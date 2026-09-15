"""Deployment-heterogeneity scenario runner (Scenarios 1-8).

Evaluation only.  Nothing here trains, selects a checkpoint, or writes into the sealed comparison
directory.  Each scenario removes information that the standard sealed protocol supplies, and every
model is scored on identical episodes.

Scenarios
---------
=========================  ===================================================================
``s1_partial_coverage``    only some candidates carry enrolment; the truth is sometimes one of
                           the others (also available standalone in ``run_partial_coverage``)
``s2_cross_placement``     enrol at one body site, deploy at another, within a dataset
``s3_cross_dataset``       enrol from a different public corpus over shared labels
``s4_missing_modality``    the gyroscope is absent from the query, the support, or both
``s5_rate_mismatch``       the query is acquired at a different sampling rate
``s6_new_domain``          rehabilitation / gym / daily-living labels no model has trained on
``s7_device_set``          the number of worn devices differs between enrolment and deployment
``s8_cold_start``          all of the above at once, on one held-out domain
=========================  ===================================================================

Severity is recorded per row on the four axes of the plan (label, support coverage, provenance,
configuration) so a degradation curve can be drawn without re-deriving which cell was which.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import baselines
from baselines import scoring
from baselines.data import load_eval_stream, load_multi_device_stream, source_slice_fingerprint
from data.scripts.curate.compatibility import PLACEMENT_SITE
from data.scripts.curate.deployment_policy import MULTI_DEVICE_EVAL_CELLS, get_stream_spec

from .partial_coverage import (
    CoverageCell,
    choose_hidden_candidates,
    hide_supports,
    hybrid_predictions,
    support_only_predictions,
)
from .run_partial_coverage import (
    cannot_attempt_rows,
    conse_scores,
    emit_rows,
    native_text_scores,
)
from .scenarios import (
    build_cross_manifest,
    derive_accel_only,
    derive_resampled,
    shared_candidates,
    stream_rows,
)
from .sealed_eval import (
    PRIMARY_BASELINES,
    SEED,
    TRAINING_BANK_ZERO_SHOT,
    _build_training_reference_bank,
    _file_hash,
    _halo_residual_predictions,
    _load_or_encode,
    _aligned_labels,
    _native_capabilities,
    _parameter_count_m,
    build_manifest,
    manifest_fingerprint,
    QueryPlan,
)
from data.datasets.mmfit.convert import PAPER_SPLITS
from training.tokenizer.eval_transfer import build_encoder

# ------------------------------------------------------------------ cell rosters

# Only streams that actually have a grid on disk are listed. Adding a placement means gridding it
# first; the runner reports a missing grid rather than quietly skipping the cell.
CROSS_PLACEMENT_STREAMS = {
    "realworld": ("phone_waist", "phone_forearm", "phone_thigh"),
    "shoaib": ("phone_right_pocket", "phone_left_pocket", "phone_belt", "watch_wrist_proxy"),
}

# Same-body-region cross-corpus pairs, plus deliberately mismatched ones so provenance and
# placement can be separated in the analysis.
CROSS_DATASET_PAIRS = (
    (("motionsense", "phone_front_pocket"), ("shoaib", "phone_right_pocket"), "same_region"),
    (("shoaib", "phone_right_pocket"), ("motionsense", "phone_front_pocket"), "same_region"),
    (("usc_had", "phone_hip"), ("realworld", "phone_waist"), "same_region"),
    (("realworld", "phone_waist"), ("usc_had", "phone_hip"), "same_region"),
    (("ut_complex", "watch_wrist"), ("shoaib", "watch_wrist_proxy"), "same_region"),
    (("inclusivehar", "phone_waist"), ("usc_had", "phone_hip"), "same_region"),
    (("motionsense", "phone_front_pocket"), ("realworld", "phone_waist"), "other_region"),
    (("ut_complex", "watch_wrist"), ("motionsense", "phone_front_pocket"), "other_region"),
)

SEALED_SINGLE_CELLS = (
    ("motionsense", "phone_front_pocket"),
    ("realworld", "phone_waist"),
    ("shoaib", "phone_right_pocket"),
    ("shoaib", "watch_wrist_proxy"),
    ("inclusivehar", "phone_waist"),
    ("usc_had", "phone_hip"),
    ("ut_complex", "watch_wrist"),
)

NEW_DOMAIN_CELLS = (
    ("mmfit", "left_wrist"),
    ("spar", "watch_left_wrist"),
    ("upper_limb_use", "control_right_wrist"),
)

RATE_TARGETS = (20.0, 25.0, 100.0)

_REGION = {
    "wrist": "arm", "forearm": "arm", "upper_arm": "arm", "hand": "arm", "ear": "head",
    "waist": "torso", "belt": "torso", "hip": "torso", "chest": "torso", "back": "torso",
    "torso": "torso", "head": "head",
    "thigh": "leg", "pocket": "leg", "knee": "leg", "shin": "leg", "calf": "leg",
    "ankle": "leg", "gastrocnemius": "leg", "hamstrings": "leg", "tibialis": "leg",
    "rectus_femoris": "leg",
}


def _region(dataset: str, stream_id: str) -> str:
    try:
        site = PLACEMENT_SITE.get(get_stream_spec(dataset, stream_id).placement, "")
    except (KeyError, ValueError):
        site = ""
    for token, region in _REGION.items():
        if token in site:
            return region
    return "unknown"


def anatomical_distance(dataset: str, a: str, b: str) -> str:
    """Coarse separation between two placements: same region, adjacent, or opposite end."""
    first, second = _region(dataset, a), _region(dataset, b)
    if "unknown" in (first, second):
        return "unknown"
    if first == second:
        return "near"
    if {first, second} in ({"arm", "leg"}, {"head", "leg"}):
        return "far"
    return "mid"


# ------------------------------------------------------------------- task model


@dataclass
class Task:
    """One scored cell: a query stream, an enrolment source, and an episode manifest."""

    scenario: str
    variant: str
    query_stream: object
    support_stream: object
    plans: tuple
    candidates: tuple[str, ...]
    offset: int
    severity: dict[str, int]
    coverage: CoverageCell | None = None
    meta: dict = field(default_factory=dict)

    @property
    def cross(self) -> bool:
        return self.support_stream is not self.query_stream


def _load(dataset: str, stream_id: str, window_seconds: float):
    return load_eval_stream(dataset, stream_id, alignment="native",
                            window_seconds=window_seconds, apply_quality_screen=True)


def _composite(dataset: str, device_ids, window_seconds: float):
    return load_multi_device_stream(dataset, tuple(device_ids), alignment="native",
                                    window_seconds=window_seconds, apply_quality_screen=True)


def _within(stream, k: int, seed: int) -> tuple:
    return tuple(build_manifest(stream, k, seed=seed))


def _within_cross_subject(stream, k: int, seed: int) -> tuple:
    if k == 0:
        return _within(stream, k, seed)
    cross = build_cross_manifest(
        stream, stream, k, seed=seed, relation="within_cross_subject", same_subject=False,
    )
    return tuple(QueryPlan(
        query=plan.query,
        support=tuple(row - cross.offset for row in plan.support),
        support_labels=plan.support_labels,
    ) for plan in cross.plans)


def _mmfit_new_domain_plans(stream, k: int, seed: int) -> tuple[tuple[QueryPlan, ...], str]:
    """Use MM-Fit's published participant split without treating workout IDs as people."""
    if k == 0:
        query_rows = _mmfit_split_rows(stream, ("cross_subject_test",))
        labels = _aligned_labels(stream)
        plans = tuple(
            QueryPlan(query=int(row), support=(), support_labels=())
            for row in query_rows
            if labels[row] in stream.eval_labels
        )
        return plans, "published_participant_test_zero_support"

    query_rows = _mmfit_split_rows(stream, ("cross_subject_test",))
    support_rows = _mmfit_split_rows(stream, ("train", "validation"))
    cross = build_cross_manifest(
        stream,
        stream,
        k,
        seed=seed,
        relation="mmfit_published_participant_split",
        candidates=tuple(stream.eval_labels),
        query_rows=query_rows,
        support_rows=support_rows,
    )
    plans = tuple(
        QueryPlan(
            query=plan.query,
            support=tuple(row - cross.offset for row in plan.support),
            support_labels=plan.support_labels,
        )
        for plan in cross.plans
    )
    return plans, "published_participant_split"


def _cross_view_of_within(plans, offset: int) -> tuple:
    """Reuse a within-stream manifest when only the sensor view changes."""
    return tuple(
        type(plan)(query=plan.query, support=tuple(offset + row for row in plan.support),
                   support_labels=plan.support_labels)
        for plan in plans
    )


def _coverage_cell(candidates, *, seed: int, dataset: str, stream: str,
                   window_seconds: float, coverage: float) -> CoverageCell:
    # k deliberately stays out of the identity: curves change evidence quantity, not which
    # activities happened to receive enrollment.
    return choose_hidden_candidates(candidates, coverage=coverage,
                                    seed_parts=(seed, dataset, stream, window_seconds, coverage))


def _zero_support_cross_plans(query, support) -> tuple[tuple[QueryPlan, ...], tuple[str, ...]]:
    """A cross-configuration k=0 cell has no support rows but keeps the shared label roster."""
    candidates, _ = shared_candidates(query, support)
    if len(candidates) < 2:
        return (), candidates
    labels = _aligned_labels(query)
    plans = tuple(QueryPlan(query=int(row), support=(), support_labels=()) for row in
                  np.flatnonzero(np.isin(labels, candidates)))
    return plans, candidates


def _matched_within_reference(
    cross,
    query,
    k: int,
    *,
    seed: int,
    same_subject: bool | None,
    relation: str,
) -> tuple[tuple[QueryPlan, ...], tuple[QueryPlan, ...]]:
    """Pair a cross-source condition with an in-query-source reference on identical queries.

    ``build_cross_manifest(query, query, ...)`` is used to enforce the requested subject relation.
    Its support indexes address a concatenated matrix, so they are translated back to the one
    within-stream feature matrix before returning.
    """
    query_rows = tuple(plan.query for plan in cross.plans)
    reference = build_cross_manifest(
        query, query, k, seed=seed, relation=relation, same_subject=same_subject,
        candidates=cross.candidates, query_rows=query_rows,
    )
    cross_by_query = {plan.query: plan for plan in cross.plans}
    reference_by_query = {plan.query: plan for plan in reference.plans}
    common = tuple(sorted(set(cross_by_query) & set(reference_by_query)))
    cross_plans = tuple(cross_by_query[row] for row in common)
    reference_plans = tuple(QueryPlan(
        query=reference_by_query[row].query,
        support=tuple(index - reference.offset for index in reference_by_query[row].support),
        support_labels=reference_by_query[row].support_labels,
    ) for row in common)
    return cross_plans, reference_plans


def _mmfit_split_rows(stream, split_names: tuple[str, ...]) -> np.ndarray:
    workout_ids = {f"w{value:02d}" for name in split_names for value in PAPER_SPLITS[name]}
    subjects = np.asarray(stream.subjects, dtype=str)
    rows = np.flatnonzero(np.isin(subjects, sorted(workout_ids)))
    if not len(rows):
        raise ValueError(f"MM-Fit paper split {split_names} has no rows after quality screening")
    return rows


# ------------------------------------------------------------------ task builders


def build_tasks(scenario: str, k: int, window_seconds: float, *, seed: int,
                coverage: float) -> list[Task]:
    """Every cell of one scenario at one evidence budget and one enrolment size."""
    tasks: list[Task] = []

    if scenario == "s1_partial_coverage":
        if k == 0:
            return tasks
        for dataset, stream_id in SEALED_SINGLE_CELLS:
            stream = _load(dataset, stream_id, window_seconds)
            plans = _within_cross_subject(stream, k, seed)
            if not plans:
                continue
            cell = _coverage_cell(stream.eval_labels, seed=seed, dataset=dataset, stream=stream_id,
                                  window_seconds=window_seconds, coverage=coverage)
            tasks.append(Task(
                scenario, f"{dataset}/{stream_id}", stream, stream,
                tuple(hide_supports(plans, cell)), tuple(stream.eval_labels),
                stream_rows(stream), {"L": 0, "S": 2, "P": 0, "C": 0}, coverage=cell,
                meta={"subject_relation": "cross_subject"}))

    elif scenario == "s2_cross_placement":
        for dataset, streams in CROSS_PLACEMENT_STREAMS.items():
            for query_id in streams:
                for support_id in streams:
                    if query_id == support_id:
                        continue
                    query = _load(dataset, query_id, window_seconds)
                    support = _load(dataset, support_id, window_seconds)
                    if k == 0:
                        plans, candidates = _zero_support_cross_plans(query, support)
                        if plans:
                            tasks.append(Task(
                                scenario, f"{dataset}/{query_id}<-{support_id}/zero_support",
                                query, support, plans, candidates, stream_rows(query),
                                {"L": 0, "S": 3, "P": 0, "C": 0},
                                meta={"distance": anatomical_distance(dataset, query_id, support_id),
                                      "subject_relation": "not_applicable_zero_support"}))
                        continue
                    for relation, same_subject, severity in (
                            ("same_subject", True, 1), ("cross_subject", False, 2)):
                        cross = build_cross_manifest(
                            query, support, k, seed=seed, relation=relation,
                            same_subject=same_subject)
                        cross_plans, reference_plans = _matched_within_reference(
                            cross, query, k, seed=seed, same_subject=same_subject,
                            relation=f"s2_reference_{relation}",
                        )
                        if not cross_plans:
                            continue
                        matched_group = f"{dataset}/{query_id}<-{support_id}/{relation}/k{k}"
                        tasks.append(Task(
                            scenario, f"{dataset}/{query_id}<-{query_id}/{relation}/matched_control",
                            query, query, reference_plans, cross.candidates, stream_rows(query),
                            {"L": 0, "S": 0, "P": 0, "C": 0},
                            meta={"matched_group": matched_group, "condition": "control",
                                  "subject_relation": relation},
                        ))
                        tasks.append(Task(
                            scenario, f"{dataset}/{query_id}<-{support_id}/{relation}",
                            query, support, cross_plans, cross.candidates, cross.offset,
                            {"L": 0, "S": 0, "P": severity, "C": 0},
                            meta={"distance": anatomical_distance(dataset, query_id, support_id),
                                  "subject_relation": relation, "matched_group": matched_group,
                                  "condition": "scenario"}))

    elif scenario == "s3_cross_dataset":
        for (q_ds, q_id), (s_ds, s_id), region in CROSS_DATASET_PAIRS:
            query = _load(q_ds, q_id, window_seconds)
            support = _load(s_ds, s_id, window_seconds)
            if k == 0:
                plans, candidates = _zero_support_cross_plans(query, support)
                if plans:
                    tasks.append(Task(
                        scenario, f"{q_ds}/{q_id}<-{s_ds}/{s_id}/zero_support",
                        query, support, plans, candidates, stream_rows(query),
                        {"L": 0, "S": 3, "P": 3, "C": 0},
                        meta={"region_match": region, "support_dataset": s_ds,
                              "subject_relation": "not_applicable_zero_support"}))
                continue
            cross = build_cross_manifest(query, support, k, seed=seed,
                                         relation="cross_dataset", same_subject=None)
            cross_plans, reference_plans = _matched_within_reference(
                cross, query, k, seed=seed, same_subject=False,
                relation="s3_in_dataset_cross_subject_reference",
            )
            if not cross_plans:
                continue
            matched_group = f"{q_ds}/{q_id}<-{s_ds}/{s_id}/k{k}"
            tasks.append(Task(
                scenario, f"{q_ds}/{q_id}<-{q_ds}/{q_id}/matched_control_for_{s_ds}",
                query, query, reference_plans, cross.candidates, stream_rows(query),
                {"L": 0, "S": 0, "P": 0, "C": 0},
                meta={"matched_group": matched_group, "condition": "control",
                      "subject_relation": "cross_subject"},
            ))
            tasks.append(Task(
                scenario, f"{q_ds}/{q_id}<-{s_ds}/{s_id}", query, support,
                cross_plans, cross.candidates, cross.offset,
                {"L": 0, "S": 0, "P": 3, "C": 0},
                meta={"region_match": region, "support_dataset": s_ds,
                      "subject_relation": "cross_dataset_identity_unmapped",
                      "matched_group": matched_group, "condition": "scenario"}))

    elif scenario == "s4_missing_modality":
        if k == 0:
            return tasks
        for dataset, stream_id in SEALED_SINGLE_CELLS:
            full = _load(dataset, stream_id, window_seconds)
            base_plans = _within_cross_subject(full, k, seed)
            try:
                accel = derive_accel_only(full)
            except ValueError as exc:
                tasks.append(Task(scenario, f"{dataset}/{stream_id}", full, full, (), (), 0,
                                  {"L": 0, "S": 0, "P": 0, "C": 1},
                                  meta={"skipped": str(exc)}))
                continue
            tasks.append(Task(
                scenario, f"{dataset}/{stream_id}/matched_full_control", full, full,
                base_plans, tuple(full.eval_labels), stream_rows(full),
                {"L": 0, "S": 0, "P": 0, "C": 0},
                meta={"matched_parent": f"{dataset}/{stream_id}/modality",
                      "condition": "control", "subject_relation": "cross_subject"},
            ))
            for variant, query, support in (
                    ("query_accel_only", accel, full),
                    ("support_accel_only", full, accel),
                    ("both_accel_only", accel, accel)):
                if query is support:
                    plans = base_plans
                    candidates, offset = tuple(query.eval_labels), stream_rows(query)
                else:
                    offset = stream_rows(query)
                    plans, candidates = _cross_view_of_within(base_plans, offset), tuple(full.eval_labels)
                if not plans:
                    continue
                tasks.append(Task(
                    scenario, f"{dataset}/{stream_id}/{variant}", query, support,
                    tuple(plans), candidates, offset, {"L": 0, "S": 0, "P": 0, "C": 1},
                    meta={"perturbation": variant,
                          "matched_parent": f"{dataset}/{stream_id}/modality",
                          "condition": "scenario", "subject_relation": "cross_subject"}))

    elif scenario == "s5_rate_mismatch":
        if k == 0:
            return tasks
        for dataset, stream_id in SEALED_SINGLE_CELLS:
            full = _load(dataset, stream_id, window_seconds)
            base_plans = _within_cross_subject(full, k, seed)
            tasks.append(Task(
                scenario, f"{dataset}/{stream_id}/matched_full_control", full, full,
                base_plans, tuple(full.eval_labels), stream_rows(full),
                {"L": 0, "S": 0, "P": 0, "C": 0},
                meta={"matched_parent": f"{dataset}/{stream_id}/rate", "condition": "control",
                      "subject_relation": "cross_subject"},
            ))
            for target in RATE_TARGETS:
                if abs(target - float(full.rate_hz)) < 1e-9:
                    continue
                query = derive_resampled(full, target)
                if not base_plans:
                    continue
                tasks.append(Task(
                    scenario, f"{dataset}/{stream_id}/query@{target:g}Hz", query, full,
                    _cross_view_of_within(base_plans, stream_rows(query)), tuple(full.eval_labels),
                    stream_rows(query),
                    {"L": 0, "S": 0, "P": 0, "C": 2},
                    meta={"query_rate_hz": target, "support_rate_hz": float(full.rate_hz),
                          "matched_parent": f"{dataset}/{stream_id}/rate",
                          "condition": "scenario", "subject_relation": "cross_subject"}))

    elif scenario == "s6_new_domain":
        for dataset, stream_id in NEW_DOMAIN_CELLS:
            stream = _load(dataset, stream_id, window_seconds)
            if dataset == "mmfit":
                plans, subject_relation = _mmfit_new_domain_plans(stream, k, seed)
            else:
                plans = _within_cross_subject(stream, k, seed)
                subject_relation = (
                    "not_applicable_zero_support" if k == 0 else "cross_subject"
                )
            if not plans:
                continue
            tasks.append(Task(
                scenario, f"{dataset}/{stream_id}", stream, stream, tuple(plans),
                tuple(stream.eval_labels), stream_rows(stream),
                {"L": 3, "S": 0, "P": 0, "C": 0},
                meta={"subject_relation": subject_relation}))
            if k > 0:
                cell = _coverage_cell(stream.eval_labels, seed=seed, dataset=dataset, stream=stream_id,
                                      window_seconds=window_seconds, coverage=coverage)
                tasks.append(Task(
                    scenario, f"{dataset}/{stream_id}/partial_coverage", stream, stream,
                    tuple(hide_supports(plans, cell)), tuple(stream.eval_labels),
                    stream_rows(stream), {"L": 3, "S": 2, "P": 0, "C": 0}, coverage=cell,
                    meta={"subject_relation": subject_relation}))

    elif scenario == "s7_device_set":
        for cell_spec in MULTI_DEVICE_EVAL_CELLS:
            dataset, device_ids = cell_spec.dataset, tuple(cell_spec.stream_ids)
            composite = _composite(dataset, device_ids, window_seconds)
            single = _load(dataset, device_ids[0], window_seconds)
            partial = _composite(dataset, device_ids[:2], window_seconds)
            if k == 0:
                plans, candidates = _zero_support_cross_plans(composite, single)
                if plans:
                    tasks.append(Task(
                        scenario, f"{dataset}/zero_support", composite, single, plans,
                        candidates, stream_rows(composite), {"L": 0, "S": 3, "P": 0, "C": 3},
                        meta={"device_variant": "zero_support",
                              "subject_relation": "not_applicable_zero_support"}))
                continue
            for variant, query, support in (
                    ("support_single_query_composite", composite, single),
                    ("support_composite_query_single", single, composite),
                    ("support_two_devices_query_all", composite, partial)):
                cross = build_cross_manifest(
                    query, support, k, seed=seed, relation=variant, same_subject=False,
                )
                cross_plans, reference_plans = _matched_within_reference(
                    cross, query, k, seed=seed, same_subject=False,
                    relation=f"s7_reference_{variant}",
                )
                if not cross_plans:
                    continue
                matched_group = f"{dataset}/{variant}/k{k}"
                tasks.append(Task(
                    scenario, f"{dataset}/{variant}/matched_control", query, query,
                    reference_plans, cross.candidates, stream_rows(query),
                    {"L": 0, "S": 0, "P": 0, "C": 0},
                    meta={"device_variant": variant, "condition": "control",
                          "matched_group": matched_group, "subject_relation": "cross_subject"},
                ))
                tasks.append(Task(
                    scenario, f"{dataset}/{variant}", query, support, cross_plans,
                    cross.candidates, cross.offset, {"L": 0, "S": 0, "P": 0, "C": 3},
                    meta={"device_variant": variant,
                          "query_devices": len(getattr(query, "devices", [1])),
                          "support_devices": len(getattr(support, "devices", [1])),
                          "condition": "scenario", "matched_group": matched_group,
                          "subject_relation": "cross_subject"}))

    elif scenario == "s8_cold_start":
        if k == 0:
            return tasks
        dataset = "mmfit"
        query = _composite(dataset, ("left_wrist", "right_pocket"), window_seconds)
        support = _load(dataset, "right_wrist", window_seconds)
        query_rows = _mmfit_split_rows(query, ("cross_subject_test",))
        support_rows = _mmfit_split_rows(support, ("train", "validation"))
        cross = build_cross_manifest(query, support, k, seed=seed,
                                     relation="cold_start_paper_split",
                                     query_rows=query_rows, support_rows=support_rows)
        if cross.plans:
            cell = _coverage_cell(cross.candidates, seed=seed, dataset=dataset, stream="cold_start",
                                  window_seconds=window_seconds, coverage=coverage)
            tasks.append(Task(
                scenario, f"{dataset}/cold_start", query, support,
                tuple(hide_supports(cross.plans, cell)), cross.candidates, cross.offset,
                {"L": 3, "S": 2, "P": 2, "C": 3}, coverage=cell,
                meta={"support_stream": "right_wrist", "query_devices": 2,
                      "provenance_unit": "published_mmfit_workout_partition",
                      "bootstrap_unit": "workout"}))
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return tasks


# --------------------------------------------------------------------- scoring


def score_task(task: Task, *, models, device, cache_dir, halo_checkpoint, bootstrap,
               banks, k: int, window_seconds: float, halo_state=None,
               provider_states=None, prediction_sink=None) -> list[dict]:
    """Every readout for every model on one task's episodes."""
    if not task.plans:
        return [{"scenario": task.scenario, "variant": task.variant, "status": "n/a",
                 "reason": task.meta.get("skipped", "no honest episode could be formed"),
                 "k": k, "window_seconds": float(window_seconds), **task.severity}]

    coverage = task.coverage or CoverageCell(
        supported=tuple(task.candidates), hidden=(), coverage=1.0, requested_coverage=1.0)
    roster = SimpleNamespace(eval_labels=list(task.candidates))
    manifest = manifest_fingerprint(task.plans)
    common = dict(k=k, window_seconds=window_seconds, bootstrap=bootstrap, manifest=manifest)
    severity_meta = {**{f"severity_{axis}": value for axis, value in task.severity.items()},
                     **task.meta, "scenario": task.scenario, "variant": task.variant,
                     "cross_stream": task.cross,
                     "support_source": f"{task.support_stream.dataset}/"
                                       f"{getattr(task.support_stream, 'stream', getattr(task.support_stream, 'cell_id', ''))}"}
    rows: list[dict] = []
    provider_states = {} if provider_states is None else provider_states
    prediction_sink = [] if prediction_sink is None else prediction_sink

    def append_emitted(predictions, *, readout: str) -> None:
        emitted = emit_rows(task.query_stream, task.plans, predictions, coverage,
                            model=name, readout=readout, extra=extra_base, **common)
        if task.coverage is not None:
            # The combined split contains both conditions; each conditional split receives the
            # severity it actually experiences rather than its cell's worst-case severity.
            for row in emitted:
                split = row.get("coverage_split")
                row["severity_S"] = 1 if split == "truth_enrolled" else 2 if split == "truth_unenrolled" else 2
        if name == "halo" and readout == "halo-classifier":
            for row in emitted:
                row["parameters_m"] = round(_parameter_count_m(
                    name, halo_state, halo_checkpoint=halo_checkpoint, include_classifier=True,
                ), 6)
        rows.extend(emitted)
        truth = _aligned_labels(task.query_stream)
        for plan, prediction in zip(task.plans, predictions):
            query = int(plan.query)
            prediction_sink.append({
                "scenario": task.scenario,
                "variant": task.variant,
                "matched_group": task.meta.get("matched_group"),
                "matched_parent": task.meta.get("matched_parent"),
                "condition": task.meta.get("condition"),
                "model": name,
                "readout": readout,
                "k": int(k),
                "window_seconds": float(window_seconds),
                "query_row": query,
                "query_event_id": str(np.asarray(task.query_stream.event_ids)[query]),
                "query_execution_id": (None if task.query_stream.execution_ids is None else
                                       str(np.asarray(task.query_stream.execution_ids)[query])),
                "query_group_id": str(np.asarray(task.query_stream.subjects)[query]),
                "truth": str(truth[query]),
                "prediction": str(prediction),
            })

    for name in models:
        try:
            if name != "halo" and name not in provider_states:
                provider_states[name] = baselines.REGISTRY[name].setup_features(device)
            query_features, fingerprint = _load_or_encode(
                name=name, stream=task.query_stream, device=device, cache_dir=cache_dir,
                halo_checkpoint=halo_checkpoint, baseline_state=provider_states.get(name),
                halo_state=halo_state)
            if task.cross:
                support_features, _ = _load_or_encode(
                    name=name, stream=task.support_stream, device=device, cache_dir=cache_dir,
                    halo_checkpoint=halo_checkpoint, baseline_state=provider_states.get(name),
                    halo_state=halo_state)
                features = np.concatenate([query_features, support_features], axis=0)
            else:
                features = query_features
            extra_base = {**severity_meta, "n_candidates": len(task.candidates),
                          "feature_fingerprint": fingerprint,
                          "parameters_m": round(_parameter_count_m(
                              name, halo_state, halo_checkpoint=halo_checkpoint,
                              include_classifier=False,
                          ), 6), **_native_capabilities(name)}

            if k > 0:
                for readout, predicted in support_only_predictions(
                        features, task.candidates, task.plans, coverage).items():
                    append_emitted(predicted, readout=readout)

            text = None
            if k == 0 or task.coverage is not None:
                if name in TRAINING_BANK_ZERO_SHOT and (name != "halo" or task.coverage is not None):
                    if name not in banks:
                        banks[name] = _build_training_reference_bank(
                            name=name, device=device,
                            cache_dir=Path("training/support_classifier/evaluations/zero_shot_feature_cache"),
                            halo_checkpoint=halo_checkpoint, halo_state=halo_state,
                            baseline_state=provider_states.get(name))
                    bank_features, bank_labels, train_labels, _ = banks[name]
                    text = conse_scores(query_features, bank_features, bank_labels, train_labels,
                                        task.candidates, device)
                elif name != "halo" and baselines.REGISTRY[name].supports_native_zero_shot():
                    state = provider_states[name]
                    text = native_text_scores(name, query_features, task.candidates, state, device)

            if k == 0:
                if name == "halo" and halo_checkpoint is not None:
                    predicted = _halo_residual_predictions(features, roster, task.plans,
                                                          halo_checkpoint, device)
                    append_emitted(predicted, readout="halo-classifier")
                elif text is not None:
                    predicted = [task.candidates[index] for index in text.argmax(axis=1).tolist()]
                    append_emitted(predicted, readout="zero-shot-native-or-bridge")
                else:
                    rows.append({"model": name, "readout": "zero-shot", "status": "inapplicable",
                                 "reason": "model exposes neither a native nor configured bridge zero-shot path",
                                 **extra_base, **common})
            elif task.coverage is not None:
                if text is not None:
                    hybrid = hybrid_predictions(text, features, task.candidates, task.plans, coverage)
                    append_emitted(hybrid, readout="hybrid-text-support")
                elif name != "halo":
                    rows.extend(cannot_attempt_rows(
                        task.query_stream, coverage, model=name, readout="hybrid-text-support",
                        k=k, window_seconds=window_seconds,
                        reason="no native or configured text-score path: hidden candidates are unreachable"))

            if name == "halo" and halo_checkpoint is not None and k > 0:
                predicted = _halo_residual_predictions(features, roster, task.plans, halo_checkpoint, device)
                append_emitted(predicted, readout="halo-classifier")
        except baselines.UnsupportedEvaluationCell as exc:
            rows.append({"model": name, "status": "unsupported", "reason": str(exc),
                         "k": k, "window_seconds": float(window_seconds), **severity_meta})
        except Exception as exc:  # Preserve other models' completed rows and make unexpected failures visible.
            rows.append({"model": name, "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                         "k": k, "window_seconds": float(window_seconds), **severity_meta})
    return rows


# ------------------------------------------------------------------------ main


def _task_artifact(task: Task, *, k: int, window_seconds: float) -> dict:
    """Persist enough episode provenance to reproduce a scored cell exactly."""
    return {
        "scenario": task.scenario,
        "variant": task.variant,
        "k": int(k),
        "window_seconds": float(window_seconds),
        "candidates": list(task.candidates),
        "severity": task.severity,
        "coverage": (None if task.coverage is None else {
            "supported": list(task.coverage.supported),
            "hidden": list(task.coverage.hidden),
            "coverage": task.coverage.coverage,
            "requested_coverage": task.coverage.requested_coverage,
            "fingerprint": task.coverage.fingerprint,
        }),
        "query_source_fingerprint": source_slice_fingerprint(task.query_stream),
        "support_source_fingerprint": source_slice_fingerprint(task.support_stream),
        "query_rate_hz": getattr(task.query_stream, "rate_hz", None),
        "support_rate_hz": getattr(task.support_stream, "rate_hz", None),
        "query_devices": list(getattr(task.query_stream, "device_ids", ())),
        "support_devices": list(getattr(task.support_stream, "device_ids", ())),
        "plans": [_plan_artifact(task, plan) for plan in task.plans],
        "meta": task.meta,
    }


def _plan_artifact(task: Task, plan: QueryPlan) -> dict:
    query = int(plan.query)
    support_local = [int(row - task.offset) if task.cross else int(row) for row in plan.support]
    event_ids = np.asarray(task.support_stream.event_ids)
    executions = task.support_stream.execution_ids
    subjects = np.asarray(task.support_stream.subjects)
    return {
        "query": query,
        "query_event_id": str(np.asarray(task.query_stream.event_ids)[query]),
        "query_execution_id": (None if task.query_stream.execution_ids is None else
                               str(np.asarray(task.query_stream.execution_ids)[query])),
        "query_group_id": str(np.asarray(task.query_stream.subjects)[query]),
        "support": [int(row) for row in plan.support],
        "support_local_rows": support_local,
        "support_event_ids": [str(event_ids[row]) for row in support_local],
        "support_execution_ids": (None if executions is None else
                                  [str(np.asarray(executions)[row]) for row in support_local]),
        "support_group_ids": [str(subjects[row]) for row in support_local],
        "support_labels": list(plan.support_labels),
    }


def _atomic_json(path: Path, value) -> None:
    """Checkpoint scenario progress without leaving a half-written JSON artifact."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def _append_jsonl(path: Path, values) -> None:
    """Append independently readable audit records without rewriting prior task output."""
    with path.open("a") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_hash(path: Path) -> str:
    """Hash a model artifact whether an adapter exposes one file or a directory."""
    if path.is_file():
        return _file_hash(path)
    if not path.is_dir():
        raise FileNotFoundError(f"model artifact does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(bytes.fromhex(_file_hash(child)))
    return digest.hexdigest()


def _run_provenance(argv: list[str], *, device: torch.device, halo_checkpoint: Path | None) -> dict:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-z"], text=True,
        )
        dirty = hashlib.sha256()
        dirty.update(status.encode())
        for entry in filter(None, status.split("\0")):
            path_text = entry[3:].split(" -> ")[-1]
            path = Path(path_text)
            if path.is_file():
                dirty.update(path_text.encode())
                dirty.update(path.read_bytes())
        dirty_digest = dirty.hexdigest() if status else None
    except (OSError, subprocess.CalledProcessError):
        dirty_digest = None
    return {
        "protocol": "deployment-scenarios-v2-20260915",
        "argv": argv,
        "git_revision": revision,
        "dirty_diff_sha256": dirty_digest,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "halo_checkpoint": (str(halo_checkpoint.resolve()) if halo_checkpoint else None),
        "halo_checkpoint_sha256": (_file_hash(halo_checkpoint) if halo_checkpoint else None),
        "seed_policy": "fixed seed; partial-coverage candidate split is invariant across k",
    }


def _paired_deltas(predictions: list[dict], *, bootstrap: int) -> list[dict]:
    """Compute scenario-minus-control deltas from persisted per-query decisions."""
    controls: dict[tuple, dict[str, dict]] = {}
    scenarios: dict[tuple, list[dict]] = {}
    for row in predictions:
        parent = row.get("matched_group") or row.get("matched_parent")
        if not parent:
            continue
        key = (parent, row["model"], row["readout"], row["k"], row["window_seconds"])
        query_key = row["query_event_id"]
        if row.get("condition") == "control":
            controls.setdefault(key, {})[query_key] = row
        elif row.get("condition") == "scenario":
            scenarios.setdefault(key, []).append(row)
    output = []
    for key, scenario_rows in scenarios.items():
        control = controls.get(key, {})
        by_variant: dict[str, list[dict]] = {}
        for row in scenario_rows:
            by_variant.setdefault(row["variant"], []).append(row)
        for variant, variant_rows in by_variant.items():
            paired = [(row, control[row["query_event_id"]]) for row in variant_rows
                      if row["query_event_id"] in control]
            if not paired:
                continue
            truth = [left["truth"] for left, _ in paired]
            scenario_prediction = [left["prediction"] for left, _ in paired]
            control_prediction = [right["prediction"] for _, right in paired]
            if any(left["truth"] != right["truth"] for left, right in paired):
                raise RuntimeError("matched scenario/control rows disagree on ground truth")
            metrics = scoring.classification_metrics(truth, scenario_prediction)
            reference = scoring.classification_metrics(truth, control_prediction)
            row = {
                "matched_group": key[0], "model": key[1], "readout": key[2],
                "k": key[3], "window_seconds": key[4], "variant": variant,
                "n_paired_queries": len(paired),
                "f1_macro_scenario": metrics["f1_macro"],
                "f1_macro_control": reference["f1_macro"],
                "f1_macro_delta": metrics["f1_macro"] - reference["f1_macro"],
            }
            if bootstrap > 0:
                row.update(scoring.paired_subject_bootstrap_difference(
                    truth, scenario_prediction, control_prediction,
                    np.asarray([left["query_group_id"] for left, _ in paired]),
                    metric="f1_macro", B=bootstrap,
                ))
            output.append(row)
    return output


def _model_artifacts(models, provider_states, halo_checkpoint) -> dict:
    output = {}
    for name in models:
        if name == "halo":
            output[name] = {"checkpoint": str(halo_checkpoint.resolve()),
                            "checkpoint_sha256": _file_hash(halo_checkpoint)}
            continue
        state = provider_states.get(name)
        if state is None:
            continue
        adapter = baselines.REGISTRY[name]
        artifacts = dict(adapter.feature_artifacts(state))
        artifacts.update(adapter.evaluation_artifacts(state))
        output[name] = {
            "artifacts": {key: {"path": str(path), "sha256": _artifact_hash(Path(path))}
                          for key, path in artifacts.items()},
            "feature_config": adapter.feature_config(state),
            "evaluation_config": adapter.evaluation_config(state),
        }
    return output


def _write_tabular_results(out: Path, rows: list[dict]) -> None:
    """Write inspectable CSV and Markdown companions without changing the JSON source of truth."""
    columns = [
        "scenario", "variant", "condition", "matched_group", "matched_parent", "dataset",
        "stream", "model", "readout", "window_seconds", "k", "coverage_split", "status",
        "n_queries", "n_candidates", "accuracy", "f1_macro", "balanced_accuracy",
        "f1_macro_seen", "f1_macro_unseen", "false_enrollment_pull", "parameters_m",
        "native_open_set_labels", "native_few_shot_adaptation", "reason",
    ]
    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    shown = rows
    header = [
        "scenario", "variant", "split", "model", "readout", "w", "k", "parameters (M)",
        "native open set", "native few shot", "status", "macro F1", "accuracy",
        "seen F1", "unseen F1", "false enrollment pull",
    ]
    lines = ["# Deployment scenario results", "", "Generated by `run_scenarios`; no result selects a checkpoint.", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in shown:
        values = [
            row.get("scenario", ""), row.get("variant", ""), row.get("coverage_split", "all"),
            row.get("model", ""),
            row.get("readout", ""), row.get("window_seconds", ""), row.get("k", ""),
            row.get("parameters_m", ""), row.get("native_open_set_labels", ""),
            row.get("native_few_shot_adaptation", ""), row.get("status", ""),
            row.get("f1_macro", ""), row.get("accuracy", ""),
            row.get("f1_macro_seen", ""), row.get("f1_macro_unseen", ""),
            row.get("false_enrollment_pull", ""),
        ]
        lines.append("| " + " | ".join(str(value) if value is not None else "-" for value in values) + " |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=[
        "s1_partial_coverage", "s2_cross_placement", "s3_cross_dataset", "s4_missing_modality",
        "s5_rate_mismatch", "s6_new_domain", "s7_device_set", "s8_cold_start"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=["halo", *PRIMARY_BASELINES])
    parser.add_argument("--halo-checkpoint", type=Path, default=None)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--window-seconds", nargs="+", type=float, default=[8.0])
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap", type=int, default=scoring.BOOTSTRAP_B)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--max-tasks-per-scenario", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="one task per scenario, one k, no bootstrap: wiring check only")
    args = parser.parse_args()

    if any(value < 0 for value in args.k):
        parser.error("--k values must be non-negative")
    if any(not np.isfinite(value) or value <= 0 for value in args.window_seconds):
        parser.error("--window-seconds values must be finite and positive")
    if not 0.0 < args.coverage < 1.0:
        parser.error("--coverage must lie strictly between zero and one")
    unknown = sorted(set(args.models) - set(PRIMARY_BASELINES) - {"halo"})
    if unknown:
        parser.error(f"models outside the registered primary roster: {unknown}")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not repeat a provider")
    if "halo" in args.models and args.halo_checkpoint is None:
        parser.error("--halo-checkpoint is required when model list includes halo")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    halo_state = None
    if "halo" in args.models:
        blob = torch.load(args.halo_checkpoint, map_location="cpu", weights_only=False)
        halo_state = (build_encoder(blob, device).eval(), _file_hash(args.halo_checkpoint))
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache or (args.out / "feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(set(args.k))
    windows = sorted(set(args.window_seconds))
    bootstrap = 0 if args.smoke else args.bootstrap
    limit = 1 if args.smoke else args.max_tasks_per_scenario
    if args.smoke:
        ks, windows = ks[:1], windows[:1]

    rows: list[dict] = []
    banks: dict[str, tuple] = {}
    failures: list[dict] = []
    paired_predictions: list[dict] = []
    provider_states: dict[str, dict] = {}
    manifest_path = args.out / "manifests.jsonl"
    prediction_path = args.out / "predictions.jsonl"
    manifest_path.write_text("")
    prediction_path.write_text("")
    _atomic_json(args.out / "run_provenance.json", _run_provenance(
        list(os.sys.argv), device=device, halo_checkpoint=args.halo_checkpoint,
    ))
    for scenario in args.scenarios:
        for window_seconds in windows:
            for k in ks:
                try:
                    tasks = build_tasks(scenario, k, window_seconds,
                                        seed=args.seed, coverage=args.coverage)
                except Exception as exc:  # noqa: BLE001
                    failures.append({"scenario": scenario, "k": k,
                                     "window_seconds": window_seconds,
                                     "stage": "build_tasks",
                                     "error": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc()})
                    _atomic_json(args.out / "failures.json", failures)
                    continue
                if not tasks:
                    requires_enrollment = scenario in {
                        "s1_partial_coverage", "s4_missing_modality", "s5_rate_mismatch",
                        "s8_cold_start",
                    }
                    if k == 0 and requires_enrollment:
                        rows.append({"scenario": scenario, "k": k,
                                     "window_seconds": float(window_seconds),
                                     "status": "inapplicable",
                                     "reason": "scenario requires at least one enrolled support"})
                        _atomic_json(args.out / "results.json", rows)
                    else:
                        # A scenario that yields no cell is a finding, not a no-op: it means every
                        # candidate pairing failed to form an honest episode at this k.
                        failures.append({"scenario": scenario, "k": k,
                                         "window_seconds": window_seconds, "stage": "build_tasks",
                                         "error": "scenario produced no evaluable cell"})
                        _atomic_json(args.out / "failures.json", failures)
                for task in tasks[:limit]:
                    _append_jsonl(
                        manifest_path,
                        [_task_artifact(task, k=k, window_seconds=window_seconds)],
                    )
                    print(f"[scenarios] {scenario} k={k} w={window_seconds:g} {task.variant}",
                          flush=True)
                    task_predictions: list[dict] = []
                    try:
                        rows.extend(score_task(
                            task, models=args.models, device=device, cache_dir=cache_dir,
                            halo_checkpoint=args.halo_checkpoint, bootstrap=bootstrap,
                            banks=banks, k=k, window_seconds=window_seconds,
                            halo_state=halo_state, provider_states=provider_states,
                            prediction_sink=task_predictions))
                    except Exception as exc:  # noqa: BLE001
                        failures.append({"scenario": scenario, "variant": task.variant, "k": k,
                                         "window_seconds": window_seconds, "stage": "score_task",
                                         "error": f"{type(exc).__name__}: {exc}",
                                         "traceback": traceback.format_exc()})
                    _append_jsonl(prediction_path, task_predictions)
                    paired_predictions.extend(
                        row for row in task_predictions
                        if row.get("matched_group") or row.get("matched_parent")
                    )
                    _atomic_json(args.out / "results.json", rows)
                    _atomic_json(args.out / "failures.json", failures)
                    _write_tabular_results(args.out, rows)

    _atomic_json(args.out / "results.json", rows)
    _atomic_json(args.out / "failures.json", failures)
    _atomic_json(args.out / "paired_deltas.json", _paired_deltas(
        paired_predictions, bootstrap=bootstrap,
    ))
    _atomic_json(args.out / "model_artifacts.json", _model_artifacts(
        args.models, provider_states, args.halo_checkpoint,
    ))
    _write_tabular_results(args.out, rows)
    print(f"wrote {len(rows)} rows and {len(failures)} failures to {args.out}")


if __name__ == "__main__":
    main()
