"""Deployment-heterogeneity scenario runner (Scenarios 1-7).

Evaluation only.  Nothing here trains, selects a checkpoint, or writes into the sealed comparison
directory.  Each scenario removes information that the standard sealed protocol supplies, and every
model is scored on identical episodes.

Scenarios
---------
=========================  ===================================================================
``s1_partial_coverage``    only some candidates carry enrolment; the truth is sometimes one of
                           the others
``s2_cross_placement``     enrol at one body site, deploy at another, within a dataset
``s3_cross_dataset``       enrol from a different public corpus over shared labels
``s4_missing_modality``    the gyroscope is absent from the query, the support, or both
``s5_rate_mismatch``       the query is acquired at a different sampling rate
``s6_new_domain``          rehabilitation / gym / daily-living labels no model has trained on
``s7_device_set``          the number of worn devices differs between enrolment and deployment
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
import time
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import baselines
from baselines import scoring
from baselines.data import load_eval_stream, load_multi_device_stream, source_slice_fingerprint
from data.scripts.curate.compatibility import PLACEMENT_SITE
from model.support.factory import (
    EVIDENCE_AWARE_ARCHITECTURE, EVIDENCE_AWARE_READOUTS,
    EVIDENCE_GATED_ARCHITECTURE, EVIDENCE_GATED_READOUTS, EVIDENCE_GATED_SEMANTIC_READOUTS,
)
from data.scripts.curate.deployment_policy import MULTI_DEVICE_EVAL_CELLS, get_stream_spec

from .partial_coverage import (
    CoverageCell,
    choose_hidden_candidates,
    hide_supports,
    equal_weight_normalized_fusion_predictions,
    classwise_neighbor_scores,
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
    FEATURE_CACHE_SCHEMA,
    FeatureMemoryCache,
    PRIMARY_BASELINES,
    SEED,
    TRAINING_BANK_ZERO_SHOT,
    _build_training_reference_bank,
    _file_hash,
    _halo_contextual_residual_predictions,
    _halo_evidence_gated_predictions,
    _halo_residual_predictions,
    _evidence_gated_readout_spec,
    halo_acquisition_rows,
    _load_or_encode,
    _aligned_labels,
    _native_capabilities,
    _parameter_count_m,
    build_manifest,
    manifest_fingerprint,
    QueryPlan,
)
from data.datasets.mmfit.protocol import partition_rows as mmfit_partition_rows
from data.datasets.mobiact.protocol import (
    candidate_labels as mobiact_candidate_labels,
    partition_rows as mobiact_partition_rows,
)
from model.support.factory import (
    CONTEXTUAL_CHECKPOINT_ARCHITECTURES, LEARNED_CLASSIFIER_ARCHITECTURES,
)
from training.tokenizer.eval_transfer import build_encoder
from halo.paths import CACHE_DIR

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
)
PROSPECTIVE_NEW_DOMAIN_CELLS = (("mobiact", "phone_trouser_pocket"),)

RATE_TARGETS = (20.0, 25.0, 100.0)
ACTIVE_SCENARIOS = (
    "s1_partial_coverage",
    "s2_cross_placement",
    "s3_cross_dataset",
    "s4_missing_modality",
    "s5_rate_mismatch",
    "s6_new_domain",
    "s7_device_set",
)
SCENARIO_PROTOCOL = "deployment-scenarios-v5-20260918"
SCENARIO_RESULT_SCHEMA = "deployment-scenarios-results-v5-20260918"
EVALUATION_ROOT = CACHE_DIR / "evaluations"
DEFAULT_SHARED_FEATURE_CACHE = EVALUATION_ROOT / f"shared_{FEATURE_CACHE_SCHEMA}"
_WITHIN_PLAN_CACHE: dict[tuple, tuple] = {}
_WITHIN_CROSS_SUBJECT_CACHE: dict[tuple, tuple] = {}

_REGION = {
    "wrist": "arm", "forearm": "arm", "upper_arm": "arm", "hand": "arm", "ear": "head",
    "waist": "torso", "belt": "torso", "hip": "torso", "chest": "torso", "back": "torso",
    "torso": "torso", "head": "head",
    "thigh": "leg", "pocket": "leg", "knee": "leg", "shin": "leg", "calf": "leg",
    "ankle": "leg", "gastrocnemius": "leg", "hamstrings": "leg", "tibialis": "leg",
    "rectus_femoris": "leg",
}


def _existing_feature_cache_dirs(write_dir: Path) -> tuple[Path, ...]:
    """Find prior compatible cache directories; individual entries remain fully validated.

    Historical evaluations stored their content-addressed arrays below each result directory.
    Searching those directories lets a scenario run reuse exact checkpoint/stream matches without
    copying files or trusting a stale filename.  ``_load_or_encode`` still verifies schema, cache
    key, source fingerprint, row count, shape, and finiteness before accepting an entry.
    """
    if not EVALUATION_ROOT.exists():
        return ()
    write_dir = write_dir.resolve()
    roots = {
        meta.parent.resolve()
        for meta in EVALUATION_ROOT.rglob("*.json")
        if meta.with_suffix(".npy").is_file() and meta.parent.resolve() != write_dir
    }
    return tuple(sorted(roots, key=str))


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


@lru_cache(maxsize=64)
def _load(dataset: str, stream_id: str, window_seconds: float):
    return load_eval_stream(dataset, stream_id, alignment="native",
                            window_seconds=window_seconds, apply_quality_screen=True,
                            candidate_labels=(mobiact_candidate_labels(window_seconds)
                                              if dataset == "mobiact" else None))


@lru_cache(maxsize=32)
def _composite(dataset: str, device_ids, window_seconds: float):
    return load_multi_device_stream(dataset, tuple(device_ids), alignment="native",
                                    window_seconds=window_seconds, apply_quality_screen=True)


def _within(stream, k: int, seed: int) -> tuple:
    key = (source_slice_fingerprint(stream), int(k), int(seed))
    if key not in _WITHIN_PLAN_CACHE:
        _WITHIN_PLAN_CACHE[key] = tuple(build_manifest(stream, k, seed=seed))
    return _WITHIN_PLAN_CACHE[key]


def _within_cross_subject(stream, k: int, seed: int) -> tuple:
    if k == 0:
        return _within(stream, k, seed)
    key = (source_slice_fingerprint(stream), int(k), int(seed))
    cached = _WITHIN_CROSS_SUBJECT_CACHE.get(key)
    if cached is not None:
        return cached
    cross = build_cross_manifest(
        stream, stream, k, seed=seed, relation="within_cross_subject", same_subject=False,
    )
    plans = tuple(QueryPlan(
        query=plan.query,
        support=tuple(row - cross.offset for row in plan.support),
        support_labels=plan.support_labels,
    ) for plan in cross.plans)
    _WITHIN_CROSS_SUBJECT_CACHE[key] = plans
    return plans


def device_set_variants(device_ids: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...], int], ...]:
    """Deterministic, rotation-balanced Scenario 7 subset pairs.

    Every source placement takes each directional role where the available device count permits.
    The output contains only mismatches; the caller constructs a same-query-set matched control
    for every returned pair.
    """
    devices = tuple(device_ids)
    if len(devices) < 2:
        return ()
    order = {device: index for index, device in enumerate(devices)}
    canonical = lambda values: tuple(sorted(values, key=order.__getitem__))
    full = devices
    rows: list[tuple[str, tuple[str, ...], tuple[str, ...], int]] = []
    for device in devices:
        rows.append(("support_single_query_full", full, (device,), 1))
        rows.append(("support_full_query_single", (device,), full, 1))
    if len(devices) >= 3:
        for device in devices:
            leave_one = tuple(value for value in devices if value != device)
            rows.append(("support_leave_one_out_query_full", full, leave_one, 2))
        for i, device in enumerate(devices):
            query = canonical((device, devices[(i + 1) % len(devices)]))
            support = canonical((devices[(i + 1) % len(devices)], devices[(i + 2) % len(devices)]))
            rows.append(("partial_overlap", query, support, 2))
    for i, device in enumerate(devices):
        rows.append(("disjoint_single", (device,), (devices[(i + 1) % len(devices)],), 3))
    return tuple(rows)


def _partitioned_plans(stream, k: int, seed: int, *, query_rows, support_rows, relation: str) -> tuple[QueryPlan, ...]:
    """Build a single-stream manifest with explicit reference/query partitions."""
    if k == 0:
        labels = _aligned_labels(stream)
        return tuple(
            QueryPlan(query=int(row), support=(), support_labels=())
            for row in query_rows if labels[row] in stream.eval_labels
        )
    cross = build_cross_manifest(
        stream, stream, k, seed=seed, relation=relation,
        candidates=tuple(stream.eval_labels), query_rows=query_rows, support_rows=support_rows,
    )
    return tuple(QueryPlan(
        query=plan.query,
        support=tuple(row - cross.offset for row in plan.support),
        support_labels=plan.support_labels,
    ) for plan in cross.plans)


def _mmfit_new_domain_plans(stream, k: int, seed: int) -> tuple[tuple[QueryPlan, ...], str]:
    """Use MM-Fit's published participant split without treating workout IDs as people."""
    plans = _partitioned_plans(
        stream, k, seed, query_rows=mmfit_partition_rows(stream, "query"),
        support_rows=mmfit_partition_rows(stream, "reference"),
        relation="mmfit_published_participant_split",
    )
    return plans, "published_participant_split"


def _mobiact_new_domain_plans(stream, k: int, seed: int) -> tuple[tuple[QueryPlan, ...], str]:
    plans = _partitioned_plans(
        stream, k, seed, query_rows=mobiact_partition_rows(stream, "query"),
        support_rows=mobiact_partition_rows(stream, "reference"),
        relation="mobiact_prospective_subject_split",
    )
    return plans, "prospective_subject_split"


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


def _partitioned_zero_support_cross(query, support, query_rows) -> tuple[tuple[QueryPlan, ...], tuple[str, ...]]:
    """Zero-support cross-view cell restricted to a frozen query partition."""
    candidates, _ = shared_candidates(query, support)
    labels = _aligned_labels(query)
    plans = tuple(
        QueryPlan(query=int(row), support=(), support_labels=())
        for row in query_rows if labels[row] in candidates
    )
    return plans, candidates


def _matched_within_reference(
    cross,
    query,
    k: int,
    *,
    seed: int,
    same_subject: bool | None,
    relation: str,
    query_rows=None,
    support_rows=None,
) -> tuple[tuple[QueryPlan, ...], tuple[QueryPlan, ...]]:
    """Pair a cross-source condition with an in-query-source reference on identical queries.

    ``build_cross_manifest(query, query, ...)`` is used to enforce the requested subject relation.
    Its support indexes address a concatenated matrix, so they are translated back to the one
    within-stream feature matrix before returning.
    """
    selected_query_rows = (tuple(query_rows) if query_rows is not None
                           else tuple(plan.query for plan in cross.plans))
    reference = build_cross_manifest(
        query, query, k, seed=seed, relation=relation, same_subject=same_subject,
        candidates=cross.candidates, query_rows=selected_query_rows,
        support_rows=support_rows,
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


# ------------------------------------------------------------------ task builders


def build_tasks(scenario: str, k: int, window_seconds: float, *, seed: int,
                coverage: float, limit: int | None = None,
                failures: list[dict] | None = None,
                include_prospective: bool = False) -> list[Task]:
    """Every cell of one scenario at one evidence budget and one enrolment size."""
    if limit is not None and limit < 1:
        return []
    tasks: list[Task] = []

    def safe_load(dataset: str, stream_id: str):
        try:
            return _load(dataset, stream_id, window_seconds)
        except Exception as exc:  # one unavailable stream must not erase sibling scenario cells
            if failures is not None:
                failures.append({
                    "scenario": scenario, "k": int(k),
                    "window_seconds": float(window_seconds), "stage": "load_stream",
                    "dataset": dataset, "stream": stream_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            return None

    def safe_composite(dataset: str, device_ids: tuple[str, ...]):
        try:
            return _composite(dataset, device_ids, window_seconds)
        except Exception as exc:
            if failures is not None:
                failures.append({
                    "scenario": scenario, "k": int(k),
                    "window_seconds": float(window_seconds), "stage": "load_stream",
                    "dataset": dataset, "stream": "+".join(device_ids),
                    "error": f"{type(exc).__name__}: {exc}",
                })
            return None

    def complete() -> bool:
        # Disclosure-only unavailable rungs do not consume the smoke/task budget; the limit must
        # still exercise real scoring work (and, for matched scenarios, control plus perturbation).
        return limit is not None and sum(bool(task.plans) for task in tasks) >= limit

    if scenario == "s1_partial_coverage":
        if k == 0:
            return tasks
        for dataset, stream_id in SEALED_SINGLE_CELLS:
            stream = safe_load(dataset, stream_id)
            if stream is None:
                continue
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
            if complete():
                return tasks

    elif scenario == "s2_cross_placement":
        for dataset, streams in CROSS_PLACEMENT_STREAMS.items():
            for query_id in streams:
                for support_id in streams:
                    if query_id == support_id:
                        continue
                    query = safe_load(dataset, query_id)
                    support = safe_load(dataset, support_id)
                    if query is None or support is None:
                        continue
                    if k == 0:
                        plans, candidates = _zero_support_cross_plans(query, support)
                        if plans:
                            tasks.append(Task(
                                scenario, f"{dataset}/{query_id}<-{support_id}/zero_support",
                                query, support, plans, candidates, stream_rows(query),
                                {"L": 0, "S": 3, "P": 0, "C": 0},
                                meta={"distance": anatomical_distance(dataset, query_id, support_id),
                                      "subject_relation": "not_applicable_zero_support"}))
                            if complete():
                                return tasks
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
                            # Absence of a planned severity rung is itself a result. Persist it so
                            # a report cannot silently look like only cross-subject deployment was
                            # intended.
                            tasks.append(Task(
                                scenario,
                                f"{dataset}/{query_id}<-{support_id}/{relation}/unavailable",
                                query, support, (), cross.candidates, cross.offset,
                                {"L": 0, "S": 0, "P": severity, "C": 0},
                                meta={
                                    "distance": anatomical_distance(dataset, query_id, support_id),
                                    "subject_relation": relation,
                                    "skipped": (
                                        "no execution-disjoint supports satisfy the requested "
                                        f"{relation} relation for every candidate"
                                    ),
                                },
                            ))
                            if complete():
                                return tasks
                            continue
                        matched_group = f"{dataset}/{query_id}<-{support_id}/{relation}/k{k}"
                        tasks.append(Task(
                            scenario, f"{dataset}/{query_id}<-{query_id}/{relation}/matched_control",
                            query, query, reference_plans, cross.candidates, stream_rows(query),
                            {"L": 0, "S": 0, "P": 0, "C": 0},
                            meta={"matched_group": matched_group, "condition": "control",
                                  "subject_relation": relation},
                        ))
                        if complete():
                            return tasks
                        tasks.append(Task(
                            scenario, f"{dataset}/{query_id}<-{support_id}/{relation}",
                            query, support, cross_plans, cross.candidates, cross.offset,
                            {"L": 0, "S": 0, "P": severity, "C": 0},
                            meta={"distance": anatomical_distance(dataset, query_id, support_id),
                                  "subject_relation": relation, "matched_group": matched_group,
                                  "condition": "scenario"}))
                        if complete():
                            return tasks

        # MM-Fit uses its published participant-disjoint split. The workout ids stored in the grid
        # are not treated as subject ids, so this deliberately avoids the generic same/cross-workout
        # branches above.
        mmfit_streams = ("left_wrist", "right_wrist", "right_pocket", "left_ear")
        for query_id in mmfit_streams:
            for support_id in mmfit_streams:
                if query_id == support_id:
                    continue
                query = safe_load("mmfit", query_id)
                support = safe_load("mmfit", support_id)
                if query is None or support is None:
                    continue
                query_rows = mmfit_partition_rows(query, "query")
                support_rows = mmfit_partition_rows(support, "reference")
                if k == 0:
                    plans, candidates = _partitioned_zero_support_cross(query, support, query_rows)
                    if plans:
                        tasks.append(Task(
                            scenario, f"mmfit/{query_id}<-{support_id}/published_split/zero_support",
                            query, support, plans, candidates, stream_rows(query),
                            {"L": 3, "S": 3, "P": 2, "C": 2},
                            meta={"subject_relation": "published_participant_test_zero_support",
                                  "condition": "scenario", "protocol": "mmfit-scenarios-v1"},
                        ))
                    if complete():
                        return tasks
                    continue
                cross = build_cross_manifest(
                    query, support, k, seed=seed, relation="mmfit_published_cross_device",
                    query_rows=query_rows, support_rows=support_rows,
                )
                cross_plans, reference_plans = _matched_within_reference(
                    cross, query, k, seed=seed, same_subject=None,
                    relation="mmfit_published_same_device_reference",
                    query_rows=query_rows,
                    support_rows=mmfit_partition_rows(query, "reference"),
                )
                if not cross_plans:
                    continue
                matched_group = f"mmfit/{query_id}<-{support_id}/published_split/k{k}"
                tasks.append(Task(
                    scenario, f"mmfit/{query_id}<-{query_id}/published_split/matched_control",
                    query, query, reference_plans, cross.candidates, stream_rows(query),
                    {"L": 3, "S": 0, "P": 0, "C": 0},
                    meta={"matched_group": matched_group, "condition": "control",
                          "subject_relation": "published_participant_split"},
                ))
                if complete():
                    return tasks
                tasks.append(Task(
                    scenario, f"mmfit/{query_id}<-{support_id}/published_split",
                    query, support, cross_plans, cross.candidates, cross.offset,
                    {"L": 3, "S": 0, "P": 2, "C": 2},
                    meta={"matched_group": matched_group, "condition": "scenario",
                          "subject_relation": "published_participant_split",
                          "protocol": "mmfit-scenarios-v1"},
                ))
                if complete():
                    return tasks

    elif scenario == "s3_cross_dataset":
        for (q_ds, q_id), (s_ds, s_id), region in CROSS_DATASET_PAIRS:
            query = safe_load(q_ds, q_id)
            support = safe_load(s_ds, s_id)
            if query is None or support is None:
                continue
            if k == 0:
                plans, candidates = _zero_support_cross_plans(query, support)
                if plans:
                    tasks.append(Task(
                        scenario, f"{q_ds}/{q_id}<-{s_ds}/{s_id}/zero_support",
                        query, support, plans, candidates, stream_rows(query),
                        {"L": 0, "S": 3, "P": 3, "C": 0},
                        meta={"region_match": region, "support_dataset": s_ds,
                              "subject_relation": "not_applicable_zero_support"}))
                    if complete():
                        return tasks
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
            if complete():
                return tasks
            tasks.append(Task(
                scenario, f"{q_ds}/{q_id}<-{s_ds}/{s_id}", query, support,
                cross_plans, cross.candidates, cross.offset,
                {"L": 0, "S": 0, "P": 3, "C": 0},
                meta={"region_match": region, "support_dataset": s_ds,
                      "subject_relation": "cross_dataset_identity_unmapped",
                      "matched_group": matched_group, "condition": "scenario"}))
            if complete():
                return tasks

    elif scenario == "s4_missing_modality":
        if k == 0:
            return tasks
        for dataset, stream_id in SEALED_SINGLE_CELLS:
            full = safe_load(dataset, stream_id)
            if full is None:
                continue
            base_plans = _within_cross_subject(full, k, seed)
            try:
                accel = derive_accel_only(full)
            except ValueError as exc:
                tasks.append(Task(scenario, f"{dataset}/{stream_id}", full, full, (), (), 0,
                                  {"L": 0, "S": 0, "P": 0, "C": 1},
                                  meta={"skipped": str(exc)}))
                if complete():
                    return tasks
                continue
            tasks.append(Task(
                scenario, f"{dataset}/{stream_id}/matched_full_control", full, full,
                base_plans, tuple(full.eval_labels), stream_rows(full),
                {"L": 0, "S": 0, "P": 0, "C": 0},
                meta={"matched_parent": f"{dataset}/{stream_id}/modality",
                      "condition": "control", "subject_relation": "cross_subject"},
            ))
            if complete():
                return tasks
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
                if complete():
                    return tasks

    elif scenario == "s5_rate_mismatch":
        if k == 0:
            return tasks
        for dataset, stream_id in SEALED_SINGLE_CELLS:
            full = safe_load(dataset, stream_id)
            if full is None:
                continue
            base_plans = _within_cross_subject(full, k, seed)
            tasks.append(Task(
                scenario, f"{dataset}/{stream_id}/matched_full_control", full, full,
                base_plans, tuple(full.eval_labels), stream_rows(full),
                {"L": 0, "S": 0, "P": 0, "C": 0},
                meta={"matched_parent": f"{dataset}/{stream_id}/rate", "condition": "control",
                      "subject_relation": "cross_subject"},
            ))
            if complete():
                return tasks
            for target in RATE_TARGETS:
                if abs(target - float(full.rate_hz)) < 1e-9:
                    continue
                try:
                    query = derive_resampled(full, target)
                except (ValueError, RuntimeError) as exc:
                    tasks.append(Task(
                        scenario, f"{dataset}/{stream_id}/query@{target:g}Hz/unavailable",
                        full, full, (), tuple(full.eval_labels), stream_rows(full),
                        {"L": 0, "S": 0, "P": 0, "C": 2},
                        meta={
                            "query_rate_hz": target,
                            "support_rate_hz": float(full.rate_hz),
                            "condition": "scenario",
                            "subject_relation": "cross_subject",
                            "skipped": f"rate perturbation unavailable: {type(exc).__name__}: {exc}",
                        },
                    ))
                    continue
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
                if complete():
                    return tasks

    elif scenario == "s6_new_domain":
        cells = NEW_DOMAIN_CELLS + (
            PROSPECTIVE_NEW_DOMAIN_CELLS if include_prospective else ()
        )
        for dataset, stream_id in cells:
            stream = safe_load(dataset, stream_id)
            if stream is None:
                continue
            if dataset == "mmfit":
                plans, subject_relation = _mmfit_new_domain_plans(stream, k, seed)
            elif dataset == "mobiact":
                plans, subject_relation = _mobiact_new_domain_plans(stream, k, seed)
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
            if complete():
                return tasks
            if k > 0:
                cell = _coverage_cell(stream.eval_labels, seed=seed, dataset=dataset, stream=stream_id,
                                      window_seconds=window_seconds, coverage=coverage)
                tasks.append(Task(
                    scenario, f"{dataset}/{stream_id}/partial_coverage", stream, stream,
                    tuple(hide_supports(plans, cell)), tuple(stream.eval_labels),
                    stream_rows(stream), {"L": 3, "S": 2, "P": 0, "C": 0}, coverage=cell,
                    meta={"subject_relation": subject_relation}))
                if complete():
                    return tasks

    elif scenario == "s7_device_set":
        for cell_spec in MULTI_DEVICE_EVAL_CELLS:
            dataset, device_ids = cell_spec.dataset, tuple(cell_spec.stream_ids)
            composite = safe_composite(dataset, device_ids)
            if composite is None:
                continue
            if k == 0:
                if dataset == "mmfit":
                    plans, candidates = _partitioned_zero_support_cross(
                        composite, composite, mmfit_partition_rows(composite, "query"),
                    )
                else:
                    plans, candidates = _zero_support_cross_plans(composite, composite)
                if plans:
                    tasks.append(Task(
                        scenario, f"{dataset}/zero_support", composite, composite, plans,
                        candidates, stream_rows(composite), {"L": 0, "S": 3, "P": 0, "C": 3},
                        meta={"device_variant": "zero_support",
                              "subject_relation": "not_applicable_zero_support"}))
                    if complete():
                        return tasks
                continue
            stream_cache: dict[tuple[str, ...], object] = {device_ids: composite}

            def subset_stream(subset: tuple[str, ...]):
                if subset not in stream_cache:
                    stream_cache[subset] = (safe_load(dataset, subset[0]) if len(subset) == 1
                                            else safe_composite(dataset, subset))
                return stream_cache[subset]

            for variant, query_ids, support_ids, severity in device_set_variants(device_ids):
                query, support = subset_stream(query_ids), subset_stream(support_ids)
                if query is None or support is None:
                    continue
                if dataset == "mmfit":
                    query_rows = mmfit_partition_rows(query, "query")
                    support_rows = mmfit_partition_rows(support, "reference")
                    cross = build_cross_manifest(
                        query, support, k, seed=seed, relation=f"mmfit_{variant}",
                        query_rows=query_rows, support_rows=support_rows,
                    )
                    reference_subject = None
                    reference_rows = mmfit_partition_rows(query, "reference")
                else:
                    query_rows = None
                    cross = build_cross_manifest(
                        query, support, k, seed=seed, relation=variant, same_subject=False,
                    )
                    reference_subject = False
                    reference_rows = None
                cross_plans, reference_plans = _matched_within_reference(
                    cross, query, k, seed=seed, same_subject=reference_subject,
                    relation=f"s7_reference_{variant}",
                    query_rows=query_rows, support_rows=reference_rows,
                )
                if not cross_plans:
                    continue
                matched_group = f"{dataset}/{variant}/{'+'.join(query_ids)}<-{'+'.join(support_ids)}/k{k}"
                tasks.append(Task(
                    scenario, f"{dataset}/{variant}/matched_control", query, query,
                    reference_plans, cross.candidates, stream_rows(query),
                    {"L": 0, "S": 0, "P": 0, "C": 0},
                    meta={"device_variant": variant, "condition": "control",
                          "matched_group": matched_group,
                          "subject_relation": ("published_participant_split" if dataset == "mmfit"
                                               else "cross_subject")},
                ))
                tasks.append(Task(
                    scenario, f"{dataset}/{variant}", query, support, cross_plans,
                    cross.candidates, cross.offset, {"L": 0, "S": 0, "P": 0, "C": severity},
                    meta={"device_variant": variant,
                          "query_devices": len(getattr(query, "devices", [1])),
                          "support_devices": len(getattr(support, "devices", [1])),
                          "query_device_ids": query_ids, "support_device_ids": support_ids,
                          "device_jaccard": len(set(query_ids) & set(support_ids)) /
                          len(set(query_ids) | set(support_ids)),
                          "device_mismatch_severity": severity,
                          "condition": "scenario", "matched_group": matched_group,
                          "subject_relation": ("published_participant_split" if dataset == "mmfit"
                                               else "cross_subject")}))
                if complete():
                    return tasks

    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return tasks


# --------------------------------------------------------------------- scoring


def _requires_cross_support_features(task: Task, k: int) -> bool:
    """Whether scoring can consume support representations for this task."""
    return bool(task.cross and int(k) > 0)


def _evidence_diagnostic_requests(
    selected_readouts: frozenset[str] | None, *, include_oracle: bool,
) -> tuple[bool, bool]:
    """Return explicit branch/oracle requests; omitted readouts mean deployable outputs only."""
    branches = (
        selected_readouts is not None
        and bool(set(EVIDENCE_AWARE_READOUTS) & set(selected_readouts))
    )
    oracle = include_oracle or (
        selected_readouts is not None
        and "halo-classifier-oracle-better-branch" in selected_readouts
    )
    return branches, oracle


def score_task(task: Task, *, models, device, cache_dir, halo_checkpoint, bootstrap,
               banks, k: int, window_seconds: float, halo_state=None,
               halo_has_classifier: bool = False,
               halo_requires_acquisition: bool = False,
               halo_architecture: str | None = None,
               selected_readouts: frozenset[str] | None = None,
               include_oracle_diagnostics: bool = False,
               provider_states=None, prediction_sink=None, cache_read_dirs=(),
               feature_memory_cache=None, manifest: str | None = None,
               timing_sink: dict[str, float] | None = None) -> list[dict]:
    """Every readout for every model on one task's episodes."""
    if not task.plans:
        severity = {f"severity_{axis}": value for axis, value in task.severity.items()}
        return [{
            "scenario": task.scenario, "variant": task.variant, "model": name,
            "readout": "all", "status": "n/a",
            "reason": task.meta.get("skipped", "no honest episode could be formed"),
            "k": k, "window_seconds": float(window_seconds), **severity, **task.meta,
        } for name in models]

    coverage = task.coverage or CoverageCell(
        supported=tuple(task.candidates), hidden=(), coverage=1.0, requested_coverage=1.0)
    roster = SimpleNamespace(eval_labels=list(task.candidates))
    manifest = manifest or manifest_fingerprint(task.plans)
    common = dict(k=k, window_seconds=window_seconds, bootstrap=bootstrap, manifest=manifest)
    severity_meta = {**{f"severity_{axis}": value for axis, value in task.severity.items()},
                     **task.meta, "scenario": task.scenario, "variant": task.variant,
                     "cross_stream": task.cross,
                     "support_source": f"{task.support_stream.dataset}/"
                                       f"{getattr(task.support_stream, 'stream', getattr(task.support_stream, 'cell_id', ''))}"}
    rows: list[dict] = []
    provider_states = {} if provider_states is None else provider_states
    prediction_sink = [] if prediction_sink is None else prediction_sink

    def append_emitted(predictions, *, readout: str, diagnostic_only: bool = False) -> None:
        emitted = emit_rows(task.query_stream, task.plans, predictions, coverage,
                            model=name, readout=readout, extra=extra_base, **common)
        if task.coverage is not None:
            # The combined split contains both conditions; each conditional split receives the
            # severity it actually experiences rather than its cell's worst-case severity.
            for row in emitted:
                split = row.get("coverage_split")
                row["severity_S"] = 1 if split == "truth_enrolled" else 2 if split == "truth_unenrolled" else 2
        if name == "halo" and readout.startswith("halo-classifier"):
            for row in emitted:
                row["parameters_m"] = round(_parameter_count_m(
                    name, halo_state, halo_checkpoint=halo_checkpoint, include_classifier=True,
                ), 6)
                if diagnostic_only:
                    row["diagnostic_only"] = True
                    row["deployable"] = False
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

    def append_inapplicable(readout: str, reason: str) -> None:
        rows.append({"model": name, "readout": readout, "status": "n/a", "reason": reason,
                     **extra_base, **common})

    for name in models:
        model_started = time.perf_counter()
        try:
            default_baseline_fusion = selected_readouts is None and name != "halo"
            baseline_fusion_requested = name != "halo" and (
                default_baseline_fusion
                or "equal-weight-normalized-fusion" in (selected_readouts or ())
            )
            halo_classifier_requested = name == "halo" and halo_has_classifier and (
                selected_readouts is None or "halo-classifier" in selected_readouts
                or bool(set(EVIDENCE_AWARE_READOUTS) & set(selected_readouts or ()))
                or bool(set(EVIDENCE_GATED_READOUTS) & set(selected_readouts or ()))
                or "halo-classifier-oracle-better-branch" in (selected_readouts or ())
            )
            companion_1nn_required = k > 0 and (
                baseline_fusion_requested or halo_classifier_requested
            )
            if name != "halo" and name not in provider_states:
                provider_states[name] = baselines.REGISTRY[name].setup_features(device)
            query_features, query_fingerprint = _load_or_encode(
                name=name, stream=task.query_stream, device=device, cache_dir=cache_dir,
                halo_checkpoint=halo_checkpoint, baseline_state=provider_states.get(name),
                halo_state=halo_state, cache_read_dirs=cache_read_dirs,
                memory_cache=feature_memory_cache)
            # A zero-support episode uses only the query and its declared semantic path. Loading
            # cross-stream support here is wasted work and can incorrectly reject a native,
            # fixed-width zero-shot representation merely because the model's enrollment feature
            # width depends on the acquisition configuration (for example NormWear channels).
            if _requires_cross_support_features(task, k):
                support_features, support_fingerprint = _load_or_encode(
                    name=name, stream=task.support_stream, device=device, cache_dir=cache_dir,
                    halo_checkpoint=halo_checkpoint, baseline_state=provider_states.get(name),
                    halo_state=halo_state, cache_read_dirs=cache_read_dirs,
                    memory_cache=feature_memory_cache)
                if query_features.ndim != 2 or support_features.ndim != 2:
                    raise baselines.UnsupportedEvaluationCell(
                        "query and support encoders must return rank-2 representations"
                    )
                if query_features.shape[1] != support_features.shape[1]:
                    raise baselines.UnsupportedEvaluationCell(
                        "query and support acquisition configurations produce incompatible "
                        f"representation dimensions ({query_features.shape[1]} versus "
                        f"{support_features.shape[1]})"
                    )
                features = np.concatenate([query_features, support_features], axis=0)
            else:
                features = query_features
                support_fingerprint = query_fingerprint if not task.cross else None
            acquisitions = None
            if name == "halo" and halo_state is not None and halo_requires_acquisition:
                query_acquisition = halo_acquisition_rows(task.query_stream, halo_state[0], device)
                if _requires_cross_support_features(task, k):
                    support_acquisition = halo_acquisition_rows(
                        task.support_stream, halo_state[0], device,
                    )
                    acquisitions = np.concatenate(
                        [query_acquisition, support_acquisition], axis=0,
                    )
                else:
                    acquisitions = query_acquisition
            extra_base = {**severity_meta, "n_candidates": len(task.candidates),
                          # Keep the legacy field for readers that predate cross-stream scoring,
                          # but record both sides explicitly so a result is fully auditable.
                          "feature_fingerprint": query_fingerprint,
                          "query_feature_fingerprint": query_fingerprint,
                          "support_feature_fingerprint": support_fingerprint,
                          "parameters_m": round(_parameter_count_m(
                              name, halo_state, halo_checkpoint=halo_checkpoint,
                              include_classifier=False,
                          ), 6), **_native_capabilities(name)}

            neighbor_scores = None
            if k > 0 and (companion_1nn_required or selected_readouts is None or bool(
                    {"1nn", "equal-weight-normalized-fusion"} & selected_readouts)):
                neighbor_scores = classwise_neighbor_scores(
                    features, task.candidates, task.plans, device=device,
                )

            if k > 0:
                # Cosine 1-NN is a mandatory companion whenever a fixed fusion or learned HALO
                # classifier row is reported. It exposes whether semantic fusion/classification
                # helps or merely dilutes the underlying representation. Explicit readout subsets
                # may add prototype/ridge but cannot suppress this fairness control.
                requested_support = (
                    {"1nn"} if selected_readouts is None else
                    set(selected_readouts & {"1nn", "prototype", "ridge"})
                )
                if companion_1nn_required:
                    requested_support.add("1nn")
                support_readouts = frozenset(requested_support)
                if support_readouts is None or support_readouts:
                    for readout, predicted in support_only_predictions(
                            features, task.candidates, task.plans, coverage, device=device,
                            readouts=support_readouts, classwise_scores=neighbor_scores).items():
                        append_emitted(predicted, readout=readout)

            text = None
            needs_text = (
                selected_readouts is None
                or "zero-shot-native-or-bridge" in selected_readouts
                or baseline_fusion_requested
            )
            if needs_text and (k == 0 or task.coverage is not None or baseline_fusion_requested):
                if name in TRAINING_BANK_ZERO_SHOT and (
                        name != "halo" or not halo_has_classifier or task.coverage is not None):
                    if name not in banks:
                        banks[name] = _build_training_reference_bank(
                            name=name, device=device,
                            cache_dir=EVALUATION_ROOT / "zero_shot_feature_cache",
                            halo_checkpoint=halo_checkpoint, halo_state=halo_state,
                            baseline_state=provider_states.get(name),
                            cache_read_dirs=cache_read_dirs,
                            memory_cache=feature_memory_cache,
                            window_seconds=window_seconds)
                    bank_features, bank_labels, train_labels, _ = banks[name]
                    text = conse_scores(query_features, bank_features, bank_labels, train_labels,
                                        task.candidates, device)
                elif name != "halo" and baselines.REGISTRY[name].supports_native_zero_shot():
                    state = provider_states[name]
                    native_features, _ = _load_or_encode(
                        name=name, stream=task.query_stream, device=device, cache_dir=cache_dir,
                        halo_checkpoint=halo_checkpoint, baseline_state=state,
                        cache_read_dirs=cache_read_dirs, memory_cache=feature_memory_cache,
                        feature_role="native_zero_shot",
                    )
                    text = native_text_scores(name, native_features, task.candidates, state, device)

            if k == 0:
                if (name == "halo" and halo_checkpoint is not None and halo_has_classifier
                        and (selected_readouts is None or "halo-classifier" in selected_readouts)):
                    predicted = _halo_residual_predictions(features, roster, task.plans,
                                                          halo_checkpoint, device,
                                                          acquisitions=acquisitions)
                    append_emitted(predicted, readout="halo-classifier")
                elif (name != "halo" and text is not None and baseline_fusion_requested):
                    # With no enrolled supports the fixed combiner has only its semantic
                    # component. Keep the readout name consistent with enrolled baseline rows;
                    # do not imply that a support term was available.
                    predicted = [task.candidates[index] for index in text.argmax(axis=1).tolist()]
                    append_emitted(predicted, readout="equal-weight-normalized-fusion")
                elif (text is not None and (selected_readouts is None
                                             or "zero-shot-native-or-bridge" in selected_readouts)):
                    predicted = [task.candidates[index] for index in text.argmax(axis=1).tolist()]
                    append_emitted(predicted, readout="zero-shot-native-or-bridge")
                elif selected_readouts is None or "zero-shot-native-or-bridge" in selected_readouts:
                    rows.append({"model": name, "readout": "zero-shot", "status": "inapplicable",
                                 "reason": "model exposes neither a native nor configured bridge zero-shot path",
                                 **extra_base, **common})
            elif task.coverage is not None or baseline_fusion_requested:
                if (text is not None and (selected_readouts is None
                                          or "equal-weight-normalized-fusion" in selected_readouts)):
                    fused = equal_weight_normalized_fusion_predictions(
                        text, features, task.candidates, task.plans, coverage,
                        classwise_scores=neighbor_scores,
                    )
                    append_emitted(fused, readout="equal-weight-normalized-fusion")
                elif (name != "halo" and (selected_readouts is None
                                           or "equal-weight-normalized-fusion" in selected_readouts)):
                    rows.extend(cannot_attempt_rows(
                        task.query_stream, coverage, model=name,
                        readout="equal-weight-normalized-fusion",
                        k=k, window_seconds=window_seconds,
                        reason="no native or configured semantic-score path: hidden candidates are unreachable",
                        scenario=task.scenario, variant=task.variant, extra=extra_base))

            if (name == "halo" and halo_checkpoint is not None and halo_has_classifier and k > 0
                    and (selected_readouts is None or "halo-classifier" in selected_readouts)):
                predicted = _halo_residual_predictions(
                    features, roster, task.plans, halo_checkpoint, device,
                    acquisitions=acquisitions,
                )
                append_emitted(predicted, readout="halo-classifier")
            if (name == "halo" and halo_checkpoint is not None
                    and halo_architecture == EVIDENCE_GATED_ARCHITECTURE
                    and selected_readouts is not None):
                # v4 branch diagnostics are opt-in, exactly like v2's, and there is no oracle row:
                # the branches are auditable decompositions, not a deployable selection rule.
                for readout in EVIDENCE_GATED_READOUTS + EVIDENCE_GATED_SEMANTIC_READOUTS:
                    if readout in selected_readouts:
                        # At k=0 the blend is the semantic branch, so its two halves are exactly
                        # the attribution the new-domain cell needs; support branches are not
                        # defined without enrollment.
                        if k == 0 and readout not in {
                            "halo-classifier-label-meaning-only",
                            *EVIDENCE_GATED_SEMANTIC_READOUTS,
                        }:
                            append_inapplicable(readout, "no enrolled support at k=0")
                            continue
                        if (readout == "halo-classifier-text-off-blend"
                                and any(set(plan.support_labels) < set(task.candidates)
                                        for plan in task.plans)):
                            append_inapplicable(
                                readout, "text-off blend is undefined with unenrolled candidates",
                            )
                            continue
                        branch, lambda_override, trust_override = _evidence_gated_readout_spec(readout)
                        append_emitted(
                            _halo_evidence_gated_predictions(
                                features, roster, task.plans, halo_checkpoint, device,
                                branch=branch, lambda_override=lambda_override,
                                trust_override=trust_override,
                            ),
                            readout=readout,
                        )
            branch_diagnostics_requested, oracle_requested = _evidence_diagnostic_requests(
                selected_readouts, include_oracle=include_oracle_diagnostics,
            )
            if (name == "halo" and halo_checkpoint is not None
                    and halo_architecture == EVIDENCE_AWARE_ARCHITECTURE
                    and (branch_diagnostics_requested or oracle_requested)):
                diagnostic_predictions = {}
                for readout, branch in zip(
                    EVIDENCE_AWARE_READOUTS,
                    ("semantic_status", "support_floor", "refined_support"),
                ):
                    branch_predictions = _halo_contextual_residual_predictions(
                        features, roster, task.plans, halo_checkpoint, device,
                        branch=branch, acquisitions=acquisitions,
                    )
                    diagnostic_predictions[branch] = branch_predictions
                    if selected_readouts is not None and readout in selected_readouts:
                        append_emitted(branch_predictions, readout=readout)
                truth = _aligned_labels(task.query_stream)
                oracle = []
                for plan, support_prediction, semantic_prediction in zip(
                    task.plans, diagnostic_predictions["refined_support"],
                    diagnostic_predictions["semantic_status"],
                ):
                    expected = str(truth[int(plan.query)])
                    oracle.append(
                        support_prediction if support_prediction == expected
                        else semantic_prediction if semantic_prediction == expected
                        else support_prediction
                    )
                if oracle_requested:
                    append_emitted(
                        oracle, readout="halo-classifier-oracle-better-branch",
                        diagnostic_only=True,
                    )
        except baselines.UnsupportedEvaluationCell as exc:
            rows.append({"model": name, "status": "unsupported", "reason": str(exc),
                         "k": k, "window_seconds": float(window_seconds), **severity_meta})
        except Exception as exc:  # Preserve other models' completed rows and make unexpected failures visible.
            rows.append({"model": name, "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                         "k": k, "window_seconds": float(window_seconds), **severity_meta})
        finally:
            if timing_sink is not None:
                timing_sink[name] = timing_sink.get(name, 0.0) + (
                    time.perf_counter() - model_started
                )
    return rows


# ------------------------------------------------------------------------ main


def _task_artifact(
    task: Task, *, k: int, window_seconds: float, compact: bool = False,
    manifest: str | None = None,
) -> dict:
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
        "manifest_fingerprint": manifest or manifest_fingerprint(task.plans),
        "n_plans": len(task.plans),
        "plans": None if compact else [_plan_artifact(task, plan) for plan in task.plans],
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
        "protocol": SCENARIO_PROTOCOL,
        "manifest_generator": "numpy-choice-json-fingerprint-v1",
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
            # Both sides must use one estimand. Otherwise a scenario that introduces a new
            # false-positive class changes its own macro-F1 denominator relative to the control.
            f1_classes = sorted(set(truth) | set(scenario_prediction) | set(control_prediction))
            metrics = scoring.classification_metrics(
                truth, scenario_prediction, f1_classes=f1_classes,
            )
            reference = scoring.classification_metrics(
                truth, control_prediction, f1_classes=f1_classes,
            )
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


class _PairedDeltaTracker:
    """Compute matched deltas while one control/scenario group is still local.

    The audit JSONL retains every decision.  This tracker only changes working memory: controls are
    released immediately after their final associated scenario instead of retaining every paired
    prediction from the complete experiment.
    """

    def __init__(self, tasks: list[Task], *, bootstrap: int):
        self.bootstrap = int(bootstrap)
        self.remaining: dict[str, int] = {}
        for task in tasks:
            parent = task.meta.get("matched_group") or task.meta.get("matched_parent")
            if parent and task.meta.get("condition") == "scenario":
                self.remaining[parent] = self.remaining.get(parent, 0) + 1
        self.controls: dict[str, list[dict]] = {}
        self.output: list[dict] = []

    def consume(self, rows: list[dict]) -> None:
        if not rows:
            return
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            parent = row.get("matched_group") or row.get("matched_parent")
            condition = row.get("condition")
            if parent and condition in {"control", "scenario"}:
                grouped.setdefault((parent, condition), []).append(row)
        for (parent, condition), values in grouped.items():
            compact = [{
                key: row.get(key)
                for key in (
                    "matched_group", "matched_parent", "condition", "model", "readout",
                    "k", "window_seconds", "variant", "query_event_id", "query_group_id",
                    "truth", "prediction",
                )
            } for row in values]
            if condition == "control":
                if self.remaining.get(parent, 0) > 0:
                    if parent in self.controls:
                        raise RuntimeError(f"duplicate matched control for {parent}")
                    self.controls[parent] = compact
                continue
            control = self.controls.get(parent)
            if control is None:
                raise RuntimeError(f"matched scenario appeared before its control: {parent}")
            self.output.extend(_paired_deltas(control + compact, bootstrap=self.bootstrap))
            self.remaining[parent] -= 1
            if self.remaining[parent] == 0:
                del self.controls[parent]

    def finish(self) -> list[dict]:
        incomplete = {key: count for key, count in self.remaining.items() if count > 0}
        if incomplete:
            raise RuntimeError(f"matched scenarios were not consumed: {incomplete}")
        if self.controls:
            raise RuntimeError(f"unused matched controls remain: {sorted(self.controls)}")
        return self.output


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
        "primary_metric", "truth_label_set", "f1_scored_classes",
        "coverage", "requested_coverage", "supported_candidates", "hidden_candidates",
        "severity_L", "severity_S", "severity_P", "severity_C",
        "query_feature_fingerprint", "support_feature_fingerprint",
        "f1_macro_seen", "f1_macro_unseen", "false_enrollment_pull", "parameters_m",
        "coverage_hmean_f1_macro", "coverage_hmean_balanced_accuracy",
        "coverage_hmean_accuracy",
        "native_open_set_labels", "native_support_conditioning",
        "published_few_label_finetuning", "reason",
    ]
    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    shown = rows
    header = [
        "scenario", "variant", "split", "model", "readout", "w", "k", "parameters (M)",
        "native open set", "native support conditioning", "published few-label finetuning",
        "status", "macro F1", "accuracy",
        "primary metric", "balanced accuracy", "truth labels", "seen F1", "unseen F1",
        "false enrollment pull", "coverage harmonic macro F1",
        "coverage harmonic balanced accuracy", "coverage harmonic accuracy",
    ]
    lines = ["# Deployment scenario results", "", "Generated by `run_scenarios`; no result selects a checkpoint.", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in shown:
        values = [
            row.get("scenario", ""), row.get("variant", ""), row.get("coverage_split", "all"),
            row.get("model", ""),
            row.get("readout", ""), row.get("window_seconds", ""), row.get("k", ""),
            row.get("parameters_m", ""), row.get("native_open_set_labels", ""),
            row.get("native_support_conditioning", ""),
            row.get("published_few_label_finetuning", ""), row.get("status", ""),
            row.get("f1_macro", ""), row.get("accuracy", ""), row.get("primary_metric", ""),
            row.get("balanced_accuracy", ""), ", ".join(row.get("truth_label_set", [])),
            row.get("f1_macro_seen", ""), row.get("f1_macro_unseen", ""),
            row.get("false_enrollment_pull", ""),
            row.get("coverage_hmean_f1_macro", ""),
            row.get("coverage_hmean_balanced_accuracy", ""),
            row.get("coverage_hmean_accuracy", ""),
        ]
        lines.append("| " + " | ".join(str(value) if value is not None else "-" for value in values) + " |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=ACTIVE_SCENARIOS,
                        default=list(ACTIVE_SCENARIOS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=["halo", *PRIMARY_BASELINES])
    parser.add_argument(
        "--readouts", nargs="+", choices=(
            "1nn", "prototype", "ridge", "equal-weight-normalized-fusion",
            "zero-shot-native-or-bridge", "halo-classifier",
            *EVIDENCE_AWARE_READOUTS, *EVIDENCE_GATED_READOUTS,
            *EVIDENCE_GATED_SEMANTIC_READOUTS,
            "halo-classifier-oracle-better-branch",
        ), default=None,
        help=("optional readout subset for a controlled diagnostic; omitted reports only "
              "equal-weight normalized fusion plus companion cosine 1-NN for baselines, and the "
              "deployment classifier plus companion cosine 1-NN for HALO"),
    )
    parser.add_argument("--halo-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--include-oracle-diagnostics", action="store_true",
        help="emit the non-deployable ground-truth branch oracle; never included by default",
    )
    parser.add_argument("--k", nargs="+", type=int, default=[0, 1, 4, 8, 32],
                        help="representative scenario curve; override explicitly for a wider sweep")
    parser.add_argument("--window-seconds", nargs="+", type=float, default=[8.0],
                        help="representative scenario duration; the sealed curve uses 4/8/16 seconds")
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap", type=int, default=scoring.BOOTSTRAP_B)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument(
        "--feature-memory-cache-gib", type=float, default=2.0,
        help="bounded RAM cache for repeatedly referenced feature arrays (default: 2 GiB)",
    )
    parser.add_argument(
        "--no-prior-feature-cache-reuse", action="store_true",
        help="do not search prior evaluation directories for independently validated cache hits",
    )
    parser.add_argument("--max-tasks-per-scenario", type=int, default=None)
    parser.add_argument(
        "--include-prospective-mobiact", action="store_true",
        help=("include the separately versioned MobiAct prospective new-domain panel in s6; "
              "it is excluded from the historical scenario suite by default"),
    )
    parser.add_argument(
        "--compact-audit", action="store_true",
        help="deprecated compatibility flag; compact audit is now the default",
    )
    parser.add_argument(
        "--full-audit", action="store_true",
        help="also persist expanded plans and per-query predictions (substantially larger/slower)",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="up to two tasks per scenario, one k, no bootstrap: wiring check only")
    args = parser.parse_args()

    if any(value < 0 for value in args.k):
        parser.error("--k values must be non-negative")
    if any(not np.isfinite(value) or value <= 0 for value in args.window_seconds):
        parser.error("--window-seconds values must be finite and positive")
    if not 0.0 < args.coverage < 1.0:
        parser.error("--coverage must lie strictly between zero and one")
    if args.compact_audit and args.full_audit:
        parser.error("--compact-audit and --full-audit are mutually exclusive")
    if not np.isfinite(args.feature_memory_cache_gib) or args.feature_memory_cache_gib < 0:
        parser.error("--feature-memory-cache-gib must be finite and non-negative")
    unknown = sorted(set(args.models) - set(PRIMARY_BASELINES) - {"halo"})
    if unknown:
        parser.error(f"models outside the registered primary roster: {unknown}")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not repeat a provider")
    if args.readouts is not None and len(set(args.readouts)) != len(args.readouts):
        parser.error("--readouts must not repeat a readout")
    if "halo" in args.models and args.halo_checkpoint is None:
        parser.error("--halo-checkpoint is required when model list includes halo")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    halo_state = None
    halo_has_classifier = False
    halo_requires_acquisition = False
    halo_architecture = None
    if "halo" in args.models:
        blob = torch.load(args.halo_checkpoint, map_location="cpu", weights_only=False)
        halo_state = (build_encoder(blob, device).eval(), _file_hash(args.halo_checkpoint))
        halo_has_classifier = blob.get("architecture_version") in LEARNED_CLASSIFIER_ARCHITECTURES
        halo_architecture = blob.get("architecture_version")
        halo_requires_acquisition = blob.get("architecture_version") in CONTEXTUAL_CHECKPOINT_ARCHITECTURES
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache or DEFAULT_SHARED_FEATURE_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_read_dirs = (() if args.no_prior_feature_cache_reuse
                       else _existing_feature_cache_dirs(cache_dir))
    feature_memory_cache = FeatureMemoryCache(
        int(args.feature_memory_cache_gib * 1024**3),
    )
    ks = sorted(set(args.k))
    windows = sorted(set(args.window_seconds))
    bootstrap = 0 if args.smoke else args.bootstrap
    # Matched scenarios emit control then perturbation. Two tasks exercise the actual comparison;
    # the historical one-task smoke only scored the control and could not catch scenario failures.
    limit = 2 if args.smoke else args.max_tasks_per_scenario
    if args.smoke:
        ks, windows = ks[:1], windows[:1]

    rows: list[dict] = []
    banks: dict[str, tuple] = {}
    failures: list[dict] = []
    paired_delta_rows: list[dict] = []
    provider_states: dict[str, dict] = {}
    manifest_path = args.out / "manifests.jsonl"
    prediction_path = args.out / "predictions.jsonl"
    manifest_path.write_text("")
    prediction_path.write_text("")
    completed_tasks = 0
    completed_cells = 0
    total_cells = len(args.scenarios) * len(windows) * len(ks)
    run_started = time.perf_counter()
    task_seconds: list[float] = []
    build_seconds: dict[str, float] = {}
    model_seconds: dict[str, float] = {}
    _atomic_json(args.out / "run_metadata.json", {
        "schema": SCENARIO_RESULT_SCHEMA,
        "complete": False,
        "status": "running_or_interrupted",
        "models": list(args.models),
        "scenarios": list(args.scenarios),
        "k": ks,
        "window_seconds": windows,
        "include_prospective_mobiact": bool(args.include_prospective_mobiact),
    })

    def checkpoint_progress(*, scenario: str | None, k: int | None,
                            window_seconds: float | None, task_done: int = 0,
                            task_total: int = 0, last_task_seconds: float | None = None) -> None:
        elapsed = time.perf_counter() - run_started
        within_cell = (task_done / task_total) if task_total else 0.0
        completed_equivalent = completed_cells + within_cell
        fraction = completed_equivalent / total_cells if total_cells else 1.0
        eta = (elapsed * (1.0 - fraction) / fraction
               if fraction > 0.0 and task_seconds else None)
        recent = task_seconds[-20:]
        payload = {
            "complete": completed_cells >= total_cells,
            "completed_cells": completed_cells,
            "total_cells": total_cells,
            "cell_fraction": fraction,
            "completed_tasks": completed_tasks,
            "current": {
                "scenario": scenario, "k": k, "window_seconds": window_seconds,
                "task_done": task_done, "task_total": task_total,
            },
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "tasks_per_minute": (60.0 * completed_tasks / elapsed) if elapsed else 0.0,
            "recent_task_seconds_mean": (sum(recent) / len(recent)) if recent else None,
            "last_task_seconds": last_task_seconds,
            "build_seconds_by_scenario": build_seconds,
            "model_seconds": model_seconds,
        }
        _atomic_json(args.out / "progress.json", payload)
        eta_text = "unknown" if eta is None else f"{eta / 60.0:.1f}m"
        print(
            f"[progress] cells={completed_cells}/{total_cells} "
            f"cell_tasks={task_done}/{task_total} tasks={completed_tasks} "
            f"elapsed={elapsed / 60.0:.1f}m eta={eta_text}",
            flush=True,
        )

    _atomic_json(args.out / "run_provenance.json", _run_provenance(
        list(os.sys.argv), device=device, halo_checkpoint=args.halo_checkpoint,
    ))
    _atomic_json(args.out / "feature_cache_provenance.json", {
        "schema": FEATURE_CACHE_SCHEMA,
        "write_directory": str(cache_dir.resolve()),
        "prior_cache_reuse": not args.no_prior_feature_cache_reuse,
        "read_directories": [str(path) for path in cache_read_dirs],
        "memory_cache_gib": float(args.feature_memory_cache_gib),
    })
    for scenario in args.scenarios:
        for window_seconds in windows:
            for k in ks:
                build_started = time.perf_counter()
                try:
                    tasks = build_tasks(scenario, k, window_seconds,
                                        seed=args.seed, coverage=args.coverage, limit=limit,
                                        failures=failures,
                                        include_prospective=args.include_prospective_mobiact)
                except Exception as exc:  # noqa: BLE001
                    failures.append({"scenario": scenario, "k": k,
                                     "window_seconds": window_seconds,
                                     "stage": "build_tasks",
                                     "error": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc()})
                    _atomic_json(args.out / "failures.json", failures)
                    completed_cells += 1
                    checkpoint_progress(scenario=scenario, k=k,
                                        window_seconds=window_seconds)
                    continue
                build_key = f"{scenario}|k={k}|w={window_seconds:g}"
                build_seconds[build_key] = time.perf_counter() - build_started
                if not tasks:
                    requires_enrollment = scenario in {
                        "s1_partial_coverage", "s4_missing_modality", "s5_rate_mismatch",
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
                selected_tasks = tasks
                paired_tracker = _PairedDeltaTracker(selected_tasks, bootstrap=bootstrap)
                for task_index, task in enumerate(selected_tasks, start=1):
                    task_manifest = manifest_fingerprint(task.plans)
                    _append_jsonl(
                        manifest_path,
                        [_task_artifact(
                            task, k=k, window_seconds=window_seconds,
                            compact=not args.full_audit, manifest=task_manifest,
                        )],
                    )
                    print(f"[scenarios] {scenario} k={k} w={window_seconds:g} {task.variant}",
                          flush=True)
                    task_predictions: list[dict] = []
                    task_started = time.perf_counter()
                    try:
                        rows.extend(score_task(
                            task, models=args.models, device=device, cache_dir=cache_dir,
                            halo_checkpoint=args.halo_checkpoint, bootstrap=bootstrap,
                            banks=banks, k=k, window_seconds=window_seconds,
                            halo_state=halo_state, halo_has_classifier=halo_has_classifier,
                            halo_requires_acquisition=halo_requires_acquisition,
                            halo_architecture=halo_architecture,
                            selected_readouts=(None if args.readouts is None
                                               else frozenset(args.readouts)),
                            include_oracle_diagnostics=args.include_oracle_diagnostics,
                            provider_states=provider_states,
                            prediction_sink=task_predictions,
                            cache_read_dirs=cache_read_dirs,
                            feature_memory_cache=feature_memory_cache,
                            manifest=task_manifest,
                            timing_sink=model_seconds))
                    except Exception as exc:  # noqa: BLE001
                        failures.append({"scenario": scenario, "variant": task.variant, "k": k,
                                         "window_seconds": window_seconds, "stage": "score_task",
                                         "error": f"{type(exc).__name__}: {exc}",
                                         "traceback": traceback.format_exc()})
                    completed_tasks += 1
                    last_task_seconds = time.perf_counter() - task_started
                    task_seconds.append(last_task_seconds)
                    if args.full_audit:
                        _append_jsonl(prediction_path, task_predictions)
                    try:
                        paired_tracker.consume(task_predictions)
                    except RuntimeError as exc:
                        failures.append({"scenario": scenario, "variant": task.variant,
                                         "k": k, "window_seconds": window_seconds,
                                         "stage": "paired_deltas", "error": str(exc)})
                    _atomic_json(args.out / "failures.json", failures)
                    # Manifests and predictions are append-only per task. Rewriting the growing
                    # JSON/CSV/Markdown result set after every cell is quadratic I/O and dominated
                    # cached evaluation runs; checkpoint it periodically and always at completion.
                    if completed_tasks % 25 == 0:
                        _atomic_json(args.out / "results.json", rows)
                        _write_tabular_results(args.out, rows)
                    checkpoint_progress(
                        scenario=scenario, k=k, window_seconds=window_seconds,
                        task_done=task_index, task_total=len(selected_tasks),
                        last_task_seconds=last_task_seconds,
                    )
                try:
                    paired_delta_rows.extend(paired_tracker.finish())
                except RuntimeError as exc:
                    # Keep every completed matched pair. One unavailable provider/cell must not
                    # discard valid deltas already computed for the rest of this scenario.
                    paired_delta_rows.extend(paired_tracker.output)
                    failures.append({"scenario": scenario, "k": k,
                                     "window_seconds": window_seconds,
                                     "stage": "paired_deltas",
                                     "error": str(exc)})
                    _atomic_json(args.out / "failures.json", failures)
                _atomic_json(args.out / "paired_deltas.json", paired_delta_rows)
                completed_cells += 1
                checkpoint_progress(scenario=scenario, k=k,
                                    window_seconds=window_seconds)

    _atomic_json(args.out / "results.json", rows)
    _atomic_json(args.out / "failures.json", failures)
    _atomic_json(args.out / "paired_deltas.json", paired_delta_rows)
    _atomic_json(args.out / "model_artifacts.json", _model_artifacts(
        args.models, provider_states, args.halo_checkpoint,
    ))
    _write_tabular_results(args.out, rows)
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    complete = not failures and not failed_rows
    _atomic_json(args.out / "run_metadata.json", {
        "schema": SCENARIO_RESULT_SCHEMA,
        "complete": complete,
        "status": "complete" if complete else "failed",
        "n_rows": len(rows),
        "n_task_failures": len(failures),
        "n_failed_rows": len(failed_rows),
        "models": list(args.models),
        "scenarios": list(args.scenarios),
        "k": ks,
        "window_seconds": windows,
        "include_prospective_mobiact": bool(args.include_prospective_mobiact),
    })
    checkpoint_progress(scenario=None, k=None, window_seconds=None)
    print(f"wrote {len(rows)} rows and {len(failures)} task failures to {args.out}")
    if not complete:
        raise SystemExit(
            f"scenario evaluation incomplete: {len(failures)} task failures and "
            f"{len(failed_rows)} failed result rows; inspect {args.out}"
        )


if __name__ == "__main__":
    main()
