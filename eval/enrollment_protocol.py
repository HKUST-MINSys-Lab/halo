"""Versioned, model-independent support/query manifests for adaptation evaluation.

The manifest is the experimental unit shared by HALO and all external baselines.  It stores row
indices into immutable evaluation grids, but binds those indices to a content fingerprint so a
rebuilt or re-ordered grid cannot silently change an experiment.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from data.scripts.curate import deployment_policy
from eval import data as eval_data
from eval.data import EvalStream, load_eval_stream
from eval.scoring import align_ground_truth_labels
from training.evidence.episode_labels import neutral_alias_vocabulary
from training.evidence.policy import PHASE_B_TEST_DATASETS


SCHEMA_VERSION = 2
PROTOCOL_NAME = "halo_matched_adaptation_v2"
DEFAULT_SUPPORT = (0, 1, 2, 4, 8, 16)
DEFAULT_SEEDS = (20260808, 20260809, 20260810, 20260811, 20260812)
ACTION_REGIMES = {
    "ordinary": (
        "inclusivehar", "usc_had", "tnda_har", "ut_complex",
        # Restored 2026-09-04. These three were never held out for being unsuitable — they were
        # the Phase-B DEVELOPMENT split, kept aside so repeated development would not consume the
        # datasets meant to support the final claim. That split no longer exists (two data roles
        # only), so the reason to withhold them is gone.
        #
        # They also correct a real imbalance: without them 8 of 11 evaluation streams are wrist,
        # and the hardest labels sit on exactly the streams with the narrowest support pools, so
        # difficulty and support-availability were confounded. All three are phone streams with
        # ordinary locomotion, which decorrelates the two.
        #
        # DISCLOSE IN THE PAPER: design decisions in the superseded Phase-B line were made while
        # looking at these three. No model was ever trained on them, so this is researcher
        # exposure rather than data leakage, but a reviewer should hear it from us.
        "motionsense", "realworld", "shoaib",
    ),
    "specialized_novel": ("monipar", "spar", "upper_limb_use"),
}

#: The seven-dataset roster the frozen ``adaptation_v1`` manifest was built from.
ADAPTATION_V1_DATASETS = (
    "inclusivehar", "usc_had", "tnda_har", "ut_complex",
    "monipar", "spar", "upper_limb_use",
)
#: The ten-dataset roster for ``adaptation_v2``.
ADAPTATION_V2_DATASETS = ADAPTATION_V1_DATASETS + ("motionsense", "realworld", "shoaib")

#: hapt is PERMANENTLY excluded: it is the same 30 subjects as uci_har, which is in the training
#: corpus, so any hapt row would be a direct subject leak. It currently has no eval_labels.json,
#: which blocks it by accident rather than by design — hence this note.
NEVER_EVALUATE = ("hapt",)


def _json_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _stream_arrays(stream: EvalStream) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stream.execution_ids is None:
        raise ValueError(f"{stream.dataset}/{stream.stream}: execution ids are required")
    labels = np.asarray(
        align_ground_truth_labels(stream.gt, stream.eval_labels), dtype=object
    )
    return (
        labels,
        np.asarray(stream.subjects, dtype=object),
        np.asarray(stream.execution_ids, dtype=object),
    )


def stream_fingerprint(stream: EvalStream) -> str:
    """Hash the screened signal tensor, row identity, and protocol metadata."""
    labels, subjects, executions = _stream_arrays(stream)
    signal = np.ascontiguousarray(stream.windows)
    return _json_hash({
        "dataset": stream.dataset,
        "stream": stream.stream,
        "alignment": stream.alignment,
        "n_windows": stream.n_windows,
        "signal_sha256": hashlib.sha256(signal.view(np.uint8)).hexdigest(),
        "labels": labels.tolist(),
        "subjects": subjects.tolist(),
        "executions": executions.tolist(),
        "eval_labels": list(stream.eval_labels),
        "channels": list(stream.channels),
        "mask": np.asarray(stream.mask, dtype=bool).tolist(),
        "rate_hz": float(stream.rate_hz),
        "quality_screen": stream.quality_screen,
        "quality_excluded": int(stream.n_quality_excluded),
    })


def _relation_seed(seed: int, relation_id: str) -> int:
    offset = int(hashlib.sha256(relation_id.encode()).hexdigest()[:8], 16)
    return int((int(seed) + offset) % (2**32))


def _rows_per_execution(
    rows: np.ndarray,
    execution_ids: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[str]]:
    executions = np.unique(execution_ids[rows])
    if len(executions) < count:
        raise ValueError("insufficient independent executions")
    executions = executions[rng.permutation(len(executions))][:count]
    selected = [
        np.sort(rows[execution_ids[rows] == execution]).astype(int).tolist()
        for execution in executions
    ]
    return selected, [str(value) for value in executions]


def _build_subject_plan(
    *,
    subject: object,
    candidate_names: Sequence[str],
    query_labels: np.ndarray,
    query_subjects: np.ndarray,
    query_executions: np.ndarray,
    support_labels: np.ndarray,
    support_subjects: np.ndarray,
    support_executions: np.ndarray,
    support_count: int,
    subject_relation: str,
    rng: np.random.Generator,
) -> dict | None:
    support_rows: list[list[int]] = []
    support_execution_names: list[list[str]] = []
    excluded_query_executions: set[str] = set()

    for label in candidate_names:
        support_mask = support_labels == label
        if subject_relation == "same_subject":
            support_mask &= support_subjects == subject
        elif subject_relation == "cross_subject":
            support_mask &= support_subjects != subject
        elif subject_relation == "unattributed":
            # Some released evaluation bundles preserve independent executions but not participant
            # identity. They can support enrollment, but no same/cross-subject claim.
            pass
        else:
            raise ValueError(f"unknown subject relation {subject_relation!r}")
        rows = np.flatnonzero(support_mask)
        if len(np.unique(support_executions[rows])) < support_count:
            return None
        selected, selected_executions = _rows_per_execution(
            rows, support_executions, support_count, rng
        )
        support_rows.append(selected)
        support_execution_names.append(selected_executions)
        if subject_relation in {"same_subject", "unattributed"}:
            excluded_query_executions.update(selected_executions)

    query_mask = (query_subjects == subject) & np.isin(query_labels, candidate_names)
    if excluded_query_executions:
        query_mask &= ~np.isin(
            query_executions.astype(str), np.asarray(sorted(excluded_query_executions))
        )
    query_rows = np.flatnonzero(query_mask)
    # Subjects need not perform every action in the dataset protocol. The candidate roster remains
    # fixed, while a subject contributes only its actual held-out executions. Requiring two query
    # classes avoids meaningless one-class subject-level F1 cells.
    if len(np.unique(query_labels[query_rows])) < 2:
        return None
    query_execution_names = sorted({str(value) for value in query_executions[query_rows]})
    if set(query_execution_names) & {
        value for values in support_execution_names for value in values
    }:
        raise AssertionError("support/query execution leakage while building manifest")
    return {
        "subject": str(subject),
        "candidate_names": list(candidate_names),
        "support_execution_rows": support_rows,
        "support_execution_ids": support_execution_names,
        "query_rows": query_rows.astype(int).tolist(),
        "query_execution_ids": query_execution_names,
    }


def _aliases(candidate_names: Sequence[str], seed: int) -> dict[str, str]:
    vocabulary = neutral_alias_vocabulary()
    if len(candidate_names) > len(vocabulary):
        raise ValueError("candidate roster is larger than neutral alias vocabulary")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(vocabulary), size=len(candidate_names), replace=False)
    return {
        str(label): vocabulary[int(index)]
        for label, index in zip(candidate_names, selected, strict=True)
    }


def _zero_shot_cell(stream: EvalStream, *, regime: str) -> dict:
    labels, subjects, executions = _stream_arrays(stream)
    observed = set(labels.tolist())
    candidates = [label for label in stream.eval_labels if label in observed]
    keep = np.isin(labels, candidates)
    plans = []
    for subject in np.unique(subjects[keep]):
        rows = np.flatnonzero(keep & (subjects == subject))
        plans.append({
            "subject": str(subject),
            "candidate_names": candidates,
            "support_execution_rows": [[] for _ in candidates],
            "support_execution_ids": [[] for _ in candidates],
            "query_rows": rows.astype(int).tolist(),
            "query_execution_ids": sorted({str(value) for value in executions[rows]}),
        })
    return {
        "kind": "zero_shot",
        "regime": regime,
        "dataset": stream.dataset,
        "query_stream": stream.stream,
        "support_stream": None,
        "configuration_relation": "none",
        "subject_relation": "none",
        "candidate_names": candidates,
        "support_ceiling": 0,
        "status": "ok" if plans else "no_eligible_queries",
        "seeds": {"0": {"plans": plans, "aliases": None}},
    }


def _positive_cell(
    query: EvalStream,
    support: EvalStream,
    *,
    regime: str,
    subject_relation: str,
    configuration_relation: str,
    support_counts: Sequence[int],
    seeds: Sequence[int],
) -> dict:
    q_labels, q_subjects, q_executions = _stream_arrays(query)
    s_labels, s_subjects, s_executions = _stream_arrays(support)
    observed_query = set(q_labels.tolist())
    observed_support = set(s_labels.tolist())
    support_vocab = set(support.eval_labels)
    # This stream-level roster is frozen before support sampling. It removes labels that the
    # acquisition stream never recorded (for example Upper Limb Use's declared but empty
    # opening-door class), but never changes with subject, seed, or k.
    candidates = [
        label for label in query.eval_labels
        if label in support_vocab and label in observed_query and label in observed_support
    ]
    requested = sorted({int(value) for value in support_counts if int(value) > 0})
    main_requested = [value for value in requested if value <= 8]
    secondary_requested = [value for value in requested if value > 8]
    relation_id = (
        f"{query.dataset}/{query.stream}/from_{support.stream}/"
        f"{configuration_relation}/{subject_relation}"
    )
    def plans_at(support_count: int, seed: int) -> list[dict]:
        rng = np.random.default_rng(_relation_seed(seed, relation_id))
        plans = []
        for subject in np.unique(q_subjects):
            plan = _build_subject_plan(
                subject=subject,
                candidate_names=candidates,
                query_labels=q_labels,
                query_subjects=q_subjects,
                query_executions=q_executions,
                support_labels=s_labels,
                support_subjects=s_subjects,
                support_executions=s_executions,
                support_count=support_count,
                subject_relation=subject_relation,
                rng=rng,
            )
            if plan is not None:
                plans.append(plan)
        return plans

    # Preserve the entire candidate roster and lower only the declared support ceiling. This keeps
    # k=1..ceiling a matched curve while making larger unsupported k values explicit N/A cells.
    ceiling = 0
    for candidate_ceiling in reversed(main_requested):
        if all(plans_at(candidate_ceiling, seed) for seed in seeds):
            ceiling = candidate_ceiling
            break
    seed_payload = {
        str(seed): {
            "plans": plans_at(ceiling, seed) if ceiling else [],
            "aliases": _aliases(candidates, _relation_seed(seed, relation_id + "/aliases")),
        }
        for seed in seeds
    }
    secondary_ceiling = 0
    for candidate_ceiling in reversed(secondary_requested):
        if all(plans_at(candidate_ceiling, seed) for seed in seeds):
            secondary_ceiling = candidate_ceiling
            break
    secondary_payload = {
        str(seed): {
            "plans": plans_at(secondary_ceiling, seed) if secondary_ceiling else [],
            "aliases": _aliases(candidates, _relation_seed(seed, relation_id + "/aliases")),
        }
        for seed in seeds
    }
    status = "ok" if candidates and ceiling > 0 else (
        "insufficient_independent_executions" if len(candidates) >= 2 else "insufficient_candidates"
    )
    return {
        "kind": "enrollment",
        "regime": regime,
        "dataset": query.dataset,
        "query_stream": query.stream,
        "support_stream": support.stream,
        "configuration_relation": configuration_relation,
        "subject_relation": subject_relation,
        "candidate_names": candidates,
        "support_ceiling": ceiling if status == "ok" else 0,
        "status": status,
        "seeds": seed_payload,
        "secondary_high_support": {
            "status": "ok" if secondary_ceiling else "insufficient_independent_executions",
            "support_ceiling": secondary_ceiling,
            "support_counts": secondary_requested,
            "cohort_policy": "separate_secondary_cohort_not_pooled_with_main_k_curve",
            "seeds": secondary_payload,
        },
    }


def build_manifest(
    *,
    datasets: Sequence[str] = PHASE_B_TEST_DATASETS,
    support_counts: Sequence[int] = DEFAULT_SUPPORT,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    alignment: str = "native",
    subject_relations: Sequence[str] = ("same_subject", "cross_subject"),
    configuration_relations: Sequence[str] = ("same_configuration", "cross_configuration"),
    protocol_name: str = PROTOCOL_NAME,
) -> dict:
    """Build the complete matched evaluation manifest from current sealed grids."""
    datasets = tuple(datasets)
    banned = sorted(set(datasets) & set(NEVER_EVALUATE))
    if banned:
        raise ValueError(
            f"{banned} may never be evaluated: they share subjects with the training corpus"
        )
    unknown = sorted(set(datasets) - set(sum((list(v) for v in ACTION_REGIMES.values()), [])))
    if unknown:
        raise ValueError(f"datasets have no declared action regime: {unknown}")
    support_counts = tuple(sorted({int(value) for value in support_counts}))
    if 0 not in support_counts or any(value < 0 for value in support_counts):
        raise ValueError("support counts must include zero and contain no negative values")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("manifest seeds must be non-empty and unique")

    cells: dict[str, dict] = {}
    fingerprints: dict[str, str] = {}
    stream_cache: dict[tuple[str, str], EvalStream] = {}

    def load(dataset: str, stream_id: str) -> EvalStream:
        key = (dataset, stream_id)
        if key not in stream_cache:
            stream_cache[key] = load_eval_stream(dataset, stream_id, alignment=alignment)
            fingerprints[f"{dataset}/{stream_id}"] = stream_fingerprint(stream_cache[key])
        return stream_cache[key]

    def relation_options(query: EvalStream, support: EvalStream) -> tuple[str, ...]:
        """Return honest subject relations for a stream pair.

        A single sentinel identity such as TNDA-HAR's ``unknown`` cannot establish either
        same-subject or cross-subject enrollment. Preserve the usable execution protocol under an
        explicit unattributed relation instead of manufacturing a subject claim.
        """

        sentinels = {"", "unknown", "none", "nan"}
        query_subjects = {str(value).strip().lower() for value in np.asarray(query.subjects)}
        support_subjects = {str(value).strip().lower() for value in np.asarray(support.subjects)}
        if not (query_subjects - sentinels) or not (support_subjects - sentinels):
            return ("unattributed",)
        return tuple(subject_relations)

    regime_for = {
        dataset: regime for regime, values in ACTION_REGIMES.items() for dataset in values
    }
    skipped: list[str] = []

    def materialized(specs):
        """Keep only streams whose grid exists for this alignment.

        A curated StreamSpec describes what the source *could* provide; the grid is what was
        actually built. shoaib, for instance, declares four placements but only
        ``phone_right_pocket`` has a native grid. Building the manifest from the spec alone raises
        halfway through; skipping silently would hide a missing conversion. So skip and report.
        """
        kept = []
        for spec in specs:
            path = eval_data.DATASETS_DIR / spec.dataset / "grids" / alignment / spec.stream_id
            if path.is_dir():
                kept.append(spec)
            else:
                skipped.append(f"{spec.dataset}/{spec.stream_id}")
        return tuple(kept)

    for dataset in datasets:
        primary = materialized(deployment_policy.stream_specs(dataset, "primary"))
        deployment = materialized(tuple(
            spec for spec in deployment_policy.stream_specs(dataset, None)
            if spec.device_profile in {"phone", "watch", "device"}
        ))
        if not primary:
            raise ValueError(
                f"{dataset} has no materialized {alignment} grid for any primary stream"
            )
        for query_spec in primary:
            query = load(dataset, query_spec.stream_id)
            zero_id = f"{dataset}/{query.stream}/zero_shot"
            cells[zero_id] = _zero_shot_cell(query, regime=regime_for[dataset])
            support_specs = []
            if "same_configuration" in configuration_relations:
                support_specs.append(("same_configuration", query_spec))
            if "cross_configuration" in configuration_relations:
                support_specs.extend(
                    ("cross_configuration", spec)
                    for spec in deployment if spec.stream_id != query_spec.stream_id
                )
            for configuration_relation, support_spec in support_specs:
                support = load(dataset, support_spec.stream_id)
                for subject_relation in relation_options(query, support):
                    cell_id = (
                        f"{dataset}/{query.stream}/from_{support.stream}/"
                        f"{configuration_relation}/{subject_relation}"
                    )
                    cells[cell_id] = _positive_cell(
                        query,
                        support,
                        regime=regime_for[dataset],
                        subject_relation=subject_relation,
                        configuration_relation=configuration_relation,
                        support_counts=support_counts,
                        seeds=seeds,
                    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": protocol_name,
        "alignment": alignment,
        "datasets": list(datasets),
        "action_regimes": {key: list(value) for key, value in ACTION_REGIMES.items()},
        "support_counts": list(support_counts),
        "support_unit": "independent_execution_per_candidate",
        "seeds": [int(value) for value in seeds],
        "candidate_policy": "fixed_dataset_stream_roster_across_curve",
        "skipped_streams": sorted(set(skipped)),
        "query_policy": "fixed_subject_cohort_within_each_serialized_support_cohort",
        "support_policy": "nested_execution_disjoint_prefix",
        "stream_fingerprints": fingerprints,
        "cells": cells,
    }
    manifest["manifest_fingerprint"] = _json_hash(manifest)
    return manifest


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.suffix == ".gz":
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            handle.write(content)
    else:
        temporary.write_text(content)
    os.replace(temporary, path)


def load_manifest(path: Path, *, validate_grids: bool = True) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported enrollment manifest schema: {manifest.get('schema_version')}")
    expected = manifest.get("manifest_fingerprint")
    payload = dict(manifest)
    payload.pop("manifest_fingerprint", None)
    if expected != _json_hash(payload):
        raise ValueError("enrollment manifest content fingerprint is invalid")
    if validate_grids:
        streams = {}
        for key, expected_stream_fp in manifest["stream_fingerprints"].items():
            dataset, stream_id = key.split("/", 1)
            stream = load_eval_stream(dataset, stream_id, alignment=manifest["alignment"])
            streams[key] = stream
            actual = stream_fingerprint(stream)
            if actual != expected_stream_fp:
                raise ValueError(
                    f"{key}: evaluation grid changed after manifest creation "
                    f"({expected_stream_fp[:12]} != {actual[:12]})"
                )
        _validate_structure(manifest, streams)
    return manifest


def _validate_structure(manifest: dict, streams: dict[str, EvalStream]) -> None:
    expected_seeds = {str(value) for value in manifest["seeds"]}
    for cell_id, cell in manifest["cells"].items():
        candidates = list(cell["candidate_names"])
        if len(candidates) < 2 or len(set(candidates)) != len(candidates):
            raise ValueError(f"{cell_id}: candidate roster must contain unique labels")
        query = streams[f"{cell['dataset']}/{cell['query_stream']}"]
        q_labels, q_subjects, q_executions = _stream_arrays(query)
        if cell["kind"] == "zero_shot":
            payload_groups = [(0, cell["seeds"])]
            if set(cell["seeds"]) != {"0"}:
                raise ValueError(f"{cell_id}: zero-shot cell must use only seed 0")
            support = None
        else:
            support = streams[f"{cell['dataset']}/{cell['support_stream']}"]
            if set(cell["seeds"]) != expected_seeds:
                raise ValueError(f"{cell_id}: main support seeds differ from manifest seeds")
            payload_groups = [(int(cell["support_ceiling"]), cell["seeds"])]
            secondary = cell.get("secondary_high_support", {})
            if secondary:
                if set(secondary["seeds"]) != expected_seeds:
                    raise ValueError(f"{cell_id}: secondary seeds differ from manifest seeds")
                payload_groups.append((int(secondary["support_ceiling"]), secondary["seeds"]))
            if cell["configuration_relation"] == "same_configuration" and (
                cell["query_stream"] != cell["support_stream"]
            ):
                raise ValueError(f"{cell_id}: same-configuration cell uses different streams")
            if cell["configuration_relation"] == "cross_configuration" and (
                cell["query_stream"] == cell["support_stream"]
            ):
                raise ValueError(f"{cell_id}: cross-configuration cell reuses one stream")

        for support_count, seed_group in payload_groups:
            if support is not None:
                s_labels, s_subjects, s_executions = _stream_arrays(support)
            for seed, seed_payload in seed_group.items():
                aliases = seed_payload.get("aliases")
                if aliases is not None and (
                    set(aliases) != set(candidates)
                    or len(set(aliases.values())) != len(candidates)
                ):
                    raise ValueError(f"{cell_id}/seed{seed}: aliases are not one-to-one")
                for plan in seed_payload["plans"]:
                    if list(plan["candidate_names"]) != candidates:
                        raise ValueError(f"{cell_id}/seed{seed}: plan candidate roster changed")
                    query_rows = np.asarray(plan["query_rows"], dtype=np.int64)
                    if np.any((query_rows < 0) | (query_rows >= query.n_windows)):
                        raise ValueError(f"{cell_id}/seed{seed}: query row out of bounds")
                    if not np.all(q_subjects[query_rows].astype(str) == str(plan["subject"])):
                        raise ValueError(f"{cell_id}/seed{seed}: query subject mismatch")
                    if not set(q_labels[query_rows]).issubset(candidates):
                        raise ValueError(f"{cell_id}/seed{seed}: query label outside candidate roster")
                    actual_query_executions = {str(value) for value in q_executions[query_rows]}
                    if actual_query_executions != set(plan["query_execution_ids"]):
                        raise ValueError(f"{cell_id}/seed{seed}: query execution ids mismatch rows")
                    if support is None:
                        continue
                    if len(plan["support_execution_rows"]) != len(candidates):
                        raise ValueError(f"{cell_id}/seed{seed}: support roster width mismatch")
                    support_execution_set = set()
                    for position, (execution_rows, executions) in enumerate(zip(
                        plan["support_execution_rows"], plan["support_execution_ids"], strict=True
                    )):
                        if len(execution_rows) != support_count or len(executions) != support_count:
                            raise ValueError(
                                f"{cell_id}/seed{seed}: support count does not match ceiling"
                            )
                        for rows, execution in zip(execution_rows, executions, strict=True):
                            rows = np.asarray(rows, dtype=np.int64)
                            if not len(rows) or np.any((rows < 0) | (rows >= support.n_windows)):
                                raise ValueError(f"{cell_id}/seed{seed}: support row out of bounds")
                            if not np.all(s_labels[rows] == candidates[position]):
                                raise ValueError(f"{cell_id}/seed{seed}: support label mismatch")
                            actual = {str(value) for value in s_executions[rows]}
                            if actual != {str(execution)}:
                                raise ValueError(f"{cell_id}/seed{seed}: support execution mismatch")
                            if cell["subject_relation"] == "same_subject" and not np.all(
                                s_subjects[rows].astype(str) == str(plan["subject"])
                            ):
                                raise ValueError(f"{cell_id}/seed{seed}: same-subject support mismatch")
                            if cell["subject_relation"] == "cross_subject" and np.any(
                                s_subjects[rows].astype(str) == str(plan["subject"])
                            ):
                                raise ValueError(f"{cell_id}/seed{seed}: cross-subject support mismatch")
                            support_execution_set.add(str(execution))
                    if support_execution_set & actual_query_executions:
                        raise ValueError(f"{cell_id}/seed{seed}: support/query execution leakage")


def iter_cells(
    manifest: dict,
    *,
    kinds: Iterable[str] | None = None,
    regimes: Iterable[str] | None = None,
) -> Iterable[tuple[str, dict]]:
    kinds = None if kinds is None else set(kinds)
    regimes = None if regimes is None else set(regimes)
    for cell_id, cell in manifest["cells"].items():
        if kinds is not None and cell["kind"] not in kinds:
            continue
        if regimes is not None and cell["regime"] not in regimes:
            continue
        yield cell_id, cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=list(ADAPTATION_V2_DATASETS))
    parser.add_argument("--protocol-name", default="halo_matched_adaptation_v2",
                        help="stamped into every cell; the assembler refuses to mix protocols, "
                             "which is what stops v1 and v2 rows landing in one table")
    parser.add_argument("--support", nargs="*", type=int, default=list(DEFAULT_SUPPORT))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--alignment", choices=("native", "non_harmonised", "harmonised"),
                        default="native")
    args = parser.parse_args()
    manifest = build_manifest(
        datasets=args.datasets,
        support_counts=args.support,
        seeds=args.seeds,
        alignment=args.alignment,
        protocol_name=args.protocol_name,
    )
    save_manifest(args.out, manifest)
    statuses = {}
    for cell in manifest["cells"].values():
        statuses[cell["status"]] = statuses.get(cell["status"], 0) + 1
    print(f"wrote {args.out}")
    print(f"fingerprint={manifest['manifest_fingerprint']}")
    print(f"cells={len(manifest['cells'])} statuses={statuses}")


if __name__ == "__main__":
    main()
