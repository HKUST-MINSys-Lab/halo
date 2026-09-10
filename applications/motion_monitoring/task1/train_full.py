"""Fit the Task-1 metric head and fix its operating point (spec section F).

Two data roles only. The head is fitted on the synthetic train manifest minus
a deterministic hold-out of background subjects; the detection threshold is
the largest score whose false-alarm rate on that hold-out stays within the
a-priori budget in ``OPERATING_POINT_PROTOCOL``. Nothing is tuned on natural
data. The untrained direct floor receives the same treatment so both arms
reach evaluation with a comparable operating point.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.batching import (
    central_duration_cap,
    nearest_length_indices,
    padding_fraction,
    plan_duration,
)
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation import fingerprint_protocol
from applications.motion_monitoring.evaluation_manifests import (
    Task1EvaluationUnit,
    read_task_manifest,
    validate_task_manifest,
)
from applications.motion_monitoring.representation_cache import open_representations
from applications.motion_monitoring.representation_cache import bounded_representation_id
from applications.motion_monitoring.task1.episodes import (
    EmbeddingSequence,
    collate_detection_episodes,
    episode_from_recordings,
    from_motion_sequence,
)
from applications.motion_monitoring.task1.full_evaluation import unit_matches
from applications.motion_monitoring.task1.full_evaluation import _reference_bucket
from applications.motion_monitoring.task1.model import DifferentiableSubsequenceMatcher
from applications.motion_monitoring.task1.training import train_step


def _crop_sequence(sequence: EmbeddingSequence, start: float, end: float) -> EmbeddingSequence:
    centers = sequence.intervals_sec.mean(dim=1)
    selected = torch.nonzero(
        (centers >= start) & (centers < end), as_tuple=False
    ).flatten()
    if not len(selected):
        raise ValueError("training crop contains no representation patches")
    left, right = int(selected[0]), int(selected[-1]) + 1
    return EmbeddingSequence(
        sequence.embeddings[left:right],
        sequence.intervals_sec[left:right],
        sequence.valid[left:right],
        metadata=sequence.metadata,
    )


def _training_episode(unit, recordings, representations, *, seconds: float | None, rng):
    reference_recording = recordings[unit.reference_cache_index]
    query_recording = recordings[unit.query_cache_index]
    reference = from_motion_sequence(
        representations.get(
            unit.dataset,
            bounded_representation_id(
                unit.reference_recording_id, unit.reference_event_index
            ),
            unit.reference_stream_id,
        )
    )
    query = from_motion_sequence(
        representations.get(unit.dataset, unit.query_recording_id, unit.query_stream_id)
    )
    # A manifest block is a declared query unit, not a training convenience crop.
    if getattr(unit, "query_interval_sec", None) is not None:
        query = _crop_sequence(query, *map(float, unit.query_interval_sec))
    timeline_start = float(query.intervals_sec[0, 0])
    timeline_end = float(query.intervals_sec[-1, 1])
    width = timeline_end - timeline_start if seconds is None else min(seconds, timeline_end - timeline_start)
    if width < timeline_end - timeline_start - 1e-9 and unit.target_intervals_sec:
        target_start, target_end = rng.choice(unit.target_intervals_sec)
        low = max(timeline_start, target_end - width)
        high = min(target_start, timeline_end - width)
        candidates = [low if high <= low else rng.uniform(low, high) for _ in range(64)]
        crop_start = next(
            (
                start
                for start in candidates
                if all(
                    end <= start + 1e-6
                    or begin >= start + width - 1e-6
                    or (begin >= start - 1e-6 and end <= start + width + 1e-6)
                    for begin, end in unit.target_intervals_sec
                )
            ),
            None,
        )
        if crop_start is None:
            raise ValueError("no training crop preserves every intersected target")
    elif width < timeline_end - timeline_start - 1e-9:
        high = max(timeline_start, timeline_end - width)
        crop_start = timeline_start if high <= timeline_start else rng.uniform(timeline_start, high)
    else:
        crop_start = timeline_start
    query = _crop_sequence(query, crop_start, crop_start + width)
    episode = episode_from_recordings(
        reference_recording,
        query_recording,
        reference,
        query,
        label=unit.label,
        reference_event_index=unit.reference_event_index,
        target_intervals_sec=unit.target_intervals_sec,
        reference_interval_sec=getattr(unit, "reference_interval_sec", None),
        # Training re-draws the grid-snap context side per sample; evaluation
        # paths leave this None for the deterministic draw.
        reference_rng=rng,
        guard_intervals_sec=unit.guard_intervals_sec,
    )
    return replace(
        episode,
        metadata={
            **episode.metadata,
            "query_view_mode": (
                "complete" if width >= timeline_end - timeline_start - 1e-9 else "cropped"
            ),
            "query_source_duration_sec": timeline_end - timeline_start,
            "query_retained_duration_sec": width,
        },
    )


def _query_duration(unit, representations) -> float:
    """Declared query duration before any stochastic training crop."""

    sequence = from_motion_sequence(
        representations.get(unit.dataset, unit.query_recording_id, unit.query_stream_id)
    )
    if getattr(unit, "query_interval_sec", None) is not None:
        sequence = _crop_sequence(sequence, *map(float, unit.query_interval_sec))
    return float(sequence.intervals_sec[-1, 1] - sequence.intervals_sec[0, 0])


def _draw_logical_units(groups, datasets_by_status, *, batch_size: int, rng):
    """Draw relationships first; duration planning cannot alter this distribution."""

    units = []
    for index in range(batch_size):
        status = bool(index % 2)
        dataset = rng.choice(datasets_by_status[status])
        units.append(rng.choice(groups[(status, dataset)]))
    return units


def _draw_length_matched_units(
    groups,
    datasets_by_status,
    *,
    batch_size: int,
    representations,
    rng,
    candidate_multiplier: int = 4,
):
    """Choose a balanced logical batch from a small, random length candidate pool."""

    if candidate_multiplier < 1:
        raise ValueError("candidate multiplier must be positive")
    pool = _draw_logical_units(
        groups, datasets_by_status, batch_size=batch_size * candidate_multiplier, rng=rng
    )
    durations = [_query_duration(unit, representations) for unit in pool]
    target = central_duration_cap(durations, rng)
    selected = []
    for status in (False, True):
        candidates = [index for index, unit in enumerate(pool) if unit.target_present == status]
        required = sum(bool(index % 2) == status for index in range(batch_size))
        if required == 0:
            continue
        local = nearest_length_indices(
            [durations[index] for index in candidates],
            count=required,
            target_seconds=target,
            rng=rng,
        )
        selected.extend(candidates[index] for index in local)
    rng.shuffle(selected)
    return [pool[index] for index in selected]


def _minimum_target_duration(units) -> float:
    """Smallest shared query cap that can retain every selected complete target."""

    durations = [
        float(end - start)
        for unit in units
        for start, end in unit.target_intervals_sec
    ]
    return max(durations, default=0.0)


def _pooled_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    tp = sum(row["true_positive_count"] for row in rows)
    fp = sum(row["false_positive_count"] for row in rows)
    fn = sum(row["false_negative_count"] for row in rows)
    hours = sum(row["query_hours"] for row in rows)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_alarms_per_hour": fp / max(hours, 1e-12),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
    }


def _event_prefix(matches, targets_sec, *, iou_threshold: float):
    """Return exact greedy TP counts for every score-ranked detection prefix."""

    ranked = sorted(matches, key=lambda item: item.score)
    targets = torch.as_tensor(targets_sec, dtype=torch.float64)
    unmatched = set(range(len(targets)))
    true_positive = 0
    cumulative = []
    for match in ranked:
        best_target = None
        best_iou = 0.0
        for target_index in unmatched:
            target_start, target_end = targets[target_index].tolist()
            intersection = max(
                0.0,
                min(match.end_sec, target_end) - max(match.start_sec, target_start),
            )
            union = max(match.end_sec, target_end) - min(match.start_sec, target_start)
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_target = target_index
        if best_target is not None and best_iou >= iou_threshold:
            unmatched.remove(best_target)
            true_positive += 1
        cumulative.append(true_positive)
    return (
        np.asarray([match.score for match in ranked], dtype=np.float64),
        np.asarray(cumulative, dtype=np.int64),
        len(targets),
    )


def _digest(*parts: object) -> str:
    return sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def split_by_subject(
    units: Sequence[Task1EvaluationUnit], *, seed: int, heldout_fraction: float
) -> tuple[list[Task1EvaluationUnit], list[Task1EvaluationUnit]]:
    """Deterministically hold out a fraction of query subjects (background wearers)."""

    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must be in (0, 1)")
    subjects = sorted({unit.query_subject_id for unit in units})
    if len(subjects) < 2:
        raise ValueError("a subject hold-out needs at least two query subjects")
    ranked = sorted(subjects, key=lambda subject: _digest(seed, "heldout", subject))
    heldout = set(ranked[: max(1, min(len(ranked) - 1, round(len(ranked) * heldout_fraction)))])
    fit = [unit for unit in units if unit.query_subject_id not in heldout]
    held = [unit for unit in units if unit.query_subject_id in heldout]
    return fit, held


# Fixed a priori (TASK1_REFERENCE_RESOLUTION_SPEC.md section F.3). Changing any
# value is a protocol revision. The budget is one false alarm every three
# minutes of monitoring; two synthetic background subjects are held out from
# head fitting, so no natural data is consumed and one individual cannot set
# the deployed threshold alone.
OPERATING_POINT_PROTOCOL: dict[str, Any] = {
    "schema_version": 1,
    "false_alarm_budget_per_hour": 20.0,
    "holdout_fraction": 0.3,
    "holdout_unit": "synthetic query background subject",
    "rule": (
        "largest score threshold whose false alarms per hour on the held-out "
        "synthetic subjects do not exceed the budget"
    ),
    "nms_iou": 0.3,
    "match_iou": 0.5,
}

# This records how the frozen-representation trainer formed query views.  It is
# saved with the head because a crop of cached contextual features is not the
# same experimental condition as cropping raw signal before an end-to-end pass.
TRAINING_VIEW_POLICY: dict[str, Any] = {
    "schema_version": 1,
    "query_views": "complete-or-central-duration-crop",
    "complete_batch_length_matching": "bounded_candidate_pool",
    "reference_views": "complete_bounded_execution_only",
    "positive_crop_rule": "retain_an_entire_target_or_reject",
}


def detection_curve(
    units: Sequence[Task1EvaluationUnit],
    recording_caches,
    representations,
    model,
    *,
    nms_iou: float = 0.3,
    match_iou: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score every unit once at an open threshold and keep its ranked prefix."""

    evaluated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        try:
            matches, episode = unit_matches(
                unit,
                recording_caches[unit.dataset],
                representations,
                score_threshold=float("inf"),
                model=model,
                nms_iou=nms_iou,
            )
        except ValueError as error:
            rejected.append({"unit": index, "label": unit.label, "reason": str(error)})
            continue
        scores, cumulative, target_count = _event_prefix(
            matches, episode.targets_sec, iou_threshold=match_iou
        )
        duration_hours = float(
            episode.query.intervals_sec[-1, 1] - episode.query.intervals_sec[0, 0]
        ) / 3600.0
        evaluated.append(
            {
                "scores": scores,
                "cumulative_tp": cumulative,
                "target_count": int(target_count),
                "query_hours": duration_hours,
                "dataset": unit.dataset,
                "relation": unit.reference_relation,
                "query_subject_id": unit.query_subject_id,
            }
        )
    return evaluated, rejected


def _rows_for_threshold(evaluated: Sequence[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    rows = []
    for item in evaluated:
        scores = item["scores"]
        detection_count = int(np.searchsorted(scores, threshold, side="right"))
        true_positive = int(item["cumulative_tp"][detection_count - 1]) if detection_count else 0
        rows.append(
            {
                "true_positive_count": float(true_positive),
                "false_positive_count": float(detection_count - true_positive),
                "false_negative_count": float(item["target_count"] - true_positive),
                "query_hours": item["query_hours"],
                "dataset": item["dataset"],
                "relation": item["relation"],
            }
        )
    return rows


def average_precision(evaluated: Sequence[dict[str, Any]]) -> float:
    """Threshold-free event AP over the pooled, score-ranked detections."""

    scores = []
    flags = []
    total_targets = 0
    for item in evaluated:
        cumulative = np.asarray(item["cumulative_tp"], dtype=np.int64)
        is_tp = np.diff(np.concatenate(([0], cumulative))) > 0
        scores.append(np.asarray(item["scores"], dtype=np.float64))
        flags.append(is_tp)
        total_targets += int(item["target_count"])
    if total_targets == 0:
        return float("nan")
    if not scores:
        return 0.0
    pooled_scores = np.concatenate(scores)
    pooled_flags = np.concatenate(flags)
    if not len(pooled_scores):
        return 0.0
    order = np.argsort(pooled_scores, kind="stable")
    hits = pooled_flags[order].astype(np.float64)
    cumulative_tp = np.cumsum(hits)
    precision = cumulative_tp / np.arange(1, len(hits) + 1)
    return float(np.sum(precision * hits) / total_targets)


def fix_operating_point(
    units: Sequence[Task1EvaluationUnit],
    recording_caches,
    representations,
    model,
    *,
    false_alarm_budget_per_hour: float,
    nms_iou: float = 0.3,
    match_iou: float = 0.5,
) -> dict[str, Any]:
    """Fix the score threshold from a false-alarm budget on held-out training units.

    Scores are distances (lower is a better match), so the threshold is the
    largest value at which the pooled held-out false alarms per hour stay at or
    under the budget. Ties are resolved downwards so the budget is never
    exceeded. Metrics at that threshold are reported for telemetry only.
    """

    if false_alarm_budget_per_hour < 0:
        raise ValueError("false-alarm budget must be non-negative")
    evaluated, rejected = detection_curve(
        units, recording_caches, representations, model, nms_iou=nms_iou, match_iou=match_iou
    )
    hours = sum(item["query_hours"] for item in evaluated)
    if not evaluated or hours <= 0:
        raise ValueError("operating point needs at least one eligible held-out unit")
    all_scores = np.concatenate([item["scores"] for item in evaluated])
    if not len(all_scores):
        raise ValueError("operating point needs at least one scored match")
    false_positive_scores = []
    for item in evaluated:
        cumulative = np.asarray(item["cumulative_tp"], dtype=np.int64)
        is_tp = np.diff(np.concatenate(([0], cumulative))) > 0
        false_positive_scores.append(np.asarray(item["scores"])[~is_tp])
    false_positives = np.sort(np.concatenate(false_positive_scores))
    allowed = int(np.floor(false_alarm_budget_per_hour * hours))
    if not len(false_positives) or allowed >= len(false_positives):
        threshold = float(np.max(all_scores))
    elif allowed == 0:
        threshold = float(np.nextafter(false_positives[0], -np.inf))
    else:
        threshold = float(false_positives[allowed - 1])
        while int(np.searchsorted(false_positives, threshold, side="right")) > allowed:
            threshold = float(np.nextafter(threshold, -np.inf))
    rows = _rows_for_threshold(evaluated, threshold)
    metrics = _pooled_metrics(rows)
    per_dataset = {
        dataset: _pooled_metrics([row for row in rows if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in rows})
    }
    return {
        "threshold": threshold,
        "false_alarm_budget_per_hour": float(false_alarm_budget_per_hour),
        "holdout_query_hours": float(hours),
        "holdout_false_alarms_allowed": allowed,
        "holdout_false_alarms": int(metrics["false_positive_count"]),
        "holdout_subjects": len({item["query_subject_id"] for item in evaluated}),
        "metrics": metrics,
        "average_precision": average_precision(evaluated),
        "per_dataset": per_dataset,
        "eligible_units": len(evaluated),
        "rejected_units": len(rejected),
        "rejections": rejected,
    }


def calibrate(
    manifest,
    recording_caches,
    representations,
    model,
    *,
    nms_iou: float = 0.3,
    match_iou: float = 0.5,
) -> dict[str, Any]:
    """Oracle F1-maximising threshold search.

    Diagnostic only: used by the synthetic-corpus controls and reported as a
    labelled upper bound. Never the deployed threshold (see ``fix_operating_point``).
    """

    evaluated = []
    rejected = []
    all_scores: list[float] = []
    for index, raw in enumerate(manifest.units):
        unit = Task1EvaluationUnit(**raw)
        try:
            matches, episode = unit_matches(
                unit,
                recording_caches[unit.dataset],
                representations,
                score_threshold=float("inf"),
                model=model,
                nms_iou=nms_iou,
            )
        except ValueError as error:
            rejected.append({"unit": index, "reason": str(error)})
            continue
        duration_hours = float(
            episode.query.intervals_sec[-1, 1]
            - episode.query.intervals_sec[0, 0]
        ) / 3600.0
        evaluated.append(
            (
                *_event_prefix(matches, episode.targets_sec, iou_threshold=match_iou),
                duration_hours,
                unit.dataset,
                _reference_bucket(
                    int(
                        episode.metadata.get(
                            "reference_positions", len(episode.reference.embeddings)
                        )
                    )
                ),
            )
        )
        all_scores.extend(match.score for match in matches)
    if not evaluated or not all_scores:
        raise ValueError("development calibration has no eligible scored matches")
    def rows_at(source, threshold_for_bucket) -> list[dict[str, Any]]:
        rows = []
        for scores, cumulative_tp, target_count, duration_hours, dataset, bucket in source:
            threshold = float(threshold_for_bucket(bucket))
            detection_count = int(np.searchsorted(scores, threshold, side="right"))
            true_positive = (
                int(cumulative_tp[detection_count - 1]) if detection_count else 0
            )
            rows.append(
                {
                    "true_positive_count": float(true_positive),
                    "false_positive_count": float(detection_count - true_positive),
                    "false_negative_count": float(target_count - true_positive),
                    "query_hours": duration_hours,
                    "dataset": dataset,
                    "reference_bucket": bucket,
                }
            )
        return rows

    def optimize(source):
        unique = np.unique(
            np.concatenate([row[0] for row in source]).astype(np.float64, copy=False)
        )
        if len(unique) > 256:
            candidates = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, 256)))
        else:
            candidates = unique
        candidates = np.concatenate(([np.nextafter(candidates[0], -np.inf)], candidates))
        best = None
        for threshold in candidates:
            rows = rows_at(source, lambda _bucket, value=float(threshold): value)
            metrics = _pooled_metrics(rows)
            key = (
                metrics["event_f1"],
                -metrics["false_alarms_per_hour"],
                -float(threshold),
            )
            if best is None or key > best[0]:
                best = (key, float(threshold), metrics, len(candidates))
        assert best is not None
        return best

    global_best = optimize(evaluated)
    buckets = sorted({row[-1] for row in evaluated})
    thresholds_by_bucket: dict[str, float] = {}
    calibration_by_bucket: dict[str, dict[str, Any]] = {}
    for bucket in buckets:
        subset = [row for row in evaluated if row[-1] == bucket]
        if len(subset) < 2 or not any(row[2] for row in subset):
            thresholds_by_bucket[bucket] = global_best[1]
            calibration_by_bucket[bucket] = {
                "threshold": global_best[1],
                "units": len(subset),
                "fallback": "global_threshold_due_to_sparse_development_stratum",
            }
            continue
        best = optimize(subset)
        thresholds_by_bucket[bucket] = best[1]
        calibration_by_bucket[bucket] = {
            "threshold": best[1],
            "units": len(subset),
            "metrics": best[2],
            "candidate_thresholds": best[3],
            "fallback": None,
        }
    chosen_rows = rows_at(
        evaluated,
        lambda bucket: thresholds_by_bucket.get(bucket, global_best[1]),
    )
    chosen_metrics = _pooled_metrics(chosen_rows)
    per_dataset = {
        dataset: _pooled_metrics([row for row in chosen_rows if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in chosen_rows})
    }
    return {
        "threshold": global_best[1],
        "thresholds_by_reference_positions": thresholds_by_bucket,
        "calibration_by_reference_positions": calibration_by_bucket,
        "metrics": chosen_metrics,
        "per_dataset": per_dataset,
        "eligible_units": len(evaluated),
        "rejected_units": len(rejected),
        "rejections": rejected,
        "candidate_thresholds": global_best[3],
    }


def select_common_units(
    manifest,
    path: Path | None,
    *,
    representation_provenance: Any | None = None,
):
    if path is None:
        return manifest, None
    common = json.loads(path.read_text(encoding="utf-8"))
    if common["task_manifest_fingerprint"] != manifest.fingerprint:
        raise ValueError("common Task-1 units belong to another task manifest")
    if representation_provenance is not None and not any(
        row.get("encoder_provenance") == representation_provenance
        for row in common.get("representations", {}).values()
    ):
        raise ValueError(
            "common Task-1 units were not built for this encoder representation"
        )
    selected = [int(index) for index in common["selected_unit_indices"]]
    if not selected:
        raise ValueError("common Task-1 unit intersection is empty")
    return (
        replace(manifest, units=tuple(manifest.units[index] for index in selected)),
        common,
    )


def fit_head(
    units: Sequence[Task1EvaluationUnit],
    recording_caches,
    representations,
    *,
    feature_dim: int,
    steps: int,
    batch_size: int,
    query_seconds: float | None,
    projection_dim: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    telemetry_every: int,
    seed: int,
    device: str,
    query_complete_probability: float = 0.0,
    adaptive_query_duration: bool = False,
) -> tuple[DifferentiableSubsequenceMatcher, list[dict[str, Any]], int]:
    """Train one head on ``units``; return (model, telemetry, episode rejections)."""

    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    if query_seconds is not None and query_seconds <= 0:
        raise ValueError("query duration cap must be positive when supplied")
    if not 0 <= query_complete_probability <= 1:
        raise ValueError("query complete probability must be in [0, 1]")
    if telemetry_every <= 0:
        raise ValueError("telemetry interval must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    model = DifferentiableSubsequenceMatcher(feature_dim, projection_dim=projection_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    groups: dict[tuple[bool, str], list[Task1EvaluationUnit]] = defaultdict(list)
    for unit in units:
        groups[(unit.target_present, unit.dataset)].append(unit)
    datasets_by_status = {
        status: sorted(dataset for present, dataset in groups if present == status)
        for status in (False, True)
    }
    if any(not datasets for datasets in datasets_by_status.values()):
        raise ValueError("Task-1 training requires present and absent source groups")
    telemetry = []
    rejection_count = 0
    for step in range(steps):
        episodes = []
        duration_plan = None
        used_cap = None
        for _ in range(20):
            # Pick the view mode before length matching. Complete views use a bounded random
            # candidate pool to control padding; cropped views preserve a directly sampled mix.
            complete = rng.random() < query_complete_probability
            if complete:
                logical_units = _draw_length_matched_units(
                    groups, datasets_by_status, batch_size=batch_size,
                    representations=representations, rng=rng,
                )
                duration_plan = plan_duration(
                    [_query_duration(unit, representations) for unit in logical_units],
                    complete_probability=1.0, rng=rng,
                )
            else:
                logical_units = _draw_logical_units(
                    groups, datasets_by_status, batch_size=batch_size, rng=rng
                )
                if adaptive_query_duration:
                    duration_plan = plan_duration(
                        [_query_duration(unit, representations) for unit in logical_units],
                        complete_probability=0.0, rng=rng,
                    )
                else:
                    if query_seconds is None:
                        raise ValueError(
                            "a fixed cropped view requires --query-seconds; "
                            "enable adaptive query duration for observed caps"
                        )
                    duration_plan = DurationPlan("cropped", query_seconds, query_seconds)
            cap = duration_plan.cap_seconds
            if cap is not None:
                cap = max(cap, _minimum_target_duration(logical_units))
            if cap is not None and query_seconds is not None:
                cap = min(cap, query_seconds)
            if cap is not None and cap + 1e-9 < _minimum_target_duration(logical_units):
                # An explicit engineering cap cannot create a partial positive target.
                rejection_count += 1
                continue
            candidate_episodes = []
            for unit in logical_units:
                try:
                    candidate_episodes.append(
                        _training_episode(
                            unit,
                            recording_caches[unit.dataset],
                            representations,
                            seconds=cap,
                            rng=rng,
                        )
                    )
                except ValueError:
                    rejection_count += 1
                    break
            if len(candidate_episodes) == batch_size:
                episodes = candidate_episodes
                used_cap = cap
                break
        if len(episodes) < batch_size:
            raise RuntimeError("could not assemble a complete eligible Task-1 batch")
        batch = collate_detection_episodes(episodes).to(device)
        result = train_step(model, batch, optimizer, grad_clip=grad_clip)
        if step == 0 or (step + 1) % telemetry_every == 0 or step + 1 == steps:
            query_lengths = [int(item.query.valid.sum()) for item in episodes]
            reference_lengths = [int(item.reference.valid.sum()) for item in episodes]
            retained = [float(item.metadata["query_retained_duration_sec"]) for item in episodes]
            source = [float(item.metadata["query_source_duration_sec"]) for item in episodes]
            telemetry.append({
                "step": step + 1,
                "loss": result.loss,
                **result.telemetry,
                "batch/query_padding_fraction": padding_fraction(query_lengths),
                "batch/reference_padding_fraction": padding_fraction(reference_lengths),
                "batch/query_source_duration_mean_sec": float(np.mean(source)),
                "batch/query_retained_duration_mean_sec": float(np.mean(retained)),
                "batch/query_complete_fraction": float(np.mean([
                    item.metadata["query_view_mode"] == "complete" for item in episodes
                ])),
                "batch/query_duration_cap_sec": (
                    0.0 if used_cap is None else float(used_cap)
                ),
            })
    model.eval()
    return model, telemetry, rejection_count


def train(args: argparse.Namespace) -> dict[str, Any]:
    cohort = read_cohort_manifest(args.cohort)
    train_manifest = read_task_manifest(args.train_manifest)
    representations = open_representations(args.representations, cohort=cohort)
    train_manifest, train_common = select_common_units(
        train_manifest,
        args.common_train_units,
        representation_provenance=representations.metadata["encoder_provenance"],
    )
    datasets = sorted({str(row["dataset"]) for row in train_manifest.units})
    recording_caches = {dataset: open_cache(dataset) for dataset in datasets}
    validate_task_manifest(train_manifest, cohort, recording_caches)
    if train_manifest.task != "task1" or train_manifest.protocol.get("split") != "train":
        raise ValueError("--train-manifest must be a Task-1 train manifest")
    first = representations.get(
        datasets[0],
        str(train_manifest.units[0]["query_recording_id"]),
        str(train_manifest.units[0]["query_stream_id"]),
    )
    units = [Task1EvaluationUnit(**row) for row in train_manifest.units]
    fit_units, heldout_units = split_by_subject(
        units,
        seed=args.seed,
        heldout_fraction=float(OPERATING_POINT_PROTOCOL["holdout_fraction"]),
    )
    model, telemetry, rejection_count = fit_head(
        fit_units,
        recording_caches,
        representations,
        feature_dim=first.embeddings.shape[1],
        steps=args.steps,
        batch_size=args.batch_size,
        query_seconds=args.query_seconds,
        query_complete_probability=args.query_complete_probability,
        adaptive_query_duration=True,
        projection_dim=args.projection_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        telemetry_every=args.telemetry_every,
        seed=args.seed,
        device=args.device,
    )
    budget = float(OPERATING_POINT_PROTOCOL["false_alarm_budget_per_hour"])
    point_kwargs = dict(
        false_alarm_budget_per_hour=budget,
        nms_iou=float(OPERATING_POINT_PROTOCOL["nms_iou"]),
        match_iou=float(OPERATING_POINT_PROTOCOL["match_iou"]),
    )
    learned_point = fix_operating_point(
        heldout_units, recording_caches, representations, model, **point_kwargs
    )
    direct_point = fix_operating_point(
        heldout_units, recording_caches, representations, None, **point_kwargs
    )

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 3,
        "feature_dim": first.embeddings.shape[1],
        "projection_dim": args.projection_dim,
        "model_state_dict": model.state_dict(),
        "cohort_fingerprint": cohort.fingerprint,
        "train_manifest_fingerprint": train_manifest.fingerprint,
        "representation_provenance": representations.metadata["encoder_provenance"],
        "representation_metadata": representations.metadata,
        "train_common_unit_count": len(train_manifest.units),
        "train_common_unit_fingerprint": (
            None if train_common is None else fingerprint_protocol(train_common)
        ),
        "operating_point_protocol": dict(OPERATING_POINT_PROTOCOL),
        "training_view_policy": {
            **TRAINING_VIEW_POLICY,
            "query_complete_probability": args.query_complete_probability,
            "query_crop_cap_seconds": args.query_seconds,
            "adaptive_query_duration": True,
        },
        "operating_point": {
            key: value for key, value in learned_point.items() if key != "rejections"
        },
        "config": vars(args),
    }
    torch.save(checkpoint, args.output / "task1_head.pt")

    report = {
        "task": "task1",
        "status": "trained_and_operating_point_fixed",
        "cohort_fingerprint": cohort.fingerprint,
        "train_manifest_fingerprint": train_manifest.fingerprint,
        "representation_provenance": representations.metadata["encoder_provenance"],
        "representation_roots": [str(path) for path in args.representations],
        "train_units": len(units),
        "fit_units": len(fit_units),
        "heldout_units": len(heldout_units),
        "train_common_units": train_common,
        "train_episode_rejections": rejection_count,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "telemetry": telemetry,
        "operating_point_protocol": dict(OPERATING_POINT_PROTOCOL),
        "training_view_policy": {
            **TRAINING_VIEW_POLICY,
            "query_complete_probability": args.query_complete_probability,
            "query_crop_cap_seconds": args.query_seconds,
            "adaptive_query_duration": True,
        },
        "learned": learned_point,
        "direct": direct_point,
    }
    (args.output / "task1_training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path, default=root / "manifests/COHORT_TASK1_V2.json"
    )
    parser.add_argument(
        "--train-manifest", type=Path, default=root / "manifests/TASK1_TRAIN_V2.json"
    )
    parser.add_argument("--common-train-units", type=Path, required=True)
    parser.add_argument(
        "--representations",
        type=Path,
        nargs="+",
        required=True,
        help="one or more representation caches (e.g. natural + synthetic) for one encoder",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--query-seconds", type=float,
        help="optional upper bound for cropped query views; complete views ignore it",
    )
    parser.add_argument(
        "--query-complete-probability", type=float, default=0.5,
        help="probability that a logical training batch uses complete query views",
    )
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--telemetry-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
