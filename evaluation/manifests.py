"""Sealed-cell rosters and execution-disjoint episode manifests shared by every rung.

Extracted verbatim from training/support_classifier/sealed_eval.py on 2026-09-23 (Phase 0 of
docs/journal/2026-09-22-rung1-rung2-implementation-plan.md). sealed_eval re-imports these names.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence
from baselines import scoring
from baselines.data import EvalStream
from data.scripts.curate.deployment_policy import (
    MULTI_DEVICE_EVAL_CELLS,
    PROSPECTIVE_EVAL_DATASETS,
    SEALED_TEST_EVAL_DATASETS,
    stream_specs,
)


SEED = 20260912


@dataclass(frozen=True)
class QueryPlan:
    """One query and its execution-disjoint enrolled rows for every candidate."""

    query: int
    support: tuple[int, ...]
    support_labels: tuple[str, ...]


def sealed_cells(scope: str = "sealed") -> tuple[tuple[str, str], ...]:
    """Resolve one declared evaluation scope without admitting caller-supplied datasets."""
    if scope == "sealed":
        datasets = SEALED_TEST_EVAL_DATASETS
    elif scope == "prospective":
        datasets = PROSPECTIVE_EVAL_DATASETS
    else:
        raise ValueError(f"unknown evaluation scope {scope!r}")
    return tuple(
        (dataset, spec.stream_id)
        for dataset in datasets
        for spec in stream_specs(dataset, "primary")
    )


def duration_cells(window_seconds: Sequence[float], *, scope: str = "sealed") -> tuple[tuple[float, str, str], ...]:
    """Expand one declared roster over evidence budgets deterministically."""
    return tuple(
        (float(duration), dataset, stream)
        for duration in sorted(set(window_seconds))
        for dataset, stream in sealed_cells(scope)
    )


def evaluation_cells(
    window_seconds: Sequence[float], *, scope: str = "sealed",
) -> tuple[tuple[float, str, str, tuple[str, ...]], ...]:
    """Single placements plus declared composites for the requested protocol scope."""
    singles = [
        (duration, dataset, stream, ())
        for duration, dataset, stream in duration_cells(window_seconds, scope=scope)
    ]
    if scope != "sealed":
        return tuple(singles)
    composites = [
        (float(duration), cell.dataset, cell.cell_id, tuple(cell.stream_ids))
        for duration in sorted(set(window_seconds))
        for cell in MULTI_DEVICE_EVAL_CELLS
        if cell.dataset in SEALED_TEST_EVAL_DATASETS
    ]
    return tuple(singles + composites)


def _aligned_labels(stream: EvalStream) -> np.ndarray:
    return np.asarray(
        scoring.align_ground_truth_labels(stream.gt, stream.eval_labels), dtype=object,
    )


def _stable_choice(values: np.ndarray, count: int, *, seed_parts: Sequence[object]) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if count < 0 or count > len(values):
        raise ValueError("choice count must lie within the available pool")
    digest = hashlib.sha256("|".join(map(str, seed_parts)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    # The exact NumPy draw is frozen protocol behavior. Optimize manifest reuse, not the mapping
    # from a public seed to episode rows.
    return np.asarray(rng.choice(values, size=count, replace=False), dtype=np.int64)


def build_manifest(
    stream: EvalStream,
    k: int,
    *,
    seed: int = SEED,
    query_rows: Sequence[int] | None = None,
    support_rows: Sequence[int] | None = None,
) -> list[QueryPlan]:
    """Create execution-disjoint target-dataset episodes without looking at representations.

    Each target candidate receives exactly k labelled support *windows* from a different physical
    execution than the query.  The candidate roster is always the frozen native vocabulary.  A
    query is omitted when that honest episode cannot be formed; this is reported rather than
    padded or silently relaxed.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    n_rows = stream.n_windows
    query_allowed = (np.arange(n_rows, dtype=np.int64) if query_rows is None
                     else np.asarray(query_rows, dtype=np.int64))
    support_allowed = (np.arange(n_rows, dtype=np.int64) if support_rows is None
                       else np.asarray(support_rows, dtype=np.int64))
    for name, rows in (("query", query_allowed), ("support", support_allowed)):
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= n_rows):
            raise ValueError(f"{name}_rows are outside the stream")
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"{name}_rows contain duplicates")
    if k == 0:
        labels = _aligned_labels(stream)
        return [QueryPlan(query=int(i), support=(), support_labels=())
                for i in query_allowed if labels[i] in stream.eval_labels]
    if not stream.execution_identity_known or stream.execution_ids is None:
        raise ValueError(
            f"{stream.dataset}/{stream.stream}: execution identity is unavailable; refusing "
            "enrollment evaluation that could leak a recording into its own support set"
        )
    labels = _aligned_labels(stream)
    valid = np.flatnonzero(labels != None)  # noqa: E711 - object-array comparison is intentional
    query_valid = query_allowed[labels[query_allowed] != None]  # noqa: E711
    plans: list[QueryPlan] = []
    candidates = tuple(stream.eval_labels)
    # Index once. The original direct expression scanned every valid row for every
    # (query, candidate) pair, which turns the optional high-k curve into needless quadratic CPU
    # work. A candidate-local index preserves exactly the same execution-disjoint rule.
    label_rows = {
        label: support_allowed[labels[support_allowed] == label]
        for label in candidates
    }
    label_execution = {
        label: np.asarray(stream.execution_ids[rows], dtype=object)
        for label, rows in label_rows.items()
    }
    execution_pools: dict[tuple[str, object], np.ndarray] = {}
    for query in query_valid.tolist():
        q_execution = stream.execution_ids[query]
        support: list[int] = []
        support_labels: list[str] = []
        possible = True
        for label in candidates:
            rows = label_rows[label]
            pool_key = (label, q_execution)
            pool = execution_pools.get(pool_key)
            if pool is None:
                pool = rows[label_execution[label] != q_execution]
                execution_pools[pool_key] = pool
            if len(pool) < k:
                possible = False
                break
            picked = _stable_choice(
                pool, k, seed_parts=(seed, stream.dataset, stream.stream, k, query, label),
            )
            support.extend(int(value) for value in picked)
            support_labels.extend([label] * k)
        if possible:
            plans.append(QueryPlan(query=query, support=tuple(support),
                                   support_labels=tuple(support_labels)))
    return plans


def manifest_fingerprint(plans: Iterable[QueryPlan]) -> str:
    payload = [asdict(plan) for plan in plans]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
