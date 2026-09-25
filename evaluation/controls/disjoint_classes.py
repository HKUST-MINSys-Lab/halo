"""Disjoint-class diagnostic: scored and pool sets have no shared labels.

A change in the N-curve shows that pool-label overlap is not required. It does not by itself
identify domain adaptation; the pool may change the decision boundary for other reasons.
"""

from __future__ import annotations

import hashlib

import numpy as np

from evaluation.rung1_unlabeled.ncurve import CellSplit


def disjoint_class_split(split: CellSplit, truth_ids: np.ndarray, n_classes: int, *,
                         holdout_fraction: float = 0.5, seed_parts=()) -> tuple[CellSplit, np.ndarray, np.ndarray]:
    """Return (new split, kept class ids, held-out class ids)."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie in (0, 1)")
    digest = hashlib.sha256("|".join(map(str, ("disjoint_classes", *seed_parts))).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    order = rng.permutation(n_classes)
    n_hold = max(1, int(round(holdout_fraction * n_classes)))
    n_hold = min(n_hold, n_classes - 2)           # keep at least two roster classes to score
    held = np.sort(order[:n_hold])
    kept = np.sort(order[n_hold:])
    scored = split.scored[np.isin(truth_ids[split.scored], kept)]
    pool = split.pool[np.isin(truth_ids[split.pool], held)]
    if len(scored) == 0 or len(pool) == 0:
        raise ValueError("disjoint-class split left an empty scored or pool set")
    return CellSplit(scored=scored, pool=pool, n_executions_scored=split.n_executions_scored,
                     n_executions_pool=split.n_executions_pool), kept, held
