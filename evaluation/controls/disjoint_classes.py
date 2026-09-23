"""Disjoint-class control (threat: the model exploits the pool's acquisition fingerprint, not its
class structure). A seeded subset of the roster is held out; the scored set contains only kept-class
windows, the pool contains only held-out-class windows from the same stream. If the N-curve still
rises, the gain is domain adaptation to the acquisition; if it vanishes, it was class structure."""

from __future__ import annotations

import hashlib

import numpy as np

from evaluation.rung2_unlabeled.ncurve import CellSplit


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
