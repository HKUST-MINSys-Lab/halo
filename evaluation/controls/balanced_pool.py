"""Balanced-pool control (threat: transductive gains are an artefact of balanced query sets —
Veilleux et al. 2022). Resamples each P_N to a uniform class marginal, so the difference between
the main run and this one bounds how much of the gain the real, imbalanced marginal costs or gives."""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np


def balanced_pool_filter(truth_ids: np.ndarray, n_classes: int, *, seed_parts=()) -> Callable[[np.ndarray], np.ndarray]:
    """Return a filter mapping a pool draw to a class-balanced subset of the same-or-smaller size.

    Uses labels — it is a control, and says so in its row. Each class contributes
    ``min(count, floor(N / classes_present))`` rows; a class absent from the draw stays absent.
    """
    digest = hashlib.sha256("|".join(map(str, ("balanced_pool", *seed_parts))).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def apply(pool_rows: np.ndarray) -> np.ndarray:
        pool_rows = np.asarray(pool_rows, dtype=np.int64)
        if len(pool_rows) == 0:
            return pool_rows
        labels = truth_ids[pool_rows]
        present = [c for c in range(n_classes) if (labels == c).any()]
        per_class = len(pool_rows) // max(len(present), 1)
        out = []
        for c in present:
            members = pool_rows[labels == c]
            take = min(len(members), per_class)
            out.extend(rng.choice(members, size=take, replace=False).tolist())
        return np.asarray(sorted(out), dtype=np.int64)

    return apply
