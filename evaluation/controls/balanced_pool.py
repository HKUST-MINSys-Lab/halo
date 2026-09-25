"""Size-matched balanced-pool control for the unlabeled N-curve."""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np


def balanced_pool_filter(truth_ids: np.ndarray, n_classes: int, *, seed_parts=()) -> Callable:
    """Resample N rows with the most uniform feasible class counts.

    Labels are used only by this diagnostic control. ``available_rows`` excludes any labeled
    supports before drawing, so the control never reuses them as unlabeled examples.
    """
    digest = hashlib.sha256("|".join(map(str, ("balanced_pool", *seed_parts))).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def apply(pool_rows: np.ndarray, *, available_rows: np.ndarray | None = None) -> np.ndarray:
        pool_rows = np.asarray(pool_rows, dtype=np.int64)
        if len(pool_rows) == 0:
            return pool_rows
        source = pool_rows if available_rows is None else np.asarray(available_rows, dtype=np.int64)
        if len(source) < len(pool_rows):
            raise ValueError("balanced-pool source has fewer rows than requested N")
        members = {c: source[truth_ids[source] == c] for c in range(n_classes)}
        counts = {c: 0 for c, rows in members.items() if len(rows)}
        for _ in range(len(pool_rows)):
            smallest = min(counts[c] for c in counts if counts[c] < len(members[c]))
            eligible = [c for c in counts if counts[c] == smallest and counts[c] < len(members[c])]
            counts[int(rng.choice(eligible))] += 1
        out = []
        for c, count in counts.items():
            out.extend(rng.choice(members[c], size=count, replace=False).tolist())
        return np.asarray(sorted(out), dtype=np.int64)

    return apply
