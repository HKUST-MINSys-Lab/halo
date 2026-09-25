"""Size-matched balanced-pool control for the unlabeled N-curve."""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np


def balanced_pool_filter(truth_ids: np.ndarray, n_classes: int, *, seed_parts=()) -> Callable:
    """Return a filter giving N rows with the most uniform feasible class counts, **nested in N**.

    Labels are used only by this diagnostic control. ``available_rows`` excludes any labeled
    supports before drawing, so the control never reuses them as unlabeled examples.

    For each distinct ``available_rows`` set the filter builds one class-interleaved order: every
    class's rows are shuffled once, then taken round-robin (class order reshuffled each round,
    exhausted classes skipped). The balanced pool of size N is the first N rows of that order, so
    B_50 ⊂ B_100 ⊂ ... exactly as the uncontrolled nested draws, and every prefix is as balanced as
    the available rows allow. The order depends only on the seed parts and the available rows, never
    on call order (the 2026-09-25 rewrite advanced one random stream across N, so successive N were
    independent re-draws rather than nested).
    """
    base_digest = hashlib.sha256("|".join(map(str, ("balanced_pool", *seed_parts))).encode()).digest()
    orders: dict[bytes, np.ndarray] = {}

    def order_for(source: np.ndarray) -> np.ndarray:
        key = hashlib.sha256(base_digest + np.sort(source).tobytes()).digest()
        if key not in orders:
            rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
            queues = {c: list(rng.permutation(source[truth_ids[source] == c]))
                      for c in range(n_classes) if bool((truth_ids[source] == c).any())}
            sequence: list[int] = []
            while queues:
                for c in rng.permutation(sorted(queues)):
                    sequence.append(int(queues[int(c)].pop()))
                    if not queues[int(c)]:
                        del queues[int(c)]
            orders[key] = np.asarray(sequence, dtype=np.int64)
        return orders[key]

    def apply(pool_rows: np.ndarray, *, available_rows: np.ndarray | None = None) -> np.ndarray:
        pool_rows = np.asarray(pool_rows, dtype=np.int64)
        if len(pool_rows) == 0:
            return pool_rows
        source = pool_rows if available_rows is None else np.asarray(available_rows, dtype=np.int64)
        if len(source) < len(pool_rows):
            raise ValueError("balanced-pool source has fewer rows than requested N")
        return np.sort(order_for(source)[:len(pool_rows)])

    return apply
