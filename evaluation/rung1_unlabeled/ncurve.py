"""The N-curve protocol: accuracy on a FIXED scored set as the unlabelled pool grows.

Why the scored set is fixed: if the transduced set *were* the scored set, accuracy at different N
would be measured on different windows and the curve would be confounded by which windows were
drawn. So per cell the queries are split **by physical execution** into a scored set S (20 %) and
a pool P (80 %); nested subsets P_50 ⊂ P_100 ⊂ … ⊂ P are drawn once from a seeded permutation;
transduction runs over S ∪ P_N; every metric is computed on S only.

* The **inductive anchor** (``method="inductive"``, recorded with N = 0) involves no transduction at
  all, and at k = 0 must reproduce the published
  sealed zero-shot rule on the same windows (the arg-max of the zero-shot scores). That is the
  self-consistency check, also emitted over *all* windows so it can be compared row for row with
  the sealed results file.
* k > 0 supports are a **cell-level shared set** of k windows per class drawn from P (execution-
  disjoint from S by construction), because transduction is joint over the pool and cannot take
  the sealed manifest's per-query support sets. The inductive k > 0 readout at N = 0 is the
  normalised class-prototype rule the sealed table calls ``prototype``, on that shared set; it is
  the same readout, not the same draw, and is labelled as such.
* The inductive row is an **anchor**, not the null of the curve. It reads a different space (the
  zero-shot arg-max, or embedding prototypes at k > 0) from every transductive row (simplex
  probability features), so "gain over inductive" mixes a change of feature space with the effect
  of the pool. The curve's null is the transductive row at **N = 0**: EM-Dirichlet over S alone
  (plus the k supports), same method, same features, no pool. Pass ``0`` in ``pool_draws`` to get
  it; "does accuracy rise with unlabelled data" is read against that row.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from evaluation.metrics import classification
from evaluation.rung1_unlabeled.transductive import INFERENCE_DEFAULTS, transduce_numpy
from evaluation.zero_shot import _normalise, probability_features

DEFAULT_POOL_SIZES: tuple[int | str, ...] = (0, 50, 100, 500, 2000, "all")
# Rung 1 is k = 0: the roster is known by name and no labelled example is ever provided. k > 0
# (labels plus an unlabelled pool) remains computable for a separately named semi-supervised
# condition, but it is not rung 1 and is not in the default grid.
DEFAULT_K: tuple[int, ...] = (0,)


def _rng(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


@dataclass(frozen=True)
class CellSplit:
    scored: np.ndarray       # row indices of S
    pool: np.ndarray         # row indices of P (execution-disjoint from S)
    n_executions_scored: int
    n_executions_pool: int


def split_scored_pool(execution_ids: np.ndarray, valid_rows: np.ndarray, *, fraction: float = 0.2,
                      seed_parts: Sequence[object] = ()) -> CellSplit:
    """Deterministic execution-level split. Executions are shuffled and accumulated into S until
    at least ``fraction`` of the valid rows are covered; the remainder is P."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie in (0, 1)")
    valid_rows = np.asarray(valid_rows, dtype=np.int64)
    executions = np.asarray(execution_ids, dtype=object)[valid_rows]
    unique = sorted(set(executions.tolist()), key=str)
    if len(unique) < 2:
        raise ValueError("at least two executions are needed to split scored and pool sets")
    order = list(unique)
    _rng("scored_pool_split", *seed_parts).shuffle(order)
    target = fraction * len(valid_rows)
    scored_exec: set = set()
    covered = 0
    for execution in order:
        if covered >= target and scored_exec:
            break
        scored_exec.add(execution)
        covered += int((executions == execution).sum())
    scored_mask = np.asarray([e in scored_exec for e in executions], dtype=bool)
    if scored_mask.all():
        raise ValueError("split left no pool executions")
    return CellSplit(scored=valid_rows[scored_mask], pool=valid_rows[~scored_mask],
                     n_executions_scored=len(scored_exec), n_executions_pool=len(unique) - len(scored_exec))


def nested_pool_draws(pool: np.ndarray, sizes: Sequence[int | str], *,
                      seed_parts: Sequence[object] = ()) -> dict[str, np.ndarray]:
    """Nested prefixes of one seeded permutation of ``pool``: P_50 ⊂ P_100 ⊂ …; ``"all"`` = P."""
    pool = np.asarray(pool, dtype=np.int64)
    order = pool[_rng("nested_pool", *seed_parts).permutation(len(pool))]
    out: dict[str, np.ndarray] = {}
    for size in sizes:
        if size == "all":
            out["all"] = order
        else:
            n = int(size)
            if n < 0:
                raise ValueError("pool sizes must be non-negative")
            out[str(n)] = order[:min(n, len(order))]
    return out


def shared_support_set(pool: np.ndarray, truth_ids: np.ndarray, k: int, n_classes: int, *,
                       seed_parts: Sequence[object] = ()) -> tuple[np.ndarray, np.ndarray] | None:
    """k rows per class from P (already execution-disjoint from S). ``None`` if any class lacks k."""
    if k <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rng = _rng("shared_support", *seed_parts)
    rows, labels = [], []
    for c in range(n_classes):
        candidates = pool[truth_ids[pool] == c]
        if len(candidates) < k:
            return None
        picked = rng.choice(candidates, size=k, replace=False)
        rows.extend(int(r) for r in picked)
        labels.extend([c] * k)
    return np.asarray(rows, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def inductive_predictions(*, scores: np.ndarray, features: np.ndarray | None, support_rows: np.ndarray,
                          support_labels: np.ndarray, n_classes: int, rows: np.ndarray) -> np.ndarray:
    """N = 0 readout. k = 0: arg-max of the zero-shot scores (the sealed rule). k > 0: normalised
    class prototypes of the shared supports on the enrollment features (sealed ``prototype``)."""
    if len(support_rows) == 0:
        return np.asarray(scores[rows].argmax(axis=1), dtype=np.int64)
    if features is None:
        raise ValueError("k > 0 inductive readout needs enrollment features")
    z = _normalise(features)
    prototypes = np.stack([z[support_rows[support_labels == c]].mean(axis=0) for c in range(n_classes)])
    return np.asarray((z[rows] @ _normalise(prototypes).T).argmax(axis=1), dtype=np.int64)


def neighbour_purity(truth: np.ndarray, neighbours: np.ndarray | None) -> float | None:
    """Share of embedding-space neighbours with the same true class (in-roster rows only).

    The affinity term's registered risk is an encoder that groups windows by subject or device
    rather than by activity, in which case the neighbour vote spreads errors. This measures that
    directly, per encoder and cell; it uses labels and is a diagnostic, never an input."""
    if neighbours is None or neighbours.size == 0:
        return None
    truth = np.asarray(truth)
    own = truth[:, None]
    other = truth[neighbours]
    counted = (own >= 0) & (other >= 0)
    return float(((own == other) & counted).sum() / max(int(counted.sum()), 1))


def pool_marginal(truth_ids: np.ndarray, rows: np.ndarray, n_classes: int) -> dict:
    counts = np.bincount(truth_ids[rows], minlength=n_classes).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {"pool_class_counts": counts.astype(int).tolist(), "pool_marginal_entropy": entropy,
            "pool_marginal_entropy_max": float(np.log(n_classes)),
            "pool_classes_present": int((counts > 0).sum())}


def run_cell(
    *,
    scores_all: np.ndarray,            # (N_total, C) zero-shot scores over the roster
    score_kind: str,
    features_all: np.ndarray | None,   # enrollment features for the k>0 inductive prototype readout
    truth_ids: np.ndarray,             # (N_total,) roster ids, -1 outside the roster
    classes: Sequence[str],
    split: CellSplit,
    pool_draws: dict[str, np.ndarray],
    ks: Sequence[int] = DEFAULT_K,
    temperature: float = 30.0,
    transduce_kwargs: dict | None = None,
    assignments: Sequence[str] = ("identity", "graph"),
    seed_parts: Sequence[object] = (),
    pool_filter: Callable[[np.ndarray], np.ndarray] | None = None,
    control: str = "none",
    device=None,
) -> list[dict]:
    """Every rung-1 row for one (encoder, cell): the N × k grid on S, plus the all-windows
    reproduction row at N = 0, k = 0. ``pool_filter`` (a control) may resample each P_N."""
    classes = list(classes)
    C = len(classes)
    kwargs = {**INFERENCE_DEFAULTS, **(transduce_kwargs or {})}
    z_all, feature_info = probability_features(scores_all, score_kind, temperature=temperature)
    truth_names = np.asarray(classes, dtype=object)
    rows: list[dict] = []

    def metrics_on(rows_idx: np.ndarray, pred_ids: np.ndarray) -> dict:
        truth = truth_names[truth_ids[rows_idx]]
        pred = truth_names[pred_ids]
        return classification(truth, pred)

    # The reproduction row: the sealed k=0 rule over every in-roster window, no transduction.
    all_rows = np.flatnonzero(truth_ids >= 0)
    rows.append({"method": "inductive", "k": 0, "N": 0, "scope": "all_windows", "control": control,
                 **metrics_on(all_rows, scores_all[all_rows].argmax(axis=1)),
                 "score_kind": score_kind, **feature_info})

    for k in ks:
        support = shared_support_set(split.pool, truth_ids, k, C, seed_parts=(*seed_parts, "k", k))
        if support is None:
            rows.append({"method": "transductive_clip_v1", "k": k, "N": None, "scope": "scored",
                         "control": control, "status": "n/a",
                         "reason": f"pool lacks {k} execution-disjoint windows for every class"})
            continue
        support_rows, support_labels = support
        pool_available = np.setdiff1d(split.pool, support_rows, assume_unique=False)
        inductive = inductive_predictions(scores=scores_all, features=features_all,
                                          support_rows=support_rows, support_labels=support_labels,
                                          n_classes=C, rows=split.scored)
        rows.append({"method": "inductive", "k": k, "N": 0, "scope": "scored", "control": control,
                     "inductive_readout": "zero_shot_argmax" if k == 0 else "prototype",
                     **metrics_on(split.scored, inductive), "n_scored": int(len(split.scored)),
                     "n_support": int(len(support_rows)), "score_kind": score_kind, **feature_info})
        for label, drawn in pool_draws.items():
            pool_n = drawn[np.isin(drawn, pool_available)] if len(support_rows) else drawn
            if pool_filter is not None:
                pool_n = pool_filter(pool_n)
            task_rows = np.concatenate([split.scored, pool_n])
            z = z_all[task_rows]
            support_z = z_all[support_rows] if k else None
            embeddings = None if features_all is None else features_all[task_rows]
            for assignment in assignments:
                if k and assignment != "identity":
                    continue           # few-shot components are pinned by supports; no matching
                preds, u, info = transduce_numpy(
                    z, support_z=support_z, support_labels=support_labels if k else None,
                    assignment=assignment, embeddings=embeddings, device=device, **kwargs,
                )
                neighbours = info.pop("neighbours")
                scored_preds = preds[:len(split.scored)]
                rows.append({
                    "method": "transductive_clip_v1", "k": k, "N": int(len(pool_n)), "N_label": label,
                    "scope": "scored", "control": control, "assignment": assignment,
                    **metrics_on(split.scored, scored_preds),
                    **pool_marginal(truth_ids, task_rows, C),
                    "n_scored": int(len(split.scored)), "n_support": int(len(support_rows)),
                    "n_transduced": int(len(task_rows)),
                    "lam": info["lam"], "n_iter": info["n_iter"], "n_iter_mm": info["n_iter_mm"],
                    "affinity_mu": info["mu"], "affinity_knn": info["knn"],
                    "neighbour_purity": neighbour_purity(truth_ids[task_rows], neighbours),
                    "cluster_sizes": info["cluster_sizes"],
                    "collapsed_components": int(sum(1 for s in info["cluster_sizes"] if s < 0.5)),
                    "score_kind": score_kind, **feature_info,
                })
    return rows
