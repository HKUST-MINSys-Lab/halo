"""Scenario 1 — partial enrollment coverage ("enrol some, ask about all").

Deployment story: a user enrols examples for the activities they can easily demonstrate and asks
the system to recognise a longer list.  Some candidates therefore carry no support at all, and the
ground truth is sometimes one of those.

This module is deliberately *additive*.  It never mutates a sealed manifest in place, never changes
the fingerprint of a fully covered manifest, and imports the immutable episode machinery from
:mod:`training.support_classifier.sealed_eval` rather than duplicating it.

Design notes
------------
* **Hiding is per cell, not per query.**  One deployment configuration is one hidden candidate set
  shared by every query in the cell.  This keeps the support count uniform across episodes (so the
  batched GPU readouts stay valid), and it is the honest experimental unit: a deployment either has
  an enrolment for an activity or it does not.
* **Support-only readouts can never name a hidden candidate.**  1-NN, prototype and ridge score the
  supported subset, so a query whose truth is hidden is necessarily wrong.  That is the measurement,
  not a bug: it is exactly the capability a support-only model lacks.
* **The hybrid readout is fixed and untrained.**  Text-path models get z-scored text evidence over
  the full roster plus z-scored support evidence where it exists.  Nothing is fitted, so no baseline
  is handicapped by our training choices and none is credited with a tuned combiner.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .sealed_eval import QueryPlan, _normalise

__all__ = [
    "CoverageCell",
    "choose_hidden_candidates",
    "hide_supports",
    "truth_split",
    "support_only_predictions",
    "hybrid_predictions",
    "zscore",
]


@dataclass(frozen=True)
class CoverageCell:
    """Which candidates carry enrolment in one partial-coverage cell."""

    supported: tuple[str, ...]
    hidden: tuple[str, ...]
    coverage: float
    requested_coverage: float

    @property
    def fingerprint(self) -> str:
        payload = "|".join(("supported", *self.supported, "hidden", *self.hidden))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _stable_rng(seed_parts: Sequence[object]) -> np.random.Generator:
    digest = hashlib.sha256("|".join(map(str, seed_parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def choose_hidden_candidates(
    candidates: Sequence[str],
    *,
    coverage: float,
    seed_parts: Sequence[object],
) -> CoverageCell:
    """Pick the candidates that receive no enrolment, stably and reproducibly.

    ``coverage`` is the fraction of candidates that keep their supports.  The number kept is
    rounded to the nearest integer and then clamped so that at least one candidate is supported and
    at least one is hidden: a cell with full or empty coverage is not this scenario and is refused
    rather than silently degenerating into the standard protocol.
    """
    roster = tuple(str(label) for label in candidates)
    if len(roster) != len(set(roster)):
        raise ValueError("candidate roster contains duplicates")
    if len(roster) < 2:
        raise ValueError("partial coverage needs at least two candidates")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie strictly between 0 and 1")

    keep = int(round(coverage * len(roster)))
    keep = max(1, min(len(roster) - 1, keep))
    rng = _stable_rng(seed_parts)
    order = rng.permutation(len(roster))
    supported = tuple(sorted(roster[index] for index in order[:keep]))
    hidden = tuple(sorted(roster[index] for index in order[keep:]))
    return CoverageCell(
        supported=supported,
        hidden=hidden,
        coverage=keep / len(roster),
        requested_coverage=float(coverage),
    )


def hide_supports(plans: Sequence[QueryPlan], cell: CoverageCell) -> list[QueryPlan]:
    """Strip every support row belonging to a hidden candidate.

    The query row, its identity and the execution-disjointness of the surviving supports are
    untouched, so the resulting episodes remain a strict subset of the sealed manifest.
    """
    hidden = set(cell.hidden)
    out: list[QueryPlan] = []
    for plan in plans:
        keep = [
            index for index, label in enumerate(plan.support_labels) if label not in hidden
        ]
        out.append(replace(
            plan,
            support=tuple(plan.support[index] for index in keep),
            support_labels=tuple(plan.support_labels[index] for index in keep),
        ))
    return out


def truth_split(
    truth: Sequence[str], cell: CoverageCell
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices whose ground truth is supported, and those whose ground truth is hidden."""
    labels = np.asarray([str(value) for value in truth], dtype=object)
    supported = np.asarray([label in set(cell.supported) for label in labels], dtype=bool)
    return np.flatnonzero(supported), np.flatnonzero(~supported)


def zscore(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Standardise each row over its unmasked entries; masked entries return 0.

    A row with fewer than two live entries, or with no spread, contributes no ranking information
    and is returned as zeros rather than as amplified noise.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if mask is None:
        mask = np.ones(scores.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != scores.shape:
        raise ValueError("mask must match the score matrix")
    out = np.zeros_like(scores)
    counts = mask.sum(axis=1)
    eligible = counts >= 2
    if not np.any(eligible):
        return out

    safe_counts = np.maximum(counts, 1)
    finite = np.where(mask, scores, 0.0)
    means = finite.sum(axis=1) / safe_counts
    centered = np.where(mask, scores - means[:, None], 0.0)
    spreads = np.sqrt(np.square(centered).sum(axis=1) / safe_counts)
    scales = np.maximum(1.0, np.abs(finite).max(axis=1))
    stable = eligible & (spreads > np.finfo(scores.dtype).eps * scales * 16.0)
    if np.any(stable):
        out[stable] = np.where(
            mask[stable], centered[stable] / spreads[stable, None], 0.0,
        )
    return out


def support_only_predictions(
    features: np.ndarray,
    candidates: Sequence[str],
    plans: Sequence[QueryPlan],
    cell: CoverageCell,
    *,
    ridge_alpha: float = 1.0,
    device=None,
) -> dict[str, list[str]]:
    """1-NN, prototype and ridge restricted to the candidates that actually carry enrolment.

    Hidden candidates are unreachable by construction, which is the point of the scenario.  The
    prototype readout skips empty classes instead of averaging an empty slice into ``nan``.
    """
    if not cell.supported:
        raise ValueError("a partial-coverage cell must support at least one candidate")
    if device is not None:
        # Hiding is cell-wide, so every surviving plan still carries the same k examples for each
        # supported class.  The sealed evaluator's batched implementation is therefore exactly the
        # same readout over a smaller candidate roster, including the same ridge system.
        from .sealed_eval import _readout_predictions

        return _readout_predictions(
            features,
            np.empty(len(features), dtype=object),
            tuple(cell.supported),
            plans,
            ridge_alpha=ridge_alpha,
            device=device,
        )
    z = _normalise(features)
    supported = list(cell.supported)
    slot_of = {label: index for index, label in enumerate(supported)}
    result: dict[str, list[str]] = {"1nn": [], "prototype": [], "ridge": []}
    for plan in plans:
        if not plan.support:
            raise ValueError("partial coverage still requires at least one enrolled support row")
        query = z[plan.query]
        x = z[np.asarray(plan.support, dtype=np.int64)]
        y = np.asarray([slot_of[label] for label in plan.support_labels], dtype=np.int64)

        result["1nn"].append(str(plan.support_labels[int(np.argmax(x @ query))]))

        present = [slot for slot in range(len(supported)) if np.any(y == slot)]
        prototypes = np.stack([x[y == slot].mean(axis=0) for slot in present])
        best = int(np.argmax(_normalise(prototypes) @ query))
        result["prototype"].append(str(supported[present[best]]))

        target = np.eye(len(supported), dtype=np.float64)[y]
        if len(x) <= x.shape[1]:
            gram = x @ x.T + ridge_alpha * np.eye(len(x), dtype=np.float64)
            scores = query @ x.T @ np.linalg.solve(gram, target)
        else:
            gram = x.T @ x + ridge_alpha * np.eye(x.shape[1], dtype=np.float64)
            scores = query @ np.linalg.solve(gram, x.T @ target)
        # Only classes with enrolment may be named; an empty column would otherwise win on ties.
        live = np.full(len(supported), -np.inf)
        for slot in present:
            live[slot] = scores[slot]
        result["ridge"].append(str(supported[int(np.argmax(live))]))
    return result


def hybrid_predictions(
    text_scores: np.ndarray,
    features: np.ndarray,
    candidates: Sequence[str],
    plans: Sequence[QueryPlan],
    cell: CoverageCell,
) -> list[str]:
    """The fixed, untrained text+support combiner used by every text-path model.

    ``text_scores`` is ``(n_windows, n_candidates)`` in the model's own candidate order.  Per query
    the text row is standardised over the full roster and the 1-NN similarity row over the supported
    subset only; the two are summed, so a hidden candidate competes on text evidence alone and a
    supported one carries both.  Nothing here is fitted to any dataset.
    """
    roster = list(str(label) for label in candidates)
    text_scores = np.asarray(text_scores, dtype=np.float64)
    if text_scores.shape[1] != len(roster):
        raise ValueError("text scores must have one column per candidate")
    z = _normalise(features)
    supported = set(cell.supported)
    support_mask = np.asarray([label in supported for label in roster], dtype=bool)

    rows = np.asarray([plan.query for plan in plans], dtype=np.int64)
    text_component = zscore(text_scores[rows])

    if not plans:
        return []
    support_counts = {len(plan.support) for plan in plans}
    if len(support_counts) != 1 or not next(iter(support_counts)):
        raise ValueError("hybrid readout requires a uniform, non-empty support set per query")
    support_rows = np.asarray([plan.support for plan in plans], dtype=np.int64)
    query_rows = np.asarray([plan.query for plan in plans], dtype=np.int64)
    similarities = np.einsum(
        "nsd,nd->ns", z[support_rows], z[query_rows], optimize=True,
    )
    candidate_index = {label: index for index, label in enumerate(roster)}
    try:
        support_slots = np.asarray([
            [candidate_index[label] for label in plan.support_labels] for plan in plans
        ], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"support label is absent from the candidate roster: {error.args[0]}") from error
    similarity = np.full((len(plans), len(roster)), -np.inf)
    episode_rows = np.broadcast_to(
        np.arange(len(plans), dtype=np.int64)[:, None], support_slots.shape,
    )
    np.maximum.at(
        similarity,
        (episode_rows.ravel(), support_slots.ravel()),
        similarities.ravel(),
    )

    live = np.isfinite(similarity) & support_mask
    finite = np.where(live, similarity, 0.0)
    support_component = zscore(finite, live)

    combined = text_component + support_component
    return [roster[index] for index in combined.argmax(axis=1).tolist()]
