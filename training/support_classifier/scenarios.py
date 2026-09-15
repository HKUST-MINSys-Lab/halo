"""Deployment-heterogeneity scenarios 2-8: derived streams and cross-stream enrolment.

Scenario 1 (partial enrolment coverage) lives in :mod:`partial_coverage`; this module supplies the
machinery every other scenario needs, all of it evaluation-only:

* **Query-side perturbations** (Scenarios 4 and 5): an accel-only view and a resampled view of a
  sealed stream.  Both return a new :class:`EvalStream`; neither mutates the original.  The sealed
  feature cache keys on a content fingerprint, so a perturbed view can never collide with the
  unperturbed one.
* **Cross-stream enrolment** (Scenarios 2, 3, 7): supports drawn from one stream, queries from
  another, over a concatenated feature matrix.  Covers other-placement, other-dataset and
  other-device-set enrolment with one implementation.

Every manifest keeps the sealed protocol's two hard rules: a support may never come from the query's
own physical execution, and a query is dropped rather than padded when an honest episode cannot be
formed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy.signal import resample_poly

from baselines.data import EvalStream
from data.scripts.labels.canonical_labels import canonicalize

from .sealed_eval import QueryPlan, _aligned_labels, _stable_choice

__all__ = [
    "GYRO_PREFIXES",
    "CrossEnrolment",
    "build_cross_manifest",
    "derive_accel_only",
    "derive_resampled",
    "shared_candidates",
    "subset_stream",
    "stream_rows",
]

GYRO_PREFIXES = ("gyro", "gyr_", "gx", "gy", "gz", "angular")


def stream_rows(stream) -> int:
    """Row count for a single-placement or composite stream."""
    return int(len(stream.gt))


# ------------------------------------------------------------ query perturbations


def _gyro_channels(channels: Sequence[str]) -> np.ndarray:
    lowered = [str(name).lower() for name in channels]
    return np.asarray(
        [any(name.startswith(prefix) for prefix in GYRO_PREFIXES) for name in lowered],
        dtype=bool,
    )


def derive_accel_only(stream: EvalStream) -> EvalStream:
    """Scenario 4: the same windows with the gyroscope removed.

    The gyroscope channels are zeroed *and* marked absent in the channel mask, which is how every
    consumer already represents a sensor the deployment does not have.  Zeroing without masking
    would instead present a broken gyroscope reading zero, which is a different and much easier
    problem. The physical stream identity stays intact; ``perturbation`` distinguishes derived
    views in cache keys and reports without breaking metadata lookup.
    """
    gyro = _gyro_channels(stream.channels)
    if not gyro.any():
        raise ValueError(f"{stream.dataset}/{stream.stream} carries no gyroscope to drop")
    windows = np.array(stream.windows, copy=True)
    windows[:, :, gyro] = 0.0
    mask = np.array(stream.mask, copy=True)
    mask[gyro] = False
    if not mask.any():
        raise ValueError("dropping the gyroscope would leave no valid channel")
    return replace(stream, windows=windows, mask=mask,
                   perturbation=_append_perturbation(stream.perturbation, "accel_only"))


def _append_perturbation(existing: str | None, current: str) -> str:
    return current if not existing else f"{existing}+{current}"


def _effective_source_rate(stream: EvalStream) -> float:
    value = stream.effective_source_rate_hz
    rate = float(stream.rate_hz if value is None else value)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("effective source rate must be finite and positive")
    return rate


def _resample_ratio(source_rate: float, target_rate: float) -> tuple[int, int]:
    ratio = Fraction(target_rate / source_rate).limit_denominator(10_000)
    if abs(float(ratio) - target_rate / source_rate) > 1e-10:
        raise ValueError(f"cannot represent sampling-rate ratio {target_rate}/{source_rate}")
    return ratio.numerator, ratio.denominator


def derive_resampled(stream: EvalStream, target_rate_hz: float) -> EvalStream:
    """Scenario 5: the same physical window observed at a different sampling rate.

    Downsampling is low-pass filtered first, so the result is band-limited acquisition rather than
    aliased garbage; upsampling is linear interpolation, which adds no information and is what a
    deployment pipeline would do.  The physical duration of a window is unchanged, so this varies
    the acquisition rate alone and not the evidence budget.
    """
    if target_rate_hz <= 0:
        raise ValueError("target rate must be positive")
    source_rate = float(stream.rate_hz)
    if abs(target_rate_hz - source_rate) < 1e-9:
        return stream
    up, down = _resample_ratio(source_rate, float(target_rate_hz))
    valid = (np.asarray(stream.lengths, dtype=np.int64) if stream.lengths is not None
             else np.full(stream.n_windows, stream.windows.shape[1], dtype=np.int64))
    if valid.shape != (stream.n_windows,) or np.any(valid <= 0) or np.any(valid > stream.windows.shape[1]):
        raise ValueError("derived resampling requires valid per-window lengths")

    out_lengths = np.asarray([(int(length) * up + down - 1) // down for length in valid], dtype=np.int64)
    max_length = int(out_lengths.max())
    resampled = np.zeros((stream.n_windows, max_length, stream.windows.shape[2]), dtype=np.float32)
    source = np.asarray(stream.windows, dtype=np.float64)
    # Equal valid lengths form dense blocks, so scipy can filter every window/channel in one call.
    for length in np.unique(valid):
        rows = np.flatnonzero(valid == length)
        expected = int((int(length) * up + down - 1) // down)
        if int(length) == 1:
            # SciPy's linear boundary extension has no slope to estimate and returns NaN for a
            # singleton. A constant is the only signal justified by one measured sample.
            block = np.repeat(source[rows, :1], expected, axis=1)
        else:
            block = resample_poly(
                source[rows, :int(length)], up, down, axis=1, padtype="line",
            )
        if block.shape[1] != expected:
            raise RuntimeError("polyphase resampler returned an unexpected valid length")
        if not np.isfinite(block).all():
            raise ValueError("polyphase resampling produced non-finite valid samples")
        resampled[rows, :expected] = block.astype(np.float32, copy=False)
    return replace(
        stream,
        windows=resampled,
        lengths=out_lengths,
        rate_hz=float(target_rate_hz),
        effective_source_rate_hz=min(_effective_source_rate(stream), float(target_rate_hz)),
        perturbation=_append_perturbation(stream.perturbation, f"rate{target_rate_hz:g}Hz-polyphase-v1"),
    )


def subset_stream(stream, rows: Sequence[int]):
    """Restrict a single or composite stream without changing event identity or metadata."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0 or np.any(rows < 0) or np.any(rows >= stream.n_windows):
        raise ValueError("stream subset rows must be a non-empty in-range vector")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("stream subset rows must be unique")
    if hasattr(stream, "devices"):
        devices = [subset_stream(member, rows) for member in stream.devices]
        return replace(stream, devices=devices, event_ids=np.asarray(stream.event_ids)[rows],
                       gt=[stream.gt[row] for row in rows], subjects=np.asarray(stream.subjects)[rows],
                       execution_ids=(np.asarray(stream.execution_ids)[rows]
                                      if stream.execution_ids is not None else None))
    return replace(stream, windows=np.asarray(stream.windows)[rows],
                   gt=[stream.gt[row] for row in rows], subjects=np.asarray(stream.subjects)[rows],
                   event_ids=np.asarray(stream.event_ids)[rows],
                   execution_ids=(np.asarray(stream.execution_ids)[rows]
                                  if stream.execution_ids is not None else None),
                   block_ids=(np.asarray(stream.block_ids)[rows] if stream.block_ids is not None else None),
                   lengths=(np.asarray(stream.lengths)[rows] if stream.lengths is not None else None))


# --------------------------------------------------------------- cross enrolment


@dataclass(frozen=True)
class CrossEnrolment:
    """Queries from one stream enrolled against supports from another.

    ``plans`` index a single concatenated feature matrix: query rows keep their own index and every
    support row is offset by the number of query rows.  :meth:`features` performs exactly that
    concatenation, so a caller cannot accidentally pair a plan with an unshifted matrix.
    """

    query_stream: object
    support_stream: object
    plans: tuple[QueryPlan, ...]
    candidates: tuple[str, ...]
    offset: int
    relation: str
    label_map: tuple[tuple[str, str], ...]

    def features(self, query_features: np.ndarray, support_features: np.ndarray) -> np.ndarray:
        if len(query_features) != self.offset:
            raise ValueError("query feature rows do not match the manifest offset")
        if query_features.shape[1] != support_features.shape[1]:
            raise ValueError("query and support features must share a dimension")
        return np.concatenate([query_features, support_features], axis=0)

    @property
    def fingerprint(self) -> str:
        payload = "|".join((
            str(self.query_stream.dataset), str(getattr(self.query_stream, "stream", "")),
            str(self.support_stream.dataset), str(getattr(self.support_stream, "stream", "")),
            self.relation, *self.candidates,
        ))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def shared_candidates(query_stream, support_stream) -> tuple[tuple[str, ...], dict[str, str]]:
    """Labels a query stream and a support stream can both express, by canonical concept.

    The roster is stated in the *query* stream's own words, because that is the vocabulary the model
    is asked to choose from.  The returned map sends a support-stream label to the query-stream
    label naming the same concept, which is what makes cross-dataset enrolment possible without
    inventing new label text.
    """
    query_by_concept = {}
    for label in query_stream.eval_labels:
        query_by_concept.setdefault(canonicalize(label), str(label))
    mapping: dict[str, str] = {}
    for label in support_stream.eval_labels:
        concept = canonicalize(label)
        if concept in query_by_concept:
            mapping[str(label)] = query_by_concept[concept]
    roster = tuple(sorted(set(mapping.values())))
    return roster, mapping


def build_cross_manifest(
    query_stream,
    support_stream,
    k: int,
    *,
    seed: int,
    relation: str = "cross_stream",
    same_subject: bool | None = None,
    allow_same_execution: bool = False,
    candidates: Sequence[str] | None = None,
    query_rows: Sequence[int] | None = None,
    support_rows: Sequence[int] | None = None,
) -> CrossEnrolment:
    """Draw k execution-disjoint supports per candidate from a *different* stream.

    ``same_subject`` selects the provenance rung: ``True`` keeps only supports recorded from the
    same person as the query (Scenario 2a), ``False`` only from a different person (2b and every
    cross-dataset cell), and ``None`` accepts either.

    ``allow_same_execution`` stays ``False`` by default even across streams.  For simultaneously
    recorded placements the *same* physical execution appears in both streams, so permitting it
    would let a query be enrolled against another view of its own moment; that is a different and
    much easier question than the one this scenario asks.
    """
    if k < 1:
        raise ValueError("cross-stream enrolment needs at least one support per candidate")
    if not bool(getattr(query_stream, "execution_identity_known", False)) or not bool(
            getattr(support_stream, "execution_identity_known", False)):
        raise ValueError("cross-stream enrolment requires known execution identity on both streams")
    if same_subject is not None and query_stream.dataset != support_stream.dataset:
        raise ValueError("subject relations across datasets require an explicit identity map")
    roster, label_map = shared_candidates(query_stream, support_stream)
    if candidates is not None:
        requested = tuple(str(label) for label in candidates)
        missing = set(requested) - set(roster)
        if missing:
            raise ValueError(f"candidates unavailable in both streams: {sorted(missing)}")
        roster = requested
    if len(roster) < 2:
        raise ValueError(
            f"{query_stream.dataset}/{getattr(query_stream, 'stream', '')} and "
            f"{support_stream.dataset}/{getattr(support_stream, 'stream', '')} share "
            f"{len(roster)} label(s); a candidate roster needs at least two"
        )

    # Both sides must be aligned to their own stream's registered vocabulary before the concept map
    # is applied. A grid's raw ``gt`` often uses a source spelling ("push_up") that the stream's
    # ``eval_labels`` canonicalises ("pushups"); comparing a raw support label against a roster built
    # from aligned labels silently matches nothing and yields an empty, unexplained manifest.
    query_labels = _aligned_labels(query_stream)
    support_aligned = _aligned_labels(support_stream)
    support_mapped = np.asarray(
        [None if value is None else label_map.get(str(value)) for value in support_aligned],
        dtype=object)

    support_exec = (np.asarray(support_stream.execution_ids, dtype=object)
                    if support_stream.execution_ids is not None else None)
    query_exec = (np.asarray(query_stream.execution_ids, dtype=object)
                  if query_stream.execution_ids is not None else None)
    if not allow_same_execution and (support_exec is None or query_exec is None):
        raise ValueError("execution identity is required to exclude a query's own execution")
    support_subjects = np.asarray(support_stream.subjects)
    query_subjects = np.asarray(query_stream.subjects)

    allowed_support = (np.arange(stream_rows(support_stream), dtype=np.int64) if support_rows is None
                       else np.asarray(support_rows, dtype=np.int64))
    allowed_query = (np.arange(stream_rows(query_stream), dtype=np.int64) if query_rows is None
                     else np.asarray(query_rows, dtype=np.int64))
    for name, rows, stream in (("query", allowed_query, query_stream),
                               ("support", allowed_support, support_stream)):
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= stream_rows(stream)):
            raise ValueError(f"{name} rows are outside the stream")
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"{name} rows contain duplicates")
    rows_by_label = {label: allowed_support[support_mapped[allowed_support] == label] for label in roster}
    offset = stream_rows(query_stream)
    plans: list[QueryPlan] = []
    # A query whose ground truth falls outside the shared roster cannot be answered by any model and
    # would silently depress every score by the same amount. Exclude it and report the exclusion.
    answerable = np.asarray([label in set(roster) for label in query_labels], dtype=bool)
    eligible_query = allowed_query[(query_labels[allowed_query] != None) & answerable[allowed_query]]  # noqa: E711
    for query in eligible_query.tolist():
        support: list[int] = []
        labels: list[str] = []
        possible = True
        for label in roster:
            pool = rows_by_label[label]
            if same_subject is not None and len(pool):
                same = support_subjects[pool] == query_subjects[query]
                pool = pool[same if same_subject else ~same]
            # Execution identifiers identify rows only within one dataset. Across independent
            # corpora equal local strings do not imply a shared physical capture.
            if not allow_same_execution and query_stream.dataset == support_stream.dataset and len(pool):
                pool = pool[support_exec[pool] != query_exec[query]]
            if len(pool) < k:
                possible = False
                break
            picked = _stable_choice(pool, k, seed_parts=(
                seed, "cross", query_stream.dataset, getattr(query_stream, "stream", ""),
                support_stream.dataset, getattr(support_stream, "stream", ""),
                relation, k, query, label,
            ))
            support.extend(offset + int(value) for value in picked)
            labels.extend([label] * k)
        if possible:
            plans.append(QueryPlan(query=int(query), support=tuple(support),
                                   support_labels=tuple(labels)))
    return CrossEnrolment(
        query_stream=query_stream, support_stream=support_stream, plans=tuple(plans),
        candidates=tuple(roster), offset=offset, relation=relation,
        label_map=tuple(sorted(label_map.items())),
    )
