"""Support-set sampling — the training curriculum, which is the paper's contribution.

WHAT AN EPISODE IS
------------------
One query recording, a candidate label roster, and K labelled *support executions* the comparator
may compare the query against. Everything the model learns about "how to compare" comes from how
these are drawn, so this module is the method rather than plumbing around it.

THE FOUR RULES
--------------
1. **Compatibility.** In ``"compatible"`` mode every support recording shares the query's exact
   acquisition key. This is a deployment filter, not a learned quantity, and no novelty is claimed
   for it — you would not offer smartwatch examples to a pocket-phone query in a real product.
2. **Never the query or its execution.** Few-shot episodes deliberately mix same-subject and
   cross-subject enrollment, but a support execution is always physically distinct from the query.
   Each support execution contributes one row, so long recordings do not receive more voting mass.
3. **Verbatim labels.** No canonicalisation beyond what the corpus already applied, no synonym
   merging, no deduplication. Two candidates may carry near-identical text; the readout handles
   that by giving them near-identical votes, which is the right answer.
4. **Ground-truth support present with probability p.** The answer is always in the candidate
   roster. In a few-shot episode every candidate has enrolled support. In a zero-shot episode none
   of the candidate labels has support; compatible rows with other labels are unbound background,
   exactly like deployed k=0 evaluation.

WHAT HAPPENS WHEN THE POOL IS TOO SMALL
---------------------------------------
K shrinks for that episode and the shrink is recorded in telemetry. Support is never padded with
incompatible rows (that would silently violate rule 1). A query that cannot form a two-candidate
decision is redrawn and counted as unusable; a requested batch that still cannot be filled after
the bounded retry budget fails loudly instead of changing the effective batch size. If shrinking or
redrawing is common, that is a finding about the corpus to report, not a bug to hide.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

from data.scripts.labels.canonical_labels import canonicalize
from data.scripts.curate.compatibility import (
    AcquisitionKey,
    is_near_miss,
    stream_key,
)
from data.scripts.eda.grid_io import discover_grids

REPO = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO / "data" / "datasets"

SamplingMode = Literal["compatible", "near_miss", "unfiltered"]
SubjectRelation = Literal["same_subject", "cross_subject"]
SupportUnit = tuple[str, str, str]  # dataset, subject, physical execution

#: A-priori constants (design doc §3). They may be varied deliberately as an experiment; they are
#: never tuned against evaluation data, because there is no development split.
DEFAULT_SUPPORT = 32
DEFAULT_P_GT_PRESENT = 0.5
DEFAULT_LABEL_SUBSET = (2, 14)
DEFAULT_SAME_SUBJECT_PROBABILITY = 0.5
MIN_RECORDING_SECONDS = 1.0


def _recording_map(dataset: str) -> dict:
    """``{event_id_without_ordinal: recording_id}``, composed exactly as ``eval/data.py`` does.

    Duplicated deliberately rather than imported: ``eval.data`` pulls in the whole evaluation
    stack, and the two paths must agree on the leakage unit even if one is refactored. The
    agreement is asserted in ``tests/test_compare_sampling.py``.
    """
    recordings_path = DATASETS_DIR / dataset / "recordings.json"
    if not recordings_path.exists():
        return {}
    recordings = json.loads(recordings_path.read_text())
    events_path = DATASETS_DIR / dataset / "events.json"
    events = json.loads(events_path.read_text()) if events_path.exists() else {}
    composed: dict = {}
    for session, recording in recordings.items():
        key = f"{dataset}:{events.get(session, session)}"
        previous = composed.setdefault(key, recording)
        if previous != recording:
            raise ValueError(
                f"{dataset}: sessions sharing physical event {key!r} disagree on their recording"
            )
    return composed


def _execution_ids(dataset: str, event_ids: Sequence[str]) -> np.ndarray:
    """The leakage unit for each window: one continuous physical capture."""
    blocks = np.asarray([
        value.rsplit(":", 1)[0]
        if ":" in str(value) and str(value).rsplit(":", 1)[1].isdigit() else value
        for value in event_ids
    ], dtype=object)
    recordings = _recording_map(dataset)
    if not recordings:
        return blocks
    return np.asarray([recordings.get(block, block) for block in blocks], dtype=object)


@dataclass(frozen=True)
class Recording:
    """One window of the corpus, addressed the way an episode needs it."""

    stream_index: int
    window_index: int
    dataset: str
    stream: str
    label: str
    subject: str
    execution: str


@dataclass
class SupportCorpus:
    """Windows indexed for balanced queries and execution-level support sampling."""

    recordings: list[Recording]
    keys: list[AcquisitionKey]                      # per stream index
    stream_names: list[tuple[str, str]]             # per stream index
    by_key: dict[AcquisitionKey, list[int]] = field(default_factory=dict)
    by_key_label: dict[tuple[AcquisitionKey, str], list[int]] = field(default_factory=dict)
    near_miss_keys: dict[AcquisitionKey, list[AcquisitionKey]] = field(default_factory=dict)
    by_key_label_unit: dict[
        tuple[AcquisitionKey, str], dict[SupportUnit, list[int]]
    ] = field(default_factory=dict)
    query_by_dataset_label: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    query_labels_by_dataset: dict[str, tuple[str, ...]] = field(default_factory=dict)
    all_labels: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.recordings)

    def key_of(self, recording: Recording) -> AcquisitionKey:
        return self.keys[recording.stream_index]

    def ensure_indexes(self) -> None:
        """Build derived lookup tables once; synthetic tests may populate the basic tables later."""
        if self.query_by_dataset_label:
            return
        if not self.by_key:
            for index, recording in enumerate(self.recordings):
                key = self.key_of(recording)
                self.by_key.setdefault(key, []).append(index)
                self.by_key_label.setdefault((key, recording.label), []).append(index)
        if not self.near_miss_keys:
            distinct = list(self.by_key)
            for key in distinct:
                self.near_miss_keys[key] = [other for other in distinct if is_near_miss(key, other)]
        labels_by_dataset: dict[str, set[str]] = defaultdict(set)
        labels: set[str] = set()
        for index, recording in enumerate(self.recordings):
            key = self.key_of(recording)
            unit = (recording.dataset, recording.subject, recording.execution)
            self.by_key_label_unit.setdefault((key, recording.label), {}).setdefault(
                unit, []
            ).append(index)
            self.query_by_dataset_label.setdefault(
                (recording.dataset, recording.label), []
            ).append(index)
            labels_by_dataset[recording.dataset].add(recording.label)
            labels.add(recording.label)
        self.query_labels_by_dataset = {
            dataset: tuple(sorted(dataset_labels))
            for dataset, dataset_labels in labels_by_dataset.items()
        }
        self.all_labels = tuple(sorted(labels))

    def summary(self) -> dict[str, object]:
        pools = {key: len(rows) for key, rows in self.by_key.items()}
        return {
            "windows": len(self.recordings),
            "streams": len(self.stream_names),
            "keys": len(pools),
            "largest_pool": max(pools.values()) if pools else 0,
            "median_pool": int(np.median(list(pools.values()))) if pools else 0,
            "keys_with_near_miss": sum(1 for v in self.near_miss_keys.values() if v),
            "datasets": len(self.query_labels_by_dataset),
            "labels": len(self.all_labels),
        }


def build_support_corpus(
    datasets: Sequence[str],
    *,
    alignment: str = "native",
    max_per_stream: int | None = None,
    seed: int = 0,
    exclude_labels: Iterable[str] = ("unlabeled",),
    min_duration_seconds: float = MIN_RECORDING_SECONDS,
) -> SupportCorpus:
    """Index the training grids into the structure the sampler draws from.

    Reads grid metadata only — never ``data.npy`` — so building the index is cheap and the encoder
    stays responsible for loading signal.
    """

    rng = np.random.default_rng(seed)
    banned = {str(label).lower() for label in exclude_labels}
    wanted = set(datasets)

    recordings: list[Recording] = []
    keys: list[AcquisitionKey] = []
    stream_names: list[tuple[str, str]] = []

    for ref in discover_grids(alignment):
        if ref.dataset not in wanted or ref.n_windows == 0:
            continue
        try:
            key = stream_key(ref.dataset, ref.stream)
        except KeyError:
            # A stream with no curated spec cannot be placed in a configuration, so it cannot
            # legitimately enter anyone's support set. Skip loudly rather than guess.
            print(f"[support-corpus] no acquisition key for {ref.dataset}/{ref.stream}; skipped")
            continue
        stream_index = len(stream_names)
        stream_names.append((ref.dataset, ref.stream))
        keys.append(key)

        executions = _execution_ids(ref.dataset, ref.event_ids)
        lengths = ref.load_lengths()
        chosen = np.arange(ref.n_windows)
        if max_per_stream is not None and ref.n_windows > max_per_stream:
            chosen = np.sort(rng.choice(ref.n_windows, size=max_per_stream, replace=False))
        for window in chosen:
            if float(lengths[int(window)]) / ref.rate_hz < min_duration_seconds:
                continue
            label = canonicalize(ref.labels[int(window)])
            if str(label).lower() in banned:
                continue
            recordings.append(Recording(
                stream_index=stream_index,
                window_index=int(window),
                dataset=ref.dataset,
                stream=ref.stream,
                label=label,
                subject=str(ref.subjects[int(window)]),
                execution=str(executions[int(window)]),
            ))

    corpus = SupportCorpus(recordings=recordings, keys=keys, stream_names=stream_names)
    for index, recording in enumerate(recordings):
        key = keys[recording.stream_index]
        corpus.by_key.setdefault(key, []).append(index)
        corpus.by_key_label.setdefault((key, recording.label), []).append(index)
    distinct = list(corpus.by_key)
    for key in distinct:
        corpus.near_miss_keys[key] = [
            other for other in distinct if is_near_miss(key, other)
        ]
    corpus.ensure_indexes()
    return corpus


@dataclass(frozen=True)
class Episode:
    """One training episode. Indices address ``SupportCorpus.recordings``."""

    query: int
    support: tuple[int, ...]             # one sampled window from each distinct execution
    support_candidate: tuple[int, ...]   # candidate slot, or -1 for zero-shot background
    candidates: tuple[str, ...]          # verbatim label strings
    gt_slot: int                         # the answer is always in the candidate roster
    mode: SamplingMode
    requested_support: int
    shrunk: bool
    zero_shot: bool = False
    subject_relation: SubjectRelation = "cross_subject"

    @property
    def is_zero_shot(self) -> bool:
        return self.zero_shot


def _keys_for(corpus: SupportCorpus, key: AcquisitionKey, mode: SamplingMode) -> list[AcquisitionKey]:
    if mode == "compatible":
        return [key] if key in corpus.by_key else []
    if mode == "near_miss":
        return list(corpus.near_miss_keys.get(key, []))
    if mode == "unfiltered":
        return list(corpus.by_key)
    raise ValueError(f"unknown sampling mode {mode!r}")


def _choose_query(corpus: SupportCorpus, rng: np.random.Generator) -> int:
    """Dataset-first, label-second sampling prevents large sources/classes dominating queries."""
    corpus.ensure_indexes()
    datasets = tuple(sorted(corpus.query_labels_by_dataset))
    dataset = str(rng.choice(datasets))
    label = str(rng.choice(corpus.query_labels_by_dataset[dataset]))
    rows = corpus.query_by_dataset_label[(dataset, label)]
    return int(rows[int(rng.integers(len(rows)))])


def balanced_query_indices(
    corpus: SupportCorpus, rng: np.random.Generator, count: int,
) -> list[int]:
    """Draw query indices with the same dataset/label hierarchy used by training episodes."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    return [_choose_query(corpus, rng) for _ in range(count)]


def _available_units(
    corpus: SupportCorpus,
    keys: Sequence[AcquisitionKey],
    query: Recording,
    relation: SubjectRelation,
) -> dict[str, dict[SupportUnit, list[int]]]:
    """Execution groups available under one subject relation, without scanning corpus windows."""
    query_subject = (query.dataset, query.subject)
    query_execution: SupportUnit = (query.dataset, query.subject, query.execution)
    output: dict[str, dict[SupportUnit, list[int]]] = defaultdict(dict)
    for key in keys:
        for label in corpus.all_labels:
            available = {
                unit: rows
                for unit, rows in corpus.by_key_label_unit.get((key, label), {}).items()
                if unit != query_execution
                and (relation != "same_subject" or unit[:2] == query_subject)
                and (relation != "cross_subject" or unit[:2] != query_subject)
            }
            if not available:
                continue
            if label not in output:
                output[label] = available
            else:
                # Rows are read-only during drawing. Copy only executions spanning keys;
                # never extend a list owned by the corpus index.
                merged = output[label]
                for unit, rows in available.items():
                    merged[unit] = merged[unit] + rows if unit in merged else rows
    return output


def _draw_support(
    units_by_label: dict[str, dict[SupportUnit, list[int]]],
    labels: Sequence[str],
    rng: np.random.Generator,
    support_size: int,
    candidate_slots: dict[str, int] | None,
) -> tuple[list[int], list[int]]:
    """Draw at most one row per physical execution, balanced across the selected labels."""
    remaining = {label: list(units_by_label[label]) for label in labels}
    for units in remaining.values():
        rng.shuffle(units)
    used: set[SupportUnit] = set()
    support: list[int] = []
    bound: list[int] = []

    if candidate_slots is not None:
        # A few-shot episode promises at least one enrolled execution for every candidate. Labels
        # can share one physical recording in continuously annotated datasets, so a greedy draw can
        # starve a later label even when a valid distinct-execution assignment exists. Find that
        # assignment first with a small bipartite matching (C <= 14), then fill the remaining budget.
        owner: dict[SupportUnit, str] = {}
        assigned: dict[str, SupportUnit] = {}

        def assign(label: str, seen: set[SupportUnit]) -> bool:
            for unit in remaining[label]:
                if unit in seen:
                    continue
                seen.add(unit)
                previous = owner.get(unit)
                if previous is None or assign(previous, seen):
                    owner[unit] = label
                    assigned[label] = unit
                    return True
            return False

        # Scarce labels first reduces needless rematching; random tie-breaking preserves stochastic
        # episode views without changing feasibility.
        label_order = list(labels)
        rng.shuffle(label_order)
        label_order.sort(key=lambda label: len(remaining[label]))
        if support_size < len(labels) or any(not assign(label, set()) for label in label_order):
            return [], []
        for label in labels:
            unit = assigned[label]
            rows = units_by_label[label][unit]
            support.append(int(rows[int(rng.integers(len(rows)))]))
            bound.append(candidate_slots[label])
            used.add(unit)
        for label in labels:
            remaining[label] = [unit for unit in remaining[label] if unit not in used]

    while len(support) < support_size:
        progressed = False
        for label in labels:
            units = remaining[label]
            while units and units[-1] in used:
                units.pop()
            if not units or len(support) >= support_size:
                continue
            unit = units.pop()
            rows = units_by_label[label][unit]
            support.append(int(rows[int(rng.integers(len(rows)))]))
            bound.append(-1 if candidate_slots is None else candidate_slots[label])
            used.add(unit)
            progressed = True
        if not progressed:
            break
    return support, bound


def draw_episode(
    corpus: SupportCorpus,
    rng: np.random.Generator,
    *,
    support_size: int = DEFAULT_SUPPORT,
    p_gt_present: float = DEFAULT_P_GT_PRESENT,
    label_subset: tuple[int, int] = DEFAULT_LABEL_SUBSET,
    mode: SamplingMode = "compatible",
    query_index: int | None = None,
    same_subject_probability: float = DEFAULT_SAME_SUBJECT_PROBABILITY,
) -> Episode | None:
    """Draw one episode, or ``None`` when the query admits no usable support at all.

    ``None`` is returned only when the pool cannot supply a single admissible row — an isolated
    configuration. The caller counts these; they are a corpus finding, not an error to swallow.
    """

    if not 0.0 <= p_gt_present <= 1.0:
        raise ValueError("p_gt_present must be in [0, 1]")
    if not 0.0 <= same_subject_probability <= 1.0:
        raise ValueError("same_subject_probability must be in [0, 1]")
    low, high = label_subset
    if low < 2 or high < low:
        raise ValueError("label_subset must be (low >= 2, high >= low)")

    corpus.ensure_indexes()
    query_index = _choose_query(corpus, rng) if query_index is None else int(query_index)
    query = corpus.recordings[query_index]
    key = corpus.key_of(query)
    keys = _keys_for(corpus, key, mode)
    if not keys:
        return None

    want_gt = bool(rng.random() < p_gt_present)
    cross = _available_units(corpus, keys, query, "cross_subject")
    same = _available_units(corpus, keys, query, "same_subject")
    if want_gt:
        feasible: list[tuple[SubjectRelation, dict[str, dict[SupportUnit, list[int]]]]] = []
        if query.label in cross and len(cross) >= 2:
            feasible.append(("cross_subject", cross))
        if query.label in same and len(same) >= 2:
            feasible.append(("same_subject", same))
        if not feasible:
            return None
        if len(feasible) == 2:
            selected = 1 if rng.random() < same_subject_probability else 0
            relation, available_units = feasible[selected]
        else:
            relation, available_units = feasible[0]
    else:
        # Deployed k=0 draws from the training corpus and therefore has no same-user enrollment.
        relation, available_units = "cross_subject", cross
        if not available_units:
            return None

    available = sorted(available_units)
    n_labels = int(rng.integers(low, high + 1))
    n_labels = min(n_labels, len(corpus.all_labels))
    if want_gt:
        n_labels = min(n_labels, support_size)
    if n_labels < 2:
        return None

    if want_gt:
        others = [label for label in available if label != query.label]
        take = min(n_labels - 1, len(others))
        if take < 1:
            return None
        picked = list(rng.choice(others, size=take, replace=False))
        picked.append(query.label)
    else:
        # All candidates must be plausible under the same support availability
        # rule. Otherwise the answer alone reveals the configuration's vocabulary.
        # Keep at least one other label for non-candidate background support.
        if query.label not in available_units or len(available) < 3:
            return None
        n_labels = min(n_labels, len(available) - 1)
        others = [label for label in available if label != query.label]
        picked = [query.label, *rng.choice(others, size=n_labels - 1, replace=False)]
    rng.shuffle(picked)
    candidates = tuple(str(label) for label in picked)
    gt_slot = candidates.index(query.label)

    if want_gt:
        support_labels = [label for label in candidates if label in available_units]
        candidate_slots = {label: candidates.index(label) for label in support_labels}
    else:
        support_labels = [label for label in available if label not in candidates]
        if len(support_labels) > high:
            support_labels = list(rng.choice(support_labels, size=high, replace=False))
        candidate_slots = None
    if not support_labels:
        return None
    support, support_candidate = _draw_support(
        available_units, support_labels, rng, support_size, candidate_slots,
    )

    if not support:
        return None
    return Episode(
        query=query_index,
        support=tuple(support),
        support_candidate=tuple(support_candidate),
        candidates=candidates,
        gt_slot=gt_slot,
        mode=mode,
        requested_support=int(support_size),
        shrunk=len(support) < support_size,
        zero_shot=not want_gt,
        subject_relation=relation,
    )


def draw_batch(
    corpus: SupportCorpus,
    rng: np.random.Generator,
    *,
    batch_size: int,
    max_attempts_per_episode: int = 32,
    **kwargs,
) -> tuple[list[Episode], dict[str, float]]:
    """Draw ``batch_size`` episodes plus the telemetry that makes the draw auditable."""

    episodes: list[Episode] = []
    attempts = 0
    unusable = 0
    while len(episodes) < batch_size and attempts < batch_size * max_attempts_per_episode:
        attempts += 1
        episode = draw_episode(corpus, rng, **kwargs)
        if episode is None:
            unusable += 1
            continue
        episodes.append(episode)
    if len(episodes) != batch_size:
        raise RuntimeError(
            f"only drew {len(episodes)} of {batch_size} requested episodes in {attempts} "
            f"attempts under mode {kwargs.get('mode', 'compatible')!r}; refusing to train "
            "with a silently smaller batch"
        )
    zero_shot = sum(1 for episode in episodes if episode.is_zero_shot)
    query_datasets = [corpus.recordings[episode.query].dataset for episode in episodes]
    query_labels = [corpus.recordings[episode.query].label for episode in episodes]
    dataset_counts = {value: query_datasets.count(value) for value in set(query_datasets)}
    duplicate_executions = []
    for episode in episodes:
        units = {
            (corpus.recordings[index].dataset, corpus.recordings[index].subject,
             corpus.recordings[index].execution)
            for index in episode.support
        }
        duplicate_executions.append(len(episode.support) - len(units))
    telemetry = {
        "sampler/realised_gt_rate": 1.0 - zero_shot / len(episodes),
        "sampler/zero_shot_rate": zero_shot / len(episodes),
        "sampler/same_subject_rate": sum(
            episode.subject_relation == "same_subject" for episode in episodes
            if not episode.is_zero_shot
        ) / max(1, len(episodes) - zero_shot),
        "sampler/shrunk_episode_fraction": sum(e.shrunk for e in episodes) / len(episodes),
        "sampler/mean_support_size": float(np.mean([len(e.support) for e in episodes])),
        "sampler/mean_candidate_count": float(np.mean([len(e.candidates) for e in episodes])),
        "sampler/unusable_query_fraction": unusable / max(attempts, 1),
        "sampler/query_dataset_count": float(len(set(query_datasets))),
        "sampler/query_label_count": float(len(set(query_labels))),
        "sampler/max_query_dataset_share": max(dataset_counts.values()) / len(episodes),
        "sampler/duplicate_support_execution_mean": float(np.mean(duplicate_executions)),
    }
    return episodes, telemetry
