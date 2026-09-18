"""Support-set sampling for the retained support-classification control.

WHAT AN EPISODE IS
------------------
One query recording, a candidate label roster, and K labelled *support executions* the classifier
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
   roster. In a few-shot episode every candidate has enrolled support. In a zero-shot episode the
   classifier receives only the query and declared candidate labels; there is no background bank.

SUPPORT-SET DRAWING
-------------------
The sampler draws exact k-shot support for every enrolled candidate and may reuse that support set
across several distinct query executions. Multiple windows from one physical execution may be
averaged, but an execution is never duplicated to manufacture k. Sparse configurations use only
feasible k values; candidate/query shortfalls and requested-subject-relation fallbacks are exposed
in telemetry. Compatibility is never relaxed. If a complete support set cannot be formed within the
bounded retry budget, drawing fails loudly rather than silently changing the task.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal, Sequence

import numpy as np

from data.scripts.labels.canonical_labels import NON_SEMANTIC_LABELS, canonicalize
from data.scripts.curate.compatibility import (
    AcquisitionKey,
    is_near_miss,
    site_group,
    stream_key,
)
from data.scripts.eda.grid_io import discover_grids
from halo.paths import DATASETS_DIR

SamplingMode = Literal[
    "compatible", "near_miss", "cross_placement", "cross_dataset", "unfiltered",
]
EnrollmentRegime = Literal["complete", "partial", "zero"]
SubjectRelation = Literal["same_subject", "cross_subject"]
SupportUnit = tuple[str, str, str]  # dataset, subject, physical execution

#: Protocol defaults (design doc §3). They may be varied deliberately using only the supervised
#: training sources and their internal subject-held-out fold; sealed sources never tune them.
DEFAULT_SUPPORT = 32
DEFAULT_P_GT_PRESENT = 0.5
# Candidate rosters should be large enough that the support task cannot devolve into a binary
# decision.  Small configurations still use every feasible label rather than being excluded.
DEFAULT_LABEL_SUBSET = (2, 32)
LARGE_C_MIN = 16
DEFAULT_SAME_SUBJECT_PROBABILITY = 0.5
DEFAULT_ENROLLMENT_K = (1, 2, 4, 8, 16, 32)
DEFAULT_QUERIES_PER_SUPPORT_SET = 4
DEFAULT_WINDOWS_PER_EXECUTION = 2
DEFAULT_ACQUISITION_MIX = (0.50, 0.25, 0.25)  # compatible, cross-placement, cross-dataset
DEPLOYMENT_SAMPLER_SCHEMA = "independent-enrollment-acquisition-v2-20260918"
DEFAULT_ENROLLMENT_MIX = (0.50, 0.25, 0.25)   # complete, partial, zero
DEFAULT_PARTIAL_COVERAGE = (0.25, 0.75)
DEFAULT_VARIABLE_SUPPORT_PROBABILITY = 0.50
MIN_RECORDING_SECONDS = 1.0


def _recording_map(dataset: str) -> dict:
    """``{event_id_without_ordinal: recording_id}``, composed exactly as ``baselines/data.py`` does.

    Duplicated deliberately rather than imported: ``baselines.data`` pulls in the whole baseline
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
    labels_by_key: dict[AcquisitionKey, tuple[str, ...]] = field(default_factory=dict)
    query_by_dataset_label: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    query_labels_by_dataset: dict[str, tuple[str, ...]] = field(default_factory=dict)
    all_labels: tuple[str, ...] = ()
    deployment_dataset_cache: dict[tuple, tuple[str, ...]] = field(
        default_factory=dict, repr=False,
    )
    deployment_label_cache: dict[tuple, tuple[str, ...]] = field(
        default_factory=dict, repr=False,
    )

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
        labels_by_key: dict[AcquisitionKey, set[str]] = defaultdict(set)
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
            labels_by_key[key].add(recording.label)
            labels.add(recording.label)
        self.query_labels_by_dataset = {
            dataset: tuple(sorted(dataset_labels))
            for dataset, dataset_labels in labels_by_dataset.items()
        }
        self.all_labels = tuple(sorted(labels))
        self.labels_by_key = {
            key: tuple(sorted(key_labels)) for key, key_labels in labels_by_key.items()
        }

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
    exclude_labels: Iterable[str] = NON_SEMANTIC_LABELS,
    min_duration_seconds: float = MIN_RECORDING_SECONDS,
) -> SupportCorpus:
    """Index the training grids into the structure the sampler draws from.

    Reads grid metadata only — never ``data.npy`` — so building the index is cheap and the encoder
    stays responsible for loading signal. The same fingerprinted implausible/duplicate-window
    exclusions used by :class:`CorpusIndex` are mandatory here; an offline evaluator must not
    silently score rows that training rejects as invalid observations.
    """

    rng = np.random.default_rng(seed)
    banned = {str(label).lower() for label in exclude_labels}
    wanted = set(datasets)
    from data.scripts.scan_implausible import load as load_implausible
    from data.scripts.scan_duplicates import load as load_duplicates
    implausible = load_implausible(alignment, require=True)
    duplicates = load_duplicates(alignment, require=True)

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
        excluded = implausible.get(ref.key, set()) | duplicates.get(ref.key, set())
        chosen = np.arange(ref.n_windows)
        if max_per_stream is not None and ref.n_windows > max_per_stream:
            chosen = np.sort(rng.choice(ref.n_windows, size=max_per_stream, replace=False))
        for window in chosen:
            if int(window) in excluded:
                continue
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
    support_candidate: tuple[int, ...]   # candidate slot; empty for zero-shot
    candidates: tuple[str, ...]          # verbatim label strings
    gt_slot: int                         # the answer is always in the candidate roster
    mode: SamplingMode
    requested_support: int
    shrunk: bool
    zero_shot: bool = False
    subject_relation: SubjectRelation = "cross_subject"
    # Optional deployment-matched representation. ``support`` remains one representative index
    # per execution for compatibility with audit/evaluation helpers; training averages the rows in
    # the corresponding group before presenting one vector to the head.
    support_window_groups: tuple[tuple[int, ...], ...] = ()
    support_per_candidate: int = 0
    support_set_id: int = -1
    requested_candidates: int = 0
    subject_relation_fallback: bool = False
    # Candidate slots whose otherwise valid enrolled rows were deliberately withheld. This is
    # training-only supervision for the text-only portion of the unified residual classifier.
    masked_candidates: tuple[int, ...] = ()
    support_counts: tuple[int, ...] = ()
    acquisition_regime: str = "compatible"
    enrollment_regime: str = "complete"
    curriculum_fallback: bool = False

    @property
    def is_zero_shot(self) -> bool:
        return self.zero_shot


def _keys_for(corpus: SupportCorpus, key: AcquisitionKey, mode: SamplingMode) -> list[AcquisitionKey]:
    if mode == "compatible":
        return [key] if key in corpus.by_key else []
    if mode == "near_miss":
        return list(corpus.near_miss_keys.get(key, []))
    if mode == "cross_placement":
        return [
            other for other in corpus.by_key
            if other.device_family == key.device_family
            and other.channels == key.channels
            and other.gravity_state == key.gravity_state
            # Laterality or an unspecified side is a near-miss within one anatomical site, not a
            # deployment placement shift. Cross-placement means a genuinely different site.
            and site_group(other.site) != site_group(key.site)
        ]
    if mode == "cross_dataset":
        return [key] if key in corpus.by_key else []
    if mode == "unfiltered":
        return list(corpus.by_key)
    raise ValueError(f"unknown sampling mode {mode!r}")


def _choose_query(
    corpus: SupportCorpus,
    rng: np.random.Generator,
    dataset: str | None = None,
    label: str | None = None,
) -> int:
    """Dataset-first, label-second sampling prevents large sources/classes dominating queries."""
    corpus.ensure_indexes()
    datasets = tuple(sorted(corpus.query_labels_by_dataset))
    dataset = str(rng.choice(datasets)) if dataset is None else str(dataset)
    if dataset not in corpus.query_labels_by_dataset:
        raise KeyError(f"unknown query dataset {dataset!r}")
    label = str(rng.choice(corpus.query_labels_by_dataset[dataset])) if label is None else str(label)
    if label not in corpus.query_labels_by_dataset[dataset]:
        raise KeyError(f"unknown query label {label!r} for dataset {dataset!r}")
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
    *,
    different_dataset: bool = False,
    same_dataset: bool = False,
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
                and (not different_dataset or unit[0] != query.dataset)
                and (not same_dataset or unit[0] == query.dataset)
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


def _draw_k_per_candidate(
    units_by_label: dict[str, dict[SupportUnit, list[int]]],
    candidates: Sequence[str],
    rng: np.random.Generator,
    k: int,
    windows_per_execution: int,
) -> tuple[list[int], list[int], list[tuple[int, ...]]]:
    """Draw exactly k globally distinct executions per candidate or fail honestly.

    The augmenting-path assignment matters for continuously annotated recordings: one physical
    execution may contain windows carrying several labels and may still be used only once.
    """
    if k < 1 or windows_per_execution < 1:
        raise ValueError("k and windows_per_execution must be positive")
    demands = [(label, occurrence) for label in candidates for occurrence in range(k)]
    choices = {label: list(units_by_label.get(label, ())) for label in candidates}
    for values in choices.values():
        rng.shuffle(values)
    if any(len(choices[label]) < k for label in candidates):
        return [], [], []

    owner: dict[SupportUnit, tuple[str, int]] = {}
    assigned: dict[tuple[str, int], SupportUnit] = {}

    def assign(demand: tuple[str, int], seen: set[SupportUnit]) -> bool:
        label, _ = demand
        for unit in choices[label]:
            if unit in seen:
                continue
            seen.add(unit)
            previous = owner.get(unit)
            if previous is None or assign(previous, seen):
                owner[unit] = demand
                assigned[demand] = unit
                return True
        return False

    # Scarce labels first; randomized choices preserve stochastic windows and subject mixtures.
    rng.shuffle(demands)
    demands.sort(key=lambda item: len(choices[item[0]]))
    if any(not assign(demand, set()) for demand in demands):
        return [], [], []

    support: list[int] = []
    slots: list[int] = []
    groups: list[tuple[int, ...]] = []
    for slot, label in enumerate(candidates):
        for occurrence in range(k):
            rows = units_by_label[label][assigned[(label, occurrence)]]
            count = min(windows_per_execution, len(rows))
            selected = np.asarray(rng.choice(rows, size=count, replace=False), dtype=np.int64)
            group = tuple(int(value) for value in selected)
            support.append(group[0])
            slots.append(slot)
            groups.append(group)
    return support, slots, groups


def _draw_feasible_candidate_roster(
    units_by_label: dict[str, dict[SupportUnit, list[int]]],
    *,
    query_label: str,
    available_labels: Sequence[str],
    requested_labels: int,
    rng: np.random.Generator,
    k: int,
    windows_per_execution: int,
) -> tuple[tuple[str, ...], list[int], list[int], list[tuple[int, ...]]]:
    """Build a random candidate roster that has a globally distinct support assignment.

    Continuously annotated recordings can expose many labels while all of those labels share only a
    few physical executions. Per-label counts therefore do not establish that the complete roster is
    drawable. Grow a feasible roster one randomly ordered label at a time and retain the matching
    produced for the final roster. A smaller roster is reported by the caller; executions are never
    duplicated merely to satisfy the requested candidate count.
    """
    if query_label not in available_labels:
        return (), [], [], []
    target = min(int(requested_labels), len(available_labels))
    others = [label for label in available_labels if label != query_label]
    rng.shuffle(others)
    selected = [query_label]
    for label in others:
        trial = [*selected, label]
        support, slots, groups = _draw_k_per_candidate(
            units_by_label, trial, rng, k, windows_per_execution,
        )
        if not support:
            continue
        selected = trial
        if len(selected) == target:
            break
    if len(selected) < 2:
        return (), [], [], []

    # Randomize candidate slot order after feasibility has been established. Re-solving cannot alter
    # feasibility and ensures the ground-truth slot is not privileged by construction.
    rng.shuffle(selected)
    support, slots, groups = _draw_k_per_candidate(
        units_by_label, selected, rng, k, windows_per_execution,
    )
    if not support:  # Defensive: bipartite feasibility is invariant to candidate ordering.
        return (), [], [], []
    return tuple(str(label) for label in selected), support, slots, groups


def _large_candidate_count(
    rng: np.random.Generator,
    *,
    available: int,
    label_subset: tuple[int, int],
) -> int:
    """Draw from the upper half of the feasible candidate range.

    A small acquisition-specific vocabulary cannot honestly provide a large-C episode, so use all
    it can provide.  Larger vocabularies are trained on broadly sized rosters without requiring a
    second, contradictory candidate-count knob.
    """
    low, high = label_subset
    capacity = min(int(high), int(available))
    if capacity < low:
        return capacity
    floor = max(low, min(LARGE_C_MIN, capacity))
    return int(rng.integers(floor, capacity + 1))


def _can_draw_candidate_count(
    units_by_label: dict[str, dict[SupportUnit, list[int]]],
    *,
    query_label: str,
    available_labels: Sequence[str],
    count: int,
    k: int,
) -> bool:
    """Conservative feasibility probe for a minimum-size deployment candidate roster."""
    if len(available_labels) < count or query_label not in available_labels:
        return False
    # Roster selection is intentionally randomized. A few fixed orders avoid declaring a source
    # unusable because one greedy order selected a label that blocked a larger feasible matching.
    for seed in range(4):
        candidates, support, _, _ = _draw_feasible_candidate_roster(
            units_by_label, query_label=query_label, available_labels=available_labels,
            requested_labels=count, rng=np.random.default_rng(seed), k=k,
            windows_per_execution=1,
        )
        if len(candidates) == count and support:
            return True
    return False


def _additional_queries(
    corpus: SupportCorpus,
    rng: np.random.Generator,
    *,
    base_query: int,
    query_keys: Sequence[AcquisitionKey],
    candidates: Sequence[str],
    support: Sequence[int],
    relation: SubjectRelation,
    count: int,
) -> list[int]:
    """Choose distinct query executions that can use one support roster without leakage."""
    base = corpus.recordings[base_query]
    support_units = {
        (corpus.recordings[index].dataset, corpus.recordings[index].subject,
         corpus.recordings[index].execution) for index in support
    }
    support_subjects = {unit[:2] for unit in support_units}
    used_units = {(base.dataset, base.subject, base.execution)} | support_units
    by_label: dict[str, dict[str, list[tuple[SupportUnit, list[int]]]]] = {}
    for label in candidates:
        choices: dict[str, list[tuple[SupportUnit, list[int]]]] = defaultdict(list)
        for key in query_keys:
            for unit, rows in corpus.by_key_label_unit.get((key, label), {}).items():
                # The support roster may span compatible sources, but all queries assigned to one
                # source-balanced support set represent the same source. Otherwise the extra
                # queries silently undo the dataset-first draw and let highly reusable datasets
                # dominate the objective again.
                if unit[0] != base.dataset:
                    continue
                if unit in used_units:
                    continue
                if relation == "cross_subject" and unit[:2] in support_subjects:
                    continue
                if relation == "same_subject" and unit[:2] != (base.dataset, base.subject):
                    continue
                choices[unit[0]].append((unit, rows))
        for rows in choices.values():
            rng.shuffle(rows)
        by_label[label] = choices

    output = [base_query]
    labels = list(candidates)
    rng.shuffle(labels)
    while len(output) < count:
        progressed = False
        for label in labels:
            datasets = []
            for dataset, choices in by_label[label].items():
                while choices and choices[-1][0] in used_units:
                    choices.pop()
                if choices:
                    datasets.append(dataset)
            if not datasets or len(output) >= count:
                continue
            dataset = str(rng.choice(datasets))
            unit, rows = by_label[label][dataset].pop()
            output.append(int(rows[int(rng.integers(len(rows)))]))
            used_units.add(unit)
            progressed = True
        if not progressed:
            break
    return output


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
    semantic_zero_shot: bool = True,
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
    if not want_gt:
        # A direct semantic head needs no background bank. Draw only plausible candidates;
        # do not inherit the old bridge's requirement for a third, non-candidate activity.
        available = sorted({label for candidate_key in keys for label in corpus.all_labels
                            if (candidate_key, label) in corpus.by_key_label_unit} | {query.label})
        requested_labels = int(rng.integers(low, high + 1))
        n_labels = min(requested_labels, len(available))
        if n_labels < 2:
            return None
        others = [label for label in available if label != query.label]
        picked = [query.label, *rng.choice(others, n_labels - 1, replace=False)]
        rng.shuffle(picked)
        candidates = tuple(str(label) for label in picked)
        return Episode(query=query_index, support=(), support_candidate=(),
                       candidates=candidates, gt_slot=candidates.index(query.label), mode=mode,
                       requested_support=0, shrunk=False, zero_shot=True,
                       requested_candidates=requested_labels)
    cross = _available_units(corpus, keys, query, "cross_subject")
    same = _available_units(corpus, keys, query, "same_subject")
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

    available = sorted(available_units)
    n_labels = int(rng.integers(low, high + 1))
    n_labels = min(n_labels, len(corpus.all_labels))
    n_labels = min(n_labels, support_size)
    if n_labels < 2:
        return None

    others = [label for label in available if label != query.label]
    take = min(n_labels - 1, len(others))
    if take < 1:
        return None
    picked = list(rng.choice(others, size=take, replace=False))
    picked.append(query.label)
    rng.shuffle(picked)
    candidates = tuple(str(label) for label in picked)
    gt_slot = candidates.index(query.label)

    support_labels = [label for label in candidates if label in available_units]
    candidate_slots = {label: candidates.index(label) for label in support_labels}
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
        zero_shot=False,
        subject_relation=relation,
    )


def _draw_deployment_support_set(
    corpus: SupportCorpus,
    rng: np.random.Generator,
    *,
    support_set_id: int,
    p_gt_present: float,
    label_subset: tuple[int, int],
    mode: SamplingMode,
    same_subject_probability: float,
    enrollment_k: Sequence[int],
    queries_per_support_set: int,
    windows_per_execution: int,
    semantic_zero_shot: bool,
    query_dataset: str | None = None,
    query_label: str | None = None,
    p_mask_candidate: float = 0.0,
    p_mask_gt: float = 0.0,
    enrollment_regime: EnrollmentRegime | None = None,
    partial_coverage: tuple[float, float] = DEFAULT_PARTIAL_COVERAGE,
    variable_support_probability: float = DEFAULT_VARIABLE_SUPPORT_PROBABILITY,
    require_query_support: bool = False,
    curriculum_fallback: bool = False,
) -> list[Episode] | None:
    """Draw one deployment-shaped support roster and several independent query executions."""
    query_index = _choose_query(corpus, rng, query_dataset, query_label)
    query = corpus.recordings[query_index]
    low, high = label_subset
    if enrollment_regime not in {None, "complete", "partial", "zero"}:
        raise ValueError(f"unknown enrollment regime {enrollment_regime!r}")
    if not 0.0 <= partial_coverage[0] <= partial_coverage[1] <= 1.0:
        raise ValueError("partial_coverage must be ordered within [0, 1]")
    if not 0.0 <= variable_support_probability <= 1.0:
        raise ValueError("variable_support_probability must be in [0, 1]")
    want_support = (
        enrollment_regime != "zero" if enrollment_regime is not None
        else bool(rng.random() < p_gt_present)
    )
    # Acquisition mismatch is undefined without evidence. A zero-support episode draws its
    # candidate vocabulary from the query's own configuration and is reported as not applicable,
    # rather than pretending that an absent bank came from another placement or dataset.
    effective_mode: SamplingMode = "compatible" if not want_support else mode
    keys = _keys_for(corpus, corpus.key_of(query), effective_mode)
    if not keys:
        return None
    different_dataset = effective_mode == "cross_dataset"
    same_dataset = effective_mode == "cross_placement"
    cross = _available_units(
        corpus, keys, query, "cross_subject", different_dataset=different_dataset,
        same_dataset=same_dataset,
    )
    same = _available_units(
        corpus, keys, query, "same_subject", different_dataset=different_dataset,
        same_dataset=same_dataset,
    )

    if not want_support:
        # The zero-shot head is a genuine query/candidate-label model.  Do not manufacture the
        # retired semantic background bank: its row count and acquisition mix are unrelated to
        # the deployment condition and would give the two heads different input semantics.
        available_labels = sorted({
            query.label, *(label for label, units in cross.items() if units),
        })
        if len(available_labels) < low:
            return None
        requested_candidates = _large_candidate_count(
            rng, available=len(available_labels), label_subset=label_subset,
        )
        others = [label for label in available_labels if label != query.label]
        selected = [query.label, *rng.choice(
            others, size=requested_candidates - 1, replace=False,
        ).tolist()]
        rng.shuffle(selected)
        candidates = tuple(str(label) for label in selected)
        base = Episode(
            query=query_index, support=(), support_candidate=(), candidates=candidates,
            gt_slot=candidates.index(query.label), mode=effective_mode, requested_support=0,
            shrunk=False, zero_shot=True, requested_candidates=requested_candidates,
        )
        query_rows = _additional_queries(
            corpus, rng, base_query=query_index, query_keys=(corpus.key_of(query),),
            candidates=base.candidates,
            support=base.support, relation="cross_subject", count=queries_per_support_set,
        )
        return [replace(
            base, query=row, gt_slot=base.candidates.index(corpus.recordings[row].label),
            requested_support=0, shrunk=False, support_set_id=support_set_id,
            acquisition_regime="not_applicable", enrollment_regime="zero",
            curriculum_fallback=curriculum_fallback,
        ) for row in query_rows]

    feasible_by_k: dict[int, list[tuple[SubjectRelation, dict, list[str]]]] = {}
    for value in dict.fromkeys(int(item) for item in enrollment_k):
        options: list[tuple[SubjectRelation, dict, list[str]]] = []
        for relation, pool in (("cross_subject", cross), ("same_subject", same)):
            labels = sorted(label for label, units in pool.items() if len(units) >= value)
            if _can_draw_candidate_count(
                pool, query_label=query.label, available_labels=labels, count=low, k=value,
            ):
                options.append((relation, pool, labels))
        if options:
            feasible_by_k[value] = options
    if not feasible_by_k:
        return None
    # Keep the dataset-first query draw. Sparse sources train at the k values their real executions
    # support instead of being rejected until a larger source happens to be selected.
    k = int(rng.choice(list(feasible_by_k)))
    options = feasible_by_k[k]
    requested_relation: SubjectRelation = (
        "same_subject" if rng.random() < same_subject_probability else "cross_subject"
    )
    by_relation = {option[0]: option for option in options}
    relation_fallback = requested_relation not in by_relation
    relation, available_units, available = by_relation.get(requested_relation, options[0])

    requested_labels = _large_candidate_count(
        rng, available=len(available), label_subset=label_subset,
    )
    candidates, support, slots, groups = _draw_feasible_candidate_roster(
        available_units, query_label=query.label, available_labels=available,
        requested_labels=requested_labels, rng=rng, k=k,
        windows_per_execution=windows_per_execution,
    )
    if len(candidates) < low or not support:
        return None
    if not 0.0 <= p_mask_candidate <= 1.0 or not 0.0 <= p_mask_gt <= 1.0:
        raise ValueError("candidate masking probabilities must be in [0, 1]")
    query_rows = None
    if not require_query_support:
        query_rows = _additional_queries(
            corpus, rng, base_query=query_index, query_keys=(corpus.key_of(query),),
            candidates=candidates,
            support=support, relation=relation, count=queries_per_support_set,
        )
    requested = len(candidates) * k
    # Unequal enrollment counts are created only by removing honestly drawn, execution-distinct
    # rows. This preserves every leakage and feasibility guarantee of the exact-k draw.
    if k > 1 and rng.random() < variable_support_probability:
        available_counts = [value for value in enrollment_k if int(value) <= k]
        target_counts = [int(rng.choice(available_counts)) for _ in candidates]
        keep = []
        seen = [0] * len(candidates)
        for index, slot in enumerate(slots):
            if seen[slot] < target_counts[slot]:
                keep.append(index)
                seen[slot] += 1
        support = [support[index] for index in keep]
        slots = [slots[index] for index in keep]
        groups = [groups[index] for index in keep]

    shared_masked: tuple[int, ...] | None = None
    effective_regime = enrollment_regime or "complete"
    if enrollment_regime == "partial":
        fraction = float(rng.uniform(*partial_coverage))
        enrolled_count = min(len(candidates) - 1, max(1, int(round(len(candidates) * fraction))))
        if require_query_support:
            # A support-only neighbor objective is undefined when the query's class has no
            # enrollment. Keep the base query's class, then hide an independently sampled subset
            # of distractors. Learned classifiers leave this flag false and retain the harder,
            # query-independent partial-enrollment protocol.
            base_slot = candidates.index(query.label)
            other_slots = [slot for slot in range(len(candidates)) if slot != base_slot]
            enrolled = {base_slot, *(int(value) for value in rng.choice(
                other_slots, size=enrolled_count - 1, replace=False,
            ))}
        else:
            enrolled = set(int(value) for value in rng.choice(
                len(candidates), size=enrolled_count, replace=False,
            ))
        shared_masked = tuple(slot for slot in range(len(candidates)) if slot not in enrolled)
    elif enrollment_regime == "complete":
        shared_masked = ()

    if require_query_support:
        allowed_candidates = tuple(
            candidate for slot, candidate in enumerate(candidates)
            if shared_masked is None or slot not in shared_masked
        )
        query_rows = _additional_queries(
            corpus, rng, base_query=query_index, query_keys=(corpus.key_of(query),),
            candidates=allowed_candidates,
            support=support, relation=relation, count=queries_per_support_set,
        )
    assert query_rows is not None

    episodes = []
    for row in query_rows:
        row_gt_slot = candidates.index(corpus.recordings[row].label)
        if shared_masked is None:
            # Historical masking path retained for exact checkpoint reproduction.
            masked = [
                slot for slot in range(len(candidates))
                if slot != row_gt_slot and rng.random() < p_mask_candidate
            ]
            if rng.random() < p_mask_gt:
                masked.append(row_gt_slot)
            masked_tuple = tuple(sorted(set(masked)))
            if not masked_tuple:
                effective_regime = "complete"
            elif len(masked_tuple) == len(candidates):
                effective_regime = "zero"
            else:
                effective_regime = "partial"
        else:
            masked_tuple = shared_masked
        masked_set = set(masked_tuple)
        keep = [index for index, slot in enumerate(slots) if slot not in masked_set]
        row_support = tuple(support[index] for index in keep)
        row_slots = tuple(slots[index] for index in keep)
        row_groups = tuple(groups[index] for index in keep)
        episodes.append(Episode(
            query=row, support=row_support, support_candidate=row_slots,
            candidates=candidates, gt_slot=row_gt_slot, mode=mode,
            requested_support=requested, shrunk=len(row_support) < requested,
            zero_shot=not row_support,
            subject_relation=relation, support_window_groups=row_groups,
            support_per_candidate=(max(row_slots.count(slot) for slot in range(len(candidates)))
                                   if row_slots else 0),
            support_set_id=support_set_id,
            requested_candidates=requested_labels,
            subject_relation_fallback=relation_fallback,
            masked_candidates=masked_tuple,
            support_counts=tuple(row_slots.count(slot) for slot in range(len(candidates))),
            acquisition_regime=mode,
            enrollment_regime=effective_regime,
            curriculum_fallback=curriculum_fallback,
        ))
    return episodes


def _eligible_deployment_datasets(
    corpus: SupportCorpus,
    *,
    p_gt_present: float,
    mode: SamplingMode,
    enrollment_k: Sequence[int],
    semantic_zero_shot: bool,
    label_subset: tuple[int, int],
) -> tuple[str, ...]:
    """Datasets that can structurally form at least one requested episode regime.

    This is cached per corpus. It prevents a held-out source with no execution-disjoint enrollment
    pair from crashing validation, while ensuring that exclusion is explicit and auditable rather
    than an accidental consequence of repeated random-query rejection.
    """
    corpus.ensure_indexes()
    cache_key = (
        float(p_gt_present), mode, tuple(dict.fromkeys(int(k) for k in enrollment_k)),
        bool(semantic_zero_shot), tuple(int(value) for value in label_subset),
    )
    cached = corpus.deployment_dataset_cache.get(cache_key)
    if cached is not None:
        return cached
    allow_support = p_gt_present > 0.0
    allow_zero = p_gt_present < 1.0 and semantic_zero_shot
    eligible: list[str] = []
    k_values = cache_key[2]
    minimum_candidates = cache_key[4][0]
    def representative_queries(dataset: str, label: str) -> list[int]:
        """One execution per acquisition key/subject, not every overlapping source window."""
        representatives: list[int] = []
        seen: set[tuple[AcquisitionKey, str]] = set()
        for index in corpus.query_by_dataset_label[(dataset, label)]:
            recording = corpus.recordings[index]
            identity = (corpus.key_of(recording), recording.subject)
            if identity in seen:
                continue
            seen.add(identity)
            representatives.append(index)
        return representatives

    def structurally_available_labels(
        query: Recording,
        keys: Sequence[AcquisitionKey],
        *,
        minimum_k: int,
        require_other_dataset: bool,
        require_same_dataset: bool,
    ) -> set[str]:
        """Cheap conservative index used only to choose queries worth an exact draw.

        The final draw still performs the global distinct-execution matching. Here we merely avoid
        retrying query configurations that cannot provide even ``minimum_k`` cross-subject units
        for the true label and one distractor.
        """
        query_subject = (query.dataset, query.subject)
        query_execution = (query.dataset, query.subject, query.execution)
        units_by_label: dict[str, set[SupportUnit]] = defaultdict(set)
        for key in keys:
            for candidate_label in corpus.labels_by_key.get(key, ()):
                units_by_label[candidate_label].update(
                    unit
                    for unit in corpus.by_key_label_unit.get((key, candidate_label), {})
                    if unit != query_execution
                    and unit[:2] != query_subject
                    and (not require_other_dataset or unit[0] != query.dataset)
                    and (not require_same_dataset or unit[0] == query.dataset)
                )
        return {
            candidate_label for candidate_label, units in units_by_label.items()
            if len(units) >= minimum_k
        }

    for dataset in sorted(corpus.query_labels_by_dataset):
        viable_query_labels: set[str] = set()
        for label in corpus.query_labels_by_dataset[dataset]:
            label_is_viable = False
            for query_index in representative_queries(dataset, label):
                query = corpus.recordings[query_index]
                keys = _keys_for(corpus, corpus.key_of(query), mode)
                if not keys:
                    continue
                if allow_zero:
                    labels = structurally_available_labels(
                        query, keys, minimum_k=1,
                        require_other_dataset=mode == "cross_dataset",
                        require_same_dataset=mode == "cross_placement",
                    )
                    labels.add(query.label)
                    if query.label in labels and len(labels) >= minimum_candidates:
                        label_is_viable = True
                        break
                if allow_support:
                    minimum_k = min(k_values)
                    available = structurally_available_labels(
                        query, keys, minimum_k=minimum_k,
                        require_other_dataset=mode == "cross_dataset",
                        require_same_dataset=mode == "cross_placement",
                    )
                    # This is a cheap eligibility index, not the final draw. Exact global
                    # execution matching still happens in _draw_k_per_candidate.
                    if query.label in available and len(available) >= minimum_candidates:
                        label_is_viable = True
                if label_is_viable:
                    break
            if label_is_viable:
                viable_query_labels.add(label)
        if len(viable_query_labels) >= 2:
            eligible.append(dataset)
        corpus.deployment_label_cache[(cache_key, dataset)] = tuple(sorted(viable_query_labels))
    result = tuple(eligible)
    corpus.deployment_dataset_cache[cache_key] = result
    return result


def draw_batch(
    corpus: SupportCorpus,
    rng: np.random.Generator,
    *,
    batch_size: int,
    max_attempts_per_episode: int = 32,
    deployment_matched: bool = False,
    enrollment_k: Sequence[int] = DEFAULT_ENROLLMENT_K,
    queries_per_support_set: int = DEFAULT_QUERIES_PER_SUPPORT_SET,
    windows_per_execution: int = DEFAULT_WINDOWS_PER_EXECUTION,
    p_mask_candidate: float = 0.0,
    p_mask_gt: float = 0.0,
    acquisition_mix: Sequence[float] | None = None,
    enrollment_mix: Sequence[float] | None = None,
    partial_coverage: tuple[float, float] = DEFAULT_PARTIAL_COVERAGE,
    variable_support_probability: float = DEFAULT_VARIABLE_SUPPORT_PROBABILITY,
    require_query_support: bool = False,
    **kwargs,
) -> tuple[list[Episode], dict[str, float]]:
    """Draw ``batch_size`` episodes plus the telemetry that makes the draw auditable."""

    if deployment_matched:
        # Keep the deployment episode contract self-contained.  The trainer passes these values
        # explicitly for provenance, but profiling, tests and downstream callers must not be able
        # to reach a half-specified path that fails only after corpus loading has finished.
        kwargs = {
            "p_gt_present": DEFAULT_P_GT_PRESENT,
            "label_subset": DEFAULT_LABEL_SUBSET,
            "mode": "compatible",
            "same_subject_probability": DEFAULT_SAME_SUBJECT_PROBABILITY,
            "semantic_zero_shot": True,
            **kwargs,
        }
        if queries_per_support_set < 1 or windows_per_execution < 1:
            raise ValueError("query and execution-window counts must be positive")
        if not enrollment_k or any(int(k) < 1 for k in enrollment_k):
            raise ValueError("enrollment_k must contain positive values")
        corpus.ensure_indexes()
        acquisition_names: tuple[SamplingMode, ...] = (
            ("compatible", "cross_placement", "cross_dataset")
            if acquisition_mix is not None else (kwargs.get("mode", "compatible"),)
        )
        enrollment_names: tuple[EnrollmentRegime | None, ...] = (
            ("complete", "partial", "zero") if enrollment_mix is not None else (None,)
        )

        def probabilities(values: Sequence[float] | None, count: int, name: str) -> np.ndarray:
            if values is None:
                return np.ones(count, dtype=np.float64) / count
            result = np.asarray(values, dtype=np.float64)
            if result.shape != (count,) or not np.isfinite(result).all() or (result < 0).any() \
                    or result.sum() <= 0:
                raise ValueError(f"{name} must contain {count} finite nonnegative weights")
            return result / result.sum()

        acquisition_probability = probabilities(
            acquisition_mix, len(acquisition_names), "acquisition_mix",
        )
        enrollment_probability = probabilities(
            enrollment_mix, len(enrollment_names), "enrollment_mix",
        )
        # Acquisition has no meaning when there is no support. Represent zero enrollment once,
        # rather than once per acquisition mode: the old Cartesian product multiplied zero-shot
        # mass whenever a dataset could not form a requested mismatch regime.
        active_pairs: list[tuple[SamplingMode, EnrollmentRegime | None]] = []
        pair_weight: dict[tuple[SamplingMode, EnrollmentRegime | None], float] = {}
        for regime_index, regime in enumerate(enrollment_names):
            if enrollment_probability[regime_index] <= 0:
                continue
            if regime == "zero":
                pair = ("compatible", regime)
                active_pairs.append(pair)
                pair_weight[pair] = float(enrollment_probability[regime_index])
                continue
            for mode_index, mode_name in enumerate(acquisition_names):
                if acquisition_probability[mode_index] <= 0:
                    continue
                pair = (mode_name, regime)
                active_pairs.append(pair)
                pair_weight[pair] = float(
                    acquisition_probability[mode_index] * enrollment_probability[regime_index]
                )
        eligible_by_pair: dict[tuple[SamplingMode, EnrollmentRegime | None], tuple[str, ...]] = {}
        for pair in active_pairs:
            mode_name, regime = pair
            eligibility_mode: SamplingMode = "compatible" if regime == "zero" else mode_name
            eligible_by_pair[pair] = _eligible_deployment_datasets(
                corpus,
                p_gt_present=(0.0 if regime == "zero" else 1.0 if regime is not None
                              else float(kwargs.get("p_gt_present", DEFAULT_P_GT_PRESENT))),
                mode=eligibility_mode,
                enrollment_k=enrollment_k,
                semantic_zero_shot=bool(kwargs.get("semantic_zero_shot", False)),
                label_subset=kwargs.get("label_subset", DEFAULT_LABEL_SUBSET),
            )
        eligible = tuple(sorted({
            dataset for datasets in eligible_by_pair.values() for dataset in datasets
        }))
        support_sets: list[list[Episode]] = []
        attempts = unusable = 0
        if not eligible:
            raise RuntimeError(
                "no dataset can form the requested deployment-matched regime; refusing to relax "
                "compatibility or duplicate executions"
            )
        datasets = np.asarray(eligible, dtype=object)
        # One shuffled cycle gives every source exactly one support set before any source repeats.
        # For a sub-cycle (the usual four-set step), sampling is without replacement.
        schedule: list[str] = []
        while len(schedule) < batch_size:
            schedule.extend(str(value) for value in rng.permutation(datasets))
        for support_set_id, query_dataset in enumerate(schedule[:batch_size]):
            feasible_pairs = [pair for pair, datasets_for_pair in eligible_by_pair.items()
                              if query_dataset in datasets_for_pair]
            if not feasible_pairs:
                raise RuntimeError(f"dataset {query_dataset!r} has no feasible curriculum condition")

            requested_enrollment = enrollment_names[int(rng.choice(
                len(enrollment_names), p=enrollment_probability,
            ))]
            feasible_enrollments = tuple(dict.fromkeys(pair[1] for pair in feasible_pairs))
            enrollment_fallback = requested_enrollment not in feasible_enrollments
            if enrollment_fallback:
                enrollment_weights = np.asarray([
                    enrollment_probability[enrollment_names.index(regime)]
                    for regime in feasible_enrollments
                ], dtype=np.float64)
                enrollment_weights /= enrollment_weights.sum()
                actual_enrollment = feasible_enrollments[int(rng.choice(
                    len(feasible_enrollments), p=enrollment_weights,
                ))]
            else:
                actual_enrollment = requested_enrollment

            if actual_enrollment == "zero":
                requested_mode: SamplingMode = "compatible"
                actual_pair = ("compatible", actual_enrollment)
                acquisition_fallback = False
            else:
                requested_mode = acquisition_names[int(rng.choice(
                    len(acquisition_names), p=acquisition_probability,
                ))]
                feasible_modes = [
                    pair[0] for pair in feasible_pairs if pair[1] == actual_enrollment
                ]
                acquisition_fallback = requested_mode not in feasible_modes
                if acquisition_fallback:
                    mode_weights = np.asarray([
                        acquisition_probability[acquisition_names.index(mode_name)]
                        for mode_name in feasible_modes
                    ], dtype=np.float64)
                    mode_weights /= mode_weights.sum()
                    requested_mode = feasible_modes[int(rng.choice(
                        len(feasible_modes), p=mode_weights,
                    ))]
                actual_pair = (requested_mode, actual_enrollment)
            initial_fallback = enrollment_fallback or acquisition_fallback
            group = None
            alternatives = [
                pair for pair in feasible_pairs
                if pair != actual_pair and pair[1] == actual_enrollment
            ]
            alternatives.extend(
                pair for pair in feasible_pairs
                if pair != actual_pair and pair[1] != actual_enrollment
            )
            rng.shuffle(alternatives)
            attempted_pairs = [actual_pair, *alternatives]
            for pair_index, attempted_pair in enumerate(attempted_pairs):
                actual_mode, actual_enrollment = attempted_pair
                fallback = initial_fallback or pair_index > 0
                for _ in range(max_attempts_per_episode):
                    attempts += 1
                    eligibility_key = (
                        0.0 if actual_enrollment == "zero" else 1.0 if actual_enrollment is not None
                        else float(kwargs.get("p_gt_present", DEFAULT_P_GT_PRESENT)),
                        actual_mode, tuple(dict.fromkeys(int(k) for k in enrollment_k)),
                        bool(kwargs.get("semantic_zero_shot", False)),
                        tuple(int(value) for value in kwargs.get("label_subset", DEFAULT_LABEL_SUBSET)),
                    )
                    viable_labels = corpus.deployment_label_cache.get(
                        (eligibility_key, query_dataset), (),
                    )
                    query_label = (
                        str(rng.choice(viable_labels))
                        if actual_mode != "compatible" and viable_labels else None
                    )
                    group = _draw_deployment_support_set(
                        corpus, rng, support_set_id=support_set_id,
                        enrollment_k=enrollment_k,
                        queries_per_support_set=queries_per_support_set,
                        windows_per_execution=windows_per_execution,
                        query_dataset=query_dataset,
                        query_label=query_label,
                        p_mask_candidate=p_mask_candidate, p_mask_gt=p_mask_gt,
                        enrollment_regime=actual_enrollment,
                        partial_coverage=partial_coverage,
                        variable_support_probability=variable_support_probability,
                        require_query_support=require_query_support,
                        curriculum_fallback=fallback,
                        **{**kwargs, "mode": actual_mode},
                    )
                    if group:
                        break
                    unusable += 1
                if group:
                    break
            if not group:
                raise RuntimeError(
                    f"could not draw a deployment-matched support set for dataset "
                    f"{query_dataset!r} under any of {attempted_pairs!r}; refusing to "
                    "relax compatibility or duplicate executions, and will not substitute an "
                    "easier source"
                )
            support_sets.append(group)
        episodes = [episode for group in support_sets for episode in group]
        query_counts = [len(group) for group in support_sets]
        support_set_episodes = [group[0] for group in support_sets]
        # Support-set-level values remain useful for data diversity, but regime and GT-support
        # telemetry must use every query because candidate masking is deliberately per query.
        regime_episodes = episodes
    else:
        query_counts = [1] * batch_size

        episodes = []
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
        regime_episodes = episodes
        support_set_episodes = episodes
    zero_shot = sum(1 for episode in episodes if episode.is_zero_shot)
    gt_supported = [any(slot == episode.gt_slot for slot in episode.support_candidate)
                    for episode in episodes]
    query_datasets = [corpus.recordings[episode.query].dataset for episode in episodes]
    query_labels = [corpus.recordings[episode.query].label for episode in episodes]
    dataset_counts = {value: query_datasets.count(value) for value in set(query_datasets)}
    support_set_datasets = [
        corpus.recordings[episode.query].dataset for episode in support_set_episodes
    ]
    support_set_dataset_counts = {
        value: support_set_datasets.count(value) for value in set(support_set_datasets)
    }
    duplicate_executions = []
    for episode in episodes:
        units = {
            (corpus.recordings[index].dataset, corpus.recordings[index].subject,
             corpus.recordings[index].execution)
            for index in episode.support
        }
        duplicate_executions.append(len(episode.support) - len(units))
    telemetry = {
        "sampler/realised_gt_rate": float(np.mean(gt_supported)),
        "sampler/zero_shot_rate": zero_shot / len(episodes),
        "sampler/gt_support_present_rate": float(np.mean(gt_supported)),
        "sampler/partial_gt_mask_rate": float(np.mean([
            (not present) and bool(episode.support) for present, episode in zip(gt_supported, episodes)
        ])),
        "sampler/masked_candidate_fraction": float(np.mean([
            len(episode.masked_candidates) / max(1, len(episode.candidates)) for episode in episodes
        ])),
        "sampler/same_subject_rate": sum(
            episode.subject_relation == "same_subject" for episode in regime_episodes
            if not episode.is_zero_shot
        ) / max(1, len(regime_episodes) - zero_shot),
        "sampler/shrunk_episode_fraction": sum(e.shrunk for e in episodes) / len(episodes),
        "sampler/mean_support_size": float(np.mean([len(e.support) for e in episodes])),
        "sampler/mean_candidate_count": float(np.mean([len(e.candidates) for e in episodes])),
        "sampler/candidate_roster_shrink_fraction": float(np.mean([
            episode.requested_candidates > len(episode.candidates)
            for episode in regime_episodes if episode.requested_candidates
        ])) if any(episode.requested_candidates for episode in regime_episodes) else 0.0,
        "sampler/subject_relation_fallback_fraction": float(np.mean([
            episode.subject_relation_fallback
            for episode in regime_episodes if not episode.is_zero_shot
        ])) if any(not episode.is_zero_shot for episode in regime_episodes) else 0.0,
        "sampler/unusable_query_fraction": unusable / max(attempts, 1),
        "sampler/query_dataset_count": float(len(set(query_datasets))),
        "sampler/query_label_count": float(len(set(query_labels))),
        "sampler/max_query_dataset_share": max(dataset_counts.values()) / len(episodes),
        "sampler/max_support_set_dataset_share": (
            max(support_set_dataset_counts.values()) / len(support_set_episodes)
        ),
        "sampler/min_support_set_dataset_share": (
            min(support_set_dataset_counts.values()) / len(support_set_episodes)
        ),
        "sampler/duplicate_support_execution_mean": float(np.mean(duplicate_executions)),
        "sampler/support_set_count": float(len(query_counts)),
        "sampler/eligible_dataset_count": (
            float(len(eligible)) if deployment_matched
            else float(len(corpus.query_labels_by_dataset))
        ),
        "sampler/ineligible_dataset_count": (
            float(len(corpus.query_labels_by_dataset) - len(eligible))
            if deployment_matched else 0.0
        ),
        "sampler/mean_queries_per_support_set": float(np.mean(query_counts)),
        "sampler/query_set_shrink_fraction": float(np.mean([
            count < queries_per_support_set for count in query_counts
        ])) if deployment_matched else 0.0,
        "sampler/mean_k_per_candidate": float(np.mean([
            len(episode.support) / max(1, len(episode.candidates))
            for episode in episodes if not episode.is_zero_shot
        ])) if any(not episode.is_zero_shot for episode in episodes) else 0.0,
        "sampler/mean_windows_per_support_execution": float(np.mean([
            len(group) for episode in episodes for group in episode.support_window_groups
        ])) if any(episode.support_window_groups for episode in episodes) else 1.0,
        "sampler/curriculum_fallback_fraction": float(np.mean([
            episode.curriculum_fallback for episode in support_set_episodes
        ])),
    }
    if deployment_matched:
        enrolled_support_sets = [
            episode for episode in support_set_episodes if not episode.is_zero_shot
        ]
        for value in dict.fromkeys(int(item) for item in enrollment_k):
            telemetry[f"sampler/k_{value}_support_set_fraction"] = sum(
                episode.support_per_candidate == value for episode in enrolled_support_sets
            ) / max(1, len(enrolled_support_sets))
        for name in ("compatible", "cross_placement", "cross_dataset"):
            telemetry[f"sampler/acquisition_{name}_fraction"] = sum(
                episode.acquisition_regime == name for episode in enrolled_support_sets
            ) / max(1, len(enrolled_support_sets))
        telemetry["sampler/acquisition_not_applicable_fraction"] = sum(
            episode.acquisition_regime == "not_applicable" for episode in support_set_episodes
        ) / len(support_set_episodes)
        for name in ("complete", "partial", "zero"):
            telemetry[f"sampler/enrollment_{name}_fraction"] = sum(
                episode.enrollment_regime == name for episode in support_set_episodes
            ) / len(support_set_episodes)
        telemetry["sampler/unequal_support_count_fraction"] = float(np.mean([
            len(set(count for count in episode.support_counts if count > 0)) > 1
            for episode in enrolled_support_sets
        ])) if enrolled_support_sets else 0.0
    return episodes, telemetry
