"""Bridge the support sampler to the existing Phase-A data path.

The sampler reasons about *recordings*; the encoder consumes the batches ``MultiScaleCollate``
produces from a :class:`PretrainDataset`. This module builds a :class:`SupportCorpus` whose
``window_index`` is a position in that dataset, so an episode's query and support rows can be
fetched, collated and encoded in one heterogeneous forward with no re-implementation of the
loading, augmentation or text-conditioning path.

Keeping this separate from :mod:`training.support_classifier.sampling` means the sampler stays testable on a
synthetic corpus with no grids, no torch and no encoder — which is what makes its rules cheap to
assert.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np

from data.scripts.curate.compatibility import AcquisitionKey, is_near_miss, stream_key
from data.scripts.labels.canonical_labels import NON_SEMANTIC_LABELS
from training.support_classifier.sampling import (
    MIN_RECORDING_SECONDS,
    Recording,
    SupportCorpus,
    _execution_ids,
)
from training.tokenizer.pretrain_data import CorpusIndex

def support_corpus_from_index(
    index: CorpusIndex,
    *,
    split: str = "train",
    exclude_labels: Sequence[str] = tuple(NON_SEMANTIC_LABELS),
    include_labels: Sequence[str] | None = None,
    min_duration_seconds: float = MIN_RECORDING_SECONDS,
) -> SupportCorpus:
    """A :class:`SupportCorpus` addressing positions in ``index.<split>``.

    ``Recording.window_index`` is the index into the key list a :class:`PretrainDataset` is built
    over, so ``dataset[recording.window_index]`` returns exactly that window.
    """

    keys = getattr(index, split)
    banned = {str(label).lower() for label in (*NON_SEMANTIC_LABELS, *exclude_labels)}
    allowed = None if include_labels is None else {str(label).lower() for label in include_labels}
    id_to_label = {value: label for label, value in index.label_ids.items()}

    executions_by_stream: dict[int, np.ndarray] = {}
    lengths_by_stream: dict[int, np.ndarray] = {}
    acquisition: dict[int, AcquisitionKey | None] = {}

    recordings: list[Recording] = []
    stream_names: list[tuple[str, str]] = []
    stream_keys: list[AcquisitionKey] = []
    stream_slot: dict[int, int] = {}

    for position, key in enumerate(keys):
        ref = index.refs[key.stream_i]
        if key.stream_i not in acquisition:
            try:
                acquisition[key.stream_i] = stream_key(ref.dataset, ref.stream)
            except KeyError:
                acquisition[key.stream_i] = None
                print(
                    f"[support-corpus] {ref.dataset}/{ref.stream} has no acquisition key; its "
                    "windows cannot enter a support set and are skipped"
                )
            executions_by_stream[key.stream_i] = _execution_ids(ref.dataset, ref.event_ids)
            lengths_by_stream[key.stream_i] = ref.load_lengths()
        acquisition_key = acquisition[key.stream_i]
        if acquisition_key is None:
            continue
        if float(lengths_by_stream[key.stream_i][key.window_i]) / ref.rate_hz \
                < min_duration_seconds:
            continue
        label = id_to_label.get(key.label_id, "")
        if str(label).lower() in banned or (allowed is not None and str(label).lower() not in allowed):
            continue
        if key.stream_i not in stream_slot:
            stream_slot[key.stream_i] = len(stream_names)
            stream_names.append((ref.dataset, ref.stream))
            stream_keys.append(acquisition_key)
        recordings.append(Recording(
            stream_index=stream_slot[key.stream_i],
            window_index=position,
            dataset=ref.dataset,
            stream=ref.stream,
            label=label,
            subject=str(ref.subjects[key.window_i]),
            execution=str(executions_by_stream[key.stream_i][key.window_i]),
        ))

    corpus = SupportCorpus(
        recordings=recordings, keys=stream_keys, stream_names=stream_names,
    )
    for position, recording in enumerate(recordings):
        acquisition_key = stream_keys[recording.stream_index]
        corpus.by_key.setdefault(acquisition_key, []).append(position)
        corpus.by_key_label.setdefault(
            (acquisition_key, recording.label), []
        ).append(position)
    distinct = list(corpus.by_key)
    for acquisition_key in distinct:
        corpus.near_miss_keys[acquisition_key] = [
            other for other in distinct if is_near_miss(acquisition_key, other)
        ]
    corpus.ensure_indexes()
    return corpus


def open_vocabulary_holdout_labels(
    index: CorpusIndex, *, fraction: float, seed: int, min_per_dataset: int = 2,
    min_remaining_per_dataset: int = 2,
) -> tuple[str, ...]:
    """Choose a reproducible global label holdout with per-dataset coverage constraints.

    A held-out canonical label is removed from every optimizer source, not merely one dataset.
    Selection only considers labels available in both subject splits. The greedy constraint keeps
    enough non-held-out labels in every source for ordinary episodic training.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("open-vocabulary holdout fraction must be in [0, 1)")
    if fraction == 0:
        return ()
    id_to_label = {value: label for label, value in index.label_ids.items()}
    labels_by_split: dict[str, dict[str, set[str]]] = {"train": {}, "val": {}}
    for split in ("train", "val"):
        for key in getattr(index, split):
            dataset = index.refs[key.stream_i].dataset
            label = str(id_to_label[key.label_id])
            if label.lower() in {str(value).lower() for value in NON_SEMANTIC_LABELS}:
                continue
            labels_by_split[split].setdefault(dataset, set()).add(label)
    eligible = {
        dataset: labels & labels_by_split["val"].get(dataset, set())
        for dataset, labels in labels_by_split["train"].items()
    }
    eligible = {dataset: labels for dataset, labels in eligible.items() if labels}
    if not eligible:
        raise ValueError("no labels occur in both subject-disjoint splits")
    memberships: dict[str, set[str]] = {}
    for dataset, labels in eligible.items():
        for label in labels:
            memberships.setdefault(label, set()).add(dataset)

    target = {
        dataset: min(
            max(0, len(labels) - min_remaining_per_dataset),
            max(min_per_dataset, int(math.ceil(fraction * len(labels)))),
        )
        for dataset, labels in eligible.items()
    }
    selected: set[str] = set()

    def rank(dataset: str, label: str) -> bytes:
        return hashlib.sha256(f"{seed}\0{dataset}\0{label}".encode()).digest()

    for dataset in sorted(eligible, key=lambda name: (len(eligible[name]), name)):
        while len(selected & eligible[dataset]) < target[dataset]:
            choices = []
            for label in eligible[dataset] - selected:
                if all(
                    len(eligible[member] - (selected | {label})) >= min_remaining_per_dataset
                    for member in memberships[label]
                ):
                    choices.append(label)
            if not choices:
                raise ValueError(
                    f"cannot reserve {target[dataset]} labels for {dataset} while retaining "
                    f"{min_remaining_per_dataset} train labels per source"
                )
            selected.add(min(choices, key=lambda label: rank(dataset, label)))
    return tuple(sorted(selected))
