"""Bridge the support sampler to the existing Phase-A data path.

The sampler reasons about *recordings*; the encoder consumes the batches ``MultiScaleCollate``
produces from a :class:`PretrainDataset`. This module builds a :class:`SupportCorpus` whose
``window_index`` is a position in that dataset, so an episode's query and support rows can be
fetched, collated and encoded in one heterogeneous forward with no re-implementation of the
loading, augmentation or text-conditioning path.

Keeping this separate from :mod:`training.compare.sampling` means the sampler stays testable on a
synthetic corpus with no grids, no torch and no encoder — which is what makes its rules cheap to
assert.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from data.scripts.curate.compatibility import AcquisitionKey, is_near_miss, stream_key
from training.compare.sampling import (
    MIN_RECORDING_SECONDS,
    Recording,
    SupportCorpus,
    _execution_ids,
)
from training.tokenizer.pretrain_data import CorpusIndex

REPO = Path(__file__).resolve().parents[2]


def support_corpus_from_index(
    index: CorpusIndex,
    *,
    split: str = "train",
    exclude_labels: Sequence[str] = ("unlabeled",),
    min_duration_seconds: float = MIN_RECORDING_SECONDS,
) -> SupportCorpus:
    """A :class:`SupportCorpus` addressing positions in ``index.<split>``.

    ``Recording.window_index`` is the index into the key list a :class:`PretrainDataset` is built
    over, so ``dataset[recording.window_index]`` returns exactly that window.
    """

    keys = getattr(index, split)
    banned = {str(label).lower() for label in exclude_labels}
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
        if str(label).lower() in banned:
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
