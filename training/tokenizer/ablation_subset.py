"""The 3-rate-core subset for controlled tokenizer experiments.

A small, deliberately diverse slice of the corpus for running tokenizer experiments quickly.
It spans sampling rate (20/50/100 Hz), on-body placement, channel modality (acc-only vs
acc+gyro), and gravity present/removed, with a held-out dataset for cross-config transfer.

    7 train streams / 5 datasets:
      wisdm phone_pocket   20 Hz  pocket  acc+gyro  gravity     (only 20 Hz — low-rate stress)
      wisdm watch_wrist    20 Hz  wrist   acc+gyro  gravity
      hhar phone_waist     50 Hz  waist   acc+gyro  gravity
      hhar phone_waist_accel_only 50 Hz waist acc-only gravity
      unimib phone_pocket  50 Hz  pocket  acc-only  gravity     (the 3-channel case)
      pamap2 watch_wrist  100 Hz  wrist   acc+gyro  gravity
      kuhar phone_waist   100 Hz  waist   acc+gyro  REMOVED

    held-out config (never trained): xrf_v2 (50 Hz, unseen dataset/device; 6 simultaneous
    placements make it ideal for the cross-config retrieval metric).

capture24 is deliberately excluded from this first read (144k acc-only free-living windows would
dominate and dilute label diversity).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

# Train datasets for the subset. wisdm contributes both its streams (20 Hz pocket + wrist); the
# rest one each. CorpusIndex splits subjects disjointly within each dataset.
# Rebuilt 2026-09-11: unimib_shar and pamap2 were retired from the head roster, so the old tuple
# was refused at launch by `assert_no_retired_sources`. The replacements keep the property the
# subset exists for - a wide native-rate spread across few sources - at 20 / 25 / 50 / 51.2 /
# 100 Hz, with xrf_v2 still held out as the unseen configuration.
SUBSET_TRAIN_DATASETS = ("wisdm", "dsads", "hhar", "forth_trace", "kuhar")

# Held-out for cross-config transfer (unseen dataset -> every window is an unseen config).
SUBSET_HELDOUT_DATASETS = ("xrf_v2",)

DEFAULT_CAP = 10_000            # per-stream window cap (keeps the subset balanced + fast)

# Placement label per (dataset, stream) for the config-decodability / cross-config metrics.
PLACEMENT = {
    ("wisdm", "phone_pocket"): "pocket", ("wisdm", "watch_wrist"): "wrist",
    ("hhar", "phone_waist"): "waist",
    ("hhar", "phone_waist_accel_only"): "waist",
    ("dsads", "left_wrist"): "wrist", ("dsads", "right_wrist"): "wrist",
    ("dsads", "chest"): "chest", ("dsads", "left_knee"): "knee",
    ("dsads", "right_knee"): "knee",
    ("forth_trace", "left_wrist"): "wrist", ("forth_trace", "right_wrist"): "wrist",
    ("forth_trace", "chest"): "chest", ("forth_trace", "torso"): "torso",
    ("forth_trace", "left_ankle"): "ankle",
    ("forth_trace", "right_thigh"): "thigh",
    ("kuhar", "phone_waist"): "waist",
    ("xrf_v2", "left_wrist"): "wrist", ("xrf_v2", "right_wrist"): "wrist",
    ("xrf_v2", "left_pocket"): "pocket", ("xrf_v2", "right_pocket"): "pocket",
    ("xrf_v2", "glasses"): "glasses", ("xrf_v2", "airpods_ear"): "ear",
}


def build_subset_index(cap: int = DEFAULT_CAP, seed: int | None = None):
    """CorpusIndex restricted to the subset train datasets, subject-disjoint train/val split."""
    from training.tokenizer.pretrain_data import CorpusIndex, SEED
    from data.scripts.curate.deployment_policy import assert_no_retired_sources
    assert_no_retired_sources(SUBSET_TRAIN_DATASETS)
    return CorpusIndex(max_per_stream=cap, seed=SEED if seed is None else seed,
                       datasets=SUBSET_TRAIN_DATASETS)


def build_heldout_index(cap: int = DEFAULT_CAP, seed: int | None = None):
    """CorpusIndex over the held-out config datasets (all windows are unseen configs)."""
    from training.tokenizer.pretrain_data import CorpusIndex, SEED
    from data.scripts.curate.deployment_policy import assert_no_retired_sources
    assert_no_retired_sources(SUBSET_HELDOUT_DATASETS)
    return CorpusIndex(max_per_stream=cap, seed=SEED if seed is None else seed,
                       datasets=SUBSET_HELDOUT_DATASETS)


def iter_split_streams(index, split: str = "val") -> Iterator[dict]:
    """Yield one dict per stream of a CorpusIndex split, ready for encode_dataset.

    Groups the split's windows by stream so each yield is a homogeneous batch (one rate / text /
    gravity), and carries the config tags (rate, placement, dataset) the metric suite needs.

    Yields: {data (N,T,6), labels (N,), texts[6], rate, gravity, channel_mask[6], dataset,
             stream, placement}.
    """
    from collections import defaultdict
    from training.tokenizer.pretrain_data import (stream_channel_descriptions,
                                                  _stream_gravity_state)

    keys = getattr(index, split)
    by_stream: dict[int, list] = defaultdict(list)
    for k in keys:
        by_stream[k.stream_i].append((k.window_i, k.label_id))

    # label_id -> label string (inverse of index.label_ids)
    id2label = {i: l for l, i in index.label_ids.items()}

    for stream_i, items in by_stream.items():
        ref = index.refs[stream_i]
        grid = ref.load_data()                                  # (n_windows, T, 6)
        wi = np.array([w for w, _ in items])
        data = grid[wi]
        labels = np.array([id2label[l] for _, l in items], dtype=object)
        yield {
            "data": data,
            "labels": labels,
            "texts": stream_channel_descriptions(ref.dataset, ref.stream),
            "rate": float(ref.rate_hz),
            "gravity": _stream_gravity_state(ref.dataset, ref.stream),
            "channel_mask": np.asarray(ref.mask, dtype=bool),
            "dataset": ref.dataset,
            "stream": ref.stream,
            "placement": PLACEMENT.get((ref.dataset, ref.stream), ref.stream),
        }


def describe() -> str:
    lines = ["tokenizer-ablation subset:"]
    from training.tokenizer.pretrain_data import _stream_gravity_state
    from data.scripts.eda.grid_io import discover_grids
    refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
    for ds in SUBSET_TRAIN_DATASETS:
        for (d, s), r in sorted(refs.items()):
            if d != ds:
                continue
            lines.append(f"  {d:12} {s:14} {r.rate_hz:>4.0f}Hz  {int(sum(r.mask))}ch  "
                         f"{PLACEMENT.get((d, s), s):8} grav={_stream_gravity_state(d, s)}")
    lines.append(f"  held-out: {SUBSET_HELDOUT_DATASETS}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
