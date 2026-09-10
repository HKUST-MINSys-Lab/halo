"""Shared corpus data-prep for LiMU-BERT (grid -> the 6-ch / 20 Hz / 120-sample
input contract both the SSL pretrain and the adapter consume).

LiMU-BERT's verified input contract (BASELINES.md): 6-channel acc+gyro, 20 Hz,
120 samples (= 6 s) per window; acceleration expressed in g (gravity retained),
gyro in rad/s, with the model's internal LayerNorm. Upstream divides acceleration
by 9.8 because its source arrays are in m/s²; our grids are already in g, so no
second division is applied. This module maps each native grid window to the
6-ch / 20 Hz / 120-sample shape:

  * 6 channels in a fixed ``acc_{x,y,z}, gyro_{x,y,z}`` order; accelerometer-only
    train sets (wisdm, unimib_shar, capture24) get their gyro channels
    ZERO-FILLED (a benign "gyro absent" encoding, not fake signal).
  * resampled to 20 Hz (polyphase, anti-aliased) and center-cropped / wrap-padded
    to exactly 120 samples.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Iterator, List, Tuple

import numpy as np
from scipy.signal import resample_poly

from data.scripts.curate.deployment_policy import EXPANDED_PHASE_A_TRAIN_DATASETS
from eval import data as eval_data

TARGET_HZ = 20
TARGET_LEN = 120          # 6 s @ 20 Hz
PREP_SCHEMA = "native-length-aware-v2"
SIX_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")

# Same-data method arm: use the design-of-record Phase-A roster from the deployment
# policy rather than maintaining a second list that can silently drift.
TRAIN_DATASETS = tuple(EXPANDED_PHASE_A_TRAIN_DATASETS)


def load_grid(dataset: str, stream: str):
    """Read a native grid; returns windows, labels, subjects, channels, rate, and lengths."""
    gdir = eval_data.DATASETS_DIR / dataset / "grids" / "native" / stream
    windows = np.load(gdir / "data.npy")
    meta = json.loads((gdir / "meta.json").read_text())
    lengths_path = gdir / meta.get("lengths_file", "lengths.npy")
    lengths = (np.load(lengths_path) if lengths_path.exists()
               else np.full(len(windows), windows.shape[1], dtype=np.int32))
    return (windows, list(meta["labels"]), list(map(str, meta["subjects"])),
            list(meta["channels"]), float(meta["rate_hz"]), lengths)


def to_six_channels(windows: np.ndarray, channels: List[str]) -> np.ndarray:
    """(N, T, C) grid -> (N, T, 6) in the fixed acc+gyro order; missing channels
    (e.g. gyro on accel-only sets) are zero-filled."""
    idx = {c: i for i, c in enumerate(channels)}
    out = np.zeros((windows.shape[0], windows.shape[1], len(SIX_CHANNELS)),
                   dtype=np.float32)
    for j, ch in enumerate(SIX_CHANNELS):
        if ch in idx:
            out[:, :, j] = windows[:, :, idx[ch]]
    return out


def resample_crop_pad(windows: np.ndarray, rate_hz: float,
                      lengths: np.ndarray | None = None) -> np.ndarray:
    """(N, T, 6) at `rate_hz` -> (N, 120, 6) at 20 Hz (resample + center-crop /
    wrap-pad)."""
    frac = Fraction(int(round(TARGET_HZ)), int(round(rate_hz))).limit_denominator(1000)
    valid = (np.asarray(lengths, dtype=np.int64) if lengths is not None
             else np.full(len(windows), windows.shape[1], dtype=np.int64))
    out = []
    for window, length in zip(windows, valid):
        y = resample_poly(window[:int(length)].astype(np.float64), frac.numerator,
                          frac.denominator, axis=0)
        if len(y) > TARGET_LEN:
            off = (len(y) - TARGET_LEN) // 2
            y = y[off:off + TARGET_LEN]
        elif len(y) < TARGET_LEN:
            total = TARGET_LEN - len(y)
            left = total // 2
            y = np.pad(y, ((left, total - left), (0, 0)), mode="wrap")
        out.append(y)
    return np.asarray(out, dtype=np.float32)


def grid_to_contract(windows: np.ndarray, channels: List[str], rate_hz: float,
                     lengths: np.ndarray | None = None) -> np.ndarray:
    """Full grid-window -> LiMU-BERT contract input (N, 120, 6), native g."""
    return resample_crop_pad(to_six_channels(windows, channels), rate_hz, lengths)


def iter_train_streams(max_per_stream: int | None = None,
                       seed: int = 3431) -> Iterator[Tuple[str, str, np.ndarray, List[str], np.ndarray]]:
    """Yield ``(dataset, stream, X6, raw_labels, subjects)`` for every training
    stream, X6 being (n, 120, 6) native-g contract input."""
    rng = np.random.RandomState(seed)
    for ds in TRAIN_DATASETS:
        for stream in eval_data.list_streams(ds, alignment="native"):
            windows, labels, subjects, channels, rate, lengths = load_grid(ds, stream)
            subjects = np.asarray(subjects)
            labels = np.asarray(labels, dtype=object)
            if max_per_stream is not None and len(windows) > max_per_stream:
                sel = rng.choice(len(windows), size=max_per_stream, replace=False)
                windows, labels, subjects, lengths = windows[sel], labels[sel], subjects[sel], lengths[sel]
            x6 = grid_to_contract(windows, channels, rate, lengths)
            yield ds, stream, x6, list(labels), subjects


def build_pretrain_array(max_per_stream: int | None = None,
                         seed: int = 3431, *, shuffle: bool = True) -> np.ndarray:
    """Pool the training corpus into a single (N, 120, 6) array for SSL pretrain."""
    parts = [x6 for _, _, x6, _, _ in iter_train_streams(max_per_stream, seed)]
    data = np.concatenate(parts, axis=0)
    if shuffle:
        rng = np.random.RandomState(seed)
        return data[rng.permutation(len(data))]
    return data
