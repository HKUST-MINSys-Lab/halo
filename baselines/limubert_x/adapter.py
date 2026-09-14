"""LiMU-BERT-X frozen representation adapter.

The released ``base_v4`` checkpoint consumes 20 samples of six-axis IMU at 20 Hz. HALO's grids
store acceleration in g, which is already the result of the authors' ``acc / 9.8`` preprocessing;
applying that normalization again would suppress acceleration by another factor of 9.8. Gyroscope
values are passed through in rad/s, as in the source HHAR preprocessing.

Evaluation windows are usually six seconds rather than one. We encode every non-overlapping
one-second clip and take a duration-weighted mean of the released transformer's hidden states. The
last partial clip is edge padded but weighted only by its real duration. This preserves the complete
window while keeping the published 20-sample positional-embedding contract.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly

from ..base import BaselineAdapter, InputContract, UnsupportedEvaluationCell, register

FEATURE_NUM = 6
HIDDEN = 72
HIDDEN_FF = 144
N_LAYERS = 4
N_HEADS = 4
TARGET_HZ = 20.0
SEQ_LEN = 20
EMBED_BATCH = 4096
SIX_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "limu_bert_x.pt"


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class _LayerNorm(nn.Module):
    """TensorFlow-style LayerNorm used by the released implementation."""

    def __init__(self, hidden: int, eps: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta = nn.Parameter(torch.zeros(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(variance + self.eps) + self.beta


class _Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(FEATURE_NUM, HIDDEN)
        self.pos_embed = nn.Embedding(SEQ_LEN, HIDDEN)
        self.norm = _LayerNorm(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(x.shape[0], -1)
        embedded = self.norm(self.lin(x))
        return self.norm(embedded + self.pos_embed(positions))


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Linear(HIDDEN, HIDDEN)
        self.proj_k = nn.Linear(HIDDEN, HIDDEN)
        self.proj_v = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        width = HIDDEN // N_HEADS
        q = self.proj_q(x).view(batch, length, N_HEADS, width).transpose(1, 2)
        k = self.proj_k(x).view(batch, length, N_HEADS, width).transpose(1, 2)
        v = self.proj_v(x).view(batch, length, N_HEADS, width).transpose(1, 2)
        weights = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(width), dim=-1)
        return (weights @ v).transpose(1, 2).contiguous().view(batch, length, HIDDEN)


class _FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN, HIDDEN_FF)
        self.fc2 = nn.Linear(HIDDEN_FF, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(_gelu(self.fc1(x)))


class _Transformer(nn.Module):
    """The authors share one attention/FFN block across four iterations."""

    def __init__(self):
        super().__init__()
        self.embed = _Embeddings()
        self.attn = _Attention()
        self.proj = nn.Linear(HIDDEN, HIDDEN)
        self.norm1 = _LayerNorm(HIDDEN)
        self.pwff = _FeedForward()
        self.norm2 = _LayerNorm(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(x)
        for _ in range(N_LAYERS):
            hidden = self.attn(hidden)
            hidden = self.norm1(hidden + self.proj(hidden))
            hidden = self.norm2(hidden + self.pwff(hidden))
        return hidden


class _Backbone(nn.Module):
    """Exact released pretraining-module layout; forward returns encoder states."""

    def __init__(self):
        super().__init__()
        self.transformer = _Transformer()
        # These unused reconstruction modules are retained so checkpoint loading is strict.
        self.fc = nn.Linear(HIDDEN, HIDDEN)
        self.linear = nn.Linear(HIDDEN, HIDDEN)
        self.norm = _LayerNorm(HIDDEN)
        self.decoder = nn.Linear(HIDDEN, FEATURE_NUM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


def _channel_indices(stream) -> list[int]:
    index = {name: position for position, name in enumerate(stream.channels)}
    missing = [name for name in SIX_CHANNELS if name not in index]
    if missing:
        raise UnsupportedEvaluationCell(
            f"LiMU-BERT-X requires measured six-axis IMU; missing channels: {missing}"
        )
    selected = [index[name] for name in SIX_CHANNELS]
    if not np.asarray(stream.mask, dtype=bool)[selected].all():
        masked = [name for name, position in zip(SIX_CHANNELS, selected) if not stream.mask[position]]
        raise UnsupportedEvaluationCell(
            f"LiMU-BERT-X requires measured six-axis IMU; masked channels: {masked}"
        )
    return selected


def _resample(x: np.ndarray, rate_hz: float, target_length: int) -> np.ndarray:
    if float(rate_hz) == TARGET_HZ and x.shape[1] == target_length:
        return x.astype(np.float32, copy=False)
    ratio = Fraction(TARGET_HZ / float(rate_hz)).limit_denominator(1000)
    output = resample_poly(x.astype(np.float64), ratio.numerator, ratio.denominator, axis=1)
    if output.shape[1] > target_length:
        output = output[:, :target_length]
    elif output.shape[1] < target_length:
        output = np.pad(output, ((0, 0), (0, target_length - output.shape[1]), (0, 0)), mode="edge")
    return output.astype(np.float32)


@torch.inference_mode()
def _encode_clips(backbone: nn.Module, clips: np.ndarray, device) -> np.ndarray:
    rows = []
    for start in range(0, len(clips), EMBED_BATCH):
        batch = torch.from_numpy(clips[start : start + EMBED_BATCH]).to(device=device, dtype=torch.float32)
        rows.append(backbone(batch).mean(dim=1).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


@torch.inference_mode()
def _window_features(stream, backbone: nn.Module, device) -> np.ndarray:
    """Encode complete valid windows without allowing padded samples to affect the result."""
    selected = _channel_indices(stream)
    lengths = (
        np.asarray(stream.lengths, dtype=np.int64)
        if stream.lengths is not None
        else np.full(stream.n_windows, stream.windows.shape[1], dtype=np.int64)
    )
    features = np.empty((stream.n_windows, HIDDEN), dtype=np.float32)

    # Grouping by valid length keeps resampling vectorized while respecting short tails.
    for valid_length in np.unique(lengths):
        rows = np.flatnonzero(lengths == valid_length)
        source = np.asarray(stream.windows[rows, :valid_length, :][:, :, selected], dtype=np.float32)
        target_length = max(1, int(round(int(valid_length) * TARGET_HZ / float(stream.rate_hz))))
        resampled = _resample(source, stream.rate_hz, target_length)

        clip_count = int(math.ceil(target_length / SEQ_LEN))
        padded_length = clip_count * SEQ_LEN
        if padded_length != target_length:
            resampled = np.pad(
                resampled, ((0, 0), (0, padded_length - target_length), (0, 0)), mode="edge",
            )
        clips = resampled.reshape(len(rows), clip_count, SEQ_LEN, FEATURE_NUM)
        clip_features = _encode_clips(backbone, clips.reshape(-1, SEQ_LEN, FEATURE_NUM), device)
        clip_features = clip_features.reshape(len(rows), clip_count, HIDDEN)

        weights = np.ones(clip_count, dtype=np.float32)
        remainder = target_length % SEQ_LEN
        if remainder:
            weights[-1] = remainder / SEQ_LEN
        features[rows] = (clip_features * weights[None, :, None]).sum(1) / weights.sum()
    return features


@register
class LiMUBERTXAdapter(BaselineAdapter):
    """Author-released LiMU-BERT-X encoder; no native open-label classifier."""

    name = "limubert_x"
    tier = "representation"
    contract = InputContract(channels=SIX_CHANNELS, rate_hz=TARGET_HZ, native_window_sec=1.0)

    def setup(self, device):
        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Official LiMU-BERT-X checkpoint missing: {CHECKPOINT}. Fetch and verify it with "
                "`python -m baselines.limubert_x.fetch`."
            )
        model = _Backbone().to(device)
        state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False).eval()
        return {"backbone": model}

    def window_features(self, stream, state, device) -> np.ndarray:
        return _window_features(stream, state["backbone"], device)

    def evaluation_artifacts(self, state):
        return {"released_checkpoint": CHECKPOINT}

    def feature_config(self, state):
        return {
            "released_config": "base_v4",
            "input_rate_hz": TARGET_HZ,
            "input_samples_per_clip": SEQ_LEN,
            "input_channels": list(SIX_CHANNELS),
            "acceleration_units": "g (authors' post-normalization convention)",
            "gyroscope_units": "rad/s",
            "feature_layer": "transformer hidden states",
            "within_clip_pool": "mean",
            "across_clip_pool": "duration-weighted mean",
        }
