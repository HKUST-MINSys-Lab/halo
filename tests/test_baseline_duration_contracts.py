"""Fast contract tests for 4/8/16-second evidence budgets."""

import numpy as np
import pytest
import torch
from dataclasses import replace

from baselines.data import EvalStream
from baselines.harnet.adapter import HarnetAdapter
from baselines.limubert_x.adapter import _window_features as limubert_features
from baselines.normwear.adapter import WINDOW_65, _normwear_chunks
from baselines.unimts.adapter import UniMTSAdapter


def _stream(seconds: int, rate: int, *, channels: int = 6) -> EvalStream:
    names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"][:channels]
    values = np.linspace(0.1, 1.0, seconds * rate * channels, dtype=np.float32)
    values = values.reshape(1, seconds * rate, channels)
    return EvalStream(
        dataset="toy", stream="phone_waist", alignment="native", windows=values,
        gt=["walk"], subjects=np.asarray(["s0"]), channels=names, rate_hz=float(rate),
        mask=np.ones(channels, dtype=bool), eval_labels=["walk"],
        event_ids=np.asarray(["e0"], dtype=object), execution_ids=np.asarray(["e0"], dtype=object),
        lengths=np.asarray([seconds * rate]), window_seconds=float(seconds),
    )


class _HARNet(torch.nn.Module):
    def feature_extractor(self, value):
        return torch.stack((value.mean((1, 2)), value.square().mean((1, 2))), dim=1).unsqueeze(-1)


class _LiMU(torch.nn.Module):
    def forward(self, value):
        # The released trunk returns one hidden state per sample.
        mean = value.mean(dim=-1, keepdim=True)
        return mean.expand(-1, -1, 72)


class _UniMTS:
    def __init__(self):
        self.rows = 0

    def encode_image(self, value):
        self.rows += len(value)
        mean = value.mean((1, 2, 3, 4))
        return torch.stack((mean, torch.ones_like(mean)), dim=1)


@pytest.mark.parametrize("seconds", [4, 8, 16])
def test_released_adapters_consume_complete_requested_duration(seconds):
    harnet = HarnetAdapter().window_features(
        _stream(seconds, 30, channels=3), {"model": _HARNet()}, torch.device("cpu"),
    )
    limubert = limubert_features(_stream(seconds, 100), _LiMU(), torch.device("cpu"))
    unimts_model = _UniMTS()
    unimts = UniMTSAdapter().window_embeddings(
        _stream(seconds, 20, channels=3), {"model": unimts_model}, torch.device("cpu"),
    )
    normwear, owners, _, _ = _normwear_chunks(_stream(seconds, 65))

    assert harnet.shape == (1, 2)
    assert limubert.shape == (1, 72)
    assert unimts.shape == (1, 2)
    # UniMTS accepts the full requested interval in one native forward pass.
    assert unimts_model.rows == 1
    assert len(normwear) == len(owners) == int(np.ceil(seconds * 65 / WINDOW_65))


def test_normwear_grouped_preprocessing_matches_individual_rows():
    base = _stream(8, 50)
    windows = np.concatenate([base.windows, base.windows * 1.7, base.windows * 0.6], axis=0)
    lengths = np.asarray([400, 317, 400], dtype=np.int64)
    stream = replace(
        base, windows=windows, gt=["walk"] * 3, subjects=np.asarray(["a", "b", "c"]),
        event_ids=np.asarray(["e0", "e1", "e2"], dtype=object),
        execution_ids=np.asarray(["e0", "e1", "e2"], dtype=object), lengths=lengths,
    )
    grouped, owners, weights, channels = _normwear_chunks(stream)
    assert channels == 6
    for row in range(3):
        single = replace(
            stream, windows=windows[row:row + 1], gt=["walk"], subjects=stream.subjects[row:row + 1],
            event_ids=stream.event_ids[row:row + 1], execution_ids=stream.execution_ids[row:row + 1],
            lengths=lengths[row:row + 1],
        )
        expected, _, expected_weights, _ = _normwear_chunks(single)
        np.testing.assert_allclose(grouped[owners == row], expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(weights[owners == row], expected_weights)
