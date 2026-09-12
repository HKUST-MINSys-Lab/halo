from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines import REGISTRY
from baselines.data import EvalStream


CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def _stream(windows: np.ndarray, *, rate_hz: float = 20.0, lengths=None, mask=None) -> EvalStream:
    count = len(windows)
    return EvalStream(
        dataset="synthetic",
        stream="phone_pocket",
        alignment="native",
        windows=windows.astype(np.float32),
        gt=["walking"] * count,
        subjects=np.asarray(["s1"] * count),
        channels=list(CHANNELS),
        rate_hz=rate_hz,
        mask=np.ones(6, dtype=bool) if mask is None else np.asarray(mask, dtype=bool),
        eval_labels=["walking"],
        lengths=(np.full(count, windows.shape[1], dtype=np.int64)
                 if lengths is None else np.asarray(lengths, dtype=np.int64)),
    )


@pytest.fixture(scope="module")
def adapter_state():
    adapter = REGISTRY["limubert_x"]
    return adapter, adapter.setup(torch.device("cpu"))


def test_released_adapter_contract(adapter_state):
    adapter, state = adapter_state
    assert adapter.supports_native_zero_shot() is False
    assert adapter.contract.rate_hz == 20.0
    assert tuple(adapter.contract.channels) == tuple(CHANNELS)
    assert sum(parameter.numel() for parameter in state["backbone"].parameters()) == 55_446


def test_complete_window_is_duration_weighted_clip_mean(adapter_state):
    adapter, state = adapter_state
    rng = np.random.default_rng(5)
    samples = rng.normal(size=(1, 40, 6)).astype(np.float32)

    full = adapter.window_features(_stream(samples), state, torch.device("cpu"))
    clips = adapter.window_features(
        _stream(np.concatenate([samples[:, :20], samples[:, 20:]], axis=0)),
        state,
        torch.device("cpu"),
    )
    np.testing.assert_allclose(full[0], clips.mean(axis=0), rtol=2e-5, atol=2e-6)


def test_partial_clip_uses_valid_duration_weight(adapter_state):
    adapter, state = adapter_state
    rng = np.random.default_rng(6)
    samples = rng.normal(size=(1, 30, 6)).astype(np.float32)

    full = adapter.window_features(_stream(samples), state, torch.device("cpu"))[0]
    first = adapter.window_features(_stream(samples[:, :20]), state, torch.device("cpu"))[0]
    tail = adapter.window_features(_stream(samples[:, 20:]), state, torch.device("cpu"))[0]
    expected = (first + 0.5 * tail) / 1.5
    np.testing.assert_allclose(full, expected, rtol=2e-5, atol=2e-6)


def test_resampling_is_finite_and_masked_channel_fails(adapter_state):
    adapter, state = adapter_state
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(2, 300, 6)).astype(np.float32)
    features = adapter.window_features(
        _stream(samples, rate_hz=50.0), state, torch.device("cpu"),
    )
    assert features.shape == (2, 72)
    assert np.isfinite(features).all()

    with pytest.raises(ValueError, match="masked channels"):
        adapter.window_features(
            _stream(samples, rate_hz=50.0, mask=[1, 1, 1, 0, 0, 0]),
            state,
            torch.device("cpu"),
        )
