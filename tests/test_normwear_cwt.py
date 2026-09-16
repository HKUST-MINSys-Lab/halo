from __future__ import annotations

import numpy as np
import torch

from baselines.normwear.adapter import _released_calc_cwt, _released_cwt_torch


def _ricker_numpy(points: int, width: float) -> np.ndarray:
    positions = np.arange(points, dtype=np.float64) - (points - 1.0) / 2.0
    amplitude = 2.0 / (np.sqrt(3.0 * width) * np.pi ** 0.25)
    ratio = positions**2 / width**2
    return amplitude * (1.0 - ratio) * np.exp(-0.5 * ratio)


def _reference(values: np.ndarray) -> np.ndarray:
    output = []
    for scale in (0.1 + np.arange(65)):
        points = max(1, min(int(10.0 * scale), len(values)))
        output.append(np.convolve(values, _ricker_numpy(points, scale), mode="same"))
    return np.stack(output, axis=1)


def test_released_cwt_matches_variable_length_reference_and_shape():
    rng = np.random.default_rng(9)
    values = rng.normal(size=(2, 47)).astype(np.float32)
    actual = _released_cwt_torch(torch.from_numpy(values)).numpy()
    expected = np.stack([_reference(row) for row in values])
    assert actual.shape == (2, 47, 65)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_released_calc_cwt_aligns_raw_and_differences():
    rng = np.random.default_rng(10)
    values = rng.normal(size=(2, 3, 50)).astype(np.float32)
    actual = _released_calc_cwt(None, values)
    assert actual.shape == (2, 3, 3, 48, 65)
    assert torch.isfinite(actual).all()


def test_released_cwt_cuda_autocast_preserves_declared_dtype():
    if not torch.cuda.is_available():
        return
    values = torch.randn(2, 47, device="cuda", dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = _released_cwt_torch(values)
    assert actual.dtype == torch.float32
    assert actual.shape == (2, 47, 65)
    assert torch.isfinite(actual).all()
