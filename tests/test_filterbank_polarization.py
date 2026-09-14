"""Contracts for fixed-filterbank triad polarization features."""

import math

import torch

from model.tokenizer.filterbank import PhysicalFilterbankTokenizer


K = 16
S = 512


def _tokenizer(**kwargs):
    return PhysicalFilterbankTokenizer(
        d_model=24, n_bands=K, dft_size=S, norm="none", use_polarization=True, **kwargs,
    ).eval()


def _ids(batch=1):
    return torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long).expand(batch, -1).clone()


def _polarization(analysis, tok):
    # energy, observability, resolution, amplitude, signed DC precede the optional block.
    start = 3 * tok.n_bands + int(tok.use_amplitude) + int(tok.use_dc)
    return analysis[..., start:]


def _motion(rate=64.0, seconds=4.0, circular=True):
    n = int(rate * seconds)
    t = torch.arange(n, dtype=torch.float32) / rate
    x = torch.zeros(1, 1, S, 6)
    phase = 2 * math.pi * 2.0 * t
    x[0, 0, :n, 0] = torch.cos(phase)
    x[0, 0, :n, 1] = torch.sin(phase) if circular else torch.cos(phase)
    x[0, 0, :n, 2] = 9.81
    # A second live triad verifies gyro shares the raw accelerometer gravity direction.
    x[0, 0, :n, 3] = torch.cos(phase)
    x[0, 0, :n, 4] = torch.sin(phase)
    return x, torch.tensor([rate]), torch.tensor([n])


def test_polarization_shape_neutral_without_sensor_metadata():
    tok = _tokenizer()
    x, rate, length = _motion()
    analysis = tok.analyze(x, rate, length)
    pol = _polarization(analysis, tok)
    assert tok.in_dim == 6 * K + 3
    assert torch.count_nonzero(pol) == 0


def test_circularity_and_linearity_are_distinguished_and_bounded():
    tok = _tokenizer()
    circular, rate, length = _motion(circular=True)
    linear, _, _ = _motion(circular=False)
    c = _polarization(tok.analyze(circular, rate, length, sensor_id=_ids()), tok)[0, 0, 0]
    l = _polarization(tok.analyze(linear, rate, length, sensor_id=_ids()), tok)[0, 0, 0]
    circularity = c[K:2 * K]
    linearity = l[K:2 * K]
    peak = circularity.argmax()
    assert circularity[peak] > 0.8
    assert linearity[peak] < 0.1
    assert c[:K].min() >= 0 and c[:2 * K].max() <= 1
    assert c[2 * K:3 * K].abs().max() <= 1
    assert 0 <= c[-1] <= 1


def test_rotation_invariance_and_time_reversal_spin_sign():
    tok = _tokenizer()
    x, rate, length = _motion()
    original = _polarization(tok.analyze(x, rate, length, sensor_id=_ids()), tok)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=x.dtype,
    )
    rotated = x.clone()
    rotated[..., :3] = x[..., :3] @ rotation.T
    rotated[..., 3:6] = x[..., 3:6] @ rotation.T
    rotated_pol = _polarization(tok.analyze(rotated, rate, length, sensor_id=_ids()), tok)
    assert torch.allclose(original, rotated_pol, rtol=2e-3, atol=2e-3)

    reversed_x = x.clone()
    valid = int(length.item())
    reversed_x[:, :, :valid] = x[:, :, :valid].flip(2)
    reversed_pol = _polarization(tok.analyze(reversed_x, rate, length, sensor_id=_ids()), tok)
    assert torch.allclose(original[..., :2 * K], reversed_pol[..., :2 * K], rtol=3e-3, atol=3e-3)
    assert torch.allclose(original[..., 2 * K:3 * K], -reversed_pol[..., 2 * K:3 * K],
                          rtol=3e-3, atol=3e-3)


def test_gravity_gate_uses_raw_accelerometer_dc_and_absent_triad_is_neutral():
    tok = _tokenizer()
    x, rate, length = _motion()
    centered = x.clone()
    valid = int(length.item())
    centered[:, :, :valid, :3] -= centered[:, :, :valid, :3].mean(dim=2, keepdim=True)
    normal = _polarization(tok.analyze(x, rate, length, sensor_id=_ids()), tok)
    no_gravity = _polarization(tok.analyze(centered, rate, length, sensor_id=_ids()), tok)
    assert normal[..., -1].mean() > 0.9
    assert no_gravity[..., -1].max() < 0.02
    assert no_gravity[..., :K].abs().max() < 0.02
    assert no_gravity[..., 2 * K:3 * K].abs().max() < 0.02
    # A missing gyro axis must not be treated as a zero-valued physical axis.
    mask = torch.tensor([[True, True, True, True, True, False]])
    masked = _polarization(tok.analyze(x, rate, length, sensor_id=_ids(), channel_mask=mask), tok)
    assert torch.count_nonzero(masked[..., 3:6, :]) == 0


def test_silent_and_degenerate_inputs_remain_finite_and_neutral():
    tok = _tokenizer()
    x = torch.zeros(1, 1, S, 6)
    result = _polarization(tok.analyze(x, torch.tensor([50.0]), torch.tensor([1]), sensor_id=_ids()), tok)
    assert torch.isfinite(result).all()
    assert result[..., :3 * K].abs().max() == 0
    assert result[..., -1].max() < 0.02


def test_rate_invariance_for_a_physical_circular_motion():
    tok = _tokenizer()
    values = []
    for rate in (32.0, 64.0, 128.0):
        x, r, length = _motion(rate=rate, seconds=3.0)
        pol = _polarization(tok.analyze(x, r, length, sensor_id=_ids()), tok)[0, 0, 0, K:2 * K]
        values.append(pol)
    for other in values[1:]:
        assert torch.nn.functional.cosine_similarity(values[0], other, dim=0) > 0.98
