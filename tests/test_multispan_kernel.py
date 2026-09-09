"""The multi-span tokenizer must keep the rate contract per span group and own its token grid.

Design of record: docs/design/CONTINUOUS_KERNEL_FRONTEND.md, "Multi-span tokenization". Each test
pins one property of the token grid (layout, metadata, rate comparability, reversal equivariance,
independence from the collate's patch grid) or of the encoder integration.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from model.tokenizer.multispan_kernel import MultiSpanKernelTokenizer


def _band_limited_signal(rate_hz: float, duration_s: float = 6.0, seed: int = 0) -> np.ndarray:
    """One physical signal recorded as a real ADC would: generate high, anti-alias decimate."""
    from scipy.signal import resample_poly

    high = 400.0
    t = np.arange(int(duration_s * high)) / high
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t)
    for freq, amp in ((1.3, 1.0), (2.0, 0.7), (4.5, 0.4)):
        x += amp * np.sin(2 * math.pi * freq * t + rng.uniform(0, 2 * math.pi))
    ratio = Fraction(rate_hz / high).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)


def _as_patches(signal: np.ndarray, rate_hz: float, patch_seconds: float = 1.0):
    per = int(round(rate_hz * patch_seconds))
    n_patches = len(signal) // per
    trimmed = signal[: n_patches * per].reshape(n_patches, per, 1)
    patches = torch.from_numpy(trimmed).unsqueeze(0).float()
    lengths = torch.full((1, n_patches), per, dtype=torch.long)
    mask = torch.ones(1, n_patches, dtype=torch.bool)
    return patches, lengths, mask


@pytest.fixture(scope="module")
def tokenizer():
    torch.manual_seed(0)
    return MultiSpanKernelTokenizer(d_model=32).eval()


def _grid(module, rate, **kw):
    patches, lengths, mask = _as_patches(_band_limited_signal(rate, **kw), rate)
    with torch.no_grad():
        return module.token_grid(patches, rate, lengths, patch_mask=mask)


def _analysis(module, rate, **kw):
    patches, lengths, mask = _as_patches(_band_limited_signal(rate, **kw), rate)
    with torch.no_grad():
        return module.analyze_grid(patches, rate, lengths, patch_mask=mask)


# ---------------------------------------------------------------------------------------------
# bank and grid layout
# ---------------------------------------------------------------------------------------------
def test_bank_is_one_gabor_per_span_harmonic(tokenizer):
    assert tokenizer.span_list == [0.25, 0.5, 1.0, 2.0]
    assert tokenizer.group_sizes == [3, 7, 12, 12]           # capped at 15 Hz and harmonic 12
    assert tokenizer.K == 34
    assert torch.allclose(tokenizer.centres, tokenizer.carrier.float() / tokenizer.spans)
    assert float(tokenizer.centres.max()) <= 15.0
    pairs = {(float(s), int(c)) for s, c in zip(tokenizer.spans, tokenizer.carrier)}
    assert len(pairs) == tokenizer.K
    # the same frequency is measured at several spans: 4 Hz sits in every group
    assert sum(1 for c in tokenizer.centres.tolist() if abs(c - 4.0) < 1e-6) == 4
    assert [float(tokenizer.centres[tokenizer.group_slice(g)].max())
            for g in range(tokenizer.G)] == [12.0, 14.0, 12.0, 6.0]


def test_token_grid_layout_follows_duration_not_rate(tokenizer):
    shapes = set()
    for rate in (20.0, 50.0, 100.0):
        grid = _grid(tokenizer, rate)
        shapes.add(tuple(grid["tokens"].shape))
        assert grid["tokens"].shape[1] == 96 + 48 + 24 + 12
        assert bool(grid["token_mask"].all())
        for g, (span, count) in enumerate(zip(tokenizer.span_list, (96, 48, 24, 12))):
            rows = grid["resolution_ids"][0] == g
            assert int(rows.sum()) == count
            assert torch.allclose(grid["durations"][0][rows], torch.full((count,), span))
            positions = grid["positions"][0][rows]
            stride = span / tokenizer.frames_per_span
            assert positions[0] == pytest.approx(0.5 * stride)
            assert positions[-1] == pytest.approx(6.0 - 0.5 * stride)
    assert len(shapes) == 1, f"token grid varies with sampling rate: {shapes}"


def test_batch_composition_does_not_change_a_recordings_tokens(tokenizer):
    """A 4 s recording padded next to a 6 s one must get the tokens it gets alone; the extra
    frames are masked, never analysed as signal."""
    rate = 50.0
    short = torch.from_numpy(_band_limited_signal(rate, duration_s=4.0, seed=3)).view(1, 4, 50, 1)
    long = torch.from_numpy(_band_limited_signal(rate, duration_s=6.0, seed=4)).view(1, 6, 50, 1)
    patches = torch.zeros(2, 6, 50, 1)
    patches[0, :4] = short[0]
    patches[1] = long[0]
    lengths = torch.tensor([[50, 50, 50, 50, 0, 0], [50] * 6])
    mask = lengths > 0
    with torch.no_grad():
        together = tokenizer.token_grid(patches, rate, lengths, patch_mask=mask)
        alone = tokenizer.token_grid(short, rate, torch.full((1, 4), 50), patch_mask=torch.ones(1, 4, dtype=torch.bool))
    assert not bool(together["token_mask"][0].all())
    assert bool(together["token_mask"][1].all())
    for g in range(tokenizer.G):
        rows = (together["resolution_ids"][0] == g) & together["token_mask"][0]
        alone_rows = alone["resolution_ids"][0] == g
        assert int(rows.sum()) == int(alone_rows.sum())
        assert torch.allclose(together["tokens"][0][rows], alone["tokens"][0][alone_rows], atol=1e-5)
        assert torch.allclose(together["positions"][0][rows], alone["positions"][0][alone_rows])


def test_collate_patch_grid_is_irrelevant(tokenizer):
    """One-second and 1.5 s patch layouts of the same recording give the same tokens: the module
    rebuilds the contiguous window and lays its own frames out in physical time."""
    rate = 50.0
    signal = _band_limited_signal(rate)
    one = _as_patches(signal, rate, 1.0)
    one_and_half = _as_patches(signal, rate, 1.5)
    with torch.no_grad():
        a = tokenizer.token_grid(one[0], rate, one[1], patch_mask=one[2])
        b = tokenizer.token_grid(one_and_half[0], rate, one_and_half[1], patch_mask=one_and_half[2])
    assert torch.allclose(a["tokens"], b["tokens"], atol=1e-5)
    assert torch.equal(a["positions"], b["positions"])


# ---------------------------------------------------------------------------------------------
# the rate contract, per span group
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rate", [20.0, 25.0, 50.0])
def test_group_magnitudes_agree_across_sampling_rates(tokenizer, rate):
    reference = _analysis(tokenizer, 100.0)
    candidate = _analysis(tokenizer, rate)
    live = tokenizer.masks(rate, torch.tensor([6.0]))[0][0] >= 1.0
    assert bool(live.any())
    for g in range(tokenizer.G):
        sl = tokenizer.group_slice(g)
        keep = live[sl]
        if not bool(keep.any()):
            continue
        span = tokenizer.span_list[g]
        frame_time = reference["groups"][g]["frame_time"]
        interior = (frame_time >= span) & (frame_time <= 6.0 - span)
        a = reference["groups"][g]["compressed"][0, 0][keep][:, interior].flatten().double()
        b = candidate["groups"][g]["compressed"][0, 0][keep][:, interior].flatten().double()
        corr = float(np.corrcoef(a.numpy(), b.numpy())[0, 1])
        assert corr > 0.97, f"group {g} ({span} s) at {rate} Hz: correlation {corr:.4f}"


def test_response_magnitude_is_not_a_function_of_sampling_rate(tokenizer):
    low = _analysis(tokenizer, 20.0)
    high = _analysis(tokenizer, 100.0)
    live = tokenizer.masks(20.0, torch.tensor([6.0]))[0][0] >= 1.0
    ratios = []
    for g in range(tokenizer.G):
        keep = live[tokenizer.group_slice(g)]
        if bool(keep.any()):
            a = low["groups"][g]["compressed"][0, 0][keep].abs().mean()
            b = high["groups"][g]["compressed"][0, 0][keep].abs().mean()
            ratios.append(float(a / b.clamp_min(1e-9)))
    assert all(0.6 < r < 1.6 for r in ratios), ratios


def test_observability_is_binary_at_initialisation(tokenizer):
    for rate in (20.0, 25.0, 50.0, 100.0):
        nyq, _ = tokenizer.masks(rate, torch.tensor([6.0]))
        carrier_live = tokenizer.centres <= tokenizer.nyquist_margin * rate / 2
        assert torch.equal(nyq[0] > 0.5, carrier_live)
        assert bool(((nyq[0] == 0) | (nyq[0] == 1)).all())


def test_kernels_are_zero_mean(tokenizer):
    offsets = torch.linspace(-1.0, 1.0, 2001)
    with torch.no_grad():
        kernels = tokenizer.kernel_at(offsets, 1000.0)                 # (K, 2, n)
    assert float(kernels.sum(-1).abs().max()) < 1e-4


# ---------------------------------------------------------------------------------------------
# what the grid keeps that a per-patch summary loses
# ---------------------------------------------------------------------------------------------
def test_time_reversal_is_equivariant_on_the_grid(tokenizer):
    """Reversing the recording reverses each group's token sequence (symmetric kernels, symmetric
    frame grid) rather than leaving the tokens unchanged. Order therefore survives into attention
    through the positions, where the fixed filterbank's per-patch energies are reversal-invariant."""
    rate = 50.0
    patches, lengths, mask = _as_patches(_band_limited_signal(rate, seed=5), rate)
    reversed_patches = patches.flip(1).flip(2)
    with torch.no_grad():
        forward = tokenizer.token_grid(patches, rate, lengths, patch_mask=mask)
        backward = tokenizer.token_grid(reversed_patches, rate, lengths, patch_mask=mask)
    # A reversed recording mirrors about its LAST SAMPLE, not the window end, so each mirrored
    # frame sits one sample (20 ms) off the original's; the kernel absorbs that as sub-sample
    # jitter. Equivariance therefore holds to a few percent, while the un-flipped comparison
    # must be far worse or the tokens carry no order at all.
    for g in range(tokenizer.G):
        rows = forward["resolution_ids"][0] == g
        a = forward["tokens"][0][rows]
        b = backward["tokens"][0][rows]
        flipped = float((a - b.flip(0)).norm() / a.norm())
        unflipped = float((a - b).norm() / a.norm())
        assert flipped < 0.08, f"group {g}: reversed-and-flipped differs by {flipped:.3f}"
        assert unflipped > 3 * flipped, (
            f"group {g}: tokens carry no temporal order ({unflipped:.3f} vs {flipped:.3f})")


def test_gradients_reach_every_parameter(tokenizer):
    module = MultiSpanKernelTokenizer(d_model=16).train()
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0), 50.0)
    grid = module.token_grid(patches, 50.0, lengths, patch_mask=mask)
    grid["tokens"].square().mean().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient is not finite"
        assert float(parameter.grad.abs().sum()) > 0, f"{name} gradient is identically zero"


def test_norm_statistics_round_trip_and_ignore_absent_channels():
    torch.manual_seed(0)
    reference = MultiSpanKernelTokenizer(d_model=16).eval()
    masked = MultiSpanKernelTokenizer(d_model=16).eval()
    masked.load_state_dict(reference.state_dict())
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0), 50.0)
    reference.fit_norm_stats(patches, 50.0, lengths, patch_mask=mask,
                             channel_mask=torch.ones(1, 1, dtype=torch.bool))
    six = torch.randn(1, 6, 50, 6) * 100.0
    six[..., 0] = patches[..., 0]
    masked.fit_norm_stats(six, 50.0, lengths, patch_mask=mask,
                          channel_mask=torch.tensor([[True, False, False, False, False, False]]))
    assert bool(reference._norm_fitted)
    assert torch.allclose(reference.norm_mu, masked.norm_mu, atol=1e-6)
    assert torch.allclose(reference.norm_sd, masked.norm_sd, atol=1e-6)
    assert torch.allclose(reference.amp_mu, masked.amp_mu, atol=1e-6)
    assert torch.allclose(reference.dc_mu, masked.dc_mu, atol=1e-6)
    assert bool((reference.norm_sd > 1e-5).all())


def test_missing_axes_are_marked_not_invented(tokenizer):
    torch.manual_seed(0)
    patches = torch.randn(1, 6, 50, 6)
    lengths = torch.full((1, 6), 50, dtype=torch.long)
    sensor_id = torch.tensor([[0, 0, 0, 1, 1, 1]])
    full = torch.ones(1, 6, dtype=torch.bool)
    accel_only = torch.tensor([[True, True, True, False, False, False]])
    with torch.no_grad():
        a = tokenizer.token_grid(patches, 50.0, lengths, sensor_id=sensor_id,
                                 channel_mask=full, n_sensors=2)["tokens"]
        b = tokenizer.token_grid(patches, 50.0, lengths, sensor_id=sensor_id,
                                 channel_mask=accel_only, n_sensors=2)["tokens"]
    assert a.shape == (1, 180, 2, 32) and torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.allclose(a[:, :, 0], b[:, :, 0], atol=1e-6), "the accelerometer token changed"
    assert not torch.allclose(a[:, :, 1], b[:, :, 1]), "an absent gyroscope produced live tokens"


def test_local_summaries_are_standardized_separately_per_span():
    module = MultiSpanKernelTokenizer(d_model=16)
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0), 50.0)
    module.fit_norm_stats(patches, 50.0, lengths, patch_mask=mask)
    analysis = module.analyze_grid(patches, 50.0, lengths, patch_mask=mask)
    for g, group in enumerate(analysis["groups"]):
        for name, key in (("amp", "amplitude"), ("dc", "dc")):
            values = group[key][..., group["valid"][0]]
            standardized = (values - getattr(module, name + "_mu")[g]) / getattr(module, name + "_sd")[g]
            assert abs(float(standardized.mean())) < 1e-5
            assert float(standardized.std(unbiased=False)) == pytest.approx(1.0, abs=1e-5)
    restored = MultiSpanKernelTokenizer(d_model=16)
    restored.load_state_dict(module.state_dict())
    assert torch.equal(restored.amp_sd, module.amp_sd)
    # With no live channels, each group's fallback must stay finite and neutral.
    module.fit_norm_stats(patches, 50.0, lengths, patch_mask=mask,
                          channel_mask=torch.zeros(1, 1, dtype=torch.bool))
    assert torch.equal(module.dc_mu, torch.zeros(module.G))
    assert torch.equal(module.dc_sd, torch.ones(module.G))


@pytest.mark.parametrize("frontend", ["continuous", "multispan"])
def test_legacy_analysis_cannot_silently_load_with_new_math(frontend):
    from model.tokenizer.continuous_kernel import ContinuousKernelTokenizer

    cls = ContinuousKernelTokenizer if frontend == "continuous" else MultiSpanKernelTokenizer
    module = cls(d_model=16)
    legacy = dict(module.state_dict())
    del legacy["_frontend_revision"]
    with pytest.raises(RuntimeError, match="saved source revision"):
        module.load_state_dict(legacy, strict=False)


def test_encoder_rejects_multiple_input_grids():
    encoder = _encoder()
    patches = torch.randn(1, 6, 50, 6)
    with pytest.raises(ValueError, match="single input patch grid"):
        encoder(patches, torch.tensor([50.0]), torch.full((1, 6), 50),
                [["x", "y", "z"] * 2], torch.zeros(1, 6),
                resolution_ids=torch.tensor([[0, 0, 0, 1, 1, 1]]),
                sensor_texts=[["accelerometer on wrist", "gyroscope on wrist"]],
                sensor_id=torch.tensor([[0, 0, 0, 1, 1, 1]]))


# ---------------------------------------------------------------------------------------------
# encoder integration
# ---------------------------------------------------------------------------------------------
def _encoder(**overrides):
    from model.tokenizer.encoder import SetTokenizerEncoder

    torch.manual_seed(0)
    kwargs = dict(
        d_model=32, num_layers=1, num_heads=4, dim_feedforward=64, dropout=0.0,
        frontend="multispan", trunk="temporal", descriptor_prediction=False,
        text_conditioning="factored", token_granularity="sensor",
        use_duration_embedding=True, duration_min_seconds=0.25, duration_max_seconds=2.0,
        num_resolutions=4, rope_min_period=0.125,
    )
    kwargs.update(overrides)
    return SetTokenizerEncoder(**kwargs).eval()


def _forward(encoder, patches, rate=50.0):
    B, P, S, C = patches.shape
    lengths = torch.full((B, P), S, dtype=torch.long)
    return encoder(
        patches, torch.full((B,), rate), lengths,
        [["x", "y", "z"] * 2] * B,
        torch.zeros(B, P),
        patch_durations=torch.ones(B, P), resolution_ids=None,
        channel_mask=torch.ones(B, C, dtype=torch.bool),
        patch_padding_mask=torch.ones(B, P, dtype=torch.bool),
        sensor_texts=[["accelerometer on the wrist", "gyroscope on the wrist"]] * B,
        sensor_id=torch.tensor([[0, 0, 0, 1, 1, 1]] * B),
        source_rate_hz=torch.full((B,), rate),
    )


def test_encoder_uses_the_frontends_token_grid():
    from model.tokenizer.multispan_kernel import MultiSpanKernelTokenizer as Tokenizer

    encoder = _encoder()
    assert isinstance(encoder.filterbank, Tokenizer)
    patches = torch.randn(2, 6, 50, 6)
    with torch.no_grad():
        out = _forward(encoder, patches)
    assert out["pooled"].shape == (2, 32)
    assert out["per_patch"].shape == (2, 180, 32)
    assert out["token_grid"]["token_mask"].shape == (2, 180)
    assert torch.isfinite(out["pooled"]).all()


def test_pooled_representation_is_sensitive_to_temporal_order():
    """Mean pooling is order-invariant on its own; RoPE inside attention is what lets the encoder
    see the order the grid preserved. Reversal must therefore change the pooled vector."""
    encoder = _encoder()
    signal = torch.from_numpy(_band_limited_signal(50.0, seed=7)).view(1, 6, 50, 1)
    patches = signal.repeat(1, 1, 1, 6) + 0.1 * torch.randn(1, 6, 50, 6)
    with torch.no_grad():
        forward = _forward(encoder, patches)["pooled"]
        backward = _forward(encoder, patches.flip(1).flip(2))["pooled"]
    assert not torch.allclose(forward, backward, atol=1e-3)


def test_encoder_refuses_jepa_masking_on_the_grid():
    encoder = _encoder()
    patches = torch.randn(1, 6, 50, 6)
    with pytest.raises(ValueError, match="token masking"):
        B, P, S, C = patches.shape
        encoder(
            patches, torch.full((B,), 50.0), torch.full((B, P), S, dtype=torch.long),
            [["x", "y", "z"] * 2], torch.zeros(B, P), patch_durations=torch.ones(B, P),
            token_mask=torch.zeros(B, P, 2, dtype=torch.bool),
            channel_mask=torch.ones(B, C, dtype=torch.bool),
            patch_padding_mask=torch.ones(B, P, dtype=torch.bool),
            sensor_texts=[["accelerometer on the wrist", "gyroscope on the wrist"]],
            sensor_id=torch.tensor([[0, 0, 0, 1, 1, 1]]),
        )
