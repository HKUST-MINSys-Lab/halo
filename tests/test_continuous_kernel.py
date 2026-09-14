"""The continuous-kernel tokenizer must be RATE-COMPARABLE, or it has no reason to exist.

Design of record: docs/design/CONTINUOUS_KERNEL_FRONTEND.md. Every test here pins one property that,
if broken, makes the front end silently behave differently on a 20 Hz phone than on a 100 Hz watch —
which is the failure the module exists to prevent, and which a loss curve would never reveal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from model.tokenizer.continuous_kernel import ContinuousKernelTokenizer


def _band_limited_signal(rate_hz: float, duration_s: float = 6.0, seed: int = 0) -> np.ndarray:
    """One physical signal, sampled as a real ADC would: generate high, anti-alias DECIMATE.

    Point-sampling a high-rate signal is NOT the same as recording at a low rate; without the
    decimation this fixture would compare aliased garbage and the test would be meaningless.
    """
    from fractions import Fraction

    from scipy.signal import resample_poly

    high = 400.0
    t = np.arange(int(duration_s * high)) / high
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t)
    for freq, amp in ((1.3, 1.0), (2.0, 0.7), (4.5, 0.4)):      # all below 9 Hz => 20 Hz-safe
        x += amp * np.sin(2 * math.pi * freq * t + rng.uniform(0, 2 * math.pi))
    ratio = Fraction(rate_hz / high).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)


def _as_patches(signal: np.ndarray, rate_hz: float, patch_seconds: float = 1.0):
    """(1, P, S, C=1) padded patches + patch_len + mask, the loader's layout."""
    per = int(round(rate_hz * patch_seconds))
    n_patches = len(signal) // per
    trimmed = signal[: n_patches * per].reshape(n_patches, per, 1)
    patches = torch.from_numpy(trimmed).unsqueeze(0).float()
    lengths = torch.full((1, n_patches), per, dtype=torch.long)
    mask = torch.ones(1, n_patches, dtype=torch.bool)
    return patches, lengths, mask


def _analyze(module, rate_hz, **kw):
    patches, lengths, mask = _as_patches(_band_limited_signal(rate_hz, **kw), rate_hz)
    return module.analyze(patches, rate_hz, lengths, patch_mask=mask)


@pytest.fixture(scope="module")
def tokenizer():
    torch.manual_seed(0)
    return ContinuousKernelTokenizer().eval()


# ---------------------------------------------------------------------------------------------
# 1. THE test — the same physical signal must produce the same features at any sampling rate
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rate", [20.0, 25.0, 50.0])
def test_band_magnitudes_agree_across_sampling_rates(tokenizer, rate):
    """The property the whole design exists for.

    An earlier draft rounded each output frame's centre to the nearest input sample. At 20 Hz that
    is 25 ms of jitter — 45 degrees of phase error on a 5 Hz component — and it dropped correlation
    to 0.844. The kernel must instead be evaluated at the EXACT offsets of the real samples so it
    absorbs the sub-sample shift.
    """
    with torch.no_grad():
        reference = _analyze(tokenizer, 100.0)["compressed"]
        candidate = _analyze(tokenizer, rate)["compressed"]
    # only kernels whose harmonics this rate can represent are comparable; the rest are, correctly,
    # a different filter (and the nyquist mask says so).
    live = tokenizer.masks(rate, torch.tensor([6.0]))[0][0] >= 1.0
    assert bool(live.any())
    a = reference[..., live, :].flatten().double().numpy()
    b = candidate[..., live, :].flatten().double().numpy()
    n = min(len(a), len(b))
    correlation = float(np.corrcoef(a[:n], b[:n])[0, 1])
    assert correlation > 0.95, f"rate {rate}: correlation {correlation:.4f}"


@pytest.mark.parametrize("rate", [20.0, 25.0, 50.0])
def test_final_tokens_remain_comparable_across_sampling_rates(tokenizer, rate):
    reference_patches, reference_len, reference_mask = _as_patches(
        _band_limited_signal(100.0), 100.0)
    candidate_patches, candidate_len, candidate_mask = _as_patches(
        _band_limited_signal(rate), rate)
    with torch.no_grad():
        reference = tokenizer(reference_patches, 100.0, reference_len,
                              patch_mask=reference_mask)
        candidate = tokenizer(candidate_patches, rate, candidate_len,
                              patch_mask=candidate_mask)
    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten(), candidate.flatten(), dim=0)
    assert float(cosine) > 0.95, f"rate {rate}: final-token cosine {float(cosine):.4f}"


def test_token_count_depends_on_duration_not_on_rate(tokenizer):
    """6 s must give the same number of tokens at 20 Hz and at 100 Hz. Shape invariance is half
    the contract; values are the other half."""
    shapes = set()
    for rate in (20.0, 50.0, 100.0):
        patches, lengths, mask = _as_patches(_band_limited_signal(rate), rate)
        with torch.no_grad():
            out = tokenizer(patches, rate, lengths, patch_mask=mask)
        shapes.add(tuple(out.shape))
    assert len(shapes) == 1, f"token shape varies with sampling rate: {shapes}"


@pytest.mark.parametrize("duration", [2.0, 4.0, 6.0])
def test_token_count_scales_with_duration(tokenizer, duration):
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0, duration_s=duration), 50.0)
    with torch.no_grad():
        out = tokenizer(patches, 50.0, lengths, patch_mask=mask)
    assert out.shape[1] == int(duration), f"{duration}s -> {out.shape[1]} tokens"


# ---------------------------------------------------------------------------------------------
# 2. The three rules that make rates comparable
# ---------------------------------------------------------------------------------------------
def test_kernels_are_zero_mean_so_they_cannot_measure_gravity(tokenizer):
    """A DC offset is gravity/tilt, handled by the separate signed-DC feature. If kernels responded
    to it they would re-introduce the orientation confound through the back door."""
    for rate in (20.0, 100.0):
        offsets = (torch.arange(-int(rate), int(rate) + 1, dtype=torch.float32) / rate)
        with torch.no_grad():
            pair = tokenizer.kernel_at(offsets, rate)
        assert torch.allclose(pair.sum(-1), torch.zeros_like(pair.sum(-1)), atol=1e-5)


def test_response_magnitude_is_not_a_function_of_sampling_rate(tokenizer):
    """`dt = 1/r` makes the convolution an integral. Without it a 20 Hz recording returns about one
    FIFTH the response of the same motion at 100 Hz, and the model learns to identify the device."""
    with torch.no_grad():
        low = _analyze(tokenizer, 20.0)["compressed"]
        high = _analyze(tokenizer, 100.0)["compressed"]
    live = tokenizer.masks(20.0, torch.tensor([6.0]))[0][0] >= 1.0
    ratio = float(low[..., live, :].abs().mean() / high[..., live, :].abs().mean().clamp_min(1e-9))
    assert 0.6 < ratio < 1.6, f"magnitude ratio 20/100 Hz = {ratio:.3f}"


def test_harmonics_above_nyquist_are_dropped(tokenizer):
    """Band-limiting is exact: harmonic m sits at m/T_k Hz, so it is a coefficient mask. A kernel
    sampled without it would alias and behave differently at every rate."""
    nyq_low, _ = tokenizer.masks(20.0, torch.tensor([6.0]))
    nyq_high, _ = tokenizer.masks(100.0, torch.tensor([6.0]))
    assert (nyq_low[0] <= nyq_high[0] + 1e-6).all()
    assert nyq_low[0].min() < 1.0, "at 20 Hz some short kernels must lose harmonics"
    assert nyq_high[0].mean() > nyq_low[0].mean()


def test_amplitude_scales_linearly_before_compression(tokenizer):
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0), 50.0)
    with torch.no_grad():
        one = tokenizer.analyze(patches, 50.0, lengths, patch_mask=mask)["compressed"]
        two = tokenizer.analyze(patches * 3.0, 50.0, lengths, patch_mask=mask)["compressed"]
    assert (torch.expm1(two) >= torch.expm1(one) - 1e-4).all()
    ratio = float(torch.expm1(two).mean() / torch.expm1(one).mean().clamp_min(1e-9))
    assert 2.0 < ratio < 4.0, ratio


# ---------------------------------------------------------------------------------------------
# 3. Robustness — the cases that have actually broken this codebase before
# ---------------------------------------------------------------------------------------------
def test_single_sample_window_produces_a_token_and_never_raises(tokenizer):
    """~0.1% of corpus windows carry ONE sample (wisdm, capture24, opportunity). This exact case
    crashed all four arms of the encoder-swap harness on 2026-08-22."""
    patches = torch.randn(2, 6, 50, 3)
    lengths = torch.zeros(2, 6, dtype=torch.long)
    lengths[:, 0] = 1
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 0] = True
    with torch.no_grad():
        out = tokenizer(patches, 50.0, lengths, patch_mask=mask)
    assert torch.isfinite(out).all()


def test_invalid_rate_and_length_fail_loudly(tokenizer):
    patches = torch.randn(1, 2, 20, 1)
    with pytest.raises(ValueError, match="positive rates"):
        tokenizer(patches, 0.0, torch.full((1, 2), 20))
    with pytest.raises(ValueError, match="cannot exceed"):
        tokenizer(patches, 20.0, torch.tensor([[21, 20]]))


def test_padded_region_cannot_leak_into_the_output(tokenizer):
    patches = torch.randn(1, 6, 60, 2)
    lengths = torch.full((1, 6), 60, dtype=torch.long)
    lengths[0, 3:] = 0
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 3:] = False
    with torch.no_grad():
        before = tokenizer(patches, 60.0, lengths, patch_mask=mask)
        patches[0, 3:] = 1e4
        after = tokenizer(patches, 60.0, lengths, patch_mask=mask)
    assert torch.allclose(before, after, atol=1e-4)


def test_gabor_init_is_a_band_pass_bank_not_noise(tokenizer):
    """EVERY kernel must peak at its own centre frequency at initialisation — that is what makes
    step 0 comparable to the physical filterbank instead of a random projection.

    The earlier version probed three kernels at 3x spacing and so never saw that the 2 s span cap
    had collapsed sixteen low kernels onto four; this sweeps all of them at ~3% spacing.
    """
    rate = 100.0
    freqs = torch.exp(torch.linspace(math.log(0.4), math.log(16.0), 120))
    t = torch.arange(int(rate * 6.0)) / rate
    signals = torch.sin(2 * math.pi * freqs.view(-1, 1) * t.view(1, -1))
    patches = signals.view(len(freqs), 6, 100, 1)
    lengths = torch.full((len(freqs), 6), 100, dtype=torch.long)
    with torch.no_grad():
        energy = tokenizer.analyze(patches, rate, lengths)["compressed"][:, 0]     # (n, K, frames)
    fpp = tokenizer.frames_per_patch
    interior = energy[..., 2 * fpp:4 * fpp].mean(-1)          # seconds 2-4: fully supported
    peak = freqs[interior.argmax(0)]
    ratio = (peak / tokenizer.centres).log().abs()
    # A kernel holding ONE cycle is inherently broadband (spectral sd ~0.72 x its centre): its
    # mirror image and the zero-mean constraint push the measured peak ~10-16% above the carrier.
    # That is the honest resolution below 1 Hz in a 2 s span, not a placement error, so those
    # kernels get a looser bound; every multi-cycle kernel must sit within the sweep's spacing.
    tolerance = torch.where(tokenizer.carrier == 1, math.log(1.25), math.log(1.08))
    worst = int((ratio - tolerance).argmax())
    assert bool((ratio < tolerance).all()), (
        f"kernel {worst} (centre {float(tokenizer.centres[worst]):.2f} Hz) peaks at "
        f"{float(peak[worst]):.2f} Hz")


def test_kernel_bank_is_distinct_with_carriers_on_centres(tokenizer):
    """A bank of 32 kernels with only 20 distinct members is 12 wasted slots that learning did not
    recover (measured on the 2026-09-08 continuous run). Spans hold whole cycles so carriers sit
    exactly on the log-spaced centres and no two (span, carrier) pairs coincide."""
    pairs = {(round(float(s), 6), int(c)) for s, c in zip(tokenizer.spans, tokenizer.carrier)}
    assert len(pairs) == tokenizer.K
    carrier_hz = tokenizer.carrier.float() / tokenizer.spans
    assert torch.allclose(carrier_hz, tokenizer.centres, rtol=1e-5)
    assert bool((tokenizer.spans <= 2.0 + 1e-6).all())
    assert int(tokenizer.carrier.min()) >= 1 and int(tokenizer.carrier.max()) <= tokenizer.M


def test_f_min_must_fit_one_cycle_in_the_longest_span():
    with pytest.raises(ValueError, match="one whole cycle"):
        ContinuousKernelTokenizer(f_min=0.3, t_max=2.0)


def test_observability_is_retained_coefficient_energy_not_slot_count():
    """An INTACT kernel must never be rescaled by the sampling rate. The harmonic-slot fraction did
    exactly that (x0.33 for an intact 8 Hz kernel at 20 Hz, x0.5-0.67 for the top kernels at 50 Hz),
    a per-device magnitude fingerprint. The mask is the coefficient energy that survives the cut."""
    module = ContinuousKernelTokenizer().eval()
    for rate in (20.0, 25.0, 50.0, 100.0):
        nyq, _ = module.masks(rate, torch.tensor([6.0]))
        carrier_live = module.carrier.float() / module.spans <= module.nyquist_margin * rate / 2
        assert torch.equal(nyq[0] > 0.5, carrier_live)
        assert bool(((nyq[0] == 0) | (nyq[0] == 1)).all()), f"{rate} Hz rescales an intact kernel"
    # Once learning spreads energy onto a harmonic a 20 Hz source cannot supply, the mask reports
    # how much of the learned filter survives — here half.
    k = module.K - 1                                   # 15 Hz kernel, span 0.267 s
    with torch.no_grad():
        module.cos_coeff.zero_()
        module.sin_coeff.zero_()
        module.cos_coeff[k, 0] = 1.0                   # harmonic 1: 3.75 Hz, live at 20 Hz
        module.sin_coeff[k, module.M - 1] = 1.0        # harmonic 12: 45 Hz, dead at 20 Hz
    nyq, _ = module.masks(20.0, torch.tensor([6.0]))
    assert float(nyq[0, k]) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("patch_seconds,per", [(0.5, 25), (1.5, 75)])
def test_patch_duration_is_a_constructor_property(patch_seconds, per):
    """Frames are laid out on the physical patch grid the module was built for, so a 1.5 s token
    packages 12 frames (6 after the stride-2 stage) and the analysis covers the whole window. The
    previous layout assumed one-second patches and silently analysed only the first P seconds."""
    torch.manual_seed(0)
    module = ContinuousKernelTokenizer(patch_seconds=patch_seconds).eval()
    rate = 50.0
    signal = torch.from_numpy(_band_limited_signal(rate)).view(1, -1, 1)
    n_patches = signal.shape[1] // per
    patches = signal[:, :n_patches * per].reshape(1, n_patches, per, 1)
    lengths = torch.full((1, n_patches), per, dtype=torch.long)
    with torch.no_grad():
        out = module.analyze(patches, rate, lengths)
        tokens = module.project(out)
    assert tokens.shape[:2] == (1, n_patches)
    assert module.frames_per_token == module.F * patch_seconds / 2
    assert out["compressed"].shape[-1] == n_patches * module.frames_per_patch
    last_centre = module._frame_geometry(rate, n_patches, torch.device("cpu"))["frame_time"][-1]
    assert float(last_centre) == pytest.approx(n_patches * patch_seconds - 0.5 / module.F, abs=1e-5)
    assert bool(out["frame_valid"].all())
    one_second, one_len, _ = _as_patches(_band_limited_signal(rate), rate)
    with pytest.raises(ValueError, match="different patch grid"):
        module.analyze(one_second, rate, one_len)


def test_default_module_refuses_a_foreign_patch_grid(tokenizer):
    patches = torch.randn(1, 4, 75, 1)
    lengths = torch.full((1, 4), 75, dtype=torch.long)
    with pytest.raises(ValueError, match="different patch grid"):
        tokenizer.analyze(patches, 50.0, lengths)


def test_a_short_final_patch_is_still_accepted(tokenizer):
    patches = torch.randn(1, 3, 50, 1)
    lengths = torch.tensor([[50, 50, 20]])
    with torch.no_grad():
        out = tokenizer(patches, 50.0, lengths, patch_mask=torch.ones(1, 3, dtype=torch.bool))
    assert out.shape[1] == 3


def test_invalid_patch_seconds_are_rejected():
    with pytest.raises(ValueError, match="even integer"):
        ContinuousKernelTokenizer(patch_seconds=0.3)


def test_norm_statistics_round_trip(tokenizer):
    module = ContinuousKernelTokenizer().eval()
    assert float(module._norm_fitted) == 0.0
    module.reset_norm_accumulator()
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0), 50.0)
    module.accumulate_norm_stats(patches, 50.0, lengths, patch_mask=mask)
    module.finalize_norm_stats()
    assert float(module._norm_fitted) == 1.0
    assert torch.isfinite(module.norm_mu).all() and (module.norm_sd > 0).all()


def test_gradients_reach_every_analysis_parameter(tokenizer):
    module = ContinuousKernelTokenizer()
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0), 50.0)
    module(patches, 50.0, lengths, patch_mask=mask).pow(2).mean().backward()
    for name in ("cos_coeff", "sin_coeff", "sigma_logit"):
        grad = getattr(module, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, f"{name} received no gradient"
    assert not module.gain_logit.requires_grad
    assert module.gain_logit.grad is None


def test_mixed_rate_batch_matches_separate_processing_and_batch_order():
    """A sample's tokens must not inherit the first row's rate or reflection boundary."""
    torch.manual_seed(0)
    module = ContinuousKernelTokenizer().eval()
    signals = [_band_limited_signal(rate) for rate in (20.0, 100.0)]
    patches = torch.zeros(2, 6, 100, 1)
    lengths = torch.zeros(2, 6, dtype=torch.long)
    for row, (signal, rate) in enumerate(zip(signals, (20, 100))):
        per = int(rate)
        patches[row, :, :per, 0] = torch.from_numpy(signal.reshape(6, per))
        lengths[row] = per
    mask = torch.ones(2, 6, dtype=torch.bool)
    rates = torch.tensor([20.0, 100.0])
    with torch.no_grad():
        together = module(patches, rates, lengths, patch_mask=mask)
        separate = torch.cat([
            module(patches[i:i + 1], rates[i], lengths[i:i + 1], patch_mask=mask[i:i + 1])
            for i in range(2)
        ])
        reversed_batch = module(patches.flip(0), rates.flip(0), lengths.flip(0),
                                patch_mask=mask.flip(0)).flip(0)
    assert torch.allclose(together, separate, atol=2e-5, rtol=2e-5)
    assert torch.allclose(together, reversed_batch, atol=2e-5, rtol=2e-5)


def test_source_rate_controls_observability_and_output():
    module = ContinuousKernelTokenizer().eval()
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0), 50.0)
    with torch.no_grad():
        native_25 = module.analyze(patches, 50.0, lengths, source_rate_hz=25.0,
                                   patch_mask=mask)
        native_50 = module.analyze(patches, 50.0, lengths, source_rate_hz=50.0,
                                   patch_mask=mask)
        out_25 = module.project(native_25)
        out_50 = module.project(native_50)
    assert (native_25["nyquist"] < native_50["nyquist"]).any()
    assert not torch.allclose(out_25, out_50)


def test_norm_calibration_excludes_absent_channels_and_padded_patches():
    torch.manual_seed(0)
    reference = ContinuousKernelTokenizer().eval()
    masked = ContinuousKernelTokenizer().eval()
    masked.load_state_dict(reference.state_dict())
    patches, lengths, patch_mask = _as_patches(_band_limited_signal(50.0), 50.0)
    reference.accumulate_norm_stats(
        patches[:, :3], 50.0, lengths[:, :3], patch_mask=patch_mask[:, :3],
        channel_mask=torch.ones(1, 1, dtype=torch.bool))
    reference.finalize_norm_stats()

    six_channels = torch.randn(1, 6, 50, 6) * 100.0
    six_channels[..., 0] = patches[..., 0]
    padded = patch_mask.clone()
    padded[:, 3:] = False
    padded_lengths = lengths.clone()
    padded_lengths[:, 3:] = 0
    masked.accumulate_norm_stats(
        six_channels, 50.0, padded_lengths, patch_mask=padded,
        channel_mask=torch.tensor([[True, False, False, False, False, False]]))
    masked.finalize_norm_stats()
    assert torch.allclose(reference.norm_mu, masked.norm_mu, atol=1e-6)
    assert torch.allclose(reference.norm_sd, masked.norm_sd, atol=1e-6)
    assert torch.allclose(reference.amp_mu, masked.amp_mu, atol=1e-6)
    assert torch.allclose(reference.dc_mu, masked.dc_mu, atol=1e-6)


def test_dc_and_amplitude_are_patch_local():
    module = ContinuousKernelTokenizer().eval()
    patches = torch.cat((torch.ones(1, 3, 20, 1), -torch.ones(1, 3, 20, 1)), dim=1)
    lengths = torch.full((1, 6), 20, dtype=torch.long)
    analysis = module.analyze(patches, 20.0, lengths)
    assert torch.allclose(analysis["dc"].flatten(),
                          torch.tensor([1., 1., 1., -1., -1., -1.]))
    assert torch.allclose(analysis["amplitude"].flatten(),
                          torch.full((6,), math.log(2.0)))


def test_edge_support_is_per_kernel_and_measures_real_samples():
    module = ContinuousKernelTokenizer().eval()
    patches = torch.randn(1, 2, 50, 1)
    lengths = torch.full((1, 2), 50, dtype=torch.long)
    edge = module.analyze(patches, 50.0, lengths)["edge"]
    assert edge.shape == (1, module.K, 2 * module.F)
    assert 0.0 < float(edge[0, 0, 0]) < 0.75
    assert float(edge[0, :, 0].std()) > 0.0
    assert float(edge[0, 0, module.F]) > float(edge[0, 0, 0])


def test_analysis_parameterization_is_bounded_and_regularized():
    module = ContinuousKernelTokenizer()
    with torch.no_grad():
        module.sigma_logit.fill_(100.0)
        module.gain_logit.fill_(-100.0)
    assert (module._sigmas() <= module.sigma_max).all()
    assert (module._sigmas() >= module.sigma_min).all()
    assert (module._gains() <= module.gain_max).all()
    assert (module._gains() >= 1.0 / module.gain_max).all()
    assert torch.isfinite(module.adaptation_regularization())


def test_sampling_geometry_deduplicates_repeated_fractional_phases():
    module = ContinuousKernelTokenizer()
    expected = {20.0: 2, 25.0: 8, 50.0: 4, 100.0: 2}
    for rate, phase_count in expected.items():
        geometry = module._frame_geometry(rate, 6, torch.device("cpu"))
        assert geometry["u"].shape[0] == phase_count
        assert geometry["phase_id"].shape[0] == 6 * module.F


def test_runtime_telemetry_is_finite_and_only_collected_on_request():
    module = ContinuousKernelTokenizer().eval()
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0), 50.0)
    assert module.runtime_summary() == {}
    module.request_runtime_telemetry()
    with torch.no_grad():
        module(patches, 50.0, lengths, patch_mask=mask)
    summary = module.runtime_summary()
    assert set(summary) == {
        "frontend/observable_fraction", "frontend/edge_support_mean",
        "frontend/response_std_mean", "frontend/dead_kernel_fraction",
    }
    assert all(math.isfinite(value) for value in summary.values())
    assert 0.0 <= summary["frontend/dead_kernel_fraction"] <= 1.0


def test_encoder_exposes_continuous_frontend_as_one_token_per_sensor():
    from model.tokenizer.encoder import SetTokenizerEncoder

    encoder = SetTokenizerEncoder(
        d_model=32, num_layers=1, num_heads=4, dim_feedforward=64,
        frontend="continuous", trunk="temporal", descriptor_prediction=False,
        token_granularity="sensor",
    ).eval()
    assert isinstance(encoder.filterbank, ContinuousKernelTokenizer)
    patches = torch.randn(2, 2, 50, 3)
    lengths = torch.tensor([[20, 20], [50, 50]])
    channel_mask = torch.ones(2, 3, dtype=torch.bool)
    sensor_id = torch.zeros(2, 3, dtype=torch.long)
    with torch.no_grad():
        tokens = encoder.tokenize(
            patches, torch.tensor([20.0, 50.0]), lengths,
            channel_mask=channel_mask, sensor_id=sensor_id, n_sensors=1,
        )
    assert tokens.shape == (2, 2, 1, 32)
    assert torch.isfinite(tokens).all()


def test_dense_sensor_cnn_keeps_modalities_separate_and_marks_missing_axes():
    torch.manual_seed(0)
    module = ContinuousKernelTokenizer(d_model=32).eval()
    patches = torch.randn(1, 2, 50, 6)
    lengths = torch.full((1, 2), 50, dtype=torch.long)
    sensor_id = torch.tensor([[0, 0, 0, 1, 1, 1]])
    both = torch.ones(1, 6, dtype=torch.bool)
    missing_acc_y = both.clone()
    missing_acc_y[:, 1] = False
    analysis = module.analyze(patches, 50.0, lengths)
    with torch.no_grad():
        complete = module.project(
            analysis, sensor_id=sensor_id, channel_mask=both, n_sensors=2,
        )
        missing = module.project(
            analysis, sensor_id=sensor_id, channel_mask=missing_acc_y, n_sensors=2,
        )
    assert complete.shape == (1, 2, 2, 32)
    # Removing an accelerometer axis changes its token but cannot alter the gyroscope token.
    assert not torch.allclose(complete[:, :, 0], missing[:, :, 0])
    assert torch.allclose(complete[:, :, 1], missing[:, :, 1], atol=1e-6)


def test_accel_only_padding_cannot_overwrite_live_sensor_presence():
    module = ContinuousKernelTokenizer(d_model=32).eval()
    patches = torch.randn(2, 2, 50, 6)
    lengths = torch.full((2, 2), 50, dtype=torch.long)
    # This is the loader's established accel-only contract: absent gyro slots keep a valid id and
    # are excluded by channel_mask rather than by a sentinel sensor id.
    sensor_id = torch.zeros(2, 6, dtype=torch.long)
    channel_mask = torch.tensor([[True, True, True, False, False, False]]).expand(2, -1)
    with torch.no_grad():
        tokens = module(
            patches, 50.0, lengths, sensor_id=sensor_id,
            channel_mask=channel_mask, n_sensors=1,
        )
    assert tokens.shape == (2, 2, 1, 32)
    assert module.sensor_presence(sensor_id, channel_mask, 1).all()


def test_dense_sensor_frontend_has_no_dead_trainable_parameters():
    torch.manual_seed(4)
    module = ContinuousKernelTokenizer(d_model=32)
    patches = torch.randn(2, 3, 50, 6)
    lengths = torch.full((2, 3), 50, dtype=torch.long)
    sensor_id = torch.tensor([[0, 0, 0, 1, 1, 1]]).expand(2, -1)
    channel_mask = torch.ones(2, 6, dtype=torch.bool)
    output = module(
        patches, 50.0, lengths, sensor_id=sensor_id,
        channel_mask=channel_mask, n_sensors=2,
    )
    output.square().mean().backward()
    dead = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if (parameter.grad is None or not torch.isfinite(parameter.grad).all()
                or float(parameter.grad.abs().sum()) == 0.0):
            dead.append(name)
    assert not dead, f"trainable parameters without a finite nonzero gradient: {dead}"


def test_features_do_not_depend_on_how_long_the_rest_of_the_recording_was():
    """A duration fingerprint is the same class of leak as a rate fingerprint.

    `GroupNorm(1, C)` normalises over (channels, time), so a frame's features come to depend on the
    statistics of the whole recording — measured at 2.43 max delta for a shared span between a 2 s
    and a 6 s window. The stack must normalise over CHANNELS at each frame only (design doc Rule 5).
    """
    torch.manual_seed(0)
    module = ContinuousKernelTokenizer().eval()
    rate, per = 50.0, 50
    head = torch.randn(1, 2, per, 1)
    tail = torch.randn(1, 4, per, 1) * 5.0                 # a loud, different remainder
    short_len = torch.full((1, 2), per, dtype=torch.long)
    long_len = torch.full((1, 6), per, dtype=torch.long)
    with torch.no_grad():
        short = module(head, rate, short_len, patch_mask=torch.ones(1, 2, dtype=torch.bool))
        long = module(torch.cat([head, tail], dim=1), rate, long_len,
                      patch_mask=torch.ones(1, 6, dtype=torch.bool))
    # the shared first patch must be unaffected by what follows it, up to the kernel's own
    # legitimate look-ahead at the boundary
    assert torch.allclose(short[:, 0], long[:, 0], atol=0.15), (
        f"first token moved by {(short[:, 0] - long[:, 0]).abs().max():.4f} when the recording "
        "was extended -- a duration fingerprint has leaked in")


def test_future_patch_mask_blocks_signal_and_gradient_from_hidden_suffix():
    torch.manual_seed(17)
    module = ContinuousKernelTokenizer(d_model=16).eval()
    patches = torch.randn(1, 6, 50, 3, requires_grad=True)
    changed = patches.detach().clone()
    changed[:, 3:] += 1000.0
    lengths = torch.full((1, 6), 50, dtype=torch.long)
    prefix = torch.tensor([[True, True, True, False, False, False]])
    with torch.no_grad():
        original = module(patches.detach(), 50.0, lengths, patch_mask=prefix)
        perturbed = module(changed, 50.0, lengths, patch_mask=prefix)
    assert torch.allclose(original, perturbed, atol=1e-6)

    output = module(patches, 50.0, lengths, patch_mask=prefix)
    output.square().mean().backward()
    assert patches.grad[:, :3].abs().sum() > 0
    assert patches.grad[:, 3:].abs().sum() == 0


def test_harmonic_count_is_enough_for_the_shapes_this_front_end_claims():
    """The motivating claim is that a time-domain kernel can match an asymmetric impact, which a
    Gaussian band cannot. Assert the basis can actually express one."""
    module = ContinuousKernelTokenizer()
    M = module.M
    n = 256
    u = torch.linspace(-0.5, 0.5, n)
    impact = torch.where(u < 0, torch.exp(u * 20), torch.exp(-u * 6)) * torch.sign(u + 1e-9)
    impact = impact - impact.mean()
    basis = []
    for m in range(1, M + 1):
        basis += [torch.cos(2 * math.pi * m * u), torch.sin(2 * math.pi * m * u)]
    design = torch.stack(basis, dim=1)
    coeff = torch.linalg.lstsq(design, impact.unsqueeze(1)).solution
    predicted = (design @ coeff).squeeze(1)
    r2 = float(1 - ((impact - predicted) ** 2).sum() / ((impact - impact.mean()) ** 2).sum())
    assert r2 > 0.80, f"only {r2:.3f} of an asymmetric impact is representable with M={M}"
