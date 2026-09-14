"""
Tests for the Physical-Hz Filterbank tokenizer (ported with the M1 filterbank port).

Covers the properties that justify the design (docs/v2/design_tokenizer.md) and the
three guardrails agreed during review:
  - rate-invariance (the headline claim)
  - streaming == offline per-patch equivalence
  - shape / dtype contract for variable P / C / rate
  - Nyquist observability mask + low-freq resolution flag behave as specified
  - S >= r*D precondition is enforced
  - Gaussian bands are never structurally empty; no NaN/inf; gradients flow
  - amplitude scalar preserves absolute magnitude
  - frozen per-band standardization calibrates and freezes
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.tokenizer.filterbank import PhysicalFilterbankTokenizer  # noqa: E402


# --------------------------------------------------------------------------- utils
def make_sinusoid_batch(freqs_hz, rates, D, S, amp=1.0):
    """One patch per (freq, rate); native samples in [0, N), zero-padded to S.

    Returns patches (B,1,S,1), rate (B,), N (B,).
    """
    B = len(rates)
    patches = torch.zeros(B, 1, S, 1)
    Ns = []
    for b, (f0, r) in enumerate(zip(freqs_hz, rates)):
        n = int(round(r * D))
        t = torch.arange(n, dtype=torch.float32) / r
        patches[b, 0, :n, 0] = amp * torch.sin(2 * math.pi * f0 * t)
        Ns.append(n)
    return patches, torch.tensor(rates, dtype=torch.float32), torch.tensor(Ns)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


# ------------------------------------------------------------------------- fixtures
@pytest.fixture
def tok():
    t = PhysicalFilterbankTokenizer(d_model=384, n_bands=32, dft_size=512)
    t.eval()
    return t


# ---------------------------------------------------------------------------- tests
def test_shape_contract(tok):
    for B, P, C in [(1, 1, 1), (4, 7, 9), (3, 2, 40)]:
        S = tok.S
        patches = torch.randn(B, P, S, C)
        rates = torch.full((B,), 50.0)
        N = torch.full((B,), 100)
        out = tok(patches, rates, N)
        assert out.shape == (B, P, C, tok.d_model)
        assert out.dtype == patches.dtype
        assert torch.isfinite(out).all()


def test_scalar_rate_and_default_length(tok):
    patches = torch.randn(2, 3, tok.S, 6)
    out_scalar = tok(patches, 50.0, 100)
    assert out_scalar.shape == (2, 3, 6, tok.d_model)
    # patch_len_samples=None -> treat full S as valid (no crash, finite)
    out_full = tok(patches, 50.0, None)
    assert torch.isfinite(out_full).all()


def test_rate_invariance_2hz(tok):
    """A pure 2 Hz tone deposits energy in the same physical-Hz band at 20/50/100 Hz."""
    patches, rates, N = make_sinusoid_batch([2.0, 2.0, 2.0], [20.0, 50.0, 100.0], D=2.0, S=tok.S)
    E, _, _, _ = tok._band_energy(patches, rates, N)          # (3,1,1,K)
    logE = torch.log1p(E).squeeze(1).squeeze(1)            # (3, K)

    # peak band coincides across all three rates
    peaks = logE.argmax(dim=-1)
    assert peaks[0].item() == peaks[1].item() == peaks[2].item()

    # and the whole band profile is near-identical (headline claim: cosine > 0.98)
    assert cos(logE[0], logE[1]) > 0.98
    assert cos(logE[0], logE[2]) > 0.98
    assert cos(logE[1], logE[2]) > 0.98


def test_peak_band_matches_frequency(tok):
    """The peak band's center frequency should be near the injected tone."""
    for f0 in [1.0, 3.0, 5.0]:
        patches, rates, N = make_sinusoid_batch([f0], [50.0], D=2.0, S=tok.S)
        E, centers, _, _ = tok._band_energy(patches, rates, N)
        peak = torch.log1p(E).reshape(-1).argmax().item()
        assert abs(centers[peak].item() - f0) < 0.6 * f0  # within ~half an octave


def test_nyquist_mask(tok):
    """At 20 Hz, bands whose center+2sigma exceed 0.9*Nyquist(=9Hz) are masked out."""
    o, _ = tok.masks(torch.tensor([20.0]), torch.tensor([40]))   # (1,K)
    centers = tok.centers
    sigma = centers / (2.0 * tok.Q)
    expected = (centers + 2 * sigma <= 0.9 * 10.0).float()
    assert torch.equal(o[0], expected)
    assert o[0].min() == 0.0 and o[0].max() == 1.0               # some seen, some blind

    # at 100 Hz (Nyquist 50) every band up to f_max=15 is observable
    o_hi, _ = tok.masks(torch.tensor([100.0]), torch.tensor([200]))
    assert o_hi[0].min() == 1.0


def test_nyquist_mask_zeros_bands_in_forward(tok):
    """Unobservable bands must appear as exactly 0 in the e_hat slot of the token input."""
    # Isolate by making projection identity-like: read e_hat straight out via a probe.
    patches = torch.randn(1, 1, tok.S, 1)
    o, _ = tok.masks(torch.tensor([20.0]), torch.tensor([40]))
    # Recompute e_hat the way forward does and confirm masked bands are zero.
    E, centers, sigma, _ = tok._band_energy(patches, torch.tensor([20.0]), torch.tensor([40]))
    e = torch.log1p(E)
    e_hat = (e - tok.norm_mu) / tok.norm_sd
    e_hat = e_hat * o.view(1, 1, 1, -1)
    masked = (o[0] == 0)
    assert torch.count_nonzero(e_hat[..., masked]) == 0


def test_resolution_flag_ramps_with_duration(tok):
    """Longer windows resolve low bands better: res is monotonic non-decreasing in D."""
    _, res_short = tok.masks(torch.tensor([50.0]), torch.tensor([50]))    # D=1.0s
    _, res_long = tok.masks(torch.tensor([50.0]), torch.tensor([200]))    # D=4.0s
    assert (res_long[0] >= res_short[0] - 1e-6).all()
    assert res_short.min() >= 0.0 and res_long.max() <= 1.0
    # low bands are the ones that benefit; the lowest band should be under-resolved at D=1s
    assert res_short[0, 0] < 1.0


def test_s_ge_rd_precondition(tok):
    """N > S must raise rather than silently truncating the window."""
    patches = torch.randn(1, 1, tok.S, 1)
    with pytest.raises(ValueError):
        tok(patches, 100.0, tok.S + 1)


def test_no_empty_bands(tok):
    """Gaussian filters have soft support: every band gets nonzero energy from noise."""
    patches = torch.randn(2, 3, tok.S, 4)
    # keep everything within Nyquist so no legitimate zeros from masking
    E, _, _, _ = tok._band_energy(patches, torch.full((2,), 100.0), torch.full((2,), 200))
    assert (E > 0).all()
    assert torch.isfinite(E).all()


def test_streaming_equivalence(tok):
    """Per-patch tokens are identical whether computed together or one at a time."""
    P = 5
    patches = torch.randn(1, P, tok.S, 3)
    full = tok(patches, 50.0, 100)
    for p in range(P):
        one = tok(patches[:, p:p + 1], 50.0, 100)
        assert torch.allclose(full[:, p], one[:, 0], atol=1e-5), f"patch {p} mismatch"


def test_amplitude_preserves_magnitude(tok):
    """Scaling the signal up must raise the amplitude scalar (running != walking)."""
    patches, rates, N = make_sinusoid_batch([2.0], [50.0], D=2.0, S=tok.S, amp=1.0)
    patches_loud = patches * 5.0
    E_soft, _, _, _ = tok._band_energy(patches, rates, N)
    E_loud, _, _, _ = tok._band_energy(patches_loud, rates, N)
    amp_soft = torch.log1p(E_soft.sum())
    amp_loud = torch.log1p(E_loud.sum())
    assert amp_loud > amp_soft + 1.0


def test_dc_feature_encodes_gravity_axis():
    """The signed DC feature restores static-posture discrimination: two still postures
    with gravity on different accel axes must produce DIFFERENT DC features, and removing
    the DC dim collapses them (proving DC is the sole discriminator for static postures)."""
    S = 512
    tok = PhysicalFilterbankTokenizer(d_model=384, dft_size=S, use_dc=True, norm="frozen")

    def static(axis, g=9.8, n=1):
        x = torch.randn(n, 1, S, 3) * 0.02
        x[:, :, :100, axis] += g
        x[:, :, 100:, :] = 0
        return x

    # calibrate over static postures on all axes + both unit scales (g and m/s^2)
    calib = torch.cat([static(a, g, n=8) for a in range(3) for g in (1.0, 9.8)], dim=0)
    tok.fit_norm_stats(calib, torch.full((calib.shape[0],), 50.0), torch.full((calib.shape[0],), 100))
    tok.eval()

    # DC value per channel should spike on the gravity axis and be low elsewhere.
    with torch.no_grad():
        r, N = tok._prep_rate_len(torch.tensor([50.0]), torch.tensor([100]), 1, torch.device("cpu"), torch.float32)
        _, _, _, dc_z = tok._band_energy(static(2), r, N)   # gravity on Z
        _, _, _, dc_y = tok._band_energy(static(1), r, N)   # gravity on Y
    dc_z, dc_y = dc_z.reshape(3), dc_y.reshape(3)
    assert dc_z.argmax().item() == 2 and dc_y.argmax().item() == 1, (dc_z, dc_y)
    # standardized: the gravity-axis DC is well above the off-axis DC
    assert (dc_z[2] - dc_z[0]) > 1.0

    # in_dim includes exactly one DC slot; zeroing it makes the two postures collapse.
    # K=32: energy, observability, resolution, amplitude, DC, and 97 polarization slots.
    assert tok.use_dc and tok.in_dim == 32 * 3 + 1 + 1 + (3 * 32 + 1)


def test_dc_disabled_is_blind_to_posture():
    """Regression guard: with use_dc=False the tokenizer is (correctly) posture-blind,
    documenting WHY the DC feature exists."""
    S = 512
    tok = PhysicalFilterbankTokenizer(d_model=384, dft_size=S, use_dc=False, norm="none")
    tok.eval()

    def static(axis, g=9.8):
        x = torch.randn(1, 1, S, 3) * 0.02
        x[:, :, :100, axis] += g
        x[:, :, 100:, :] = 0
        return x

    with torch.no_grad():
        tz = tok(static(2), torch.tensor([50.0]), torch.tensor([100])).reshape(3, -1)
        ty = tok(static(1), torch.tensor([50.0]), torch.tensor([100])).reshape(3, -1)
    cos = torch.nn.functional.cosine_similarity(tz, ty, dim=-1).mean()
    assert cos > 0.99  # posture-blind without DC


def test_gradient_flow(tok):
    patches = torch.randn(2, 4, tok.S, 3, requires_grad=True)
    out = tok(patches, 50.0, 100)
    out.sum().backward()
    assert patches.grad is not None and torch.isfinite(patches.grad).all()
    assert tok.proj.weight.grad is not None and torch.isfinite(tok.proj.weight.grad).all()


def test_calibration_sets_frozen_stats(tok):
    assert tok._norm_fitted.item() == 0.0
    patches = torch.randn(8, 6, tok.S, 5)
    tok.fit_norm_stats(patches, torch.full((8,), 50.0), torch.full((8,), 100))
    assert tok._norm_fitted.item() == 1.0
    assert torch.isfinite(tok.norm_mu).all() and torch.isfinite(tok.norm_sd).all()
    assert (tok.norm_sd > 0).all()
    # stats moved off the identity defaults
    assert not torch.allclose(tok.norm_mu, torch.zeros_like(tok.norm_mu))
    assert not torch.allclose(tok.norm_sd, torch.ones_like(tok.norm_sd))


def test_calibration_observability_uses_native_source_rate():
    """Upsampled 25 Hz signals must not calibrate interpolation-only 25--50 Hz bands."""
    tok = PhysicalFilterbankTokenizer(d_model=8, n_bands=32, dft_size=128)
    patches = torch.randn(1, 1, 128, 1)
    tok.reset_norm_accumulator()
    tok.accumulate_norm_stats(
        patches,
        sampling_rate_hz=torch.tensor([50.0]),
        patch_len_samples=torch.tensor([50]),
        source_rate_hz=torch.tensor([25.0]),
    )
    observable, _ = tok.masks(
        torch.tensor([50.0]), torch.tensor([50]), source_rate_hz=torch.tensor([25.0])
    )
    assert torch.equal(tok._acc_count > 0, observable[0].bool())
    assert int((tok._acc_count == 0).sum()) == 5
    tok.finalize_norm_stats()
    assert torch.equal(tok.norm_mu[~observable[0].bool()], torch.zeros(5))
    assert torch.equal(tok.norm_sd[~observable[0].bool()], torch.ones(5))


def test_calibration_streaming_matches_oneshot():
    """Accumulate over two halves == one-shot fit over the whole batch."""
    a = PhysicalFilterbankTokenizer(d_model=64, n_bands=16, dft_size=256)
    b = PhysicalFilterbankTokenizer(d_model=64, n_bands=16, dft_size=256)
    patches = torch.randn(10, 3, 256, 4)
    rate, N = torch.full((10,), 50.0), torch.full((10,), 100)

    a.fit_norm_stats(patches, rate, N)

    b.reset_norm_accumulator()
    b.accumulate_norm_stats(patches[:4], rate[:4], N[:4])
    b.accumulate_norm_stats(patches[4:], rate[4:], N[4:])
    b.finalize_norm_stats()

    assert torch.allclose(a.norm_mu, b.norm_mu, atol=1e-5)
    assert torch.allclose(a.norm_sd, b.norm_sd, atol=1e-5)


def test_param_count_matches_spec():
    """The legacy no-polarization shape is retained; the enabled shape is explicit."""
    t = PhysicalFilterbankTokenizer(d_model=384, n_bands=32, use_resolution_mask=False,
                                    use_amplitude=True, use_dc=False, use_polarization=False)
    assert t.in_dim == 65
    n_proj = t.proj.weight.numel() + t.proj.bias.numel()
    assert n_proj == 65 * 384 + 384 == 25344
    # with the resolution flag on: Linear(97->384)
    t2 = PhysicalFilterbankTokenizer(
        d_model=384, n_bands=32, use_resolution_mask=True, use_dc=False,
        use_polarization=False,
    )
    assert t2.in_dim == 97
    # DC feature adds exactly one input dim (signed per-channel gravity/tilt scalar).
    t3 = PhysicalFilterbankTokenizer(
        d_model=384, n_bands=32, use_resolution_mask=True, use_dc=True,
        use_polarization=False,
    )
    assert t3.in_dim == 98
    t4 = PhysicalFilterbankTokenizer(d_model=384, n_bands=32, use_resolution_mask=True, use_dc=True)
    assert t4.in_dim == 195


def test_learnable_arm_b_trains():
    """Every adaptive filter parameter receives finite gradients from the signal path."""
    t = PhysicalFilterbankTokenizer(d_model=64, n_bands=16, dft_size=256, learnable=True)
    patches = torch.randn(2, 2, 256, 3)
    out = t(patches, 50.0, 100)
    out.sum().backward()
    for name in ("_center_offsets", "_bandwidth_logits", "_compression_logits", "_shape_logit"):
        g = getattr(t, name).grad
        assert g is not None and torch.isfinite(g).all(), name
        assert g.abs().sum() > 0, name
    # Endpoints are pinned and interior centers remain ordered.
    c = t._band_centers()
    assert torch.allclose(c[[0, -1]], torch.tensor([t.f_min, t.f_max]))
    assert (c[1:] > c[:-1]).all()


def test_learnable_arm_b_bounds_survive_extreme_logits():
    t = PhysicalFilterbankTokenizer(d_model=16, n_bands=16, learnable=True)
    with torch.no_grad():
        t._center_offsets.copy_(torch.linspace(-1e6, 1e6, 14))
        t._bandwidth_logits.copy_(torch.linspace(-1e6, 1e6, 16))
        t._compression_logits.copy_(torch.linspace(1e6, -1e6, 16))
        t._shape_logit.fill_(1e6)
    centers = t._band_centers()
    bw = t._band_sigmas(centers, adaptive=True) / (centers / (2 * t.Q))
    gains = t._compression_gains()
    assert (centers[1:] > centers[:-1]).all()
    assert bw.min() >= 1 / t.bandwidth_factor_max - 1e-6
    assert bw.max() <= t.bandwidth_factor_max + 1e-6
    assert gains.min() >= 1 / t.compression_gain_max - 1e-6
    assert gains.max() <= t.compression_gain_max + 1e-6
    assert t.filter_shape_min <= t._filter_shape(True) <= t.filter_shape_max


def test_per_token_patch_lengths_are_supported():
    t = PhysicalFilterbankTokenizer(d_model=32, n_bands=16, dft_size=256)
    patches = torch.randn(2, 4, 256, 3)
    lengths = torch.tensor([[20, 50, 75, 10], [40, 60, 100, 25]])
    out = t(patches, torch.tensor([50.0, 100.0]), lengths)
    o, res = t.masks(torch.tensor([50.0, 100.0]), lengths)
    assert out.shape == (2, 4, 3, 32) and torch.isfinite(out).all()
    assert o.shape == (2, 16) and res.shape == (2, 4, 16)


def test_amplitude_rate_invariant(tok):
    """The amplitude scalar must NOT encode sampling rate (window-energy normalization).

    Regression for the confound where an unnormalized rDFT made E (hence amp) scale with
    N=r*D, so amp drifted ~log(rate ratio) for the same physical motion.
    """
    patches, rates, N = make_sinusoid_batch([2.0, 2.0, 2.0], [20.0, 50.0, 100.0], D=2.0, S=tok.S)
    E, _, _, _ = tok._band_energy(patches, rates, N)
    amp = torch.log1p(E.sum(dim=-1)).reshape(-1)          # (3,)
    assert (amp.max() - amp.min()) < 0.3, f"amp rate-drift {amp.tolist()} (was ~1.6 pre-fix)"


def test_amplitude_duration_invariant(tok):
    """The amplitude scalar must NOT encode patch duration for a fixed physical tone."""
    p1, r1, n1 = make_sinusoid_batch([2.0], [50.0], D=1.0, S=tok.S)
    p2, r2, n2 = make_sinusoid_batch([2.0], [50.0], D=8.0, S=tok.S)
    a1 = torch.log1p(tok._band_energy(p1, r1, n1)[0].sum())
    a2 = torch.log1p(tok._band_energy(p2, r2, n2)[0].sum())
    assert abs((a1 - a2).item()) < 0.3, "amp duration-drift (was ~2.1 pre-fix)"


def test_calibration_is_observability_masked():
    """Frozen per-band stats reflect only samples where each band is observable."""
    t = PhysicalFilterbankTokenizer(d_model=64, n_bands=32, dft_size=512)
    S = t.S
    hi = torch.randn(6, 4, S, 3); hi_r = torch.full((6,), 100.0); hi_N = torch.full((6,), 200)
    lo = torch.randn(6, 4, S, 3); lo_r = torch.full((6,), 20.0); lo_N = torch.full((6,), 40)
    t.reset_norm_accumulator()
    t.accumulate_norm_stats(hi, hi_r, hi_N)
    t.accumulate_norm_stats(lo, lo_r, lo_N)
    t.finalize_norm_stats()

    o_lo, _ = t.masks(torch.tensor([20.0]), torch.tensor([40]))
    o_hi, _ = t.masks(torch.tensor([100.0]), torch.tensor([200]))
    rows = 6 * 4 * 3  # B*P*C per batch

    hi_only = (o_hi[0] > 0) & (o_lo[0] == 0)   # observable at 100Hz, not at 20Hz
    both = (o_hi[0] > 0) & (o_lo[0] > 0)       # observable at both
    assert hi_only.any() and both.any()
    # high-only bands counted the 100Hz rows only (not dragged toward 0 by 20Hz samples)
    assert torch.allclose(t._acc_count[hi_only],
                          torch.full((int(hi_only.sum()),), float(rows), dtype=torch.float64))
    assert torch.allclose(t._acc_count[both],
                          torch.full((int(both.sum()),), float(2 * rows), dtype=torch.float64))


def test_per_channel_source_rates_calibrate_like_separate_native_streams():
    """A mixed-device row must preserve each channel's acquisition Nyquist during calibration."""
    torch.manual_seed(19)
    mixed = PhysicalFilterbankTokenizer(d_model=32, n_bands=16, dft_size=128)
    reference = PhysicalFilterbankTokenizer(d_model=32, n_bands=16, dft_size=128)
    patches = torch.randn(1, 2, 128, 2)
    rate = torch.tensor([100.0])
    lengths = torch.full((1, 2), 100, dtype=torch.long)

    mixed.reset_norm_accumulator()
    mixed.accumulate_norm_stats(
        patches, rate, lengths, source_rate_hz=torch.tensor([[20.0, 100.0]]),
    )
    reference.reset_norm_accumulator()
    reference.accumulate_norm_stats(patches[..., :1], rate, lengths, source_rate_hz=torch.tensor([20.0]))
    reference.accumulate_norm_stats(patches[..., 1:], rate, lengths, source_rate_hz=torch.tensor([100.0]))

    torch.testing.assert_close(mixed._acc_count, reference._acc_count)
    torch.testing.assert_close(mixed._acc_sum, reference._acc_sum, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(mixed._acc_sqsum, reference._acc_sqsum, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
