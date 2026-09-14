"""Physical-Hz constant-Q filterbank tokenizer (M1 port).

Ported verbatim from `legacy_code/model/feature_extractor.py` (the battle-tested V2
tokenizer, with its unit-test suite at `tests/test_filterbank.py`) into the
HALO representation encoder; see docs/design/DESIGN_OF_RECORD.md.
Key hyperparameters and the physics that pins each value are collected below
so they are easy to find and hard to second-guess.

`learnable=True` is the constrained-learnable arm: mildly adaptive ordered centers,
bandwidths, compression knees, and a shared filter shape are mixed with the fixed
physical bank through a learned residual gate. `SetTokenizerEncoder(frontend=...)`
selects either the fixed or constrained-learnable arm.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple

# ============================================================================
# PhysicalFilterbankTokenizer — default hyperparameters (justified)
# ============================================================================
# These are physically motivated, not tuned knobs. Change them in config.py for a
# run; change the DEFAULTS here only with a reason that supersedes the ones below.
#
# FB_F_MIN_HZ / FB_F_MAX_HZ — the analysis band, in physical Hz (constant, rate-
#   independent). Human activity energy lives ~0.3–15 Hz: gait cadence ~0.5–3 Hz,
#   harmonics + hand/limb motion up to ~10–12 Hz. 15 Hz is also <= the Nyquist of
#   the lowest native rate in the corpus (20 Hz -> 10 Hz Nyquist; bands above that
#   are simply Nyquist-masked per-sample, so 15 Hz costs nothing at low rates and
#   captures the extra content at 50/100 Hz). Below 0.3 Hz is quasi-DC (gravity/
#   tilt), which is handled separately by the signed DC feature, not a band.
FB_F_MIN_HZ = 0.3
FB_F_MAX_HZ = 15.0
#
# FB_N_BANDS — number of log-spaced constant-Q bands across [f_min, f_max].
#   Log spacing matches how HAR discriminability scales (fine at low freq where
#   cadence lives, coarse up high). 32 is generous; at Q=4 and ~1.5 s windows the
#   top bands overlap / are resolution-limited, so 16–24 is likely equivalent and
#   cheaper — but that is an ABLATION (measure before cutting), so the default
#   stays at the safe, higher value.
FB_N_BANDS = 32
#
# FB_Q — constant-Q quality factor (center / bandwidth). Q=4 gives ~1/4-octave
#   bands: selective enough to separate gait harmonics, wide enough that a band
#   still contains signal energy in a short (~1–1.5 s) window. Standard for
#   audio/vibration constant-Q analysis.
FB_Q = 4.0
#
# FB_DFT_SIZE (S) — the fixed zero-pad/rDFT length. MUST satisfy S >= max patch
#   length N = sampling_rate * patch_seconds over the whole corpus, because the
#   filterbank matrix is built for exactly S//2+1 bins (one fixed shape so any
#   patch length maps onto one shared filterbank; batched). Zero-padding beyond N
#   is pure frequency-domain interpolation — it does NOT change the band-energy
#   output (verified: S=256 vs 512 give band-cosine 1.000000 and identical Nyquist/
#   resolution masks at 20/25/50/100 Hz). So S only trades compute vs headroom.
#   The active multi-resolution classifier recipe emits a shared 512-sample patch tensor for
#   0.5/1/2/4-second grids. Long high-rate views are bounded before analysis, but the common
#   collate/tokenizer contract must still agree on S=512. Overflow is a hard ValueError (never
#   silent truncation), so if a future patch exceeds this the run fails loudly. Keep it a power
#   of two for FFT efficiency.
FB_DFT_SIZE = 512
#
# FB_NYQUIST_MARGIN — a band is "observable" only if center + 2*sigma <= margin *
#   (rate/2). 0.9 keeps a 10% guard below Nyquist so a band's Gaussian tail does
#   not straddle the aliasing edge.
FB_NYQUIST_MARGIN = 0.9
#
# FB_RESOLUTION_MIN_CYCLES — a band at f is "resolved" once ~this many cycles fit
#   in the window D = N/rate (a 0.3 Hz band needs ~3 s to see one cycle). Below
#   that the band is present-but-blurry and gets FLAGGED (resolution mask), not
#   zeroed. 1.0 cycle is the minimum to estimate a frequency's energy at all.
FB_RESOLUTION_MIN_CYCLES = 1.0
# ============================================================================


class PhysicalFilterbankTokenizer(nn.Module):
    """
    Physical-Hz constant-Q filterbank tokenizer (PHz-FB). The previous design rationale remains
    available through docs/LEGACY.md; the active use is documented in DESIGN_OF_RECORD.md.

    Turns each native-rate, zero-padded patch of each channel into one d_model
    token, entirely in the *physical frequency* domain, so the representation is
    rate-invariant and anti-aliased by construction (no interpolation exists in the
    path). Pipeline, per patch / per channel:

        Hann window + DC removal
          -> native-rate zero-padded rDFT (size S); bin m -> physical Hz  phi[m] = m*r/S
          -> fixed constant-Q Gaussian filterbank (K bands, centers fixed in Hz) -> E_k
          -> log1p compression + frozen per-band standardization
          -> concat[ e_hat(K), nyquist_mask(K), (resolution_flag(K)), amplitude(1) ]
          -> shared Linear(-> d_model)

    Contract (drop-in for the old extractors):
        forward(patches, sampling_rate_hz, patch_len_samples=None)
          patches:            (B, P, S, C)   native-rate, zero-padded to S
          sampling_rate_hz:   scalar | (B,)  ONE rate per sample (see note)
          patch_len_samples:  scalar | (B,) | None   true N per sample, for the Hann
                                                      window / DC / masks (None -> S)
        returns tokens:       (B, P, C, d_model)

    Rate is bound at the *sample* level, not per channel — all channels within one
    sample share one rate. This matches the corpus (each device resamples all
    channels to a common grid). Genuinely mixed-rate channels within one sample
    would need a (B, C) rate/length signature; out of scope by design.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_bands: int = FB_N_BANDS,
        f_min: float = FB_F_MIN_HZ,
        f_max: float = FB_F_MAX_HZ,
        Q: float = FB_Q,
        dft_size: int = FB_DFT_SIZE,
        nyquist_margin: float = FB_NYQUIST_MARGIN,
        learnable: bool = False,               # True -> Arm B (learnable Gaussian centers)
        use_amplitude: bool = True,
        use_dc: bool = True,                   # signed per-channel DC (gravity/tilt) feature
        use_resolution_mask: bool = True,      # low-freq mirror of the Nyquist mask
        resolution_min_cycles: float = FB_RESOLUTION_MIN_CYCLES,  # band "resolved" once ~this many cycles fit in D
        norm: str = "frozen",                  # 'frozen' | 'none' (per-band standardization)
        center_shift_fraction: float = 0.45,   # fraction of adjacent log-spacing; <0.5 preserves order
        bandwidth_factor_max: float = 1.5,
        compression_gain_max: float = 2.0,
        filter_shape_min: float = 1.5,
        filter_shape_max: float = 2.5,
        adaptive_gate_init: float = 0.1,
        use_polarization: bool = True,
        polarization_energy_kappa: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_bands = int(n_bands)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.Q = float(Q)
        self.S = int(dft_size)
        self.M = self.S // 2                    # rDFT returns M+1 bins
        self.nyquist_margin = float(nyquist_margin)
        self.use_amplitude = bool(use_amplitude)
        self.use_dc = bool(use_dc)
        self.use_resolution_mask = bool(use_resolution_mask)
        self.resolution_min_cycles = float(resolution_min_cycles)
        self.norm = norm
        self.learnable = bool(learnable)
        self.center_shift_fraction = float(center_shift_fraction)
        self.bandwidth_factor_max = float(bandwidth_factor_max)
        self.compression_gain_max = float(compression_gain_max)
        self.filter_shape_min = float(filter_shape_min)
        self.filter_shape_max = float(filter_shape_max)
        self.adaptive_gate_init = float(adaptive_gate_init)
        self.use_polarization = bool(use_polarization)
        self.polarization_energy_kappa = float(polarization_energy_kappa)
        if not 0.0 <= self.center_shift_fraction < 0.5:
            raise ValueError("center_shift_fraction must be in [0, 0.5) to preserve center order")
        if self.bandwidth_factor_max < 1.0 or self.compression_gain_max < 1.0:
            raise ValueError("adaptive multiplicative bounds must be >= 1")
        if not self.filter_shape_min < 2.0 < self.filter_shape_max:
            raise ValueError("filter-shape bounds must strictly contain the Gaussian exponent 2")
        if not 0.0 < adaptive_gate_init < 1.0:
            raise ValueError("adaptive_gate_init must be in (0, 1)")
        if not math.isfinite(self.polarization_energy_kappa) \
                or self.polarization_energy_kappa < 0.0:
            raise ValueError("polarization_energy_kappa must be finite and non-negative")

        # Log-spaced physical-Hz band centers f_1..f_K
        k = torch.arange(self.n_bands, dtype=torch.float32)
        centers = self.f_min * (self.f_max / self.f_min) ** (k / (self.n_bands - 1))
        self.register_buffer("centers", centers)
        if self.learnable:
            # Endpoints stay physically pinned. Interior shifts are less than half an
            # adjacent log-band spacing, so centers cannot cross even at their limits.
            self._center_offsets = nn.Parameter(torch.zeros(max(self.n_bands - 2, 0)))
            self._bandwidth_logits = nn.Parameter(torch.zeros(self.n_bands))
            self._compression_logits = nn.Parameter(torch.zeros(self.n_bands))
            self._shape_logit = nn.Parameter(torch.zeros(()))
            gate_logit = math.log(adaptive_gate_init / (1.0 - adaptive_gate_init))
            self._adaptive_gate_logit = nn.Parameter(torch.tensor(gate_logit))

        # Frozen per-band standardization buffers (identity until calibrated via
        # fit_norm_stats / the accumulate+finalize API over the augmented (r,D) mix).
        self.register_buffer("norm_mu", torch.zeros(self.n_bands))
        self.register_buffer("norm_sd", torch.ones(self.n_bands))
        self.register_buffer("_norm_fitted", torch.zeros(1))
        # Running accumulators for streaming calibration (not persisted). float64 to
        # avoid catastrophic cancellation in the two-pass variance (band log-energies
        # can have large means relative to their variance).
        self.register_buffer("_acc_count", torch.zeros(self.n_bands, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sum", torch.zeros(self.n_bands, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sqsum", torch.zeros(self.n_bands, dtype=torch.float64), persistent=False)

        # Frozen signed-DC standardization (scalar). DC = per-channel patch mean in
        # native units (the gravity/tilt component the band path removes in _band_energy).
        # It is SIGNED, so it does NOT go through log1p; it needs its own accumulator.
        # Stats are pooled over all channels+patches (gravity can land on any axis, so
        # one shared (mu,sd) keeps the feature axis-agnostic and unit-normalized —
        # standardizing away the m/s^2-vs-g device fingerprint while preserving the
        # relative direction/tilt across the 3 accel channels within a patch).
        self.register_buffer("dc_mu", torch.zeros(1))
        self.register_buffer("dc_sd", torch.ones(1))
        self.register_buffer("_dc_acc_count", torch.zeros(1, dtype=torch.float64), persistent=False)
        self.register_buffer("_dc_acc_sum", torch.zeros(1, dtype=torch.float64), persistent=False)
        self.register_buffer("_dc_acc_sqsum", torch.zeros(1, dtype=torch.float64), persistent=False)

        in_dim = self.n_bands + self.n_bands                 # e_hat + nyquist mask
        if self.use_resolution_mask:
            in_dim += self.n_bands                           # resolution flag
        if self.use_amplitude:
            in_dim += 1                                      # amplitude scalar
        if self.use_dc:
            in_dim += 1                                      # signed DC (gravity/tilt) scalar
        # Per-triad bounded polarization values (vertical energy share, circularity, signed spin)
        # plus gravity confidence. They are replicated onto the three axis rows of a triad so the
        # existing per-axis projection and SensorFold contract remain unchanged.
        self.polarization_dim = 3 * self.n_bands + 1 if self.use_polarization else 0
        in_dim += self.polarization_dim
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, d_model)

    # ------------------------------------------------------------------ helpers
    def _band_centers(self) -> torch.Tensor:
        if self.learnable and self.n_bands > 2:
            log_base = self.centers.log()
            left = log_base[1:-1] - log_base[:-2]
            right = log_base[2:] - log_base[1:-1]
            limit = self.center_shift_fraction * torch.minimum(left, right)
            shifted = (log_base[1:-1] + limit * torch.tanh(self._center_offsets)).exp()
            return torch.cat((self.centers[:1], shifted, self.centers[-1:]))
        return self.centers

    def _band_sigmas(self, centers: torch.Tensor, adaptive: bool) -> torch.Tensor:
        sigma = centers / (2.0 * self.Q)
        if adaptive and self.learnable:
            log_bound = math.log(self.bandwidth_factor_max)
            sigma = sigma * torch.exp(log_bound * torch.tanh(self._bandwidth_logits))
        return sigma

    def _filter_shape(self, adaptive: bool) -> torch.Tensor:
        if not (adaptive and self.learnable):
            return self.centers.new_tensor(2.0)
        midpoint = 0.5 * (self.filter_shape_min + self.filter_shape_max)
        radius = 0.5 * (self.filter_shape_max - self.filter_shape_min)
        return midpoint + radius * torch.tanh(self._shape_logit)

    def _compression_gains(self) -> torch.Tensor:
        if not self.learnable:
            return torch.ones_like(self.centers)
        return torch.exp(math.log(self.compression_gain_max) * torch.tanh(self._compression_logits))

    def adaptive_gate(self) -> torch.Tensor:
        return (torch.sigmoid(self._adaptive_gate_logit) if self.learnable
                else self.centers.new_zeros(()))

    def adaptation_parameters(self) -> tuple[nn.Parameter, ...]:
        """Analysis parameters that use the slower physical-adaptation optimizer group."""
        if not self.learnable:
            return ()
        parameters = []
        if self.n_bands > 2:
            parameters.append(self._center_offsets)
        parameters.extend((self._bandwidth_logits, self._compression_logits,
                           self._shape_logit, self._adaptive_gate_logit))
        return tuple(parameters)

    def adaptation_regularization(self) -> torch.Tensor:
        """Dimensionless pull toward the fixed physical initialization."""
        if not self.learnable:
            return self.centers.new_zeros(())
        terms = []
        if self._center_offsets.numel():
            terms.append(torch.tanh(self._center_offsets).square().mean())
        terms.extend((
            torch.tanh(self._bandwidth_logits).square().mean(),
            torch.tanh(self._compression_logits).square().mean(),
            torch.tanh(self._shape_logit).square(),
        ))
        return torch.stack(terms).mean()

    @torch.no_grad()
    def adaptation_summary(self) -> dict[str, float]:
        """Compact telemetry for detecting saturation or a collapsed residual gate."""
        centers = self._band_centers()
        center_oct = torch.log2(centers / self.centers).abs()
        bw = self._band_sigmas(centers, adaptive=True) / (centers / (2.0 * self.Q))
        gain = self._compression_gains()
        return {
            "frontend/gate": float(self.adaptive_gate()),
            "frontend/center_shift_oct_max": float(center_oct.max()),
            "frontend/bandwidth_factor_min": float(bw.min()),
            "frontend/bandwidth_factor_max": float(bw.max()),
            "frontend/compression_gain_min": float(gain.min()),
            "frontend/compression_gain_max": float(gain.max()),
            "frontend/filter_shape": float(self._filter_shape(adaptive=True)),
        }

    def _prep_rate_len(self, sampling_rate_hz, patch_len_samples, B, device, dtype, P=None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize rate to (B,) and length to (B,) or, when P is given, (B,P)."""
        if not torch.is_tensor(sampling_rate_hz):
            sampling_rate_hz = torch.as_tensor(sampling_rate_hz)
        r = sampling_rate_hz.to(device=device, dtype=dtype).reshape(-1)
        if r.numel() == 1:
            r = r.expand(B)
        assert r.numel() == B, f"sampling_rate_hz must be scalar or length B={B}, got {r.numel()}"

        if patch_len_samples is None:
            shape = (B, P) if P is not None else (B,)
            N = torch.full(shape, self.S, dtype=torch.long, device=device)
        else:
            if not torch.is_tensor(patch_len_samples):
                patch_len_samples = torch.as_tensor(patch_len_samples)
            N = patch_len_samples.to(device=device).long()
            if P is None:
                N = N.reshape(-1)
                if N.numel() == 1:
                    N = N.expand(B)
                assert N.numel() == B, f"patch_len_samples must be scalar or length B={B}"
            else:
                if N.numel() == 1:
                    N = N.reshape(1, 1).expand(B, P)
                elif N.numel() == B:
                    N = N.reshape(B, 1).expand(B, P)
                elif N.numel() == B * P:
                    N = N.reshape(B, P)
                else:
                    raise AssertionError(
                        f"patch_len_samples must be scalar, (B,), or (B,P)=({B},{P})"
                    )
        # Guardrail: native samples must fit the DFT window, else the zero-pad
        # silently becomes a truncation that destroys low-frequency resolution.
        length_valid = N.max() <= self.S
        if N.device.type == "cpu":
            if not bool(length_valid):
                n_max = int(N.max())
                raise ValueError(
                    f"patch_len_samples max {n_max} exceeds dft_size S={self.S}; raise "
                    f"dft_size so that r*D <= S for every sample."
                )
        else:
            # This invariant is checked on every tokenizer pass. Keep the CUDA hot path
            # asynchronous; a Python int conversion here serialized the entire stream.
            torch._assert_async(
                length_valid,
                f"patch_len_samples exceeds dft_size S={self.S}",
            )
        return r, N

    def _prep_source_rate(self, source_rate_hz, patch_len_samples, B, C, device, dtype, P=None):
        """Normalize source bandwidth as either one rate per recording or per channel.

        The stored timeline has one common rate, but exact multi-device recordings can combine
        channels acquired at different native rates.  Their Nyquist masks must remain channel
        specific in both calibration and the regular forward path.
        """
        supplied = torch.as_tensor(source_rate_hz, device=device, dtype=dtype)
        if supplied.ndim == 2:
            if supplied.shape != (B, C):
                raise ValueError(
                    f"per-channel source_rate_hz must have shape {(B, C)}, got {tuple(supplied.shape)}"
                )
            return supplied
        source_rate, _ = self._prep_rate_len(
            supplied, patch_len_samples, B, device, dtype, P=P,
        )
        return source_rate

    def _hann_and_valid(self, N, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-token Hann window (...,S) placed in [0,N) plus a validity mask."""
        idx = torch.arange(self.S, device=device).view(*([1] * N.ndim), self.S)
        valid = (idx < N.unsqueeze(-1)).to(dtype)
        Nf = N.to(dtype).clamp(min=2).unsqueeze(-1)
        hann = 0.5 * (1.0 - torch.cos(2 * math.pi * idx / (Nf - 1.0)))
        window = hann * valid                                      # zero outside [0, N)
        return window, valid

    def _spectral_power(self, patches, N):
        """Compute one shared normalized complex spectrum and its power.

        The complex spectrum is normalized by the Hann-window energy, so amplitude-domain band
        pooling has squared magnitude on the same scale as the ordinary band-energy path.
        """
        B, P, S, _ = patches.shape
        window, valid = self._hann_and_valid(N, patches.device, patches.dtype)  # (B,P,S)
        vm = valid.unsqueeze(-1)
        Nf = N.to(patches.dtype).clamp(min=1).view(B, P, 1, 1)
        mean = (patches * vm).sum(dim=2, keepdim=True) / Nf
        dc = mean.squeeze(2)
        x_win = (patches - mean) * vm * window.unsqueeze(-1)
        X = torch.fft.rfft(x_win, n=S, dim=2)
        win_energy = window.square().sum(dim=2).clamp(min=1e-8)
        spectrum = X / win_energy.sqrt().view(B, P, 1, 1)
        power = spectrum.real.square() + spectrum.imag.square()
        return spectrum, power, dc

    def _apply_filterbank(self, power, r, centers, sigma, shape):
        H = self._filter_weights(r, centers, sigma, shape, dtype=power.dtype)
        return torch.einsum("bkm,bpmc->bpck", H, power)

    def _filter_weights(self, r, centers, sigma, shape, *, dtype):
        m = torch.arange(self.M + 1, device=r.device, dtype=dtype)
        phi = m.unsqueeze(0) * r.unsqueeze(1) / self.S
        diff = (phi.unsqueeze(1) - centers.view(1, -1, 1)).abs()
        # clamp avoids the undefined shape-gradient 0**p * log(0) when a center lands
        # exactly on an FFT bin; the value remains numerically indistinguishable from 0.
        ratio = (diff / sigma.view(1, -1, 1)).clamp_min(1e-12)
        return torch.exp(-0.5 * ratio.pow(shape))

    def _apply_filterbank_complex(self, spectrum, r, centers, sigma, shape):
        """Complex amplitude pooling paired with :meth:`_apply_filterbank`.

        ``sqrt(H)`` is essential: a pure tone then gives ``|Z|² == E`` in the matched band.
        Complex arithmetic stays in the caller's FP32 spectral-analysis island.
        """
        H = self._filter_weights(r, centers, sigma, shape, dtype=spectrum.real.dtype)
        return torch.einsum("bkm,bpmc->bpck", H.sqrt().to(spectrum.dtype), spectrum)

    def _band_energy(self, patches, r, N):
        """
        Core DSP. (B,P,S,C) + per-sample (r,N) -> band energy E (B,P,C,K), the band
        centers / sigmas used for the observability masks, and the signed per-channel
        DC mean (B,P,C) that the band path removes (the gravity/tilt component).
        """
        B, P, S, C = patches.shape
        device, dtype = patches.device, patches.dtype
        if N.ndim == 1:
            N = N.view(B, 1).expand(B, P)
        _, power, dc = self._spectral_power(patches, N)
        centers = self._band_centers().to(device=device, dtype=dtype)
        sigma = self._band_sigmas(centers, adaptive=True).to(dtype)
        E = self._apply_filterbank(power, r, centers, sigma,
                                   self._filter_shape(adaptive=True).to(dtype))
        return E, centers, sigma, dc

    def _polarization_features(
        self,
        spectrum: torch.Tensor,
        r: torch.Tensor,
        centers: torch.Tensor,
        sigma: torch.Tensor,
        dc: torch.Tensor,
        observability: torch.Tensor,
        sensor_id: torch.Tensor | None,
        device_id: torch.Tensor | None,
        channel_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Bounded triad polarization features, replicated to the triad's axis rows.

        The recorded six-slot convention is [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].
        ``sensor_id`` is required to opt into the feature: callers without triad metadata receive
        neutral zero slots, preserving the tensor contract for generic legacy call sites.
        """
        B, P, _, C = spectrum.shape
        output = spectrum.real.new_zeros(B, P, C, self.polarization_dim)
        if sensor_id is None or C < 3:
            return output
        sensor_id = torch.as_tensor(sensor_id, device=spectrum.device, dtype=torch.long)
        if sensor_id.shape != (B, C):
            raise ValueError(f"sensor_id must have shape {(B, C)}, got {tuple(sensor_id.shape)}")
        if channel_mask is None:
            live = torch.ones(B, C, dtype=torch.bool, device=spectrum.device)
        else:
            live = torch.as_tensor(channel_mask, device=spectrum.device, dtype=torch.bool)
            if live.shape != (B, C):
                raise ValueError(
                    f"channel_mask must have shape {(B, C)}, got {tuple(live.shape)}"
                )

        # ``sqrt(H)`` amplitude pooling keeps a matched pure tone on the same scale as ordinary
        # band energy. Observability zeros interpolation-only high bands before ratios are formed.
        Z_all = self._apply_filterbank_complex(
            spectrum, r, centers, sigma, centers.new_tensor(2.0),
        ) * (observability.view(B, 1, C, self.n_bands)
             if observability.ndim == 3 else observability.view(B, 1, 1, self.n_bands)).to(spectrum.dtype)

        starts = tuple(range(0, C - 2, 3))
        sensor_device = None
        if device_id is not None:
            device_id = torch.as_tensor(device_id, device=spectrum.device, dtype=torch.long)
            if device_id.ndim != 2 or device_id.shape[0] != B:
                raise ValueError("device_id must have shape (B,N_sensor)")
            sensor_device = torch.gather(device_id, 1, sensor_id)

        for start in starts:
            stop = start + 3
            same_sensor = sensor_id[:, start:stop].eq(sensor_id[:, start:start + 1]).all(dim=1)
            triad_live = live[:, start:stop].all(dim=1) & same_sensor
            if sensor_device is None:
                gravity_dc = dc[:, :, :3]
                acc_live = live[:, :3].all(dim=1)
            else:
                triad_devices = torch.stack([sensor_device[:, value] for value in starts], dim=1)
                triad_sensors = torch.stack([sensor_id[:, value] for value in starts], dim=1)
                target_device = sensor_device[:, start:start + 1]
                candidates = triad_sensors.masked_fill(
                    triad_devices != target_device, torch.iinfo(torch.long).max,
                )
                gravity_slot = candidates.argmin(dim=1)
                triad_dc = torch.stack([dc[:, :, value:value + 3] for value in starts], dim=2)
                gather = gravity_slot.view(B, 1, 1, 1).expand(B, P, 1, 3)
                gravity_dc = torch.gather(triad_dc, 2, gather).squeeze(2)
                live_triads = torch.stack([
                    live[:, value:value + 3].all(dim=1) for value in starts
                ], dim=1)
                acc_live = torch.gather(live_triads, 1, gravity_slot[:, None]).squeeze(1)
            gravity_norm = gravity_dc.square().sum(dim=-1).sqrt()
            gravity = gravity_dc / gravity_norm.unsqueeze(-1).clamp_min(1e-8)
            grav_ok = torch.sigmoid((gravity_norm - 0.5) / 0.1) * acc_live[:, None].to(dc.dtype)
            Z = Z_all[:, :, start:stop]                                      # (B,P,3,K)
            tr = Z.real.square().add(Z.imag.square()).sum(dim=2)             # (B,P,K)
            mean_energy = tr.mean(dim=-1, keepdim=True)
            energy_gate = tr / (tr + self.polarization_energy_kappa * mean_energy).clamp_min(1e-8)

            # Im(conj(Z) x Z), expanded explicitly to keep the rotational sign convention clear.
            vx = 2.0 * (Z[:, :, 1].conj() * Z[:, :, 2]).imag
            vy = 2.0 * (Z[:, :, 2].conj() * Z[:, :, 0]).imag
            vz = 2.0 * (Z[:, :, 0].conj() * Z[:, :, 1]).imag
            rotary = torch.stack((vx, vy, vz), dim=2)                         # (B,P,3,K)
            denom = tr.clamp_min(1e-8)
            vertical_complex = (gravity.unsqueeze(-1) * Z).sum(dim=2)
            vertical = vertical_complex.real.square().add(vertical_complex.imag.square()) / denom
            circular = rotary.square().sum(dim=2).sqrt() / denom
            spin = (rotary * gravity.unsqueeze(-1)).sum(dim=2) / denom

            valid = triad_live[:, None, None].to(tr.dtype)
            vertical = vertical.clamp(0.0, 1.0) * energy_gate * grav_ok.unsqueeze(-1) * valid
            circular = circular.clamp(0.0, 1.0) * energy_gate * valid
            spin = spin.clamp(-1.0, 1.0) * energy_gate * grav_ok.unsqueeze(-1) * valid
            values = torch.cat((vertical, circular, spin, grav_ok.unsqueeze(-1)), dim=-1)
            # An incomplete/non-triad group contributes no partially meaningful metadata.
            values = values * valid
            output[:, :, start:stop] = values.unsqueeze(2).expand(B, P, 3, -1)
        return output

    def _observability_masks(self, r, N, centers, sigma, source_r=None
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nyquist observability o (B,K) and low-freq resolution flag res (B,K).

        ``source_r`` (optional, (B,)) is the rate the signal was ACQUIRED at, when that differs from
        the rate the array is stored at. Upsampling cannot create information: a stream captured at
        25 Hz and stored at 50 Hz has nothing above 12.5 Hz except interpolation artifact, so the
        Nyquist bound must come from the SOURCE rate while the DFT bin->frequency mapping keeps
        using the true array rate. Defaults to ``r`` (no resampling)."""
        dtype = r.dtype
        nyq = self.nyquist_margin * ((r if source_r is None else source_r) * 0.5)
        if nyq.ndim == 2:
            o = (centers.view(1, 1, -1) + 2.0 * sigma.view(1, 1, -1)
                 <= nyq.unsqueeze(-1)).to(dtype)                       # (B,C,K)
        else:
            o = (centers.view(1, -1) + 2.0 * sigma.view(1, -1)
                 <= nyq.view(-1, 1)).to(dtype)                          # (B,K)
        if N.ndim == 1:
            D = (N.to(dtype) / r).clamp(min=1e-6)
            res = (centers.view(1, -1) * D.view(-1, 1)
                   / self.resolution_min_cycles).clamp(0.0, 1.0)
        else:
            D = (N.to(dtype) / r.view(-1, 1)).clamp(min=1e-6)
            res = (centers.view(1, 1, -1) * D.unsqueeze(-1)
                   / self.resolution_min_cycles).clamp(0.0, 1.0)
        return o, res

    def masks(self, sampling_rate_hz, patch_len_samples=None, source_rate_hz=None
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Public: (nyquist observability o, resolution flag res), each (B, K)."""
        device, dtype = self.norm_mu.device, self.norm_mu.dtype
        B = torch.as_tensor(sampling_rate_hz).reshape(-1).numel()
        p_tensor = torch.as_tensor(patch_len_samples) if patch_len_samples is not None else None
        P = p_tensor.shape[1] if p_tensor is not None and p_tensor.ndim == 2 else None
        r, N = self._prep_rate_len(sampling_rate_hz, patch_len_samples, B, device, dtype, P=P)
        centers = self.centers.to(device=device, dtype=dtype)
        sigma = centers / (2.0 * self.Q)
        src_r = None
        if source_rate_hz is not None:
            src_r, _ = self._prep_rate_len(source_rate_hz, patch_len_samples, B, device, dtype, P=P)
        return self._observability_masks(r, N, centers, sigma, source_r=src_r)

    # ---------------------------------------------------------------- calibration
    def reset_norm_accumulator(self):
        self._acc_count.zero_()
        self._acc_sum.zero_()
        self._acc_sqsum.zero_()
        self._dc_acc_count.zero_()
        self._dc_acc_sum.zero_()
        self._dc_acc_sqsum.zero_()

    @torch.no_grad()
    def accumulate_norm_stats(self, patches, sampling_rate_hz, patch_len_samples=None,
                              patch_mask=None, channel_mask=None, source_rate_hz=None):
        """Fold one (augmented) batch into the running per-band log-energy stats.

        Only *observable* bands are folded in (Nyquist mask applied per band), so a band
        above a low-rate sample's Nyquist is not dragged toward zero by out-of-band
        filter-tail energy. This keeps the frozen mean equal to the observable-conditional
        mean, so the neutral 0 that forward() imputes for masked bands matches their mean.

        patch_mask:   optional (B, P) bool — padded/phantom patches excluded from stats.
        channel_mask: optional (B, C) bool — ABSENT (zero-filled) channels excluded, so
                      the frozen mean/sd reflect REAL sensor energy and are not dragged
                      toward the all-zero signature (~67% of the corpus is accel-only).
        source_rate_hz: optional native acquisition bandwidth. A converter may store an
                       upsampled signal at a higher rate; those interpolation-only bands must
                       not enter the observable-conditional normalization statistics.
        """
        B, P, S, C = patches.shape
        r, N = self._prep_rate_len(sampling_rate_hz, patch_len_samples, B,
                                   patches.device, patches.dtype, P=P)
        _, power, dc = self._spectral_power(patches, N)
        centers = self.centers.to(device=patches.device, dtype=patches.dtype)
        sigma = centers / (2.0 * self.Q)
        E = self._apply_filterbank(power, r, centers, sigma, centers.new_tensor(2.0))
        src_r = (self._prep_source_rate(
            source_rate_hz, patch_len_samples, B, C, patches.device, patches.dtype, P=P,
        ) if source_rate_hz is not None else None)
        o, _ = self._observability_masks(r, N, centers, sigma, source_r=src_r)
        e = torch.log1p(E).to(torch.float64)                           # (B,P,C,K)
        w = (o.view(B, 1, C, self.n_bands).expand_as(e)
             if o.ndim == 3 else o.view(B, 1, 1, self.n_bands).expand_as(e))
        w = w.to(torch.float64)
        if patch_mask is not None:
            w = w * patch_mask.view(B, P, 1, 1).to(torch.float64)      # exclude padded patches
        if channel_mask is not None:
            w = w * channel_mask.view(B, 1, C, 1).to(torch.float64)    # exclude absent channels
        e = e.reshape(-1, self.n_bands)
        w = w.reshape(-1, self.n_bands)
        self._acc_count += w.sum(dim=0)                                 # per-band counts
        self._acc_sum += (e * w).sum(dim=0)
        self._acc_sqsum += (e * e * w).sum(dim=0)

        # Signed DC (gravity/tilt) stats: pooled over all channels+patches (one scalar
        # mu/sd), so the feature is axis-agnostic and the m/s^2-vs-g scale is normalized
        # out. No log1p (dc is signed). Padded patches + absent channels excluded.
        if self.use_dc:
            d = dc.to(torch.float64).reshape(B, P, -1)                 # (B,P,C)
            dw = torch.ones_like(d)
            if patch_mask is not None:
                dw = dw * patch_mask.view(B, P, 1).to(torch.float64)
            if channel_mask is not None:
                dw = dw * channel_mask.view(B, 1, C).to(torch.float64)
            d = d.reshape(-1)
            dw = dw.reshape(-1)
            self._dc_acc_count += dw.sum()
            self._dc_acc_sum += (d * dw).sum()
            self._dc_acc_sqsum += (d * d * dw).sum()

    @torch.no_grad()
    def finalize_norm_stats(self, eps: float = 1e-5):
        """Set frozen mu/sd from the per-band accumulators. Call once after calibration.

        Bands never observed during calibration fall back to identity (mu=0, sd=1) so an
        unseen-but-later-observable band cannot blow up e_hat at inference.
        """
        seen = self._acc_count > 0
        safe_n = self._acc_count.clamp(min=1.0)
        mu = self._acc_sum / safe_n
        var = (self._acc_sqsum / safe_n) - mu * mu
        sd = var.clamp(min=eps).sqrt()
        mu = torch.where(seen, mu, torch.zeros_like(mu))
        sd = torch.where(seen, sd, torch.ones_like(sd))
        self.norm_mu.copy_(mu)
        self.norm_sd.copy_(sd)

        # Frozen signed-DC standardization (scalar). Falls back to identity (0,1) if
        # the DC feature is disabled or never accumulated.
        if self.use_dc and self._dc_acc_count.item() > 0:
            dc_n = self._dc_acc_count.clamp(min=1.0)
            dc_mu = self._dc_acc_sum / dc_n
            dc_var = (self._dc_acc_sqsum / dc_n) - dc_mu * dc_mu
            dc_sd = dc_var.clamp(min=eps).sqrt()
            self.dc_mu.copy_(dc_mu.to(self.dc_mu.dtype))
            self.dc_sd.copy_(dc_sd.to(self.dc_sd.dtype))
        else:
            self.dc_mu.zero_()
            self.dc_sd.fill_(1.0)
        self._norm_fitted.fill_(1.0)

    @torch.no_grad()
    def fit_norm_stats(self, patches, sampling_rate_hz, patch_len_samples=None, eps: float = 1e-5,
                       source_rate_hz=None):
        """Convenience one-shot calibration over a single (large) batch."""
        self.reset_norm_accumulator()
        self.accumulate_norm_stats(
            patches, sampling_rate_hz, patch_len_samples, source_rate_hz=source_rate_hz)
        self.finalize_norm_stats(eps)

    # --------------------------------------------------------------------- forward
    def analyze(self, patches, sampling_rate_hz, patch_len_samples=None,
                source_rate_hz=None, sensor_id=None, device_id=None,
                channel_mask=None) -> torch.Tensor:
        """Physical feature (B, P, C, in_dim) — everything BEFORE the learnable projection.

        Split out from ``forward`` so callers holding two copies of this module can share the
        expensive part. On the FIXED arm this whole path is parameter-free (rDFT, constant-Q
        filterbank against ``self.centers``, plus frozen norm/DC buffers), so an EMA teacher's
        analysis is bit-identical to the student's and recomputing it is pure waste. On the
        LEARNABLE arm the adaptive residual branch reads parameters that EMA-diverge, so the
        teacher must run its own ``analyze``.
        """
        B, P, S, C = patches.shape
        assert S == self.S, (
            f"patch time dim {S} != dft_size {self.S}; zero-pad patches to S before the tokenizer"
        )
        device, dtype = patches.device, patches.dtype
        r, N = self._prep_rate_len(sampling_rate_hz, patch_len_samples, B, device, dtype, P=P)
        spectrum, power, dc = self._spectral_power(patches, N)
        fixed_centers = self.centers.to(device=device, dtype=dtype)
        fixed_sigma = fixed_centers / (2.0 * self.Q)
        E_fixed = self._apply_filterbank(power, r, fixed_centers, fixed_sigma,
                                         fixed_centers.new_tensor(2.0))

        # Arm B remains anchored to the calibrated fixed log-energy feature. Only the
        # compressed adaptive residual is mixed in; amplitude/DC/metadata stay physical.
        e_fixed = torch.log1p(E_fixed)
        if self.learnable:
            centers = self._band_centers().to(device=device, dtype=dtype)
            sigma = self._band_sigmas(centers, adaptive=True).to(dtype)
            E_adapt = self._apply_filterbank(power, r, centers, sigma,
                                             self._filter_shape(adaptive=True).to(dtype))
            gains = self._compression_gains().to(device=device, dtype=dtype)
            e_adapt = torch.log1p(E_adapt * gains.view(1, 1, 1, -1))
            e = e_fixed + self.adaptive_gate().to(dtype) * (e_adapt - e_fixed)
        else:
            e = e_fixed
        e_hat = (e - self.norm_mu) / self.norm_sd if self.norm == "frozen" else e

        # Amplitude scalar: total log-energy, preserves absolute magnitude.
        amp = torch.log1p(E_fixed.sum(dim=-1, keepdim=True))        # (B,P,C,1)

        # Signed DC (gravity/tilt) feature: the per-channel patch mean the band path
        # removed. Frozen-standardized (scalar mu/sd) so the m/s^2-vs-g device scale is
        # normalized out and only the relative cross-channel gravity direction survives.
        # This restores static-posture discrimination (stand/sit/lie differ only in DC).
        dc_feat = ((dc - self.dc_mu) / self.dc_sd).unsqueeze(-1) if self.norm == "frozen" \
            else dc.unsqueeze(-1)                                    # (B,P,C,1)

        # Nyquist observability mask (o) zeroes bands above native Nyquist (neutral,
        # since e_hat is standardized). Resolution flag (res) is the low-freq mirror:
        # a band at f_k needs ~resolution_min_cycles cycles within D=N/r to be resolved;
        # below that the value is present-but-blurry, so we *flag* it rather than zero it.
        src_r = (self._prep_source_rate(
            source_rate_hz, patch_len_samples, B, C, device, dtype, P=P,
        ) if source_rate_hz is not None else None)
        o, res = self._observability_masks(r, N, fixed_centers, fixed_sigma, source_r=src_r)
        o_bpck = (o.view(B, 1, C, self.n_bands).expand(B, P, C, self.n_bands)
                  if o.ndim == 3 else o.view(B, 1, 1, self.n_bands).expand(B, P, C, self.n_bands))
        e_hat = e_hat * o_bpck

        feats = [e_hat, o_bpck]
        if self.use_resolution_mask:
            if res.ndim == 2:
                res = res.view(B, 1, self.n_bands).expand(B, P, self.n_bands)
            feats.append(res.unsqueeze(2).expand(B, P, C, self.n_bands))

        if self.use_amplitude:
            feats.append(amp)

        if self.use_dc:
            feats.append(dc_feat)

        if self.use_polarization:
            feats.append(self._polarization_features(
                spectrum, r, fixed_centers, fixed_sigma, dc, o, sensor_id, device_id, channel_mask,
            ))

        return torch.cat(feats, dim=-1)                            # (B,P,C,in_dim)

    def project(self, token_in: torch.Tensor) -> torch.Tensor:
        """The learnable half: (B,P,C,in_dim) -> (B,P,C,d_model)."""
        return self.proj(token_in)

    @torch.no_grad()
    def copy_normalization_from(self, other: "PhysicalFilterbankTokenizer") -> None:
        """Copy only frozen corpus calibration, never the learned projection.

        The JEPA physical-target analyzer intentionally has a smaller input when the student
        enables polarization.  It must share the fixed feature calibration, but copying a full
        state dict would incorrectly couple its projection shape to the student frontend.
        """
        if self.n_bands != other.n_bands:
            raise ValueError("cannot share normalization across different band counts")
        for name in ("norm_mu", "norm_sd", "dc_mu", "dc_sd", "_norm_fitted"):
            getattr(self, name).copy_(getattr(other, name))

    def forward(self, patches, sampling_rate_hz, patch_len_samples=None,
                source_rate_hz=None, sensor_id=None, device_id=None,
                channel_mask=None) -> torch.Tensor:
        return self.project(
            self.analyze(
                patches, sampling_rate_hz, patch_len_samples, source_rate_hz,
                sensor_id=sensor_id, device_id=device_id, channel_mask=channel_mask,
            )
        )
