"""Continuous-time kernel tokenizer (CKT) — a rate-flexible, shape-preserving front end.

Design of record: `docs/design/CONTINUOUS_KERNEL_FRONTEND.md`. This module is integrated as the
`--frontend continuous` arm in Phase A and end-to-end episodic training.

WHAT IT IS
----------
A bank of `K` band-pass kernels, each defined as a CONTINUOUS function of real time and sampled at
whatever rate the recording arrives at. Kernel `k` spans `T_k` **seconds** and is parametrised in
normalised time `u = t/T_k` as a truncated Fourier series with `M` harmonics::

    w_k(u) = exp(-u^2 / 2 sigma_k^2) * sum_{m=1..M} [ a_km cos(2 pi m u) + b_km sin(2 pi m u) ]

`M = 12` gives **24 coefficients per kernel**. Three properties fall out of this basis, and each one
solves a problem a spline-through-control-points would leave open:

* harmonic `m` sits at exactly `m / T_k` Hz, so **band-limiting is a coefficient mask** rather than a
  smoothing heuristic — the kernel can never alias when sampled;
* there is no `m = 0` term, so `int w = 0` by construction and kernels **cannot measure gravity**
  (the signed DC component stays a separate feature, as in the filterbank);
* setting `a_k,carrier = 1` makes the kernel a Gabor wavelet, so **initialisation reproduces the
  physical filterbank** and any later difference is attributable to learning rather than to a
  different starting point.

BANK GEOMETRY (2026-09-09)
--------------------------
Centres are log-spaced on `[f_min, f_max]`. A kernel spans as many WHOLE cycles of its centre as
fit in `t_max`, up to `n_cycles`::

    cycles_k = clamp(floor(f_k * t_max), 1, n_cycles),   T_k = cycles_k / f_k,   carrier = cycles_k

so the carrier harmonic sits at exactly `f_k` for every kernel and every (span, carrier) pair is
distinct. The previous rule (`T_k = clamp(n_cycles / f_k, t_min, t_max)`, carrier rounded) snapped
all sixteen kernels below 2 Hz onto the 0.5 Hz grid of a 2 s span: eight of them were the same
0.59 Hz kernel, four the same 1 Hz kernel, and only 20 of 32 kernels were distinct. Nothing below
`1 / t_max` can hold one cycle, so `f_min` defaults to 0.5 Hz; the 0.3-0.5 Hz range was never
reachable and is quasi-DC territory handled by the signed DC feature anyway.

WHY IT MIGHT BEAT THE FILTERBANK (and why it might not)
-------------------------------------------------------
The filterbank is already rate-invariant, by working in physical frequency. What it cannot represent
is *waveform shape within a band* (it keeps magnitude, discards phase) and *sub-second timing* (one
vector per patch). Three of the four motion scales that matter — impact transients ~60 ms, arm-swing
reversal ~150 ms, step ~500 ms — are invisible at a 1 s token boundary. This module keeps them by
emitting several ordered frames per token.

Against that: our own constrained-learnable filterbank arm measured **inert**, and a frozen
filterbank probe beat a raw-waveform CNN of the same budget. The prior is unfavourable, which is why
the design doc pre-registers a kill criterion on **cross-rate transfer** rather than accuracy.

THE THREE RULES THAT MAKE RATES COMPARABLE (all measured; see the design doc S12)
--------------------------------------------------------------------------------
1. **Evaluate the kernel at the exact offsets of the real samples.** For output time `t_f`, use
   `w_k((t_n - t_f) / T_k)` at the true sample times `t_n = n/r`. Rounding the frame centre to the
   nearest sample costs 25 ms of jitter at 20 Hz = 45 degrees of phase error at 5 Hz, and drops
   cross-rate correlation from 0.986 to 0.844. Never resample the signal; let the kernel absorb the
   sub-sample offset. That is the entire point of it being continuous.
2. **Convolution is an INTEGRAL: multiply by `dt = 1/r`.** A plain sum makes a 20 Hz recording return
   one fifth the response of the same physical motion at 100 Hz, and the model then learns to
   identify the device from response magnitude. L1 normalisation is equivalent; **L2 is actively
   harmful** (it rescales per rate and destroys amplitude comparability).
3. **Re-zero-mean after sampling.** Discretisation breaks the exact `int w = 0`.
4. **The observability mask is the fraction of a kernel's COEFFICIENT ENERGY that survives the
   Nyquist cut, not the fraction of its harmonic slots.** At Gabor initialisation that is exactly
   1 when the carrier is live and 0 otherwise. The earlier slot-count fraction multiplied an intact
   8 Hz kernel by 0.33 at 20 Hz and by 0.5-0.67 at 50 Hz: a per-rate rescaling of the standardised
   response, i.e. the device fingerprint Rule 2 exists to prevent.

Measured cross-rate agreement with all three applied, against 100 Hz: **0.986 at 20 Hz, 0.987 at
25 Hz, 0.998 at 50 Hz**. The residual is honest quadrature error (20 taps vs 100 taps approximating
the same integral), not a defect.

CONTRACT
--------
`forward(patches, sampling_rate_hz, patch_len_samples=None, source_rate_hz=None,
patch_mask=None) -> (B, P, C, d_model)` — identical to `PhysicalFilterbankTokenizer`. Stored and
native acquisition rates may vary across a batch; token count is a function of duration only.

The module is built for ONE physical patch duration (`patch_seconds`, default 1 s): each token
packages `frames_per_second * patch_seconds` analysis frames, halved by the stride-2 stage. The
analysis frames are laid out in physical time from the patch grid, so `analyze` refuses a batch
whose full patches are not `patch_seconds` long instead of silently analysing the wrong seconds.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------
# Defaults — physically motivated, mirroring filterbank.py's constants where they correspond.
# ---------------------------------------------------------------------------------------------
CK_N_KERNELS = 32          # one per band; same count and log grid as FB_N_BANDS
CK_N_HARMONICS = 12        # -> 24 coefficients per kernel (12 cos + 12 sin).
                           # Chosen by measuring representational capacity: the fraction of an
                           # arbitrary kernel shape's variance M harmonics can express is
                           #   shape                       M=8     M=12    M=16
                           #   asymmetric impact (heel strike)  0.76   0.84   0.88
                           #   chirp                            0.62   0.92   0.99
                           #   sharp spike                      0.75   0.92   0.98
                           # M=8 is marginal for exactly the shapes this front end exists to
                           # capture. The cost is rate-fragility: a kernel is fully representable
                           # only where f_k <= 1.8*r/M, so at 20 Hz M=8 keeps 22/32 kernels intact
                           # and M=12 keeps 19/32. Cross-rate correlation is unaffected for the
                           # kernels that ARE intact (0.9948 vs 0.9946). M=12 buys real shape
                           # capacity for three more Nyquist-masked bands at the lowest rate.
CK_F_MIN_HZ = 0.5          # = 1 / CK_T_MAX_S: the lowest frequency with one whole cycle in the
                           # longest span. FB_F_MIN_HZ is 0.3, but no kernel of <= 2 s can carry
                           # 0.3 Hz; below 0.5 Hz is quasi-DC, handled by the signed DC feature.
CK_F_MAX_HZ = 15.0         # == FB_F_MAX_HZ; source-rate masking limits low-rate recordings
CK_N_CYCLES = 4.0          # cycles of the centre frequency inside the span -> carrier at harmonic 4,
                           # so harmonics 1..8 span [f/4, 2f]: constant RELATIVE bandwidth (constant-Q
                           # arrived at from the time side rather than the frequency side)
CK_T_MIN_S = 0.05
CK_T_MAX_S = 2.0           # a 4 s kernel in a 6 s window is 67% edge-padded; 2.0 keeps two-thirds of
                           # frames fully supported for EVERY kernel and still reaches 2 cycles at 1 Hz
CK_FRAMES_PER_SECOND = 8   # output frame rate: 125 ms, resolves the arm-swing and step scales
CK_NYQUIST_MARGIN = 0.9    # == FB_NYQUIST_MARGIN
CK_ENVELOPE_SIGMA = 0.22   # Gaussian envelope width in normalised time
CK_FRAME_CHUNK = 24        # amortize gather/einsum launches while keeping allocator pressure modest
CK_PATCH_SECONDS = 1.0     # physical duration of one token; frames_per_second * this must be even
CK_SIGMA_MIN = 0.05        # envelope width as a fraction of span; an experiment capacity bound
CK_SIGMA_MAX = 0.50
CK_GAIN_MAX = 2.0          # bounds learned amplification independently of coefficient norm


class ContinuousKernelTokenizer(nn.Module):
    """Continuous-time kernels followed by a conventional dense sensor-triad CNN.

    The physical kernels are shared across axes.  Their responses are then packed into fixed xyz
    slots for each accelerometer or gyroscope before the learned CNN, so the CNN can model joint
    directional motion without mixing distinct sensors or placements.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_kernels: int = CK_N_KERNELS,
        n_harmonics: int = CK_N_HARMONICS,
        f_min: float = CK_F_MIN_HZ,
        f_max: float = CK_F_MAX_HZ,
        n_cycles: float = CK_N_CYCLES,
        t_min: float = CK_T_MIN_S,
        t_max: float = CK_T_MAX_S,
        frames_per_second: int = CK_FRAMES_PER_SECOND,
        conv_channels: Tuple[int, int] = (64, 128),
        nyquist_margin: float = CK_NYQUIST_MARGIN,
        envelope_sigma: float = CK_ENVELOPE_SIGMA,
        gabor_init: bool = True,
        norm: str = "frozen",
        patch_seconds: float = CK_PATCH_SECONDS,
    ):
        super().__init__()
        self._validate_analysis_config(n_harmonics, nyquist_margin, norm)
        if not all(math.isfinite(v) for v in (f_min, f_max, t_min, t_max, patch_seconds)):
            raise ValueError("frequency, span and patch settings must be finite")
        if not 0 < f_min < f_max:
            raise ValueError("frequencies must satisfy 0 < f_min < f_max")
        if n_kernels < 2 or n_harmonics < 1:
            raise ValueError("need at least 2 kernels and 1 harmonic")
        if not 0.0 < t_min < t_max:
            raise ValueError("require 0 < t_min < t_max")
        if frames_per_second < 2 or frames_per_second % 2:
            raise ValueError("frames_per_second must be even and >= 2 (one stride-2 stage)")
        if not 1 <= int(n_cycles) <= n_harmonics or n_cycles != int(n_cycles):
            raise ValueError("n_cycles must be a whole number of carrier harmonics within n_harmonics")
        if f_min * t_max < 1.0 - 1e-9:
            raise ValueError(
                f"f_min={f_min} Hz cannot hold one whole cycle in t_max={t_max} s; raise f_min to "
                f"at least {1.0 / t_max:.3f} Hz or lengthen t_max"
            )
        frames_per_patch = float(frames_per_second) * float(patch_seconds)
        if patch_seconds <= 0 or abs(frames_per_patch - round(frames_per_patch)) > 1e-6 \
                or int(round(frames_per_patch)) % 2:
            raise ValueError(
                f"patch_seconds={patch_seconds} must make frames_per_second * patch_seconds an even "
                f"integer (got {frames_per_patch}) so one stride-2 stage yields whole frames per token"
            )
        self.K = int(n_kernels)
        self.register_buffer("_frontend_revision", torch.tensor(2, dtype=torch.long))
        self.M = int(n_harmonics)
        self.F = int(frames_per_second)
        self.patch_seconds = float(patch_seconds)
        self.frames_per_patch = int(round(frames_per_patch))
        self.d_model = int(d_model)
        self.nyquist_margin = float(nyquist_margin)
        self.envelope_sigma = float(envelope_sigma)
        self.norm = norm
        self.learnable = True
        self.emits_sensor_tokens = True
        self.axes_per_sensor = 3
        self._geometry_cache: dict[tuple, dict[str, torch.Tensor]] = {}
        self._telemetry_requested = False
        self._runtime_summary: dict[str, torch.Tensor] = {}
        self.sigma_min = CK_SIGMA_MIN
        self.sigma_max = CK_SIGMA_MAX
        self.gain_max = CK_GAIN_MAX

        # --- band centres and spans (fixed physics, not learned) ---
        # Each kernel spans as many whole cycles of its centre as fit in t_max (at most n_cycles),
        # so its carrier harmonic sits at EXACTLY f_k and no two kernels coincide. See the module
        # docstring, "bank geometry".
        k = torch.arange(self.K, dtype=torch.float32)
        centres = f_min * (f_max / f_min) ** (k / (self.K - 1))
        cycles = torch.clamp(torch.floor(centres * t_max + 1e-6), 1.0, float(n_cycles))
        spans = cycles / centres
        if bool((spans < t_min - 1e-9).any()):
            raise ValueError("a kernel span fell below t_min; lower t_min or f_max")
        self.max_span = float(spans.max())
        self.register_buffer("centres", centres)                 # (K,) Hz
        self.register_buffer("spans", spans)                     # (K,) seconds
        self.register_buffer("carrier", cycles.long())           # (K,) harmonic index == cycles

        # Coefficient direction describes kernel shape; a separate bounded gain describes scale.
        # Normalising the coefficient pair removes an otherwise unidentifiable scale degree of
        # freedom between coefficients and gain.
        self.cos_coeff = nn.Parameter(torch.zeros(self.K, self.M))
        self.sin_coeff = nn.Parameter(torch.zeros(self.K, self.M))
        sigma_fraction = (envelope_sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)
        if not 0.0 < sigma_fraction < 1.0:
            raise ValueError(
                f"envelope_sigma must be in ({self.sigma_min}, {self.sigma_max}), got "
                f"{envelope_sigma}"
            )
        self.sigma_logit = nn.Parameter(
            torch.full((self.K,), math.log(sigma_fraction / (1.0 - sigma_fraction)))
        )
        self.gain_logit = nn.Parameter(torch.zeros(self.K))
        if gabor_init:
            # A Gabor wavelet at the carrier: the quadrature pair is (cos, sin) of that harmonic,
            # so at step 0 this bank approximates the physical filterbank we already trust.
            with torch.no_grad():
                self.cos_coeff[torch.arange(self.K), self.carrier - 1] = 1.0
        else:
            nn.init.normal_(self.cos_coeff, std=0.3)
            nn.init.normal_(self.sin_coeff, std=0.3)
        self.register_buffer("initial_cos_coeff", self._normalised_coefficients()[0].detach().clone())
        self.register_buffer("initial_sin_coeff", self._normalised_coefficients()[1].detach().clone())
        self.register_buffer("initial_sigma", torch.full((self.K,), float(envelope_sigma)))

        # --- frozen per-kernel standardisation of the compressed magnitude ---
        self.register_buffer("norm_mu", torch.zeros(self.K))
        self.register_buffer("norm_sd", torch.ones(self.K))
        self.register_buffer("_norm_fitted", torch.zeros(1))
        self.register_buffer("_acc_count", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sum", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sqsum", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        for name in ("amp", "dc"):
            self.register_buffer(f"{name}_mu", torch.zeros(()))
            self.register_buffer(f"{name}_sd", torch.ones(()))
            self.register_buffer(f"_{name}_acc_count", torch.zeros((), dtype=torch.float64),
                                 persistent=False)
            self.register_buffer(f"_{name}_acc_sum", torch.zeros((), dtype=torch.float64),
                                 persistent=False)
            self.register_buffer(f"_{name}_acc_sqsum", torch.zeros((), dtype=torch.float64),
                                 persistent=False)

        # --- ordinary dense CNN over all xyz kernel responses on the fixed real-time grid ---
        c1, c2 = conv_channels
        self.conv1 = nn.Conv1d(self.axes_per_sensor * self.K, c1, 3, stride=2, padding=1)
        # LayerNorm over the CHANNEL dim at each frame -- never over time. `GroupNorm(1, C)` looks
        # equivalent but pools over (C, L), which makes a frame's features depend on how long the
        # rest of the recording was: measured, a 2 s and a 6 s window sharing their first 2 s got
        # tokens differing by 2.43 for that shared span. That is a duration fingerprint, the same
        # class of leak as a rate fingerprint, and it is what design-doc Rule 5 forbids.
        self.ln1 = nn.LayerNorm(c1)
        self.conv2 = nn.Conv1d(c1, c2, 3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(c2)
        self.frames_per_token = self.frames_per_patch // 2       # after the single stride-2
        self.in_dim = (c2 * self.frames_per_token   # ORDERED frames — the point of the design
                       + self.K                     # nyquist mask
                       + self.K                     # resolution flag
                       + self.K * self.frames_per_token  # per-kernel, per-frame real support
                       + self.axes_per_sensor       # per-axis amplitude
                       + self.axes_per_sensor       # per-axis signed DC
                       + self.axes_per_sensor)      # axis-validity bits
        self.proj = nn.Linear(self.in_dim, self.d_model)

    @staticmethod
    def _validate_analysis_config(n_harmonics: int, nyquist_margin: float, norm: str) -> None:
        if not math.isfinite(n_harmonics) or n_harmonics < 1 or int(n_harmonics) != n_harmonics:
            raise ValueError("n_harmonics must be a positive integer")
        if not math.isfinite(nyquist_margin) or not 0 < nyquist_margin <= 1:
            raise ValueError("nyquist_margin must be finite and in (0, 1]")
        if norm not in {"frozen", "none"}:
            raise ValueError("norm must be 'frozen' or 'none'")

    def _normalised_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        joined = torch.cat((self.cos_coeff, self.sin_coeff), dim=1)
        joined = F.normalize(joined, dim=1, eps=1e-8)
        return joined[:, :self.M], joined[:, self.M:]

    def _sigmas(self) -> torch.Tensor:
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(self.sigma_logit)

    def _gains(self) -> torch.Tensor:
        return torch.exp(math.log(self.gain_max) * torch.tanh(self.gain_logit))

    def adaptation_parameters(self) -> tuple[nn.Parameter, ...]:
        """Mildly learned analysis-bank parameters, excluding the ordinary CNN/projection."""
        return (self.cos_coeff, self.sin_coeff, self.sigma_logit, self.gain_logit)

    def adaptation_regularization(self) -> torch.Tensor:
        """Dimensionless pull toward the physical Gabor initialisation."""
        cos_coeff, sin_coeff = self._normalised_coefficients()
        shape = (cos_coeff - self.initial_cos_coeff).square().mean()
        shape = shape + (sin_coeff - self.initial_sin_coeff).square().mean()
        sigma = ((self._sigmas() - self.initial_sigma) /
                 (self.sigma_max - self.sigma_min)).square().mean()
        gain = torch.tanh(self.gain_logit).square().mean()
        return torch.stack((shape, sigma, gain)).mean()

    @torch.no_grad()
    def adaptation_summary(self) -> dict[str, float]:
        cos_coeff, sin_coeff = self._normalised_coefficients()
        displacement = torch.cat((cos_coeff - self.initial_cos_coeff,
                                  sin_coeff - self.initial_sin_coeff), dim=1).norm(dim=1)
        sigma = self._sigmas()
        gain = self._gains()
        return {
            "frontend/kernel_shape_shift_mean": float(displacement.mean()),
            "frontend/kernel_shape_shift_max": float(displacement.max()),
            "frontend/kernel_sigma_min": float(sigma.min()),
            "frontend/kernel_sigma_max": float(sigma.max()),
            "frontend/kernel_gain_min": float(gain.min()),
            "frontend/kernel_gain_max": float(gain.max()),
        }

    def request_runtime_telemetry(self, enabled: bool = True) -> None:
        self._telemetry_requested = bool(enabled)

    @torch.no_grad()
    def runtime_summary(self) -> dict[str, float]:
        return {name: float(value) for name, value in self._runtime_summary.items()}

    # ------------------------------------------------------------------ kernel construction
    def kernel_at(self, offsets: torch.Tensor, rate_hz: float,
                  source_rate_hz: Optional[float] = None) -> torch.Tensor:
        """Evaluate every kernel at exact time offsets in SECONDS -> (K, 2, n) quadrature pair.

        `offsets` is (n,) seconds relative to an output-frame centre. Returns cos-phase and
        sin-phase kernels so the magnitude |z| is phase-invariant like a band energy.
        """
        return self._kernel_frames(offsets.unsqueeze(0), rate_hz, source_rate_hz)[0]

    def _kernel_frames(self, offsets: torch.Tensor, rate_hz: float,
                       source_rate_hz: Optional[float] = None) -> torch.Tensor:
        """Vectorized kernel evaluation for offsets shaped (frames, taps)."""
        spans = self.spans.to(offsets.device)                     # (K,)
        u = offsets.unsqueeze(1) / spans.view(1, self.K, 1)       # (F,K,n)
        inside = u.abs() <= 0.5
        m = torch.arange(1, self.M + 1, device=offsets.device, dtype=offsets.dtype)  # (M,)
        phase = 2 * math.pi * m.view(1, 1, self.M, 1) * u.unsqueeze(2)  # (F,K,M,n)
        # Rule 1 of the rate contract: per-harmonic band-limit. Harmonic m sits at m/T_k Hz, so
        # this is an exact mask on a coefficient, never a smoothing approximation.
        observable_rate = rate_hz if source_rate_hz is None else source_rate_hz
        limit = self.nyquist_margin * observable_rate / 2.0
        live = (m.view(1, self.M) / spans.unsqueeze(1)) <= limit                     # (K, M)
        cos_coeff, sin_coeff = self._normalised_coefficients()
        cos_c = (cos_coeff.to(offsets.dtype) * live).view(1, self.K, self.M, 1)
        sin_c = (sin_coeff.to(offsets.dtype) * live).view(1, self.K, self.M, 1)
        cos_k = (cos_c * torch.cos(phase)).sum(2)
        sin_k = (sin_c * torch.cos(phase)).sum(2)
        # quadrature partner: same coefficients, sine basis
        cos_q = (cos_c * torch.sin(phase)).sum(2)
        sin_q = (sin_c * torch.sin(phase)).sum(2)
        real = cos_k + sin_q
        imag = sin_k - cos_q
        sigma = self._sigmas().to(offsets.dtype).view(1, self.K, 1)
        env = torch.exp(-u.pow(2) / (2 * sigma.pow(2)))
        real, imag = real * env * inside, imag * env * inside
        pair = torch.stack((real, imag), dim=2)                                       # (F,K,2,n)
        # Rule 3: re-zero-mean after sampling (discretisation breaks the exact integral).
        count = inside.sum(-1, keepdim=True).clamp_min(1).unsqueeze(2)
        pair = (pair - pair.sum(-1, keepdim=True) / count) * inside.unsqueeze(2)
        # Rule 2: convolution is an INTEGRAL. dt = 1/r. Never L2-normalise here.
        gain = self._gains().to(offsets.dtype).view(1, self.K, 1, 1)
        return pair * gain / float(rate_hz)

    def _frame_geometry(self, rate: float, P: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Cache rate-dependent sample geometry and Fourier bases, never learned values."""
        key = (device.type, device.index, float(rate), int(P))
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached
        half = int(math.ceil(self.max_span * rate / 2.0)) + 2
        offsets_base = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
        frame = torch.arange(P * self.frames_per_patch, device=device, dtype=torch.float32)
        frame_time = (frame + 0.5) / self.F
        centre = torch.floor(frame_time * rate)
        sample_index = centre.unsqueeze(1) + offsets_base.unsqueeze(0)
        fractional_phase = frame_time * rate - centre
        unique_phase, phase_id = torch.unique(fractional_phase, return_inverse=True)
        unique_offsets = (offsets_base.unsqueeze(0) - unique_phase.unsqueeze(1)) / rate
        spans = self.spans.to(device)
        u = unique_offsets.unsqueeze(1) / spans.view(1, self.K, 1)
        inside = u.abs() <= 0.5
        harmonic = torch.arange(1, self.M + 1, device=device, dtype=torch.float32)
        phase = 2 * math.pi * harmonic.view(1, 1, self.M, 1) * u.unsqueeze(2)
        support = inside.index_select(0, phase_id)
        cached = {
            "sample_index": sample_index.long(),
            "frame_time": frame_time,
            "phase_id": phase_id,
            "u": u,
            "inside": inside,
            "cos_phase": torch.cos(phase),
            "sin_phase": torch.sin(phase),
            "support": support,
        }
        self._geometry_cache[key] = cached
        return cached

    def _kernels_from_geometry(self, geometry: dict[str, torch.Tensor], rate: float,
                               source_rate: float) -> torch.Tensor:
        """Apply current learnable coefficients to cached constant bases."""
        device = self.spans.device
        harmonic = torch.arange(1, self.M + 1, device=device, dtype=self.spans.dtype)
        live = (harmonic.view(1, self.M) / self.spans.unsqueeze(1)) <= (
            self.nyquist_margin * source_rate / 2.0)
        cos_coeff, sin_coeff = self._normalised_coefficients()
        cos_coeff = (cos_coeff * live).view(1, self.K, self.M, 1)
        sin_coeff = (sin_coeff * live).view(1, self.K, self.M, 1)
        cos_phase = geometry["cos_phase"]
        sin_phase = geometry["sin_phase"]
        cos_k = (cos_coeff * cos_phase).sum(2)
        sin_k = (sin_coeff * cos_phase).sum(2)
        cos_q = (cos_coeff * sin_phase).sum(2)
        sin_q = (sin_coeff * sin_phase).sum(2)
        real, imag = cos_k + sin_q, sin_k - cos_q
        sigma = self._sigmas().view(1, self.K, 1)
        envelope = torch.exp(-geometry["u"].square() / (2 * sigma.square()))
        inside = geometry["inside"]
        pair = torch.stack((real * envelope * inside, imag * envelope * inside), dim=2)
        count = inside.sum(-1, keepdim=True).clamp_min(1).unsqueeze(2)
        pair = (pair - pair.sum(-1, keepdim=True) / count) * inside.unsqueeze(2)
        pair = pair * self._gains().view(1, self.K, 1, 1) / float(rate)
        return pair.index_select(0, geometry["phase_id"])

    # ------------------------------------------------------------------ masks
    @staticmethod
    def _rate_vector(value, B: int, device: torch.device, dtype: torch.dtype,
                     name: str) -> torch.Tensor:
        value = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
        if value.numel() == 1:
            value = value.expand(B)
        if value.numel() != B:
            raise ValueError(f"{name} must be scalar or length B={B}, got {value.numel()}")
        positive = ((value > 0) & torch.isfinite(value)).all()
        if value.device.type == "cpu":
            if not bool(positive):
                raise ValueError(f"{name} must contain only finite positive rates")
        else:
            torch._assert_async(positive, f"{name} must contain only finite positive rates")
        return value

    def masks(self, rate_hz, duration_s: torch.Tensor,
              source_rate_hz=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """(nyquist observability, resolution flag), each (B, K).

        `nyquist`: fraction of this kernel's normalised COEFFICIENT ENERGY carried by harmonics
        representable at the source rate. It is exactly binary at Gabor initialisation (the carrier
        is live or it is not) and only becomes fractional once learning spreads energy onto harmonics
        a low-rate source cannot supply. A harmonic-slot count would rescale intact kernels per rate.
        `resolution`: how much of the kernel's span the recording can actually supply.
        """
        duration_s = torch.as_tensor(duration_s, device=self.spans.device,
                                     dtype=self.spans.dtype).reshape(-1)
        B = duration_s.numel()
        rate = self._rate_vector(rate_hz, B, self.spans.device, self.spans.dtype,
                                 "sampling_rate_hz")
        source_rate = (rate if source_rate_hz is None else
                       self._rate_vector(source_rate_hz, B, self.spans.device,
                                         self.spans.dtype, "source_rate_hz"))
        spans = self.spans
        m = torch.arange(1, self.M + 1, device=spans.device, dtype=spans.dtype)
        harmonic_hz = m.view(1, self.M) / spans.unsqueeze(1)                    # (K,M)
        live = harmonic_hz.view(1, self.K, self.M) <= (
            self.nyquist_margin * source_rate.view(B, 1, 1) / 2.0)
        cos_coeff, sin_coeff = self._normalised_coefficients()
        energy = (cos_coeff.square() + sin_coeff.square()).detach().to(spans.dtype)   # (K,M), rows sum to 1
        nyq = (energy.view(1, self.K, self.M) * live.to(spans.dtype)).sum(dim=2)     # (B,K)
        res = (duration_s.unsqueeze(1) / spans.unsqueeze(0)).clamp(0.0, 1.0)    # (B, K)
        return nyq, res

    # ------------------------------------------------------------------ window reconstruction
    @staticmethod
    def contiguous_window(patches: torch.Tensor, patch_len: torch.Tensor,
                          patch_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B,P,S,C) padded patches -> (B, T, C) contiguous signal + (B,) valid lengths.

        Padding slots and heterogeneous rates make a naive `reshape(B, P*S, C)` wrong twice over: it
        interleaves a 20 Hz item's real samples with the zero padding sized for a 100 Hz item in the
        same batch. Index the VALID samples instead. (Same failure and same fix as
        `model/tokenizer/baseline_backbone.py`.)
        """
        B, P, S, C = patches.shape
        valid = patch_len.clamp_min(0)
        if patch_mask is not None:
            valid = valid * patch_mask.long()
        cumulative = torch.cat([valid.new_zeros((B, 1)), valid.cumsum(dim=1)], dim=1).float()
        total = cumulative[:, -1]
        width = int(total.max().item()) if total.numel() else 0
        width = max(width, 1)
        position = torch.arange(width, device=patches.device, dtype=torch.float32).view(1, -1)
        inside = position < total.unsqueeze(1)
        index = torch.minimum(position, (total - 1).clamp_min(0).unsqueeze(1))
        patch_of = (torch.searchsorted(cumulative, index.contiguous(), right=True) - 1).clamp(0, P - 1)
        offset = (index - cumulative.gather(1, patch_of)).long().clamp_min(0)
        offset = torch.minimum(offset, (patch_len.gather(1, patch_of) - 1).clamp_min(0))
        flat = patches.reshape(B, P * S, C)
        window = flat.gather(1, (patch_of * S + offset).unsqueeze(-1).expand(-1, -1, C))
        return window * inside.unsqueeze(-1).to(window.dtype), total

    @staticmethod
    def _reflect_indices(index: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Reflect arbitrary integer indices at each row's own valid recording boundary."""
        lengths = lengths.long().clamp_min(1).view(-1, 1, 1)
        period = (2 * (lengths - 1)).clamp_min(1)
        folded = torch.remainder(index.view(1, *index.shape), period)
        reflected = torch.where(folded <= lengths - 1, folded, period - folded)
        return torch.where(lengths > 1, reflected, torch.zeros_like(reflected)).long()

    @staticmethod
    def _patch_summaries(patches: torch.Tensor, patch_len: torch.Tensor,
                         patch_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P, S, _ = patches.shape
        sample_valid = torch.arange(S, device=patches.device).view(1, 1, S) \
            < patch_len.clamp_min(0).unsqueeze(-1)
        sample_valid = sample_valid & patch_mask.view(B, P, 1)
        weight = sample_valid.unsqueeze(-1).to(patches.dtype)
        count = weight.sum(dim=2).clamp_min(1.0)
        amplitude = torch.log1p((patches.abs() * weight).sum(dim=2) / count)
        dc = (patches * weight).sum(dim=2) / count
        valid = patch_mask.unsqueeze(-1).to(patches.dtype)
        return amplitude * valid, dc * valid

    def _analyze_rate_group(self, window: torch.Tensor, total: torch.Tensor, P: int,
                            rate: float, source_rate: float
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Analyze a homogeneous stored/native-rate group without sharing recording boundaries."""
        B, _, C = window.shape
        device = window.device
        n_frames = P * self.frames_per_patch
        geometry = self._frame_geometry(rate, P, device)
        sample_index_all = geometry["sample_index"]
        taps = sample_index_all.shape[1]
        signal = window.permute(0, 2, 1)                                         # (B,C,T)
        duration = total / rate
        with torch.autocast(device_type=device.type, enabled=False):
            kernels_all = self._kernels_from_geometry(geometry, rate, source_rate)
        responses, edge_support, frame_valid = [], [], []
        for start in range(0, n_frames, CK_FRAME_CHUNK):
            stop = min(start + CK_FRAME_CHUNK, n_frames)
            t_f = geometry["frame_time"][start:stop]
            sample_index = sample_index_all[start:stop]
            reflected = self._reflect_indices(sample_index, total)               # (B,f,t)
            gather = reflected.reshape(B, 1, -1).expand(B, C, -1)
            segment = signal.gather(2, gather).reshape(B, C, stop - start, taps)
            with torch.autocast(device_type=device.type, enabled=False):
                kernels = kernels_all[start:stop]
                z = torch.einsum("bcft,fkqt->bckfq", segment.float(), kernels)
            responses.append(z.square().sum(-1).clamp_min(1e-12).sqrt())

            kernel_support = geometry["support"][start:stop]
            real_sample = ((sample_index.view(1, stop - start, taps) >= 0) &
                           (sample_index.view(1, stop - start, taps) <
                            total.view(B, 1, 1)))
            supported = (kernel_support.unsqueeze(0) & real_sample.unsqueeze(2)).sum(-1)
            denominator = kernel_support.sum(-1).clamp_min(1).unsqueeze(0)
            edge_support.append((supported / denominator).permute(0, 2, 1).float())
            frame_valid.append((t_f.view(1, -1) < duration.view(B, 1)))
        return (torch.cat(responses, dim=3), torch.cat(edge_support, dim=2),
                torch.cat(frame_valid, dim=1))

    def _check_patch_grid(self, patch_len: torch.Tensor, patch_mask: torch.Tensor,
                          rates: torch.Tensor) -> None:
        """Every full patch must be `patch_seconds` long; only the last valid patch may be shorter.

        Frames are laid out on the physical patch grid this module was built for. Feeding it a
        different grid would attribute the wrong seconds to every token, silently.
        """
        B, P = patch_len.shape
        # The collate partitions on ROUNDED physical boundaries, so a full patch holds
        # rate * patch_seconds samples give or take one (25 Hz x 1.5 s alternates 37/38).
        expected = rates.to(torch.float32) * self.patch_seconds
        n_valid = patch_mask.sum(dim=1)
        is_last = torch.arange(P, device=patch_len.device).view(1, P) == (n_valid - 1).view(B, 1)
        must_be_full = patch_mask & ~is_last
        length = patch_len.to(torch.float32)
        fits = length <= expected.view(B, 1) + 1.0 + 1e-6
        full = (length - expected.view(B, 1)).abs() <= 1.0 + 1e-6
        ok = ((fits | ~patch_mask) & (full | ~must_be_full)).all()
        message = (f"continuous frontend built for {self.patch_seconds:g} s patches received a "
                   f"different patch grid (full patches must hold rate * {self.patch_seconds:g} "
                   f"samples within one sample; only the final valid patch may be shorter)")
        if patch_len.device.type == "cpu":
            if not bool(ok):
                raise ValueError(message)
        else:
            torch._assert_async(ok, message)

    # ------------------------------------------------------------------ the analysis stage
    def analyze(self, patches, sampling_rate_hz, patch_len_samples=None,
                source_rate_hz=None, patch_mask=None) -> dict:
        """Per-frame band magnitudes and the masks. Returns a dict; `project` turns it into tokens."""
        B, P, S, C = patches.shape
        device = patches.device
        rates = self._rate_vector(sampling_rate_hz, B, device, patches.dtype,
                                  "sampling_rate_hz")
        source_rates = (rates if source_rate_hz is None else
                        self._rate_vector(source_rate_hz, B, device, patches.dtype,
                                          "source_rate_hz"))
        if patch_len_samples is None:
            patch_len_samples = torch.full((B, P), S, dtype=torch.long, device=device)
        else:
            patch_len_samples = torch.as_tensor(
                patch_len_samples, device=device, dtype=torch.long)
            if patch_len_samples.numel() == 1:
                patch_len_samples = patch_len_samples.reshape(1, 1).expand(B, P)
            elif patch_len_samples.numel() == B:
                patch_len_samples = patch_len_samples.reshape(B, 1).expand(B, P)
            elif patch_len_samples.numel() == B * P:
                patch_len_samples = patch_len_samples.reshape(B, P)
            else:
                raise ValueError(f"patch_len_samples must be scalar, (B,), or (B,P)=({B},{P})")
        lengths_fit = (patch_len_samples <= S).all()
        if device.type == "cpu":
            if not bool(lengths_fit):
                raise ValueError(f"patch_len_samples cannot exceed padded sample width S={S}")
        else:
            torch._assert_async(lengths_fit, f"patch_len_samples exceeds sample width S={S}")
        length_valid = patch_len_samples > 0
        patch_mask = (length_valid if patch_mask is None else
                      torch.as_tensor(patch_mask, device=device, dtype=torch.bool).reshape(B, P)
                      & length_valid)
        self._check_patch_grid(patch_len_samples, patch_mask, rates)
        window, total = self.contiguous_window(patches, patch_len_samples, patch_mask)
        duration = total / rates
        n_frames = P * self.frames_per_patch
        magnitude = patches.new_zeros(B, C, self.K, n_frames, dtype=torch.float32)
        edge_support = patches.new_zeros(B, self.K, n_frames, dtype=torch.float32)
        frame_valid = torch.zeros(B, n_frames, device=device, dtype=torch.bool)
        pairs = torch.stack((rates, source_rates), dim=1)
        unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
        # One synchronization obtains the small list of distinct physical configurations. The
        # previous implementation synchronized twice per group via scalar .item() calls.
        pair_values = unique_pairs.detach().cpu().tolist()
        for group_id, (rate, source_rate) in enumerate(pair_values):
            rows = torch.nonzero(inverse == group_id, as_tuple=False).flatten()
            group_magnitude, group_edge, group_valid = self._analyze_rate_group(
                window.index_select(0, rows), total.index_select(0, rows), P,
                float(rate), float(source_rate))
            # These destinations are freshly allocated assembly buffers. Updating them in place
            # avoids copying the complete batch once per physical-rate group; IndexCopyBackward
            # still carries gradients from every group into the learned kernels and signal path.
            magnitude.index_copy_(0, rows, group_magnitude)
            edge_support.index_copy_(0, rows, group_edge)
            frame_valid.index_copy_(0, rows, group_valid)
        compressed = torch.log1p(magnitude)
        nyq, res = self.masks(rates, duration, source_rates)
        amplitude, dc = self._patch_summaries(patches, patch_len_samples, patch_mask)
        if self._telemetry_requested:
            valid_frames = frame_valid.view(B, 1, 1, n_frames).to(compressed.dtype)
            valid_count = (valid_frames.sum() * C).clamp_min(1.0)
            mean = (compressed * valid_frames).sum(dim=(0, 1, 3)) / valid_count
            variance = ((compressed - mean.view(1, 1, self.K, 1)).square()
                        * valid_frames).sum(dim=(0, 1, 3)) / valid_count
            edge_weight = frame_valid.view(B, 1, n_frames).to(edge_support.dtype)
            self._runtime_summary = {
                "frontend/observable_fraction": nyq.mean().detach(),
                "frontend/edge_support_mean": (
                    (edge_support * edge_weight).sum()
                    / (edge_weight.sum() * self.K).clamp_min(1.0)
                ).detach(),
                "frontend/response_std_mean": variance.sqrt().mean().detach(),
                "frontend/dead_kernel_fraction": (variance < 1e-10).float().mean().detach(),
            }
            self._telemetry_requested = False
        return {"compressed": compressed, "nyquist": nyq, "resolution": res,
                "edge": edge_support, "window": window, "duration": duration,
                "frame_valid": frame_valid, "patch_valid": patch_mask,
                "amplitude": amplitude, "dc": dc,
                "n_patches": P, "rate": rates, "source_rate": source_rates}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        revision = state_dict.get(prefix + "_frontend_revision")
        if revision is None or not torch.equal(revision.cpu(), self._frontend_revision.cpu()):
            error_msgs.append(
                f"{prefix}continuous frontend checkpoint predates or differs from the current "
                "analysis/normalization revision. Reproduce historical results with their saved "
                "source revision; train a new checkpoint for the corrected frontend."
            )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    # ------------------------------------------------------------------ normalisation stats
    def reset_norm_accumulator(self):
        self._acc_count.zero_(); self._acc_sum.zero_(); self._acc_sqsum.zero_()
        for name in ("amp", "dc"):
            getattr(self, f"_{name}_acc_count").zero_()
            getattr(self, f"_{name}_acc_sum").zero_()
            getattr(self, f"_{name}_acc_sqsum").zero_()

    @torch.no_grad()
    def accumulate_norm_stats(self, patches, sampling_rate_hz, patch_len_samples=None,
                              patch_mask=None, channel_mask=None, source_rate_hz=None):
        out = self.analyze(patches, sampling_rate_hz, patch_len_samples,
                           source_rate_hz=source_rate_hz, patch_mask=patch_mask)
        value = out["compressed"]                                                # (B,C,K,f)
        B, C, _, n_frames = value.shape
        weight = out["frame_valid"].view(B, 1, 1, n_frames).to(torch.float64)
        weight = weight * out["nyquist"].view(B, 1, self.K, 1).to(torch.float64)
        if channel_mask is not None:
            channel_mask = torch.as_tensor(channel_mask, device=value.device, dtype=torch.bool)
            weight = weight * channel_mask.view(B, C, 1, 1).to(torch.float64)
        else:
            weight = weight.expand(B, C, self.K, n_frames)
        value64 = value.to(torch.float64)
        self._acc_count += weight.sum(dim=(0, 1, 3))
        self._acc_sum += (value64 * weight).sum(dim=(0, 1, 3))
        self._acc_sqsum += (value64.square() * weight).sum(dim=(0, 1, 3))

        patch_weight = out["patch_valid"].view(B, -1, 1).to(torch.float64)
        if channel_mask is not None:
            patch_weight = patch_weight * channel_mask.view(B, 1, C).to(torch.float64)
        else:
            patch_weight = patch_weight.expand(B, out["patch_valid"].shape[1], C)
        for name, sample in (("amp", out["amplitude"]), ("dc", out["dc"])):
            sample = sample.to(torch.float64)
            getattr(self, f"_{name}_acc_count").add_(patch_weight.sum())
            getattr(self, f"_{name}_acc_sum").add_((sample * patch_weight).sum())
            getattr(self, f"_{name}_acc_sqsum").add_((sample.square() * patch_weight).sum())

    @torch.no_grad()
    def finalize_norm_stats(self, eps: float = 1e-5):
        seen = self._acc_count > 0
        count = self._acc_count.clamp_min(1.0)
        mu = self._acc_sum / count
        var = (self._acc_sqsum / count - mu.pow(2)).clamp_min(0.0)
        self.norm_mu.copy_(torch.where(seen, mu, torch.zeros_like(mu)).float())
        self.norm_sd.copy_(torch.where(
            seen, var.sqrt().clamp_min(eps), torch.ones_like(var)).float())
        for name in ("amp", "dc"):
            observed = getattr(self, f"_{name}_acc_count")
            count = observed.clamp_min(1.0)
            mu = getattr(self, f"_{name}_acc_sum") / count
            var = getattr(self, f"_{name}_acc_sqsum") / count - mu.square()
            getattr(self, f"{name}_mu").copy_(
                torch.where(observed > 0, mu, torch.zeros_like(mu)).float())
            getattr(self, f"{name}_sd").copy_(torch.where(
                observed > 0, var.clamp_min(0.0).sqrt().clamp_min(eps),
                torch.ones_like(var)).float())
        self._norm_fitted.fill_(1.0)

    @torch.no_grad()
    def fit_norm_stats(self, patches, sampling_rate_hz, patch_len_samples=None, eps: float = 1e-5,
                       patch_mask=None, channel_mask=None, source_rate_hz=None):
        self.reset_norm_accumulator()
        self.accumulate_norm_stats(
            patches, sampling_rate_hz, patch_len_samples, patch_mask=patch_mask,
            channel_mask=channel_mask, source_rate_hz=source_rate_hz)
        self.finalize_norm_stats(eps)

    # ------------------------------------------------------------------ projection to tokens
    def _sensor_layout(
        self,
        batch: int,
        channels: int,
        device: torch.device,
        sensor_id: torch.Tensor | None,
        channel_mask: torch.Tensor | None,
        n_sensors: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return the fixed sensor/axis slot for each channel and the live sensor mask."""
        if sensor_id is None:
            sensor_id = (torch.arange(channels, device=device) // self.axes_per_sensor).view(1, -1)
            sensor_id = sensor_id.expand(batch, -1)
        else:
            sensor_id = sensor_id.to(device=device, dtype=torch.long)
        if sensor_id.shape != (batch, channels):
            raise ValueError(
                f"sensor_id must have shape {(batch, channels)}, got {tuple(sensor_id.shape)}"
            )
        if channel_mask is None:
            channel_mask = torch.ones(batch, channels, dtype=torch.bool, device=device)
        else:
            channel_mask = channel_mask.to(device=device, dtype=torch.bool)
        if channel_mask.shape != (batch, channels):
            raise ValueError(
                f"channel_mask must have shape {(batch, channels)}, got {tuple(channel_mask.shape)}"
            )
        inferred = (channels + self.axes_per_sensor - 1) // self.axes_per_sensor
        sensors = int(n_sensors if n_sensors is not None else inferred)
        if sensors <= 0:
            raise ValueError("n_sensors must be positive")
        ids_valid = (~channel_mask | ((sensor_id >= 0) & (sensor_id < sensors))).all()
        if device.type == "cpu":
            if not bool(ids_valid):
                raise ValueError("a live channel maps outside the available sensor descriptions")
        else:
            torch._assert_async(
                ids_valid, "a live channel maps outside the available sensor descriptions",
            )

        # Channel order is the global [acc xyz, gyro xyz] contract.  Using channel position rather
        # than a live-channel ordinal preserves y/z when x is dropped instead of shifting them left.
        axis = torch.arange(channels, device=device).remainder(self.axes_per_sensor)
        slot = sensor_id * self.axes_per_sensor + axis.view(1, channels)
        live_slot = torch.where(channel_mask, slot, torch.full_like(slot, -1))
        if device.type == "cpu":
            for row in range(batch):
                values = live_slot[row][live_slot[row] >= 0]
                if values.unique().numel() != values.numel():
                    raise ValueError("two live channels map to the same sensor-axis slot")
        # Reduce with OR, never overwrite: accel-only streams intentionally give their six padded
        # channel slots sensor id 0 and rely on channel_mask to mark gyro absent.  A plain scatter
        # would let a later False gyro slot overwrite the earlier True accelerometer slot.
        membership = F.one_hot(
            sensor_id.clamp(0, sensors - 1), num_classes=sensors,
        ).to(torch.bool)
        sensor_present = (membership & channel_mask.unsqueeze(-1)).any(dim=1)
        return slot, channel_mask, sensor_present, sensors

    @staticmethod
    def _pack_axis_values(
        values: torch.Tensor, slot: torch.Tensor, live: torch.Tensor, slots: int,
    ) -> torch.Tensor:
        """Pack (B,...,C) values into fixed sensor-axis slots without breaking gradients."""
        batch, channels = slot.shape
        middle = values.shape[1:-1]
        source = values * live.view(batch, *((1,) * len(middle)), channels).to(values.dtype)
        out = values.new_zeros(batch, *middle, slots + 1)
        trash = torch.full_like(slot, slots)
        target = torch.where(live, slot, trash)
        index = target.view(batch, *((1,) * len(middle)), channels).expand_as(source)
        out.scatter_(dim=-1, index=index, src=source)
        return out[..., :slots]

    def sensor_presence(
        self, sensor_id: torch.Tensor, channel_mask: torch.Tensor, n_sensors: int,
    ) -> torch.Tensor:
        """Public presence helper used by the encoder after direct sensor-token projection."""
        batch, channels = sensor_id.shape
        _, _, present, _ = self._sensor_layout(
            batch, channels, sensor_id.device, sensor_id, channel_mask, n_sensors,
        )
        return present

    def project(
        self,
        analysis: dict,
        *,
        sensor_id: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        n_sensors: int | None = None,
    ) -> torch.Tensor:
        compressed = analysis["compressed"]                                      # (B,C,K,f)
        B, C, K, n_frames = compressed.shape
        P = analysis["n_patches"]
        slot, live, _, sensors = self._sensor_layout(
            B, C, compressed.device, sensor_id, channel_mask, n_sensors,
        )
        total_slots = sensors * self.axes_per_sensor
        if self.norm == "frozen":
            compressed = (compressed - self.norm_mu.view(1, 1, K, 1)) / self.norm_sd.view(1, 1, K, 1)
        compressed = compressed * analysis["nyquist"].view(B, 1, K, 1)
        # Invalid frames still contain reflected analysis values. Zero them before the temporal CNN
        # so padding, or signal beyond a future-JEPA context boundary, cannot enter a valid token.
        frame_valid = analysis["frame_valid"].view(B, 1, 1, n_frames)
        compressed = compressed * frame_valid.to(compressed.dtype)
        # (B,C,K,T) -> (B,S,xyz,K,T).  The extra trash slot receives absent channels and is sliced.
        source = compressed * live.view(B, C, 1, 1).to(compressed.dtype)
        packed = compressed.new_zeros(B, total_slots + 1, K, n_frames)
        target = torch.where(live, slot, torch.full_like(slot, total_slots))
        packed.scatter_(1, target.view(B, C, 1, 1).expand(B, C, K, n_frames), source)
        x = packed[:, :total_slots].reshape(B * sensors, self.axes_per_sensor * K, n_frames)
        x = self.conv1(x)
        # Conv1 output j is centred on input frame 2*j. Using max-pool validity would mark the first
        # future output valid merely because its left neighbour is observed; Conv2 could then carry
        # that boundary activation back into the final context token.
        conv_valid = analysis["frame_valid"][:, ::2].unsqueeze(1)
        conv_valid = conv_valid.expand(B, sensors, -1).reshape(B * sensors, 1, -1)
        x = F.gelu(self.ln1(x.transpose(1, 2)).transpose(1, 2))
        x = x * conv_valid.to(x.dtype)
        x = self.conv2(x)
        x = F.gelu(self.ln2(x.transpose(1, 2)).transpose(1, 2))                  # (B*S, c2, f/2)
        x = x * conv_valid.to(x.dtype)
        c2, reduced = x.shape[1], x.shape[2]
        per_token = self.frames_per_token
        if reduced != per_token * P:
            raise RuntimeError(
                f"expected {per_token} pooled frames per token ({P} tokens), got {reduced} frames"
            )
        x = x.reshape(B, sensors, c2, P, per_token)
        # ORDERED flatten: a rising and a falling ramp must be different inputs. An order-invariant
        # pool (mean/max/std) cannot distinguish them, which would re-destroy the sub-second
        # structure this front end exists to keep.
        ordered = x.permute(0, 3, 1, 2, 4).reshape(B, P, sensors, c2 * per_token)
        metadata_dim = 2 * K + K * self.frames_per_token + 3 * self.axes_per_sensor
        if ordered.shape[-1] < self.in_dim - metadata_dim:
            pad = self.in_dim - metadata_dim - ordered.shape[-1]
            ordered = F.pad(ordered, (0, pad))
        nyq = analysis["nyquist"].view(B, 1, 1, K).expand(B, P, sensors, K)
        res = analysis["resolution"].view(B, 1, 1, K).expand(B, P, sensors, K)
        edge = F.avg_pool1d(analysis["edge"], kernel_size=2, stride=2)            # (B,K,f/2)
        edge = edge[..., :per_token * P].reshape(B, K, P, per_token)
        edge = edge.permute(0, 2, 1, 3).reshape(B, P, 1, K * per_token)
        edge = edge.expand(B, P, sensors, K * per_token)
        if edge.shape[-1] < K * self.frames_per_token:
            edge = F.pad(edge, (0, K * self.frames_per_token - edge.shape[-1]))
        edge = edge[..., :K * self.frames_per_token]
        amplitude = analysis["amplitude"]
        dc = analysis["dc"]
        if self.norm == "frozen":
            amplitude = (amplitude - self.amp_mu) / self.amp_sd
            dc = (dc - self.dc_mu) / self.dc_sd
        amplitude = self._pack_axis_values(amplitude, slot, live, total_slots).reshape(
            B, P, sensors, self.axes_per_sensor)
        dc = self._pack_axis_values(dc, slot, live, total_slots).reshape(
            B, P, sensors, self.axes_per_sensor)
        validity = live.new_zeros(B, total_slots + 1)
        validity.scatter_(1, target, live)
        validity = validity[:, :total_slots].reshape(B, 1, sensors, self.axes_per_sensor)
        validity = validity.expand(B, P, sensors, self.axes_per_sensor).to(ordered.dtype)
        features = torch.cat([ordered[..., :self.in_dim - metadata_dim],
                              nyq, res, edge, amplitude, dc, validity], dim=-1)
        return self.proj(features)

    def forward(self, patches, sampling_rate_hz, patch_len_samples=None,
                source_rate_hz=None, patch_mask=None, sensor_id=None,
                channel_mask=None, n_sensors=None) -> torch.Tensor:
        return self.project(self.analyze(patches, sampling_rate_hz, patch_len_samples,
                                         source_rate_hz=source_rate_hz, patch_mask=patch_mask),
                            sensor_id=sensor_id, channel_mask=channel_mask,
                            n_sensors=n_sensors)
