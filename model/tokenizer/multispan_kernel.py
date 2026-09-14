"""Multi-span continuous-kernel tokenizer: one token per (span group, frame, sensor).

Design of record: `docs/design/CONTINUOUS_KERNEL_FRONTEND.md`, section "Multi-span tokenization".
Selected with `--frontend multispan`.

WHAT IT IS
----------
The single-span `ContinuousKernelTokenizer` runs one constant-Q bank at 8 frames per second and
packs four ordered frames into a one-second token through a dense CNN. This module keeps the
continuous kernels and the rate contract but changes what a token IS:

* the bank is organised by physical span (default 0.5, 1, and 2 seconds), with log-spaced physical
  centres projected onto each span's harmonic basis;
* every span is analyzed on the same dense 16 Hz physical-time grid;
* a sensor-shared causal convolutional stem composes local response patterns and reduces each span
  to four output tokens per second. Tokens carry physical centre time, span, and resolution id.

The kernels themselves are exactly the single-span module's: a Gaussian-windowed Fourier series in
normalised time, evaluated at the exact offsets of the real samples, scaled by `dt`, re-zero-meaned,
with the observability mask equal to the coefficient energy that survives the Nyquist cut.

The collate's patch grid is irrelevant here: the contiguous window is rebuilt from the patches and
the frames are laid out in physical time from zero. Different collate partitions give the same
tokens.

CONTRACT
--------
`token_grid(patches, sampling_rate_hz, patch_len_samples, source_rate_hz=None, patch_mask=None,
sensor_id=None, channel_mask=None, n_sensors=None)` returns a dict::

    tokens          (B, N, S, d_model)   N = sum over groups of output-token counts
    positions       (B, N)  physical centre time in seconds
    durations       (B, N)  the token's span in seconds
    resolution_ids  (B, N)  group index
    token_mask      (B, N)  True where the frame centre lies inside the recording

`forward` is `token_grid`. `analyze`/`project` of the parent are not defined for this layout.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .continuous_kernel import (
    CK_ENVELOPE_SIGMA,
    CK_F_MAX_HZ,
    CK_N_HARMONICS,
    CK_NYQUIST_MARGIN,
    CK_SIGMA_MIN,
    CK_SIGMA_MAX,
    CK_GAIN_MAX,
    ContinuousKernelTokenizer,
)

MS_SPANS_S = (0.5, 1.0, 2.0)
MS_FRAME_RATE_HZ = 16
MS_TOKEN_RATE_HZ = 4
MS_FRAMES_PER_SPAN = 4  # Legacy checkpoints only: their hop was span / frames_per_span.


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over channels independently at every physical-time frame."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value.transpose(1, 2)).transpose(1, 2)


class _CausalDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, dilation=dilation, bias=False,
        )
        self.norm = _ChannelLayerNorm(channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(F.pad(value, (self.left_padding, 0)))
        update = self.pointwise(F.gelu(self.norm(update)))
        return (value + update) * mask.unsqueeze(1).to(value.dtype)


class _CausalDownsample(nn.Module):
    """Learned causal stride-2 reduction with explicit validity propagation."""

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, stride=2, groups=channels, bias=False,
        )
        self.norm = _ChannelLayerNorm(channels)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.conv(F.pad(value, (self.left_padding, 0)))
        mask = mask[:, ::2][:, :value.shape[-1]]
        return F.gelu(self.norm(value)) * mask.unsqueeze(1).to(value.dtype), mask


class MultiSpanConvStem(nn.Module):
    """Per-group entry projection followed by a sensor-shared causal temporal stem."""

    def __init__(
        self,
        input_dims: Sequence[int],
        d_model: int,
        channels: int = 128,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4),
        shared: bool = True,
    ):
        super().__init__()
        self.shared = bool(shared)
        self.entry = nn.ModuleList(nn.Conv1d(dim, channels, 1) for dim in input_dims)

        def body():
            return nn.ModuleList(
                _CausalDepthwiseBlock(channels, kernel_size, int(d)) for d in dilations
            )

        def downsamples():
            return nn.ModuleList((_CausalDownsample(channels), _CausalDownsample(channels)))

        self.body = body() if self.shared else nn.ModuleList(body() for _ in input_dims)
        self.downsample = downsamples() if self.shared else nn.ModuleList(
            downsamples() for _ in input_dims
        )
        self.out = nn.Linear(channels, d_model) if self.shared else nn.ModuleList(
            nn.Linear(channels, d_model) for _ in input_dims
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor, group: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # features (B,F,S,D) -> one independent sequence per sensor; sensors never mix here.
        batch, frames, sensors, width = features.shape
        value = features.permute(0, 2, 3, 1).reshape(batch * sensors, width, frames)
        valid = mask[:, None, :].expand(batch, sensors, frames).reshape(batch * sensors, frames)
        value = F.gelu(self.entry[group](value)) * valid.unsqueeze(1).to(value.dtype)
        body = self.body if self.shared else self.body[group]
        downsample = self.downsample if self.shared else self.downsample[group]
        for block in body:
            value = block(value, valid)
        for layer in downsample:
            value, valid = layer(value, valid)
        value = value.transpose(1, 2)
        projection = self.out if self.shared else self.out[group]
        value = projection(value).reshape(batch, sensors, -1, projection.out_features)
        value = value.permute(0, 2, 1, 3)
        valid = valid.reshape(batch, sensors, -1).all(dim=1)
        return value, valid


def multispan_frame_count(
    spans: Sequence[float], duration_s: float, frames_per_span: int | None = None,
    *, token_rate_hz: float = MS_TOKEN_RATE_HZ,
) -> int:
    """Number of physical-time tokens emitted for one sensor and recording."""

    spans = tuple(float(span) for span in spans)
    if (not spans or not math.isfinite(duration_s) or duration_s <= 0
            or (frames_per_span is not None and frames_per_span <= 0)
            or token_rate_hz <= 0
            or any(not math.isfinite(span) or span <= 0 for span in spans)):
        raise ValueError("spans, duration, and frames_per_span must be finite and positive")
    if frames_per_span is None:
        return len(spans) * max(1, int(math.ceil(duration_s * token_rate_hz - 1e-6)))
    return sum(max(1, int(math.ceil(duration_s * frames_per_span / float(span) - 1e-6)))
               for span in spans)


class MultiSpanKernelTokenizer(ContinuousKernelTokenizer):
    """Span-grouped continuous kernels emitting a physical-time token grid."""

    def __init__(
        self,
        d_model: int = 128,
        spans: Sequence[float] = MS_SPANS_S,
        frame_rate_hz: int | None = MS_FRAME_RATE_HZ,
        frames_per_span: int | None = None,
        n_harmonics: int = CK_N_HARMONICS,
        f_max: float = CK_F_MAX_HZ,
        nyquist_margin: float = CK_NYQUIST_MARGIN,
        envelope_sigma: float = CK_ENVELOPE_SIGMA,
        gabor_init: bool = True,
        norm: str = "frozen",
        centre_spacing: str = "log",
        centres_per_span: Sequence[int] | None = None,
        compression_scale: str = "calibrated",
        stem: str = "conv",
        stem_channels: int = 128,
        stem_kernel: int = 5,
        stem_dilations: Sequence[int] = (1, 2, 4),
        stem_shared: bool = True,
    ):
        nn.Module.__init__(self)
        self._validate_analysis_config(n_harmonics, nyquist_margin, norm)
        if not math.isfinite(f_max) or f_max <= 0:
            raise ValueError("f_max must be finite and positive")
        spans = tuple(sorted(float(s) for s in spans))
        if not spans or not all(math.isfinite(s) and s > 0 for s in spans) \
                or len(set(spans)) != len(spans):
            raise ValueError("spans must be distinct finite positive durations in seconds")
        if frame_rate_hz is None and frames_per_span is None:
            raise ValueError("frame_rate_hz or legacy frames_per_span is required")
        if frame_rate_hz is not None and (
            not math.isfinite(frame_rate_hz) or frame_rate_hz < 1 or frame_rate_hz != int(frame_rate_hz)
        ):
            raise ValueError("frame_rate_hz must be a positive integer")
        if frames_per_span is not None and (
            not math.isfinite(frames_per_span) or frames_per_span < 1
            or frames_per_span != int(frames_per_span)
        ):
            raise ValueError("frames_per_span must be a positive integer")
        if centre_spacing not in {"harmonic", "log"}:
            raise ValueError("centre_spacing must be 'harmonic' or 'log'")
        if compression_scale not in {"none", "calibrated"}:
            raise ValueError("compression_scale must be 'none' or 'calibrated'")
        if stem not in {"none", "conv"}:
            raise ValueError("stem must be 'none' or 'conv'")
        if (not isinstance(stem_channels, int) or isinstance(stem_channels, bool)
                or stem_channels < 1):
            raise ValueError("stem_channels must be a positive integer")
        if (not isinstance(stem_kernel, int) or isinstance(stem_kernel, bool)
                or stem_kernel < 1):
            raise ValueError("stem_kernel must be a positive integer")
        if (not stem_dilations or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in stem_dilations)):
            raise ValueError("stem_dilations must contain positive integers")
        legacy_layout = (
            frame_rate_hz is None and frames_per_span is not None
            and centre_spacing == "harmonic" and compression_scale == "none" and stem == "none"
        )
        self.register_buffer(
            "_frontend_revision", torch.tensor(2 if legacy_layout else 3, dtype=torch.long),
        )
        if n_harmonics < 1:
            raise ValueError("need at least 1 harmonic")
        self.M = int(n_harmonics)
        self.d_model = int(d_model)
        self.frame_rate_hz = int(frame_rate_hz) if frame_rate_hz is not None else None
        self.frames_per_span = int(frames_per_span) if frames_per_span is not None else None
        self.centre_spacing = centre_spacing
        self.compression_scale_mode = compression_scale
        self.stem_type = stem
        self.stem_channels = int(stem_channels)
        self.stem_kernel = int(stem_kernel)
        self.stem_dilations = tuple(int(value) for value in stem_dilations)
        self.stem_shared = bool(stem_shared)
        self.nyquist_margin = float(nyquist_margin)
        self.envelope_sigma = float(envelope_sigma)
        self.norm = norm
        self.learnable = True
        self.emits_sensor_tokens = True
        self.emits_token_grid = True
        self.axes_per_sensor = 3
        self._geometry_cache: dict[tuple, dict[str, torch.Tensor]] = {}
        self._telemetry_requested = False
        self._runtime_summary: dict[str, torch.Tensor] = {}
        self.sigma_min = CK_SIGMA_MIN
        self.sigma_max = CK_SIGMA_MAX
        self.gain_max = CK_GAIN_MAX

        centre_counts = tuple(
            min(self.M, max(1, int(math.floor(f_max * span + 1e-6)))) for span in spans
        ) if centres_per_span is None else tuple(int(value) for value in centres_per_span)
        if len(centre_counts) != len(spans) or any(value < 1 for value in centre_counts):
            raise ValueError("centres_per_span must provide one positive count per span")

        # The legacy path retains exact integer harmonics. New runs distribute projected Gabor
        # centres logarithmically within every span's representable physical-frequency range.
        span_of, carrier_of, group_of = [], [], []
        centre_of: list[float] = []
        for g, T in enumerate(spans):
            n_harm = min(self.M, int(math.floor(f_max * T + 1e-6)))
            if n_harm < 1:
                raise ValueError(f"span {T} s holds no harmonic at or below f_max={f_max} Hz")
            count = n_harm if centre_spacing == "harmonic" else centre_counts[g]
            high = min(float(f_max), self.M / T)
            if count == 1:
                frequencies = [1.0 / T]
            elif centre_spacing == "harmonic":
                frequencies = [m / T for m in range(1, count + 1)]
            else:
                frequencies = torch.logspace(
                    math.log10(1.0 / T), math.log10(high), count,
                ).tolist()
            for frequency in frequencies:
                span_of.append(T)
                carrier_of.append(max(1, min(self.M, int(round(frequency * T)))))
                centre_of.append(float(frequency))
                group_of.append(g)
        self.K = len(span_of)
        self.G = len(spans)
        self.group_sizes = [group_of.count(g) for g in range(self.G)]
        self.group_offsets = [sum(self.group_sizes[:g]) for g in range(self.G)]
        self.span_list = list(spans)
        self.max_span = spans[-1]
        self.register_buffer("spans", torch.tensor(span_of, dtype=torch.float32))      # (K,)
        self.register_buffer("carrier", torch.tensor(carrier_of, dtype=torch.long))    # (K,)
        self.register_buffer("group", torch.tensor(group_of, dtype=torch.long))        # (K,)
        self.register_buffer("centres", torch.tensor(centre_of, dtype=torch.float32))  # (K,) Hz
        self.register_buffer("group_spans", torch.tensor(spans, dtype=torch.float32))  # (G,)

        # --- learnable analysis parameters, exactly as the single-span module ---
        self.cos_coeff = nn.Parameter(torch.zeros(self.K, self.M))
        self.sin_coeff = nn.Parameter(torch.zeros(self.K, self.M))
        sigma_fraction = (envelope_sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)
        if not 0.0 < sigma_fraction < 1.0:
            raise ValueError(
                f"envelope_sigma must be in ({self.sigma_min}, {self.sigma_max}), got {envelope_sigma}"
            )
        self.sigma_logit = nn.Parameter(
            torch.full((self.K,), math.log(sigma_fraction / (1.0 - sigma_fraction)))
        )
        self.gain_logit = nn.Parameter(torch.zeros(self.K), requires_grad=False)
        if gabor_init:
            with torch.no_grad():
                if centre_spacing == "harmonic":
                    self.cos_coeff[torch.arange(self.K), self.carrier - 1] = 1.0
                else:
                    projected_cos, projected_sin = self._projected_gabor_coefficients()
                    self.cos_coeff.copy_(projected_cos)
                    self.sin_coeff.copy_(projected_sin)
        else:
            nn.init.normal_(self.cos_coeff, std=0.3)
            nn.init.normal_(self.sin_coeff, std=0.3)
        self.register_buffer("initial_cos_coeff", self._normalised_coefficients()[0].detach().clone())
        self.register_buffer("initial_sin_coeff", self._normalised_coefficients()[1].detach().clone())
        self.register_buffer("initial_sigma", torch.full((self.K,), float(envelope_sigma)))

        # --- frozen standardisation of the compressed magnitude, per kernel ---
        self.register_buffer("norm_mu", torch.zeros(self.K))
        self.register_buffer("norm_sd", torch.ones(self.K))
        self.register_buffer("compression_knee", torch.ones(self.K))
        self.register_buffer("_norm_fitted", torch.zeros(1))
        self.register_buffer("_acc_count", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sum", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sqsum", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        self.register_buffer("_scale_log_sum", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        self.register_buffer("_scale_count", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        self._scale_samples: list[torch.Tensor] = []
        for name in ("amp", "dc"):
            self.register_buffer(f"{name}_mu", torch.zeros(self.G, 2))
            self.register_buffer(f"{name}_sd", torch.ones(self.G, 2))
            self.register_buffer(f"_{name}_acc_count", torch.zeros(self.G, 2, dtype=torch.float64),
                                 persistent=False)
            self.register_buffer(f"_{name}_acc_sum", torch.zeros(self.G, 2, dtype=torch.float64),
                                 persistent=False)
            self.register_buffer(f"_{name}_acc_sqsum", torch.zeros(self.G, 2, dtype=torch.float64),
                                 persistent=False)

        input_dims = [self.group_in_dim(g) for g in range(self.G)]
        self.proj = (nn.ModuleList(nn.Linear(width, self.d_model) for width in input_dims)
                     if self.stem_type == "none" else nn.ModuleList())
        self.stem = (
            MultiSpanConvStem(
                input_dims, self.d_model, channels=self.stem_channels,
                kernel_size=self.stem_kernel, dilations=self.stem_dilations,
                shared=self.stem_shared,
            )
            if self.stem_type == "conv" else None
        )

    # ------------------------------------------------------------------ layout helpers
    def group_in_dim(self, g: int) -> int:
        k = self.group_sizes[g]
        return (self.axes_per_sensor * k     # standardised magnitudes on the three axes
                + k                          # observability per kernel
                + 1                          # edge support of the span at this frame
                + self.axes_per_sensor       # local log amplitude per axis
                + self.axes_per_sensor       # local signed DC per axis
                + self.axes_per_sensor)      # axis validity bits

    def group_slice(self, g: int) -> slice:
        return slice(self.group_offsets[g], self.group_offsets[g] + self.group_sizes[g])

    def _projected_gabor_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Least-squares projection of non-harmonic analytic carriers onto the Fourier basis."""
        samples = max(8 * self.M, 256)
        u = torch.linspace(-0.5, 0.5, samples, dtype=torch.float64)
        harmonic = torch.arange(1, self.M + 1, dtype=torch.float64)
        angle = 2.0 * math.pi * u[:, None] * harmonic[None, :]
        cos_basis, sin_basis = angle.cos(), angle.sin()
        envelope = torch.exp(-u.square() / (2.0 * float(self.envelope_sigma) ** 2))

        def shaped(value: torch.Tensor) -> torch.Tensor:
            value = value * envelope[:, None] if value.ndim == 2 else value * envelope
            return value - value.mean(dim=0, keepdim=value.ndim == 2)

        cos_out = torch.zeros(self.K, self.M, dtype=torch.float64)
        sin_out = torch.zeros_like(cos_out)
        for index, (centre, span) in enumerate(zip(self.centres.double(), self.spans.double())):
            target_angle = 2.0 * math.pi * centre * span * u
            # Match the actual zero-mean, Gaussian-windowed quadrature kernel, not an unwindowed
            # sinusoid. The envelope is what lets a compact harmonic basis represent centres that
            # sit between integer harmonics without sacrificing response energy.
            system = torch.cat((
                torch.cat((shaped(cos_basis), shaped(sin_basis)), dim=1),
                torch.cat((shaped(-sin_basis), shaped(cos_basis)), dim=1),
            ), dim=0)
            target = torch.cat((shaped(target_angle.cos()), shaped(-target_angle.sin())))
            solution = torch.linalg.lstsq(system, target).solution
            cos_out[index], sin_out[index] = solution[:self.M], solution[self.M:]
        return cos_out.float(), sin_out.float()

    @property
    def token_rate_hz(self) -> float:
        return float(self._analysis_rate(0) / 4.0 if self.stem is not None
                     else self._analysis_rate(0))

    def _analysis_rate(self, group: int) -> float:
        if self.frame_rate_hz is not None:
            return float(self.frame_rate_hz)
        assert self.frames_per_span is not None
        return float(self.frames_per_span) / self.span_list[group]

    def _output_rate(self, group: int) -> float:
        return float(self._analysis_rate(group) / 4.0 if self.stem is not None
                     else self._analysis_rate(group))

    def frame_counts(self, duration_s: float) -> list[int]:
        """Analysis frames per group covering `duration_s` seconds from zero."""
        return [max(1, int(math.ceil(duration_s * self._analysis_rate(g) - 1e-6)))
                for g in range(self.G)]

    def token_counts(self, duration_s: float) -> list[int]:
        """Output token counts after the optional convolutional reduction."""
        return [max(1, int(math.ceil(duration_s * self._output_rate(g) - 1e-6)))
                for g in range(self.G)]

    def token_metadata(self, duration_seconds: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build the signal-independent physical token grid for a batch of recordings."""
        if duration_seconds.ndim != 1 or not bool((duration_seconds > 0).all()):
            raise ValueError("duration_seconds must be a positive (B,) tensor")
        batch = duration_seconds.shape[0]
        device = duration_seconds.device
        counts = self.token_counts(float(duration_seconds.max()))
        positions, durations, ids, masks = [], [], [], []
        for group, count in enumerate(counts):
            span = self.span_list[group]
            stride = 1.0 / self._output_rate(group)
            center = (torch.arange(count, device=device, dtype=torch.float32) + 0.5) * stride
            positions.append(center.view(1, count).expand(batch, count))
            durations.append(torch.full((batch, count), span, device=device))
            ids.append(torch.full((batch, count), group, device=device, dtype=torch.long))
            masks.append(center.view(1, count) < duration_seconds.view(batch, 1))
        return {
            "positions": torch.cat(positions, dim=1),
            "durations": torch.cat(durations, dim=1),
            "resolution_ids": torch.cat(ids, dim=1),
            "token_mask": torch.cat(masks, dim=1),
            "frame_counts": counts,
        }

    # ------------------------------------------------------------------ geometry and kernels
    def _group_geometry(self, rate: float, g: int, n_frames: int,
                        device: torch.device) -> dict[str, torch.Tensor]:
        key = (device.type, device.index, float(rate), int(g), int(n_frames))
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached
        T = self.span_list[g]
        stride = 1.0 / self._analysis_rate(g)
        frame_time = (torch.arange(n_frames, device=device, dtype=torch.float32) + 0.5) * stride
        centre = torch.floor(frame_time * rate)
        phase = frame_time * rate - centre
        half = int(math.ceil(T * rate / 2.0)) + 2
        offsets_base = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
        sample_index = (centre.unsqueeze(1) + offsets_base.unsqueeze(0)).long()     # (n, taps)
        unique_phase, phase_id = torch.unique(phase, return_inverse=True)
        u = ((offsets_base.unsqueeze(0) - unique_phase.unsqueeze(1)) / rate) / T   # (n_phase, taps)
        inside = u.abs() <= 0.5
        harmonic = torch.arange(1, self.M + 1, device=device, dtype=torch.float32)
        angle = 2 * math.pi * harmonic.view(1, self.M, 1) * u.unsqueeze(1)        # (n_phase, M, taps)
        cached = {
            "frame_time": frame_time,
            "sample_index": sample_index,
            "phase_id": phase_id,
            "u": u,
            "inside": inside,
            "cos_phase": torch.cos(angle),
            "sin_phase": torch.sin(angle),
            "support": inside.index_select(0, phase_id),                             # (n, taps)
        }
        self._geometry_cache[key] = cached
        return cached

    def _group_kernels(self, geometry: dict[str, torch.Tensor], g: int, rate: float,
                       source_rate: float) -> torch.Tensor:
        """Current learnable kernels of group `g` on the cached bases -> (n, K_g, 2, taps)."""
        sl = self.group_slice(g)
        k = self.group_sizes[g]
        device = self.spans.device
        harmonic = torch.arange(1, self.M + 1, device=device, dtype=self.spans.dtype)
        live = (harmonic / self.span_list[g]) <= (self.nyquist_margin * source_rate / 2.0)  # (M,)
        cos_coeff, sin_coeff = self._normalised_coefficients()
        cos_coeff = (cos_coeff[sl] * live).view(1, k, self.M, 1)
        sin_coeff = (sin_coeff[sl] * live).view(1, k, self.M, 1)
        cos_phase = geometry["cos_phase"].unsqueeze(1)                              # (n_phase,1,M,taps)
        sin_phase = geometry["sin_phase"].unsqueeze(1)
        real = (cos_coeff * cos_phase).sum(2) + (sin_coeff * sin_phase).sum(2)
        imag = (sin_coeff * cos_phase).sum(2) - (cos_coeff * sin_phase).sum(2)
        sigma = self._sigmas()[sl].view(1, k, 1)
        envelope = torch.exp(-geometry["u"].unsqueeze(1).square() / (2 * sigma.square()))
        inside = geometry["inside"].unsqueeze(1)                                    # (n_phase,1,taps)
        pair = torch.stack((real * envelope * inside, imag * envelope * inside), dim=2)
        count = inside.sum(-1, keepdim=True).clamp_min(1).unsqueeze(2)
        pair = (pair - pair.sum(-1, keepdim=True) / count) * inside.unsqueeze(2)
        pair = pair * self._gains()[sl].view(1, k, 1, 1) / float(rate)
        return pair.index_select(0, geometry["phase_id"])

    # ------------------------------------------------------------------ analysis
    def _analyze_rows(self, window: torch.Tensor, total: torch.Tensor, rate: float,
                      source_rate: float, n_frames: list[int]) -> list[dict[str, torch.Tensor]]:
        """One homogeneous (stored rate, source rate) group of rows -> per-span-group results."""
        B, _, C = window.shape
        device = window.device
        signal = window.permute(0, 2, 1)                                             # (B,C,T)
        duration = total / rate
        out = []
        for g in range(self.G):
            geometry = self._group_geometry(rate, g, n_frames[g], device)
            sample_index = geometry["sample_index"]                                 # (n,taps)
            n, taps = sample_index.shape
            with torch.autocast(device_type=device.type, enabled=False):
                kernels = self._group_kernels(geometry, g, rate, source_rate)
            reflected = self._reflect_indices(sample_index, total)                   # (B,n,taps)
            gather = reflected.reshape(B, 1, -1).expand(B, C, -1)
            segment = signal.gather(2, gather).reshape(B, C, n, taps)
            with torch.autocast(device_type=device.type, enabled=False):
                z = torch.einsum("bcft,fkqt->bckfq", segment.float(), kernels)
            magnitude = z.square().sum(-1).clamp_min(1e-12).sqrt()                  # (B,C,K_g,n)
            real_sample = (sample_index.view(1, n, taps) >= 0) & (
                sample_index.view(1, n, taps) < total.view(B, 1, 1))
            support = geometry["support"]                                            # (n,taps)
            local = support.unsqueeze(0) & real_sample                               # (B,n,taps)
            count = local.sum(-1).clamp_min(1)                                       # (B,n)
            edge = (count.float() / support.sum(-1).clamp_min(1).float().view(1, n))  # (B,n)
            weight = local.to(segment.dtype).unsqueeze(1)                            # (B,1,n,taps)
            denominator = count.to(segment.dtype).view(B, 1, n)
            dc = (segment * weight).sum(-1) / denominator                             # (B,C,n)
            amplitude = torch.log1p((segment.abs() * weight).sum(-1) / denominator)  # (B,C,n)
            valid = geometry["frame_time"].view(1, n) < duration.view(B, 1)
            out.append({"magnitude": magnitude, "edge": edge, "valid": valid,
                        "dc": dc, "amplitude": amplitude})
        return out

    @staticmethod
    def _patch_lengths(patch_len_samples, B: int, P: int, S: int,
                       device: torch.device) -> torch.Tensor:
        if patch_len_samples is None:
            return torch.full((B, P), S, dtype=torch.long, device=device)
        patch_len = torch.as_tensor(patch_len_samples, device=device, dtype=torch.long)
        if patch_len.numel() == 1:
            return patch_len.reshape(1, 1).expand(B, P)
        if patch_len.numel() == B:
            return patch_len.reshape(B, 1).expand(B, P)
        if patch_len.numel() == B * P:
            return patch_len.reshape(B, P)
        raise ValueError(f"patch_len_samples must be scalar, (B,), or (B,P)=({B},{P})")

    def analyze_grid(self, patches, sampling_rate_hz, patch_len_samples=None,
                     source_rate_hz=None, patch_mask=None,
                     grid_duration_seconds: torch.Tensor | None = None) -> dict:
        """Per-group frame magnitudes, edge support, validity and local summaries."""
        B, P, S, C = patches.shape
        collect_runtime = self._telemetry_requested
        device = patches.device
        rates = self._rate_vector(sampling_rate_hz, B, device, patches.dtype, "sampling_rate_hz")
        source_rates = (rates if source_rate_hz is None else
                        self._rate_vector(source_rate_hz, B, device, patches.dtype,
                                          "source_rate_hz"))
        patch_len = self._patch_lengths(patch_len_samples, B, P, S, device)
        lengths_fit = (patch_len <= S).all()
        if device.type == "cpu":
            if not bool(lengths_fit):
                raise ValueError(f"patch_len_samples cannot exceed padded sample width S={S}")
        else:
            torch._assert_async(lengths_fit, f"patch_len_samples exceeds sample width S={S}")
        length_valid = patch_len > 0
        patch_mask = (length_valid if patch_mask is None else
                      torch.as_tensor(patch_mask, device=device, dtype=torch.bool).reshape(B, P)
                      & length_valid)
        window, total = self.contiguous_window(patches, patch_len, patch_mask)
        duration = total / rates
        # A past-only student can analyze a shorter prefix while retaining the teacher's complete
        # output shape. This keeps physical target indices aligned without exposing future signal.
        grid_duration = duration if grid_duration_seconds is None else torch.as_tensor(
            grid_duration_seconds, device=device, dtype=duration.dtype,
        ).reshape(B)
        if not bool((grid_duration + 1e-6 >= duration).all()):
            raise ValueError("grid duration cannot be shorter than analyzed signal duration")
        n_frames = self.frame_counts(float(grid_duration.max()))
        groups = []
        for g in range(self.G):
            k, n = self.group_sizes[g], n_frames[g]
            groups.append({
                "magnitude": patches.new_zeros(B, C, k, n, dtype=torch.float32),
                "compressed": patches.new_zeros(B, C, k, n, dtype=torch.float32),
                "edge": patches.new_zeros(B, n, dtype=torch.float32),
                "valid": torch.zeros(B, n, device=device, dtype=torch.bool),
                "dc": patches.new_zeros(B, C, n, dtype=torch.float32),
                "amplitude": patches.new_zeros(B, C, n, dtype=torch.float32),
                "frame_time": (torch.arange(n, device=device, dtype=torch.float32) + 0.5)
                / self._analysis_rate(g),
            })
        pairs = torch.stack((rates, source_rates), dim=1)
        unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
        for pair_id, (rate, source_rate) in enumerate(unique_pairs.detach().cpu().tolist()):
            rows = torch.nonzero(inverse == pair_id, as_tuple=False).flatten()
            results = self._analyze_rows(window.index_select(0, rows), total.index_select(0, rows),
                                         float(rate), float(source_rate), n_frames)
            for g, result in enumerate(results):
                sl = self.group_slice(g)
                groups[g]["magnitude"].index_copy_(0, rows, result["magnitude"])
                knee = (self.compression_knee[sl] if self.compression_scale_mode == "calibrated"
                        else torch.ones_like(self.compression_knee[sl]))
                compressed = torch.log1p(
                    result["magnitude"] / knee.view(1, 1, -1, 1).clamp_min(1e-8)
                )
                groups[g]["compressed"].index_copy_(0, rows, compressed)
                groups[g]["edge"].index_copy_(0, rows, result["edge"])
                groups[g]["valid"].index_copy_(0, rows, result["valid"])
                groups[g]["dc"].index_copy_(0, rows, result["dc"])
                groups[g]["amplitude"].index_copy_(0, rows, result["amplitude"])
        nyq, res = self.masks(rates, duration, source_rates)                           # (B,K)
        if self._telemetry_requested:
            stds, edges, weights = [], [], []
            for g, group in enumerate(groups):
                valid = group["valid"].view(B, 1, 1, -1).to(torch.float32)
                count = (valid.sum() * C).clamp_min(1.0)
                mean = (group["compressed"] * valid).sum(dim=(0, 1, 3)) / count
                variance = ((group["compressed"] - mean.view(1, 1, -1, 1)).square()
                            * valid).sum(dim=(0, 1, 3)) / count
                stds.append(variance.sqrt())
                edges.append((group["edge"] * group["valid"]).sum())
                weights.append(group["valid"].sum())
            std = torch.cat(stds)
            self._runtime_summary = {
                "frontend/observable_fraction": nyq.mean().detach(),
                "frontend/edge_support_mean": (
                    torch.stack(edges).sum() / torch.stack(weights).sum().clamp_min(1)).detach(),
                "frontend/response_std_mean": std.mean().detach(),
                "frontend/dead_kernel_fraction": (std < 1e-5).float().mean().detach(),
            }
            self._telemetry_requested = False
        return {"groups": groups, "nyquist": nyq, "resolution": res, "duration": duration,
                "rate": rates, "source_rate": source_rates, "n_frames": n_frames,
                "collect_runtime": collect_runtime}

    # ------------------------------------------------------------------ normalisation stats
    @torch.no_grad()
    def reset_compression_accumulator(self) -> None:
        self._scale_samples = []

    @torch.no_grad()
    def accumulate_compression_stats(self, patches, sampling_rate_hz, patch_len_samples=None,
                                     patch_mask=None, channel_mask=None, source_rate_hz=None):
        if self.compression_scale_mode != "calibrated":
            return
        analysis = self.analyze_grid(
            patches, sampling_rate_hz, patch_len_samples,
            source_rate_hz=source_rate_hz, patch_mask=patch_mask,
        )
        batch, _, _, channels = patches.shape
        nyq = analysis["nyquist"]
        channel_weight = (torch.ones(batch, channels, device=patches.device, dtype=torch.bool)
                          if channel_mask is None else torch.as_tensor(
                              channel_mask, device=patches.device, dtype=torch.bool))
        collected = []
        for g, group in enumerate(analysis["groups"]):
            sl = self.group_slice(g)
            valid = (group["valid"].view(batch, 1, 1, -1)
                     & channel_weight.view(batch, channels, 1, 1)
                     & nyq[:, sl].view(batch, 1, -1, 1).gt(0))
            values = group["magnitude"].permute(2, 0, 1, 3).reshape(self.group_sizes[g], -1)
            weights = valid.expand_as(group["magnitude"]).permute(2, 0, 1, 3).reshape(
                self.group_sizes[g], -1
            )
            for row, keep in zip(values, weights):
                selected = row[keep]
                if selected.numel() > 4096:
                    stride = max(1, selected.numel() // 4096)
                    selected = selected[::stride][:4096]
                collected.append(selected.detach().float().cpu())
        self._scale_samples.append(torch.nn.utils.rnn.pad_sequence(
            collected, batch_first=True, padding_value=float("nan")
        ))

    @torch.no_grad()
    def finalize_compression_stats(self) -> None:
        if self.compression_scale_mode != "calibrated":
            self.compression_knee.fill_(1.0)
            return
        if not self._scale_samples:
            raise RuntimeError("no observations were accumulated for compression calibration")
        rows = []
        for kernel in range(self.K):
            parts = [batch[kernel][torch.isfinite(batch[kernel])] for batch in self._scale_samples]
            values = torch.cat(parts)
            rows.append(values.median() if values.numel() else torch.tensor(1.0))
        knee = torch.stack(rows).clamp_min(1e-8).to(self.compression_knee)
        self.compression_knee.copy_(knee)
        self._scale_samples = []

    @torch.no_grad()
    def accumulate_norm_stats(self, patches, sampling_rate_hz, patch_len_samples=None,
                              patch_mask=None, channel_mask=None, source_rate_hz=None):
        analysis = self.analyze_grid(patches, sampling_rate_hz, patch_len_samples,
                                     source_rate_hz=source_rate_hz, patch_mask=patch_mask)
        B, _, _, C = patches.shape
        if channel_mask is None:
            channel_weight = torch.ones(B, C, device=patches.device, dtype=torch.float64)
        else:
            channel_weight = torch.as_tensor(channel_mask, device=patches.device).to(torch.float64)
        nyq = analysis["nyquist"].to(torch.float64)
        for g, group in enumerate(analysis["groups"]):
            sl = self.group_slice(g)
            value = group["compressed"].to(torch.float64)                             # (B,C,K_g,n)
            weight = (group["valid"].view(B, 1, 1, -1).to(torch.float64)
                      * nyq[:, sl].view(B, 1, -1, 1) * channel_weight.view(B, C, 1, 1))
            self._acc_count[sl] += weight.sum(dim=(0, 1, 3))
            self._acc_sum[sl] += (value * weight).sum(dim=(0, 1, 3))
            self._acc_sqsum[sl] += (value.square() * weight).sum(dim=(0, 1, 3))
            token_weight = group["valid"].view(B, 1, -1).to(torch.float64) * channel_weight.view(B, C, 1)
            for name in ("amp", "dc"):
                sample = group["amplitude" if name == "amp" else "dc"].to(torch.float64)
                channel_kind = (torch.arange(C, device=sample.device) >= 3).long()
                for kind in range(2):
                    kind_weight = token_weight * channel_kind.eq(kind).view(1, C, 1)
                    getattr(self, f"_{name}_acc_count")[g, kind].add_(kind_weight.sum())
                    getattr(self, f"_{name}_acc_sum")[g, kind].add_(
                        (sample * kind_weight).sum()
                    )
                    getattr(self, f"_{name}_acc_sqsum")[g, kind].add_(
                        (sample.square() * kind_weight).sum()
                    )

    # ------------------------------------------------------------------ projection
    def project_grid(self, analysis: dict, *, sensor_id=None, channel_mask=None,
                     n_sensors=None) -> dict[str, torch.Tensor]:
        groups = analysis["groups"]
        B, C = groups[0]["compressed"].shape[:2]
        device = groups[0]["compressed"].device
        slot, live, _, sensors = self._sensor_layout(B, C, device, sensor_id, channel_mask, n_sensors)
        total_slots = sensors * self.axes_per_sensor
        target = torch.where(live, slot, torch.full_like(slot, total_slots))
        validity = live.new_zeros(B, total_slots + 1)
        validity.scatter_(1, target, live)
        validity = validity[:, :total_slots].reshape(B, 1, sensors, self.axes_per_sensor).float()
        nyq_all = analysis["nyquist"]
        tokens, positions, durations, ids, masks = [], [], [], [], []
        standardized_values = []
        for g, group in enumerate(groups):
            sl = self.group_slice(g)
            k, n = self.group_sizes[g], group["compressed"].shape[-1]
            compressed = group["compressed"]
            if self.norm == "frozen":
                compressed = (compressed - self.norm_mu[sl].view(1, 1, k, 1)) \
                    / self.norm_sd[sl].view(1, 1, k, 1)
            nyq = nyq_all[:, sl]                                                      # (B,k)
            compressed = compressed * nyq.view(B, 1, k, 1)
            source = compressed * live.view(B, C, 1, 1).to(compressed.dtype)
            packed = compressed.new_zeros(B, total_slots + 1, k, n)
            packed.scatter_(1, target.view(B, C, 1, 1).expand(B, C, k, n), source)
            magnitudes = packed[:, :total_slots].reshape(B, sensors, self.axes_per_sensor * k, n)
            magnitudes = magnitudes.permute(0, 3, 1, 2)                               # (B,n,S,3k)
            amplitude, dc = group["amplitude"], group["dc"]                           # (B,C,n)
            if self.norm == "frozen":
                channel_kind = (torch.arange(C, device=device) >= 3).long()
                amp_mu, amp_sd = self.amp_mu[g][channel_kind], self.amp_sd[g][channel_kind]
                dc_mu, dc_sd = self.dc_mu[g][channel_kind], self.dc_sd[g][channel_kind]
                amplitude = (amplitude - amp_mu.view(1, C, 1)) / amp_sd.view(1, C, 1)
                dc = (dc - dc_mu.view(1, C, 1)) / dc_sd.view(1, C, 1)
            amplitude = self._pack_axis_values(amplitude.permute(0, 2, 1), slot, live, total_slots)
            dc = self._pack_axis_values(dc.permute(0, 2, 1), slot, live, total_slots)
            features = torch.cat([
                magnitudes,
                nyq.view(B, 1, 1, k).expand(B, n, sensors, k),
                group["edge"].view(B, n, 1, 1).expand(B, n, sensors, 1),
                amplitude.reshape(B, n, sensors, self.axes_per_sensor),
                dc.reshape(B, n, sensors, self.axes_per_sensor),
                validity.expand(B, n, sensors, self.axes_per_sensor),
            ], dim=-1)
            frame_valid = group["valid"]
            features = features * frame_valid.view(B, n, 1, 1).to(features.dtype)
            if self.stem is None:
                projected = self.proj[g](features)
                output_valid = frame_valid
            else:
                projected, output_valid = self.stem(features, frame_valid, g)
            output_count = projected.shape[1]
            tokens.append(projected)
            stride = 1.0 / self._output_rate(g)
            positions.append(
                (torch.arange(output_count, device=device, dtype=torch.float32) + 0.5)
                .mul(stride).view(1, output_count).expand(B, output_count)
            )
            durations.append(torch.full((B, output_count), self.span_list[g], device=device))
            ids.append(torch.full((B, output_count), g, device=device, dtype=torch.long))
            masks.append(output_valid)
            valid_values = compressed.permute(0, 1, 3, 2)[
                group["valid"].view(B, 1, n, 1).expand(B, C, n, k)
                & live.view(B, C, 1, 1).expand(B, C, n, k)
                & nyq.gt(0).view(B, 1, 1, k).expand(B, C, n, k)
            ]
            standardized_values.append(valid_values)
            if analysis.get("collect_runtime", False) and valid_values.numel():
                self._runtime_summary[f"frontend/group_{g}_standardised_mean_abs"] = (
                    valid_values.mean().abs().detach()
                )
                self._runtime_summary[f"frontend/group_{g}_standardised_sd"] = (
                    valid_values.std(unbiased=False).detach()
                )
        if analysis.get("collect_runtime", False) and standardized_values:
            values = torch.cat(standardized_values)
            self._runtime_summary["frontend/standardised_mean_abs"] = values.mean().abs().detach()
            self._runtime_summary["frontend/standardised_sd"] = values.std(unbiased=False).detach()
        return {
            "tokens": torch.cat(tokens, dim=1),
            "positions": torch.cat(positions, dim=1),
            "durations": torch.cat(durations, dim=1),
            "resolution_ids": torch.cat(ids, dim=1),
            "token_mask": torch.cat(masks, dim=1),
        }

    def token_grid(self, patches, sampling_rate_hz, patch_len_samples=None,
                   source_rate_hz=None, patch_mask=None, sensor_id=None,
                   grid_duration_seconds=None,
                   channel_mask=None, n_sensors=None) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=patches.device.type, enabled=False):
            analysis = self.analyze_grid(patches.float(), sampling_rate_hz, patch_len_samples,
                                         source_rate_hz=source_rate_hz, patch_mask=patch_mask,
                                         grid_duration_seconds=grid_duration_seconds)
        return self.project_grid(analysis, sensor_id=sensor_id, channel_mask=channel_mask,
                                 n_sensors=n_sensors)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        revision = state_dict.get(prefix + "_frontend_revision")
        if revision is not None and int(revision) == 2 and int(self._frontend_revision) == 2:
            state_dict.setdefault(prefix + "compression_knee", self.compression_knee.detach().clone())
            for name in ("amp_mu", "amp_sd", "dc_mu", "dc_sd"):
                key = prefix + name
                value = state_dict.get(key)
                if value is not None and value.ndim == 1:
                    state_dict[key] = value[:, None].expand(-1, 2).clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, patches, sampling_rate_hz, patch_len_samples=None,
                source_rate_hz=None, patch_mask=None, sensor_id=None,
                channel_mask=None, n_sensors=None) -> dict[str, torch.Tensor]:
        return self.token_grid(patches, sampling_rate_hz, patch_len_samples,
                               source_rate_hz=source_rate_hz, patch_mask=patch_mask,
                               sensor_id=sensor_id, channel_mask=channel_mask,
                               n_sensors=n_sensors)

    # The parent's single-grid entry points are not defined for this layout.
    def analyze(self, *args, **kwargs):
        raise NotImplementedError("MultiSpanKernelTokenizer emits a token grid; use token_grid()")

    def project(self, *args, **kwargs):
        raise NotImplementedError("MultiSpanKernelTokenizer emits a token grid; use token_grid()")
