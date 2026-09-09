"""Multi-span continuous-kernel tokenizer: one token per (span group, frame, sensor).

Design of record: `docs/design/CONTINUOUS_KERNEL_FRONTEND.md`, section "Multi-span tokenization".
Selected with `--frontend multispan`.

WHAT IT IS
----------
The single-span `ContinuousKernelTokenizer` runs one constant-Q bank at 8 frames per second and
packs four ordered frames into a one-second token through a dense CNN. This module keeps the
continuous kernels and the rate contract but changes what a token IS:

* the bank is organised by physical SPAN. Group `g` has span `T_g` seconds (default 0.25, 0.5, 1,
  2 s) and holds one Gabor kernel per harmonic of `1 / T_g`, capped at both `f_max` and
  `n_harmonics / T_g`, so the same frequency is
  measured narrowband at long spans and broadband at short ones. A two-dimensional tiling of the
  time-frequency plane instead of one constant-Q line;
* each group's envelope is sampled at its OWN temporal resolution: stride `T_g / frames_per_span`,
  i.e. four frames per span by default. This is a temporal-resolution/compute choice, not a strict
  no-aliasing guarantee for the learned responses;
* every frame of every group becomes one token per sensor, carrying its physical centre time (for
  the trunk's RoPE), its span (the duration embedding) and its group (the resolution id). Sub-second
  order is positional inside the attention sequence, which is what the CNN's ordered flatten was
  approximating; the trunk's per-resolution pooling then gives one vector per recording.

The kernels themselves are exactly the single-span module's: a Gaussian-windowed Fourier series in
normalised time, evaluated at the exact offsets of the real samples, scaled by `dt`, re-zero-meaned,
with the observability mask equal to the coefficient energy that survives the Nyquist cut.

The collate's patch grid is irrelevant here: the contiguous window is rebuilt from the patches and
the frames are laid out in physical time from zero. One-second and 1.5 s collates give the same
tokens.

CONTRACT
--------
`token_grid(patches, sampling_rate_hz, patch_len_samples, source_rate_hz=None, patch_mask=None,
sensor_id=None, channel_mask=None, n_sensors=None)` returns a dict::

    tokens          (B, N, S, d_model)   N = sum over groups of that group's frame count
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

MS_SPANS_S = (0.25, 0.5, 1.0, 2.0)   # octave-spaced so the harmonic grids nest (4, 2, 1, 0.5 Hz)
MS_FRAMES_PER_SPAN = 4               # default envelope stride T/4; validate fidelity empirically


class MultiSpanKernelTokenizer(ContinuousKernelTokenizer):
    """Span-grouped continuous kernels emitting a physical-time token grid."""

    def __init__(
        self,
        d_model: int = 128,
        spans: Sequence[float] = MS_SPANS_S,
        frames_per_span: int = MS_FRAMES_PER_SPAN,
        n_harmonics: int = CK_N_HARMONICS,
        f_max: float = CK_F_MAX_HZ,
        nyquist_margin: float = CK_NYQUIST_MARGIN,
        envelope_sigma: float = CK_ENVELOPE_SIGMA,
        gabor_init: bool = True,
        norm: str = "frozen",
    ):
        nn.Module.__init__(self)
        self._validate_analysis_config(n_harmonics, nyquist_margin, norm)
        if not math.isfinite(f_max) or f_max <= 0:
            raise ValueError("f_max must be finite and positive")
        self.register_buffer("_frontend_revision", torch.tensor(2, dtype=torch.long))
        spans = tuple(sorted(float(s) for s in spans))
        if not spans or not all(math.isfinite(s) and s > 0 for s in spans) \
                or len(set(spans)) != len(spans):
            raise ValueError("spans must be distinct finite positive durations in seconds")
        if not math.isfinite(frames_per_span) or frames_per_span < 1 \
                or frames_per_span != int(frames_per_span):
            raise ValueError("frames_per_span must be a positive integer")
        if n_harmonics < 1:
            raise ValueError("need at least 1 harmonic")
        self.M = int(n_harmonics)
        self.d_model = int(d_model)
        self.frames_per_span = int(frames_per_span)
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

        # Each span has at most M carriers; long spans intentionally cover lower frequencies.
        span_of, carrier_of, group_of = [], [], []
        for g, T in enumerate(spans):
            n_harm = min(self.M, int(math.floor(f_max * T + 1e-6)))
            if n_harm < 1:
                raise ValueError(f"span {T} s holds no harmonic at or below f_max={f_max} Hz")
            for m in range(1, n_harm + 1):
                span_of.append(T)
                carrier_of.append(m)
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
        self.register_buffer("centres", self.carrier.float() / self.spans)             # (K,) Hz
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
        self.gain_logit = nn.Parameter(torch.zeros(self.K))
        if gabor_init:
            with torch.no_grad():
                self.cos_coeff[torch.arange(self.K), self.carrier - 1] = 1.0
        else:
            nn.init.normal_(self.cos_coeff, std=0.3)
            nn.init.normal_(self.sin_coeff, std=0.3)
        self.register_buffer("initial_cos_coeff", self._normalised_coefficients()[0].detach().clone())
        self.register_buffer("initial_sin_coeff", self._normalised_coefficients()[1].detach().clone())
        self.register_buffer("initial_sigma", torch.full((self.K,), float(envelope_sigma)))

        # --- frozen standardisation of the compressed magnitude, per kernel ---
        self.register_buffer("norm_mu", torch.zeros(self.K))
        self.register_buffer("norm_sd", torch.ones(self.K))
        self.register_buffer("_norm_fitted", torch.zeros(1))
        self.register_buffer("_acc_count", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sum", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sqsum", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        for name in ("amp", "dc"):
            self.register_buffer(f"{name}_mu", torch.zeros(self.G))
            self.register_buffer(f"{name}_sd", torch.ones(self.G))
            self.register_buffer(f"_{name}_acc_count", torch.zeros(self.G, dtype=torch.float64),
                                 persistent=False)
            self.register_buffer(f"_{name}_acc_sum", torch.zeros(self.G, dtype=torch.float64),
                                 persistent=False)
            self.register_buffer(f"_{name}_acc_sqsum", torch.zeros(self.G, dtype=torch.float64),
                                 persistent=False)

        # --- one linear projection per span group ---
        self.proj = nn.ModuleList([
            nn.Linear(self.group_in_dim(g), self.d_model) for g in range(self.G)
        ])

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

    def frame_counts(self, duration_s: float) -> list[int]:
        """Frames per group covering `duration_s` seconds from zero."""
        return [max(1, int(math.ceil(duration_s * self.frames_per_span / T - 1e-6)))
                for T in self.span_list]

    # ------------------------------------------------------------------ geometry and kernels
    def _group_geometry(self, rate: float, g: int, n_frames: int,
                        device: torch.device) -> dict[str, torch.Tensor]:
        key = (device.type, device.index, float(rate), int(g), int(n_frames))
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached
        T = self.span_list[g]
        stride = T / self.frames_per_span
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
                     source_rate_hz=None, patch_mask=None) -> dict:
        """Per-group frame magnitudes, edge support, validity and local summaries."""
        B, P, S, C = patches.shape
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
        # One synchronisation: the grid length follows the longest recording in the batch.
        n_frames = self.frame_counts(float(duration.max()))
        groups = []
        for g in range(self.G):
            k, n = self.group_sizes[g], n_frames[g]
            groups.append({
                "compressed": patches.new_zeros(B, C, k, n, dtype=torch.float32),
                "edge": patches.new_zeros(B, n, dtype=torch.float32),
                "valid": torch.zeros(B, n, device=device, dtype=torch.bool),
                "dc": patches.new_zeros(B, C, n, dtype=torch.float32),
                "amplitude": patches.new_zeros(B, C, n, dtype=torch.float32),
                "frame_time": (torch.arange(n, device=device, dtype=torch.float32) + 0.5)
                * (self.span_list[g] / self.frames_per_span),
            })
        pairs = torch.stack((rates, source_rates), dim=1)
        unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
        for pair_id, (rate, source_rate) in enumerate(unique_pairs.detach().cpu().tolist()):
            rows = torch.nonzero(inverse == pair_id, as_tuple=False).flatten()
            results = self._analyze_rows(window.index_select(0, rows), total.index_select(0, rows),
                                         float(rate), float(source_rate), n_frames)
            for g, result in enumerate(results):
                groups[g]["compressed"].index_copy_(0, rows, torch.log1p(result["magnitude"]))
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
                "rate": rates, "source_rate": source_rates, "n_frames": n_frames}

    # ------------------------------------------------------------------ normalisation stats
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
                getattr(self, f"_{name}_acc_count")[g].add_(token_weight.sum())
                getattr(self, f"_{name}_acc_sum")[g].add_((sample * token_weight).sum())
                getattr(self, f"_{name}_acc_sqsum")[g].add_((sample.square() * token_weight).sum())

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
                amplitude = (amplitude - self.amp_mu[g]) / self.amp_sd[g]
                dc = (dc - self.dc_mu[g]) / self.dc_sd[g]
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
            tokens.append(self.proj[g](features))                                     # (B,n,S,d)
            positions.append(group["frame_time"].view(1, n).expand(B, n))
            durations.append(torch.full((B, n), self.span_list[g], device=device))
            ids.append(torch.full((B, n), g, device=device, dtype=torch.long))
            masks.append(group["valid"])
        return {
            "tokens": torch.cat(tokens, dim=1),
            "positions": torch.cat(positions, dim=1),
            "durations": torch.cat(durations, dim=1),
            "resolution_ids": torch.cat(ids, dim=1),
            "token_mask": torch.cat(masks, dim=1),
        }

    def token_grid(self, patches, sampling_rate_hz, patch_len_samples=None,
                   source_rate_hz=None, patch_mask=None, sensor_id=None,
                   channel_mask=None, n_sensors=None) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=patches.device.type, enabled=False):
            analysis = self.analyze_grid(patches.float(), sampling_rate_hz, patch_len_samples,
                                         source_rate_hz=source_rate_hz, patch_mask=patch_mask)
        return self.project_grid(analysis, sensor_id=sensor_id, channel_mask=channel_mask,
                                 n_sensors=n_sensors)

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
