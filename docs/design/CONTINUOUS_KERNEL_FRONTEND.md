# Continuous physical-time frontend

> **Implemented encoder arm, 2026-08-24; audited and corrected 2026-09-09** (see "Corrections"
> below). Retained because temporal resolution may matter directly for sequence matching and
> phase-local movement comparison. Both the single-span and multi-span arms are implemented.
> The multi-span arm is ready for short end-to-end experiments; it has no trained result yet.

## Purpose

The fixed HALO filterbank summarizes motion in physical frequency bands. The continuous frontend
adds learnable sub-second temporal structure while defining every analysis kernel and stride in
seconds rather than samples. A 25 Hz and a 100 Hz recording therefore evaluate the same continuous
kernel at different sample locations and emit the same physical output frame rate.

## Current shape

### Parameter and stability policy

Physical constraints come from the recording metadata, while architecture choices are experiment
settings. No dataset-name-specific gains, thresholds or branches belong in this frontend.

| Quantity | Rule and rationale |
|---|---|
| Sampling and kernel spacing | Use seconds and the actual stored/source rates; never assume a fixed sample count means a fixed duration. |
| Feature scale | Fit frozen statistics on source/label-balanced training samples. Kernel statistics are per kernel; multi-span amplitude/DC statistics are per span. Exclude missing channels and padding. Never recalibrate on test data. |
| Physical units | Converters must supply the agreed units before encoding. Statistical normalization does not repair incorrect units or metadata. Avoid per-record amplitude normalization because intensity can be useful signal. |
| Temporal identity | Derive duration-embedding bounds, resolution count and RoPE period from the configured spans. Preserve actual time positions and mask unavailable tokens. |
| Span, harmonic and frame counts | Configurable capacity/compute choices, not physical laws or demonstrated optima. Keep the same settings across datasets in a comparison and save them with the checkpoint. |
| Envelope/gain bounds | Shared dimensionless constants in `continuous_kernel.py`; prevent unconstrained scaling and degenerate envelopes. These are engineering choices, not dataset-specific fitted values. |
| Numerical floors | Prevent division by zero or undefined magnitude derivatives; they do not define normal feature scale. Calibration spread and finite-gradient checks determine whether the model is operating away from those floors. |

New adaptive rules need a measured failure and a clear contract. Prefer an existing shared helper
over a parallel implementation; add a configuration knob only when it represents a useful
experiment. Reject invalid rates, normalization modes and geometry instead of silently converting
them into a different experiment. Keep train/evaluation preprocessing and saved model semantics
consistent, and test mixed rates, lengths, missing channels and gradients when changing them.

The default frame stride is a compute/resolution tradeoff. The Gaussian envelope is not strictly
band-limited, and learning can change the response bandwidth, so four frames per span is not an
exact no-aliasing guarantee. Validate temporal fidelity empirically when changing spans or frames.

### Single-span layout

For each accelerometer or gyroscope xyz triad:

```text
native xyz samples
  -> 32 shared continuous physical-time kernels on each axis
  -> 96 response channels at 8 frames/second
  -> dense Conv1d(96 -> 64, kernel 3, stride 2) + LayerNorm + GELU
  -> dense Conv1d(64 -> 128, kernel 3, stride 1) + LayerNorm + GELU
  -> four ordered frames per one-second patch
  -> concatenate observability, edge support, amplitude, signed DC, and axis-validity features
  -> project to one token for the physical sensor
```

The dense CNN mixes all three axes within one sensor but never crosses accelerometer/gyroscope or
device boundaries. Missing axes remain explicit through validity bits.

## Kernel parameterization

Each analysis kernel is a smooth, windowed continuous curve represented by a small Fourier basis and
sampled at the recording's native rate. Read it as a filterbank run in reverse: the 24 coefficients
of kernel `k` ARE its amplitude spectrum on the grid `m / T_k` Hz (m = 1..12) under a Gaussian
window; the module synthesises that spectrum into a time-domain kernel and correlates it with the
signal at the exact sample offsets. The cos/sin pair is the analytic kernel, so the frame value
`|z|` is a band envelope: carrier phase is discarded by design, and waveform shape inside a band is
carried by multi-harmonic templates plus the timing of their envelope peaks across frames.

**Bank geometry (2026-09-09).** Centres are log-spaced on [0.5, 15] Hz. Each kernel spans as many
whole cycles of its centre as fit in 2 s, at most four, so the carrier harmonic sits exactly at the
centre and all 32 (span, carrier) pairs are distinct. The frequency resolution of a kernel is set by
its own span (`1 / T_k`, smeared by the envelope to about `0.72 / T_k`), NOT by the token duration:
changing `patch_seconds` only changes how many frames are grouped per token. One-cycle kernels
(0.5-1 Hz) are inherently broadband; that is the honest resolution below 1 Hz in a 2 s span.

**Observability.** The mask is the fraction of a kernel's normalised coefficient energy carried by
harmonics below the source Nyquist. At Gabor initialisation it is exactly binary (the carrier is
live or not); it becomes fractional only once learning puts energy on harmonics a low-rate source
cannot supply. Kernels are still built with the dead harmonics zeroed.

**Patch grid.** The module is built for one physical patch duration (`patch_seconds`, default 1 s)
and lays its 8 Hz analysis frames on that grid, so a 1.5 s token packages 12 frames (6 after the
stride-2 stage). `analyze` refuses a batch whose full patches do not match the grid within one
sample. The compare trainer passes `--patch-seconds` through and records it in the checkpoint
config; `--resolutions` (several grids in one sequence) is still refused for this frontend.

The implementation has 32 active analysis kernels and 135,808 frontend parameters, of which 832 are
the continuous analysis bank. Gradients reach every learnable parameter. Mixed-rate, source-rate,
missing-axis, accel-only, modality-isolation, and end-to-end paths have focused regression tests in
`tests/test_continuous_kernel.py`.

## Corrections (2026-09-09 audit)

Verified correct: kernel synthesis and the Hilbert-pair convention, integral scaling, re-zero-mean,
exact fractional-offset geometry, reflection padding, per-rate grouping, gradient reach, channel-only
LayerNorm. A pure tone at a carrier gives a flat envelope across frames (min/max 0.998). Fixed:

1. **Duplicate kernels.** The old rule `T = clamp(4 / f, 0.05, 2)` with a rounded carrier snapped
   all sixteen kernels below 2 Hz onto the 0.5 Hz grid of a 2 s span: eight were the same 0.59 Hz
   kernel, four the same 1 Hz kernel; 20 of 32 were distinct. After 35k steps of the 2026-09-08
   continuous run the eight still had pairwise tuning-curve correlations of 0.89-0.99. Now every
   kernel is distinct with its carrier exactly on its centre; `f_min` moved from 0.3 to 0.5 Hz
   because nothing below `1 / t_max` was ever reachable.
2. **Rate fingerprint from the mask.** The harmonic-slot fraction multiplied the standardised
   response of INTACT kernels by 0.33 (8 Hz kernel at 20 Hz) and 0.42-0.67 (top kernels at 50 Hz),
   the per-device magnitude difference Rule 2 forbids. Replaced by the retained-energy fraction.
   The cross-rate tests had selected only fully live kernels and could not see this.
3. **Hard-coded one-second layout.** Frames were `P x 8` regardless of patch length, so four 1.5 s
   patches were analysed as seconds 0-4 with the last two seconds dropped and every frame marked
   valid. The layout now follows `patch_seconds`; foreign grids raise.
4. **Trainer wiring.** The compare trainer trained the analysis bank at the full encoder rate with
   no pull toward the Gabor initialisation and no telemetry (the pretraining path had all three).
   It now has `--frontend-lr-scale`, `--frontend-reg-weight` (both default to the old behaviour) and
   logs the `frontend/*` shape-shift, envelope, gain and observability summaries.
5. The Gabor-initialisation test probed three kernels at 3x spacing and fed one 6 s "patch", so it
   analysed only its first second; it now sweeps every kernel at 3% spacing on a real grid.

Not changed: the resolution flag (`duration / span`, clamped) is a constant 1.0 for every 6 s
window and therefore 32 dead input dimensions; kept for checkpoint compatibility. Existing
continuous checkpoints load unchanged (spans and carriers are buffers) but score under the new mask.

**Pooled-frame agreement with the fixed filterbank.** Averaging the squared envelope over a
patch's frames reproduces the fixed filterbank's per-patch log band energy with per-band
correlation 0.84-0.95 for every band above 0.8 Hz on stationary band-limited noise (0.74 for the
one-cycle kernels). Reversing a window in time changes the continuous token by 32% and the fixed
filterbank token by exactly 0. So the frames are a superset of the filterbank at initialisation:
band energy plus its 125 ms envelope timing, minus carrier phase.

## Measured prior result

Under the previous generic HAR protocol, the dense continuous arm improved HALO's zero-shot point
estimate and the learned reranker but slightly reduced direct 1-NN representation performance. The
dataset-bootstrap intervals crossed zero for the headline frontend differences. That result is mixed
and does not promote this frontend universally.

The application comparison is more diagnostic: evaluate fixed and continuous frontends with the
same Task-1 sequence matcher and Task-2 phase-local score. The continuous arm earns its cost only if
its ordered sub-second features improve event boundaries, same-motion verification, or localization
of known execution changes.

## Cost

The last measured RTX 4090 end-to-end profile used about 129 ms per historical episodic training step
and 3.89 GiB allocated VRAM. In an isolated mixed-rate batch of 512 windows, continuous analysis was
the dominant frontend cost; triad packing, dense CNN, and projection were comparatively small.

Application inference must re-profile complete continuous sessions. Reusing overlapping analysis
frames is likely more important than optimizing the ordinary dense CNN.

## Multi-span tokenization (active encoder variant)

Direction from the user: a bank is only meaningful with a real variety of physical spans, each
span's responses kept at a temporal resolution that does not throw away what that span resolved,
all of it self-attended with explicit time and span identity, and pooled into one vector for
comparison; query and support must go through the same encoder with gradients. Implemented in
`model/tokenizer/multispan_kernel.py` as a separate frontend arm; the single-span continuous arm
remains available. It reuses the multi-resolution filterbank machinery (tagged token
grids, centre-time RoPE, log-duration embedding, equal-weight per-resolution pooling) rather than
duplicating it.

**Bank.** Span groups `T in {0.5, 1, 1.5}` s (`--spans`). Within a group the kernels are the
harmonics of `1 / T`, capped at both 15 Hz and harmonic 12: 7, 12 and 12 kernels, 31 in total.
Their highest initial carrier frequencies are 14, 12 and 8 Hz respectively. This intentional
capacity cap keeps the experiment small; it is not full coverage to 15 Hz at every span. The same
frequency is measured at every retained span, narrowband at long spans and broadband at short ones:
a two-dimensional tiling of the
time-frequency plane. Each kernel keeps its 24 learnable coefficients, envelope and gain; the rate
contract (exact sample offsets, integral scaling, re-zero-mean, energy-retained observability mask)
is unchanged and tested per group (cross-rate correlation > 0.97 at 20/25/50 vs 100 Hz).

**Frames per group.** Stride `T / frames_per_span` (`--frames-per-span`, default 4, an initial
temporal-resolution choice rather than an exact bandwidth guarantee). The eight-second JEPA window
yields 64 + 32 + 22 = 118 tokens per
sensor. The grid follows the longest recording in the batch; shorter recordings are masked beyond
their own duration and get exactly the tokens they get alone (tested).

**Tokens.** One token per (group, frame, sensor): the group's standardised kernel magnitudes on the
three axes, its observability entries, the span's edge support at that frame, the local log
amplitude and signed DC over the span, and the axis-validity bits, through one linear map per
group. Every token carries its physical centre time (RoPE, fastest period = two strides of the
finest group, 0.25 s by default), its span (duration embedding over [0.5, 1.5] s) and its group
(`resolution_id`, `num_resolutions = 3`). The encoder's forward takes these from the frontend
instead of the collate, so the collate's patch grid only supplies the contiguous window: 1 s and
1.5 s collates give identical tokens (tested). The dense CNN and ordered flatten are gone;
reversing a recording reverses each group's token sequence and changes the pooled vector through
RoPE, where the fixed filterbank is reversal-invariant (tested).

Kernel magnitudes have frozen per-kernel statistics. Local amplitude and signed DC have frozen
per-span statistics, fitted on live training samples only. Missing axes and padding never
contribute to calibration; unobserved groups fall back to mean zero and scale one.

Checkpoints record `multiresolution=true`, `token_grid_owner=frontend`, and the actual spans in
`eval_resolutions`. These describe the output tokens. Input still uses one non-overlapping patch
grid; `--patching checkpoint` selects that packaging automatically. Explicit evaluation
`--patching multiresolution` and multiple input resolution IDs are rejected because concatenating
those grids would duplicate the raw recording.

**Checkpoint revision.** Current continuous modules save `_frontend_revision=2`. Earlier
checkpoints without this marker are rejected with an actionable error: they were trained with
different observability mathematics or, for the multi-span pilot, shared amplitude/DC statistics.
Reproduce those historical runs using their original commit and saved source patch. Training the
corrected frontend requires a fresh run and fresh evaluation. Do not add a revision marker to an
old checkpoint to bypass the check. The 2026-09-08 continuous score is historical, not a score for
this implementation and remains archived outside the live result record.

**Pooling and gradients.** Mean within group, equal weight across groups (the existing rule).
Every query and support window of an episode goes through the same encoder in one forward pass and
the retained support-classification control keeps support vectors attached, so gradients reach the
kernels from both sides; `frontend/*` telemetry and `--frontend-lr-scale` /
`--frontend-reg-weight` apply.

**Cost, measured on the RTX 4090** (40 steps, `neighbors` readout, four episodes per step, capped
corpus, four loader workers): 43 ms per step against 17 ms for the fixed filterbank, i.e. 2.6x,
so a 35k-step run is about 25 minutes. Per-token export for the evaluation adapter follows the
frontend's grid (`out["token_grid"]`).

**Future JEPA.** The past-only future objective supports this frontend directly. Student kernels see
only the raw prefix; the EMA teacher sees the complete window. The trainer derives RoPE's fastest
period from the shortest span and frame density, audits targets per span, logs frontend gradient,
observability and dead-kernel telemetry, and sizes the batch from the actual emitted token count.
The historical bidirectional masked objective remains unsupported because it cannot provide honest
raw-signal masking for kernels whose support overlaps a masked interval.

Smoke and tests: `tests/test_multispan_kernel.py` plus the encoder/export suites, including
normalization, checkpoint revision checks and refusal of duplicated input grids.

Verification on 2026-09-09: 179 focused tests passed. A three-step real-corpus RTX 4090 smoke
completed with finite training/validation losses and nonzero query, support, kernel and duration
gradients. Reloading its checkpoint through the detailed evaluator exported 120 tokens for a
4 s recording and 180 for a 6 s recording, with valid physical times. The invalid multi-resolution
input override raised an error before encoding. Local diagnostic artifacts are in
`/tmp/halo_multispan_fixes_20260909/`; these smoke scores are not performance results.

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python -m training.support_classifier.train --frontend multispan \
  --out training/support_classifier/outputs/<run> [--spans 0.5 1 1.5] \
  [--frames-per-span 4] [--frontend-lr-scale 1.0] [--frontend-reg-weight 0.0]
```
