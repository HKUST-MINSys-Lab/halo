# Frontend compute budget, and how long the JEPA window should be

**Date:** 2026-09-12
**Status:** immutable entry — see `docs/journal/README.md`. Design analysis with measurements;
implementation is specified in `docs/design/FRONTEND_PLAN_20260912.md`. Follows
`2026-09-12-fixed-filterbank-decision.md` (0.5/1/2/4/8 s ladder on 8 s windows) and answers two
questions raised against it: how to add the long patches without wasting compute, and whether the
JEPA pretraining window should grow to ~32 s.

## The waste is in the shared DFT length, not in the long patches

The filterbank zero-pads every patch to one shared rDFT length `S` and analyses all patch lengths
through it (`filterbank.py:50-61`, guard at `:319-334`). That coupling is what makes long patches
look expensive: an 8 s patch at 240 Hz is 1,920 samples, so a shared `S` must become 2048, and
then **every short patch is padded to 2048 too**. A 0.5 s patch at 20 Hz carries 10 real samples
in a 2048-point transform.

Relative rDFT work per sensor-channel (`sum over patches of S log2 S`), with today's
configuration as 1.00:

| design | tokens/sensor | rDFT work | vs today |
|---|---:|---:|---:|
| **today** — 8 s, 0.5/1/1.5, shared S=512 | 30 | 138,240 | 1.00× |
| 5-ladder 8 s, shared S (forced to 2048) | 31 | 698,368 | **5.05×** |
| 5-ladder 8 s, **per-resolution S** (128/256/512/1024/2048) | 31 | 92,160 | **0.67×** |
| 5-ladder 8 s, **decimated to a 40 Hz analysis rate** (S 32/64/128/256/512) | 31 | 17,920 | **0.13×** |

Two conclusions. **Per-resolution `S` makes the five-resolution ladder cheaper than today's
three-resolution one** — 0.67×, while adding two octaves of frequency resolution. And the naive
shared-`S` route is the only expensive option; it is 7.6× the cost of doing the same thing with
per-resolution lengths.

## The larger win: the analysis rate should follow `f_max`, not the acquisition rate

The bank measures 32 bands spanning 0.3–15 Hz. Nothing above ~15 Hz is ever read. But every patch
is transformed at its *native* rate, so a 240 Hz Nymeria patch carries 6× the samples the analysis
can use, and a 100 Hz Capture-24 patch carries 2.5×. Anti-alias-decimating each patch to
`min(source_rate, ~40 Hz)` before the rDFT (40 Hz gives Nyquist 20 Hz, comfortably above the 15 Hz
ceiling) makes the transform length a function of patch *duration* alone: 8 s → 320 samples → S=512,
0.5 s → 20 samples → S=32. That is **0.13× today's cost, a 7.7× reduction**, and it shrinks the
patch tensor correspondingly (the `(B, P, S, C)` input at S=2048 is ~400 MB at batch 264; at
S=512 it is ~100 MB).

Three implementation cautions, because this touches the rate contract:

- **The Nyquist observability mask must keep keying off the original source rate**, not the
  analysis rate. After decimation every patch looks like 40 Hz; which bands are physically real
  still depends on how the data was acquired. This is the one place where getting it wrong would
  silently fabricate bands.
- **Never upsample.** A 20 Hz source stays at 20 Hz (`min(source_rate, 40)`); its existing
  observability mask already cuts everything above 9 Hz.
- **It may improve, not harm, rate invariance.** Decimating heterogeneous sources onto a common
  analysis grid removes a source of cross-rate mismatch; the existing test (correlation > 0.97 at
  20/25/50 vs 100 Hz, `tests/test_multispan_kernel.py`) is the check, and it should get *better*.
  That is a hypothesis to verify, not an assumption.

A useful implementation note: the patch groups are contiguous along the patch axis, and patches
are zero-padded at the end, so per-resolution lengths can be obtained by slicing
`patches[:, group_slice, :S_g, :]` before the transform — no change to the collate contract is
strictly required to get the 0.67× version. The 0.13× version does need the collate to decimate.

## The JEPA window: 16 s, not 32 s

The case for a longer pretraining window is strong and was made in
`2026-09-12-long-window-proposal.md`: on an 8 s window with horizon bins of 0–3 s, "the future
looks like the present" is true almost everywhere, which is the measured cause of the horizon-flat
loss. A longer window buys genuinely harder horizons.

The constraint is the label-free corpus. Fraction of continuous executions long enough to supply
one window, and the resulting non-overlapping window yield:

| source | rate | ≥8 s | ≥16 s | ≥24 s | ≥32 s | windows @8 s | @16 s | @32 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| capture24_pretrain | 100 | 1.00 | 0.99 | 0.99 | 0.99 | 1,747,520 | 873,716 | 436,816 |
| extrasensory_pretrain | 50 | 1.00 | **0.89** | 0.19 | **0.00** | 305,279 | 108,461 | **9** |
| nymeria_xsens | 240 | 1.00 | 1.00 | 1.00 | 1.00 | 51,095 | 25,443 | 12,606 |

**At 32 s, ExtraSensory yields nine windows — it is gone.** That is not a minor trim: it is the
only label-free source with phone placements and a consumer-grade watch, leaving a corpus of
Capture-24 (wrist accelerometer only) plus Nymeria (research-grade Xsens). BenchHAR's finding that
consumer-grade data generalises better to research-grade than the reverse (arXiv 2605.08296) makes
that a real loss, not a bookkeeping one.

**At 16 s ExtraSensory survives at 89% of executions and 108k windows**, and the horizon still
roughly triples: with the context boundary at 0.4–0.7 of the window, prediction horizons run to
~9.6 s instead of ~3 s. Cost is 124 tokens per sensor (248 per two-sensor window), so the
`MAX_BATCH_TOKENS = 16,384` ceiling puts batch at 132 rather than today's 273 — a 2× reduction,
against 4× (batch 66) at 32 s. Batch matters here beyond throughput: the VICReg variance term
estimates per-dimension statistics across the batch, and 66 samples over 256 dimensions is a noisy
estimate.

Combined with per-resolution `S`, a 16 s five-resolution window costs 1.33× today's frontend work;
with decimation it is 0.26×. So the longer window is affordable either way, and essentially free
with decimation.

**Recommendation:** JEPA pretraining at **16 s**, supervised/sealed at **8 s**. The encoder is
duration-agnostic by construction, so the 2× ratio is supported, and it is a far milder version of
the pretrain-long/adapt-short asymmetry than the 60 s proposal. The horizon bins must be widened
from the current `((0,1),(1,2),(2,3))` to something like `((0,2),(2,5),(5,10))` or the extra window
length buys nothing. If a 32 s arm is wanted later it should be run explicitly as a
Capture-24 + Nymeria corpus and disclosed as such, not as a drop-in replacement.

**Related:** `docs/design/FRONTEND_PLAN_20260912.md`,
`2026-09-12-fixed-filterbank-decision.md`, `2026-09-12-long-window-proposal.md`,
`2026-09-12-jepa-and-encoder-findings.md`.
