# Decision: focus on the fixed filterbank; add 4 s and 8 s patches

**Date:** 2026-09-12
**Status:** immutable entry — see `docs/journal/README.md`. This records a **design decision and
the measurements behind it**. It does not supersede
`2026-09-12-long-window-proposal.md`; it narrows where that proposal applies (long windows remain
relevant to pretraining, not to the supervised/sealed path). Decision is Alex's; the supporting
measurements were run the same day and are reproducible from
`training/tokenizer/diagnostics/frontend/`.

## The decision

Concentrate encoder work on the **fixed multiresolution filterbank**, and extend its patch
ladder with **4 s and 8 s patches** (giving 0.5 / 1 / 2 / 4 / 8 s). The continuous-kernel
multi-span arm is retained as an ablation/control but is no longer the primary development
target.

## The reasoning

The continuous-kernel design buys three things, and on inspection the fixed multiresolution bank
already has two of them:

1. **Rate and duration agnosticism through continuity.** Real, but not exclusive: the fixed
   physical filterbank is *also* rate-aware by construction (physical-Hz bands, Nyquist
   observability mask), and measured rate-invariant — correlation 0.990–0.992 between embeddings
   of the same physical signal presented at 20/25/50/100 vs 240 Hz, with 26 of 32 bands still
   observable at 20 Hz (probe: `knn_probe.py` / earlier rate probe, 2026-09-12).
2. **Imitating different frequency resolutions by varying the physical time a kernel spans.**
   This is exactly what a multiresolution patch ladder does. Varying patch length *is* the
   mechanism; continuous kernels are one way to implement it, not the only one.
3. **Learnable kernel shape.** The one genuinely exclusive property — and measured inert. Over a
   full 20k-step pretraining run the kernels moved ~1% of their allowed range (σ 0.220 →
   0.217–0.223 within [0.05, 0.50]; gain 1.00 → 0.98–1.01 within [0, 2]); the multi-span arm's
   sealed-test win was produced with the kernels effectively frozen at Gabor initialisation.
   Detail in `2026-09-12-jepa-and-encoder-findings.md` and
   `halo-continuous-kernel-sweep-20260912`.

What would make the continuous arm worth its complexity is a convolutional stack on top of the
kernel responses, building a temporal hierarchy — the wav2vec/HuBERT shape. **That requires long
windows, and the supervised and sealed rosters cannot supply them.** Measured fraction of
continuous single-label executions long enough to supply one window (execution = the leakage
unit, the hard ceiling on window length):

| role | ≥ 6 s | ≥ 8 s | ≥ 12 s | ≥ 16 s | ≥ 30 s |
|---|---:|---:|---:|---:|---:|
| supervised head (8 sources, pooled) | 0.99 | **0.83** | 0.61 | 0.52 | 0.12 |
| sealed test (6 sources, pooled) | 1.00 | **0.96** | 0.96 | 0.90 | 0.71 |

Worst cases at 16 s: `xrf_v2` **0.00** (max execution 15 s), `hhar` 0.52, `kuhar` 0.56,
`harmes` 0.75; sealed `usc_had` 0.83. So the window lengths a deep convolutional stack would need
are exactly the lengths that cost us sources. On 8–16 s of signal there is very little for a CNN
hierarchy to compose that attention across patches does not already reach. Alex's framing: the
continuous kernel needs a CNN; we have no headroom for one where it matters.

## Why 4 s and 8 s patches, quantitatively

This is the strongest single argument, and it is about honest frequency resolution. A window of
length *T* cannot resolve frequency structure finer than ≈1/T (the time–frequency uncertainty
relation, Gabor 1946). The fixed bank **declares 32 log-spaced bands from 0.3 to 15 Hz**, but at
the current patch lengths most of the low end is never populated. Bands receiving at least two
full cycles within the patch:

| patch | 1 cycle needs | bands with ≥1 cycle | 2 cycles needs | **bands with ≥2 cycles** |
|---:|---:|---:|---:|---:|
| 0.5 s | 2.00 Hz | 16/32 | 4.00 Hz | **11/32** |
| 1.0 s | 1.00 Hz | 22/32 | 2.00 Hz | **16/32** |
| 1.5 s | 0.67 Hz | 25/32 | 1.33 Hz | **20/32** |
| 2.0 s | 0.50 Hz | 27/32 | 1.00 Hz | **22/32** |
| **4.0 s** | 0.25 Hz | 32/32 | 0.50 Hz | **27/32** |
| **8.0 s** | 0.12 Hz | 32/32 | 0.25 Hz | **32/32** |

**Twelve of the thirty-two declared bands (0.30–1.20 Hz) never reach two cycles at any patch
length currently in the ladder.** The code is honest about this — `use_resolution_mask` /
`FB_RESOLUTION_MIN_CYCLES` flag such bands as "present-but-blurry" rather than pretending — but
a third of the bank is being carried as blurry. And that band is not incidental to the task: human
stride frequency is roughly 0.5–1.1 Hz, arm swing 0.5–1 Hz, and sit/stand and postural transitions
sit below that. The proposed 4 s and 8 s patches are what make those bands real measurements
rather than flagged approximations.

**The token cost is essentially zero**, because long patches contribute few tokens. Per sensor in
an 8 s window:

| ladder | tokens per resolution | total |
|---|---|---:|
| current 0.5/1/1.5 | 16, 8, 6 | 30 |
| planned 0.5/1/2 | 16, 8, 4 | 28 |
| **0.5/1/2/4/8** | 16, 8, 4, 2, 1 | **31** |

So the full ladder costs one extra token over today's and three over the planned one, while adding
two octaves of frequency resolution.

## Window length: 8 s, which also unifies the two corpus contracts

4 s and 8 s patches require a window of at least 8 s; the labelled grids are currently 6 s. At 8 s
the cost is 17% of head executions and 4% of sealed executions (table above) — affordable, and far
cheaper than the 16 s+ a convolutional hierarchy would want. It also has a clean side effect: the
label-free pretraining corpus is **already** at 8 s (`PRETRAIN_WINDOW_SECONDS = 8.0`), so moving
the labelled grids from 6 s to 8 s collapses the current dual-contract arrangement
(`validate_source_window_contract` enforcing 6 s labelled / 8 s label-free) into one window
length end to end.

## What the literature supports here

- **A fixed, engineered spectral frontend is not a handicap at the frontier.** Whisper uses a
  completely non-learnable 80-channel log-mel frontend and places all learnable depth after it;
  the Audio Spectrogram Transformer likewise takes a fixed 128-bin mel spectrogram and applies a
  *linear* embedding to 16×16 time–frequency patches before its transformer — architecturally the
  same shape as our fixed bank plus attention.
- **Learnable filterbanks frequently do not learn, and this is a known, named phenomenon.**
  Anderson, Kinnunen & Harte, *"Learnable Frontends that do not Learn: Quantifying Sensitivity to
  Filterbank Initialisation"* (ICASSP 2023): *"a recurring finding reported independently in
  learnable filterbank studies is that the learned filters do not differ substantially from their
  initialised values… We feel this is more likely an optimisation problem."* Their conclusion:
  the lack of movement *"demonstrates a shortcoming in the overall optimisation strategy."* They
  also found **initialisation mattered more than learning** — a linear initialisation beat mel and
  bark on both of their tasks. The follow-up work is titled *"EfficientLEAF: A Faster LEarnable
  Audio Frontend of Questionable Use"* (Schlüter & Gutenbrunner, 2022). Our measured 1% kernel
  movement is this phenomenon, not a local bug.
- **Fixed/random feature banks are strong baselines for time series specifically.** ROCKET
  (Dempster et al., *DMKD* 2020) reaches state of the art on the UCR archive using *random*
  convolutional kernels plus a linear classifier; breadth of features, not learned filters, is
  what carries time-series classification. Consistent with our own measurement that a 50-dimension
  hand-crafted spectral-statistics vector reaches 0.82/0.78/0.70/0.61/0.56 subject-disjoint 5-NN
  balanced accuracy on hhar/kuhar/realdisp/dsads/wisdm — above a random-initialised encoder on
  every stream.
- **Caveat in the other direction, recorded honestly:** BenchHAR (arXiv 2605.08296, cross-dataset
  sensor HAR, 8 SSL methods × 12 architectures) reports that *CNN encoders exhibit the strongest
  ability to learn generalisable sensor representations*. That is an argument for convolutional
  depth somewhere in the stack, and it is not answered by this decision — only deferred, on the
  grounds that the window lengths that make it pay are not available in our supervised and sealed
  rosters.

## Consequences and open items

- **`DFT_SIZE` must rise, or long patches must be decimated.** The filterbank zero-pads every
  patch to one shared rDFT length `S` and fails loudly if `rate × patch_seconds > S`
  (`filterbank.py:319-334`). An 8 s patch at 240 Hz is 1,920 samples against the current
  `S = 512`. Two routes: raise `S` to 2048 (simple, but every *short* patch is then zero-padded
  4× further, so frontend cost rises across the board), or anti-alias-decimate long patches to a
  fixed analysis rate before the rDFT — the bank only measures to 15 Hz, so ~40 Hz suffices and
  an 8 s patch becomes ~320 samples, fitting the existing `S = 512`. The second is the principled
  option (DFT length should be set by `f_max`, not by source rate) but changes the rate contract
  and must be re-verified against the cross-rate invariance tests.
- **Long patches are degenerate under intra-window future-JEPA.** With an 8 s window and an 8 s
  patch, that patch spans the whole window and is therefore neither pure context nor pure target
  for a boundary at 0.4–0.7 of the window; 4 s patches straddle the boundary much of the time.
  The workable arrangement is that JEPA uses the short subset of the ladder and the long patches
  serve the supervised path — acceptable given JEPA's measured contribution, but it should be
  explicit rather than discovered at runtime.
- **The multi-span arm's remaining measured advantage is token density, not kernels** (117 tokens
  per 8 s window versus the fixed bank's ~30). If the extended fixed ladder still trails it after
  these changes, the difference to test next is an *overlapping* patch grid for the fixed bank,
  not a return to learnable kernels.
- Grids must be rebuilt at 8 s to a separate `--out-root`; the grid path does not encode window
  length and a rebuild would otherwise overwrite the corpus an in-flight run depends on.

**Related:** `docs/design/FRONTEND_PLAN_20260912.md` (implementation plan; X0/X6/X9 re-prioritised
by this decision), `docs/design/CONTINUOUS_KERNEL_FRONTEND.md`,
`2026-09-12-jepa-and-encoder-findings.md`, `2026-09-12-long-window-proposal.md`.
