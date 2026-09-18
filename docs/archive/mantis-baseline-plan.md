# Plan: add Mantis as a frozen (and briefly fine-tuned) encoder baseline

**Written 2026-09-13.** Implementation brief for another agent. "Implement" here means
**build + tests + short smoke**. Do **not** run sealed evaluation, and do not launch any long
training run, without Alex's explicit go. See `docs/design/EVALUATION_PROTOCOL.md`.

## Why

Mantis is a lightweight time-series-classification foundation model (8M params) that ZARA
(ACL 2026) used as its retrieval embedder, where it beat DTW retrieval by ~10 accuracy points.
It is open-weighted, pip-installable, fine-tunable on one GPU, and much cheaper to add than the
baselines we already carry. It is also the clearest architectural foil to our fixed filterbank:
it interpolates every input to 512 samples and instance-normalises each channel, i.e. it discards
exactly the two things (physical Hz and gravity/DC) our design is built to preserve.

Reference: V. Feofanov, S. Wen, M. Alonso, R. Ilbert, H. Guo, M. Tiomoko, L. Pan, J. Zhang and
I. Redko, "Mantis: Lightweight calibrated foundation model for user-friendly time series
classification," [arXiv:2502.15637](https://arxiv.org/abs/2502.15637), 2025.
Code: [github.com/vfeofanov/mantis](https://github.com/vfeofanov/mantis) ·
weights: `paris-noah/Mantis-8M` on HuggingFace · package: `mantis-tsfm`.

## What Mantis does (verified from the paper — re-verify against the code before relying on it)

* Inputs are **interpolated to a fixed length of 512** and **instance-standardised per channel**
  (subtract mean, divide by std over time), both inside the forward pass.
* Token generator emits **32 tokens of dim 256**, concatenating three streams: (i) normalised
  signal → 1D conv → mean pool; (ii) first difference → same; (iii) raw per-patch mean/std via a
  Multi-Scaled Scalar Encoder.
* ViT: class token prepended, sinusoidal positions, **6 layers × 8 heads**, d=256. Documented
  output is the **class-token embedding** from the last layer.
* **Univariate.** Pre-trained on univariate series and "applied to multivariate settings by
  treating channels independently." Channel adapters (LComb etc.) exist but are an appendix
  add-on and require training.
* Pre-training: contrastive (InfoNCE, temperature 0.1), single augmentation RandomCropResize
  (0–20% crop). Corpus is a union of UCR, UEA, ECG, EMG, Epilepsy, FD-A/FD-B, Gesture,
  **HAR (Anguita et al. 2013 = UCI-HAR)**, SleepEEG.
  *Note a discrepancy to resolve:* arXiv v1 says 7M pre-training samples; the later ICML/OpenReview
  version says 1.89M. Record whichever the released checkpoint's own card states.

## Step 0 — provenance and contamination (do this first, it can kill the task)

1. Fetch `paris-noah/Mantis-8M`, record the checksum, and mirror it following the existing
   convention in `docs/v2/` for baseline weights (see the other adapters; weights are gitignored).
2. Create `references/baselines/mantis/` with the paper PDF and `citation.json`, matching the
   layout of `references/baselines/unimts/` etc.
3. **Contamination check.** Read the pre-training corpus list in the paper's Appendix A.1 and
   cross-check it against our sealed roster: motionsense, realworld, shoaib, inclusivehar,
   usc_had, ut_complex. On the published list none of the six appears (the only HAR entry is
   UCI-HAR, which is in our *retired* roster, not our sealed one). **Verify this rather than
   trusting the sentence above**, write the finding into the adapter docstring, and wire any
   overlap into `is_incompatible()` so a contaminated cell is a disclosed `N/A`, not a silent
   number. If a sealed stream turns out to be in the corpus, stop and report — do not work around it.

## Step 0b — how Mantis interacts with the 4 s / 8 s / 16 s evaluation

Read alongside `docs/design/EVAL_EXPANSION_PLAN_20260913.md`, which sets the evaluation windows to
**N = 4, 8, 16 seconds**.

Mantis is the only baseline with a **fixed token budget rather than a fixed time budget**. It
interpolates every input to 512 samples and its token generator always emits exactly 32 patches, so
the window length changes the resolution of each token, not their number:

| N | effective rate after resizing | seconds per token | tokens |
|---:|---:|---:|---:|
| 4 s | 128 Hz | 0.125 s | 32 |
| 8 s | 64 Hz | 0.25 s | 32 |
| 16 s | 32 Hz | 0.5 s | 32 |

Consequences for the implementation and the write-up:

* **The C0 length probe will pass at every N.** Mantis needs no chunking, no padding and no crop at
  any window length — it is the most length-flexible model in the comparison. Feed it the full N
  seconds in one pass and record `padded: false` for every cell.
* **Its effective Nyquist at N=16 is 16 Hz**, barely above the 15 Hz upper edge of the motion band
  established in `docs/journal/2026-09-12-spectral-frontend-literature.md`. It is not losing signal
  at 16 s, but it has no headroom beyond it. Note this rather than discovering it later.
* **Longer windows are not "more evidence" for Mantis.** For every other model N=4 -> N=16 adds
  chunks, patches or tokens. For Mantis it is the same 32 tokens, each four times coarser. Mantis
  may therefore fail to improve — or regress — across the duration sweep while the others improve.
  **That is a result, not a bug.** Report Mantis's duration curve alongside the others and say
  explicitly that its token budget is fixed; do not "fix" it by feeding overlapping crops.
* **Pin the duration in `feature_config`.** Because Mantis reasons in cycles-per-window rather than
  Hz, its features are only comparable within a single N. Assert the incoming duration matches the
  declared N and stamp it into the feature fingerprint, so a 4 s cache can never be read for a 16 s
  cell.

## Step 1 — the adapter

Create `baselines/mantis/adapter.py` + `__init__.py`, registered as `mantis`. Model on
`baselines/harnet/adapter.py`, which is the closest reference (frozen released trunk, explicit
input contract, honest `is_incompatible`).

**Input contract.** `InputContract(channels=None, rate_hz=None, window_sec=None)` — Mantis accepts
native rate and length because it resamples internally. **But** it works in *cycles-per-window*,
not Hz, so comparability across streams requires a **single pinned window duration**. Assert the
incoming window duration equals the pinned value and record it in `feature_config`. Do not let
variable-duration windows through silently.

**The only thing that must be implemented is `window_features(stream, state, device) -> (N, D)`.**
`sealed_eval` already applies 1-NN, prototype, ridge and differentiable-neighbours to whatever
`window_features` returns, so implementing this one method yields the entire k-curve for free.
Subclass `BaselineAdapter` directly (not `ConSEAdapter`/`CosineAdapter` — Mantis has no text tower
and no native zero-shot path). `k=0` cells must be a declared `N/A` with a reason string.

**Per-window pipeline:**

1. Take the native window, `(T, C)`.
2. Run Mantis **per channel** (it is univariate), giving `(C, 32, 256)` patch tokens.
   The documented output is the class token only; the 32 patch tokens are the last layer's
   sequence. **Verify what the released API exposes.** If patch tokens are not reachable through
   the public API, take them via a forward hook on the final transformer layer rather than
   reimplementing the model. If neither works, fall back to the class token — `(C, 256)` — and say
   so loudly in the docstring, because it changes what the pooling step below can do.
3. **Pool tokens → one vector per window.** Two stages, both of which must be parameter-free in
   the frozen arm:
   * over the 32 token positions: concatenate mean and std (`(C, 512)`);
   * over channels: mean and std again, so the result is **invariant to channel count**
     (`(1024,)`). Do not concatenate per-channel vectors — our streams have different channel
     counts and orders, and a fixed concatenation would break on the first mismatch.
   L2-normalise the final vector. Record the exact pooling recipe in `feature_config` so the
   feature cache invalidates if it changes.
4. Return `(N, 1024)` float32.

**Two limitations to state in the docstring, not discover later.** Because instance normalisation
happens inside Mantis's forward pass, (a) gravity/DC is destroyed, so orientation and posture
information is unavailable to it, and (b) *relative* amplitude between channels is unavailable,
since each channel is normalised independently. Consequence: unlike harnet, Mantis needs **no
gravity guard** — gravity-removed sources such as KU-HAR are neither better nor worse for it.

## Step 2 — the fine-tuned arm

Add `--frontend`-style support for a `mantis` encoder in the support-classifier trainer so it can
be trained under the existing parameter-free differentiable-neighbours objective, exactly as the
`random_fixed_adapted` arm is. Reuse `training/support_classifier/neighbors.py`; add no new loss.

* **5,000 steps**, 500 warmup, BF16, same `--data-seed` as the existing adapted arms, training
  sources only. This is deliberately matched to the arms already in
  `training/support_classifier/evaluations/jepa_representation_20260912/` so the rows are directly
  comparable.
* In this arm the channel/token pooling **may** be learned — reuse `RecordingAttentionPool` from
  `model/tokenizer/encoder.py` rather than writing a new pooling module.
* Mantis is 8M params against our 2.7–3.2M trunk. Use a reduced encoder learning-rate multiplier
  (the trainer's existing 0.05 warm-start multiplier is the right starting point) and report the
  trainable-parameter count in the run config.

## Step 3 — what the finished work should produce

A table with these rows, on the six sealed streams at `k = 1…128`, macro-F1, slotting into the
existing ladder:

| arm | encoder updates |
|---|---|
| random frozen (exists) | none |
| JEPA frozen (exists) | none |
| **Mantis frozen** | none |
| random adapted 5k (exists) | yes |
| JEPA adapted 5k (exists) | yes |
| **Mantis adapted 5k** | yes |
| end-to-end 35k (exists) | yes |

The interesting comparison is not "does Mantis win" but **Mantis frozen vs our frozen arms** (does
8M params of cross-domain contrastive pre-training beat 32 engineered bands?) and **Mantis adapted
vs our adapted arms** (does the gap survive supervised adaptation, the way JEPA's did not?).

## Tests

1. Adapter registers; `REGISTRY["mantis"]` resolves.
2. `window_features` returns finite `(N, 1024)` on a small real batch from a **training** source.
3. **Channel-count invariance:** a 3-channel and a 6-channel stream both produce 1024-d output.
4. **Determinism:** two calls on identical input give bit-identical features; feature fingerprint
   is stable across processes.
5. **Window-duration assertion** fires on a mismatched duration rather than silently resampling.
6. Degenerate inputs (all-zero window, single channel, NaN-free guarantee) produce finite output.
7. `k=0` returns a declared `N/A` with a reason, not a crash and not a number.
8. Feature-cache round-trip: cached features reload and match.
9. Contamination guard: `is_incompatible` returns a reason for any dataset found in Step 0's
   corpus cross-check.

## Acceptance

* `pytest tests/` green (currently 726 passed, 1 skipped).
* The nine tests above passing.
* A short smoke: `window_features` on one training stream, plus a ≤50-step smoke of the fine-tuned
  arm. Report wall-clock and peak memory.
* **No sealed evaluation.** Produce the adapter, the trainer wiring, and the smoke; then stop and
  report, so Alex can decide whether to spend the sealed run.

## Notes for whoever picks this up

* Do not "fix" Mantis's instance normalisation or its 512-point interpolation to make it more like
  our frontend. The comparison is only meaningful if Mantis is run as published.
* `MantisPlus` / `MantisV2` checkpoints exist (pre-trained on the synthetic CauKer-2M set). Use
  **`Mantis-8M`** — it is the one ZARA evaluated. Note the others in the docstring as future work.
* If `mantis-tsfm` pulls a heavy or conflicting dependency set, vendor only the architecture module
  rather than taking the dependency; record which you did.
