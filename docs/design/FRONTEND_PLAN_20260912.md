# Historical frontend and JEPA plan

> **Historical planning record.** Future-JEPA is retired from the active recipe. Preserve this
> document for its measurements and rationale, but do not treat its JEPA commands as active work.

> Hand-off and attribution document, 2026-09-12. Findings were measured on the tree at commit
> `d71d24f`; probes are in `training/tokenizer/diagnostics/frontend/`. X0-X6 are implemented in the
> current tree and verified by focused tests plus both trainer smokes. The E0-E7 experiment ladder
> remains unrun and no new sealed evaluation has been consumed.

> **STATUS UPDATE 2026-09-12 (read before starting).** Alex has narrowed the direction: the
> **fixed multiresolution filterbank is now the primary arm**, extended to a
> **0.5 / 1 / 2 / 4 / 8 s patch ladder on 8 s windows**; the continuous-kernel multi-span arm is
> demoted to an ablation/control. Rationale and measurements:
> `docs/journal/2026-09-12-fixed-filterbank-decision.md`. Re-prioritisation of this plan:
> **X0 is superseded** (spans become 0.5/1/2/4/8, window 6 s -> 8 s, and `DFT_SIZE` must be
> raised to 2048 or long patches decimated before the rDFT — see the journal entry);
> **X1-X5 still apply** to the continuous arm but only as far as is needed to keep it a fair
> ablation; **X6 (dense hop + conv stem) and X9 (long windows) are deferred**, since both pay off
> only at window lengths the supervised and sealed rosters cannot supply (measured: only 52% of
> head executions reach 16 s, and `xrf_v2` reaches 0%). The E0-E7 ladder in section 3 still
> stands as the attribution method; re-point its rows at the fixed arm.

## 0. Ground rules (do not skip)

1. **Two data roles only.** Every decision in this plan is made on the eight supervised training
   sources: the trainer's internal subject-held-out panel (`validation/enrolled_dataset_macro_f1`)
   and the subject-disjoint kNN probe (`knn_probe.py`). `training.support_classifier.sealed_eval`
   runs once per *final* arm, after every choice in this document is frozen. Do not add sealed
   rows while iterating.
2. **No training launches without Alex's go.** "Implement" = code + tests + `--smoke`. Each
   experiment row lists its budget so the go can be given per row.
3. **Every new knob is serialised** into the checkpoint `config` and read back by
   `training/tokenizer/eval_transfer.py::build_encoder`, with a round-trip test
   (`tests/test_eval_transfer_detailed.py` pattern). A knob that is not in the config does not
   exist.
4. **Rate contract is inviolable.** Nothing after the physical decomposition may see sample
   indices; everything on the frame grid is in seconds. Cross-rate correlation test
   (`tests/test_multispan_kernel.py`, > 0.97 at 20/25/50 vs 100 Hz) must keep passing.
5. **No statistics over time inside the frontend** (duration-fingerprint rule, see the
   GroupNorm note in `continuous_kernel.py:263-268`). Channel-only LayerNorm per frame is allowed.
6. Another agent is editing the JEPA objective concurrently. Do not revert unfamiliar edits in
   `future_jepa.py`/`pretrain.py`; note them.
7. Baseline to beat for any frontend claim: the 50-dim hand-crafted vector in `knn_probe.py`
   (subject-disjoint 5-NN BA hhar/kuhar/realdisp/dsads/wisdm = 0.82/0.78/0.70/0.61/0.56).

## 1. Findings (what is wrong, with evidence)

| # | finding | evidence | where |
|---|---|---|---|
| F1 | Kernels never trained: sigma 0.220→0.217-0.223 of an allowed [0.05, 0.50]; gain 1.00→0.98-1.01 of [0, 2]; shape shift mean 0.07 | JEPA run telemetry `frontend/*`; e2e neighbour run identical | `pretrain.py:238-239` (`frontend_lr_scale=0.1`, `frontend_reg_weight=1e-3`), `continuous_kernel.py:307` (anchor to init) |
| F2 | Gradients DO reach every kernel parameter (zero-frac 0.000) but are sign-inconsistent once the trunk is trained (batch-gradient cosine 0.08-0.13 vs 0.30 trunk); from random init sigma/gain are consistent (SNR>1 on 81-84% of elements, cosine 0.72-0.78). LR-bounded random walk explains F1 | `ck_grad_probe.py` | — |
| F3 | Frequency allocation: integer harmonics of 1/T give only 4 distinct centres below 2.5 Hz (0.67, 1.0, 1.33, 2.0) while 2/4/6/8 Hz are each triplicated; the single-span module is log-spaced (15/32 centres < 2.5 Hz) | `ck_response_probe.py`; design doc admits "intentional capacity cap" | `multispan_kernel.py:136-152` |
| F4 | `log1p(|z|)` is in its linear regime: |z| median 3e-4..7e-3, p99 ≤ 0.22, fraction > 0.5 is 0.000 → features are standardised *linear* magnitude; frozen `norm_sd` 0.006-0.093 then amplifies high-frequency kernels ×10-170 | `ck_response_probe.py` | `multispan_kernel.py:398`, `:472` |
| F5 | `gain_logit` is functionally redundant with the projection (×2 gain → 0.03% non-affine feature change) → inert parameter | `ck_response_probe.py` | `continuous_kernel.py:300`, `multispan_kernel.py:195` |
| F6 | Quadrature (sin) coefficients get almost no consistent gradient in either state (phase-invariant |z| readout; one inert rotation direction per kernel) | `ck_grad_probe.py` | `continuous_kernel.py:292` |
| F7 | No local temporal composition: token = one spectral column (21-36 magnitudes + extras) → one `Linear` → attention. No CNN exists in the multispan path (conv1/conv2 live only in the single-span class). Random-init encoder scores *below* the hand-crafted floor; frozen JEPA equals it | `knn_probe.py`; `multispan_kernel.py:90-198` | — |
| F8 | Hop too coarse / tokens redundant: `frames_per_span=4` with sigma 0.22T → FWHM/stride ≈ 2; groups emit 8/4/2.7 frames/s; the fixed arm emits ≤ 2/s | `stage_probe_v4.py` (within-window token cosine 0.95 at init) | `multispan_kernel.py:212-215` |
| F9 | Nyquist-mask vector appended per token is a source-rate fingerprint (20 Hz: 25/31 live, 25 Hz: 28/31, ≥ 50 Hz: 31/31) | `ck_response_probe.py` | `multispan_kernel.py:476-495` |
| F10 | Masked kernels read as 0 = corpus mean after standardisation, not "absent" | code read | `multispan_kernel.py:472-476` |
| F11 | amp/dc standardisation pooled across accelerometer (g) and gyroscope (rad/s) axes | `ck_response_probe.py` (amp_mu 0.34 shared) | `multispan_kernel.py:429-453` |
| F12 | Frozen per-kernel standardisation goes stale once kernels move (mostly absorbable by `proj`, but unmonitored) | code read | `multispan_kernel.py:472` |
| F13 | Spans (0.5, 1, 1.5) leave a 0.5 s tail token in 8 s windows (17% of resolution-2 targets, per the JEPA review) and give f_min 0.67 Hz | JEPA review | `pretrain.py:228`, `future_patch_durations` |

Checked and clean (no action): `index_copy_` assembly preserves gradient; sigma sigmoid not
saturated; float32 analysis under disabled autocast; sqrt-of-clamped magnitude cannot blow up;
geometry caches hold only constant bases; reflection edge handling benign for a magnitude readout.

Context from the sealed table (do not re-read sealed data to confirm): the multispan arm is the
best arm (ridge 57.2 / 72.7 / 77.6 / 80.3 at k = 1/8/32/128) **with the kernels effectively frozen
at Gabor init**. On current evidence the win comes from the token grid, not learned kernels. The
ladder in §3 exists to attribute that.

## 2. Fixes

Each fix: change → files → config/serialisation → tests → acceptance. Order matters; X1-X4 are
prerequisites for X6.

### X0 — Spans 0.5 / 1 / 2 s (both arms)

- **Change:** `MS_SPANS_S = (0.5, 1.0, 2.0)`; `PretrainConfig.multispan_durations` and
  `future_patch_durations` default to `(0.5, 1.0, 2.0)`; `train.py --resolutions` and `--spans`
  defaults likewise. 2 s satisfies `CK_T_MAX_S = 2.0` and makes `CK_F_MIN_HZ = 0.5` real (first
  harmonic of the 2 s span). 8 s and 6 s windows tile exactly (4 and 3 tokens), removing the
  1.5 s tail-token defect.
- **Files:** `model/tokenizer/multispan_kernel.py:64` (`MS_SPANS_S`),
  `training/tokenizer/pretrain.py:227` (`future_patch_durations`), `:228`
  (`multispan_durations`), `training/support_classifier/train.py:858` (`--spans` /
  `--resolutions` defaults), and in `training/tokenizer/pretrain_data.py`: `:62`
  `LONG_PATCH_SECONDS_CHOICES` (extend to 2.0), `:67` `VAL_RESOLUTION_PAIR = (0.5, 1.5)` →
  `(0.5, 2.0)`, and the `DFT_SIZE` justification comment at `:72-76` (240 Hz × 2 s = 480 ≤ 512
  still holds; say so).
- **Check:** `dft_size=512` still covers 240 Hz × 2 s = 480 samples (`filterbank.S`); the
  multispan taps per kernel become 480+4 at 240 Hz (einsum cost ≈ +33%; measure step time in the
  smoke). `min_resolution_ratio=1.75`: ratios 2.0 and 2.0 pass.
- **Tests:** update every test that asserts `(0.5, 1.0, 1.5)`; add a test that an 8 s and a 6 s
  window produce no partial token at any span.
- **Note:** checkpoints trained at 1.5 s are not comparable with the new grid; the sealed rows
  above were at 1.5 s. Record this in RESULTS.md when the time comes.

### X0b — Efficient five-resolution frontend (supersedes X0's sizing; measurements in `docs/journal/2026-09-12-frontend-efficiency-and-jepa-window.md`)

**Target:** patch ladder `0.5 / 1 / 2 / 4 / 8 s`, supervised window `8 s`, JEPA window `16 s`.
Do **not** implement this by raising the shared `DFT_SIZE` to 2048 — measured 5.05x today's rDFT
work. Two staged options, both cheaper than today:

1. **Per-resolution transform length (stage 1, 0.67x today, low risk).** `S_g = next_pow2(max_rate
   * T_g)` -> 128 / 256 / 512 / 1024 / 2048. No resampling, rate contract untouched. Patch groups
   are contiguous along the patch axis and patches are zero-padded at the end, so this can be done
   by slicing `patches[:, group_slice, :S_g, :]` before the rDFT without changing the collate
   contract. The `(B,P,S,C)` tensor still carries `S_max`, so memory is not yet recovered.
2. **Decimate to a fixed analysis rate (stage 2, 0.13x today).** The bank reads 0.3-15 Hz, so
   anti-alias-decimate each patch to `min(source_rate, 40)` Hz in the collate before the rDFT;
   then `S_g = next_pow2(40 * T_g)` -> 32 / 64 / 128 / 256 / 512 and the patch tensor shrinks with
   it. Required care: (a) the **Nyquist observability mask must still key off the ORIGINAL source
   rate**, not 40 Hz, or bands will be fabricated for low-rate sources; (b) never upsample
   (`min(source_rate, 40)`); (c) re-run the cross-rate invariance test — it should improve, since
   all sources land on a common analysis grid, and that improvement is the acceptance signal.

**Also update:** `future_horizon_bins_seconds` from `((0,1),(1,2),(2,3))` to approximately
`((0,2),(2,5),(5,10))` for the 16 s JEPA window, or the longer window buys nothing.
`PRETRAIN_WINDOW_SECONDS` 8.0 -> 16.0; labelled `WINDOW_SECONDS` 6.0 -> 8.0; rebuild grids to a
separate `--out-root`. Expect batch ~132 at 16 s under `MAX_BATCH_TOKENS=16384` (today 273).

**Do not** use a 32 s JEPA window without explicit sign-off: measured, `extrasensory_pretrain`
yields 9 windows at 32 s (vs 305k at 8 s, 108k at 16 s), i.e. the corpus silently becomes
Capture-24 + Nymeria only.

### X1 — Unpin the kernels

- **Change:** `PretrainConfig.frontend_lr_scale: 0.1 → 1.0`, `frontend_reg_weight: 1e-3 → 0.0`.
  Keep the anchor regulariser available (`continuous_kernel.py:307`) but off by default.
- **Add** `--freeze-kernels` (both trainers): puts `adaptation_parameters()` on
  `requires_grad=False`. Replace the guard at `train.py:919-920` (`frontend_lr_scale <= 0`
  errors) with: `frontend_lr_scale == 0` ⇒ freeze. Record `freeze_kernels` in the config.
- **Files:** `pretrain.py:238-239, 2128-2140`; `train.py:821-825, 919-920, 1169-1188`.
- **Acceptance:** after a 3k-step e2e run from random init (E1 below),
  `frontend/kernel_shape_shift_mean` > 0.2 and `kernel_sigma_min/max` leave the [0.216, 0.224]
  band. If they still do not move at lr scale 1.0, stop and report — the objective, not the
  optimiser, is then the constraint.

### X2 — Log-spaced centres within each span (with projected Gabor init)

- **Problem:** the kernel is a Fourier series on the harmonics m/T of its own span, so a centre
  that is not an integer harmonic cannot be one coefficient. Two options; implement (a), keep (b)
  as fallback.
- **(a) Projected init.** Choose per-span centres log-spaced on `[1/T, min(f_max, M/T)]`
  (counts per span configurable, default 7/12/12 kept), and initialise `cos/sin_coeff` by
  least-squares projection of the analytic Gabor at that centre (sigma 0.22T) onto the M
  harmonic cos/sin basis evaluated on the span. Store the chosen centres in the `centres` buffer;
  drop the assumption `carrier` is an integer (it is only used at init and in `masks()` — see
  below).
- **(b) Harmonic subset.** Keep integer harmonics but pick a log-like subset per span and drop
  cross-span duplicates, e.g. 2 s: m∈{1,2,3,4,6,8,11}; 1 s: {1,2,3,4,6,8,10,12}; 0.5 s:
  {1,2,3,4,5,7}. Exact, zero new code paths, 21 kernels.
- **Observability mask (`masks()`, `continuous_kernel.py:461`)** already uses per-coefficient
  energy, so a projected (multi-coefficient) kernel gets a fractional mask automatically. Keep.
- **Config:** `centre_spacing: "harmonic" | "log"` (+ `centres_per_span`), serialised; default
  `"log"` for new runs; `build_encoder` must reconstruct old checkpoints with `"harmonic"`.
- **Tests:** projected Gabor reconstructs ≥ 95% of the analytic kernel's energy for every centre
  at every span; at least 8 distinct centres < 2.5 Hz across the bank; cross-rate correlation
  test unchanged.

### X3 — Compression scale

- **Change:** `compressed = log1p(|z| / s_k)` with a per-kernel scale `s_k` fitted during the
  existing calibration pass (median |z| over the calibration batches, stored as a buffer next to
  `norm_mu/sd`), so log1p operates around 1 rather than at 1e-3. Refit `norm_mu/sd` on the new
  compressed values (the calibration loop already does this; just make sure `s_k` is finalised
  *before* the standardisation accumulators run — one extra pass or a two-stage finalize).
- **Also check the fixed filterbank** (`filterbank.py:91`, same `log1p`): run the regime check
  (`ck_response_probe.py` logic) on `PhysicalFilterbankTokenizer`; apply the same scale if its
  band energies are also ≪ 1.
- **Files:** `multispan_kernel.py:398, 429-453, 472`; `continuous_kernel.py:711-770`.
- **Config:** `compression_scale: "none" | "calibrated"`, serialised.
- **Tests:** fraction of compressed values with |z|/s_k > 0.5 is ≥ 0.3 on a calibration batch;
  standardised feature mean ≈ 0, sd ≈ 1 on the calibration set; old checkpoints load with
  `"none"`.
- **Fixed-filterbank check (completed):** the retained fixed frontend is not globally trapped in
  the same linear regime (`norm_mu` median 0.289, range 0.008-0.434 on the reference checkpoint),
  so calibrated compression was not added to that separate frontend.

### X4 — Remove the inert gain; tolerate the sin null direction

- **Change:** exclude `gain_logit` from `adaptation_parameters()` and freeze it at 0 (keep the
  tensor for state-dict compatibility). Document that the projection and `norm_sd` own scale.
  No change for `sin_coeff`: it becomes live once shapes turn asymmetric; just do not count it as
  capacity in the paper (effective ≈ 13 params/kernel at init).
- **Files:** `continuous_kernel.py:300-306`.

### X5 — Standardisation hygiene

- amp/dc `mu/sd` per channel type (accel vs gyro), not per group only
  (`multispan_kernel.py:429-453, 483-486`). Serialised buffers; old checkpoints map the shared
  value to both.
- Telemetry: `frontend/standardised_mean_abs` and `frontend/standardised_sd` per group every log
  step; alert if drift > 0.5 sd from the calibration set (that is the F12 staleness signal).
- F9/F10 are design choices: keep the mask features, but add one sentence to
  `docs/data/DATA_HETEROGENEITY.md` disclosing that the observability vector encodes source rate
  class. Optional experiment row E7 below tests dropping the vector.

### X6 — Dense hop + convolutional stem (the structural fix; both arms via one module)

**Shape (v1, keeps the three-stream token contract):**

- **Hop:** replace `frames_per_span` with `frame_rate_hz` (default 16) for every span group.
  8 s → 128 frames per group per sensor. `token_metadata`/`frame_counts`
  (`multispan_kernel.py:212-240`) and `_group_geometry` (`:242`) take the hop from this.
- **Per group, per sensor (weights shared across sensors; sensors never mixed):**
  1. entry: pointwise `Conv1d(F_g → 128)` + GELU (F_g = existing 38/58-dim frame feature);
  2. body: 3 × [depthwise `Conv1d(128, k=5, dilation 1/2/4, same)` → channel-LayerNorm → GELU
     → pointwise `Conv1d(128→128)` → residual]. Receptive field 29 frames ≈ 1.8 s;
  3. downsample: two mask-aware stride-2 stages (strided depthwise k=4) → 32 tokens/group at
     4 tokens/s;
  4. out: `Linear(128 → d_model)`.
  Share the body across the three groups (per-group entry projections only) unless E4 shows
  otherwise. ≈ 110k params shared, ≈ 285k unshared.
- **Token metadata after the stem:** `positions = (i + 0.5) × 0.25 s`; `durations` = kernel
  span (duration embedding keeps meaning "resolution identity"); `token_mask` = centre frame
  inside the recording (`frame_time < duration`); `resolution_ids` = group. Downsampling of the
  mask: a token is valid iff its centre frame is valid. `frontend_rope_min_period`
  (`pretrain.py:154`) → `2 × token_stride = 0.5 s`.
- **`stem: "none" | "conv"`** so dense hop without depth is runnable (E2).
- **Insertion point:** `project_grid` (`multispan_kernel.py:455-507`) — build `features`
  exactly as now, then `features → stem → tokens` instead of `self.proj[g](features)`.
- **JEPA leakage (critical):** `pretrain.py:2685-2691` zeroes *tokens* with `future_context_mask`.
  With a stem, zero the *frames* before the stem instead: expose a `frame_mask` argument on
  `token_grid`/`analyze_grid` that the student passes (frames with centre > boundary → zeroed
  features and marked invalid), and keep the token-level `student_patch_valid` for attention.
  Planner inputs (`positions ± durations/2`) stay as they are; the frame mask is what guarantees
  no future samples reach the student. Add a test: a 20-sigma spike in every future frame changes
  no valid context token (reuse `b7_multispan_probe.py` logic).
- **Calibration:** unchanged (statistics are on pre-stem magnitudes).
- **Physical decoder / planner:** unchanged; token count per window changes, so
  `future_tokens_per_window` (`pretrain.py:126`) and the validation at `:1650-1720` must be
  updated for the new grid.
- **Serialisation:** `frame_rate_hz`, `stem`, `stem_channels`, `stem_kernel`, `stem_dilations`,
  `stem_shared`, `token_stride_s` in the config; `build_encoder` (`eval_transfer.py:211-217`)
  reads them; round-trip test.
- **Tests:** (i) rate invariance of stem *output* tokens (20/25/50 vs 100 Hz correlation > 0.95);
  (ii) duration fingerprint: a 2 s and a 6 s window sharing their first 2 s give identical tokens
  for that span (< 1e-4); (iii) leakage test above; (iv) token count = 3 × ⌈duration × 4⌉;
  (v) mask propagation on partial windows.
- **Smoke:** `pretrain --smoke --frontend multispan`, `train --smoke --frontend multispan`,
  plus `--freeze-kernels` and `stem=none` variants.

### X7 — v2 (only if E4 wins): single frame grid with spans as channels

Concatenate the three groups' frame features (154 dims) at 16 frames/s → one stem → one token
stream at 4 tokens/s (32 tokens per 8 s). Sets `num_resolutions=1`; touches
`future_resolution_durations`, per-resolution pooling and `balanced_future_latent_loss`. Not in
v1 scope.

### X8 — Later: second-order (modulation) kernel layer

A continuous kernel bank applied to the 16 frames/s magnitude envelopes (scattering-style
|x∗ψ₁|∗ψ₂), capturing 0.5-4 Hz envelope modulation. Only meaningful after X6; frame rate ≥ 8/s
required. Design note only; no code in this plan.

## 3. Experiment ladder (decisions on training sources only)

Common setup for every row: same trunk (d_model 256, 3 layers, 8 heads, ffn 1024 — the JEPA-run
trunk, NOT `build_random_encoder`'s d128 which is a capacity confound), spans 0.5/1/2,
`training.support_classifier.train --classifier neighbors`, **5,000 steps from random init**
(supervised adaptation is the dominant lever in the sealed table; JEPA warm-start is a second
pass once the v2 objective is verified), `encoder_lr_scale 1.0`, two seeds for the reference
row to establish the noise floor before any comparison is called.

Metrics per row: `validation/enrolled_dataset_macro_f1` at step 5000 (predeclared; do not pick
`best_internal`), and `knn_probe.py` on the five training streams (frozen, subject-disjoint).
Report per-dataset numbers, not just the mean.

| row | frontend | hop | stem | kernels | what it isolates | est. cost |
|---|---|---|---|---|---|---|
| E0 | fixed multires (unchanged) | 1/patch | none | — | reference + noise floor (2 seeds) | 2 × ~15 min |
| E1 | multispan, X0-X4 applied | 4/span (as now) | none | learnable (lr 1.0) | does unpinning + log centres + compression move anything (F1-F4) | ~15 min |
| E1f | same | same | none | frozen | learnability alone | ~15 min |
| E2 | multispan | 16/s | none | frozen | hop density alone | ~20 min |
| E3 | multispan | 16/s | conv | frozen | depth on top of hop ≈ "fixed bank + depth" | ~25 min |
| E4 | multispan | 16/s | conv | learnable | the continuous arm proper | ~25 min |
| E5 | E4 warm-started from the v2 JEPA multispan checkpoint | — | — | — | does pretraining still add on top (compare to E4) | ~25 min |
| E6 | E3/E4 at 35k steps | — | — | — | budget-matched vs the legacy e2e 35k arm | ~2 h each |
| E7 (opt.) | E3 without the nyq feature vector | | | | does the rate fingerprint carry the result | ~25 min |

**Decision rules.**

- A component is promoted only if it beats the previous row by more than 2 × the E0 seed spread
  on the internal macro-F1 *and* does not lose on the kNN probe on more than one stream.
- E1 vs E1f decides whether "learnable kernels" appears in the paper at all. E3 vs E2 decides
  the stem. E4 vs E3 decides whether learnable kernels matter *given* depth. If E4 ≈ E3 the
  arm is renamed "dense-hop physical bank + stem" and the continuous kernels become an ablation.
- Acceptance for X1 is read from E1 telemetry (see X1).
- Only the rows that survive get a single `sealed_eval` run, together, after this document is
  updated with the frozen choices.

## 4. JEPA-side items to verify (owned by the other agent; do not re-implement)

Landed in `d71d24f`: time-to-boundary encoding on predictor memory; absolute position removed
from the query (log-horizon features); per-query loss weights; `latest_index` helper.
Still to confirm before E5: (i) target normalisation over *time* (data2vec instance-norm for
speech-like modalities) — `future_jepa.py:319` still layer-norms over features only; (ii) the
per-query weight is a motion weight (ssl-wearables: sample ∝ window std); (iii) the telemetry
controls (persistence, same-window-other-target, no-context) are logged; (iv) `stage_probe_v4.py`
on the v2 checkpoint shows shuffled-context Δ ≫ 0.002 and own-similarity *declining* with horizon.

## 5. Deliverables

1. Code + tests for X0-X6 (`pytest tests/` green; currently 744 passed, 1 skipped).
2. Smoke logs for both trainers with the new frontend (attach `run_config.json`).
3. Probe outputs in `training/tokenizer/diagnostics/frontend/out/` for E0-E4.
4. A results table in this file's §3 with per-dataset numbers and the seed spread, and the
   promoted configuration written into `docs/design/CONTINUOUS_KERNEL_FRONTEND.md`.
5. No sealed run until Alex signs off the frozen configuration.

## 6. X9 — Long analysis windows (gated follow-on, do not start before E3/E4)

Recorded 2026-09-12 from Alex's proposal; rationale and the feasibility measurement are in
`docs/journal/2026-09-12-long-window-proposal.md`. Summary for an implementer:

**Idea.** Run the continuous-kernel arm on a much longer window (30-60 s) so the kernels plus the
X6 conv stem have room to build a genuine temporal hierarchy; the output is a sequence of
sub-patch vectors along time, used directly as JEPA targets and fused for classification by
attention pooling over time *and* across span groups (extend `RecordingAttentionPool`,
`model/tokenizer/encoder.py`).

**Why it is gated behind X6.** The conv stem is what makes a long window useful; a long window
without it just adds tokens. Build the stem, show it earns its place at the current window length
(rows E2/E3/E4), then lengthen.

**Feasibility (measured, seconds per continuous single-label execution):** pretraining is fine —
capture24_pretrain median 93,600 s, nymeria_xsens 932 s; extrasensory_pretrain caps at 40 s
(median 16 s) and must be carried as short windows in mixed-duration batches, not excluded.
Supervised/sealed are **not** feasible at 60 s: xrf_v2 (max 15 s) and harmes (0% of executions
>= 60 s) are lost from head training, usc_had (8%) from the sealed test, motionsense halved.
Pretrain long / adapt short is therefore the intended shape, which the duration-agnostic encoder
already supports.

**Implementation notes.**
- Dilation schedule must keep early blocks at 1-3 s receptive field so the stack still fires on a
  6 s episode recording; only later blocks reach tens of seconds.
- `build_grids.py --window-seconds` exists but the grid path does not encode window length —
  **write to a separate `--out-root`**, never overwrite the 8 s label-free grids.
- `PRETRAIN_WINDOW_SECONDS` (`data/pretraining/corpus_plan.py:47`) and `WINDOW_SECONDS`
  (`data/scripts/build_grids.py:57`, `training/tokenizer/pretrain_data.py:56`) are enforced by
  `validate_source_window_contract` (`pretrain.py:165`): a long-window corpus is a new contract
  value, declared and serialised, not a silent override.
- Cost scales linearly in frames: ~7.5x the frames at 60 s vs 8 s, so batch drops roughly
  proportionally at constant memory.
- Optional cheaper variant worth pricing first: lengthen *episode* recordings 6 s -> ~20 s, which
  six of eight head sources and five of six sealed streams support without roster loss.
