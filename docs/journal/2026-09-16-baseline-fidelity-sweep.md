# Baseline fidelity sweep (2026-09-16)

**Report only. Nothing was fixed.** Question: are the four released baselines used the way their
authors published them, and represented fairly according to the abilities they publish? Method:
one independent reader per baseline comparing our adapter against the vendored upstream code and
paper, plus cross-cutting hardware probes on the sealed grids. Direction of every deviation is
stated as *favours* or *disadvantages* the baseline.

## 1. Cross-cutting probes (all four adapters, MotionSense front pocket)

### 1.1 Units in the sealed grids — verified consistent

| stream | rate | accel at rest (g) | gyro median / p95 (rad/s) |
|---|---:|---:|---|
| motionsense / front pocket | 50 | 1.006 | 0.55 / 4.09 |
| shoaib / right pocket | 50 | 1.014 | 1.30 / 4.21 |
| usc_had / hip | 100 | 0.950 | 0.33 / 2.51 |
| ut_complex / wrist | 50 | 1.009 | 0.34 / 3.51 |
| inclusivehar / waist | 50 | 0.992 | 0.46 / 2.60 |
| realworld / waist | 50 | 1.005 | no gyroscope |

Accelerometer is in g with gravity retained everywhere; gyroscope magnitudes are rad/s (a deg/s
stream would read ~57x larger), and the USC-HAD converter explicitly rescales its deg/s source.
This is the contract LiMU-BERT-X expects (g after the authors' /9.8, rad/s) and what UniMTS
(g → m/s² in the adapter) and HARNet (g) require.

### 1.2 Sampling-rate handling — same 8 s physical signal presented at 50, 100 and 25 Hz

| adapter | cos(50, 100 Hz) | 1-NN neighbour agreement (50 vs 100) | cos(50, 25 Hz) | agreement (50 vs 25) | feature spread (mean pairwise cos) |
|---|---:|---:|---:|---:|---:|
| UniMTS | 1.0000 | 1.000 | 0.9997 | 0.938 | 0.746 |
| LiMU-BERT-X | 1.0000 | 0.953 | 1.0000 | 0.969 | 0.813 |
| HARNet | 0.9993 | **0.859** | 0.9439 | **0.312** | 0.620 |
| NormWear | 0.9994 | **0.547** | 0.9961 | **0.125** | **0.987** |

UniMTS and LiMU-BERT-X are rate-invariant through their adapters. HARNet changes 14% of nearest
neighbours under a **lossless** 50→100 Hz change: its features are well spread (rank 59), so this
is genuine sensitivity to the anti-aliasing filter that `resample_poly` builds for a 3/10 versus
3/5 ratio, not numerical noise. NormWear flips 45% and 88%: its MSiTF vector is hyper-concentrated
(mean pairwise cosine 0.987, rank 11 in 2048 dims), so a rate change the size of fp16 rounding
reorders neighbours. Both are properties of the models on our resampling path rather than errors,
but they mean the rate-mismatch scenario measures resampler sensitivity for those two, and NormWear's
enrolled rows are unstable for a reason unrelated to activity (see the readout finding of 09-15).

### 1.3 Which part of a long window each adapter consumes (cosine to the untouched window)

| adapter | 16 s: tail 8–16 s noised | 16 s: tail 10–16 s noised | 16 s: head 0–5.5 s noised | 4 s: tail 2–4 s noised |
|---|---:|---:|---:|---:|
| HARNet | 0.886 | 0.926 | 0.894 | 0.771 |
| UniMTS | 0.806 | 0.855 | 0.843 | 0.882 |
| LiMU-BERT-X | 0.969 | 0.984 | 0.970 | 0.957 |
| NormWear | 0.993 | 0.991 | 0.995 | 0.987 |

Every adapter responds to every part of a 16 s window. For LiMU-BERT-X and NormWear that is the
documented chunk-and-average rule. For **HARNet and UniMTS it contradicts both their published
input contracts and our own docstrings**, which say centre-crop to 150 samples (5 s) and
wrap-pad/truncate to 200 frames (10 s) respectively. The code feeds the full window in one pass
(`harnet/adapter.py:212-235 _window_chunk_features`; `unimts/adapter.py:236-262`, comment: "One
full evaluation interval is therefore one forward input"). Both networks are fully convolutional
with global pooling, so they run; but a 16 s input is 3.2x HARNet's and 1.6x UniMTS's training
length, and a 4 s input is 0.8x and 0.4x.

### 1.4 Published window rule versus our adapter — measured effect (300 windows, LOSO 1-NN and native zero-shot)

| model, window | frames the adapter feeds | published frames | cos(ours, published) | zero-shot F1 ours / published | 1-NN F1 ours / published |
|---|---:|---:|---:|---|---|
| UniMTS, 4 s | 80 | 200 (wrap-pad) | 0.893 | **39.6 / 34.5** | 14.3 / 14.0 |
| UniMTS, 16 s | 320 | 200 (truncate) | 0.971 | **41.4 / 37.9** | **83.7 / 81.5** |
| HARNet, 4 s | 120 | 150 (wrap-pad) | 0.929 | — | 13.5 / 13.0 |
| HARNet, 16 s | 480 | 150 (centre crop) | 0.902 | — | **74.8 / 63.0** |

**Both deviations favour the baseline**: UniMTS gains 3.5–5.1 zero-shot F1 and HARNet gains 11.8
1-NN F1 at 16 s from being given the whole window. That is consistent with the fairness rule
"choose the setting most favourable to the baseline", and it is the same amount of evidence HALO
receives, so the 16 s column stays a like-for-like comparison. It is not, however, what the
artifacts and documents say we do:

- `unimts/adapter.py:163,171` report `input_samples = 200` and every UniMTS result row carries
  `padded: True` with `padded_fraction` 0.60 (4 s) / 0.22 (8 s) — a padding-to-200 that the
  feature path does not perform.
- `harnet/citation.json` and `harnet/adapter.py:15-17, 168-171` describe a centre crop to 150 and
  say "our eval grids are ≤6 s"; `docs/baselines/BASELINES.md:7` says "published resampling and
  crop contract". None is true under the 4/8/16 s protocol.
- The adapter's own ConSE head-fit (`harnet/adapter.py:_fit_head`, the `harnet5_conse_head*.pt`
  caches, `HARNET_CORPUS=matched`) is **dead code** on every current runner: the sealed k=0 row is
  the shared training-bank 1-NN → ConSE path in `sealed_eval.py`, gated so `adapter.setup` is never
  called for a `conse`-tier model. There is no train/eval mismatch, but there is a large apparatus
  that contributes to no reported number.

### 1.5 Multi-device handling — what the artifact records

| adapter | composite mode | mechanism |
|---|---|---|
| UniMTS | `native` | each device at its own SMPL joint in one graph (adapter refuses joint collisions) |
| NormWear | `native` | devices concatenated as extra channels into MSiTF |
| HARNet | `per-device-pooled` | mean of per-device features (`base.py:86-88`) |
| LiMU-BERT-X | `per-device-pooled` | same |
| HALO | `native` | learned recording pool |

Whether "native" is what each upstream trained to fuse is assessed per baseline in §2.

### 1.6 The k=0 training bank is built at 6 s for every bank model

`_build_training_reference_bank` calls `load_eval_stream(dataset, stream_id, alignment="native",
apply_quality_screen=True, candidate_labels=…)` with **no `window_seconds`**
(`sealed_eval.py:1024-1027`), so it falls through to the legacy unqualified 6 s grid — the only
one the eight bank sources have on disk — while queries are 4/8/16 s. This applies to HALO,
HARNet and LiMU-BERT alike. It matters most for HARNet, the most length-sensitive adapter (§1.2).

### 1.7 Precision — cleared

NormWear fp16 versus fp32: zero-shot argmin agreement 1.000, feature cosine 0.99998.

## 2. Per-baseline findings

Each subsection: what is verified faithful, then deviations with direction. "Reader" = the
independent per-baseline code reader; "probe" = my own measurement.

### 2.1 UniMTS

**Faithful.** Released checkpoint loads strict; accelerometer-only weights (`acc.data_bn` is 66 =
3×22, first conv `(192,3,1,1)`) and no gyroscope ever reaches it; g→m/s² ×9.80665 with gravity;
20 Hz; no per-window normalisation (the paper's "normalisation" is unit standardisation); tensor
layout `(N,3,T,22,1)`; joint indices decode UniMTS's re-indexed graph correctly; **multi-device is
its native multi-joint fusion**, exactly what upstream `load_custom_data` does for multi-IMU sets;
no test-time rotation augmentation on either side; cosine scoring ≡ upstream normalised dot.

**Deviations.**

- **U1 (P1, both directions, undisclosed).** The published window rule — resample to 20 Hz, wrap-pad
  short to 200 frames, truncate long to the first 200 (`UniMTS/data.py:222-233`, README: "only the
  first 10 seconds") — is defined (`PAD_LEN`) and reported but never applied; the adapter feeds the
  whole interval (§1.3–1.4). Probe: at 4 s our 80-frame input scores +5.1 zero-shot F1 over the
  published rule on MotionSense; at 16 s our 320 frames score +3.5 zero-shot / +2.2 1-NN. Reader's
  12-cell grid: prediction agreement with the published preprocessing is 0.50 on RealWorld 4 s and
  0.62 on MotionSense 16 s; mean effect −0.45 F1 (cell-dependent, ±8). Every UniMTS row carries
  `padded: True` and `input_samples: 200`, describing padding that does not happen.
- **U2 (P1, favours UniMTS on average, undisclosed).** Zero-shot label text is not the published
  protocol. Upstream tokenises one bare string per class (`data.py:240-242`); the GPT paraphrases
  enrich *pre-training* descriptions only. We average 8 CLIP embeddings — the bare label plus 7
  paraphrases from HALO's **own training-set** synonym/template tables ("heterogeneous device
  seated", "mobile sensing in a seated position"). Probe, 8 s zero-shot F1, ours vs bare label:
  MotionSense 35.6 / 34.6, UT-Complex 20.7 / **22.1**, Shoaib **40.9** / 30.7. Reader's 12-cell
  mean +3.0 in UniMTS's favour, −5 on UT-Complex 16 s. The sealed artifact records nothing about
  this; only `run_scenarios` writes `label_text_ensemble`.
- **U3 (P1, misstates a published ability).** `native_few_shot_adaptation = no` in the results
  table. The paper's headline settings are zero-shot, **few-shot fine-tuning** (k∈{1,2,3,5,10},
  "+16.3%") and full-shot; `finetune.py` ships it. Our frozen 1-NN is a protocol choice, but the
  column asserts the model has no native adaptation mechanism, which is false.
- **U4 (P2).** The stale `input_samples=200` is hashed into the feature-cache key, so fixing U1
  would not invalidate cached features. `citation.json` says the input is 6-dim acc+gyro; the
  released checkpoint is accel-only. Two UniMTS contracts coexist: `baseline_backbone.py`
  implements the published 200-frame rule exactly, the sealed adapter does not.
- **U5 (P3, measured inert).** Joint choices differ from upstream's own for RealWorld/Shoaib/
  MotionSense (side-mirrored, belt 9 vs 0, MotionSense one joint vs two); reader measured ≤ ±2 F1,
  ours slightly better on Shoaib belt. Resampler is polyphase, upstream is FFT `resample`.

### 2.2 NormWear

**Faithful.** L1 argmin over the 2048-d MSiTF vector, native query "What is the current
activity?" and answer template "This subject is presently {}." (both verbatim from
`zero_shot/sentence_template.py`); TinyLlama revision pinned; text batching identical; 65 Hz
resampling is *more* faithful than the released code (which only resamples above 256 Hz and would
otherwise misread the ricker scales); multi-device as extra channels is exactly the mechanism
MSiTF is trained and benchmarked on (their WESAD cell is a 10-channel concatenation); channel
order is non-load-bearing by their permutation invariance; fp16 cleared (§1.7).

**Deviations.**

- **N1 (P1, disadvantages, confirmed 09-15).** `window_features` returns the MSiTF text-alignment
  vector; upstream's downstream recipe pools backbone patch tokens. Worth +8 to +16 macro F1 on
  enrolled rows. **Correction to the 09-15 finding's fix:** the reader is right that NormWear is
  never compared across streams (it has no training bank), so upstream's `flatten(nvar×768)` is
  feasible within a cell and should be the default; channel-mean is only needed for a
  cross-stream use that does not exist.
- **N2 (P1, unknown sign, reader-measured).** The released zero-shot path runs
  `scipy.signal.cwt` (`optimized_cwt=False`); we force the torch re-implementation because SciPy
  removed `cwt`. The reader reimplemented SciPy's and found the finest scale row effectively dead
  and sign-flipped in the torch path (correlation −0.97 at scale 0.1, one of 13 frequency patch
  columns corrupted). I did not reproduce this; treat as reader-verified pending the probe it
  specifies (vendor the 12-line SciPy CWT, re-encode 512 windows, report flips).
- **N3 (P3, measured inert).** Our 6 s chunking with trailing edge-pad and per-chunk amplitude
  normalisation is an invention; upstream runs any length in one pass via `pos_adjust`. Probe at
  8 s, single pass vs chunked: feature cosine 0.993, zero-shot agreement 0.99, F1 identical. Not a
  live problem at 8 s; still worth replacing with the native single pass.
- **N4 (P2).** Preprocessing provenance is ambiguous on both sides: the paper lists detrend +
  Gaussian smoothing (σ=1.3 samples); upstream's `preproc_all`/`basic_preproc` implement that plus
  an "optional" amplitude normalisation, but **no released inference path calls them**. We apply
  detrend + amplitude normalisation and no smoothing — the step the paper omits, minus the step it
  names. Needs the 2×2 probe the reader specifies before either is claimed.
- **N5 (P2).** The 1,293.86 M parameter count is 85% frozen third-party TinyLlama (backbone 136.1 M
  + MSiTF 57.7 M); the paper reports no count. The 19x headline ratio against HARNet is an artifact
  of the LM choice; `MATCHED_CORPUS_BUDGET` already uses 136 M, so the repo contradicts itself.
- **N6 (P2).** Our zero-shot number is template-filled over an open vocabulary with macro F1; the
  paper's UCI-HAR 71.2 AUROC used hand-written per-class option sentences. Not comparable, not
  stated. NormWear is absent from `BASELINE_FAIRNESS_POLICY.md`; `citation.json` still says
  "6 real acc+gyro channels, 6 s windows" while the adapter sets `channels=None` and chunks.

### 2.3 LiMU-BERT-X

**Faithful, decisively.** Units: grids hold g and rad/s (§1.1), the authors' `/9.8` is not
reapplied, gyro is never rescaled — the reader additionally showed m/s² or deg/s inputs explode the
model's own reconstruction error. `_Backbone` matches the released layout module-for-module and
loads strict (55,446 parameters; `pos_embed` is `(20,72)`, so **20 steps is the checkpoint's own
contract**, not our assumption — the public repo's `seq_len=120` configs belong to the original
LIMU-BERT). Missing-gyro refusal is the *right* representation of a 6-axis model with a single
6-wide input projection; zero-filling would invent an ability. k=0 ConSE bridge is disclosed as ours
(`native_open_set_labels = no`). Multi-device per-device pooling is shared and disclosed; upstream
publishes no fusion mechanism.

**Deviations.**

- **L1 (P1, disadvantages).** The released model is a **10 Hz** model. The LiMU-BERT-X paper:
  "we reduced the IMU data sampling rate from 20 Hz to 10 Hz … we adopted a 10 Hz sampling rate for
  our implementation" and "the collected sensors are the accelerometer and gyroscope sampled at
  10 Hz". Our adapter feeds 20 Hz (`TARGET_HZ = 20.0`), i.e. each 20-step clip covers 1 s of the
  2 s the model was trained to see. The reader measured the checkpoint's own reconstruction head
  preferring 10 Hz and +0.3 to +1.6 LOSO 1-NN F1 at 10 Hz on three sealed streams; **my own
  reconstruction probe was mis-wired and is inconclusive**, so the rate claim rests on the paper
  text, which is unambiguous. `native_window_sec = 1.0` and the padding disclosure inherit the
  same error.
- **L2 (P1, disadvantages, and it is our data).** RealWorld ships gyroscope zips for every activity
  (16 `gyr_*` per subject on disk). Our converter's acc/gyro part pairing keeps gyroscope in **15
  of 132 sessions**, so every RealWorld grid is accelerometer-only. That is why LiMU-BERT-X is
  "excluded from all of RealWorld" (98 `unsupported` cells) and why the journal's "accelerometer-
  only device → LiMU-BERT cannot run" is demonstrated on a stream that is accelerometer-only because
  of our pipeline. It also silently withholds gyroscope from HALO and NormWear on RealWorld.
- **L3 (P2, wrong attribution).** Token-mean pooling over 20 steps and duration-weighted clip
  averaging are ours; upstream feeds the full token sequence to a GRU and takes the last state.
  Reader measured mean vs last-token vs concat as a wash (MotionSense 83.2/77.8/83.2), so the
  journal's "mean-pooling gives a smoothed, ambiguous descriptor" explains the k=1 failure by a
  property of *our* readout and is not supported.
- **L4 (P2).** The published few-label ability is joint fine-tuning (700 epochs, 1% labels); we
  never exercise it, and `base.py:make_probe`'s stated rationale ("moves LiMU-BERT toward its
  paper's classifier") is inert — no probe is fitted in the sealed path.
- **L5 (P2, retract).** The matched-corpus plan's note that the released checkpoint saw
  MotionSense/Shoaib in pretraining is wrong for *this* checkpoint: Phase-II pretraining is courier
  data only; Shoaib is an ablation corpus and an explicitly failed external-pretraining experiment.
  The note over-states a leakage advantage.
- **L6 (P3).** `released_config: "base_v4"` is unverifiable — no clone of the Experience repo is
  vendored, and the checkpoint matches neither the paper's stated widths nor its "137 thousand
  parameters". Resampling method unverifiable for the same reason.

### 2.4 HARNet (ssl-wearables)

**Faithful.** Pinned tag and entrypoint; frozen trunk with BatchNorm in eval, fp32, matching
upstream's own frozen protocol; raw g with gravity and no normalisation (as upstream's
`NormalDataset`); accelerometer-only by channel name; axis order is a non-issue because the authors
trained axis-swap/rotation invariance in; **the representation is bit-identical to upstream at
native length** (feature map `(N,512,1)` at 150 samples, so flatten ≡ our temporal mean; probe:
66.3 = 66.3 F1); no sealed data in the k=0 bank.

**Deviations.**

- **H1 (P1, disadvantages, disclosed only in citation.json).** We run **harnet5**; every published
  ability belongs to **harnet10** (paper: "ten-second-long windows", "feature vector of size 1024",
  README benchmarks all 10 s; 10.98 M vs 4.49 M parameters). The stated reason — harnet10 "does not
  fit our ≤6 s grids" — is stale and now false: probe shows harnet10's trunk runs natively at 300
  and 480 samples (errors at 150), so a 16 s window is a single-position, unpadded harnet10 input.
  `BASELINES.md`, `RESULTS.md` and the journal all say "HARNet".
- **H2 (P1, disadvantages every k=0 row).** The gravity guard is **dataset-scoped**: it reads
  `streams[0]` only (`adapter.py:304-316`). xrf_v2's first stream is the gravity-removed earbud, so
  all six xrf_v2 streams — five of which retain gravity, including both wrists — are excluded from
  HARNet's zero-shot bank. Confirmed: the bank cache holds 0 xrf_v2 HARNet files against 156 HALO
  files. That is ~17% of the bank, and the exclusion appears nowhere in the artifact (folded into a
  fingerprint hash).
- **H3 (P2, favours, undisclosed).** Full-window feeding (§1.3–1.4): +7.9 / +3.9 1-NN F1 at 8 s
  (MotionSense / Shoaib) and +11.8 at 16 s over the published 5 s crop. Every row still declares
  `input_samples: 150`; the padding disclosure uses a 4 s floor against a declared 5 s native
  length, so 4 s rows read `padded: False` below the contract; `BASELINES.md` says "crop
  contract"; the function that implements the crop has no callers.
- **H4 (P2).** The published ability is fine-tuning (full fine-tune .810 vs frozen+FC .759 vs
  scratch .684 on WISDM); we test a parameter-free readout the authors never published, weaker than
  even their frozen row, while the journal says HARNet "fails at short windows" and its pretraining
  "buys nothing visible".
- **H5 (P2).** REALWORLD is one of ssl-wearables' own downstream benchmarks and one of our sealed
  six; WISDM is in our bank. No label leakage (the released weights are SSL-only), but the policy
  asks for upstream-overlap disclosure and none exists.
- **H6 (P3).** Resampler: upstream is linear interpolation (the model was pretrained on linearly
  downsampled 100→30 Hz data); ours is polyphase; cos 0.971 between the two. No ±3 g clip (0.2% of
  samples exceed it). The guard is inert on RealWorld (`list_streams` defaults to 6 s and RealWorld
  has only w4/w8/w16) and fails open on missing grids. `HARNET_CORPUS=matched` renames the adapter
  out of the roster and silently drops the k=0 row. The journal's "4 s is shorter than its native
  input" mechanism is wrong: at 120 samples the trunk emits the same `(N,512,1)` map and no padding
  occurs; the 4→16 s gain is four averaged positions versus one.

## 3. What to do first (report only; nothing fixed)

1. **Data, not adapters:** repair RealWorld's gyroscope pairing (L2). It withholds a modality from
   three baselines and HALO on one of six sealed sources and manufactures a "cannot run" cell.
2. **HARNet:** per-stream gravity guard (H2) and a harnet10 arm for 16 s (H1) — both move reported
   numbers in HARNet's favour.
3. **LiMU-BERT-X:** 10 Hz (L1); retract the leakage note (L5).
4. **UniMTS:** report the published bare-label zero-shot row beside the E=8 row, and either apply
   the 200-frame rule or state that both models receive the full window (U1, U2); fix the few-shot
   column (U3).
5. **NormWear:** backbone features with upstream flatten (N1); resolve the CWT implementation (N2)
   and the preprocessing 2×2 (N4) before any NormWear number is cited.
6. **Disclosure everywhere:** `input_samples`, `padded`, bank exclusions, bank window length,
   parameter-count composition, upstream benchmark overlap.
