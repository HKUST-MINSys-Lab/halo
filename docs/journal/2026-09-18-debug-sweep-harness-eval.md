# Debug sweep 3: training harness, curriculum, evaluation metrics and scenarios (2026-09-18)

**Report only. Nothing was fixed.** Third pass, after the fixing agent's 13 commits of 2026-09-16/17
(`080a87a` .. `a5e3e71`), the representative scenario run, the sealed baseline run and the four-arm
curriculum screen. Scope: `train.py`, `sampling.py`, `residual_classifier.py`, `pretrain_data.py`,
`augmentations.py`, the new `curriculum_audit.py` and `development_panel.py`, `run_scenarios.py`,
`scenarios.py`, `partial_coverage.py`, `run_partial_coverage.py`, `sealed_eval.py`,
`merge_sealed_results.py`, `baselines/scoring.py`, the result records under `docs/results/` and the
artifacts under `training/support_classifier/evaluations/`. The other agent's in-progress
MM-Fit/MobiAct integration (uncommitted) is only noted where it changes a protocol. Method: inline
read of every diff since `c480f50`, then probes on the real corpus, the stored artifacts and the
GPU. Suite: 939 passed, 1 skipped; one test failed during the sweep and passed on rerun because the
other agent's `deployment_policy` edit was mid-flight (§5).

## 0. Status of the second sweep's findings

| id | finding | status | evidence |
|---|---|---|---|
| N1 | stored manifests unreproducible | **fixed, verified** | NumPy draw and JSON fingerprint restored; rebuilding s1 MotionSense 8 s k=8 reproduces all 3,693 stored support rows |
| N2 | baseline fusion-only, 1-NN hidden | **half fixed, and the promoted run violates the half that was fixed** | `sealed_eval` now always emits 1-NN beside fusion (273 rows each per model in the sealed baseline run); `run_scenarios` still suppresses baseline 1-NN by default and the promoted scenario run was executed with `--readouts equal-weight-normalized-fusion halo-classifier`; see §1 |
| N3 | Stage B trained on the old sampler | disclosed, still promoted as "HALO" | the 2026-09-17 results record scores that checkpoint and says so in its limitations; the screen record says the same recipe must not be promoted; §2 |
| N4 | resume refused for older checkpoints | **fixed** | six missing trajectory keys now default from the saved args |
| N5, N6 | cross-placement was cross-dataset; string-inequality sites | **fixed** | `same_dataset=True` pools and `site_group` inequality; audit: 0 of 302 cross-placement sets contain another dataset |
| N7 | cross-dataset is body-IMU only | structural, now disclosed | audit: 3 datasets, 5 query labels; §3 |
| N8 | rate perturbation mostly upsampled | **fixed** | half of applicable draws now sample at or below the hardware rate; `padtype="line"` on both sides |
| N9 | asymmetric row sets | open | baselines still get only fusion in scenarios by default (§1) |
| N10 .. N14, C5, C13c | telemetry, gate crash, val sizing | **fixed** | conditional keys carry fractions; support perturbation reported as a fraction; replay seed no longer advances the episode RNG; `--val-repeats-per-dataset` is read; regime-split gate no longer copies `None` |
| C6 | gate reads post-residual logits | declared intentional | readiness note |

## 1. The headline scenario table compares HALO with the readout the protocol says must not stand alone

`docs/results/2026-09-17-scenario-evaluation-results.md` reports at k=8 that HALO leads all seven
scenarios by 9.6 to 26.2 points over "the strongest released baseline". That baseline row is the
fixed equal-weight fusion readout only. The same record's own readout-fairness table says fusion is
12 to 21 points *below* the baselines' cosine 1-NN in every complete-enrollment scenario at k=8,
and `EVALUATION_PROTOCOL.md` (commit `f94ce02`) says "External baselines always report cosine 1-NN
beside equal-weight normalized fusion". The promoted run does not contain a single baseline 1-NN
row: its argv is `--readouts equal-weight-normalized-fusion halo-classifier`, and the
`run_scenarios` default still hides baseline 1-NN (`support_readouts = frozenset()` when no
readouts are requested, with the comment "representation-only 1-NN is a HALO development
diagnostic, not a baseline headline row"). The code and the protocol text disagree, and the
artifact follows the code.

Recomputed from the tracked artifacts on identical manifests (0 manifest mismatches across runs),
8 s, split=all, primary cells, mean over variants, macro-F1 points:

| scenario | k | HALO classifier | best baseline, fusion | HALO minus fusion | best baseline, 1-NN | HALO minus 1-NN |
|---|---:|---:|---:|---:|---:|---:|
| partial enrollment | 8 | 48.6 | HARNet-10 38.4 | +10.2 | UniMTS 30.4 | +18.1 |
| cross placement | 8 | 44.9 | UniMTS 35.2 | +9.7 | UniMTS 48.1 | **-3.2** |
| cross dataset | 8 | 75.3 | HARNet-5 51.6 | +23.7 | UniMTS 65.4 | +9.9 |
| missing modality | 8 | 66.6 | UniMTS 42.6 | +24.0 | UniMTS 71.3 | **-4.7** |
| rate mismatch | 1 | 61.2 | UniMTS 43.3 | +17.9 | UniMTS 67.5 | **-6.3** |
| cross placement | 1 | 43.9 | HARNet-5 34.2 | +9.7 | UniMTS 42.9 | +1.0 |
| missing modality | 1 | 58.1 | UniMTS 42.5 | +15.5 | UniMTS 57.7 | +0.4 |
| cross placement | 32 | 57.1 | HARNet-5 45.3 | +11.8 | UniMTS 58.8 | **-1.7** |
| missing modality | 32 | 68.6 | HARNet-5 42.8 | +25.9 | UniMTS 74.8 | **-6.2** |

The 1-NN rows come from `scenarios_baselines_v3_20260917_final`, which shares the manifests but
stopped during rate mismatch at k=1 (its log ends there; it has no `run_metadata.json`), so rate
mismatch at k>1, new domain and device set have no 1-NN row anywhere on disk. Partial enrollment
is the one scenario where fusion has a structural advantage (hidden truths) and where the HALO lead
is robust to the choice of readout. Everywhere else the headline lead is mostly the fusion
penalty. The 364-row "readout fairness control" cited by the record is not in the repository;
its numbers are consistent with the recomputation above, but they cannot be audited.

What a reader will do with this: the record's summary sentence "HALO leads all seven retained
scenarios" will be quoted. Under the protocol's own mandatory companion readout it leads three of
the four scenarios that have a 1-NN row at k=8, and it trails on cross placement and missing
modality at every k where the row exists.

## 2. The curriculum screen decides between arms inside the panel's noise

The screen (`curriculum_ablation_screen_20260917.md`, journal
`2026-09-17-classifier-curriculum-screen.md`) trains four arms for 3,000 steps with one seed and
scores each on nine development panels of 64 support sets (136 to 207 queries, 2 to 7 datasets
per panel). It concludes that the adaptive gate alone is "the targeted winner" (+10.0 cross
dataset, +7.2 zero, +4.5 partial, +2.5 complete, -0.6 compatible, -2.8 cross placement) and that
the combined recipe "has a genuine interaction problem".

Measured this sweep: the corrected Stage A checkpoint re-scored on three panel seeds (the same
episode-drawing code, a different draw), dataset-macro F1 in points:

| panel | seed 0 (the screen's) | seed 1 | seed 2 | sd |
|---|---:|---:|---:|---:|
| clean mixed, enrolled | 65.7 | 56.4 | 53.9 | 6.2 |
| clean mixed, zero-shot | 55.0 | 43.7 | 62.1 | 9.3 |
| gyro dropout, enrolled | 53.1 | 49.1 | 48.5 | 2.5 |
| cross dataset | 70.4 | 82.5 | 83.6 | 7.3 |
| cross placement | 50.7 | 53.4 | 51.1 | 1.5 |
| partial enrollment | 51.7 | 58.2 | 54.1 | 3.3 |
| complete enrollment | 58.2 | 59.8 | 57.7 | 1.1 |
| zero enrollment | 49.6 | 48.5 | 49.3 | 0.6 |
| net gain over neighbour floor, clean | 11.9 | 5.8 | 4.3 | 4.0 |

The screen's own panel (seed 0) happens to be the lowest of three draws on cross dataset and the
highest on clean mixed. Every arm difference the screen interprets is smaller than or comparable
to the between-panel spread of a single checkpoint, except gyro-dropout robustness (+6.7 against a
spread of 2.5). Arms share a panel, so panel sampling is common-mode for the *difference*; the
paired per-panel deltas below (§2.1) are the right quantity, and they were not computed before the
decision was written.

Two smaller defects in the same record: "selected step 2,500 versus 3,000" is a two-point choice
(validation ran only at 2,500 and 3,000), so the selection rule adds a second noise source; and the
"rate_downsample_mixture" panel uses `rate_p=0.5` with the new rate rule, so half of its perturbed
recordings are upsampled.

### 2.1 Paired arm deltas over three panels

All four screen checkpoints re-scored on the three panel seeds; arm minus corrected Stage A on the
same panel, dataset-macro F1 points (mean of three paired deltas, sd of the three):

| panel | metric | gate only | perturbations only | perturbations + gate |
|---|---|---:|---:|---:|
| clean mixed | enrolled | +1.1 (3.9) | -5.0 (2.9) | -3.6 (2.1) |
| clean mixed | zero-shot | -2.5 (8.5) | -6.5 (5.1) | -1.8 (8.9) |
| gyro dropout | enrolled | -2.2 (3.4) | +1.1 (5.1) | +2.9 (2.5) |
| gyro dropout | zero-shot | +0.4 (7.4) | **+9.7 (4.9)** | **+12.2 (7.7)** |
| compatible | enrolled | +2.1 (2.4) | -0.7 (7.2) | -1.4 (9.3) |
| cross placement | enrolled | **-3.3 (0.4)** | -2.3 (4.4) | -4.9 (4.2) |
| cross dataset | enrolled | +3.3 (5.8) | -0.5 (1.1) | -2.9 (3.7) |
| complete enrolment | enrolled | +3.5 (2.9) | -0.2 (4.2) | +2.7 (3.4) |
| partial enrolment | enrolled | **+3.9 (1.9)** | +2.3 (2.2) | +1.9 (1.3) |
| zero enrolment | zero-shot | +3.7 (3.3) | -2.6 (3.8) | -0.6 (0.8) |

What survives replication: the gate's partial-enrolment gain (about +4, consistent across panels),
its cross-placement loss (about -3, tight), and a gyroscope-dropout gain for the perturbation arms
that lives in the *zero-shot* half of the panel (+10 to +12), not in the enrolled half the screen
record reports (+6.7 came from seed 0 alone; over three panels it is +1.1 with sd 5). What does not
survive: the gate's headline cross-dataset gain (+10.0 on the screen's panel, +0.4 and -0.6 on
the other two), its zero-enrolment gain (+7.4, +2.6, +1.1), the perturbation arm's clean cost as a
precise number, and any claim that the combined arm "overturns more correct neighbours" (its net
gain versus Stage A is -3.2 with sd 5.0). The screen's ranking of the four arms is not established
by the screen. Panel replication should be part of `development_panel` before any decision is
recorded, and the decision paragraph of the screen record should be reworded to what the
replicated numbers support.

## 3. Curriculum: what the corrected sampler delivers

`curriculum_audit_20260917` (2,000 sets, train split): enrolled shares 0.748 / 0.151 / 0.101
against the 0.50 / 0.25 / 0.25 default, fallback 0.225, zero enrollment 0.343 against 0.25.
Cross placement is feasible for dsads, forth_trace, realdisp and xrf_v2 only; cross dataset for
dsads, forth_trace and xrf_v2 only; no phone or watch dataset has either. The mechanism is the
fallback rule in `draw_batch`: when a (regime, mode) pair is infeasible for the scheduled dataset,
the draw falls back to a feasible pair weighted by the product weights, and the zero-enrollment
pairs of the infeasible modes are always feasible (they resolve to compatible). For a phone or
watch dataset the feasible mass is compatible 0.25/0.125/0.125 plus two zero pairs of 0.0625, so
those four datasets get 40% zero-shot and never a mismatch. Per-dataset measurement in §3.1.

The validation split is worse: the screen's mixed validation panel realized cross placement 0.023,
cross dataset 0.25, zero 0.31 (from the screen JSON's sampler telemetry). Checkpoint selection
uses `validation/selection_dataset_macro_f1`, the enrolled dataset-macro F1 over that panel, so
selection is almost blind to cross placement and sees cross dataset at more than twice its
training share.

### 3.1 Per-dataset realized shares, train and validation splits

Probe: 100 batches of 4 support sets with the trainer's exact `draw_kwargs` and `episode_rng`,
shares of support sets per query dataset (acquisition regime; `n/a` is the zero-support share).

| dataset | split | n | fallback | compatible | cross placement | cross dataset | zero support |
|---|---|---:|---:|---:|---:|---:|---:|
| dsads | train | 60 | 0.00 | 0.33 | 0.20 | 0.18 | 0.28 |
| forth_trace | train | 49 | 0.00 | 0.33 | 0.24 | 0.18 | 0.24 |
| xrf_v2 | train | 51 | 0.00 | 0.39 | 0.10 | 0.20 | 0.31 |
| realdisp | train | 45 | 0.18 | 0.33 | 0.24 | - | 0.42 |
| harmes | train | 53 | 0.32 | 0.68 | - | - | 0.32 |
| hhar | train | 48 | 0.40 | 0.56 | - | - | 0.44 |
| kuhar | train | 44 | 0.41 | 0.61 | - | - | 0.39 |
| wisdm | train | 50 | 0.36 | 0.68 | - | - | 0.32 |
| dsads | val | 67 | 0.16 | 0.39 | - | 0.24 | 0.37 |
| forth_trace | val | 53 | 0.21 | 0.45 | - | 0.19 | 0.36 |
| xrf_v2 | val | 53 | 0.00 | 0.34 | 0.25 | 0.21 | 0.21 |
| realdisp | val | 53 | 0.23 | 0.43 | 0.30 | - | 0.26 |
| harmes | val | 61 | 0.36 | 0.62 | - | - | 0.38 |
| kuhar | val | 58 | 0.38 | 0.55 | - | - | 0.45 |
| wisdm | val | 55 | 0.33 | 0.60 | - | - | 0.40 |
| hhar | val | 0 | - | - | - | - | - |

Three facts the design document does not state. The curriculum is two curricula: three body-IMU
sources get roughly the designed mixture, and the four phone/watch sources get compatible-only
enrolment with 32 to 44 percent zero-support sets (hhar: 25 percent complete enrolment). The
validation split has no HHAR episodes at all, and cross placement in validation comes from
RealDisp and XRF only. The sealed evaluator's cross-placement, cross-dataset and device-set
scenarios are phone and watch cells; the training and validation mismatch curricula never
contain a phone or watch query.

## 4. Evaluator, metrics and artifacts

**E-a - P2. `merge_sealed_results` rejects every current sealed run.** It requires
`result_schema == "sealed-results-v2-20260916"`; `sealed_eval` writes `sealed-results-v3-20260916`
(and the uncommitted version adds `prospective-results-v1-20260918`). The completed sealed baseline
run `sealed_baselines_v2_20260917_final` (2,925 rows, 1-NN and fusion for all five baselines at
k=0..64 and 4/8/16 s, `complete: true`) therefore cannot be merged, which is presumably why
`docs/results/RESULTS.md` still shows the 2026-09-14 baseline rows.

**E-b - P2. Incomplete runs without completion markers.** `scenarios_baselines_v3_20260917_final`
(25,793 rows, stopped in s5 at k=1), `scenarios_baselines_v3_20260917_representative*` and
`scenarios_halo_classifier_v3_20260917_k0_1_4_8_32` all lack `run_metadata.json`; only the
`profiled_v2` HALO run is complete. The "final" name on an aborted run is a trap for the next
reader; the readiness note's rule that a run without metadata is not a result should be applied
to the directory names.

**E-c - P2. The promoted HALO checkpoint is the recipe the screen says not to promote.** The
results record scores `halo_fixed_mr_residual_v3_curriculum1234_40k_20260916/best_internal.pt`
(perturbations plus gate, stopped at 20k, old sampler). The screen record's decision is "do not
promote or extend the combined perturbation-plus-gate recipe". Both are dated 2026-09-17. Either
the scenario table is provisional and should say so in its title, or it should be re-scored with
the checkpoint the project intends to promote.

**E-d - P2. HALO's own floor is absent from the promoted artifact.** With
`--readouts equal-weight-normalized-fusion halo-classifier`, HALO has no 1-NN/prototype/ridge rows
in `scenarios_halo_classifier_v3_20260917_profiled_v2`, so the classifier-versus-floor question the
scenario experiment was built to answer (first sweep, §0) cannot be read from the tracked artifact.

**E-e - P3. Protocol tags drift.** `_run_provenance` says `deployment-scenarios-v4-20260917`;
`run_metadata.json` still says schema `deployment-scenarios-results-v3-20260916`; the v3 artifact's
`run_metadata` lists `s8_cold_start` among scenarios. Harmless, but the tag pair is the thing a
merge tool will key on next.

**E-f - P3. "Exact neighbour floor" in the comparison telemetry is the temperature-0.07 soft vote
argmax** (`neighbor_logits = base_logits`), not cosine 1-NN. The panel and record call it the
neighbour floor; the sealed tables' "1-NN" is a different rule. Not wrong, but the two must not be
compared numerically.

**E-g - landed during this sweep as `093d8e6`, protocol-relevant.** The active scenario roster
changed: `NEW_DOMAIN_CELLS` is now MM-Fit left wrist only (`spar` and `upper_limb_use` retired,
disclosed in `EVAL_DATASET_AND_TEXT_AUDIT_20260918.md`), MobiAct sits in a separate
`prospective` scope that cannot touch the sealed-six mean, s2 gains a published-split MM-Fit
cross-device block with severity `L=3`, and the MM-Fit composite in `MULTI_DEVICE_EVAL_CELLS` now
enters s7. The results schema string was bumped to `deployment-scenarios-results-v4-20260918`, but
`_run_provenance` still writes `deployment-scenarios-v4-20260917`, and `EVALUATION_PROTOCOL` still
says v4 differs from v3 only by retiring cold start. A run under this code will claim the same
protocol tag as the 2026-09-17 record while scoring a different s2, s6 and s7 roster, so the two
are not comparable cell-for-cell and the tag should say so. The five dataset/protocol test files
pass (62 tests). The transient failure earlier in the sweep came from this edit mid-flight.

## 5. Verified correct this sweep

- Manifests: 3,693 of 3,693 stored support rows reproduced for s1 MotionSense 8 s k=8; sealed and
  scenario runs on 2026-09-17 share manifests (0 mismatches over every scenario/variant/k).
- Sampler contract (audit + probes): 0 cross-placement sets with another dataset's support;
  cross-dataset purity 1.0; fallback and regimes stamped; the replay seed no longer advances the
  episode RNG; rate augmentation downsamples in half of the applicable draws with the same
  polyphase edge policy as evaluation.
- Trainer: resume of a pre-curriculum and a curriculum checkpoint reaches the `--steps` guard;
  gate telemetry keys survive validation aggregation; support perturbation is a fraction.
- Evaluator: one unloadable stream is recorded as a `load_stream` failure without erasing sibling
  cells; paired deltas keep completed pairs when a scenario is incomplete; the fusion rule refuses
  non-uniform support widths and mismatched score matrices; `zscore` degenerates to zero rather
  than amplified noise; the residual head is cached per checkpoint/configuration; the sealed
  baseline run carries both readouts for every model and k.
- Tests: 939 passed, 1 skipped.

## 6. Priority for whoever acts on this

1. §1 - re-score the promoted scenario run with baseline 1-NN (a cached re-run: features are on
   disk), rewrite the headline table with both readouts, and make `run_scenarios` follow the
   protocol text by emitting baseline 1-NN by default.
2. §2 - do not act on the screen's arm ranking until the paired per-panel deltas (§2.1) are read;
   add panel replication (three seeds) to `development_panel` as the default.
3. E-a - fix the schema string in `merge_sealed_results` and merge the completed sealed baseline
   run into `RESULTS.md`.
4. E-c - decide which checkpoint is "HALO" in the results record and say so in its title.
5. §3 - state the per-dataset delivered curriculum in `CLASSIFIER_CURRICULUM_EXPERIMENT` and
   either accept that phone/watch sources see no mismatch, or add a within-dataset phone/watch
   source that can supply one.
6. E-b, E-e, E-g - directory hygiene and protocol tags.

## 7. Addendum (2026-09-18, later): the acquisition-conditioning-v2 metadata pipeline

**Report only. Nothing was fixed.** The other agent's uncommitted change replaces the combined
sensor sentence with two branches: device/placement text only ("a phone located at the front
pocket", identical for a co-located accelerometer and gyroscope; neutral text "a device at an
unspecified placement") plus a `StructuredSensorConditioner` that embeds modality (2), gravity
state (4) and `log2(rate/50)` for the stored and effective rates, gated in parallel with the text
residual. Contract: `docs/design/ACQUISITION_CONDITIONING_CONTRACT.md`. Files: `sensor_tokens.py`,
`encoder.py`, `pretrain_data.py`, `augmentations.py`, `eval_transfer.py`, `encoding.py`,
`pretrain.py`. The tree was still being edited during this pass (one crash and one stale test I
hit were fixed by the other agent within the hour); every statement below was re-checked against
the file state at the end of the pass.

**M1 - P1. The legacy path does not reproduce legacy checkpoints; the contract says it does.**
`stream_sensor_texts` was rewritten for all callers, so a checkpoint reconstructed under
`combined-text-v1` (every checkpoint that exists: promoted classifier, Stage A, Stage B, the four
screens) is now fed the new sentences at evaluation and in the development panel. Measured on the
promoted checkpoint, MotionSense front pocket, 8 s, 3,693 windows, against its own cached feature
matrix from the 2026-09-17 run (same checkpoint hash, same source fingerprint, same cache key):

| text fed to the legacy checkpoint | max abs diff | mean cosine | min cosine | leave-one-out 1-NN agreement with cache |
|---|---:|---:|---:|---:|
| new v2 sentences (current code) | 2.66 | 0.908 | 0.430 | 38.5% |
| legacy sentences (reconstructed) | 0.015 | 1.000 | 1.000 | - |

The contract's compatibility section ("Checkpoints without this field ... reconstruct under the
historical combined-text-v1 contract ... The legacy path exists only for exact reproduction") is
therefore false as implemented: the weights are rebuilt correctly, the input is not. Because
`FEATURE_CACHE_SCHEMA` was not bumped and the text is not part of the cache key, a re-run that
partly hits the shared cache would mix old-text and new-text features for one checkpoint inside one
results file. The legacy renderer has to be kept and selected by `conditioning_schema`, or the
legacy path has to be declared unsupported and the cache schema bumped.

**M2 - P2 (design). The stored sampling rate is now an explicit learned input, and it is a dataset
identifier.** Training rates: dsads 25, wisdm 20, forth_trace 51.2, kuhar 100, the other four 50 Hz;
combined with the device/placement sentence, `(text, stored rate)` identifies each of the eight
training sources uniquely. The sealed cells are 50 Hz except USC-HAD and MM-Fit at 100 Hz. The
effective source rate is a physical quantity (bandwidth) and belongs here; the stored rate after
resampling carries no physical information beyond it, so its embedding is a shortcut channel by
construction, and the rate-mismatch scenario resamples the query to 20/25/100 Hz, exactly the
rates that name WISDM, DSADS and KU-HAR in training. The 2026-09-17 heterogeneity audit itself
asked for a metadata-only shortcut test and for the learned rate feature to be a separate ablation
from physical correctness; the implementation makes it the default. Recommendation: embed the
effective rate only, or keep both and run the audit's metadata-only prediction test before any
training run uses schema v2.

**M3 - P2 (provenance). Schema v2 is the silent default for every new encoder.**
`build_random_encoder` and `PretrainConfig` default to v2, so the corrected Stage A / Stage B
re-runs that §1-§3 call for would change the conditioning at the same time as the curriculum, which
the audit's own sequencing rule forbids ("do not change the text backend simultaneously"). The
smoke checkpoint written this pass records `conditioning_schema: acquisition-conditioning-v2` and
50,816 new parameters, so provenance is at least explicit; the comparison design is not.

**M4 - fixed during the pass, recorded for the log.** At 02:0x the working tree crashed the trainer
smoke in `_pad_sensor_rows` (composite recordings received a `(devices, N)` matrix instead of a
concatenated row for `sensor_modality`); by 02:12 `merge_device_items` used a new
`_merge_conditioning_metadata` that concatenates per device, and the smoke passed (three steps,
validation, checkpoint). One stale test that still expected the old sentences was updated in the
same interval. A second failure (`test_eval_encoding_builds_factored_text_from_actual_channel_mask`)
appeared once in a six-file batch and never again in three reruns; it is order-dependent and
unresolved, most likely module-level caches (`_GRAVITY_STATE_CACHE`, the frozen text encoder's
descriptor cache) shared across test files.

**M5 - P3. Modality is no longer recoverable from text, which degenerates one ablation.** With
identical sentences for accel and gyro, the Phase-A descriptor-retrieval head (off by default) can
no longer be asked to reconstruct which modality was masked; `descriptor_retrieval_loss` collapses
the duplicate rows, so the objective silently becomes "which device/placement", not "which sensor".
Fine while it stays off; the contract should say so.

**M6 - P3.** `sensor_rates_hz` is cast to the token dtype before `log2`, i.e. bfloat16 under
autocast; 51.2/50 survives, but the two rate features differ by less than bf16 resolution for
several training sources (50 vs 51.2 collapses). Compute the log-ratio in float32 and cast after.

**M7 - P3 (other agent's dataset work, touches this pipeline).** The MobiAct grid on disk is the
old phantom: unqualified `meta.json` with `rate_hz: 0.0` and no labels, and no 4/8/16 s grid
directories. `list_streams` now ignores unqualified metadata for those durations, so the
prospective scope will fail at load rather than feed a zero rate into the structured conditioner
(which rejects non-positive rates on CPU only). Worth knowing before the first prospective run.

**Verified correct.** Gravity augmentation updates the structured gravity enum and no longer
rewrites the sentence; gyro dropout removes the sensor row and compacts `sensor_id`; the text
dropout neutral string equals the neutral-mode string; padded sensor rows are neutralised before
embedding and masked by `sensor_present`; effective rate is `min(hardware, stored)` on the
training, single-stream evaluation and multi-device evaluation paths; per-device rates are
concatenated, not overwritten, for composites; legacy checkpoints still load strictly (no
`structured_conditioner` keys) and the schema is inferred from the state dict when the config
lacks it; `MatchedCorpusEncoder.forward` absorbs the new keyword arguments; the JEPA pretrain path
passes the three tensors to every encoder call; the trainer smoke on the current tree completes; full suite on the tree at the end of the pass: 945 passed, 1 skipped, 2 warnings in 63.73s (0:01:03).
