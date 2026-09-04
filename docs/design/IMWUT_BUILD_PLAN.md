# IMWUT build plan — from `main` to the Nov 1 2026 submission state

> ## ⚠️ HISTORICAL DOCUMENT
>
> This was written on 2026-09-03 and parts of it were superseded within days. It is kept because it
> records *how* the work was decomposed and what the sweep found, which is still useful. **It is not
> the design of record.**
>
> Known to be out of date here:
> - **`adaptation_v2` (10 datasets) is the current protocol**, not `adaptation_v1` (7).
> - **Warm start is no longer the plan**; end-to-end from random init is the default.
> - The week-by-week schedule was written before the build finished. The build is done; what remains
>   is the run budget, which has not been cut to fit.
>
> For the current design read [`IMWUT_COMPARE_DESIGN.md`](IMWUT_COMPARE_DESIGN.md); for orientation
> and standing conventions read [`../IMWUT_START_HERE.md`](../IMWUT_START_HERE.md).


**Status: historical implementation plan, 2026-09-03.** It is retained as an audit trail only;
current behavior is defined by `docs/design/IMWUT_COMPARE_DESIGN.md`. Venue read:
`docs/research/IMWUT_VENUE_READ.md`.

This document records (1) a read-only debug sweep of the training and evaluation code on `main`
as of `1c187f6`, (2) what is reusable as-is, (3) the model-side, data-side, evaluation-side and
baseline-side changes needed, and (4) a week-by-week schedule against Nov 1.

---

## 1. Debug sweep — what the code on `main` actually is

Read: `training/tokenizer/{pretrain,pretrain_episodic,episodic,pretrain_data}.py`,
`model/evidence/{engine,evidence_mixer}.py`, `model/tokenizer/encoder.py`,
`eval/{enrollment_protocol,run_adaptation_baselines,data}.py`, every `baselines/*/adapter.py`,
the `adaptation_v1` manifest, the result artifacts under `eval/adaptation_results/`, the checkpoints
under `training/tokenizer/outputs/`, and the archived 2026-08-22 harness audit. The classification-
era test suite (`tests/` minus `tests/applications`) passes in full.

### 1.1 Sound and reusable without change

| piece | state | note |
|---|---|---|
| **Evaluation manifest `adaptation_v1`** | frozen, fingerprinted, 61 cells, 7 datasets, k in {0,1,2,4,8,16}, 5 seeds | support and query are execution-disjoint; every enrollment cell is `same_configuration` (support stream == query stream), i.e. **config-compatible by construction**; both `cross_subject` and `same_subject` relations exist. This is exactly the Section-5 protocol of the design doc. |
| **Enrollment protocol + adaptation runner** | `eval/enrollment_protocol.py`, `eval/run_adaptation_baselines.py` | nearest / prototype / ridge / linear_head readouts on frozen features, plus a `native` path for models that bring their own mechanism. Leakage assertion at manifest build. |
| **Scoring path** | audited 2026-08-22, no correctness bug | ConSE bridge, CIs, ground-truth alignment all verified against upstream code. |
| **Released-checkpoint baselines** | harnet, UniMTS, ImageBind, NormWear adapters | contracts verified against upstream; checkpoints on disk; `label_text_ensemble=8` for UniMTS already applied (closes audit F2). |
| **Phase-A pretraining** | `pretrain.py`, JEPA + aug-VICReg, calibrate-once-at-2k | CLI default is already **single-resolution, sensor granularity, factored text, fixed frontend**. A 15k-step batch-1024 run took **26 min** on the 4090 at 5 GB. |
| **Text-OFF pathway** | `--neutral-acquisition-text` in the episodic trainer; `neutral=True` in `stream_channel_descriptions` | acquisition text is stripped; modality text survives (needed by sensor identity). This *is* Arm A's conditioning. |
| **Comparator core** | `EvidenceMixer` (set attention over candidate/query/evidence tokens, zero-init residual head) and `mixed_vote` (attention weights x rectified label-text cosine, enrolled-identity vote) | the design's readout equation already exists here. |
| **Training datasets** | 18 sources (12 primary + 6 direct-converted), grids on disk, sampler tempered by dataset/subject | disjoint from the 7 evaluation datasets. |
| **Compatibility key** | `SensorCompatibilityKey` (device family, placement, channels, gravity state; rate excluded) in `applications/motion_monitoring/data/compatibility.py` | defined exactly as the design wants; lives in the application package and must be lifted. |

### 1.2 Not sound for the new design — must change

| finding | severity | consequence |
|---|---|---|
| **The episodic trainer is bank-and-retrieval shaped.** `pretrain_episodic.py` requires `--bank-windows >= 1`, attaches a stratified corpus bank to every episode, computes retrieval scores, and feeds `top_k` evidence rows plus a retrieval-score attention bias into the mixer. There is no "support only, attend over all K" mode. | **blocking** | A new episode path is needed: support set = the K compatible examples, no corpus bank, no top-k, no retrieval bias. ~1,950 lines of trainer are entangled with the bank; write a narrow new trainer rather than thread flags through it. |
| **Support sampling has no compatibility filter.** `EpisodeSpec.disjointness` is `subject` or `stream`; neither is "same configuration family across datasets, subject-disjoint". `policy.py` states outright: "Subject identity and configuration never constrain support sampling." | **blocking** | Arm A needs a compatibility-keyed sampler; Arm B is the unfiltered variant. |
| **Phase-A checkpoints do not match the compact shape.** Every Phase-A `best.pt` on disk is d=256 / 6 layers / dual trunk. The compact engine is d=128 / temporal trunk, and the compact e2e checkpoint records `phase_a_checkpoint: None` — it was trained from random init. | **blocking for warm-start** | A Phase-A run at the compact shape is required (26 min). The old Phase-A checkpoints are not reusable for the headline model. |
| **Baseline results are not clean enough to publish.** The canonical baseline rows (`v1_d85761d`) predate the UniMTS ensemble fix (`label_text_ensemble: None`) and the UniMTS row was produced from a **dirty** git tree; the later `_shared` rerun has the ensemble but is also dirty. | **must rerun** | Re-run all four released-checkpoint baselines from a clean tree on the final protocol. Cost is small: harnet 31 s, UniMTS 50 s, ImageBind 48 s, NormWear 485 s per manifest pass, plus head fits. |
| **CrossHAR / LiMU-BERT** | policy | dropped: self-pretrained, not released. Their adapters stay in the repo but leave the table. |
| **The classification results docs were deleted from `main` by the pivot.** 13 docs (adaptation tables, step-0 control, plateau diagnosis, harness audit) and the figures exist only on `archive/pre-application-main-20260830`; `docs/results/RESULTS.md` on `main` now holds application-task tables. | **must restore** | Restore them under `docs/results/classification/` on the `imwut/compare` branch so the paper's numbers have a provenance trail. |
| **Zero-shot at k=0 in the manifest has no support rows** | fine | k=0 cells are scored by each model's native rule; for us that will be the text vote over an *empty* support set — which is degenerate. The design says ZS is disclosed secondary; we score it with the Arm-A comparator given a small config-compatible corpus draw that excludes the candidate labels (the previous "bank" idea, kept only for the ZS row). Decide before build (Section 3.3). |

### 1.3 Minor findings carried forward

- `EvidenceMixer` expects a `retrieval_score` bias; with no retrieval the bias input becomes a
  constant and `score_bias_gain` is dead. Remove the input in the new path rather than feed zeros.
- The compact adapter's native path builds a 512-window corpus bank keyed by checkpoint hash; the
  new adapter must not.
- `git_dirty: True` in result JSONs is recorded but not refused. The final runs should be made from
  a tagged, clean commit; the assembler should refuse dirty rows for the paper table.

---

## 2. What is reused vs built

| Reused unchanged | Reused with a narrow change | New |
|---|---|---|
| Phase-A pretraining recipe and CLI | `EvidenceMixer` (drop the retrieval-score bias input) | `SupportEpisodeSampler`: compatibility-keyed (Arm A) or unfiltered (Arm B), subject-disjoint, never-the-query, random label subset, GT present w.p. p |
| `adaptation_v1` manifest and enrollment protocol | `mixed_vote` (support-only rows; no corpus rows) | `train_compare.py`: warm-start encoder + comparator, joint ZS/FS episodes, soft ZS targets, fixed step budget, seeds |
| Scoring, CIs, assembler | `baselines/halo_compact` -> `baselines/halo_compare` (native path = support-only comparator) | `SensorCompatibilityKey` lifted to `data/scripts/curate/compatibility.py` and applied to every training stream via its `StreamSpec` |
| harnet / UniMTS / ImageBind / NormWear adapters | assembler refuses `git_dirty` rows for the paper table | Compatibility *distance* for Arm B2 (same device family + different placement, etc.) |
| Text-OFF (`neutral`) description path | | Cost-table script (params, encode latency for K support + query, memory, named device) |

---

## 3. Work items

### 3.1 Data side (wire-up, no new data)

1. Lift `SensorCompatibilityKey` out of the application package; derive a key for every training
   stream from `deployment_policy.StreamSpec` (device_profile -> family, placement, channel set,
   gravity_state). Audit the result by hand: 18 datasets, ~60 streams, print the family table.
2. Define the **compatibility distance** for Arm B2 (0 = same key; 1 = same family+placement,
   different channels/gravity; 2 = same family, different placement; 3 = different family). One
   table, checked in.
3. Confirm every training stream has verbatim label strings and execution ids (the sampler's
   never-the-query rule needs execution granularity, as the eval manifest already does).
4. No new grids. No dataset is added or removed for Nov 1.

### 3.2 Model side

1. **Phase-A pretrain at the compact shape**: d=128, temporal trunk, single-res, sensor
   granularity, factored text, `--neutral-acquisition-text` semantics applied at pretrain too for
   Arm A (so the warm-start encoder never saw acquisition text). One run, ~30 min. A second run with
   text ON for Arm B.
2. **Support-only comparator**: `EvidenceMixer` over `[candidates | query rows | K support rows +
   their labels]`, no retrieval bias; readout through `mixed_vote` with support rows only.
3. **`train_compare.py`**: loads the Phase-A checkpoint, draws episodes from
   `SupportEpisodeSampler`, loss = CE on FS episodes + soft-target CE on ZS episodes, fixed budget
   (35k steps), final-step checkpoint, seed as a CLI argument. Telemetry: encoder effective rank
   (the collapse watchdog), enrolled-vote vs text-vote mass share, per-episode positive share.
4. **Step-0 predictor**: the same checkpoint format at initialisation, for the paired control.

### 3.3 Evaluation side

1. `baselines/halo_compare/adapter.py`: `window_features` (pooled rows, for the frozen-feature
   readouts) and `native` enrollment = support-only comparator over the manifest's support rows.
2. Zero-shot row: decide one of (a) comparator over a small config-compatible corpus draw that
   excludes candidate labels, or (b) report ZS via ConSE on our frozen features like the closed-set
   baselines. (a) is the deployed mechanism and the design's intent; (b) is one line of code. Do
   (a), disclose it.
3. Same-subject cells already exist in the manifest; nothing to build.
4. Cost table script.

### 3.4 Baselines

1. Re-run harnet, UniMTS, ImageBind, NormWear from a clean tagged commit on `adaptation_v1`.
   ~12 min of GPU plus head fits. Keep the result JSONs and their fingerprints in the branch.
2. Weight audit for Wonderwall, IMUZero, LanHAR, GOAT, MOMENT (reading task). Add an adapter only
   if weights are released and the contract can be met by Oct 10; otherwise cite and exclude with
   the stated reason.

### 3.5 Documentation / provenance

1. Restore the archived classification results docs to `docs/results/classification/`.
2. Retire `halo_compact` and `halo_evidence` adapters from the table (keep code).
3. The paper's numbers come only from result JSONs whose `git_dirty` is false and whose
   `manifest_fingerprint` matches `adaptation_v1`.

---

## 4. Schedule against Nov 1 (eight weeks, one shared 24 GB GPU)

| week | ends | deliverable | gate |
|---|---|---|---|
| 1 | Sep 10 | 3.1 data wiring + compatibility table; 3.5.1 docs restored; baseline weight audit | family table reviewed by hand |
| 2 | Sep 17 | 3.2.1 Phase-A compact-shape pretrain (Arm A text-OFF, Arm B text-ON); 3.2.2 support-only comparator + tests | encoder effective rank healthy; mixer identity-at-init test passes |
| 3 | Sep 24 | 3.2.3 `train_compare.py` + sampler; smoke on real caches; step-0 predictor | every batch has the planned positive share; no-retrieval path scores the untrained floor |
| 4 | Oct 1 | first full Arm A run (35k, seed 1) + `halo_compare` adapter; baselines rerun clean | **the k-curve result is in hand** — this is the Nov-1 go/no-go |
| 5 | Oct 8 | seeds 2-3; Arm B run; same-subject and cross-subject tables; cost table | paired step-0 control computed for every run |
| 6 | Oct 15 | ablations (comparator vs 1-NN/prototype; filter on/off; p sweep); heterogeneity-axis figure | |
| 7 | Oct 22 | paper draft complete, blind; figures final | internal read by supervisor |
| 8 | Oct 29 | revisions, supplement, anonymised repo, submission Nov 1 | |

Budget notes. Phase-A 26 min; the old 35k episodic run was ~3.3 h at the compact shape, so a
support-only run should be comparable or cheaper (no bank). Three seeds x two arms x 3.3 h fits in
week 4-5 on one GPU. Baseline reruns are minutes. The schedule has no slack for a second model
redesign; if the week-4 gate fails, the honest move is Feb 1.

---

## 5. Decisions needed before build starts

1. Zero-shot row mechanism (3.3.2): comparator over a candidate-excluded compatible corpus draw
   (recommended) vs ConSE.
2. Whether Phase-A for Arm A is pretrained text-OFF (recommended: the headline encoder never sees
   acquisition text at any stage) or shared with Arm B and only fine-tuned text-OFF.
3. Confirm the Arm B2 compatibility-distance table once 3.1.2 is drafted.
