# Matched-corpus baseline experiment — plan (2026-09-15)

**Status:** M2 is implemented, unit-tested, and smoke-tested but has not had a full run. M0 is
available through the shared evaluation readouts; M1 is not built. Implementation details and the
measured compute budget are in `MATCHED_CORPUS_BUDGET_20260915.md`. This plan answers the
supervisor's second point from the 2026-09-15 meeting: released checkpoints differ in training
corpus and scale, so each matched condition must state exactly what it controls.

Companion documents: `DEPLOYMENT_SCENARIOS_PLAN_20260915.md` (where we look for an edge) and
`ABLATION_PLAN_20260915.md` (why the edge exists once found).

## 1. The question, and why one arm cannot answer it

"Train the baselines on our data" is three different experiments. They answer different questions
and have different faithfulness costs, so they must be reported as separate rows and never merged.

| level | what is held constant | what it isolates | faithfulness cost |
|---|---|---|---|
| **M0 — matched readout** | released encoder frozen; only the readout is fit on our corpus | nothing about the encoder; removes readout advantage | none: this is how the model is deployed |
| **M1 — matched corpus, own objective** | our corpus, their architecture, **their** pretraining objective | what their method learns from our data | high for corpus-scale methods: retraining a 700k-person-day model on 224k windows is not that model |
| **M2 — matched corpus and objective** | our corpus, their architecture, **our** support-conditioned objective and readout | **the encoder architecture alone** | their objective is discarded, so this is not "their model" at all |

**M2 is the arm that answers the supervisor's challenge.** If a fixed physical filterbank beats a
from-scratch HARNet or LiMU-BERT backbone under identical data, identical objective, identical
episodes and identical readout, the encoder design is a contribution. If it does not, we learn that
the objective and curriculum are the contribution, which is a publishable finding and a better paper
than a defended ambiguity. M1 is the arm reviewers will ask for; M0 is bookkeeping we partly have.

**Nothing here replaces the released-checkpoint rows.** Those stay as the deployment comparison: a
practitioner downloading HARNet gets UK Biobank scale, and that is a real advantage they really have.

## 2. What "our corpus" is, exactly

HALO's promoted run (`halo_fixed_mr_residual_v3_8s_4res_40k_20260914`) has `phase_a: None` — **no
label-free pretraining at all**. Its entire training signal is the supervised episodic stage over
the eight labelled head datasets. That makes the matched corpus unambiguous and small:

| dataset | streams gridded | windows (8 s, native) |
|---|---:|---:|
| realdisp | 9 | 49,212 |
| xrf_v2 | 6 | 44,898 |
| wisdm | 2 | 41,813 |
| dsads | 5 | 29,005 |
| harmes | 1 | 27,318 |
| hhar | 2 | 12,940 |
| forth_trace | 5 | 9,450 |
| kuhar | 1 | 8,951 |
| **total** | **31** | **223,587** |

Same subject holdout, same execution-disjointness, same sealed six never touched. The label-free
pretraining corpus (`capture24_pretrain`, `nymeria_xsens`, `extrasensory_pretrain`, 106 GB) is
**out of scope**: HALO does not use it, so including it for baselines would hand them an advantage
HALO declined. If a baseline's objective is unlabelled-only (LiMU-BERT, NormWear), it trains on the
unlabelled signal of these same eight datasets.

**A leakage note that currently favours a baseline.** LiMU-BERT's released checkpoints are pretrained
on HHAR, UCI-HAR, MotionSense and Shoaib. Two of those — MotionSense and Shoaib — are **in our sealed
evaluation set**. Verify this against `baselines/limubert_x/citation.json` and the source paper
before asserting it, but if it holds, the released LiMU-BERT-X row has seen our test signal
unlabelled, and the M1 arm is the fix, not a handicap. This belongs in the paper either way.

## 3. Corpus export

One export per input contract, written once and reused by every arm. Build as
`data/scripts/export_matched_corpus.py`, writing under `data/matched_corpus/<contract>/`.

| contract | channels | rate | window | consumers |
|---|---|---|---|---|
| `acc3_30hz_10s` | acc x/y/z | 30 Hz | 10 s (HARNet native 5 s, tiled) | HARNet |
| `imu6_20hz_1s` | acc + gyro | 20 Hz | 1 s clips | LiMU-BERT |
| `acc3_20hz_10s_joints` | acc x/y/z placed on 22 SMPL joints, rest masked | 20 Hz | 10 s (200 samples) | UniMTS |
| `raw_native` | as gridded | native | 8 s | NormWear, M2 arms |

Rules for the export, all of which must be asserted in code:

1. **Subjects**: the same training-subject set HALO used. No sealed subject may appear.
2. **Resampling** uses each adapter's own `InputContract` path, so a baseline is fed exactly what its
   adapter feeds it at evaluation time. Do not write a second resampler.
3. **Gravity**: preserve each stream's declared `gravity_state`. Do not silently high-pass.
4. **Per-stream cap**: reuse `MATCHED_MAX_PER_STREAM` from `baselines/harnet/adapter.py`, which the
   existing matched ConSE head already applies, so no model gets more of one placement than another.
5. **Label text** for text-objective arms comes from `training/support_classifier/label_text.py`,
   the same paraphrase table HALO uses. Using a different table would confound text quality with
   encoder quality.

## 4. Per-baseline plan

### 4.1 LiMU-BERT-X — run this one first

* **Why first**: it is the only baseline whose original pretraining scale is *comparable to ours*
  (four small HAR corpora), so M1 is scientifically fair rather than a handicap; its architecture is
  **defined in our own repo** (`baselines/limubert_x/adapter.py`, `_Backbone`), so M2 needs no
  external checkout; and it is 55,446 parameters, so training is minutes, not hours.
* **M1**: masked-reconstruction pretraining, the authors' objective, on `imu6_20hz_1s` from our eight
  datasets. Reference implementation: `legacy_code/auxiliary_repos/LIMU-BERT-Public/pretrain.py` and
  `models.py`. Port the loss rather than vendoring the training loop; the architecture already
  matches ours module for module.
* **M2**: instantiate `_Backbone`, mean-pool its hidden states exactly as the adapter's
  `_window_features` does, and hand that pooled vector to our support classifier in place of the
  filterbank encoder. Train with `training/support_classifier/train.py` arguments identical to the
  promoted run.
* **Caveat to state**: the released weights were trained on corpora that include two of our sealed
  test sets. M1 removes that; report both rows.
* **Effort**: ~1 day including the M2 encoder shim. **Cost**: under an hour of GPU per arm.

### 4.2 HARNet

* **M0**: already exists. `baselines/harnet/adapter.py` supports `HARNET_CORPUS=matched`, which fits
  the ConSE head on our corpus with our per-stream cap (`harnet5_conse_head_matched.pt`). This is
  readout matching only — **the encoder is still UK Biobank** — and the current results table must
  say which mode produced its HARNet row.
* **M1**: HARNet's contribution is 700k person-days of self-supervised UK Biobank wrist data. Its
  training code is upstream (`OxWearables/ssl-wearables`), not vendored. Retraining its multi-task
  SSL on 224k windows produces a model that is not HARNet in any meaningful sense. **Recommendation:
  skip M1 and say why.** Scale is the method; removing the scale removes the method.
* **M2**: worth doing and cheap. Load the harnet5 ResNet trunk **randomly initialised**, train under
  our objective on `raw_native`. This is a legitimate "strong 1-D CNN encoder" control and it is the
  single most informative comparison for the frontend claim, because HARNet's trunk is exactly the
  convolutional-depth alternative our `spectral-frontend-literature` journal entry names as the
  honest counter-evidence.
* **Effort**: M2 ~half a day. **Cost**: one 40k run.

### 4.3 UniMTS

* **The hard one, and the most interesting.** Its pretraining sample is `(T, 22, 3)`: simulated
  per-joint accelerometry over a 22-node SMPL skeleton, contrastively aligned to text
  (`auxiliary_repos/UniMTS/pretrain.py`, `data.py:CLIPDataset`). We have at most six real body sites
  and no skeleton.
* **M1 is feasible and principled**, because their own pipeline already masks joints: `CLIPDataset`
  randomly selects 1-5 joints per sample and zeroes the rest, and the evaluation path masks to the
  joints a dataset actually has. So the matched export places each of our real streams at its
  corresponding SMPL joint and masks the other joints — which is exactly the distribution their
  model is trained to handle. The joint mapping already exists in
  `baselines/unimts/adapter.py` (`N_JOINTS = 22` plus the documented joint semantics); reuse it, do
  not re-derive it.
  * Multi-placement datasets (realdisp 9, xrf_v2 6, dsads 5, forth_trace 5) populate several joints
    per sample, which is closer to their training distribution than single-phone datasets.
  * Text comes from our paraphrase table; their objective needs one sentence per sample.
  * Run `pretrain.py --gyro 0 --padding_size 200` with `data_path` pointing at the export.
* **M2**: instantiate `ST_GCN_18` randomly initialised over the same masked-joint tensor and train
  under our objective. This isolates the graph encoder from the mocap corpus.
* **Caveat to state plainly**: UniMTS's contribution is the synthetic-mocap corpus and the text
  alignment together. The M1 arm is "the UniMTS architecture and objective trained on real IMU at our
  scale", and it must be labelled that way in every table. It is not a reproduction and not a
  refutation of their paper.
* **Effort**: ~2 days, most of it the joint-placement export. **Cost**: their default is 100 epochs;
  budget one overnight run and check convergence at 10.

### 4.4 NormWear — **revised 2026-09-15 after a measurement; the earlier decline was unsound**

An earlier draft of this plan declined both arms on the grounds that the released checkpoint is
"collapsed on our data — effective rank 1.4 in 2,048 dimensions". **That reasoning does not hold.**
Direct measurement (`NORMWEAR_READOUT_FINDING_20260915.md`) shows two things:

1. We feed the **wrong tensor** to the enrolment readouts. `window_features` returns the MSiTF
   query-conditioned, text-aligned vector — NormWear's *zero-shot* head — while NormWear's own
   downstream benchmark pools **backbone patch tokens** for representation tasks. Correcting this is
   worth **+16.1 macro F1** on MotionSense and **+7.8** on InclusiveHAR under an identical probe.
2. The collapse diagnosis is the wrong mechanism. The correct backbone features have a *higher*
   mean pairwise cosine (0.996 against 0.983) and a similarly low uncentred rank, yet retrieve 14
   points better. High cosine is a common-mode offset that centring removes; it is not evidence of
   an uninformative representation. The quoted rank of 1.4 also does not reproduce under the repo's
   own centred definition, which gives 18.5.

**Revised plan.**

* **Prerequisite, before any arm**: fix `window_features` to return pooled backbone tokens and
  re-run NormWear's enrolled rows everywhere. Until that lands, every NormWear enrolled number in
  this project — the results table, the failure analysis, the scenario runs — understates it. The
  zero-shot row is unaffected: the MSiTF path is its native zero-shot mechanism and is implemented
  faithfully.
* **M0 — matched readout: required, not optional.** With corrected features NormWear must be
  re-measured before any claim about it is made. This is now the arm that matters most for NormWear.
* **M1 — decline, but for the right reason.** NormWear is a 1.29B-parameter multi-modal foundation
  model pretrained across physiological signals far beyond IMU; retraining its objective on 224k
  wrist/waist IMU windows tests our ability to train their model, not their model. This is the same
  scale argument used for HARNet, and it is sound. State it that way.
* **M2 — optional, low priority.** The backbone is a channel-independent ViT over ricker-CWT
  scalograms, which is a genuinely different frontend hypothesis from ours and would be an
  interesting encoder control. But it is the largest model in the set and the slowest to train, and
  HARNet's trunk already provides a convolutional control more cheaply. Do it only if steps 2 and 4
  leave the frontend question open.

**Lesson for the fairness contract.** A baseline that looks broken should be assumed mis-adapted
until its own repository's usage has been checked. Add that check to §5.

### 4.5 Mantis (optional, not currently a baseline)

If a matched-corpus table is being built anyway, Mantis is the cleanest architectural foil for the
frontend argument: it interpolates every series to 512 samples and instance-normalises gravity away,
the two things we refuse. `MANTIS_BASELINE_PLAN.md` already specifies the adapter. M2 only.

## 5. Fairness contract

Every one of these must hold or the table is worthless, and each should be asserted in code:

1. **Same windows.** Every arm trains on the identical exported windows for its contract; the export
   is fingerprinted and the fingerprint is recorded in each run's config.
2. **Same subject holdout, same sealed six.** Assert no sealed subject or dataset enters any arm.
3. **Same evaluation.** Identical sealed manifests, identical readouts, identical episodes. Nothing
   about §4 changes `sealed_eval.py`'s protocol.
4. **No handicapping.** Each baseline gets its own preprocessing, its own augmentation, its own
   optimiser defaults, and its own best-known hyperparameters. Where we must choose, choose the
   setting most favourable to the baseline and record the choice.
4b. **Use each model the way its own repository uses it.** Before any arm runs, confirm against the
   upstream code which tensor that model uses for *representation* tasks and which for *zero-shot*,
   and feed each readout the matching one. NormWear failed this check (§4.4): we fed a text-alignment
   head into nearest-neighbour retrieval and lost up to 16 macro F1. Record the upstream file and
   line that justifies each choice.
5. **No advantage HALO declined.** No arm gets the 106 GB label-free corpus, because HALO did not.
6. **Capacity is reported, never matched.** Do not shrink a baseline to HALO's parameter count;
   report `parameters_m` beside every row and let the reader judge. Matching capacity would cripple
   models whose design assumes scale.
7. **Convergence is evidenced, not assumed.** Each arm publishes its training curve and the step its
   checkpoint came from. A baseline that did not converge is a failed arm, reported as such, not a
   low number.
8. **A priori stopping.** Fixed step budget per arm, decided before the first run, matching HALO's
   40k where the architecture allows. No arm is selected on any sealed result.

## 6. Reporting

One table, three blocks, in `docs/results/MATCHED_CORPUS_RESULTS_*.md`:

```
released checkpoint      (deployment comparison — the current table)
M1 matched corpus        (their objective, our data)   [UniMTS, LiMU-BERT; HARNet/NormWear: declined, with reason]
M2 matched everything    (our objective, our data)     [all architectures + HALO's own encoder]
```

The M2 block is the only one where a row difference is attributable to the encoder, and the text
must say so. Each row carries `parameters_m`, the corpus fingerprint, the step count, and the
convergence evidence. HALO appears in all three blocks unchanged — it is already matched to itself.

## 7. Order and cost

| step | arm | effort | GPU |
|---|---|---|---|
| 1 | corpus export + fingerprint + assertions | 1 day | none |
| 2 | LiMU-BERT M2 (encoder shim under our objective) | 0.5 day | <1 h |
| 3 | LiMU-BERT M1 (their masked objective) | 0.5 day | <1 h |
| 4 | HARNet M2 (random-init trunk, our objective) | 0.5 day | ~1 h |
| 5 | UniMTS export with joint placement | 1.5 days | none |
| 6 | UniMTS M1, then M2 | 0.5 day | overnight |
| 7 | table + convergence appendix | 0.5 day | none |

About a week, most of it export code rather than compute. Steps 2 and 4 alone give a defensible M2
block and are the highest value per day.

## 8. What this cannot fix

Matching the corpus does not make these deployed-system comparisons into parameter-matched
ablations, and the paper should stop implying otherwise. It also cannot recover what a released
checkpoint learned from data we do not have. The honest framing after this work is: *"against
released checkpoints we are competitive at their own task and far ahead under deployment
heterogeneity; under a matched corpus and a matched objective, our encoder does X."* X is currently
unknown, and steps 2 and 4 determine it within two days.
