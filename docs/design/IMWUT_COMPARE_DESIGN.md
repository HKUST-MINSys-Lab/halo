# Recognize by comparison — agreed design and paper shape for the IMWUT submission

## Current decision: staged encoder and prediction heads (2026-09-08)

The current implementation is `dual_attention` plus the separate `neighbors` encoder-training
experiment. The September 7 `sensor_only` model remains the source of the completed
[matched results](../results/IMWUT_MATCHED_ADAPTATION_20260908.md); those scores do not describe
the new heads. No full training or evaluation of the new design has been performed.

### Model

Both heads use the same `RecordingAttentionHead` architecture with independent weights:

- k=0: one query-recording token and candidate-label tokens, no background support bank.
- k>0: query, candidate labels, support-recording tokens and separate support-label tokens.
- Shared sensor projection for query/support; shared text projection for support/candidate labels
  within each head. Projection + LayerNorm precedes token composition.
- Four role embeddings identify the token types. Each support recording and its label share a
  positive instance ID; different recordings have different IDs even when their labels agree.
  Query and candidate tokens have no instance tag. These IDs do not identify classes or subjects.
- Training randomly assigns instance tags. At inference explicit IDs must travel with support
  pairs when reordered; without supplied IDs, the head assigns IDs in input order. Thus permutation
  equivariance is exact for tokens WITH their tags, not an assertion of invariance to changing the
  tag vocabulary itself. The default capacity is 512 support instances and fails loudly above it.
- Content/role/instance directions have controlled learned gains (initially 1/.25/.25).
  Two pre-LayerNorm attention + GELU FFN blocks, residual connections, depth-scaled residual
  initialization and final LayerNorm use the existing `SetAttentionStack`.
- Final scores are scaled cosine similarities of contextualized query and candidate vectors.
  There is no vote, residual correction, hard retrieval, or candidate-specific classifier weight.
- Enrollment availability selects the head, never the query's ground-truth label. Zero-shot
  episodes do not encode background windows. Support-removal diagnostics hold the head fixed.

The encoder is shared, but the first head experiment freezes it to isolate the head's effect.
`--no-freeze-encoder` explicitly enables later end-to-end training. Old `sensor_only`/`fused`
readouts and checkpoint shapes remain available for reproduction, not as the new default.

### Separate neighbor experiment

`neighbors` trains the encoder through raw cosine similarities of normalized recording outputs.
It has no trainable projection, attention head, or episode centering. Every support has an enrolled
candidate identity; every training query must have actual correct-label support. Cross-entropy on
per-class log-sum-exp neighbor scores equals negative log probability assigned to correct-label
neighbors. Both query and support encoder paths receive gradients. k=0 is explicitly unsupported
for this control, rather than inventing a text bridge. Its native enrollment rule is soft neighbors;
the evaluation's generic 1-NN remains the hard-neighbor control.

Examples (mechanically runnable; full experiments have not yet been launched):

```bash
python -m training.compare.train --comparator-readout neighbors --out training/compare/outputs/neighbor_encoder
python -m training.compare.train --comparator-readout dual_attention --phase-a training/compare/outputs/neighbor_encoder/last.pt --out training/compare/outputs/attention_heads
```

Use the project venv interpreter. `--phase-a` is the existing encoder-checkpoint argument and also
accepts a neighbor-trained checkpoint; it does not imply JEPA pretraining. The default attention
run requires an encoder checkpoint and freezes it. Three-step CPU smokes of both stages passed.
Smoke scores are wiring checks, not evidence of model quality. Checkpoint readout and source
  fingerprints distinguish the experiments. Published result tables remain unchanged.

### Deployment-matched training episodes

- A training step draws four independent support sets by default. Their base query datasets are
  sampled without replacement within a step; failed draws retry within that dataset rather than
  silently replacing sparse sources. Each set is reused by up to four
  distinct query executions, yielding up to 16 classified queries without encoding its support
  repeatedly.
- Enrollment episodes sample k from the feasible members of {1, 2, 4, 8} for that query/configuration
  and draw exactly k distinct physical executions for every candidate. Two windows per execution are
  sampled and averaged when available. Deployment averages all windows in an enrolled execution:
  this is the lower-variance version of the same execution mean, while random train-time subsampling
  keeps the batch bounded and exposes the encoder to within-execution variation. Neither path
  normalizes before averaging. No support is manually inserted, duplicated, or made compatible by
  relaxing acquisition constraints.
- Cross-subject support is requested 80% of the time. If one relation is impossible, the feasible
  relation is used and the fallback rate is reported. Candidate and additional-query shortages are
  likewise visible in telemetry. Independently drawn support sets receive equal loss weight even if
  one has fewer usable queries.
- Direct zero-shot episodes contain candidate text and queries but no fake background memory. The
  neighbor experiment is enrollment-only by construction.
- Staged candidate counts default to 6-14, matching the current evaluation manifest's observed
  range. The neighbor encoder is trained from scratch at the base `3e-4` learning rate; the `0.05`
  encoder multiplier remains the default only for head/legacy fine-tuning.
- Validation requests one fixed support set per eligible held-out source by default. Sources that cannot
  structurally form the requested execution-disjoint regime in the held-out split are excluded and
  counted in telemetry; the remaining sources are each represented before any is repeated.

### Remaining agreed work before substantive training

1. Run the neighbor-only encoder experiment first, then train the two attention heads separately
   from the shared encoder. Measure k=0,1,2,4,8; report the different k=0 dataset cohort explicitly.
2. Track head-specific gradients, query/support signal, support ablations, rank and losses.
   For new heads `base_logits` is a neighbor floor (uniform without enrollment), not the old vote;
   `neighbor_floor/*` telemetry describes this floor, NOT internal attention weights.
3. Encoder learning-rate changes and frontend/temporal-resolution changes are postponed. JEPA
   future-latent prediction and physical-feature reconstruction remain proposals, not active losses.

### Sources

- [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html): attention over sets.
- [Pre-LayerNorm analysis](https://proceedings.mlr.press/v119/xiong20b.html): gradient stability.
- [Neighborhood Components Analysis](https://www.cs.toronto.edu/~hinton/absps/nca.pdf): differentiable
  neighbor classification; our log probability objective is an adaptation, not an exact reproduction.

---

## Historical design snapshot (September 3-7, superseded above)

The remaining sections record the rationale and settings of the previously evaluated model.
Their defaults, readiness statements and preliminary protocol fingerprints are historical, not
instructions for configuring the new staged experiment. See the linked matched report for results.

This supersedes the clinical / motion-monitoring pivot as the paper target. The
classification-era code (Phase-A tokenizer, evidence/compact engine, baseline adapters, the
`adaptation_v1` evaluation manifest) stays the substrate; the application tasks remain on `main`
untouched.

---

## 0. Why this direction

Supervisor decision, 2026-09-03: clinical venues want a concrete solution to one condition with
data gathered for it; a generic technique that happens to apply to several conditions will not
land there. HALO started as an ML paper and should stay one. The right venue for a well-evaluated
incremental improvement on the HALO line is IMWUT, where the recent comparable work lives
(CrossHAR, GOAT, oneHAR, IMUZero, LanHAR, MASTER, Customizable-FM, Wonderwall).

What we learned across v1 and v2 (see `docs/results/` and the memory ledger) points at one
increment:

- The encoder was never the bottleneck: frozen HALO features linearly separate activities at
  84 macro-F1 while zero-shot through a text bridge sits near 40.
- Comparison against labelled exemplars is the right primitive: the *untrained* retrieve-and-vote
  mechanism reached 44.1 macro-F1 like-for-like, versus 47.3 for harnet and 42.7 for ConSE.
  Prototype/ridge over frozen features beat every trained head we built. The earlier 47.5 figure was
  confounded and is not evidence that the untrained mechanism beat harnet.
- Plain cosine retrieval over a heterogeneous bank fails for a specific reason: it ranks by
  acquisition configuration (x7 lift) rather than by activity. Learned retrieval was worth
  exactly nothing. The stage should go.
- The one learned component that helped (the attention mixer, +0.12) helped only because the
  language channel was in the loop: a scrambled-vocabulary control inverted the gain.
- Training with the answer absent from the *candidate roster* pushed zero-shot below chance under
  closed-vocabulary cross-entropy. The corrected task withholds its support but keeps the answer
  selectable. Zero-shot is still capped by the weak relation between signal similarity and
  label-name similarity (r ~ 0.11).
- A short end-to-end run showed an early effective-rank drop, but longer historical runs from random
  initialisation performed best. Warm-starting from self-supervised pretraining remains an untested
  comparison arm, not a requirement.

The literature has moved the same way: sensor heterogeneity (user, device, placement) is named as
the dominant barrier in every 2025-26 survey, ZARA (ACL 2026) gets training-free transfer by
reasoning over *retrieved reference statistics* with an LLM, and HARBench (PerCom 2026) now scores
position-robustness and few-shot as separate axes. We take ZARA's structure — features plus
labelled reference recordings — and replace the LLM with a small learned comparator.

---

## 1. Thesis

**Recognise by comparison, not classification.** Given a query window and a handful of labelled
recordings that were acquired under a compatible sensor configuration, a small comparator decides
which examples the query resembles and reads the label off them in language space. The model is
not asked to learn a universal representation of all IMU data; it is asked to embed motion into
features good enough that attention over a compatible support set can make the call.

The contribution we market is the **training curriculum**: how support sets are sampled during
end-to-end episodic training so that the comparator learns to compare rather than to memorise a
vocabulary. Nothing in the architecture is claimed as novel.

---

## 2. Model (kept deliberately simple)

| Component | Decision | Rationale |
|---|---|---|
| Front end | **Fixed physical filterbank, single resolution, not learnable** | The learnable arm stayed pinned at its init; simplicity wins. Multiresolution's +0.034 is a documented option we deliberately do not take. |
| Feature extraction | **Existing Phase-A tokenizer + temporal trunk, unchanged** | No evidence that complicating it buys anything. |
| Conditioning | **Rate and patch-duration pathways kept. Acquisition-configuration text is OFF in the core design.** Compatibility is handled at support construction instead (Section 3). | The encoder is trained on every configuration anyway; once support is compatible by construction there is nothing left for the text to tell it. The text pathway is kept in the code for the Section 6 experiment. |
| Comparator | **Attention over the query and every support example** (no retrieval stage, no top-k) | Removes the config-ranking defect and the non-differentiable selection; K is small enough to attend over fully. |
| Feature centering | **ON by default**: each episode's mean feature is subtracted from the query and support rows before similarity and attention | Acquisition configuration is close to a common mode within an episode, since the support all shares the query's key; the previous design's retrieval ranked by configuration at a 7.0x lift. Removing the mean leaves only how rows differ. `--no-center-features` is the ablation. |
| Readout | Fixed query/support cosine vote, plus a **sensor-only** learned reweighting: attention over the query's and support rows' signal vectors emits one zero-initialised scalar per support row that shifts its logit before the softmax (`--comparator-readout sensor_only`, design of record 2026-09-07). Label and candidate text never enter the learned path. | Sensors are compared with sensors and labels with labels, paired by index in the vote. The learned part decides only which examples to trust; it cannot become a label-text-to-motion bridge, which is the pathway every earlier learned text component hurt through. Step 0 is exactly the closed-form vote; unseen candidates use the same shared operation and no candidate-specific parameter. The `fused` readout (label text fused into support tokens, residual per candidate) is kept as an ablation arm. |
| Label / text tower | Frozen sentence encoder (MiniLM), one vector per label string | Zero-shot evaluation ensembles support draws, not label paraphrases. |
| Size | Compact engine budget, ~1M trainable parameters | Efficiency is part of the story. |

Support examples are **encoded per query, with gradients**, at train time. Nothing is cached.

**Correctness revision, 2026-09-07.** Each support recording's projected signal, acquisition
description and label are fused into one support token before set attention. Their role embeddings
are retained inside that fusion; candidate bindings remain separate from recording identity.
This keeps the feature-label association available for unbound zero-shot background as well as
enrolled support. Candidate tokens and query feature/description tokens remain explicit. Padding
is excluded from the cosine-softmax denominator, and the residual head still starts at exactly zero.
Older comparison checkpoints without the support-fusion parameters require the earlier branch
revision; they must not be silently loaded as this design or used as its paired step-0 control.

---

## 3. Support-set contract (the method)

A support set is the list of labelled executions the comparator sees alongside the query. Training
samples one window from each distinct execution on each draw, which is a stochastic view of that
execution and gives every execution equal influence. Evaluation deterministically pools all windows
of each enrolled execution into one mean vector **at the encoder's native scale** — the same scale as
a training support row, never L2-normalised, because episode centering averages query and support
rows together and the residual head was fitted to raw-scale rows
(`tests/test_compare_eval_parity.py`). Neither path lets a long execution cast more votes than a
short one.

1. **Compatibility is a filter, not a learned quantity.** Support examples must share the query's
   acquisition configuration family: device family, placement, channel set, gravity state.
   Sampling rate is *not* part of the key (the filterbank is rate-invariant by construction). We
   do not offer a smartwatch example to a pocket-phone query. This is a plain deployment
   consideration; we claim no novelty for it. The compatibility key already exists in the
   application code (`SensorCompatibilityKey`) and is reused as-is.
2. **Never the query or its physical execution.** Few-shot episodes mix same-subject and
   cross-subject enrollment with equal probability when both are feasible. Zero-shot support comes
   from other subjects, matching the external training-corpus support used at evaluation.
3. **Random label subset.** Draw 2–14 candidate labels and K support executions. Grid labels are
   canonicalized at the corpus boundary; the episode sampler applies no additional synonym merging
   or deduplication.
4. **Ground-truth support present with fixed probability.** The answer always remains in the
   candidate roster. With probability p all candidates have enrolled support (few-shot); otherwise
   no candidate has enrolled support and compatible examples of other labels form unbound
   background (zero-shot). The two regimes are interleaved.
   Zero-shot distractors are drawn from the same available compatible label pool as the answer,
   not preferentially from other configurations. The roster shrinks to leave at least one
   non-candidate background label. Pools with fewer than three available labels cannot supply this
   regime and the draw is retried; the realized mixture and rejection rate are logged.
5. **Both regimes use one-hot cross-entropy** because the correct answer is present in both candidate
   rosters. Loss is averaged over episodes, so p is the objective-mixture weight.
6. **Support may span datasets** as long as every example passes the compatibility filter. The
   encoder is trained on all configurations; the filter only constrains what is compared.

Constants are set by judgement before the first run. The training corpus's deterministic
subject-held-out split is health telemetry, not an external development set and not permission to
tune against evaluation data:

| constant | value | note |
|---|---|---|
| total support executions (train) | 32 | shared across the episode's candidate or background labels; it may shrink when the compatible pool is smaller |
| p (GT present, train) | 0.5 | joint ZS/FS; a sweep over p in {0.25, 0.5, 0.75} is a cheap experiment |
| label-subset size | uniform in [2, min(14, labels available)] | |
| fine-tune schedule | 35k-50k steps | the plateau we saw at 6k was a schedule artifact |
| checkpoint | final step of the fixed budget | no selection on held-out data |
| seeds | >= 3 | validation-draw variance dominates single runs |

At inference, `k` means **enrolled executions per candidate**, not the total support size used by
training. The evaluation therefore presents up to `candidate count * k` support executions. The
comparator is set-based and accepts that variable size; this train/test difference is explicit and
must be inspected in the k-curve rather than described as if the two quantities were identical.

---

## 4. Training

**One stage: end to end from random initialisation** (revised 2026-09-04). Encoder and comparator
are trained together on episodes drawn by the Section 3 sampler. The sole loss is mean
cross-entropy over candidates in both regimes.

This corrects an earlier draft that made a Phase-A warm start mandatory on the grounds that
from-scratch training collapsed the encoder's effective rank. The checkpoints contradict it: every
compact-engine checkpoint on disk, including `long_4h_20260821` — the one leading 33 of 40
enrolment columns at selection 0.5424 — records `phase_a_checkpoint=None`. The rank collapse was
measured on a 6,000-step schedule where the step-1,750 "peak" was head saturation, not convergence;
at 35,000 steps the same from-scratch recipe produced the best result the project has.

Rank collapse therefore stays as **telemetry worth watching**, not a demonstrated failure mode, and
`encoder/effective_rank` is logged every validation.

The Phase-A warm start remains available as a second arm (`--phase-a`). Which wins at a 35k
schedule has never been tested head to head; that is an experiment, not a settled question.

---

## 5. Evaluation

- **Protocol**: the frozen `adaptation_v2` manifest — **10** held-out datasets, subject-disjoint,
  69 cells (48 usable), fingerprint `5058cf41…`. `adaptation_v1` (7 datasets) stays intact so the
  pre-pivot numbers remain comparable; the two carry different protocol names so the assembler
  cannot mix their rows.

  **motionsense, realworld and shoaib were restored 2026-09-04.** They were never unsuitable — they
  were the Phase-B *development* split, withheld so repeated development would not consume the
  datasets meant to support the final claim. That split no longer exists. Restoring them doubles
  the ordinary-activity enrolment cells (5 -> 10) and lifts ordinary zero-shot cells (4 -> 7)
  without losing a single clinical cell, and it decorrelates two things that were confounded: with
  only the seven, 8 of 11 streams were wrist and the hardest labels sat on exactly the streams with
  the narrowest support pools. All three are phone streams carrying ordinary locomotion.

  **To disclose in the paper:** design decisions in the superseded Phase-B line were made while
  looking at these three. No model was ever trained on them, so this is researcher exposure rather
  than data leakage — but the reviewer should hear it from us. `motionsense` additionally has no
  config-compatible training stream, so like `upper_limb_use` its k=0 row is unsupported while its
  k>=1 rows are unaffected.

  `hapt` is permanently excluded and now raises if requested: it is the same 30 subjects as
  `uci_har`, which is in the training corpus.
- **Primary matched benchmark**: the cross-subject, same-configuration enrollment curve at
  k in {1, 2, 4, 8}. Every point uses the same relation cells, subjects, candidate rosters, and
  nested support prefixes; cells unable to reach k=8 remain available only in a clearly labelled
  partial-coverage supplement. Scores are dataset-macro F1 over five serialized support seeds.
  Paired uncertainty is a dataset-balanced subject bootstrap and is labelled with that estimand.
  Zero-shot (k=0) is a disclosed secondary row using the released model's native rule.
  The zero-shot headline uses only the intersection of scored cells across models with results,
  with per-model dataset/cell coverage shown separately. Variant cell-level deltas and subject-level
  deltas are separate statistics; a subject-bootstrap interval belongs only to the latter.
- **k=16 is a separate cohort**, not the final point of the primary curve. Only six datasets have
  enough independent executions, and their eligible query subjects differ from the k=1..8 cohort.
- **Support at test time** comes from the held-out dataset's own enrollment pool, which is
  config-compatible by construction. Two enrollment modes are first-class:
  **cross-subject** (support from other people) and **same-subject** (support from the user's own
  recordings). The available data support a multi-dataset fixed k=1..8 curve only for the former;
  same-subject results are therefore reported per supported k and never pooled into that curve.
  TNDA-HAR's released bundle omits participant identity, so its enrollment result is explicitly
  **subject-unattributed** and contributes to neither same-subject claims nor subject bootstraps.
- **Mandatory control rows**: the *untrained floor* (same mechanism at initialisation) and the
  *step-0 control* (paired against each trained run), following the methodology rule that every
  learned component is guilty until a control clears it.
- **Baselines = released checkpoints only.** Training regimen and data are part of each method;
  we do not match them. Keep harnet, UniMTS, ImageBind, NormWear. Drop CrossHAR and LiMU-BERT
  (we pretrained those ourselves). Audit for released weights before writing: Wonderwall
  (IMWUT'26), IMUZero (IMWUT'25), LanHAR (IMWUT'24), GOAT (IMWUT'24), MOMENT. For k >= 1 every
  representation receives the same 1-NN, prototype, and closed-form ridge
  readouts; HALO's support comparator is an additional row. Native rules are used only for k = 0.
- **Closed-set models have no k = 0 row.** Historical ConSE results remain reproducible, but a
  locally fitted semantic bridge is not the released model's native rule and is excluded from the
  current zero-shot table. Those encoders still enter every k >= 1 comparison through the matched
  representation readouts above.

---

## 6. Two arms, then ablations

**Arm A — explicit compatibility control (the core design).** No acquisition text reaches the
encoder. Support is filtered to the query's configuration family at construction time. This is the
headline configuration and every claim in Section 7 is made about it.

**Arm B — learned compatibility (an experiment, not a core design).** Acquisition text is ON and
support construction allows *anything* — compatible and incompatible examples mixed. The question
is whether the comparator learns on its own to gauge which examples are compatible, which to
trust and which to discount. Two sub-questions fall out of it:

- B1. With mixed support, does Arm B recover Arm A's accuracy (learned filter ~ explicit filter)?
- B2. When *no* perfectly compatible example exists — the support holds only near misses, e.g. a
  different placement on the same device family — can the model still draw something useful from
  them, and how does accuracy degrade with compatibility distance?

Arm B is reported as a result about the model, not as a claim the paper rests on.

**Ablations on Arm A** (each answers one reviewer question):

| Ablation | Question it answers |
|---|---|
| **Episode mean-centering OFF** (`--no-center-features`; centering is the DEFAULT as of 2026-09-04) | Does removing what the support rows have in common force the model to discriminate? Configuration is close to a common mode within an episode — the support all shares the query's key — and the previous design's retrieval ranked by configuration at a 7.0x lift, so this targets the measured defect directly. **Changes the step-0 function: compare raw scores at matched seeds, never paired gain.** |
| From-scratch vs Phase-A warm start | Does one-stage training beat two, at a 35k schedule? Never tested head to head. |
| **Fixed filterbank vs continuous kernel** (`--frontend`) | The encoder's two real front-end modes; see 6.1 — the existing head-to-head is inside the noise and must be re-run matched. |
| Fixed single-res filterbank vs multiresolution vs learnable | Is the simple front end leaving accuracy on the table? |
| Attention comparator vs cosine 1-NN vs prototype over the same encoder | Is the learned comparison worth having? (the untrained floor lives here) |
| **Sensor-only reweighting vs fused label-in-attention** (`--comparator-readout`) | Does letting the learned part see label text help or hurt? Both share the same step-0 function, so compare by paired gain. The run launched 2026-09-07 14:27 (`imwut_compare_arm_a_fixed_35k_20260907`) is the fused arm. |
| Compatible support vs unfiltered support, text OFF | How much does the explicit filter buy on its own? |
| Text vote vs one-hot vote | Is the language channel load-bearing? (plus the scrambled-vocabulary control) |
| p in {0, 0.25, 0.5, 0.75, 1} | Does joint ZS/FS training cost few-shot accuracy? |
| Step-0 control | Did fine-tuning help at all, paired? |

### 6.1 On the two front-end modes

The encoder has always had two genuine front ends — the fixed physical filterbank and the
continuous kernel bank — and both are selectable in the comparison trainer (`--frontend`), verified
to run from scratch at the compact shape.

A head-to-head already exists on disk, and it settles nothing:

| run | frontend | step | selection |
|---|---|---:|---:|
| `e2e_pb04_fixed_filterbank_35k_20260824` | fixed | 10,000 | 0.3642 |
| `e2e_pb04_continuous_dense_35k_20260824` | continuous | 13,000 | 0.3724 |

Continuous leads by 0.0082, but the two runs stopped at different steps, neither recorded a seed,
and the measured screening noise on this setup is sd 0.0065 — the standing rule being that nothing
under about 0.012 is real. So the gap sits inside the noise and was measured off unmatched
schedules. Only the continuous arm has downstream adaptation results on disk, so even the
second-stage comparison is one-sided.

The honest statement is that **we do not know which front end is better**, and the paper must
either say so or run the comparison properly: matched steps, matched seeds, at least three of them,
compared by raw score rather than paired gain (the front end changes the step-0 function). The
design of record stays *fixed* because it is the simpler claim and the one the thesis rests on;
continuous is the challenger, not the default.

## 7. Paper shape

**Working title.** Recognize by Comparison: In-Context Activity Recognition Across Heterogeneous
IMU Setups.

**Claims.**
1. Test whether a compact comparator over a fixed physical filterbank, told nothing about the
   sensor configuration but given K compatible labelled recordings at test time, improves over its
   exact closed-form vote and released foundation-model representations across ten held-out
   datasets. This remains a hypothesis until the declared runs are complete.
2. Any measured gain is tested against the exact step-0 vote and matched frozen-feature readouts;
   the curriculum interleaves support-present and support-withheld episodes.
3. No episode-specific label homogenisation is needed after the corpus's canonical mapping;
   canonical label strings vote through language similarity.
4. The whole system runs on device (parameter, latency and memory table).

**Sections.** Introduction (heterogeneity is a comparison problem) -> Related work (HAR foundation
models, language-aligned HAR, training-free/LLM reasoning, few-shot HAR) -> Method (front end,
comparator, support contract, curriculum) -> Evaluation protocol (manifest, leakage discipline,
baseline contract) -> Results (k-curve table, per-dataset, heterogeneity axes) -> Ablations ->
Cost -> Limitations (zero-shot cap stated plainly; compatibility filter assumes known placement)
-> Conclusion.

**Heterogeneity-axis experiment** (kept from the earlier thesis, now with a mechanism that can use
it): placement shift, rate shift, orientation perturbation, gravity present vs removed — ours vs
baselines under identical inputs. Built 2026-09-05 as `--perturb` on the adaptation runner
(`eval/perturbation.py`; placement shift is the existing cross-configuration cells), applied to the
frozen `adaptation_v2` cells on the query side (the figure) or the support side (the
incompatible-exemplar cell for Arm A vs Arm B). Not yet run.

---

## 8. Reuse map

| Reused as-is | New |
|---|---|
| Phase-A tokenizer + pretraining recipe | Episode sampler with no retrieval and the compatibility filter |
| Evidence mixer (becomes the comparator) | Joint support-present/support-withheld cross-entropy |
| `adaptation_v1` manifest, eval harness, adapters | Baseline weight audit for the 2025-26 comparators |
| Results/methodology tooling (paired gain, step-0 predictor) | Cost table |

---

## 9. Historical pre-build decisions

All three are now **DECIDED** (2026-09-03) and recorded in `docs/design/IMWUT_HANDOFF.md` §1:

1. **k=0 mechanism** — comparator over a candidate-excluded, config-compatible corpus draw (not
   ConSE), so one mechanism spans the whole k-curve.
2. **Arm A never sees acquisition text** — text is OFF at pretraining *and* fine-tuning; Arm A and
   Arm B get separate Phase-A runs (~26 min each).
3. **Arm B2 compatibility distance is binary, not a ladder** — distance 1 means same device family
   plus *equivalent* placement (left↔right wrist, pocket variants). Wrist↔ankle and watch↔phone are
   out of scope; we do not claim a graded degradation curve.

Remaining: constants in Section 3 are set by judgement and may be varied in training; baseline
weight audit (Section 5). Target deadline is **Nov 1 2026** (user decision), with the Oct 1 k-curve
gate in `docs/design/IMWUT_BUILD_PLAN.md`. Venue read: `docs/research/IMWUT_VENUE_READ.md`.

The former free-text placement blocker is resolved by the checked-in `PLACEMENT_SITE` mapping in
`data/scripts/curate/compatibility.py`. New placement prose now fails loudly until mapped.

The implementation now exists on `imwut/compare`; launch readiness is tracked in
`docs/IMWUT_START_HERE.md` and must be re-audited before a full run.
