# START HERE — branch `imwut/compare`

You are on the branch for the **IMWUT comparison-model line**. If you read only one document before
touching anything, read this one. It exists because this repository has carried three successive
research lines and most of the confusion available here comes from mistaking one for another.

Last updated 2026-09-08.

## Latest matched results (2026-09-08)

Completed HALO trained/initial controls and all four released baselines on the same current manifest
and clean evaluation source `83058a3`. The current tables and interpretation are in
[Matched adaptation results](results/IMWUT_MATCHED_ADAPTATION_20260908.md).
HALO 1-NN improves with training but its native engine still trails 1-NN at higher k; UniMTS leads
the primary aggregate. Zero-shot remains weak. The readiness notes below describe the preceding
audit; its clean-source and matched-manifest reruns have now been completed. Upstream overlap and
prior test-set exposure remain disclosure limitations.

## Evaluation readiness (2026-09-07)

Use `eval/manifests/adaptation_v2_20260907.json.gz` for every model in the next comparison.
The September 4 baseline tables and the preliminary dirty-source September 7 HALO output are
historical artifacts, not a final matched table. The evaluation CLI requires committed clean
source before loading models; the assembler also validates source, checkpoint, and manifest hashes.
Do not edit old JSON provenance or bypass these checks. Preserve concurrent agents' changes when
preparing the intended source commit. Run released HARNet, UniMTS, ImageBind, and NormWear on the
same manifest as HALO, with `nearest prototype ridge`; HALO additionally reports its native engine.

The assembler retains variant-qualified model names, reports every HALO native variant's paired
subject comparisons, writes `per_dataset.csv`, and adds a common-stream-coverage zero-shot table.
Its all-available averages are descriptive only when coverage differs. Unsupported predictions are
N/A, not zero scores. The current HALO mechanism lacks compatible zero-shot evidence on MotionSense,
RealWorld, Upper Limb Use, and USC-HAD; report that limitation, not only its supported average.

Table F1 pools query windows within each cell, averages cells/seeds per dataset, then datasets.
Paired subject-macro differences and their bootstrap intervals use a different, explicitly labelled
estimand. `k=16` remains a separate cohort from the fixed `k=1,2,4,8` curve. Random aliases test
label renaming, not novel physical actions. Models retain their published input contracts (including
different modalities and temporal padding/cropping), so this is a deployed-system comparison, not
a matched-data/architecture superiority claim. Upstream checkpoint-specific training-data overlap
must be disclosed; released weights alone do not prove that test sources were unseen. Repeated use
of these test datasets for design feedback must also be disclosed rather than called sealed testing.

---

## 1. Motivation — the problem, and why this is the response

### The problem the field has

Wearable activity recognition does not fail for lack of model capacity. It fails because **every
deployment is a different sensing problem**. Change the device, where it sits on the body, which
axes it reports, whether gravity is present, and the same activity produces a different signal.
Every 2025-26 survey of the area names this — heterogeneity across users, devices and placements —
as the dominant barrier to models that work outside the dataset they were fitted on.

The field's standard answer is to build a bigger, more invariant encoder: train on more sources,
augment away the nuisances, and hope one representation covers everything. We tried that here for
two years. It produces a representation that separates activities beautifully *within* a
configuration and transfers poorly *across* one.

### Why comparison, and not a better classifier

The measurement that redirected this project: our frozen features separate activities at **84.3
macro-F1** under a supervised linear probe, while zero-shot recognition through the label-text
bridge sits at **39.6** — a 44.7-point gap
([`RESULTS_V2.md`](results/classification/RESULTS_V2.md)). The encoder was never the bottleneck.
What was missing was a way to turn "this recording resembles that one" into an answer.

And when we gave the model exemplars to compare against, an **entirely untrained** retrieve-and-vote
rule reached **44.1** like-for-like — against 47.3 for a 4.2M-parameter model trained on roughly
700k person-days of wrist data, and against 42.7 for the ConSE bridge.

**Read that carefully, because it is easy to overclaim.** 44.1 does *not* beat harnet. An earlier
write-up quoted **47.5** for this mechanism and said it did; that figure carried two confounds and
its like-for-like value is 44.1 — the correction is recorded at the top of
[`EVIDENCE_ENGINE_FINDINGS.md`](results/classification/EVIDENCE_ENGINE_FINDINGS.md). What survives
is still the point worth building on: a mechanism with **no trained parameters at all** landed
within about three points of a model trained on four orders of magnitude more sensor data, and
above the standard open-vocabulary bridge. The comparison was doing real work before any learning
happened. That is a reason to invest in the comparison, not a claim to have won.

So the bet is: **stop asking for one representation of all human motion, and start asking a much
easier question.** Not "what activity is this?" but "which of these few labelled examples does this
resemble?" — with the examples chosen to be acquisition-compatible with the query, the way any real
deployment would choose them. That question is well-posed under heterogeneity in a way the first
one is not, because the query and its examples share a sensing configuration by construction.

### Thesis

**Recognise by comparison, not classification.** Given a query recording and a handful of labelled
example recordings acquired under a compatible sensor configuration, a small comparator decides
which examples the query resembles and reads the answer off their labels in language space.

The model is *not* asked to learn a universal representation of all IMU data. It is asked to embed
motion well enough that attention over a compatible support set can make the call.

### Why it should be publishable

Two capabilities fall out of the design rather than being bolted on. Labels are read through
language and no parameter is tied to a candidate, so **an unseen activity name is scored by the same
operation as a seen one** — **44 of the 63 candidate labels** in the `adaptation_v2` roster name
an activity the training corpus does not contain (48 by exact string; 4 more are synonyms of a
training label, e.g. `climbingup` for `walking_upstairs`, and the k = 0 support draw excludes those
by concept). And a user
enrolling a few examples on their own device is the *native* mode of the model, not a fine-tuning
afterthought, which is the deployment story a ubicomp venue actually cares about.

**What we claim as novel: the training curriculum** — how support sets are sampled. Nothing in the
architecture is claimed as new; the comparator is a matching network with a language readout, and
the related-work section says so.

**Target venue: IMWUT. Deadline: 1 November 2026.** Double-blind, 8–10k words, and a
major-revision-then-accept path is the norm rather than a failure. The venue read is in
[`docs/research/IMWUT_VENUE_READ.md`](research/IMWUT_VENUE_READ.md).

Why this line at all: the supervisor ruled out a clinical pivot on 2026-09-03 (clinical venues want
one condition with data gathered for it, not a generic method that touches several), and HALO began
as an ML paper. The evidence that shaped the design is a ledger of measured failures from the
earlier lines — summarised in §7.

---

## 2. Design philosophy — read this before you "fix" anything

The design is the product of a long run of *negative* results (§7). Almost everything clever that
was tried here made things worse, and the things that worked were subtractions. That history is why
the code looks the way it does, and it is the single most important thing to understand before
changing it.

**Five principles, in priority order.**

1. **Simplicity is a claim, not laziness.** A fixed filterbank rather than a learned front end, no
   retrieval stage, no per-candidate parameters, one shared scalar head. Each of those is a
   position we defend in the paper, and each was reached by *deleting* something that had been
   measured and found not to pay. Adding machinery back needs evidence, not intuition.
2. **The mechanism must be one mechanism.** The same comparison rule runs at every point on the
   k-curve, k = 0 included. A model that quietly switches to a different rule when the support set
   is empty has two mechanisms and one misleading graph.
3. **Deployment realism beats benchmark convenience.** The compatibility filter exists because no
   real product would compare a pocket-phone query against smartwatch examples. We claim no novelty
   for it — it is a considered constraint, not a contribution, and it is applied at support
   construction rather than learned.
4. **Honesty over impressiveness.** An unmeasurable cell is reported unmeasurable. A gap inside the
   noise floor is reported as "we do not know". A dataset that cannot answer a question is
   documented as such rather than coerced into a number.
5. **Controls before claims.** Every learned component is guilty until its control clears it,
   because on this codebase the majority of learned components have been net-negative on first
   measurement. The untrained floor and the paired step-0 control are not paperwork; they are the
   only reason we can tell learning from luck.

### Choices that look like bugs and are not

If you are sweeping this code for defects, these will all look wrong. They are deliberate. Each one
has a measurement or an argument behind it; changing any of them is a design decision for the user,
not a fix.

| Looks like a bug | Why it is there |
|---|---|
| **No retrieval stage.** The comparator attends over the whole support set with no top-k. | Retrieval was measured to rank by *acquisition configuration* rather than activity (×7.0 lift), and a learned retrieval stage was worth exactly +0.0000 paired. K is small enough to attend over completely. |
| **Episode labels use the corpus's canonical strings verbatim** — the sampler performs no further synonym merging or deduplication. Two candidates can carry near-identical text. | The readout gives them near-identical votes. Canonicalization already happens in `CorpusIndex`; claiming that it never happens would be false. |
| **Some k = 0 cells report "unsupported"** rather than a number. | Those streams have no config-compatible training partner, so the deployed mechanism genuinely does not apply. Substituting ConSE or padding with incompatible rows would report a different mechanism under this model's name. Principle 4. |
| **Support sets shrink below K instead of being padded.** | Padding with foreign rows would silently violate the compatibility rule; skipping the episode would bias the corpus toward well-populated configurations. Shrinking is logged in telemetry. |
| **Training starts from random init**, not from the Phase-A checkpoint that exists. | Every compact checkpoint that ever led the table was trained this way. An earlier draft of the docs claimed warm start was mandatory; that claim was wrong and is corrected. |
| **Encoder effective rank falls early in training.** | Measured before, on a 6k schedule where the apparent peak was head saturation. At 35k the same recipe produced the project's best result. It is logged as a watchdog, not treated as a failure. |
| **The learned head is zero-initialised** (`shift_head`, or `residual_head` in the fused arm), so the model does nothing at step 0. | That is what makes step 0 *exactly* the closed-form vote, which is what the untrained floor and every paired comparison are defined against. It is asserted to 1e-6. Do not "helpfully" initialise it. |
| **Absent channels are exact zeros with a mask**, never imputed. | Fabricating a channel would hand the model information no deployment has. Principle 4. |
| **`adaptation_v1` still exists** alongside v2. | Kept for comparability with pre-pivot numbers. Different protocol names stop the assembler mixing rows. |
| **Retired baseline adapters (`halo`, `halo_compact`, `halo_evidence`) are still registered.** | They reproduce the frozen classification results. They are retired from the *table*, not from the repo. |
| **`hapt` raises if requested for evaluation.** | Same 30 subjects as `uci_har`, which is in training. This is a leak guard, not an oversight. |

### Known defect NOT to confuse with the above

`baselines/halo_compact/adapter.py` passes `stream.mask` straight into `encode_dataset_detailed`,
which requires the canonical 6-slot layout. It fails on accelerometer-only streams — `monipar`'s
native alignment in particular. This is a *real* bug. It was fixed in `halo_compare` via the
`_six_slot` helper and deliberately left alone in `halo_compact`, which is being retired. Fix it
there only if that adapter is needed again.

---

## 3. Which code is live, and which is a previous life

| Line | Status | Where |
|---|---|---|
| **IMWUT comparison** (this branch) | **LIVE — this is the work** | `data/scripts/curate/compatibility.py`, `model/evidence/comparator.py`, `training/compare/`, `baselines/halo_compare/`, `eval/compare_cost.py`, `eval/enrollment_protocol.py` |
| Classification / evidence-engine | **superseded**, kept because it still produces the baseline table and the frozen results | `model/evidence/{engine,evidence_reranker,patch_retrieval,relational_decoder,confidence}.py`, `training/evidence/`, `training/tokenizer/`, `baselines/{halo,halo_compact,halo_evidence}/` |
| Application / motion-monitoring (Tasks 1–3) | **not the paper target**, but actively maintained by another agent on `main` | `applications/motion_monitoring/`, `docs/tasks/` |

**Do not delete the superseded code.** The classification stack still encodes evaluation streams,
still fits the ConSE bridge for closed-vocabulary baselines, and still reproduces the numbers quoted
in `docs/results/classification/`. It is superseded as a *design*, not as infrastructure.

**Do not touch `applications/`** unless asked. Another agent works there concurrently; unexpected
edits in this repository are usually that agent, not corruption.

**`docs/results/classification/` is frozen.** It is the provenance trail for numbers the paper
cites. Never edit those files; add new results elsewhere.

---

## 4. The model, end to end

1. **Draw an episode** (`training/compare/sampling.py`). Pick a query recording; look up its
   acquisition key; then choose same-subject or cross-subject support from distinct physical
   executions. The answer is always one of 2–14 candidate labels. With probability `p` the
   candidates have enrolled support; otherwise every candidate is withheld from support and
   compatible examples of other labels form unbound background. Draw at most one window from each
   support execution and balance support across labels.
2. **Encode** the query and all support recordings together in one forward pass. Support is encoded
   **per query, with gradients** — nothing is cached, so the comparison itself trains the encoder.
3. **Centre** each episode on its own mean feature (default), so only how rows *differ* can drive
   anything downstream.
4. **Compare** (`model/evidence/comparator.py`, `--comparator-readout sensor_only`, design of
   record 2026-09-07). One set-attention sequence per query over `[query row | support rows]` —
   **signal vectors only**. No retrieval, no top-k, and no label or candidate text in the learned
   path: sensors are compared with sensors, labels with labels, paired by index in the vote. The
   sensor description is already fused into every encoder feature, so it re-enters here only for
   Arm B (`use_descriptor`), where support may mix configurations; in Arm A it is constant within
   an episode. The `fused` readout (label text fused into support tokens, residual per candidate)
   is the ablation arm.
5. **Vote and correct.** A fixed query/support cosine softmax weights each support row. An enrolled
   row votes for its bound candidate; unbound background votes through label-text cosine. The
   attention stack emits one learned scalar per support row that shifts its logit before that
   softmax — the model learns which examples to trust, nothing else. There are no label-specific
   learned parameters. In a zero-shot episode the shift still reweights the background rows; the
   label-to-candidate relation itself is the frozen text cosine and is never learned.
6. **Loss.** Both regimes use cross-entropy on the true candidate. The batch mean is the sole
   objective, so `p` directly controls the relative training mass of zero- and few-shot episodes.

At evaluation, k ≥ 1 uses one equally weighted pooled vector per enrolled execution, with **no
corpus bank**. k = 0 is an **ensemble of 8 training-shaped draws** (4 seen labels × 8 independent
executions = K 32) with every candidate label excluded from support, combined before the argmax.
The primary external comparison is the fixed-cell, cross-subject/same-configuration k=1..8 curve.
Same-subject and cross-configuration cells are reported separately; the k=16 point uses a separate
high-support cohort and is never joined to the main curve. TNDA-HAR has no participant identifiers
in its released bundle, so its enrollment row is subject-unattributed rather than same-subject.

The full design of record, including every ablation, is
[`docs/design/IMWUT_COMPARE_DESIGN.md`](design/IMWUT_COMPARE_DESIGN.md). **When this file and that
one disagree, that one wins** — it is the design of record and this is the orientation.

---

## 5. Standing conventions — violate these and the work is worthless

These are project rules, not preferences. Several exist because they were violated before and cost
a result.

1. **Never launch a training, control, or evaluation run without the user's explicit go.**
   "Implement" means build + tests + a short smoke on real data. Nothing longer.
2. **Two data roles only: training and evaluation.** The training corpus has a deterministic
   subject-held-out split used for health telemetry and writes `best_internal.pt`, but the declared
   result checkpoint remains the final fixed-budget step. Never select a checkpoint or tune a
   hyper-parameter on evaluation data.
3. **Paired-gain rule.** Report a run as `score(trained) − score(its own step 0)`. The validation
   draw once contributed five times the run-to-run scatter of the thing being measured, so a raw
   number carries the draw, not the method. **Paired gain is only valid when both arms share the
   same step-0 function** — an architecture change (centering, front end) moves step 0, so those
   arms are compared by raw score at matched seeds instead.
4. **Every learned component is guilty until a control clears it.** Minimum: the untrained floor and
   the paired step-0 control; add a scrambled-vocabulary control wherever the language channel is
   involved. On this codebase almost every learned addition has been net-negative at first pass.
5. **Never fabricate a data channel.** Absent channels are exact zeros with a validity mask, never
   imputed, never randomly filled.
6. **"Unsupported" is a result.** When a dataset cannot answer a test, report that. Do not widen a
   relation or substitute a different mechanism to manufacture a number.
7. **The interpreter is `/home/alex/code/HALO/legacy_code/.venv/bin/python`**, run from the repo
   root. There is no venv in this repository.
8. **Another agent edits this repo concurrently.** Check `git status` before staging; never
   `checkout`/`reset` over someone else's work.
9. **Do not commit or push unless asked.**
10. **The GPU is shared and has 24 GB.** Estimate cost before proposing a run, and emit progress.
11. **The training step is still partly CPU-bound.** The trainer vectorises
    the episode indexing (`LabelTextTable`, gather-based `split_encoded`/`episode_text`), uses
    the fused AdamW, and prefetches upcoming steps in forked worker processes (`PrefetchLoader`,
    `--loader-workers`, default 4; steps are seeded per step by `episode_rng(data_seed, step)` so
    the episode sequence is identical for any worker count and across resume). The September 7
    full-corpus probe measured about 24 ms/step after the CPU preparation improvements; see the
    bounded measurements below rather than extrapolating older capped-corpus timings. Do not
    reintroduce thread pools for loading (the per-window work holds the GIL; 8 threads were
    2.5-3.5x slower than none) and do not fork workers after threads exist (deadlock).

### Training performance check (2026-09-07)

RTX 4090, fixed frontend, 8 episodes/step, up to 32 support windows/episode, neutral acquisition
text, native-rate full training corpus (1,744,767 eligible windows), BF16 and fused AdamW.
CPU threads limited with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`. No objective, support budget,
label pool, learning rate, or precision policy changed in this optimization pass.

| Probe | Workers | Timed steps after warmup | Mean step | Mean loader wait |
|---|---:|---:|---:|---:|
| Corrected code before optimizations | 2 | 16 | 37.83 ms | 21.96 ms |
| Corrected code before optimizations | 4 | 16 | 28.33 ms | 13.18 ms |
| Optimized sampler, copies, and collation | 2 | 64 | 32.08 ms | 16.31 ms |
| Optimized sampler, copies, and collation (default) | 4 | 64 | 23.83 ms | 9.38 ms |
| Same optimized code, experimental synchronous pinning | 4 | 64 | 25.00 ms | 10.25 ms |

These short, warm-cache probes include forward/backward, clipping, optimizer update and GPU
synchronization, but exclude startup, normal telemetry/checkpoint writing, and validation.
Timings vary with host contention; these are not full-run completion measurements. At 24-32 ms,
35k optimizer steps alone extrapolate to 14-19 minutes; budget roughly 20-30 minutes including
overheads, then use the actual run telemetry to refine the estimate. Peak allocated GPU tensor
memory in these probes was 0.21 GiB (not total CUDA/process memory). Low VRAM does not imply that
increasing the episode batch is a semantics-preserving speed optimization.

Kept: reuse read-only support-row lists (copy only cross-key merges), owned NumPy-to-tensor
window copies, reuse patch boundaries by shape/rate within a batch, fill small metadata in NumPy,
four process workers, and avoid waiting for unused queue buffers at normal shutdown.
Dropped: synchronous pinned transfers, since no speedup was measured. No additional background
pipeline or GPU stream scheduler was added.

Episode hashes and final losses matched across worker counts and the transfer probes; scalar
reference tests check exact collation equivalence, including fractional native rates and short
tails. The final correctness/optimization suite passed 188 tests in 22.56 seconds, including
the worker-shutdown regression. A real three-step trainer smoke completed forward/backward,
validation, checkpoint writing, and shutdown with finite gradients and normalized support votes.

Reproduce a bounded probe (no checkpoints or full training):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m training.compare.profile --workers 2 4 --steps 80 --warmup 16 --out /tmp/compare-profile.json
```

---

## 6. Datasets: what trains, what evaluates, and why

**Training — 18 datasets, 56 streams, 31 acquisition configs, 166 labels.**
`deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS`.

**Evaluation — 10 datasets, manifest `adaptation_v2`** (69 cells, 48 usable, fingerprint
`5058cf41…`). `adaptation_v1` (7 datasets) is kept intact for comparability but is **not** the
current protocol; the two carry different protocol names so the assembler cannot mix their rows.

- **44 of the 63 candidate labels** in the `adaptation_v2` roster are **never seen in training**
  by concept (48 by exact string) — that is the open-vocabulary story. Count by concept, not by
  string: `climbingup`, `climbingdown`, `ascending_stairs`, `descending_stairs` are the training
  labels `walking_upstairs` / `walking_downstairs` under another name. Per dataset, in the primary
  cohort: monipar 9/9 and spar 7/7 unseen; usc_had 6/12, ut_complex 5/13, inclusivehar 2/6;
  motionsense, realworld and shoaib 0. (An earlier note said 44 of 55; that counted the 7-dataset
  roster's observed grid labels rather than v2's pre-registered candidate vocabulary.)
- 4 of the 8 original evaluation acquisition configs are **unseen in training** — that is the
  heterogeneity story.
- `motionsense`, `realworld` and `shoaib` were restored on 2026-09-04. They were never unsuitable;
  they were the old *development* split, and that split no longer exists. They fix a confound —
  previously 8 of 11 streams were wrist and the hardest labels sat on exactly the streams with the
  narrowest support pools. **Disclose in the paper** that design decisions in the superseded
  Phase-B line were made while looking at them: no model was trained on them, so this is researcher
  exposure, not data leakage, and the a-priori-constants rule closes the channel by which it would
  otherwise inflate results.
- **`hapt` may never be evaluated.** It is the same 30 subjects as `uci_har`, which is in training.
  `build_manifest` raises if it is requested.
- `upper_limb_use` (4 streams), `motionsense`, and `usc_had` have no exactly config-compatible
  training stream, so their **k = 0 row is unsupported**. Their k >= 1 rows are unaffected. The
  evaluator must record these cells as unsupported rather than aborting or substituting a method.
- The full compatibility audit also reports unsupported `shoaib/phone_belt` and
  `shoaib/watch_wrist_proxy`; `adaptation_v2` skips those streams, so they are not additional result
  cells.

---

## 7. The evidence this design rests on

Every one of these is a measurement from the earlier lines, recorded in
`docs/results/classification/`. They are why the design looks like it does.

| Measurement | Consequence |
|---|---|
| Frozen features separate activities at 84 macro-F1; zero-shot through text sits near 40 | The encoder was never the bottleneck — the bridge was |
| Untrained retrieve-and-vote scored **44.1 like-for-like** (harnet 47.3, ConSE 42.7). The widely-quoted 47.5 was confounded — see §1 | Comparison against exemplars does real work *before* any training, but it did **not** beat harnet. Do not repeat the 47.5 claim |
| Retrieval ranked by **acquisition configuration**, ×7.0 lift, not by activity | Delete the retrieval stage; and centre the episode, since configuration is a common mode |
| Learned retrieval was worth **+0.0000** paired | The stage is not worth its complexity |
| The attention mixer gained +0.12, and a scrambled-vocabulary control inverted it | The gain is *semantic*; keep language in the loop |
| Asking CE to predict an answer omitted from the candidate roster pushed zero-shot **below chance** (16.18 → 9.44) | Hold out the answer's support, not the answer itself; the current one-hot target is then well-defined |
| Every compact checkpoint that ever led the table was trained **from random init** | End-to-end from scratch is the default; warm start is a secondary arm |
| Signal-similarity vs label-name-similarity: r ≈ 0.11 | Zero-shot is structurally capped; headline on k ≥ 1 and disclose k = 0 |

---

## 8. Decided and closed

- Arm A (core) sees **no acquisition text at any stage**; Arm B (experiment) has text ON with
  unfiltered support. Separate pretrains per arm; the trainer refuses a mismatched checkpoint.
- Compatibility is **binary**: `are_compatible` (identical key) and `is_near_miss` (same device
  family, equivalent site). There is deliberately **no graded distance ladder**.
- Sampling rate is **not** part of the acquisition key — the filterbank is rate-invariant by
  construction, so rate is nuisance within a configuration.
- Laterality is preserved but wording is not: `"the left wrist"` and `"left wrist"` are one site,
  while left and right wrist are a near miss. Non-dominant, affected and unaffected wrists are their
  own sites.
- Constants, fixed a priori: 32 total support executions per training episode, `p = 0.5`, label
  subset 2–14, 35k steps, final-step
  checkpoint, ≥ 3 seeds; zero-shot ensemble R = 8 draws × 4 labels × 8 rows, combiner
  `probability`.
- Centering is **ON by default**; `--no-center-features` is the ablation.
- Fixed filterbank is the design of record; the continuous kernel is the challenger, not the
  default.

## 9. Still open

- **Readout decision 2026-09-07:** `sensor_only` is the design of record (labels never enter the
  learned path; one zero-init shift per support row). The run launched at 14:27 the same day,
  `imwut_compare_arm_a_fixed_35k_20260907`, started minutes before the change and is therefore the
  **fused ablation arm**, not the core. Its checkpoints load as `fused` via
  `comparator_config_from_checkpoint`; a step-0 control for it must be written with
  `--comparator-readout fused`. The core Arm A run has not been launched.

- **Correctness revision 2026-09-07:** support feature/description/label binding is now explicit
  through a fused support-row token; padding cannot dilute the base vote. Zero-shot candidates
  all come from the compatible activity pool, with other labels reserved for background support.
  Training remains the same single-stage end-to-end cross-entropy objective, not Phase-A SSL.
  The zero residual head still receives gradients immediately and upstream comparator modules
  begin learning after its first update. New training and a new paired initial checkpoint are
  required for this architecture. No performance defaults were changed in this correctness pass.
- **Evaluation correctness 2026-09-07:** zero-shot support encoding now preserves valid lengths;
  persistent cache hits restore adapter stream context; cache provenance includes encoder,
  preprocessing and perturbation code. Inapplicable perturbations are N/A cells, and support-side
  perturbations do not masquerade as a changed k=0 corpus bank. Single-window recordings remain
  eligible when execution identity is known; TNDA-HAR is excluded because that identity is missing,
  not because its windows are short. Zero-shot headline coverage is matched across models, and
  variant subject deltas/intervals are kept separate from cell-level deltas with label mode visible.
  Valid lengths and execution provenance enter manifest fingerprints: regenerate the manifest and
  result artifacts before reporting new numbers. Historical baseline tables remain historical,
  not results from this revision. Optimizations and a full evaluation rerun are deferred.
- **No training run has been launched.** Nothing here has evidence of beating its untrained floor.
- **Evaluation audit 2026-09-05, fixed in code but not yet in the frozen artifacts:** (a) the
  adapter's enrolled support rows are now the raw execution mean at the encoder's scale, matching a
  training row (they were L2-normalised while the query was not; `tests/test_compare_eval_parity.py`
  pins parity); (b) the k = 0 support draw excludes candidates by canonical concept, not string;
  (c) `eval.run_adaptation_baselines --variant step0` gives the paired control its own model
  identity, and the assembler refuses two artifacts with one identity (they used to be averaged);
  (d) TNDA-HAR's enrollment cells lack verified execution identity and are refused at manifest
  build and reclassified as unsupported at assembly. **The frozen `adaptation_v2` manifest still
  carries the TNDA-HAR enrollment cells**; rebuilding it changes the fingerprint and requires the
  released-checkpoint baselines to be rerun (about 8 minutes) — batch that with the first HALO
  evaluation rather than spending it now.
- **Evaluation audit 2026-09-05, open protocol questions** (not bugs; decisions): the primary curve
  is same-configuration by construction so it never varies acquisition between query and support;
  same-subject enrollment is supported at k ≥ 2 on three datasets only (the user's call 2026-09-05:
  the paper is cross-subject few-shot HAR, same-subject is a supplement); the headline pools the
  ordinary and specialized regimes and emits no confidence intervals or chance floor — per-dataset
  rows are always kept in `cells.csv` / `dataset_macro.csv`.
- **Heterogeneity-axis experiment: BUILT 2026-09-05, NOT RUN.** `eval.run_adaptation_baselines
  --perturb {rate,channel,orientation,gravity} --perturb-side {query,support}` applies one controlled
  change (`eval/perturbation.py`) to one side of every frozen `adaptation_v2` cell and assembles as a
  `model@<perturbation>` variant; `variant_deltas.json` in the assembled tables is the retention
  (variant minus base) per readout and condition, with paired subject intervals. Query side = the
  §7 figure; support side = the incompatible-exemplar cell for the two arms. **Arm A's deployed
  rule is now in the adapter:** with acquisition text OFF it refuses support whose acquisition
  key differs from the query's (near-miss included) and the runner records an honest native n/a
  while the frozen-feature readouts stay; Arm B attends over anything and records
  `support_compatibility`. Rate and orientation leave the key unchanged (they test what
  "compatible" absorbs); channel and gravity change it (they test the filter itself).
- The GPU budget across nine queued ablation arms does not fit the schedule; the cut has not been
  made.
- The clean released-checkpoint baseline rerun is complete and recorded in
  [`results/IMWUT_BASELINE_ADAPTATION_V2_20260904.md`](results/IMWUT_BASELINE_ADAPTATION_V2_20260904.md).
  HALO comparison-model training and matched evaluation remain open.
- Baseline weight audit for the 2025–26 comparators (Wonderwall, IMUZero, LanHAR, GOAT, MOMENT).
- Whether to spend GPU settling fixed-vs-continuous, or report it open.

---

## 10. How to run it

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python   # from the repo root

# audit the acquisition-key table (read-only)
$PY -m data.scripts.curate.audit_compatibility

# train, end to end from scratch, Arm A. --smoke does three steps on capped real data
$PY -m training.compare.train --out training/compare/outputs/arm_a \
    --device cuda --neutral-acquisition-text --smoke

# `initial.pt` is written automatically after frontend calibration and step-0 validation

# build an evaluation manifest
$PY -m eval.enrollment_protocol --out eval/manifests/adaptation_v2.json.gz

# cost table, on a named device
$PY -m eval.compare_cost --checkpoint <ckpt> --device cuda

# paired step-0 control: same adapter, its own identity (assembles as halo_compare@step0)
HALO_COMPARE_CKPT=<step0.pt> $PY -m eval.run_adaptation_baselines \
    --manifest eval/manifests/adaptation_v2.json.gz --baselines halo_compare --variant step0 \
    --device cuda --out-dir eval/adaptation_results/<run>

# heterogeneity axis: perturb every query (or --perturb-side support) at test time
$PY -m eval.run_adaptation_baselines --manifest eval/manifests/adaptation_v2.json.gz \
    --baselines harnet unimts imagebind normwear halo_compare --device cuda \
    --perturb orientation --perturb-side query --out-dir eval/adaptation_results/<run>

# tests
$PY -m pytest tests -q
```

---

## 11. Document map for this line

| Document | What it owns | Status |
|---|---|---|
| **this file** | orientation, conventions, current state | live |
| [`design/IMWUT_COMPARE_DESIGN.md`](design/IMWUT_COMPARE_DESIGN.md) | **the design of record** — model, support contract, ablations, paper shape | live, authoritative |
| [`research/IMWUT_VENUE_READ.md`](research/IMWUT_VENUE_READ.md) | venue mechanics and the recalibration they forced | live |
| [`design/COMPATIBILITY_AUDIT.txt`](design/COMPATIBILITY_AUDIT.txt) | generated stream → acquisition-key table | regenerate, do not hand-edit |
| [`design/IMWUT_BUILD_PLAN.md`](design/IMWUT_BUILD_PLAN.md) | the 2026-09-03 sweep and schedule | **historical** — superseded in parts |
| [`design/IMWUT_HANDOFF.md`](design/IMWUT_HANDOFF.md) | implementation spec written for another agent | **historical** — the work is done |
| [`results/IMWUT_BASELINE_ADAPTATION_V2_20260904.md`](results/IMWUT_BASELINE_ADAPTATION_V2_20260904.md) | current released-checkpoint zero-shot and enrollment results | live result of record |
| `results/classification/` | pre-pivot measurements | **frozen — never edit** |
| `tasks/`, `docs/data/APPLICATION_DATASETS.md` | the application line | not this paper |
