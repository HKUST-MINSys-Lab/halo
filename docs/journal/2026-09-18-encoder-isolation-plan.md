# Isolating the encoder from the classifier: plan (2026-09-18)

**Status:** plan only. No model code, training or evaluation is changed by this document.
Requested 2026-09-18: attach frozen baseline encoders to our learnable classifier, train only the
classifier, and run both a released-checkpoint variant and a from-scratch-on-our-data variant.

Prerequisite reading: [matched-corpus plan](2026-09-15-matched-corpus-plan.md) (level definitions),
[budget](2026-09-15-matched-corpus-budget.md) (what is built, measured throughput),
[sweep](2026-09-15-matched-corpus-sweep.md) (four defects already fixed).

## 1. Most of this already exists

`MatchedCorpusEncoder` presents a baseline trunk to the support classifier under HALO's encoder
contract, with `--encoder-arch {halo,limubert,harnet,unimts}`, `--matched-pretrained` and
`--matched-d-model`. It is unit-tested (21 tests), smoke-tested, throughput-profiled and rebuilt by
`eval_transfer.build_encoder`, so the sealed evaluator scores it unchanged. **It has never had a
full run.** Everything downstream of the pooled recording vector is byte-identical to the HALO arm.

So the from-scratch-on-our-data variant is an existing, unexercised capability rather than new work.
The frozen variant is the genuinely new piece, and it is small.

## 2. Levels, and which question each answers

| level | encoder | trained on our corpus | our classifier | isolates |
|---|---|---|---|---|
| M0 | released, frozen | no | no (1-NN / fusion readout) | nothing about the encoder; this is the deployment row we published today |
| **M2f (new)** | released **or** random, **trunk frozen** | classifier + projection only | yes | representation quality of a fixed encoder, under our classifier |
| M2 | baseline architecture, random init, trained end to end | yes | yes | the encoder **architecture**, everything else held identical |
| M1 | their architecture, their objective | yes | no | what their method learns from our data (not built, out of scope here) |

The request maps onto M2f for "frozen baseline encoders with our learnable classifier", and onto the
existing M2 for "train it from scratch using our dataset". `--matched-pretrained` supplies the
released-checkpoint variant at either level.

**Why this isolates dataset drift.** Every arm sees the same 223,587 windows from the same eight
training datasets, the same subject holdout, the same sampler, the same episodes, the same loss and
the same sealed manifests. Corpus scale, which is the released models' real advantage and their
main confound, is removed by construction at M2 and M2f-random. It is *not* removed at
M2f-pretrained, which is the point of running both.

## 3. The one real design decision: what "frozen" means

`MatchedCorpusEncoder` is a baseline trunk followed by a trainable projection
(`Linear(out_dim, 2d) → GELU → Dropout → Linear(2d, d)`) and a `LayerNorm`. Freezing the whole
encoder would leave the classifier reading a **randomly initialised** MLP projection of the trunk,
which measures nothing and would quietly hand every frozen arm a crippling disadvantage.

**Proposal: freeze the trunk, train the projection and its norm.** That is standard probing
practice, it keeps the frozen arms comparable to each other, and it leaves the classifier the same
input width and conditioning it gets everywhere else. The projection is ~0.1 M parameters against
trunks of 0.1–5.3 M.

This needs a new flag, because `--freeze-encoder` freezes everything. Proposed `--freeze-trunk`,
mutually exclusive with `--freeze-encoder`, recorded in the trajectory and the run config.

## 4. Code gaps, all small

1. **`--freeze-encoder` is rejected without `--phase-a` or `--resume`** (`train.py`, argument
   validation). A frozen released baseline has neither. The check must allow
   `--encoder-arch != halo`.
2. **No `--freeze-trunk`.** Add it; it sets `requires_grad_(False)` and `eval()` on
   `encoder.net` only, leaving `proj` and `row_norm` trainable. Dropout inside the frozen trunk
   must be off; dropout in the projection stays on.
3. **Parameter accounting and telemetry** currently report encoder parameters as one number.
   Split into trunk versus adapter, and report trainable versus total, or the frozen arms will look
   like 5 M-parameter models when 0.1 M are learning.
4. **Optimizer groups** put every encoder parameter in one group at `lr * encoder_lr_scale`. With a
   frozen trunk the only encoder-side parameters are the projection; confirm it lands in a group
   with a sensible rate rather than inheriting an encoder scale meant for a full trunk.
5. **`--matched-pretrained` provenance.** Each trunk loads released weights internally. Record the
   resolved weight file and its SHA-256 in the run config, as the released-baseline adapters do, so
   a pretrained arm is reproducible and auditable.
6. **`--encoder-arch` is not accepted by the contextual head path** in any special way, but the
   contextual classifier takes `input_dim`; confirm `ContextualClassifierConfig(input_dim=...)` is
   set from the encoder width when the two differ. Today both are 128, so this is a latent trap.

None of these touch the residual classifier, the sampler, the manifests or the evaluators.

## 5. Proposed matrix

Sealed and scenario evaluation is identical for every arm and uses the manifests already fixed.
Times are the 2026-09-15 measurements scaled by 1.53, the ratio between the HALO reference then
(34 min) and the residual arm we actually ran today (52 min) under the heavier current curriculum.
They must be re-profiled before launch, not trusted.

**Tier 1, the decisive set (about 5 GPU-hours plus evaluation).**

| arm | encoder | init | trunk | est. train |
|---|---|---|---|---|
| A (have) | HALO | random | trained | done, 52 min |
| B | HALO | our residual-arm checkpoint | frozen | ~35 min |
| C | harnet5 | released | frozen | ~50 min |
| D | LiMU-BERT | released | frozen | ~65 min |
| E | harnet5 | random | trained | ~76 min |
| F | LiMU-BERT | random | trained | ~98 min |

B versus A separates the classifier from the encoder within HALO. C and D versus E and F separate
released pretraining from architecture. A versus E and F is the supervisor's question: is our
encoder the contribution, with everything else identical?

**Tier 2, expensive, run only if Tier 1 is ambiguous.** UniMTS frozen (~7 h) and UniMTS end to end
(~11 h); it needs gradient checkpointing to fit at all. NormWear is declined at roughly 280 GPU-hours
for one row, and that number is reported rather than paid.

## 6. Confounds that must be stated in the record

* **LiMU-BERT's released checkpoint is pretrained on MotionSense and Shoaib, two of our six sealed
  test datasets.** Arm D therefore has seen sealed signal, unlabelled. Verify against
  `baselines/limubert_x/citation.json` and the paper, and if it holds, mark arm D as leaked and
  treat arm F as the clean LiMU-BERT row. This also applies to the released LiMU-BERT-X rows already
  in the results record.
* **Arm B is favourable to HALO** and is not symmetric with C and D: HALO's frozen encoder was
  trained on-task with this very classifier, while theirs was trained on their own objective. B
  answers "how much does our classifier add to our own encoder", not "whose encoder is better".
  The symmetric comparison is A versus E and F.
* **Input contracts are deliberately not matched.** Each trunk gets the rate, window and channels its
  authors specify. Feeding a baseline something it was never designed to read would measure our
  adaptation rather than its architecture.
* **Two of 31 training streams are gravity-removed** and are fed to every arm rather than dropped,
  so the corpus stays identical across arms. This mildly disadvantages the gravity-dependent
  accelerometer trunks and is disclosed rather than corrected.
* **Capacity is reported, never matched.** Trunks span 0.1 M to 5.3 M parameters.

## 7. Acceptance tests before any full run

1. A frozen trunk receives no gradient and its parameters are bit-identical before and after an
   optimizer step, while the projection, the classifier and the temperatures do change.
2. Freezing changes nothing else: with the trunk frozen and the projection also frozen at fixed
   weights, the pooled vectors match an unfrozen forward on the same batch exactly.
3. The frozen trunk is in eval mode, so its dropout and any normalisation running statistics do not
   move during training.
4. A pretrained arm records a resolved weight path and digest; a random arm records that it has none.
5. Parameter accounting reports trunk, adapter, classifier, and trainable-versus-total separately,
   and the sealed evaluator's `parameters_m` agrees with the trainer's count for the same checkpoint.
6. `eval_transfer.build_encoder` round-trips a frozen-trunk checkpoint and reproduces its training
   forward bit-exactly, so sealed scoring cannot silently differ from training.
7. A three-step CUDA smoke per arm, plus a scenario smoke at k=0 and k=1, both completing with zero
   failed rows, exactly as the contextual head was gated today.

## 8. Order of work

1. Fix the six gaps in section 4, with the tests in section 7. No run yet.
2. Profile each Tier 1 arm for 150 steps and replace the estimates above with measurements.
3. Launch Tier 1 sequentially, evaluating each arm as it finishes, as the three current arms were.
4. Promote a single table that reports trainable parameters beside every row, with the leakage and
   asymmetry disclosures from section 6 attached to the specific rows they affect.
