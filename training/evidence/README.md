# Phase-B Evidence Training

Phase B trains the learned retriever and relational evidence decoder on clean Phase-A patch
embeddings. The canonical motivation and behavioral contract are in
[`docs/design/PHASE_B_TRAINING_INTENT.md`](../../docs/design/PHASE_B_TRAINING_INTENT.md).

## Active Recipe

Each of the eight independent episodes in an optimizer step samples:

- `C` candidate labels uniformly from 2 through 16;
- one support count `k` uniformly from 0 through 8;
- eight query executions; and
- `k` random eligible real support executions for every candidate.

The query execution is excluded from support. Subjects, configurations, and support sources are
otherwise not scheduled. Every candidate receives the same `k`. A k=0 episode uses coherent label
text. Every positive-k episode assigns fresh semantically neutral names to the candidates and their
support, forcing the model to recover the episode-local support-to-label binding. Phase B otherwise
uses clean embeddings, fixed retrieval settings, and candidate cross-entropy only. It does not train
with physical augmentation, synthetic subject characters, partial enrollment, hard distractors,
bootstrap stages, or auxiliary losses.

Support is placed in the episode's allowed memory view. The learned query retriever still has to
select it. Support identities are never used to append or force rows into the evidence roster.

## Commands

Build the memory bank after Phase A:

```bash
python -m training.evidence.build_memory --device cuda
```

Run the synthetic CPU integration test and the real-data smoke test:

```bash
python -m training.evidence.train_patch_decoder --smoke
python -m training.evidence.train_patch_decoder --device cuda --real-smoke
```

Launch the default frozen-tokenizer training:

```bash
python -m training.evidence.train_patch_decoder --device cuda --evidence-budget 64
```

The default compute shape is eight episodes by eight queries. `--episodes-per-step` and
`--queries-per-episode` are explicit profiling controls; there is no ambiguous `--batch` argument.
Label-text variants are disabled by default.

The optional later end-to-end experiment is:

```bash
python -m training.evidence.train_patch_decoder --device cuda \
  --tokenizer-mode ema_finetune \
  --checkpoint training/tokenizer/outputs/phase_a_headline/best.pt
```

## Evaluation

Run the development enrollment protocol first:

```bash
python -m training.evidence.eval_enrollment --device cuda
```

Evaluate the positive-support arbitrary-label condition separately from coherent zero-shot
prediction:

```bash
python -m training.evidence.eval_enrollment --device cuda --random-aliases
```

Use the sealed test roster only after development decisions are frozen:

```bash
python -m training.evidence.eval_enrollment --device cuda --protocol-role test
```

Evaluation reports `k=0,1,2,4,8`, support-removed and label-shuffled interventions, a closed-form
vote over the same retrieved rows, prototype and fitted ridge controls, same- and cross-subject
cohorts where real metadata supports them, and paired subject-bootstrap intervals. Unsupported
cohorts are marked rather than silently substituted.

Internal checkpoint validation uses one coherent C=8 k=0 condition and one arbitrary-name C=8
k=1,2,4,8 curve for each of three transfer folds. A learned checkpoint is eligible only when it
benefits from support presence, uses the support-label binding, and matches the closed-form low-k
control.

## Telemetry

Training updates telemetry roughly once per minute:

```text
training/evidence/outputs/telemetry/patch_evidence_predictor/
```

Render and monitor it without using the training GPU:

```bash
python -m training.evidence.monitor_training \
  --telemetry-dir training/evidence/outputs/telemetry/patch_evidence_predictor \
  --render --watch 60
```

Telemetry includes loss and accuracy by `C` and `k`, validation controls, support recall, evidence
attention mass and entropy, selected-score gradients, row and subspace diversity, candidate-logit
spread, component gradient coverage and RMS, clipping, throughput, and VRAM use. Non-finite losses
or gradients stop training.

Training writes atomic resumable state beside the output as `*.last.pt`. Resume with the same command
and `--resume <state>`. Memory identity, source behavior, fixed validation canaries, and all
trajectory-affecting options are checked before restoration.

The separate confidence-calibration experiment is parked and is not part of the current Phase-B
launch or claim.
