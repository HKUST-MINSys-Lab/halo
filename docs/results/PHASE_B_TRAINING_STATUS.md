# Phase-B Training Status

> **Canonical empirical record for Phase-B development.** The behavioral contract is in
> [`PHASE_B_TRAINING_INTENT.md`](../design/PHASE_B_TRAINING_INTENT.md); commands are in
> [`training/evidence/README.md`](../../training/evidence/README.md). This file owns completed-run
> results and their interpretation.
>
> Last updated: 2026-08-11. The sealed external test roster has **not** been consumed.

## 1. Current Verdict

The v22 clean arbitrary-label experiment is a **clear development-level adaptation success**.
Unlike the preceding coherent-only run, the learned evidence engine passed every internal mechanism
gate and was selected instead of the closed-form fallback. On genuinely held development datasets,
performance rises consistently as labeled support is added. Removing support or shuffling its
episode-local names destroys that gain.

The result does not establish a final claim. It uses one Phase-B seed, the coherent zero-support path
is weak, and prototype/ridge controls remain several F1 points stronger. The correct next question is
how to preserve zero-support semantics while closing the remaining gap to direct support methods.

## 2. Active Training Recipe

| setting | value |
|---|---:|
| Phase-A tokenizer | frozen |
| optimizer | AdamW |
| steps | 3,000 |
| independent episodes per step | 8 |
| queries per episode | 8 |
| candidate count | uniform integer from 2 through 16 |
| support per candidate | uniform integer from 0 through 8 |
| k=0 label presentation | coherent activity names |
| k>0 label presentation | fresh neutral aliases per episode |
| signal views | clean stored patch embeddings |
| objective | candidate-set cross-entropy only |
| evidence budget | 64 patch rows |
| learning rate | `2e-4`, 300-step warmup, cosine decay |
| tokenizer fine-tuning | disabled |
| seed | `20260725` |

Every candidate receives the same `k`. A positive-support episode assigns a one-to-one random name
such as `protocol amber` to each candidate and to that candidate's support rows. The assignment is
redrawn for every episode. Retrieval remains learned and query driven; no support row is manually
inserted into top-k evidence.

The run processed 24,000 independent episodes and 192,000 query windows in 601 seconds on the local
RTX 4090. It completed without non-finite values, dead gradient paths, or clipping instability.

## 3. Internal Checkpoint Selection

Step 1000 was selected as the learned relational decoder. It passed all declared requirements:

- learned low-k performance exceeded the closed-form vote;
- support presence improved prediction;
- removing support reduced correct-label probability;
- shuffling support labels reduced correct-label probability.

Fixed held-family C=8 balanced accuracy at the selected checkpoint:

| support per candidate | learned engine | identity retrieval vote |
|---:|---:|---:|
| 0, coherent | 0.142 | 0.338 |
| 1, arbitrary names | 0.350 | 0.259 |
| 2, arbitrary names | 0.360 | 0.301 |
| 4, arbitrary names | 0.400 | 0.354 |
| 8, arbitrary names | 0.455 | 0.389 |

Removing support reduced mean true-label probability by 0.151. Cyclically shuffling support names
reduced it by 0.168. Training and held-family macro BA were nearly identical at selection
(`0.3425` versus `0.3414`), unlike the large gap in the failed coherent-only run.

Later checkpoints retained adaptation but did not improve low-k selection. The final step reached
0.325 low-k BA versus 0.355 at step 1000, while k=8 remained 0.440. This supports checkpointing the
development optimum rather than treating the final optimizer state as authoritative.

## 4. External Development Evaluation

The development roster is MotionSense, RealWorld, and Shoaib. Values below are unweighted means of
available dataset/protocol macro F1 scores. Cross-subject cohorts use genuinely different recorded
people, not synthetic subject transformations.

### Arbitrary-label full enrollment

| relation | k | learned engine | identity vote | prototype | ridge | support removed | labels shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| same subject | 1 | 74.51 | 78.94 | 82.82 | 79.66 | 42.65 | 14.54 |
| same subject | 2 | 77.66 | 80.20 | 80.75 | 79.35 | 42.65 | 14.79 |
| cross subject | 1 | 49.87 | 50.22 | 54.84 | 54.49 | 14.27 | 11.96 |
| cross subject | 2 | 55.29 | 56.70 | 59.40 | 58.83 | 14.27 | 11.01 |
| cross subject | 4 | 61.29 | 64.71 | 63.73 | 64.09 | 14.27 | 10.10 |
| cross subject | 8 | 65.71 | 69.37 | 68.76 | 68.85 | 14.27 | 9.21 |

Same-subject k>0 averages contain MotionSense and RealWorld; Shoaib has no valid paired same-subject
support cohort. Cross-subject averages contain all three datasets and 17,837 query windows per row.

The intervention columns establish mechanism use. For example, cross-subject k=8 falls from 65.71
F1 to 14.27 when support is removed and to 9.21 when support receives incorrect names. The model is
using both the physical examples and the support-to-name binding.

### Coherent names

| relation | k/shape | learned engine | identity vote | prototype | ridge |
|---|---|---:|---:|---:|---:|
| same subject | k=0 | 31.54 | 33.49 | N/A | N/A |
| same subject | k=1 full | 73.93 | 78.92 | 82.82 | 79.66 |
| same subject | k=2 full | 77.56 | 81.33 | 80.75 | 79.35 |
| cross subject | k=0 | 10.09 | 24.56 | N/A | N/A |
| cross subject | k=1 full | 50.52 | 50.82 | 54.84 | 54.49 |
| cross subject | k=2 full | 55.65 | 56.70 | 59.40 | 58.83 |
| cross subject | k=4 full | 61.63 | 64.78 | 63.73 | 64.09 |
| cross subject | k=8 full | 65.92 | 69.54 | 68.76 | 68.85 |

The nearly identical positive-k coherent and arbitrary-name curves show that enrolled evidence, not
activity-name semantics, drives the successful adaptation path. Coherent partial enrollment also
generalizes despite not being trained directly: cross-subject F1 rises from 35.22 at k=1 to 39.91
at k=8, compared with the 10.09 zero-support floor.

## 5. Comparison With the Failed Minimal Run

The v21 run used coherent names for every episode. It reached approximately 0.94 training BA while
held-out BA stayed near 0.19; support removal and label shuffling were nearly inert. External
cross-subject F1 only rose from 11.86 at k=0 to 14.48 at k=8.

Under v22, positive-support candidate names are arbitrary. External coherent-name cross-subject F1
now rises from 10.09 at k=0 to 50.52, 55.65, 61.63, and 65.92 at k=1,2,4,8. This isolates the former
failure: coherent positive-support episodes allowed direct activity classification and did not make
support binding necessary.

Zero-support quality did not improve. It fell modestly on the external development aggregate
(11.86 to 10.09 cross-subject; 34.04 to 31.54 same-subject) and more substantially on the fixed
held-family canary. Uniform sampling over k=0..8 allocates only about one ninth of episodes to the
semantic path, while every positive-k episode trains arbitrary-name binding.

## 6. Remaining Limitations

1. The learned engine remains about 3-5 macro F1 points below prototype/ridge controls on most
   cross-subject cells. Retrieval and evidence interpretation therefore still leave usable
   representation quality on the table.
2. Coherent k=0 generalization is weak, especially across subjects. Future work should rebalance or
   separate the semantic path without weakening the successful adaptation objective.
3. Configuration-only internal transfer is markedly weaker than subject-only transfer. The current
   external development roster has no genuine cross-configuration enrollment cohort, so that claim
   cannot yet be tested adequately.
4. This is one Phase-B seed selected on development canaries. Replicates are required before opening
   the sealed test roster.
5. Arbitrary labels are intentionally unanswerable at k=0 and are therefore evaluated only when
   support is present.

## 7. Artifact Index

Current v22 run:

- root: `training/evidence/outputs/phase_b_v22_alias_support_20260811/`
- selected predictor: `patch_evidence_predictor.pt`
- resumable final state: `patch_evidence_predictor.last.pt`
- milestone predictors: `patch_evidence_predictor.milestones/`
- complete training log: `train.log`
- raw telemetry and rendered plot: `telemetry/`
- arbitrary-label development result: `eval_alias_dev.json`
- coherent-name development result: `eval_coherent_dev.json`

Superseded runs remain available for forensic comparison:

- coherent-only v21: `training/evidence/outputs/phase_b_v21_minimal_20260811/`
- earlier complex v20: `training/evidence/outputs/phase_b_v20_20260811/`
- historical diagnostics: `training/evidence/outputs/diagnostics/phase_b_20260808/`

## 8. Interpretation Boundary

This result demonstrates learned memory adaptation on held development datasets. It does not yet
establish final unseen-dataset performance, a multi-seed estimate, or superiority over direct
prototype/ridge adaptation. Do not report sealed-test numbers until the remaining development
decisions and replication policy are frozen.
