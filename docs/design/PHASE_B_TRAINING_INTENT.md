# Phase-B Training Intent

> **Canonical Phase-B motivation and executable contract. Read this before configuring,
> launching, or interpreting Phase-B training.**
>
> Status: implementation-aligned as of 2026-08-11. Phase A is complete. Phase B is intentionally
> using a minimal training protocol so that memory adaptation can be tested before adding more
> difficult episode conditions.

This document owns the reason Phase B exists and the behavior of the active trainer.
`training/evidence/README.md` owns commands. Historical evidence-engine designs and earlier training
runs are retained under `docs/archive/` and `docs/results/`; they are not configuration guidance.

## 1. Thesis

Phase B learns to use a labeled memory as an adaptation mechanism. It is not a classifier with a
fixed output vocabulary. At inference time, the caller provides a set of candidate labels and may
enroll labeled examples for those candidates. The model retrieves patch-level evidence and predicts
one of the supplied labels without fitting a new classifier.

The intended application is personalized activity and rehabilitation monitoring. A clinician may
record a movement a few times and give it a task-specific name. Later recordings should benefit from
those examples without creating and maintaining a separate fine-tuned model for every person.

Phase B cannot recover information absent from the sensor. It also cannot reliably separate two
labels whose physical executions are indistinguishable from the available IMU channels. Candidate
labels constrain the decision; they do not make an unobservable distinction observable.

## 2. Terms

- **Candidate labels:** the labels allowed as answers for one episode.
- **Archive:** the immutable CPU-resident collection of Phase-A patch embeddings and metadata.
- **Active memory:** the rotating subset of archive windows available to the learned retriever.
- **Support:** a labeled execution made available in memory for one candidate.
- **k:** the number of independent support executions made available for every candidate.
- **Evidence:** rows selected from active memory by learned, query-driven retrieval.
- **Query:** the recording whose candidate label the model must predict.

Support and evidence are not synonyms. Support describes what is available in memory. Evidence is
what the learned retriever actually selects. The support identity is retained for telemetry and
intervention tests, but it is not an input to retrieval ranking.

## 3. Minimal Training Episode

Every episode is sampled independently:

1. Draw the number of candidate labels uniformly from 2 through 16.
2. Draw the candidate labels uniformly from labels feasible in the training split.
3. Draw `k` uniformly from 0 through 8.
4. Draw query executions from the selected candidates.
5. For `k > 0`, draw `k` eligible real support executions independently for every candidate.
6. Remove all other rows carrying a candidate's canonical label from that episode's memory view.
7. Retrieve evidence using only learned query-to-memory similarity.
8. Predict the correct candidate using ordinary cross-entropy.

The query execution itself is never eligible as support. When verified event identifiers exist, the
whole query event is excluded. Otherwise the source window is excluded. Support is not constrained
to be from the same subject, another subject, the same placement, or another placement. Those
relations occur according to the real data and are measured rather than scheduled.

All candidates receive the same `k`, so support count cannot identify the answer. At `k=0`, all
candidate concepts are absent from memory and the model receives coherent activity names. It must
use label semantics and transferable background evidence. At `k>0`, the candidates receive fresh,
semantically neutral names such as `protocol amber`. Each candidate concept can enter memory only
through its randomly sampled support. The same episode-local name is attached to a candidate token
and all of that candidate's support rows. The assignment is redrawn independently for every episode.
This removes the direct activity-name shortcut and makes support-to-name binding necessary.

One optimizer step contains eight independent episodes with eight query executions each. Episodes
share model parameters and the active memory table, but not candidate sets, support draws, query
executions, or random state.

## 4. Deliberate Omissions

The current trainer does **not** use:

- signal augmentation in Phase B;
- synthetic subject characters or virtual-person transformations;
- partial enrollment, where only some candidates receive support;
- same-subject or cross-subject episode types;
- hard-distractor mining;
- an easy-to-hard curriculum;
- a decoder bootstrap or retriever freeze;
- a changing retrieval budget or temperature;
- auxiliary retrieval, support-ranking, or counterfactual training losses.

Candidate cross-entropy is the only optimization objective. Support removal and support-label
shuffling remain evaluation interventions, not losses. Label-text variants are disabled by default.

These omissions do not assert that the removed ideas are invalid. They make the first experiment
interpretable. If this protocol fails, augmentation or a more elaborate curriculum cannot be used
to explain away a failure of the basic learned retrieval and evidence-decoding mechanism.

## 5. Memory and Retrieval

The archive stores patch embeddings, not only pooled session embeddings. Each patch retains its
canonical label, subject, dataset/configuration, source window, verified event where available,
sensor, physical time, duration, resolution, and source provenance.

The active memory contains at most 16 source windows per label and refreshes every five optimizer
steps. It balances configurations and subjects so that a large data source does not dominate. It no
longer reserves part of a label's budget for a selected anchor subject.

The deployment and training evidence budget is 64 patch rows. Retrieval uses four learned projected
subspaces. The number selected from each subspace is derived from the evidence budget and number of
query patches. Training and deployment use the same fixed score temperature of 0.07.

Retrieval is entirely learned and query driven:

```text
query patches -> learned projected keys -> hard top-k rows -> unique evidence roster
```

No oracle support row is appended to that roster. If a support row is not selected, it cannot affect
the decoder on that forward pass. Candidate loss reaches selected retrieval scores through their
attention bias. Telemetry therefore reports support recall, support attention mass, score-gradient
direction, roster diversity, and subspace overlap.

## 6. Evidence Decoder

The relational decoder applies self-attention to candidate-label tokens, background-label tokens,
query-patch tokens, and retrieved-evidence tokens. Role embeddings distinguish these token types.
Episode-local binding slots connect selected support evidence to the corresponding candidate without
creating a permanent classifier parameter for that label.

A candidate logit is read directly from its candidate token. There is no prototype or nearest-
neighbor score added to the learned logit. Closed-form retrieval voting, prototypes, and fitted ridge
classifiers are reported only as controls.

Patch vectors, label-text vectors, and retrieval projections are L2-normalized at their boundaries.
Each additive decoder component has a learned positive scale before token-level normalization. The
retrieval score enters attention as a relative log-probability over selected evidence, so a common
cosine offset does not change the result.

## 7. Validation and Success Criteria

Internal validation uses fixed clean C=8 conditions across held-subject, held-configuration, and
jointly held folds. It evaluates coherent k=0 once per fold and an arbitrary-name k=1,2,4,8 curve.
Query executions, candidate identities, and arbitrary-name assignments stay fixed within the
positive-support curve while support grows through nested prefixes.

A useful learned checkpoint must:

1. improve with enrolled support;
2. lose correct-label probability when true support is removed;
3. lose correct-label probability when support labels are shuffled; and
4. match or exceed the closed-form vote over the same retrieved rows at low k.

External evaluation additionally reports prototype and ridge controls, genuine same-subject and
cross-subject cohorts where source metadata supports them, and per-subject uncertainty intervals.
Internal validation matches the two training conditions: coherent k=0 and arbitrary-name
k=1,2,4,8. External evaluation reports coherent names and arbitrary names separately. Physical
perturbations remain an out-of-distribution stress test and must be described as such.

The first scientific question is narrow: can the learned retriever and decoder use randomly enrolled
clean examples at least as well as a simple vote over the rows they retrieve? If not, the next step is
to simplify or revise the evidence engine, not to add curriculum complexity.

## 8. Tokenizer Policy

The default freezes the Phase-A tokenizer and uses the clean stored patch embeddings that built the
archive. This isolates Phase-B learning. The code retains an explicit `ema_finetune` experiment for
later end-to-end work: retrieval keys come from a detached EMA encoder and selected raw rows are
re-encoded through the online tokenizer for gradients. That mode is not the current headline run.

## 9. Telemetry and Artifacts

Training writes an atomic telemetry snapshot roughly once per minute and a run-specific JSONL
history. It reports loss, accuracy, candidate and support-count strata, learned-versus-control
validation, support retrieval recall, retrieval-score gradients, evidence diversity, attention
concentration, component gradients, clipping, throughput, and VRAM use.

Predictor checkpoints record the complete episode policy, objective, retrieval configuration,
memory-bank fingerprint, source fingerprint, and fixed-canary fingerprint. A state from an earlier
training regime cannot be resumed under this protocol.

## 10. Source of Truth

- training policy: `training/evidence/policy.py`;
- episode construction and support sampling: `training/evidence/patch_episodes.py`;
- training loop and objective: `training/evidence/train_patch_decoder.py`;
- learned retrieval: `model/evidence/patch_retrieval.py`;
- evidence decoder: `model/evidence/relational_decoder.py`;
- external enrollment evaluation: `training/evidence/eval_enrollment.py`;
- commands and telemetry paths: `training/evidence/README.md`.
