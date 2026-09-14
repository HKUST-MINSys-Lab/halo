# Experiment Roadmap

This is the single forward-looking experiment record for the active support-conditioned HAR
system. It separates completed exploratory work from experiments that can produce a current result.
The architecture and protocol are defined in `DESIGN_OF_RECORD.md` and
`EVALUATION_PROTOCOL.md`; this document defines the order in which claims are tested.

## Question

Can a heterogeneous physical-time encoder and support-conditioned semantic classifier improve
adaptation beyond a simple nearest-neighbor rule?

The answer requires separating representation quality from classifier reasoning. A better score
from a learned head alone is not evidence of a better encoder; a better encoder score alone does
not establish that semantic mixing is useful.

## Established Exploratory Evidence

The following work was useful for choosing experiments, but is not a promoted result under the
current sealed protocol:

| finding | historical artifact | current use |
|---|---|---|
| End-to-end differentiable-neighbor training substantially improved the encoder over random initialization. | `IMWUT_DIFFERENTIABLE_NEIGHBORS_20260908.md` | Retain neighbors as the encoder-only control. |
| Future-JEPA improved frozen representations but not the end-to-end adapted route. | `journal/2026-09-13-jepa-value-measured.md` | Retired; preserve as a negative result, not an active arm. |
| Earlier learned support engines did not consistently beat the same encoder with 1-NN. | `IMWUT_MATCHED_ADAPTATION_20260908.md` | Require a direct same-encoder neighbor comparison for every new semantic head. |

Those experiments used earlier source rosters, candidate ranges, manifests, and classifier designs.
They may be cited as ablation history, but their scores must not be pooled with current sealed-test
results or presented as current baseline comparisons.

## Current Experiment Ladder

Each row uses the current disjoint rosters and the immutable manifest produced by
`training.support_classifier.sealed_eval`.

| order | condition | encoder updates | head updates | purpose |
|---:|---|---|---|---|
| 0 | released external baselines | none | none | Establish released-checkpoint 1-NN, prototype, ridge, and native `k=0` reference rows. |
| 1 | HALO initialized / neighbors | none after initialization | none | Random-encoder and implementation floor. |
| 2 | HALO supervised end-to-end / neighbors | yes | none | Measure what direct enrollment training can teach the encoder without semantic-head capacity. |
| 3 | HALO frozen encoder / token mixer | no | yes | Test classifier reasoning independently of encoder adaptation. |
| 4 | HALO end-to-end / token mixer | yes | yes | Test the complete system against the matched neighbors control. |

The one-second fixed filterbank is retained as the compact architectural control. The current
multiresolution filterbank is the only planned extension. Do not add a new frontend until this
ladder identifies a specific representation failure.

## Retired JEPA experiment

Future-JEPA is not in the active ladder. Its frozen-versus-adapted comparison is recorded in
[2026-09-13-jepa-value-measured.md](../journal/2026-09-13-jepa-value-measured.md): it improved a
frozen representation, but the gain disappeared once the supervised neighbor objective adapted the
encoder. The code and exact result rows remain reproducible but must not be used as current model
references.

## Classifier Experiment

The active token mixer has two parameter sets and one shared encoder:

* **Zero-shot (`k=0`):** query plus candidate-label tokens, followed by query/candidate cosine
  scoring.
* **Enrollment (`k>0`):** query, all support recordings, paired support-label tokens, and all
  candidate-label tokens. Role embeddings and pair/candidate tags identify the relationships;
  attention contextualises the full set, then refined query/support cosine scores vote to labels.

The necessary comparison is always the same encoder and the same manifest under `neighbors` versus
`retrieve-mix-vote`. A token mixer that does not exceed the neighbor control is not promoted as a
useful component, even if its absolute score is high.

## Reporting Rules

* Use `k = 0, 1, 2, 4, 8, 16, 32, 64, 128` when a sealed stream can form the requested
  execution-disjoint enrollment. `k=0` and `k>0` are separate information conditions.
* Report macro F1, balanced accuracy, subject bootstrap intervals, candidate count, and query
  count per sealed dataset/stream before any aggregate.
* Baselines use author-released checkpoints. At `k>0`, show 1-NN, prototype, and ridge. At `k=0`,
  show only a declared published native scoring path; unsupported cells are `N/A`.
* Record commit, checkpoint hash, corpus fingerprint, immutable manifest fingerprint, runtime,
  peak memory, and whether the encoder/head were frozen.
* A historical table is comparable only to another result with the same representation checkpoint,
  source roster, candidate policy, enrolled-execution definition, manifest, and metric. Otherwise
  it is an ablation reference, not a numeric baseline.

## Promotion Gate

Add a run to `docs/results/RESULTS.md` only after the sealed evaluator produced per-stream machine
readouts from a frozen checkpoint. Internal validation selects training checkpoints and detects
failures; it is never a test result.
