# Experiment Roadmap

Last verified against code: 2026-09-18.

This is the single forward-looking experiment record for the active support-conditioned HAR
system. It separates completed exploratory work from experiments that can produce a current result.
The architecture and protocol are defined in
[`design_of_record.md`](../contracts/design_of_record.md) and
[`evaluation_protocol.md`](../contracts/evaluation_protocol.md); this document defines the order
in which claims are tested.

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
| 3 | HALO frozen encoder / residual classifier | no | yes | Test classifier reasoning independently of encoder adaptation. |
| 4 | HALO end-to-end / residual classifier | yes | yes | Test the complete system against the matched neighbors control. |
| 5 | released encoder / HALO classifier | no | yes | Isolate classifier value from HALO encoder value under both aggregate and scenario manifests. |

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

The implemented learned head is the v3 residual classifier, normally using one shared parameter
set. The earlier two-head token mixer is a historical architecture, not the current default. A
candidate-specific mixture remains a proposal in the
[2026-09-17 contextual classifier plan](../journal/2026-09-17-contextual-classifier-plan.md).

That proposal contextualizes query, support, support-label and candidate tokens, then combines
semantic support voting and direct query/candidate scoring with a candidate-specific learned
mixture. It has not been implemented or evaluated and is not part of the active contract.

The necessary comparison is always the same encoder and the same manifest under `neighbors` versus
`residual-classifier`. A learned classifier that does not exceed the neighbor control is not promoted as a
useful component, even if its absolute score is high.

## Reporting Rules

* Use `k = 0, 1, 2, 4, 8, 16, 32, 64, 128` when a sealed stream can form the requested
  execution-disjoint enrollment. `k=0` and `k>0` are separate information conditions.
* Report macro F1, balanced accuracy, subject bootstrap intervals, candidate count, and query
  count per sealed dataset/stream before any aggregate.
* Baselines use author-released checkpoints. On the complete-enrollment curve at `k>0`, show
  equal-weight normalized fusion and 1-NN; every candidate receives `k` supports. At `k=0`, show
  equal-weight normalized fusion and a separate released-native row where available. Scenario
  tables use equal-weight normalized fusion as the default baseline readout. Unsupported native
  cells are `N/A`.
* Every result artifact contains macro F1, balanced accuracy, and accuracy. Partial-enrollment
  artifacts also contain truth-enrolled, truth-unenrolled, combined, and harmonic-mean summaries.
* Record commit, checkpoint hash, corpus fingerprint, immutable manifest fingerprint, runtime,
  peak memory, and whether the encoder/head were frozen.
* A historical table is comparable only to another result with the same representation checkpoint,
  source roster, candidate policy, enrolled-execution definition, manifest, and metric. Otherwise
  it is an ablation reference, not a numeric baseline.

## Active curriculum experiment

The 2026-09-17 [heterogeneity and metadata audit](../journal/2026-09-17-heterogeneity-metadata-audit.md)
separates deployment heterogeneity from nuisance augmentation and from metadata/text-backend
ablations. Complete the source-label/device/exposure checks before making stronger unseen-domain
claims. Baseline-encoder plus HALO-classifier training remains planned but explicitly deferred.

The staged deployment-challenge experiment is specified in
[`curriculum.md`](../contracts/curriculum.md).
It first changes acquisition/enrollment episode construction, then adds truthful rate/modality
perturbations and an evidence-dependent semantic gate. Explicit acquisition tokens are deferred.

## Promotion Gate

Add a run to `docs/results/RESULTS.md` only after the sealed evaluator produced per-stream machine
readouts from a frozen checkpoint. Internal validation selects training checkpoints and detects
failures; it is never a test result.
