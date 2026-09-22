# Experiment Roadmap

> **Superseded 2026-09-22.** This roadmap's question — can a learned support-conditioned classifier
> beat a simple nearest-neighbour rule — was answered by v4 (see `docs/results/RESULTS.md`) and is now
> regime 3 of a three-regime plan. The live plan is [`docs/overview/roadmap.md`](../overview/roadmap.md);
> the decision record is [the 2026-09-22 journal entry](../journal/2026-09-22-three-regimes-and-unsupervised-adaptation-decisions.md). Kept for reproducibility only.

Last verified against code: 2026-09-19.

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
| 3 | HALO frozen encoder / evidence-aware v2 classifier | no | yes | Test classifier reasoning independently of encoder adaptation. |
| 4 | HALO end-to-end / evidence-aware v2 classifier | yes | yes | Test the complete system against the matched neighbors control. |
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

The active experiment is `support_evidence_aware_v2`. It computes auditable support and semantic
status-quo paths before contextualization, then learns candidate-specific support corrections and
semantic reliance. It is mechanically validated but has no full-duration result yet. The promoted
control remains `support_classifier_v3`. Both `support_contextual_mixture_v1` and
`support_contextual_residual_v1` are abandoned negative results; the two-head token mixer is
retired. They remain loadable only for historical reproduction. The current implementation record
is [the 2026-09-19 handoff](../journal/2026-09-19-new-classifier-design-handoff.md), followed by the
[second-review fixes](../journal/2026-09-19-evidence-aware-v2-second-review-fixes.md).

The necessary comparison is always the same encoder and the same manifest under `neighbors` versus
`evidence-aware-v2`. A learned classifier that does not exceed the neighbor control is not promoted as a
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
It changes acquisition/enrollment episode construction and can add truthful rate/modality
perturbations. The current classifier consumes the encoder's single pooled runtime-acquisition
vector and trains with modular path-improvement objectives; explicit per-field classifier tokens
remain unnecessary unless the pooled contract fails an ablation.

## Promotion Gate

Add a run to `docs/results/RESULTS.md` only after the sealed evaluator produced per-stream machine
readouts from a frozen checkpoint. Internal validation selects training checkpoints and detects
failures; it is never a test result.
