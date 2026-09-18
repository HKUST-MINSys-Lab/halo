# Classifier isolation plan - 2026-09-15

**Status:** completed on 2026-09-15. This is a temporary decision record for the September 15
classifier investigation. Results and reproducible artifacts are in
`training/support_classifier/evaluations/classifier_isolation_20260915/RESULTS.md`; this file can
be folded into the experiment roadmap once the decision is made.

The cached 8-second evaluation completed in 6.1 minutes without retraining. The result bundle
contains raw predictions, aggregate and per-dataset tables, plots, immutable episode manifests,
neighbor-error strata, and a separate k=0 seen-versus-unseen concept summary.

## Question

The learned classifier is justified only if query/support context, label semantics, and token
bindings improve on the same encoder's parameter-free neighbour floor. Determine which learned
score component helps, which causes the high-support regression, and whether the classifier uses
the support-label relationship at all.

## Evidence already available

- Full classifier and centred-neighbour floor at 4, 8, and 16 seconds for every feasible
  `k in {0,1,2,4,8,16,32,64,128}`.
- Raw 1-NN, prototype, and ridge on the identical immutable manifests.
- Neighbour error ranks and support-removal sensitivity at `k=1` and `k=8`.
- Shared versus fully regime-split classifier parameters.
- Per-placement and native multi-device cells. These do not constitute a cross-configuration
  support test because each support still comes from the query's evaluation stream.

## Fast experiment (run first)

Use the promoted residual-classifier checkpoint, existing 8-second feature cache, and immutable
manifests. Do not encode data or train a model. Evaluate:

1. full classifier;
2. centred-neighbour floor;
3. base plus text term only;
4. support-reranking residual only;
5. candidate residual only;
6. both residuals with no text term; and
7. full classifier after a deterministic cyclic permutation of support-label text while preserving
   sensor vectors, vote destinations, masks, pair tags, and candidate labels.

Report dataset-balanced macro-F1 for every k, paired deltas from the floor, and per-dataset deltas.
Parse the training log for lambda, text margin, and residual magnitudes. A delta smaller than one
macro-F1 point is treated as unresolved because enrollment-draw variation is approximately
0.36-0.54 points.

## Decision rules

- Text-only reproduces the high-k loss: extend training through `k=32` and replace the flat `8+`
  text-weight bucket with a smooth support-dependent gate.
- Candidate-only regresses: remove or constrain the candidate residual.
- Support-reranking-only regresses: add an improvement-over-floor objective and shrink unsupported
  corrections toward zero.
- Shuffling support-label text has negligible effect: the attention path is not using the binding;
  do not add more metadata until that credit-assignment problem is understood.
- Components behave correctly but only on seen concepts: add concept-level label holdout and
  paraphrase evaluation before making an open-set claim.

## Follow-up experiments (not part of the fast pass)

In order: concept-level seen/unseen k=0 stratification; controlled semantically hard candidate
sets; partial enrollment; same- versus cross-subject enrollment; and finally matched
cross-configuration supports. The last item requires an explicit acquisition-descriptor input to
the classifier and a carefully defined near-compatible support protocol, so it must not be inferred
from the current multi-device results.

## Decision from the completed experiment

- The direct query-to-candidate text score accounts for essentially all of the learned classifier's
  aggregate behavior. It helps by 2.11 macro-F1 points over the neighbor floor at k=1, but hurts by
  2.90 points at k=8 and 6.03 points at k=128.
- Cyclically shuffling support-label text changes aggregate macro-F1 by at most 0.05 points. The
  checkpoint has not learned a measurable dependence on support-label semantic binding.
- Support and candidate residual branches are each slightly worse than the neighbor floor. Adding
  mixer capacity is therefore not supported by the evidence.
- Error stratification shows why the aggregate curve changes: at k=1 the text path rescues many
  cases where the true support ranks 2--5; at k>=8 it increasingly overrides already-correct rank-1
  sensor decisions.
- At k=0 the learned classifier reaches 23.83 macro-F1 on genuinely unseen concepts versus 18.60
  for the training-bank 1-NN + ConSE bridge. This is a real improvement, but still weak in absolute
  terms and must not be described as the 49.29 all-concept score.

The next classifier iteration should preserve the neighbor floor, train through at least k=32, and
learn a smooth evidence-dependent gate that can decline the semantic prior when sensor evidence is
decisive. Acquisition-aware and support-label-binding claims require new matched challenges before
they can be evaluated; the current protocol does not exercise them.
