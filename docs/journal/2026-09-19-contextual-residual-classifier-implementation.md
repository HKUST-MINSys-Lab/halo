# Contextual residual classifier implementation

**Date:** 2026-09-19  
**Status:** implemented and smoke-tested; no full-duration result yet.

`support_contextual_residual_v1` replaces the active `contextual` training mode. The historical
`support_contextual_mixture_v1` remains strictly loadable so its recorded negative result is
reproducible, but new runs cannot accidentally instantiate it.

The replacement keeps the centered differentiable-neighbor vote as an explicit floor. A set
transformer sees query/support motion, one learned runtime acquisition vector per recording,
support labels, candidate labels, role embeddings, and support-label pair tags. It produces
candidate-specific corrections to support comparisons and a candidate-specific semantic weight;
it never emits unconstrained candidate logits. Separate zero-support and enrolled semantic
projections use the same mechanics and independent weights.

Training adds a generic, modular better-over-reference objective based on true-class log-odds.
The default comparisons are contextual support over the support floor and final prediction over
the better individual branch. References are detached targets; all encoder, conditioner, pooling,
and classifier parameters remain trainable through the improved path.

Verification performed before this record:

- unit tests for floor preservation, zero-support semantics, set permutation invariance, dynamic
  candidate count, complete gradient flow, strict checkpoint restoration, and evaluator plumbing;
- three end-to-end steps on the real eight-source corpus, including validation and checkpoint
  writing; all losses and gradients finite;
- one resumed optimizer step from the smoke checkpoint;
- acquisition vectors reconstructed separately for query and support streams in sealed, partial-
  coverage, scenario, and multi-device evaluation paths.

The smoke is a mechanical test only and is not a result claim.
