# Baseline comparison policy

Last verified against code: 2026-09-24.

The goal is a useful deployment comparison, not an attribution claim across unmatched upstream
corpora. Each retained encoder keeps its author-released weights and published input preprocessing.

For an eligible query/support episode:

1. Build the recording split once, before embedding any method.
2. Feed the same recordings to every adapter, preserving each adapter's required channel order,
   units, rate conversion, crop, padding, and normalization.
3. Apply the same support/candidate episode manifest to every exported representation.
4. On the complete-enrollment aggregate curve, report equal-weight normalized fusion and cosine
   1-NN where `k >= 1`. Every candidate receives exactly `k` supports. Prototype and ridge are
   optional diagnostics rather than headline rows.
5. At `k = 0`, report equal-weight normalized fusion and the released native method separately
   where a valid native target-candidate scorer exists. Mark the native row `N/A` otherwise. Equal
   fusion has no neighbour component at `k=0` and therefore reduces to its declared semantic route.
6. In partial-enrollment and deployment-heterogeneity scenario tables, use equal-weight normalized
   fusion as the default external-baseline readout. Preserve truth-enrolled, truth-unenrolled,
   combined, and harmonic-mean diagnostics.
7. Report HALO's learned residual-classifier result separately from its matched `neighbors` and 1-NN
   controls.
8. Mark unsupported source/model combinations instead of inventing a replacement input path.

The classifier-attribution study is a project method, not a native baseline result. It freezes each
released encoder, attaches the identical HALO learnable classifier, trains only that classifier,
and evaluates it under both the aggregate and scenario manifests. Label these rows `HALO classifier
with frozen <encoder>` and disclose trainable parameters, steps, runtime, and seeds.

## Corpus-matched baseline arms (added 2026-09-22)

**Status (2026-09-24): built, not trained** (≈ 15.5 GPU-h, a separate go). They are rung 1's
tier 2 in the [roadmap](../overview/roadmap.md).

The headline tables above evaluate every encoder frozen, because the claim under test is enrollment
**without parameter updates**. That is the experimental condition, not an approximation of any
author's downstream recipe. Three of the four retained baselines are fine-tuned rather than frozen
by their own authors, so the frozen tables likely understate them; that is accepted, disclosed, and
does not change the deployment claim.

Two further arms answer the separate question of whether HALO's lead reflects its training corpus
sitting closer to the test distribution than the baselines' pretraining corpora do. They are
reported separately and never merged into a deployment row.

1. **Corpus-matched from scratch (M2).** The baseline's architecture, randomly initialised, trained
   on our training corpus under our sampler, episodes, objective and step budget. Precedented by
   three of the four baselines' own papers, which each run the same random-init control. It
   measures the architecture given our data, and it necessarily **discards the pretraining that is
   the released model's contribution** — so the frozen released row must be reported beside it.
2. **Frozen trunk with a corpus-fitted projection.** The released encoder is unchanged; its own
   `window_features` produces the representation in both fitting and evaluation, and a projection
   fitted with the differentiable-neighbour objective is applied before the readout. This is the
   only arm offered for an encoder whose authors never fine-tune it, or whose training cost is
   prohibitive; both reasons must be stated, protocol first.

Rules for both: no released preprocessing may be reimplemented where the adapter can be called, and
any reimplementation must be pinned against the released artifact by a test. Disclose trainable
versus total parameters, steps, runtime, seed, and the measured cost of any arm declined. Report
encoder-only readouts (1-NN) for the encoder comparison, so no classifier enters it. An arm that
scores below its own frozen released row is reported as such.

All model-selection decisions are fixed a priori or use the internal subject-held-out fold of the
supervised training sources; there is no separate development-source roster. Test results are per
dataset with subject-level uncertainty, checkpoint provenance, upstream data-overlap disclosure,
runtime, and memory. See the full [evaluation protocol](../contracts/evaluation_protocol.md).
