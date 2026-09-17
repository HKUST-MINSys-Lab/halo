# Baseline comparison policy

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
7. Report HALO's learned retrieve-mix-vote result separately from its matched `neighbors` and 1-NN
   controls.
8. Mark unsupported source/model combinations instead of inventing a replacement input path.

The classifier-attribution study is a project method, not a native baseline result. It freezes each
released encoder, attaches the identical HALO learnable classifier, trains only that classifier,
and evaluates it under both the aggregate and scenario manifests. Label these rows `HALO classifier
with frozen <encoder>` and disclose trainable parameters, steps, runtime, and seeds.

All model-selection decisions are fixed a priori or use the internal subject-held-out fold of the
supervised training sources; there is no separate development-source roster. Test results are per
dataset with subject-level uncertainty, checkpoint provenance, upstream data-overlap disclosure,
runtime, and memory. See the full [evaluation protocol](../design/EVALUATION_PROTOCOL.md).
