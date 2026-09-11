# Baseline comparison policy

The goal is a useful deployment comparison, not an attribution claim across unmatched upstream
corpora. Each retained encoder keeps its author-released weights and published input preprocessing.

For an eligible query/support episode:

1. Build the recording split once, before embedding any method.
2. Feed the same recordings to every adapter, preserving each adapter's required channel order,
   units, rate conversion, crop, padding, and normalization.
3. Apply the same support/candidate episode manifest to every exported representation.
4. Report 1-NN, prototype, and ridge as separate readouts where `k >= 1`; label fitted methods as
   adaptation rather than zero shot.
5. Report HALO's learned support-vote result separately from its 1-NN control.
6. Mark unsupported source/model combinations instead of inventing a replacement input path.

All model-selection decisions use development data. Test results are per dataset with subject-level
uncertainty, checkpoint provenance, upstream data-overlap disclosure, runtime, and memory. See the
full [evaluation protocol](../design/EVALUATION_PROTOCOL.md).
