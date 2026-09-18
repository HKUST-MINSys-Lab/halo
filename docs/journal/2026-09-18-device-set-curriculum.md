# Joint Device-Set Curriculum And Evaluation

## Purpose

Multi-device support is not a claim that every sensor placement is interchangeable. It tests a
more precise deployment condition: an enrolled example and a query may expose the same activity
through different subsets of simultaneously worn devices. The encoder receives every selected
device; the classifier must decide how much to trust the resulting support evidence.

## Training

The standard eight-source classification corpus is retained. Only the aligned multi-device sources
(`realdisp`, `xrf_v2`, `dsads`, and `forth_trace`) can form a joint device-set episode.

For each eligible support set, before any signal is loaded:

1. The sampler has already selected the query recordings, support executions, candidate labels,
   subject relation, and acquisition regime. Device planning does not inspect labels or signals.
2. The planner finds device stream IDs available for every query and every support recording in
   that support set. A relation is then selected uniformly from feasible relations:
   `matched_single`, `matched_composite`, `query_superset`, `support_superset`,
   `partial_overlap`, or `disjoint`.
3. All queries sharing the support set receive the same query subset; every support execution
   receives the same support subset. Exact-time alignment and existing subject/execution exclusion
   rules remain mandatory.
4. If the set spans datasets, has no common aligned devices, or cannot realize a relation, it takes
   the legacy independent-composition path and is recorded as `not_applicable`. No compatibility
   rule is relaxed as a fallback.

`--device-set-challenge-probability` controls the chance of attempting this planner (default
`0.5`). `--multi-device-probability` continues to govern ordinary independent composition for
rows outside a planned relation.

The task loss is unchanged cross-entropy. The first experiment should not add a consistency loss:
matched controls and relation-stratified telemetry establish whether the existing encoder and
classifier already learn useful invariance.

## Telemetry

Every train step records the per-relation fraction, loss, and accuracy, plus mean planned query
and support device counts, Jaccard overlap, and feasibility fallback fraction. Existing overall
device count metrics remain a separate statement about encoded rows, not a substitute for the
query/support relationship.

## Scenario 7

At `k > 0`, each sealed multi-device cell now includes all feasible directional pairs:

- every single device supporting the full device set and the inverse direction;
- every leave-one-device-out support subset against the full query set;
- rotation-balanced two-device partial-overlap pairs when at least three devices exist;
- rotation-balanced disjoint single-device pairs.

Every mismatch has an exact matched control using the same query stream, candidates, support
counts, query rows, and subject partition. Scenario rows store the explicit query/support device
IDs, Jaccard overlap, and mismatch severity. `k=0` remains a separate label-bridge condition: it
has no enrolled support set and therefore cannot test device-set mismatch.

The standard sealed aggregate remains unchanged. Scenario reporting uses 8-second windows and
`k={1,4,8,32}`, reports per-dataset rows and paired subject-bootstrap confidence intervals, and
does not imply a new aggregate headline score.
