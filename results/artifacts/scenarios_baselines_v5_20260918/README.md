# Fresh baseline deployment-scenario evaluation

Run date: 2026-09-18

This artifact is the completed `deployment-scenarios-v5-20260918` baseline-only evaluation.
It uses the five retained released providers (`harnet5`, `harnet10`, `limubert_x`, `unimts`, and
`normwear`), seven active scenarios, 8-second evidence, and `k = 0, 1, 4, 8, 32`.

`run_metadata.json` records 1,132 scored task units, 11,211 result rows, zero task failures, and
zero failed rows. `results.json.gz` is the exhaustive metric payload; `manifests.jsonl.gz` records
the deterministic episodes; `paired_deltas.json` contains matched-control comparisons. `RESULTS.md`
is generated directly from the result rows.

The source invocation, commit, model artifact fingerprints, cache provenance, and final progress
record are retained beside the data. This baseline-only artifact must not be numerically combined
with historical HALO scenario rows; a matching current HALO evaluation is required.
