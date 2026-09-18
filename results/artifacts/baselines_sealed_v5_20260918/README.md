# Fresh baseline sealed evaluation

Run date: 2026-09-18

This is the completed baseline-only aggregate evaluation under
`sealed-manifest-v2-20260916` and feature-cache schema
`sealed-feature-v6-20260918`. It covers the six sealed datasets, 4/8/16-second
evidence windows, and `k = 0, 1, 2, 4, 8, 16, 32, 64, 128` for HARNet-5,
HARNet-10, LiMU-BERT-X, UniMTS, and NormWear.

The run completed all 39 protocol cells in 2,521.9 seconds. It emitted 3,354
successful rows and 213 explicit `n/a` rows for unsupported model/input or
insufficient execution-disjoint support conditions; there are no failed rows.
At `k=0`, the artifact retains each available native zero-support path and the
declared unifying fusion path. At `k>=1`, it retains both cosine 1-NN and
equal-weight normalized fusion.

`results.json.gz` is the exhaustive result payload, `episode_manifests.json.gz`
records the deterministic episodes, and `RESULTS.md` is generated directly from
those rows. This artifact is baseline-only: it must be compared with HALO only
after a current HALO checkpoint has been scored under the same protocol.

The evaluator ran from commit `1f5f1124527589e96bdc17530b666f5286aa7f95`.
The uncompressed source hashes are recorded in `source_hashes.txt`.
