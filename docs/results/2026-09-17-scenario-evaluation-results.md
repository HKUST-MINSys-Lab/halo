# Deployment-scenario evaluation results - 2026-09-17

This is the durable result record for the completed representative deployment-scenario run. It is
not a checkpoint-selection result.

## Protocol

- Active protocol: `deployment-scenarios-v4-20260917`
- Source artifact: `deployment-scenarios-v3-20260916`; its cold-start rows are retired and excluded
- Window duration: 8 seconds
- Enrollment counts: `k = 0, 1, 4, 8, 32`
- Scenarios: partial enrollment, cross placement, cross dataset, missing modality, rate mismatch,
  new domain, and device-set mismatch
- HALO readout: learned support classifier
- External readout: fixed equal-weight normalized fusion
- Models: HALO, HARNet-5, HARNet-10, LiMU-BERT-X, UniMTS, and NormWear
- Promoted result rows: 4,341 (65 retired cold-start rows excluded)
- Source-artifact scored tasks: 651
- Task failures: 0
- Failed result rows: 0
- Runtime: 1,600.05 seconds

Every provider used the same model-independent query, candidate, and support manifests. Unsupported
model/input combinations remain explicit rather than being assigned a replacement input path. The
HALO checkpoint was
`training/support_classifier/outputs/halo_fixed_mr_residual_v3_curriculum1234_40k_20260916/best_internal.pt`
with SHA-256
`7c62680eb1f38305cd636e94086b1c232e2a332f4b3d0179e5bdba32dc542ce6`.

## Heterogeneous-condition result at k=8

Macro F1 is on a 0-100 scale. Each row compares HALO with the strongest eligible released baseline
on matched scenario cells. This is the concise scenario view; the tracked machine-readable
artifact retains every model, variant, coverage split, and enrollment count.

| scenario | HALO classifier | strongest released baseline | baseline | HALO minus baseline |
|---|---:|---:|---|---:|
| Partial enrollment | 46.25 | 36.63 | HARNet-10 | +9.62 |
| Cross placement | 44.90 | 35.21 | UniMTS | +9.69 |
| Cross dataset | 76.59 | 50.41 | HARNet-5 | +26.19 |
| Missing modality | 63.39 | 40.92 | UniMTS | +22.46 |
| Rate mismatch | 66.35 | 40.98 | UniMTS | +25.38 |
| New domain | 50.36 | 27.98 | UniMTS | +22.39 |
| Device-set mismatch | 71.60 | 46.90 | UniMTS | +24.69 |

HALO leads all seven retained scenarios. These are system-level comparisons: HALO received
task-specific episodic training, while the external encoders use released weights with the fixed
common adapter. They do not by themselves attribute the gain to HALO's encoder or classifier.

## Readout fairness control

A current-code paired diagnostic held every episode and feature cache fixed and changed only the
external readout between cosine 1-NN and equal-weight normalized fusion. It completed 364 rows
without failure.

| condition | fusion minus 1-NN at k=1 | fusion minus 1-NN at k=8 |
|---|---:|---:|
| Partial enrollment, all queries | +14.45 | +11.67 |
| Partial enrollment, truth enrolled | -42.41 | -50.91 |
| Partial enrollment, truth unenrolled | +18.42 | +17.70 |
| Cross placement | -11.55 | -12.49 |
| Cross dataset | -5.00 | -14.85 |
| Missing modality | -7.89 | -19.85 |
| Rate mismatch | -5.94 | -16.69 |
| Device-set mismatch | -10.60 | -21.00 |

Fusion is the primary fixed adaptation protocol because it can name candidates without enrollment.
The diagnostic shows why representation-only 1-NN must remain available as a companion control:
fusion helps when the truth is unenrolled but can dilute strong neighbor geometry when every truth
has support. No per-model or test-selected fusion weight is used.

## Artifacts

The source artifact's complete rows and immutable manifests are tracked in
`docs/results/artifacts/scenarios_halo_classifier_v3_20260917/`.

That immutable v3 artifact retains 65 cold-start rows for reproducibility. Readers and summary
generators must exclude `scenario == "s8_cold_start"`; new runs use v4 and cannot generate it.

| artifact | uncompressed SHA-256 |
|---|---|
| `results.json.gz` | `cbca4772343d2da5abdc2de1c11df7db2c29fa7d173471fb73633c42c0acfe46` |
| `manifests.jsonl.gz` | `cf5bca9d23f22d0a430f067b20b145aa7ad3cf8d2ad4c705017247fa6820c4d7` |
| `run_metadata.json` | `fab9df55609813caf5c844c1395181f4aa900492eca0d7ca8cb6ecedc22f2457` |
| `run_provenance.json` | `afeffddca2a849dfdc85fb599a3f8da9b98fefecd4ce9c4c9a26c4dfb6951116` |
| `model_artifacts.json` | `16b86185052581c29c3c9a6cd35c269e0297418d29d9e0076022bdc57928daf6` |
| `failures.json` | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |

The run provenance records Git revision `b528942a727360fd314d67585289378375600bd2` plus its
working-tree diff hash. The raw local evaluation directory remains
`training/support_classifier/evaluations/scenarios_halo_classifier_v3_20260917_profiled_v2_k0_1_4_8_32/`.

## Limitations and next attribution experiment

The evaluated HALO checkpoint predates the final sampler corrections. Its evaluation is valid, but
it must not be described as trained with the final corrected curriculum. The matching
differentiable-neighbours-trained HALO encoder arm also remains pending.

The next controlled attribution experiment freezes each released encoder and trains the identical
HALO classifier, input projection, curriculum, optimizer schedule, and checkpoint-selection rule.
Those arms must be named `HALO classifier with <encoder>`, because they test HALO's classifier on a
baseline representation rather than a baseline's native prediction mechanism.
