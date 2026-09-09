# Temporal-Resolution Comparison (2026-09-09)

## Purpose

This controlled comparison holds the corpus, end-to-end neighbor objective, episode sampler, seed, optimizer schedule, and 35,000-step budget fixed while changing the front end's temporal resolution. It is not a comparison with external baselines.

All runs used corpus fingerprint `47ed542baddb3f0f`, 56 streams, approximately 1.75 M training windows, 209 K development windows, and seed `20260901`. Runtime was PyTorch 2.9.0 with bf16 autocast on an RTX 4090.

## Arms And Selected Checkpoints

| ID | Front end | Physical scales | Selected step | Development selection F1 |
|---|---|---|---:|---:|
| `fixed_1s` | fixed physical filterbank | 1.0 s | 30,000 | 0.701 |
| `fixed_mr_050_100_150` | fixed filterbank, duration embedding, joint attention | 0.5, 1.0, 1.5 s | 35,000 | 0.736 |
| `multispan_025_050_100_200` | continuous-kernel multi-span front end | 0.25, 0.5, 1.0, 2.0 s | 25,000 | 0.747 |
| `fixed_mr_025_050_100_200` | fixed filterbank, duration embedding, joint attention | 0.25, 0.5, 1.0, 2.0 s | 30,000 | 0.742 |

The final arm completed in 43 minutes. Its development selection F1 was 0.399 at initialization, 0.619 at 10k, 0.717 at 20k, 0.742 at 30k, and 0.738 at 35k. `best_internal.pt` is therefore the 30k checkpoint rather than the final checkpoint.

## Primary External Protocol

The primary sealed manifest is `eval/manifests/adaptation_v2_20260907.json.gz`. Each `k` is the number of independent enrolled executions per candidate. Results are coherent-label, cross-subject, same-configuration adaptation macro F1, aggregated within each dataset and then averaged across datasets. The ordinary cohort contains six datasets and the specialized cohort two, so they remain separate.

`support comparator` is HALO's native neighbor readout. The native neighbor-only arm has no zero-shot head, so native `k=0` is correctly unavailable. It is evaluated beside the same-vector 1-NN, prototype, and ridge controls in the raw artifacts.

### Ordinary Activities: Cross-Subject, Same Configuration (6 datasets)

| arm / support comparator | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| fixed_1s | 65.08 | 69.34 | 73.00 | 75.44 |
| fixed_mr_050_100_150 | **66.56** | **71.05** | **74.52** | 77.04 |
| multispan_025_050_100_200 | 65.24 | 69.98 | 73.94 | **77.14** |
| fixed_mr_025_050_100_200 | 65.17 | 70.05 | 73.74 | 76.31 |

### Specialized Activities: Cross-Subject, Same Configuration (2 datasets)

| arm / support comparator | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| fixed_1s | 53.15 | 58.24 | 62.88 | 66.33 |
| fixed_mr_050_100_150 | **53.97** | **59.44** | **64.20** | **67.47** |
| multispan_025_050_100_200 | 53.40 | 58.57 | 63.23 | 66.16 |
| fixed_mr_025_050_100_200 | 52.39 | 57.94 | 63.54 | 66.63 |

The exact-scale fixed multi-resolution control does not explain the 0.5/1.0/1.5 arm simply by adding more scales. The 0.5/1.0/1.5 fixed-filterbank configuration is strongest across this primary comparison. Continuous multi-span is competitive at ordinary `k=8` but trails that fixed multi-resolution arm elsewhere.

## High-Enrollment Supplement

The separate `eval/manifests/adaptation_v2_high_support_20260909.json.gz` manifest tests `k=16,32,64`. It must not be pooled with the primary curve: only five relations reach `k=16`, two reach `k=64`, and only one specialized relation reaches `k=32`.

| arm / support comparator | ordinary k=16 (3 ds) | ordinary k=32 (2 ds) | ordinary k=64 (1 ds) |
|---|---:|---:|---:|
| fixed_1s | 67.54 | 77.35 | 68.00 |
| fixed_mr_050_100_150 | 68.74 | 78.82 | 67.42 |
| multispan_025_050_100_200 | **70.51** | **80.73** | **69.19** |
| fixed_mr_025_050_100_200 | 67.33 | 77.85 | 66.37 |

## Artifacts

Raw per-cell results are in `eval/adaptation_results/temporal_resolution_20260909/`. Checkpoint-specific feature caches are under `eval/adaptation_feature_cache/temporal_resolution_*`. Regenerate aggregates and per-dataset tables with `python -m eval.assemble_adaptation` and the corresponding manifest. Do not mix primary and high-support outputs because their available relations differ.
