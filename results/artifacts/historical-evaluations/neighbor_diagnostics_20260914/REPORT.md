# Neighbor-control diagnostics

Exploratory sealed-test analysis using cached embeddings and immutable episodes. No encoder or classifier was trained.

## Decision Rules

| k | rule | macro-F1, mean across cells | accuracy, mean across cells |
|---:|---|---:|---:|
| 1 | 1nn | 65.2993 | 66.3354 |
| 1 | prototype | 65.2993 | 66.3354 |
| 1 | ridge | 65.2045 | 66.4863 |
| 1 | soft-neighbor | 65.2993 | 66.3354 |
| 8 | 1nn | 75.7002 | 76.0640 |
| 8 | prototype | 76.4477 | 76.8259 |
| 8 | ridge | 78.0338 | 78.4807 |
| 8 | soft-neighbor | 77.3463 | 77.5340 |

## Support Draw Variance

| k | rule | mean macro-F1 | seed standard deviation |
|---:|---|---:|---:|
| 1 | 1nn | 65.3050 | 0.5416 |
| 1 | prototype | 65.3050 | 0.5416 |
| 1 | soft-neighbor | 65.3050 | 0.5416 |
| 8 | 1nn | 75.6787 | 0.3849 |
| 8 | prototype | 76.7214 | 0.3720 |
| 8 | soft-neighbor | 77.4583 | 0.3561 |

## Interpretation

A low correct-label support rank on incorrect queries points to retrieval geometry. A high rank but an incorrect soft vote points to aggregation/calibration, which is the narrow case where a learned support-conditioned re-ranker is justified. Support-removal flip rates measure dependence on individual enrolled examples; they are not a robustness claim.
