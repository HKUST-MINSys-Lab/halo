# Tier 1 (rung 1): unlabelled adaptation — EM-Dirichlet + affinity, calibrated temperatures

Macro-F1 on the fixed scored set (20 % of executions, execution-disjoint from the pool); k = 0 throughout. `anchor` = per-window zero-shot arg-max (equals each model's published sealed k = 0 row). The curve's null is N=0 (the same transductive method over the scored set alone).

### Dataset-balanced macro-F1 vs unlabelled pool size N

| encoder | anchor | N=0 | N=50 | N=100 | N=500 | N=2000 | N=all | Δ all−0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| halo | 47.4 | 50.6 | 51.3 | 52.1 | 52.6 | 51.8 | 51.9 | 1.3 |
| harnet10 | 32.0 | 31.6 | 31.6 | 31.5 | 30.8 | 31.6 | 31.4 | -0.3 |
| harnet5 | 31.2 | 30.5 | 30.5 | 30.6 | 31.0 | 30.2 | 31.1 | 0.6 |
| limubert_x | 21.9 | 23.0 | 22.4 | 22.5 | 22.7 | 22.3 | 22.6 | -0.4 |
| matched:harnet | 33.7 | 39.3 | 40.6 | 39.6 | 39.0 | 39.1 | 40.8 | 1.5 |
| normwear | 10.4 | 16.3 | 16.4 | 17.0 | 17.7 | 18.6 | 18.4 | 2.1 |
| unimts | 31.9 | 35.1 | 36.3 | 36.1 | 35.1 | 35.1 | 34.7 | -0.4 |

### Same, graph cluster-to-class assignment

| encoder | anchor | N=0 | N=50 | N=100 | N=500 | N=2000 | N=all | Δ all−0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| halo | 47.4 | 50.3 | 50.9 | 51.7 | 52.1 | 51.8 | 51.6 | 1.3 |
| harnet10 | 32.0 | 31.8 | 31.6 | 31.5 | 30.7 | 31.5 | 31.2 | -0.6 |
| harnet5 | 31.2 | 30.6 | 30.7 | 30.7 | 31.0 | 30.2 | 31.0 | 0.4 |
| limubert_x | 21.9 | 21.8 | 21.2 | 22.0 | 21.8 | 22.8 | 22.2 | 0.4 |
| matched:harnet | 33.7 | 40.5 | 41.8 | 39.4 | 40.7 | 40.8 | 41.2 | 0.6 |
| normwear | 10.4 | 16.8 | 16.7 | 16.9 | 18.7 | 17.9 | 18.6 | 1.8 |
| unimts | 31.9 | 36.5 | 36.9 | 35.0 | 36.6 | 36.6 | 36.9 | 0.5 |

### Cells helped / hurt by more than 5 points at N=all

| encoder | cells | vs anchor: helped | hurt | vs N=0: helped | hurt |
|---|---:|---:|---:|---:|---:|
| halo | 11 | 3 | 0 | 3 | 1 |
| harnet10 | 11 | 1 | 3 | 0 | 0 |
| harnet5 | 11 | 2 | 3 | 0 | 0 |
| limubert_x | 11 | 4 | 1 | 2 | 1 |
| matched:harnet | 11 | 7 | 0 | 2 | 0 |
| normwear | 11 | 8 | 0 | 2 | 0 |
| unimts | 11 | 5 | 2 | 2 | 3 |

### Per dataset (anchor → N=0 → N=all)

| encoder | inclusivehar | motionsense | realworld | shoaib | usc_had | ut_complex |
|---|---:|---:|---:|---:|---:|---:|
| halo | 38.1 → 43.6 → 46.6 | 67.5 → 71.8 → 71.3 | 45.1 → 43.6 → 47.6 | 63.2 → 63.9 → 65.5 | 31.2 → 37.4 → 38.0 | 39.4 → 43.5 → 42.8 |
| harnet10 | 31.2 → 27.9 → 26.1 | 41.0 → 39.5 → 39.5 | 24.3 → 29.6 → 28.7 | 46.7 → 40.6 → 41.1 | 22.3 → 24.4 → 22.8 | 26.6 → 27.6 → 29.9 |
| harnet5 | 28.0 → 24.6 → 25.9 | 46.1 → 39.2 → 39.2 | 25.2 → 29.0 → 31.1 | 42.8 → 40.3 → 38.8 | 21.9 → 23.7 → 24.7 | 23.0 → 26.2 → 26.7 |
| limubert_x | 27.0 → 24.0 → 22.4 | 36.8 → 38.4 → 38.1 | 18.0 → 25.4 → 28.1 | 23.9 → 22.4 → 23.2 | 16.4 → 18.0 → 13.9 | 9.4 → 9.8 → 9.6 |
| matched:harnet | 34.4 → 36.7 → 36.2 | 56.0 → 63.7 → 63.1 | 33.6 → 33.5 → 34.0 | 40.1 → 53.5 → 56.9 | 22.9 → 28.3 → 30.7 | 15.5 → 20.0 → 23.9 |
| normwear | 16.6 → 19.1 → 20.5 | 16.7 → 28.0 → 29.9 | 9.0 → 12.9 → 13.9 | 7.9 → 20.1 → 22.8 | 7.3 → 7.1 → 7.7 | 5.0 → 10.8 → 15.8 |
| unimts | 28.1 → 30.1 → 29.3 | 46.7 → 44.8 → 38.5 | 36.5 → 38.4 → 43.3 | 33.1 → 31.7 → 31.7 | 27.5 → 34.0 → 34.5 | 19.4 → 31.6 → 30.9 |

### Per cell: N=0 → N=all (identity assignment)

| dataset / stream | halo | harnet10 | harnet5 | limubert_x | matched:harnet | normwear | unimts |
|---|---:|---:|---:|---:|---:|---:|---:|
| inclusivehar / phone_waist | 43.6 → 46.6 | 27.9 → 26.1 | 24.6 → 25.9 | 24.0 → 22.4 | 36.7 → 36.2 | 19.1 → 20.5 | 30.1 → 29.3 |
| motionsense / phone_front_pocket | 71.8 → 71.3 | 39.5 → 39.5 | 39.2 → 39.2 | 38.4 → 38.1 | 63.7 → 63.1 | 28.0 → 29.9 | 44.8 → 38.5 |
| realworld / phone_forearm | 36.9 → 41.1 | 28.0 → 27.3 | 27.5 → 27.3 | 26.4 → 31.4 | 36.5 → 34.3 | 11.0 → 12.7 | 35.0 → 36.1 |
| realworld / phone_thigh | 45.3 → 46.7 | 31.2 → 29.8 | 28.2 → 30.9 | 23.0 → 22.0 | 34.3 → 37.1 | 17.0 → 17.9 | 35.0 → 45.3 |
| realworld / phone_waist | 48.6 → 54.9 | 29.6 → 28.9 | 31.3 → 35.1 | 26.7 → 30.9 | 29.5 → 30.4 | 10.7 → 11.1 | 45.2 → 48.5 |
| shoaib / phone_belt | 73.7 → 81.0 | 48.3 → 50.8 | 42.9 → 40.0 | 15.2 → 16.1 | 68.4 → 76.4 | 18.9 → 21.9 | 42.0 → 46.2 |
| shoaib / phone_left_pocket | 59.9 → 65.9 | 36.4 → 38.0 | 35.1 → 31.6 | 29.6 → 29.1 | 49.9 → 45.2 | 14.6 → 21.6 | 30.5 → 21.9 |
| shoaib / phone_right_pocket | 51.6 → 52.8 | 34.0 → 31.7 | 29.5 → 29.9 | 31.2 → 24.6 | 39.4 → 41.6 | 18.9 → 19.4 | 5.9 → 17.4 |
| shoaib / watch_wrist_proxy | 70.3 → 62.2 | 43.6 → 43.9 | 53.5 → 53.5 | 13.8 → 23.0 | 56.3 → 64.4 | 27.9 → 28.3 | 48.5 → 41.3 |
| usc_had / phone_hip | 37.4 → 38.0 | 24.4 → 22.8 | 23.7 → 24.7 | 18.0 → 13.9 | 28.3 → 30.7 | 7.1 → 7.7 | 34.0 → 34.5 |
| ut_complex / watch_wrist | 43.5 → 42.8 | 27.6 → 29.9 | 26.2 → 26.7 | 9.8 → 9.6 | 20.0 → 23.9 | 10.8 → 15.8 | 31.6 → 30.9 |

### Diagnostics at N=all (mean over cells)

| encoder | T | neighbour purity | same-execution share | purity, other executions | collapsed components |
|---|---:|---:|---:|---:|---:|
| halo | 13.91 | 0.88 | 0.54 | 0.78 | 0.00 |
| harnet10 | 4.53 | 0.71 | 0.24 | 0.63 | 0.00 |
| harnet5 | 4.27 | 0.68 | 0.24 | 0.59 | 0.00 |
| limubert_x | 4.23 | 0.88 | 0.66 | 0.68 | 0.00 |
| matched:harnet | 7.43 | 0.79 | 0.41 | 0.66 | 0.00 |
| normwear | 0.33 | 0.86 | 0.54 | 0.72 | 0.00 |
| unimts | 372.42 | 0.84 | 0.45 | 0.73 | 0.00 |

