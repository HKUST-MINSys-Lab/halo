# 2026-09-13 - Retired future-JEPA promoted-result snapshot

**Status:** archival snapshot. Future-JEPA is no longer an active HALO training route because its
measured benefit disappears after the supervised encoder adaptation used by the current system.
This preserves the JEPA rows formerly promoted in `docs/results/RESULTS.md` before their removal
from the live comparison table. The underlying artifacts remain untouched.

The protocol used six sealed datasets and execution-disjoint supports. `k` is enrolled executions
per candidate label. Values are equal-dataset means on a 0-100 scale.

## Macro F1

| model | parameters | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HALO fixed multiresolution, JEPA + 5k tune | 3.22M | 24.23 | 55.67 | 60.51 | 64.42 | 68.18 | 70.65 | 72.80 | 74.55 | 75.74 |
| HALO multispan kernel, JEPA + 5k tune | 2.71M | 26.21 | 57.21 | 62.72 | 66.24 | 69.35 | 71.59 | 73.79 | 75.31 | 76.11 |

## Accuracy

| model | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HALO fixed multiresolution, JEPA + 5k tune | 30.94 | 55.77 | 60.83 | 64.74 | 68.47 | 70.85 | 73.10 | 74.83 | 76.00 |
| HALO multispan kernel, JEPA + 5k tune | 31.79 | 57.09 | 62.67 | 66.22 | 69.35 | 71.70 | 73.93 | 75.48 | 76.22 |

## Per-dataset k=0 Macro F1

| model | MotionSense | RealWorld | Shoaib | InclusiveHAR | USC-HAD | UT-Complex | equal-dataset mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| HALO fixed multiresolution, JEPA + 5k tune | 31.16 | 26.90 | 34.31 | 20.75 | 15.11 | 17.15 | 24.23 |
| HALO multispan kernel, JEPA + 5k tune | 35.46 | 32.63 | 24.89 | 24.13 | 19.35 | 20.82 | 26.21 |

The causal interpretation and the full frozen-versus-adapted ladder are in
[2026-09-13-jepa-value-measured.md](2026-09-13-jepa-value-measured.md).
