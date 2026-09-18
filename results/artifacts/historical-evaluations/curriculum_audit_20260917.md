# Support Curriculum Audit

Protocol: `support-curriculum-audit-v1-20260917`. Seed: `20260917`. Window: `8.0` s.

Sampled **2,000 support sets** and **7,457 queries** from the active training roster.

| condition | realised share | datasets | labels | acquisition keys |
|---|---:|---:|---:|---:|
| acquisition/compatible (enrolled only) | 0.748 | 8 | 144 | 26 |
| acquisition/cross_placement (enrolled only) | 0.151 | 4 | 78 | 19 |
| acquisition/cross_dataset (enrolled only) | 0.101 | 3 | 5 | 3 |
| enrollment/complete | 0.427 | - | - | - |
| enrollment/partial | 0.230 | - | - | - |
| enrollment/zero | 0.343 | - | - | - |

- Curriculum fallback share: **0.225**
- Truth-enrolled query share: **0.521**
- Cross-placement sets containing another dataset: **0.000**

## Dataset Coverage

- `compatible`: dsads, forth_trace, harmes, hhar, kuhar, realdisp, wisdm, xrf_v2
- `cross_placement`: dsads, forth_trace, realdisp, xrf_v2
- `cross_dataset`: dsads, forth_trace, xrf_v2

## Candidate And Support Distributions

- Candidate count: C=2: 0.025, C=3: 0.021, C=4: 0.018, C=5: 0.009, C=6: 0.128, C=7: 0.001, C=8: 0.007, C=9: 0.007, C=10: 0.001, C=11: 0.002, C=12: 0.016, C=13: 0.002, C=14: 0.001, C=15: 0.005, C=16: 0.151, C=17: 0.137, C=18: 0.135, C=19: 0.050, C=20: 0.027, C=21: 0.021, C=22: 0.025, C=23: 0.022, C=24: 0.026, C=25: 0.025, C=26: 0.021, C=27: 0.024, C=28: 0.021, C=29: 0.022, C=30: 0.020, C=31: 0.014, C=32: 0.015
- Enrolled K: K=1: 0.237, K=2: 0.225, K=4: 0.217, K=8: 0.145, K=16: 0.107, K=32: 0.069
