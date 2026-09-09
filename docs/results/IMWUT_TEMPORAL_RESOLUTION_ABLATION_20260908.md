# HALO temporal-resolution ablation — 2026-09-08

## Result in one sentence

Short fixed-filterbank patches hurt transfer, while a 1.5 s fixed-filterbank patch and the
continuous frontend are only small, statistically unresolved improvements over the 1.0 s control;
the tested 0.5 + 1.5 s dual-resolution construction was not explicitly scale-aware and therefore
does not settle the value of properly tagged multi-resolution fusion.

**Post-run implementation audit.** The comparison encoder did receive physical centre times and
filterbank resolution flags, but its sensor-level tokens had no duration embedding and the compact
comparison path failed to forward `resolution_ids`. The reported dual-resolution score remains an
accurate result for that untagged construction; it is not evidence against the corrected design.
Commit `90d44f3` contains the original experiment implementation. The subsequent scale-aware path
uses 0.5, 1.0, and 1.5 s grids, adds a bounded learned embedding of log physical duration before
self-attention, forwards resolution identities, and balances all active scales during pooling.

**RoPE reconstruction warning (2026-09-09).** The historical fixed dual-resolution checkpoint did
not record its actual fastest RoPE period (0.5 s). The shared loader inferred 0.4 s from an older
Phase-A convention, so the table below evaluated a slightly altered temporal encoder. Current
comparison checkpoints persist the exact period and round-trip it. The historical dual-resolution
number remains exploratory and must be rerun before it is used as evidence for the corrected arm.

**Continuous frontend version warning (2026-09-09).** All continuous scores below describe the
2026-09-08 implementation from commit `90d44f3` and the corresponding run's saved source patch.
Later corrections changed the initial kernel bank and replaced the harmonic-slot observability
fraction with retained coefficient energy. Loading an old checkpoint under that new math changes
its low-rate representations even though its saved kernel parameters are unchanged. Current code
therefore rejects checkpoints without the current frontend revision marker. These tables remain
historical results; the corrected single-span arm and new multi-span arm need separate training
and evaluation before any performance comparison. No corrected-arm score is available here.

These are exploratory model-selection results, not confirmatory paper results. The external
evaluation datasets have now informed design decisions and must not be described as untouched test
data in a final paper.

## Questions

1. Does a broader, dataset-balanced development panel select checkpoints that transfer better?
2. Does preserving within-patch temporal order with the continuous frontend improve transfer?
3. What single fixed-filterbank patch duration works best?
4. Does encoding both 0.5 s and 1.5 s resolutions improve over one resolution?

## Controlled protocol

All arms used the same corpus fingerprint (`47ed542baddb3f0f`), subject split, seed (`20260901`),
35,000 optimizer steps, four episodes per step, 32 support rows per episode, four queries per
support set, two sampled windows per execution, candidate range 6–14, and enrollment counts
`k={1,2,4,8}`. The encoder was trained end to end through the differentiable-neighbor objective.
Acquisition text was neutralized. Only frontend or patch construction changed.

External evaluation used the frozen encoder, 1-NN, coherent labels, same-configuration
cross-subject enrollment, and the immutable manifest
`eval/manifests/adaptation_v2_20260907.json.gz`. The headline is the dataset-macro F1 over the eight
datasets that provide every `k={1,2,4,8}` cell. This is the same aggregation used for the preceding
1.0 s control. Raw result artifacts are under
`eval/adaptation_results/resolution_experiments_20260908/`.

The five new arms were trained concurrently on one RTX 4090. They completed in 30.8–35.3 minutes
of per-process wall time, or about 35 minutes total elapsed for all five. Those timings measure the
concurrent experiment campaign and are not clean single-model throughput comparisons. The older
1.0 s control took 19.5 minutes when run by itself.

## External results

Macro F1 (%), dataset-macro over the common eight-dataset cohort:

| Encoder arm, final checkpoint | k=1 | k=2 | k=4 | k=8 | Mean |
|---|---:|---:|---:|---:|---:|
| Fixed filterbank, 0.5 s | 60.02 | 64.52 | 68.22 | 70.66 | 65.85 |
| Fixed filterbank, 0.75 s | 60.95 | 65.01 | 68.50 | 71.27 | 66.43 |
| **Fixed filterbank, 1.0 s control** | **62.73** | **66.73** | **70.22** | **72.62** | **68.08** |
| Fixed filterbank, 1.5 s | 62.58 | 67.16 | 70.87 | **73.49** | **68.53** |
| Fixed filterbank, 0.5 + 1.5 s | 62.44 | 66.90 | 70.32 | 73.09 | 68.19 |
| Continuous frontend, 1.0 s | 62.40 | 66.86 | **71.04** | 73.17 | 68.37 |

Relative to the 1.0 s control, averaging each dataset over k before bootstrapping datasets:

| Arm | Mean delta | 95% dataset-bootstrap interval | Datasets improved |
|---|---:|---:|---:|
| Fixed 0.5 s | -2.22 | [-3.76, -0.80] | 1 / 8 |
| Fixed 0.75 s | -1.65 | [-3.05, -0.38] | 2 / 8 |
| Fixed 1.5 s | +0.45 | [-1.05, +2.29] | 3 / 8 |
| Fixed 0.5 + 1.5 s | +0.11 | [-0.98, +1.30] | 4 / 8 |
| Continuous 1.0 s | +0.29 | [-1.53, +2.35] | 3 / 8 |

The intervals quantify variation across the eight datasets, not uncertainty across training seeds.
There is only one training seed per arm.

### Per-dataset k=4

| Arm | Inclusive-HAR | MoniPar | MotionSense | RealWorld | Shoaib | SPAR | USC-HAD | UT Complex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 0.5 s | 42.45 | 47.08 | 74.50 | 73.31 | 92.89 | **76.59** | 55.85 | **83.06** |
| Fixed 0.75 s | **43.62** | 47.66 | 75.30 | 77.54 | 92.30 | 73.86 | 55.77 | 81.98 |
| Fixed 1.0 s | 43.23 | 47.58 | 79.42 | 77.70 | 93.28 | 75.77 | **61.20** | **83.61** |
| Fixed 1.5 s | 41.93 | 47.67 | **85.33** | 77.60 | 96.30 | 75.77 | 61.03 | 81.31 |
| Fixed 0.5 + 1.5 s | 43.03 | **48.45** | 83.39 | 76.57 | 94.67 | 74.88 | 59.93 | 81.68 |
| Continuous 1.0 s | 41.97 | 46.66 | 85.01 | **80.61** | **97.69** | 75.16 | 59.52 | 81.71 |

The continuous frontend's aggregate gain is concentrated in MotionSense, RealWorld, and Shoaib.
It loses on five of eight datasets. This is evidence that within-patch temporal order helps some
motion families, not evidence of a general replacement for the fixed filterbank.

## Checkpoint-selection experiment

The development panel now draws eight deterministic support sets from every eligible held-out
dataset (136 support sets and 425 query windows in these runs). Selection uses the dataset-macro
hard-1NN F1 rather than the learned training loss. Across the five final arm checkpoints, the new
development score tracks external mean F1 reasonably well (Spearman rho 0.90, p=0.037), so it is
useful for ranking substantially different encoder designs.

| Arm | Selected step | Development F1 at selected step | Development F1 at 35k | External mean: selected | External mean: 35k |
|---|---:|---:|---:|---:|---:|
| Fixed 0.5 s | 25,000 | 71.85 | 70.06 | 65.91 | 65.85 |
| Fixed 0.75 s | 30,000 | 70.92 | 70.62 | 66.48 | 66.43 |
| Fixed 1.5 s | 35,000 | 72.93 | 72.93 | 68.53 | 68.53 |
| Fixed 0.5 + 1.5 s | 35,000 | 72.76 | 72.76 | 68.19 | 68.19 |
| Continuous 1.0 s | 25,000 | 75.82 | 75.51 | 68.16 | 68.37 |

Within a run, the selected and final external scores differ by at most 0.21 points, and selection is
not consistently better. The panel is therefore suitable as a training-health and coarse-ranking
measure, but this experiment does not establish that it can reliably resolve small checkpoint
differences. Final checkpoint choice should not be based on sub-point movements from one seed.

## Interpretation

1. **Do not use 0.5 s or 0.75 s as the sole fixed-filterbank resolution.** They discard too much
   low-frequency context and lose reliably, especially on MotionSense, RealWorld, and USC-HAD.
2. **Retain 1.0 s as the conservative default.** The 1.5 s arm has the highest mean, but its +0.45
   point gain is inconsistent across datasets and unresolved with one seed.
3. **Do not enable the tested untagged dual-resolution path by default.** Its +0.11 point mean
   change is a tie in practical terms. A corrected scale-aware three-resolution arm is required
   before drawing a conclusion about multi-resolution fusion itself.
4. **The temporal-order hypothesis remains plausible but unproven.** The fixed filterbank is
   invariant to time reversal within a patch up to floating-point error, whereas the continuous
   frontend responds to reversal. The continuous arm's gains on three locomotion-oriented datasets
   show that the information can matter, but its five regressions show that the current learned
   frontend does not exploit it robustly.
5. **Do not combine continuous and multi-resolution yet.** The continuous frontend currently emits
   a fixed physical frame layout designed for one-second tokens. The training CLI now rejects other
   patch durations rather than silently producing invalid masks.

## Next decision

The next experiment is the corrected fixed-filterbank 0.5 + 1.0 + 1.5 s arm against the fixed 1.0 s
control. If it shows a material gain, repeat both over two or three seeds before adopting the added
complexity. The continuous frontend remains a separate temporal-order experiment.

## Reproducibility and verification

- Experiment-support source commit: `90d44f3`.
- Focused tests: 158 passed.
- Smoke runs passed for fixed 0.5, 0.75, 1.5 s; fixed 0.5 + 1.5 s; continuous 1.0 s; and checkpoint
  reconstruction.
- The deterministic development panel produced the same fingerprint
  (`ffe71ba0199c35a2`) under three independent `PYTHONHASHSEED` values.
- External result JSON files include manifest, checkpoint, source, and runtime provenance.
