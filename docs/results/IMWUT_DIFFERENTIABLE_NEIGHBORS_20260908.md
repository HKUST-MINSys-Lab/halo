# Differentiable-neighbor encoder experiment - 2026-09-08

## Conclusion

Training HALO end to end through a parameter-free differentiable-neighbor objective produced a
large, consistent improvement in enrollment classification. At the predeclared 35,000-step endpoint,
HALO with hard 1-NN reaches **62.73 / 66.73 / 70.22 / 72.62 macro F1** at
k=`1 / 2 / 4 / 8`. This is `+13.50 / +12.03 / +11.41 / +10.26` points over the same run's
initialized encoder and `+6.92 / +6.16 / +5.96 / +5.44` over released UniMTS on the matched primary
cohort.

This supports the narrow hypothesis: the encoder can be trained directly for enrollment-style
nearest-neighbor adaptation, and the previous objective was leaving substantial representation
quality unused. It does not test semantic zero-shot recognition, and it does not establish that a
learned attention head adds value beyond the trained encoder.

## Experiment

- Source commit: `4b460fd` (clean worktree, empty persisted source patch).
- Run: `training/compare/outputs/neighbor_encoder_20260908_seed20260901/`.
- Checkpoints: `initial.pt`, `best_internal.pt` at step 19,500, and `last.pt` at step 35,000.
- Frontend: fixed physical filterbank; acquisition text neutralized.
- Objective: normalized recording cosine similarities, aggregated into per-class log-sum-exp
  scores, followed by cross-entropy. There is no trainable comparator, projection, or classifier.
- Both query and support recordings pass through the same trainable encoder and receive gradients.
- Four independently sampled support sets per step, four query executions per set, k sampled from
  `{1,2,4,8}`, 6-14 candidates, and up to two windows averaged per execution.
- Training corpus: 56 streams, 1,753,141 train windows, 209,339 validation windows, 166 labels.
- Runtime: 1,167.8 seconds (**19.46 minutes**) on one RTX 4090 using bfloat16.
- Before launch, 1,224 tests passed, including CPU/GPU training smokes and train/evaluation parity.

## Training health

The encoder continued improving after the early part of training. These are means over logged
training steps in each interval.

| steps | loss | hard 1-NN accuracy | correct-label mass | cosine margin | effective rank |
|---|---:|---:|---:|---:|---:|
| 1-2k | 1.305 | 60.51% | 0.456 | 0.327 | 44.8 |
| 2-10k | 0.927 | 66.45% | 0.569 | 0.389 | 50.3 |
| 10-20k | 0.742 | 71.98% | 0.649 | 0.458 | 57.3 |
| 20-35k | 0.673 | 74.52% | 0.693 | 0.503 | 62.9 |

Query and support embedding gradients remained nonzero throughout. In the final interval their mean
norms were 0.0383 and 0.0033 respectively; the support value is per support row and there are many
more support than query rows. Mean pre-clip encoder gradient norm was 5.26 and the mean clipping
coefficient was 0.255. No non-finite loss, gradient, or representation collapse was observed.

The fixed internal validation set selected step 19,500 by minimum loss (`0.8047`). Its external
transfer is slightly weaker than the predeclared 35,000-step endpoint at every k. This means the
small internal validation suite is useful for failure detection but is not yet a reliable proxy for
cross-dataset transfer. The final checkpoint is reported as the primary endpoint because its step was
fixed before external evaluation, not because it was selected on the test results.

## Matched primary curve

Scores are dataset-macro F1 on a 0-100 scale. The primary cohort is the same eight datasets with
cross-subject, same-configuration support and k=1,2,4,8 coverage used by the September 8 matched
report. Each support item is an independent enrolled execution. Released baselines use their
unchanged frozen checkpoints and the exact same serialized support/query plans.

| encoder / readout | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| **HALO neighbor-trained, 35k / 1-NN** | **62.73** | **66.73** | **70.22** | **72.62** |
| HALO neighbor-trained, 19.5k / 1-NN | 61.64 | 65.77 | 69.31 | 71.83 |
| HALO initialized / 1-NN | 49.23 | 54.70 | 58.81 | 62.37 |
| previous HALO sensor-only run / 1-NN | 50.68 | 56.84 | 61.42 | 64.81 |
| UniMTS / 1-NN | 55.81 | 60.58 | 64.27 | 67.18 |
| HARNet / 1-NN | 46.13 | 50.03 | 53.66 | 56.67 |
| ImageBind / 1-NN | 40.54 | 46.06 | 50.67 | 55.13 |
| NormWear / 1-NN | 23.71 | 26.42 | 28.95 | 31.77 |

The parameter-free soft-neighbor readout scores `62.75 / 67.00 / 70.69 / 73.31` at the final
checkpoint, a small gain over hard 1-NN. Prototype scores `62.73 / 66.81 / 70.31 / 72.81`; ridge
scores `61.89 / 65.78 / 69.93 / 73.06`. Thus the central result is robust to the simple readout,
rather than depending on a fitted classifier.

## Per-dataset result at k=8

| dataset | HALO 35k / 1-NN | UniMTS / 1-NN | delta |
|---|---:|---:|---:|
| InclusiveHAR | 44.68 | 44.24 | +0.44 |
| MoniPar | 50.71 | 42.65 | +8.06 |
| MotionSense | 82.18 | 87.48 | -5.30 |
| RealWorld | 79.53 | 75.18 | +4.35 |
| Shoaib | 95.20 | 95.14 | +0.06 |
| SPAR | 78.23 | 64.98 | +13.25 |
| USC-HAD | 64.74 | 56.99 | +7.75 |
| UT-Complex | 85.73 | 70.82 | +14.90 |

HALO leads UniMTS on seven of eight datasets, with MotionSense the exception. The largest gains are
on SPAR, USC-HAD, and UT-Complex; this pattern should be investigated rather than generalized as a
universal advantage.

## Artifacts and limits

- Raw evaluation: `eval/adaptation_results/neighbor_encoder_20260908/`.
- Standalone tables: `eval/adaptation_tables/neighbor_encoder_20260908/`.
- Tables assembled with released baselines:
  `eval/adaptation_tables/neighbor_encoder_matched_20260908/`.
- Manifest: `eval/manifests/adaptation_v2_20260907.json.gz`, fingerprint
  `c88099170efe04f8842533bb15df8c0c5121e75e004784e49639f5857fe928cd`.
- External evaluations took 89.8 seconds for 19.5k and about 60.2 seconds each for initialization
  and 35k. Feature extraction used the adapter's checkpoint-keyed cache.

There is one trained seed. The test datasets have influenced prior development, so this is not a
fresh sealed-test claim. Released encoders differ in size, modality, and upstream data; this is a
matched deployed-system comparison, not an architecture-only ablation. The neighbor arm has no
semantic k=0 mechanism, and unsupported k=0 cells are recorded as N/A. A second training seed and a
development protocol that predicts cross-dataset transfer are required before treating the numerical
margin as publication-grade evidence.
