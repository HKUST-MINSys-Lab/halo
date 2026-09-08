# Matched HALO and released-baseline results - 2026-09-08

Current matched result for the sensor-only comparator on branch `imwut/compare`.
This supersedes the September 4 baseline-only snapshot for comparisons with HALO.

## Conclusion

Training improves HALO's enrollment representation modestly, but the learned native engine does
not improve on the same trained encoder with 1-NN in the aggregate at k=2,4,8. UniMTS remains the
strongest released baseline on the primary curve. Zero-shot remains weak. These results do not
establish overall superiority of HALO, nor a benefit from the learned engine over 1-NN.

## Provenance and protocol

- Evaluation source commit: `83058a3`; all six runs recorded clean source.
- Manifest: `eval/manifests/adaptation_v2_20260907.json.gz`.
- Manifest SHA256 identity: `c88099170efe04f8842533bb15df8c0c5121e75e004784e49639f5857fe928cd`.
- HALO checkpoint directory: `training/compare/outputs/imwut_compare_arm_a_sensor_only_35k_20260907/`.
- Trained: `last.pt` at 35,000 steps; control: the same run's `initial.pt`.
- Model identities: `halo_compare@sensor_only_trained` and `halo_compare@sensor_only_step0`.
- Frozen released baselines: HARNet-5, UniMTS, ImageBind, NormWear; no locally pretrained baselines.
- Five support seeds, identical serialized query/support plans for every model.
- k is independent enrolled executions **per candidate**, not windows or optimizer steps.
- Primary enrollment: cross-subject, same-configuration; fixed eight-dataset cohort for k=1,2,4,8.
- Each support execution is pooled to one vector. Generic readouts use normalized window vectors;
  HALO's native engine uses raw-scale means consistent with its training path.
- All scores are macro F1 on a 0-100 scale. Pool query predictions within a cell, average cells/seeds
  within each dataset, then give each dataset equal weight. The primary tables below combine all
  eight datasets, without giving the historical ordinary/specialized groups equal weight.
- Paired subject-macro differences and bootstrap intervals in the machine-readable tables are
  separate estimands, not confidence intervals for these cell-based aggregate scores.
- No training or test-driven hyperparameter selection was performed during this evaluation.

## Primary enrollment curve

| encoder | readout | k=1 | k=2 | k=4 | k=8 |
|---|---|---:|---:|---:|---:|
| HALO trained | native support comparator | 50.71 | 55.61 | 58.99 | 61.28 |
| HALO trained | 1-NN | 50.68 | 56.84 | 61.42 | 64.81 |
| HALO trained | prototype | 50.68 | 54.99 | 58.03 | 60.38 |
| HALO trained | ridge | 47.77 | 52.03 | 56.58 | 60.64 |
| HARNet | 1-NN | 46.13 | 50.03 | 53.66 | 56.67 |
| HARNet | prototype | 46.13 | 49.53 | 52.57 | 54.57 |
| HARNet | ridge | 45.42 | 49.25 | 53.41 | 57.13 |
| UniMTS | 1-NN | 55.81 | 60.58 | 64.27 | 67.18 |
| UniMTS | prototype | 55.81 | 58.57 | 61.17 | 62.99 |
| UniMTS | ridge | 53.61 | 56.87 | 59.82 | 62.28 |
| ImageBind | 1-NN | 40.54 | 46.06 | 50.67 | 55.13 |
| ImageBind | prototype | 40.54 | 44.20 | 47.67 | 50.38 |
| ImageBind | ridge | 40.62 | 45.32 | 50.08 | 54.85 |
| NormWear | 1-NN | 23.71 | 26.42 | 28.95 | 31.77 |
| NormWear | prototype | 23.71 | 24.52 | 25.37 | 25.49 |
| NormWear | ridge | 19.53 | 19.84 | 20.10 | 19.84 |

## Initial-checkpoint control

| readout at initial checkpoint | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| native support comparator | 48.13 | 52.65 | 56.55 | 59.41 |
| 1-NN | 49.23 | 54.70 | 58.81 | 62.37 |
| prototype | 49.23 | 53.42 | 56.44 | 58.31 |
| ridge | 41.15 | 44.76 | 49.18 | 53.76 |

The k=8 training gain is +2.44 points for 1-NN and +1.87 for the native engine. This compares
different encoder checkpoints, not just turning the learned residual off on a trained encoder.
The initial checkpoint includes the normal initial calibration; it is not a separately pretrained
encoder. It remains surprisingly competitive, so the training gain should not be overstated.

## Per-dataset enrollment at k=8

| dataset | HALO native | HALO 1-NN | HARNet 1-NN | UniMTS 1-NN | ImageBind 1-NN | NormWear 1-NN |
|---|---:|---:|---:|---:|---:|---:|
| InclusiveHAR | 41.54 | 41.43 | 34.48 | 44.24 | 32.69 | 29.81 |
| MoniPar | 47.98 | 46.40 | 44.27 | 42.65 | 29.75 | 27.78 |
| MotionSense | 61.47 | 72.72 | 73.30 | 87.48 | 62.76 | 36.49 |
| RealWorld | 65.99 | 70.62 | 56.77 | 75.18 | 51.72 | 27.14 |
| Shoaib | 88.81 | 89.94 | 80.23 | 95.14 | 79.79 | 39.48 |
| SPAR | 72.54 | 71.58 | 52.33 | 64.98 | 70.62 | 29.08 |
| USC-HAD | 36.38 | 51.38 | 45.51 | 56.99 | 43.71 | 24.77 |
| UT-Complex | 75.57 | 74.43 | 66.45 | 70.82 | 69.99 | 39.63 |

The native engine helps on some datasets but loses about 15 points on USC-HAD and 11.25 on
MotionSense relative to HALO 1-NN. This is a measured failure pattern, not a demonstrated cause.
All k values, readouts, and supplemental datasets are retained in `per_dataset.csv` below.

## Native zero-shot on common coverage

The common intersection is seven stream cells from six datasets: InclusiveHAR, MoniPar, Shoaib,
SPAR (both wrists), TNDA-HAR, and UT-Complex. Average streams within datasets, then datasets.

| model | macro F1 | available datasets in full protocol |
|---|---:|---:|
| HALO trained | 10.01 | 6/10 |
| HALO initial control | 10.27 | 6/10 |
| UniMTS | 29.99 | 10/10 |
| ImageBind | 11.90 | 10/10 |
| NormWear | 4.33 | 10/10 |
| HARNet | N/A: no native open-vocabulary rule | 0/10 |

HALO cannot supply its native zero-shot rule for MotionSense, RealWorld, Upper Limb Use, or USC-HAD
under its compatibility/candidate-exclusion policy. These are availability limitations, not zeros
or excluded evidence of success. TNDA is allowed for zero-shot, but its unverified independent
execution identity excludes enrollment claims. No ConSE bridge or added 1-NN/text bridge is used.

## Runtime

| model/run | evaluation seconds |
|---|---:|
| HALO trained | 106.9 |
| HALO initial control | 108.9 |
| HARNet | 27.1 |
| UniMTS | 63.1 |
| ImageBind | 56.9 |
| NormWear | 328.7 |

Total measured model-run time was about 11.5 minutes on the RTX 4090, excluding orchestration,
tests, and report assembly. No new model training was launched.

## Artifacts and limitations

- `eval/adaptation_results/matched_20260908/`: six JSON artifacts with exact source/checkpoint
  fingerprints, clean-worktree provenance, metrics, support details, and timing.
- `eval/adaptation_tables/matched_20260908/`: `tables.md`, `cells.csv`, `per_dataset.csv`,
  `dataset_macro.csv`, `zero_shot_common_cells.csv`, and `paired_deltas.json`.
- Assembler accepted all six artifacts and produced 16,096 scored cell/readout rows and 1,292
  paired subject-comparison groups. These are repeated conditions, not independent experiments.
- Raw outputs stay local and Git-ignored; this compact report and source/manifest are tracked.
- k=16, same-subject, and cross-configuration results are supplemental, not merged into the primary
  fixed-cohort curve. Generic controls remain available when HALO's native compatibility rule
  declines a support configuration.
- Random aliases test renaming, not new physical actions. Generic readouts do not use their text.
- One HALO training seed was tested; five enrollment draws are not five independent trained models.
- The test datasets have previously informed development. This is not a fresh sealed-test claim.
- Released models have different pretraining, modalities, sizes, and native temporal inputs. This
  is a system comparison, not a controlled architecture-superiority experiment. Exact upstream
  checkpoint training-data overlap remains a disclosure limitation, not a verified absence of overlap.

Reassemble with:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python -m eval.assemble_adaptation \
  --manifest eval/manifests/adaptation_v2_20260907.json.gz \
  --inputs eval/adaptation_results/matched_20260908/*.json \
  --out-dir eval/adaptation_tables/matched_20260908
```
