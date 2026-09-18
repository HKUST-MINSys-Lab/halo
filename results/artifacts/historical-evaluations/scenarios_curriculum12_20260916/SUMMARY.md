# Stage A deployment-scenario summary

**Checkpoint:** `halo_fixed_mr_residual_v3_curriculum12_40k_20260916/best_internal.pt`

**Selection:** step 30,000, selected only by subject-held-out validation macro-F1 (0.760)
**Protocol:** 8-second windows, `k in {1, 8}`, identical immutable manifests for HALO and every
released baseline, and no scenario result used for checkpoint selection.

This is the concise entry point for reasoning about the deployment scenarios. `RESULTS.md` in this
directory retains every HALO cell, split, readout, and confidence interval. Released-baseline rows
come from `../scenarios_full_20260916`, whose `manifests.jsonl` is byte-identical to this run.

## Scenario glossary

1. **Partial enrollment:** only part of the declared candidate vocabulary has target-domain
   supports. This tests whether semantic evidence can recover a true label with missing enrollment.
2. **Cross placement:** query and support perform the same candidate activities but come from
   different body placements. This tests acquisition-location transfer.
3. **Cross dataset:** query and support come from different datasets under comparable sensor
   configurations. This tests corpus and collection-protocol transfer.
4. **Missing modality:** gyroscope input is removed from the query, support, or both. The table
   averages those perturbed conditions and excludes the full-sensor control.
5. **Rate mismatch:** the query is resampled to 20, 25, or 100 Hz while support stays native. The
   table averages the mismatched conditions and excludes the matched-rate control.
6. **New domain:** MM-Fit, SPAR, and Upper Limb Use are absent from classifier training. Both full
   and partial enrollment variants are retained because each is a real deployment condition.
7. **Device-set mismatch:** query and support contain different single/composite device sets. The
   matched-device controls are excluded from the table.
8. **Cold start:** the MM-Fit deployment domain is introduced without prior in-domain classifier
   training. This is one domain-level cell, so it should not be generalized by itself.

## Aggregate macro-F1

Values are unweighted means over valid perturbed cells. Both HALO rows use the same Stage A
checkpoint, whose encoder and residual classifier were trained end to end together. `HALO 1-NN`
removes the classifier only at evaluation, so it diagnoses that jointly trained encoder; it is not
the separate differentiable-neighbours-trained encoder. External models use the shared 1-NN
readout. `N/A` is an explicit incompatibility.

### k = 1

| scenario | HALO classifier | HALO 1-NN | HARNet | LiMU-BERT-X | UniMTS | NormWear |
|---|---:|---:|---:|---:|---:|---:|
| partial enrollment | **49.9** | 27.7 | 24.2 | 25.6 | 26.1 | 12.8 |
| cross placement | **59.7** | 54.5 | 42.6 | 47.5 | 50.7 | 16.9 |
| cross dataset | **69.6** | 65.4 | 52.7 | 35.1 | 58.6 | 30.2 |
| missing modality | 54.3 | 53.8 | 47.4 | N/A | **56.6** | 22.0 |
| rate mismatch | 56.6 | 56.0 | 41.4 | 53.7 | **57.4** | 21.6 |
| new domain | 38.1 | **39.9** | 23.3 | 38.1 | 28.3 | 13.1 |
| device-set mismatch | 64.1 | 62.7 | 49.0 | **76.6** | 55.0 | 19.8 |
| cold start | 23.3 | **23.8** | 17.6 | 4.6 | 19.6 | 8.2 |

### k = 8

| scenario | HALO classifier | HALO 1-NN | HARNet | LiMU-BERT-X | UniMTS | NormWear |
|---|---:|---:|---:|---:|---:|---:|
| partial enrollment | **51.1** | 30.6 | 27.7 | 30.2 | 30.3 | 16.4 |
| cross placement | **64.5** | 60.7 | 50.3 | 53.2 | 56.7 | 21.2 |
| cross dataset | **73.4** | 72.0 | 59.9 | 39.4 | 65.0 | 35.3 |
| missing modality | 61.9 | 63.6 | 59.2 | N/A | **70.7** | 29.6 |
| rate mismatch | 62.3 | 66.4 | 50.3 | 70.8 | **71.5** | 29.0 |
| new domain | **49.0** | 47.8 | 32.0 | 44.0 | 36.3 | 18.0 |
| device-set mismatch | 71.3 | 70.9 | 60.7 | **85.9** | 63.0 | 29.1 |
| cold start | 26.2 | **29.6** | 20.0 | 4.8 | 25.7 | 9.1 |

## Historical DN-trained encoder comparison

The historical differentiable-neighbours checkpoint was trained before the deployment curriculum
with `k in {1, 2, 4, 8}` and then evaluated here using a plain 1-NN readout on the same scenario
protocol. The Stage A column is the encoder trained jointly through the learned classifier with
curriculum changes 1–2. Positive deltas favor Stage A.

| scenario | historical DN k1 | Stage A k1 | delta | historical DN k8 | Stage A k8 | delta |
|---|---:|---:|---:|---:|---:|---:|
| partial enrollment | 27.5 | 27.7 | +0.2 | 30.6 | 30.6 | +0.0 |
| cross placement | 55.1 | 54.5 | -0.6 | 61.0 | 60.7 | -0.3 |
| cross dataset | 60.7 | 65.4 | +4.7 | 68.4 | 72.0 | +3.6 |
| missing modality | 53.1 | 53.8 | +0.7 | 62.0 | 63.6 | +1.6 |
| rate mismatch | 56.8 | 56.0 | -0.8 | 65.7 | 66.4 | +0.7 |
| new domain | 36.7 | 39.9 | +3.2 | 45.9 | 47.8 | +1.9 |
| device-set mismatch | 64.3 | 62.7 | -1.6 | 72.8 | 70.9 | -1.9 |
| cold start | 16.8 | 23.8 | +7.0 | 19.6 | 29.6 | +10.0 |

This comparison does not isolate the curriculum: the objectives and training budgets differ. A
40k-step curriculum-matched DN arm is smoke-tested but deliberately not launched pending approval.
Its support-only objective uses the same acquisition mixture and conditional complete/partial
ratio, but excludes zero-support queries because no differentiable neighbour target exists there.
The exhaustive historical rows are in
[`../scenarios_neighbors_precurriculum_20260916/RESULTS.md`](../scenarios_neighbors_precurriculum_20260916/RESULTS.md).

## Per-dataset macro-F1

`Best external` is the strongest valid released-model 1-NN result in that cell; the exhaustive
artifact contains every external model separately. Multiple perturbation variants within one
scenario/dataset are averaged with equal weight.

### Partial enrollment

| dataset | HALO cls k1 | HALO 1-NN k1 | best external k1 | HALO cls k8 | HALO 1-NN k8 | best external k8 |
|---|---:|---:|---|---:|---:|---|
| InclusiveHAR | 39.0 | 19.1 | 20.0 HARNet | 36.7 | 20.4 | 24.4 UniMTS |
| MotionSense | 60.5 | 25.1 | 27.8 UniMTS | 63.8 | 29.1 | 31.8 UniMTS |
| RealWorld | 40.2 | 19.7 | 19.4 UniMTS | 40.8 | 22.3 | 21.7 UniMTS |
| Shoaib | 64.2 | 41.0 | 38.5 UniMTS | 65.7 | 44.6 | 43.2 UniMTS |
| USC-HAD | 44.5 | 18.9 | 15.0 LiMU-BERT-X | 45.6 | 22.5 | 20.0 UniMTS |
| UT-Complex | 36.7 | 29.1 | 27.6 LiMU-BERT-X | 39.6 | 30.6 | 32.0 LiMU-BERT-X |

### Cross placement

| dataset | HALO cls k1 | HALO 1-NN k1 | best external k1 | HALO cls k8 | HALO 1-NN k8 | best external k8 |
|---|---:|---:|---|---:|---:|---|
| RealWorld | 35.1 | 30.5 | 40.4 UniMTS | 38.5 | 34.3 | 46.3 UniMTS |
| Shoaib | 71.9 | 66.5 | 55.9 UniMTS | 77.5 | 73.9 | 61.9 UniMTS |

### Cross dataset

| dataset | HALO cls k1 | HALO 1-NN k1 | best external k1 | HALO cls k8 | HALO 1-NN k8 | best external k8 |
|---|---:|---:|---|---:|---:|---|
| InclusiveHAR | 66.9 | 63.1 | 53.9 HARNet | 71.4 | 70.3 | 51.8 HARNet |
| MotionSense | 65.6 | 58.6 | 60.7 UniMTS | 67.2 | 62.1 | 67.6 UniMTS |
| RealWorld | 64.2 | 62.8 | 54.7 UniMTS | 68.5 | 66.5 | 63.3 UniMTS |
| Shoaib | 77.0 | 69.5 | 66.2 UniMTS | 84.5 | 80.4 | 79.2 UniMTS |
| USC-HAD | 79.6 | 67.9 | 56.3 UniMTS | 80.9 | 72.7 | 58.7 HARNet |
| UT-Complex | 68.9 | 71.2 | 59.8 UniMTS | 73.5 | 81.0 | 68.8 UniMTS |

### Missing modality

| dataset | HALO cls k1 | HALO 1-NN k1 | best external k1 | HALO cls k8 | HALO 1-NN k8 | best external k8 |
|---|---:|---:|---|---:|---:|---|
| InclusiveHAR | 35.4 | 32.4 | 33.5 UniMTS | 37.9 | 35.7 | 42.7 UniMTS |
| MotionSense | 59.4 | 54.8 | 68.4 UniMTS | 66.3 | 63.6 | 84.0 UniMTS |
| RealWorld | 54.1 | 61.4 | 58.3 UniMTS | 65.0 | 75.3 | 71.7 UniMTS |
| Shoaib | 67.7 | 66.2 | 72.9 UniMTS | 76.5 | 78.4 | 88.8 UniMTS |
| USC-HAD | 44.4 | 43.4 | 38.6 UniMTS | 49.0 | 51.3 | 54.2 UniMTS |
| UT-Complex | 51.5 | 52.2 | 51.5 UniMTS | 62.0 | 62.1 | 64.9 UniMTS |

### Rate mismatch

| dataset | HALO cls k1 | HALO 1-NN k1 | best external k1 | HALO cls k8 | HALO 1-NN k8 | best external k8 |
|---|---:|---:|---|---:|---:|---|
| InclusiveHAR | 39.8 | 34.5 | 33.3 UniMTS | 40.1 | 38.1 | 42.9 UniMTS |
| MotionSense | 70.1 | 65.1 | 68.3 UniMTS | 76.8 | 77.2 | 83.9 UniMTS |
| RealWorld | 47.1 | 54.8 | 58.3 UniMTS | 54.4 | 65.3 | 71.6 UniMTS |
| Shoaib | 71.5 | 68.8 | 72.8 UniMTS | 78.2 | 81.9 | 89.8 LiMU-BERT-X |
| USC-HAD | 43.9 | 42.3 | 45.2 LiMU-BERT-X | 45.9 | 50.0 | 58.5 LiMU-BERT-X |
| UT-Complex | 48.1 | 53.1 | 55.5 LiMU-BERT-X | 56.8 | 65.1 | 72.5 LiMU-BERT-X |

### New domain

| dataset | HALO cls k1 | HALO 1-NN k1 | best external k1 | HALO cls k8 | HALO 1-NN k8 | best external k8 |
|---|---:|---:|---|---:|---:|---|
| MM-Fit | 46.8 | 47.4 | 51.2 LiMU-BERT-X | 57.8 | 55.5 | 55.5 LiMU-BERT-X |
| SPAR | 49.4 | 53.4 | 44.8 LiMU-BERT-X | 60.9 | 57.8 | 52.4 LiMU-BERT-X |
| Upper Limb Use | 18.1 | 19.0 | 18.3 LiMU-BERT-X | 28.3 | 30.1 | 24.0 LiMU-BERT-X |

### Device-set mismatch

| dataset | HALO cls k1 | HALO 1-NN k1 | best external k1 | HALO cls k8 | HALO 1-NN k8 | best external k8 |
|---|---:|---:|---|---:|---:|---|
| RealWorld | 51.0 | 49.5 | 44.4 UniMTS | 59.3 | 56.1 | 50.9 UniMTS |
| Shoaib | 77.2 | 76.0 | 76.6 LiMU-BERT-X | 83.3 | 85.8 | 85.9 LiMU-BERT-X |

### Cold start

| dataset | HALO cls k1 | HALO 1-NN k1 | best external k1 | HALO cls k8 | HALO 1-NN k8 | best external k8 |
|---|---:|---:|---|---:|---:|---|
| MM-Fit | 23.3 | 23.8 | 19.6 UniMTS | 26.2 | 29.6 | 25.7 UniMTS |

## Interpretation

Stage A is strongest on partial enrollment, cross-dataset transfer, and Shoaib cross-placement.
Its encoder also improves the cold-start 1-NN floor. The learned classifier still gives back useful
encoder geometry under rate mismatch, missing modalities, and several high-k cells; that measured
failure is the direct motivation for Stage B's acquisition perturbations and adaptive semantic gate.
