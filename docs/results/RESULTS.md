# HALO results record

This is the promoted result record for the current support-conditioned HAR design. Retired
future-JEPA variants are archived separately and are not mixed into this table.

> **Protocol boundary (2026-09-18):** historical tables retain their original artifacts but must
> not be extended with new checkpoints. New runs use `deployment-scenarios-v5-20260918` and
> `sealed-manifest-v2-20260916`, with the restored NumPy manifest draw, corrected k=0 filtering,
> independent enrollment/acquisition sampler, and mandatory companion cosine 1-NN rows. Matched
> comparisons must re-score every checkpoint under one protocol. The v4 table below is historical.

## Representative scenario protocol - 2026-09-17

Deployment scenarios are a fixed set of seven stress tests, not a full Cartesian sweep. The
completed run uses every scenario at `k=0`, `k=1`, `k=4`, `k=8`, and `k=32` with an 8-second
analysis window. It evaluates HALO and the five retained baselines (`HARNet-5`, `HARNet-10`,
`LiMU-BERT-X`, `UniMTS`, and `NormWear`) on identical episodes. HALO uses its learned classifier;
external encoders use the fixed equal-weight normalized fusion adapter. The complete run contains
4,341 promoted result rows after excluding the retired cold-start condition. The immutable source
artifact contains 65 additional cold-start rows retained only for reproducibility.

The ordinary sealed evaluation is separate and may use the larger `k=0..64`, 4/8/16-second grid.
Do not multiply that sealed grid into the scenario experiment: `7 scenarios x 8 k values x 3
windows x all dataset variants` is an exploratory Cartesian sweep, not the representative
scenario protocol. Such exploratory runs must use a separate output directory and must not
replace the representative tables.

The concise results, readout-fairness diagnostic, limitations, hashes, and tracked machine-readable
artifacts are in the
[2026-09-17 scenario results](../journal/2026-09-17-scenario-evaluation-results.md). The original local
run is
`training/support_classifier/evaluations/scenarios_halo_classifier_v3_20260917_profiled_v2_k0_1_4_8_32/`.

Every scenario run writes `progress.json` after each scored task. It records exact completed and
total protocol cells, progress within the current cell, elapsed time, a rolling task rate, ETA,
episode-construction time, and cumulative time attributed to each model. Console `[progress]`
lines mirror the durable file. Use these measurements for status and duration estimates; counting
plain `[scenarios]` log lines is not a reliable ETA because multi-device tasks are substantially
more expensive than ordinary cached cells.

## Deployment-scenario diagnostics - 2026-09-16

The Stage A acquisition/enrollment curriculum is reported separately from the ordinary sealed
k-curve because it historically tested eight perturbed deployment conditions. Cold start is now
retired; active v4 runs test the seven nonredundant conditions. The concise summary contains a
scenario glossary, aggregate comparisons against every released baseline, and per-dataset tables;
the adjacent exhaustive artifact retains every split, readout, and confidence interval:

- [Stage A scenario summary](../../training/support_classifier/evaluations/scenarios_curriculum12_20260916/SUMMARY.md)
- [Stage A exhaustive scenario results](../../training/support_classifier/evaluations/scenarios_curriculum12_20260916/RESULTS.md)

These rows use the same immutable manifests as the released-baseline scenario run. They are not
mixed numerically with the ordinary 4/8/16-second sealed comparison below. In the scenario summary,
`HALO 1-NN` is a diagnostic readout of the encoder jointly trained with the residual classifier;
it is not the older differentiable-neighbours-trained encoder.

## Current sealed comparison - 2026-09-14

This supersedes the 2026-09-13 summary below. It evaluates the validation-selected checkpoint from
the direct end-to-end fixed multi-resolution filterbank run with the **learned residual support
classifier**: 8-second training windows, `0.5/1/2/4 s` patch durations, gravity-referenced
polarization features, multi-device training. The encoder has **0.789M parameters**; the learned
classifier adds **1.41M** (2.203M total on classifier rows). Evaluation uses immutable,
execution-disjoint episodes identical for every provider, at three evidence durations. No sealed
result selected the checkpoint.

Full per-dataset matrices for all `k` and every readout, with confidence intervals, device
disclosures and padding flags, are in
[the canonical combined artifact](../../training/support_classifier/evaluations/sealed_comparison_residual_v3_8s_4res_40k_20260914/combined/RESULTS.md).

![k-curve](assets/k_curve_residual_v3_20260914.png)

Dataset-balanced macro F1, single-device cells. HALO rows name their readout: `classifier` is the
learned head; `centred neighbours` is the same head with the residual and text term disabled, which
is exactly centred differentiable neighbours and is HALO's parameter-free floor. Baselines use the
shared 1-NN readout; at `k=0` each model uses its disclosed zero-support path.

| window | model / readout | k=0 | k=1 | k=8 | k=32 | k=128 |
|---|---|---:|---:|---:|---:|---:|
| 4 s | HALO / classifier | **48.1** | **57.6** | 65.6 | 67.1 | 67.5 |
| 4 s | HALO / centred neighbours | - | 54.3 | 67.3 | 71.6 | 73.5 |
| 4 s | UniMTS / 1-NN | 29.3 | 50.4 | 63.6 | 67.2 | 70.9 |
| 4 s | LiMU-BERT-X / 1-NN | - | 46.8 | 62.4 | 67.9 | 70.7 |
| 4 s | HARNet / 1-NN | 33.8 | 39.5 | 50.0 | 55.5 | 60.0 |
| 4 s | NormWear / 1-NN | - | 20.9 | 28.0 | 32.9 | 37.8 |
| 8 s | HALO / classifier | **49.3** | **60.5** | 68.4 | 69.9 | 71.3 |
| 8 s | HALO / centred neighbours | - | 58.3 | 71.6 | 75.4 | 77.5 |
| 8 s | UniMTS / 1-NN | 30.7 | 52.4 | 66.0 | 70.6 | 75.1 |
| 8 s | LiMU-BERT-X / 1-NN | - | 49.2 | 65.9 | 71.4 | 73.5 |
| 8 s | HARNet / 1-NN | 35.6 | 44.4 | 56.4 | 62.0 | 66.3 |
| 8 s | NormWear / 1-NN | 3.5 | 21.4 | 29.4 | 34.9 | 40.2 |
| 16 s | HALO / classifier | **49.2** | **63.5** | 69.6 | 71.4 | 87.3 |
| 16 s | HALO / centred neighbours | - | 61.7 | 73.4 | 76.2 | 92.2 |
| 16 s | UniMTS / 1-NN | 30.6 | 54.3 | 67.7 | 72.1 | 90.6 |
| 16 s | LiMU-BERT-X / 1-NN | - | 51.0 | 67.2 | 74.0 | 91.7 |
| 16 s | HARNet / 1-NN | 36.9 | 47.8 | 60.2 | 66.0 | 86.3 |
| 16 s | NormWear / 1-NN | - | 23.1 | 31.1 | 38.3 | 54.3 |

### Per-dataset decomposition

These are the existing single-device rows underlying the aggregate table above, not a new
evaluation. Where a dataset has multiple eligible placements, its entry is the mean across those
placements, matching the aggregate calculation. `—` means that the cell was unavailable. These
2026-09-14 baseline rows predate the 2026-09-16 baseline-fidelity repairs and will be replaced after
the next full baseline evaluation; they are retained here only to decompose the reported aggregate.

#### 4-second windows: macro F1

| dataset | model / readout | k=0 | k=1 | k=8 | k=32 | k=128 |
|---|---|---:|---:|---:|---:|---:|
| MotionSense | HALO / classifier | 67.8 | 71.5 | 79.2 | 82.4 | 83.5 |
| MotionSense | HALO / centred neighbours | — | 66.0 | 79.7 | 85.6 | 88.7 |
| MotionSense | UniMTS / 1-NN | 38.0 | 64.5 | 80.3 | 85.2 | 89.6 |
| MotionSense | LiMU-BERT-X / 1-NN | 40.9 | 53.1 | 74.6 | 81.6 | 86.6 |
| MotionSense | HARNet / 1-NN | 46.7 | 49.4 | 62.3 | 70.4 | 75.7 |
| MotionSense | NormWear / 1-NN | 6.3 | 23.9 | 31.0 | 37.6 | 44.1 |
| RealWorld | HALO / classifier | 43.7 | 50.7 | 62.9 | 64.4 | 63.5 |
| RealWorld | HALO / centred neighbours | — | 48.8 | 65.2 | 69.6 | 71.7 |
| RealWorld | UniMTS / 1-NN | 29.2 | 49.4 | 62.4 | 64.4 | 67.0 |
| RealWorld | LiMU-BERT-X / 1-NN | — | — | — | — | — |
| RealWorld | HARNet / 1-NN | 25.4 | 33.4 | 43.5 | 48.6 | 51.9 |
| RealWorld | NormWear / 1-NN | 3.5 | 21.2 | 26.7 | 30.1 | 32.4 |
| Shoaib | HALO / classifier | 67.7 | 79.0 | 85.5 | 86.3 | 87.0 |
| Shoaib | HALO / centred neighbours | — | 73.5 | 85.9 | 88.5 | 89.5 |
| Shoaib | UniMTS / 1-NN | 36.8 | 71.6 | 86.1 | 89.5 | 91.4 |
| Shoaib | LiMU-BERT-X / 1-NN | 28.6 | 64.5 | 79.8 | 82.3 | 83.2 |
| Shoaib | HARNet / 1-NN | 48.0 | 54.3 | 69.1 | 75.0 | 80.5 |
| Shoaib | NormWear / 1-NN | 3.6 | 23.6 | 34.1 | 43.0 | 50.4 |
| InclusiveHAR | HALO / classifier | 36.5 | 34.8 | 37.6 | 36.1 | 36.4 |
| InclusiveHAR | HALO / centred neighbours | — | 31.0 | 36.4 | 37.6 | 38.5 |
| InclusiveHAR | UniMTS / 1-NN | 25.3 | 33.8 | 41.9 | 40.6 | 43.0 |
| InclusiveHAR | LiMU-BERT-X / 1-NN | 27.8 | 22.8 | 27.2 | 28.6 | 28.1 |
| InclusiveHAR | HARNet / 1-NN | 25.0 | 30.2 | 29.5 | 28.7 | 30.2 |
| InclusiveHAR | NormWear / 1-NN | 4.8 | 23.3 | 25.4 | 24.2 | 27.4 |
| USC-HAD | HALO / classifier | 34.8 | 50.7 | 57.8 | 59.8 | 60.4 |
| USC-HAD | HALO / centred neighbours | — | 50.4 | 64.7 | 70.1 | 72.9 |
| USC-HAD | UniMTS / 1-NN | 24.9 | 40.8 | 55.3 | 62.5 | 69.2 |
| USC-HAD | LiMU-BERT-X / 1-NN | 12.8 | 44.6 | 64.9 | 74.3 | 79.7 |
| USC-HAD | HARNet / 1-NN | 23.6 | 29.9 | 42.3 | 49.8 | 57.2 |
| USC-HAD | NormWear / 1-NN | 1.4 | 15.0 | 21.9 | 26.4 | 31.0 |
| UT-Complex | HALO / classifier | 38.1 | 58.8 | 70.7 | 73.6 | 74.5 |
| UT-Complex | HALO / centred neighbours | — | 56.1 | 72.0 | 77.9 | 79.7 |
| UT-Complex | UniMTS / 1-NN | 21.8 | 42.6 | 55.8 | 60.9 | 65.2 |
| UT-Complex | LiMU-BERT-X / 1-NN | 9.6 | 48.9 | 65.5 | 72.5 | 76.2 |
| UT-Complex | HARNet / 1-NN | 34.0 | 39.9 | 53.4 | 60.3 | 64.3 |
| UT-Complex | NormWear / 1-NN | 1.1 | 18.2 | 28.9 | 36.1 | 41.6 |

#### 8-second windows: macro F1

| dataset | model / readout | k=0 | k=1 | k=8 | k=32 | k=128 |
|---|---|---:|---:|---:|---:|---:|
| MotionSense | HALO / classifier | 69.9 | 74.6 | 84.1 | 85.2 | 86.2 |
| MotionSense | HALO / centred neighbours | — | 70.3 | 85.6 | 90.1 | 91.8 |
| MotionSense | UniMTS / 1-NN | 36.1 | 67.6 | 83.0 | 88.8 | 90.7 |
| MotionSense | LiMU-BERT-X / 1-NN | 43.1 | 53.3 | 78.8 | 85.9 | 89.3 |
| MotionSense | HARNet / 1-NN | 51.7 | 58.4 | 71.4 | 78.5 | 82.8 |
| MotionSense | NormWear / 1-NN | 6.2 | 20.9 | 31.3 | 40.9 | 48.5 |
| RealWorld | HALO / classifier | 44.3 | 51.7 | 65.0 | 67.0 | 73.5 |
| RealWorld | HALO / centred neighbours | — | 51.1 | 68.8 | 72.1 | 80.6 |
| RealWorld | UniMTS / 1-NN | 30.5 | 51.3 | 66.0 | 68.4 | 77.2 |
| RealWorld | LiMU-BERT-X / 1-NN | — | — | — | — | — |
| RealWorld | HARNet / 1-NN | 28.3 | 37.7 | 48.5 | 53.4 | 61.9 |
| RealWorld | NormWear / 1-NN | 3.6 | 18.9 | 25.0 | 28.7 | 32.3 |
| Shoaib | HALO / classifier | 68.7 | 82.6 | 88.1 | 89.4 | 89.6 |
| Shoaib | HALO / centred neighbours | — | 78.6 | 89.7 | 91.6 | 92.1 |
| Shoaib | UniMTS / 1-NN | 37.6 | 73.8 | 88.9 | 91.8 | 93.5 |
| Shoaib | LiMU-BERT-X / 1-NN | 28.9 | 69.7 | 85.5 | 87.5 | 87.2 |
| Shoaib | HARNet / 1-NN | 46.3 | 61.3 | 76.6 | 82.1 | 86.6 |
| Shoaib | NormWear / 1-NN | 3.6 | 25.3 | 38.4 | 47.8 | 54.3 |
| InclusiveHAR | HALO / classifier | 36.6 | 35.2 | 37.6 | 36.8 | 36.5 |
| InclusiveHAR | HALO / centred neighbours | — | 32.2 | 38.7 | 41.0 | 40.9 |
| InclusiveHAR | UniMTS / 1-NN | 31.6 | 32.1 | 37.6 | 39.1 | 41.3 |
| InclusiveHAR | LiMU-BERT-X / 1-NN | 28.0 | 23.8 | 27.0 | 27.9 | 27.1 |
| InclusiveHAR | HARNet / 1-NN | 26.2 | 28.6 | 30.3 | 30.5 | 29.5 |
| InclusiveHAR | NormWear / 1-NN | 4.8 | 23.5 | 23.4 | 23.6 | 26.0 |
| USC-HAD | HALO / classifier | 35.4 | 53.1 | 59.3 | 61.5 | 62.3 |
| USC-HAD | HALO / centred neighbours | — | 53.9 | 68.2 | 73.7 | 75.1 |
| USC-HAD | UniMTS / 1-NN | 26.5 | 38.9 | 55.2 | 66.7 | 74.8 |
| USC-HAD | LiMU-BERT-X / 1-NN | 12.6 | 45.4 | 66.5 | 77.7 | 82.0 |
| USC-HAD | HARNet / 1-NN | 27.4 | 33.1 | 48.7 | 58.3 | 64.4 |
| USC-HAD | NormWear / 1-NN | 1.4 | 16.2 | 24.7 | 28.9 | 35.3 |
| UT-Complex | HALO / classifier | 40.8 | 65.5 | 76.7 | 79.4 | 79.8 |
| UT-Complex | HALO / centred neighbours | — | 63.9 | 78.4 | 83.9 | 84.7 |
| UT-Complex | UniMTS / 1-NN | 21.8 | 50.3 | 65.5 | 68.9 | 73.1 |
| UT-Complex | LiMU-BERT-X / 1-NN | 9.5 | 53.7 | 71.6 | 78.2 | 81.7 |
| UT-Complex | HARNet / 1-NN | 33.6 | 47.5 | 63.2 | 69.1 | 72.6 |
| UT-Complex | NormWear / 1-NN | 1.1 | 23.5 | 33.5 | 39.3 | 44.9 |

#### 16-second windows: macro F1

| dataset | model / readout | k=0 | k=1 | k=8 | k=32 | k=128 |
|---|---|---:|---:|---:|---:|---:|
| MotionSense | HALO / classifier | 70.0 | 76.3 | 83.7 | 86.1 | 87.3 |
| MotionSense | HALO / centred neighbours | — | 72.6 | 85.8 | 90.6 | 92.2 |
| MotionSense | UniMTS / 1-NN | 43.3 | 68.4 | 84.1 | 87.4 | 90.6 |
| MotionSense | LiMU-BERT-X / 1-NN | 42.7 | 56.0 | 78.4 | 87.6 | 91.7 |
| MotionSense | HARNet / 1-NN | 50.4 | 61.3 | 75.6 | 83.9 | 86.3 |
| MotionSense | NormWear / 1-NN | 6.1 | 24.8 | 32.6 | 43.6 | 54.3 |
| RealWorld | HALO / classifier | 44.3 | 54.2 | 67.0 | 67.3 | — |
| RealWorld | HALO / centred neighbours | — | 54.0 | 70.7 | 70.1 | — |
| RealWorld | UniMTS / 1-NN | 31.2 | 53.5 | 67.6 | 67.3 | — |
| RealWorld | LiMU-BERT-X / 1-NN | — | — | — | — | — |
| RealWorld | HARNet / 1-NN | 29.3 | 41.4 | 52.1 | 56.3 | — |
| RealWorld | NormWear / 1-NN | 3.6 | 19.2 | 26.0 | 30.3 | — |
| Shoaib | HALO / classifier | 68.5 | 83.9 | 89.4 | 90.6 | — |
| Shoaib | HALO / centred neighbours | — | 80.9 | 91.4 | 92.8 | — |
| Shoaib | UniMTS / 1-NN | 38.2 | 72.9 | 88.7 | 92.1 | — |
| Shoaib | LiMU-BERT-X / 1-NN | 28.8 | 69.8 | 86.3 | 89.1 | — |
| Shoaib | HARNet / 1-NN | 48.8 | 65.6 | 81.6 | 87.0 | — |
| Shoaib | NormWear / 1-NN | 3.6 | 26.4 | 40.9 | 50.5 | — |
| InclusiveHAR | HALO / classifier | 36.8 | 38.0 | 35.7 | 36.0 | — |
| InclusiveHAR | HALO / centred neighbours | — | 34.2 | 37.8 | 40.0 | — |
| InclusiveHAR | UniMTS / 1-NN | 26.7 | 34.7 | 36.3 | 40.7 | — |
| InclusiveHAR | LiMU-BERT-X / 1-NN | 28.0 | 24.5 | 27.4 | 31.3 | — |
| InclusiveHAR | HARNet / 1-NN | 27.6 | 28.7 | 29.7 | 27.5 | — |
| InclusiveHAR | NormWear / 1-NN | 4.9 | 24.3 | 23.4 | 24.8 | — |
| USC-HAD | HALO / classifier | 35.8 | 53.6 | 60.1 | 64.2 | — |
| USC-HAD | HALO / centred neighbours | — | 56.6 | 69.6 | 75.8 | — |
| USC-HAD | UniMTS / 1-NN | 25.5 | 39.2 | 57.4 | 68.4 | — |
| USC-HAD | LiMU-BERT-X / 1-NN | 12.8 | 45.6 | 67.5 | 78.8 | — |
| USC-HAD | HARNet / 1-NN | 28.3 | 34.5 | 52.4 | 64.3 | — |
| USC-HAD | NormWear / 1-NN | 2.2 | 17.6 | 25.5 | 33.8 | — |
| UT-Complex | HALO / classifier | 39.9 | 74.8 | 81.9 | 84.2 | — |
| UT-Complex | HALO / centred neighbours | — | 72.2 | 85.1 | 88.1 | — |
| UT-Complex | UniMTS / 1-NN | 18.8 | 57.1 | 71.8 | 76.3 | — |
| UT-Complex | LiMU-BERT-X / 1-NN | 9.1 | 59.3 | 76.4 | 83.5 | — |
| UT-Complex | HARNet / 1-NN | 37.2 | 55.2 | 70.1 | 76.7 | — |
| UT-Complex | NormWear / 1-NN | 1.1 | 26.4 | 37.9 | 46.9 | — |

### Reading this table

* **`k=0` is the headline change.** The learned classifier reaches 49.3 at 8 s against HARNet's
  training-bank bridge at 35.6 and UniMTS's released native text head at 30.7. HALO's own bridge
  scores 38.3, so +11.0 of the gain is the classifier rather than the encoder.
* **`k=1` is the other regime the classifier owns**: at one support per candidate, 1-NN, prototype,
  ridge and the soft vote are mathematically identical, so only a second evidence source can move
  it. 60.5 against 58.6 clears the ~1-point enrollment-draw noise.
* **Above `k=2` the learned residual currently regresses** (−1.9 to −6.3 against centred
  neighbours), traced to a λ schedule whose top bucket covers all `k>=8` and a training range that
  stopped at k=8. Until that is fixed, **centred neighbours is HALO's strongest enrolled readout**
  and is the row to compare against baselines at high k.
* `k=128` at 16 s is **MotionSense only** and is not a six-dataset aggregate.
* Compare within a window duration, never across: query counts differ, so confidence intervals do.

Multi-device composite cells are reported separately in the combined artifact. At 8 s and k=1,
HALO gains **+11.1** (RealWorld) and **+8.9** (Shoaib) over its own single-placement cells, against
UniMTS's native skeleton fusion at +3.7 / +4.1.

**How HALO fuses devices.** One token per `(patch, sensor)`, where a *sensor* is one 3-axis modality
on one device, so an accelerometer and a gyroscope on the same wrist stay separate tokens. Device,
modality and placement identity enter through per-sensor **text** conditioning (axis lives in the
role text, so no fact is injected twice), and a single learned `RecordingAttentionPool` query
attends over every `patch x sensor` token at once. There is no separate device-fusion stage and no
placement-specific parameter: an unseen placement is expressible because identity is text. The pool
is ~0.13M of the 0.789M encoder and is trained end to end with everything else. An earlier version
of this line credited the gain to a parameter-free hierarchical device mean; that path exists in
`encoder.py` but is **overridden** by the learned pool in every trained checkpoint, so it produced
none of the numbers in this file. See
[docs/journal/2026-09-14-multi-device-pooling-correction.md](../journal/2026-09-14-multi-device-pooling-correction.md).

The narrative behind these numbers — why JEPA and the continuous kernel were dropped, what is and
is not a controlled comparison, and the open limitations — is in
[docs/journal/2026-09-14-design-narrative.md](../journal/2026-09-14-design-narrative.md).

**Protocol note.** These numbers use the 4/8/16-second multi-device protocol introduced on
2026-09-13 and are **not comparable** to any result measured under the previous 6-second protocol,
including the tables below and the retired JEPA record. See
[docs/journal/2026-09-14-evaluation-rebuild.md](../journal/2026-09-14-evaluation-rebuild.md).

## Superseded sealed comparison - 2026-09-13

The table below evaluated the same encoder configuration with the parameter-free
differentiable-neighbours control and no learned classifier. It is retained as the immediately
preceding record; its encoder matches the current one under identical readouts (1-NN at 8 s: 59.0
vs 58.6 at k=1, 76.5 vs 77.1 at k=128).

## Historical: sealed enrollment comparison - 2026-09-12 (6-second protocol)

Both metrics are on a 0-100 scale and are averaged equally across the six sealed datasets. Accuracy
is the fraction of correct predictions within each dataset. Macro F1 gives every ground-truth class
equal weight within each dataset, regardless of its frequency. For every `k >= 1`, all rows use the
same parameter-free hard 1-NN classifier and the same deterministic,
execution-disjoint support/query manifests. `k` is the number of enrolled executions **per
candidate label**, so an episode with `C` candidates contains `C * k` supports. At `k=0`, no sealed
target recording is enrolled. HALO, HARNet, and LiMU-BERT-X instead use a labelled reference bank
built only from the eight classifier-training datasets: hard 1-NN predicts a training label and
ConSE bridges that prediction to the sealed candidate vocabulary. UniMTS and NormWear use their
released native text-aligned prediction paths.

"Open-set labels" means that the deployed model can emit labels outside its classifier-training
vocabulary, either from labelled supports or through its semantic bridge. "Native adaptation"
means that the model itself accepts labelled examples at inference. Applying the common 1-NN
evaluator to an external encoder does not turn that encoder into a native adaptation model. The
explicit `k=0 readout` column prevents a shared ConSE bridge from being mistaken for a released
model's native capability.

### Macro F1

| model | parameters | open-set labels | native adaptation | k=0 readout | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HARNet, released | 4.49M | no | no | train-bank 1-NN + ConSE | **36.97** | 42.40 | 46.78 | 50.80 | 53.51 | 56.50 | 59.48 | 61.07 | 63.89 |
| LiMU-BERT-X, released | 0.055M | no | no | train-bank 1-NN + ConSE | 24.97 | 50.37 | 57.17 | 62.03 | 66.50 | 69.61 | 70.98 | 72.42 | 73.53 |
| UniMTS, released | 68.61M | yes | no | native zero-support | 31.61 | 56.08 | 61.73 | 66.22 | 68.83 | 71.44 | 72.92 | 74.05 | 75.57 |
| NormWear, released | 1,293.86M | yes | no | native zero-support | 5.14 | 21.90 | 24.68 | 28.39 | 31.71 | 35.51 | 39.15 | 41.61 | 44.89 |

### Accuracy

| model | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HARNet, released | **41.07** | 43.56 | 47.75 | 51.46 | 53.98 | 56.78 | 59.56 | 61.09 | 63.82 |
| LiMU-BERT-X, released | 29.03 | 51.65 | 58.67 | 63.26 | 67.45 | 70.40 | 71.74 | 73.10 | 74.02 |
| UniMTS, released | 39.90 | 55.52 | 61.23 | 65.59 | 68.15 | 70.84 | 72.37 | 73.57 | 75.07 |
| NormWear, released | 15.95 | 22.42 | 25.15 | 28.86 | 32.12 | 35.97 | 39.68 | 42.06 | 45.50 |

Bold identifies the strongest score in each metric column. Promoted HALO rows will be added only
after the current direct end-to-end fixed-filterbank runs are evaluated with these manifests.

## Interpretation

- These are deployed-system comparisons, not parameter- or upstream-data-matched architecture
  ablations. The released models differ substantially in size and pretraining corpus.
- `k=0` does not mean an empty world model. It means zero target-dataset enrollment. The disclosed
  training-bank bridge lets representation-only models participate without fitting on a sealed
  recording, but it is not a native released zero-shot head. HARNet leads this condition at 36.97.

### Zero-target-enrollment results by sealed dataset

All entries are macro F1 on a 0-100 scale. The readout is identified in the main table.

| model | MotionSense | RealWorld | Shoaib | InclusiveHAR | USC-HAD | UT-Complex | equal-dataset mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| HARNet, released | **54.22** | 26.64 | **48.07** | 28.31 | **29.27** | **35.29** | **36.97** |
| LiMU-BERT-X, released | 42.61 | 22.43 | 34.83 | 27.58 | 12.17 | 10.20 | 24.97 |
| UniMTS, released | 34.04 | **36.25** | 37.03 | **32.16** | 26.09 | 24.09 | 31.61 |
| NormWear, released | 7.09 | 4.85 | 3.58 | 8.60 | 4.85 | 1.86 | 5.14 |

## Retired JEPA record

The prior JEPA rows, per-dataset values, telemetry interpretation, and machine-readable artifacts
are preserved in
[2026-09-13-retired-jepa-promoted-results.md](../journal/2026-09-13-retired-jepa-promoted-results.md)
and [2026-09-13-jepa-value-measured.md](../journal/2026-09-13-jepa-value-measured.md). They are
historical evidence, not active model references.

No sealed result was used to select a checkpoint or tune an operating threshold.
