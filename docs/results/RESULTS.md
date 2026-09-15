# HALO results record

This is the promoted result record for the current support-conditioned HAR design. Retired
future-JEPA variants are archived separately and are not mixed into this table.

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
