# 2026-09-14 — Where each baseline fails, and what our readout was leaving on the table

**Status:** analysis of sealed results, plus the independent neighbour-control diagnostics that
motivated the classifier design. No new runs; both read existing artifacts.

Sources: `evaluations/sealed_comparison_20260913/`,
`evaluations/neighbor_diagnostics_20260914/REPORT.md` (other agent).

---

## 1. Each baseline fails in a different deployment scenario

Per-cell macro F1 at 8 s, k=8, single-device:

| cell | HALO | UniMTS | LiMU-BERT-X | HARNet | NormWear |
|---|---:|---:|---:|---:|---:|
| inclusivehar / phone_waist | 36.8 | 37.6 | 27.0 | 30.3 | 23.4 |
| motionsense / phone_front_pocket | 85.4 | 83.0 | 78.8 | 71.4 | 31.3 |
| realworld / phone_forearm | 57.9 | 56.4 | – | 44.2 | 23.3 |
| realworld / phone_thigh | 65.7 | 70.2 | – | 49.9 | 27.6 |
| realworld / phone_waist | 77.2 | 71.3 | – | 51.3 | 24.1 |
| shoaib / phone_belt | 90.2 | 84.5 | 77.8 | 75.2 | 35.2 |
| shoaib / phone_left_pocket | 92.7 | 94.5 | 88.3 | 79.6 | 39.1 |
| shoaib / phone_right_pocket | 94.5 | 94.5 | 91.5 | 76.9 | 35.4 |
| shoaib / watch_wrist_proxy | 87.0 | 82.1 | 84.2 | 74.8 | 43.8 |
| usc_had / phone_hip | 65.8 | 55.2 | **66.5** | 48.7 | 24.7 |
| ut_complex / watch_wrist | 79.5 | 65.5 | 71.6 | 63.2 | 33.5 |

**HARNet** (4.49M, accel-only, 30 Hz, 5 s native): fails at short windows and low k. Its 4 s→16 s
gain is **+10.6**, more than double anyone else's, because 4 s is shorter than its native input. Its
k=1 is the lowest of the working models (47.8). Its Biobank *wrist* pretraining buys nothing visible
— UT-Complex wrist 63.2, Shoaib wrist-proxy 74.8, no better than its phone cells. But it leads
zero-shot at 37.2 via our ConSE bridge over its frozen features.

**UniMTS** (68.6M, ST-GCN over a 22-node SMPL skeleton, text-contrastive): fails on anything that
is not whole-body locomotion, and on the wrist. UT-Complex costs it **−10.2** against its locomotion
mean — the largest activity-type penalty in the table. Shoaib's wrist-proxy is 82.1 against 94.5 in
either pocket, a 12.5-point placement spread. The mechanism is its prior: a single populated wrist
joint doing a hand gesture is far off its synthetic full-body motion-capture manifold.

**LiMU-BERT-X** (0.055M, 1 s clips mean-pooled): fails at one-shot enrollment (k=1 = 56.8) but has
the **largest enrollment gain of anyone, +20.6 from k=1→32**. Mean-pooling eight 1-second clips
gives a smoothed, ambiguous descriptor — one example cannot place it, thirty-two can. It is also
**excluded from all of RealWorld** by a hard six-axis contract ("requires measured six-axis IMU;
masked channels: gyro"). For 55k parameters it wins USC-HAD outright (66.5 vs our 65.8).

**NormWear** (1.29B): fails everywhere, and the reason is mechanistic rather than "weak features".
Its embeddings on MotionSense have **mean pairwise cosine 0.966 and effective rank 1.4 in 2048
dimensions** — near-constant across windows. 1-NN on that is near chance; its native zero-shot head
*is* chance (3.5). Reporting the collapse number alongside the score is the fair statement.

**HALO's own worst cells:** InclusiveHAR 36.8, and RealWorld forearm 57.9. Our RealWorld placement
spread is **19.3** (forearm 57.9 / thigh 65.7 / waist 77.2) — **the largest of any model**, against
UniMTS's 14.9. On this dataset the heterogeneity claim cuts against us, and it should be said.

### Scenario matrix

| scenario | fails | holds |
|---|---|---|
| short window, k≤2 | HARNet, LiMU-BERT | HALO, UniMTS |
| wrist device, hand/complex activity | UniMTS | HALO, LiMU-BERT |
| accelerometer-only device | LiMU-BERT (cannot run) | everyone else |
| zero-shot | UniMTS native, NormWear native | HARNet bridge, HALO |
| upper-limb placement transfer | **HALO worst**, HARNet | UniMTS least bad |
| incline / posture pairs (InclusiveHAR) | everyone | no one |

## 2. Three cross-cutting findings

**Released text-aligned models lose zero-shot to a frozen encoder plus a bridge.** Ordering at 8 s:
HARNet bridge 37.2 > HALO bridge 35.1 > **UniMTS native 32.5** > LiMU-BERT bridge 26.1 >
**NormWear native 3.5**. The two models with *native* text heads are not the top two. Same lesson as
the classifier design: learn the metric, not the label map.
*(Since superseded on our side — the learned classifier reaches 49.3; see the 09-14 result entry.)*

**Every baseline's embedding space is far more concentrated than ours.** Mean pairwise cosine on one
shared cell: HALO **0.27** (effective rank 10.8), HARNet 0.58, UniMTS 0.78, LiMU-BERT 0.83, NormWear
0.97 (rank 1.4). This is exactly the regime where centring helps most, and it predicted the
measured +1.7 centring gain.

**InclusiveHAR is a universal failure with a dataset-level explanation.** 20 subjects, six classes:
walking, jogging, sitting, standing, **ramp_ascent, ramp_descent**. Two ramp classes against level
walking is an incline distinction that is subtle in a waist phone; sit/stand is a posture pair
needing gravity orientation. Its k-curve is flat (34→38), so enrollment is not the limiting factor
and no readout will fix it. Per-class error analysis is the right next step, not a model change.

## 3. What the neighbour diagnostics found — the classifier's actual target

Independent analysis on cached embeddings with matched decision rules on identical supports.

| k | measure | value |
|---:|---|---:|
| 1 | soft-neighbour accuracy | 66.3% |
| 1 | on **wrong** predictions, correct label's best support in top-5 | **92.5%** |
| 8 | on **wrong** predictions, correct label's best support ranked **first** | **15.6%** |
| 8 | on wrong predictions, correct support in top-5 | 72.6% |

Matched decision rules at 8 s: 1-NN 75.70, prototype 76.45, soft neighbours 77.35, **ridge 78.03**.

Two calibration facts that make any classifier claim assessable: enrollment-draw std is **0.36–0.54**
macro F1, so a real delta must clear ~1 point; and removing the top support flips **16.5%** of
predictions against **0.42%** for a random one, so the vote correctly relies on strong evidence and
a classifier must not diffuse it.

**Reading.** At k=1 the correct evidence is nearly always *close*, just not *closest* — so only a
second, independent source of evidence can help, which is the text term. At k≥8, 15.6% of errors
had the right answer ranked first and were outvoted by several mediocre wrong-label supports — a
*voting* failure, which a per-support scalar reweighting targets directly. Those two readings are
what the residual classifier was built from; see
`docs/design/SUPPORT_CLASSIFIER_DESIGN_20260914.md` §1.

## 4. Caveats

- Section 1 numbers are from the 2026-09-13 comparison (neighbours-trained encoder). Baseline rows
  are bit-identical in the 09-14 comparison, so the baseline conclusions carry; HALO's own rows have
  since moved.
- The neighbour diagnostics ran on cached **sealed** embeddings. They generate hypotheses; the
  architecture must be chosen on training-subject holdout.
- RealWorld is accelerometer-only in every grid, so our 19.3 spread is measured on three channels
  with no gyro polarization available.
