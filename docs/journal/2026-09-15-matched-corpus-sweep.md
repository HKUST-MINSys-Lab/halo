# Matched-corpus arms: debug sweep and profiling (2026-09-15)

Sweep and profile of the M2 code before any full run. Four defects found and fixed with regression
tests; one disclosure recorded; optimisation measured rather than assumed. Suite: **883 passed,
1 skipped**. No full run has been made.

## 1. Defects found and fixed

**D1 — mixed-rate groups were padded to the batch's longest signal before being cut to the
trunk's clip.** Windows are grouped by acquisition rate for anti-aliased resampling. The first
implementation padded every group to the longest resampled length by repeating the final sample,
then applied each model's clip rule. For LiMU-BERT that padding became **extra one-second clips
that were averaged into the representation**; for harnet it shifted the centre crop into fabricated
data. A window's embedding therefore depended on the acquisition rates of *other windows in its
batch*. Fixed by cutting each group by its own length and batching once.
Test: `test_mixed_rate_groups_are_encoded_at_their_own_length`.

**D2 — the UniMTS joint lookup read one GPU element per sensor.** `int(device_of_sensor[row,
sensor])` inside a per-window, per-sensor loop forced roughly a thousand synchronisations per step.
Resolved on one host copy and transferred once.

**D3 — a window with no usable device still received an embedding.** In the native-fusion path an
all-zero skeleton was encoded and the result stored, so a recording that never met the channel
contract produced a vector that looked like evidence. Now zeroed, with `device_present` false.
Test: `test_unimts_window_with_no_usable_device_is_not_given_an_embedding`.

**D4 — harnet cannot accept a short input.** Its ResNet pads circularly and raises
`Padding value causes wrapping around more than once` below its 5 s contract. The adaptive clip in
§3 had to become per-contract: harnet never shortens, UniMTS may.
Test: `test_a_trunk_that_cannot_take_a_short_input_is_never_given_one`.

## 2. Disclosure, not a defect: gravity

Two of the 31 training streams are gravity-removed (`kuhar/phone_waist`, `xrf_v2/airpods_ear`),
about 4% of windows. harnet and UniMTS are gravity-dependent accelerometer models and cannot
express that a stream has had gravity removed; HALO gates its polarization features on it
explicitly. `baseline_backbone.py` handles this by dropping such rows from *every* arm.

**We feed them to every arm instead, and disclose it.** Dropping them would change the corpus per
arm and break the single invariant M2 exists to hold. The resulting disadvantage is a real
architectural difference, not one we introduced. State the 4% in the results table.

## 3. Optimisation: measured, mostly negative

Per-stage timings, 24 steps, 4 workers, mean milliseconds. `encode` is forward only.

| arm | step | loader wait | encode | backward | other |
|---|---:|---:|---:|---:|---:|
| HALO | 55 | 21 (38%) | 16 | 11 | 7 |
| harnet5 | 86 | 24 (28%) | 30 | 25 | 7 |
| LiMU-BERT | 91 | 26 (29%) | 26 | 33 | 6 |
| UniMTS | 584 | 31 (5%) | 173 | 371 | 9 |

**What worked.** Two changes to UniMTS, in isolation on a realistic batch (157 windows, 4 devices):

| configuration | encoder fwd+bwd | peak memory |
|---|---:|---:|
| original: checkpoint chunk 32, wrap-pad to the released 10 s | 307 ms | 1.7 GiB |
| chunk 64, real 8 s window | **185 ms** (1.66x) | 2.7 GiB |
| the above, compiled | 146 ms (2.1x) | 2.7 GiB |

*Chunk size.* Gradient checkpointing bounds peak memory by the **chunk**, not the batch: measured
3.4 GiB at both 144 and 429 recordings. The original chunk of 32 was needlessly conservative.

*Real window length.* UniMTS's released convention is 200 samples (10 s at 20 Hz); our windows are
8 s, so the previous code wrap-padded 160 to 200 and **fabricated a quarter of every input**. The
ST-GCN is fully convolutional in time and there is no released weight to match in a random-init
arm, so the real length is both faithful and faster. A `--matched-pretrained` arm keeps 200.

**What did not work, and is worth recording so nobody retries it.**

* **End-to-end, UniMTS improved only 6%**: 1.50 -> 1.60 steps/s, 7.4 h -> 7.0 h for 40k. A 1.66x
  encoder speedup did not translate, and the profiler's `backward` stage did not move at all. I
  could not reconcile the isolated and end-to-end measurements in this session. **Treat 7.0 h as
  the number and this as an open question**; the remaining cost is not where the isolated benchmark
  says it is.
* **`torch.compile` is not usable.** On UniMTS it fails inside the trainer with
  `module 'model' has no attribute 'torch'`: its ST-GCN lives in a repo-local module also named
  `model`, and the adapter restores HALO's package after import, so TorchDynamo cannot resolve the
  class globals. It is now refused with that explanation. On harnet (13.19 vs 13.42 steps/s) and
  LiMU-BERT (10.14 vs 10.49) compilation is *slightly worse* than eager: those arms are not
  encoder-bound.
* **More loader workers do not help.** HALO at 4, 6 and 8 workers: 56, 57, 56 ms per step. The 38%
  "loader wait" is not worker starvation.

## 4. Revised budget

| arm | steps/s | 40k steps |
|---|---:|---:|
| HALO | 19.4 | 34 min |
| harnet5 | 13.4 | 50 min |
| LiMU-BERT | 10.5 | 64 min |
| UniMTS | 1.60 | 7.0 h |

Unchanged conclusion: LiMU-BERT plus harnet5 is about 2 GPU-hours and is the block to run first.

## 5. Answering "is it an efficiency thing?"

Partly, and the honest split is:

* **UniMTS** was carrying real waste: a quarter of its input was fabricated padding and its
  checkpoint chunk was too small. That is fixed, worth 1.66x on the encoder, but only 6%
  end-to-end. Its remaining cost is inherent: a 22-node graph convolution over 160 frames runs the
  full skeleton even though at most four nodes carry data.
* **NormWear** is not an efficiency problem. Its 136 M-parameter backbone emits 560 patch tokens
  per channel and measures 46.6 windows/s forward-only, giving ~183 GPU-hours for 40k steps. No
  chunking or compilation changes that order of magnitude.
* **HALO, harnet and LiMU-BERT** are already near their floor. Nothing in this sweep moved them,
  and compilation made them marginally worse.
