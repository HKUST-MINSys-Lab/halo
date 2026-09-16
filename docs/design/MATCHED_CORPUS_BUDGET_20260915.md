# Matched-corpus M2: implementation, hyperparameters and compute budget (2026-09-15)

**Status:** M2 built, unit-tested (21 tests), smoke-tested and throughput-measured on the real
corpus. **No full run has been made.** Implements level M2 of
`MATCHED_CORPUS_PLAN_20260915.md`. M1 (each baseline's own objective) is not built.

## 1. What was built

| file | role |
|---|---|
| `model/tokenizer/matched_encoder.py` | `MatchedCorpusEncoder`: a baseline trunk wearing HALO's support-classifier encoder contract, trainable end to end |
| `training/support_classifier/train.py` | `--encoder-arch {halo,limubert,harnet}`, `--matched-pretrained`, `--matched-d-model` |
| `training/tokenizer/eval_transfer.py` | `build_encoder` rebuilds a matched arm, so `sealed_eval.py` scores it unchanged |
| `tests/test_matched_encoder.py` | 21 tests |

Full suite after the change: **871 passed, 1 skipped**.

```bash
python -m training.support_classifier.train --out <run-dir> --encoder-arch limubert \
  --classifier residual --steps 40000 --seed 20260901 --data-seed 20260901
```

Everything downstream of `pooled` is byte-identical to the HALO arm: same corpus, same sampler, same
episodes, same residual classifier, same readouts, same sealed evaluation. A row difference is
therefore attributable to the encoder.

## 2. Input contracts (each model's own, deliberately not matched)

| arm | rate | clip | channels | long windows | device fusion | trunk params |
|---|---:|---:|---|---|---|---:|
| HALO | native | 0.5/1/2/4 s patches | acc + gyro | native | learned attention pool | 0.789 M |
| LiMU-BERT | 20 Hz | 20 samples (1 s) | acc + gyro | non-overlapping 1 s clips, mean-pooled | mean over devices | 0.107 M |
| harnet5 | 30 Hz | 150 samples (5 s) | acc only | centre crop | mean over devices | 4.392 M |
| UniMTS | 20 Hz | 200 samples (10 s) | acc only | start crop | **native**: every device placed at its own SMPL joint in one graph | 5.344 M |

UniMTS is the exception to mean pooling. Its whole design is that different placements are
different graph nodes, so pooling per device afterwards would replace its own contribution with
ours. Each device is placed at the joint its **sensor description** names (UniMTS's own placement
tables, matched against our sensor text rather than the stream key, which is what makes composite
recordings work), and the graph fuses them. This is also the setting most favourable to the
baseline, per fairness rule §5.4.

Capacity is reported, never matched (plan §5.6). The 1 s clip rule for LiMU-BERT is its released
positional-embedding contract and is the same rule its evaluation adapter uses.

**Device fusion is the parameter-free unweighted mean over devices present.** HALO's learned
recording pool is HALO's own component; handing it to a baseline would confound the encoder
comparison with HALO's fusion design. Note this is a *handicap disclosure*, not a handicap: it is
the same rule the released-checkpoint rows use.

## 3. Bugs the build caught, all fixed with regression tests

1. **`_load_harnet` returns a frozen, eval-mode deployment model.** Training it without
   `requires_grad_(True)` would have trained the projection alone and reported an architecture
   failure. Test: `test_trunk_is_trainable`.
2. **Random init by tensor rank zeroed BatchNorm scale.** A zero gamma outputs zeros and kills the
   gradient; the arm trains to nothing and looks like the architecture's fault. Init is now per
   module type. Test: `test_random_init_does_not_zero_normalisation_scale`.
3. **Composite recordings carry more than six channels.** The first implementation scanned only the
   canonical six, silently feeding device 0's signal for every device — in exactly the multi-device
   cells HALO is strongest at. Channel selection is now per device via the sensor map. Tests:
   `test_composite_recording_reads_every_device_not_just_the_first`, `test_device_pooling_is_an_unweighted_mean`.
4. **Frontend calibration ran for encoders with no filterbank**, burning forward passes for nothing.
   Now skipped with a printed reason.

A fifth issue was my own test, not the code: summing a LayerNorm output has exactly zero gradient,
so the first trainability check reported a false failure. Tests now use a randomly weighted loss.

## 4. Measured throughput (this machine, default recipe, 150 steps)

| arm | steps/s | 40k steps | vs HALO |
|---|---:|---:|---:|
| HALO (reference) | 19.4 | **34 min** | 1.00x |
| harnet5 | 13.4 | **50 min** | 1.45x |
| LiMU-BERT | 10.5 | **64 min** | 1.85x |
| UniMTS | 1.5 | **7.4 h** | 13x |
| NormWear | — | **~183 h** (not run) | ~320x |

LiMU-BERT is the *smallest* model and the *slowest* arm. The cost is not its 0.107 M parameters, it
is that an 8 s window becomes eight separate 1 s transformer calls. That is its published contract,
so the cost is real and must not be optimised away by feeding it longer clips.

**LiMU-BERT plus harnet5 is about 2 GPU-hours; adding UniMTS makes it about 9.5.** Reproducing
other people's models is not the expense; deciding what to run is.

**UniMTS needs gradient checkpointing to fit at all.** A 22-joint by 200-frame graph at the full
episode batch exceeds 24 GB of activations and the first attempt died with a CUDA out-of-memory.
Checkpointing recomputes each 32-window chunk in the backward pass instead of storing it:
numerically identical, roughly 30% more compute, and it keeps the **recipe** intact. Shrinking
episodes-per-step would have fitted too, but it would have changed the objective and voided the
comparison, so it was rejected.

**NormWear is declined on measured cost, not opinion.** Its sensor backbone is 136 M parameters
(not the 1.29 B often quoted, which includes the frozen TinyLlama text tower) and emits 560 patch
tokens per channel. Measured forward-only throughput is 46.6 windows/s, so one training step of
~256 windows with backward is about 16.5 s, and 40k steps is **~183 GPU-hours**. That is 320x the
HALO arm for one row. Decline it and report this number.

## 5. Hyperparameters: what to hold and what to allow

The purpose of M2 is to isolate the encoder, so **every non-encoder hyperparameter is copied from
the promoted HALO run and frozen**:

| knob | value | why frozen |
|---|---|---|
| steps | 40,000 | same budget as the promoted run |
| warmup | 500 | same |
| lr | 3e-4 | same |
| episodes/step | 4 | same |
| support size | 32 | same |
| queries per support set | 4 | same |
| enrollment k | 1, 2, 4, 8 | same curriculum |
| p_gt_present | 0.5 | same open-set construction |
| p_mask_candidate / p_mask_gt | 0.25 / 0.10 | same |
| same-subject prob. | 0.5 | same |
| multi-device prob. / max devices | 0.5 / 4 | same |
| seeds | 20260901 | same |

**The one hyperparameter that may legitimately differ is the encoder learning rate.** `train.py`
already exposes `--encoder-lr-scale`. A 4.4 M-parameter ResNet and a 0.1 M transformer do not have
the same optimal step size, and the fairness contract says to choose the setting most favourable to
the baseline. So:

* Run each arm first at `--encoder-lr-scale 1.0` (identical to HALO).
* Then run **one** additional arm per baseline at the more favourable of `0.3` or `3.0`, chosen from
  the first 2,000 steps' training loss, not from any sealed result.
* Report the better of the two and disclose that two were tried. Two extra hours total.

This is the only place where tuning is permitted, it is disclosed, and it can only help the
baseline. Everything else is frozen a priori.

## 6. Convergence evidence, decided before the runs

A baseline that did not converge is a **failed arm**, reported as such, not a low number (plan
§5.7). Criteria, fixed now:

1. Training loss at step 40k is below its value at step 20k, or within noise of it.
2. The validation dataset-macro F1 curve is flat or rising over the last 10k steps.
3. The encoder's effective rank at 40k is above 20. A rank in the single digits means collapse, and
   the arm is reported as collapsed rather than as an architecture verdict.
4. If any criterion fails, re-run once at the alternative encoder learning rate before reporting.

Publish the loss curve and the selected step for every arm.

## 7. Recommended run order

| # | arm | flags | wall clock |
|---|---|---|---:|
| 1 | LiMU-BERT M2 | `--encoder-arch limubert` | 64 min |
| 2 | harnet5 M2 | `--encoder-arch harnet` | 50 min |
| 3 | sealed evaluation of both | `--models halo --halo-checkpoint <run>` | ~36 min |
| 4 | lr-scale retry for whichever arm fails §6 | `--encoder-lr-scale` | up to 64 min |
| 5 | UniMTS M2 (overnight) | `--encoder-arch unimts` | 7.4 h |
| 6 | sealed evaluation of UniMTS | as above | ~18 min |

**Under four hours** for a two-architecture M2 block against the promoted HALO run; **one extra
overnight** adds the graph encoder, which is the most architecturally distant of the three and
therefore the most informative if the first two are inconclusive.
The existing sealed comparison already supplies every baseline's released-checkpoint row, so no
baseline needs re-evaluating.

## 8. Not built

* **M1** (each baseline's own pretraining objective on our corpus) and the corpus export it needs.
  LiMU-BERT M1 is the cheap one and is portable: its architecture is already vendored in this repo,
  so its masked-reconstruction objective can be implemented here without the external checkout.
  UniMTS M1 needs the 22-joint placement export described in plan §4.3.
* **NormWear M2.** Declined on the measured 183 GPU-hour cost above. Its readout defect
  (`NORMWEAR_READOUT_FINDING_20260915.md`) is a separate and much cheaper fix that should still
  happen, because it affects its released-checkpoint rows everywhere.
* **Mantis.** Not in the registry; `MANTIS_BASELINE_PLAN.md` specifies the adapter. M2 only, and
  only if the frontend question is still open after the three arms above.
