# 2026-09-14 — Correction: multi-device fusion is a learned attention pool, not a hierarchical mean

**Corrects:** [2026-09-14-evaluation-rebuild.md](2026-09-14-evaluation-rebuild.md) (the paragraph
after the multi-device table) and [2026-09-14-design-narrative.md](2026-09-14-design-narrative.md)
(§8, the "Multi-device from a plain hierarchical mean beat a purpose-built graph" bullet).
Both entries remain as written, per this folder's immutability rule. This entry supersedes their
description of the mechanism. **The numbers in both entries are unaffected and stand.**

## What the two entries claimed

That HALO's multi-device gain at k=1 (+11.1 RealWorld, +8.9 Shoaib, against UniMTS's native
SMPL-skeleton fusion at +3.7 / +4.1) came from a *parameter-free hierarchical mean* — accelerometer
and gyroscope averaged within a device, then devices averaged equally — and that the headline was
therefore "a mean beat a purpose-built graph."

## What the code actually does

`model/tokenizer/encoder.py` contains **two** recording-pooling paths:

1. `hierarchical_device_pool` (`encoder.py:50`) — the parameter-free mean described above.
2. `RecordingAttentionPool` (`encoder.py:78`) — one learned query attending over every
   `patch x sensor` token, flattened, with a padding mask.

At `encoder.py:817` the learned pool **unconditionally overwrites** the hierarchical result when it
is enabled, with a comment stating why: pre-averaging devices would make accelerometer and
gyroscope indistinguishable to a trained checkpoint. The hierarchical tensor is still returned as
`per_device`, but its only consumer is a device *count* used for telemetry
(`training/support_classifier/train.py:332`). It is never a feature.

`learnable_recording_pool=True` is set unconditionally for the support-classifier path
(`training/support_classifier/encoding.py`, `train.py:1442`). Verified directly in the saved
checkpoints — all three trained arms carry 15 `recording_pool.*` tensors and `config.
learnable_recording_pool == True`:

| checkpoint | pool tensors | flag |
|---|---:|---|
| `halo_fixed_mr_neighbors_8s_4res_e2e_20260913_continue95k` | 15 | True |
| `halo_fixed_mr_neighbors95k_frozen_residual_40k_20260914` | 15 | True |
| `halo_fixed_mr_neighbors95k_frozen_residual_regimesplit_40k_20260914` | 15 | True |
| `halo_fixed_mr_residual_v3_8s_4res_40k_20260914` (promoted) | 15 | True |

So **every row of the 2026-09-14 promoted table used learned attention pooling.** The
parameter-free mean produced none of those numbers. It is live only for checkpoints with no learned
pool — random-init frozen and JEPA-frozen arms, since Phase-A JEPA trains patch representations and
never builds a recording pool.

This was already documented correctly in
[EVAL_EXPANSION_PLAN_20260913.md §D5b](../design/EVAL_EXPANSION_PLAN_20260913.md), which carries an
explicit "Correction to the first draft" making exactly this point. The two journal entries above
were written later and regressed to the superseded wording. The design decision recorded as
"option 3 — hierarchical parameter-free pooling" applies **only to the mean path**, not to the
trained arms.

## What HALO's multi-device fusion actually is

- **Token granularity.** One token per `(patch, sensor)`, where a *sensor* is one 3-axis modality on
  one device. Accelerometer and gyroscope on the same wrist are two tokens and are never
  pre-averaged. The x/y/z axes are folded inside the token by `SensorFold` over the fixed xyz
  channel order, which is why no axis positional encoding is needed. A three-device recording
  carrying both modalities is six sensor tokens per patch.
- **Identity is text, and it is factored.** `role_texts` carries the **axis only**; `sensor_texts`
  carries **device + modality + placement** per present modality, with the gravity convention riding
  on the accelerometer text alone. Placement/device/modality appear in exactly one of the two, so
  no configuration fact is injected twice when they are summed
  (`training/tokenizer/pretrain_data.py:330`).
- **One pooling stage, not two.** A single learned query attends over all `patch x sensor` tokens
  across devices, modalities, patches and resolutions simultaneously. There is no separate
  device-fusion stage.
- **Cost.** The pool is **132,864 parameters of the encoder's 789,508** (0.133M of 0.789M), measured
  from the promoted checkpoint.
- **Fully differentiable end to end.** The filterbank is fixed but differentiable; the chain through
  sensor folding, text conditioning, the transformer, the attention pool, centring and the
  differentiable-neighbours softmax has no gradient cut. Every `detach()` in `model/support/` and
  `training/support_classifier/train.py` is telemetry or the `corpus_mean` buffer. Gradient reaches
  support embeddings as well as the query, since both pass through the same trunk.

## Why the correction matters for the claim

The measured comparison is unchanged and is still favourable: HALO's fusion gains far more at k=1
than UniMTS's purpose-built skeleton fusion. But the *argument* changes, in both directions.

- **Weaker as stated.** "A parameter-free mean beat a purpose-built graph" was the stronger and more
  quotable claim, and it is not what happened. We are comparing a learned pool against a learned
  structured fusion, which is a fairer fight and a less surprising result. Do not put the old
  sentence on a slide.
- **Stronger in a different way.** What actually beat the skeleton graph is a **placement-agnostic**
  pool with no per-device or per-placement parameters, where device identity enters only as text.
  That is directly on-thesis: it is the same "one language interface for unseen acquisition
  configurations" mechanism as the zero-shot label path, and it is why an unseen placement is
  expressible at all. UniMTS's fusion, by contrast, is tied to a fixed SMPL skeleton.
- **An ablation is now owed.** Because the two paths both exist and only one is used, the honest
  version of this claim needs the learned pool measured *against* the hierarchical mean on the same
  checkpoint. That is a cheap evaluation-only run and it is currently unmeasured. Until it is run,
  we cannot say how much of the +11.1 / +8.9 the learned pool is responsible for versus simply
  having more devices present.

## Also corrected

`docs/results/RESULTS.md` carried the same wording in its multi-device paragraph and has been
corrected in place, since it is a promoted-results record rather than a journal entry.
`docs/design/SUPPORT_CLASSIFIER_DESIGN_20260914.md` §"Not targets" described the same gain as
coming from a hierarchical mean and has been corrected in place.

## Follow-up

Add to the open list: **measure the learned recording pool against the parameter-free hierarchical
mean on the promoted checkpoint, on the multi-device composite cells.** Evaluation-only, no
training. This is the ablation that decides which sentence we are entitled to write.
