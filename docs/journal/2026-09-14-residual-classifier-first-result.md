# 2026-09-14 — The learned classifier's first sealed result: zero-shot solved, high-k regressed

**Status:** result. Sealed, manifest-matched. First evaluation of the identity-initialised residual
support classifier designed in `docs/design/SUPPORT_CLASSIFIER_DESIGN_20260914.md`.

**Run:** `training/support_classifier/outputs/halo_fixed_mr_residual_v3_8s_4res_40k_20260914`
— fixed multiresolution filterbank, `0.5/1/2/4 s` patches, 8 s training windows, 40,000 steps,
`--classifier residual --centring support_mean --p-gt-present 0.5 --p-mask-candidate 0.25
--p-mask-gt 0.10 --enrollment-k 1 2 4 8`.
**Evaluation:** `evaluations/sealed_comparison_residual_v3_8s_4res_40k_20260914/`.
Baseline rows are bit-identical to the 2026-09-13 evaluation, confirming the manifests and cached
features are matched across the two comparisons.

---

## 1. Headline — 8-second windows, single-device, dataset-balanced macro F1

| model / readout | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **HALO classifier** | **49.3** | **60.5** | 64.0 | 66.2 | 68.4 | 69.6 | 69.9 | 70.0 | 71.3 |
| HALO classifier, residual off (= centred DN) | – | 58.3 | 63.8 | 68.1 | 71.6 | 73.5 | 75.4 | 76.3 | 77.5 |
| HALO ridge | – | 58.4 | 64.1 | 68.8 | 71.5 | 74.2 | 75.7 | 76.7 | **78.5** |
| HALO 1-NN | – | 58.6 | 63.3 | 67.2 | 69.9 | 71.8 | 73.7 | 75.2 | 77.1 |
| HALO prototype | – | 58.6 | 64.2 | 68.6 | 70.7 | 72.6 | 73.4 | 73.9 | 75.0 |
| HALO training-bank 1-NN + ConSE | 38.3 | – | – | – | – | – | – | – | – |
| HARNet training-bank 1-NN + ConSE | 35.6 | – | – | – | – | – | – | – | – |
| UniMTS native zero-support | 30.7 | – | – | – | – | – | – | – | – |
| UniMTS 1-NN | – | 52.4 | 58.2 | 63.1 | 66.0 | 68.6 | 70.6 | 73.1 | 75.1 |
| LiMU-BERT-X 1-NN | – | 49.2 | 56.1 | 61.5 | 65.9 | 69.8 | 71.4 | 72.9 | 73.5 |
| HARNet 1-NN | – | 44.4 | 48.9 | 53.1 | 56.4 | 59.2 | 62.0 | 64.4 | 66.3 |
| NormWear 1-NN | – | 21.4 | 23.5 | 26.3 | 29.4 | 32.1 | 34.9 | 38.0 | 40.2 |

The same shape holds at 4 s and 16 s. Classifier parameters are 2.203M against the 0.789M encoder,
so the learned head roughly triples the trainable model.

## 2. What worked

**Zero-shot, decisively.** 38.3 → **49.3**, a +11.0 gain over our own bridge and **+13.7 over
HARNet's** (35.6), which had led this condition since 2026-09-12. Against UniMTS's released native
text-aligned head (30.7) the margin is +18.6. This was the weakest cell in the system and the
classifier's primary design target; it is now the largest relative lead we hold.

Multi-device zero-shot is better still: 63.1 on the three-device RealWorld composite and 72.4 on
the four-device Shoaib composite, against 49.3 single-device. More simultaneous devices sharply
improve recognition from label text alone.

**k=1, as designed.** 60.5 against 58.6 for 1-NN and 58.4 for ridge: **+1.9**, above the ~1-point
enrollment-draw noise floor. This is the regime where every parameter-free readout is
mathematically identical and only a second source of evidence can help. The text term supplied it.

**Centring, for free.** Residual-off *is* centred differentiable neighbours, so the gap to plain
1-NN is the parameter-free preprocessing gain: **+1.7 macro F1 at k=8–32**, tapering to ≈0 at k=1
and k=128. At k=128 centred DN reaches 77.5 against ridge's 78.5 while remaining a single
vectorised operation rather than a per-query linear solve. SimpleShot's centre-then-normalise,
confirmed on our geometry.

**The curriculum change cost the encoder nothing.** Under the identical 1-NN readout the residual
run's encoder matches the 95k-step neighbours run (8 s: 58.6 vs 59.0 at k=1; 73.7 vs 73.0 at k=32;
77.1 vs 76.5 at k=128) in 40k steps rather than 95k, despite `p_gt_present=0.5` halving the
enrolled-episode gradient and masking removing further supports. Training for zero-shot was not
paid for out of few-shot representation quality.

## 3. What failed — the learned residual above k=2

Attribution at 8 s, single-device:

| k | 1-NN | centred DN | Δ centring | classifier | Δ learned |
|---:|---:|---:|---:|---:|---:|
| 1 | 58.6 | 58.3 | −0.3 | 60.5 | **+2.1** |
| 2 | 63.3 | 63.8 | +0.5 | 64.0 | +0.2 |
| 4 | 67.2 | 68.1 | +0.9 | 66.2 | −1.9 |
| 8 | 69.9 | 71.6 | +1.7 | 68.4 | −3.1 |
| 16 | 71.8 | 73.5 | +1.7 | 69.6 | −3.9 |
| 32 | 73.7 | 75.4 | +1.7 | 69.9 | −5.5 |
| 64 | 75.2 | 76.3 | +1.1 | 70.0 | −6.3 |
| 128 | 77.1 | 77.5 | +0.4 | 71.3 | −6.2 |

The learned part helps at k=1, is neutral at k=2, and then costs progressively more, reaching
−6.3 at k=64. **Acceptance bar 2 of the design (≥ ridge at k≥16, ≥ soft vote at every k) is failed
decisively.** The same pattern appears on both multi-device composites.

**Diagnosis — a design error in the λ schedule, not a training failure.** Final learned values:

| bucket | λ(0) | λ(1) | λ(2) | λ(4) | λ(8+) |
|---|---:|---:|---:|---:|---:|
| final | 1.93 | 1.13 | 1.01 | 0.92 | **0.79** |

λ does decay with k, exactly as the design hoped — but the bucket table ends at 8, so **every k from
8 to 128 shares λ(8+) = 0.79**. At k=128 the vote is highly reliable and the text term is still
being added at weight 0.79, injecting evidence the episode no longer needs. The design explicitly
anticipated this case ("whether λ(8+) → 0. If text evidence remains useful at high k, report it")
and wrote the bucket edges as `(0, 1, 2, 4, 8)`. That was my choice and it is wrong: the top bucket
is trained at k=8, where text still helps slightly, and then applied at k=128, where it hurts.

Two compounding factors. The training curriculum used `--enrollment-k 1 2 4 8`, so **k=16 through
128 received no gradient at all** — both λ and the per-support residual are pure extrapolation
there. And `mean_abs_r_candidate` grew to 0.47, larger than `mean_abs_r_support` (0.07–0.37): the
head leans on an unconditioned per-candidate bias, a second additive term with no mechanism to
fade as evidence accumulates.

Telemetry also shows `text_score_gt_minus_max_other` starting at **−1.55** — the closed-form
least-squares initialisation of `P_text` put the ground-truth label *below* the best distractor —
and climbing to ≈+1.4. Consistent with the open sweep finding that `P_text` is applied to the
attention-refined query while the fit was computed on raw pooled features, so the init describes an
input the model never presents.

## 4. Interpretation

The classifier does precisely what it was designed to do in the two regimes the parameter-free vote
cannot enter — no supports, and one support that might be wrong — and damages the regime where the
vote is already strong. The failure is confined, understood, and attributable to two named
decisions (bucket edges, training k range) rather than to the residual formulation.

Centred differentiable neighbours is the better parameter-free control on every count and should
replace 1-NN as HALO's reported floor.

## 5. What this changes

1. **Extend the λ schedule past 8** and train with `--enrollment-k 1 2 4 8 16 32`, so the high-k
   buckets receive gradient. Consider replacing the bucket table with a smooth function of log k so
   extrapolation beyond the trained range is principled rather than flat.
2. **Fix the `P_text` input** to the raw pooled query so the closed-form initialisation applies to
   the tensor it was fitted on (open sweep finding H1).
3. **Reconsider `r_candidate`**: gate it by k, or remove it.
4. **Promote centred DN** to the reported parameter-free readout.
5. The k=0 result stands on its own and does not depend on any of the above.

## 6. Caveats

- Sealed numbers; no sealed result selected the checkpoint. The fixes above must be validated on
  training-subject holdout before another sealed run.
- k=128 at 16 s is MotionSense-only and is not a six-dataset aggregate.
- Three of the six sealed datasets have every label verbatim in the training vocabulary, so the
  k=0 figure is mostly *known labels without supports*, not unseen labels. The seen/unseen
  stratification of this number is outstanding and is the honest test of the open-set claim.
- The classifier adds 1.41M parameters to a 0.789M encoder.

## 7. Links

- Design: `docs/design/SUPPORT_CLASSIFIER_DESIGN_20260914.md`
- Audit: `docs/journal/RESIDUAL_CLASSIFIER_AUDIT_PROFILE_20260914.md` (other agent)
- Neighbour diagnostics that motivated the design:
  `training/support_classifier/evaluations/neighbor_diagnostics_20260914/REPORT.md`
- Previous promoted comparison: `docs/journal/2026-09-13-jepa-value-measured.md`
