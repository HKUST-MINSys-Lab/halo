# 2026-09-14 — The argument: how the system got here, and what the evidence says

**Status:** narrative synthesis. Every number below is drawn from a dated entry or a sealed
artifact; this entry adds no new measurement. It exists so the design decisions of 2026-09-12 to
09-14 can be explained in order, with the evidence attached to each.

**One-line version.** We removed two learned components that measurement showed were not earning
their cost (label-free pretraining, learnable kernels), spent the savings on engineered physics and
a corrected evaluation, and added one learned component targeted at the two regimes a
parameter-free readout provably cannot enter. Zero-shot went from our weakest cell to our largest
lead; the enrolled regime is currently a partial regression with a diagnosed cause.

---

## 1. The question

Can one model serve unseen activity labels *and* unseen acquisition configurations, adapting from a
handful of labelled examples? Two axes: label vocabulary and device/placement/rate. The system is
an encoder plus a support-conditioned comparison, evaluated on six **sealed** datasets never used
for training or selection, with execution-disjoint enrollment.

## 2. Why we dropped Future-JEPA

**It works, and it is redundant.** The four-arm ladder
([2026-09-13-jepa-value-measured.md](2026-09-13-jepa-value-measured.md)) is unambiguous:

| comparison | Δ macro F1 | streams improved |
|---|---:|---|
| JEPA frozen − random frozen (differentiable neighbours) | **+9.24** | **6/6** |
| JEPA frozen − random frozen (ridge) | **+12.80** | **6/6** |
| JEPA adapted − random adapted | +0.67 | 4/6, sign flips |
| end-to-end 35k − JEPA-adapted 5k | +0.68 | 4/6 |

JEPA buys 9–13 points of representation quality for a **frozen** encoder — real, large, consistent.
But 5,000 steps of ordinary supervised adaptation reproduce it, and the JEPA-vs-random gap after
adaptation (+0.67) sits below the ~1.2-point noise floor with the sign flipping across streams —
the same signature as the config-conditioning inertness finding in commit `9b7d75d`.

**The honest conclusion:** JEPA is a *substitute* for early supervised training, not a complement.
Since we always have some labels for few-shot adaptation, we were paying a full pretraining run for
something adaptation gives free. Also worth naming: it did what it was asked. The predictor beat a
persistence baseline (0.28 vs 0.58 loss) and did not collapse (effective rank 12 → 101). It just
learned mostly *window identity* — with the context state zeroed, metadata alone picked the right
target 61% of the time, and two different target residuals from one window already agreed at 0.66
cosine. Only ~15% of the available headroom was genuine dynamics.

A deeper reason it could not have helped much
([2026-09-13-why-a-simple-encoder.md](2026-09-13-why-a-simple-encoder.md)): with a **frozen**
engineered frontend, predictive pretraining can only reshape how patches are combined. It cannot
enrich what each patch measures. The ceiling on pretraining is set by the same decision that makes
the encoder robust — a trade we took deliberately.

## 3. Why we dropped the continuous-kernel frontend

Learnable Gabor kernels were meant to give frequency *and* time-frequency resolution. Three
measurements closed it
([halo-continuous-kernel-sweep](../design/CONTINUOUS_KERNEL_FRONTEND.md),
[2026-09-12-jepa-and-encoder-findings.md](2026-09-12-jepa-and-encoder-findings.md)):

1. **Gradients reach every parameter** (zero-fraction 0.000) but are **sign-inconsistent** once the
   trunk is trained (batch cosine 0.08–0.13). The kernels perform an LR-bounded random walk — an
   optimisation failure, and a named published phenomenon ("learnable frontends that do not learn").
2. **Parameters were never the constraint** — 806 kernel parameters, and the multispan path has no
   CNN after the kernels at all.
3. **There is no headroom to go deeper.** wav2vec/HuBERT stack seven conv layers because they start
   at 16 kHz. At 20–100 Hz there is nothing to convolve down. One continuous-kernel layer largely
   approximates a fixed filterbank, and without temporal resolution to spare there is no second
   layer worth having.

Against that, a fixed constant-Q bank has the invariances **by construction** and costs no
parameters. The decision was to focus the fixed bank rather than keep fighting the optimiser
([2026-09-12-fixed-filterbank-decision.md](2026-09-12-fixed-filterbank-decision.md)).

## 4. Why we improved the fixed filterbank the way we did

Having chosen it, the question became what it *cannot* express. Two answers.

**Longer patches.** The bank ran 0.5/1/1.5 s. Time–frequency uncertainty says a patch of length T
cannot resolve below ~1/T, so slow structure was unreachable. Adding **4 s** patches is nearly free:
decimating to a 40 Hz analysis rate keeps an 8 s patch at 320 samples, so `DFT_SIZE` stays 512 and
the ladder costs ~0.13× the previous rDFT work
([2026-09-12-frontend-efficiency-and-jepa-window.md](2026-09-12-frontend-efficiency-and-jepa-window.md)).

**Phase, in the only form that survives heterogeneity.** A magnitude spectrum is invariant to time
shift *and* time reversal, so within a patch it cannot distinguish a straight-line shake from a
circular stir, or a motion from its reverse. Raw inter-axis phase would encode sensor mounting and
break the heterogeneity claim. The gravity-referenced, rotation-invariant forms do not: `vert`,
`circ`, `spin` from the 3×3 cross-spectral matrix
([2026-09-13-filterbank-polarization-features.md](2026-09-13-filterbank-polarization-features.md)).
This is standard three-component polarization analysis imported from geophysics (Samson & Olson
1980; Gonella 1972; Mooers 1973), with direct wearables precedent for the gravity reference (Mizell,
ISWC 2003) and for cross-spectral IMU features (Kobayashi et al., ICASSP 2011).

**The principle underneath both** — and the reason the encoder stays small — is that our job is to
*measure what the body is doing*, not to learn what activities look like. The learnable surface is
deliberately small: 0.789M parameters against MOMENT's 385M.

## 5. How it compares to the previous version

Two comparisons, with different strengths.

**Clean (encoder identical, classifier only):** the 09-13 and 09-14 runs share frontend, resolutions
`0.5/1/2/4`, polarization on, 8 s windows. Only the classifier and `p_gt_present` differ. This
isolates the classifier — §6.

**Bundled (the frontend arc):** the JEPA-era fixed multiresolution encoder scored k=0 **24.2**,
k=1 55.7, k=8 68.2, k=128 75.7 at 6 s
([2026-09-13-retired-jepa-promoted-results.md](2026-09-13-retired-jepa-promoted-results.md)). The
current encoder scores k=0 **38.3** (bridge) / **49.3** (classifier), k=1 60.5, k=8 71.6, k=128 77.5
at 8 s. **This is not a controlled comparison** — it bundles polarization, the 4 s patch, the
resolution change, 6 s→8 s, the protocol rebuild and dropping JEPA, and the 6 s and 8 s protocols
are not comparable at all ([2026-09-14-evaluation-rebuild.md](2026-09-14-evaluation-rebuild.md)).
Cite it as *where the system was and where it is*, never as a frontend ablation. **The controlled
polarization and 4 s ablations have not been run and remain outstanding.**

## 6. The learned classifier

The design followed measurement, not architecture taste. Two deficiencies were established
([2026-09-14-baseline-failure-analysis.md](2026-09-14-baseline-failure-analysis.md) §3): at k=1 every
parameter-free readout is *mathematically identical* and on wrong predictions the correct support is
in the top-5 **92.5%** of the time — so only a second evidence source can help. At k≥8, **15.6%** of
wrong predictions had the correct label's best support ranked **first** and were outvoted — a
*voting* failure a per-support reweighting targets directly.

So: differentiable neighbours as the base score, on centred vectors, plus a learned scalar residual
and a label-text term weighted by how many supports a candidate has, **initialised so the untrained
classifier equals the control bit-for-bit**. Trained with per-candidate support masking so some
candidates must be scored from their name alone.

**Results** ([2026-09-14-residual-classifier-first-result.md](2026-09-14-residual-classifier-first-result.md)),
8 s, dataset-balanced macro F1:

| | k=0 | k=1 | k=8 | k=32 | k=128 |
|---|---:|---:|---:|---:|---:|
| **HALO classifier** | **49.3** | **60.5** | 68.4 | 69.9 | 71.3 |
| HALO centred neighbours | – | 58.3 | 71.6 | 75.4 | 77.5 |
| HALO 1-NN | – | 58.6 | 69.9 | 73.7 | 77.1 |
| UniMTS | 30.7 | 52.4 | 66.0 | 70.6 | 75.1 |
| HARNet | 35.6 | 44.4 | 56.4 | 62.0 | 66.3 |

![k-curve](../results/assets/k_curve_residual_v3_20260914.png)

## 7. What went well

- **Zero-shot, decisively.** 38.3 → **49.3**; +13.7 over HARNet's bridge, which had led that
  condition, and +18.6 over UniMTS's released native text head. Our weakest cell became our largest
  lead.
- **k=1 improved where no parameter-free readout can.** 60.5 vs 58.6, above the ~1-point
  enrollment-draw noise.
- **Centring: +1.7 macro F1 at k=8–32 for zero parameters**, nearly closing the gap to ridge
  (77.5 vs 78.5 at k=128) without ridge's per-query solve. It should be our reported floor.
- **Multi-device from a plain hierarchical mean beat a purpose-built graph** — +11.1/+8.9 at k=1
  against UniMTS's native SMPL fusion at +3.7/+4.1. Zero-shot gains most: RealWorld composite 63.1
  against 44.3 single.
- **The curriculum change cost the encoder nothing.** Under identical 1-NN the 40k residual run
  matches the 95k neighbours run (58.6 vs 59.0 at k=1; 77.1 vs 76.5 at k=128) — zero-shot bought
  without paying in few-shot representation quality, in fewer steps.
- **A real defect found by the fairness probe:** HARNet had been centre-cropped from 6 s to 5 s,
  discarding evidence for nothing.

## 8. What did not go well

- **The learned residual regresses above k=2**: +2.1 at k=1, then −1.9, −3.1, −3.9, −5.5, −6.3, −6.2.
  Cause identified and owned: the λ bucket edges were written `(0,1,2,4,8)`, so **every k from 8 to
  128 shares λ=0.79** — text evidence still injected at weight 0.79 when the vote is already
  reliable — compounded by training only to k=8, leaving k=16–128 pure extrapolation.
- **`P_text` is fed the wrong tensor.** Its closed-form initialisation was fitted on raw pooled
  features but applied to the attention-refined query; telemetry shows the ground-truth label
  starting **below** the best distractor (−1.55).
- **The k=0 number is not yet the open-set claim.** Three of six sealed datasets have every label
  verbatim in the training vocabulary, so 49.3 is largely *known labels without supports*. The
  seen/unseen stratification is outstanding and free to compute.
- **RealWorld placement spread is 19.3 — the largest of any model** (forearm 57.9 / thigh 65.7 /
  waist 77.2). On the dataset that most directly tests placement heterogeneity, our claim currently
  reads worse than UniMTS's 14.9.
- **InclusiveHAR fails for everyone** (23–38 at k=8) with a flat k-curve, so it is an encoder/data
  problem no readout will fix.
- **The training curriculum does not yet support the motivation.** It never produces a
  cross-configuration episode (`mode=compatible` always) nor an unseen-label episode, and trains
  only to k=8. Both heterogeneity axes are currently hoped-for rather than trained-for.

## 9. Limitations to state plainly

Deployed-system comparisons, not parameter- or corpus-matched ablations. The controlled polarization
and 4 s ablations are unrun. `k=128` at 16 s is MotionSense-only. RealWorld is accelerometer-only in
every grid. The classifier adds 1.41M parameters to a 0.789M encoder. All architecture choices must
be validated on training-subject holdout — the neighbour diagnostics that motivated the design were
computed on sealed embeddings and are hypothesis-generating only.

## 10. What follows

1. Fix λ (smooth in log k), train to k=32, correct the `P_text` input, gate or drop `r_candidate`.
2. Stratify the k=0 result by seen/unseen label — free, and it is the real open-set test.
3. Run the controlled polarization and 4 s ablations.
4. Close the curriculum gaps: cross-placement episodes and label-group holdout, each with a matching
   evaluation cell, which is what would make the training design a contribution rather than
   competent practice.
