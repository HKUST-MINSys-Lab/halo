# 2026-09-13 — Mantis as a design foil, and a correction to today's ZARA reading

**Corrects:** [2026-09-13-why-a-simple-encoder.md](2026-09-13-why-a-simple-encoder.md) §5.
Journal entries are immutable, so this is the correction rather than an edit.

## 1. The correction

That entry said ZARA's retrieval-embedder ablation "credits a *learned* retrieval embedder with
+10.6 points" (DTW 71.0 → Mantis 81.6). The number is right; the framing overstates what it
supports.

The full row is **DTW 71.0 → Moment-small 79.4 → Moment-large 80.8 → Mantis 81.6**. All three
learned encoders cluster at 79–82 while the only non-learned entry is raw DTW distance on
multivariate IMU, which is a weak control. **There is no engineered-features-plus-retrieval row in
that ablation.** So it establishes *learned embedding beats raw elastic distance*, not *learned
beats engineered*. Cite it as the former.

The conclusion of §5 — learn the metric, not the label map — is unaffected; only the strength of
the evidence for it changes.

## 2. Mantis as the architectural foil

Worth recording because it is the cleanest available contrast to our frontend, and because it
makes our design choices legible as choices rather than defaults.

Mantis (Feofanov et al., [arXiv:2502.15637](https://arxiv.org/abs/2502.15637), 8M params, ViT
6×8, d=256, contrastive with a single RandomCropResize augmentation) does two things in its
forward pass that we deliberately refuse:

* **Interpolates every input to a fixed length of 512** — "inspired by fixed-resolution inputs in
  computer vision." This absorbs sampling-rate heterogeneity by resampling, but it means the model
  reasons in *cycles-per-window*, not Hz, and it cannot know that a 20 Hz recording upsampled to
  512 points has nothing real above 10 Hz. We anchor to physical Hz and carry an explicit
  **Nyquist observability mask derived from the source rate**. This is the heterogeneity claim, and
  Mantis has no answer to it.
* **Instance-standardises each channel** (subtract mean, divide by std over time). That removes the
  DC, which *is* the gravity vector — so orientation and posture are unavailable to it, and so is
  relative amplitude between channels. We keep signed DC deliberately. Telling detail: ZARA's
  hand-crafted feature list separately includes **tilt angle**, i.e. they had to put back what
  instance normalisation takes away.

It is also **univariate** — "pre-trained on univariate data and applied to multivariate settings by
treating channels independently" — against our explicit cross-channel sensor fold.

| | Mantis | ours |
|---|---|---|
| time axis | interpolate to 512 samples | physical seconds, constant-Q bands in Hz |
| rate heterogeneity | absorbed by resampling; no observability signal | Nyquist mask from source rate |
| DC / gravity | destroyed by instance norm | signed DC retained |
| tokenizer | learned 1D conv → 32 patches | fixed rDFT → 32 constant-Q bands |
| channels | independent, univariate | cross-channel sensor fold |
| resolutions | one | ladder 0.5/1/2 s, interleaved |
| conditioning | none | acquisition / placement text |
| params | 8M | 2.7–3.2M trunk |

## 3. What we are missing, honestly

1. **Calibration.** Mantis's headline claim is being the most calibrated classification foundation
   model. We have no calibration story at all, and we score across candidate labels and support
   sets at k=0…128 where calibration matters. Reliability diagrams and ECE per k would be a real
   contribution obtainable without new training.
2. **Transient / impulsive structure.** *Not* the derivative spectrum — differencing is
   multiplication by jω, so the derivative's band energy is ≈(2πf_k)² times ours, a per-band
   reweighting the projection can already learn. What we lack is peak-to-peak, crest factor, max
   jerk, zero-crossing rate: band energies are patch averages, so a heel strike and a smooth
   oscillation of equal energy are identical to us. Mantis captures some of this through its
   differenced stream *before* pooling; ZARA lists jerk and zero-crossings explicitly.
3. **Augmentation discipline.** Mantis uses exactly one augmentation, chosen to be
   label-preserving. Our stack was elaborate enough that commit `9b7d75d` found it undifferentiated
   and blamed it for conditioning inertness.
4. **Cross-domain pre-training diversity** (UCR, UEA, ECG, EMG, Epilepsy, FD, Gesture, HAR,
   SleepEEG) against our HAR-only corpus.

## 4. Two things this unblocks

**A cheap, strong, uncarried baseline.** Open weights, 8M params, frozen-encoder usage, and it is
the encoder ZARA credits. Implementation brief written at
`docs/design/MANTIS_BASELINE_PLAN.md` — frozen arm plus a 5,000-step fine-tuned arm matched to the
existing adapted arms, so the rows drop straight into the ladder in
[2026-09-13-jepa-value-measured.md](2026-09-13-jepa-value-measured.md). Not yet built; no sealed
run authorised.

**Verified baseline parameter counts**, which partly closes the action item left open on
2026-09-12 (the recalled "~60M next-best baseline" that could not be checked). From Mantis's own
comparison: **MOMENT 385M params / 1.13B pre-training samples**, GPT4TS 80M, Mantis 8M, NuTime 2M,
UniTS 1M. Against our 2.7–3.2M trunk, MOMENT is the ~100× figure and is the right anchor for a
parameters-vs-performance plot. These are counts reported by a third party, not measured by us
from materialised checkpoints — that check is still outstanding before they go in a paper.
