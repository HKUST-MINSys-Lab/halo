# Is a fixed spectral frontend principled for our regime? What the literature says

**Date:** 2026-09-12
**Status:** immutable entry — see `docs/journal/README.md`. Literature check requested against the
decision recorded in `2026-09-12-fixed-filterbank-decision.md`. Implementation plan:
`docs/design/FIXED_FRONTEND_PLAN_20260912.md`.

**Short answer: yes, and the principle is specific.** Body-worn acceleration is bandlimited to
~15 Hz with its energy concentrated in 0.3–5 Hz, so a physical filterbank spanning that range is
close to a sufficient statistic for the frequency content of the signal. That is *not* true of
16 kHz audio, which is why wav2vec-style learned frontends earn seven convolutional layers there
and do not obviously earn them here. The depth is better spent on temporal composition —
which, in our design, attention across patches already provides.

## The bandwidth argument, with sources

- Bouten et al., *IEEE Trans. Biomed. Eng.* 1997 (a triaxial accelerometer for assessing daily
  physical activity), reporting Antonsson & Mann: *"By measuring these 'worst case' accelerations
  at heel strike with a force platform Antonsson and Mann demonstrated that **99% of the
  acceleration power during walking with bare feet is concentrated below 15 Hz**. Higher
  frequencies are caused by the impact between foot and walking surface and do not directly result
  from voluntary muscular work."* The same paper: *"in walking at natural velocity the bulk of
  acceleration power in the upper body ranges from **0.8–5 Hz**"*, and citing Sun & Hill, *"Fast
  Fourier analysis of daily activities performed on a force platform revealed the major energy band
  to be between **0.3–3.5 Hz**."*
- A 2024–25 wearable locomotor-discrimination parameter study reaches the same place empirically:
  *"At **40 Hz** the mean criterion score of the discrimination **plateaued**, with any variation in
  results over 40 Hz relating to noise"*, and *"99% of amplitude information from the Fast Fourier
  Transform spectrum resides within signal frequencies below 15 Hz for walking and 18 Hz for
  running."*

Our bank is 32 log-spaced bands over 0.3–15 Hz — almost exactly the documented support of the
signal. This is also the direct justification for the 40 Hz analysis rate adopted in the
implementation plan: it is the rate at which the published discrimination curve flattens, and it
sits comfortably above the 33.3 Hz our own `f_max` and Nyquist margin require.

## Evidence that fixed/engineered features are competitive here

- **ROCKET / MiniROCKET** (Dempster et al., *DMKD* 2020; *MiniRocket*, KDD 2021): random
  convolutional kernels plus a linear classifier reach state of the art on the UCR archive.
  Breadth of fixed features, not learned filters, carries time-series classification.
- **Learnable filterbanks often do not learn.** Anderson, Kinnunen & Harte (ICASSP 2023):
  *"a recurring finding reported independently in learnable filterbank studies is that the learned
  filters do not differ substantially from their initialised values… we feel this is more likely an
  optimisation problem."* Follow-up work is titled *"EfficientLEAF: A Faster LEarnable Audio
  Frontend of Questionable Use."* Our own measurement — kernels moving ~1% of their range over a
  20k-step run — is this phenomenon.
- **Whisper** (fixed 80-channel log-mel) and **AST** (fixed 128-bin mel + linear patch embedding)
  are frontier systems with entirely non-learnable spectral frontends.
- Recent HAR work is moving *toward* spectral priors, not away: *SPECTRA: An Efficient
  Spectral-Informed Neural Network for Sensor-Based Activity Recognition* (2026), *Triple Spectral
  Fusion for Sensor-based HAR* (2026), and *Feature Anchors for Time-Series Sensor-Based HAR*
  (2026), which notes handcrafted time-series features *"capture meaningful motion statistics and
  remain competitive."*

## A caution that applies to us, recorded honestly

Leite et al. (Aalto, arXiv:2410.13605) ran >500 experiments comparing transformer and
non-transformer HAR architectures and found *"Transformer-based models show consistently lower
performance than ResBiLSTM and InnoHAR… transformers can only outperform their counterparts in
less than 3% of the experiments,"* attributing it to the data scarcity of the field: transformers
*"require extensive pre-training, strong data augmentations, availability of excessive amounts of
data… In HAR, these endeavors have not been explored given the scarcity of data."* Our system is
transformer-based, so this is a caution about our trunk, not only about CNNs. Their prescription —
restore inductive bias — is exactly what a physical filterbank does, but at the frontend rather
than in the trunk. Worth keeping in view if the fixed arm underperforms.

Two further counter-points, for balance: BenchHAR (arXiv 2605.08296) finds CNN encoders learn the
most generalisable sensor representations across its cross-dataset benchmark; and a 2025 classical-
vs-deep HAR benchmark (Hossain et al.) reports *"CNN models offer superior performance across all
datasets… Classical models like Random Forest do well on smaller datasets but face challenges with
larger, more complex data."* Both compare end-to-end learned *pipelines* against hand-designed
*classifiers*, which is a different question from whether the **first layer** should be learned —
but they are the strongest available arguments for putting convolutional depth somewhere in the
stack, and the decision to defer that is a deliberate bet, not a settled result.

## Where this leaves the claim

Defensible as stated: *the accelerometer signal is narrowband and low-frequency, so a physically
grounded filterbank spanning 0.3–15 Hz captures essentially all of its spectral content; we
therefore fix the frontend and spend learnable capacity on comparison and temporal context
instead.* The honest qualifier is that this is a claim about the **frequency** content only —
temporal composition within a window is a separate axis, and the literature does support depth
there. Our current bet is that multiresolution patches plus attention cover it; the ladder
extension to 0.5/1/2/4/8 s is what makes the low-frequency half of that claim true rather than
flagged-as-blurry.

**Related:** `2026-09-12-fixed-filterbank-decision.md`,
`2026-09-12-frontend-efficiency-and-jepa-window.md`,
`docs/design/FIXED_FRONTEND_PLAN_20260912.md`.
