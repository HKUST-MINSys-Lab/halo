# 2026-09-13 — Adding polarization features to the fixed filterbank

**Status:** designed and specified, **not yet built**. This entry records the design, the
motivation, and the published precedent it rests on, so the change can be justified without
re-deriving it.

**Context:** [2026-09-12-fixed-filterbank-decision.md](2026-09-12-fixed-filterbank-decision.md)
committed the project to the fixed multiresolution filterbank as the primary frontend. The
immediate follow-up question was whether that frontend is *expressive enough*, given that it
keeps only per-channel band energies. This entry answers "no, and here is the specific thing it
is missing".

---

## 1. What the current frontend discards, and why it matters

`PhysicalFilterbankTokenizer` computes a real DFT per channel per patch and immediately forms
`power = X.real² + X.imag²`. Each DFT bin is a complex number `X[m] = A·e^{iφ}`; squaring and
summing keeps the amplitude `A` and destroys the phase `φ`.

Discarding per-channel phase was the right call in isolation. `φ` depends on where the patch
boundary happens to fall relative to the motion: shifting the window by `τ` maps every phase to
`φ + 2πfτ`, and our patch boundaries come from a grid laid over the recording, so a per-channel
phase feature is arbitrary.

But the magnitude spectrum has two invariances that are *not* harmless here:

1. **Shift invariance.** `|FFT(x(t−τ))| = |FFT(x(t))|` — where in the patch an event occurs is
   invisible.
2. **Time-reversal invariance.** `FFT(x(−t)) = conj(FFT(x(t)))`, so `|FFT(x(−t))| = |FFT(x(t))|`
   — a motion and its reverse have identical band energies.

Consequence: within a single patch the frontend cannot distinguish a motion from its reverse, nor
a straight-line oscillation from a circular one. This is not hypothetical for our roster — KU-HAR
carries `repeated_standing_and_sitting` and `repeated_standing_and_lying`, and several training
sources contain rotational gestures whose only distinguishing feature at a given frequency is the
*shape and sense* of the trajectory, not its energy.

The key observation is that while per-channel phase is arbitrary, the phase *difference between
axes of the same triad* is not: all three axes are sampled at the same instants, so a window shift
adds the same `2πfτ` to all three and cancels in the difference. That difference is what separates
linear from elliptical from circular motion, and clockwise from counter-clockwise.

## 2. What is being added

Reuse the complex spectrum `X` that `_spectral_power` already computes, before it is squared, and
pool it with the amplitude-domain band filters `G = sqrt(H)` (so `|Z|² = E` for a pure tone):

```
Z[k] = Σ_m G[k,m] · X[m]          # one complex number per band per channel
```

For each sensor triad, form the 3×3 Hermitian cross-spectral matrix `S[k] = Z[k] Z[k]ᴴ` with
`Z[k] = (Z_x, Z_y, Z_z)`, and the rotary vector `v[k] = Im(conj(Z) × Z) = 2(a × b)` where
`Z = a + ib`. Then emit three normalized, dimensionless, bounded scalars per band per triad:

| feature | formula | range | meaning |
|---|---|---|---|
| `vert` | `(ĝᵀ Re S ĝ) / tr S` | [0,1] | share of the band's energy along gravity vs horizontal |
| `circ` | `‖v‖ / tr S` | [0,1] | 0 = straight-line oscillation, 1 = circular |
| `spin` | `(v · ĝ) / tr S` | [−1,1] | signed rotation sense about the vertical |

plus a scalar `grav_ok` gate. `ĝ` is the existing per-patch DC feature of the accelerometer triad,
normalized. The bound `‖v‖ ≤ tr S` follows from `2|a||b| sin θ ≤ |a|² + |b|²` (AM–GM), with
equality exactly for circular motion — which is what makes `circ` a clean [0,1] shape descriptor.

`circ` is invariant to sensor rotation outright (both `‖v‖` and `tr S` are scalar invariants of
`S`). `vert` and `spin` are invariant because `v` and `ĝ` rotate together with the device frame —
gravity is measured in the same frame as the motion. This is the property that keeps the feature
compatible with our cross-configuration heterogeneity claim: raw inter-axis phase would encode
sensor mounting, these forms do not. `spin` flips sign under time reversal; `circ` and `vert` do
not.

Full implementation spec (config flag, calibration, attachment point after `SensorFold`, gating,
serialization, tests, acceptance criterion) was written up separately and handed to the
implementing agent on 2026-09-13.

## 3. Where this comes from — precedent

This is not a new invention. It is the standard toolkit for three-component vector time series,
imported from geophysics and oceanography into our frontend. The citations below are the ones to
use in the paper.

**Cross-spectral / spectral-matrix polarization analysis.**

* J. C. Samson and J. V. Olson, "Some comments on the descriptions of the polarization states of
  waves," *Geophysical Journal of the Royal Astronomical Society*, 61(1):115–129, 1980.
  DOI [10.1111/j.1365-246X.1980.tb04308.x](https://doi.org/10.1111/j.1365-246X.1980.tb04308.x)
  — the canonical reference for our construction. It defines the spectral matrix `S` for
  n-dimensional (in particular three-dimensional) waves, and builds degree-of-polarization
  measures that "are functions only of the scalar invariants of `S`," writing the polarization
  vector as `u = r₁ + i r₂` with `r₁`, `r₂` real and orthogonal, where "`r₁` and `r₂` locate the
  major and minor axes of the ellipse, and the ellipticity is given by the ratio of their
  magnitudes." Our `a`/`b` decomposition of `Z` and our `circ` ratio are that construction; using
  scalar invariants rather than diagonalizing `S` is exactly their recommendation, and is why our
  version costs three numbers per band rather than an eigendecomposition.

* J. Park, F. L. Vernon and C. R. Lindberg, "Frequency dependent polarization analysis of
  high-frequency seismograms," *Journal of Geophysical Research*, 92(B12):12664–12674, 1987.
  DOI [10.1029/JB092iB12p12664](https://doi.org/10.1029/JB092iB12p12664)
  — establishes polarization estimated *as a function of frequency* from three-component data,
  where the complex polarization vector's elements "specify the relative amplitudes and phases of
  motion measured along the recorded components within a chosen frequency band." That is our
  per-band `Z[k]`, and their band width is set by the taper's time–bandwidth product exactly as
  ours is set by the Gaussian band filter `H`.

**Rotary spectra (the signed rotation sense).**

* J. Gonella, "A rotary-component method for analysing meteorological and oceanographic vector
  time series," *Deep-Sea Research*, 19(12):833–846, 1972.
  DOI [10.1016/0011-7471(72)90002-2](https://doi.org/10.1016/0011-7471(72)90002-2)
* C. N. K. Mooers, "A technique for the cross spectrum analysis of pairs of complex-valued time
  series, with emphasis on properties of polarized components and rotational invariants,"
  *Deep-Sea Research*, 20(12):1129–1141, 1973.
  DOI [10.1016/0011-7471(73)90027-2](https://doi.org/10.1016/0011-7471(73)90027-2)
  — rotary-spectrum decomposition splits a vector time series into counter-rotating circular
  components at each frequency; the difference of their powers is the signed rotary quantity our
  `spin` measures, and Mooers' title names the "rotational invariants" our normalization produces.
  Cite these for `spin` specifically.

**Circularity of complex signals (the signal-processing framing).**

* B. Picinbono, "On circularity," *IEEE Transactions on Signal Processing*, 42(12):3473–3482, 1994.
  DOI [10.1109/78.340781](https://doi.org/10.1109/78.340781)
  — the standard signal-processing reference for circularity/propriety of complex-valued signals.
  Use this when framing the feature for a signal-processing audience rather than a geophysics one.

**Gravity as the reference direction, in our own domain.**

* D. Mizell, "Using gravity to estimate accelerometer orientation," *Proc. 7th IEEE International
  Symposium on Wearable Computers (ISWC)*, pp. 252–253, 2003.
  — establishes the exact move `vert` makes, in wearable sensing: estimate the gravity vector by
  averaging accelerometer samples over the window, then decompose motion into its vertical
  component and the magnitude of its horizontal component, "independently of how the three-axis
  accelerometer system is oriented." Our DC feature already computes this gravity estimate; `vert`
  is the per-band version of Mizell's decomposition. Twenty-three years old and directly on point —
  this is the cheapest possible justification for the `vert` half of the change.

* A. Yurtman and B. Barshan, "Activity recognition invariant to sensor orientation with wearable
  motion sensors," *Sensors*, 17(8):1838, 2017.
  DOI [10.3390/s17081838](https://doi.org/10.3390/s17081838)
  — recent HAR-side evidence that orientation invariance is worth engineering in explicitly:
  "the ordinary activity recognition system cannot handle incorrectly oriented sensors," while
  their transformations "achieve nearly the same activity recognition performance as the ordinary
  system for which the sensor units are not rotatable." Cite for the motivation, not the method
  (theirs is a time-domain preprocessing transform, ours is a frequency-domain feature).

## 4. Closest prior work — and how ours differs

* T. Kobayashi, K. Hasida and N. Otsu, "Rotation invariant feature extraction from 3-D
  acceleration signals," *Proc. IEEE ICASSP*, pp. 3684–3687, 2011.
  DOI [10.1109/ICASSP.2011.5947150](https://doi.org/10.1109/ICASSP.2011.5947150)

This is the nearest published thing to what we are adding, and it is in our domain. They FFT the
three-axis signal to `f(ω) ∈ ℂ³`, stack it as `F ∈ ℂ^{3×n}`, and take the correlation matrix
`R = F*F ∈ ℂ^{n×n}`. They prove rotation invariance (`R̂ = F*AᵀAF = F*F = R` for any rotation `A`)
and shift invariance (`|R̃_jk| = |R_jk|`), and argue the method "can extract more discriminative
features than the standard power spectrum-based methods." They report it beating power-spectrum
baselines on accelerometer gait identification. **This is the single best citation for the claim
that cross-spectral structure carries discriminative information the magnitude spectrum does
not.**

Two honest differences, both worth stating rather than glossing:

1. **Different contraction of the same object.** Their `R = F*F` is `n×n` and contracts the *axis*
   index — cross-frequency correlations summed over axes. Ours is `S[k] = Z[k]Z[k]ᴴ`, `3×3` per
   band, contracting the *frequency* index — cross-axis correlations at one frequency. Theirs is
   automatically rotation-invariant because `AᵀA = I` cancels; ours is not (`S → A S Aᵀ`), which is
   precisely why we take scalar invariants of `S` and introduce the gravity reference. Their
   feature is `O(n²)` numbers; ours is 3 per band.
2. **They take absolute values; we keep the sign.** `|R_jk|` discards the rotation sense, so their
   feature is time-reversal invariant in the same way the plain magnitude spectrum is. Our `spin`
   retains it, which is the part that addresses the KU-HAR reversed-transition problem. That
   retention is bought by referencing gravity, and is therefore gated off on gravity-removed
   sources.

**Follow-up (specified, default off):** a harmonic-asymmetry feature — the phase relation between
a band and its second harmonic, which is shift-invariant and sign-flipping under time reversal and
captures *waveform* asymmetry (sharp heel strike, slow recovery) rather than rotational asymmetry.
The gait-analysis precedent is the harmonic ratio of trunk accelerations: H. B. Menz, S. R. Lord
and R. C. Fitzpatrick, "Acceleration patterns of the head and pelvis when walking on level and
irregular surfaces," *Gait & Posture*, 18(1):35–46, 2003,
DOI [10.1016/S0966-6362(02)00159-5](https://doi.org/10.1016/S0966-6362(02)00159-5) (623 citations).
That literature uses harmonic *amplitudes*; the phase-coupling version is the natural extension
and is kept behind its own flag so it can be cut independently.

## 5. Cost and decision rule

The change adds 97 numbers per triad per patch and widens the filterbank projection input from 98
to 195 dimensions (+~25k parameters). One extra einsum, no extra FFT.

That is a >2× widening of the frontend input on a label-scarce problem, so it carries a real
overfitting cost. **This is a hypothesis, not a fix.** The acceptance criterion is the
subject-disjoint kNN probe on training sources only, with the flag on and off, against the
hand-crafted feature floor (BA 0.82 / 0.78 / 0.70 / 0.61 / 0.56 on hhar / kuhar / realdisp / dsads
/ wisdm). If it does not move that probe or the internal validation panel, it gets reverted rather
than kept on the grounds of being principled. No sealed evaluation is involved in that decision.

## 6. Caveats recorded at the time of writing

* `spin` catches *rotational* asymmetry, not all time asymmetry. A purely linear sit→stand has
  `v ≈ 0`; what separates it from stand→sit is the DC trajectory across consecutive patches, which
  the bank already carries. Do not claim `spin` resolves transition direction in general.
* `vert` and `spin` require gravity and are gated to ≈0 on gravity-removed sources (KU-HAR; see
  [[halo-corpus-data-quality]] — recgym is additionally min–max normalized, which corrupts the DC
  feature and therefore `ĝ`). `circ` needs no gravity and works everywhere.
* None of the numbers in §5 are measured yet; they are the design's own cost accounting.
* A `use_phase` flag existed in an earlier repo state (per memory `halo-phase-feature`) and was
  never transfer-tested before the 2026-09-11 consolidation removed it. That was raw cross-channel
  phase, which is *not* what is proposed here — it was mounting-dependent and would have
  undermined the heterogeneity claim. The gravity-referenced and rotation-invariant forms above
  are the replacement.

## 7. Links

* Prior entry: [2026-09-12-fixed-filterbank-decision.md](2026-09-12-fixed-filterbank-decision.md)
* Prior entry: [2026-09-12-spectral-frontend-literature.md](2026-09-12-spectral-frontend-literature.md)
* Target file: `model/tokenizer/filterbank.py`; attachment point `model/tokenizer/encoder.py`
  (after `SensorFold`)
