# Evaluation readiness fixes - 2026-09-16

**Status:** implemented and smoke-tested. This note resolves the second evaluation/curriculum
debug sweep and defines the protocol boundary for subsequent scenario and sealed runs.

> **Later protocol update:** v5 (`deployment-scenarios-v5-20260918`) retires the redundant compound
> cold-start scenario, moves new-domain scoring to MM-Fit, adds MM-Fit cross-device/device-set
> cells, and keeps MobiAct in a separate prospective scope. It is not cell-for-cell comparable to
> the historical v4 artifact.

## Protocol and fairness

- Restored the historical NumPy without-replacement draw and canonical JSON manifest fingerprint.
  A 3,693-query MotionSense 8 s, k=8 rebuild exactly reproduces fingerprint
  `fc5a984f438ab59d1992b6ebaf6e556d9a7b66dfe9a8e38fe39bd61987e2d012`.
- New runs identify `deployment-scenarios-v3-20260916` and
  `sealed-manifest-v2-20260916`. Historical rows cannot be extended with a checkpoint scored under
  another manifest protocol.
- Equal-weight normalized fusion remains the declared external-baseline deployment readout. Cosine
  1-NN is now mandatory beside it in every enrolled cell and in matched severity deltas. Prototype
  and ridge remain opt-in. There is no per-cell oracle selection.
- One unloadable stream is recorded as a cell failure; it no longer aborts every sibling dataset in
  the same scenario/k/window group.

## Curriculum corrections

- Cross-placement support now means another anatomical site group in the same dataset. Wrist
  laterality/unspecified-wrist variants do not masquerade as a placement change.
- Cross-dataset support remains exact-key support from another dataset. Its device-family coverage
  is limited by the real eight-source corpus and is disclosed rather than overstated.
- Half of applicable rate-augmentation draws explicitly sample at or below the hardware rate, so
  training includes bandwidth loss instead of mostly interpolation. Training and evaluation use
  the same `resample_poly(..., padtype="line")` edge policy.
- Augmentation replay derives a seed without advancing the episode RNG. Device subsets therefore
  remain identical to the pre-replay trajectory for a fixed data seed.
- `support_per_candidate` and k-share telemetry now reflect the realized post-downsample maximum.
  Support perturbation telemetry reports the fraction of support recordings perturbed, rather than
  the near-certain event that any row in a large support set was perturbed.
- Conditional support-count and sensor-margin gate telemetry carries explicit fractions and survives
  validation aggregation. `--val-repeats-per-dataset` now determines the automatic validation size.

## Checkpoints and model path

- Historical trajectory migration covers acquisition/enrollment mixtures, partial coverage,
  variable support, rate augmentation, and modality dropout. Both a curriculum checkpoint and a
  pre-curriculum checkpoint reached the expected `already reached --steps` guard instead of failing
  trajectory comparison.
- Regime-split plus adaptive-gate plus text-off no longer attempts to copy a `None` gate tensor.
- The adaptive gate intentionally reads the post-residual sensor logits: it decides how much to
  trust the classifier's final sensor evidence, not only its immutable 1-NN floor.
- The stopped Stage B checkpoint predates current sampler corrections and is diagnostic only. Any
  Stage A/B claim requires both arms to be retrained on the current sampler and re-scored on one
  manifest protocol.

## Evaluation efficiency

- HALO residual heads are cached by checkpoint/configuration instead of loaded once per task.
- Sealed evaluation retains released provider states, uses a bounded decoded-feature LRU, and
  computes each model/stream semantic score matrix once for the complete k curve.
- Existing vectorized neighbor scores, fusion, feature caches, paired bootstrap, and periodic result
  writes remain enabled. Numerical baseline frontends were not changed merely for speed, avoiding
  feature-cache invalidation immediately before the evaluation run.

## Verification

- Focused changed-path suite: **168 passed**.
- Baseline duration/view/LiMU-BERT-X/NormWear-CWT suite: **21 passed**.
- Broader available-environment suite: **816 passed, 1 skipped**; the remaining collection/errors
  were missing optional `sentence-transformers`, `torchaudio`, `pyarrow`, or BVH dependencies in
  the temporary test interpreter, not failures on the changed paths.
- Real-corpus curriculum probe: 120 support sets, zero cross-placement or cross-dataset contract
  violations.
- GPU Stage B smoke: calibration, three optimizer steps, validation, telemetry, and checkpoint write
  completed.
- Eight-scenario HARNet smoke: **46 rows**, both 1-NN and fusion on every scored task, **zero task
  failures**.

The code is ready to run the scenario evaluation. Existing pre-v3 scenario tables remain historical
evidence; they are not episode-matched to new checkpoints.
