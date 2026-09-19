# Bounded contextual residual v1: training and evaluation

**Date:** 2026-09-19  
**Status:** completed, mechanically valid negative result  
**Architecture:** `support_contextual_residual_v1`

## Purpose

This run tested the replacement for the failed contextualise-first mixture. The new classifier
keeps a closed-form support floor, adds bounded candidate-aware contextual support corrections,
computes a separate semantic path, and learns a per-candidate residual weight. Its optional
auxiliary terms all use the same path-improvement objective: a named path should improve the
true-class log-odds over a detached reference path.

## Readiness and fixes

The pre-run sweep found no classifier math or gradient-flow defect. It added explicit tests for
partial-enrollment candidate reachability and acquisition-conditioning gradients, repaired stale
profiler assumptions, and fixed a full-corpus loader failure caused by PyTorch's file-descriptor
sharing strategy under 16 workers. The loader now defers eager prefetch until first use and uses
the shared-memory file strategy. The full-corpus worker smoke, classifier gradient smoke, sealed
smoke, scenario smoke, and focused post-fix tests passed. The full suite before the transport-only
loader fix was 986 passed and 1 skipped; the affected loader tests and full-corpus probe passed
after it.

## Training

- 40,000 optimizer steps, four episodes per step, end-to-end encoder and classifier training.
- Fixed multi-resolution filterbank spans: 0.5, 1, 2, and 4 seconds; 8-second training windows.
- Current independent enrollment/acquisition and device-set curriculum.
- Wall time: 2,649 seconds (44.2 minutes).
- Best internal checkpoint: step 35,000, selected only by subject-held-out training-source
  validation; dataset-macro F1 0.7139.
- Best checkpoint SHA-256:
  `471ec05f45b83e563de30e489a875b57ed4e77b1002eba59b6319e8b7ff7ebdd`.
- Final step-40,000 validation dataset-macro F1: 0.6812. The run did not collapse and all monitored
  core gradients remained finite and nonzero.

## Sealed result

Dataset-balanced, single-device macro F1:

| readout | window | k=0 | k=1 | k=8 | k=32 | k=128 |
|---|---:|---:|---:|---:|---:|---:|
| full classifier | 4 s | 49.9 | 58.0 | 62.3 | 65.7 | 68.2 |
| support floor | 4 s | - | 56.7 | 69.0 | 72.6 | 73.9 |
| full classifier | 8 s | 50.0 | 59.7 | 64.0 | 67.6 | 70.3 |
| support floor | 8 s | - | 59.8 | 72.0 | 75.4 | 76.3 |
| full classifier | 16 s | 50.6 | 62.7 | 66.0 | 69.3 | 84.6* |
| support floor | 16 s | - | 63.4 | 74.3 | 76.9 | 93.0* |

`*` At 16 seconds, `k=128` is MotionSense only and is not an aggregate comparable to the other
columns. At `k=64`, the 16-second full classifier/support floor scores are 70.9/77.7.

At 8 seconds the branch decomposition is:

| readout | k=1 | k=8 | k=32 | k=128 |
|---|---:|---:|---:|---:|
| full classifier | 59.7 | 64.0 | 67.6 | 70.3 |
| contextual support | **61.2** | 69.7 | 72.5 | 74.2 |
| closed-form support floor | 59.8 | **72.0** | **75.4** | 76.3 |
| cosine 1-NN | 60.4 | 71.5 | 74.6 | **76.5** |
| semantic path | 47.7 | 43.5 | 37.8 | 32.0 |

The design repaired the prior all-semantic gate collapse: enrollment now changes predictions and
the contextual support path improves the floor at `k=1`. It still fails the central monotonic
safety goal. As support grows, the semantic path degrades sharply and the final learned combination
overrides stronger support evidence. The auxiliary objective improves the contextual support path
locally but does not guarantee that the final path preserves the support floor on unseen labels.

## Scenario result

The representative seven-scenario suite completed 1,132 task units and 2,395 rows with zero
failures. It preserves the expected qualitative benefit in partial enrollment, but it regresses
from the promoted residual classifier in the aggregate partial-coverage and new-domain cells. The
full rows and paired controls are retained for follow-up analysis; no scenario result selected the
checkpoint.

## Reproducibility

The sealed manifest SHA-256 is
`3117900ff269125ae3cdefe3d3fd1067ad0bd9338b6dcf8dcf7ff8630d0fb985`; the scenario manifest
SHA-256 is `0b39f297102fe3d4d54194de99e72d5e98c53e9a96e16571273c1ab813ca96d2`.
Both are byte-identical to the corresponding 2026-09-18 baseline artifacts.

Artifacts:

- [`halo_contextual_residual_v1_sealed_v5_20260919`](../../results/artifacts/halo_contextual_residual_v1_sealed_v5_20260919/)
- [`halo_contextual_residual_v1_scenarios_v5_20260919`](../../results/artifacts/halo_contextual_residual_v1_scenarios_v5_20260919/)
