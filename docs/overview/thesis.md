# Thesis and scope

Last verified against code: 2026-09-18.

HALO means **Heterogeneity-Adaptive, Lightweight, Open-vocabulary activity recognition**. Its
contribution is a compact end-to-end system for a realistic deployment condition: IMU channels,
sampling rates, placements, devices, durations, subjects, and candidate activities differ, while a
deployment may provide zero or a few labelled examples.

The system is deliberately support-conditioned. A candidate label is always declared by the task;
available support recordings determine how much sensor evidence exists for that label. Text gives a
bridge when support is absent or incomplete, but text-only arbitrary-motion understanding is not the
claim. “Open-vocabulary” means candidate labels need not belong to one fixed training classifier;
it does not mean unknown-class rejection.

The active research questions are:

1. How much representation quality can a sub-million-parameter physical-time encoder retain across
   heterogeneous wearable acquisitions?
2. Can a learned classifier use support labels and acquisition context without becoming worse than
   a parameter-free nearest-neighbor control?
3. Which deployment mismatches benefit from that learned comparison, and at what clean-condition
   cost?

The project no longer presents itself as a universal IMU foundation model. Its strongest claim is
the complete lightweight system, its explicit deployment protocol, and transparent per-dataset and
per-scenario evidence.

