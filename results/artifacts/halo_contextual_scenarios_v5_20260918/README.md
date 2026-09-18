# HALO contextual arm, scenarios evaluation (recorded negative result)

Run date: 2026-09-18. Architecture `support_contextual_mixture_v1`, the contextualise-first
semantic-voting head of the approved 2026-09-17 plan. Training recipe identical to the residual and
neighbours arms except `--classifier contextual`; final step-40,000 checkpoint, SHA-256 `49370bc308b4c97721556b34a5b89292aebb304632d201619bec2fb542199d47`.

**This head failed and is not a candidate design.** Its learned gate collapsed onto the semantic
branch (measured mean weight 0.996, every candidate above 0.9), so the support path is computed and
discarded and the headline readout barely responds to enrollment. Retained as evidence, with the
branch decompositions (`halo-classifier-semantic-only`, `-support-only`, `-fixed-half-mixture`) that
diagnose it. See the results record for the full reading.

Rows: 2338 scored, 57 explicit
non-scored, 0 failed. Manifests are identical to the baseline artifacts.
