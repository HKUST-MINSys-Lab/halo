# HALO evidence-aware v2, step 32,500 (companion) — sealed evaluation

Run date: 2026-09-19.
Same training run as the step-40k primary (`halo_evidence_aware_v2_40k_20260919`, code `a9e8d91`).
**Checkpoint: `best_internal.pt`, step 32,500**, chosen by v2's internal rule (equal weight on the
seen-label scenario-balanced and label-held-out subject-held-out panels); declared as the companion
before any sealed number existed. SHA-256 `be0dd82e709a5812de8acffbb29dbcae280192875fc3c4a46bde6a7d997876ea`.
Manifests identical to the released-baseline artifacts on every shared cell.

**Result: same negative as the primary, within about one point.** The classifier stays flat near 34-36
macro F1 at every k; the router routes to label meaning. See `docs/results/RESULTS.md`, section
"The evidence-aware v2 follow-up", for the diagnosis and caveats (label holdout of core activities;
the evaluator's seen/unseen flag ignores the holdout).

8 s dataset-balanced macro F1, classifier: 34.2/34.5/35.4/35.7/36.0 at `k=0/1/8/32/128`; unmodified support vote 55.1/66.8/70.5/72.2 at `k=1/8/32/128`.
