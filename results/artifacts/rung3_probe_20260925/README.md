# Tier 3 (rung 3), cached-feature treatments — 2026-09-25

**What:** for all six encoders, two treatments fitted on the same k labelled windows per class:
`enrollment_frozen` (the parameter-free class-prototype vote on frozen features; the rung-2 readout on
this draw) and `linear_probe` (one logistic-regression layer on frozen features, TransfHAR's
mechanism). k ∈ {1, 4, 16}; **3 independent support draws** per (cell, k); 11 sealed 8 s cells;
scored on rung 1's fixed scored set; supports drawn from the execution-disjoint pool.

The fine-tuning treatments (full fine-tune, from-scratch specialist, LoRA) are a separate GPU run.

**Files:** `results.json.gz` (1,188 rows), `RESULTS.md` (from `results/tools/rung3_report.py`),
`run_provenance.json`.

**Headline (dataset-balanced macro-F1, mean over draws):** HALO leads every baseline at every k under
both treatments — probe 53.5 / 66.8 / 73.3 at k = 1 / 4 / 16 against the best baseline UniMTS
45.4 / 51.6 / 60.8. The within-cell spread across support draws at k = 1 is ≈ 5 points for every
model, which is why single-draw rows would have been unreliable.
