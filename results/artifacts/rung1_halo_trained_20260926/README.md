# Tier 1 (rung 1), level A: HALO trained through EM-Dirichlet — 2026-09-26

**What:** HALO trained from scratch for 40k steps with the rung-1 objective (`--rung1-training`:
cross-entropy through an unrolled EM-Dirichlet + affinity transduction over a sampled pool, T = 30
inside training, roster logits RMS-capped at 4), then scored under exactly the tier-1 protocol of
`results/artifacts/rung1_tier1_20260925` (same cells, split, pool draws, method).
Checkpoint `runs/support-classifier/halo_rung1_40k_20260925/last.pt` (sha256 prefix
0d913376740e4979). Training config and log: `training_run_config.json`, `training_log.jsonl.gz`.

**Scoring policy (declared before any result, journal 2026-09-26 §8):** primary = `last.pt` at its
own calibrated temperature (fitted on held-out training-source windows: T = 31.8), the same procedure
as every other encoder; secondary = T = 30, the temperature it was trained through.
`best_internal.pt` is the same step-40k weights (verified tensor-equal), so it has no separate row.

**Files:** `last_calibrated/` (primary), `last_T30/` (diagnostic) — `results.json.gz` (132 rows each,
per-label telemetry), provenance, calibration; `RESULTS.md` (generated with v4's tier-1 rows for
comparison).

## Result: negative

| encoder | anchor | N=0 | N=100 | N=500 | N=all | pool effect |
|---|---:|---:|---:|---:|---:|---:|
| HALO v4 (reference) | 47.4 | 50.6 | 52.1 | 52.6 | 51.9 | +1.3 |
| HALO rung-1-trained, calibrated T (primary) | 42.9 | 48.3 | 49.3 | 48.9 | 47.9 | −0.4 |
| HALO rung-1-trained, T = 30 | 42.9 | 48.8 | 49.5 | 49.1 | 48.1 | −0.7 |

Training through the adaptation procedure made transduction worth more relative to the model's own
anchor (+5.4 anchor → N=0, vs v4's +3.2) but lowered the anchor by 4.5 points and removed the pool
gain. It is below v4 at every N and on 4 of 6 datasets at N=all.

## Caveats

- Not a single-factor ablation: rung-1 training refuses text corruption, so this arm also lacks v4's
  auxiliary corrupted-text view (journal §7).
- One seed, one run. Not re-tuned: doing so would select on sealed data.

## Inductive floor (T1-E): fails by ~2 points

`INDUCTIVE_FLOOR.md`, `sealed_1nn/` — the trained encoder under the sealed tier-2 cosine 1-NN
readout, vs v4's published k-curve on the 13 shared 8 s cells: k = 1 60.3 → 58.5 (−1.9), k = 8
70.9 → 68.7 (−2.2), k = 32 74.6 → 72.1 (−2.5). Training through EM-Dirichlet slightly degraded the
representation itself, not only the transductive score.
