# Overnight 2026-09-25/26: tier-1 results, temperature calibration, speed fixes, tier-3 probe

Date: 2026-09-26 (HKT night of 25→26). Status: **results for tier 1 (six encoders + controls) and the
tier-3 cached-feature treatments; HALO's rung-1-trained arm and the fine-tuning runs in progress.**
Plan of record: [experiment runbook](../overview/experiments.md).

## 1. Temperature calibration (method change, before any tier-1 result)

The smoke tests showed one shared temperature (T = 30) made the providers' probability features
incomparable. Each encoder now gets the maximum-likelihood temperature on subject-held-out windows of
the 8 training sources (`evaluation/rung1_unlabeled/calibration.py`). Fitted at 8 s: HALO 13.9
(≈ 1/0.07, the temperature its text head was trained with — a sanity check), HARNet-5 4.3, HARNet-10
4.5, LiMU-BERT-X 4.2, UniMTS 372, NormWear 0.33 (spread units). Held-out calibration error fell for
all six (e.g. HARNet-5 0.52 → 0.045, NormWear 0.81 → 0.03).

## 2. Speed and robustness (profiling-driven)

- **py-spy on rung-1 training:** 93 % of wall time was one validation pass, 88 % inside
  `update_alpha`'s MM loop — kernel-launch-bound on (B, C, C) tensors. `torch.compile` fuses one MM
  step on CUDA (0.094 → 0.034 ms, equal to 5e-7); rung-1 validation batches 8× more support sets per
  call (its cost is flat in batch size). 60 steps + one validation: ~430 s → 56 s; training
  0.12 s/step, so a 40k run is ~1.3 h.
- **Calibration** first encoded the whole training corpus through UniMTS/NormWear although only
  held-out windows are scored; fixed (native providers encode only the sampled held-out rows).
- **A tier-1 run died of GPU OOM** from orphaned data-loader workers of a killed profiling run plus
  co-scheduled jobs. The rung-1 driver now checkpoints rows after every cell (`--resume`); jobs are
  queued by GPU-memory budget (`runs/evaluations/queue_*.sh`).

## 3. Tier 1 results (six encoders, 11 cells) — `results/artifacts/rung1_tier1_20260925/`

Dataset-balanced macro-F1, k = 0, identity assignment:

| encoder | anchor (no unlabelled data) | N=0 (scored set only) | N=all | peak |
|---|---:|---:|---:|---:|
| HALO v4 | 47.4 | 50.6 | 51.9 | 52.6 @ N=500 |
| UniMTS | 31.9 | 35.1 | 34.7 | 36.3 @ N=50 |
| HARNet-10 | 32.0 | 31.6 | 31.4 | — |
| HARNet-5 | 31.2 | 30.5 | 31.1 | — |
| LiMU-BERT-X | 21.9 | 23.0 | 22.6 | — |
| NormWear | 10.4 | 16.3 | 18.4 | 18.6 @ N=2000 |

Reading: most of the unlabelled-data effect is between the anchor and N=0 — "N=0" already transduces
over the scored set (hundreds to thousands of windows), so it is not a no-unlabelled-data condition.
The pool then adds ≤ 2 points. HALO is best on every dataset and gains +4.5 over its anchor at
N=all; Gate A (pool curve rises for ≥ 1 encoder) passes narrowly (HALO +1.3, NormWear +2.1 from N=0).

## 4. Tier 3, cached-feature treatments — `results/artifacts/rung3_probe_20260925/`

Frozen enrollment and linear probe for all six, k ∈ {1, 4, 16}, 3 support draws. HALO leads at every
k under both (probe 53.5 / 66.8 / 73.3 vs UniMTS 45.4 / 51.6 / 60.8). Support-draw spread at k = 1
≈ 5 points for every model.

## 5. Pending at time of writing

Controls (μ = 0, balanced pool, disjoint classes); HALO trained through EM-Dirichlet (40k steps,
`last.pt` primary — declared before results); its tier-1 score and inductive-floor check; the
corpus-matched HARNet arm; tier-3 fine-tuning (timing first). Sections below are appended as they land.
