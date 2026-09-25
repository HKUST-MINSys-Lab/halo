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

## 5. Control: the published method (μ = 0) — the affinity term is what lets HALO use the pool

Same protocol, same temperatures, affinity term off (`runs/evaluations/rung1_tier1_mu0_20260925`):

| encoder | N=0 μ=0 → μ=1 | N=all μ=0 → μ=1 | pool effect (all − 0) μ=0 → μ=1 |
|---|---|---|---:|
| HALO v4 | 51.1 → 50.6 | 49.1 → 51.9 | −2.0 → **+1.3** |
| UniMTS | 34.8 → 35.1 | 36.5 → 34.7 | +1.8 → −0.4 |
| NormWear | 15.1 → 16.3 | 17.1 → 18.4 | +2.0 → +2.1 |
| LiMU-BERT-X | 23.3 → 23.0 | 20.2 → 22.6 | −3.1 → −0.4 |
| HARNet-10 | 32.8 → 31.6 | 31.8 → 31.4 | −1.0 → −0.3 |
| HARNet-5 | 30.5 → 30.5 | 30.1 → 31.1 | −0.4 → +0.6 |

Under the published EM-Dirichlet, a bigger unlabelled pool *hurts* HALO (−2.0). With the affinity
term (neighbours in the encoder's own space vote with their text probabilities) the pool helps it
(+1.3; +2.8 over μ = 0 at N = all). The term is not uniformly good: it costs UniMTS 1.8 at N = all.
μ = 1 was fixed before any result, so this is a finding about the method, not a tuned choice.

## 6. Tier-3 timing (T3-0) found two defects before any fine-tune result

- **Raw-window treatments could not run on CUDA.** `encode_dataset_detailed` returns host tensors on
  its no-grad path; prediction fed them to a CUDA head. The earlier smokes ran on CPU. Fixed
  (`encode_rows` moves features to the head's device).
- **The head measured feature scale.** Per-dimension feature spread differs by >1000× across trunks
  (from-scratch UniMTS 0.0005, released LiMU-BERT-X 0.03, HALO 0.7, from-scratch HARNet 1.4). With
  one plain linear head at lr 1e-3 for 300 steps, released LiMU-BERT-X ended at loss 1.45 and
  scored *below* its own linear probe (55.9 vs 57.9). Raw-window heads are now cosine classifiers
  (scale 10, Baseline++), matching the L2-normalised probe; LiMU-BERT-X full fine-tune on the timing
  cell becomes 65.0 and every model's full fine-tune is ≥ its probe there. Decided from training
  losses on one cell, applied uniformly, before any Phase-A row.
- Cost per fit (one small cell, shared GPU): HALO 19 s, HARNet-5 8 s, LiMU-BERT-X 9 s, UniMTS 46 s
  (full) / 34 s (scratch). Serial Phase A with 3 draws ≈ 4.3 h, so draw 0 runs first as four
  memory-capped per-model processes in parallel; draws 1–2 follow (`--first-draw 1`).
- Profile (py-spy, `runs/evaluations/rung3_timing_20260926/profile.speedscope.json`): 88 % of wall
  in the fit loop; ~26 % of it is CPU collation/preprocessing per step, the rest GPU-bound forward/
  backward. Worth caching the collated support batch if fine-tuning becomes the bottleneck.

## 7. T1-C is not a single-factor change from v4

Rung-1 training refuses text corruption, so the rung-1-trained HALO lacks v4's auxiliary
corrupted-text view as well as gaining the EM-Dirichlet objective. T1-D vs v4 therefore measures
both changes together. (The contextual/evidence switches recorded True in v4's config are forced off
for the evidence-gated head by current code; they were inert for v4 too.)

## 8. Pending at time of writing

Controls (μ = 0, balanced pool, disjoint classes); HALO trained through EM-Dirichlet (40k steps,
`last.pt` primary — declared before results); its tier-1 score and inductive-floor check; the
corpus-matched HARNet arm; tier-3 fine-tuning (timing first). Sections below are appended as they land.
