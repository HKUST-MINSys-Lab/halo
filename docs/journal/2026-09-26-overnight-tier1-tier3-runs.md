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

**Balanced pool** (`..._balanced_...`): HALO 50.6 → 51.9, the same pool effect with a smoother
curve (52.1 at N = 2000) — not an artefact of the natural class imbalance.

**Disjoint classes** (pool holds none of the scored classes; F1 over the scored half of the roster,
hence higher anchors): the pool still helps HALO (+3.6), UniMTS (+4.8), NormWear (+3.0); LiMU-BERT-X
drops (−6.1). So the pool effect is not "learning the target classes from unlabelled data". Likely
mechanism — explaining away: without a pool, scored windows the text wrongly assigns to other classes
become those classes' clusters and reinforce the error; pool windows of the real other classes anchor
those clusters elsewhere and release the scored windows. The paper must describe the gain this way.
All three controls are in `results/artifacts/rung1_tier1_20260925/controls/`.

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

## 8. Scoring policy for the rung-1-trained HALO (declared 03:15 HKT, before any T1-D result)

T1-C ran EM-Dirichlet inside training at a fixed T = 30; the tier-1 protocol scores every encoder at
its own calibrated temperature. **Primary:** calibrated T (the same procedure as every other
encoder; fitted on held-out training-source windows). **Secondary diagnostic:** T = 30, the
temperature it was trained through. `last.pt` primary, `best_internal.pt` secondary (declared
earlier). Training ended 03:13 HKT; internal validation dataset-macro F1 0.14 (2.5k) → 0.367 (40k).

## 9. T1-D: HALO trained through EM-Dirichlet is worse than v4 (primary, `last.pt`)

`runs/evaluations/rung1_halo_trained_last_20260925`; calibrated T = 31.8 (≈ its training T = 30,
so the declared T = 30 diagnostic cannot differ much). Dataset-balanced macro-F1:

| encoder | anchor | N=0 | N=100 | N=500 | N=all | pool effect |
|---|---:|---:|---:|---:|---:|---:|
| HALO v4 | 47.4 | 50.6 | 52.1 | 52.6 | 51.9 | +1.3 |
| HALO trained through EM (40k, from scratch) | 42.9 | 48.3 | 49.3 | 48.9 | 47.9 | −0.4 |

Lower at every N. Training did make transduction worth more *relative to its own anchor* (+5.4 to
N=0 vs v4's +3.2), but from a base 4.5 points lower, and the pool no longer helps (ut_complex N=0
39.3 → N=all 30.7). Per dataset it wins only shoaib at N=all (67.2 vs 65.5) and usc_had (38.6 vs
38.0). Confounds (§7): the arm also lacks v4's auxiliary corrupted-text view. Reading: as with the
2026-08/09 Phase-B findings, training through the adaptation procedure did not beat the untrained
procedure on a good encoder; level A is **negative** on this run. Not re-tuned: that would select
on sealed data.

## 10. Tier 3 Phase A, draw 0: HALO frozen already beats every fine-tuned baseline

`results/artifacts/rung3_phaseA_20260926` (draw 0; like-for-like against the probe run's draw 0).
Dataset-balanced macro-F1, k = 1 / 4 / 16:

| encoder | frozen enrollment | full fine-tune | from scratch |
|---|---|---|---|
| HALO v4 | 53.1 / 66.0 / 71.8 | 50.1 / 69.7 / 76.2 | 42.2 / 59.6 / 73.2 |
| UniMTS | 47.7 / 60.6 / 60.1 | 50.2 / 64.4 / 67.4 | 34.7 / 44.0 / 59.9 |
| HARNet-5 | 34.7 / 42.1 / 48.2 | 42.6 / 54.9 / 68.1 | 36.3 / 47.9 / 63.3 |
| LiMU-BERT-X | 45.1 / 51.6 / 55.3 | 45.1 / 58.3 / 60.8 | 41.6 / 52.2 / 61.7 |

HALO with no parameter update is ≥ every baseline's full fine-tune at every k; fine-tuning HALO pays
from k = 4 and costs 3 points at k = 1. At k = 16 even a from-scratch HALO architecture (73.2) beats
every fine-tuned baseline. One draw only: k = 1 differences under ~5 are unresolved until draws
1–2 land.

**Update 04:01 — draws 1–2 for HARNet-5 and LiMU-BERT-X weaken the k = 16 claim.** HARNet-5 full
fine-tune at k = 16 is 68.1 / 71.6 / 72.2 across draws (mean 70.7) vs HALO frozen's 3-draw 71.3: a
near-tie, not a win. Draw 0 happened to be HARNet-5's worst k = 16 draw. The claim to carry is "HALO
frozen matches the best fine-tuned baseline at k = 16 and leads at k ≤ 4", pending HALO's and
UniMTS's draws 1–2.

## 11. Speed: sealed evaluation was 59 % one unvectorised line

The T1-E sealed run projected 45 min for 13 cells. A 3-minute py-spy copy showed 59 % of wall time
on one line of `_halo_evidence_gated_predictions` — a per-episode `torch.as_tensor(..., device=cuda)`
plus an indexed device write inside the batch loop (the classifier forward was 23 %). Five sites in
`sealed_eval.py` had the pattern; all now fill host arrays and transfer once per batch. Predictions
bit-identical (checked on 150 plans incl. zero-support); throughput ≈ 2× on the first cells. The
running T1-E was left on the old code (a restart would not have finished sooner).

Also noted, not changed tonight: the tier-3 shards and evals run as separate processes, which
**time-slice** the GPU (no MPS), and batch-32 fine-tuning of small trunks leaves SMs idle while
`nvidia-smi` reports 99 %. Enabling MPS, or fitting several (k, draw) jobs in one process, would
raise real utilisation.

**T1-E (inductive floor), 03:56:** the rung-1-trained encoder's sealed cosine 1-NN k-curve is below
v4's by 1.9 / 2.2 / 2.5 at k = 1 / 8 / 32 (13 shared cells). The pre-registered floor check fails:
the representation itself got slightly worse. The T = 30 diagnostic (48.1 at N=all) confirms the
negative T1-D result does not hinge on the temperature. Artifact:
`results/artifacts/rung1_halo_trained_20260926/`.

## 12. T1-F: HARNet trained on our corpus closes about half the gap

`results/artifacts/rung1_matched_harnet_20260926` — HARNet architecture trained from scratch on our 8 sources with
our pipeline (neighbour head, 40k steps, 03:56–04:43; peaks ~22 GB GPU, so it ran alone), scored under
tier 1 via a ridge text bridge (T = 7.4): 33.7 / 39.3 / 40.8 (anchor / N=0 / N=all), pool effect
+1.5. Strongest non-HALO row; released HARNet-5 is 31.2 / 30.5 / 31.1 and HALO 47.4 / 50.6 / 51.9.
About 10 of HALO's ~21-point lead over released HARNet-5 is reproduced by our corpus and recipe on
a HARNet trunk; ~11 stays with HALO. The pool effect is not unique to HALO's space.

## 13. Pending at time of writing

Controls (μ = 0, balanced pool, disjoint classes); HALO trained through EM-Dirichlet (40k steps,
`last.pt` primary — declared before results); its tier-1 score and inductive-floor check; the
corpus-matched HARNet arm; tier-3 fine-tuning (timing first). Sections below are appended as they land.
