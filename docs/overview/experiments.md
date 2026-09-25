# Experiment runbook

Last updated: 2026-09-25. The single list of what we train and evaluate, in order, with the exact
commands, where outputs go, and status. Design and reasoning live in [roadmap.md](roadmap.md); this
file is operational. Update the **status** column when something runs; results go to
[RESULTS.md](../results/RESULTS.md) and `results/artifacts/<name>/`, each with a README.

**Tiers** (the docs also call them rungs): **1** unlabelled adaptation (k = 0, a growing pool);
**2** k labelled examples, parameters frozen; **3** k labelled examples, fine-tuning allowed.

## Standing rules

- Two data roles: 8 supervised training sources, 6 sealed evaluation datasets. No dev split.
  Nothing is selected on sealed data; sealed runs happen only on an explicit go (given for tier 1
  on 2026-09-25).
- One inference procedure per tier, identical for every encoder. Every row carries its route,
  temperature and provenance; every artifact is per cell (dataset × stream) so results can be
  re-aggregated or diagnosed later.
- Before any long run: time it, and profile anything slow (see the 2026-09-25 speed commit).
- Commit and push a clean state before and after each experiment.

## Shared inputs

| input | path |
|---|---|
| HALO v4 checkpoint | `runs/support-classifier/halo_t6_40k_20260920/last.pt` |
| HALO v4 feature cache | `cache/evaluations/feature_cache_halo_t6_40k_20260920_final40k` |
| baseline feature caches | `cache/evaluations/feature_cache_baselines_v2_20260917_final`, `..._20260917` |
| rung-1 feature cache (new rows) | `runs/evaluations/rung1_cache` |
| per-encoder temperatures | `runs/evaluations/rung1_tier1_20260925/temperature_calibration_w8.json` |

`CACHES="--cache-read-dirs cache/evaluations/feature_cache_halo_t6_40k_20260920_final40k cache/evaluations/feature_cache_baselines_v2_20260917_final cache/evaluations/feature_cache_baselines_v2_20260917 --feature-cache runs/evaluations/rung1_cache"`

## Tier 1 — unlabelled adaptation

| id | experiment | command | cost | status |
|---|---|---|---|---|
| T1-A | six released encoders + HALO v4 under EM-Dirichlet + affinity, per-encoder calibrated T, N ∈ {0, 50, 100, 500, 2000, all}, 11 sealed 8 s cells | `halo-rung1 --out runs/evaluations/rung1_tier1_20260925 --halo-checkpoint <v4> $CACHES` | ~25 min | running 2026-09-25 (relaunched after a GPU OOM from co-scheduled jobs; now checkpoints every cell) |
| T1-B | controls: published method (μ = 0), balanced pool, disjoint classes | same + `--affinity-mu 0` / `--control balanced_pool` / `--control disjoint_classes`, each with `--temperature-file <T1-A calibration>` | ~25 min each | queued (`runs/evaluations/queue_tier1_controls.sh`) |
| T1-C | **HALO trained through EM-Dirichlet** (ladder step 3, from scratch, 40k steps) | `halo-train --out runs/support-classifier/halo_rung1_40k_20260925 --rung1-training --pool-mode transductive --pool-size 500 --steps 40000` | ~1.5 h (0.12 s/step + validation) | **running** since 01:55 HKT 2026-09-25. Checkpoint policy declared before any result: `last.pt` (step 40k) primary, `best_internal.pt` secondary diagnostic. Gate A read from T1-A when it lands |
| T1-D | score T1-C under the same protocol | `halo-rung1 --out ... --models halo --encoder-label halo:rung1-trained --halo-checkpoint <T1-C best>` | ~5 min | after T1-C |
| T1-E | inductive-floor check: T1-C's encoder must not fall below v4's tier-2 1-NN k-curve | `halo-sealed-eval --models halo --halo-checkpoint <T1-C> --readouts 1nn ...` | ~20 min | after T1-C |
| T1-F | corpus-matched baseline arms (tier "2" of rung 1): HARNet, LiMU-BERT, UniMTS trunks trained on our corpus with differentiable neighbours, then scored like T1-A | `halo-train --encoder-arch {harnet,limubert,unimts} --classifier neighbors --steps 40000`; then `halo-rung1 --encoder-label matched:<arm>` | HARNet 32 min, LiMU-BERT ~75 min, UniMTS 13.6 h (cloud) | not started |
| T1-G | level B (learnable method constants) | not built | — | later |

## Tier 2 — labels, frozen (the case study)

| id | experiment | status |
|---|---|---|
| T2-A | v4 sealed + scenario tables vs the five baselines | **done** (RESULTS.md, 2026-09-20) |
| T2-B | v5 online memory reader: train (`train_memory`, from v4, training data only), debug on internal validation, then online predict-then-update evaluation against its fixed-vote floor, no memory, v4 and (tier-1 setting) EM-Dirichlet | trainer built; **online evaluator not built**; after tier 1 |

## Tier 3 — fine-tuning

| id | experiment | command | status |
|---|---|---|---|
| T3-0 | time one cell for the Phase-A treatments | `halo-rung3 --out ... --cells 1 --k 4 --treatments linear_probe full_finetune scratch_specialist` | not started |
| T3-A | Phase A: linear probe (all six), full fine-tune + from-scratch specialist (HALO, HARNet-5, LiMU-BERT-X, UniMTS); k ∈ {1, 4, 16}; 3 support draws; 11 cells | `halo-rung3 --out runs/evaluations/rung3_phaseA --k 1 4 16 --support-draws 3 --treatments linear_probe full_finetune scratch_specialist $CACHES --halo-checkpoint <v4>` | after T3-0 |
| T3-B | Phase B: LoRA, small classifier, k ∈ {2, 8, 32} | same with the other treatments | if Phase A is interesting |

Fine-tuning boundaries (all a priori): probe = one linear layer on frozen features; LoRA = rank 8 on
every linear and ungrouped conv layer, BatchNorm statistics frozen; full = whole encoder at 0.1× the
head's learning rate; specialist = same architecture from random init. 300 steps, batch 32, AdamW,
cosine, head lr 1e-3, weight decay 0.05. Heads sit on each model's own features.

## Telemetry every result must carry

Per row: dataset, stream, window, model/encoder label, route, temperature, k, N, method, control,
assignment, macro-F1 / accuracy / balanced accuracy (+ per-label F1 where available), n scored, n
transduced, λ, μ, knn, neighbour purity, same-execution share, other-execution purity, collapsed
components, pool class marginal. Per artifact: provenance (rung, method, readout version,
checkpoint + manifest fingerprints, argv, git state), calibration file, run log.
