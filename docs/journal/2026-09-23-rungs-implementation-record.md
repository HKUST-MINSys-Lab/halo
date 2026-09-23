# Rungs 1, 2 and 4 built; the unlabelled-pool training arm built — implementation record

Date: 2026-09-23. Status: **built, unit-tested, not smoke-tested on real data, nothing run.**
Branch `feat/evaluation-package-20260923`; anchor tag
`hist/v3-support-conditioned/pre-evaluation-package-20260923` is the last state before any of this.
`main` is not updated until the bit-exact extraction gate and a debug sweep pass. Plan followed:
[the implementation plan](2026-09-22-rung1-rung2-implementation-plan.md); design:
[`docs/overview/roadmap.md`](../overview/roadmap.md).

## What exists now

| piece | where | status |
|---|---|---|
| Phase 0 — shared modules extracted from `sealed_eval.py` by exact source segment | `evaluation/{features,manifests,zero_shot,provenance}.py` | `sealed_eval` 2,383 → 1,747 lines, re-exports every moved name; 162 pre-existing tests green; `tests/test_evaluation_extraction.py` pins byte-identity against the anchor tag, re-export completeness, one-way dependency; the bit-exact cached-cell check is `slow` and gated on a golden (see below) |
| Phase 3 — registry | `evaluation/provenance.py` | `Rung`, `Method`, `READOUT_VERSION`, `ArtifactProvenance` (method/encoder fixed per artifact or validated per row), `write_artifact` refuses unregistered rows |
| uniform zero-shot route | `evaluation/zero_shot.py` | `zero_shot_scores` (HALO: closed-form SBERT bridge refit on the training bank, **no v4 head**; HARNet/LiMU-BERT: bank + ConSE; UniMTS/NormWear: native), `probability_features` (softmax T = 30 from the released config; NormWear's negative-L1 std-scaled, disclosed), `ProviderScorer` — the single feature + score path all three drivers use |
| rung 1 | `evaluation/rung1_discovery/{cluster,rsa,name,run}.py`, `evaluation/metrics.py` | k-means++ ×10 / Ward, raw + PCA-64, K̂ (silhouette, Davies–Bouldin) with \|K̂ − K\|, AMI/ARI/NMI/Hungarian, RSA on true-class and oracle-matched cluster centroids vs SBERT geometry with shuffled null and top-PC ablation, three-way naming through each provider's own route, UMAP-silhouette proxy; `evaluate_cell` is pure and tested |
| rung 2 | `evaluation/rung2_unlabeled/{transductive,ncurve,run}.py`, `evaluation/controls/` | EM-Dirichlet ported from the released repo (`master`, fetched 2026-09-23), batched, `candidate_mask` and `row_mask` exact, differentiable at fixed iteration counts; identity assignment for training + the reference's graph/basic matching for inference; execution-level scored/pool split, nested draws, cell-level shared supports at k > 0, N = 0 inductive rows (also over all windows for row-for-row comparison with the sealed file); balanced-pool and disjoint-class controls |
| Phase 4 — the training arm | `training/support_classifier/sampling.py`, `train.py` | `Episode.pool*`, `attach_pool`, `draw_batch(pool_*)`, pool telemetry; `split_pool`, `probability_features_torch`, `pooled_episode_logits` (`transductive` = the evaluator's `transduce` unrolled; `soft_kmeans` = one Ren-2018 E-step), `run_step(pool_mode, …)`, nine `--pool-*` flags persisted and resume-checked |
| rung 4 | `evaluation/rung4_finetune/{lora,finetune,run}.py` | LoRA wrapper; `enrollment_frozen` / `linear_probe` / `small_classifier` on cached features (all six); `lora` / `full_finetune` / `scratch_specialist` on raw windows (HALO; HARNet-5, LiMU-BERT-X, UniMTS as released trunks under the contract; NormWear unsupported by declaration); scored on rung 2's split |
| entry points | `pyproject.toml` | `halo-rung1`, `halo-rung2`, `halo-rung4`; `halo-train --pool-size … --pool-mode …` |
| tests | `tests/test_{evaluation_extraction,rung1_discovery,rung2_transductive,rung4_finetune,pool_curriculum}.py` | all synthetic; no cached data needed |

## Decisions made during the build (not in the plan)

1. **`metrics.classification` delegates to `baselines.scoring.classification_metrics`** so rung 2's
   and rung 4's accuracy / macro-F1 are the sealed table's definitions, not a second one.
2. **Two RSA variants.** `rsa_class` uses true-label centroids (the geometry claim, free of
   clustering error); `rsa_cluster` uses discovered clusters mapped by the oracle assignment (the
   end-to-end number). Naming for native-tier providers scores centroids of their *native
   zero-shot* features under the clustering's labels, because a centroid in enrollment space cannot
   be scored in a different native space.
3. **Rung 2's pool protocol fixes the scored set** (20 % of executions) so accuracy at different N
   is on the same windows. k > 0 supports are a **cell-level shared set** drawn from the pool
   partition — transduction is joint, so the sealed manifest's per-query supports cannot be used.
   Consequence: the N = 0 row is bit-comparable to the sealed table at **k = 0 only** (the
   `all_windows` reproduction row); at k > 0 it is the same *readout* (`prototype`) on a different
   draw, and is labelled so.
4. **Centring is not applied by default in rung 2.** The reference does not centre; pool-mean
   centring was in the roadmap as the `corpus_mean` knob. It is *not* wired as an option in this
   build — recorded as an open item rather than silently added.
5. **`paper_lambda` keeps the released integer-division rule**, so a roster of fewer than five
   classes gets λ = 0 (partition term off). Recorded per row, not "fixed".
6. **Rung 4 registers `enrollment_frozen`** as a rung-4 method: the rung-3 readout re-run on rung
   4's own draw, so the crossover is like-for-like; named distinctly from the sealed rows.
7. **Rung 4's small classifier is the corpus-matched projection MLP + linear head trained by CE on
   the supports**, not the differentiable-neighbour objective (k = 1 makes leave-one-out prototypes
   impossible). `linear_probe` is deterministic L-BFGS logistic regression (TransfHAR's mechanism).
8. **HARNet-10 has no fine-tuning path** (only the harnet5 trunk exists under the contract); it
   gets the cached-feature treatments only.
9. **The trainer's `soft_kmeans` mode falls back to the head for zero-shot episodes** (no supports
   to seed from); `transductive` handles k = 0 natively.

## Smoke commands for the sweep (none run here)

```bash
# Phase 0 functional gate — produce the golden ONCE at the anchor tag, then compare (slow test)
git worktree add /tmp/halo-anchor hist/v3-support-conditioned/pre-evaluation-package-20260923
# see tests/test_evaluation_extraction.py::test_sealed_results_on_a_cached_cell_are_bit_identical_to_the_anchor_golden

# rung 1 / 2 / 4 on one cached cell, HALO only
python -m evaluation.rung1_discovery.run --out /tmp/r1 --models halo --halo-checkpoint <v4> \
    --feature-cache <shared cache> --cells 1 --no-umap --null-draws 20 --k-range 2 6
python -m evaluation.rung2_unlabeled.run --out /tmp/r2 --models halo --halo-checkpoint <v4> \
    --feature-cache <shared cache> --cells 1 --k 0 1 --pool-sizes 0 50 --n-iter 3 --n-iter-mm 10
python -m evaluation.rung4_finetune.run --out /tmp/r4 --models halo --halo-checkpoint <v4> \
    --feature-cache <shared cache> --cells 1 --k 1 --steps 5 --treatments enrollment_frozen linear_probe lora

# the training arm, a few steps: transductive (ladder step 4) and soft_kmeans (step 3)
halo-train --smoke --steps 3 --pool-size 16 --pool-mode transductive --classifier evidence_gated
halo-train --smoke --steps 3 --pool-size 16 --pool-mode soft_kmeans --classifier evidence_gated
```

Things the sweep should look at first: (i) that rung 2's `all_windows` N = 0, k = 0 row equals the
sealed `native_zero_support` / bank row for the same cell; (ii) feature-cache hits (no re-encoding)
in all three drivers; (iii) the trainer's pooled step cost vs `--pool-size 0`; (iv) `transduce`
numerics in bf16 autocast (the reference runs float32 — `pooled_episode_logits` upcasts features
to float32 before the softmax, but the unrolled MM has not been exercised under autocast).

## Not done / open

- Pool-mean centring as a rung-2 option (roadmap says "the existing `corpus_mean` knob"; not wired).
- Rung 3's runners are not moved into `evaluation/rung3_frozen/` (planned last; nothing depends on it).
- Tier 2 of rung 2 needs the corpus-matched arms **trained** (≈ 15.5 GPU-h, separate go).
- No real-data smoke has been run; the golden for the extraction gate has not been produced.
- Docs: `docs/contracts/evaluation_protocol.md` and `curriculum.md` do not yet describe rungs 1/2/4
  or the pool curriculum; to be written once the sweep has confirmed behaviour.
