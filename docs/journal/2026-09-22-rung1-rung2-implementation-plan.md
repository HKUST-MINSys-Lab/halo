# Rungs 1 and 2: implementation plan

Date: 2026-09-22 (late). Status: **plan; nothing built, nothing run.** Design is in
[`docs/overview/roadmap.md`](../overview/roadmap.md); this entry says what exists and is reused,
what is extracted, what is new, how each piece is verified, and in what order. Effort estimates are
build+tests+smoke; every run remains gated on explicit go.

## 0. The interface that makes both rungs uniform

Every rung-1/rung-2 readout consumes exactly two things per model and cell, both of which
[`sealed_eval.py`](../../training/support_classifier/sealed_eval.py) already produces for all six:

| model | (a) features, N×D | (b) zero-shot scores over the roster, N×C |
|---|---|---|
| HALO (v4's encoder) | `_halo_features` — encoder + recording pool, cached | **closed-form text projection refit** on the training corpus (`train.fit_text_projection`, ridge, features→SBERT) then cosine against SBERT(candidates). No v4 head. |
| UniMTS | `adapter.features_for_stream`, cached | native: `native_zero_shot_features_for_stream` (tier `cosine`) vs its own text encoder |
| NormWear | same | native candidate scoring (tier `bespoke`; already used at k=0 in the sealed table) |
| HARNet-5 / -10 | same | training-bank 1-NN + ConSE: `_training_bank_conse_predictions(return_scores=True)` |
| LiMU-BERT-X | same | same bank bridge |
| corpus-matched arms | `MatchedCorpusEncoder` under HALO's encoder contract → the `_halo_features` path with a different checkpoint | text-projection refit, exactly as HALO |

Transductive-CLIP consumes only **(b)** — it softmaxes the scores into probability features on the
simplex and never looks at D. Rung 1 consumes only **(a)**. So neither rung touches a
model-specific readout, and the HALO row is the encoder plus a closed-form bridge, symmetric with
the ConSE bridge the representation-tier baselines get.

**Reading of the method, from the paper (arXiv:2405.18437) and the released repo
`SegoleneMartin/transductive-CLIP` (`src/methods/{zero_shot,few_shot}`, `config/`):**
probability features z_n = softmax(zero-shot logits); one Dirichlet per class with parameter α_k;
objective = Dirichlet data-fitting + a **partition-complexity term that discourages overly balanced
predictions** + support constraints at k>0; solved by block Majorization–Minimization — the α
update is "MM-quadratic" (Algorithm 1: per-coordinate positive root of a quadratic built from
digamma and log-gamma terms; no inner Minka iterations), the assignment update is closed-form
(Appendix C). Appendix F reports zero-shot accuracy **as a function of query-set size** — the N-curve
exists in their paper, which is a shape to compare against. The partition-complexity term is the
opposite of TIM's uniform-marginal push, so the method is *less* exposed to the balanced-query
artefact than TIM; the balanced-pool control stays regardless.

**Unrollability:** every operation in the block MM is an elementary or special function available
in `torch` (`digamma`, `lgamma`, `sqrt`, softmax); with fixed iteration counts it is a finite
differentiable graph. The only non-differentiable step is the paper's zero-shot post-hoc **graph
cluster-to-class assignment** (§4.3 / Appendix D). We fix component identity at initialisation
instead — α_k initialised from the probability mass of samples whose argmax is k — which is
differentiable and needs no matching; the paper's assignment is kept as an inference-only option and
both are reported on one cell. **This is a disclosed deviation.** Provisional choice stands;
PADDLE remains the fallback if the port surfaces a problem.

## Phase 0 — extract the shared modules from `sealed_eval.py`, bit-exact

*Why first:* rungs 1 and 2 need feature loading, manifests, zero-shot scores and provenance, and
must not import them from rung 3's 2,382-line runner.

| new module | moved from `sealed_eval.py` (names unchanged) |
|---|---|
| `evaluation/features.py` | `FeatureMemoryCache`, `feature_cache_schema`, `_cache_key`, `_file_hash`, `_file_hash_for_stat`, `_load_or_encode`, `_halo_features`, `_baseline_feature_state`, `halo_acquisition_vector/rows` |
| `evaluation/manifests.py` | `QueryPlan`, `sealed_cells`, `duration_cells`, `evaluation_cells`, `_aligned_labels`, `_stable_choice`, `build_manifest`, `manifest_fingerprint` |
| `evaluation/zero_shot.py` | `_normalise`, `_build_training_reference_bank`, `_training_bank_conse_predictions` **+ new** `zero_shot_scores(model, features, candidates, …) -> (N×C, info)` dispatching the four routes in §0, and `fit_halo_text_bridge(encoder, bank) -> (W, corpus_mean)` wrapping `fit_text_projection` |
| `evaluation/provenance.py` | `_run_provenance`, `_atomic_json`, `validate_result_rows` **+ new** registry (Phase 3) |

`sealed_eval.py` keeps every readout and its `main()`, and re-imports the moved names so nothing
that imports them today breaks. Import direction is one-way: `evaluation/*` imports from
`baselines/`, `model/`, `training/support_classifier/{neighbors,encoding,collate}` — **never** from
`sealed_eval` or `run_scenarios`.

**Verification — `tests/test_evaluation_extraction.py`:** pick one cached sealed cell
(`motionsense/phone_front_pocket/w=4`, k ∈ {0, 8}, model `halo` + one baseline); run the readout
through `sealed_eval` at the pre-extraction commit and after; assert the result rows, the feature
cache filenames, the artifact fingerprints and the manifest fingerprint are byte-identical. Plus
`test_sealed_eval.py`, `test_scenarios.py`, `test_matched_encoder*.py` green, untouched.
Effort ~½ day. Risk: circular imports; the one-way rule above is the mitigation.

## Phase 1 — rung 1, discovery

Files: `evaluation/metrics.py` (new functions), `evaluation/rung1_discovery/{cluster,rsa,name,run}.py`,
CLI `halo-rung1`.

- **`metrics.py`:** `ami`, `ari`, `nmi` (sklearn, already a dependency, 1.7.2), `hungarian_accuracy`
  (`scipy.optimize.linear_sum_assignment` on the confusion matrix), `silhouette`, `davies_bouldin`,
  `rsa_spearman(centroids, text_vectors)` (`pdist` on each → `spearmanr` of the condensed vectors),
  `naming_accuracy(cos_matrix, truth, mode ∈ {oracle_hungarian, model_hungarian, greedy})`.
- **`cluster.py`:** `cluster_cell(features, *, k, algorithm ∈ {kmeans_pp, ward}, restarts=10,
  seed, pca_dim ∈ {None, 64}) -> labels, centroids`; `estimate_k(features, k_range=range(2, 31))
  -> (K̂_silhouette, K̂_davies_bouldin)`; features L2-normalised first.
- **`rsa.py`:** centroid RDM vs SBERT(roster) RDM; dimension-ablation = drop the top-m principal
  components and recompute, m ∈ {1, 4, 16}; shuffled-label null (`controls/shuffled_labels.py`).
- **`name.py`:** cluster × roster cosine — cosine tier through the native text space; HALO through
  the text bridge; representation tier by running `_training_bank_conse_predictions` **on the
  centroids** (a centroid is a feature vector, so the bridge applies unchanged); three assignment
  modes → the oracle − model gap.
- **`run.py`:** for each (model, 8 s, dataset, stream) in `evaluation_cells` (single-device only):
  features via `_load_or_encode` (cache hit, no encoding); truth via `_aligned_labels`; all subjects
  pooled (subject-independent); K_true = |roster|; cluster at K_true and at each K̂; UMAP-5 +
  silhouette as the label-free proxy (umap-learn present in the environment); rows carry
  `RUNG=1`, `METHOD ∈ {kmeans_pp_10, ward}`, `READOUT_VERSION=discovery-v1`, `pca_dim`, `K_true`,
  `K_hat_*`, all metrics.

**Tests — `tests/test_rung1_discovery.py`:** synthetic well-separated blobs → AMI, ARI, Hungarian
accuracy = 1; RSA = 1 when centroids are an isometry of the text vectors and ≈ 0 under shuffled
labels; oracle ≥ model ≥ greedy naming on synthetic; `estimate_k` recovers K on blobs;
`run.py --smoke` on one cached cell writes to the scratch directory, never to `results/`.
Effort ~1 day. Compute: CPU, seconds–minutes per cell; 39 cells × 6 models → hours.

## Phase 2 — rung 2, the established arm (tiers 1 and 2)

Files: `evaluation/rung2_unlabeled/{transductive,ncurve,run}.py`, `evaluation/controls/{balanced_pool,disjoint_classes}.py`,
CLI `halo-rung2`.

- **`transductive.py` — the one implementation, imported by the evaluator and (Phase 4) the
  trainer.** Port from `src/methods/{zero_shot,few_shot}` of the released repo — port, do not
  re-derive from the paper alone; cite the repo commit in provenance. Signature:
  `transduce(logits: (N,C), *, support_onehot: (N,C)|None, n_outer, n_inner, lam_partition,
  init="text") -> assignments (N,C)`. Pure tensor ops, no in-place writes into the graph, so it is
  differentiable when called with grad enabled. Hyperparameters **verbatim from the repo's
  `config/`** for the zero-shot and few-shot settings, frozen a priori, recorded in provenance;
  nothing is tuned on sealed data.
- **`ncurve.py` — the pool protocol.** The accuracy at different N must be measured on the *same*
  windows or the curve is confounded. So per cell: split queries by physical execution 20/80 with
  the manifest seed → **scored set S (fixed across N)** and pool P; nested draws
  P_50 ⊂ P_100 ⊂ P_500 ⊂ … ⊂ P; transduction runs over S ∪ P_N; accuracy and macro-F1 are reported
  on S only. **N=0 is inductive** (no transduction at all) and must reproduce the published sealed
  row for the same windows — the self-consistency check. Pool-mean centring subtracts the mean of
  S ∪ P_N from features before scoring (the existing `corpus_mean` knob); at N=0 the centring is
  today's. At k>0 the supports are the manifest's `QueryPlan` rows — the same k windows as the
  sealed table — entering as constraints. The class marginal of S ∪ P_N is recorded per cell (it is
  the real, imbalanced one); the balanced-pool control resamples P_N to a uniform marginal.
- **`run.py`:** loops models × cells × k ∈ {0,1,2,4,8} × N ∈ {0,50,100,500,2000,all}; features
  from cache; scores via `zero_shot_scores`; rows carry `RUNG=2`,
  `METHOD=transductive_clip_v1`, `READOUT_VERSION=ncurve-v1`, `N`, `k`, `pool_marginal`,
  `centring`, `init`. Tier 2 = the same run with `--encoder-arch/--matched-pretrained` pointing at
  a corpus-matched checkpoint (the existing flags).
- **Controls:** `balanced_pool.py` (uniform resample of P_N); `disjoint_classes.py` (pool drawn from
  the same stream but from classes outside the roster — reuses the partial-coverage manifests from
  `run_partial_coverage.py`, which already hold out roster classes).

**Tests — `tests/test_rung2_transductive.py`:** (i) N=0 path reproduces the sealed k=0 and k=8
rows bit-exactly on one cached cell; (ii) on synthetic simplex data drawn from known Dirichlet
components, the block MM recovers α and assignments; (iii) `torch.autograd.gradcheck` through
n_outer × n_inner unrolled steps on a tiny problem; (iv) nested pool draws are deterministic under
the seed and P_50 ⊂ P_100; (v) the balanced control yields a uniform marginal; (vi) text-init and
paper graph-assignment agree on a separable synthetic task. Effort: port ~1 day; protocol + runner
~1 day; tests ~½ day. Compute: CPU; one MM run over ≤ 17k × C per (cell, k, N) is seconds; the
grid is hours.

**Tier 2 prerequisite, stated plainly:** the corpus-matched arms are *built and unlaunched* —
HARNet 32 min, LiMU-BERT ~40 min (after the 2026-09-22 rate-contract fix), UniMTS 13.6 h, NormWear
frozen+projection ~40 min: ≈ 15.5 GPU-h that has not happened. Tier 2 needs those checkpoints. No
new *design*, but it is new *compute*, and it is gated like everything else.

## Phase 3 — provenance registry

`evaluation/provenance.py`: `RUNG = {1, 2, 3, 4}`, `METHOD` enum, per-rung `READOUT_VERSION`;
`write_artifact(rows, provenance)` refuses to write without all three; `run_provenance` gains the
fields beside the checkpoint and manifest fingerprints. Rung-3 runners adopt it when they move.
Effort: hours.

## Phase 4 — HALO's unrolled arm (gated on the rung-2 established curve rising; sketch only)

Trainer changes only; the evaluator does not change, which is the point.

- `sampling.Episode` gains `pool: tuple[int, ...]`, `pool_acquisition: str`, `pool_marginal:
  tuple[float, ...]`; `draw_episode` gains `pool_size`, `pool_regime ∈ {compatible,
  cross_placement, cross_dataset}` (the scenario taxonomy), a Dirichlet concentration for the pool
  marginal, a distractor fraction, and reuse of the existing partial-coverage mask.
- `train.run_step`: encode pool rows with the same encoder (`encode_recording_rows`); zero-shot
  logits for query ∪ pool from the text bridge alone (`p_text` cosine — expose it as a function so
  the residual head is not in the path); call `evaluation.rung2_unlabeled.transductive.transduce`
  with grad enabled; the query's soft assignment becomes the (B, C) logits handed to
  `episode_loss`, which is unchanged. Ladder step 3 (curriculum-only) is the same episodes with
  the loss left on the differentiable-neighbour path — one flag.
- Cost: the head's attention is untouched (the pool never enters it); the extra cost is one encoder
  forward over pool rows per episode plus negligible MM iterations — estimate 1.5–2× per step at
  pool_size 32. Sized properly before launch, per the standing rule.

## Done criteria and order

1. Phase 0 green (bit-exact test + all existing tests) — **before anything else touches the
   working pipeline.**
2. Phase 3 (small; the others write through it).
3. Phase 1, Phase 2 in either order; each done = unit tests green + `--smoke` on one cached cell
   per model to scratch, no sealed number read.
4. On go: rung 1 and rung 2 tier 1. Tier 2 after the matched arms are trained (separate go).
5. Gate → Phase 4.

## Decisions needed before the build starts

1. Rung 1 headline at 8 s only (matches the sealed headline) — recommend yes.
2. Scored/pool split 20/80 by execution — recommend yes.
3. Label-free proxy dimensionality: UMAP-5 — check against Lowe et al.'s setting during the build.
4. Launch the corpus-matched arms (≈ 15.5 GPU-h) so tier 2 exists — a separate go.
5. Sealed six only, or MM-Fit too (open since the morning).
