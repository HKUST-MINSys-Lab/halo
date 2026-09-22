# Experiment roadmap: the three-regime plan

Last verified against code: 2026-09-22. This is the single forward-looking plan. It supersedes the
classifier-versus-neighbours roadmap, archived at
[`docs/archive/roadmap-classifier-vs-neighbours-20260919.md`](../archive/roadmap-classifier-vs-neighbours-20260919.md).
The reasoning, literature and retractions behind every choice here are in the
[2026-09-22 decision record](../journal/2026-09-22-three-regimes-and-unsupervised-adaptation-decisions.md);
this file states only what is decided.

**Status: pre-registration. Nothing in regimes 1 or 2 has been run.** Stage U reads sealed data, so
this plan is written before any number is read, and it runs only on explicit go.

## Standing rules

- Two data roles only: supervised training sources and the sealed six. No development split.
- Every readout is applied **identically to all six encoders** (HALO v4 and the five released
  baselines). A method that cannot be applied to one of them is not used.
- Pre-register, then read. Registered predictions and threats are listed below and are not edited
  after a result is read; a later journal entry names what changed.
- Report metrics uniformly: accuracy **and** macro-F1 wherever a label is available.
- Declare subject-independent evaluation everywhere (all subjects pooled; the harder setting).

## Regime 1 — discovery

*Deployment provides unlabelled recordings. Roster and K are unknown. Tests the encoder, not a
clustering algorithm.*

Protocol copied from Lowe et al., *An Empirical Study into Clustering of Unseen Datasets with
Self-Supervised Encoders* (TMLR 2024, arXiv:2406.02465) — the same experiment in vision.

| element | decision |
|---|---|
| algorithm | k-means++ with 10 restarts. Agglomerative (Ward) is a robustness check on the same result, not a second method. |
| features | L2-normalised; report raw and PCA-to-64-d, because encoder dimensionality spans 72-d to 4608-d. |
| K | estimated by silhouette and Davies–Bouldin over k ∈ [2, 30]. **|K̂ − K| is reported as a metric in its own right.** |
| metric 1 — partition quality | AMI and ARI primary; Hungarian accuracy for comparability; NMI reported only with its cluster-count bias disclosed. |
| metric 2 — semantic geometry | **Representational Similarity Analysis**: Spearman correlation between the K×K centroid-distance matrix and the K×K label-text-distance matrix (Kriegeskorte 2008; Dwivedi & Roig CVPR 2019). Projection-free, therefore fair to HARNet and LiMU-BERT, which have no text encoder. |
| metric 2, secondary | cluster-naming accuracy: oracle-Hungarian (literature's number) vs model-Hungarian on the encoder's own cluster×name cosine matrix vs greedy argmax. The oracle − model gap **is the naming cost**. HARNet and LiMU-BERT are named through our ConSE bridge, disclosed. |
| label-free proxy | silhouette in a UMAP-reduced space, which Lowe et al. find tracks clustering quality without labels — a signal a real deployment could use. |
| data | existing sealed feature caches, all six encoders, CPU. |

Registered caveat on RSA: alignment metrics can be driven by a small subset of dimensions (Bertram
et al. 2026, arXiv:2605.05907); report a dimension-ablation check alongside.

## Regime 2 — unlabelled adaptation: the N-curve

*Deployment provides the roster and a growing pool of unlabelled recordings. No label is ever
provided. The load-bearing regime.*

Per-window zero-shot is stateless, so its accuracy is provably flat in the amount of unlabelled
data. The experiment is a curve in **N**, the size of the unlabelled pool, at fixed k:

> N ∈ {0, 50, 100, 500, 2000, all}, log-spaced; k ∈ {0, 1, 2, 4, 8}. At N=0 the method must
> reproduce the published sealed number exactly — that is the self-consistency check.

Pools exist: the sealed manifests give 39 cells at k=8, 552–16,828 query windows, median ~450 per
class.

### The one established method

**Transductive Zero-Shot and Few-Shot CLIP** (Martin et al., CVPR 2024,
10.1109/cvpr52733.2024.02722) — provisional. The latest method designed for text-initialised
prototypes at k=0, handling k=0 and k>0 in one framework; same lineage as PADDLE. Applied
identically to all six encoders, with pool-mean centring (the existing `corpus_mean` knob).

Two things are checked before the choice is final, in the build: that its block
majorisation–minimisation update can be unrolled differentiably for training (see below); and how
it behaves under imbalanced, partially-covered pools. **Fallback: PADDLE** (Martin et al. 2022,
arXiv:2210.14545) — hyperparameter-free, built for query classes drawn from a larger set than the
support set. Same first author; the swap costs nothing narratively.

Cited for properties, **not** run as arms: OSLO (open-set), α-TIM (imbalance), Burzer et al. 2026
MAP-EM (the HAR-native competitor; its stated limitation — assumes known active classes — is our
regime).

### HALO's contribution arm: trained through the procedure

The curriculum gains episodes that contain an **unlabelled pool** alongside the k supports, with the
pool structured the way deployment pools are:

- drawn from a **different acquisition** (device / placement / rate) than the supports — the axis
  no vision paper has;
- **Dirichlet-imbalanced** marginals, so the robustness Veilleux et al. show is missing is trained in
  rather than hoped for;
- **partial coverage**: some roster entries absent from the pool; **distractors**: pool windows from
  outside the roster.

Mechanism: **the transductive method is the head.** The pool enters training exactly where it
enters inference — through the unrolled transductive iterations — not through a new token role. What
is trained is the encoder, the recording pool, and `p_text` (the projection that supplies HALO's k=0
text prototypes; the analogue of CLIP's image–text alignment). **The v4 learned head — residual
attention stack, gated text blend, corruption auxiliary, unenrolled calibration — is not used in
rung 2**, at training or at inference; it remains the rung-3 case-study vehicle. An earlier draft of
this section proposed a `ROLE_UNLABELED` token acting through v4's residual. That would give HALO a
private inference path and break the identical-inference invariant, so it is withdrawn (to be
recorded in the next journal entry).

Two N=0 checks, not one: (i) **architectural** — the same checkpoint with the pool removed must give
a bit-identical forward pass; (ii) **inductive floor** — the retrained encoder's own N=0 k-curve must
not fall below v4's encoder at any k. The sealed table remains the N=0 column for v4's encoder, not
for the new arm.

**End-to-end means unrolled.** The transductive method above runs inside the training forward pass
on the episode's pool, and the loss backpropagates through it into the encoder — meta-learning the
deployment procedure (cf. Hu et al. 2020, arXiv:2004.12696). Exposure alone is not assumed to
suffice: Ochal et al. (IEEE TAI, 10.1109/tai.2023.3298303) show many meta-learners "will not
automatically learn to balance from exposure to imbalanced training tasks".

### Comparison structure

Three baseline tiers, two of which need no new baseline training, and a four-step HALO ladder in
which each step changes one thing.

| tier | baselines | isolates | new training |
|---|---|---|---|
| 1 | frozen released checkpoints + the transductive method | off-the-shelf reality | none |
| 2 | the corpus-matched arms (built 2026-09-22; trained with differentiable neighbours on our corpus) + the transductive method | "you just saw our corpus" | none — existing checkpoints, new readout |
| 3 | HALO trained through the unrolled method | the claim | HALO only |

HALO ladder, all from random initialisation on our corpus: (1) plain cross-entropy → (2) +
heterogeneity curriculum, **which is today's v4 encoder** → (3) + unlabelled-pool episodes with the
differentiable-neighbour objective, no unrolling (Ren 2018 style; objective-matched to tier 2) →
(4) + unrolled transductive loss. Step 3→4 is "trained through the procedure"; 2→3 is exposure alone;
1→2 is the existing curriculum. Optional tier 4: the unrolled loss dropped into the cheapest matched
arm (HARNet, 32 min) as a transfer check — not required for the claim.

**Invariant: inference is identical for everyone, HALO included.** Unrolling is training-time only.

### Registered threats and their matched controls

| threat | source | control |
|---|---|---|
| transductive gains are an artefact of balanced pools | Veilleux et al. 2022 (arXiv:2204.11181): drops "even below inductive" under Dirichlet marginals | balanced-pool arm; per-cell imbalance reported |
| exposure to heterogeneous pools does not by itself confer robustness | Ochal et al. | curriculum-only arm (no unrolling) vs unrolled arm |
| episodic training is unnecessary; plain CE suffices | Laenen & Bertinetto NeurIPS 2021; Burzer et al. validate for HAR; Zhang 2024 (arXiv:2402.00092). Counter: LibFewShot (TPAMI) | plain-CE control |
| the model exploits the pool's acquisition fingerprint, not its class structure | — | same-config **disjoint-class** pool: gain survives ⇒ domain adaptation; vanishes ⇒ class structure |
| pseudo-label confirmation bias and cluster collapse | Wang et al. CVPR 2022 (10.1109/cvpr52688.2022.01424) | confidence threshold, per-class cap, **collapse rate reported** |
| encoder overfits the unrolled method and regresses inductively | — | k=0 and k-curve at N=0 must not regress |

### What is dropped, and why

- **Class-prior correction (SLD).** Esuli, Molinari & Sebastiani (ACM TOIS 2020, 10.1145/3433164):
  helps only with ≤5 classes and a calibrated classifier; otherwise "negative rather than positive".
  Our rosters are 6–8 and text-cosine is uncalibrated.
- **TENT.** Needs BatchNorm affines; not all six encoders have them — unfair by construction.
- **LaplacianShot** as primary. Balanced-benchmark era; superseded for our regime.
- **"Clustering with known K" (the former U-0b).** With the roster known it is a worse version of
  zero-shot; the only justification is transduction, which is what regime 2 now measures.
- **Estimated-K as the deployment case (U-0c) and kNN purity as a headline (U-0a).** K is given by
  the roster in regime 2; in regime 1 |K̂ − K| is one metric among several. kNN purity survives only as
  a one-line diagnostic.
- **Seeded k-means as U-2.** The unrolled transductive method subsumes it.

## Regime 3 — labelled adaptation

*Deployment provides k labelled examples per class.*

- **k-curve, parameters frozen** (the existing sealed table, v4): demoted to a **case study**. It
  answers a niche need and is no longer the headline.
- **Fine-tuning at matched k, every model including HALO**: the headline adaptation result. Ladder
  by cost — linear probe on frozen features (closed form; TransfHAR's mechanism) → small classifier
  (the frozen-projection path built 2026-09-22) → LoRA / full fine-tune. Scored against the honest
  closed-set bar of **~77–80 %** (DAGHAR LODO, fully supervised, multi-subject/device/dataset). Build
  cost: HALO's own-encoder fine-tune path does not exist yet; NormWear has none and its authors never
  fine-tune.

## Registered predictions

1. Regime 1: HALO leads AMI/ARI on the sealed six by roughly its 1-NN margin; RSA is where the gap
   opens; MM-Fit at k=0 is where we lose. Absolute clustering numbers will be low (subject-independent
   HAR clustering runs ACC 26–51 even for trained deep models); the criterion is a margin.
2. Regime 2, established method on all six: modest gains (a few points), **negative on some cells
   for some encoders** — going negative is the norm in this literature (Burzer's baselines: −12.8 pp).
   The curve rises steeply and plateaus within a few hundred windows per class.
3. Regime 2, HALO curriculum arm: a reliably non-negative, steeper curve than any baseline under the
   same procedure. If the established method is flat on all six, the curriculum arm has nothing to
   improve on and is not built.
4. Regime 3: HALO's fine-tuned curve matches or exceeds the baselines' at every k; reaching ~77–80 %
   at large k is the bar, not a prediction.

## Sequencing and gates

1. **This document and the journal entry** — done 2026-09-22.
2. **Build regime 1 and the regime-2 established arm** on cached features. Build + tests + smoke;
   nothing runs without explicit go. Includes the shared-module extraction in the code plan below.
3. **Run regimes 1 and 2 (established)** on go. CPU, ~hours.
4. **Gate:** build the curriculum arm only if prediction 2's curve rises for at least one encoder.
5. **Regime 3** fine-tune paths. The expensive build; last.

## Open decisions

- Regime 1 and 2 on the sealed six only, or MM-Fit as well.
- Build the curriculum arm now or after the gate in step 4 (recommendation: after).
- An embodied evaluation source (Ego-Exo4D IMU) so the headline framing can be scored.

## Code organisation plan

*A plan. No code is moved by this document.* Regime 3 is the working pipeline that produced every
published number, so the rule is **extract, don't rewrite**, and every move is verified bit-exact
against an existing cached artifact before it lands.

**Principle: separate by concern, not by regime.** All three regimes share the six encoders, the
feature caches, the sealed manifests, the metrics, and provenance. What differs is only the readout on
top of cached features. So regimes are thin drivers over shared modules, not parallel copies.

Today
[`training/support_classifier/sealed_eval.py`](../../training/support_classifier/sealed_eval.py)
(2,382 lines) bundles feature caching, manifest construction, eight model-specific readouts, metrics,
provenance and the CLI. Regimes 1 and 2 need the first two and the last two but none of the readouts.
Copying them creates two versions; importing them makes regime 1 depend on regime 3's file.

Target layout:

```text
evaluation/                      NEW package — consumes checkpoints + manifests, produces artifacts
  encoders.py                    ← from sealed_eval: HALO + baseline feature extraction, _load_or_encode
  manifests.py                   ← from sealed_eval: QueryPlan, *_cells, build_manifest, manifest_fingerprint
  features.py                    ← from sealed_eval: FeatureMemoryCache, cache keys, file hashes
  metrics.py                     ← from sealed_eval: _metric_row  +  NEW: ami, ari, hungarian_acc, rsa, naming_acc
  provenance.py                  ← from sealed_eval: run provenance, atomic JSON  +  NEW: regime/method registry
  regime1_discovery/             NEW: cluster.py, rsa.py, name.py, run.py
  regime2_unlabeled/             NEW: transductive.py (the one method, also importable by training), ncurve.py, run.py
  regime3_labeled/               sealed_eval.py, run_scenarios.py, run_partial_coverage.py,
                                 frozen_baseline_adaptation.py — moved last, behaviour unchanged
  controls/                      NEW: balanced_pool.py, disjoint_classes.py, shuffled_labels.py
training/support_classifier/     keeps train.py, curriculum, sampling, collate, objectives, neighbors.py
                                 (neighbors.py stays: model/ imports it; it is shared train/eval code)
results/tools/                   reporting only — tables, plots, provenance readers
```

Rules that make "which version is which" unambiguous:

1. **Every artifact declares its regime and method.** `evaluation/provenance.py` holds a registry
   (`REGIME ∈ {1, 2, 3}`, `METHOD` enum, `READOUT_VERSION` string such as `discovery-v1`,
   `ncurve-v1`). An artifact cannot be written without them, and they sit in `run_provenance.json`
   beside the checkpoint and manifest fingerprints — the code analogue of RESULTS.md's "which run is
   which" index.
2. **Readout versions bump when the procedure changes**, exactly as `sealed-manifest-v2` and
   `deployment-scenarios-v5` do today, so an old artifact is never mistaken for a new protocol.
3. **Classifier names are unchanged**: v4 / T-numbers per RESULTS.md. Regimes 1 and 2 use no learned
   head, so they name the encoder checkpoint, not a classifier.
4. **The transductive method lives in one file** (`evaluation/regime2_unlabeled/transductive.py`)
   and is imported by both the evaluator and the trainer's unrolled loop. One implementation, so the
   method unrolled in training is provably the one applied to the baselines at test.
5. **Order of migration:** extract the shared modules first (regimes 1 and 2 need them; each
   extraction is `git mv` + import shim + a bit-exact diff of `results.json` on one cached sealed cell);
   build regimes 1 and 2 on them; move regime 3's runners last, and only once the extraction is proven.
   Entry points in `pyproject.toml` are re-pointed at the same time; `halo-sealed-eval` keeps its name.
6. **Case studies and other experiments** are drivers under the regime they read from, never new
   top-level trees. A case study that reads regime-3 caches is `evaluation/regime3_labeled/case_*.py`.
