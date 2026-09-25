# Experiment roadmap: the three-rung plan

Last verified against code: 2026-09-24. This is the single forward-looking plan. It supersedes the
classifier-versus-neighbours roadmap, archived at
[`docs/archive/roadmap-classifier-vs-neighbours-20260919.md`](../archive/roadmap-classifier-vs-neighbours-20260919.md).
The reasoning, literature and retractions behind every choice here are in the
[2026-09-22 decision record](../journal/2026-09-22-three-regimes-and-unsupervised-adaptation-decisions.md),
its [addendum](../journal/2026-09-22-four-rungs-and-rung2-head-correction.md), and the
[2026-09-23 renumbering](../journal/2026-09-23-three-rungs-discovery-dropped.md), the
[2026-09-23 learnability decision](../journal/2026-09-23-rung1-learnability-ceiling-and-fixes.md) and
the [2026-09-24 affinity and reorganisation entry](../journal/2026-09-24-rung1-affinity-and-repository-reorganisation.md); this file states
only what is decided. **Numbering:** the 09-22 entries say "three regimes" and then "four rungs";
the discovery rung was dropped on 2026-09-23 and the remaining three were renumbered. This file and
the Notion ⭐ Latest series are authoritative; older entries keep their own numbering.

**Status: pre-registration. Nothing on rungs 1 or 3 has been run.** Rungs 1 and 3 read sealed
data, so this plan is written before any number is read, and it runs only on explicit go.
**Built 2026-09-23/24 on `main`** — rungs 1 and 3, the embedding-affinity term and the pool training
arm; unit-tested, not smoke-tested on real data; see the
[implementation record](../journal/2026-09-23-rungs-implementation-record.md) (written under the
old numbering). All work is on `main`; the state before the `evaluation/` extraction is tagged
`hist/v3-support-conditioned/pre-evaluation-package-20260923`, and the functional golden for the
extraction check (`tests/test_evaluation_extraction.py`, slow) has not yet been produced.

## The ladder

Three rungs, hardest first, ordered by how much a deployment provides. The same six encoders — HALO
and five released baselines — are scored on every rung through the same inference procedure.

| rung | the deployment provides | status |
|---|---|---|
| 1 | the roster, and a growing pool of unlabelled recordings; no labels ever | built, gated |
| 2 | k labelled examples per class; parameters frozen | **done** — the case study |
| 3 | k labelled examples per class; fine-tuning allowed | built |

## Standing rules

- Two data roles only: supervised training sources and the sealed six. No development split.
- Every readout is applied **identically to all six encoders** (HALO and the five released
  baselines). A method that cannot be applied to one of them is not used.
- Pre-register, then read. Registered predictions and threats are listed below and are not edited
  after a result is read; a later journal entry names what changed.
- Report metrics uniformly: accuracy **and** macro-F1 wherever a label is available.
- Declare subject-independent evaluation everywhere (all subjects pooled; the harder setting).

## Rung 1 — unlabelled adaptation: the N-curve

*Deployment provides the roster and a growing pool of unlabelled recordings. No label is ever
provided. The load-bearing rung.*

Per-window zero-shot is stateless, so its accuracy is provably flat in the amount of unlabelled
data. The experiment is a curve in **N**, the size of the unlabelled pool, with **no labelled
example at any point (k = 0)**:

> N ∈ {0, 50, 100, 500, 2000, all}, log-spaced. The scored set is a fixed 20 % of executions, so
> every N is measured on the same windows. **The curve's null is N = 0 under the same method**
> (transduction over the scored set alone). A separate *inductive* anchor — the per-window zero-shot
> arg-max — reproduces the published sealed k=0 row: exactly for every baseline, and for HALO v4
> too, whose k=0 classifier reduces to its `p_text` arg-max (verified equal on all 11 sealed 8 s
> cells, 2026-09-25). It reads a different space and is never the baseline for a gain.

Pools exist: the sealed single-device cells hold 552–16,828 windows each, median ~450 per class.
(k > 0 plus a pool — labels *and* unlabelled data — is computable with the same code, but it is a
semi-supervised condition, not rung 1, and is not in the plan.)

### The one established method

**Transductive Zero-Shot and Few-Shot CLIP** (Martin et al., CVPR 2024,
10.1109/cvpr52733.2024.02722) — provisional. The latest method designed for text-initialised
prototypes at k=0, handling k=0 and k>0 in one framework; same lineage as PADDLE. Applied
identically to all six encoders. Ported from the released repository
(`evaluation/rung1_unlabeled/transductive.py`), not re-derived; unrollable — every step is a torch
primitive and fixed iteration counts make it a finite differentiable graph.

Checked in the build: unrollability holds; behaviour under imbalanced, partially-covered pools is
a registered threat below, not assumed.

Two disclosed deviations from the reference, both applied to all six encoders:

- **λ = N for every roster.** The reference's `int(C / k_eff) · N` (zero-shot `int(C / 5) · N`) was
  written for 1,000-class label spaces with 3–10 classes present; our label space is the declared
  roster, all of it deployed, so the non-oracle `k_eff` is the roster size.
- **An embedding-affinity term** (added 2026-09-24). EM-Dirichlet sees each window only through its
  text probabilities, so rung 1 would otherwise compare text heads, not encoders. Each window's
  update also gets `μ · log(neighbour vote)`: the cosine-weighted mean of the *text probabilities*
  of its 10 nearest neighbours in **that encoder's own embedding space** — a local prior beside the
  method's global class-proportion prior (cf. LaplacianShot's Laplacian term). μ = 1 a priori;
  μ = 0 is the published method. The vote uses fixed evidence, not current assignments: on a
  synthetic check, voting on assignments collapsed to one class when neighbourhoods were
  uninformative, while voting on evidence gains on clean clusters and is neutral on uninformative
  ones. Every row reports `neighbour_purity` (share of neighbours with the same true class) so an
  encoder that groups by subject or device rather than activity is visible.

HALO's zero-shot scores come from its own learned text projection `p_text` — the head the pool arm
trains through; baselines use their native text heads or the training-bank ConSE bridge.

### How learnable the adaptation may be (decided 2026-09-23)

| level | what is learned | status |
|---|---|---|
| A | HALO's encoder and `p_text`, trained **through** the fixed method, unrolled | in — built |
| B | the method unrolled with a few learnable **constants** (temperature, λ, damping, μ), initialised at the values above and fitted for every frozen baseline too | in — not built |
| C | a fully learned adapter that reads the pool (ARM, ICRM, TPN, a set transformer) | **out** |

C is out because ARM is below plain training on real shifts and collapses with an empty pool,
ICRM plateaus after ~25 samples (it identifies the environment, not the classes), a fixed
transductive loss on good features beats meta-learned transductive modules, and every learned head
we put over the support vote collapsed or fell below its floor. B's parameters are constants, never
functions of the pool; once they depend on the pool it is C. **Fallback: PADDLE** (Martin et al. 2022,
arXiv:2210.14545) — hyperparameter-free, built for query classes drawn from a larger set than the
support set. Same first author; the swap costs nothing narratively.

A [deployment memory-reader proposal](../journal/2026-09-25-deployment-memory-reader-proposal.md)
with a [storage/token literature audit](../journal/2026-09-25-memory-reader-literature-audit.md)
and an [online-episode addendum](../journal/2026-09-25-memory-reader-online-episode-addendum.md)
record a separate HALO-specific classifier experiment for future implementation. It does not
replace this rung's registered shared inference procedure or change the current paper protocol.

Cited for properties, **not** run as arms: OSLO (open-set), α-TIM (imbalance), Burzer et al. 2026
MAP-EM (the HAR-native competitor; its stated limitation — assumes known active classes — is our
regime).

### HALO's contribution arm: trained through the procedure

The curriculum gains episodes that contain an **unlabelled pool** (alongside supports when the
episode has any), with the pool structured the way deployment pools are:

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
attention stack, gated text blend, corruption auxiliary, unenrolled calibration — is not used on
rung 1**, at training or at inference; it remains the rung-2 case-study vehicle. (An earlier draft
proposed a `ROLE_UNLABELED` token acting through v4's residual; withdrawn — it would give HALO a
private inference path.) Built as `halo-train --rung1-training --pool-size MAX_N --pool-mode transductive`
(zero-support query groups with a shared pool; affinity on by default:
`--pool-affinity-mu 1 --pool-affinity-knn 10`, so gradients also reach the encoder through its
neighbourhood geometry). `--pool-mode soft_kmeans` is an enrolled-only diagnostic, not a k=0
rung-1 arm. The rung-1 objective mixes transductive and inductive text cross-entropy; validation
selects checkpoints across N with the full inference solver.

Two N=0 checks, not one: (i) **architectural** — the same checkpoint with the pool removed must give
a bit-identical forward pass; (ii) **inductive floor** — the retrained encoder's own N=0 k-curve must
not fall below v4's encoder at any k. The sealed table remains the N=0 column for v4's encoder, not
for the new arm.

**End-to-end means unrolled.** The transductive method runs inside the training forward pass on the
episode's pool, and the loss backpropagates through it into the encoder — meta-learning the
deployment procedure (cf. Hu et al. 2020, arXiv:2004.12696). Exposure alone is not assumed to
suffice: Ochal et al. (IEEE TAI, 10.1109/tai.2023.3298303) show many meta-learners "will not
automatically learn to balance from exposure to imbalanced training tasks".

### Comparison structure

Three baseline tiers and a four-step HALO ladder in which each step changes one thing. Tier 1
needs no training; tier 2 needs the matched arms trained (≈ 15.5 GPU-h, a separate go).

| tier | baselines | isolates | new training |
|---|---|---|---|
| 1 | frozen released checkpoints + the transductive method | off-the-shelf reality | none |
| 2 | the corpus-matched arms (built 2026-09-22, trained with differentiable neighbours on our corpus) + the transductive method | "you just saw our corpus" | no new *design*; the arms are built but **unlaunched** — HARNet 32 min, LiMU-BERT ~40 min, UniMTS 13.6 h, NormWear ~40 min ≈ 15.5 GPU-h |
| 3 | HALO trained through the unrolled method | the claim | HALO only |
| B | every encoder of tiers 1–2 with the level-B constants fitted on our training corpus (encoder frozen) | whether learning the method's constants helps any encoder, separately from training through it | a few constants per encoder; not built |

HALO ladder, all from random initialisation on our corpus: (1) plain cross-entropy → (2) +
heterogeneity curriculum, **which is today's v4 encoder** → (3) zero-support grouped-pool training
with the shared unrolled transductive method → (4, proposed, not built) learned method constants.
Step 2→3 asks whether optimizing through the deployment method helps; step 3→4 would ask whether
learned constants help, matched by tier B. Exposure alone without an inference path is not a
meaningful k=0 training arm: the unlabeled pool cannot change a support-only prediction when no
supports exist. Optional tier 4: the unrolled loss dropped into the cheapest matched arm
(HARNet, 32 min) as a transfer check — not required for the claim.

**Invariant: inference is identical for everyone, HALO included.** Unrolling is training-time only.

### Registered threats and their matched controls

| threat | source | control |
|---|---|---|
| transductive gains are an artefact of balanced pools | Veilleux et al. 2022 (arXiv:2204.11181): drops "even below inductive" under Dirichlet marginals | balanced-pool arm; per-cell imbalance reported |
| exposure to heterogeneous pools does not by itself confer robustness | Ochal et al. | compare the inductive N=0 anchor with training through the transductive method |
| episodic training is unnecessary; plain CE suffices | Laenen & Bertinetto NeurIPS 2021; Burzer et al. validate for HAR; Zhang 2024 (arXiv:2402.00092). Counter: LibFewShot (TPAMI) | ladder step 1 |
| the model exploits the pool's acquisition fingerprint, not its class structure | — | same-config **disjoint-class** pool tests whether label overlap is required; it cannot alone establish domain adaptation |
| pseudo-label confirmation bias and cluster collapse | Wang et al. CVPR 2022 (10.1109/cvpr52688.2022.01424) | confidence threshold, per-class cap, **collapse rate reported** |
| encoder overfits the unrolled method and regresses inductively | — | N=0 check (ii): the retrained encoder's k-curve must not fall below v4's |
| the affinity vote spreads errors for an encoder that groups windows by subject or device rather than activity | synthetic check, 2026-09-24 | `neighbour_purity` reported per encoder and cell; the μ = 0 ablation (the published method) reported for every encoder |

### What is dropped, and why

- **The discovery rung** (roster and K unknown: cluster, RSA against label-text geometry, name the
  clusters). Dropped 2026-09-23 as not interesting enough. The code is retained, retired, under
  `evaluation/discovery/` (`halo-discovery`) and is not part of the paper.
- **Class-prior correction (SLD).** Esuli, Molinari & Sebastiani (ACM TOIS 2020, 10.1145/3433164):
  helps only with ≤5 classes and a calibrated classifier; otherwise "negative rather than positive".
  Our rosters are 6–8 and text-cosine is uncalibrated.
- **TENT.** Needs BatchNorm affines; not all six encoders have them — unfair by construction.
- **LaplacianShot** as primary. Balanced-benchmark era; superseded for our regime. Its idea — a
  pairwise term over feature-space neighbours — is what the affinity term borrows.
- **`ROLE_UNLABELED`.** A private HALO inference path; see above.
- **"Clustering with known K".** With the roster known it is a worse version of zero-shot; the only
  justification is transduction, which is what rung 1 measures.
- **Seeded k-means as a refinement step.** The unrolled transductive method subsumes it.

## Rung 2 — labels, parameters frozen: the case study

*Deployment provides k labelled examples per class; the model may not change. This is the existing
work: v4's 333 sealed cells and 737 deployment-scenario cells, the learned classifier, the k-curve,
and the per-scenario results against the baselines under equal-weight normalised fusion.*

It is demoted from headline to case study, and it is written as an **encoder** result with v4 as
the vehicle, because that is what the numbers show (sealed aggregate, 8 s, dataset-balanced
macro-F1, [RESULTS.md](../results/RESULTS.md)):

| readout | k=1 | k=8 | k=32 | k=128 |
|---|---:|---:|---:|---:|
| HALO v4 learned classifier | 62.4 | 72.7 | 75.8 | 76.5 |
| HALO v4 parameter-free support vote (the head's own floor) | 60.5 | 72.4 | 76.1 | 77.4 |
| HALO v4 encoder, cosine 1-NN, no learned head | 60.8 | 71.7 | 74.8 | 76.3 |
| best baseline (UniMTS), same 1-NN | 53.9 | 66.9 | 71.5 | 73.9 |

With no learned head, the HALO encoder beats every baseline under an identical parameter-free
readout at every k. The learned head earns its keep at k ∈ {0, 1} — the text term is what makes
k=0 work (50.2 vs 38.6 for the bank bridge) — and trails its own floor at k ≥ 32. So the rung-2
claim is the encoder's; the head's design history (T1–T8) is a section on how the enrollment arm was
built, not the claim. The frozen-enrollment arm on rung 3's crossover is represented by the
parameter-free vote.

The scenario taxonomy (737 cells of acquisition mismatch) is reused as the axis along which rung 1
draws its unlabelled pools.

## Rung 3 — labels, fine-tuning allowed

*Deployment provides k labelled examples per class; every model may fine-tune. The headline
adaptation result, and where the "just retrain" objection is answered head-on.*

Ladder by cost, every treatment applied to every model including HALO: linear probe on frozen
features (closed form; TransfHAR's mechanism) → small classifier (the frozen-projection path built
2026-09-22) → LoRA / full fine-tune → a specialist trained from scratch on the same k windows.
Scored on rung 1's execution split, so the crossover is on identical windows. Built as
`halo-rung3`.

Priority, in order:

1. **The must:** HALO fine-tuned ≥ every baseline fine-tuned, at every matched k.
2. **The absolute claim:** the **crossover** — HALO fine-tuned on k windows against a specialist
   trained from scratch on the same k windows; the k at which the curves cross is a number the
   protocol cannot make uninterpretable.
3. **Context, not pass/fail:** where we land against ~77–80 % (DAGHAR LODO, fully supervised,
   harmonised, ~5 classes, 225 subjects). Our protocol is six unharmonised datasets at ~8.7 classes
   scored by macro-F1, so an absolute target on it is not apples-to-apples; committing to one would
   manufacture a miss.

Build cost: NormWear has no fine-tuning path (no trunk under the encoder contract; its authors never
fine-tune) and is declared unsupported rather than approximated; HARNet-10 has no trunk under the
contract and gets the cached-feature treatments only.

## Registered predictions

1. **Rung 1, established method on all six:** modest gains (a few points), **negative on some cells
   for some encoders** — going negative is the norm in this literature (Burzer's baselines: −12.8
   pp). The curve rises steeply and plateaus within a few hundred windows per class.
2. **Rung 1, HALO unrolled arm:** a reliably non-negative, steeper curve than any baseline under the
   same procedure. If the established method is flat on all six, the arm has nothing to improve on
   and is not run.
3. **Rung 2:** no prediction — done.
4. **Rung 3:** HALO's fine-tuned curve matches or exceeds every baseline's at every k; the crossover
   sits at a k small enough to matter (registered qualitatively, not as a number).

## Sequencing and gates

1. **This document and the journal entries** — done 2026-09-22/24.
2. **Build rung 1 and rung 3** — **done 2026-09-23**, with the pool training arm, per the
   [implementation plan](../journal/2026-09-22-rung1-rung2-implementation-plan.md) (old numbering);
   the embedding-affinity term added 2026-09-24. Level B is not built.
3. **Debug sweep** — done 2026-09-25, all findings fixed
   ([record](../journal/2026-09-25-debug-sweep-and-fixes.md)). Still open: the extraction golden,
   and a smoke on one cached cell.
4. **Run rung 1, tier 1** on go. CPU, ~hours. Tier 2 after the matched arms are trained (separate go).
5. **Gate:** run the HALO unrolled arm only if prediction 1's curve rises for at least one encoder.
6. **Rung 3** fine-tuning. The expensive part; last in sequence, not least in weight.
7. **v5, after rung 1** (decided 2026-09-25). v5 is a separate HALO-only experiment, not part of
   the rung-1 comparison. It is trained and evaluated only once rung 1's EM-Dirichlet + affinity
   curve exists, and is then compared against that method, its own fixed-vote floor and
   no-memory control, and v4 under matched verified enrollments, on the same streams and with the
   same information at each step. That needs an online predict-then-update evaluator, not yet
   built.

## Open decisions

- Rung 1 on the sealed six only, or MM-Fit as well.
- Pool-mean centring as a rung-1 option (the `corpus_mean` knob; not wired).
- An embodied evaluation source (Ego-Exo4D IMU) so the headline framing can be scored.
- Launching the corpus-matched arms (≈ 15.5 GPU-h) so tier 2 exists.
- Building level B (the learnable constants and tier B).

## Code organisation

The separate **v5 online-memory experiment** is implemented in
[`model/support/memory_classifier.py`](../../model/support/memory_classifier.py),
[`training/support_classifier/memory_bank.py`](../../training/support_classifier/memory_bank.py),
[`training/support_classifier/memory_episodes.py`](../../training/support_classifier/memory_episodes.py),
and [`train_memory.py`](../../training/support_classifier/train_memory.py). It predicts a stream
causally and may retain earlier unlabelled predictions or verified enrollments. It is not a
registered rung-1 readout: rung 1 remains the common fixed-pool transductive comparison. The
[implementation record](../journal/2026-09-25-v5-online-memory-implementation.md) states what is
built, smoke-tested, and still awaiting a dedicated online evaluation protocol. It comes after
rung 1 (step 7 of the sequencing below).

*Built 2026-09-23; rung 2's runners moved in on 2026-09-24.* Rung 2 is the working pipeline that
produced every published number, so the rule was **extract, don't rewrite**, verified byte-exact
against the anchor tag at the source level (the functional golden is still to be produced).

**Principle: separate by concern, not by rung.** All rungs share the six encoders, the feature
caches, the sealed manifests, the metrics, and provenance. What differs is only the readout on top of
cached features (rungs 1–2) or the treatment applied before it (rung 3).

```text
evaluation/                      consumes checkpoints + manifests, produces artifacts
  features.py                    ← from sealed_eval (verbatim): feature extraction and caches
  manifests.py                   ← from sealed_eval (verbatim): rosters and episode manifests
  zero_shot.py                   ← from sealed_eval (verbatim) + zero_shot_scores, probability_features, ProviderScorer
  metrics.py                     one definition each; classification delegates to baselines.scoring
  provenance.py                  ← from sealed_eval (verbatim) + the rung / method / readout-version registry
  rung1_unlabeled/               transductive.py (the one method; also imported by the trainer), ncurve.py, run.py
  rung2_frozen/                  sealed_eval.py, run_scenarios.py, run_partial_coverage.py (moved 2026-09-24; three-line
                                 aliases remain at training/support_classifier/ for recorded commands)
  rung3_finetune/                lora.py, finetune.py, run.py
  controls/                      balanced_pool.py, disjoint_classes.py
  discovery/                     RETIRED 2026-09-23 (was rung 1 of the four-rung plan); kept runnable, not part of the paper
training/support_classifier/     train.py (+ the pool arm), sampling.py (+ attach_pool), scenarios.py, partial_coverage.py,
                                 neighbors.py (model/ imports it)
training/tokenizer/              shared encoder infrastructure (pretrain_data.py, eval_transfer.build_encoder) + the
                                 retired Future-JEPA entry point; see its README
results/tools/                   reporting only
```

Rules that make "which version is which" unambiguous:

1. **Every artifact declares its rung and method.** `evaluation/provenance.py` holds a registry
   (`RUNG ∈ {1, 2, 3}`; `DISCOVERY = 0` retired; `METHOD` enum; `READOUT_VERSION` such as `ncurve-v2`,
   `finetune-v1`). An artifact cannot be written without them.
2. **Readout versions bump when the procedure changes**, as `sealed-manifest-v2` and
   `deployment-scenarios-v5` do today.
3. **Classifier names are unchanged**: v4 / T-numbers per RESULTS.md. Rung 1 uses no learned head,
   so it names the encoder checkpoint, not a classifier.
4. **The transductive method lives in one file** and is imported by both the evaluator and the
   trainer's unrolled loop, so the method unrolled in training is provably the one applied to the
   baselines at test.
5. **Case studies and other experiments** are drivers under the rung they read from, never new
   top-level trees.
