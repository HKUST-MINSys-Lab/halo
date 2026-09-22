# The three-regime plan, and how the unsupervised tier was redesigned twice in one day

Date: 2026-09-22 (evening). Status: **decision record + literature record; nothing run.** This
entry supersedes the Stage U section (U-0a/b/c, U1, U2) of
[the morning's pivot entry](2026-09-22-pivot-unsupervised-then-finetune-and-framing.md); the rest
of that entry stands. The live plan is [`docs/overview/roadmap.md`](../overview/roadmap.md); the
live thesis is [`docs/overview/thesis.md`](../overview/thesis.md). This entry records what was
discussed, what the literature said, what was retracted, and why each decision was made.

## 1. What was discussed, in order

1. **"How does U-2 work?"** U-2 had been registered as seeded k-means from the enrolled windows. The
   question exposed that a support-conditioned *classifier* has no obvious clustering
   interpretation. The answer found: the head's support set is just vectors with candidate bindings,
   so pseudo-labelled pool windows can be supports — the head is its own EM step. And `corpus_mean`
   centring, already plumbed in
   [`residual_classifier.py`](../../model/support/residual_classifier.py), is a one-step
   transductive method.
2. **The objection that killed U-0b.** If the roster is known, "clustering with known K" is a worse
   version of zero-shot classification, and per-window zero-shot is *stateless*: no amount of
   unlabelled deployment data changes it. Alex's point exactly: "the model never gets better." The
   reframe: the only reason to look at the pool is **transduction** — priors, deployment geometry,
   and boundaries that respect the data's own clumps — and the experiment is therefore a **curve in
   N (unlabelled), not k (labelled)**. Every table we had swept k. None swept N.
3. **Alex's three regimes**, which became the paper's structure: (1) discovery — roster and K
   unknown, two metrics: same-label-same-cluster, and centroid geometry mirroring label-text
   geometry; (2) known roster, growing unlabelled pool, no labels ever — "extremely important
   because it makes this whole adaptation idea actually worth something"; (3) labelled — the k-curve
   demoted to a case study, and fine-tuning for everyone against the ~77–80 % bar.
4. **A thin literature search, caught.** I ran three web searches, one scite call and one Exa call
   (Perplexity failed, out of quota) and recommended LaplacianShot as the regime-2 method. Alex asked
   "did you use Consensus and scite and alphaXiv?" I had not. The full five-tool search overturned the
   recommendation (§2). **New standing rule: every literature search uses scite, Exa, WebSearch,
   Consensus and alphaXiv; never Perplexity.**
5. **The curriculum idea.** Alex: the k-curve's novelty is not the mechanism but the training
   curriculum that mirrors deployment heterogeneity, so the model "comes with adaptation as a
   primary objective". Can the same be done for unlabelled adaptation — episodes with unlabelled
   pools the model must learn to use? Search: yes in vision since 2018 (§2.4); not on the
   acquisition axis; and the literature explicitly says it is needed (§2.5).
6. **"How many methods are we using? I think only two."** Correct; I had listed PADDLE, OSLO,
   α-TIM and Burzer as arms in two consecutive answers. Back to one established method per regime,
   everything else cited for properties.
7. **Build our own unsupervised method, or reuse SOTA and innovate in training?** Reuse at
   inference, innovate in training, and "end-to-end" means the inference method is **unrolled inside
   training** so the encoder is optimised for it (§3).

## 2. What the literature said

Tools: scite (Smart Citations), Consensus, alphaXiv (flaky — three "Connection closed" failures,
then worked), Exa, WebSearch. Per-tool attribution is in the transcript; the findings:

### 2.1 Regime 1 has an off-the-shelf protocol

Lowe et al., *An Empirical Study into Clustering of Unseen Datasets with Self-Supervised Encoders*
(TMLR 2024, arXiv:2406.02465): frozen encoders deployed on unseen datasets, conventional clustering,
encoders compared. Findings we inherit: SSL encoders win far-OOD, supervised win in-domain;
**silhouette in UMAP space tracks clustering quality without labels** — a deployable selection
signal. k-means++ ×10 with Hungarian matching is the accepted default (also *Beyond Supervised vs.
Unsupervised*, arXiv:2206.08347).

For metric 2: Representational Similarity Analysis (Kriegeskorte 2008; Nili 2014) is built to
compare representations "acting on the same stimuli" across "an unrelated medium"; Dwivedi & Roig
(CVPR 2019, 10.1109/cvpr.2019.01267) use it to compare pretrained models with no training. Caveat
registered: Bertram et al. 2026 (arXiv:2605.05907) show alignment metrics can be driven by a small,
non-representative subset of dimensions.

### 2.2 Regime 2: the method families, and when they fail

Three families that get conflated: transductive inference (TIM/α-TIM, LaplacianShot, PADDLE, OSLO,
transductive-CLIP, GATE, TPA — nothing in the encoder updates), source-free test-time adaptation
(SHOT into the encoder; TENT into BatchNorm; T3A prototypes only), and closed-form prototype
refinement (Burzer MAP-EM, PDA, soft k-means). Nearly all reduce to: recentre by pool statistics,
EM between assignments and prototypes, a regulariser on the marginal.

**They work when** the shift is a global offset (Burzer: one EM step plus centring gives +32 pp on
HARTH), the starting point is decent, the pool is balanced, the pool's classes match the roster,
and the shift is mild. **They fail** under severe shift (T3A by its authors' account), poor starting
points (Burzer's baselines: PDA −12.81 pp, OFTTA −3.57 — *going negative is the norm*), imbalanced
pools (Veilleux et al. 2022, arXiv:2204.11181: "substantial performance drops, even below inductive
methods"; Ochal et al., IEEE TAI 10.1109/tai.2023.3298303: up to 17 %), partial or unknown coverage
(Burzer's stated limitation), and uncalibrated scores.

**Prior correction is dead for us.** Esuli, Molinari & Sebastiani (ACM TOIS 2020, 10.1145/3433164)
re-examined SLD: improves posteriors "only when the number of classes is very small and the classifier
is calibrated" (|Y| ≤ 5); beyond that "performance degrades rapidly, and the impact ... becomes
negative rather than positive". Rosters are 6–8; text-cosine is uncalibrated.

**The method chosen:** Transductive Zero-Shot and Few-Shot CLIP (Martin et al., CVPR 2024,
10.1109/cvpr52733.2024.02722; ~20 % ImageNet gain over zero-shot from batches of 75), provisional on
unrollability; fallback PADDLE (Martin 2022, arXiv:2210.14545; hyperparameter-free; query classes
drawn from a larger set — our partial coverage). OSLO (CVPR 2023, 10.1109/cvpr52729.2023.02299) and
α-TIM cited for open-set and imbalance. α-TIM applied to satellite time series (Mohammadi et al.,
*Remote Sensing* 2024, 10.3390/rs16061026) shows the Dirichlet-realistic protocol already travels
outside vision.

### 2.3 The HAR-native competitor

Burzer, Riedel, Beigl, Röddiger, *Uncertainty-Aware (Un)Supervised Few-Shot User Adaptation for
On-Device Personalized HAR* (June 2026, arXiv:2606.04798). Gradient-free; repurposes pretrained HAR
classifiers as Prototypical Networks with prior prototypes; supervised = closed-form Bayesian
prototype update, unsupervised = MAP-EM over latent classes; **+0.56 to +32.13 pp unsupervised**.
Same-dataset LOSO on TinierHAR, four datasets, known active classes — an easier shift than ours.
Two things matter: they centre by the unlabelled support mean and say why ("removes global
subject-specific offsets ... rather than compensating for a global shift that can be easily
corrected by centering") — independent confirmation that pool centring is load-bearing; and their
stated limitation is our regime: "MAP-EM currently assumes prior knowledge of which activity classes
are present ... Future work should infer active classes directly from the calibration data."

### 2.4 "Unlabelled data in training episodes" is a 2018 idea

Ren et al., *Meta-Learning for Semi-Supervised Few-Shot Classification* (2018, arXiv:1803.00676,
1,455 citations): unlabelled examples "within each episode", with and without distractor classes,
Prototypical Networks "trained in an end-to-end way on episodes, to learn to leverage the unlabeled
examples". A mature line follows: Learning to Self-Train (arXiv:1906.00562), TransMatch (CVPR 2020),
PLATINUM, TACO (AAAI 2021), Task-Adaptive Clustering (arXiv:2003.08221), Empirical-Bayes
Transductive Meta-Learning (arXiv:2004.12696). **The mechanism is not a contribution.** Every one of
them draws the pool from the same acquisition as the supports; their hard case is distractor
*classes*. None varies device, placement or rate. That is the slice our corpus uniquely supports.

### 2.5 The literature says it is needed

- Cao et al. (arXiv:1909.11722): "the shot number used in meta-training **should match** the one
  used in meta-testing" — theory for Prototypical Networks. *A citation for why the k-curve
  curriculum matters, which we did not previously have.*
- Han et al., *Meta-learning for few-shot open task recognition* (*Sci Rep* 2026,
  10.1038/s41598-026-36291-x): "the target configuration is unknown at training time ... a
  **structural shift between training and test episodes**"; open-task evaluation is "a necessary
  complement". They formalise cross-way/cross-shot; nobody formalises cross-acquisition.
- Ochal et al.: episodes are sampled "to mimic tasks seen during evaluation" but "standard training
  procedures overlook the real-world dynamics" — and the warning: many meta-learners "will **not**
  automatically learn to balance from exposure to imbalanced training tasks".
- Counter-current, to be engaged: Laenen & Bertinetto (NeurIPS 2021) argue plain cross-entropy
  suffices; Burzer et al. validate that for HAR; Zhang 2024 (arXiv:2402.00092) challenges "the
  principle that training conditions must match testing conditions". LibFewShot (TPAMI,
  10.1109/tpami.2023.3312125) finds episodic training still necessary with pretraining. A live
  controversy; the plain-CE control is mandatory.

## 3. What was decided, and why

1. **Three regimes, one established method each, applied identically to all six encoders.**
   Attribution: if the inference method is identical and only the encoder differs, the gain is the
   encoder's. This is the k-curve's logic (parameter-free vote for everyone) carried over.
2. **Regime 2 is a curve in N, not a clustering metric.** Flat is the null and is Alex's objection
   made quantitative. Rising is the paper.
3. **Reuse the established method; innovate in training; "end-to-end" = unrolled.** The inference
   side is saturated (six methods 2022–2026); the training side is the gap the literature names. Two
   versions exist — curriculum-only (Ren 2018) and unrolling the method in the forward pass — and only
   the second answers Ochal's finding that exposure alone may not suffice. The same implementation is
   used at test for all six, so the claim is: *given identical test-time adaptation, an encoder
   trained through that adaptation beats encoders that were not.*
4. **Mechanism for the pool: `ROLE_UNLABELED`.** Unlabelled tokens have no candidate binding, so they
   cannot enter `differentiable_neighbor_logits`; they act only through the residual. Hence N=0 is
   bit-identical to the current model and `residual_enabled=False` is a provable zero — the ablation
   is clean by construction. Pool summarised to a few tokens to keep attention cost flat.
5. **The novelty claim is framed as a citation, not an assertion:** *the literature says episodes
   should mirror deployment, several papers name it the open problem, and nobody has done it on the
   acquisition axis.* We cite them saying it.
6. **Metric 2 of regime 1 is RSA**, because it is projection-free and therefore fair to the two
   encoders without a text head; naming accuracy is secondary and its ConSE dependence disclosed.
7. **k-curve demoted to a case study; fine-tuning for everyone is the headline of regime 3.**

## 4. Retracted or corrected today

Recorded so the paper never repeats them.

- **LaplacianShot as the regime-2 primary** (this afternoon) — based on a thin search; withdrawn for
  transductive-CLIP / PADDLE.
- **Class-prior correction as a gain source** (this afternoon) — withdrawn on Esuli et al.
- **"Transduction helps" asserted without naming what it extracts** — replaced by the three named
  sources (priors, geometry, boundaries), of which priors then died.
- **"Nobody clusters FM embeddings for HAR"** — retracted this morning (Takatsu et al. 2025 et al.);
  stands retracted.
- **U-0a / U-0b / U-0c / U1 / U2 as registered** — superseded by regimes 1 and 2 above. Zhong 2006's
  seeded-vs-constrained distinction, and the seeded-k-means U-2, are moot.
- **The unlabelled-episode curriculum as novel** — it is Ren 2018; only its acquisition-axis
  instantiation is unclaimed.

## 5. Housekeeping done with this entry

- `docs/overview/thesis.md` and `docs/overview/roadmap.md` rewritten for the three regimes; the
  classifier-era versions archived under `docs/archive/` with supersession notices.
- `docs/overview/architecture.md` and `README.md` corrected: v4 is promoted (they still said v3 and
  named evidence-aware v2 as the active experiment).
- A code-organisation plan is in the roadmap: `evaluation/` package, shared modules extracted from
  `sealed_eval.py` bit-exactly before regimes 1 and 2 are built on them, regime/method/readout-version
  registry in provenance, one implementation of the transductive method shared by trainer and
  evaluator. **No code moved yet.**

## 6. Open

- Sealed six only, or MM-Fit too, for regimes 1 and 2.
- Curriculum arm now, or after the established arm shows a rising curve (recommendation: after).
- An embodied evaluation source (Ego-Exo4D IMU).
- Alex's message cut off at "Our learnable classifier" — thought not yet captured.
