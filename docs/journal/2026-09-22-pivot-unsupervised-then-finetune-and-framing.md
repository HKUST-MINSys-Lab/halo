# The pivot: unsupervised first, fine-tuning second — and how the paper is framed

Date: 2026-09-22. Status: **decision + research record; no experiment run.** Written after a
supervisor meeting and a long design conversation, so the reasoning is registered before results.

## 1. The objection that started this

A closed-set, domain-specific model reaches ~99% in its domain. Our standing argument — "domain
models fall apart under open labels or unseen heterogeneity" — is weak, because the rebuttal is
trivial: retrain for the deployment. And few-shot enrollment also needs labelled deployment data.
So what is the purpose of what we built?

Two honest answers, in order of strength:

1. **The 99% is a within-distribution, often leaked, number.** Same subjects, one device,
   frequently random splits over overlapping windows (our own audit found this in `uci_har`,
   `sp_sw_har`, `unimib`). Cross-subject closed-set on real benchmarks is typically 80–92%. Our
   sealed 72.7 is cross-subject, manifest-controlled, six unseen datasets. Different protocol.
   The literature ceilings are tabulated with citations in
   [`docs/related_work/2026-09-22-published-ceilings-by-regime.md`](../related_work/2026-09-22-published-ceilings-by-regime.md).
   Headline: the 99% UCI-HAR figure is 10-fold CV over 50%-overlapping windows (Mekruksavanich &
   Jitpattanakul 2021, Table 8); the same model under LOSO is 96.7 / 93.9 and the dataset's own
   subject-disjoint baseline is 96 (Anguita 2013). Honest cross-subject ceilings: UCI-HAR ~93 mF1,
   MotionSense ~89, RealWorld ~85, PAMAP2 ~85–86, USC-HAD ~61, Opportunity 45–69, WISDM-19 ~32;
   two 2026 multi-dataset benchmarks plateau at 61–68 mean mF1. Placement/device shift costs
   more than subject shift. The genuine unseen-dataset zero-shot ceiling is UniMTS 34.3 mF1 ≈
   AnyMo 29.5; every higher number relaxes an "unseen" axis.
2. **What HALO replaces is not the specialist, it is the pipeline that produces specialists.**
   Enrollment at k=8 is a minute of "show me walking" by the end user; a specialist needs
   hundreds of windows per class across subjects plus an engineer and a training loop, and
   HARNet's own paper shows from-scratch loses to a random forest on 6/7 datasets when data is
   small. Heterogeneity is combinatorial (device × placement × rate × subject × vocabulary), so
   "just retrain" is a per-instance pipeline, not a one-time cost.

But neither answer has been **measured**. The experiment the objection actually demands — same k
labelled windows per class, enrollment versus a specialist trained on them, giving a
data-efficiency crossover — has never been in a table. That is the fine-tuning stage below.

## 2. The regime, named: arbitrary activity detection

The label vocabulary is defined **after** deployment, by whoever is using the device, not by
whoever trained the model. Value is proportional to how far three things are from the model
developer's control: the vocabulary (user / operator defines it), the acquisition (bring your own
device), and the labelled data (rare, new, private, or uneconomic to collect). When all three are
developer-controlled — a phone vendor's fixed 20-class step counter — a specialist wins and the
paper should say so. That concession is what makes the rest credible.

**Framing rules, from the supervisor:** no clinical or health framing. Relying on a foundation
model instead of collecting disease-specific data is frowned upon, and the reviewer's first
question in a health frame ("why didn't you collect disease data?") has no good answer. So:
rehab, MoniPar, KneE-PAD, trial endpoints, "patient", "digital biomarker" — **out of the headline.**
InclusiveHAR stays as a *robustness* row ("users whose movement differs from the training
population"), never as an accessibility-health claim.

**Framings that survive, ranked by how well our data supports them:**

| framing | why the label is unknown in advance | our data |
|---|---|---|
| Embodied AI / learning from demonstration | operator defines the task's primitives that day | Ego-Exo4D, Nymeria in the pretraining plan; **need an eval source** |
| Industrial — OpenPack | steps change per SKU and site; supervisor demonstrates | Task-1 infrastructure exists |
| End-user-defined gestures and routines | shipped consumer feature; user teaches by demonstration | the sealed six as robustness |
| Fitness — MM-Fit | every trainer invents movements | already our foreign-vocabulary test → becomes the flagship |
| Annotation acceleration | cluster, name a few, propagate — the community's own bottleneck | any source |

Ubicomp has a lineage for this — end-user programming of sensor recognisers, "teach the device by
demonstration" — that IMWUT reviewers accept without a clinical justification.

## 3. The staged plan (supervisor's, adopted)

**Stage U — unsupervised, on the open-set / multi-subject / multi-device regime.** Feature caches
for HALO v4 and all five baselines already cover every sealed stream, so this is CPU work on
existing artifacts.

* U-0a — kNN purity / R@k: do a window's nearest neighbours share its activity? No algorithm, no
  K, no seed, no labels at query time. Added 2026-09-22 so the headline geometry claim does not
  rest on a clustering algorithm's assumptions.
* U-0b — cluster with **known K** (= candidate roster): spherical k-means, 10 seeds; **AMI, ARI and
  Hungarian/clustering accuracy** (see metric note); agglomerative as a second algorithm so nothing
  is a k-means artefact.
* U-0c — cluster with **estimated K** (silhouette / eigengap). The deployment-realistic case, and
  where the AMI-vs-NMI choice actually bites, since K then varies between models.
* U1 — name the clusters: centroid → the model's text bridge → argmax over the roster.
  Cluster-naming accuracy and end-to-end unsupervised accuracy. Open-vocabulary variant over the
  155 training labels + roster is the harder, more honest version. Disclose that HARNet and
  LiMU-BERT-X are named through *our* ConSE bridge.
* U2 — refine as labels arrive: seed with the k windows the manifests already enrol,
  k ∈ {0,1,2,4,8,16,32}; **seeded k-means (not constrained k-means — see below)**, label
  propagation over the kNN graph, and HALO's parameter-free vote.

### Design review against the literature (2026-09-22) — three corrections

**1. Declare subject-independent clustering explicitly; it is the dominant design choice.**
Mahon & Lukasiewicz (*Efficient Deep Clustering of Human Activities and How to Improve Evaluation*,
arXiv:2209.08335) identify this as the central ambiguity in the field — whether each subject's data
is clustered separately (subject-dependent) or all subjects together (subject-independent) — and
show the *same model* swings enormously between them: WISDM-watch ACC 78.40 → 25.58, ARI 72.22 →
12.60; REALDISP ACC 89.60 → 51.37; PAMAP 66.28 → 48.30. Their Table 1 shows almost every prior HAR
clustering paper is ambiguous about which it used. **We cluster subject-independently** — harder, and
the only setting consistent with the deployment framing. Stated here so it can never be ambiguous.
They also test window-wise vs point-wise labelling and find "essentially no diﬀerence", so our
window-level evaluation needs no further justification beyond being declared.

**2. NMI is a biased metric; use AMI and ARI.** Jerdee et al. (*Nature Communications*,
10.1038/s41467-025-66150-8) and Mahmoudi et al. (*Scientific Reports*,
10.1038/s41598-024-59073-9) both show NMI is biased with respect to the number of clusters — the
latter proving it formally and calling NMI "unsuitable ... for evaluating clustering". Jerdee et al.
show the bias changes *which algorithm looks best*. Romano et al. (*JMLR* 2015) give the operational
rule: **ARI when the reference clustering has large equal-sized clusters, AMI when it is unbalanced
with small clusters.** Our sealed rosters are imbalanced, so AMI is the right primary, with ARI
alongside. NMI may be reported for comparability with the HAR clustering literature (which uses it
almost universally) but never as the deciding number, and always with the bias disclosed.

**3. Expected absolute numbers are far lower than the sealed table would suggest.** In the
subject-independent setting, Mahon & Lukasiewicz's *trained deep clustering model* reaches ACC
25.6–51.4, NMI 28.4–71.8, ARI 12.6–43.9 across six datasets. We are running k-means over frozen
embeddings with no training at all, on held-out datasets. Low absolute numbers are the norm here,
not a failure signal — the claim is *relative* (does HALO's geometry cluster better than the
baselines'), and the registered success criterion is a margin, not a threshold.

**Why seeded k-means and not constrained k-means for U2.** Zhong (*Machine Learning* 2006,
10.1007/s10994-006-6540-7) compares seeded, constrained and feedback-based variants and finds "the
constrained approach is the best when available labels are complete whereas the feedback-based
approach excels when available labels are incomplete." Our partial-coverage regime has *incomplete*
label sets by construction — some roster entries are enrolled, others deliberately are not — so
constrained k-means is the wrong tool: it would force every seeded point to stay in its seed's
cluster and leave no mechanism for an unenrolled activity to form its own. Seeded k-means uses the
labelled windows only to initialise centroids and then lets assignment proceed freely, which is
what an open-set setting requires. Basu, Banerjee & Mooney (2002) is the canonical reference;
González-Almagro et al. (*Artificial Intelligence Review* 2025, 10.1007/s10462-024-11103-8) is the
315-method taxonomy with a pitfalls section to check against before implementing.

**U2 is the missing left end of a curve we have already published.** At k=0 it is U-0b; at k≥1 with
HALO's own parameter-free vote it should reproduce our published sealed k-curve exactly. That makes
it both a new result and a self-consistency check on the existing table.

Registered caveats: known K is generous (hence U-0c); clustering metrics are imbalance-sensitive
(report per dataset); this measures the representation, not the classifier, which is the right
object here; feature dimensionality varies 72-d to 4608-d across models, so report raw plus a
PCA-to-64-d robustness check.

**Stage F — fine-tuning on, same k-window budgets, every treatment applied to HALO too.**
Linear probe on frozen features (closed form; NormWear's own protocol; TransfHAR's mechanism —
see §4); small classifier (the frozen-projection path built 2026-09-22, fitted on the cell's k
windows as a specialist and on our corpus as transfer); LoRA / full fine-tune of the encoder on
the cell's k windows — the crossover experiment. HALO's own-encoder fine-tune path does not exist
yet; three baselines have `MatchedCorpusEncoder`; NormWear has none and its authors never fine-tune.

Registered prediction: HALO leads U0 on the sealed six by roughly its 1-NN margin; U1 naming is
where the gap opens; MM-Fit at k=0 is where we lose. If U0 is flat against UniMTS, the "more useful
unsupervised" claim is dead before Stage F.

**Rule:** Stage U is a new readout on sealed data, so it is pre-registered before it is read, and
run only on explicit go.

## 4. What the literature sweep found (2026-09-22; alphaXiv, scite, Consensus, arXiv)

### The direct competitor for the framing: TransfHAR (Bradshaw, Arakawa, Liu, Ahuja — Northwestern / CMU / Google; arXiv 2608.15861, Aug 2026)

*"users define and expand their own activity set for personalized recognition from only a few
demonstrations"* — this is our regime, stated by Google and Northwestern, shipped as an Apple
Watch app. Details that matter:

* MAE-pretrained 1D-ViT (12 layers, 384-d) on eight public datasets, 2,760 h, 381 subjects;
  wrist only; 50 Hz; **2.56 s windows**; 3- or 6-axis.
* **Adaptation is a frozen encoder plus a single linear layer (384×C+C) trained on-device in
  under a second.** Not enrollment by example; a probe.
* Few-shot on SAMoSA (27 kitchen/workshop classes) 64.9% BA vs 56.8 supervised; PrISM (19
  procedural steps) 36.5 vs 31.0; UTD-MHAD (21) 69.1 vs 64.1. In-lab, 10 participants × 7
  self-chosen activities from a bank of 55: **75.2% (1 shot) / 86.7% (5 shots) / 90.4% (one
  1-minute recording per class)**, cross-session.
* **Compares to no foundation model** — not UniMTS, HARNet, LiMU-BERT, or NormWear. Only
  task-specific supervised baselines and architecture-matched from-scratch/fine-tuned ViTs.
* Stated limitation: window-based, no long temporal structure; untested across body types.
* Venue: UIST 2026 (per secondary listings). **No code or weights located** as of 2026-09-22; the one
  GitHub link a search surfaced is a 404.

What this means for us. It validates the on-demand framing from outside. It is also the cleanest
possible Stage-F "linear probe" treatment to adopt verbatim. Our differentiators against it are
exactly the things it does not do: any device / placement / rate rather than one wrist; enrollment
with **no training at all**, not even a probe; open-set "none of the above"; and naming through
text. If its weights are released it belongs in the baseline table; if not, its probe protocol
does.

### The zero-shot competitor we are not comparing against: AnyMo (Chen … Salim — UNSW / HKUST-GZ; arXiv 2605.22715, May 2026)

* Pretrained **entirely on synthetic IMU** from physics simulation over dense body-surface
  placements; real Nymeria streams only for noise priors and sim-to-real evaluation. 60 Hz,
  6-axis, 5 s windows, 23 anatomical segments, LLM-aligned tokens.
* Zero-shot on 14 unseen datasets — including **OpenPack, EgoExo4D, Ego4D**, RealWorld, USC-HAD,
  WISDM, DSADS, PAMAP2, UCI-HAR, UTD-MHAD, Opportunity: AnyMo 35.7 Acc / 29.5 F1 / 57.5 R@2
  against UniMTS 31.9 / 26.4 / 46.9, NormWear 8.0 / 2.2 / 16.6, ImageBind 13.4 / 8.8, IMU2CLIP
  18.0 / 13.0, HARGPT 10.2 / 5.5, Gemma-4-26B ~19 / 11.
* **No few-shot, fine-tuning, linear-probe or clustering evaluation.** Code is released
  (`github.com/Breezelled/AnyMo`, MIT) with an AnyMo-Bench dataset on HuggingFace, but **no
  pretrained checkpoints** and no standalone inference entry point — verified 2026-09-22. Reproducing
  its zero-shot row means retraining from its synthetic pipeline.

What this means. It beats UniMTS zero-shot, it evaluates on our embodied and occupational targets,
and it is from a neighbouring lab. It should be a zero-shot baseline if weights are public. And its
absence of any adaptation or unsupervised row is the gap Stages U and F fill.

### Everything else worth carrying

* ~~**Nobody in this sweep evaluates foundation-model embeddings by unsupervised clustering for
  HAR**~~ — **corrected 2026-09-22, this was wrong.** A targeted search found an established HAR
  deep-clustering line: Abedin et al. (ISWC 2020), Ahmed et al. (ISWC 2022, adapting SCAN to HAR),
  Mahon & Lukasiewicz (2022), Sheng et al. (2023), Amrani et al. (ICCE-Berlin 2022), and a 2020
  *Sensors* review (10.3390/s20092702). Most damagingly for the original claim, **Takatsu et al.**
  (*IEEE Access* 2025, 10.1109/access.2025.3562897) run cross-dataset pretrained representations →
  unsupervised deep clustering → fine-tuning with 50 samples, i.e. essentially our U-0 → U-2
  pipeline, reporting F1 0.441–0.781 clustering and 0.66–0.88 after 50 labels. What remains
  defensible as novel is narrower and should be stated that way: a *head-to-head comparison of
  released foundation models as frozen feature extractors* on held-out datasets (rather than
  proposing another clustering method), and the text-bridge **naming** step (U1), for which no
  counterpart was found.
* **Nobody reports an enrollment-versus-fine-tuning crossover at matched label budgets.** This
  claim survived the scite digest — FSID (Sci Rep 2025) has the matched design but only at a fixed
  5-sample budget on spectrogram/gait data, and HARLLM gives a fine-tuning curve with no
  enrollment arm.
* *Are they ready for prime-time?* (arXiv 2608.13316; fetched directly — the ceilings sweep did not
  surface it, so it is absent from that table) — three modes (linear, frozen+attention
  head, fine-tune); **UniMTS is "the best frozen feature extractor overall"**; NormWear and UniMTS
  are "evaluated below their modality-native potential" on triaxial input. Same caveat applies to us.
* **Inertia-1** (Xu … Yang — JHU / UCLA / Duke; arXiv 2607.06617; `yang-ai-lab/Inertia-1`): ten
  objectives, 5M/30M/100M, 15 datasets, linear probe + full fine-tune; pretraining beats from-scratch
  CNN/ViT consistently (HHAR LP AUROC 89.1 → 93.5); **no label-efficiency curve**; "more (diverse)
  pretraining data yields steadier gains than larger model sizes"; linear-probe quality plateaus
  with size.
* **Haresamudram et al. 2024** (arXiv 2408.12023): natural-language supervision "performs
  substantially worse than standard end-to-end training and self-supervision"; causes are sensor
  heterogeneity and poor text descriptions. Directly explains our primitive-path and text-path
  results; numbers pending.
* **IMWUT'26 survey** (Bian et al., 10.1145/3810230): lifecycle taxonomy with an explicit
  *adaptation* axis — linear probe / fine-tune / few-shot / personalisation — the vocabulary to
  use when describing Stage F.
* Recent open-vocabulary lineage to cite: IMUZero (IMWUT'25), LanHAR (IMWUT'24), OV-HAR
  (PerCom'25 WS), ActivityNarrated (2026), SensorLM (Google, 59.7M h; few-shot and label
  efficiency), HARGPT.
* Embodied anchors: **Ego-METAS** (2026) — online temporal action segmentation over EgoExo4D +
  CMU-MMAC + CaptainCook4D with IMU among five modalities under energy budgets — the benchmark
  shape our embodied row should match; **IMU-to-4D "Seeing Without Eyes"** (2026) — motion, text
  and scene from earbud/watch/phone IMUs, framed as "embodied AI that perceives the world through
  motion rather than sight."

### Addendum from the scite digests (same day)

* **The end-user-teaching lineage needs a targeted search.** A generic "programming by
  demonstration" query returned robotics PbD, not the HCI canon. The lineage to cite is uWave
  (Liu et al., PerCom 2009 — "allows users to define their own personal gestures" from a single
  training sample), Exemplar (Hartmann, CHI 2007), MAGIC (Ashbrook & Starner, CHI 2010), the
  $1-family recognisers, and IMWUT "interactive machine teaching" work. Confirmed citable via
  scite (same day): uWave — Liu, Wang, Zhong, PerCom 2009, 10.1109/percom.2009.4912759, 562
  citing publications; MAGIC — Ashbrook & Starner, CHI 2010, 10.1145/1753326.1753653; Exemplar —
  Hartmann et al., CHI 2007, 10.1145/1240624.1240646 ("authoring sensor-based interactions by
  demonstration"). Full texts are closed-access; cite from metadata.
* **Nobody reports a samples-per-class crossover between enrollment/prototypes and
  fine-tuning.** FSID (Belal et al., Sci Rep 2025, 10.1038/s41598-025-04323-7) has the matched
  design — frozen self-supervised transformer + weight-imprinted prototypes vs fine-tuning at 5
  novel samples — but on spectrogram images and gait sets only, and its preprint's best
  configuration still fine-tunes. HARLLM (arXiv 2605.12019, 2026) gives a fine-tuning
  label-efficiency curve only: LoRA on a frozen LLM, HHAR weighted-F1 > 80 at 1% labels, 93.68 at
  10%. So the Stage-F crossover is unclaimed territory.
* **RAG-HAR** (arXiv 2512.08984, 2025) is a training-free retrieval competitor: statistical
  descriptors → vector DB → LLM, "recognition and meaningful labelling of multiple unseen human
  activities" without training or fine-tuning. Conceptually the closest thing to enrollment in
  the recent literature; benchmarks not listed in what we read.
* **Haresamudram 2024, the numbers** (users split 60/20/20, no overlap): natural-language
  supervision zero-shot mean-F1 vs the best supervised/SSL baseline — HHAR 31.05 vs 59.25,
  MotionSense 38.97 vs 89.35, PAMAP2 10.88 vs 59.43, Mobiact 16.93 vs 82.36, MHEALTH 11.15 vs
  53.79, Myogym 1.47 vs 40.87: gaps of **28 to 65 points**. Adapting only the projection layer on
  100 labelled windows per class (~4 min of data) recovers 20–40 points; ChatGPT-diversified label
  sentences +5–7; body-part knowledge +6–10; combined (CLIP text encoder + SLIP loss + better
  sentences) HHAR 50.2 → 63.4. This is the external evidence for two things we found the hard way:
  text alignment alone is weak on heterogeneous sensors, and a small adaptation on the deployment
  side recovers most of it.

## 5. Decisions

1. Adopt the two-stage plan; Stage U first, on existing caches, after pre-registration and go.
2. Frame the paper as arbitrary / end-user-defined activity detection, with embodied AI, OpenPack
   and MM-Fit as the headline settings and the sealed six as the controlled robustness study.
   No clinical framing anywhere.
3. Add TransfHAR's frozen-probe protocol as a Stage-F treatment; evaluate AnyMo zero-shot if
   weights are obtainable; ask for both.
4. Add an embodied evaluation source (Ego-Exo4D IMU) — the framing that carries most weight is
   the one we currently cannot score.
5. The concession stays in the paper: fixed vocabulary, controlled device, plentiful labels →
   train a specialist.
