# JEPA / latent world-model SSL: best-practice survey vs. the HALO Phase-A design

> **Historical literature audit.** The "design under review" below is the retired masked-JEPA and
> pooled-VICReg recipe. Its findings informed the implemented future objective in
> [`JEPA_PRETRAINING_OBJECTIVE.md`](JEPA_PRETRAINING_OBJECTIVE.md); do not read the restatement below
> as the current trainer behavior.

Compiled 2026-09-09. Numbers marked **[V]** were read by me directly out of the primary PDF
(downloaded from arXiv, text extracted locally). Numbers marked **[S]** come from a subagent's
search pass and were **not** independently verified by me — treat those as leads.

**Design under review (restated):**
1. Student sees a masked view: ONE contiguous temporal block ≈50% of patches, occasionally a whole
   co-located sensor. Predicts EMA-teacher contextual tokens at hidden positions. **No separate
   predictor** — the student encoder emits predictions at mask-token positions. Cosine loss.
   **No target normalisation.**
2. Collapse control = VICReg (25/25/1) on pooled embeddings of two augmented views.
3. Optional decode-to-physics: MSE onto parameter-free filterbank band energies at hidden positions.
4. Loss weights calibrated once (gradient-norm share) after warmup.

---

## Q1. Predictor design — is omitting the separate predictor safe?

### The direct evidence that it is *not* (in the pure joint-embedding setting)

**BYOL** (Grill et al., NeurIPS 2020, arXiv:2006.07733), **Table 5b** [V] — 300-epoch ImageNet linear
eval, intermediate variants between BYOL and SimCLR. With β=0 (no negatives):

| predictor | target network | β | top-1 |
|---|---|---|---|
| ✓ | ✓ | 0 | **72.5** (BYOL) |
| ✓ | ✗ | 0 | 0.3 |
| ✗ | ✓ | 0 | 0.2 |
| ✗ | ✗ | 0 | 0.1 |

Paper text [V]: *"Removing the predictor in BYOL results in an unsupervised version of Mean Teacher
with no classification loss… This variant of BYOL collapses (Row 7 of Table 5) which suggests that
the additional predictor is critical to prevent collapse in an unsupervised scenario."* And: *"the
representation does collapse when either is removed."*

They further show [V] you can drop the *target network* if you keep the predictor **near-optimal**
— optimal linear predictor by per-batch regression → 52.5%; raising only the predictor's LR → 66.5%;
raising both projector and predictor LR → ≈25%. Conclusion: *"keeping the predictor near-optimal at
all times is important to preventing collapse, which may be one of the roles of BYOL's target
network."*

**Tian, Chen & Ganguli, "Understanding self-supervised learning dynamics without contrastive pairs"
(ICML 2021, arXiv:2102.06810)** [V, abstract verified verbatim] — the theory behind the above: they
analyse the nonlinear dynamics of the predictor + stop-gradient + EMA + weight-decay system in linear
networks and derive **DirectPred**, which *sets* the linear predictor in closed form from the
eigenspectrum of its input covariance rather than learning it. DirectPred beats a learned linear
predictor by **+2.5%** at 300 epochs and **+5%** at 60 epochs on ImageNet. The load-bearing point for
this review: in their account the predictor is not an optional convenience, it is the component whose
eigen-alignment dynamics *is* the collapse-avoidance mechanism.

### Predictor capacity matters, in a specific direction

**I-JEPA** (Assran et al., CVPR 2023, arXiv:2301.08243) [V]. Architecture: predictor is a
*deliberately narrow* ViT, embedding dim fixed at **384** for every backbone, depth 6 (ViT-B/16),
12 (ViT-L/H), 16 (ViT-G).

- **Table 12** (predictor depth, ViT-L/16, 500 ep, IN-1% linear probe): depth 6 → **64.0**, depth 12 → **66.9**.
- **Table 14** (predictor width, ViT-L/16 = 1024-ch encoder, 600 ep, IN-1% fine-tune): width 384 → **70.7**, width 1024 → **68.4**. Paper: *"Having a width bottleneck in the predictor improves the downstream performances."*

So the empirically good predictor is **deep but narrow** — a lossy, low-rank map. That geometry is
exactly what an encoder-emits-its-own-predictions design cannot express: the prediction head has the
encoder's full width and zero extra depth.

**V-JEPA** (Bardes et al. 2024, arXiv:2404.08471) [V]: predictor = *"a narrow transformer implemented
using 12 blocks with an embedding dimension of 384"*, taking encoder outputs plus learnable mask
tokens with positional embeddings. Same recipe.

**data2vec 2.0** (Baevski et al., ICML 2023, arXiv:2212.07525) [V]: separate lightweight
**convolutional** decoder — D=4 layers, dim 384, kernel 7, 16 groups (Table 9, speech). Mask tokens
can be **random Gaussian noise** rather than a learned vector — they found that sufficient.

**V-JEPA 2 / 2-AC** (arXiv:2506.09985): predictor retained; 2-AC is a ~300M block-causal predictor on
a frozen encoder [S].

### The two published precedents for dropping the predictor

**DINO** (Caron et al., ICCV 2021, arXiv:2104.14294) [V], Table 7 discussion: *"adding a predictor to
the student network has little impact (row 6) while it is critical in BYOL to prevent collapse."* But
this is only true because DINO substitutes a different anti-collapse mechanism (centering +
sharpening on the teacher softmax) — and *"in the absence of momentum, our framework does not work
(row 2)."*

**LeJEPA** (Balestriero & LeCun 2025, arXiv:2511.08544) [V]: *"prior work has shown both empirically
and theoretically that predictors in image JEPA (without asymmetric information) and teacher-student
architectures serve primarily to prevent collapse… Removing these components produces collapsed
encoders, i.e., with performances at chance-level. Thanks to LeJEPA's SIGReg loss, we can remove both
the predictor and teacher-student architecture without suffering from collapse, as shown in Table 4."*
Note the parenthetical **"without asymmetric information"** — LeJEPA's JEPA is view-based (8 crops),
not positionally-conditioned masked prediction.

**Verdict for the design under review.** The design is not literally "BYOL minus predictor" — mask
tokens + positional conditioning supply real asymmetric information, and MaskFeat/BERT show an
encoder with a thin head can learn from masked prediction against a *fixed* target. But against an
**EMA teacher**, dropping the predictor is the one configuration BYOL Table 5b measures at 0.2% and
LeJEPA describes as chance-level absent a distributional guarantee. The literature's preferred
geometry (deep + width-bottlenecked, I-JEPA Tables 12/14) is precisely what the encoder cannot
supply, and folding the prediction map into the encoder's own output space forces one representation
to serve two jobs. **This is the largest single deviation.** A 4–6 block, 256–384-dim predictor is
the cheapest, best-evidenced fix and is the near-universal default (I-JEPA, V-JEPA, V-JEPA 2,
data2vec 2.0, TS-JEPA [S], S-JEPA [S], HAR-JEPA [S]).

---

## Q2. Target construction and normalisation

### What each system actually does [V unless noted]

| system | target | normalisation | loss |
|---|---|---|---|
| I-JEPA | last-layer EMA target-encoder output at masked positions | none stated | mean **L2²** |
| V-JEPA | masked *output* of EMA y-encoder ("contextualized targets", crediting data2vec 2.0) | none stated | **L1** (*"we modify it to use an ℓ1 regression, which we found to be more stable"*) |
| data2vec | avg of top-K teacher FFN blocks | **per-block normalisation before averaging**: instance-norm (speech), parameter-free LayerNorm (vision/NLP) | Smooth L1 |
| data2vec 2.0 | avg of top-K FFN blocks | **IN → AVG → LN** (vision/NLP), **IN → AVG** (speech) — Tables 8–10 | **L2** |
| DINO | teacher softmax over prototypes | **centering + sharpening** | cross-entropy |
| BYOL | ℓ2-normalised projection | ℓ2 | normalised MSE ≡ 2−2·cos |

### The evidence that normalisation is an anti-collapse device

**data2vec** (Baevski et al., ICML 2022, arXiv:2202.03555) §3.3 [V], verbatim:

> *"Normalizing the targets helps prevent the model from collapsing into a constant representation
> for all time-steps and it also prevents layers with high norm to dominate the target features. For
> speech representations, we use instance normalization … while for NLP and vision we found
> parameter-less layer normalization to be sufficient. Variance-Invariance-Covariance regularization
> (Bardes et al., 2021) also addresses this problem but we found the above strategy to perform well
> and it does not introduce additional hyper-parameters."*

and §6 "Representation Collapse" [V]:

> *"we found collapse to be more likely for modalities where **adjacent targets are very correlated
> and where longer spans need to be masked, e.g., speech**. We address this by promoting variance
> through normalizing target representations over the sequence or batch. For models where targets are
> less correlated, such as vision and NLP, momentum tracking is sufficient."*

This is the single most on-point paragraph in the survey for the design under review. 1-second IMU
patches with a ~50% contiguous block mask sit squarely in the *"adjacent targets highly correlated,
long spans masked"* regime that data2vec names as the highest-collapse-risk case — the case where
they say momentum tracking alone is **not** sufficient and per-sequence normalisation is the remedy.
Note also that data2vec explicitly names VICReg as the *alternative* to target normalisation, not a
complement — but see Q5 on where the design applies VICReg.

**DINO** §5.3 + Fig. 7 [V]: centering and sharpening are complementary and *both* required —
*"If one operation is missing, the KL converges to zero, indicating a collapse"* (entropy →0 with no
centering, →−log(1/K) with no sharpening: two different collapse modes).

### Top-K layer averaging

**data2vec Fig. 2** [V]: predicting the average of K teacher layers beats the top layer alone (K=1)
for all three modalities; *"The effect is very pronounced for speech and NLP while for vision there
is still a slight advantage."* data2vec 2.0 uses K=8–10 (speech), K=10–32 (vision/NLP) [V, Tables 8–9].

### L1 vs L2 vs cosine

**data2vec Appendix B, Table 7** [V] — Librispeech dev-other WER, lower better:
L2 17.1 · L1 17.2 · SmoothL1(β=0.08) 17.2 · **β=0.25 16.8** · **β=0.5 16.8** · β=1 17.3.
Paper: *"different choices of the loss function have a relatively small effect on final performance."*
V-JEPA prefers L1 for *stability*, not accuracy. BYOL's loss is literally cosine. So the loss family
is a minor lever; **cosine is well-precedented and not itself a defect**.

**Verdict for the design under review.** Cosine is fine. Last-layer-only targets are I-JEPA/V-JEPA
practice and acceptable, though data2vec Fig. 2 says top-K averaging is a cheap win and is *most*
pronounced for sequential signals. The real deviation is **no target normalisation**: cosine
normalises each token's *magnitude* but does nothing to stop every masked position converging on the
same *direction*, which is the collapse mode data2vec's instance-norm-over-the-sequence exists to
kill — in exactly the correlated-adjacent-targets/long-span regime IMU occupies. Adding a
parameter-free per-sequence normalisation of teacher targets is a one-line change with the strongest
prior in this survey.

---

## Q3. Masking geometry and ratio

### Images/video: many blocks, not one

**I-JEPA Table 6** [V] — ViT-B/16, 300 ep, IN-1% linear:

| strategy | targets | context | top-1 |
|---|---|---|---|
| **multi-block** | 4 × Block(0.15,0.2) | Block(0.85,1.0) ∖ targets | **54.2** |
| block | 1 × Block(0.6) | complement | 20.2 |
| random | 1 × Random(0.6) | complement | 17.6 |
| rasterized | 3 quadrants | 1 quadrant | 15.5 |

**I-JEPA Table 10** [V] — number of target blocks, all else fixed:
1 → **9.0** · 2 → 22.0 · 3 → 48.5 · 4 → **54.2**.

That is a 45-point spread between one target block and four. **Table 8** [V]: target block scale
(0.15,0.2) is best (54.2), with both smaller (0.075–0.2 → 19.2) and larger (0.2–0.3 → 33.6) worse.
**Table 9** [V]: shrinking the context hurts monotonically (0.85–1.0 → 54.2 down to 0.40–1.0 → 31.2).

**V-JEPA** [V]: masks are the union of many blocks — short-range (8 blocks, 15% of each frame) and
long-range (2 blocks, 70% of each frame), each extended over the full temporal extent, giving an
**effective ratio ≈90%**. Two masks per clip (multi-mask).

**V-JEPA Table 4** [V] — ViT-L/16, attentive probe, frozen (K400 / SSv2 / IN1K):

| masking | K400 | SSv2 | IN1K |
|---|---|---|---|
| random-tube[0.9] | 51.5 | 46.4 | 55.6 |
| causal multi-block[6] | 61.3 | 49.8 | 66.9 |
| causal multi-block[12] | 71.9 | 63.6 | 72.2 |
| **multi-block** | **72.9** | **67.4** | **72.8** |

Note this corrects a premise in the brief: V-JEPA does **not** use tube masking. Random tube masking
at 90% is their *worst* row (−21 K400). Appendix E.4 / Fig. 8 [V]: *"We find that sampling several
blocks to perform better than sampling a single large block"* (8b) and *"low spatial or temporal
coverage results in a trivial prediction task, which degrades downstream performance"* (8c); Fig. 8a:
two masks per clip beats one.

### Masking the teacher's *output*, not its input

**I-JEPA Table 11** [V] — ViT-H/16, 300 ep, IN-1% linear: mask the target-encoder **output** →
**67.3**; mask its **input** → **56.1** (+11.2 for the former). V-JEPA does the same, explicitly
crediting data2vec 2.0's "contextualized targets". The design under review gives the teacher the
clean input and reads out at masked positions — **aligned, and this is one of the highest-leverage
choices in the paper.**

### Temporal / physiological data: ratios cluster lower, spans beat scatter

| system | ratio | geometry | source |
|---|---|---|---|
| **data2vec 2.0, speech** | **R = 0.50 / 0.55**, block width B = 5 | inverse block | [V] Table 9 |
| data2vec 2.0, vision | R = 0.75 / 0.80, B = 3 | inverse block | [V] Table 8 |
| CAE (images) | **50% best** (60% helps probes, −0.2 seg) | random | [V] Table 7 |
| LSM (wearables) | **0.8** best of 0.3–0.9 | random | [V] Table 4 |
| Ti-MAE | 75% best (0.45→.308, .60→.265, **.75→.210**, .90→.248 MSE) | random | [S] |
| SimMTM | **50% best**, 75% worse (0.409 vs 0.422 MSE) | — | [S] |
| TS-JEPA | 70% | random patch | [S] |
| LIMU-BERT (IMU) | **15%**, geometric spans, π=0.2, K≤10 | span | [S] |
| TST | 15%, geometric spans, mean length 3 | span, per-variable | [S] |
| TimeMAE | 60% | random sub-series | [S] |
| PatchTST-SSL | 40% of patches | random patch | [S] |

The **most reproduced finding for temporal data** is that *contiguous spans beat scattered points*:
LIMU-BERT (Fig. 10, span > single-mask), TST (geometric-span-per-variable > independent Bernoulli),
SimMTM (geometric > random) [all S]. Reason given consistently: short scattered masks are trivially
solved by interpolating neighbours. The design's contiguous-block choice is **on the right side of
this**. But note that "contiguous span" in every one of those papers means *several* spans, not one.

### Whole-sensor / channel masking

**LSM, "Scaling Wearable Foundation Models"** (Narayanswamy et al., arXiv:**2410.13638**, 2024 — note
the ID in the brief, 2412.09758, is wrong), **Table 5** [V] — MAE/MSE at fixed 0.8 ratio:

| strategy | Interp-60m MAE/MSE | Extrap-60m MAE/MSE |
|---|---|---|
| **Random** | **0.24 / 0.26** | **0.37 / 0.44** |
| Structured (Temporal) | 0.24 / 0.26 | 0.37 / 0.44 |
| **Structured (Sensor)** | **0.54 / 0.71** | **0.53 / 0.73** |
| Temporal Interpolation | 0.41 / 0.48 | 0.52 / 0.66 |
| Temporal Extrapolation | 0.43 / 0.51 | 0.51 / 0.64 |

Whole-sensor masking is the *worst* of five as a **sole** strategy, and they selected random for all
downstream work. Caveat: these are reconstruction errors on a harder task, not downstream
representation quality.

**LSM-2 / AIM** (arXiv:2506.05321, 2025) [S] mixes artificial masks: 80% random imputation, 50%
temporal-slice (all sensors at a timepoint), **50% signal-slice (all timepoints of one sensor)** —
specifically to buy robustness to real sensor dropout, reporting ~73% smaller degradation across 12
missingness settings vs LSM-1. That is a *robustness* result, not a representation-quality result.

**Verdict for the design under review.** Contiguous > scattered is correct for temporal data. ~50% is
right in line with data2vec 2.0's speech setting (R=0.5, B=5) and CAE, and sits below the 70–90% of
the video/vision JEPAs — defensible, since no time-series JEPA in this survey goes near 90%. The
clear deviation is **ONE block**: I-JEPA Table 10 (1 block → 9.0 vs 4 → 54.2), I-JEPA Table 6
(block 20.2 vs multi-block 54.2), and V-JEPA Fig. 8b all say several blocks beat one large one, and
that is the cheapest fix available. Occasional whole-sensor masking is defensible *as a minority
component* on LSM-2 grounds (missingness robustness), and LSM Table 5 says it must not become primary.

---

## Q4. Reconstruction / hand-crafted-feature decoding as an anti-collapse anchor

This is the question where the literature genuinely splits, so both sides are laid out.

### Evidence AGAINST decoding (all of it is "latent INSTEAD OF pixels")

**I-JEPA Table 7** [V] — IN-1% linear: target-encoder output (500 ep) **66.9** vs pixels (800 ep) **40.7**.

**V-JEPA Table 1** [V] — ViT-L/16, 90K iters, same masking:

| target | K400 frozen | SSv2 frozen | IN1K frozen | K400 fine-tune |
|---|---|---|---|---|
| Pixels | 68.6 | 66.0 | 73.3 | 85.4 |
| **Features** | **73.7** | 66.2 | 74.8 | 85.6 |

Important nuance often lost: the gain is **+5.1 in frozen K400 and ~0 under fine-tuning** (85.4 vs
85.6) and ~0 on SSv2 frozen. V-JEPA's case against pixels is a case about *frozen-representation
semantic level*, not about optimisation or collapse.

**DINO-WM** (Zhou, Pan, LeCun, Pinto, arXiv:2411.04983), Appendix A.4.3 Table 7 [S]: letting the
decoder's reconstruction gradient reach the latent predictor drops PushT success **0.92 → 0.80**;
they keep the decoder fully decoupled (visualisation only) and conclude this *"underscores the
advantage of disentangling feature learning from reconstruction objectives."* This is the closest
structural analogue to the design's coupled decode term and it points negative.

**TD-MPC2** (Hansen et al., ICLR 2024, arXiv:2310.16828) [S], verbatim rationale: *"Learning a
generative model of the environment using a reconstruction (decoder) objective is tempting due to its
rich learning signal. However, accurately predicting raw future observations … is a difficult
problem, and does not necessarily lead to effective control."* Decoder-free by construction; same for
SPR and PLDM (which uses **VICReg-style variance/covariance + inverse dynamics** instead of a decoder)
[S]. **Ni et al. 2024** (arXiv:2401.08898, ICLR) [S]: stop-gradient targets *provably* avoid
collapse in linear self-predictive models — collapse-avoidance comes from the target mechanism, not
from a decoder.

### Evidence FOR — and specifically for *hand-crafted feature* targets

**MaskFeat** (Wei et al., CVPR 2022, arXiv:2112.09133) [V] — ViT-B, IN-1K, 300 ep, fine-tune top-1:
scratch 81.8 · **pixel 82.5** · **HOG 83.6** · dVAE token 82.8 · MoCo v3 feat 83.9 · DINO feat 84.0 ·
supervised feat 82.6. On video (K400, MViTv2-S, 300 ep): scratch 81.1 · pixel **80.7** (worse than
scratch) · **HOG 82.2** · DINO feat 82.5.

So a parameter-free hand-crafted descriptor is a *strictly better* target than raw signal and is
competitive with learned features — direct support for filterbank band energies over raw
reconstruction. **But two caveats from the same paper:**

**MaskFeat Table 12a** [V]: local contrast normalisation of HOG is **essential** — vs the default ℓ2
normalisation, ℓ1 costs **−0.8%** and *no normalisation at all costs* **−1.4%**.

**MaskFeat Table 13** [V] — the closest published experiment to the design's combined objective:

| targets | top-1 |
|---|---|
| pixel | 82.5 (−1.1) |
| **HOG** | **83.6** |
| **pixel + HOG** (two linear heads, losses averaged) | **82.3 (−1.3)** |

Paper: *"Though further tuning the loss weighting might improve this result, it signals that the two
objectives can not benefit each other. This is reasonable, as HOG targets are locally normalized
while pixel colors are strongly influenced by local brightness changes."*

**CAE — Context Autoencoder** (Chen et al., IJCV / arXiv:2202.03026) [V] — the closest *architectural*
precedent to the design's U-shape: encoder → latent contextual regressor predicting masked-patch
**representations** (alignment loss, in latent space) → decoder mapping those predicted latents to
DALL-E-token targets. **Table 5**, IN-1K 300 ep:

| decoder | alignment | LIN | ATT | FT | ADE Seg | COCO Det |
|---|---|---|---|---|---|---|
| ✗ | ✗ | 60.3 | 71.2 | 82.9 | 47.0 | 46.9 |
| ✓ | ✗ | 63.1 | 72.7 | 83.4 | 47.1 | 47.2 |
| ✗ | ✓ | 62.0 | 71.5 | 83.4 | 47.1 | 47.2 |
| **✓** | **✓** | **64.1** | **73.8** | **83.6** | **48.3** | **48.4** |

and, critically, the paper text [V]: *"We observe that if the pretraining task, masked patch
reconstruction, is not included, the training collapses, leading to a trivial solution."* CAE has no
EMA teacher — its decode-to-low-level-target head **is** its anti-collapse anchor, which is precisely
the role the design assigns to its filterbank term. Cost: 1.24× params, 1.24× training time.

**MAGE** (Li et al., CVPR 2023, arXiv:2211.09117) [S] Table 7, IN-1K linear probe: contrastive-only
72.9 · reconstruction-only 73.3 · **C+R 77.1**; the paper states *"the reconstructive loss acts as a
regularizer that prevents the encoder from learning shortcut solutions."* λ sweep: 0→73.3, 0.01→75.5,
0.05→76.7, **0.1→77.1**, 0.2→76.9, 0.5→76.6 — a clean unimodal curve.

**Dreamer V3** (Hafner et al., Nature 2025) Fig. 6b [S]: *"the performance of Dreamer predominantly
rests on the unsupervised reconstruction loss of its world model."*

**Verdict for the design under review.** The apparent V-JEPA-says-no objection does not actually apply:
V-JEPA/I-JEPA measured latent **instead of** pixels; the design does latent **plus** a decode head,
which is the CAE/MAGE configuration and there it helps (+3.8 LIN / +2.6 ATT / +1.3 ADE / +1.5 COCO
over decoder-less in CAE Table 5). Choosing a hand-crafted filterbank descriptor over raw signal is
the MaskFeat result and is right. **Two concrete risks, both with numbers:** (a) MaskFeat Table 13 is
the one published test of *combining* a hand-crafted-feature target with a second target and it lost
1.3 points, with the stated cause being a **normalised-vs-unnormalised target-statistics mismatch** —
"band energies of the raw signal" are unnormalised and reproduce that mismatch exactly, so per-window
log/contrast normalisation of the filterbank target is indicated (MaskFeat Table 12a: −1.4% without);
(b) DINO-WM Table 7 shows coupling a reconstruction gradient into a latent predictor can cost
0.92→0.80, so the term should be ablated on/off and not assumed free.

---

## Q5. Collapse prevention: EMA vs EMA+predictor vs VICReg vs SIGReg

**BYOL Table 5a** [V] — target mode, 300 ep IN linear: constant random network (τ=1) 18.8 ±0.7 ·
τ=0.999 69.8 · **τ=0.99 72.5** · τ=0.9 68.4 · stop-gradient-of-online (τ=0) **0.3**. All τ ∈ [0.9,
0.999] land above 68.4 — the EMA rate is forgiving but the *presence* of the lag is not optional.

**VICReg** (Bardes, Ponce & LeCun, ICLR 2022, arXiv:2105.04906) **Table 4** [V] — ResNet-50, 100 ep
IN linear; ME = momentum encoder, SG = stop-grad, PR = predictor:

| method | ME | SG | PR | No Reg | Var Reg | Var/Cov Reg |
|---|---|---|---|---|---|---|
| BYOL | ✓ | ✓ | ✓ | 69.3† | **70.2** | 69.5 |
| SimSiam | ✗ | ✓ | ✓ | 67.9† | 68.1 | 67.6 |
| SimSiam | ✗ | ✓ | ✗ | 35.1 | 67.3 | 67.1 |
| SimSiam | ✗ | ✗ | ✗ | **collapse** | 56.8 | 66.1 |
| VICReg | ✗ | ✗ | ✗ | **collapse** | 56.2 → 57.5 | 67.3 → **68.6†** |

Read carefully: **VICReg added on top of an architecture that already has ME+SG+PR is roughly a wash**
(BYOL 69.3 → 69.5 with Var/Cov; +0.9 with variance only). VICReg's value in this table is as a
*substitute* for those components, not a supplement. The exception in the JEPA setting is **C-JEPA**
(Mo & Tong, NeurIPS 2024, arXiv:2410.19560) [S], reported Table 4: I-JEPA baseline 63.7 lin / 82.5 FT
→ full Var+Cov+Invariance **69.5 / 83.6** at 100 ep ViT-B/16 — a large gain, **not verified by me**.

**VICReg Table 7 (Appendix D.4)** [V] — coefficient sensitivity, IN linear:

| λ | μ | ν | top-1 |
|---|---|---|---|
| 1 | 0 | 0 | collapse |
| 25 | 0 | 1 | collapse |
| 0 | 25 | 1 | collapse |
| 1 | 1 | 0 | 57.5 |
| 1 | 1 | 1 | collapse |
| 5 | 5 | 1 | 68.1 |
| 10 | 10 | 1 | 68.2 |
| **25** | **25** | **1** | **68.6** |
| 50 | 50 | 1 | 68.3 |

Text [V]: *"using very different values for λ and µ, or taking λ = µ with ν > µ leads to unstable
training … taking λ = µ and picking ν < µ leads to stable convergence, with the exact value picked
for mu having very limited influence."* So **25/25/1 is not a magic number** — anything in 5–50 with
λ=μ>ν gives 68.1–68.6. Good: the design's choice is in the flat region and does not need defending.

**LeJEPA** [V]: identifies the isotropic Gaussian as the distribution minimising downstream prediction
risk (Thm. 1) and enforces it with **SIGReg** (sketched/sliced Epps–Pulley test), linear time and
memory, single trade-off λ. Claims: no stop-grad, no teacher-student, no predictor, no schedulers.
Table 4 shows a teacher-student config gives *"a small performance boost for ViT models … but it is
not necessary to prevent collapse."* λ sensitivity (Fig. 8, ResNet-50/IN-100, 400 ep) is flat across
10⁻³–10⁻¹.

Direct criticism of VICReg [V, §5.2]: setting the SIGReg test statistic to
`mean(x)² + (std(x)−1)²` *"recovers the VICReg SSL method in the limit of large number of slices …
we however strongly advocate against such a setting as it would lead to shortcut solutions — a
phenomenon already observed in VICReg."* I.e. VICReg only matches the first two moments, leaving room
for non-Gaussian shortcut geometries that SIGReg's full distributional test forbids. No time-series
results in LeJEPA (image domains only: ImageNet-1K/100, Galaxy10, Food101). A LeJEPA-for-EEG paper
("Laya") exists [S, unverified].

**Verdict for the design under review.** The mechanism stack (EMA teacher + stop-grad + VICReg) is
each individually well-evidenced, and 25/25/1 sits in VICReg's flat region. **The structural concern
is where VICReg is applied**: on *pooled* embeddings of two augmented views. The JEPA loss operates on
*per-token* contextual embeddings at masked positions, and the characteristic masked-prediction
collapse mode — every masked position emitting the same vector — leaves the pooled embedding perfectly
well-conditioned and is therefore invisible to a pooled VICReg. That is exactly the failure
data2vec's per-sequence instance normalisation of targets is built to catch (Q2). Removing the
predictor (Q1) makes this gap matter more, not less. Cheapest fixes, in order: token-level variance
term, or target normalisation, or restore the predictor.

---

## Q6. Multi-objective weighting; is one-shot gradient-norm calibration a known practice?

**GradNorm** (Chen et al., ICML 2018, arXiv:1711.02257) [S]: per-step gradient update on the task
weights themselves, driving each task's gradient norm at a shared layer toward a target set by its
relative inverse training rate; weights renormalised so Σw=T. **Continuous, every step.** Their own
§5.3 "GradNorm Static" ablation takes the time-averaged weights E_t[w_i(t)] from a full GradNorm run
and retrains from scratch with them fixed — close to, but below, full GradNorm, and close to grid
search. That is the nearest published relative of the design's procedure, but it needs a complete
prior GradNorm run to produce the numbers.

**PCGrad** (Yu et al., NeurIPS 2020, arXiv:2001.06782) [S]: per-step projection of conflicting
gradients onto each other's normal plane. **Kendall, Gal & Cipolla** (CVPR 2018, arXiv:1705.07115)
[S]: learned homoscedastic log-variances, trained jointly; notably their weights **converge in ~100
iterations** while the network needs 30,000+ — but they never freeze them. **Sener & Koltun**
(NeurIPS 2018), **IMTL** (ICLR 2021), **Nash-MTL** (ICML 2022) [S]: all per-step.

**The deflationary literature** [S], which is the relevant precedent here:

- **Kurin et al., "In Defense of the Unitary Scalarization for Deep Multi-Task Learning"** (NeurIPS
  2022, arXiv:2201.04122): no specialized multi-task optimizer consistently beats plain fixed-weight
  summation once the baseline gets standard regularisation; SMTOs act mostly as implicit regularisers.
- **Xin et al., "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"** (NeurIPS
  2022, arXiv:2209.11379): MTO methods don't beat static weighting on the scalarization Pareto front,
  and — directly relevant — *"the dynamically assigned task weights do not move significantly"*
  during training in most runs (their Fig. 3). Also ~2.4× slowdown.
- **Lin, Ye & Zhang, random loss weighting (RLW/RGW)** (TMLR 2022, arXiv:2111.10603): literally random
  per-step weights are competitive with 8 SOTA gradient-balancing methods.

**What SSL papers actually do**: VICReg — hand-set constants from an ablation grid, flat over 5–50
[V]. MAGE — fixed λ=0.1 from a 6-point grid [S]. DINOv2 — fixed coefficients set by ablation [S].
MAE — single loss. **None** report a gradient-norm calibration step.

**Verdict for the design under review.** I found **no named precedent** for "measure each loss's
gradient-norm share once after warmup, freeze the weights thereafter." The nearest relatives are
GradNorm's own Static ablation (which requires a full dynamic run first) and a structurally identical
warmup-measure-then-freeze pattern applied to *learning rates* rather than loss weights in recent LLM
optimizer work [S, unverified]. It should be described as a cheap heuristic substitute for a weight
sweep, not as principled balancing. It is nonetheless well-defended by Xin et al. (adaptive weights
barely move anyway) and Kurin et al. (tuned fixed scalarization is competitive); the honest way to
report it is a small sensitivity sweep around the calibrated point showing a flat region, exactly as
VICReg Table 7 and MAGE Table 8 do.

---

## Q7. Explicit "U-shaped" encode → predict-latent → decode designs

**CAE** (arXiv:2202.03026) — see Q4 for the full Table 5. This *is* the design's shape: latent
regressor with an alignment loss in representation space, then a decoder from the **predicted latents**
to a low-level target. Both components help and are additive; the decode branch is load-bearing for
collapse. Mask ratio 50% best (Table 7) [V]. **This is the citation the design should be anchored to.**

**SiamMAE** (Gupta et al., NeurIPS 2023, arXiv:2305.14344) [S]: asymmetric masking (frame 1 at 0%,
frame 2 at 95%) with a cross-attention decoder; pixel target only, no latent-prediction head. Their
Table 2a shows the asymmetry is the load-bearing part: 90% symmetric too hard, 50% symmetric too easy.

**CrossMAE** (Fu et al. 2024, arXiv:2401.14391) [S]: decoder self-attention among mask tokens is
unnecessary; cross-attention-only decoding of a **25% random subset** of tokens reaches 83.5% IN-1K,
above full-reconstruction MAE, at 2.5–3.7× less decode compute. Relevant as a cost-reduction lever if
the filterbank decode term proves expensive.

**BEiT v2** (arXiv:2208.06366) [S]: a second, shallow (1–2 layer) MIM head on [CLS]+intermediate patch
tokens, loss summed with the main MIM loss; Table 5 — helps, and **shallower beats deeper**, and
sharing head parameters helps further.

**MimCo** [S]: patch-level 81.55 · image-level 81.59 · both **81.66** — small but positive.
**DINOv2** [S] Table 3: adding the iBOT MIM term gives ~+3% on dense prediction; KoLeo +8% on retrieval.
**dBOT** (ICLR 2024, arXiv:2209.03917) [S]: with multi-stage bootstrapping the *choice* of target
representation stops mattering — even a randomly initialised teacher works.

**Counterweight**: MaskFeat Table 13 (Q4) is the one clean case of two targets in one loss making
things worse, and its diagnosis is target-statistics mismatch, not the U-shape per se.

**Verdict for the design under review.** The U-shape is a real, published, *positively-evidenced*
pattern (CAE most directly; MAGE, BEiT v2, MimCo, DINOv2 supporting), so the "authors describe it as
U-shaped" framing is defensible and should cite CAE rather than being defended against V-JEPA. Two
design details the literature is specific about: keep the decode head **shallow** (BEiT v2: 1–2 layers
beat 3; CrossMAE: cross-attention-only, subset decoding), and **normalise the low-level target** so
its statistics don't fight the latent target (MaskFeat Tables 12a/13).

---

## Consolidated scorecard

| # | Design choice | Verdict | Strongest citation |
|---|---|---|---|
| 1 | Teacher sees clean input, mask applied to teacher **output** | **Aligned, high-leverage** | I-JEPA Table 11: 67.3 vs 56.1 [V] |
| 2 | Cosine loss on latents | **Aligned** (loss family is a minor lever) | data2vec Table 7: L1/L2/SmoothL1 span 16.8–17.3 [V]; BYOL loss ≡ cosine |
| 3 | ~50% mask ratio, contiguous | **Aligned for temporal data** | data2vec 2.0 Table 9 speech R=0.5/0.55 B=5 [V]; CAE Table 7 [V]; SimMTM 50>75 [S] |
| 4 | Decode to hand-crafted filterbank features (vs raw) | **Aligned** | MaskFeat Table 2: HOG 83.6 > pixel 82.5 > scratch 81.8 [V] |
| 5 | U-shape: latent prediction + decode head | **Aligned** (contra the V-JEPA framing — V-JEPA tested latent *instead of* pixels) | CAE Table 5: (✓,✓) 64.1/73.8/48.3 vs (✗,✗) 60.3/71.2/47.0 [V] |
| 6 | VICReg 25/25/1 coefficients | **Aligned** and insensitive | VICReg Table 7: 5–50 gives 68.1–68.6 [V] |
| 7 | **No separate predictor** | **Deviates — largest gap** | BYOL Table 5b: no-predictor + target net = 0.2% [V]; I-JEPA Tables 12/14 want deep+narrow [V] |
| 8 | **One contiguous block** (rather than several) | **Deviates** | I-JEPA Table 10: 1 block 9.0 vs 4 blocks 54.2 [V]; V-JEPA Fig. 8b [V] |
| 9 | **No target normalisation** | **Deviates** — and in the exact regime data2vec flags | data2vec §3.3 + §6 [V]; data2vec 2.0 IN→AVG→LN [V] |
| 10 | **VICReg on pooled embeddings only** | **Gap** — cannot see per-token collapse | VICReg Table 4 [V] (VICReg on top of ME+SG+PR ≈ neutral: 69.3→69.5) |
| 11 | Unnormalised band-energy target alongside a latent target | **Risk** | MaskFeat Table 13: pixel+HOG 82.3 < HOG 83.6 [V]; Table 12a: no-norm −1.4% [V] |
| 12 | Whole-sensor masking, occasional | **Defensible as a minority component only** | LSM Table 5: Structured(Sensor) 0.54/0.71 vs Random 0.24/0.26 [V]; LSM-2 AIM mixes it at 50% [S] |
| 13 | Last-layer targets (no top-K averaging) | **Acceptable; cheap win available** | data2vec Fig. 2, "very pronounced for speech" [V] |
| 14 | One-shot gradient-norm weight calibration | **No named precedent found** — novel; defensible but must be labelled as heuristic | GradNorm §5.3 Static [S]; Xin et al. Fig. 3 "weights do not move significantly" [S]; Kurin et al. [S] |
| 15 | Coupling the decode gradient into the latent path | **Ablate it — evidence exists both ways** | CAE Table 5 (+) [V] vs DINO-WM Table 7: 0.92→0.80 (−) [S] |

## Explicitly not verified
- C-JEPA (arXiv:2410.19560) Table 4 numbers (63.7 → 69.5) — subagent-reported only.
- DINO-WM Appendix A.4.3 Table 7 (0.92 → 0.80) — subagent-reported only; this is a load-bearing negative and worth checking before citing.
- MAGE Table 7/8, MimCo Table 8, BEiT v2 Table 5, DINOv2 Table 3 — subagent-reported.
- All time-series rows marked [S] (Ti-MAE, SimMTM, TS-JEPA, LIMU-BERT, TST, TimeMAE, PatchTST, LSM-2).
- LeJEPA Table 4's individual cells: the PDF's table extraction was ambiguous about which row is which (w/predictor × w/SWA); the qualitative claims quoted are verbatim and reliable, the per-cell numbers are not.
- No JEPA/masked-SSL paper found that reports removing the predictor while keeping an EMA teacher and *succeeding* on masked prediction — absence of evidence, not evidence of absence.
