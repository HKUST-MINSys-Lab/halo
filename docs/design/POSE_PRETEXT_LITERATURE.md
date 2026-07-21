# Pose-pretext direction — literature go/no-go brief

> **Generated 2026-07-21** by a 37-agent survey → deep-read → adversarial-verify → synthesize pass
> (Consensus + Exa; WebSearch/Perplexity budgets exhausted mid-run — see the coverage caveat in §7).
> **12 of 24 load-bearing numeric claims were REFUTED** by the verification pass and removed. What
> remains survived an independent attempt to refute it. Numbers here still need spot-checking
> against the PDFs before anything is written into a paper.

# GO / NO-GO Brief: IMU→Pose as a Pretext Task for Wearable Activity Recognition

## 1. Verdict

**NO-GO as framed. A narrow GO-WITH-MODIFICATION survives, but it is a much smaller contribution than the proposal assumes.**

The single strongest reason: the central hypothesis — *pose is a config-invariant canonical space* — is refuted by the pose literature itself. Every deployed sparse-IMU pose system recovers pose **conditional on** knowing the acquisition config: T-pose calibration, known or actively-tracked on-body placement, sensor-to-bone rotation, and a fixed sampling rate [1][2][6][9]. IMUPoser must classify which of 5 body locations the phone is in (90.8% instance-wise accuracy) *before* it can do pose [1]; WHIP re-solves the device-to-joint mounting rotation every 20 s and calls this "an idealized setting" [4]. Pose is not a config-invariant space you decode into; it is a space you can only reach *after* the config is resolved. Using it as the invariance mechanism inverts the dependency.

Secondary reason: the mechanism is already published. IMUCoCo (UIST 2025) trains an IMU encoder against SMPL pose + kinematics, states verbatim that "the KRs and PR are dropped once the training is completed," then freezes the encoder and feeds it to an activity-recognition head [10]. A reviewer will find it.

---

## 2. What sparse-IMU pose estimation can and cannot do today

**Metric warning:** MPJVE (mesh-vertex error), MPJPE (root-aligned joint error), N-MPJPE (scale-normalized), SIP error (shoulders+hips only) and MPJRE (all-joint rotation) are four to five different quantities. DiffusionPoser states outright that "there is no exact correspondence between metrics across papers" [3]. Do not build a single leaderboard from the table below; read it column-by-column.

| Config | Metric | Error | Source |
|---|---|---|---|
| 1 consumer device (pocket/wrist/head avg) | MPJVE | **16.27 cm** (SD 9.93) | IMUPoser [1] |
| 2 consumer devices | MPJVE | **13.9 cm** (SD 8.36) | IMUPoser [1] |
| 3 consumer devices | MPJVE | **11.1 cm** (SD 6.51) | IMUPoser [1] |
| 1–3 devices, better decoder | MPJVE | 10.6 cm (DIP-IMU) / 12.6 cm (TotalCapture) | MobilePoser [2] |
| Watch + phone, real 2026 consumer hw | N-MPJPE | **83.0 mm** (unseen actor) / 99.5 mm (unseen action) | WHIP [4] |
| All 4 IMUs, real consumer hw | N-MPJPE | 61.7 mm (WHIP) / 75.5 mm (IMUPoser retrained) | WHIP [4] |
| 6 research-grade Xsens (2018–22 methods) | pos err | 5–7 cm; SIP 13–17° | PIP/DynaIP [3][7] |
| 6 **cheap off-the-shelf** IMUs, real | pos err | **13.62 cm**, SIP 30.47° (PIP) | Ultra Inertial Poser [6] |
| Global translation, consumer, 1 s window | RTE | **17.6–27.6 cm per second** | MobilePoser [2] |
| Global translation, 6 Xsens, 10 s | RE10 | 0.25–0.32 m (worst trial 0.75–1.4 m) | DiffusionPoser [3] |
| Activity dependence, consumer | MPJVE | walking 7.6 → sitting 11.5 → push-ups 16.1 cm | MobilePoser [2] |
| Low-acceleration motion, 6 IMUs | pos err | 15.87 cm (PIP), "reverts to a standing pose" | Ultra Inertial Poser [6] |

**Direct answer: no.** From 1–2 consumer-placement IMUs you do not recover pose in any sense that would make it a useful supervision target for HAR. Three facts settle it:

1. **The 1→3 device slope is ~5 cm across the entire useful range** [1]. Sensors barely move the number. Something other than the sensors is doing most of the work — almost certainly the motion prior.
2. **Nobody has ever run the prior-only control.** I searched specifically: no published MPJPE for predicting the dataset-mean pose, a static standing pose, or an unconditional prior sample with sensors zeroed. The field has never measured how much of 10–16 cm is sensor-driven. Indirect proxies point the wrong way: PIP/TIP demonstrably fall back to a standing pose when acceleration is weak [6], and WHIP's weakest modality (insoles) still lands at 121.8–151.7 mm [4] — not catastrophically worse than watch+phone's 83–99.5 mm.
3. **Accuracy is worst exactly where HAR needs discrimination.** Cyclic locomotion is the best case; anything off the upright walking manifold is the worst [2][6]. And your own weak classes are stairs/ramp/elevator — which are gait-cadence-similar and separated by *altitude*, a channel pose does not contain.

Three claims commonly cited in support of this direction are **refuted** and must not be used: (a) "one pocket phone recovers both legs to ~10 cm" — IMUPoser never measures the uninstrumented leg [1]; (b) "uninstrumented limbs land at 20–27 cm" — true for arms, false for legs (~10 cm) [1]; (c) "the 2-sensor optimum is never a consumer placement" — on DiffusionPoser's genuinely exhaustive real-IMU table the best 2-sensor config is **both wrists** [3].

---

## 3. Is the idea taken?

**Yes, the mechanism is taken. The evaluation is not.**

**Closest work: IMUCoCo, Zhou/Arakawa/Agarwal/Goel, UIST 2025** [10]. It is your recipe, in order: LSTM IMU encoder → 24 SMPL joint-node features; pretraining loss = kinematic MSE (velocity, position, global+local orientation) + full-body SMPL pose MSE + cosine alignment; training data = AMASS forward-simulated IMU + DIP-IMU/Xsens; **"The KRs and PR are dropped once the training is completed"**; then "one can freeze the IMUCoCo model and feed the IMU data ... for the task," with an ST-GCN HAR head over wrist + thigh-pocket + ear. 23M params, 200 GPU-hours. HAR result: 73.7 macro-F1.

Second-closest: **Multi3Net** (ISWC 2024) [11] — pose↔IMU contrastive alignment + Pose2IMU regression, then unfrozen fine-tune against a no-pretraining control (+18.8% on OpenPack). **PIM** (IEEE ABC 2025) [12] — the identical *recipe shape* (physics pseudo-labels → per-task heads → drop heads → fine-tune) with a proper no-pretraining control, but scalar physics targets rather than pose. **MotionBERT** (ICCV 2023) [13] — pose-as-pretext → action recognition, but skeleton/video only, never inertial.

**The one thing that is genuinely open:** IMUCoCo's *only* ablations (Tables 3 and 4) are pose-estimation ablations. There is **no "w/o IMUCoCo" control for HAR anywhere in the paper**, no random-init baseline, no comparison to any standard HAR baseline (DeepConvLSTM, harnet), and no public HAR benchmark — the 73.7 is on the authors' own 12-participant, 3-hour, 10-activity Apple Watch dataset. So the question *"does pose pretraining actually help HAR?"* is unanswered by the paper that owns the mechanism. That absence is your only defensible foothold, and it is an ablation, not a method.

---

## 4. The sim-to-real gap

Numbers, with a correction: **the gap does not grow as sensors are removed** — that first-pass claim is refuted by the table it came from. DiffusionPoser real/synth global-angular ratios: 6 IMUs 2.06×, 5 IMUs 1.81×, 4 IMUs 1.12×, 3 IMUs 1.30×, 2 IMUs 1.31× [3]. The relative gap is *largest at the densest config*; synthetic degrades faster with sparsity than real does, so synthetic benchmarks **overstate** the marginal value of adding sensors. Mechanism quantified: mean sensor-to-bone orientation error 9.4° on real TotalCapture vs 0° synthetic [3]. (Caveat: DiffusionPoser's Table 2 conflicts with its own appendix Table 5 for the 4-IMU row, and the appendix tables contain verified duplicated rows.)

For real-data value, the corrected reading of MobilePoser [2] is the opposite of the headline: per-dataset, real-IMU fine-tuning buys **12–19% in-domain** and ~0 out-of-domain. On the consumer configuration specifically it is the largest single effect in the ablation (11.7 → 9.5 cm, −18.8%). Real data matters a lot; it just doesn't transfer across domains.

**Does this explain UniMTS 34.0 vs harnet 47.3?** Directionally consistent, but **the literature does not establish it**, and this is a thin spot you should not smooth over:

- No published head-to-head zero-shot comparison of any mocap-pretrained model against harnet/ssl-wearables exists. harnet has no zero-shot head; UniMTS benchmarks BioBankSSL only in few-shot/full-shot, where a classifier is trained. **Your 34.0 vs 47.3 has no literature precedent.** Whatever ConSE adapter gives harnet a zero-shot score is the load-bearing, unvalidated part — expect a reviewer to attack exactly there.
- The direction reverses with protocol: at full-shot, UniMTS (87.5) **beats** BioBankSSL (82.7) on the same 18 datasets. The Oxford group states zero-shot "remains unstable and does not scale reliably" and that comparisons "are more reliable when some degree of fine-tuning is undertaken"; their re-implementation scores 0.3070 vs UniMTS's published 0.343 — a 3.6-point reproducibility spread on the very metric [14].
- Independent re-measurements of UniMTS put it at 32.1 and 26.4, not 34.3 — but both come from the *same lab* (Salim, UNSW) [15]. There is one outside lab, not two. There is still no neutral third-party zero-shot IMU HAR benchmark.

**A go/no-go resting solely on a zero-shot delta is exposed. Pair it with a few-shot curve.**

---

## 5. AMASS practicalities

**Scale:** unverified. The only quantitative figure MPI-IS publishes today is "45 hours and growing"; the landing page still shows the 2019 abstract. There is no authoritative hours/motions/subjects count for the current release [16].

**Licence: this is an operational blocker for your cloud-burst workflow.** Beyond "one copy for archive purposes only," the AMASS licence states the Dataset "may not be reproduced, modified and/or made available in any form to any third party without Max-Planck's prior written permission." "Modified" plausibly reaches derived tensors. **Mirroring AMASS or AMASS-derived arrays into R2 for vast.ai pods is in clear tension with this**; no published interpretation exists either way. SMPL/SMPL-H/SMPL-X/DMPL require separate registrations. Per-constituent terms (CMU, KIT, BMLrub, …) could not be confirmed. Nymeria is plain CC BY-NC 4.0 and is a materially better fit for that workflow.

**ADL coverage: bad, and the evidence is indirect but convergent.** AMASS ships zero action labels. (a) 92.4% of AMASS duration comes from five scripted lab-protocol datasets; (b) **none of eight target everyday ADLs appears in BABEL's released 150-class taxonomy**; (c) every documented post-2019 addition is handball/dance/yoga/4D-scan-derived. BABEL's Zipf tail is brutal: the 100th category has 86 occurrences, the 200th has 8 — unusable at training scale even if the labels exist behind registration.

**Most damaging:** Darwish, Nicholson, Doherty & Yuan, *"Motion Capture is Not the Target Domain"* (arXiv 2602.11064) [17] runs essentially the proposed experiment inside the UniMTS framework and concludes mocap is the wrong pretraining domain, with 40% of the pretraining data nearly matching 100%. Caveats: unreviewed preprint, single-seed headline tables, and the scaling study subsamples HumanML3D (24,661 pairs), not full AMASS — nobody has published a full-AMASS-scale ablation.

---

## 6. Is HAR saturated?

"Saturated" is false as a blanket claim and true for what most papers benchmark on.

- **Done, subject-independently, at 93–99%:** UCI-HAR (~96.6% LOSO mean per-subject F1 — note this is *my* arithmetic mean over a published per-subject table, not an author-reported aggregate), MotionSense, WISDM, MM-Fit.
- **Not done, but ~12% of remaining headroom is label noise:** PAMAP2, Opportunity, MHEALTH. Opportunity numbers are not comparable across papers because the label track differs (5-class locomotion vs 18-class gesture) — never put them in one column.
- **No rigorous subject-independent USC-HAD number exists.** The circulating 92.28/98.93 figures are random-split.
- **"LOSO always costs ~10 points" is false** — WISDM BiGRU: 97.91 k-fold vs 98.02 LOSO.

**Where headroom genuinely remains:** (i) free-living fine-grained ADLs (0.576 macro-F1 on CAPTURE-24 vs 0.800 coarse), (ii) in-the-wild multi-label context (BA 0.773 on ExtraSensory, 51 labels, 5-fold subject-disjoint) [18], (iii) generalization across users/placements/devices.

**Critically for this proposal:** fine-grained locomotion is **not** the headroom. Ramp-vs-stairs is largely solved from lower-limb IMUs (3.13% error, subject-independent; 99% steady-state), and the strongest elevator/escalator systems use barometer or magnetometer (>0.90 F1) [19]. Your near-zero stairs/ramp/elevator F1 is a **placement and modality** problem — the discriminative channel is altitude, not gait geometry. **Pose pretraining cannot fix it**, because pose from a wrist/pocket IMU does not contain altitude either (15–30 cm of translation error accumulates per second [2]).

Also relevant: your 84.3 probe vs 39.6 zero-shot gap is normal, not pathological. IMU2CLIP (EMNLP Findings 2023) puts exactly this comparison in adjacent rows of one table — 18.46 → 62.52 F1 on Aria, a 44.1-point gap vs your 44.7 [20]. That is your best citation *and* it partly deflates the "novel ablation" framing. (Metric caveat: IMU2CLIP says only "measured via F1," never macro or micro.)

---

## 7. What would actually be novel

Almost nothing in the proposal as written. Being blunt about what is already gone:

- IMU→pose pretext, drop the head, freeze, HAR — **gone** (IMUCoCo, UIST'25) [10].
- Pose as an IMU-encoder pretraining signal — **gone** (Multi3Net, ISWC'24) [11].
- Pose-as-pretext → action recognition, conceptually — **gone** (MotionBERT, ICCV'23) [13].
- Physics/biomechanical pseudo-labels with dropped heads — **gone** (PIM, ABC'25) [12].
- AMASS-simulated IMU pretraining for HAR — **gone** (UniMTS), and argued to be the *wrong* domain [17].

The narrowest defensible residual, stated honestly as a **negative-results / measurement** contribution rather than a method:

> *The first controlled measurement of whether IMU→pose pretraining transfers to activity recognition — same encoder, same data, with and without the pose objective — on public, subject-disjoint, cross-dataset HAR benchmarks, including the prior-only control that the pose literature has never published.*

Two supporting facts make this real: (a) IMUCoCo has zero HAR ablations and evaluates on its own 3-hour custom dataset; (b) **no paper in the sparse-IMU pose literature reports a prior-only control** — mean-pose, static-standing, or sensors-zeroed. Running it would be a genuine service to two fields.

That is a workshop-to-mid-tier paper, not a MobiCom-class contribution, and it is more likely to produce a null than a win. If the project needs a headline method, this is not it.

**Search coverage caveat:** WebSearch and Perplexity budgets were exhausted; evidence is Consensus + Exa. IMUCoCo surfaced at UIST, not a HAR venue — keyword search over HAR literature systematically misses this idea. **Before writing any novelty sentence, hand-sweep ACM DL: UIST/CHI/IMWUT/ISWC 2024–2026 for "IMU + pose + activity recognition."** Two high-value papers remain unread behind paywalls: Wonderwall (IMWUT 10(1) Art. 14, doi 10.1145/3789688 — the only work containing a fine-tuning-vs-probing comparison across zero/few/full-shot on 10 benchmarks) and IMUZero (IMWUT 9(4), doi 10.1145/3770669).

---

## 8. Cheapest decisive experiment

**Run the prior-only control first. It costs about a day and it can kill the direction outright.**

IMUPoser and MobilePoser both release code and pretrained checkpoints; WHIP is trained with per-modality dropout at p=0.5, so the empty-subset case is directly evaluable in its existing checkpoint [4].

**Experiment A (half a day, no training).** Take a released consumer-config pose checkpoint. Evaluate three conditions on the same test set: (1) full sensor input, (2) sensors zeroed / all modalities dropped, (3) constant predicted-mean pose. Report MPJPE for all three.

> **Kill criterion:** if the sensors-zeroed condition lands within ~3 cm of the full-input condition (i.e. within the 1→3 device slope of ~5 cm), then the pose "signal" your encoder would be distilling is dominated by the motion prior, the pretext task carries almost no IMU-specific information, and the direction is dead. Given the shallow device slope and the documented standing-pose fallback [6], **I expect this outcome.**

**Experiment B (2–3 days, only if A survives).** The exact ablation IMUCoCo omits: identical encoder, identical AMASS-simulated data budget, three arms — (i) pose-pretrained, (ii) random init, (iii) your existing SSL/evidence-engine pretraining — all evaluated frozen-linear-probe and few-shot on your public subject-disjoint ZS-XD split.

> **Kill criterion:** if arm (i) does not beat arm (ii) by more than seed variance on ≥3 of your held-out datasets, there is nothing to publish. Note PIM's precedent: on MM-Fit its pretraining is **net negative** versus no pretraining once labels are plentiful (0.605 vs 0.672 at 32/class) [12].

**Do not** invest in a forward-simulation pipeline, biomechanical constraint losses, or an AMASS R2 mirror before A returns. The licence question alone (§5) should gate the mirror.

---

## 9. References

1. Mollyn, Arakawa, Goel, Harrison, Ahuja. *IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds.* CHI 2023. https://ar5iv.labs.arxiv.org/html/2304.12518
2. Xu, Gao, Hoffmann, Ahuja. *MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices.* UIST 2024. https://arxiv.org/html/2504.12492v1
3. Van Wouwe, Lee, Falisse, Delp, Liu. *DiffusionPoser: Real-time Human Motion Reconstruction From Arbitrary Sparse Sensors Using Autoregressive Diffusion.* CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Van_Wouwe_DiffusionPoser_Real-time_Human_Motion_Reconstruction_From_Arbitrary_Sparse_Sensors_Using_CVPR_2024_paper.pdf
4. Boscolo Camiletto, Dabral, Alvarado, Beeler, Habermann, Theobalt. *Towards Real-World Wearable Motion Reconstruction (WHIP).* arXiv:2607.09780. https://arxiv.org/html/2607.09780v1
5. Li et al. *UltraPoser: Pushing the Limits of IMU-based Full-Body Pose Estimation with Ultrasound Sensing on Consumer Wearables.* UIST 2025. https://samsonsjarkal.github.io/KeSun/files/uist25_ultraposer.pdf
6. *Ultra Inertial Poser: Scalable Motion Capture and Tracking from Sparse Inertial Sensors and Ultra-Wideband Ranging.* SIGGRAPH 2024. https://static.siplab.org/papers/siggraph2024-ultra_inertial_poser.pdf
7. Yi et al. *Physical Inertial Poser (PIP).* CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/papers/Yi_Physical_Inertial_Poser_PIP_Physics-Aware_Real-Time_Human_Motion_Tracking_From_Sparse_Inertial_Sensors_CVPR_2022_paper.pdf
8. Huang et al. *Deep Inertial Poser (DIP).* SIGGRAPH Asia 2018. https://dip.is.tuebingen.mpg.de/assets/dip.pdf
9. Zhang, Yi, Xu. *BaroPoser: Real-time Human Motion Tracking from IMUs and Barometers in Everyday Devices.* UIST 2025, arXiv:2508.03313. https://arxiv.org/html/2508.03313v1
10. Zhou, Arakawa, Agarwal, Goel. *IMUCoCo: Enabling Flexible On-Body IMU Placement for Human Pose Estimation and Activity Recognition.* UIST 2025. https://doi.org/10.1145/3746059.3747695 — full text http://smashlab.io/pdfs/imucoco.pdf
11. Fortes Rey, Ray, Xia, Wu, Lukowicz. *Enhancing Inertial Hand based HAR through Joint Representation of Language, Pose and Synthetic IMUs (Multi3Net).* ISWC 2024. https://arxiv.org/abs/2406.01316
12. Nshimyimana, Fortes Rey, Suh, Zhou, Lukowicz. *PIM: Physics-Informed Multi-task Pretraining.* IEEE ABC 2025. https://arxiv.org/abs/2503.17978
13. Zhu et al. *MotionBERT: A Unified Perspective on Learning Human Motion Representations.* ICCV 2023.
14. Awasthi, Moya Rueda, Fink. *Video-based Pose-Estimation Data as Source for Transfer Learning in Human Activity Recognition.* ICPR 2022. https://arxiv.org/abs/2212.01353
15. Li, Chen, Xue, Salim. *ZARA.* arXiv:2508.04038 (preprint; no stated venue). https://arxiv.org/abs/2508.04038
16. AMASS: Mahmood, Ghorbani, Troje, Pons-Moll, Black. ICCV 2019. https://amass.is.tue.mpg.de/
17. Darwish, Nicholson, Doherty, Yuan. *Motion Capture is Not the Target Domain.* arXiv:2602.11064 (unreviewed preprint, single-seed).
18. Vaizman, Weibel, Lanckriet. *Context Recognition In-the-Wild (ExtraSensory).* IMWUT 2017/2018.
19. *ELESON: elevator/escalator detection.* IEEE TMC 2024; Kang 2022 / Camargo 2021 locomotion-mode from lower-limb IMUs (medium confidence — Consensus abstracts only).
20. Moon et al. *IMU2CLIP.* EMNLP Findings 2023. https://arxiv.org/abs/2210.14395
21. Yi et al. *TransPose.* ACM TOG 40(4), SIGGRAPH 2021.
22. *Transformer IMU Calibrator.* ACM TOG 44(4)/SIGGRAPH 2025. https://arxiv.org/abs/2506.10580