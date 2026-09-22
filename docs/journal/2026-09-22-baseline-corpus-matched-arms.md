# Giving the baselines our corpus: what we run, and why each choice is defensible

Date: 2026-09-22. Status: **built and smoke-tested; no arm has been trained.** Supersedes the
frozen-trunk proposal in [the encoder isolation plan](2026-09-18-encoder-isolation-plan.md), whose
M2f level we are not running — see §4.

## 1. The question

Alex's supervisor asked for each baseline to be fine-tuned on our training corpus. The concern
behind the request is a real confound: HALO's encoder trains on `dsads, forth_trace, harmes, hhar,
kuhar, realdisp, wisdm, xrf_v2` and is tested on six sealed datasets, while every baseline is an
author-released checkpoint pretrained on something else — UK Biobank wrist accelerometry, a
delivery-rider phone corpus, synthetic mocap, clinical wearables. If our training distribution
simply sits closer to our test distribution than theirs does, our lead measures proximity rather
than method.

Nothing in the current tables controls for this. We also checked whether a free control existed
and it does not: the contract **excludes** the original LiMU-BERT and CrossHAR precisely *because*
their usable checkpoints were trained in this project, so the one self-pretrained comparison that
would have controlled for corpus proximity was removed for a different fairness reason.

## 2. The headline protocol does not change

Our claim is enrollment **without parameter updates** — zero-shot and few-shot by example. Under
that claim, evaluating every model frozen is the experimental condition, not an approximation of
each author's recipe. Nobody gets gradient steps at deployment, ourselves included.

This matters because the papers would otherwise push the other way. Of the four baselines, three
are not frozen by their own authors: HARNet reports frozen and full fine-tuning separately and
makes fine-tuning the headline; UniMTS never freezes its signal encoder in any reported
experiment; LiMU-BERT-X deliberately *replaced* the original's frozen protocol with joint
fine-tuning "to enhance the model's fitting capabilities". Only NormWear is frozen throughout.

So the arms below are **a separate study answering a separate question**, reported separately, and
never folded into the deployment table. That is also how the external benchmark
([*Foundation models for movement data: are they ready for prime-time?*](https://arxiv.org/html/2608.13316))
treats it, with linear, frozen and fine-tuned as three distinct modes.

## 3. The arms

Every arm sees the same 223,587 windows, the same subject holdout, the same sampler, the same
episodes, the same differentiable-neighbour objective, the same step budget and the same sealed
manifests. Only the encoder differs. Corpus scale — the released models' real advantage and the
confound in question — is removed by construction in the from-scratch arms.

| arm | encoder | command | measured cost |
|---|---|---|---|
| reference | HALO, from scratch | already run: `halo_neighbors_stage_a_40k_20260918` | — |
| M2 HARNet | HARNet architecture, random init | `--encoder-arch harnet --classifier neighbors` | 32 min |
| M2 LiMU-BERT | LiMU-BERT-X architecture, random init | `--encoder-arch limubert --classifier neighbors` | ~75 min |
| M2 UniMTS | UniMTS ST-GCN, random init | `--encoder-arch unimts --classifier neighbors` | **13.6 h** |
| frozen NormWear | released NormWear, trunk frozen | `frozen_baseline_adaptation --baseline normwear` | ~35 min encode + minutes |

The comparison is read as **1-NN against 1-NN**: pure encoder versus encoder, no classifier
anywhere in it. Today's baseline 1-NN at 8 s, k=8, against HALO's neighbours arm at 70.6: UniMTS
66.9, LiMU-BERT-X 63.9, NormWear 60.6, HARNet-10 52.1.

### Registered before any arm runs

* **Adapted or retrained baselines close most of the gap to HALO's 1-NN** → the supervisor's
  concern is validated; the headline comparison needs reframing before submission.
* **They improve but plateau well short** → we can state a quantity rather than an argument:
  training on our corpus recovers *X* of the *Y*-point gap, and the remainder is not explained by
  training-distribution proximity.
* A from-scratch arm landing *below* the frozen released row is **not** a defect. It says that
  model's pretraining is carrying it, which is a finding worth reporting.

## 4. Why each decision is what it is

**From-scratch is well precedented, by the baselines' own authors.** Three of the four papers run
the random-init same-architecture control themselves: HARNet makes it the denominator of its
headline claim ("a network of the same architecture but fully trained from scratch"), UniMTS names
it `Random`, LiMU-BERT-X reports "w/o pretraining" across label budgets. Two calibrations from
their numbers: in the full-label regime the pretraining gap is modest (UniMTS 5.9 accuracy points,
LiMU-BERT-X 1.3 at 90% labels), so with 223,587 windows these arms should be competitive and it is
informative if they are not; but HARNet's from-scratch control **loses to a random forest on six
of seven datasets**, so this control collapses on small data. Ours is not small.

**We are not running M2f (frozen trunk inside `MatchedCorpusEncoder`).** It needs a new
`--freeze-trunk` flag and a relaxed guard — new surface to argue about — and it is strictly *less*
generous than a full retrain. If training on our corpus does not close the gap, freezing will not.

**We are not reducing UniMTS's skeleton**, although our corpus populates only 9 of its 22 SMPL
joints. The saving would be real (the adjacency einsum is O(V²), so perhaps 2–4×, taking 13.6 h to
4–7 h) but the zero joints are not inert: after the first spatial convolution they carry signal and
feed back into the populated ones, so removing them changes the computation, and the nine are not
contiguous in the kinematic tree, so a subgraph needs a redefined adjacency — and ST-GCN's spatial
partitioning is *defined* by graph distance. It would stop being UniMTS's architecture. The
zero-filling is also in-distribution by design: pre-training masks "1 to 5 joints" and zeroes the
rest. **Seven hours is not worth "you modified their model".**

**NormWear is frozen because that is its protocol, not because it is expensive.** Across 11
datasets and 18 applications its authors only ever freeze the encoder and fit a closed-form probe —
logistic regression by Newton's method, ridge by Cholesky — citing Yuan's protocol so that
"performance differences are not due to variations in learning rate, regularization, or data
augmentation". They report no fine-tuned number and run no random-init control. Their framing is a
"starting point … with minimal tuning".

The cost is the second argument, and it is now measured rather than extrapolated: **27.4 windows/s
with a backward pass**, which at our 369-window step is 13.5 s/step and **150 GPU-hours** for 40k —
before the gradient checkpointing it would need, since it OOMs above ~8–12 windows on 24 GB while
the neighbour loss requires all 369 rows in one graph. Call it 200 hours. Partial unfreezing does
not rescue it (~100 h). And training it at all means stripping a `@torch.no_grad()` from the
authors' released `get_signal_embedding` — nobody has ever run a training step through it.

Frozen and cached instead: **108 windows/s forward-only, the whole corpus in ~35 minutes, a 1.9 GiB
cache**, then minutes to fit the projection. 300× cheaper, and it is what its authors do.

## 5. Two defects found while building this

**The matched LiMU-BERT contract was wrong.** `BACKBONE_CONTRACTS["limubert"]` declared
`rate_hz = 20.0`, feeding the trunk one-second clips. Three sources agree on 10 Hz: the released
checkpoint's positional-embedding table is `(20, 72)`; the deployment paper states "we reduced the
IMU data sampling rate from 20 Hz to 10 Hz"; and the released adapter encodes at `TARGET_HZ = 10.0`
with `native_window_sec = 2.0`. Every positional embedding covered half the physical time it was
pretrained for, and the window produced eight clips instead of four — also double the compute. The
stale "20" in the literature is the mask width (20 of 120) and the original SenSys classifier's
slice length, neither of which is the input window.

This was caught by comparing the two implementations, not by reading either. Our *published*
baseline rows are unaffected — they go through the adapter, which was correct — and the matched
path had never been run. `tests/test_matched_encoder_fidelity.py` now pins every backbone's
contract against its released adapter and, for LiMU-BERT, against the checkpoint tensor itself.

**NormWear's released feature extractor is `@torch.no_grad()`-decorated**
(`modules/normwear.py:508`). It is written as a feature extractor, not a trainable module. Recorded
because it is the clearest evidence that full training is outside its intended use.

## 6. What is built

* `model/tokenizer/matched_encoder.py` — LiMU-BERT contract corrected to 10 Hz.
* `tests/test_matched_encoder_fidelity.py` — the contract gate, 7 tests.
* `training/support_classifier/frozen_baseline_adaptation.py` — encode the corpus once through a
  baseline's **own** `window_features`, cache it, fit a projection with the differentiable-neighbour
  objective. The projection has the same shape as `MatchedCorpusEncoder`'s adapter so a frozen row
  and a matched row differ only in the encoder.
* `--baseline-projection NAME=PATH` in the sealed evaluator, applied after the frozen feature cache
  and folded into the feature fingerprint so an adapted row cannot be mistaken for a native one.

Channel slotting deserves a note. NormWear's enrollment feature is 768 × *real channels*, so an
accelerometer-only stream returns 2304 and a six-axis stream 4608, and one corpus-wide matrix needs
a fixed width. Averaging channels is ruled out by the adapter itself ("previously cost substantial
enrolled accuracy"); a projection per channel count would put configurations in different learned
spaces, which a support vote across heterogeneous streams must not do. So each channel's sub-vector
goes to a fixed slot and absent channels are zeros — the way absence is represented everywhere else
here. Nothing is altered or mixed, and a channel the device never carried contributes nothing to a
cosine comparison.

**Smoke tested, not trained:** HARNet from scratch runs 3 steps end to end through the real trainer
and `eval_transfer` rebuilds it; the NormWear path caches, slots, fits and round-trips through the
sealed evaluator.

## 7. Caveats to carry into any write-up

* These rows are a **project method, not native baseline results**. Label them as such, and
  disclose trainable parameters, steps, runtime and seed, per `docs/contracts/baseline_fairness.md`.
* A from-scratch arm **discards the pretraining that is these models' contribution**. It measures
  architecture given our data. That is the point, and it is also the limitation; the frozen released
  row must be reported beside it or the pair is misleading.
* The asymmetry is real and should be stated: HALO trains its encoder end to end from scratch; the
  frozen NormWear arm adapts a released trunk through a projection and keeps its large-scale
  pretraining, which we do not have.
* **NormWear has no from-scratch row**, and the reason to give is its protocol first and the
  measured 200 GPU-hours second.
* Our 40k-step arms are not comparable in scale to the baselines' pretraining — HARNet reports 420
  GPU-hours, LiMU-BERT-X 780. The from-scratch arms are not "their model retrained"; they are their
  architecture, given our corpus, at our budget.
* The external benchmark notes NormWear and UniMTS are "evaluated below their modality-native
  potential" when restricted to triaxial acceleration. The same applies here.
