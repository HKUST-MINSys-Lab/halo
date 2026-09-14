# JEPA pretraining and encoder-design findings

**Date:** 2026-09-12
**Status:** immutable entry — see `docs/journal/README.md` for the journal's rules. Some items
below reference a re-run currently in progress; where that matters it is flagged explicitly so a
later reader does not mistake an in-progress measurement for a concluded one.
**Detail:** the numbers and probes summarized here are in memory notes
`halo-jepa-run-analysis-20260912`, `halo-continuous-kernel-sweep-20260912`,
`halo-ssl-literature-verdict-20260912`, and the hand-off plan `docs/design/FRONTEND_PLAN_20260912.md`
(issues, fixes, and an experiment ladder for an implementing agent). This entry is the narrative
summary for slides; those documents are the technical record.

## The two encoder arms we're actually deciding between

The design of record (`docs/design/DESIGN_OF_RECORD.md`) carries three encoder arms. Two of them
are where the live work is: the **fixed multiresolution filterbank** (log-spaced physical
frequency bands at 0.5/1.0/1.5 s, no learnable frontend parameters) and the **continuous-kernel
multi-span** design (parametric Gabor kernels, nominally learnable, organised by physical span so
the same frequency is measured narrowband at long spans and broadband at short ones — a genuine
attempt at joint time–frequency resolution rather than a frequency-only decomposition). The
multi-span arm is currently the stronger of the two on sealed evaluation (see below), but — as of
this writing — with its kernels essentially frozen at initialization; whether the *learnable* part
of the design is earning its complexity is an open question addressed in
`FRONTEND_PLAN_20260912.md`, not yet answered.

## JEPA pretraining: implemented, and it doesn't clear the bar

We built and ran the Future-JEPA label-free predictive objective on both arms (past-only student,
EMA teacher, predictor forecasting future latent patch states, physical decoder to filterbank
targets) on a three-source label-free corpus (Capture-24, Nymeria Xsens, ExtraSensory).

**Measured (2026-09-12 sealed sweep, pre-fix objective — see the caveat below), ridge readout,
mean macro-F1 over the six sealed test streams:**

| arm | supervised adaptation steps | k=1 | k=8 | k=32 | k=128 |
|---|---:|---|---|---|---|
| random encoder, frozen | 0 | 42.2 | 54.1 | 62.9 | 69.9 |
| JEPA-pretrained encoder, frozen | 0 | 53.5 | 69.2 | 76.1 | 79.3 |
| random encoder + 5k adaptation steps | 5,000 | 53.7 | 69.3 | 76.2 | 79.2 |
| JEPA-pretrained + 5k adaptation | 5,000 | 55.5 | 70.7 | 76.6 | 79.6 |

**The finding:** frozen JEPA and random-plus-5k-labelled-adaptation land on the *same row* to
within noise. Roughly 15,000 steps of label-free pretraining bought exactly what 5,000 steps of
labelled episodic training bought, no more. Once both are combined the gain over adaptation alone
is small (+1–2 points), and a from-scratch end-to-end run at a larger labelled budget (35k steps)
matches or exceeds JEPA-then-adapt at every k. So: **under this protocol, JEPA pretraining did
not outperform simply training the encoder end-to-end on our (small) labelled adaptation set, by
more than noise.**

**Why we think this is happening, not just "JEPA doesn't work":**

1. **Our encoder is structurally simple, and that isn't a choice we can walk back cheaply.** The
   large learnable frontends that make JEPA/masked pretraining pay off elsewhere (audio, video)
   universally assume a very high native sampling rate — wav2vec 2.0 and HuBERT run seven strided
   convolution layers over 16 kHz audio, using the sample-rate headroom to compound abstraction
   through repeated downsampling before attention ever sees the signal. Our IMU sources run
   20–240 Hz. At that rate there is very little room to convolve *down* before hitting the
   physical content floor, so both our arms currently rely on a largely engineered spectral
   decomposition (fixed log-spaced bands, or physically parameterised Gabor kernels) with very
   few learnable parameters at the front (some 800–25,000, versus low millions in a comparable
   audio frontend), and no compositional stage after the decomposition before attention. This
   plausibly limits how much a self-supervised *representation-learning* signal like JEPA can
   actually improve — there is comparatively little undetermined structure in the frontend for a
   predictive loss to shape. This is being addressed (see "Encoder design: still open," below).
2. **Within a short window, target representations are often close to trivially predictable —
   and we checked whether this is collapse. It largely isn't.** Direct probing of the JEPA
   teacher targets showed same-window, same-resolution targets sitting at 0.6–0.85 cosine
   similarity, rising toward 0.95+ on the stillest third of windows. Initial suspicion was
   representation collapse. Two checks argue against that reading: encoder effective rank stayed
   high through training (roughly 140 of 256 dimensions, not the single-digit ranks a collapsed
   encoder would show), and the representations remain well separated for classification (the
   trained/adapted encoder clears a hand-crafted 50-dimensional spectral-statistics baseline on
   subject-disjoint nearest-neighbour classification across every training source tested). The
   more likely explanation is that free-living wrist motion genuinely *is* close to stationary
   over sub-second-to-few-second spans much of the time — two-thirds of sampled windows were in
   the stillest tercile of motion energy — so "the future looks like the present" is often an
   honest property of the signal, not a degenerate shortcut. That said, it does mean the
   predictive objective, as specified, has little to predict on most of its training data; see
   the motion-weighting fix in `FRONTEND_PLAN_20260912.md` / the JEPA objective fix commit
   (`d71d24f`, "Strengthen future JEPA temporal objective").
3. **This matches the published picture for the regime we're actually in.** A recent controlled
   comparison of SSL objectives on time series (arXiv:2605.19462) found plain JEPA often sits
   *below* a no-pretraining linear-probe baseline on classification, with masked reconstruction
   and a regularised JEPA variant (LeJEPA) the two consistent winners; the closest published
   benchmark to our setting (BenchHAR, arXiv:2605.08296 — cross-dataset sensor HAR, 8 SSL methods
   × 12 architectures) found a *hybrid* of reconstruction and contrastive pretraining wins
   overall, and — notably — that unlabeled data from activity classes *outside* the downstream
   task does not improve generalization, which is close to our own corpus design (a label-free
   pretraining corpus deliberately disjoint from the supervised training sources). Several of
   these same works, and the strongest published free-living wearable SSL result we found
   (ssl-wearables / Yuan et al., 700k person-days of UK Biobank accelerometry), are explicit that
   predictive/self-supervised pretraining earns its keep specifically in the *no-labels-at-all*
   regime. **We are not quite in that regime**: our support-conditioned few-shot adaptation
   training requires labelled support sets, so some labelled data is a hard requirement of the
   downstream method regardless of whether pretraining is used. Once that labelled training signal
   exists, the literature's own account of *when* JEPA helps predicts a smaller — and easily
   noise-sized — incremental gain from adding label-free pretraining on top, which is what we
   measured.

**Caveat on the numbers above, stated because a fix is mid-flight:** the table reflects the JEPA
objective *before* the 2026-09-12 fixes (commit `d71d24f` — removing an absolute-position leak
from the predictor's query, adding time-to-boundary information to the predictor's context, and
per-query loss weighting toward higher-motion targets). A second full pretraining run under the
fixed objective was started the same day and was not complete as of this entry. If that changes
the picture materially, it will be recorded in a new dated journal entry rather than edited into
this one.

**This is not "JEPA is useless."** It is implemented, it runs cleanly (no collapse, healthy
gradient/rank telemetry), and the literature review above is explicit that it is one of the
stronger available options specifically for the zero-label regime — worth keeping and revisiting
once the encoder-depth and motion-weighting issues are addressed, and worth stating plainly as a
negative result in the write-up rather than quietly dropping.

## Two encouraging results from the same sweep

1. **The enrollment curve now extends to k = 32, 64, 128** (previously reported only to k = 8),
   and HALO's advantage over every released baseline *grows* with k rather than shrinking. At
   k = 128, ridge readout: HALO (multi-span, JEPA-pretrained + adapted) 80.3 mean macro-F1 versus
   69.6 for the strongest released baseline (UniMTS) and 69.6 for HARNet — roughly a 6–11 point
   margin that widens from k = 1 (55.5 vs 54.3) through k = 128. This is a genuinely strong result
   independent of the JEPA question above, and it holds across the full six-dataset sealed roster
   with execution-disjoint support.
2. **HALO's trainable parameter count is small relative to the baselines it's beating.** The
   encoder trunk actually producing these numbers is 2.7M parameters (multi-span arm) to 3.2M
   (fixed arm) — confirmed by direct count of `requires_grad` parameters in the built checkpoints.
   A trainable-parameters-vs-performance plot is worth building for the paper/slides: informally,
   the next-best baseline (UniMTS) is recalled as roughly two orders of magnitude larger
   (~60M, *not yet independently verified in this repo* — needs a confirmed count before it goes
   in a table), which would make the parameter-efficiency story a second, separate contribution
   worth stating alongside the accuracy numbers. **Action item:** verify and tabulate exact
   trainable-parameter counts for every baseline adapter (HARNet, UniMTS, NormWear, LiMU-BERT-X)
   the same way it was done for HALO, before this claim is used in any figure.

One scoping note for both results above: they were produced by the **parameter-free readouts**
(ridge regression fit per episode, differentiable-neighbours soft vote, 1-NN) on frozen or
lightly-adapted encoder features — **not** by the trained semantic token-mixer classifier
(`model/support/token_mixer.py`), which exists in code and passes its tests but has not yet been
the subject of dedicated experimentation. Improving the classifier side of the system is
essentially unstarted work as of this entry.

## Encoder design: still open

The continuous-kernel design's premise — joint time- and frequency-domain resolution, rather than
the frequency-only decomposition of the fixed filterbank — is the right idea in principle and is
still being worked out in practice, on two fronts:

- **Where to add trainable capacity, and how to aggregate it into more abstract features.**
  Measured: the kernel decomposition currently feeds attention directly (one spectral column per
  frame → one linear projection → transformer), with no local, sub-second compositional stage in
  between — the equivalent of using wav2vec's learned filterbank without any of its seven
  convolutional layers on top. Audio/vision frontends earn their depth partly from very high
  sampling rates giving room to stride down repeatedly; at 20–240 Hz that specific mechanism
  doesn't transfer, but the *depth* itself still can, reshaped: a small stack of dilated,
  non-strided convolutions operating on the frontend's physical-time frame grid (rate-invariant
  by construction, since frame spacing is in seconds) rather than on raw samples. This is
  specified in detail, with an integration plan and a four-row attribution experiment to
  determine whether it (and/or the learnable kernels themselves) actually earns its complexity,
  in `docs/design/FRONTEND_PLAN_20260912.md`.
- **Training the added parameters stably.** Directly measured (gradient probes, not inference):
  the continuous kernels *do* receive nonzero gradient at every step, so this is not a dead- or
  blocked-gradient bug. But once the encoder is otherwise trained, those gradients are close to
  sign-inconsistent across batches (batch-to-batch gradient cosine ~0.1, versus ~0.3 for the rest
  of the encoder), so under Adam they resolve into a small, roughly LR-bounded random walk around
  initialization rather than sustained learning — consistent with a known phenomenon in the
  learnable-filterbank literature (learned audio filterbanks reported to move little from
  initialization across several independent studies). From a random initial encoder, the same
  kernel parameters *do* receive consistent, learnable gradient (80%+ of elements with acceptable
  signal-to-noise), so the mechanism is not fundamentally broken — the pinned learning rate and an
  anchor-to-initialization regularizer on the kernels are the more likely proximate cause, and are
  first on the fix list.

Bottom line for this entry: we do not yet have the feature-extraction capability we want from
either arm, in a specific and now-diagnosed sense — the fixed arm is deliberately and
defensibly a compact engineered-features control, while the continuous-kernel arm's differentiator
(learnable, jointly time-and-frequency-resolved features) is not yet demonstrated to be doing
anything beyond what a frozen, denser sampling of the same kernel family would do. Settling that
is the immediate next block of work; see `docs/design/FRONTEND_PLAN_20260912.md` for the concrete
plan.

**Related:** `docs/design/CONTINUOUS_KERNEL_FRONTEND.md`, `docs/design/JEPA_PRETRAINING_OBJECTIVE.md`,
`docs/design/EXPERIMENT_ROADMAP.md`, `docs/design/FRONTEND_PLAN_20260912.md`, memory notes
`halo-jepa-run-analysis-20260912`, `halo-continuous-kernel-sweep-20260912`,
`halo-ssl-literature-verdict-20260912`.
