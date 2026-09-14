# 2026-09-13 — Why a deliberately simple encoder: the design rationale

**Status:** rationale, not a result. This entry records *why* we chose a fixed, engineered,
low-parameter frontend over a learned deep encoder, so the argument can be reused in a paper or a
supervisor meeting without re-deriving it. The empirical entries it rests on are
[2026-09-12-jepa-and-encoder-findings.md](2026-09-12-jepa-and-encoder-findings.md) and
[2026-09-12-fixed-filterbank-decision.md](2026-09-12-fixed-filterbank-decision.md).

---

## 1. The observation that started it

Across our own probes, the learned encoder repeatedly fails to beat classical engineered features.
The subject-disjoint hand-crafted floor (50 standard time/frequency features, balanced accuracy)
is **0.82 / 0.78 / 0.70 / 0.61 / 0.56** on hhar / kuhar / realdisp / dsads / wisdm, and it beats a
randomly-initialised encoder on *every* stream. That floor has since become the acceptance
criterion for frontend changes, precisely because clearing it is not automatic.

This is not a local anomaly. It is the normal state of the field, and it has a reason.

## 2. The argument: the learnable surface is small

The clearest way to see what a HAR encoder actually has to do is a thought experiment.

Suppose you could take raw recordings from an IMU at an arbitrary body location and reliably
translate them into a statement about the *skeleton*: which limb is involved and how it is moving,
in primitive terms — "forearm swings forward", "wrist flexes", "trunk rotates". Emit a sequence of
those descriptors and you are done: a language model can infer the activity from that sequence
with little difficulty, and can do so for activities it has never seen, because the inference is
ordinary commonsense reasoning over a physical description rather than pattern matching over a
fixed label set.

So the job of the sensor-side model is: **raw signal → a faithful, low-dimensional description of
how the body is moving.** Nothing more. That target is far smaller than "learn a representation of
human activity", and it is mostly a *physics* problem — orientation relative to gravity, which
frequencies carry energy, whether motion is linear or rotational, how the segments co-move. These
are quantities with closed forms. They do not need to be discovered from data.

This reframes the encoder's purpose and is the load-bearing argument of the current design: we are
not trying to learn *what activities look like*; we are trying to measure *what the body is doing*,
in a way that survives a change of sensor. Measurement wants correct physics and invariance, not
capacity.

## 3. Why we do not train IMU→pose explicitly

The obvious response is "then train a pose estimator." We deliberately do not, and the reason is
**hallucination**.

Reconstructing a full skeleton from a handful of body-worn IMUs is severely underdetermined — six
sensors cannot observe 20+ joint angles. The literature resolves this with strong learned motion
priors:

* Y. Huang, M. Kaufmann, E. Aksan, M. J. Black, O. Hilliges and G. Pons-Moll, "Deep Inertial
  Poser: Learning to reconstruct human pose from sparse inertial measurements in real time,"
  *ACM Transactions on Graphics* (SIGGRAPH Asia), 2018. arXiv:1810.04703,
  DOI [10.48550/arXiv.1810.04703](https://doi.org/10.48550/arXiv.1810.04703)

Those priors are exactly the problem for us. They are trained to produce poses that look
*plausible*, and a plausible pose is not a measured pose. The failure mode is silent: the model
emits a confident, natural-looking skeleton that is wrong, and nothing downstream can detect it.
For animation that is acceptable and even desirable. For recognition it is poison — we would be
classifying the prior's imagination rather than the subject's movement, and the error would
correlate with the prior's training distribution, i.e. exactly with the populations and activities
already well represented.

So the design keeps the *motivation* of the skeleton view (describe the movement physically) while
refusing the *mechanism* (generate a skeleton). We emit measured physical descriptors with known
semantics and known failure modes, and let the comparison happen in representation space.

## 4. Why added capacity is actively harmful here

Two reasons, one statistical and one physical.

**Statistical.** The problem is label-scarce and the heterogeneity axes are exactly the axes a
high-capacity encoder will latch onto. Every additional learnable feature extractor is additional
opportunity to fit subject identity, device identity and session artefacts — all of which are
perfectly predictive within a dataset and useless across one. We have measured this failure
directly: within-execution random-split balanced accuracy is 0.98 for *every* feature set, so any
non-subject-disjoint number is void, and the encoder will happily reach that 0.98 by memorising
the wrong thing. Capacity buys overfitting before it buys recognition.

**Physical.** Changing the sensor model, the mounting orientation, the body location or the
sampling rate makes the numbers look completely different while the underlying movement is
unchanged. A learned extractor has no way to know which differences are physical and which are
instrumental; it has to be *taught* that, from data that does not densely cover the product space
of configurations. Engineered descriptors can be given the invariance by construction — this is
what the constant-Q filterbank does for sampling rate, and what the gravity-referenced
polarization features of
[2026-09-13-filterbank-polarization-features.md](2026-09-13-filterbank-polarization-features.md)
do for mounting orientation. Invariance obtained by construction costs no parameters and cannot be
unlearned.

The general version of "simple beats complex when the task is more constrained than the
architecture" is by now well established in time series:

* A. Zeng, M. Chen, L. Zhang and Q. Xu, "Are Transformers effective for time series forecasting?"
  *AAAI 2023*, 37(9):11121–11128.
  DOI [10.1609/aaai.v37i9.26317](https://doi.org/10.1609/aaai.v37i9.26317)
  — one-layer linear models beat every published Transformer LTSF model, "in all cases, and often
  by a large margin." Cite for the general principle, not for HAR specifically.

And within our own field, the systematic assessment of the pretrain-then-finetune paradigm is:

* H. Haresamudram, I. Essa and T. Plötz, "Assessing the state of self-supervised human activity
  recognition using wearables," *Proc. ACM IMWUT*, 6(3):1–47, 2022.
  DOI [10.1145/3550299](https://doi.org/10.1145/3550299)
  — seven state-of-the-art SSL methods across nine benchmarks, framed explicitly against "the
  classic activity recognition chain" of hand-crafted features. Same venue we are submitting to;
  this is the reference for "SSL in HAR has not settled the question."

## 5. The external work that took the same route: ZARA

* Z. Li et al., "ZARA: Training-free motion time-series reasoning via evidence-grounded LLM
  agents," *ACL 2026*, pp. 14985–…, [aclanthology.org/2026.acl-long.684](https://aclanthology.org/2026.acl-long.684/).
  Earlier version: "ZARA: Zero-shot motion time-series analysis via knowledge and retrieval
  augmented LLM agents," [arXiv:2508.04038](https://arxiv.org/abs/2508.04038).
  Code: [github.com/cruiseresearchgroup/ZARA](https://github.com/cruiseresearchgroup/ZARA)

ZARA does not train an encoder on the classification objective at all. It extracts ~40
"low-cost, human-interpretable statistics" — time-domain (mean, variance, RMS, signal-magnitude
area), frequency-domain (dominant frequency, spectral entropy, band power) and **cross-channel
(channel correlations, tilt angle)** — builds a pairwise knowledge base of which statistics
discriminate which activity pair, retrieves evidence from a labelled support set, and has an LLM
reason over the retrieved evidence. Its stated motivation is ours: applying LLMs "directly to
numerical time-series often leads to hallucinations and weak grounding," so the fix is to
translate signal statistics into explicit, verifiable language rather than rely on "black-box
projections."

Three things in it matter to us directly:

1. **It wins, decisively, without a trained classifier.** 81.6% mean accuracy / 81.4 macro-F1
   across eight HAR benchmarks, against UniMTS at 39.4/32.1 and IMU2CLIP at 22.7/17.9 —
   "exceeding the strongest baselines by 2.53× in macro F1."
2. **Model scale is not where the gain comes from.** Their backbone ablation runs Qwen-30B →
   GPT-4.1-mini → Gemini (71.0 → 77.5 → 81.6) and concludes the gains "originate from the
   knowledge and retrieval-augmented framework design rather than LLM backbone scale," with every
   backbone beating every baseline. This is external support for our own observation that
   parameter count is not the axis that matters here.
3. **It independently diagnoses the weakness of our baselines.** On UniMTS and NormWear — two of
   our comparison models — it reports performance that "degrades sharply on unseen activities,
   revealing a strong dependence on label exposure." That is the same open-set brittleness our
   thesis targets, found by an unrelated group.

Its architecture is also structurally close to ours — statistics rather than a learned classifier,
class-wise retrieval over a labelled support set, evidence aggregated to a decision — which is
reassuring convergence rather than a threat, since our contribution is the heterogeneity-invariant
*measurement* side and a trained comparison, not the agentic reasoning side.

### The honest wrinkle

**ZARA is not evidence that learned encoders are useless, and we must not cite it that way.** Its
own retrieval-embedder ablation is the opposite: swapping DTW for the pretrained Mantis time-series
encoder moves mean accuracy **71.0 → 81.6**. More than ten points of their headline number comes
from a *learned* encoder used as a retrieval embedder. Moment-small and Moment-large land in
between (79.4, 80.8).

That is precisely our position, stated by someone else: the learned encoder earns its keep as a
**retrieval/comparison embedder**, not as a classifier trained on a closed label set. It is also
consistent with our own history — [halo-phaseB-m4a-results] and [halo-phaseB-tier2-build] both
found the trained closed-vocabulary decoder net-negative against an untrained retrieval control.
So the lesson is not "do not learn"; it is "learn the metric, not the label map."

## 6. What this rationale does *not* license

Recorded so a future reader does not overextend the argument:

* **It is an argument for a small learnable surface, not for zero learning.** Our trunk still
  learns composition, folding and pooling, and ZARA's own ablation says that part is worth ~10
  points.
* **The descriptor vocabulary is a real limit.** The thought experiment in §2 assumes the movement
  can be described in language. Our own labelled-corpus audit found label text cannot express
  direction or side (see [[halo-labelled-corpus-audit]]), which caps how far the "describe it and
  let an LLM reason" route can go — and is part of why the clinical pivot was abandoned.
* **The parameter-count comparison is still unverified.** The claim that the next-best baseline has
  ~60M parameters against our 2.7–3.2M trunk has *not* been checked against materialised
  checkpoints; it was flagged as an action item in the 2026-09-12 entry and remains open. Do not
  put it in a paper until it is measured.
* **§4's statistical argument is reasoning, not a measurement.** We have not run a capacity sweep
  showing degradation with added parameters. The 0.98 within-execution figure shows the *hazard*
  exists; it does not show that our specific added capacity fell into it.

## 7. How this composes with the JEPA finding

The JEPA result and this rationale are the same argument seen from two sides. A frozen, engineered
frontend means a predictive pretraining objective can only reshape how patches are *combined* — it
cannot enrich what each patch measures, because that is fixed by construction. So the ceiling on
what pretraining can contribute is set by the same decision that makes the encoder robust. We
accepted that trade deliberately: a smaller learnable surface that generalises across
configurations, rather than a larger one that needs data we do not have to learn invariances we
can write down.

## 8. Links

* [2026-09-12-jepa-and-encoder-findings.md](2026-09-12-jepa-and-encoder-findings.md)
* [2026-09-12-fixed-filterbank-decision.md](2026-09-12-fixed-filterbank-decision.md)
* [2026-09-12-spectral-frontend-literature.md](2026-09-12-spectral-frontend-literature.md)
* [2026-09-13-filterbank-polarization-features.md](2026-09-13-filterbank-polarization-features.md)
