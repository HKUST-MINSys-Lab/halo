# Thesis and scope

Last verified against code: 2026-09-24. Supersedes the classifier-era thesis, archived at
[`docs/archive/thesis-support-conditioned-classifier-era-20260918.md`](../archive/thesis-support-conditioned-classifier-era-20260918.md);
the reasoning is in the
[2026-09-22 decision record](../journal/2026-09-22-three-regimes-and-unsupervised-adaptation-decisions.md)
and the [2026-09-23 renumbering](../journal/2026-09-23-three-rungs-discovery-dropped.md).

HALO means **Heterogeneity-Adaptive, Lightweight, Open-vocabulary activity recognition**: a
sub-million-parameter support-conditioned recogniser for wearable IMU whose deployment condition —
device, placement, sampling rate, subject, vocabulary, and the amount and kind of deployment data —
is decided by whoever uses it, not by whoever trained it.

## The regime: arbitrary activity detection

The label vocabulary is defined **after** deployment, by the user or operator. Value grows with the
distance of three things from the model developer's control: the vocabulary, the acquisition, and
the labelled data. When all three are developer-controlled — a fixed vocabulary on one device with
plentiful labels — a closed-set specialist wins, and the paper says so. That concession is what
makes the rest credible.

Framings the paper leads with: embodied AI / learning from demonstration, industrial task steps,
end-user-defined gestures and routines, fitness with trainer-invented movements, and annotation
acceleration. **No clinical or health framing.** A foundation model is not a substitute for
collecting disease-specific data and the paper does not argue otherwise; InclusiveHAR is a
robustness row only.

## The claim, in three rungs

The paper is organised as a ladder from the least to the most a deployment can provide, hardest
rung first. The same six encoders — HALO and five released baselines — are scored on every rung
through the same inference procedure, so that every difference is attributable to the encoder and
how it was trained.

| rung | the deployment provides | the question | what "better" is measured against |
|---|---|---|---|
| **1 — unlabelled adaptation** | the roster, plus a growing pool of unlabelled recordings; no label ever | does accuracy rise with unlabelled data, with no label ever provided? | a flat curve, which is exactly per-window zero-shot |
| **2 — labelled, parameters frozen** | k labelled examples per class; the model may not change | does the encoder beat the baselines under an identical parameter-free readout at every k? | done — the existing sealed and scenario tables; the case study |
| **3 — labelled, fine-tuning allowed** | k labelled examples per class; every model may fine-tune | with the same treatment for everyone, does HALO still lead at every k, and where does it cross a specialist trained from scratch on the same k? | every baseline fine-tuned; the crossover; ~77–80 % as context |

The contribution is **not** any single mechanism. Every inference-time procedure is an established
one, applied identically to all six encoders. The contribution is that HALO is **trained through
the deployment procedure**: its curriculum mirrors the heterogeneity and the information conditions
it will meet — few or no labels, an unlabelled pool from a different acquisition, imbalanced and
partially-covered rosters — so adaptation is a primary training objective rather than a property a
strong encoder happens to have. The few-shot literature states this should be done and names it as
open; nobody has done it on the acquisition axis.

Two limits on that claim, both decided in advance. The adaptation procedure itself may gain only a
handful of learnable constants, initialised at the published values and fitted for every baseline
too — never a learned adapter that reads the pool, which the literature shows is unreliable on real
shifts ([decision](../journal/2026-09-23-rung1-learnability-ceiling-and-fixes.md)). And rung 1's
procedure lets each encoder's own embedding neighbourhoods vote alongside its text scores, so the
rung measures how well an encoder groups the same activity together, not only how good its text
head is ([roadmap](roadmap.md#the-one-established-method)).

What the system does **not** claim: that language alone describes arbitrary motion; unknown-class
rejection as a headline; or that a foundation model replaces domain data where domain data is cheap.
