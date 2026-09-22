# Thesis and scope

Last verified against code: 2026-09-22. Supersedes the classifier-era thesis, archived at
[`docs/archive/thesis-support-conditioned-classifier-era-20260918.md`](../archive/thesis-support-conditioned-classifier-era-20260918.md);
the reasoning is in the
[2026-09-22 decision record](../journal/2026-09-22-three-regimes-and-unsupervised-adaptation-decisions.md).

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

## The claim, in three regimes

The paper is organised from the least to the most information a deployment can provide. The same
six encoders — HALO v4 and five released baselines — are scored in every regime, through the same
inference procedure, so that every difference is attributable to the encoder.

| regime | the deployment provides | the question | what "better" is measured against |
|---|---|---|---|
| **1 — discovery** | unlabelled recordings; roster and K unknown | does the encoder's geometry recover the activities, and does it mirror how their names relate in language? | the baselines' geometry |
| **2 — unlabelled adaptation** | the roster, plus a growing pool of unlabelled recordings | does accuracy rise with unlabelled data, with no label ever provided? | a flat curve, which is exactly per-window zero-shot |
| **3 — labelled adaptation** | k labelled examples per class | with labels and fine-tuning switched on for every model, does HALO still lead, and reach the honest closed-set bar? | ~77–80 % — the fully supervised multi-subject, multi-device, multi-dataset ceiling |

The contribution is **not** any single mechanism. Every inference-time procedure is an established
one, applied identically to all six encoders. The contribution is that HALO is **trained through
the deployment procedure**: its curriculum mirrors the heterogeneity and the information conditions
it will meet — few or no labels, an unlabelled pool from a different acquisition, imbalanced and
partially-covered rosters — so adaptation is a primary training objective rather than a property a
strong encoder happens to have. The few-shot literature states this should be done and names it as
open; nobody has done it on the acquisition axis.

What the system does **not** claim: that language alone describes arbitrary motion; unknown-class
rejection as a headline; or that a foundation model replaces domain data where domain data is cheap.
