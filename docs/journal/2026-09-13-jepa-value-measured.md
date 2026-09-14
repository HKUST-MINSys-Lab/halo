# 2026-09-13 — What JEPA pretraining buys, measured

**Status:** result. Sealed evaluation, manifest-matched across all arms. This entry supersedes the
*speculative* parts of [2026-09-12-jepa-and-encoder-findings.md](2026-09-12-jepa-and-encoder-findings.md)
(which said JEPA "does not beat end-to-end training beyond noise" before the four-arm ladder had
been read) by putting numbers on it. The conclusion is unchanged but the picture is more
interesting than "it does nothing".

**Source:** `training/support_classifier/evaluations/jepa_representation_20260912/`.
Six sealed streams (inclusivehar/phone_waist, motionsense/phone_front_pocket,
realworld/phone_waist, shoaib/phone_right_pocket, usc_had/phone_hip, ut_complex/watch_wrist),
`k = 1, 2, 4, 8, 16, 32, 64, 128`, macro-F1, subject-bootstrap CIs. All five arms share identical
per-cell manifests and each has a single feature fingerprint, so the rows are directly comparable.
Only the fixed multiresolution filterbank is shown — the continuous-kernel/multispan arm was
demoted to ablation on 2026-09-12 and is omitted deliberately.

---

## 1. The headline table

Mean macro-F1 over the six sealed streams, differentiable-neighbours readout (the parameter-free
scoring rule the encoder-only arms are actually trained under):

| arm | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random frozen | 50.8 | 53.4 | 54.6 | 56.5 | 58.7 | 59.9 | 60.9 | 61.3 | 57.0 |
| **JEPA frozen** | 54.0 | 59.6 | 63.5 | 67.2 | 69.8 | 71.3 | 72.1 | 72.5 | **66.3** |
| random adapted 5k | 54.4 | 60.1 | 64.7 | 68.9 | 71.9 | 73.5 | 74.4 | 75.1 | 67.9 |
| **JEPA adapted 5k** | 55.7 | 61.1 | 65.6 | 69.5 | 72.2 | 73.9 | 74.7 | 75.6 | **68.5** |
| end-to-end 35k | 56.8 | 62.3 | 66.7 | 70.1 | 72.9 | 74.2 | 75.2 | 75.5 | 69.2 |

Ridge readout, same arms:

| arm | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random frozen | 42.2 | 45.8 | 49.2 | 54.1 | 58.7 | 62.9 | 66.6 | 69.9 | 56.2 |
| **JEPA frozen** | 53.5 | 59.2 | 63.7 | 69.2 | 73.0 | 76.1 | 77.6 | 79.3 | **69.0** |
| random adapted 5k | 53.7 | 59.5 | 64.6 | 69.3 | 73.6 | 76.2 | 77.6 | 79.2 | 69.2 |
| **JEPA adapted 5k** | 55.5 | 61.6 | 66.2 | 70.7 | 74.4 | 76.6 | 78.2 | 79.6 | **70.3** |
| end-to-end 35k | 56.0 | 61.9 | 67.5 | 71.9 | 75.7 | 77.5 | 79.1 | 79.8 | 71.2 |

1-NN readout, means only: random frozen 65.0, JEPA frozen 67.2, random adapted 67.3, JEPA adapted
67.8, end-to-end 69.1.

## 2. What JEPA helps with

**It substantially improves a frozen representation, and the effect is consistent.**
Paired per-stream deltas, JEPA-frozen minus random-frozen:

| readout | overall Δ macro-F1 | sd | streams improved |
|---|---:|---:|---|
| differentiable-neighbours | **+9.24** | 7.90 | **6/6** |
| ridge | **+12.80** | 6.44 | **6/6** |
| 1-NN | +2.21 | 3.31 | 5/6 |

Every sealed stream improves under both parametric readouts, with per-stream gains up to +19.3
(ut_complex, ridge) and +18.4 (ut_complex, differentiable-neighbours). This is not a noise-level
effect and it is not driven by one dataset. If the deployment constraint were "encoder must stay
frozen", JEPA would be clearly worth keeping.

**Interpretation (reasoning, not measured):** note the readout dependence. The 1-NN gain is +2.2
while ridge is +12.8. 1-NN is scale-free — only the *ranking* of neighbours matters — whereas ridge
and the temperature-scaled neighbour vote depend on the *geometry* of the space. JEPA's large gains
land almost entirely on the readouts that care about geometry. The most likely reading is that
JEPA mainly fixes the **conditioning** of the embedding space (isotropy, usable scale, the VICReg
effect — encoder effective rank rises 12 → 101 during pretraining) rather than discovering better
neighbourhood structure. The +2.2 on 1-NN is the part that is genuinely better neighbours.

## 3. What JEPA does not help with

**Once the encoder is trained at all, the advantage disappears.** Paired per-stream deltas,
JEPA-adapted minus random-adapted, both 5,000 neighbour-only steps from the same data seed:

| readout | overall Δ | sd | streams improved | worst stream |
|---|---:|---:|---|---|
| differentiable-neighbours | +0.67 | 3.88 | 4/6 | realworld −5.21 |
| ridge | +1.15 | 4.08 | 4/6 | realworld −3.81 |

Two things make this uninterpretable as a real gain:

1. The magnitude sits at or below the established screening noise floor (sd 0.0065 on the
   3k-step screen ⇒ nothing under ≈1.2 points is real; see [[halo-phasea-noise-floor]]).
2. **The sign flips across sealed streams** — realworld −5.2 and inclusivehar −3.0 against
   usc_had +5.3 and ut_complex +3.3, with a between-stream sd (3.9–4.1) six times the mean.

That signature has appeared in this project before and meant the same thing both times: commit
`9b7d75d` found acquisition-config conditioning contributed "+0.0086 kNN-BA against a 0.0065 noise
floor, and the sign flips on two of four held-out datasets," and concluded the architectural claim
did nothing measurable. Same pattern here.

**And end-to-end training is not beaten by JEPA-warm-started training.** End-to-end 35k minus
JEPA-adapted 5k: +0.68 (differentiable-neighbours) and +0.84 (ridge), 4/6 streams — also within
noise, and in the *other* direction. Honest reading: the three trained routes (random-adapted,
JEPA-adapted, end-to-end) converge to the same place. The caveat is that the end-to-end arm ran
35,000 steps against 5,000 for the adapted arms, so this row is not step-matched and should not be
cited as "end-to-end wins" — only as "warm-starting from JEPA does not win."

## 4. The one-sentence summary

**JEPA buys roughly 9–13 macro-F1 points of representation quality that the supervised neighbour
objective then reproduces on its own within 5,000 steps.** It is a substitute for early supervised
training, not a complement to it. Because we always have some labels for few-shot adaptation, we
are paying a full pretraining run for something the adaptation step gives us free.

## 5. What this experiment does *not* answer

* **Zero-shot is entirely unmeasured.** Every `k=0` / `retrieve-mix-vote` cell in all arms is
  `n/a` — "HALO token-mixer readout requires a current support-classifier checkpoint". So we have
  **no evidence at all** on whether JEPA helps zero-shot classification. Given JEPA has no text in
  its objective there is no mechanism by which it should, but that is an argument, not a result.
* **Label efficiency at very small k is not isolated.** JEPA-adapted leads at k=1 by +1.3 on every
  readout, which is the largest relative margin in the table. It stays inside the noise band, but
  if a low-k claim ever matters, that is the cell to power properly rather than read off this run.
* **Only one pretraining recipe was tested**, at one scale, on one frontend. "Future-JEPA as
  configured on 2026-09-12 is not worth its cost" does not generalise to "label-free pretraining
  cannot help here". The cross-placement variant discussed on 2026-09-13 is untested.

## 6. Decision this supports

Keep the ladder result, stop spending on future-JEPA for the fixed arm, and report it as a
measured negative with a measured positive inside it: *label-free pretraining gives a frozen
encoder most of what supervised adaptation would give it, and nothing beyond that.* That is a
publishable observation about the regime, and it is more useful than a null.

## 7. Links

* [2026-09-12-jepa-and-encoder-findings.md](2026-09-12-jepa-and-encoder-findings.md) (earlier, pre-numbers)
* [2026-09-13-why-a-simple-encoder.md](2026-09-13-why-a-simple-encoder.md) (why the ceiling exists)
* `docs/design/EXPERIMENT_ROADMAP.md` rows 1–4; `docs/design/JEPA_REPRESENTATION_EVALUATION.md`
