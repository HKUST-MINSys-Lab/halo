# Enrollment by demonstration — design notes

> **Status: brainstorm, 2026-07-21.** Nothing here is built or verified. Two load-bearing
> assumptions are explicitly unverified (§8). Branch: `pose-pretext-exploration`.
> Supersedes nothing; this is the direction we landed on after two others were killed.

## 0. What this replaces, and why

Three framings were considered and dropped in one session. Recording them so nobody resurrects them.

| framing | why it died |
|---|---|
| **Zero-shot open-set HAR** (the current pitch) | Our own numbers refute the practical case: supervised linear probe **84.3** vs zero-shot ConSE **39.6** on the same frozen encoder. If a few hundred labels double the score, "no finetuning needed" is a 45-point discount, not a feature. Also: open-set *labels* is table stakes (ConSE, 2014). |
| **IMU → pose as a pretext task** | Killed by literature (`POSE_PRETEXT_LITERATURE.md`). The central hypothesis — pose is a config-invariant space — is backwards: sparse-IMU pose systems resolve the config *first* (IMUPoser classifies which of 5 body locations the phone is in before estimating pose). And the mechanism is already published: IMUCoCo, UIST 2025, drops the pose heads and freezes the encoder for activity recognition, verbatim. |
| **Open-world discovery** ("find unnamed behaviours") | Killed by the obvious objection: everything worth naming has a name. Clustering free-living IMU yields transitions, subject idiosyncrasies and device artifacts. With encoder purity at 0.68, clusters would be dominated by nuisance factors. A cluster that cleanly separates "subject 7" is a failure, not a discovery. |

**What survived the discovery critique** is not discovery but two weaker, sharper claims:

1. **Abstention** — knowing when something is *not* in the vocabulary, instead of silently
   misassigning it. Free-living data is mostly "other" (CAPTURE-24: 0.800 coarse vs **0.576**
   fine-grained), and epidemiological totals are computed by summing classifier outputs over it.
   Retrieval gives a *distance*; a fitted softmax is normalized to always pick something.
2. **Enrollment by demonstration** — the subject of this document.

## 1. The task

> The user demonstrates a movement a handful of times. The system then finds and counts every
> other instance of it in their recordings. No label, no taxonomy, no retraining.

The model must never emit a softmax over a fixed vocabulary. It answers a **verification**
question: *is this segment the same movement as the one I was shown?* — a calibrated same/different
score against an arbitrary enrolled prototype.

**Why verification and not classification, in our own evidence:** both of our biggest failures
(M4a's trained head at 40.9 vs 47.5 untrained; the Tier-2 decoder net-negative twice) were
closed-vocabulary cross-entropy destroying open-vocabulary geometry. A verification objective does
not have that failure mode by construction. **Our negative results predict this should work where
what we tried did not.**

### Applications (chosen for social value, not benchmark convenience)

* **Rehabilitation adherence.** The prescribed exercise is specific to the patient and appears in no
  dataset. Demonstrate once in clinic; count occurrences at home. "Just finetune" genuinely does not
  apply — three examples is not a training set and the concept is defined per-user.
* **Occupational repetitive strain.** The motion has no name in any taxonomy, but *count per shift*
  is what predicts injury.
* Secondary: retrospective cohort queries over archives, where the question is posed after
  collection.

### Division of labour

* **Contrastive training** shapes the *metric* — what "same movement" means.
* **The evidence engine** provides *test-time enrollment* (add exemplars to the memory bank, no
  gradient step) and *calibrated uncertainty* (distance, not a normalized softmax).

Training never sees the rehab exercise. It only learns a metric good enough that three exemplars
suffice at test time.

## 2. Free supervision: motif / periodicity mining

Rehab exercises, yoga holds, chiropractic drills and line work are **repetitive by nature**, and
repetition 3 and repetition 7 of one movement are the same concept executed differently — different
speed, amplitude, fatigue, posture drift. That is exactly the invariance we want, sitting in
unlabeled data for free.

This is a **deterministic data-preparation step, not something learned**. See §7 for the concrete
recipe. It yields, with no annotation:

* **positive pairs** (cycle *i* ↔ cycle *j* of the same motif),
* **cycle boundaries and phase**,
* **ground-truth repetition counts** (we detected the cycles, so we know how many).

So detection, counting and localisation supervision all come from one unsupervised signal.

Contrast with SimCLR-style augmentation positives, which only teach invariance to the augmentations
*we* chose. Repetition positives teach invariance to how humans actually vary a movement.

## 3. Positive-pair hierarchy — one axis per source

The flaw in a naive version: if every positive pair comes from one session, subject/device/placement/
rate/gravity are all **constant**, so the model gets no gradient signal to be config-invariant.
Contrastive learning only buys invariance to what the positives actually vary over.

| pair source | axis it teaches | labels needed | verdict |
|---|---|---|---|
| consecutive cycles, same session | execution variation (speed, amplitude, fatigue) | none | core |
| **simultaneous streams, different placement/device** | **acquisition config** | none (needs time sync) | **core — this is the fix** |
| same subject, different day/session | donning, re-mounting | none (session ids) | core |
| same label, different subject | subject | labels | ⚠️ **weight low or drop** |
| same label, different dataset | everything at once | labels | ⚠️ use sparingly |

**Subject invariance may be actively harmful.** Enrollment is *personalized* — we find *this*
patient's instances. Same-person matching is the easy win. Hard subject-invariance trains away the
specificity the application depends on.

**Cross-dataset positives are the most valuable and least reliable.** Treadmill "walking" and
free-living "walking" share a canonicalized label and are genuinely different movements. Forcing
them together teaches the model to ignore real distinctions. Measure whether they help.

**Negatives** are same-session / same-subject *different movement*, so the only way to win is to
encode the movement rather than the context. **Same-movement-different-config must never be sampled
as a negative** — that bug would train config-*sensitivity* straight back in.

### The multi-stream asset (measured 2026-07-21)

Simultaneously-recorded streams give cross-config positives for free: same movement, same instant,
different placement.

| dataset | streams | subjects present in **all** streams |
|---|---|---|
| **xrf_v2** | 6 — airpods_ear, glasses, left/right_pocket, left/right_wrist | 16 |
| **wisdm** | 2 — phone_pocket, watch_wrist | 51 |
| **sp_sw_har** | 2 — phone_pocket, watch_wrist | 23 |
| **nfi_fared** | 2 — back, wrist | 14 |

Only 4 of 12 datasets. **And we discarded more during curation:** PAMAP2 and MHEALTH each ship
*three* IMUs (chest/hand/ankle, chest/wrist/ankle) in their raw sources; we curated one stream from
each. Recoverable from data already on disk — same story as the dropped magnetometer channels.

## 4. The atom: cycles, not patches, not sessions

A repetition is a **trajectory, not a point**. The bottom and top of a squat are both "squat" and
look nothing alike, so naive patch-to-patch positives force phase-misaligned segments together and
teach the metric nonsense.

* **Session as atom** — too coarse. Contains many activities, varies in length, destroys
  localisation, and localisation is what counting needs.
* **Patch as atom** — hits the phase problem head-on.
* **Cycle as atom** — phase-*complete*, so two cycles are comparable regardless of where each
  starts, and slow vs fast executions both reduce to one atom. ✅

Motif mining supplies cycle boundaries *and* phase, so we can either pool a whole cycle
(phase-invariant by construction) or align patches by phase explicitly. That is temporal
cycle-consistency obtained analytically rather than learned.

```
patches (multi-scale) ──attention within window──> contextualized patches
                                 │
                     attention-pool over one cycle
                                 ▼
                        cycle embedding      ← the contrastive / matching unit
```

* **Detection** matches at cycle level → phase-robust.
* **Counting** comes from cycle boundaries → needs patch resolution underneath.
* **Localisation** comes from patch-level scores → same.
* **Speed variation** → soft-DTW or phase-resampled alignment at patch level.

### Multi-scale patches — supported by our own ablation

Our Phase-A learnable-tokenizer arm lifted held-out transfer 0.801 → 0.824, and attribution showed
the gain came from **multiresolution, not the (inert) learnable filterbank**. Multi-scale is the one
component of that arm that demonstrably worked.

It is also physically necessary: a walking stride is ~1 s, a squat rep ~3 s, a tremor ~0.1 s. Any
fixed patch size is wrong for most movements.

## 5. ⚠️ The session-context shortcut

Contextualising within a session, **combined with same-session positive pairs**, is a shortcut
generator. If a patch embedding absorbs session-level context, any two patches from that session
become trivially similar and the model scores perfectly by recognising the *session*, not the
movement — reintroducing through the architecture the exact failure the sampling was designed to
avoid.

**Rule:** context may inform *how to interpret* a patch (acquisition config, normalization); it must
not carry *what* the patch is. Identity-revealing session information is kept out of that pathway or
explicitly suppressed.

**The useful flip side:** placement, orientation, rate and gravity convention are constant within a
session, so **session-level statistics are a free unlabeled estimator of the acquisition config** —
including a config never seen in training. That is the input-side conditioning claim obtained
without anyone describing the sensor in text.

## 6. Scope

**Train on everything. Evaluate and pitch on phone + watch.**

* Restricting *training* pairs to pocket↔wrist would teach one config transformation rather than a
  general mechanism, and would leave **nothing to hold out** — no unseen config, hence no way to
  test the compositional-generalization thesis at all. The held-out config *is* the experiment.
* xrf_v2's exotic streams are **AirPods and smart glasses** — consumer devices, so including them
  widens the framing defensibly rather than breaking it.
* Hold out at least one config never trained on, ideally a *phone or watch* config so the demo
  matches the deployment story, with exotic placements as a harder secondary check.

The deployment story that makes the thesis concrete:
**enroll the exercise in clinic on a watch; detect it at home on a pocket phone.**

## 7. Evaluation

Protocol: **hold out entire movement types.** Enroll with k = 1 / 3 / 5 exemplars. Report:

* **detection** — average precision
* **counting** — MAE against mined or annotated rep counts
* **localisation** — IoU
* **abstention** — AUROC for "this is not any enrolled concept"
* **negative control** — cluster/match agreement with *subject* and *device* identity. If matches are
  predicted by who or what rather than by movement, the result is void.

Candidate data: **RecGym** (already in corpus — but our notes flag it as min–max normalized, which
corrupts the DC/gravity component and must be checked), **MM-Fit** (exercises with rep counts),
**OpenPack** (occupational, repetitive; what Multi3Net evaluated on).

## 8. Unverified assumptions — check before building

1. **Are the multi-stream datasets time-aligned at the sample level**, or merely from the same
   subjects? §3's entire cross-config mechanism collapses if not. Cheap to check.
2. **How much of the corpus is periodic enough to mine motifs from?** If only walking and running
   yield clean cycles, the mined-concept pool is thin and the objective is under-supplied. Directly
   measurable on the grids.
3. **Prior art**, given how the pose direction went. Few-shot / query-by-example HAR, IMU repetition
   counting, and rehab-adherence sensing all exist in some form. Run the same adversarial sweep
   before committing.

## 9. Related open items inherited from elsewhere

* Our stairs/ramp/elevator weakness is a **modality** problem (altitude), not a representation one —
  the literature says elevator/escalator needs barometer or magnetometer. PAMAP2 and MHEALTH ship
  magnetometer in raw sources we did not ingest.
* The 84.3-vs-39.6 gap is **normal, not pathological**: IMU2CLIP reports 18.46 → 62.52 F1, a
  44.1-point gap, in adjacent rows of one table. Best available citation for that framing.
