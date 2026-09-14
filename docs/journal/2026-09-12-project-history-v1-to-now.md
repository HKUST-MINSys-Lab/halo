# Project history: HALO v1 → v2 → clinical pivot → current design

**Date:** 2026-09-12
**Status:** immutable retrospective entry — see `docs/journal/README.md` for the journal's rules.
**Purpose:** orient a new reader (agent or human) on how the project arrived at its current
design, and give the motivation/contribution framing needed for a supervisor-meeting slide. This
intentionally does not go earlier than HALO v1; it is a summary, not a full archaeology. Sources:
`docs/HISTORY.md`, `docs/design/DESIGN_OF_RECORD.md`, and the memory notes cited inline.

## v1 — language-aligned open-set HAR ("HALO" / "TSFM")

The original design: an IMU encoder aligned to a frozen text embedding space, doing open-vocabulary
activity recognition by cosine similarity against label-text embeddings. Code name `TSFM`, paper
name `HALO` ("Heterogeneity-Aware Language-aligned Open-set" IMU foundation model) — the two names
refer to the same system. Headline configuration: TSFM-Small-Deep, d=384, 8 layers, ~35M trainable
parameters.

**Motivation / contribution, as pinned down before v1's submission:** open-vocabulary recognition
of *unseen labels* alone is not a sufficient contribution — ConSE (Norouzi 2014) already bridges a
closed classifier to unseen label text, and every closed-vocabulary baseline in the comparison was
given ConSE for fairness, so "we can name new activities" invites the reviewer response "ConSE
already does that." The claim that survived scrutiny was narrower and two-sided: **one natural-
language interface for both what to recognize (unseen labels) and how the signal was acquired
(unseen sensor configuration/placement/transform)**. Input-side conditioning in language is what
ConSE cannot do (it only bridges the output side) and what a fixed configuration one-hot cannot do
(it only covers configurations seen at train time). The clinching experiment was to show a model
*told* about a real acquisition transform (rotation, gravity present/removed, placement, rate) at
test time holds accuracy where an equally-trained baseline without that information degrades.

**Outcome:** submitted to MobiCom 2026 (#1698), rejected (5 reviewers, mean merit 2.4).

## v2 — clean rebuild, retrieval and evidence-engine experiments

**2026-07-12:** the v1 codebase (`legacy_code/`, formerly `code/`) had accumulated enough
complexity that the project restarted from scratch in a new, concern-oriented repo
(`/home/alex/code/HALO/halo`, private, `github.com/alxdofficial/halo`). The core v1 thesis (one
language interface for label- and configuration-openness) was retained; the data pipeline,
baseline adapters, and evaluation harness were rebuilt cleanly and re-verified end to end (12
datasets, 6 baseline adapters, subject-disjoint leakage-safe scoring).

From that foundation, v2 explored a sequence of architectures beyond the pure text-alignment
model: explicit admissibility scoring, a retrieval/"evidence engine" design (salient-not-invariant,
configuration-conditional retrieval over a learned support bank), and a Phase-B vector/vote
mechanism. This line of work is preserved as an immutable reference (tag/branch
`phaseb-vector8-vote-20260824`) but is no longer live.

## The clinical / movement-monitoring pivot (abandoned)

For a period the project pivoted toward a three-task movement-monitoring application aimed at
clinical populations (Parkinson's motor symptoms, shoulder physiotherapy, hemiparesis arm use —
`applications/` package, Tasks 0–3, evaluation on MoniPar/SPAR/upper-limb-use).

**Supervisor decision, meeting 2026-09-03:** the pivot was dropped. Clinical venues expect a
technique tailored to one disease with matched data, not a generic ML method applied across
3–4 conditions; that framing was judged unlikely to land. The direction set instead was: take an
earlier HALO design (v1 or v2), improve it incrementally, clean up the experimental protocol, and
target a venue friendly to incremental, well-evaluated work (IMWUT) rather than a from-scratch
architecture pitch.

This application-pivot work is preserved as an immutable archive (tag
`archive-pre-classifier-cleanup-20260911`) and is not part of the current design. The three
clinical datasets were subsequently dropped from the evaluation roster entirely — the live
question is general human activity recognition, not a clinical readout.

## Current design — support-conditioned heterogeneous HAR

**Consolidated 2026-09-11** (commit `23665e9`, "Consolidate repository around support-conditioned
HAR") into the one live system this journal's sibling entries describe in detail. In brief:

- Encode a query recording and a small set of labelled support recordings with a shared HALO
  encoder (rate-aware, physically grounded — see the encoder-arms discussion in the companion
  entry).
- Compare query and support in representation space; aggregate support evidence into candidate
  scores (a differentiable-neighbours soft vote today; a trained semantic token-mixer classifier
  is built and available but not yet the focus of experimentation).
- Optional label-free Future-JEPA predictive pretraining for any encoder arm, evaluated as a
  genuinely separate stage rather than baked into the design.

**Motivation, restated for this design (`docs/design/DESIGN_OF_RECORD.md`):** "The intended claim
is not that language alone identifies any action. It is that a rate-aware, physically grounded
encoder and an explicit support comparison can provide useful recognition when the acquisition
configuration and support availability are honestly specified." This is a narrowing from v1's
pure zero-shot text-alignment framing toward a support-conditioned, few-shot-capable system whose
central evaluation axis is the enrollment curve (k = 0, 1, 2, 4, 8, 16, 32, 64, 128) under
execution-disjoint, honestly-disclosed acquisition heterogeneity — not open-vocabulary zero-shot
alone. The v1/v2 concern with configuration-openness survives as the acquisition-heterogeneity
axis (compatible / near-miss / unfiltered sampling modes over device family, site, channels,
gravity state); what changed is that recognition is now support-conditioned rather than
text-only.

**Protocol discipline adopted alongside this design (2026-09-11, reaffirmed):** three disjoint
source roles — label-free pretraining, supervised support-classifier training, sealed test — with
**no development-data split**. All operating decisions (checkpoints, thresholds, candidate
distributions) are fixed a priori or read from training data; the sealed roster is touched once,
after those choices are frozen. This is a deliberate, stricter departure from the three-way
train/dev/test convention used in some of the v2-era work.

## Where this leaves the paper's contribution

The throughline from v1 to now is "make recognition robust to acquisition heterogeneity that is
*disclosed*, not assumed away by invariance." What has changed across the versions is the
mechanism: v1 argued language conditioning does this; the current design argues a physically
grounded, rate-aware encoder plus explicit support comparison does this, with language playing a
narrower, auxiliary role (channel/acquisition description, not the primary recognition path). See
`docs/design/DESIGN_OF_RECORD.md` and the companion journal entry
`2026-09-12-jepa-and-encoder-findings.md` for where the encoder-design and pretraining questions
stand as of this writing.

**Related:** `docs/HISTORY.md` (artifact-level index of the eras above), memory notes
`halo-new-repo`, `halo-core-thesis`, `halo-tsfm-naming`, `halo-imwut-direction`,
`halo-support-conditioned-consolidation-20260911`.
