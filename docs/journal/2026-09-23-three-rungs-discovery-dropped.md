# Addendum: the discovery rung is dropped; three rungs, renumbered

Date: 2026-09-23. Status: **decision record; nothing run.** Supersedes the numbering in
[the 09-22 evening entry](2026-09-22-three-regimes-and-unsupervised-adaptation-decisions.md), its
[addendum](2026-09-22-four-rungs-and-rung2-head-correction.md), the
[implementation plan](2026-09-22-rung1-rung2-implementation-plan.md) and the
[implementation record](2026-09-23-rungs-implementation-record.md); their content stands under
their own numbering. Live plan: [`docs/overview/roadmap.md`](../overview/roadmap.md).

## The decision

Alex, 2026-09-23: *"rung 1 is not interesting enough. Let's get rid of it and make rung 2 rung 1."*
The discovery readout — roster and K unknown; k-means / Ward on frozen features; AMI, ARI, RSA
against label-text geometry; three-way cluster naming — is dropped from the paper.

| old (four rungs, 09-22) | new (three rungs, 09-23) |
|---|---|
| 1 — discovery | **dropped**; code retained, retired |
| 2 — unlabelled N-curve | **1** |
| 3 — labels, parameters frozen (case study) | **2** |
| 4 — labels, fine-tuning allowed | **3** |

What rung 2 (new numbering) contains, in Alex's words: the existing learnable classifier, the
aggregate k-curve and the per-scenario results, compared against the baselines using equal-weight
normalised fusion — as a case study.

## What changed in the repo

- `evaluation/rung2_unlabeled/` → `evaluation/rung1_unlabeled/`; `evaluation/rung4_finetune/` →
  `evaluation/rung3_finetune/`; `evaluation/rung1_discovery/` → `evaluation/discovery/` (retired,
  runnable, not part of the paper). Moves are `git mv`, so history follows.
- `evaluation.provenance.Rung`: `UNLABELED = 1`, `FROZEN = 2`, `FINETUNE = 3`; `DISCOVERY = 0`
  kept so the retired readout still runs and can never be mistaken for a rung of the paper.
- Entry points: `halo-rung1` (unlabelled), `halo-rung3` (fine-tune), `halo-discovery`.
- Tests renamed to match; docstrings renumbered; the full suite green after the move.
- `docs/overview/roadmap.md` and `thesis.md` rewritten for three rungs. Journal entries are not
  edited; each keeps the numbering it was written under, and the roadmap header says how they map.

## Notion

The 2026-09-07 ⭐ Latest series (`imwut/compare`, Arm A / Arm B, "no HALO training run exists")
was archived under 🗄️ Archive as *2026-09-07 — compare-arm plan (pre-rungs)* with a supersession
banner; its page 4 had already been deleted and was not restored. ⭐ Latest was rewritten as a new
four-page series under the three-rung numbering, from the repo's documents of record (thesis,
roadmap, RESULTS.md, the ceilings ladder). It is the shareable version of this plan.
