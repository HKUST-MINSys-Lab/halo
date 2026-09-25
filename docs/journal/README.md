# Project journal

A dated, append-only record of architecture pivots, experiments, findings, and decisions. Historical
entries preserve what was believed at the time; current behavior is defined only by
[`docs/contracts/`](../contracts/) and [`docs/overview/`](../overview/).

## Rules

- Use one `YYYY-MM-DD-short-slug.md` file per entry.
- Never silently rewrite a historical conclusion. Add a later entry that names what it supersedes.
- Keep entries self-contained and link to code, artifacts, contracts, or commits.
- Run `python tools/generate_journal_index.py` after adding or moving an entry.

## Index

| date | entry | title |
|---|---|---|
| 2026-09-09 | [2026-09-09-jepa-literature-notes.md](2026-09-09-jepa-literature-notes.md) | JEPA / latent world-model SSL: best-practice survey vs. the HALO Phase-A design |
| 2026-09-12 | [2026-09-12-fixed-filterbank-decision.md](2026-09-12-fixed-filterbank-decision.md) | Decision: focus on the fixed filterbank; add 4 s and 8 s patches |
| 2026-09-12 | [2026-09-12-frontend-efficiency-and-jepa-window.md](2026-09-12-frontend-efficiency-and-jepa-window.md) | Frontend compute budget, and how long the JEPA window should be |
| 2026-09-12 | [2026-09-12-frontend-plan.md](2026-09-12-frontend-plan.md) | Historical frontend and JEPA plan |
| 2026-09-12 | [2026-09-12-jepa-and-encoder-findings.md](2026-09-12-jepa-and-encoder-findings.md) | JEPA pretraining and encoder-design findings |
| 2026-09-12 | [2026-09-12-long-window-proposal.md](2026-09-12-long-window-proposal.md) | Design proposal: long analysis windows for the continuous-kernel arm |
| 2026-09-12 | [2026-09-12-project-history-v1-to-now.md](2026-09-12-project-history-v1-to-now.md) | Project history: HALO v1 → v2 → clinical pivot → current design |
| 2026-09-12 | [2026-09-12-spectral-frontend-literature.md](2026-09-12-spectral-frontend-literature.md) | Is a fixed spectral frontend principled for our regime? What the literature says |
| 2026-09-13 | [2026-09-13-eval-expansion-plan.md](2026-09-13-eval-expansion-plan.md) | Plan: correct the evaluation, and evaluate at 4 s / 8 s / 16 s windows |
| 2026-09-13 | [2026-09-13-filterbank-polarization-features.md](2026-09-13-filterbank-polarization-features.md) | 2026-09-13 — Adding polarization features to the fixed filterbank |
| 2026-09-13 | [2026-09-13-jepa-value-measured.md](2026-09-13-jepa-value-measured.md) | 2026-09-13 — What JEPA pretraining buys, measured |
| 2026-09-13 | [2026-09-13-mantis-comparison-and-zara-correction.md](2026-09-13-mantis-comparison-and-zara-correction.md) | 2026-09-13 — Mantis as a design foil, and a correction to today's ZARA reading |
| 2026-09-13 | [2026-09-13-readiness-repair-plan.md](2026-09-13-readiness-repair-plan.md) | Training and evaluation readiness repairs |
| 2026-09-13 | [2026-09-13-retired-jepa-promoted-results.md](2026-09-13-retired-jepa-promoted-results.md) | 2026-09-13 - Retired future-JEPA promoted-result snapshot |
| 2026-09-13 | [2026-09-13-why-a-simple-encoder.md](2026-09-13-why-a-simple-encoder.md) | 2026-09-13 — Why a deliberately simple encoder: the design rationale |
| 2026-09-14 | [2026-09-14-baseline-failure-analysis.md](2026-09-14-baseline-failure-analysis.md) | 2026-09-14 — Where each baseline fails, and what our readout was leaving on the table |
| 2026-09-14 | [2026-09-14-design-narrative.md](2026-09-14-design-narrative.md) | 2026-09-14 — The argument: how the system got here, and what the evidence says |
| 2026-09-14 | [2026-09-14-evaluation-rebuild.md](2026-09-14-evaluation-rebuild.md) | 2026-09-14 — The evaluation was rebuilt: 4/8/16 s windows, multi-device cells, six fairness guarantees |
| 2026-09-14 | [2026-09-14-multi-device-pooling-correction.md](2026-09-14-multi-device-pooling-correction.md) | 2026-09-14 — Correction: multi-device fusion is a learned attention pool, not a hierarchical mean |
| 2026-09-14 | [2026-09-14-regime-split-vs-shared-classifier.md](2026-09-14-regime-split-vs-shared-classifier.md) | 2026-09-14 — Shared vs. fully-separate zero-shot/few-shot classifier parameters |
| 2026-09-14 | [2026-09-14-residual-classifier-first-result.md](2026-09-14-residual-classifier-first-result.md) | 2026-09-14 — The learned classifier's first sealed result: zero-shot solved, high-k regressed |
| 2026-09-14 | [2026-09-14-support-classifier-design.md](2026-09-14-support-classifier-design.md) | Support classifier design — 2026-09-14 |
| 2026-09-15 | [2026-09-15-ablation-plan.md](2026-09-15-ablation-plan.md) | Ablation plan — strengthening the contribution claims (2026-09-15) |
| 2026-09-15 | [2026-09-15-classifier-isolation-plan.md](2026-09-15-classifier-isolation-plan.md) | Classifier isolation plan - 2026-09-15 |
| 2026-09-15 | [2026-09-15-deployment-scenarios-implementation.md](2026-09-15-deployment-scenarios-implementation.md) | Deployment-heterogeneity scenarios — implementation handoff (2026-09-15) |
| 2026-09-15 | [2026-09-15-deployment-scenarios-plan.md](2026-09-15-deployment-scenarios-plan.md) | Deployment-heterogeneity scenarios — evaluation plan (2026-09-15) |
| 2026-09-15 | [2026-09-15-deployment-scenarios-review.md](2026-09-15-deployment-scenarios-review.md) | Deployment scenario implementation review - 2026-09-15 |
| 2026-09-15 | [2026-09-15-matched-corpus-budget.md](2026-09-15-matched-corpus-budget.md) | Matched-corpus M2: implementation, hyperparameters and compute budget (2026-09-15) |
| 2026-09-15 | [2026-09-15-matched-corpus-plan.md](2026-09-15-matched-corpus-plan.md) | Matched-corpus baseline experiment — plan (2026-09-15) |
| 2026-09-15 | [2026-09-15-matched-corpus-sweep.md](2026-09-15-matched-corpus-sweep.md) | Matched-corpus arms: debug sweep and profiling (2026-09-15) |
| 2026-09-15 | [2026-09-15-normwear-readout-finding.md](2026-09-15-normwear-readout-finding.md) | NormWear is fed the wrong tensor for enrolled readouts (2026-09-15) |
| 2026-09-16 | [2026-09-16-baseline-fidelity-fix-plan.md](2026-09-16-baseline-fidelity-fix-plan.md) | Baseline fidelity and runtime fix plan (2026-09-16) |
| 2026-09-16 | [2026-09-16-baseline-fidelity-sweep.md](2026-09-16-baseline-fidelity-sweep.md) | Baseline fidelity sweep (2026-09-16) |
| 2026-09-16 | [2026-09-16-debug-sweep-eval-curriculum-b.md](2026-09-16-debug-sweep-eval-curriculum-b.md) | Debug sweep 2: evaluation code and the scenario-imitating curriculum (2026-09-16, second pass) |
| 2026-09-16 | [2026-09-16-debug-sweep-scenarios-curriculum.md](2026-09-16-debug-sweep-scenarios-curriculum.md) | Debug sweep: scenario evaluator and Stage A curriculum (2026-09-16) |
| 2026-09-16 | [2026-09-16-evaluation-readiness-fixes.md](2026-09-16-evaluation-readiness-fixes.md) | Evaluation readiness fixes - 2026-09-16 |
| 2026-09-17 | [2026-09-17-classifier-curriculum-screen.md](2026-09-17-classifier-curriculum-screen.md) | Classifier Curriculum Screen - 2026-09-17 |
| 2026-09-17 | [2026-09-17-contextual-classifier-plan.md](2026-09-17-contextual-classifier-plan.md) | Contextual semantic-voting classifier: implementation plan |
| 2026-09-17 | [2026-09-17-heterogeneity-metadata-audit.md](2026-09-17-heterogeneity-metadata-audit.md) | Heterogeneity, metadata, and domain-shift audit |
| 2026-09-17 | [2026-09-17-scenario-evaluation-results.md](2026-09-17-scenario-evaluation-results.md) | Deployment-scenario evaluation results - 2026-09-17 |
| 2026-09-18 | [2026-09-18-debug-sweep-harness-eval.md](2026-09-18-debug-sweep-harness-eval.md) | Debug sweep 3: training harness, curriculum, evaluation metrics and scenarios (2026-09-18) |
| 2026-09-18 | [2026-09-18-device-set-curriculum.md](2026-09-18-device-set-curriculum.md) | Joint Device-Set Curriculum And Evaluation |
| 2026-09-18 | [2026-09-18-encoder-isolation-plan.md](2026-09-18-encoder-isolation-plan.md) | Isolating the encoder from the classifier: plan (2026-09-18) |
| 2026-09-18 | [2026-09-18-eval-dataset-text-audit.md](2026-09-18-eval-dataset-text-audit.md) | Evaluation-dataset expansion and text-pipeline audit |
| 2026-09-18 | [2026-09-18-gradient-scale-audit.md](2026-09-18-gradient-scale-audit.md) | Support-classifier gradient-scale audit |
| 2026-09-18 | [2026-09-18-repo-docs-consolidation-plan.md](2026-09-18-repo-docs-consolidation-plan.md) | Repository, documentation and naming consolidation plan (2026-09-18, second pass) |
| 2026-09-18 | [2026-09-18-repository-consolidation-executed.md](2026-09-18-repository-consolidation-executed.md) | Repository and documentation consolidation executed |
| 2026-09-18 | [2026-09-18-training-evaluation-readiness.md](2026-09-18-training-evaluation-readiness.md) | Training and evaluation readiness sweep |
| 2026-09-19 | [2026-09-19-bounded-contextual-residual-v1-results.md](2026-09-19-bounded-contextual-residual-v1-results.md) | Bounded contextual residual v1: training and evaluation |
| 2026-09-19 | [2026-09-19-contextual-residual-classifier-implementation.md](2026-09-19-contextual-residual-classifier-implementation.md) | Contextual residual classifier implementation |
| 2026-09-19 | [2026-09-19-evidence-aware-classifier-repair-plan.md](2026-09-19-evidence-aware-classifier-repair-plan.md) | Evidence-aware classifier repair plan |
| 2026-09-19 | [2026-09-19-evidence-aware-v2-readiness-audit.md](2026-09-19-evidence-aware-v2-readiness-audit.md) | Evidence-aware v2: implementation and readiness audit |
| 2026-09-19 | [2026-09-19-evidence-aware-v2-readiness-fixes.md](2026-09-19-evidence-aware-v2-readiness-fixes.md) | Evidence-aware v2 readiness fixes |
| 2026-09-19 | [2026-09-19-evidence-aware-v2-second-review-fixes.md](2026-09-19-evidence-aware-v2-second-review-fixes.md) | Evidence-aware v2 second-review fixes |
| 2026-09-19 | [2026-09-19-new-classifier-design-handoff.md](2026-09-19-new-classifier-design-handoff.md) | New classifier design: implementation handoff |
| 2026-09-20 | [2026-09-20-classifier-t6-design.md](2026-09-20-classifier-t6-design.md) | T6: what T4 and the corruption-free control taught, put into one design |
| 2026-09-20 | [2026-09-20-classifier-v4-evidence-gated-design.md](2026-09-20-classifier-v4-evidence-gated-design.md) | Classifier v4: evidence-gated blend on the promoted residual head (agreed design) |
| 2026-09-20 | [2026-09-20-classifier-v4-implementation.md](2026-09-20-classifier-v4-implementation.md) | Classifier v4 implementation notes, and four deviations from the design entry |
| 2026-09-20 | [2026-09-20-primitive-semantic-path-design.md](2026-09-20-primitive-semantic-path-design.md) | Primitive-driven semantic alignment: design, label-side audit, and implementation |
| 2026-09-21 | [2026-09-21-classifier-t8-grounded-primitives.md](2026-09-21-classifier-t8-grounded-primitives.md) | T8: the primitive path with a grounded label side |
| 2026-09-21 | [2026-09-21-v4-promotion-and-primitive-path-postmortem.md](2026-09-21-v4-promotion-and-primitive-path-postmortem.md) | v4 promoted (the T6 recipe), and why the primitive semantic path failed on foreign vocabulary |
| 2026-09-22 | [2026-09-22-baseline-corpus-matched-arms.md](2026-09-22-baseline-corpus-matched-arms.md) | Giving the baselines our corpus: what we run, and why each choice is defensible |
| 2026-09-22 | [2026-09-22-four-rungs-and-rung2-head-correction.md](2026-09-22-four-rungs-and-rung2-head-correction.md) | Addendum: four rungs, and rung 2 does not use the v4 head |
| 2026-09-22 | [2026-09-22-pivot-unsupervised-then-finetune-and-framing.md](2026-09-22-pivot-unsupervised-then-finetune-and-framing.md) | The pivot: unsupervised first, fine-tuning second — and how the paper is framed |
| 2026-09-22 | [2026-09-22-rung1-rung2-implementation-plan.md](2026-09-22-rung1-rung2-implementation-plan.md) | Rungs 1 and 2: implementation plan |
| 2026-09-22 | [2026-09-22-three-regimes-and-unsupervised-adaptation-decisions.md](2026-09-22-three-regimes-and-unsupervised-adaptation-decisions.md) | The three-regime plan, and how the unsupervised tier was redesigned twice in one day |
| 2026-09-23 | [2026-09-23-rung1-learnability-ceiling-and-fixes.md](2026-09-23-rung1-learnability-ceiling-and-fixes.md) | Rung 1: how learnable the adaptation may be, and five fixes to the rung-1 build |
| 2026-09-23 | [2026-09-23-rungs-implementation-record.md](2026-09-23-rungs-implementation-record.md) | Rungs 1, 2 and 4 built; the unlabelled-pool training arm built — implementation record |
| 2026-09-23 | [2026-09-23-three-rungs-discovery-dropped.md](2026-09-23-three-rungs-discovery-dropped.md) | Addendum: the discovery rung is dropped; three rungs, renumbered |
| 2026-09-24 | [2026-09-24-rung1-affinity-and-repository-reorganisation.md](2026-09-24-rung1-affinity-and-repository-reorganisation.md) | Rung 1 gets an embedding-affinity term and is k = 0 only; the repository moves to main-only and is reorganised |
| 2026-09-25 | [2026-09-25-debug-sweep-and-fixes.md](2026-09-25-debug-sweep-and-fixes.md) | Debug sweep of everything built 2026-09-22 to 09-25, and the fixes |
| 2026-09-25 | [2026-09-25-deployment-memory-reader-proposal.md](2026-09-25-deployment-memory-reader-proposal.md) | Deployment memory reader: proposed design for a later classifier experiment |
| 2026-09-25 | [2026-09-25-memory-reader-literature-audit.md](2026-09-25-memory-reader-literature-audit.md) | Memory-reader literature audit: storage and token contract |
| 2026-09-25 | [2026-09-25-memory-reader-online-episode-addendum.md](2026-09-25-memory-reader-online-episode-addendum.md) | Memory reader: online episodes and age decision |
| 2026-09-25 | [2026-09-25-v5-online-memory-implementation.md](2026-09-25-v5-online-memory-implementation.md) | V5 online memory reader: implementation record |
| 2026-09-25 | [2026-09-25-v5-verified-feedback.md](2026-09-25-v5-verified-feedback.md) | V5 verified-label feedback |
