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
| 2026-09-18 | [2026-09-18-eval-dataset-text-audit.md](2026-09-18-eval-dataset-text-audit.md) | Evaluation-dataset expansion and text-pipeline audit |
| 2026-09-18 | [2026-09-18-repo-docs-consolidation-plan.md](2026-09-18-repo-docs-consolidation-plan.md) | Repository, documentation and naming consolidation plan (2026-09-18, second pass) |
| 2026-09-18 | [2026-09-18-repository-consolidation-executed.md](2026-09-18-repository-consolidation-executed.md) | Repository and documentation consolidation executed |
| 2026-09-18 | [2026-09-18-training-evaluation-readiness.md](2026-09-18-training-evaluation-readiness.md) | Training and evaluation readiness sweep |
