# Repository and documentation consolidation executed

**Date:** 2026-09-18

**Supersedes:** [the consolidation plan](2026-09-18-repo-docs-consolidation-plan.md) as an
implementation status record. The plan remains the historical rationale.

The active repository now presents one support-conditioned HALO design. Runtime paths are owned by
`halo/paths.py`; stable console commands replace layout-dependent module commands; living documents
are under `docs/contracts/` and `docs/overview/`; dated decisions and audits are indexed under the
journal; superseded designs are visibly archived. The active name is **Heterogeneity-Adaptive,
Lightweight, Open-vocabulary activity recognition**.

Generated state was separated from source without deleting it. Local checkpoints moved to
`runs/` and evaluation caches to `cache/`, with ignored compatibility links at their former paths.
Promoted and historical machine-readable evidence moved to `results/artifacts/`. Generated debug
figures and unused modules were removed from the tracked tree; retained publication figures live
with the result artifacts. Continuous/multispan frontends, Future-JEPA, and the old token mixer are
reproduction-only and require explicit acknowledgement flags.

Verification on the consolidated tree:

- complete suite: 959 passed, 1 skipped;
- focused contract and compatibility tests: passed;
- three-step CUDA support-classifier smoke: completed, including validation and checkpoint write;
- seven-scenario HARNet smoke: 14 tasks, 40 rows, zero failures;
- all new console entry points import and render help;
- living Markdown links resolve, data rosters match code, and only promoted PNGs are tracked.

The lab-owned private repository is the source of truth after this entry lands. The personal
repository remains private as a transition mirror. Canonical historical references use annotated
`hist/<era>/<topic>-<date>` tags; old tag names remain temporarily for compatibility.
