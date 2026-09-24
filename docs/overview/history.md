# HALO history and archives

Last verified against Git references: 2026-09-24.

**All work happens on `main`.** There are no feature, fix or experiment branches; `main` contains
only the active system. Historical states are immutable annotated tags under `hist/<era>/...`.
The older untagged-prefix names (`archive-*`, `phaseb-*`, `results-pre-vocab-fix`) remain as
compatibility aliases of the same commits.

| Era | Scope | Tags |
|---|---|---|
| `v1-language-aligned` | language-aligned general HAR (the MobiCom'26 submission line) | `hist/v1-language-aligned/{results-pre-vocab-fix,docs-consistency,pose-pretext-exploration}-20260721`, `.../{factored-sensor-rework,ssl-pretrain-recipe}-20260724` |
| `v2-evidence-engine` | explicit admissibility, retrieval, reranking, and Phase-B evidence experiments | `hist/v2-evidence-engine/decoder-diagnostics-base-20260811`, `.../decoder-diagnostics-uncommitted-20260811`, `.../{contextual-scalar-reranker,recording-reranker-pretrain,vector8-vote}-20260824` |
| `v2-applications` | abandoned rehabilitation, movement-monitoring, and occupational-strain pivot | `hist/v2-applications/pre-application-main-20260830`, `.../pre-classifier-cleanup-20260911` |
| `v3-support-conditioned` | fixed-filterbank encoder and support-conditioned classifier — **the live era, on `main`** | `hist/v3-support-conditioned/imwut-comparison-pre-cleanup-20260910`, `.../repository-consolidation-20260918`, `.../pre-evaluation-package-20260923`, `.../pre-cleanup-20260924` |

Two v3 snapshots matter for reproduction:

- `pre-evaluation-package-20260923` — the last state in which the rung-2 sealed/scenario pipeline
  was untouched, before its shared modules were extracted into `evaluation/`. Every published
  number was produced by code at or before this tag.
- `pre-cleanup-20260924` — the last state before the 2026-09-24 cleanup, which removed the M1
  primitives module, the retired Future-JEPA health and monitoring scripts, two one-off reporting
  scripts, and moved the rung-2 runners to `evaluation/rung2_frozen/`.

Branches retired on 2026-09-24: `feat/evaluation-package-20260923` and
`fix/training-eval-readiness-20260918` (both fully contained in `main`), and
`codex/phase-b-decoder-diagnostics-20260811`, whose uncommitted worktree state is preserved as
`hist/v2-evidence-engine/decoder-diagnostics-uncommitted-20260811` (its 163 MB of checkpoints are
outside Git, in `halo_archives/phase-b-decoder-diagnostics-uncommitted-20260811/` beside the
repository).

The personal repository is a private transition mirror. The lab-owned private repository is the
source of truth. Archive tags support reproducibility but do not define live architecture,
evaluation, or result documentation.
