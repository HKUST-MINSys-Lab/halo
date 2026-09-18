# HALO history and archives

Last verified against Git references: 2026-09-18.

`main` contains only the active support-conditioned HAR system. Historical systems are immutable
annotated tags under `hist/<era>/...`; old tag names remain temporarily as compatibility aliases.

| Era | Scope | Canonical references |
|---|---|---|
| `v1-language-aligned` | language-aligned general HAR | `hist/v1-language-aligned/results-pre-vocab-fix-20260712` |
| `v2-evidence-engine` | explicit admissibility, retrieval, reranking, and Phase-B evidence experiments | `hist/v2-evidence-engine/*-20260824` |
| `v2-applications` | abandoned rehabilitation, movement-monitoring, and occupational-strain pivot | `hist/v2-applications/pre-classifier-cleanup-20260911` |
| `v3-support-conditioned` | fixed-filterbank encoder and support-conditioned classifier | `main`; dated snapshots under `hist/v3-support-conditioned/` |

The personal repository is a private transition mirror. The lab-owned private repository is the
source of truth. Archive tags support reproducibility but do not define live architecture,
evaluation, or result documentation.
