# HALO history and archives

This branch contains only the active support-conditioned HAR system. Historical systems are kept
as immutable Git references, not parallel live implementations.

| Era | Scope | Preserved reference |
|---|---|---|
| HALO v1 | language-aligned general HAR | `results-pre-vocab-fix` |
| HALO v2 | retrieval, explicit admissibility, and Phase-B evidence experiments | `archive/phaseb-vector8-vote-20260824`, tag `phaseb-vector8-vote-20260824` |
| Application pivot | movement monitoring Tasks 0-3 | tag `archive-pre-classifier-cleanup-20260911` |
| Current | strong encoder plus support-conditioned activity classification | `main` after this cleanup |

Additional older snapshots remain under `archive/*`. The secondary worktree branch
`codex/phase-b-decoder-diagnostics-20260811` is intentionally retained while its owner finishes
its independent diagnostics; it is not part of the current design.

Archive references support reproducibility. They must not be cited as live architecture,
evaluation, or result documentation.
