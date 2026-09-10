# Dataset storage inventory

> **Retention policy, 2026-09-10.** Source archives, derived timelines, live task artifacts, and
> historical training artifacts are deliberately separate. The active branch carries no retired
> Phase-B checkpoint or result directory.

The machine-readable inventory is
[`data/quality/storage_inventory.json`](../../data/quality/storage_inventory.json). Regenerate it
after acquiring or rebuilding data with:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python -m data.scripts.storage_audit
```

The audit excludes symlinks, `__pycache__`, and incomplete `.part` downloads. It hashes same-size
files of at least 20 MiB to identify byte-identical copies. An identical payload is not automatically
safe to delete because some paths are part of explicit data contracts.

## Retention policy

1. Keep one verified source archive when it is the only compact, reproducible copy of a release.
2. Read ZIP and nested-ZIP payloads lazily when practical. Do not expand a whole continuous corpus.
3. Keep canonical sessions when application code needs physical timelines and native annotations.
4. Keep `native`, `harmonised`, and `non_harmonised` grids while active code still references all
   three. HALO pretraining, fixed-layout baselines, and legacy evaluations do not use the same layout.
5. Store long free-living Task-2 representations at minute resolution. Do not retain every decoded
   CSV or every one-second embedding after aggregation has been verified.
6. Never count a partial download as retained data. Completed payloads must match the frozen source
   byte count or checksum.

## Task-2 longitudinal sources

| source | retained raw storage | expansion policy | derived-cache target |
|---|---:|---|---:|
| MoniPar | 0.40 GB already present | retain its complete 174-session canonical source; no new raw copy | under 0.1 GB |
| ALAMEDA | 4.79 GB compressed | read only four eligible same-placement campaigns for participants 4 and 11 directly from ZIP; do not materialize their 2.33 GB Parquet subset separately | under 0.1 GB |
| COPS | 47.87 GB compressed for all 66 participants | read nested hourly archives directly; never materialize the projected roughly 297 GiB CSV corpus | under 0.5 GB at minute resolution |
| WATCH-PD | pending access | request only longitudinal inertial tasks, boundaries, configuration metadata, and linked clinical measures | measure after access |
| REHAB-120 | 2.47 GB currently quarantined | blocked and not part of Task 2; removable after the audit artifacts are preserved | none |

All 66 COPS participants are retained because subject-disjoint short-term fluctuation evaluation
benefits from the complete cohort. "Only what we use" therefore means compressed source archives
plus selected streaming reads, not a hand-picked participant subset that would weaken or bias the
evaluation.

## Current core corpus

The original `data/datasets` tree occupies approximately **167.1 GB**. Its largest sources are:

| dataset | total | main reason |
|---|---:|---|
| Capture-24 | 57.7 GB | 14.2 GB downloads, 6.7 GB sessions, and 36.7 GB across three grid layouts |
| PHYTMO | 17.7 GB | optical and inertial source payloads retained beside derived sessions/grids |
| HMOG | 13.1 GB | 6.1 GB archive, 4.1 GB sessions, and 2.8 GB native grids |
| REALDISP | 11.6 GB | source archive plus extracted logs and three grid layouts |
| RealWorld | 8.3 GB | source archive plus extracted per-sensor archives |
| ExtraSensory | 7.4 GB | 2.5 GB sessions and 4.8 GB native grids |
| MM-Fit | 6.0 GB | source archive plus extracted multimodal release |
| TNDA-HAR | 5.6 GB | full UniMTS archive plus a selectively extracted 0.62 GB bundle |
| HHAR | 5.6 GB | source archives and extracted activity CSVs |
| NFI-FARED | 4.8 GB | source CSVs, sessions, and three grid layouts |

These layers are rebuildable but not necessarily waste. Removing one should be an explicit tradeoff
between disk space and the ability to reproduce conversion without another download or extraction.

## Confirmed reclaim candidates

The audit currently identifies:

- about **1.90 GB** of embedded Git histories in downloaded SPAR, Forth-Trace, Upper Limb Use, and
  SP-SW-HAR repositories. Dataset conversion does not require repository history.
- **2.47 GB** from the blocked REHAB-120 download and duplicate extraction trees.
- about **0.38 GB** of byte-identical files above 20 MiB. Most are logically separate native and
  non-harmonised grid paths and should not be deleted casually; a duplicated NFI-FARED source CSV and
  duplicated REHAB extraction are genuine source-level redundancy.
- about **0.08 GB** of Python/type-check/debug caches, which are disposable but operationally small.

This gives approximately **4.4 GB of conservative dataset cleanup** without touching source archives,
sessions, or required grid layouts. Source/extracted pairs could recover substantially more, but that
is a cold-storage decision rather than proven waste.

## Training-artifact retention

On 2026-09-10, 27.9 GB of retired comparison outputs were moved out of the live repository to
`/home/alex/code/HALO/halo_archives/imwut-comparison-pre-cleanup-20260910/`:

| archive | size | contents |
|---|---:|---|
| `evidence_outputs` | 21 GB | retired retrieval/admissibility runs and memory banks |
| `compare_outputs` | 984 MB | retired support-classifier variants |
| `tokenizer_outputs` | 5.9 GB | pre-cleanup tokenizer checkpoints and probes |

Live training output directories begin empty under `training/tokenizer/outputs/` and
`training/support_classifier/outputs/`. Active movement-monitoring representation caches remain in
`applications/motion_monitoring/artifacts/`; they are not historical classifier artifacts.
