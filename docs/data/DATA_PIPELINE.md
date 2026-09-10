# Data pipeline - source sessions, training grids, and application timelines

How raw datasets become the gridded corpus. Every stage is one module in `data/scripts/` (see also
`DATA_HETEROGENEITY.md` for the per-dataset rationale).

> **Application requirement, 2026-08-28:** the six-second grids are an encoder-training and legacy
> HAR representation. Tasks 1-3 operate on complete source-session timelines. Application loaders
> must reconstruct overlapping temporal embeddings from the converted sessions, preserve real gaps
> and event boundaries, and never treat six-second blocks as independent source recordings.

## Stages

| # | Stage | Module | Output |
|---|---|---|---|
| 1 | **Convert** | `data/datasets/<ds>/convert.py` | `data/datasets/<ds>/sessions/*.parquet` + `manifest.json` + `labels.json` *(gitignored)* |
| 2 | **Curate** | `curate/deployment_policy.py` | one phone/watch device stream, acc(+gyro), gravity reconstructed |
| 3 | **Unit** | `curate/accel_units.py` | accelerometer → g |
| 4 | **Assemble** | `assembly/assemble.py` | (resample ->) window -> one training/evaluation view per alignment (`Grid`) |
| 5 | **Orchestrate** | `build_grids.py` | per-stream grids under `data/datasets/<ds>/grids/<alignment>/<stream>/` |
| 6 | **Screen** | `scan_implausible.py`, `scan_duplicates.py` | per-alignment exclusion caches in `data/quality/` |

**Three alignments** are built (`build_grids._ALIGNMENTS`), not two:

| alignment | rate | channels | labels | who consumes it |
|---|---|---|---|---|
| `native` | native (20/50/100 Hz) | 6-ch `[acc,gyro]` + mask | canonical | HALO representation training and historical evidence-engine experiments |
| `harmonised` | 60 Hz | 6-ch `[acc,gyro]` + mask | canonical | legacy fixed-rate exports and explicit ablations |
| `non_harmonised` | native | native 3/6-ch | native | the default `run_baselines` scoring alignment |

Native grids retain the valid samples from each recording but partition them into non-overlapping
contexts of at most six seconds. The final shorter context is right-padded on disk and accompanied by
`lengths.npy`. HALO loaders slice to that valid length before augmentation and form fixed one-second
patches, including one honest final short patch. Corpus-matched CrossHAR, LiMU-BERT, and HARNet
training also starts from these native windows, trims each row by `lengths.npy`, and then performs
anti-aliased resampling plus crop/padding to the published model input contract. Their old fitted
backbones and heads are intentionally rejected by preprocessing schema checks and must be retrained.

`placement_strict` → phones only (drops the watch datasets).

Stage 6 is **not optional**. `CorpusIndex`, `build_memory`, `resolvability`, `sensor_bias`, `eval/data.py`
and `run_baselines` all load these caches with `require=True`: a missing or stale cache is a hard
failure, because silently readmitting byte-identical stale-buffer windows would corrupt both training
and scoring. Rebuild them after every grid rebuild, once per alignment you intend to use.

The application path needs an equivalent session-level quality screen. It must map every rejected
grid interval back to source time, reject duplicate or stale buffers before motif discovery, and
retain the reason in the application manifest. Otherwise Task 3 could report a sensor replay defect
as a highly recurrent human motion.

## Run

```bash
python -m data.scripts.download_datasets                  # (1) download raw → data/datasets/<ds>/downloads/
python -m data.datasets.<ds>.convert                      #     convert one dataset → sessions/ + labels.json + manifest.json
python -m data.scripts.build_grids                        # (2–5) all three alignments; default roster =
                                                          #       expanded Phase-A train ∪ primary eval
python -m data.scripts.build_grids --dataset hhar wisdm   #       ...or just some datasets
python -m data.scripts.build_grids --alignment native     #       ...or just one alignment

python -m data.scripts.scan_implausible                   # (6) REQUIRED — native exclusion cache
python -m data.scripts.scan_duplicates                    # (6) REQUIRED — native exclusion cache
python -m data.scripts.scan_implausible --alignment non_harmonised   # needed by run_baselines
python -m data.scripts.scan_duplicates  --alignment non_harmonised   # needed by run_baselines

python -m data.scripts.labels.build_global_label_mapping  # legacy HAR vocabulary; not required by Tasks 1-3
python -m data.scripts.curate.sensor_bias --build         # frozen per-sensor physics; Phase A fails closed without it
```

`build_grids` with no `--dataset` no longer means "every dataset on disk": it resolves
`EXPANDED_PHASE_A_TRAIN_DATASETS ∪ PRIMARY_EVAL_DATASETS` via `build_stream_specs()`, so a clean
rebuild reproduces the trainer's roster instead of depending on grids left behind by an earlier
one-off build. Optional scale sources (`extrasensory`, `nhanes`, `hmog`, `kneepad`) must be named
explicitly.

`placement_strict` (phones only, "harmonised-strict") is not a separate build — it is the phone
subset of the harmonised grids, selected at training time via `deployment_streams(placement_strict=True)`.

## Converter contract (the recipe every converter follows)

Run as `python -m data.datasets.<ds>.convert` from the repo root. Each converter: (a) reads raw from
`data/datasets/<ds>/downloads/`, writes to `data/datasets/<ds>/` (`DS_DIR = Path(__file__).resolve().parent`);
(b) emits **raw whole-recording sessions** `sessions/<id>/data.parquet` (build_grids does the 6 s
windowing — converters must NOT pre-window); (c) includes a `subject`
column for subject-disjoint splits and writes per-session activity to `labels.json`; (d) emits the exact
source column names `deployment_policy` selects for that dataset's stream. All 35 converters under
`data/datasets/*/convert.py` satisfy this and are verified against real output.

One deliberate exception to (b): the 2026-08 multi-placement sources (realdisp, forth_trace, dsads,
opportunity, phytmo, mmfit, kneepad, monipar, spar, upper_limb_use) write **every placement of one
recording into a single frame** with a per-placement column prefix. That is what makes window *i* of
each placement the same physical instant, which the paired-contrast measurement in
`training/evidence/resolvability.py` depends on. Copy this pattern for any future multi-sensor source.
