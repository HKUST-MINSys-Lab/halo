# ExtraSensory label-free pretraining adapter

This adapter exposes the raw ExtraSensory phone and Pebble accelerometer captures to
JEPA pretraining without reusing the supervised converter's activity filtering or
six-second concatenation. Its dataset name is `extrasensory_pretrain`, deliberately
different from the labelled `extrasensory` dataset.

## Source and storage

The upstream publication and local copy are in
`references/datasets/extrasensory/`. The raw archives are shared read-only from:

```text
data/datasets/extrasensory/downloads/
  raw_acc.zip
  watch_acc.zip
  labels.zip
  cv5Folds.zip
```

No archive is copied or linked into this directory. The adapter fetcher writes missing
archives directly into that shared directory, resumes interrupted `.part` downloads,
and validates the published byte sizes before conversion:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m data.pretraining.extrasensory_pretrain.fetch
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m data.pretraining.extrasensory_pretrain.convert
```

Both commands accept `--raw-dir` for an alternate archive location. Conversion also
accepts `--limit-subjects` and `--limit-examples-per-subject` for bounded smoke tests.

## Conversion contract

- Storage is one Parquet per participant and stream: at most 60 files for each of
  `watch_wrist`, `phone_hand`, and `phone_pocket`, rather than roughly 622,000 tiny
  capture files. The converter writes bounded 100,000-row Parquet batches and never
  holds more than one participant's decoded data.
- Every raw phone/watch capture or clock-split part receives one integer `segment_id`.
  The shared grid reader expands these into independent logical sessions before
  resampling and windowing, so JEPA context cannot cross unrelated annotated minutes.
- Activity columns are not loaded. `labels.json` contains only `__unlabeled__`.
- Watch captures are retained whenever their signal and clock are valid.
- Phone captures are retained when exactly one of the two deployment-valid placements,
  hand or pocket, is explicitly present and no conflicting phone placement is active.
  Placement metadata is needed to avoid assigning false sensor context; it is not an
  activity target. Bag, table, unknown, and ambiguous phone captures are excluded.
- The authors' CV split is used only to identify Android versus iPhone. Android
  acceleration is converted from m/s^2 to g; iPhone acceleration is already in g.
  Pebble values are converted from milli-g when required. All streams retain gravity.
- Each uninterrupted capture is independently regularized to 50 Hz. A clock gap or
  reset creates another segment rather than an interpolation across missing time.
- No fixed-duration crop is applied here. Downstream grid construction decides the
  JEPA window length and uses all complete windows from each capture. Short trailing
  regions are not padded into examples because they cannot supply the configured future horizons.

Conversion is resumable by default. Each completed participant is checkpointed in
`manifest.json`; an interrupted participant's one-to-three aggregate files are rebuilt
atomically. Converter schema version 2 is intentionally incompatible with the earlier
one-file-per-capture pilot, so use `--fresh` once when replacing that output. Pass
`--fresh` later only when intentionally rebuilding all converted sessions. Generated
`sessions/`, `labels.json`, and `manifest.json` are ignored by git.
The tracked `metadata.json` declares this source as label-free and pretraining-only.
