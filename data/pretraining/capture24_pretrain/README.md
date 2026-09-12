# Capture-24 label-free pretraining view

This adapter makes a source-isolated Future-JEPA corpus from the existing local public
Capture-24 archive. It does not download or copy the source archive.

The labelled Capture-24 converter under `data/datasets/capture24/` splits participants at
activity-label transitions. That is appropriate for supervised recognition, but would make a
predictive objective see annotation-defined rather than physical sequence boundaries. This adapter
reads only `time`, `x`, `y`, and `z` from each original `P*.csv.gz`. It never reads the `annotation`
column or its dictionary. One packed Parquet per participant carries `segment_id`; only invalid
timestamps and genuine clock gaps create a new segment.

All output labels are the reserved `__unlabeled__` marker. The view is therefore usable only by the
label-free JEPA roster, never by a support episode, label vocabulary, labelled selection probe, or test.

The raw source is the dominant-wrist, 100 Hz, gravity-present Axivity AX3 data from 151 people. Its
publication and archive already live under `references/datasets/capture24/`.

```bash
python -m data.pretraining.build_corpus --stage fetch --datasets capture24_pretrain --yes
python -m data.pretraining.build_corpus --stage convert --datasets capture24_pretrain
python -m data.pretraining.build_corpus --stage grids --datasets capture24_pretrain
```

Generated `sessions/`, `labels.json`, `manifest.json`, and `grids/` are ignored by Git.
