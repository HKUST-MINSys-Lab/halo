# Evaluation Artifact Status

An evaluation directory is a result only when `run_metadata.json` exists and contains
`"complete": true`. The runner writes `complete: false` with status `running_or_interrupted` before
the first cell, then atomically replaces it at completion. A directory with only `progress.json`,
or with no completion metadata, is an interrupted cache/work directory regardless of words such as
`final` in its historical name.

Known interrupted 2026-09-17 directories include:

- `scenarios_baselines_v3_20260917_final`
- `scenarios_baselines_v3_20260917_representative*`
- `scenarios_halo_classifier_v3_20260917_k0_1_4_8_32`

Their feature arrays may be reused only through the content-addressed cache validator. Their result
rows must not be merged, promoted, or quoted as a completed experiment.
