# Exploratory Data Analysis

This directory owns exploratory analysis code and generated artifacts for HALO.
EDA is read-only with respect to raw downloads, converted sessions, grids,
checkpoints, and evaluation results.

## Layout

- Python modules live directly in this package and run as
  `python -m data.scripts.eda.<module>`.
- `outputs/` contains generated figures, tables, and deterministic sample
  manifests, grouped by analysis family. Its contents are ignored by Git.

```text
outputs/
  inventory/
  signals/
    trajectories/
    magnitudes/
    samples/
  gravity_alignment/
    overlays/
    samples/
  tokenizer_features/
    raw/
    gravity_aligned/
  activity_signatures/
    cross_dataset/
    within_dataset/
    summaries/
```

`--output-dir` sets the root of this same structure; it does not flatten the
analysis-specific subdirectories.

Every sampled example must record the random seed and enough provenance to
recover its dataset, subject, session, activity, device stream, channels,
sampling rate, window bounds, and source alignment.

## Existing commands

```bash
python -m data.scripts.eda.inventory_streams
python -m data.scripts.eda.plot_sensor_trajectories --label walking
python -m data.scripts.eda.plot_gravity_aligned_overlays --label walking sitting standing cycling
python -m data.scripts.eda.plot_tokenizer_features --label walking sitting standing cycling
python -m data.scripts.eda.plot_tokenizer_features --orientation gravity_aligned --label sitting standing
python -m data.scripts.eda.plot_activity_signatures
```

The trajectory command writes a raw 3D sensor-state figure, a magnitude figure,
and a JSON sample manifest. Pass exact `dataset/stream` values with `--streams`
to change the comparison set.

The gravity-aligned view estimates pitch/roll from mean acceleration, applies one
rigid rotation to both co-located sensors, and then makes a unitless overlay after
per-modality centering and RMS normalization. It does not infer yaw.

The tokenizer-feature script is a historical EDA utility and still imports the
reference physical-Hz filterbank from the sibling `legacy_code` repository. The
current repository now contains HALO model code, but this script has not yet been
migrated to its checkpoint interface. It plots pre-projection log-band energies only;
it does not claim to show current learned encoder embeddings or checkpoint behavior.
Raw device-frame input is the default. `--orientation gravity_aligned` is a
sensitivity view because alignment changes the signed DC gravity/tilt features
that the tokenizer intentionally preserves for static-posture discrimination.

`plot_activity_signatures` uses up to 96 deterministic windows per
dataset/activity and produces cross-dataset percentile profiles, cross-activity
and within-dataset heatmaps, and absolute-energy distributions. Its spectral
shape sums energy across xyz before normalization, so it is invariant to any
rigid 3D sensor rotation. It is a diagnostic companion to the actual tokenizer,
which remains per-axis and orientation-sensitive.

## Initial analysis set

1. **Channel and stream inventory**: summarize available raw channels and the
   phone/watch stream retained by `data.scripts.curate.deployment_policy`.
2. **Cross-dataset sensor trajectories**: choose an activity, sample a fixed
   number of windows per dataset, and plot accelerometer and gyroscope vector
   states in 3D. These are trajectories through measurement space
   `(x(t), y(t), z(t))`, not estimates of physical position.
3. **Orientation-aware companion views**: pair raw-axis plots with vector
   magnitude and other rotation-invariant summaries. Gravity alignment may be
   shown only when gravity is present and estimable; it does not recover global
   yaw or make differently worn devices fully coordinate-equivalent.
4. **Dataset/activity comparisons**: use shared canonical labels while retaining
   the original label when the selected grid exposes it, plus available
   acquisition metadata in the manifest.
5. **Application representation views**: compare frozen encoder sequences, aligned
   demonstrations, per-phase deviations, and recurrent motifs. These views remain
   to be implemented in a dedicated future analysis package rather than the active classifier.

Primary figures use deployment-plausible phone or watch streams. Analyses of
ankle, chest, ECG, magnetometer, or other pruned inputs must be explicitly
labelled as non-deployment stress analyses and kept separate from primary
figures.
