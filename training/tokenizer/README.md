# Retired encoder pretraining

`training.tokenizer.pretrain` is retained solely to reproduce the retired label-free future-JEPA
experiments. It requires `--allow-retired-jepa` and is not an active HALO training entry point.

## Historical Future-JEPA

The student receives only a physical-time prefix of an eight-second source window. An EMA teacher
encodes the clean window. Target-query tokens request later intervals without exposing their signal.
The predictor forecasts the change from a past-only, same-sensor/same-resolution EMA reference to
the normalized teacher state of a later patch. It receives relative time-to-boundary metadata on
every observed context token, but no target absolute position. Adding the predicted residual back to
that reference yields the future state passed to a small decoder for frozen, standardized physical
measurements. The predictor and decoder are removed after pretraining.

The implementation accepts any frontend that returns the common token-grid contract: token values,
physical start/center/end times, durations or resolution IDs, sensor metadata, and validity masks.
The retained arms are fixed one-second filterbank, fixed multiresolution filterbank at
0.5/1.0/2.0 seconds, and revision-3 continuous multispan kernels at 0.5/1.0/2.0 seconds.
The loader reconstructs historical revision-2 0.5/1.0/1.5-second checkpoints from their saved
configuration; those checkpoints do not silently inherit current frontend math.

## Historical commands

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
$PY -m training.tokenizer.pretrain --allow-retired-jepa --help
$PY -m training.tokenizer.objective_health --allow-retired-jepa --frontend fixed --corpus label_free --out /tmp/halo_health.json
```

These commands are historical reproducibility utilities only. The corpus and timing record live in
[PRETRAINING_CORPUS.md](../../docs/data/PRETRAINING_CORPUS.md).

## Retained historical utilities

- `pretrain.py`: configuration, training, validation, checkpointing, and telemetry;
- `pretrain_data.py`: corpus indexing, balanced sampling, and collate contract;
- `future_jepa.py`: future-target construction and objective modules;
- `losses_repr.py`: shared numerical utilities and the explicit legacy masked control;
- `objective_health.py`, `grad_check.py`, `monitor_training.py`, `plot_training.py`: targeted
  readiness and health diagnostics; and
- `eval_transfer.py`, `eval_quality.py`: encoder reconstruction and controlled representation
  probes.

The removed fleet launchers, contrastive experiment, and old Phase-A diagnostic scripts are
available through Git history rather than as competing live entry points.
