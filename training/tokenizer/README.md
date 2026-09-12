# Encoder pretraining

`training.tokenizer.pretrain` trains a selected HALO encoder on label-free IMU windows. The default
live objective is future-JEPA; `--jepa-mode masked` remains a named historical control, not the
default recipe.

## Future-JEPA

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
0.5/1.0/1.5 seconds, and continuous multispan kernels at 0.5/1.0/1.5 seconds.

## Operational commands

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
$PY -m training.tokenizer.pretrain --help
$PY -m training.tokenizer.objective_health --frontend fixed --corpus label_free --out /tmp/halo_health.json
$PY -m training.tokenizer.monitor_training --run-dir training/tokenizer/outputs/<run> --render
```

Before a full run, use `objective_health` on the chosen corpus/frontend to confirm finite targets,
valid future horizons, teacher/student separation, loss scales, and gradient reach. During a run,
`monitor_training` reads the lightweight JSON telemetry without touching the GPU.

The current full-corpus batches, step counts, measured RTX 4090 throughput, and wall-time planning
budget are maintained in [PRETRAINING_CORPUS.md](../../docs/data/PRETRAINING_CORPUS.md). Do not use
historical support-classifier timings to estimate JEPA pretraining.

## Retained utilities

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
