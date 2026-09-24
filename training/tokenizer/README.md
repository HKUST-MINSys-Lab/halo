# `training/tokenizer/`: shared encoder infrastructure, plus the retired JEPA pretraining

Last verified against code: 2026-09-24.

The directory name is historical. It holds two different things, and the difference matters:

## Active shared infrastructure (used by training and every evaluation rung)

- `pretrain_data.py` — corpus indexing, the multiresolution collate and the stream/channel
  description helpers. Imported by the support-classifier trainer, `evaluation/`, the baselines and
  the data scripts.
- `eval_transfer.py` — `build_encoder` (rebuilds any HALO or corpus-matched encoder from a
  checkpoint) and the dataset encoders the feature caches use.
- `pretrain.py` — besides the retired entry point below, it provides the provenance and corpus
  fingerprint helpers (`capture_runtime_provenance`, `capture_source_provenance`,
  `corpus_fingerprint`) and constants (`DFT_SIZE`, `TRAIN_DATASETS`) that the support-classifier
  trainer imports.

## Retired: label-free Future-JEPA pretraining

`python -m training.tokenizer.pretrain` is retained solely to reproduce the retired label-free
Future-JEPA experiments. It requires `--allow-retired-jepa` and is not an active HALO training
entry point. Its objective modules are `future_jepa.py`, `losses_repr.py` and `ablation_subset.py`.
Why it was retired: [the retired-JEPA record](../../docs/journal/2026-09-13-retired-jepa-promoted-results.md)
and [the measured value of JEPA](../../docs/journal/2026-09-13-jepa-value-measured.md).

The student receives only a physical-time prefix of an eight-second source window; an EMA teacher
encodes the clean window; the predictor forecasts the change from a past-only EMA reference to the
teacher state of a later patch, and a small decoder maps it to frozen physical measurements. The
predictor and decoder are removed after pretraining. Historical revision-2 checkpoints are rebuilt
from their saved configuration and do not inherit current frontend math.

```bash
.venv/bin/python -m training.tokenizer.pretrain --allow-retired-jepa --help
```

The JEPA health and monitoring scripts (`objective_health.py`, `grad_check.py`,
`monitor_training.py`, `plot_training.py`, `eval_quality.py`) and the frontend probes under
`diagnostics/frontend/` were removed from `main` on 2026-09-24; they are preserved at the tag
`hist/v3-support-conditioned/pre-cleanup-20260924`.
