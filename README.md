# HALO

**HALO: Heterogeneity-Adaptive, Lightweight, Open-vocabulary activity recognition** is a
support-conditioned system for zero- and few-shot recognition from heterogeneous wearable IMUs in
under one million trainable parameters.

HALO receives a query recording, a declared candidate-label set, and optionally labelled support
recordings. It does not claim that language alone describes arbitrary motion, nor does it perform
open-set rejection. Instead, it learns a compact physical-time representation and judges the query
against the evidence available at deployment.

```text
native-rate IMU + acquisition metadata
    -> fixed multiresolution physical filterbank (0.5, 1, 2, 4 s)
    -> contextual temporal encoder and learned recording pool
    -> support-conditioned classifier
    -> scores over the declared candidate labels
```

The active experiment uses 8-second training windows. The parameter-free differentiable-neighbor
path is the encoder control; the learned residual classifier combines support evidence with a
semantic candidate path. Future-JEPA, continuous kernels, explicit admissibility, and hidden-bank
retrieval are preserved as historical work, not active defaults.

Start with [the documentation index](docs/README.md). The living architecture, conditioning,
curriculum, data, baseline, and evaluation contracts are under `docs/contracts/`; promoted results
are in [RESULTS.md](docs/results/RESULTS.md); the dated research record is under `docs/journal/`.

## Commands

```bash
uv sync --extra model --extra dev
uv run halo-train --help
uv run halo-sealed-eval --help
uv run halo-scenarios --help
uv run pytest -q
```

For released-baseline evaluation, install `uv sync --extra model --extra dev --extra baselines`.
The extra includes the CLIP and timm imports used by UniMTS and NormWear; released weights and
their upstream source checkouts remain separate prerequisites.

Generated datasets remain under `data/datasets/`. Checkpoints, feature caches, and local result
artifacts are ignored and can be redirected with `HALO_RUNS_DIR`, `HALO_CACHE_DIR`, and
`HALO_RESULTS_DIR`; all path variables are defined in `halo/paths.py`.
