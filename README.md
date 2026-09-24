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

The paper is organised as three rungs, scored identically for HALO and five released baselines
(HARNet-5, HARNet-10, LiMU-BERT-X, UniMTS, NormWear): **1** unlabelled adaptation — the roster is
known by name and a pool of unlabelled recordings grows, with no labelled example ever; **2** k
labelled examples with parameters frozen — the existing sealed and scenario results, now a case
study; **3** k labelled examples with fine-tuning for every model. Rung 2 is done; rungs 1 and 3 are
built and pre-registered but not run. The plan is [the roadmap](docs/overview/roadmap.md).

The encoder uses 8-second training windows. The promoted classifier is **v4**
(`support_classifier_v4`, T6 recipe, promoted 2026-09-21); v3 is superseded and the evidence-aware
v2, T7 and T8 tries are recorded negative results — see [RESULTS.md](docs/results/RESULTS.md) for
the try-number index, because architecture strings and try numbers do not line up. Future-JEPA,
continuous kernels, explicit admissibility and hidden-bank retrieval are retired.

Start with [the documentation index](docs/README.md). The living contracts are under
`docs/contracts/`, the overview (thesis, architecture, roadmap, history) under `docs/overview/`,
promoted results in [RESULTS.md](docs/results/RESULTS.md), and the dated research record under
`docs/journal/`. All work happens on `main`; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Commands

```bash
uv sync --extra model --extra dev
uv run halo-train --help
uv run halo-sealed-eval --help
uv run halo-scenarios --help
uv run halo-rung1 --help
uv run halo-rung3 --help
uv run pytest -q
```

For released-baseline evaluation, install `uv sync --extra model --extra dev --extra baselines`.
The extra includes the CLIP and timm imports used by UniMTS and NormWear; released weights and
their upstream source checkouts remain separate prerequisites.

Generated datasets remain under `data/datasets/`. Checkpoints, feature caches, and local result
artifacts are ignored and can be redirected with `HALO_RUNS_DIR`, `HALO_CACHE_DIR`, and
`HALO_RESULTS_DIR`; all path variables are defined in `halo/paths.py`.
