# data/scripts

Shared, cross-dataset data logic, grouped by pipeline stage. Import as `data.scripts.<group>.<module>`.

Pipeline order: **curate → assemble** (→ grids). `labels/`, `debug/`, and
`eda/` are cross-cutting.

## `curate/` — which signal (device/channel selection + units)
| Module | Role |
|---|---|
| `deployment_policy.py` | phone/watch device-stream selection; channels/placement per dataset (owns **gravity**) |
| `accel_units.py` | accelerometer **unit → g** (runs *after* deployment_policy) |
| `channels.py` | channel-name helpers (`group_channels_by_sensor`) |

## `assembly/` — signal → windowed harmonised/non-harmonised grids
| Module | Role |
|---|---|
| `baseline_view.py` | **harmonised** (fixed 6-ch, zero-pad+mask) vs **non-harmonised** (native) channel views |
| `assemble.py` | full pipeline: curate → unit→g → (resample) → window → view → `Grid` |

## `labels/` — label vocabulary + text
| Module | Role |
|---|---|
| `canonical_labels.py` | unified **canonical** training vocabulary (merge synonyms) |
| `label_augmentation.py` | per-dataset label synonyms + templates |
| `build_global_label_mapping.py` | legacy shared HAR label vocabulary; not used by the current support-classifier path |

## `debug/` — per-session plots
`visualization_utils.py`, `plot_sessions.py`

## `eda/` — reproducible exploratory analysis
Read-only channel inventories, cross-dataset sensor plots, rotation-aware views,
and generated artifacts under `eda/outputs/`. See [`eda/README.md`](eda/README.md).

## top level
| Module | Role |
|---|---|
| `augmentations.py` | legacy representation-training augmentations; application matching starts from clean signals |
| `download_datasets.py` | download raw datasets into `../datasets/<ds>/downloads/` (per-dataset entry point) |
| `build_grids.py` | assemble converted sessions → windowed harmonised/non-harmonised grids |

Per-dataset pieces (converter, `metadata.json`, `labels.json`) live under [`../datasets/<name>/`](../datasets).
The application timeline contract is in `../../docs/data/DATA_PIPELINE.md`; dataset task roles are in
`../../docs/data/APPLICATION_DATASETS.md`. Harmonised versus native baseline inputs are documented in
`../../docs/baselines/BASELINE_FAIRNESS_POLICY.md` and `../../docs/data/DATA_HETEROGENEITY.md`.
