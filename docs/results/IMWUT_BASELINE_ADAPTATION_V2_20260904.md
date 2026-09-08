# Released-checkpoint adaptation baselines — 2026-09-04

This is the tracked result of the first complete released-checkpoint baseline run on the corrected
`adaptation_v2` protocol. This is a historical baseline snapshot, not a matched comparison
against the September 7 sensor-only HALO run. The current manifest is
`eval/manifests/adaptation_v2_20260907.json.gz`; rerun all models on committed current source
before producing the new comparison. Do not reuse these numbers in that table.
HALO is intentionally absent: no comparison-model checkpoint was available for this run. Add HALO
only after its checkpoint passes the same readiness and provenance checks.

## Experimental record

| item | value |
|---|---|
| code commit | `abe8df56315bc9d589e56df32beb05ee377f1488` |
| protocol | `halo_matched_adaptation_v2` |
| manifest fingerprint | `5058cf414c5ed2e4e52d9e09aa2317c87329bfa0bfbc3f19e091a8f2471c2bca` |
| evaluation datasets | 10 |
| manifest cells | 69 total; 48 usable |
| candidate vocabulary | 63 labels; 48 absent from the 166-label training vocabulary |
| support seeds | 5 |
| device | NVIDIA RTX 4090, 24 GiB |
| full wall time | 480.82 s (8 min 1 s) |

The primary enrollment benchmark is the fixed, common cohort that supports every point at
`k in {1, 2, 4, 8}` under **cross-subject, same-configuration** enrollment. It contains eight
datasets: InclusiveHAR, MoniPar, MotionSense, RealWorld, Shoaib, SPAR, USC-HAD, and UT-Complex.
Every support item is one equally weighted vector pooled from one independent execution. The
reported score pools query-window predictions across subjects within each cell, computes macro F1,
averages cells/seeds within each dataset, then gives every dataset equal weight. Subjects with more
query windows contribute more to the cell score. Paired subject-macro deltas and bootstrap intervals
are separate statistics, not confidence intervals for these cell-based table scores.

TNDA-HAR is subject-unattributed in the released files and is therefore excluded from the primary
cross-subject claim. Upper Limb Use cannot support the complete fixed `k=1..8` cohort. Both remain
in the supplemental outputs. `k=16` uses a smaller high-support cohort and is not joined to the
primary curve.

## Native zero-shot result

Macro F1 is dataset-macro over all ten evaluation datasets. HARNet is a released representation
encoder without a native open-vocabulary classifier, so reporting a locally fitted text bridge as
its native zero-shot method would be misleading.

| released model | native macro F1 at k=0 | datasets |
|---|---:|---:|
| UniMTS | **28.58** | 10 |
| ImageBind | 10.90 | 10 |
| NormWear | 4.67 | 10 |
| HARNet | not supported | 10 |

## Primary enrollment result

Macro F1, dataset-balanced over the eight common cross-subject/same-configuration datasets.
`1-NN`, prototype, and closed-form ridge are the same non-gradient readouts for every encoder.

| encoder | readout | k=1 | k=2 | k=4 | k=8 |
|---|---|---:|---:|---:|---:|
| HARNet | 1-NN | 46.13 | 50.03 | 53.66 | 56.67 |
| HARNet | prototype | 46.13 | 49.53 | 52.57 | 54.57 |
| HARNet | ridge | 45.42 | 49.25 | 53.41 | 57.13 |
| UniMTS | 1-NN | **55.81** | **60.58** | **64.27** | **67.18** |
| UniMTS | prototype | **55.81** | **58.57** | **61.17** | **62.99** |
| UniMTS | ridge | **53.61** | **56.87** | **59.82** | **62.28** |
| ImageBind | 1-NN | 40.54 | 46.06 | 50.67 | 55.13 |
| ImageBind | prototype | 40.54 | 44.20 | 47.67 | 50.38 |
| ImageBind | ridge | 40.62 | 45.32 | 50.08 | 54.85 |
| NormWear | 1-NN | 23.71 | 26.42 | 28.95 | 31.77 |
| NormWear | prototype | 23.71 | 24.52 | 25.37 | 25.49 |
| NormWear | ridge | 19.53 | 19.84 | 20.10 | 19.84 |

UniMTS is the strongest released representation on this protocol at every primary support count.
The best readout is not universal: 1-NN leads UniMTS through `k=8`, while ridge narrowly leads
HARNet at `k=8`. This is why all three matched readouts must remain in the table.

## Per-dataset 1-NN result

Each entry is `k=1 / k=2 / k=4 / k=8` macro F1 on the same primary cohort.

| dataset | HARNet | UniMTS | ImageBind | NormWear |
|---|---:|---:|---:|---:|
| InclusiveHAR | 33.04 / 34.37 / 34.26 / 34.48 | 35.40 / 38.63 / 41.93 / 44.24 | 23.15 / 27.60 / 29.12 / 32.69 | 25.65 / 26.69 / 27.71 / 29.81 |
| MoniPar | 30.02 / 35.32 / 39.89 / 44.27 | 37.96 / 39.34 / 41.32 / 42.65 | 20.25 / 23.39 / 26.57 / 29.75 | 18.96 / 22.43 / 25.10 / 27.78 |
| MotionSense | 62.04 / 66.12 / 70.10 / 73.30 | 75.48 / 81.55 / 84.77 / 87.48 | 50.36 / 55.69 / 58.52 / 62.76 | 27.97 / 31.07 / 33.71 / 36.49 |
| RealWorld | 47.30 / 50.92 / 54.54 / 56.77 | 68.45 / 72.49 / 73.84 / 75.18 | 42.35 / 46.97 / 49.79 / 51.72 | 21.99 / 23.89 / 26.15 / 27.14 |
| Shoaib | 67.76 / 71.80 / 75.96 / 80.23 | 84.20 / 89.61 / 93.09 / 95.14 | 52.22 / 62.79 / 71.62 / 79.79 | 29.38 / 31.09 / 34.80 / 39.48 |
| SPAR | 38.48 / 43.47 / 48.65 / 52.33 | 45.55 / 52.32 / 58.71 / 64.98 | 53.36 / 58.34 / 64.71 / 70.62 | 21.11 / 24.77 / 26.58 / 29.08 |
| USC-HAD | 36.33 / 39.18 / 42.56 / 45.51 | 42.58 / 48.15 / 52.77 / 56.99 | 32.00 / 37.13 / 41.22 / 43.71 | 17.77 / 20.39 / 22.81 / 24.77 |
| UT-Complex | 54.09 / 59.06 / 63.30 / 66.45 | 56.89 / 62.55 / 67.70 / 70.82 | 50.64 / 56.60 / 63.77 / 69.99 | 26.81 / 31.03 / 34.73 / 39.63 |

## Runtime and provenance

| encoder | elapsed | setup | peak VRAM | feature-cache result |
|---|---:|---:|---:|---|
| HARNet | 28.32 s | 0.13 s | 0.22 GiB | 14 misses, 4,374 execution vectors |
| UniMTS | 64.06 s | 2.17 s | 4.94 GiB | 14 misses, 4,374 execution vectors |
| ImageBind | 54.78 s | 7.76 s | 5.10 GiB | 14 misses, 4,374 execution vectors |
| NormWear | 324.51 s | 2.57 s | 7.67 GiB | 14 misses, 4,374 execution vectors |

All four outputs record a clean worktree, exact source fingerprints, and hashes of the released
model artifacts. NormWear also records the pinned TinyLlama revision used by its released zero-shot
path. This was the cold full run, so every stream was a cache miss; later evaluations reuse the
validated stream features and execution pools.

Generic enrollment readouts do not consume label text. Their coherent-label and random-alias
scores are therefore identical by construction; duplicating both tables would imply a text
robustness experiment that these readouts do not perform.

## Reproduction artifacts

The large machine-readable outputs are deliberately ignored by Git but remain in this checkout:

- `eval/adaptation_results/released_v2_20260904/` — four per-model JSON files;
- `eval/adaptation_tables/released_v2_20260904/cells.csv` — all subject/seed/cell results;
- `eval/adaptation_tables/released_v2_20260904/dataset_macro.csv` — grouped summaries;
- `eval/adaptation_tables/released_v2_20260904/tables.md` — all supplemental tables; and
- `eval/adaptation_feature_cache/` — strict-key representation caches.

Reproduce the model outputs with:

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
$PY -m eval.run_adaptation_baselines \
  --manifest eval/manifests/adaptation_v2.json.gz \
  --baselines harnet unimts imagebind normwear \
  --device cuda \
  --out-dir eval/adaptation_results/released_v2_20260904 \
  --feature-cache-dir eval/adaptation_feature_cache \
  --methods nearest prototype ridge \
  --label-modes coherent random_alias
```
