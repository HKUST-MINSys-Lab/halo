# Label-free pretraining corpus

> Operational source of truth: `data/pretraining/corpus_plan.py`. This document explains the live
> plan; it does not activate a source that lacks a tested converter and loader.

## Purpose and isolation

Future-JEPA uses no activity labels. Label-free sources live under `data/pretraining/`, carry the
reserved `__unlabeled__` marker, and are excluded from label vocabularies, support-classifier
episodes, and evaluation splits. This makes a transfer claim auditable: a dataset used as a held-out
recognition test cannot silently have entered encoder pretraining.

## Current operational plan

| source | contribution | subjects | streams | usable windows | usable stream-hours |
|---|---|---:|---:|---:|---:|
| Capture-24 | broad free-living dominant-wrist coverage | 151 | 1 | 1,747,520 | 3,883.38 |
| Nymeria Xsens | synchronized bilateral limb and torso six-axis IMU | 39 | 11 | 51,085 | 113.52 |
| ExtraSensory | free-living phone hand/pocket and wrist accelerometry | 60 | 3 | 300,594 | 667.99 |

The materialized three-source corpus contains 2,099,199 usable eight-second windows and 4,664.89
stream-hours after quality exclusions (audit 2026-09-11). Stream-hours count simultaneous body
placements separately because the encoder consumes each placement as an independent sensor stream.
Training samples sources and subjects explicitly rather than in proportion to raw duration, so
Capture-24 cannot dominate merely by being longer.

The production schedule draws 7.68 million windows for either retained JEPA arm. This is 3.66 raw
corpus equivalents, not 3.66 deterministic epochs: source- and subject-balanced sampling deliberately
permits repeats and does not promise that every individual window appears once before repetition.

Three sources is the honest count, not a placeholder for four. The roster covers three distinct
roles: subject breadth on the wrist every evaluation set uses (Capture-24, 151 participants),
in-the-wild phone placements (ExtraSensory), and synchronized multi-placement body IMU at a high
clock (Nymeria Xsens). No available source occupies a fourth role. NHANES would be a second
free-living wrist stream behind Capture-24; synthetic IMU is one actor and is simulated. The way
to add label-free signal here is to grow Nymeria Xsens from its 40-sequence slice toward the
verified 1,100-sequence manifest, which deepens the placement axis that is currently thinnest at
114 materialized stream-hours.

ArWISE was investigated and rejected. The accessible s01 and s02 waveform archives measured about
1 Hz rather than the documented 10 Hz, while the 38 GB s10 archive could not be transferred reliably
enough to verify its cadence. It is not part of the corpus, build tooling, or source roster.

Synthetic IMU, Nymeria Aria, and Ego-Exo4D remain optional ablations rather than members of the main
corpus. The main experiment uses measured human motion only. Add or remove a source only through a
reviewed change to `CORPUS_PLAN`, its adapter, source documentation, and quality tests.

## Data contract

All sources must preserve native timestamps, subject IDs, stream identity, canonical acceleration
and gyroscope units, gravity state, placement, channel masks, and real recording/gap boundaries.
The common grid reserves six physical slots; acceleration-only inputs zero-pad and mask the gyro
slots rather than fabricate data. Grid arrays are float16 on disk and are converted to float32 at
read time.

Pretraining samples eight-second regions. This is a context region for predictive learning, not a
downstream recording-length policy. It may not cross a session, subject, stream, configuration, or
clock gap.

## Operation

```bash
python -m data.pretraining.build_corpus --plan
python -m data.pretraining.build_corpus --stage fetch --datasets capture24_pretrain --yes
python -m data.pretraining.build_corpus --stage convert --datasets capture24_pretrain
python -m data.pretraining.build_corpus --stage grids --datasets capture24_pretrain
```

Stages are resumable. Fetching is always explicit and licence-gated sources fail closed until their
approved local assets or URL manifest are available. Before a full run, build one source at a time,
inspect loader outputs, and record the exact `CORPUS_PLAN` revision with the checkpoint.

## Measured training budget

Production-shape profiling on 2026-09-12 used the full materialized corpus, the full JEPA model,
BF16 neural computation, and an otherwise idle RTX 4090. Each bounded probe included the ordinary
50-step telemetry and used the same LR and EMA schedule as a complete run; periodic transfer
evaluation was disabled so its cost is not hidden in the optimizer throughput.

| encoder arm | batch | full steps | measured steady throughput | optimizer-time projection | peak allocated VRAM |
|---|---:|---:|---:|---:|---:|
| fixed multiresolution filterbank, 0.5/1.0/1.5 s | 512 | 15,000 | about 14.3 steps/s | about 17.5 min | 3.32 GiB |
| continuous multispan kernels, 0.5/1.0/1.5 s | 384 | 20,000 | about 10.8 steps/s | about 30.9 min | 7.27 GiB |

Allow roughly 20 minutes and 35 minutes respectively for startup, frontend calibration, final
checkpointing, and normal run-to-run variation. Sequentially training both arms should therefore
take about 55 minutes on the reference machine, with one hour as the practical planning budget.
These figures supersede the older 2026-09-11 JEPA profiles. Timings in the continuous-frontend
design for historical support-classifier experiments measure a different workload and must not be
used to estimate JEPA pretraining.
