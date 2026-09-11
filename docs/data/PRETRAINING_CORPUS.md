# Label-free pretraining corpus

> Operational source of truth: `data/pretraining/corpus_plan.py`. This document explains the live
> plan; it does not activate a source that lacks a tested converter and loader.

## Purpose and isolation

Future-JEPA uses no activity labels. Label-free sources live under `data/pretraining/`, carry the
reserved `__unlabeled__` marker, and are excluded from label vocabularies, support-classifier
episodes, and evaluation splits. This makes a transfer claim auditable: a dataset used as a held-out
recognition test cannot silently have entered encoder pretraining.

## Current operational plan

| source | contribution | access | state |
|---|---|---|---|
| NHANES | broad wrist accelerometer subject coverage | public | fetcher and converter available; pilot materialized |
| Nymeria Xsens | synchronized bilateral limb and torso six-axis streams | user-authorized URL manifest | converter available; current manifest group must be accepted before full fetch |
| Synthetic IMU | virtual six-axis motion diversity | gated upstream assets | ablation only, never the real-data base |

The build plan currently targets about 41,100 stream-hours and 163 GB of float16 grids: 36,000
NHANES stream-hours, 3,300 Nymeria Xsens stream-hours, and 1,800 synthetic stream-hours. These are
stream-hours, so simultaneous body placements each count as one encoder input stream. The sampler
must balance sources and subjects; raw duration alone must not let NHANES dominate training.

Nymeria Aria, Ego-Exo4D, ExtraSensory, ArWISE, and Embody3D remain research candidates. They are
not live corpus members because one or more of access, converter, metadata, overlap, or loader
validation is unfinished. Add one only through a reviewed change to `CORPUS_PLAN`, its source
adapter, source documentation, and quality tests.

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
python -m data.pretraining.build_corpus --stage fetch --datasets nhanes --yes
python -m data.pretraining.build_corpus --stage convert --datasets nhanes
python -m data.pretraining.build_corpus --stage grids --datasets nhanes
```

Stages are resumable. Fetching is always explicit and licence-gated sources fail closed until their
approved local assets or URL manifest are available. Before a full run, build one source at a time,
inspect loader outputs, and record the exact `CORPUS_PLAN` revision with the checkpoint.
