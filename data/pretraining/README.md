# Label-free pretraining sources

This tree exists only for optional encoder pretraining. Its sessions carry the reserved
`__unlabeled__` marker and must never enter supervised label mapping, support-classifier episodes,
or evaluation splits.

`data.pretraining.corpus_plan.CORPUS_PLAN` is the executable source of truth. The current build
plan is intentionally limited:

| source | role | expected data |
|---|---|---|
| Capture-24 | subject breadth | continuous dominant-wrist accelerometer, 100 Hz, 151 people |
| Nymeria Xsens | synchronized placement diversity | bilateral limb/torso suit accelerometer and gyroscope, 240 Hz |
| ExtraSensory | deployment-device diversity | free-living phone hand/pocket and wrist accelerometry, 50 Hz store |

The planned source window is eight seconds. It is a pretraining context region, not a claim that
downstream query or support recordings must be cropped to eight seconds. Grid building respects
recording, session, and clock-gap boundaries.

## Safe staged use

```bash
python -m data.pretraining.build_corpus --plan
python -m data.pretraining.build_corpus --stage fetch --datasets capture24_pretrain --yes
python -m data.pretraining.build_corpus --stage convert --datasets capture24_pretrain
python -m data.pretraining.build_corpus --stage grids --datasets capture24_pretrain
```

Each stage is resumable and requires explicit confirmation before a large fetch. Raw assets,
converted sessions, and grids remain ignored by Git. Each source must preserve timestamps, native
rate, canonical units, channel availability, placement, gravity state, and a stable subject ID.

Nymeria is licence-gated and its fetcher fails closed until the user-issued URL manifest is present.
Synthetic motion remains available as an explicit ablation but is not part of the main real-IMU
corpus. NHANES remains wired as an explicit non-dominant-wrist ablation, but its official host was
too slow for the initial production cohort. No script bypasses access controls.

For the research rationale and expected corpus status, see
[PRETRAINING_CORPUS.md](../../docs/data/PRETRAINING_CORPUS.md).
