# Label-free pretraining sources

This tree exists only for optional encoder pretraining. Its sessions carry the reserved
`__unlabeled__` marker and must never enter supervised label mapping, support-classifier episodes,
or evaluation splits.

`data.pretraining.corpus_plan.CORPUS_PLAN` is the executable source of truth. The current build
plan is intentionally limited:

| source | role | expected data |
|---|---|---|
| NHANES | subject breadth | wrist accelerometer, 80 Hz, selected motion-aware hours |
| Nymeria Xsens | synchronized placement diversity | bilateral limb/torso suit accelerometer and gyroscope, 240 Hz |
| Synthetic IMU | optional diversity ablation | virtual six-axis limb streams, never the corpus base |

The planned source window is eight seconds. It is a pretraining context region, not a claim that
downstream query or support recordings must be cropped to eight seconds. Grid building respects
recording, session, and clock-gap boundaries.

## Safe staged use

```bash
python -m data.pretraining.build_corpus --plan
python -m data.pretraining.build_corpus --stage fetch --datasets nhanes --yes
python -m data.pretraining.build_corpus --stage convert --datasets nhanes
python -m data.pretraining.build_corpus --stage grids --datasets nhanes
```

Each stage is resumable and requires explicit confirmation before a large fetch. Raw assets,
converted sessions, and grids remain ignored by Git. Each source must preserve timestamps, native
rate, canonical units, channel availability, placement, gravity state, and a stable subject ID.

Nymeria and synthetic inputs are licence-gated. Their fetchers fail closed until the required URL
manifest or locally acquired assets are present; no script bypasses access controls.

For the research rationale and expected corpus status, see
[PRETRAINING_CORPUS.md](../../docs/data/PRETRAINING_CORPUS.md).
