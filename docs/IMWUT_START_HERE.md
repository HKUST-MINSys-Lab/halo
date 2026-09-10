# HALO for IMWUT: movement monitoring

This branch contains one live research system: a wearable movement-monitoring
pipeline built on pretrained IMU representations. It does not contain the retired
Phase-B retrieval, admissibility, or language-voting systems.

## System

The paper evaluates three operations over timestamped, native-rate motion sequences:

1. **Task 1: arbitrary task detection.** Given one or more bounded reference
   executions, locate that movement in a later complete recording.
2. **Task 2: change quantification.** Learn one person's ordinary variation for
   a confirmed movement and identify meaningful later change.
3. **Task 3: recurrent motion discovery.** Find and consolidate recurring motion
   motifs in a complete occupational timeline.

The authoritative system description is [DESIGN_OF_RECORD.md](design/DESIGN_OF_RECORD.md).
Task definitions and data roles are in [RESEARCH_TASKS.md](design/RESEARCH_TASKS.md).
The sealed measurement rules are in [EVALUATION_PROTOCOL.md](design/EVALUATION_PROTOCOL.md).

## Encoder variants

Every task uses the same `MotionSequence` interface. The compared HALO frontends are:

- `fixed_1s`: fixed physical filterbank with one-second patches;
- `filterbank_multiresolution`: fixed filterbank with joint 0.5, 1.0, and 1.5 second tokens; and
- `continuous_multispan`: continuous physical-time kernels with 0.5, 1.0, and 1.5 second spans.

JEPA-style label-free pretraining is defined in
[JEPA_PRETRAINING_OBJECTIVE.md](design/JEPA_PRETRAINING_OBJECTIVE.md). It is a
pretraining stage, not a separate numbered project phase.

## Code map

| Purpose | Location |
|---|---|
| Recording contract, representation caches, task evaluation | `applications/motion_monitoring/` |
| Task 1, Task 2, Task 3 heads | `applications/motion_monitoring/task{1,2,3}/` |
| HALO frontends and encoder | `model/tokenizer/` |
| JEPA pretraining | `training/tokenizer/pretrain.py`, `training/tokenizer/future_jepa.py` |
| Support-conditioned classification control | `training/support_classifier/`, `model/support/comparator.py` |
| Released comparison encoders | `baselines/harnet/`, `baselines/unimts/`, `baselines/normwear/` |

The support-conditioned classifier is an encoder-transfer control. It is distinct
from Task 1: Task 1 aligns a reference sequence against a complete query timeline;
the classifier scores a bounded query against candidate-labelled support executions.

The agreed direction for complete recordings, training crops, and efficient length grouping is
specified in [RECORDING_LENGTH_BATCHING_PLAN.md](design/RECORDING_LENGTH_BATCHING_PLAN.md).
Its status section distinguishes the implemented Task-1 frozen-representation path from the pending
raw-input and end-to-end encoder work.

## Data and results

The application manifests define the only labelled sources that may enter each
task. The label-free pretraining roster is generated from
[`data/pretraining/corpus_plan.py`](../data/pretraining/corpus_plan.py). Every
used external dataset has a local source record under `references/datasets/`.

Official results belong only in [RESULTS.md](results/RESULTS.md). Historical
classification experiments, obsolete checkpoints, and former design documents are
available through [HISTORY.md](HISTORY.md), not duplicated in this branch.

## Mechanical check

Use the project virtual environment for imports requiring numerical libraries:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.smoke --steps 3 --device cpu
```

This checks mechanics only. It is not an experimental result.
