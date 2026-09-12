# HALO: current research entry point

## Research question

HALO studies activity recognition when IMU recordings differ in sampling rate, available channels,
device placement, and recording duration. The system receives a query recording, a bounded
candidate set, and optionally labelled examples of those candidates. It must decide which
candidate best matches the query without treating free-form label language as a sufficient motion
description.

## Current system

1. Convert and curate native-rate IMU while preserving units, channel masks, placement, gravity
   state, and source boundaries.
2. Encode recordings with one selected HALO frontend and the contextual temporal encoder.
3. Compare query and support recordings in the shared representation space.
4. Contextualise query, support, paired support-label, and candidate-label tokens, then aggregate
   support evidence into explicit candidate scores. A separate query/candidate head handles `k=0`.

The three encoder arms are a fixed one-second filterbank control, fixed multiresolution filterbank
at 0.5/1.0/1.5 seconds, and continuous multispan kernels at 0.5/1.0/1.5 seconds. Future-JEPA is
optional pretraining for any compatible arm, not a second classifier.

## Read in this order

1. [DESIGN_OF_RECORD.md](design/DESIGN_OF_RECORD.md)
2. [JEPA_PRETRAINING_OBJECTIVE.md](design/JEPA_PRETRAINING_OBJECTIVE.md)
3. [EXPERIMENT_ROADMAP.md](design/EXPERIMENT_ROADMAP.md)
4. [PRETRAINING_CORPUS.md](data/PRETRAINING_CORPUS.md)
5. [EVALUATION_PROTOCOL.md](design/EVALUATION_PROTOCOL.md)
6. [BASELINES.md](baselines/BASELINES.md)
7. [RESULTS.md](results/RESULTS.md)

## What is not current

The language-alignment v1 work, explicit admissibility and hidden-memory retrieval experiments, and
the three-task movement-monitoring application pivot are archived. The current classifier may be
described as retrieve-mix-vote, but it means the bounded support set and semantic token mixer in
[DESIGN_OF_RECORD.md](design/DESIGN_OF_RECORD.md), not the retired Phase-B memory-bank system.
Archived designs remain recoverable through [HISTORY.md](HISTORY.md), but no current command, paper
claim, or default configuration should depend on them.
