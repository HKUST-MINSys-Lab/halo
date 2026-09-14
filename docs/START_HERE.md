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

The active encoder arm is the fixed physical filterbank, with one-second and multiresolution
controls. Future-JEPA and the continuous-kernel branch are retained only as historical
reproducibility code; neither is an active recipe or model reference.

## Read in this order

1. [DESIGN_OF_RECORD.md](design/DESIGN_OF_RECORD.md)
2. [EXPERIMENT_ROADMAP.md](design/EXPERIMENT_ROADMAP.md)
3. [EVALUATION_PROTOCOL.md](design/EVALUATION_PROTOCOL.md)
4. [BASELINES.md](baselines/BASELINES.md)
5. [RESULTS.md](results/RESULTS.md)
6. [journal/README.md](journal/README.md) — dated, append-only record of how the design got
   here. Start with [the design narrative](journal/2026-09-14-design-narrative.md) for the
   argument end to end; individual entries carry the measurements behind each decision.

## What is not current

The language-alignment v1 work, future-JEPA, continuous-kernel experimentation, explicit
admissibility and hidden-memory retrieval experiments, and the three-task movement-monitoring
application pivot are archived. The current classifier may be
described as retrieve-mix-vote, but it means the bounded support set and semantic token mixer in
[DESIGN_OF_RECORD.md](design/DESIGN_OF_RECORD.md), not the retired Phase-B memory-bank system.
Archived designs remain recoverable through [HISTORY.md](HISTORY.md), but no current command, paper
claim, or default configuration should depend on them.
