# Design of record: support-conditioned heterogeneous HAR

> Current design, 2026-09-11. This document supersedes the earlier language-alignment,
> admissibility, evidence-engine, and movement-monitoring designs on `main`.

## Objective

Given native-rate IMU recordings from heterogeneous sources, learn a representation in which a
query recording can be compared with a small collection of labelled support recordings. The task
has a declared candidate set. At zero support the system has no enrolled example for the target;
at few support it receives one or more labelled examples. These are different operating
conditions, not interchangeable scores.

The intended claim is not that language alone identifies any action. It is that a rate-aware,
physically grounded encoder and an explicit support comparison can provide useful recognition when
the acquisition configuration and support availability are honestly specified.

## Input contract

Each recording retains its native samples and metadata:

- accelerometer xyz and, when present, gyroscope xyz in canonical units;
- sampling rate and source rate;
- per-channel validity masks and gravity state;
- sensor placement and device description; and
- recording, subject, and dataset provenance.

Preprocessing never invents a missing modality. Acceleration-only streams are masked rather than
treated as measured gyroscope signals. All time spans are in seconds, not sample counts.

## Encoder arms

All arms return valid contextual patch vectors and a recording representation produced after
temporal context. Their interfaces preserve duration, physical time, resolution identity, and
validity masks.

| arm | frontend | intended role |
|---|---|---|
| fixed control | one-second physical filterbank | small, interpretable baseline |
| fixed multiresolution | filterbank at 0.5, 1.0, and 1.5 seconds, jointly contextualized | frequency features at multiple physical spans |
| continuous multispan | learnable continuous kernels at 0.5, 1.0, and 1.5 seconds | temporal as well as frequency-sensitive representation |

Acquisition-description conditioning is auxiliary context. It may explain which channels are
present and how they were acquired; it must not become a shortcut for dataset or label identity.

## Optional predictive pretraining

Future-JEPA uses a student encoder that sees only a prefix of an unlabeled region and an EMA
teacher that supplies later patch targets. A lightweight predictor forecasts future latent states;
a physical decoder reconstructs standardized future physical measurements from the prediction. The
student is the only encoder retained after pretraining. Full leakage and masking rules are in
[JEPA_PRETRAINING_OBJECTIVE.md](JEPA_PRETRAINING_OBJECTIVE.md).

## Support classifier

For each episode, the encoder produces query and support recording representations. A fixed,
temperature-scaled cosine distribution gives each support row a weight. Explicitly enrolled
supports vote for their bound candidate. Background rows may only contribute through the fixed
label-semantic bridge used by the declared protocol.

The optional learned comparator receives the query and every support **sensor vector** with role
and episode-slot embeddings. Its set-attention stack produces one scalar adjustment per support
row. It cannot read candidate, support-label, or acquisition-description embeddings; this prevents
the learned path from bypassing sensor evidence through a text shortcut. The resulting adjusted
support distribution feeds the same vote as the fixed control.

The initial scalar head is zero so the first prediction exactly equals the fixed support vote. Its
weights receive gradients on the first step; the attention stack receives gradients after the head
has moved away from zero. Training telemetry must confirm that the residual-head norm and support
weight change become non-zero.

## Training and evaluation boundary

Encoder pretraining and support-classifier training are independent stages. A classifier experiment
may freeze a selected encoder or train a dedicated HALO copy end to end. Query and support encodings
must both receive gradients in the end-to-end arm, including any learned recording pool.

Evaluation uses subject- and recording-disjoint train/development/test splits. Candidate count and
support count are episode properties, recorded with every score. Thresholds, checkpoint choice,
and hyperparameters come from development data only. The test protocol and retained baselines are
defined in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).

## Exclusions

The retired explicit admissibility table, separate Phase-B memory bank, candidate-token mixer,
arbitrary-label curriculum, and Task 0-3 movement-monitoring packages are not part of this design.
They are archived in Git and must not be revived through a default flag or undocumented import.
