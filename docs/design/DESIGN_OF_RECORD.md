# Design of record: support-conditioned heterogeneous HAR

> Current design, 2026-09-12. This document supersedes the earlier language-alignment,
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
teacher that supplies later patch targets. A lightweight predictor forecasts each later latent's
residual from a past-only EMA context reference; the reconstructed future latent is decoded to
standardized physical measurements. The student is the only encoder retained after pretraining.
Full leakage and masking rules are in
[JEPA_PRETRAINING_OBJECTIVE.md](JEPA_PRETRAINING_OBJECTIVE.md).

## Support classifier

For each episode, the encoder produces one learnable pooled vector for the query recording and for
each enrolled support recording. The active classifier is a small set transformer with **separate
parameters** for the two information conditions:

* With enrollment (`k > 0`), the token set is the query vector, every support vector, each
  support's paired label token, and every candidate-label token. Role embeddings distinguish the
  four token types. Pair tags bind a support vector to its own label token; candidate tags bind
  that label token to its candidate. The head refines the set jointly, scores the refined query
  against every refined support vector by cosine similarity, and softly votes the scores to the
  bound candidate labels.
* With no enrollment (`k = 0`), the token set is only the query vector and candidate-label tokens.
  The zero-shot head refines them jointly, then scores the refined query against each refined
  candidate label by cosine similarity.

There is no top-k retrieval or hidden background bank. Every supplied support row participates in
attention and receives a differentiable score. The `neighbors` control removes the token mixer and
uses the same encoder with a temperature-scaled soft support vote; it is the fast diagnostic of
encoder quality without classifier reasoning.

## Training and evaluation boundary

Encoder pretraining and support-classifier training are independent stages. A classifier experiment
may freeze a selected encoder or train a dedicated HALO copy end to end. Query and support encodings
must both receive gradients in the end-to-end arm, including the learned recording pool. The
zero-shot and enrolled losses are averaged by regime when both appear in a batch, so the shared
encoder is not dominated by whichever condition happened to supply more queries.

The current paper roster has three pairwise-disjoint source roles. This table is explanatory; the
executable authority is `data/scripts/curate/deployment_policy.py`.

| role | count | datasets |
|---|---:|---|
| label-free JEPA pretraining | 3 | `capture24_pretrain`, `nymeria_xsens`, `extrasensory_pretrain` |
| support-classifier training | 8 | `hhar`, `wisdm`, `kuhar`, `harmes`, `xrf_v2`, `dsads`, `forth_trace`, `realdisp` |
| sealed test | 6 | `motionsense`, `realworld`, `shoaib`, `inclusivehar`, `usc_had`, `ut_complex` |

The eight supervised sources are split by subject into optimizer data and an internal validation
fold. That fold selects support-classifier checkpoints, but it is not a separate development-source
roster. Candidate count and support count are episode properties recorded with every score. JEPA
uses a fixed training schedule and takes its final checkpoint because its three label-free sources
provide no meaningful label probe. The sealed sources never select a checkpoint or hyperparameter
and are touched only after the protocol is frozen. See
[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).

## Exclusions

The retired explicit admissibility table, separate Phase-B memory bank, memory-wide retrieval and
candidate-scoring path, arbitrary-label curriculum, and Task 0-3 movement-monitoring packages are
not part of this design. The active bounded-set token mixer described above is distinct from that
retired path. Archived components must not be revived through a default flag or undocumented import.
