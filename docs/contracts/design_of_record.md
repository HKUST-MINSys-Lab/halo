# Design of record: support-conditioned heterogeneous HAR

> Last verified against code: 2026-09-24. This document is the design of the **system** (input
> contract, encoder, classifier, training boundary). The **plan** — the three rungs, their methods
> and what is claimed — is owned by [`../overview/roadmap.md`](../overview/roadmap.md) and
> [`../overview/thesis.md`](../overview/thesis.md). It supersedes the earlier language-alignment,
> admissibility, evidence-engine, and movement-monitoring designs.

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

Several simultaneous devices are represented as additional `(device, modality)` sensor rows with
an explicit device id. The parameter-free recording path first averages modalities within each
device and then averages devices, so an accelerometer-plus-gyroscope device does not count twice as
much as an accelerometer-only device. The learned recording pool receives every valid sensor token.

Preprocessing never invents a missing modality. Acceleration-only streams are masked rather than
treated as measured gyroscope signals. All time spans are in seconds, not sample counts.

## Encoder arms

All arms return valid contextual patch vectors and a recording representation produced after
temporal context. Their interfaces preserve duration, physical time, resolution identity, and
validity masks.

| arm | frontend | intended role |
|---|---|---|
| fixed control | one-second physical filterbank | small, interpretable baseline |
| fixed multiresolution | filterbank at 0.5, 1.0, 2.0, and 4.0 seconds, jointly contextualized | frequency features at multiple physical spans |
| continuous multispan | retired experimental frontend; retained only for checkpoint loading/reproduction | not part of the active recipe or reported comparison |

New fixed-filterbank runs append bounded per-triad polarization features when complete xyz metadata
is available. This preserves the physical band-energy path while adding rotation-aware motion
geometry. The complete contract, including checkpoint compatibility, is in
[filterbank.md](filterbank.md).

Acquisition-description conditioning is auxiliary context. It may explain which channels are
present and how they were acquired; it must not become a shortcut for dataset or label identity.

## Support classifier

For each episode, the encoder produces one learned pooled motion vector and one pooled runtime
acquisition vector for the query recording and each support recording. The promoted classifier is
**v4** — the T6 recipe on `support_classifier_v4`
([`model/support/evidence_gated_classifier.py`](../../model/support/evidence_gated_classifier.py)),
promoted 2026-09-21 and the trainer default (`--classifier evidence_gated`). Its floor is the
closed-form, parameter-free support vote. On top of it, two bounded, label-blind gates: a
per-support trust scalar (`|t| <= 2`) that reweights the vote, and a per-candidate blend weight
`lambda <= 0.85` that mixes the vote with the label-meaning path (`p_text` cosine against the
frozen label text). Neither gate sees label text, text confidence or slot identity, and
`lambda_max < 1` means no setting can delete the support path. The T6 recipe adds a label-text
corruption view as a gate-only auxiliary and a label-blind calibration term for candidates with no
support of their own. The head cannot emit unrestricted class logits.

Superseded and negative heads remain loadable for reproduction only: v3 (`support_classifier_v3`,
promoted 2026-09-18 to 2026-09-21), the evidence-aware v2 head (T3), the two v1 contextual heads
(T1, T2), and the primitive semantic path on the v4 architecture — T7 (best sealed aggregate,
disqualified by a 15-point foreign-vocabulary regression) and T8 (built, never trained). The try-number index is in [RESULTS.md](../results/RESULTS.md), because
architecture strings and try numbers do not line up; lifecycle strings in
`model/support/factory.py` are authoritative.

On rung 1 the head is not used at all: pooled episodes are scored by the transductive method
through `p_text` alone (the unlabelled-pool arm in [curriculum.md](curriculum.md)).

There is no top-k retrieval or hidden background bank. Every supplied support row participates in
attention and receives a differentiable score. The `neighbors` control removes the learned classifier and
uses the same encoder with a temperature-scaled soft support vote; it is the fast diagnostic of
encoder quality without classifier reasoning.

## Training and evaluation boundary

Support-classifier training is the active optimization stage. A classifier experiment may freeze a
selected encoder or train a dedicated HALO copy end to end. Query and support encodings must both
receive gradients in the end-to-end arm, including the learned recording pool. The
zero-shot and enrolled losses are averaged by regime when both appear in a batch, so the shared
encoder is not dominated by whichever condition happened to supply more queries.

*Historical (T3 only; not part of the v4 recipe):* the evidence-aware arm optionally added modular path-improvement losses: semantic branch
preservation, refined-support improvement over the unmodified vote where truth has direct support,
and final-output non-regression against the detached better available branch. The last uses raw
per-view regret, normalized group log-mean-exp, then one smooth hinge. Active terms are averaged and
weighted once; no module is manually frozen or assigned to a bespoke objective.

For that arm only, a deterministic 20% per-source target selects a global union of eligible
canonical labels before the optimizer corpus is built. They remain available only in the subject-held-out internal
open-vocabulary panel. Primary checkpoint selection equally weights that panel and the ordinary
seen-label deployment-family panel. This prevents routing decisions from being selected solely on
the training vocabulary. Historical classifiers keep their original corpus and selection rule.

The current paper roster has two pairwise-disjoint active source roles. This table is explanatory; the
executable authority is `data/scripts/curate/deployment_policy.py`.

| role | count | datasets |
|---|---:|---|
| support-classifier training | 8 | `hhar`, `wisdm`, `kuhar`, `harmes`, `xrf_v2`, `dsads`, `forth_trace`, `realdisp` |
| sealed test | 6 | `motionsense`, `realworld`, `shoaib`, `inclusivehar`, `usc_had`, `ut_complex` |

The eight supervised sources are split by subject into optimizer data and an internal validation
fold. That fold selects support-classifier checkpoints, but it is not a separate development-source
roster. Candidate count and support count are episode properties recorded with every score. The
sealed sources never select a checkpoint or hyperparameter and are touched only after the protocol
is frozen. See
[evaluation_protocol.md](evaluation_protocol.md).

Training mixes single-device examples with exact event-aligned 2-4-device examples from `realdisp`,
`xrf_v2`, `dsads`, and `forth_trace`. Eligible training support sets also receive coordinated
query/support device-set challenges; other examples use independent composition. Both synchronous
and worker-based loaders use the same seeded planner. Internal checkpoint validation is
subject-disjoint, deterministic and unaugmented, but includes independently composed multi-device
recordings at the configured composition probability. It does not currently use the coordinated
training device-set planner. The sealed protocol measures both single placements and fixed
all-device composites at 4, 8, and 16 seconds.

## Exclusions

The retired explicit admissibility table, separate Phase-B memory bank, memory-wide retrieval and
candidate-scoring path, arbitrary-label curriculum, and Task 0-3 movement-monitoring packages are
not part of this design. The abandoned `support_contextual_mixture_v1` and
`support_contextual_residual_v1` heads are also outside the active design. Archived or abandoned
components must not be revived through a default flag or undocumented import.
