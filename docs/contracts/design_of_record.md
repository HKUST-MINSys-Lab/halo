# Design of record: support-conditioned heterogeneous HAR

> Last verified against code: 2026-09-19. This document supersedes the earlier language-alignment,
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
acquisition vector for the query recording and each support recording. The active
`support_evidence_aware_v2` classifier first calculates two pre-context status-quo distributions:
an execution-distinct support vote and a query-to-candidate label-meaning score. It then
contextualizes query, support, paired support-label, candidate-label, runtime acquisition, and
the status-quo evidence with a shared set transformer. The contextualizer can refine
query/support-to-candidate comparisons and choose a candidate-specific semantic reliance, but it
cannot emit unrestricted candidate logits. Final probabilities are a normalized mixture of the
refined support vote and semantic distribution. With no supports, the availability mask makes the
result exactly the semantic distribution.

The semantic status path is query-only before contextualization, so changing a support set cannot
alter it. Support corrections and status-evidence additions initialize at exactly zero through
learned scalar gates applied after well-conditioned normalized projections; semantic
reliance initializes at `1e-3` when supports exist. The v1 residual and failed v1
contextualize-first heads remain loadable solely for historical reproduction; `--classifier
contextual` creates v2. The CLI default remains the promoted residual control so an omitted model
flag cannot silently launch an experimental head.

The support status uses a learned query/support temperature, while the separately named support
floor remains a fixed-temperature diagnostic. Support-label/candidate semantic binding has its own
learned temperature. One uniform pseudo-support supplies a continuous finite prior over candidates;
its influence decays with the number of real supports instead of changing discontinuously when an
off-roster support assigns an arbitrarily small semantic mass.

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

The evidence-aware arm optionally adds modular path-improvement losses: semantic branch
preservation, refined-support improvement over the unmodified vote where truth has direct support,
and final-output non-regression against the detached better available branch. The last uses raw
per-view regret, normalized group log-mean-exp, then one smooth hinge. Active terms are averaged and
weighted once; no module is manually frozen or assigned to a bespoke objective.

For this arm only, a deterministic 20% per-source target selects a global union of eligible
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
not part of this design. The active bounded-set contextual residual classifier described above is distinct from that
retired path. Archived components must not be revived through a default flag or undocumented import.
