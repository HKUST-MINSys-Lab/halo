# Design of record: detect, compare, and discover

> **Design of record, 2026-09-10.** This is the sole live design for the IMWUT
> movement-monitoring submission. Historical classification and retrieval systems
> are available only through `docs/HISTORY.md`.

## 1. System boundary

The system accepts phone, smartwatch, or compatible consumer-wearable IMU recordings and produces a
timestamped movement representation. It does not require an activity vocabulary for its core tasks.

```text
native-rate IMU + acquisition metadata
    -> faithful encoder adapter
    -> timestamped patch embeddings and physical measurements
    -> task-specific dense matching, alignment, or recurrence discovery
    -> inspectable physical-time intervals and measurements
```

No generic segmentation stage is required. Long recordings remain intact until the task-specific
algorithm localizes the evidence it needs.

## 2. Common representation contract

Every encoder adapter returns a `MotionSequence` with:

- one normalized embedding per temporal patch or sliding analysis frame;
- physical start, end, and represented duration for every embedding;
- sensor modality, placement, sampling rate, gravity state, and channel-validity metadata;
- execution, session, subject, dataset, and source-recording provenance;
- aligned raw acceleration and gyroscope references; and
- parameter-free physical summaries required by Task 2.

Adapters may use a model's native temporal features. A model exposing only a pooled embedding is
applied with a common physical-time sliding window and stride. Duplicate windows never receive extra
weight merely because one model emits a denser temporal grid.

## 3. Task 1: reference-to-stream detection

Task 1 applies constrained cosine-cost subsequence DTW between one or more reference sequences and a
complete query timeline. The alignment endpoints localize each match. Development-calibrated score
thresholding and temporal non-maximum suppression turn the dense endpoint scores into zero or more
non-duplicate physical-time detections.

The initial method is non-parametric. A diagonal feature weighting, linear projection, or
differentiable alignment is added only as a measured comparison. Synthetic timelines may train or
stress the system, but primary evaluation uses natural continuous recordings and real event
intervals. [`TASK1_ARBITRARY_DETECTION.md`](../tasks/TASK1_ARBITRARY_DETECTION.md) owns the contract.

## 4. Task 2: aligned difference measurement

Task 2 receives independently bounded executions from source annotations, user-guided recordings, or
Task-1 detections. It aligns them once, then keeps several views separate:

1. latent shape and phase-local embedding cost;
2. total duration, phase duration, cadence, and time-warp ratio;
3. acceleration and angular-velocity intensity;
4. dominant and band-limited physical-frequency energy;
5. smoothness, stability, and repetition variability; and
6. sensitivity to session, remounting, rate, and channel availability.

Accepted baseline executions define personal variation and measurement noise. A later execution is
reported as changed only relative to that distribution. Directional claims require external ground
truth. [`TASK2_CHANGE_QUANTIFICATION.md`](../tasks/TASK2_CHANGE_QUANTIFICATION.md) owns the contract.

## 5. Task 3: dense multiscale recurrence discovery

The encoder runs once over a complete recording at a fine physical-time stride. Adjacent base patch
embeddings are pooled over a declared set of durations to form a temporal pyramid. Every candidate
retains its start time, end time, duration, scale, and embedding.

A learned or fixed same-motion metric connects similar candidates. Temporal consolidation removes
duplicate overlapping intervals using a declared rule such as Soft-NMS or weighted interval
selection. A recurrence graph then groups non-overlapping occurrences into motifs. Hidden source
labels are used only for training targets or evaluation, never as semantic input.

Raw or engineered-feature variable-length matrix-profile search, pooled cosine, and constrained DTW
remain controls. The system reports compactness, nearest-negative separation, occurrence count,
duration spread, boundary stability, and representative examples. A person confirms or names a motif
before it is called an activity. [`TASK3_RECURRENT_MOTION_DISCOVERY.md`](../tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md)
owns the contract.

## 6. Optional motion-proposal acceleration

Physical motion evidence, hysteresis, and bounded PELT can screen long still intervals before dense
matching. This is an optional speed baseline only. Every primary Task-1 and Task-3 result must remain
available from direct complete-timeline processing so proposal recall cannot cap model quality.
[`MOTION_PROPOSAL_BASELINE.md`](../methods/MOTION_PROPOSAL_BASELINE.md) records that method.

## 7. Learning boundary

Phase A and Phase B are upstream HALO representation-training stages. They are not names for the
three application tasks. The application comparison starts from released encoder checkpoints and
uses a separate task-specific model for each task.

Every external encoder remains frozen and receives the same small learned Task-1, Task-2, or Task-3
module. HALO is reported both frozen under that matched protocol and as an additional end-to-end arm
in which the task loss may update a task-specific copy of the HALO encoder. This does not require one
HALO checkpoint to become simultaneously optimal for detection, change measurement, and recurrence
discovery.

Each task also retains a non-parametric floor. The floor establishes how much information is already
available and whether the learned module adds value; it is not a claim that a classification-trained
encoder is already an ideal application metric. End-to-end fine-tuning keeps the same task
definition, split, decoder, and evaluation.

- positives are independent executions of the same source movement;
- hard negatives are explicitly different movements from the same source context where possible;
- ambiguous or incompletely annotated relationships are ignored;
- augmentations model sensor nuisance but never manufacture the only positive evidence; and
- test identities, subjects, sessions, and datasets remain held out as declared.

## 8. Role of HALO and external encoders

HALO contributes physical-time features, temporal patch embeddings, explicit missing-information
masks, and heterogeneous-sensor metadata. These are hypotheses, not assumed advantages. They are
tested against raw signals, physical features, and released encoders under the same downstream
pipeline. The system remains publishable as an application result if an external encoder wins a task.

The previous candidate-label interface, zero-shot classifier, memory-label voting, admissibility gate,
and retrieve-mix-vote mechanism are not part of this design. They remain recoverable from Git commit
`32267b6`.

## 9. Implementation principles

- Use physical seconds, not sample counts, for every duration and stride.
- Preserve source timestamps and hard recording/session boundaries.
- Keep reference and evaluated occurrences execution-disjoint.
- Preserve complete timelines for Tasks 1 and 3.
- Separate latent measurements from interpretable physical measurements.
- Treat synthetic streams as training aids, not primary publication evidence.
- Keep the first system non-parametric and auditable.
- Never describe recurrence as intent or difference as clinical improvement without ground truth.
