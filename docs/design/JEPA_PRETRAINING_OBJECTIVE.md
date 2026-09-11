# HALO predictive pretraining objective

> **Implemented design of record, updated 2026-09-10.** The default path in
> `training/tokenizer/pretrain.py` implements this multi-horizon future-JEPA objective. The former
> bidirectional masked JEPA plus VICReg recipe remains available only as the explicitly selected
> `--jepa-mode masked` control. Literature evidence and the audit that motivated the change remain in
> [`LABEL_FREE_SCALE_AND_JEPA_AUDIT_20260909.md`](LABEL_FREE_SCALE_AND_JEPA_AUDIT_20260909.md).

## 1. Purpose

The experiment asks whether unlabeled temporal prediction can improve HALO's patch representation.
It reuses the existing HALO tokenizer and encoder. It changes the pretraining task rather than the
inference architecture.

Given past sensor context, the model must:

1. predict the encoder representation of one or more later physical-time intervals; and
2. retain enough physical information in that prediction to reconstruct a compact set of future
   filterbank measurements.

The first objective rewards predictable motion dynamics. The second prevents the predicted latent
from preserving only abstract information that cannot describe the motion. A small distributional
regularizer prevents representation collapse.

This is predictive representation learning, not a complete action-conditioned world model. The
model predicts likely future sensor state from observed motion; it does not receive actions or try
to generate raw sensor waveforms.

## 2. Reused HALO components

The objective is encoder-agnostic at the physical-time token-grid boundary. Every supported
frontend supplies signal tokens, center times, represented durations, resolution IDs, and validity
masks. The default implemented arm uses:

- the fixed physical filterbank;
- 0.5, 1.0, and 1.5 second patch resolutions;
- physical patch start, center, end, and duration metadata;
- sensor acquisition-description conditioning;
- temporal and cross-sensor attention; and
- the final per-patch latent representation used by downstream tasks.

The EMA teacher has the same encoder architecture as the student. The future predictor and physical
decoder exist only during pretraining. Downstream inference retains only the trained student
tokenizer and encoder.

The constrained-learnable filterbank uses the same joint patch grid. The continuous-kernel arm uses
its native fixed one-second token grid because its ordered sub-frame projection is constructed for
one declared physical token duration. This changes tokenization, not the JEPA loss or its
past/future contract.

The multi-span continuous-kernel arm emits a joint physical-time grid at 0.5, 1.0, and 1.5 second
spans by default. It uses the same duration and resolution metadata as the filterbank arms. Both
continuous arms truncate the raw signal presented to the student frontend at the context boundary;
excluding future transformer tokens alone is insufficient because a kernel evaluated on the full
recording can already contain future samples.

### 2.1 Representation levels

The encoder must keep patch-level and recording-level representations distinct:

```text
native-rate IMU
    -> physical-time frontend at one or more resolutions
    -> temporal, cross-sensor, and cross-resolution context
    -> contextual patch states                         (JEPA representation)
    -> optional downstream recording pool
    -> one recording state                             (task representation)
```

JEPA predicts contextual patch states. It does not predict the pooled recording state. The encoder
continues to expose both `per_patch` and `pooled` outputs for compatibility, but the current
duration-aware mean pool is only a non-learned recording baseline; it is not the JEPA target.

### 2.2 Cross-resolution and cross-sensor context

The 0.5, 1.0, and 1.5 second token grids are presented jointly. Every token carries its physical
start, center, and end time; represented duration; resolution identity; sensor role; acquisition
configuration; and validity mask. Temporal attention may exchange information among all observed
tokens from one sensor. Cross-sensor attention may exchange information among sensors describing
the same local physical interval. Because the joint temporal sequence contains every resolution,
its temporal attention also performs cross-resolution fusion; a separate averaging stage is not
required.

Resolution identity or duration must be injected explicitly before joint attention. Physical-time
position alone is insufficient because tokens at the same center time but with different spans
contain different measurements. The encoder remains a sequence model after fusion: it does not
collapse all resolutions or times to one vector during pretraining.

### 2.3 Structural no-leakage contract

If the sampled student context ends at physical time `T`, a signal token may enter the student only
when:

```text
token.end_time <= T
```

Testing only the center time is invalid: a long patch centered before `T` may contain samples from
after `T`. Future signal tokens are removed or marked as key/value padding before any student
temporal or cross-resolution attention. Replacing them with ordinary learned mask tokens inside a
bidirectional encoder is not sufficient for this experiment, because those positions can still
participate in contextual mixing.

The student may receive empty target-query tokens containing the requested future time, horizon,
duration, sensor role, and acquisition configuration. They contain no signal-derived value. The
predictor may let these queries attend to observed student context. The teacher may attend over the
complete clean region, but teacher states are stop-gradient targets and are never copied into the
student input.

No context or target may cross a source recording, gap, session, subject, stream, or acquisition-
configuration boundary. Multi-resolution overlap leakage is checked in physical time rather than
by token index.

### 2.4 Downstream recording pooling

Tasks requiring one vector per recording add a separate recording pool after the contextual patch
states. The primary learned option is an attention-pooling query over all valid patch states, with
physical time, duration, resolution, sensor identity, and missing-sensor masks retained. A
duration-aware mean remains the parameter-free control.

The downstream task loss trains the attention pool. Under a frozen-encoder comparison, every
encoder receives the same-capacity pooling head and fitting protocol. Under an end-to-end HALO arm,
gradients may update the pool and the student encoder together. The pretraining-only future
predictor and physical decoder are not reused as a classifier.

Pooling must first normalize within each resolution and then combine active resolutions with equal
declared weight. Otherwise a 0.5-second grid dominates a 1.5-second grid merely because it emits
more tokens.

## 3. Training example

Draw one continuous region without crossing a session, subject, stream, acquisition configuration,
or known recording-gap boundary. The design-of-record label-free corpus uses an eight-second source
region. Within that region:

1. Sample a context boundary so the student observes approximately 40-70% of the region.
2. Sample one or more target intervals strictly after the observed context.
3. Vary the gap between context and target and the prediction horizon.
4. Keep context and targets disjoint in physical time.
5. Require targets to contain real, valid samples for the sensor being predicted.

The prediction horizon is measured in seconds, not token indices. This keeps the task meaningful
across native sampling rates and patch resolutions. The default short, medium, and long center-time
horizon bins are `[0,1)`, `[1,2)`, and `[2,3)` seconds. One-second bins remain attainable by every
supported token grid; the earlier `[0,0.5)` bin was structurally empty for one-second continuous
tokens and is not part of the implemented design.

### Multi-resolution masking

Target selection happens in physical time before tokens are selected. The student receives only
tokens whose end time is at or before the sampled context boundary; all later signal tokens are
zeroed and excluded as attention keys and values. This prefix rule is stricter and simpler than
removing only selected target intervals. It prevents a fine-resolution target from being visible
through an overlapping coarse token, or the reverse.

Each token identifies its physical center time, represented duration, resolution, sensor role, and
acquisition configuration. Losses are averaged within each resolution and then across resolutions,
so a resolution with more tokens does not dominate merely because its grid is denser.

## 4. Student, teacher, and predictor

### Student

The student is the trainable HALO encoder. It receives only context tokens and their truthful
metadata. It must not receive signal values, filterbank features, or contextual embeddings from a
target interval.

### Teacher

The teacher is a stop-gradient copy of the HALO encoder. It receives the complete clean region and
produces contextualized target representations at the selected future intervals. A target is the
normalized average of a small declared set of upper teacher layers rather than an unnormalized,
single-layer moving value.

For a target interval `t`, the target is conceptually:

```text
teacher_target(t) = normalize(mean(selected upper-layer teacher states at t))
```

The exact layer set is a recorded hyperparameter and must remain fixed within a run.

### Future predictor

A lightweight, narrower predictor receives the student's context states plus target-query tokens.
Each target query contains only information available before observing the target: requested future
time, horizon, duration/resolution, sensor role, and acquisition configuration. It emits one
predicted latent vector per target.

The predictor is intentionally smaller than the encoder. Its role is to model temporal transition,
not to duplicate the encoder or hide poor representations behind a high-capacity head.

## 5. Objectives

### 5.1 Future latent prediction

The primary loss compares each predicted future latent against its normalized EMA-teacher target.
Use Smooth-L1 or L1 initially because these are less sensitive than squared error to occasional
large residuals. The primary loss is:

```text
L_future = mean_over_valid_targets(
    smooth_l1(normalize(predicted_future), teacher_target)
)
```

The live trainer reports the loss separately by prediction horizon and patch duration, plus target
counts by source dataset. Per-sensor and per-acquisition-configuration breakdowns are offline
acceptance analyses so they do not add synchronization or aggregation overhead to every update. A
shuffled-target control must remain measurably worse than the true target; otherwise the task is not
learning temporal correspondence.

### 5.2 Decode predicted latents to physical measurements

A small decoder receives the **predicted future latent**, not the observed teacher latent. It
reconstructs a frozen, standardized target derived from an independent parameter-free physical
measurement analyzer. That analyzer is calibrated once, checkpointed, excluded from optimization,
and is not the selected student frontend. Consequently, fixed, constrained-learnable, and
continuous-kernel encoders receive the same kind of physical supervision without being allowed to
move their own target. The initial target contains only
measurements whose scale and meaning are already controlled by the frozen corpus calibration:

- normalized filterbank-band energies; and
- signed DC terms for each present accelerometer or gyroscope axis.

The frontend's raw total-energy scalar is intentionally excluded because it is not standardized.
A temporal-envelope target is also deferred until it has a fixed corpus-level normalization and a
measured downstream benefit. This avoids adding nominal physical targets that dominate only because
of scale.

The frozen analyzer's corpus calibration covers every target duration emitted by the selected
frontend. In particular, multi-span targets must not apply statistics fitted only to one-second
patches to their shorter and longer intervals.

Channel validity is a hard mask. Nyquist observability and frequency resolution are continuous
confidence weights, preserving partial information without an arbitrary threshold. These values
weight the loss and are not reconstructed as outputs; otherwise the decoder could reduce its loss
by recovering acquisition configuration rather than motion.

The decoder does not reconstruct raw waveforms. Its purpose is to require the predicted latent to
retain interpretable motion information that is not guaranteed by latent matching alone.

```text
L_physical = masked_mean(
    smooth_l1(physical_decoder(predicted_future), standardized_future_features)
)
```

The live trainer reports error relative to a zero predictor. Training-mean, previous-context, and
temporal-interpolation controls belong in the offline objective audit. A physical target that cannot
beat suitable non-learned controls is removed rather than retained as decorative supervision.

### 5.3 Collapse and redundancy control

Apply variance and covariance regularization directly to the encoder's per-patch latent states used
downstream. Do not rely only on a projector or pooled session vector, because either can remain
healthy while patch representations collapse.

This term initially contains VICReg's variance and covariance components only. Its invariance term
is enabled only when there are two physically justified views that should represent the same motion.
It must not force independence between genuinely different future states or erase orientation that
defines the motion.

```text
L_collapse = L_variance(student_patch_latents) + L_covariance(student_patch_latents)
```

### 5.4 Combined loss

```text
L_total = L_future
        + lambda_physical * L_physical
        + lambda_collapse * L_collapse
```

`L_future` is the main objective. Initial gradient-share targets, used only to choose fixed scalar
weights after a short warmup, are:

- future latent prediction: 70%;
- physical reconstruction: 20%; and
- collapse control: 10%.

The numerically stable launch coefficients before that measurement are `1.0`, `5.0`, and `0.01`,
respectively. They are initialization values, not an assertion that the three raw losses are equally
important.

Measure gradients on the shared encoder, solve the weights once after warmup, and then freeze them.
Do not continuously equalize objective gradients: equal gradient magnitude does not imply equal
scientific importance and moving weights change the optimization target throughout training.

## 6. EMA teacher update

The teacher receives no optimizer update. After every successful student optimizer step:

```text
teacher = momentum * teacher + (1 - momentum) * student
```

Use an example-count-adjusted base momentum and increase it smoothly toward 1 over training. Do not
update the teacher after an overflowed or skipped optimizer step. Teacher parameters, normalization
statistics, and any learnable frontend state used to form targets must all follow the same declared
EMA policy.

The teacher is initialized from the student, stored in checkpoints, and restored exactly on resume.
The run record stores the EMA schedule and effective half-life in processed examples.

## 7. Augmentation policy

Start without independent student/teacher signal augmentation. The information asymmetry comes from
past-only context versus a complete teacher target, so extra corruption is not required to create
the task.

When rotation is tested, apply the same physically valid rotation to student context and teacher
target for a given placement. Independent rotations change the prediction target and can teach the
encoder to discard orientation-dependent motion. Rate, channel-dropout, and text perturbations are
separate one-at-a-time ablations after the clean objective is shown to work.

## 8. Required telemetry and acceptance checks

The live trainer records:

- weighted and unweighted values for all three losses;
- shared-encoder gradient norm and pairwise gradient cosine for each objective;
- gradient norms for the encoder, predictor, physical decoder, and conditioning paths;
- teacher/student parameter distance and EMA update half-life;
- latent standard deviation, covariance, and effective rank;
- true-target loss versus shuffled-target loss and their margin;
- physical reconstruction error versus a zero predictor;
- valid target count and target coverage by horizon, resolution, and source dataset;
- mask-overlap leakage count, which must remain zero; and
- fixed downstream development probes evaluated without fitting on sealed test data.

The offline acceptance audit additionally checks duplicate-vector rate, sensor/configuration target
coverage, and physical prediction against training-mean, previous-context, and temporal-interpolation
baselines. These diagnostics are not represented as minute-level telemetry when they require a
corpus scan or extra model pass.

A run is invalid if target information reaches the student, any resolution exposes an overlapping
target, the predictor or decoder has zero gradient, the teacher is updated by backpropagation, a
large source dominates solely through token count, or resume changes the teacher trajectory.

## 9. Controlled experiment

Run three sample-matched arms with the same encoder, corpus, optimizer budget, and development
selection protocol:

| arm | objective | purpose |
|---|---|---|
| A | historical bidirectional masked JEPA plus VICReg (`--jepa-mode masked`) | external/control comparison |
| B | multi-horizon future JEPA plus collapse control | tests temporal-transition learning |
| C | arm B plus the physical decoder | tests whether physical grounding adds useful information |

Arm C is promoted only if it improves fixed downstream representation probes, not merely its own
reconstruction loss. The current released or best historical encoder remains an external control.

Use `--physical-weight 0` for arm B. Arm C is the default. Both use the same future target planner,
student/teacher encoders, corpus sampler, and downstream selection probes.

## 10. Implementation map

- `training/tokenizer/future_jepa.py`: physical-time target planner, metadata-only future predictor,
  normalized teacher targets, balanced losses, and fixed objective calibration.
- `training/tokenizer/pretrain.py`: corpus integration, generic token-grid handling, past-only
  student/full teacher forwards, independent physical target extraction, optimization, telemetry,
  checkpointing, and resume.
- `model/tokenizer/encoder.py`: explicit duration conditioning and optional upper-layer states.
- `model/tokenizer/transformer.py`: upper-layer state exposure for EMA targets.
- `tests/test_future_jepa.py`: leakage, target, balancing, gradient, and active-head contracts.

The future path requires sensor-granularity tokens but supports the fixed filterbank,
constrained-learnable filterbank, single-span continuous-kernel, and multi-span continuous-kernel
frontends. The filterbank arms use fixed 0.5/1.0/1.5-second grids; the single-span continuous arm
uses its native one-second token grid; and the multi-span arm uses its declared span grid. In every arm,
future selection and leakage checks operate on the emitted physical intervals rather than assuming
a sampling rate or interpreting token indices as time.

## 11. Literature basis

- [I-JEPA](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html): asymmetric context/target prediction with an EMA target encoder and lightweight predictor.
- [data2vec](https://proceedings.mlr.press/v162/baevski22a.html): normalized contextualized targets from multiple EMA-teacher layers.
- [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748): multi-horizon future latent prediction.
- [TimeSiam](https://proceedings.mlr.press/v235/dong24e.html): temporal-distance-conditioned prediction between past and current time-series segments.
- [Context Autoencoder](https://arxiv.org/abs/2202.03026): joint latent prediction and reconstruction from the predicted representation.
- [VICReg](https://arxiv.org/abs/2105.04906): explicit variance and covariance control against collapse and redundancy.
- [V-JEPA](https://arxiv.org/abs/2404.08471): evidence that bidirectional multi-block prediction remains a necessary control rather than assuming causal prediction must win.
