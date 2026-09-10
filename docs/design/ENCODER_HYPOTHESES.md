# Encoder hypotheses for personalized motion monitoring

> **Design and evidence record, 2026-08-28.** This document asks whether HALO's encoder mechanisms
> address the representation failures that matter for Tasks 1-3. It separates implemented behavior,
> literature-based motivation, prior project evidence, and claims that still require an experiment.

## 1. Bottom line

There is no single encoder deficit shared by all three application tasks. Most published wearable HAR
encoders are optimized and evaluated for classifying fixed windows. Our tasks instead require:

- accurate event boundaries in a continuous timeline;
- local sequence geometry that survives moderate speed changes;
- sensitivity to meaningful differences between executions of the same movement;
- stability under rate, placement, remounting, and missing-sensor changes; and
- local representations that preserve repeated temporal motifs and separate them from background.

HALO directly and credibly addresses **rate-aware physical analysis, explicit observability, missing
channels, and timestamped local representations**. Those are implemented properties, not yet proven
application advantages. The fixed filterbank and continuous-kernel frontend make different tradeoffs
that should remain matched experimental arms.

HALO does **not** yet establish that free-text acquisition descriptions improve a movement
representation. The existing parity experiment found a mean held-out gain of only `+0.0086` balanced
accuracy and a sign reversal on two of four datasets. Text conditioning remains an ablation and
nuisance-control hypothesis, not a headline contribution.

The likely gap is therefore broader than frontend design. It includes a mismatch between generic HAR
pretraining and the information each application needs. A representation trained to collapse all
examples of one activity can erase exactly the within-activity differences needed by Task 2. A pooled
window classifier can also discard boundaries and phase structure needed by Tasks 1 and 3 and by
the optional motion-proposal baseline.

Four levels of evidence must not be conflated:

1. **Input compatibility:** the encoder can run on different rates and channel sets.
2. **Physical comparability:** controlled versions of the same physical signal produce comparable
   analyses, with unavailable content marked rather than invented.
3. **Learned cross-domain robustness:** independent executions remain useful across subjects,
   placements, devices, and sessions.
4. **Task sufficiency:** the representation preserves the exact boundary, recurrence, or execution
   difference needed by an application.

HALO has strong evidence for levels 1 and 2. Levels 3 and 4 remain experimental questions. Calling
the encoder "heterogeneity-aware" should therefore mean that it accepts and exposes acquisition
differences, not that it has already learned invariance to every difference.

## 2. What existing encoders already solve

HALO should not claim that prior models ignore heterogeneity:

- CrossHAR treats cross-dataset changes in user, device, and placement as domain shift and combines
  physically motivated augmentation with hierarchical self-supervision
  ([Hong et al., 2024](https://doi.org/10.1145/3659597)).
- UniMTS uses synthetic motion, a skeleton graph, and rotation augmentation to cover device location
  and mounting orientation ([Zhang et al., 2024](https://doi.org/10.48550/arxiv.2410.19818)).
- NormWear uses channel-aware processing to support heterogeneous multivariate wearable inputs
  ([Luo et al., 2024](https://doi.org/10.1145/3803808)).
- HARNet demonstrates that representation quality and transfer can improve substantially through
  self-supervision on 700,000 person-days of accelerometer data
  ([Yuan et al., 2024](https://doi.org/10.1038/s41746-024-01062-3)).
- Dataset benchmarks already identify sampling rate, hardware, placement, attachment, protocol, and
  participant behavior as separate sources of shift
  ([Napoli et al., 2024](https://doi.org/10.1038/s41597-024-03951-4)).

These methods commonly obtain compatibility through a fixed input contract, resampling, normalized
channels, rotation augmentation, synthetic coverage, channel-independent processing, or data scale.
HALO's narrower hypothesis is that **explicit physical rate and observability information can retain
honest differences without forcing every source into one apparently uniform input**.

## 3. What the three tasks require

### 3.1 Task 1: arbitrary demonstrated movement detection

This is query-by-example sequence matching, not activity-label classification. The representation
must keep the order of local motion while tolerating bounded speed variation and ordinary execution
variation. Activity detection followed by DTW is established for inertial gesture matching
([Li and Hu, 2024](https://doi.org/10.3390/jimaging10050123)).

Relevant HALO mechanisms:

- physical seconds for patch timing and alignment;
- comparable features across native rates;
- local patch embeddings for constrained subsequence DTW; and
- masks that prevent absent or unobservable channels from looking like measured zeros.

The continuous frontend may help when the same frequency content occurs in a different within-patch
order. The fixed filterbank may be sufficient when cadence and energy dominate. This is a direct
fixed-versus-continuous ablation, not a reason to promote the continuous arm in advance.

Acquisition descriptions could reduce false matches across incompatible placements, but only if
they outperform both neutral text and structured non-language metadata on unseen sessions. They do
not encode the demonstrated movement and should never be used as a semantic shortcut.

### 3.2 Task 2: change quantification

This task creates the hardest invariance tradeoff. The representation should ignore acquisition
nuisance while retaining changes in phase, duration, intensity, smoothness, range, and compensation.
Clinical movement-quality work emphasizes accurate kinematics, measurement reliability, uncertainty,
and minimum detectable change rather than class accuracy alone
([Unger et al., 2024](https://doi.org/10.3389/fdgth.2024.1359776),
[Felius et al., 2022](https://doi.org/10.3390/s22030908)). Wearable features can estimate clinical
movement-quality scores, but published systems combine signal features, task structure, and clinical
supervision rather than relying on a generic class embedding
([Adans-Dester et al., 2020](https://doi.org/10.1038/s41746-020-00328-w)).

Relevant HALO mechanisms:

- signed DC and amplitude preserve posture and movement magnitude;
- physical frequency bands preserve cadence and spectral change;
- continuous kernels may preserve phase-local waveform differences; and
- configuration metadata and masks can stratify, rather than silently merge, remounted sessions.

Main risk:

- JEPA/VICReg or rotation objectives can deliberately collapse differences that Task 2 needs. A
  generic HAR transfer score cannot show that the retained geometry is suitable for change
  measurement.

Task 2 therefore keeps raw physical measurements beside latent distance. It must report test-retest
reliability, false change under remounting, and sensitivity to known execution changes. A latent
metric that is discriminative but unreliable is not useful.

### 3.3 Task 3: recurrent motion discovery

The representation must preserve repeated subsequences, variable duration, and separation from
incidental or outlier motion. Prior factory work finds that outlier segments can shift inferred
operation timing unless the discovery method models multiple motifs and process structure
([Xia et al., 2020](https://doi.org/10.1145/3411836)). Personalized multidimensional motif discovery
has also been proposed specifically where patient exercises are too individual for a fixed
supervised vocabulary ([Balasubramanian et al., 2016](https://doi.org/10.1109/JSTSP.2016.2543679)).

Relevant HALO mechanisms:

- a dense timestamped patch sequence;
- physical-time comparability across recordings; and
- local temporal structure from the continuous frontend when recurrence depends on waveform order.

Weak or irrelevant mechanisms:

- acquisition text does not define recurrence;
- a six-second encoder context does not solve long-horizon motif search; and
- cross-sensor attention cannot recover a missing sensor or infer intent.

The same-motion metric, matrix-profile control, recurrence graph, DTW re-scoring, overlap
suppression, and human confirmation remain downstream algorithms. HALO is useful only if its
sequence improves held-out-identity matching, motif coverage, background separation, boundary
stability, or runtime over raw and physical-feature controls.

## 4. Mechanism-by-mechanism assessment

| HALO mechanism | Implemented property | Gap it may address | Evidence status | Main risk |
|---|---|---|---|---|
| fixed physical-Hz filterbank | 32 constant-Q bands, explicit Nyquist and duration resolution masks, amplitude and signed DC | cross-rate comparison, cadence and physical-frequency structure | implementation tested; application benefit unproven | power discards within-band phase and sub-second order |
| continuous physical-time kernels | native-time integral kernels, exact sample offsets, four ordered subframes per one-second token | onsets, waveform shape, phase-local matching | cross-rate mechanics tested; prior generic-HAR result mixed | extra capacity can learn source/subject fingerprints |
| source-rate observability | high bands are masked according to the native source, even after storage-rate conversion | prevents an upsampled low-rate source from claiming unavailable detail | strong implementation rationale and tests | mask itself exposes acquisition configuration, which may become a shortcut |
| channel and sensor masks | absent channels remain absent rather than zero-valued evidence | variable sensor sets and missing modalities | implementation tested | cannot restore information that was never measured |
| signed acceleration DC | preserves gravity direction and static posture separately from AC energy | posture, tilt, and execution geometry | implementation tested | remounting and body orientation can look like movement change |
| physical-time positions | transformer positions use seconds rather than token indices | variable patch spacing and sequence alignment | implementation tested | current checkpoints were trained on local six-second windows, not long sessions |
| temporal attention | contextualizes each patch with nearby patches | local phase and transition context | implemented | generic pretraining can smooth boundaries or collapse useful variation |
| cross-sensor attention | mixes simultaneous sensor streams in the dual trunk | coordinated multi-placement motion | implemented | benefit may not transfer to single-watch or single-phone deployment |
| factored acquisition text | MiniLM sensor identity plus axis-role projection | unseen wording and configuration context | operational, but prior parity result is effectively null | source-string memorization; text cannot correct calibration or observability |
| future-JEPA plus physical grounding | predicts multi-horizon EMA-teacher states from past-only context, decodes fixed physical features, and controls patch collapse | temporal dynamics and physically informative local representations without labels | implemented; application benefit unproven | predictable dynamics can still smooth rare changes or omit unpredictable detail |

### 4.1 What the current clean pretraining recipe does not enforce

The current defaults set rotation, rate augmentation, channel dropout, signal jitter, scaling,
gravity removal, and text perturbation probabilities to zero. This makes the reference experiment
simple and avoids training away unmeasured task information. It also means robustness to those
changes does not arise from an explicit positive-pair objective.

The label-free future-JEPA objective operates within one source window. It does not explicitly
identify walking from different datasets, two subjects performing the same exercise, or two
placements observing one synchronized movement as equivalent. The frontend can make those inputs
physically better formed, but the objective is not directly teaching cross-source movement identity.

Neither frontend is inherently rotation invariant. The fixed arm computes per-axis band energy and
signed DC. The continuous arm feeds ordered xyz responses into a dense CNN. Both may retain useful
orientation and posture information, but a remounted device can therefore change the representation.
The default factored text path describes placement and gravity state; it does not geometrically align
axes or calibrate a remounting. Placement and orientation robustness remain open measurements.

## 5. What has already been measured

### 5.1 Frontend and conditioning mechanics

On 2026-08-28, the focused CPU suite passed `102` tests across:

- `tests/test_filterbank.py`;
- `tests/test_continuous_kernel.py`;
- `tests/test_factored_conditioning.py`; and
- `tests/test_sensor_granularity.py`.

These tests cover rate comparability, source-rate observability, duration-dependent token counts,
missing axes, accel-only streams, sensor folding, text perturbation, and gradient-bearing continuous
parameters. This is correctness evidence, not downstream efficacy evidence.

The continuous frontend's synthetic cross-rate comparison against 100 Hz is recorded as `0.986` at
20 Hz, `0.987` at 25 Hz, and `0.998` at 50 Hz. The difference that remains is expected numerical
integration error. This validates its physical-time implementation on controlled signals.

### 5.2 Acquisition-text parity

The prior full-description versus neutral-description probe used the same trained checkpoint and
held-out subject split:

| dataset | full text BA | neutral text BA | difference |
|---|---:|---:|---:|
| MotionSense | 0.7912 | 0.7564 | +0.0348 |
| RealWorld | 0.6486 | 0.6562 | -0.0076 |
| Shoaib | 0.8962 | 0.8314 | +0.0648 |
| InclusiveHAR | 0.4290 | 0.4864 | -0.0574 |
| **mean** | **0.6912** | **0.6826** | **+0.0086** |

The project screening noise floor treated effects below about `0.012` as inconclusive. Because the
model was trained with full text and only neutralized at inference, neutral text was also the harder,
out-of-distribution arm. The lack of a consistent degradation is evidence that the checkpoint did
not depend strongly on acquisition text. The definitive application experiment still retrains
matched full, neutral, and no-text arms rather than treating this inference perturbation as final.

This result agrees with an empirical study finding that wearable natural-language supervision can
underperform supervised and self-supervised learning when sensor heterogeneity is high and text is
not rich or diverse ([Haresamudram et al., 2024](https://doi.org/10.48550/arxiv.2408.12023)). It also
explains why large sensor-language systems build rich statistical, structural, and semantic captions
rather than relying on a small vocabulary of acquisition strings
([Zhang et al., 2025](https://doi.org/10.48550/arxiv.2506.09108)).

### 5.3 Fixed versus continuous frontend

The prior generic-HAR comparison was mixed: the continuous arm improved a zero-shot point estimate
and a learned reranker, while direct 1-NN representation quality was slightly lower. Dataset-level
bootstrap intervals crossed zero. It is therefore incorrect to call either frontend universally
better. The application tasks provide the more relevant test because they directly score temporal
localization, matching, and change sensitivity.

## 6. Decisive experiment ladder

Every encoder must feed the same Task-1 full-timeline matcher, Task-2 alignment/report, and Task-3
dense motif evaluator. Thresholds and tiny metric heads are fit only on development data. Test subjects,
sessions, and tasks remain sealed.

### 6.1 Floors and external encoders

1. Raw IMU with the task's established algorithm.
2. Magnitude/gravity-aligned and physical-feature sequences.
3. Primary released HARNet, UniMTS, and NormWear representations through faithful adapters, with
   ImageBind reserved for an optional generic multimodal appendix control.
4. HALO representations under the matched task algorithm.

This comparison asks which available representation is most useful. Because upstream data and model
scale differ, it does not isolate architecture.

### 6.2 Within-HALO architecture ablations

Use the same corpus, split, optimizer, encoder width, trunk, checkpoint-selection rule, and task
algorithm. Change one factor at a time:

1. fixed filterbank versus continuous kernels;
2. temporal patch sequence versus pooled representation;
3. native-rate physical analysis and observability masks versus a common-rate resampled control;
4. complete masks versus a fixed complete-channel subset;
5. temporal-only versus dual temporal/cross-sensor trunk on genuinely synchronous multi-sensor data;
6. no acquisition text versus neutral text versus full descriptions; and
7. full descriptions versus compact structured configuration IDs, if text shows a gain.

The structured control is important. If arbitrary text and a categorical configuration ID perform
equally, the gain is configuration lookup rather than language generalization.

### 6.3 Nuisance and information probes

For each representation, measure:

- source-rate, placement, dataset, subject, and session predictability;
- same-motion similarity across rates, placements, sessions, and subjects;
- different-motion separation within the same subject and configuration;
- boundary sharpness around annotated event starts and ends;
- speed-warp stability without collapsing different movements;
- Task-2 test-retest reliability and minimum detectable change; and
- false change caused by remounting, rotation, channel removal, and rate conversion.

Dataset identity that remains predictable after controlling for physical configuration is a warning
that a frontend or text path learned a source shortcut. Perfect nuisance invariance is not the goal:
Task 2 must retain real subject-specific execution differences. The target is a measured tradeoff.

### 6.4 Component decision gates

- **Continuous frontend:** keep as the preferred arm only if it improves boundary F1, Task-1 event
  AP, Task-2 known-change sensitivity at matched reliability, or Task-3 motif coverage with a
  subject-level confidence interval excluding a negligible effect.
- **Acquisition text:** promote only if full text beats both neutral/no-text and structured controls
  on unseen descriptions or configurations, while reducing remount false positives.
- **Cross-sensor attention:** retain as an application claim only if synchronous multi-sensor tasks
  improve without harming single-sensor inputs.
- **HALO overall:** claim a representation contribution only if it beats raw/physical controls and
  released encoders with the same downstream algorithm, or offers a clear robustness/efficiency gain.
- **Task pipeline:** if the same small metric head improves every encoder similarly, attribute the
  contribution to the task formulation and method, not the HALO frontend.

## 7. Current claim boundary

The defensible statement today is:

> HALO implements a native-rate, physical-time, mask-aware temporal IMU representation designed for
> heterogeneous acquisition. Its fixed and continuous frontends provide complementary frequency and
> local waveform views. Whether these properties improve event localization, demonstrated-motion
> matching, execution-change measurement, or motif discovery is the subject of the matched
> application experiments.

The following statements are not yet supported:

- acquisition-language conditioning is the missing gap in current wearable encoders;
- the continuous frontend is better than the fixed filterbank;
- generic HAR transfer implies reliable rehabilitation change measurement;
- handling rate and missing channels implies placement or subject invariance; or
- a stronger encoder alone solves long-horizon segmentation or motif discovery.

## 8. Practical recommendation

Keep the current low-complexity application plan. Start with frozen encoders and non-parametric Task
0-2 floors. Task 3 trains the same small pairwise metric independently over each frozen
representation, because arbitrary event identity is its direct supervision. Run HALO fixed/no-text
and continuous/no-text as the core representation ablation, with full text as a controlled arm. Add
end-to-end encoder fine-tuning only after the frozen experiment identifies a concrete representation
failure.

This ordering can produce a publishable result under either outcome: a HALO mechanism wins under a
matched test, an established encoder is the better foundation for the application, or simple
physical methods are sufficient and the contribution is the validated application pipeline.
