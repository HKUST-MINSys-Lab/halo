# Recording lengths, training crops, and efficient batching

Status: active implementation plan, 2026-09-10. The frozen-representation Task-1 trainer now
implements complete/cropped query views, central-rank crop caps, complete-batch length matching,
and padding telemetry. Raw-input cropping, end-to-end encoder batching, and correspondence-aware
support-fragment supervision remain the next implementation stages. No training was launched.
Baseline: commit `cb6ce68`. Re-read the worktree before editing because other agents may be active.

## 1. Outcome and scope

Use variable-length, complete query recordings and independently bounded support executions.
During training also expose the model to contiguous crops. Group similar lengths for efficient
encoding without changing which subjects, activities, and support relationships were sampled.
Each enrolled execution remains a separate support, regardless of its duration or chunk count.

This work covers the common recording/view/batching infrastructure, Task-1 detection, and the
retained support-classification control. They have different outputs and supervision; do not merge
their heads or losses. Task 2 and Task 3 may reuse packing without enabling crop augmentation:
truncation can change Task-2 change measurements and Task-3 event/count supervision. JEPA keeps
its separately specified prefix/future-target rules and context-duration configuration.

Primary development and test evaluation uses the complete declared query interval and complete
enrolled execution. A declared interval can be a source-native, exhaustively annotated block rather
than an entire file. Never extend evaluation into unannotated time and call it negative background.
Optional partial-support evaluation is a separately named protocol, fixed before test scoring.

## 2. Current implementation to inspect

| File or component | Current behavior | Required change |
|---|---|---|
| `applications/motion_monitoring/data/contracts.py`, `data/cache.py`, `data/examples.py` | Raw recordings, timestamps, masks, source provenance, crop utilities | Reuse contracts; validate crop boundaries and provenance |
| `applications/motion_monitoring/sequence.py` | Native patching, honest final partial patch; HALO context-overlap chunking | Batch independent recording views; preserve output coverage and physical times |
| `applications/motion_monitoring/baseline_encoder.py` | Released encoders use their required receptive windows and group equal sample lengths | Preserve native input contracts; permit efficient batching of views |
| `applications/motion_monitoring/task1/train_full.py` | `_training_episode` takes a query crop; CLI default is 60 seconds | Replace the fixed crop with the explicit complete/cropped policy |
| `applications/motion_monitoring/task1/episodes.py` | Separate padded reference/query tensors; source and execution checks | Keep checks, add view provenance and explicit partial-event handling |
| `applications/motion_monitoring/task1/full_evaluation.py` | Complete manifest query interval, separately bounded reference | Preserve deterministic interval coverage and reference semantics |
| `applications/motion_monitoring/representation_cache.py` | Frozen full-timeline and bounded-event caches | Distinguish full-context representations from independently encoded cropped views |
| `training/support_classifier/sampling.py` | `Recording` actually indexes a grid window; optional multiple windows per support | Index bounded recordings/executions, not arbitrary grid fragments |
| `training/support_classifier/train.py` | Loads `PretrainDataset` windows; optional means over sampled support windows | Load planned raw views; produce one vector per query/support execution |
| `training/support_classifier/encoding.py`, `collate.py` | Shared encoder metadata and collate boundary | Reuse encoder call contract and extend to length-grouped batches |

Do not assume increasing `windows_per_execution` recovers a complete execution. Random windows may
overlap, omit signal, or belong to a multi-action session. For the classification corpus, inspect
converted `data/datasets/<dataset>/sessions/*/data.parquet`, manifests, and assembly utilities.
Build a lightweight interval index into source sessions. Reconstruct from grids only when exact
coverage, ordering, overlap removal, boundaries, and tail preservation can be proved. A source with
only pre-windowed samples remains explicitly pre-windowed; never invent a longer recording.

Also verify the application HALO wrapper supports the selected frontend's actual output contract.
Its existing `_encode_chunk` reads `per_patch`; do not assume this automatically provides correct
joint multiresolution/multispan features or a common temporal grid. Verify all three retained
frontends: fixed 1 second, fixed 0.5/1/1.5 seconds, continuous spans 0.5/1/1.5 seconds.

## 3. Define the units before sampling

Reuse `RawRecording` and `SensorStream`. Add small immutable records, preferably in
`applications/motion_monitoring/data/recording_views.py`, for:

- An indexed source interval: dataset, source/session/subject/execution IDs, stream/config key,
  source time bounds, annotation granularity, source reference and split.
- A planned view: interval identity, crop start/end on the source clock, complete/cropped flag,
  parent execution identity, valid-duration estimate, and deterministic view seed.
- A batch plan: sampled episode identities, views, encoding groups, and mappings back to episodes.

Do not put arbitrary arrays, encoder state, or CUDA tensors in worker-side planning records.
Measure durations from valid source timestamps/sample support, not padded storage lengths.
Distinguish source session identity (leakage grouping) from bounded execution identity (support
counting). Label runs are not automatically individual repetitions; preserve annotation quality.
Select the subject/session split before crops or grouping. All descendants inherit that split.

## 4. Sampling and crop policy

First draw a logical optimizer-step group using the existing task's source, label, subject,
compatibility, support-count, and negative/absent rules. Freeze those relationships before duration
optimization. A logical group is the set of episodes contributing to one optimizer update; it can
be processed in several smaller physical batches. Do not select supports because they happen to
have convenient lengths or redraw an inconvenient dataset away.

Query policy:

1. Randomly choose complete or cropped mode for each logical query group.
2. Complete mode retains each declared query interval.
3. In cropped mode, sort the group's real durations. Choose an observed duration uniformly from
   the central half by rank, excluding strict shortest/longest ranks when possible. This becomes
   the group's maximum query duration. Use the median for groups of one or two; all ties are valid.
4. Each longer query gets a contiguous crop with an independently drawn start time. Shorter
   queries stay complete. Never stretch, repeat, wrap, or concatenate signal to meet this cap.
5. Validate annotations and feasible matching after cropping. If no valid crop exists after a
   small bounded number of attempts, keep the full query and record the fallback. Do not silently
   remove the episode or relax compatibility.

Support policy:

1. Independently for each enrolled execution, choose complete or cropped mode.
2. In cropped mode, draw a retained fraction of that execution and a contiguous start position.
   Clamp the minimum duration to task/encoder feasibility, not a universal fixed number of seconds.
3. Draw the same policy for every support class, including distractors. Crop choices cannot depend
   on whether a support is correct for the query. Do not synchronize query and support crop phases.
4. If too short or semantically ineligible, retain the original support and record why.
5. Reuse one planned support view for all queries sharing that support set in the same logical
   group. Distinct planned views of the same source are not eligible for embedding deduplication.

Starting configuration, to be recorded as engineering defaults rather than established optima:

| Setting | Initial proposal |
|---|---|
| `query_complete_probability` | 0.5 |
| `support_complete_probability` | 0.5, for task-eligible support crops |
| `support_retained_fraction` | Uniform in [0.5, 1.0] |
| Duration grouping | Reorder within the already sampled logical group |
| Memory budget | Calibrate from a short profile for each frontend/device |

Keep the central-rank rule internal to the versioned policy initially. Avoid separate duration
ranges for each dataset or separate knobs for every packing heuristic. Provide `complete` and
`mixed` recipe presets; preserve explicit overrides in saved configuration. Keep a small existing
logical episode count initially, then increase only on measured throughput and coverage evidence.

## 5. Preserve task meaning when cropping

Classification: use complete bounded single-action intervals. A retained crop must still be
covered by the same source label with sufficient valid signal. Ambiguous transitions are excluded
by a declared rule. Keep a support's label and execution ID; k counts original independent
executions, not windows, patches, or augmented copies.

Task 1 query: keep all complete targets inside the selected interval. For an event cut by a crop,
either choose another crop or keep the intersected event explicitly loss-ignored; never relabel the
fragment as background. Matcher paths cannot cross missing-data gaps or synthetic seam guards.
Respect existing present/absent sampling; random cropping must not silently change that mixture.

Task 1 support: an arbitrary fragment of a discrete execution is NOT automatically a template
with the full execution's onset and offset. Keep complete references as the initial default for
such examples. Enable cropped references with boundary supervision only when a correspondence is
known, for example the exact time mapping retained from a synthetic insertion of the same donor.
The target is then the matching fragment's true interval, with metrics named accordingly. Independent
same-action repetitions without a correspondence do not gain invented phase boundaries. A separate
presence-only objective would be a design extension; do not add it as part of this batching change.
For periodic/bout references follow the existing source-specific protocol and disclose its extent.

Read `docs/tasks/TASK1_REFERENCE_RESOLUTION_SPEC.md` section F before updating source rules. Earlier
sections contain historical decisions. Do not revive retired per-repetition inference merely to
make a crop fit. If the source lacks correspondence, report a crop-ineligible fallback. Thus 0.5
support-complete probability applies to eligible cases; realized complete share may be higher.

## 6. Encoding and pooling efficiently

Implement one planner/packer, preferably `applications/motion_monitoring/batching.py`, reused by
the task and classification callers. Avoid building a second general training framework.

- Flatten the views from all logical episodes into encoding work. Deduplicate only identical
  source interval, crop, augmentation seed, frontend/config, and conditioning inputs.
- Group by encoder-required input layout, native rate where necessary, resolution layout, and
  approximate valid token count. Query and support work can share an encoding batch because their
  role is assigned downstream. Role alone should not prevent efficient packing.
- Use ordinary dense padded tensors within each small group first. Keep sample, patch, channel,
  sensor, and resolution masks distinct. Packing unrelated recordings into one attention sequence
  would require block-diagonal isolation; do not add that complexity in the first implementation.
- Restore output ordering with explicit view-to-episode indices. Maintain gradients through all
  gathers, concatenations, resolution fusion, and pooling.
- For classification use the encoder's existing trainable recording pooling over all valid tokens
  where feasible. Each complete/cropped execution yields one vector. Do not mean-pool a random
  selection of windows and call it complete-recording pooling.
- For detection retain the sequence and source-clock intervals. Multiresolution tokens must map
  to the downstream time grid with a declared fusion rule; a scale is not an extra independent
  occurrence. Preserve the frontend's duration conditioning and native receptive fields.
- Long recordings use bounded encoder context/chunking. Keep each represented output once, with
  context overlap to reduce chunk-edge effects. This is a declared local-context model, not assumed
  mathematically identical to unrestricted full-recording attention. Train/eval use the same rule.
- For classification beyond one encoder chunk, use a documented hierarchical masked pooling path
  retaining all signal. Treat adding trainable pooling as an architecture change and test its
  gradients/checkpoint metadata. Avoid equally weighting chunks that represent unequal durations
  through an accidental unweighted mean.

Estimate cost using the actual frontend output: raw CNN sample storage, valid/padded tokens,
attention sequence lengths (quadratic where applicable), and Task-1 query-by-reference alignment
matrix size. Equal seconds across two frontends do not imply equal memory or compute.
Keep a modest safety margin based on a short measurement. Process outliers singly or in chunks;
do not silently drop them. Do not claim that accumulating gradients automatically bounds memory:
graphs retained for all chunks of one full-recording pooled loss can still grow. Use activation
checkpointing or recomputation where necessary, and measure the real peak.

CPU workers plan views and load raw slices; the main process owns GPU encoding. Use a bounded
prefetch queue, pinned buffers when beneficial, and nonblocking transfers. Preserve existing AMP
policy and numerically sensitive FP32 computations. Avoid per-sample device synchronization,
unbounded caches, and repeated encoding of the same unmodified support view.

## 7. Loss weighting and reproducibility

Keep logical sampling and loss normalization independent of physical batch size. Compute one
optimizer update after all pieces of the logical group. A short physical batch must not receive
the same total weight as a much larger one merely because both invoke backward once.

For classification, sum per-query losses divided by the logical query count (or retain the explicitly
defined episode-balanced objective). For the existing Task-1 endpoint loss, compute positive and
negative valid counts over the logical group, sum the corresponding loss terms across physical
batches, and reproduce its balanced reduction exactly. If changing to episode-balanced loss,
declare that separately and compare it; do not hide it inside packing.

Gradient clipping, scheduler advance, optimizer update, and any EMA update occur once per logical
step. Calibration sees a representative mixture of real durations. Seed view draws from stable
run/step/episode/view identities; worker timing and bucket ordering must not affect crop choices.
Resume stores the data-policy version and logical progress. Strictly validate incompatible resumes.

## 8. Frozen caches and evaluation

Cropping cached contextual embeddings does not reproduce encoding a cropped raw signal. Use one
of two explicit modes:

- Independent input crops: crop raw data before encoding. Frozen encoders may use a finite,
  predetermined cache of such views shared across compared models. Fingerprint crop bounds, model
  weights, preprocessing, context policy, and augmentation. Trainable encoders re-encode online.
- Full-context feature crops: slice a complete timeline's cached embeddings for a head-only
  experiment. Name this mode accurately; it does not test input-truncation robustness.

The primary mixed-input recipe uses the first mode. Do not regenerate all caches before mechanics
and timing are verified. Reuse unchanged full-view caches when provenance matches.

Primary evaluation covers every valid part of the declared query and the full declared enrolled
execution. Respect explicit existing exemplar intervals for bout data until a separately versioned
protocol replaces them. Apply the same raw intervals, validity rules, and support identities to all
encoders, while preserving each released encoder's native rate/window requirements. Do not pool
Task-1 templates to a single classification vector. Partial-support metrics must state whether they
score an aligned fragment or a whole execution. Tune thresholds only on the declared held-out
development/calibration data and preserve the existing source split protocol.

## 9. Telemetry and short verification

Log aggregate values at the existing reporting cadence, not per-patch GPU synchronizations:

- Original/retained duration quantiles by query/support; complete/cropped proportions; fallback
  counts; valid tail coverage; annotation-based rejection reasons.
- Valid versus allocated raw samples, tokens, and DTW cells; padding fraction; logical versus
  physical batch sizes; real seconds/tokens processed; unique executions and reused views.
- CPU draw/load/collate time, GPU encoding/head/backward time, throughput, and peak VRAM.
- Source/label/subject/config and duration-bin coverage before/after packing; observed k; present
  versus absent queries; support length distribution by correct/distractor status.
- Query-encoder, support-encoder, frontend, fusion/pooling, and head gradient health on short probes.
  Changing padding must not alter valid outputs/loss beyond expected numeric tolerance.

Required tests and measurements:

1. Pure planning tests: mixed/tied lengths, tiny batches, deterministic worker-independent draws,
   resume reproducibility, unchanged sampled identities, and full recording fallback.
2. Provenance tests: crops cannot cross source gaps, subject/config boundaries, or split boundaries;
   copies do not create independent supports. Reject concatenating noncontiguous grids.
3. Crop tests: final sample coverage, arbitrary native rates, short honest tails, complete-target
   preservation, loss-ignored partial events, crop-ineligible Task-1 references, known synthetic
   fragment mapping, and invalid/empty interval handling.
4. Packing tests: compare individually encoded views with grouped views in eval mode; perturb
   padding; permute work order; check exact reconstruction and no recording-to-recording leakage.
   Training dropout need not be bitwise identical across layouts; disable it for equivalence tests.
5. Gradient tests: compare packed versus reference logical loss/gradients with dropout disabled;
   show nonzero finite gradients through query and support encoding/pooling for all three frontends.
   Include all-negative Task-1 batches and uneven final physical batches.
6. Real-data smoke: a few short/medium/long eligible examples across sources and each selected
   released encoder. Frozen encoders have no gradients; their heads do. Use synthetic outliers to
   test tails without loading an hour-long signal for the first test.
7. Shared-GPU-aware profile: check GPU use, run a few warmup steps and 10-20 measured steps with
   preloaded examples, then a short loader-inclusive run. Compare the SAME logical episodes and
   crop plans for naive and grouped batching. Report throughput/VRAM; no claimed one-hour training
   estimate without measured example counts and full-pipeline timing.

## 10. Delivery and acceptance

Implement in reviewable commits: (1) interval inventory and view contracts, (2) planning/cropping,
(3) encoder batching/masks, (4) Task-1 integration, (5) classification full-execution loading and
pooling, (6) validation/telemetry/documentation. Preserve concurrent changes and work on the single
active branch; do not restore retired evidence/retrieval systems.

Update the existing task documents and entry points to reference this contract. Replace the fixed
60-second primary crop setting with the versioned policy; remove competing flags once their callers
are migrated. Record changes in result/cache/checkpoint protocol IDs. Historical results remain
identified as historical; do not silently relabel them as variable-length results.

Done means: complete coverage in primary evaluation; complete/cropped training views where semantics
permit; execution-disjoint enrollment; retained dataset mix; masks and logical losses verified;
all three HALO frontends and selected frozen encoders smoke-tested; measured padding/throughput;
clear counts of sources still limited by pre-windowed storage. Report residual limitations honestly.
Do not launch full training or fleet-wide cache generation as part of this implementation handoff.
