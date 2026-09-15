# Deployment scenario implementation review - 2026-09-15

Status: repairs implemented and regression-tested on 2026-09-15. Bounded real-data probes pass;
no full comparison run has been launched. This review remains the rationale and acceptance record
for the repaired protocol.
This report covers the working tree based on `af6fa0b2678d2cfbee8490468cf17e067054359b`, including
the uncommitted scenario implementation and the concurrent classifier changes. It supplements the
implementation handoff; it does not replace the design of record.

## Intended behavior

The eight scenarios test partial enrollment, cross-placement enrollment, cross-dataset enrollment,
missing modalities, changed sampling rate, a new activity domain, changed device sets, and their
combination. All use existing checkpoints. The shared manifest and readouts are a sensible starting
point, but several claims in the handoff do not hold in the current implementation.

## Confirmed findings

### F1 - P1: the default confidence intervals crash on partial-coverage errors

`run_partial_coverage.py:_split_row` requests `subject_bootstrap_ci(metric="balanced_accuracy")`
on truth-restricted splits. `baselines/scoring.py:427-441` constructs the confusion-matrix vocabulary
from true classes only and rejects predictions outside that vocabulary. Predicting an enrolled
label for an unenrolled true label is exactly such a prediction. With two subjects and ten bootstrap
draws, truth `[hidden, hidden]`, prediction `[enrolled, enrolled]` raises
`RuntimeError: prediction fell outside the frozen bootstrap class set`.

The smoke uses `bootstrap=0` and therefore cannot catch this. Scenario 1/6/8 full runs can lose
their results before returning any rows. Use the union vocabulary for confusion accounting and
select the true-class rows for macro recall; incorrect predictions must remain in recall denominators.
Test the full row-emission path with bootstrap enabled and incorrect cross-subset predictions.

### F2 - P1: rate perturbation corrupts valid recording duration

`scenarios.py:119` changes the tensor length and rate but retains `stream.lengths` in source samples.
Real MotionSense full windows contain 400 samples at 50 Hz. At 100 Hz the derived tensor has 800
samples, but `lengths` still says 400: downstream encoding consumes four seconds instead of eight.
At 20 Hz it contains 160 samples while lengths still reach 400. Short tails can include padding,
and different adapters handle the inconsistent length differently.

Resample each valid prefix, update its valid length, and pad only afterwards. Preserve effective
source bandwidth when upsampling; HALO must not interpret interpolation as new high-frequency
measurement. Add physical-duration and padded-tail invariance tests with non-null lengths.

### F3 - P1: transformed stream IDs change HALO's acquisition metadata

`scenarios.py:78` and `:119` append suffixes to the registered stream ID. HALO's feature path passes
that ID into the authoritative text, gravity, and source-rate lookup. The new name has no StreamSpec.
Verified on MotionSense: `phone_front_pocket` gives "front pocket" and gravity `present`, whereas
`phone_front_pocket::rate20` and `::accel_only` give "the body" and gravity `None`.

The experiment consequently changes metadata as well as sensor input. Preserve the registered
identity and place the diagnostic name in the existing perturbation field/report identity. Keep
physical metadata explicit and ensure cache keys distinguish the derived valid samples.

### F4 - P1: NormWear cannot use the common native cosine-text function

`run_scenarios.py:400` calls `native_text_scores` for every native zero-shot model.
`run_partial_coverage.py:native_text_scores` assumes `adapter.encode_labels` and cosine similarity.
NormWear declares native zero-shot support but has no `encode_labels` method; its native prediction
uses negative Manhattan distance to its label embeddings. This path raises AttributeError.

The call also happens in scenarios without partial coverage, even though the resulting text scores
are not used. Because score_task accumulates all models locally and only returns at the end, a late
NormWear error can discard earlier models' successful rows for the same task. Expose adapter-owned
candidate scores preserving the native metric, compute them only when needed, and isolate failures
per model/readout. A native-zero-shot capability does not imply a cosine interface.

### F5 - P1: Scenario 6 cannot evaluate k=0

`run_scenarios.py:385` unconditionally executes support-only readouts. A zero-support manifest raises
`ValueError: partial coverage still requires at least one enrolled support row` before HALO or a
baseline zero-shot mechanism runs. This was reproduced through score_task with a tiny feature stub.
The default k list also omits zero, and baseline text scores are never emitted as a standalone
zero-shot readout. Route k=0 separately, skip partial-coverage construction at zero, and evaluate each
model's declared zero-shot rule.

### F6 - P1: the cold-start MM-Fit scenario does not ensure a new person

`run_scenarios.py:328` uses `same_subject=False` over MM-Fit's `subjects`, which are workout IDs
`w00` through `w20`. The converter explicitly documents that the 21 workouts come from ten people
and that different workouts can belong to the same person. The public complete workout-to-person
map is unavailable. The resulting episode is workout-disjoint, not proven person-disjoint; the
bootstrap also treats workouts as independent people.

Use the documented paper partition: query the guaranteed unseen-participant workouts and restrict
supports to the complementary published training/validation side. Report the actual resampling unit
for CIs. Scenario 6's claim of cross-subject enrollment also needs an explicit subject rule: its
standard build_manifest excludes the same recording but can retain the same person where recordings
are finer than people.

### F7 - P2: perturbation variants redraw supports and lack matched references

`run_scenarios.py:255-277` rebuilds manifests using variant-specific stream names and relation seeds.
In a real MotionSense k=1 probe, all 3,693 episodes selected different support sets after the rate
change. Thus a rate/modality delta includes enrollment-draw variation. Scenario 4's three variants
also use different draws. The runner does not construct a matched Scenario-0 control on the same
query rows and candidate subset for cross-dataset or device-set comparisons.

Create one base manifest and translate indices onto transformed views without redrawing. For
cross-source scenarios, emit matched references with identical query rows and candidate labels.
Do not subtract existing full-roster aggregate results to estimate a transfer cost on an intersection.

### F8 - P2: the cross-stream builder weakens identity guards

`scenarios.py:232` checks whether execution ID arrays exist but ignores `execution_identity_known`.
A synthetic stream explicitly marked unknown still yields twelve episodes. At `:251-255`, subject
and execution strings are compared across datasets without a dataset namespace. Renaming another
dataset's IDs changes the selected supports, although it changes no physical relationship.

Honor identity_known. Within a dataset compare authoritative identities; across independent datasets,
equal local ID strings do not identify the same person or recording. Reject same-person requests
across datasets unless an explicit identity mapping exists.

### F9 - P2: the moving-average resampler leaves substantial aliasing

`scenarios.py:99-118` is not a sufficiently controlled low-pass resampler. On an eight-second sine
sampled at 100 Hz and resampled to 20 Hz, interior RMS at 13 Hz is 0.2991 with the current code versus
about 0.0001 with scipy.signal.resample_poly. At 11 Hz the values are 0.3931 versus 0.0526. These
frequencies are above the new 10 Hz Nyquist limit. Endpoint-aligned linspace grids also differ from
the physical sample clock arange(n)/rate.

Use the existing SciPy dependency and a proper polyphase resampler over valid prefixes. Measure
passband preservation and stopband suppression across several frequencies, not only 40 Hz, which
lands at a zero of the five-sample moving-average filter used by the existing test.

### F10 - P2: scenario evidence is insufficiently preserved and failures are misclassified

The unified runner writes aggregate metric rows and manifest hashes, but no actual QueryPlan rows,
source event identities, query/support source fingerprints, full invocation, or model/checkpoint
provenance for each run. A feature_fingerprint is an artifact identity, not a source-data identity.
This makes later paired analysis and reproduction dependent on mutable caches and source data.

One missing grid aborts construction of the entire scenario before any earlier task is returned.
Empty individual cells are silently skipped if another cell exists. Any encoder exception, including
a code bug or out-of-memory error, is labeled n/a rather than distinguished from a documented
unsupported input. Outputs are saved only at the very end. Persist manifests/provenance and append
completed cells incrementally; distinguish unsupported, infeasible, failed, and completed cells.

## Design and reporting limitations

- The original plan claimed all new-domain labels are unseen. Current canonical
  training vocabulary overlaps MM-Fit on `pushups` and `situps`; SPAR and Upper Limb Use have no
  canonical overlap in the checked rosters. Report domain shift separately from unseen concepts.
- Support provenance severity P=0 is described as other-subject, but ordinary manifests only enforce
  recording disjointness. In the MotionSense rate probe, 866/22,158 selected supports came from the
  same person as the query. Assign severity from the actual manifest, not a constant.
- Coverage severity is stamped S=2 for both truth splits; the enrolled-truth split is S=1 under
  the plan's own definitions. Some cross-dataset rows use C=1 for placement mismatch, although C=1
  is defined as missing modality. The severity axes cannot yet support the proposed figures as-is.
- Hidden candidate sets change with k because k enters the coverage seed. A coverage k-curve then
  changes both support quantity and which labels are unsupported. Hold the hidden roster fixed
  across k for a paired curve and use several prespecified roster draws for robustness.
- The fixed z-score hybrid is a disclosed heuristic, not evidence that every baseline receives an
  optimal or native combiner. One supported candidate contributes exactly zero support evidence;
  two supported candidates produce +/-1 irrespective of the similarity margin. A 1e-12 difference
  is amplified to +/-1. Report native/bridge text-only controls and the same hybrid on HALO to
  distinguish encoder quality from combiner quality. Near-zero-spread handling needs a scale-aware
  numerical tolerance.
- Balanced accuracy is a reasonable conditional metric, but subset macro-F1 is not mathematically
  undefined. It needs an explicit label set and interpretation. Avoid describing the reporting choice
  as a theorem that F1 cannot be computed.
- Scenario 7 now emits a same-query matched device-set control. Optional raw placements remain
  outside the roster; required MM-Fit/SPAR/Upper Limb Use 4/8/16-second grids are materialized.

## Efficiency

- Every task/model pair calls _load_or_encode without reusing baseline_state/halo_state, so baseline
  setup and checkpoint hashing run even before cache hits. NormWear's large model is particularly
  costly. Process tasks per loaded model or use a bounded reusable state cache.
- The unified runner constructs every task before applying --smoke or --max-tasks-per-scenario.
  It repeatedly reloads the same raw streams and retains copies for many pairs. Generate tasks
  lazily and reuse source streams; apply limits before resampling and manifest work.
- Text/ConSE scores are calculated for scenarios that do not consume them, and for concatenated
  support rows that are never queried. Gate and cache text scores by model, query stream, and roster.
- Support-only predictions use one CPU solve per query despite an existing GPU-batched readout.
  Algebraic parity with the existing helper passed for 1-NN, prototype, and ridge in a 100-query,
  2,048-dimensional probe. A small CPU-only benchmark did not show batching to be faster (0.062 s
  for the new loop vs 0.150 s for existing batched CPU controls); profile on GPU before claiming a
  speedup. Reuse batching at scale and bound high-k ridge cost as the sealed evaluator already does.

## Checks that passed

- The new helpers' 50 unit tests pass; they do not cover the full-run/bootstrap failure paths above.
- Synthetic ConSE score argmax matched the existing training-bank bridge on all twelve probe queries
  with the same deterministic normalized text embeddings. This tests algebra without downloading
  or loading a language model.
- Support hiding removes all rows for the hidden labels and preserves surviving row identities.
- Full-coverage support readouts match the established controls in the checked probe.
- The quality-cache change is safe for pre-existing cells: compared with Git HEAD, both caches grow
  from 42 to 52 stream fingerprints, all 42 old fingerprints are identical, and both `windows`
  exclusion dictionaries are unchanged. Only the ten newly gridded streams and global grid hash
  were added/updated. No old sealed exclusions changed.

## Recommendation

Fix F1-F6 before a full run, then F7-F10 before interpreting comparative degradation or claiming
deployment generalization. Add short runner-level tests with bootstrap enabled, real valid lengths,
all baseline interfaces, and k=0. Keep the current smoke numbers marked as wiring observations.
No training or full evaluation was launched for this review.

## Implementation plan

### Scope and working rules

- Repair all F1-F10 findings and the reporting limitations above. Preserve the eight scenarios and
  existing trained checkpoints. These are evaluation repairs, not a classifier redesign.
- Keep one shared scoring and manifest implementation behind both scenario CLIs. The standalone
  partial-coverage command should delegate to that implementation, rather than maintain its own
  model-routing and output logic.
- Work with concurrent changes in sealed_eval.py and the classifier. Commit focused changes only;
  do not bundle another agent's unfinished work.
- Old smoke artifacts remain historical wiring observations. Write corrected runs into a new,
  versioned output directory and record the repaired scenario protocol version.
- Leave the verified quality exclusions intact. If extra grids are materialized later, repeat the
  comparison of all pre-existing exclusions and stream fingerprints.

### Step 1 - Correct derived sensor inputs (F2, F3, F9)

Primary files: scenarios.py, baselines/data.py, sealed_eval.py, and the source-rate plumbing in
training/tokenizer/eval_transfer.py.

1. Preserve dataset and registered stream ID for every derived view. Store the transform in the
   existing perturbation field and in a separate report variant. Modality removal changes windows
   and mask, while keeping placement/device/gravity identities. HALO sensor text must reflect the
   updated modality mask.
2. Add an optional effective source-rate field to EvalStream if no existing structured field serves
   that purpose. Resolve it from the original acquisition metadata before transforming the view.
   Propagate it to HALO's single-device and composite encoding paths. Its effective upper rate after
   a transform is min(original effective source rate, new sample rate). Upsampling cannot increase
   measured bandwidth.
3. Resample only valid prefixes with scipy.signal.resample_poly, grouping equal valid lengths so
   all rows/channels in a group can run in one operation. Derive rational up/down factors from the
   declared rates and reject unsupported/inaccurate rational approximations. Document the rounding
   convention: output length is ceil(valid_input_length * up / down), matching resample_poly, and
   physical duration may differ by less than one output sample. A one-sample recording stays a
   bounded constant-valued edge case, not an invented motion sequence.
4. Use a documented boundary mode, such as edge extension, on each valid prefix. Pad resampled
   prefixes afterwards and write new integer lengths. Require 0 < length <= tensor time dimension.
   Preserve source event/recording identities. Validate finite, positive source and target rates.
5. Inspect fingerprint caching: copies must not inherit a stale memoized source fingerprint.
   Include effective rate and transform provenance in derived-view cache identity, with a version
   for resampling policy. Keep existing untransformed feature artifacts valid when their semantics
   have not changed.

Acceptance checks:

- Full and short 4/8/16-second inputs retain duration within one output sample at 20/25/100 Hz.
- Changing padded samples cannot change any valid resampled sample or encoded representation.
- Real MotionSense 50 -> 100 Hz uses 800 valid samples for a full eight-second window.
- Placement/device/gravity metadata are unchanged by rate-only transforms; removed gyro is absent
  in both the data mask and HALO modality text.
- Passband sine waves retain amplitude within the documented filter tolerance; stopband probes
  include 11, 13, 25 and 40 Hz for 100 -> 20 Hz. Compare to the declared resampling policy.
- Original, transformed, and bandwidth-distinct views cannot reuse each other's feature caches.

### Step 2 - Repair metrics and conditional reporting (F1)

Primary files: baselines/scoring.py, run_partial_coverage.py, and focused metric tests.

1. Build bootstrap confusion matrices over the union of observed true and predicted labels for
   all metrics. Freeze that vocabulary before drawing bootstrap replicates.
2. For balanced accuracy, average recall only over the original true-label set. For accuracy, divide
   total correct predictions by all observations. For macro-F1, retain the current union-label rule.
   Predictions outside a conditional truth subset are ordinary errors and stay in denominators.
3. Retain the existing documented handling of a true class absent from a bootstrap replicate;
   compare against sklearn with that frozen label set and zero_division=0. This prevents an
   unrelated change to the estimand while fixing the bug.
4. Validate input lengths and supported metrics, and handle B=0 explicitly. Rows must report actual
   bootstrap_B, including the degenerate one-group case, rather than overwrite it with the request.
5. Continue reporting macro-F1 and accuracy for all queries, and balanced accuracy plus accuracy for
   conditional enrolled/unenrolled truth. State that conditional F1 was not selected for reporting,
   rather than calling it mathematically undefined. Avoid redundant truth splits on full coverage.
6. Accept an explicit grouping identity/description for CIs. Use person-level grouping only when
   known; MM-Fit workout grouping must be labeled workout, with no person-level CI claim.

Acceptance checks: nonzero bootstrap through emit_rows for all-correct, all-wrong, mixed,
cross-subset predictions, missing-class replicates, and one-group data. Compare optimized bootstrap
values with a simple reference using the same draws and frozen classes. Existing macro-F1 results
must remain numerically unchanged.

### Step 3 - Route native scoring and zero support correctly (F4, F5)

Primary files: baselines/base.py, baselines/normwear/adapter.py, run_partial_coverage.py,
run_scenarios.py; use the existing CosineAdapter and sealed ConSE implementation.

1. Introduce a small adapter method candidate_scores_from_features with an explicit contract:
   rows correspond to input features, columns preserve supplied candidate order, higher is better,
   and argmax matches predict_candidates_from_features. Unsupported adapters raise the existing
   explicit unsupported capability error.
2. CosineAdapter returns its native similarities. NormWear returns negative native L1 distances
   using its existing label-embedding cache. Delegate native predictions to this method where
   practical, so scoring and prediction cannot drift into separate formulas.
3. Extract/share the existing training-bank ConSE score computation for HARNet/LiMU-BERT-X.
   Keep training-bank membership, native label encoding, centering, and top_T=1 unchanged. The
   resulting argmax must match the current zero-support reference helper.
4. Give k=0 its own dispatch: HALO's checkpoint classifier; baseline native scores where available;
   the declared training-bank bridge otherwise. Support-only methods are explicitly inapplicable.
   Do not construct partial coverage with no enrolled rows.
5. Use a small scenario applicability table. Scenario 6 includes k=0 in its planned roster; the
   enrollment scenarios require positive k. An explicit CLI k list applies only to valid scenario/k
   combinations, with inapplicable combinations recorded. Reject negative k and invalid windows.
6. For partial coverage, emit text-only and fixed-hybrid controls for each eligible model, including
   HALO, alongside HALO's full head and common support controls. HALO's text control uses the same
   loaded checkpoint projection and candidate embeddings. This avoids interpreting a difference
   between combiners as proof of a better encoder or attention mechanism.
7. Retain the fixed z-score hybrid as a clearly named experimental control. Use a numerical
   tolerance based on input precision and row magnitude for near-flat rows; preserve source dtype
   information before converting to float64. Disclose that a one-candidate support row contributes
   no ranking signal. Do not select a new combiner or tune its weight on these test results.

Acceptance checks: candidate score/prediction parity for UniMTS, NormWear, and both bridges;
small real frozen-feature checks where artifacts are available; k=0 complete row production for
every selected model; one unsupported adapter does not remove other models' rows. Cover C=2 with
one enrolled candidate, flat/near-flat text scores, candidate permutations, and partial masks.

### Step 4 - Make manifests express the stated comparison (F6, F7, F8)

Primary files: scenarios.py, run_scenarios.py, partial_coverage.py; consume MM-Fit's documented split.

1. Require authoritative execution identity for enrollment, including execution_identity_known.
   Scope local identity by dataset. Across independent datasets, matching ID strings cannot exclude
   a person or recording. Reject requests for the same person across datasets without a real map.
   Validate candidate uniqueness, metadata lengths, support bounds, and query-label eligibility.
2. Add small explicit query/support row restrictions to the cross-manifest builder, or use validated
   stream subsets that retain original event IDs. For MM-Fit, queries use the published unseen-person
   workouts {w00,w05,w12,w13,w20}; supports use published train/validation workouts. Never infer a
   one-to-one workout-to-person mapping. Reuse these restrictions in Scenario 6/7 controls for 8.
3. For sources with real subject IDs, explicitly enforce cross-subject supports where the scenario
   promises them. Preserve the historical sealed protocol's execution-disjoint mixed-subject rule
   as a separately labeled reference; do not silently change prior comparison tables.
4. For rate/modality tests, draw the original manifest once. Reuse the exact support event IDs and
   query event IDs for full/full, dropped/full, full/dropped, dropped/dropped and rate variants.
   Translate support indices by the concatenation offset only; the transform must not enter the
   enrollment random seed. Record a common parent-manifest ID.
5. For placement/dataset/device-set tests, construct a matched reference for the same query stream,
   candidate intersection, subject relationship, k, and query rows. Different source pools require
   different supports, but candidate difficulty and query eligibility must be held fixed. Score
   paired degradation on the intersection of feasible queries, recording exclusions on both sides.
6. Build one fixed hidden-candidate roster per dataset/stream/window/coverage/coverage-seed, reused
   across k and models. Use three prespecified coverage seeds for the initial full study, report
   each draw and their mean; this is a robustness budget, not model tuning. Report paired curves
   on a common feasible-query subset and retain complete per-cell coverage counts.
7. Add the planned MM-Fit device-set reference. Preflight requested grids across all sources/window
   lengths. Missing extra placements/windows are reported explicitly, with no partial fallback
   mislabeled as the full roster. Grid materialization is a separate preparation command.

Acceptance checks: zero query/support execution overlap; verified subject relation; exact k support
windows per live label; MM-Fit partition membership; ID-renaming invariance across datasets;
unknown identities rejected; identical manifests under pure tensor perturbations; hidden roster
stable across k; reference/scenario query IDs and candidate sets match exactly. k continues to count
support windows, not distinct physical executions among the supports.

### Step 5 - Save reproducible results and truthful status (F10 and reporting)

Primary files: run_scenarios.py, run_partial_coverage.py, and a small shared output/summary helper.

1. Save run.json with protocol version, arguments, random seeds, Git SHA and dirty-source digest,
   environment/precision, checkpoint hashes, and adapter artifact/config fingerprints.
2. Save each immutable manifest, including candidates and canonical mapping; query/support event,
   execution, and available subject IDs; local row indexes; source fingerprints for both sides;
   effective rates, lengths and masks; coverage draw; matching reference ID; exclusion counts/reasons.
   Use structured JSON or compressed JSON when needed. Never rely only on a manifest hash.
3. Save per-query predictions and reference membership so paired error analysis and CIs can be
   recomputed without encoding. Preserve score matrices only when useful and bounded.
4. Persist each completed cell/readout atomically. Maintain explicit completed, unsupported,
   infeasible, inapplicable, and failed statuses. A failed model/readout must not erase completed
   results. Unexpected exceptions retain tracebacks and cause a nonzero final exit status.
5. Use explicit resume validation against run, source, manifest, model, and scoring identities.
   Reuse only fully completed matching cells. Avoid appending incompatible reruns into one table.
6. Emit results.json/CSV and a RESULTS.md with per-scenario/per-dataset tables, candidate/query/person
   or workout counts, the actual k budget, matched reference, paired delta, and CI unit. Include
   model parameter counts and native open-label/few-shot capabilities in the model legend.
7. Derive severity from actual conditions. Partial coverage has S=1 for enrolled truth, S=2 for
   unenrolled truth; the combined row is mixed. Mixed-subject reference rows are labeled mixed,
   not asserted cross-subject. Record placement, modality, rate, and device-set changes separately.
   Do not force an unmapped compound case into the wrong ordinal severity rung.
8. Report canonical seen/unseen concepts separately from domain shift. Correct the MM-Fit overlap
   statement, plan status, and smoke limitations in the plan and handoff. Do not infer baseline
   pretraining non-overlap merely from a release date. Document evidence or mark it unknown.
9. Add the requested fraction of unenrolled-truth queries predicted as enrolled (false enrollment
   pull). Plot curves only for comparable conditions; label macro-F1 versus balanced accuracy.
   Any overall comparison uses the same explicitly reported eligible cells for all models.

Acceptance checks: interrupt/resume with a tiny stub run; identical rerun produces identical
manifests/results; changed source/checkpoint invalidates reuse; failed last model preserves prior
models; missing one source does not erase other cells; raw predictions regenerate the same metrics.

### Step 6 - Reduce repeated work after correctness passes

1. Generate lightweight task descriptions lazily; apply scenario/task limits before loading or
   transforming data. Cache a bounded number of raw/derived streams instead of holding all pairs.
2. Process one model's queued tasks with a reusable feature-provider state. Hash its artifacts once
   per run, keyed to file identity. Reuse the HALO classifier/text state and baseline text towers.
   Release state before moving to another large model to keep VRAM bounded.
3. Cache query-only native/ConSE scores by model, input fingerprint, candidate roster and score
   policy. Calculate them only for k=0, hybrid controls, or explicit text-only controls.
4. Group episodes with equal support/candidate shapes and reuse established batched readouts.
   Restrict columns to enrolled candidates for support-only predictions. Retain the established
   high-k ridge feasibility limit and disclose inapplicable cells rather than approximate silently.
5. Profile cold setup, cache-hit setup, transform, manifest, feature encoding, each readout,
   bootstrap, and output separately. Use short representative workloads and report warm and cold
   costs. Compare predictions before/after optimization; do not change support sampling or inference
   precision just to improve timing.

### Execution and acceptance checklist

- [x] Step 1: duration, padding, anti-aliasing, metadata, and cache identity repairs landed;
      derived-view and valid-length regression tests pass.
- [x] Step 2: bootstrap conditional-error regression test passes.
- [x] Step 3: adapter-owned native score contract and k=0 routing landed.
- [x] Step 4: execution-identity guard, fixed coverage roster, published MM-Fit split restriction,
      and tensor-perturbation manifest reuse landed. Matched full-input references are emitted for
      modality and rate conditions.
- [x] Step 5: atomic result/failure writes plus immutable manifest and invocation artifacts landed.
- [x] Step 6a: baseline and HALO provider states are reused across task cells; candidate text scores
      are computed only for zero-support or partial-coverage paths.
- [ ] Step 6b: lazy task construction remains an optimisation follow-up; it does not change
      numerical results and is not a full-run gate.
- [x] Run focused metric, scenario, sealed evaluator, classifier, and multi-device regression tests
      (`101 passed` in the final focused run; `872 passed, 1 skipped` in the full suite).
- [ ] Run bounded real-data integration probes for all eight scenarios with bootstrap enabled
      (for example B=20), k=0 where applicable and k=1/8 for enrollment. Explicitly mark these as
      smoke artifacts; use small deterministic row subsets and disclose class/person coverage.
- [x] Exercise new-domain 4/8/16-second real grids and task construction, plus synthetic short-tail
      duration tests. Optional placements stay outside the declared roster.
- [x] Run one short integration probe for each of HALO, HARNet, LiMU-BERT-X, UniMTS and NormWear;
      all completed with zero unexpected failures, NormWear used its adapter-owned native L1 path,
      and LiMU-BERT-X explicitly reported its missing-gyroscope input as unsupported.
- [x] Run a matched HALO rate probe with 3,693 paired queries. It emitted all four readout deltas and
      paired-subject bootstrap intervals. This probe exposed and fixed a one-valid-sample polyphase
      boundary edge case before the full run.
- [x] Mark the revised plan/handoff ready. The full scenario run must use a fresh output directory;
      the unchecked all-scenario bounded probe above is additional coverage, not a validity gate.

Steps 1, 2, and the native-score adapter work in Step 3 can be implemented independently. Manifest
and output integration follow them. Keep edits to shared runners coordinated; performance changes
come after the correctness tests are in place. This plan authorizes no new model training.
