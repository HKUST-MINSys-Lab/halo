# Training and evaluation readiness repairs

Date: 2026-09-13
Status: code repairs partially implemented on 2026-09-13; corrected data artifacts and
authoritative evaluation caches remain pending regeneration.

This is the consolidated implementation plan for the two verified readiness sweeps.
It takes precedence over conflicting implementation details in
[EVAL_EXPANSION_PLAN_20260913.md](EVAL_EXPANSION_PLAN_20260913.md), not over the scientific
goals of that protocol. It is a temporary repair checklist, not another model specification.
Update task status and evidence here as work lands; keep architecture in DESIGN_OF_RECORD
and evaluation rules in EVALUATION_PROTOCOL. Journal entries and historical results stay intact.

## Scope and decisions

- Fix code, contracts, tests, and current documentation. Do not launch full training, a full
  sealed evaluation, or an expensive HARNet head refit without a separate go-ahead.
- Brief CPU tests and synthetic/development-data GPU smokes are allowed. Inspecting sealed
  metadata, clocks, and preprocessing is allowed; do not use sealed scores to choose repairs.
- Keep the fixed multi-resolution filterbank, polarization features, shared encoder, learned
  recording pooling, parameterless neighbours control, and separate zero/enrolled token-mixer
  weights. Do not redesign the classifier, add auxiliary losses, or alter the enrollment task.
- Preserve k = 0, 1, 2, 4, 8, 16, 32, 64, 128, subject CIs, per-dataset results, and 4/8/16 s
  evaluation. Report feasibility and query coverage per cell; do not assume all k remain feasible
  after correcting data. k counts support windows per candidate, not unique subjects/executions.
  All supports must remain execution-disjoint from their query.
- Preserve legacy RealWorld artifacts for historical reproduction. Correct physical alignment
  takes priority over byte identity for newly generated data. Do not silently relabel new data
  as the historical protocol.
- Mantis is a separate pending feature under MANTIS_BASELINE_PLAN, not a readiness repair.
  Do not add it to a reported roster until its independent implementation/provenance tests pass.
- Do not modify unrelated dirty-worktree changes, delete old checkpoints, or commit generated
  caches and large arrays. Capture git HEAD/status before work and commit only reviewed hunks.

## Verified starting evidence

- The first sweep passed 104 scoped tests; it did not include compatibility-key tests.
- Rechecking `tests/test_compatibility_key.py` produced 4 failures, 19 passes. All four failures
  arise from unmapped `the forearm` and `the thigh` placements.
- RealWorld native waist metadata: legacy 11,237 windows / 132 executions; w6 11,266 / 130;
  98 event IDs removed and 127 added. Raw proband1 running timestamps also show about 0.5 s
  placement offsets that independent timestamp zeroing erases.
- A 4 s single-resolution patch at 200 Hz raises `patch length 800 exceeds dft_size 512`.
- A nonexistent 99 s grid request resolves to the existing legacy directory instead of failing.
- Previous bounded probes found changed single-device learned-pool embeddings, source-rate
  collapse in XRF composites, and native 4/8/16 s forward support in UniMTS and NormWear.
- The other report's full-suite count is external evidence, not independently rerun here.

## Execution order

1. Add regression tests and implement W1-W5 (model/data contracts). Small independent changes
   can be separate commits. Do not regenerate full artifacts yet.
2. Implement W6-W8 (cache identities, evaluation dispatch, baseline contracts).
3. Implement W9-W10 (validation coverage, active recipe/docs).
4. With correctness tests passing, implement W11 (bounded optimizations and durability).
5. Execute W12: short integrated tests, artifact rebuild preflight, then the necessary
   preprocessing/cache rebuilds. Report measured duration before any unexpectedly lengthy job.

Parallel ownership, if subagents are used: encoder/filterbank; RealWorld converter/grids;
baseline adapters; trainer RNG/validation. A single integrator owns sealed_eval.py and final
cache/interface changes. Agree W4's source-rate schema before editing callers in parallel.

## W1. Restore learned pooling without changing the mean path

Files: `model/tokenizer/encoder.py`, `tests/test_multi_device_support.py`, encoder/pooling tests.

- Restore `_learned_recording_pool(h, weights.gt(0))` on the temporal path. Do not feed
  pre-averaged `per_device` tokens to the learned pool.
- Keep hierarchical device-balanced averaging only for the parameter-free mean path and
  diagnostics. Preserve every valid patch x sensor token for learned pooling.
- Verify both query and support encoding retain the gradient path through this pool and encoder.
- Version the pooling semantics in new checkpoint provenance. For old checkpoints, distinguish
  documented historical pooling from any runs produced during this regression. Do not silently
  claim numerical equivalence for checkpoints trained with the accidental pre-average behavior;
  mark those runs affected and reproduce from their saved source snapshot if necessary.

Acceptance:
- Learned-pool input contains P*S tokens, including separate accel/gyro tokens on one device.
- Fixed old/new-reference weights give identical single-device outputs in float32 when restoring
  the documented historical computation; compare with the historical function, not only the mean.
- Equal per-device means but different modality vectors need not produce the same learned output.
- Valid tokens and pooling parameters receive finite nonzero gradients in both query/support
  roles; padded tokens cannot affect valid outputs. Existing mean-pool identity tests still pass.

## W2. Repair placement vocabulary and RealWorld clocks/populations

Files: `data/scripts/curate/compatibility.py`, `data/datasets/realworld/convert.py`, deployment
metadata, grid-builder integration, compatibility/converter tests.

- Explicitly map `the forearm` to the existing side-unspecified forearm site and `the thigh`
  to the existing side-unspecified thigh site. Do not invent laterality or equate wrist/forearm.
- Retain Unix timestamps through CSV parsing. Match recording parts by explicit part identity,
  with timestamp checks; do not trust list position where one member can have a missing part.
- Validate finite, ordered clocks. Handle duplicates under a documented deterministic policy;
  split/reject backward seams and acquisition gaps rather than interpolating across them.
  Derive gap handling from the existing time-grid policy and observed/native sampling cadence,
  not a new dataset-specific silent tolerance.
- Construct one uniform physical clock using integer sample indices at the output rate, e.g.
  origin + arange(n)/rate. Do not stretch an arbitrary duration to n samples with linspace.
- Resample onto valid coverage only, preserving true start offsets. No endpoint extension may
  masquerade as measured sensor data; absent channels/intervals must stay absent or excluded.
- Preserve valid single-placement recordings when another placement is missing. For composites,
  derive an explicit common-overlap view and exclude unavailable members/events transparently.
- Use stable execution IDs, absolute interval identity and a shared window anchor. If the current
  session format cannot express differing member coverage, extend the converter/grid contract
  deliberately; do not fix this by silently intersecting already-misaligned window IDs.
- Keep each matched single/composite comparison on declared physical intervals. Document where
  composite eligibility reduces coverage; do not claim that all single-only recordings belong
  to the composite population.
- Write corrected artifacts to a versioned/staging namespace, validate, then promote atomically.
  Preserve old waist grids/manifests under their historical protocol identifier, outside active
  discovery. Do not overwrite history to satisfy a byte-identity assertion.

Acceptance:
- All 23 compatibility tests pass.
- Synthetic clocks with 0.5 s offsets align common impulses; shorter/missing sensors never
  create constant extrapolated tails or remove valid unrelated single-placement executions.
- Missing/reordered nested recording parts cannot join unrelated executions.
- A bounded raw RealWorld sample agrees with preserved physical times, sample spacing and
  overlap bounds across accel/gyro and placements.
- Emit old/new execution/window counts and added/removed IDs with reasons, for every duration.
  Fixing the population is a protocol revision, not evidence of a performance improvement.

## W3. Make grid discovery and duration loading fail closed

Files: `baselines/data.py`, `data/scripts/eda/grid_io.py`, `data/scripts/build_grids.py`, quality
scanners and their tests.

- `_grid_dir`: use the requested qualified duration if valid; allow legacy fallback only for
  a 6 s request with a verified legacy 6 s schema. Missing 4/8/16 s must raise an actionable error.
- Validate requested duration against metadata before loading/scoring, and validate tensor/length
  bounds. Legacy metadata missing duration needs an explicit known-schema rule, not the requested
  duration substituted as truth. Small floating-point serialization differences may use tolerance.
- `discover_grids(None)`: preserve legacy-first behavior, then select explicit w6 if no legacy
  exists. Never pick an arbitrary w4/w8/w16. For a stream with neither, raise a clear error
  requiring a duration (or use an existing explicit caller policy), never silently pick or omit it.
- Explicit-duration discovery and quality scans must use the same resolution rule as evaluation.
- Make new grid writes explicit about their duration; legacy output is a deliberate compatibility
  operation. Assert manifest duration, source duration and reported duration agree everywhere.

Acceptance: shuffled directory creation order has no effect; missing/mislabelled duration raises;
legacy 6 s remains deliberately loadable; all 4/8/16 quality and evaluation references agree.

## W4. Preserve each sensor's acquisition bandwidth and device identity

Files: `pretrain_data.py`, `eval_transfer.py`, support encoding/trainer forwarding,
`model/tokenizer/encoder.py`, `filterbank.py`, related tests.

- Define a canonical per-channel source-rate representation: item `(C,)`, collated `(B,C)`.
  Retain scalar/per-recording `(B,)` input compatibility and broadcast it explicitly. Avoid
  guessing whether a 1D tensor is per-batch or per-channel when sizes happen to match.
- Keep sample-grid rate separate: resampling to 50 Hz does not turn a 25 Hz acquisition into
  a 50 Hz source. Resolve hardware bandwidth consistently in train, single eval and composite eval.
- `merge_device_items` concatenates source rates, masks and sensor/device IDs; no global min/max.
- Filterbank observability and polarization gates use each triad's source rate. Validate coherent
  triad rates or use the limiting rate within that triad only. Adding another device cannot
  change existing devices' feature observability.
- Carry device_id through public filterbank `forward()` as well as `analyze()`. Preserve the old
  positional API by adding a keyword argument, not inserting it between existing positional args.
- Document padded-channel handling; masked channels cannot contribute features or invalid divides.

Acceptance: mixed 25/50 Hz composite retains the 50 Hz member's valid high-frequency bands while
the 25 Hz member stays masked above its bandwidth; solo/composite feature observability matches;
old scalar inputs match broadcast inputs; batch size equal to channel count is unambiguous;
forward/analyze-project paths agree with device-aware gravity on a two-device example.

## W5. Share bounded analysis and preserve recording-tail coverage

Files: both collators in `pretrain_data.py`, export callers in `eval_transfer.py`, duration tests.

- Factor the existing bounded-analysis preprocessing into one helper used by single- and
  multi-resolution collation in both training and evaluation. Reuse the current anti-aliasing
  policy; retain native source-rate metadata and true physical time. No repeated resampling.
- Remove the silent short-tail discard. Prefer shifting the final nominal-width patch back
  to the true recording end, retaining a bounded overlap with its predecessor, rather than
  constructing an oversized merged patch that exceeds the DFT budget. Deduplicate equal bounds.
  For recordings shorter than a nominal patch, retain one valid partial patch.
- Record actual starts/ends/durations. Where parameter-free duration weights combine overlapping
  patches, account for unique represented coverage rather than pretending all durations are
  disjoint. Learned pooling continues to receive the valid tokens and true metadata.
- Apply any shared patch-boundary change consistently to live train/export code. Test or guard
  retired JEPA callers without re-enabling pretraining; do not silently change historical runs.
- Version this preprocessing change. Avoid claiming native baseline internals preserve every
  sample identically; the guarantee is no avoidable wrapper crop/drop and equal supplied evidence.

Acceptance: 4/8 s single patches at 200 Hz fit the bounded DFT; exact multiples have no duplicate
tail; 4.07 s input is represented through 4.07 s at every resolution; changing its tail changes
features on a deterministic signal probe; padding changes do not; train/export contracts match.

## W6. Version feature caches and rebuild only affected dependencies

Files: `sealed_eval.py`, `baselines/data.py`, baseline feature-config hooks and reference-bank cache.

- Centralize extraction identity: checkpoint hashes + effective configuration + extraction schema
  version + relevant encoder/adapter/preprocessing implementation digest. Hash relevant source
  dependencies, not the entire dirty repo; memoize immutable artifact digests within a run.
- Include input masks, channel order, true lengths, source rates, gravity/placement/config text,
  device mapping, execution identity and valid raw samples in the relevant input/cache identity.
  Use canonical serialization with shapes/dtypes and field names. Avoid fragile string concatenation.
- Treat fingerprinted input objects as immutable or explicitly invalidate their memoized fingerprint
  on mutation. Validate loaded cache rank, row count, dimension, dtype and finiteness too.
- Write arrays and metadata atomically. A partial cache is not a cache hit. Propagate invalidation
  through query features, training reference-bank features, and derived heads/results.
- Preserve historical caches/results under their own version; do not eagerly delete them.

Acceptance: changing extraction code, masks, config text or source bandwidth misses cache;
padding-only changes do not change measured-input identity; unchanged input hits; malformed or
partial arrays fail/rebuild clearly; a changed query extraction cannot reuse an old reference bank.

## W7. Correct zero-shot dispatch and reporting

Files: `training/support_classifier/sealed_eval.py`, token-mixer evaluation helpers/tests.

- Route by checkpoint mechanism/capability, not membership of HALO in one blanket bridge set.
  A token-mixer checkpoint at k=0 uses its zero-shot parameter branch, query recording vector,
  candidate label tokens and proper role tags, with no fabricated support tokens. At k>0 use
  its enrolled branch. Preserve one shared encoder and checkpoint-specific pooling/text settings.
- Neighbours-only HALO checkpoints keep the documented training-support semantic bridge at k=0;
  do not call an absent learned head. At k>0 retain the agreed parameterless readouts.
- Keep training-bank bridge results explicitly named as a control, not labelled native learned
  HALO. Native text-aligned baselines use their own candidate scoring for both single/composite
  features where their adapter declares support.
- Remove the unconditional composite k=0 N/A. Apply N/A only to an unsupported mechanism, e.g.
  a bank-dependent method without a properly matched composite reference bank. Do not fabricate
  such a bank by relabelling single-device examples; implementing that bank is not required here.
- All result statuses, including N/A and errors, include duration, devices, protocol/manifest ID,
  mechanism, requested k, eligible query count and exclusion reason. Score identical query plans
  for methods within a cell; report coverage changes across k instead of assuming comparability.
- Add cheap actual CLI/driver tests with fake encoders so helpers cannot pass while dispatch is wrong.

Acceptance: sentinel zero/enrolled heads prove branch use at k=0/k=1; neighbours checkpoints
cannot masquerade as mixers; native composite baselines reach prediction; only unsupported
bank paths return N/A; no head fitting, threshold tuning or checkpoint selection touches sealed data.

## W8. Align baseline training/inference contracts and remove needless chunking

Files: HARNet, UniMTS, NormWear, LiMU-BERT-X adapters; `baselines/base.py`; baseline tests.

- HARNet: make retained `_fit_head`, native probability prediction and feature export call the
  same length-aware feature extraction helper with the same source preprocessing. Bump head-fit
  preprocessing identity. Never accept the old cropped-feature head as current. Do not launch a
  full refit during repair; the current sealed training-bank path does not need this fitted head.
- UniMTS/NormWear: verify native 4/8/16 s handling and internal padding/positional behavior with
  the actual released model. Use full-window native forward where supported. Handle true partial
  lengths explicitly; batch by compatible lengths without allowing padding to contaminate pooling.
- LiMU-BERT-X: retain justified fixed-length chunking; use all measured chunks, correct valid-tail
  weighting and original channel order/units. Preserve baseline-native internal pooling.
- Consolidate wrapper aggregation through one shared weighted-mean primitive that supports
  grouped/ragged chunk owners, so vectorized implementations need not materialize giant stacks.
  Distinguish equal-device aggregation from measured-duration chunk aggregation.
- Remove mutable per-call channel-count state where it can be passed locally. Do not introduce
  concurrent sharing of a baseline instance until its inference state is demonstrably reentrant.

Acceptance: head-fit/export helpers agree on the same 4/8/16 s signal; a crop-trained head cache
is rejected; native models actually receive the full declared interval; a tail impulse outside
an old crop reaches the input/features; shared pooling matches current mathematics; multi-device
channel/joint ordering and collision/missing-modality tests still pass. No baseline weight updates.

## W9. Deterministic device sampling and multi-device validation

Files: dataset sample-loading API, trainer prefetch/draw/validation code, resume tests.

- Pass an owned per-step/per-sample RNG into device selection. Derive it from data seed, step and
  stable sample occurrence/role using SeedSequence or existing helpers, never Python hash or
  global np.random. A worker's completion order must not change a sampled batch.
- Keep normal `__getitem__` compatible through a deterministic default; the trainer should use
  the explicit seeded sampling API. Repeated query/support occurrences may have independent
  subsets, but that independence must replay exactly.
- Include device probability/max count, selection policy version, data fingerprint and corrected
  preprocessing/pooling semantics in the trajectory guard and checkpoint provenance.
- Add deterministic single-device and multi-device development episodes using only validation
  subjects/executions. If no composite is eligible, report coverage/N/A; never substitute sealed data.
- Report validation separately by k=0/k>0 and single/composite. Freeze manifests and the checkpoint
  selection rule before training. Do not silently change the existing declared selector to a new
  averaged score; initially retain it and add composite health reporting, or explicitly version a
  documented selector change based only on development data.
- Telemetry: actual device-count distribution, eligible versus drawn composites, per-source mix,
  valid samples versus padding/tokens, separate query/support counts, per-regime metrics and
  finite gradient norms for encoder, pooling and each active head. Throttle diagnostics, not training.

Acceptance: same step under 0/1/2 workers gives identical sampled IDs/device subsets and tensors;
uninterrupted and resumed next steps match on CPU; changed trajectory settings reject resume;
query/support draws remain independently seeded; validation composites are fixed and leak-free.

## W10. Unify active defaults and documentation

Files: trainer CLI/config, DESIGN_OF_RECORD, EVALUATION_PROTOCOL, EXPERIMENT_ROADMAP,
training/model READMEs, RESULTS status/banner.

- Resolve one active recipe: fixed multi-resolution filterbank with the agreed expanded
  0.5/1/2/4/8 s ladder and implemented polarization policy. Keep 1 s as an explicit named control,
  not an accidental default. A patch span is not the query/support recording length; preserve
  valid partial spans on shorter recordings and the existing recording-length curriculum.
- Persist effective resolved configuration. Explicit overrides and old checkpoint configurations
  must not inherit new defaults silently. Avoid a default resolution list that makes an explicit
  single-patch CLI option get ignored; reject ambiguous combinations or use existing recipe logic.
- Disconnect JEPA and continuous/multispan from active commands and promoted rows, retaining
  explicit historical loading/reproduction through the existing archival guard/path. Do not
  delete classes needed to load archived artifacts; no new archival framework is required.
- Update current architecture/pooling, baseline capabilities, native duration consumption,
  zero-shot mechanisms and data-version disclosures. Mark old mismatched evaluations historical
  rather than current; do not replace their numbers before reruns exist.
- Mantis stays visibly planned/unimplemented until its separate work is complete.

Acceptance: a CLI config-resolution test gives the intended active recipe; explicit 1 s control
works; incompatible resume fails; current docs agree; archived journal/results remain unchanged.

## W11. Optimize only after correctness is locked

Files: evaluator provider lifecycle, batching, cache setup and result output.

- Process providers with a bounded lifecycle: load each baseline once, reuse across duration/cells,
  release it before the next large provider. Reuse HALO encoder and mixer states. Do not retain all
  models in VRAM at once. Prebuild shared immutable manifests independently of provider order.
- Resolve artifact/config hashes without loading heavy model weights when possible. Cache-hit
  runs should not instantiate unused GPU models. Reuse digests within a run, not across changed files.
- Batch learned-head inference by actual token count and attention memory, including C*k support
  recording/label tokens. Microbatch queries, never silently truncate supports/candidates or reduce k.
  If one episode exceeds capacity, report it honestly; do not change full attention semantics.
- Profile native long-window baselines after W8 before selecting encode batch sizes. Keep a
  conservative explicit memory ceiling; measure 4/8/16 s and representative high-k synthetic cells.
- BF16/autocast may cover verified operations; keep sensitive normalization, cosine reductions,
  softmax/loss and reported metrics appropriately stable. Match float32 outputs within a documented
  tolerance and check ranking changes/ties; never quantize caches silently as an optimization.
- Save completed results and manifests atomically per cell/readout, with a resumable completion
  key covering all fingerprints. Do not lose an entire suite if a later provider fails.

Acceptance: one provider initialization across several cells; cached rerun avoids extraction;
batch-size/chunking changes preserve float32 predictions; interruption resumes without duplicate
rows or mixing versions. Report step/encode latency and peak VRAM, not an unsupported runtime promise.

## W12. Integration, artifact rebuilds and release gate

Run in this order and record commands, true exit codes, elapsed time and outcomes:

1. Focused tests: compatibility, converter/grid IO, filterbank/polarization, multi-device
   train/export, support loader/sampling, token mixer, baseline durations, sealed driver.
   Run pytest directly or preserve pipefail; a successful `tail` is not a successful test run.
2. CPU synthetic integrated train-forward/backward and fake-provider evaluator at k=0/1/4,
   all durations, single/composite, with mask/tail/permutation/provenance edge cases.
3. Brief GPU development smoke for both neighbours and token-mixer classifiers (about 3-5
   optimizer steps each). Query/support encoder and pool gradients finite; the appropriate
   zero/enrolled head has gradients when its regime occurs. Inactive-branch zero gradients are
   expected, not failure. Check loss/weights remain finite and frozen baselines remain frozen.
4. Bounded real-checkpoint native-duration probes for retained baselines; synthetic high-k
   learned-head capacity probe with actual candidate sizes. No sealed classification scoring.
5. Preflight artifact sizes/elapsed sample rate. Rebuild corrected RealWorld sessions, then
   duration grids, then quality exclusions, then manifests, then features/reference banks.
   Other datasets need no raw reconversion unless their inputs changed; feature extraction
   identities may nevertheless require new caches. Refitting the optional HARNet head is a
   separate authorized job, not a prerequisite for its training-bank evaluation.
6. Before promotion, validate artifact metadata, finite data, lengths, common clocks, coverage,
   label/execution identity and source fingerprints. Promote corrected artifacts atomically.
   Produce a machine-readable readiness summary with missing/pending jobs and exact reasons.
7. Final broader relevant test suite (exclude long training/download integrations explicitly),
   git diff review and documentation check. Commit reviewed fixes in logical units; never stage
   unrelated agent work or large untracked caches. Report branch/commit IDs and what remains.

Full-run authorization criteria:
- All above correctness regressions pass, not just the original narrow suite.
- Corrected data and quality artifacts are materialized, not merely code-ready.
- No stale feature/head/reference-bank cache is accepted under the current protocol.
- Zero-shot and enrolled mechanisms are explicitly identifiable in every output row.
- Known unsupported cells are honest N/A; valid cells cannot silently fall back to another window.
- Short end-to-end smokes pass and a bounded high-k memory/latency measurement is recorded.
- Remaining nonblocking items (Mantis, historical refits, optional optimizations) are listed without
  claiming they are implemented. Do not declare all possible models/k ready if only a subset ran.

## Suggested commit boundaries

1. Learned pooling restoration + gradient/parity regressions.
2. Placement vocabulary + RealWorld clock/coverage correction + converter tests.
3. Strict grid duration selection + source-rate/device metadata + shared analysis/tails.
4. Cache provenance/invalidation + checkpoint-aware evaluation routing.
5. Baseline feature parity/native duration + common aggregation.
6. Deterministic sampling/resume + development composite coverage.
7. Active recipe/docs + bounded evaluator optimizations + final readiness evidence.

## Implementation status

| package | status | note |
|---|---|---|
| W1 | implemented | Learned pooling again receives all patch x sensor tokens. |
| W2 | code implemented; artifacts pending | Placement vocabulary and converter clock handling changed; RealWorld sessions/grids must be rebuilt before use. |
| W3 | implemented | Missing non-6 s duration grids now fail closed; discovery chooses w6 deterministically. |
| W4 | implemented for fixed filterbank | Per-channel acquisition-rate metadata now reaches fixed-filterbank observability. Retired continuous frontends retain their historical scalar-rate interface. |
| W5 | implemented | Both collators use bounded analysis; multi-resolution tails are retained. |
| W6 | partially implemented | Cache schema/input fingerprint validation is versioned; atomic writes and source-code digests remain a follow-up. |
| W7 | implemented | Current token-mixer checkpoints use their zero-shot branch before the legacy bridge. |
| W8 | partially implemented | HARNet shared-fit feature contract and UniMTS full-window input are fixed; NormWear wrapper refactor remains a follow-up. |
| W9 | implemented | Device subsets are drawn from per-step RNG and validation enables the declared device regime. |
| W10 | partially implemented | Active design document retires continuous multispan; CLI/default cleanup remains a follow-up. |
| W11-W12 | pending | Require final integration review and corrected artifact/cache generation. |
