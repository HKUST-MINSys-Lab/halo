# Residual classifier audit and training profile

Date: 2026-09-14. Scope: current `support_classifier_v3`, fixed multi-resolution filterbank,
classification training and its checkpoint/evaluation handoff. Independent code inspection,
bounded real-training-roster GPU profiles, synthetic numerical probes, and focused tests.
The initial findings were implemented and verified with focused tests plus a three-step real-data
smoke; no full training or sealed evaluation was run.

## Implementation update

Findings 1-6 below are resolved in `support_classifier_v3`: new checkpoints serialize DFT size;
known support-classifier v1/v2 snapshots retain their 512-point migration while historical Phase-A
snapshots retain the 256-point fallback; mixed information regimes are loss-grouped correctly;
unsupported candidates use a uniform log-prior reference; resume inherits the saved trajectory;
telemetry reports actual GT support; and learned-head rows include classifier parameters.

Token composition now normalizes/scales content, role and instance ingredients before attention.
This changes learned-head state layout, hence v3. v2 remains evaluable with its original raw sum.
The compiler is opt-in and fails loudly if unavailable; the default uses eager BF16.

## Decision

**The corrected classifier is ready for a training run.**
About **35-40 minutes for 35,000 steps is a reasonable planning estimate**, using eight loader
workers and eager BF16 execution. The measured training-loop extrapolation is 29.1 minutes;
the additional allowance is not a measured full-run guarantee. No reduction in model, examples,
window duration, resolution count, or step count is needed to reach that estimate.

## Confirmed issues

### 1. Encoder checkpoint restoration changes the frontend

`training/support_classifier/encoding.py:114` builds a 512-point DFT encoder but omits
`dft_size` from its serialized configuration. `training/tokenizer/eval_transfer.py:175` defaults
missing values to 256. A fresh encoder state round-trip succeeds but changes `filterbank.S`
from **512 to 256**. The prior three-step smoke checkpoint also omits this field.

This is not caught by a head-only strict load: the DFT size is structural, not a weight shape
checked by that load. Restoring a different frontend can change extracted features and rejects
long patches or 512-sample padded input. A 4-second patch at 100 Hz already exceeds 256.

Resolved: serialize the live encoder's DFT size, restore matching collators, and explicitly migrate
identifiable historical snapshots. Do not blindly change every legacy fallback: older 256-point
checkpoints exist. Require an explicit value when provenance cannot resolve it. Test the complete
encoder-plus-head numerical round-trip with a long native-rate patch.

### 2. Per-query masking makes loss weighting depend on query order

`sampling.py:862` independently removes candidate supports for different queries sharing one
support-set ID. A set can consequently contain both supportless and supported queries.
`train.py:530` averages that set's losses, then assigns its entire regime using the first query.

Probe: reordering otherwise identical query rows changes CE **1.70229 -> 2.92224**. It halves
the gradient for one unrelated supported query and doubles it for an unrelated supportless query.
The probe establishes an allowed edge case; its frequency over a full run was not measured.

Resolved: preserve equal-regime weighting, but group within `(support_set_id, regime)` before computing
regime means, or define an equivalent explicit per-query weighting. Test row-order invariance,
mixed-regime sets, and sparse draws. This does not require abandoning the intended independent
per-query masks.

### 3. Partially enrolled candidates use incompatible score references

`model/support/residual_classifier.py:207` combines supported candidates' **log vote probabilities**
(non-positive) with unsupported candidates' **raw cosine/text-temperature scores**, replacing
their metric term with zero. These scores do not share a neutral reference or calibrated scale.

With a deliberately neutral text bridge, zero residuals, and counts `[2, 0, 2]`, two probe queries
produce logits `[-2.6927, 0, -0.0701]` and `[-0.4672, 0, -0.9855]`: the unsupported candidate wins
both without any text advantage. Real text cosines need not be zero and can amplify or reverse
this effect. This is a mathematical problem in the supplied design as well as its implementation.

Resolved: establish a common candidate-score/prior convention for metric evidence and text-only
evidence. Preserve the ordinary neighbor decision when every candidate has supports, but do not
treat an unsupported candidate's arbitrary zero as a neutral probability. Test neutral text,
constant text offsets, variable candidate counts, and GT-present/GT-absent partial enrollment.

Related edge: with one surviving support, support-mean centring makes that vector exactly zero.
The probe gives zero metric gradients to query and support. **A single-support normalized vote
is already constant even without centring**, so simply disabling centring is not a complete fix.
Partial enrollment needs a defined treatment of absolute match evidence versus text evidence.

The initialization contract is exact equality to **centred** neighbors when all candidates are
supported. It is not a guarantee of never underperforming raw neighbors, especially after training.

### 4. Resume does not faithfully preserve the sampling configuration

`train.py:1461`'s trajectory guard omits `p_mask_candidate` and `p_mask_gt`, while resumed draws
use current CLI values. Changing these flags, or omitting previously customized values, silently
changes the curriculum. Several new classifier CLI flags are also not reconciled with the restored
classifier configuration, so a requested setting can be ignored instead of rejected.

Additionally, the default encoder LR scale is selected as 0.05 whenever `--resume` is present,
even for a from-scratch run originally using 1.0; the trajectory guard then rejects the ordinary
resume unless the user repeats the original scale. This is a loud failure rather than corruption.

Resolved: inherit saved configuration when flags are omitted, reject explicit incompatible overrides,
and include support-mask probabilities in the trajectory contract. Keep deliberate continuation
of the step horizon separate from changes to the data or architecture.

### 5. Telemetry conflates GT support with any support

`sampling.py:1054` reports regimes using the first query of each support set.
`sampler/realised_gt_rate` is computed as one minus the zero-support rate, but GT supports can be
masked while other candidates still have supports. Therefore this no longer measures its name.

Resolved: report actual per-query GT support presence, zero-support frequency, per-candidate counts,
and realized mask frequencies; report support-set statistics separately. Monitor accuracy/loss
for fully supported, partially supported GT-present, partially supported GT-absent, and all-zero
conditions. Existing combined summaries can hide the score-reference problem above.

### 6. Native evaluation parameter count excludes the learned head

`sealed_eval.py:87` counts only encoder parameters for every HALO row, including the learned
classifier. Here the encoder has **789,409** parameters and the classifier **1,413,895**:
**2,203,304 total**, excluding the external frozen text tower under the existing counting convention.

Resolved: count the actual method being reported. Parameter-free readouts retain encoder-only
counts; learned-head rows must include the head. Do not silently change baseline accounting.

## Design risks, not proven performance regressions

- **Identity tags dominate text-token magnitudes.** Fresh 128-dimensional projected unit text has
  mean norm 0.576; role, candidate and pair embeddings have mean norms 11.920, 11.254 and 11.293.
  Pre-LayerNorm avoids large activation magnitudes, but does not restore the relative contribution
  of small text content. Use the repo's existing normalized/scaled composition pattern or explicitly
  matched initialization scales; verify semantic sensitivity rather than merely finite gradients.
- **Support tags account for most head parameters.** The 8193-by-128 pair table contains 1,048,704
  parameters, about 74% of the head. Training only visits tags up to the sampled support count;
  evaluation at much larger k can use never-trained tags. Tag reassignment is not the same as tensor
  permutation: a learned tag table does not guarantee invariance to new tag identities. Compare
  deterministic tag permutations and high-k behavior before claiming generalization.
- **Centring remains an unverified improvement.** The design's subject-held-out centring ablation
  has not been run. Exact equality to the centred control verifies mechanics, not that centring
  improves the uncentred encoder. Keep both controls when evaluating the learned head.
- **Residuals can still hurt.** Initializing output layers to zero only protects the starting point;
  unconstrained support/candidate corrections can later override useful metric information. Track
  paired improvements and harms relative to the exact same centered and uncentered controls.

## Checks that passed

- 66 focused tests passed in 2.44 seconds: residual classifier, sampler, loader, sealed-eval units.
- Real-training-roster BF16 forward/backward and optimizer steps were finite in all bounded profiles.
- Both query and support input vectors receive gradients. Recording pooling, encoder, residual
  heads and text bridge receive gradients in the real-data profile.
- Zero-initialized residual output weights intentionally give the attention trunk zero gradient
  on the first step. On the second BF16 step its gradient norm becomes 0.00187: it is not dead.
- With nonzero learned residual weights: support permutation preserving bindings/tags changes
  logits by at most 4.77e-7; adding masked large-valued support padding changes them by 0;
  changing another batch row changes the first row by 0.
- The all-enrolled exact centered-neighbor initialization tests pass. Shared neighbor controls
  were not modified for this audit.

These are bounded checks, not a claim that all data, training, or evaluation paths are bug-free.

## Performance measurements

RTX 4090, BF16, fixed polarized filterbank, 8-second source windows, patches 0.5/1/2/4 seconds,
four independent support sets per step, four queries per set, candidate range 2-32, enrollment
k in 1/2/4/8, semantic-zero probability 0.5, other-candidate masking 0.25, GT masking 0.10,
and multi-device probability 0.5 with at most four devices. Full eight-source training roster;
no per-stream cap. Worker comparisons use identical episode hashes and shapes.

Each worker arm: 96 steps, first 16 excluded. Calibration was shortened to one batch of 64,
and profiling uses a constant LR without warmup. These are throughput probes, not learning curves
or accuracy comparisons. They start from a fresh encoder; a promoted encoder has the same tensor
work but a different learning trajectory.

| Loader processes | Mean step | Loader wait | 35k-loop estimate | Peak allocated VRAM |
|---|---:|---:|---:|---:|
| 2 | 92.32 ms | 59.33 ms | 53.85 min | 2.09 GiB |
| 4 | 63.61 ms | 30.40 ms | 37.11 min | 2.09 GiB |
| 8 | 49.96 ms | 15.91 ms | 29.15 min | 2.09 GiB |

Eight-process median: 35.13 ms; p90: 88.73 ms. Average unique recording windows encoded per step:
216.1; average queries: 15.36 (sparse sets can supply fewer than four). Thus this is not a
16-recording encoder batch: the support windows also get encoded with gradients.

Eight-process component intervals: encoder **including pooling** 17.60 ms, episode assembly
about 1 ms, classifier plus CE 3.35 ms, backward 10.12 ms, clipping about 0.4 ms, optimizer about
1.2 ms. CUDA event intervals can include GPU idle time while Python issues work; they are not pure
kernel occupancy measurements. The old `recording_pool` timing field is an empty event boundary,
not evidence that pooling costs zero. Last-step gradient norms are post-clipping, not loss shares.

One full default validation panel (64 training-subject-held-out support sets) measured **1.96 s**.
At every 2,500 steps this is about 27 seconds over 35k, before any separate initial validation.
Corpus indexing took 1.11 seconds. Full 20-by-256 calibration, text-tower startup, checkpoint
serialization, telemetry and sustained-load variability were not included in the 29.15 minutes.

### Compiler caveat

The short compile trial did **not** successfully compile: Inductor's TF32 legacy getter raises
after the runtime sets the newer precision API, and `suppress_errors=True` falls back to eager.
The trainer nevertheless prints that compilation is enabled. The trial's 65.59 ms mean is not a
valid compiled-vs-eager speed comparison; fallback and compilation overhead contaminate it.

Fix the precision setup against this installed PyTorch version and expose actual compile/fallback
status. Meanwhile use `--no-compile-transformer`; the speed target does not depend on compilation.

### Low-complexity optimization priorities

1. **Eight loader processes:** measured 21% faster overall than four, with unchanged episodes.
2. **Avoid the broken compile attempt:** eager BF16 already meets the projected budget.
3. **Deduplicate corrected voting:** the head computes the base vote, corrected vote, and a zero-
   correction vote separately, repeating normalization, cosine, softmax, scatter and log. Reuse
   invariant terms without detaching the base, preserving exact init output and gradient contracts.
4. **Reduce scalar synchronization:** per-step shape/range checks, Python `bool(cuda_tensor)`
   branches and loss grouping launch many small operations. Move static validity checks to batch
   construction and tensorize the corrected loss reduction. Keep strict input checks at API edges.
5. **Do not start with complicated ragged attention:** support padding averaged 61.6%, attention
   token padding 54.6%, but the entire head plus loss is only 3.35 ms. Grouping episodes by length
   may help larger k later; it is not the current main bottleneck. Increasing batch size solely to
   fill VRAM changes optimization and is unnecessary for this target.

No saving beyond the measured worker change is assumed in the 35-40-minute planning estimate.
Recheck the short profile after correctness fixes before launching.

## Reproducibility

Harness: `training/support_classifier/training_profile.py` (now explicitly exercises the residual head).
Artifacts: `training/support_classifier/evaluations/residual_audit_20260914/`:

- `workers.json`: complete timings, shapes, episode hashes and clipped module gradient norms.
- `eager.json`, `eager.w4.trace.json`, `eager.w4.operators.txt`: one detailed operator trace.
- `compile.json`, `compile.log`: requested-compile trial and explicit fallback traceback.
- `validation_timing.json`: isolated default-panel validation cost.
- `probes.py`, `probes.json`: reproducible correctness probes and source-file SHA256 hashes.

From the `halo/` directory, using `/home/alex/code/HALO/legacy_code/.venv/bin/python`:

```bash
python -m training.support_classifier.training_profile --workers 2 4 8 --steps 96 --warmup 16 --out /tmp/residual-workers.json
python -m training.support_classifier.training_profile --workers 8 --steps 12 --warmup 4 --validation-support-sets 64 --out /tmp/residual-validation.json
python training/support_classifier/evaluations/residual_audit_20260914/probes.py
python -m pytest -q tests/test_residual_classifier.py tests/test_support_classifier_sampling.py tests/test_support_classifier_loader.py tests/test_sealed_eval.py
```

The shared worktree contains concurrent changes. This report describes the source fingerprint
recorded with the probes, not a clean committed release. Model and trainer fixes remain pending.
