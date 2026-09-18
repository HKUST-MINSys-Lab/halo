# Training and evaluation readiness sweep

Scope: bounded code review and smoke tests after repository consolidation, starting at
`57aeb15`. No full training, feature-bank regeneration, or full evaluation was launched.
No subagents were used. This entry records mechanical readiness, not model quality.

## Confirmed issues and repairs

1. HALO neighbor-only checkpoints lacked their training-bank/ConSE zero-support route in
   non-coverage scenario tasks. The scenario dispatch now uses that route without requiring a
   partial-coverage task; regression tests cover both task types.
2. The sealed evaluator computed the same checkpoint's zero-support result but hid it as
   diagnostic-only. It is now a visible HALO readout. Classifier-checkpoint bank diagnostics and
   external baseline reporting are unchanged.
3. Scenario CLI defaults still requested the expensive full grid. Defaults now implement the
   representative budget: eight-second windows and k = 0, 1, 4, 8, 32. Explicit larger grids remain
   possible, and the separate sealed-curve defaults are unchanged.
4. Subject bootstrap with zero replicates crashed when multiple subjects were present. It now
   explicitly reports disabled confidence intervals, not fabricated uncertainty; negative counts
   are rejected. Positive bootstrap counts retain their existing implementation.
5. The environment lacked declared CLIP/timm dependencies for released UniMTS and NormWear.
   The optional `baselines` extra now installs these. OpenAI CLIP 1.0.1 requires the historical
   `pkg_resources` interface, so this extra constrains setuptools below 81. Weights and upstream
   checkouts remain separate requirements, not downloaded implicitly by package installation.
6. The zero-worker trainer bypassed joint device-set planning and per-step augmentation seeds.
   The synchronous and prefetched paths now call the same batch-preparation function. Tests cover
   planned and independently composed batches; worker count no longer changes this curriculum.
7. The design contract incorrectly said internal validation remained single-device. It already
   uses deterministic, independently composed multi-device examples. The contract now states that
   behavior and distinguishes it from the training-only coordinated device-set planner.

## Verification

| check | outcome |
|---|---|
| focused pre-fix model/data/evaluation suite | 250 passed in 28.75 seconds |
| focused regression suite after fixes | 73 passed in 2.55 seconds |
| complete test suite | 965 passed, 1 skipped, 23 warnings in 76.32 seconds |
| HALO neighbors, actual CUDA training | three steps, validation, checkpoint write and reload passed |
| HALO residual classifier, actual CUDA training | three steps, validation, checkpoint write and reload passed |
| residual training with zero loader workers | three steps passed; logged losses match the two-worker smoke |
| released HARNet-5, HARNet-10, LiMU-BERT-X, UniMTS, NormWear | weights loaded and short inference passed on compatible views |
| actual scenario CLI, HARNet-5 / 1-NN | all seven scenarios built; 14 tasks, 20 output rows, zero task failures |

The inference probes used two real windows per view: 4/8/16-second MotionSense, acceleration-only,
20 Hz resampling, and an aligned two-device RealWorld composition. Features were finite and
nonzero. UniMTS and NormWear also exercised their actual native semantic-scoring paths. Both HALO
smoke checkpoints reloaded for all six views; the residual classifier predicted with zero and one
support. Fusion arithmetic was checked with small controlled score matrices, not a newly encoded
full training bank.

Residual smoke training losses were 3.6324, 3.2739, 2.8754 in both loader configurations. Neighbor
losses were 2.2974, 1.6754, 2.0811. Both runs had nonzero encoder, duration-conditioning,
acquisition-text and structured-conditioning gradients; the residual classifier also had nonzero
gradients. Learned recording-pool weights and biases changed in both arms. These three-step
checks do not establish convergence or an expected training time.

Reproduction commands (bounded by 120-150 second shell timeouts):

```bash
uv sync --extra model --extra dev --extra baselines
uv run pytest -q
uv run halo-train --out /tmp/halo-readiness-20260918-neighbors \
  --classifier neighbors --smoke --loader-workers 2 --no-compile-transformer
uv run halo-train --out /tmp/halo-readiness-20260918-residual \
  --classifier residual --smoke --loader-workers 2 --no-compile-transformer
uv run halo-train --out /tmp/halo-readiness-20260918-sync \
  --classifier residual --smoke --loader-workers 0 --no-compile-transformer
uv run halo-scenarios --out /tmp/halo-readiness-scenarios-20260918 \
  --smoke --models harnet5 --k 1 --window-seconds 8 --readouts 1nn
```

Machine-local logs and small probe summaries are under `/tmp/halo-readiness-*`; these are temporary
diagnostics, not publication artifacts. Their conclusions and test counts are persisted here.

## Readiness boundaries

- The implemented residual classifier and neighbor control are mechanically runnable. The
  proposed replacement contextual semantic-voting classifier remains a separate implementation
  task; passing this sweep does not mean that proposed architecture exists.
- Neighbor training requires enrolled ground truth and cannot optimize zero-support episodes.
  It shares the eligible acquisition/device curriculum, not an identical complete episode
  distribution with the semantic classifier. Report that distinction in matched-training claims.
- LiMU-BERT-X correctly rejects acceleration-only input: its released contract requires measured
  acceleration and gyroscope channels. NormWear's flattened features change width with channel
  count; unequal-width enrollment comparisons remain explicitly unsupported.
- No full zero-support reference bank was rebuilt, no compiled training path was benchmarked,
  and no long-run convergence, resume, or all-cells performance claim follows from these smokes.
  Regression tests exercised zero-support runner dispatch using tiny controlled feature banks.
- MobiAct availability is not unlocked by these fixes; its prospective scope remains separate
  from the prepared sealed-six and MM-Fit scenario paths.
- Existing promoted performance tables were not overwritten with smoke metrics. Fresh results
  need their own checkpoint/protocol provenance and complete manifests.

## Current-code follow-up

After the initial sweep, the active branch received two checkpoint-persistence repairs. A resumed
run now preserves the closed-form text-projection calibration record (`p_text_init`), and the
polarization enable flag and energy-gate coefficient are part of the trajectory compatibility
contract. This prevents run metadata from silently disagreeing with the restored encoder. The
repairs are commit `8a30afa`.

The exact current residual recipe was then exercised on CUDA for three fresh optimizer steps and
resumed for a fourth step. Calibration, worker-prefetched episode assembly, mixed-precision
forward/backward, validation, atomic checkpoint writes, optimizer/RNG restoration, and strict
encoder/classifier reload all passed. The restored model contains 840,097 encoder parameters and
1,413,908 classifier parameters. Saved validation metrics were finite, the text calibration record
survived resume, and the checkpoint retained polarization enabled with kappa 0.05.

All monitored trainable paths had nonzero finite gradients: encoder, classifier, duration
embedding, acquisition-text conditioner, structured conditioner, and recording pool. Effective
rank stayed noncollapsed in the three-step probe (50.1--73.4). Early total gradient norms were
19.7--107.0 and therefore strongly clipped by the declared norm-1.0 guard. Historical 3k and 40k
runs show the same persistent clipping regime, so this is not a new implementation regression; it
is a training-design characteristic that must remain visible in telemetry.

The active curriculum audit sampled 256 support sets (969 queries): 46.1% complete, 22.3% partial,
31.6% zero enrollment; among enrolled sets, 80.0% compatible, 9.7% cross-placement, and 10.3%
cross-dataset acquisition. It found no cross-placement dataset contamination. The requested
acquisition mix cannot be fully realized because the eight-source corpus has only four datasets
with valid within-dataset cross-placement pairs and three with valid cross-dataset acquisition-key
pairs; fallback was 19.1%. This is a measured corpus limitation, not a sampler leak.

Two model-selection limitations remain explicit rather than silently changed:

- `best_internal.pt` is selected on enrolled-support dataset-macro F1. Zero-support F1 is logged
  separately but does not participate in selection, even though the shared model is reported in
  both regimes.
- internal validation uses deterministic independent multi-device composition, but not the
  coordinated query/support device-set challenge used during training. The sealed scenario panel
  measures that challenge only after checkpoint selection.

Neither limitation blocks optimization or checkpoint restoration. Before the next publication
run, the checkpoint-selection contract should be frozen explicitly: retain the enrolled-primary
policy and name a zero-support companion checkpoint, or predeclare a joint criterion plus a fixed
coordinated device-set development panel. Sealed results must never make that choice.
