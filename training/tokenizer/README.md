# HALO representation pretraining

This directory contains HALO's label-free encoder pretraining. The design of record is the
multi-horizon, physically grounded future-JEPA objective described in
[`docs/design/JEPA_PRETRAINING_OBJECTIVE.md`](../../docs/design/JEPA_PRETRAINING_OBJECTIVE.md).
Activity labels are used only by development probes; they do not enter the training objective.

## Default objective

For the default fixed-filterbank arm, every native-rate eight-second source window becomes aligned
0.5, 1.0, and 1.5 second patch grids. The constrained-learnable filterbank uses the same grid; the
continuous-kernel arm uses its native one-second token grid. A random physical-time boundary leaves
roughly 40-70% of the window visible to the student. Targets begin strictly after that boundary, so
a token interval that crosses the boundary belongs to neither side.

The multi-span continuous arm is also supported. It emits separate 0.5, 1.0, and 1.5 second
physical-time grids by default. For both continuous arms, the frontend itself receives only the
raw prefix available to the student; masking only the later transformer rows would leak future
samples through kernels evaluated on the complete recording.

The live recipe has three terms:

| term | behavior | initial weight |
|---|---|---:|
| `future` | predict normalized EMA-teacher states at sampled short, medium, and long future horizons | 1.0 |
| `physical` | decode each predicted future state to standardized filterbank energies and signed DC | 5.0 |
| `collapse` | enforce variance and reduce covariance directly on visible student patch/sensor states | 0.01 |

The physical target excludes the filterbank's unstandardized amplitude scalar. It uses continuous
confidence weights for bands limited by Nyquist frequency or patch frequency resolution. This avoids
making an unobservable feature a hard target without introducing a dataset-specific cutoff.

After warmup, the trainer measures encoder-gradient norms on representative batches, solves one
fixed scalarization targeting 70% future / 20% physical / 10% collapse influence, applies it once,
and freezes the coefficients. Transfer validation, not gradient equality, selects the checkpoint.

## Student and teacher

- The **student** receives only patches ending at or before the sampled boundary. Future signal
  values are zeroed before encoding and future rows are excluded as attention keys and values.
- The **EMA teacher** receives the complete clean window under `torch.no_grad()`.
- The target is the normalized mean of the teacher's top two transformer-layer states.
- A narrow predictor receives visible student states plus target time, horizon, duration,
  resolution, sensor descriptor, and a learned query role. It never receives target signal values.
- The physical decoder operates on the predicted future state, so it cannot bypass future
  prediction by reading the teacher target.

The fixed-filterbank arm shares its parameter-free analysis between student and teacher. Learnable
and continuous frontends run the EMA teacher's own frontend parameters. All arms retain separate
student/teacher projection and transformer weights. The EMA decay may be fixed or follow a cosine
schedule; both the instantaneous decay and its half-life in examples are logged.

## Scope and retained control

The future objective currently requires:

- sensor-granularity tokens (one token per accelerometer or gyroscope xyz triad);
- a frontend that emits tokens with truthful physical intervals and validity masks; and
- duration/resolution conditioning whenever more than one token scale is present.

The fixed, constrained-learnable, single-span continuous-kernel, and multi-span continuous-kernel
frontends are supported. Physical
reconstruction always uses a separate frozen filterbank analyzer whose calibrated state is stored
in the checkpoint. It never follows the student or EMA update, so a learnable frontend cannot
co-adapt with its target. For multi-span training, that analyzer is calibrated across every
declared target duration rather than reusing one-second statistics.

The previous bidirectional masked-JEPA plus VICReg recipe remains only as an explicit historical
control:

```bash
python -m training.tokenizer.pretrain --jepa-mode masked --no-multiresolution ...
```

Its predictor, VICReg projector, two-view data path, and legacy options are instantiated only in
that mode. They are not active modules in a default checkpoint.

## Data and sampling

The sampler is label-free and hierarchical:

```text
P(dataset) proportional to n_dataset^0.25, capped at 25%
P(subject | dataset) proportional to n_subject^0.5
P(window | subject) uniform
```

Splits are subject-disjoint. Native acquisition rate is carried separately from stored array rate,
so an upsampled stream is not treated as containing spectral information above its real Nyquist
limit. Final partial source windows and final partial patches remain represented with honest masks
and durations. A source window too short to provide both context and a future target is marked
ineligible rather than given a fabricated target.

The exact dataset roster, corpus fingerprint, source provenance, and resolved objective weights are
serialized with each run. Corpus sizes change as opt-in pretraining sources are added, so use
`run_config.json` and the generated corpus report for run-specific counts rather than copying a
global row count into result claims.

## Model defaults

| setting | value |
|---|---:|
| encoder | d=256, 3 dual-branch transformer layers, 8 heads |
| frontend | fixed physical filterbank |
| token unit | one token per sensor modality triad |
| source context | up to eight seconds |
| patch grids | 0.5, 1.0, 1.5 seconds, jointly encoded |
| predictor | d=128, 2 decoder layers, 4 heads |
| optimizer | AdamW |
| LR / warmup / weight decay | derived from the resolved batch; recorded in `run_config.json` |
| gradient clip | 1.0 |
| precision on CUDA | BF16 neural autocast; FP32 master weights, physical analysis and loss statistics |
| default fixed-filterbank batch / steps | 512 / 15,000 |

Multi-span defaults use a separate 61,440 transformer-token ceiling and a reference batch of 384.
With an eight-second context and spans 0.5, 1.0, and 1.5 seconds at four frames per span, the resolved
schedule is batch 384 and 20,000 updates, preserving the fixed arm's 7.68 million sampled windows.
The fixed-filterbank token ceiling is 16,384. Durations, frame density, token budget, and resolved
schedule are serialized in the checkpoint. These ceilings describe batch times temporal tokens;
sensor count and frontend workspace also affect memory usage.

FP16 remains an explicit `--amp-dtype fp16` option with dynamic loss scaling; BF16 uses no scaler
and aborts on non-finite gradients before the optimizer/EMA update. Compilation is opt-in because
it was slower for the measured dynamic workloads. Neither optimization reduces encoder capacity.

Run-specific values in `run_config.json` are authoritative. The trainer rejects incompatible
future-objective combinations instead of silently falling back to a different experiment.

## Launch and smoke tests

Use the project environment for commands that import Torch, SciPy, pandas, or h5py:

The full-roster commands below require **all** configured sources to have usable native grids.
`--smoke` reduces model/run size, not this requirement. For an explicitly limited local pilot,
replace `--corpus label_free` with `--datasets nhanes synthetic_imu`; do not call that a full-corpus
run. See the dated verification below for the current local data state.

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python

$PY -m training.tokenizer.pretrain --smoke --steps 2 --device cpu --corpus label_free \
  --out /tmp/halo_future_jepa_smoke --force

$PY -m training.tokenizer.objective_health --frontend fixed --corpus label_free \
  --out /tmp/halo_fixed_objective_health.json
$PY -m training.tokenizer.objective_health --frontend multispan --corpus label_free \
  --out /tmp/halo_multispan_objective_health.json

$PY -m training.tokenizer.pretrain --device cuda --corpus label_free \
  --objective-calibration-mode apply \
  --out training/tokenizer/outputs/<run>

$PY -m training.tokenizer.pretrain --device cuda --corpus label_free --frontend multispan \
  --multispan-durations 0.5 1.0 1.5 \
  --out training/tokenizer/outputs/<multispan-run>
```

Production prerequisites include all requested native grids, the eight-second source-window
contract, current grid-quality caches, and the labelled development grids used for selection.
Sensor-bias augmentation is disabled by default and requires no bias artifact in this recipe.
Resume from `last.pt` with the same trajectory-defining configuration. Checkpoints restore the
student, EMA teacher, frozen physical-target analyzer, active pretraining heads, optimizer,
scheduler, AMP scaler, RNG states, corpus fingerprint, and the already-resolved objective
coefficients.

## Optimization verification (2026-09-11)

The vectorized future planner was checked against an independent scalar specification on ragged,
permuted and missing-resolution grids. It preserves uniform sampling over unique physical centers,
not over duplicated resolution tokens. Random numbers are consumed in a different order from the
old per-recording loop: sampling rules are preserved, but seeds do not reproduce the old trajectory
across this source change. Resume provenance checks intentionally reject changed source code.

This sweep fixed an absent-resolution selection erasing a previously selected target at index zero,
handled empty/invalid planner inputs, and removed a quadratic feasibility temporary. The continuous
student's learned projection now follows the same mixed-precision policy as the teacher while
signal analysis remains FP32. CUDA memory fields now report binary GiB; older fields named `gib`
actually contained decimal GB and need multiplying by `1e9 / 1024**3` when comparing histories.

Verification: 221 focused tests passed. Both full-size encoders also completed 52 CUDA BF16 updates,
objective calibration, reduced development probes, checkpoint reconstruction, and resume to step
54. Encoder, frontend projection, future predictor and physical decoder gradients were finite and
nonzero; reported future-context leakage and skipped updates were zero. Frozen physical-target
analyzer weights were identical before and after resume.

| bounded GPU probe | fixed multiresolution | continuous multispan |
|---|---:|---:|
| batch size | 512 | 384 |
| measured windows/s (step-50 telemetry interval) | 8,736 | 2,988 |
| peak allocated memory (converted to GiB) | 3.77 | 8.17 |
| encoder gradient norm at step 50 | 0.841 | 0.407 |
| future-predictor gradient norm at step 50 | 0.400 | 0.450 |
| physical-decoder gradient norm at step 50 | 0.180 | 0.122 |

These are **mechanical smoke measurements, not model-quality results or full-corpus timing promises**.
They used only NHANES and synthetic IMU (60/80 Hz), three normalization-calibration batches, warmup
10, objective calibration at step 20 over five batches, and a reduced development cap of 128 windows
per stream. The full 15k/20k LR/EMA schedules were retained with `--stop-after`; the other overrides
are diagnostic only. Raw logs/checkpoints are local at
`/tmp/halo_jepa_optimization_audit_20260911/` and are temporary, not publication artifacts.
At these measured rates, 7.68 million windows alone extrapolate to about 15/43 minutes, excluding
full-corpus loading, full calibration, validation and checkpoint overhead. Higher-rate and
additional-placement data must be profiled once materialized; neither arm has a verified
30-minute full-corpus guarantee.

**Local production readiness: blocked on corpus materialization, not a known remaining model-code
failure.** The configured label-free roster is NHANES, Nymeria Xsens, Nymeria Aria, Ego-Exo4D and
synthetic IMU. As of this audit, the three Nymeria/Ego-Exo4D sources have no usable grids. NHANES has
eight subjects and 37,512 retained windows; synthetic IMU has one actor and 31,073 retained windows
across head, pelvis and sternum. Its other five grid directories contain zero windows. Total usable
pilot data: **68,585 windows**, not the planned full corpus. The loader correctly refuses the default
full-roster launch rather than silently substituting this subset. Complete/verify the requested
sources, then run a brief representative high-rate probe before starting production training.

## Monitoring

The trainer writes structured telemetry every 50 updates. Refresh the CPU-only health report and
dashboard without touching the training GPU:

```bash
$PY -m training.tokenizer.monitor_training \
  --run-dir training/tokenizer/outputs/<run> --render --watch 60
```

The future-JEPA telemetry includes:

- weighted and raw future, physical, and collapse losses;
- target counts and losses by horizon and resolution;
- true-target versus shuffled-target similarity margin;
- physical-decoder improvement over a zero predictor;
- context, target, ineligible, and leakage rates;
- encoder, frontend, predictor, decoder, and objective-specific gradient norms and cosines;
- multi-span observability, dead-kernel fraction, response spread, and duration-gate state;
- visible-state and teacher representation spread/effective rank;
- EMA student-teacher distance and half-life;
- AMP skips, clipping, throughput, VRAM, source balance, and input validity; and
- fixed subject-disjoint transfer probes used for checkpoint selection.

Detailed per-configuration or duplicate-vector analyses are offline acceptance checks, not extra
per-step work.

## Code map

- `pretrain.py`: configuration, model assembly, train/validation loop, telemetry, checkpoints.
- `future_jepa.py`: future-target planner, predictor, physical targets, balanced losses, calibration.
- `pretrain_data.py`: corpus index, sampler, dataset, and aligned multi-resolution collate.
- `eval_transfer.py`: checkpoint reconstruction and fixed downstream transfer probes.
- `monitor_training.py` / `plot_training.py`: CPU-only live health reporting.
- `losses_repr.py`: historical masked-JEPA/VICReg control and shared diagnostics.
