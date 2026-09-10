# HALO representation pretraining

This directory contains HALO's label-free encoder pretraining. The design of record is the
multi-horizon, physically grounded future-JEPA objective described in
[`docs/design/JEPA_PRETRAINING_OBJECTIVE.md`](../../docs/design/JEPA_PRETRAINING_OBJECTIVE.md).
Activity labels are used only by development probes; they do not enter the training objective.

## Default objective

For the default fixed-filterbank arm, every native-rate six-second source window becomes aligned
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
| source context | up to six seconds |
| patch grids | 0.5, 1.0, 1.5 seconds, jointly encoded |
| predictor | d=128, 2 decoder layers, 4 heads |
| optimizer | AdamW |
| LR / warmup / weight decay | 4.24e-4 / 500 steps / 0.0707 |
| gradient clip | 1.0 |
| precision on CUDA | FP16 autocast with dynamic loss scaling; FP32 master weights and reductions |
| default batch / steps | 512 / 15,000 |

Multi-span defaults are derived from the same 12,288 transformer-token budget. With spans 0.5,
1.0, and 1.5 seconds, this is batch 128 and 60,000 updates, preserving the default arm's number of
sampled windows. `--multispan-durations` is serialized in the checkpoint.

Run-specific values in `run_config.json` are authoritative. The trainer rejects incompatible
future-objective combinations instead of silently falling back to a different experiment.

## Launch and smoke tests

Use the project environment for commands that import Torch, SciPy, pandas, or h5py:

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python

$PY -m training.tokenizer.pretrain --smoke --steps 2 --device cpu \
  --out /tmp/halo_future_jepa_smoke --force

$PY -m training.tokenizer.pretrain --device cuda \
  --calibrate-objectives-at 1000 \
  --objective-calibration-mode apply \
  --out training/tokenizer/outputs/<run>

$PY -m training.tokenizer.pretrain --device cuda --frontend continuous \
  --out training/tokenizer/outputs/<continuous-run>

$PY -m training.tokenizer.pretrain --device cuda --frontend multispan \
  --multispan-durations 0.5 1.0 1.5 \
  --out training/tokenizer/outputs/<multispan-run>
```

Production prerequisites include current grid-quality caches and any required sensor-bias artifact.
Resume from `last.pt` with the same trajectory-defining configuration. Checkpoints restore the
student, EMA teacher, frozen physical-target analyzer, active pretraining heads, optimizer,
scheduler, AMP scaler, RNG states, corpus fingerprint, and the already-resolved objective
coefficients.

## Monitoring

The trainer writes structured telemetry every 50 updates. Refresh the CPU-only health report and
dashboard without touching the training GPU:

```bash
$PY -m training.tokenizer.monitor_training \
  --run-dir training/tokenizer/outputs/<run> --render
```

The future-JEPA telemetry includes:

- weighted and raw future, physical, and collapse losses;
- target counts and losses by horizon and resolution;
- true-target versus shuffled-target similarity margin;
- physical-decoder improvement over a zero predictor;
- context, target, ineligible, and leakage rates;
- encoder, predictor, decoder, and objective-specific gradient norms and cosines;
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
