# Retired JEPA representation evaluation

> **Historical experiment record.** Do not use these commands for a current HALO run.

This document defines the narrow experiment that asks whether the two 2026-09-12 label-free JEPA
checkpoints produce useful support-classification representations. It is an experiment harness,
not a model-design document.

The commands below reproduce revision-2 checkpoints whose physical durations were
`0.5/1.0/1.5` seconds. They are historical controls, not templates for the revision-3 frontend in
[CONTINUOUS_KERNEL_FRONTEND.md](CONTINUOUS_KERNEL_FRONTEND.md), which uses `0.5/1.0/2.0` seconds.

## Arms

Each retained frontend has four arms: random frozen, JEPA frozen, random adapted, and JEPA
adapted.  Frozen arms use the deterministic duration-weighted recording pool already present in
the JEPA encoder. Adapted arms train only the encoder through the parameter-free differentiable
neighbour loss for a fixed 5,000 steps. The recording pool is then learnable and is counted as part
of the adapted encoder.

The random-adapted arm is mandatory. It controls for improvement caused by the supervised
neighbour objective rather than label-free pretraining.

## Frozen readout

At `k > 0`, the sealed evaluator applies 1-NN, differentiable neighbours, prototype, and ridge to
the same execution-disjoint support/query manifest. The differentiable-neighbour readout is a
softmax over temperature-scaled query/support cosine similarities, with weights summed by support
label. It has no trainable parameters. `k=0` is not applicable to any support-only readout.

The evaluator's `--embedding-diagnostics` switch writes visual evidence without affecting scores.
It is deliberately opt-in because sealed data must only be read after the experiment choices are
frozen.

## Adaptation schedule

Use 5,000 optimizer steps, 500 warmup steps, the fixed current episode curriculum, BF16, and the
same `--data-seed` for a JEPA/random pair. Warm-started JEPA runs use the trainer's 0.05 encoder
learning-rate multiplier. Internal subject-held-out validation is diagnostic only; evaluate the
predeclared final step-5,000 checkpoint on sealed data.

## Commands

The following are templates, not an instruction to consume sealed data now.

```bash
# Encoder-only adaptation, fixed multiresolution JEPA arm.
python -m training.support_classifier.train \
  --phase-a training/tokenizer/outputs/jepa_fixed_multires_20260912_full/last.pt \
  --frontend fixed --resolutions 0.5 1.0 1.5 --classifier neighbors \
  --steps 5000 --warmup-steps 500 --out training/support_classifier/outputs/jepa_fixed_neighbors_5k

# Frozen sealed evaluation, with optional representation figures.
python -m training.support_classifier.sealed_eval \
  --models halo --halo-checkpoint training/tokenizer/outputs/jepa_fixed_multires_20260912_full/last.pt \
  --out training/support_classifier/evaluations/jepa_fixed_frozen --embedding-diagnostics
```

Use the multispan checkpoint with `--frontend multispan --spans 0.5 1.0 1.5` for the parallel arm.
All reportable runs must follow [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).
