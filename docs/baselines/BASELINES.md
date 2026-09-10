# Baseline Encoders

The motion-monitoring study compares timestamped representations, not each model's native HAR
classifier. Every retained encoder feeds the same Task 1 matcher, Task 2 change model, and Task 3
recurrent-motion model. Frozen controls use the authors' released checkpoints. HALO may be trained
end to end with the task head in its own arm.

## Retained released-checkpoint roster

| Encoder | Family | Why retained | Input constraint |
|---|---|---|---|
| HARNet / ssl-wearables | large-scale wrist accelerometry SSL | low-cost temporal-CNN control | accelerometer, 30 Hz, 5 s receptive field |
| UniMTS | synthetic motion and body-configuration encoding | configuration-aware recent foundation-model control | accelerometer mapped to its body model |
| NormWear | channel-independent time-frequency wearable model | closest external time-frequency control | substantially slower inference |
| HALO | physical-time representation | project model: base fixed, multiresolution fixed, and multispan continuous variants | accepts heterogeneous IMU layouts |

The application adapter preserves each baseline's published input contract, derives timestamped
embeddings at a common evaluation stride, and passes no label information to the encoder.

## Excluded models

LiMU-BERT, CrossHAR, ImageBind, and internally trained HALO variants are not active comparison
arms. Their retained or reproducible history belongs to
`archive/imwut-comparison-pre-cleanup-20260910`, not this branch. Excluding them keeps the paper
limited to published checkpoint provenance and a small, interpretable representation roster.

See [BASELINE_FAIRNESS_POLICY.md](BASELINE_FAIRNESS_POLICY.md) for the common application-task
protocol and source-specific constraints.
