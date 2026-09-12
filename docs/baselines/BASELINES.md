# Retained baseline encoders

The primary comparison is deliberately small and uses only author-released checkpoints.

| encoder | representation family | required input | role in comparison |
|---|---|---|---|
| HARNet / SSL-Wearables | large-scale wrist accelerometry self-supervision | accelerometer, published resampling and crop contract | low-cost temporal HAR control |
| LiMU-BERT-X | large-scale phone IMU masked-reconstruction pretraining | 20 Hz, one-second, six-axis IMU clips | compact real-world deployment control |
| UniMTS | synthetic-motion and body-configuration foundation model | accelerometer mapped to its published body configuration | recent heterogeneous-motion control |
| NormWear | channel-independent wearable time-frequency model | channels and preprocessing required by its checkpoint | time-frequency control |
| HALO | physical-time IMU representation | heterogeneous masked accelerometer/gyroscope streams | project model |

Every adapter must record its checkpoint source and published preprocessing. It may resample or pad
only as required by that model's documented contract. It must not use test labels, candidate text,
or support labels while producing an embedding.

The original LiMU-BERT and CrossHAR are excluded because their locally usable checkpoints were
trained in this project. LiMU-BERT-X is included separately because its authors released the
checkpoint trained on their large-scale delivery corpus. Its transformer consumes 20 samples at
20 Hz. For a longer evaluation window, the adapter encodes every contiguous one-second clip and
duration-weights the clip embeddings; it does not crop the window to one second. LiMU-BERT-X has no
released open-label classifier, so `k=0` is `N/A` and only the common enrollment readouts are
reported for `k>0`. ImageBind remains diagnostic only.
