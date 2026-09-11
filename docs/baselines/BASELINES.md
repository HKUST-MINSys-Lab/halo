# Retained baseline encoders

The primary comparison is deliberately small and uses only author-released checkpoints.

| encoder | representation family | required input | role in comparison |
|---|---|---|---|
| HARNet / SSL-Wearables | large-scale wrist accelerometry self-supervision | accelerometer, published resampling and crop contract | low-cost temporal HAR control |
| UniMTS | synthetic-motion and body-configuration foundation model | accelerometer mapped to its published body configuration | recent heterogeneous-motion control |
| NormWear | channel-independent wearable time-frequency model | channels and preprocessing required by its checkpoint | time-frequency control |
| HALO | physical-time IMU representation | heterogeneous masked accelerometer/gyroscope streams | project model |

Every adapter must record its checkpoint source and published preprocessing. It may resample or pad
only as required by that model's documented contract. It must not use test labels, candidate text,
or support labels while producing an embedding.

LiMU-BERT and CrossHAR are excluded because locally usable checkpoints were trained in this project.
ImageBind is retained only for diagnostic work. These exclusions reduce provenance ambiguity; their
historical results are not part of the live paper comparison.
