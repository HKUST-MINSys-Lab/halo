# Retained baseline encoders

The primary comparison is deliberately small and uses only author-released checkpoints.

| encoder | representation family | required input | role in comparison |
|---|---|---|---|
| HARNet-5 and HARNet-10 / SSL-Wearables | large-scale wrist accelerometry self-supervision | accelerometer at 30 Hz; separate 5 s and 10 s released contracts | low-cost temporal HAR controls |
| LiMU-BERT-X | large-scale phone IMU masked-reconstruction pretraining | 10 Hz, two-second, six-axis IMU clips | compact real-world deployment control |
| UniMTS | synthetic-motion and body-configuration foundation model | accelerometer mapped to its published body configuration | recent heterogeneous-motion control |
| NormWear | channel-independent wearable time-frequency model | channels and preprocessing required by its checkpoint | time-frequency control |
| HALO | physical-time IMU representation | heterogeneous masked accelerometer/gyroscope streams | project model |

### Audited representation and metadata surfaces

| encoder | frozen representation used for enrollment | acquisition/text inputs used |
|---|---|---|
| HARNet-5 / HARNet-10 | released `feature_extractor` output immediately consumed by `EvaClassifier` (`T=1` is squeezed, not pooled) | no metadata or text; accelerometer is linearly resampled to 30 Hz and crop/wrap-padded to the released 150/300-sample contract |
| LiMU-BERT-X | released transformer hidden sequence, mean-pooled within each 20-sample clip and duration-weighted across clips | no metadata or text; six-axis IMU is resampled to the published 10 Hz clock |
| UniMTS | released accelerometer `acc_st_gcn` sensor embedding | stream placement is mapped to the released SMPL joint; source-dictionary candidate text is used only by the native zero-shot text path; input is 20 Hz and 200 frames |
| NormWear | released sensor-backbone patch tokens, mean-pooled over patches and flattened over measured channels | no metadata or text for enrollment; the separate native zero-shot path uses the released fixed activity query and candidate-label text; input is resampled to 65 Hz; tails shorter than the backbone's 11-sample minimum repeat their final measured sample |

These choices are serialized in each result's `model_artifacts.json`; a cache key therefore changes
when the layer, pooling policy, sampling rate, window policy, label dictionary, or metadata input
changes.

Every adapter must record its checkpoint source and published preprocessing. It may resample or pad
only as required by that model's documented contract. It must not use test labels, candidate text,
or support labels while producing an embedding.

The original LiMU-BERT and CrossHAR are excluded because their locally usable checkpoints were
trained in this project. LiMU-BERT-X is included separately because its authors released the
checkpoint trained on their large-scale delivery corpus. Its transformer consumes 20 samples at
10 Hz. For a longer evaluation window, the adapter encodes every contiguous two-second clip and
duration-weights the clip embeddings; it does not crop the window to two seconds. The released
downstream classifier applies a GRU to the hidden sequence, so mean pooling is explicitly a HALO
common-protocol representation readout rather than a claimed native LiMU-BERT-X layer. LiMU-BERT-X
has no released open-label classifier, so `k=0` uses the disclosed training-bank 1-NN plus ConSE
bridge; it is not described as a native zero-shot result. ImageBind remains diagnostic only.

HARNet exports the released `feature_extractor` output immediately before `EvaClassifier`; the
published fixed input makes its temporal dimension one, so no evaluation-only temporal pooling is
introduced. HARNet and LiMU-BERT-X consume no text or acquisition metadata. UniMTS consumes sensor
placement by mapping each stream to the released SMPL joint and uses its text tower only for native
zero-shot scoring. NormWear's native path uses its fixed activity query and candidate label text;
its common enrollment representation is pooled from sensor-backbone patch tokens without text.
Because that released downstream representation flattens measured channels, enrolled comparisons
whose query and support have different channel/device counts are reported as unsupported rather
than introducing an evaluation-only projection. Its native fixed-width zero-shot path remains
valid in those cells because it does not consume support features.
