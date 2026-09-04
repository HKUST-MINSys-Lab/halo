# Representation and signal baselines

The application study compares movement representations, not native HAR classifiers. Every method
feeds the same Task-1 matcher, Task-2 comparison pipeline, and Task-3 motif evaluator where its
temporal output permits that use.

## Required non-learned controls

| control | purpose |
|---|---|
| raw IMU subsequence DTW | establishes whether a pretrained representation beats direct waveform matching |
| magnitude/gravity-aligned DTW | tests simple orientation robustness without a learned encoder |
| physical-feature sequence | compares duration, intensity, frequency, smoothness, and stability directly |
| raw or feature matrix profile | established floor for recurrent subsequence discovery |

A pretrained encoder is not useful merely because it beats another encoder. It must improve on these
simple, interpretable application methods or provide materially better robustness or efficiency.

## Primary released-checkpoint roster

The IMWUT comparison branch uses four released-checkpoint external encoders. This is the smallest
set that covers the main representation families used by the classification/adaptation benchmark;
the application-task study on `main` may omit ImageBind when its generic multimodal control is not
informative for a task.

| encoder | publication | family represented | why it is retained | main limitation |
|---|---|---|---|---|
| **HARNet / ssl-wearables** | npj Digital Medicine, 2024 | large-scale convolutional self-supervision on real wrist accelerometry | strongest low-cost control for whether scale and a conventional temporal CNN are already sufficient | accelerometer only; fixed 5 s, 30 Hz contract |
| **UniMTS** | NeurIPS, 2024 | synthetic motion, skeleton-graph encoding, rotation augmentation, and sensor-text alignment | recent control for explicit placement/orientation generalization and language-aligned motion representations | accelerometer only; body placement must map to its 22-joint skeleton |
| **NormWear** | ACM TCH, 2025 | channel-independent time-frequency wearable foundation model | recent and most direct external comparison to HALO's frequency-domain and variable-channel design | approximately an order of magnitude slower than the other primary baselines |
| **ImageBind** | CVPR, 2023 | generic multimodal text-aligned IMU representation | tests whether broad multimodal alignment transfers without wearable-specific adaptation | IMU tower was trained on head-mounted Ego4D data; large checkpoint |
| **HALO** | project model | physical-time representation | fixed and continuous physical-time arms under matched within-HALO ablations | project model, not an external baseline |

ImageBind remains optional for routine application-task runs because it requires a roughly 4.5 GB
checkpoint and overlaps with UniMTS's language-alignment question. It is retained in the final
classification/adaptation comparison so that the released-checkpoint roster matches the design.

CrossHAR and LiMU-BERT remain excluded because the repository's usable backbones were pretrained by
us rather than released by their authors. That would reintroduce choices about corpus, schedule,
augmentation, and checkpoint selection.

## Measured evaluation cost

The following measurements come from the same existing adaptation manifest
(`1bd89d35f5aed197fdce73db87c8442a949249337bbe703a870ae7c2253a5469`) on the project RTX 4090.
They include the shared readout/evaluation work, so they are a matched operational screen rather
than pure encoder microbenchmarks. The manifest and code path are identical across rows.

| encoder | elapsed time | setup time | peak GPU memory | time relative to HARNet |
|---|---:|---:|---:|---:|
| **HARNet** | 31.45 s | 0.13 s | 0.26 GiB | 1.0x |
| **UniMTS** | 50.23 s | 2.19 s | 4.98 GiB | 1.6x |
| **ImageBind (optional)** | 47.78 s | 7.97 s | 5.10 GiB | 1.5x |
| **NormWear** | 484.65 s | 2.34 s | 8.82 GiB | 15.4x |

The compact measurements above were promoted in Git at commit `87d3c04`; the large per-model JSON
outputs remain intentionally ignored. Treat this as an operational screening result, not a
reproducibility-grade benchmark. New publication results must retain the compact manifest,
checkpoint hashes, and per-model timing summary in a tracked result artifact.

Use two execution tiers:

- **Rapid development:** HALO, raw/physical controls, HARNet, and UniMTS.
- **Final frozen benchmark:** add NormWear, encode each recording once, and cache timestamped
  embeddings for reuse by Tasks 1-3.

The shared application adapter is implemented in
`applications/motion_monitoring/baseline_encoder.py`. It slides each released model at a one-second
stride while preserving its native receptive field: HARNet 5 s, UniMTS 10 s, and NormWear 6 s.
The application task heads receive only timestamped unit-normalized representations; native HAR
classifiers and label text do not enter Tasks 1-3. Real-data smokes cover all three task heads and
MoniPar, ALAMEDA, and bilateral COPS streams. This establishes input compatibility only, not model
quality.

The primary study compares frozen representations. Do not fine-tune every foundation model merely
to make the table larger. If a task-specific learned arm is justified, fit the same small head to
each frozen representation. Any end-to-end encoder fine-tuning is a separate experiment with its
own compute and training-data disclosure. For that optional arm, HARNet and the UniMTS sensor tower
are the practical external candidates; keep NormWear frozen unless a result shows that its roughly
136M-parameter backbone warrants the additional cost.

## What is shared

- source recordings and real event boundaries;
- subject/session splits and application manifests;
- reference counts and target-absent intervals;
- matching, alignment, event-merging, threshold-selection, and motif-scoring code;
- task metrics and subject-level uncertainty; and
- physical-time windows and strides whenever an encoder lacks a native temporal sequence.

Each adapter preserves its checkpoint's documented units, rate, channels, normalization, and shape.
An encoder is never modified to accept information absent from its released architecture.

## What the comparison means

The primary comparison asks which available frozen representation is the best component for the
application. Different upstream datasets and model sizes are therefore documented rather than
artificially matched. A result does **not** prove that one architecture is intrinsically superior.

HALO architecture claims require within-HALO controls using the same data and optimization, such as
fixed filterbank versus continuous kernels, acquisition conditioning versus neutral metadata, or
pooled versus temporal patch output.

## Primary sources

- [HARNet paper](https://doi.org/10.1038/s41746-024-01062-3) and
  [official repository](https://github.com/OxWearables/ssl-wearables)
- [UniMTS paper](https://doi.org/10.48550/arxiv.2410.19818) and
  [official repository](https://github.com/xiyuanzh/UniMTS)
- [NormWear paper](https://doi.org/10.1145/3803808) and
  [official repository](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear)
- [ImageBind paper](https://doi.org/10.1109/CVPR52729.2023.01457) and
  [official repository](https://github.com/facebookresearch/ImageBind) (optional appendix control)

The detailed treatment contract is in
[`BASELINE_FAIRNESS_POLICY.md`](BASELINE_FAIRNESS_POLICY.md).
