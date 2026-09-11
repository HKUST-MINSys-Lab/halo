"""Label-free HALO representation pretraining.

The default objective is multi-horizon future JEPA: a past-only student predicts normalized
contextual states from a full-view EMA teacher, the predicted states decode back to fixed physical
measurements, and variance/covariance regularization acts directly on student patch states. The
older bidirectional masked JEPA plus VICReg recipe remains an explicit ``--jepa-mode masked``
control.

Labels are used only by the validation probes. The corpus sampler is hierarchical and label-free:
capped dataset-temperature mass, subject-temperature mass, then windows.

Other invariants:
  * Acquisition conditioning is carried by sensor text descriptors and explicit duration/
    resolution metadata. Text augmentation is an opt-in ablation, not a default objective.
  * Gravity alignment and SO(3) augmentation are disabled by default; signed DC preserves
    gravity-frame posture information.
  * The encoder's inner filterbank norm is CALIBRATED before training.  (M3 lesson)

Model selection: subject-disjoint val kNN recall macro-averaged over label/stream cells, not loss.
Checkpoints carry config + label map + filterbank norm stats + provenance.

Run (CPU smoke):   .../python -m training.tokenizer.pretrain --steps 20 --smoke
Run (real, GPU):   .../python -m training.tokenizer.pretrain --device cuda
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.scripts.augmentations import AugmentationConfig
from data.scripts.labels.canonical_labels import NON_SEMANTIC_LABELS
from data.pretraining.corpus_plan import PRETRAIN_WINDOW_SECONDS
from model.tokenizer.encoder import SetTokenizerEncoder
from model.tokenizer.continuous_kernel import ContinuousKernelTokenizer
from model.tokenizer.filterbank import PhysicalFilterbankTokenizer
from model.tokenizer.multispan_kernel import MS_FRAMES_PER_SPAN, multispan_frame_count
from model.tokenizer.sensor_tokens import descriptor_retrieval_loss
from training.tokenizer.losses_repr import (
    MASK_RATIO_TIME,
    make_mask_plan,
    make_per_resolution_mask_plan,
    make_sensor_mask_plan,
    masked_ema_latent_loss,
    pair_contrast,
    phase_a_loss,
    vicreg,
    fold_analysis_to_sensors,
    masked_analysis_reconstruction_loss,
    VICRegOutput,
)
from training.tokenizer.eval_transfer import (
    PHASE_A_SELECTION_DATASETS,
    assert_selection_roster_is_untrained,
    development_transfer_score,
)
from training.tokenizer.future_jepa import (
    DEFAULT_HORIZON_BINS_SECONDS,
    FuturePredictor,
    balanced_future_latent_loss,
    balanced_physical_loss,
    combine_future_losses,
    pack_selected_interval_patches,
    fixed_filterbank_physical_targets,
    gather_token_rows,
    make_future_target_plan,
    normalized_teacher_target,
    patch_variance_covariance,
    recommend_fixed_objective_weights,
    truncate_patch_lengths_at_time,
)
from training.tokenizer.pretrain_data import (
    DFT_SIZE,
    LONG_PATCH_SECONDS_CHOICES,
    PATCH_SECONDS,
    MIN_RESOLUTION_RATIO,
    SHORT_PATCH_SECONDS_CHOICES,
    VAL_RESOLUTION_PAIR,
    CorpusIndex,
    MultiResolutionCollate,
    MultiScaleCollate,
    PretrainDataset,
    CORPUS_MATCHED_TRAIN_DATASETS,
    TRAIN_DATASETS,
    MAX_BATCH_TOKENS,
    WINDOW_SECONDS,
    TemperatureSampler,
    modalities_present,
    _seed_worker,
)

GYRO_IDX = [3, 4, 5]
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "pretrain"


def future_resolution_durations(cfg: "PretrainConfig") -> tuple[float, ...]:
    """Token durations represented by the collate grid used for future JEPA."""

    if cfg.frontend == "multispan":
        return tuple(cfg.multispan_durations)
    return (tuple(cfg.future_patch_durations) if cfg.multiresolution
            else (float(cfg.patch_seconds),))


def future_tokens_per_window(cfg: "PretrainConfig") -> int:
    """Conservative token count used to size a future-JEPA batch."""

    if cfg.frontend == "multispan":
        return multispan_frame_count(
            cfg.multispan_durations, cfg.source_window_seconds, cfg.frames_per_span,
        )
    return sum(
        max(1, int(math.ceil(cfg.source_window_seconds / duration - 1e-6)))
        for duration in future_resolution_durations(cfg)
    )


def batch_under_token_budget(tokens_per_window: int, reference_batch: int = 512) -> int:
    """Largest multiple of 32 no larger than the reference or token budget."""

    if tokens_per_window <= 0:
        raise ValueError("tokens_per_window must be positive")
    ceiling = max(MAX_BATCH_TOKENS // tokens_per_window, 1)
    candidate = min(reference_batch, ceiling)
    return max((candidate // 32) * 32, 1) if candidate >= 32 else candidate


def frontend_rope_min_period(cfg: "PretrainConfig") -> float:
    """Fastest physical period resolvable by the selected token grid."""

    if cfg.frontend == "multispan":
        return 2.0 * min(cfg.multispan_durations) / cfg.frames_per_span
    return 0.4 if cfg.multiresolution else 0.5


def validate_source_window_contract(index: CorpusIndex, expected_seconds: float) -> None:
    """Reject stale label-free grids built with a different physical context."""

    mismatches = []
    for ref in index.refs:
        if ref.rate_hz <= 0 or len(ref.shape) < 2:
            continue
        actual = ref.shape[1] / ref.rate_hz
        tolerance = max(1.0 / ref.rate_hz, 1e-6)
        if abs(actual - expected_seconds) > tolerance:
            mismatches.append(f"{ref.key}={actual:g}s")
    if mismatches:
        shown = ", ".join(mismatches[:8])
        suffix = " ..." if len(mismatches) > 8 else ""
        raise ValueError(
            f"label-free JEPA requires {expected_seconds:g}s source grids, but found "
            f"{shown}{suffix}. Rebuild with `python -m data.pretraining.build_corpus --stage grids`."
        )


@dataclass
class PretrainConfig:
    # The depth sweep found that three layers retain activity information better than six while
    # preserving temporal and cross-sensor context.
    d_model: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    frontend: str = "fixed"               # fixed | constrained-learnable | continuous kernels
    corpus_name: str = "label_free"
    source_window_seconds: float = PRETRAIN_WINDOW_SECONDS
    dft_size: int = DFT_SIZE               # serialized architecture/collate capacity
    trunk: str = "dual"                   # dual (checkpoint-compatible) | temporal (compact engine)
    # Omit the Phase-A-only descriptor head unless its explicit objective is enabled. Serialize this
    # shape decision so strict reconstruction never has to infer it from state-dict prefixes.
    descriptor_prediction: bool = False
    # Config-text conditioning (docs/design/TEXT_CONDITIONING.md §4b). 'per_channel' (default) is the
    # legacy one-description-per-channel path; 'factored' splits it into per-channel ROLE text +
    # per-sensor IDENTITY text. Default MUST stay 'per_channel' (do-no-harm). asdict(cfg) serializes
    # both into the checkpoint config, so eval/reconstruction picks up the arm automatically.
    text_conditioning: str = "per_channel"
    # Arm A of the IMWUT comparison line: strip DEVICE / PLACEMENT / GRAVITY from the conditioning
    # text so the encoder is never told the acquisition configuration at ANY stage. Sensor MODALITY
    # survives, because the masked-sensor objective needs to know what it is reconstructing.
    # Compatibility is handled when the support set is built instead, and the claim only holds if
    # pretraining honours it too — fine-tuning alone would leave the encoder shaped by text it saw
    # here. Serialized into the checkpoint so evaluation cannot silently mix the two arms.
    neutral_acquisition_text: bool = False
    # Direct constructors retain the legacy channel path for explicit tests/ablations. The Phase-A
    # CLI sets the design-of-record sensor path below.
    token_granularity: str = "channel"
    # Retained in checkpoint metadata solely to reconstruct legacy encoders that used the artifact.
    sensor_bias_dim: int = 14
    use_sensor_bias_conditioning: bool = False
    use_sensor_isolated_retrieval: bool = True
    descriptor_weight: float = 0.0        # explicit ablation; default JEPA predicts signal latents only
    gate_bias_init: float = -2.0          # factored fusion identity-gate bias at init (sigma~=0.12)
    # ``future`` is the design of record. ``masked`` reproduces the earlier bidirectional control.
    jepa_mode: str = "future"              # future | masked
    # Future JEPA aligns all target choices in physical time. Fixed durations make runs directly
    # comparable and let the predictor condition explicitly on temporal support.
    multiresolution: bool = True
    future_patch_durations: tuple[float, ...] = (0.5, 1.0, 1.5)
    multispan_durations: tuple[float, ...] = (0.5, 1.0, 1.5)
    frames_per_span: int = MS_FRAMES_PER_SPAN
    rope_min_period: float = 0.5
    future_context_fraction: tuple[float, float] = (0.4, 0.7)
    future_horizon_bins_seconds: tuple[tuple[float, float], ...] = DEFAULT_HORIZON_BINS_SECONDS
    future_teacher_top_layers: int = 2
    future_predictor_dim: int = 128
    future_predictor_layers: int = 2
    future_predictor_heads: int = 4
    patch_seconds: float = PATCH_SECONDS
    frontend_lr_scale: float = 0.1         # physical adaptation moves slower than the encoder
    frontend_reg_weight: float = 1e-3
    center_shift_fraction: float = 0.45
    bandwidth_factor_max: float = 1.5
    compression_gain_max: float = 2.0
    filter_shape_min: float = 1.5
    filter_shape_max: float = 2.5
    adaptive_gate_init: float = 0.1
    duration_gate_init: float = 0.1
    short_patch_choices: tuple[float, ...] = SHORT_PATCH_SECONDS_CHOICES
    long_patch_choices: tuple[float, ...] = LONG_PATCH_SECONDS_CHOICES
    min_resolution_ratio: float = MIN_RESOLUTION_RATIO
    val_resolution_pair: tuple[float, float] = VAL_RESOLUTION_PAIR
    # Single-resolution starts from the 1024 x 7,500 recipe. Future-JEPA schedules are derived
    # below to preserve the 7.68-million-window reference under the actual token budget.
    steps: int = 7_500                    # ~4.4 aggregate expanded-corpus equivalents
    # Conservative square-root LR scaling from 3e-4 at batch 256. Doubling weight decay preserves
    # approximately the same integrated AdamW shrink over one quarter as many optimizer updates.
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 250               # same 256k warmup windows as 1,000 steps at batch 256
    grad_clip: float = 1.0
    # In future mode these are the three visible objective coefficients. The legacy VICReg weight
    # remains below for the masked control and is not active in future mode.
    jepa_weight: float = 1.0
    # Warmup starts near the intended gradient proportions: the collapse term already contains
    # VICReg's internal 25x variance coefficient, while the randomly initialized physical decoder
    # attenuates its encoder gradient. The post-warmup calibration measures and freezes exact values.
    physical_weight: float = 5.0
    collapse_weight: float = 0.01
    vicreg_weight: float = 1.0
    # 0.996^4 preserves the EMA half-life in examples when batch 256 -> 1024 and updates divide by 4.
    jepa_ema_decay: float = 0.984095744256
    # BYOL RAMPS the teacher decay rather than fixing it (cosine to exactly 1.0):
    #     tau_k = 1 - (1 - tau_base) * (cos(pi*k/K) + 1) / 2,  tau_base -> 1.0
    # data2vec also anneals but is NOT the source of this shape: it uses a LINEAR ramp for speech
    # and NLP, and a constant 0.9998 for vision. The 'cosine' arm here is BYOL's.
    # The trade-off is real -- early training wants a teacher that moves (the student has
    # nothing to learn from a frozen random target), late training wants one that is stable.
    # The reference batch-256 momentum was 0.996. The batch-1024 value above is its exact
    # example-time equivalent, not a claim that a lower-momentum teacher is intrinsically better.
    jepa_ema_schedule: str = "cosine"     # fixed | cosine (BYOL ramp to 1.0)
    # Realised mask fraction is (L + patch)/W, so nominal 0.5 currently masks ~0.56 of short
    # tokens and ~0.63 of long ones. Comparable methods sit HIGHER: BEiT 40%, MAE 75%
    # (its ablation shows linear-probe accuracy climbing steadily to 75%, a ~20-point gap over
    # low ratios), data2vec 2.0 ~80%, I-JEPA context 0.7-1.0. Raising this also reduces the
    # cross-resolution leak, since fewer short tokens stay visible inside a masked long one.
    mask_ratio_time: float = MASK_RATIO_TIME
    vicreg_invariance_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    vicreg_target_std: float = 1.0
    collapse_variance_weight: float = 25.0
    collapse_covariance_weight: float = 1.0
    collapse_target_std: float = 1.0
    # VICReg expander, hidden and output widths kept separate; see PipelineAModel. Both defaults
    # reproduce the historical hard-coded 256->256->128 exactly, so `pretrain` with no width flag
    # is the control. Output 128 is HALF d_model -- the literature ratio is several times LARGER
    # -- so this is swept, not assumed. Sweep caveat: the 128/512/1024 arms already run moved
    # hidden and output TOGETHER, and only the 512/1024 arms differed from the historical hidden
    # width, so they do not isolate which of the two widths mattered.
    vicreg_proj_hidden: int = 256
    vicreg_proj_dim: int = 128
    retrieval_vicreg_fraction: float = 0.5
    rotation_p: float = 0.0
    rotation_pairing: str = "shared"
    rate_augmentation_p: float = 0.0
    channel_dropout_p: float = 0.0
    jitter_p: float = 0.0
    scale_p: float = 0.0
    gravity_p: float = 0.0
    channel_text_phrase_p: float = 0.0
    channel_text_dropout_p: float = 0.0
    # Masked reconstruction of the PARAMETER-FREE filterbank analysis features. Set > 0 (with
    # jepa_weight 0) to swap the self-referential EMA-latent target for a fixed physical one.
    mae_weight: float = 0.0
    # Optional one-time post-warmup calibration. Report mode writes the recommendation and stops;
    # apply mode installs it after the calibration step, freezes it, and continues the same run.
    objective_calibration_at: int = 0       # 0 disables; recommended full pilot: 2_000
    objective_calibration_batches: int = 50
    objective_target_jepa_share: float = 0.70
    objective_target_physical_share: float = 0.20
    objective_calibration_mode: str = "report"  # report (stop) | apply (freeze and continue)
    # Label-free hierarchical corpus sampler. Dataset mass is tempered and capped, then distributed
    # within each dataset as P(subject) ∝ n_subject^subject_alpha. This keeps Capture-24's useful
    # scale without letting one acc-only wrist corpus or its longest subjects define the encoder.
    sampler_alpha: float = 0.25
    sampler_max_dataset_share: float = 0.25
    sampler_subject_alpha: float = 0.5
    homogeneous_sensor_batches: bool = True
    batch_size: int = 1_024               # measured peak: 4.75 GiB on a 24 GB RTX 4090
    calib_batches: int = 50               # frontend norm calibration pass
    val_every: int = 500                  # about every 43 s at the measured batch-1024 rate
    val_per_label: int = 40               # kNN val: windows PER LABEL (stratified, all classes scored)
    knn_k: int = 5
    selection_datasets: tuple[str, ...] = PHASE_A_SELECTION_DATASETS
    selection_every: int = 2_000
    # Compile only the transformer, leaving ragged text conditioning eager. Compiling encoder.encode
    # specialized on each batch's changing unique-text cardinality and was slower. Transformer-only
    # compilation is checkpoint-neutral; fall back to eager if the backend cannot lower a shape.
    compile_encoder: bool = False
    num_workers: int = 12                 # re-profiled 2026-08-07 on the production two-view loader:
                                          # steady wait 38.8 ms (nw=8) -> 24.1 ms (nw=12), and the
                                          # live GPU loop's exposed wait fell 7.7 ms -> <0.2 ms.
    seed: int = 20260718                  # MODEL seed: weight init, augmentation, batch order (varies per replicate)
    # DATA seed: the subject train/val split. Held FIXED across arms AND replicates so every run — and
    # the metric harness — sees the SAME subject-disjoint split (audit 2026-07-23 #1: the eval used the
    # default split regardless of --seed, so seed!=default leaked ~19 train subjects into metric-val).
    data_seed: int = 20260718
    # Serialized explicitly into every checkpoint. Never leave this as an implicit reference to a
    # mutable module-level default: changing the roster after training must not change bank rebuilds.
    train_datasets: tuple | None = None
    max_per_stream: int | None = None     # None = use all windows; temperature sampling controls sources
    device: str = "cpu"


def hydrate_calibrated_objective_weights(
    cfg: PretrainConfig,
    saved_cfg: dict,
    saved_step: int,
    explicit_fields: set[str] | None = None,
) -> bool:
    """Restore calibration trajectory state before strict resume validation.

    This applies both before and after the calibration step. Explicit CLI overrides are left alone
    so the ordinary resume validator can reject an attempted trajectory change.
    """
    del saved_step  # The schedule is immutable from step zero, even before calibration executes.
    explicit = explicit_fields or set()
    fields = {
        "objective_calibration_at": int,
        "objective_calibration_batches": int,
        "objective_target_jepa_share": float,
        "objective_target_physical_share": float,
        "objective_calibration_mode": str,
        "jepa_weight": float,
        "physical_weight": float,
        "collapse_weight": float,
        "vicreg_weight": float,
    }
    applied = False
    for key, cast in fields.items():
        if key in saved_cfg and key not in explicit:
            setattr(cfg, key, cast(saved_cfg[key]))
            applied = True
    return applied


class PipelineAModel(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        duration_choices = (future_resolution_durations(cfg) if cfg.jepa_mode == "future"
                            else tuple(cfg.short_patch_choices) + tuple(cfg.long_patch_choices))
        duration_min = min(duration_choices)
        duration_max = max(duration_choices)
        if duration_min == duration_max:
            # Single-scale controls do not activate duration conditioning, but the shared encoder
            # still requires a non-degenerate normalization interval for checkpoint compatibility.
            duration_min, duration_max = 0.5 * duration_min, 1.5 * duration_max
        self.encoder = SetTokenizerEncoder(
            d_model=cfg.d_model, num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout, dft_size=cfg.dft_size,
            frontend=cfg.frontend,                # 'fixed' (default) | 'learnable'
            trunk=cfg.trunk,
            descriptor_prediction=cfg.descriptor_prediction,
            text_conditioning=cfg.text_conditioning,  # 'per_channel' (default) | 'factored'
            token_granularity=cfg.token_granularity,
            sensor_bias_dim=cfg.sensor_bias_dim,
            use_sensor_bias_conditioning=cfg.use_sensor_bias_conditioning,
            use_sensor_isolated_retrieval=cfg.use_sensor_isolated_retrieval,
            gate_bias_init=cfg.gate_bias_init,
            # Center time alone cannot distinguish overlapping tokens with different physical
            # support. Duration is therefore explicit at both channel and sensor granularity.
            use_duration_embedding=(cfg.multiresolution or cfg.frontend == "multispan"),
            duration_min_seconds=duration_min,
            duration_max_seconds=duration_max,
            duration_gate_init=cfg.duration_gate_init,
            num_resolutions=max(2, len(future_resolution_durations(cfg))),
            rope_min_period=frontend_rope_min_period(cfg),
            center_shift_fraction=cfg.center_shift_fraction,
            bandwidth_factor_max=cfg.bandwidth_factor_max,
            compression_gain_max=cfg.compression_gain_max,
            filter_shape_min=cfg.filter_shape_min,
            filter_shape_max=cfg.filter_shape_max,
            adaptive_gate_init=cfg.adaptive_gate_init,
            **({"spans": cfg.multispan_durations, "frames_per_span": cfg.frames_per_span}
               if cfg.frontend == "multispan" else {}),
        )
        self.encoder.multiresolution = cfg.multiresolution or cfg.frontend == "multispan"
        self.encoder.eval_resolutions = tuple(
            future_resolution_durations(cfg) if cfg.jepa_mode == "future"
            else (cfg.val_resolution_pair if cfg.multiresolution else (cfg.patch_seconds,))
        )
        self.encoder.eval_resolution_pair = self.encoder.eval_resolutions
        self.encoder.eval_patch_seconds = float(cfg.patch_seconds)
        self.encoder.min_resolution_ratio = float(cfg.min_resolution_ratio)
        if cfg.token_granularity == "sensor" and cfg.descriptor_weight <= 0:
            self.encoder.descriptor_prediction_enabled = False
            if self.encoder.descriptor_head is not None:
                self.encoder.descriptor_head.requires_grad_(False)
        self.jepa_predictor = None
        self.future_predictor = None
        self.physical_decoder = None
        self.vicreg_projector = None
        self.physical_target_analyzer = None
        if cfg.jepa_mode == "future":
            # Future rows are removed as attention keys/values rather than replaced by a learned
            # mask. Keep the legacy parameter serialized for encoder checkpoint compatibility, but
            # do not present dead state to the optimizer or gradient-health diagnostics.
            self.encoder.mask_token.requires_grad_(False)
            self.future_predictor = FuturePredictor(
                cfg.d_model,
                predictor_dim=cfg.future_predictor_dim,
                num_layers=cfg.future_predictor_layers,
                num_heads=cfg.future_predictor_heads,
                dropout=cfg.dropout,
                max_resolutions=max(8, len(future_resolution_durations(cfg))),
            )
            if cfg.physical_weight > 0:
                # This frozen analyzer defines the physical target independently of the selected
                # student frontend. A learnable/continuous encoder must not be allowed to move its
                # own reconstruction target. Its projection exists only to keep this a normal,
                # serializable module; JEPA reads `analyze()` and never calls `project()`.
                self.physical_target_analyzer = PhysicalFilterbankTokenizer(
                    d_model=cfg.d_model, dft_size=cfg.dft_size, learnable=False,
                )
                self.physical_target_analyzer.requires_grad_(False)
                # Standardized band energies and signed DC for each xyz axis. Raw total energy is
                # omitted until it has an independently frozen corpus calibration.
                physical_axis_dim = (
                    self.physical_target_analyzer.n_bands
                    + int(self.physical_target_analyzer.use_dc)
                )
                self.physical_decoder = nn.Sequential(
                    nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                    nn.Linear(cfg.d_model, 3 * physical_axis_dim),
                )
        else:
            self.jepa_predictor = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
        # MAE head: sensor token -> the three axes' physical analysis features concatenated.
        # Built only when requested so the reference recipe's state_dict is unchanged.
        self.mae_head = None
        if cfg.mae_weight > 0:
            analysis_dim = self.encoder.filterbank.proj.in_features
            self.mae_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                nn.Linear(cfg.d_model, 3 * analysis_dim),
            )
        # VICReg/Barlow-Twins expander. Width is a measured knob, not a detail: VICReg Table 12
        # reports ImageNet linear top-1 of 55.9 / 59.2 / 62.4 / 65.1 / 67.3 / 68.6 / 68.8 at
        # expander dims 256 / 512 / 1024 / 2048 / 4096 / 8192 / 16384 -- monotonic, +12.7 points
        # from 256 to 8192 -- and states the rule directly: "performance improves when the size of
        # the expander layers is larger than the dimension of the representation" (their encoder
        # is 2048, their expander 8192, i.e. 4x WIDER). The hard-coded 128 here was HALF our
        # d_model=256, the opposite ratio, and below the smallest point they tested.
        # Caveat kept in view: that curve is ImageNet-scale. On CIFAR/STL-scale corpora wide
        # expanders overfit (Guarding Barlow Twins Against Overfitting, 2023), and our corpus is
        # 1.5M windows -- so this is swept, not assumed.
        # Hidden and output widths are separate. Folding them into one knob silently rewrote the
        # DEFAULT architecture (256->256->128, 98,688 params) into 256->128->128 (49,408), so the
        # unchanged default command stopped being a control-equivalent run.
        if cfg.jepa_mode == "masked":
            self.vicreg_projector = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.vicreg_proj_hidden), nn.GELU(),
                nn.Linear(cfg.vicreg_proj_hidden, cfg.vicreg_proj_dim),
            )

    def pretraining_heads(self) -> dict[str, nn.Module]:
        """Return only active objective heads for checkpointing and gradient telemetry."""
        names = ("jepa_predictor", "future_predictor", "physical_decoder",
                 "vicreg_projector", "mae_head")
        return {name: module for name in names if (module := getattr(self, name, None)) is not None}


@torch.no_grad()
def update_ema_encoder(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """Update teacher parameters by EMA with one foreach launch per arithmetic operation.

    The teacher is deep-copied after filterbank calibration. Its non-parameter state consists only
    of frozen calibration statistics and RoPE frequencies, so copying those immutable buffers on
    every optimizer update is unnecessary.

    ``decay == 1.0`` (a frozen teacher) is accepted because the BYOL cosine ramp reaches exactly
    1.0 on its final step; rejecting it crashed every cosine run at ``step == cfg.steps``, after
    the last checkpoint. It is still rejected as a *fixed* value by the CLI, where it would mean
    a randomly-initialised teacher for the entire run.
    """
    if not 0.0 <= float(decay) <= 1.0:
        raise ValueError("EMA decay must be in [0, 1]")
    cache = getattr(teacher, "_ema_update_cache", None)
    if cache is None or cache[0] != id(student):
        student_named = dict(student.named_parameters())
        teacher_named = dict(teacher.named_parameters())
        if student_named.keys() != teacher_named.keys():
            raise ValueError("EMA student and teacher parameter structures differ")
        cache = (id(student), tuple(student_named.values()), tuple(teacher_named.values()))
        object.__setattr__(teacher, "_ema_update_cache", cache)
    _, student_params, teacher_params = cache
    torch._foreach_mul_(teacher_params, float(decay))
    torch._foreach_add_(teacher_params, student_params, alpha=1.0 - float(decay))


@torch.no_grad()
def representation_health(z: torch.Tensor, prefix: str = "repr") -> dict[str, float]:
    """Small-batch collapse diagnostics over an embedding matrix."""
    # Collapse diagnostics are an FP32 island. Merely calling `.float()` is insufficient inside a
    # surrounding CUDA autocast region: the covariance matmul is cast back to BF16, whose symmetric
    # eigensolver is not implemented and whose spectrum would be too coarse for effective rank.
    with torch.autocast(device_type=z.device.type, enabled=False):
        x = z.detach().float()
        std = x.std(dim=0, unbiased=False)
        centered = x - x.mean(0)
        denominator = max(len(x) - 1, 1)
        cov = centered.T @ centered / denominator
        # The telemetry batch is usually much narrower than the embedding dimension, so the
        # covariance is rank-deficient by construction. Solving its repeated zero eigenspace can
        # fail to converge on CUDA even for finite inputs. The nonzero covariance eigenvalues are
        # exactly squared singular values of the centered sample matrix divided by (n - 1), and the
        # rectangular SVD is both smaller and numerically better conditioned here.
        eig = torch.linalg.svdvals(centered).square().div(denominator).clamp_min(0)
        probs = eig / eig.sum().clamp_min(1e-12)
        effective_rank = torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum())
        offdiag = cov - torch.diag_embed(torch.diagonal(cov))
        names = tuple(f"{prefix}/{name}" for name in (
            "min_std", "mean_std", "effective_rank", "mean_norm",
            "cov_offdiag_abs_mean", "cov_max_eigenvalue",
        ))
        # One device-to-host synchronization for the whole diagnostic instead of one per scalar.
        values = torch.stack((
            std.min(), std.mean(), effective_rank, x.norm(dim=1).mean(),
            offdiag.abs().mean(), eig.max(),
        )).cpu().tolist()
    return dict(zip(names, values))


@torch.no_grad()
def ema_teacher_health(student: nn.Module, teacher: nn.Module) -> dict[str, float]:
    """Distance between the student and EMA teacher with one device synchronization."""
    delta_sq = None
    student_sq = None
    for student_parameter, teacher_parameter in zip(student.parameters(), teacher.parameters()):
        delta = student_parameter.detach().float() - teacher_parameter.detach().float()
        delta_term = delta.square().sum()
        student_term = student_parameter.detach().float().square().sum()
        delta_sq = delta_term if delta_sq is None else delta_sq + delta_term
        student_sq = student_term if student_sq is None else student_sq + student_term
    if delta_sq is None:
        return {"ema/parameter_distance": 0.0, "ema/relative_parameter_distance": 0.0}
    values = torch.stack((delta_sq.sqrt(), delta_sq.sqrt() / student_sq.sqrt().clamp_min(1e-12)))
    distance, relative = values.cpu().tolist()
    return {"ema/parameter_distance": distance, "ema/relative_parameter_distance": relative}


def objective_encoder_grad_geometry(
    losses: dict[str, torch.Tensor], encoder: nn.Module,
) -> dict[str, dict[str, float]]:
    """Norms, dots, and cosines of objective gradients on the shared encoder.

    Loss values are not comparable across objectives. This geometry is computed before clipping and
    after every configured loss coefficient already present in ``losses``. Callers that need unit
    coefficients must pass unit-weight tensors explicitly.
    """
    parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    gradients: dict[str, list[torch.Tensor | None]] = {}
    for name, loss in losses.items():
        if loss.requires_grad:
            gradients[name] = list(torch.autograd.grad(
                loss, parameters, retain_graph=True, allow_unused=True,
            ))
        else:
            gradients[name] = [None] * len(parameters)

    norm_tensors: dict[str, torch.Tensor] = {}
    for name, values in gradients.items():
        pieces = [value.detach().float().square().sum()
                  for value in values if value is not None]
        reference = losses[name]
        squared = (torch.stack(pieces).sum() if pieces else
                   reference.detach().new_zeros((), dtype=torch.float32))
        norm_tensors[name] = squared.sqrt()

    norm_names = list(norm_tensors)
    norm_values = torch.stack([norm_tensors[name] for name in norm_names]).cpu().tolist()
    norms = dict(zip(norm_names, norm_values))

    dots: dict[str, float] = {}
    cosines: dict[str, float] = {}
    names = list(losses)
    for position, left in enumerate(names):
        for right in names[position + 1:]:
            key = f"{left}|{right}"
            pieces = [(a.detach().float() * b.detach().float()).sum()
                      for a, b in zip(gradients[left], gradients[right])
                      if a is not None and b is not None]
            dot = float(torch.stack(pieces).sum()) if pieces else 0.0
            dots[key] = dot
            denom = norms[left] * norms[right]
            cosines[key] = dot / denom if denom > 0 else 0.0
    return {"norms": norms, "dots": dots, "cosines": cosines}


def recommend_objective_weights(
    samples: list[dict[str, dict[str, float]]],
    *,
    current_jepa_weight: float,
    current_vicreg_weight: float,
    target_jepa_share: float = 0.45,
) -> dict[str, object]:
    """Solve one frozen JEPA/VICReg scalarization from post-warmup gradient samples.

    ``samples`` contain unit-weight gradients named ``jepa`` and ``vicreg``. A common scale
    preserves the pilot's median combined encoder gradient, avoiding an accidental effective
    learning-rate or clipping change when the relative coefficient is corrected.
    """
    if not samples:
        raise ValueError("objective calibration needs at least one gradient sample")
    if not 0 < target_jepa_share < 1:
        raise ValueError("target JEPA gradient share must be in (0,1)")

    required = {"jepa", "vicreg"}
    for sample in samples:
        if set(sample.get("norms", {})) != required:
            raise ValueError("calibration samples must contain JEPA and VICReg gradients")

    def _median_norm(name: str) -> float:
        return statistics.median(float(sample["norms"][name]) for sample in samples)

    jepa_norm = _median_norm("jepa")
    vicreg_norm = _median_norm("vicreg")
    if jepa_norm <= 0 or vicreg_norm <= 0:
        raise ValueError("JEPA and VICReg must both produce non-zero calibration gradients")

    def _dot(sample: dict[str, dict[str, float]], left: str, right: str) -> float:
        direct = f"{left}|{right}"
        reverse = f"{right}|{left}"
        return float(sample["dots"].get(direct, sample["dots"].get(reverse, 0.0)))

    def _combined_norm(sample: dict[str, dict[str, float]], coefficients: dict[str, float]) -> float:
        total = 0.0
        names = list(coefficients)
        for name in names:
            total += coefficients[name] ** 2 * float(sample["norms"][name]) ** 2
        for position, left in enumerate(names):
            for right in names[position + 1:]:
                total += 2.0 * coefficients[left] * coefficients[right] * _dot(
                    sample, left, right
                )
        return math.sqrt(max(total, 0.0))

    def _solve_median_share(share_at, target: float) -> float:
        """Monotonic bisection for a coefficient whose median per-batch share is ``target``."""
        if target <= 0:
            return 0.0
        low, high = 0.0, 1.0
        while statistics.median(share_at(high)) < target:
            high *= 2.0
            if high > 1e12:
                raise ValueError("could not bracket objective calibration coefficient")
        for _ in range(80):
            middle = 0.5 * (low + high)
            if statistics.median(share_at(middle)) < target:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    jepa_to_vicreg = _solve_median_share(
        lambda weight: [
            weight * float(sample["norms"]["jepa"])
            / max(weight * float(sample["norms"]["jepa"])
                  + float(sample["norms"]["vicreg"]), 1e-12)
            for sample in samples
        ],
        target_jepa_share,
    )

    baseline_coefficients = {
        "jepa": float(current_jepa_weight),
        "vicreg": float(current_vicreg_weight),
    }
    proposed_coefficients = {
        "jepa": jepa_to_vicreg,
        "vicreg": 1.0,
    }
    baseline_norm = statistics.median(
        _combined_norm(sample, baseline_coefficients) for sample in samples
    )
    proposed_norm = statistics.median(
        _combined_norm(sample, proposed_coefficients) for sample in samples
    )
    common_scale = baseline_norm / proposed_norm if proposed_norm > 0 else 1.0

    jepa_shares = [
        jepa_to_vicreg * float(sample["norms"]["jepa"])
        / max(jepa_to_vicreg * float(sample["norms"]["jepa"])
              + float(sample["norms"]["vicreg"]), 1e-12)
        for sample in samples
    ]

    def _distribution(values: list[float]) -> dict[str, float]:
        return {
            "min": min(values),
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "median": statistics.median(values),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "max": max(values),
        }

    return {
        "recommended": {
            "jepa_weight": common_scale * jepa_to_vicreg,
            "vicreg_weight": common_scale,
        },
        "target_norm_shares": {
            "jepa_vs_vicreg": float(target_jepa_share),
        },
        "median_unit_gradient_norms": {
            "jepa": jepa_norm,
            "vicreg": vicreg_norm,
        },
        "median_combined_encoder_grad_norm": {
            "pilot_weights": baseline_norm,
            "recommended_weights": proposed_norm * common_scale,
        },
        "realized_norm_share_distribution": {
            "jepa_vs_vicreg": _distribution(jepa_shares),
        },
        "common_scale": common_scale,
    }


_SOURCE_SUFFIXES = {
    ".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt", ".sh", ".ini", ".cfg",
}
# Source and configuration that can affect a Phase-A trajectory. Scoping the snapshot prevents an
# unrelated Phase-B edit in the shared repository from invalidating a Phase-A resume, while the
# corpus fingerprint below independently covers the realised grid contents.
_PHASE_A_SOURCE_ROOTS = (
    "training/tokenizer", "model/tokenizer", "data/datasets", "data/pretraining", "data/scripts",
    "data/labels", "data/quality/duplicate_windows.json",
    "data/quality/implausible_windows.json", "pyproject.toml",
)
_RUN_ARTIFACT_NAMES = {
    "log.jsonl", "run_config.json", "objective_calibration.json",
    "source.patch", "source_provenance.json", "runtime_provenance.json",
    "health.json", "health.txt", "telemetry.png",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=_repo_root(), capture_output=True, check=check, timeout=30,
    )


def capture_source_provenance(
    out: Path,
    *,
    write: bool = True,
    roots: tuple[str, ...] = _PHASE_A_SOURCE_ROOTS,
) -> dict:
    """Persist a reconstructable patch for tracked and untracked source files."""
    head = _git(["rev-parse", "HEAD"]).stdout.decode().strip()
    tracked = _git([
        "diff", "HEAD", "--binary", "--", *roots,
        # Results from Phase A, Phase B, and diagnostics are runtime artifacts, not source. Including
        # another job's changing JSON here made an otherwise faithful Phase-A resume fail its source
        # fingerprint check and inflated the snapshot by several megabytes in a shared worktree.
        ":(exclude,glob)**/outputs/**",
    ]).stdout
    untracked_raw = _git(["ls-files", "--others", "--exclude-standard", "-z"]).stdout
    untracked = []
    chunks = [tracked]
    for raw in untracked_raw.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8")
        path = _repo_root() / rel
        in_phase_a_source = any(
            rel == root or rel.startswith(root.rstrip("/") + "/")
            for root in roots
        )
        if (not in_phase_a_source or "outputs" in Path(rel).parts
                or path.suffix.lower() not in _SOURCE_SUFFIXES):
            continue
        # Source/config files should be small. Fail rather than silently omit a file required to
        # reconstruct this worktree.
        if path.stat().st_size > 10 * 1024 * 1024:
            raise RuntimeError(f"untracked source file is too large for provenance patch: {rel}")
        diff = _git(["diff", "--no-index", "--binary", "--", "/dev/null", rel], check=False)
        if diff.returncode not in (0, 1):
            raise RuntimeError(diff.stderr.decode(errors="replace"))
        chunks.append(diff.stdout)
        untracked.append(rel)
    patch = b"".join(chunks)
    patch_sha256 = hashlib.sha256(patch).hexdigest()
    metadata = {
        "git": f"{head[:7]}-dirty" if patch else head[:7],
        "head": head,
        "dirty": bool(patch),
        "patch_sha256": patch_sha256,
        "patch_bytes": len(patch),
        "untracked_source_files": sorted(untracked),
    }
    if write:
        (out / "source.patch").write_bytes(patch)
        (out / "source_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metadata["_patch"] = patch
    return metadata


def write_source_provenance(out: Path, metadata: dict) -> dict:
    """Write a previously captured source snapshot and return its serializable metadata."""
    serializable = {key: value for key, value in metadata.items() if key != "_patch"}
    (out / "source.patch").write_bytes(metadata.get("_patch", b""))
    (out / "source_provenance.json").write_text(json.dumps(serializable, indent=2) + "\n")
    return serializable


def capture_runtime_provenance(device: torch.device) -> dict[str, object]:
    """Software and accelerator identity needed to reconstruct a reported training run."""
    import scipy

    info: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "mixed_precision": "fp16" if device.type == "cuda" else "fp32",
        "dynamic_loss_scaling": device.type == "cuda",
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update({
            "gpu_name": props.name,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            "gpu_total_memory_bytes": props.total_memory,
        })
    return info


def prepare_output_dir(out: Path, *, force: bool, smoke: bool, resume: bool) -> list[Path]:
    """Create a run directory and remove every known artifact for an explicit fresh overwrite."""
    out.mkdir(parents=True, exist_ok=True)
    stale = sorted(
        {path for path in out.glob("*.pt")}
        | {path for path in out.glob("objective_calibration*.json")}
        | {out / name for name in _RUN_ARTIFACT_NAMES if (out / name).exists()}
    )
    if stale and not resume:
        if force or smoke:
            for path in stale:
                path.unlink()
        else:
            raise SystemExit(
                f"output dir {out} already contains {[path.name for path in stale]}; "
                "choose a fresh --out or pass --force to overwrite (or --resume)."
            )
    return stale


def corpus_fingerprint(index) -> str:
    """Stable 16-hex signature of the assembled TRAINING corpus (per-stream dataset/rate/shape +
    label vocab + the cap/seed/selected-window counts that determine WHICH windows were drawn),
    stored in the checkpoint so it records the exact corpus that produced the weights (F5 — grid
    meta.json carries no raw fingerprint; audit #10: two subset runs with different cap/seed must
    NOT collide, so the sampling knobs and realised split sizes are part of the identity)."""
    import hashlib

    import numpy as _np
    sig = []
    for r in sorted(index.refs, key=lambda r: (r.dataset, r.stream)):
        # CONTENT-sensitive (audit F5): shape+rate+label-COUNT alone collided for two corpora that
        # differed in their per-window label/subject assignment. Hash the actual per-window labels
        # and subjects (small, exact) plus a deterministic strided digest of the signal itself, so a
        # regenerated or re-ordered grid cannot silently reuse an earlier run's identity.
        lab = ",".join(map(str, r.labels)).encode()
        sub = ",".join(map(str, r.subjects)).encode()
        arr = _np.ascontiguousarray(r.load_data()[::97, ::13, :], dtype=_np.float32)
        sig.append(f"{r.dataset}/{r.stream}:{r.rate_hz}:{tuple(r.shape)}"
                   f":{hashlib.sha256(lab).hexdigest()[:12]}"
                   f":{hashlib.sha256(sub).hexdigest()[:12]}"
                   f":{hashlib.sha256(arr.tobytes()).hexdigest()[:12]}")
    sig.append("labels=" + ",".join(sorted(index.label_ids)))
    sig.append("implausible=" + str(getattr(index, "n_implausible_dropped", 0)))
    sig.append(f"cap={getattr(index, 'max_per_stream', None)}:seed={getattr(index, 'seed', None)}"
               f":ntrain={len(index.train)}:nval={len(index.val)}")
    return hashlib.sha256("|".join(sig).encode()).hexdigest()[:16]


def stratified_eval_subset(keys, per_label: int, seed: int, allowed_labels=None):
    """Select a deterministic, stream-covered validation/support subset.

    Every available ``(label, stream)`` cell receives one row before a second row is assigned to
    any cell, then the round-robin repeats until the per-label cap is full. Stream IDs are global
    within a corpus, so this covers both dataset and placement/configuration heterogeneity.
    """
    from collections import defaultdict

    if per_label <= 0:
        raise ValueError("per_label must be positive")
    groups = defaultdict(lambda: defaultdict(list))
    for key in keys:
        if allowed_labels is None or key.label_id in allowed_labels:
            groups[key.label_id][key.stream_i].append(key)

    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(groups):
        cells = groups[label]
        for stream_i in cells:
            order = rng.permutation(len(cells[stream_i]))
            cells[stream_i] = [cells[stream_i][int(i)] for i in order]
        target = min(per_label, sum(len(rows) for rows in cells.values()))
        cursor = {stream_i: 0 for stream_i in cells}
        n_selected = 0
        while n_selected < target:
            active = [stream_i for stream_i in sorted(cells)
                      if cursor[stream_i] < len(cells[stream_i])]
            if not active:
                break
            for position in rng.permutation(len(active)):
                stream_i = active[int(position)]
                selected.append(cells[stream_i][cursor[stream_i]])
                cursor[stream_i] += 1
                n_selected += 1
                if n_selected >= target:
                    break
    return selected


def knn_predict(train_z, train_y, test_z, k: int) -> torch.Tensor:
    if not len(test_z):
        return torch.empty(0, dtype=train_y.dtype)
    d = torch.cdist(test_z.float(), train_z.float())
    nn_lab = train_y[d.topk(min(k, d.shape[1]), largest=False).indices]
    return nn_lab.mode(dim=1).values


def knn_balanced_acc(train_z, train_y, test_z, test_y, k: int) -> float:
    # Score EVERY query label (F1 fix). A query class absent from the support scores 0 — kNN
    # retrieves other-class neighbours — instead of being dropped from the metric. The old
    # `set(train_y) & set(test_y)` intersection silently omitted unsupported query classes,
    # inflating the number and making best.pt selection depend on which classes the random
    # support cap happened to include. Vectorized (cdist+topk+mode) — was a per-query Python loop.
    labels = sorted(set(test_y.tolist()))
    if not labels:
        return float("nan")
    pred = knn_predict(train_z, train_y, test_z, k)
    per_class = [float((pred[test_y == label] == label).float().mean())
                 for label in labels if (test_y == label).any()]
    return float(np.mean(per_class)) if per_class else float("nan")


def balanced_acc(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Macro-averaged per-class recall (balanced accuracy) from hard predictions."""
    per_class = []
    for label in sorted(set(true.tolist())):
        m = true == label
        if m.any():
            per_class.append(float((pred[m] == label).float().mean()))
    return float(np.mean(per_class)) if per_class else float("nan")


def label_group_balanced_acc(pred: torch.Tensor, true: torch.Tensor, groups) -> float:
    """Macro recall over observed ``(label, group)`` cells."""
    if len(pred) != len(true) or len(groups) != len(true):
        raise ValueError("predictions, labels, and groups must have equal length")
    cells = {}
    for row, (label, group) in enumerate(zip(true.tolist(), groups)):
        cells.setdefault((int(label), str(group)), []).append(row)
    recalls = [float((pred[rows] == true[rows]).float().mean())
               for rows in cells.values()]
    return float(np.mean(recalls)) if recalls else float("nan")


@torch.no_grad()
def label_text_prototypes(model: PipelineAModel, label_ids: dict) -> torch.Tensor:
    """(L, 384) L2-normalized frozen-LM embedding of each label's name, indexed BY label id.

    Turns "brushing_teeth" -> "a person brushing teeth" -> mean-pooled MiniLM vector. These are
    the class prototypes for the ConSE-style text-cosine probe: the same frozen text tower the
    encoder already uses, so the probe measures whether the sensor representation aligns to label
    SEMANTICS (the downstream zero-shot target), not just cluster purity like kNN."""
    id2str = {i: s for s, i in label_ids.items()}
    prompts = [f"a person {id2str[i].replace('_', ' ')}" for i in range(len(id2str))]
    emb, mask = model.encoder.text_encoder.encode(prompts, device=torch.device("cpu"))  # (L,S,384)
    m = mask.unsqueeze(-1).float()
    proto = (emb * m).sum(1) / m.sum(1).clamp(min=1.0)
    return torch.nn.functional.normalize(proto, dim=1)


def _l2n(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)


def conse_probe_predict(train_z, train_y, val_z, val_y, protos,
                        ridge_lambda: float = 1.0) -> torch.Tensor:
    """CRUDE-but-comparable zero-shot head: ridge-fit a linear map sensor_emb -> label-text space
    on the TRAIN support (this IS ConSE's semantic projection), then cosine-match each val window
    to the candidate labels' text prototypes (candidates = the val label set). Returns predicted
    label ids (aligned to val_z rows). Fit fresh each val, no calibration — a live proxy for the
    downstream ZS protocol."""
    Zt, Zv = _l2n(train_z.float()), _l2n(val_z.float())
    T = protos[train_y]                                         # (N, 384) target text vectors
    d = Zt.shape[1]
    ridge = ridge_lambda * torch.eye(d, device=Zt.device, dtype=Zt.dtype)
    W = torch.linalg.solve(Zt.t() @ Zt + ridge, Zt.t() @ T)       # (d, 384)
    proj = _l2n(Zv @ W)                                          # val projected into text space
    cand = torch.tensor(sorted(set(val_y.tolist())), device=val_y.device, dtype=val_y.dtype)
    sims = proj @ _l2n(protos[cand]).t()                        # (Nval, Ncand) cosine
    return cand[sims.argmax(dim=1)]


@torch.no_grad()
def embed_stratified(model: PipelineAModel, loader: DataLoader, device, per_label: int,
                     target_labels: set | None = None, label_totals: dict | None = None):
    """Embed EXACTLY ``min(per_label, available)`` windows per label (deterministic — the loader
    must be shuffle=False / pre-shuffled). Guarantees every label is represented so the kNN metric
    covers all classes and best.pt selection is stable.

    Two things the earlier version got wrong, both measured on the live split:

    * The cap was evaluated for a whole batch BEFORE ``counts`` was updated, so a label sitting at
      39/40 admitted every one of its occurrences in that batch. Realised counts ran 3-79 per label
      on val and 27-89 on the train support, i.e. the kNN support bank was never balanced and the
      ConSE ridge fit was weighted toward over-represented labels. ``pending`` now closes the gap
      inside the batch, so the cap is exact.
    * The early exit required every target label to REACH ``per_label``. Labels with fewer than
      ``per_label`` windows in total can never reach it, so the loop always drained the whole
      loader — 186k val + 1.5M train rows per evaluation. ``label_totals`` (the label histogram of
      the keys backing this loader) makes the target ``min(per_label, total)``, which is reachable.
    """
    from collections import Counter
    model.eval()
    zs, ys, srcs, streams = [], [], [], []
    counts: Counter = Counter()
    done: set = set()

    def _target_for(label: int) -> int:
        if label_totals is None:
            return per_label
        return min(per_label, int(label_totals.get(label, per_label)))

    wanted = set(target_labels) if target_labels is not None else None
    for batch in loader:
        lab = batch["labels"]
        take = []
        pending: Counter = Counter()
        for j, l in enumerate(lab.tolist()):
            if wanted is not None and l not in wanted:
                continue
            if counts[l] + pending[l] >= _target_for(l):
                continue
            pending[l] += 1
            take.append(j)
        if take:
            # Match the training precision policy: spectral analysis remains FP32, while the neural
            # projection/conditioning/transformer path uses FP16 autocast on CUDA. Probe embeddings
            # are converted back to FP32 before the CPU kNN/ridge calculations below.
            sensor_granularity = model.encoder.token_granularity == "sensor"
            factored = model.encoder.text_conditioning == "factored" or sensor_granularity
            texts = batch["role_texts"] if factored else batch["texts"]
            patches = batch["patches"].to(device, non_blocking=True).float()
            rates = batch["rates"].to(device, non_blocking=True)
            plen = batch["patch_len"].to(device, non_blocking=True)
            source_rates = batch.get("source_rates", batch["rates"]).to(
                device, non_blocking=True,
            )
            with torch.amp.autocast(
                device.type, enabled=device.type == "cuda", dtype=torch.float16,
            ):
                out = model.encoder(
                    patches, rates, plen, texts,
                    batch["positions"].to(device, non_blocking=True),
                    patch_durations=(batch["patch_durations"].to(device, non_blocking=True)
                                     if "patch_durations" in batch else None),
                    resolution_ids=(batch["resolution_ids"].to(device, non_blocking=True)
                                    if "resolution_ids" in batch else None),
                    channel_mask=batch["channel_mask"].to(device, non_blocking=True),
                    patch_padding_mask=batch["patch_padding_mask"].to(device, non_blocking=True),
                    sensor_texts=(batch["sensor_texts"] if factored else None),
                    sensor_id=(batch["sensor_id"].to(device, non_blocking=True) if factored else None),
                    source_rate_hz=source_rates,
                    sensor_bias=(batch["sensor_bias"].to(device, non_blocking=True)
                                 if sensor_granularity and "sensor_bias" in batch else None),
                )
            pooled = out["pooled"].float().cpu()
            zs.append(pooled[take])
            ys.append(lab[take])
            srcs.extend(batch["sources"][j] for j in take)      # per-window source (telemetry)
            streams.extend(batch["streams"][j] for j in take)
            for l in lab[take].tolist():
                counts[l] += 1
                if counts[l] >= _target_for(l):
                    done.add(l)
        # Exit as soon as every label we are still waiting on has hit its ACHIEVABLE target.
        if wanted is not None and wanted <= done:
            break
        if wanted is None and label_totals is not None and set(label_totals) <= done:
            break
    model.train()
    return torch.cat(zs), torch.cat(ys), srcs, streams


def module_grad_norms(model) -> dict:
    """Per-module gradient L2 norm (call AFTER unscale_, BEFORE clip → real, un-clipped scale).
    A cheap reduction — computed only on log steps, so no hot-loop cost. Diagnoses vanish/explode
    per component (encoder vs each head)."""
    def _gn(params) -> float:
        pieces = [p.grad.detach().float().square().sum()
                  for p in params if p.grad is not None]
        return float(torch.stack(pieces).sum().sqrt()) if pieces else 0.0

    mods = [("encoder", model.encoder), ("frontend", model.encoder.filterbank)]
    mods.extend(model.pretraining_heads().items())
    if model.encoder.token_granularity == "sensor":
        if model.encoder.sensor_fold is not None:
            mods.append(("sensor_fold", model.encoder.sensor_fold))
        mods.append(("descriptor_projection", model.encoder.descriptor_proj))
        if model.encoder.descriptor_head is not None:
            mods.append(("descriptor_head", model.encoder.descriptor_head))
        if model.encoder.use_sensor_bias_conditioning:
            mods.append(("bias_projection", model.encoder.bias_proj))
    return {f"grad/{name}": _gn(mod.parameters()) for name, mod in mods}


def per_source_mean(values: torch.Tensor, sources: list) -> dict:
    """Group a (B,) tensor of per-window values by source dataset (NaN-safe) for telemetry."""
    agg: dict = {}
    for s, v in zip(sources, values.tolist()):
        if v == v:                                              # skip NaN
            agg.setdefault(s, []).append(v)
    return {s: round(float(np.mean(vs)), 4) for s, vs in agg.items()}


def _corpus_datasets(name: str) -> tuple[str, ...]:
    """Resolve a named Phase-A recipe to its dataset roster.

    ``label_free`` is the 2026-09-09 split: the encoder pretrains only on sources that carry no
    activity annotation at all, leaving every labelled corpus genuinely out-of-sample for it.
    The historical ``expanded`` and ``matched`` recipes train the encoder on the same labelled
    data the comparator and classification head later use, which makes a downstream gain
    ambiguous — it can always be read as the encoder having already met those subjects, devices
    and activities. They remain available so earlier runs stay reproducible.
    """
    from data.scripts.curate.deployment_policy import (
        LABEL_FREE_PRETRAIN_DATASETS,
        assert_pretraining_is_label_free,
    )

    if name == "label_free":
        assert_pretraining_is_label_free(LABEL_FREE_PRETRAIN_DATASETS)
        return tuple(LABEL_FREE_PRETRAIN_DATASETS)
    if name == "expanded":
        return tuple(TRAIN_DATASETS)
    if name == "matched":
        return tuple(CORPUS_MATCHED_TRAIN_DATASETS)
    raise ValueError(f"unknown corpus recipe {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--stop-after", type=int, default=None,
                        help="stop and checkpoint at this step while retaining --steps as the full "
                             "LR/EMA schedule (for bounded trajectory monitors)")
    parser.add_argument("--lr", type=float, default=None,
                        help="optimizer peak learning rate (future-JEPA default scales with batch)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="AdamW weight decay (future-JEPA default scales with batch)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="linear LR warmup steps (future-JEPA default preserves warmup examples)")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="global gradient-norm clipping threshold (default 1.0)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny corpus + tiny model for a fast CPU end-to-end check")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing non-empty output dir (default: refuse)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="warm-resume from a checkpoint (restore encoder/heads/opt/sched/scaler/"
                             "RNG + step and continue the remaining steps)")
    parser.add_argument("--frontend", choices=("fixed", "learnable", "continuous", "multispan"),
                        default="fixed",
                        help="tokenizer arm. 'fixed' = physical-Hz constant-Q filterbank (default); "
                             "'learnable' = the historical constrained-adaptive filterbank arm "
                             "(see docs/LEGACY.md); 'continuous' = the "
                             "continuous-time temporal-kernel arm; 'multispan' = span-grouped "
                             "continuous kernels on a shared physical-time grid.")
    parser.add_argument("--jepa-mode", choices=("future", "masked"), default="future",
                        help="future = past-only multi-horizon predictive objective (default); "
                             "masked = historical bidirectional masked-JEPA + VICReg control")
    parser.add_argument("--neutral-acquisition-text", action="store_true", default=None,
                        help="IMWUT Arm A: strip device/placement/gravity from the conditioning "
                             "text so the encoder is never told the acquisition configuration")
    parser.add_argument("--text-conditioning", choices=("per_channel", "factored"), default=None,
                        help="config-text conditioning (docs/design/TEXT_CONDITIONING.md §4b). "
                             "'per_channel' = one description per channel; 'factored' (the CLI "
                             "DEFAULT) = per-channel ROLE text + per-sensor IDENTITY text.")
    parser.add_argument("--token-granularity", choices=("channel", "sensor"), default=None,
                        help="token granularity (docs/design/DESIGN_OF_RECORD.md). 'channel' = one "
                             "token per channel (legacy); 'sensor' = fold each modality triad into "
                             "one token (required by future JEPA).")
    parser.add_argument("--multiresolution", action=argparse.BooleanOptionalAction, default=None,
                        help="jointly encode multiple physical patch durations (default ON for "
                             "fixed/learnable future JEPA; continuous kernels use their native "
                             "one-second token grid)")
    parser.add_argument("--patch-seconds", type=float, default=None,
                        help="single-resolution patch duration in seconds (default 1.0)")
    parser.add_argument("--future-patch-durations", type=float, nargs="+", default=None,
                        help="fixed physical patch durations used jointly by future JEPA "
                             "(default: 0.5 1.0 1.5 seconds)")
    parser.add_argument("--multispan-durations", type=float, nargs="+", default=None,
                        help="physical spans represented by the multi-span continuous frontend "
                             "(default: 0.5 1.0 1.5 seconds)")
    parser.add_argument("--frames-per-span", type=int, default=None,
                        help="multi-span analysis frames per physical span (default: 4)")
    parser.add_argument("--future-context-fraction", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="fractional range from which the past-only context boundary is drawn")
    parser.add_argument("--future-teacher-top-layers", type=int, default=None,
                        help="number of upper EMA-teacher layers averaged into each target")
    parser.add_argument("--future-predictor-dim", type=int, default=None,
                        help="narrow future-predictor width (default 128)")
    parser.add_argument("--future-predictor-layers", type=int, default=None,
                        help="future-predictor decoder depth (default 2)")
    parser.add_argument("--future-predictor-heads", type=int, default=None,
                        help="future-predictor attention heads (default 4)")
    parser.add_argument("--descriptor-weight", type=float, default=None,
                        help="historical masked-control descriptor-reconstruction weight "
                             "(default 0=disabled; rejected by future JEPA)")
    parser.add_argument("--rotation-p", type=float, default=None,
                        help="SO(3) rotation probability (default 0 for the clean reference)")
    parser.add_argument("--rotation-pairing", choices=("shared", "independent"), default=None,
                        help="masked-control only: shared rotates both VICReg views identically; "
                             "independent explicitly trains rotation invariance")
    parser.add_argument("--rate-augmentation-p", type=float, default=None,
                        help="anti-aliased sampling-rate augmentation probability (default 0)")
    parser.add_argument("--mae-weight", type=float, default=None,
                        help="historical masked-control reconstruction of parameter-free "
                             "filterbank features (rejected by future JEPA)")
    parser.add_argument("--gravity-p", type=float, default=None,
                        help="gravity-removal probability; CONFIG-group, widens the acquisition "
                             "distribution (does not demand invariance)")
    parser.add_argument("--channel-text-phrase-p", type=float, default=None,
                        help="channel-text paraphrase probability; NUISANCE-group")
    parser.add_argument("--channel-text-dropout-p", type=float, default=None,
                        help="channel-text dropout probability; CONFIG-group")
    parser.add_argument("--jitter-p", type=float, default=None,
                        help="per-view additive-noise probability; NUISANCE-group, suppresses "
                             "subject/device idiosyncrasy without touching gravity orientation")
    parser.add_argument("--scale-p", type=float, default=None,
                        help="per-view amplitude-scaling probability; NUISANCE-group, same rationale")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="trunk depth. Measured 2026-08-18: activity information peaks at "
                             "depth 1-3 and decays 4-8 points by depth 6 in every trained arm, "
                             "while a random-init trunk is flat")
    parser.add_argument("--trunk", choices=("dual", "temporal"), default=None,
                        help="dual = historical cross-sensor trunk; temporal = compact sensor-"
                             "isolated trunk used by the compact evidence engine")
    parser.add_argument("--d-model", type=int, default=None,
                        help="encoder width (compact small uses 128; historical reference uses 256)")
    parser.add_argument("--dim-feedforward", type=int, default=None,
                        help="trunk feed-forward width (compact small uses 256)")
    parser.add_argument("--channel-dropout-p", type=float, default=None,
                        help="whole-modality dropout probability (default 0)")
    parser.add_argument("--val-every", type=int, default=None,
                        help="steps between validation passes (selection can only fire on one)")
    parser.add_argument("--selection-every", type=int, default=None,
                        help="steps between held-out development-transfer selection scores; must be "
                             "a multiple of --val-every to fire on schedule")
    parser.add_argument("--retrieval-vicreg-fraction", type=float, default=None,
                        help="fraction of VICReg assigned directly to the sensor rows stored in "
                             "the evidence bank (default 0.5)")
    parser.add_argument("--jepa-weight", type=float, default=None,
                        help="future-latent prediction weight (or masked prediction weight in the "
                             "historical --jepa-mode masked control)")
    parser.add_argument("--vicreg-weight", type=float, default=None,
                        help="augmentation VICReg weight in --jepa-mode masked")
    parser.add_argument("--physical-weight", type=float, default=None,
                        help="future-latent physical-decoder objective weight")
    parser.add_argument("--collapse-weight", type=float, default=None,
                        help="patch-level variance/covariance objective weight")
    parser.add_argument("--num-heads", type=int, default=None,
                        help="Attention heads (default 8 -> head dim d_model/heads = 32). The "
                             "literature range for head dim is 64-128; 4 heads gives 64 at "
                             "IDENTICAL parameter count.")
    parser.add_argument("--jepa-ema-schedule", choices=("fixed", "cosine"), default=None,
                        help="Teacher decay schedule. 'cosine' is the BYOL ramp from "
                             "jepa_ema_decay to 1.0 and is the default; 'fixed' is the control.")
    parser.add_argument("--mask-ratio-time", type=float, default=None,
                        help="Nominal JEPA temporal mask fraction (default 0.5 -> ~0.6 realised). "
                             "MAE/data2vec use 0.75-0.8.")
    parser.add_argument("--jepa-ema-decay", type=float, default=None,
                        help="EMA teacher momentum (default 0.984095744256 at batch 1024; "
                             "example-time equivalent to 0.996 at batch 256).")
    parser.add_argument("--vicreg-proj-dim", type=int, default=None,
                        help="VICReg expander OUTPUT width (default 128); also the width of the "
                             "DxD covariance matrices, so memory grows quadratically. "
                             "VICReg/Barlow-Twins both find wider is better and recommend it "
                             "exceed the representation dim (ours is d_model=256).")
    parser.add_argument("--vicreg-proj-hidden", type=int, default=None,
                        help="VICReg expander HIDDEN width (default 256 = d_model). Separate from "
                             "--vicreg-proj-dim so the default stays the historical control.")
    parser.add_argument("--calibrate-objectives-at", type=int, default=None,
                        help="collect post-warmup objective gradients ending at this step and "
                             "resolve one frozen scalarization (future-JEPA default: 1000; "
                             "single-resolution masked-control default: 500; 0 disables)")
    parser.add_argument("--objective-calibration-batches", type=int, default=None,
                        help="number of consecutive post-warmup batches used by calibration "
                             "(default 50)")
    parser.add_argument("--objective-calibration-mode", choices=("report", "apply"), default=None,
                        help="'report' writes the recommendation and stops; 'apply' installs it "
                             "once after the calibration step, freezes it, and continues training "
                             "(real-run default: apply)")
    parser.add_argument("--target-jepa-gradient-share", type=float, default=None,
                        help="target future-JEPA gradient share (default 0.70; masked control 0.45)")
    parser.add_argument("--target-physical-gradient-share", type=float, default=None,
                        help="target physical-decoder gradient share (default 0.20; remainder is "
                             "patch collapse control)")
    parser.add_argument("--sampler-alpha", type=float, default=None,
                        help="temperature-sampler exponent: 1=proportional, 0=uniform-per-source, "
                             "0.25 is the default.")
    parser.add_argument("--sampler-max-dataset-share", type=float, default=None,
                        help="maximum ordinary-draw probability for one dataset (default 0.25; "
                             "raised automatically only when too few datasets make it infeasible).")
    parser.add_argument("--sampler-subject-alpha", type=float, default=None,
                        help="within-dataset subject-size exponent: 1=proportional, 0=uniform over "
                             "subjects, 0.5=square-root tempering (default).")
    parser.add_argument("--batch", type=int, default=None,
                        help="batch size for the temperature sampler (future-JEPA default 512; "
                             "historical single-resolution control default 1024)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="training DataLoader workers (default 12, profiled for a Ryzen 7900X + "
                             "RTX 4090; use 0 for in-process loading)")
    parser.add_argument("--mixed-sensor-batches", dest="homogeneous_sensor_batches",
                        action="store_false", default=None,
                        help="mix one- and two-sensor windows in each batch (slower compatibility arm)")
    parser.add_argument("--subset", action="store_true",
                        help="train on the tokenizer-ablation 3-rate-core subset (5 datasets, xrf_v2 "
                             "held out) instead of the full corpus. See ablation_subset.py.")
    parser.add_argument("--corpus", choices=("label_free", "expanded", "matched"),
                        default="label_free",
                        help="named Phase-A recipe. label_free=the label-free pretraining corpus "
                             "in data/pretraining/, disjoint from the labelled data the "
                             "comparator and classification head train on, so a downstream gain "
                             "is attributable and is the default; expanded=the active labelled "
                             "historical roster (14 sources since 2026-09-10; the frozen 18 is "
                             "EXPANDED_18_TRAIN_DATASETS); "
                             "matched=the frozen original 12-source corpus for technique-only "
                             "baseline comparisons (contains retired sources: needs --allow-retired)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="explicit train dataset list (overrides --corpus; incompatible with "
                             "--subset).")
    parser.add_argument("--allow-retired", action="store_true",
                        help="permit datasets in deployment_policy.RETIRED_TRAIN_DATASETS. Only "
                             "for reproducing a run that predates their 2026-09-10 retirement "
                             "(e.g. --corpus matched, or the --subset ablation).")
    parser.add_argument("--max-per-stream", type=int, default=None,
                        help="per-stream window cap (default: None=all; --subset defaults to the "
                             "ablation DEFAULT_CAP so train and metric-eval share one corpus).")
    parser.add_argument("--seed", type=int, default=None,
                        help="MODEL seed (init/augmentation/batch order). Vary this across replicates.")
    parser.add_argument("--compile", dest="compile_encoder", action="store_true", default=None,
                        help="torch.compile only the transformer core (not ragged text conditioning). "
                             "Enabled by default for CUDA Phase-A runs.")
    parser.add_argument("--no-compile", dest="compile_encoder", action="store_false",
                        help="disable the default CUDA transformer compilation")
    parser.add_argument("--data-seed", type=int, default=None,
                        help="DATA seed = the subject train/val split. Keep FIXED across all arms and "
                             "replicates so the split (and the metric harness) stays identical (#1).")
    args = parser.parse_args()

    cfg = PretrainConfig(
        device=args.device,
        frontend=args.frontend,
        jepa_mode=args.jepa_mode,
        token_granularity="sensor",
        # Future JEPA consumes a generic physical-time token grid. The filterbank arms use the
        # established joint-resolution collate; the single-span continuous arm has its own ordered
        # sub-frame analysis and therefore uses the one-second grid it was built for.
        multiresolution=args.jepa_mode == "future"
                        and args.frontend not in {"continuous", "multispan"},
        text_conditioning="factored",  # PAPER default (F8): factored role+sensor conditioning is the
                                       # committed arm; --text-conditioning per_channel is the ablation.
                                       # (The dataclass default stays per_channel for direct/test ctors.)
        objective_calibration_at=500,
        objective_calibration_mode="apply",
        objective_target_jepa_share=(0.70 if args.jepa_mode == "future" else 0.45),
        compile_encoder=args.device.startswith("cuda") and not args.smoke,
        train_datasets=_corpus_datasets(args.corpus),
        corpus_name=args.corpus,
        source_window_seconds=(PRETRAIN_WINDOW_SECONDS
                               if args.corpus == "label_free" else WINDOW_SECONDS),
    )
    if args.neutral_acquisition_text is not None:
        cfg.neutral_acquisition_text = bool(args.neutral_acquisition_text)
    if args.text_conditioning is not None:
        cfg.text_conditioning = args.text_conditioning
    if args.token_granularity is not None:
        cfg.token_granularity = args.token_granularity
    if args.multiresolution is not None:
        cfg.multiresolution = args.multiresolution
    if args.patch_seconds is not None:
        cfg.patch_seconds = args.patch_seconds
    if args.future_patch_durations is not None:
        cfg.future_patch_durations = tuple(args.future_patch_durations)
    if args.multispan_durations is not None:
        cfg.multispan_durations = tuple(args.multispan_durations)
    if args.frames_per_span is not None:
        cfg.frames_per_span = args.frames_per_span
    if args.future_context_fraction is not None:
        cfg.future_context_fraction = tuple(args.future_context_fraction)
    if args.future_teacher_top_layers is not None:
        cfg.future_teacher_top_layers = args.future_teacher_top_layers
    if args.future_predictor_dim is not None:
        cfg.future_predictor_dim = args.future_predictor_dim
    if args.future_predictor_layers is not None:
        cfg.future_predictor_layers = args.future_predictor_layers
    if args.future_predictor_heads is not None:
        cfg.future_predictor_heads = args.future_predictor_heads
    if args.descriptor_weight is not None:
        cfg.descriptor_weight = args.descriptor_weight
    cfg.descriptor_prediction = cfg.descriptor_weight > 0
    if args.rotation_p is not None:
        cfg.rotation_p = args.rotation_p
    if args.rotation_pairing is not None:
        cfg.rotation_pairing = args.rotation_pairing
    if args.rate_augmentation_p is not None:
        cfg.rate_augmentation_p = args.rate_augmentation_p
    if args.channel_dropout_p is not None:
        cfg.channel_dropout_p = args.channel_dropout_p
    if args.mae_weight is not None:
        cfg.mae_weight = args.mae_weight
    if args.gravity_p is not None:
        cfg.gravity_p = args.gravity_p
    if args.channel_text_phrase_p is not None:
        cfg.channel_text_phrase_p = args.channel_text_phrase_p
    if args.channel_text_dropout_p is not None:
        cfg.channel_text_dropout_p = args.channel_text_dropout_p
    if args.jitter_p is not None:
        cfg.jitter_p = args.jitter_p
    if args.scale_p is not None:
        cfg.scale_p = args.scale_p
    if args.num_layers is not None:
        cfg.num_layers = args.num_layers
    if args.trunk is not None:
        cfg.trunk = args.trunk
    if args.d_model is not None:
        cfg.d_model = args.d_model
    if args.dim_feedforward is not None:
        cfg.dim_feedforward = args.dim_feedforward
    if args.retrieval_vicreg_fraction is not None:
        cfg.retrieval_vicreg_fraction = args.retrieval_vicreg_fraction
    if args.val_every is not None:
        cfg.val_every = args.val_every
    if args.selection_every is not None:
        cfg.selection_every = args.selection_every
    if args.jepa_weight is not None:
        cfg.jepa_weight = args.jepa_weight
    if args.vicreg_weight is not None:
        cfg.vicreg_weight = args.vicreg_weight
    if args.physical_weight is not None:
        cfg.physical_weight = args.physical_weight
    if args.collapse_weight is not None:
        cfg.collapse_weight = args.collapse_weight
    if args.num_heads is not None:
        cfg.num_heads = args.num_heads
    if args.jepa_ema_schedule is not None:
        cfg.jepa_ema_schedule = args.jepa_ema_schedule
    if args.mask_ratio_time is not None:
        cfg.mask_ratio_time = args.mask_ratio_time
    if args.jepa_ema_decay is not None:
        cfg.jepa_ema_decay = args.jepa_ema_decay
    if args.vicreg_proj_dim is not None:
        cfg.vicreg_proj_dim = args.vicreg_proj_dim
    if args.vicreg_proj_hidden is not None:
        cfg.vicreg_proj_hidden = args.vicreg_proj_hidden
    if args.calibrate_objectives_at is not None:
        cfg.objective_calibration_at = args.calibrate_objectives_at
    if args.objective_calibration_batches is not None:
        cfg.objective_calibration_batches = args.objective_calibration_batches
    if args.objective_calibration_mode is not None:
        cfg.objective_calibration_mode = args.objective_calibration_mode
    if args.target_jepa_gradient_share is not None:
        cfg.objective_target_jepa_share = args.target_jepa_gradient_share
    if args.target_physical_gradient_share is not None:
        cfg.objective_target_physical_share = args.target_physical_gradient_share
    if args.sampler_alpha is not None:
        cfg.sampler_alpha = args.sampler_alpha
    if args.sampler_max_dataset_share is not None:
        cfg.sampler_max_dataset_share = args.sampler_max_dataset_share
    if args.sampler_subject_alpha is not None:
        cfg.sampler_subject_alpha = args.sampler_subject_alpha
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.homogeneous_sensor_batches is not None:
        cfg.homogeneous_sensor_batches = args.homogeneous_sensor_batches
    if args.seed is not None:
        cfg.seed = args.seed
    if args.compile_encoder is not None:
        cfg.compile_encoder = args.compile_encoder
    if args.data_seed is not None:
        cfg.data_seed = args.data_seed
    if args.datasets is not None and args.subset:
        parser.error("--datasets and --subset are mutually exclusive")
    if args.datasets is not None:
        cfg.train_datasets = tuple(args.datasets)
        from data.scripts.curate.deployment_policy import LABEL_FREE_PRETRAIN_DATASETS
        requested = set(cfg.train_datasets)
        label_free = set(LABEL_FREE_PRETRAIN_DATASETS)
        if requested.issubset(label_free):
            cfg.corpus_name = "label_free_subset"
            cfg.source_window_seconds = PRETRAIN_WINDOW_SECONDS
        elif requested.isdisjoint(label_free):
            cfg.corpus_name = "custom"
            cfg.source_window_seconds = WINDOW_SECONDS
        else:
            parser.error(
                "--datasets cannot mix eight-second label-free grids with six-second labelled "
                "grids; choose one source-window contract per run"
            )
    elif args.subset:
        from training.tokenizer.ablation_subset import SUBSET_TRAIN_DATASETS, DEFAULT_CAP
        cfg.train_datasets = SUBSET_TRAIN_DATASETS
        cfg.corpus_name = "subset"
        cfg.source_window_seconds = WINDOW_SECONDS
        # Apply the SAME per-stream cap the metric harness uses (build_subset_index(cap=DEFAULT_CAP)),
        # so TRAIN and EVAL share one corpus definition (audit 2026-07-23 #3: --subset previously left
        # max_per_stream=None -> trained on ~94k windows while metrics used the 10k cap).
        if args.max_per_stream is None:
            cfg.max_per_stream = DEFAULT_CAP
    if args.max_per_stream is not None:
        cfg.max_per_stream = args.max_per_stream
    # The single gate between every roster path (--corpus, --datasets, --subset) and training.
    from data.scripts.curate.deployment_policy import assert_no_retired_sources
    assert_no_retired_sources(cfg.train_datasets or TRAIN_DATASETS, allow=args.allow_retired)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.warmup_steps is not None:
        cfg.warmup_steps = args.warmup_steps
    if args.grad_clip is not None:
        cfg.grad_clip = args.grad_clip
    if not math.isfinite(cfg.source_window_seconds) or cfg.source_window_seconds <= 0:
        parser.error("source-window duration must be finite and positive")
    if cfg.frames_per_span <= 0:
        parser.error("--frames-per-span must be a positive integer")
    if cfg.frontend == "multispan" and (
        any(not math.isfinite(value) or value <= 0 for value in cfg.multispan_durations)
        or tuple(sorted(set(cfg.multispan_durations))) != cfg.multispan_durations
    ):
        parser.error("multi-span durations must be unique, finite, positive, and increasing")

    # Resolve time position at the fastest frontend stride. For multi-span this is twice the
    # shortest T/frames_per_span interval (the temporal Nyquist period).
    cfg.rope_min_period = frontend_rope_min_period(cfg)

    # Derive schedule defaults from the real token count. Both arms see the same 7.68 million
    # source windows as the audited 512 x 15,000 reference without exceeding the transformer-token
    # budget. Explicit oversized batches fail rather than silently dropping a resolution.
    if cfg.multiresolution:
        tokens_per_window = future_tokens_per_window(cfg)
        max_batch = max(MAX_BATCH_TOKENS // tokens_per_window, 1)
        default_batch = batch_under_token_budget(tokens_per_window)
        if args.batch is None:
            cfg.batch_size = default_batch
        if args.steps is None:
            cfg.steps = round(15_000 * 512 / cfg.batch_size)
        batch_ratio = cfg.batch_size / 512.0
        if args.lr is None:
            cfg.lr = 4.242640687119285e-4 * math.sqrt(batch_ratio)
        if args.weight_decay is None:
            cfg.weight_decay = 0.07071067811865475 * math.sqrt(batch_ratio)
        if args.warmup_steps is None:
            cfg.warmup_steps = round(500 / batch_ratio)
        if args.calibrate_objectives_at is None:
            cfg.objective_calibration_at = round(1_000 / batch_ratio)
        if args.jepa_ema_decay is None:
            cfg.jepa_ema_decay = 0.992016 ** batch_ratio
        if cfg.batch_size > max_batch:
            parser.error(
                f"multi-resolution durations {future_resolution_durations(cfg)} emit "
                f"{tokens_per_window} tokens/window and require --batch <= {max_batch} "
                f"under the {MAX_BATCH_TOKENS}-token budget"
            )
    elif cfg.frontend == "multispan":
        tokens_per_window = future_tokens_per_window(cfg)
        max_batch = max(MAX_BATCH_TOKENS // tokens_per_window, 1)
        default_batch = batch_under_token_budget(tokens_per_window)
        if args.batch is None:
            cfg.batch_size = default_batch
        if args.steps is None:
            cfg.steps = round(15_000 * 512 / cfg.batch_size)
        batch_ratio = cfg.batch_size / 512.0
        if args.lr is None:
            cfg.lr = 4.242640687119285e-4 * math.sqrt(batch_ratio)
        if args.weight_decay is None:
            cfg.weight_decay = 0.07071067811865475 * math.sqrt(batch_ratio)
        if args.warmup_steps is None:
            cfg.warmup_steps = round(500 / batch_ratio)
        if args.calibrate_objectives_at is None:
            cfg.objective_calibration_at = round(1_000 / batch_ratio)
        if args.jepa_ema_decay is None:
            # Preserve the reference teacher half-life in examples.
            cfg.jepa_ema_decay = 0.992016 ** batch_ratio
        if cfg.batch_size > max_batch:
            parser.error(
                f"--frontend multispan with spans {cfg.multispan_durations} emits "
                f"{tokens_per_window} tokens/window and requires --batch <= {max_batch} "
                f"under the {MAX_BATCH_TOKENS}-token budget"
            )
    if cfg.jepa_mode == "future":
        if cfg.token_granularity != "sensor":
            parser.error("--jepa-mode future currently requires sensor-granularity tokens")
        if cfg.descriptor_weight > 0 or cfg.mae_weight > 0:
            parser.error(
                "descriptor reconstruction and legacy masked MAE are not part of future JEPA"
            )
        durations = future_resolution_durations(cfg)
        if any(duration <= 0 for duration in durations):
            parser.error("future JEPA patch durations must be positive")
        if cfg.multiresolution and len(durations) < 2:
            parser.error("multi-resolution future JEPA requires at least two patch durations")
        if tuple(sorted(set(durations))) != durations:
            parser.error("future patch durations must be unique and strictly increasing")
        context_lo, context_hi = cfg.future_context_fraction
        if not 0 < context_lo <= context_hi < 1:
            parser.error("future context fractions must satisfy 0 < MIN <= MAX < 1")
        horizon_bins = tuple(cfg.future_horizon_bins_seconds)
        if (not horizon_bins
                or any(lo < 0 or hi <= lo for lo, hi in horizon_bins)
                or any(left[1] > right[0]
                       for left, right in zip(horizon_bins, horizon_bins[1:]))):
            parser.error(
                "future horizon bins must be ordered, non-overlapping [MIN,MAX) intervals"
            )
        if cfg.future_teacher_top_layers <= 0 or cfg.future_teacher_top_layers > cfg.num_layers:
            parser.error("future teacher top-layer count must be in [1, num_layers]")
        if (cfg.future_predictor_dim <= 0 or cfg.future_predictor_layers <= 0
                or cfg.future_predictor_heads <= 0
                or cfg.future_predictor_dim % cfg.future_predictor_heads):
            parser.error("future predictor width must be positive and divisible by its head count")
        if cfg.jepa_weight <= 0 or cfg.collapse_weight <= 0 or cfg.physical_weight < 0:
            parser.error("future/collapse weights must be positive and physical weight nonnegative")
        if not 0 < cfg.objective_target_physical_share < 1:
            parser.error("target physical gradient share must be in (0,1)")
        if cfg.objective_target_jepa_share + cfg.objective_target_physical_share >= 1:
            parser.error("future and physical target shares must leave a positive collapse share")
    # The paper recipe calibrates by default. Tiny smoke runs cannot reach step 500, and the
    # VICReg-only control has no second objective to balance; disable only when the user did not
    # explicitly request a calibration experiment.
    if args.smoke and args.calibrate_objectives_at is None:
        cfg.objective_calibration_at = 0
    if cfg.jepa_weight <= 0 and args.calibrate_objectives_at is None:
        cfg.objective_calibration_at = 0
    # Architecture/masking validation runs BEFORE --smoke so a smoke rejects the same configs a
    # real run does. An unvalidated --mask-ratio-time -1 previously trained to completion, leaving
    # 75% of windows with no JEPA mask at all -- a silently different experiment, not an error.
    if cfg.num_heads <= 0 or cfg.d_model % cfg.num_heads:
        parser.error(f"--num-heads must be positive and divide d_model={cfg.d_model}")
    if cfg.jepa_mode == "masked" and not 0 < cfg.mask_ratio_time < 1:
        parser.error("--mask-ratio-time must be in (0,1)")
    if cfg.vicreg_proj_dim <= 0 or cfg.vicreg_proj_hidden <= 0:
        parser.error("expander widths must be positive")
    if not 0.0 <= cfg.retrieval_vicreg_fraction <= 1.0:
        parser.error("retrieval_vicreg_fraction must be in [0,1]")
    for name in ("rotation_p", "rate_augmentation_p", "channel_dropout_p", "jitter_p", "scale_p",
                 "gravity_p", "channel_text_phrase_p", "channel_text_dropout_p"):
        if not 0.0 <= float(getattr(cfg, name)) <= 1.0:
            parser.error(f"{name} must be in [0,1]")
    if cfg.patch_seconds <= 0:
        parser.error("--patch-seconds must be positive")
    if cfg.frontend in {"continuous", "multispan"}:
        if cfg.multiresolution or abs(cfg.patch_seconds - 1.0) > 1e-6:
            parser.error(
                "continuous frontends require a single one-second input partition; multispan "
                "kernels create their own output resolution grid"
            )
        if cfg.mae_weight > 0:
            parser.error(
                "--mae-weight is not defined for the continuous frontend's structured analysis; "
                "use the reference JEPA + VICReg objective"
            )
    if cfg.frontend == "multispan" and cfg.jepa_mode != "future":
        parser.error("the multi-span frontend is currently defined for future JEPA only")
    if cfg.selection_every <= 0:
        parser.error("selection_every must be positive")
    if cfg.vicreg_proj_dim > 2048:
        # VICReg materialises two dense DxD covariance matrices per step (losses_repr.vicreg), so
        # fp32 memory is 8*D^2 bytes before autograd: 128MiB at 4096, 512MiB at 8192.
        print(f"[warn] vicreg_proj_dim={cfg.vicreg_proj_dim} -> VICReg covariance matrices "
              f"alone need {8 * cfg.vicreg_proj_dim ** 2 / 2**20:.0f} MiB", flush=True)
    if args.smoke:
        # Preserve every experimental override and change only resource/budget knobs. Rebuilding
        # PretrainConfig field-by-field repeatedly dropped newly added objective settings, so a
        # passing smoke could exercise a different arm than the requested run.
        # d_model shrinks to 64, so the requested head count only survives if it still divides it.
        # Silently pinning num_heads=4 here meant a smoke "passed" for an arm nobody requested.
        if 64 % cfg.num_heads:
            parser.error(f"--smoke uses d_model=64, which --num-heads {cfg.num_heads} "
                         f"does not divide; smoke cannot exercise this arm")
        cfg = replace(
            cfg,
            d_model=64, num_layers=2, dim_feedforward=128,
            batch_size=32,
            steps=args.steps if args.steps is not None else 10,
            warmup_steps=(args.warmup_steps if args.warmup_steps is not None
                          else min(2, max((args.steps if args.steps is not None else 10) - 1, 0))),
            calib_batches=3,
            val_every=max(args.steps if args.steps is not None else 10, 5), val_per_label=10,
            num_workers=0, max_per_stream=200, selection_datasets=(),
        )
    if cfg.sampler_alpha < 0:
        parser.error("--sampler-alpha must be nonnegative")
    if not 0 < cfg.sampler_max_dataset_share <= 1:
        parser.error("--sampler-max-dataset-share must be in (0,1]")
    if not 0 <= cfg.sampler_subject_alpha <= 1:
        parser.error("--sampler-subject-alpha must be in [0,1]")
    for field in ("jepa_weight", "frontend_reg_weight", "descriptor_weight"):
        if float(getattr(cfg, field)) < 0:
            parser.error(f"--{field.replace('_', '-')} must be nonnegative")
    if cfg.jepa_mode == "masked" and cfg.vicreg_weight <= 0:
        parser.error("--vicreg-weight must be positive")
    if cfg.steps <= 0 or cfg.batch_size <= 0:
        parser.error("--steps and --batch must be positive")
    if cfg.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")
    if cfg.max_per_stream is not None and cfg.max_per_stream <= 0:
        parser.error("--max-per-stream must be positive when provided")
    if not 0 <= cfg.jepa_ema_decay < 1:
        parser.error("--jepa-ema-decay must be in [0,1)")
    if cfg.lr <= 0 or cfg.grad_clip <= 0:
        parser.error("learning rate and gradient clipping threshold must be positive")
    if cfg.weight_decay < 0:
        parser.error("--weight-decay must be nonnegative")
    if not 0 <= cfg.warmup_steps < cfg.steps:
        parser.error("--warmup-steps must be nonnegative and smaller than --steps")
    if cfg.objective_calibration_at < 0 or cfg.objective_calibration_batches <= 0:
        parser.error("objective calibration step must be nonnegative and batch count positive")
    if cfg.objective_calibration_at:
        first_calibration_step = (
            cfg.objective_calibration_at - cfg.objective_calibration_batches + 1
        )
        if cfg.objective_calibration_at > cfg.steps:
            parser.error("--calibrate-objectives-at cannot exceed --steps")
        if first_calibration_step <= cfg.warmup_steps:
            parser.error("objective calibration batches must all occur after warmup")
        if cfg.jepa_weight <= 0:
            parser.error("objective calibration requires an active JEPA objective")
    if not 0 < cfg.objective_target_jepa_share < 1:
        parser.error("--target-jepa-gradient-share must be in (0,1)")
    device = torch.device(cfg.device)
    if device.type == "cuda":
        # TF32 for CUDA matrix products inside the FP32 filterbank/diagnostic islands. The
        # transformer already runs FP16 under autocast; validation kNN/ridge stays CPU FP32.
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.manual_seed(cfg.seed)
    prepare_output_dir(
        args.out, force=args.force, smoke=args.smoke, resume=args.resume is not None,
    )
    source_provenance = capture_source_provenance(args.out, write=False)
    runtime_provenance = capture_runtime_provenance(device)
    precision = "fp16-mixed" if device.type == "cuda" else "fp32"
    print(f"frontend={cfg.frontend} multiresolution={cfg.multiresolution} "
          f"precision={precision}", flush=True)
    # NB: run_config.json is written AFTER resume validation (below), so a rejected --resume can't
    # overwrite the metadata with the bad attempted config (audit 2026-07-23 #6).

    # ------------------------------------------------------------------ data
    # DATA seed (fixed subject split), NOT the model seed, so the split is identical across replicates
    # and reconstructable by the metric harness (#1).
    index = CorpusIndex(max_per_stream=cfg.max_per_stream, seed=cfg.data_seed,
                        datasets=cfg.train_datasets or TRAIN_DATASETS)
    if cfg.corpus_name in {"label_free", "label_free_subset"}:
        validate_source_window_contract(index, cfg.source_window_seconds)
    corpus_fp = corpus_fingerprint(index)
    print(f"corpus: {index.summary()}  (datasets={sorted(cfg.train_datasets or TRAIN_DATASETS)})",
          flush=True)
    if cfg.selection_datasets:
        # Selecting on a trained source turns held-out transfer into a training probe. Assert it.
        assert_selection_roster_is_untrained(cfg.train_datasets or TRAIN_DATASETS)
        print(f"selection: {list(cfg.selection_datasets)} every {cfg.selection_every} steps "
              f"(held out; posture canaries reported per dataset)", flush=True)
    augmentation_cfg = AugmentationConfig.phase_a(
        rotation_p=cfg.rotation_p,
        rate_p=cfg.rate_augmentation_p,
        channel_dropout_p=cfg.channel_dropout_p,
        jitter_p=cfg.jitter_p,
        scale_p=cfg.scale_p,
        gravity_p=cfg.gravity_p,
        channel_text_phrase_p=cfg.channel_text_phrase_p,
        channel_text_dropout_p=cfg.channel_text_dropout_p,
    )
    two_view = cfg.jepa_mode == "masked"
    train_ds = PretrainDataset(
        index, index.train, augment=True, two_view=two_view,
        augmentation_config=augmentation_cfg, rotation_pairing=cfg.rotation_pairing,
        neutral_acquisition_text=cfg.neutral_acquisition_text,
    )
    calibration_ds = PretrainDataset(
        index, index.train, augment=True, two_view=False,
        augmentation_config=augmentation_cfg, rotation_pairing=cfg.rotation_pairing,
        neutral_acquisition_text=cfg.neutral_acquisition_text,
    )
    # Preselecting keeps evaluation cheap. The helper covers every label/stream cell before filling
    # additional slots, preventing a large source from monopolizing a common label's cap.
    semantic_label_ids = {
        label_id for label, label_id in index.label_ids.items()
        if label not in NON_SEMANTIC_LABELS
    }
    val_keys = stratified_eval_subset(
        index.val, cfg.val_per_label, cfg.data_seed,
        allowed_labels=semantic_label_ids,
    )
    has_internal_classification_probe = bool(val_keys)
    val_ds = PretrainDataset(index, val_keys, augment=False,
                             neutral_acquisition_text=cfg.neutral_acquisition_text)
    fixed_train_durations = (cfg.future_patch_durations if cfg.jepa_mode == "future" else None)
    train_collate = (MultiResolutionCollate(
        short_choices=cfg.short_patch_choices, long_choices=cfg.long_patch_choices,
        fixed_patch_seconds=fixed_train_durations,
        min_resolution_ratio=cfg.min_resolution_ratio, seed=cfg.seed,
        dft_size=cfg.dft_size,
        two_view=two_view,
    ) if cfg.multiresolution
                     else MultiScaleCollate(fixed_patch_seconds=cfg.patch_seconds,
                                            seed=cfg.seed, two_view=True,
                                            dft_size=cfg.dft_size))
    calibration_collate = (MultiResolutionCollate(
        short_choices=cfg.short_patch_choices, long_choices=cfg.long_patch_choices,
        fixed_patch_seconds=fixed_train_durations,
        min_resolution_ratio=cfg.min_resolution_ratio, seed=cfg.seed,
        dft_size=cfg.dft_size,
        two_view=False,
    ) if cfg.multiresolution else MultiScaleCollate(fixed_patch_seconds=cfg.patch_seconds,
                                                     seed=cfg.seed, two_view=False,
                                                     dft_size=cfg.dft_size))
    loader_kwargs = dict(
        collate_fn=train_collate, num_workers=cfg.num_workers, worker_init_fn=_seed_worker,
        persistent_workers=cfg.num_workers > 0, pin_memory=device.type == "cuda")
    sensor_batch_groups = (
        [len(modalities_present(index.refs[key.stream_i].mask)) for key in index.train]
        if cfg.homogeneous_sensor_batches else None
    )
    temperature_sampler = TemperatureSampler(
        index.train, index.stream_datasets,
        num_samples=cfg.steps * cfg.batch_size,
        alpha=cfg.sampler_alpha, seed=cfg.seed,
        batch_size=cfg.batch_size,
        subject_ids=index.train_subject_ids,
        subject_alpha=cfg.sampler_subject_alpha,
        max_dataset_share=cfg.sampler_max_dataset_share,
        batch_group_ids=sensor_batch_groups,
    )
    calibration_sampler = TemperatureSampler(
        index.train, index.stream_datasets,
        num_samples=cfg.calib_batches * cfg.batch_size,
        alpha=cfg.sampler_alpha, seed=cfg.seed - 1,
        batch_size=cfg.batch_size,
        subject_ids=index.train_subject_ids,
        subject_alpha=cfg.sampler_subject_alpha,
        max_dataset_share=cfg.sampler_max_dataset_share,
        batch_group_ids=sensor_batch_groups,
    )
    shares = ", ".join(
        f"{dataset}={share:.1%}"
        for dataset, share in sorted(
            temperature_sampler.dataset_probabilities.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    print(
        f"sampler: dataset alpha={cfg.sampler_alpha:g}, "
        f"max_share={cfg.sampler_max_dataset_share:.1%}, "
        f"subject alpha={cfg.sampler_subject_alpha:g} ({shares})",
        flush=True,
    )
    train_loader = DataLoader(
        train_ds, sampler=temperature_sampler,
        batch_size=cfg.batch_size, drop_last=True, **loader_kwargs)
    calibration_loader = DataLoader(
        calibration_ds, sampler=calibration_sampler,
        batch_size=cfg.batch_size, drop_last=True,
        collate_fn=calibration_collate, num_workers=cfg.num_workers,
        worker_init_fn=_seed_worker, persistent_workers=cfg.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    target_calibration_loader = None
    if cfg.frontend == "multispan" and cfg.physical_weight > 0:
        # The student frontend consumes a one-second input partition, but the frozen physical
        # target is evaluated at every emitted span. Fit its normalization on all those durations
        # using the same sampled corpus instead of applying one-second statistics to every target.
        target_calibration_sampler = TemperatureSampler(
            index.train, index.stream_datasets,
            num_samples=cfg.calib_batches * cfg.batch_size,
            alpha=cfg.sampler_alpha, seed=cfg.seed - 1,
            batch_size=cfg.batch_size, subject_ids=index.train_subject_ids,
            subject_alpha=cfg.sampler_subject_alpha,
            max_dataset_share=cfg.sampler_max_dataset_share,
            batch_group_ids=sensor_batch_groups,
        )
        target_calibration_loader = DataLoader(
            calibration_ds, sampler=target_calibration_sampler,
            batch_size=cfg.batch_size, drop_last=True,
            collate_fn=MultiResolutionCollate(
                fixed_patch_seconds=cfg.multispan_durations,
                min_resolution_ratio=cfg.min_resolution_ratio,
                dft_size=cfg.dft_size,
            ),
            num_workers=cfg.num_workers, worker_init_fn=_seed_worker,
            persistent_workers=cfg.num_workers > 0, pin_memory=device.type == "cuda",
        )
    # Validation uses deterministic fixed patch durations and no augmentation.
    val_workers = min(6, cfg.num_workers)
    val_collate = (
        MultiResolutionCollate(fixed_patch_seconds=(
            cfg.future_patch_durations if cfg.jepa_mode == "future" else cfg.val_resolution_pair
        ),
                               min_resolution_ratio=cfg.min_resolution_ratio,
                               dft_size=cfg.dft_size)
        if cfg.multiresolution else
        MultiScaleCollate(fixed_patch_seconds=cfg.patch_seconds, dft_size=cfg.dft_size)
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=val_collate,
                            num_workers=val_workers, persistent_workers=val_workers > 0,
                            pin_memory=device.type == "cuda")
    # Fixed generator so the kNN SUPPORT bank is deterministic across evals AND identical between the
    # two arms (audit 2026-07-23 #5: shuffle=True with no generator drew a different support set per
    # arm/eval, so matched training seeds did NOT give matched validation). Seeded from DATA seed (the
    # split), not the model seed, and RESET before each val (#4) so the support is identical at every
    # evaluation and across replicates — not just across arms.
    # The support bank only ever needs the labels the val queries actually carry, and val is now a
    # fixed set, so both subsets can be chosen once here instead of re-scanned every evaluation.
    from collections import Counter as _Counter
    val_label_totals = dict(_Counter(k.label_id for k in val_keys))
    _val_label_set = set(val_label_totals)
    train_keys = stratified_eval_subset(index.train, cfg.val_per_label, cfg.data_seed,
                                        allowed_labels=_val_label_set)
    train_label_totals = dict(_Counter(k.label_id for k in train_keys))
    print(f"eval subsets: val {len(index.val):,} -> {len(val_keys):,} rows · "
          f"support {len(index.train):,} -> {len(train_keys):,} rows "
          f"({len(_val_label_set)} labels)", flush=True)

    train_eval_gen = torch.Generator().manual_seed(cfg.data_seed)
    train_eval_loader = DataLoader(
        PretrainDataset(index, train_keys, augment=False), batch_size=256,
        shuffle=False, collate_fn=val_collate, generator=train_eval_gen,
        num_workers=val_workers, persistent_workers=val_workers > 0,
        pin_memory=device.type == "cuda", worker_init_fn=_seed_worker,
    )

    # ------------------------------------------------------------------ model
    model = PipelineAModel(cfg).to(device)
    # Calibrate the selected encoder frontend and the independent physical-target analyzer. The
    # latter remains fixed even when the selected frontend is learnable or continuous.
    fe = model.encoder.filterbank
    target_fe = model.physical_target_analyzer
    print(f"calibrating frontend norm on {cfg.calib_batches} batches ...", flush=True)
    fe.reset_norm_accumulator()
    if target_fe is not None and cfg.frontend != "fixed":
        target_fe.reset_norm_accumulator()
    for b in calibration_loader:
        calibration_args = (
            b["patches"].to(device), b["rates"].to(device), b["patch_len"].to(device),
        )
        calibration_kwargs = {
            "patch_mask": b["patch_padding_mask"].to(device),
            "channel_mask": b["channel_mask"].to(device),
            "source_rate_hz": b.get("source_rates", b["rates"]).to(device),
        }
        fe.accumulate_norm_stats(*calibration_args, **calibration_kwargs)
        if target_fe is not None and cfg.frontend not in {"fixed", "multispan"}:
            target_fe.accumulate_norm_stats(*calibration_args, **calibration_kwargs)
    fe.finalize_norm_stats()
    if target_fe is not None:
        if cfg.frontend == "fixed":
            # Identical fixed analyzers can share the already measured calibration exactly.
            target_fe.load_state_dict(fe.state_dict())
        elif cfg.frontend == "multispan":
            assert target_calibration_loader is not None
            for b in target_calibration_loader:
                target_fe.accumulate_norm_stats(
                    b["patches"].to(device), b["rates"].to(device), b["patch_len"].to(device),
                    patch_mask=b["patch_padding_mask"].to(device),
                    channel_mask=b["channel_mask"].to(device),
                    source_rate_hz=b.get("source_rates", b["rates"]).to(device),
                )
            target_fe.finalize_norm_stats()
        else:
            target_fe.finalize_norm_stats()
    # Drop the one-view calibration iterator and its mmap handles before training workers start.
    del calibration_loader, calibration_sampler, calibration_ds, target_calibration_loader

    jepa_teacher = None
    if cfg.jepa_weight > 0:
        jepa_teacher = copy.deepcopy(model.encoder).to(device).eval()
        for parameter in jepa_teacher.parameters():
            parameter.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {n_params / 1e6:.2f}M trainable params · device={device}", flush=True)

    # Label-text prototypes for the live ConSE-style zero-shot probe (built once, frozen LM).
    label_protos = (
        label_text_prototypes(model, index.label_ids)
        if has_internal_classification_probe else None
    )

    adaptive_ids = {id(parameter) for parameter in fe.adaptation_parameters()}
    adaptive_params, base_params = [], []
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in adaptive_ids:
            adaptive_params.append(parameter)
        else:
            base_params.append(parameter)
    param_groups = [{"params": base_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    if adaptive_params:
        # Explicit physical regularization replaces AdamW's logit-space decay. In particular,
        # weight decay would pull the residual gate logit toward zero, i.e. gate=0.5, not fixed=0.
        param_groups.append({"params": adaptive_params, "lr": cfg.lr * cfg.frontend_lr_scale,
                             "weight_decay": 0.0})
    # CUDA's fused implementation is state/checkpoint compatible and cut the measured optimizer +
    # clip + EMA stage from 5.8 ms to 2.0 ms on the RTX 4090. Keep the ordinary implementation on CPU.
    opt = torch.optim.AdamW(param_groups, fused=device.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(cfg.warmup_steps, 1), 1.0)
        * 0.5 * (1 + np.cos(np.pi * min(s / cfg.steps, 1.0))),
    )
    # The 1k-step RTX 4090 pilot settled at 2^14 after two harmless startup overflows from PyTorch's
    # 2^16 default. Start there to avoid throwing away those first optimizer updates while retaining
    # dynamic growth/backoff for any later change in gradient range.
    scaler = torch.amp.GradScaler(
        enabled=device.type == "cuda", init_scale=16_384.0,
    )
    def ema_decay_at(step: int) -> float:
        """BYOL cosine ramp from cfg.jepa_ema_decay to 1.0, or the fixed value.

        Reaches exactly 1.0 at ``step == cfg.steps``; ``update_ema_encoder`` accepts that.
        """
        if cfg.jepa_ema_schedule != "cosine":
            return cfg.jepa_ema_decay
        progress = min(max(step / max(cfg.steps, 1), 0.0), 1.0)
        return 1.0 - (1.0 - cfg.jepa_ema_decay) * (math.cos(math.pi * progress) + 1.0) / 2.0

    log_path = args.out / "log.jsonl"
    best_ba = -1.0
    latest_selection_ba: float | None = None
    latest_selection_step: int | None = None
    t0 = time.time()
    calibration_samples: list[dict[str, dict[str, float]]] = []
    calibration_steps: list[int] = []
    calibration_source_counts: dict[str, int] = {}
    calibration_completed = False

    def checkpoint(name: str, step: int, val_ba: float):
        import random as _stdrandom
        torch.save({
            "encoder": model.encoder.state_dict(),
            "heads": {name: head.state_dict()
                      for name, head in model.pretraining_heads().items()},
            "physical_target_analyzer": (
                model.physical_target_analyzer.state_dict()
                if model.physical_target_analyzer is not None else None
            ),
            "config": asdict(cfg),
            "jepa_teacher": (jepa_teacher.state_dict() if jepa_teacher is not None else None),
            "label_ids": index.label_ids,
            "step": step, "val_ba": val_ba,
            "selection_metric": ("development_transfer_knn_ba"
                                 if cfg.selection_datasets else "val_knn_label_stream_ba"),
            "selection_value": latest_selection_ba,
            "selection_step": latest_selection_step,
            "selection_datasets": list(cfg.selection_datasets),
            "best_ba": max(best_ba, latest_selection_ba if latest_selection_ba is not None else -1.0),
            "git": source_provenance["git"],
            "source_provenance": source_provenance,
            "runtime_provenance": runtime_provenance,
            "corpus": index.summary(),
            "corpus_fingerprint": corpus_fp,   # which corpus produced this (F5)
            # Full restart state so a killed run resumes without silently diverging (F5).
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": (torch.cuda.get_rng_state_all() if device.type == "cuda" else None),
                    "numpy": np.random.get_state(),
                    "python": _stdrandom.getstate()},
        }, args.out / name)

    # F5b: warm-resume — restore weights + optimizer/scheduler/scaler/RNG + step, then continue the
    # REMAINING steps. Not bit-exact (the sampler re-draws a fresh epoch for the remaining steps), but
    # training continues correctly from the saved state rather than restarting from scratch.
    start_step = 0
    if args.resume:
        rk = torch.load(args.resume, map_location=device, weights_only=False)
        saved_cfg = rk.get("config", {})
        saved_step = int(rk["step"])
        # One-time calibration legitimately changes these scalar coefficients after the original
        # command is parsed. They are immutable checkpoint trajectory state on resume; all other
        # structural and user settings remain strictly validated below.
        explicit_resume_fields = {
            key for key, value in {
                "objective_calibration_at": args.calibrate_objectives_at,
                "objective_calibration_batches": args.objective_calibration_batches,
                "objective_target_jepa_share": args.target_jepa_gradient_share,
                "objective_target_physical_share": args.target_physical_gradient_share,
                "objective_calibration_mode": args.objective_calibration_mode,
                "jepa_weight": args.jepa_weight,
                "physical_weight": args.physical_weight,
                "collapse_weight": args.collapse_weight,
                "vicreg_weight": args.vicreg_weight,
            }.items() if value is not None
        }
        hydrate_calibrated_objective_weights(
            cfg, saved_cfg, saved_step, explicit_fields=explicit_resume_fields,
        )
        # A faithful resume must reproduce the SAME optimization trajectory, so validate the ENTIRE
        # serialized config against the checkpoint — NOT a hand-listed subset. Only knobs that touch
        # neither the training trajectory
        # NOR the saved model's meaning may differ — runtime + eval-cadence. (steps stays checked: it
        # rescales the cosine LR schedule, so a faithful resume passes the same --steps.)
        _RESUME_RUNTIME_ONLY = {"device", "num_workers", "val_every", "val_per_label", "knn_k"}

        def _norm(v):
            if isinstance(v, (list, tuple)):
                return [_norm(item) for item in v]
            if isinstance(v, dict):
                return {key: _norm(value) for key, value in v.items()}
            return v
        cur_cfg = asdict(cfg)
        missing_trajectory = sorted(
            set(cur_cfg) - set(saved_cfg) - _RESUME_RUNTIME_ONLY
        )
        if missing_trajectory:
            raise ValueError(
                "resume checkpoint predates trajectory-defining configuration fields "
                f"{missing_trajectory}; start a new attributable run instead of silently applying "
                "today's defaults to an older optimization trajectory"
            )
        for key in sorted(set(cur_cfg) | set(saved_cfg)):
            if key in _RESUME_RUNTIME_ONLY:
                continue
            saved = saved_cfg.get(key)
            cur = cur_cfg.get(key)
            # train_datasets round-trips through asdict as a list; compare order-insensitively as sets
            if key == "train_datasets" and saved is not None and cur is not None:
                if set(saved) != set(cur):
                    raise ValueError(f"resume mismatch for {key}: checkpoint={saved!r}, requested={cur!r}")
            elif _norm(saved) != _norm(cur):
                raise ValueError(
                    f"resume configuration mismatch for {key}: checkpoint={saved!r}, requested={cur!r} "
                    f"— a resume must reproduce the run; only {sorted(_RESUME_RUNTIME_ONLY)} may differ.")
        saved_fp = rk.get("corpus_fingerprint")
        if saved_fp is not None and saved_fp != corpus_fp:
            raise ValueError(
                f"resume corpus fingerprint mismatch: checkpoint={saved_fp}, "
                f"current={corpus_fp} — the corpus/cap/seed changed since the run started.")
        saved_source = rk.get("source_provenance")
        if saved_source is not None:
            if saved_source.get("patch_sha256") != source_provenance.get("patch_sha256"):
                raise ValueError(
                    "resume source fingerprint mismatch: the tracked/untracked source patch differs "
                    "from the checkpoint; resume from the recorded source or start a new run."
                )
        else:
            print("[warn] resume checkpoint predates reconstructable source provenance", flush=True)
        model.encoder.load_state_dict(rk["encoder"])
        for key, head in model.pretraining_heads().items():
            if key not in rk.get("heads", {}):
                raise ValueError(
                    f"resume checkpoint predates the consolidated Phase-A objective: missing {key!r}"
                )
            head.load_state_dict(rk["heads"][key])
        if model.physical_target_analyzer is not None:
            target_state = rk.get("physical_target_analyzer")
            if target_state is None:
                raise ValueError("resume checkpoint is missing the frozen physical target analyzer")
            model.physical_target_analyzer.load_state_dict(target_state)
        if jepa_teacher is not None:
            if rk.get("jepa_teacher") is not None:
                jepa_teacher.load_state_dict(rk["jepa_teacher"])
            else:
                raise ValueError("resume checkpoint is missing the JEPA teacher state")
        opt.load_state_dict(rk["optimizer"])
        sched.load_state_dict(rk["scheduler"])
        scaler.load_state_dict(rk["scaler"])
        if "rng" in rk:
            import random as _sr
            torch.set_rng_state(rk["rng"]["torch"].cpu())
            if device.type == "cuda" and rk["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in rk["rng"]["cuda"]])
            np.random.set_state(rk["rng"]["numpy"])
            _sr.setstate(rk["rng"]["python"])
        start_step = saved_step
        # Restore the RUNNING best, not this checkpoint's own val_ba (#6): resuming from last.pt
        # (whose val_ba is the latest, not the best) must not let a later worse val overwrite best.pt.
        best_ba = float(rk.get("best_ba", rk.get("selection_value", rk["val_ba"])))
        latest_selection_ba = rk.get("selection_value")
        latest_selection_step = rk.get("selection_step")
        # Draw FRESH windows for the remaining steps instead of REPLAYING the sampler prefix (audit F1):
        # advance the temperature sampler's epoch so the resumed run's training draw differs from the
        # interrupted run's. Not bit-exact (the design accepts a fresh epoch for the remaining steps),
        # but it no longer re-trains on the exact windows the prefix already saw.
        _samp = getattr(train_loader, "sampler", None)
        if isinstance(_samp, TemperatureSampler):
            _samp.epoch += start_step
        print(f"resumed from {args.resume} at step {start_step} (best_ba {best_ba:.3f})", flush=True)

    # Write run metadata only AFTER resume validation passed, so a rejected resume leaves it untouched (#6).
    source_provenance = write_source_provenance(args.out, source_provenance)
    (args.out / "runtime_provenance.json").write_text(
        json.dumps(runtime_provenance, indent=2) + "\n"
    )
    (args.out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    run_until_step = min(args.stop_after or cfg.steps, cfg.steps)
    if start_step >= run_until_step:
        print(f"checkpoint already reached requested stop step {run_until_step}; nothing to resume",
              flush=True)
        return

    def encode_clean_view_b(batch: dict) -> dict:
        """Encode the SECOND augmentation-positive view (the collate's ``*_b`` keys), clean/no mask.
        Only ['pooled'] is consumed by the VICReg projector. Filterbank analysis runs FP32 like
        view A; its projection and transformer run under the caller's FP16 autocast. The
        conditioning path MATCHES view A (per_channel vs factored), so the encoder sees the same
        text mode — factored view B carries its OWN independently-augmented role/sensor text."""
        p_b = batch["patches_b"].to(device, non_blocking=True).float()
        r_b = batch["rates_b"].to(device, non_blocking=True)
        pl_b = batch["patch_len_b"].to(device, non_blocking=True)
        pos_b = batch["positions_b"].to(device, non_blocking=True)
        pdur_b = (batch["patch_durations_b"].to(device, non_blocking=True)
                  if "patch_durations_b" in batch else None)
        rids_b = (batch["resolution_ids_b"].to(device, non_blocking=True)
                  if "resolution_ids_b" in batch else None)
        cmask_b = batch["channel_mask_b"].to(device, non_blocking=True)
        ppad_b = batch["patch_padding_mask_b"].to(device, non_blocking=True)
        sensor_granularity = cfg.token_granularity == "sensor"
        sid_b = (batch["sensor_id_b"].to(device, non_blocking=True)
                 if sensor_granularity else None)
        tokens_b = model.encoder.tokenize(
            p_b, r_b, pl_b,
            channel_mask=cmask_b,
            source_rate_hz=batch.get("source_rates_b", r_b).to(device, non_blocking=True),
            sensor_id=sid_b,
            n_sensors=(max(map(len, batch["sensor_texts_b"]))
                       if sensor_granularity else None))
        sensor_descriptors_b = None
        if sensor_granularity:
            sensor_descriptors_b, sensor_text_ids_b = \
                model.encoder.encode_sensor_descriptors_unique(batch["sensor_texts_b"], device)
            te_b = tm_b = role_ids_b = ste_b = stm_b = None
        elif cfg.text_conditioning == "factored":
            te_b, tm_b, role_ids_b, ste_b, stm_b, sensor_text_ids_b = \
                model.encoder.encode_texts_factored_unique(
                    batch["role_texts_b"], batch["sensor_texts_b"], device)
            sid_b = batch["sensor_id_b"].to(device, non_blocking=True)
        else:
            te_b, tm_b = model.encoder.encode_texts(batch["texts_b"], device)
            ste_b = stm_b = sid_b = role_ids_b = sensor_text_ids_b = None
        return encode_fn(tokens_b, te_b, tm_b, pos_b,
                                    patch_durations=pdur_b, resolution_ids=rids_b,
                                    channel_mask=cmask_b, patch_padding_mask=ppad_b,
                                    sensor_text_embs=ste_b, sensor_text_masks=stm_b,
                                    sensor_descriptors=sensor_descriptors_b,
                                    sensor_id=sid_b, role_text_ids=role_ids_b,
                                    sensor_text_ids=sensor_text_ids_b)

    # Keep conditioning/folding eager and compile the stable transformer core. Install runtime
    # callables rather than wrapping modules so state_dict keys remain identical.
    encode_fn = model.encoder.encode
    if cfg.compile_encoder and device.type == "cuda":
        # Objective gradient telemetry re-walks the graph with retain_graph=True; disable Inductor's
        # donated-buffer optimization so the diagnostic backward remains valid.
        import torch._functorch.config as functorch_config
        import torch._dynamo.config as dynamo_config
        functorch_config.donated_buffer = False
        # Backward executes after leaving autocast below. This explicit setting follows PyTorch's
        # compiled-AMP contract instead of relying on its "same_as_forward" default assumption.
        functorch_config.backward_pass_autocast = "off"
        # Dynamic patch/sensor shapes can expose an unsupported Inductor specialization even after
        # hundreds of successful compiled updates (observed immediately after the step-1000
        # validation on torch 2.9.0+cu128). Compilation is an optimization, not part of the model:
        # fall back to the identical eager transformer for a graph the backend cannot lower instead
        # of terminating an otherwise healthy run.
        dynamo_config.suppress_errors = True
        model.encoder._compiled_transformer_forward = torch.compile(
            model.encoder.transformer.forward, dynamic=True)
        if jepa_teacher is not None:
            jepa_teacher._compiled_transformer_forward = torch.compile(
                jepa_teacher.transformer.forward, dynamic=True)
        compiled_models = "student + EMA" if jepa_teacher is not None else "student"
        print(f"torch.compile: {compiled_models} transformer core(s) "
              "(dynamic, checkpoint-neutral) — step 1 pays the compile", flush=True)

    if cfg.jepa_mode == "future":
        print(
            "future JEPA: past-only context "
            f"{cfg.future_context_fraction[0]:g}-{cfg.future_context_fraction[1]:g}, "
            f"durations={future_resolution_durations(cfg)}, "
            f"horizons={cfg.future_horizon_bins_seconds}, "
            f"teacher_top_layers={cfg.future_teacher_top_layers}",
            flush=True,
        )
    else:
        mask_description = (
            "contiguous time + same-placement whole-sensor"
            if cfg.token_granularity == "sensor" else
            ("independent contiguous block per resolution" if cfg.multiresolution
             else "contiguous time + whole-channel")
        )
        if cfg.descriptor_weight > 0:
            mask_description += " + descriptor"
        if cfg.jepa_weight > 0 or cfg.mae_weight > 0:
            print(f"masked JEPA: {mask_description}, ratio={cfg.mask_ratio_time:g}", flush=True)
        else:
            print("masked JEPA: disabled (VICReg-only control)", flush=True)

    # Rolling CPU-side data telemetry. These counters cover every batch between scalar log records,
    # rather than sampling only the batch that happens to land on a logging step.
    data_batches = 0
    data_examples = 0
    source_counts: Counter = Counter()
    stream_counts: Counter = Counter()
    augmentation_counts: Counter = Counter()
    augmentation_examples = 0
    channel_count_counts: Counter = Counter()
    patch_pair_counts: Counter = Counter()
    observed_rates: list[float] = []
    observed_source_rates: list[float] = []
    last_log_wall = time.perf_counter()
    last_log_step = start_step
    amp_skipped_since_log = 0
    amp_skipped_total = 0
    amp_consecutive_skips = 0
    zero_target_count = torch.zeros((), device=device)
    jepa_ineligible_count = torch.zeros((), device=device)
    zero_target_examples = 0
    descriptor_target_count = torch.zeros((), device=device)
    descriptor_target_windows = torch.zeros((), device=device)

    model.train()
    for step, batch in enumerate(train_loader, start=start_step + 1):
        calibration_report = None
        do_log = step % 50 == 0 or step == 1
        do_objective_grad_log = step == 1 or step % 500 == 0
        if hasattr(fe, "request_runtime_telemetry"):
            fe.request_runtime_telemetry(do_log)
        patches = batch["patches"].to(device, non_blocking=True)   # NOT gravity-aligned (2026-07-19 design)
        rates = batch["rates"].to(device, non_blocking=True)
        patch_len = batch["patch_len"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        patch_durations = (batch["patch_durations"].to(device, non_blocking=True)
                           if "patch_durations" in batch else None)
        resolution_ids = (batch["resolution_ids"].to(device, non_blocking=True)
                          if "resolution_ids" in batch else None)
        channel_mask = batch["channel_mask"].to(device, non_blocking=True)
        patch_pad = batch["patch_padding_mask"].to(device, non_blocking=True)
        raw_patch_len = patch_len
        raw_patch_pad = patch_pad
        raw_patch_starts = batch["patch_starts"].to(device, non_blocking=True)
        raw_positions = positions
        raw_patch_durations = patch_durations
        raw_resolution_ids = resolution_ids
        B, _, _, C = patches.shape
        full_recording_duration = (
            (raw_patch_len * raw_patch_pad).sum(dim=1).float() / rates.clamp_min(1e-6)
        )
        if cfg.frontend == "multispan":
            grid_meta = fe.token_metadata(full_recording_duration.detach().cpu())
            positions = grid_meta["positions"].to(device, non_blocking=True)
            patch_durations = grid_meta["durations"].to(device, non_blocking=True)
            resolution_ids = grid_meta["resolution_ids"].to(device, non_blocking=True)
            patch_pad = grid_meta["token_mask"].to(device, non_blocking=True)
            planner_starts = grid_meta["positions"] - 0.5 * grid_meta["durations"]
            planner_ends = grid_meta["positions"] + 0.5 * grid_meta["durations"]
            planner_valid = grid_meta["token_mask"]
            planner_resolutions = grid_meta["resolution_ids"]
        else:
            planner_starts = batch["patch_starts"]
            planner_ends = batch["patch_ends"]
            planner_valid = batch["patch_padding_mask"]
            planner_resolutions = batch["resolution_ids"]
        P = positions.shape[1]

        data_batches += 1
        data_examples += B
        source_counts.update(batch["sources"])
        stream_counts.update(batch["streams"])
        channel_count_counts.update(int(value) for value in batch["channel_mask"].sum(dim=1).tolist())
        pair = batch["patch_seconds"]
        pair_key = "+".join(f"{float(value):g}" for value in pair) \
            if isinstance(pair, (tuple, list)) else f"{float(pair):g}"
        patch_pair_counts[pair_key] += 1
        observed_rates.extend(float(value) for value in batch["rates"].tolist())
        observed_source_rates.extend(float(value) for value in batch["source_rates"].tolist())
        for traces in (batch.get("augmentations", []), batch.get("augmentations_b", [])):
            augmentation_examples += len(traces)
            for names in traces:
                augmentation_counts.update(names)

        sensor_granularity = cfg.token_granularity == "sensor"
        sensor_placement = batch.get("sensor_placement")
        if sensor_placement is not None:
            sensor_placement = sensor_placement.to(device, non_blocking=True)
        descriptor_mask = None

        future_context_mask = future_target_mask = future_horizon = future_eligible = None
        future_leakage_count = 0
        if cfg.jepa_mode == "future":
            # Plan from signal-independent physical intervals, so the same objective applies to a
            # collate-owned filterbank grid or a frontend-owned continuous multi-span grid.
            future_plan = make_future_target_plan(
                planner_starts, planner_ends, planner_valid,
                planner_resolutions, context_fraction=cfg.future_context_fraction,
                horizon_bins_seconds=cfg.future_horizon_bins_seconds,
            )
            future_context_mask = future_plan.context_mask.to(device, non_blocking=True)
            future_target_mask = future_plan.target_mask.to(device, non_blocking=True)
            future_horizon = future_plan.horizon_seconds.to(device, non_blocking=True)
            future_eligible = future_plan.eligible.to(device, non_blocking=True)
            # Fail closed if a planner change ever violates the physical-time prefix contract.
            future_context_end = future_plan.context_end.to(device, non_blocking=True)
            context_end = future_plan.context_end[:, None]
            leakage = ((future_plan.context_mask
                        & (planner_ends > context_end + 1e-7)).sum()
                       + (future_plan.target_mask
                          & (planner_starts < context_end - 1e-7)).sum())
            future_leakage_count = int(leakage)
            if future_leakage_count:
                raise RuntimeError(
                    f"future JEPA planner exposed {future_leakage_count} overlapping tokens"
                )
        elif cfg.jepa_weight > 0 or cfg.mae_weight > 0:
            if sensor_granularity:
                # Sensor granularity: the mask grid is (B,P,S), and the planner also emits the
                # descriptor mask. `sensor_present` is not known until the fold runs inside the
                # encoder, so approximate it here from which sensor slots carry a real description
                # (a padded slot has an all-zero bias row and no text) — the encoder re-derives the
                # authoritative mask and intersects, so an over-permissive guess cannot mask a
                # sensor that does not exist.
                # n_max across the batch, NOT sample 0's count: sensor counts are ragged (an
                # accel-only stream carries one, acc+gyro carry two) and the text-id table is padded
                # to the batch maximum, which is the width the encoder derives N from.
                sensor_text_lists = batch.get("sensor_texts") or [[]]
                n_sensors = max((len(texts) for texts in sensor_text_lists), default=1) or 1
                present = torch.tensor(
                    [[i < len(texts) for i in range(n_sensors)]
                     for texts in sensor_text_lists],
                    dtype=torch.bool, device=device)
                plan = make_sensor_mask_plan(
                    B, P, n_sensors, device=device, time_ratio=cfg.mask_ratio_time,
                    valid_patches=patch_pad, sensor_present=present,
                    sensor_placement=sensor_placement,
                    descriptor_event_p=(0.25 if cfg.descriptor_weight > 0 else 0.0))
                descriptor_mask = plan.descriptor_mask
                descriptor_target_count = descriptor_target_count + descriptor_mask.sum()
                descriptor_target_windows = (
                    descriptor_target_windows + descriptor_mask.any(dim=1).sum()
                )
            elif cfg.multiresolution:
                plan = make_per_resolution_mask_plan(
                    resolution_ids, C, GYRO_IDX, channel_mask=channel_mask,
                    valid_patches=patch_pad, time_ratio=cfg.mask_ratio_time)
            else:
                plan = make_mask_plan(B, P, C, GYRO_IDX, device=device,
                                      time_ratio=cfg.mask_ratio_time,
                                      valid_patches=patch_pad, channel_mask=channel_mask)

        with torch.amp.autocast(
            device.type, enabled=device.type == "cuda", dtype=torch.float16,
        ):
            projection_sensor_id = (
                batch["sensor_id"].to(device, non_blocking=True)
                if sensor_granularity else None
            )
            projection_n_sensors = (
                max(map(len, batch["sensor_texts"])) if sensor_granularity else None
            )
            # The neural projection/transformer path uses FP16. The filterbank DSP (rDFT +
            # constant-Q reduction) stays FP32 because FP16 has too little range for raw spectral
            # energy; this is a narrow numerical island and sensor tokens retain gradients.
            with torch.amp.autocast(device.type, enabled=False):
                # Split analysis from projection so the EMA teacher can reuse the DSP. On the
                # fixed arm `analyze` is parameter-free, so the teacher's analysis would be
                # bit-identical anyway and running the rDFT + constant-Q einsum twice per step
                # was pure waste. The learnable arm's analysis reads EMA-diverging parameters,
                # so it keeps its own pass (shared_analysis stays None).
                _src_rate = batch.get("source_rates", rates).to(device, non_blocking=True)
                if cfg.frontend == "multispan":
                    student_lengths, student_raw_mask = truncate_patch_lengths_at_time(
                        raw_patch_len, raw_patch_starts, rates, future_context_end,
                    )
                    student_grid = fe.token_grid(
                        patches.float(), rates, student_lengths, source_rate_hz=_src_rate,
                        patch_mask=student_raw_mask, sensor_id=projection_sensor_id,
                        channel_mask=channel_mask, n_sensors=projection_n_sensors,
                        grid_duration_seconds=full_recording_duration,
                    )
                    student_analysis = shared_analysis = None
                    sensor_tokens = student_grid["tokens"]
                    if sensor_tokens.shape[1] != P:
                        raise RuntimeError("multi-span student and physical planner grids disagree")
                elif cfg.frontend == "fixed":
                    shared_analysis = model.encoder.analyze(
                        patches.float(), rates, patch_len, source_rate_hz=_src_rate)
                else:
                    shared_analysis = None
                if cfg.frontend != "multispan":
                    student_analysis = (shared_analysis if shared_analysis is not None else
                                        model.encoder.analyze(
                                            patches.float(), rates, patch_len,
                                            source_rate_hz=_src_rate,
                                            patch_mask=(future_context_mask
                                                        if cfg.jepa_mode == "future"
                                                        and cfg.frontend == "continuous"
                                                        else None),
                                        ))
                physical_target_analysis = None
                if model.physical_target_analyzer is not None:
                    # A frozen, separately calibrated analyzer defines the decoder target for
                    # every encoder arm. `no_grad` prevents needless FFT autograd state while the
                    # predictor-to-decoder path remains fully differentiable.
                    if cfg.frontend == "fixed":
                        # Calibration and fixed analysis parameters are exact copies, so this is
                        # the same target without paying for a duplicate FFT.
                        physical_target_analysis = shared_analysis.detach()
                    elif cfg.frontend != "multispan":
                        with torch.no_grad():
                            physical_target_analysis = model.physical_target_analyzer.analyze(
                                patches.float(), rates, patch_len, source_rate_hz=_src_rate,
                            )
                enc_channel_mask = channel_mask
                enc_texts = batch["texts"]
            if cfg.frontend != "multispan":
                sensor_tokens = model.encoder.project_tokens(
                    student_analysis, sensor_id=projection_sensor_id,
                    channel_mask=enc_channel_mask, n_sensors=projection_n_sensors,
                )
            # Config-text conditioning, built ONCE and reused by the clean and masked encode passes.
            # per_channel (default): per-channel descriptions -> (B,C,S,384); UNCHANGED from before.
            # factored: ROLE text -> text_embs/text_masks; the per-sensor IDENTITY carried separately
            # (sensor_text_embs/masks/id), summed inside the fusion (docs/design/TEXT_CONDITIONING.md).
            sensor_descriptors = None
            descriptor_target_descriptors = descriptor_target_ids = None
            if sensor_granularity:
                sensor_descriptors, sensor_text_ids = \
                    model.encoder.encode_sensor_descriptors_unique(batch["sensor_texts"], device)
                if cfg.jepa_weight > 0 and cfg.descriptor_weight > 0:
                    descriptor_target_descriptors, descriptor_target_ids = \
                        model.encoder.encode_sensor_descriptors_unique(
                            batch["sensor_target_texts"], device,
                        )
                text_embs = text_masks = role_text_ids = None
                sensor_text_embs = sensor_text_masks = None
                enc_sensor_id = projection_sensor_id
            elif cfg.text_conditioning == "factored":
                (text_embs, text_masks, role_text_ids,
                 sensor_text_embs, sensor_text_masks, sensor_text_ids) = \
                    model.encoder.encode_texts_factored_unique(
                        batch["role_texts"], batch["sensor_texts"], device)
                enc_sensor_id = batch["sensor_id"].to(device, non_blocking=True)
            else:
                text_embs, text_masks = model.encoder.encode_texts(enc_texts, device)
                sensor_text_embs = sensor_text_masks = enc_sensor_id = None
                role_text_ids = sensor_text_ids = None
            # Future signal values are zeroed before the student and future positions are excluded
            # as attention keys/values. This is stronger than relying on a bidirectional mask token:
            # no student context state can contain samples after the selected boundary.
            student_patch_valid = patch_pad
            student_tokens = sensor_tokens
            if cfg.jepa_mode == "future":
                student_patch_valid = future_context_mask
                student_tokens = sensor_tokens * future_context_mask.view(B, P, 1, 1).to(
                    sensor_tokens.dtype
                )
            clean = encode_fn(student_tokens, text_embs, text_masks, positions,
                              patch_durations=patch_durations,
                              resolution_ids=resolution_ids,
                              channel_mask=enc_channel_mask,
                              patch_padding_mask=student_patch_valid,
                              sensor_text_embs=sensor_text_embs,
                              sensor_text_masks=sensor_text_masks,
                              sensor_descriptors=sensor_descriptors,
                              sensor_id=enc_sensor_id, role_text_ids=role_text_ids,
                              sensor_text_ids=sensor_text_ids,
                              return_retrieval_tokens=cfg.jepa_mode == "masked")
            z = (model.vicreg_projector(clean["pooled"])
                 if cfg.jepa_mode == "masked" else clean["pooled"])
            if cfg.jepa_mode == "future":
                masked = clean
                jepa_mask = (future_target_mask.unsqueeze(2)
                             & clean["sensor_present"].unsqueeze(1))
            elif cfg.jepa_weight > 0 or cfg.mae_weight > 0:
                masked = encode_fn(sensor_tokens, text_embs, text_masks, positions,
                                   patch_durations=patch_durations,
                                   resolution_ids=resolution_ids,
                                   token_mask=plan.token_mask,
                                   channel_mask=enc_channel_mask,
                                   patch_padding_mask=patch_pad,
                                   sensor_text_embs=sensor_text_embs,
                                   sensor_text_masks=sensor_text_masks,
                                   sensor_descriptors=sensor_descriptors,
                                   sensor_id=enc_sensor_id, role_text_ids=role_text_ids,
                                   sensor_text_ids=sensor_text_ids,
                                   return_retrieval_tokens=False,
                                   **({"descriptor_mask": descriptor_mask}
                                      if sensor_granularity else {}))
                if sensor_granularity:
                    # The authoritative presence mask comes from the fold, not the pre-encoder
                    # guess used to plan the mask.
                    jepa_mask = (plan.token_mask & masked["sensor_present"].unsqueeze(1)
                                 & patch_pad.unsqueeze(2))
                else:
                    jepa_mask = (plan.token_mask & enc_channel_mask.unsqueeze(1)
                                 & patch_pad.unsqueeze(2))
            else:
                masked = clean
                jepa_mask = torch.zeros(clean["tokens"].shape[:3], dtype=torch.bool, device=device)

            if cfg.jepa_mode == "future":
                jepa_eligible = future_eligible
                targetless = ~future_target_mask.any(dim=1)
                zero_target_count = zero_target_count + (targetless & jepa_eligible).sum()
                jepa_ineligible_count = jepa_ineligible_count + (~jepa_eligible).sum()
                zero_target_examples += B
            elif cfg.jepa_weight > 0 or cfg.mae_weight > 0:
                if sensor_granularity:
                    observable_tokens = (
                        patch_pad.sum(dim=1) * masked["sensor_present"].sum(dim=1)
                    )
                else:
                    observable_tokens = patch_pad.sum(dim=1) * enc_channel_mask.sum(dim=1)
                jepa_eligible = observable_tokens > 1
                targetless = jepa_mask.flatten(1).sum(dim=1).eq(0)
                zero_target_count = zero_target_count + (targetless & jepa_eligible).sum()
                jepa_ineligible_count = jepa_ineligible_count + (~jepa_eligible).sum()
                zero_target_examples += B

            jepa_loss = clean["pooled"].new_zeros(())
            jepa_parts: dict[str, float] = {}
            teacher_clean = None
            if jepa_teacher is not None:
                # The teacher sees the clean view and never receives gradients. Reuse frozen text-LM
                # outputs, but run the teacher's own frontend/fusion/transformer weights.
                with torch.no_grad():
                    if cfg.frontend == "multispan":
                        teacher_grid = jepa_teacher.filterbank.token_grid(
                            patches.float(), rates, raw_patch_len,
                            source_rate_hz=_src_rate, patch_mask=raw_patch_pad,
                            sensor_id=projection_sensor_id, channel_mask=enc_channel_mask,
                            n_sensors=projection_n_sensors,
                            grid_duration_seconds=full_recording_duration,
                        )
                        teacher_sensor_tokens = teacher_grid["tokens"]
                        if teacher_sensor_tokens.shape[1] != P:
                            raise RuntimeError("multi-span teacher and physical planner grids disagree")
                        teacher_analysis = None
                    elif shared_analysis is not None:
                        # Fixed arm: reuse the student's parameter-free analysis, then apply the
                        # TEACHER's own (EMA-lagged) projection under the outer FP16 autocast.
                        teacher_analysis = shared_analysis.detach()
                    else:
                        with torch.amp.autocast(device.type, enabled=False):
                            teacher_analysis = jepa_teacher.analyze(
                                patches.float(), rates, patch_len,
                                source_rate_hz=batch.get("source_rates", rates).to(
                                    device, non_blocking=True),
                            )
                    if cfg.frontend != "multispan":
                        teacher_sensor_tokens = jepa_teacher.project_tokens(
                            teacher_analysis, sensor_id=projection_sensor_id,
                            channel_mask=enc_channel_mask, n_sensors=projection_n_sensors,
                        )
                    teacher_clean = jepa_teacher.encode(
                        teacher_sensor_tokens, text_embs, text_masks, positions,
                        patch_durations=patch_durations,
                        resolution_ids=resolution_ids,
                        channel_mask=enc_channel_mask,
                        patch_padding_mask=patch_pad,
                        sensor_text_embs=sensor_text_embs,
                        sensor_text_masks=sensor_text_masks,
                        sensor_descriptors=sensor_descriptors,
                        sensor_id=enc_sensor_id,
                        role_text_ids=role_text_ids,
                        sensor_text_ids=sensor_text_ids,
                        return_retrieval_tokens=False,
                        return_layer_states=cfg.jepa_mode == "future",
                        # The teacher consumes the SAME `patches` as view A, so it inherits view A's
                        # acquisition config automatically — no separate config draw exists here.
                        # It sees the descriptor unmasked: the target must be the fully-informed
                        # representation, or the student would be chasing a teacher handicapped the
                        # same way it is.
                    )
                if cfg.jepa_mode == "future":
                    teacher_target_grid = normalized_teacher_target(
                        teacher_clean["layer_states"], top_k=cfg.future_teacher_top_layers,
                    )
                    context_valid = (future_context_mask.unsqueeze(2)
                                     & clean["sensor_present"].unsqueeze(1))
                    jepa_prediction, future_indices, future_query_valid = model.future_predictor(
                        clean["tokens"], context_valid, future_target_mask,
                        positions, patch_durations, resolution_ids, future_horizon,
                        clean["descriptor"], clean["sensor_present"],
                    )
                    teacher_targets = gather_token_rows(teacher_target_grid, future_indices)
                    future_query_resolutions = resolution_ids.gather(
                        1, future_indices[..., 0]
                    )
                    jepa_loss = balanced_future_latent_loss(
                        jepa_prediction, teacher_targets, future_query_valid,
                        future_query_resolutions,
                        num_resolutions=len(future_resolution_durations(cfg)),
                    )
                else:
                    jepa_prediction = model.jepa_predictor(masked["tokens"])
                    jepa_loss = masked_ema_latent_loss(
                        jepa_prediction, teacher_clean["tokens"], jepa_mask,
                        token_groups=resolution_ids,
                        token_durations=patch_durations,
                    )
                # JEPA has no negatives, so its loss alone cannot distinguish "learned to predict
                # the teacher" from "the teacher collapsed and anything predicts it". Log the
                # margin over a random masked-position pairing (see losses_repr.pair_contrast).
                if do_log and bool(jepa_mask.any()):
                    if cfg.jepa_mode == "future":
                        valid_prediction = jepa_prediction[future_query_valid]
                        valid_target = teacher_targets[future_query_valid]
                    else:
                        valid_prediction = jepa_prediction[jepa_mask].flatten(1)
                        valid_target = teacher_clean["tokens"][jepa_mask].flatten(1)
                    jepa_diag = pair_contrast(valid_prediction, valid_target)
                    jepa_parts = {f"jepa/{k}": v for k, v in jepa_diag.items()}
                    if cfg.jepa_mode == "masked" and resolution_ids is not None:
                        with torch.no_grad():
                            # Token-axis validity: sensors at sensor granularity, channels
                            # otherwise. Using the channel mask against a (B,P,S) grid would
                            # broadcast-error, and silently would be worse.
                            token_valid = (masked["sensor_present"] if sensor_granularity
                                           else enc_channel_mask)
                            real = patch_pad.unsqueeze(2) & token_valid.unsqueeze(1)
                            for group, name in ((0, "short"), (1, "long")):
                                group_tokens = resolution_ids.eq(group).unsqueeze(2) & real
                                selected = jepa_mask & group_tokens
                                jepa_parts[f"jepa/target_fraction_{name}"] = float(
                                    selected.sum().float() / group_tokens.sum().clamp(min=1)
                                )
                                jepa_parts[f"jepa/loss_{name}"] = float(masked_ema_latent_loss(
                                    jepa_prediction.detach(), teacher_clean["tokens"], selected,
                                    token_durations=patch_durations,
                                ))

            descriptor_loss = jepa_loss.new_zeros(())
            descriptor_acc = jepa_loss.new_zeros(())
            mae_loss = jepa_loss.new_zeros(())
            physical_loss = jepa_loss.new_zeros(())
            physical_zero_baseline = jepa_loss.new_zeros(())
            collapse_result = None
            retrieval_health_rows = None
            z_b = None

            if cfg.jepa_mode == "future":
                context_valid = (future_context_mask.unsqueeze(2)
                                 & clean["sensor_present"].unsqueeze(1))
                collapse_result = patch_variance_covariance(
                    clean["tokens"], context_valid,
                    resolution_ids=resolution_ids,
                    num_resolutions=len(future_resolution_durations(cfg)),
                    variance_weight=cfg.collapse_variance_weight,
                    covariance_weight=cfg.collapse_covariance_weight,
                    target_std=cfg.collapse_target_std,
                )
                retrieval_health_rows = clean["tokens"][context_valid]

                if cfg.physical_weight > 0:
                    target_analyzer = model.physical_target_analyzer
                    physical_target_indices = future_indices
                    if cfg.frontend == "multispan":
                        # Analyze only the few JEPA-selected intervals, not the complete dense
                        # multi-span grid. The lookup maps every predictor patch index back to its
                        # packed physical-target row without changing target semantics.
                        with torch.amp.autocast(device.type, enabled=False), torch.no_grad():
                            full_window, full_samples = ContinuousKernelTokenizer.contiguous_window(
                                patches.float(), raw_patch_len, raw_patch_pad,
                            )
                            target_patches, target_lengths, slot_by_patch, _ = (
                                pack_selected_interval_patches(
                                    full_window, full_samples, rates,
                                    planner_starts.to(device, non_blocking=True),
                                    planner_ends.to(device, non_blocking=True),
                                    future_target_mask, cfg.dft_size,
                                )
                            )
                            physical_target_analysis = target_analyzer.analyze(
                                target_patches, rates, target_lengths,
                                source_rate_hz=_src_rate,
                            )
                        physical_target_indices = torch.stack((
                            slot_by_patch.gather(1, future_indices[..., 0]),
                            future_indices[..., 1],
                        ), dim=-1)
                    if physical_target_analysis is None:
                        raise RuntimeError("physical objective has no frozen target analysis")
                    physical_channels, physical_channel_valid = fixed_filterbank_physical_targets(
                        physical_target_analysis,
                        n_bands=target_analyzer.n_bands,
                        use_resolution_mask=target_analyzer.use_resolution_mask,
                        # The frontend amplitude scalar is not standardized. Keep it out until a
                        # frozen corpus calibration is defined rather than letting it dominate.
                        use_amplitude=target_analyzer.use_amplitude,
                        use_dc=target_analyzer.use_dc,
                        include_amplitude=False,
                    )
                    physical_target_grid, _ = fold_analysis_to_sensors(
                        physical_channels, enc_sensor_id, enc_channel_mask,
                        n_sensors=clean["sensor_present"].shape[1],
                    )
                    physical_valid_grid, _ = fold_analysis_to_sensors(
                        physical_channel_valid.to(physical_channels.dtype),
                        enc_sensor_id, enc_channel_mask,
                        n_sensors=clean["sensor_present"].shape[1],
                    )
                    physical_targets = gather_token_rows(
                        physical_target_grid, physical_target_indices,
                    )
                    physical_feature_valid = gather_token_rows(
                        physical_valid_grid, physical_target_indices,
                    )
                    physical_prediction = model.physical_decoder(jepa_prediction)
                    physical_loss = balanced_physical_loss(
                        physical_prediction, physical_targets, future_query_valid,
                        future_query_resolutions, physical_feature_valid,
                        num_resolutions=len(future_resolution_durations(cfg)),
                    )
                    physical_zero_baseline = balanced_physical_loss(
                        torch.zeros_like(physical_prediction), physical_targets,
                        future_query_valid, future_query_resolutions, physical_feature_valid,
                        num_resolutions=len(future_resolution_durations(cfg)),
                    )

                out = combine_future_losses(
                    jepa_loss, physical_loss, collapse_result.total,
                    future_weight=cfg.jepa_weight,
                    physical_weight=cfg.physical_weight,
                    collapse_weight=cfg.collapse_weight,
                )
                jepa_objective = jepa_loss
                parts = {}
                if do_log:
                    parts = {
                        "future/loss": float(jepa_loss.detach()),
                        "physical/loss": float(physical_loss.detach()),
                        "physical/zero_baseline": float(physical_zero_baseline.detach()),
                        "physical/improvement_over_zero": float(
                            (physical_zero_baseline - physical_loss).detach()),
                        "collapse/total": float(collapse_result.total.detach()),
                        "collapse/variance": float(collapse_result.variance.detach()),
                        "collapse/covariance": float(collapse_result.covariance.detach()),
                        "collapse/min_std": float(collapse_result.min_std.detach()),
                        "future/eligible_fraction": float(future_eligible.float().mean()),
                        "future/context_fraction_tokens": float(
                            future_context_mask.sum().float() / patch_pad.sum().clamp_min(1)),
                        "future/target_fraction_tokens": float(
                            future_target_mask.sum().float() / patch_pad.sum().clamp_min(1)),
                        "future/leakage_count": future_leakage_count,
                        **jepa_parts,
                    }
                    selected_horizons = future_horizon.gather(1, future_indices[..., 0])
                    with torch.no_grad():
                        per_query_error = torch.nn.functional.smooth_l1_loss(
                            torch.nn.functional.layer_norm(
                                jepa_prediction.float(), (jepa_prediction.shape[-1],)
                            ),
                            torch.nn.functional.layer_norm(
                                teacher_targets.float(), (teacher_targets.shape[-1],)
                            ),
                            reduction="none",
                        ).mean(dim=-1)
                        for horizon_i, (horizon_lo, horizon_hi) in enumerate(
                            cfg.future_horizon_bins_seconds
                        ):
                            selected = (future_query_valid
                                        & selected_horizons.ge(horizon_lo)
                                        & selected_horizons.lt(horizon_hi))
                            parts[f"future/horizon_{horizon_i}_count"] = int(selected.sum())
                            parts[f"future/horizon_{horizon_i}_loss"] = float(
                                per_query_error[selected].mean() if bool(selected.any())
                                else per_query_error.new_zeros(())
                            )
                        for resolution_i, duration in enumerate(future_resolution_durations(cfg)):
                            selected = future_query_valid & future_query_resolutions.eq(resolution_i)
                            parts[f"future/resolution_{duration:g}s_count"] = int(selected.sum())
                            parts[f"future/resolution_{duration:g}s_loss"] = float(
                                per_query_error[selected].mean() if bool(selected.any())
                                else per_query_error.new_zeros(())
                            )
            else:
                clean_b = encode_clean_view_b(batch)
                z_b = model.vicreg_projector(clean_b["pooled"])
                pooled_vicreg = vicreg(
                    z, z_b,
                    invariance_weight=cfg.vicreg_invariance_weight,
                    variance_weight=cfg.vicreg_variance_weight,
                    covariance_weight=cfg.vicreg_covariance_weight,
                    target_std=cfg.vicreg_target_std,
                )
                vicreg_result = pooled_vicreg
                retrieval_vicreg = None
                if sensor_granularity and cfg.retrieval_vicreg_fraction > 0:
                    ra = clean.get("retrieval_tokens")
                    rb = clean_b.get("retrieval_tokens")
                    if ra is None or rb is None or ra.shape != rb.shape:
                        raise RuntimeError("aligned sensor-row VICReg requires matching retrieval tokens")
                    patch_pad_b = batch["patch_padding_mask_b"].to(device, non_blocking=True).bool()
                    if patch_pad.shape != patch_pad_b.shape:
                        raise RuntimeError("sensor-row VICReg requires aligned patch grids")
                    sensor_valid = clean["sensor_present"] & clean_b["sensor_present"]
                    row_valid = patch_pad.unsqueeze(2) & patch_pad_b.unsqueeze(2) \
                        & sensor_valid.unsqueeze(1)
                    retrieval_vicreg = vicreg(
                        ra[row_valid], rb[row_valid],
                        invariance_weight=cfg.vicreg_invariance_weight,
                        variance_weight=cfg.vicreg_variance_weight,
                        covariance_weight=cfg.vicreg_covariance_weight,
                        target_std=cfg.vicreg_target_std,
                    )
                    retrieval_health_rows = ra[row_valid]
                    fraction = float(cfg.retrieval_vicreg_fraction)
                    vicreg_result = VICRegOutput(
                        total=(1.0 - fraction) * pooled_vicreg.total
                              + fraction * retrieval_vicreg.total,
                        invariance=(1.0 - fraction) * pooled_vicreg.invariance
                                   + fraction * retrieval_vicreg.invariance,
                        variance=(1.0 - fraction) * pooled_vicreg.variance
                                 + fraction * retrieval_vicreg.variance,
                        covariance=(1.0 - fraction) * pooled_vicreg.covariance
                                   + fraction * retrieval_vicreg.covariance,
                        min_std=torch.minimum(pooled_vicreg.min_std, retrieval_vicreg.min_std),
                    )

                # Explicit legacy controls remain isolated to masked mode.
                if (cfg.descriptor_weight > 0 and sensor_granularity
                        and descriptor_mask is not None and bool(descriptor_mask.any())):
                    score_rows = descriptor_mask & masked["sensor_present"]
                    target_descriptor = descriptor_target_descriptors.index_select(
                        0, descriptor_target_ids.clamp_min(0).reshape(-1),
                    ).reshape(*descriptor_target_ids.shape, -1)
                    descriptor_loss, descriptor_acc = descriptor_retrieval_loss(
                        masked["descriptor_pred"], target_descriptor,
                        target_ids=descriptor_target_ids,
                        candidate_descriptors=descriptor_target_descriptors,
                        row_mask=score_rows,
                    )
                if cfg.mae_weight > 0 and model.mae_head is not None and sensor_granularity:
                    mae_target, axis_valid = fold_analysis_to_sensors(
                        student_analysis.detach(), enc_sensor_id, enc_channel_mask,
                        n_sensors=masked["sensor_present"].shape[1],
                    )
                    mae_loss = masked_analysis_reconstruction_loss(
                        model.mae_head(masked["tokens"]), mae_target, jepa_mask, axis_valid,
                    )
                jepa_objective = jepa_loss + cfg.descriptor_weight * descriptor_loss
                out = phase_a_loss(
                    jepa_objective, vicreg_result.total,
                    mae=(mae_loss if cfg.mae_weight > 0 else None),
                    mae_weight=cfg.mae_weight,
                    jepa_weight=cfg.jepa_weight,
                    vicreg_weight=cfg.vicreg_weight,
                )
                parts = {}
                if do_log:
                    parts = {
                        "jepa": float(jepa_loss.detach()),
                        "vicreg": float(vicreg_result.total.detach()),
                        "vicreg/invariance": float(vicreg_result.invariance.detach()),
                        "vicreg/variance": float(vicreg_result.variance.detach()),
                        "vicreg/covariance": float(vicreg_result.covariance.detach()),
                        "vicreg/min_std": float(vicreg_result.min_std.detach()),
                        "mae/loss": float(mae_loss.detach()),
                        "mae/loss_weighted": float((cfg.mae_weight * mae_loss).detach()),
                        "vicreg/pooled_total": float(pooled_vicreg.total.detach()),
                        "vicreg/retrieval_total": float(
                            retrieval_vicreg.total.detach() if retrieval_vicreg is not None
                            else pooled_vicreg.total.new_zeros(())
                        ),
                        **jepa_parts,
                        "descriptor/loss": float(descriptor_loss.detach()),
                        "descriptor/top1": float(descriptor_acc.detach()),
                    }
            frontend_reg = model.encoder.filterbank.adaptation_regularization()
            out.total = out.total + cfg.frontend_reg_weight * frontend_reg
            # Converting a CUDA scalar to float synchronizes the whole stream. These values are
            # telemetry only, so materialize them on log steps instead of stalling every update.
            if do_log:
                parts.update({
                    "frontend_reg": float(frontend_reg.detach()),
                    "frontend_reg_weighted": float(
                        (cfg.frontend_reg_weight * frontend_reg).detach()),
                })
                if cfg.jepa_mode == "masked":
                    parts.update({f"vicreg/{key}": value
                                  for key, value in pair_contrast(z, z_b).items()})

        objective_grad_norms = {}
        if do_objective_grad_log:
            top_geometry = objective_encoder_grad_geometry(out.terms, model.encoder)
            objective_grad_norms = {
                **{f"grad_objective/{name}": value
                   for name, value in top_geometry["norms"].items()},
                **{f"grad_cosine/{name.replace('|', '_vs_')}": value
                   for name, value in top_geometry["cosines"].items()},
            }
            norm_sum = sum(top_geometry["norms"].values())
            objective_grad_norms.update({
                f"grad_objective/{name}_share": value / max(norm_sum, 1e-12)
                for name, value in top_geometry["norms"].items()
            })

        calibrating = (
            cfg.objective_calibration_at > 0
            and cfg.objective_calibration_at - cfg.objective_calibration_batches < step
            <= cfg.objective_calibration_at
        )
        if calibrating:
            if cfg.jepa_mode == "future":
                unit_losses = {
                    "future": jepa_loss,
                    "collapse": collapse_result.total,
                }
                if cfg.physical_weight > 0:
                    unit_losses["physical"] = physical_loss
            else:
                # VICReg retains its published internal 25/25/1 coefficients.
                unit_losses = {"jepa": jepa_objective, "vicreg": vicreg_result.total}
            unit_geometry = objective_encoder_grad_geometry(unit_losses, model.encoder)
            calibration_samples.append(unit_geometry)
            calibration_steps.append(step)
            for source in batch["sources"]:
                calibration_source_counts[source] = calibration_source_counts.get(source, 0) + 1

            if step == cfg.objective_calibration_at:
                if cfg.jepa_mode == "future":
                    current_weights = {
                        "future": cfg.jepa_weight,
                        "collapse": cfg.collapse_weight,
                    }
                    target_shares = {
                        "future": cfg.objective_target_jepa_share,
                        "collapse": 1.0 - cfg.objective_target_jepa_share
                                    - (cfg.objective_target_physical_share
                                       if cfg.physical_weight > 0 else 0.0),
                    }
                    if cfg.physical_weight > 0:
                        current_weights["physical"] = cfg.physical_weight
                        target_shares["physical"] = cfg.objective_target_physical_share
                    calibration_report = recommend_fixed_objective_weights(
                        calibration_samples,
                        current_weights=current_weights,
                        target_shares=target_shares,
                    )
                else:
                    calibration_report = recommend_objective_weights(
                        calibration_samples,
                        current_jepa_weight=cfg.jepa_weight,
                        current_vicreg_weight=cfg.vicreg_weight,
                        target_jepa_share=cfg.objective_target_jepa_share,
                    )
                cosine_names = sorted({
                    name for sample in calibration_samples for name in sample["cosines"]
                })
                calibration_report.update({
                    "mode": cfg.objective_calibration_mode,
                    "sample_steps": calibration_steps,
                    "n_batches": len(calibration_samples),
                    "current": ({
                        name: weight for name, weight in {
                            "future": cfg.jepa_weight,
                            "physical": cfg.physical_weight,
                            "collapse": cfg.collapse_weight,
                        }.items() if weight > 0
                    } if cfg.jepa_mode == "future" else {
                        "jepa_weight": cfg.jepa_weight,
                        "vicreg_weight": cfg.vicreg_weight,
                    }),
                    "gradient_cosine": {
                        name: {
                            "median": statistics.median(
                                sample["cosines"].get(name, 0.0)
                                for sample in calibration_samples
                            ),
                            "min": min(sample["cosines"].get(name, 0.0)
                                       for sample in calibration_samples),
                            "max": max(sample["cosines"].get(name, 0.0)
                                       for sample in calibration_samples),
                        }
                        for name in cosine_names
                    },
                    "sampled_source_share": {
                        source: count / sum(calibration_source_counts.values())
                        for source, count in sorted(calibration_source_counts.items())
                    },
                    "config": asdict(cfg),
                    "git": source_provenance["git"],
                    "source_provenance": source_provenance,
                    "corpus_fingerprint": corpus_fp,
                    "samples": calibration_samples,
                })
        if device.type == "cuda":
            # Keep the safety check on device; converting it to bool stalls the CPU every update.
            torch._assert_async(torch.isfinite(out.total), "non-finite Phase-A loss")
        elif not bool(torch.isfinite(out.total)):
            failure_parts = {name: float(value.detach()) for name, value in out.terms.items()}
            failure_parts["frontend_reg"] = float(frontend_reg.detach())
            raise FloatingPointError(f"non-finite Phase-A loss at step {step}: {failure_parts}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(out.total).backward()
        scaler.unscale_(opt)
        gnorms = module_grad_norms(model) if do_log else {}     # pre-clip per-module grad norms
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        clip_coefficient = (min(
            1.0, float(cfg.grad_clip) / (float(total_grad_norm.detach()) + 1e-6)
        ) if do_log else None)
        old_scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        # GradScaler lowers its scale when it skips a non-finite optimizer step. The EMA target and
        # LR schedule must advance only with real student updates, or the teacher drifts toward
        # unchanged weights while the optimizer trajectory silently loses a scheduler step.
        optimizer_stepped = scaler.get_scale() >= old_scale
        if optimizer_stepped:
            amp_consecutive_skips = 0
        else:
            amp_skipped_since_log += 1
            amp_skipped_total += 1
            amp_consecutive_skips += 1
        if jepa_teacher is not None and optimizer_stepped:
            update_ema_encoder(model.encoder, jepa_teacher, ema_decay_at(step))
        if optimizer_stepped:
            sched.step()
        if do_log:
            lrs = sched.get_last_lr()
            log_wall = time.perf_counter()
            steps_in_window = max(step - last_log_step, 1)
            seconds_in_window = max(log_wall - last_log_wall, 1e-9)
            steps_per_second = steps_in_window / seconds_in_window
            rec = {"step": step, "lr": lrs[0],
                   "elapsed_s": round(time.time() - t0, 1),
                   "patch_seconds": batch["patch_seconds"],
                   "total": round(float(out.total.detach()), 4), **parts, **gnorms}
            rec.update({
                "perf/steps_per_s": steps_per_second,
                "perf/examples_per_s": steps_per_second * cfg.batch_size,
                "perf/eta_minutes": max(run_until_step - step, 0) / steps_per_second / 60.0,
                "amp/scale": float(scaler.get_scale()),
                "amp/dtype": "float16" if device.type == "cuda" else "float32",
                "amp/skipped_updates_window": int(amp_skipped_since_log),
                "amp/skipped_updates_total": int(amp_skipped_total),
                "amp/consecutive_skips": int(amp_consecutive_skips),
                "data/source_share_window": {
                    key: value / max(data_examples, 1)
                    for key, value in sorted(source_counts.items())
                },
                "data/batches_window": int(data_batches),
                "data/examples_window": int(data_examples),
                "data/source_share_target": dict(sorted(
                    temperature_sampler.dataset_probabilities.items()
                )),
                "data/stream_share_window": {
                    key: value / max(data_examples, 1)
                    for key, value in sorted(stream_counts.items())
                },
                "data/channel_count_share_window": {
                    str(key): value / max(data_examples, 1)
                    for key, value in sorted(channel_count_counts.items())
                },
                "data/patch_pair_share_window": {
                    key: value / max(data_batches, 1)
                    for key, value in sorted(patch_pair_counts.items())
                },
                "data/augmentation_rate_window": {
                    key: value / max(augmentation_examples, 1)
                    for key, value in sorted(augmentation_counts.items())
                },
                "data/rate_hz_min": min(observed_rates, default=float("nan")),
                "data/rate_hz_median": float(np.median(observed_rates)) if observed_rates else float("nan"),
                "data/rate_hz_max": max(observed_rates, default=float("nan")),
                "data/source_rate_hz_min": min(observed_source_rates, default=float("nan")),
                "data/source_rate_hz_median": (
                    float(np.median(observed_source_rates)) if observed_source_rates else float("nan")
                ),
                "data/source_rate_hz_max": max(observed_source_rates, default=float("nan")),
                "jepa_zero_target_frac_window": (
                    float(zero_target_count) / max(zero_target_examples, 1)
                    if cfg.jepa_weight > 0 else 0.0
                ),
                "jepa_ineligible_frac_window": (
                    float(jepa_ineligible_count) / max(zero_target_examples, 1)
                    if cfg.jepa_weight > 0 else 0.0
                ),
                "descriptor/targets_window": int(descriptor_target_count),
                "descriptor/target_window_fraction": (
                    float(descriptor_target_windows) / max(zero_target_examples, 1)
                    if sensor_granularity and cfg.jepa_weight > 0 else 0.0
                ),
            })
            rec.update({f"loss_weighted/{name}": float(value.detach())
                        for name, value in out.terms.items()})
            if cfg.jepa_mode == "future":
                rec.update({
                    "objective_weight/future": float(cfg.jepa_weight),
                    "objective_weight/physical": float(cfg.physical_weight),
                    "objective_weight/collapse": float(cfg.collapse_weight),
                })
            else:
                rec.update({
                    "loss_weighted/jepa_latent": float(
                        (cfg.jepa_weight * jepa_loss).detach()),
                    "loss_weighted/descriptor": float(
                        (cfg.jepa_weight * cfg.descriptor_weight * descriptor_loss).detach()),
                    "objective_weight/jepa": float(cfg.jepa_weight),
                    "objective_weight/vicreg": float(cfg.vicreg_weight),
                })
            input_values = patches.detach().float()
            finite_input = torch.isfinite(input_values)
            rec.update({
                "data/input_finite_fraction": float(finite_input.float().mean()),
                "data/input_abs_max": float(
                    torch.where(finite_input, input_values.abs(), 0.0).max()
                ),
                "data/input_rms": float(torch.sqrt(
                    torch.where(finite_input, input_values.square(), 0.0).mean()
                )),
            })
            rec["grad/total_preclip"] = float(total_grad_norm.detach())
            rec["grad/clip_coefficient"] = clip_coefficient
            rec["grad/clipped"] = float(clip_coefficient < 1.0)
            rec.update(objective_grad_norms)
            rec.update(representation_health(clean["pooled"], "repr_encoder"))
            if retrieval_health_rows is not None:
                rec.update(representation_health(retrieval_health_rows, "repr_retrieval"))
            if cfg.jepa_mode == "masked":
                rec.update(representation_health(z, "repr_projector"))
            if teacher_clean is not None and do_objective_grad_log:
                rec.update(representation_health(teacher_clean["pooled"], "repr_teacher"))
            if jepa_teacher is not None:
                rec.update(ema_teacher_health(model.encoder, jepa_teacher))
                decay = ema_decay_at(step)
                rec["ema/decay"] = decay
                rec["ema/half_life_examples"] = (
                    float("inf") if decay >= 1.0
                    else math.log(0.5) / math.log(max(decay, 1e-12)) * cfg.batch_size
                )
            if cfg.jepa_weight > 0 or cfg.mae_weight > 0:
                # Separate planner failures from physically ineligible one-token windows. The former
                # must remain zero; the latter legitimately receive VICReg but no JEPA loss.
                with torch.no_grad():
                    zero_target = ((jepa_mask.flatten(1).sum(dim=1) == 0)
                                   & jepa_eligible).float()
                    ineligible = (~jepa_eligible).float()
                rec["jepa_zero_target_frac_by_source"] = per_source_mean(
                    zero_target, batch["sources"])
                rec["jepa_ineligible_frac_by_source"] = per_source_mean(
                    ineligible, batch["sources"])
                if cfg.jepa_mode == "future":
                    rec["future/targets_per_window_by_source"] = per_source_mean(
                        future_target_mask.sum(dim=1).float(), batch["sources"]
                    )
            if len(lrs) > 1:
                rec["lr_frontend"] = lrs[1]
            if model.encoder.filterbank.learnable:
                rec.update(model.encoder.filterbank.adaptation_summary())
            if hasattr(fe, "runtime_summary"):
                rec.update(fe.runtime_summary())
            if model.encoder.use_duration_embedding:
                rec["duration/gate"] = float(torch.sigmoid(
                    model.encoder.duration_gate_logit.detach()))
            if device.type == "cuda":
                rec.update({
                    "memory/allocated_gib": torch.cuda.memory_allocated(device) / 1e9,
                    "memory/reserved_gib": torch.cuda.memory_reserved(device) / 1e9,
                    "memory/peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1e9,
                })
            print(json.dumps(rec), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            data_batches = 0
            data_examples = 0
            source_counts.clear()
            stream_counts.clear()
            augmentation_counts.clear()
            augmentation_examples = 0
            channel_count_counts.clear()
            patch_pair_counts.clear()
            observed_rates.clear()
            observed_source_rates.clear()
            amp_skipped_since_log = 0
            zero_target_count.zero_()
            jepa_ineligible_count.zero_()
            zero_target_examples = 0
            descriptor_target_count.zero_()
            descriptor_target_windows.zero_()
            last_log_wall = log_wall
            last_log_step = step

        if calibration_report is not None:
            report_path = args.out / "objective_calibration.json"
            report_path.write_text(json.dumps(calibration_report, indent=2))
            if cfg.objective_calibration_mode == "apply":
                recommended = calibration_report["recommended"]
                if cfg.jepa_mode == "future":
                    cfg.jepa_weight = float(recommended["future"])
                    cfg.collapse_weight = float(recommended["collapse"])
                    if "physical" in recommended:
                        cfg.physical_weight = float(recommended["physical"])
                else:
                    cfg.jepa_weight = float(recommended["jepa_weight"])
                    cfg.vicreg_weight = float(recommended["vicreg_weight"])
                # This is now the resolved trajectory configuration used from step+1 onward.
                # Rewriting run_config makes a later resume pass the same frozen coefficients.
                (args.out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))
            event = {
                "step": step,
                "event": ("objective_calibration_applied"
                          if cfg.objective_calibration_mode == "apply"
                          else "objective_calibration_complete"),
                "recommended": calibration_report["recommended"],
                "report": str(report_path),
            }
            print(json.dumps(event), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(event) + "\n")
            if cfg.objective_calibration_mode == "report":
                print("calibration pilot complete; recommendation written without changing weights",
                      flush=True)
                calibration_completed = True
                break
            print("objective coefficients applied once and frozen for the remaining steps", flush=True)

        if not (
            cfg.objective_calibration_at > 0 and cfg.objective_calibration_mode == "report"
        ) and (
            step % cfg.val_every == 0 or step == run_until_step
        ):
            # Query/support cover every available label/stream cell up to the fixed per-label cap.
            # These small fixed subsets take ~1.8 s eagerly. Sending their different batch shapes
            # through the training compiler costs ~35 s once and does not amortize over one run.
            # Temporarily bypass only the runtime compile hook; parameters and calculations match.
            if has_internal_classification_probe:
                compiled_transformer = model.encoder._compiled_transformer_forward
                model.encoder._compiled_transformer_forward = None
                try:
                    val_z, val_y, val_src, val_stream = embed_stratified(
                        model, val_loader, device, cfg.val_per_label,
                        label_totals=val_label_totals,
                    )
                    # Same support bank at every validation and across comparison arms.
                    train_eval_gen.manual_seed(cfg.data_seed)
                    train_z, train_y, _, _ = embed_stratified(
                        model, train_eval_loader, device, cfg.val_per_label,
                        target_labels=set(val_y.tolist()), label_totals=train_label_totals,
                    )
                finally:
                    model.encoder._compiled_transformer_forward = compiled_transformer
                knn_pred = knn_predict(train_z, train_y, val_z, cfg.knn_k)
                ba = balanced_acc(knn_pred, val_y)
                hetero_ba = label_group_balanced_acc(knn_pred, val_y, val_stream)
                # ConSE-style text-cosine probe: ridge-map sensor->label-text space on the train
                # support, cosine-classify val against the label prototypes.
                conse_pred = conse_probe_predict(train_z, train_y, val_z, val_y, label_protos)
                conse_ba = balanced_acc(conse_pred, val_y)
                conse_hetero_ba = label_group_balanced_acc(conse_pred, val_y, val_stream)
            else:
                # A single `__unlabeled__` class gives every representation 100% kNN/ConSE. Keep
                # those metrics explicitly absent; checkpoint selection uses the held-out labelled
                # development roster below.
                val_y = torch.empty(0, dtype=torch.long)
                val_src, val_stream = [], []
                knn_pred = conse_pred = torch.empty(0, dtype=torch.long)
                ba = hetero_ba = conse_ba = conse_hetero_ba = float("nan")
            run_selection = bool(cfg.selection_datasets) and (
                step % cfg.selection_every == 0 or step == run_until_step
            )
            if run_selection:
                compiled_transformer = model.encoder._compiled_transformer_forward
                model.encoder._compiled_transformer_forward = None
                try:
                    selection_ba, selection_by_dataset = development_transfer_score(
                        model.encoder, device, tuple(cfg.selection_datasets), patching="checkpoint",
                    )
                finally:
                    model.encoder._compiled_transformer_forward = compiled_transformer
                latest_selection_ba = selection_ba
                latest_selection_step = step
            elif not cfg.selection_datasets:
                selection_ba, selection_by_dataset = hetero_ba, {}
                latest_selection_ba = selection_ba
                latest_selection_step = step
            else:
                selection_ba, selection_by_dataset = latest_selection_ba, {}
            # per-source val BA — which datasets cluster (kNN) / align to text (conse) well
            vs = np.asarray(val_src)
            ba_by_src, conse_by_src = {}, {}
            for s in sorted(set(val_src)):
                mt = torch.from_numpy(vs == s)
                if int(mt.sum()) >= cfg.knn_k:
                    ba_by_src[s] = round(balanced_acc(knn_pred[mt], val_y[mt]), 4)
                    conse_by_src[s] = round(balanced_acc(conse_pred[mt], val_y[mt]), 4)
            rec = {"step": step, "val_knn_ba": ba,
                   "val_knn_label_stream_ba": round(hetero_ba, 4),
                   "val_conse_ba": round(conse_ba, 4),
                   "val_conse_label_stream_ba": round(conse_hetero_ba, 4),
                   "development_transfer_knn_ba": (
                       round(selection_ba, 4) if selection_ba is not None else None
                   ),
                   "development_transfer_step": latest_selection_step,
                   "development_transfer_by_dataset": {
                       key: round(value, 4) for key, value in selection_by_dataset.items()
                   },
                   "val_ba_by_source": ba_by_src, "val_conse_by_source": conse_by_src}
            if device.type == "cuda":
                # peak so far (train step + val embedding) — memory telemetry.
                rec["peak_gib"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            print(json.dumps(rec), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            checkpoint("last.pt", step, hetero_ba)
            if (run_selection or not cfg.selection_datasets) and selection_ba > best_ba:
                best_ba = selection_ba
                checkpoint("best.pt", step, hetero_ba)
            # The next throughput window measures training only, not this deliberately expensive
            # validation/checkpoint interval.
            last_log_wall = time.perf_counter()
            last_log_step = step
        if step >= run_until_step:
            break

    if calibration_completed:
        print(f"done: objective calibration report in {args.out}", flush=True)
    else:
        if run_until_step < cfg.steps:
            print(f"bounded monitor stopped at step {run_until_step}; full schedule remains "
                  f"{cfg.steps} steps and this checkpoint can be resumed", flush=True)
        metric = ("development transfer kNN" if cfg.selection_datasets
                  else "val label/stream-macro kNN")
        print(f"done: best {metric} {best_ba:.3f} · checkpoints in {args.out}", flush=True)


if __name__ == "__main__":
    main()
