"""Sensor-level tokens — the (B,P,C,d) -> (B,P,S,d) fold, and the two conditioning projections.

DESIGN OF RECORD (docs/design/DESIGN_OF_RECORD.md). A **sensor** is one modality triad at one
placement (accel xyz, or gyro xyz); a 6-channel stream is two sensors. Three things change from the
per-channel encoder:

1. ``SensorFold`` folds a sensor's three axis tokens into ONE token. Sequence length drops 3x on
   6-channel streams, and role text ("x"/"y"/"z") becomes unnecessary — axis order is fixed inside
   the token, so position carries the axis identity that text used to.

2. ``ConditioningProjection`` gives ``text_descriptor`` and ``sensor_bias`` each a LEARNABLE MLP over
   a FROZEN artifact. That split is load-bearing: the projections adapt, the underlying SBERT vector
   and closed-form statistics do not. If the statistics themselves were learned they would stop
   being measurements, and the declared-vs-measured audit property would die with them.

3. The optional ``DescriptorHead`` ablation reconstructs a masked ``text_descriptor`` from the
   sensor's context. It is disabled in the reference recipe. When enabled, the caller scores it by
   retrieval rather than MSE because regression to corpus-mean text is a trivial solution.

WHY FOLDING NEEDS AN AXIS VALIDITY MASK
---------------------------------------
Per-channel tokens let ``channel_mask`` drop a dead axis outright. Once three axes share a token,
a dead axis (mhealth's degenerate gyro, or a channel-dropout event) would otherwise contribute its
raw value to the concatenation and silently poison the sensor. The fold therefore zeroes invalid
axes AND appends an explicit 3-dim validity indicator, so "this axis read zero" and "this axis is
absent" stay distinguishable — the same present-vs-unobservable distinction the Nyquist mask draws
in the frequency domain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

AXES_PER_SENSOR = 3
CONDITIONING_SCHEMA_V2 = "acquisition-conditioning-v2"
LEGACY_CONDITIONING_SCHEMA = "combined-text-v1"

MODALITY_ACCELEROMETER = 0
MODALITY_GYROSCOPE = 1
N_MODALITIES = 2

GRAVITY_PRESENT = 0
GRAVITY_REMOVED = 1
GRAVITY_UNKNOWN = 2
GRAVITY_NOT_APPLICABLE = 3
N_GRAVITY_STATES = 4
RATE_REFERENCE_HZ = 50.0


class SensorFold(nn.Module):
    """(B,P,C,d) per-channel tokens -> (B,P,S,d) per-sensor tokens.

    ``sensor_id`` (B,C) maps each channel to its sensor index, and channels belonging to one sensor
    are taken in their existing order — which is the fixed xyz order of ``CHANNELS``. That fixed
    order is what makes a positional encoding over axes unnecessary.
    """

    def __init__(self, d_model: int, axes: int = AXES_PER_SENSOR, dropout: float = 0.1):
        super().__init__()
        self.axes = axes
        self.d_model = d_model
        # +axes for the validity indicator: the projection can tell an absent axis from a zero one.
        self.proj = nn.Sequential(
            nn.Linear(d_model * axes + axes, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,                    # (B,P,C,d)
        sensor_id: torch.Tensor,                 # (B,C) long
        channel_mask: torch.Tensor | None = None,  # (B,C) True = channel exists
        n_sensors: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(sensor_tokens (B,P,S,d), sensor_mask (B,S))``."""
        B, P, C, d = tokens.shape
        if sensor_id.shape != (B, C):
            raise ValueError(f"sensor_id must be {(B, C)}, got {tuple(sensor_id.shape)}")
        S = int(n_sensors if n_sensors is not None else (sensor_id.max().item() + 1))
        if channel_mask is None:
            channel_mask = tokens.new_ones(B, C, dtype=torch.bool)
        channel_mask = channel_mask.bool()

        # Axis ordinal WITHIN each sensor, counted over LIVE channels only.
        #
        # This must ignore dead channels, because `stream_sensor_texts` gives an accel-only stream
        # sensor_id = [0,0,0,0,0,0] — absent-modality channels keep a *valid* index and rely on
        # channel_mask to pool to nothing. That is fine for per-channel tokens and wrong here: naive
        # counting would make sensor 0 own six channels. Counting live channels only, and routing
        # dead ones to a trash slot that is sliced off, makes the fold agree with the existing
        # sensor_id contract instead of demanding every caller change it.
        live = channel_mask.to(torch.int64)                                  # (B,C)
        member = F.one_hot(sensor_id, num_classes=S).to(torch.int64) * live.unsqueeze(-1)
        ordinal = member.cumsum(dim=1) - member                              # (B,C,S)
        axis_idx = (ordinal * member).sum(dim=2)                             # (B,C)
        axes_valid = ((axis_idx * live) < self.axes).all()
        if tokens.device.type == "cpu":
            if not bool(axes_valid):
                raise ValueError(
                    f"a sensor owns more than {self.axes} live channels; "
                    f"SensorFold assumes a fixed {self.axes}-axis triad"
                )
        else:
            # Avoid a device-wide synchronization in each of the four encoder passes per step.
            torch._assert_async(axes_valid, f"a sensor owns more than {self.axes} live channels")

        trash = S * self.axes
        slot = torch.where(channel_mask, sensor_id * self.axes + axis_idx,
                           torch.full_like(sensor_id, trash))                # (B,C)
        grid = tokens.new_zeros(B, P, trash + 1, d)
        valid = tokens.new_zeros(B, trash + 1)
        # Zero out absent channels BEFORE scatter so a dead axis contributes nothing to the concat.
        masked_tokens = tokens * channel_mask.view(B, 1, C, 1).to(tokens.dtype)
        grid.scatter_(2, slot.view(B, 1, C, 1).expand(B, P, C, d), masked_tokens)
        valid.scatter_(1, slot, channel_mask.to(tokens.dtype))
        grid, valid = grid[:, :, :trash], valid[:, :trash]

        flat = grid.reshape(B, P, S, self.axes * d)
        vflag = valid.reshape(B, S, self.axes).unsqueeze(1).expand(B, P, S, self.axes)
        fused = self.proj(torch.cat([flat, vflag.to(flat.dtype)], dim=-1))
        sensor_mask = valid.reshape(B, S, self.axes).any(dim=2)             # (B,S)
        return self.norm(fused), sensor_mask


class ConditioningProjection(nn.Module):
    """Learnable MLP over a frozen conditioning artifact, injected as a GATED residual.

    Gate bias inits negative so the conditioner is do-no-harm at initialisation and the model learns
    how much to use it — and so setting the bias very negative is a clean on/off ablation. Same
    contract as ``FactoredChannelTextFusion``'s gate, which this replaces at sensor granularity.
    """

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1,
                 gate_bias_init: float = -2.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias_init)

    def embed(self, artifact: torch.Tensor) -> torch.Tensor:
        """(B,S,in_dim) frozen artifact -> (B,S,d) learnable embedding."""
        return self.norm(self.mlp(artifact))

    def forward(
        self,
        sensor_tokens: torch.Tensor,     # (B,P,S,d)
        artifact: torch.Tensor,          # (B,S,in_dim) frozen
        valid: torch.Tensor | None = None,   # (B,S) True = artifact present
    ) -> torch.Tensor:
        return sensor_tokens + self.delta(sensor_tokens, artifact, valid)

    def delta(
        self,
        sensor_tokens: torch.Tensor,
        artifact: torch.Tensor,
        valid: torch.Tensor | None = None,
        *,
        embedded: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return only the gated residual, permitting parallel conditioners."""
        emb = self.embed(artifact) if embedded is None else embedded  # (B,S,d)
        if valid is not None:
            emb = emb * valid.unsqueeze(-1).to(emb.dtype)
        B, P, S, d = sensor_tokens.shape
        e = emb.unsqueeze(1).expand(B, P, S, d)
        gate = torch.sigmoid(self.gate(torch.cat([sensor_tokens, e], dim=-1)))
        return gate * e


class StructuredSensorConditioner(nn.Module):
    """Exact runtime acquisition facts, kept separate from natural-language semantics.

    Supported modalities are deliberately closed to accelerometer and gyroscope. Rates use a
    fixed physical reference rather than corpus-fitted normalization, so a new dataset requires no
    refitting. Gravity is four-way because it is inapplicable to gyroscopes rather than unknown.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, gate_bias_init: float = -2.0):
        super().__init__()
        self.modality = nn.Embedding(N_MODALITIES, d_model)
        self.gravity = nn.Embedding(N_GRAVITY_STATES, d_model)
        self.rate = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias_init))

    def embed(
        self,
        modality: torch.Tensor,
        gravity: torch.Tensor,
        rates_hz: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if modality.shape != gravity.shape or rates_hz.shape != (*modality.shape, 2):
            raise ValueError("structured sensor metadata has inconsistent shapes")
        if valid.shape != modality.shape:
            raise ValueError("structured sensor validity mask has inconsistent shape")

        # Ragged sensor batches use zero padding. Validate only live rows and replace padding with
        # legal neutral values before embedding so padding cannot trigger an error or NaN. Avoid
        # Python truth tests on CUDA tensors: those synchronize every training step. The loader
        # performs the full semantic validation on CPU; embedding lookup still fails loudly for a
        # live out-of-range id if a custom caller violates the contract.
        live_modality = modality[valid]
        live_gravity = gravity[valid]
        live_rates = rates_hz[valid]
        if modality.device.type == "cpu":
            if not torch.isfinite(live_rates).all() or not (live_rates > 0).all():
                raise ValueError("stored and effective sensor rates must be finite and positive")
            if not ((live_modality >= 0) & (live_modality < N_MODALITIES)).all():
                raise ValueError("unsupported modality: HALO currently accepts only accel and gyro")
            if not ((live_gravity >= 0) & (live_gravity < N_GRAVITY_STATES)).all():
                raise ValueError("invalid gravity-state id")
            accel = live_modality.eq(MODALITY_ACCELEROMETER)
            if not torch.where(accel, live_gravity.ne(GRAVITY_NOT_APPLICABLE),
                               live_gravity.eq(GRAVITY_NOT_APPLICABLE)).all():
                raise ValueError("gravity must be present/removed/unknown for accel and N/A for gyro")

        safe_modality = torch.where(valid, modality, torch.zeros_like(modality))
        safe_gravity = torch.where(valid, gravity, torch.full_like(gravity, GRAVITY_UNKNOWN))
        safe_rates = torch.where(
            valid.unsqueeze(-1), rates_hz, torch.full_like(rates_hz, RATE_REFERENCE_HZ),
        )
        # Only effective source rate carries physical bandwidth information. Stored rate is kept
        # in the transport contract for resampling provenance, but feeding it to the model creates
        # an avoidable dataset-identity shortcut. Compute the physical feature in float32 even
        # under autocast; ``delta`` casts the combined structured embedding back to the token
        # dtype before it enters the gated residual path.
        rate_features = torch.log2(
            safe_rates[..., 1:2].float() / RATE_REFERENCE_HZ,
        )
        rate_embedding = self.rate(rate_features)
        return self.norm(
            self.modality(safe_modality) + self.gravity(safe_gravity) + rate_embedding
        )

    def delta(
        self,
        sensor_tokens: torch.Tensor,
        modality: torch.Tensor,
        gravity: torch.Tensor,
        rates_hz: torch.Tensor,
        valid: torch.Tensor,
        *,
        embedded: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = (self.embed(modality, gravity, rates_hz, valid)
               if embedded is None else embedded).to(sensor_tokens.dtype)
        emb = emb * valid.unsqueeze(-1).to(emb.dtype)
        expanded = emb.unsqueeze(1).expand_as(sensor_tokens)
        gate = torch.sigmoid(self.gate(torch.cat([sensor_tokens, expanded], dim=-1)))
        return gate * expanded


class DescriptorHead(nn.Module):
    """Reconstruct a masked ``text_descriptor`` from the encoder's contextual sensor token.

    Predicts into the FROZEN text space (384-d SBERT), L2-normalised, so the caller can score by
    cosine against the batch's true descriptors. Predicting in text space rather than d_model is
    what makes the retrieval scoring meaningful — the candidates are real descriptors, not learned
    slots the head could co-adapt with.
    """

    def __init__(self, d_model: int, text_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, text_dim),
        )

    def forward(self, sensor_context: torch.Tensor) -> torch.Tensor:
        """(B,S,d) -> (B,S,text_dim), L2-normalised."""
        return F.normalize(self.net(sensor_context), dim=-1)


def descriptor_retrieval_loss(
    predicted: torch.Tensor,      # (..., text_dim) L2-normalised predictions
    targets: torch.Tensor,        # (..., text_dim) L2-normalised true descriptors
    temperature: float = 0.07,
    target_ids: torch.Tensor | None = None,
    candidate_descriptors: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """InfoNCE over the batch's DISTINCT descriptors. Returns ``(loss, top1_accuracy)``.

    Scored retrievally rather than by MSE: with ~20 distinct sensor descriptions in the corpus, an
    MSE head minimises by predicting the corpus mean, which looks like a falling loss while learning
    nothing. Ranking the true descriptor against its true competitors cannot be gamed that way.

    Duplicate descriptors within a batch are collapsed to unique rows first — otherwise two rows
    carrying the same sensor text become each other's negatives, and the task becomes unsolvable in
    exactly the cases where it should be easy.
    """
    if predicted.numel() == 0:
        zero = predicted.new_zeros(())
        return zero, zero
    if target_ids is not None:
        if candidate_descriptors is None:
            raise ValueError("candidate_descriptors are required with target_ids")
        if predicted.shape[:-1] != target_ids.shape:
            raise ValueError("target_ids must match predicted except for its descriptor dimension")
        # The caller's candidate table is already unique and target_ids index it directly. Scoring
        # every table row supplies true in-batch distractors and avoids a dynamic torch.unique.
        uniq = candidate_descriptors
        inverse = target_ids.reshape(-1)
        predicted = predicted.reshape(-1, predicted.shape[-1])
        valid = inverse.ge(0)
        if row_mask is not None:
            if row_mask.shape != target_ids.shape:
                raise ValueError("row_mask must have the same shape as target_ids")
            valid = valid & row_mask.reshape(-1)
        inverse = inverse.clamp_min(0)
    else:
        if row_mask is not None:
            raise ValueError("row_mask requires target_ids")
        uniq, inverse = torch.unique(targets, dim=0, return_inverse=True)
    logits = predicted @ uniq.t() / temperature                 # (N, U)
    if target_ids is None:
        loss = F.cross_entropy(logits, inverse)
        acc = (logits.argmax(dim=1) == inverse).float().mean()
        return loss, acc
    weights = valid.to(logits.dtype)
    denominator = weights.sum()
    active = denominator.gt(0).to(logits.dtype)
    per_row = F.cross_entropy(logits, inverse, reduction="none")
    loss = (per_row * weights).sum() / denominator.clamp(min=1.0) * active
    correct = logits.argmax(dim=1).eq(inverse).to(logits.dtype)
    acc = (correct * weights).sum() / denominator.clamp(min=1.0) * active
    return loss, acc
