"""The config-conditional set encoder (M3) — completes "the tokenizer, broadly".

Assembly (build plan M3; EVIDENCE_ENGINE.md §5.2.1):

    device-frame IMU patches ── fixed filterbank ──> axis features ── SensorFold(xyz) ─┐
                           └── continuous kernels ──> dense xyz sensor CNN ────────────┤
                                                                                       ▼
                                                                         sensor tokens (B,P,S,d)
                                                      │
    sensor description ──frozen LM──> gated projection ──────> acquisition semantics
                                                      │
    JEPA token_mask ──> learned [MASK] token (BEFORE conditioning, so the model knows WHICH
                      sensor is hidden — masked-sensor modeling needs the identity
                      of the thing it must reconstruct)
                                                      │
    DualBranchTransformer: temporal attention with PHYSICAL-TIME RoPE (seconds, never
    patch index) + cross-sensor attention (sensor-mask aware). Sensors carry no positional
    index; identity is text, so sensor count/order are free.
                                                      ▼
    {tokens (B,P,S,d) · per_patch (B,P,d) · pooled (B,d)}

Configuration conditioning is the sensor text (for example, "watch accelerometer on the left
wrist"). Measured sensor statistics remain bank metadata, but are excluded from the representation
trunk because held-out devices have no fitted statistics and the training values expose source ID.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .channel_text import ChannelTextFusion, FactoredChannelTextFusion, TokenTextEncoder
from .continuous_kernel import ContinuousKernelTokenizer
from .multispan_kernel import MultiSpanKernelTokenizer
from .filterbank import PhysicalFilterbankTokenizer
from .sensor_tokens import ConditioningProjection, DescriptorHead, SensorFold
from ..blocks import AttentionSpec
from .transformer import DualBranchTransformer, TemporalTrunk

# RoPE periods in seconds. The reference grid is one second; 0.5 seconds preserves headroom for
# the retained multi-resolution ablation. The slowest period exceeds any training context.
ROPE_MIN_PERIOD_S = 0.5
ROPE_MAX_PERIOD_S = 600.0


class SetTokenizerEncoder(nn.Module):
    """signal patches + channel TEXT + physical time -> representation.

    Permutation- and count-invariant over channels by construction (identity via text,
    attention via masks); rate- and patch-duration-agnostic via the physical-Hz
    filterbank + physical-time RoPE.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        text_model: str = "all-MiniLM-L6-v2",
        frontend: str = "fixed",                  # 'fixed'|'learnable'|'continuous'
        trunk: str = "dual",                     # representation trunk: 'dual'|'temporal'
        descriptor_prediction: bool = True,      # build the Phase-A descriptor-prediction head
        text_conditioning: str = "per_channel",  # 'per_channel' (legacy) | 'factored' (role+sensor)
        token_granularity: str = "channel",      # direct-constructor compatibility; CLI uses sensor
        sensor_bias_dim: int = 14,               # 7 standardized values + 7 support bits
        use_sensor_bias_conditioning: bool = False,
        use_sensor_isolated_retrieval: bool = False,
        gate_bias_init: float = -2.0,             # factored: negative => identity lightly injected @ init
        use_duration_embedding: bool = False,
        duration_min_seconds: float = 0.4,
        duration_max_seconds: float = 1.5,
        duration_gate_init: float = 0.1,
        num_resolutions: int = 2,
        rope_min_period: float = ROPE_MIN_PERIOD_S,
        **filterbank_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_duration_embedding = bool(use_duration_embedding)
        self.duration_min_seconds = float(duration_min_seconds)
        self.duration_max_seconds = float(duration_max_seconds)
        if not 0 < self.duration_min_seconds < self.duration_max_seconds:
            raise ValueError("duration bounds must satisfy 0 < min < max")
        self.num_resolutions = int(num_resolutions)
        if self.num_resolutions < 2:
            raise ValueError("num_resolutions must be at least 2")
        if text_conditioning not in ("per_channel", "factored"):
            raise ValueError("text_conditioning must be 'per_channel' or 'factored'")
        self.text_conditioning = text_conditioning
        if token_granularity not in ("channel", "sensor"):
            raise ValueError("token_granularity must be 'channel' or 'sensor'")
        self.token_granularity = token_granularity
        self.sensor_bias_dim = int(sensor_bias_dim)
        self.use_sensor_bias_conditioning = bool(use_sensor_bias_conditioning)
        if trunk not in ("dual", "temporal"):
            raise ValueError("trunk must be 'dual' or 'temporal'")
        self.trunk = trunk
        # A temporal trunk never mixes sensors, so its FINAL layer is already the
        # sensor-isolated representation retrieval wants. Reading a shallow prefix instead —
        # which is all `use_sensor_isolated_retrieval` ever bought — would now cost depth for
        # an isolation the architecture already guarantees.
        self.use_sensor_isolated_retrieval = (
            False if trunk == "temporal" else bool(use_sensor_isolated_retrieval)
        )
        if frontend not in {"fixed", "learnable", "continuous", "multispan"}:
            raise ValueError("frontend must be 'fixed', 'learnable', 'continuous' or 'multispan'")
        if frontend in ("continuous", "multispan") and token_granularity != "sensor":
            raise ValueError(
                "the continuous frontends jointly model each xyz triad and therefore require "
                "token_granularity='sensor'"
            )
        self.frontend = frontend
        # Attribute stays named `filterbank` for checkpoint compatibility.
        if frontend in ("continuous", "multispan"):
            # dft_size is padding capacity for the physical FFT, not a continuous-kernel setting.
            continuous_kwargs = dict(filterbank_kwargs)
            for physical_only in (
                "dft_size", "center_shift_fraction", "bandwidth_factor_max",
                "compression_gain_max", "filter_shape_min", "filter_shape_max",
                "adaptive_gate_init",
            ):
                continuous_kwargs.pop(physical_only, None)
            if frontend == "continuous":
                self.filterbank = ContinuousKernelTokenizer(d_model=d_model, **continuous_kwargs)
            else:
                # Span-grouped kernels emitting a physical-time token grid; the collate's patch
                # grid only supplies the contiguous window (see multispan_kernel.py).
                self.filterbank = MultiSpanKernelTokenizer(d_model=d_model, **continuous_kwargs)
        else:
            self.filterbank = PhysicalFilterbankTokenizer(
                learnable=frontend == "learnable", d_model=d_model, **filterbank_kwargs,
            )
        self.text_encoder = TokenTextEncoder(model_name=text_model)   # frozen, cached
        if token_granularity == "sensor":
            # DESIGN OF RECORD: a sensor is one modality triad. Folding xyz into one token makes the
            # role text ("x"/"y"/"z") redundant — axis identity becomes positional inside the token —
            # and sensor descriptions get a learnable projection over frozen text embeddings.
            # The continuous front end already fuses a fixed xyz triad with a dense CNN.  The
            # physical filterbank still emits per-axis rows and needs the separate fold.
            self.sensor_fold = (
                None if frontend in ("continuous", "multispan")
                else SensorFold(d_model=d_model, dropout=dropout)
            )
            self.descriptor_proj = ConditioningProjection(384, d_model, dropout=dropout,
                                                          gate_bias_init=gate_bias_init)
            if self.use_sensor_bias_conditioning:
                self.bias_proj = ConditioningProjection(
                    self.sensor_bias_dim, d_model, dropout=dropout,
                    gate_bias_init=gate_bias_init,
                )
            # PipelineAModel disables and freezes this head in the reference recipe, so building
            # it unconditionally put ~132k parameters with no path to any loss into every
            # checkpoint. It stays default-on so existing checkpoints load strictly; a model
            # that will never run the ablation should pass descriptor_prediction=False.
            self.descriptor_head = (
                DescriptorHead(d_model, text_dim=384, dropout=dropout)
                if descriptor_prediction else None
            )
            # PipelineAModel disables this explicit ablation in the reference recipe. The module
            # remains in the state dict for checkpoint compatibility, but an inactive head should
            # not build an unused prediction graph on every encoder call.
            self.descriptor_prediction_enabled = bool(descriptor_prediction)
        if token_granularity == "sensor":
            # Sensor tokens receive their two conditioning paths after xyz folding below.  Keeping a
            # channel fusion module here would create trainable parameters that can never participate
            # in a sensor-granularity forward pass.
            self.fusion = None
        elif text_conditioning == "factored":
            # per-channel ROLE text + per-sensor IDENTITY text (docs/design/TEXT_CONDITIONING.md)
            self.fusion = FactoredChannelTextFusion(d_model=d_model, text_dim=384,
                                                    gate_bias_init=gate_bias_init)
        else:
            self.fusion = ChannelTextFusion(d_model=d_model, text_dim=384)
        if self.use_duration_embedding:
            self.duration_proj = nn.Sequential(
                nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_model),
            )
            if not 0.0 < duration_gate_init < 1.0:
                raise ValueError("duration_gate_init must be in (0, 1)")
            gate_logit = torch.logit(torch.tensor(float(duration_gate_init)))
            self.duration_gate_logit = nn.Parameter(gate_logit)
        # MAE-style small-random init (NOT zeros: a zero mask token is symmetric across
        # masked positions and starts with no signal to distinguish "hidden here").
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        if trunk == "temporal":
            if dim_feedforward % d_model:
                raise ValueError(
                    "a temporal trunk states its width as one AttentionSpec, so "
                    f"dim_feedforward ({dim_feedforward}) must be a multiple of d_model "
                    f"({d_model})"
                )
            self.attention_spec = AttentionSpec(
                d_model=d_model, n_heads=num_heads,
                ffn_mult=dim_feedforward // d_model, dropout=dropout,
            )
            self.transformer = TemporalTrunk(
                self.attention_spec, num_layers=num_layers, use_rope=True,
                rope_min_period=rope_min_period, rope_max_period=ROPE_MAX_PERIOD_S,
            )
        else:
            self.attention_spec = None
            self.transformer = DualBranchTransformer(
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_rope=True,
                rope_min_period=rope_min_period,
                rope_max_period=ROPE_MAX_PERIOD_S,
            )
        # Runtime-only acceleration hook. The trainer may install a compiled bound ``forward`` here;
        # keeping the actual module untouched preserves ordinary state_dict keys and eager eval loads.
        self._compiled_transformer_forward = None

    def _add_duration_embedding(
        self,
        tokens: torch.Tensor,
        patch_durations: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Condition every patch token on its physical temporal support.

        Centre-time RoPE says *where* a token is located. This embedding says how much physical
        time the token summarizes, which is essential when several patch grids share one attention
        sequence. Log time makes multiplicative scale changes (0.5 -> 1.0 -> 2.0 seconds) evenly
        spaced; the bounded gate introduces the new signal gently without blocking gradients.
        """
        if not self.use_duration_embedding:
            return tokens
        if patch_durations is None:
            raise ValueError("patch_durations are required when duration embedding is enabled")
        if patch_durations.shape != tokens.shape[:2]:
            raise ValueError(
                f"patch_durations must have shape {tuple(tokens.shape[:2])}, "
                f"got {tuple(patch_durations.shape)}"
            )
        lo = math.log(self.duration_min_seconds)
        span = math.log(self.duration_max_seconds) - lo
        valid_duration = patch_durations > 0
        log_d = patch_durations.clamp(min=self.duration_min_seconds).log()
        normalized = (2.0 * (log_d - lo) / span - 1.0).clamp(-1.0, 1.0)
        duration_emb = self.duration_proj(normalized.unsqueeze(-1).to(tokens.dtype))
        duration_emb = duration_emb * valid_duration.unsqueeze(-1)
        return tokens + torch.sigmoid(self.duration_gate_logit) * duration_emb.unsqueeze(2)

    def _resolution_weights(
        self,
        resolution_ids: torch.Tensor,
        patch_weight: torch.Tensor,
        patch_durations: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return duration weights per patch/resolution and their per-resolution sums."""
        if resolution_ids.shape != patch_weight.shape:
            raise ValueError(
                f"resolution_ids must have shape {tuple(patch_weight.shape)}, "
                f"got {tuple(resolution_ids.shape)}"
            )
        valid = resolution_ids.ge(0) & resolution_ids.lt(self.num_resolutions) \
            & patch_weight.gt(0)
        one_hot = F.one_hot(
            resolution_ids.clamp(0, self.num_resolutions - 1),
            num_classes=self.num_resolutions,
        ).to(patch_weight.dtype)
        duration = (
            patch_durations.to(patch_weight.dtype)
            if patch_durations is not None else patch_weight
        )
        weights = one_hot * (valid.to(patch_weight.dtype) * duration).unsqueeze(-1)
        return weights, weights.sum(dim=1)

    # ------------------------------------------------------------------ shareable stages
    # The forward splits into (tokenize · encode_texts · encode) so a training step that
    # needs masked and clean views computes the filterbank and the
    # text embeddings ONCE and only re-runs the cheap transformer tail.

    def tokenize(self, patches, sampling_rate_hz, patch_len_samples, channel_mask=None,
                 source_rate_hz=None, sensor_id=None, n_sensors=None) -> torch.Tensor:
        """Signal tokens identical across masked/clean views.

        The physical filterbank returns ``(B,P,C,d)`` axis tokens.  The continuous front end
        jointly projects fixed xyz slots and returns ``(B,P,S,d)`` sensor tokens directly.

        Spectral analysis is an explicit FP32 island. The learned projection runs in the caller's
        autocast context, so CUDA training gets Tensor Core FP16 without exposing FFT power or
        log-energy standardization to FP16's limited dynamic range.
        """
        with torch.amp.autocast(patches.device.type, enabled=False):
            token_in = self.filterbank.analyze(
                patches.float(), sampling_rate_hz, patch_len_samples,
                source_rate_hz=source_rate_hz,
            )
        if getattr(self.filterbank, "emits_sensor_tokens", False):
            return self.filterbank.project(
                token_in, sensor_id=sensor_id, channel_mask=channel_mask,
                n_sensors=n_sensors,
            )
        return self.filterbank.project(token_in)

    def analyze(self, patches, sampling_rate_hz, patch_len_samples, source_rate_hz=None):
        """Parameter-free (fixed arm) physical feature, shareable across encoder copies."""
        return self.filterbank.analyze(patches, sampling_rate_hz, patch_len_samples,
                                       source_rate_hz=source_rate_hz)

    def project_tokens(self, token_in, *, sensor_id=None, channel_mask=None,
                       n_sensors=None) -> torch.Tensor:
        """Apply THIS encoder's learnable filterbank projection to a shared analysis."""
        if getattr(self.filterbank, "emits_sensor_tokens", False):
            return self.filterbank.project(
                token_in, sensor_id=sensor_id, channel_mask=channel_mask,
                n_sensors=n_sensors,
            )
        return self.filterbank.project(token_in)

    def encode_texts(
        self, channel_texts: Sequence[Sequence[str]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B lists of C strings) -> token embeddings (B, C, S, 384) + mask (B, C, S).

        Dedupes to UNIQUE strings before the (cached) encoder + before the pad/stack —
        a batch has at most (#streams × 6) distinct descriptions, not B×C, so the
        per-step assembly cost is bounded by variety, not batch size.
        """
        embs_u, masks_u, gather = self.encode_texts_unique(channel_texts, device)
        B, C = gather.shape
        S = embs_u.shape[1]
        embs = embs_u.index_select(0, gather.reshape(-1)).reshape(B, C, S, -1)
        masks = masks_u.index_select(0, gather.reshape(-1)).reshape(B, C, S)
        return embs, masks

    def encode_texts_unique(
        self, channel_texts: Sequence[Sequence[str]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode unique strings and return ``(embeddings, masks, inverse_ids)``.

        ``inverse_ids`` has shape (B,C). Keeping text unique through the learnable pooler avoids
        repeating identical attention/MLP work for every batch row. With pooler dropout disabled (or
        in eval), values and gradients are exactly the dense computation; training shares one dropout
        realization among rows carrying the same description.
        """
        B, C = len(channel_texts), len(channel_texts[0])
        for texts in channel_texts:
            assert len(texts) == C, "all samples in a batch must have the same channel count"
        flat = [t for texts in channel_texts for t in texts]
        unique = list(dict.fromkeys(flat))
        embs_u, masks_u = self.text_encoder.encode(unique, device=device)   # (U, S, 384)
        idx = {t: i for i, t in enumerate(unique)}
        gather = torch.tensor([idx[t] for t in flat], device=device, dtype=torch.long).reshape(B, C)
        return embs_u, masks_u, gather

    def encode_texts_factored(self, role_texts, sensor_texts, device):
        """Encode the two factored text sources with the same frozen LM.

        role_texts:   B lists of C role strings   -> (role_embs (B,C,S,384), role_masks (B,C,S))
        sensor_texts: B ragged lists of sensor strings -> padded
                      (sensor_embs (B,N_max,S,384), sensor_masks (B,N_max,S)).
        """
        role_embs, role_masks = self.encode_texts(role_texts, device)
        if not sensor_texts or any(len(texts) == 0 for texts in sensor_texts):
            raise ValueError("factored text conditioning requires at least one sensor description "
                             "per sample")
        B, n_max = len(sensor_texts), max(map(len, sensor_texts))
        flat = [text for texts in sensor_texts for text in texts]
        unique = list(dict.fromkeys(flat))
        embs_u, masks_u = self.text_encoder.encode(unique, device=device)
        lookup = {text: i for i, text in enumerate(unique)}
        S, D = embs_u.shape[1], embs_u.shape[2]
        sensor_embs = embs_u.new_zeros(B, n_max, S, D)
        sensor_masks = torch.zeros(B, n_max, S, dtype=torch.bool, device=device)
        for b, texts in enumerate(sensor_texts):
            gather = torch.tensor([lookup[text] for text in texts], device=device)
            sensor_embs[b, :len(texts)] = embs_u.index_select(0, gather)
            sensor_masks[b, :len(texts)] = masks_u.index_select(0, gather)
        return role_embs, role_masks, sensor_embs, sensor_masks

    def encode_texts_factored_unique(self, role_texts, sensor_texts, device):
        """Unique-token equivalent of :meth:`encode_texts_factored` for the hot path.

        Returns unique role token rows + (B,C) inverse IDs and unique sensor token rows + a padded
        (B,N_max) inverse-ID table. ``-1`` marks a padded sensor slot that no channel may reference.
        """
        role_embs, role_masks, role_ids = self.encode_texts_unique(role_texts, device)
        if not sensor_texts or any(len(texts) == 0 for texts in sensor_texts):
            raise ValueError("factored text conditioning requires at least one sensor description "
                             "per sample")
        B, n_max = len(sensor_texts), max(map(len, sensor_texts))
        flat = [text for texts in sensor_texts for text in texts]
        unique = list(dict.fromkeys(flat))
        sensor_embs, sensor_masks = self.text_encoder.encode(unique, device=device)
        lookup = {text: i for i, text in enumerate(unique)}
        # Build the whole table on CPU, then ONE host-to-device copy. The previous form issued a
        # separate `torch.tensor(..., device=device)` per batch row — about 144 tiny transfers per
        # episode on this corpus, and a visible share of the `aten::copy_` calls in the profile.
        ids_cpu = torch.full((B, n_max), -1, dtype=torch.long)
        for b, texts in enumerate(sensor_texts):
            ids_cpu[b, :len(texts)] = torch.tensor([lookup[text] for text in texts],
                                                   dtype=torch.long)
        sensor_text_ids = ids_cpu.to(device, non_blocking=True)
        return (role_embs, role_masks, role_ids,
                sensor_embs, sensor_masks, sensor_text_ids)

    def encode_sensor_descriptors_unique(self, sensor_texts, device):
        """Frozen pooled sensor descriptors plus a padded per-sample inverse-ID table.

        This is the sensor-granularity text path.  Axis-role strings and token-level LM outputs are
        intentionally absent because xyz identity is encoded by the fixed fold order and the sensor
        conditioner consumes only the frozen mean-pooled descriptor.
        """
        if not sensor_texts or any(len(texts) == 0 for texts in sensor_texts):
            raise ValueError("sensor granularity requires at least one sensor description per sample")
        B, n_max = len(sensor_texts), max(map(len, sensor_texts))
        flat = [text for texts in sensor_texts for text in texts]
        unique = list(dict.fromkeys(flat))
        descriptors = self.text_encoder.encode_pooled(unique, device=device)
        lookup = {text: i for i, text in enumerate(unique)}
        # Build the small ragged inverse table on the host and transfer it once. Creating one CUDA
        # tensor per sample generated 2*B pageable H2D copies on every two-view training step (515
        # transfers at batch 256 in the operator trace), which became a launch bottleneck at the
        # larger batches the 4090 can comfortably hold.
        sensor_text_ids = torch.tensor(
            [
                [*(lookup[text] for text in texts), *([-1] * (n_max - len(texts)))]
                for texts in sensor_texts
            ],
            dtype=torch.long,
            device=device,
        )
        return descriptors, sensor_text_ids

    def encode(
        self,
        sensor_tokens: torch.Tensor,                 # (B, P, C, d) from tokenize()
        text_embs: torch.Tensor,                     # (B, C, S, 384) from encode_texts()
        text_masks: torch.Tensor,
        positions: torch.Tensor,                     # (B, P) patch-center times in SECONDS
        patch_durations: Optional[torch.Tensor] = None,  # (B, P) true temporal support in seconds
        resolution_ids: Optional[torch.Tensor] = None,   # (B, P) 0=short, 1=long, -1=padding
        token_mask: Optional[torch.Tensor] = None,   # (B, P, C) True = hide for JEPA
        channel_mask: Optional[torch.Tensor] = None, # (B, C) True = channel exists
        patch_padding_mask: Optional[torch.Tensor] = None,  # (B, P) True = real patch
        # --- factored text conditioning (only when text_conditioning='factored') ---
        sensor_text_embs: Optional[torch.Tensor] = None,   # (B, N_sensors, S, 384)
        sensor_text_masks: Optional[torch.Tensor] = None,  # (B, N_sensors, S)
        sensor_descriptors: Optional[torch.Tensor] = None, # unique (U,384) or dense (B,N,384)
        sensor_id: Optional[torch.Tensor] = None,          # (B, C) long
        role_text_ids: Optional[torch.Tensor] = None,      # (B,C), when text rows stay unique
        sensor_text_ids: Optional[torch.Tensor] = None,    # (B,N_sensors), -1 = padding
        # --- sensor granularity (token_granularity='sensor') ---
        sensor_bias: Optional[torch.Tensor] = None,        # (B,N_sensors,sensor_bias_dim) frozen
        descriptor_mask: Optional[torch.Tensor] = None,    # (B,N_sensors) True = hide the descriptor
        return_retrieval_tokens: bool = True,
        retrieval_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.token_granularity == "sensor":
            return self._encode_sensor(
                sensor_tokens, text_embs, text_masks, positions,
                patch_durations=patch_durations, resolution_ids=resolution_ids,
                token_mask=token_mask, channel_mask=channel_mask,
                patch_padding_mask=patch_padding_mask,
                sensor_text_embs=sensor_text_embs, sensor_text_masks=sensor_text_masks,
                sensor_descriptors=sensor_descriptors,
                sensor_id=sensor_id, sensor_text_ids=sensor_text_ids,
                sensor_bias=sensor_bias, descriptor_mask=descriptor_mask,
                return_retrieval_tokens=return_retrieval_tokens,
                retrieval_only=retrieval_only,
            )
        if retrieval_only:
            raise ValueError("retrieval_only is implemented only for sensor-granularity tokens")
        B, P, C, _ = sensor_tokens.shape

        # JEPA masking BEFORE fusion: the [MASK] token then receives its channel's text
        # identity, so the encoder knows *which* channel it must reconstruct.
        tokens = sensor_tokens
        if token_mask is not None:
            tokens = torch.where(
                token_mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens
            )
        tokens = self._add_duration_embedding(tokens, patch_durations)
        if self.text_conditioning == "factored":
            if sensor_text_embs is None or sensor_text_masks is None or sensor_id is None:
                raise ValueError("factored text_conditioning requires sensor_text_embs / "
                                 "sensor_text_masks / sensor_id; use encode_texts_factored()")
            sensor_id = sensor_id.to(device=tokens.device, dtype=torch.long)
            if sensor_id.shape != (B, C):
                raise ValueError(f"sensor_id must have shape {(B, C)}, got {tuple(sensor_id.shape)}")
            sensor_count = (sensor_text_ids.shape[1] if sensor_text_ids is not None
                            else sensor_text_embs.shape[1])
            if sensor_id.numel() and (
                int(sensor_id.min().item()) < 0
                or int(sensor_id.max().item()) >= sensor_count
            ):
                raise ValueError("sensor_id contains an index without a corresponding sensor description")
            valid_sensor = (sensor_text_ids.ge(0) if sensor_text_ids is not None
                            else sensor_text_masks.any(dim=2))
            selected_sensor_valid = torch.gather(valid_sensor, 1, sensor_id)
            if not bool(selected_sensor_valid.all()):
                raise ValueError(
                    "sensor_id points to a padded/missing sensor description for at least one sample"
                )
            # `text_embs`/`text_masks` carry the ROLE tokens in the factored path.
            tokens = self.fusion(tokens, text_embs, text_masks,
                                 sensor_text_embs, sensor_text_masks, sensor_id,
                                 role_text_ids=role_text_ids, sensor_text_ids=sensor_text_ids)
        else:
            tokens = self.fusion(tokens, text_embs, text_masks)

        transformer_forward = self._compiled_transformer_forward or self.transformer
        h = transformer_forward(
            tokens,
            channel_mask=channel_mask,
            patch_padding_mask=patch_padding_mask,
            positions=positions,
        )                                                                # (B,P,C,d)

        # Pooling respects the masks: absent channels / padded patches contribute nothing.
        weights = h.new_ones(B, P, C)
        if channel_mask is not None:
            weights = weights * channel_mask.view(B, 1, C)
        if patch_padding_mask is not None:
            weights = weights * patch_padding_mask.view(B, P, 1)
        denom_c = weights.sum(dim=2, keepdim=True).clamp(min=1.0)
        per_patch = (h * weights.unsqueeze(-1)).sum(dim=2) / denom_c.squeeze(2).unsqueeze(-1)
        patch_w = weights.amax(dim=2)                                    # (B,P) patch validity
        if resolution_ids is None:
            temporal_w = patch_w
            if patch_durations is not None:
                temporal_w = temporal_w * patch_durations.to(per_patch.dtype)
            pooled = (per_patch * temporal_w.unsqueeze(-1)).sum(dim=1) \
                / temporal_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        else:
            # Equal resolution weight: twelve short tokens must not outweigh four long
            # tokens merely because their temporal grid is denser.
            # Within a resolution, weight each patch by its REPRESENTED duration so a partial tail
            # patch (a short window-crop remainder) contributes proportionally, not as a full-length
            # patch (F1). When every patch in a resolution has equal duration (e.g. eval's evenly
            # divided window) the constant factor cancels — this reduces to the uniform mean.
            scale_w, denom = self._resolution_weights(
                resolution_ids, patch_w, patch_durations,
            )
            summaries = torch.einsum("bpd,bps->bsd", per_patch, scale_w) \
                / denom.clamp(min=1e-6).unsqueeze(-1)
            active = (denom > 0).to(per_patch.dtype)
            pooled = (summaries * active.unsqueeze(-1)).sum(dim=1) \
                / active.sum(dim=1, keepdim=True).clamp(min=1.0)

        return {"tokens": h, "per_patch": per_patch, "pooled": pooled}

    # ------------------------------------------------------------------ sensor granularity
    def _encode_sensor(
        self,
        channel_tokens: torch.Tensor,                # (B,P,C,d) from tokenize()
        text_embs: torch.Tensor,                     # unused at sensor granularity (role text)
        text_masks: torch.Tensor,
        positions: torch.Tensor,
        patch_durations=None,
        resolution_ids=None,
        token_mask: Optional[torch.Tensor] = None,   # (B,P,S) True = hide this sensor-patch
        channel_mask: Optional[torch.Tensor] = None, # (B,C)
        patch_padding_mask: Optional[torch.Tensor] = None,
        sensor_text_embs: Optional[torch.Tensor] = None,   # (B,N,S_tok,384)
        sensor_text_masks: Optional[torch.Tensor] = None,  # (B,N,S_tok)
        sensor_descriptors: Optional[torch.Tensor] = None, # unique (U,384) or dense (B,N,384)
        sensor_id: Optional[torch.Tensor] = None,          # (B,C)
        sensor_text_ids: Optional[torch.Tensor] = None,
        sensor_bias: Optional[torch.Tensor] = None,        # (B,N,sensor_bias_dim)
        descriptor_mask: Optional[torch.Tensor] = None,    # (B,N) True = hide the descriptor
        return_retrieval_tokens: bool = True,
        retrieval_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        """The design-of-record forward: fold to sensor tokens, condition, attend, pool.

        Order is load-bearing. The JEPA [MASK] is applied to the FOLDED sensor token but BEFORE the
        descriptor and bias are injected, so a masked sensor still carries its identity and the model
        knows *which* sensor it must reconstruct — the same principle as the per-channel path, one
        granularity up. Reversing it would ask the model to reconstruct an anonymous hole.
        """
        if sensor_id is None or sensor_descriptors is None:
            raise ValueError("sensor granularity requires sensor_id and sensor_descriptors")
        B, P, token_count, d = channel_tokens.shape
        # `sensor_text_embs` may be UNIQUE rows (U,S_tok,384) with `sensor_text_ids` (B,N) indexing
        # them — the hot path dedupes text, so a batch holds at most a few distinct descriptions.
        # The per-sample sensor count therefore comes from the id table, never from the embedding.
        unique_text = sensor_text_ids is not None and sensor_descriptors.dim() == 2
        N = sensor_text_ids.shape[1] if sensor_text_ids is not None else sensor_descriptors.shape[1]
        sensor_id = sensor_id.to(device=channel_tokens.device, dtype=torch.long)

        if getattr(self.filterbank, "emits_sensor_tokens", False):
            if token_count != N:
                raise ValueError(
                    f"continuous frontend emitted {token_count} sensor slots, expected {N}"
                )
            if channel_mask is None:
                raise ValueError("continuous sensor projection requires channel_mask")
            tokens = channel_tokens
            folded_mask = self.filterbank.sensor_presence(sensor_id, channel_mask, N)
        else:
            if self.sensor_fold is None:
                raise RuntimeError("physical axis tokens require SensorFold")
            tokens, folded_mask = self.sensor_fold(
                channel_tokens, sensor_id, channel_mask, n_sensors=N,
            )
        # A sensor is present iff it has a live channel AND a real (non-padding) description.
        sensor_present = folded_mask
        if sensor_text_ids is not None:
            sensor_present = sensor_present & sensor_text_ids.ge(0).to(sensor_present.device)

        if token_mask is not None:
            if token_mask.shape != (B, P, N):
                raise ValueError(f"sensor-granularity token_mask must be {(B, P, N)}, "
                                 f"got {tuple(token_mask.shape)}")
            tokens = torch.where(token_mask.unsqueeze(-1),
                                 self.mask_token.expand_as(tokens), tokens)

        # A centre-time encoding cannot distinguish overlapping 0.5, 1.0 and 1.5 second tokens
        # whose centres coincide. Add the physical span before descriptor conditioning and temporal
        # attention so all scales can contextualize one another explicitly.
        tokens = self._add_duration_embedding(tokens, patch_durations)

        # Legacy dual-trunk checkpoints can request a shallow sensor-isolated retrieval branch.
        # The active temporal trunk forces this option off: its final output below is already
        # sensor-isolated, carries all temporal layers, and includes descriptor conditioning.
        retrieval_tokens = None
        if return_retrieval_tokens and self.use_sensor_isolated_retrieval:
            retrieval_tokens = self.transformer.retrieval_context(
                tokens, patch_padding_mask=patch_padding_mask, positions=positions,
            )

        if unique_text:
            # (U,384) -> (B,N,384). Padding slots (-1) clamp to row 0 and are zeroed by
            # `sensor_present` downstream, so they contribute nothing.
            descriptor = sensor_descriptors.index_select(
                0, sensor_text_ids.clamp_min(0).reshape(-1),
            )
            descriptor = descriptor.reshape(B, N, -1)
        else:
            descriptor = sensor_descriptors                                      # (B,N,384)

        if retrieval_only:
            if not self.use_sensor_isolated_retrieval or retrieval_tokens is None:
                raise ValueError(
                    "retrieval_only requires use_sensor_isolated_retrieval=True"
                )
            # Phase B stores this layer-0 temporal row. Descriptor conditioning, cross-sensor
            # attention, later temporal layers, pooling, and the Phase-A descriptor head cannot
            # affect it, so running them only burns compute and advertises parameters with no path
            # to the loss. Return exactly the metadata live_sensor_rows consumes.
            return {
                "retrieval_tokens": retrieval_tokens,
                "sensor_present": sensor_present,
                "descriptor": descriptor,
            }

        descriptor_visible = sensor_present
        if descriptor_mask is not None:
            descriptor_visible = descriptor_visible & ~descriptor_mask.to(sensor_present.device)
        tokens = self.descriptor_proj(tokens, descriptor, descriptor_visible)
        if self.use_sensor_bias_conditioning:
            if sensor_bias is None:
                raise ValueError("this legacy checkpoint requires sensor_bias conditioning")
            if sensor_bias.shape != (B, N, self.sensor_bias_dim):
                raise ValueError(
                    f"sensor_bias must be {(B, N, self.sensor_bias_dim)}, got {tuple(sensor_bias.shape)}"
                )
            tokens = self.bias_proj(tokens, sensor_bias.to(tokens.dtype), sensor_present)

        transformer_forward = self._compiled_transformer_forward or self.transformer
        h = transformer_forward(tokens, channel_mask=sensor_present,
                                patch_padding_mask=patch_padding_mask, positions=positions)
        if return_retrieval_tokens and retrieval_tokens is None:
            retrieval_tokens = h

        weights = h.new_ones(B, P, N) * sensor_present.view(B, 1, N).to(h.dtype)
        if patch_padding_mask is not None:
            weights = weights * patch_padding_mask.view(B, P, 1).to(h.dtype)
        denom = weights.sum(dim=2, keepdim=True).clamp(min=1.0)
        per_patch = (h * weights.unsqueeze(-1)).sum(dim=2) / denom.squeeze(2).unsqueeze(-1)
        patch_w = weights.amax(dim=2)
        if resolution_ids is None:
            temporal_w = patch_w
            if patch_durations is not None:
                temporal_w = temporal_w * patch_durations.to(per_patch.dtype)
            pooled = (per_patch * temporal_w.unsqueeze(-1)).sum(dim=1) \
                / temporal_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        else:
            scale_w, denom_r = self._resolution_weights(
                resolution_ids, patch_w, patch_durations,
            )
            sums = torch.einsum("bpd,bpr->brd", per_patch, scale_w)
            means = sums / denom_r.clamp(min=1e-6).unsqueeze(-1)
            present = (denom_r > 0).to(per_patch.dtype)
            pooled = (means * present.unsqueeze(-1)).sum(dim=1) \
                / present.sum(dim=1, keepdim=True).clamp(min=1.0)

        # Per-sensor context for descriptor prediction. Match session pooling: physical-duration
        # weighting within each grid and equal weight across active grids. Otherwise the denser short
        # grid and a partial tail patch would dominate descriptor supervision.
        pw = (patch_padding_mask.to(h.dtype) if patch_padding_mask is not None
              else h.new_ones(B, P))
        if resolution_ids is None:
            context_w = pw
            if patch_durations is not None:
                context_w = context_w * patch_durations.to(h.dtype)
            sensor_context = (h * context_w.view(B, P, 1, 1)).sum(dim=1) \
                / context_w.sum(dim=1).clamp(min=1e-6).view(B, 1, 1)
        else:
            context_w = scale_w.to(h.dtype)
            context_denom = context_w.sum(dim=1)
            context_by_resolution = torch.einsum("bpsd,bpr->brsd", h, context_w) \
                / context_denom.clamp(min=1e-6).view(B, self.num_resolutions, 1, 1)
            active_context = (context_denom > 0).to(h.dtype)
            sensor_context = (
                context_by_resolution * active_context.view(B, self.num_resolutions, 1, 1)
            ).sum(dim=1) / active_context.sum(dim=1).clamp(min=1.0).view(B, 1, 1)

        descriptor_pred = (self.descriptor_head(sensor_context)
                           if self.descriptor_prediction_enabled
                           and self.descriptor_head is not None else None)
        return {"tokens": h, "retrieval_tokens": retrieval_tokens,
                "per_patch": per_patch, "pooled": pooled,
                "sensor_context": sensor_context, "sensor_present": sensor_present,
                "descriptor": descriptor,
                "descriptor_pred": descriptor_pred}

    # ------------------------------------------------------------------------ forward
    def forward(
        self,
        patches: torch.Tensor,                       # (B, P, S, C) zero-padded native-rate
        sampling_rate_hz,                            # scalar | (B,)
        patch_len_samples,                           # scalar | (B,) | (B,P) true N
        channel_texts: Sequence[Sequence[str]],      # per_channel: B lists of C descriptions;
                                                     # factored: B lists of C ROLE strings
        positions: torch.Tensor,                     # (B, P) patch-center times in SECONDS
        patch_durations: Optional[torch.Tensor] = None,
        resolution_ids: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,   # (B, P, C) True = hide for JEPA
        channel_mask: Optional[torch.Tensor] = None, # (B, C) True = channel exists
        patch_padding_mask: Optional[torch.Tensor] = None,  # (B, P) True = real patch
        sensor_texts: Optional[Sequence[Sequence[str]]] = None,  # factored: B lists of N_sensor strings
        sensor_id: Optional[torch.Tensor] = None,                # factored: (B, C) long
        source_rate_hz=None,                         # scalar | (B,) acquisition bandwidth bound
        sensor_bias: Optional[torch.Tensor] = None,  # sensor granularity: (B, N, sensor_bias_dim)
        descriptor_mask: Optional[torch.Tensor] = None,  # sensor granularity: (B, N)
        return_retrieval_tokens: bool = True,
        retrieval_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        s_embs = s_masks = None
        if self.token_granularity == "sensor":
            # Sensor granularity always uses the factored text sources, but only the SENSOR half:
            # role text is redundant once xyz is folded into one token (axis identity is positional).
            if sensor_texts is None or sensor_id is None:
                raise ValueError("sensor granularity requires sensor_texts and sensor_id")
            device = patches.device
            sensor_descriptors, sensor_text_ids = self.encode_sensor_descriptors_unique(
                sensor_texts, device,
            )
            grid = None
            if getattr(self.filterbank, "emits_token_grid", False):
                # The frontend defines the token grid: centre times, spans and group ids replace
                # the collate's patch metadata for RoPE, the duration embedding and pooling.
                if token_mask is not None:
                    raise ValueError("JEPA token masking is not defined on the multi-span token grid")
                if resolution_ids is not None:
                    live = (resolution_ids >= 0 if patch_padding_mask is None
                            else patch_padding_mask.bool())
                    # Raw patches must partition one recording, not repeat it at several scales.
                    single_grid = ((resolution_ids == 0) | ~live).all()
                    if patches.device.type == "cpu":
                        if not bool(single_grid):
                            raise ValueError("multi-span frontend requires a single input patch grid")
                    else:
                        torch._assert_async(single_grid,
                                            "multi-span frontend requires a single input patch grid")
                grid = self.filterbank.token_grid(
                    patches, sampling_rate_hz, patch_len_samples, source_rate_hz=source_rate_hz,
                    patch_mask=patch_padding_mask, sensor_id=sensor_id, channel_mask=channel_mask,
                    n_sensors=sensor_text_ids.shape[1],
                )
                sensor_tokens = grid["tokens"]
                positions, patch_durations = grid["positions"], grid["durations"]
                resolution_ids, patch_padding_mask = grid["resolution_ids"], grid["token_mask"]
            else:
                sensor_tokens = self.tokenize(
                    patches, sampling_rate_hz, patch_len_samples,
                    channel_mask=channel_mask, source_rate_hz=source_rate_hz,
                    sensor_id=sensor_id, n_sensors=sensor_text_ids.shape[1],
                )
            encoded = self._encode_sensor(
                sensor_tokens, None, None, positions,
                patch_durations=patch_durations, resolution_ids=resolution_ids,
                token_mask=token_mask, channel_mask=channel_mask,
                patch_padding_mask=patch_padding_mask,
                sensor_descriptors=sensor_descriptors,
                sensor_id=sensor_id, sensor_text_ids=sensor_text_ids,
                sensor_bias=sensor_bias, descriptor_mask=descriptor_mask,
                return_retrieval_tokens=return_retrieval_tokens,
                retrieval_only=retrieval_only,
            )
            if grid is not None:
                encoded["token_grid"] = {
                    key: grid[key] for key in ("positions", "durations", "resolution_ids", "token_mask")
                }
            return encoded
        sensor_tokens = self.tokenize(
            patches, sampling_rate_hz, patch_len_samples,
            channel_mask=channel_mask, source_rate_hz=source_rate_hz,
        )
        device = sensor_tokens.device
        if retrieval_only:
            raise ValueError("retrieval_only is implemented only for sensor-granularity tokens")
        if self.text_conditioning == "factored":
            if sensor_texts is None or sensor_id is None:
                raise ValueError("factored text_conditioning requires sensor_texts and sensor_id")
            text_embs, text_masks, role_ids, s_embs, s_masks, sensor_text_ids = \
                self.encode_texts_factored_unique(channel_texts, sensor_texts, device)
        else:
            text_embs, text_masks = self.encode_texts(channel_texts, device)
        return self.encode(sensor_tokens, text_embs, text_masks, positions,
                           patch_durations=patch_durations, resolution_ids=resolution_ids,
                           token_mask=token_mask, channel_mask=channel_mask,
                           patch_padding_mask=patch_padding_mask,
                           sensor_text_embs=s_embs, sensor_text_masks=s_masks, sensor_id=sensor_id,
                           role_text_ids=(role_ids if self.text_conditioning == "factored" else None),
                           sensor_text_ids=(sensor_text_ids
                                            if self.text_conditioning == "factored" else None))
