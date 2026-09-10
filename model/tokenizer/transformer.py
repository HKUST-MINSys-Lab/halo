"""Dual-branch transformer (M3 port from legacy_code/model/transformer.py).

Temporal attention (per channel, physical-time RoPE over SECONDS not indices) +
cross-channel attention (per patch, channel-mask aware -> variable channel counts by
construction).

NOTE: this is FACTORIZED (dual-branch) attention, not the flat T×C attention sketched
in EVIDENCE_ENGINE.md §5.2.1 — ported as the battle-tested design; flat attention is
an ablation if ever needed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..blocks import AttentionSpec


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor):
    """positions (BC, P) in seconds, inv_freq (hd/2,) rad/s -> cos, sin (BC, 1, P, hd)."""
    ang = positions.to(inv_freq.dtype).unsqueeze(-1) * inv_freq        # (BC, P, hd/2)
    emb = torch.cat((ang, ang), dim=-1)                                # (BC, P, hd)
    return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)              # (BC, 1, P, hd)


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention over the temporal (patch) dimension.

    Processes each channel independently, allowing the model to learn
    temporal dependencies within each sensor channel.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_rope: bool = False,
        rope_min_period: float = 1.0,
        rope_max_period: float = 1000.0,
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_rope: Apply rotary position embedding indexed by physical time (seconds)
            rope_min_period: Period (s) of the fastest rotary component (finest patch spacing)
            rope_max_period: Period (s) of the slowest component (> max session span)
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # One GEMM for Q/K/V materially reduces launch overhead at HALO's short sequence lengths.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5

        # RoPE frequency band calibrated to HAR time-scales (seconds), not integer token
        # indices. Geometric periods: index 0 = fastest (min_period), last = slowest.
        self.use_rope = use_rope
        if use_rope:
            half = self.head_dim // 2
            k = torch.arange(half, dtype=torch.float32)
            periods = rope_min_period * (rope_max_period / rope_min_period) ** (k / max(half - 1, 1))
            self.register_buffer("rope_inv_freq", 2.0 * math.pi / periods, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply temporal self-attention.

        Args:
            x: Input tensor of shape (batch_size * num_channels, num_patches, d_model)
            mask: Optional attention mask of shape (num_patches, num_patches)
            key_padding_mask: Optional patch validity mask of shape (batch_channels, num_patches)
                             True = valid patch, False = padded patch

        Returns:
            Output tensor of same shape as input
        """
        batch_channels, num_patches, d_model = x.shape

        # Project to Q, K, V in one kernel.
        qkv = self.qkv_proj(x).view(
            batch_channels, num_patches, 3, self.num_heads, self.head_dim,
        )
        Q, K, V = qkv.unbind(dim=2)

        # Transpose to (batch_channels, num_heads, num_patches, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Rotary position embedding over physical time, applied to Q,K before attention.
        if self.use_rope and positions is not None:
            cos, sin = _rope_cos_sin(positions, self.rope_inv_freq)     # (BC,1,P,hd)
            cos, sin = cos.to(Q.dtype), sin.to(Q.dtype)
            Q = Q * cos + _rotate_half(Q) * sin
            K = K * cos + _rotate_half(K) * sin

        # Build an optional attention mask for SDPA (True = attend, False = masked).
        attn_mask = None
        if mask is not None:
            attn_mask = mask.bool()
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)                     # (BC,1,P,P)
        if key_padding_mask is not None:
            # Column mask: prevent attention TO padded patches -> (BC,1,1,P)
            key_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = key_mask if attn_mask is None else (attn_mask & key_mask)

        # Use PyTorch's fused attention (dispatches to Flash Attention / memory-efficient backend)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        # Reshape back and concatenate heads
        # (batch_channels, num_patches, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(batch_channels, num_patches, d_model)

        # Final projection
        output = self.out_proj(attn_output)

        return output

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Migrate pre-fusion checkpoints without weakening strict checkpoint loading."""
        fused_weight = prefix + "qkv_proj.weight"
        old_weights = [prefix + f"{name}_proj.weight" for name in ("q", "k", "v")]
        if fused_weight not in state_dict and all(name in state_dict for name in old_weights):
            state_dict[fused_weight] = torch.cat([state_dict.pop(name) for name in old_weights], dim=0)
            old_biases = [prefix + f"{name}_proj.bias" for name in ("q", "k", "v")]
            state_dict[prefix + "qkv_proj.bias"] = torch.cat(
                [state_dict.pop(name) for name in old_biases], dim=0,
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class CrossChannelSelfAttention(nn.Module):
    """
    Multi-head self-attention over the channel dimension.

    Allows different sensor channels to communicate and share information
    within each patch at the same temporal position.

    This enables the model to learn cross-channel dependencies and interactions
    (e.g., correlation between accelerometer and gyroscope).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-channel self-attention.

        Args:
            x: Input tensor of shape (batch_size * num_patches, num_channels, d_model)
            channel_mask: Optional channel validity mask of shape (batch_size * num_patches, num_channels)
                         True = valid channel, False = padded channel

        Returns:
            Output tensor of same shape as input
        """
        batch_patches, num_channels, d_model = x.shape

        if num_channels == 1:
            # Attention over a singleton set is identically its value vector: Q, K, scaling, and
            # softmax cannot affect the result. Sensor-homogeneous Phase-A batches hit this path for
            # every accelerometer-only stream, avoiding two thirds of this projection's GEMM work.
            value = F.linear(
                x,
                self.qkv_proj.weight[2 * d_model:],
                (self.qkv_proj.bias[2 * d_model:] if self.qkv_proj.bias is not None else None),
            ).view(batch_patches, 1, self.num_heads, self.head_dim).transpose(1, 2)
            if channel_mask is not None:
                value = value * channel_mask.view(batch_patches, 1, 1, 1).to(value.dtype)
            if self.training and self.dropout.p:
                # SDPA applies dropout to its (one-element) attention probability independently per
                # head. Using that same mask shape preserves the distribution, including 1/(1-p)
                # scaling, rather than applying elementwise dropout to the value coordinates.
                keep = F.dropout(
                    value.new_ones(batch_patches, self.num_heads, 1, 1),
                    p=self.dropout.p, training=True,
                )
                value = value * keep
            return self.out_proj(
                value.transpose(1, 2).reshape(batch_patches, 1, d_model)
            )

        qkv = self.qkv_proj(x).view(
            batch_patches, num_channels, 3, self.num_heads, self.head_dim,
        )
        Q, K, V = qkv.unbind(dim=2)

        # Transpose to (batch_patches, num_heads, num_channels, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Build attention mask (True = attend, False = masked).
        attn_mask = None
        if channel_mask is not None:
            # Column mask: prevent attention TO padded channels
            # (batch_patches, num_channels) -> (batch_patches, 1, 1, num_channels)
            attn_mask = channel_mask.unsqueeze(1).unsqueeze(2)

        if num_channels <= 4:
            # HALO's sensor-token design has at most two entries here. The generic masked SDPA
            # kernel is optimized for longer sequences and is substantially launch-bound at N=2;
            # the direct definition is mathematically identical and measured 4-5x faster on the
            # target RTX 4090. Keep SDPA below for legacy per-channel token sets.
            scores = (Q @ K.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            probabilities = scores.softmax(dim=-1)
            probabilities = F.dropout(
                probabilities,
                p=self.dropout.p,
                training=self.training,
            )
            attn_output = probabilities @ V
        else:
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale,
            )

        # Reshape back and concatenate heads
        # (batch_patches, num_channels, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(batch_patches, num_channels, d_model)

        # Final projection
        output = self.out_proj(attn_output)

        return output

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Migrate pre-fusion checkpoints without weakening strict checkpoint loading."""
        fused_weight = prefix + "qkv_proj.weight"
        old_weights = [prefix + f"{name}_proj.weight" for name in ("q", "k", "v")]
        if fused_weight not in state_dict and all(name in state_dict for name in old_weights):
            state_dict[fused_weight] = torch.cat([state_dict.pop(name) for name in old_weights], dim=0)
            old_biases = [prefix + f"{name}_proj.bias" for name in ("q", "k", "v")]
            state_dict[prefix + "qkv_proj.bias"] = torch.cat(
                [state_dict.pop(name) for name in old_biases], dim=0,
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two-layer MLP with GELU activation.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Input/output dimension
            dim_feedforward: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (*, d_model)

        Returns:
            Output tensor of same shape
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class DualBranchTransformerBlock(nn.Module):
    """
    Dual-branch transformer block with both temporal and cross-channel attention.

    Architecture:
    1. Temporal self-attention (patches within each channel)
    2. Cross-channel self-attention (channels within each patch)
    3. Feed-forward network

    Each step includes residual connections and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_rope: bool = False,
        rope_min_period: float = 1.0,
        rope_max_period: float = 1000.0,
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward network
            dropout: Dropout probability
            use_rope / rope_min_period / rope_max_period: physical-time RoPE for temporal attn
        """
        super().__init__()

        # Temporal attention (over patches within channel)
        self.temporal_attn = TemporalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            rope_min_period=rope_min_period,
            rope_max_period=rope_max_period,
        )

        # Cross-channel attention (over channels within patch)
        self.cross_channel_attn = CrossChannelSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through dual-branch transformer block.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            temporal_mask: Optional mask for temporal attention (num_patches, num_patches)
            channel_mask: Optional mask for channel attention (batch_size, num_channels)
                         True = valid channel, False = padded
            patch_padding_mask: Optional patch validity mask (batch_size, num_patches)
                               True = valid patch, False = padded

        Returns:
            Output tensor of shape (batch_size, num_patches, num_channels, d_model)
        """
        batch_size, num_patches, num_channels, d_model = x.shape

        # 1. Temporal self-attention (patches within each channel)
        x_temporal = x.permute(0, 2, 1, 3)
        x_temporal = x_temporal.reshape(batch_size * num_channels, num_patches, d_model)

        if patch_padding_mask is not None:
            temporal_key_padding_mask = patch_padding_mask.unsqueeze(1).expand(
                batch_size, num_channels, num_patches
            ).reshape(batch_size * num_channels, num_patches)
        else:
            temporal_key_padding_mask = None

        temporal_positions = None
        if positions is not None:
            temporal_positions = positions.unsqueeze(1).expand(
                batch_size, num_channels, num_patches
            ).reshape(batch_size * num_channels, num_patches)
        tmask = temporal_mask
        if tmask is not None and tmask.dim() == 3:
            tmask = tmask.unsqueeze(1).expand(
                batch_size, num_channels, num_patches, num_patches
            ).reshape(batch_size * num_channels, num_patches, num_patches)

        temporal_output = self.temporal_attn(
            x_temporal, tmask,
            key_padding_mask=temporal_key_padding_mask,
            positions=temporal_positions,
        )
        temporal_output = temporal_output.reshape(
            batch_size, num_channels, num_patches, d_model,
        ).permute(0, 2, 1, 3)

        # Residual and norm
        x = x + self.dropout(temporal_output)
        x = self.norm1(x)

        # 2. Cross-channel self-attention (channels within each patch)
        x_channel = x.reshape(batch_size * num_patches, num_channels, d_model)
        if channel_mask is not None:
            channel_mask_expanded = channel_mask.unsqueeze(1).expand(
                batch_size, num_patches, num_channels
            ).reshape(batch_size * num_patches, num_channels)
        else:
            channel_mask_expanded = None
        channel_output = self.cross_channel_attn(x_channel, channel_mask_expanded)
        channel_output = channel_output.reshape(
            batch_size, num_patches, num_channels, d_model,
        )

        # Residual and norm
        x = x + self.dropout(channel_output)
        x = self.norm2(x)

        # 3. Feed-forward network
        ff_output = self.feed_forward(x)

        # Residual and norm
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x

    def temporal_context(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply this block's temporal branch without cross-sensor mixing."""
        batch_size, num_patches, num_channels, d_model = x.shape
        temporal = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_channels, num_patches, d_model,
        )
        padding = None
        if patch_padding_mask is not None:
            padding = patch_padding_mask.unsqueeze(1).expand(
                batch_size, num_channels, num_patches,
            ).reshape(batch_size * num_channels, num_patches)
        temporal_positions = None
        if positions is not None:
            temporal_positions = positions.unsqueeze(1).expand(
                batch_size, num_channels, num_patches,
            ).reshape(batch_size * num_channels, num_patches)
        tmask = temporal_mask
        if tmask is not None and tmask.dim() == 3:
            tmask = tmask.unsqueeze(1).expand(
                batch_size, num_channels, num_patches, num_patches,
            ).reshape(batch_size * num_channels, num_patches, num_patches)
        output = self.temporal_attn(
            temporal, tmask, key_padding_mask=padding, positions=temporal_positions,
        ).reshape(batch_size, num_channels, num_patches, d_model).permute(0, 2, 1, 3)
        return self.norm1(x + self.dropout(output))


class DualBranchTransformer(nn.Module):
    """
    Dual-branch transformer with both temporal and cross-channel attention.

    Processes patches with two types of attention:
    1. Temporal attention: Models dependencies across patches within each channel
    2. Cross-channel attention: Models interactions between channels within each patch

    Input:  (batch, patches, channels, d_model)
    Process: Temporal attention → Cross-channel attention → FFN (per block)
    Output: (batch, patches, channels, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_rope: bool = False,
        rope_min_period: float = 1.0,
        rope_max_period: float = 1000.0,
    ):
        """
        Args:
            d_model: Feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward networks
            dropout: Dropout probability
            use_rope / rope_min_period / rope_max_period: physical-time RoPE for temporal attn
        """
        super().__init__()

        # Stack of dual-branch transformer blocks
        self.layers = nn.ModuleList([
            DualBranchTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_rope=use_rope,
                rope_min_period=rope_min_period,
                rope_max_period=rope_max_period,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        return_layer_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Process input through dual-branch transformer.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            temporal_mask: Optional (num_patches, num_patches) or per-sample (B,P,P) mask
            channel_mask: Optional mask for channel attention (batch_size, num_channels)
            patch_padding_mask: Optional patch validity mask (batch_size, num_patches)
            positions: Optional per-patch physical times (batch_size, num_patches) for RoPE

        Returns:
            Output tensor of shape (batch_size, num_patches, num_channels, d_model)
        """
        layer_states = []
        for layer in self.layers:
            x = layer(x, temporal_mask, channel_mask, patch_padding_mask, positions=positions)
            if return_layer_states:
                layer_states.append(x)

        return (x, tuple(layer_states)) if return_layer_states else x

    def retrieval_context(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sensor-isolated temporal representation used by retrieval and its Phase-A loss."""
        if not self.layers:
            return x
        return self.layers[0].temporal_context(
            x, temporal_mask=temporal_mask,
            patch_padding_mask=patch_padding_mask, positions=positions,
        )


class TemporalTrunk(nn.Module):
    """Temporal-only representation trunk: a few pre-norm self-attention layers per sensor.

    WHAT CHANGED FROM :class:`DualBranchTransformer`, AND WHY
    ---------------------------------------------------------
    The dual-branch trunk interleaves a cross-sensor branch so an accelerometer token can read its
    companion gyroscope. That branch is ~40% of the trunk's parameters, and it fuses sensors at a
    point where the model knows nothing about the sensors beyond their descriptions.

    In the evidence engine that fusion happens later and better: the mixer attends over every
    retrieved row, and rows from the same recording share a co-membership group id, so cross-sensor
    relations are formed there — alongside the labels and descriptions that give them meaning. The
    trunk's job is narrower: turn one sensor's patch sequence into one comparable vector. Temporal
    attention is what that needs.

    Consequences worth stating plainly. Each sensor is encoded in isolation, so the retrieval
    feature for a wrist accelerometer is identical whether or not a gyroscope was present on the
    same device — which is the invariance the cross-configuration claim wants, and which the
    dual-branch trunk did not have. It also means a sensor pair carries no joint representation
    until the mixer sees it.

    Input/output ``(batch, patches, sensors, d_model)``, matching the dual-branch trunk so the
    encoder can hold either.
    """

    def __init__(
        self,
        spec: AttentionSpec,
        num_layers: int = 3,
        use_rope: bool = True,
        rope_min_period: float = 1.0,
        rope_max_period: float = 1000.0,
    ):
        super().__init__()
        self.spec = spec
        self.num_layers = int(num_layers)
        self.attn_norm = nn.ModuleList(
            [nn.LayerNorm(spec.d_model) for _ in range(num_layers)]
        )
        self.attn = nn.ModuleList([
            TemporalSelfAttention(
                d_model=spec.d_model,
                num_heads=spec.n_heads,
                dropout=spec.dropout,
                use_rope=use_rope,
                rope_min_period=rope_min_period,
                rope_max_period=rope_max_period,
            )
            for _ in range(num_layers)
        ])
        self.ffn_norm = nn.ModuleList(
            [nn.LayerNorm(spec.d_model) for _ in range(num_layers)]
        )
        self.ffn = nn.ModuleList([
            FeedForward(spec.d_model, spec.dim_feedforward, spec.dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(spec.dropout)
        self.out_norm = nn.LayerNorm(spec.d_model)

    def _flatten(self, x: torch.Tensor):
        batch, patches, sensors, dim = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(batch * sensors, patches, dim)
        return flat, (batch, patches, sensors, dim)

    @staticmethod
    def _expand(per_sample: Optional[torch.Tensor], sensors: int) -> Optional[torch.Tensor]:
        """(B, P[, P]) -> (B*S, P[, P]); sensors share the patch grid of their window."""
        if per_sample is None:
            return None
        batch, patches = per_sample.shape[0], per_sample.shape[1]
        trailing = per_sample.shape[2:]
        return (per_sample.unsqueeze(1)
                .expand(batch, sensors, patches, *trailing)
                .reshape(batch * sensors, patches, *trailing))

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        stop_after: Optional[int] = None,
        return_layer_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """``channel_mask`` is accepted and ignored: sensors never mix in this trunk.

        ``stop_after`` runs a prefix of the stack, which is how the retrieval representation is
        read out at a chosen depth without a second forward pass.
        """
        flat, (batch, patches, sensors, dim) = self._flatten(x)
        padding = self._expand(patch_padding_mask, sensors)
        pos = self._expand(positions, sensors)
        mask = temporal_mask
        if mask is not None and mask.dim() == 3:
            mask = self._expand(mask, sensors)

        depth = self.num_layers if stop_after is None else min(int(stop_after), self.num_layers)
        layer_states = []
        for layer in range(depth):
            attended = self.attn[layer](
                self.attn_norm[layer](flat), mask, key_padding_mask=padding, positions=pos,
            )
            flat = flat + self.dropout(attended)
            flat = flat + self.dropout(self.ffn[layer](self.ffn_norm[layer](flat)))
            if return_layer_states:
                state = self.out_norm(flat).reshape(
                    batch, sensors, patches, dim,
                ).permute(0, 2, 1, 3)
                layer_states.append(state)
        flat = self.out_norm(flat)
        output = flat.reshape(batch, sensors, patches, dim).permute(0, 2, 1, 3)
        return (output, tuple(layer_states)) if return_layer_states else output

    def retrieval_context(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Kept for interface parity with the dual-branch trunk.

        There, retrieval had to read a shallow prefix to stay sensor-isolated. Here the whole trunk
        is sensor-isolated, so retrieval reads the full depth and this is the ordinary forward.
        """
        return self.forward(
            x, temporal_mask=temporal_mask,
            patch_padding_mask=patch_padding_mask, positions=positions,
        )
