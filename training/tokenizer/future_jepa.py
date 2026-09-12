"""Multi-horizon, physically grounded JEPA primitives.

The student encodes only tokens ending at or before a sampled context boundary. A narrow predictor
uses metadata-only future queries to predict normalized EMA-teacher states. A small decoder maps the
predicted states to fixed physical measurements. All interval decisions are made in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# One-second bins are resolvable by every supported token grid, including the continuous frontend's
# one-second rows. Narrower first bins made an objective structurally inactive for that encoder.
DEFAULT_HORIZON_BINS_SECONDS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0))


@dataclass(frozen=True)
class FutureTargetPlan:
    """Dense plan over the collate token grid.

    ``context_mask`` and ``target_mask`` have shape ``(B, P)``. A true context entry is the only
    kind of signal token the student may attend to. Target entries identify teacher states and
    metadata-only predictor queries; they are never student signal inputs.
    """

    context_end: torch.Tensor
    context_mask: torch.Tensor
    target_mask: torch.Tensor
    horizon_seconds: torch.Tensor
    eligible: torch.Tensor


def truncate_patch_lengths_at_time(
    patch_lengths: torch.Tensor,
    patch_starts: torch.Tensor,
    rates: torch.Tensor,
    context_end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a raw patch partition containing exactly the prefix ending at ``context_end``."""

    if patch_lengths.shape != patch_starts.shape or patch_lengths.ndim != 2:
        raise ValueError("patch lengths and starts must have matching (B,P) shapes")
    batch = patch_lengths.shape[0]
    if rates.reshape(-1).shape[0] != batch or context_end.reshape(-1).shape[0] != batch:
        raise ValueError("rates and context_end must contain one value per recording")
    available = torch.floor(
        (context_end.reshape(batch, 1) - patch_starts).clamp_min(0)
        * rates.reshape(batch, 1) + 1e-5
    ).long()
    lengths = torch.minimum(patch_lengths.long().clamp_min(0), available.clamp_min(0))
    return lengths, lengths.gt(0)


def extract_interval_patches(
    window: torch.Tensor,
    total_samples: torch.Tensor,
    rates: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract physical intervals from contiguous recordings for frozen target analysis."""

    if window.ndim != 3 or starts.shape != ends.shape or starts.ndim != 2:
        raise ValueError("expected window (B,T,C) and interval metadata (B,P)")
    batch, intervals = starts.shape
    if window.shape[0] != batch or capacity <= 0:
        raise ValueError("interval batch and positive capacity are required")
    rates = rates.reshape(batch, 1)
    total_samples = total_samples.long().reshape(batch, 1)
    first = torch.floor(starts.clamp_min(0) * rates + 1e-5).long()
    last = torch.ceil(ends.clamp_min(0) * rates - 1e-5).long()
    first = torch.minimum(first, total_samples)
    last = torch.minimum(last, total_samples)
    lengths = (last - first).clamp(min=0, max=capacity)
    offsets = torch.arange(capacity, device=window.device).view(1, 1, capacity)
    indices = first.unsqueeze(-1) + offsets
    sample_valid = offsets < lengths.unsqueeze(-1)
    indices = indices.clamp(min=0, max=max(window.shape[1] - 1, 0))
    source = window.unsqueeze(1).expand(batch, intervals, window.shape[1], window.shape[2])
    values = source.gather(
        2, indices.unsqueeze(-1).expand(batch, intervals, capacity, window.shape[2]),
    )
    return values * sample_valid.unsqueeze(-1).to(values.dtype), lengths


def pack_selected_interval_patches(
    window: torch.Tensor,
    total_samples: torch.Tensor,
    rates: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    selected: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack selected physical intervals and retain a dense patch-to-slot lookup.

    Future JEPA selects only a few physical anchors from a potentially dense multi-span grid.
    Extracting every interval wastes memory in proportion to the entire grid. This helper packs
    only selected rows to ``Q=max(selected.sum(1))`` and returns the slot for every original patch
    index, so predictor queries can gather the matching frozen physical target exactly.
    """

    if selected.shape != starts.shape or selected.dtype != torch.bool:
        raise ValueError("selected must be a boolean mask matching interval metadata")
    batch, patches = selected.shape
    counts = selected.sum(dim=1)
    packed_width = max(int(counts.max().item()), 1)
    packed_indices = torch.zeros(
        batch, packed_width, dtype=torch.long, device=selected.device,
    )
    packed_valid = torch.arange(packed_width, device=selected.device).unsqueeze(0) < counts[:, None]
    # A stable sort places selected patch indices first while preserving physical-time order.
    order = torch.argsort((~selected).long(), dim=1, stable=True)
    packed_indices.copy_(order[:, :packed_width])
    packed_starts = starts.gather(1, packed_indices)
    packed_ends = ends.gather(1, packed_indices)
    values, lengths = extract_interval_patches(
        window, total_samples, rates, packed_starts, packed_ends, capacity,
    )
    values = values * packed_valid[:, :, None, None].to(values.dtype)
    lengths = lengths * packed_valid.to(lengths.dtype)

    slot_by_patch = torch.zeros(batch, patches, dtype=torch.long, device=selected.device)
    slots = torch.arange(packed_width, device=selected.device).expand(batch, -1)
    slot_by_patch.scatter_(1, packed_indices, slots * packed_valid.long())
    return values, lengths, slot_by_patch, packed_valid


def make_future_target_plan(
    patch_starts: torch.Tensor,
    patch_ends: torch.Tensor,
    patch_valid: torch.Tensor,
    resolution_ids: torch.Tensor | None = None,
    *,
    context_fraction: tuple[float, float] = (0.4, 0.7),
    horizon_bins_seconds: Sequence[tuple[float, float]] = DEFAULT_HORIZON_BINS_SECONDS,
    generator: torch.Generator | None = None,
) -> FutureTargetPlan:
    """Sample context prefixes and aligned future targets without signal overlap.

    One physical anchor is sampled from each feasible horizon bin. Every resolution token that
    contains that anchor and starts at or after the context boundary becomes a target. Thus
    resolutions supervise the same future event while losses can still balance them separately.
    Windows unable to provide both honest context and a future target are marked ineligible.
    """

    if patch_starts.shape != patch_ends.shape or patch_starts.shape != patch_valid.shape:
        raise ValueError("patch starts, ends, and validity must have matching (B,P) shapes")
    if patch_starts.ndim != 2:
        raise ValueError("future planning expects (B,P) patch metadata")
    lo, hi = map(float, context_fraction)
    if not 0.0 < lo <= hi < 1.0:
        raise ValueError("context fractions must satisfy 0 < min <= max < 1")
    bins = tuple((float(a), float(b)) for a, b in horizon_bins_seconds)
    if not bins or any(not math.isfinite(a) or not math.isfinite(b) or a < 0 or b <= a
                       for a, b in bins):
        raise ValueError("horizon bins must be non-empty increasing positive-width intervals")
    if resolution_ids is not None and resolution_ids.shape != patch_valid.shape:
        raise ValueError("resolution_ids must match the (B,P) token grid")

    device = patch_starts.device
    starts = patch_starts.float()
    ends = patch_ends.float()
    valid = patch_valid.bool() & torch.isfinite(starts) & torch.isfinite(ends) & (ends > starts)
    if resolution_ids is not None:
        valid &= resolution_ids.ge(0)
    batch, patches = valid.shape
    context_end = starts.new_zeros(batch)
    context_mask = torch.zeros_like(valid)
    target_mask = torch.zeros_like(valid)
    horizon = starts.new_zeros(batch, patches)
    if not batch or not patches:
        return FutureTargetPlan(context_end, context_mask, target_mask, horizon,
                                torch.zeros(batch, dtype=torch.bool, device=device))

    # Build all context candidates in one tensor. Random candidates retain priority; sorted token
    # boundaries are deterministic fallbacks for short or unusually staggered grids. Duplicated
    # boundaries are harmless and avoid a Python-side unique/sort for every recording.
    live_count = valid.sum(dim=1)
    inf, neg_inf = float("inf"), float("-inf")
    window_start = starts.masked_fill(~valid, inf).min(dim=1).values
    window_end = ends.masked_fill(~valid, neg_inf).max(dim=1).values
    duration = window_end - window_start
    row_viable = (live_count >= 2) & torch.isfinite(duration) & duration.gt(0)
    fractions = lo + (hi - lo) * torch.rand(
        batch, 8, generator=generator, device=device,
    )
    random_boundaries = window_start[:, None] + fractions * duration[:, None]
    interior_boundaries = torch.cat((
        starts.masked_fill(~valid, inf), ends.masked_fill(~valid, inf),
    ), dim=1).sort(dim=1).values
    boundary_candidates = torch.cat((random_boundaries, interior_boundaries), dim=1)
    candidate_valid = torch.cat((
        row_viable[:, None].expand(-1, random_boundaries.shape[1]),
        row_viable[:, None]
        & interior_boundaries.gt(window_start[:, None])
        & interior_boundaries.lt(window_end[:, None]),
    ), dim=1)
    # Existence needs only the earliest end and latest start. Avoid a (B, 2P+8, P)
    # temporary when dense multi-span grids contain hundreds of intervals.
    earliest_end = ends.masked_fill(~valid, inf).min(dim=1).values
    latest_start = starts.masked_fill(~valid, neg_inf).max(dim=1).values
    observed_any = earliest_end[:, None].le(boundary_candidates + 1e-7)
    future_any = latest_start[:, None].ge(boundary_candidates - 1e-7)
    feasible = candidate_valid & observed_any & future_any
    has_boundary = feasible.any(dim=1)
    first_feasible = feasible.to(torch.int8).argmax(dim=1)
    boundary = boundary_candidates.gather(1, first_feasible[:, None]).squeeze(1)
    boundary = torch.where(has_boundary, boundary, torch.zeros_like(boundary))
    context_mask = (
        valid & ends.le(boundary[:, None] + 1e-7) & has_boundary[:, None]
    )
    future_mask = (
        valid & starts.ge(boundary[:, None] - 1e-7) & has_boundary[:, None]
    )
    context_end.copy_(boundary)

    centers = 0.5 * (starts + ends)
    center_horizons = centers - boundary[:, None]
    if resolution_ids is None:
        resolution_groups: tuple[int | None, ...] = (None,)
    else:
        resolution_groups = tuple(
            int(group) for group in torch.unique(resolution_ids[valid]).tolist()
            if int(group) >= 0
        )

    def select_covering(anchor: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Select the closest covering interval per row and resolution."""

        covering = (
            future_mask & rows[:, None]
            & starts.le(anchor[:, None] + 1e-7)
            & ends.gt(anchor[:, None] + 1e-7)
        )
        chosen = torch.zeros_like(valid)
        for group in resolution_groups:
            candidates = covering if group is None else covering & resolution_ids.eq(group)
            distance = (centers - anchor[:, None]).abs().masked_fill(~candidates, inf)
            index = distance.argmin(dim=1)
            present = candidates.any(dim=1)
            # An absent group has argmin index zero; it must not erase a previous
            # group's selection at that position.
            keep = chosen.gather(1, index[:, None]) | present[:, None]
            chosen.scatter_(1, index[:, None], keep)
        return chosen

    # Draw one unique physical center per horizon and select every resolution that covers it.
    # The only loops are over the fixed number of horizons and resolutions, never over the batch.
    for bin_lo, bin_hi in bins:
        choices = (
            future_mask & center_horizons.ge(bin_lo) & center_horizons.lt(bin_hi)
        )
        sorted_centers = centers.masked_fill(~choices, inf).sort(dim=1).values
        unique_center = torch.isfinite(sorted_centers)
        if patches > 1:
            unique_center[:, 1:] &= sorted_centers[:, 1:].ne(sorted_centers[:, :-1])
        unique_count = unique_center.sum(dim=1)
        # A uniform draw over unique centers prevents denser resolution grids from gaining extra
        # anchor probability. Rows without choices are masked before they can affect the plan.
        draw = torch.floor(
            torch.rand(batch, generator=generator, device=device)
            * unique_count.clamp_min(1).to(starts.dtype)
        ).long()
        unique_rank = unique_center.long().cumsum(dim=1) - 1
        picked = unique_center & unique_rank.eq(draw[:, None])
        anchor = torch.where(picked, sorted_centers, torch.zeros_like(sorted_centers)).sum(dim=1)
        rows = unique_count.gt(0)
        covered = select_covering(anchor, rows)
        newly_selected = covered & ~target_mask
        target_mask |= covered
        horizon = torch.where(
            newly_selected, (anchor - boundary)[:, None], horizon,
        )

    # A feasible future outside the configured bins remains useful. Align all available
    # resolutions to its earliest physical center.
    fallback_rows = has_boundary & ~target_mask.any(dim=1)
    fallback_anchor = centers.masked_fill(~future_mask, inf).min(dim=1).values
    fallback_anchor = torch.where(
        fallback_rows, fallback_anchor, torch.zeros_like(fallback_anchor),
    )
    fallback = select_covering(fallback_anchor, fallback_rows)
    target_mask |= fallback
    horizon = torch.where(
        fallback, (fallback_anchor - boundary)[:, None], horizon,
    )
    eligible = has_boundary & context_mask.any(dim=1) & target_mask.any(dim=1)

    return FutureTargetPlan(
        context_end=context_end,
        context_mask=context_mask,
        target_mask=target_mask,
        horizon_seconds=horizon,
        eligible=eligible,
    )


def normalized_teacher_target(layer_states: Sequence[torch.Tensor], top_k: int = 2) -> torch.Tensor:
    """Average the last ``top_k`` EMA layers and normalize each token without learned state."""

    if not layer_states:
        raise ValueError("teacher target needs at least one encoder layer")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    shape = layer_states[-1].shape
    if any(state.shape != shape for state in layer_states):
        raise ValueError("teacher layer states must have matching shapes")
    target = torch.stack(tuple(layer_states)[-top_k:], dim=0).mean(dim=0).float()
    return F.layer_norm(target, (target.shape[-1],)).detach()


class FuturePredictor(nn.Module):
    """Narrow decoder whose queries contain metadata but no future signal values."""

    def __init__(
        self,
        d_model: int,
        predictor_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        descriptor_dim: int = 384,
        dropout: float = 0.1,
        max_resolutions: int = 8,
    ):
        super().__init__()
        if predictor_dim <= 0 or predictor_dim % num_heads:
            raise ValueError("predictor_dim must be positive and divisible by num_heads")
        if num_layers <= 0:
            raise ValueError("future predictor needs at least one layer")
        self.d_model = int(d_model)
        self.predictor_dim = int(predictor_dim)
        self.context_proj = nn.Linear(d_model, predictor_dim)
        self.descriptor_proj = nn.Linear(descriptor_dim, predictor_dim, bias=False)
        self.numeric_proj = nn.Sequential(
            nn.Linear(3, predictor_dim), nn.GELU(), nn.Linear(predictor_dim, predictor_dim),
        )
        self.resolution_embedding = nn.Embedding(max_resolutions, predictor_dim)
        self.query_role = nn.Parameter(torch.randn(predictor_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=4 * predictor_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, d_model)

    def forward(
        self,
        context_states: torch.Tensor,
        context_valid: torch.Tensor,
        target_mask: torch.Tensor,
        positions: torch.Tensor,
        durations: torch.Tensor,
        resolution_ids: torch.Tensor,
        horizon_seconds: torch.Tensor,
        context_end: torch.Tensor,
        sensor_descriptors: torch.Tensor,
        sensor_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return packed predictions, ``(patch,sensor)`` indices, and validity.

        Context has shape ``(B,P,S,D)`` and is flattened only after invalid future rows are excluded
        as keys/values. Target queries are packed to the batch maximum; an ineligible row receives
        one harmless dummy query so PyTorch never sees an all-padded decoder sequence.
        """

        if context_states.ndim != 4:
            raise ValueError("context states must have shape (B,P,S,D)")
        batch, patches, sensors, dim = context_states.shape
        if dim != self.d_model:
            raise ValueError(f"expected context width {self.d_model}, got {dim}")
        if context_valid.shape != (batch, patches, sensors):
            raise ValueError("context_valid must have shape (B,P,S)")
        if target_mask.shape != (batch, patches):
            raise ValueError("target_mask must have shape (B,P)")
        for name, tensor in {
            "positions": positions,
            "durations": durations,
            "resolution_ids": resolution_ids,
            "horizon_seconds": horizon_seconds,
        }.items():
            if tensor.shape != (batch, patches):
                raise ValueError(f"{name} must have shape (B,P)")
        if context_end.reshape(-1).shape[0] != batch:
            raise ValueError("context_end must contain one value per recording")
        if sensor_descriptors.shape[:2] != (batch, sensors):
            raise ValueError("sensor_descriptors must have shape (B,S,descriptor_dim)")
        if sensor_present.shape != (batch, sensors):
            raise ValueError("sensor_present must have shape (B,S)")

        # Use one fixed query slot for every patch/sensor pair. The validity mask identifies the
        # sampled targets. This avoids per-example packing, ``.item()`` GPU synchronizations, and
        # ragged decoder shapes; at the intended 22 patches x at most two sensors the padding is
        # small and predictable.
        query_grid = target_mask.unsqueeze(2) & sensor_present.unsqueeze(1)
        patch_index = torch.arange(patches, device=context_states.device).repeat_interleave(sensors)
        sensor_index = torch.arange(sensors, device=context_states.device).repeat(patches)
        indices = torch.stack((patch_index, sensor_index), dim=-1).unsqueeze(0).expand(batch, -1, -1)
        query_valid = query_grid.flatten(1)

        patch_index = indices[..., 0]
        sensor_index = indices[..., 1]
        target_duration = durations.gather(1, patch_index)
        target_horizon = horizon_seconds.gather(1, patch_index)
        target_resolution = resolution_ids.gather(1, patch_index).clamp(
            min=0, max=self.resolution_embedding.num_embeddings - 1,
        )
        descriptor = sensor_descriptors.gather(
            1, sensor_index.unsqueeze(-1).expand(-1, -1, sensor_descriptors.shape[-1]),
        )

        # Queries ask "what follows this observed prefix?" rather than "what belongs at this
        # absolute place in the recording?"  Bounded log-time features preserve physical units
        # across recordings with different lengths without exposing target absolute position.
        log_scale = math.log1p(8.0)
        target_horizon_feature = torch.tanh(torch.log1p(target_horizon.clamp_min(0)) / log_scale)
        target_duration_feature = torch.tanh(torch.log1p(target_duration.clamp_min(0)) / log_scale)
        numeric = torch.stack((
            target_horizon_feature,
            target_duration_feature,
            target_duration / (target_duration + target_horizon + 1.0),
        ), dim=-1)
        query = (
            self.query_role.view(1, 1, -1)
            + self.numeric_proj(numeric.to(context_states.dtype))
            + self.descriptor_proj(descriptor.to(context_states.dtype))
            + self.resolution_embedding(target_resolution)
        )
        query = query * query_valid.unsqueeze(-1).to(query.dtype)

        context_positions = positions.repeat_interleave(sensors, dim=1)
        context_durations = durations.repeat_interleave(sensors, dim=1)
        context_resolutions = resolution_ids.repeat_interleave(sensors, dim=1).clamp(
            min=0, max=self.resolution_embedding.num_embeddings - 1,
        )
        context_sensor_index = torch.arange(sensors, device=context_states.device).repeat(patches)
        context_descriptor = sensor_descriptors.gather(
            1, context_sensor_index.view(1, -1, 1).expand(
                batch, -1, sensor_descriptors.shape[-1],
            ),
        )
        time_to_boundary = (context_end.reshape(batch, 1) - context_positions).clamp_min(0)
        context_numeric = torch.stack((
            torch.tanh(torch.log1p(time_to_boundary) / log_scale),
            torch.tanh(torch.log1p(context_durations.clamp_min(0)) / log_scale),
            context_durations / (context_durations + time_to_boundary + 1.0),
        ), dim=-1)
        memory = (
            self.context_proj(context_states.flatten(1, 2))
            + self.numeric_proj(context_numeric.to(context_states.dtype))
            + self.descriptor_proj(context_descriptor.to(context_states.dtype))
            + self.resolution_embedding(context_resolutions)
        )
        memory_valid = context_valid.flatten(1, 2)
        # Every eligible row has context. For ineligible rows, expose one zero memory element to
        # avoid an all-masked attention softmax; query_valid keeps it out of every objective.
        no_memory = ~memory_valid.any(dim=1)
        memory = memory.clone()
        memory_valid = memory_valid.clone()
        memory[:, 0] = torch.where(no_memory.unsqueeze(1), 0, memory[:, 0])
        memory_valid[:, 0] |= no_memory
        # TransformerDecoder rejects an all-padded target row. Expose one zero dummy query to the
        # decoder only; the original query_valid remains false so no objective consumes it.
        decoder_query_valid = query_valid.clone()
        decoder_query_valid[:, 0] |= ~query_valid.any(dim=1)
        predicted = self.decoder(
            query,
            memory,
            tgt_key_padding_mask=~decoder_query_valid,
            memory_key_padding_mask=~memory_valid,
        )
        predicted = self.out_proj(self.out_norm(predicted))
        return predicted, indices, query_valid


def gather_token_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``(patch,sensor)`` rows from a dense ``(B,P,S,D)`` tensor."""

    if values.ndim != 4 or indices.ndim != 3 or indices.shape[-1] != 2:
        raise ValueError("expected values (B,P,S,D) and indices (B,Q,2)")
    batch = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
    return values[batch, indices[..., 0], indices[..., 1]]


def past_context_references(
    context_targets: torch.Tensor,
    context_valid: torch.Tensor,
    positions: torch.Tensor,
    resolution_ids: torch.Tensor,
    num_resolutions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached mean and latest past targets for each ``(sensor, resolution)``.

    ``context_targets`` must come from an encoder invocation whose temporal keys are restricted to
    the observed prefix.  The final boolean records whether a same-resolution reference exists;
    callers may fall back to the all-resolution mean when a sparse grid lacks one.
    """

    if context_targets.ndim != 4 or context_valid.shape != context_targets.shape[:3]:
        raise ValueError("context targets and validity must have shapes (B,P,S,D) and (B,P,S)")
    batch, patches, sensors, width = context_targets.shape
    if positions.shape != (batch, patches) or resolution_ids.shape != (batch, patches):
        raise ValueError("context metadata must match the patch grid")
    if num_resolutions <= 0:
        raise ValueError("num_resolutions must be positive")
    groups = resolution_ids.clamp(0, num_resolutions - 1)
    group_mask = torch.nn.functional.one_hot(groups, num_classes=num_resolutions).bool()
    valid = context_valid.unsqueeze(-1) & group_mask.unsqueeze(2)
    count = valid.sum(dim=1).unsqueeze(-1)
    summed = torch.einsum("bpsd,bpsr->bsrd", context_targets.float(), valid.to(torch.float32))
    mean = summed / count.clamp_min(1).to(summed.dtype)

    # Choose the latest real context patch per sensor/resolution. ``argmax`` has a stable first
    # tie-break; overlapping resolutions may share a center without ambiguity for this control.
    time = positions.unsqueeze(2).unsqueeze(-1).expand(-1, -1, sensors, num_resolutions)
    latest_index = time.masked_fill(~valid, float("-inf")).argmax(dim=1)
    gather = latest_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, width)
    rows = context_targets.permute(0, 2, 1, 3).unsqueeze(2).expand(
        -1, -1, num_resolutions, -1, -1,
    )
    latest = rows.gather(3, gather).squeeze(3)
    return mean.detach(), latest.detach(), count.squeeze(-1).gt(0)


def balanced_future_latent_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    query_valid: torch.Tensor,
    query_resolution_ids: torch.Tensor,
    num_resolutions: int | None = None,
    query_weights: torch.Tensor | None = None,
    normalize_rows: bool = True,
) -> torch.Tensor:
    """Smooth-L1 normalized latent loss, averaged equally over active resolutions."""

    if prediction.shape != target.shape or query_valid.shape != prediction.shape[:2]:
        raise ValueError("future latent prediction, target, and validity shapes disagree")
    pred = F.layer_norm(prediction.float(), (prediction.shape[-1],)) if normalize_rows else prediction.float()
    tgt = F.layer_norm(target.detach().float(), (target.shape[-1],)) if normalize_rows else target.detach().float()
    per_query = F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=-1)
    live = query_valid & query_resolution_ids.ge(0)
    if query_weights is None:
        weights = torch.ones_like(per_query)
    elif query_weights.shape != per_query.shape:
        raise ValueError("query_weights must be finite and match (B,Q)")
    else:
        # Do not use a host-side ``Tensor.all()`` here: this loss runs every accelerator step.
        weights = torch.nan_to_num(
            query_weights.detach().to(per_query.dtype), nan=0.0, posinf=0.0, neginf=0.0,
        ).clamp_min(0)
    if num_resolutions is None:
        num_resolutions = max(int(query_resolution_ids.detach().max().item()) + 1, 1)
    groups = query_resolution_ids.clamp(0, num_resolutions - 1)
    sums = per_query.new_zeros(num_resolutions).scatter_add_(
        0, groups[live], per_query[live] * weights[live],
    )
    counts = per_query.new_zeros(num_resolutions).scatter_add_(
        0, groups[live], weights[live],
    )
    means = sums / counts.clamp_min(1)
    active = counts.gt(0).to(means.dtype)
    return (means * active).sum() / active.sum().clamp_min(1)


@dataclass(frozen=True)
class CollapseOutput:
    total: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    min_std: torch.Tensor


@dataclass
class FutureLossOutput:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


def combine_future_losses(
    future: torch.Tensor,
    physical: torch.Tensor,
    collapse: torch.Tensor,
    *,
    future_weight: float,
    physical_weight: float,
    collapse_weight: float,
) -> FutureLossOutput:
    """Apply fixed top-level coefficients without hiding any objective in another."""

    weights = {
        "future": float(future_weight),
        "physical": float(physical_weight),
        "collapse": float(collapse_weight),
    }
    if weights["future"] <= 0 or weights["collapse"] <= 0 or weights["physical"] < 0:
        raise ValueError("future/collapse weights must be positive and physical nonnegative")
    terms = {
        "future": weights["future"] * future,
        "collapse": weights["collapse"] * collapse,
    }
    if weights["physical"] > 0:
        terms["physical"] = weights["physical"] * physical
    return FutureLossOutput(total=sum(terms.values()), terms=terms)


def recommend_fixed_objective_weights(
    samples: list[dict[str, dict[str, float]]],
    current_weights: dict[str, float],
    target_shares: dict[str, float],
) -> dict[str, object]:
    """Choose one fixed multi-objective scalarization after warmup.

    Coefficients first match the requested median gradient-norm shares. A single common multiplier
    then preserves the pilot's median combined encoder-gradient norm, including measured pairwise
    dot products, so calibration does not silently alter the effective learning rate.
    """

    if not samples:
        raise ValueError("objective calibration needs at least one sample")
    names = tuple(current_weights)
    if set(names) != set(target_shares) or not names:
        raise ValueError("current weights and target shares must name the same objectives")
    if any(float(current_weights[name]) <= 0 for name in names):
        raise ValueError("every calibrated objective needs a positive current weight")
    share_total = sum(float(target_shares[name]) for name in names)
    if share_total <= 0 or any(float(target_shares[name]) <= 0 for name in names):
        raise ValueError("target objective shares must be positive")
    shares = {name: float(target_shares[name]) / share_total for name in names}
    for sample in samples:
        if set(sample.get("norms", {})) != set(names):
            raise ValueError("calibration sample objective names do not match requested weights")

    median_norm = {
        name: statistics.median(float(sample["norms"][name]) for sample in samples)
        for name in names
    }
    if any(value <= 0 or not math.isfinite(value) for value in median_norm.values()):
        raise ValueError("every objective must produce a finite non-zero calibration gradient")
    relative = {name: shares[name] / median_norm[name] for name in names}

    def _dot(sample, left, right):
        return float(sample["dots"].get(
            f"{left}|{right}", sample["dots"].get(f"{right}|{left}", 0.0),
        ))

    def _combined(sample, coefficients):
        squared = sum(
            coefficients[name] ** 2 * float(sample["norms"][name]) ** 2
            for name in names
        )
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                squared += 2 * coefficients[left] * coefficients[right] * _dot(
                    sample, left, right,
                )
        return math.sqrt(max(squared, 0.0))

    pilot_norm = statistics.median(_combined(sample, current_weights) for sample in samples)
    relative_norm = statistics.median(_combined(sample, relative) for sample in samples)
    if relative_norm <= 0:
        raise ValueError("calibrated objectives cancel to a zero combined gradient")
    common = pilot_norm / relative_norm
    recommended = {name: common * relative[name] for name in names}
    achieved = {
        name: recommended[name] * median_norm[name]
        / sum(recommended[item] * median_norm[item] for item in names)
        for name in names
    }
    return {
        "recommended": recommended,
        "target_gradient_shares": shares,
        "achieved_median_gradient_shares": achieved,
        "median_unit_gradient_norms": median_norm,
        "median_combined_encoder_grad_norm": {
            "pilot_weights": pilot_norm,
            "recommended_weights": statistics.median(
                _combined(sample, recommended) for sample in samples
            ),
        },
    }


def fixed_filterbank_physical_targets(
    analysis: torch.Tensor,
    *,
    n_bands: int,
    use_resolution_mask: bool,
    use_amplitude: bool,
    use_dc: bool,
    include_amplitude: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select stable motion values and return continuous observability weights.

    ``PhysicalFilterbankTokenizer.analyze`` concatenates values and acquisition metadata. The
    Nyquist/resolution values weight this target but are not themselves reconstructed.
    """

    if analysis.ndim != 4 or n_bands <= 0:
        raise ValueError("filterbank analysis must be (B,P,C,F) with positive n_bands")
    cursor = 0
    energy = analysis[..., cursor:cursor + n_bands]
    cursor += n_bands
    nyquist = analysis[..., cursor:cursor + n_bands].clamp(0, 1)
    cursor += n_bands
    resolved = torch.ones_like(nyquist)
    if use_resolution_mask:
        # Resolution is a confidence based on cycles present in the patch, not a categorical mask.
        # Retain it as a continuous loss weight instead of introducing an arbitrary threshold.
        resolved = analysis[..., cursor:cursor + n_bands].clamp(0, 1)
        cursor += n_bands
    values = [energy]
    validity = [nyquist * resolved]
    if use_amplitude:
        if include_amplitude:
            values.append(analysis[..., cursor:cursor + 1])
            validity.append(torch.ones_like(analysis[..., cursor:cursor + 1]))
        cursor += 1
    if use_dc:
        values.append(analysis[..., cursor:cursor + 1])
        validity.append(torch.ones_like(analysis[..., cursor:cursor + 1]))
        cursor += 1
    if cursor != analysis.shape[-1]:
        raise ValueError(
            f"unexpected filterbank analysis width {analysis.shape[-1]} (parsed {cursor})"
        )
    return torch.cat(values, dim=-1), torch.cat(validity, dim=-1)


def patch_variance_covariance(
    patch_states: torch.Tensor,
    valid: torch.Tensor,
    *,
    resolution_ids: torch.Tensor | None = None,
    num_resolutions: int | None = None,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    target_std: float = 1.0,
) -> CollapseOutput:
    """VICReg variance/covariance terms directly on downstream patch states."""

    if valid.shape != patch_states.shape[:-1]:
        raise ValueError("collapse validity must match patch-state leading dimensions")
    if resolution_ids is not None:
        if resolution_ids.shape != patch_states.shape[:2]:
            raise ValueError("collapse resolution ids must have shape (B,P)")
        if num_resolutions is None:
            num_resolutions = max(int(resolution_ids.detach().max().item()) + 1, 1)
        outputs = []
        active_groups = []
        for group in range(num_resolutions):
            group_valid = valid & resolution_ids.eq(group).unsqueeze(-1)
            active_groups.append(group_valid.sum().ge(2))
            outputs.append(patch_variance_covariance(
                patch_states, group_valid,
                variance_weight=variance_weight,
                covariance_weight=covariance_weight,
                target_std=target_std,
            ))
        active = torch.stack(active_groups)
        active_float = active.to(patch_states.dtype)
        denominator = active_float.sum().clamp_min(1)
        variance = (torch.stack([output.variance for output in outputs])
                    * active_float).sum() / denominator
        covariance = (torch.stack([output.covariance for output in outputs])
                      * active_float).sum() / denominator
        total = float(variance_weight) * variance + float(covariance_weight) * covariance
        std_values = torch.stack([output.min_std for output in outputs])
        min_std = torch.where(
            active, std_values, torch.full_like(std_values, float("inf")),
        ).min()
        min_std = torch.where(active.any(), min_std, min_std.new_zeros(()))
        return CollapseOutput(total, variance, covariance, min_std)

    rows = patch_states[valid].float()
    if rows.shape[0] < 2:
        zero = patch_states.sum() * 0.0
        return CollapseOutput(zero, zero, zero, zero.detach())
    centered = rows - rows.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=True) + 1e-4)
    variance = F.relu(float(target_std) - std).mean()
    cov = centered.T @ centered / max(rows.shape[0] - 1, 1)
    covariance = (cov.flatten()[:-1].view(cov.shape[0] - 1, cov.shape[0] + 1)[:, 1:]
                  .flatten().square().sum() / rows.shape[1])
    total = float(variance_weight) * variance + float(covariance_weight) * covariance
    return CollapseOutput(total, variance, covariance, std.min().detach())


def balanced_physical_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    query_valid: torch.Tensor,
    query_resolution_ids: torch.Tensor,
    feature_valid: torch.Tensor,
    num_resolutions: int | None = None,
) -> torch.Tensor:
    """Smooth-L1 physical loss balanced by resolution and excluding unavailable features."""

    if prediction.shape != target.shape or feature_valid.shape != target.shape:
        raise ValueError("physical prediction, target, and feature-valid shapes disagree")
    error = F.smooth_l1_loss(prediction.float(), target.detach().float(), reduction="none")
    feature_weight = feature_valid.to(error.dtype).clamp(0, 1) \
        * query_valid.unsqueeze(-1).to(error.dtype)
    per_query_sum = (error * feature_weight).sum(dim=-1)
    per_query_count = feature_weight.sum(dim=-1)
    live = query_valid & query_resolution_ids.ge(0)
    if num_resolutions is None:
        num_resolutions = max(int(query_resolution_ids.detach().max().item()) + 1, 1)
    groups = query_resolution_ids.clamp(0, num_resolutions - 1)
    sums = per_query_sum.new_zeros(num_resolutions).scatter_add_(
        0, groups[live], per_query_sum[live],
    )
    counts = per_query_count.new_zeros(num_resolutions).scatter_add_(
        0, groups[live], per_query_count[live],
    )
    means = sums / counts.clamp_min(1)
    active = counts.gt(0).to(means.dtype)
    return (means * active).sum() / active.sum().clamp_min(1)
