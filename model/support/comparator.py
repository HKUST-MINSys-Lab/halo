"""Current support-conditioned classification control.

The control starts as a closed-form soft vote over all supplied support recordings. Its sole
learned operation is a sensor-vector reweighting of support rows. Candidate and support-label text
remain in the fixed voting rule; the learned path cannot form a text-to-motion shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import AttentionSpec, ScaledSum, SetAttentionStack

ROLE_QUERY, ROLE_SUPPORT = range(2)
N_ROLES = 2
UNBOUND_SLOT = 0
READOUTS = ("sensor_only",)


@dataclass(frozen=True)
class ComparatorConfig:
    """Configuration for the retained sensor-only support reweighter."""

    text_dim: int = 384
    n_layers: int = 2
    n_slots: int = 64
    identity_gain_init: float = 0.25
    readout: str = "sensor_only"
    use_descriptor: bool = False

    def __post_init__(self) -> None:
        if self.readout != "sensor_only":
            raise ValueError("only the current sensor_only support classifier is retained")
        if self.n_layers < 0 or self.n_slots < 2:
            raise ValueError("attention depth must be nonnegative and slot capacity at least two")


def comparator_config_from_checkpoint(saved: dict) -> ComparatorConfig:
    """Restore only checkpoints made by the retained architecture."""
    config = dict(saved)
    if config.get("readout", "sensor_only") != "sensor_only":
        raise ValueError("checkpoint uses a retired support-classifier readout")
    return ComparatorConfig(**config)


class SupportComparator(nn.Module):
    """Set-attend over the query and all support sensor vectors to shift row logits."""

    def __init__(self, spec: AttentionSpec, cfg: ComparatorConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or ComparatorConfig()
        d = spec.d_model
        self.proj_signal = nn.Linear(d, d)
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.slot_emb = nn.Embedding(self.cfg.n_slots, d)
        self.compose = ScaledSum(3, init=[1.0, self.cfg.identity_gain_init,
                                          self.cfg.identity_gain_init])
        self.stack = SetAttentionStack(spec, self.cfg.n_layers)
        self.shift_head = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.shift_head.weight)

    @property
    def head(self) -> nn.Linear:
        return self.shift_head

    def _token(self, content: torch.Tensor, role: int, slot: torch.Tensor) -> torch.Tensor:
        roles = self.role_emb(torch.full(content.shape[:-1], role, dtype=torch.long,
                                         device=content.device))
        return self.compose(content, roles, self.slot_emb(slot))

    def forward(self, *, candidate_text: torch.Tensor, query_feature: torch.Tensor,
                query_descriptor: torch.Tensor, query_mask: torch.Tensor,
                support_feature: torch.Tensor, support_descriptor: torch.Tensor,
                support_label_text: torch.Tensor, support_mask: torch.Tensor,
                candidate_slot: torch.Tensor, support_slot: torch.Tensor,
                candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return one additive temperature-scale shift per valid support recording."""
        del candidate_text, query_descriptor, support_descriptor, support_label_text, candidate_slot
        if query_feature.shape[0] != support_feature.shape[0]:
            raise ValueError("query and support batches must match")
        if support_mask.shape != support_feature.shape[:2] or support_slot.shape != support_mask.shape:
            raise ValueError("support metadata must match support rows")
        if candidate_mask is not None and not bool(candidate_mask.any(dim=1).all()):
            raise ValueError("each episode needs at least one candidate")
        valid_slots = support_slot[support_mask]
        if bool((valid_slots < UNBOUND_SLOT).any()) or bool((valid_slots >= self.cfg.n_slots).any()):
            raise ValueError("support slot exceeds ComparatorConfig.n_slots")
        B, Q, _ = query_feature.shape
        query_slot = torch.full((B, Q), UNBOUND_SLOT, dtype=torch.long, device=query_feature.device)
        query = self._token(self.proj_signal(query_feature), ROLE_QUERY, query_slot)
        support = self._token(self.proj_signal(support_feature), ROLE_SUPPORT, support_slot)
        hidden = self.stack(torch.cat([query, support], dim=1),
                            key_padding_mask=torch.cat([query_mask, support_mask], dim=1))
        with torch.autocast(device_type=query_feature.device.type, enabled=False):
            shift = self.shift_head(hidden[:, Q:].float()).squeeze(-1)
        return shift.masked_fill(~support_mask, 0.0)

    def telemetry(self) -> dict[str, float]:
        gains = self.compose.log_gain.detach().exp()
        return {
            "comparator/residual_head_norm": float(self.shift_head.weight.detach().norm()),
            "comparator/content_gain": float(gains[0]),
            "comparator/identity_gain_mean": float(gains[1:].mean()),
        }


def support_vote(*, candidate_text: torch.Tensor, support_label_text: torch.Tensor,
                 support_bound: torch.Tensor, support_mask: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    """Map support-row weights to candidate logits with fixed label semantics."""
    B, C, _ = candidate_text.shape
    K = support_label_text.shape[1]
    if support_bound.shape != (B, K):
        raise ValueError("support binding must match support rows")
    index = torch.arange(C, device=candidate_text.device).view(1, 1, C)
    bound = support_bound.view(B, K, 1)
    enrolled = (bound.ge(0) & bound.eq(index)).to(weights.dtype)
    semantic = F.relu(torch.bmm(F.normalize(support_label_text.float(), dim=-1),
                                 F.normalize(candidate_text.float(), dim=-1).transpose(1, 2)))
    vote = enrolled + semantic.to(weights.dtype) * bound.lt(0).to(weights.dtype)
    return (weights * support_mask.unsqueeze(-1).to(weights.dtype) * vote).sum(dim=1)


def center_episode(query_feature: torch.Tensor, support_feature: torch.Tensor,
                   query_mask: torch.Tensor, support_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the episode-common feature component from query and support rows together."""
    rows = torch.cat([query_feature, support_feature], dim=1)
    mask = torch.cat([query_mask, support_mask], dim=1).unsqueeze(-1).to(rows.dtype)
    mean = (rows * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1e-6)
    return query_feature - mean, support_feature - mean


def comparator_logits(
    comparator: SupportComparator | None, *, candidate_text: torch.Tensor,
    query_feature: torch.Tensor, query_descriptor: torch.Tensor, query_mask: torch.Tensor,
    support_feature: torch.Tensor, support_descriptor: torch.Tensor,
    support_label_text: torch.Tensor, support_bound: torch.Tensor, support_mask: torch.Tensor,
    candidate_slot: torch.Tensor | None = None, candidate_mask: torch.Tensor | None = None,
    temperature: float = 0.07, vote_scale: float = 10.0, center: bool = True,
) -> dict[str, torch.Tensor]:
    """Score candidates by the fixed vote plus an optional learned support-row shift."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    B, C, _ = candidate_text.shape
    K = support_feature.shape[1]
    if candidate_mask is None:
        candidate_mask = torch.ones((B, C), dtype=torch.bool, device=candidate_text.device)
    if center:
        query_feature, support_feature = center_episode(query_feature, support_feature,
                                                        query_mask, support_mask)
    valid_query = query_mask.unsqueeze(-1).to(query_feature.dtype)
    pooled = (query_feature * valid_query).sum(1) / valid_query.sum(1).clamp_min(1e-6)
    similarity = torch.bmm(F.normalize(pooled.float(), dim=-1).unsqueeze(1),
                           F.normalize(support_feature.float(), dim=-1).transpose(1, 2)).squeeze(1)
    similarity = similarity.masked_fill(~support_mask, float("-inf"))
    empty = ~support_mask.any(dim=1, keepdim=True)

    def row_weights(logits: torch.Tensor) -> torch.Tensor:
        value = torch.softmax(torch.where(empty, torch.zeros_like(logits), logits), dim=1)
        value = torch.where(support_mask, value, torch.zeros_like(value))
        value = torch.where(empty, torch.zeros_like(value), value)
        return value.unsqueeze(-1).expand(B, K, C)

    base_weights = row_weights(similarity / temperature)
    base = vote_scale * support_vote(candidate_text=candidate_text,
                                     support_label_text=support_label_text,
                                     support_bound=support_bound, support_mask=support_mask,
                                     weights=base_weights)
    used_weights, logits = base_weights, base
    if comparator is not None:
        if candidate_slot is None:
            raise ValueError("a comparator needs episode-local candidate slots")
        support_slot = torch.where(support_bound.ge(0),
                                   torch.gather(candidate_slot, 1, support_bound.clamp_min(0)),
                                   torch.full_like(support_bound, UNBOUND_SLOT))
        shift = comparator(candidate_text=candidate_text, query_feature=query_feature,
                           query_descriptor=query_descriptor, query_mask=query_mask,
                           support_feature=support_feature, support_descriptor=support_descriptor,
                           support_label_text=support_label_text, support_mask=support_mask,
                           candidate_slot=candidate_slot, support_slot=support_slot,
                           candidate_mask=candidate_mask)
        used_weights = row_weights(similarity / temperature + shift.to(similarity.dtype))
        logits = vote_scale * support_vote(candidate_text=candidate_text,
                                           support_label_text=support_label_text,
                                           support_bound=support_bound, support_mask=support_mask,
                                           weights=used_weights)
    logits = logits.masked_fill(~candidate_mask, -1e30)
    return {"logits": logits, "base_logits": base.masked_fill(~candidate_mask, -1e30),
            "residual": logits - base, "support_weight": used_weights[..., 0]}
