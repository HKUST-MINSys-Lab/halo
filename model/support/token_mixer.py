"""Full support-conditioned token mixer.

The classifier is deliberately a small, explicit layer above the shared motion encoder.  It sees
query motion, enrolled support motion, the labels attached to those supports, and the declared
candidate labels as distinct tokens.  There is no top-k retrieval: every supplied support row is
present in the set attention calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import AttentionSpec, SetAttentionStack
from model.support.roles import (
    N_ROLES, ROLE_CANDIDATE, ROLE_QUERY, ROLE_SUPPORT, ROLE_SUPPORT_LABEL,
)


@dataclass(frozen=True)
class TokenMixerConfig:
    text_dim: int = 384
    n_layers: int = 2
    # Sealed datasets may expose their complete label roster.  These limits are contracts, not
    # training-tuned vocabulary sizes: reject an infeasible quadratic attention problem loudly
    # instead of silently dropping candidate labels or enrolled examples.
    max_candidates: int = 256
    max_supports: int = 8192
    temperature: float = 0.07

    def __post_init__(self) -> None:
        if self.n_layers < 0 or self.max_candidates < 2 or self.max_supports < 1:
            raise ValueError("invalid token-mixer capacity")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


class SemanticEpisodeHead(nn.Module):
    """One parameter set for either zero-shot or enrolled-support episodes."""

    def __init__(self, spec: AttentionSpec, cfg: TokenMixerConfig | None = None, *, mode: str):
        super().__init__()
        if mode not in {"zero", "few"}:
            raise ValueError("mode must be 'zero' or 'few'")
        self.spec = spec
        self.cfg = cfg or TokenMixerConfig()
        self.mode = mode
        d = spec.d_model
        self.signal_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.label_proj = nn.Sequential(nn.LayerNorm(self.cfg.text_dim), nn.Linear(self.cfg.text_dim, d))
        self.role_emb = nn.Embedding(N_ROLES, d)
        # Candidate tags bind candidate labels to labels attached to enrolled support.  Support-pair
        # tags bind one recording token to its own label token.  Their slot order is randomized by
        # episode construction, so neither table can become a label lookup.
        self.candidate_tag = nn.Embedding(self.cfg.max_candidates + 1, d)
        self.support_pair_tag = nn.Embedding(self.cfg.max_supports + 1, d)
        self.stack = SetAttentionStack(spec, self.cfg.n_layers)

    def _compose(
        self,
        content: torch.Tensor,
        role: int,
        *,
        candidate_tag: torch.Tensor | None = None,
        pair_tag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = content.shape[:-1]
        roles = self.role_emb(torch.full(shape, role, device=content.device, dtype=torch.long))
        value = content + roles
        if candidate_tag is not None:
            value = value + self.candidate_tag(candidate_tag)
        if pair_tag is not None:
            value = value + self.support_pair_tag(pair_tag)
        return value

    def _check(self, candidate_mask: torch.Tensor, support_mask: torch.Tensor | None = None) -> None:
        if candidate_mask.ndim != 2 or bool((~candidate_mask.any(dim=1)).any()):
            raise ValueError("every episode needs at least one candidate")
        if candidate_mask.shape[1] > self.cfg.max_candidates:
            raise ValueError("candidate count exceeds TokenMixerConfig.max_candidates")
        if support_mask is not None and support_mask.shape[1] > self.cfg.max_supports:
            raise ValueError("support count exceeds TokenMixerConfig.max_supports")

    def zero_shot(
        self, *, query_feature: torch.Tensor, candidate_text: torch.Tensor,
        candidate_mask: torch.Tensor, candidate_slot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.mode != "zero":
            raise RuntimeError("few-shot head cannot score zero-shot episodes")
        self._check(candidate_mask)
        if query_feature.ndim != 2 or candidate_text.shape[:2] != candidate_mask.shape:
            raise ValueError("zero-shot tensor shapes do not agree")
        b, c = candidate_mask.shape
        if candidate_slot.shape != (b, c):
            raise ValueError("candidate slots must have one entry per candidate")
        query = self._compose(self.signal_proj(query_feature).unsqueeze(1), ROLE_QUERY)
        candidates = self._compose(
            self.label_proj(candidate_text), ROLE_CANDIDATE,
            candidate_tag=candidate_slot,
        )
        values = torch.cat((query, candidates), dim=1)
        valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=values.device), candidate_mask), dim=1)
        hidden = self.stack(values, key_padding_mask=valid)
        refined_query, refined_candidates = hidden[:, 0], hidden[:, 1:]
        logits = torch.einsum(
            "bd,bcd->bc", F.normalize(refined_query.float(), dim=-1),
            F.normalize(refined_candidates.float(), dim=-1),
        ) / self.cfg.temperature
        return {"logits": logits.masked_fill(~candidate_mask, -1e30),
                "query": refined_query, "candidate": refined_candidates}

    def few_shot(
        self, *, query_feature: torch.Tensor, support_feature: torch.Tensor,
        support_label_text: torch.Tensor, candidate_text: torch.Tensor,
        support_bound: torch.Tensor, support_mask: torch.Tensor,
        candidate_slot: torch.Tensor, support_pair_slot: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.mode != "few":
            raise RuntimeError("zero-shot head cannot score enrolled episodes")
        self._check(candidate_mask, support_mask)
        b, k, _ = support_feature.shape
        c = candidate_mask.shape[1]
        if query_feature.shape != (b, support_feature.shape[-1]):
            raise ValueError("query feature must have one row per episode")
        if support_label_text.shape[:2] != (b, k) or support_bound.shape != (b, k):
            raise ValueError("support label/binding shapes do not match support features")
        if candidate_slot.shape != (b, c) or support_pair_slot.shape != (b, k):
            raise ValueError("candidate and support-pair slots must follow their token rows")
        if bool((support_bound[support_mask] < 0).any()) or bool((support_bound[support_mask] >= c).any()):
            raise ValueError("every few-shot support must be bound to a valid candidate")
        query = self._compose(self.signal_proj(query_feature).unsqueeze(1), ROLE_QUERY)
        pair = support_pair_slot * support_mask
        support = self._compose(self.signal_proj(support_feature), ROLE_SUPPORT, pair_tag=pair)
        # Carry the candidate's identity tag to its paired support-label token. ``support_bound``
        # is an index for the final vote; it is deliberately not itself an embedding id.
        bound_tag = candidate_slot.gather(1, support_bound.clamp_min(0)) * support_mask
        support_label = self._compose(
            self.label_proj(support_label_text), ROLE_SUPPORT_LABEL,
            candidate_tag=bound_tag, pair_tag=pair,
        )
        candidate = self._compose(
            self.label_proj(candidate_text), ROLE_CANDIDATE, candidate_tag=candidate_slot,
        )
        values = torch.cat((query, support, support_label, candidate), dim=1)
        valid = torch.cat((
            torch.ones((b, 1), dtype=torch.bool, device=values.device),
            support_mask, support_mask, candidate_mask,
        ), dim=1)
        hidden = self.stack(values, key_padding_mask=valid)
        refined_query = hidden[:, 0]
        refined_support = hidden[:, 1:1 + k]
        similarity = torch.einsum(
            "bd,bkd->bk", F.normalize(refined_query.float(), dim=-1),
            F.normalize(refined_support.float(), dim=-1),
        ) / self.cfg.temperature
        weights = torch.softmax(similarity.masked_fill(~support_mask, float("-inf")), dim=1)
        weights = torch.where(support_mask, weights, torch.zeros_like(weights))
        vote = torch.zeros((b, c), device=weights.device, dtype=weights.dtype)
        vote.scatter_add_(1, support_bound.clamp_min(0), weights * support_mask.to(weights.dtype))
        logits = vote.clamp_min(1e-12).log().masked_fill(~candidate_mask, -1e30)
        return {"logits": logits, "support_weight": weights, "query": refined_query,
                "support": refined_support, "candidate": hidden[:, 1 + 2 * k:]}

    def telemetry(self) -> dict[str, float]:
        return {
            f"head/{self.mode}/signal_projection_norm": float(self.signal_proj[-1].weight.detach().norm()),
            f"head/{self.mode}/label_projection_norm": float(self.label_proj[-1].weight.detach().norm()),
        }


class SupportTokenMixer(nn.Module):
    """Two independently parameterised semantic heads sharing an external motion encoder."""

    def __init__(self, spec: AttentionSpec, cfg: TokenMixerConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or TokenMixerConfig()
        self.zero_shot_head = SemanticEpisodeHead(spec, self.cfg, mode="zero")
        self.few_shot_head = SemanticEpisodeHead(spec, self.cfg, mode="few")

    def forward(self, *, is_zero_shot: torch.Tensor, query_feature: torch.Tensor,
                candidate_text: torch.Tensor, candidate_mask: torch.Tensor,
                candidate_slot: torch.Tensor,
                support_feature: torch.Tensor | None = None,
                support_label_text: torch.Tensor | None = None,
                support_bound: torch.Tensor | None = None,
                support_mask: torch.Tensor | None = None,
                support_pair_slot: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Route each episode to its own head while preserving original row order."""
        if is_zero_shot.dtype != torch.bool or is_zero_shot.shape != candidate_mask.shape[:1]:
            raise ValueError("is_zero_shot must be one boolean per episode")
        b, c = candidate_mask.shape
        logits = query_feature.new_full((b, c), -1e30, dtype=torch.float32)
        support_weight = None
        zero_rows = torch.nonzero(is_zero_shot, as_tuple=False).flatten()
        few_rows = torch.nonzero(~is_zero_shot, as_tuple=False).flatten()
        if len(zero_rows):
            result = self.zero_shot_head.zero_shot(
                query_feature=query_feature.index_select(0, zero_rows),
                candidate_text=candidate_text.index_select(0, zero_rows),
                candidate_mask=candidate_mask.index_select(0, zero_rows),
                candidate_slot=candidate_slot.index_select(0, zero_rows),
            )
            logits.index_copy_(0, zero_rows, result["logits"].to(logits.dtype))
        if len(few_rows):
            if any(value is None for value in (support_feature, support_label_text, support_bound,
                                               support_mask, support_pair_slot)):
                raise ValueError("few-shot rows require support tensors")
            result = self.few_shot_head.few_shot(
                query_feature=query_feature.index_select(0, few_rows),
                support_feature=support_feature.index_select(0, few_rows),
                support_label_text=support_label_text.index_select(0, few_rows),
                candidate_text=candidate_text.index_select(0, few_rows),
                support_bound=support_bound.index_select(0, few_rows),
                support_mask=support_mask.index_select(0, few_rows),
                candidate_slot=candidate_slot.index_select(0, few_rows),
                support_pair_slot=support_pair_slot.index_select(0, few_rows),
                candidate_mask=candidate_mask.index_select(0, few_rows),
            )
            logits.index_copy_(0, few_rows, result["logits"].to(logits.dtype))
            support_weight = query_feature.new_zeros(support_mask.shape, dtype=torch.float32)
            support_weight.index_copy_(0, few_rows, result["support_weight"].to(support_weight.dtype))
        return {"logits": logits, "support_weight": support_weight}

    def telemetry(self) -> dict[str, float]:
        return {**self.zero_shot_head.telemetry(), **self.few_shot_head.telemetry()}
