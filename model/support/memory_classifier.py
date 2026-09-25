"""Causal, support-and-unlabelled memory reader (experimental v5).

The reader may reweight evidence but cannot create an independent candidate-logit path.
It is deliberately separate from the registered transductive rung-1 readout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model.blocks import AttentionSpec


ARCHITECTURE_VERSION = "support_memory_reader_v5"


@dataclass(frozen=True)
class MemoryReaderConfig:
    text_dim: int = 384
    hidden_dim: int = 128
    num_heads: int = 4
    semantic_temperature: float = 0.07
    neighbor_temperature: float = 0.07
    max_entries: int = 64

    def __post_init__(self) -> None:
        if self.text_dim < 1 or self.hidden_dim < 1 or self.max_entries < 1:
            raise ValueError("memory reader dimensions and capacity must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.semantic_temperature <= 0 or self.neighbor_temperature <= 0:
            raise ValueError("temperatures must be positive")


@dataclass
class MemoryReadout:
    probabilities: torch.Tensor
    semantic: torch.Tensor
    fixed: torch.Tensor
    vote: torch.Tensor
    reliability: torch.Tensor
    semantic_weight: torch.Tensor


class MemoryReaderClassifier(nn.Module):
    """Three grouped tokens per entry; permutation-invariant attention over entries.

    All tensors are for a single deployment episode. ``verified`` is -1 for an unlabelled
    entry and a candidate index otherwise. ``evidence`` is the detached *original* semantic
    distribution, recomputed against the active roster when that roster changes.
    """

    def __init__(self, spec: AttentionSpec, cfg: MemoryReaderConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or MemoryReaderConfig(hidden_dim=spec.d_model)
        d, h = spec.d_model, self.cfg.hidden_dim
        self.p_text = nn.Linear(d, self.cfg.text_dim)
        self.motion_proj = nn.Linear(d, h)
        self.acquisition_proj = nn.Linear(d, h)
        self.evidence_proj = nn.Linear(self.cfg.text_dim, h)
        self.query_proj = nn.Linear(d, h)
        self.role = nn.Embedding(4, h)  # query, motion, acquisition, label evidence
        self.entry_attention = nn.MultiheadAttention(h, self.cfg.num_heads, batch_first=True)
        self.bank_attention = nn.MultiheadAttention(h, self.cfg.num_heads, batch_first=True)
        self.entry_norm = nn.LayerNorm(h)
        self.bank_norm = nn.LayerNorm(h)
        self.trust = nn.Sequential(nn.Linear(2 * h + 6, h), nn.GELU(), nn.Linear(h, 1))
        self.gate = nn.Sequential(nn.Linear(2 * h + 4, h), nn.GELU(), nn.Linear(h, 1))
        # Initial state is close to the fixed equal blend, without freezing inner gradients.
        nn.init.normal_(self.trust[-1].weight, std=1e-3)
        nn.init.zeros_(self.trust[-1].bias)
        nn.init.normal_(self.gate[-1].weight, std=1e-3)
        nn.init.zeros_(self.gate[-1].bias)

    def semantic(self, motion: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if motion.shape[-1] != self.spec.d_model or candidates.shape[-1] != self.cfg.text_dim:
            raise ValueError("motion or candidate text width does not match the checkpoint")
        scores = F.normalize(self.p_text(motion).float(), dim=-1) @ F.normalize(
            candidates.float(), dim=-1,
        ).T
        return F.softmax(scores / self.cfg.semantic_temperature, dim=-1)

    def forward(
        self, query: torch.Tensor, query_acquisition: torch.Tensor,
        candidates: torch.Tensor, memory_motion: torch.Tensor,
        memory_acquisition: torch.Tensor, evidence: torch.Tensor,
        verified: torch.Tensor,
    ) -> MemoryReadout:
        if query.ndim != 1 or query_acquisition.shape != query.shape:
            raise ValueError("one query motion/acquisition vector is required")
        if candidates.ndim != 2 or candidates.shape[0] < 2:
            raise ValueError("at least two candidate labels are required")
        n, c = evidence.shape
        if n > self.cfg.max_entries or c != candidates.shape[0]:
            raise ValueError("memory capacity or evidence roster mismatch")
        if (memory_motion.shape != (n, self.spec.d_model)
                or memory_acquisition.shape != memory_motion.shape
                or verified.shape != (n,)):
            raise ValueError("memory tensor shapes do not agree")
        if n and bool(((verified < -1) | (verified >= c)).any()):
            raise ValueError("verified label index is outside the roster")
        semantic = self.semantic(query, candidates)
        if not n:
            empty = query.new_empty(0)
            return MemoryReadout(semantic, semantic, semantic, semantic, empty,
                                 semantic.new_ones(c))
        if not bool(torch.isfinite(evidence).all()):
            raise ValueError("memory evidence must be finite")
        # Pseudo-label evidence is a snapshot, not a backprop path through an earlier decision.
        pseudo = evidence.detach().float()
        label_probs = torch.where(
            (verified >= 0)[:, None],
            F.one_hot(verified.clamp_min(0).long(), c).float(),
            pseudo,
        )
        evidence_mass = label_probs.sum(-1)
        if bool(((label_probs < 0).any(dim=-1) | (evidence_mass < 0) | ((evidence_mass > 1e-8)
                 & ((evidence_mass - 1).abs() > 1e-3))).any()):
            raise ValueError("memory evidence must be normalized or explicitly absent")
        has_evidence = evidence_mass > 1e-8
        label_probs = label_probs / evidence_mass[:, None].clamp_min(1e-8)
        if not bool(has_evidence.any()):
            empty = query.new_empty(0)
            return MemoryReadout(semantic, semantic, semantic, semantic, empty,
                                 semantic.new_ones(c))
        label_vector = label_probs @ F.normalize(candidates.float(), dim=-1)
        tokens = torch.stack((
            self.motion_proj(memory_motion.float()) + self.role.weight[1],
            self.acquisition_proj(memory_acquisition.float()) + self.role.weight[2],
            self.evidence_proj(label_vector) + self.role.weight[3],
        ), dim=1)
        # Attention within each entry supplies a shared group identity without a positional code.
        local, _ = self.entry_attention(tokens, tokens, tokens, need_weights=False)
        entry = self.entry_norm(tokens + local).mean(dim=1)
        q = self.query_proj(query.float()) + self.role.weight[0]
        global_context, _ = self.bank_attention(
            q[None, None], entry[None], entry[None], need_weights=False,
        )
        q = self.bank_norm(q + global_context[0, 0])
        similarity = F.normalize(query.float(), dim=-1) @ F.normalize(
            memory_motion.float(), dim=-1,
        ).T
        acquisition_similarity = F.normalize(query_acquisition.float(), dim=-1) @ F.normalize(
            memory_acquisition.float(), dim=-1,
        ).T
        base_weights = F.softmax(
            (similarity / self.cfg.neighbor_temperature).masked_fill(~has_evidence, float("-inf")),
            dim=0,
        )
        fixed_vote = base_weights @ label_probs
        fixed = 0.5 * (semantic + fixed_vote)
        entropy = -(label_probs * label_probs.clamp_min(1e-8).log()).sum(-1) / torch.log(
            similarity.new_tensor(float(c)),
        )
        if n > 1:
            pairwise = F.normalize(memory_motion.float(), dim=-1) @ F.normalize(
                memory_motion.float(), dim=-1,
            ).T
            pairwise = pairwise.masked_fill(
                torch.eye(n, dtype=torch.bool, device=pairwise.device), float("-inf"),
            )
            neighbor_density = pairwise.topk(min(3, n - 1), dim=-1).values.mean(-1)
            neighbor_weight = F.softmax(pairwise / self.cfg.neighbor_temperature, dim=-1)
            neighbor_usable = neighbor_weight * has_evidence[None].float()
            neighbor_label = (neighbor_usable @ label_probs) / neighbor_usable.sum(
                -1, keepdim=True,
            ).clamp_min(1e-8)
            neighbor_consensus = (label_probs * neighbor_label).sum(-1)
        else:
            neighbor_density = similarity.new_zeros(n)
            neighbor_consensus = similarity.new_zeros(n)
        trust_features = torch.cat((
            entry, q.expand(n, -1), similarity[:, None], acquisition_similarity[:, None],
            entropy[:, None], (verified >= 0).float()[:, None],
            neighbor_density[:, None], neighbor_consensus[:, None],
        ), dim=-1)
        adjustment = self.trust(trust_features).squeeze(-1)
        weights = F.softmax(
            (similarity / self.cfg.neighbor_temperature + adjustment).masked_fill(
                ~has_evidence, float("-inf"),
            ), dim=0,
        )
        vote = weights @ label_probs
        semantic_candidate = F.normalize(candidates.float(), dim=-1)
        gate_features = torch.cat((
            q.expand(c, -1), self.evidence_proj(semantic_candidate),
            semantic[:, None], vote[:, None], fixed_vote[:, None],
            torch.full((c, 1), n / self.cfg.max_entries, device=q.device),
        ), dim=-1)
        semantic_weight = torch.sigmoid(self.gate(gate_features).squeeze(-1))
        combined = semantic_weight * semantic + (1 - semantic_weight) * vote
        combined = combined / combined.sum().clamp_min(1e-8)
        return MemoryReadout(combined, semantic, fixed, vote, weights, semantic_weight)
