"""Parameter-free differentiable-neighbour readout shared by training and evaluation.

This is an *experiment control*, not a separate HALO model.  It maps one query recording vector
and labelled support recording vectors to candidate logits with temperature-scaled cosine weights.
Keeping it here prevents the training and sealed-evaluation implementations from diverging.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_TEMPERATURE = 0.07


def differentiable_neighbor_logits(
    query: torch.Tensor,
    support: torch.Tensor,
    support_candidate: torch.Tensor,
    support_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return candidate log-probabilities and per-support soft weights.

    Shapes are ``query=(B,D)``, ``support=(B,K,D)``, and candidate-index/mask tensors
    ``(B,K)/(B,C)``.  Every valid support participates, which keeps the path differentiable for
    encoder tuning.  The function deliberately rejects zero-support episodes; they are a
    different information condition and do not have a neighbour interpretation.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if query.ndim != 2 or support.ndim != 3 or support.shape[:1] != query.shape[:1] \
            or support.shape[-1] != query.shape[-1]:
        raise ValueError("query must be (B,D) and support must be matching (B,K,D)")
    if support_candidate.shape != support_mask.shape or support_mask.shape != support.shape[:2]:
        raise ValueError("support_candidate and support_mask must be (B,K)")
    if candidate_mask.ndim != 2 or candidate_mask.shape[0] != query.shape[0]:
        raise ValueError("candidate_mask must be (B,C)")
    if not bool(support_mask.any(dim=1).all()):
        raise ValueError("differentiable neighbours require at least one valid support per query")

    similarity = torch.einsum(
        "bd,bkd->bk", F.normalize(query.float(), dim=-1), F.normalize(support.float(), dim=-1),
    ) / temperature
    weight = torch.softmax(similarity.masked_fill(~support_mask, float("-inf")), dim=1)
    weight = torch.where(support_mask, weight, torch.zeros_like(weight))
    vote = torch.zeros_like(candidate_mask, dtype=weight.dtype)
    invalid_binding = support_mask & ((support_candidate < 0) | (support_candidate >= vote.shape[1]))
    if bool(invalid_binding.any()):
        raise ValueError("valid support has an invalid candidate binding")
    vote.scatter_add_(1, support_candidate.clamp_min(0), weight)
    logits = vote.clamp_min(1e-12).log().masked_fill(~candidate_mask, -1e30)
    return logits, weight
