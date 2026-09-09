"""Separate semantic/enrollment heads and the parameter-free encoder-training control."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from model.blocks import AttentionSpec, ScaledSum, SetAttentionStack

QUERY, SUPPORT, SUPPORT_LABEL, CANDIDATE = range(4)


def cosine_execution_pool(
    window_features: torch.Tensor,
    window_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool windows into one execution under the cosine-retrieval geometry.

    Each window contributes direction rather than unconstrained encoder norm.  The resulting
    execution is normalised again so this operation is identical whether it is consumed by a dot
    product or by :func:`neighbor_logits`.  ``window_mask`` is useful for padded execution groups.
    """
    if window_features.ndim < 2:
        raise ValueError("execution pooling expects (..., windows, features)")
    if window_features.shape[-2] == 0:
        raise ValueError("an execution must contain at least one window")
    values = F.normalize(window_features.float(), dim=-1)
    if window_mask is None:
        pooled = values.mean(dim=-2)
    else:
        if window_mask.shape != window_features.shape[:-1]:
            raise ValueError("window mask must match every execution window")
        weights = window_mask.unsqueeze(-1).to(values.dtype)
        count = weights.sum(dim=-2)
        if bool((count == 0).any()):
            raise ValueError("each execution needs at least one valid window")
        pooled = (values * weights).sum(dim=-2) / count
    return F.normalize(pooled, dim=-1)


class RecordingAttentionHead(nn.Module):
    """One shared architecture; query/candidate cosine is the only classification readout.

    Instance IDs identify recordings, NOT classes. Carry IDs along when permuting support pairs.
    Training randomizes the ID vocabulary to discourage learning from an arbitrary pair number.
    """

    def __init__(self, spec: AttentionSpec, *, text_dim: int, n_layers: int,
                 max_instances: int = 512, identity_gain: float = 0.25):
        super().__init__()
        d = spec.d_model
        self.signal = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.text = nn.Sequential(nn.Linear(text_dim, d), nn.LayerNorm(d))
        self.roles = nn.Embedding(4, d)
        # Zero denotes no support instance (query/candidate); it is not a learned pair tag.
        self.instances = nn.Embedding(max_instances + 1, d, padding_idx=0)
        self.compose = ScaledSum(3, init=[1.0, identity_gain, identity_gain])
        self.stack = SetAttentionStack(spec, n_layers)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def _token(self, content, role, ids):
        roles = self.roles(torch.full_like(ids, role))
        return self.compose(content, roles, self.instances(ids))

    def forward(self, *, query, candidates, candidate_mask, support=None,
                support_labels=None, support_mask=None, instance_ids=None):
        b, c, _ = candidates.shape
        q = self._token(self.signal(query).unsqueeze(1), QUERY,
                        torch.zeros((b, 1), dtype=torch.long, device=query.device))
        labels = self._token(self.text(candidates), CANDIDATE,
                             torch.zeros((b, c), dtype=torch.long, device=query.device))
        tokens = [q, labels]
        masks = [torch.ones((b, 1), dtype=torch.bool, device=query.device), candidate_mask]
        if support is not None:
            k = support.shape[1]
            if k > self.instances.num_embeddings - 1:
                raise ValueError("support count exceeds the configured instance-tag capacity")
            if support_labels is None or support_mask is None:
                raise ValueError("support vectors require paired labels and validity masks")
            if instance_ids is None:
                ids = torch.arange(1, k + 1, device=query.device).expand(b, k)
                if self.training:
                    vocabulary = torch.stack([
                        torch.randperm(self.instances.num_embeddings - 1, device=query.device)[:k]
                        + 1 for _ in range(b)
                    ])
                    ids = vocabulary
            else:
                ids = instance_ids
            if ids.shape != support_mask.shape:
                raise ValueError("one instance ID is required per support recording")
            valid_ids = ids[support_mask]
            if bool(((valid_ids <= 0) | (valid_ids >= self.instances.num_embeddings)).any()):
                raise ValueError("support instance IDs must be positive and within capacity")
            for row in range(b):
                active = ids[row, support_mask[row]]
                if len(active.unique()) != len(active):
                    raise ValueError("distinct support recordings must have distinct instance IDs")
            ids = ids.masked_fill(~support_mask, 0)
            tokens += [self._token(self.signal(support), SUPPORT, ids),
                       self._token(self.text(support_labels), SUPPORT_LABEL, ids)]
            masks += [support_mask, support_mask]
        hidden = self.stack(torch.cat(tokens, dim=1), key_padding_mask=torch.cat(masks, dim=1))
        # Score in fp32; one bounded positive scale is shared across every candidate. Autocast
        # would otherwise run the einsum in bf16 (``.float()`` on its inputs does not prevent that),
        # which quantises the logits and made ``index_copy`` into the fp32 output fail on CUDA.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            query_out = F.normalize(hidden[:, 0].float(), dim=-1)
            label_out = F.normalize(hidden[:, 1:1 + c].float(), dim=-1)
            scale = self.log_scale.float().clamp(math.log(0.01), math.log(100.0)).exp()
            scores = scale * torch.einsum("bd,bcd->bc", query_out, label_out)
        return scores.masked_fill(~candidate_mask, -1e30)


class DualRecordingAttention(nn.Module):
    """Independent head weights; dispatch depends only on enrollment availability."""

    def __init__(self, spec, **kwargs):
        super().__init__()
        self.zero_shot = RecordingAttentionHead(spec, **kwargs)
        self.enrollment = RecordingAttentionHead(spec, **kwargs)

    def forward(self, *, query, candidates, candidate_mask, support, support_labels,
                support_mask, enrolled, instance_ids=None):
        out = query.new_zeros(query.shape[0], candidates.shape[1], dtype=torch.float32)
        for selected, head in ((~enrolled, self.zero_shot), (enrolled, self.enrollment)):
            index = selected.nonzero(as_tuple=True)[0]
            if not len(index):
                continue
            kwargs = {}
            if head is self.enrollment:
                kwargs = dict(support=support[index], support_labels=support_labels[index],
                              support_mask=support_mask[index],
                              instance_ids=None if instance_ids is None else instance_ids[index])
            value = head(query=query[index], candidates=candidates[index],
                         candidate_mask=candidate_mask[index], **kwargs)
            out = out.index_copy(0, index, value.to(out.dtype))
        return out


def neighbor_logits(query, support, support_bound, support_mask, candidate_mask,
                    temperature=0.07):
    """Log summed neighbor mass: CE equals -log P(correct-label neighbor).

    No learned projection, centering, second probability squash, or detached support vectors.
    Empty sets produce a uniform diagnostic floor; training rejects missing true-label supports.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("neighbor temperature must be finite and positive")
    b, k = support_mask.shape
    c = candidate_mask.shape[1]
    if not bool(candidate_mask.any(dim=1).all()):
        raise ValueError("each episode needs at least one valid candidate")
    if bool(((support_bound[support_mask] < 0) | (support_bound[support_mask] >= c)).any()):
        raise ValueError("neighbor classification requires enrolled support labels")
    if k and bool((support_mask & ~candidate_mask.gather(1, support_bound.clamp(0, c - 1))).any()):
        raise ValueError("support cannot refer to a padded candidate")
    if k == 0:
        return query.new_zeros((b, c), dtype=torch.float32), query.new_zeros((b, 0), dtype=torch.float32)
    # fp32 throughout: under autocast the einsum would run in bf16 and the temperature-scaled
    # scores (up to ~14) would be quantised to ~0.06 before the log-sum-exp.
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = torch.einsum("bd,bkd->bk", F.normalize(query.float(), dim=-1),
                              F.normalize(support.float(), dim=-1)) / temperature
    scores = scores.masked_fill(~support_mask, -1e30)
    assignment = support_bound.unsqueeze(-1).eq(torch.arange(c, device=query.device))
    assignment = assignment & support_mask.unsqueeze(-1)
    logits = torch.logsumexp(scores.unsqueeze(-1).masked_fill(~assignment, -1e30), dim=1)
    logits = logits.masked_fill(~assignment.any(dim=1), -1e30)
    empty = ~support_mask.any(dim=1)
    logits = torch.where(empty.unsqueeze(1), torch.zeros_like(logits), logits)
    weights = scores.softmax(dim=1).masked_fill(~support_mask, 0.0)
    return logits.masked_fill(~candidate_mask, -1e30), weights
