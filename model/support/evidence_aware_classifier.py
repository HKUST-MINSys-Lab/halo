"""Evidence-aware support classifier.

This is the active successor to the contextual residual head.  It keeps the
two auditable sources of evidence separate: a support vote over recordings and
a query-to-candidate-label semantic score.  Set attention sees the *status quo*
from both paths and only refines the support comparison and candidate-local
mixture weight; it never emits unconstrained class logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import AttentionSpec, ScaledSum, SetAttentionStack
from model.support.roles import N_ROLES, ROLE_CANDIDATE, ROLE_QUERY, ROLE_SUPPORT, ROLE_SUPPORT_LABEL
from training.support_classifier.neighbors import differentiable_neighbor_logits

ARCHITECTURE_VERSION = "support_evidence_aware_v2"
NEG = -1e30
LOG_FILL = -1e4


@dataclass(frozen=True)
class EvidenceAwareClassifierConfig:
    text_dim: int = 384
    input_dim: int | None = None
    acquisition_dim: int | None = None
    n_layers: int = 2
    max_candidates: int = 256
    max_supports: int = 8192
    support_temperature: float = 0.07
    semantic_temperature: float = 0.07
    temperature_floor: float = 1e-3
    router_hidden: int | None = None
    semantic_reliance_init: float = 1e-3

    def __post_init__(self) -> None:
        if self.text_dim < 1 or self.n_layers < 0 or self.max_candidates < 2 or self.max_supports < 0:
            raise ValueError("invalid evidence-aware classifier capacity")
        if self.input_dim is not None and self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.acquisition_dim is not None and self.acquisition_dim < 1:
            raise ValueError("acquisition_dim must be positive")
        if min(self.support_temperature, self.semantic_temperature) <= self.temperature_floor:
            raise ValueError("initial temperatures must exceed temperature_floor")
        if not 0.0 < self.semantic_reliance_init < 1.0:
            raise ValueError("semantic_reliance_init must be in (0, 1)")


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


def _masked_log_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.log_softmax(scores.masked_fill(~mask, LOG_FILL), dim=dim)


def _identity_linear(width: int) -> nn.Linear:
    layer = nn.Linear(width, width, bias=False)
    nn.init.eye_(layer.weight)
    return layer


class EvidenceAwareSupportClassifier(nn.Module):
    """Contextual support matching with candidate-specific evidence arbitration."""

    def __init__(self, spec: AttentionSpec, cfg: EvidenceAwareClassifierConfig | None = None) -> None:
        super().__init__()
        self.spec, self.cfg = spec, cfg or EvidenceAwareClassifierConfig()
        d, e, a = spec.d_model, self.cfg.input_dim or spec.d_model, self.cfg.acquisition_dim or spec.d_model
        hidden = self.cfg.router_hidden or max(1, d // 2)
        self.motion_adapter = nn.Sequential(nn.LayerNorm(e), nn.Linear(e, d))
        self.acquisition_adapter = nn.Sequential(nn.LayerNorm(a), nn.Linear(a, d))
        self.text_adapter = nn.Sequential(nn.LayerNorm(self.cfg.text_dim), nn.Linear(self.cfg.text_dim, d))
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.support_pair_tag = nn.Embedding(self.cfg.max_supports + 1, d)

        # These adapters define the pre-context semantic path.  It intentionally does not see a
        # support token, ensuring the semantic branch is invariant to enrollment changes.
        self.semantic_query = _identity_linear(d)
        self.semantic_candidate = _identity_linear(d)
        self.query_token_sum = ScaledSum(3, init=[1.0, 0.25, 0.15])
        self.support_token_sum = ScaledSum(5, init=[1.0, 0.25, 0.15, 0.10, 0.10])
        self.label_token_sum = ScaledSum(3, init=[1.0, 0.15, 0.10])
        self.candidate_token_sum = ScaledSum(3, init=[1.0, 0.15, 0.10])
        self.support_evidence = nn.Linear(1, d)
        self.candidate_evidence = nn.Linear(3, d)
        nn.init.zeros_(self.support_evidence.weight); nn.init.zeros_(self.support_evidence.bias)
        nn.init.zeros_(self.candidate_evidence.weight); nn.init.zeros_(self.candidate_evidence.bias)
        self.stack = SetAttentionStack(spec, self.cfg.n_layers)

        self.correction_query = nn.Linear(d, d, bias=False)
        self.correction_support = nn.Linear(d, d, bias=False)
        self.correction_label = nn.Linear(d, d, bias=False)
        self.correction_candidate = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.correction_candidate.weight)
        self.raw_correction_scale = nn.Parameter(torch.tensor(0.0))

        self.router = nn.Sequential(nn.LayerNorm(2 * d + 3), nn.Linear(2 * d + 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.router[-1].weight); nn.init.zeros_(self.router[-1].bias)
        self.semantic_reliance_bias = nn.Parameter(torch.tensor(_inverse_sigmoid(self.cfg.semantic_reliance_init)))
        self.raw_temperatures = nn.Parameter(torch.tensor([
            _inverse_softplus(self.cfg.support_temperature - self.cfg.temperature_floor),
            _inverse_softplus(self.cfg.semantic_temperature - self.cfg.temperature_floor),
        ], dtype=torch.float32))

    def temperatures(self) -> torch.Tensor:
        return F.softplus(self.raw_temperatures.float()) + self.cfg.temperature_floor

    def telemetry(self) -> dict[str, float]:
        tau = self.temperatures().detach()
        return {"classifier/tau_support": float(tau[0]), "classifier/tau_semantic": float(tau[1]),
                "classifier/semantic_reliance_prior": float(torch.sigmoid(self.semantic_reliance_bias).detach()),
                "classifier/correction_scale": float(F.softplus(self.raw_correction_scale).detach())}

    def _role(self, content: torch.Tensor, role: int) -> torch.Tensor:
        return self.role_emb(torch.full(content.shape[:-1], role, dtype=torch.long, device=content.device))

    @staticmethod
    def _centre(query: torch.Tensor, support: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        denom = mask.sum(1, keepdim=True).clamp_min(1).to(support.dtype)
        mean = (support * mask.unsqueeze(-1).to(support.dtype)).sum(1) / denom
        mean = torch.where(mask.any(1, keepdim=True), mean, torch.zeros_like(mean))
        return query.float() - mean.float(), support.float() - mean[:, None].float()

    @staticmethod
    def _counts(bound: torch.Tensor, mask: torch.Tensor, c: int) -> torch.Tensor:
        counts = torch.zeros((bound.shape[0], c), device=bound.device, dtype=torch.float32)
        exact = mask & bound.ge(0)
        if bound.shape[1]: counts.scatter_add_(1, bound.clamp_min(0), exact.float())
        return counts

    def _check(self, query, support, q_acq, s_acq, support_text, bound, mask, pair, candidate_text, candidate_mask) -> None:
        b, c = candidate_mask.shape; s = support.shape[1]
        e, a = self.cfg.input_dim or self.spec.d_model, self.cfg.acquisition_dim or self.spec.d_model
        if query.shape != (b, e) or support.shape != (b, s, e) or q_acq.shape != (b, a) or s_acq.shape != (b, s, a):
            raise ValueError("motion/acquisition feature shapes do not agree")
        if support_text.shape != (b, s, self.cfg.text_dim) or candidate_text.shape != (b, c, self.cfg.text_dim):
            raise ValueError("label-text shapes do not agree")
        if bound.shape != mask.shape or pair.shape != mask.shape or c > self.cfg.max_candidates or s > self.cfg.max_supports:
            raise ValueError("invalid evidence-aware episode layout")
        if not bool(candidate_mask.any(1).all()): raise ValueError("every query needs a candidate")
        if bool((mask & ((bound < -1) | (bound >= c))).any()): raise ValueError("invalid support binding")

    def forward(self, *, query_feature: torch.Tensor, support_feature: torch.Tensor,
                query_acquisition: torch.Tensor, support_acquisition: torch.Tensor,
                support_label_text: torch.Tensor, support_bound: torch.Tensor,
                support_mask: torch.Tensor, support_pair_slot: torch.Tensor,
                candidate_text: torch.Tensor, candidate_mask: torch.Tensor,
                candidate_slot: torch.Tensor | None = None, return_diagnostics: bool = False) -> dict[str, torch.Tensor]:
        del candidate_slot
        mask, cmask = support_mask.bool(), candidate_mask.bool()
        self._check(query_feature, support_feature, query_acquisition, support_acquisition, support_label_text, support_bound, mask, support_pair_slot, candidate_text, cmask)
        b, c, s = cmask.shape[0], cmask.shape[1], support_feature.shape[1]
        support_feature = torch.where(mask[..., None], support_feature, torch.zeros_like(support_feature))
        support_acquisition = torch.where(mask[..., None], support_acquisition, torch.zeros_like(support_acquisition))
        support_label_text = torch.where(mask[..., None], support_label_text, torch.zeros_like(support_label_text))
        candidate_text = torch.where(cmask[..., None], candidate_text, torch.zeros_like(candidate_text))
        q0, s0 = self._centre(query_feature, support_feature, mask)
        q_motion, s_motion = self.motion_adapter(q0).unsqueeze(1), self.motion_adapter(s0)
        q_acq, s_acq = self.acquisition_adapter(query_acquisition).unsqueeze(1), self.acquisition_adapter(support_acquisition)
        labels, candidates = self.text_adapter(support_label_text), self.text_adapter(candidate_text)
        tau_support, tau_semantic = self.temperatures()
        has_any = mask.any(1); counts = self._counts(support_bound, mask, c); has_direct = counts.gt(0)
        uniform = -cmask.sum(1, keepdim=True).float().log()

        # Status quo, calculated before attention. Semantic scores cannot use support state.
        # Do not centre the semantic query by support rows.  This is the explicit
        # query-only branch used when supports are absent or untrustworthy.
        semantic_query = self.motion_adapter(query_feature.float()) + self.acquisition_adapter(query_acquisition.float())
        semantic_status = _masked_log_softmax(torch.einsum("bd,bcd->bc", F.normalize(self.semantic_query(semantic_query.float()), dim=-1), F.normalize(self.semantic_candidate(candidates.float()), dim=-1)) / tau_semantic, cmask, 1)
        support_status = uniform.expand(-1, c).clone(); support_weight = q0.new_zeros((b, s))
        exact = mask & support_bound.ge(0)
        if s and bool(exact.any(1).any()):
            rows = exact.any(1)
            floor, weight = differentiable_neighbor_logits(q0[rows], s0[rows], support_bound[rows], exact[rows], cmask[rows], temperature=self.cfg.support_temperature)
            support_status[rows] = torch.where(counts[rows].gt(0), floor, uniform[rows])
            support_weight[rows] = weight
        support_status = _masked_log_softmax(support_status, cmask, 1)
        qsn = F.normalize(q0.float(), dim=-1); ssn = F.normalize(s0.float(), dim=-1)
        query_support_cosine = (torch.einsum("bd,bsd->bs", qsn, ssn) / tau_support).masked_fill(~mask, 0.0)
        status_features = torch.stack((support_status - uniform, semantic_status - uniform, has_direct.float()), dim=-1)
        pair = self.support_pair_tag(support_pair_slot.clamp(0, self.cfg.max_supports))
        tokens = torch.cat((
            self.query_token_sum(q_motion, q_acq, self._role(q_motion, ROLE_QUERY)),
            self.support_token_sum(s_motion, s_acq, self._role(s_motion, ROLE_SUPPORT), pair, self.support_evidence(query_support_cosine[..., None])),
            self.label_token_sum(labels, self._role(labels, ROLE_SUPPORT_LABEL), pair),
            self.candidate_token_sum(candidates, self._role(candidates, ROLE_CANDIDATE), self.candidate_evidence(status_features)),
        ), 1)
        valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=tokens.device), mask, mask, cmask), 1)
        hidden = self.stack(tokens, key_padding_mask=valid)
        q_h, s_h, l_h, c_h = hidden[:, 0], hidden[:, 1:1+s], hidden[:, 1+s:1+2*s], hidden[:, 1+2*s:]
        if s:
            u = torch.tanh(self.correction_query(q_h.float())[:, None] + self.correction_support(s_h.float()) + self.correction_label(l_h.float()))
            correction = F.softplus(self.raw_correction_scale) * torch.einsum("bsd,bcd->bsc", u, self.correction_candidate(c_h.float())) / math.sqrt(self.spec.d_model)
            binding = torch.einsum("bsd,bcd->bsc", F.normalize(l_h.float(), dim=-1), F.normalize(c_h.float(), dim=-1)) / tau_semantic
            binding = _masked_log_softmax(binding, cmask[:, None, :].expand(-1, s, -1), 2)
            exact_binding = binding.new_full((b, s, c), LOG_FILL)
            exact_binding.scatter_(2, support_bound.clamp_min(0)[..., None], 0.0)
            label_logp = torch.where(exact[..., None], exact_binding, binding)
            pair_valid = mask[..., None] & cmask[:, None, :]
            pair_logp = _masked_log_softmax((query_support_cosine[..., None] + correction + label_logp).flatten(1), pair_valid.flatten(1), 1)
            refined_support = torch.logsumexp(pair_logp.reshape(b, s, c), 1)
            evidence_present = has_direct | (mask & support_bound.lt(0)).any(1, keepdim=True)
            refined_support = torch.where(evidence_present, refined_support, uniform)
        else:
            correction = q0.new_zeros((b, 0, c)); pair_logp = q0.new_zeros((b, 0)); refined_support = uniform.expand(-1, c).clone()
        refined_support = _masked_log_softmax(refined_support, cmask, 1)
        router_features = torch.cat((q_h.float()[:, None].expand(-1, c, -1), c_h.float(), status_features), -1)
        semantic_reliance = torch.sigmoid(self.semantic_reliance_bias + self.router(router_features).squeeze(-1))
        semantic_reliance = torch.where(has_any[:, None], semantic_reliance, torch.ones_like(semantic_reliance)).masked_fill(~cmask, 0.0)
        log_g = semantic_reliance.clamp_min(torch.finfo(semantic_reliance.dtype).tiny).log()
        log_not_g = (1.0 - semantic_reliance).clamp_min(torch.finfo(semantic_reliance.dtype).tiny).log()
        logits = _masked_log_softmax(torch.logaddexp(log_not_g + refined_support, log_g + semantic_status), cmask, 1)
        out = {"logits": logits.masked_fill(~cmask, NEG), "support_status_logits": support_status.masked_fill(~cmask, NEG),
               "support_floor_logits": support_status.masked_fill(~cmask, NEG), "neighbor_logits": support_status.masked_fill(~cmask, NEG),
               "refined_support_logits": refined_support.masked_fill(~cmask, NEG), "contextual_support_logits": refined_support.masked_fill(~cmask, NEG),
               "semantic_status_logits": semantic_status.masked_fill(~cmask, NEG), "semantic_logits": semantic_status.masked_fill(~cmask, NEG),
               "semantic_reliance": semantic_reliance, "semantic_weight": semantic_reliance, "query_support_cosine": query_support_cosine,
               "support_weight": support_weight, "support_correction": correction, "candidate_has_direct_support": has_direct,
               "has_any_support": has_any, "has_support": has_any, "k_c": counts, "temperatures": torch.stack((tau_support, tau_semantic)),
               "correction_scale": F.softplus(self.raw_correction_scale)}
        if return_diagnostics: out["pair_log_probability"] = pair_logp
        return out

    @staticmethod
    def branch_logits(output: dict[str, torch.Tensor], branch: str) -> torch.Tensor:
        keys = {"final": "logits", "support_status": "support_status_logits", "support_floor": "support_status_logits",
                "refined_support": "refined_support_logits", "contextual_support": "refined_support_logits",
                "semantic": "semantic_status_logits"}
        if branch not in keys: raise ValueError(f"unknown evidence-aware branch {branch!r}")
        return output[keys[branch]]
