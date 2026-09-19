"""ABANDONED NEGATIVE RESULT: contextual residual classifier with a neighbour floor.

Architecture ``support_contextual_residual_v1`` preserved useful zero-support behavior but
degraded its stronger support floor as enrollment grew. It remains strictly loadable only to
reproduce the recorded 2026-09-19 result; new ``--classifier contextual`` runs do not build it.

The head preserves the deployment-time evidence decomposition instead of asking a transformer to
invent logits directly.  Query, support, support-label, candidate-label and acquisition tokens are
first contextualised as one unordered set.  The resulting states may (1) correct individual
query/support comparisons and (2) estimate a candidate-local semantic weight.  Candidate logits
are still assembled from an auditable support vote plus query/label semantic evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import AttentionSpec, ScaledSum, SetAttentionStack
from model.support.roles import (
    N_ROLES, ROLE_CANDIDATE, ROLE_QUERY, ROLE_SUPPORT, ROLE_SUPPORT_LABEL,
)
from training.support_classifier.neighbors import differentiable_neighbor_logits

ARCHITECTURE_VERSION = "support_contextual_residual_v1"
NEG = -1e30
LOG_FILL = -1e4


@dataclass(frozen=True)
class ContextualResidualClassifierConfig:
    text_dim: int = 384
    input_dim: int | None = None
    acquisition_dim: int | None = None
    n_layers: int = 2
    max_candidates: int = 256
    max_supports: int = 8192
    support_temperature: float = 0.07
    semantic_temperature: float = 0.07
    temperature_floor: float = 1e-3
    semantic_hidden: int | None = None
    supported_semantic_init: float = 1e-3
    zero_support_semantic_init: float = 1.0
    max_log_weight_residual: float = 3.0

    def __post_init__(self) -> None:
        if self.text_dim < 1 or self.n_layers < 0:
            raise ValueError("invalid contextual residual capacity")
        if self.max_candidates < 2 or self.max_supports < 0:
            raise ValueError("invalid contextual residual episode capacity")
        if self.input_dim is not None and self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.acquisition_dim is not None and self.acquisition_dim < 1:
            raise ValueError("acquisition_dim must be positive")
        if min(self.support_temperature, self.semantic_temperature) <= self.temperature_floor:
            raise ValueError("initial temperatures must exceed temperature_floor")
        if min(self.supported_semantic_init, self.zero_support_semantic_init) <= 0:
            raise ValueError("semantic initial weights must be positive")
        if self.max_log_weight_residual <= 0:
            raise ValueError("max_log_weight_residual must be positive")


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _masked_log_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.log_softmax(scores.masked_fill(~mask, LOG_FILL), dim=dim)


def _identity_linear(width: int) -> nn.Linear:
    layer = nn.Linear(width, width, bias=False)
    nn.init.eye_(layer.weight)
    return layer


class ContextualResidualSupportClassifier(nn.Module):
    """Contextual corrections around a centred differentiable-neighbour support floor."""

    def __init__(
        self, spec: AttentionSpec,
        cfg: ContextualResidualClassifierConfig | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.cfg = cfg or ContextualResidualClassifierConfig()
        d = spec.d_model
        e = self.cfg.input_dim or d
        a = self.cfg.acquisition_dim or d
        hidden = self.cfg.semantic_hidden or max(1, d // 2)

        self.motion_adapter = nn.Sequential(nn.LayerNorm(e), nn.Linear(e, d))
        self.acquisition_adapter = nn.Sequential(nn.LayerNorm(a), nn.Linear(a, d))
        self.text_adapter = nn.Sequential(nn.LayerNorm(self.cfg.text_dim), nn.Linear(self.cfg.text_dim, d))
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.support_pair_tag = nn.Embedding(self.cfg.max_supports + 1, d)
        self.query_token_sum = ScaledSum(3, init=[1.0, 0.25, 0.15])
        self.support_token_sum = ScaledSum(4, init=[1.0, 0.25, 0.15, 0.10])
        self.label_token_sum = ScaledSum(3, init=[1.0, 0.15, 0.10])
        self.candidate_token_sum = ScaledSum(2, init=[1.0, 0.15])
        self.stack = SetAttentionStack(spec, self.cfg.n_layers)

        # Candidate-specific support correction.  The candidate projection starts at zero, so the
        # complete support path is exactly the centred neighbour floor at initialisation.
        self.correction_query = nn.Linear(d, d, bias=False)
        self.correction_support = nn.Linear(d, d, bias=False)
        self.correction_label = nn.Linear(d, d, bias=False)
        self.correction_candidate = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.correction_candidate.weight)

        # Zero-support and enrolled semantic maps are mechanically identical but independently
        # parameterised because they solve different information conditions.
        self.semantic_query_zero = _identity_linear(d)
        self.semantic_candidate_zero = _identity_linear(d)
        self.semantic_query_enrolled = _identity_linear(d)
        self.semantic_candidate_enrolled = _identity_linear(d)

        # One shared candidate-local network works for any C.  Its zero output preserves the two
        # explicit initial priors. The output projection learns first; upstream gate layers begin
        # receiving gradient as soon as that projection leaves zero.
        self.semantic_gate = nn.Sequential(
            nn.LayerNorm(2 * d + 1), nn.Linear(2 * d + 1, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.semantic_gate[-1].weight)
        nn.init.zeros_(self.semantic_gate[-1].bias)
        self.raw_semantic_base = nn.Parameter(torch.tensor([
            _inverse_softplus(self.cfg.zero_support_semantic_init),
            _inverse_softplus(self.cfg.supported_semantic_init),
        ], dtype=torch.float32))
        self.raw_temperatures = nn.Parameter(torch.tensor([
            _inverse_softplus(self.cfg.support_temperature - self.cfg.temperature_floor),
            _inverse_softplus(self.cfg.semantic_temperature - self.cfg.temperature_floor),
        ], dtype=torch.float32))

    def temperatures(self) -> torch.Tensor:
        return F.softplus(self.raw_temperatures.float()) + self.cfg.temperature_floor

    def telemetry(self) -> dict[str, float]:
        tau = self.temperatures().detach()
        return {
            "classifier/tau_support": float(tau[0]),
            "classifier/tau_semantic": float(tau[1]),
            "classifier/semantic_base_zero": float(F.softplus(self.raw_semantic_base[0]).detach()),
            "classifier/semantic_base_enrolled": float(F.softplus(self.raw_semantic_base[1]).detach()),
        }

    def _role(self, content: torch.Tensor, role: int) -> torch.Tensor:
        ids = torch.full(content.shape[:-1], role, dtype=torch.long, device=content.device)
        return self.role_emb(ids)

    def _check(
        self, query: torch.Tensor, support: torch.Tensor,
        query_acquisition: torch.Tensor, support_acquisition: torch.Tensor,
        support_text: torch.Tensor, support_bound: torch.Tensor,
        support_mask: torch.Tensor, pair_slot: torch.Tensor,
        candidate_text: torch.Tensor, candidate_mask: torch.Tensor,
    ) -> None:
        b, c = candidate_mask.shape
        s = support.shape[1]
        e = self.cfg.input_dim or self.spec.d_model
        a = self.cfg.acquisition_dim or self.spec.d_model
        if query.shape != (b, e) or support.shape != (b, s, e):
            raise ValueError("query/support feature shapes do not agree")
        if query_acquisition.shape != (b, a) or support_acquisition.shape != (b, s, a):
            raise ValueError("query/support acquisition shapes do not agree")
        if support_text.shape != (b, s, self.cfg.text_dim):
            raise ValueError("support-label text shape does not agree")
        if candidate_text.shape != (b, c, self.cfg.text_dim):
            raise ValueError("candidate-label text shape does not agree")
        if support_bound.shape != support_mask.shape or support_mask.shape != (b, s):
            raise ValueError("support binding/mask shapes do not agree")
        if pair_slot.shape != (b, s):
            raise ValueError("support pair-slot shape does not agree")
        if c > self.cfg.max_candidates or s > self.cfg.max_supports:
            raise ValueError("episode exceeds contextual residual capacity")
        if not bool(candidate_mask.any(dim=1).all()):
            raise ValueError("every query needs at least one candidate")
        invalid = support_mask & ((support_bound < -1) | (support_bound >= c))
        if bool(invalid.any()):
            raise ValueError("valid support has an invalid candidate binding")

    @staticmethod
    def _centre(
        query: torch.Tensor, support: torch.Tensor, support_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        denom = support_mask.sum(dim=1, keepdim=True).clamp_min(1).to(support.dtype)
        mean = (support * support_mask.unsqueeze(-1).to(support.dtype)).sum(dim=1) / denom
        # Zero-support rows must not be translated by an artificial zero mean.
        mean = torch.where(support_mask.any(1, keepdim=True), mean, torch.zeros_like(mean))
        return query.float() - mean.float(), support.float() - mean[:, None].float()

    @staticmethod
    def _counts(bound: torch.Tensor, mask: torch.Tensor, candidates: int) -> torch.Tensor:
        count = torch.zeros((bound.shape[0], candidates), device=bound.device, dtype=torch.float32)
        exact = mask & bound.ge(0)
        if bound.shape[1]:
            count.scatter_add_(1, bound.clamp_min(0), exact.float())
        return count

    def forward(
        self, *, query_feature: torch.Tensor, support_feature: torch.Tensor,
        query_acquisition: torch.Tensor, support_acquisition: torch.Tensor,
        support_label_text: torch.Tensor, support_bound: torch.Tensor,
        support_mask: torch.Tensor, support_pair_slot: torch.Tensor,
        candidate_text: torch.Tensor, candidate_mask: torch.Tensor,
        candidate_slot: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        del candidate_slot
        support_mask = support_mask.bool()
        candidate_mask = candidate_mask.bool()
        self._check(
            query_feature, support_feature, query_acquisition, support_acquisition,
            support_label_text, support_bound, support_mask, support_pair_slot,
            candidate_text, candidate_mask,
        )
        b, c = candidate_mask.shape
        s = support_feature.shape[1]
        device_type = query_feature.device.type

        support_feature = torch.where(
            support_mask.unsqueeze(-1), support_feature, torch.zeros_like(support_feature),
        )
        support_acquisition = torch.where(
            support_mask.unsqueeze(-1), support_acquisition, torch.zeros_like(support_acquisition),
        )
        support_label_text = torch.where(
            support_mask.unsqueeze(-1), support_label_text, torch.zeros_like(support_label_text),
        )
        candidate_text = torch.where(
            candidate_mask.unsqueeze(-1), candidate_text, torch.zeros_like(candidate_text),
        )
        q0, s0 = self._centre(query_feature, support_feature, support_mask)

        q_motion = self.motion_adapter(q0).unsqueeze(1)
        s_motion = self.motion_adapter(s0)
        q_acq = self.acquisition_adapter(query_acquisition).unsqueeze(1)
        s_acq = self.acquisition_adapter(support_acquisition)
        labels = self.text_adapter(support_label_text)
        candidates = self.text_adapter(candidate_text)
        pair = self.support_pair_tag(support_pair_slot.clamp(min=0, max=self.cfg.max_supports))
        tokens = torch.cat((
            self.query_token_sum(q_motion, q_acq, self._role(q_motion, ROLE_QUERY)),
            self.support_token_sum(s_motion, s_acq, self._role(s_motion, ROLE_SUPPORT), pair),
            self.label_token_sum(labels, self._role(labels, ROLE_SUPPORT_LABEL), pair),
            self.candidate_token_sum(candidates, self._role(candidates, ROLE_CANDIDATE)),
        ), dim=1)
        valid = torch.cat((
            torch.ones((b, 1), dtype=torch.bool, device=tokens.device),
            support_mask, support_mask, candidate_mask,
        ), dim=1)
        hidden = self.stack(tokens, key_padding_mask=valid)
        q_h = hidden[:, 0]
        s_h = hidden[:, 1:1 + s]
        l_h = hidden[:, 1 + s:1 + 2 * s]
        c_h = hidden[:, 1 + 2 * s:]

        with torch.autocast(device_type=device_type, enabled=False):
            tau_support, tau_semantic = self.temperatures()
            has_any_support = support_mask.any(dim=1)
            count = self._counts(support_bound, support_mask, c)
            candidate_has_support = count.gt(0)
            uniform = -candidate_mask.sum(dim=1, keepdim=True).float().log()

            support_floor = uniform.expand(-1, c).clone()
            support_weight = q0.new_zeros((b, s))
            exact_mask = support_mask & support_bound.ge(0)
            exact_episode = exact_mask.any(dim=1)
            if s and bool(exact_episode.any()):
                floor, weight = differentiable_neighbor_logits(
                    q0[exact_episode], s0[exact_episode], support_bound[exact_episode],
                    exact_mask[exact_episode], candidate_mask[exact_episode],
                    # The reference floor remains the fixed differentiable-neighbour control;
                    # the contextual support path may learn its own temperature above it.
                    temperature=self.cfg.support_temperature,
                )
                # Missing candidates in partial enrollment retain a neutral uniform prior instead
                # of becoming impossible solely because no demonstration was supplied.
                local_count = count[exact_episode]
                floor = torch.where(local_count.gt(0), floor, uniform[exact_episode])
                support_floor[exact_episode] = floor
                support_weight[exact_episode] = weight

            qn = F.normalize(q0.float(), dim=-1)
            sn = F.normalize(s0.float(), dim=-1)
            motion_score = torch.einsum("bd,bsd->bs", qn, sn) / tau_support
            motion_score = motion_score.masked_fill(~support_mask, LOG_FILL)
            if s:
                u = torch.tanh(
                    self.correction_query(q_h.float()).unsqueeze(1)
                    + self.correction_support(s_h.float())
                    + self.correction_label(l_h.float())
                )
                v = self.correction_candidate(c_h.float())
                correction = torch.einsum("bsd,bcd->bsc", u, v) / math.sqrt(self.spec.d_model)
                correction = torch.tanh(correction)

                semantic_binding = torch.einsum(
                    "bsd,bcd->bsc", F.normalize(l_h.float(), dim=-1),
                    F.normalize(c_h.float(), dim=-1),
                ) / tau_semantic
                semantic_binding = _masked_log_softmax(
                    semantic_binding,
                    candidate_mask.unsqueeze(1).expand(-1, s, -1), dim=2,
                )
                exact = support_mask & support_bound.ge(0)
                exact_binding = semantic_binding.new_full((b, s, c), LOG_FILL)
                if s:
                    exact_binding.scatter_(2, support_bound.clamp_min(0).unsqueeze(-1), 0.0)
                label_logp = torch.where(exact.unsqueeze(-1), exact_binding, semantic_binding)
                pair_valid = support_mask.unsqueeze(-1) & candidate_mask.unsqueeze(1)
                pair_score = motion_score.unsqueeze(-1) + correction + label_logp
                pair_logp = _masked_log_softmax(pair_score.flatten(1), pair_valid.flatten(1), dim=1)
                support_mass = torch.logsumexp(pair_logp.reshape(b, s, c), dim=1)
                off_roster = (support_mask & support_bound.lt(0)).any(dim=1, keepdim=True)
                evidence_present = candidate_has_support | off_roster
                contextual_support = torch.where(evidence_present, support_mass, uniform)
            else:
                correction = q_h.new_zeros((b, 0, c), dtype=torch.float32)
                pair_logp = q_h.new_zeros((b, 0), dtype=torch.float32)
                contextual_support = uniform.expand(-1, c).clone()

            q_zero = F.normalize(self.semantic_query_zero(q_h.float()), dim=-1)
            c_zero = F.normalize(self.semantic_candidate_zero(c_h.float()), dim=-1)
            q_enrolled = F.normalize(self.semantic_query_enrolled(q_h.float()), dim=-1)
            c_enrolled = F.normalize(self.semantic_candidate_enrolled(c_h.float()), dim=-1)
            semantic_zero = torch.einsum("bd,bcd->bc", q_zero, c_zero) / tau_semantic
            semantic_enrolled = torch.einsum("bd,bcd->bc", q_enrolled, c_enrolled) / tau_semantic
            semantic_score = torch.where(has_any_support.unsqueeze(1), semantic_enrolled, semantic_zero)
            semantic_logp = _masked_log_softmax(semantic_score, candidate_mask, dim=1)

            gate_input = torch.cat((
                q_h.float().unsqueeze(1).expand(-1, c, -1), c_h.float(),
                torch.log1p(count).unsqueeze(-1),
            ), dim=-1)
            gate_residual = self.cfg.max_log_weight_residual * torch.tanh(
                self.semantic_gate(gate_input).squeeze(-1)
            )
            state = has_any_support.long().unsqueeze(1).expand(-1, c)
            semantic_weight = F.softplus(self.raw_semantic_base[state] + gate_residual)
            semantic_weight = semantic_weight.masked_fill(~candidate_mask, 0.0)
            semantic_evidence = semantic_logp - uniform
            combined = contextual_support + semantic_weight * semantic_evidence
            logits = _masked_log_softmax(combined, candidate_mask, dim=1)

        out = {
            "logits": logits.masked_fill(~candidate_mask, NEG),
            "support_floor_logits": support_floor.masked_fill(~candidate_mask, NEG),
            "neighbor_logits": support_floor.masked_fill(~candidate_mask, NEG),
            "contextual_support_logits": contextual_support.masked_fill(~candidate_mask, NEG),
            "semantic_logits": semantic_logp.masked_fill(~candidate_mask, NEG),
            "semantic_weight": semantic_weight,
            "support_weight": support_weight,
            "support_correction": correction,
            "k_c": count,
            "has_support": has_any_support,
            "temperatures": torch.stack((tau_support, tau_semantic)),
        }
        if return_diagnostics:
            out["pair_log_probability"] = pair_logp
            out["semantic_gate_residual"] = gate_residual
        return out

    @staticmethod
    def branch_logits(output: dict[str, torch.Tensor], branch: str) -> torch.Tensor:
        key = {
            "final": "logits",
            "support_floor": "support_floor_logits",
            "contextual_support": "contextual_support_logits",
            "semantic": "semantic_logits",
        }.get(branch)
        if key is None:
            raise ValueError(f"unknown contextual residual branch {branch!r}")
        return output[key]
