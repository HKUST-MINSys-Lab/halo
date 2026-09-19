"""ABANDONED NEGATIVE RESULT: contextualize-first semantic-voting classifier.

Architecture ``support_contextual_mixture_v1`` collapsed onto its semantic branch and is not a
candidate design. It remains strictly loadable only to reproduce the recorded 2026-09-18 result.

Implements the approved 2026-09-17 plan: every supplied token (query, support recordings,
support labels, candidate labels) is contextualised by one set-attention stack *before* either
comparison is made. The support path votes with similarity-weighted supports whose votes are
distributed among candidates by label similarity; the semantic path compares the contextualised
query with the contextualised candidates directly. A per-candidate sigmoid gate mixes the two
normalised distributions in log space. There is no support-to-candidate binding, no scalar
residual, no count-bucket lambda and no hand-crafted gate feature.

Unlike the residual head this model is not identity-initialised to a closed-form floor and makes
no claim that it cannot perform worse than 1-NN; that floor is kept elsewhere as a diagnostic.
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

ARCHITECTURE_VERSION = "support_contextual_mixture_v1"
NEG = -1e30          # output sentinel for invalid candidates
LOG_FILL = -1e4      # finite log-space fill: exp() underflows to 0 with a zero, finite gradient


@dataclass(frozen=True)
class ContextualClassifierConfig:
    text_dim: int = 384
    input_dim: int | None = None        # encoder recording-vector width E; None means d_model
    n_layers: int = 2
    max_candidates: int = 256
    max_supports: int = 8192
    init_temperature: float = 0.07      # shared initial value of the three learned temperatures
    temperature_floor: float = 1e-3
    gate_hidden: int | None = None      # None means d_model // 2
    gate_output_std: float = 1e-3

    def __post_init__(self) -> None:
        if self.text_dim < 1 or self.n_layers < 0 or self.max_candidates < 2 or self.max_supports < 0:
            raise ValueError("invalid contextual-classifier capacity")
        if self.input_dim is not None and self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.temperature_floor <= 0 or self.init_temperature <= self.temperature_floor:
            raise ValueError("init_temperature must exceed a positive temperature_floor")
        if self.gate_hidden is not None and self.gate_hidden < 1:
            raise ValueError("gate_hidden must be positive")
        if self.gate_output_std <= 0:
            raise ValueError("gate_output_std must be positive")


def masked_log_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Log-softmax over the valid entries of ``dim``.

    Masked entries are filled with a large finite negative rather than -inf: their softmax mass
    underflows to exactly zero, so valid entries are unchanged, while downstream ``logsumexp`` and
    ``logaddexp`` backward passes never see ``-inf - -inf`` (which is NaN even under a zero upstream
    gradient). Rows with no valid entry are the caller's responsibility.
    """
    return torch.log_softmax(scores.masked_fill(~mask, LOG_FILL), dim=dim)


class ContextualSupportClassifier(nn.Module):
    """Contextualise first, then vote through supports and compare to labels directly."""

    def __init__(self, spec: AttentionSpec, cfg: ContextualClassifierConfig | None = None):
        super().__init__()
        self.spec, self.cfg = spec, cfg or ContextualClassifierConfig()
        d = spec.d_model
        e = self.cfg.input_dim or d
        t = self.cfg.text_dim
        self.motion_adapter = nn.Sequential(nn.LayerNorm(e), nn.Linear(e, d))
        self.text_adapter = nn.Sequential(nn.LayerNorm(t), nn.Linear(t, d))
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.support_pair_tag = nn.Embedding(self.cfg.max_supports + 1, d)
        # Content leads; identity channels enter conservatively (see ScaledSum).
        self.query_token_sum = ScaledSum(2, init=[1.0, 0.15])
        self.support_token_sum = ScaledSum(3, init=[1.0, 0.15, 0.10])
        self.label_token_sum = ScaledSum(3, init=[1.0, 0.15, 0.10])
        self.candidate_token_sum = ScaledSum(2, init=[1.0, 0.15])
        self.stack = SetAttentionStack(spec, self.cfg.n_layers)
        self.p_motion = nn.Linear(d, d, bias=False)
        self.p_text = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.p_motion.weight)
        nn.init.eye_(self.p_text.weight)
        hidden = self.cfg.gate_hidden or max(1, d // 2)
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.gate[-1].weight, std=self.cfg.gate_output_std)
        nn.init.zeros_(self.gate[-1].bias)
        raw = math.log(math.expm1(self.cfg.init_temperature - self.cfg.temperature_floor))
        # Order: motion (query-support), label (support-label to candidate), semantic (query-candidate).
        self.raw_temperatures = nn.Parameter(torch.full((3,), raw))

    # ------------------------------------------------------------------ helpers
    def temperatures(self) -> torch.Tensor:
        return F.softplus(self.raw_temperatures.float()) + self.cfg.temperature_floor

    def telemetry(self) -> dict[str, float]:
        tau = self.temperatures().detach()
        return {"classifier/tau_motion": float(tau[0]), "classifier/tau_label": float(tau[1]),
                "classifier/tau_semantic": float(tau[2])}

    def _role(self, content: torch.Tensor, role: int) -> torch.Tensor:
        return self.role_emb(torch.full(content.shape[:-1], role, dtype=torch.long, device=content.device))

    def _check(self, query, support, support_text, support_mask, pair_slot, candidate_text, candidate_mask):
        b, c = candidate_mask.shape
        s = support.shape[1]
        e = self.cfg.input_dim or self.spec.d_model
        if query.shape != (b, e) or support.shape != (b, s, e):
            raise ValueError("query/support recording shapes do not agree with the head's input width")
        if support_text.shape != (b, s, self.cfg.text_dim) or candidate_text.shape != (b, c, self.cfg.text_dim):
            raise ValueError("label text dimensions do not agree")
        if support_mask.shape != (b, s) or pair_slot.shape != (b, s):
            raise ValueError("support mask/tag shapes do not agree")
        if c > self.cfg.max_candidates or s > self.cfg.max_supports:
            raise ValueError("episode exceeds contextual-classifier capacity")
        if not bool(candidate_mask.any(dim=1).all()):
            raise ValueError("every query needs at least one valid candidate")
        if s and bool(((pair_slot < 0) | (pair_slot > self.cfg.max_supports)).any()):
            raise ValueError("support pair tag outside capacity")

    # ------------------------------------------------------------------ forward
    def forward(
        self, *,
        query_feature: torch.Tensor,        # (B, E)
        support_feature: torch.Tensor,      # (B, S, E), S may be 0
        support_label_text: torch.Tensor,   # (B, S, T)
        support_mask: torch.Tensor,         # (B, S) bool
        support_pair_slot: torch.Tensor,    # (B, S) long, 0 = padding
        candidate_text: torch.Tensor,       # (B, C, T)
        candidate_mask: torch.Tensor,       # (B, C) bool
        support_bound: torch.Tensor | None = None,   # accepted for call compatibility; unused
        candidate_slot: torch.Tensor | None = None,  # accepted for call compatibility; unused
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        del support_bound, candidate_slot
        support_mask = support_mask.bool()
        candidate_mask = candidate_mask.bool()
        self._check(query_feature, support_feature, support_label_text, support_mask,
                    support_pair_slot, candidate_text, candidate_mask)
        b, c = candidate_mask.shape
        s = support_feature.shape[1]
        device_type = query_feature.device.type

        # Zero invalid rows before any adapter so padding can never carry non-finite content.
        # A select is required: multiplying NaN padding by a zero mask leaves NaN.
        support_feature = torch.where(support_mask.unsqueeze(-1), support_feature, torch.zeros_like(support_feature))
        support_label_text = torch.where(support_mask.unsqueeze(-1), support_label_text, torch.zeros_like(support_label_text))
        candidate_text = torch.where(candidate_mask.unsqueeze(-1), candidate_text, torch.zeros_like(candidate_text))

        q_in = self.motion_adapter(query_feature.unsqueeze(1))
        s_in = self.motion_adapter(support_feature)
        l_in = self.text_adapter(support_label_text)
        c_in = self.text_adapter(candidate_text)
        pair = self.support_pair_tag(support_pair_slot.clamp_min(0))
        tokens = torch.cat((
            self.query_token_sum(q_in, self._role(q_in, ROLE_QUERY)),
            self.support_token_sum(s_in, self._role(s_in, ROLE_SUPPORT), pair),
            self.label_token_sum(l_in, self._role(l_in, ROLE_SUPPORT_LABEL), pair),
            self.candidate_token_sum(c_in, self._role(c_in, ROLE_CANDIDATE)),
        ), dim=1)
        valid = torch.cat((
            torch.ones((b, 1), dtype=torch.bool, device=tokens.device),
            support_mask, support_mask, candidate_mask,
        ), dim=1)
        hidden = self.stack(tokens, key_padding_mask=valid)
        q_h = hidden[:, 0]
        zero = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
        s_h = torch.where(support_mask.unsqueeze(-1), hidden[:, 1:1 + s], zero)
        l_h = torch.where(support_mask.unsqueeze(-1), hidden[:, 1 + s:1 + 2 * s], zero)
        c_h = torch.where(candidate_mask.unsqueeze(-1), hidden[:, 1 + 2 * s:], zero)

        # Gate on contextualised query/candidate states (bf16-safe MLP), read in FP32.
        gate_in = torch.cat((q_h.unsqueeze(1).expand(-1, c, -1), c_h), dim=-1)
        g = self.gate(gate_in).squeeze(-1).float()

        with torch.autocast(device_type=device_type, enabled=False):
            tau = self.temperatures()
            q_m = F.normalize(self.p_motion(q_h.float()), dim=-1)
            c_t = F.normalize(self.p_text(c_h.float()), dim=-1)
            z_sem = torch.einsum("bd,bcd->bc", q_m, c_t) / tau[2]
            log_p_sem = masked_log_softmax(z_sem, candidate_mask, dim=1)

            has_support = support_mask.any(dim=1)
            log_p_sup = log_p_sem
            log_w = q_m.new_full((b, s), LOG_FILL)
            log_label = None
            if s and bool(has_support.any()):
                s_m = F.normalize(self.p_motion(s_h.float()), dim=-1)
                l_t = F.normalize(self.p_text(l_h.float()), dim=-1)
                a = torch.einsum("bd,bsd->bs", q_m, s_m) / tau[0]
                rows = has_support.nonzero(as_tuple=True)[0]
                log_w = log_w.clone()
                log_w[rows] = masked_log_softmax(a[rows], support_mask[rows], dim=1)
                label_scores = torch.einsum("bsd,bcd->bsc", l_t, c_t) / tau[1]
                log_label = masked_log_softmax(label_scores, candidate_mask.unsqueeze(1).expand(-1, s, -1), dim=2)
                voted = torch.logsumexp(log_w.unsqueeze(-1) + log_label, dim=1)      # (B, C)
                # Rows without support keep the semantic distribution; never let a -inf/NaN
                # support vote reach the mixture.
                log_p_sup = torch.where(has_support.unsqueeze(1), voted, log_p_sem)

            log_mass = torch.logaddexp(F.logsigmoid(-g) + log_p_sup, F.logsigmoid(g) + log_p_sem)
            log_mass = log_mass.masked_fill(~candidate_mask, LOG_FILL)
            mixed = log_mass - torch.logsumexp(log_mass, dim=1, keepdim=True)
            log_p = torch.where(has_support.unsqueeze(1), mixed, log_p_sem)

        alpha = torch.sigmoid(g).masked_fill(~candidate_mask, 0.0)
        support_weight = torch.where(support_mask, log_w.exp(), torch.zeros_like(log_w))
        out = {
            "logits": log_p.masked_fill(~candidate_mask, NEG),
            "log_p_support": log_p_sup.masked_fill(~candidate_mask, NEG),
            "log_p_semantic": log_p_sem.masked_fill(~candidate_mask, NEG),
            "semantic_weight": alpha,
            "gate_logit": g,
            "has_support": has_support,
            "support_weight": support_weight,
            "temperatures": tau,
        }
        if return_diagnostics:
            out["log_support_weight"] = log_w
            out["log_label_distribution"] = log_label
        return out

    # ------------------------------------------------------------------ readouts
    @staticmethod
    def branch_logits(output: dict[str, torch.Tensor], branch: str) -> torch.Tensor:
        """Readout decompositions of one forward: ``mixture`` (learned), ``semantic``,
        ``support``, or ``fixed_half``. These reuse the contextualised states and are not
        absence-of-support interventions."""
        if branch == "mixture":
            return output["logits"]
        if branch == "semantic":
            return output["log_p_semantic"]
        if branch == "support":
            return output["log_p_support"]
        if branch == "fixed_half":
            half = math.log(0.5)
            valid = output["logits"] > NEG / 2
            mass = torch.logaddexp(half + output["log_p_support"].clamp_min(LOG_FILL),
                                   half + output["log_p_semantic"].clamp_min(LOG_FILL))
            mass = mass.masked_fill(~valid, LOG_FILL)
            return (mass - torch.logsumexp(mass, dim=1, keepdim=True)).masked_fill(~valid, NEG)
        raise ValueError(f"unknown readout branch {branch!r}")
