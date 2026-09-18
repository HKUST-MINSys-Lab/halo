"""Identity-initialised support-conditioned residual classifier.

The shared nearest-neighbour controls remain in ``training.support_classifier.neighbors``.
This module is HALO-only: it centres an episode locally, takes its differentiable-neighbour vote
as the floor, and learns scalar corrections without rewriting the encoder's geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import AttentionSpec, ScaledSum, SetAttentionStack
from model.support.token_mixer import (
    N_ROLES, ROLE_CANDIDATE, ROLE_QUERY, ROLE_SUPPORT, ROLE_SUPPORT_LABEL,
)
from training.support_classifier.neighbors import differentiable_neighbor_logits


@dataclass(frozen=True)
class ResidualClassifierConfig:
    text_dim: int = 384
    n_layers: int = 2
    max_candidates: int = 256
    max_supports: int = 8192
    temperature: float = 0.07
    text_temperature: float = 0.07
    centring: str = "support_mean"
    lambda_buckets: tuple[int, ...] = (0, 1, 2, 4, 8)
    residual_enabled: bool = True
    text_term_enabled: bool = True
    shared_trunk: bool = True
    # Two complete, independently parameterised heads routed by episode regime
    # (zero-support vs enrolled), as in the retired SupportTokenMixer. Nothing is
    # shared between them. `shared_trunk` splits metric-vs-text WITHIN one head and is
    # a different axis; the two flags compose.
    regime_split: bool = False
    normalized_token_composition: bool = True
    adaptive_text_gate: bool = False
    text_gate_hidden: int = 16

    def __post_init__(self) -> None:
        if self.text_dim < 1 or self.n_layers < 0 or self.max_candidates < 2 or self.max_supports < 0:
            raise ValueError("invalid residual-classifier capacity")
        if self.temperature <= 0 or self.text_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.centring not in {"none", "support_mean", "corpus_mean"}:
            raise ValueError(
                "centring must be none, support_mean, or corpus_mean; CSLS is an "
                "evaluation-time decision rule and is not implemented by this head"
            )
        if not self.lambda_buckets or self.lambda_buckets[0] != 0 or tuple(sorted(set(self.lambda_buckets))) != self.lambda_buckets:
            raise ValueError("lambda_buckets must be unique, sorted, and start with zero")
        if self.text_gate_hidden < 1:
            raise ValueError("text_gate_hidden must be positive")


class ResidualSupportClassifier(nn.Module):
    """One k-agnostic set-attention scorer with its differentiable support vote as the floor."""

    def __init__(self, spec: AttentionSpec, cfg: ResidualClassifierConfig | None = None):
        super().__init__()
        self.spec, self.cfg = spec, cfg or ResidualClassifierConfig()
        d = spec.d_model
        self.signal_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        # Text remains in the frozen SBERT coordinate system. This fixed projection only gives
        # attention a d-dimensional content token; it cannot learn a label lookup or move text.
        projection = torch.empty(self.cfg.text_dim, d)
        nn.init.orthogonal_(projection)
        self.register_buffer("text_to_model", projection, persistent=True)
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.candidate_tag = nn.Embedding(self.cfg.max_candidates + 1, d)
        self.support_pair_tag = nn.Embedding(self.cfg.max_supports + 1, d)
        # Tags identify roles and bindings, but their raw sqrt(d) magnitude must not drown the
        # encoder/text content. Tags are redrawn per episode, so begin them conservatively.
        # The old v2 checkpoint format deliberately retains its original raw sum for reproducible
        # historical evaluation; all new v3 heads use normalized composition.
        if self.cfg.normalized_token_composition:
            self.query_token_sum = ScaledSum(2, init=[1.0, 0.15])
            self.support_token_sum = ScaledSum(4, init=[1.0, 0.15, 0.10, 0.10])
            self.label_token_sum = ScaledSum(4, init=[1.0, 0.15, 0.10, 0.10])
            self.candidate_token_sum = ScaledSum(3, init=[1.0, 0.15, 0.10])
        self.metric_stack = SetAttentionStack(spec, self.cfg.n_layers)
        self.text_stack = self.metric_stack if self.cfg.shared_trunk else SetAttentionStack(spec, self.cfg.n_layers)
        self.r_support_head = nn.Linear(d, 1)
        self.r_candidate_head = nn.Linear(d, 1)
        nn.init.zeros_(self.r_support_head.weight); nn.init.zeros_(self.r_support_head.bias)
        nn.init.zeros_(self.r_candidate_head.weight); nn.init.zeros_(self.r_candidate_head.bias)
        self.p_text = nn.Linear(d, self.cfg.text_dim)
        self.lambda_table = nn.Parameter(torch.zeros(len(self.cfg.lambda_buckets)))
        if self.cfg.adaptive_text_gate:
            # Positive base weights under the softplus parameterization. Enrolled candidates begin
            # arbitrarily close to the historical zero text weight without creating a dead path.
            desired = torch.full_like(self.lambda_table, 1e-3)
            desired[0] = 1.0
            with torch.no_grad():
                self.lambda_table.copy_(torch.log(torch.expm1(desired)))
            self.text_gate_norm = nn.LayerNorm(6)
            self.text_gate = nn.Sequential(
                nn.Linear(6, self.cfg.text_gate_hidden), nn.SiLU(),
                nn.Linear(self.cfg.text_gate_hidden, 1),
            )
            # A tiny nonzero output weight lets every gate layer receive gradient on step one while
            # keeping initialization within 1e-3 of the support-count prior.
            nn.init.normal_(self.text_gate[-1].weight, std=1e-3)
            nn.init.zeros_(self.text_gate[-1].bias)
        else:
            with torch.no_grad():
                self.lambda_table[0] = 1.0
        # Keep persistent-buffer shape invariant so strict checkpoint restoration works. NaNs mark
        # an unfitted mean and make accidental corpus centring fail loudly before any scoring.
        self.register_buffer("corpus_mean", torch.full((d,), float("nan")), persistent=True)

    def set_corpus_mean(self, value: torch.Tensor) -> None:
        if value.shape != (self.spec.d_model,):
            raise ValueError("corpus mean must match d_model")
        self.corpus_mean.copy_(value.detach().to(device=self.corpus_mean.device, dtype=torch.float32))

    def _check(self, query: torch.Tensor, support: torch.Tensor, text: torch.Tensor,
               bound: torch.Tensor, support_mask: torch.Tensor, candidate_mask: torch.Tensor) -> None:
        b, c = candidate_mask.shape
        if query.shape != (b, self.spec.d_model) or support.shape[:2] != bound.shape or support.shape[:2] != support_mask.shape:
            raise ValueError("query/support shapes do not agree")
        if support.shape[-1] != self.spec.d_model or text.shape != (b, c, self.cfg.text_dim):
            raise ValueError("feature/text dimensions do not agree")
        if c > self.cfg.max_candidates or support.shape[1] > self.cfg.max_supports or not bool(candidate_mask.any(1).all()):
            raise ValueError("episode exceeds residual-classifier capacity")
        if bound.numel() and bool(((bound < -1) | (bound >= c)).any()):
            raise ValueError("support bound outside candidate range")

    def _compose(self, content: torch.Tensor, role: int, *, candidate_tag: torch.Tensor | None = None,
                 pair_tag: torch.Tensor | None = None) -> torch.Tensor:
        roles = self.role_emb(torch.full(content.shape[:-1], role, dtype=torch.long, device=content.device))
        if not self.cfg.normalized_token_composition:
            out = content + roles
            if candidate_tag is not None:
                out = out + self.candidate_tag(candidate_tag)
            if pair_tag is not None:
                out = out + self.support_pair_tag(pair_tag)
            return out
        if candidate_tag is None and pair_tag is None:
            return self.query_token_sum(content, roles)
        if pair_tag is None:
            return self.candidate_token_sum(content, roles, self.candidate_tag(candidate_tag))
        combine = self.support_token_sum if role == ROLE_SUPPORT else self.label_token_sum
        return combine(content, roles, self.candidate_tag(candidate_tag),
                       self.support_pair_tag(pair_tag))

    def _centre(self, query: torch.Tensor, support: torch.Tensor, mask: torch.Tensor,
                corpus_mean: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.centring == "none":
            return query.float(), support.float()
        if self.cfg.centring == "corpus_mean":
            mean = corpus_mean if corpus_mean is not None else self.corpus_mean
            if mean.shape != (self.spec.d_model,) or not bool(torch.isfinite(mean).all()):
                raise ValueError("corpus_mean centring requires a fitted corpus mean")
            mean = mean.to(query).view(1, -1)
        elif self.cfg.centring == "support_mean":
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(support.dtype)
            mean = (support * mask.unsqueeze(-1).to(support.dtype)).sum(dim=1) / denom
        # The shared differentiable-neighbor helper owns cosine normalization. Returning only the
        # centred vectors avoids a second normalization and preserves its output bit-for-bit.
        return query.float() - mean.float(), support.float() - mean[:, None].float()

    def _k_per_candidate(self, bound: torch.Tensor, mask: torch.Tensor, c: int) -> torch.Tensor:
        counts = torch.zeros((bound.shape[0], c), dtype=torch.float32, device=bound.device)
        if bound.shape[1]:
            counts.scatter_add_(1, bound.clamp_min(0), mask.float())
        return counts

    def _lambda(self, k_c: torch.Tensor) -> torch.Tensor:
        bucket = torch.zeros_like(k_c, dtype=torch.long)
        for index, lower in enumerate(self.cfg.lambda_buckets[1:], start=1):
            bucket = torch.where(k_c >= lower, torch.full_like(bucket, index), bucket)
        return self.lambda_table[bucket]

    def _adaptive_lambda(
        self,
        *,
        k_c: torch.Tensor,
        query: torch.Tensor,
        support: torch.Tensor,
        support_bound: torch.Tensor,
        support_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        metric_logits: torch.Tensor,
        text_cosine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Candidate-local semantic weight from observed sensor evidence, never label identity."""
        b, c = k_c.shape
        bucket = torch.zeros_like(k_c, dtype=torch.long)
        for index, lower in enumerate(self.cfg.lambda_buckets[1:], start=1):
            bucket = torch.where(k_c >= lower, torch.full_like(bucket, index), bucket)

        sums = k_c.new_zeros((b, c))
        squares = k_c.new_zeros((b, c))
        strongest = k_c.new_full((b, c), -1.0)
        if support.shape[1]:
            similarity = torch.einsum(
                "bd,bkd->bk", F.normalize(query, dim=-1), F.normalize(support, dim=-1),
            ).float()
            slots = support_bound.clamp_min(0)
            valid_similarity = similarity * support_mask.to(similarity.dtype)
            sums.scatter_add_(1, slots, valid_similarity)
            squares.scatter_add_(1, slots, valid_similarity.square())
            strongest.scatter_reduce_(
                1, slots,
                similarity.masked_fill(~support_mask, -1.0),
                reduce="amax", include_self=True,
            )
        denom = k_c.clamp_min(1.0)
        mean = sums / denom
        # k=1 has exactly zero variance. sqrt'(0) is infinite, so use a small FP32 floor rather
        # than letting ordinary one-shot enrollment create NaN gradients in the gate.
        spread = (squares / denom - mean.square()).clamp_min(0.0).add(1e-6).sqrt()
        strongest = torch.where(k_c.gt(0), strongest, torch.zeros_like(strongest))

        valid_metric = metric_logits.masked_fill(~candidate_mask, float("-inf"))
        top_values, top_slots = valid_metric.topk(k=min(2, c), dim=1)
        if c == 1:  # Defensive; config and sampler both require at least two candidates.
            competitor = top_values[:, :1].expand_as(metric_logits)
        else:
            candidate = torch.arange(c, device=metric_logits.device).unsqueeze(0)
            competitor = torch.where(
                top_slots[:, :1].eq(candidate), top_values[:, 1:2], top_values[:, :1],
            )
        margin = torch.where(
            candidate_mask, torch.tanh(metric_logits - competitor), torch.zeros_like(metric_logits),
        )
        sensor_score = 2.0 * torch.softmax(valid_metric.float(), dim=1).to(text_cosine) - 1.0
        disagreement = torch.where(
            candidate_mask, text_cosine - sensor_score, torch.zeros_like(text_cosine),
        )
        features = torch.stack((
            torch.log1p(k_c), strongest, mean, spread, margin, disagreement,
        ), dim=-1)
        features = features.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        residual = self.text_gate(self.text_gate_norm(features)).squeeze(-1)
        value = F.softplus(self.lambda_table[bucket] + residual)
        return value.masked_fill(~candidate_mask, 0.0), features

    def forward(self, *, query_feature: torch.Tensor, support_feature: torch.Tensor,
                support_label_text: torch.Tensor, support_bound: torch.Tensor,
                support_mask: torch.Tensor, support_pair_slot: torch.Tensor,
                candidate_text: torch.Tensor, candidate_mask: torch.Tensor,
                candidate_slot: torch.Tensor, corpus_mean: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        self._check(query_feature, support_feature, candidate_text, support_bound, support_mask, candidate_mask)
        b, c = candidate_mask.shape
        k = support_feature.shape[1]
        q, s = self._centre(query_feature, support_feature, support_mask, corpus_mean)
        k_c = self._k_per_candidate(support_bound, support_mask, c)
        base_logits = q.new_zeros((b, c)); base_weight = q.new_zeros((b, k))
        has_support = support_mask.any(dim=1)
        if k and bool(has_support.any()):
            selected_logits, selected_weight = differentiable_neighbor_logits(
                q[has_support], s[has_support], support_bound[has_support], support_mask[has_support],
                candidate_mask[has_support], temperature=self.cfg.temperature,
            )
            base_logits[has_support] = selected_logits
            base_weight[has_support] = selected_weight
        r_support = q.new_zeros((b, k)); r_candidate = q.new_zeros((b, c))
        if self.cfg.residual_enabled:
            q_token = self._compose(self.signal_proj(query_feature).unsqueeze(1), ROLE_QUERY)
            s_token = self._compose(self.signal_proj(support_feature), ROLE_SUPPORT,
                                    candidate_tag=torch.where(support_mask, candidate_slot.gather(1, support_bound.clamp_min(0)), torch.zeros_like(support_bound)),
                                    pair_tag=support_pair_slot)
            sl_token = self._compose(support_label_text @ self.text_to_model, ROLE_SUPPORT_LABEL,
                                     candidate_tag=torch.where(support_mask, candidate_slot.gather(1, support_bound.clamp_min(0)), torch.zeros_like(support_bound)),
                                     pair_tag=support_pair_slot)
            c_token = self._compose(candidate_text @ self.text_to_model, ROLE_CANDIDATE, candidate_tag=candidate_slot)
            tokens = torch.cat((q_token, s_token, sl_token, c_token), dim=1)
            valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=q.device), support_mask, support_mask, candidate_mask), dim=1)
            metric_hidden = self.metric_stack(tokens, key_padding_mask=valid)
            text_hidden = self.text_stack(tokens, key_padding_mask=valid) if self.text_stack is not self.metric_stack else metric_hidden
            if k:
                r_support = self.r_support_head(metric_hidden[:, 1:1 + k]).squeeze(-1) * support_mask
            r_candidate = self.r_candidate_head(text_hidden[:, 1 + 2 * k:]).squeeze(-1) * candidate_mask
        if not self.cfg.residual_enabled and not self.cfg.text_term_enabled and bool((~has_support).any()):
            raise ValueError("supportless candidates require the text term or residual scorer")
        if k and self.cfg.residual_enabled:
            q_unit = F.normalize(q, dim=-1)
            s_unit = F.normalize(s, dim=-1)

            def corrected_vote(correction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                scores = torch.einsum("bd,bkd->bk", q_unit, s_unit) / self.cfg.temperature + correction
                scores = scores.masked_fill(~support_mask, float("-inf"))
                corrected_weight = torch.zeros(scores.shape, dtype=torch.float32, device=scores.device)
                if bool(has_support.any()):
                    corrected_weight[has_support] = torch.softmax(
                        scores[has_support].float(), dim=1,
                    ).masked_fill(~support_mask[has_support], 0.0)
                vote = torch.zeros((b, c), dtype=corrected_weight.dtype, device=corrected_weight.device)
                vote.scatter_add_(1, support_bound.clamp_min(0), corrected_weight)
                corrected_logits = vote.clamp_min(1e-12).log()
                return corrected_logits, corrected_weight

            corrected_logits, weight = corrected_vote(r_support)
            zero_logits, _ = corrected_vote(torch.zeros_like(r_support))
            # The shared helper is the exact floor. Subtracting this branch's own zero-correction
            # value makes the learned reranking a genuine residual while retaining gradients.
            metric_logits = base_logits + (corrected_logits - zero_logits)
            metric_logits = torch.where(has_support[:, None], metric_logits, base_logits)
        else:
            weight, metric_logits = base_weight, base_logits
        # P_text is fitted in closed form on raw pooled encoder features. Keep that exact input at
        # initialization; the attention trunk contributes only through the scalar residual heads.
        text_cosine = torch.einsum(
            "bd,bcd->bc", F.normalize(self.p_text(query_feature).float(), dim=-1),
            F.normalize(candidate_text.float(), dim=-1),
        )
        text_score = text_cosine / self.cfg.text_temperature
        uniform_log_prior = -candidate_mask.sum(dim=1, keepdim=True).to(metric_logits.dtype).log()
        gate_features = None
        if self.cfg.text_term_enabled and self.cfg.adaptive_text_gate:
            # A candidate with no sensor evidence is represented by the same uninformative prior
            # used by final scoring. Giving the gate a raw zero logit here would make missing
            # candidates appear artificially stronger whenever enrolled candidates are negative.
            gate_metric = torch.where(k_c.eq(0), uniform_log_prior, metric_logits)
            lam, gate_features = self._adaptive_lambda(
                k_c=k_c, query=q, support=s, support_bound=support_bound,
                support_mask=support_mask, candidate_mask=candidate_mask,
                metric_logits=gate_metric, text_cosine=text_cosine,
            )
        else:
            lam = self._lambda(k_c) if self.cfg.text_term_enabled else torch.zeros_like(k_c)
        # A missing candidate has no neighbour probability, not a logit of zero.  Its reference
        # is the uniform prior of an uninformative complete vote.  Thus text competes with a
        # defined prior rather than receiving an arbitrary numerical advantage.
        base_part = torch.where(k_c.eq(0), uniform_log_prior, base_logits)
        metric_part = torch.where(k_c.eq(0), uniform_log_prior, metric_logits)
        text_part = lam * text_score
        logits = metric_part + text_part + r_candidate
        return {"logits": logits.masked_fill(~candidate_mask, -1e30), "support_weight": weight,
                "k_c": k_c, "r_support": r_support, "r_candidate": r_candidate,
                "text_score": text_score, "lambda": lam,
                "neighbor_logits": base_logits, "base_part": base_part,
                "metric_part": metric_part,
                "text_part": text_part, "text_gate_features": gate_features}

    def telemetry(self) -> dict[str, float]:
        values = F.softplus(self.lambda_table.detach()) if self.cfg.adaptive_text_gate \
            else self.lambda_table.detach()
        return {f"classifier/lambda_{bucket}": float(values[index])
                for index, bucket in enumerate(self.cfg.lambda_buckets)}


class RegimeSplitSupportClassifier(nn.Module):
    """Two complete residual heads, routed by whether an episode has any support at all.

    This is the no-sharing control for the unified scorer: the zero-support head and the enrolled
    head share no parameter, not even the frozen text projection buffer. An episode reaches the
    enrolled head if any candidate retains support, so a partially masked episode trains the
    enrolled head's text term rather than the zero-support head.
    """

    def __init__(self, spec: AttentionSpec, cfg: ResidualClassifierConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or ResidualClassifierConfig()
        member = replace(self.cfg, regime_split=False)
        self.zero_head = ResidualSupportClassifier(spec, member)
        self.few_head = ResidualSupportClassifier(spec, member)

    def set_corpus_mean(self, value: torch.Tensor) -> None:
        self.zero_head.set_corpus_mean(value)
        self.few_head.set_corpus_mean(value)

    @property
    def p_text(self) -> nn.Linear:
        """Least-squares initialisation writes here; mirror it into both heads afterwards."""
        return self.few_head.p_text

    def sync_text_projection(self) -> None:
        self.zero_head.p_text.load_state_dict(self.few_head.p_text.state_dict())

    def forward(self, *, query_feature: torch.Tensor, support_feature: torch.Tensor,
                support_mask: torch.Tensor, candidate_mask: torch.Tensor,
                **kwargs) -> dict[str, torch.Tensor]:
        enrolled = support_mask.any(dim=1) if support_mask.numel() else \
            torch.zeros(len(query_feature), dtype=torch.bool, device=query_feature.device)
        b, c = candidate_mask.shape
        k = support_feature.shape[1]
        out = {"logits": query_feature.new_zeros((b, c), dtype=torch.float32),
               "support_weight": query_feature.new_zeros((b, k), dtype=torch.float32),
               "k_c": query_feature.new_zeros((b, c)), "r_support": query_feature.new_zeros((b, k)),
               "r_candidate": query_feature.new_zeros((b, c)),
               "text_score": query_feature.new_zeros((b, c)),
               "lambda": query_feature.new_zeros((b, c)),
               "neighbor_logits": query_feature.new_zeros((b, c)),
               "base_part": query_feature.new_zeros((b, c)),
               "metric_part": query_feature.new_zeros((b, c)),
               "text_part": query_feature.new_zeros((b, c)),
               "text_gate_features": (query_feature.new_zeros((b, c, 6))
                                      if self.cfg.text_term_enabled and self.cfg.adaptive_text_gate
                                      else None)}
        for head, rows in ((self.few_head, torch.nonzero(enrolled).flatten()),
                           (self.zero_head, torch.nonzero(~enrolled).flatten())):
            if not len(rows):
                continue
            part = head(
                query_feature=query_feature.index_select(0, rows),
                support_feature=support_feature.index_select(0, rows),
                support_mask=support_mask.index_select(0, rows),
                candidate_mask=candidate_mask.index_select(0, rows),
                **{name: value.index_select(0, rows) if torch.is_tensor(value) and value.shape[:1] == (b,)
                   else value for name, value in kwargs.items()},
            )
            for name, value in out.items():
                if value is not None:
                    value.index_copy_(0, rows, part[name].to(value.dtype))
        return out

    def telemetry(self) -> dict[str, float]:
        return {**{f"{key}_zero": value for key, value in self.zero_head.telemetry().items()},
                **{f"{key}_few": value for key, value in self.few_head.telemetry().items()}}


def build_support_classifier(spec: AttentionSpec, cfg: ResidualClassifierConfig):
    """One construction path for the trainer and the evaluator."""
    return RegimeSplitSupportClassifier(spec, cfg) if cfg.regime_split \
        else ResidualSupportClassifier(spec, cfg)
