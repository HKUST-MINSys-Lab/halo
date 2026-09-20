"""ACTIVE EXPERIMENT: evidence-gated support classifier (``support_classifier_v4``).

Design of record: ``docs/journal/2026-09-20-classifier-v4-evidence-gated-design.md``.

This head keeps the promoted v3 lineage's discipline — a closed-form support vote that is always
in the output path, perturbed by bounded learned scalars — and removes the two unbounded text side
channels v3 carries (``r_support`` and ``r_candidate``, both produced by an attention stack that
reads label tokens).  Label text reaches the output at exactly one place, weighted by
``lambda_c <= lambda_max``.

Order of computation, deliberately the reverse of the abandoned contextual lineage:

1. closed-form paths: the centred differentiable-neighbour vote, and the text log-softmax;
2. statistics derived from those paths (similarity structure and vote structure only);
3. two small MLPs on those statistics: per-support trust ``t_j`` and per-candidate ``lambda_c``;
4. the blend ``(1 - lambda_c) * metric_c + lambda_c * text_c``.

Both gates are **label-blind**: no feature is derived from candidate text, support label text,
text similarity, text confidence, candidate slot identity or raw encoder embeddings.  The collapse
policy that sank ``support_contextual_mixture_v1``, ``support_contextual_residual_v1`` and
``support_evidence_aware_v2`` ("this is a training-vocabulary label, so trust its text") is
therefore inexpressible here, and ``lambda_max`` keeps the support path from ever being deleted.

Context between supports and between candidates enters only through fixed permutation-invariant
pooling inside the feature vectors.  There is no learned mixing between rows: each MLP sees one
support (or one candidate) at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import AttentionSpec
from model.support.primitive_semantics import (
    PRIMITIVE_VOCABULARY_VERSION, PrimitiveSemanticConfig, PrimitiveSemanticHead,
)
from training.support_classifier.neighbors import differentiable_neighbor_logits

ARCHITECTURE_VERSION = "support_classifier_v4"

# Feature widths are part of the checkpoint contract; changing either requires a new version.
N_TRUST_FEATURES = 11
N_GATE_FEATURES = 13

# Every statistic is bounded and computed in fp32, but a padded row can still carry an arbitrary
# encoder value, so masked entries are *replaced* rather than merely multiplied by zero.
_STD_FLOOR = 1e-6

# Prior blend weight per support-count bucket, before the learned candidate-local correction.
# This re-expresses v3's learned ordering (more evidence -> less text: 1.80/1.33/1.19/1.09/0.98
# over buckets 0/1/2/4/8) on the bounded [0, lambda_max] scale.  It is not a numerical port: v3
# scales a raw cosine/tau score, this head blends two log-probabilities, so only the ordering and
# the direction carry over.  The bucket-0 entry is inert because k=0 forces lambda to 1.
DEFAULT_LAMBDA_PRIOR = (0.99, 0.55, 0.45, 0.38, 0.30)


@dataclass(frozen=True)
class EvidenceGatedClassifierConfig:
    text_dim: int = 384
    max_candidates: int = 256
    max_supports: int = 8192
    temperature: float = 0.07
    text_temperature: float = 0.07
    centring: str = "support_mean"
    lambda_buckets: tuple[int, ...] = (0, 1, 2, 4, 8)
    # Declared a priori.  An enrolled candidate always keeps at least (1 - lambda_max) of its
    # support evidence, so no setting of the gate can delete the support path.
    lambda_max: float = 0.85
    # Bounds |t_j|, and therefore the support reweighting, to exp(+-trust_scale).
    trust_scale: float = 2.0
    trust_hidden: int = 32
    gate_hidden: int = 32
    trust_enabled: bool = True
    text_term_enabled: bool = True
    # Which semantic path feeds the blend. "text" is the promoted cosine path and the default;
    # "primitives" is the compositional path; "text+primitives" combines the two by a FIXED
    # equal-weight sum in log space, never a learned router.
    semantic_mode: str = "text"
    primitive_version: str = PRIMITIVE_VOCABULARY_VERSION
    primitive_combiner: str = "projection"
    primitive_projection_rank: int = 384
    primitive_profile_temperature: float = 0.05
    primitive_logit_scale: float = 10.0

    def __post_init__(self) -> None:
        if self.semantic_mode not in SEMANTIC_MODES:
            raise ValueError(f"semantic_mode must be one of {SEMANTIC_MODES}")
        if self.text_dim < 1 or self.max_candidates < 2 or self.max_supports < 0:
            raise ValueError("invalid evidence-gated capacity")
        if self.temperature <= 0 or self.text_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.centring not in {"none", "support_mean", "corpus_mean"}:
            raise ValueError("centring must be none, support_mean, or corpus_mean")
        if (not self.lambda_buckets or self.lambda_buckets[0] != 0
                or tuple(sorted(set(self.lambda_buckets))) != self.lambda_buckets):
            raise ValueError("lambda_buckets must be unique, sorted, and start with zero")
        if not 0.0 < self.lambda_max < 1.0:
            raise ValueError("lambda_max must be in (0, 1) so the support path is never deleted")
        if self.trust_scale <= 0:
            raise ValueError("trust_scale must be positive")
        if self.trust_hidden < 1 or self.gate_hidden < 1:
            raise ValueError("gate hidden widths must be positive")


SEMANTIC_MODES = ("text", "primitives", "text+primitives")


def _logit(value: float) -> float:
    return float(torch.logit(torch.tensor(value, dtype=torch.float64)))


class EvidenceGatedSupportClassifier(nn.Module):
    """Closed-form paths, then two bounded label-blind scalars, then one blend."""

    def __init__(self, spec: AttentionSpec, cfg: EvidenceGatedClassifierConfig | None = None):
        super().__init__()
        self.spec, self.cfg = spec, cfg or EvidenceGatedClassifierConfig()
        d = spec.d_model
        # The only learned map into text space, fitted in closed form before training like v3's.
        self.p_text = nn.Linear(d, self.cfg.text_dim)
        self.trust_norm = nn.LayerNorm(N_TRUST_FEATURES)
        self.trust_mlp = nn.Sequential(
            nn.Linear(N_TRUST_FEATURES, self.cfg.trust_hidden), nn.GELU(),
            nn.Linear(self.cfg.trust_hidden, 1),
        )
        self.gate_norm = nn.LayerNorm(N_GATE_FEATURES)
        self.gate_mlp = nn.Sequential(
            nn.Linear(N_GATE_FEATURES, self.cfg.gate_hidden), nn.GELU(),
            nn.Linear(self.cfg.gate_hidden, 1),
        )
        # Near-zero, NOT exactly zero, output layers. Two reasons, and the second is decisive:
        #  * dL/dW_hidden is proportional to W_out, so a zero output weight makes every hidden
        #    layer's gradient exactly zero (this is how v2 shipped 34 dead parameter tensors);
        #  * the trust head's output BIAS is shift-invariant — a constant added to every support's
        #    score cancels in the softmax over supports — so with a zero output weight the trust
        #    MLP would receive no gradient at all, ever, not merely at step 0.
        # A 1e-3 draw keeps step 0 within a few 1e-3 of the closed-form vote while every gate
        # parameter is trainable from step one. v3's text gate uses the same remedy.
        for mlp in (self.trust_mlp, self.gate_mlp):
            nn.init.normal_(mlp[-1].weight, std=1e-3)
            nn.init.zeros_(mlp[-1].bias)
        buckets = len(self.cfg.lambda_buckets)
        prior = (DEFAULT_LAMBDA_PRIOR if buckets == len(DEFAULT_LAMBDA_PRIOR)
                 else (0.99,) + (0.5,) * (buckets - 1))
        self.lambda_prior = nn.Parameter(torch.tensor(
            [_logit(min(value / self.cfg.lambda_max, 1.0 - 1e-4)) for value in prior],
            dtype=torch.float32,
        ))
        self.primitive_head = None
        if self.cfg.semantic_mode != "text":
            self.primitive_head = PrimitiveSemanticHead(d, PrimitiveSemanticConfig(
                version=self.cfg.primitive_version, text_dim=self.cfg.text_dim,
                combiner=self.cfg.primitive_combiner,
                projection_rank=self.cfg.primitive_projection_rank,
                profile_temperature=self.cfg.primitive_profile_temperature,
                logit_scale=self.cfg.primitive_logit_scale,
            ))
        self.register_buffer("corpus_mean", torch.full((d,), float("nan")), persistent=True)

    # ------------------------------------------------------------------ helpers
    def set_corpus_mean(self, value: torch.Tensor) -> None:
        if value.shape != (self.spec.d_model,):
            raise ValueError("corpus mean must match d_model")
        self.corpus_mean.copy_(
            value.detach().to(device=self.corpus_mean.device, dtype=torch.float32),
        )

    def _check(self, query: torch.Tensor, support: torch.Tensor, text: torch.Tensor,
               bound: torch.Tensor, support_mask: torch.Tensor,
               candidate_mask: torch.Tensor) -> None:
        b, c = candidate_mask.shape
        if (query.shape != (b, self.spec.d_model) or support.shape[:2] != bound.shape
                or support.shape[:2] != support_mask.shape):
            raise ValueError("query/support shapes do not agree")
        if support.shape[-1] != self.spec.d_model or text.shape != (b, c, self.cfg.text_dim):
            raise ValueError("feature/text dimensions do not agree")
        if (c > self.cfg.max_candidates or support.shape[1] > self.cfg.max_supports
                or not bool(candidate_mask.any(1).all())):
            raise ValueError("episode exceeds evidence-gated capacity")
        if bound.numel() and bool(((bound < -1) | (bound >= c)).any()):
            raise ValueError("support bound outside candidate range")

    def _centre(self, query: torch.Tensor, support: torch.Tensor, mask: torch.Tensor,
                corpus_mean: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Identical contract to v3: return centred vectors, let the shared helper normalise."""
        if self.cfg.centring == "none":
            return query.float(), support.float()
        if self.cfg.centring == "corpus_mean":
            mean = corpus_mean if corpus_mean is not None else self.corpus_mean
            if mean.shape != (self.spec.d_model,) or not bool(torch.isfinite(mean).all()):
                raise ValueError("corpus_mean centring requires a fitted corpus mean")
            mean = mean.to(query).view(1, -1)
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(support.dtype)
            # Replace padded rows rather than multiplying them by zero: a padded slot may hold a
            # non-finite encoder value, and NaN * 0 is NaN, which would poison the whole episode.
            kept = torch.where(mask.unsqueeze(-1), support, torch.zeros_like(support))
            mean = kept.sum(dim=1) / denom
        return query.float() - mean.float(), support.float() - mean[:, None].float()

    def _k_per_candidate(self, bound: torch.Tensor, mask: torch.Tensor, c: int) -> torch.Tensor:
        counts = torch.zeros((bound.shape[0], c), dtype=torch.float32, device=bound.device)
        if bound.shape[1]:
            counts.scatter_add_(1, bound.clamp_min(0), mask.float())
        return counts

    def _bucket(self, k_c: torch.Tensor) -> torch.Tensor:
        bucket = torch.zeros_like(k_c, dtype=torch.long)
        for index, lower in enumerate(self.cfg.lambda_buckets[1:], start=1):
            bucket = torch.where(k_c >= lower, torch.full_like(bucket, index), bucket)
        return bucket

    # ------------------------------------------------------------------ statistics
    @staticmethod
    def _candidate_stats(
        similarity: torch.Tensor, slots: torch.Tensor, support_mask: torch.Tensor,
        k_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-candidate mean, std and max of the query-support similarity, masked-safe."""
        valid = similarity * support_mask.to(similarity.dtype)
        sums = torch.zeros_like(k_c).scatter_add_(1, slots, valid)
        squares = torch.zeros_like(k_c).scatter_add_(1, slots, valid.square())
        strongest = torch.full_like(k_c, -1.0).scatter_reduce_(
            1, slots, similarity.masked_fill(~support_mask, -1.0),
            reduce="amax", include_self=True,
        )
        denom = k_c.clamp_min(1.0)
        mean = sums / denom
        # k=1 has exactly zero variance; sqrt'(0) is infinite, so floor before the square root.
        std = (squares / denom - mean.square()).clamp_min(0.0).add(_STD_FLOOR).sqrt()
        empty = k_c.eq(0)
        zero = torch.zeros_like(mean)
        return (torch.where(empty, zero, mean), torch.where(empty, zero, std),
                torch.where(empty, zero, strongest))

    @staticmethod
    def _normalised_rank(similarity: torch.Tensor, support_mask: torch.Tensor,
                         n_support: torch.Tensor) -> torch.Tensor:
        """Tie-aware fraction of valid supports ranked above each support.

        The earlier stable-sort ordinal rank made equal similarities depend on support-row order.
        This formulation gives tied rows the same rank and is invariant to support permutation.
        The support axis is bounded by the episode contract, so the explicit pairwise comparison is
        both clearer and negligible relative to encoder execution.
        """
        above = similarity.unsqueeze(1) > similarity.unsqueeze(2)
        valid_other = support_mask.unsqueeze(2)
        rank = (above & valid_other).sum(dim=2).to(similarity.dtype)
        return torch.where(
            support_mask, rank / n_support.clamp_min(1.0), torch.zeros_like(rank),
        )

    def _trust_features(
        self, *, similarity: torch.Tensor, slots: torch.Tensor, support_mask: torch.Tensor,
        k_c: torch.Tensor, mean_c: torch.Tensor, max_c: torch.Tensor,
        episode_mean: torch.Tensor, episode_std: torch.Tensor, episode_max: torch.Tensor,
        n_support: torch.Tensor,
    ) -> torch.Tensor:
        """One label-blind feature vector per (query, support) pair."""
        own_mean = mean_c.gather(1, slots)
        own_max = max_c.gather(1, slots)
        own_k = k_c.gather(1, slots)
        # Best support of any OTHER candidate, computed exactly by masking this support's own
        # candidate out of the per-candidate maxima rather than approximating with the global max.
        c = k_c.shape[1]
        axis = torch.arange(c, device=slots.device).view(1, c, 1)
        other = max_c.unsqueeze(-1).masked_fill(
            axis.eq(slots.unsqueeze(1)) | k_c.eq(0).unsqueeze(-1), -1.0,
        ).amax(dim=1)
        z = ((similarity - episode_mean) / episode_std.clamp_min(_STD_FLOOR)).clamp(-8.0, 8.0)
        features = torch.stack((
            similarity,                                  # absolute closeness
            similarity - own_mean,                       # outlier or anchor within its candidate
            similarity - own_max,                        # distance from its candidate's best
            z,                                           # closeness at this episode's scale
            similarity - other,                          # discriminative against rival candidates
            self._normalised_rank(similarity, support_mask, n_support),
            torch.log1p(own_k),                          # sibling count
            own_k.eq(1.0).to(similarity.dtype),          # marks undefined within-candidate spread
            episode_mean.expand_as(similarity),          # episode context
            episode_max.expand_as(similarity),
            episode_std.expand_as(similarity),
        ), dim=-1)
        return torch.where(support_mask.unsqueeze(-1), features, torch.zeros_like(features))

    def _gate_features(
        self, *, vote_logits: torch.Tensor, k_c: torch.Tensor, candidate_mask: torch.Tensor,
        mean_c: torch.Tensor, std_c: torch.Tensor, max_c: torch.Tensor,
        mean_trust: torch.Tensor, episode_mean: torch.Tensor, episode_max: torch.Tensor,
    ) -> torch.Tensor:
        """One label-blind feature vector per candidate, read AFTER the trust reweighting."""
        c = k_c.shape[1]
        valid = vote_logits.masked_fill(~candidate_mask, float("-inf"))
        top_values, top_slots = valid.topk(k=min(2, c), dim=1)
        if c == 1:  # defensive; the sampler requires at least two candidates
            competitor = top_values[:, :1].expand_as(vote_logits)
        else:
            axis = torch.arange(c, device=vote_logits.device).unsqueeze(0)
            competitor = torch.where(
                top_slots[:, :1].eq(axis), top_values[:, 1:2], top_values[:, :1],
            )
        margin = torch.where(
            candidate_mask, torch.tanh(vote_logits - competitor), torch.zeros_like(vote_logits),
        )
        probability = torch.softmax(valid.float(), dim=1)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
        roster = candidate_mask.sum(dim=1, keepdim=True).to(vote_logits.dtype)
        coverage = (k_c.gt(0) & candidate_mask).sum(dim=1, keepdim=True).to(vote_logits.dtype) \
            / roster.clamp_min(1.0)
        features = torch.stack((
            vote_logits.clamp(-30.0, 0.0),               # this candidate's vote share (log)
            margin,                                      # how decisive the vote is
            entropy.expand_as(vote_logits),              # vote concentration over the roster
            torch.log1p(k_c),                            # evidence quantity
            k_c.eq(1.0).to(vote_logits.dtype),
            max_c, mean_c, std_c,                        # evidence quality and agreement
            mean_trust,                                  # the trust stage's summary
            episode_max.expand_as(vote_logits),          # is the query matched to anything at all
            episode_mean.expand_as(vote_logits),
            coverage.expand_as(vote_logits),             # enrollment structure of the roster
            roster.log().expand_as(vote_logits),         # makes margin/entropy comparable
        ), dim=-1)
        return torch.where(candidate_mask.unsqueeze(-1), features, torch.zeros_like(features))

    # ------------------------------------------------------------------ forward
    def forward(
        self, *, query_feature: torch.Tensor, support_feature: torch.Tensor,
        support_label_text: torch.Tensor, support_bound: torch.Tensor,
        support_mask: torch.Tensor, support_pair_slot: torch.Tensor,
        candidate_text: torch.Tensor, candidate_mask: torch.Tensor,
        candidate_slot: torch.Tensor, corpus_mean: torch.Tensor | None = None,
        lambda_override: float | None = None, trust_override: float | None = None,
        text_stop_gradient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """``support_label_text``, ``support_pair_slot`` and ``candidate_slot`` are accepted for
        one shared trainer/evaluator call signature and are deliberately unused: no label-derived
        or slot-derived quantity may reach either gate.

        ``lambda_override`` / ``trust_override`` force a constant for ablation readouts only.
        ``text_stop_gradient`` is a (B,) boolean marking rows whose semantic path must not send
        gradient into ``p_text`` or the encoder: the label-text corruption curriculum uses it so
        that a scrambled roster prices the reliability of the semantic path without also training
        the alignment to fit the scrambled labels.
        """
        with torch.autocast(device_type=query_feature.device.type, enabled=False):
            return self._forward(
                query_feature=query_feature.float(), support_feature=support_feature.float(),
                support_bound=support_bound, support_mask=support_mask,
                candidate_text=candidate_text.float(), candidate_mask=candidate_mask,
                corpus_mean=corpus_mean, lambda_override=lambda_override,
                trust_override=trust_override, text_stop_gradient=text_stop_gradient,
            )

    def _forward(
        self, *, query_feature: torch.Tensor, support_feature: torch.Tensor,
        support_bound: torch.Tensor, support_mask: torch.Tensor,
        candidate_text: torch.Tensor, candidate_mask: torch.Tensor,
        corpus_mean: torch.Tensor | None, lambda_override: float | None,
        trust_override: float | None, text_stop_gradient: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        self._check(query_feature, support_feature, candidate_text, support_bound,
                    support_mask, candidate_mask)
        b, c = candidate_mask.shape
        k = support_feature.shape[1]
        q, s = self._centre(query_feature, support_feature, support_mask, corpus_mean)
        k_c = self._k_per_candidate(support_bound, support_mask, c)
        has_support = support_mask.any(dim=1)
        slots = support_bound.clamp_min(0)

        # 1. closed-form support vote: the floor, always in the output path.
        base_logits = q.new_zeros((b, c))
        base_weight = q.new_zeros((b, k))
        if k and bool(has_support.any()):
            selected_logits, selected_weight = differentiable_neighbor_logits(
                q[has_support], s[has_support], support_bound[has_support],
                support_mask[has_support], candidate_mask[has_support],
                temperature=self.cfg.temperature,
            )
            base_logits[has_support] = selected_logits
            base_weight[has_support] = selected_weight

        # 2. similarity statistics, and 3a. the bounded per-support trust scalar.
        trust = q.new_zeros((b, k))
        trust_features = q.new_zeros((b, k, N_TRUST_FEATURES))
        mean_c = std_c = max_c = q.new_zeros((b, c))
        episode_mean = episode_max = q.new_zeros((b, 1))
        if k:
            similarity = torch.einsum(
                "bd,bkd->bk", F.normalize(q, dim=-1), F.normalize(s, dim=-1),
            ).float()
            # A padded slot may hold any encoder value, including a non-finite one; replace it.
            similarity = torch.where(support_mask, similarity, torch.zeros_like(similarity))
            n_support = support_mask.sum(dim=1, keepdim=True).to(similarity.dtype)
            denom = n_support.clamp_min(1.0)
            episode_mean = (similarity * support_mask).sum(dim=1, keepdim=True) / denom
            episode_square = (similarity.square() * support_mask).sum(dim=1, keepdim=True) / denom
            episode_std = (episode_square - episode_mean.square()).clamp_min(0.0) \
                .add(_STD_FLOOR).sqrt()
            episode_max = torch.where(
                has_support[:, None],
                similarity.masked_fill(~support_mask, -1.0).amax(dim=1, keepdim=True),
                torch.zeros_like(episode_mean),
            )
            mean_c, std_c, max_c = self._candidate_stats(similarity, slots, support_mask, k_c)
            if self.cfg.trust_enabled:
                trust_features = self._trust_features(
                    similarity=similarity, slots=slots, support_mask=support_mask, k_c=k_c,
                    mean_c=mean_c, max_c=max_c, episode_mean=episode_mean,
                    episode_std=episode_std, episode_max=episode_max, n_support=n_support,
                )
                raw = self.trust_mlp(self.trust_norm(trust_features)).squeeze(-1)
                trust = self.cfg.trust_scale * torch.tanh(raw)
                trust = torch.where(support_mask, trust, torch.zeros_like(trust))
            if trust_override is not None:
                trust = torch.where(
                    support_mask, torch.full_like(trust, float(trust_override)),
                    torch.zeros_like(trust),
                )

        # 3b. the reweighted vote, as an exact residual on the closed-form floor.  At trust == 0
        # the correction is identically zero, and the floor keeps its own gradient path.
        weight, metric_logits = base_weight, base_logits
        mean_trust = q.new_zeros((b, c))
        if k and bool(has_support.any()) and (self.cfg.trust_enabled or trust_override is not None):
            q_unit, s_unit = F.normalize(q, dim=-1), F.normalize(s, dim=-1)
            cosine = torch.einsum("bd,bkd->bk", q_unit, s_unit)

            def weighted_vote(correction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                scores = (cosine / self.cfg.temperature + correction).masked_fill(
                    ~support_mask, float("-inf"),
                )
                row_weight = torch.zeros(scores.shape, dtype=torch.float32, device=scores.device)
                row_weight[has_support] = torch.softmax(
                    scores[has_support].float(), dim=1,
                ).masked_fill(~support_mask[has_support], 0.0)
                vote = torch.zeros((b, c), dtype=row_weight.dtype, device=row_weight.device)
                vote.scatter_add_(1, slots, row_weight)
                return vote.clamp_min(1e-12).log(), row_weight

            trusted_logits, weight = weighted_vote(trust)
            zero_logits, _ = weighted_vote(torch.zeros_like(trust))
            metric_logits = torch.where(
                has_support[:, None], base_logits + (trusted_logits - zero_logits), base_logits,
            )
            mean_trust = (torch.zeros_like(k_c).scatter_add_(1, slots, trust * support_mask)
                          / k_c.clamp_min(1.0))

        # 4. semantic path as a log-probability over the roster, so lambda is a true blend.
        text_cosine = torch.einsum(
            "bd,bcd->bc", F.normalize(self.p_text(query_feature).float(), dim=-1),
            F.normalize(candidate_text.float(), dim=-1),
        )
        text_logits = torch.log_softmax(
            (text_cosine / self.cfg.text_temperature).masked_fill(~candidate_mask, float("-inf")),
            dim=-1,
        )
        primitive = None
        primitive_logits = None
        if self.primitive_head is not None:
            primitive = self.primitive_head(query_feature, candidate_text, candidate_mask)
            primitive_logits = primitive["logits"]
        if self.cfg.semantic_mode == "text":
            semantic_logits = text_logits
        elif self.cfg.semantic_mode == "primitives":
            semantic_logits = primitive_logits
        else:
            # Fixed equal weights in log space: the product of the two distributions, renormalised.
            # Deliberately not learned; a learned weight here is the router we removed.
            semantic_logits = torch.log_softmax(
                (text_logits + primitive_logits).masked_fill(~candidate_mask, float("-inf")),
                dim=-1,
            )
        if text_stop_gradient is not None:
            if text_stop_gradient.shape != (b,):
                raise ValueError("text_stop_gradient must be one boolean per episode")
            # Detach the whole semantic branch on corrupted rows: a scrambled roster must price
            # the reliability of meaning without training p_text, the primitive keys or the
            # encoder to fit the scramble. lambda still receives gradient, because d(loss)/d(lambda)
            # needs the branch's value, not its gradient.
            semantic_logits = torch.where(
                text_stop_gradient.view(b, 1), semantic_logits.detach(), semantic_logits,
            )
        # A candidate with no support has no vote probability; its reference is the uniform prior
        # of an uninformative complete vote, exactly as in v3.
        uniform_log_prior = -candidate_mask.sum(dim=1, keepdim=True).to(metric_logits.dtype).log()
        metric_part = torch.where(k_c.eq(0), uniform_log_prior, metric_logits)

        # 5. the per-candidate blend weight, read from the reweighted vote.
        gate_features = self._gate_features(
            vote_logits=metric_part, k_c=k_c, candidate_mask=candidate_mask,
            mean_c=mean_c, std_c=std_c, max_c=max_c, mean_trust=mean_trust,
            episode_mean=episode_mean, episode_max=episode_max,
        )
        raw_gate = self.gate_mlp(self.gate_norm(gate_features)).squeeze(-1)
        lam = self.cfg.lambda_max * torch.sigmoid(self.lambda_prior[self._bucket(k_c)] + raw_gate)
        if not self.cfg.text_term_enabled:
            if bool((~has_support).any()):
                raise ValueError("supportless candidates require the text term")
            lam = torch.zeros_like(lam)
        if lambda_override is not None:
            lam = torch.full_like(lam, float(lambda_override))
        # A candidate with no support of its own can only be named by meaning.  This is forced,
        # not learned, and it is the only place lambda may reach 1.
        lam = torch.where(k_c.eq(0), torch.ones_like(lam), lam).masked_fill(~candidate_mask, 0.0)

        logits = (1.0 - lam) * metric_part + lam * semantic_logits
        return {
            "logits": logits.masked_fill(~candidate_mask, -1e30),
            "support_weight": weight, "k_c": k_c,
            "trust": trust, "lambda": lam, "mean_trust": mean_trust,
            "neighbor_logits": base_logits, "metric_logits": metric_logits,
            "metric_part": metric_part, "text_logits": text_logits,
            "semantic_logits": semantic_logits,
            **({} if primitive is None else {
                "primitive_logits": primitive_logits,
                "primitive_profile": primitive["primitive_profile"],
                "candidate_primitive_profile": primitive["candidate_primitive_profile"],
                "primitive_axis_entropy": primitive["axis_entropy"],
            }),
            "text_score": text_cosine / self.cfg.text_temperature,
            "trust_features": trust_features, "gate_features": gate_features,
            # The declared bounds travel with the output so telemetry can report saturation
            # against them without reaching into the config from the training loop.
            "lambda_max": torch.as_tensor(self.cfg.lambda_max, device=q.device),
            "trust_scale": torch.as_tensor(self.cfg.trust_scale, device=q.device),
        }

    @staticmethod
    def branch_logits(output: dict[str, torch.Tensor], branch: str) -> torch.Tensor:
        """Named auditable branches of one forward pass, for evaluation readouts."""
        keys = {"final": "logits", "support": "metric_part", "support_floor": "neighbor_logits",
                "semantic": "semantic_logits", "semantic_text": "text_logits",
                "semantic_primitives": "primitive_logits"}
        if branch not in keys:
            raise ValueError(f"unknown evidence-gated branch {branch!r}")
        if keys[branch] not in output:
            raise ValueError(
                f"branch {branch!r} is not available for this checkpoint's semantic mode"
            )
        return output[keys[branch]]

    def telemetry(self) -> dict[str, float]:
        prior = (self.cfg.lambda_max * torch.sigmoid(self.lambda_prior.detach())).tolist()
        out = {f"classifier/lambda_prior_{bucket}": float(value)
               for bucket, value in zip(self.cfg.lambda_buckets, prior)}
        if self.primitive_head is not None:
            out.update(self.primitive_head.telemetry())
        return out
