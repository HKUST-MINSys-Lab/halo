"""Modular path-improvement objectives for support classifiers.

Every auxiliary objective has the same contract: a named prediction path should improve the true
class log-odds over a detached reference path.  The references define targets, never parameter
freezing, so gradients remain free to update the complete encoder, conditioner and classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


COMPARISONS = (
    "contextual_support_over_support_floor",
    "final_over_best_branch",
)


@dataclass(frozen=True)
class ImprovementObjectiveConfig:
    enabled: bool = True
    weight: float = 0.1
    margin: float = 0.0
    temperature: float = 0.1
    comparisons: tuple[str, ...] = COMPARISONS

    def __post_init__(self) -> None:
        if self.weight < 0 or self.temperature <= 0 or self.margin < 0:
            raise ValueError("invalid path-improvement objective scale")
        unknown = set(self.comparisons) - set(COMPARISONS)
        if unknown:
            raise ValueError(f"unknown path-improvement comparisons: {sorted(unknown)}")


def true_class_log_odds(
    logits: torch.Tensor, target: torch.Tensor, candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """True-class score minus log-sum-exp of all valid alternatives."""
    valid = candidate_mask.bool()
    if logits.shape != valid.shape or target.shape != logits.shape[:1]:
        raise ValueError("logit, target and candidate-mask shapes do not agree")
    true = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    other = valid.clone()
    other.scatter_(1, target.unsqueeze(1), False)
    # Episode construction requires at least two candidates. Fail loudly for custom callers.
    if not bool(other.any(dim=1).all()):
        raise ValueError("log-odds improvement requires at least two valid candidates")
    competitor = torch.logsumexp(logits.masked_fill(~other, float("-inf")), dim=1)
    return true - competitor


def improvement_loss(
    better_quality: torch.Tensor,
    reference_quality: torch.Tensor,
    selected: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor | None:
    """Smooth ranking loss; the reference is detached by contract."""
    selected = selected.bool()
    if not bool(selected.any()):
        return None
    delta = (reference_quality.detach() + margin - better_quality) / temperature
    return F.softplus(delta[selected]).mean() * temperature


def contextual_path_improvement_objective(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    candidate_mask: torch.Tensor,
    cfg: ImprovementObjectiveConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the configured comparisons and average the active terms.

    Averaging, rather than summing, keeps the global auxiliary weight stable as comparisons are
    toggled.  This makes later counterfactual paths additive configuration, not a loss retuning.
    """
    zero = output["logits"].new_zeros(())
    if not cfg.enabled or cfg.weight == 0 or not cfg.comparisons:
        return zero, {"aux/path_improvement": zero.detach(), "aux/active_comparisons": zero.detach()}

    qualities = {
        name: true_class_log_odds(output[key], target, candidate_mask)
        for name, key in {
            "support_floor": "support_floor_logits",
            "contextual_support": "contextual_support_logits",
            "semantic": "semantic_logits",
            "final": "logits",
        }.items()
    }
    truth_enrolled = output["k_c"].gather(1, target.unsqueeze(1)).squeeze(1).gt(0)
    terms: list[torch.Tensor] = []
    metrics: dict[str, torch.Tensor] = {}

    if "contextual_support_over_support_floor" in cfg.comparisons:
        term = improvement_loss(
            qualities["contextual_support"], qualities["support_floor"], truth_enrolled,
            margin=cfg.margin, temperature=cfg.temperature,
        )
        if term is not None:
            terms.append(term)
            metrics["aux/contextual_support_over_support_floor"] = term.detach()

    if "final_over_best_branch" in cfg.comparisons:
        reference = torch.maximum(qualities["contextual_support"], qualities["semantic"])
        term = improvement_loss(
            qualities["final"], reference, torch.ones_like(truth_enrolled),
            margin=cfg.margin, temperature=cfg.temperature,
        )
        if term is not None:
            terms.append(term)
            metrics["aux/final_over_best_branch"] = term.detach()

    value = torch.stack(terms).mean() if terms else zero
    metrics["aux/path_improvement"] = value.detach()
    metrics["aux/active_comparisons"] = value.new_tensor(float(len(terms)))
    return value, metrics


@dataclass(frozen=True)
class EvidenceAwareObjectiveConfig:
    """Auxiliaries for ``support_evidence_aware_v2``.

    The three terms are deliberately all path-improvement contracts.  They do
    not supervise a router or freeze a branch: each asks an evidence path to be
    at least as useful as an auditable detached reference.
    """
    enabled: bool = True
    weight: float = 0.1
    margin: float = 0.0
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if self.weight < 0 or self.margin < 0 or self.temperature <= 0:
            raise ValueError("invalid evidence-aware auxiliary scale")


def _group_mean(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """Mean each non-negative group; ``-1`` makes a row its own group."""
    if values.ndim != 1 or group_ids.shape != values.shape:
        raise ValueError("group values and identifiers must be one-dimensional and aligned")
    ids = group_ids.long().clone()
    private = ids.lt(0)
    ids[private] = torch.arange(ids.numel(), device=ids.device)[private] + ids.numel()
    _, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    total = torch.zeros(int(inverse.max()) + 1, device=values.device, dtype=values.dtype)
    count = torch.zeros_like(total)
    total.scatter_add_(0, inverse, values)
    count.scatter_add_(0, inverse, torch.ones_like(values))
    return total / count.clamp_min(1)


def evidence_aware_objective(
    output: dict[str, torch.Tensor], target: torch.Tensor, candidate_mask: torch.Tensor,
    cfg: EvidenceAwareObjectiveConfig, *, group_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Branch preservation and group-level best-path non-regression.

    ``group_ids`` identifies counterfactual variants of a fixed query/candidate
    episode.  Ordinary independently drawn episodes pass ``None`` and are each
    treated as their own group.
    """
    zero = output["logits"].new_zeros(())
    if not cfg.enabled or cfg.weight == 0:
        return zero, {"aux/evidence_aware": zero.detach(), "aux/active_terms": zero.detach()}
    b = target.numel()
    groups = torch.arange(b, device=target.device) if group_ids is None else group_ids.to(target.device)
    support_status = true_class_log_odds(output["support_status_logits"], target, candidate_mask)
    refined = true_class_log_odds(output["refined_support_logits"], target, candidate_mask)
    semantic = true_class_log_odds(output["semantic_status_logits"], target, candidate_mask)
    final = true_class_log_odds(output["logits"], target, candidate_mask)
    direct = output["candidate_has_direct_support"].gather(1, target[:, None]).squeeze(1)
    # Support-only counterfactual views share a query/candidate semantic problem.  Count that
    # semantic CE once per explicit group rather than letting four enrollment views multiply it.
    semantic_ce = F.nll_loss(output["semantic_status_logits"], target, reduction="none")
    branch_terms = [_group_mean(semantic_ce, groups).mean()]
    if bool(direct.any()):
        branch_terms.append(F.nll_loss(output["refined_support_logits"][direct], target[direct]))
    improve = improvement_loss(refined, support_status, direct, margin=cfg.margin, temperature=cfg.temperature)
    if improve is not None:
        branch_terms.append(improve)
    branch = torch.stack(branch_terms).mean()
    reference = torch.maximum(refined.detach(), semantic.detach())
    regret = F.softplus((reference + cfg.margin - final) / cfg.temperature) * cfg.temperature
    # Aggregate counterfactual variants before applying the soft hinge.  A single
    # easy view cannot hide a regression in another deployment condition.
    grouped_regret = _group_mean(regret, groups)
    best = F.softplus(grouped_regret / cfg.temperature).mean() * cfg.temperature
    active = torch.stack((branch, best)).mean()
    metrics = {
        "aux/evidence_aware": active.detach(), "aux/branch_preservation": branch.detach(),
        "aux/best_path_non_regression": best.detach(), "aux/active_terms": active.new_tensor(2.0),
        "aux/group_count": active.new_tensor(float(grouped_regret.numel())),
    }
    return active, metrics
