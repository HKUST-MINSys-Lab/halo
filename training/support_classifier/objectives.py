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
