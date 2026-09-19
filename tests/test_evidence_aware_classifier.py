import numpy as np
import torch

from model.blocks import AttentionSpec
from model.support.evidence_aware_classifier import (
    EvidenceAwareClassifierConfig, EvidenceAwareSupportClassifier,
)
from training.support_classifier.objectives import EvidenceAwareObjectiveConfig, evidence_aware_objective
from training.support_classifier.sampling import Episode, enrollment_counterfactual_group


def _model() -> EvidenceAwareSupportClassifier:
    return EvidenceAwareSupportClassifier(
        AttentionSpec(d_model=16, n_heads=4, dropout=0.0),
        EvidenceAwareClassifierConfig(text_dim=8, acquisition_dim=16, max_candidates=4, max_supports=5),
    )


def _inputs(*, support_bound=None):
    torch.manual_seed(7)
    bound = torch.tensor([[0, 1, -1]]) if support_bound is None else support_bound
    return dict(
        query_feature=torch.randn(1, 16), support_feature=torch.randn(1, 3, 16),
        query_acquisition=torch.randn(1, 16), support_acquisition=torch.randn(1, 3, 16),
        support_label_text=torch.randn(1, 3, 8), support_bound=bound,
        support_mask=torch.ones(1, 3, dtype=torch.bool), support_pair_slot=torch.tensor([[0, 1, 2]]),
        candidate_text=torch.randn(1, 3, 8), candidate_mask=torch.ones(1, 3, dtype=torch.bool),
    )


def test_semantic_status_is_independent_of_support_rows():
    model = _model().eval()
    first = _inputs()
    second = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in first.items()}
    second["support_feature"] = torch.randn_like(second["support_feature"]) * 50
    second["support_acquisition"] = torch.randn_like(second["support_acquisition"]) * 50
    second["support_label_text"] = torch.randn_like(second["support_label_text"]) * 50
    with torch.no_grad():
        a, b = model(**first), model(**second)
    torch.testing.assert_close(a["semantic_status_logits"], b["semantic_status_logits"], atol=0, rtol=0)


def test_zero_support_is_exact_semantic_path_and_gradients_reach_adapters():
    model = _model()
    data = _inputs(support_bound=torch.full((1, 3), -1))
    data["support_mask"].zero_()
    output = model(**data)
    torch.testing.assert_close(output["logits"], output["semantic_status_logits"], atol=1e-6, rtol=1e-6)
    output["logits"].sum().backward()
    assert model.motion_adapter[1].weight.grad is not None
    assert model.acquisition_adapter[1].weight.grad is not None


def test_evidence_aware_objective_groups_counterfactual_variants():
    model = _model()
    data = _inputs()
    # Duplicate a valid episode so grouping has a meaningful, deterministic contract.
    data = {key: value.repeat((2,) + (1,) * (value.ndim - 1)) for key, value in data.items()}
    output = model(**data)
    value, metrics = evidence_aware_objective(
        output, torch.tensor([0, 0]), data["candidate_mask"], EvidenceAwareObjectiveConfig(),
        group_ids=torch.tensor([9, 9]),
    )
    assert torch.isfinite(value)
    assert metrics["aux/group_count"].item() == 1


def test_enrollment_counterfactuals_hold_the_recognition_problem_fixed():
    base = Episode(query=7, support=(10, 11, 12, 13), support_candidate=(0, 0, 1, 1),
                   candidates=("walk", "sit"), gt_slot=0, mode="compatible",
                   requested_support=4, shrunk=False, support_window_groups=((10,), (11,), (12,), (13,)),
                   enrollment_regime="complete")
    group = enrollment_counterfactual_group(base, group_id=4, rng=np.random.default_rng(1))
    assert group.axis == "enrollment"
    assert len(group.views) == 4
    for view in group.views:
        assert (view.query, view.candidates, view.gt_slot, view.counterfactual_group) == (7, ("walk", "sit"), 0, 4)
    assert not group.views[-1].support and group.views[-1].zero_shot
