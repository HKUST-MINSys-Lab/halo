from __future__ import annotations

import torch
import numpy as np
from types import SimpleNamespace

from model.blocks import AttentionSpec
from model.support.contextual_residual_classifier import (
    ContextualResidualClassifierConfig, ContextualResidualSupportClassifier,
)
from training.support_classifier.objectives import (
    ImprovementObjectiveConfig, contextual_path_improvement_objective,
)
from model.support.factory import build_classifier_from_blob


def _case(*, supports: int = 4):
    torch.manual_seed(7)
    b, c, d, t = 2, 3, 16, 12
    query = torch.randn(b, d, requires_grad=True)
    support = torch.randn(b, supports, d, requires_grad=True)
    query_acquisition = torch.randn(b, d, requires_grad=True)
    support_acquisition = torch.randn(b, supports, d, requires_grad=True)
    support_text = torch.randn(b, supports, t)
    support_mask = torch.ones(b, supports, dtype=torch.bool)
    bound = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 1]], dtype=torch.long)[:, :supports]
    pair = torch.arange(1, supports + 1).unsqueeze(0).expand(b, -1)
    candidate_text = torch.randn(b, c, t)
    candidate_mask = torch.ones(b, c, dtype=torch.bool)
    return dict(
        query_feature=query, support_feature=support,
        query_acquisition=query_acquisition, support_acquisition=support_acquisition,
        support_label_text=support_text, support_bound=bound, support_mask=support_mask,
        support_pair_slot=pair, candidate_text=candidate_text,
        candidate_mask=candidate_mask,
    )


def _head():
    spec = AttentionSpec(d_model=16, n_heads=4, ffn_mult=2, dropout=0.0)
    cfg = ContextualResidualClassifierConfig(
        text_dim=12, input_dim=16, acquisition_dim=16, n_layers=2,
        max_candidates=8, max_supports=16,
    )
    return ContextualResidualSupportClassifier(spec, cfg)


def test_initial_contextual_support_is_neighbour_floor_for_exact_full_enrollment():
    head = _head().eval()
    output = head(**_case())
    assert torch.allclose(
        output["contextual_support_logits"], output["support_floor_logits"], atol=1e-6,
    )
    assert torch.count_nonzero(output["support_correction"]) == 0


def test_zero_support_reduces_exactly_to_semantic_path():
    head = _head().eval()
    case = _case(supports=0)
    output = head(**case)
    assert output["support_weight"].shape == (2, 0)
    assert torch.allclose(output["logits"], output["semantic_logits"], atol=1e-6)


def test_support_permutation_is_equivariant_and_predictions_are_invariant():
    head = _head().eval()
    case = _case()
    reference = head(**case)["logits"]
    permutation = torch.tensor([2, 0, 3, 1])
    for key in ("support_feature", "support_acquisition", "support_label_text",
                "support_bound", "support_mask", "support_pair_slot"):
        case[key] = case[key][:, permutation]
    actual = head(**case)["logits"]
    assert torch.allclose(actual, reference, atol=2e-6)


def test_main_and_generic_auxiliary_objective_reach_all_learnable_paths():
    head = _head().train()
    case = _case()
    output = head(**case)
    target = torch.tensor([0, 2])
    main = torch.nn.functional.cross_entropy(output["logits"], target)
    auxiliary, metrics = contextual_path_improvement_objective(
        output, target, case["candidate_mask"], ImprovementObjectiveConfig(),
    )
    (main + 0.1 * auxiliary).backward()
    assert torch.isfinite(main + auxiliary)
    assert metrics["aux/active_comparisons"].item() == 2
    assert case["query_feature"].grad is not None
    assert case["query_acquisition"].grad is not None
    assert case["support_acquisition"].grad is not None
    assert head.correction_candidate.weight.grad is not None
    assert head.semantic_gate[-1].weight.grad is not None
    assert float(head.correction_candidate.weight.grad.abs().sum()) > 0
    assert float(head.semantic_gate[-1].weight.grad.abs().sum()) > 0


def test_candidate_count_is_dynamic_for_shared_gate():
    head = _head().eval()
    case = _case()
    for key in ("candidate_text", "candidate_mask"):
        case[key] = case[key][:, :2]
    case["support_bound"] = case["support_bound"].clamp_max(1)
    output = head(**case)
    assert output["logits"].shape == (2, 2)


def test_checkpoint_factory_round_trip_is_exact():
    head = _head().eval()
    blob = {
        "architecture_version": "support_contextual_residual_v1",
        "classifier": head.state_dict(),
        "classifier_config": head.cfg.__dict__,
        "attention_spec": head.spec.__dict__,
    }
    loaded, version = build_classifier_from_blob(blob, device=torch.device("cpu"))
    assert version == "support_contextual_residual_v1"
    expected = head(**_case())["logits"]
    actual = loaded(**_case())["logits"]
    assert torch.equal(expected, actual)


def test_sealed_scorer_requires_and_uses_aligned_acquisition_rows(tmp_path, monkeypatch):
    from training.support_classifier import sealed_eval
    from training.support_classifier.sealed_eval import QueryPlan

    head = _head().eval()
    checkpoint = tmp_path / "contextual-residual.pt"
    torch.save({
        "architecture_version": "support_contextual_residual_v1",
        "classifier": head.state_dict(),
        "classifier_config": head.cfg.__dict__,
        "attention_spec": head.spec.__dict__,
    }, checkpoint)
    # Avoid constructing the frozen text tower in this unit: the scorer contract is the aligned
    # feature/acquisition plumbing, not SBERT itself.
    table = SimpleNamespace(
        labels=("walking", "sitting"), index={"walking": 0, "sitting": 1},
        matrix=torch.randn(2, 12),
    )
    table.ids = lambda labels: [table.index[label] for label in labels]
    monkeypatch.setattr(sealed_eval, "make_label_text", lambda labels, device: table)
    features = np.random.default_rng(3).normal(size=(5, 16)).astype(np.float32)
    acquisitions = np.random.default_rng(4).normal(size=(5, 16)).astype(np.float32)
    stream = SimpleNamespace(eval_labels=("walking", "sitting"))
    plans = (QueryPlan(query=0, support=(1, 2), support_labels=("walking", "sitting")),)
    prediction = sealed_eval._halo_contextual_residual_predictions(
        features, stream, plans, checkpoint, torch.device("cpu"),
        acquisitions=acquisitions,
    )
    assert len(prediction) == 1 and prediction[0] in stream.eval_labels
