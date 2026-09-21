import numpy as np
import torch

from model.blocks import AttentionSpec
from model.support.evidence_aware_classifier import (
    ARCHITECTURE_VERSION, EvidenceAwareClassifierConfig, EvidenceAwareSupportClassifier,
)
from model.support.factory import (
    ACTIVE_EXPERIMENTAL_CLASSIFIER_ARCHITECTURE,
    ABANDONED_CLASSIFIER_ARCHITECTURES, CLASSIFIER_ARCHITECTURE_STATUS,
    LEARNED_CLASSIFIER_ARCHITECTURES,
    PROMOTED_CLASSIFIER_ARCHITECTURE,
    build_classifier_from_blob,
)
from training.support_classifier.objectives import EvidenceAwareObjectiveConfig, evidence_aware_objective
from training.support_classifier.sampling import Episode, enrollment_counterfactual_group


def _model() -> EvidenceAwareSupportClassifier:
    return EvidenceAwareSupportClassifier(
        AttentionSpec(d_model=16, n_heads=4, dropout=0.0),
        EvidenceAwareClassifierConfig(text_dim=8, acquisition_dim=16, max_candidates=4, max_supports=5),
    )


def test_every_learned_checkpoint_architecture_has_an_explicit_lifecycle():
    assert LEARNED_CLASSIFIER_ARCHITECTURES <= CLASSIFIER_ARCHITECTURE_STATUS.keys()
    assert CLASSIFIER_ARCHITECTURE_STATUS[PROMOTED_CLASSIFIER_ARCHITECTURE] == "promoted-control"
    # v2 completed its matched sealed and scenario evaluation on 2026-09-19 and did not beat the
    # v3 control, so it is no longer the active experiment; see docs/results/RESULTS.md.
    assert CLASSIFIER_ARCHITECTURE_STATUS[ARCHITECTURE_VERSION] == "abandoned-negative-result"
    assert ARCHITECTURE_VERSION in ABANDONED_CLASSIFIER_ARCHITECTURES
    # Since the 2026-09-21 promotion the active experiment (T7) is a recipe on the promoted
    # architecture, so that architecture's lifecycle status is the promoted one.
    assert CLASSIFIER_ARCHITECTURE_STATUS[ACTIVE_EXPERIMENTAL_CLASSIFIER_ARCHITECTURE] in {
        "active-experimental", "promoted-control",
    }


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


def _objective_output(logits: torch.Tensor, *, has_support: torch.Tensor | None = None):
    batch, candidates = logits.shape
    return {
        "logits": logits,
        "support_status_logits": logits.clone(),
        "refined_support_logits": logits.clone(),
        "semantic_status_logits": logits.clone(),
        "candidate_has_direct_support": torch.zeros_like(logits, dtype=torch.bool),
        "has_any_support": (torch.zeros(batch, dtype=torch.bool)
                            if has_support is None else has_support),
    }


def test_best_path_zero_regret_has_one_smooth_hinge():
    logits = torch.zeros(1, 2)
    value, _ = evidence_aware_objective(
        _objective_output(logits), torch.tensor([0]), torch.ones(1, 2, dtype=torch.bool),
        EvidenceAwareObjectiveConfig(
            branch_preservation=False, temperature=0.1, group_temperature=0.1,
        ), group_ids=torch.tensor([-1]),
    )
    torch.testing.assert_close(value, value.new_tensor(0.1 * np.log(2.0)), atol=1e-7, rtol=0)


def test_private_group_ids_never_collide_with_explicit_ids():
    logits = torch.zeros(2, 2)
    _, metrics = evidence_aware_objective(
        _objective_output(logits), torch.tensor([0, 0]), torch.ones(2, 2, dtype=torch.bool),
        EvidenceAwareObjectiveConfig(branch_preservation=False),
        group_ids=torch.tensor([3, -1]),
    )
    assert metrics["aux/group_count"].item() == 2


def test_group_best_path_is_order_invariant_and_penalizes_a_worse_view():
    base = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    worse = base.clone(); worse[1] = torch.tensor([-1.0, 1.0])
    cfg = EvidenceAwareObjectiveConfig(branch_preservation=False)
    mask = torch.ones(2, 2, dtype=torch.bool); target = torch.tensor([0, 0]); groups = torch.tensor([4, 4])
    base_value, _ = evidence_aware_objective(_objective_output(base), target, mask, cfg, group_ids=groups)
    worse_output = _objective_output(worse)
    # Keep the detached reference fixed while worsening only the final path.
    worse_output["support_status_logits"] = base
    worse_output["refined_support_logits"] = base
    worse_output["semantic_status_logits"] = base
    worse_value, _ = evidence_aware_objective(worse_output, target, mask, cfg, group_ids=groups)
    reordered = {key: value.flip(0) for key, value in worse_output.items()}
    reordered_value, _ = evidence_aware_objective(
        reordered, target.flip(0), mask.flip(0), cfg, group_ids=groups.flip(0),
    )
    assert worse_value > base_value
    torch.testing.assert_close(worse_value, reordered_value)


def test_refined_support_starts_at_status_quo_for_exact_bindings():
    model = _model().eval()
    with torch.no_grad():
        output = model(**_inputs(support_bound=torch.tensor([[0, 1, 0]])))
    torch.testing.assert_close(
        output["refined_support_logits"], output["support_status_logits"], atol=1e-6, rtol=1e-6,
    )


def test_off_roster_support_contributes_semantic_evidence():
    model = _model().eval()
    data = _inputs(support_bound=torch.full((1, 3), -1))
    with torch.no_grad():
        first = model(**data)["support_status_logits"]
        data["support_label_text"] = -data["support_label_text"]
        second = model(**data)["support_status_logits"]
    assert not torch.equal(first, second)


def test_support_permutation_is_exactly_equivariant():
    model = _model().eval()
    data = _inputs()
    order = torch.tensor([2, 0, 1])
    permuted = dict(data)
    for name in ("support_feature", "support_acquisition", "support_label_text",
                 "support_bound", "support_mask", "support_pair_slot"):
        permuted[name] = data[name][:, order]
    with torch.no_grad():
        first, second = model(**data), model(**permuted)
    for name in ("logits", "support_status_logits", "refined_support_logits",
                 "semantic_status_logits", "semantic_reliance"):
        torch.testing.assert_close(first[name], second[name], atol=2e-6, rtol=2e-6)


def test_all_active_v2_blocks_receive_finite_gradients():
    model = _model().train()
    data = _inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    reached = {name: False for name, _ in model.named_parameters()}
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        output = model(**data)
        objective, _ = evidence_aware_objective(
            output, torch.tensor([0]), data["candidate_mask"], EvidenceAwareObjectiveConfig(),
        )
        (-output["logits"][0, 0] + 0.1 * objective).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
                reached[name] |= bool(parameter.grad.abs().max() > 0)
        optimizer.step()
    missing = [name for name, was_reached in reached.items() if not was_reached]
    assert not missing, f"parameters never received a nonzero gradient after gates opened: {missing}"


def test_support_prior_is_continuous_for_tiny_off_roster_mass():
    sensor = torch.tensor([[0.0]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    candidates = torch.ones(1, 3, dtype=torch.bool)
    uniform = torch.tensor([[-np.log(3.0)]])
    exact = torch.tensor([[[0.0, -100.0, -100.0]]])
    almost_exact = torch.tensor([[[-1e-6, -14.0, -14.0]]])
    first, _ = EvidenceAwareSupportClassifier._support_path(
        sensor, exact, mask, candidates, uniform,
    )
    second, _ = EvidenceAwareSupportClassifier._support_path(
        sensor, almost_exact, mask, candidates, uniform,
    )
    torch.testing.assert_close(first, second, atol=3e-6, rtol=0)


def test_fixed_support_floor_is_not_the_learned_temperature_branch():
    model = _model().eval()
    data = _inputs(support_bound=torch.tensor([[0, 1, 0]]))
    with torch.no_grad():
        first = model(**data)
        model.raw_temperatures[0].add_(2.0)
        second = model(**data)
    assert not torch.equal(first["support_status_logits"], second["support_status_logits"])
    torch.testing.assert_close(
        first["support_floor_logits"], second["support_floor_logits"], atol=0, rtol=0,
    )


def test_cpu_bfloat16_autocast_is_finite():
    model = _model().eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(**_inputs())
    assert torch.isfinite(output["logits"]).all()


def test_v2_checkpoint_round_trip_is_strict_and_exact():
    model = _model().eval()
    blob = {
        "architecture_version": ARCHITECTURE_VERSION,
        "attention_spec": dict(d_model=16, n_heads=4, dropout=0.0),
        "classifier_config": model.cfg.__dict__,
        "classifier": model.state_dict(),
    }
    restored, version = build_classifier_from_blob(blob)
    assert version == ARCHITECTURE_VERSION
    with torch.no_grad():
        expected, actual = model(**_inputs()), restored(**_inputs())
    torch.testing.assert_close(expected["logits"], actual["logits"], atol=0, rtol=0)


def test_auxiliary_switches_disable_terms_exactly():
    logits = torch.zeros(1, 2)
    value, metrics = evidence_aware_objective(
        _objective_output(logits), torch.tensor([0]), torch.ones(1, 2, dtype=torch.bool),
        EvidenceAwareObjectiveConfig(
            branch_preservation=False, best_path_non_regression=False,
        ),
    )
    assert value.item() == 0.0
    assert metrics["aux/active_terms"].item() == 0.0


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
