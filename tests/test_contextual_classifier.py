"""Acceptance tests for the contextual semantic-voting head (plan of 2026-09-17, section 6)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model.blocks import AttentionSpec
from model.support.contextual_classifier import (
    ARCHITECTURE_VERSION, ContextualClassifierConfig, ContextualSupportClassifier,
)
from model.support.factory import build_classifier_from_blob

D, T = 12, 16


def _head(seed=0, n_layers=1):
    torch.manual_seed(seed)
    return ContextualSupportClassifier(
        AttentionSpec(d_model=D, n_heads=3, dropout=0.0),
        ContextualClassifierConfig(text_dim=T, n_layers=n_layers),
    ).eval()


def _episode(b=2, s=6, c=3, seed=3, *, support_mask=None):
    g = torch.Generator().manual_seed(seed)
    values = dict(
        query_feature=torch.randn(b, D, generator=g),
        support_feature=torch.randn(b, s, D, generator=g),
        support_label_text=torch.randn(b, s, T, generator=g),
        support_mask=torch.ones(b, s, dtype=torch.bool) if support_mask is None else support_mask,
        support_pair_slot=torch.arange(1, s + 1).unsqueeze(0).expand(b, -1).clone(),
        candidate_text=torch.randn(b, c, T, generator=g),
        candidate_mask=torch.ones(b, c, dtype=torch.bool),
    )
    return values


def _fp64_reference(head, values):
    """Recompute the scoring rule with an explicit double-precision loop from the head's own
    contextualised states (returned as diagnostics), independent of the vectorised code."""
    out = head(**values, return_diagnostics=True)
    log_w = out["log_support_weight"].double()
    log_label = out["log_label_distribution"].double()
    b, s = log_w.shape
    c = log_label.shape[-1]
    support = torch.full((b, c), -math.inf, dtype=torch.double)
    for row in range(b):
        for cand in range(c):
            terms = [log_w[row, k] + log_label[row, k, cand] for k in range(s) if values["support_mask"][row, k]]
            support[row, cand] = torch.logsumexp(torch.stack(terms), 0) if terms else -math.inf
    return out, support


def test_reference_arithmetic_matches_fp64_loop_and_distributions_sum_to_one():
    head = _head()
    values = _episode()
    out, support = _fp64_reference(head, values)
    assert torch.allclose(out["log_p_support"].double(), support, atol=1e-5)
    for key in ("logits", "log_p_support", "log_p_semantic"):
        assert torch.allclose(out[key].exp().sum(dim=1), torch.ones(2), atol=1e-5), key
    # A constant alpha of one half is the arithmetic average of the two branch probabilities.
    with torch.no_grad():
        head.gate[-1].weight.zero_(); head.gate[-1].bias.zero_()
    out = head(**values)
    assert torch.allclose(out["semantic_weight"], torch.full_like(out["semantic_weight"], 0.5))
    average = 0.5 * (out["log_p_support"].exp() + out["log_p_semantic"].exp())
    assert torch.allclose(out["logits"].exp(), average, atol=1e-5)
    assert torch.allclose(ContextualSupportClassifier.branch_logits(out, "fixed_half"), out["logits"], atol=1e-5)


def test_identity_initialisation_and_temperatures():
    head = _head()
    assert torch.equal(head.p_motion.weight, torch.eye(D))
    assert torch.equal(head.p_text.weight, torch.eye(D))
    assert torch.allclose(head.temperatures(), torch.full((3,), 0.07), atol=1e-6)
    assert head.gate[-1].bias.abs().max() == 0 and head.gate[-1].weight.abs().max() > 0


def test_masks_ragged_empty_mixed_and_extreme_inputs_stay_finite():
    head = _head()
    # Ragged supports and candidates, one support in row 1, one candidate live in row 0.
    values = _episode(b=2, s=4, c=3)
    values["support_mask"] = torch.tensor([[True, True, True, False], [True, False, False, False]])
    values["candidate_mask"] = torch.tensor([[True, False, False], [True, True, True]])
    values["support_feature"][:, 3] = float("nan")          # padded rows may carry garbage
    values["support_label_text"][1, 1:] = 1e6
    out = head(**values)
    live = values["candidate_mask"]
    assert torch.isfinite(out["logits"][live]).all() and (out["logits"][~live] < -1e29).all()
    assert torch.isfinite(out["log_p_support"][live]).all()
    # The training failure mode: padded candidates must not poison the backward pass.
    torch.nn.functional.nll_loss(out["logits"].masked_fill(~live, float("-inf")).log_softmax(-1),
                                 torch.tensor([0, 2])).backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    total = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
    assert torch.isfinite(total)
    head.zero_grad(set_to_none=True)
    # Mixed zero/enrolled batch: row 0 has no support and must equal its semantic distribution.
    values = _episode(b=2, s=3, c=4)
    values["support_mask"][0] = False
    out = head(**values)
    assert not bool(out["has_support"][0]) and bool(out["has_support"][1])
    assert torch.equal(out["logits"][0], out["log_p_semantic"][0])
    assert torch.equal(out["log_p_support"][0], out["log_p_semantic"][0])
    assert out["support_weight"][0].abs().sum() == 0
    assert torch.allclose(out["support_weight"][1].sum(), torch.tensor(1.0), atol=1e-5)
    # All-empty batch with S=0.
    values = _episode(b=3, s=0, c=5)
    out = head(**values)
    assert out["support_weight"].shape == (3, 0)
    assert torch.equal(out["logits"], out["log_p_semantic"])
    # Large logits (scaled features) remain finite and normalised.
    values = _episode(b=2, s=6, c=3)
    values["query_feature"] *= 1e3; values["support_feature"] *= 1e3; values["candidate_text"] *= 1e3
    out = head(**values)
    assert torch.isfinite(out["logits"]).all()
    assert torch.allclose(out["logits"].exp().sum(dim=1), torch.ones(2), atol=1e-4)
    loss = -out["logits"][:, 0].sum(); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)


def test_missing_candidate_fails_early():
    head = _head()
    values = _episode()
    values["candidate_mask"][1] = False
    with pytest.raises(ValueError, match="valid candidate"):
        head(**values)


def test_set_behaviour_permutation_padding_and_sibling_independence():
    head = _head()
    values = _episode(b=2, s=5, c=4)
    base = head(**values)["logits"]
    # Jointly permuting supports (features, labels, masks, tags) preserves the output.
    perm = torch.tensor([3, 0, 4, 1, 2])
    permuted = dict(values)
    for key in ("support_feature", "support_label_text", "support_mask", "support_pair_slot"):
        permuted[key] = values[key][:, perm]
    assert torch.allclose(head(**permuted)["logits"], base, atol=1e-5)
    # Permuting candidates permutes predictions.
    cperm = torch.tensor([2, 0, 3, 1])
    permuted = dict(values)
    for key in ("candidate_text", "candidate_mask"):
        permuted[key] = values[key][:, cperm]
    assert torch.allclose(head(**permuted)["logits"], base[:, cperm], atol=1e-5)
    # Added masked padding cannot change valid predictions.
    padded = dict(values)
    padded["support_feature"] = torch.cat((values["support_feature"], torch.full((2, 2, D), 7.0)), 1)
    padded["support_label_text"] = torch.cat((values["support_label_text"], torch.full((2, 2, T), -9.0)), 1)
    padded["support_mask"] = torch.cat((values["support_mask"], torch.zeros(2, 2, dtype=torch.bool)), 1)
    padded["support_pair_slot"] = torch.cat((values["support_pair_slot"], torch.zeros(2, 2, dtype=torch.long)), 1)
    padded["candidate_text"] = torch.cat((values["candidate_text"], torch.full((2, 1, T), 3.0)), 1)
    padded["candidate_mask"] = torch.cat((values["candidate_mask"], torch.zeros(2, 1, dtype=torch.bool)), 1)
    assert torch.allclose(head(**padded)["logits"][:, :4], base, atol=1e-5)
    # Changing a sibling query cannot change this query's prediction.
    sibling = dict(values)
    sibling["query_feature"] = values["query_feature"].clone(); sibling["query_feature"][1] += 5.0
    sibling["support_feature"] = values["support_feature"].clone(); sibling["support_feature"][1] *= -1
    assert torch.allclose(head(**sibling)["logits"][0], base[0], atol=1e-5)


def test_gradients_reach_every_parameter_on_enrolled_batch_and_semantic_path_on_zero_support():
    head = _head(n_layers=1).train()
    values = _episode(b=3, s=4, c=3)
    values["support_mask"][2] = False                      # mixed regime
    out = head(**values)
    loss = torch.nn.functional.nll_loss(out["logits"], torch.tensor([0, 1, 2]))
    loss.backward()
    missing = [name for name, p in head.named_parameters()
               if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    assert not missing, missing
    # A pure zero-support batch trains the semantic path without touching support-only parameters.
    head.zero_grad(set_to_none=True)
    values = _episode(b=2, s=0, c=3)
    out = head(**values)
    torch.nn.functional.nll_loss(out["logits"], torch.tensor([0, 2])).backward()
    assert head.text_adapter[1].weight.grad is not None and head.p_text.weight.grad.abs().sum() > 0
    assert head.raw_temperatures.grad is not None and float(head.raw_temperatures.grad[2]) != 0.0
    assert float(head.raw_temperatures.grad[0]) == 0.0 and float(head.raw_temperatures.grad[1]) == 0.0
    assert head.support_pair_tag.weight.grad is None or float(head.support_pair_tag.weight.grad.abs().sum()) == 0.0


def test_encoder_input_gradients_flow_to_query_and_supports():
    head = _head()
    values = _episode(b=2, s=4, c=3)
    values["query_feature"].requires_grad_(True); values["support_feature"].requires_grad_(True)
    out = head(**values)
    out["logits"][:, 0].sum().backward()
    assert values["query_feature"].grad.abs().sum() > 0
    assert values["support_feature"].grad.abs().sum() > 0


def test_factory_round_trip_is_strict_and_rejects_residual_ablation_flags(tmp_path):
    head = _head(seed=5)
    values = _episode()
    blob = {
        "architecture_version": ARCHITECTURE_VERSION,
        "classifier": head.state_dict(),
        "classifier_config": {"text_dim": T, "n_layers": 1},
        "attention_spec": {"d_model": D, "n_heads": 3, "dropout": 0.0},
    }
    restored, version = build_classifier_from_blob(blob)
    assert version == ARCHITECTURE_VERSION
    assert torch.allclose(restored(**values)["logits"], head(**values)["logits"])
    with pytest.raises(ValueError, match="not defined for the contextual head"):
        build_classifier_from_blob(blob, overrides={"residual_enabled": False})
    broken = dict(blob); broken["classifier"] = {k: v for k, v in head.state_dict().items() if "gate" not in k}
    with pytest.raises(RuntimeError):
        build_classifier_from_blob(broken)


def test_evaluator_gathers_off_roster_support_labels_without_candidate_binding(tmp_path):
    """Sealed/scenario readout must accept a support label outside the candidate roster."""
    from training.support_classifier import sealed_eval
    from training.support_classifier.sealed_eval import QueryPlan
    try:
        table = sealed_eval.make_label_text(("walking", "sitting"), torch.device("cpu"))
    except Exception as error:  # pragma: no cover - text model unavailable in this environment
        pytest.skip(f"label text encoder unavailable: {error}")
    text_dim = int(table.matrix.shape[-1])
    torch.manual_seed(1)
    head = ContextualSupportClassifier(
        AttentionSpec(d_model=D, n_heads=3, dropout=0.0),
        ContextualClassifierConfig(text_dim=text_dim, n_layers=1),
    ).eval()
    checkpoint = tmp_path / "contextual.pt"
    torch.save({
        "architecture_version": ARCHITECTURE_VERSION, "classifier": head.state_dict(),
        "classifier_config": {"text_dim": text_dim, "n_layers": 1},
        "attention_spec": {"d_model": D, "n_heads": 3, "dropout": 0.0},
    }, checkpoint)
    features = np.random.default_rng(0).standard_normal((6, D)).astype(np.float32)
    stream = SimpleNamespace(eval_labels=("walking", "sitting"))
    plans = (
        QueryPlan(query=0, support=(1, 2, 3), support_labels=("walking", "sitting", "jogging")),
        QueryPlan(query=4, support=(), support_labels=()),
    )
    predicted = sealed_eval._halo_contextual_predictions(features, stream, plans, checkpoint, torch.device("cpu"))
    assert len(predicted) == 2 and set(predicted) <= {"walking", "sitting"}
    for branch in ("semantic", "support", "fixed_half"):
        out = sealed_eval._halo_contextual_predictions(features, stream, plans, checkpoint,
                                                       torch.device("cpu"), branch=branch)
        assert len(out) == 2 and set(out) <= {"walking", "sitting"}
