"""The sensor-only readout: labels never enter the learned path, and step 0 is the closed-form vote."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from model.blocks import AttentionSpec
from model.evidence.comparator import (
    ComparatorConfig,
    SupportComparator,
    comparator_config_from_checkpoint,
    comparator_logits,
)
from training.compare.step0 import assert_identity_at_init

D, TEXT = 32, 24
SPEC = AttentionSpec(d_model=D, n_heads=4, ffn_mult=2, dropout=0.0)


def _episode(B=2, C=4, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    unit = lambda *s: F.normalize(n(*s), dim=-1)  # noqa: E731
    bound = torch.randint(-1, C, (B, K), generator=g)
    return {
        "candidate_text": unit(B, C, TEXT),
        "query_feature": n(B, 1, D),
        "query_descriptor": unit(B, 1, TEXT),
        "query_mask": torch.ones(B, 1, dtype=torch.bool),
        "support_feature": n(B, K, D),
        "support_descriptor": unit(B, K, TEXT),
        "support_label_text": unit(B, K, TEXT),
        "support_bound": bound,
        "support_mask": torch.ones(B, K, dtype=torch.bool),
        "candidate_slot": (1 + torch.arange(C)).unsqueeze(0).expand(B, C).contiguous(),
        "candidate_mask": torch.ones(B, C, dtype=torch.bool),
    }


def _module(use_descriptor=False, wake=0.0):
    torch.manual_seed(0)
    m = SupportComparator(SPEC, ComparatorConfig(
        text_dim=TEXT, n_layers=2, n_slots=16, readout="sensor_only", use_descriptor=use_descriptor,
    )).eval()
    if wake:
        torch.nn.init.normal_(m.shift_head.weight, std=wake)
    return m


def test_default_config_is_sensor_only_without_descriptor():
    cfg = ComparatorConfig()
    assert cfg.readout == "sensor_only" and cfg.use_descriptor is False
    with pytest.raises(ValueError):
        ComparatorConfig(readout="attention")


def test_sensor_only_has_no_text_parameters_and_fused_has_them():
    m = _module()
    names = {name for name, _ in m.named_parameters()}
    assert not any(name.startswith(("proj_text", "support_fusion", "residual_head")) for name in names)
    assert "shift_head.weight" in names
    fused = SupportComparator(SPEC, ComparatorConfig(text_dim=TEXT, readout="fused"))
    fused_names = {name for name, _ in fused.named_parameters()}
    assert {"proj_text.weight", "support_fusion.weight", "residual_head.weight"} <= fused_names


@pytest.mark.parametrize("center", [False, True])
def test_identity_at_init(center):
    gap = assert_identity_at_init(_module(), center=center)
    assert gap == 0.0
    episode = _episode()
    learned = comparator_logits(_module(), **episode, center=center)
    closed = comparator_logits(None, **episode, center=center)
    torch.testing.assert_close(learned["logits"], closed["logits"])
    torch.testing.assert_close(learned["support_weight"], closed["support_weight"])
    assert learned["residual"].abs().max() == 0.0


def test_learned_path_is_blind_to_every_text_input():
    """Change candidate text, label text and both descriptors: the shift and the support weights
    must not move. Only the closed-form vote may respond to text."""
    m = _module(wake=0.5)
    episode = _episode()
    reference = comparator_logits(m, **episode)
    g = torch.Generator().manual_seed(99)
    altered = dict(episode)
    for key in ("candidate_text", "support_label_text", "support_descriptor", "query_descriptor"):
        altered[key] = F.normalize(torch.randn(*episode[key].shape, generator=g), dim=-1)
    other = comparator_logits(m, **altered)
    torch.testing.assert_close(other["support_weight"], reference["support_weight"])
    # and the raw comparator output itself
    slot = torch.where(episode["support_bound"].ge(0),
                       torch.gather(episode["candidate_slot"], 1, episode["support_bound"].clamp_min(0)),
                       torch.zeros_like(episode["support_bound"]))
    kwargs = {k: v for k, v in episode.items() if k != "support_bound"}
    kwargs["support_slot"] = slot
    a = m(**kwargs)
    kwargs.update({k: altered[k] for k in ("candidate_text", "support_label_text",
                                            "support_descriptor", "query_descriptor")})
    b = m(**kwargs)
    torch.testing.assert_close(a, b)
    assert a.shape == (2, 6), "sensor_only emits one shift per support row"


def test_descriptor_enters_only_when_enabled():
    episode = _episode()
    changed = {**episode, "support_descriptor": episode["support_descriptor"].flip(1)}
    off = _module(use_descriptor=False, wake=0.5)
    torch.testing.assert_close(
        comparator_logits(off, **episode)["support_weight"],
        comparator_logits(off, **changed)["support_weight"],
    )
    on = _module(use_descriptor=True, wake=0.5)
    assert "proj_text.weight" in dict(on.named_parameters())
    assert not torch.allclose(
        comparator_logits(on, **episode)["support_weight"],
        comparator_logits(on, **changed)["support_weight"],
    )


def test_woken_shift_reweights_support_but_keeps_the_vote_structure():
    m = _module(wake=0.5)
    episode = _episode()
    out = comparator_logits(m, **episode)
    closed = comparator_logits(None, **episode)
    assert not torch.allclose(out["support_weight"], closed["support_weight"])
    torch.testing.assert_close(out["support_weight"].sum(1), torch.ones(2))
    torch.testing.assert_close(out["base_logits"], closed["logits"])
    torch.testing.assert_close(out["logits"] - out["base_logits"], out["residual"])


def test_support_permutation_equivariance_and_logit_invariance():
    m = _module(wake=0.5)
    episode = _episode()
    perm = torch.randperm(6, generator=torch.Generator().manual_seed(3))
    permuted = {
        **episode,
        **{k: episode[k][:, perm] for k in ("support_feature", "support_descriptor",
                                            "support_label_text", "support_bound", "support_mask")},
    }
    a, b = comparator_logits(m, **episode), comparator_logits(m, **permuted)
    torch.testing.assert_close(a["logits"], b["logits"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(a["support_weight"][:, perm], b["support_weight"], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("center", [False, True])
def test_padding_rows_change_nothing(center):
    m = _module(wake=0.5)
    episode = _episode(B=1, K=3)
    original = comparator_logits(m, **episode, center=center)
    padded = dict(episode)
    for key in ("support_feature", "support_descriptor", "support_label_text",
                "support_bound", "support_mask"):
        value = episode[key]
        tail = value.new_full((1, 20, *value.shape[2:]), -1 if key == "support_bound" else 0)
        padded[key] = torch.cat([value, tail], dim=1)
    actual = comparator_logits(m, **padded, center=center)
    torch.testing.assert_close(actual["logits"], original["logits"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual["support_weight"].sum(1), torch.ones(1))
    assert actual["support_weight"][:, 3:].abs().max() == 0.0


def test_empty_support_set_is_safe():
    m = _module(wake=0.5)
    episode = _episode(B=1, K=2)
    episode["support_mask"].zero_()
    out = comparator_logits(m, **episode)
    assert torch.isfinite(out["logits"]).all()
    assert out["support_weight"].abs().max() == 0.0


def test_every_parameter_learns_within_two_steps():
    m = _module().train()
    optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
    episode = _episode()
    for step in range(2):
        optimizer.zero_grad()
        out = comparator_logits(m, **episode)
        F.cross_entropy(out["logits"], torch.tensor([0, 1])).backward()
        assert m.shift_head.weight.grad.norm() > 0
        if step == 1:
            dead = [name for name, p in m.named_parameters()
                    if p.grad is None or p.grad.norm() == 0]
            assert not dead, dead
        optimizer.step()
    assert not torch.allclose(comparator_logits(m.eval(), **episode)["support_weight"],
                              comparator_logits(None, **episode)["support_weight"])


def test_gradient_reaches_the_encoder_features_through_the_shift():
    m = _module(wake=0.5)
    episode = _episode()
    episode["support_feature"].requires_grad_()
    comparator_logits(m, **episode)["logits"].square().sum().backward()
    assert episode["support_feature"].grad.abs().sum() > 0


def test_legacy_checkpoint_config_loads_as_fused_and_new_round_trips():
    legacy = {"text_dim": TEXT, "n_layers": 2, "n_slots": 16, "identity_gain_init": 0.25}
    cfg = comparator_config_from_checkpoint(legacy)
    assert cfg.readout == "fused" and cfg.use_descriptor is True
    fused = SupportComparator(SPEC, cfg)
    fused.load_state_dict(SupportComparator(SPEC, ComparatorConfig(**legacy, readout="fused")).state_dict())
    new = ComparatorConfig(text_dim=TEXT, n_layers=1, use_descriptor=True)
    assert comparator_config_from_checkpoint(dataclasses.asdict(new)) == new
    with pytest.raises(RuntimeError):
        SupportComparator(SPEC, new).load_state_dict(fused.state_dict())


def test_telemetry_reports_the_live_head():
    m = _module(wake=0.5)
    t = m.telemetry()
    assert t["comparator/residual_head_norm"] > 0 and t["comparator/readout_is_fused"] == 0.0
