"""Contract tests for the annotated label side of the primitive path (T8).

What must hold: the annotations cover the supervised training vocabulary exactly and nothing
else; every value is a value of its axis; the frozen map returns one distribution per axis and
recovers an annotated label's own profile; the exact mixture cannot fall below half the better
half; and the defaults leave T7 checkpoints unchanged.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from model.blocks import AttentionSpec
from model.support.evidence_gated_classifier import (
    ARCHITECTURE_VERSION, EvidenceGatedClassifierConfig, EvidenceGatedSupportClassifier,
    probability_semantic_mixture,
)
from model.support.factory import build_classifier_from_blob, classifier_try_name
from model.support.primitive_annotations import (
    ANNOTATION_VERSION, ANNOTATIONS, ANNOTATIONS_V1, SCRAMBLED_ANNOTATION_VERSION,
    annotated_labels, annotation_hash, annotation_profile_matrix,
)
from model.support.primitive_semantics import (
    VOCABULARIES, PrimitiveSemanticConfig, PrimitiveSemanticHead, axis_names, axis_slices,
    n_primitives,
)

D_MODEL, TEXT_DIM = 32, 384


# --------------------------------------------------------------------- the annotations
def test_annotations_cover_the_training_vocabulary_exactly():
    """Every supervised training concept is annotated; no evaluation-only concept is added."""
    try:
        from baselines.data import load_global_labels
        training = {str(label) for label in load_global_labels()}
    except Exception as exc:  # pragma: no cover - corpus not materialised in this environment
        pytest.skip(f"training label list unavailable: {exc}")
    assert set(ANNOTATIONS_V1) == training, {
        "missing": sorted(training - set(ANNOTATIONS_V1)), "extra": sorted(set(ANNOTATIONS_V1) - training),
    }
    assert len(ANNOTATIONS_V1) == 155


def test_every_annotation_value_belongs_to_its_axis():
    allowed = {axis: {name for name, _ in values} for axis, values in VOCABULARIES["primitives-v1"]}
    axes = axis_names()
    for label, values in ANNOTATIONS_V1.items():
        assert len(values) == len(axes), label
        for axis, value in zip(axes, values):
            assert value in allowed[axis], (label, axis, value)


def test_scrambled_control_is_a_permutation_with_its_own_hash():
    assert sorted(ANNOTATIONS[SCRAMBLED_ANNOTATION_VERSION].values()) == sorted(ANNOTATIONS_V1.values())
    assert ANNOTATIONS[SCRAMBLED_ANNOTATION_VERSION] != ANNOTATIONS_V1
    assert annotation_hash(SCRAMBLED_ANNOTATION_VERSION) != annotation_hash()


def test_profile_matrix_is_one_distribution_per_axis():
    labels, matrix = annotation_profile_matrix(smoothing=0.05)
    assert matrix.shape == (len(labels), n_primitives())
    for block in axis_slices():
        torch.testing.assert_close(matrix[:, block].sum(-1), torch.ones(len(labels)))
        assert float(matrix[:, block].max()) < 1.0 and float(matrix[:, block].min()) > 0.0


# --------------------------------------------------------------------- the annotated head
def fake_anchors(seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(len(annotated_labels()), TEXT_DIM, generator=generator), dim=-1)


def annotated_head(**overrides) -> PrimitiveSemanticHead:
    cfg = PrimitiveSemanticConfig(text_dim=TEXT_DIM, label_side="annotated", combiner="bilinear",
                                  **overrides)
    return PrimitiveSemanticHead(D_MODEL, cfg, values=fake_anchors()).eval()


def test_annotated_head_carries_anchors_as_frozen_buffers():
    head = annotated_head()
    buffers = dict(head.named_buffers())
    assert buffers["anchors"].shape == (155, TEXT_DIM)
    assert buffers["anchor_profiles"].shape == (155, n_primitives())
    assert not any(name.startswith(("anchors", "anchor_profiles")) for name, _ in head.named_parameters())
    assert "values" not in buffers, "the sentence bank has no role on the annotated label side"


def test_annotated_label_recovers_its_own_profile_at_low_temperature():
    head = annotated_head(label_map_temperature=0.01)
    anchors = head.anchors
    text = anchors[:6].unsqueeze(0)
    mask = torch.ones(1, 6, dtype=torch.bool)
    profile = head.candidate_profile(text, mask)[0]
    _, truth = annotation_profile_matrix(smoothing=0.05)
    for block in axis_slices():
        assert torch.equal(profile[:, block].argmax(-1), truth[:6, block].argmax(-1))
        torch.testing.assert_close(profile[:, block].sum(-1), torch.ones(6))


def test_unseen_label_inherits_a_convex_combination_of_anchor_profiles():
    """An unseen label's profile is a weighted average of annotated ones: attributes, not names."""
    head = annotated_head(label_map_temperature=0.1)
    generator = torch.Generator().manual_seed(9)
    unseen = F.normalize(torch.randn(1, 4, TEXT_DIM, generator=generator), dim=-1)
    profile = head.candidate_profile(unseen, torch.ones(1, 4, dtype=torch.bool))[0]
    for block in axis_slices():
        torch.testing.assert_close(profile[:, block].sum(-1), torch.ones(4))
    low, high = head.anchor_profiles.min(0).values, head.anchor_profiles.max(0).values
    assert bool((profile >= low - 1e-6).all()) and bool((profile <= high + 1e-6).all())


def test_config_rejects_combiners_that_do_not_apply():
    with pytest.raises(ValueError, match="not defined"):
        PrimitiveSemanticConfig(label_side="annotated", combiner="projection")
    with pytest.raises(ValueError, match="not defined"):
        PrimitiveSemanticConfig(label_side="sentences", combiner="bilinear")


def test_bilinear_repair_is_identity_initialised_and_trainable():
    head = annotated_head()
    assert all(torch.equal(weight, torch.eye(weight.shape[0]))
               for weight in head.interaction_blocks)
    query = torch.randn(3, D_MODEL)
    text = head.anchors[:5].unsqueeze(0).expand(3, -1, -1)
    out = head(query, text, torch.ones(3, 5, dtype=torch.bool))
    F.nll_loss(out["logits"], torch.tensor([0, 1, 2])).backward()
    assert sum(float(matrix.grad.abs().sum()) for matrix in head.interaction_blocks) > 0
    assert float(head.keys.weight.grad.abs().sum()) > 0
    assert all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
               for parameter in head.parameters())


def test_bilinear_repair_is_bounded_and_cannot_mix_axes():
    head = annotated_head()
    profile = torch.zeros(1, n_primitives())
    candidate = torch.zeros(1, 1, n_primitives())
    for block in axis_slices():
        profile[:, block.start] = 1.0
        candidate[:, :, block.start] = 1.0
    with torch.no_grad():
        for matrix in head.interaction_blocks:
            matrix.mul_(100.0)
    score = head.agreement(profile, candidate)
    assert float(score.detach().abs().max()) <= 1.0 + 1e-6
    assert sum(matrix.numel() for matrix in head.interaction_blocks) == sum(
        (block.stop - block.start) ** 2 for block in axis_slices()
    )


def test_dense_bilinear_smoke_checkpoint_migrates_strictly():
    head = annotated_head()
    state = head.state_dict()
    dense = torch.zeros(n_primitives(), n_primitives())
    for block, matrix in zip(axis_slices(), head.interaction_blocks):
        dense[block, block] = matrix.detach()
    for key in [key for key in state if key.startswith("interaction_blocks.")]:
        del state[key]
    state["interaction"] = dense
    restored = annotated_head()
    restored.load_state_dict(state, strict=True)
    for expected, actual in zip(head.interaction_blocks, restored.interaction_blocks):
        torch.testing.assert_close(expected, actual)


# --------------------------------------------------------------------- the mixture in v4
def v4(**overrides) -> EvidenceGatedSupportClassifier:
    spec = AttentionSpec(d_model=D_MODEL, n_heads=4, ffn_mult=2, dropout=0.0)
    head = EvidenceGatedSupportClassifier(spec, EvidenceGatedClassifierConfig(
        text_dim=TEXT_DIM, semantic_mode="text+primitives", primitive_label_side="annotated",
        primitive_combiner="bilinear", **overrides,
    )).eval()
    with torch.no_grad():
        head.primitive_head.anchors.copy_(fake_anchors())
    return head


def test_mixture_never_falls_below_half_the_better_half():
    # Adversarial large roster: clamp-then-renormalize used to violate this bound here.
    candidates = 256
    first = torch.full((1, candidates), -1000.0)
    second = torch.full((1, candidates), -torch.log(torch.tensor(float(candidates))))
    first[:, 0] = 0.0
    mask = torch.ones_like(first, dtype=torch.bool)
    mixed = probability_semantic_mixture(first, second, mask)
    better = torch.maximum(first, second)
    assert bool((mixed >= better - torch.log(torch.tensor(2.0)) - 1e-6).all())
    torch.testing.assert_close(mixed.exp().sum(-1), torch.ones(1))


def test_mixture_smoothing_is_total_mass_and_remains_normalized():
    first = torch.log_softmax(torch.tensor([[20.0, -20.0, -20.0]]), dim=-1)
    second = torch.log_softmax(torch.tensor([[-20.0, 20.0, -20.0]]), dim=-1)
    mask = torch.ones_like(first, dtype=torch.bool)
    epsilon = 1e-2
    mixed = probability_semantic_mixture(first, second, mask, smoothing=epsilon).exp()
    torch.testing.assert_close(mixed.sum(-1), torch.ones(1))
    expected = (1.0 - epsilon) * 0.5 * (first.exp() + second.exp()) + epsilon / 3.0
    torch.testing.assert_close(mixed, expected)


def test_mixture_has_finite_backward_with_padded_candidates():
    first = torch.tensor([[0.0, -1.0, float("-inf")]], requires_grad=True)
    second = torch.tensor([[-2.0, 0.0, float("-inf")]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    mixed = probability_semantic_mixture(first, second, mask)
    (-mixed[:, 0]).backward()
    assert bool(torch.isfinite(first.grad).all())
    assert bool(torch.isfinite(second.grad).all())


def test_log_sum_default_is_unchanged_for_t7_checkpoints():
    from tests.test_evidence_gated_classifier import make_episode

    head = v4()  # default combination is log_sum, default label side sentences overridden above
    assert head.cfg.semantic_combination == "log_sum"
    batch = make_episode(seed=14)
    out = head(**batch)
    valid = batch["candidate_mask"]
    expected = torch.log_softmax(
        (out["text_logits"] + out["primitive_logits"]).masked_fill(~valid, float("-inf")), dim=-1,
    )
    torch.testing.assert_close(out["semantic_logits"], expected)


def test_gate_only_still_isolates_the_blend_gate_with_the_annotated_side():
    from tests.test_evidence_gated_classifier import make_episode

    head = v4(semantic_combination="mixture", unenrolled_calibration=True)
    batch = make_episode(seed=15)
    out = head(**batch, gate_only=True)
    valid = batch["candidate_mask"]
    F.cross_entropy(out["logits"].masked_fill(~valid, -1e30), torch.tensor([0, 1, 0])).backward()
    live = {name for name, p in head.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert live and all(name.startswith(("gate_mlp", "gate_norm", "lambda_prior")) for name in live), sorted(live)


def test_checkpoint_round_trip_and_try_name():
    head = v4(semantic_combination="mixture", unenrolled_calibration=True)
    blob = {"architecture_version": ARCHITECTURE_VERSION, "classifier": head.state_dict(),
            "classifier_config": dataclasses.asdict(head.cfg), "attention_spec": dataclasses.asdict(head.spec)}
    restored, _ = build_classifier_from_blob(blob)
    assert torch.equal(restored.primitive_head.anchors, head.primitive_head.anchors)
    assert torch.equal(restored.primitive_head.anchor_profiles, head.primitive_head.anchor_profiles)
    assert classifier_try_name(ARCHITECTURE_VERSION, trajectory={
        "text_corruption_mode": "auxiliary", "text_corruption_probability": 1.0,
        "unenrolled_calibration": True, "semantic_mode": "text+primitives",
        "primitive_label_side": "annotated", "primitive_combiner": "bilinear",
        "primitive_annotations": "annotations-v1", "semantic_combination": "mixture",
    }) == "T8"
    assert classifier_try_name(ARCHITECTURE_VERSION, trajectory={
        "text_corruption_mode": "auxiliary", "text_corruption_probability": 1.0,
        "unenrolled_calibration": True, "semantic_mode": "text+primitives",
        "primitive_label_side": "annotated", "primitive_combiner": "bilinear",
        "primitive_annotations": "annotations-v1-scrambled", "semantic_combination": "mixture",
    }) == "T8-scrambled-control"


def test_checkpoint_restore_uses_persisted_anchors_without_text_encoder(monkeypatch):
    head = v4(semantic_combination="mixture", unenrolled_calibration=True)
    blob = {"architecture_version": ARCHITECTURE_VERSION, "classifier": head.state_dict(),
            "classifier_config": dataclasses.asdict(head.cfg),
            "attention_spec": dataclasses.asdict(head.spec),
            "primitive_provenance": head.primitive_head.provenance}

    def forbidden(_labels):
        raise AssertionError("checkpoint restore attempted to re-encode primitive anchors")

    monkeypatch.setattr("model.support.primitive_semantics._anchor_text_matrix", forbidden)
    restored, _ = build_classifier_from_blob(blob)
    torch.testing.assert_close(restored.primitive_head.anchors, head.primitive_head.anchors)
