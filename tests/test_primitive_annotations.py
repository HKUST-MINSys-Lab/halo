"""Contract tests for the annotated label side of the primitive path (T8).

What must hold: the annotations cover the supervised training vocabulary exactly and nothing
else; every value is a value of its axis; the frozen map returns one distribution per axis and
recovers an annotated label's own profile; the mixture combination can never fall below half the
better half; and the defaults leave T7 checkpoints unchanged.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from model.blocks import AttentionSpec
from model.support.evidence_gated_classifier import (
    ARCHITECTURE_VERSION, EvidenceGatedClassifierConfig, EvidenceGatedSupportClassifier,
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
    """Training labels only: no sealed label may appear, and none may be missing."""
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
    assert torch.equal(head.interaction, torch.eye(n_primitives()))
    query = torch.randn(3, D_MODEL)
    text = head.anchors[:5].unsqueeze(0).expand(3, -1, -1)
    out = head(query, text, torch.ones(3, 5, dtype=torch.bool))
    F.nll_loss(out["logits"], torch.tensor([0, 1, 2])).backward()
    assert float(head.interaction.grad.abs().sum()) > 0
    assert float(head.keys.weight.grad.abs().sum()) > 0


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
    from tests.test_evidence_gated_classifier import make_episode

    head = v4(semantic_combination="mixture", semantic_floor=1e-3)
    batch = make_episode(seed=12)
    out = head(**batch)
    valid = batch["candidate_mask"]
    better = torch.maximum(out["text_logits"], out["primitive_logits"])
    # log(0.5 * max(p_text, p_prim)) is a lower bound on the mixture before renormalisation, and
    # renormalisation can only raise a probability that was below its share.
    assert bool((out["semantic_logits"][valid] >= (better[valid] + torch.log(torch.tensor(0.5))) - 1e-5).all())
    assert bool((out["semantic_logits"][valid].exp().sum(-1) if out["semantic_logits"].ndim == 1
                 else out["semantic_logits"].masked_fill(~valid, -1e30).exp().sum(-1) - 1).abs().max() < 1e-4)


def test_mixture_floor_bounds_how_wrong_a_half_can_be():
    from tests.test_evidence_gated_classifier import make_episode

    head = v4(semantic_combination="mixture", semantic_floor=1e-2)
    batch = make_episode(seed=13)
    out = head(**batch)
    valid = batch["candidate_mask"]
    # No valid candidate can be driven below the floor's share of the mixture.
    assert float(out["semantic_logits"].detach()[valid].exp().min()) >= 0.5 * 1e-2 / 2


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
        "primitive_label_side": "annotated",
    }) == "T8"
