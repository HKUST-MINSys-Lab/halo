"""Contract tests for the primitive-driven semantic path.

The properties that matter are the ones that keep this from becoming the collapse the contextual
lineage produced: the value bank is frozen, the compatibility projection is shared between
primitive sentences and candidate labels (so it cannot single out a vocabulary), it is identity-
initialised, and a uniform sensor axis is neutral across candidates.
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
from model.support.factory import build_classifier_from_blob
from model.support.primitive_semantics import (
    PRIMITIVE_VOCABULARY_VERSION, SCRAMBLED_VERSION, VOCABULARIES, PrimitiveSemanticConfig,
    PrimitiveSemanticHead, axis_names, axis_sizes, axis_slices, n_primitives, primitive_names,
    primitive_sentences, vocabulary_hash,
)

D_MODEL, TEXT_DIM = 32, 384


def fake_values(version: str = PRIMITIVE_VOCABULARY_VERSION) -> torch.Tensor:
    """A deterministic stand-in for the frozen sentence encoder, so tests need no model."""
    generator = torch.Generator().manual_seed(11)
    return F.normalize(torch.randn(n_primitives(version), TEXT_DIM, generator=generator), dim=-1)


def make_head(**overrides) -> PrimitiveSemanticHead:
    cfg = PrimitiveSemanticConfig(text_dim=TEXT_DIM, **overrides)
    return PrimitiveSemanticHead(D_MODEL, cfg, values=fake_values(cfg.version)).eval()


def make_batch(b=3, c=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(b, D_MODEL, generator=generator)
    text = F.normalize(torch.randn(b, c, TEXT_DIM, generator=generator), dim=-1)
    mask = torch.ones(b, c, dtype=torch.bool)
    mask[:, -1] = False
    return query, text * mask.unsqueeze(-1), mask


# --------------------------------------------------------------------- vocabulary
def test_vocabulary_is_well_formed_and_hashed():
    assert n_primitives() == 32 and len(axis_names()) == 10
    assert sum(axis_sizes()) == n_primitives()
    assert len(set(primitive_names())) == n_primitives()
    sentences = primitive_sentences()
    assert len(set(sentences)) == len(sentences), "a sentence is repeated"
    assert all(sentence.endswith(".") for sentence in sentences)
    assert vocabulary_hash() != vocabulary_hash(SCRAMBLED_VERSION)


def test_vocabulary_names_no_activity_from_our_corpora():
    """The vocabulary must not encode the evaluation's label set.

    Naming activities lifts label-side agreement (52% -> 78% in the 2026-09-20 audit) precisely
    because it smuggles label-to-label similarity in, which would make a zero-shot gain an
    artefact of our word choice. Checked against a fixed list of activity nouns rather than the
    live corpus so the test cannot start passing because a roster changed.
    """
    forbidden = ("walking", "running", "jogging", "sitting", "standing", "lying", "cycling",
                 "biking", "stairs", "jumping", "typing", "vacuuming", "sweeping", "rowing",
                 "squat", "burpee", "push-up", "pushup", "elevator", "smoking", "eating",
                 "drinking", "sleeping", "talking", "climbing", "hiking", "stomping", "shaking")
    import re

    # Whole words only: "repeating" contains "eating", and a substring match would forbid
    # ordinary English rather than activity names.
    for version in VOCABULARIES:
        for sentence in primitive_sentences(version):
            lowered = sentence.lower()
            hits = [word for word in forbidden
                    if re.search(rf"\b{re.escape(word)}\b", lowered)]
            assert not hits, f"{version}: {hits} named in {sentence!r}"


def test_scrambled_control_is_a_permutation_of_the_same_sentences():
    assert sorted(primitive_sentences()) == sorted(primitive_sentences(SCRAMBLED_VERSION))
    assert primitive_sentences() != primitive_sentences(SCRAMBLED_VERSION)
    assert axis_names() == axis_names(SCRAMBLED_VERSION)


# --------------------------------------------------------------------- head structure
def test_value_bank_is_frozen():
    """A trainable value bank drifts into a lookup over the training vocabulary."""
    head = make_head()
    assert "values" in dict(head.named_buffers())
    assert not any(name.startswith("values") for name, _ in head.named_parameters())


def test_fixed_combiner_is_genuinely_fixed_and_bounded():
    head = make_head(combiner="fixed")
    assert "axis_weight" not in dict(head.named_parameters())
    query, text, mask = make_batch(seed=12)
    agreement = head.agreement(head.profile(query), head.candidate_profile(text, mask))
    assert bool((agreement >= 0).all()) and bool((agreement <= 1).all())


def test_profiles_are_per_axis_distributions():
    head = make_head()
    query, text, mask = make_batch()
    out = head(query, text, mask)
    for block in axis_slices():
        torch.testing.assert_close(
            out["primitive_profile"][:, block].sum(-1), torch.ones(query.shape[0]),
        )
        valid = out["candidate_primitive_profile"][mask][:, block].sum(-1)
        torch.testing.assert_close(valid, torch.ones_like(valid))
    assert torch.allclose(
        out["logits"].masked_fill(~mask, -1e30).exp().sum(-1), torch.ones(query.shape[0]),
        atol=1e-5,
    )


def test_projection_is_identity_initialised():
    """Step 0 of the learnable compatibility is exactly the fixed cosine design."""
    learned = make_head(combiner="projection")
    fixed = make_head(combiner="fixed")
    with torch.no_grad():  # the fixed combiner's axis weights start at 1, so both reduce to cosine
        query, text, mask = make_batch(seed=4)
        a = learned._compatibility(text)
        b = fixed._compatibility(text)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_uniform_axis_is_neutral_across_candidates():
    """The implied observability mask: an axis the head cannot resolve must not tilt the ranking.

    This is what replaces a declared mask table. Set one axis to uniform and the candidate
    ranking must be unchanged from dropping that axis entirely.
    """
    head = make_head(combiner="projection")
    query, text, mask = make_batch(seed=5)
    profile = head.profile(query)
    candidate = head.candidate_profile(text, mask)
    block = axis_slices()[-1]  # the head axis
    uniform = profile.clone()
    uniform[:, block] = 1.0 / (block.stop - block.start)
    full = head.agreement(uniform, candidate)
    without = head.agreement(uniform, candidate) - torch.einsum(
        "bv,bcv->bc", uniform[:, block], candidate[..., block],
    )
    difference = full - without
    # Constant across the valid candidates of a row, therefore invisible after the softmax.
    high = difference.masked_fill(~mask, float("-inf")).amax(dim=1)
    low = difference.masked_fill(~mask, float("inf")).amin(dim=1)
    assert float((high - low).abs().max().detach()) < 1e-6


def test_label_side_is_a_pure_function_of_the_label_embedding():
    head = make_head()
    query, text, mask = make_batch(seed=6)
    first = head.candidate_profile(text, mask)
    shuffled = torch.roll(text, shifts=2, dims=1)
    second = head.candidate_profile(shuffled, mask)
    torch.testing.assert_close(
        first.roll(2, dims=1)[mask.roll(2, dims=1) & mask],
        second[mask.roll(2, dims=1) & mask],
    )


def test_gradients_reach_keys_and_projection_but_not_values():
    head = make_head(combiner="projection")
    query, text, mask = make_batch(seed=7)
    out = head(query, text, mask)
    F.nll_loss(out["logits"], torch.tensor([0, 1, 2])).backward()
    assert float(head.keys.weight.grad.abs().sum()) > 0
    assert float(head.projection.grad.abs().sum()) > 0
    assert head.values.grad is None


# --------------------------------------------------------------------- integration with v4
def v4(mode: str) -> EvidenceGatedSupportClassifier:
    spec = AttentionSpec(d_model=D_MODEL, n_heads=4, ffn_mult=2, dropout=0.0)
    head = EvidenceGatedSupportClassifier(
        spec, EvidenceGatedClassifierConfig(text_dim=TEXT_DIM, semantic_mode=mode),
    ).eval()
    if head.primitive_head is not None:
        with torch.no_grad():
            head.primitive_head.values.copy_(fake_values())
    return head


def v4_batch(seed=0):
    from tests.test_evidence_gated_classifier import make_episode

    return make_episode(seed=seed)


def test_default_mode_is_text_and_constructs_no_primitive_head():
    head = v4("text")
    assert head.primitive_head is None
    out = head(**v4_batch())
    assert "primitive_logits" not in out
    torch.testing.assert_close(out["semantic_logits"], out["text_logits"])


@pytest.mark.parametrize("mode", ["primitives", "text+primitives"])
def test_primitive_modes_produce_finite_blended_logits(mode):
    head = v4(mode)
    batch = v4_batch(seed=1)
    out = head(**batch)
    valid = batch["candidate_mask"]
    assert torch.isfinite(out["logits"].masked_select(valid)).all()
    assert out["primitive_profile"].shape[-1] == n_primitives()
    if mode == "primitives":
        torch.testing.assert_close(out["semantic_logits"], out["primitive_logits"])
    else:
        assert not torch.allclose(out["semantic_logits"], out["text_logits"])
        assert not torch.allclose(out["semantic_logits"], out["primitive_logits"])


def test_semantic_combination_has_no_learned_weight():
    """text+primitives is a FIXED equal-weight sum in log space; a learned weight is the router."""
    head = v4("text+primitives")
    names = [name for name, _ in head.named_parameters()]
    assert not any("semantic_weight" in name or "semantic_mix" in name for name in names)
    batch = v4_batch(seed=2)
    out = head(**batch)
    valid = batch["candidate_mask"]
    expected = torch.log_softmax(
        (out["text_logits"] + out["primitive_logits"]).masked_fill(~valid, float("-inf")), dim=-1,
    )
    torch.testing.assert_close(out["semantic_logits"], expected)


def test_corruption_stop_gradient_covers_both_semantic_halves():
    head = v4("text+primitives")
    batch = v4_batch(seed=3)
    batch["query_feature"] = batch["query_feature"].requires_grad_(True)
    stop = torch.tensor([True, False, False])
    out = head(**batch, text_stop_gradient=stop)
    out["semantic_logits"].sum().backward()
    grad = batch["query_feature"].grad
    assert float(grad[0].abs().sum()) == 0.0, "a corrupted row still trained the semantic path"
    assert float(grad[1].abs().sum()) > 0.0


def test_branch_logits_reports_each_semantic_half():
    head = v4("text+primitives")
    out = head(**v4_batch(seed=8))
    assert torch.equal(head.branch_logits(out, "semantic_text"), out["text_logits"])
    assert torch.equal(head.branch_logits(out, "semantic_primitives"), out["primitive_logits"])
    assert torch.equal(head.branch_logits(out, "semantic"), out["semantic_logits"])
    text_only = v4("text")
    with pytest.raises(ValueError, match="not available"):
        text_only.branch_logits(text_only(**v4_batch(seed=9)), "semantic_primitives")


def test_checkpoint_round_trip_carries_the_vocabulary():
    head = v4("text+primitives")
    blob = {
        "architecture_version": ARCHITECTURE_VERSION,
        "classifier": head.state_dict(),
        "classifier_config": dataclasses.asdict(head.cfg),
        "attention_spec": dataclasses.asdict(head.spec),
        "primitive_provenance": head.primitive_head.provenance,
    }
    restored, _ = build_classifier_from_blob(blob)
    assert restored.cfg.semantic_mode == "text+primitives"
    # The frozen value bank travels in the state dict, so an evaluator never re-encodes it.
    assert torch.equal(restored.primitive_head.values, head.primitive_head.values)
    batch = v4_batch(seed=10)
    with torch.no_grad():
        torch.testing.assert_close(head(**batch)["logits"], restored(**batch)["logits"])
    blob["primitive_provenance"] = {
        **blob["primitive_provenance"], "primitive_vocabulary_hash": "mismatch",
    }
    with pytest.raises(ValueError, match="provenance"):
        build_classifier_from_blob(blob)


def test_primitive_head_sanitizes_nonfinite_padded_candidate_text():
    head = make_head()
    query, text, mask = make_batch(seed=13)
    query.requires_grad_()
    text[~mask] = float("nan")
    output = head(query, text, mask)
    torch.nn.functional.nll_loss(output["logits"], torch.tensor([0, 1, 2])).backward()
    assert torch.isfinite(query.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in head.parameters())


def test_scrambled_vocabulary_is_selectable_and_changes_scores():
    spec = AttentionSpec(d_model=D_MODEL, n_heads=4, ffn_mult=2, dropout=0.0)
    batch = v4_batch(seed=11)
    outputs = []
    for version in (PRIMITIVE_VOCABULARY_VERSION, SCRAMBLED_VERSION):
        head = EvidenceGatedSupportClassifier(spec, EvidenceGatedClassifierConfig(
            text_dim=TEXT_DIM, semantic_mode="primitives", primitive_version=version,
        )).eval()
        with torch.no_grad():
            head.primitive_head.values.copy_(fake_values(version))
        outputs.append(head(**batch)["primitive_logits"])
    assert not torch.allclose(outputs[0], outputs[1])
