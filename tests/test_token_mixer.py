"""Contracts for HALO's active support-conditioned semantic classifier."""

from __future__ import annotations

import torch

from model.blocks import AttentionSpec
from model.support.token_mixer import SupportTokenMixer, TokenMixerConfig


def _inputs(batch: int = 2, supports: int = 3, candidates: int = 3):
    torch.manual_seed(7)
    return {
        "is_zero_shot": torch.tensor([False, True][:batch], dtype=torch.bool),
        "query_feature": torch.randn(batch, 8),
        "support_feature": torch.randn(batch, supports, 8),
        "support_label_text": torch.randn(batch, supports, 4),
        "support_bound": torch.tensor([[0, 1, 2], [0, 1, 2]])[:batch, :supports],
        "support_mask": torch.ones(batch, supports, dtype=torch.bool),
        "support_pair_slot": torch.tensor([[2, 1, 3], [1, 3, 2]])[:batch, :supports],
        "candidate_text": torch.randn(batch, candidates, 4),
        "candidate_mask": torch.ones(batch, candidates, dtype=torch.bool),
        "candidate_slot": torch.tensor([[3, 1, 2], [2, 3, 1]])[:batch, :candidates],
    }


def _mixer() -> SupportTokenMixer:
    return SupportTokenMixer(
        AttentionSpec(d_model=8, n_heads=2, ffn_mult=2, dropout=0.0),
        TokenMixerConfig(text_dim=4, n_layers=1, max_candidates=8, max_supports=8),
    )


def test_mixed_zero_and_few_rows_have_finite_logits_and_gradients():
    mixer = _mixer()
    inputs = _inputs()
    output = mixer(**inputs)
    assert output["logits"].shape == (2, 3)
    assert torch.isfinite(output["logits"]).all()
    assert output["support_weight"] is not None
    output["logits"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in mixer.zero_shot_head.parameters())
    assert any(parameter.grad is not None for parameter in mixer.few_shot_head.parameters())


def test_few_shot_token_order_is_equivariant_when_identity_tags_move_with_tokens():
    mixer = _mixer().eval()
    inputs = _inputs(batch=1)
    inputs["is_zero_shot"] = torch.tensor([False])
    original = mixer(**inputs)["logits"]

    candidate_perm = torch.tensor([2, 0, 1])
    support_perm = torch.tensor([1, 2, 0])
    inverse = torch.argsort(candidate_perm)
    permuted = {key: value.clone() if torch.is_tensor(value) else value for key, value in inputs.items()}
    for key in ("candidate_text", "candidate_mask", "candidate_slot"):
        permuted[key] = permuted[key][:, candidate_perm]
    for key in ("support_feature", "support_label_text", "support_mask", "support_pair_slot"):
        permuted[key] = permuted[key][:, support_perm]
    # Candidate indices are vote destinations, so remap them after candidate permutation and then
    # move the support rows as one bound pair.
    old_to_new = inverse
    permuted["support_bound"] = old_to_new[inputs["support_bound"][:, support_perm]]
    rearranged = mixer(**permuted)["logits"][:, inverse]
    torch.testing.assert_close(original, rearranged, atol=1e-5, rtol=1e-5)


def test_padded_support_rows_do_not_change_valid_prediction():
    mixer = _mixer().eval()
    inputs = _inputs(batch=1, supports=3)
    inputs["is_zero_shot"] = torch.tensor([False])
    original = mixer(**inputs)["logits"]
    padded = {key: value.clone() if torch.is_tensor(value) else value for key, value in inputs.items()}
    for key in ("support_feature", "support_label_text"):
        padded[key] = torch.cat((padded[key], torch.randn(1, 2, padded[key].shape[-1])), dim=1)
    padded["support_bound"] = torch.cat((padded["support_bound"], torch.zeros(1, 2, dtype=torch.long)), dim=1)
    padded["support_pair_slot"] = torch.cat((padded["support_pair_slot"], torch.zeros(1, 2, dtype=torch.long)), dim=1)
    padded["support_mask"] = torch.cat((padded["support_mask"], torch.zeros(1, 2, dtype=torch.bool)), dim=1)
    torch.testing.assert_close(original, mixer(**padded)["logits"], atol=1e-5, rtol=1e-5)
