"""Unit contracts for the shared parameter-free neighbour control."""

import torch

from training.support_classifier.neighbors import differentiable_neighbor_logits
from training.support_classifier.representation_diagnostics import embedding_summary


def test_neighbour_vote_is_normalized_and_prefers_matching_support():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    support = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    bindings = torch.tensor([[0, 1], [0, 1]])
    mask = torch.ones((2, 2), dtype=torch.bool)
    candidates = torch.ones((2, 2), dtype=torch.bool)
    logits, weights = differentiable_neighbor_logits(query, support, bindings, mask, candidates)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert logits.argmax(dim=1).tolist() == [0, 1]


def test_neighbour_vote_rejects_zero_support_and_invalid_binding():
    query = torch.ones((1, 2))
    support = torch.ones((1, 1, 2))
    candidates = torch.ones((1, 2), dtype=torch.bool)
    try:
        differentiable_neighbor_logits(query, support, torch.tensor([[0]]),
                                       torch.tensor([[False]]), candidates)
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("zero-support episode was accepted")
    try:
        differentiable_neighbor_logits(query, support, torch.tensor([[2]]),
                                       torch.tensor([[True]]), candidates)
    except ValueError as error:
        assert "invalid candidate" in str(error)
    else:
        raise AssertionError("invalid candidate binding was accepted")


def test_embedding_summary_handles_embedding_dimension_unrelated_to_label_count():
    features = torch.eye(5)[:4].numpy()
    summary, _ = embedding_summary(features, ["a", "a", "b", "b"], max_points=4, pair_samples=4)
    assert summary["dimension"] == 5
    assert summary["n_labels"] == 2
