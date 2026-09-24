from __future__ import annotations

import torch

from training.evidence.eval_enrollment import (
    _prediction_contingency,
    _restricted_view,
    _zero_module_output,
)
from training.evidence.patch_episodes import EpisodeMemoryView


def _view() -> EpisodeMemoryView:
    return EpisodeMemoryView(
        allowed=torch.tensor([[[True, True, False], [False, False, False]]]),
        support_mask=torch.tensor([True, False, False]),
        support_candidate=torch.tensor([0, -1, -1]),
        candidate_ids=torch.tensor([3, 5]),
        query_label=torch.tensor([3]),
        support_units_per_candidate=torch.tensor([1, 0]),
        episode_type="ordinary_few_support",
        label_mode="coherent",
    )


def test_restricted_view_changes_only_the_allowed_roster():
    view = _view()
    restricted = _restricted_view(view, view.support_mask)
    assert torch.equal(restricted.allowed, torch.tensor([[[True, False, False], [False, False, False]]]))
    assert restricted.support_mask is view.support_mask
    assert restricted.candidate_ids is view.candidate_ids


def test_prediction_contingency_is_paired_and_exhaustive():
    result = _prediction_contingency(
        torch.tensor([0, 1, 2, 0]),
        torch.tensor([0, 2, 1, 1]),
        torch.tensor([0, 1, 1, 2]),
    )
    assert result == {
        "both_correct": 1,
        "decoder_only_correct": 1,
        "identity_only_correct": 1,
        "both_wrong": 1,
        "prediction_changed": 3,
        "queries": 4,
    }


def test_zero_module_output_is_scoped_and_restores_forward():
    layer = torch.nn.Linear(3, 2)
    value = torch.ones(4, 3)
    expected = layer(value)
    with _zero_module_output(layer):
        assert torch.equal(layer(value), torch.zeros(4, 2))
    assert torch.equal(layer(value), expected)
