"""Contracts for joint query/support device-set composition."""

from __future__ import annotations

import numpy as np

from training.support_classifier.device_sets import plan_device_sets
from training.support_classifier.sampling import Episode, Recording, SupportCorpus


class _Dataset:
    def __init__(self):
        self.members = {
            position: {name: position * 10 + offset for offset, name in enumerate(("a", "b", "c"))}
            for position in range(4)
        }

    def aligned_device_members(self, position):
        return self.members[position]


def _episode(query: int) -> Episode:
    return Episode(
        query=query, support=(2, 3), support_candidate=(0, 1),
        candidates=("walk", "run"), gt_slot=0, mode="compatible", requested_support=2,
        shrunk=False, support_window_groups=((2,), (3,)), support_set_id=7,
    )


def test_joint_plan_uses_one_relation_for_all_queries_and_all_support_rows():
    corpus = SupportCorpus(
        recordings=[Recording(0, pos, "dsads", "stream", "walk", "s", f"e{pos}")
                    for pos in range(4)],
        keys=[], stream_names=[],
    )
    episodes = [_episode(0), _episode(1)]
    composition, plans = plan_device_sets(
        episodes, corpus, _Dataset(), np.random.default_rng(7), probability=1.0,
    )
    first, second = plans[id(episodes[0])], plans[id(episodes[1])]
    assert first == second
    assert first.relation != "not_applicable"
    assert composition[0] != () and composition[1] != ()
    assert len(composition[0]) == len(first.query_devices)
    assert len(composition[2]) == len(first.support_devices)
    # Each event uses different dataset positions, but it must select the same named stream set.
    assert tuple(value % 10 for value in composition[2]) == tuple(value % 10 for value in composition[3])


def test_ineligible_cross_dataset_set_is_explicitly_not_applicable():
    corpus = SupportCorpus(
        recordings=[
            Recording(0, 0, "dsads", "stream", "walk", "s", "e0"),
            Recording(0, 1, "other", "stream", "walk", "s", "e1"),
            Recording(0, 2, "dsads", "stream", "run", "s", "e2"),
        ], keys=[], stream_names=[],
    )
    episode = Episode(0, (1, 2), (0, 1), ("walk", "run"), 0, "cross_dataset", 2, False,
                      support_window_groups=((1,), (2,)))
    composition, plans = plan_device_sets(
        [episode], corpus, _Dataset(), np.random.default_rng(1), probability=1.0,
    )
    assert not composition
    assert plans[id(episode)].relation == "not_applicable"


def test_counterfactual_enrollment_views_share_one_device_intervention():
    corpus = SupportCorpus(
        recordings=[Recording(0, pos, "dsads", "stream", "walk", "s", f"e{pos}")
                    for pos in range(4)],
        keys=[], stream_names=[],
    )
    complete = _episode(0)
    complete = Episode(**{**complete.__dict__, "counterfactual_group": 19})
    partial = Episode(**{
        **complete.__dict__, "support": (2,), "support_candidate": (0,),
        "support_window_groups": ((2,),), "counterfactual_view": "partial",
    })
    zero = Episode(**{
        **complete.__dict__, "support": (), "support_candidate": (),
        "support_window_groups": (), "zero_shot": True, "counterfactual_view": "zero",
    })
    episodes = [complete, partial, zero]
    composition, plans = plan_device_sets(
        episodes, corpus, _Dataset(), np.random.default_rng(7), probability=1.0,
    )
    assert plans[id(complete)] == plans[id(partial)] == plans[id(zero)]
    assert plans[id(complete)].relation != "not_applicable"
    assert 0 in composition and 2 in composition and 3 in composition
