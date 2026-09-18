"""Joint device-set planning for support-classifier episodes.

The encoder may compose aligned devices, but choosing every recording independently cannot teach
the classifier what it means for a query to have more, fewer, or overlapping devices than its
enrolment set.  This module plans those relationships once per support set without examining
labels or signal values.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .sampling import Episode, SupportCorpus

DeviceRelation = Literal[
    "not_applicable", "matched_single", "matched_composite", "query_superset",
    "support_superset", "partial_overlap", "disjoint",
]

RELATIONS: tuple[DeviceRelation, ...] = (
    "matched_single", "matched_composite", "query_superset", "support_superset",
    "partial_overlap", "disjoint",
)


@dataclass(frozen=True)
class DeviceSetPlan:
    """A relation and the chosen stream names for one support set."""

    relation: DeviceRelation
    query_devices: tuple[str, ...] = ()
    support_devices: tuple[str, ...] = ()
    fallback: bool = False

    @property
    def jaccard(self) -> float:
        query, support = set(self.query_devices), set(self.support_devices)
        return len(query & support) / max(1, len(query | support))


def _episode_groups(episodes: list[Episode]) -> list[list[Episode]]:
    """Group all queries that reuse the same physical support executions."""
    grouped: dict[tuple[tuple[int, ...], ...], list[Episode]] = defaultdict(list)
    for episode in episodes:
        groups = episode.support_window_groups or tuple((index,) for index in episode.support)
        grouped[tuple(groups)].append(episode)
    return list(grouped.values())


def _common_streams(dataset, positions: list[int]) -> tuple[str, ...]:
    if not positions:
        return ()
    members = [dataset.aligned_device_members(position) for position in positions]
    if not all(members):
        return ()
    common = set(members[0])
    for value in members[1:]:
        common.intersection_update(value)
    return tuple(sorted(common))


def _select_relation(devices: tuple[str, ...], relation: DeviceRelation,
                     rng: np.random.Generator) -> DeviceSetPlan | None:
    """Draw balanced, role-level subsets from streams shared by every selected row."""
    n = len(devices)
    if n < 2:
        return None
    order = list(rng.permutation(np.asarray(devices, dtype=object)).tolist())
    if relation == "matched_single":
        chosen = (order[0],)
        return DeviceSetPlan(relation, chosen, chosen)
    if relation == "matched_composite":
        if n < 2:
            return None
        count = int(rng.integers(2, n + 1))
        chosen = tuple(sorted(order[:count]))
        return DeviceSetPlan(relation, chosen, chosen)
    if relation == "query_superset":
        if n < 2:
            return None
        support_count = int(rng.integers(1, n))
        support = tuple(sorted(order[:support_count]))
        query = tuple(sorted(order[:support_count + 1]))
        return DeviceSetPlan(relation, query, support)
    if relation == "support_superset":
        if n < 2:
            return None
        query_count = int(rng.integers(1, n))
        query = tuple(sorted(order[:query_count]))
        support = tuple(sorted(order[:query_count + 1]))
        return DeviceSetPlan(relation, query, support)
    if relation == "partial_overlap":
        if n < 3:
            return None
        return DeviceSetPlan(relation, tuple(sorted((order[0], order[1]))),
                             tuple(sorted((order[1], order[2]))))
    if relation == "disjoint":
        return DeviceSetPlan(relation, (order[0],), (order[1],))
    raise ValueError(f"unknown device-set relation {relation!r}")


def plan_device_sets(
    episodes: list[Episode], corpus: SupportCorpus, dataset, rng: np.random.Generator,
    *, probability: float,
) -> tuple[dict[int, tuple[int, ...]], dict[int, DeviceSetPlan]]:
    """Return ``recording-index -> aligned members`` and per-episode relation telemetry.

    A plan is only legal when every query and every support recording of the set belongs to one
    source dataset and exposes the necessary aligned streams.  This preserves label balance,
    subject/execution exclusions, and the episode sampler's acquisition contract.  Ineligible
    sets retain ordinary independent composition and are explicitly marked ``not_applicable``.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("device-set challenge probability must be in [0, 1]")
    composition: dict[int, tuple[int, ...]] = {}
    per_episode: dict[int, DeviceSetPlan] = {}
    for grouped in _episode_groups(episodes):
        queries = [episode.query for episode in grouped]
        supports = [index for episode in grouped
                    for group in (episode.support_window_groups or tuple((i,) for i in episode.support))
                    for index in group]
        all_indices = [*queries, *supports]
        records = [corpus.recordings[index] for index in all_indices]
        if rng.random() >= probability or len({record.dataset for record in records}) != 1:
            plan = DeviceSetPlan("not_applicable")
        else:
            query_positions = [corpus.recordings[index].window_index for index in queries]
            support_positions = [corpus.recordings[index].window_index for index in supports]
            available = tuple(sorted(set(_common_streams(dataset, query_positions))
                                     & set(_common_streams(dataset, support_positions))))
            choices = list(RELATIONS)
            rng.shuffle(choices)
            plan = next((candidate for name in choices
                         if (candidate := _select_relation(available, name, rng)) is not None),
                        DeviceSetPlan("not_applicable", fallback=bool(available)))
        proposed: dict[int, tuple[int, ...]] = {}
        if plan.relation != "not_applicable":
            for index in queries:
                members = dataset.aligned_device_members(corpus.recordings[index].window_index)
                proposed[index] = tuple(members[name] for name in plan.query_devices)
            for index in supports:
                members = dataset.aligned_device_members(corpus.recordings[index].window_index)
                proposed[index] = tuple(members[name] for name in plan.support_devices)
            if any(index in composition and composition[index] != members
                   for index, members in proposed.items()):
                plan = DeviceSetPlan("not_applicable", fallback=True)
                proposed = {}
        for episode in grouped:
            per_episode[id(episode)] = plan
        if plan.relation == "not_applicable":
            continue
        composition.update(proposed)
        # This loop is intentionally only an invariant check; all selected members were built
        # above before any map was mutated, so overlapping support sets cannot be half-planned.
        for index in queries:
            members = dataset.aligned_device_members(corpus.recordings[index].window_index)
            if composition[index] != tuple(members[name] for name in plan.query_devices):
                raise RuntimeError("joint query device-set plan was not preserved")
        for index in supports:
            members = dataset.aligned_device_members(corpus.recordings[index].window_index)
            if composition[index] != tuple(members[name] for name in plan.support_devices):
                raise RuntimeError("joint support device-set plan was not preserved")
    return composition, per_episode
