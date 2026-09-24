"""Patch-bank indexing, leakage masks, and evidence assembly for Phase-B episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from model.evidence.patch_retrieval import PatchRetrieval

# Shared empty result for labels with no eligible windows under a memory mask.
_EMPTY_LONG = torch.empty(0, dtype=torch.long)


@dataclass
class QueryPatches:
    Z: torch.Tensor
    mask: torch.Tensor
    time: torch.Tensor
    duration: torch.Tensor
    resolution: torch.Tensor
    window: torch.Tensor
    event: torch.Tensor
    subj: torch.Tensor
    cfg: torch.Tensor
    sensor: torch.Tensor
    row: torch.Tensor


@dataclass
class EvidenceBatch:
    index: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor
    mask: torch.Tensor
    head: torch.Tensor
    query_patch: torch.Tensor
    local_index: torch.Tensor | None = None


@dataclass
class EpisodeMemoryView:
    """One batch-level overlay on the immutable active memory index."""

    allowed: torch.Tensor
    support_mask: torch.Tensor
    support_candidate: torch.Tensor
    candidate_ids: torch.Tensor
    query_label: torch.Tensor
    support_units_per_candidate: torch.Tensor
    episode_type: str
    label_mode: str

    @property
    def support_rows(self) -> torch.Tensor:
        return torch.nonzero(self.support_mask, as_tuple=True)[0]


class PatchTable:
    """CPU-resident patch metadata with efficient parent-window and label row lookup."""

    def __init__(self, bank: dict):
        self.patch = bank["patch"]
        self.n_windows = len(bank["Z"])
        self.window_label = torch.as_tensor(bank["y"], dtype=torch.long).cpu()
        self.window_event = torch.as_tensor(bank["event"], dtype=torch.long).cpu()
        self.window_event_verified = torch.as_tensor(
            bank["event_verified"], dtype=torch.bool
        ).cpu()
        self.window_cfg = torch.as_tensor(bank["cfg"], dtype=torch.long).cpu()
        self.window_subj = torch.as_tensor(bank["subj"], dtype=torch.long).cpu()
        window = torch.as_tensor(self.patch["window"], dtype=torch.long).cpu()
        order = torch.argsort(window, stable=True)
        self.order = order
        sorted_window = window[order]
        counts = torch.bincount(sorted_window, minlength=self.n_windows)
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        self._label_candidate_cache: dict[bytes, dict[int, torch.Tensor]] = {}
        self.label_rows: dict[int, torch.Tensor] = {}
        labels = torch.as_tensor(self.patch["y"], dtype=torch.long).cpu()
        for label in torch.unique(labels).tolist():
            self.label_rows[int(label)] = torch.nonzero(labels == label, as_tuple=True)[0]
        self.event_windows: dict[int, torch.Tensor] = {}
        verified_rows = torch.nonzero(self.window_event_verified, as_tuple=True)[0]
        if len(verified_rows):
            event_order = torch.argsort(self.window_event[verified_rows], stable=True)
            sorted_rows = verified_rows[event_order]
            sorted_events = self.window_event[sorted_rows]
            events, counts = torch.unique_consecutive(sorted_events, return_counts=True)
            offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
            for position, event in enumerate(events.tolist()):
                self.event_windows[int(event)] = sorted_rows[
                    int(offsets[position]):int(offsets[position + 1])
                ]

    def label_candidate_windows(self, memory_window_mask: torch.Tensor) -> dict[int, torch.Tensor]:
        """Eligible window ids per label, ascending — the grouping `sample_index_rows` scans for.

        Recomputing `nonzero(mask & window_label.eq(label))` per label walked all
        ``n_windows`` twice for each of the 93 labels on every active-index refresh, which was the
        whole 300 ms cost of a refresh. The grouping depends only on the mask and the (immutable)
        window labels, so it is built once per distinct mask — in practice two, the training and
        validation memories. One stable sort reproduces the per-label ascending order exactly.
        """
        mask = memory_window_mask.detach().cpu().bool()
        key = mask.numpy().tobytes()
        cached = self._label_candidate_cache.get(key)
        if cached is not None:
            return cached

        eligible = torch.nonzero(mask, as_tuple=True)[0]
        grouped: dict[int, torch.Tensor] = {label: _EMPTY_LONG for label in self.label_rows}
        if len(eligible):
            labels = self.window_label[eligible]
            order = torch.argsort(labels, stable=True)
            # `eligible` is ascending and the sort is stable, so each run stays ascending.
            sorted_labels, sorted_windows = labels[order], eligible[order]
            values, counts = torch.unique_consecutive(sorted_labels, return_counts=True)
            bounds = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
            for position, label in enumerate(values.tolist()):
                if int(label) in grouped:
                    grouped[int(label)] = sorted_windows[
                        int(bounds[position]):int(bounds[position + 1])
                    ]

        # A handful of entries covers the training and validation memories a run actually uses.
        if len(self._label_candidate_cache) >= 4:
            self._label_candidate_cache.pop(next(iter(self._label_candidate_cache)))
        self._label_candidate_cache[key] = grouped
        return grouped

    def rows_for_windows(self, windows: torch.Tensor) -> list[torch.Tensor]:
        rows = []
        for window in windows.detach().cpu().tolist():
            lo, hi = int(self.offsets[window]), int(self.offsets[window + 1])
            rows.append(self.order[lo:hi])
        return rows

    def gather_queries(
        self,
        windows: torch.Tensor,
        device,
        *,
        expand_verified_events: bool = True,
        allowed_window_mask: torch.Tensor | None = None,
    ) -> QueryPatches:
        window_groups = []
        allowed_cpu = (
            None if allowed_window_mask is None
            else allowed_window_mask.detach().cpu().bool()
        )
        for window in windows.detach().cpu().tolist():
            group = torch.tensor([window], dtype=torch.long)
            if expand_verified_events and bool(self.window_event_verified[window]):
                siblings = self.event_windows.get(int(self.window_event[window]), group)
                # An event id is only a positive identity when labels agree. This catches converter
                # clock/segmentation mistakes instead of silently making a multi-label query set.
                siblings = siblings[self.window_label[siblings].eq(self.window_label[window])]
                if allowed_cpu is not None:
                    siblings = siblings[allowed_cpu[siblings]]
                if len(siblings):
                    group = siblings
            window_groups.append(group)
        rows = []
        for group in window_groups:
            parts = self.rows_for_windows(group)
            rows.append(torch.cat(parts) if parts else torch.empty(0, dtype=torch.long))
        if not rows or min(map(len, rows)) < 1:
            raise ValueError("every query window must have at least one valid patch")
        B, Q = len(rows), max(map(len, rows))
        d = int(self.patch["Z"].shape[1])
        Z = torch.zeros(B, Q, d, dtype=torch.float32, device=device)
        mask = torch.zeros(B, Q, dtype=torch.bool, device=device)

        def allocate(dtype):
            return torch.zeros(B, Q, dtype=dtype, device=device)

        time = allocate(torch.float32)
        duration = allocate(torch.float32)
        resolution = torch.full((B, Q), -1, dtype=torch.long, device=device)
        window = allocate(torch.long)
        event = allocate(torch.long)
        subj = allocate(torch.long)
        cfg = allocate(torch.long)
        sensor = allocate(torch.long)
        row_id = torch.full((B, Q), -1, dtype=torch.long, device=device)
        for b, row in enumerate(rows):
            n = len(row)
            mask[b, :n] = True
            Z[b, :n] = torch.as_tensor(self.patch["Z"])[row].float().to(device)
            time[b, :n] = torch.as_tensor(self.patch["time"])[row].float().to(device)
            duration[b, :n] = torch.as_tensor(self.patch["duration"])[row].float().to(device)
            resolution[b, :n] = torch.as_tensor(self.patch["resolution"])[row].long().to(device)
            window[b, :n] = torch.as_tensor(self.patch["window"])[row].long().to(device)
            event[b, :n] = torch.as_tensor(self.patch["event"])[row].long().to(device)
            subj[b, :n] = torch.as_tensor(self.patch["subj"])[row].long().to(device)
            cfg[b, :n] = torch.as_tensor(self.patch["cfg"])[row].long().to(device)
            sensor[b, :n] = torch.as_tensor(self.patch["sensor"])[row].long().to(device)
            row_id[b, :n] = row.to(device)
        return QueryPatches(
            Z=Z, mask=mask, time=time, duration=duration, resolution=resolution,
            window=window, event=event, subj=subj, cfg=cfg, sensor=sensor, row=row_id,
        )

    def sample_index_rows(
        self,
        memory_window_mask: torch.Tensor,
        windows_per_label: int,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Label/config/subject-balanced source windows, retaining all of their patch grids."""
        if windows_per_label < 1:
            raise ValueError("windows_per_label must be positive")
        label_candidates = self.label_candidate_windows(memory_window_mask)
        selected_windows = []
        for label in sorted(self.label_rows):
            candidates = label_candidates[label]
            if len(candidates) <= windows_per_label:
                selected_windows.append(candidates)
                continue
            # Balance the retrievable corpus across configurations and subjects, but do not reserve
            # any rows from a chosen subject or otherwise shape it for enrollment. Episode support
            # is sampled later from this active view without subject constraints.
            picked = []
            remaining_candidates = candidates
            remaining_budget = windows_per_label
            remaining_cfg = self.window_cfg[remaining_candidates]
            configs = sorted(remaining_cfg.unique().tolist())
            # When a label occurs in more configurations than the active-window budget, assigning
            # the remainder to sorted config ids would permanently exclude later datasets/streams.
            # Randomize only the quota order; the supplied generator keeps the roster reproducible.
            configs = [int(value) for value in rng.permutation(configs)]
            base, extra = divmod(remaining_budget, max(1, len(configs)))
            for position, config in enumerate(configs):
                quota = base + int(position < extra)
                if quota == 0:
                    continue
                cfg_windows = remaining_candidates[remaining_cfg.eq(config)]
                # Temper large subjects by drawing subject round-robin within a configuration.
                # Grouped with one stable sort rather than a mask per subject; `unique` is ascending
                # and the sort is stable, so each bucket holds the same windows in the same order
                # and `rng.permutation` is called on identical arrays in identical sequence.
                cfg_subj = self.window_subj[cfg_windows]
                bucket_order = torch.argsort(cfg_subj, stable=True)
                bucket_subj = cfg_subj[bucket_order]
                bucket_windows = cfg_windows[bucket_order].numpy()
                values, counts = torch.unique_consecutive(bucket_subj, return_counts=True)
                edges = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
                subject_buckets = {
                    int(subject): list(
                        rng.permutation(bucket_windows[int(edges[i]):int(edges[i + 1])])
                    )
                    for i, subject in enumerate(values.tolist())
                }
                cfg_pick = []
                while len(cfg_pick) < quota and any(subject_buckets.values()):
                    for subject in sorted(subject_buckets):
                        if subject_buckets[subject] and len(cfg_pick) < quota:
                            cfg_pick.append(subject_buckets[subject].pop())
                picked.extend(cfg_pick)
            if len(picked) < windows_per_label:
                remaining = remaining_candidates[
                    ~torch.isin(remaining_candidates, torch.tensor(picked))
                ]
                take = min(windows_per_label - len(picked), len(remaining))
                picked.extend(rng.choice(remaining.numpy(), size=take, replace=False).tolist())
            selected_windows.append(torch.tensor(sorted(picked), dtype=torch.long))
        selected = [
            row
            for windows in selected_windows
            for row in self.rows_for_windows(windows)
        ]
        if not selected or not any(len(rows) for rows in selected):
            raise ValueError("no patch rows are eligible for the training memory index")
        return torch.cat(selected)


def _support_units(patch: dict, rows: torch.Tensor, device) -> torch.Tensor:
    """Return one support identity per patch: verified event, otherwise source window."""
    window = torch.as_tensor(patch["window"])[rows].long().to(device)
    event = torch.as_tensor(patch["event"])[rows].long().to(device)
    verified = torch.as_tensor(patch["event_verified"])[rows].bool().to(device)
    # Offset verified events so their ids cannot collide with ordinary source-window ids.
    offset = int(torch.as_tensor(patch["window"]).max()) + 1
    return torch.where(verified, event + offset, window)


def build_episode_memory_view(
    patch: dict,
    index_rows: torch.Tensor,
    query: QueryPatches,
    query_label: torch.Tensor,
    candidates: torch.Tensor,
    *,
    support_count: int | Sequence[int],
    episode_type: str,
    label_mode: str,
    rng: np.random.Generator,
    truth_present: bool = True,
) -> EpisodeMemoryView:
    """Construct support/background roles without mutating labels stored in the archive.

    Candidate concepts are removed from ordinary background memory. ``support_count`` event/window
    identities per candidate are then restored as provided support. All patches from one selected
    identity are retained, including verified synchronous placements.

    ``support_count`` is either one integer applied to every candidate, or a per-candidate sequence.
    The per-candidate form is what creates a **partially enrolled** episode: some candidates carry
    enrolled examples and the rest must be recognized from background memory and their name alone.
    That mixture is the regime the evidence engine exists for — with a uniform count, every episode
    is either "nothing is enrolled" (no substrate) or "everything is enrolled" (a prototype
    classifier is optimal), and neither needs retrieval over background memory.

    A candidate given 0 support keeps its concept ERASED from background memory, exactly as in a
    zero-support episode, so it can never be recognized by retrieving its own stored examples.
    """
    if episode_type not in {
        "semantic_zero_support", "ordinary_few_support",
        "cross_subject_few_support", "same_subject_enrollment",
    }:
        raise ValueError(f"unknown episode type {episode_type!r}")
    per_candidate = (
        [int(support_count)] * len(candidates)
        if isinstance(support_count, (int, np.integer))
        else [int(value) for value in support_count]
    )
    if len(per_candidate) != len(candidates):
        raise ValueError(
            f"support_count has {len(per_candidate)} entries for {len(candidates)} candidates"
        )
    if any(value < 0 for value in per_candidate):
        raise ValueError("support_count must be nonnegative")
    if episode_type == "semantic_zero_support" and any(per_candidate):
        raise ValueError("semantic_zero_support requires support_count=0")
    # A candidate presented under a meaningless episode-local alias and given no support is
    # unanswerable: the name carries no semantics to fall back on. Partial enrollment is therefore
    # a COHERENT-label construct by definition.
    if label_mode == "random_alias" and any(value == 0 for value in per_candidate):
        raise ValueError("random aliases are unanswerable without provided support")

    device = query.Z.device
    rows_cpu = index_rows.detach().cpu().long()
    y = torch.as_tensor(patch["y"])[rows_cpu].long().to(device)
    subj = torch.as_tensor(patch["subj"])[rows_cpu].long().to(device)
    event = torch.as_tensor(patch["event"])[rows_cpu].long().to(device)
    window = torch.as_tensor(patch["window"])[rows_cpu].long().to(device)
    support_unit = _support_units(patch, rows_cpu, device)
    candidates = candidates.long().to(device)
    query_label = query_label.long().to(device)
    query_truth_is_candidate = torch.isin(query_label, candidates)
    if truth_present and bool((~query_truth_is_candidate).any()):
        raise ValueError("every query label must be in an answerable episode candidate set")
    if not truth_present and bool(query_truth_is_candidate.any()):
        raise ValueError("truth-absent episodes cannot include a query label as a candidate")

    # A support identity can never be the query execution itself. For verified events this also
    # excludes every synchronous placement, rather than leaking a sibling sensor as enrollment.
    query_windows = torch.unique(query.window[query.mask])
    query_events = torch.unique(query.event[query.mask])
    eligible_support = ~torch.isin(window, query_windows) & ~torch.isin(event, query_events)
    support_mask = torch.zeros(len(rows_cpu), dtype=torch.bool, device=device)
    support_candidate = torch.full(
        (len(rows_cpu),), -1, dtype=torch.long, device=device
    )
    realized = []
    for candidate_position, label in enumerate(candidates.tolist()):
        support_count = per_candidate[candidate_position]
        label_rows = y.eq(label) & eligible_support
        if episode_type == "cross_subject_few_support":
            # Person-disjointness only. Acquisition disjointness is deliberately NOT required: 35 of
            # the 93 corpus labels exist in exactly one stream, so demanding it would silently drop
            # them from this regime entirely. Support recorded on another stream is allowed and
            # happens naturally wherever the label supports it; how often it happened is measured by
            # describe_episode_composition rather than enforced here.
            query_rows = query_label.eq(label)
            related_query_mask = (
                query_rows.unsqueeze(1) & query.mask if truth_present else query.mask
            )
            query_subjects = torch.unique(query.subj[related_query_mask])
            label_rows &= ~torch.isin(subj, query_subjects)
        elif episode_type == "same_subject_enrollment":
            query_rows = query_label.eq(label)
            if truth_present and bool(query_rows.any()):
                related_query_mask = query_rows.unsqueeze(1) & query.mask
                query_subjects = torch.unique(query.subj[related_query_mask])
                if len(query_subjects) != 1:
                    raise ValueError(
                        "same-subject enrollment requires one real query subject per candidate"
                    )
                label_rows &= subj.eq(query_subjects[0])
        units = torch.unique(support_unit[label_rows])
        if len(units) < support_count:
            raise ValueError(
                f"candidate label {label} has {len(units)} eligible support units, "
                f"needs {support_count}"
            )
        if support_count:
            chosen = rng.choice(
                units.detach().cpu().numpy(), size=support_count, replace=False
            )
            chosen_t = torch.as_tensor(chosen, dtype=torch.long, device=device)
            selected = y.eq(label) & torch.isin(support_unit, chosen_t)
            support_mask |= selected
            support_candidate[selected] = candidate_position
        realized.append(support_count)

    B, Q = query.mask.shape
    allowed = query.mask.unsqueeze(-1).expand(B, Q, len(rows_cpu)).clone()
    allowed &= window.view(1, 1, -1).ne(query.window.unsqueeze(-1))
    allowed &= event.view(1, 1, -1).ne(query.event.unsqueeze(-1))
    # Candidate classes enter memory only through explicitly provided support. Background labels
    # remain available as distractors and as semantic bridges in coherent-label episodes.
    candidate_rows = torch.isin(y, candidates)
    allowed &= (~candidate_rows | support_mask).view(1, 1, -1)
    if not truth_present:
        # An unanswerable example must not leak its true concept through ordinary background rows.
        allowed &= y.view(1, 1, -1).ne(query_label.view(B, 1, 1))
    if not bool((allowed.any(-1) | ~query.mask).all()):
        raise ValueError("at least one query patch has no eligible episodic memory rows")
    return EpisodeMemoryView(
        allowed=allowed,
        support_mask=support_mask,
        support_candidate=support_candidate,
        candidate_ids=candidates,
        query_label=query_label,
        support_units_per_candidate=torch.tensor(realized, device=device),
        episode_type=episode_type,
        label_mode=label_mode,
    )


def simultaneous_stream_pairs(
    window_cfg: torch.Tensor,
    window_event: torch.Tensor,
) -> set[frozenset]:
    """Stream pairs captured at the same instant, i.e. sharing verified event ids.

    xrf_v2's six placements and nfi_fared's back/wrist are recorded synchronously and therefore
    share an event; wisdm's phone and watch cover the same activities in separate sessions and
    share none. The distinction is invisible to the config id alone, so precompute it once and let
    :func:`describe_episode_composition` report which kind an episode actually drew.
    """
    cfg = window_cfg.detach().cpu().long()
    event = window_event.detach().cpu().long()
    order = torch.argsort(event, stable=True)
    events, counts = torch.unique_consecutive(event[order], return_counts=True)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    pairs: set[frozenset] = set()
    for position in torch.nonzero(counts > 1, as_tuple=True)[0].tolist():
        members = sorted({
            int(value) for value in
            cfg[order[int(offsets[position]):int(offsets[position + 1])]].tolist()
        })
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pairs.add(frozenset((a, b)))
    return pairs


def describe_episode_composition(
    patch: dict,
    index_rows: torch.Tensor,
    query: QueryPatches,
    view: EpisodeMemoryView,
    simultaneous_pairs: set[frozenset] | None = None,
) -> dict:
    """Plain-language account of what an episode's enrolled support actually consists of.

    Reports counts in whole recorded executions (and the patch rows they contribute) rather than
    enforcing anything, so an episode whose label only exists on one stream is described honestly
    instead of being excluded.
    """
    rows = index_rows.detach().cpu().long()
    support_local = view.support_rows.detach().cpu()
    subj = torch.as_tensor(patch["subj"])[rows].long()
    cfg = torch.as_tensor(patch["cfg"])[rows].long()
    unit = _support_units(patch, rows, torch.device("cpu"))
    candidate_of = view.support_candidate.detach().cpu()
    candidates = view.candidate_ids.detach().cpu().tolist()
    query_label = view.query_label.detach().cpu()
    q_mask = query.mask.detach().cpu()
    q_subj = query.subj.detach().cpu()
    q_cfg = query.cfg.detach().cpu()
    pairs = simultaneous_pairs or set()

    # NOTE on the stream categories: provided support can never be the query's own execution, so
    # these describe the RELATIONSHIP between the two streams, not whether these two particular
    # recordings happened at once. "worn simultaneously" means the corpus captures those streams
    # together as one rig (xrf_v2's placements, nfi_fared back+wrist); "independently recorded"
    # means the same activity was covered by both streams in unrelated sessions (wisdm's phone and
    # watch, or two different studies).
    counts = {
        "enrolled_examples": 0, "enrolled_patches": int(len(support_local)),
        "enrolled_examples_without_matching_query": 0,
        "performed_by_a_different_person": 0, "performed_by_the_same_person": 0,
        "from_the_query_s_own_stream": 0,
        "from_a_second_stream_worn_simultaneously": 0,
        "from_an_independently_recorded_stream": 0,
    }
    for position, label in enumerate(candidates):
        picked = support_local[candidate_of[support_local].eq(position)]
        if not len(picked):
            continue
        rows_for_label = q_mask & query_label.eq(int(label)).unsqueeze(1)
        has_matching_query = bool(rows_for_label.any())
        query_subjects = set(q_subj[rows_for_label].tolist()) if has_matching_query else set()
        query_configs = set(q_cfg[rows_for_label].tolist()) if has_matching_query else set()
        for execution in torch.unique(unit[picked]).tolist():
            member = picked[unit[picked].eq(execution)]
            counts["enrolled_examples"] += 1
            if not has_matching_query:
                # Distractor support has no corresponding query execution. Comparing it with an
                # unrelated query would overstate same-person and same-stream enrollment.
                counts["enrolled_examples_without_matching_query"] += 1
                continue
            key = ("performed_by_the_same_person"
                   if set(subj[member].tolist()) & query_subjects
                   else "performed_by_a_different_person")
            counts[key] += 1
            streams = set(cfg[member].tolist())
            if streams & query_configs:
                counts["from_the_query_s_own_stream"] += 1
            outside_streams = streams - query_configs
            paired_within_support = any(
                frozenset((left, right)) in pairs
                for left in streams for right in streams if left < right
            )
            paired_outside = {
                stream for stream in outside_streams
                if any(frozenset((stream, query_stream)) in pairs
                       for query_stream in query_configs)
            }
            if paired_within_support or paired_outside:
                counts["from_a_second_stream_worn_simultaneously"] += 1
            if any(stream not in paired_outside for stream in outside_streams):
                counts["from_an_independently_recorded_stream"] += 1
    counts["synthetic_persona"] = {
        "same_subject_enrollment": "one persona shared by the query and its support",
        "cross_subject_few_support": "a different persona for the query and its support",
    }.get(view.episode_type, "none applied")
    return counts


def support_capacity_by_label(
    patch: dict,
    index_rows: torch.Tensor,
    n_labels: int,
) -> torch.Tensor:
    """Count independent verified-event/window support identities per canonical label."""
    rows = index_rows.detach().cpu().long()
    labels = torch.as_tensor(patch["y"])[rows].long()
    units = _support_units(patch, rows, torch.device("cpu"))
    capacity = torch.zeros(n_labels, dtype=torch.long)
    if not len(labels):
        return capacity
    # Count distinct (label, unit) pairs in one pass instead of rescanning the whole active index
    # once per label. The pair is packed into a single integer rather than using
    # `torch.unique(..., dim=0)`, which sorts row-wise and measured 5.6x *slower* than the per-label
    # loop it would replace. Labels and units are both non-negative ids, so the packing is injective.
    stride = int(units.max()) + 1
    distinct = torch.unique(labels * stride + units)
    capacity.index_add_(
        0, torch.div(distinct, stride, rounding_mode="floor"),
        torch.ones(len(distinct), dtype=torch.long),
    )
    return capacity


def queries_from_encoded(
    encoded: dict[str, torch.Tensor],
    windows: torch.Tensor,
    device,
    *,
    sensor_id: int = -1,
    subject_ids: torch.Tensor | None = None,
) -> QueryPatches:
    """Pack selected windows from ``encode_dataset_detailed`` into a padded query patch set."""
    selected_rows = []
    patch_window = encoded["patch_window"].long()
    for window in windows.detach().cpu().tolist():
        selected_rows.append(torch.nonzero(patch_window.eq(window), as_tuple=True)[0])
    if not selected_rows or min(map(len, selected_rows)) < 1:
        raise ValueError("every external query window must contain at least one valid patch")
    B, Q = len(selected_rows), max(map(len, selected_rows))
    d = int(encoded["patch_Z"].shape[1])
    Z = torch.zeros(B, Q, d, dtype=torch.float32, device=device)
    mask = torch.zeros(B, Q, dtype=torch.bool, device=device)
    time = torch.zeros(B, Q, dtype=torch.float32, device=device)
    duration = torch.zeros(B, Q, dtype=torch.float32, device=device)
    resolution = torch.full((B, Q), -1, dtype=torch.long, device=device)
    window_id = torch.zeros(B, Q, dtype=torch.long, device=device)
    # Negative ids cannot collide with nonnegative memory ids. Each external window is its own
    # event/window; all its patches share a local sensor-membership id.
    event = torch.zeros(B, Q, dtype=torch.long, device=device)
    subj = torch.zeros(B, Q, dtype=torch.long, device=device)
    cfg = torch.full((B, Q), int(sensor_id), dtype=torch.long, device=device)
    if subject_ids is not None:
        subject_ids = subject_ids.detach().cpu().long()
        if tuple(subject_ids.shape) != (B,):
            raise ValueError(f"subject_ids must have shape {(B,)}, got {tuple(subject_ids.shape)}")
    for b, rows in enumerate(selected_rows):
        n = len(rows)
        mask[b, :n] = True
        Z[b, :n] = encoded["patch_Z"][rows].float().to(device)
        time[b, :n] = encoded["patch_time"][rows].float().to(device)
        duration[b, :n] = encoded["patch_duration"][rows].float().to(device)
        resolution[b, :n] = encoded["patch_resolution"][rows].long().to(device)
        external_id = -(b + 1)
        window_id[b, :n] = external_id
        event[b, :n] = external_id
        subj[b, :n] = external_id if subject_ids is None else int(subject_ids[b])
    return QueryPatches(
        Z=Z, mask=mask, time=time, duration=duration, resolution=resolution,
        window=window_id, event=event, subj=subj, cfg=cfg, sensor=cfg.clone(),
        row=torch.full((B, Q), -1, dtype=torch.long, device=device),
    )


def assemble_evidence(
    retrieval: PatchRetrieval,
    online_scores: torch.Tensor,
    index_rows: torch.Tensor,
    *,
    max_evidence: int,
    tau: float,
) -> EvidenceBatch:
    """Keep the highest-scoring unique memory rows from learned query retrieval."""
    if max_evidence < 1:
        raise ValueError("max_evidence must be positive")
    if tau <= 0:
        raise ValueError("retrieval temperature must be positive")
    if online_scores.shape != retrieval.index.shape:
        raise ValueError("online scores must align with retrieved indices")

    B, Q, H, K = retrieval.index.shape
    device = online_scores.device
    flat_count = Q * H * K
    flat_local = retrieval.index.reshape(B, flat_count)
    flat_valid = retrieval.valid.reshape(B, flat_count)
    flat_scores = online_scores.reshape(B, flat_count)
    sortable = flat_scores.detach().masked_fill(~flat_valid, float("-inf"))
    order = torch.argsort(sortable, dim=1, descending=True, stable=True)
    sorted_local = torch.gather(flat_local, 1, order)
    sorted_valid = torch.gather(flat_valid, 1, order)
    sorted_scores = torch.gather(flat_scores, 1, order)

    # Keep the first (therefore highest-scoring) occurrence of each active-memory row.  scatter-reduce
    # avoids the historical Python scan and CPU synchronization for every query/head/top-k tuple.
    rank = torch.arange(flat_count, device=device).view(1, -1).expand(B, -1)
    first_rank = torch.full(
        (B, len(index_rows)), flat_count, dtype=torch.long, device=device
    )
    first_rank.scatter_reduce_(
        1, sorted_local, rank.masked_fill(~sorted_valid, flat_count),
        reduce="amin", include_self=True,
    )
    unique = sorted_valid & first_rank.gather(1, sorted_local).eq(rank)
    unique_count = unique.sum(1)
    if bool((unique_count == 0).any()):
        raise ValueError("learned query retrieval produced no evidence")

    E = min(max_evidence, int(unique_count.max()))
    accepted_rank = rank.masked_fill(~unique, flat_count).topk(
        E, dim=1, largest=False, sorted=True
    ).values
    mask = accepted_rank.lt(flat_count)
    gather_rank = accepted_rank.clamp_max(flat_count - 1)
    local_index = torch.gather(sorted_local, 1, gather_rank).masked_fill(~mask, 0)
    scores = torch.gather(sorted_scores, 1, gather_rank).masked_fill(~mask, 0.0)

    # Flattening order is q -> head -> top-k, matching the selector tensor layout.
    original_position = torch.gather(order, 1, gather_rank)
    query_patch = torch.div(original_position, H * K, rounding_mode="floor")
    head = torch.div(original_position, K, rounding_mode="floor").remainder(H)
    query_patch = query_patch.masked_fill(~mask, 0)
    head = head.masked_fill(~mask, 0)
    global_rows = index_rows.to(device=device, dtype=torch.long)
    index = global_rows[local_index].masked_fill(~mask, 0)

    weights = torch.softmax(scores.masked_fill(~mask, float("-inf")) / tau, dim=1)
    weights = weights.masked_fill(~mask, 0.0)
    return EvidenceBatch(
        index=index,
        weights=weights,
        scores=scores,
        mask=mask,
        head=head,
        query_patch=query_patch,
        local_index=local_index,
    )
