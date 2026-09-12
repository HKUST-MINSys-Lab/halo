"""Focused contracts for multi-horizon HALO JEPA."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from training.tokenizer.future_jepa import (
    FuturePredictor,
    balanced_future_latent_loss,
    balanced_physical_loss,
    combine_future_losses,
    fixed_filterbank_physical_targets,
    gather_token_rows,
    make_future_target_plan,
    normalized_teacher_target,
    past_context_references,
    pack_selected_interval_patches,
    patch_variance_covariance,
    recommend_fixed_objective_weights,
)
from training.tokenizer.pretrain import (
    MULTISPAN_MAX_BATCH_TOKENS, MULTISPAN_REFERENCE_BATCH,
    PipelineAModel, PretrainConfig, batch_under_token_budget,
    frontend_rope_min_period, future_tokens_per_window,
    validate_source_window_contract,
)


def _three_scale_grid(batch: int = 2):
    entries = []
    for resolution, duration in enumerate((0.5, 1.0, 1.5)):
        start = 0.0
        while start + duration <= 6.0 + 1e-8:
            entries.append((start, start + duration, resolution))
            start += duration
    entries.sort(key=lambda item: ((item[0] + item[1]) / 2, item[2]))
    starts = torch.tensor([[item[0] for item in entries]]).expand(batch, -1).clone()
    ends = torch.tensor([[item[1] for item in entries]]).expand(batch, -1).clone()
    groups = torch.tensor([[item[2] for item in entries]]).expand(batch, -1).clone()
    valid = torch.ones_like(starts, dtype=torch.bool)
    return starts, ends, groups, valid


@pytest.mark.parametrize(
    ("frontend", "multiresolution", "expected_resolutions"),
    (("fixed", True, 3), ("learnable", True, 3), ("continuous", False, 1),
     ("multispan", False, 3)),
)
def test_future_jepa_model_is_frontend_agnostic(
    frontend: str, multiresolution: bool, expected_resolutions: int,
):
    cfg = PretrainConfig(
        d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
        frontend=frontend, token_granularity="sensor", text_conditioning="factored",
        multiresolution=multiresolution, future_predictor_dim=64,
        future_predictor_heads=4,
    )
    model = PipelineAModel(cfg)
    assert model.future_predictor.resolution_embedding.num_embeddings >= expected_resolutions
    assert model.physical_target_analyzer is not None
    assert model.physical_target_analyzer.learnable is False
    assert not any(parameter.requires_grad
                   for parameter in model.physical_target_analyzer.parameters())
    assert model.physical_decoder[-1].out_features == 3 * (
        model.physical_target_analyzer.n_bands
        + int(model.physical_target_analyzer.use_dc)
    )


def test_eight_second_future_schedules_fit_the_token_budget():
    fixed = PretrainConfig(source_window_seconds=8.0)
    assert future_tokens_per_window(fixed) == 30
    assert batch_under_token_budget(future_tokens_per_window(fixed)) == 512

    multispan = PretrainConfig(
        frontend="multispan", multiresolution=False, source_window_seconds=8.0,
    )
    assert future_tokens_per_window(multispan) == 118
    assert batch_under_token_budget(
        future_tokens_per_window(multispan),
        reference_batch=MULTISPAN_REFERENCE_BATCH,
        max_batch_tokens=MULTISPAN_MAX_BATCH_TOKENS,
    ) == 384
    assert frontend_rope_min_period(multispan) == pytest.approx(0.25)


def test_label_free_window_contract_rejects_a_stale_grid():
    index = SimpleNamespace(refs=[SimpleNamespace(
        rate_hz=80.0, shape=(10, 480, 6), key="nhanes/watch_wrist",
    )])
    with pytest.raises(ValueError, match="nhanes/watch_wrist=6s"):
        validate_source_window_contract(index, 8.0)


def test_future_plan_is_deterministic_disjoint_and_span_safe():
    starts, ends, groups, valid = _three_scale_grid()
    a = make_future_target_plan(
        starts, ends, valid, groups, generator=torch.Generator().manual_seed(7),
    )
    b = make_future_target_plan(
        starts, ends, valid, groups, generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(a.context_mask, b.context_mask)
    assert torch.equal(a.target_mask, b.target_mask)
    assert a.eligible.all()
    assert not (a.context_mask & a.target_mask).any()
    assert (ends[a.context_mask] <= a.context_end[:, None].expand_as(ends)[a.context_mask]).all()
    assert (starts[a.target_mask] >= a.context_end[:, None].expand_as(starts)[a.target_mask]).all()
    assert set(groups[a.target_mask].tolist()) == {0, 1, 2}
    assert (a.horizon_seconds[a.target_mask] >= 0).all()
    assert (a.horizon_seconds[a.target_mask] < 3.0).all()


def test_future_plan_marks_single_token_window_ineligible():
    plan = make_future_target_plan(
        torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.tensor([[True]]),
        torch.tensor([[0]]), generator=torch.Generator().manual_seed(1),
    )
    assert not plan.eligible.item()
    assert not plan.context_mask.any()
    assert not plan.target_mask.any()


def test_missing_resolution_cannot_erase_another_resolution_target():
    # The future interval deliberately occupies column zero and another group is context-only.
    starts = torch.tensor([[2.0, 0.0]])
    plan = make_future_target_plan(
        starts, starts + 1, torch.ones_like(starts, dtype=torch.bool),
        torch.tensor([[0, 1]]), context_fraction=(0.5, 0.5),
    )
    assert plan.eligible.tolist() == [True]
    assert plan.context_mask.tolist() == [[False, True]]
    assert plan.target_mask.tolist() == [[True, False]]


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (2, 3)])
def test_empty_or_fully_invalid_future_grids_are_finite_and_ineligible(shape):
    starts = torch.full(shape, float("nan"))
    plan = make_future_target_plan(starts, starts, torch.zeros(shape, dtype=torch.bool))
    assert not plan.eligible.any()
    assert not plan.context_mask.any()
    assert not plan.target_mask.any()
    assert torch.isfinite(plan.context_end).all()
    assert torch.isfinite(plan.horizon_seconds).all()


@pytest.mark.parametrize("with_groups", [False, True])
def test_batched_planner_matches_scalar_spec_on_ragged_permuted_grids(with_groups):
    # A scalar specification independently replays the batch RNG draws. It exercises missing
    # groups, invalid metadata, overlapping spans, unsorted columns, and fallback boundaries.
    rng = torch.Generator().manual_seed(37)
    starts = torch.randint(0, 12, (24, 18), generator=rng).float() / 2
    ends = starts + torch.randint(1, 7, starts.shape, generator=rng).float() / 2
    groups = torch.randint(0, 3, starts.shape, generator=rng) if with_groups else None
    valid = torch.rand(starts.shape, generator=rng) > 0.3
    valid[0] = False
    starts[1, :2] = float("nan")
    fractions = (0.01, 0.02)  # forces deterministic boundary fallback on some rows
    bins = ((0., 1.), (1., 2.), (2., 3.))
    seed = 83
    plan = make_future_target_plan(
        starts, ends, valid, groups, context_fraction=fractions,
        horizon_bins_seconds=bins, generator=torch.Generator().manual_seed(seed),
    )
    replay = torch.Generator().manual_seed(seed)
    draws = torch.rand(24, 8, generator=replay)
    anchors = torch.stack([torch.rand(24, generator=replay) for _ in bins])
    live = valid & torch.isfinite(starts) & torch.isfinite(ends) & (ends > starts)
    for row in range(24):
        indices = torch.where(live[row])[0].tolist()
        if len(indices) < 2:
            assert not plan.eligible[row]
            continue
        s, e = starts[row].tolist(), ends[row].tolist()
        first, last = min(s[i] for i in indices), max(e[i] for i in indices)
        candidates = [first + float(fractions[0] + (fractions[1] - fractions[0]) * u)
                      * (last - first) for u in draws[row]]
        candidates += sorted({t for i in indices for t in (s[i], e[i]) if first < t < last})
        boundary = next((t for t in candidates if any(e[i] <= t + 1e-7 for i in indices)
                         and any(s[i] >= t - 1e-7 for i in indices)), None)
        if boundary is None:
            assert not plan.eligible[row]
            continue
        assert float(plan.context_end[row]) == pytest.approx(boundary, abs=1e-6)
        future = [i for i in indices if s[i] >= boundary - 1e-7]
        centers = [(a + b) / 2 for a, b in zip(s, e)]
        selected = {}

        def cover(anchor):
            covering = [i for i in future if s[i] <= anchor + 1e-7 < e[i]]
            labels = {int(groups[row, i]) if groups is not None else 0 for i in covering}
            for label in labels:
                matching = [i for i in covering if groups is None or int(groups[row, i]) == label]
                chosen = min(matching, key=lambda i: (abs(centers[i] - anchor), i))
                selected.setdefault(chosen, anchor - boundary)

        for bin_id, (lo, hi) in enumerate(bins):
            choices = sorted({centers[i] for i in future if lo <= centers[i] - boundary < hi})
            if choices:
                cover(choices[int(float(anchors[bin_id, row]) * len(choices))])
        if not selected:
            cover(min(centers[i] for i in future))
        assert torch.where(plan.target_mask[row])[0].tolist() == sorted(selected)
        for index, horizon in selected.items():
            assert float(plan.horizon_seconds[row, index]) == pytest.approx(horizon, abs=1e-6)


def test_duplicate_centers_do_not_bias_physical_anchor_sampling():
    batch = 4096
    # Four duplicate tokens at 2.5s and one at 3.5s must still give each time 50% probability.
    starts = torch.tensor([[0., 2., 2., 2., 2., 3.]]).expand(batch, -1)
    plan = make_future_target_plan(
        starts, starts + 1, torch.ones_like(starts, dtype=torch.bool),
        context_fraction=(0.25, 0.25), horizon_bins_seconds=((0., 5.),),
        generator=torch.Generator().manual_seed(9),
    )
    assert plan.eligible.all()
    assert plan.target_mask.sum(1).eq(1).all()
    late_fraction = plan.target_mask[:, -1].float().mean().item()
    assert 0.47 < late_fraction < 0.53


def test_one_physical_anchor_selects_at_most_one_patch_per_resolution():
    starts, ends, groups, valid = _three_scale_grid(batch=1)
    for seed in range(20):
        plan = make_future_target_plan(
            starts, ends, valid, groups,
            horizon_bins_seconds=((0.0, 6.0),),
            generator=torch.Generator().manual_seed(seed),
        )
        assert plan.eligible.item()
        counts = [int((plan.target_mask & groups.eq(group)).sum()) for group in range(3)]
        assert max(counts) <= 1


def test_overlapping_multispan_grid_selects_one_frame_per_resolution_and_horizon():
    entries = []
    for resolution, span in enumerate((0.5, 1.0, 1.5)):
        stride = span / 4
        for frame in range(int(6.0 / stride)):
            center = (frame + 0.5) * stride
            entries.append((center - span / 2, center + span / 2, resolution))
    starts = torch.tensor([[item[0] for item in entries]])
    ends = torch.tensor([[item[1] for item in entries]])
    groups = torch.tensor([[item[2] for item in entries]])
    valid = torch.ones_like(starts, dtype=torch.bool)
    plan = make_future_target_plan(
        starts, ends, valid, groups, generator=torch.Generator().manual_seed(4),
    )
    # Three horizon bins times three resolutions is the absolute upper bound.
    assert 1 <= int(plan.target_mask.sum()) <= 9
    selected_centers = 0.5 * (starts + ends)
    for horizon in torch.unique(plan.horizon_seconds[plan.target_mask]):
        at_horizon = plan.target_mask & plan.horizon_seconds.eq(horizon)
        for group in range(3):
            assert int((at_horizon & groups.eq(group)).sum()) <= 1
        assert torch.unique(selected_centers[at_horizon]).numel() <= 3


def test_teacher_target_is_normalized_average_and_detached():
    a = torch.randn(2, 4, 1, 8, requires_grad=True)
    b = torch.randn(2, 4, 1, 8, requires_grad=True)
    target = normalized_teacher_target((a, b), top_k=2)
    expected = torch.nn.functional.layer_norm((a + b).detach().float() / 2, (8,))
    assert torch.allclose(target, expected)
    assert not target.requires_grad
    assert torch.allclose(target.mean(dim=-1), torch.zeros(2, 4, 1), atol=1e-5)


def test_predictor_has_gradients_without_future_signal_input():
    torch.manual_seed(3)
    model = FuturePredictor(16, predictor_dim=16, num_layers=2, num_heads=4, dropout=0.0)
    context = torch.randn(2, 5, 2, 16, requires_grad=True)
    context_valid = torch.zeros(2, 5, 2, dtype=torch.bool)
    context_valid[:, :3] = True
    target_mask = torch.zeros(2, 5, dtype=torch.bool)
    target_mask[:, 3:] = True
    positions = torch.arange(5).float().view(1, 5).expand(2, -1)
    durations = torch.ones(2, 5)
    groups = torch.tensor([[0, 1, 0, 1, 0]]).expand(2, -1)
    horizons = torch.where(target_mask, positions - 2.0, torch.zeros_like(positions))
    descriptions = torch.randn(2, 2, 384)
    prediction, indices, query_valid = model(
        context, context_valid, target_mask, positions, durations, groups,
        horizons, torch.full((2,), 2.0), descriptions, torch.ones(2, 2, dtype=torch.bool),
    )
    target = torch.randn_like(prediction)
    query_groups = groups.gather(1, indices[..., 0])
    loss = balanced_future_latent_loss(prediction, target, query_valid, query_groups)
    loss.backward()
    assert context.grad is not None and context.grad[:, :3].abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert torch.isfinite(prediction).all()


def test_future_prediction_cannot_read_invalid_context_rows():
    torch.manual_seed(9)
    model = FuturePredictor(16, predictor_dim=16, num_layers=1, num_heads=4, dropout=0.0).eval()
    context = torch.randn(1, 5, 1, 16)
    changed = context.clone()
    changed[:, 3:] += 10_000
    context_valid = torch.tensor([[[True], [True], [True], [False], [False]]])
    target_mask = torch.tensor([[False, False, False, True, True]])
    positions = torch.arange(5).float().view(1, 5)
    durations = torch.ones(1, 5)
    groups = torch.zeros(1, 5, dtype=torch.long)
    horizons = torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0]])
    descriptions = torch.randn(1, 1, 384)
    present = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        a = model(context, context_valid, target_mask, positions, durations, groups,
                  horizons, torch.tensor([2.0]), descriptions, present)[0]
        b = model(changed, context_valid, target_mask, positions, durations, groups,
                  horizons, torch.tensor([2.0]), descriptions, present)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_predictor_uses_context_time_order_and_is_translation_invariant():
    torch.manual_seed(13)
    model = FuturePredictor(8, predictor_dim=8, num_layers=1, num_heads=2, dropout=0.0).eval()
    context = torch.randn(1, 5, 1, 8)
    context_valid = torch.tensor([[[True], [True], [True], [False], [False]]])
    target_mask = torch.tensor([[False, False, False, True, True]])
    positions = torch.arange(5).float().view(1, 5)
    durations = torch.ones(1, 5)
    groups = torch.zeros(1, 5, dtype=torch.long)
    horizons = torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0]])
    descriptors = torch.randn(1, 1, 384)
    present = torch.ones(1, 1, dtype=torch.bool)
    shuffled = context.clone()
    shuffled[:, :3] = shuffled[:, torch.tensor([2, 0, 1])]
    with torch.no_grad():
        baseline = model(
            context, context_valid, target_mask, positions, durations, groups, horizons,
            torch.tensor([2.0]), descriptors, present,
        )[0]
        reordered = model(
            shuffled, context_valid, target_mask, positions, durations, groups, horizons,
            torch.tensor([2.0]), descriptors, present,
        )[0]
        shifted = model(
            context, context_valid, target_mask, positions + 17.0, durations, groups, horizons,
            torch.tensor([19.0]), descriptors, present,
        )[0]
    assert (baseline - reordered).abs().max() > 1e-5
    assert torch.allclose(baseline, shifted, atol=1e-6)


def test_past_context_references_are_resolution_specific_and_detached():
    targets = torch.tensor([[
        [[1.0, 2.0], [10.0, 20.0]],
        [[3.0, 4.0], [30.0, 40.0]],
        [[9.0, 9.0], [90.0, 90.0]],
    ]], requires_grad=True)
    valid = torch.tensor([[[True, True], [True, True], [False, False]]])
    positions = torch.tensor([[1.0, 2.0, 3.0]])
    groups = torch.tensor([[0, 0, 1]])
    mean, latest, available = past_context_references(targets, valid, positions, groups, 2)
    assert available.tolist() == [[[True, False], [True, False]]]
    assert torch.allclose(mean[0, :, 0], torch.tensor([[2.0, 3.0], [20.0, 30.0]]))
    assert torch.allclose(latest[0, :, 0], torch.tensor([[3.0, 4.0], [30.0, 40.0]]))
    assert not mean.requires_grad and not latest.requires_grad


def test_predictor_handles_ineligible_rows_without_nan_or_loss_leakage():
    model = FuturePredictor(16, predictor_dim=16, num_layers=1, num_heads=4, dropout=0.0)
    context = torch.randn(2, 4, 1, 16)
    context_valid = torch.tensor([
        [[True], [True], [False], [False]],
        [[False], [False], [False], [False]],
    ])
    target_mask = torch.tensor([[False, False, True, False], [False, False, False, False]])
    positions = torch.arange(4).float().expand(2, -1)
    durations = torch.ones(2, 4)
    groups = torch.zeros(2, 4, dtype=torch.long)
    horizons = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    prediction, indices, valid = model(
        context, context_valid, target_mask, positions, durations, groups, horizons,
        torch.full((2,), 1.0), torch.randn(2, 1, 384), torch.ones(2, 1, dtype=torch.bool),
    )
    target = torch.randn_like(prediction)
    loss = balanced_future_latent_loss(
        prediction, target, valid, groups.gather(1, indices[..., 0]), num_resolutions=1,
    )
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(loss)
    assert not valid[1].any()


def test_direct_patch_collapse_control_penalizes_constant_rows():
    diverse = torch.randn(64, 1, 16)
    collapsed = torch.zeros_like(diverse)
    valid = torch.ones(64, 1, dtype=torch.bool)
    good = patch_variance_covariance(diverse, valid)
    bad = patch_variance_covariance(collapsed, valid)
    assert bad.total > good.total
    assert bad.min_std < good.min_std


def test_balanced_losses_ignore_padding_and_average_resolutions_equally():
    prediction = torch.zeros(1, 3, 2)
    target = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    valid = torch.tensor([[True, True, False]])
    groups = torch.tensor([[0, 1, 1]])
    latent = balanced_future_latent_loss(prediction, target, valid, groups)
    assert torch.isfinite(latent)

    feature_valid = valid.unsqueeze(-1).expand_as(target)
    physical = balanced_physical_loss(prediction, target, valid, groups, feature_valid)
    expected = (0.5 + 2.5) / 2  # smooth-L1 errors for magnitudes 1 and 3, equal group weight
    assert physical == pytest.approx(expected)


def test_gather_token_rows_preserves_patch_sensor_pairing():
    values = torch.arange(1 * 3 * 2 * 4).reshape(1, 3, 2, 4)
    indices = torch.tensor([[[2, 1], [0, 0]]])
    gathered = gather_token_rows(values, indices)
    assert torch.equal(gathered[0, 0], values[0, 2, 1])
    assert torch.equal(gathered[0, 1], values[0, 0, 0])


def test_selected_physical_intervals_are_packed_with_exact_lookup():
    window = torch.arange(2 * 12, dtype=torch.float32).reshape(2, 12, 1)
    total = torch.tensor([12, 10])
    rates = torch.tensor([2.0, 2.0])
    starts = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    ends = starts + 1.0
    selected = torch.tensor([[False, True, False, True], [True, False, False, False]])
    values, lengths, slot_by_patch, valid = pack_selected_interval_patches(
        window, total, rates, starts, ends, selected, capacity=3,
    )
    assert values.shape == (2, 2, 3, 1)
    assert valid.tolist() == [[True, True], [True, False]]
    assert lengths.tolist() == [[2, 2], [2, 0]]
    assert values[0, slot_by_patch[0, 1], :2, 0].tolist() == [2.0, 3.0]
    assert values[0, slot_by_patch[0, 3], :2, 0].tolist() == [6.0, 7.0]
    assert values[1, slot_by_patch[1, 0], :2, 0].tolist() == [12.0, 13.0]
    assert not values[1, 1].any()


def test_filterbank_masks_gate_physical_values_but_are_not_targets():
    # K=2 layout: energy(2), nyquist(2), resolution(2), amplitude(1), dc(1).
    analysis = torch.tensor([[[[1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 3.0, -4.0]]]])
    values, valid = fixed_filterbank_physical_targets(
        analysis, n_bands=2, use_resolution_mask=True, use_amplitude=True, use_dc=True,
    )
    assert values.tolist() == [[[[1.0, 2.0, 3.0, -4.0]]]]
    assert valid.tolist() == [[[[1.0, 0.0, 1.0, 1.0]]]]


def test_filterbank_resolution_confidence_is_not_hard_thresholded():
    # K=1 layout: energy, nyquist, resolution, amplitude, dc.
    analysis = torch.tensor([[[[2.0, 1.0, 0.25, 3.0, -4.0]]]])
    _, weight = fixed_filterbank_physical_targets(
        analysis, n_bands=1, use_resolution_mask=True, use_amplitude=True,
        use_dc=True, include_amplitude=False,
    )
    assert weight.tolist() == [[[[0.25, 1.0]]]]


def test_three_objective_calibration_hits_requested_shares_and_preserves_scale():
    sample = {
        "norms": {"future": 1.0, "physical": 2.0, "collapse": 4.0},
        "dots": {
            "future|physical": 0.0,
            "future|collapse": 0.0,
            "physical|collapse": 0.0,
        },
        "cosines": {},
    }
    report = recommend_fixed_objective_weights(
        [copy.deepcopy(sample) for _ in range(4)],
        {"future": 1.0, "physical": 1.0, "collapse": 1.0},
        {"future": 0.7, "physical": 0.2, "collapse": 0.1},
    )
    assert report["achieved_median_gradient_shares"] == pytest.approx({
        "future": 0.7, "physical": 0.2, "collapse": 0.1,
    })
    norms = report["median_combined_encoder_grad_norm"]
    assert norms["recommended_weights"] == pytest.approx(norms["pilot_weights"])


def test_combined_future_loss_keeps_each_term_visible():
    output = combine_future_losses(
        torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0),
        future_weight=0.7, physical_weight=0.2, collapse_weight=0.1,
    )
    assert set(output.terms) == {"future", "physical", "collapse"}
    assert output.total == pytest.approx(1.4)


def test_patch_collapse_control_averages_resolutions_equally():
    states = torch.randn(1, 6, 1, 8)
    valid = torch.ones(1, 6, 1, dtype=torch.bool)
    groups = torch.tensor([[0, 0, 0, 0, 1, 1]])
    grouped = patch_variance_covariance(
        states, valid, resolution_ids=groups, num_resolutions=2,
    )
    first = patch_variance_covariance(states, valid & groups.eq(0).unsqueeze(-1))
    second = patch_variance_covariance(states, valid & groups.eq(1).unsqueeze(-1))
    assert grouped.variance == pytest.approx((first.variance + second.variance) / 2)
    assert grouped.covariance == pytest.approx((first.covariance + second.covariance) / 2)


def test_patch_collapse_control_ignores_inactive_resolution():
    states = torch.randn(2, 3, 1, 8)
    valid = torch.tensor([
        [[True], [True], [False]],
        [[True], [True], [False]],
    ])
    groups = torch.tensor([[0, 0, 1], [0, 0, 1]])
    grouped = patch_variance_covariance(
        states, valid, resolution_ids=groups, num_resolutions=2,
    )
    active_only = patch_variance_covariance(states, valid & groups.eq(0).unsqueeze(-1))
    assert grouped.total == pytest.approx(active_only.total)
    assert grouped.min_std == pytest.approx(active_only.min_std)


def test_future_recipe_constructs_only_active_pretraining_heads():
    model = PipelineAModel(PretrainConfig())
    assert set(model.pretraining_heads()) == {"future_predictor", "physical_decoder"}
    assert not model.encoder.mask_token.requires_grad
    assert model.jepa_predictor is None
    assert model.vicreg_projector is None
    assert model.mae_head is None
