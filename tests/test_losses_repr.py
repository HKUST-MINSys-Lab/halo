"""Unit tests for the consolidated JEPA + augmentation-VICReg Phase-A losses."""

from __future__ import annotations

import copy

import pytest
import torch

from training.tokenizer.losses_repr import (
    MIN_VISIBLE_TIME,
    make_mask_plan,
    make_per_resolution_mask_plan,
    masked_ema_latent_loss,
    phase_a_loss,
    vicreg,
)
from training.tokenizer.pretrain import (
    PipelineAModel,
    PretrainConfig,
    objective_encoder_grad_geometry,
    recommend_objective_weights,
    update_ema_encoder,
)

GYRO = [3, 4, 5]


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ------------------------------------------------------------------------- mask plan
def test_mask_ratio_and_floor():
    plan = make_mask_plan(B=64, T=10, C=6, gyro_channels=GYRO, generator=gen())
    mask = plan.token_mask
    assert mask.shape == (64, 10, 6)
    ratio = mask.float().mean()
    assert 0.3 < ratio < 0.8, f"overall mask ratio implausible: {ratio}"
    visible_t = (~mask).any(dim=2).sum(dim=1)
    assert (visible_t >= MIN_VISIBLE_TIME).all(), "visible-time floor violated"


def test_mask_adapts_to_small_T():
    """Multi-scale patch_seconds means T can be tiny — the floor must hold at T=3."""
    plan = make_mask_plan(B=32, T=3, C=6, gyro_channels=GYRO, generator=gen(1))
    visible_t = (~plan.token_mask).any(dim=2).sum(dim=1)
    assert (visible_t >= MIN_VISIBLE_TIME).all()


def test_one_patch_window_gets_cross_channel_jepa_target():
    channels = torch.tensor([[True, True, True, False, False, False]])
    plan = make_mask_plan(
        B=1, T=1, C=6, gyro_channels=GYRO, generator=gen(9),
        channel_event_p=0.0, valid_patches=torch.ones(1, 1, dtype=torch.bool),
        channel_mask=channels,
    )
    assert int(plan.token_mask.sum()) == 1
    assert not bool(plan.token_mask[..., 3:].any())


# -------------------------------------------------------------- per-resolution mask plan
def _per_res_grid(n_short: int, n_long: int):
    """Interleaved (B=1) resolution ids: the collate sorts tokens by physical centre time, so a
    grid's tokens are NOT contiguous along the patch axis. Alternate them to prove the block is
    contiguous in each grid's OWN order, not in the raw patch index."""
    ids = [0] * n_short + [1] * n_long
    ids = [x for pair in zip(ids[:min(n_short, n_long)], ids[n_short:]) for x in pair] \
        + ids[min(n_short, n_long):n_short] + ids[n_short + min(n_short, n_long):]
    return torch.tensor([ids])


def test_per_resolution_block_is_contiguous_within_each_grid():
    rids = _per_res_grid(12, 4)
    T = rids.shape[1]
    plan = make_per_resolution_mask_plan(
        rids, C=3, gyro_channels=None, generator=gen(3), channel_event_p=0.0,
        valid_patches=torch.ones(1, T, dtype=torch.bool),
        channel_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    masked = plan.token_mask[0, :, 0]
    for group in (0, 1):
        hits = [bool(masked[p]) for p in range(T) if int(rids[0, p]) == group]
        runs = sum(1 for i in range(1, len(hits)) if hits[i] and not hits[i - 1]) + int(hits[0])
        assert runs == 1, f"grid {group} masked in {runs} runs, expected one contiguous block"
        assert any(hits) and not all(hits)


def test_per_resolution_realises_the_requested_ratio_in_each_grid():
    """Counting each block in tokens makes the ratio independent of patch duration."""
    rids = _per_res_grid(12, 4)
    T = rids.shape[1]
    for ratio, want_short, want_long in ((0.5, 6, 2), (0.75, 9, 3)):
        plan = make_per_resolution_mask_plan(
            rids, C=3, gyro_channels=None, generator=gen(4), channel_event_p=0.0,
            time_ratio=ratio, valid_patches=torch.ones(1, T, dtype=torch.bool),
            channel_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        masked = plan.token_mask[0, :, 0]
        got = {g: sum(bool(masked[p]) for p in range(T) if int(rids[0, p]) == g) for g in (0, 1)}
        assert got == {0: want_short, 1: want_long}, (ratio, got)


def test_per_resolution_masks_a_one_token_grid_in_full():
    """The sp_sw_har case: the other temporal grid contextualizes a one-token resolution."""
    rids = torch.tensor([[0, 1, 0]])            # 2 short tokens, 1 long token
    plan = make_per_resolution_mask_plan(
        rids, C=3, gyro_channels=None, generator=gen(5), channel_event_p=0.0,
        valid_patches=torch.ones(1, 3, dtype=torch.bool),
        channel_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    # ALL channels, not merely any: this is temporal rather than channel-only supervision.
    assert bool(plan.token_mask[0, 1].all()), "the single long token got no temporal mask"
    assert not bool(plan.token_mask[0].all()), "nothing visible left to predict from"


def test_per_resolution_always_leaves_one_visible_real_token():
    for seed in range(25):
        rids = torch.tensor([[0, 1]])           # smallest possible grid: one token each
        plan = make_per_resolution_mask_plan(
            rids, C=2, gyro_channels=None, generator=gen(seed), channel_event_p=1.0,
            time_ratio=0.9, valid_patches=torch.ones(1, 2, dtype=torch.bool),
            channel_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        assert not bool(plan.token_mask.all()), f"seed {seed} masked every real token"


def test_per_resolution_never_masks_padded_or_absent_entries():
    rids = torch.tensor([[0, 0, 1, -1]])
    valid = torch.tensor([[True, True, True, False]])
    channels = torch.tensor([[True, True, False]])
    plan = make_per_resolution_mask_plan(
        rids, C=3, gyro_channels=None, generator=gen(6), channel_event_p=1.0,
        valid_patches=valid, channel_mask=channels,
    )
    assert not bool(plan.token_mask[0, 3].any())        # padded patch
    assert not bool(plan.token_mask[..., 2].any())      # absent channel


def test_gyro_triad_dropped_jointly():
    """Modality drops take the WHOLE triad; a single-channel drop may hit one gyro axis
    (dead-channel event), so the joint-unity property is: >=2 gyro channels fully
    dropped implies all 3 (only the triad event can drop more than one)."""
    plan = make_mask_plan(B=512, T=8, C=6, gyro_channels=GYRO, generator=gen(2))
    full_channel = plan.token_mask.all(dim=1)            # (B, C) masked at EVERY t
    gyro_count = full_channel[:, GYRO].sum(dim=1)
    assert (gyro_count == 3).sum() > 0, "gyro triad drops never happened at B=512"
    assert not bool((gyro_count == 2).any()), "2-of-3 gyro drop: triad event not joint"


def test_validity_aware_mask_guarantees_supervision():
    """With valid_patches + channel_mask, every window with >=2 real patches must get
    at least one masked REAL token (the JEPA zero-supervision fix), and no non-real token
    is ever masked."""
    B, T, C = 256, 6, 6
    g = gen(5)
    # random per-window real-patch counts (1..6) and accel-only vs full-IMU
    usable = torch.randint(1, T + 1, (B,), generator=g)
    valid_patches = torch.arange(T).unsqueeze(0) < usable.unsqueeze(1)      # (B,T) prefix
    channel_mask = torch.ones(B, C, dtype=torch.bool)
    accel_only = torch.rand(B, generator=g) < 0.6
    channel_mask[accel_only, 3:] = False                                    # drop gyro
    plan = make_mask_plan(B, T, C, GYRO, generator=g,
                          valid_patches=valid_patches, channel_mask=channel_mask)
    real = valid_patches.unsqueeze(2) & channel_mask.unsqueeze(1)
    # no non-real token masked
    assert not bool((plan.token_mask & ~real).any())
    # every window with >=2 real patches has >=1 masked real token
    sup = (plan.token_mask & real).flatten(1).sum(1)
    assert (sup[usable >= 2] >= 1).all(), "zero JEPA supervision on a >=2-patch window"


# --------------------------------------------------------------------------- VICReg / EMA targets
def test_vicreg_prefers_aligned_pairs_without_negative_mining():
    torch.manual_seed(4)
    z = torch.randn(32, 16)
    aligned = vicreg(z, z + 0.01 * torch.randn_like(z))
    shuffled = vicreg(z, z.roll(1, dims=0))
    assert aligned.total < shuffled.total
    assert torch.isfinite(aligned.total)


def test_vicreg_penalizes_collapsed_features():
    diverse = torch.randn(64, 16)
    collapsed = torch.zeros_like(diverse)
    good = vicreg(diverse, diverse.clone())
    bad = vicreg(collapsed, collapsed.clone())
    assert bad.variance > good.variance
    assert bad.min_std < 0.02


def test_vicreg_gradient_flows_to_both_views():
    a = torch.randn(16, 8, requires_grad=True)
    b = torch.randn(16, 8, requires_grad=True)
    vicreg(a, b).total.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()


def test_masked_ema_latent_is_stop_gradient_and_masked():
    pred = torch.randn(2, 3, 2, 8, requires_grad=True)
    target = pred.detach().clone().requires_grad_(True)
    target.data[:, 0] *= -1
    mask = torch.zeros(2, 3, 2, dtype=torch.bool)
    mask[:, 1:] = True
    loss = masked_ema_latent_loss(pred, target, mask)
    assert loss < 1e-6
    loss.backward()
    assert pred.grad is not None
    assert target.grad is None


def test_masked_ema_latent_weights_resolutions_equally():
    target = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]])
    pred = target.clone()
    pred[:, 3] = torch.tensor([-1.0, 0.0])
    mask = torch.ones(1, 4, 1, dtype=torch.bool)
    groups = torch.tensor([[0, 0, 0, 1]])
    loss = masked_ema_latent_loss(pred, target, mask, token_groups=groups)
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_masked_ema_latent_duration_weights_partial_tail():
    target = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    pred = target.clone()
    pred[:, 1] = torch.tensor([-1.0, 0.0])
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    durations = torch.tensor([[1.0, 0.1]])
    loss = masked_ema_latent_loss(pred, target, mask, token_durations=durations)
    assert torch.allclose(loss, torch.tensor(2.0 / 11.0), atol=1e-6)


def test_phase_a_loss_has_exactly_two_weighted_terms():
    jepa = torch.tensor(2.0)
    vicreg_loss = torch.tensor(3.0)
    out = phase_a_loss(jepa, vicreg_loss, jepa_weight=0.5, vicreg_weight=2.0)
    assert set(out.terms) == {"jepa", "vicreg"}
    assert out.total == pytest.approx(7.0)


def test_ema_teacher_updates_without_gradients_and_roundtrips_state():
    student = torch.nn.Linear(3, 2, bias=False)
    teacher = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
        teacher.weight.zero_()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    update_ema_encoder(student, teacher, decay=0.5)
    assert torch.allclose(teacher.weight, torch.ones_like(teacher.weight))
    assert not teacher.weight.requires_grad

    restored = torch.nn.Linear(3, 2, bias=False)
    restored.load_state_dict(teacher.state_dict())
    assert torch.equal(restored.weight, teacher.weight)


def test_ema_accepts_a_fully_frozen_teacher():
    """decay == 1.0 must be legal: BYOL's cosine ramp lands on it exactly at the final step.

    Rejecting it crashed every --jepa-ema-schedule cosine run at step == steps, after the last
    checkpoint was written. The CLI still rejects 1.0 as a FIXED decay, where it would mean a
    randomly-initialised teacher for the whole run.
    """
    student, teacher = torch.nn.Linear(3, 2, bias=False), torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
        teacher.weight.zero_()
    update_ema_encoder(student, teacher, decay=1.0)
    assert torch.equal(teacher.weight, torch.zeros_like(teacher.weight))
    with pytest.raises(ValueError):
        update_ema_encoder(student, teacher, decay=1.0 + 1e-6)


def test_foreach_ema_reuses_a_validated_parameter_mapping():
    student = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(1.0)
    before = [parameter.clone() for parameter in teacher.parameters()]
    update_ema_encoder(student, teacher, decay=0.25)
    cache = teacher._ema_update_cache
    update_ema_encoder(student, teacher, decay=0.25)
    assert teacher._ema_update_cache is cache
    for initial, source, target in zip(before, student.parameters(), teacher.parameters()):
        expected = (0.25 ** 2) * initial + (1.0 - 0.25 ** 2) * source
        assert torch.allclose(target, expected)


def test_default_expander_is_the_historical_control_architecture():
    """d_model -> d_model -> 128. Adding a width knob must not silently rewrite the default.

    Folding hidden and output width into one flag turned the unchanged default command into
    256 -> 128 -> 128 (49,408 params vs 98,688), so it stopped being control-equivalent to every
    earlier run it was being compared against.
    """
    projector = PipelineAModel(PretrainConfig(jepa_mode="masked")).vicreg_projector
    assert [projector[0].in_features, projector[0].out_features] == [256, 256]
    assert [projector[2].in_features, projector[2].out_features] == [256, 128]
    assert sum(p.numel() for p in projector.parameters()) == 98_688


def test_objective_gradient_geometry_reports_direction_and_scale():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    module = torch.nn.Module()
    module.register_parameter("value", parameter)
    losses = {
        "parallel": 2.0 * parameter[0],
        "same": 3.0 * parameter[0],
        "orthogonal": 4.0 * parameter[1],
    }
    geometry = objective_encoder_grad_geometry(losses, module)
    assert geometry["norms"] == pytest.approx({
        "parallel": 2.0, "same": 3.0, "orthogonal": 4.0,
    })
    assert geometry["cosines"]["parallel|same"] == pytest.approx(1.0)
    assert geometry["cosines"]["parallel|orthogonal"] == pytest.approx(0.0)


def test_objective_calibration_hits_share_and_preserves_scale():
    # Unit gradients are orthogonal with norms JEPA=1 and VICReg=4.
    sample = {
        "norms": {"jepa": 1.0, "vicreg": 4.0},
        "dots": {"jepa|vicreg": 0.0},
        "cosines": {"jepa|vicreg": 0.0},
    }
    report = recommend_objective_weights(
        [sample] * 5,
        current_jepa_weight=1.0,
        current_vicreg_weight=1.0,
        target_jepa_share=0.5,
    )
    recommended = report["recommended"]
    weighted_jepa = recommended["jepa_weight"] * 1.0
    weighted_vicreg = recommended["vicreg_weight"] * 4.0
    assert weighted_jepa / (weighted_jepa + weighted_vicreg) == pytest.approx(0.5)
    norms = report["median_combined_encoder_grad_norm"]
    assert norms["recommended_weights"] == pytest.approx(norms["pilot_weights"])


def test_objective_calibration_rejects_missing_vicreg_signal():
    sample = {
        "norms": {"jepa": 1.0, "vicreg": 0.0},
        "dots": {"jepa|vicreg": 0.0},
        "cosines": {"jepa|vicreg": 0.0},
    }
    with pytest.raises(ValueError, match="non-zero calibration gradients"):
        recommend_objective_weights(
            [sample], current_jepa_weight=1.0, current_vicreg_weight=1.0,
        )
