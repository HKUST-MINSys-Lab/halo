"""Gate tests for the sensor-granularity design of record.

These assert the properties the design DEPENDS on, not merely that shapes line up:

  * a dead axis must be distinguishable from an axis that read zero (mhealth's degenerate gyro);
  * the fold must honour the existing `sensor_id` contract, where an accel-only stream maps ALL six
    channel slots to sensor 0 and relies on `channel_mask`;
  * whole-sensor JEPA masking must NEVER cross a placement boundary — the constraint that separates
    the well-posed same-site objective from the ill-posed cross-placement one deleted 2026-08-06;
  * a sensor must never have its signal AND its descriptor hidden at once (nothing left to
    reconstruct from);
  * descriptor scoring must be retrieval-based, and must not turn duplicate descriptions into each
    other's negatives.
"""

from __future__ import annotations

import torch

from model.tokenizer.sensor_tokens import (
    GRAVITY_NOT_APPLICABLE, GRAVITY_PRESENT,
    MODALITY_ACCELEROMETER, MODALITY_GYROSCOPE,
    ConditioningProjection, DescriptorHead, SensorFold, descriptor_retrieval_loss,
)
from training.tokenizer.losses_repr import make_sensor_mask_plan
from training.tokenizer.pretrain_data import SENSOR_BIAS_DIM


D = 32
SEED = 20260811


def _conditioning(batch: int, *, accel_only: bool = False, rate: float = 50.0) -> dict:
    modalities = [MODALITY_ACCELEROMETER]
    gravity = [GRAVITY_PRESENT]
    if not accel_only:
        modalities.append(MODALITY_GYROSCOPE)
        gravity.append(GRAVITY_NOT_APPLICABLE)
    return {
        "sensor_modality": torch.tensor([modalities] * batch),
        "sensor_gravity": torch.tensor([gravity] * batch),
        "sensor_rates_hz": torch.tensor([[[rate, rate]] * len(modalities)] * batch),
    }


# ----------------------------------------------------------------------------- SensorFold
def _fold_inputs(B=2, P=4, C=6):
    tokens = torch.randn(B, P, C, D)
    sensor_id = torch.tensor([[0, 0, 0, 1, 1, 1]] * B)
    channel_mask = torch.ones(B, C, dtype=torch.bool)
    return tokens, sensor_id, channel_mask


def test_fold_produces_one_token_per_sensor():
    tokens, sid, cm = _fold_inputs()
    fold = SensorFold(D)
    out, mask = fold(tokens, sid, cm)
    assert out.shape == (2, 4, 2, D)
    assert mask.all()


def test_absent_axis_is_distinguishable_from_zero_valued_axis():
    """The validity indicator, not just zeroing — otherwise a dead sensor reads as a resting one."""
    tokens, sid, cm = _fold_inputs()
    fold = SensorFold(D).eval()
    cm_dead = cm.clone()
    cm_dead[0, 4] = False                       # gyro-y absent
    tokens_zero = tokens.clone()
    tokens_zero[0, :, 4] = 0.0                  # gyro-y present but reading zero
    with torch.no_grad():
        absent, _ = fold(tokens, sid, cm_dead)
        zeroed, _ = fold(tokens_zero, sid, cm)
    assert not torch.allclose(absent[0, :, 1], zeroed[0, :, 1])


def test_accel_only_stream_honours_the_existing_sensor_id_contract():
    """`stream_sensor_texts` maps ALL six slots to sensor 0 when gyro is absent.

    Absent channels keep a valid index and rely on `channel_mask`. The fold must count only LIVE
    channels; counting all six would make sensor 0 own six channels and raise.
    """
    tokens, _, cm = _fold_inputs()
    sid_all_zero = torch.zeros(2, 6, dtype=torch.long)
    cm_accel = cm.clone()
    cm_accel[:, 3:] = False
    out, mask = SensorFold(D)(tokens, sid_all_zero, cm_accel, n_sensors=1)
    assert out.shape == (2, 4, 1, D)
    assert mask.all()


def test_fold_rejects_a_sensor_with_too_many_live_channels():
    tokens, _, cm = _fold_inputs()
    sid = torch.zeros(2, 6, dtype=torch.long)       # 6 LIVE channels on one sensor
    try:
        SensorFold(D)(tokens, sid, cm, n_sensors=1)
    except ValueError as exc:
        assert "more than 3 live channels" in str(exc)
    else:                                            # pragma: no cover
        raise AssertionError("expected a ValueError for a 6-live-channel sensor")


def test_absent_sensor_reports_absent():
    tokens, sid, cm = _fold_inputs()
    cm2 = cm.clone()
    cm2[0, 3:] = False
    _, mask = SensorFold(D)(tokens, sid, cm2)
    assert mask[0].tolist() == [True, False]
    assert mask[1].tolist() == [True, True]


# ------------------------------------------------------------------- ConditioningProjection
def test_conditioning_is_do_no_harm_at_init():
    """Negative gate bias => the conditioner barely moves the token before training."""
    torch.manual_seed(SEED)
    proj = ConditioningProjection(9, D).eval()
    tokens = torch.randn(2, 4, 2, D)
    with torch.no_grad():
        out = proj(tokens, torch.randn(2, 2, 9), torch.ones(2, 2, dtype=torch.bool))
    assert (out - tokens).abs().max() < 0.5


def test_conditioning_is_suppressed_for_absent_sensors():
    torch.manual_seed(SEED)
    proj = ConditioningProjection(9, D).eval()
    tokens = torch.randn(2, 4, 2, D)
    valid = torch.tensor([[True, False], [True, True]])
    with torch.no_grad():
        out = proj(tokens, torch.randn(2, 2, 9), valid)
    assert torch.allclose(out[0, :, 1], tokens[0, :, 1])       # absent sensor untouched


# ------------------------------------------------------------------------ descriptor scoring
def test_descriptor_loss_collapses_duplicate_descriptions():
    """Two rows sharing a sensor description must not become each other's negatives."""
    torch.nn.functional.normalize
    target = torch.nn.functional.normalize(torch.randn(1, 384), dim=-1).repeat(4, 1)
    pred = torch.nn.functional.normalize(torch.randn(4, 384), dim=-1)
    loss, acc = descriptor_retrieval_loss(pred, target)
    assert float(loss) == 0.0 and float(acc) == 1.0


def test_descriptor_loss_integer_ids_match_float_deduplication():
    """The optimized ID path must preserve the original descriptor-retrieval objective exactly."""
    torch.manual_seed(SEED)
    candidates = torch.nn.functional.normalize(torch.randn(3, 384), dim=-1)
    ids = torch.tensor([2, 0, 2, 1, 0])
    targets = candidates.index_select(0, ids)
    predicted = torch.nn.functional.normalize(torch.randn(5, 384), dim=-1)
    reference = descriptor_retrieval_loss(predicted, targets)
    optimized = descriptor_retrieval_loss(
        predicted, targets, target_ids=ids, candidate_descriptors=candidates,
    )
    assert torch.allclose(reference[0], optimized[0])
    assert torch.equal(reference[1], optimized[1])


def test_descriptor_loss_fixed_shape_mask_matches_compact_all_candidate_scoring():
    torch.manual_seed(SEED)
    candidates = torch.nn.functional.normalize(torch.randn(4, 384), dim=-1)
    ids = torch.tensor([[0, 1], [2, 3], [1, -1]])
    predicted = torch.nn.functional.normalize(torch.randn(3, 2, 384), dim=-1)
    selected = torch.tensor([[True, False], [False, True], [True, False]])

    loss, acc = descriptor_retrieval_loss(
        predicted, candidates[ids.clamp_min(0)], target_ids=ids,
        candidate_descriptors=candidates, row_mask=selected,
    )
    rows = predicted[selected]
    targets = ids[selected]
    logits = rows @ candidates.t() / 0.07
    expected_loss = torch.nn.functional.cross_entropy(logits, targets)
    expected_acc = logits.argmax(dim=1).eq(targets).float().mean()

    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(acc, expected_acc)


def test_descriptor_head_emits_unit_vectors_in_frozen_text_space():
    head = DescriptorHead(D).eval()
    with torch.no_grad():
        out = head(torch.randn(2, 3, D))
    assert out.shape == (2, 3, 384)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2, 3), atol=1e-5)


# ------------------------------------------------------------------------- sensor mask plan
def _plan(placement, present=None, B=256, T=6, S=4, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    present = present if present is not None else torch.ones(B, S, dtype=torch.bool)
    return make_sensor_mask_plan(B, T, S, generator=g, sensor_present=present,
                                 sensor_placement=placement.expand(B, S).contiguous())


def test_whole_sensor_masking_never_crosses_a_placement():
    """THE constraint. A sensor alone at its placement has no co-tenant to be predicted from."""
    plan = _plan(torch.tensor([[0, 1, 2, 3]]))          # every sensor alone
    assert int(plan.token_mask.all(dim=1).sum()) == 0


def test_whole_sensor_masking_only_hits_co_tenants():
    plan = _plan(torch.tensor([[0, 0, 1, 2]]))          # only sensors 0,1 share a placement
    per_sensor = plan.token_mask.all(dim=1).sum(dim=0)
    assert per_sensor[2] == 0 and per_sensor[3] == 0
    assert per_sensor[0] > 0 and per_sensor[1] > 0


def test_signal_and_descriptor_are_never_both_hidden():
    plan = _plan(torch.tensor([[0, 0, 1, 1]]))
    assert int((plan.token_mask.all(dim=1) & plan.descriptor_mask).sum()) == 0


def test_absent_sensors_are_never_masked():
    present = torch.ones(256, 4, dtype=torch.bool)
    present[:, 3] = False
    plan = _plan(torch.tensor([[0, 0, 1, 1]]), present=present)
    assert int(plan.token_mask[:, :, 3].sum()) == 0
    assert int(plan.descriptor_mask[:, 3].sum()) == 0


def test_mask_events_fire_at_roughly_the_configured_rate():
    plan = _plan(torch.tensor([[0, 0, 1, 1]]), B=2048)
    sensor_rate = float(plan.token_mask.all(dim=1).any(dim=1).float().mean())
    descriptor_rate = float(plan.descriptor_mask.any(dim=1).float().mean())
    assert 0.15 < sensor_rate < 0.35
    assert 0.15 < descriptor_rate < 0.35


def test_one_patch_two_modalities_always_gets_a_jepa_target():
    """Temporal masking is impossible at T=1, so co-located accel/gyro is the fallback task."""
    B = 512
    plan = make_sensor_mask_plan(
        B, 1, 2,
        generator=torch.Generator().manual_seed(91),
        sensor_present=torch.ones(B, 2, dtype=torch.bool),
        sensor_placement=torch.zeros(B, 2, dtype=torch.long),
        valid_patches=torch.ones(B, 1, dtype=torch.bool),
    )
    assert plan.token_mask.flatten(1).any(dim=1).all()
    assert (~plan.token_mask.flatten(1).all(dim=1)).all()


def test_one_patch_one_sensor_remains_honestly_ineligible_for_jepa():
    plan = make_sensor_mask_plan(
        32, 1, 1,
        generator=torch.Generator().manual_seed(92),
        sensor_present=torch.ones(32, 1, dtype=torch.bool),
        sensor_placement=torch.zeros(32, 1, dtype=torch.long),
        valid_patches=torch.ones(32, 1, dtype=torch.bool),
    )
    assert not plan.token_mask.any()


# --------------------------------------------------------------------------- encoder wiring
def test_sensor_encoder_forward_shapes_and_masking():
    from model.tokenizer.encoder import SetTokenizerEncoder

    B, P, C, S_PAD, N_TRUE = 2, 5, 6, 256, 64
    enc = SetTokenizerEncoder(d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
                              dropout=0.0, token_granularity="sensor",
                              sensor_bias_dim=SENSOR_BIAS_DIM, dft_size=S_PAD).eval()
    patches = torch.zeros(B, P, S_PAD, C)
    patches[:, :, :N_TRUE] = torch.randn(B, P, N_TRUE, C)
    sensor_texts = [["a phone accelerometer on the front pocket; includes gravity",
                     "a phone gyroscope on the front pocket"]] * B
    role = [["x", "y", "z", "x", "y", "z"]] * B
    sid = torch.tensor([[0, 0, 0, 1, 1, 1]] * B)
    pos = torch.arange(P).float().unsqueeze(0).expand(B, P).contiguous()
    cm = torch.ones(B, C, dtype=torch.bool)
    bias = torch.randn(B, 2, SENSOR_BIAS_DIM)

    with torch.no_grad():
        out = enc(patches, 50.0, N_TRUE, role, pos, channel_mask=cm, sensor_texts=sensor_texts,
                  sensor_id=sid, sensor_bias=bias, **_conditioning(B))
    assert out["tokens"].shape == (B, P, 2, 64)          # per SENSOR, not per channel
    assert out["descriptor"].shape == (B, 2, 384)
    assert out["descriptor_pred"].shape == (B, 2, 384)
    assert out["sensor_present"].all()

    token_mask = torch.zeros(B, P, 2, dtype=torch.bool)
    token_mask[:, :, 1] = True
    with torch.no_grad():
        masked = enc(patches, 50.0, N_TRUE, role, pos, channel_mask=cm,
                     sensor_texts=sensor_texts, sensor_id=sid, sensor_bias=bias,
                     token_mask=token_mask,
                     descriptor_mask=torch.tensor([[True, False]] * B), **_conditioning(B))
    assert not torch.allclose(out["pooled"], masked["pooled"])


def test_per_sensor_row_export_doubles_six_channel_streams():
    """Bank layout: one row per (patch, SENSOR). 6-channel -> 2x, accel-only -> 1x, never phantom."""
    from model.tokenizer.encoder import SetTokenizerEncoder
    from training.tokenizer.eval_transfer import encode_dataset_detailed

    enc = SetTokenizerEncoder(d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
                              dropout=0.0, token_granularity="sensor",
                              sensor_bias_dim=SENSOR_BIAS_DIM).eval()
    g = torch.Generator().manual_seed(SEED)
    data = torch.randn(6, 300, 6, generator=g).numpy()
    texts = [f"c{i}" for i in range(6)]

    both = encode_dataset_detailed(enc, data, texts, torch.device("cpu"), 50.0,
                                   gravity_state="present", channel_mask=[True] * 6,
                                   dataset="wisdm", stream="phone_pocket",
                                   export_sensor_rows=True)
    assert both["sensor_Z"].shape[0] == 2 * both["patch_Z"].shape[0]
    assert sorted(set(both["sensor_slot"].tolist())) == [0, 1]
    # Every per-sensor column must be the same length or a consumer silently mispairs rows.
    n = both["sensor_Z"].shape[0]
    for key in ("sensor_window", "sensor_slot", "sensor_time", "sensor_duration",
                "sensor_resolution"):
        assert both[key].shape[0] == n, key

    accel_only = encode_dataset_detailed(enc, data, texts, torch.device("cpu"), 50.0,
                                         gravity_state="present",
                                         channel_mask=[True, True, True, False, False, False],
                                         dataset="capture24", stream="watch_wrist",
                                         export_sensor_rows=True)
    assert accel_only["sensor_Z"].shape[0] == accel_only["patch_Z"].shape[0]
    assert sorted(set(accel_only["sensor_slot"].tolist())) == [0]      # no phantom gyroscope


def test_channel_granularity_exports_no_sensor_rows():
    """The flag must be inert on a channel-granularity encoder, not silently emit patch rows."""
    from model.tokenizer.encoder import SetTokenizerEncoder
    from training.tokenizer.eval_transfer import encode_dataset_detailed

    enc = SetTokenizerEncoder(d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
                              dropout=0.0, text_conditioning="factored").eval()
    g = torch.Generator().manual_seed(SEED)
    data = torch.randn(4, 300, 6, generator=g).numpy()
    out = encode_dataset_detailed(enc, data, [f"c{i}" for i in range(6)], torch.device("cpu"),
                                  50.0, gravity_state="present", channel_mask=[True] * 6,
                                  dataset="wisdm", stream="phone_pocket",
                                  export_sensor_rows=True)
    assert out["sensor_Z"].shape[0] == 0
    assert out["patch_Z"].shape[0] > 0


def test_sensor_encoder_handles_accel_only_streams():
    from model.tokenizer.encoder import SetTokenizerEncoder

    B, P, C, S_PAD, N_TRUE = 2, 5, 6, 256, 64
    enc = SetTokenizerEncoder(d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
                              dropout=0.0, token_granularity="sensor",
                              sensor_bias_dim=SENSOR_BIAS_DIM, dft_size=S_PAD).eval()
    patches = torch.zeros(B, P, S_PAD, C)
    patches[:, :, :N_TRUE] = torch.randn(B, P, N_TRUE, C)
    cm = torch.ones(B, C, dtype=torch.bool)
    cm[:, 3:] = False
    pos = torch.arange(P).float().unsqueeze(0).expand(B, P).contiguous()
    with torch.no_grad():
        out = enc(patches, 50.0, N_TRUE, [["x", "y", "z", "x", "y", "z"]] * B, pos,
                  channel_mask=cm,
                  sensor_texts=[["a watch accelerometer on the wrist; includes gravity"]] * B,
                  sensor_id=torch.zeros(B, C, dtype=torch.long),
                  sensor_bias=torch.randn(B, 1, SENSOR_BIAS_DIM),
                  **_conditioning(B, accel_only=True))
    assert out["tokens"].shape == (B, P, 1, 64)
    assert out["sensor_present"].all()


def test_retrieval_rows_are_sensor_isolated():
    """Adding a co-located gyroscope must not rewrite the accelerometer evidence row."""
    from model.tokenizer.encoder import SetTokenizerEncoder

    torch.manual_seed(18)
    enc = SetTokenizerEncoder(
        d_model=64, num_layers=2, num_heads=4, dim_feedforward=128, dropout=0.0,
        token_granularity="sensor", use_sensor_isolated_retrieval=True, dft_size=256,
    ).eval()
    patches = torch.randn(2, 4, 256, 6)
    positions = torch.arange(4).float().expand(2, 4)
    sensor_id = torch.tensor([[0, 0, 0, 1, 1, 1]] * 2)
    common = dict(
        sampling_rate_hz=50.0, patch_len_samples=64,
        channel_texts=[["x", "y", "z", "x", "y", "z"]] * 2,
        positions=positions,
        patch_padding_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    with torch.no_grad():
        both = enc(
            patches, channel_mask=torch.ones(2, 6, dtype=torch.bool),
            sensor_texts=[["accelerometer at wrist", "gyroscope at wrist"]] * 2,
            sensor_id=sensor_id,
            **_conditioning(2),
            **common,
        )
        accel = enc(
            patches, channel_mask=torch.tensor([[1, 1, 1, 0, 0, 0]] * 2).bool(),
            sensor_texts=[["accelerometer at wrist"]] * 2,
            sensor_id=torch.zeros(2, 6, dtype=torch.long),
            **_conditioning(2, accel_only=True),
            **common,
        )
    assert torch.allclose(
        both["retrieval_tokens"][:, :, 0], accel["retrieval_tokens"][:, :, 0],
        atol=1e-6, rtol=1e-6,
    )


def test_retrieval_only_shortcut_matches_full_forward_rows():
    """Skipping the Phase-A trunk must be an exact compute optimization, not a new representation."""
    from model.tokenizer.encoder import SetTokenizerEncoder

    torch.manual_seed(23)
    enc = SetTokenizerEncoder(
        d_model=32, num_layers=2, num_heads=4, dim_feedforward=64, dropout=0.0,
        token_granularity="sensor", use_sensor_isolated_retrieval=True, dft_size=256,
    ).eval()
    patches = torch.randn(2, 3, 256, 6)
    kwargs = dict(
        sampling_rate_hz=50.0, patch_len_samples=64,
        channel_texts=[["x", "y", "z", "x", "y", "z"]] * 2,
        positions=torch.arange(3).float().expand(2, 3),
        channel_mask=torch.ones(2, 6, dtype=torch.bool),
        patch_padding_mask=torch.ones(2, 3, dtype=torch.bool),
        sensor_texts=[["accelerometer at wrist", "gyroscope at wrist"]] * 2,
        sensor_id=torch.tensor([[0, 0, 0, 1, 1, 1]] * 2),
        **_conditioning(2),
    )
    with torch.no_grad():
        full = enc(patches, **kwargs)
        short = enc(patches, retrieval_only=True, **kwargs)
    assert set(short) == {"retrieval_tokens", "sensor_present", "descriptor"}
    assert torch.equal(short["sensor_present"], full["sensor_present"])
    assert torch.allclose(short["descriptor"], full["descriptor"], atol=0, rtol=0)
    assert torch.allclose(
        short["retrieval_tokens"], full["retrieval_tokens"], atol=1e-6, rtol=1e-6,
    )


def test_new_sensor_encoder_has_no_source_statistics_projection():
    from model.tokenizer.encoder import SetTokenizerEncoder

    enc = SetTokenizerEncoder(token_granularity="sensor")
    assert not enc.use_sensor_bias_conditioning
    assert not hasattr(enc, "bias_proj")


def test_multiresolution_sensor_pooling_is_duration_weighted():
    """A short tail patch must contribute in proportion to physical time, not as a full patch."""
    from model.tokenizer.encoder import SetTokenizerEncoder

    B, P, C, S_PAD, N_TRUE = 1, 4, 6, 256, 64
    enc = SetTokenizerEncoder(d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
                              dropout=0.0, token_granularity="sensor",
                              sensor_bias_dim=SENSOR_BIAS_DIM, dft_size=S_PAD).eval()
    patches = torch.zeros(B, P, S_PAD, C)
    patches[:, :, :N_TRUE] = torch.randn(B, P, N_TRUE, C)
    positions = torch.tensor([[0.25, 0.6, 0.5, 1.1]])
    durations = torch.tensor([[0.5, 0.2, 1.0, 0.2]])
    resolution_ids = torch.tensor([[0, 0, 1, 1]])
    with torch.no_grad():
        out = enc(
            patches, 50.0, N_TRUE, [["x", "y", "z", "x", "y", "z"]], positions,
            patch_durations=durations, resolution_ids=resolution_ids,
            channel_mask=torch.ones(B, C, dtype=torch.bool),
            patch_padding_mask=torch.ones(B, P, dtype=torch.bool),
            sensor_texts=[["accelerometer at wrist", "gyroscope at wrist"]],
            sensor_id=torch.tensor([[0, 0, 0, 1, 1, 1]]),
            sensor_bias=torch.randn(B, 2, SENSOR_BIAS_DIM),
            **_conditioning(B),
        )
    per_patch = out["per_patch"]
    short = (per_patch[:, :2] * durations[:, :2, None]).sum(1) / durations[:, :2].sum(1, keepdim=True)
    long = (per_patch[:, 2:] * durations[:, 2:, None]).sum(1) / durations[:, 2:].sum(1, keepdim=True)
    expected = (short + long) / 2
    assert torch.allclose(out["pooled"], expected, atol=1e-5, rtol=1e-5)
    tokens = out["tokens"]
    short_sensor = (tokens[:, :2] * durations[:, :2, None, None]).sum(1) \
        / durations[:, :2].sum(1).view(B, 1, 1)
    long_sensor = (tokens[:, 2:] * durations[:, 2:, None, None]).sum(1) \
        / durations[:, 2:].sum(1).view(B, 1, 1)
    assert torch.allclose(out["sensor_context"], (short_sensor + long_sensor) / 2,
                          atol=1e-5, rtol=1e-5)


def test_sensor_tokens_receive_trainable_duration_identity():
    from model.tokenizer.encoder import SetTokenizerEncoder

    enc = SetTokenizerEncoder(
        d_model=16, num_layers=1, num_heads=4, dim_feedforward=32, dropout=0.0,
        token_granularity="sensor", use_duration_embedding=True,
        duration_min_seconds=0.5, duration_max_seconds=1.5, num_resolutions=3,
    )
    tokens = torch.zeros(1, 3, 2, 16, requires_grad=True)
    durations = torch.tensor([[0.5, 1.0, 1.5]])
    conditioned = enc._add_duration_embedding(tokens, durations)
    assert not torch.allclose(conditioned[:, 0], conditioned[:, 1])
    assert not torch.allclose(conditioned[:, 1], conditioned[:, 2])
    conditioned.square().sum().backward()
    assert enc.duration_gate_logit.grad is not None
    assert float(enc.duration_gate_logit.grad.abs()) > 0
    assert all(parameter.grad is not None for parameter in enc.duration_proj.parameters())


def test_checkpoint_reconstruction_preserves_sensor_design():
    from dataclasses import asdict

    from training.tokenizer.eval_transfer import build_encoder, encode_dataset_detailed
    from training.tokenizer.pretrain import PipelineAModel, PretrainConfig

    cfg = PretrainConfig(
        d_model=64, num_layers=2, num_heads=4, dim_feedforward=128, dropout=0.0,
        token_granularity="sensor", text_conditioning="factored", multiresolution=True,
        sensor_bias_dim=SENSOR_BIAS_DIM,
    )
    original = PipelineAModel(cfg).encoder.eval()
    restored = build_encoder(
        {"config": asdict(cfg), "encoder": original.state_dict()}, torch.device("cpu"),
    )
    assert restored.token_granularity == "sensor"
    assert restored.sensor_bias_dim == SENSOR_BIAS_DIM
    assert restored.multiresolution is True
    assert restored.use_duration_embedding is True
    assert restored.num_resolutions == 3
    assert restored.eval_resolutions == (0.5, 1.0, 2.0)
    assert restored.fusion is None

    data = torch.randn(2, 300, 6).numpy()
    out = encode_dataset_detailed(
        restored, data, [f"channel {i}" for i in range(6)], torch.device("cpu"), 50.0,
        gravity_state="present", channel_mask=[True] * 6,
        dataset="wisdm", stream="phone_pocket",
    )
    assert out["pooled"].shape == (2, 64)
    assert torch.isfinite(out["pooled"]).all()


# ---------------------------------------------------------------- one presence rule, two callers
# `stream_sensor_texts` fixes the sensor count N and the `sensor_id` map; `stream_sensor_bias` must
# return exactly N rows. These lived as two separate expressions, and eval_transfer's used `.all()`
# while the text path used `.any()` — so a PARTIAL TRIAD (some but not all axes live) produced a
# sensor with a description and no bias row. No native grid carries one today, which is exactly why
# the divergence went unnoticed; `modalities_present` is now the single rule both callers derive from.
def test_modalities_present_counts_a_partial_triad_as_present():
    from training.tokenizer.pretrain_data import modalities_present

    # Two accel axes live, third dead — still an accelerometer. SensorFold's axis-validity
    # indicator is what handles the dead axis; dropping the sensor entirely would lose the other two.
    assert modalities_present([True, True, False, False, False, False]) == ["accel"]
    assert modalities_present([False] * 3 + [False, True, True]) == ["gyro"]
    assert modalities_present([True, False, False, False, False, True]) == ["accel", "gyro"]
    assert modalities_present([False] * 6) == []


def test_modalities_present_rejects_a_wrong_width_mask():
    from training.tokenizer.pretrain_data import modalities_present

    for bad in ([True] * 3, [True] * 7, []):
        try:
            modalities_present(bad)
        except ValueError:
            continue
        raise AssertionError(f"a {len(bad)}-slot mask must be rejected, not silently reinterpreted")


def test_bias_rows_match_sensor_texts_for_every_mask_pattern():
    """The invariant the encoder's shape check enforces, over all 64 masks including partial triads."""
    from itertools import product

    from training.tokenizer.pretrain_data import (modalities_present, stream_sensor_bias,
                                                  stream_sensor_texts)

    checked = 0
    for bits in product([False, True], repeat=6):
        modalities = modalities_present(list(bits))
        if not modalities:
            continue                                   # no live channel: no sensor, nothing to align
        _, sensor_texts, sensor_id = stream_sensor_texts(
            "wisdm", "phone_pocket",
            has_accel="accel" in modalities, has_gyro="gyro" in modalities,
        )
        bias = stream_sensor_bias("wisdm", "phone_pocket", modalities)
        assert bias.shape[0] == len(sensor_texts), (bits, bias.shape, sensor_texts)
        assert bias.shape[1] == SENSOR_BIAS_DIM
        assert int(max(sensor_id)) < len(sensor_texts), (bits, sensor_id)
        checked += 1
    assert checked == 63                               # every mask but the all-dead one
