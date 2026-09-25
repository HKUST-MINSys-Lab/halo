"""Matched-corpus M2 encoder shims — unit tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.tokenizer.matched_encoder import (
    BACKBONE_CONTRACTS,
    MatchedCorpusEncoder,
    build_matched_encoder,
    reconstruct_native_window,
)

ARCHS = ("limubert", "harnet")


def _batch(B=4, P=4, S=64, C=6, valid=50, rate=50.0):
    patches = torch.randn(B, P, S, C)
    # Padding beyond the valid length must never reach the trunk; make it obviously poisonous.
    patches[:, :, valid:, :] = 1e4
    return {
        "patches": patches,
        "patch_len": torch.full((B, P), valid, dtype=torch.long),
        "patch_padding_mask": torch.ones(B, P, dtype=torch.bool),
        "sensor_id": torch.tensor([[0, 0, 0, 1, 1, 1]] * B),
        "channel_mask": torch.ones(B, C, dtype=torch.bool),
        "rates": torch.full((B,), rate),
        "device_id": torch.zeros(B, 2, dtype=torch.long),
    }


def _forward(encoder, batch):
    return encoder(
        batch["patches"], batch["rates"], batch["patch_len"], None, None,
        channel_mask=batch["channel_mask"], patch_padding_mask=batch["patch_padding_mask"],
        sensor_id=batch["sensor_id"], device_id=batch["device_id"],
    )


# ------------------------------------------------------------ window reconstruction


def test_reconstruction_skips_dft_padding():
    """Patch buffers are padded to the DFT width; splicing that padding in would corrupt the signal."""
    B, P, S, valid = 2, 3, 16, 5
    patches = torch.zeros(B, P, S, 1)
    for p in range(P):
        patches[:, p, :valid, 0] = torch.arange(valid).float() + p * 10
        patches[:, p, valid:, 0] = -999.0
    window, total = reconstruct_native_window(
        patches, torch.full((B, P), valid, dtype=torch.long), torch.ones(B, P, dtype=torch.bool))
    assert total.tolist() == [P * valid] * B
    assert not (window == -999.0).any(), "padding leaked into the reconstructed signal"
    expected = torch.cat([torch.arange(valid).float() + p * 10 for p in range(P)])
    assert torch.allclose(window[0, :, 0], expected)


def test_reconstruction_honours_the_patch_padding_mask():
    B, P, S, valid = 1, 3, 8, 4
    patches = torch.zeros(B, P, S, 1)
    for p in range(P):
        patches[:, p, :valid, 0] = p + 1
    mask = torch.tensor([[True, False, True]])
    window, total = reconstruct_native_window(
        patches, torch.full((B, P), valid, dtype=torch.long), mask)
    assert int(total[0]) == 2 * valid, "a masked patch must contribute no samples"
    assert set(window[0, :2 * valid, 0].tolist()) == {1.0, 3.0}


# ------------------------------------------------------------------- the shim


@pytest.mark.parametrize("arch", ARCHS)
def test_forward_returns_the_support_classifier_contract(arch):
    encoder = build_matched_encoder(arch)
    batch = _batch()
    out = _forward(encoder, batch)
    assert out["pooled"].shape == (4, encoder.d_model)
    assert torch.isfinite(out["pooled"]).all()
    assert out["sensor_present"].shape[0] == 4
    assert out["device_present"].dim() == 3
    assert out["descriptor"].shape[0] == 4


@pytest.mark.parametrize("arch", ARCHS)
def test_trunk_is_trainable(arch):
    """M2 trains the backbone. A frozen trunk would silently train the projection alone."""
    encoder = build_matched_encoder(arch)
    out = _forward(encoder, _batch())
    torch.manual_seed(0)
    # A plain sum has zero gradient through the final LayerNorm, so weight the output randomly.
    (out["pooled"] * torch.randn_like(out["pooled"])).sum().backward()
    grads = [p.grad for p in encoder.net.parameters()]
    moved = sum(1 for g in grads if g is not None and float(g.abs().sum()) > 0)
    # Not every tensor is reached by every batch (unused positional-embedding rows, for instance),
    # but a trunk that is training must move most of its parameters.
    assert moved > len(grads) // 2, f"only {moved}/{len(grads)} trunk parameters received gradient"


@pytest.mark.parametrize("arch", ARCHS)
def test_random_init_does_not_zero_normalisation_scale(arch):
    """A zero BatchNorm gamma outputs zeros and looks like an architecture failure."""
    encoder = build_matched_encoder(arch)
    for module in encoder.net.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
            if getattr(module, "weight", None) is not None:
                assert float(module.weight.abs().sum().detach()) > 0, \
                    "normalisation scale was zeroed"
    out = _forward(encoder, _batch())
    assert float(out["pooled"].std().detach()) > 1e-3, "representation collapsed at initialisation"


@pytest.mark.parametrize("arch", ARCHS)
def test_output_depends_on_the_signal(arch):
    encoder = build_matched_encoder(arch).eval()
    batch = _batch()
    first = _forward(encoder, batch)["pooled"]
    batch["patches"] = batch["patches"] * 3.0 + 1.0
    second = _forward(encoder, batch)["pooled"]
    assert not torch.allclose(first, second, atol=1e-4)


@pytest.mark.parametrize("arch", ARCHS)
def test_rate_is_respected_not_ignored(arch):
    """The same physical signal at two acquisition rates must reach the trunk at its own rate."""
    encoder = build_matched_encoder(arch).eval()
    slow = _batch(rate=25.0)
    fast = _batch(rate=100.0)
    fast["patches"] = slow["patches"].clone()
    a = _forward(encoder, slow)["pooled"]
    b = _forward(encoder, fast)["pooled"]
    assert not torch.allclose(a, b, atol=1e-5)


def test_accelerometer_only_trunk_ignores_the_gyroscope():
    """harnet5 is accelerometer-only; a gyroscope change must not move its output."""
    encoder = build_matched_encoder("harnet").eval()
    batch = _batch()
    before = _forward(encoder, batch)["pooled"]
    batch["patches"][:, :, :, 3:] += 5.0          # gyroscope channels only
    after = _forward(encoder, batch)["pooled"]
    assert torch.allclose(before, after, atol=1e-5)


def test_six_axis_trunk_uses_the_gyroscope():
    encoder = build_matched_encoder("limubert").eval()
    batch = _batch()
    before = _forward(encoder, batch)["pooled"]
    batch["patches"][:, :, :, 3:] += 5.0
    after = _forward(encoder, batch)["pooled"]
    assert not torch.allclose(before, after, atol=1e-5)


def test_composite_recording_reads_every_device_not_just_the_first():
    """A composite carries more than six channels; each device must contribute its own signal."""
    encoder = build_matched_encoder("limubert").eval()
    B, C = 2, 12
    batch = _batch(B=B, C=C)
    # Two devices, each with an accelerometer sensor and a gyroscope sensor.
    batch["sensor_id"] = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]] * B)
    batch["device_id"] = torch.tensor([[0, 0, 1, 1]] * B)
    batch["channel_mask"] = torch.ones(B, C, dtype=torch.bool)
    out = _forward(encoder, batch)
    assert out["device_present"].shape[-1] == 2
    assert bool(out["device_present"].all()), "both devices must be encoded"
    before = out["pooled"].clone()
    # Perturb ONLY the second device's channels; the pooled vector must move.
    batch["patches"][:, :, :, 6:] += 5.0
    after = _forward(encoder, batch)["pooled"]
    assert not torch.allclose(before, after, atol=1e-5), "the second device was ignored"


def test_device_pooling_is_an_unweighted_mean():
    """Device fusion stays parameter-free: HALO's learned pool is HALO's own component."""
    encoder = build_matched_encoder("limubert").eval()
    B, C = 1, 12
    batch = _batch(B=B, C=C)
    batch["sensor_id"] = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]] * B)
    batch["device_id"] = torch.tensor([[0, 0, 1, 1]] * B)
    batch["channel_mask"] = torch.ones(B, C, dtype=torch.bool)
    # Make both devices carry the identical signal; the mean must equal either device alone.
    batch["patches"][:, :, :, 6:] = batch["patches"][:, :, :, :6]
    both = _forward(encoder, batch)["pooled"]
    single = dict(batch)
    single["channel_mask"] = torch.tensor([[True] * 6 + [False] * 6])
    only_first = _forward(encoder, single)["pooled"]
    assert torch.allclose(both, only_first, atol=1e-4)


def test_a_batch_with_no_usable_window_fails_loudly():
    encoder = build_matched_encoder("harnet")
    batch = _batch()
    batch["channel_mask"] = torch.zeros_like(batch["channel_mask"])
    with pytest.raises(ValueError, match="channel contract"):
        _forward(encoder, batch)


def test_unknown_backbone_is_refused():
    with pytest.raises(ValueError, match="backbone must be"):
        MatchedCorpusEncoder("mantis")


@pytest.mark.parametrize("arch", ARCHS)
def test_checkpoint_round_trip(arch):
    encoder = build_matched_encoder(arch).eval()
    batch = _batch()
    expected = _forward(encoder, batch)["pooled"]
    clone = build_matched_encoder(arch).eval()
    clone.load_state_dict(encoder.state_dict())
    assert torch.allclose(_forward(clone, batch)["pooled"], expected, atol=1e-6)


def test_contracts_match_each_models_published_preprocessing():
    """Pinned against the released adapters, never against a literal copied from this file.

    This test previously asserted ``limubert rate_hz == 20.0``, which is what the contract said
    and what the released checkpoint does not do. Restating a value cannot detect that the value
    is wrong; only comparing it to the artifact can. The rate and clip therefore come from the
    adapters themselves here, and `tests/test_matched_encoder_fidelity.py` additionally checks
    LiMU-BERT's clip against the checkpoint's positional-embedding table.
    """
    from baselines.harnet import adapter as harnet
    from baselines.limubert_x import adapter as limubert

    assert BACKBONE_CONTRACTS["limubert"]["rate_hz"] == limubert.TARGET_HZ
    assert BACKBONE_CONTRACTS["limubert"]["clip"] == limubert.SEQ_LEN
    assert BACKBONE_CONTRACTS["limubert"]["channels"] == 6
    # 20 samples at 10 Hz is the released two-second clip.
    assert (BACKBONE_CONTRACTS["limubert"]["clip"]
            / BACKBONE_CONTRACTS["limubert"]["rate_hz"]) == 2.0

    assert BACKBONE_CONTRACTS["harnet"]["rate_hz"] == harnet.TARGET_HZ
    assert BACKBONE_CONTRACTS["harnet"]["clip"] == harnet.TARGET_LEN     # 5 s at 30 Hz
    assert BACKBONE_CONTRACTS["harnet"]["channels"] == 3                 # accelerometer only


# --------------------------------------------------- UniMTS: native skeleton fusion

def _multi_device_batch(B=3, C=12):
    batch = _batch(B=B, C=C)
    batch["sensor_id"] = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]] * B)
    batch["device_id"] = torch.tensor([[0, 0, 1, 1]] * B)
    batch["channel_mask"] = torch.ones(B, C, dtype=torch.bool)
    batch["sensor_texts"] = [[
        "a phone accelerometer on the right wrist", "a phone gyroscope on the right wrist",
        "a phone accelerometer on the thigh", "a phone gyroscope on the thigh",
    ]] * B
    return batch


def _forward_texts(encoder, batch):
    return encoder(
        batch["patches"], batch["rates"], batch["patch_len"], None, None,
        channel_mask=batch["channel_mask"], patch_padding_mask=batch["patch_padding_mask"],
        sensor_id=batch["sensor_id"], device_id=batch["device_id"],
        sensor_texts=batch.get("sensor_texts"),
    )


def test_sensor_joint_reads_placement_from_the_sensor_description():
    from model.tokenizer.matched_encoder import sensor_joint
    wrist = sensor_joint("a phone accelerometer on the right wrist")
    thigh = sensor_joint("a phone accelerometer on the thigh")
    left = sensor_joint("a wearable device accelerometer on the left wrist")
    assert wrist != thigh, "different placements must be different graph nodes"
    assert left != wrist, "side must be distinguished"


def test_unimts_places_every_device_on_the_skeleton():
    encoder = build_matched_encoder("unimts").eval()
    batch = _multi_device_batch()
    out = _forward_texts(encoder, batch)
    assert out["pooled"].shape == (3, encoder.d_model)
    assert bool(out["device_present"].all()), "both placements must reach the graph"


def test_unimts_output_moves_when_the_second_placement_changes():
    encoder = build_matched_encoder("unimts").eval()
    batch = _multi_device_batch()
    before = _forward_texts(encoder, batch)["pooled"].clone()
    batch["patches"][:, :, :, 6:] += 5.0          # second device only
    after = _forward_texts(encoder, batch)["pooled"]
    assert not torch.allclose(before, after, atol=1e-5)


def test_unimts_fuses_devices_natively_rather_than_pooling_them():
    """The graph does the fusion; pooling per device afterwards would replace its contribution."""
    encoder = build_matched_encoder("unimts")
    assert encoder.fuses_devices_natively
    assert not build_matched_encoder("limubert").fuses_devices_natively


def test_unimts_trunk_is_trainable():
    encoder = build_matched_encoder("unimts")
    out = _forward_texts(encoder, _multi_device_batch())
    torch.manual_seed(0)
    (out["pooled"] * torch.randn_like(out["pooled"])).sum().backward()
    grads = [p.grad for p in encoder.net.parameters()]
    moved = sum(1 for g in grads if g is not None and float(g.abs().sum()) > 0)
    assert moved > len(grads) // 2, f"only {moved}/{len(grads)} trunk parameters received gradient"


def test_unimts_is_accelerometer_only():
    encoder = build_matched_encoder("unimts").eval()
    batch = _multi_device_batch()
    before = _forward_texts(encoder, batch)["pooled"].clone()
    # Gyroscope channels of device 0 are columns 3:6; device 1's are 9:12.
    batch["patches"][:, :, :, 3:6] += 5.0
    batch["patches"][:, :, :, 9:12] += 5.0
    after = _forward_texts(encoder, batch)["pooled"]
    assert torch.allclose(before, after, atol=1e-5)


# ------------------------------------------- regressions found by the 2026-09-15 sweep

def test_mixed_rate_groups_are_encoded_at_their_own_length():
    """Padding every group to the longest in the batch fabricated data the trunk then consumed.

    For LiMU-BERT the padding became extra one-second clips that diluted the mean; for harnet it
    shifted the centre crop into repeated samples. Each group must be encoded at its own length.
    """
    encoder = build_matched_encoder("limubert").eval()
    slow = _batch(B=2, rate=20.0)
    # A batch whose rows resample to different lengths: same patches, different acquisition rates.
    mixed = _batch(B=4, rate=20.0)
    mixed["rates"] = torch.tensor([20.0, 20.0, 50.0, 50.0])
    mixed["patches"][:2] = slow["patches"]
    out_mixed = _forward(encoder, mixed)["pooled"]
    out_alone = _forward(encoder, slow)["pooled"]
    assert torch.allclose(out_mixed[:2], out_alone, atol=1e-4), (
        "a row's embedding changed because of the acquisition rate of other rows in its batch"
    )


def test_unimts_joint_lookup_needs_no_per_element_device_read():
    """Regression for a synchronisation stall, asserted behaviourally: joints must still be right."""
    from model.tokenizer.matched_encoder import sensor_joint
    encoder = build_matched_encoder("unimts").eval()
    batch = _multi_device_batch()
    out = _forward_texts(encoder, batch)
    assert bool(out["device_present"].all())
    # Swapping the two devices' placement text must change the representation.
    swapped = _multi_device_batch()
    swapped["sensor_texts"] = [[
        "a phone accelerometer on the thigh", "a phone gyroscope on the thigh",
        "a phone accelerometer on the right wrist", "a phone gyroscope on the right wrist",
    ]] * 3
    assert sensor_joint("a phone accelerometer on the thigh") != \
        sensor_joint("a phone accelerometer on the right wrist")
    assert not torch.allclose(out["pooled"], _forward_texts(encoder, swapped)["pooled"], atol=1e-5)


def test_unimts_window_with_no_usable_device_is_not_given_an_embedding():
    """An all-zero skeleton must not produce a vector that looks like evidence."""
    encoder = build_matched_encoder("unimts").eval()
    batch = _multi_device_batch(B=2)
    batch["channel_mask"] = torch.zeros_like(batch["channel_mask"])
    batch["channel_mask"][0, :] = True          # only the first window is usable
    out = _forward_texts(encoder, batch)
    assert bool(out["device_present"][0].any())
    assert not bool(out["device_present"][1].any())
    assert float(out["pooled"][1].abs().sum()) == 0.0
    assert float(out["pooled"][0].abs().sum()) > 0.0


def test_a_trunk_that_cannot_take_a_short_input_is_never_given_one():
    """harnet5's ResNet pads circularly and raises below its 5 s contract; UniMTS's graph does not."""
    from model.tokenizer.matched_encoder import BACKBONE_CONTRACTS
    assert BACKBONE_CONTRACTS["harnet"]["min_clip"] == BACKBONE_CONTRACTS["harnet"]["clip"]
    assert BACKBONE_CONTRACTS["unimts"]["min_clip"] < BACKBONE_CONTRACTS["unimts"]["clip"]
    # A two-second window at 100 Hz resamples well below harnet's 150 samples; it must wrap-pad up.
    encoder = build_matched_encoder("harnet").eval()
    short = _batch(rate=100.0)
    out = _forward(encoder, short)
    assert torch.isfinite(out["pooled"]).all()


def test_unimts_uses_the_real_window_rather_than_fabricating_padding():
    """Wrap-padding an 8 s window to the released 10 s convention invents a quarter of the input."""
    encoder = build_matched_encoder("unimts").eval()
    assert encoder.min_clip < encoder.clip
    pretrained_rule = build_matched_encoder("unimts", pretrained=False)
    assert pretrained_rule._clip_length([torch.zeros(2, 160, 3)]) == 160
    assert pretrained_rule._clip_length([torch.zeros(2, 400, 3)]) == 200, "still capped at the contract"


@pytest.mark.parametrize("arch", ARCHS)
def test_projection_free_trunk_exposes_its_own_feature(arch):
    """Rung 3 heads sit on the trunk feature: no freshly initialised layer between them."""
    encoder = build_matched_encoder(arch, projection=False).eval()
    assert isinstance(encoder.proj, torch.nn.Identity) and isinstance(encoder.row_norm, torch.nn.Identity)
    assert encoder.d_model == encoder.net.out_dim
    out = _forward(encoder, _batch())
    assert out["pooled"].shape == (4, encoder.net.out_dim)
    assert torch.isfinite(out["pooled"]).all()
    assert not any(isinstance(m, torch.nn.Linear) for m in encoder.modules() if m is not encoder.net
                   and not any(m is n for n in encoder.net.modules()))
