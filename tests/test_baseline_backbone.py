"""The encoder-swap harness must be a faithful stand-in, or the comparison measures the harness.

Each test here pins one property that, if broken, would silently make a third-party backbone look
worse than it is: rate-correct resampling, honest accelerometer-only rows, matched row scale,
placement, gravity compatibility, and real freezing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.tokenizer.baseline_backbone import (
    BaselineRowEncoder,
    _center_crop_or_wrap,
    _start_crop_or_wrap,
    _stream_joint,
)


def _batch(n=8, patches=6, samples=64, channels=6, rate=50.0, slots=2):
    torch.manual_seed(0)
    sensor_id = torch.zeros(n, channels, dtype=torch.long)
    sensor_id[:, 3:] = 1
    channel_mask = torch.ones(n, channels, dtype=torch.bool)
    if slots == 1:
        channel_mask[:, 3:] = False
    return {
        "patches": torch.randn(n, patches, samples, channels),
        "rates": torch.full((n,), rate),
        "patch_len": torch.full((n, patches), samples, dtype=torch.long),
        "positions": torch.arange(patches, dtype=torch.float32).repeat(n, 1),
        "patch_durations": torch.ones(n, patches),
        "channel_mask": channel_mask,
        "patch_padding_mask": torch.ones(n, patches, dtype=torch.bool),
        "sensor_id": sensor_id,
        "role_texts": [["x", "y", "z"] * 2 for _ in range(n)],
        "sensor_texts": [(["a wrist accelerometer"] if slots == 1 else
                          ["a wrist accelerometer", "a wrist gyroscope"])
                         for _ in range(n)],
        "source_rates": torch.full((n,), rate),
        "streams": ["mhealth/watch_wrist"] * n,
        "gravity_state": ["present"] * n,
    }


def _run(encoder, batch):
    return encoder(
        batch["patches"], batch["rates"], batch["patch_len"], batch["role_texts"],
        batch["positions"], patch_durations=batch["patch_durations"],
        channel_mask=batch["channel_mask"], patch_padding_mask=batch["patch_padding_mask"],
        sensor_texts=batch["sensor_texts"], sensor_id=batch["sensor_id"],
        source_rate_hz=batch["source_rates"],
        streams=batch["streams"], gravity_state=batch["gravity_state"],
    )


@pytest.fixture(scope="module")
def harnet():
    return BaselineRowEncoder("harnet", d_model=128, freeze=True, device=torch.device("cpu"))


def test_rejects_an_unknown_backbone():
    with pytest.raises(ValueError):
        BaselineRowEncoder("resnet50")


def test_rows_carry_the_engine_contract(harnet):
    batch = _batch()
    out = _run(harnet, batch)
    assert out["retrieval_tokens"].shape == (8, 1, 2, 128)
    assert out["sensor_present"].shape == (8, 2)
    assert out["sensor_present"][:, 0].all()
    assert not out["sensor_present"][:, 1].any()       # accel-only published backbones
    assert out["retrieval_window_rows"] is True
    assert out["descriptor"] is not None                 # live_sensor_rows indexes this
    assert torch.isfinite(out["retrieval_tokens"]).all()


def test_row_scale_matches_our_trunks_contract(harnet):
    """Our trunk emits rows of norm sqrt(d); an arm with a different scale tests a different
    numerical regime in the mixer rather than its own representation."""
    out = _run(harnet, _batch())
    present = out["sensor_present"]
    rows = out["retrieval_tokens"][:, 0][present]
    norms = rows.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 128 ** 0.5), rtol=2e-3)


def test_frozen_trunk_projection_receives_gradient(harnet):
    harnet.zero_grad(set_to_none=True)
    _run(harnet, _batch(n=2))["retrieval_tokens"][..., 0].mean().backward()
    assert harnet.proj.weight.grad is not None
    assert torch.isfinite(harnet.proj.weight.grad).all()
    assert harnet.proj.weight.grad.norm() > 0
    assert all(parameter.grad is None for parameter in harnet.net.parameters())


def test_resampling_follows_the_sampling_RATE_not_the_array_length(harnet):
    """The same waveform recorded at 25 Hz and at 50 Hz must reach the backbone alike.

    Resampling by array length instead of by rate time-compresses the signal -- an artefact that
    would handicap every third-party backbone and stay invisible in the loss.
    """
    def sine_batch(rate, samples):
        batch = _batch(n=2, samples=samples, rate=rate, slots=1)
        t = torch.arange(6 * samples, dtype=torch.float32) / rate      # seconds
        wave = torch.stack([torch.sin(2 * np.pi * f * t) for f in (1.0, 2.5, 4.0)], dim=-1)
        patched = wave.reshape(1, 6, samples, 3).repeat(2, 1, 1, 1)
        batch["patches"] = torch.zeros(2, 6, samples, 6)
        batch["patches"][:, :, :, :3] = patched
        return batch

    seen = []
    original = harnet._features

    def spy(window, joint, **kwargs):
        seen.append(window.clone())
        return original(window, joint, **kwargs)

    harnet._features = spy
    try:
        _run(harnet, sine_batch(25.0, 32))
        _run(harnet, sine_batch(50.0, 64))
    finally:
        harnet._features = original

    slow, fast = seen[0], seen[1]
    assert slow.shape == fast.shape == (2, 150, 3)
    # Both describe the same seconds of the same sinusoids resampled to 30 Hz. They are not
    # bit-identical because each source clock samples at different phases, but anti-aliased physical
    # resampling must preserve the same waveform rather than create a time-scaled copy.
    assert torch.allclose(slow, fast, atol=0.08)
    a = (slow - slow.mean()).flatten()
    b = (fast - fast.mean()).flatten()
    correlation = float(a @ b / (a.norm() * b.norm()))
    assert correlation > 0.999, correlation
    # a length-based resample would time-compress the 25 Hz batch by 2x: assert that it does NOT
    shifted = torch.nn.functional.interpolate(
        slow[:, :75].permute(0, 2, 1), size=150, mode="linear", align_corners=True
    ).permute(0, 2, 1)
    assert torch.norm(slow - fast) < torch.norm(shifted - fast)


def test_a_frozen_backbone_stays_frozen_through_encoder_train():
    """`nn.Module.train()` recurses; without an override the frozen arm's BatchNorm statistics
    would drift on our corpus and the arm would not be frozen at all."""
    frozen = BaselineRowEncoder("harnet", freeze=True, device=torch.device("cpu"))
    frozen.train()
    norms = [m for m in frozen.net.modules()
             if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert norms and all(not m.training for m in norms)
    assert not any(p.requires_grad for p in frozen.net.parameters())


def test_fine_tuned_third_party_arm_is_not_part_of_the_experiment():
    with pytest.raises(ValueError, match="keeps released third-party backbones frozen"):
        BaselineRowEncoder("harnet", freeze=False, device=torch.device("cpu"))


def test_an_accel_row_ignores_the_gyroscope(harnet):
    """Rows are the retrieval unit; one sensor's row must not depend on a co-resident sensor."""
    batch = _batch()
    before = _run(harnet, batch)["retrieval_tokens"][:, :, 0]
    batch["patches"][:, :, :, 3:] = torch.randn_like(batch["patches"][:, :, :, 3:])
    after = _run(harnet, batch)["retrieval_tokens"][:, :, 0]
    assert torch.equal(before, after)


def test_gyro_never_becomes_an_accelerometer_row(harnet):
    out = _run(harnet, _batch())
    assert out["sensor_present"][:, 0].all()
    assert not out["sensor_present"][:, 1].any()


def test_gravity_removed_acceleration_is_explicitly_incompatible(harnet):
    batch = _batch(n=4)
    batch["gravity_state"] = ["removed"] * 4
    out = _run(harnet, batch)
    assert not out["sensor_present"].any()


def test_unimts_placement_mapping_uses_the_stream_key():
    assert _stream_joint("xrf_v2/left_wrist") == 17
    assert _stream_joint("c_mhad/right_wrist") == 21
    assert _stream_joint("oca/imu0_right upper arm") == 19
    assert _stream_joint("oca/imu2_left upper arm") == 15
    assert _stream_joint("xrf_v2/right_pocket") == 5
    assert _stream_joint("nfi_fared/lower_back") == 9
    assert _stream_joint("unknown/device") == 0


def test_backbones_keep_their_released_crop_and_padding_conventions():
    x = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    assert _center_crop_or_wrap(x, 2).flatten().tolist() == [1.0, 2.0]
    assert _start_crop_or_wrap(x, 2).flatten().tolist() == [0.0, 1.0]
    assert _center_crop_or_wrap(x, 6).flatten().tolist() == [3.0, 0.0, 1.0, 2.0, 3.0, 0.0]
    assert _start_crop_or_wrap(x, 6).flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 0.0, 1.0]


def test_padding_and_mixed_rates_do_not_inject_zeros(harnet):
    """A short window must resample from its own valid samples, never from padding."""
    batch = _batch(n=4)
    batch["patch_len"][:, 3:] = 0
    batch["patch_padding_mask"][:, 3:] = False
    batch["patches"][:, 3:] = 12345.0                    # poison the padded region
    out = _run(harnet, batch)
    assert torch.isfinite(out["retrieval_tokens"]).all()
    assert out["retrieval_tokens"].abs().max() < 1e3


def test_a_single_sample_window_still_produces_a_row(harnet):
    """~0.1% of corpus windows (wisdm, capture24, opportunity) carry exactly one sample.

    Our encoder emits rows for them, so refusing them would shrink the baseline arms' row
    population and stop the comparison being matched -- and it crashed all four arms at
    validation before this was handled.
    """
    batch = _batch(n=4)
    batch["patch_len"][:] = 0
    batch["patch_len"][:, 0] = 1
    batch["patch_padding_mask"][:] = False
    batch["patch_padding_mask"][:, 0] = True
    out = _run(harnet, batch)
    assert torch.isfinite(out["retrieval_tokens"]).all()
    assert out["sensor_present"].any()


def test_checkpoint_factory_reconstructs_a_baseline_encoder(harnet):
    from training.tokenizer.eval_transfer import build_encoder

    checkpoint = {
        "config": {
            "encoder_backbone": "harnet",
            "freeze_backbone": True,
            "retrieval_granularity": "window",
            "d_model": 128,
        },
        "encoder": harnet.state_dict(),
    }
    restored = build_encoder(checkpoint, torch.device("cpu"))
    assert isinstance(restored, BaselineRowEncoder)
    assert restored.retrieval_granularity == "window"
    assert torch.equal(
        _run(harnet, _batch(n=2))["retrieval_tokens"],
        _run(restored, _batch(n=2))["retrieval_tokens"],
    )


def test_runtime_acceleration_does_not_change_checkpoint_keys(harnet):
    """Compiled inference is an execution detail, not a new checkpoint architecture."""
    assert not any("_compiled_net" in key or "_orig_mod" in key for key in harnet.state_dict())
