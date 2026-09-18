"""Factored text metadata must stay synchronized with physics/channel augmentations."""

import random

import numpy as np
import torch

from data.scripts.augmentations import AugmentationConfig, IMUAugmenter, IMUSample
from training.tokenizer.pretrain_data import CHANNELS, stream_sensor_texts


def _sample() -> IMUSample:
    roles, sensors, sensor_id = stream_sensor_texts("hhar", "phone_waist")
    data = torch.zeros(360, 6)
    data[:, 0] = 1.0
    data[:, 1] = 0.05 * torch.sin(torch.linspace(0, 20, 360))
    return IMUSample(
        data=data,
        channel_names=list(CHANNELS),
        sampling_rate=60.0,
        channel_descriptions=[
            f"{name}; includes gravity" if name.startswith("acc") else name
            for name in CHANNELS
        ],
        channel_mask=[True] * 6,
        role_descriptions=roles,
        sensor_descriptions=sensors,
        sensor_id=sensor_id,
        gravity_state="present",
    )


def test_gravity_removal_updates_structured_state_without_polluting_natural_text():
    cfg = AugmentationConfig.none()
    cfg.gravity.enabled = True
    cfg.gravity.p = 1.0
    out = IMUAugmenter(cfg)(_sample())
    assert out.gravity_state == "removed"
    assert out.sensor_descriptions[0] == "a phone located at the waist"
    assert out.data[:, :3].mean(0).norm() < 0.1
    assert out.applied_augmentations == ["gravity"]


def test_gyro_dropout_drops_the_gyro_sensor():
    cfg = AugmentationConfig.none()
    cfg.channel_dropout.enabled = True
    cfg.channel_dropout.p = 1.0
    out = IMUAugmenter(cfg)(_sample())
    assert out.channel_names == list(CHANNELS[:3])          # only the accel triad survives
    assert out.sensor_id == [0, 0, 0]
    # accel & gyro are separate modality-level sensors: dropping the gyro group REMOVES the gyro
    # sensor entirely (no phantom "accelerometer only" phrase on a shared description).
    assert len(out.sensor_descriptions) == 1
    assert out.sensor_descriptions == ["a phone located at the waist"]
    assert out.applied_augmentations == ["channel_dropout"]


def test_text_dropout_respects_channel_budget_without_erasing_sensor_identity():
    cfg = AugmentationConfig.none()
    cfg.channel_text_dropout.enabled = True
    cfg.channel_text_dropout.p = 1.0
    cfg.channel_text_dropout.max_frac = 0.5
    random.seed(12)
    np.random.seed(12)
    sample = _sample()
    original_sensor = list(sample.sensor_descriptions)
    out = IMUAugmenter(cfg)(sample)
    # Role slots are AXIS-only, so they neutralize to the axis-specific `role_neutral`, not to the
    # channel neutral (which states the wrong kind of fact for an axis slot).
    dropped = [i for i, text in enumerate(out.role_descriptions)
               if text == cfg.channel_text_dropout.role_neutral]
    assert 1 <= len(dropped) <= 3
    assert out.sensor_descriptions == original_sensor      # sensor identity survives channel dropout
    assert all(out.channel_descriptions[i] == cfg.channel_text_dropout.neutral for i in dropped)
    # every surviving role slot is still a bare axis
    for i, text in enumerate(out.role_descriptions):
        if i not in dropped:
            assert text in {"x", "y", "z"}, f"role slot {i} is not an axis: {text!r}"
    assert out.applied_augmentations == ["channel_text_dropout"]


def test_sensor_text_dropout_is_bounded_and_keeps_one_with_two_sensors():
    cfg = AugmentationConfig.none()
    cfg.sensor_text_dropout.enabled = True
    cfg.sensor_text_dropout.p = 1.0
    random.seed(3)
    np.random.seed(3)
    out = IMUAugmenter(cfg)(_sample())          # hhar phone_waist -> accel + gyro sensors
    assert len(out.sensor_descriptions) == 2    # both slots kept; identity may be neutralized
    survived = [d for d in out.sensor_descriptions if d != cfg.sensor_text_dropout.neutral]
    assert len(survived) >= 1, "bounded dropout must keep >=1 sensor described when there are 2"


def test_sensor_text_dropout_decision_is_shared_across_relation_views():
    """Config-conditional thesis guard: the two relation views of one window must never disagree on
    WHETHER the acquisition config was described. With independent draws ~2p(1-p) of positive pairs
    are 'config in A, neutralised in B', and relation learning trains embed(config) ==
    embed(no config), i.e. to IGNORE the config — the opposite of salient-not-invariant."""
    cfg = AugmentationConfig.none()
    cfg.sensor_text_dropout.enabled = True
    cfg.sensor_text_dropout.p = 0.5            # high rate so disagreement would be common
    aug = IMUAugmenter(cfg)
    neutral = cfg.sensor_text_dropout.neutral
    asymmetric = 0
    for seed in range(60):
        random.seed(seed); np.random.seed(seed)
        shared = random.randrange(2**31)
        a = aug(_sample(), shared_config_seed=shared)
        b = aug(_sample(), shared_config_seed=shared)
        na = any(s == neutral for s in a.sensor_descriptions)
        nb = any(s == neutral for s in b.sensor_descriptions)
        if na != nb:
            asymmetric += 1
        if na and nb:                           # same sensors chosen, not just the same count
            assert a.sensor_descriptions == b.sensor_descriptions
    assert asymmetric == 0, f"{asymmetric}/60 pairs had config in one view only"


def test_shared_configuration_is_applied_before_independent_crop():
    """A nuisance crop must not make a shared rate transform fire in only one positive view."""
    cfg = AugmentationConfig.none()
    cfg.rate.enabled = True
    cfg.rate.p = 1.0
    cfg.rate.min_hz = cfg.rate.max_hz = 30.0
    cfg.window_crop.enabled = True
    cfg.window_crop.p = 1.0
    aug = IMUAugmenter(cfg)

    def short_sample():
        sample = _sample()
        sample.data = sample.data[:70].clone()  # rate passes before crop; some crops would make it skip
        return sample

    for seed in range(40):
        random.seed(seed)
        np.random.seed(seed)
        shared = random.randrange(2**31)
        a = aug(short_sample(), shared_config_seed=shared)
        b = aug(short_sample(), shared_config_seed=shared)
        assert a.sampling_rate == b.sampling_rate == 30.0
        assert "rate" in a.applied_augmentations and "rate" in b.applied_augmentations


def test_padding_only_accelerometer_is_not_treated_as_physical_gravity():
    cfg = AugmentationConfig.none()
    cfg.gravity.enabled = True
    cfg.gravity.p = 1.0
    roles, sensors, sensor_id = stream_sensor_texts(
        "hhar", "phone_waist", has_accel=False, has_gyro=True
    )
    sample = IMUSample(
        data=torch.zeros(120, 6),
        channel_names=list(CHANNELS),
        sampling_rate=50.0,
        channel_descriptions=["accelerometer includes gravity"] * 3 + ["gyroscope"] * 3,
        channel_mask=[False, False, False, True, True, True],
        role_descriptions=roles,
        sensor_descriptions=sensors,
        sensor_id=sensor_id,
        gravity_state=None,
    )
    out = IMUAugmenter(cfg)(sample)
    assert out.gravity_state is None
    # has_accel=False -> only a gyroscope sensor is advertised; no phantom accelerometer sensor.
    assert len(out.sensor_descriptions) == 1
    assert out.sensor_descriptions == ["a phone located at the waist"]
