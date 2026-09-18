from __future__ import annotations

import pytest
import torch

from model.tokenizer.sensor_tokens import (
    LEGACY_CONDITIONING_SCHEMA,
    GRAVITY_NOT_APPLICABLE,
    GRAVITY_PRESENT,
    MODALITY_ACCELEROMETER,
    MODALITY_GYROSCOPE,
    StructuredSensorConditioner,
)
from training.tokenizer.pretrain_data import (
    stream_sensor_texts,
    structured_sensor_metadata,
)


def test_natural_language_contains_only_runtime_device_and_placement():
    _, descriptions, sensor_id = stream_sensor_texts("wisdm", "watch_wrist")
    assert descriptions == [
        "a watch located at the dominant wrist",
        "a watch located at the dominant wrist",
    ]
    assert sensor_id == [0, 0, 0, 1, 1, 1]
    forbidden = ("wisdm", "accelerometer", "gyroscope", "gravity", "hz", "walking")
    assert all(term not in descriptions[0].lower() for term in forbidden)


def test_unregistered_stream_has_no_name_parsing_fallback():
    with pytest.raises(KeyError, match="No deployment stream"):
        stream_sensor_texts("unregistered_dataset", "phone_wrist")


def test_non_runtime_stress_device_cannot_enter_conditioning():
    with pytest.raises(ValueError, match="non-runtime device profile"):
        stream_sensor_texts("harth", "stress_thigh")


def test_legacy_renderer_reconstructs_the_historical_combined_sensor_text():
    _, descriptions, _ = stream_sensor_texts(
        "wisdm", "watch_wrist", conditioning_schema=LEGACY_CONDITIONING_SCHEMA,
    )
    assert descriptions == [
        "a watch accelerometer on the dominant wrist; includes gravity; "
        "recorded alongside a gyroscope",
        "a watch gyroscope on the dominant wrist; recorded alongside an accelerometer",
    ]


def test_structured_metadata_separates_modality_and_gravity_semantics():
    modality, gravity = structured_sensor_metadata(
        has_accel=True, has_gyro=True, gravity_state="removed",
    )
    assert modality.tolist() == [MODALITY_ACCELEROMETER, MODALITY_GYROSCOPE]
    assert gravity.tolist() == [1, GRAVITY_NOT_APPLICABLE]


def test_structured_conditioner_masks_padding_and_all_components_receive_gradient():
    conditioner = StructuredSensorConditioner(d_model=12, dropout=0.0)
    tokens = torch.randn(2, 3, 2, 12, requires_grad=True)
    modality = torch.tensor([
        [MODALITY_ACCELEROMETER, MODALITY_GYROSCOPE],
        [MODALITY_ACCELEROMETER, 0],
    ])
    gravity = torch.tensor([
        [GRAVITY_PRESENT, GRAVITY_NOT_APPLICABLE],
        [GRAVITY_PRESENT, 0],
    ])
    rates = torch.tensor([
        [[50.0, 50.0], [50.0, 50.0]],
        [[100.0, 80.0], [0.0, 0.0]],
    ])
    valid = torch.tensor([[True, True], [True, False]])

    delta = conditioner.delta(tokens, modality, gravity, rates, valid)
    assert torch.isfinite(delta).all()
    assert torch.equal(delta[1, :, 1], torch.zeros_like(delta[1, :, 1]))
    delta.square().sum().backward()
    assert tokens.grad is not None
    assert all(parameter.grad is not None for parameter in conditioner.parameters())


def test_structured_conditioner_rejects_gravity_on_gyroscope():
    conditioner = StructuredSensorConditioner(d_model=8, dropout=0.0)
    with pytest.raises(ValueError, match="gravity must be"):
        conditioner.embed(
            torch.tensor([[MODALITY_GYROSCOPE]]),
            torch.tensor([[GRAVITY_PRESENT]]),
            torch.tensor([[[50.0, 50.0]]]),
            torch.tensor([[True]]),
        )


def test_stored_rate_is_provenance_only_and_rate_math_stays_float32():
    conditioner = StructuredSensorConditioner(d_model=8, dropout=0.0).eval()
    modality = torch.tensor([[MODALITY_ACCELEROMETER]])
    gravity = torch.tensor([[GRAVITY_PRESENT]])
    valid = torch.tensor([[True]])
    captured = []
    hook = conditioner.rate[0].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].dtype)
    )
    first = conditioner.embed(
        modality, gravity, torch.tensor([[[20.0, 50.0]]], dtype=torch.bfloat16), valid,
    )
    second = conditioner.embed(
        modality, gravity, torch.tensor([[[100.0, 50.0]]], dtype=torch.bfloat16), valid,
    )
    changed_bandwidth = conditioner.embed(
        modality, gravity, torch.tensor([[[100.0, 25.0]]], dtype=torch.bfloat16), valid,
    )
    hook.remove()
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, changed_bandwidth)
    assert captured == [torch.float32, torch.float32, torch.float32]
