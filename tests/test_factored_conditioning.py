"""Tests for factored text conditioning and acquisition-conditioning-v2.

The refactor splits the tokenizer's per-channel placement-repeating text into:
  * per-CHANNEL role text (axis only),
  * per-SENSOR natural text (device role and placement only), and
  * exact structured modality, gravity, and rate fields.

These tests pin the properties that make the change safe and honest: the sensor identity really is
broadcast by ``sensor_id``; independent residual branches avoid double-injection; and the legacy
per-channel path remains reconstructible for old checkpoints.
"""

import numpy as np
import torch

from model.tokenizer.channel_text import FactoredChannelTextFusion, _TextPooler
from model.tokenizer.encoder import SetTokenizerEncoder
from training.tokenizer.pretrain_data import CHANNELS, stream_sensor_texts


# ----------------------------------------------------------------------------- data-side factoring
def test_role_text_is_axis_only_and_sensor_text_carries_device_placement():
    """The compounding hazard (§6): a config fact lives in exactly ONE source. Accel and gyro are
    two modality-level sensors; role text is the positional axis only."""
    role, sensor, sensor_id = stream_sensor_texts("wisdm", "watch_wrist")
    assert len(role) == len(CHANNELS) and len(sensor) == 2      # accel + gyro sensors
    assert sensor_id == [0, 0, 0, 1, 1, 1]
    for r in role:
        assert r in {"x", "y", "z"}, f"role must be axis-only, got {r!r}"
    accel_s, gyro_s = sensor
    assert accel_s == gyro_s
    for s in sensor:
        assert "wrist" in s and "watch" in s          # placement + device on each sensor
        assert "axis" not in s                        # axis lives only in the role text
        assert all(word not in s for word in ("accelerometer", "gyroscope", "gravity", "hz"))


def test_role_text_is_constant_across_streams_sensor_text_is_not():
    r1, s1, _ = stream_sensor_texts("wisdm", "watch_wrist")
    r2, s2, _ = stream_sensor_texts("wisdm", "phone_pocket")
    assert r1 == r2, "role text must be corpus-constant (axis/modality only)"
    assert s1 != s2, "sensor text must vary with device/placement"


def test_accel_only_stream_advertises_one_sensor_with_gravity_state():
    _, sensor, sensor_id = stream_sensor_texts(
        "capture24", "watch_wrist", has_accel=True, has_gyro=False,
        gravity_removed=True,
    )
    assert len(sensor) == 1                            # no phantom gyroscope sensor
    assert "watch" in sensor[0] and "wrist" in sensor[0]
    assert "accelerometer" not in sensor[0] and "gravity" not in sensor[0]
    assert sensor_id == [0, 0, 0, 0, 0, 0]             # every slot -> the single accel sensor


def test_eval_encoding_builds_factored_text_from_actual_channel_mask():
    from training.tokenizer.eval_transfer import encode_dataset

    class SpyEncoder:
        text_conditioning = "factored"
        use_duration_embedding = False

        def __init__(self):
            self.sensor_texts = None
            self.kwargs = None

        def __call__(self, patches, rates, patch_len, role_texts, positions, **kwargs):
            self.sensor_texts = kwargs["sensor_texts"]
            self.kwargs = kwargs
            return {"pooled": torch.zeros(len(patches), 4)}

    enc = SpyEncoder()
    encode_dataset(
        enc,
        np.zeros((2, 60, 6), dtype=np.float32),
        ["legacy"] * 6,
        torch.device("cpu"),
        rate=60.0,
        gravity_state="removed",
        channel_mask=[True, True, True, False, False, False],
        dataset="capture24",
        stream="watch_wrist",
    )
    text = enc.sensor_texts[0][0]
    assert "watch" in text and "wrist" in text
    assert all(word not in text for word in ("accelerometer", "gyroscope", "gravity", "hz"))
    assert len(enc.sensor_texts[0]) == 1               # acc-only mask -> one advertised sensor
    assert enc.kwargs["sensor_modality"].tolist() == [[0], [0]]
    assert enc.kwargs["sensor_gravity"].tolist() == [[1], [1]]


# ------------------------------------------------------------------------------------ the pooler
def test_text_pooler_handles_all_masked_row_without_nan():
    torch.manual_seed(0)
    pool = _TextPooler(d_model=32, text_dim=16).eval()
    toks = torch.randn(3, 5, 16)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[1] = False                                # a padding row
    with torch.no_grad():
        out = pool(toks, mask)
    assert out.shape == (3, 32) and torch.isfinite(out).all()


# --------------------------------------------------------------------------------- the fusion
def _fusion_inputs(B=2, P=4, C=6, d=32, text_dim=16, S=5, n_sensors=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        sensor_tokens=torch.randn(B, P, C, d, generator=g),
        role_tokens=torch.randn(B, C, S, text_dim, generator=g),
        role_mask=torch.ones(B, C, S, dtype=torch.bool),
        sensor_text_tokens=torch.randn(B, n_sensors, S, text_dim, generator=g),
        sensor_text_mask=torch.ones(B, n_sensors, S, dtype=torch.bool),
        sensor_id=torch.zeros(B, C, dtype=torch.long),
    )


def test_gate_bias_makes_identity_negligible_at_init():
    """Do-no-harm / ablatable: a very negative gate bias => fusion ≈ identity map on sensor tokens."""
    torch.manual_seed(0)
    fusion = FactoredChannelTextFusion(d_model=32, text_dim=16, gate_bias_init=-30.0).eval()
    x = _fusion_inputs()
    with torch.no_grad():
        out = fusion(**x)
    assert torch.allclose(out, x["sensor_tokens"], atol=1e-4), "identity should be ~off at very negative gate bias"


def test_sensor_identity_is_broadcast_by_sensor_id():
    """Two channels in the SAME sensor get the SAME sensor contribution; different sensors differ."""
    torch.manual_seed(1)
    fusion = FactoredChannelTextFusion(d_model=16, text_dim=8, gate_bias_init=0.0).eval()
    B, P, C, d, S = 1, 1, 4, 16, 3
    # zero the role path so we isolate the sensor contribution, and zero sensor tokens so the
    # output IS the gated sensor identity.
    role_tokens = torch.zeros(B, C, S, 8)
    role_mask = torch.ones(B, C, S, dtype=torch.bool)
    sensor_text_tokens = torch.randn(B, 2, S, 8)     # two DISTINCT sensors
    sensor_text_mask = torch.ones(B, 2, S, dtype=torch.bool)
    sensor_id = torch.tensor([[0, 0, 1, 1]])         # ch0,1 -> sensor0 ; ch2,3 -> sensor1
    sensor_tokens = torch.zeros(B, P, C, d)
    with torch.no_grad():
        out = fusion(sensor_tokens, role_tokens, role_mask,
                     sensor_text_tokens, sensor_text_mask, sensor_id)[0, 0]   # (C, d)
    assert torch.allclose(out[0], out[1], atol=1e-6), "same-sensor channels must match"
    assert torch.allclose(out[2], out[3], atol=1e-6), "same-sensor channels must match"
    assert not torch.allclose(out[0], out[2], atol=1e-4), "different sensors must differ"


def test_identity_is_role_plus_sensor_sum():
    """identity = role + sensor: with the gate forced to 1, output - sensor_tokens = role_pool + sensor_pool."""
    torch.manual_seed(2)
    fusion = FactoredChannelTextFusion(d_model=16, text_dim=8, gate_bias_init=0.0).eval()
    x = _fusion_inputs(d=16, text_dim=8)
    B, P, C, d = x["sensor_tokens"].shape
    # Force gate == 1 everywhere so the residual is exactly the identity vector.
    with torch.no_grad():
        fusion.gate_sensor.weight.zero_()
        fusion.gate_identity.weight.zero_()
        fusion.gate_identity.bias.fill_(30.0)        # sigmoid(30) ~ 1
        out = fusion(**x)
        role = fusion.pool(x["role_tokens"].reshape(B * C, -1, 8),
                           x["role_mask"].reshape(B * C, -1)).reshape(B, C, d)
        sens = fusion.pool(x["sensor_text_tokens"].reshape(B, -1, 8),
                           x["sensor_text_mask"].reshape(B, -1)).reshape(B, 1, d)
        expected_identity = role + sens              # sensor_id all 0 -> broadcast sensor 0
        got_identity = (out - x["sensor_tokens"]).mean(dim=1)   # constant over patches
    assert torch.allclose(got_identity, expected_identity, atol=1e-4)


def test_unique_text_pooling_matches_dense_values_and_gradients():
    """Deduplicating repeated text rows is an execution optimization, not a model change."""
    torch.manual_seed(12)
    dense_fusion = FactoredChannelTextFusion(
        d_model=16, text_dim=8, dropout=0.0, gate_bias_init=0.0
    ).eval()
    unique_fusion = FactoredChannelTextFusion(
        d_model=16, text_dim=8, dropout=0.0, gate_bias_init=0.0
    ).eval()
    unique_fusion.load_state_dict(dense_fusion.state_dict())

    B, P, C, S = 3, 2, 6, 5
    role_unique = torch.randn(4, S, 8)
    role_unique_mask = torch.ones(4, S, dtype=torch.bool)
    role_ids = torch.tensor([[0, 1, 2, 0, 1, 2],
                             [0, 1, 2, 3, 3, 3],
                             [2, 1, 0, 2, 1, 0]])
    sensor_unique = torch.randn(3, S, 8)
    sensor_unique_mask = torch.ones(3, S, dtype=torch.bool)
    sensor_text_ids = torch.tensor([[0, -1], [1, 2], [2, -1]])
    sensor_id = torch.tensor([[0] * C, [0, 0, 0, 1, 1, 1], [0] * C])

    role_dense = role_unique[role_ids]
    role_mask_dense = role_unique_mask[role_ids]
    sensor_dense = sensor_unique[sensor_text_ids.clamp_min(0)]
    sensor_mask_dense = sensor_unique_mask[sensor_text_ids.clamp_min(0)]
    sensor_mask_dense[sensor_text_ids < 0] = False
    sensor_dense = sensor_dense.masked_fill(~sensor_mask_dense.unsqueeze(-1), 0.0)
    dense_tokens = torch.randn(B, P, C, 16, requires_grad=True)
    unique_tokens = dense_tokens.detach().clone().requires_grad_(True)

    dense = dense_fusion(dense_tokens, role_dense, role_mask_dense,
                         sensor_dense, sensor_mask_dense, sensor_id)
    unique = unique_fusion(unique_tokens, role_unique, role_unique_mask,
                           sensor_unique, sensor_unique_mask, sensor_id,
                           role_text_ids=role_ids, sensor_text_ids=sensor_text_ids)
    assert torch.allclose(dense, unique, atol=1e-6, rtol=1e-5)

    dense.square().mean().backward()
    unique.square().mean().backward()
    assert torch.allclose(dense_tokens.grad, unique_tokens.grad, atol=1e-6, rtol=1e-5)
    for dense_parameter, unique_parameter in zip(
        dense_fusion.parameters(), unique_fusion.parameters()
    ):
        assert torch.allclose(
            dense_parameter.grad, unique_parameter.grad, atol=1e-6, rtol=1e-5
        )


# --------------------------------------------------------------------------- encoder integration
def test_encoder_factored_forward_runs_and_legacy_is_unchanged():
    """Both paths run; and constructing 'per_channel' still builds the legacy fusion."""
    from model.tokenizer.channel_text import ChannelTextFusion
    legacy = SetTokenizerEncoder(d_model=32, num_layers=1, num_heads=4, dim_feedforward=64)
    assert isinstance(legacy.fusion, ChannelTextFusion)

    enc = SetTokenizerEncoder(d_model=32, num_layers=1, num_heads=4, dim_feedforward=64,
                              text_conditioning="factored").eval()
    assert isinstance(enc.fusion, FactoredChannelTextFusion)

    B, P, C, S = 2, 3, 6, enc.filterbank.S
    patches = torch.randn(B, P, S, C)
    role_texts = [["accelerometer x-axis"] * C for _ in range(B)]
    sensor_texts = [
        ["a watch on the left wrist; accelerometer includes gravity"],
        ["a phone in the left pocket; accelerometer includes gravity",
         "a watch on the right wrist; accelerometer includes gravity"],
    ]
    sensor_id = torch.tensor([[0] * C, [0, 0, 0, 1, 1, 1]])
    positions = torch.arange(P).float().unsqueeze(0).repeat(B, 1)
    out = enc(patches, 50.0, S, role_texts, positions,
              channel_mask=torch.ones(B, C, dtype=torch.bool),
              patch_padding_mask=torch.ones(B, P, dtype=torch.bool),
              sensor_texts=sensor_texts, sensor_id=sensor_id)
    assert out["pooled"].shape == (B, 32) and torch.isfinite(out["pooled"]).all()


def test_transformer_runtime_hook_keeps_checkpoint_keys_and_values():
    """The compile hook may replace execution, but it must not rewrite checkpoint structure."""
    torch.manual_seed(21)
    enc = SetTokenizerEncoder(
        d_model=24, num_layers=1, num_heads=4, dim_feedforward=48, dropout=0.0,
        text_conditioning="factored",
    ).eval()
    keys_before = tuple(enc.state_dict())
    enc._compiled_transformer_forward = enc.transformer.forward
    assert tuple(enc.state_dict()) == keys_before

    B, P, C, S = 1, 2, 6, enc.filterbank.S
    with torch.no_grad():
        out = enc(
            torch.randn(B, P, S, C), 50.0, S, [["x", "y", "z", "x", "y", "z"]],
            torch.arange(P).float().unsqueeze(0),
            channel_mask=torch.ones(B, C, dtype=torch.bool),
            patch_padding_mask=torch.ones(B, P, dtype=torch.bool),
            sensor_texts=[["a phone accelerometer in a pocket; includes gravity",
                           "a phone gyroscope in a pocket"]],
            sensor_id=torch.tensor([[0, 0, 0, 1, 1, 1]]),
        )
    assert torch.isfinite(out["pooled"]).all()


def test_factored_conditioning_changes_embedding_when_sensor_text_changes():
    torch.manual_seed(8)
    enc = SetTokenizerEncoder(d_model=24, num_layers=1, num_heads=4, dim_feedforward=48,
                              dropout=0.0, text_conditioning="factored").eval()
    B, P, C, S = 1, 2, 6, enc.filterbank.S
    patches = torch.randn(B, P, S, C)
    roles = [["accelerometer x-axis"] * C]
    positions = torch.arange(P).float().unsqueeze(0)
    kwargs = dict(
        channel_mask=torch.ones(B, C, dtype=torch.bool),
        patch_padding_mask=torch.ones(B, P, dtype=torch.bool),
        sensor_id=torch.zeros(B, C, dtype=torch.long),
    )
    with torch.no_grad():
        wrist = enc(
            patches, 50.0, S, roles, positions,
            sensor_texts=[["a watch on the left wrist; accelerometer includes gravity"]],
            **kwargs,
        )["pooled"]
        pocket = enc(
            patches, 50.0, S, roles, positions,
            sensor_texts=[["a phone in the right pocket; accelerometer includes gravity"]],
            **kwargs,
        )["pooled"]
    assert not torch.allclose(wrist, pocket, atol=1e-6), "sensor-level text has no operational effect"


def test_encoder_accepts_two_complete_sensors_with_separate_descriptions():
    torch.manual_seed(9)
    enc = SetTokenizerEncoder(d_model=24, num_layers=1, num_heads=4, dim_feedforward=48,
                              dropout=0.0, text_conditioning="factored").eval()
    B, P, C, S = 1, 2, 12, enc.filterbank.S
    roles = [[
        "accelerometer x-axis", "accelerometer y-axis", "accelerometer z-axis",
        "gyroscope x-axis", "gyroscope y-axis", "gyroscope z-axis",
    ] * 2]
    with torch.no_grad():
        out = enc(
            torch.randn(B, P, S, C),
            50.0,
            S,
            roles,
            torch.arange(P).float().unsqueeze(0),
            channel_mask=torch.ones(B, C, dtype=torch.bool),
            patch_padding_mask=torch.ones(B, P, dtype=torch.bool),
            sensor_texts=[[
                "a phone in the pocket; accelerometer and gyroscope; accelerometer includes gravity",
                "a watch on the wrist; accelerometer and gyroscope; accelerometer includes gravity",
            ]],
            sensor_id=torch.tensor([[0] * 6 + [1] * 6]),
        )
    assert out["tokens"].shape == (B, P, C, 24)
    assert torch.isfinite(out["pooled"]).all()


def test_encoder_factored_requires_sensor_inputs():
    enc = SetTokenizerEncoder(d_model=16, num_layers=1, num_heads=2, dim_feedforward=32,
                              text_conditioning="factored")
    B, P, C, S = 1, 2, 6, enc.filterbank.S
    patches = torch.randn(B, P, S, C)
    positions = torch.zeros(B, P)
    role_texts = [["accelerometer x-axis"] * C]
    try:
        enc(patches, 50.0, S, role_texts, positions)     # missing sensor_texts/sensor_id
        raise AssertionError("expected ValueError for missing factored inputs")
    except ValueError:
        pass


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
