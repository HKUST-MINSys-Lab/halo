"""M3 gate tests (build plan M3): the set encoder must be permutation- and
count-invariant over channels (identity = TEXT, not position), physical-time aware
(RoPE over seconds), and support JEPA masking across temporal resolutions.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.tokenizer.encoder import SetTokenizerEncoder

RATE = 60.0
N = 60          # 1 s patches at 60 Hz
S = 256


class _PassFusion(nn.Module):
    def forward(self, sensor_tokens, text_tokens, text_mask):
        return sensor_tokens


class _PassTransformer(nn.Module):
    def forward(self, x, **kwargs):
        return x


@pytest.fixture(scope="module")
def enc():
    torch.manual_seed(0)
    e = SetTokenizerEncoder(d_model=64, num_layers=2, num_heads=4,
                            dropout=0.0, dft_size=S)
    e.eval()
    return e


def make_batch(b=2, p=6, c=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    patches = torch.zeros(b, p, S, c)
    patches[:, :, :N] = torch.randn(b, p, N, c, generator=g) * 0.1
    texts = [
        [f"accelerometer {a}-axis at the wrist" for a in "xyz"] +
        [f"gyroscope {a}-axis at the wrist" for a in "xyz"]
    ][0][:c]
    positions = (torch.arange(p).float() * 1.0 + 0.5).unsqueeze(0).expand(b, p)
    return patches, [list(texts)] * b, positions


def test_duration_metadata_is_bounded_continuous_and_changes_tokens():
    torch.manual_seed(3)
    model = SetTokenizerEncoder(
        d_model=16, num_layers=1, num_heads=4, dim_feedforward=32, dropout=0.0,
        dft_size=S, use_duration_embedding=True,
    )
    model.fusion = _PassFusion()
    model.transformer = _PassTransformer()
    sensor = torch.zeros(1, 2, 1, 16)
    text = torch.zeros(1, 1, 1, 384)
    text_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    positions = torch.tensor([[1.0, 1.0]])
    durations = torch.tensor([[0.4, 1.5]])
    out = model.encode(sensor, text, text_mask, positions, patch_durations=durations)
    assert torch.isfinite(out["tokens"]).all()
    assert not torch.allclose(out["tokens"][:, 0], out["tokens"][:, 1])


def test_multiresolution_pooling_weights_scales_not_tokens():
    model = SetTokenizerEncoder(
        d_model=4, num_layers=1, num_heads=1, dim_feedforward=8, dropout=0.0,
        dft_size=S,
    )
    model.fusion = _PassFusion()
    model.transformer = _PassTransformer()
    sensor = torch.tensor([0.0, 0.0, 0.0, 10.0]).view(1, 4, 1, 1).expand(1, 4, 1, 4)
    text = torch.zeros(1, 1, 1, 384)
    text_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    out = model.encode(
        sensor, text, text_mask, torch.arange(4).view(1, 4).float(),
        resolution_ids=torch.tensor([[0, 0, 0, 1]]),
        patch_padding_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    assert torch.allclose(out["pooled"], torch.full((1, 4), 5.0))


def test_three_resolution_pooling_weights_each_scale_equally():
    model = SetTokenizerEncoder(
        d_model=4, num_layers=1, num_heads=1, dim_feedforward=8, dropout=0.0,
        dft_size=S, num_resolutions=3,
    )
    model.fusion = _PassFusion()
    model.transformer = _PassTransformer()
    values = torch.tensor([0.0, 0.0, 10.0, 20.0, 20.0])
    sensor = values.view(1, 5, 1, 1).expand(1, 5, 1, 4)
    text = torch.zeros(1, 1, 1, 384)
    text_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    out = model.encode(
        sensor, text, text_mask, torch.arange(5).view(1, 5).float(),
        resolution_ids=torch.tensor([[0, 0, 1, 2, 2]]),
        patch_padding_mask=torch.ones(1, 5, dtype=torch.bool),
    )
    assert torch.allclose(out["pooled"], torch.full((1, 4), 10.0))


def test_resolutions_contextualize_each_other():
    torch.manual_seed(7)
    model = SetTokenizerEncoder(
        d_model=16, num_layers=1, num_heads=4, dim_feedforward=32, dropout=0.0,
        dft_size=S,
    ).eval()
    model.fusion = _PassFusion()
    sensor = torch.randn(1, 4, 1, 16)
    changed = sensor.clone()
    changed[:, 2:] += 100.0                       # perturb only resolution 1
    text = torch.zeros(1, 1, 1, 384)
    text_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    positions = torch.tensor([[0.25, 0.75, 0.5, 1.5]])
    groups = torch.tensor([[0, 0, 1, 1]])
    a = model.encode(sensor, text, text_mask, positions, resolution_ids=groups)
    b = model.encode(changed, text, text_mask, positions, resolution_ids=groups)
    assert not torch.allclose(a["tokens"][:, :2], b["tokens"][:, :2], atol=1e-5)


def test_encoder_can_expose_teacher_layer_states_without_changing_final_output():
    torch.manual_seed(11)
    model = SetTokenizerEncoder(
        d_model=16, num_layers=2, num_heads=4, dim_feedforward=32, dropout=0.0,
        dft_size=S,
    ).eval()
    model.fusion = _PassFusion()
    sensor = torch.randn(1, 4, 1, 16)
    text = torch.zeros(1, 1, 1, 384)
    text_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    positions = torch.arange(4).float().view(1, 4)
    ordinary = model.encode(sensor, text, text_mask, positions)
    layered = model.encode(
        sensor, text, text_mask, positions, return_layer_states=True,
    )
    assert len(layered["layer_states"]) == 2
    assert all(state.shape == layered["tokens"].shape for state in layered["layer_states"])
    assert torch.allclose(ordinary["tokens"], layered["tokens"])


def test_padded_future_signal_cannot_change_observed_encoder_states():
    torch.manual_seed(12)
    model = SetTokenizerEncoder(
        d_model=16, num_layers=2, num_heads=4, dim_feedforward=32, dropout=0.0,
        dft_size=S,
    ).eval()
    model.fusion = _PassFusion()
    tokens = torch.randn(1, 5, 1, 16)
    changed = tokens.clone()
    changed[:, 3:] += 10_000
    text = torch.zeros(1, 1, 1, 384)
    text_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    positions = torch.arange(5).float().view(1, 5)
    visible = torch.tensor([[True, True, True, False, False]])
    with torch.no_grad():
        ordinary = model.encode(
            tokens, text, text_mask, positions, patch_padding_mask=visible,
        )["tokens"]
        perturbed = model.encode(
            changed, text, text_mask, positions, patch_padding_mask=visible,
        )["tokens"]
    assert torch.allclose(ordinary[:, :3], perturbed[:, :3], atol=1e-6)


# ------------------------------------------------------------- variable channel counts
@pytest.mark.parametrize("c", [3, 6, 9, 12])
def test_consumes_any_channel_count_unchanged(enc, c):
    b, p = 2, 4
    patches = torch.zeros(b, p, S, c)
    patches[:, :, :N] = torch.randn(b, p, N, c) * 0.1
    texts = [[f"sensor {i} at some placement" for i in range(c)]] * b
    positions = (torch.arange(p).float() + 0.5).unsqueeze(0).expand(b, p)
    with torch.no_grad():
        out = enc(patches, RATE, torch.tensor([N] * b), texts, positions)
    assert out["tokens"].shape == (b, p, c, 64)
    assert out["pooled"].shape == (b, 64)
    assert torch.isfinite(out["tokens"]).all()


# ---------------------------------------------------------- permutation invariance
def test_channel_permutation_equivariance(enc):
    """Permuting channels AND their texts must permute per-channel outputs and leave
    the pooled representation unchanged — channels carry no positional identity."""
    patches, texts, positions = make_batch()
    perm = torch.tensor([4, 2, 0, 5, 1, 3])
    patches_p = patches[:, :, :, perm]
    texts_p = [[t[i] for i in perm.tolist()] for t in texts]
    with torch.no_grad():
        out = enc(patches, RATE, torch.tensor([N, N]), texts, positions)
        out_p = enc(patches_p, RATE, torch.tensor([N, N]), texts_p, positions)
    assert torch.allclose(out["tokens"][:, :, perm], out_p["tokens"], atol=1e-4), \
        "per-channel outputs must permute with the channels"
    assert torch.allclose(out["pooled"], out_p["pooled"], atol=1e-4), \
        "pooled representation must be permutation-INVARIANT"


def test_channel_identity_comes_from_text(enc):
    """Same signal, different channel TEXT -> different representation (the text is
    load-bearing, not decorative)."""
    patches, texts, positions = make_batch()
    texts_alt = [[t.replace("wrist", "ankle") for t in sample] for sample in texts]
    with torch.no_grad():
        out = enc(patches, RATE, torch.tensor([N, N]), texts, positions)
        out_alt = enc(patches, RATE, torch.tensor([N, N]), texts_alt, positions)
    assert not torch.allclose(out["pooled"], out_alt["pooled"], atol=1e-3), \
        "changing the placement text must change the representation"


def test_forward_respects_native_source_bandwidth(enc):
    patches, texts, positions = make_batch(b=2, p=3, c=6, seed=12)
    rates = torch.tensor([50.0, 50.0])
    lengths = torch.tensor([50, 50])
    with torch.no_grad():
        stored_clock = enc(
            patches[:, :3], rates, lengths, texts, positions[:, :3],
            source_rate_hz=rates,
        )
        acquisition_clock = enc(
            patches[:, :3], rates, lengths, texts, positions[:, :3],
            source_rate_hz=torch.tensor([25.0, 25.0]),
        )
    assert not torch.allclose(stored_clock["pooled"], acquisition_clock["pooled"]), \
        "public forward ignored the acquisition-rate observability bound"


# ------------------------------------------------------------------- physical time
def test_rope_time_shift_invariance(enc):
    """RoPE is relative: shifting ALL patch times by a constant leaves attention —
    and thus the output — unchanged. (Index-PE would fail this only if indices moved;
    the real content of the test is that ABSOLUTE time does not leak in.)"""
    patches, texts, positions = make_batch()
    with torch.no_grad():
        out = enc(patches, RATE, torch.tensor([N, N]), texts, positions)
        out_shift = enc(patches, RATE, torch.tensor([N, N]), texts, positions + 137.0)
    assert torch.allclose(out["pooled"], out_shift["pooled"], atol=1e-3)


def test_rope_spacing_matters(enc):
    """Patch SPACING (physical seconds) must matter: the same patches presented at
    1 s spacing vs 3 s spacing are different physical stories. Checked on PER-PATCH
    outputs — mean-pooling over patches largely cancels attention re-weighting, so
    `pooled` is the wrong place to look for this effect."""
    patches, texts, positions = make_batch()
    with torch.no_grad():
        out1 = enc(patches, RATE, torch.tensor([N, N]), texts, positions)
        out3 = enc(patches, RATE, torch.tensor([N, N]), texts, positions * 3.0)
    diff = (out1["per_patch"] - out3["per_patch"]).abs().max()
    assert diff > 1e-3, f"physical patch spacing had no effect on per-patch outputs ({diff})"


# ------------------------------------------------------------------ mask machinery
def test_token_mask_changes_masked_positions_only_at_input(enc):
    patches, texts, positions = make_batch()
    mask = torch.zeros(2, 6, 6, dtype=torch.bool)
    mask[:, 2:4, :] = True
    with torch.no_grad():
        out = enc(patches, RATE, torch.tensor([N, N]), texts, positions, token_mask=mask)
    assert torch.isfinite(out["tokens"]).all()
    with torch.no_grad():
        out_clean = enc(patches, RATE, torch.tensor([N, N]), texts, positions)
    assert not torch.allclose(out["tokens"], out_clean["tokens"], atol=1e-3)


def test_channel_mask_blocks_contribution(enc):
    """A masked-out channel must not influence the pooled output: 6 channels with the
    last 3 masked == the 3-channel forward of the same signal."""
    patches, texts, positions = make_batch()
    cmask = torch.tensor([[True, True, True, False, False, False]] * 2)
    with torch.no_grad():
        out6 = enc(patches, RATE, torch.tensor([N, N]), texts, positions,
                   channel_mask=cmask)
        out3 = enc(patches[:, :, :, :3], RATE, torch.tensor([N, N]),
                   [t[:3] for t in texts], positions)
    assert torch.allclose(out6["pooled"], out3["pooled"], atol=1e-4), \
        "masked channels leaked into the pooled representation"


# ---------------------------------------------------------------------- training
def test_gradients_flow_everywhere_except_frozen_text(enc):
    patches, texts, positions = make_batch()
    mask = torch.zeros(2, 6, 6, dtype=torch.bool)
    mask[:, 1, :] = True                       # exercise the [MASK] token so it gets grad
    enc.train()
    out = enc(patches, RATE, torch.tensor([N, N]), texts, positions, token_mask=mask)
    out["pooled"].sum().backward()
    grads = {n: p.grad is not None for n, p in enc.named_parameters()}
    assert all(grads.values()), [n for n, ok in grads.items() if not ok]
    enc.zero_grad()
    enc.eval()
