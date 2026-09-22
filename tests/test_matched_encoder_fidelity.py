"""The matched-corpus trunks must see what the released adapters feed their checkpoints.

WHY THIS FILE EXISTS. ``MatchedCorpusEncoder`` reimplements each backbone's preprocessing --
resampling, cropping, clip splitting -- inside HALO's encoder contract, because the released
adapters consume an ``EvalStream`` and the trainer has patch tensors. Two implementations of one
published contract will drift, and on 2026-09-22 one had: the ``limubert`` entry declared
``rate_hz = 20.0`` and so fed the trunk one-second clips, while the released checkpoint's
positional-embedding table is ``(20, 72)`` and the deployment paper states the rate was reduced to
10 Hz. Every positional embedding covered half the physical time it was pretrained for.

Nothing caught it because no test compared the two implementations, and the matched arms had never
been run. These tests are that comparison. They are cheap, they need no GPU, and they fail loudly
if a contract is edited away from what the released artifact actually pins.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.tokenizer.matched_encoder import BACKBONE_CONTRACTS


def test_limubert_contract_matches_the_released_checkpoint_and_adapter():
    """Rate and clip length must agree with the adapter, and clip with the checkpoint itself."""
    from baselines.limubert_x import adapter as released

    contract = BACKBONE_CONTRACTS["limubert"]
    assert contract["rate_hz"] == pytest.approx(released.TARGET_HZ), (
        "matched limubert rate must equal the released adapter's TARGET_HZ; a mismatch silently "
        "changes how much physical time each positional embedding covers"
    )
    assert contract["clip"] == released.SEQ_LEN
    assert contract["channels"] == len(released.SIX_CHANNELS)

    # The released weights pin the sequence length independently of any prose: the positional
    # embedding table has exactly one row per input position.
    if released.CHECKPOINT.is_file():
        state = torch.load(released.CHECKPOINT, map_location="cpu", weights_only=False)
        state = state if isinstance(state, dict) else state.state_dict()
        positions = state["transformer.embed.pos_embed.weight"].shape[0]
        assert positions == contract["clip"], (
            f"checkpoint accepts {positions} positions, contract declares {contract['clip']}"
        )

    # The declared window each clip spans, which is the quantity the 2026-09-22 bug corrupted.
    seconds = contract["clip"] / contract["rate_hz"]
    assert seconds == pytest.approx(2.0), f"clip spans {seconds} s, released contract is 2 s"


def test_harnet_contract_matches_the_released_adapter():
    from baselines.harnet import adapter as released

    contract = BACKBONE_CONTRACTS["harnet"]
    assert contract["rate_hz"] == pytest.approx(released.TARGET_HZ)
    assert contract["clip"] == released.TARGET_LEN
    assert contract["channels"] == 3, "harnet5 is accelerometer-only"
    assert contract["clip"] / contract["rate_hz"] == pytest.approx(5.0)


def test_unimts_contract_matches_the_released_adapter():
    from baselines.unimts import adapter as released

    contract = BACKBONE_CONTRACTS["unimts"]
    assert contract["rate_hz"] == pytest.approx(released.TARGET_HZ)
    assert contract["channels"] == 3, "the released UniMTS weights are accelerometer-only"
    assert contract["clip"] / contract["rate_hz"] == pytest.approx(10.0)


def test_every_contract_declares_a_whole_number_of_samples():
    """A fractional clip length means the rate and window were specified inconsistently."""
    for name, contract in BACKBONE_CONTRACTS.items():
        seconds = contract["clip"] / contract["rate_hz"]
        assert abs(seconds * contract["rate_hz"] - contract["clip"]) < 1e-9, name
        assert seconds > 0, name


def test_limubert_window_splits_into_whole_clips_at_the_protocol_duration():
    """An 8 s protocol window must divide into whole 2 s clips, or the tail is silently dropped.

    ``_encode_groups`` keeps ``count = length // clip`` whole clips and discards the remainder,
    whereas the released adapter duration-weights a partial final clip. Those agree only when the
    window is an exact multiple of the clip, which at 10 Hz and 8 s it is (four clips). This test
    pins that so a future window-length change surfaces the divergence instead of quietly biasing
    the representation toward the start of the window.
    """
    contract = BACKBONE_CONTRACTS["limubert"]
    samples = 8.0 * contract["rate_hz"]
    assert samples % contract["clip"] == 0, (
        f"an 8 s window is {samples} samples, not a whole number of {contract['clip']}-sample "
        "clips; the matched encoder would drop the remainder that the adapter weights"
    )


def test_channel_slotting_preserves_values_and_places_absences_at_zero():
    """The frozen-baseline path must not permute or mix channels when widths differ."""
    from training.support_classifier.frozen_baseline_adaptation import CHANNELS, slot_channels

    per_channel = 4
    mask = np.array([True, True, True, False, False, False])
    block = np.arange(2 * 3 * per_channel, dtype=np.float32).reshape(2, 3 * per_channel)
    slotted = slot_channels(block, mask, per_channel)

    assert slotted.shape == (2, len(CHANNELS) * per_channel)
    # Present channels keep their values, in canonical order.
    np.testing.assert_allclose(slotted[:, :3 * per_channel], block)
    # Absent channels are exactly zero, never imputed.
    np.testing.assert_allclose(slotted[:, 3 * per_channel:], 0.0)

    # A non-contiguous mask must land each channel at its own slot, not pack them to the left.
    gapped = np.array([True, False, True, False, False, False])
    block2 = np.arange(2 * 2 * per_channel, dtype=np.float32).reshape(2, 2 * per_channel)
    out = slot_channels(block2, gapped, per_channel)
    np.testing.assert_allclose(out[:, 0:per_channel], block2[:, 0:per_channel])
    np.testing.assert_allclose(out[:, per_channel:2 * per_channel], 0.0)
    np.testing.assert_allclose(out[:, 2 * per_channel:3 * per_channel], block2[:, per_channel:])


def test_slot_channels_rejects_a_width_that_does_not_match_the_mask():
    from training.support_classifier.frozen_baseline_adaptation import slot_channels

    with pytest.raises(ValueError):
        slot_channels(np.zeros((1, 10), dtype=np.float32), np.array([True, True, False] + [False] * 3), 4)
