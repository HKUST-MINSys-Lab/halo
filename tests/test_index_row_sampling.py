"""`sample_index_rows` must select exactly what the per-label rescan selected.

Building the active memory bank used to rescan all windows once per label, which was the entire
~300 ms cost of an active-index refresh. The grouping is now precomputed per memory mask. This is
only a legitimate optimisation if the chosen windows are identical. The function consumes its RNG
through configuration permutation, per-subject shuffles, and top-up, so a changed grouping order
would silently reshuffle later draws. `_reference_candidates` reproduces the original per-label scan
as the oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.evidence.patch_episodes import PatchTable


def _reference_candidates(table, mask, label):
    """The original `nonzero(mask & window_label.eq(label))` this replaced."""
    return torch.nonzero(mask & table.window_label.eq(label), as_tuple=True)[0]


def _bank(seed, *, n_windows=400, n_labels=9, n_subjects=11, n_cfg=4, patches_per_window=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_labels, n_windows)
    subj = rng.integers(0, n_subjects, n_windows)
    cfg = rng.integers(0, n_cfg, n_windows)
    event = rng.integers(0, max(2, n_windows // 3), n_windows)
    verified = rng.random(n_windows) < 0.5
    window = np.repeat(np.arange(n_windows), patches_per_window)
    return {
        "Z": torch.zeros(n_windows, 2),
        "y": torch.as_tensor(y, dtype=torch.long),
        "subj": torch.as_tensor(subj, dtype=torch.long),
        "cfg": torch.as_tensor(cfg, dtype=torch.long),
        "event": torch.as_tensor(event, dtype=torch.long),
        "event_verified": torch.as_tensor(verified, dtype=torch.bool),
        "patch": {
            "y": torch.as_tensor(np.repeat(y, patches_per_window), dtype=torch.long),
            "subj": torch.as_tensor(np.repeat(subj, patches_per_window), dtype=torch.long),
            "cfg": torch.as_tensor(np.repeat(cfg, patches_per_window), dtype=torch.long),
            "window": torch.as_tensor(window, dtype=torch.long),
            "event": torch.as_tensor(np.repeat(event, patches_per_window), dtype=torch.long),
            "event_verified": torch.as_tensor(
                np.repeat(verified, patches_per_window), dtype=torch.bool
            ),
        },
    }


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("windows_per_label", [1, 4, 16, 64])
def test_grouping_matches_the_per_label_rescan(seed, windows_per_label):
    bank = _bank(seed)
    table = PatchTable(bank)
    rng = np.random.default_rng(seed)
    mask = torch.as_tensor(rng.random(len(bank["y"])) < 0.8)

    grouped = table.label_candidate_windows(mask)
    for label in sorted(table.label_rows):
        assert grouped[label].tolist() == _reference_candidates(table, mask, label).tolist(), label

    # And the sampler built on it consumes the RNG identically, so the roster is unchanged.
    first = table.sample_index_rows(mask, windows_per_label, np.random.default_rng(99))
    second = table.sample_index_rows(mask, windows_per_label, np.random.default_rng(99))
    assert first.tolist() == second.tolist()


def test_grouping_is_cached_per_mask_and_does_not_go_stale():
    bank = _bank(3)
    table = PatchTable(bank)
    n = len(bank["y"])
    wide = torch.ones(n, dtype=torch.bool)
    narrow = torch.zeros(n, dtype=torch.bool)
    narrow[: n // 3] = True

    # Alternate so a single-entry or key-blind cache would return the wrong mask's grouping.
    for mask in (wide, narrow, wide, narrow):
        grouped = table.label_candidate_windows(mask)
        for label in sorted(table.label_rows):
            assert grouped[label].tolist() == _reference_candidates(table, mask, label).tolist()

    # Different masks must give different rosters; identical masks identical ones.
    assert (
        table.sample_index_rows(wide, 4, np.random.default_rng(7)).tolist()
        != table.sample_index_rows(narrow, 4, np.random.default_rng(7)).tolist()
    )
    assert (
        table.sample_index_rows(narrow, 4, np.random.default_rng(7)).tolist()
        == table.sample_index_rows(narrow.clone(), 4, np.random.default_rng(7)).tolist()
    )


def test_labels_with_no_eligible_windows_contribute_nothing():
    bank = _bank(5)
    table = PatchTable(bank)
    mask = torch.zeros(len(bank["y"]), dtype=torch.bool)
    mask[torch.as_tensor(bank["y"]).eq(0)] = True   # only label 0 survives
    grouped = table.label_candidate_windows(mask)
    assert len(grouped[0]) > 0
    assert all(len(grouped[label]) == 0 for label in table.label_rows if label != 0)
    rows = table.sample_index_rows(mask, 4, np.random.default_rng(1))
    assert torch.as_tensor(bank["patch"]["y"])[rows].unique().tolist() == [0]
