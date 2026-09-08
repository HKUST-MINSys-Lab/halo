from types import SimpleNamespace

import numpy as np
import torch
import pytest

from training.tokenizer.pretrain_data import PretrainDataset, WindowKey


def test_raw_window_copy_owns_storage_and_preserves_valid_prefix(tmp_path, monkeypatch):
    import training.tokenizer.pretrain_data as data

    original = np.arange(2 * 12 * 6, dtype=np.float64).reshape(2, 12, 6)
    path = tmp_path / "grid.npy"
    np.save(path, original)
    grid = np.load(path, mmap_mode="r")
    dataset = PretrainDataset.__new__(PretrainDataset)
    dataset.neutral_acquisition_text = True
    dataset._grid = lambda _: grid
    dataset._lengths = lambda _: np.array([12, 7])
    monkeypatch.setattr(data, "stream_sensor_texts", lambda *args, **kw: (
        ["x", "y", "z"] * 2, ["accelerometer", "gyroscope"], [0, 0, 0, 1, 1, 1],
    ))
    monkeypatch.setattr(data, "_stream_gravity_state", lambda *args: "present")
    ref = SimpleNamespace(dataset="test", stream="wrist", mask=[True] * 6, rate_hz=50.)
    sample = dataset._raw_sample(ref, WindowKey(0, 1, 0), ["channel"] * 6)
    assert sample.data.dtype == torch.float32
    torch.testing.assert_close(sample.data, torch.tensor(original[1, :7], dtype=torch.float32))
    sample.data.add_(1000)
    np.testing.assert_array_equal(grid, original)


@pytest.mark.parametrize("seconds", [0.4, 1., 1.4])
def test_collate_matches_scalar_reference_for_native_rates_and_short_windows(seconds):
    from training.tokenizer.pretrain_data import MultiScaleCollate, _physical_patch_bounds

    items = []
    for length, rate in [(1, 50.), (17, 100.), (300, 50.), (307, 51.2), (599, 100.)]:
        items.append({"data": torch.arange(length * 6, dtype=torch.float32).reshape(length, 6),
                      "rate": rate, "source_rate": rate + 1, "texts": ["channel"] * 6,
                      "label_id": 0, "channel_mask": torch.ones(6, dtype=torch.bool)})
    collated = MultiScaleCollate(fixed_patch_seconds=seconds)(items)
    names = ("patches", "patch_len", "rates", "source_rates", "positions",
             "patch_durations", "patch_padding_mask")
    expected = {name: torch.zeros_like(collated[name]) for name in names}
    for b, item in enumerate(items):
        rate = item["rate"]
        for p, (start, end) in enumerate(_physical_patch_bounds(len(item["data"]), rate, seconds)):
            length = end - start
            expected["patches"][b, p, :length] = item["data"][start:end]
            expected["patch_len"][b, p] = length
            expected["patch_durations"][b, p] = length / rate
            expected["positions"][b, p] = (start + 0.5 * length) / rate
            expected["patch_padding_mask"][b, p] = True
        expected["rates"][b] = rate
        expected["source_rates"][b] = item["source_rate"]
    for name in names:
        torch.testing.assert_close(collated[name], expected[name], rtol=0, atol=0)
