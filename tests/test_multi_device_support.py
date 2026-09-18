"""Contracts shared by multi-device training and the expanded sealed evaluator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.base import BaselineAdapter
from baselines.data import (
    EvalStream, MultiDeviceEvalStream, load_multi_device_stream, source_slice_fingerprint,
)
from baselines.unimts.adapter import _joint_for
from data.scripts.scan_duplicates import cache_path as duplicate_cache_path
from data.scripts.scan_implausible import cache_path as implausible_cache_path
from data.scripts.build_grids import build_stream_specs
from model.tokenizer.encoder import hierarchical_device_pool
from training.support_classifier.sealed_eval import _cache_key, evaluation_cells
from training.tokenizer.pretrain_data import MultiScaleCollate, merge_device_items


def _stream(name: str, *, event_ids=("e0", "e1"), channels=3) -> EvalStream:
    names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"][:channels]
    return EvalStream(
        dataset="toy", stream=name, alignment="native",
        windows=np.zeros((len(event_ids), 8, channels), dtype=np.float32),
        gt=["walk"] * len(event_ids), subjects=np.asarray(["s"] * len(event_ids)),
        channels=names, rate_hz=4.0, mask=np.ones(channels, dtype=bool),
        eval_labels=["walk"], event_ids=np.asarray(event_ids, dtype=object),
        execution_ids=np.asarray(event_ids, dtype=object), lengths=np.full(len(event_ids), 8),
    )


def test_composite_intersects_exact_event_rows(monkeypatch):
    streams = {"a": _stream("a"), "b": _stream("b", event_ids=("e0", "other"))}
    monkeypatch.setattr("baselines.data.load_eval_stream", lambda dataset, stream, *a, **k: streams[stream])
    composite = load_multi_device_stream("toy", ("a", "b"), apply_quality_screen=False)
    assert composite.event_ids.tolist() == ["e0"]
    assert composite.n_alignment_excluded == 1


def test_composite_rejects_disagreeing_common_event_metadata(monkeypatch):
    streams = {"a": _stream("a"), "b": _stream("b")}
    streams["b"].gt[0] = "run"
    monkeypatch.setattr("baselines.data.load_eval_stream", lambda dataset, stream, *a, **k: streams[stream])
    with pytest.raises(ValueError, match="gt"):
        load_multi_device_stream("toy", ("a", "b"), apply_quality_screen=False)


def test_feature_cache_distinguishes_single_and_ordered_composite():
    a, b = _stream("a"), _stream("b")
    composite = MultiDeviceEvalStream(
        dataset="toy", cell_id="a+b", devices=[a, b], device_ids=["a", "b"],
        event_ids=a.event_ids, gt=a.gt, subjects=a.subjects, eval_labels=a.eval_labels,
        execution_ids=a.execution_ids, execution_identity_known=True,
        quality_screen="applied", n_quality_excluded=0, quality_excluded_by_device={},
        window_seconds=6.0, alignment="native",
    )
    assert len({_cache_key("m", a, "fp"), _cache_key("m", b, "fp"),
                _cache_key("m", composite, "fp")}) == 3


def test_source_fingerprint_covers_valid_samples_but_not_padding():
    original = _stream("a")
    changed_padding = _stream("a")
    changed_padding.lengths[:] = 4
    original.lengths[:] = 4
    changed_padding.windows[:, 4:] = 99
    assert source_slice_fingerprint(original) == source_slice_fingerprint(changed_padding)

    changed_signal = _stream("a")
    changed_signal.lengths[:] = 4
    changed_signal.windows[0, 0, 0] = 1
    assert source_slice_fingerprint(original) != source_slice_fingerprint(changed_signal)


def test_declared_unimts_multi_device_cells_use_distinct_joints():
    realworld = [_stream(name) for name in ("phone_forearm", "phone_thigh", "phone_waist")]
    shoaib = [_stream(name) for name in (
        "phone_left_pocket", "phone_right_pocket", "watch_wrist_proxy", "phone_belt",
    )]
    assert len({_joint_for(stream) for stream in realworld}) == len(realworld)
    assert len({_joint_for(stream) for stream in shoaib}) == len(shoaib)


def test_evaluation_matrix_expands_all_durations_and_composites():
    cells = evaluation_cells((16, 4, 8, 4))
    assert {duration for duration, *_ in cells} == {4.0, 8.0, 16.0}
    assert sum(bool(devices) for *_, devices in cells) == 6


def test_explicit_grid_build_keeps_primary_placement_proxies():
    keys = {(spec.dataset, spec.stream_id) for spec in build_stream_specs(("shoaib", "ut_complex"))}
    assert ("shoaib", "watch_wrist_proxy") in keys
    assert ("ut_complex", "watch_wrist") in keys


def test_capped_real_multi_device_corpus_keeps_aligned_smoke_rows():
    from training.tokenizer.pretrain_data import CorpusIndex, PretrainDataset

    index = CorpusIndex(max_per_stream=20, datasets=("dsads",), alignment="native")
    dataset = PretrainDataset(
        index, index.train, augment=False, two_view=False,
        multi_device_probability=1.0, max_devices=4,
    )

    assert dataset._aligned_devices
    item = dataset[next(iter(dataset._aligned_devices))]
    assert int(item["device_id"].max()) >= 1


def test_driver_fans_out_non_native_adapter_once():
    class Adapter(BaselineAdapter):
        def window_features(self, stream, state, device):
            value = 1.0 if stream.stream == "a" else 3.0
            return np.asarray([[value, 1.0], [value, 1.0]], dtype=np.float32)

    a, b = _stream("a"), _stream("b")
    composite = MultiDeviceEvalStream(
        "toy", "a+b", [a, b], ["a", "b"], a.event_ids, a.gt, a.subjects,
        a.eval_labels, a.execution_ids, True, "applied", 0, {}, 6.0, "native",
    )
    result = Adapter().features_for_stream(composite, None, "cpu")
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)


def test_hierarchical_pool_weights_devices_not_modalities():
    # Device 0 has accel+gyro rows with value 2; device 1 has accel only with value 6.
    h = torch.tensor([[[[2.0], [2.0], [6.0]]]])
    weights = torch.ones(1, 1, 3)
    present = torch.ones(1, 3, dtype=torch.bool)
    pooled, per_device, valid = hierarchical_device_pool(
        h, weights, present, torch.tensor([[0, 0, 1]]),
    )
    torch.testing.assert_close(per_device.flatten(), torch.tensor([2.0, 6.0]))
    torch.testing.assert_close(pooled.flatten(), torch.tensor([4.0]))
    assert valid.all()


def test_single_device_pool_is_bit_identical_to_historical_mean():
    torch.manual_seed(4)
    h = torch.randn(2, 3, 2, 8)
    weights = torch.randint(0, 2, (2, 3, 2)).float()
    weights[:, :, 0] = 1
    expected = (h * weights.unsqueeze(-1)).sum(2) / weights.sum(2).clamp_min(1).unsqueeze(-1)
    actual, _, _ = hierarchical_device_pool(h, weights, weights.amax(1).bool(), torch.zeros(2, 2, dtype=torch.long))
    assert torch.equal(actual, expected)


def test_composite_collate_pads_channels_and_preserves_device_ids():
    def item(value, sensors):
        channels = sensors * 3
        return {
            "data": torch.full((8, channels), value), "rate": 4.0, "source_rate": 4.0,
            "texts": ["axis"] * channels, "role_texts": ["axis"] * channels,
            "sensor_texts": ["sensor"] * sensors, "sensor_target_texts": ["sensor"] * sensors,
            "sensor_id": torch.arange(sensors).repeat_interleave(3),
            "sensor_modality": torch.tensor([0, 1][:sensors]),
            "sensor_gravity": torch.tensor([0, 3][:sensors]),
            "sensor_rates_hz": torch.tensor([[4.0, 4.0]] * sensors),
            "device_id": torch.zeros(sensors, dtype=torch.long),
            "sensor_bias": torch.zeros(sensors, 14), "sensor_placement": torch.zeros(sensors, dtype=torch.long),
            "channel_mask": torch.ones(channels, dtype=torch.bool), "label_id": 0,
        }
    composite = merge_device_items([item(1.0, 2), item(2.0, 1)])
    output = MultiScaleCollate(fixed_patch_seconds=1.0, dft_size=16)([item(0.0, 2), composite])
    assert output["patches"].shape[-1] == 9
    assert output["channel_mask"].sum(1).tolist() == [6, 9]
    assert output["device_id"][1].tolist() == [0, 0, 1]
    assert output["sensor_modality"][1].tolist() == [0, 1, 0]
    assert output["sensor_rates_hz"].shape == (2, 3, 2)


def test_quality_artifacts_are_duration_qualified_without_moving_legacy_six_second_path():
    assert duplicate_cache_path("native", 6.0).name == "duplicate_windows.json"
    assert implausible_cache_path("native", 6.0).name == "implausible_windows.json"
    assert duplicate_cache_path("native", 4.0).name == "duplicate_windows_w4.json"
    assert implausible_cache_path("native", 16.0).name == "implausible_windows_w16.json"
