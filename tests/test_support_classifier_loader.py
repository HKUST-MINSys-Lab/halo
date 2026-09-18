import queue
from types import SimpleNamespace

import pytest
import torch

from training.support_classifier.collate import SupportCollate
from training.support_classifier.encoding import materialize_deferred_patches
from training.support_classifier.train import PrefetchLoader, build_dataset, episode_rng
from training.tokenizer.pretrain_data import (
    MultiResolutionCollate,
    _bounded_analysis_view,
    _bounded_analysis_views,
)


def _item(samples: int, channels: int, value: float) -> dict:
    sensors = channels // 3
    return {
        "data": torch.arange(samples * channels, dtype=torch.float32).reshape(samples, channels)
        * value,
        "rate": 50.0,
        "source_rate": 50.0,
        "channel_source_rates": torch.full((channels,), 50.0),
        "texts": ["axis"] * channels,
        "role_texts": ["axis"] * channels,
        "sensor_texts": ["sensor"] * sensors,
        "sensor_target_texts": ["sensor"] * sensors,
        "sensor_id": torch.arange(channels) // 3,
        "device_id": torch.arange(sensors) // 2,
        "sensor_placement": torch.arange(sensors) // 2,
        "label_id": 0,
        "channel_mask": torch.ones(channels, dtype=torch.bool),
        "gravity_state": "present",
        "source": "synthetic",
        "stream": "synthetic_stream",
        "window_index": 0,
        "subject": "synthetic_subject",
    }


def test_build_dataset_returns_the_configured_pretrain_dataset(monkeypatch):
    captured = {}

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr("training.support_classifier.train.PretrainDataset", FakeDataset)
    monkeypatch.setattr(
        "training.support_classifier.train.AugmentationConfig.phase_a",
        lambda **kwargs: ("augmentation", kwargs),
    )
    index = SimpleNamespace(train=[1, 2])
    args = SimpleNamespace(
        rate_augmentation_probability=0.0, modality_dropout_probability=0.0,
        neutral_acquisition_text=False, multi_device_probability=0.25, max_devices=3,
    )
    dataset = build_dataset(index, args)
    assert isinstance(dataset, FakeDataset)
    assert captured["args"] == (index, index.train)
    assert captured["kwargs"]["multi_device_probability"] == 0.25
    assert captured["kwargs"]["max_devices"] == 3


def test_deferred_patch_materialization_is_exact():
    base = MultiResolutionCollate(fixed_patch_seconds=(0.5, 1.0, 1.5))
    items = [_item(300, 6, 0.01), _item(237, 6, -0.02)]
    expected = base(items)
    deferred = base.deferred(items)
    actual = materialize_deferred_patches(deferred, torch.device("cpu"))
    assert torch.equal(actual, expected["patches"])
    for key in ("patch_len", "patch_padding_mask", "positions", "patch_durations",
                "resolution_ids"):
        assert torch.equal(deferred[key], expected[key])


def test_batched_long_patch_analysis_matches_individual_resampling():
    items = [_item(600, 6, 0.01), _item(600, 6, -0.02), _item(120, 6, 0.03)]
    items[0]["rate"] = items[1]["rate"] = 100.0
    items[2]["rate"] = 20.0
    durations = (0.5, 1.0, 2.0, 4.0)
    actual = _bounded_analysis_views(items, durations)
    for item, (values, rate) in zip(items, actual):
        expected, expected_rate = _bounded_analysis_view(
            item["data"], float(item["rate"]), durations,
        )
        assert rate == expected_rate
        assert torch.allclose(values, torch.as_tensor(expected), atol=1e-6, rtol=1e-6)


def test_support_collate_buckets_by_channel_width_and_restores_order():
    collate = SupportCollate(MultiResolutionCollate(
        fixed_patch_seconds=(0.5, 1.0, 1.5),
    ))
    bucketed = collate.bucketed([
        _item(300, 12, 0.01), _item(300, 6, 0.02), _item(300, 12, 0.03),
    ])
    assert [batch["compact_data"].shape[-1] for batch in bucketed.batches] == [6, 12]
    assert [indices.tolist() for indices in bucketed.row_indices] == [[1], [0, 2]]
    assert bucketed.restore_order.tolist() == [1, 0, 2]
    restored = torch.cat([
        indices.float().unsqueeze(1) for indices in bucketed.row_indices
    ]).index_select(0, bucketed.restore_order)
    assert restored.squeeze(1).tolist() == [0.0, 1.0, 2.0]


def _loader():
    loader = PrefetchLoader.__new__(PrefetchLoader)
    loader._pending = {}
    loader._consumed = set()
    loader._first_step = 1
    loader._next_request = 2
    loader._ahead = 0
    loader._requests = queue.Queue()
    loader._results = queue.Queue()
    return loader


def test_worker_death_is_reported_instead_of_hanging():
    loader = _loader()
    loader._processes = [SimpleNamespace(is_alive=lambda: False, pid=123, exitcode=-9)]
    with pytest.raises(RuntimeError, match="123, -9"):
        loader.get(1)


def test_consumed_step_cannot_block_on_second_request():
    loader = _loader()
    loader._results.put((1, [], {}, {}, None))
    assert loader.get(1) == ([], {}, {})
    with pytest.raises(ValueError, match="consumed"):
        loader.get(1)


def test_loader_returns_device_plans_when_worker_provides_them():
    loader = _loader()
    plans = [SimpleNamespace(relation="aligned")]
    loader._results.put((1, ["episode"], {"draw": 1}, {"batch": 1}, plans, None))
    assert loader.get(1) == (["episode"], {"draw": 1}, {"batch": 1}, plans)


def test_worker_exception_is_propagated():
    loader = _loader()
    loader._results.put((1, None, None, None, "bad data"))
    with pytest.raises(RuntimeError, match="bad data"):
        loader.get(1)


def test_step_rng_is_independent_of_worker_order():
    expected = episode_rng(42, 9).integers(0, 100, size=10).tolist()
    episode_rng(42, 8).integers(0, 100, size=100)
    assert episode_rng(42, 9).integers(0, 100, size=10).tolist() == expected


def test_normal_worker_shutdown_discards_unused_queue_buffers(monkeypatch):
    import training.support_classifier.train as train

    cancelled = []
    monkeypatch.setattr(train.torch, "set_num_threads", lambda _: None)
    train._prefetch_worker(
        None, None, None, 0, 8, {}, SimpleNamespace(get=lambda: None),
        SimpleNamespace(cancel_join_thread=lambda: cancelled.append(True)),
    )
    assert cancelled == [True]


@pytest.mark.parametrize("challenge_probability", [0.0, 1.0])
def test_synchronous_and_worker_batch_preparation_match(monkeypatch, challenge_probability):
    import training.support_classifier.train as train
    from training.support_classifier.sampling import Episode, Recording, SupportCorpus

    class Dataset:
        def aligned_device_members(self, position):
            return {name: position * 10 + offset for offset, name in enumerate(("a", "b", "c"))}

        def item_with_members(self, members, rng):
            return ("planned", tuple(members), float(rng.random()))

        def item_with_rng(self, position, rng):
            return ("independent", position, float(rng.random()))

    corpus = SupportCorpus(
        recordings=[Recording(0, pos, "dsads", "stream", "walk", "s", f"e{pos}")
                    for pos in range(3)], keys=[], stream_names=[],
    )
    episode = Episode(0, (1, 2), (0, 1), ("walk", "run"), 0, "compatible", 2,
                      False, support_window_groups=((1,), (2,)), support_set_id=7)
    monkeypatch.setattr(train, "draw_batch", lambda *a, **kw: ([episode], {"draw": 1}))
    monkeypatch.setattr(train.torch, "set_num_threads", lambda _: None)
    expected = train.prepare_training_batch(
        corpus, Dataset(), list, 42, 1, 1, {}, challenge_probability,
    )
    requests = iter((1, None))
    outputs = []
    train._prefetch_worker(
        corpus, Dataset(), list, 42, 1, {}, SimpleNamespace(get=lambda: next(requests)),
        SimpleNamespace(put=outputs.append, cancel_join_thread=lambda: None),
        challenge_probability,
    )
    assert outputs == [(1, *expected, None)]
    assert {item[0] for item in expected[2]} == {
        "planned" if challenge_probability else "independent"
    }
    assert (expected[3][0].relation != "not_applicable") == bool(challenge_probability)


def test_support_collate_row_chunking_preserves_rows_and_order():
    """A row-chunked bucketing must reassemble exactly the unchunked row order."""
    items = [_item(300, 12, 0.01), _item(300, 6, 0.02), _item(300, 12, 0.03),
             _item(300, 6, 0.04), _item(300, 12, 0.05)]
    plain = SupportCollate(MultiResolutionCollate(fixed_patch_seconds=(0.5, 1.0, 1.5)))
    chunked = SupportCollate(MultiResolutionCollate(fixed_patch_seconds=(0.5, 1.0, 1.5)),
                             max_rows_per_batch=2)
    a, b = plain.bucketed(items), chunked.bucketed(items)

    assert a.row_count == b.row_count == len(items)
    # Chunking splits the wide bucket; it must not merge or drop any row.
    assert len(b.batches) > len(a.batches)
    assert max(int(batch["compact_data"].shape[0]) for batch in b.batches) <= 2

    def restored(bucket):
        return torch.cat([
            index.float().unsqueeze(1) for index in bucket.row_indices
        ]).index_select(0, bucket.restore_order).squeeze(1).tolist()

    assert restored(a) == restored(b) == [float(row) for row in range(len(items))]

    # Every row's own payload must survive the split unchanged.
    def payload_by_row(bucket):
        out = {}
        for batch, index in zip(bucket.batches, bucket.row_indices):
            for position, row in enumerate(index.tolist()):
                out[row] = batch["compact_data"][position]
        return out

    left, right = payload_by_row(a), payload_by_row(b)
    assert set(left) == set(right)
    for row in left:
        assert torch.equal(left[row], right[row])


def test_support_collate_rejects_negative_chunk_size():
    with pytest.raises(ValueError):
        SupportCollate(MultiResolutionCollate(fixed_patch_seconds=(0.5,)), max_rows_per_batch=-1)
