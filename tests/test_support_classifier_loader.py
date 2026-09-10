import queue
from types import SimpleNamespace

import pytest

from training.support_classifier.train import PrefetchLoader, episode_rng


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
