from pathlib import Path

import numpy as np
import torch

from eval.run_adaptation_baselines import (
    _feature_cache_key,
    _load_cached_features,
    _save_cached_features,
    _source_fingerprint,
    score_positive_cell,
)


def test_feature_cache_round_trip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "features.npy"
    expected = np.arange(12, dtype=np.float32).reshape(4, 3)
    _save_cached_features(path, expected)

    np.testing.assert_array_equal(_load_cached_features(path, 4), expected)
    assert _load_cached_features(path, 5) is None


def test_feature_cache_key_covers_model_stream_source_artifact_and_config() -> None:
    base = dict(
        baseline_name="model", dataset="data", stream_id="stream",
        stream_fingerprint="stream-hash", source_fingerprint="source-hash",
        artifacts={"weights": {"sha256": "weights-hash"}}, config={"rate": 50},
    )
    expected = _feature_cache_key(**base)

    for field, replacement in (
        ("stream_fingerprint", "other-stream"),
        ("source_fingerprint", "other-source"),
        ("artifacts", {"weights": {"sha256": "other-weights"}}),
        ("config", {"rate": 100}),
    ):
        changed = dict(base)
        changed[field] = replacement
        assert _feature_cache_key(**changed) != expected


def test_external_source_tree_changes_source_fingerprint(tmp_path: Path) -> None:
    source = tmp_path / "upstream"
    source.mkdir()
    module = source / "model.py"
    module.write_text("VALUE = 1\n")

    class Adapter:
        name = "fake"

        def evaluation_source_paths(self):
            return (source,)

    before = _source_fingerprint(Adapter())
    module.write_text("VALUE = 2\n")

    assert _source_fingerprint(Adapter()) != before


def test_execution_pool_cache_is_reused_without_changing_scores() -> None:
    support = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    plan = {
        "subject": "s1", "candidate_names": ["a", "b"],
        "support_execution_rows": [[[0], [1]], [[2], [3]]],
        "support_execution_ids": [["a0", "a1"], ["b0", "b1"]],
        "query_rows": [0, 1], "query_execution_ids": ["q0", "q1"],
    }
    cache = {}
    kwargs = dict(
        query_features=query, support_features=support,
        query_labels=np.asarray(["a", "b"], dtype=object), plans=[plan],
        support_count=2, device=torch.device("cpu"), seed=7,
        methods=("nearest", "prototype", "ridge"), execution_feature_cache=cache,
    )

    first = score_positive_cell(**kwargs)
    size = len(cache)
    second = score_positive_cell(**kwargs)

    assert size == 4 == len(cache)
    assert first["nearest"] == second["nearest"]
    assert first["prototype"] == second["prototype"]
    assert first["ridge"] == second["ridge"]
