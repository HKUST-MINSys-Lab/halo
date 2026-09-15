"""Parity contracts for deployment-scenario performance optimizations."""

from __future__ import annotations

import gc
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

from baselines.data import EvalStream, source_slice_fingerprint
from baselines.scoring import paired_subject_bootstrap_difference
from training.support_classifier.partial_coverage import zscore
from training.support_classifier.run_scenarios import _PairedDeltaTracker
from training.support_classifier.sealed_eval import (
    FEATURE_CACHE_SCHEMA,
    FeatureMemoryCache,
    _cache_key,
    _file_hash,
    _load_or_encode,
)


def _legacy_paired_bootstrap(gt, pred, control, subjects, *, metric, B, seed):
    """The pre-vectorization estimator, retained here solely as a parity oracle."""
    gt, pred, control, subjects = map(np.asarray, (gt, pred, control, subjects))
    unique = np.unique(subjects)
    if metric == "f1_macro":
        classes = sorted(set(gt.tolist()) | set(pred.tolist()) | set(control.tolist()))
        score = lambda y, p: f1_score(  # noqa: E731
            y, p, labels=classes, average="macro", zero_division=0,
        ) * 100
    elif metric == "balanced_accuracy":
        classes = sorted(set(gt.tolist()))
        score = lambda y, p: recall_score(  # noqa: E731
            y, p, labels=classes, average="macro", zero_division=0,
        ) * 100
    else:
        score = lambda y, p: accuracy_score(y, p) * 100  # noqa: E731
    rows = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(B):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([rows[subject] for subject in sampled])
        values.append(score(gt[selected], pred[selected]) - score(gt[selected], control[selected]))
    return score(gt, pred) - score(gt, control), np.percentile(values, [2.5, 97.5])


@pytest.mark.parametrize("metric", ["f1_macro", "balanced_accuracy", "accuracy"])
def test_vectorized_paired_bootstrap_is_exactly_the_historical_estimator(metric):
    rng = np.random.default_rng(41)
    classes = np.asarray(["a", "b", "c", "d"])
    truth = classes[rng.integers(0, len(classes), size=240)]
    prediction = classes[rng.integers(0, len(classes), size=240)]
    control = classes[rng.integers(0, len(classes), size=240)]
    subjects = np.asarray([f"s{value}" for value in rng.integers(0, 12, size=240)])
    expected_point, expected_ci = _legacy_paired_bootstrap(
        truth, prediction, control, subjects, metric=metric, B=80, seed=17,
    )
    actual = paired_subject_bootstrap_difference(
        truth, prediction, control, subjects, metric=metric, B=80, seed=17,
    )
    assert actual[f"{metric}_difference"] == pytest.approx(expected_point, abs=1e-12)
    assert actual[f"{metric}_difference_ci_lo"] == pytest.approx(expected_ci[0], abs=1e-12)
    assert actual[f"{metric}_difference_ci_hi"] == pytest.approx(expected_ci[1], abs=1e-12)


def test_feature_memory_cache_is_bounded_and_lru():
    cache = FeatureMemoryCache(max_bytes=32)
    first = np.zeros((4,), dtype=np.float32)
    second = np.ones((4,), dtype=np.float32)
    third = np.full((4,), 2.0, dtype=np.float32)
    cache.put("first", first)
    cache.put("second", second)
    assert cache.get("first") is first  # make first most recently used
    cache.put("third", third)
    assert cache.get("second") is None
    assert cache.get("first") is first
    assert cache.get("third") is third


def test_stream_fingerprint_cache_does_not_retain_raw_streams():
    cache = FeatureMemoryCache(max_bytes=0)
    stream = EvalStream(
        dataset="toy", stream="wrist", alignment="native",
        windows=np.ones((1, 20, 3), dtype=np.float32), gt=["walk"],
        subjects=np.asarray(["a"]), channels=["acc_x", "acc_y", "acc_z"],
        rate_hz=20.0, mask=np.ones(3, dtype=bool), eval_labels=["walk"],
        window_seconds=1.0, event_ids=np.arange(1), execution_ids=np.asarray(["e1"]),
    )
    identity = id(stream)
    cache.stream_fingerprint(stream)
    assert identity in cache._stream_fingerprints
    del stream
    gc.collect()
    assert identity not in cache._stream_fingerprints


def test_vectorized_zscore_matches_rowwise_definition():
    rng = np.random.default_rng(9)
    scores = rng.normal(size=(97, 13))
    mask = rng.random(scores.shape) > 0.35
    expected = np.zeros_like(scores)
    for row in range(len(scores)):
        live = scores[row, mask[row]]
        if len(live) < 2:
            continue
        spread = float(live.std())
        scale = max(1.0, float(np.abs(live).max()))
        if spread > np.finfo(scores.dtype).eps * scale * 16.0:
            expected[row, mask[row]] = (live - live.mean()) / spread
    assert np.allclose(zscore(scores, mask), expected, rtol=1e-12, atol=1e-12)


def test_load_or_encode_reuses_a_valid_prior_cache(tmp_path):
    stream = EvalStream(
        dataset="toy", stream="wrist", alignment="native",
        windows=np.ones((3, 20, 3), dtype=np.float32),
        gt=["walk"] * 3, subjects=np.asarray(["a", "b", "c"]),
        channels=["acc_x", "acc_y", "acc_z"], rate_hz=20.0,
        mask=np.ones(3, dtype=bool), eval_labels=["walk"], window_seconds=1.0,
        event_ids=np.arange(3), execution_ids=np.asarray(["e1", "e2", "e3"]),
    )
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"immutable checkpoint identity")
    source = source_slice_fingerprint(stream)
    fingerprint = _file_hash(checkpoint)
    key = _cache_key("halo", stream, fingerprint, source_fingerprint=source)
    prior = tmp_path / "prior"
    prior.mkdir()
    filename = f"toy__wrist__halo__{key}.npy"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(prior / filename, expected)
    (prior / filename).with_suffix(".json").write_text(json.dumps({
        "cache_schema": FEATURE_CACHE_SCHEMA,
        "cache_key": key,
        "n_windows": 3,
        "artifact_fingerprint": fingerprint,
        "source_slice_fingerprint": source,
    }))
    result, result_fingerprint = _load_or_encode(
        name="halo", stream=stream, device=torch.device("cpu"),
        cache_dir=tmp_path / "new", halo_checkpoint=checkpoint,
        cache_read_dirs=(prior,), memory_cache=FeatureMemoryCache(1024),
    )
    assert np.array_equal(result, expected)
    assert result_fingerprint == fingerprint
    assert not (tmp_path / "new" / filename).exists()


def test_paired_tracker_releases_a_completed_control_group():
    tasks = [
        SimpleNamespace(meta={"matched_group": "g", "condition": "control"}),
        SimpleNamespace(meta={"matched_group": "g", "condition": "scenario"}),
    ]
    tracker = _PairedDeltaTracker(tasks, bootstrap=0)
    common = {"matched_group": "g", "model": "m", "readout": "1nn", "k": 1,
              "window_seconds": 8.0}
    control = [
        {**common, "condition": "control", "variant": "control", "query_event_id": str(i),
         "query_group_id": f"s{i % 2}", "truth": value, "prediction": value}
        for i, value in enumerate(("a", "a", "b", "b"))
    ]
    scenario = [
        {**common, "condition": "scenario", "variant": "shift", "query_event_id": str(i),
         "query_group_id": f"s{i % 2}", "truth": value, "prediction": "a"}
        for i, value in enumerate(("a", "a", "b", "b"))
    ]
    tracker.consume(control)
    tracker.consume(scenario)
    output = tracker.finish()
    assert len(output) == 1
    assert output[0]["f1_macro_delta"] < 0
    assert not tracker.controls
