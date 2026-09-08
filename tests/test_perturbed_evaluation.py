"""The perturbation flag on the adaptation runner, and the Arm A rule it exercises."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from baselines.base import BaselineAdapter, UnsupportedEvaluationCell
from eval.data import EvalStream
from eval.enrollment_protocol import _positive_cell, _zero_shot_cell
from eval.perturbation import Perturbation


def _stream(name="wrist"):
    labels, subjects, executions, windows = [], [], [], []
    rng = np.random.default_rng(0)
    for subject in ("s1", "s2", "s3"):
        for label in ("walk", "sit"):
            for execution in range(4):
                for _ in range(2):
                    labels.append(label)
                    subjects.append(subject)
                    executions.append(f"{subject}:{label}:{execution}")
                    base = np.array([1.0, 0.0, 0.0] if label == "walk" else [0.0, 1.0, 0.0])
                    windows.append((base + 0.05 * rng.standard_normal(3))[None].repeat(12, 0))
    return EvalStream(
        dataset="synthetic", stream=name, alignment="native",
        windows=np.stack(windows).astype(np.float32), gt=labels,
        subjects=np.asarray(subjects, dtype=object), channels=["acc_x", "acc_y", "acc_z"],
        rate_hz=2.0, mask=np.ones(3, dtype=bool), eval_labels=["walk", "sit"],
        execution_ids=np.asarray(executions, dtype=object),
    )


def _manifest(stream):
    positive = _positive_cell(
        stream, stream, regime="ordinary", subject_relation="cross_subject",
        configuration_relation="same_configuration", support_counts=[0, 1, 2], seeds=[3],
    )
    return {
        "manifest_fingerprint": "manifest-test", "alignment": "native",
        "support_counts": [0, 1, 2],
        "cells": {
            "synthetic/wrist/zero_shot": _zero_shot_cell(stream, regime="ordinary"),
            "synthetic/wrist/from_wrist/same_configuration/cross_subject": positive,
        },
    }


class _Recorder(BaselineAdapter):
    """Frozen-feature fake that remembers which stream view it was asked to encode."""

    name = "perturb_fake"
    tier = "fake"
    seen: list = []

    def setup(self, device):
        return None

    def supports_native_zero_shot(self):
        return True

    def predict_candidates(self, value, candidates, state, device):
        return list(value.gt), {}

    def predict_candidates_from_features(self, features, candidates, state, device):
        names = np.asarray(list(candidates), dtype=object)
        return names[np.asarray(features).argmax(1)].tolist(), {}

    def window_features(self, value, state, device):
        type(self).seen.append(value)
        return np.asarray(value.windows[:, 0, :2], dtype=np.float32)


class _NativeRecorder(_Recorder):
    """Native adapter whose deployed rule refuses perturbed support, like Arm A."""

    name = "perturb_native_fake"

    def supports_native_enrollment(self):
        return True

    def predict_enrollment(self, query_stream, support_stream, plan, support_count,
                           candidate_texts, state, device, *, seed):
        if support_stream.perturbation is not None:
            raise UnsupportedEvaluationCell("deployed rule filters perturbed support out")
        names = list(plan["candidate_names"])
        return [names[0]] * len(plan["query_rows"]), {"rows": 1}


def _run(monkeypatch, tmp_path, adapter, perturbation, variant=None):
    import baselines
    import eval.run_adaptation_baselines as runner

    stream = _stream()
    manifest = _manifest(stream)
    type(adapter).seen = []
    baselines.REGISTRY[adapter.name] = adapter
    monkeypatch.setattr(runner, "load_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(runner, "load_eval_stream", lambda *a, **k: stream)
    try:
        return runner.run(
            baseline_name=adapter.name, manifest_path=tmp_path / "unused.json", device="cpu",
            out=tmp_path / "result.json", perturbation=perturbation, variant=variant,
        )
    finally:
        baselines.REGISTRY.pop(adapter.name, None)


def test_query_side_perturbation_touches_only_the_query_view(monkeypatch, tmp_path):
    payload = _run(monkeypatch, tmp_path, _Recorder(), Perturbation("orientation", "query"))
    views = {view.perturbation for view in _Recorder.seen}
    assert views == {None, "query_orientation_per_stream"}, "one grid, two roles, two views"
    assert payload["model"] == "perturb_fake@query_orientation_per_stream"
    assert payload["variant"] == "query_orientation_per_stream"
    assert payload["perturbation"]["axis"] == "orientation"
    enrollment = [v for v in payload["results"].values() if v.get("kind") == "enrollment"]
    assert enrollment and all("prototype" in v for v in enrollment)


def test_support_side_perturbation_is_an_honest_native_n_a(monkeypatch, tmp_path):
    payload = _run(monkeypatch, tmp_path, _NativeRecorder(), Perturbation("orientation", "support"))
    enrollment = [v for v in payload["results"].values() if v.get("kind") == "enrollment"]
    assert enrollment
    for value in enrollment:
        assert "support_comparator" not in value
        assert "filters perturbed support" in value["native_unsupported"]
        assert "prototype" in value, "matched frozen-feature readouts are still reported"
    views = {view.perturbation for view in _NativeRecorder.seen}
    assert views == {None, "support_orientation_per_stream"}, "the query side was untouched"


def test_perturbed_run_without_variant_still_gets_an_identity(monkeypatch, tmp_path):
    payload = _run(monkeypatch, tmp_path, _Recorder(), Perturbation("rate", "query", rate_hz=1.0))
    assert payload["model"] == "perturb_fake@query_rate_1hz"
    named = _run(monkeypatch, tmp_path, _Recorder(), Perturbation("rate", "query", rate_hz=1.0),
                 variant="custom")
    assert named["model"] == "perturb_fake@custom"


def test_unperturbed_run_is_unchanged(monkeypatch, tmp_path):
    payload = _run(monkeypatch, tmp_path, _Recorder(), None)
    assert payload["model"] == "perturb_fake" and payload["perturbation"] is None
    assert {view.perturbation for view in _Recorder.seen} == {None}


def test_inapplicable_channel_shift_records_na_without_aborting(monkeypatch, tmp_path):
    payload = _run(monkeypatch, tmp_path, _Recorder(), Perturbation("channel", "query"))
    assert payload["results"]
    assert all(row.get("status") == "n/a" for row in payload["results"].values())
    assert all("accelerometer-only" in row["reason"] for row in payload["results"].values())


def test_support_perturbation_does_not_report_unmodified_zero_shot(monkeypatch, tmp_path):
    payload = _run(monkeypatch, tmp_path, _Recorder(), Perturbation("orientation", "support"))
    rows = [r for r in payload["results"].values() if r.get("kind") == "zero_shot"]
    assert rows and all(r.get("status") == "n/a" for r in rows)


def test_warm_cache_restores_adapter_context(monkeypatch, tmp_path):
    import baselines
    import eval.run_adaptation_baselines as runner

    class OwnerAdapter(_Recorder):
        name = "owner_fake"
        encodes = 0

        def setup(self, device):
            return {}

        def window_features(self, stream, state, device):
            self.encodes += 1
            return super().window_features(stream, state, device)

        def restore_window_features(self, stream, features, state, device):
            state[id(features)] = stream

        def predict_candidates_from_features(self, features, candidates, state, device):
            assert state[id(features)].dataset == "synthetic"
            return super().predict_candidates_from_features(features, candidates, state, device)

    adapter = OwnerAdapter()
    stream = _stream()
    manifest = _manifest(stream)
    manifest["stream_fingerprints"] = {"synthetic/wrist": "fixture"}
    monkeypatch.setitem(baselines.REGISTRY, adapter.name, adapter)
    monkeypatch.setattr(runner, "load_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(runner, "load_eval_stream", lambda *a, **k: stream)
    for index in range(2):
        runner.run(baseline_name=adapter.name, manifest_path=tmp_path / "unused", device="cpu",
                   out=tmp_path / f"result{index}.json", feature_cache_dir=tmp_path / "cache")
    assert adapter.encodes == 1
