import dataclasses
import json

import numpy as np
import pytest

import eval.data as data
from eval.enrollment_protocol import stream_fingerprint
from eval.perturbation import Perturbation, perturb_stream


def _grid(tmp_path, monkeypatch):
    root = tmp_path / "synthetic" / "grids" / "native" / "wrist"
    root.mkdir(parents=True)
    np.save(root / "data.npy", np.zeros((3, 300, 6), dtype=np.float32))
    np.save(root / "mask.npy", np.ones(6, dtype=bool))
    np.save(root / "lengths.npy", np.array([108, 300, 200], dtype=np.int32))
    (root / "meta.json").write_text(json.dumps({
        "labels": ["walking"] * 3, "subjects": ["s1"] * 3,
        "event_ids": ["session:0", "session:1", "session:2"], "rate_hz": 50.,
        "channels": [f"{sensor}_{axis}" for sensor in ("acc", "gyro") for axis in "xyz"],
        "lengths_file": "lengths.npy",
    }))
    monkeypatch.setattr(data, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(data, "load_eval_labels", lambda *args: ["walking"])
    monkeypatch.setattr(data, "_recording_map", lambda *args: {})
    monkeypatch.setattr(data, "_quality_excluded", lambda *args: (np.array([1]), "applied"))
    return root


def test_quality_screen_keeps_lengths_aligned_and_fingerprinted(tmp_path, monkeypatch):
    _grid(tmp_path, monkeypatch)
    stream = data.load_eval_stream("synthetic", "wrist", alignment="native")
    assert stream.lengths.tolist() == [108, 200]
    assert stream.execution_identity_known
    changed = dataclasses.replace(stream, lengths=np.array([107, 200]))
    assert stream_fingerprint(stream) != stream_fingerprint(changed)


def test_declared_lengths_must_exist(tmp_path, monkeypatch):
    root = _grid(tmp_path, monkeypatch)
    (root / "lengths.npy").unlink()
    with pytest.raises(FileNotFoundError, match="declared lengths"):
        data.load_eval_stream("synthetic", "wrist", alignment="native")


def test_partial_gravity_filter_and_rate_resample_preserve_validity(tmp_path, monkeypatch):
    _grid(tmp_path, monkeypatch)
    stream = data.load_eval_stream("synthetic", "wrist", alignment="native")
    stream.gravity_state = "present"
    stream.channel_descriptions = ["acceleration; includes gravity"] * 6
    stream.windows[0, :108, 0] = 1 + .2 * np.sin(np.arange(108) / 5)
    stream.windows[1, :200, 0] = 1
    short = dataclasses.replace(stream, windows=stream.windows[:1, :108].copy(),
                                lengths=np.array([108]))
    padded = perturb_stream(stream, Perturbation("gravity", "query"))
    honest = perturb_stream(short, Perturbation("gravity", "query"))
    np.testing.assert_allclose(padded.windows[0, :108], honest.windows[0], atol=1e-6)
    assert not padded.windows[0, 108:].any()
    changed = perturb_stream(stream, Perturbation("rate", "query", rate_hz=25))
    assert changed.lengths.tolist() == [54, 100]
    assert not changed.windows[0, 54:].any()
