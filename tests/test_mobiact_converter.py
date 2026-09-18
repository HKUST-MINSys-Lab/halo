from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.datasets.mobiact.convert import discover_trials, find_annotated_root, load_trial


def _write_trial(path, *, missing=()):
    time = np.arange(100, dtype=np.float64) / 100.0
    frame = pd.DataFrame({
        "rel_time": time,
        "acc_x": np.sin(time), "acc_y": np.cos(time), "acc_z": np.ones_like(time) * 9.80665,
        "gyro_x": time, "gyro_y": time * 2, "gyro_z": time * 3,
    })
    frame = frame.drop(columns=list(missing))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_annotated_release_discovery_and_trial_conversion(tmp_path):
    root = tmp_path / "release" / "Annotated Data" / "WAL"
    source = root / "WAL_1_1_annotated.csv"
    _write_trial(source)
    annotated = find_annotated_root(tmp_path)
    trials = discover_trials(annotated)
    assert len(trials) == 1
    segments, stats = load_trial(source)
    assert len(segments) == 1
    assert stats["source_rate_hz_median"] == pytest.approx(100.0)
    frame = segments[0]
    assert list(frame.columns) == ["timestamp_sec", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    assert len(frame) == 50
    assert np.isfinite(frame.to_numpy()).all()


def test_mobiact_rejects_missing_sensor_columns(tmp_path):
    source = tmp_path / "Annotated Data" / "WAL" / "WAL_1_1_annotated.csv"
    _write_trial(source, missing=("gyro_z",))
    with pytest.raises(ValueError, match="missing required columns"):
        load_trial(source)


def test_mobifall_text_release_is_not_silently_accepted(tmp_path):
    raw = tmp_path / "MobiFall_Dataset_v2.0"
    raw.mkdir()
    with pytest.raises(FileNotFoundError, match="MobiFall"):
        find_annotated_root(tmp_path)
