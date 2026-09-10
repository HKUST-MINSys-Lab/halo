import json

import numpy as np
import pandas as pd
import pytest


def test_wisdm_mixed_rate_is_anti_aliased_to_real_20_hz():
    from data.datasets.wisdm.convert import _regularize

    source_hz = 50.0
    t = np.arange(500) / source_hz
    # 2 Hz must survive; 15 Hz is above the target Nyquist and must not alias strongly to 5 Hz.
    values = np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 15 * t)
    frame = pd.DataFrame({"timestamp": t * 1e9,
                          "x": values, "y": values, "z": values})
    out_t, out = _regularize(frame, ["x", "y", "z"])
    assert np.allclose(np.diff(out_t), 1 / 20.0)
    freq = np.fft.rfftfreq(len(out), 1 / 20.0)
    power = np.abs(np.fft.rfft(out[:, 0]))
    p2 = power[np.argmin(np.abs(freq - 2.0))]
    p5 = power[np.argmin(np.abs(freq - 5.0))]
    assert p5 < 0.1 * p2


def test_hhar_resampling_filters_before_decimation():
    from data.datasets.hhar.convert import resample_stream

    source_hz = 200.0
    t = np.arange(1200) / source_hz
    signal = np.sin(2 * np.pi * 40 * t)
    frame = pd.DataFrame({"timestamp_ns": t * 1e9, **{
        column: signal for column in
        ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    }})
    out = resample_stream(frame)[0]["acc_x"].to_numpy()
    assert np.sqrt(np.mean(out ** 2)) < 0.05


def test_hhar_retains_accelerometer_only_device(tmp_path):
    from data.datasets.hhar.convert import load_and_merge_sensor_data

    columns = ["Index", "Arrival_Time", "Creation_Time", "x", "y", "z",
               "User", "Model", "Device", "gt"]
    accel = pd.DataFrame([
        [0, 0, 0, 1, 2, 3, "a", "m", "galaxy", "walk"],
        [1, 1, 20_000_000, 2, 3, 4, "a", "m", "galaxy", "walk"],
    ], columns=columns)
    # A valid but unrelated gyro key ensures the input file itself is non-empty.
    gyro = pd.DataFrame([
        [0, 0, 0, 1, 2, 3, "b", "m", "nexus", "walk"],
        [1, 1, 20_000_000, 2, 3, 4, "b", "m", "nexus", "walk"],
    ], columns=columns)
    acc_path, gyro_path = tmp_path / "acc.csv", tmp_path / "gyro.csv"
    accel.to_csv(acc_path, index=False)
    gyro.to_csv(gyro_path, index=False)
    merged = load_and_merge_sensor_data(acc_path, gyro_path)
    assert len(merged) == 2
    assert merged[["gyro_x", "gyro_y", "gyro_z"]].isna().all().all()


def test_unimib_trial_id_groups_overlapping_windows(monkeypatch, tmp_path):
    from data.datasets.unimib_shar import convert

    monkeypatch.setattr(convert, "OUTPUT_DIR", tmp_path)
    data = np.zeros((2, 3 * convert.WINDOW_LEN), dtype=np.float64)
    labels = np.asarray([[3, 1, 4], [3, 1, 4]], dtype=np.uint8)
    convert.process(data, labels)
    recordings = json.loads((tmp_path / "recordings.json").read_text())
    assert len(set(recordings.values())) == 1
    assert next(iter(recordings.values())).endswith("trial04")


def test_harmes_comment_markers_are_preserved_as_open_labels(tmp_path):
    from data.datasets.harmes.convert import event_segments

    events = tmp_path / "events.csv"
    pd.DataFrame({
        "Description": ["RECORD", "Doing a puzzle start", "Doing a puzzle end"],
        "Type": ["start", "comment", "comment"],
        "Time": [1000.0, 1001.0, 1010.0],
    }).to_csv(events, index=False)
    assert list(event_segments(str(events), 1000.0)) == [("doing_a_puzzle", 1001.0, 1010.0)]


@pytest.mark.parametrize("module_name", ["baselines.limubert.prep", "baselines.crosshar.prep"])
def test_layout_locked_baseline_resampling_ignores_padded_tail(module_name):
    import importlib

    prep = importlib.import_module(module_name)
    clean = np.zeros((1, 20, 6), dtype=np.float32)
    dirty = clean.copy()
    dirty[:, 10:] = 10_000.0
    lengths = np.array([10])
    left = prep.resample_crop_pad(clean, rate_hz=10.0, lengths=lengths)
    right = prep.resample_crop_pad(dirty, rate_hz=10.0, lengths=lengths)
    assert np.array_equal(left, right)


def test_memory_vocab_refuses_silent_historical_relabeling():
    from training.evidence.build_memory import _load_vocab

    with pytest.raises(ValueError, match="semantic vocabulary differs"):
        _load_vocab({"label_ids": {"definitely_not_the_current_vocabulary": 0}})
    assert _load_vocab({"label_ids": {"__unlabeled__": 0}})
