"""Contracts for the annotation-free Capture-24 JEPA adapter."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.pretraining.capture24_pretrain import convert, fetch


def _write_raw(path: Path, *, gap: bool = False) -> None:
    rate = 100.0
    count = 1_200
    seconds = np.arange(count, dtype=np.float64) / rate
    if gap:
        seconds[600:] += 1.0
    timestamps = pd.Timestamp("2020-01-01") + pd.to_timedelta(seconds, unit="s")
    frame = pd.DataFrame(
        {
            "time": timestamps.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "x": 0.1 * np.sin(seconds),
            "y": np.zeros(count),
            "z": np.ones(count),
            # Deliberately varied labels: conversion must not see this column at all.
            "annotation": np.where(np.arange(count) < 600, "walking", "sitting"),
        }
    )
    with gzip.open(path, "wt") as handle:
        frame.to_csv(handle, index=False)


def test_converter_ignores_annotations_and_splits_only_real_clock_gaps(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    source.mkdir()
    raw = source / "P001.csv.gz"
    _write_raw(raw, gap=True)
    monkeypatch.setattr(convert, "validate_source", lambda _root: (raw,))

    manifest = convert.convert(source_root=source, output_root=tmp_path / "out", chunk_rows=257)
    output = tmp_path / "out"
    labels = json.loads((output / "labels.json").read_text())
    assert labels == {"capture24_P001_watch_wrist": ["__unlabeled__"]}
    assert manifest["activity_labels_used"] is False
    assert manifest["num_subjects"] == 1
    frame = pd.read_parquet(output / "sessions" / "capture24_P001_watch_wrist" / "data.parquet")
    assert list(frame.columns) == list(convert.OUTPUT_COLUMNS)
    assert set(frame["segment_id"]) == {0, 1}
    assert frame.groupby("segment_id")["timestamp_sec"].first().tolist() == [0.0, 0.0]
    assert np.median(np.linalg.norm(frame[["acc_x", "acc_y", "acc_z"]], axis=1)) == pytest.approx(1.0, abs=0.01)


def test_converter_does_not_drop_a_chunk_that_continues_a_clock_segment(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    source.mkdir()
    raw = source / "P001.csv.gz"
    _write_raw(raw, gap=False)
    monkeypatch.setattr(convert, "validate_source", lambda _root: (raw,))

    convert.convert(source_root=source, output_root=tmp_path / "out", chunk_rows=211)
    frame = pd.read_parquet(tmp_path / "out" / "sessions" / "capture24_P001_watch_wrist" / "data.parquet")
    assert len(frame) == 1_200
    assert frame["segment_id"].nunique() == 1
    assert np.diff(frame["timestamp_sec"]).min() > 0.0


def test_fetch_validation_requires_raw_waveforms_but_no_annotation_dictionary(tmp_path):
    for subject in range(1, fetch.EXPECTED_PARTICIPANTS + 1):
        (tmp_path / f"P{subject:03d}.csv.gz").touch()
    assert len(fetch.validate_source(tmp_path)) == fetch.EXPECTED_PARTICIPANTS


def test_metadata_and_policy_keep_capture24_pretraining_label_free_and_routable():
    metadata = json.loads((Path(convert.__file__).parent / "metadata.json").read_text())
    assert metadata["activities"] == []
    assert metadata["phase_a_only"] is True
    assert metadata["full_windows_only"] is True
    from data.scripts.curate.deployment_policy import session_stream_specs

    matches = session_stream_specs(
        "capture24_pretrain", "capture24_P001_watch_wrist", role="phase_a_scale"
    )
    assert [spec.stream_id for spec in matches] == ["watch_wrist"]
