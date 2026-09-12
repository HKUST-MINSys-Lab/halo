"""Focused synthetic-fixture tests for the label-free ExtraSensory adapter."""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from data.pretraining.extrasensory_pretrain import convert
from data.pretraining.extrasensory_pretrain import fetch
from data.pretraining.extrasensory_pretrain.fetch import REQUIRED_ARCHIVES, validate_archives


def _signal(rate: float, seconds: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    t = 1_700_000_000.0 + np.arange(int(rate * seconds), dtype=np.float64) / rate
    phase = np.arange(len(t), dtype=np.float64) / rate
    xyz_g = np.column_stack((0.05 * np.sin(2 * np.pi * phase), np.zeros(len(t)), np.ones(len(t))))
    return t, xyz_g


def _payload(values: np.ndarray) -> str:
    return "\n".join(" ".join(f"{value:.9f}" for value in row) for row in values)


def _fixture_archives(root: Path) -> None:
    subject = "subject-android"
    t, xyz_g = _signal(40.0)
    with ZipFile(root / "raw_acc.zip", "w") as archive:
        for timestamp in (100, 200, 400, 500):
            archive.writestr(
                f"raw_acc/{subject}/{timestamp}.m_raw_acc.dat",
                _payload(np.column_stack((t, xyz_g * convert.GRAVITY_MS2))),
            )
    watch_t = np.arange(500, dtype=np.float64) * 40.0
    watch_mg = np.column_stack((np.zeros(500), np.zeros(500), np.full(500, 1000.0)))
    with ZipFile(root / "watch_acc.zip", "w") as archive:
        archive.writestr(
            f"watch_acc/{subject}/100.m_watch_acc.dat",
            _payload(np.column_stack((watch_t, watch_mg))),
        )
        # Watch data without a corresponding label row must still survive.
        archive.writestr(
            f"watch_acc/{subject}/300.m_watch_acc.dat",
            _payload(np.column_stack((watch_t, watch_mg))),
        )
    labels = pd.DataFrame(
        {
            "timestamp": [100, 200, 400, 500],
            "label:PHONE_IN_HAND": [1, 1, 0, 0],
            "label:PHONE_IN_POCKET": [0, 1, 1, 0],
            "label:PHONE_IN_BAG": [0, 0, 0, 1],
            "label:PHONE_ON_TABLE": [0, 0, 0, 0],
            # Activity-like data is intentionally present to prove the converter does not need it.
            "label:FIX_walking": [1, 1, 0, 1],
        }
    )
    with ZipFile(root / "labels.zip", "w") as archive:
        archive.writestr(
            f"labels/{subject}.features_labels.csv.gz",
            gzip.compress(labels.to_csv(index=False).encode()),
        )
    with ZipFile(root / "cv5Folds.zip", "w") as archive:
        archive.writestr("cv_5_folds/fold_0_train_android_uuids.txt", subject)
        archive.writestr("cv_5_folds/fold_0_test_android_uuids.txt", "")
        archive.writestr("cv_5_folds/fold_0_train_iphone_uuids.txt", "")
        archive.writestr("cv_5_folds/fold_0_test_iphone_uuids.txt", "")


def test_validate_archives_is_non_copying(tmp_path):
    _fixture_archives(tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())
    paths = validate_archives(tmp_path)
    assert tuple(paths) == REQUIRED_ARCHIVES
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_download_resumes_part_file_atomically(tmp_path, monkeypatch):
    destination = tmp_path / "archive.zip"
    destination.with_suffix(".zip.part").write_bytes(b"abc")
    requests = []

    class Response(io.BytesIO):
        status = 206

    def fake_urlopen(request):
        requests.append(request)
        return Response(b"def")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    fetch._download_resumable("https://example.invalid/archive.zip", destination, 6)
    assert requests[0].headers["Range"] == "bytes=3-"
    assert destination.read_bytes() == b"abcdef"
    assert not destination.with_suffix(".zip.part").exists()


def test_phone_platform_units_produce_the_same_physical_signal(tmp_path):
    archive_path = tmp_path / "phone.zip"
    t, xyz_g = _signal(40.0)
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("iphone.dat", _payload(np.column_stack((t, xyz_g))))
        archive.writestr(
            "android.dat",
            _payload(np.column_stack((t, xyz_g * convert.GRAVITY_MS2))),
        )
    with ZipFile(archive_path) as archive:
        iphone = convert.read_phone(archive, "iphone.dat", "iphone")
        android = convert.read_phone(archive, "android.dat", "android")
    assert len(iphone) == len(android) == 1
    assert np.allclose(iphone[0], android[0], atol=2e-4)


def test_convert_preserves_examples_and_uses_no_activity_targets(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    _fixture_archives(raw)

    assert convert.convert(raw_dir=raw, output_dir=out)
    labels = json.loads((out / "labels.json").read_text())
    # Four source segments are packed into three participant/stream Parquets. Timestamp
    # 300 proves watch data does not depend on an activity or placement row.
    assert len(labels) == 3
    assert all(value == ["__unlabeled__"] for value in labels.values())
    assert set(labels) == {
        "extrasensory_subject-android_phone_hand",
        "extrasensory_subject-android_phone_pocket",
        "extrasensory_subject-android_watch_wrist",
    }

    segments_by_stream = {}
    for session_id in labels:
        frame = pd.read_parquet(out / "sessions" / session_id / "data.parquet")
        assert list(frame.columns) == [
            "timestamp_sec", "acc_x", "acc_y", "acc_z", "subject", "segment_id",
        ]
        assert len(frame) > 900
        segment_ids = frame["segment_id"].drop_duplicates().tolist()
        assert segment_ids == list(range(len(segment_ids)))
        segments_by_stream[session_id.rsplit("_", 2)[-2] + "_" + session_id.rsplit("_", 1)[-1]] = len(segment_ids)
        for _, segment in frame.groupby("segment_id", sort=False):
            assert np.diff(segment["timestamp_sec"]).min() == pytest.approx(0.02)
            assert segment["timestamp_sec"].iloc[0] == 0.0
        magnitude = np.linalg.norm(frame[list(convert.OUTPUT_COLUMNS)].to_numpy(), axis=1)
        assert np.median(magnitude) == pytest.approx(1.0, abs=0.02)
    assert segments_by_stream == {"phone_hand": 1, "phone_pocket": 1, "watch_wrist": 2}

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["dataset"] == "extrasensory_pretrain"
    assert manifest["activity_labels_used"] is False
    assert manifest["placement_metadata_used"] is True
    assert manifest["sampling_rate_hz"] == 50.0
    assert manifest["unit"] == "g"
    assert manifest["gravity_state"] == "present"
    assert manifest["phase_a_only"] is True
    assert manifest["converter_version"] == 2
    assert manifest["num_sessions"] == 3
    assert manifest["num_segments"] == 4
    assert manifest["segments_by_stream"] == {
        "phone_hand": 1, "phone_pocket": 1, "watch_wrist": 2,
    }
    assert manifest["skips"] == {"phone_without_unique_placement": 2}
    assert manifest["completed_subjects"] == ["subject-android"]


def test_conversion_resumes_completed_subject_without_rewriting(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    _fixture_archives(raw)
    assert convert.convert(raw_dir=raw, output_dir=out)
    parquet = next((out / "sessions").glob("*/data.parquet"))
    stamp = parquet.stat().st_mtime_ns
    assert convert.convert(raw_dir=raw, output_dir=out)
    assert parquet.stat().st_mtime_ns == stamp
    assert len(json.loads((out / "labels.json").read_text())) == 3


def test_clock_gap_splits_instead_of_interpolating(tmp_path):
    archive_path = tmp_path / "phone.zip"
    t, xyz = _signal(50.0, seconds=4.0)
    t[100:] += 2.0
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("phone.dat", _payload(np.column_stack((t, xyz))))
    with ZipFile(archive_path) as archive:
        parts = convert.read_phone(archive, "phone.dat", "iphone")
    assert len(parts) == 2
    assert [len(part) for part in parts] == [100, 100]


def test_tracked_metadata_is_label_free_and_collision_safe():
    metadata = json.loads((Path(convert.__file__).parent / "metadata.json").read_text())
    assert metadata["dataset"] == "extrasensory_pretrain"
    assert metadata["activities"] == []
    assert metadata["phase_a_only"] is True
    assert metadata["pre_windowed"] is False
    assert metadata["streaming_grid"] is True
    assert metadata["full_windows_only"] is True


def test_packed_session_suffixes_route_to_exactly_one_stream():
    from data.scripts.curate.deployment_policy import session_stream_specs

    subject = "subject-android"
    for stream in convert.STREAMS:
        session_id = f"extrasensory_{subject}_{stream}"
        matches = session_stream_specs(
            "extrasensory_pretrain", session_id, role="phase_a_scale"
        )
        assert [spec.stream_id for spec in matches] == [stream]
