"""Focused regressions for the optional Phase-A scale-source converters."""

from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from data.datasets.extrasensory.convert import (
    GRAVITY_MS2,
    _activity,
    _phone_platforms,
    read_phone,
)
from data.pretraining.nhanes.convert import RATE_HZ, WINDOW_SAMPLES, _complete_blocks
from data.pretraining.nhanes.fetch import _ArchiveLinkParser


REPO = Path(__file__).resolve().parents[1]


def _extra_row(**active):
    row = {
        "label:LYING_DOWN": 0,
        "label:SITTING": 0,
        "label:OR_standing": 0,
        "label:FIX_walking": 0,
        "label:FIX_running": 0,
        "label:BICYCLING": 0,
        "label:STAIRS_-_GOING_UP": 0,
        "label:STAIRS_-_GOING_DOWN": 0,
    }
    row.update(active)
    return row


def test_extrasensory_stair_direction_overrides_generic_walking_only_when_unambiguous():
    assert _activity(_extra_row(**{
        "label:FIX_walking": 1,
        "label:STAIRS_-_GOING_UP": 1,
    })) == "walking_upstairs"
    assert _activity(_extra_row(**{
        "label:FIX_walking": 1,
        "label:STAIRS_-_GOING_DOWN": 1,
    })) == "walking_downstairs"
    assert _activity(_extra_row(**{
        "label:FIX_walking": 1,
        "label:STAIRS_-_GOING_UP": 1,
        "label:STAIRS_-_GOING_DOWN": 1,
    })) is None
    assert _activity(_extra_row(**{"label:SITTING": 1})) == "sitting"
    assert _activity(_extra_row(**{
        "label:SITTING": 1,
        "label:OR_standing": 1,
    })) is None


def test_extrasensory_uses_published_platform_assignment(tmp_path):
    archive_path = tmp_path / "splits.zip"
    android = [f"android-{i:02d}" for i in range(26)]
    iphone = [f"iphone-{i:02d}" for i in range(34)]
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "cv_5_folds/fold_0_train_android_uuids.txt",
            "\n".join(android[:20]),
        )
        archive.writestr(
            "cv_5_folds/fold_0_test_android_uuids.txt",
            "\n".join(android[20:]),
        )
        archive.writestr(
            "cv_5_folds/fold_0_train_iphone_uuids.txt",
            "\n".join(iphone[:27]),
        )
        archive.writestr(
            "cv_5_folds/fold_0_test_iphone_uuids.txt",
            "\n".join(iphone[27:]),
        )
    with ZipFile(archive_path) as archive:
        platforms = _phone_platforms(archive)
    assert len(platforms) == 60
    assert {platforms[subject] for subject in android} == {"android"}
    assert {platforms[subject] for subject in iphone} == {"iphone"}


def test_extrasensory_phone_units_are_platform_driven(tmp_path):
    archive_path = tmp_path / "phone.zip"
    t = 1_700_000_000 + np.arange(800) / 40.0
    signal_g = np.column_stack([
        0.05 * np.sin(np.arange(800) / 20.0),
        np.zeros(800),
        np.ones(800),
    ])
    with ZipFile(archive_path, "w") as archive:
        for platform, values in (
            ("iphone", signal_g),
            ("android", signal_g * GRAVITY_MS2),
        ):
            payload = "\n".join(
                " ".join(f"{value:.9f}" for value in row)
                for row in np.column_stack([t, values])
            )
            archive.writestr(f"{platform}.m_raw_acc.dat", payload)
    with ZipFile(archive_path) as archive:
        iphone = read_phone(archive, "iphone.m_raw_acc.dat", "iphone")
        android = read_phone(archive, "android.m_raw_acc.dat", "android")
    assert iphone.shape == android.shape == (900, 3)
    assert np.allclose(iphone, android, atol=2e-4)
    assert 0.95 < np.median(np.linalg.norm(android, axis=1)) < 1.05


def test_nhanes_gap_and_qc_boundaries_never_form_cross_gap_windows():
    n = WINDOW_SAMPLES * 3
    start = datetime(2020, 1, 1)
    timestamps = np.array([
        start + timedelta(seconds=i / RATE_HZ + (2.0 if i >= 2 * WINDOW_SAMPLES else 0.0))
        for i in range(n)
    ])
    xyz = np.zeros((n, 3), dtype=np.float32)
    xyz[:, 2] = 1.0
    frame = pd.DataFrame({
        "HEADER_TIMESTAMP": timestamps,
        "X": xyz[:, 0],
        "Y": xyz[:, 1],
        "Z": xyz[:, 2],
    })
    qc_start = timestamps[WINDOW_SAMPLES]
    qc_end = timestamps[2 * WINDOW_SAMPLES - 1]
    blocks = _complete_blocks(frame, [(qc_start, qc_end)])
    assert [len(block) for block in blocks] == [WINDOW_SAMPLES, WINDOW_SAMPLES]
    assert all(np.allclose(block[:, 2], 1.0) for block in blocks)


def test_nhanes_backward_timestamp_is_a_hard_boundary():
    n = WINDOW_SAMPLES * 2
    start = datetime(2020, 1, 1)
    timestamps = np.array([start + timedelta(seconds=i / RATE_HZ) for i in range(n)])
    # A device clock reset between complete windows must not create a deceptively continuous run.
    timestamps[WINDOW_SAMPLES:] = [
        start + timedelta(seconds=i / RATE_HZ) for i in range(WINDOW_SAMPLES)
    ]
    frame = pd.DataFrame({
        "HEADER_TIMESTAMP": timestamps,
        "X": np.zeros(n), "Y": np.zeros(n), "Z": np.ones(n),
    })
    blocks = _complete_blocks(frame, [])
    assert [len(block) for block in blocks] == [WINDOW_SAMPLES, WINDOW_SAMPLES]


def test_nhanes_live_index_absolute_links_are_parsed():
    parser = _ArchiveLinkParser()
    parser.feed(
        '<A HREF="/pub/pax_g/62161.tar.bz2">62161.tar.bz2</A>'
        '<a href="62163.tar.bz2">62163.tar.bz2</a>'
        '<a href="/pub/">parent</a>'
    )
    assert parser.subjects == {"62161", "62163"}


def test_integrated_scale_datasets_have_local_primary_sources():
    required = {
        "capture24": ("citation.json", "paper.pdf"),
        "extrasensory": ("citation.json", "paper.pdf"),
        "nhanes": (
            "citation.json",
            "procedures_manual.pdf",
            "pax80_g_documentation.html",
        ),
    }
    for dataset, files in required.items():
        directory = REPO / "references" / "datasets" / dataset
        for filename in files:
            path = directory / filename
            assert path.is_file() and path.stat().st_size > 0, path
