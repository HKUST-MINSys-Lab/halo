from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from data.datasets.realworld.convert import _inner_part_id, load_sensor_parts, resample_part


def _csv_bytes(offset: float) -> bytes:
    frame = pd.DataFrame({
        "id": range(12),
        "attr_time": [offset + 20 * row for row in range(12)],
        "attr_x": range(12),
        "attr_y": range(12),
        "attr_z": range(12),
    })
    return frame.to_csv(index=False).encode()


def _inner_zip(member: str, offset: float) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(member, _csv_bytes(offset))
    return output.getvalue()


def test_realworld_unnumbered_inner_archive_is_physical_part_one(tmp_path):
    archive_path = tmp_path / "gyr_walking_csv.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "gyr_walking_csv.zip",
            _inner_zip("Gyroscope_walking_waist.csv", 1_000),
        )
        archive.writestr(
            "gyr_walking_2_csv.zip",
            _inner_zip("Gyroscope_walking_waist.csv", 2_000),
        )

    parts = load_sensor_parts(archive_path, "gyro")

    assert sorted(parts) == [1, 2]
    assert parts[1]["timestamp_sec"].iloc[0] == 1.0
    assert parts[2]["timestamp_sec"].iloc[0] == 2.0


def test_realworld_accelerometer_and_gyroscope_part_ids_align():
    assert _inner_part_id("acc_walking_1_csv.zip") == 1
    assert _inner_part_id("gyr_walking_csv.zip") == 1
    assert _inner_part_id("acc_walking_3_csv.zip") == 3
    assert _inner_part_id("gyr_walking_3_csv.zip") == 3


def test_resample_uses_measured_accelerometer_gyroscope_overlap():
    time_acc = np.arange(0.0, 2.0, 0.02)
    time_gyro = np.arange(0.02, 1.98, 0.02)
    acc = pd.DataFrame({"timestamp_sec": time_acc, **{
        f"acc_{axis}": np.sin(time_acc) for axis in "xyz"
    }})
    gyro = pd.DataFrame({"timestamp_sec": time_gyro, **{
        f"gyro_{axis}": np.cos(time_gyro) for axis in "xyz"
    }})

    output = resample_part(acc, gyro, rate=50.0)

    assert output is not None
    assert all(f"gyro_{axis}" in output for axis in "xyz")
    assert output["timestamp_sec"].iloc[0] == 0.0
    assert len(output) == len(time_gyro)
