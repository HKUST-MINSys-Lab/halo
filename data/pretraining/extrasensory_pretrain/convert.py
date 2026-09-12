"""Convert raw ExtraSensory captures into label-free JEPA sessions.

Unlike ``data.datasets.extrasensory.convert``, this adapter never reads activity
columns, never filters by activity, and never concatenates separate raw examples.
The label archive supplies phone placement only; the published CV split supplies
phone platform only, which is required for correct physical units.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import butter, sosfiltfilt

from data.pretraining.extrasensory_pretrain.fetch import DEFAULT_RAW_DIR, validate_archives


DS_DIR = Path(__file__).resolve().parent
OUTPUT_RATE_HZ = 50.0
GRAVITY_MS2 = 9.80665
UNLABELED = "__unlabeled__"
OUTPUT_COLUMNS = ("acc_x", "acc_y", "acc_z")
PHONE_PLACEMENT_COLUMNS = {
    "label:PHONE_IN_HAND": "phone_hand",
    "label:PHONE_IN_POCKET": "phone_pocket",
}
OTHER_PHONE_PLACEMENT_COLUMNS = ("label:PHONE_IN_BAG", "label:PHONE_ON_TABLE")
STREAMS = ("watch_wrist", "phone_hand", "phone_pocket")
# About 2.4 MB of numeric data per flush. This is large enough to avoid tiny Parquet
# row groups while keeping three active stream buffers comfortably bounded.
PARQUET_BUFFER_ROWS = 100_000


def _member_index(archive: ZipFile, suffix: str) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    for name in archive.namelist():
        if not name.endswith(suffix):
            continue
        parts = name.split("/")
        if len(parts) < 3:
            continue
        try:
            timestamp = int(parts[-1].split(".", 1)[0])
        except ValueError:
            continue
        key = (parts[-2], timestamp)
        if key in result:
            raise ValueError(f"duplicate raw example key {key}: {result[key]} and {name}")
        result[key] = name
    return result


def _load_text_array(archive: ZipFile, member: str) -> np.ndarray:
    array = np.loadtxt(io.BytesIO(archive.read(member)))
    array = np.atleast_2d(array).astype(np.float64, copy=False)
    if array.shape[1] not in (3, 4):
        raise ValueError(f"{member}: expected 3 or 4 columns, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{member}: non-finite values")
    return array


def _split_clock(t: np.ndarray, xyz_g: np.ndarray, nominal_rate: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split resets/gaps in source order and remove consecutive duplicate stamps."""
    if len(t) != len(xyz_g) or len(t) < 3:
        raise ValueError("too few samples")
    dt = np.diff(t)
    positive = dt[dt > 1e-9]
    if not len(positive):
        raise ValueError("clock has no positive steps")
    median_dt = float(np.median(positive))
    gap_seconds = max(0.5, 5.0 * median_dt, 5.0 / nominal_rate)
    boundaries = np.flatnonzero((dt < -1e-9) | (dt > gap_seconds)) + 1
    result = []
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(t)]):
        part_t = np.asarray(t[start:stop], dtype=np.float64)
        part_xyz = np.asarray(xyz_g[start:stop], dtype=np.float64)
        keep = np.r_[True, np.diff(part_t) > 1e-9]
        part_t, part_xyz = part_t[keep], part_xyz[keep]
        if len(part_t) >= 3:
            result.append((part_t, part_xyz))
    return result


def _regularize_parts(
    t: np.ndarray,
    xyz_g: np.ndarray,
    *,
    nominal_rate: float,
) -> list[np.ndarray]:
    median_g = float(np.median(np.linalg.norm(xyz_g, axis=1)))
    if not 0.2 <= median_g <= 3.0:
        raise ValueError(f"implausible median acceleration {median_g:.3f} g")

    output = []
    for part_t, part_xyz in _split_clock(t, xyz_g, nominal_rate):
        part_t = part_t - part_t[0]
        source_rate = float(1.0 / np.median(np.diff(part_t)))
        filtered = part_xyz
        if source_rate > OUTPUT_RATE_HZ * 1.1 and len(part_xyz) >= 32:
            cutoff_hz = 0.45 * OUTPUT_RATE_HZ
            sos = butter(6, cutoff_hz / (0.5 * source_rate), btype="low", output="sos")
            filtered = sosfiltfilt(sos, part_xyz, axis=0)
        n_out = int(round(float(part_t[-1]) * OUTPUT_RATE_HZ)) + 1
        if n_out < 3:
            continue
        target_t = np.arange(n_out, dtype=np.float64) / OUTPUT_RATE_HZ
        # Rounding can place the final target infinitesimally beyond the last observation.
        target_t = target_t[target_t <= part_t[-1] + 1e-8]
        if len(target_t) < 3:
            continue
        values = np.column_stack(
            [np.interp(target_t, part_t, filtered[:, axis]) for axis in range(3)]
        )
        if np.isfinite(values).all():
            output.append(values.astype(np.float32))
    return output


def read_phone(archive: ZipFile, member: str, platform: str) -> list[np.ndarray]:
    array = _load_text_array(archive, member)
    if array.shape[1] != 4:
        raise ValueError(f"{member}: phone recording has no timestamp column")
    t, xyz = array[:, 0], array[:, 1:4]
    if platform == "android":
        xyz = xyz / GRAVITY_MS2
    elif platform != "iphone":
        raise ValueError(f"{member}: unknown phone platform {platform!r}")
    positive_dt = np.diff(t)
    positive_dt = positive_dt[positive_dt > 1e-9]
    if not len(positive_dt):
        raise ValueError(f"{member}: unusable phone clock")
    nominal_rate = float(1.0 / np.median(positive_dt))
    return _regularize_parts(t, xyz, nominal_rate=nominal_rate)


def read_watch(archive: ZipFile, member: str) -> list[np.ndarray]:
    array = _load_text_array(archive, member)
    if array.shape[1] == 4:
        t = (array[:, 0] - array[0, 0]) / 1000.0
        xyz = array[:, 1:4]
        positive_dt = np.diff(t)
        positive_dt = positive_dt[positive_dt > 1e-9]
        nominal_rate = float(1.0 / np.median(positive_dt)) if len(positive_dt) else 25.0
    else:
        xyz = array[:, :3]
        nominal_rate = 25.0
        t = np.arange(len(xyz), dtype=np.float64) / nominal_rate
    if float(np.median(np.linalg.norm(xyz, axis=1))) > 20.0:
        xyz = xyz / 1000.0
    return _regularize_parts(t, xyz, nominal_rate=nominal_rate)


def _phone_platforms(split_archive: ZipFile) -> dict[str, str]:
    platforms: dict[str, str] = {}
    for platform in ("android", "iphone"):
        for split in ("train", "test"):
            member = f"cv_5_folds/fold_0_{split}_{platform}_uuids.txt"
            for subject in split_archive.read(member).decode("ascii").split():
                previous = platforms.setdefault(subject, platform)
                if previous != platform:
                    raise ValueError(f"{subject}: assigned to both phone platforms")
    if not platforms:
        raise ValueError("published platform split is empty")
    return platforms


def _label_member_index(labels_archive: ZipFile) -> dict[str, str]:
    result = {}
    for member in labels_archive.namelist():
        if not member.endswith(".features_labels.csv.gz"):
            continue
        subject = Path(member).name.split(".", 1)[0]
        if subject in result:
            raise ValueError(f"duplicate label table for participant {subject}")
        result[subject] = member
    return result


def _placements_for_subject(labels_archive: ZipFile, member: str | None) -> dict[int, str]:
    result: dict[int, str] = {}
    if member is None:
        return result
    columns = ["timestamp", *PHONE_PLACEMENT_COLUMNS, *OTHER_PHONE_PLACEMENT_COLUMNS]
    payload = gzip.decompress(labels_archive.read(member))
    frame = pd.read_csv(io.BytesIO(payload), usecols=columns)
    for values in frame.itertuples(index=False, name=None):
        row = dict(zip(frame.columns, values))
        valid = [
            placement
            for column, placement in PHONE_PLACEMENT_COLUMNS.items()
            if row[column] == 1
        ]
        other_active = any(row[column] == 1 for column in OTHER_PHONE_PLACEMENT_COLUMNS)
        if len(valid) == 1 and not other_active:
            result[int(row["timestamp"])] = valid[0]
    return result


class _StreamWriter:
    """Buffered, atomic Parquet writer for one participant and placement stream."""

    def __init__(self, sessions_dir: Path, subject: str, stream: str):
        self.subject = subject
        self.stream = stream
        self.session_id = f"extrasensory_{subject}_{stream}"
        self.destination = sessions_dir / self.session_id
        self.destination.mkdir(parents=True, exist_ok=True)
        self.final_path = self.destination / "data.parquet"
        self.temp_path = self.destination / "data.parquet.part"
        self.temp_path.unlink(missing_ok=True)
        self._writer: pq.ParquetWriter | None = None
        self._values: list[np.ndarray] = []
        self._segment_ids: list[int] = []
        self._buffered_rows = 0
        self.num_segments = 0
        self.num_rows = 0

    def append(self, values: np.ndarray) -> None:
        if values.ndim != 2 or values.shape[1] != len(OUTPUT_COLUMNS) or not len(values):
            raise ValueError(f"invalid ExtraSensory segment shape {values.shape}")
        self._values.append(np.asarray(values, dtype=np.float32))
        self._segment_ids.append(self.num_segments)
        self._buffered_rows += len(values)
        self.num_segments += 1
        self.num_rows += len(values)
        if self._buffered_rows >= PARQUET_BUFFER_ROWS:
            self._flush()

    def _flush(self) -> None:
        if not self._values:
            return
        values = np.concatenate(self._values, axis=0)
        timestamps = np.concatenate(
            [np.arange(len(part), dtype=np.float64) / OUTPUT_RATE_HZ for part in self._values]
        )
        segment_ids = np.concatenate(
            [np.full(len(part), segment_id, dtype=np.int64)
             for part, segment_id in zip(self._values, self._segment_ids)]
        )
        table = pa.table(
            {
                "timestamp_sec": pa.array(timestamps),
                "acc_x": pa.array(values[:, 0]),
                "acc_y": pa.array(values[:, 1]),
                "acc_z": pa.array(values[:, 2]),
                "subject": pa.array([self.subject] * len(values)),
                "segment_id": pa.array(segment_ids),
            }
        )
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.temp_path,
                table.schema,
                compression="zstd",
                use_dictionary=("subject", "segment_id"),
            )
        self._writer.write_table(table, row_group_size=PARQUET_BUFFER_ROWS)
        self._values.clear()
        self._segment_ids.clear()
        self._buffered_rows = 0

    def close(self) -> bool:
        self._flush()
        if self._writer is None:
            self.abort()
            return False
        self._writer.close()
        self._writer = None
        self.temp_path.replace(self.final_path)
        return True

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.temp_path.unlink(missing_ok=True)


def _write_progress(
    output_dir: Path,
    labels: dict[str, list[str]],
    completed_subjects: set[str],
    *,
    subject_stats: dict[str, dict],
    raw_dir: Path,
) -> None:
    session_counts: Counter[str] = Counter()
    segment_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    for stats in subject_stats.values():
        session_counts.update(stats.get("sessions_by_stream", {}))
        segment_counts.update(stats.get("segments_by_stream", {}))
        row_counts.update(stats.get("rows_by_stream", {}))
        skips.update(stats.get("skips", {}))
    labels_tmp = output_dir / "labels.json.part"
    labels_tmp.write_text(json.dumps(labels, indent=2) + "\n")
    labels_tmp.replace(output_dir / "labels.json")
    manifest = {
        "dataset_name": "ExtraSensory (label-free pretraining view)",
        "dataset": "extrasensory_pretrain",
        "converter_version": 2,
        "source": "http://extrasensory.ucsd.edu/",
        "num_subjects": sum(bool(stats.get("sessions_by_stream")) for stats in subject_stats.values()),
        "num_subjects_processed": len(subject_stats),
        "completed_subjects": sorted(completed_subjects),
        "num_sessions": len(labels),
        "num_segments": sum(segment_counts.values()),
        "sampling_rate_hz": OUTPUT_RATE_HZ,
        "source_sampling_rate_hz": "phone device-dependent; Pebble watch approximately 25 Hz",
        "channels": list(OUTPUT_COLUMNS),
        "unit": "g",
        "gravity_state": "present",
        "phase_a_only": True,
        "activity_labels_used": False,
        "placement_metadata_used": True,
        "sessions_by_stream": dict(sorted(session_counts.items())),
        "segments_by_stream": dict(sorted(segment_counts.items())),
        "rows_by_stream": dict(sorted(row_counts.items())),
        "skips": dict(sorted(skips.items())),
        "subject_stats": subject_stats,
        "raw_archive_directory": str(raw_dir.resolve()),
        "note": (
            "One Parquet per participant and stream; segment_id preserves every raw capture "
            "or clock part as an independent windowing boundary. Every label is __unlabeled__."
        ),
    }
    manifest_tmp = output_dir / "manifest.json.part"
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_tmp.replace(output_dir / "manifest.json")


def convert(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DS_DIR,
    limit_subjects: int | None = None,
    limit_examples_per_subject: int | None = None,
    fresh: bool = False,
) -> bool:
    paths = validate_archives(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = output_dir / "sessions"
    labels_path = output_dir / "labels.json"
    manifest_path = output_dir / "manifest.json"
    if fresh and sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    if fresh:
        labels_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = (
        json.loads(labels_path.read_text()) if labels_path.is_file() and not fresh else {}
    )
    old_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() and not fresh else {}
    )
    if old_manifest and old_manifest.get("converter_version") != 2:
        raise ValueError("existing output uses another converter version; rerun with --fresh")
    completed_subjects = set(old_manifest.get("completed_subjects", []))
    subject_stats: dict[str, dict] = dict(old_manifest.get("subject_stats", {}))

    with (
        ZipFile(paths["raw_acc.zip"]) as phone_archive,
        ZipFile(paths["watch_acc.zip"]) as watch_archive,
        ZipFile(paths["labels.zip"]) as labels_archive,
        ZipFile(paths["cv5Folds.zip"]) as split_archive,
    ):
        phone_members = _member_index(phone_archive, ".m_raw_acc.dat")
        watch_members = _member_index(watch_archive, ".m_watch_acc.dat")
        label_members = _label_member_index(labels_archive)
        platforms = _phone_platforms(split_archive)
        keys_by_subject: dict[str, set[tuple[str, int]]] = defaultdict(set)
        for key in phone_members:
            keys_by_subject[key[0]].add(key)
        for key in watch_members:
            keys_by_subject[key[0]].add(key)
        all_subjects = sorted(keys_by_subject)
        if limit_subjects is not None:
            all_subjects = all_subjects[:limit_subjects]

        for subject in all_subjects:
            if subject in completed_subjects:
                continue
            for stream in STREAMS:
                session_id = f"extrasensory_{subject}_{stream}"
                labels.pop(session_id, None)
                shutil.rmtree(sessions_dir / session_id, ignore_errors=True)
            subject_stats.pop(subject, None)

            accepted = 0
            truncated = False
            local_skips: Counter[str] = Counter()
            writers: dict[str, _StreamWriter] = {}

            def writer(stream: str) -> _StreamWriter:
                if stream not in writers:
                    writers[stream] = _StreamWriter(sessions_dir, subject, stream)
                return writers[stream]

            placements = _placements_for_subject(labels_archive, label_members.get(subject))
            keys = sorted(keys_by_subject[subject], key=lambda key: key[1])
            try:
                for key in keys:
                    if limit_examples_per_subject is not None and accepted >= limit_examples_per_subject:
                        truncated = True
                        break
                    timestamp = key[1]
                    watch_member = watch_members.get(key)
                    if watch_member is not None:
                        try:
                            for values in read_watch(watch_archive, watch_member):
                                writer("watch_wrist").append(values)
                            accepted += 1
                        except (ValueError, OSError):
                            local_skips["bad_watch"] += 1

                    phone_member = phone_members.get(key)
                    if phone_member is None:
                        continue
                    placement = placements.get(timestamp)
                    platform = platforms.get(subject)
                    if placement is None:
                        local_skips["phone_without_unique_placement"] += 1
                        continue
                    if platform is None:
                        raise ValueError(
                            f"{subject}: phone data has no Android/iPhone assignment in the "
                            "published split"
                        )
                    try:
                        for values in read_phone(phone_archive, phone_member, platform):
                            writer(placement).append(values)
                        accepted += 1
                    except (ValueError, OSError):
                        local_skips["bad_phone"] += 1

                sessions_by_stream = {}
                segments_by_stream = {}
                rows_by_stream = {}
                for stream, stream_writer in writers.items():
                    if stream_writer.close():
                        labels[stream_writer.session_id] = [UNLABELED]
                        sessions_by_stream[stream] = 1
                        segments_by_stream[stream] = stream_writer.num_segments
                        rows_by_stream[stream] = stream_writer.num_rows
            except BaseException:
                for stream_writer in writers.values():
                    stream_writer.abort()
                raise

            if not truncated:
                completed_subjects.add(subject)
            subject_stats[subject] = {
                "complete": not truncated,
                "sessions_by_stream": sessions_by_stream,
                "segments_by_stream": segments_by_stream,
                "rows_by_stream": rows_by_stream,
                "skips": dict(sorted(local_skips.items())),
            }
            _write_progress(
                output_dir,
                labels,
                completed_subjects,
                subject_stats=subject_stats,
                raw_dir=raw_dir,
            )

    if not labels:
        return False
    _write_progress(
        output_dir,
        labels,
        completed_subjects,
        subject_stats=subject_stats,
        raw_dir=raw_dir,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DS_DIR)
    parser.add_argument("--limit-subjects", type=int)
    parser.add_argument("--limit-examples-per-subject", type=int)
    parser.add_argument("--fresh", action="store_true", help="discard prior converted sessions")
    args = parser.parse_args()
    if not convert(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        limit_subjects=args.limit_subjects,
        limit_examples_per_subject=args.limit_examples_per_subject,
        fresh=args.fresh,
    ):
        raise SystemExit("no ExtraSensory pretraining sessions were produced")


if __name__ == "__main__":
    main()
