"""Convert Ego-Exo4D take VRS files into label-free, Phase-A-only HALO sessions.

What we take
------------
The head-mounted Project Aria IMU, and nothing else. Ego-Exo4D ships 1,286 camera-hours,
from 740+ participants across 13 sites; the glasses carry two IMUs (``imu-right``
= stream 1202-1, ~800 Hz; ``imu-left`` = stream 1202-2, ~1 kHz) plus a
magnetometer and a barometer. We keep ONE IMU per take -- ``imu-right`` by
preference, ``imu-left`` as a fallback -- and record which one in the manifest.
Video, audio, gaze, trajectory, magnetometer and barometer are all discarded.

Signal decisions (each one is a decision, not an accident)
---------------------------------------------------------
Units. Aria reports acceleration in m/s^2 and angular rate in rad/s, both in the
device frame, with gravity PRESENT. We divide acceleration by 9.80665 to reach
the corpus-wide unit of g and leave gravity in; gyro passes through untouched.
A stationarity probe reports the gravity magnitude of the quietest two seconds
of each session so a mis-scaled take is visible rather than silent.

Rate. Aria IMU timestamps are non-uniform device timestamps in nanoseconds. We
anti-alias low-pass (zero-phase FIR at 80 Hz, i.e. 0.4x the target rate) and then
place samples on a uniform 200 Hz grid. 200 Hz is chosen because the frontend's
f_max is about 14 Hz: storing 800 Hz or 1 kHz would cost 4-5x for information the
model never reads, while 200 Hz still leaves an order of magnitude of headroom.

Gaps. Aria drops IMU packets. Interpolating across a dropout invents motion, so
any gap longer than 0.5 s SPLITS the session instead. Parts are suffixed
(``..._aria_head_part02``) and ``aria_head`` stays a matchable substring so the
deployment-policy stream token still resolves. Parts shorter than one 6 s window
are dropped.

Subjects. HALO splits are subject-disjoint, so a take must resolve to a stable
person. takes.json carries ``participant_uid`` (a release-wide integer id, joined
to participants.json), which is exactly what we want; the subject id is then
``p<participant_uid>``. When it is missing we fall back to the take's
``capture_uid`` -- multiple takes share a capture, so a capture is a single
recording session with a single wearer, which keeps splits disjoint at the cost
of splitting one person across several pseudo-subjects. The fallback is counted
and reported in the manifest under ``subject_id_source``; a per-take id is never
invented, because that would silently break subject-disjoint splitting.

Output contract
---------------
``sessions/<session_id>/data.parquet`` with ``timestamp_sec`` (float64, from 0,
uniform 1/200 s), ``acc_{x,y,z}`` (float32, g, gravity present, device frame),
``gyro_{x,y,z}`` (float32, rad/s, device frame) and ``subject`` (string), plus
``labels.json`` (every session marked ``__unlabeled__``), ``manifest.json`` and
``metadata.json``.

Usage::

    python -m data.pretraining.ego_exo4d.convert --takes 40 --max-hours-per-take 1
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin

DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"

DATASET = "ego_exo4d"
SESSION_PREFIX = "egoexo"
STREAM_TOKEN = "aria_head"
UNLABELED = "__unlabeled__"
SOURCE_URL = "https://docs.ego-exo4d-data.org/"

RATE_HZ = 200.0
WINDOW_SECONDS = 6.0
MIN_SESSION_SECONDS = WINDOW_SECONDS
MAX_GAP_SECONDS = 0.5
GRAVITY_MS2 = 9.80665

# Anti-alias FIR: pass to 0.4*fs_out, stop by 0.5*fs_out (the output Nyquist).
ANTIALIAS_PASS_FRACTION = 0.4
ANTIALIAS_TRANSITION_FRACTION = 0.1
_HAMMING_TAPS_CONSTANT = 3.3  # taps ~= 3.3 * fs / transition_width for a Hamming window

ACC_COLUMNS = ("acc_x", "acc_y", "acc_z")
GYRO_COLUMNS = ("gyro_x", "gyro_y", "gyro_z")
SIGNAL_COLUMNS = ACC_COLUMNS + GYRO_COLUMNS

# imu-right (1202-1, ~800 Hz) first; imu-left (1202-2, ~1 kHz) as fallback.
IMU_PREFERENCE = (("imu-right", "1202-1"), ("imu-left", "1202-2"))

PROJECTARIA_HELP = (
    "projectaria_tools is required to read Ego-Exo4D VRS files.\n"
    "  pip install projectaria-tools\n"
    "  (import path is `from projectaria_tools.core import data_provider`)"
)


class MissingDependency(RuntimeError):
    """Raised when projectaria_tools is not importable."""


# --------------------------------------------------------------------------
# VRS access
# --------------------------------------------------------------------------


def _data_provider_module():
    try:
        from projectaria_tools.core import data_provider  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - exercised via monkeypatch
        raise MissingDependency(PROJECTARIA_HELP) from error
    return data_provider


def _stream_id(raw: str):
    try:
        from projectaria_tools.core.stream_id import StreamId  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return StreamId(raw)
    except Exception:  # noqa: BLE001 - a malformed id is simply "not available"
        return None


def open_provider(vrs_path: Path):
    provider = _data_provider_module().create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"projectaria_tools could not open {vrs_path}")
    return provider


def resolve_imu_stream(provider) -> tuple[str, object, int]:
    """Pick one head IMU, preferring imu-right (1202-1) over imu-left (1202-2)."""
    for label, raw_id in IMU_PREFERENCE:
        stream_id = None
        try:
            stream_id = provider.get_stream_id_from_label(label)
        except Exception:  # noqa: BLE001
            stream_id = None
        if stream_id is None:
            stream_id = _stream_id(raw_id)
        if stream_id is None:
            continue
        try:
            count = int(provider.get_num_data(stream_id))
        except Exception:  # noqa: BLE001
            continue
        if count > 0:
            return label, stream_id, count
    raise RuntimeError(
        "no usable IMU stream in this VRS: tried "
        + ", ".join(f"{label} ({raw})" for label, raw in IMU_PREFERENCE)
    )


def read_imu(provider, stream_id, count: int, max_seconds: float | None = None):
    """Read (time_sec, accel m/s^2, gyro rad/s) from one IMU stream, time-ordered."""
    stamps = np.empty(count, dtype=np.float64)
    accel = np.empty((count, 3), dtype=np.float64)
    gyro = np.empty((count, 3), dtype=np.float64)
    for index in range(count):
        record = provider.get_imu_data_by_index(stream_id, index)
        stamps[index] = float(record.capture_timestamp_ns)
        accel[index] = np.asarray(record.accel_msec2, dtype=np.float64)
        gyro[index] = np.asarray(record.gyro_radsec, dtype=np.float64)

    order = np.argsort(stamps, kind="stable")
    stamps, accel, gyro = stamps[order], accel[order], gyro[order]
    # Duplicate device timestamps break interpolation; keep the first of each.
    keep = np.r_[True, np.diff(stamps) > 0]
    stamps, accel, gyro = stamps[keep], accel[keep], gyro[keep]

    time_sec = (stamps - stamps[0]) / 1e9
    finite = np.isfinite(accel).all(axis=1) & np.isfinite(gyro).all(axis=1)
    time_sec, accel, gyro = time_sec[finite], accel[finite], gyro[finite]
    if max_seconds is not None:
        within = time_sec <= max_seconds
        time_sec, accel, gyro = time_sec[within], accel[within], gyro[within]
    return time_sec, accel, gyro


# --------------------------------------------------------------------------
# Signal handling
# --------------------------------------------------------------------------


def split_on_gaps(time_sec: np.ndarray, max_gap: float = MAX_GAP_SECONDS) -> list[slice]:
    """Contiguous index runs, cut wherever the sample interval exceeds `max_gap`.

    Splitting rather than interpolating matters: a 1 s dropout linearly bridged
    would read as a slow, smooth head movement that never happened.
    """
    if len(time_sec) == 0:
        return []
    breaks = np.flatnonzero(np.diff(time_sec) > max_gap) + 1
    bounds = np.r_[0, breaks, len(time_sec)]
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def native_rate(time_sec: np.ndarray) -> float:
    if len(time_sec) < 2:
        return float("nan")
    step = float(np.median(np.diff(time_sec)))
    return 1.0 / step if step > 0 else float("nan")


def antialias(values: np.ndarray, fs_in: float, fs_out: float = RATE_HZ) -> np.ndarray:
    """Zero-phase FIR low-pass ahead of decimation to `fs_out`.

    Passband to 0.4*fs_out (80 Hz at 200 Hz out) so the frontend's ~14 Hz band is
    untouched; stopband by 0.5*fs_out so nothing can fold back over the new
    Nyquist. `filtfilt` keeps the passband gain at 1 and the phase at 0.
    """
    if not np.isfinite(fs_in) or fs_in <= fs_out:
        return values
    cutoff = ANTIALIAS_PASS_FRACTION * fs_out
    transition = ANTIALIAS_TRANSITION_FRACTION * fs_out
    taps = int(np.ceil(_HAMMING_TAPS_CONSTANT * fs_in / transition)) | 1
    # filtfilt needs more than 3*(taps-1) samples; shrink the filter before giving up.
    max_taps = (len(values) // 3) | 1
    taps = min(taps, max_taps)
    if taps < 9:
        return values
    kernel = firwin(taps, cutoff, fs=fs_in, window="hamming")
    return filtfilt(kernel, [1.0], values, axis=0)


def resample_uniform(
    time_sec: np.ndarray,
    values: np.ndarray,
    rate_hz: float = RATE_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Anti-alias, then place `values` on a uniform `rate_hz` grid starting at 0.

    Interpolating against the *measured* timestamps (rather than assuming the
    nominal 800 Hz / 1 kHz) absorbs the device-clock jitter that VRS records;
    the low-pass ahead of it is what makes the rate change legal.
    """
    if len(time_sec) < 2:
        raise ValueError("need at least two samples to resample")
    filtered = antialias(values, native_rate(time_sec), rate_hz)
    duration = float(time_sec[-1] - time_sec[0])
    count = int(np.floor(duration * rate_hz)) + 1
    grid = time_sec[0] + np.arange(count, dtype=np.float64) / rate_hz
    out = np.column_stack(
        [np.interp(grid, time_sec, filtered[:, channel]) for channel in range(filtered.shape[1])]
    )
    return grid - grid[0], out


def stationary_gravity_g(acc_g: np.ndarray, rate_hz: float = RATE_HZ, seconds: float = 2.0) -> float:
    """Mean |acc| over the quietest `seconds`: should read ~1.0 g if units are right."""
    window = max(int(rate_hz * seconds), 1)
    norm = np.linalg.norm(acc_g, axis=1)
    if len(norm) < window:
        return float(norm.mean())
    kernel = np.ones(window) / window
    mean = np.convolve(norm, kernel, mode="valid")
    energy = np.convolve((norm - norm.mean()) ** 2, kernel, mode="valid")
    return float(mean[int(np.argmin(energy))])


# --------------------------------------------------------------------------
# Take bookkeeping
# --------------------------------------------------------------------------


def subject_id(take: Mapping) -> tuple[str, str]:
    """(subject, source). ``p<participant_uid>`` when known, else the capture uid."""
    participant = take.get("participant_uid")
    if participant is not None and str(participant).strip() not in {"", "None", "null"}:
        return f"p{participant}", "participant_uid"
    capture = take.get("capture_uid")
    if capture:
        return f"capture_{capture}", "capture_uid_fallback"
    raise ValueError(
        f"take {take.get('take_uid')} has neither participant_uid nor capture_uid; refusing to "
        "invent a per-take subject id (it would break subject-disjoint splits)"
    )


def session_id(take_uid: str, subject: str, part: int, total_parts: int) -> str:
    base = f"{SESSION_PREFIX}_{take_uid}_{subject}_{STREAM_TOKEN}"
    return base if total_parts == 1 else f"{base}_part{part:02d}"


def parse_session_id(sid: str) -> tuple[str, str, int]:
    """Inverse of :func:`session_id`: ``(take_uid, subject, part)``.

    Take uids are hyphenated uuids and never contain ``_``, so the first
    underscore after the prefix separates the take from the subject -- which is
    what lets ``capture_<uuid>`` fallback subjects survive the round trip.
    """
    if not sid.startswith(f"{SESSION_PREFIX}_") or STREAM_TOKEN not in sid:
        raise ValueError(f"not an {DATASET} session id: {sid}")
    body = sid[len(SESSION_PREFIX) + 1 :]
    head, _, tail = body.partition(f"_{STREAM_TOKEN}")
    take_uid, _, subject = head.partition("_")
    part = int(tail.rsplit("part", 1)[1]) if "part" in tail else 1
    return take_uid, subject, part


def _session_rows(path: Path) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    return int(pq.ParquetFile(path).metadata.num_rows)


def take_directory(root: Path, take: Mapping) -> Path | None:
    name = str(take.get("take_name") or "")
    candidates = []
    if take.get("root_dir"):
        candidates.append(root / str(take["root_dir"]))
    if name:
        candidates.append(root / "takes" / name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    if name:
        matches = sorted(
            (path for path in root.rglob(name) if path.is_dir()),
            key=lambda p: (len(p.parts), str(p)),
        )
        if matches:
            return matches[0]
    return None


def find_take_vrs(take_dir: Path, take: Mapping | None = None) -> Path | None:
    """Pick the take's Aria VRS, preferring the image-stream-free file.

    ``take_vrs_noimagestream`` writes files suffixed ``_noimagestreams.vrs``; the
    full ``take_vrs`` part writes ``<cam_id>.vrs``. Either is readable, so we take
    whichever is present and prefer the small one. Cam ids come from the take's
    own embedded capture record where available, so an exo GoPro can never win.
    """
    candidates = sorted(take_dir.rglob("*.vrs"))
    if not candidates:
        return None
    ego_ids: list[str] = []
    capture = (take or {}).get("capture") or {}
    for camera in capture.get("cameras", []) or []:
        if camera.get("is_ego") and str(camera.get("device_type", "")).lower() == "aria":
            ego_ids.append(str(camera.get("cam_id", "")))

    def rank(path: Path) -> tuple:
        stem = path.stem.lower()
        no_image = 0 if "noimagestream" in stem else 1
        ego = 0 if any(cam and stem.startswith(cam.lower()) for cam in ego_ids) else 1
        aria = 0 if "aria" in stem else 1
        return (no_image, ego, aria, str(path))

    return sorted(candidates, key=rank)[0]


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


def session_frames(
    time_sec: np.ndarray,
    accel_ms2: np.ndarray,
    gyro_rads: np.ndarray,
    subject: str,
    rate_hz: float = RATE_HZ,
    min_seconds: float = MIN_SESSION_SECONDS,
    max_gap: float = MAX_GAP_SECONDS,
) -> list[pd.DataFrame]:
    """Gap-split, anti-alias, resample and package one take's IMU as HALO frames."""
    frames: list[pd.DataFrame] = []
    for span in split_on_gaps(time_sec, max_gap):
        segment_time = time_sec[span]
        if len(segment_time) < 2:
            continue
        if float(segment_time[-1] - segment_time[0]) < min_seconds:
            continue
        values = np.hstack([accel_ms2[span] / GRAVITY_MS2, gyro_rads[span]])
        grid, resampled = resample_uniform(segment_time, values, rate_hz)
        if len(grid) < int(min_seconds * rate_hz):
            continue
        frame = pd.DataFrame(
            resampled.astype(np.float32), columns=list(SIGNAL_COLUMNS)
        )
        frame.insert(0, "timestamp_sec", grid.astype(np.float64))
        frame["subject"] = str(subject)
        frames.append(frame)
    return frames


def convert(
    raw_dir: Path = DOWNLOADS,
    output_dir: Path = DS_DIR,
    limit_takes: int | None = None,
    max_hours_per_take: float | None = 1.0,
    keep_existing: bool = False,
) -> bool:
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    try:
        from data.pretraining.ego_exo4d.fetch import load_takes  # noqa: PLC0415
    except ImportError:  # pragma: no cover - direct-script execution
        from fetch import load_takes  # type: ignore

    takes = sorted(load_takes(raw_dir), key=lambda t: str(t["take_uid"]))
    if limit_takes is not None:
        takes = takes[:limit_takes]
    if not takes:
        raise FileNotFoundError(
            f"no takes listed under {raw_dir}; run "
            "`python -m data.pretraining.ego_exo4d.fetch --takes <N>`"
        )

    sessions_dir = output_dir / "sessions"
    labels_path = output_dir / "labels.json"
    labels: dict[str, list[str]] = {}
    carried_takes: set[str] = set()
    #: take_uid -> subject, across carried-forward AND newly converted sessions.
    take_subjects: dict[str, str] = {}
    if keep_existing and sessions_dir.is_dir():
        # Incremental mode: the batch loop in README.md deletes each batch's VRS after
        # converting it, so wiping sessions/ here would destroy work that can no longer
        # be rebuilt. Existing sessions are carried forward and their takes are skipped.
        existing = sorted(
            path.name for path in sessions_dir.iterdir() if (path / "data.parquet").is_file()
        )
        labels = {sid: [UNLABELED] for sid in existing}
        parsed = [parse_session_id(sid) for sid in existing]
        carried_takes = {uid for uid, _, _ in parsed}
        take_subjects = {uid: subject for uid, subject, _ in parsed}
    elif sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    subjects: set[str] = set(take_subjects.values())
    stats["sessions"] = len(labels)
    stats["samples"] = sum(
        _session_rows(sessions_dir / sid / "data.parquet") for sid in labels
    )
    stats["takes_carried_forward"] = len(carried_takes)
    imu_labels: Counter = Counter()
    gravity_probes: list[float] = []
    max_seconds = None if max_hours_per_take is None else float(max_hours_per_take) * 3600.0

    for index, take in enumerate(takes, start=1):
        take_uid = str(take["take_uid"])
        if take_uid in carried_takes:
            continue
        take_dir = take_directory(raw_dir, take)
        if take_dir is None:
            stats["takes_without_directory"] += 1
            continue
        vrs_path = find_take_vrs(take_dir, take)
        if vrs_path is None:
            stats["takes_without_vrs"] += 1
            continue
        try:
            subject, _source = subject_id(take)
        except ValueError as error:
            print(f"[ego_exo4d] skip {take_uid}: {error}")
            stats["takes_without_subject"] += 1
            continue

        provider = open_provider(vrs_path)
        try:
            imu_label, stream_id, count = resolve_imu_stream(provider)
        except RuntimeError as error:
            print(f"[ego_exo4d] skip {take_uid}: {error}")
            stats["takes_without_imu"] += 1
            continue
        time_sec, accel, gyro = read_imu(provider, stream_id, count, max_seconds)
        if len(time_sec) < 2:
            stats["takes_without_samples"] += 1
            continue

        frames = session_frames(time_sec, accel, gyro, subject)
        if not frames:
            stats["takes_without_usable_segments"] += 1
            continue

        for part, frame in enumerate(frames, start=1):
            sid = session_id(take_uid, subject, part, len(frames))
            destination = sessions_dir / sid
            destination.mkdir()
            frame.to_parquet(destination / "data.parquet", index=False)
            labels[sid] = [UNLABELED]
            gravity_probes.append(
                stationary_gravity_g(frame[list(ACC_COLUMNS)].to_numpy(np.float64))
            )
            stats["sessions"] += 1
            stats["samples"] += len(frame)

        subjects.add(subject)
        take_subjects[take_uid] = subject
        imu_labels[imu_label] += 1
        stats["takes"] += 1
        hours = stats["samples"] / RATE_HZ / 3600.0
        print(
            f"[ego_exo4d] {index:04d}/{len(takes)} {take_uid} {subject} {imu_label}: "
            f"{len(frames)} session(s), {sum(len(f) for f in frames) / RATE_HZ / 60:.1f} min "
            f"(running total {hours:.2f} h)"
        )

    if not labels:
        return False

    median_gravity = float(np.median(gravity_probes)) if gravity_probes else None
    if median_gravity is not None and not (0.8 <= median_gravity <= 1.2):
        print(
            f"[ego_exo4d] WARNING: quietest-window |acc| median is {median_gravity:.3f} g, "
            "not ~1.0 g. Check the m/s^2 -> g conversion or the source units."
        )

    # Derived from every session on disk, not just this run, so an incremental
    # convert can never report a clean provenance while carrying fallback sessions.
    fallback_subjects = {name for name in subjects if name.startswith("capture_")}
    fallback_takes = sum(
        1 for name in take_subjects.values() if name.startswith("capture_")
    )
    if fallback_subjects and len(fallback_subjects) < len(subjects):
        subject_source = "mixed"
    elif fallback_subjects:
        subject_source = "capture_uid_fallback"
    elif subjects:
        subject_source = "participant_uid"
    else:
        subject_source = "unknown"

    note = (
        "Head-mounted Project Aria IMU only (no video/audio/gaze). One IMU per take, "
        "imu-right (1202-1) preferred over imu-left (1202-2). Accel m/s^2 -> g, gravity present; "
        "gyro rad/s. Anti-aliased and resampled from ~800 Hz/1 kHz to a uniform 200 Hz grid. "
        f"Sessions split at gaps > {MAX_GAP_SECONDS} s instead of interpolating across dropouts. "
        "No activity labels: every session carries the reserved marker __unlabeled__ and this "
        "dataset is Phase-A only."
    )
    if fallback_takes:
        note += (
            f" WARNING: {fallback_takes} take(s) had no participant_uid and fall back to a "
            "capture-level subject id; those takes are subject-disjoint at capture granularity, "
            "not participant granularity."
        )

    labels_path.write_text(json.dumps(dict(sorted(labels.items())), indent=2) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "Ego-Exo4D Aria head IMU (bounded subset)",
                "source": SOURCE_URL,
                "num_subjects": len(subjects),
                "sampling_rate_hz": RATE_HZ,
                "channels": list(SIGNAL_COLUMNS),
                "unit": "g",
                "gravity_state": "present",
                "phase_a_only": True,
                "num_takes": stats["takes"] + stats["takes_carried_forward"],
                "takes_converted_this_run": stats["takes"],
                "num_sessions": stats["sessions"],
                "hours": round(stats["samples"] / RATE_HZ / 3600.0, 3),
                "imu_streams": dict(imu_labels),
                "imu_streams_scope": (
                    "this run only; carried-forward sessions are not re-inspected"
                    if stats["takes_carried_forward"]
                    else "all takes"
                ),
                "subject_id_source": subject_source,
                "subject_id_fallback_takes": fallback_takes,
                "subject_id_fallback_subjects": len(fallback_subjects),
                "gap_split_seconds": MAX_GAP_SECONDS,
                "native_rate_note": "resampled from Aria imu-right ~800 Hz / imu-left ~1 kHz",
                "stationary_gravity_g_median": (
                    None if median_gravity is None else round(median_gravity, 4)
                ),
                "max_hours_per_take": max_hours_per_take,
                "skipped": {key: value for key, value in stats.items() if key.startswith("takes_")},
                "note": note,
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "display_name": "Ego-Exo4D Aria head IMU (bounded subset)",
                "sampling_rate_hz": RATE_HZ,
                "pre_windowed": False,
                "streaming_grid": True,
                "role": "pretrain_scale",
                "phase_a_only": True,
                "activities": [],
                "num_subjects": None,
                "channels": list(SIGNAL_COLUMNS),
                "core_channels": {name: name for name in SIGNAL_COLUMNS},
                "extra_channels": [],
                # Label-free scale source: two bytes a sample is what makes the corpus fit.
                # float16 resolves ~0.001 g at 1 g, well below anything the <14 Hz frontend reads.
                "grid_dtype": "float16",
                "placement": "the head (smart glasses)",
                "note": note,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[ego_exo4d] complete: {dict(stats)}; subjects={len(subjects)}")
    return True


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=DOWNLOADS)
    parser.add_argument("--output-dir", type=Path, default=DS_DIR)
    parser.add_argument("--takes", type=int, default=None, help="convert at most N takes")
    parser.add_argument("--max-hours-per-take", type=float, default=1.0)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="carry forward already-converted sessions instead of rebuilding from scratch "
        "(required by the fetch/convert/delete batch loop in README.md)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not convert(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        limit_takes=args.takes,
        max_hours_per_take=args.max_hours_per_take,
        keep_existing=args.keep_existing,
    ):
        raise SystemExit("no Ego-Exo4D sessions were produced")


if __name__ == "__main__":
    main()
