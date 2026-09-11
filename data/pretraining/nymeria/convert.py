"""Convert downloaded Nymeria sequences into label-free HALO Phase-A sessions.

Nymeria records, simultaneously and time-synchronised, an Xsens MVN Link suit
(17 inertial trackers, 240 Hz) and Project Aria headset + two miniAria wristband
IMUs.  There are no activity annotations here, so every session carries the
reserved ``__unlabeled__`` marker: Phase-A self-supervision may use them, label
vocabulary construction and the Phase-B evidence bank must not.

TWO OUTPUT DATASETS, NOT ONE
----------------------------
``data/scripts/build_grids.py`` reads ONE scalar ``sampling_rate_hz`` per dataset
directory (``native_rate = float(json.loads((ds_dir/"metadata.json").read_text())
["sampling_rate_hz"])``).  The Xsens streams are 240 Hz and the Aria streams are
200 Hz, so a single dataset directory cannot describe both without lying about one
of them.  The converter therefore writes two sibling dataset directories, each a
complete, self-consistent instance of the repo-wide session contract::

    data/pretraining/nymeria_xsens/{sessions/,labels.json,manifest.json,metadata.json}  240 Hz
    data/pretraining/nymeria_aria/{sessions/,labels.json,manifest.json,metadata.json}   200 Hz

which is exactly where ``corpus_roots.dataset_root(name)`` resolves them.  Each
``metadata.json`` is copied from the authored template in this package
(``nymeria_<family>.metadata.json``).  This package also keeps a ``labels.json``
(the union) and a ``manifest.json`` (a summary naming both datasets and carrying no
scalar rate of its own).  Note that this package deliberately holds NO
``metadata.json``: ``corpus_roots`` treats any directory with one as a dataset.

SIGNAL HANDLING
---------------
Xsens (MVNX):
  * The device-frame measurement is reconstructed, never taken from the global
    frame.  ``sensorFreeAcceleration`` is gravity-REMOVED and expressed globally, so
    the accelerometer reading is recovered as
    ``a_device = R(sensorOrientation)^T @ (freeAcc_global + g_up)`` with
    ``g_up = (0, 0, +9.80665)`` in MVN's Z-up global frame, then divided by
    9.80665 to reach g.  A stationary sensor therefore reads |a| = 1 g with
    gravity PRESENT, as the contract requires.
  * MVNX carries no sensor-level angular velocity.  The sensor is rigid with its
    segment, so segment ``angularVelocity`` (global frame) is the sensor's angular
    velocity and is rotated by the same ``sensorOrientation``:
    ``w_device = R^T @ w_global``.
  * If the sensor-level fields are absent, the segment fields ``acceleration`` +
    ``orientation`` are used identically; the manifest records which path ran
    (``acceleration_source``: ``sensor`` or ``segment``).
  * Angular units are read from the file's own unit declarations when present.
    When the file declares nothing, the MVNX SI default (rad/s) is assumed, a
    warning is printed, and ``angular_unit_source`` in the manifest says
    ``assumed_mvnx_si_default`` rather than pretending it was verified.  A
    magnitude heuristic additionally warns when the values look like deg/s.

Aria (VRS):
  * One IMU per device: ``imu-right`` (1202-1, ~800 Hz) preferred, ``imu-left``
    (1202-2, ~1 kHz) as fallback; the choice is recorded per session.
  * Accel arrives in m/s^2 and gyro in rad/s, both already in the device frame.
  * Timestamps are non-uniform device clocks in nanoseconds.  Each contiguous run
    is linearly resampled onto the sensor's nominal uniform clock, then
    anti-alias filtered and decimated to exactly 200 Hz with
    ``scipy.signal.resample_poly`` (Kaiser-window FIR, cutoff at the output
    Nyquist).  The frontend's f_max is ~14 Hz, so 200 Hz is far above what is
    used and storing 1 kHz would cost 5x for nothing.

GAP POLICY (both modalities)
  A time gap longer than 0.5 s is a hard boundary, never interpolated across.
  The session is split into ``..._p01``, ``..._p02`` parts; the stream token stays
  a matchable substring of the session id.  Each part is then truncated to
  complete eight-second windows, so no downstream grid window can cross a gap.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from data.pretraining.corpus_plan import PRETRAIN_WINDOW_SECONDS

DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"
#: The label-free corpus root. `corpus_roots.dataset_root(name)` resolves a dataset to
#: `data/pretraining/<name>`, so the two dataset directories are SIBLINGS of this module
#: package, not children of it.
CORPUS_ROOT = DS_DIR.parent

GRAVITY_MS2 = 9.80665
XSENS_RATE_HZ = 240.0
ARIA_RATE_HZ = 200.0
ARIA_NOMINAL_RATES = (800.0, 1000.0)
WINDOW_SECONDS = PRETRAIN_WINDOW_SECONDS
MAX_GAP_SECONDS = 0.5
UNLABELED = "__unlabeled__"

XSENS_DATASET = "nymeria_xsens"
ARIA_DATASET = "nymeria_aria"

#: Authored, tracked metadata.json templates. `corpus_roots._is_dataset_dir` recognises a
#: dataset by the presence of metadata.json, so each dataset directory needs its own copy;
#: they are copied rather than authored in place because the dataset directories are
#: generated output (see .gitignore).
METADATA_TEMPLATES = {
    XSENS_DATASET: DS_DIR / f"{XSENS_DATASET}.metadata.json",
    ARIA_DATASET: DS_DIR / f"{ARIA_DATASET}.metadata.json",
}

#: Stream token -> the MVNX sensor/segment labels that may carry it, normalised
#: (lowercased, non-alphanumerics stripped).  MVN Link's 17-tracker configuration
#: names the trunk tracker ``T8``; some exports call it ``Sternum``.
XSENS_STREAMS: Mapping[str, tuple[str, ...]] = {
    "xsens_head": ("head",),
    "xsens_sternum": ("t8", "sternum"),
    "xsens_pelvis": ("pelvis",),
    "xsens_lforearm": ("leftforearm", "lforearm", "leftlowerarm"),
    "xsens_rforearm": ("rightforearm", "rforearm", "rightlowerarm"),
    "xsens_lupperarm": ("leftupperarm", "lupperarm"),
    "xsens_rupperarm": ("rightupperarm", "rupperarm"),
    "xsens_lthigh": ("leftupperleg", "leftthigh", "lupperleg"),
    "xsens_rthigh": ("rightupperleg", "rightthigh", "rupperleg"),
    "xsens_lshank": ("leftlowerleg", "leftshank", "llowerleg"),
    "xsens_rshank": ("rightlowerleg", "rightshank", "rlowerleg"),
}

#: Recording directory -> stream token.  ``recording_observer`` is deliberately
#: absent: it is a second person following the participant.
ARIA_STREAMS: Mapping[str, str] = {
    "recording_head": "aria_head",
    "recording_lwrist": "aria_lwrist",
    "recording_rwrist": "aria_rwrist",
}
ARIA_IMU_PREFERENCE = ("imu-right", "imu-left")

#: Where a participant identity may hide in ``<seq>/metadata.json``.  Checked in
#: order, at any nesting depth.  UNVERIFIED against the real release -- see
#: ``resolve_subject`` and the README.
SUBJECT_KEYS = (
    "participant_id",
    "participant",
    "subject_id",
    "subject",
    "wearer_id",
    "wearer",
    "actor_id",
    "actor",
    "user_id",
)

_ARIA_TOOLS_HINT = (
    "projectaria_tools is required to read Nymeria's motion.vrs files.\n"
    "    pip install projectaria-tools\n"
    "(or 'pip install projectaria-tools[all]'). Install it, or run with "
    "--streams xsens to convert only the Xsens suit."
)


class NpzSchemaUnknown(RuntimeError):
    """body/xdata.npz did not carry arrays this converter recognises."""


class MvnxParseError(RuntimeError):
    """The MVNX file did not contain the fields the contract needs."""


# ======================================================================================
# small numeric primitives (all pure, all unit-tested)
# ======================================================================================


def normalise_label(label: str) -> str:
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def quat_to_matrix(quat: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """Rotation matrices from unit quaternions, shape (..., 4) -> (..., 3, 3).

    The returned ``R`` maps device/sensor-frame vectors into the global frame
    (``v_global = R @ v_device``), which is the MVNX ``sensorOrientation`` /
    ``orientation`` convention.  MVNX stores the real part FIRST (``wxyz``).
    """
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"expected quaternions of shape (...,4), got {quat.shape}")
    if order == "wxyz":
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    elif order == "xyzw":
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    else:
        raise ValueError(f"unknown quaternion order {order!r}")
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    norm = np.where(norm == 0.0, 1.0, norm)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    matrix = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def to_device_frame(vectors_global: np.ndarray, quat: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """Rotate global-frame vectors into the device frame: ``v_device = R^T @ v_global``."""
    matrix = quat_to_matrix(quat, order)
    return np.einsum("...ji,...j->...i", matrix, np.asarray(vectors_global, dtype=np.float64))


def gravity_vector(up_axis: str = "z") -> np.ndarray:
    """Specific force measured by a stationary accelerometer, in the global frame.

    An accelerometer at rest reads proper acceleration ``+g`` along the world UP
    axis (not the free-fall direction), which is why a stationary device reads
    +1 g rather than -1 g.  MVN's default global frame is Z-up, right-handed.
    """
    index = {"x": 0, "y": 1, "z": 2}.get(up_axis.lower())
    if index is None:
        raise ValueError(f"up_axis must be one of x/y/z, got {up_axis!r}")
    vector = np.zeros(3, dtype=np.float64)
    vector[index] = GRAVITY_MS2
    return vector


def free_acceleration_to_device_g(
    free_acc_global_ms2: np.ndarray,
    quat: np.ndarray,
    *,
    up_axis: str = "z",
    order: str = "wxyz",
) -> np.ndarray:
    """Gravity-free global acceleration -> gravity-PRESENT device-frame acceleration in g."""
    specific_global = np.asarray(free_acc_global_ms2, dtype=np.float64) + gravity_vector(up_axis)
    return to_device_frame(specific_global, quat, order) / GRAVITY_MS2


def contiguous_blocks(
    times_sec: np.ndarray,
    max_gap_seconds: float = MAX_GAP_SECONDS,
    expected_rate_hz: float | None = None,
) -> list[tuple[int, int]]:
    """Split a timestamp array at any gap longer than ``max_gap_seconds``.

    Returns half-open ``(start, stop)`` index pairs.  Non-increasing steps are
    boundaries too: a clock that jumps backwards is a gap, not something to sort
    silently.
    """
    times = np.asarray(times_sec, dtype=np.float64)
    if times.size == 0:
        return []
    if times.size == 1:
        return [(0, 1)]
    step = np.diff(times)
    # Xsens timestamps are frame-index based. A single dropped frame must remain a timing
    # boundary: rebuilding timestamps at the nominal rate would otherwise compress physical time.
    max_step = (1.5 / expected_rate_hz if expected_rate_hz else max_gap_seconds)
    boundaries = np.flatnonzero((step > max_step) | (step <= 0.0)) + 1
    bounds = np.concatenate(([0], boundaries, [times.size]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def resample_uniform(times_sec: np.ndarray, values: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate a jittered but gap-free run onto a uniform ``rate_hz`` grid."""
    times = np.asarray(times_sec, dtype=np.float64)
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if values.shape[0] != times.shape[0]:
        values = values.T
    if times.size < 2:
        return times.copy(), values.copy()
    count = int(np.floor((times[-1] - times[0]) * rate_hz)) + 1
    grid = times[0] + np.arange(count, dtype=np.float64) / rate_hz
    out = np.column_stack(
        [np.interp(grid, times, values[:, column]) for column in range(values.shape[1])]
    )
    return grid, out


def estimate_rate(times_sec: np.ndarray) -> float:
    times = np.asarray(times_sec, dtype=np.float64)
    if times.size < 2 or times[-1] <= times[0]:
        return float("nan")
    return float((times.size - 1) / (times[-1] - times[0]))


def snap_nominal_rate(
    estimate: float, candidates: Sequence[float] = ARIA_NOMINAL_RATES, tolerance: float = 0.05
) -> float:
    """Round a measured rate to a documented nominal rate, or keep the estimate."""
    if not np.isfinite(estimate):
        return float("nan")
    best = min(candidates, key=lambda c: abs(c - estimate))
    return float(best) if abs(best - estimate) <= tolerance * best else float(estimate)


def decimate_to(values: np.ndarray, rate_in: float, rate_out: float) -> np.ndarray:
    """Anti-alias low-pass then decimate ``values`` (T, C) from ``rate_in`` to ``rate_out``.

    ``scipy.signal.resample_poly`` designs a Kaiser-window FIR whose cutoff is the
    OUTPUT Nyquist, so content above ``rate_out / 2`` is filtered out before
    downsampling rather than folded back onto the passband.
    """
    from scipy.signal import resample_poly  # local import: scipy only needed on this path

    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(rate_in) or rate_in <= 0:
        raise ValueError(f"input rate must be positive and finite, got {rate_in}")
    if abs(rate_in - rate_out) < 1e-9:
        return values
    ratio = Fraction(rate_out / rate_in).limit_denominator(1000)
    if ratio.numerator <= 0:
        raise ValueError(f"cannot resample {rate_in} Hz -> {rate_out} Hz")
    return resample_poly(values, up=ratio.numerator, down=ratio.denominator, axis=0)


def truncate_to_windows(
    count: int, rate_hz: float, window_seconds: float = WINDOW_SECONDS
) -> int:
    """Largest sample count that is a whole number of ``window_seconds`` windows."""
    window = int(round(rate_hz * window_seconds))
    return (count // window) * window if window else 0


# ======================================================================================
# MVNX (Xsens)
# ======================================================================================


@dataclass
class StreamSignal:
    token: str
    acc_g: np.ndarray
    gyro_rads: np.ndarray | None
    times_sec: np.ndarray
    rate_hz: float
    detail: dict = field(default_factory=dict)


@dataclass
class MvnxRecording:
    frame_rate: float
    times_sec: np.ndarray
    sensor_labels: list[str]
    segment_labels: list[str]
    subject_label: str | None
    acceleration_source: str  # "sensor" | "segment"
    angular_unit: str
    angular_unit_source: str
    declared_units: dict
    fields: dict[str, np.ndarray]


def _tag(element: ET.Element) -> str:
    return element.tag.rpartition("}")[2]


def _floats(element: ET.Element | None) -> np.ndarray | None:
    if element is None or not (element.text or "").strip():
        return None
    return np.fromstring(element.text.replace(",", " "), sep=" ", dtype=np.float64)


def _collect_unit_declarations(root_attrs: Iterable[tuple[str, Mapping[str, str]]]) -> dict:
    declared: dict = {}
    for tag, attrs in root_attrs:
        for key, value in attrs.items():
            if "unit" in key.lower():
                declared[f"{tag}.{key}"] = value
    return declared


#: A body-worn MTw gyroscope tops out around 2000 deg/s ~= 35 rad/s, so a p99 above
#: this is far more likely to be degrees mislabelled as radians than real motion.
DEGREES_HEURISTIC_RAD_S = 35.0


def looks_like_degrees(angular_velocity: np.ndarray) -> float | None:
    """Return the p99 magnitude when it is implausibly large for rad/s, else None."""
    values = np.asarray(angular_velocity, dtype=np.float64)
    if values.size == 0:
        return None
    magnitude = float(np.percentile(np.abs(values), 99))
    return magnitude if magnitude > DEGREES_HEURISTIC_RAD_S else None


def _resolve_angular_unit(declared: dict, override: str | None) -> tuple[str, str]:
    if override in ("rad", "deg"):
        return override, "cli_override"
    for key, value in declared.items():
        text = str(value).lower()
        if "deg" in text:
            return "deg", f"declared:{key}={value}"
        if "rad" in text:
            return "rad", f"declared:{key}={value}"
    return "rad", "assumed_mvnx_si_default"


def parse_mvnx(
    path: Path,
    *,
    angular_unit: str | None = None,
    prefer_sensor_fields: bool = True,
) -> MvnxRecording:
    """Stream-parse an MVNX file into the arrays the contract needs.

    Uses ``iterparse`` and clears each frame after reading it: a 15-minute
    sequence at 240 Hz is ~216,000 frames and the file is hundreds of MB.
    """
    frame_rate = None
    subject_label = None
    sensor_labels: list[str] = []
    segment_labels: list[str] = []
    unit_sources: list[tuple[str, Mapping[str, str]]] = []
    wanted = (
        "sensorFreeAcceleration",
        "sensorOrientation",
        "acceleration",
        "orientation",
        "angularVelocity",
    )
    columns: dict[str, list[np.ndarray]] = {name: [] for name in wanted}
    indices: list[float] = []
    stamps: list[float] = []
    in_sensors = in_segments = False
    frame_count = 0

    for event, element in ET.iterparse(str(path), events=("start", "end")):
        tag = _tag(element)
        if event == "start":
            if tag == "sensors":
                in_sensors = True
            elif tag == "segments":
                in_segments = True
            elif tag == "subject":
                subject_label = element.attrib.get("label")
                if "frameRate" in element.attrib:
                    frame_rate = float(element.attrib["frameRate"])
                unit_sources.append((tag, dict(element.attrib)))
            elif tag in ("mvnx", "mvn", "frames"):
                unit_sources.append((tag, dict(element.attrib)))
            continue
        # end events
        if tag == "sensors":
            in_sensors = False
        elif tag == "segments":
            in_segments = False
        elif tag == "sensor" and in_sensors:
            sensor_labels.append(element.attrib.get("label", f"sensor{len(sensor_labels)}"))
        elif tag == "segment" and in_segments:
            segment_labels.append(element.attrib.get("label", f"segment{len(segment_labels)}"))
        elif tag == "frame":
            if element.attrib.get("type", "normal") != "normal":
                element.clear()
                continue
            children = {_tag(child): child for child in element}
            for name in wanted:
                values = _floats(children.get(name))
                columns[name].append(values if values is not None else np.empty(0))
            index_attr = element.attrib.get("index")
            indices.append(float(index_attr) if index_attr is not None else float(frame_count))
            time_attr = element.attrib.get("time")
            stamps.append(float(time_attr) / 1000.0 if time_attr is not None else float("nan"))
            frame_count += 1
            element.clear()

    if frame_count == 0:
        raise MvnxParseError(f"{path}: no <frame type='normal'> elements")
    if frame_rate is None:
        frame_rate = XSENS_RATE_HZ
        print(f"[nymeria] WARNING {path.name}: no frameRate attribute; assuming {frame_rate} Hz")

    fields: dict[str, np.ndarray] = {}
    for name, rows in columns.items():
        widths = {row.size for row in rows}
        if widths == {0}:
            continue
        if len(widths) != 1:
            raise MvnxParseError(f"{path}: ragged <{name}> rows ({sorted(widths)})")
        fields[name] = np.asarray(rows, dtype=np.float64)

    has_sensor = "sensorFreeAcceleration" in fields and "sensorOrientation" in fields
    has_segment = "acceleration" in fields and "orientation" in fields
    if prefer_sensor_fields and has_sensor:
        source = "sensor"
    elif has_segment:
        source = "segment"
        if prefer_sensor_fields:
            print(
                f"[nymeria] {path.name}: sensor-level fields absent; "
                "falling back to segment acceleration + orientation"
            )
    else:
        raise MvnxParseError(
            f"{path}: needs either (sensorFreeAcceleration, sensorOrientation) or "
            f"(acceleration, orientation); found {sorted(fields)}"
        )
    if "angularVelocity" not in fields:
        print(f"[nymeria] WARNING {path.name}: no <angularVelocity>; sessions will be acc-only")

    declared = _collect_unit_declarations(unit_sources)
    unit, unit_source = _resolve_angular_unit(declared, angular_unit)
    if unit_source == "assumed_mvnx_si_default":
        print(
            f"[nymeria] WARNING {path.name}: file declares no angular unit; assuming rad/s "
            "(MVNX SI default). Pass --angular-unit deg if your export uses degrees."
        )
        if "angularVelocity" in fields:
            magnitude = looks_like_degrees(fields["angularVelocity"])
            if magnitude is not None:
                print(
                    f"[nymeria] WARNING {path.name}: |angularVelocity| p99 = {magnitude:.1f}, "
                    f"above the {DEGREES_HEURISTIC_RAD_S:g} rad/s plausibility ceiling for a "
                    "body-worn gyroscope. This export may be in deg/s -- verify and re-run "
                    "with --angular-unit deg if so."
                )

    # Frame index is the authoritative clock: an index jump is a dropped-frame gap,
    # which the `time` attribute (a nominal ms counter) would hide.
    times = (np.asarray(indices, dtype=np.float64) - float(indices[0])) / float(frame_rate)
    return MvnxRecording(
        frame_rate=float(frame_rate),
        times_sec=times,
        sensor_labels=sensor_labels,
        segment_labels=segment_labels,
        subject_label=subject_label,
        acceleration_source=source,
        angular_unit=unit,
        angular_unit_source=unit_source,
        declared_units=declared,
        fields=fields,
    )


def _label_index(labels: Sequence[str], aliases: Sequence[str]) -> int | None:
    normalised = [normalise_label(label) for label in labels]
    for alias in aliases:
        if alias in normalised:
            return normalised.index(alias)
    return None


def mvnx_stream_signals(
    recording: MvnxRecording,
    *,
    streams: Mapping[str, tuple[str, ...]] = XSENS_STREAMS,
    up_axis: str = "z",
    quat_order: str = "wxyz",
) -> list[StreamSignal]:
    """Reconstruct device-frame acc (g) + gyro (rad/s) for each requested placement."""
    signals: list[StreamSignal] = []
    if recording.acceleration_source == "sensor":
        acc_field, quat_field = "sensorFreeAcceleration", "sensorOrientation"
        acc_labels = recording.sensor_labels
    else:
        acc_field, quat_field = "acceleration", "orientation"
        acc_labels = recording.segment_labels
    acc_all = recording.fields[acc_field].reshape(len(recording.times_sec), -1, 3)
    quat_all = recording.fields[quat_field].reshape(len(recording.times_sec), -1, 4)
    omega_all = None
    if "angularVelocity" in recording.fields:
        omega_all = recording.fields["angularVelocity"].reshape(len(recording.times_sec), -1, 3)
    gyro_scale = np.pi / 180.0 if recording.angular_unit == "deg" else 1.0

    for token, aliases in streams.items():
        acc_index = _label_index(acc_labels, aliases)
        if acc_index is None or acc_index >= acc_all.shape[1] or acc_index >= quat_all.shape[1]:
            continue
        quat = quat_all[:, acc_index, :]
        acc_g = free_acceleration_to_device_g(
            acc_all[:, acc_index, :], quat, up_axis=up_axis, order=quat_order
        )
        gyro = None
        if omega_all is not None:
            # angularVelocity is per-SEGMENT; the sensor is rigid with its segment so
            # they share an angular velocity, but it must be rotated by the SENSOR's
            # orientation to land in the sensor's own axes.
            segment_index = _label_index(recording.segment_labels, aliases)
            if segment_index is not None and segment_index < omega_all.shape[1]:
                gyro = to_device_frame(
                    omega_all[:, segment_index, :] * gyro_scale, quat, quat_order
                )
        signals.append(
            StreamSignal(
                token=token,
                acc_g=acc_g,
                gyro_rads=gyro,
                times_sec=recording.times_sec,
                rate_hz=recording.frame_rate,
                detail={
                    "acceleration_source": recording.acceleration_source,
                    "label": acc_labels[acc_index],
                    "angular_unit": recording.angular_unit,
                    "angular_unit_source": recording.angular_unit_source,
                },
            )
        )
    return signals


# --- body/xdata.npz -------------------------------------------------------------------

#: Candidate array names in ``body/xdata.npz``.  These are CANDIDATES, not verified
#: facts: the file was never opened during development (the release is licence-gated
#: and 80 TB).  ``load_npz_recording`` refuses rather than guessing when none match,
#: and ``--inspect-npz`` prints the real keys so this table can be corrected.
NPZ_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "sensorFreeAcceleration": ("sensorFreeAcceleration", "sensor_free_acceleration", "free_acceleration", "sensorFreeAcc"),
    "sensorOrientation": ("sensorOrientation", "sensor_orientation", "sensorQuat", "sensor_quat"),
    "acceleration": ("acceleration", "segment_acceleration", "segmentAcceleration"),
    "orientation": ("orientation", "segment_orientation", "segmentOrientation"),
    "angularVelocity": ("angularVelocity", "angular_velocity", "segmentAngularVelocity"),
    "sensor_labels": ("sensorLabels", "sensor_labels", "sensors"),
    "segment_labels": ("segmentLabels", "segment_labels", "segments"),
    "frame_rate": ("frameRate", "frame_rate", "fps", "rate"),
}


def inspect_npz(path: Path) -> dict[str, dict]:
    """Report every array in an npz: shape and dtype.  Used by ``--inspect-npz``."""
    with np.load(str(path), allow_pickle=True) as bundle:
        return {
            key: {"shape": list(np.shape(bundle[key])), "dtype": str(np.asarray(bundle[key]).dtype)}
            for key in bundle.files
        }


def _npz_lookup(bundle, canonical: str):
    for candidate in NPZ_CANDIDATES[canonical]:
        if candidate in bundle.files:
            return bundle[candidate]
    return None


def load_npz_recording(path: Path, *, angular_unit: str | None = None) -> MvnxRecording:
    """Load ``body/xdata.npz`` if -- and only if -- it carries recognisable arrays."""
    with np.load(str(path), allow_pickle=True) as bundle:
        found = {name: _npz_lookup(bundle, name) for name in NPZ_CANDIDATES}
        sensor_ok = found["sensorFreeAcceleration"] is not None and found["sensorOrientation"] is not None
        segment_ok = found["acceleration"] is not None and found["orientation"] is not None
        if not (sensor_ok or segment_ok):
            raise NpzSchemaUnknown(
                f"{path}: none of the expected arrays were present. Actual keys and shapes:\n"
                + json.dumps(inspect_npz(path), indent=2)
                + "\nUpdate NPZ_CANDIDATES in convert.py, or use --body-source mvnx."
            )
        source = "sensor" if sensor_ok else "segment"
        acc = np.asarray(found["sensorFreeAcceleration" if sensor_ok else "acceleration"], dtype=np.float64)
        quat = np.asarray(found["sensorOrientation" if sensor_ok else "orientation"], dtype=np.float64)
        frames = acc.shape[0]
        fields = {
            ("sensorFreeAcceleration" if sensor_ok else "acceleration"): acc.reshape(frames, -1),
            ("sensorOrientation" if sensor_ok else "orientation"): quat.reshape(frames, -1),
        }
        if found["angularVelocity"] is not None:
            fields["angularVelocity"] = np.asarray(found["angularVelocity"], dtype=np.float64).reshape(frames, -1)
        rate = float(np.asarray(found["frame_rate"]).ravel()[0]) if found["frame_rate"] is not None else XSENS_RATE_HZ
        labels = {
            key: [str(label) for label in np.asarray(found[key]).ravel()]
            for key in ("sensor_labels", "segment_labels")
            if found[key] is not None
        }
    sensor_labels = labels.get("sensor_labels", [])
    segment_labels = labels.get("segment_labels", sensor_labels)
    if not sensor_labels and not segment_labels:
        raise NpzSchemaUnknown(
            f"{path}: arrays present but no sensor/segment label array, so placements "
            "cannot be identified. Use --body-source mvnx."
        )
    unit, unit_source = _resolve_angular_unit({}, angular_unit)
    return MvnxRecording(
        frame_rate=rate,
        times_sec=np.arange(frames, dtype=np.float64) / rate,
        sensor_labels=sensor_labels or segment_labels,
        segment_labels=segment_labels or sensor_labels,
        subject_label=None,
        acceleration_source=source,
        angular_unit=unit,
        angular_unit_source=f"npz:{unit_source}",
        declared_units={},
        fields=fields,
    )


# ======================================================================================
# Aria VRS
# ======================================================================================


def _imu_attribute(sample, names: Sequence[str]):
    for name in names:
        if hasattr(sample, name):
            return getattr(sample, name)
    raise RuntimeError(
        f"projectaria_tools IMU sample exposes none of {list(names)}; available: "
        f"{[a for a in dir(sample) if not a.startswith('_')]}"
    )


def read_aria_imu(
    vrs_path: Path, preference: Sequence[str] = ARIA_IMU_PREFERENCE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Read ONE IMU stream from a Nymeria ``motion.vrs``.

    Returns ``(times_sec, accel_ms2, gyro_rads, stream_label)``.  ``imu-right``
    (1202-1, ~800 Hz) is preferred over ``imu-left`` (1202-2, ~1 kHz); the label
    actually used is returned so the manifest can record it per session.
    """
    try:
        from projectaria_tools.core import data_provider
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatched module
        raise RuntimeError(_ARIA_TOOLS_HINT) from exc

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"{vrs_path}: projectaria_tools could not open the VRS container")

    for label in preference:
        stream_id = provider.get_stream_id_from_label(label)
        if stream_id is None:
            continue
        count = int(provider.get_num_data(stream_id))
        if count <= 0:
            continue
        times = np.empty(count, dtype=np.float64)
        accel = np.empty((count, 3), dtype=np.float64)
        gyro = np.empty((count, 3), dtype=np.float64)
        for i in range(count):
            sample = provider.get_imu_data_by_index(stream_id, i)
            if isinstance(sample, (tuple, list)):
                sample = sample[0]
            times[i] = float(_imu_attribute(sample, ("capture_timestamp_ns", "captureTimestampNs"))) / 1e9
            accel[i] = np.asarray(_imu_attribute(sample, ("accel_msec2", "accel_msc2", "accel")), dtype=np.float64)
            gyro[i] = np.asarray(_imu_attribute(sample, ("gyro_radsec", "gyro_rad_sec", "gyro")), dtype=np.float64)
        return times, accel, gyro, label
    raise RuntimeError(
        f"{vrs_path}: none of {list(preference)} carried samples "
        "(expected imu-right = 1202-1 or imu-left = 1202-2)"
    )


def aria_stream_signals(
    times_sec: np.ndarray,
    accel_ms2: np.ndarray,
    gyro_rads: np.ndarray,
    token: str,
    stream_label: str,
    *,
    target_rate: float = ARIA_RATE_HZ,
    max_gap_seconds: float = MAX_GAP_SECONDS,
) -> list[StreamSignal]:
    """Gap-split, resample to a uniform native clock, anti-alias, decimate to 200 Hz."""
    signals: list[StreamSignal] = []
    for start, stop in contiguous_blocks(times_sec, max_gap_seconds):
        block_times = np.asarray(times_sec[start:stop], dtype=np.float64)
        if block_times.size < 2:
            continue
        native = snap_nominal_rate(estimate_rate(block_times))
        if not np.isfinite(native) or native < target_rate:
            continue
        stacked = np.column_stack(
            [np.asarray(accel_ms2[start:stop], dtype=np.float64), np.asarray(gyro_rads[start:stop], dtype=np.float64)]
        )
        _, uniform = resample_uniform(block_times, stacked, native)
        decimated = decimate_to(uniform, native, target_rate)
        keep = truncate_to_windows(len(decimated), target_rate)
        if keep == 0:
            continue
        decimated = decimated[:keep]
        signals.append(
            StreamSignal(
                token=token,
                acc_g=decimated[:, :3] / GRAVITY_MS2,
                gyro_rads=decimated[:, 3:],
                times_sec=np.arange(keep, dtype=np.float64) / target_rate,
                rate_hz=target_rate,
                detail={
                    "imu_stream": stream_label,
                    "native_rate_hz": native,
                    "source_span_sec": float(block_times[-1] - block_times[0]),
                },
            )
        )
    return signals


# ======================================================================================
# session writing
# ======================================================================================


def session_frame(signal: StreamSignal, subject: str) -> pd.DataFrame:
    """Build the exact repo-wide session frame for one stream."""
    keep = truncate_to_windows(len(signal.acc_g), signal.rate_hz)
    if keep == 0:
        raise ValueError(f"{signal.token}: fewer than one {WINDOW_SECONDS:g}s window")
    frame = pd.DataFrame(
        {
            "timestamp_sec": np.arange(keep, dtype=np.float64) / float(signal.rate_hz),
            "acc_x": signal.acc_g[:keep, 0].astype(np.float32),
            "acc_y": signal.acc_g[:keep, 1].astype(np.float32),
            "acc_z": signal.acc_g[:keep, 2].astype(np.float32),
        }
    )
    if signal.gyro_rads is not None:
        for axis, column in enumerate("xyz"):
            frame[f"gyro_{column}"] = signal.gyro_rads[:keep, axis].astype(np.float32)
    frame["subject"] = str(subject)
    return frame


def write_session(sessions_dir: Path, session_id: str, frame: pd.DataFrame) -> Path:
    destination = sessions_dir / session_id
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "data.parquet"
    frame.to_parquet(path, index=False)
    return path


def _search_nested(node, keys: Sequence[str]):
    if isinstance(node, Mapping):
        for key in keys:
            value = node.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                return str(value), key
        for value in node.values():
            found = _search_nested(value, keys)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _search_nested(value, keys)
            if found:
                return found
    return None


def resolve_subject(sequence_dir: Path, mvnx_subject: str | None = None) -> tuple[str, str]:
    """Best available stable participant id for a sequence, plus how it was obtained.

    Nymeria has 264 participants across ~1,200 sequences, so falling back to the
    sequence uid INFLATES the subject count and would break subject-disjoint
    splitting.  The fallback is therefore reported loudly, not silently accepted.
    """
    metadata_path = sequence_dir / "metadata.json"
    if metadata_path.exists():
        try:
            found = _search_nested(json.loads(metadata_path.read_text()), SUBJECT_KEYS)
        except json.JSONDecodeError:
            found = None
        if found:
            return found[0], f"metadata.json:{found[1]}"
    if mvnx_subject and mvnx_subject.strip():
        return mvnx_subject.strip(), "mvnx:subject@label"
    raise ValueError(
        f"{sequence_dir.name}: no verified participant identifier in metadata.json or MVNX; "
        "refusing the sequence because a recording id would invalidate subject-level balancing"
    )


def _session_id(sequence: str, subject: str, token: str, part: int, parts: int) -> str:
    safe_subject = "".join(ch if ch.isalnum() else "-" for ch in str(subject))
    base = f"nymeria_{sequence}_{safe_subject}_{token}"
    return base if parts == 1 else f"{base}_p{part:02d}"


# ======================================================================================
# top-level conversion
# ======================================================================================


#: Detail keys that describe ONE stream or ONE part, not the dataset. Rolling them up to
#: the dataset manifest with "first wins" would report one placement's tracker label as if
#: it applied to all eight.
_PER_SESSION_DETAIL_KEYS = frozenset({"label", "source_span_sec"})


@dataclass
class DatasetOutput:
    name: str
    rate_hz: float
    channels: tuple[str, ...]
    directory: Path
    labels: dict[str, list[str]] = field(default_factory=dict)
    subjects: set = field(default_factory=set)
    seconds: float = 0.0
    detail: dict = field(default_factory=dict)


def _sequence_dirs(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        return []
    candidates = [
        path
        for path in sorted(raw_dir.iterdir())
        if path.is_dir()
        and not path.name.startswith(".")
        and (
            (path / "body").is_dir()
            or (path / "metadata.json").exists()
            or any((path / recording).is_dir() for recording in ARIA_STREAMS)
        )
    ]
    return candidates


def _load_body(
    sequence_dir: Path, body_source: str, angular_unit: str | None
) -> MvnxRecording | None:
    npz = sequence_dir / "body" / "xdata.npz"
    mvnx = sequence_dir / "body" / "xdata.mvnx"
    if body_source in ("auto", "npz") and npz.exists():
        try:
            return load_npz_recording(npz, angular_unit=angular_unit)
        except NpzSchemaUnknown as exc:
            if body_source == "npz":
                raise
            print(f"[nymeria] {exc}\n[nymeria] falling back to {mvnx.name}")
    if body_source == "npz":
        raise FileNotFoundError(f"{npz} not found")
    if mvnx.exists():
        return parse_mvnx(mvnx, angular_unit=angular_unit)
    return None


def convert(
    *,
    raw_dir: Path = DOWNLOADS,
    output_dir: Path = CORPUS_ROOT,
    sequences: int | None = None,
    streams: Sequence[str] = ("xsens", "aria"),
    max_hours_per_sequence: float | None = None,
    body_source: str = "auto",
    angular_unit: str | None = None,
    up_axis: str = "z",
    quat_order: str = "wxyz",
) -> bool:
    sequence_dirs = _sequence_dirs(raw_dir)
    if sequences is not None:
        sequence_dirs = sequence_dirs[:sequences]
    if not sequence_dirs:
        raise FileNotFoundError(
            f"no Nymeria sequence directories under {raw_dir}; run "
            "`python -m data.pretraining.nymeria.fetch --sequences <N>`"
        )

    outputs = {
        "xsens": DatasetOutput(
            XSENS_DATASET, XSENS_RATE_HZ, ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
            output_dir / XSENS_DATASET,
        ),
        "aria": DatasetOutput(
            ARIA_DATASET, ARIA_RATE_HZ, ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
            output_dir / ARIA_DATASET,
        ),
    }
    active = {key: outputs[key] for key in streams if key in outputs}
    if not active:
        raise ValueError(f"--streams must name some of {sorted(outputs)}, got {list(streams)}")
    for output in active.values():
        sessions = output.directory / "sessions"
        if sessions.exists():
            shutil.rmtree(sessions)
        sessions.mkdir(parents=True)

    stats: Counter = Counter()
    max_samples = None if max_hours_per_sequence is None else max_hours_per_sequence * 3600.0

    for position, sequence_dir in enumerate(sequence_dirs, start=1):
        sequence = sequence_dir.name
        signals: list[StreamSignal] = []
        mvnx_subject = None

        if "xsens" in active:
            try:
                recording = _load_body(sequence_dir, body_source, angular_unit)
            except (MvnxParseError, NpzSchemaUnknown) as exc:
                print(f"[nymeria] SKIP {sequence} body: {exc}")
                recording = None
                stats["body_unreadable"] += 1
            if recording is None:
                stats["body_missing"] += 1
            else:
                mvnx_subject = recording.subject_label
                outputs["xsens"].detail.setdefault("acceleration_source", recording.acceleration_source)
                outputs["xsens"].detail.setdefault("angular_unit", recording.angular_unit)
                outputs["xsens"].detail.setdefault("angular_unit_source", recording.angular_unit_source)
                for signal in mvnx_stream_signals(
                    recording, up_axis=up_axis, quat_order=quat_order
                ):
                    for start, stop in contiguous_blocks(
                        signal.times_sec, expected_rate_hz=signal.rate_hz
                    ):
                        signals.append(
                            StreamSignal(
                                token=signal.token,
                                acc_g=signal.acc_g[start:stop],
                                gyro_rads=None if signal.gyro_rads is None else signal.gyro_rads[start:stop],
                                times_sec=signal.times_sec[start:stop] - signal.times_sec[start],
                                rate_hz=signal.rate_hz,
                                detail=signal.detail,
                            )
                        )

        if "aria" in active:
            for recording_name, token in ARIA_STREAMS.items():
                vrs = sequence_dir / recording_name / "data" / "motion.vrs"
                if not vrs.exists():
                    stats[f"missing_{token}"] += 1
                    continue
                times, accel, gyro, label = read_aria_imu(vrs)
                signals.extend(aria_stream_signals(times, accel, gyro, token, label))

        if not signals:
            stats["sequences_without_signals"] += 1
            continue

        try:
            subject, subject_source = resolve_subject(sequence_dir, mvnx_subject)
        except ValueError as exc:
            print(f"[nymeria] SKIP {sequence}: {exc}")
            stats["sequences_without_verified_subject"] += 1
            continue
        by_token: dict[str, list[StreamSignal]] = {}
        for signal in signals:
            by_token.setdefault(signal.token, []).append(signal)

        for token, parts in by_token.items():
            key = "xsens" if token.startswith("xsens_") else "aria"
            output = active.get(key)
            if output is None:
                continue
            budget = max_samples
            for part_number, signal in enumerate(parts, start=1):
                if signal.gyro_rads is None:
                    # A grid stream has one channel mask. Allowing an acc-only recording into an
                    # IMU stream makes that mask depend on whichever session happens to be first.
                    stats["parts_without_gyro"] += 1
                    continue
                if budget is not None:
                    allowed = int(budget * signal.rate_hz)
                    if allowed <= 0:
                        break
                    if len(signal.acc_g) > allowed:
                        signal = StreamSignal(
                            token=signal.token,
                            acc_g=signal.acc_g[:allowed],
                            gyro_rads=None if signal.gyro_rads is None else signal.gyro_rads[:allowed],
                            times_sec=signal.times_sec[:allowed],
                            rate_hz=signal.rate_hz,
                            detail=signal.detail,
                        )
                try:
                    frame = session_frame(signal, subject)
                except ValueError:
                    stats["parts_below_one_window"] += 1
                    continue
                session_id = _session_id(sequence, subject, token, part_number, len(parts))
                write_session(output.directory / "sessions", session_id, frame)
                output.labels[session_id] = [UNLABELED]
                output.subjects.add(subject)
                seconds = len(frame) / signal.rate_hz
                output.seconds += seconds
                output.detail.setdefault("subject_source", subject_source)
                for detail_key, detail_value in signal.detail.items():
                    if detail_key in _PER_SESSION_DETAIL_KEYS:
                        continue  # varies per stream/part; "first wins" would misreport it
                    output.detail.setdefault(detail_key, detail_value)
                if budget is not None:
                    budget -= seconds
                stats[f"sessions_{key}"] += 1
        print(
            f"[nymeria] {position:04d}/{len(sequence_dirs)} {sequence}: "
            f"{len(signals)} stream parts, subject={subject} ({subject_source})"
        )

    written = {key: output for key, output in active.items() if output.labels}
    if not written:
        return False

    union: dict[str, list[str]] = {}
    for key, output in written.items():
        template = METADATA_TEMPLATES[output.name]
        metadata = json.loads(template.read_text())
        if float(metadata["sampling_rate_hz"]) != output.rate_hz:
            raise AssertionError(
                f"{template.name} declares {metadata['sampling_rate_hz']} Hz but the converter "
                f"wrote {output.rate_hz} Hz sessions"
            )
        metadata["num_subjects"] = len(output.subjects)
        (output.directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (output.directory / "labels.json").write_text(json.dumps(output.labels, indent=2) + "\n")
        (output.directory / "manifest.json").write_text(
            json.dumps(_dataset_manifest(output), indent=2) + "\n"
        )
        union.update(output.labels)

    # The union + module summary live in the module package in production, but follow an
    # explicit --output-dir so a test or scratch build stays hermetic.
    module_dir = DS_DIR if output_dir == CORPUS_ROOT else output_dir
    module_dir.mkdir(parents=True, exist_ok=True)
    module_dir.joinpath("labels.json").write_text(json.dumps(union, indent=2) + "\n")
    module_dir.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "module": "nymeria",
                "source": "https://www.projectaria.com/datasets/nymeria/",
                "licence": "CC BY-NC 4.0",
                "phase_a_only": True,
                "datasets": {output.name: _dataset_manifest(output) for output in written.values()},
                "num_sessions": sum(len(output.labels) for output in written.values()),
                "stats": dict(stats),
                "note": (
                    "Deliberately carries NO scalar sampling_rate_hz: the Xsens streams are "
                    "240 Hz and the Aria streams are 200 Hz, and build_grids.py reads one "
                    "rate per dataset directory. Point the grid builder at the per-rate "
                    f"dataset directories ({', '.join(sorted(o.name for o in written.values()))}), "
                    "not at this one."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[nymeria] complete: {dict(stats)}")
    for output in written.values():
        print(
            f"[nymeria]   {output.name}: {len(output.labels)} sessions, "
            f"{len(output.subjects)} subjects, {output.seconds / 3600:.2f} h @ {output.rate_hz:g} Hz"
        )
    return True


def _dataset_manifest(output: DatasetOutput) -> dict:
    return {
        "dataset_name": output.name,
        "source": "https://www.projectaria.com/datasets/nymeria/",
        "num_subjects": len(output.subjects),
        "num_sessions": len(output.labels),
        "sampling_rate_hz": output.rate_hz,
        "channels": list(output.channels),
        "unit": "g",
        "gravity_state": "present",
        "phase_a_only": True,
        "hours": round(output.seconds / 3600.0, 4),
        "gap_policy": f"split at gaps > {MAX_GAP_SECONDS:g}s; parts truncated to whole {WINDOW_SECONDS:g}s windows",
        "detail": output.detail,
        "note": "No activity labels; reserved marker __unlabeled__.",
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=DOWNLOADS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CORPUS_ROOT,
        help="corpus root the two dataset directories are written into "
             "(default: data/pretraining, where corpus_roots.dataset_root resolves them)",
    )
    parser.add_argument("--sequences", type=int, help="convert only the first N sequences")
    parser.add_argument("--streams", nargs="+", default=["xsens", "aria"], choices=["xsens", "aria"])
    parser.add_argument("--max-hours-per-sequence", type=float, default=None)
    parser.add_argument("--body-source", choices=["auto", "mvnx", "npz"], default="auto")
    parser.add_argument(
        "--angular-unit",
        choices=["rad", "deg"],
        default=None,
        help="override the MVNX angular-velocity unit instead of using the file's declaration",
    )
    parser.add_argument("--up-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--quat-order", choices=["wxyz", "xyzw"], default="wxyz")
    parser.add_argument(
        "--inspect-npz",
        type=Path,
        help="print the arrays in one body/xdata.npz and exit (documents the real schema)",
    )
    args = parser.parse_args(argv)

    if args.inspect_npz is not None:
        print(json.dumps(inspect_npz(args.inspect_npz), indent=2))
        return

    ok = convert(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        sequences=args.sequences,
        streams=tuple(args.streams),
        max_hours_per_sequence=args.max_hours_per_sequence,
        body_source=args.body_source,
        angular_unit=args.angular_unit,
        up_axis=args.up_axis,
        quat_order=args.quat_order,
    )
    if not ok:
        print("no Nymeria sessions were produced", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
