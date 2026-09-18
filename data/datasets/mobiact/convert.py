"""Convert the official MobiAct v2 annotated CSV release into HALO sessions.

This converter intentionally supports one source format: ``Annotated Data/<CODE>/*_annotated.csv``
from MobiAct v2. MobiFall's separate text files are a different release and must not silently
masquerade as MobiAct. Use :mod:`data.datasets.mobiact.setup` for the full preparation workflow.

Android acceleration remains in m/s^2 and angular velocity in rad/s. The shared grid assembler
converts accelerometer channels to g exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data.scripts.assembly.assemble import resample_signal

HERE = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = HERE / "downloads"
TARGET_RATE_HZ = 50.0
RELEASE_ID = "mobiact-v2-annotated"
PROTOCOL_VERSION = "mobiact-prospective-v1-20260918"

ACTIVITIES: dict[str, str] = {
    "STD": "standing", "WAL": "walking", "JOG": "jogging", "JUM": "jumping",
    "STU": "stairs_up", "STN": "stairs_down", "SCH": "sitting_chair",
    "CHU": "chair_up", "SIT": "sitting", "CSI": "car_step_in",
    "CSO": "car_step_out", "LYI": "lying", "FOL": "fall_forward",
    "FKL": "fall_forward_knees", "BSC": "fall_backward_sitting", "SDL": "fall_sideways",
}
FALL_CODES = frozenset({"FOL", "FKL", "BSC", "SDL"})
TRANSITION_CODES = frozenset({"SCH", "CHU", "CSI", "CSO"})
REQUIRED_COLUMNS = ("rel_time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
FILENAME = re.compile(
    r"^(?P<activity>[A-Za-z]+)_(?P<subject>\d+)_(?P<trial>\d+)_annotated\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Trial:
    path: Path
    activity_code: str
    subject: int
    trial: int

    @property
    def recording_id(self) -> str:
        return f"{self.activity_code}_s{self.subject:03d}_t{self.trial:02d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_annotated_root(raw_root: Path) -> Path:
    """Find exactly one populated ``Annotated Data`` directory below ``raw_root``."""
    candidates = [
        path for path in Path(raw_root).rglob("*")
        if path.is_dir() and path.name.casefold() == "annotated data"
        and any(path.rglob("*_annotated.csv"))
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one populated 'Annotated Data' directory below {raw_root}, "
            f"found {len(candidates)}. MobiFall text files are not MobiAct v2."
        )
    return candidates[0]


def discover_trials(annotated_root: Path) -> list[Trial]:
    trials: list[Trial] = []
    unknown: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    for path in sorted(Path(annotated_root).rglob("*.csv")):
        match = FILENAME.match(path.name)
        if match is None or match.group("activity").upper() not in ACTIVITIES:
            unknown.append(str(path.relative_to(annotated_root)))
            continue
        trial = Trial(
            path, match.group("activity").upper(),
            int(match.group("subject")), int(match.group("trial")),
        )
        key = (trial.activity_code, trial.subject, trial.trial)
        if key in seen:
            raise ValueError(f"duplicate MobiAct trial identity {key}: {path}")
        seen.add(key)
        trials.append(trial)
    if unknown:
        raise ValueError(
            f"unrecognized CSV files in MobiAct release ({len(unknown)}): {', '.join(unknown[:5])}"
        )
    if not trials:
        raise ValueError(f"no MobiAct v2 annotated trials found in {annotated_root}")
    return trials


def _split_runs(time_s: np.ndarray) -> list[tuple[int, int]]:
    delta = np.diff(time_s)
    positive = delta[delta > 0]
    if not len(positive):
        return []
    threshold = max(0.25, 10.0 * float(np.median(positive)))
    boundaries = np.r_[0, np.flatnonzero(delta > threshold) + 1, len(time_s)]
    return [(int(start), int(stop)) for start, stop in zip(boundaries[:-1], boundaries[1:])]


def load_trial(path: Path) -> tuple[list[pd.DataFrame], dict]:
    """Validate and anti-alias one annotated trial onto a uniform 50 Hz clock."""
    source = pd.read_csv(path)
    source.columns = [str(column).strip().lower() for column in source.columns]
    missing = sorted(set(REQUIRED_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    numeric = source.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{path}: required sensor columns contain non-finite values")
    numeric = numeric.sort_values("rel_time", kind="stable").drop_duplicates("rel_time")
    # Pandas may expose a read-only Arrow/Copy-on-Write view; the origin normalization below is
    # deliberately local and must never mutate the source frame.
    time_s = numeric["rel_time"].to_numpy(np.float64, copy=True)
    time_s -= time_s[0]
    if len(time_s) < 2 or np.any(np.diff(time_s) <= 0):
        raise ValueError(f"{path}: relative time is not strictly increasing")

    signals = numeric.loc[:, REQUIRED_COLUMNS[1:]].to_numpy(np.float64)
    segments: list[pd.DataFrame] = []
    source_rates: list[float] = []
    for start, stop in _split_runs(time_s):
        local_t = time_s[start:stop] - time_s[start]
        if len(local_t) < 10 or local_t[-1] <= 0:
            continue
        source_rate = 1.0 / float(np.median(np.diff(local_t)))
        if not 5.0 <= source_rate <= 1000.0:
            raise ValueError(f"{path}: implausible source sampling rate {source_rate:.3f} Hz")
        source_grid = np.arange(int(np.floor(local_t[-1] * source_rate)) + 1) / source_rate
        regular = np.column_stack([
            np.interp(source_grid, local_t, signals[start:stop, column])
            for column in range(signals.shape[1])
        ])
        sampled = resample_signal(regular, source_rate, TARGET_RATE_HZ)
        if len(sampled) < 10 or not np.isfinite(sampled).all():
            continue
        frame = pd.DataFrame(sampled, columns=REQUIRED_COLUMNS[1:])
        frame.insert(0, "timestamp_sec", np.arange(len(frame), dtype=np.float64) / TARGET_RATE_HZ)
        segments.append(frame)
        source_rates.append(source_rate)
    if not segments:
        raise ValueError(f"{path}: no valid contiguous sensor segment")
    return segments, {
        "source_rows": int(len(numeric)),
        "source_rate_hz_median": float(np.median(source_rates)),
        "segments": len(segments),
        "duration_seconds": float(sum(len(frame) for frame in segments) / TARGET_RATE_HZ),
    }


def _stable_subject_split(
    subjects: Iterable[int], *, seed: int, query_fraction: float,
) -> tuple[list[int], list[int]]:
    if not 0.0 < query_fraction < 0.5:
        raise ValueError("query_fraction must be between zero and 0.5")
    ranked = sorted(
        set(map(int, subjects)),
        key=lambda subject: hashlib.sha256(f"{seed}:{subject}".encode()).digest(),
    )
    n_query = max(1, int(round(len(ranked) * query_fraction)))
    return sorted(ranked[n_query:]), sorted(ranked[:n_query])


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def convert(
    raw_root: Path = DEFAULT_RAW_ROOT,
    *, split_seed: int = 20260918,
    query_fraction: float = 0.30,
    archive_sha256: str | None = None,
) -> dict:
    annotated_root = find_annotated_root(Path(raw_root))
    trials = discover_trials(annotated_root)
    staging = HERE / "sessions.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    labels: dict[str, list[str]] = {}
    recordings: dict[str, str] = {}
    inventory: list[dict] = []
    label_subjects: dict[str, set[int]] = defaultdict(set)
    try:
        for index, trial in enumerate(trials, 1):
            segments, stats = load_trial(trial.path)
            activity = ACTIVITIES[trial.activity_code]
            label_subjects[activity].add(trial.subject)
            for segment_index, frame in enumerate(segments):
                session_id = trial.recording_id
                if len(segments) > 1:
                    session_id += f"_segment_{segment_index:02d}"
                frame["subject"] = f"s{trial.subject:03d}"
                destination = staging / session_id
                destination.mkdir()
                frame.to_parquet(destination / "data.parquet", index=False)
                labels[session_id] = [activity]
                recordings[session_id] = trial.recording_id
            inventory.append({
                "recording_id": trial.recording_id, "activity_code": trial.activity_code,
                "activity": activity, "subject": f"s{trial.subject:03d}", "trial": trial.trial,
                "source_file": str(trial.path.relative_to(annotated_root.parent)),
                "source_sha256": _sha256(trial.path), **stats,
            })
            if index % 250 == 0:
                print(f"[mobiact] converted {index}/{len(trials)} trials", flush=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    reference, query = _stable_subject_split(
        (trial.subject for trial in trials), seed=split_seed, query_fraction=query_fraction,
    )
    reference_set, query_set = set(reference), set(query)
    uncovered = sorted(
        label for label, subjects in label_subjects.items()
        if not (subjects & reference_set and subjects & query_set)
    )
    if uncovered:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(
            "frozen subject split lacks reference/query coverage for labels "
            f"{uncovered}; choose another split seed before evaluation"
        )

    sessions = HERE / "sessions"
    if sessions.exists():
        shutil.rmtree(sessions)
    staging.rename(sessions)
    emitted_labels = sorted(set(value[0] for value in labels.values()))
    subjects = sorted({trial.subject for trial in trials})
    activity_counts = Counter(row["activity"] for row in inventory)

    manifest = {
        "dataset_name": "MobiAct v2", "release_id": RELEASE_ID,
        "description": (
            "MobiAct v2 annotated smartphone trials: ADLs and simulated falls, Samsung Galaxy S3 "
            "in a freely selected trouser pocket orientation."
        ),
        "source": "https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/",
        "citation": (
            "Vavoulas G, Chatzaki C, Malliotakis T, Pediaditis M, Tsiknakis M. The MobiAct "
            "Dataset: Recognition of Activities of Daily Living using Smartphones. ICT4AWE 2016."
        ),
        "license": "MobiAct database usage agreement; redistribution is not implied",
        "sampling_rate_hz": TARGET_RATE_HZ,
        "channels": [
            {"name": name, "sampling_rate_hz": TARGET_RATE_HZ,
             "unit": "m/s^2" if name.startswith("acc_") else "rad/s"}
            for name in REQUIRED_COLUMNS[1:]
        ],
        "subjects": len(subjects), "trials": len(inventory), "activities": emitted_labels,
        "placement": "freely oriented trouser pocket", "gravity_state": "present",
    }
    metadata = {
        "dataset": "mobiact", "display_name": "MobiAct v2", "release_id": RELEASE_ID,
        "sampling_rate_hz": TARGET_RATE_HZ, "pre_windowed": False, "full_windows_only": True,
        "num_subjects": len(subjects), "num_trials": len(inventory), "num_sessions": len(labels),
        "hours": sum(float(row["duration_seconds"]) for row in inventory) / 3600.0,
        "activity_trial_counts": dict(sorted(activity_counts.items())),
        "archive_sha256": archive_sha256,
    }
    protocol = {
        "protocol_version": PROTOCOL_VERSION, "release_id": RELEASE_ID,
        "status": "converted_needs_grid_finalization", "split_seed": split_seed,
        "query_fraction": query_fraction,
        "reference_subjects": [f"s{subject:03d}" for subject in reference],
        "query_subjects": [f"s{subject:03d}" for subject in query],
        "candidate_labels": emitted_labels, "candidate_labels_by_window_seconds": {},
        "panels": {
            "falls": sorted(ACTIVITIES[code] for code in FALL_CODES if ACTIVITIES[code] in emitted_labels),
            "transitions": sorted(
                ACTIVITIES[code] for code in TRANSITION_CODES if ACTIVITIES[code] in emitted_labels
            ),
            "ordinary_adl": sorted(
                label for code, label in ACTIVITIES.items()
                if code not in FALL_CODES | TRANSITION_CODES and label in emitted_labels
            ),
        },
    }
    artifacts = (
        ("labels.json", labels), ("recordings.json", recordings), ("manifest.json", manifest),
        ("metadata.json", metadata), ("eval_protocol.json", protocol),
        ("eval_labels.json", {
            "dataset": "mobiact", "protocol_version": PROTOCOL_VERSION,
            "source": "Native MobiAct v2 activity names, frozen before model scoring.",
            "labels": emitted_labels,
        }),
        ("source_inventory.json", {"release_id": RELEASE_ID, "trials": inventory}),
    )
    for filename, value in artifacts:
        _write_json(HERE / filename, value)
    print(
        f"[mobiact] wrote {len(labels)} sessions from {len(inventory)} trials, "
        f"{len(subjects)} subjects, {len(emitted_labels)} labels",
        flush=True,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--split-seed", type=int, default=20260918)
    parser.add_argument("--query-fraction", type=float, default=0.30)
    parser.add_argument("--archive-sha256", default=None)
    args = parser.parse_args()
    convert(args.raw_root, split_seed=args.split_seed, query_fraction=args.query_fraction,
            archive_sha256=args.archive_sha256)


if __name__ == "__main__":
    main()
