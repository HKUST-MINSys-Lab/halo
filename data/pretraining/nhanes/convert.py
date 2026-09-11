"""Convert a bounded NHANES PAX80_G subset into Phase-A-only HALO sessions.

NHANES 2011-2012 uses an ActiGraph GT3X+ on the non-dominant wrist, sampled at
80 Hz. Values are calibrated triaxial acceleration in g and include gravity.
Each participant archive contains hourly CSV files and a QC interval CSV.

There are no activity annotations or diaries. Sessions therefore carry the
reserved ``__unlabeled__`` marker. Pipeline A may use them for self-supervised
objectives, but label vocabulary construction, validation probes, and the
Phase-B evidence bank must exclude that marker.

The released QC file flags sensor malfunction (~0.26% of the release), not
off-body time, and no non-wear detector is applied here. The result is
free-living as recorded: 43% of windows sit below 0.003 g of motion and 13.3%
are byte-identical repeats of a motionless posture. The exact repeats are
dropped by ``data.scripts.scan_duplicates``; the rest is genuine sedentary and
sleep time and is kept on purpose.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import tarfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.pretraining.corpus_plan import PRETRAIN_WINDOW_SECONDS


DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"
RATE_HZ = 80.0
WINDOW_SECONDS = PRETRAIN_WINDOW_SECONDS
WINDOW_SAMPLES = int(RATE_HZ * WINDOW_SECONDS)
UNLABELED = "__unlabeled__"
#: Max-axis standard deviation, in g, below which one JEPA source window is indistinguishable
#: from a motionless device. Measured as the NHANES noise floor in the 2026-07 corpus audit.
STILL_G = 0.003
#: Share of a subject's hour budget deliberately drawn from the low-motion end, so sleep and
#: sedentary posture stay represented instead of being selected away.
STILL_FRACTION = 1.0 / 3.0
#: Default hours per participant. Twelve spans a day and a night while keeping breadth of
#: SUBJECTS the dominant axis: under the sampler's per-subject n^0.5 tempering, 3,000 people
#: at 12 h is worth far more than 500 people at a full week.
DEFAULT_MAX_HOURS = 12
OUTPUT_COLUMNS = ("acc_x", "acc_y", "acc_z")
_STAMP = re.compile(r"\.(2000-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-000-P0000\.sensor\.csv$")


def _member_start(name: str) -> datetime:
    match = _STAMP.search(name)
    if not match:
        raise ValueError(f"cannot parse NHANES timestamp from {name}")
    return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")


def _read_qc(archive: tarfile.TarFile, members: list[tarfile.TarInfo]) -> pd.DataFrame:
    log = next((member for member in members if member.name.endswith("_Logs.csv")), None)
    if log is None:
        return pd.DataFrame()
    handle = archive.extractfile(log)
    if handle is None:
        return pd.DataFrame()
    frame = pd.read_csv(handle)
    return frame.dropna(how="all")


def _qc_intervals(qc: pd.DataFrame, first_day: datetime) -> list[tuple[datetime, datetime]]:
    result = []
    if qc.empty:
        return result
    day0 = datetime.combine(first_day.date(), datetime.min.time())
    for row in qc.itertuples(index=False):
        try:
            day = int(row.DAY_OF_DATA)
            start_time = pd.to_timedelta(str(row.START_TIME))
            end_time = pd.to_timedelta(str(row.END_TIME))
        except (TypeError, ValueError):
            continue
        start = day0 + timedelta(days=day - 1) + start_time.to_pytimedelta()
        end = day0 + timedelta(days=day - 1) + end_time.to_pytimedelta()
        if end >= start:
            result.append((start, end))
    return result


def _evenly_spaced(members: list[tarfile.TarInfo], max_hours: int | None) -> list[tarfile.TarInfo]:
    if max_hours is None or len(members) <= max_hours:
        return members
    # The first and last files are usually partial hours. Avoid spending a small
    # bounded budget on those endpoints when enough full interior hours exist.
    start, stop = (1, len(members) - 2) if len(members) - 2 >= max_hours else (0, len(members) - 1)
    indices = np.unique(np.round(np.linspace(start, stop, max_hours)).astype(int))
    return [members[int(index)] for index in indices]


def motion_score(xyz: np.ndarray) -> float:
    """Fraction of JEPA source windows in ``xyz`` that carry more than sensor noise.

    A window counts as moving when its largest per-axis standard deviation exceeds
    ``STILL_G``. That threshold is the release's own noise floor, not a tuned constant: the
    2026-07 audit measured 43% of NHANES windows below 0.003 g of max-axis std, which is the
    signature of an ActiGraph resting against a mattress rather than of human movement.

    Pure and cheap so hour selection can be unit-tested without tar archives.
    """
    usable = (len(xyz) // WINDOW_SAMPLES) * WINDOW_SAMPLES
    if usable < WINDOW_SAMPLES:
        return 0.0
    windows = xyz[:usable].reshape(-1, WINDOW_SAMPLES, xyz.shape[1])
    spread = np.nanstd(windows, axis=1).max(axis=1)
    return float(np.mean(spread > STILL_G))


def select_hours(
    scores: "list[float] | np.ndarray",
    max_hours: int | None,
    still_fraction: float = STILL_FRACTION,
) -> list[int]:
    """Choose which hourly files to keep, given each one's :func:`motion_score`.

    Free-living wrist data is mostly stillness, and a masked-prediction objective learns
    nothing from predicting a motionless window from its motionless neighbours. But an
    all-motion corpus is not free-living either, and sleep and sedentary posture are exactly
    the content the DC/gravity part of the frontend reads. So the budget is split: the top
    ``1 - still_fraction`` of hours by motion, plus a ``still_fraction`` minority drawn from
    the low-motion end, both kept in chronological order.

    Returns indices into ``scores``, sorted, so the caller preserves time order.
    """
    total = len(scores)
    if max_hours is None or total <= max_hours:
        return list(range(total))
    order = list(np.argsort(np.asarray(scores, dtype=float), kind="stable"))
    n_still = min(int(round(max_hours * still_fraction)), max_hours)
    n_moving = max_hours - n_still
    chosen = set(order[:n_still]) | set(order[total - n_moving:] if n_moving else [])
    # Rounding collisions (an hour landing in both ends of a short record) can leave the set
    # short of budget; top it up from the unused middle, most-moving first.
    if len(chosen) < max_hours:
        for index in reversed(order):
            if len(chosen) >= max_hours:
                break
            chosen.add(index)
    return sorted(chosen)


def _complete_blocks(
    frame: pd.DataFrame,
    intervals: list[tuple[datetime, datetime]],
) -> list[np.ndarray]:
    timestamps = pd.to_datetime(
        frame["HEADER_TIMESTAMP"],
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce",
    )
    xyz = frame[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    good = np.isfinite(xyz).all(axis=1) & timestamps.notna().to_numpy()
    for start, end in intervals:
        good &= ~((timestamps >= start) & (timestamps <= end)).to_numpy()

    # Sensor gaps are boundaries even when no published QC interval covers them.
    stamp_ns = timestamps.astype("int64", copy=False).to_numpy()
    dt = np.diff(stamp_ns) / 1e9
    boundaries = np.flatnonzero(
        (~good[1:]) | (~good[:-1]) | (dt > 2.5 / RATE_HZ) | (dt <= 0.0)
    ) + 1
    bounds = np.r_[0, boundaries, len(frame)]
    blocks = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if not good[start:end].all():
            continue
        usable = ((end - start) // WINDOW_SAMPLES) * WINDOW_SAMPLES
        if usable:
            blocks.append(xyz[start : start + usable])
    return blocks


def convert(
    raw_dir: Path = DOWNLOADS,
    output_dir: Path = DS_DIR,
    limit_subjects: int | None = None,
    max_hours_per_subject: int | None = DEFAULT_MAX_HOURS,
    selection: str = "motion",
) -> bool:
    archives = sorted(raw_dir.glob("*.tar.bz2"))
    if limit_subjects is not None:
        archives = archives[:limit_subjects]
    if not archives:
        raise FileNotFoundError(
            f"no participant archives under {raw_dir}; run "
            "`python -m data.pretraining.nhanes.fetch --subjects <N>`"
        )

    sessions_dir = output_dir / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True)
    labels: dict[str, list[str]] = {}
    stats: Counter = Counter()

    for archive_i, archive_path in enumerate(archives, start=1):
        seqn = archive_path.name.split(".", 1)[0]
        with tarfile.open(archive_path, mode="r:bz2") as archive:
            members = archive.getmembers()
            sensor_members = sorted(
                (member for member in members if member.name.endswith(".sensor.csv")),
                key=lambda member: _member_start(member.name),
            )
            if not sensor_members:
                stats["subjects_without_sensor_files"] += 1
                continue
            qc = _read_qc(archive, members)
            intervals = _qc_intervals(qc, _member_start(sensor_members[0].name))
            if selection == "evenly_spaced":
                chosen = _evenly_spaced(sensor_members, max_hours_per_subject)
                hour_blocks = {}
            else:
                # Motion-aware selection has to read every candidate hour to score it, so the
                # decoded blocks are cached and reused rather than decompressed twice.
                hour_blocks = {}
                scores = []
                for member in sensor_members:
                    handle = archive.extractfile(member)
                    if handle is None:
                        stats["unreadable_hour"] += 1
                        hour_blocks[member.name] = []
                        scores.append(-1.0)
                        continue
                    frame = pd.read_csv(
                        io.BytesIO(handle.read()),
                        usecols=["HEADER_TIMESTAMP", "X", "Y", "Z"],
                    )
                    blocks_here = _complete_blocks(frame, intervals)
                    hour_blocks[member.name] = blocks_here
                    scores.append(
                        motion_score(np.concatenate(blocks_here, axis=0))
                        if blocks_here else -1.0
                    )
                keep = select_hours(scores, max_hours_per_subject)
                chosen = [sensor_members[index] for index in keep]
                stats["hours_scored"] += len(sensor_members)

            blocks: list[np.ndarray] = []
            for member in chosen:
                if member.name in hour_blocks:
                    blocks.extend(hour_blocks[member.name])
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    stats["unreadable_hour"] += 1
                    continue
                frame = pd.read_csv(
                    io.BytesIO(handle.read()),
                    usecols=["HEADER_TIMESTAMP", "X", "Y", "Z"],
                )
                blocks.extend(_complete_blocks(frame, intervals))

        if not blocks:
            stats["subjects_without_complete_blocks"] += 1
            continue
        data = np.concatenate(blocks, axis=0)
        out = pd.DataFrame(data, columns=OUTPUT_COLUMNS)
        out.insert(0, "timestamp_sec", np.arange(len(out), dtype=np.float64) / RATE_HZ)
        out["subject"] = seqn
        session_id = f"nhanes_{seqn}_watch_wrist"
        destination = sessions_dir / session_id
        destination.mkdir()
        out.to_parquet(destination / "data.parquet", index=False)
        labels[session_id] = [UNLABELED]
        stats["windows"] += len(out) // WINDOW_SAMPLES
        stats["subjects"] += 1
        print(
            f"[nhanes] {archive_i:03d}/{len(archives)} {seqn}: "
            f"{len(out) / RATE_HZ / 3600:.2f} h, {len(out) // WINDOW_SAMPLES} windows"
        )

    if not labels:
        return False
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "NHANES PAX80_G (bounded subset)",
                "source": "https://ftp.cdc.gov/pub/pax_g/",
                "num_subjects": stats["subjects"],
                "sampling_rate_hz": RATE_HZ,
                "channels": list(OUTPUT_COLUMNS),
                "unit": "g",
                "gravity_state": "present",
                "phase_a_only": True,
                "max_hours_per_subject": max_hours_per_subject,
                "hour_selection": selection,
                "still_threshold_g": STILL_G,
                "still_fraction": STILL_FRACTION,
                "note": "No activity labels; reserved marker __unlabeled__.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[nhanes] complete: {dict(stats)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DOWNLOADS)
    parser.add_argument("--output-dir", type=Path, default=DS_DIR)
    parser.add_argument("--limit-subjects", type=int)
    parser.add_argument("--max-hours-per-subject", type=int, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--selection", choices=("motion", "evenly_spaced"), default="motion",
                        help="How to spend the hour budget. 'motion' ranks every candidate hour "
                             "by moving-window fraction and keeps the most active plus a "
                             "one-third still minority; 'evenly_spaced' takes an unbiased spread "
                             "and decompresses only the chosen hours (much cheaper, more stillness).")
    args = parser.parse_args()
    if not convert(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        limit_subjects=args.limit_subjects,
        max_hours_per_subject=args.max_hours_per_subject,
        selection=args.selection,
    ):
        raise SystemExit("no NHANES sessions were produced")


if __name__ == "__main__":
    main()
