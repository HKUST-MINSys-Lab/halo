"""Convert HARMES (Hand Activity Recognition from Multimodal Egocentric Sensing)
right-wrist IMU subset to the HALO session format.

Source: Zenodo 19425719 (HARMES-RAW.zip, CC-BY-4.0). See fetch.py — we pull ONLY the
right-wrist WearOS recordings + event logs out of the 19.9 GB zip via HTTP range
requests, skipping 18.9 GB of audio and the left-wrist Puck.js files.

Per recording folder `downloads/HARMES-RAW/<PP>/<REC>/`:
  * `recording_<date>.csv` — RIGHT wrist, WearOS smartwatch. Columns:
        timestamp(epoch ms) | acc_{x,y,z}(m/s^2) | gyro_{x,y,z}(rad/s) | ts_sync
    Verified on real data: at-rest |acc| = 9.81 m/s^2 (1.013 g); gyro p99 = 5.6 rad/s
    (already canonical rad/s). Accelerometer is m/s^2 -> classified ACC_UNIT_MS2 (the
    corpus rescales acc to g at windowing; gyro is never touched).
  * `<epoch>.csv` — the start/end EVENT LOG: columns Time(epoch s) | Type(start|end) |
    Description(activity). This is the labelled ground truth (the `_TASKLIST.csv` is only
    the planned protocol order, no timestamps, and is not downloaded).

Why RIGHT wrist only (left Puck.js dropped): the left `*_merged.csv` accelerometer is
cleanly 8192 counts/g (LSM6DS3 +/-4g, cross-calibrated against the right wrist during
still segments via the shared `ts_sync` clock), BUT its gyroscope has an undocumented
full-scale and saturates at +/-32764 (int16 rail) on most hand motion — unrecoverable
without the authors' sensor config. A stream needs correct 6-axis; we do not ship a
corrupted gyro. HARMES's value to us is its 15 fine-grained hand-ADL LABELS, all of
which the right wrist carries. (Same discipline as mhealth's degenerate gyro / recgym.)

Timeline hygiene: WearOS timestamps arrive slightly out of order (p1 dt ~= -16 ms) and
have ~149 dropout gaps (>250 ms) per recording. Per activity segment we sort by epoch,
drop duplicate timestamps, SPLIT at gaps > 250 ms, and resample each continuous run to a
uniform 50 Hz grid (the filterbank assumes uniform sampling at metadata.sampling_rate_hz).

Labels (15): brushing_teeth / drinking / vacuum_cleaning map to existing canonicals; the
other 12 are added as NEW canonicals (user decision "add all as new canonical").
"""

from __future__ import annotations

import glob
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads"
NATIVE_RATE = 50.0          # WearOS right wrist, resampled to a uniform grid
GAP_MS = 250.0              # split a segment at any inter-sample gap larger than this
MIN_RUN_SEC = 6.0          # drop continuous runs shorter than one corpus window

LABEL_MAP = {
    # existing canonicals
    "Brushing teeth": "brushing_teeth",
    "Drinking": "drinking",
    "Vacuum Cleaning": "vacuum_cleaning",
    # --- new canonical labels (fine-grained hand ADLs) ---
    "Floor cleaning": "floor_cleaning",
    "Window cleaning": "window_cleaning",
    "Making tea": "making_tea",
    "Washing dishes": "washing_dishes",
    "Putting away the dishes": "putting_away_dishes",
    "Cutting vegetables": "cutting_vegetables",
    "Cleaning table": "cleaning_table",
    "Watering plant": "watering_plants",
    "Cleaning out the dishwasher": "emptying_dishwasher",
    "Washing hands": "washing_hands",
    "Cream hands": "applying_hand_cream",
    "Disinfecting hands": "disinfecting_hands",
}

COMMENT_LABEL_ALIASES = {
    "cleaning the cupboar": "cleaning the cupboard",
    "cleaning the door": "cleaning door",
    "building with lego": "build with lego",
    "pet the dog": "petting the dog",
    "play with the dog": "playing with the dog",
    "working in the gartden": "working in the garden",
}


def _comment_marker(description: str) -> tuple[str, str] | None:
    """Parse visit-four ``'<activity> start/end'`` comment markers."""
    match = re.fullmatch(r"\s*(.*?)\s+(start|end)\s*", description, flags=re.IGNORECASE)
    if match is None:
        return None
    name = re.sub(r"\s+", " ", match.group(1).strip().lower())
    return COMMENT_LABEL_ALIASES.get(name, name), match.group(2).lower()


def _open_label(name: str) -> str:
    """Stable readable identifier for a clearly delimited free-living activity."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def event_segments(event_csv: str, wrist_epoch0: float):
    """Yield (canonical_label, t0_epoch_s, t1_epoch_s) from a start/end event log.

    HARMES has a per-recording clock offset: some event logs differ from the WearOS wrist clock by
    exactly one hour (DST/timezone) behind the WearOS wrist clock, so the labels would not
    overlap the IMU at all. We correct by snapping the event->wrist start offset to the
    NEAREST WHOLE HOUR (0 for the well-aligned majority, +/-3600 for the offenders). Only
    whole-hour corrections are applied; anything else is a genuine mismatch left as-is.
    """
    e = pd.read_csv(event_csv)
    e["Description"] = e["Description"].astype(str).str.strip()
    e["Type"] = e["Type"].astype(str).str.strip()
    off = round((wrist_epoch0 - float(e["Time"].min())) / 3600.0) * 3600.0
    stack: dict[str, float] = {}
    for _, row in e.iterrows():
        d, ty, t = row["Description"], row["Type"], float(row["Time"]) + off
        if d == "RECORD" or d.lower().startswith("deleted"):
            continue
        if ty == "comment":
            marker = _comment_marker(d)
            if marker is None:
                continue
            d, ty = marker
        if ty == "start":
            stack[d] = t
        elif ty == "end" and d in stack:
            t0 = stack.pop(d)
            canon = LABEL_MAP.get(d, LABEL_MAP.get(d.title()))
            if canon is None and row["Type"].strip() == "comment":
                canon = _open_label(d)
            if canon is not None and t > t0:
                yield canon, t0, t


def resample_runs(epoch_s: np.ndarray, sig: np.ndarray):
    """Sort, dedup, split at >GAP_MS gaps, resample each run to a uniform 50 Hz grid.

    `sig` is (N, 6) acc+gyro. Yields (T, 6) uniformly-sampled arrays, one per run.
    """
    order = np.argsort(epoch_s, kind="stable")
    epoch_s, sig = epoch_s[order], sig[order]
    keep = np.concatenate([[True], np.diff(epoch_s) > 0])   # strictly increasing for interp
    epoch_s, sig = epoch_s[keep], sig[keep]
    if len(epoch_s) < 2:
        return
    gaps = np.flatnonzero(np.diff(epoch_s) > GAP_MS / 1000.0) + 1
    for lo, hi in zip(np.concatenate([[0], gaps]), np.concatenate([gaps, [len(epoch_s)]])):
        t = epoch_s[lo:hi]
        if t[-1] - t[0] < MIN_RUN_SEC:
            continue
        grid = np.arange(t[0], t[-1], 1.0 / NATIVE_RATE)
        out = np.empty((len(grid), sig.shape[1]), dtype=np.float32)
        for c in range(sig.shape[1]):
            out[:, c] = np.interp(grid, t, sig[lo:hi, c])
        yield out


def create_manifest() -> dict:
    chans = [{"name": c, "stream": "wrist",
              "description": (f"{'accelerometer' if 'acc' in c else 'gyroscope'} {c[-1]}-axis of a "
              f"WearOS smartwatch on the right wrist in "
                              f"{'m/s^2 (gravity present)' if 'acc' in c else 'rad/s'}"),
              "sampling_rate_hz": NATIVE_RATE}
             for c in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")]
    return {
        "dataset_name": "HARMES (right-wrist IMU)",
        "description": ("Hand Activity Recognition from Multimodal Egocentric Sensing: right-wrist "
                        "WearOS IMU, 20 participants, 15 fine-grained kitchen/bathroom hand ADLs "
                        "(cleaning, cooking, hygiene) at 50 Hz. Accelerometer m/s^2 (gravity present); "
                        "gyroscope rad/s. Left-wrist Puck.js excluded (unrecoverable gyro scale)."),
        "channels": chans,
    }


def main() -> None:
    sessions_dir = HERE / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True)

    labels: dict[str, list[str]] = {}
    dist: Counter = Counter()
    n_sess = 0
    rec_dirs = sorted({str(Path(p).parent) for p in
                       glob.glob(str(RAW / "HARMES-RAW" / "*" / "*" / "recording_*.csv"))})
    for rec_dir in rec_dirs:
        d = Path(rec_dir)
        subject = d.parent.name                     # participant PP (01..20)
        rec = d.name                                # recording id (e.g. 0102)
        right = glob.glob(str(d / "recording_*.csv"))
        event = [p for p in glob.glob(str(d / "*.csv"))
                 if Path(p).stem.isdigit()]          # numeric-name event log
        if not right or not event:
            continue
        R = pd.read_csv(right[0])
        epoch = R["timestamp"].to_numpy(np.float64) / 1000.0        # epoch ms -> s
        sig = R[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].to_numpy(np.float32)
        for canon, t0, t1 in event_segments(event[0], float(epoch.min())):
            m = (epoch >= t0) & (epoch < t1)
            if m.sum() < 2:
                continue
            for r, run in enumerate(resample_runs(epoch[m], sig[m])):
                sid = f"pp{subject}_{rec}_{canon}_seg{n_sess}_run{r}"
                out = pd.DataFrame({
                    "timestamp_sec": np.arange(len(run), dtype=np.float64) / NATIVE_RATE,
                    **{f"acc_{a}": run[:, i] for i, a in enumerate("xyz")},
                    **{f"gyro_{a}": run[:, 3 + i] for i, a in enumerate("xyz")},
                    "subject": subject,
                })
                seg_dir = sessions_dir / sid
                seg_dir.mkdir()
                out.to_parquet(seg_dir / "data.parquet", index=False)
                labels[sid] = [canon]
                dist[canon] += 1
                n_sess += 1

    (HERE / "labels.json").write_text(json.dumps(labels, indent=2))
    (HERE / "metadata.json").write_text(json.dumps(
        {"dataset": "harmes", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False}, indent=2))
    (HERE / "manifest.json").write_text(json.dumps(create_manifest(), indent=2))
    print(f"harmes: {n_sess} sessions from {len(rec_dirs)} recordings; "
          f"subjects={len({s.split('_')[0] for s in labels})}; sessions/label {dict(dist)}")


if __name__ == "__main__":
    main()
