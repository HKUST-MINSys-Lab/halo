"""
Convert KU-HAR dataset to standardized format.

Input: data/datasets/kuhar/downloads/
Output: data/datasets/kuhar/
  - manifest.json (channel metadata)
  - labels.json (activity labels per session)
  - sessions/session_XXX/data.parquet (sensor data)

KU-HAR format:
  - Folders organized as: N.ActivityName (e.g., "16.Stair-down")
  - CSV files: timestamp, acc_x, acc_y, acc_z, timestamp2, gyro_x, gyro_y, gyro_z
  - No headers in CSV files
  - 90 subjects, 18 activities
  - 100 Hz sampling rate
"""

import json
import re
import shutil
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from math import gcd
from scipy.signal import resample_poly

# Run as `python -m data.datasets.kuhar.convert` from repo root; repo root on path for shared imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _estimate_rate(timestamps: np.ndarray, nominal: float) -> float:
    """Estimate the sampling rate (Hz) from a timestamp column in seconds.

    KU-HAR is nominally 100 Hz, but at least one subject (1016) records ~111 Hz.
    Windowing counts samples, so using a wrong rate mis-sizes windows in seconds.
    Returns the nominal rate when timestamps are unusable (constant / non-monotone).
    """
    t = np.asarray(timestamps, dtype=float)
    if t.size < 10:
        return nominal
    dt = np.diff(t)
    dt = dt[(dt > 0) & (dt < 1.0)]
    if dt.size < 5:
        return nominal
    return float(1.0 / np.median(dt))


# Activity mapping - folder name to standardized name
ACTIVITIES = {
    "Stand": "standing",
    "Sit": "sitting",
    "Talk-sit": "talking_sitting",
    "Talk-stand": "talking_standing",
    # The release describes these as repeated bidirectional sequences, not one-way
    # transitions. Keeping them distinct avoids contaminating transition classes.
    "Stand-sit": "repeated_standing_and_sitting",
    "Lay": "lying",
    "Lay-stand": "repeated_standing_and_lying",
    "Pick": "picking_up",
    "Jump": "jumping",
    "Push-up": "push_up",
    "Sit-up": "sit_up",
    "Walk": "walking",
    "Walk-backwards": "walking_backwards",
    "Walk-circle": "walking_in_circles",
    "Run": "running",
    "Stair-up": "walking_upstairs",
    "Stair-down": "walking_downstairs",
    "Table-tennis": "table_tennis",
}

# Paths (new repo layout: this converter lives in data/datasets/kuhar/)
DS_DIR = Path(__file__).resolve().parent
RAW_DIR = DS_DIR / "downloads"
OUTPUT_DIR = DS_DIR

# Dataset parameters
SAMPLE_RATE = 100.0  # Hz
CLOCK_GAP_INTERVALS = 5


def split_clock_runs(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a KU-HAR file at genuine clock resets and material timing gaps.

    The release contains millisecond-scale backwards timestamp jitter, which must remain in the
    same physical recording, as well as rare 18--41 second resets and missing stretches.  Estimate
    the normal cadence from positive intervals, then use a cadence-relative threshold so this does
    not encode a source-specific magic number.
    """
    if frame.empty:
        return []
    clock = pd.to_numeric(frame["timestamp_sec"], errors="coerce").to_numpy(np.float64)
    positive = np.diff(clock)
    positive = positive[np.isfinite(positive) & (positive > 0) & (positive < 1.0)]
    observed_dt = float(np.median(positive)) if positive.size >= 5 else 1.0 / SAMPLE_RATE
    # A small subset of releases records bursts at 1--2 ms followed by ~20 ms idle periods.  The
    # median positive delta then describes the serializer, not the physical 100 Hz acquisition;
    # using it alone would turn each ordinary burst into a fake gap.  A true gap must exceed both
    # the observed cadence and the documented sensor cadence by this many intervals.
    threshold = CLOCK_GAP_INTERVALS * max(observed_dt, 1.0 / SAMPLE_RATE)
    delta = np.diff(clock)
    cuts = np.flatnonzero(
        ~np.isfinite(delta) | (delta < -threshold) | (delta > threshold)
    ) + 1
    bounds = np.r_[0, cuts, len(frame)]
    runs: list[pd.DataFrame] = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        run = frame.iloc[int(start):int(stop)].copy().reset_index(drop=True)
        # Preserve even a short terminal run. Grid construction later applies the experiment's
        # minimum-window rule; dropping it here would silently turn a clock repair into data loss.
        if len(run) >= 2:
            # Each run receives its own uniform local clock after any anti-aliased resampling.
            runs.append(run)
    return runs


def parse_folder_name(folder_name: str) -> Optional[str]:
    """Parse activity name from folder like '16.Stair-down'."""
    # Remove numeric prefix
    match = re.match(r'\d+\.(.+)', folder_name)
    if match:
        activity_key = match.group(1)
        return ACTIVITIES.get(activity_key)
    return None


def load_csv_file(filepath: Path) -> Optional[pd.DataFrame]:
    """Load and parse CSV file (no header)."""
    try:
        # CSV format: timestamp, acc_x, acc_y, acc_z, timestamp2, gyro_x, gyro_y, gyro_z
        df = pd.read_csv(filepath, header=None)

        if len(df.columns) < 7:
            return None

        # Build standardized DataFrame
        result = pd.DataFrame()

        # Use first timestamp column
        result["timestamp_sec"] = df.iloc[:, 0].values

        # Accelerometer (columns 1-3)
        result["acc_x"] = df.iloc[:, 1].values
        result["acc_y"] = df.iloc[:, 2].values
        result["acc_z"] = df.iloc[:, 3].values

        # Gyroscope (columns 5-7, skipping timestamp2 at column 4)
        result["gyro_x"] = df.iloc[:, 5].values
        result["gyro_y"] = df.iloc[:, 6].values
        result["gyro_z"] = df.iloc[:, 7].values

        channels = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        values = result[channels].apply(pd.to_numeric, errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        complete = values.notna().all(axis=1).to_numpy()
        if not complete.all():
            valid_rows = np.flatnonzero(complete)
            if valid_rows.size == 0:
                print(f"  ERROR {filepath}: no row has a complete finite IMU sample")
                return None
            # A modality occasionally stops a few samples before the other one (for example the last
            # four gyro rows of 1101_O_11.csv). Those rows are not six-channel observations and must
            # not be extrapolated. Trim unsupported edges, then interpolate only bounded interior gaps.
            first, last = int(valid_rows[0]), int(valid_rows[-1]) + 1
            result = result.iloc[first:last].reset_index(drop=True)
            result[channels] = values.iloc[first:last].interpolate(
                method="linear", limit_area="inside",
            ).to_numpy()
            if not np.isfinite(result[channels].to_numpy(dtype=np.float64)).all():
                print(f"  ERROR {filepath}: unresolved non-finite IMU samples")
                return None
            print(f"  trimmed/interpolated {int((~complete).sum())} incomplete rows in {filepath.name}")

        return result

    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None


def find_data_files(raw_dir: Path) -> List[Dict]:
    """Find all data files with activity labels."""
    data_files = []

    # Look in Raw_time_domian_data folder
    raw_data_dir = raw_dir / "1.Raw_time_domian_data"
    if not raw_data_dir.exists():
        # Try without the typo
        raw_data_dir = raw_dir / "1.Raw_time_domain_data"
    if not raw_data_dir.exists():
        # Just use any subfolder with activity folders
        for subdir in raw_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                raw_data_dir = subdir
                break

    if not raw_data_dir or not raw_data_dir.exists():
        return []

    print(f"Using data directory: {raw_data_dir}")

    # Iterate through activity folders
    for activity_folder in raw_data_dir.iterdir():
        if not activity_folder.is_dir():
            continue

        activity = parse_folder_name(activity_folder.name)
        if not activity:
            continue

        # Find CSV files in this activity folder
        for csv_file in activity_folder.glob("*.csv"):
            # Extract subject ID from filename (e.g., "1038_T_1.csv" -> subject 1038).
            # Deterministic fallback (zlib.crc32, not hash()) so re-runs are stable.
            subject_match = re.match(r'(\d+)_', csv_file.stem)
            subject = int(subject_match.group(1)) if subject_match else zlib.crc32(csv_file.stem.encode()) % 10000

            data_files.append({
                "filepath": csv_file,
                "activity": activity,
                "subject": subject,
            })

    return data_files


def convert_kuhar():
    """Convert KU-HAR dataset to standardized format."""
    print("=" * 80)
    print("KU-HAR → Standardized Format Converter")
    print("=" * 80)

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Run: python -m data.scripts.download_datasets kuhar")
        return False

    # Find data files
    data_files = find_data_files(RAW_DIR)
    print(f"Found {len(data_files)} data files")

    if len(data_files) == 0:
        print("ERROR: No valid data files found")
        return False

    # Create output directory. Clear any prior conversion so a re-run doesn't mix
    # stale sessions in with the new ones.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions_dir = OUTPUT_DIR / "sessions"
    shutil.rmtree(sessions_dir, ignore_errors=True)
    sessions_dir.mkdir(exist_ok=True)

    all_labels = {}
    session_count = 0
    skipped_count = 0
    activity_counts = {}
    subject_window_counts: Dict[int, int] = {}
    rate_warned = set()

    for i, file_info in enumerate(data_files):
        filepath = file_info["filepath"]
        activity = file_info["activity"]
        subject = file_info["subject"]

        # Documented KU-HAR subject ids are 1001..1090. Flag anything outside that
        # range (e.g. 1101, which is a single high-volume ~8.7%-of-windows id) so a
        # re-fetch surfaces the anomaly instead of silently over-weighting it.
        if not (1001 <= subject <= 1090):
            print(f"  ⚠ subject {subject} outside documented range 1001–1090 ({filepath.name})")

        # Load CSV
        df = load_csv_file(filepath)
        if df is None or len(df) < 20:
            skipped_count += 1
            continue

        runs = split_clock_runs(df)
        if not runs:
            skipped_count += 1
            continue
        if len(runs) > 1:
            print(f"  split {filepath.name} into {len(runs)} clock-contiguous runs")
        for part, run in enumerate(runs):
            # Estimate each run's true rate (subject 1016 is about 111 Hz). Windowing counts
            # samples, so a wrong rate mis-sizes physical windows.
            file_rate = _estimate_rate(run["timestamp_sec"].values, SAMPLE_RATE)
            if abs(file_rate - SAMPLE_RATE) / SAMPLE_RATE > 0.05 and subject not in rate_warned:
                print(f"  ⚠ subject {subject}: est. rate {file_rate:.1f} Hz (nominal {SAMPLE_RATE}) — resampling")
                rate_warned.add(subject)

            # Normalize each contiguous run to 100 Hz.  Never interpolate across a clock seam.
            chans = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
            if abs(file_rate - SAMPLE_RATE) / SAMPLE_RATE > 0.02:
                s, d = int(round(file_rate)), int(round(SAMPLE_RATE))
                g = gcd(s, d)
                arr = resample_poly(run[chans].to_numpy(np.float64), up=d // g, down=s // g, axis=0)
                run = pd.DataFrame(arr, columns=chans)
            run["timestamp_sec"] = np.arange(len(run), dtype=np.float64) / SAMPLE_RATE
            run["subject"] = f"s{subject:04d}"
            session_id = f"s{subject:04d}_{activity}_{i:04d}_p{part:02d}"
            session_dir = sessions_dir / session_id
            session_dir.mkdir(exist_ok=True)
            run.to_parquet(session_dir / "data.parquet", index=False)

            all_labels[session_id] = [activity]
            subject_window_counts[subject] = subject_window_counts.get(subject, 0) + 1
            session_count += 1
            activity_counts[activity] = activity_counts.get(activity, 0) + 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(data_files)} files, {session_count} sessions...")

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\n✓ Created labels.json ({len(all_labels)} sessions)")

    # Create manifest.json
    create_manifest()

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions converted")
    print(f"  - {skipped_count} files skipped")
    print(f"  - {len(activity_counts)} unique activities")
    print(f"  - {SAMPLE_RATE} Hz sampling rate")
    print("\nActivity distribution:")
    for activity, count in sorted(activity_counts.items(), key=lambda x: -x[1]):
        print(f"  {activity}: {count}")

    # Surface per-subject over-representation (subject 1101 held ~8.7% of windows).
    if subject_window_counts and session_count > 0:
        top = sorted(subject_window_counts.items(), key=lambda x: -x[1])[:5]
        print("\nTop subjects by window share (watch for >5% single-subject dominance):")
        for subj, cnt in top:
            print(f"  s{subj}: {cnt} ({100 * cnt / session_count:.1f}%)")

    # Generate visualizations
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.scripts.debug.visualization_utils import generate_debug_visualizations
        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError:
        pass

    return True


def create_manifest():
    """Create manifest.json with channel metadata.

    Kept in sync with the committed ``data/kuhar/manifest.json`` so re-running the
    converter does not silently degrade the manifest. Notes captured here:
      - 18 unique activities. In particular, the repeated stand/sit and stand/lie
        protocols remain bidirectional classes, and circular walking is not folded
        into straight walking.
      - Accelerometer is GRAVITY-REMOVED (linear acceleration): |acc| over static
        activities (standing/sitting/lying) is ~0.06, not ~9.8/1.0. Verified
        empirically. The gravity-canonicalization augmentation must treat these
        channels as gravity-free.
      - Nominal 100 Hz. One subject (1016) records at ~111 Hz; HALO's timestamp-
        native tokenizer handles this from ``timestamp_sec`` directly.
    """
    manifest = {
        "dataset_name": "KU-HAR",
        "description": "KU-HAR (Khulna University Human Activity Recognition) dataset. 89 subjects performing 18 activities including walking, running, jumping, stairs, sitting, standing, lying, repeated posture transitions, and exercises. Smartphone IMU with triaxial accelerometer (gravity removed / linear acceleration) and gyroscope.",
        "source": "https://www.kaggle.com/datasets/niloy333/kuhar",
        "num_subjects": 89,
        "channels": [
            {"name": "acc_x", "description": "Smartphone accelerometer X-axis (waist), gravity removed", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_y", "description": "Smartphone accelerometer Y-axis (waist), gravity removed", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_z", "description": "Smartphone accelerometer Z-axis (waist), gravity removed", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_x", "description": "Smartphone gyroscope X-axis (waist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_y", "description": "Smartphone gyroscope Y-axis (waist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_z", "description": "Smartphone gyroscope Z-axis (waist)", "sampling_rate_hz": SAMPLE_RATE},
        ],
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("✓ Created manifest.json")


def main():
    """Main entry point."""
    success = convert_kuhar()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
