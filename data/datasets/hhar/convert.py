"""
Convert HHAR (Heterogeneity Activity Recognition) dataset to standardized format.

Input: data/datasets/hhar/downloads/Activity recognition exp/
Output: data/datasets/hhar/
  - manifest.json (minimal, human-readable)
  - labels.json (activity labels per session)
  - sessions/<session_id>/data.parquet (raw whole-recording session; build_grids does the windowing)

HHAR Dataset Info:
- 9 users (a-i)
- 6 activities: stand, walk, bike, sit, stairsup, stairsdown
- Multiple device models (phones and watches)
- We use phone data for consistency with other datasets
- ~43.9 million samples total
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from data.scripts.assembly.assemble import resample_signal


# Activity mapping (original names to standardized)
ACTIVITIES = {
    "stand": "standing",
    "sit": "sitting",
    "walk": "walking",
    "bike": "cycling",
    "stairsup": "walking_upstairs",
    "stairsdown": "walking_downstairs",
}

# Paths (new repo layout: this converter lives in data/datasets/hhar/)
DS_DIR = Path(__file__).resolve().parent
RAW_DIR = DS_DIR / "downloads" / "Activity recognition exp"
OUTPUT_DIR = DS_DIR

# Processing parameters
TARGET_SAMPLE_RATE = 50  # Hz (HHAR native rates vary 50-200 Hz; resampled to a uniform 50 Hz here)
MIN_SESSION_SAMPLES = 300  # 6 s @ 50 Hz — the shared 6 s grid window; shorter segments yield no window


def load_and_merge_sensor_data(acc_path: Path, gyro_path: Path) -> pd.DataFrame:
    """
    Load and merge accelerometer and gyroscope data.

    Returns DataFrame with columns:
    - timestamp_ns: Creation_Time in nanoseconds
    - acc_x, acc_y, acc_z: Accelerometer data
    - gyro_x, gyro_y, gyro_z: Gyroscope data
    - user: User ID (a-i)
    - activity: Ground truth activity
    """
    print("  Loading accelerometer data...")
    # Read in chunks to handle large file
    acc_chunks = []
    for chunk in pd.read_csv(acc_path, chunksize=1000000):
        # Filter out null activities and keep only needed columns
        chunk = chunk[chunk['gt'].notna() & (chunk['gt'] != 'null')]
        chunk = chunk[['Creation_Time', 'x', 'y', 'z', 'User', 'Device', 'gt']]
        acc_chunks.append(chunk)

    acc_df = pd.concat(acc_chunks, ignore_index=True)
    acc_df.columns = ['timestamp_ns', 'acc_x', 'acc_y', 'acc_z', 'user', 'device', 'activity']
    print(f"    Loaded {len(acc_df):,} accelerometer samples")

    print("  Loading gyroscope data...")
    gyro_chunks = []
    for chunk in pd.read_csv(gyro_path, chunksize=1000000):
        chunk = chunk[chunk['gt'].notna() & (chunk['gt'] != 'null')]
        chunk = chunk[['Creation_Time', 'x', 'y', 'z', 'User', 'Device', 'gt']]
        gyro_chunks.append(chunk)

    gyro_df = pd.concat(gyro_chunks, ignore_index=True)
    gyro_df.columns = ['timestamp_ns', 'gyro_x', 'gyro_y', 'gyro_z', 'user', 'device', 'activity']
    print(f"    Loaded {len(gyro_df):,} gyroscope samples")

    # Merge on timestamp, user, and activity (approximate matching)
    print("  Merging sensor data...")

    # Sort by user, DEVICE, activity, timestamp. Streams are physical: never merge
    # across devices (their rates AND biases differ — the whole point of HHAR).
    acc_df = acc_df.sort_values(['user', 'device', 'activity', 'timestamp_ns']).reset_index(drop=True)
    gyro_df = gyro_df.sort_values(['user', 'device', 'activity', 'timestamp_ns']).reset_index(drop=True)

    merged_data = []
    key_columns = ['user', 'device', 'activity']
    # The old loop boolean-scanned both 10M-row frames once per key. Grouping once preserves the
    # exact physical key while avoiding hundreds of full-corpus passes.
    gyro_groups = gyro_df.groupby(key_columns, sort=False, observed=True)
    gyro_keys = set(gyro_groups.groups)
    for (user, device, activity), acc_subset in acc_df.groupby(
        key_columns, sort=False, observed=True
    ):
        acc_subset = acc_subset.sort_values('timestamp_ns')
        key = (user, device, activity)
        gyro_subset = (gyro_groups.get_group(key).sort_values('timestamp_ns')
                       if key in gyro_keys else None)
        if gyro_subset is None or len(gyro_subset) == 0:
            # Galaxy S+ has no gyroscope rows in the release. Preserve it as an honest
            # accelerometer-only stream rather than silently deleting the only 50 Hz phone model.
            missing = acc_subset.copy()
            for column in ('gyro_x', 'gyro_y', 'gyro_z'):
                missing[column] = np.nan
            merged_data.append(missing)
            continue

        # Match gyro to acc timestamps within one physical device (50ms tolerance)
        merged = pd.merge_asof(
            acc_subset,
            gyro_subset[['timestamp_ns', 'gyro_x', 'gyro_y', 'gyro_z']],
            on='timestamp_ns',
            direction='nearest',
            tolerance=50_000_000,  # 50ms in nanoseconds
        )
        merged = merged.dropna(subset=['gyro_x', 'gyro_y', 'gyro_z'])
        merged_data.append(merged)

    result = pd.concat(merged_data, ignore_index=True)
    n_accel_only = int(result['gyro_x'].isna().sum())
    print(f"    Prepared: {len(result):,} samples "
          f"({n_accel_only:,} from accelerometer-only devices)")

    return result


def resample_stream(group, rate=TARGET_SAMPLE_RATE):
    """Resample one physical (user,device,activity) stream to a uniform `rate` Hz on its REAL
    Creation_Time clock. The stream is FIRST split into contiguous segments at timestamp gaps so no
    gap is filled with a fabricated linear ramp — a single corrupt Creation_Time would otherwise
    open a multi-hour fake 'standing' stream (interpolated dead-flat). Native HHAR rates are
    50-200 Hz. Returns a LIST of resampled segment DataFrames (one per contiguous run)."""
    g = group.sort_values('timestamp_ns').drop_duplicates('timestamp_ns')
    if len(g) < 2:
        return []
    t = g['timestamp_ns'].to_numpy(np.float64) / 1e9
    t = t - t[0]
    dt = np.diff(t)
    med = float(np.median(dt)) if len(dt) else 0.0
    gap = max(0.25, 10.0 * med)               # a real clock break, not sample jitter
    breaks = np.nonzero(dt > gap)[0] + 1       # a new segment starts after each gap
    bounds = np.concatenate([[0], breaks, [len(g)]]).astype(int)
    all_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    cols = [column for column in all_cols if column in g and g[column].notna().any()]
    arr = {c: g[c].to_numpy(np.float64) for c in cols}
    segments = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        ts = t[a:b] - t[a]
        if len(ts) < 2 or ts[-1] <= 0:
            continue
        local_dt = np.diff(ts)
        local_dt = local_dt[local_dt > 0]
        if len(local_dt) == 0:
            continue
        source_rate = 1.0 / float(np.median(local_dt))
        source_grid = np.arange(int(np.floor(ts[-1] * source_rate)) + 1) / source_rate
        uniform = np.column_stack([np.interp(source_grid, ts, arr[c][a:b]) for c in cols])
        sampled = resample_signal(uniform, source_rate, rate)
        out = {'timestamp_sec': np.arange(len(sampled), dtype=np.float64) / rate}
        for ci, c in enumerate(cols):
            out[c] = sampled[:, ci]
        segments.append(pd.DataFrame(out))
    return segments


def convert_sessions(df: pd.DataFrame, sessions_dir: Path) -> Dict[str, List[str]]:
    """Save each continuous (user, device, activity) recording as ONE RAW session.

    Preserves HHAR's device heterogeneity: every physical (user, device, activity) stream is
    gap-split and resampled to a uniform 50 Hz on its real Creation_Time clock (see
    resample_stream), then EACH contiguous resampled segment is written whole as one session.
    No pre-windowing here — the shared build_grids.py cuts the fixed 6 s windows (pre-windowing
    would double-window the grid). Accel stays in native m/s²; the shared pipeline rescales to g.

    Returns labels_dict mapping session_id -> [standardized_activity].
    """
    labels_dict: Dict[str, List[str]] = {}
    session_count = 0
    skipped_short = 0

    # Group by user, DEVICE, activity — never merge across devices (rates AND biases differ).
    for (user, device, activity), group in df.groupby(['user', 'device', 'activity']):
        std_activity = ACTIVITIES.get(activity, activity)
        # Each CONTIGUOUS resampled segment (gap-split) becomes one whole session — no window ever
        # spans a fabricated across-gap interpolation ramp.
        for seg_idx, seg in enumerate(resample_stream(group)):
            if len(seg) < MIN_SESSION_SAMPLES:
                skipped_short += 1
                continue
            seg = seg.copy()
            # Subject id (user a-i) for subject-disjoint splits; read by build_grids.iter_sessions.
            seg['subject'] = str(user)

            accel_only = not {'gyro_x', 'gyro_y', 'gyro_z'} <= set(seg.columns)
            kind = "accel_only" if accel_only else "imu"
            session_id = f"hhar_{kind}_{user}_{device}_{activity}_{seg_idx:03d}"
            session_dir = sessions_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            # Columns: timestamp_sec, acc_x/y/z, gyro_x/y/z (native m/s²), subject.
            seg.to_parquet(session_dir / "data.parquet", index=False)
            labels_dict[session_id] = [std_activity]
            session_count += 1

    print(f"    Wrote {session_count} raw sessions "
          f"({skipped_short} segments skipped: < {MIN_SESSION_SAMPLES} samples / 6 s)")
    return labels_dict


def create_manifest():
    """Create minimal manifest.json."""
    manifest = {
        "dataset_name": "HHAR",
        "description": "Heterogeneity Human Activity Recognition dataset. 9 users performing 6 activities with smartphones (Nexus 4, Galaxy S+, Galaxy S3, S3 mini) carried in a waist pouch. Phones sampled at 50-200 Hz and are anti-aliased to 50 Hz. Galaxy S+ is accelerometer-only because the release has no gyroscope rows for that model; the other phones retain both sensors.",
        "channels": [
            {
                "name": "acc_x",
                "description": "Accelerometer X-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "acc_y",
                "description": "Accelerometer Y-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "acc_z",
                "description": "Accelerometer Z-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gyro_x",
                "description": "Gyroscope X-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gyro_y",
                "description": "Gyroscope Y-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gyro_z",
                "description": "Gyroscope Z-axis",
                "sampling_rate_hz": 50.0
            }
        ]
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Created manifest: {manifest_path}")


def main():
    """Convert HHAR to standardized format."""
    print("=" * 80)
    print("HHAR -> Standardized Format Converter")
    print("=" * 80)

    # Check input
    acc_path = RAW_DIR / "Phones_accelerometer.csv"
    gyro_path = RAW_DIR / "Phones_gyroscope.csv"

    if not acc_path.exists():
        print(f"ERROR: Accelerometer data not found at {acc_path}")
        print("Run: python -m data.scripts.download_datasets hhar")
        return

    if not gyro_path.exists():
        print(f"ERROR: Gyroscope data not found at {gyro_path}")
        print("Run: python -m data.scripts.download_datasets hhar")
        return

    # Create output directory
    import shutil
    shutil.rmtree(OUTPUT_DIR / "sessions", ignore_errors=True)   # clear stale windows -> no orphan dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and merge sensor data
    print("\nStep 1: Loading and merging sensor data...")
    merged_df = load_and_merge_sensor_data(acc_path, gyro_path)

    # Emit raw whole-recording sessions (build_grids does the fixed 6 s windowing)
    print("\nStep 2: Writing raw sessions...")
    labels_dict = convert_sessions(merged_df, OUTPUT_DIR / "sessions")

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(labels_dict, f, indent=2)

    print(f"\nCreated labels: {labels_path}")
    print(f"  Total sessions: {len(labels_dict)}")

    # Create manifest
    create_manifest()
    metadata_path = OUTPUT_DIR / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["num_sessions"] = len(labels_dict)
        metadata.pop("num_sessions_subsampled", None)
        metadata_path.write_text(json.dumps(metadata, indent=2))

    # Print summary
    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(labels_dict)} raw sessions (whole recordings; not pre-windowed)")
    print("  - 3 channels for Galaxy S+; 6 channels for devices with gyroscope data")
    print(f"  - {TARGET_SAMPLE_RATE} Hz sampling rate")

    # Activity distribution
    activity_counts = {}
    for session_id, labels in labels_dict.items():
        for label in labels:
            activity_counts[label] = activity_counts.get(label, 0) + 1

    print("\nActivity distribution:")
    for activity, count in sorted(activity_counts.items()):
        print(f"  {activity}: {count}")


if __name__ == "__main__":
    main()
