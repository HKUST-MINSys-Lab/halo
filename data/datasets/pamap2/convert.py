"""
Convert PAMAP2 dataset to standardized format.

Input: data/datasets/pamap2/downloads/PAMAP2_Dataset/
Output: data/datasets/pamap2/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet
"""

import shutil
import sys
import json
import pandas as pd
from pathlib import Path

# Run as `python -m data.datasets.pamap2.convert` from repo root; repo root on path for shared imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# Activity mapping
ACTIVITIES = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping"
}

# Column names for PAMAP2 (54 columns). Per the Reiss thesis Appendix B Table B.2,
# each IMU is 17 columns with orientation INTERLEAVED after the magnetometer (NOT
# appended at the end of the file). The previous code appended all 12 orientation
# columns at the end, which shifted every chest/ankle channel and fed garbage
# (temperature / orientation / other-IMU magnetometer) into chest/ankle acc/gyro/mag.
_IMU_SENSORS = ['temp',
                'acc16_x', 'acc16_y', 'acc16_z',   # 3D-acc, +/-16g
                'acc6_x', 'acc6_y', 'acc6_z',      # 3D-acc, +/-6g
                'gyro_x', 'gyro_y', 'gyro_z',      # 3D-gyroscope
                'mag_x', 'mag_y', 'mag_z',         # 3D-magnetometer
                'ori_1', 'ori_2', 'ori_3', 'ori_4']  # orientation (invalid/off)
COLUMN_NAMES = ['timestamp', 'activity_id', 'heart_rate'] + [
    f'{location}_{sensor}'
    for location in ['hand', 'chest', 'ankle']
    for sensor in _IMU_SENSORS
]

# Paths (new repo layout: this converter lives in data/datasets/pamap2/)
DS_DIR = Path(__file__).resolve().parent
RAW_DIR = DS_DIR / "downloads" / "PAMAP2_Dataset" / "Protocol"
OUTPUT_DIR = DS_DIR


def segment_continuous_activity(df: pd.DataFrame, min_duration_sec: float = 5.0):
    """Segment continuous recording into sessions based on activity changes."""
    sessions = []

    # Find activity boundaries
    activity_changes = df['activity_id'].ne(df['activity_id'].shift())
    segment_ids = activity_changes.cumsum()

    for seg_id in segment_ids.unique():
        segment = df[segment_ids == seg_id].copy()

        # Skip transient activities and very short segments
        activity_id = segment['activity_id'].iloc[0]
        if activity_id == 0:
            continue

        duration = (segment['timestamp'].iloc[-1] - segment['timestamp'].iloc[0])
        if duration < min_duration_sec:
            continue

        sessions.append({
            'data': segment,
            'activity_id': activity_id,
            'duration_sec': duration
        })

    return sessions


def convert_subject(subject_file: Path):
    """Convert one subject file to sessions with variable-length windowing."""
    print(f"  Processing: {subject_file.name}")

    # Load data (space-separated, 54 columns)
    try:
        df = pd.read_csv(subject_file, sep=' ', header=None, names=COLUMN_NAMES)
    except Exception as e:
        print(f"    ERROR loading {subject_file.name}: {e}")
        return [], {}

    # Interpolate NaN values (wireless data loss is common in PAMAP2)
    # Use linear interpolation for sensor data, forward-fill for activity_id
    df['activity_id'] = df['activity_id'].ffill().bfill()  # Forward-fill activity labels
    df = df.interpolate(method='linear', limit_direction='both')  # Linear interpolation for sensors
    df = df.fillna(0)  # Fill any remaining NaN at edges with 0

    # Segment into sessions
    sessions = segment_continuous_activity(df)

    print(f"    Found {len(sessions)} activity segments")

    subject_id = subject_file.stem.replace('subject', '')
    session_data = []
    labels_dict = {}
    total_windows = 0

    for idx, session in enumerate(sessions):
        # Create base session ID
        base_session_id = f"subject{subject_id}_seg{idx:03d}"
        activity_name = ACTIVITIES.get(session['activity_id'], 'unknown')

        # Prepare DataFrame (drop activity_id column, keep only sensor data)
        data = session['data'].copy()

        # Reset timestamp to start from 0
        data['timestamp_sec'] = data['timestamp'] - data['timestamp'].iloc[0]

        # Drop activity_id (it's in labels.json)
        data = data.drop(columns=['activity_id'])

        # Subject id for subject-disjoint splits (read by build_grids.iter_sessions).
        data['subject'] = f"subject{subject_id}"

        # Reorder: timestamp first, then all channels
        cols = ['timestamp_sec'] + [c for c in data.columns if c != 'timestamp_sec' and c != 'timestamp']
        data = data[cols]

        # Save the whole continuous single-activity segment as ONE session; build_grids does the
        # fixed 6 s windowing (avoids double-windowing).
        session_dir = OUTPUT_DIR / "sessions" / base_session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        data.to_parquet(session_dir / "data.parquet", index=False)

        labels_dict[base_session_id] = [activity_name]
        session_data.append(base_session_id)
        total_windows += 1

    print(f"    Created {total_windows} sessions from {len(sessions)} segments")
    return session_data, labels_dict


def create_manifest():
    """Create minimal manifest.json."""
    # Create channel metadata (only for valid sensors, skip orientation)
    channels = [
        {
            "name": "heart_rate",
            "description": "Heart rate from chest monitor",
            "sampling_rate_hz": 9.0
        }
    ]

    # IMU sensors on 3 locations
    for location in ['hand', 'chest', 'ankle']:
        location_desc = {
            'hand': 'wrist-mounted',
            'chest': 'chest-mounted',
            'ankle': 'ankle-mounted'
        }[location]

        # Temperature
        channels.append({
            "name": f"{location}_temp",
            "description": f"Temperature from {location_desc} IMU",
            "sampling_rate_hz": 100.0
        })

        # Accelerometers (16g and 6g scales)
        for scale in ['16', '6']:
            for axis in ['x', 'y', 'z']:
                channels.append({
                    "name": f"{location}_acc{scale}_{axis}",
                    "description": f"Acceleration {axis}-axis (±{scale}g sensor range, values in m/s^2 including gravity) from {location_desc} IMU",
                    "sampling_rate_hz": 100.0
                })

        # Gyroscope
        for axis in ['x', 'y', 'z']:
            channels.append({
                "name": f"{location}_gyro_{axis}",
                "description": f"Angular velocity {axis}-axis from {location_desc} IMU",
                "sampling_rate_hz": 100.0
            })

        # Magnetometer
        for axis in ['x', 'y', 'z']:
            channels.append({
                "name": f"{location}_mag_{axis}",
                "description": f"Magnetic field {axis}-axis from {location_desc} IMU",
                "sampling_rate_hz": 100.0
            })

        # Orientation (quaternion) — invalid in PAMAP2 (sensor orientation was off)
        for i in range(1, 5):
            channels.append({
                "name": f"{location}_ori_{i}",
                "description": f"Orientation quaternion component {i} from {location_desc} IMU (INVALID: orientation turned off in PAMAP2; constant placeholder)",
                "sampling_rate_hz": 100.0
            })

    manifest = {
        "dataset_name": "PAMAP2",
        "description": "Physical activity monitoring with 3 IMUs (hand, chest, ankle) and heart rate. 9 subjects performing the 12 required protocol activities including walking, running, cycling, and household tasks. Optional-protocol recordings are intentionally excluded so included subjects share one protocol.",
        "channels": channels
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created manifest: {manifest_path}")


def main():
    """Convert PAMAP2 to standardized format."""
    print("=" * 80)
    print("PAMAP2 → Standardized Format Converter")
    print("=" * 80)

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Run: python -m data.scripts.download_datasets pamap2")
        return

    # Create output directory. Clear any prior conversion so re-runs don't leave
    # stale orphan session dirs (the loader indexes labels.json, so orphans are
    # inert but they inflate disk and desync on-disk vs labels.json counts).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUTPUT_DIR / "sessions", ignore_errors=True)

    # Process all subject files
    subject_files = sorted(RAW_DIR.glob("subject*.dat"))
    print(f"\nFound {len(subject_files)} subject files")

    all_labels = {}
    all_sessions = []

    for subject_file in subject_files:
        sessions, labels = convert_subject(subject_file)
        all_sessions.extend(sessions)
        all_labels.update(labels)

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(all_labels, f, indent=2)

    print(f"\n✓ Created labels: {labels_path}")
    print(f"  Total sessions: {len(all_labels)}")

    # Create manifest
    create_manifest()
    metadata_path = OUTPUT_DIR / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["num_sessions"] = len(all_labels)
        metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(all_labels)} sessions")
    print("  - ~40 channels (3 IMUs + heart rate)")
    print("  - 100 Hz IMU, 9 Hz heart rate")

    # Generate debug visualizations
    try:
        from data.scripts.debug.visualization_utils import generate_debug_visualizations

        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError as e:
        print(f"\n⚠ Could not generate visualizations: {e}")
        print("Install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
