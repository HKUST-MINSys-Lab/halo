"""
Convert WISDM dataset to standardized format.

Input: data/datasets/wisdm/downloads/wisdm-dataset/
Output: data/datasets/wisdm/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from data.scripts.assembly.assemble import resample_signal

# Run as `python -m data.datasets.wisdm.convert` from repo root; repo root on path for shared imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# Activity mapping (A-S, excluding N)
ACTIVITIES = {
    'A': 'walking',
    'B': 'jogging',
    'C': 'stairs',
    'D': 'sitting',
    'E': 'standing',
    'F': 'typing',
    'G': 'brushing_teeth',
    'H': 'eating_soup',
    'I': 'eating_chips',
    'J': 'eating_pasta',
    'K': 'drinking',
    'L': 'eating_sandwich',
    'M': 'kicking',
    'O': 'playing_catch',
    'P': 'dribbling',
    'Q': 'writing',
    'R': 'clapping',
    'S': 'folding_clothes'
}

# Paths (new repo layout: this converter lives in data/datasets/wisdm/)
DS_DIR = Path(__file__).resolve().parent
RAW_DIR = DS_DIR / "downloads" / "wisdm-dataset" / "raw"
OUTPUT_DIR = DS_DIR


def load_sensor_data(device: str, sensor: str):
    """Load data for one device-sensor combination."""
    sensor_dir = RAW_DIR / device / sensor

    if not sensor_dir.exists():
        print(f"    WARNING: {sensor_dir} not found")
        return {}

    # Data organized by subject (1600-1650)
    subject_files = sorted(sensor_dir.glob("*.txt"))

    subject_data = {}
    for subject_file in subject_files:
        subject_id = subject_file.stem

        # Parse CSV format: subject_id,activity_code,timestamp,x,y,z;
        # Note: WISDM data has semicolons at the end of each line
        try:
            # Read without dtype specification to handle semicolons
            df = pd.read_csv(
                subject_file,
                names=['subject', 'activity', 'timestamp', 'x', 'y', 'z'],
                sep=',',
                on_bad_lines='skip'  # Skip malformed lines
            )
            # Remove trailing semicolons from the last column and convert to numeric
            df['z'] = df['z'].astype(str).str.rstrip(';').astype(float)
            df['x'] = df['x'].astype(float)
            df['y'] = df['y'].astype(float)

            subject_data[subject_id] = df
        except Exception as e:
            print(f"      ERROR loading {subject_file.name}: {e}")

    return subject_data


# Merge / segmentation parameters
GAP_SPLIT_NS = 200_000_000       # split a subject/activity at timestamp gaps > 200 ms — WISDM has
                                 # multi-second clock gaps within an activity (max ~56 s); never
                                 # concatenate discontinuous segments across them.
MIN_SEG_ROWS = 20                # skip segments shorter than ~1 s @ 20 Hz


def _by_subject(sensor_dict: dict) -> dict:
    """Re-key a {file_stem: df} sensor dict by the NUMERIC subject id (the raw first column), so a
    subject's accelerometer and gyroscope files (different stems) pair up."""
    out = {}
    for df in sensor_dict.values():
        if len(df):
            out[str(int(df['subject'].iloc[0]))] = df
    return out


def _clock_runs(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Return strictly increasing contiguous timestamp runs."""
    ordered = frame.sort_values("timestamp").drop_duplicates("timestamp")
    if ordered.empty:
        return []
    cuts = np.flatnonzero(np.diff(ordered["timestamp"].to_numpy(np.float64)) > GAP_SPLIT_NS) + 1
    bounds = [0, *cuts.tolist(), len(ordered)]
    return [ordered.iloc[lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:])
            if hi - lo >= MIN_SEG_ROWS]


def _regularize(frame: pd.DataFrame, columns: list[str], target_hz: float = 20.0):
    """Regularize one clock run and anti-alias it to ``target_hz``.

    WISDM contains per-subject clocks near 20, 25, 50, and 100 Hz. Interpolating first onto the
    run's measured native clock makes the small timestamp jitter regular; polyphase resampling then
    applies the required low-pass filter before decimation.
    """
    t = frame["timestamp"].to_numpy(np.float64) / 1e9
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return np.empty(0), np.empty((0, len(columns)), np.float32)
    source_hz = 1.0 / float(np.median(dt))
    if not np.isfinite(source_hz) or source_hz < 1.0:
        return np.empty(0), np.empty((0, len(columns)), np.float32)
    native_t = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * source_hz)) + 1) / source_hz
    values = frame[columns].to_numpy(np.float64)
    uniform = np.column_stack([np.interp(native_t, t, values[:, i])
                               for i in range(values.shape[1])]).astype(np.float32)
    sampled = resample_signal(uniform, source_hz, target_hz)
    sampled_t = native_t[0] + np.arange(len(sampled), dtype=np.float64) / target_hz
    return sampled_t, sampled


def merged_sessions(device_label: str, accel_dict: dict, gyro_dict: dict):
    """Merge co-located accel+gyro into 6-channel sessions per subject/activity, split at clock gaps.

    The release contains several real acquisition rates despite advertising a nominal 20 Hz rate.
    Acceleration and gyroscope are therefore gap-split independently, regularized from their real
    timestamps, and anti-aliased to a common 20 Hz clock before they are joined. This prevents a
    120-row 50 Hz segment from being misrepresented as six seconds and prevents nearest-neighbour
    sample holding when the two sensor clocks differ.
    """
    accel_by_subj, gyro_by_subj = _by_subject(accel_dict), _by_subject(gyro_dict)
    sessions = []
    for subject_id, adf in accel_by_subj.items():
        gdf = gyro_by_subj.get(subject_id)
        for activity_code in sorted(adf['activity'].unique()):
            a = adf[adf['activity'] == activity_code]
            g = None if gdf is None else gdf[gdf['activity'] == activity_code]
            if g is None or len(g) == 0:
                continue     # require gyro so every wisdm session is a uniform 6-channel grid
            a = a.rename(columns={'x': f'{device_label}_accel_x', 'y': f'{device_label}_accel_y',
                                  'z': f'{device_label}_accel_z'})
            g = g[['timestamp', 'x', 'y', 'z']].rename(
                columns={'x': f'{device_label}_gyro_x', 'y': f'{device_label}_gyro_y',
                         'z': f'{device_label}_gyro_z'})
            activity_name = ACTIVITIES.get(activity_code, 'unknown')
            si = 0
            for arun in _clock_runs(a):
                for grun in _clock_runs(g):
                    lo = max(float(arun.timestamp.iloc[0]), float(grun.timestamp.iloc[0])) / 1e9
                    hi = min(float(arun.timestamp.iloc[-1]), float(grun.timestamp.iloc[-1])) / 1e9
                    if hi - lo < MIN_SEG_ROWS / 20.0:
                        continue
                    at, av = _regularize(arun, [f'{device_label}_accel_{x}' for x in 'xyz'])
                    gt, gv = _regularize(grun, [f'{device_label}_gyro_{x}' for x in 'xyz'])
                    if len(at) == 0 or len(gt) == 0:
                        continue
                    target = at[(at >= lo) & (at <= hi) & (at >= gt[0]) & (at <= gt[-1])]
                    if len(target) < MIN_SEG_ROWS:
                        continue
                    ai = np.clip(np.rint((target - at[0]) * 20.0).astype(int), 0, len(av) - 1)
                    gyro = np.column_stack([np.interp(target, gt, gv[:, c]) for c in range(3)])
                    seg = pd.DataFrame({"timestamp": target * 1e9})
                    for c, axis in enumerate("xyz"):
                        seg[f'{device_label}_accel_{axis}'] = av[ai, c]
                        seg[f'{device_label}_gyro_{axis}'] = gyro[:, c]
                    sessions.append({
                        'session_id': f"{device_label}_{subject_id}_{activity_code}_{si}",
                        'data': seg, 'activity_name': activity_name,
                        'device': device_label, 'subject': str(subject_id),
                    })
                    si += 1
    return sessions


def merge_device_sensors():
    """Build 6-channel (accel+gyro) sessions for phone and watch."""
    print("\nLoading sensor data...")
    phone_accel, phone_gyro = load_sensor_data("phone", "accel"), load_sensor_data("phone", "gyro")
    watch_accel, watch_gyro = load_sensor_data("watch", "accel"), load_sensor_data("watch", "gyro")
    print(f"  phone accel/gyro files: {len(phone_accel)}/{len(phone_gyro)} · "
          f"watch: {len(watch_accel)}/{len(watch_gyro)}")
    return (merged_sessions("phone", phone_accel, phone_gyro)
            + merged_sessions("watch", watch_accel, watch_gyro))


def save_sessions(sessions):
    """Save each merged 6-channel session to parquet (build_grids does the fixed 6 s windowing)."""
    labels_dict = {}
    for session in sessions:
        sid = session['session_id']
        data = session['data'].copy()
        # timestamp -> seconds from the segment start (WISDM timestamps are nanoseconds)
        data['timestamp_sec'] = ((data['timestamp'] - data['timestamp'].min()) / 1e9).astype(float)
        device = session['device']
        cols = ['timestamp_sec',
                f'{device}_accel_x', f'{device}_accel_y', f'{device}_accel_z',
                f'{device}_gyro_x', f'{device}_gyro_y', f'{device}_gyro_z']
        out = data[cols].astype(float)
        out['subject'] = session['subject']

        session_dir = OUTPUT_DIR / "sessions" / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(session_dir / "data.parquet", index=False)
        labels_dict[sid] = [session['activity_name']]

    print(f"  Created {len(labels_dict)} merged 6-channel sessions")
    return labels_dict


def create_manifest():
    """Create minimal manifest.json."""
    manifest = {
        "dataset_name": "WISDM",
        "description": "Smartphone and smartwatch activity recognition. 51 subjects performing 18 activities with accelerometer and gyroscope on a phone in the right trouser pocket and a watch on the dominant wrist. Mixed native clocks are anti-aliased to 20 Hz during conversion.",
        "channels": [
            {
                "name": "phone_accel_x",
                "description": "Phone accelerometer X-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "phone_accel_y",
                "description": "Phone accelerometer Y-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "phone_accel_z",
                "description": "Phone accelerometer Z-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "phone_gyro_x",
                "description": "Phone gyroscope X-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "phone_gyro_y",
                "description": "Phone gyroscope Y-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "phone_gyro_z",
                "description": "Phone gyroscope Z-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_accel_x",
                "description": "Smartwatch accelerometer X-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_accel_y",
                "description": "Smartwatch accelerometer Y-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_accel_z",
                "description": "Smartwatch accelerometer Z-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_gyro_x",
                "description": "Smartwatch gyroscope X-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_gyro_y",
                "description": "Smartwatch gyroscope Y-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_gyro_z",
                "description": "Smartwatch gyroscope Z-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            }
        ]
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created manifest: {manifest_path}")


def main():
    """Convert WISDM to standardized format."""
    print("=" * 80)
    print("WISDM → Standardized Format Converter")
    print("=" * 80)

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Run: python -m data.scripts.download_datasets wisdm")
        return

    # Create output directory; wipe any prior sessions (old accel-only/gyro-only layout must not
    # linger — stale `phone_accel_*` sessions would still route into the merged 6-ch stream).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    if (OUTPUT_DIR / "sessions").exists():
        shutil.rmtree(OUTPUT_DIR / "sessions")

    # Merge and segment all data
    sessions = merge_device_sensors()
    print(f"\nCreated {len(sessions)} sessions")

    # Save sessions and labels
    print("\nSaving sessions...")
    labels_dict = save_sessions(sessions)

    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(labels_dict, f, indent=2)

    print(f"\n✓ Created labels: {labels_path}")
    print(f"  Total sessions: {len(labels_dict)}")

    # Create manifest
    create_manifest()
    metadata_path = OUTPUT_DIR / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["num_sessions"] = len(labels_dict)
        metadata.pop("num_sessions_subsampled", None)
        metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(labels_dict)} sessions")
    print("  - 12 channels (phone + watch, accel + gyro)")
    print("  - 20 Hz sampling rate")
    print("\nNOTE: Each session contains data from ONE device-sensor combination.")
    print("      To use multi-modal data, merge sessions by subject and activity.")

    # Generate debug visualizations
    try:
        from data.scripts.debug.visualization_utils import generate_debug_visualizations

        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError as e:
        print(f"\n⚠ Could not generate visualizations: {e}")
        print("Install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
