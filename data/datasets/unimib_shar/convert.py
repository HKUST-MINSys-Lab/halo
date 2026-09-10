"""
Convert UniMiB SHAR dataset to standardized format.

Input:  data/datasets/unimib_shar/downloads/UniMiB/  (the ORIGINAL .npy release)
Output: data/datasets/unimib_shar/{sessions/,labels.json,manifest.json}

UniMiB SHAR: 30 subjects, 17 activities (9 ADL + 8 falls), accelerometer ONLY, 50 Hz.
Pre-windowed: each distributed segment is a fixed 151-sample window (metadata `pre_windowed: true`).

IMPORTANT (fixed 2026-07-12): we read the RAW `UniMiB/acc_data.npy` (physical **m/s²**, gravity
present) + `acc_labels.npy`, NOT the Kaggle `unimib_*.csv`. The Kaggle CSVs are per-axis **z-score
normalized** (mean 0, std 1) — using them destroys the DC/gravity component and anisotropically warps
each axis, feeding a physically meaningless signal to a gravity-present-g pipeline. The raw `.npy` also
carries the real subject ids (lost in the CSV), enabling subject-disjoint splits.

Raw layout:
  acc_data:   (N, 453) float64 — each row is [ax(151), ay(151), az(151)] in m/s².
  acc_labels: (N, 3)   uint8   — columns [activity_id (1-17), subject_id (1-30), trial_id].
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# Activity mapping (acc_labels column 0, 1-indexed, 17 activities: 9 ADL + 8 falls)
ACTIVITIES = {
    1: 'standing_up_from_sitting', 2: 'standing_up_from_laying', 3: 'walking', 4: 'running',
    5: 'going_up_stairs', 6: 'jumping', 7: 'going_down_stairs', 8: 'lying_down_from_standing',
    9: 'sitting_down', 10: 'falling_forward', 11: 'falling_right', 12: 'falling_backward',
    13: 'falling_hitting_obstacle', 14: 'falling_with_protection', 15: 'falling_backward_sitting',
    16: 'syncope', 17: 'falling_left',
}

# Paths (new repo layout: this converter lives in data/datasets/unimib_shar/)
DS_DIR = Path(__file__).resolve().parent
RAW_NPY_DIR = DS_DIR / "downloads" / "UniMiB"
OUTPUT_DIR = DS_DIR

WINDOW_LEN = 151      # samples per distributed segment (pre-windowed)
SAMPLING_RATE_HZ = 50.0


def load_raw():
    """Load the raw UniMiB .npy release: (acc_data (N,453) m/s², acc_labels (N,3))."""
    # Plain numeric arrays (float64 / uint8) from the public UniMiB release — no pickled objects,
    # so allow_pickle stays False (default) for safety.
    acc_data = np.load(RAW_NPY_DIR / "acc_data.npy").astype(np.float64)
    acc_labels = np.load(RAW_NPY_DIR / "acc_labels.npy")
    if acc_data.shape[1] != 3 * WINDOW_LEN:
        raise ValueError(f"unexpected acc_data width {acc_data.shape[1]} (expected {3 * WINDOW_LEN})")
    print(f"Loaded raw UniMiB: {acc_data.shape[0]} windows, {acc_labels.shape} labels")
    return acc_data, acc_labels


def process(acc_data, acc_labels):
    """Each row → one pre-windowed session (151 samples) with acc_x/y/z (m/s²) + subject."""
    sessions_dir = OUTPUT_DIR / "sessions"
    shutil.rmtree(sessions_dir, ignore_errors=True)   # clear stale (e.g. old CSV-based) sessions
    sessions_dir.mkdir(parents=True, exist_ok=True)
    labels_dict = {}
    recordings = {}
    subjects = set()

    for i in range(acc_data.shape[0]):
        activity_id = int(acc_labels[i][0])
        subject_id = int(acc_labels[i][1])
        trial_id = int(acc_labels[i][2])
        if activity_id not in ACTIVITIES:
            continue
        activity_name = ACTIVITIES[activity_id]

        # Block layout [ax(151), ay(151), az(151)] -> (151, 3)
        w = acc_data[i].reshape(3, WINDOW_LEN).T
        frame = pd.DataFrame({
            "timestamp_sec": np.arange(WINDOW_LEN) / SAMPLING_RATE_HZ,
            "acc_x": w[:, 0], "acc_y": w[:, 1], "acc_z": w[:, 2],
            "subject": f"subject{subject_id:02d}",   # real subject id for subject-disjoint splits
        })
        if not np.isfinite(frame[["acc_x", "acc_y", "acc_z"]].to_numpy()).all():
            continue

        session_id = f"subject{subject_id:02d}_act{activity_id:02d}_{i:05d}"
        session_dir = sessions_dir / session_id
        session_dir.mkdir(exist_ok=True)
        frame.to_parquet(session_dir / "data.parquet", index=False)
        labels_dict[session_id] = [activity_name]
        # Distributed windows from one subject/activity/trial heavily overlap. The release's trial
        # id is therefore the physical execution boundary used by enrollment leakage guards.
        recordings[session_id] = f"subject{subject_id:02d}_act{activity_id:02d}_trial{trial_id:02d}"
        subjects.add(subject_id)

    (OUTPUT_DIR / "labels.json").write_text(json.dumps(labels_dict, indent=2))
    (OUTPUT_DIR / "recordings.json").write_text(json.dumps(recordings, indent=2))
    print(f"Created {len(labels_dict)} sessions across {len(subjects)} subjects")
    return labels_dict, sorted(subjects)


def create_manifest():
    manifest = {
        "dataset_name": "UniMiB SHAR",
        "description": ("Smartphone accelerometer HAR. 30 subjects, 17 activities (9 ADL + 8 falls), "
                        "50 Hz, accelerometer only (no gyroscope). Raw units m/s² (gravity present)."),
        "channels": [{"name": c, "description": f"Accelerometer {c[-1].upper()}-axis (m/s²)",
                      "sampling_rate_hz": SAMPLING_RATE_HZ} for c in ("acc_x", "acc_y", "acc_z")],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("✓ Created manifest.json (accelerometer only, m/s²)")


def main():
    print("=" * 80)
    print("UniMiB SHAR → Standardized Format Converter (raw .npy, m/s², real subjects)")
    print("=" * 80)
    if not (RAW_NPY_DIR / "acc_data.npy").exists():
        print(f"ERROR: raw UniMiB .npy not found at {RAW_NPY_DIR} "
              "(expected acc_data.npy + acc_labels.npy from the Kaggle download's UniMiB/ folder)")
        return
    acc_data, acc_labels = load_raw()
    labels_dict, subjects = process(acc_data, acc_labels)
    if not labels_dict:
        print("ERROR: no sessions created."); return
    create_manifest()
    acts = {}
    for v in labels_dict.values():
        acts[v[0]] = acts.get(v[0], 0) + 1
    print(f"\nDone: {len(labels_dict)} sessions, {len(subjects)} subjects, {len(acts)} activities.")


if __name__ == "__main__":
    main()
