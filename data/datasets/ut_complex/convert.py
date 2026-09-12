"""Convert UT-Complex (University of Twente complex activities) into HALO session format.

Shoaib et al. 2016, "Complex Human Activity Recognition Using Smartphone and Wrist-Worn Motion
Sensors" (Sensors 16(4):426). 10 participants, 13 activities (incl. hand-gesture activities:
typing, writing, drinking coffee, giving a talk, smoking, eating). Two Samsung Galaxy S2 phones
were carried in the RIGHT POCKET and on the RIGHT WRIST (emulating a smartwatch).

We keep the wrist-mounted-phone stream only (ut_complex/watch_wrist) because hand-gesture
activities need a wrist placement.  It is a phone proxy, not a genuine smartwatch stream; that
distinction lives in the deployment policy's ``watch_proxy`` device family.
Accelerometer is the raw (gravity-present) accelerometer, in m/s^2 (|acc| ~ 9.8); gyro in rad/s.

Source: https://www.utwente.nl/en/eemcs/ps/research/dataset/  (ut-data-complex.rar)
File layout (no header), per Readme.txt:
  timestamp, acc(x,y,z), linear_acc(x,y,z), gyro(x,y,z), mag(x,y,z), activity_code

Each activity is stored as one contiguous 30-min block = the 10 participants' 3-min recordings
concatenated with no per-row subject id. We recover subjects by splitting each activity block into
10 equal contiguous chunks (participant order is consistent across activity blocks), which yields
clean per-subject sessions. build_grids does the fixed 6 s windowing.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DS_DIR = Path(__file__).resolve().parent
RAW_CSV = DS_DIR / "downloads" / "extracted" / "UT_Data_Complex" / "smartphoneatwrist.csv"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

SAMPLING_RATE = 50  # Hz
N_SUBJECTS = 10

# activity code (Readme.txt) -> native label, cleaned/underscored.
ACTIVITY_MAP = {
    11111: "walking",
    11112: "standing",
    11113: "jogging",
    11114: "sitting",
    11115: "biking",
    11116: "walking_upstairs",
    11117: "walking_downstairs",
    11118: "typing",
    11119: "writing",
    11120: "drinking_coffee",
    11121: "talking",
    11122: "smoking",
    11123: "eating",
}


def main():
    if not RAW_CSV.exists():
        print(f"ERROR: raw wrist CSV not found at {RAW_CSV}")
        print("Download ut-data-complex.rar from the UTwente PS dataset page and extract into downloads/.")
        return 1

    df = pd.read_csv(RAW_CSV, header=None)
    # cols: 0 ts, 1-3 acc, 4-6 linacc, 7-9 gyro, 10-12 mag, 13 label
    acc = df.iloc[:, 1:4].to_numpy(dtype=np.float64)
    gyro = df.iloc[:, 7:10].to_numpy(dtype=np.float64)
    lab = df.iloc[:, 13].to_numpy()

    sessions_dir = DS_DIR / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # contiguous single-activity runs (one per activity code).
    change = np.where(np.diff(lab) != 0)[0] + 1
    bounds = [0, *change.tolist(), len(lab)]

    labels = {}
    n_sessions = 0
    subjects = set()
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        code = int(lab[b0])
        if code not in ACTIVITY_MAP:
            continue
        activity = ACTIVITY_MAP[code]
        # split the block into N_SUBJECTS equal contiguous chunks = one recording per participant.
        for k, sl in enumerate(np.array_split(np.arange(b0, b1), N_SUBJECTS)):
            if len(sl) < 10:
                continue
            a, g = acc[sl], gyro[sl]
            n = len(a)
            out = pd.DataFrame({
                "timestamp_sec": np.arange(n) / SAMPLING_RATE,
                "acc_x": a[:, 0], "acc_y": a[:, 1], "acc_z": a[:, 2],
                "gyro_x": g[:, 0], "gyro_y": g[:, 1], "gyro_z": g[:, 2],
                "subject": f"s{k:02d}",
            })
            session_id = f"s{k:02d}_{activity}"
            (sessions_dir / session_id).mkdir(parents=True, exist_ok=True)
            out.to_parquet(sessions_dir / session_id / "data.parquet", index=False)
            labels[session_id] = [activity]
            subjects.add(k)
            n_sessions += 1

    (DS_DIR / "labels.json").write_text(json.dumps(labels, indent=2))
    acts = sorted({v[0] for v in labels.values()})
    print(f"UT-Complex: {n_sessions} sessions, {len(subjects)} subjects, {len(acts)} activities")
    print("  activities:", acts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
