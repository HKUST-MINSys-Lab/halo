"""Publication-backed semantic contracts for the active KU-HAR converter."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.datasets.kuhar.convert import ACTIVITIES, split_clock_runs


def test_repeated_and_circular_protocols_are_not_collapsed():
    # The release describes Stand-sit and Lay-stand as five repeated, bidirectional
    # executions and Walk-circle as a circular path. None is a one-way/plain class.
    assert ACTIVITIES["Stand-sit"] == "repeated_standing_and_sitting"
    assert ACTIVITIES["Lay-stand"] == "repeated_standing_and_lying"
    assert ACTIVITIES["Walk-circle"] == "walking_in_circles"
    assert len(set(ACTIVITIES.values())) == 18


def test_checked_in_metadata_declares_the_converter_vocabulary():
    path = Path(__file__).resolve().parents[1] / "data/datasets/kuhar/metadata.json"
    metadata = json.loads(path.read_text())
    # Clock-contiguous source sessions: 1,945 source files plus 507 genuine post-seam runs.
    assert metadata["num_sessions"] == 2452
    assert set(metadata["activities"]) == set(ACTIVITIES.values())


def test_clock_split_ignores_millisecond_jitter_but_cuts_real_seams_and_gaps():
    clock = np.arange(40, dtype=float) / 100.0
    clock[12] -= 0.002  # observed harmless backwards jitter
    clock[20:] += 18.0  # reset/discontinuity
    clock[31:] += 0.20  # material missing interval
    frame = pd.DataFrame({
        "timestamp_sec": clock,
        "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0,
        "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
    })
    runs = split_clock_runs(frame)
    assert [len(run) for run in runs] == [20, 11, 9]
