"""Integration tests for the grid-build orchestrator's assembly core (`stream_grid`).

Uses synthetic in-memory sessions, so it exercises the accumulation + canonicalization + both
alignments without needing converted parquet on disk.
"""

import numpy as np
import pandas as pd

from data.scripts.build_grids import _greedy_class_cap, iter_logical_segments, stream_grid
from data.scripts.curate.accel_units import GRAVITY_MS2
from data.scripts.curate.deployment_policy import all_source_channels, get_stream_spec


def _session_frame(dataset, spec, n, activity, acc=GRAVITY_MS2, gyro=0.3):
    cols = {"timestamp_sec": np.arange(n, dtype=float) / 50.0}
    for s in all_source_channels(dataset, role=spec.role):
        cols[s] = np.full(n, acc if "acc" in s else (gyro if "gyro" in s else 1.0), dtype=float)
    cols["activity"] = [activity] * n
    return pd.DataFrame(cols)


def test_stream_grid_harmonised_stacks_resamples_and_canonicalizes():
    spec = get_stream_spec("hhar", "phone_waist")
    sessions = [  # (frame, native_rate_hz, subject)
        (_session_frame("hhar", spec, 300, "laying"), 50.0, "s1"),   # laying -> lying (canonical)
        (_session_frame("hhar", spec, 300, "walking"), 50.0, "s2"),
    ]
    grid, subjects = stream_grid("hhar", spec, sessions, alignment="harmonised",
                                 resample_to=60.0, canonical_labels=True, view="harmonised")
    assert grid.rate_hz == 60.0                                       # resampled
    assert grid.channels == ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    assert grid.data.shape == (2, 360, 6)                            # 300@50Hz -> 360@60Hz, one 6 s window each
    assert np.allclose(grid.data[:, 100:200, :3], 1.0, atol=0.05)    # m/s^2 -> g, gravity preserved
    assert set(grid.labels) == {"lying", "walking"}                 # 'laying' canonicalized to 'lying'
    assert len(subjects) == len(grid.data) and set(subjects) == {"s1", "s2"}


def test_stream_grid_persists_session_event_identity():
    spec = get_stream_spec("hhar", "phone_waist")
    frame = _session_frame("hhar", spec, 600, "walking")
    frame.attrs["halo_session_id"] = "subject1_recording7"
    grid, _ = stream_grid(
        "hhar", spec, [(frame, 50.0, "s1")], alignment="native",
        resample_to=None, canonical_labels=True, view="harmonised",
    )
    assert grid.event_ids == [
        "hhar:subject1_recording7:0",
        "hhar:subject1_recording7:1",
    ]


def test_aggregated_segments_never_form_a_cross_boundary_window():
    spec = get_stream_spec("hhar", "phone_waist")
    first = _session_frame("hhar", spec, 200, "walking", acc=GRAVITY_MS2)
    second = _session_frame("hhar", spec, 200, "walking", acc=2 * GRAVITY_MS2)
    frame = pd.concat([first, second], ignore_index=True)
    frame["segment_id"] = np.repeat([10, 11], 200)
    frame.attrs["halo_session_id"] = "subject1_aggregate"
    logical = [(part, 50.0, "s1") for part in iter_logical_segments(frame)]

    grid, subjects = stream_grid(
        "hhar", spec, logical, alignment="native", resample_to=None,
        canonical_labels=True, view="harmonised",
    )

    # Each 4-second capture becomes one honest partial window. Without segment-aware
    # assembly, the 400 concatenated rows would create a 6-second window mixing captures.
    assert grid.lengths.tolist() == [200, 200]
    assert grid.data.shape == (2, 300, 6)
    assert np.allclose(grid.data[0, :200, :3], 1.0)
    assert np.allclose(grid.data[1, :200, :3], 2.0)
    assert np.count_nonzero(grid.data[:, 200:, :]) == 0
    assert subjects == ["s1", "s1"]
    assert grid.event_ids == [
        "hhar:subject1_aggregate:segment:10:0",
        "hhar:subject1_aggregate:segment:11:0",
    ]


def test_aggregated_segment_id_cannot_reappear_later():
    frame = pd.DataFrame({"acc_x": [0.0, 1.0, 2.0], "segment_id": [0, 1, 0]})
    with np.testing.assert_raises_regex(ValueError, "contiguous run"):
        list(iter_logical_segments(frame))


def test_native_grid_retains_final_partial_context_with_honest_length():
    spec = get_stream_spec("hhar", "phone_waist")
    frame = _session_frame("hhar", spec, 425, "walking")  # 8.5 seconds at 50 Hz
    grid, subjects = stream_grid(
        "hhar", spec, [(frame, 50.0, "s1")], alignment="native",
        resample_to=None, canonical_labels=True, view="harmonised",
    )
    assert grid.data.shape == (2, 300, 6)
    assert grid.lengths.tolist() == [300, 125]
    assert np.count_nonzero(grid.data[1, 125:]) == 0
    assert grid.labels == ["walking", "walking"]
    assert subjects == ["s1", "s1"]


def test_stream_grid_can_drop_partial_context_for_predictive_sources():
    spec = get_stream_spec("hhar", "phone_waist")
    frame = _session_frame("hhar", spec, 425, "walking")
    grid, subjects = stream_grid(
        "hhar", spec, [(frame, 50.0, "s1")], alignment="native",
        resample_to=None, canonical_labels=True, view="harmonised",
        include_partial=False,
    )
    assert grid.data.shape == (1, 300, 6)
    assert grid.lengths.tolist() == [300]
    assert subjects == ["s1"]


def test_baseline_grids_still_drop_partial_contexts():
    spec = get_stream_spec("hhar", "phone_waist")
    frame = _session_frame("hhar", spec, 425, "walking")
    grid, _ = stream_grid(
        "hhar", spec, [(frame, 50.0, "s1")], alignment="non_harmonised",
        resample_to=None, canonical_labels=False, view="non_harmonised",
    )
    assert grid.data.shape == (1, 300, 6)
    assert grid.lengths.tolist() == [300]


def test_stream_grid_prefers_shared_physical_event_over_device_session_id():
    spec = get_stream_spec("hhar", "phone_waist")
    grids = []
    for device_session in ("recording_phone", "recording_watch"):
        frame = _session_frame("hhar", spec, 300, "walking")
        frame.attrs["halo_session_id"] = device_session
        frame.attrs["halo_event_id"] = "subject1_physical_event7"
        grid, _ = stream_grid(
            "hhar", spec, [(frame, 50.0, "s1")], alignment="native",
            resample_to=None, canonical_labels=True, view="harmonised",
        )
        grids.append(grid)
    assert grids[0].event_ids == grids[1].event_ids == [
        "hhar:subject1_physical_event7:0"
    ]


def test_stream_grid_non_harmonised_keeps_native_rate_and_labels():
    spec = get_stream_spec("hhar", "phone_waist")
    sessions = [(_session_frame("hhar", spec, 120, "laying"), 50.0, "s1")]
    grid, subjects = stream_grid("hhar", spec, sessions, alignment="non_harmonised",
                                 resample_to=None, canonical_labels=False,
                                 view="non_harmonised", window_seconds=1.0)
    assert grid.rate_hz == 50.0                                      # native rate kept
    assert grid.data.shape == (2, 50, 6)                            # 120@50Hz, 1 s windows, stride=window
    assert grid.labels == ["laying", "laying"]                     # native label, NOT canonicalized


def test_stream_grid_native_keeps_rate_but_canonicalizes_and_pads_to_six():
    """The HALO 'native' regime: native RATE (no 60 Hz resample) yet canonical labels + the fixed
    6-ch pad+mask layout the tokenizer loader expects."""
    spec = get_stream_spec("hhar", "phone_waist")
    sessions = [
        (_session_frame("hhar", spec, 300, "laying"), 50.0, "s1"),   # laying -> lying (canonical)
        (_session_frame("hhar", spec, 300, "walking"), 50.0, "s2"),
    ]
    grid, subjects = stream_grid("hhar", spec, sessions, alignment="native",
                                 resample_to=None, canonical_labels=True, view="harmonised")
    assert grid.alignment == "native"
    assert grid.rate_hz == 50.0                                      # NATIVE rate preserved (not 60)
    assert grid.channels == ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    assert grid.data.shape == (2, 300, 6)                           # 300@50Hz, one 6 s window each, 6-ch
    assert set(grid.labels) == {"lying", "walking"}                 # canonicalized like harmonised
    assert len(subjects) == len(grid.data) and set(subjects) == {"s1", "s2"}


def test_stream_grid_acc_only_dataset_harmonised_pads_gyro():
    spec = get_stream_spec("capture24", "watch_wrist")
    sessions = [(_session_frame("capture24", spec, 600, "walking", acc=1.0), 100.0, "p1")]
    grid, _ = stream_grid("capture24", spec, sessions, alignment="harmonised",
                          resample_to=60.0, canonical_labels=True, view="harmonised")
    assert grid.channels == ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    assert list(grid.mask) == [True, True, True, False, False, False]
    assert np.count_nonzero(grid.data[..., 3:]) == 0                # gyro zero-padded, never fabricated


def test_stream_grid_native_acc_only_pads_gyro_at_native_rate():
    """Native regime on an accel-only stream: native 100 Hz kept, gyro slots zero-padded + masked."""
    spec = get_stream_spec("capture24", "watch_wrist")
    sessions = [(_session_frame("capture24", spec, 600, "walking", acc=1.0), 100.0, "p1")]
    grid, _ = stream_grid("capture24", spec, sessions, alignment="native",
                          resample_to=None, canonical_labels=True, view="harmonised")
    assert grid.rate_hz == 100.0                                    # native 100 Hz kept (not 60)
    assert grid.data.shape == (1, 600, 6)                           # 600@100Hz = 6 s, 6-ch pad
    assert list(grid.mask) == [True, True, True, False, False, False]
    assert np.count_nonzero(grid.data[..., 3:]) == 0


def test_greedy_class_cap_balances_and_keeps_rare_classes():
    per_class = {
        "sitting": [(f"sit_{i}", 3600.0) for i in range(10)],       # 10 h, each 1 h
        "sports":  [("sp_0", 1800.0), ("sp_1", 1800.0)],            # 1 h total, under the cap
    }
    keep = _greedy_class_cap(per_class, max_hours=2.0)              # 2 h/class
    assert len([s for s in keep if s.startswith("sit_")]) == 2      # capped to 2 h
    assert {"sp_0", "sp_1"} <= keep                                 # rare class kept whole (under cap)


def test_greedy_class_cap_always_keeps_at_least_one():
    per_class = {"vehicle": [("v0", 1e9), ("v1", 1e9)]}             # first session already exceeds cap
    keep = _greedy_class_cap(per_class, max_hours=1.0)
    assert len(keep) == 1                                          # never drops a class to zero, but stops
