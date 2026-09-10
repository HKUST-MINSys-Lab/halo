import random

import numpy as np
import pytest
import torch

from applications.motion_monitoring.batching import (
    central_duration_cap,
    nearest_length_indices,
    padding_fraction,
    plan_duration,
)
from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)
from applications.motion_monitoring.evaluation_manifests import Task1EvaluationUnit
from applications.motion_monitoring.representation_cache import bounded_representation_id
from applications.motion_monitoring.sequence import MotionSequence
from applications.motion_monitoring.task1.train_full import _training_episode


def test_duration_plan_draws_from_central_observed_lengths():
    rng = random.Random(3)
    cap = central_duration_cap([1.0, 3.0, 7.0, 100.0], rng)
    assert cap in {3.0, 7.0}
    assert plan_duration([2.0, 3.0], complete_probability=1.0, rng=rng).mode == "complete"
    cropped = plan_duration([2.0, 3.0], complete_probability=0.0, rng=rng)
    assert cropped.mode == "cropped"
    assert cropped.cap_seconds == 3.0


def test_padding_fraction_is_zero_for_equal_lengths_and_bounded_otherwise():
    assert padding_fraction([5, 5, 5]) == 0.0
    assert padding_fraction([2, 5]) == pytest.approx(0.3)


def test_nearest_length_selection_prefers_the_requested_duration():
    selected = nearest_length_indices(
        [1.0, 4.0, 5.0, 20.0], count=2, target_seconds=4.5, rng=random.Random(9)
    )
    assert {selected[0], selected[1]} == {1, 2}


def _recording(recording_id: str, event: EventInterval) -> RawRecording:
    timestamps = np.arange(10, dtype=np.float64)
    values = np.stack((timestamps, timestamps + 1, timestamps + 2), axis=1).astype(np.float32)
    stream = SensorStream(
        stream_id="wrist_acc",
        placement="wrist",
        device="watch",
        timestamps_sec=timestamps,
        values=values,
        channels=("acc_x", "acc_y", "acc_z"),
        valid=np.ones_like(values, dtype=bool),
        gravity_state="present",
        nominal_rate_hz=1.0,
    )
    return RawRecording(
        dataset="unit",
        recording_id=recording_id,
        subject_id=recording_id,
        session_id=recording_id,
        streams=(stream,),
        events=(event,),
    )


def _sequence(recording: RawRecording) -> MotionSequence:
    count = 10
    embeddings = torch.nn.functional.normalize(torch.arange(1, count * 4 + 1).reshape(count, 4).float(), dim=-1)
    intervals = torch.column_stack((torch.arange(count), torch.arange(1, count + 1))).double()
    physical = torch.zeros(count, 1)
    return MotionSequence(
        embeddings=embeddings,
        intervals_sec=intervals,
        valid=torch.ones(count, dtype=torch.bool),
        physical_features=physical,
        physical_feature_mask=torch.ones_like(physical, dtype=torch.bool),
        physical_feature_names=("valid_fraction",),
        dataset=recording.dataset,
        recording_id=recording.recording_id,
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        stream_id="wrist_acc",
        placement="wrist",
        device="watch",
        channels=("acc_x", "acc_y", "acc_z"),
        gravity_state="present",
        sampling_rate_hz=1.0,
    )


class _Representations:
    def __init__(self, rows):
        self.rows = rows

    def get(self, dataset, recording_id, stream_id):
        return self.rows[(dataset, recording_id, stream_id)]


def test_task1_training_crop_retains_complete_target_and_marks_view():
    reference = _recording("reference", EventInterval(1.0, 4.0, "pour"))
    query = _recording("query", EventInterval(4.0, 6.0, "pour"))
    rows = {
        ("unit", "query", "wrist_acc"): _sequence(query),
        ("unit", bounded_representation_id("reference", 0), "wrist_acc"): _sequence(reference),
    }
    unit = Task1EvaluationUnit(
        dataset="unit",
        query_cache_index=0,
        query_recording_id="query",
        query_subject_id="query",
        query_stream_id="wrist_acc",
        reference_cache_index=1,
        reference_recording_id="reference",
        reference_subject_id="reference",
        reference_stream_id="wrist_acc",
        reference_event_index=0,
        label="pour",
        target_intervals_sec=((4.0, 6.0),),
        target_present=True,
        reference_interval_sec=(1.0, 4.0),
        reference_rule="unit",
    )
    episode = _training_episode(
        unit, [query, reference], _Representations(rows), seconds=5.0, rng=random.Random(1)
    )
    assert episode.metadata["query_view_mode"] == "cropped"
    assert episode.metadata["query_retained_duration_sec"] == 5.0
    assert episode.targets_sec.tolist() == [[4.0, 6.0]]
    assert episode.query.intervals_sec[0, 0] <= 4.0
    assert episode.query.intervals_sec[-1, 1] >= 6.0
