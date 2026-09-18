"""Publication-split helpers for MM-Fit scenario evaluation."""

from __future__ import annotations

import json
from pathlib import Path

PROTOCOL_PATH = Path(__file__).resolve().parent / "eval_protocol.json"


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if not {"protocol_version", "reference_workouts", "query_workouts"} <= set(protocol):
        raise ValueError("MM-Fit scenario protocol is incomplete")
    if set(protocol["reference_workouts"]) & set(protocol["query_workouts"]):
        raise ValueError("MM-Fit reference/query workout sets overlap")
    return protocol


def partition_rows(stream, partition: str) -> list[int]:
    key = {
        "reference": "reference_workouts",
        "query": "query_workouts",
        "seen_person_test": "seen_person_test_workouts",
    }.get(partition)
    if key is None:
        raise ValueError("unknown MM-Fit partition")
    workouts = {f"w{int(value):02d}" for value in load_protocol()[key]}
    rows = [index for index, subject in enumerate(map(str, stream.subjects)) if subject in workouts]
    if not rows:
        raise ValueError(f"MM-Fit {partition} partition has no surviving rows")
    return rows
