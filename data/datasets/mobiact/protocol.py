"""Frozen MobiAct prospective-evaluation protocol helpers."""

from __future__ import annotations

import json
from pathlib import Path

PROTOCOL_PATH = Path(__file__).resolve().parent / "eval_protocol.json"


def load_protocol(*, require_ready: bool = True) -> dict:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(
            f"missing {PROTOCOL_PATH}; run `python -m data.datasets.mobiact.setup --archive ...`"
        )
    protocol = json.loads(PROTOCOL_PATH.read_text())
    required = {"protocol_version", "release_id", "reference_subjects", "query_subjects"}
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"MobiAct protocol is missing fields: {missing}")
    if set(protocol["reference_subjects"]) & set(protocol["query_subjects"]):
        raise ValueError("MobiAct reference/query subject partitions overlap")
    if require_ready and protocol.get("status") != "ready":
        raise RuntimeError("MobiAct protocol is not finalized; run the complete setup workflow")
    return protocol


def candidate_labels(window_seconds: float) -> list[str]:
    protocol = load_protocol()
    key = f"{float(window_seconds):g}"
    labels = protocol.get("candidate_labels_by_window_seconds", {}).get(key)
    if not labels:
        raise ValueError(f"MobiAct has no registered candidate panel for {key}-second windows")
    return list(labels)


def partition_rows(stream, partition: str) -> list[int]:
    if partition not in {"reference", "query"}:
        raise ValueError("partition must be 'reference' or 'query'")
    protocol = load_protocol()
    allowed = set(protocol[f"{partition}_subjects"])
    return [index for index, subject in enumerate(map(str, stream.subjects)) if subject in allowed]
