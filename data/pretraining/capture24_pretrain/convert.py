"""Build a continuous, annotation-free Capture-24 session view for JEPA.

The labelled Capture-24 converter deliberately splits each participant into contiguous activity
runs. That is correct for classification but inappropriate for predictive self-supervision: it
would use the annotation stream to suppress genuine transitions. This adapter reads only
``time, x, y, z`` from the original participant CSV files. It emits one packed Parquet per
participant, splitting only invalid timestamps and clock discontinuities into ``segment_id``.
``build_grids.iter_logical_segments`` respects those boundaries without ever seeing activities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.pretraining.capture24_pretrain.fetch import SOURCE_ROOT, validate_source

REPO = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO / "data" / "pretraining" / "capture24_pretrain"
SESSIONS_DIRNAME = "sessions"
UNLABELED = "__unlabeled__"
RATE_HZ = 100.0
OUTPUT_COLUMNS = ("timestamp_sec", "acc_x", "acc_y", "acc_z", "subject", "segment_id")
# The recorder's nominal cadence is 10 ms. A quarter-second gap is a deliberately generous
# boundary: ordinary cadence jitter remains continuous, while an actual missing span cannot be
# interpolated across by downstream resampling.
GAP_SECONDS = 0.25
DEFAULT_CHUNK_ROWS = 250_000


def _timestamp_seconds(series: pd.Series) -> np.ndarray:
    # Most participants use one ISO layout, but a small number mix fractional-second layouts
    # within the same raw file. ``mixed`` preserves both without falling back to slow,
    # warning-producing elementwise dateutil parsing.
    parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    # datetime64 NaT has int64.min. Convert it to NaN before differencing.
    values = parsed.astype("int64", copy=False).to_numpy(dtype=np.float64) / 1e9
    values[values < 0] = np.nan
    return values


def _chunk_parts(
    chunk: pd.DataFrame,
    *,
    previous_time: float | None,
    segment_id: int,
    segment_origin: float | None,
    subject: str,
) -> tuple[list[pd.DataFrame], float | None, int, float | None]:
    """Convert one CSV chunk without joining over invalid or discontinuous timestamps."""
    timestamps = _timestamp_seconds(chunk["time"])
    xyz = chunk[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
    valid = np.isfinite(timestamps) & np.isfinite(xyz).all(axis=1)
    parts: list[pd.DataFrame] = []
    start = 0
    n = len(chunk)

    while start < n:
        while start < n and not valid[start]:
            previous_time = None
            segment_origin = None
            start += 1
        if start >= n:
            break
        stop = start + 1
        while stop < n and valid[stop]:
            stop += 1
        times = timestamps[start:stop]
        values = xyz[start:stop]
        boundaries = np.zeros(len(times), dtype=bool)
        if previous_time is None or times[0] - previous_time <= 0 or times[0] - previous_time > GAP_SECONDS:
            boundaries[0] = True
        if len(times) > 1:
            delta = np.diff(times)
            boundaries[1:] = (delta <= 0) | (delta > GAP_SECONDS)
        # Index zero is always a piece start. ``boundaries[0]`` only says whether this
        # piece begins a NEW clock segment or continues the one carried from the prior CSV
        # chunk; omitting zero here would silently drop every continuation chunk.
        indices = np.r_[0, np.flatnonzero(boundaries[1:]) + 1, len(times)]
        for begin, end in zip(indices[:-1], indices[1:]):
            if boundaries[begin]:
                segment_id += 1
                segment_origin = float(times[begin])
            assert segment_origin is not None
            frame = pd.DataFrame(
                {
                    "timestamp_sec": times[begin:end] - segment_origin,
                    "acc_x": values[begin:end, 0],
                    "acc_y": values[begin:end, 1],
                    "acc_z": values[begin:end, 2],
                    "subject": subject,
                    "segment_id": segment_id,
                }
            )
            parts.append(frame)
        previous_time = float(times[-1])
        start = stop
    return parts, previous_time, segment_id, segment_origin


def _convert_participant(source: Path, destination: Path, *, chunk_rows: int) -> dict[str, int | float]:
    """Write one participant atomically and return compact manifest statistics."""
    subject = source.name.split(".")[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.part")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    segments = 0
    previous_time: float | None = None
    segment_origin: float | None = None
    segment_id = -1
    try:
        for chunk in pd.read_csv(
            source,
            usecols=["time", "x", "y", "z"],
            chunksize=chunk_rows,
            dtype={"x": np.float32, "y": np.float32, "z": np.float32},
        ):
            parts, previous_time, segment_id, segment_origin = _chunk_parts(
                chunk,
                previous_time=previous_time,
                segment_id=segment_id,
                segment_origin=segment_origin,
                subject=subject,
            )
            for frame in parts:
                table = pa.Table.from_pandas(frame.loc[:, OUTPUT_COLUMNS], preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                rows += len(frame)
        if writer is None:
            raise ValueError(f"{source}: no finite time/x/y/z samples")
        writer.close()
        writer = None
        temporary.replace(destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    segments = segment_id + 1
    return {"rows": rows, "segments": segments, "seconds": rows / RATE_HZ}


def convert(
    *,
    source_root: Path = SOURCE_ROOT,
    output_root: Path = OUTPUT_ROOT,
    limit: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, object]:
    """Convert every available participant without reading Capture-24 activity annotations."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    sources = validate_source(source_root)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when provided")
        sources = sources[:limit]
    sessions_root = output_root / SESSIONS_DIRNAME
    sessions_root.mkdir(parents=True, exist_ok=True)

    previous_manifest_path = output_root / "manifest.json"
    previous_stats: dict[str, dict[str, int | float]] = {}
    if previous_manifest_path.is_file():
        try:
            previous_stats = json.loads(previous_manifest_path.read_text()).get("session_stats", {})
        except json.JSONDecodeError:
            # A half-written or user-edited manifest must not make valid existing Parquets
            # unusable. It is overwritten below from the authoritative output files.
            previous_stats = {}
    completed: dict[str, dict[str, int | float]] = {}
    for index, source in enumerate(sources, start=1):
        subject = source.name.split(".")[0]
        session_id = f"capture24_{subject}_watch_wrist"
        parquet_path = sessions_root / session_id / "data.parquet"
        if parquet_path.is_file():
            metadata = pq.ParquetFile(parquet_path).metadata
            completed[session_id] = previous_stats.get(session_id, {
                "rows": int(metadata.num_rows),
                "segments": -1,
                "seconds": float(metadata.num_rows) / RATE_HZ,
            })
            print(f"  [{index:3d}/{len(sources)}] {subject}: already converted", flush=True)
            continue
        stats = _convert_participant(source, parquet_path, chunk_rows=chunk_rows)
        completed[session_id] = stats
        print(
            f"  [{index:3d}/{len(sources)}] {subject}: {stats['rows']:,} samples, "
            f"{stats['segments']} clock segments",
            flush=True,
        )

    labels = {session_id: [UNLABELED] for session_id in sorted(completed)}
    (output_root / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    total_rows = sum(int(stats["rows"]) for stats in completed.values())
    known_segments = [int(stats["segments"]) for stats in completed.values() if int(stats["segments"]) >= 0]
    manifest = {
        "dataset": "capture24_pretrain",
        "source": "Capture-24 raw participant CSV files",
        "source_root": str(source_root),
        "activity_labels_used": False,
        "sampling_rate_hz": RATE_HZ,
        "unit": "g",
        "gravity_state": "present",
        "num_subjects": len(completed),
        "num_sessions": len(completed),
        "num_clock_segments": sum(known_segments) if len(known_segments) == len(completed) else None,
        "num_samples": total_rows,
        "stream_hours": total_rows / RATE_HZ / 3600.0,
        "completed_subjects": [source.name.split(".")[0] for source in sources],
        "session_stats": completed,
        "converter_version": 1,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    args = parser.parse_args()
    manifest = convert(
        source_root=args.source_root,
        output_root=args.output_root,
        limit=args.limit,
        chunk_rows=args.chunk_rows,
    )
    print(
        f"Capture-24 label-free conversion complete: {manifest['num_subjects']} subjects, "
        f"{manifest['stream_hours']:.1f} stream-hours."
    )


if __name__ == "__main__":
    main()
