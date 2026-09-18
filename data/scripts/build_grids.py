"""Build the corpus grids from converted sessions — the harmonisation entry point.

Run AFTER the per-dataset converters have produced `data/datasets/<ds>/sessions/*.parquet`
(+ `manifest.json` with the native rate, + `labels.json`):

    python -m data.scripts.build_grids                     # all streams: harmonised + non-harmonised

`harmonised-strict` (phones only) is NOT a separate build — it is the phone subset of the harmonised
grids, selected at training time via `deployment_streams(placement_strict=True)`.

For every device stream (`deployment_policy.deployment_streams`), each session is assembled into a
windowed `Grid`, in three regimes (`_ALIGNMENTS`):

  * **harmonised**     — resampled to 60 Hz, fixed 6-ch [acc,gyro] pad+mask, **canonical** labels.
                         The fixed-rate crutch the layout-locked baselines (CrossHAR/LiMU-BERT/harnet)
                         require.
  * **non_harmonised** — native rate, native 3/6-ch, **native** labels. The raw eval/baseline source.
  * **native**         — **native rate** (no resample), fixed 6-ch pad+mask, **canonical** labels.
                         HALO's source: the filterbank tokenizer is rate-invariant, so HALO trains on
                         the corpus's REAL native rates rather than a 60 Hz base + synthetic rate aug.

Grids are written per dataset+stream under `data/datasets/<ds>/grids/` (gitignored). The assembly
core (`stream_grid`) is unit-tested on synthetic sessions; only the disk I/O below needs real data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.scripts.assembly import baseline_view
from data.scripts.assembly.assemble import Grid, assemble, resample_signal
from data.scripts.curate.corpus_roots import dataset_root as _resolve_dataset_root
from data.scripts.curate.corpus_roots import roots_under
from data.scripts.curate.deployment_policy import (
    DIRECT_CONVERTED_TRAIN_DATASETS,
    EXPANDED_PHASE_A_TRAIN_DATASETS,
    PRIMARY_EVAL_DATASETS,
    STANDARD_CHANNEL_ORDER,
    StreamSpec,
    channel_names_for_frame,
    deployment_streams,
    session_stream_specs,
    stream_specs,
)
from data.scripts.labels.canonical_labels import canonicalize
from halo.paths import REPO_ROOT

REPO = REPO_ROOT  # compatibility alias for converter helpers
HARMONISED_RATE_HZ = 60
WINDOW_SECONDS = 6.0

# The three build regimes. (name/dir, resample target, canonical labels, channel view):
#   harmonised     — 60 Hz + canonical labels + 6-ch pad+mask. Legacy fixed-rate exports.
#                    Current CrossHAR/LiMU-BERT/HARNet adapters start from `native`, trim each
#                    partial window by `lengths.npy`, then anti-alias/resample to their contract.
#   non_harmonised — native rate + native labels + native 3/6-ch width. The raw eval/baseline source.
#   native         — native rate + canonical labels + 6-ch pad+mask. HALO's source: the filterbank
#                    tokenizer is rate-invariant, so HALO trains on REAL native rates (no 60 Hz
#                    resample) with the same canonical vocab + 6-slot layout its loader expects.
_ALIGNMENTS = (
    ("harmonised",     float(HARMONISED_RATE_HZ), True,  "harmonised"),
    ("non_harmonised", None,                      False, "non_harmonised"),
    ("native",         None,                      True,  "harmonised"),
)

# A session as the assembler needs it: the raw frame + its native rate + the subject id (for splits).
Session = Tuple[pd.DataFrame, float, object]


def stream_grid(dataset: str, spec: StreamSpec, sessions: Iterable[Session], *,
                alignment: str, resample_to: Optional[float], canonical_labels: bool,
                view: str, pre_windowed: bool = False,
                window_seconds: float = WINDOW_SECONDS,
                include_partial: Optional[bool] = None) -> Tuple[Grid, List]:
    """Assemble every session of ONE device stream into a single stacked `Grid`.

    The three build concerns are decoupled (a stream can mix them — see ``_ALIGNMENTS``):
      ``resample_to``      target rate before windowing (60 for the harmonised corpus; ``None`` keeps
                           the session's NATIVE rate — the HALO/tokenizer path, which is rate-invariant).
      ``canonical_labels`` map labels to the canonical vocabulary (harmonised + native corpora); the
                           raw ``non_harmonised`` eval view keeps native label strings.
      ``view``             channel layout passed to ``baseline_view.to_view``: ``"harmonised"`` (fixed
                           6-ch ``[acc,gyro]`` zero-pad+mask) or ``"non_harmonised"`` (native 3/6-ch).
    ``alignment`` is only the OUTPUT name/dir (``"harmonised" | "non_harmonised" | "native"``).
    ``pre_windowed=True`` → the dataset is distributed as fixed-length segments (e.g. UCI HAR's
    2.56 s windows), too short for the 6 s corpus window; treat each segment as exactly ONE window
    (window = the segment's own resampled length) instead of cutting 6 s windows from it.
    Returns ``(grid, subjects)`` where ``subjects`` is one entry per window (for subject-disjoint splits).
    """
    sessions = list(sessions)
    if include_partial is None:
        include_partial = alignment == "native"
    datas: List[np.ndarray] = []
    labels: List = []
    subjects: List = []
    event_ids: List[str] = []
    lengths: List[np.ndarray] = []
    channels: Optional[Tuple[str, ...]] = None
    mask = None
    rate_out = float(resample_to) if resample_to is not None else None

    forced_window: Optional[int] = None
    if pre_windowed and sessions:
        f0, r0, _ = sessions[0]
        n0 = len(f0)
        # The segment's length after (optional) resampling — one whole window, no truncation.
        forced_window = (len(resample_signal(np.zeros((n0, 1), np.float32), r0, resample_to))
                         if resample_to is not None else n0)

    for frame, native_rate, subject in sessions:
        out_rate = resample_to if resample_to is not None else native_rate
        window = forced_window if forced_window is not None else max(1, round(window_seconds * out_rate))
        g = assemble(frame, dataset, spec, alignment=view, window=window,
                     rate_hz=native_rate, resample_to=resample_to,
                     include_partial=include_partial)
        if len(g.data) == 0:
            continue
        datas.append(g.data)
        lengths.append(g.lengths if g.lengths is not None
                       else np.full(len(g.data), g.data.shape[1], dtype=np.int32))
        channels, mask, rate_out = g.channels, g.mask, g.rate_hz
        if canonical_labels:
            labels.extend(canonicalize(l) for l in g.labels)   # unify synonyms in the canonical corpora
        else:
            labels.extend(g.labels)                            # native labels for the non-harmonised view
        subjects.extend([subject] * len(g.data))
        # Converters may map several device-specific session ids onto one verified physical event
        # through events.json. The local-window ordinal then identifies simultaneous windows within
        # that event. Fall back to the stream-local session id for ordinary single-stream datasets.
        event_id = frame.attrs.get("halo_event_id") or frame.attrs.get("halo_session_id")
        if event_id is None:
            event_id = f"{spec.stream_id}:anonymous:{len(datas) - 1}"
        event_ids.extend(f"{dataset}:{event_id}:{i}" for i in range(len(g.data)))

    if not datas:  # nothing assembled — return a well-formed empty grid at the right width
        declared = tuple(c for c in STANDARD_CHANNEL_ORDER
                         if c in (set(spec.required) | set(spec.optional or {})))
        _, out_channels, empty_mask = baseline_view.to_view(
            np.zeros((0, len(declared)), np.float32), declared, view)
        return Grid(np.zeros((0, 0, len(out_channels)), np.float32), empty_mask, out_channels, [],
                    alignment, dataset, float(resample_to) if resample_to is not None else 0.0,
                    lengths=np.empty(0, dtype=np.int32)), []
    return Grid(np.concatenate(datas, axis=0), mask, channels, labels, alignment, dataset,
                float(rate_out), event_ids, np.concatenate(lengths)), subjects


# --------------------------------------------------------------------------------------------------
# Disk I/O — reads what the converters produce. Thin by design; the logic above is what's tested.
# --------------------------------------------------------------------------------------------------


def iter_logical_segments(frame: pd.DataFrame) -> Iterable[pd.DataFrame]:
    """Yield independent logical sessions encoded inside one physical Parquet.

    Most converters store one logical session per file and therefore have no
    ``segment_id`` column. Large snippet corpora may aggregate many captures into one
    participant/stream file to avoid hundreds of thousands of tiny files. For those,
    this is the sole expansion point: grid assembly sees each segment independently,
    so resampling, fixed-window slicing, and final partial padding can never cross a
    source capture or clock-gap boundary.
    """
    if "segment_id" not in frame.columns:
        yield frame
        return
    raw_ids = frame["segment_id"].to_numpy()
    if not len(raw_ids):
        return
    try:
        numeric_ids = np.asarray(raw_ids, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("segment_id must contain finite integers") from exc
    if not np.isfinite(numeric_ids).all() or not np.equal(numeric_ids, np.floor(numeric_ids)).all():
        raise ValueError("segment_id must contain finite integers")
    segment_ids = numeric_ids.astype(np.int64)
    starts = np.r_[0, np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1]
    stops = np.r_[starts[1:], len(frame)]
    ordered_ids = segment_ids[starts]
    if len(np.unique(ordered_ids)) != len(ordered_ids):
        raise ValueError("each segment_id must occupy one contiguous run")
    source_attrs = dict(frame.attrs)
    base_session = source_attrs.get("halo_session_id")
    base_event = source_attrs.get("halo_event_id") or base_session
    for start, stop, segment_id in zip(starts, stops, ordered_ids):
        segment = frame.iloc[int(start):int(stop)].drop(columns="segment_id")
        segment.attrs.update(source_attrs)
        if base_session is not None:
            segment.attrs["halo_session_id"] = f"{base_session}:segment:{int(segment_id)}"
        if base_event is not None:
            segment.attrs["halo_event_id"] = f"{base_event}:segment:{int(segment_id)}"
        yield segment


def iter_sessions(dataset: str, spec: StreamSpec,
                  keep_ids: Optional[set] = None) -> Iterable[Session]:
    """Yield the converted sessions belonging to `spec`'s device stream.

    Expects `data/datasets/<ds>/sessions/*.parquet` + `manifest.json` (native `sampling_rate_hz`).
    For multi-stream datasets (wisdm phone/watch) sessions are routed by `session_stream_specs`.
    ``keep_ids`` (if given) restricts to that set of session ids — used by the class-balanced cap for
    huge free-living datasets (capture24) so we never load the full corpus into memory.
    """
    ds_dir = dataset_root(dataset)
    # Native rate is a dataset property (metadata.json). The session manifest stores rate per-channel;
    # a single deployment stream shares one rate, so the dataset rate is authoritative.
    native_rate = float(json.loads((ds_dir / "metadata.json").read_text())["sampling_rate_hz"])
    # Converters write per-session activity to labels.json (session_id -> [activity]); the parquet has
    # only sensor channels, so inject `activity` here for the assembler to majority-vote.
    labels_map = json.loads((ds_dir / "labels.json").read_text()) if (ds_dir / "labels.json").exists() else {}
    # Optional converter-authored map: device-specific session id -> shared physical event id.
    # Only datasets with verified simultaneous placements provide it.
    events_map = (
        json.loads((ds_dir / "events.json").read_text())
        if (ds_dir / "events.json").exists() else {}
    )
    # `labels.json` is the converter's declaration of what it emitted, so a session directory that
    # is not in it is an ORPHAN from an earlier run of that converter — not data. Re-running a
    # converter overwrites the sessions it still emits but does not delete the ones it no longer
    # does, and this glob would otherwise happily ingest them. That is exactly the failure mode a
    # data-quality fix creates: MM-Fit's gap fix dropped 61 sets, and without this guard all 61
    # would have survived in the grid, defeating the fix. Verified 2026-08-11 that labels.json
    # covers every session directory exactly for all 34 converted datasets.
    orphans = 0
    unavailable = 0

    # One session per subdir: sessions/<session_id>/data.parquet.
    for pq in sorted((ds_dir / "sessions").glob("*/data.parquet")):
        sid = pq.parent.name
        if keep_ids is not None and sid not in keep_ids:
            continue
        if labels_map and sid not in labels_map:
            orphans += 1
            continue
        if spec.stream_id not in {s.stream_id for s in session_stream_specs(dataset, sid, role=spec.role)}:
            continue
        frame = pd.read_parquet(pq)
        if "activity" not in frame.columns and sid in labels_map:
            act = labels_map[sid]
            frame = frame.assign(activity=act[0] if isinstance(act, list) else act)
        # A converted physical session may retain one placement when another device's clock has
        # no valid overlap.  That is valid single-device evidence, but it is not evidence for a
        # stream whose required source triad is absent.  Check this before assembly rather than
        # treating an optional placement as a converter failure or emitting fabricated zero rows.
        try:
            channel_names_for_frame(frame, spec)
        except ValueError:
            unavailable += 1
            continue
        # Set this AFTER DataFrame transformations: pandas does not make attrs propagation part of
        # the assign/copy contract. Losing it would silently disable verified placement positives.
        frame.attrs["halo_session_id"] = sid
        frame.attrs["halo_event_id"] = events_map.get(sid, sid)
        # Subject for subject-disjoint splits: a `subject` column if present, else the session-id
        # prefix (converters encode the subject in the id, e.g. "sub01_..." / "subject3_...").
        subject = frame["subject"].iloc[0] if "subject" in frame.columns else sid.split("_")[0]
        yield from (
            (segment, native_rate, subject)
            for segment in iter_logical_segments(frame)
        )

    if orphans:
        print(f"  [{dataset}/{spec.stream_id}] skipped {orphans} session directories absent from "
              f"labels.json (stale output of an earlier converter run)", flush=True)
    if unavailable:
        print(f"  [{dataset}/{spec.stream_id}] skipped {unavailable} sessions without this "
              "stream's required source channels", flush=True)


def _pre_windowed(dataset: str) -> bool:
    """Whether a dataset is distributed as fixed short segments (metadata `pre_windowed: true`)."""
    meta = dataset_root(dataset) / "metadata.json"
    return bool(meta.exists() and json.loads(meta.read_text()).get("pre_windowed", False))


def _max_hours_per_class(dataset: str) -> Optional[float]:
    """Per-class hour cap for a huge free-living dataset (metadata `max_hours_per_class`), else None."""
    meta = dataset_root(dataset) / "metadata.json"
    if not meta.exists():
        return None
    v = json.loads(meta.read_text()).get("max_hours_per_class")
    return float(v) if v else None


def dataset_root(dataset: str) -> Path:
    """Directory holding ``dataset``, resolved under THIS module's ``REPO``.

    Going through the module global rather than importing the resolved path keeps the
    long-standing test contract that monkeypatching ``build_grids.REPO`` redirects the whole
    pipeline at an isolated tree.
    """
    return _resolve_dataset_root(dataset, roots=roots_under(REPO))


def _grid_root(out_root: Optional[Path], dataset: str) -> Path:
    """Directory holding ``dataset``'s grids.

    ``out_root`` is an explicit override used by tests and scratch builds. Left as ``None``
    (the production default) each dataset writes beside its own sessions, which resolves to
    whichever corpus root actually holds it — ``data/datasets`` or ``data/pretraining``.
    """
    return (out_root / dataset) if out_root is not None else dataset_root(dataset)


def _store_dtype(dataset: str) -> np.dtype:
    """On-disk sample dtype for this dataset's grids (metadata ``grid_dtype``, default float32).

    Label-free scale sources declare ``"grid_dtype": "float16"``. Halving grid bytes is what
    makes a corpus of tens of thousands of stream-hours fit on one disk, and it costs nothing
    the model can see: the frontend reads band energies below ~14 Hz, and float16 resolves
    ~0.001 g at 1 g, which is exactly the precision NHANES itself publishes.

    float16 rather than a scaled int16 on purpose. Both are two bytes, but a reader that
    forgets to dequantize an int16 grid gets values wrong by the scale factor and silently
    trains on garbage, whereas a reader that forgets a float16 grid gets correct values in a
    narrower dtype. Every existing read path already ends in ``np.asarray(..., float32)``,
    which upcasts float16 transparently.
    """
    meta = dataset_root(dataset) / "metadata.json"
    if not meta.exists():
        return np.dtype(np.float32)
    name = json.loads(meta.read_text()).get("grid_dtype", "float32")
    if name not in ("float32", "float16"):
        raise ValueError(
            f"{dataset}: unsupported grid_dtype {name!r}; use 'float32' or 'float16'"
        )
    return np.dtype(name)


def _streaming_grid_enabled(dataset: str) -> bool:
    """Whether grids for this source must be written incrementally.

    Large free-living sources can fit on disk while exceeding RAM during the
    final concatenate. This flag is authored in metadata.json and affects only
    the write strategy, never corpus membership or sample selection.
    """
    meta = dataset_root(dataset) / "metadata.json"
    return bool(meta.exists() and json.loads(meta.read_text()).get("streaming_grid", False))


def _full_windows_only(dataset: str) -> bool:
    """Whether native grids omit incomplete source contexts.

    Future-prediction corpora made from independent snippets can otherwise be dominated by
    padded examples that cannot supply all configured prediction horizons.
    """
    meta = dataset_root(dataset) / "metadata.json"
    return bool(meta.exists() and json.loads(meta.read_text()).get("full_windows_only", False))


def _greedy_class_cap(per_class: dict, max_hours: float) -> set:
    """Select session ids so each class contributes at most `max_hours` of data.

    `per_class` maps canonical-label -> list of (session_id, duration_seconds). Within a class,
    sessions are taken in a deterministic hash order (mixes subjects, reproducible) until the hour
    budget is reached — always keeping at least one session so a rare class never vanishes.
    Pure/deterministic so it is unit-testable without disk.
    """
    budget = max_hours * 3600.0
    keep: set = set()
    for items in per_class.values():
        ordered = sorted(items, key=lambda it: hashlib.md5(str(it[0]).encode()).hexdigest())
        acc = 0.0
        for i, (sid, dur) in enumerate(ordered):
            if i > 0 and acc >= budget:
                break
            keep.add(sid)
            acc += dur
    return keep


def _capped_session_ids(dataset: str, spec: StreamSpec, max_hours: float) -> set:
    """Class-balanced keep-set for `dataset`'s stream, reading only parquet row-counts (cheap)."""
    import pyarrow.parquet as pq_meta  # metadata-only read; no data loaded
    from collections import defaultdict
    ds_dir = dataset_root(dataset)
    native_rate = float(json.loads((ds_dir / "metadata.json").read_text())["sampling_rate_hz"])
    labels_map = json.loads((ds_dir / "labels.json").read_text()) if (ds_dir / "labels.json").exists() else {}
    per_class = defaultdict(list)
    for pqf in sorted((ds_dir / "sessions").glob("*/data.parquet")):
        sid = pqf.parent.name
        if spec.stream_id not in {s.stream_id for s in session_stream_specs(dataset, sid, role=spec.role)}:
            continue
        act = labels_map.get(sid)
        label = canonicalize(act[0] if isinstance(act, list) else act) if act else "?"
        nrows = pq_meta.ParquetFile(pqf).metadata.num_rows
        per_class[label].append((sid, nrows / native_rate))
    return _greedy_class_cap(per_class, max_hours)


def _forced_window_for_factory(
    sessions_factory: Callable[[], Iterable[Session]],
    pre_windowed: bool,
    resample_to: Optional[float],
) -> Optional[int]:
    if not pre_windowed:
        return None
    first = next(iter(sessions_factory()), None)
    if first is None:
        return None
    frame, native_rate, _ = first
    if resample_to is None:
        return len(frame)
    return len(
        resample_signal(
            np.zeros((len(frame), 1), np.float32),
            native_rate,
            resample_to,
        )
    )


def _session_grid(
    dataset: str,
    spec: StreamSpec,
    frame: pd.DataFrame,
    native_rate: float,
    *,
    resample_to: Optional[float],
    view: str,
    include_partial: bool,
    forced_window: Optional[int],
    window_seconds: float = WINDOW_SECONDS,
) -> Grid:
    out_rate = resample_to if resample_to is not None else native_rate
    window = (
        forced_window
        if forced_window is not None
        else max(1, round(window_seconds * out_rate))
    )
    return assemble(
        frame,
        dataset,
        spec,
        alignment=view,
        window=window,
        rate_hz=native_rate,
        resample_to=resample_to,
        include_partial=include_partial,
    )


def _window_dir_name(window_seconds: float) -> str:
    """Stable schema component for a physical grid duration (``w4``, ``w0p5``)."""
    value = float(window_seconds)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("window_seconds must be finite and positive")
    text = f"{value:g}".replace(".", "p")
    return f"w{text}"


def _grid_destination(
    out_root: Path | None, dataset: str, alignment: str, stream_id: str, window_seconds: float,
) -> Path:
    return (_grid_root(out_root, dataset) / "grids" / alignment / stream_id
            / _window_dir_name(window_seconds))


def _write_streaming_grid(
    out_root: Path,
    dataset: str,
    spec: StreamSpec,
    sessions_factory: Callable[[], Iterable[Session]],
    *,
    alignment: str,
    resample_to: Optional[float],
    canonical_labels: bool,
    view: str,
    pre_windowed: bool = False,
    window_seconds: float = WINDOW_SECONDS,
    include_partial: bool = False,
) -> None:
    """Two-pass, constant-signal-memory grid writer.

    Pass one computes the exact output shape one session at a time. Pass two
    writes directly into an ``open_memmap`` NPY file and retains only compact
    labels/subjects/event metadata in memory. The resulting on-disk schema is
    byte-for-byte compatible with ``GridRef.load_data()``.
    """
    forced_window = _forced_window_for_factory(
        sessions_factory, pre_windowed, resample_to
    )
    total = 0
    sample_shape: tuple[int, int] | None = None
    mask: np.ndarray | None = None
    channels: tuple[str, ...] | None = None
    rate_out = float(resample_to) if resample_to is not None else 0.0

    for frame, native_rate, _ in sessions_factory():
        grid = _session_grid(
            dataset,
            spec,
            frame,
            native_rate,
            resample_to=resample_to,
            view=view,
            include_partial=include_partial,
            forced_window=forced_window,
            window_seconds=window_seconds,
        )
        if not len(grid.data):
            continue
        shape = (int(grid.data.shape[1]), int(grid.data.shape[2]))
        if sample_shape is None:
            sample_shape = shape
            mask = grid.mask
            channels = grid.channels
            rate_out = float(grid.rate_hz)
        elif shape != sample_shape or grid.channels != channels or not np.array_equal(grid.mask, mask):
            raise ValueError(
                f"{dataset}/{spec.stream_id}/{alignment}: inconsistent session grid "
                f"shape/channels/mask: first={sample_shape, channels, mask.tolist()}, "
                f"got={shape, grid.channels, grid.mask.tolist()}"
            )
        total += len(grid.data)

    if sample_shape is None or mask is None or channels is None:
        # Preserve the ordinary empty-grid behavior without adding a second
        # parallel definition of channel padding.
        empty, subjects = stream_grid(
            dataset,
            spec,
            [],
            alignment=alignment,
            resample_to=resample_to,
            canonical_labels=canonical_labels,
            view=view,
            pre_windowed=pre_windowed,
            window_seconds=window_seconds,
        )
        _save(out_root, spec, empty, subjects, window_seconds=window_seconds)
        return

    destination = _grid_destination(
        out_root, dataset, alignment, spec.stream_id, window_seconds,
    )
    destination.mkdir(parents=True, exist_ok=True)
    temp_data = destination / "data.npy.part"
    if temp_data.exists():
        temp_data.unlink()
    store_dtype = _store_dtype(dataset)
    mapped = np.lib.format.open_memmap(
        temp_data,
        mode="w+",
        dtype=store_dtype,
        shape=(total, *sample_shape),
    )
    labels: list[str] = []
    subjects: list[str] = []
    event_ids: list[str] = []
    lengths = np.empty(total, dtype=np.int32)
    cursor = 0
    release_cursor = 0
    committed = False
    try:
        for frame, native_rate, subject in sessions_factory():
            grid = _session_grid(
                dataset,
                spec,
                frame,
                native_rate,
                resample_to=resample_to,
                view=view,
                include_partial=include_partial,
                forced_window=forced_window,
                window_seconds=window_seconds,
            )
            count = len(grid.data)
            if not count:
                continue
            mapped[cursor : cursor + count] = grid.data
            lengths[cursor : cursor + count] = (
                grid.lengths if grid.lengths is not None
                else np.full(count, grid.data.shape[1], dtype=np.int32)
            )
            if canonical_labels:
                labels.extend(canonicalize(label) for label in grid.labels)
            else:
                labels.extend(map(str, grid.labels))
            subjects.extend([str(subject)] * count)
            event = frame.attrs.get("halo_event_id") or frame.attrs.get("halo_session_id")
            if event is None:
                event = f"{spec.stream_id}:anonymous:{cursor}"
            event_ids.extend(f"{dataset}:{event}:{index}" for index in range(count))
            cursor += count
            # np.memmap is disk-backed, but Linux charges dirty mapped pages to
            # process RSS until writeback. Flush and release them in ~1 GB
            # signal chunks so a 20+ GB grid remains practical on smaller pods.
            if cursor - release_cursor >= 65_536:
                mapped.flush()
                try:
                    mapped._mmap.madvise(mmap.MADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
                release_cursor = cursor
        if cursor != total:
            raise RuntimeError(
                f"{dataset}/{spec.stream_id}/{alignment}: pass count changed "
                f"from {total} to {cursor}"
            )
        mapped.flush()
        os.replace(temp_data, destination / "data.npy")
        committed = True
    finally:
        del mapped
        if not committed:
            temp_data.unlink(missing_ok=True)

    np.save(destination / "mask.npy", mask)
    np.save(destination / "lengths.npy", lengths)
    (destination / "meta.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "stream_id": spec.stream_id,
                "alignment": alignment,
                "rate_hz": rate_out,
                "channels": list(channels),
                "labels": labels,
                "subjects": subjects,
                "event_ids": event_ids,
                "lengths_file": "lengths.npy",
                "store_dtype": str(store_dtype),
                "window_seconds": sample_shape[0] / rate_out,
                "partial_windows": int(np.count_nonzero(lengths < sample_shape[0])),
            }
        )
    )
    print(
        f"  {dataset}/{spec.stream_id}/{alignment}: "
        f"{(total, *sample_shape)} [streaming]"
    )


def build(out_root: Optional[Path] = None, datasets: Optional[Sequence[str]] = None,
          alignments: Optional[Sequence[str]] = None,
          window_seconds: float = WINDOW_SECONDS) -> None:
    """Assemble + save the grid regimes for EVERY device stream (phone + watch + body).

    We build the full phone+watch+body set once; `harmonised-strict` is the phone-only subset selected
    at training time (`deployment_streams(placement_strict=True)`), not a separate materialization.
    Pass `datasets` to build only those (e.g. after a single converter re-run); pass `alignments` to
    build only some regimes (e.g. `["native"]` to add the HALO grids without rebuilding the rest).
    """
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    # ``out_root`` is an explicit override (tests, scratch builds). Left unset, each dataset
    # writes beside its own sessions, which may be either corpus root.
    want = set(datasets) if datasets else None
    regimes = [a for a in _ALIGNMENTS if alignments is None or a[0] in set(alignments)]
    # Optional scale streams are considered only when explicitly named. The default includes every
    # primary train/eval stream plus all direct-converted Phase-A streams, including the three whose
    # non-phone/watch placements intentionally carry role="stress". This makes a clean default build
    # reproduce the trainer's 18-source roster instead of depending on grids left by an explicit
    # one-off build.
    specs = build_stream_specs(datasets)
    for spec in specs:
        if want and spec.dataset not in want:
            continue
        pw = _pre_windowed(spec.dataset)
        full_windows_only = _full_windows_only(spec.dataset)
        cap = _max_hours_per_class(spec.dataset)
        keep_ids = _capped_session_ids(spec.dataset, spec, cap) if cap else None
        if keep_ids is not None:
            print(f"  {spec.dataset}/{spec.stream_id}: class-balanced cap {cap}h/class → {len(keep_ids)} sessions")
        sessions_factory = lambda s=spec, ids=keep_ids: iter_sessions(
            s.dataset, s, keep_ids=ids
        )
        for name, resample_to, canonical, view in regimes:
            include_partial = name == "native" and not full_windows_only
            if _streaming_grid_enabled(spec.dataset):
                _write_streaming_grid(
                    out_root,
                    spec.dataset,
                    spec,
                    sessions_factory,
                    alignment=name,
                    resample_to=resample_to,
                    canonical_labels=canonical,
                    view=view,
                    pre_windowed=pw,
                    window_seconds=window_seconds,
                    include_partial=include_partial,
                )
            else:
                sessions = list(sessions_factory())
                grid, subjects = stream_grid(
                    spec.dataset,
                    spec,
                    sessions,
                    alignment=name,
                    resample_to=resample_to,
                    canonical_labels=canonical,
                    view=view,
                    pre_windowed=pw,
                    window_seconds=window_seconds,
                    include_partial=include_partial,
                )
                _save(out_root, spec, grid, subjects, window_seconds=window_seconds)


def build_stream_specs(datasets: Optional[Sequence[str]] = None) -> Tuple[StreamSpec, ...]:
    """Resolve the exact stream manifest a grid build will materialize.

    Kept separate from disk I/O so tests can prove that a default clean rebuild contains every
    expanded Phase-A source, including direct-converted stress streams.
    """
    want = set(datasets) if datasets else None
    if want:
        deployment = tuple(
            spec for spec in deployment_streams(placement_strict=False, role=None)
            if spec.dataset in want
        )
        # Explicit rebuilds must also materialise every declared primary evaluation stream.
        # Deployment filtering intentionally excludes disclosed placement proxies, but sealed
        # evaluation still needs their grids (currently UT-Complex and Shoaib wrist proxies).
        primary = tuple(
            spec for dataset in sorted(want) for spec in stream_specs(dataset, "primary")
        )
        merged: dict[tuple[str, str], StreamSpec] = {}
        for spec in deployment + primary:
            merged.setdefault((spec.dataset, spec.stream_id), spec)
        return tuple(merged.values())
    default_datasets = set(EXPANDED_PHASE_A_TRAIN_DATASETS) | set(PRIMARY_EVAL_DATASETS)
    # The sealed roster may include an explicitly disclosed placement proxy (currently
    # UT-Complex's wrist-mounted phone).  It is not eligible for phone/watch compatibility
    # pooling, but it must still be materialised when the declared sealed roster is rebuilt.
    # Resolve primary specs directly rather than through ``deployment_streams``, which correctly
    # excludes proxies from the deployment-compatible training view.
    primary = tuple(
        spec for dataset in default_datasets for spec in stream_specs(dataset, "primary")
    )
    direct = tuple(
        spec for spec in deployment_streams(placement_strict=False, role=None)
        if spec.dataset in DIRECT_CONVERTED_TRAIN_DATASETS
    )
    seen: set[tuple[str, str]] = set()
    merged = []
    for spec in primary + direct:
        key = (spec.dataset, spec.stream_id)
        if key not in seen:
            merged.append(spec)
            seen.add(key)
    return tuple(merged)


def _save(
    out_root: Optional[Path], spec: StreamSpec, grid: Grid, subjects: List,
    *, window_seconds: float | None = None,
) -> None:
    # Private legacy callers/tests that omit ``window_seconds`` retain the direct-stream layout.
    # ``build()`` always passes it explicitly and therefore always writes the versioned schema.
    d = (_grid_destination(out_root, spec.dataset, grid.alignment, spec.stream_id, window_seconds)
         if window_seconds is not None else
         _grid_root(out_root, spec.dataset) / "grids" / grid.alignment / spec.stream_id)
    d.mkdir(parents=True, exist_ok=True)
    store_dtype = _store_dtype(spec.dataset)
    np.save(d / "data.npy", np.asarray(grid.data, dtype=store_dtype))
    np.save(d / "mask.npy", grid.mask)
    lengths = (grid.lengths if grid.lengths is not None
               else np.full(len(grid.data), grid.data.shape[1], dtype=np.int32))
    np.save(d / "lengths.npy", lengths)
    (d / "meta.json").write_text(json.dumps({
        "dataset": grid.dataset, "stream_id": spec.stream_id, "alignment": grid.alignment,
        "rate_hz": grid.rate_hz, "channels": list(grid.channels),
        "labels": list(map(str, grid.labels)), "subjects": list(map(str, subjects)),
        "event_ids": list(map(str, grid.event_ids or [])),
        "lengths_file": "lengths.npy",
        "store_dtype": str(store_dtype),
        "window_seconds": (
            grid.data.shape[1] / grid.rate_hz
            if grid.data.ndim == 3 and grid.data.shape[1] and grid.rate_hz > 0 else None
        ),
        "partial_windows": int(np.count_nonzero(lengths < (grid.data.shape[1]
                                                               if grid.data.ndim == 3 else 0))),
    }))
    print(f"  {spec.dataset}/{spec.stream_id}/{grid.alignment}: {grid.data.shape}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", nargs="*", default=None,
                   help="Build only these datasets (default: all primary train/eval streams plus "
                        "the expanded Phase-A direct-converted roster).")
    p.add_argument("--alignment", nargs="*", default=None,
                   choices=[a[0] for a in _ALIGNMENTS],
                   help="Build only these grid regimes (default: all three).")
    p.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS,
                   help="physical source-window duration (default: 6; label-free JEPA uses 8)")
    args = p.parse_args()
    build(datasets=args.dataset, alignments=args.alignment,
          window_seconds=args.window_seconds)


if __name__ == "__main__":
    main()
