"""Grid loader shared by external encoder adapters and historical zero-support scoring.

Each dataset stores windowed grids under::

    data/datasets/<ds>/grids/{harmonised,non_harmonised}/<stream>/w<seconds>/
        data.npy   float32 (N, T, C)   accelerometer (+gyro) in g
        mask.npy   bool    (C,)         per-channel validity (False = zero-pad)
        meta.json  {dataset, stream_id, alignment, rate_hz, channels[list],
                    labels[per-window list], subjects[per-window list]}

and a pre-registered candidate label vocabulary at
``data/datasets/<ds>/eval_labels.json`` (the sealed dataset's declared candidate strings for that
dataset). Harmonized grid construction may store a genuine synonym under its training-corpus
canonical name; :func:`baselines.scoring.align_ground_truth_labels` maps that internal representation
back to the unique frozen target string before scoring. The global ConSE training vocabulary lives at
``data/labels/global_labels.json``.

Unlike the legacy loader (which majority-voted raw per-timestep activity codes
through an ``idx_to_label`` map, with an offset bug), the grid meta already
carries a decoded per-window label string and subject id — so ground truth is
read directly, offset-free. Native (`non_harmonised`) is the default eval source
because baseline adapters resample per their own input contract; harmonised is
exposed via the `alignment` argument.
"""

from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO / "data" / "datasets"
GLOBAL_LABELS_PATH = REPO / "data" / "labels" / "global_labels.json"

ALIGNMENTS = ("native", "non_harmonised", "harmonised")


@dataclass
class EvalStream:
    """One dataset/stream grid, ready for model-agnostic scoring.

    Attributes:
        dataset:     dataset name (e.g. ``"motionsense"``).
        stream:      stream / placement id (e.g. ``"phone_front_pocket"``).
        alignment:   ``"native"`` (current converter output), legacy ``"non_harmonised"``,
                     or resampled ``"harmonised"``.
        windows:     (N, T, C) float32 sensor windows (accel + optional gyro), in g.
        gt:          per-window ground-truth label strings, length N (verbatim from
                     the grid meta — NOT yet aligned or filtered to `eval_labels`).
        subjects:    (N,) subject ids, one per window.
        channels:    the C channel names of `windows`, in grid order.
        rate_hz:     sampling rate of `windows`.
        mask:        (C,) bool per-channel validity (False = zero-padded absence).
        eval_labels: the dataset's pre-registered candidate label vocabulary.
        event_ids:   converter-provided window event ids when available.
        execution_ids: the leakage unit — one continuous physical capture. Derived by removing the
                     window ordinal from ``event_ids`` and then, where the converter provides a
                     ``recordings.json``, mapping the resulting label block onto the recording it
                     was cut out of. See :func:`_recording_map`.
        block_ids:   the finer pre-grouping value (one contiguous label block). Kept for diagnostics
                     only; anything deciding what may be enrolled against what must use
                     ``execution_ids``.
        execution_granularity: ``"recording"`` when a ``recordings.json`` applied, else ``"block"``.
        quality_screen: ``"applied"``, ``"not requested"``, or ``"unavailable: <reason>"`` when the
                     duplicate/implausible caches do not cover this alignment. Never silently empty.
        n_quality_excluded: windows dropped by that screen.
    """
    dataset: str
    stream: str
    alignment: str
    windows: np.ndarray
    gt: List[str]
    subjects: np.ndarray
    channels: List[str]
    rate_hz: float
    mask: np.ndarray
    eval_labels: List[str]
    window_seconds: float = 6.0
    event_ids: Optional[np.ndarray] = None
    execution_ids: Optional[np.ndarray] = None
    block_ids: Optional[np.ndarray] = None
    execution_granularity: str = "block"
    quality_screen: str = "not requested"
    n_quality_excluded: int = 0
    # Synthetic diagnostic overrides. Production grid loads leave these as None and adapters derive
    # the authoritative values from deployment_policy exactly as before.
    gravity_state: Optional[str] = None
    channel_descriptions: Optional[list] = None
    # Label of the controlled acquisition perturbation applied to this view of the grid, or None
    # for the grid as converted. Retained diagnostic views must give it a unique name; adapters that cache per
    # stream must key on it so a perturbed and an unperturbed view of one grid never collide.
    perturbation: Optional[str] = None
    # Highest physical sampling rate at which this view contains measured information. A derived
    # upsampled view keeps its original ceiling so consumers do not treat interpolation as new data.
    effective_source_rate_hz: Optional[float] = None
    lengths: Optional[np.ndarray] = None
    execution_identity_known: bool = True

    @property
    def n_windows(self) -> int:
        return self.windows.shape[0]


@dataclass
class MultiDeviceEvalStream:
    """Aligned simultaneous-device evaluation cell.

    Members remain native-rate streams; consumers must encode/resample each member themselves.
    The common arrays are deliberately exposed under the single-stream names used by episode
    construction, while raw signal access is only possible through ``devices``.
    """
    dataset: str
    cell_id: str
    devices: list[EvalStream]
    device_ids: list[str]
    event_ids: np.ndarray
    gt: List[str]
    subjects: np.ndarray
    eval_labels: List[str]
    execution_ids: np.ndarray | None
    execution_identity_known: bool
    quality_screen: str
    n_quality_excluded: int
    quality_excluded_by_device: dict[str, int]
    window_seconds: float
    alignment: str
    # Rows available to a single placement but not to every member.  They are excluded rather
    # than aligned approximately: a composite example must retain the converter's exact event id.
    n_alignment_excluded: int = 0

    @property
    def stream(self) -> str:
        return self.cell_id

    @property
    def n_windows(self) -> int:
        return len(self.event_ids)


def source_slice_fingerprint(stream: EvalStream | MultiDeviceEvalStream) -> str:
    """Hash the exact valid raw samples and row metadata shared by every provider.

    The evaluator records this independently of model feature caches. It makes the fairness
    contract auditable: two reported provider rows with the same cell must name the same source
    fingerprint, even though each released model performs its own documented resampling.
    """
    cached = getattr(stream, "_source_slice_fingerprint", None)
    if cached is not None:
        return str(cached)
    digest = hashlib.sha256()
    members = stream.devices if isinstance(stream, MultiDeviceEvalStream) else [stream]
    digest.update(str(stream.dataset).encode())
    digest.update(str(stream.stream).encode())
    digest.update(np.asarray(stream.event_ids, dtype=str).tobytes())
    digest.update(np.asarray(stream.gt, dtype=str).tobytes())
    for member in members:
        digest.update(member.stream.encode())
        digest.update(str(float(member.rate_hz)).encode())
        digest.update("\0".join(member.channels).encode())
        digest.update(np.asarray(member.mask, dtype=np.bool_).tobytes())
        for field in ("source_rate_hz", "effective_source_rate_hz", "gravity_state", "config_text",
                      "device_ids", "perturbation"):
            if hasattr(member, field):
                digest.update(field.encode())
                digest.update(repr(getattr(member, field)).encode())
        lengths = (np.asarray(member.lengths, dtype=np.int64) if member.lengths is not None
                   else np.full(member.n_windows, member.windows.shape[1], dtype=np.int64))
        digest.update(lengths.tobytes())
        # Padding is excluded deliberately: it is not measured evidence and adapters must use
        # ``lengths``. Hashing row-by-row avoids materialising a second corpus-sized tensor.
        for row, length in enumerate(lengths.tolist()):
            digest.update(np.ascontiguousarray(member.windows[row, :length]).view(np.uint8))
    value = digest.hexdigest()
    setattr(stream, "_source_slice_fingerprint", value)
    return value


def load_multi_device_stream(
    dataset: str,
    device_ids: Sequence[str],
    alignment: str = "non_harmonised",
    *,
    window_seconds: float = 6.0,
    apply_quality_screen: bool = True,
) -> MultiDeviceEvalStream:
    """Load an ordered composite on the exact event-id intersection.

    Each device grid may legitimately be a superset when a recording retains one placement after
    another placement's clock fails coverage validation.  The composite therefore uses only event
    ids emitted verbatim by *every* member.  It never joins by timestamp, nearest neighbour, or
    window index, and it still rejects any disagreement in metadata for a retained physical event.
    """
    ids = list(device_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("a multi-device cell requires at least two distinct ordered stream ids")
    # Load unfiltered first: quality is applied once to the composite intersection below.
    members = [load_eval_stream(dataset, device, alignment, window_seconds=window_seconds,
                                apply_quality_screen=False) for device in ids]
    reference = members[0]
    common_ids = set(map(str, reference.event_ids))
    for member in members[1:]:
        common_ids.intersection_update(map(str, member.event_ids))
        if member.eval_labels != reference.eval_labels:
            raise ValueError(f"{dataset}: multi-device members disagree on candidate labels")
    if not common_ids:
        raise ValueError(f"{dataset}: multi-device members have no exact common event ids")
    row_indices = []
    for member in members:
        lookup = {str(event_id): row for row, event_id in enumerate(member.event_ids)}
        if len(lookup) != member.n_windows:
            raise ValueError(f"{dataset}/{member.stream}: duplicate event ids prevent exact composite alignment")
        row_indices.append(np.asarray(
            [lookup[str(event_id)] for event_id in reference.event_ids if str(event_id) in common_ids],
            dtype=np.int64,
        ))
    reference_indices = row_indices[0]
    for member, indices in zip(members[1:], row_indices[1:]):
        for field in ("gt", "subjects", "execution_ids"):
            expected = np.asarray(getattr(reference, field), dtype=object)[reference_indices]
            actual = np.asarray(getattr(member, field), dtype=object)[indices]
            if not np.array_equal(expected, actual):
                raise ValueError(f"{dataset}: multi-device members disagree on {field} for common event ids")
    n_alignment_excluded = int(reference.n_windows - len(reference_indices))
    def align(member: EvalStream, indices: np.ndarray) -> EvalStream:
        return EvalStream(**{**member.__dict__, "windows": member.windows[indices],
                             "gt": [member.gt[row] for row in indices],
                             "subjects": member.subjects[indices], "event_ids": member.event_ids[indices],
                             "execution_ids": member.execution_ids[indices] if member.execution_ids is not None else None,
                             "block_ids": member.block_ids[indices] if member.block_ids is not None else None,
                             "lengths": member.lengths[indices] if member.lengths is not None else None})
    members = [align(member, indices) for member, indices in zip(members, row_indices)]
    reference = members[0]
    keep = np.ones(reference.n_windows, dtype=bool)
    excluded_by_device: dict[str, int] = {}
    quality = "not requested"
    if apply_quality_screen:
        quality = "applied"
        for member in members:
            excluded, status = _quality_excluded(
                dataset, member.stream, alignment, window_seconds,
            )
            if status != "applied":
                quality = status
                break
            # Quality artifacts refer to pre-intersection grid rows.  Map only those exact rows
            # which survived event-id alignment onto the composite's shared index space.
            original_rows = row_indices[len(excluded_by_device)]
            original_to_composite = {int(row): pos for pos, row in enumerate(original_rows)}
            composite_rows = np.asarray(
                [original_to_composite[int(row)] for row in excluded
                 if int(row) in original_to_composite], dtype=np.int64,
            )
            excluded_by_device[member.stream] = int(len(composite_rows))
            keep[composite_rows] = False
        if quality != "applied":
            raise RuntimeError(f"{dataset}/{'+'.join(ids)}: composite quality screen {quality}")
    def subset(member: EvalStream) -> EvalStream:
        return EvalStream(**{**member.__dict__, "windows": member.windows[keep],
                             "gt": [x for x, ok in zip(member.gt, keep) if ok],
                             "subjects": member.subjects[keep], "event_ids": member.event_ids[keep],
                             "execution_ids": member.execution_ids[keep] if member.execution_ids is not None else None,
                             "block_ids": member.block_ids[keep] if member.block_ids is not None else None,
                             "lengths": member.lengths[keep] if member.lengths is not None else None,
                             "quality_screen": quality, "n_quality_excluded": int((~keep).sum())})
    members = [subset(member) for member in members]
    return MultiDeviceEvalStream(
        dataset=dataset, cell_id="+".join(ids), devices=members, device_ids=ids,
        event_ids=members[0].event_ids, gt=members[0].gt, subjects=members[0].subjects,
        eval_labels=members[0].eval_labels, execution_ids=members[0].execution_ids,
        execution_identity_known=all(member.execution_identity_known for member in members),
        quality_screen=quality, n_quality_excluded=int((~keep).sum()),
        quality_excluded_by_device=excluded_by_device, window_seconds=float(window_seconds),
        alignment=alignment, n_alignment_excluded=n_alignment_excluded,
    )


def _window_dir_name(window_seconds: float) -> str:
    value = float(window_seconds)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("window_seconds must be finite and positive")
    return f"w{value:g}".replace(".", "p")


def _grid_dir(dataset: str, stream: str, alignment: str, window_seconds: float = 6.0) -> Path:
    if alignment not in ALIGNMENTS:
        raise ValueError(f"alignment must be one of {ALIGNMENTS}, got {alignment!r}")
    root = DATASETS_DIR / dataset / "grids" / alignment / stream
    qualified = root / _window_dir_name(window_seconds)
    # Migration: the historical 6 s schema stored files directly under <stream>.
    # Prefer duration-qualified materializations when present, but never make old sealed grids
    # unreadable merely because the schema has advanced.
    if qualified.exists():
        return qualified
    # Only the historical unqualified schema is a documented 6 s compatibility path.
    # Never serve a different evidence duration under the caller's requested label.
    if np.isclose(float(window_seconds), 6.0) and (root / "meta.json").exists():
        return root
    raise FileNotFoundError(
        f"missing {float(window_seconds):g} s grid for {dataset}/{stream}/{alignment}; "
        f"expected {qualified}"
    )


def list_streams(dataset: str, alignment: str = "non_harmonised", *, window_seconds: float = 6.0) -> List[str]:
    """Stream ids available for a dataset under the given alignment."""
    root = DATASETS_DIR / dataset / "grids" / alignment
    if not root.exists():
        return []
    result = []
    duration_dir = _window_dir_name(window_seconds)
    for stream_dir in root.iterdir():
        if not stream_dir.is_dir():
            continue
        if (stream_dir / "meta.json").exists() or (stream_dir / duration_dir / "meta.json").exists():
            result.append(stream_dir.name)
    return sorted(result)


def load_eval_labels(dataset: str, stream: Optional[str] = None) -> List[str]:
    """The pre-registered candidate label vocabulary, restricted to what `stream` can observe.

    `eval_labels.json` carries the dataset-wide vocabulary under ``labels``. Several sources are not
    uniform across their placements, and scoring those against the dataset-wide set penalises a
    model for labels the acquisition configuration physically never records. Where that is a
    property of the PROTOCOL rather than of what happened to survive windowing, the file declares it
    under ``streams``:

        {"labels": [...], "streams": {"left_arm": [...], "left_shin": [...]}}

    Measured 2026-08-11: PHYTMO's arm and forearm units observe 6 of its 20 labels while its shin
    and thigh units observe 14 — the upper-limb exercises and the lower-limb ones are separate
    protocols recorded on separate units. Restriction is opt-in per stream; a stream absent from
    ``streams`` gets the dataset-wide vocabulary, and a declared subset must be one.
    """
    path = DATASETS_DIR / dataset / "eval_labels.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No eval_labels.json for '{dataset}' at {path}. This dataset is not "
            "set up as a sealed evaluation target."
        )
    blob = json.loads(path.read_text())
    labels = list(blob["labels"])
    declared = blob.get("streams", {}).get(stream) if stream else None
    if declared is None:
        return labels
    unknown = sorted(set(declared) - set(labels))
    if unknown:
        raise ValueError(
            f"{dataset}/{stream}: eval_labels.json declares stream labels absent from the "
            f"dataset vocabulary: {unknown}"
        )
    return [label for label in labels if label in set(declared)]


def _recording_map(dataset: str) -> dict:
    """``{event_id_without_ordinal: recording_id}`` for one dataset, or ``{}`` if it declares none.

    A converter emits one session per contiguous label block, so several sessions routinely come out
    of ONE continuous capture. Those blocks are seconds apart and are not independent enrollment
    executions — measured 2026-08-11, Opportunity carries 30 blocks per (subject, label) against 6
    real recordings. `recordings.json` (written by the converter, regenerable with
    `data.scripts.curate.build_recording_maps`) maps each session back onto its capture.

    `events.json`, where present, separately maps device-specific session ids onto one verified
    simultaneous physical event, and `build_grids` uses that value as the event id. The two maps are
    composed here so the result is keyed the way :func:`load_eval_stream` sees ids. A dataset whose
    events and recordings disagree — two sessions sharing an event but not a recording — is a
    converter bug rather than something to paper over, so it raises.
    """
    root = DATASETS_DIR / dataset
    recordings_path = root / "recordings.json"
    if not recordings_path.exists():
        return {}
    recordings = json.loads(recordings_path.read_text())
    events_path = root / "events.json"
    events = json.loads(events_path.read_text()) if events_path.exists() else {}

    composed: dict = {}
    for session, recording in recordings.items():
        key = f"{dataset}:{events.get(session, session)}"
        previous = composed.setdefault(key, recording)
        if previous != recording:
            raise ValueError(
                f"{dataset}: sessions sharing physical event {key!r} disagree on their recording "
                f"({previous!r} vs {recording!r}). events.json and recordings.json must nest."
            )
    return composed


def load_global_labels() -> List[str]:
    """The global ConSE training-label vocabulary (closed-vocab baselines)."""
    if not GLOBAL_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Global label vocabulary missing at {GLOBAL_LABELS_PATH}. Run "
            "`python -m data.scripts.labels.build_global_label_mapping`."
        )
    return list(json.loads(GLOBAL_LABELS_PATH.read_text())["labels"])


@lru_cache(maxsize=12)
def _quality_exclusion_cache(alignment: str, window_seconds: float = 6.0) -> dict[str, set[int]]:
    """Validate each corpus-wide quality artifact once per process."""
    from data.scripts.scan_duplicates import load as load_duplicates
    from data.scripts.scan_implausible import load as load_implausible

    duplicate = load_duplicates(alignment, require=True, window_seconds=window_seconds)
    implausible = load_implausible(alignment, require=True, window_seconds=window_seconds)
    return {
        key: set(duplicate.get(key, ())) | set(implausible.get(key, ()))
        for key in set(duplicate) | set(implausible)
    }


def _quality_excluded(dataset: str, stream: str, alignment: str,
                      window_seconds: float = 6.0) -> tuple[np.ndarray, str]:
    """Window indices this stream must not serve, plus a one-word provenance string.

    ``scan_duplicates`` (byte-identical stale-buffer windows) and ``scan_implausible`` (windows
    outside any consumer sensor's full-scale range) cache their verdicts per stream. Phase-A
    training and historical support-bank builds applied them; **evaluation did not**, so a
    window that is too corrupt to train on was still scored. That is live, not hypothetical:
    `motionsense/phone_front_pocket` is a sealed test stream and carries 34 flagged
    duplicates, and Opportunity's 70 fabricated hole-windows were excluded from training while
    remaining available to the evaluator.

    The caches are built for one alignment at a time. When they do not cover the requested
    alignment this returns the reason rather than an empty set, so a silently unscreened load is
    distinguishable from a genuinely clean one — :attr:`EvalStream.quality_screen` records which.
    """
    key = f"{dataset}/{stream}"
    try:
        excluded = _quality_exclusion_cache(alignment, float(window_seconds)).get(key, set())
    except (FileNotFoundError, ValueError) as error:
        return np.zeros(0, dtype=int), f"unavailable: {error}"
    return np.asarray(sorted(excluded), dtype=int), "applied"


def load_eval_stream(
    dataset: str,
    stream: str,
    alignment: str = "non_harmonised",
    *,
    window_seconds: float = 6.0,
    apply_quality_screen: bool = True,
    candidate_labels: Optional[List[str]] = None,
) -> EvalStream:
    """Load one dataset/stream grid as an :class:`EvalStream`.

    Args:
        dataset:   dataset name under ``data/datasets/``.
        stream:    stream / placement id (see :func:`list_streams`).
        alignment: ``"non_harmonised"`` (default; native channels/rate — the eval
                   source, since adapters resample per baseline) or ``"harmonised"``.
        apply_quality_screen: drop windows flagged by ``scan_duplicates`` /
                   ``scan_implausible`` (see :func:`_quality_excluded`). Pass ``False`` only to
                   inspect the raw grid; scoring on an unscreened stream reports windows the
                   trainer itself refused.
        candidate_labels: explicit candidate vocabulary for a training-only stream. Sealed
                   evaluation callers must omit this and use the dataset's registered
                   ``eval_labels.json``; the zero-shot reference-bank builder supplies the frozen
                   global training vocabulary because training datasets intentionally have no
                   sealed-evaluation candidate file.

    The returned `gt` / `subjects` are 1:1 with `windows` (length N) and verbatim
    from the grid — align and restrict `gt` to `eval_labels` at scoring time via
    :func:`baselines.scoring.filter_ground_truth`.
    """
    gdir = _grid_dir(dataset, stream, alignment, window_seconds)
    if not gdir.exists():
        avail = list_streams(dataset, alignment, window_seconds=window_seconds)
        raise FileNotFoundError(
            f"No grid for {dataset}/{stream} ({alignment}) at {gdir}. "
            f"Available {alignment} streams: {avail}"
        )

    windows = np.load(gdir / "data.npy")
    mask = np.load(gdir / "mask.npy")
    meta = json.loads((gdir / "meta.json").read_text())
    lengths_path = gdir / meta.get("lengths_file", "lengths.npy")
    if "lengths_file" in meta and not lengths_path.exists():
        raise FileNotFoundError(f"{dataset}/{stream}: declared lengths file is missing: {lengths_path}")
    lengths = (np.load(lengths_path) if lengths_path.exists()
               else np.full(len(windows), windows.shape[1], dtype=np.int64))
    if (lengths.shape != (len(windows),) or not np.issubdtype(lengths.dtype, np.integer)
            or np.any(lengths <= 0) or np.any(lengths > windows.shape[1])):
        raise ValueError(f"{dataset}/{stream}: invalid per-window valid lengths")

    gt = list(meta["labels"])
    subjects = np.asarray(meta["subjects"])
    event_ids = np.asarray(
        meta.get("event_ids", [f"{dataset}:{stream}:window_{i}" for i in range(len(gt))]),
        dtype=object,
    )
    block_ids = np.asarray([
        value.rsplit(":", 1)[0]
        if ":" in str(value) and str(value).rsplit(":", 1)[1].isdigit()
        else str(value)
        for value in event_ids
    ], dtype=object)
    # Group contiguous label blocks back onto the continuous capture they were cut out of. Without
    # this, two blocks of one bout look like two independent enrollment executions.
    recordings = _recording_map(dataset)
    execution_ids = (
        np.asarray([recordings.get(block, block) for block in block_ids], dtype=object)
        if recordings else block_ids
    )
    channels = list(meta["channels"])

    # Structural invariants — fail loud rather than silently misalign scoring.
    n = windows.shape[0]
    if not (len(gt) == len(subjects) == len(event_ids) == n):
        raise ValueError(
            f"{dataset}/{stream}: meta labels ({len(gt)}) / subjects "
            f"({len(subjects)}) do not match window count ({n})."
        )
    if windows.shape[2] != len(channels):
        raise ValueError(
            f"{dataset}/{stream}: window channel dim ({windows.shape[2]}) != "
            f"len(channels) ({len(channels)})."
        )
    if mask.shape != (len(channels),):
        raise ValueError(
            f"{dataset}/{stream}: mask shape {mask.shape} != ({len(channels)},)."
        )

    # Applied AFTER the structural invariants so a length mismatch is still reported against the
    # grid as written, not against the screened view of it.
    screen = "not requested"
    n_excluded = 0
    if apply_quality_screen:
        actual_window = float(meta.get("window_seconds", window_seconds))
        excluded, screen = _quality_excluded(dataset, stream, alignment, actual_window)
        if len(excluded):
            keep = np.ones(n, dtype=bool)
            keep[excluded[excluded < n]] = False
            n_excluded = int((~keep).sum())
            windows = windows[keep]
            lengths = lengths[keep]
            gt = [label for label, take in zip(gt, keep) if take]
            subjects = subjects[keep]
            event_ids = event_ids[keep]
            block_ids = block_ids[keep]
            execution_ids = execution_ids[keep]

    return EvalStream(
        dataset=dataset,
        stream=stream,
        alignment=alignment,
        window_seconds=float(meta.get("window_seconds", window_seconds)),
        windows=windows,
        gt=gt,
        subjects=subjects,
        channels=channels,
        rate_hz=float(meta["rate_hz"]),
        mask=mask.astype(bool),
        eval_labels=(list(candidate_labels) if candidate_labels is not None
                     else load_eval_labels(dataset, stream)),
        event_ids=event_ids,
        execution_ids=execution_ids,
        block_ids=block_ids,
        execution_granularity="recording" if recordings else "block",
        quality_screen=screen,
        n_quality_excluded=n_excluded,
        lengths=lengths,
        execution_identity_known=(
            dataset != "tnda_har" and "event_ids" in meta
            and bool(meta.get("execution_identity_known", True))
        ),
    )
