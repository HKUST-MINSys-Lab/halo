"""Loader for the new-repo grid format, feeding the ZS-XD evaluation.

Each dataset stores windowed grids under::

    data/datasets/<ds>/grids/{harmonised,non_harmonised}/<stream>/
        data.npy   float32 (N, T, C)   accelerometer (+gyro) in g
        mask.npy   bool    (C,)         per-channel validity (False = zero-pad)
        meta.json  {dataset, stream_id, alignment, rate_hz, channels[list],
                    labels[per-window list], subjects[per-window list]}

and a pre-registered candidate label vocabulary at
``data/datasets/<ds>/eval_labels.json`` (the ZS-XD target strings for that
dataset). Harmonized grid construction may store a genuine synonym under its training-corpus
canonical name; :func:`eval.scoring.align_ground_truth_labels` maps that internal representation
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
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
        eval_labels: the dataset's pre-registered ZS-XD candidate label vocabulary.
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
    # for the grid as converted. Set only by :mod:`eval.perturbation`; adapters that cache per
    # stream must key on it so a perturbed and an unperturbed view of one grid never collide.
    perturbation: Optional[str] = None
    lengths: Optional[np.ndarray] = None
    execution_identity_known: bool = True

    @property
    def n_windows(self) -> int:
        return self.windows.shape[0]


def _grid_dir(dataset: str, stream: str, alignment: str) -> Path:
    if alignment not in ALIGNMENTS:
        raise ValueError(f"alignment must be one of {ALIGNMENTS}, got {alignment!r}")
    return DATASETS_DIR / dataset / "grids" / alignment / stream


def list_streams(dataset: str, alignment: str = "non_harmonised") -> List[str]:
    """Stream ids available for a dataset under the given alignment."""
    root = DATASETS_DIR / dataset / "grids" / alignment
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


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
            "set up as a ZS-XD evaluation target."
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


@lru_cache(maxsize=3)
def _quality_exclusion_cache(alignment: str) -> dict[str, set[int]]:
    """Validate each corpus-wide quality artifact once per process."""
    from data.scripts.scan_duplicates import load as load_duplicates
    from data.scripts.scan_implausible import load as load_implausible

    duplicate = load_duplicates(alignment, require=True)
    implausible = load_implausible(alignment, require=True)
    return {
        key: set(duplicate.get(key, ())) | set(implausible.get(key, ()))
        for key in set(duplicate) | set(implausible)
    }


def _quality_excluded(dataset: str, stream: str, alignment: str) -> tuple[np.ndarray, str]:
    """Window indices this stream must not serve, plus a one-word provenance string.

    ``scan_duplicates`` (byte-identical stale-buffer windows) and ``scan_implausible`` (windows
    outside any consumer sensor's full-scale range) cache their verdicts per stream. Phase-A
    training and the memory-bank build have always applied them; **evaluation did not**, so a
    window that is too corrupt to train on was still scored. That is live, not hypothetical:
    `motionsense/phone_front_pocket` is a Phase-B development stream and carries 34 flagged
    duplicates, and Opportunity's 70 fabricated hole-windows were excluded from training while
    remaining available to the evaluator.

    The caches are built for one alignment at a time. When they do not cover the requested
    alignment this returns the reason rather than an empty set, so a silently unscreened load is
    distinguishable from a genuinely clean one — :attr:`EvalStream.quality_screen` records which.
    """
    key = f"{dataset}/{stream}"
    try:
        excluded = _quality_exclusion_cache(alignment).get(key, set())
    except (FileNotFoundError, ValueError) as error:
        return np.zeros(0, dtype=int), f"unavailable: {error}"
    return np.asarray(sorted(excluded), dtype=int), "applied"


def load_eval_stream(
    dataset: str,
    stream: str,
    alignment: str = "non_harmonised",
    *,
    apply_quality_screen: bool = True,
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

    The returned `gt` / `subjects` are 1:1 with `windows` (length N) and verbatim
    from the grid — align and restrict `gt` to `eval_labels` at scoring time via
    :func:`eval.scoring.filter_ground_truth`.
    """
    gdir = _grid_dir(dataset, stream, alignment)
    if not gdir.exists():
        avail = list_streams(dataset, alignment)
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
        excluded, screen = _quality_excluded(dataset, stream, alignment)
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
        windows=windows,
        gt=gt,
        subjects=subjects,
        channels=channels,
        rate_hz=float(meta["rate_hz"]),
        mask=mask.astype(bool),
        eval_labels=load_eval_labels(dataset, stream),
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
