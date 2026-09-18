"""Read-only discovery and deterministic sampling for generated HALO grids."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


from data.scripts.curate.corpus_roots import grid_search_roots
from halo.paths import DATASETS_DIR

# Retained as the labelled-tree constant for callers that address it directly. Grid DISCOVERY
# spans every corpus root (labelled + label-free pretraining) via ``grid_search_roots()``.
_FINGERPRINT_CACHE: dict[tuple, str] = {}
_WINDOW_DIR_RE = re.compile(r"^w(?:\d+(?:p\d+)?|p\d+)$")


@lru_cache(maxsize=4096)
def _grid_signal_digests(
    grid_dir_text: str,
    stat_key: tuple[tuple[str, int, int, int], ...],
) -> tuple[str, str | None]:
    """Hash expensive arrays without changing the public fingerprint algorithm."""
    del stat_key
    grid_dir = Path(grid_dir_text)
    data = np.load(grid_dir / "data.npy", mmap_mode="r")
    sampled = np.ascontiguousarray(data[::97, ::13, :], dtype=np.float32)
    sampled_digest = hashlib.sha256(sampled.tobytes()).hexdigest()
    lengths_path = grid_dir / "lengths.npy"
    lengths_digest = None
    if lengths_path.exists():
        lengths = np.ascontiguousarray(np.load(lengths_path, mmap_mode="r"), dtype=np.int32)
        lengths_digest = hashlib.sha256(lengths.tobytes()).hexdigest()
    return sampled_digest, lengths_digest


def _is_window_dir(path: Path) -> bool:
    """True only for version directories such as ``w4`` or ``w0p5``.

    A prefix check is invalid because deployment streams commonly start with ``watch_``.
    """
    return bool(_WINDOW_DIR_RE.fullmatch(path.name))


@dataclass(frozen=True)
class GridRef:
    """Metadata and paths for one persisted dataset/stream grid."""

    dataset: str
    stream: str
    alignment: str
    rate_hz: float
    channels: tuple[str, ...]
    mask: tuple[bool, ...]
    labels: tuple[str, ...]
    subjects: tuple[str, ...]
    event_ids: tuple[str, ...]
    event_ids_explicit: bool
    shape: tuple[int, int, int]
    grid_dir: Path

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.stream}"

    @property
    def n_windows(self) -> int:
        return self.shape[0]

    @property
    def duration_seconds(self) -> float:
        # An empty placeholder grid (a failed/absent conversion) carries rate 0. Report no
        # duration rather than raising ZeroDivisionError in every caller that sums hours.
        if self.rate_hz <= 0 or self.shape[0] == 0:
            return 0.0
        return float(self.load_lengths().sum()) / self.rate_hz

    def load_data(self) -> np.ndarray:
        """Memory-map the grid so callers read only selected windows.

        The array dtype is whatever the build stored: ``float32`` for labelled sources,
        ``float16`` for label-free scale sources (see ``build_grids._store_dtype``). Values
        are in the same physical units either way, and every read path in the repo ends in
        ``np.asarray(..., dtype=np.float32)``, which upcasts transparently. Read
        :attr:`store_dtype` if a caller genuinely needs to branch on it.
        """
        return np.load(self.grid_dir / "data.npy", mmap_mode="r")

    @property
    def store_dtype(self) -> np.dtype:
        """On-disk sample dtype, read through NumPy's stable memory-map API."""
        return np.dtype(np.load(self.grid_dir / "data.npy", mmap_mode="r").dtype)

    def load_lengths(self) -> np.ndarray:
        """True samples per row; legacy grids without the sidecar are all full-length."""
        path = self.grid_dir / "lengths.npy"
        if path.exists():
            return np.load(path, mmap_mode="r")
        return np.full(self.shape[0], self.shape[1], dtype=np.int32)


def discover_grids(
    alignment: str = "harmonised",
    datasets_dir: Path | None = None,
    *,
    window_seconds: float | None = None,
) -> list[GridRef]:
    """Discover and validate all persisted grids for one alignment.

    Searches every corpus root by default (``data/datasets`` and ``data/pretraining``), so a
    label-free scale source is found without callers knowing which tree holds it. Pass
    ``datasets_dir`` to restrict discovery to one root.
    """
    refs: list[GridRef] = []
    if window_seconds is None:
        # Keep existing corpus discovery stable: legacy direct-stream grids remain its default.
        # A duration-qualified grid must be requested explicitly so 4/8/16 s evaluation assets
        # cannot silently multiply the label-free/pretraining corpus.
        patterns = (f"*/grids/{alignment}/*/meta.json", f"*/grids/{alignment}/*/w*/meta.json")
    else:
        token = f"w{float(window_seconds):g}".replace(".", "p")
        # During migration, requested 6 s grids may still live directly under the stream. The
        # evaluator prefers an explicit w6 when present and falls back per stream otherwise;
        # discovery and quality scans must resolve the identical corpus.
        patterns = ((f"*/grids/{alignment}/*/meta.json", f"*/grids/{alignment}/*/{token}/meta.json")
                    if np.isclose(float(window_seconds), 6.0)
                    else (f"*/grids/{alignment}/*/{token}/meta.json",))
    roots = (datasets_dir,) if datasets_dir is not None else grid_search_roots()
    def _grid_key(path: Path) -> tuple[str, str]:
        stream_dir = path.parent.parent if _is_window_dir(path.parent) else path.parent
        return stream_dir.parents[2].name, stream_dir.name

    candidates = [path for root in roots for pattern in patterns for path in root.glob(pattern)]
    if window_seconds is None or np.isclose(float(window_seconds), 6.0):
        # Default corpus discovery keeps a direct legacy grid authoritative. An explicit 6 s
        # request mirrors ``baselines.data._grid_dir`` and prefers w6, with legacy fallback.
        selected: dict[tuple[str, str], Path] = {}
        for path in candidates:
            key = _grid_key(path)
            old = selected.get(key)
            def rank(value: Path) -> int:
                # A no-duration call is the historical six-second corpus.  New streams
                # have no direct legacy directory, so choose explicit w6 deterministically.
                if not _is_window_dir(value.parent):
                    return 1 if window_seconds is not None else 0
                if value.parent.name == "w6":
                    return 0 if window_seconds is not None else 1
                return 2
            if old is None or rank(path) < rank(old) or (rank(path) == rank(old) and str(path) < str(old)):
                selected[key] = path
        candidates = list(selected.values())
    meta_paths = sorted(candidates, key=lambda path: (*_grid_key(path), str(path)))
    for meta_path in meta_paths:
        grid_dir = meta_path.parent
        data_path = grid_dir / "data.npy"
        mask_path = grid_dir / "mask.npy"
        if not data_path.exists() or not mask_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        data = np.load(data_path, mmap_mode="r")
        mask = np.load(mask_path).astype(bool)
        channels = tuple(map(str, meta["channels"]))
        labels = tuple(map(str, meta["labels"]))
        subjects = tuple(map(str, meta["subjects"]))
        event_ids_explicit = "event_ids" in meta and len(meta["event_ids"]) == data.shape[0]
        event_ids = (
            tuple(map(str, meta["event_ids"]))
            if event_ids_explicit
            else tuple(f"{meta['dataset']}/{meta['stream_id']}:{i}" for i in range(data.shape[0]))
        )

        if data.ndim != 3:
            raise ValueError(f"{grid_dir}: expected (N,T,C), got {data.shape}")
        if len(labels) != data.shape[0] or len(subjects) != data.shape[0]:
            raise ValueError(f"{grid_dir}: labels/subjects do not match window count")
        if len(event_ids) != data.shape[0]:
            raise ValueError(f"{grid_dir}: event_ids do not match window count")
        if len(channels) != data.shape[2] or mask.shape != (data.shape[2],):
            raise ValueError(f"{grid_dir}: channels/mask do not match channel dimension")
        lengths_path = grid_dir / "lengths.npy"
        if lengths_path.exists():
            lengths = np.load(lengths_path, mmap_mode="r")
            if lengths.shape != (data.shape[0],):
                raise ValueError(f"{grid_dir}: lengths do not match window count")
            if len(lengths) and (int(lengths.min()) < 1 or int(lengths.max()) > data.shape[1]):
                raise ValueError(f"{grid_dir}: lengths must be in [1, {data.shape[1]}]")

        refs.append(GridRef(
            dataset=str(meta["dataset"]),
            stream=str(meta["stream_id"]),
            alignment=str(meta["alignment"]),
            rate_hz=float(meta["rate_hz"]),
            channels=channels,
            mask=tuple(map(bool, mask)),
            labels=labels,
            subjects=subjects,
            event_ids=event_ids,
            event_ids_explicit=event_ids_explicit,
            shape=tuple(map(int, data.shape)),
            grid_dir=grid_dir,
        ))
    return refs


def grid_corpus_fingerprint(
    alignment: str = "harmonised",
    refs: Sequence[GridRef] | None = None,
) -> str:
    """Fingerprint the grids consumed by corpus-wide quality scans.

    Labels and subjects are hashed in full. Signal content is sampled deterministically and paired
    with file size, so an edited grid invalidates exclusion indices without forcing every training
    process to hash tens of gigabytes. File modification time is deliberately excluded: a
    byte-identical clean rebuild on another machine must accept the same reviewed cache.
    """
    selected = list(refs) if refs is not None else discover_grids(alignment)
    cache_parts = []
    for ref in sorted(selected, key=lambda item: item.key):
        paths = [ref.grid_dir / "data.npy", ref.grid_dir / "meta.json", ref.grid_dir / "mask.npy"]
        lengths = ref.grid_dir / "lengths.npy"
        if lengths.exists():
            paths.append(lengths)
        cache_parts.append((
            ref.key,
            tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                  for path in paths),
        ))
    cache_key = (alignment, tuple(cache_parts))
    cached = _FINGERPRINT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()

    def add(value: object) -> None:
        payload = str(value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)

    stats_by_key = dict(cache_parts)
    for ref in sorted(selected, key=lambda item: item.key):
        add((ref.key, ref.alignment, ref.rate_hz, ref.channels, ref.mask, ref.shape))
        for values in (ref.labels, ref.subjects):
            values_digest = hashlib.sha256()
            for value in values:
                encoded = value.encode("utf-8")
                values_digest.update(len(encoded).to_bytes(4, "little"))
                values_digest.update(encoded)
            add(values_digest.hexdigest())
        data_path = ref.grid_dir / "data.npy"
        add(data_path.stat().st_size)
        sampled_digest, lengths_digest = _grid_signal_digests(
            str(ref.grid_dir.resolve()), stats_by_key[ref.key],
        )
        add(sampled_digest)
        if lengths_digest is not None:
            add(lengths_digest)

    value = digest.hexdigest()
    _FINGERPRINT_CACHE[cache_key] = value
    return value


def triad_indices(ref: GridRef, modality: str) -> tuple[int, int, int] | None:
    """Return valid xyz indices for ``acc`` or ``gyro``, else ``None``."""
    names = tuple(f"{modality}_{axis}" for axis in "xyz")
    if not all(name in ref.channels for name in names):
        return None
    indices = tuple(ref.channels.index(name) for name in names)
    if not all(ref.mask[index] for index in indices):
        return None
    return indices


def matching_refs(
    refs: Iterable[GridRef],
    label: str,
    selectors: Sequence[str] | None = None,
) -> list[GridRef]:
    """Select grids containing ``label`` by dataset or dataset/stream selector."""
    eligible = [ref for ref in refs if label in ref.labels]
    if not selectors:
        return eligible

    selected: list[GridRef] = []
    for selector in selectors:
        for ref in eligible:
            if ref in selected:
                continue
            if selector == ref.dataset or selector == ref.key:
                selected.append(ref)
    return selected


def sample_indices(ref: GridRef, label: str, count: int, seed: int) -> np.ndarray:
    """Choose reproducible windows, preferring distinct subjects when possible."""
    candidates = np.fromiter(
        (index for index, value in enumerate(ref.labels) if value == label),
        dtype=np.int64,
    )
    if len(candidates) < count:
        raise ValueError(f"{ref.key}: requested {count} {label!r} windows, found {len(candidates)}")
    digest = hashlib.blake2b(
        f"{seed}:{ref.key}:{label}".encode("utf-8"), digest_size=8
    ).digest()
    local_seed = int.from_bytes(digest, "little")
    rng = np.random.default_rng(local_seed)

    by_subject: dict[str, list[int]] = {}
    for index in candidates:
        by_subject.setdefault(ref.subjects[int(index)], []).append(int(index))
    subject_order = np.asarray(sorted(by_subject), dtype=object)
    rng.shuffle(subject_order)

    chosen = [int(rng.choice(by_subject[str(subject)])) for subject in subject_order[:count]]
    if len(chosen) < count:
        remaining = np.asarray([index for index in candidates if int(index) not in chosen])
        chosen.extend(map(int, rng.choice(remaining, size=count - len(chosen), replace=False)))
    return np.sort(np.asarray(chosen, dtype=np.int64))


def output_dir(path: Path | None = None) -> Path:
    result = path or Path(__file__).resolve().parent / "outputs"
    result.mkdir(parents=True, exist_ok=True)
    return result


def output_subdir(root: Path, *parts: str) -> Path:
    """Create and return an analysis-specific directory below an output root."""
    result = root.joinpath(*parts)
    result.mkdir(parents=True, exist_ok=True)
    return result
