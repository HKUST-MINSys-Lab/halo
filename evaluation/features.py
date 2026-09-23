"""Feature extraction and on-disk/in-memory feature caches shared by every rung.

Extracted verbatim from training/support_classifier/sealed_eval.py on 2026-09-23 (Phase 0 of
docs/journal/2026-09-22-rung1-rung1-implementation-plan.md). sealed_eval re-imports these names.
"""

from __future__ import annotations

import baselines
import hashlib
import json
import numpy as np
import torch
import weakref
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Sequence
from baselines.data import EvalStream, MultiDeviceEvalStream, source_slice_fingerprint
from training.tokenizer.eval_transfer import (
    build_encoder,
    encode_dataset,
    encode_multi_device_dataset,
)
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions


# Bump whenever feature extraction semantics, cache inputs, or pooling changes.  This avoids
# treating an old embedding array as valid after a code-only correction.
# v6 includes the acquisition-conditioning schema in the encoder reconstruction contract. v5
# caches cannot distinguish historical combined sensor prose from v2 device/placement text.
FEATURE_CACHE_SCHEMA = "sealed-feature-v6-20260918"


UNCHANGED_BASELINE_FEATURE_CACHE_SCHEMA = "sealed-feature-v5-20260916"


def feature_cache_schema(model_name: str) -> str:
    """Invalidate HALO text-conditioned features without discarding unchanged baseline work."""
    return (FEATURE_CACHE_SCHEMA if model_name == "halo"
            else UNCHANGED_BASELINE_FEATURE_CACHE_SCHEMA)


class FeatureMemoryCache:
    """Bounded process-local cache for repeatedly referenced feature matrices.

    Scenario cells intentionally reuse the same immutable stream under several perturbation and
    support conditions.  Keeping recently used arrays avoids repeated ``np.load`` calls without
    making evaluation memory grow with the complete experiment.
    """

    def __init__(self, max_bytes: int = 2 * 1024**3):
        if max_bytes < 0:
            raise ValueError("feature memory-cache size must be non-negative")
        self.max_bytes = int(max_bytes)
        self._bytes = 0
        self._values: OrderedDict[str, np.ndarray] = OrderedDict()
        self._stream_fingerprints: dict[int, tuple[weakref.ReferenceType, str]] = {}

    def stream_fingerprint(self, stream) -> str:
        identity = id(stream)
        cached = self._stream_fingerprints.get(identity)
        if cached is not None and cached[0]() is stream:
            return cached[1]
        value = source_slice_fingerprint(stream)
        try:
            reference = weakref.ref(
                stream,
                lambda ref, key=identity: self._drop_stream_fingerprint(key, ref),
            )
        except TypeError:
            # Extension-owned stream wrappers need not support weak references. Recomputing their
            # fingerprint is preferable to retaining an unbounded strong-reference side cache.
            return value
        self._stream_fingerprints[identity] = (reference, value)
        return value

    def _drop_stream_fingerprint(self, identity: int, reference: weakref.ReferenceType) -> None:
        cached = self._stream_fingerprints.get(identity)
        if cached is not None and cached[0] is reference:
            del self._stream_fingerprints[identity]

    def get(self, key: str) -> np.ndarray | None:
        value = self._values.pop(key, None)
        if value is not None:
            self._values[key] = value
        return value

    def put(self, key: str, value: np.ndarray) -> None:
        if self.max_bytes == 0 or value.nbytes > self.max_bytes:
            return
        previous = self._values.pop(key, None)
        if previous is not None:
            self._bytes -= previous.nbytes
        self._values[key] = value
        self._bytes += value.nbytes
        while self._bytes > self.max_bytes:
            _, evicted = self._values.popitem(last=False)
            self._bytes -= evicted.nbytes


def _cache_key(
    name: str,
    stream: EvalStream,
    fingerprint: str,
    *,
    source_fingerprint: str | None = None,
    feature_role: str = "enrollment",
) -> str:
    devices = tuple(getattr(stream, "device_ids", (stream.stream,)))
    source_fingerprint = source_fingerprint or source_slice_fingerprint(stream)
    text = (f"{feature_cache_schema(name)}|{feature_role}|{name}|{stream.dataset}|{stream.stream}|{stream.alignment}|"
            f"{stream.window_seconds:g}|{devices}|{fingerprint}|{source_fingerprint}")
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@lru_cache(maxsize=128)
def _file_hash_for_stat(path_text: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    del size, mtime_ns, ctime_ns  # Cache-key material; digest covers the complete file.
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    # ctime closes the practical stale-cache hole where a tool preserves mtime while replacing a
    # same-sized checkpoint. It changes on inode metadata/content replacement on this platform.
    return _file_hash_for_stat(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _halo_features(
    stream: EvalStream,
    checkpoint: Path,
    device: torch.device,
    state: tuple[torch.nn.Module, str] | None = None,
) -> tuple[np.ndarray, str]:
    if state is None:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder = build_encoder(blob, device).eval()
        fingerprint = _file_hash(checkpoint)
    else:
        encoder, fingerprint = state
    if isinstance(stream, MultiDeviceEvalStream):
        features = encode_multi_device_dataset(
            encoder, stream.devices, device,
            amp_dtype=torch.bfloat16 if device.type == "cuda" else None,
        )
    else:
        features = encode_dataset(
            encoder, stream.windows, stream_channel_descriptions(stream.dataset, stream.stream), device,
            stream.rate_hz, _stream_gravity_state(stream.dataset, stream.stream),
            channel_mask=stream.mask, dataset=stream.dataset, stream=stream.stream,
            source_rate=(stream.effective_source_rate_hz
                         if stream.effective_source_rate_hz is not None else None),
            lengths=stream.lengths, amp_dtype=torch.bfloat16 if device.type == "cuda" else None,
            batch_size=512 if device.type == "cuda" else 256,
        )
    return np.asarray(features.cpu(), dtype=np.float32), fingerprint


def _baseline_feature_state(
    name: str,
    stream: EvalStream,
    device: torch.device,
    state: dict | None = None,
    feature_role: str = "enrollment",
):
    """Load a released feature provider and fingerprint it without encoding a stream."""
    adapter = baselines.REGISTRY[name]
    reason = adapter.incompatibility_for_stream(stream)
    if reason is not None:
        raise baselines.UnsupportedEvaluationCell(reason)
    state = adapter.setup_features(device) if state is None else state
    fingerprint_key = f"_feature_fingerprint_{feature_cache_schema(name)}_{feature_role}"
    fingerprint = state.get(fingerprint_key)
    if fingerprint is None:
        artifacts = (adapter.native_feature_artifacts(state) if feature_role == "native_zero_shot"
                     else adapter.feature_artifacts(state))
        config = (adapter.native_feature_config(state) if feature_role == "native_zero_shot"
                  else adapter.feature_config(state))
        fingerprint = hashlib.sha256(json.dumps({
            "artifacts": {key: _file_hash(Path(path)) for key, path in artifacts.items()},
            "config": config,
        }, sort_keys=True, default=str).encode()).hexdigest()
        state[fingerprint_key] = fingerprint
    return adapter, state, fingerprint


def _load_or_encode(
    *, name: str, stream: EvalStream, device: torch.device, cache_dir: Path,
    halo_checkpoint: Path | None, baseline_state: dict | None = None,
    halo_state: tuple[torch.nn.Module, str] | None = None,
    cache_read_dirs: Sequence[Path] = (),
    memory_cache: FeatureMemoryCache | None = None,
    feature_role: str = "enrollment",
) -> tuple[np.ndarray, str]:
    if name == "halo":
        if halo_checkpoint is None:
            raise ValueError("--halo-checkpoint is required when model list includes halo")
        probe = _file_hash(halo_checkpoint)
    else:
        # Feature artifacts can be verified only after adapter setup. Do not reuse a baseline
        # cache under a guessed key; a changed released checkpoint must force re-extraction.
        adapter, state, probe = _baseline_feature_state(
            name, stream, device, state=baseline_state, feature_role=feature_role,
        )
    source_fingerprint = (memory_cache.stream_fingerprint(stream) if memory_cache is not None
                          else source_slice_fingerprint(stream))
    key = _cache_key(name, stream, probe, source_fingerprint=source_fingerprint,
                     feature_role=feature_role)
    cache_schema = feature_cache_schema(name)
    filename = f"{stream.dataset}__{stream.stream}__{name}__{key}.npy"
    memory_key = f"{name}:{key}"
    if memory_cache is not None:
        cached = memory_cache.get(memory_key)
        if cached is not None:
            return cached, probe

    roots = (Path(cache_dir), *(Path(root) for root in cache_read_dirs if Path(root) != Path(cache_dir)))
    for root in roots:
        candidate = root / filename
        candidate_meta = candidate.with_suffix(".json")
        if not candidate.exists() or not candidate_meta.exists():
            continue
        meta = json.loads(candidate_meta.read_text())
        if (meta.get("cache_schema") != cache_schema or meta.get("cache_key") != key
                or meta.get("n_windows") != stream.n_windows
                or meta.get("source_slice_fingerprint") != source_fingerprint):
            continue
        cached = np.load(candidate)
        if cached.ndim == 2 and cached.shape[0] == stream.n_windows and np.isfinite(cached).all():
            if memory_cache is not None:
                memory_cache.put(memory_key, cached)
            return cached, str(meta["artifact_fingerprint"])
    if name == "halo":
        values, fingerprint = _halo_features(
            stream, halo_checkpoint, device, state=halo_state,
        )
    else:
        extractor = (adapter.native_zero_shot_features_for_stream
                     if feature_role == "native_zero_shot" else adapter.features_for_stream)
        values = np.asarray(extractor(stream, state, device), dtype=np.float32)
        fingerprint = probe
    if values.shape[0] != stream.n_windows or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name}: invalid feature matrix {values.shape} for {stream.dataset}/{stream.stream}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path = cache_dir / filename
    meta_path = array_path.with_suffix(".json")
    np.save(array_path, values)
    meta_path.write_text(json.dumps({"cache_schema": cache_schema, "cache_key": key, "n_windows": stream.n_windows,
                                     "artifact_fingerprint": fingerprint,
                                     "source_slice_fingerprint": source_fingerprint},
                                    indent=2) + "\n")
    if memory_cache is not None:
        memory_cache.put(memory_key, values)
    return values, fingerprint
