"""Protocol-controlled sealed evaluation for support-conditioned HAR.

This is intentionally separate from ``train.py``.  It is the only entry point allowed to read the
six sealed sources, creates one immutable support/query manifest per stream and k, and applies the
same readouts to every representation provider.  It does not select checkpoints or tune settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

import baselines
from baselines import scoring
from baselines.data import EvalStream, list_streams, load_eval_stream, load_global_labels
from data.scripts.curate.deployment_policy import (
    SEALED_TEST_EVAL_DATASETS,
    SUPERVISED_HEAD_TRAIN_DATASETS,
    assert_no_retired_sources,
    stream_specs,
)
from model.blocks import AttentionSpec
from model.support.token_mixer import SupportTokenMixer, TokenMixerConfig
from training.support_classifier.train import make_label_text
from training.support_classifier.neighbors import differentiable_neighbor_logits
from training.support_classifier.representation_diagnostics import write_embedding_diagnostics
from training.support_classifier.sampling import MIN_RECORDING_SECONDS
from training.tokenizer.eval_transfer import build_encoder, encode_dataset
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

SEED = 20260912
# Full enrollment curve. All six sealed sources have at least 128 execution-disjoint
# support windows per candidate for every currently valid query (verified 2026-09-12).
# ``build_manifest`` still fails closed per query if a future corpus revision no longer
# satisfies a requested support count.
DEFAULT_K = (0, 1, 2, 4, 8, 16, 32, 64, 128)
PRIMARY_BASELINES = ("harnet", "limubert_x", "unimts", "normwear")
TRAINING_BANK_ZERO_SHOT = frozenset({"halo", "harnet", "limubert_x"})


@dataclass(frozen=True)
class QueryPlan:
    """One query and its execution-disjoint enrolled rows for every candidate."""

    query: int
    support: tuple[int, ...]
    support_labels: tuple[str, ...]


def sealed_cells() -> tuple[tuple[str, str], ...]:
    """The executable sealed roster; no caller-supplied datasets are accepted."""
    return tuple(
        (dataset, spec.stream_id)
        for dataset in SEALED_TEST_EVAL_DATASETS
        for spec in stream_specs(dataset, "primary")
    )


def _aligned_labels(stream: EvalStream) -> np.ndarray:
    return np.asarray(
        scoring.align_ground_truth_labels(stream.gt, stream.eval_labels), dtype=object,
    )


def _stable_choice(values: np.ndarray, count: int, *, seed_parts: Sequence[object]) -> np.ndarray:
    digest = hashlib.sha256("|".join(map(str, seed_parts)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.asarray(rng.choice(values, size=count, replace=False), dtype=np.int64)


def build_manifest(stream: EvalStream, k: int, *, seed: int = SEED) -> list[QueryPlan]:
    """Create execution-disjoint target-dataset episodes without looking at representations.

    Each target candidate receives exactly k labelled support *windows* from a different physical
    execution than the query.  The candidate roster is always the frozen native vocabulary.  A
    query is omitted when that honest episode cannot be formed; this is reported rather than
    padded or silently relaxed.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return [QueryPlan(query=i, support=(), support_labels=()) for i in range(stream.n_windows)]
    if not stream.execution_identity_known or stream.execution_ids is None:
        raise ValueError(
            f"{stream.dataset}/{stream.stream}: execution identity is unavailable; refusing "
            "enrollment evaluation that could leak a recording into its own support set"
        )
    labels = _aligned_labels(stream)
    valid = np.flatnonzero(labels != None)  # noqa: E711 - object-array comparison is intentional
    plans: list[QueryPlan] = []
    candidates = tuple(stream.eval_labels)
    # Index once. The original direct expression scanned every valid row for every
    # (query, candidate) pair, which turns the optional high-k curve into needless quadratic CPU
    # work. A candidate-local index preserves exactly the same execution-disjoint rule.
    label_rows = {
        label: valid[labels[valid] == label]
        for label in candidates
    }
    label_execution = {
        label: np.asarray(stream.execution_ids[rows], dtype=object)
        for label, rows in label_rows.items()
    }
    for query in valid.tolist():
        q_execution = stream.execution_ids[query]
        support: list[int] = []
        support_labels: list[str] = []
        possible = True
        for label in candidates:
            rows = label_rows[label]
            pool = rows[label_execution[label] != q_execution]
            if len(pool) < k:
                possible = False
                break
            picked = _stable_choice(
                pool, k, seed_parts=(seed, stream.dataset, stream.stream, k, query, label),
            )
            support.extend(int(value) for value in picked)
            support_labels.extend([label] * k)
        if possible:
            plans.append(QueryPlan(query=query, support=tuple(support),
                                   support_labels=tuple(support_labels)))
    return plans


def manifest_fingerprint(plans: Iterable[QueryPlan]) -> str:
    payload = [asdict(plan) for plan in plans]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _normalise(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


@torch.no_grad()
def _training_bank_conse_predictions(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    reference_label_ids: np.ndarray,
    train_labels: Sequence[str],
    target_labels: Sequence[str],
    device: torch.device,
    *,
    query_batch_size: int = 1024,
    reference_batch_size: int = 32768,
) -> tuple[list[str], dict]:
    """Bridge a training-bank nearest neighbour onto an unseen candidate vocabulary.

    ``k=0`` means zero *target-dataset enrollment*, not zero prior labelled evidence. The closest
    training-corpus recording supplies a distribution over the fixed training vocabulary; ConSE
    then maps that distribution to the target label strings. Chunking the exact matrix search
    bounds VRAM without changing the selected neighbour.
    """
    query = _normalise(query_features).astype(np.float32, copy=False)
    reference = _normalise(reference_features).astype(np.float32, copy=False)
    label_ids = np.asarray(reference_label_ids, dtype=np.int64)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("query and training-bank features must be matching matrices")
    if label_ids.shape != (len(reference),) or not len(reference):
        raise ValueError("training-bank labels must align with a non-empty feature matrix")
    if label_ids.min() < 0 or label_ids.max() >= len(train_labels):
        raise ValueError("training-bank label id outside the declared training vocabulary")

    nearest: list[np.ndarray] = []
    reference_device = [
        torch.as_tensor(reference[start:start + reference_batch_size], device=device)
        for start in range(0, len(reference), reference_batch_size)
    ]
    for start in range(0, len(query), query_batch_size):
        q = torch.as_tensor(query[start:start + query_batch_size], device=device)
        best_score = torch.full((len(q),), -torch.inf, device=device)
        best_row = torch.zeros((len(q),), dtype=torch.long, device=device)
        offset = 0
        for block in reference_device:
            score, row = (q @ block.T).max(dim=1)
            replace = score > best_score
            best_score = torch.where(replace, score, best_score)
            best_row = torch.where(replace, row + offset, best_row)
            offset += len(block)
        nearest.append(best_row.cpu().numpy())
    nearest_rows = np.concatenate(nearest)
    probs = np.zeros((len(query), len(train_labels)), dtype=np.float32)
    probs[np.arange(len(query)), label_ids[nearest_rows]] = 1.0
    predictions, info = scoring.conse_predict(probs, train_labels, target_labels, top_T=1)
    info.update({
        "zero_support_protocol": "training_bank_1nn_conse_v1",
        "reference_rows": int(len(reference)),
        "reference_labels": int(len(np.unique(label_ids))),
    })
    del reference_device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, info


def _readout_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    candidates: Sequence[str],
    plans: Sequence[QueryPlan],
    *,
    ridge_alpha: float = 1.0,
    device: torch.device | None = None,
) -> dict[str, list[str]]:
    """Matched enrollment readouts over one frozen representation matrix."""
    z = _normalise(features)
    result = {"1nn": [], "prototype": [], "ridge": []}
    candidate_index = {label: index for index, label in enumerate(candidates)}
    for plan in plans:
        query = z[plan.query]
        support = np.asarray(plan.support, dtype=np.int64)
        if not len(support):
            continue
        x = z[support]
        y = np.asarray([candidate_index[label] for label in plan.support_labels], dtype=np.int64)
        result["1nn"].append(str(plan.support_labels[int(np.argmax(x @ query))]))
        prototypes = np.stack([
            x[y == slot].mean(axis=0) for slot in range(len(candidates))
        ])
        result["prototype"].append(str(candidates[int(np.argmax(_normalise(prototypes) @ query))]))
        if device is None:
            # Reference path used by unit tests and CPU-only callers.
            target = np.eye(len(candidates), dtype=np.float64)[y]
            if len(x) <= x.shape[1]:
                gram = x @ x.T + ridge_alpha * np.eye(len(x), dtype=np.float64)
                coefficient = np.linalg.solve(gram, target)
                ridge_scores = query @ x.T @ coefficient
            else:
                gram = x.T @ x + ridge_alpha * np.eye(x.shape[1], dtype=np.float64)
                coefficient = np.linalg.solve(gram, x.T @ target)
                ridge_scores = query @ coefficient
            result["ridge"].append(str(candidates[int(np.argmax(ridge_scores))]))
    if device is not None:
        result["ridge"] = _ridge_predictions_batched(
            z, candidates, plans, device, ridge_alpha=ridge_alpha,
        )
    return result


@torch.no_grad()
def _ridge_predictions_batched(
    normalized_features: np.ndarray,
    candidates: Sequence[str],
    plans: Sequence[QueryPlan],
    device: torch.device,
    *,
    ridge_alpha: float,
    batch_size: int = 32,
) -> list[str]:
    """GPU-batched ridge readout for the large enrollment curve.

    Episodes keep their immutable, query-specific supports.  Only the independent ridge solves
    are batched; this is algebraically the same primal/dual ridge classifier as the reference
    path above.  ``alpha=1`` keeps the float32 systems well-conditioned on the normalized inputs.
    """
    if not plans:
        return []
    candidate_index = {label: index for index, label in enumerate(candidates)}
    output: list[str] = []
    eye_cache: dict[int, torch.Tensor] = {}
    for start in range(0, len(plans), batch_size):
        chunk = plans[start:start + batch_size]
        support_rows = np.asarray([plan.support for plan in chunk], dtype=np.int64)
        labels = np.asarray(
            [[candidate_index[label] for label in plan.support_labels] for plan in chunk],
            dtype=np.int64,
        )
        query_rows = np.asarray([plan.query for plan in chunk], dtype=np.int64)
        x = torch.as_tensor(normalized_features[support_rows], dtype=torch.float32, device=device)
        query = torch.as_tensor(normalized_features[query_rows], dtype=torch.float32, device=device)
        target = torch.nn.functional.one_hot(
            torch.as_tensor(labels, device=device), num_classes=len(candidates),
        ).to(dtype=x.dtype)
        support_count, dim = x.shape[1], x.shape[2]
        if support_count <= dim:
            if support_count not in eye_cache:
                eye_cache[support_count] = torch.eye(support_count, dtype=x.dtype, device=device)
            gram = x @ x.transpose(-1, -2) + ridge_alpha * eye_cache[support_count]
            coefficient = torch.linalg.solve(gram, target)
            scores = (query.unsqueeze(1) @ x.transpose(-1, -2) @ coefficient).squeeze(1)
        else:
            if dim not in eye_cache:
                eye_cache[dim] = torch.eye(dim, dtype=x.dtype, device=device)
            gram = x.transpose(-1, -2) @ x + ridge_alpha * eye_cache[dim]
            coefficient = torch.linalg.solve(gram, x.transpose(-1, -2) @ target)
            scores = (query.unsqueeze(1) @ coefficient).squeeze(1)
        output.extend(candidates[index] for index in scores.argmax(dim=1).cpu().tolist())
    return output


@torch.no_grad()
def _differentiable_neighbor_predictions(
    features: np.ndarray,
    candidates: Sequence[str],
    plans: Sequence[QueryPlan],
    device: torch.device,
    *,
    batch_size: int = 256,
) -> list[str]:
    """Apply the exact parameter-free training control to an immutable enrolled manifest."""
    if any(not plan.support for plan in plans):
        raise ValueError("differentiable neighbours are undefined for zero-support episodes")
    normalized = _normalise(features).astype(np.float32, copy=False)
    candidate_index = {label: index for index, label in enumerate(candidates)}
    output: list[str] = []
    for start in range(0, len(plans), batch_size):
        chunk = plans[start:start + batch_size]
        query_rows = np.asarray([plan.query for plan in chunk], dtype=np.int64)
        support_rows = np.asarray([plan.support for plan in chunk], dtype=np.int64)
        support_bound = torch.as_tensor(
            [[candidate_index[label] for label in plan.support_labels] for plan in chunk],
            dtype=torch.long, device=device,
        )
        logits, _ = differentiable_neighbor_logits(
            torch.as_tensor(normalized[query_rows], device=device),
            torch.as_tensor(normalized[support_rows], device=device),
            support_bound,
            torch.ones(support_bound.shape, dtype=torch.bool, device=device),
            torch.ones((len(chunk), len(candidates)), dtype=torch.bool, device=device),
        )
        output.extend(candidates[index] for index in logits.argmax(dim=1).cpu().tolist())
    return output


def _metric_row(
    stream: EvalStream,
    plans: Sequence[QueryPlan],
    predictions: Sequence[str],
    *,
    bootstrap: int,
) -> dict:
    indices = np.asarray([plan.query for plan in plans], dtype=np.int64)
    truth = _aligned_labels(stream)[indices].tolist()
    subjects = np.asarray(stream.subjects)[indices]
    metrics = scoring.classification_metrics(truth, list(predictions))
    metrics.update(scoring.subject_bootstrap_ci(
        truth, list(predictions), subjects, metric="f1_macro", B=bootstrap,
    ))
    metrics.update({
        "dataset": stream.dataset,
        "stream": stream.stream,
        "n_queries": int(len(indices)),
        "n_candidates": int(len(stream.eval_labels)),
        "quality_screen": stream.quality_screen,
        "quality_excluded": int(stream.n_quality_excluded),
    })
    return metrics


def _cache_key(name: str, stream: EvalStream, fingerprint: str) -> str:
    text = f"{name}|{stream.dataset}|{stream.stream}|{stream.alignment}|{fingerprint}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    features = encode_dataset(
        encoder, stream.windows, stream_channel_descriptions(stream.dataset, stream.stream), device,
        stream.rate_hz, _stream_gravity_state(stream.dataset, stream.stream),
        channel_mask=stream.mask, dataset=stream.dataset, stream=stream.stream,
        lengths=stream.lengths, amp_dtype=torch.bfloat16 if device.type == "cuda" else None,
    )
    return np.asarray(features.cpu(), dtype=np.float32), fingerprint


@torch.no_grad()
def _halo_token_mixer_predictions(
    features: np.ndarray,
    stream: EvalStream,
    plans: Sequence[QueryPlan],
    checkpoint: Path,
    device: torch.device,
    batch_size: int = 64,
) -> list[str]:
    """Run HALO's semantic token mixer on an immutable manifest.

    The zero-support path receives only the query and the declared candidate labels.  The enrolled
    path receives every support recording plus its paired label token and the candidate tokens;
    it never has an implicit retrieval bank outside the manifest.
    """
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if blob.get("architecture_version") != "support_token_mixer_v1" \
            or "classifier" not in blob or "classifier_config" not in blob \
            or "attention_spec" not in blob:
        raise ValueError("HALO token-mixer readout requires a current support-classifier checkpoint")
    if blob["classifier"] is None or blob["classifier_config"] is None:
        raise ValueError("the checkpoint contains the differentiable-neighbors control, not a token mixer")
    mixer = SupportTokenMixer(
        AttentionSpec(**blob["attention_spec"]), TokenMixerConfig(**blob["classifier_config"]),
    ).to(device).eval()
    mixer.load_state_dict(blob["classifier"])
    candidates = tuple(stream.eval_labels)
    table = make_label_text(candidates, device)
    candidate_text = table.matrix[torch.as_tensor(table.ids(candidates), device=device)]
    candidate_text = candidate_text.unsqueeze(0)
    c = len(candidates)
    candidate_mask = torch.ones((1, c), dtype=torch.bool, device=device)
    label_to_slot = {label: slot for slot, label in enumerate(candidates)}
    output: list[str] = []
    for start in range(0, len(plans), batch_size):
        chunk = plans[start:start + batch_size]
        query_rows = np.asarray([plan.query for plan in chunk], dtype=np.int64)
        b = len(chunk)
        query_feature = torch.as_tensor(features[query_rows], dtype=torch.float32, device=device)
        common = {
            "is_zero_shot": torch.tensor([not plan.support for plan in chunk],
                                         dtype=torch.bool, device=device),
            "query_feature": query_feature,
            "candidate_text": candidate_text.expand(b, -1, -1),
            "candidate_mask": candidate_mask.expand(b, -1),
            "candidate_slot": (1 + torch.arange(c, device=device)).unsqueeze(0).expand(b, -1),
        }
        if any(plan.support for plan in chunk):
            if not all(plan.support for plan in chunk):
                raise ValueError("manifest batches must not mix zero-shot and enrolled plans")
            support_rows = np.asarray([plan.support for plan in chunk], dtype=np.int64)
            support_label_ids = torch.as_tensor(
                [[label_to_slot[label] for label in plan.support_labels] for plan in chunk],
                dtype=torch.long, device=device,
            )
            common.update({
                "support_feature": torch.as_tensor(features[support_rows], dtype=torch.float32, device=device),
                "support_label_text": candidate_text.expand(b, -1, -1).gather(
                    1, support_label_ids.unsqueeze(-1).expand(-1, -1, candidate_text.shape[-1]),
                ),
                "support_bound": support_label_ids,
                "support_mask": torch.ones(support_label_ids.shape, dtype=torch.bool, device=device),
                "support_pair_slot": (1 + torch.arange(support_label_ids.shape[1], device=device))
                    .unsqueeze(0).expand(b, -1),
            })
        logits = mixer(**common)["logits"]
        output.extend(candidates[index] for index in logits.argmax(dim=1).cpu().tolist())
    return output


def _baseline_feature_state(
    name: str,
    stream: EvalStream,
    device: torch.device,
    state: dict | None = None,
):
    """Load a released feature provider and fingerprint it without encoding a stream."""
    adapter = baselines.REGISTRY[name]
    reason = adapter.is_incompatible(stream.dataset)
    if reason is not None:
        raise baselines.UnsupportedEvaluationCell(reason)
    state = adapter.setup_features(device) if state is None else state
    artifacts = adapter.feature_artifacts(state)
    fingerprint = hashlib.sha256(json.dumps({
        "artifacts": {key: _file_hash(Path(path)) for key, path in artifacts.items()},
        "config": adapter.feature_config(state),
    }, sort_keys=True, default=str).encode()).hexdigest()
    return adapter, state, fingerprint


def _load_or_encode(
    *, name: str, stream: EvalStream, device: torch.device, cache_dir: Path,
    halo_checkpoint: Path | None, baseline_state: dict | None = None,
    halo_state: tuple[torch.nn.Module, str] | None = None,
) -> tuple[np.ndarray, str]:
    if name == "halo":
        if halo_checkpoint is None:
            raise ValueError("--halo-checkpoint is required when model list includes halo")
        probe = _file_hash(halo_checkpoint)
    else:
        # Feature artifacts can be verified only after adapter setup. Do not reuse a baseline
        # cache under a guessed key; a changed released checkpoint must force re-extraction.
        adapter, state, probe = _baseline_feature_state(
            name, stream, device, state=baseline_state,
        )
    key = _cache_key(name, stream, probe)
    array_path = cache_dir / f"{stream.dataset}__{stream.stream}__{name}__{key}.npy"
    meta_path = array_path.with_suffix(".json")
    if array_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("cache_key") == key and meta.get("n_windows") == stream.n_windows:
            return np.load(array_path), str(meta["artifact_fingerprint"])
    if name == "halo":
        values, fingerprint = _halo_features(
            stream, halo_checkpoint, device, state=halo_state,
        )
    else:
        values = np.asarray(adapter.window_features(stream, state, device), dtype=np.float32)
        fingerprint = probe
    if values.shape[0] != stream.n_windows or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name}: invalid feature matrix {values.shape} for {stream.dataset}/{stream.stream}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(array_path, values)
    meta_path.write_text(json.dumps({"cache_key": key, "n_windows": stream.n_windows,
                                     "artifact_fingerprint": fingerprint}, indent=2) + "\n")
    return values, fingerprint


def _build_training_reference_bank(
    *,
    name: str,
    device: torch.device,
    cache_dir: Path,
    halo_checkpoint: Path | None,
    halo_state: tuple[torch.nn.Module, str] | None = None,
    baseline_state: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Encode the active labelled training roster for zero-target-enrollment scoring."""
    assert_no_retired_sources(SUPERVISED_HEAD_TRAIN_DATASETS)
    train_labels = load_global_labels()
    label_to_id = {label: index for index, label in enumerate(train_labels)}
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    provenance: list[dict] = []
    excluded: list[dict] = []
    for dataset in SUPERVISED_HEAD_TRAIN_DATASETS:
        streams = list_streams(dataset, alignment="native")
        if not streams:
            raise FileNotFoundError(f"zero-shot training bank has no native grid for {dataset}")
        for stream_id in streams:
            stream = load_eval_stream(
                dataset, stream_id, alignment="native", apply_quality_screen=True,
                candidate_labels=train_labels,
            )
            if stream.quality_screen != "applied":
                raise RuntimeError(
                    f"{dataset}/{stream_id}: quality screen unavailable ({stream.quality_screen})"
                )
            aligned = np.asarray(
                scoring.align_ground_truth_labels(stream.gt, train_labels), dtype=object,
            )
            lengths = (
                np.asarray(stream.lengths, dtype=np.int64)
                if stream.lengths is not None
                else np.full(stream.n_windows, stream.windows.shape[1], dtype=np.int64)
            )
            eligible_duration = lengths / float(stream.rate_hz) >= MIN_RECORDING_SECONDS
            keep = np.flatnonzero((aligned != None) & eligible_duration)  # noqa: E711
            if not len(keep):
                continue
            bank_stream = replace(
                stream,
                windows=stream.windows[keep],
                gt=[stream.gt[row] for row in keep],
                subjects=np.asarray(stream.subjects)[keep],
                event_ids=(np.asarray(stream.event_ids)[keep] if stream.event_ids is not None else None),
                execution_ids=(
                    np.asarray(stream.execution_ids)[keep]
                    if stream.execution_ids is not None else None
                ),
                block_ids=(np.asarray(stream.block_ids)[keep] if stream.block_ids is not None else None),
                lengths=lengths[keep],
                perturbation=f"training-bank-min-{MIN_RECORDING_SECONDS:g}s",
            )
            if name == "limubert_x":
                required = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
                channel_index = {channel: index for index, channel in enumerate(bank_stream.channels)}
                missing = [channel for channel in required if channel not in channel_index]
                masked = [
                    channel for channel in required
                    if channel in channel_index and not bool(bank_stream.mask[channel_index[channel]])
                ]
                if missing or masked:
                    excluded.append({
                        "dataset": dataset,
                        "stream": stream_id,
                        "reason": f"LiMU-BERT-X requires measured six-axis IMU; "
                                  f"missing={missing}, masked={masked}",
                    })
                    continue
            try:
                features, fingerprint = _load_or_encode(
                    name=name,
                    stream=bank_stream,
                    device=device,
                    cache_dir=cache_dir,
                    halo_checkpoint=halo_checkpoint,
                    halo_state=halo_state,
                    baseline_state=baseline_state,
                )
            except baselines.UnsupportedEvaluationCell as error:
                excluded.append({"dataset": dataset, "stream": stream_id, "reason": str(error)})
                continue
            feature_parts.append(np.asarray(features, dtype=np.float32))
            label_parts.append(np.asarray([label_to_id[str(aligned[row])] for row in keep], dtype=np.int64))
            provenance.append({
                "dataset": dataset,
                "stream": stream_id,
                "rows": int(len(keep)),
                "feature_fingerprint": fingerprint,
                "quality_excluded": int(stream.n_quality_excluded),
            })
    if not feature_parts:
        raise RuntimeError(f"no valid labelled training references for {name}")
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    fingerprint = hashlib.sha256(json.dumps({
        "provider": name,
        "roster": list(SUPERVISED_HEAD_TRAIN_DATASETS),
        "vocabulary": train_labels,
        "parts": provenance,
        "excluded": excluded,
        "protocol": "training_bank_1nn_conse_v1",
    }, sort_keys=True).encode()).hexdigest()
    return features, labels, train_labels, fingerprint


def _write_markdown(rows: Sequence[dict], path: Path) -> None:
    columns = ("model", "readout", "k", "dataset", "stream", "f1_macro", "balanced_accuracy",
               "f1_macro_ci_lo", "f1_macro_ci_hi", "n_queries", "n_candidates", "status")
    lines = ["# Sealed support-conditioned HAR results", "",
             "Generated by `training.support_classifier.sealed_eval`; no values select a checkpoint.", "",
             "| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_BASELINES))
    parser.add_argument("--halo-checkpoint", type=Path, default=None)
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap", type=int, default=scoring.BOOTSTRAP_B)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="optional shared sealed-stream feature cache; defaults to OUT/feature_cache",
    )
    parser.add_argument(
        "--zero-shot-bank-cache",
        type=Path,
        default=Path("training/support_classifier/evaluations/zero_shot_feature_cache"),
        help="shared cache for labelled training-corpus representations used only at k=0",
    )
    parser.add_argument("--embedding-diagnostics", action="store_true",
                        help="write opt-in representation figures beside each encoded stream; this never "
                             "changes predictions, manifests, or checkpoint selection")
    args = parser.parse_args()
    if not args.k or min(args.k) < 0:
        parser.error("--k must contain non-negative support counts")
    unknown = sorted(set(args.models) - set(PRIMARY_BASELINES) - {"halo"})
    if unknown:
        parser.error(f"models outside the registered primary roster: {unknown}")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not repeat a provider")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache or args.out / "feature_cache"
    halo_state: tuple[torch.nn.Module, str] | None = None
    if "halo" in args.models:
        if args.halo_checkpoint is None:
            parser.error("--halo-checkpoint is required when model list includes halo")
        halo_blob = torch.load(args.halo_checkpoint, map_location="cpu", weights_only=False)
        halo_state = (
            build_encoder(halo_blob, device).eval(),
            _file_hash(args.halo_checkpoint),
        )
    # A very large native model (notably NormWear) must be instantiated once and reused across
    # sealed streams. The same state also serves training-bank encoding so ordinary released
    # encoders are not reconstructed once per source stream.
    persistent_states: dict[str, dict] = {}
    if len(args.models) == 1 and args.models[0] != "halo":
        name = args.models[0]
        persistent_states[name] = baselines.REGISTRY[name].setup_features(device)
    zero_shot_banks: dict[str, tuple[np.ndarray, np.ndarray, list[str], str]] = {}
    if 0 in args.k:
        for name in args.models:
            if name not in TRAINING_BANK_ZERO_SHOT:
                continue
            print(f"[sealed-eval] building/loading k=0 training bank for {name}", flush=True)
            zero_shot_banks[name] = _build_training_reference_bank(
                name=name,
                device=device,
                cache_dir=args.zero_shot_bank_cache,
                halo_checkpoint=args.halo_checkpoint,
                halo_state=halo_state,
                baseline_state=persistent_states.get(name),
            )
    all_rows: list[dict] = []
    manifests: dict[str, dict] = {}
    for dataset, stream_id in sealed_cells():
        stream = load_eval_stream(dataset, stream_id, alignment="native", apply_quality_screen=True)
        if stream.quality_screen != "applied":
            raise RuntimeError(f"{dataset}/{stream_id}: quality screen unavailable ({stream.quality_screen})")
        # A representation depends only on the provider and the stream, never on enrollment k.
        # Encode/cache it once, then run all protocol readouts on immutable manifests.
        features_by_model: dict[str, tuple[np.ndarray, str]] = {}
        feature_errors: dict[str, str] = {}
        # Enrollment readouts and the training-bank bridge consume recording representations.
        # Native text-aligned baselines can bypass them at k=0.
        if args.models:
            for name in args.models:
                try:
                    features_by_model[name] = _load_or_encode(
                        name=name, stream=stream, device=device, cache_dir=cache_dir,
                        halo_checkpoint=args.halo_checkpoint,
                        baseline_state=persistent_states.get(name),
                        halo_state=halo_state,
                    )
                    if args.embedding_diagnostics:
                        features, _ = features_by_model[name]
                        write_embedding_diagnostics(
                            features, _aligned_labels(stream),
                            args.out / "embedding_diagnostics" / name / dataset / stream_id,
                            title=f"{name}: {dataset}/{stream_id}", seed=args.seed,
                        )
                except baselines.UnsupportedEvaluationCell as exc:
                    feature_errors[name] = str(exc)
        for k in sorted(set(args.k)):
            plans = build_manifest(stream, k, seed=args.seed)
            manifest_id = f"{dataset}/{stream_id}/k={k}"
            manifests[manifest_id] = {"fingerprint": manifest_fingerprint(plans),
                                      "n_queries": len(plans), "k": k,
                                      "candidates": stream.eval_labels,
                                      # The complete row lists can be regenerated exactly from
                                      # the sealed stream, this protocol version, and ``seed``.
                                      # Persisting them duplicated gigabytes of index data at
                                      # k=64/128 without adding audit value.
                                      "seed": args.seed,
                                      "construction": "execution_disjoint_stable_choice_v1"}
            if not plans:
                for name in args.models:
                    all_rows.append({"model": name, "readout": "all", "k": k,
                                     "dataset": dataset, "stream": stream_id,
                                     "status": "n/a", "reason": "no honest execution-disjoint episode"})
                continue
            for name in args.models:
                if k == 0:
                    if name in TRAINING_BANK_ZERO_SHOT:
                        if name in feature_errors:
                            all_rows.append({"model": name, "readout": "training-bank-1nn-conse", "k": k,
                                             "dataset": dataset, "stream": stream_id,
                                             "status": "n/a", "reason": feature_errors[name]})
                            continue
                        features, fingerprint = features_by_model[name]
                        bank_features, bank_labels, train_labels, bank_fingerprint = zero_shot_banks[name]
                        predicted, info = _training_bank_conse_predictions(
                            features, bank_features, bank_labels, train_labels,
                            stream.eval_labels, device,
                        )
                        metric = _metric_row(stream, plans, predicted, bootstrap=args.bootstrap)
                        metric.update({
                            "model": name,
                            "readout": "training-bank-1nn-conse",
                            "k": k,
                            "status": "ok",
                            "feature_fingerprint": fingerprint,
                            "training_bank_fingerprint": bank_fingerprint,
                            "manifest": manifests[manifest_id]["fingerprint"],
                            **info,
                        })
                        all_rows.append(metric)
                        continue
                    # Text-aligned released models retain their own candidate-scoring mechanism.
                    if not baselines.REGISTRY[name].supports_native_zero_shot():
                        all_rows.append({"model": name, "readout": "native_zero_support", "k": k,
                                         "dataset": dataset, "stream": stream_id, "status": "n/a",
                                         "reason": "no declared native zero-support path"})
                        continue
                    if name in feature_errors:
                        all_rows.append({"model": name, "readout": "native_zero_support", "k": k,
                                         "dataset": dataset, "stream": stream_id, "status": "n/a",
                                         "reason": feature_errors[name]})
                        continue
                    adapter = baselines.REGISTRY[name]
                    state = persistent_states.get(name)
                    if state is None:
                        state = adapter.setup(device)
                    features, fingerprint = features_by_model[name]
                    prediction, _ = adapter.predict_candidates_from_features(
                        features, stream.eval_labels, state, device,
                    )
                    aligned = _aligned_labels(stream)
                    keep = np.flatnonzero(aligned != None)  # noqa: E711
                    metric = scoring.classification_metrics(aligned[keep].tolist(), [prediction[i] for i in keep])
                    metric.update(scoring.subject_bootstrap_ci(aligned[keep].tolist(), [prediction[i] for i in keep],
                                                               stream.subjects[keep], B=args.bootstrap))
                    metric.update({"model": name, "readout": "native_zero_support", "k": k,
                                   "dataset": dataset, "stream": stream_id, "status": "ok",
                                   "n_queries": int(len(keep)), "n_candidates": len(stream.eval_labels),
                                   "feature_fingerprint": fingerprint})
                    all_rows.append(metric)
                    if name not in persistent_states:
                        del state
                    if device.type == "cuda" and name not in persistent_states:
                        torch.cuda.empty_cache()
                    continue
                if name in feature_errors:
                    all_rows.append({"model": name, "readout": "all", "k": k,
                                     "dataset": dataset, "stream": stream_id,
                                     "status": "n/a", "reason": feature_errors[name]})
                    continue
                features, fingerprint = features_by_model[name]
                predictions = _readout_predictions(
                    features, _aligned_labels(stream), stream.eval_labels, plans, device=device,
                )
                for readout, predicted in predictions.items():
                    metric = _metric_row(stream, plans, predicted, bootstrap=args.bootstrap)
                    metric.update({"model": name, "readout": readout, "k": k,
                                   "status": "ok", "feature_fingerprint": fingerprint,
                                   "manifest": manifests[manifest_id]["fingerprint"]})
                    all_rows.append(metric)
                predicted = _differentiable_neighbor_predictions(
                    features, stream.eval_labels, plans, device,
                )
                metric = _metric_row(stream, plans, predicted, bootstrap=args.bootstrap)
                metric.update({"model": name, "readout": "differentiable-neighbors", "k": k,
                               "status": "ok", "feature_fingerprint": fingerprint,
                               "manifest": manifests[manifest_id]["fingerprint"]})
                all_rows.append(metric)
                if name == "halo":
                    try:
                        predicted = _halo_token_mixer_predictions(
                            features, stream, plans, args.halo_checkpoint, device,
                        )
                    except ValueError as exc:
                        all_rows.append({"model": name, "readout": "retrieve-mix-vote", "k": k,
                                         "dataset": dataset, "stream": stream_id,
                                         "status": "n/a", "reason": str(exc)})
                    else:
                        metric = _metric_row(stream, plans, predicted, bootstrap=args.bootstrap)
                        metric.update({"model": name, "readout": "retrieve-mix-vote", "k": k,
                                       "status": "ok", "feature_fingerprint": fingerprint,
                                       "manifest": manifests[manifest_id]["fingerprint"]})
                        all_rows.append(metric)
    (args.out / "episode_manifests.json").write_text(json.dumps(manifests, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(all_rows, indent=2, allow_nan=True) + "\n")
    _write_markdown(all_rows, args.out / "RESULTS.md")
    print(f"[sealed-eval] wrote {args.out / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
