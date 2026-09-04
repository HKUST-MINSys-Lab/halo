"""Run zero-shot and few-shot external baselines on a shared enrollment manifest.

The runner never constructs an episode.  It consumes the exact support/query rows serialized by
``eval.enrollment_protocol`` and therefore cannot silently give different methods easier candidate
sets, support executions, or query cohorts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

import baselines
from baselines.base import UnsupportedEvaluationCell
from eval.data import load_eval_stream
from eval.enrollment_protocol import iter_cells, load_manifest
from eval.scoring import align_ground_truth_labels, classification_metrics


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "eval" / "adaptation_results"
DEFAULT_FEATURE_CACHE = REPO / "eval" / "adaptation_feature_cache"
METHODS = ("nearest", "prototype", "ridge", "linear_head")
DEFAULT_METHODS = ("nearest", "prototype", "ridge")
NATIVE_METHOD = "support_comparator"
LINEAR_HEAD_STEPS = 200
LINEAR_HEAD_LR = 5e-2
LINEAR_HEAD_WEIGHT_DECAY = 1e-3
LINEAR_EPISODE_BATCH = 32


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    os.replace(temporary, path)


def _source_fingerprint(adapter) -> str:
    digest = hashlib.sha256()
    fixed = [
        "eval/run_adaptation_baselines.py",
        "eval/enrollment_protocol.py",
        "baselines/base.py",
        "eval/scoring.py",
        "data/labels/global_labels.json",
    ]
    module_parts = type(adapter).__module__.split(".")
    baseline_dir = REPO.joinpath(*module_parts[:-1])
    sources: list[tuple[str, Path | None]] = [
        (relative, REPO / relative) for relative in fixed
    ]
    if baseline_dir.exists():
        sources.extend(
            (path.relative_to(REPO).as_posix(), path)
            for path in sorted(baseline_dir.rglob("*.py"))
        )
    else:
        relative = f"baselines/{adapter.name}/adapter.py"
        sources.append((relative, REPO / relative))
    for source_index, source in enumerate(adapter.evaluation_source_paths()):
        source = Path(source)
        candidates = sorted(source.rglob("*.py")) if source.is_dir() else [source]
        for path in candidates:
            if not path.exists():
                continue
            relative = path.relative_to(source) if source.is_dir() else Path(path.name)
            identity = f"external/{source_index}/{source.name}/{relative.as_posix()}"
            sources.append((identity, path))
    for identity, path in sorted(set(sources), key=lambda item: item[0]):
        digest.update(identity.encode())
        if path is not None and path.exists():
            digest.update(path.read_bytes())
        else:
            # Synthetic test adapters need no repository file; production adapters always have one.
            digest.update(b"<adapter-source-unavailable>")
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"evaluation artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def _feature_cache_key(
    *, baseline_name: str, dataset: str, stream_id: str, stream_fingerprint: str,
    source_fingerprint: str, artifacts: dict, config: dict,
) -> str:
    payload = {
        "schema": 1,
        "baseline": baseline_name,
        "dataset": dataset,
        "stream": stream_id,
        "stream_fingerprint": stream_fingerprint,
        "source_fingerprint": source_fingerprint,
        "artifacts": artifacts,
        "config": config,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_cached_features(path: Path, expected_rows: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if value.ndim != 2 or value.shape[0] != expected_rows or not np.isfinite(value).all():
        return None
    return np.asarray(value, dtype=np.float32)


def _save_cached_features(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)


def _git_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def _unit_features(features: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    if tensor.ndim != 2 or not torch.isfinite(tensor).all():
        raise ValueError("baseline window features must be a finite (N,D) matrix")
    return F.normalize(tensor, dim=-1)


def _batched_linear_head_predictions(
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    n_candidates: int,
    *,
    device: torch.device,
    seed: int,
) -> list[np.ndarray]:
    """Fit independent episode heads in padded batches to avoid tiny-kernel launch overhead."""
    output: list[np.ndarray] = []
    for chunk_start in range(0, len(episodes), LINEAR_EPISODE_BATCH):
        chunk = episodes[chunk_start:chunk_start + LINEAR_EPISODE_BATCH]
        max_support = max(len(item[0]) for item in chunk)
        feature_dim = chunk[0][0].shape[1]
        x = torch.zeros(len(chunk), max_support, feature_dim, device=device)
        y = torch.zeros(len(chunk), max_support, dtype=torch.long, device=device)
        mask = torch.zeros(len(chunk), max_support, dtype=torch.bool, device=device)
        initial = torch.stack([item[3] for item in chunk])
        for index, (support, target, _, _) in enumerate(chunk):
            x[index, :len(support)] = support
            y[index, :len(target)] = target
            mask[index, :len(support)] = True
        devices = [] if device.type == "cpu" else [device.index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed + chunk_start))
            weight = torch.nn.Parameter(initial.clone())
            bias = torch.nn.Parameter(torch.zeros(len(chunk), n_candidates, device=device))
            optimizer = torch.optim.AdamW(
                (weight, bias), lr=LINEAR_HEAD_LR,
                weight_decay=LINEAR_HEAD_WEIGHT_DECAY,
            )
            for _ in range(LINEAR_HEAD_STEPS):
                optimizer.zero_grad(set_to_none=True)
                logits = torch.einsum("bsd,bcd->bsc", x, weight) + bias[:, None, :]
                losses = F.cross_entropy(
                    logits.flatten(0, 1), y.flatten(), reduction="none"
                ).view(len(chunk), max_support)
                per_episode = (losses * mask).sum(1) / mask.sum(1).clamp_min(1)
                loss = per_episode.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite supervised target-head loss")
                loss.backward()
                optimizer.step()
            for index, (_, _, query, _) in enumerate(chunk):
                output.append(
                    F.linear(query, weight[index], bias[index]).argmax(1).detach().cpu().numpy()
                )
    return output


def _ridge_predictions(
    support: torch.Tensor,
    target_position: torch.Tensor,
    query: torch.Tensor,
    n_candidates: int,
) -> np.ndarray:
    target = F.one_hot(target_position, n_candidates).float()
    if len(support) <= support.shape[1]:
        alpha = torch.linalg.solve(
            support @ support.T
            + torch.eye(len(support), device=support.device, dtype=support.dtype),
            target,
        )
        weight = support.T @ alpha
    else:
        weight = torch.linalg.solve(
            support.T @ support
            + torch.eye(support.shape[1], device=support.device, dtype=support.dtype),
            support.T @ target,
        )
    return (query @ weight).argmax(1).cpu().numpy()


def support_predictions(
    features: np.ndarray,
    support_rows: np.ndarray,
    support_positions: np.ndarray,
    query_rows: np.ndarray,
    n_candidates: int,
    *,
    device: torch.device,
    seed: int,
    methods: Sequence[str] = METHODS,
) -> dict[str, np.ndarray]:
    """Fit matched support rules and return candidate-position predictions."""
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown adaptation methods: {unknown}")
    z = _unit_features(features)
    x = z[torch.as_tensor(support_rows, dtype=torch.long)].to(device)
    q = z[torch.as_tensor(query_rows, dtype=torch.long)].to(device)
    y = torch.as_tensor(support_positions, dtype=torch.long, device=device)
    if len(x) == 0 or any(not bool(y.eq(index).any()) for index in range(n_candidates)):
        raise ValueError("every candidate must have at least one support execution")
    result: dict[str, np.ndarray] = {}

    if "nearest" in methods:
        result["nearest"] = y[(q @ x.T).argmax(1)].cpu().numpy()

    centroids = torch.stack([
        F.normalize(x[y.eq(index)].mean(0), dim=0) for index in range(n_candidates)
    ])
    if "prototype" in methods:
        result["prototype"] = (q @ centroids.T).argmax(1).cpu().numpy()

    if "ridge" in methods:
        result["ridge"] = _ridge_predictions(x, y, q, n_candidates)

    if "linear_head" in methods:
        # A target classifier is the common frozen-representation supervised adaptation control.
        result["linear_head"] = _batched_linear_head_predictions(
            [(x, y, q, centroids)], n_candidates, device=device, seed=seed
        )[0]
    return result


def _score_predictions(
    truth: list[str], predictions: dict[str, list[str]], subject_records: dict
) -> dict:
    result = {
        "queries": len(truth),
        "subjects": len(subject_records),
        "subject_results": subject_records,
    }
    for method, values in predictions.items():
        metrics = classification_metrics(truth, values)
        result[method] = {
            "f1_macro": metrics["f1_macro"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        }
    return result


def _macro_f1_positions(
    truth: np.ndarray, prediction: np.ndarray, n_candidates: int
) -> float:
    """Macro-F1 over the truth/prediction union for integer class positions."""
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    active = np.unique(np.concatenate((truth, prediction)))
    if len(active) == 0:
        return 0.0
    true_count = np.bincount(truth, minlength=n_candidates)
    predicted_count = np.bincount(prediction, minlength=n_candidates)
    true_positive = np.bincount(
        truth[truth == prediction], minlength=n_candidates
    )
    denominator = true_count[active] + predicted_count[active]
    return float(
        np.mean(2.0 * true_positive[active] / np.maximum(denominator, 1)) * 100.0
    )


def score_positive_cell(
    *,
    query_features: np.ndarray | torch.Tensor,
    support_features: np.ndarray | torch.Tensor,
    query_labels: np.ndarray,
    plans: list[dict],
    support_count: int,
    device: torch.device,
    seed: int,
    methods: Sequence[str],
    execution_feature_cache: dict[tuple[int, ...], torch.Tensor] | None = None,
) -> dict:
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown adaptation methods: {unknown}")
    all_truth: list[str] = []
    all_predictions = {method: [] for method in methods}
    subject_records = {}
    query_z = (
        query_features
        if isinstance(query_features, torch.Tensor)
        else _unit_features(query_features).to(device)
    )
    support_z = (
        query_z if support_features is query_features
        else support_features if isinstance(support_features, torch.Tensor)
        else _unit_features(support_features).to(device)
    )
    linear_episodes = []
    linear_metadata = []
    for plan_index, plan in enumerate(plans):
        candidates = list(plan["candidate_names"])
        query_rows = np.asarray(plan["query_rows"], dtype=np.int64)
        # k is an execution budget. Collapse every enrolled execution to one normalized vector so a
        # 56-window recording cannot outweigh a one-window recording in the generic controls.
        execution_features = []
        execution_positions = []
        support_window_counts = []
        for position, executions in enumerate(plan["support_execution_rows"]):
            for execution_rows in executions[:support_count]:
                cache_key = tuple(int(row) for row in execution_rows)
                pooled = (
                    execution_feature_cache.get(cache_key)
                    if execution_feature_cache is not None else None
                )
                if pooled is None:
                    row_index = torch.as_tensor(execution_rows, dtype=torch.long, device=device)
                    pooled = F.normalize(support_z[row_index].mean(0), dim=0)
                    if execution_feature_cache is not None:
                        execution_feature_cache[cache_key] = pooled
                execution_features.append(pooled)
                execution_positions.append(position)
                support_window_counts.append(len(execution_rows))
        x = torch.stack(execution_features)
        q = query_z[torch.as_tensor(query_rows, dtype=torch.long, device=device)]
        y = torch.as_tensor(execution_positions, dtype=torch.long, device=device)
        if any(not bool(y.eq(index).any()) for index in range(len(candidates))):
            raise ValueError("every candidate must have at least one support execution")
        centroids = torch.stack([
            F.normalize(x[y.eq(index)].mean(0), dim=0) for index in range(len(candidates))
        ])
        predicted = {}
        if "nearest" in methods:
            predicted["nearest"] = y[(q @ x.T).argmax(1)].cpu().numpy()
        if "prototype" in methods:
            predicted["prototype"] = (q @ centroids.T).argmax(1).cpu().numpy()
        if "ridge" in methods:
            predicted["ridge"] = _ridge_predictions(x, y, q, len(candidates))
        names = np.asarray(candidates, dtype=object)
        truth = query_labels[query_rows].tolist()
        candidate_position = {name: position for position, name in enumerate(candidates)}
        truth_positions = np.asarray(
            [candidate_position[name] for name in truth], dtype=np.int64
        )
        all_truth.extend(truth)
        record = {
            "queries": len(query_rows),
            "query_executions": len(plan["query_execution_ids"]),
            "support_executions": int(len(candidates) * support_count),
            "support_windows": int(sum(support_window_counts)),
            "support_windows_per_execution_min": int(min(support_window_counts)),
            "support_windows_per_execution_max": int(max(support_window_counts)),
        }
        for method, positions in predicted.items():
            values = names[positions].tolist()
            all_predictions[method].extend(values)
            record[f"{method}_f1_macro"] = _macro_f1_positions(
                truth_positions, positions, len(candidates)
            )
        subject_records[str(plan["subject"])] = record
        if "linear_head" in methods:
            linear_episodes.append((x, y, q, centroids))
            linear_metadata.append((names, truth, truth_positions, record))
    if linear_episodes:
        positions_by_episode = _batched_linear_head_predictions(
            linear_episodes, len(plans[0]["candidate_names"]),
            device=device, seed=seed + support_count,
        )
        for positions, (names, truth, truth_positions, record) in zip(
            positions_by_episode, linear_metadata, strict=True
        ):
            values = names[positions].tolist()
            all_predictions["linear_head"].extend(values)
            record["linear_head_f1_macro"] = _macro_f1_positions(
                truth_positions, positions, len(names)
            )
    return _score_predictions(all_truth, all_predictions, subject_records)


def score_native_enrollment_cell(
    adapter,
    query_stream,
    support_stream,
    query_labels: np.ndarray,
    plans: list[dict],
    support_count: int,
    candidate_texts: Sequence[str],
    state,
    device,
    *,
    seed: int,
) -> dict:
    """Score one model's native memory mechanism on the shared episode plans."""
    all_truth: list[str] = []
    all_predictions: list[str] = []
    subject_records = {}
    info_records = []
    for plan_index, plan in enumerate(plans):
        predictions, info = adapter.predict_enrollment(
            query_stream, support_stream, plan, support_count, candidate_texts,
            state, device, seed=seed + plan_index,
        )
        query_rows = np.asarray(plan["query_rows"], dtype=np.int64)
        if len(predictions) != len(query_rows):
            raise ValueError(
                f"{adapter.name}: native enrollment returned {len(predictions)} predictions for "
                f"{len(query_rows)} query windows"
            )
        truth = query_labels[query_rows].tolist()
        all_truth.extend(truth)
        all_predictions.extend(predictions)
        subject_records[str(plan["subject"])] = {
            "queries": len(query_rows),
            "query_executions": len(plan["query_execution_ids"]),
            "support_executions": int(len(plan["candidate_names"]) * support_count),
            f"{NATIVE_METHOD}_f1_macro": classification_metrics(
                truth, predictions,
            )["f1_macro"],
        }
        info_records.append(info)
    metrics = classification_metrics(all_truth, all_predictions)
    return {
        NATIVE_METHOD: {
            "f1_macro": metrics["f1_macro"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        },
        "native_adapter_info": info_records,
        "native_subject_results": subject_records,
    }


def score_zero_cell(adapter, stream, features, state, device, cell: dict) -> dict:
    predictions, info = adapter.predict_candidates_from_features(
        features, cell["candidate_names"], state, device
    )
    labels = np.asarray(
        align_ground_truth_labels(stream.gt, stream.eval_labels), dtype=object
    )
    all_truth: list[str] = []
    all_predictions: list[str] = []
    subject_records = {}
    for plan in cell["seeds"]["0"]["plans"]:
        rows = np.asarray(plan["query_rows"], dtype=np.int64)
        truth = labels[rows].tolist()
        pred = [predictions[int(row)] for row in rows]
        all_truth.extend(truth)
        all_predictions.extend(pred)
        subject_records[str(plan["subject"])] = {
            "queries": len(rows),
            "query_executions": len(plan["query_execution_ids"]),
            "zero_shot_f1_macro": classification_metrics(truth, pred)["f1_macro"],
        }
    metrics = classification_metrics(all_truth, all_predictions)
    return {
        "queries": len(all_truth),
        "subjects": len(subject_records),
        "zero_shot": {
            "f1_macro": metrics["f1_macro"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        },
        "subject_results": subject_records,
        "adapter_info": info,
    }


def run(
    *,
    baseline_name: str,
    manifest_path: Path,
    device: str,
    out: Path,
    methods: Sequence[str] = DEFAULT_METHODS,
    label_modes: Sequence[str] = ("coherent", "random_alias"),
    loaded_manifest: dict | None = None,
    feature_cache_dir: Path | None = None,
) -> dict:
    manifest = (
        loaded_manifest if loaded_manifest is not None
        else load_manifest(manifest_path, validate_grids=True)
    )
    if baseline_name not in baselines.REGISTRY:
        raise ValueError(f"unknown baseline {baseline_name!r}; available={sorted(baselines.REGISTRY)}")
    adapter = baselines.REGISTRY[baseline_name]
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    started = time.time()
    needs_native_model = (
        adapter.supports_native_zero_shot() or adapter.supports_native_enrollment()
    )
    state = (
        adapter.setup(resolved_device)
        if needs_native_model else adapter.setup_features(resolved_device)
    )
    setup_seconds = time.time() - started
    source_fingerprint = _source_fingerprint(adapter)
    artifact_fingerprint_cache: dict[Path, dict] = {}

    def artifact_fingerprint(path: Path) -> dict:
        resolved = Path(path).resolve()
        if resolved not in artifact_fingerprint_cache:
            artifact_fingerprint_cache[resolved] = _file_fingerprint(resolved)
        return artifact_fingerprint_cache[resolved]

    feature_artifacts = {
        name: artifact_fingerprint(path)
        for name, path in adapter.feature_artifacts(state).items()
    }
    evaluation_artifacts = {
        name: artifact_fingerprint(path)
        for name, path in (
            adapter.evaluation_artifacts(state).items()
            if needs_native_model else adapter.feature_artifacts(state).items()
        )
    }
    feature_config = adapter.feature_config(state)
    adapter_config = (
        adapter.evaluation_config(state) if needs_native_model else feature_config
    )
    streams = {}
    feature_cache: dict[tuple[str, str], np.ndarray] = {}
    unit_feature_cache: dict[tuple[str, str], torch.Tensor] = {}
    pooled_execution_cache: dict[
        tuple[str, str], dict[tuple[int, ...], torch.Tensor]
    ] = {}
    feature_cache_hits = 0
    feature_cache_misses = 0

    def load_stream(dataset: str, stream_id: str):
        key = (dataset, stream_id)
        if key not in streams:
            streams[key] = load_eval_stream(
                dataset, stream_id, alignment=manifest["alignment"]
            )
        return streams[key]

    def features(dataset: str, stream_id: str) -> np.ndarray:
        nonlocal feature_cache_hits, feature_cache_misses
        key = (dataset, stream_id)
        if key not in feature_cache:
            stream = load_stream(dataset, stream_id)
            cache_path = None
            value = None
            if feature_cache_dir is not None:
                stream_key = f"{dataset}/{stream_id}"
                if stream_key not in manifest["stream_fingerprints"]:
                    raise ValueError(f"manifest has no fingerprint for {stream_key}")
                cache_key = _feature_cache_key(
                    baseline_name=baseline_name,
                    dataset=dataset,
                    stream_id=stream_id,
                    stream_fingerprint=manifest["stream_fingerprints"][stream_key],
                    source_fingerprint=source_fingerprint,
                    artifacts=feature_artifacts,
                    config=feature_config,
                )
                safe_stream = stream_id.replace("/", "_")
                cache_path = Path(feature_cache_dir) / baseline_name / (
                    f"{dataset}__{safe_stream}__{cache_key[:16]}.npy"
                )
                value = _load_cached_features(cache_path, stream.n_windows)
            loaded_from_cache = value is not None
            if value is None:
                feature_cache_misses += 1
                value = np.asarray(adapter.window_features(stream, state, resolved_device))
            else:
                feature_cache_hits += 1
            if value.ndim != 2 or value.shape[0] != stream.n_windows:
                raise ValueError(
                    f"{baseline_name}/{dataset}/{stream_id}: feature shape {value.shape} is not "
                    f"({stream.n_windows}, D)"
                )
            if not np.isfinite(value).all():
                raise ValueError(
                    f"{baseline_name}/{dataset}/{stream_id}: non-finite feature values"
                )
            if cache_path is not None and not loaded_from_cache:
                _save_cached_features(cache_path, value)
            feature_cache[key] = value.astype(np.float32, copy=False)
        return feature_cache[key]

    def unit_features(dataset: str, stream_id: str) -> torch.Tensor:
        key = (dataset, stream_id)
        if key not in unit_feature_cache:
            unit_feature_cache[key] = _unit_features(
                features(dataset, stream_id)
            ).to(resolved_device)
        return unit_feature_cache[key]

    results = {}
    for cell_id, cell in iter_cells(manifest):
        dataset = cell["dataset"]
        incompatible = adapter.is_incompatible(dataset)
        if cell["status"] != "ok" or incompatible:
            results[cell_id] = {
                "status": "n/a",
                "reason": incompatible or cell["status"],
                "regime": cell["regime"],
            }
            continue
        if cell["kind"] == "zero_shot":
            if not adapter.supports_native_zero_shot():
                results[f"{cell_id}/coherent/k0"] = {
                    "status": "n/a",
                    "reason": "model has no native open-vocabulary prediction rule",
                    "kind": "zero_shot",
                    "regime": cell["regime"],
                    "support_count": 0,
                    "label_mode": "coherent",
                }
                continue
            query_stream = load_stream(dataset, cell["query_stream"])
            try:
                result = score_zero_cell(
                    adapter, query_stream, features(dataset, cell["query_stream"]),
                    state, resolved_device, cell,
                )
            except UnsupportedEvaluationCell as exc:
                results[f"{cell_id}/coherent/k0"] = {
                    "status": "n/a", "reason": str(exc), "kind": "zero_shot",
                    "regime": cell["regime"], "support_count": 0,
                    "label_mode": "coherent",
                }
                print(f"[{baseline_name}] {cell_id}: unsupported ({exc})", flush=True)
                continue
            result.update({"kind": "zero_shot", "regime": cell["regime"], "support_count": 0})
            results[f"{cell_id}/coherent/k0"] = result
            continue

        query_stream = load_stream(dataset, cell["query_stream"])
        query_labels = np.asarray(
            align_ground_truth_labels(query_stream.gt, query_stream.eval_labels), dtype=object
        )
        query_z = unit_features(dataset, cell["query_stream"])
        support_z = unit_features(dataset, cell["support_stream"])
        for seed_text, seed_payload in cell["seeds"].items():
            seed = int(seed_text)
            for support_count in manifest["support_counts"]:
                if support_count <= 0:
                    continue
                cohort = "main"
                active_payload = seed_payload
                active_ceiling = int(cell["support_ceiling"])
                if support_count > active_ceiling:
                    secondary = cell.get("secondary_high_support", {})
                    if secondary.get("status") != "ok" or support_count > int(
                        secondary.get("support_ceiling", 0)
                    ):
                        continue
                    cohort = "secondary_high_support"
                    active_payload = secondary["seeds"][seed_text]
                plans = active_payload["plans"]
                adaptation_started = time.time()
                scored = score_positive_cell(
                    query_features=query_z,
                    support_features=support_z,
                    query_labels=query_labels,
                    plans=plans,
                    support_count=support_count,
                    device=resolved_device,
                    seed=seed,
                    methods=methods,
                    execution_feature_cache=pooled_execution_cache.setdefault(
                        (dataset, cell["support_stream"]), {}
                    ),
                )
                scored["fit_and_predict_seconds"] = time.time() - adaptation_started
                scored["feature_dim"] = int(query_z.shape[1])
                if "linear_head" in methods:
                    scored["linear_head_trainable_parameters"] = int(
                        len(cell["candidate_names"]) * (query_z.shape[1] + 1)
                    )
                for label_mode in label_modes:
                    mode_scored = scored
                    if adapter.supports_native_enrollment():
                        aliases = active_payload.get("aliases", {})
                        candidate_texts = (
                            list(cell["candidate_names"])
                            if label_mode == "coherent"
                            else [aliases[name] for name in cell["candidate_names"]]
                        )
                        native_started = time.time()
                        native = score_native_enrollment_cell(
                            adapter, query_stream,
                            load_stream(dataset, cell["support_stream"]), query_labels,
                            plans, support_count, candidate_texts, state, resolved_device,
                            seed=seed,
                        )
                        mode_scored = {
                            **scored,
                            NATIVE_METHOD: native[NATIVE_METHOD],
                            "native_fit_and_predict_seconds": time.time() - native_started,
                            "native_adapter_info": native["native_adapter_info"],
                            "subject_results": {
                                subject: {
                                    **record,
                                    **native["native_subject_results"].get(subject, {}),
                                }
                                for subject, record in scored["subject_results"].items()
                            },
                        }
                    # Generic readouts use aliases only as class IDs and are intentionally identical
                    # across label modes. A native semantic mechanism is re-run with the actual text.
                    key = f"{cell_id}/{label_mode}/seed{seed}/k{support_count}"
                    results[key] = {
                        **mode_scored,
                        "kind": "enrollment",
                        "regime": cell["regime"],
                        "label_mode": label_mode,
                        "support_count": support_count,
                        "seed": seed,
                        "cohort": cohort,
                        "candidate_count": len(cell["candidate_names"]),
                    }
        print(f"[{baseline_name}] {cell_id}: complete", flush=True)

    payload = {
        "schema_version": 2,
        "baseline": baseline_name,
        "adapter": f"{type(adapter).__module__}.{type(adapter).__name__}",
        "manifest": str(manifest_path.resolve()),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "source_fingerprint": source_fingerprint,
        "methods": list(methods) + ([NATIVE_METHOD] if adapter.supports_native_enrollment() else []),
        "primary_positive_k_method": (
            NATIVE_METHOD if adapter.supports_native_enrollment() else "nearest"
        ),
        "evaluation_artifacts": evaluation_artifacts,
        "adapter_config": adapter_config,
        "feature_cache": {
            "enabled": feature_cache_dir is not None,
            "directory": str(Path(feature_cache_dir).resolve()) if feature_cache_dir else None,
            "stream_hits": feature_cache_hits,
            "stream_misses": feature_cache_misses,
            "pooled_execution_vectors": int(sum(
                len(values) for values in pooled_execution_cache.values()
            )),
        },
        "support_protocol": {
            "support_unit": manifest.get(
                "support_unit", "independent_execution_per_candidate"
            ),
            "k_definition": "independent_enrolled_executions_per_candidate",
            "generic_readout_representation": (
                "one_normalized_mean_pooled_vector_per_enrolled_execution"
            ),
            "halo_native_representation": (
                "one_normalized_mean_pooled_vector_per_enrolled_execution"
            ),
        },
        "linear_head_recipe": {
            "scope": "linear_head_finetuning_on_frozen_representation",
            "optimizer": "AdamW",
            "steps": LINEAR_HEAD_STEPS,
            "learning_rate": LINEAR_HEAD_LR,
            "weight_decay": LINEAR_HEAD_WEIGHT_DECAY,
            "initialization": "normalized_class_prototypes",
            "selection": "fixed_before_test_no_query_early_stopping",
        } if "linear_head" in methods else None,
        "supervised_adaptation_scope": {
            "primary": "not_in_default_report",
            "model_native_end_to_end": "optional_separate_experiment",
            "reason": "the main enrollment report uses only non-gradient matched readouts",
        },
        "setup_seconds": setup_seconds,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(resolved_device))
            if resolved_device.type == "cuda" else 0
        ),
        **_git_provenance(),
        "results": results,
    }
    _atomic_write(out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--no-feature-cache", action="store_const", const=None,
                        dest="feature_cache_dir")
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--label-modes", nargs="*", choices=("coherent", "random_alias"),
                        default=["coherent", "random_alias"])
    args = parser.parse_args()
    failures = []
    shared_manifest = load_manifest(args.manifest, validate_grids=True)
    manifest_tag = args.manifest.name
    for suffix in (".gz", ".json"):
        if manifest_tag.endswith(suffix):
            manifest_tag = manifest_tag[:-len(suffix)]
    for baseline_name in args.baselines:
        out = args.out_dir / f"{baseline_name}__{manifest_tag}.json"
        try:
            run(
                baseline_name=baseline_name,
                manifest_path=args.manifest,
                device=args.device,
                out=out,
                methods=args.methods,
                label_modes=args.label_modes,
                loaded_manifest=shared_manifest,
                feature_cache_dir=args.feature_cache_dir,
            )
            print(f"-> {out}", flush=True)
        except Exception as error:
            failures.append((baseline_name, repr(error)))
            print(f"[{baseline_name}] FAILED: {error}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"baseline adaptation failures: {failures}")


if __name__ == "__main__":
    main()
