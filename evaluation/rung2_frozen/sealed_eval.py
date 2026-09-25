"""Protocol-controlled sealed evaluation for support-conditioned HAR.

This is intentionally separate from ``train.py``.  It is the only entry point allowed to read the
six sealed sources, creates one immutable support/query manifest per stream and k, and applies the
same readouts to every representation provider.  It does not select checkpoints or tune settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import baselines
from baselines import scoring
from halo.paths import CACHE_DIR
from baselines.data import (
    EvalStream, MultiDeviceEvalStream, list_streams, load_eval_stream, load_global_labels,
    load_multi_device_stream, source_slice_fingerprint,
)
from data.scripts.curate.deployment_policy import (
    MULTI_DEVICE_EVAL_CELLS,
    PROSPECTIVE_EVAL_DATASETS,
    SEALED_TEST_EVAL_DATASETS,
    SUPERVISED_HEAD_TRAIN_DATASETS,
    assert_no_retired_sources,
    stream_specs,
)
from data.datasets.mobiact.protocol import (
    candidate_labels as mobiact_candidate_labels,
    partition_rows as mobiact_partition_rows,
)
from data.scripts.labels.canonical_labels import canonicalize
from model.blocks import AttentionSpec
from model.support.token_mixer import SupportTokenMixer, TokenMixerConfig
from model.support.residual_classifier import ResidualClassifierConfig, build_support_classifier
from model.support.contextual_classifier import ContextualSupportClassifier
from model.support.contextual_residual_classifier import ContextualResidualSupportClassifier
from model.support.evidence_aware_classifier import EvidenceAwareSupportClassifier
from model.support.factory import (
    CONTEXTUAL_RESIDUAL_ARCHITECTURE, EVIDENCE_AWARE_ARCHITECTURE,
    EVIDENCE_GATED_ARCHITECTURE, EVIDENCE_GATED_READOUTS, EVIDENCE_GATED_SEMANTIC_READOUTS,
    LEGACY_CONTEXTUAL_ARCHITECTURE,
    CONTEXTUAL_CHECKPOINT_ARCHITECTURES, CONTEXTUAL_READOUTS, EVIDENCE_AWARE_READOUTS,
    LEGACY_CONTEXTUAL_READOUTS, LEARNED_CLASSIFIER_ARCHITECTURES,
    RESIDUAL_ARCHITECTURES, build_classifier_from_blob,
)
from training.support_classifier import frozen_baseline_adaptation
from training.support_classifier.train import make_label_text
from training.support_classifier.neighbors import differentiable_neighbor_logits
from training.support_classifier.representation_diagnostics import write_embedding_diagnostics
from training.support_classifier.sampling import MIN_RECORDING_SECONDS
from training.support_classifier.partial_coverage import (
    CoverageCell,
    classwise_neighbor_scores,
    equal_weight_normalized_fusion_predictions,
)
from training.tokenizer.eval_transfer import build_encoder, encode_dataset, encode_multi_device_dataset
from training.tokenizer.pretrain_data import (
    STREAM_SOURCE_RATE_HZ, _stream_gravity_state, modalities_present,
    stream_channel_descriptions, stream_sensor_texts, structured_sensor_metadata,
)

# Shared evaluation modules, extracted 2026-09-23 (Phase 0 of the rung 1/2 implementation plan).
# Re-exported so existing imports and test monkeypatches of ``sealed_eval.<name>`` keep working;
# sealed_eval must never be imported by ``evaluation/`` (one-way dependency).
from evaluation.features import (
    FEATURE_CACHE_SCHEMA,
    FeatureMemoryCache,
    UNCHANGED_BASELINE_FEATURE_CACHE_SCHEMA,
    _baseline_feature_state,
    _cache_key,
    _file_hash,
    _file_hash_for_stat,
    _halo_features,
    _load_or_encode,
    feature_cache_schema,
)
from evaluation.manifests import (
    QueryPlan,
    SEED,
    _aligned_labels,
    _stable_choice,
    build_manifest,
    duration_cells,
    evaluation_cells,
    manifest_fingerprint,
    sealed_cells,
)
from evaluation.zero_shot import (
    TRAINING_BANK_ZERO_SHOT,
    _build_training_reference_bank,
    _normalise,
    _training_bank_conse_predictions,
)
from evaluation.provenance import _atomic_json, _run_provenance, validate_result_rows
from evaluation.acquisition import halo_acquisition_rows, halo_acquisition_vector  # noqa: F401

# Full enrollment curve. All six sealed sources have at least 128 execution-disjoint
# support windows per candidate for every currently valid query (verified 2026-09-12).
# ``build_manifest`` still fails closed per query if a future corpus revision no longer
# satisfies a requested support count.
DEFAULT_K = (0, 1, 2, 4, 8, 16, 32, 64, 128)
PRIMARY_BASELINES = ("harnet5", "harnet10", "limubert_x", "unimts", "normwear")
# Full released-model parameter counts in millions, measured from the pinned artifacts used by
# the adapters.  This is model capacity, not the parameter-free common enrollment readout.
_PARAMETER_COUNT_M = {
    "harnet5": 4.491,
    "harnet10": 10.983,
    "limubert_x": 0.055,
    "unimts": 68.61,
    "normwear": 1293.86,
}
_PARAMETER_BREAKDOWN_M = {
    "normwear": {
        "sensor_backbone": 136.1,
        "msitf_aggregator": 57.7,
        "frozen_tinyllama_text_tower": 1100.06,
    },
}
_HALO_CLASSIFIER_PARAMETER_CACHE: dict[str, int] = {}
_HALO_RESIDUAL_HEAD_CACHE: dict[tuple[str, str, bool, bool], torch.nn.Module] = {}


MAX_EXACT_RIDGE_SYSTEM = 512


def _native_capabilities(name: str) -> dict[str, bool]:
    """Disclose capabilities supplied by the model itself, not shared controls.

    The common 1-NN/prototype/ridge readouts deliberately give every representation the same
    post-hoc enrollment protocol.  They must not be mistaken for a released model's native
    open-set or adaptation mechanism.
    """
    if name == "halo":
        return {
            "native_open_set_labels": True,
            "native_support_conditioning": True,
            "published_few_label_finetuning": False,
        }
    adapter = baselines.REGISTRY[name]
    return {
        "native_open_set_labels": bool(adapter.supports_native_zero_shot()),
        "native_support_conditioning": bool(adapter.supports_native_enrollment()),
        "published_few_label_finetuning": name in {"harnet5", "harnet10", "limubert_x", "unimts", "normwear"},
    }


def _parameter_count_m(
    name: str,
    halo_state: tuple[torch.nn.Module, str] | None = None,
    *,
    halo_checkpoint: Path | None = None,
    include_classifier: bool = False,
) -> float:
    """Return full model capacity for every result row.

    HALO is checkpoint-dependent, so count its materialized encoder at runtime; released
    baselines use the audited counts above.  The second tuple member is the checkpoint
    fingerprint used for cache invalidation, not a module.
    """
    if name != "halo":
        return _PARAMETER_COUNT_M[name]
    if halo_state is None:
        raise ValueError("HALO parameter count requires the loaded evaluation state")
    encoder, _ = halo_state
    count = sum(parameter.numel() for parameter in encoder.parameters())
    if include_classifier:
        if halo_checkpoint is None:
            raise ValueError("learned HALO classifier count requires its checkpoint")
        cache_key = str(halo_checkpoint.resolve())
        if cache_key not in _HALO_CLASSIFIER_PARAMETER_CACHE:
            blob = torch.load(halo_checkpoint, map_location="cpu", weights_only=False)
            if blob.get("architecture_version") not in LEARNED_CLASSIFIER_ARCHITECTURES:
                raise ValueError("only a learned support-classifier checkpoint exposes the learned count")
            head, _ = build_classifier_from_blob(blob)
            _HALO_CLASSIFIER_PARAMETER_CACHE[cache_key] = sum(
                parameter.numel() for parameter in head.parameters()
            )
        count += _HALO_CLASSIFIER_PARAMETER_CACHE[cache_key]
    return count / 1_000_000.0


def _readout_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    candidates: Sequence[str],
    plans: Sequence[QueryPlan],
    *,
    ridge_alpha: float = 1.0,
    device: torch.device | None = None,
    readouts: frozenset[str] | None = None,
) -> dict[str, list[str]]:
    """Matched enrollment readouts over one frozen representation matrix."""
    z = _normalise(features)
    requested = frozenset(("1nn", "prototype", "ridge")) if readouts is None else readouts
    unknown = requested - {"1nn", "prototype", "ridge"}
    if unknown:
        raise ValueError(f"unknown support-only readouts: {sorted(unknown)}")
    result = {name: [] for name in requested}
    candidate_index = {label: index for index, label in enumerate(candidates)}
    if device is None:
        for plan in plans:
            query = z[plan.query]
            support = np.asarray(plan.support, dtype=np.int64)
            if not len(support):
                continue
            x = z[support]
            y = np.asarray([candidate_index[label] for label in plan.support_labels], dtype=np.int64)
            if "1nn" in requested:
                result["1nn"].append(str(plan.support_labels[int(np.argmax(x @ query))]))
            if "prototype" in requested:
                prototypes = np.stack([
                    x[y == slot].mean(axis=0) for slot in range(len(candidates))
                ])
                result["prototype"].append(
                    str(candidates[int(np.argmax(_normalise(prototypes) @ query))])
                )
            if "ridge" in requested:
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
    else:
        neighbor_readouts = requested & {"1nn", "prototype"}
        if neighbor_readouts:
            result.update(_neighbor_prototype_predictions_batched(
                z, candidates, plans, device, readouts=neighbor_readouts,
            ))
        support_count = len(plans[0].support) if plans else 0
        # Ridge needs one query-specific solve because every immutable episode has different
        # enrolled rows. Beyond this exact-system order, especially for 2,048-D NormWear features,
        # the requested high-k control takes hours. High-k 1-NN/prototype remain exact and are the
        # primary parameter-free enrollment controls; disclose ridge as N/A rather than approximate it.
        if "ridge" in requested and min(support_count, z.shape[1]) <= MAX_EXACT_RIDGE_SYSTEM:
            result["ridge"] = _ridge_predictions_batched(
                z, candidates, plans, device, ridge_alpha=ridge_alpha,
            )
        elif "ridge" in requested:
            result.pop("ridge", None)
    return result


@torch.no_grad()
def _neighbor_prototype_predictions_batched(
    normalized_features: np.ndarray,
    candidates: Sequence[str],
    plans: Sequence[QueryPlan],
    device: torch.device,
    *,
    readouts: frozenset[str] = frozenset(("1nn", "prototype")),
) -> dict[str, list[str]]:
    """GPU-batch exact 1-NN and prototype scoring for query-specific episodes."""
    if not plans:
        return {name: [] for name in readouts}
    candidate_index = {label: index for index, label in enumerate(candidates)}
    support_count = len(plans[0].support)
    if not support_count or any(len(plan.support) != support_count for plan in plans):
        raise ValueError("batched enrollment plans must have one non-zero support width")
    dim = int(normalized_features.shape[1])
    # Bound the gathered support tensor to roughly 96 MiB. This scales safely from compact
    # released encoders to NormWear's 2,048-dimensional representation and large-k episodes.
    batch_size = max(1, min(256, (24 * 1024 * 1024) // max(1, support_count * dim)))
    nearest: list[str] = []
    prototype: list[str] = []
    for start in range(0, len(plans), batch_size):
        chunk = plans[start:start + batch_size]
        support_rows = np.asarray([plan.support for plan in chunk], dtype=np.int64)
        query_rows = np.asarray([plan.query for plan in chunk], dtype=np.int64)
        bindings = torch.as_tensor(
            [[candidate_index[label] for label in plan.support_labels] for plan in chunk],
            dtype=torch.long, device=device,
        )
        x = torch.as_tensor(normalized_features[support_rows], dtype=torch.float32, device=device)
        query = torch.as_tensor(normalized_features[query_rows], dtype=torch.float32, device=device)

        if "1nn" in readouts:
            similarities = torch.einsum("bsd,bd->bs", x, query)
            selected = similarities.argmax(dim=1)
            selected_labels = bindings.gather(1, selected[:, None]).squeeze(1)
            nearest.extend(candidates[index] for index in selected_labels.cpu().tolist())

        if "prototype" in readouts:
            sums = torch.zeros((len(chunk), len(candidates), dim), dtype=x.dtype, device=device)
            sums.scatter_add_(1, bindings.unsqueeze(-1).expand(-1, -1, dim), x)
            counts = torch.zeros((len(chunk), len(candidates)), dtype=x.dtype, device=device)
            counts.scatter_add_(1, bindings, torch.ones_like(bindings, dtype=x.dtype))
            means = sums / counts.clamp_min(1).unsqueeze(-1)
            means = torch.nn.functional.normalize(means, dim=-1)
            scores = torch.einsum("bcd,bd->bc", means, query)
            prototype.extend(candidates[index] for index in scores.argmax(dim=1).cpu().tolist())
    output = {}
    if "1nn" in readouts:
        output["1nn"] = nearest
    if "prototype" in readouts:
        output["prototype"] = prototype
    return output


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
    per_label = scoring.per_class_f1(truth, list(predictions))
    training_concepts = {canonicalize(label) for label in load_global_labels()}
    seen = {label: canonicalize(label) in training_concepts for label in per_label}
    seen_scores = [score for label, score in per_label.items() if seen[label]]
    unseen_scores = [score for label, score in per_label.items() if not seen[label]]
    metrics.update({
        "per_label_f1": per_label,
        "label_seen_in_training": seen,
        "f1_macro_seen": float(np.mean(seen_scores)) if seen_scores else None,
        "f1_macro_unseen": float(np.mean(unseen_scores)) if unseen_scores else None,
    })
    metrics.update(scoring.subject_bootstrap_ci(
        truth, list(predictions), subjects, metric="f1_macro", B=bootstrap,
    ))
    metrics.update({
        "dataset": stream.dataset,
        "stream": stream.stream,
        "window_seconds": float(stream.window_seconds),
        "n_queries": int(len(indices)),
        "n_candidates": int(len(stream.eval_labels)),
        "quality_screen": stream.quality_screen,
        "quality_excluded": int(stream.n_quality_excluded),
        "source_slice_fingerprint": source_slice_fingerprint(stream),
    })
    return metrics


@torch.no_grad()
def _halo_residual_predictions(
    features: np.ndarray, stream: EvalStream, plans: Sequence[QueryPlan], checkpoint: Path,
    device: torch.device, *, residual_enabled: bool = True, text_term_enabled: bool = True,
    batch_size: int = 64, acquisitions: np.ndarray | None = None,
) -> list[str]:
    """Evaluate v2's unified scorer; shared parameter-free controls remain elsewhere unchanged."""
    fingerprint = _file_hash(checkpoint)
    cache_key = (fingerprint, str(device), bool(residual_enabled), bool(text_term_enabled))
    head = _HALO_RESIDUAL_HEAD_CACHE.get(cache_key)
    if head is None:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if blob.get("architecture_version") in {CONTEXTUAL_RESIDUAL_ARCHITECTURE, EVIDENCE_AWARE_ARCHITECTURE}:
            if not (residual_enabled and text_term_enabled):
                raise ValueError("residual ablation flags are not defined for the contextual head")
            return _halo_contextual_residual_predictions(
                features, stream, plans, checkpoint, device,
                batch_size=batch_size, acquisitions=acquisitions,
            )
        if blob.get("architecture_version") == EVIDENCE_GATED_ARCHITECTURE:
            # v4 is a different parameterisation: its ablations are the named branch readouts
            # (label meaning / trust-weighted vote / untrusted vote), not these two flags.
            if not (residual_enabled and text_term_enabled):
                raise ValueError(
                    "residual ablation flags are not defined for the evidence-gated head; "
                    "use the branch readouts instead"
                )
            return _halo_evidence_gated_predictions(
                features, stream, plans, checkpoint, device, batch_size=batch_size,
            )
        if blob.get("architecture_version") == LEGACY_CONTEXTUAL_ARCHITECTURE:
            if not (residual_enabled and text_term_enabled):
                raise ValueError("residual ablation flags are not defined for the contextual head")
            return _halo_contextual_predictions(
                features, stream, plans, checkpoint, device, batch_size=batch_size,
            )
        if blob.get("architecture_version") not in RESIDUAL_ARCHITECTURES:
            raise ValueError("residual readout requires a residual support-classifier checkpoint")
        classifier_config = dict(blob["classifier_config"])
        if blob.get("architecture_version") == "support_classifier_v2":
            classifier_config.setdefault("normalized_token_composition", False)
        cfg = ResidualClassifierConfig(**classifier_config)
        cfg = replace(cfg, residual_enabled=residual_enabled, text_term_enabled=text_term_enabled)
        head = build_support_classifier(AttentionSpec(**blob["attention_spec"]), cfg).to(device).eval()
        head.load_state_dict(blob["classifier"], strict=True)
        _HALO_RESIDUAL_HEAD_CACHE[cache_key] = head
    candidates = tuple(stream.eval_labels)
    table = make_label_text(candidates, device)
    candidate_text = table.matrix[torch.as_tensor(table.ids(candidates), device=device)].unsqueeze(0)
    label_to_slot = {label: slot for slot, label in enumerate(candidates)}
    output: list[str] = []
    for start in range(0, len(plans), batch_size):
        chunk = plans[start:start + batch_size]
        b, c = len(chunk), len(candidates)
        width = max((len(plan.support) for plan in chunk), default=0)
        rows = np.zeros((b, width), dtype=np.int64)
        bound = torch.full((b, width), -1, dtype=torch.long, device=device)
        support_mask = torch.zeros((b, width), dtype=torch.bool, device=device)
        for row, plan in enumerate(chunk):
            if plan.support:
                rows[row, :len(plan.support)] = plan.support
                bound[row, :len(plan.support)] = torch.tensor(
                    [label_to_slot[label] for label in plan.support_labels], device=device,
                )
                support_mask[row, :len(plan.support)] = True
        support_feature = torch.as_tensor(features[rows], dtype=torch.float32, device=device)
        support_feature = support_feature * support_mask.unsqueeze(-1)
        safe_bound = bound.clamp_min(0)
        support_text = candidate_text.expand(b, -1, -1).gather(
            1, safe_bound.unsqueeze(-1).expand(-1, -1, candidate_text.shape[-1]),
        ) * support_mask.unsqueeze(-1)
        result = head(
            query_feature=torch.as_tensor(features[[plan.query for plan in chunk]], dtype=torch.float32, device=device),
            support_feature=support_feature, support_label_text=support_text, support_bound=bound,
            support_mask=support_mask,
            support_pair_slot=torch.arange(1, width + 1, device=device).unsqueeze(0).expand(b, -1),
            candidate_text=candidate_text.expand(b, -1, -1),
            candidate_mask=torch.ones((b, c), dtype=torch.bool, device=device),
            candidate_slot=torch.arange(1, c + 1, device=device).unsqueeze(0).expand(b, -1),
        )
        output.extend(candidates[index] for index in result["logits"].argmax(dim=1).cpu().tolist())
    return output


_HALO_CONTEXTUAL_HEAD_CACHE: dict[tuple, ContextualSupportClassifier] = {}
_HALO_CONTEXTUAL_RESIDUAL_HEAD_CACHE: dict[tuple, ContextualResidualSupportClassifier] = {}
_ACQUISITION_CONDITIONED_ARCHITECTURES = CONTEXTUAL_CHECKPOINT_ARCHITECTURES


@torch.no_grad()
def _halo_contextual_predictions(
    features: np.ndarray, stream: EvalStream, plans: Sequence[QueryPlan], checkpoint: Path,
    device: torch.device, *, branch: str = "mixture", batch_size: int = 64,
) -> list[str]:
    """Score plans with the contextual mixture head.

    Support-label text is gathered from a table built over the union of the candidate roster and
    every actual support label in the manifest, so off-roster support labels are legal and no
    candidate binding is ever used. At k=0 the head receives only the query and the candidates.
    """
    fingerprint = _file_hash(checkpoint)
    cache_key = (fingerprint, str(device))
    head = _HALO_CONTEXTUAL_HEAD_CACHE.get(cache_key)
    if head is None:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if blob.get("architecture_version") != LEGACY_CONTEXTUAL_ARCHITECTURE:
            raise ValueError("contextual readout requires a contextual support-classifier checkpoint")
        head, _ = build_classifier_from_blob(blob, device=device)
        _HALO_CONTEXTUAL_HEAD_CACHE[cache_key] = head
    candidates = tuple(stream.eval_labels)
    extra = sorted({label for plan in plans for label in plan.support_labels} - set(candidates))
    table = make_label_text(tuple(candidates) + tuple(extra), device)
    candidate_text = table.matrix[torch.as_tensor(table.ids(candidates), device=device)].unsqueeze(0)
    text_dim = candidate_text.shape[-1]
    output: list[str] = []
    for start in range(0, len(plans), batch_size):
        chunk = plans[start:start + batch_size]
        b, c = len(chunk), len(candidates)
        width = max((len(plan.support) for plan in chunk), default=0)
        rows = np.zeros((b, width), dtype=np.int64)
        label_ids = np.zeros((b, width), dtype=np.int64)
        support_mask = torch.zeros((b, width), dtype=torch.bool, device=device)
        for row, plan in enumerate(chunk):
            if plan.support:
                rows[row, :len(plan.support)] = plan.support
                label_ids[row, :len(plan.support)] = table.ids(list(plan.support_labels))
                support_mask[row, :len(plan.support)] = True
        # Fancy indexing already yields (b, width, feature_dim); an explicit reshape with -1 is
        # ambiguous when width is 0, which is exactly the zero-enrollment scenario path.
        support_feature = torch.as_tensor(
            np.asarray(features)[rows], dtype=torch.float32, device=device,
        ) * support_mask.unsqueeze(-1)
        support_text = table.matrix[torch.as_tensor(label_ids, device=device)] \
            * support_mask.unsqueeze(-1)
        if support_feature.shape != (b, width, features.shape[-1]):
            raise RuntimeError("contextual support gather produced an unexpected shape")
        if support_text.shape != (b, width, text_dim):
            raise RuntimeError("contextual support-label gather produced an unexpected shape")
        result = head(
            query_feature=torch.as_tensor(features[[plan.query for plan in chunk]], dtype=torch.float32, device=device),
            support_feature=support_feature, support_label_text=support_text,
            support_mask=support_mask,
            support_pair_slot=torch.arange(1, width + 1, device=device).unsqueeze(0).expand(b, -1),
            candidate_text=candidate_text.expand(b, -1, -1),
            candidate_mask=torch.ones((b, c), dtype=torch.bool, device=device),
        )
        scores = ContextualSupportClassifier.branch_logits(result, branch)
        output.extend(candidates[index] for index in scores.argmax(dim=1).cpu().tolist())
    return output


@torch.no_grad()
def _halo_contextual_residual_predictions(
    features: np.ndarray, stream: EvalStream, plans: Sequence[QueryPlan], checkpoint: Path,
    device: torch.device, *, branch: str = "final", batch_size: int = 64,
    acquisitions: np.ndarray | None,
) -> list[str]:
    """Score immutable plans with the current contextual residual classifier."""
    if acquisitions is None:
        raise ValueError(
            "contextual residual evaluation requires acquisition rows aligned with features"
        )
    features = np.asarray(features)
    acquisitions = np.asarray(acquisitions)
    if acquisitions.ndim != 2 or len(acquisitions) != len(features):
        raise ValueError("acquisition rows must align 1:1 with feature rows")
    fingerprint = _file_hash(checkpoint)
    cache_key = (fingerprint, str(device))
    head = _HALO_CONTEXTUAL_RESIDUAL_HEAD_CACHE.get(cache_key)
    if head is None:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if blob.get("architecture_version") not in {CONTEXTUAL_RESIDUAL_ARCHITECTURE, EVIDENCE_AWARE_ARCHITECTURE}:
            raise ValueError("contextual residual readout requires its matching checkpoint")
        loaded, _ = build_classifier_from_blob(blob, device=device)
        if not isinstance(loaded, (ContextualResidualSupportClassifier, EvidenceAwareSupportClassifier)):
            raise TypeError("classifier factory returned the wrong contextual architecture")
        head = loaded
        _HALO_CONTEXTUAL_RESIDUAL_HEAD_CACHE[cache_key] = head

    candidates = tuple(stream.eval_labels)
    extra = sorted({label for plan in plans for label in plan.support_labels} - set(candidates))
    table = make_label_text(tuple(candidates) + tuple(extra), device)
    candidate_ids = torch.as_tensor(table.ids(candidates), device=device)
    candidate_text = table.matrix[candidate_ids].unsqueeze(0)
    candidate_to_slot = {label: slot for slot, label in enumerate(candidates)}
    output: list[str] = []
    for start in range(0, len(plans), batch_size):
        chunk = plans[start:start + batch_size]
        b, c = len(chunk), len(candidates)
        width = max((len(plan.support) for plan in chunk), default=0)
        rows = np.zeros((b, width), dtype=np.int64)
        label_ids = np.zeros((b, width), dtype=np.int64)
        bound = torch.full((b, width), -1, dtype=torch.long, device=device)
        support_mask = torch.zeros((b, width), dtype=torch.bool, device=device)
        for row, plan in enumerate(chunk):
            if not plan.support:
                continue
            rows[row, :len(plan.support)] = plan.support
            label_ids[row, :len(plan.support)] = table.ids(list(plan.support_labels))
            bound[row, :len(plan.support)] = torch.tensor(
                [candidate_to_slot.get(label, -1) for label in plan.support_labels],
                dtype=torch.long, device=device,
            )
            support_mask[row, :len(plan.support)] = True
        support_feature = torch.as_tensor(features[rows], dtype=torch.float32, device=device)
        support_feature = support_feature * support_mask.unsqueeze(-1)
        support_acquisition = torch.as_tensor(
            acquisitions[rows], dtype=torch.float32, device=device,
        ) * support_mask.unsqueeze(-1)
        support_text = table.matrix[torch.as_tensor(label_ids, device=device)] \
            * support_mask.unsqueeze(-1)
        query_rows = np.asarray([plan.query for plan in chunk], dtype=np.int64)
        result = head(
            query_feature=torch.as_tensor(features[query_rows], dtype=torch.float32, device=device),
            support_feature=support_feature,
            query_acquisition=torch.as_tensor(
                acquisitions[query_rows], dtype=torch.float32, device=device,
            ),
            support_acquisition=support_acquisition,
            support_label_text=support_text, support_bound=bound,
            support_mask=support_mask,
            support_pair_slot=torch.arange(1, width + 1, device=device).unsqueeze(0).expand(b, -1),
            candidate_text=candidate_text.expand(b, -1, -1),
            candidate_mask=torch.ones((b, c), dtype=torch.bool, device=device),
            candidate_slot=torch.arange(1, c + 1, device=device).unsqueeze(0).expand(b, -1),
        )
        logits = head.branch_logits(result, branch)
        output.extend(candidates[index] for index in logits.argmax(dim=1).cpu().tolist())
    return output


@torch.no_grad()
def _halo_residual_diagnostic_predictions(
    features: np.ndarray, stream: EvalStream, plans: Sequence[QueryPlan], checkpoint: Path,
    device: torch.device, *, batch_size: int = 64,
) -> dict[str, list[str]]:
    """Decompose one residual-head forward and perturb only support-label text bindings."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if blob.get("architecture_version") not in {"support_classifier_v2", "support_classifier_v3"}:
        raise ValueError("classifier diagnostics require a residual support-classifier checkpoint")
    classifier_config = dict(blob["classifier_config"])
    if blob.get("architecture_version") == "support_classifier_v2":
        classifier_config.setdefault("normalized_token_composition", False)
    head = build_support_classifier(
        AttentionSpec(**blob["attention_spec"]), ResidualClassifierConfig(**classifier_config),
    ).to(device).eval()
    head.load_state_dict(blob["classifier"], strict=True)
    candidates = tuple(stream.eval_labels)
    table = make_label_text(candidates, device)
    candidate_text = table.matrix[torch.as_tensor(table.ids(candidates), device=device)].unsqueeze(0)
    label_to_slot = {label: slot for slot, label in enumerate(candidates)}
    names = (
        "halo-classifier", "halo-classifier-floor", "halo-classifier-text-only",
        "halo-classifier-support-residual-only", "halo-classifier-candidate-residual-only",
        "halo-classifier-residual-only", "halo-classifier-support-label-shuffled",
    )
    output = {name: [] for name in names}
    for start in range(0, len(plans), batch_size):
        chunk = plans[start:start + batch_size]
        b, c = len(chunk), len(candidates)
        width = max((len(plan.support) for plan in chunk), default=0)
        rows = np.zeros((b, width), dtype=np.int64)
        bound = torch.full((b, width), -1, dtype=torch.long, device=device)
        support_mask = torch.zeros((b, width), dtype=torch.bool, device=device)
        for row, plan in enumerate(chunk):
            if plan.support:
                rows[row, :len(plan.support)] = plan.support
                bound[row, :len(plan.support)] = torch.as_tensor(
                    [label_to_slot[label] for label in plan.support_labels], device=device,
                )
                support_mask[row, :len(plan.support)] = True
        support_feature = torch.as_tensor(features[rows], dtype=torch.float32, device=device)
        support_feature = support_feature * support_mask.unsqueeze(-1)
        safe_bound = bound.clamp_min(0)
        expanded_text = candidate_text.expand(b, -1, -1)
        support_text = expanded_text.gather(
            1, safe_bound.unsqueeze(-1).expand(-1, -1, candidate_text.shape[-1]),
        ) * support_mask.unsqueeze(-1)
        common = {
            "query_feature": torch.as_tensor(
                features[[plan.query for plan in chunk]], dtype=torch.float32, device=device,
            ),
            "support_feature": support_feature,
            "support_bound": bound,
            "support_mask": support_mask,
            "support_pair_slot": torch.arange(1, width + 1, device=device).unsqueeze(0).expand(b, -1),
            "candidate_text": expanded_text,
            "candidate_mask": torch.ones((b, c), dtype=torch.bool, device=device),
            "candidate_slot": torch.arange(1, c + 1, device=device).unsqueeze(0).expand(b, -1),
        }
        result = head(support_label_text=support_text, **common)
        component_logits = {
            "halo-classifier": result["logits"],
            "halo-classifier-floor": result["base_part"],
            "halo-classifier-text-only": result["base_part"] + result["text_part"],
            "halo-classifier-support-residual-only": result["metric_part"],
            "halo-classifier-candidate-residual-only": result["base_part"] + result["r_candidate"],
            "halo-classifier-residual-only": result["metric_part"] + result["r_candidate"],
        }
        for name, logits in component_logits.items():
            output[name].extend(candidates[index] for index in logits.argmax(dim=1).cpu().tolist())

        if width:
            shuffled_bound = torch.where(support_mask, (safe_bound + 1) % c, safe_bound)
            shuffled_text = expanded_text.gather(
                1, shuffled_bound.unsqueeze(-1).expand(-1, -1, candidate_text.shape[-1]),
            ) * support_mask.unsqueeze(-1)
        else:
            shuffled_text = support_text
        shuffled = head(support_label_text=shuffled_text, **common)["logits"]
        output["halo-classifier-support-label-shuffled"].extend(
            candidates[index] for index in shuffled.argmax(dim=1).cpu().tolist()
        )
    return output


_HALO_EVIDENCE_GATED_HEAD_CACHE: dict[tuple, object] = {}


def _evidence_gated_semantic_readouts(checkpoint: Path) -> tuple[str, ...]:
    """The semantic-half readouts, only for a checkpoint whose blend actually has two halves."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if blob.get("architecture_version") != EVIDENCE_GATED_ARCHITECTURE:
        return ()
    mode = (blob.get("classifier_config") or {}).get("semantic_mode", "text")
    return EVIDENCE_GATED_SEMANTIC_READOUTS if mode == "text+primitives" else ()


def _evidence_gated_readout_spec(readout: str) -> tuple[str, float | None, float | None]:
    """Map a diagnostic name to its branch and optional in-forward ablation override."""
    specs = {
        "halo-classifier-label-meaning-only": ("semantic", None, None),
        "halo-classifier-support-vote": ("support", None, None),
        "halo-classifier-untrusted-support-vote": ("support_floor", None, None),
        # These retain the complete output path and differ in exactly one source of evidence.
        "halo-classifier-text-off-blend": ("final", 0.0, None),
        "halo-classifier-trust-off-blend": ("final", None, 0.0),
        # Halves of the semantic branch; defined only when the checkpoint has a primitive path.
        "halo-classifier-semantic-text-only": ("semantic_text", None, None),
        "halo-classifier-semantic-primitives-only": ("semantic_primitives", None, None),
    }
    try:
        return specs[readout]
    except KeyError as exc:
        raise ValueError(f"unknown evidence-gated readout {readout!r}") from exc


@torch.no_grad()
def _halo_evidence_gated_predictions(
    features: np.ndarray, stream: EvalStream, plans: Sequence[QueryPlan], checkpoint: Path,
    device: torch.device, *, branch: str = "final", lambda_override: float | None = None,
    trust_override: float | None = None, batch_size: int = 64,
) -> list[str]:
    """Score ``support_classifier_v4`` on an immutable manifest.

    The head takes exactly the v3 call signature (it consumes no acquisition vector), so this
    mirrors the residual path. ``branch`` selects an auditable decomposition of the same forward.
    The optional overrides retain the complete blend while ablating only text or trust.
    """
    fingerprint = _file_hash(checkpoint)
    cache_key = (fingerprint, str(device))
    head = _HALO_EVIDENCE_GATED_HEAD_CACHE.get(cache_key)
    if head is None:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if blob.get("architecture_version") != EVIDENCE_GATED_ARCHITECTURE:
            raise ValueError("checkpoint is not an evidence-gated classifier")
        head, _ = build_classifier_from_blob(blob, device=device)
        _HALO_EVIDENCE_GATED_HEAD_CACHE[cache_key] = head
    candidates = tuple(stream.eval_labels)
    table = make_label_text(candidates, device)
    candidate_text = table.matrix[torch.as_tensor(table.ids(candidates), device=device)].unsqueeze(0)
    label_to_slot = {label: slot for slot, label in enumerate(candidates)}
    predictions: list[str] = []
    for start in range(0, len(plans), batch_size):
        chunk = plans[start:start + batch_size]
        b, c = len(chunk), len(candidates)
        width = max((len(plan.support) for plan in chunk), default=0)
        rows = np.zeros((b, width), dtype=np.int64)
        bound = torch.full((b, width), -1, dtype=torch.long, device=device)
        support_mask = torch.zeros((b, width), dtype=torch.bool, device=device)
        for row, plan in enumerate(chunk):
            if plan.support:
                rows[row, :len(plan.support)] = plan.support
                bound[row, :len(plan.support)] = torch.as_tensor(
                    [label_to_slot[label] for label in plan.support_labels], device=device,
                )
                support_mask[row, :len(plan.support)] = True
        support_feature = torch.as_tensor(features[rows], dtype=torch.float32, device=device)
        support_feature = support_feature * support_mask.unsqueeze(-1)
        expanded_text = candidate_text.expand(b, -1, -1)
        support_text = expanded_text.gather(
            1, bound.clamp_min(0).unsqueeze(-1).expand(-1, -1, candidate_text.shape[-1]),
        ) * support_mask.unsqueeze(-1)
        result = head(
            query_feature=torch.as_tensor(
                features[[plan.query for plan in chunk]], dtype=torch.float32, device=device,
            ),
            support_feature=support_feature, support_label_text=support_text,
            support_bound=bound, support_mask=support_mask,
            support_pair_slot=torch.arange(1, width + 1, device=device).unsqueeze(0).expand(b, -1),
            candidate_text=expanded_text,
            candidate_mask=torch.ones((b, c), dtype=torch.bool, device=device),
            candidate_slot=torch.arange(1, c + 1, device=device).unsqueeze(0).expand(b, -1),
            lambda_override=lambda_override,
            trust_override=trust_override,
        )
        logits = head.branch_logits(result, branch)
        predictions.extend(candidates[index] for index in logits.argmax(dim=1).cpu().tolist())
    return predictions


@torch.no_grad()
def _halo_token_mixer_predictions(
    features: np.ndarray,
    stream: EvalStream,
    plans: Sequence[QueryPlan],
    checkpoint: Path,
    device: torch.device,
    batch_size: int = 64,
    acquisitions: np.ndarray | None = None,
) -> list[str]:
    """Run HALO's semantic token mixer on an immutable manifest.

    The zero-support path receives only the query and the declared candidate labels.  The enrolled
    path receives every support recording plus its paired label token and the candidate tokens;
    it never has an implicit retrieval bank outside the manifest.
    """
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if blob.get("architecture_version") in RESIDUAL_ARCHITECTURES:
        return _halo_residual_predictions(features, stream, plans, checkpoint, device)
    if blob.get("architecture_version") in {CONTEXTUAL_RESIDUAL_ARCHITECTURE, EVIDENCE_AWARE_ARCHITECTURE}:
        return _halo_contextual_residual_predictions(
            features, stream, plans, checkpoint, device, acquisitions=acquisitions,
        )
    if blob.get("architecture_version") == EVIDENCE_GATED_ARCHITECTURE:
        return _halo_evidence_gated_predictions(features, stream, plans, checkpoint, device)
    if blob.get("architecture_version") == LEGACY_CONTEXTUAL_ARCHITECTURE:
        return _halo_contextual_predictions(features, stream, plans, checkpoint, device)
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


def _write_markdown(rows: Sequence[dict], path: Path) -> None:
    columns = ("model", "readout", "window_seconds", "k", "dataset", "stream", "n_devices",
               "multi_device_mode", "padded", "padded_fraction", "f1_macro", "balanced_accuracy",
               "accuracy", "f1_macro_ci_lo", "f1_macro_ci_hi", "n_queries", "n_candidates",
               "parameters_m", "parameter_breakdown_m", "native_open_set_labels", "native_support_conditioning",
               "published_few_label_finetuning", "status")
    lines = ["# Sealed support-conditioned HAR results", "",
             "Generated by `evaluation.rung2_frozen.sealed_eval`; no values select a checkpoint.", "",
             "| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    # Keep diagnostic controls in results.json for auditability, but do not let them look like
    # competing baseline classifiers in the publication-facing table.
    for row in rows:
        if row.get("diagnostic_only"):
            continue
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("sealed", "prospective"), default="sealed",
        help=("sealed keeps the historical six-dataset protocol; prospective runs only the "
              "separately frozen MobiAct expansion"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_BASELINES))
    parser.add_argument("--halo-checkpoint", type=Path, default=None)
    parser.add_argument("--allow-retired-jepa-checkpoint", action="store_true",
                        help="allow a future-JEPA checkpoint only to reproduce historical results")
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument("--window-seconds", type=float, nargs="+", default=[4.0, 8.0, 16.0],
                        help="physical evidence budgets to score; each requires its own grid")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap", type=int, default=scoring.BOOTSTRAP_B)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="optional shared sealed-stream feature cache; defaults to OUT/feature_cache",
    )
    parser.add_argument(
        "--feature-memory-cache-gib", type=float, default=2.0,
        help="bounded in-process LRU for decoded feature matrices (default: 2 GiB)",
    )
    parser.add_argument(
        "--zero-shot-bank-cache",
        type=Path,
        default=CACHE_DIR / "evaluations" / "zero_shot_feature_cache",
        help="shared cache for labelled training-corpus representations used only at k=0",
    )
    parser.add_argument("--embedding-diagnostics", action="store_true",
                        help="write opt-in representation figures beside each encoded stream; this never "
                             "changes predictions, manifests, or checkpoint selection")
    parser.add_argument(
        "--baseline-diagnostic-readouts", action="store_true",
        help=("also compute prototype and ridge controls for external encoders; cosine 1-NN is "
              "always emitted and the primary baseline readout remains equal-weight normalized fusion"),
    )
    parser.add_argument(
        "--baseline-projection", action="append", default=[], metavar="NAME=PATH",
        help="frozen-trunk adaptation: apply a projection fitted by "
             "training.support_classifier.frozen_baseline_adaptation to that baseline's frozen "
             "features before every readout. The released encoder is never modified. Report the "
             "resulting rows as a project method, not as a native baseline result.",
    )
    parser.add_argument(
        "--classifier-isolation", action="store_true",
        help="add HALO-only component and support-label-binding diagnostics",
    )
    args = parser.parse_args()
    if not args.k or min(args.k) < 0:
        parser.error("--k must contain non-negative support counts")
    if (not args.window_seconds or any(value <= 0 or not np.isfinite(value)
                                       for value in args.window_seconds)):
        parser.error("--window-seconds must contain finite positive durations")
    if not np.isfinite(args.feature_memory_cache_gib) or args.feature_memory_cache_gib < 0:
        parser.error("--feature-memory-cache-gib must be finite and non-negative")
    unknown = sorted(set(args.models) - set(PRIMARY_BASELINES) - {"halo"})
    if unknown:
        parser.error(f"models outside the registered primary roster: {unknown}")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not repeat a provider")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    # Frozen-trunk adaptations, loaded once. Empty unless --baseline-projection was passed, so
    # the default protocol is byte-identical to a run without this feature.
    baseline_projections = frozen_baseline_adaptation.parse_projection_arguments(
        args.baseline_projection, device=device)
    for _name, (_module, _blob, _source) in baseline_projections.items():
        print(f"[sealed-eval] {_name}: frozen trunk + projection from {_source} "
              f"({_blob['trainable_parameters']:,} trainable, {_blob['steps']} steps, "
              f"seed {_blob['seed']})", flush=True)
    halo_state: tuple[torch.nn.Module, str] | None = None
    halo_has_classifier = False
    halo_architecture = None
    if "halo" in args.models:
        if args.halo_checkpoint is None:
            parser.error("--halo-checkpoint is required when model list includes halo")
        halo_blob = torch.load(args.halo_checkpoint, map_location="cpu", weights_only=False)
        halo_has_classifier = halo_blob.get("classifier") is not None
        halo_architecture = halo_blob.get("architecture_version")
        if "jepa_mode" in halo_blob.get("config", {}) \
                and not args.allow_retired_jepa_checkpoint:
            parser.error(
                "future-JEPA checkpoints are retired from the active HALO recipe; pass "
                "--allow-retired-jepa-checkpoint only for historical reproduction"
            )
        halo_state = (
            build_encoder(halo_blob, device).eval(),
            _file_hash(args.halo_checkpoint),
        )
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache or args.out / "feature_cache"
    # A very large native model (notably NormWear) must be instantiated once and reused across
    # sealed streams. The same state also serves training-bank encoding so ordinary released
    # encoders are not reconstructed once per source stream.
    persistent_states: dict[str, dict] = {
        name: baselines.REGISTRY[name].setup_features(device)
        for name in args.models if name != "halo"
    }
    feature_memory_cache = FeatureMemoryCache(
        int(args.feature_memory_cache_gib * 1024**3),
    )
    # Banks are duration-specific because every provider must see the same physical evidence budget.
    zero_shot_banks: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, list[str], str]] = {}
    all_rows: list[dict] = []
    manifests: dict[str, dict] = {}
    requested_cells = list(evaluation_cells(args.window_seconds, scope=args.scope))
    started = time.perf_counter()
    total_cells = len(requested_cells)

    def checkpoint_progress(*, completed_cells: int, current: tuple[float, str, str] | None) -> None:
        elapsed = time.perf_counter() - started
        fraction = completed_cells / total_cells if total_cells else 1.0
        eta = None if completed_cells == 0 else elapsed * (1.0 - fraction) / fraction
        _atomic_json(args.out / "progress.json", {
            "complete": completed_cells == total_cells,
            "completed_cells": completed_cells,
            "total_cells": total_cells,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "current": (None if current is None else {
                "window_seconds": float(current[0]), "dataset": current[1], "stream": current[2],
            }),
        })
        eta_text = "unknown" if eta is None else f"{eta / 60.0:.1f}m"
        print(f"[sealed-eval] cells={completed_cells}/{total_cells} "
              f"elapsed={elapsed / 60.0:.1f}m eta={eta_text}", flush=True)

    checkpoint_progress(completed_cells=0, current=None)
    for cell_index, (window_seconds, dataset, stream_id, device_ids) in enumerate(requested_cells, start=1):
        checkpoint_progress(completed_cells=cell_index - 1,
                            current=(window_seconds, dataset, stream_id))
        cell_row_start = len(all_rows)
        stream = (
            load_multi_device_stream(
                dataset, device_ids, alignment="native", window_seconds=window_seconds,
                apply_quality_screen=True,
            )
            if device_ids else
            load_eval_stream(
                dataset, stream_id, alignment="native", window_seconds=window_seconds,
                apply_quality_screen=True,
                candidate_labels=(
                    mobiact_candidate_labels(window_seconds)
                    if args.scope == "prospective" and dataset == "mobiact" else None
                ),
            )
        )
        if stream.quality_screen != "applied":
            raise RuntimeError(f"{dataset}/{stream_id}: quality screen unavailable ({stream.quality_screen})")
        halo_acquisitions = (
            halo_acquisition_rows(stream, halo_state[0], device)
            if halo_state is not None and halo_architecture in _ACQUISITION_CONDITIONED_ARCHITECTURES else None
        )
        # Freeze every episode before any provider is loaded or invoked. All models consume these
        # same query/support row ids; representation extraction cannot influence episode creation.
        plans_by_k: dict[int, list[QueryPlan]] = {}
        for k in sorted(set(args.k)):
            if args.scope == "prospective" and dataset == "mobiact":
                plans = build_manifest(
                    stream, k, seed=args.seed,
                    query_rows=mobiact_partition_rows(stream, "query"),
                    support_rows=mobiact_partition_rows(stream, "reference"),
                )
            else:
                plans = build_manifest(stream, k, seed=args.seed)
            plans_by_k[k] = plans
            manifest_id = f"{dataset}/{stream_id}/w={window_seconds:g}/k={k}"
            manifests[manifest_id] = {
                "fingerprint": manifest_fingerprint(plans), "n_queries": len(plans), "k": k,
                "candidates": stream.eval_labels,
                "device_ids": list(getattr(stream, "device_ids", (stream.stream,))),
                "window_seconds": float(window_seconds),
                "source_slice_fingerprint": source_slice_fingerprint(stream),
                "seed": args.seed,
                "construction": (
                    "prospective_subject_disjoint_numpy_choice_json_v1"
                    if args.scope == "prospective" else "execution_disjoint_numpy_choice_json_v2"
                ),
            }
        # A representation depends only on the provider and the stream, never on enrollment k.
        # Encode/cache it once, then run all protocol readouts on immutable manifests.
        features_by_model: dict[str, tuple[np.ndarray, str]] = {}
        feature_errors: dict[str, str] = {}
        semantic_by_model: dict[str, tuple[np.ndarray, dict]] = {}
        # Enrollment readouts and the training-bank bridge consume recording representations.
        # Native text-aligned baselines can bypass them at k=0.
        if args.models:
            for name in args.models:
                try:
                    features_by_model[name] = _load_or_encode(
                        name=name, stream=stream, device=device, cache_dir=cache_dir,
                        halo_checkpoint=args.halo_checkpoint,
                        baseline_state=persistent_states.get(name),
                        halo_state=halo_state, memory_cache=feature_memory_cache,
                    )
                    if name in baseline_projections:
                        # A frozen-trunk arm: the released encoder is unchanged and its cached
                        # features are reused, then mapped through a projection fitted on our
                        # training corpus. Applied after the cache so the frozen features stay
                        # shared with the unadapted run, and folded into the fingerprint so a
                        # projected row can never be mistaken for a native baseline row.
                        module, blob, source = baseline_projections[name]
                        raw, fingerprint = features_by_model[name]
                        features_by_model[name] = (
                            frozen_baseline_adaptation.apply_projection(
                                module, blob, raw,
                                frozen_baseline_adaptation.stream_channel_mask(stream),
                                device=device),
                            f"{fingerprint}+proj:{_file_hash(Path(source))[:16]}",
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
            plans = plans_by_k[k]
            manifest_id = f"{dataset}/{stream_id}/w={window_seconds:g}/k={k}"
            if not plans:
                for name in args.models:
                    all_rows.append({"model": name, "readout": "all", "k": k,
                                     "dataset": dataset, "stream": stream_id,
                                     "status": "n/a", "reason": "no honest execution-disjoint episode"})
                continue
            for name in args.models:
                if k == 0:
                    # A current token-mixer checkpoint has an explicit zero-support head.
                    # Exercise it before considering the historical training-bank bridge; it
                    # works for single and native multi-device features alike.
                    if name == "halo" and name not in feature_errors:
                        features, fingerprint = features_by_model[name]
                        halo_architecture = torch.load(
                            args.halo_checkpoint, map_location="cpu", weights_only=False,
                        ).get("architecture_version")
                        is_v2 = halo_architecture in LEARNED_CLASSIFIER_ARCHITECTURES
                        is_residual = halo_architecture in RESIDUAL_ARCHITECTURES
                        try:
                            predicted = _halo_token_mixer_predictions(
                                features, stream, plans, args.halo_checkpoint, device,
                                acquisitions=halo_acquisitions,
                            )
                        except ValueError:
                            # The differentiable-neighbours control intentionally has no mixer.
                            pass
                        else:
                            metric = _metric_row(stream, plans, predicted, bootstrap=args.bootstrap)
                            metric.update({"model": name, "readout": (
                                "halo-classifier" if is_v2 else "retrieve-mix-vote-zero-shot"
                            ),
                                           "k": k, "status": "ok",
                                           "feature_fingerprint": fingerprint,
                                           "manifest": manifests[manifest_id]["fingerprint"]})
                            all_rows.append(metric)
                            # At k=0 the blend IS the semantic branch, so the question worth
                            # answering here is which half of it produced the answer. This is the
                            # attribution for the zero-shot claim and the primary MM-Fit cell.
                            for readout in _evidence_gated_semantic_readouts(args.halo_checkpoint):
                                branch, _, _ = _evidence_gated_readout_spec(readout)
                                semantic_metric = _metric_row(
                                    stream, plans,
                                    _halo_evidence_gated_predictions(
                                        features, stream, plans, args.halo_checkpoint, device,
                                        branch=branch,
                                    ),
                                    bootstrap=args.bootstrap,
                                )
                                semantic_metric.update({
                                    "model": name, "readout": readout, "k": k, "status": "ok",
                                    "feature_fingerprint": fingerprint,
                                    "manifest": manifests[manifest_id]["fingerprint"],
                                })
                                all_rows.append(semantic_metric)
                            if is_v2 and args.classifier_isolation:
                                diagnostic_predictions = _halo_residual_diagnostic_predictions(
                                    features, stream, plans, args.halo_checkpoint, device,
                                )
                                for readout, diagnostic_prediction in diagnostic_predictions.items():
                                    if readout in {"halo-classifier", "halo-classifier-floor"}:
                                        continue
                                    diagnostic_metric = _metric_row(
                                        stream, plans, diagnostic_prediction,
                                        bootstrap=args.bootstrap,
                                    )
                                    diagnostic_metric.update({
                                        "model": name, "readout": readout, "k": k,
                                        "status": "ok", "feature_fingerprint": fingerprint,
                                        "manifest": manifests[manifest_id]["fingerprint"],
                                        "diagnostic_only": True,
                                    })
                                    all_rows.append(diagnostic_metric)
                            # v2's raw-query text bridge and the historical training-bank ConSE
                            # bridge answer the same k=0 question differently. Keep both as named,
                            # disclosed comparison rows; v1 retains its historical single row.
                            if not is_v2:
                                continue
                    if isinstance(stream, MultiDeviceEvalStream) and name in TRAINING_BANK_ZERO_SHOT:
                        all_rows.append({"model": name, "readout": "training-bank-1nn-conse", "k": k,
                                         "window_seconds": float(window_seconds),
                                         "dataset": dataset, "stream": stream_id, "status": "n/a",
                                         "reason": "no matching multi-device training reference bank",
                                         "diagnostic_only": True})
                        if name != "halo":
                            all_rows.append({
                                "model": name, "readout": "equal-weight-normalized-fusion", "k": k,
                                "window_seconds": float(window_seconds),
                                "dataset": dataset, "stream": stream_id, "status": "n/a",
                                "reason": "no matching multi-device semantic reference bank",
                            })
                            all_rows.append({
                                "model": name, "readout": "native_zero_support", "k": k,
                                "window_seconds": float(window_seconds),
                                "dataset": dataset, "stream": stream_id, "status": "n/a",
                                "reason": "no declared native zero-support path",
                            })
                        continue
                    if name in TRAINING_BANK_ZERO_SHOT:
                        if name in feature_errors:
                            all_rows.append({"model": name, "readout": "training-bank-1nn-conse", "k": k,
                                             "dataset": dataset, "stream": stream_id,
                                             "status": "n/a", "reason": feature_errors[name],
                                             "diagnostic_only": True})
                            if name != "halo":
                                all_rows.append({
                                    "model": name, "readout": "equal-weight-normalized-fusion", "k": k,
                                    "dataset": dataset, "stream": stream_id,
                                    "status": "n/a", "reason": feature_errors[name],
                                })
                                all_rows.append({
                                    "model": name, "readout": "native_zero_support", "k": k,
                                    "dataset": dataset, "stream": stream_id, "status": "n/a",
                                    "reason": "no declared native zero-support path",
                                })
                            continue
                        features, fingerprint = features_by_model[name]
                        bank_key = (name, float(window_seconds))
                        if bank_key not in zero_shot_banks:
                            zero_shot_banks[bank_key] = _build_training_reference_bank(
                                name=name, device=device, cache_dir=args.zero_shot_bank_cache,
                                halo_checkpoint=args.halo_checkpoint, halo_state=halo_state,
                                baseline_state=persistent_states.get(name),
                                window_seconds=window_seconds,
                            )
                        bank_features, bank_labels, train_labels, bank_fingerprint = zero_shot_banks[bank_key]
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
                            "diagnostic_only": name != "halo" or halo_has_classifier,
                        })
                        all_rows.append(metric)
                        if name != "halo":
                            fusion_metric = dict(metric)
                            fusion_metric.update({
                                "readout": "equal-weight-normalized-fusion",
                                "diagnostic_only": False,
                                "zero_support_fusion_degenerate": True,
                            })
                            all_rows.append(fusion_metric)
                            all_rows.append({
                                "model": name, "readout": "native_zero_support", "k": k,
                                "dataset": dataset, "stream": stream_id, "status": "n/a",
                                "reason": "no declared native zero-support path",
                            })
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
                    features, fingerprint = _load_or_encode(
                        name=name, stream=stream, device=device, cache_dir=cache_dir,
                        halo_checkpoint=args.halo_checkpoint, baseline_state=state,
                        feature_role="native_zero_shot", memory_cache=feature_memory_cache,
                    )
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
                    fusion_metric = dict(metric)
                    fusion_metric.update({
                        "readout": "equal-weight-normalized-fusion",
                        "zero_support_fusion_degenerate": True,
                    })
                    all_rows.append(fusion_metric)
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
                # Every enrolled external baseline exposes 1-NN beside the primary normalized
                # fusion row. Prototype/ridge remain opt-in diagnostics.
                predictions = _readout_predictions(
                    features, _aligned_labels(stream), stream.eval_labels, plans, device=device,
                    readouts=(None if name == "halo" or args.baseline_diagnostic_readouts
                              else frozenset(("1nn",))),
                )
                for readout, predicted in predictions.items():
                    metric = _metric_row(stream, plans, predicted, bootstrap=args.bootstrap)
                    metric.update({"model": name, "readout": readout, "k": k,
                                   "status": "ok", "feature_fingerprint": fingerprint,
                                   "manifest": manifests[manifest_id]["fingerprint"],
                                   "diagnostic_only": name != "halo" and readout != "1nn"})
                    all_rows.append(metric)
                if (name == "halo" or args.baseline_diagnostic_readouts) and "ridge" not in predictions:
                    all_rows.append({
                        "model": name, "readout": "ridge", "k": k,
                        "dataset": dataset, "stream": stream_id,
                        "status": "n/a",
                        "reason": (
                            "exact query-specific ridge system exceeds "
                            f"{MAX_EXACT_RIDGE_SYSTEM} dimensions"
                        ),
                        "diagnostic_only": name != "halo",
                    })
                if name != "halo":
                    state = persistent_states.get(name)
                    if name in TRAINING_BANK_ZERO_SHOT:
                        if name not in semantic_by_model:
                            bank_key = (name, float(window_seconds))
                            if bank_key not in zero_shot_banks:
                                zero_shot_banks[bank_key] = _build_training_reference_bank(
                                    name=name, device=device, cache_dir=args.zero_shot_bank_cache,
                                    halo_checkpoint=args.halo_checkpoint, halo_state=halo_state,
                                    baseline_state=state, window_seconds=window_seconds,
                                    memory_cache=feature_memory_cache,
                                )
                            bank_features, bank_labels, train_labels, bank_fingerprint = zero_shot_banks[bank_key]
                            semantic_scores, semantic_info = _training_bank_conse_predictions(
                                features, bank_features, bank_labels, train_labels,
                                stream.eval_labels, device, return_scores=True,
                            )
                            semantic_info["training_bank_fingerprint"] = bank_fingerprint
                            semantic_by_model[name] = (semantic_scores, semantic_info)
                        semantic_scores, semantic_info = semantic_by_model[name]
                    elif baselines.REGISTRY[name].supports_native_zero_shot():
                        if name not in semantic_by_model:
                            native_features, _ = _load_or_encode(
                                name=name, stream=stream, device=device, cache_dir=cache_dir,
                                halo_checkpoint=args.halo_checkpoint, baseline_state=state,
                                feature_role="native_zero_shot", memory_cache=feature_memory_cache,
                            )
                            semantic_by_model[name] = (
                                np.asarray(
                                    baselines.REGISTRY[name].candidate_scores_from_features(
                                        native_features, stream.eval_labels, state, device,
                                    ), dtype=np.float64,
                                ),
                                {"zero_support_protocol": "native_candidate_scores"},
                            )
                        semantic_scores, semantic_info = semantic_by_model[name]
                    else:
                        all_rows.append({
                            "model": name, "readout": "equal-weight-normalized-fusion", "k": k,
                            "dataset": dataset, "stream": stream_id, "status": "n/a",
                            "reason": "no native or configured semantic-score path",
                        })
                        continue
                    neighbor_scores = classwise_neighbor_scores(
                        features, stream.eval_labels, plans, device=device,
                    )
                    full_coverage = CoverageCell(
                        supported=tuple(stream.eval_labels), hidden=(), coverage=1.0,
                        requested_coverage=1.0,
                    )
                    fused = equal_weight_normalized_fusion_predictions(
                        semantic_scores, features, stream.eval_labels, plans, full_coverage,
                        classwise_scores=neighbor_scores,
                    )
                    metric = _metric_row(stream, plans, fused, bootstrap=args.bootstrap)
                    metric.update({
                        "model": name, "readout": "equal-weight-normalized-fusion", "k": k,
                        "status": "ok", "feature_fingerprint": fingerprint,
                        "manifest": manifests[manifest_id]["fingerprint"], **semantic_info,
                    })
                    all_rows.append(metric)
                    continue
                if name == "halo":
                    # Differentiable neighbours is used only to train and diagnose the HALO
                    # encoder. Sealed reports use the deployment readouts: 1-NN, prototype,
                    # ridge, and the learned HALO classifier.
                    try:
                        halo_architecture = torch.load(
                            args.halo_checkpoint, map_location="cpu", weights_only=False,
                        ).get("architecture_version")
                        is_v2 = halo_architecture in LEARNED_CLASSIFIER_ARCHITECTURES
                        is_residual = halo_architecture in RESIDUAL_ARCHITECTURES
                        predicted = _halo_token_mixer_predictions(
                            features, stream, plans, args.halo_checkpoint, device,
                            acquisitions=halo_acquisitions,
                        )
                    except ValueError as exc:
                        all_rows.append({"model": name, "readout": "retrieve-mix-vote", "k": k,
                                         "dataset": dataset, "stream": stream_id,
                                         "status": "n/a", "reason": str(exc)})
                    else:
                        metric = _metric_row(stream, plans, predicted, bootstrap=args.bootstrap)
                        metric.update({"model": name, "readout": (
                            "halo-classifier" if is_v2 else "retrieve-mix-vote"
                        ), "k": k,
                                       "status": "ok", "feature_fingerprint": fingerprint,
                                       "manifest": manifests[manifest_id]["fingerprint"]})
                        all_rows.append(metric)
                        if halo_architecture == EVIDENCE_GATED_ARCHITECTURE:
                            # v4 diagnostics. Support evidence does not exist at k=0, and a
                            # text-off blend is undefined for an unenrolled candidate, so do not
                            # fabricate branch scores for those information conditions.
                            for readout in (EVIDENCE_GATED_READOUTS
                                            + _evidence_gated_semantic_readouts(
                                                args.halo_checkpoint)):
                                if k == 0 and readout not in {
                                    "halo-classifier-label-meaning-only",
                                    "halo-classifier-semantic-text-only",
                                    "halo-classifier-semantic-primitives-only",
                                }:
                                    all_rows.append({
                                        "model": name, "readout": readout, "k": k,
                                        "dataset": dataset, "stream": stream_id,
                                        "status": "n/a", "reason": "no enrolled support at k=0",
                                    })
                                    continue
                                if (readout == "halo-classifier-text-off-blend"
                                        and any(set(plan.support_labels) < set(stream.eval_labels)
                                                for plan in plans)):
                                    all_rows.append({
                                        "model": name, "readout": readout, "k": k,
                                        "dataset": dataset, "stream": stream_id,
                                        "status": "n/a",
                                        "reason": "text-off blend is undefined with unenrolled candidates",
                                    })
                                    continue
                                branch, lambda_override, trust_override = _evidence_gated_readout_spec(readout)
                                branch_predicted = _halo_evidence_gated_predictions(
                                    features, stream, plans, args.halo_checkpoint, device,
                                    branch=branch, lambda_override=lambda_override,
                                    trust_override=trust_override,
                                )
                                branch_metric = _metric_row(stream, plans, branch_predicted,
                                                            bootstrap=args.bootstrap)
                                branch_metric.update({"model": name, "readout": readout, "k": k,
                                                      "status": "ok",
                                                      "feature_fingerprint": fingerprint,
                                                      "manifest": manifests[manifest_id]["fingerprint"]})
                                all_rows.append(branch_metric)
                        elif is_v2 and not is_residual:
                            # Contextual head: branch decompositions of the same forward.
                            branches = (
                                ("semantic", "support_floor", "contextual_support")
                                if halo_architecture in _ACQUISITION_CONDITIONED_ARCHITECTURES
                                else ("semantic", "support", "fixed_half")
                            )
                            readout_names = (
                                EVIDENCE_AWARE_READOUTS if halo_architecture == EVIDENCE_AWARE_ARCHITECTURE
                                else CONTEXTUAL_READOUTS if halo_architecture in _ACQUISITION_CONDITIONED_ARCHITECTURES
                                else LEGACY_CONTEXTUAL_READOUTS
                            )
                            for readout, branch in zip(readout_names, branches):
                                if halo_architecture in _ACQUISITION_CONDITIONED_ARCHITECTURES:
                                    branch_predicted = _halo_contextual_residual_predictions(
                                        features, stream, plans, args.halo_checkpoint, device,
                                        branch=branch, acquisitions=halo_acquisitions,
                                    )
                                else:
                                    branch_predicted = _halo_contextual_predictions(
                                        features, stream, plans, args.halo_checkpoint, device,
                                        branch=branch,
                                    )
                                branch_metric = _metric_row(stream, plans, branch_predicted,
                                                            bootstrap=args.bootstrap)
                                branch_metric.update({"model": name, "readout": readout, "k": k,
                                                      "status": "ok", "feature_fingerprint": fingerprint,
                                                      "manifest": manifests[manifest_id]["fingerprint"]})
                                all_rows.append(branch_metric)
                        if is_residual:
                            predicted = _halo_residual_predictions(
                                features, stream, plans, args.halo_checkpoint, device,
                                residual_enabled=False, text_term_enabled=False,
                            )
                            metric = _metric_row(stream, plans, predicted, bootstrap=args.bootstrap)
                            metric.update({"model": name, "readout": "halo-classifier-residual-off",
                                           "k": k, "status": "ok", "feature_fingerprint": fingerprint,
                                           "manifest": manifests[manifest_id]["fingerprint"]})
                            all_rows.append(metric)
                            if args.classifier_isolation:
                                diagnostic_predictions = _halo_residual_diagnostic_predictions(
                                    features, stream, plans, args.halo_checkpoint, device,
                                )
                                for readout, diagnostic_prediction in diagnostic_predictions.items():
                                    # Canonical full and floor-equivalent rows were written above.
                                    if readout in {"halo-classifier", "halo-classifier-floor"}:
                                        continue
                                    diagnostic_metric = _metric_row(
                                        stream, plans, diagnostic_prediction,
                                        bootstrap=args.bootstrap,
                                    )
                                    diagnostic_metric.update({
                                        "model": name, "readout": readout, "k": k,
                                        "status": "ok", "feature_fingerprint": fingerprint,
                                        "manifest": manifests[manifest_id]["fingerprint"],
                                        "diagnostic_only": True,
                                    })
                                    all_rows.append(diagnostic_metric)
        # Attach model-input disclosures uniformly, including N/A rows. This is deliberately done
        # once at the cell boundary so no readout can forget the duration/device fairness fields.
        for row in all_rows[cell_row_start:]:
            name = row["model"]
            row.update(_native_capabilities(name))
            learned_head = name == "halo" and row.get("readout") in {
                "halo-classifier", "halo-classifier-residual-off",
                "halo-classifier-text-only", "halo-classifier-support-residual-only",
                "halo-classifier-candidate-residual-only", "halo-classifier-residual-only",
                "halo-classifier-support-label-shuffled", *CONTEXTUAL_READOUTS,
            }
            row["parameters_m"] = round(_parameter_count_m(
                name, halo_state, halo_checkpoint=args.halo_checkpoint,
                include_classifier=learned_head,
            ), 3)
            if name in _PARAMETER_BREAKDOWN_M:
                row["parameter_breakdown_m"] = _PARAMETER_BREAKDOWN_M[name]
            if name == "halo":
                accounting = {"padded": False, "padded_fraction": 0.0}
                mode = "native" if isinstance(stream, MultiDeviceEvalStream) else "single-device"
            else:
                adapter = baselines.REGISTRY[name]
                accounting = adapter.input_accounting(stream)
                mode = ("native" if adapter.supports_multi_device else "per-device-pooled") \
                    if isinstance(stream, MultiDeviceEvalStream) else "single-device"
            row.setdefault("window_seconds", float(window_seconds))
            row["n_devices"] = len(stream.devices) if isinstance(stream, MultiDeviceEvalStream) else 1
            row["multi_device_mode"] = mode
            row.update(accounting)
            row["source_slice_fingerprint"] = source_slice_fingerprint(stream)
        checkpoint_progress(completed_cells=cell_index, current=None)
    validate_result_rows(
        all_rows,
        expected_cells=[(duration, dataset, stream_id)
                        for duration, dataset, stream_id, _ in requested_cells],
        models=args.models,
        k_values=sorted(set(args.k)),
    )
    _atomic_json(args.out / "run_metadata.json", {
        "result_schema": "prospective-results-v1-20260918" if args.scope == "prospective"
        else "sealed-results-v3-20260916",
        "manifest_protocol": "prospective-mobiact-v1-20260918" if args.scope == "prospective"
        else "sealed-manifest-v2-20260916",
        "manifest_generator": "numpy-choice-json-fingerprint-v1",
        "feature_cache_schema": FEATURE_CACHE_SCHEMA,
        "models": args.models,
        "scope": args.scope,
        "k": sorted(set(args.k)),
        "window_seconds": sorted(set(map(float, args.window_seconds))),
        "n_cells": len(requested_cells),
        "complete": True,
    })
    _atomic_json(args.out / "run_provenance.json", _run_provenance(
        list(sys.argv), device=device, halo_checkpoint=args.halo_checkpoint,
    ))
    (args.out / "episode_manifests.json").write_text(json.dumps(manifests, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(all_rows, indent=2, allow_nan=True) + "\n")
    _write_markdown(all_rows, args.out / "RESULTS.md")
    checkpoint_progress(completed_cells=total_cells, current=None)
    print(f"[sealed-eval] wrote {args.out / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
