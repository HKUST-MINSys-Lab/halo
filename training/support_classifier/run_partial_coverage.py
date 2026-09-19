"""Scenario 1 runner — partial enrollment coverage across the sealed roster.

Evaluation only.  Loads the same sealed streams, the same execution-disjoint manifests and the same
cached features as ``sealed_eval``, then hides the supports of a fixed candidate subset per cell and
scores every model three ways:

* **support-only readouts** (1-NN / prototype / ridge) restricted to enrolled candidates,
* **equal-weight normalized fusion** of semantic + cosine 1-NN evidence, using one fixed rule,
* **HALO's own classifier**, which handles unenrolled candidates natively because it was trained to.

Rows are emitted three times per readout: over all queries, over queries whose ground truth is
enrolled, and over queries whose ground truth is *not*.  The third split is the scenario.

This module never writes into the sealed comparison directory and never selects a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import baselines
from model.support.factory import CONTEXTUAL_ARCHITECTURE
from baselines import scoring
from halo.paths import CACHE_DIR
from baselines.data import load_eval_stream, load_global_labels, load_multi_device_stream
from data.scripts.labels.canonical_labels import canonicalize

from .partial_coverage import (
    CoverageCell,
    choose_hidden_candidates,
    hide_supports,
    equal_weight_normalized_fusion_predictions,
    classwise_neighbor_scores,
    support_only_predictions,
    truth_split,
)
from .sealed_eval import (
    PRIMARY_BASELINES,
    SEED,
    TRAINING_BANK_ZERO_SHOT,
    _aligned_labels,
    _halo_residual_predictions,
    halo_acquisition_rows,
    _build_training_reference_bank,
    _file_hash,
    _load_or_encode,
    evaluation_cells,
    manifest_fingerprint,
    QueryPlan,
)
from .scenarios import build_cross_manifest
from training.tokenizer.eval_transfer import build_encoder

DEFAULT_K = (1, 4, 8)
DEFAULT_COVERAGE = 0.5


# --------------------------------------------------------------------- text scores


def conse_scores(
    query_features: np.ndarray,
    bank_features: np.ndarray,
    bank_label_ids: np.ndarray,
    train_labels,
    target_labels,
    device: torch.device,
) -> np.ndarray:
    """ConSE bridge similarities, ``(n_queries, n_candidates)``.

    Reproduces the score matrix that :func:`baselines.scoring.conse_predict` takes its arg-max over,
    so fusion ranks a text-bridge model exactly as its own zero-support path would.
    """
    query = query_features / np.maximum(
        np.linalg.norm(query_features, axis=1, keepdims=True), 1e-12)
    bank = bank_features / np.maximum(
        np.linalg.norm(bank_features, axis=1, keepdims=True), 1e-12)
    label_ids = np.asarray(bank_label_ids, dtype=np.int64)

    q = torch.as_tensor(query.astype(np.float32), device=device)
    b = torch.as_tensor(bank.astype(np.float32), device=device)
    nearest = torch.empty(len(q), dtype=torch.long)
    for start in range(0, len(q), 1024):
        nearest[start:start + 1024] = (q[start:start + 1024] @ b.T).argmax(dim=1).cpu()
    del q, b
    if device.type == "cuda":
        torch.cuda.empty_cache()

    probs = np.zeros((len(query), len(train_labels)), dtype=np.float32)
    probs[np.arange(len(query)), label_ids[nearest.numpy()]] = 1.0

    return scoring.conse_score_matrix(
        probs, list(train_labels), list(target_labels), top_T=1,
    )


def native_text_scores(name: str, features: np.ndarray, candidates, state, device) -> np.ndarray:
    """Native candidate scores from a released model, preserving its own metric."""
    adapter = baselines.REGISTRY[name]
    roster = list(candidates)
    scores = np.asarray(adapter.candidate_scores_from_features(features, roster, state, device),
                        dtype=np.float64)
    if scores.shape != (len(features), len(roster)) or not np.isfinite(scores).all():
        raise ValueError(f"{name}: invalid native candidate-score matrix {scores.shape}")
    predicted, _ = adapter.predict_candidates_from_features(features, roster, state, device)
    if predicted != [roster[index] for index in scores.argmax(axis=1).tolist()]:
        raise RuntimeError(f"{name}: native score argmax does not reproduce native prediction")
    return scores


# --------------------------------------------------------------------- metric rows


def _split_row(stream, plans, predictions, rows: np.ndarray, *, bootstrap: int,
               truth_restricted: bool) -> dict:
    """Metrics over one subset of the episodes, or a disclosed empty row.

    ``truth_restricted`` marks the two splits whose ground truth spans only part of the roster.
    Balanced accuracy remains the primary conditional metric because it averages recall over the
    classes that can be true in that split. Macro F1 is still a valid secondary measurement when
    its vocabulary is defined as the observed truth/prediction union, so it is retained rather than
    suppressed.
    """
    if not len(rows):
        return {"status": "n/a", "reason": "no query in this split", "n_queries": 0}
    indices = np.asarray([plans[i].query for i in rows], dtype=np.int64)
    truth = _aligned_labels(stream)[indices].tolist()
    predicted = [predictions[i] for i in rows]
    subjects = np.asarray(stream.subjects)[indices]
    metrics = scoring.classification_metrics(truth, predicted)
    metrics["truth_label_set"] = sorted(set(truth))
    metrics["f1_scored_classes"] = scoring.macro_f1_classes(truth, predicted)
    per_label = scoring.per_class_f1(truth, predicted)
    training_concepts = {canonicalize(label) for label in load_global_labels()}
    seen_scores = [score for label, score in per_label.items()
                   if canonicalize(label) in training_concepts]
    unseen_scores = [score for label, score in per_label.items()
                     if canonicalize(label) not in training_concepts]
    metrics.update({
        "per_label_f1": per_label,
        "f1_macro_seen": float(np.mean(seen_scores)) if seen_scores else None,
        "f1_macro_unseen": float(np.mean(unseen_scores)) if unseen_scores else None,
    })
    metric_name = "balanced_accuracy" if truth_restricted else "f1_macro"
    if truth_restricted:
        metrics["f1_macro_note"] = (
            "secondary conditional metric over the observed truth/prediction label union; "
            "balanced_accuracy is primary"
        )
    if bootstrap > 0:
        metrics.update(scoring.subject_bootstrap_ci(
            truth, predicted, subjects, metric=metric_name, B=bootstrap))
    metrics.update({"status": "ok", "n_queries": int(len(rows)),
                    "bootstrap_B": int(bootstrap), "primary_metric": metric_name})
    return metrics


def _harmonic_mean(left: float | None, right: float | None) -> float | None:
    """Symmetric summary that cannot hide failure on either coverage split."""
    if left is None or right is None:
        return None
    left = float(left)
    right = float(right)
    if not np.isfinite(left) or not np.isfinite(right) or left < 0.0 or right < 0.0:
        raise ValueError("coverage metrics must be finite and non-negative")
    total = left + right
    return 0.0 if total == 0.0 else 2.0 * left * right / total


def emit_rows(stream, plans, predictions, cell: CoverageCell, *, model: str, readout: str,
              k: int, window_seconds: float, bootstrap: int, manifest: str,
              scenario: str = "s1_partial_coverage", variant: str | None = None,
              extra: dict | None = None) -> list[dict]:
    """One row per truth split, each carrying the coverage configuration."""
    truth = _aligned_labels(stream)[
        np.asarray([plan.query for plan in plans], dtype=np.int64)].tolist()
    supported_rows, hidden_rows = truth_split(truth, cell)
    splits = {"all": np.arange(len(plans))}
    if cell.hidden:
        splits.update({"truth_enrolled": supported_rows, "truth_unenrolled": hidden_rows})
    out = []
    for split, rows in splits.items():
        row = _split_row(stream, plans, predictions, rows, bootstrap=bootstrap,
                         truth_restricted=split != "all")
        row.update({
            "model": model, "readout": readout, "k": k, "coverage_split": split,
            "dataset": stream.dataset,
            "stream": getattr(stream, "stream", None) or getattr(stream, "cell_id", ""),
            "window_seconds": float(window_seconds),
            "n_candidates": len(stream.eval_labels),
            "coverage": cell.coverage, "requested_coverage": cell.requested_coverage,
            "supported_candidates": list(cell.supported),
            "hidden_candidates": list(cell.hidden),
            "coverage_fingerprint": cell.fingerprint,
            "manifest": manifest,
            "scenario": scenario,
        })
        if variant is not None:
            row["variant"] = variant
        if extra:
            row.update(extra)
        if split == "truth_unenrolled" and len(rows):
            supported = set(cell.supported)
            row["false_enrollment_pull"] = float(np.mean(
                [predictions[index] in supported for index in rows]
            ))
        out.append(row)
    if cell.hidden:
        by_split = {row["coverage_split"]: row for row in out}
        enrolled = by_split["truth_enrolled"]
        unenrolled = by_split["truth_unenrolled"]
        for metric in ("f1_macro", "balanced_accuracy", "accuracy"):
            value = _harmonic_mean(enrolled.get(metric), unenrolled.get(metric))
            for row in out:
                row[f"coverage_hmean_{metric}"] = value
    return out


def cannot_attempt_rows(stream, cell, *, model, readout, k, window_seconds, reason,
                        scenario: str = "s1_partial_coverage", variant: str | None = None,
                        extra: dict | None = None) -> list[dict]:
    """A model with no text path cannot name an unenrolled candidate; say so, do not score it."""
    row = {
        "model": model, "readout": readout, "k": k, "coverage_split": "truth_unenrolled",
        "dataset": stream.dataset,
        "stream": getattr(stream, "stream", None) or getattr(stream, "cell_id", ""),
        "window_seconds": float(window_seconds), "status": "cannot_attempt", "reason": reason,
        "coverage": cell.coverage, "coverage_fingerprint": cell.fingerprint,
        "supported_candidates": list(cell.supported),
        "hidden_candidates": list(cell.hidden),
        "scenario": scenario,
    }
    if variant is not None:
        row["variant"] = variant
    if extra:
        row.update(extra)
    return [row]


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=["halo", *PRIMARY_BASELINES])
    parser.add_argument("--halo-checkpoint", type=Path, default=None)
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument("--window-seconds", nargs="+", type=float, default=[8.0])
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE,
                        help="fraction of candidates that keep their enrolment")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap", type=int, default=scoring.BOOTSTRAP_B)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--zero-shot-bank-cache", type=Path,
                        default=CACHE_DIR / "evaluations" / "zero_shot_feature_cache")
    parser.add_argument("--smoke", action="store_true",
                        help="one cell, one k, no bootstrap: wiring check only")
    args = parser.parse_args()

    if any(value < 1 for value in args.k):
        parser.error("partial coverage needs at least one enrolled support per live candidate")
    if any(not np.isfinite(value) or value <= 0 for value in args.window_seconds):
        parser.error("--window-seconds values must be finite and positive")
    if not 0.0 < args.coverage < 1.0:
        parser.error("--coverage must lie strictly between zero and one")
    unknown = sorted(set(args.models) - set(PRIMARY_BASELINES) - {"halo"})
    if unknown:
        parser.error(f"models outside the registered primary roster: {unknown}")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not repeat a provider")
    if "halo" in args.models and args.halo_checkpoint is None:
        parser.error("--halo-checkpoint is required when model list includes halo")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    halo_state = None
    halo_requires_acquisition = False
    if "halo" in args.models:
        blob = torch.load(args.halo_checkpoint, map_location="cpu", weights_only=False)
        halo_state = (build_encoder(blob, device).eval(), _file_hash(args.halo_checkpoint))
        halo_requires_acquisition = blob.get("architecture_version") == CONTEXTUAL_ARCHITECTURE
    provider_states = {
        name: baselines.REGISTRY[name].setup_features(device)
        for name in args.models if name != "halo"
    }
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache or (args.out / "feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cells = [cell for cell in evaluation_cells(args.window_seconds) if not cell[3]]
    ks = sorted(set(args.k))
    bootstrap = 0 if args.smoke else args.bootstrap
    if args.smoke:
        cells, ks = cells[:1], ks[:1]

    rows: list[dict] = []
    banks: dict[str, tuple] = {}
    coverage_cells: dict[str, dict] = {}
    for window_seconds, dataset, stream_id, device_ids in cells:
        stream = (load_multi_device_stream(dataset, device_ids, alignment="native",
                                           window_seconds=window_seconds, apply_quality_screen=True)
                  if device_ids else
                  load_eval_stream(dataset, stream_id, alignment="native",
                                   window_seconds=window_seconds, apply_quality_screen=True))
        features: dict[str, tuple[np.ndarray, str]] = {}
        native_features: dict[str, np.ndarray] = {}
        errors: dict[str, str] = {}
        for name in args.models:
            try:
                features[name] = _load_or_encode(
                    name=name, stream=stream, device=device, cache_dir=cache_dir,
                    halo_checkpoint=args.halo_checkpoint,
                    baseline_state=provider_states.get(name), halo_state=halo_state,
                )
            except baselines.UnsupportedEvaluationCell as exc:
                errors[name] = str(exc)

        for k in ks:
            # Match the unified scenario runner: partial-enrolment supports are cross-subject.
            cross = build_cross_manifest(
                stream, stream, k, seed=args.seed, relation="within_cross_subject",
                same_subject=False,
            )
            plans = [QueryPlan(
                query=plan.query,
                support=tuple(row - cross.offset for row in plan.support),
                support_labels=plan.support_labels,
            ) for plan in cross.plans]
            if not plans:
                continue
            cell = choose_hidden_candidates(
                stream.eval_labels, coverage=args.coverage,
                # A coverage cell names one deployment condition, independent of k.
                seed_parts=(args.seed, dataset, stream_id, window_seconds, args.coverage),
            )
            covered = hide_supports(plans, cell)
            fingerprint = manifest_fingerprint(covered)
            key = f"{dataset}/{stream_id}/w={window_seconds:g}/k={k}"
            coverage_cells[key] = {**asdict(cell), "manifest": fingerprint,
                                   "n_episodes": len(covered)}

            for name in args.models:
                if name in errors:
                    rows.append({"model": name, "readout": "all", "k": k, "status": "n/a",
                                 "reason": errors[name], "dataset": dataset, "stream": stream_id,
                                 "scenario": "s1_partial_coverage"})
                    continue
                matrix, feature_fingerprint = features[name]
                neighbor_scores = classwise_neighbor_scores(
                    matrix, stream.eval_labels, covered, device=device,
                )
                shared = dict(k=k, window_seconds=window_seconds, bootstrap=bootstrap,
                              manifest=fingerprint,
                              extra={"feature_fingerprint": feature_fingerprint})

                for readout, predicted in support_only_predictions(
                        matrix, stream.eval_labels, covered, cell,
                        classwise_scores=neighbor_scores).items():
                    rows.extend(emit_rows(stream, covered, predicted, cell,
                                          model=name, readout=readout, **shared))

                text = None
                if name in TRAINING_BANK_ZERO_SHOT:
                    if name not in banks:
                        banks[name] = _build_training_reference_bank(
                            name=name, device=device, cache_dir=args.zero_shot_bank_cache,
                            halo_checkpoint=args.halo_checkpoint,
                            halo_state=halo_state,
                            baseline_state=provider_states.get(name),
                            window_seconds=window_seconds)
                    bank_features, bank_labels, train_labels, _ = banks[name]
                    text = conse_scores(matrix, bank_features, bank_labels, train_labels,
                                        stream.eval_labels, device)
                elif baselines.REGISTRY.get(name) is not None and \
                        baselines.REGISTRY[name].supports_native_zero_shot():
                    state = provider_states[name]
                    if name not in native_features:
                        native_features[name], _ = _load_or_encode(
                            name=name, stream=stream, device=device, cache_dir=cache_dir,
                            halo_checkpoint=args.halo_checkpoint, baseline_state=state,
                            halo_state=halo_state, feature_role="native_zero_shot",
                        )
                    text = native_text_scores(
                        name, native_features[name], stream.eval_labels, state, device,
                    )

                if text is not None:
                    rows.extend(emit_rows(
                        stream, covered,
                        equal_weight_normalized_fusion_predictions(
                            text, matrix, stream.eval_labels, covered, cell,
                            classwise_scores=neighbor_scores,
                        ),
                        cell, model=name, readout="equal-weight-normalized-fusion", **shared))
                elif name != "halo":
                    rows.extend(cannot_attempt_rows(
                        stream, cell, model=name, readout="equal-weight-normalized-fusion", k=k,
                        window_seconds=window_seconds,
                        reason="no text path: unenrolled candidates are unreachable"))

                if name == "halo" and args.halo_checkpoint is not None:
                    acquisitions = (
                        halo_acquisition_rows(stream, halo_state[0], device)
                        if halo_requires_acquisition else None
                    )
                    predicted = _halo_residual_predictions(
                        matrix, stream, covered, args.halo_checkpoint, device,
                        acquisitions=acquisitions)
                    rows.extend(emit_rows(stream, covered, predicted, cell,
                                          model=name, readout="halo-classifier", **shared))

    (args.out / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    (args.out / "coverage_cells.json").write_text(json.dumps(coverage_cells, indent=2, sort_keys=True))
    print(f"wrote {len(rows)} rows to {args.out/'results.json'}")


if __name__ == "__main__":
    main()
