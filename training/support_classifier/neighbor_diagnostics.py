"""Fast, exploratory diagnostics for the parameter-free HALO neighbor control.

This script consumes an existing sealed-evaluation feature cache and reconstructs the immutable
episodes from ``sealed_eval``.  It never invokes an encoder, alters a checkpoint, or selects a
checkpoint.  Its purpose is to identify limitations of a plain neighbor rule before adding a
learned support-conditioned classifier.

The produced report is exploratory sealed-test analysis.  It must not be used for architecture
selection without confirming the resulting hypothesis on the development protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from baselines import scoring
from baselines.data import load_eval_stream, load_multi_device_stream, source_slice_fingerprint
from training.support_classifier.sealed_eval import (
    SEED, _aligned_labels, _differentiable_neighbor_predictions, _readout_predictions,
    _neighbor_prototype_predictions_batched, _normalise, build_manifest, evaluation_cells,
)


def _cache_features(cache_dir: Path, stream) -> np.ndarray:
    """Load the one historical HALO cache matching this exact sealed stream."""
    matches: list[Path] = []
    fingerprint = source_slice_fingerprint(stream)
    for meta_path in cache_dir.glob(f"{stream.dataset}__{stream.stream}__halo__*.json"):
        meta = json.loads(meta_path.read_text())
        if (meta.get("n_windows") == stream.n_windows
                and meta.get("source_slice_fingerprint") == fingerprint):
            matches.append(meta_path.with_suffix(".npy"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one cached embedding matrix for {stream.dataset}/{stream.stream}, "
            f"found {len(matches)}. Use the cache belonging to the promoted checkpoint."
        )
    values = np.load(matches[0])
    if values.ndim != 2 or values.shape[0] != stream.n_windows or not np.isfinite(values).all():
        raise ValueError(f"invalid cached features at {matches[0]}")
    return _normalise(values).astype(np.float32, copy=False)


def _plans_to_arrays(plans, candidates: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_to_id = {label: index for index, label in enumerate(candidates)}
    query = np.asarray([plan.query for plan in plans], dtype=np.int64)
    support = np.asarray([plan.support for plan in plans], dtype=np.int64)
    support_labels = np.asarray(
        [[label_to_id[label] for label in plan.support_labels] for plan in plans], dtype=np.int64,
    )
    return query, support, support_labels, np.arange(len(candidates), dtype=np.int64)


def _soft_logits(query: np.ndarray, support: np.ndarray, bindings: np.ndarray, candidates: int,
                 *, temperature: float = 0.07, present: np.ndarray | None = None) -> np.ndarray:
    """Exact NumPy equivalent of the differentiable-neighbor readout, with optional row removal."""
    scores = np.einsum("bd,bsd->bs", query, support) / temperature
    if present is not None:
        scores = np.where(present, scores, -np.inf)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    votes = np.zeros((len(query), candidates), dtype=np.float32)
    np.add.at(votes, (np.arange(len(query))[:, None], bindings), weights)
    return np.log(np.maximum(votes, 1e-12))


def _metrics(truth: np.ndarray, predictions: Iterable[str]) -> dict[str, float]:
    values = scoring.classification_metrics(truth.tolist(), list(predictions))
    return {key: float(values[key]) for key in ("accuracy", "balanced_accuracy", "f1_macro")}


def _standard_predictions(features: np.ndarray, candidates: tuple[str, ...], plans, device: torch.device,
                          *, include_ridge: bool) -> dict[str, list[str]]:
    """Run the cheap exact controls by default; ridge is explicit because it is query-solve bound."""
    if include_ridge:
        return _readout_predictions(features, np.empty(0), candidates, plans, device=device)
    return _neighbor_prototype_predictions_batched(features, candidates, plans, device)


def _neighborhood_summary(features: np.ndarray, truth: np.ndarray, candidates: tuple[str, ...], plans) -> dict:
    query_rows, support_rows, bindings, _ = _plans_to_arrays(plans, candidates)
    q, s = features[query_rows], features[support_rows]
    truth_ids = np.asarray([candidates.index(str(label)) for label in truth[query_rows]], dtype=np.int64)
    similarity = np.einsum("bd,bsd->bs", q, s)
    soft = _soft_logits(q, s, bindings, len(candidates)).argmax(axis=1)
    order = np.argsort(-similarity, axis=1)
    is_true = bindings == truth_ids[:, None]
    true_best = np.where(is_true, similarity, -np.inf).max(axis=1)
    rank = 1 + (similarity > true_best[:, None]).sum(axis=1)
    result = {
        "n_queries": int(len(q)),
        "soft_accuracy": float(np.mean(soft == truth_ids)),
        "correct_label_best_support_rank_median": float(np.median(rank)),
        "correct_label_best_support_top1": float(np.mean(rank == 1)),
        "correct_label_best_support_top5": float(np.mean(rank <= 5)),
    }
    wrong = soft != truth_ids
    if wrong.any():
        result.update({
            "wrong_queries": int(wrong.sum()),
            "wrong_correct_label_best_support_rank_median": float(np.median(rank[wrong])),
            "wrong_correct_label_best_support_top1": float(np.mean(rank[wrong] == 1)),
            "wrong_correct_label_best_support_top5": float(np.mean(rank[wrong] <= 5)),
        })
    return result


def _support_sensitivity(features: np.ndarray, truth: np.ndarray, candidates: tuple[str, ...], plans,
                         *, maximum: int, seed: int) -> dict:
    """Measure whether one support row can disproportionately change a soft-neighbor decision."""
    if not plans:
        return {"n_sampled": 0}
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(plans), size=min(maximum, len(plans)), replace=False)
    sample = [plans[int(index)] for index in chosen]
    query_rows, support_rows, bindings, _ = _plans_to_arrays(sample, candidates)
    q, s = features[query_rows], features[support_rows]
    truth_ids = np.asarray([candidates.index(str(label)) for label in truth[query_rows]], dtype=np.int64)
    base = _soft_logits(q, s, bindings, len(candidates)).argmax(axis=1)
    similarities = np.einsum("bd,bsd->bs", q, s)
    top = similarities.argmax(axis=1)
    random_row = rng.integers(0, similarities.shape[1], size=len(sample))

    def removed_prediction(rows: np.ndarray) -> np.ndarray:
        present = np.ones(similarities.shape, dtype=bool)
        present[np.arange(len(sample)), rows] = False
        return _soft_logits(q, s, bindings, len(candidates), present=present).argmax(axis=1)

    remove_top = removed_prediction(top)
    remove_random = removed_prediction(random_row)
    return {
        "n_sampled": int(len(sample)),
        "top_similarity_support_removal_flip_rate": float(np.mean(remove_top != base)),
        "random_support_removal_flip_rate": float(np.mean(remove_random != base)),
        "base_accuracy": float(np.mean(base == truth_ids)),
        "after_top_removal_accuracy": float(np.mean(remove_top == truth_ids)),
        "after_random_removal_accuracy": float(np.mean(remove_random == truth_ids)),
    }


def _aggregate(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
    return {"mean": float(values.mean()), "std_across_cells": float(values.std()), "n_cells": int(len(values))}


def _historical_ridge(eval_dir: Path, *, window_seconds: float, k: int) -> list[dict]:
    """Reuse the exact ridge rows already computed for this immutable checkpoint/manifest.

    Ridge is deliberately not recomputed during a quick diagnostic pass: its query-specific linear
    solves are much slower than the other parameter-free rules, while the promoted sealed run has
    already recorded the exact same readout over the same cache and episode construction.
    """
    rows = json.loads((eval_dir / "results.json").read_text())
    return [
        {metric: float(row[metric]) for metric in ("accuracy", "balanced_accuracy", "f1_macro")}
        for row in rows
        if row.get("model") == "halo" and row.get("readout") == "ridge"
        and row.get("status") == "ok" and row.get("k") == k
        and float(row.get("window_seconds", -1)) == window_seconds
    ]


def _markdown(result: dict) -> str:
    lines = ["# Neighbor-control diagnostics", "",
             "Exploratory sealed-test analysis using cached embeddings and immutable episodes. "
             "No encoder or classifier was trained.", "",
             "## Decision Rules", "",
             "| k | rule | macro-F1, mean across cells | accuracy, mean across cells |", "|---:|---|---:|---:|"]
    for k, rules in result["decision_rules"].items():
        for name, row in rules.items():
            lines.append(f"| {k} | {name} | {row['f1_macro']['mean']:.4f} | {row['accuracy']['mean']:.4f} |")
    lines.extend(["", "## Support Draw Variance", "",
                  "| k | rule | mean macro-F1 | seed standard deviation |", "|---:|---|---:|---:|"])
    for k, rules in result["support_draws"].items():
        for name, row in rules.items():
            lines.append(
                f"| {k} | {name} | {row['mean_f1_macro']:.4f} | "
                f"{row['mean_within_cell_seed_std']:.4f} |"
            )
    lines.extend(["", "## Interpretation", "",
                  "A low correct-label support rank on incorrect queries points to retrieval geometry. "
                  "A high rank but an incorrect soft vote points to aggregation/calibration, which is the narrow "
                  "case where a learned support-conditioned re-ranker is justified. Support-removal flip rates "
                  "measure dependence on individual enrolled examples; they are not a robustness claim.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path,
                        default=Path("training/support_classifier/evaluations/sealed_halo_fixed_mr_neighbors_8s_4res_e2e_20260913"))
    parser.add_argument("--out", type=Path,
                        default=Path("training/support_classifier/evaluations/neighbor_diagnostics_20260914"))
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260912, 20260913, 20260914])
    parser.add_argument("--max-sensitivity-queries", type=int, default=256)
    parser.add_argument("--include-ridge", action="store_true",
                        help="also run exact query-specific ridge; this can take minutes per cell")
    args = parser.parse_args()
    if any(k <= 0 for k in args.k):
        parser.error("this diagnostic is defined only for enrolled k > 0")
    cache_dir = args.eval_dir / "feature_cache"
    if not cache_dir.exists():
        parser.error(f"missing cache directory: {cache_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict = {"protocol": {"checkpoint_eval_dir": str(args.eval_dir), "window_seconds": args.window_seconds,
                                  "k": args.k, "seeds": args.seeds, "seed": SEED,
                                  "note": "exploratory sealed analysis; cached embeddings only"},
                    "decision_rules": {}, "support_draws": {}, "neighborhoods": {}, "support_sensitivity": {}}
    per_k_rules: dict[int, dict[str, list[dict]]] = {k: {name: [] for name in ("1nn", "prototype", "ridge", "soft-neighbor")} for k in args.k}
    # Keep draw scores grouped by evaluation cell. Pooling them first would mistake ordinary
    # between-dataset difficulty for enrollment-set instability.
    per_k_draw: dict[int, dict[str, list[list[float]]]] = {
        k: {name: [] for name in ("1nn", "prototype", "soft-neighbor")} for k in args.k
    }
    for duration, dataset, stream_id, device_ids in evaluation_cells([args.window_seconds]):
        stream = (load_multi_device_stream(dataset, device_ids, alignment="native", window_seconds=duration,
                                           apply_quality_screen=True) if device_ids else
                  load_eval_stream(dataset, stream_id, alignment="native", window_seconds=duration,
                                   apply_quality_screen=True))
        features = _cache_features(cache_dir, stream)
        truth = _aligned_labels(stream)
        cell = f"{dataset}/{stream_id}"
        result["neighborhoods"][cell] = {}
        result["support_sensitivity"][cell] = {}
        for k in args.k:
            plans = build_manifest(stream, k, seed=SEED)
            if not plans:
                continue
            predicted = _standard_predictions(
                features, stream.eval_labels, plans, device, include_ridge=args.include_ridge,
            )
            predicted["soft-neighbor"] = _differentiable_neighbor_predictions(features, stream.eval_labels, plans, device)
            labels = truth[np.asarray([plan.query for plan in plans], dtype=np.int64)]
            for name, values in predicted.items():
                if name in per_k_rules[k]:
                    per_k_rules[k][name].append(_metrics(labels, values))
            result["neighborhoods"][cell][str(k)] = _neighborhood_summary(features, truth, stream.eval_labels, plans)
            result["support_sensitivity"][cell][str(k)] = _support_sensitivity(
                features, truth, stream.eval_labels, plans, maximum=args.max_sensitivity_queries,
                seed=SEED + k,
            )
            cell_draws = {name: [] for name in per_k_draw[k]}
            for seed in args.seeds:
                draw_plans = build_manifest(stream, k, seed=seed)
                draw = _standard_predictions(
                    features, stream.eval_labels, draw_plans, device, include_ridge=False,
                )
                draw["soft-neighbor"] = _differentiable_neighbor_predictions(
                    features, stream.eval_labels, draw_plans, device,
                )
                draw_labels = truth[np.asarray([plan.query for plan in draw_plans], dtype=np.int64)]
                for name in per_k_draw[k]:
                    cell_draws[name].append(_metrics(draw_labels, draw[name])["f1_macro"])
            for name, values in cell_draws.items():
                per_k_draw[k][name].append(values)
        print(f"[neighbor-diagnostics] completed {cell}", flush=True)
    for k, rules in per_k_rules.items():
        historical_ridge = _historical_ridge(args.eval_dir, window_seconds=args.window_seconds, k=k)
        if historical_ridge:
            rules["ridge"] = historical_ridge
        result["decision_rules"][str(k)] = {
            name: {metric: _aggregate(rows, metric) for metric in ("accuracy", "balanced_accuracy", "f1_macro")}
            for name, rows in rules.items() if rows
        }
    for k, rules in per_k_draw.items():
        result["support_draws"][str(k)] = {
            name: {"mean_f1_macro": float(np.mean(values)),
                   "mean_within_cell_seed_std": float(np.mean([np.std(cell) for cell in values])),
                   "max_within_cell_seed_std": float(np.max([np.std(cell) for cell in values])),
                   "n_cells": int(len(values)), "n_seeds": int(len(args.seeds))}
            for name, values in rules.items() if values
        }
    (args.out / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "REPORT.md").write_text(_markdown(result))
    print(f"[neighbor-diagnostics] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
