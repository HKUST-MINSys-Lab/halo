"""Discovery driver (retired): cluster every sealed single-device cell for every encoder from cached features.

``evaluate_cell`` is the pure core (features + truth + roster + text vectors → rows) and is what
the unit tests exercise; ``main`` only does I/O, mirroring the sealed evaluator's stream loading so
the feature cache is hit rather than re-encoded. Nothing is run without an explicit invocation.

Smoke: ``python -m evaluation.discovery.run --out /tmp/r1 --models halo --halo-checkpoint
<ckpt> --feature-cache <shared cache> --cells 1 --no-umap --null-draws 20 --k-range 2 6``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

import baselines
from baselines.data import load_eval_stream, source_slice_fingerprint
from evaluation.features import FeatureMemoryCache
from evaluation.manifests import SEED, _aligned_labels, evaluation_cells
from evaluation.metrics import ami, ari, davies_bouldin, hungarian_accuracy, nmi, silhouette
from evaluation.provenance import ArtifactProvenance, Rung, _atomic_json, write_artifact
from evaluation.discovery.cluster import (
    ALGORITHMS, DEFAULT_K_RANGE, cluster_cell, estimate_k, prepare_features, umap_silhouette,
)
from evaluation.discovery.name import name_clusters, naming_outcomes, oracle_assignment
from evaluation.discovery.rsa import (
    class_centroids, cluster_centroids_by_label, dimension_ablation, rsa, shuffled_null,
)
from evaluation.zero_shot import ProviderScorer, _normalise, zero_shot_feature_role

RUNG_MODELS = ("halo", "harnet5", "harnet10", "limubert_x", "unimts", "normwear")


def evaluate_cell(
    *,
    features_raw: np.ndarray,
    naming_features: np.ndarray | None,
    truth: Sequence[str],
    classes: Sequence[str],
    text_vectors: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray] | None,
    seed: int = SEED,
    pca_dims: Sequence[int | None] = (None, 64),
    algorithms: Sequence[str] = ALGORITHMS,
    restarts: int = 10,
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    estimate: bool = True,
    umap: bool = True,
    umap_components: int = 5,
    null_draws: int = 200,
) -> list[dict]:
    """All discovery rows for one (encoder, cell). Label-free quantities never see ``truth``."""
    classes = list(classes)
    truth = np.asarray(truth, dtype=object)
    index = {label: i for i, label in enumerate(classes)}
    truth_ids = np.asarray([index.get(t, -1) for t in truth], dtype=np.int64)
    valid = truth_ids >= 0
    K = len(classes)
    rows: list[dict] = []
    for pca_dim in pca_dims:
        x = prepare_features(features_raw, pca_dim=pca_dim, seed=seed)
        class_cent = class_centroids(x[valid], truth[valid], classes)
        rsa_class_value = rsa(class_cent, text_vectors)
        null = shuffled_null(class_cent, text_vectors, draws=null_draws, seed=seed)
        ablation = (dimension_ablation(features_raw[valid], truth[valid], classes, text_vectors, seed=seed)
                    if pca_dim is None else {})
        k_est = estimate_k(x, k_range=tuple(k_range), seed=seed) if estimate else None
        for algorithm in algorithms:
            settings = [("k_true", K)]
            if k_est is not None:
                settings.append(("k_hat_silhouette", k_est["k_hat_silhouette"]))
            for k_setting, k in settings:
                if k < 2 or k > len(x):
                    continue
                clustering = cluster_cell(x, k=k, algorithm=algorithm, restarts=restarts, seed=seed)
                labels = clustering.labels
                oracle = oracle_assignment(labels[valid], truth_ids[valid], k, K)
                row = {
                    "method": algorithm, "k_setting": k_setting, "k": int(k), "k_true": K,
                    "pca_dim": 0 if pca_dim is None else int(pca_dim),
                    "ami": ami(truth_ids[valid], labels[valid]),
                    "ari": ari(truth_ids[valid], labels[valid]),
                    "nmi": nmi(truth_ids[valid], labels[valid]),
                    "hungarian_accuracy_pct": 100.0 * hungarian_accuracy(truth_ids[valid], labels[valid]),
                    "silhouette": silhouette(x, labels),
                    "davies_bouldin": davies_bouldin(x, labels),
                    "rsa_class": rsa_class_value,
                    "rsa_class_null_mean": null["mean"], "rsa_class_null_p95": null["p95"],
                    "rsa_cluster": rsa(cluster_centroids_by_label(x, labels, oracle, K), text_vectors),
                    **{f"rsa_class_{key}": value for key, value in ablation.items()},
                    "n_windows": int(valid.sum()),
                    "n_clusters_nonempty": int(len(np.unique(labels))),
                }
                if k_est is not None:
                    row.update({
                        "k_hat_silhouette": k_est["k_hat_silhouette"],
                        "k_hat_davies_bouldin": k_est["k_hat_davies_bouldin"],
                        "k_abs_error_silhouette": abs(k_est["k_hat_silhouette"] - K),
                        "k_abs_error_davies_bouldin": abs(k_est["k_hat_davies_bouldin"] - K),
                    })
                if score_fn is not None and naming_features is not None and k_setting == "k_true":
                    scores = name_clusters(naming_features, labels, k, score_fn)
                    outcomes = naming_outcomes(labels[valid], truth_ids[valid], k, K, scores)
                    for key, value in outcomes.items():
                        if key.endswith("_assignment"):
                            continue
                        row[f"{key}_pct" if isinstance(value, float) else key] = (
                            100.0 * value if isinstance(value, float) else value)
                if umap and algorithm == "kmeans_pp_10" and pca_dim is None and k_setting == "k_true":
                    row["umap_silhouette"] = umap_silhouette(x, labels, n_components=umap_components, seed=seed)
                rows.append(row)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(RUNG_MODELS))
    parser.add_argument("--halo-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-seconds", type=float, nargs="+", default=[8.0])
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="the shared sealed feature cache; default <out>/feature_cache")
    parser.add_argument("--cache-read-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--feature-memory-cache-gib", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pca-dim", type=int, default=64, help="second feature setting beside raw")
    parser.add_argument("--k-range", type=int, nargs=2, default=[2, 30], metavar=("LOW", "HIGH"))
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--null-draws", type=int, default=200)
    parser.add_argument("--umap-components", type=int, default=5)
    parser.add_argument("--no-umap", action="store_true")
    parser.add_argument("--no-k-estimate", action="store_true")
    parser.add_argument("--no-naming", action="store_true")
    parser.add_argument("--cells", type=int, default=None, help="evaluate only the first N cells (smoke)")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if len(set(args.models)) != len(args.models) or any(m not in RUNG_MODELS for m in args.models):
        raise SystemExit(f"--models must be unique names from {RUNG_MODELS}")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache or args.out / "feature_cache"
    scorer = ProviderScorer(
        models=args.models, device=device, cache_dir=cache_dir, halo_checkpoint=args.halo_checkpoint,
        cache_read_dirs=args.cache_read_dirs,
        memory_cache=FeatureMemoryCache(int(args.feature_memory_cache_gib * 1024**3)),
    )
    cells = [cell for cell in evaluation_cells(args.window_seconds, scope="sealed") if not cell[3]]
    if args.cells is not None:
        cells = cells[:args.cells]
    fingerprints: dict[str, str] = {}
    rows: list[dict] = []
    started = time.perf_counter()
    for cell_index, (window_seconds, dataset, stream_id, _) in enumerate(cells, start=1):
        stream = load_eval_stream(dataset, stream_id, alignment="native", window_seconds=window_seconds,
                                  apply_quality_screen=True)
        truth = _aligned_labels(stream)
        classes = list(stream.eval_labels)
        text_vectors = _normalise(np.asarray(scorer.sbert(classes), dtype=np.float32))
        base = {"dataset": dataset, "stream": stream_id, "window_seconds": float(window_seconds),
                "source_slice_fingerprint": source_slice_fingerprint(stream)}
        for name in args.models:
            try:
                features, fingerprint = scorer.features(name, stream)
                score_fn = naming_features = None
                if not args.no_naming:
                    role = zero_shot_feature_role(name)
                    naming_features = features if role == "enrollment" else scorer.features(name, stream, role=role)[0]
                    score_fn = scorer.score_fn(name, classes, window_seconds)
            except baselines.UnsupportedEvaluationCell as exc:
                rows.append({**base, "model": name, "encoder": name, "status": "n/a", "reason": str(exc)})
                continue
            cell_rows = evaluate_cell(
                features_raw=features, naming_features=naming_features, truth=truth, classes=classes,
                text_vectors=text_vectors, score_fn=score_fn, seed=args.seed,
                pca_dims=(None, args.pca_dim), restarts=args.restarts,
                k_range=range(args.k_range[0], args.k_range[1] + 1), estimate=not args.no_k_estimate,
                umap=not args.no_umap, umap_components=args.umap_components, null_draws=args.null_draws,
            )
            for row in cell_rows:
                row.update({**base, "model": name, "encoder": name, "status": "ok",
                            "artifact_fingerprint": fingerprint,
                            "naming_route": "none" if score_fn is None else
                            ("halo_text_bridge" if name == "halo" else
                             "training_bank_conse" if zero_shot_feature_role(name) == "enrollment" else "native")})
            rows.extend(cell_rows)
            fingerprints[name] = fingerprint
        elapsed = time.perf_counter() - started
        print(f"[discovery] cells={cell_index}/{len(cells)} elapsed={elapsed / 60:.1f}m", flush=True)
        _atomic_json(args.out / "progress.json", {"completed_cells": cell_index, "total_cells": len(cells)})
    provenance = ArtifactProvenance(
        rung=Rung.DISCOVERY,
        checkpoint_fingerprint=hashlib.sha256(json.dumps(sorted(fingerprints.items())).encode()).hexdigest(),
        manifest_fingerprint=hashlib.sha256(json.dumps({
            "cells": [list(c[:3]) for c in cells], "seed": args.seed, "pca_dim": args.pca_dim,
            "k_range": args.k_range, "restarts": args.restarts}, sort_keys=True).encode()).hexdigest(),
        extra={"per_model_fingerprints": fingerprints, "subject_independent": True},
    )
    path = write_artifact(args.out, rows, provenance, argv=sys.argv, device=device,
                          halo_checkpoint=args.halo_checkpoint)
    _write_summary(args.out / "RESULTS.md", rows)
    print(f"[discovery] wrote {path} ({len(rows)} rows)")


def _write_summary(path: Path, rows: Sequence[dict]) -> None:
    keys = ("ami", "ari", "hungarian_accuracy_pct", "rsa_class", "rsa_cluster",
            "naming_model_hungarian_pct", "naming_cost_pct", "k_abs_error_silhouette")
    lines = ["# Discovery readout (retired 2026-09-23; not a rung of the plan) (means over cells; k_true, raw features, k-means++)", "",
             "| encoder | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    ok = [r for r in rows if r.get("status") == "ok" and r.get("k_setting") == "k_true"
          and r.get("pca_dim") == 0 and r.get("method") == "kmeans_pp_10"]
    for encoder in sorted({r["encoder"] for r in ok}):
        sub = [r for r in ok if r["encoder"] == encoder]
        cells = []
        for key in keys:
            values = [r[key] for r in sub if key in r and r[key] is not None and np.isfinite(r[key])]
            cells.append(f"{np.mean(values):.3f}" if values else "-")
        lines.append(f"| {encoder} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
