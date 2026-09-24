"""Rung 1 driver: the N-curve for every encoder on every sealed single-device cell, from cached
features and the uniform zero-shot score matrix. ``ncurve.run_cell`` is the pure core.

Tier 1: the released checkpoints (``--models``). Tier 2: a corpus-matched arm is scored as
``halo`` with ``--halo-checkpoint`` pointing at its checkpoint (``build_encoder`` rebuilds the
matched trunk from the checkpoint's ``encoder_arch``); name it with ``--encoder-label``.

Smoke: ``python -m evaluation.rung1_unlabeled.run --out /tmp/r2 --models halo --halo-checkpoint
<ckpt> --feature-cache <shared cache> --cells 1 --k 0 1 --pool-sizes 0 50 --n-iter 3 --n-iter-mm 10``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import baselines
from baselines.data import load_eval_stream, source_slice_fingerprint
from evaluation.controls import CONTROLS, balanced_pool_filter, disjoint_class_split
from evaluation.features import FeatureMemoryCache
from evaluation.manifests import SEED, _aligned_labels, evaluation_cells
from evaluation.provenance import ArtifactProvenance, Rung, _atomic_json, write_artifact
from evaluation.rung1_unlabeled.ncurve import (
    DEFAULT_K, DEFAULT_POOL_SIZES, nested_pool_draws, run_cell, split_scored_pool,
)
from evaluation.rung1_unlabeled.transductive import INFERENCE_DEFAULTS
from evaluation.zero_shot import ProviderScorer, zero_shot_feature_role

RUNG_MODELS = ("halo", "harnet5", "harnet10", "limubert_x", "unimts", "normwear")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(RUNG_MODELS))
    parser.add_argument("--halo-checkpoint", type=Path, default=None)
    parser.add_argument("--encoder-label", default=None,
                        help="row label for the halo slot, e.g. matched:harnet (tier 2); default halo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-seconds", type=float, nargs="+", default=[8.0])
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--cache-read-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--feature-memory-cache-gib", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--pool-sizes", nargs="+", default=[str(s) for s in DEFAULT_POOL_SIZES],
                        help="pool sizes N (integers and/or 'all'); N=0 is transduction over the "
                             "scored set alone, the null of the curve (the inductive anchor is always emitted)")
    parser.add_argument("--scored-fraction", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=30.0, help="released config: T 30")
    parser.add_argument("--n-iter", type=int, default=INFERENCE_DEFAULTS["n_iter"])
    parser.add_argument("--n-iter-mm", type=int, default=INFERENCE_DEFAULTS["n_iter_mm"])
    parser.add_argument("--lam", type=float, default=None, help="override lambda (default: N)")
    parser.add_argument("--affinity-mu", type=float, default=INFERENCE_DEFAULTS["mu"],
                        help="weight of the embedding-affinity term; 0 = the published EM-Dirichlet")
    parser.add_argument("--affinity-knn", type=int, default=INFERENCE_DEFAULTS["knn"],
                        help="neighbours per window in each encoder's embedding space")
    parser.add_argument("--assignments", nargs="+", default=["identity", "graph"])
    parser.add_argument("--control", choices=CONTROLS, default="none")
    parser.add_argument("--holdout-fraction", type=float, default=0.5, help="disjoint_classes control")
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
    pool_sizes: list[int | str] = [s if s == "all" else int(s) for s in args.pool_sizes]
    transduce_kwargs = {"n_iter": args.n_iter, "n_iter_mm": args.n_iter_mm,
                        "early_stop": args.n_iter_mm >= 100, "lam": args.lam,
                        "mu": args.affinity_mu, "knn": args.affinity_knn}
    cells = [cell for cell in evaluation_cells(args.window_seconds, scope="sealed") if not cell[3]]
    if args.cells is not None:
        cells = cells[:args.cells]
    label_of = {name: (args.encoder_label if name == "halo" and args.encoder_label else name)
                for name in args.models}
    fingerprints: dict[str, str] = {}
    rows: list[dict] = []
    started = time.perf_counter()
    for cell_index, (window_seconds, dataset, stream_id, _) in enumerate(cells, start=1):
        stream = load_eval_stream(dataset, stream_id, alignment="native", window_seconds=window_seconds,
                                  apply_quality_screen=True)
        base = {"dataset": dataset, "stream": stream_id, "window_seconds": float(window_seconds),
                "source_slice_fingerprint": source_slice_fingerprint(stream)}
        classes = list(stream.eval_labels)
        truth = _aligned_labels(stream)
        index = {label: i for i, label in enumerate(classes)}
        truth_ids = np.asarray([index.get(t, -1) for t in truth], dtype=np.int64)
        valid_rows = np.flatnonzero(truth_ids >= 0)
        if not stream.execution_identity_known or stream.execution_ids is None:
            for name in args.models:
                rows.append({**base, "model": name, "encoder": label_of[name], "status": "n/a",
                             "reason": "execution identity unavailable; cannot split scored and pool sets"})
            continue
        seed_parts = (args.seed, dataset, stream_id, float(window_seconds), args.scored_fraction)
        split = split_scored_pool(stream.execution_ids, valid_rows, fraction=args.scored_fraction,
                                  seed_parts=seed_parts)
        kept = held = None
        if args.control == "disjoint_classes":
            split, kept, held = disjoint_class_split(split, truth_ids, len(classes),
                                                     holdout_fraction=args.holdout_fraction,
                                                     seed_parts=seed_parts)
        pool_draws = nested_pool_draws(split.pool, pool_sizes, seed_parts=seed_parts)
        pool_filter = (balanced_pool_filter(truth_ids, len(classes), seed_parts=seed_parts)
                       if args.control == "balanced_pool" else None)
        for name in args.models:
            try:
                enrollment, fingerprint = scorer.features(name, stream)
                role = zero_shot_feature_role(name)
                zero_shot_features = enrollment if role == "enrollment" else scorer.features(name, stream, role=role)[0]
                scores, info = scorer.scores(name, zero_shot_features, classes, window_seconds)
            except baselines.UnsupportedEvaluationCell as exc:
                rows.append({**base, "model": name, "encoder": label_of[name], "status": "n/a", "reason": str(exc)})
                continue
            cell_rows = run_cell(
                scores_all=scores, score_kind=info["kind"], features_all=enrollment, truth_ids=truth_ids,
                classes=classes, split=split, pool_draws=pool_draws, ks=args.k,
                temperature=args.temperature, transduce_kwargs=transduce_kwargs,
                assignments=args.assignments, seed_parts=(*seed_parts, name),
                pool_filter=pool_filter, control=args.control, device=device,
            )
            for row in cell_rows:
                row.update({**base, "model": name, "encoder": label_of[name],
                            "status": row.get("status", "ok"), "artifact_fingerprint": fingerprint,
                            "zero_shot_route": info["route"],
                            "n_executions_scored": split.n_executions_scored,
                            "n_executions_pool": split.n_executions_pool,
                            **({"roster_kept": [classes[i] for i in kept],
                                "roster_held_out": [classes[i] for i in held]} if kept is not None else {})})
            rows.extend(cell_rows)
            fingerprints[label_of[name]] = fingerprint
        elapsed = time.perf_counter() - started
        print(f"[rung1] cells={cell_index}/{len(cells)} elapsed={elapsed / 60:.1f}m", flush=True)
        _atomic_json(args.out / "progress.json", {"completed_cells": cell_index, "total_cells": len(cells)})
    provenance = ArtifactProvenance(
        rung=Rung.UNLABELED,
        checkpoint_fingerprint=hashlib.sha256(json.dumps(sorted(fingerprints.items())).encode()).hexdigest(),
        manifest_fingerprint=hashlib.sha256(json.dumps({
            "cells": [list(c[:3]) for c in cells], "seed": args.seed, "scored_fraction": args.scored_fraction,
            "pool_sizes": [str(s) for s in pool_sizes], "k": args.k, "control": args.control},
            sort_keys=True).encode()).hexdigest(),
        extra={"per_model_fingerprints": fingerprints, "temperature": args.temperature,
               "transduce": transduce_kwargs, "assignments": args.assignments, "control": args.control,
               "reference_implementation": "github.com/SegoleneMartin/transductive-CLIP@master (fetched 2026-09-23)",
               "subject_independent": True},
    )
    path = write_artifact(args.out, rows, provenance, argv=sys.argv, device=device,
                          halo_checkpoint=args.halo_checkpoint)
    _write_summary(args.out / "RESULTS.md", rows)
    print(f"[rung1] wrote {path} ({len(rows)} rows)")


def _dataset_balanced(rows: Sequence[dict]) -> float | None:
    """Mean over datasets of the per-dataset mean: the sealed table's aggregation."""
    by_dataset: dict[str, list[float]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(float(row["f1_macro"]))
    return float(np.mean([np.mean(v) for v in by_dataset.values()])) if by_dataset else None


def _write_summary(path: Path, rows: Sequence[dict]) -> None:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("scope") == "scored"
          and r.get("assignment", "identity") == "identity"]
    lines = ["# Rung 1 — unlabelled adaptation: dataset-balanced macro-F1 on the scored set vs pool size N",
             "",
             "`inductive` is the anchor (zero-shot arg-max at k = 0, embedding prototypes at k > 0) and "
             "reads a different feature space; the curve's null is `N=0`, the same transductive method "
             "over the scored set alone.", ""]
    for k in sorted({r["k"] for r in ok}):
        sub = [r for r in ok if r["k"] == k]
        labels = sorted({r["N_label"] for r in sub if r["method"] != "inductive"},
                        key=lambda s: (s == "all", int(s) if s != "all" else 0))
        header = ["inductive"] + [f"N={label}" for label in labels]
        lines += [f"## k = {k}", "", "| encoder | " + " | ".join(header) + " |",
                  "|---|" + "---:|" * len(header)]
        for encoder in sorted({r["encoder"] for r in sub}):
            mine = [r for r in sub if r["encoder"] == encoder]
            groups = [[r for r in mine if r["method"] == "inductive"]] \
                + [[r for r in mine if r.get("N_label") == label] for label in labels]
            values = [_dataset_balanced(group) for group in groups]
            lines.append(f"| {encoder} | " + " | ".join("-" if v is None else f"{v:.1f}" for v in values) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
