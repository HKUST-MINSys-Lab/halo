"""Rung 3 driver: the fine-tuning ladder for every provider on every sealed single-device cell.

The scored set and the k-per-class support draws are rung 1's (same seed parts), so a rung-3 row
and a rung-1 row for the same (cell, k) are on identical windows and the enrollment-vs-fine-tuning
crossover is a like-for-like comparison. Raw-window treatments re-instantiate an encoder per
(cell, k, treatment) and train it for a fixed budget; that is the expensive part of the paper and
is why ``--cells`` / ``--k`` / ``--steps`` exist for smoke runs.

Smoke: ``python -m evaluation.rung3_finetune.run --out /tmp/r4 --models halo --halo-checkpoint
<ckpt> --feature-cache <shared cache> --cells 1 --k 1 --steps 5 --treatments enrollment_frozen
linear_probe lora``.
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
from evaluation.features import FeatureMemoryCache
from evaluation.manifests import SEED, _aligned_labels, evaluation_cells
from evaluation.provenance import ArtifactProvenance, Rung, _atomic_json, write_artifact
from evaluation.rung1_unlabeled.ncurve import split_scored_pool
from evaluation.rung3_finetune.finetune import (
    CACHED_FEATURE_TREATMENTS, TREATMENTS, FineTuneConfig, run_cell,
)
from evaluation.zero_shot import ProviderScorer

RUNG_MODELS = ("halo", "harnet5", "harnet10", "limubert_x", "unimts", "normwear")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(RUNG_MODELS))
    parser.add_argument("--halo-checkpoint", type=Path, default=None)
    parser.add_argument("--encoder-label", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-seconds", type=float, nargs="+", default=[8.0])
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--cache-read-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--feature-memory-cache-gib", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scored-fraction", type=float, default=0.2, help="must match rung 1")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--treatments", nargs="+", default=list(TREATMENTS), choices=TREATMENTS)
    parser.add_argument("--steps", type=int, default=FineTuneConfig.steps)
    parser.add_argument("--lr", type=float, default=FineTuneConfig.lr)
    parser.add_argument("--encoder-lr-scale", type=float, default=FineTuneConfig.encoder_lr_scale)
    parser.add_argument("--batch-size", type=int, default=FineTuneConfig.batch_size)
    parser.add_argument("--lora-rank", type=int, default=FineTuneConfig.lora_rank)
    parser.add_argument("--lora-alpha", type=float, default=FineTuneConfig.lora_alpha)
    parser.add_argument("--cells", type=int, default=None, help="evaluate only the first N cells (smoke)")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if len(set(args.models)) != len(args.models) or any(m not in RUNG_MODELS for m in args.models):
        raise SystemExit(f"--models must be unique names from {RUNG_MODELS}")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache or args.out / "feature_cache"
    cfg = FineTuneConfig(steps=args.steps, lr=args.lr, encoder_lr_scale=args.encoder_lr_scale,
                         batch_size=args.batch_size, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                         seed=args.seed)
    needs_cached = bool(set(args.treatments) & CACHED_FEATURE_TREATMENTS)
    scorer = ProviderScorer(
        models=args.models, device=device, cache_dir=cache_dir, halo_checkpoint=args.halo_checkpoint,
        cache_read_dirs=args.cache_read_dirs,
        memory_cache=FeatureMemoryCache(int(args.feature_memory_cache_gib * 1024**3)),
    ) if needs_cached or "halo" in args.models else None
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
        # Identical to rung 1's split for this cell: same seed parts, same fraction.
        seed_parts = (args.seed, dataset, stream_id, float(window_seconds), args.scored_fraction)
        split = split_scored_pool(stream.execution_ids, valid_rows, fraction=args.scored_fraction,
                                  seed_parts=seed_parts)
        for name in args.models:
            features = fingerprint = None
            treatments = list(args.treatments)
            if needs_cached:
                try:
                    features, fingerprint = scorer.features(name, stream)
                except baselines.UnsupportedEvaluationCell as exc:
                    # Only the cached-feature treatments depend on these features; the raw-window
                    # treatments build their own trunk and are still attempted.
                    for treatment in (t for t in treatments if t in CACHED_FEATURE_TREATMENTS):
                        rows.append({**base, "model": name, "encoder": label_of[name], "method": treatment,
                                     "status": "n/a", "reason": f"cached features unavailable: {exc}"})
                    treatments = [t for t in treatments if t not in CACHED_FEATURE_TREATMENTS]
                    if not treatments:
                        continue
            cell_rows = run_cell(
                name=name, stream=stream, features=features, truth_ids=truth_ids, classes=classes,
                split=split, ks=args.k, treatments=treatments, cfg=cfg, device=device,
                halo_checkpoint=args.halo_checkpoint, seed_parts=(*seed_parts, name),
            )
            for row in cell_rows:
                row.update({**base, "model": name, "encoder": label_of[name],
                            "artifact_fingerprint": fingerprint,
                            "n_executions_scored": split.n_executions_scored,
                            "n_executions_pool": split.n_executions_pool})
            rows.extend(cell_rows)
            if fingerprint:
                fingerprints[label_of[name]] = fingerprint
        elapsed = time.perf_counter() - started
        print(f"[rung3] cells={cell_index}/{len(cells)} elapsed={elapsed / 60:.1f}m", flush=True)
        _atomic_json(args.out / "progress.json", {"completed_cells": cell_index, "total_cells": len(cells)})
    provenance = ArtifactProvenance(
        rung=Rung.FINETUNE,
        checkpoint_fingerprint=hashlib.sha256(json.dumps(sorted(fingerprints.items())).encode()).hexdigest()
        if fingerprints else hashlib.sha256(str(args.halo_checkpoint).encode()).hexdigest(),
        manifest_fingerprint=hashlib.sha256(json.dumps({
            "cells": [list(c[:3]) for c in cells], "seed": args.seed, "scored_fraction": args.scored_fraction,
            "k": args.k, "treatments": args.treatments}, sort_keys=True).encode()).hexdigest(),
        extra={"per_model_fingerprints": fingerprints, "config": cfg.as_dict(),
               # Same split as rung 1: execution-disjoint, not subject-disjoint.
               "subject_independent": False, "execution_disjoint_scored_pool": True},
    )
    path = write_artifact(args.out, rows, provenance, argv=sys.argv, device=device,
                          halo_checkpoint=args.halo_checkpoint)
    _write_summary(args.out / "RESULTS.md", rows)
    print(f"[rung3] wrote {path} ({len(rows)} rows)")


def _write_summary(path: Path, rows: Sequence[dict]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    lines = ["# Rung 3 — fine-tuning ladder: mean macro-F1 on the scored set", ""]
    for treatment in TREATMENTS:
        sub = [r for r in ok if r["method"] == treatment]
        if not sub:
            continue
        ks = sorted({r["k"] for r in sub})
        lines += [f"## {treatment}", "", "| encoder | " + " | ".join(f"k={k}" for k in ks) + " |",
                  "|---|" + "---:|" * len(ks)]
        for encoder in sorted({r["encoder"] for r in sub}):
            cells = []
            for k in ks:
                values = [r["f1_macro"] for r in sub if r["encoder"] == encoder and r["k"] == k]
                cells.append(f"{np.mean(values):.1f}" if values else "-")
            lines.append(f"| {encoder} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
