"""Online (predict-then-update) evaluation of the v5 memory reader on the sealed 8 s cells.

Per cell and arrival order, the scored set S (rung 1's split) is replayed as a stream. Settings:
``unlabelled`` (memory starts empty; every window inserted unverified after prediction — tier 1) and
``enrolled`` (k verified windows per class from the execution-disjoint pool first — tier 2; the same
support draw rungs 1/3 use for draw 0). Every prediction reports the learned reader, its fixed-vote
floor and the no-memory semantic path from one forward pass (see ``stream.py``).

Example:
    python -m evaluation.online_memory.run --out runs/evaluations/v5_online \
        --v5-checkpoint runs/support-classifier/v5/best_internal.pt \
        --feature-checkpoint runs/support-classifier/halo_t6_40k_20260920/last.pt \
        --cache-read-dirs cache/evaluations/feature_cache_halo_t6_40k_20260920_final40k
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from baselines.data import load_eval_stream, source_slice_fingerprint
from evaluation.features import FeatureMemoryCache, _file_hash
from evaluation.manifests import SEED, _aligned_labels, evaluation_cells
from evaluation.online_memory.stream import run_stream, summarise
from evaluation.provenance import _atomic_json, _run_provenance
from evaluation.rung1_unlabeled.ncurve import shared_support_set, split_scored_pool
from evaluation.zero_shot import ProviderScorer


def _encoders_identical(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--v5-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-checkpoint", type=Path, default=None,
                        help="checkpoint whose cached features to reuse; must have v5's exact encoder weights")
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--cache-read-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scored-fraction", type=float, default=0.2)
    parser.add_argument("--orders", type=int, default=3, help="independent arrival orders per cell")
    parser.add_argument("--settings", nargs="+", default=["unlabelled", "enrolled"],
                        choices=("unlabelled", "enrolled"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 4, 16], help="enrolled setting only")
    parser.add_argument("--cells", type=int, default=None)
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    from model.support.factory import build_classifier_from_blob
    from model.support.memory_classifier import ARCHITECTURE_VERSION
    from training.support_classifier.train import label_text_matrix
    from evaluation.rung2_frozen.sealed_eval import halo_acquisition_rows

    blob = torch.load(args.v5_checkpoint, map_location="cpu", weights_only=False)
    if blob.get("architecture_version") != ARCHITECTURE_VERSION:
        raise SystemExit("--v5-checkpoint is not a v5 memory-reader checkpoint")
    reader, _ = build_classifier_from_blob(blob, device=device)
    reader.eval()
    feature_checkpoint = args.feature_checkpoint or args.v5_checkpoint
    if args.feature_checkpoint is not None:
        other = torch.load(args.feature_checkpoint, map_location="cpu", weights_only=False)
        if not _encoders_identical(blob["encoder"], other["encoder"]):
            raise SystemExit("--feature-checkpoint's encoder differs from v5's; its features would be stale")
    scorer = ProviderScorer(models=["halo"], device=device, cache_dir=args.feature_cache or args.out / "feature_cache",
                            halo_checkpoint=feature_checkpoint, cache_read_dirs=args.cache_read_dirs,
                            memory_cache=FeatureMemoryCache(4 * 1024 ** 3))
    encoder = scorer.halo_state[0]
    cells = [c for c in evaluation_cells([args.window_seconds], scope="sealed") if not c[3]]
    if args.cells is not None:
        cells = cells[:args.cells]
    rows: list[dict] = []
    started = time.perf_counter()
    for index, (window_seconds, dataset, stream_id, _) in enumerate(cells, start=1):
        stream = load_eval_stream(dataset, stream_id, alignment="native", window_seconds=window_seconds,
                                  apply_quality_screen=True)
        base = {"dataset": dataset, "stream": stream_id, "window_seconds": float(window_seconds),
                "source_slice_fingerprint": source_slice_fingerprint(stream)}
        if not stream.execution_identity_known or stream.execution_ids is None:
            rows.append({**base, "status": "n/a", "reason": "execution identity unavailable"})
            continue
        roster = tuple(stream.eval_labels)
        truth = _aligned_labels(stream)
        lookup = {label: i for i, label in enumerate(roster)}
        truth_ids = np.asarray([lookup.get(t, -1) for t in truth], dtype=np.int64)
        valid = np.flatnonzero(truth_ids >= 0)
        seed_parts = (args.seed, dataset, stream_id, float(window_seconds), args.scored_fraction)
        split = split_scored_pool(stream.execution_ids, valid, fraction=args.scored_fraction, seed_parts=seed_parts)
        motion, fingerprint = scorer.features("halo", stream)
        acquisition = halo_acquisition_rows(stream, encoder, device)
        text = label_text_matrix(list(roster), device)
        executions = np.asarray(stream.execution_ids, dtype=object)
        plans = [("unlabelled", 0, None)] if "unlabelled" in args.settings else []
        if "enrolled" in args.settings:
            for k in args.k:
                support = shared_support_set(split.pool, truth_ids, k, len(roster),
                                             seed_parts=(*seed_parts, "halo", "k", k))
                plans.append(("enrolled", k, None if support is None else support[0]))
        for setting, k, enrolled in plans:
            if setting == "enrolled" and enrolled is None:
                rows.append({**base, "setting": setting, "k": k, "status": "n/a",
                             "reason": f"pool lacks {k} windows for every class"})
                continue
            for order in range(args.orders):
                result = run_stream(reader, motion=motion, acquisition=acquisition, truth_ids=truth_ids,
                                    execution_ids=executions, roster=roster, candidate_text=text,
                                    scored_rows=split.scored, seed=int(args.seed) + order, device=device,
                                    setting=setting, enrolled_rows=enrolled,
                                    model_version=_file_hash(args.v5_checkpoint))
                rows.append({**base, "setting": setting, "k": k, "order": order, "status": "ok",
                             "feature_fingerprint": fingerprint, **summarise(result, roster)})
        print(f"[online] cells={index}/{len(cells)} elapsed={(time.perf_counter() - started) / 60:.1f}m", flush=True)
        _atomic_json(args.out / "results_partial.json", rows)
    _atomic_json(args.out / "results.json", rows)
    _atomic_json(args.out / "run_provenance.json", {
        "argv": sys.argv, "v5_checkpoint": str(args.v5_checkpoint), "v5_sha256": _file_hash(args.v5_checkpoint),
        "feature_checkpoint": str(feature_checkpoint),
        "provenance": _run_provenance(sys.argv, device=device, halo_checkpoint=feature_checkpoint),
        "protocol": "online-memory-v1", "fingerprint": hashlib.sha256(json.dumps(
            [list(c[:3]) for c in cells]).encode()).hexdigest()})
    _write_summary(args.out / "RESULTS.md", rows)
    print(f"[online] wrote {args.out}/results.json ({len(rows)} rows)")


def _write_summary(path: Path, rows: list[dict]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    lines = ["# v5 online memory reader — dataset-balanced macro-F1 on the scored stream", "",
             "Mean over arrival orders and cells within a dataset, then over datasets.", "",
             "| setting | k | reader | fixed vote (floor) | no memory | corrections | new errors |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for setting, k in sorted({(r["setting"], r["k"]) for r in ok}):
        sub = [r for r in ok if r["setting"] == setting and r["k"] == k]
        def bal(key):
            by = {}
            for r in sub:
                by.setdefault(r["dataset"], []).append(r[key])
            return float(np.mean([np.mean(v) for v in by.values()]))
        lines.append(f"| {setting} | {k} | {bal('reader_f1_macro'):.1f} | {bal('fixed_f1_macro'):.1f} | "
                     f"{bal('semantic_f1_macro'):.1f} | {sum(r['corrections'] for r in sub)} | "
                     f"{sum(r['new_errors'] for r in sub)} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
