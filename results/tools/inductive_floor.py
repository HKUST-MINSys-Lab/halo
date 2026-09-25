"""Inductive-floor check (runbook T1-E): a candidate encoder's cosine 1-NN k-curve against a reference.

Usage:
    python results/tools/inductive_floor.py \
        --reference results/artifacts/halo_evidence_gated_v4_step40k_sealed_v5_20260920 \
        --candidate runs/evaluations/sealed_1nn_halo_rung1_trained_20260925 \
        --out results/artifacts/<name>/INDUCTIVE_FLOOR.md

Both are sealed-evaluator result directories (``results.json`` or ``results.json.gz``). Only cells
present in both, at the same window, are compared. Dataset-balanced = mean over datasets of the mean
over that dataset's cells.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict]:
    path = Path(path)
    for name in ("results.json.gz", "results.json"):
        f = path / name
        if f.is_file():
            blob = json.load(gzip.open(f) if name.endswith(".gz") else f.open())
            rows = blob["rows"] if isinstance(blob, dict) else blob
            return [r for r in rows if r.get("status", "ok") == "ok"]
    raise FileNotFoundError(f"no results.json(.gz) in {path}")


def one_nn(rows: list[dict], window: float) -> dict:
    return {(r["dataset"], r["stream"], int(r["k"])): float(r["f1_macro"]) for r in rows
            if r.get("readout") == "1nn" and float(r.get("window_seconds", 0)) == window}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ref, cand = one_nn(load(args.reference), args.window_seconds), one_nn(load(args.candidate), args.window_seconds)
    shared = sorted(set(ref) & set(cand))
    ks = sorted({k for _, _, k in shared})
    lines = ["# Inductive floor: cosine 1-NN k-curve, candidate vs reference", "",
             f"reference: `{args.reference}`  ", f"candidate: `{args.candidate}`  ",
             f"window {args.window_seconds:g} s; {len({(d, s) for d, s, _ in shared})} shared cells.", "",
             "| k | reference | candidate | Δ | cells below reference by > 5 |", "|---:|---:|---:|---:|---:|"]
    for k in ks:
        per = defaultdict(lambda: ([], []))
        worse = 0
        for d, s, kk in shared:
            if kk == k:
                per[d][0].append(ref[(d, s, k)])
                per[d][1].append(cand[(d, s, k)])
                worse += cand[(d, s, k)] < ref[(d, s, k)] - 5
        r = float(np.mean([np.mean(v[0]) for v in per.values()]))
        c = float(np.mean([np.mean(v[1]) for v in per.values()]))
        lines.append(f"| {k} | {r:.1f} | {c:.1f} | {c - r:+.1f} | {worse} |")
    lines += ["", "## Per cell", "", "| dataset / stream | k | reference | candidate |", "|---|---:|---:|---:|"]
    lines += [f"| {d} / {s} | {k} | {ref[(d, s, k)]:.1f} | {cand[(d, s, k)]:.1f} |" for d, s, k in shared]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:7 + len(ks)]))


if __name__ == "__main__":
    main()
