"""Tier-1 (rung 1) report: per-cell rows -> per-dataset and dataset-balanced tables, controls,
calibration and neighbour diagnostics, as one Markdown file.

Usage:
    python results/tools/rung1_report.py --main runs/evaluations/rung1_tier1_20260925 \
        --control mu0=runs/evaluations/rung1_tier1_mu0_20260925 \
        --control balanced=... --control disjoint=... \
        --variant runs/evaluations/rung1_halo_trained_20260925 \
        --out results/artifacts/rung1_tier1_20260925/RESULTS.md

Every number is macro-F1 on the fixed scored set (20 % of executions), identity cluster assignment
unless stated. "Dataset-balanced" = mean over datasets of the per-dataset mean over cells.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_ORDER = ("0", "50", "100", "500", "2000", "all")


def load(path: Path) -> list[dict]:
    rows = json.loads((Path(path) / "results.json").read_text())
    return [r for r in rows if r.get("status", "ok") == "ok"]


def _n_key(r):
    return r.get("N_label") if r.get("method") != "inductive" else "inductive"


def curve_table(rows: list[dict], *, assignment: str = "identity") -> dict:
    """{encoder: {dataset: {N: mean over cells}}} for scored-set k=0 rows."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r.get("scope") != "scored" or r.get("k") != 0:
            continue
        if r.get("method") != "inductive" and r.get("assignment", "identity") != assignment:
            continue
        out[r["encoder"]][r["dataset"]][_n_key(r)].append(float(r["f1_macro"]))
    return {e: {d: {n: float(np.mean(v)) for n, v in ns.items()} for d, ns in ds.items()} for e, ds in out.items()}


def balanced(table: dict, encoder: str, n: str) -> float | None:
    vals = [ns[n] for ns in table.get(encoder, {}).values() if n in ns]
    return float(np.mean(vals)) if vals else None


def fmt(x, digits=1):
    return "-" if x is None else f"{x:.{digits}f}"


def aggregate_section(title: str, rows: list[dict], assignment: str = "identity") -> list[str]:
    table = curve_table(rows, assignment=assignment)
    cols = ["inductive"] + [n for n in N_ORDER if any(n in ns for e in table.values() for ns in e.values())]
    lines = [f"### {title}", "", "| encoder | " + " | ".join(f"N={c}" if c != "inductive" else "anchor" for c in cols)
             + " | Δ all−0 |", "|---|" + "---:|" * (len(cols) + 1)]
    for encoder in sorted(table):
        vals = [balanced(table, encoder, c) for c in cols]
        a, z = balanced(table, encoder, "all"), balanced(table, encoder, "0")
        delta = None if a is None or z is None else a - z
        lines.append(f"| {encoder} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(delta)} |")
    return lines + [""]


def per_dataset_section(rows: list[dict]) -> list[str]:
    table = curve_table(rows)
    datasets = sorted({d for e in table.values() for d in e})
    lines = ["### Per dataset (anchor → N=0 → N=all)", "",
             "| encoder | " + " | ".join(datasets) + " |", "|---|" + "---:|" * len(datasets)]
    for encoder in sorted(table):
        cells = []
        for d in datasets:
            ns = table[encoder].get(d, {})
            cells.append(f"{fmt(ns.get('inductive'))} → {fmt(ns.get('0'))} → {fmt(ns.get('all'))}")
        lines.append(f"| {encoder} | " + " | ".join(cells) + " |")
    return lines + [""]


def per_cell_section(rows: list[dict]) -> list[str]:
    cells = defaultdict(dict)
    for r in rows:
        if r.get("scope") == "scored" and r.get("k") == 0 and r.get("assignment", "identity") == "identity":
            cells[(r["dataset"], r["stream"])][(r["encoder"], _n_key(r))] = r["f1_macro"]
    encoders = sorted({e for c in cells.values() for e, _ in c})
    lines = ["### Per cell: N=0 → N=all (identity assignment)", "",
             "| dataset / stream | " + " | ".join(encoders) + " |", "|---|" + "---:|" * len(encoders)]
    for (d, s), vals in sorted(cells.items()):
        lines.append(f"| {d} / {s} | " + " | ".join(
            f"{fmt(vals.get((e, '0')))} → {fmt(vals.get((e, 'all')))}" for e in encoders) + " |")
    return lines + [""]


def diagnostics_section(rows: list[dict]) -> list[str]:
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("method") == "transductive_clip_v1" and r.get("assignment") == "identity" and r.get("N_label") == "all":
            for key in ("neighbour_purity", "neighbour_same_execution", "neighbour_purity_other_execution",
                        "collapsed_components", "temperature"):
                if r.get(key) is not None:
                    acc[r["encoder"]][key].append(float(r[key]))
    lines = ["### Diagnostics at N=all (mean over cells)", "",
             "| encoder | T | neighbour purity | same-execution share | purity, other executions | collapsed components |",
             "|---|---:|---:|---:|---:|---:|"]
    for e in sorted(acc):
        m = {k: (float(np.mean(v)) if v else None) for k, v in acc[e].items()}
        lines.append(f"| {e} | {fmt(m.get('temperature'), 2)} | {fmt(m.get('neighbour_purity'), 2)} | "
                     f"{fmt(m.get('neighbour_same_execution'), 2)} | {fmt(m.get('neighbour_purity_other_execution'), 2)} | "
                     f"{fmt(m.get('collapsed_components'), 2)} |")
    return lines + [""]


def harm_section(rows: list[dict], threshold: float = 5.0) -> list[str]:
    """How often transduction helps or hurts a cell by more than ``threshold`` points."""
    anchor = {(r["encoder"], r["dataset"], r["stream"]): r["f1_macro"] for r in rows
              if r.get("method") == "inductive" and r.get("scope") == "scored" and r.get("k") == 0}
    zero = {(r["encoder"], r["dataset"], r["stream"]): r["f1_macro"] for r in rows
            if r.get("method") == "transductive_clip_v1" and r.get("assignment") == "identity" and r.get("N_label") == "0"}
    counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in rows:
        if r.get("method") != "transductive_clip_v1" or r.get("assignment") != "identity" or r.get("N_label") != "all":
            continue
        key = (r["encoder"], r["dataset"], r["stream"])
        c = counts[r["encoder"]]
        c[0] += 1
        c[1] += r["f1_macro"] - anchor[key] > threshold
        c[2] += r["f1_macro"] - anchor[key] < -threshold
        if key in zero:
            c[3] += r["f1_macro"] - zero[key] > threshold
            c[4] += r["f1_macro"] - zero[key] < -threshold
    lines = [f"### Cells helped / hurt by more than {threshold:g} points at N=all", "",
             "| encoder | cells | vs anchor: helped | hurt | vs N=0: helped | hurt |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {e} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} |" for e, c in sorted(counts.items())]
    return lines + [""]


def calibration_section(main: Path) -> list[str]:
    files = sorted(Path(main).glob("temperature_calibration_w*.json"))
    if not files:
        return []
    blob = json.loads(files[0].read_text())["models"]
    lines = ["### Per-encoder temperature (held-out training-source windows)", "",
             "| encoder | route | T | NLL (T=30 → fitted) | ECE (T=30 → fitted) | held-out top-1 acc | windows |",
             "|---|---|---:|---|---|---:|---:|"]
    for name, info in blob.items():
        lines.append(f"| {name} | {info['route']} | {info['temperature']:.3g} | {info['nll_at_30']:.2f} → {info['nll']:.2f} | "
                     f"{info['ece_at_30']:.3f} → {info['ece']:.3f} | {100 * info['accuracy']:.1f} | {info['n_windows']} |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--control", action="append", default=[], help="name=path")
    parser.add_argument("--variant", action="append", default=[], type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.main)
    for path in args.variant:
        rows += [r for r in load(path)]
    lines = ["# Tier 1 (rung 1): unlabelled adaptation — EM-Dirichlet + affinity, calibrated temperatures", "",
             "Macro-F1 on the fixed scored set (20 % of executions, execution-disjoint from the pool); "
             "k = 0 throughout. `anchor` = per-window zero-shot arg-max (equals each model's published sealed "
             "k = 0 row). The curve's null is N=0 (the same transductive method over the scored set alone).", ""]
    lines += calibration_section(args.main)
    lines += aggregate_section("Dataset-balanced macro-F1 vs unlabelled pool size N", rows)
    lines += aggregate_section("Same, graph cluster-to-class assignment", rows, assignment="graph")
    for spec in args.control:
        name, _, path = spec.partition("=")
        lines += aggregate_section(f"Control: {name}", load(Path(path)))
    lines += harm_section(rows)
    lines += per_dataset_section(rows)
    lines += per_cell_section(rows)
    lines += diagnostics_section(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
