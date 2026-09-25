"""Tier-3 (rung 3) report: fine-tuning ladder rows -> dataset-balanced tables with support-draw spread,
per-dataset tables, and trainable-parameter / wall-time context.

Usage:
    python results/tools/rung3_report.py --run runs/evaluations/rung3_probe_20260925 \
        [--run runs/evaluations/rung3_phaseA_20260925] --out results/artifacts/<name>/RESULTS.md

Macro-F1 on rung 1's fixed scored set. Each (cell, k) is fitted on ``support_draws`` independent
support sets; a row's value is the mean over draws, and ``±`` is the mean over datasets of the
within-cell standard deviation across draws.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ("enrollment_frozen", "linear_probe", "small_classifier", "lora", "full_finetune", "scratch_specialist")


def load(paths) -> list[dict]:
    rows = []
    for path in paths:
        rows += json.loads((Path(path) / "results.json").read_text())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.run)
    ok = [r for r in rows if r.get("status") == "ok"]
    na = [r for r in rows if r.get("status") != "ok"]
    ks = sorted({r["k"] for r in ok})
    cell_draws = defaultdict(list)                     # (model, method, k, dataset, stream) -> [f1]
    params = defaultdict(list)
    for r in ok:
        cell_draws[(r["encoder"], r["method"], r["k"], r["dataset"], r["stream"])].append(float(r["f1_macro"]))
        if r.get("trainable_params") is not None:
            params[(r["encoder"], r["method"])].append(int(r["trainable_params"]))
    by_dataset = defaultdict(lambda: defaultdict(list))
    spread = defaultdict(lambda: defaultdict(list))
    for (enc, method, k, dataset, _), vals in cell_draws.items():
        by_dataset[(enc, method, k)][dataset].append(float(np.mean(vals)))
        spread[(enc, method, k)][dataset].append(float(np.std(vals)))
    encoders = sorted({key[0] for key in by_dataset})
    methods = [m for m in ORDER if any(key[1] == m for key in by_dataset)]
    lines = ["# Tier 3 (rung 3): adaptation with k labelled windows per class", "",
             "Dataset-balanced macro-F1 on the fixed scored set (mean over support draws; ± = mean "
             "within-cell std across draws). Heads sit on each model's own features.", ""]
    for method in methods:
        lines += [f"## {method}", "", "| encoder | " + " | ".join(f"k={k}" for k in ks) + " | trainable params |",
                  "|---|" + "---:|" * (len(ks) + 1)]
        for enc in encoders:
            cells = []
            for k in ks:
                d = by_dataset.get((enc, method, k))
                if not d:
                    cells.append("-")
                    continue
                mean = float(np.mean([np.mean(v) for v in d.values()]))
                sd = float(np.mean([np.mean(v) for v in spread[(enc, method, k)].values()]))
                cells.append(f"{mean:.1f} ± {sd:.1f}")
            p = params.get((enc, method))
            lines.append(f"| {enc} | " + " | ".join(cells) + f" | {int(np.median(p)) if p else '-'} |")
        lines.append("")
    datasets = sorted({key[3] for key in cell_draws})
    lines += ["## Per dataset (mean over cells and draws)", ""]
    for method in methods:
        lines += [f"### {method}", "", "| encoder | k | " + " | ".join(datasets) + " |",
                  "|---|---:|" + "---:|" * len(datasets)]
        for enc in encoders:
            for k in ks:
                d = by_dataset.get((enc, method, k), {})
                if d:
                    lines.append(f"| {enc} | {k} | " + " | ".join(
                        f"{np.mean(d[x]):.1f}" if x in d else "-" for x in datasets) + " |")
        lines.append("")
    fits = defaultdict(lambda: defaultdict(list))
    for r in ok:
        fit = fits[(r["encoder"], r["method"])]
        fit["cells"].append((r["dataset"], r["stream"]))
        fit["draws"].append(r.get("support_draw", 0))
        for key in ("fit_seconds", "final_loss"):
            if r.get(key) is not None:
                fit[key].append(float(r[key]))
    lines += ["## Fit diagnostics", "",
              "Coverage and cost per (encoder, method). A final loss far above 0 means the head did not "
              "fit its supports within the fixed budget.", "",
              "| encoder | method | cells | draws | rows | median fit s | mean final loss |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for (enc, method), fit in sorted(fits.items(), key=lambda kv: (kv[0][0], ORDER.index(kv[0][1]))):
        med = f"{np.median(fit['fit_seconds']):.2f}" if fit["fit_seconds"] else "-"
        loss = f"{np.mean(fit['final_loss']):.3f}" if fit["final_loss"] else "-"
        lines.append(f"| {enc} | {method} | {len(set(fit['cells']))} | {len(set(fit['draws']))} | "
                     f"{len(fit['cells'])} | {med} | {loss} |")
    lines.append("")
    if na:
        reasons = defaultdict(int)
        for r in na:
            reasons[(r.get("encoder") or r.get("model"), r.get("method"), r.get("reason", "")[:80])] += 1
        lines += ["## Not applicable", "", "| encoder | method | reason | rows |", "|---|---|---|---:|"]
        lines += [f"| {e} | {m} | {why} | {n} |" for (e, m, why), n in sorted(reasons.items(), key=str)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
