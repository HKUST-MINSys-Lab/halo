"""Tables and figures for v5-protocol HALO arms against the released-baseline artifacts.

Every comparison is restricted to cells whose episode-manifest fingerprint is identical in the
HALO artifact and the baseline artifact; the script prints how many cells matched and refuses to
join a cell whose manifest differs.

Aggregation rules (fixed, shared by every arm):
  * sealed: dataset-balanced macro F1 = mean over single-device streams within a dataset, then
    mean over the six sealed datasets;
  * scenarios: mean over primary scenario cells (``condition == "scenario"``), whole-query split,
    excluding same-subject variants; matched controls are used only for paired costs.

Usage:
  python results/tools/v5_report.py --arm "label=path/to/artifact_or_run_dir" ... \
      [--figures OUT_DIR] [--markdown OUT.md]
An arm path is a directory containing ``results.json.gz`` (sealed or scenarios; detected from rows).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "results/artifacts"
BASE_SEALED = ART / "baselines_sealed_v5_20260918/results.json.gz"
BASE_SCEN = ART / "scenarios_baselines_v5_20260918/results.json.gz"
KS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
SCEN_KS = (0, 1, 4, 8, 32)
WINDOWS = (4.0, 8.0, 16.0)
BASELINES = {"harnet5": "HARNet-5", "harnet10": "HARNet-10", "limubert_x": "LiMU-BERT-X",
             "unimts": "UniMTS", "normwear": "NormWear"}
SCENARIOS = {"s1_partial_coverage": "partial coverage", "s2_cross_placement": "cross placement",
             "s3_cross_dataset": "cross dataset", "s4_missing_modality": "missing modality",
             "s5_rate_mismatch": "rate mismatch", "s6_new_domain": "new domain (MM-Fit)",
             "s7_device_set": "device set"}


def load(path: Path) -> list[dict]:
    path = Path(path)
    if path.is_dir():
        path = path / "results.json.gz" if (path / "results.json.gz").exists() else path / "results.json"
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)
    return payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload


def is_scenario_payload(rows: list[dict]) -> bool:
    return any(row.get("scenario") for row in rows)


# ----------------------------------------------------------------------------- manifests
def sealed_cell(row):
    return (float(row["window_seconds"]), int(row["k"]), row["dataset"], row.get("stream"),
            row.get("multi_device_mode"))


def scen_cell(row):
    return (row.get("scenario"), row.get("variant"), int(row.get("k", -1)),
            row.get("coverage_split", "all"), row.get("subject_relation"),
            float(row.get("window_seconds", 0)))


def identical_cells(a: list[dict], b: list[dict], cell_fn) -> tuple[set, set]:
    def index(rows):
        out = {}
        for row in rows:
            if row.get("status") == "ok" and row.get("manifest") and row.get("k") is not None:
                out.setdefault(cell_fn(row), row["manifest"])
        return out
    ia, ib = index(a), index(b)
    shared = set(ia) & set(ib)
    same = {cell for cell in shared if ia[cell] == ib[cell]}
    return same, shared - same


# ----------------------------------------------------------------------------- sealed
def sealed_value(rows, model, readout, window, k, cells, dataset=None):
    per = collections.defaultdict(list)
    for row in rows:
        if (row.get("status") == "ok" and row.get("model") == model and row.get("readout") == readout
                and float(row["window_seconds"]) == window and int(row["k"]) == k
                and row.get("multi_device_mode") == "single-device"
                and (cells is None or sealed_cell(row) in cells)
                and row.get("f1_macro") is not None
                and (dataset is None or row["dataset"] == dataset)):
            per[row["dataset"]].append(row["f1_macro"])
    if not per:
        return None, 0
    return st.mean(st.mean(v) for v in per.values()), len(per)


def halo_readouts(rows):
    order = ["halo-classifier", "halo-classifier-residual-off",
             "halo-classifier-label-meaning-only", "halo-classifier-unmodified-support-vote",
             "halo-classifier-contextual-support-vote", "1nn", "training-bank-1nn-conse"]
    present = {row.get("readout") for row in rows if row.get("model") == "halo"}
    return [r for r in order if r in present]


def sealed_tables(arms, base, window, lines):
    ks_line = " | ".join(f"k={k}" for k in KS)
    lines.append(f"\n### Sealed, {window:g} s, dataset-balanced macro F1 (single-device, identical manifests)\n")
    lines.append(f"| row | {ks_line} |")
    lines.append("|---" * (len(KS) + 1) + "|")
    for label, rows, cells in arms:
        for readout in halo_readouts(rows):
            vals = []
            for k in KS:
                v, n = sealed_value(rows, "halo", readout, window, k, cells)
                vals.append("-" if v is None else f"{v:.1f}" + ("" if n == 6 else f"({n})"))
            if any(x != "-" for x in vals):
                lines.append(f"| {label} / {readout} | " + " | ".join(vals) + " |")
    cells = arms[0][2] if arms else None
    for model, name in BASELINES.items():
        for readout in ("equal-weight-normalized-fusion", "1nn", "native_zero_support"):
            vals = []
            for k in KS:
                v, n = sealed_value(base, model, readout, window, k, cells)
                vals.append("-" if v is None else f"{v:.1f}" + ("" if n == 6 else f"({n})"))
            if any(x != "-" for x in vals):
                lines.append(f"| {name} / {readout} | " + " | ".join(vals) + " |")


# ----------------------------------------------------------------------------- scenarios
def is_control(row):
    variant = str(row.get("variant") or "")
    return (row.get("condition") == "control" or "matched_control" in variant
            or "matched_full_control" in variant)


def primary(row, cells):
    return (row.get("status") == "ok" and not is_control(row)
            and row.get("coverage_split", "all") == "all"
            and row.get("subject_relation") != "same_subject"
            and row.get("f1_macro") is not None
            and (cells is None or scen_cell(row) in cells))


def scenario_value(rows, model, readout, scenario, k, cells, split="all"):
    values = [row["f1_macro"] for row in rows
              if row.get("model") == model and row.get("readout") == readout
              and row.get("scenario") == scenario and int(row.get("k", -1)) == k
              and (primary(row, cells) if split == "all" else (
                  row.get("status") == "ok" and not is_control(row)
                  and row.get("coverage_split") == split
                  and row.get("subject_relation") != "same_subject"
                  and row.get("f1_macro") is not None
                  and (cells is None or scen_cell(row) in cells)))]
    return (st.mean(values), len(values)) if values else (None, 0)


def best_baseline(base, scenario, k, cells):
    best = None
    readouts = ("1nn",) if k > 0 else ("equal-weight-normalized-fusion", "native_zero_support",
                                       "training-bank-1nn-conse")
    for model, name in BASELINES.items():
        for readout in readouts:
            v, _ = scenario_value(base, model, readout, scenario, k, cells)
            if v is not None and (best is None or v > best[0]):
                best = (v, f"{name} {'1-NN' if readout == '1nn' else readout}")
    return best


def pair_key(row):
    """Scenario cells link to their control by matched_group (s2/s3/s7) or matched_parent (s4/s5)."""
    return row.get("matched_group") or row.get("matched_parent")


def paired_cost(rows, model, readout, scenario, k):
    """Scenario minus its matched control on the same queries (matched_parent linkage)."""
    controls = {}
    for row in rows:
        if (row.get("status") == "ok" and is_control(row)
                and row.get("model") == model and row.get("readout") == readout
                and row.get("scenario") == scenario and int(row.get("k", -1)) == k
                and row.get("coverage_split", "all") == "all" and row.get("f1_macro") is not None):
            controls[pair_key(row)] = row["f1_macro"]
    deltas = [row["f1_macro"] - controls[pair_key(row)] for row in rows
              if primary(row, None) and row.get("model") == model and row.get("readout") == readout
              and row.get("scenario") == scenario and int(row.get("k", -1)) == k
              and pair_key(row) in controls]
    return (st.mean(deltas), len(deltas)) if deltas else (None, 0)


def scenario_tables(arms, base, lines):
    lines.append("\n### Scenarios, 8 s, macro F1: mean over primary scenario cells "
                 "(matched controls excluded, whole-query split, no same-subject variants)\n")
    header = ["scenario", "k"] + [f"{label} / {r}" for label, rows, _ in arms
                                  for r in ("halo-classifier", "1nn")] + ["best baseline"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---" * len(header) + "|")
    cells = arms[0][2] if arms else None
    for scenario, sname in SCENARIOS.items():
        for k in SCEN_KS:
            vals = []
            for label, rows, arm_cells in arms:
                for readout in ("halo-classifier", "1nn"):
                    v, n = scenario_value(rows, "halo", readout, scenario, k, None)
                    vals.append("-" if v is None else f"{v:.1f}")
            if all(v == "-" for v in vals):
                continue
            b = best_baseline(base, scenario, k, None)
            vals.append("-" if b is None else f"{b[0]:.1f} {b[1]}")
            lines.append(f"| {sname} | {k} | " + " | ".join(vals) + " |")
    lines.append("\n### Partial coverage by truth split, 8 s, macro F1 (k=8)\n")
    lines.append("| row | all | truth enrolled | truth unenrolled |")
    lines.append("|---|---:|---:|---:|")
    for label, rows, arm_cells in arms:
        for readout in halo_readouts(rows):
            vals = [scenario_value(rows, "halo", readout, "s1_partial_coverage", 8, None, s)[0]
                    for s in ("all", "truth_enrolled", "truth_unenrolled")]
            if any(v is not None for v in vals):
                lines.append(f"| {label} / {readout} | " + " | ".join(
                    "-" if v is None else f"{v:.1f}" for v in vals) + " |")
    lines.append("\n### Paired mismatch cost at k=8 (scenario minus matched control, same queries)\n")
    cost_cols = [(rows, "halo", r, f"{label} / {r}") for label, rows, _ in arms
                 for r in ("halo-classifier", "1nn")]
    cost_cols += [(base, "unimts", "1nn", "UniMTS 1-NN"), (base, "harnet10", "1nn", "HARNet-10 1-NN")]
    lines.append("| scenario | " + " | ".join(c[3] for c in cost_cols) + " |")
    lines.append("|---" * (1 + len(cost_cols)) + "|")
    for scenario, sname in SCENARIOS.items():
        vals = []
        for rows, model, readout, _ in cost_cols:
                v, n = paired_cost(rows, model, readout, scenario, 8)
                vals.append("-" if v is None else f"{v:+.1f} ({n})")
        if any(v != "-" for v in vals):
            lines.append(f"| {sname} | " + " | ".join(vals) + " |")


# ----------------------------------------------------------------------------- figures
PALETTE = ["#1f5fa8", "#d1495b", "#2e8b57", "#e08e0b", "#6a4c93", "#6c757d", "#17a2b8"]


def k_curve_figure(arms, base, out: Path, title: str, halo_rows_spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    positions = {k: i for i, k in enumerate(KS)}
    for ax, window in zip(axes, WINDOWS):
        series = []
        for (label, rows, cells), (readouts, style) in zip(arms, halo_rows_spec):
            for readout, name, color, ls in readouts:
                series.append((rows, "halo", readout, name, color, ls, cells))
        for i, (model, name) in enumerate(BASELINES.items()):
            if model == "harnet5":
                continue
            series.append((base, model, "1nn", f"{name} 1-NN", "#9aa0a6", [":", "--", "-.", (0, (1, 1))][i % 4],
                           arms[0][2]))
        for rows, model, readout, name, color, ls, cells in series:
            xs, ys = [], []
            for k in KS[1:]:
                v, _ = sealed_value(rows, model, readout, window, k, cells)
                if v is not None:
                    xs.append(positions[k]); ys.append(v)
            if xs:
                ax.plot(xs, ys, color=color, linestyle=ls, marker="o", ms=3.5, lw=1.8, label=name)
            v0, _ = sealed_value(rows, model, readout if readout != "1nn" else "training-bank-1nn-conse",
                                 window, 0, cells)
            if model == "halo" and readout.startswith("halo-classifier") and v0 is None:
                v0, _ = sealed_value(rows, model, readout, window, 0, cells)
            if v0 is not None and model == "halo" and readout.startswith("halo-classifier"):
                ax.plot([0], [v0], marker="*", ms=13, color=color, linestyle="none")
        ax.set_xticks(range(len(KS)), ["0"] + [str(k) for k in KS[1:]])
        ax.set_title(f"{window:g}-second windows")
        ax.set_xlabel("enrolled executions per candidate (k); ★ = k=0")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("dataset-balanced macro F1")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", default=[], help="label=directory")
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None, help="k-curve PNG (sealed arms only)")
    parser.add_argument("--figure-title", default="HALO vs released baselines, sealed v5")
    args = parser.parse_args()
    base_sealed, base_scen = load(BASE_SEALED), load(BASE_SCEN)
    sealed_arms, scen_arms = [], []
    for spec in args.arm:
        label, _, path = spec.partition("=")
        rows = load(Path(path))
        if not rows:
            raise SystemExit(f"no results.json.gz under {path}")
        if is_scenario_payload(rows):
            same, diff = identical_cells(rows, base_scen, scen_cell)
            scen_arms.append((label, rows, same))
        else:
            same, diff = identical_cells(rows, base_sealed, sealed_cell)
            sealed_arms.append((label, rows, same))
        print(f"{label}: {len(same)} manifest-identical cells with baselines, {len(diff)} differing")
        if diff:
            print(f"  WARNING: {len(diff)} shared cells have different manifests and are excluded")
    lines = [f"<!-- generated by results/tools/v5_report.py -->"]
    for window in WINDOWS:
        if sealed_arms:
            sealed_tables(sealed_arms, base_sealed, window, lines)
    if scen_arms:
        scenario_tables(scen_arms, base_scen, lines)
    text = "\n".join(lines) + "\n"
    if args.markdown:
        args.markdown.write_text(text)
    print(text)
    if args.figure and sealed_arms:
        spec = []
        for i, (label, rows, _) in enumerate(sealed_arms):
            color = PALETTE[i % len(PALETTE)]
            spec.append(([("halo-classifier", f"{label}: classifier", color, "-")], None))
        k_curve_figure(sealed_arms, base_sealed, args.figure, args.figure_title, spec)
        print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
