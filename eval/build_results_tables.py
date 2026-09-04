"""Emit current headline and diagnostic adaptation tables from assembled cells.

    python -m eval.build_results_tables \
      --cells eval/adaptation_tables/<run>/cells.csv \
      --out docs/results/TABLES.md

1. Zero-shot: HALO against released-checkpoint baselines with a native zero-shot rule.
2. Label efficiency: the same non-gradient 1-NN, prototype, and ridge readouts for every
   released-checkpoint representation, with HALO's learned support comparator shown separately.

The input must come from :mod:`eval.assemble_adaptation`, which validates the manifest, source and
checkpoint fingerprints before writing it. Aggregation first averages seeds within each dataset and
then gives every available dataset equal weight, so a dataset with more seeds does not outvote one
with fewer.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL_NAMES = {
    "halo_compare": "HALO (ours)",
    "harnet": "HARNet", "unimts": "UniMTS", "normwear": "NormWear",
    "imagebind": "ImageBind",
}
REPORT_MODEL_ORDER = (
    "halo_compare", "unimts", "harnet", "imagebind", "normwear",
)
# HARNet has a released representation checkpoint but no native open-vocabulary decision rule.
# Its locally fitted ConSE bridge is intentionally omitted from the paper's zero-shot comparison.
ZERO_SHOT_MODEL_ORDER = ("halo_compare", "unimts", "imagebind", "normwear")
DATASET_NAMES = {
    "motionsense": "MotionSense",
    "realworld": "RealWorld HAR",
    "shoaib": "Shoaib",
    "inclusivehar": "Inclusive-HAR",
    "usc_had": "USC-HAD",
    "tnda_har": "TNDA-HAR",
    "ut_complex": "UT Complex",
    "monipar": "MoniPar",
    "spar": "SPAR",
    "upper_limb_use": "Upper Limb Use",
}


def _emit(header: list[str], rows: list[tuple[str, list[float]]], higher_better=True) -> list[str]:
    """Markdown table with the best value in each column bolded, matching the house format."""
    best = []
    for column in range(len(header) - 1):
        values = [r[1][column] for r in rows if r[1][column] == r[1][column]]
        best.append(max(values) if values and higher_better else (min(values) if values else None))
    out = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    for name, values in rows:
        cells = []
        for column, value in enumerate(values):
            text = "n/a" if value != value else f"{value:.2f}"
            if value == value and best[column] is not None and abs(value - best[column]) < 1e-9:
                text = f"**{text}**"
            cells.append(text)
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return out


def _dataset_macro(rows: list[dict]) -> tuple[float, int]:
    """Mean over datasets of the per-dataset seed mean."""
    per_dataset = defaultdict(list)
    for row in rows:
        per_dataset[row["dataset"]].append(float(row["f1_macro"]))
    if not per_dataset:
        return float("nan"), 0
    return float(np.mean([np.mean(v) for v in per_dataset.values()])), len(per_dataset)


def _cells(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def _validate_current_cells(cells: list[dict]) -> None:
    present = {row["model"] for row in cells}
    missing = set(MODEL_NAMES) - present
    if missing:
        raise ValueError(
            "current report requires the released-checkpoint model roster; missing: "
            + ", ".join(sorted(missing))
        )
    if not any(
        row["model"] == "halo_compare" and row["method"] == "support_comparator"
        and int(row["k"]) > 0
        for row in cells
    ):
        raise ValueError(
            "HALO evidence-engine enrollment rows are missing; pooled-feature controls cannot be "
            "used as the current HALO headline"
        )
    required_readouts = {"nearest", "prototype", "ridge"}
    missing_readouts = {
        (model, method)
        for model in MODEL_NAMES
        for method in required_readouts
        if not any(
            row["model"] == model and row["method"] == method and int(row["k"]) > 0
            for row in cells
        )
    }
    if missing_readouts:
        formatted = ", ".join(f"{model}/{method}" for model, method in sorted(missing_readouts))
        raise ValueError(f"current report is missing matched enrollment readouts: {formatted}")


def table_zero_shot(cells: list[dict]) -> str:
    out = ["## 1. Zero-shot", "",
           "No labelled examples. Macro F1, equally averaged over all held-out datasets.", ""]
    scored = []
    for model in ZERO_SHOT_MODEL_ORDER:
        overall, _ = _dataset_macro([
            c for c in cells
            if c["model"] == model and c["method"] == "zero_shot"
            and c["label_mode"] == "coherent"
        ])
        scored.append((MODEL_NAMES[model], [overall]))
    scored.sort(key=lambda r: -np.nanmean(r[1]))
    out += _emit(["model", "all datasets"], scored)
    dataset_count = _dataset_macro([
        c for c in cells
        if c["method"] == "zero_shot" and c["label_mode"] == "coherent"
    ])[1]
    out += ["", f"Held-out datasets: {dataset_count}.", ""]
    return "\n".join(out)


def _enrollment_model_methods() -> list[tuple[str, str, str]]:
    rows = []
    readout_names = {"nearest": "1-NN", "prototype": "prototype", "ridge": "ridge"}
    for model in REPORT_MODEL_ORDER:
        display_model = "HALO" if model == "halo_compare" else MODEL_NAMES[model]
        if model == "halo_compare":
            rows.append(("HALO / support comparator", model, "support_comparator"))
        rows.extend(
            (f"{display_model} / {display_method}", model, method)
            for method, display_method in readout_names.items()
        )
    return rows


def table_label_efficiency(cells: list[dict]) -> str:
    ks = sorted({int(c["k"]) for c in cells if int(c["k"]) > 0})
    out = ["## 2. Label efficiency", "",
           "`k` is the number of independent enrolled executions per candidate. HALO is shown "
           "with its learned support comparator in addition to the same three non-gradient "
           "readouts used for every representation: one-nearest-neighbor, support prototypes, "
           "and closed-form ridge regression. All readouts see only the enrolled support "
           "executions. Macro F1, mean over datasets.", ""]
    scored = []
    for display_name, model, method in _enrollment_model_methods():
        row = [
            _dataset_macro([
                c for c in cells
                if c["model"] == model and c["method"] == method
                and c["label_mode"] == "coherent" and int(c["k"]) == k
            ])[0]
            for k in ks
        ]
        scored.append((display_name, row))
    out += _emit(["model"] + [f"k={k}" for k in ks], scored)
    out.append("")
    return "\n".join(out)


def table_per_dataset(cells: list[dict]) -> str:
    ks = sorted({int(c["k"]) for c in cells if int(c["k"]) > 0})
    out = [
        "## 3. Per-dataset performance", "",
        "These tables use the same protocol as the aggregate results. Values are macro F1 averaged "
        "over seeds within each held-out dataset.", "", "### Native zero-shot by dataset", "",
    ]
    datasets = sorted({c["dataset"] for c in cells}, key=lambda d: DATASET_NAMES.get(d, d))
    zero_rows = []
    for model in ZERO_SHOT_MODEL_ORDER:
        values = [
            _dataset_macro([
                c for c in cells
                if c["model"] == model and c["method"] == "zero_shot"
                and c["dataset"] == dataset and c["label_mode"] == "coherent"
                and int(c["k"]) == 0
            ])[0]
            for dataset in datasets
        ]
        zero_rows.append((MODEL_NAMES[model], values))
    out += _emit(["model"] + [DATASET_NAMES.get(d, d) for d in datasets], zero_rows)
    out.append("")

    out += ["### Enrollment by dataset", ""]
    for dataset in datasets:
        scored = []
        for display_name, model, method in _enrollment_model_methods():
            values = [
                _dataset_macro([
                    c for c in cells
                    if c["model"] == model and c["method"] == method
                    and c["dataset"] == dataset and c["label_mode"] == "coherent"
                    and int(c["k"]) == k
                ])[0]
                for k in ks
            ]
            scored.append((display_name, values))
        out += [f"#### {DATASET_NAMES.get(dataset, dataset)}", ""]
        out += _emit(["model / readout"] + [f"k={k}" for k in ks], scored)
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cells = _cells(args.cells)
    _validate_current_cells(cells)
    text = "\n".join([
        "# Results", "",
        table_zero_shot(cells),
        table_label_efficiency(cells),
        table_per_dataset(cells),
    ])
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
