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
# Runs of the HALO adapter that are controls, not the headline: assembled under
# ``halo_compare@<variant>`` (see ``eval.run_adaptation_baselines --variant``). They are listed
# beside the trained row whenever present and never substitute for it.
CONTROL_MODEL_NAMES = {
    "halo_compare@step0": "HALO step-0 control (untrained)",
    "halo_compare@untrained_floor": "HALO untrained floor",
}
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
MAIN_KS = (1, 2, 4, 8)
PRIMARY_SUBJECT_RELATION = "cross_subject"
PRIMARY_CONFIGURATION_RELATION = "same_configuration"


def _is_main_curve_row(row: dict) -> bool:
    """Select the fixed cell cohort used for comparable k=1..8 curves."""

    return (
        row.get("analysis_set", "main_common_k1_8") == "main_common_k1_8"
        and row.get("cohort", "main") == "main"
    )


def _has_relation_fields(cells: list[dict]) -> bool:
    return any("subject_relation" in row for row in cells)


def _relation_rows(cells: list[dict], subject: str, configuration: str) -> list[dict]:
    if not _has_relation_fields(cells):
        return [row for row in cells if _is_main_curve_row(row)]
    return [
        row for row in cells
        if _is_main_curve_row(row)
        and row.get("subject_relation") == subject
        and row.get("configuration_relation") == configuration
    ]


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
           "No labelled examples. Macro F1 on the intersection of scored cells shared by all "
           "models with zero-shot results. Datasets receive equal weight. Missing models are "
           "shown as n/a; coverage outside the matched cohort is not a head-to-head score.", ""]
    models = list(ZERO_SHOT_MODEL_ORDER) + _control_models(cells)
    def cell_key(row):
        return (row["dataset"], row.get("cell", row["dataset"]))
    by_model = {
        model: [r for r in cells if r["model"] == model and r["method"] == "zero_shot"
                and r["label_mode"] == "coherent" and np.isfinite(float(r["f1_macro"]))]
        for model in models
    }
    available = [set(map(cell_key, rows)) for rows in by_model.values() if rows]
    common = set.intersection(*available) if available else set()
    scored = []
    for model in models:
        overall, _ = _dataset_macro([r for r in by_model[model] if cell_key(r) in common])
        scored.append((MODEL_NAMES.get(model, CONTROL_MODEL_NAMES.get(model, model)), [overall]))
    scored.sort(key=lambda r: -r[1][0] if np.isfinite(r[1][0]) else float("inf"))
    out += _emit(["model", "matched cells"], scored)
    out += ["", f"Matched held-out datasets: {len({k[0] for k in common})}; cells: {len(common)}.",
            "", "| model | scored datasets | scored cells |", "|---|---:|---:|"]
    for model in models:
        keys = set(map(cell_key, by_model[model]))
        name = MODEL_NAMES.get(model, CONTROL_MODEL_NAMES.get(model, model))
        out.append(f"| {name} | {len({k[0] for k in keys})} | {len(keys)} |")
    out.append("")
    return "\n".join(out)


def _control_models(cells: list[dict]) -> list[str]:
    present = {row["model"] for row in cells}
    return [model for model in CONTROL_MODEL_NAMES if model in present]


def _enrollment_model_methods(cells: list[dict] | None = None) -> list[tuple[str, str, str]]:
    rows = []
    readout_names = {"nearest": "1-NN", "prototype": "prototype", "ridge": "ridge"}
    for model in REPORT_MODEL_ORDER:
        display_model = "HALO" if model == "halo_compare" else MODEL_NAMES[model]
        if model == "halo_compare":
            rows.append(("HALO / support comparator", model, "support_comparator"))
            for control in _control_models(cells or []):
                rows.append((CONTROL_MODEL_NAMES[control], control, "support_comparator"))
        rows.extend(
            (f"{display_model} / {display_method}", model, method)
            for method, display_method in readout_names.items()
        )
    return rows


def table_label_efficiency(cells: list[dict]) -> str:
    ks = list(MAIN_KS)
    out = ["## 2. Label efficiency", "",
           "`k` is the number of independent enrolled executions per candidate. HALO is shown "
           "with its learned support comparator in addition to the same three non-gradient "
           "readouts used for every representation: one-nearest-neighbor, support prototypes, "
           "and closed-form ridge regression. All readouts see only the enrolled support "
           "executions. Every k=1..8 column uses the same protocol cells. Macro F1 is averaged "
           "over datasets. Subject and configuration relations are never pooled.", ""]
    relations = (
        ("Same configuration, cross subject", "cross_subject", "same_configuration"),
        ("Same configuration, same subject", "same_subject", "same_configuration"),
        ("Cross configuration, cross subject", "cross_subject", "cross_configuration"),
        ("Cross configuration, same subject", "same_subject", "cross_configuration"),
        ("Subject identity unavailable", "unattributed", "same_configuration"),
    )
    for title, subject_relation, configuration_relation in relations:
        relation_cells = _relation_rows(cells, subject_relation, configuration_relation)
        if not relation_cells:
            continue
        scored = []
        for display_name, model, method in _enrollment_model_methods(cells):
            row = [
                _dataset_macro([
                    c for c in relation_cells
                    if c["model"] == model and c["method"] == method
                    and c["label_mode"] == "coherent" and int(c["k"]) == k
                ])[0]
                for k in ks
            ]
            scored.append((display_name, row))
        datasets = sorted({c["dataset"] for c in relation_cells})
        out += [f"### {title}", "", f"Fixed-cohort datasets: {len(datasets)}.", ""]
        out += _emit(["model"] + [f"k={k}" for k in ks], scored)
        out.append("")

    high_support = [
        c for c in cells
        if c.get("analysis_set") == "secondary_high_support"
        and c.get("label_mode") == "coherent" and int(c["k"]) == 16
    ]
    if high_support:
        out += ["### Secondary k=16 cohort", "",
                "This is a separately eligible subject/cell cohort and is not an extension of the "
                "fixed k=1..8 curve.", ""]
        scored = []
        for display_name, model, method in _enrollment_model_methods(cells):
            value = _dataset_macro([
                c for c in high_support if c["model"] == model and c["method"] == method
            ])[0]
            scored.append((display_name, [value]))
        out += _emit(["model", "k=16"], scored)
        out.append("")
    return "\n".join(out)


def table_per_dataset(cells: list[dict]) -> str:
    ks = list(MAIN_KS)
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

    out += ["### Enrollment by dataset", "",
            "Fixed k=1..8 cohort; same-configuration, cross-subject relation only. Other relations "
            "are reported separately in the aggregate table.", ""]
    enrollment_cells = _relation_rows(
        cells, PRIMARY_SUBJECT_RELATION, PRIMARY_CONFIGURATION_RELATION
    )
    enrollment_datasets = sorted(
        {c["dataset"] for c in enrollment_cells}, key=lambda d: DATASET_NAMES.get(d, d)
    )
    for dataset in enrollment_datasets:
        scored = []
        for display_name, model, method in _enrollment_model_methods(cells):
            values = [
                _dataset_macro([
                    c for c in enrollment_cells
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
