"""Combine independently-run sealed baseline outputs without changing any result row.

The sealed evaluator runs one provider at a time to avoid incompatible released-model
allocations.  This utility only concatenates already-written ``results.json`` files and writes
the same Markdown table format as the evaluator; it never constructs episodes or scores a model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path



_NATIVE_CAPABILITIES = {
    "halo": {"native_open_set_labels": True, "native_few_shot_adaptation": True},
    "harnet": {"native_open_set_labels": False, "native_few_shot_adaptation": False},
    "limubert_x": {"native_open_set_labels": False, "native_few_shot_adaptation": False},
    "unimts": {"native_open_set_labels": True, "native_few_shot_adaptation": False},
    "normwear": {"native_open_set_labels": True, "native_few_shot_adaptation": False},
}

# Audited full released-model capacity in millions. The shared 1-NN/prototype/ridge readouts
# are parameter-free and are deliberately not included in these values.
_PARAMETER_COUNT_M = {
    "harnet": 4.49,
    "limubert_x": 0.055,
    "unimts": 68.61,
    "normwear": 1293.86,
}


def _table(columns: tuple[str, ...], rows: list[dict]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines.extend("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows)
    return lines


def _report_markdown(rows: list[dict], path: Path) -> None:
    """Write duration- and dataset-grouped results plus a dataset-balanced mean.

    A dataset may expose more than one single-device placement and an optional composite cell.
    The mean first averages those cells inside each dataset, then averages datasets, so RealWorld's
    additional placements do not carry more weight than another sealed dataset.
    """
    detail_columns = (
        "model", "parameters_m", "readout", "k", "native_open_set_labels", "native_few_shot_adaptation",
        "stream", "n_devices", "accuracy", "f1_macro", "balanced_accuracy", "n_queries", "status",
    )
    lines = ["# Sealed support-conditioned HAR results", "",
             "Each section uses one physical evidence duration. Rows are immutable sealed episodes; ",
             "the mean is dataset-balanced after averaging a dataset's available placement cells.", ""]
    valid = [row for row in rows if row.get("status") == "ok" and
             isinstance(row.get("accuracy"), (int, float)) and math.isfinite(float(row["accuracy"]))]
    for duration in sorted({float(row["window_seconds"]) for row in rows if "window_seconds" in row}):
        at_duration = [row for row in rows if float(row.get("window_seconds", -1)) == duration]
        lines.extend([f"## {duration:g}-second windows", ""])
        for dataset in sorted({row.get("dataset", "") for row in at_duration}):
            section = [row for row in at_duration if row.get("dataset") == dataset]
            lines.extend([f"### {dataset}", * _table(detail_columns, section), ""])
        grouped: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in valid:
            if float(row["window_seconds"]) != duration:
                continue
            grouped[(row["model"], row["readout"], int(row["k"]))][row["dataset"]].append(row)
        means = []
        for (model, readout, k), by_dataset in sorted(grouped.items()):
            dataset_rows = [
                {
                    "accuracy": sum(float(row["accuracy"]) for row in values) / len(values),
                    "f1_macro": sum(float(row["f1_macro"]) for row in values) / len(values),
                    "native_open_set_labels": values[0]["native_open_set_labels"],
                    "native_few_shot_adaptation": values[0]["native_few_shot_adaptation"],
                    "parameters_m": values[0]["parameters_m"],
                }
                for values in by_dataset.values()
            ]
            means.append({
                "model": model, "readout": readout, "k": k,
                "native_open_set_labels": dataset_rows[0]["native_open_set_labels"],
                "native_few_shot_adaptation": dataset_rows[0]["native_few_shot_adaptation"],
                "parameters_m": dataset_rows[0]["parameters_m"],
                "accuracy": round(sum(row["accuracy"] for row in dataset_rows) / len(dataset_rows), 3),
                "f1_macro": round(sum(row["f1_macro"] for row in dataset_rows) / len(dataset_rows), 3),
                "n_datasets": len(dataset_rows),
            })
        lines.extend(["### Dataset-Balanced Mean", * _table(
            ("model", "parameters_m", "readout", "k", "native_open_set_labels", "native_few_shot_adaptation",
             "accuracy", "f1_macro", "n_datasets"), means), ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="directory containing one subdirectory per provider")
    parser.add_argument("--models", nargs="+", required=True,
                        help="provider subdirectories, in desired table order")
    parser.add_argument("--out", type=Path, default=None,
                        help="combined output directory; defaults to ROOT/combined")
    args = parser.parse_args()

    rows = []
    sources = []
    for model in args.models:
        source = args.root / model / "results.json"
        if not source.exists():
            raise FileNotFoundError(f"{model}: completed sealed result missing: {source}")
        provider_rows = json.loads(source.read_text())
        if any(row.get("model") != model for row in provider_rows):
            raise ValueError(f"{source}: contains rows for a different provider")
        # Differentiable neighbours is an encoder-training control, not a deployment
        # readout. Historical evaluator output may contain it; never include it in reports.
        provider_rows = [row for row in provider_rows
                         if row.get("readout") != "differentiable-neighbors"]
        if model not in _NATIVE_CAPABILITIES:
            raise ValueError(f"{model}: native capability disclosure is not registered")
        for row in provider_rows:
            row.update(_NATIVE_CAPABILITIES[model])
            if model in _PARAMETER_COUNT_M:
                row["parameters_m"] = _PARAMETER_COUNT_M[model]
            elif "parameters_m" not in row:
                raise ValueError(f"{model}: result rows must declare checkpoint parameter count")
        rows.extend(provider_rows)
        sources.append(str(source))

    output = args.out or args.root / "combined"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=True) + "\n")
    (output / "sources.json").write_text(json.dumps({"sources": sources}, indent=2) + "\n")
    _report_markdown(rows, output / "RESULTS.md")
    print(f"[merge-sealed-results] wrote {output / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
