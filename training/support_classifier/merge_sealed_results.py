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
    "halo": {"native_open_set_labels": True, "native_support_conditioning": True,
             "published_few_label_finetuning": False},
    "harnet5": {"native_open_set_labels": False, "native_support_conditioning": False,
                "published_few_label_finetuning": True},
    "harnet10": {"native_open_set_labels": False, "native_support_conditioning": False,
                 "published_few_label_finetuning": True},
    "limubert_x": {"native_open_set_labels": False, "native_support_conditioning": False,
                   "published_few_label_finetuning": True},
    "unimts": {"native_open_set_labels": True, "native_support_conditioning": False,
               "published_few_label_finetuning": True},
    "normwear": {"native_open_set_labels": True, "native_support_conditioning": False,
                 "published_few_label_finetuning": True},
}

# Audited full released-model capacity in millions. The shared 1-NN/prototype/ridge readouts
# are parameter-free and are deliberately not included in these values.
_PARAMETER_COUNT_M = {
    "harnet5": 4.491,
    "harnet10": 10.983,
    "limubert_x": 0.055,
    "unimts": 68.61,
    "normwear": 1293.86,
}
_PARAMETER_BREAKDOWN_M = {
    "normwear": {
        "sensor_backbone": 136.1,
        "msitf_aggregator": 57.7,
        "frozen_tinyllama_text_tower": 1100.06,
    },
}
SEALED_RESULT_SCHEMA = "sealed-results-v3-20260916"


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
        "model", "parameters_m", "parameter_breakdown_m", "readout", "k", "native_open_set_labels",
        "native_support_conditioning", "published_few_label_finetuning",
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
                    "native_support_conditioning": values[0]["native_support_conditioning"],
                    "published_few_label_finetuning": values[0]["published_few_label_finetuning"],
                    "parameters_m": values[0]["parameters_m"],
                }
                for values in by_dataset.values()
            ]
            means.append({
                "model": model, "readout": readout, "k": k,
                "native_open_set_labels": dataset_rows[0]["native_open_set_labels"],
                "native_support_conditioning": dataset_rows[0]["native_support_conditioning"],
                "published_few_label_finetuning": dataset_rows[0]["published_few_label_finetuning"],
                "parameters_m": dataset_rows[0]["parameters_m"],
                "accuracy": round(sum(row["accuracy"] for row in dataset_rows) / len(dataset_rows), 3),
                "f1_macro": round(sum(row["f1_macro"] for row in dataset_rows) / len(dataset_rows), 3),
                "n_datasets": len(dataset_rows),
            })
        lines.extend(["### Dataset-Balanced Mean", * _table(
            ("model", "parameters_m", "readout", "k", "native_open_set_labels",
             "native_support_conditioning", "published_few_label_finetuning",
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
    combined_source = args.root / "results.json"
    combined_metadata = args.root / "run_metadata.json"
    if combined_source.exists() or combined_metadata.exists():
        if not combined_source.exists() or not combined_metadata.exists():
            raise FileNotFoundError("combined sealed run needs results.json and run_metadata.json")
        run = json.loads(combined_metadata.read_text())
        if not run.get("complete") or run.get("result_schema") != SEALED_RESULT_SCHEMA:
            raise ValueError(f"{combined_metadata}: stale or incomplete sealed run")
        available = set(run.get("models", ()))
        missing = sorted(set(args.models) - available)
        if missing:
            raise ValueError(f"combined sealed run is missing requested providers: {missing}")
        combined_rows = json.loads(combined_source.read_text())
        rows_by_model = {
            model: [row for row in combined_rows if row.get("model") == model]
            for model in args.models
        }
        sources.append(str(combined_source))
    else:
        rows_by_model = {}
        for model in args.models:
            source = args.root / model / "results.json"
            metadata = args.root / model / "run_metadata.json"
            if not source.exists():
                raise FileNotFoundError(f"{model}: completed sealed result missing: {source}")
            if not metadata.exists():
                raise FileNotFoundError(f"{model}: run metadata missing: {metadata}")
            run = json.loads(metadata.read_text())
            if not run.get("complete") or run.get("result_schema") != SEALED_RESULT_SCHEMA:
                raise ValueError(f"{metadata}: stale or incomplete sealed run")
            rows_by_model[model] = json.loads(source.read_text())
            sources.append(str(source))

    for model in args.models:
        provider_rows = rows_by_model[model]
        if any(row.get("model") != model for row in provider_rows):
            raise ValueError(f"sealed rows selected for {model} contain a different provider")
        if any(row.get("status") not in {"ok", "n/a"} for row in provider_rows):
            raise ValueError(f"{model}: contains failed or unfinished sealed rows")
        # Differentiable neighbours is an encoder-training control, not a deployment
        # readout. Historical evaluator output may contain it; never include it in reports.
        provider_rows = [row for row in provider_rows
                         if row.get("readout") != "differentiable-neighbors"]
        provider_rows = [row for row in provider_rows if not row.get("diagnostic_only")]
        if model not in _NATIVE_CAPABILITIES:
            raise ValueError(f"{model}: native capability disclosure is not registered")
        for row in provider_rows:
            row.update(_NATIVE_CAPABILITIES[model])
            if model in _PARAMETER_COUNT_M:
                row["parameters_m"] = _PARAMETER_COUNT_M[model]
            if model in _PARAMETER_BREAKDOWN_M:
                row["parameter_breakdown_m"] = _PARAMETER_BREAKDOWN_M[model]
            elif "parameters_m" not in row:
                raise ValueError(f"{model}: result rows must declare checkpoint parameter count")
        rows.extend(provider_rows)

    output = args.out or args.root / "combined"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=True) + "\n")
    (output / "sources.json").write_text(json.dumps({"sources": sources}, indent=2) + "\n")
    _report_markdown(rows, output / "RESULTS.md")
    print(f"[merge-sealed-results] wrote {output / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
