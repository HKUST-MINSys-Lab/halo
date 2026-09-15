"""Summarize opt-in residual-classifier isolation rows and training telemetry."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


DISPLAY = {
    "halo-classifier": "full",
    "halo-classifier-residual-off": "floor",
    "halo-classifier-text-only": "text only",
    "halo-classifier-support-residual-only": "support residual",
    "halo-classifier-candidate-residual-only": "candidate residual",
    "halo-classifier-residual-only": "both residuals",
    "halo-classifier-support-label-shuffled": "shuffled support-label text",
}
ORDER = tuple(DISPLAY)


def _dataset_means(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        if (row.get("status") == "ok" and row.get("model") == "halo"
                and row.get("readout") in DISPLAY
                and row.get("multi_device_mode") == "single-device"):
            grouped[(row["readout"], int(row["k"]), row["dataset"])].append(
                float(row["f1_macro"])
            )
    return [
        {"readout": readout, "k": k, "dataset": dataset, "f1_macro": mean(values)}
        for (readout, k, dataset), values in sorted(grouped.items())
    ]


def _aggregate(dataset_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in dataset_rows:
        grouped[(row["readout"], row["k"])].append(row["f1_macro"])
    floors = {
        k: mean(values) for (readout, k), values in grouped.items()
        if readout == "halo-classifier-residual-off"
    }
    result = []
    for (readout, k), values in sorted(grouped.items()):
        value = mean(values)
        result.append({
            "readout": readout, "k": k, "f1_macro": value,
            "delta_from_floor": value - floors[k] if k in floors else None,
            "n_datasets": len(values),
        })
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_summary(log_path: Path) -> tuple[dict[str, float], list[dict]]:
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "classifier/mean_abs_r_support" in row]
    tail = rows[-20:]
    keys = (
        "classifier/lambda_0", "classifier/lambda_1", "classifier/lambda_2",
        "classifier/lambda_4", "classifier/lambda_8",
        "classifier/mean_abs_r_support", "classifier/mean_abs_r_candidate",
        "classifier/text_score_gt_minus_max_other",
    )
    summary = {key: median(float(row[key]) for row in tail) for key in keys}
    return summary, rows


def _plots(aggregate: list[dict], telemetry: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    ks = sorted({row["k"] for row in aggregate})
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for readout in ORDER:
        points = {row["k"]: row["f1_macro"] for row in aggregate if row["readout"] == readout}
        if points:
            ax.plot([ks.index(k) for k in points], list(points.values()), marker="o",
                    linewidth=2, label=DISPLAY[readout])
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_xlabel("Enrollment per candidate (k)")
    ax.set_ylabel("Dataset-balanced macro-F1")
    ax.set_title("HALO classifier component isolation, 8-second windows")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "component_k_curves.png", dpi=180)
    plt.close(fig)

    steps = [row["step"] for row in telemetry]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for bucket in (0, 1, 2, 4, 8):
        key = f"classifier/lambda_{bucket}"
        axes[0].plot(steps, [row[key] for row in telemetry], label=f"k={bucket}+")
    axes[0].set_title("Learned text weights")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("lambda")
    axes[0].legend(fontsize=8)
    for key, label in (
        ("classifier/mean_abs_r_support", "support residual"),
        ("classifier/mean_abs_r_candidate", "candidate residual"),
        ("classifier/text_score_gt_minus_max_other", "text GT margin"),
    ):
        axes[1].plot(steps, [row[key] for row in telemetry], label=label)
    axes[1].set_title("Classifier behavior during training")
    axes[1].set_xlabel("Training step")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "training_components.png", dpi=180)
    plt.close(fig)


def _markdown(aggregate: list[dict], dataset_rows: list[dict], telemetry: dict[str, float]) -> str:
    by_key = {(row["readout"], row["k"]): row for row in aggregate}
    ks = sorted({row["k"] for row in aggregate})
    lines = [
        "# HALO classifier isolation - 2026-09-15", "",
        "Eight-second, single-device sealed cells. Every component uses the same checkpoint,",
        "cached recording representations, and immutable execution-disjoint manifests. Values",
        "are macro-F1 percentages, first averaged across a dataset's streams and then equally",
        "across datasets. These are diagnostic ablations, not checkpoint-selection results.", "",
        "![Component k-curves](component_k_curves.png)", "", "## Component curves", "",
        "| readout | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|" + "---:|" * len(ks),
    ]
    for readout in ORDER:
        cells = []
        for k in ks:
            row = by_key.get((readout, k))
            cells.append("-" if row is None else f"{row['f1_macro']:.2f}")
        if any(cell != "-" for cell in cells):
            lines.append(f"| {DISPLAY[readout]} | " + " | ".join(cells) + " |")
    lines += ["", "## Delta from centred-neighbour floor", "",
              "| readout | " + " | ".join(f"k={k}" for k in ks if k > 0) + " |",
              "|---|" + "---:|" * sum(k > 0 for k in ks)]
    for readout in ORDER:
        if readout == "halo-classifier-residual-off":
            continue
        cells = []
        for k in ks:
            if k == 0:
                continue
            row = by_key.get((readout, k))
            delta = None if row is None else row["delta_from_floor"]
            cells.append("-" if delta is None else f"{delta:+.2f}")
        if any(cell != "-" for cell in cells):
            lines.append(f"| {DISPLAY[readout]} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-dataset diagnostic deltas", "",
              "Deltas from the centred-neighbour floor at the three most informative support counts.", "",
              "| dataset | k | text only | support residual | candidate residual | both residuals | full | shuffled labels |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    dataset_lookup = {(row["dataset"], row["k"], row["readout"]): row["f1_macro"]
                      for row in dataset_rows}
    for dataset in sorted({row["dataset"] for row in dataset_rows}):
        for k in (1, 8, 32):
            floor = dataset_lookup.get((dataset, k, "halo-classifier-residual-off"))
            if floor is None:
                continue
            columns = []
            for readout in (
                "halo-classifier-text-only", "halo-classifier-support-residual-only",
                "halo-classifier-candidate-residual-only", "halo-classifier-residual-only",
                "halo-classifier", "halo-classifier-support-label-shuffled",
            ):
                value = dataset_lookup.get((dataset, k, readout))
                columns.append("-" if value is None else f"{value - floor:+.2f}")
            lines.append(f"| {dataset} | {k} | " + " | ".join(columns) + " |")

    lines += ["", "## Training telemetry", "", "![Training components](training_components.png)", "",
              "Median over the final 20 logged training batches:", "",
              "| measure | value |", "|---|---:|"]
    for key, value in telemetry.items():
        lines.append(f"| `{key}` | {value:.4f} |")
    lines += ["", "## Interpretation", "",
              "Interpretation is added after reviewing the generated component and per-dataset deltas.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.results.read_text())
    dataset_rows = _dataset_means(rows)
    aggregate = _aggregate(dataset_rows)
    telemetry_summary, telemetry_rows = _training_summary(args.training_log)
    _write_csv(args.out / "aggregate.csv", aggregate)
    _write_csv(args.out / "per_dataset.csv", dataset_rows)
    _plots(aggregate, telemetry_rows, args.out)
    (args.out / "summary.json").write_text(json.dumps({
        "aggregate": aggregate, "per_dataset": dataset_rows,
        "telemetry_final20_median": telemetry_summary,
    }, indent=2) + "\n")
    (args.out / "RESULTS.md").write_text(
        _markdown(aggregate, dataset_rows, telemetry_summary)
    )


if __name__ == "__main__":
    main()
