"""Plot current matched adaptation k-curves from an assembled cells table.

The two outputs answer different questions and must remain separate:

1. ``knn_representation_curves`` applies the same 1-NN readout to every representation.
2. ``primary_adaptation_curves`` shows HALO's support comparator beside the same no-fitting 1-NN rule
   on every frozen representation.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_NAMES = {
    "halo_compare": "HALO",
    "harnet": "HARNet",
    "unimts": "UniMTS",
    "normwear": "NormWear",
    "imagebind": "ImageBind",
}
KS = (1, 2, 4, 8)
COLORS = {
    "HALO / support comparator": "#c43c39",
    "HALO / 1-NN": "#111111",
    "HALO": "#111111",
    "HARNet": "#2878b5",
    "UniMTS": "#5a9f4b",
    "ImageBind": "#8c6d5c",
    "NormWear": "#6f7b83",
}


def _load(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def _dataset_macro(
    rows: list[dict], *, model: str, method: str, k: int,
    subject_relation: str, configuration_relation: str,
) -> float:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (row["model"] == model and row["method"] == method
                and row["label_mode"] == "coherent"
                and row.get("analysis_set") == "main_common_k1_8"
                and row.get("cohort") == "main"
                and row.get("subject_relation") == subject_relation
                and row.get("configuration_relation") == configuration_relation
                and int(row["k"]) == k):
            by_dataset[row["dataset"]].append(float(row["f1_macro"]))
    if not by_dataset:
        return float("nan")
    return float(np.mean([np.mean(values) for values in by_dataset.values()]))


def _curve(
    rows: list[dict], model: str, method: str,
    subject_relation: str, configuration_relation: str,
) -> list[float]:
    return [
        _dataset_macro(
            rows, model=model, method=method, k=k,
            subject_relation=subject_relation,
            configuration_relation=configuration_relation,
        )
        for k in KS
    ]


def _style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="semibold", pad=8)
    ax.set_xticks(range(len(KS)), [str(k) for k in KS])
    ax.set_xlabel("Enrolled executions per candidate (k)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_knn(
    rows: list[dict], out_dir: Path, subject_relation: str, configuration_relation: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    for model, name in MODEL_NAMES.items():
        values = _curve(rows, model, "nearest", subject_relation, configuration_relation)
        ax.plot(
            range(len(KS)), values, marker="o", markersize=4,
            linewidth=2.8 if model == "halo_compare" else 1.5,
            color=COLORS[name], label=name, zorder=5 if model == "halo_compare" else 2,
        )
    _style_axes(ax, f"{subject_relation.replace('_', ' ')}, {configuration_relation.replace('_', ' ')}")
    ax.set_ylabel("Macro F1")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=8.5)
    fig.suptitle("Matched 1-NN comparison of frozen representations", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.subplots_adjust(bottom=0.23)
    _save(fig, out_dir, "knn_representation_curves")


def plot_primary(
    rows: list[dict], out_dir: Path, subject_relation: str, configuration_relation: str,
) -> None:
    series = [
        ("HALO / support comparator", "halo_compare", "support_comparator"),
        ("HALO / 1-NN", "halo_compare", "nearest"),
        *((MODEL_NAMES[model], model, "nearest")
          for model in MODEL_NAMES if model != "halo_compare"),
    ]
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    for name, model, method in series:
        halo = model == "halo_compare"
        ax.plot(
            range(len(KS)), _curve(
                rows, model, method, subject_relation, configuration_relation
            ),
            marker="o", markersize=4, linewidth=2.8 if halo else 1.4,
            linestyle="--" if method == "support_comparator" else "-",
            color=COLORS[name], label=name, zorder=5 if halo else 2,
        )
    _style_axes(ax, f"{subject_relation.replace('_', ' ')}, {configuration_relation.replace('_', ' ')}")
    ax.set_ylabel("Macro F1")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08), fontsize=8.5)
    fig.suptitle("Label-efficient adaptation: HALO support comparator and matched 1-NN", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.subplots_adjust(bottom=0.27)
    _save(fig, out_dir, "primary_adaptation_curves")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subject-relation", default="cross_subject")
    parser.add_argument("--configuration-relation", default="same_configuration")
    args = parser.parse_args()
    rows = _load(args.cells)
    plot_knn(rows, args.out_dir, args.subject_relation, args.configuration_relation)
    plot_primary(rows, args.out_dir, args.subject_relation, args.configuration_relation)
    print(f"wrote k-curve figures to {args.out_dir}")


if __name__ == "__main__":
    main()
