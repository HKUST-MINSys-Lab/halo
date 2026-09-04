"""Assemble matched adaptation outputs into dataset-macro tables and paired comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import baselines
from eval.enrollment_protocol import load_manifest
from eval.run_adaptation_baselines import _source_fingerprint

NATIVE_ZERO_SHOT_MODELS = {"halo_compare", "unimts", "normwear", "imagebind"}
TARGET_KEY = ("halo_compare", "support_comparator")
MAIN_CURVE_MAX_K = 8


def _cell_id(result_key: str, kind: str) -> str:
    suffix_parts = 3 if kind == "enrollment" else 2
    return result_key.rsplit("/", suffix_parts)[0]


def _analysis_set(result: dict, cell: dict) -> str:
    if result["kind"] == "zero_shot":
        return "zero_shot"
    if result.get("cohort", "main") == "secondary_high_support":
        return "secondary_high_support"
    if int(cell["support_ceiling"]) >= MAIN_CURVE_MAX_K:
        return "main_common_k1_8"
    return "supplemental_partial_coverage"


def _external_rows(payload: dict, manifest: dict) -> tuple[list[dict], list[dict]]:
    model = payload["baseline"]
    rows, subjects = [], []
    for key, result in payload["results"].items():
        if result.get("status"):
            continue
        cell_id = _cell_id(key, result["kind"])
        if cell_id not in manifest["cells"]:
            raise ValueError(f"result references unknown manifest cell: {cell_id}")
        cell = manifest["cells"][cell_id]
        dataset = cell["dataset"]
        common = {
            "model": model,
            "dataset": dataset,
            "regime": result["regime"],
            "label_mode": result.get("label_mode", "coherent"),
            "k": int(result["support_count"]),
            "seed": int(result.get("seed", 0)),
            "cell": cell_id,
            "subject_relation": cell["subject_relation"],
            "configuration_relation": cell["configuration_relation"],
            "cohort": result.get("cohort", "zero_shot"),
            "analysis_set": _analysis_set(result, cell),
        }
        methods = ["zero_shot"] if result["kind"] == "zero_shot" else payload["methods"]
        for method in methods:
            metric = result.get(method)
            if not isinstance(metric, dict):
                continue
            rows.append({**common, "method": method, "f1_macro": float(metric["f1_macro"])})
            field = f"{method}_f1_macro"
            for subject, record in result.get("subject_results", {}).items():
                if common["subject_relation"] in {"none", "unattributed"}:
                    continue
                if field in record:
                    subjects.append({
                        **common, "method": method, "subject": f"{dataset}:{subject}",
                        "f1_macro": float(record[field]),
                    })
    return rows, subjects


def load_rows(paths: list[Path], manifest: dict) -> tuple[list[dict], list[dict]]:
    rows, subjects = [], []
    for path in paths:
        payload = json.loads(path.read_text())
        if "baseline" not in payload:
            raise ValueError(
                f"{path}: legacy model-specific result payload; rerun through "
                "eval.run_adaptation_baselines"
            )
        if int(payload.get("schema_version", 0)) < 2:
            raise ValueError(f"{path}: legacy result artifact; rerun with provenance schema 2")
        model = payload["baseline"]
        if model not in baselines.REGISTRY:
            raise ValueError(f"{path}: unknown adapter {model!r} in current source tree")
        current_source = _source_fingerprint(baselines.REGISTRY[model])
        if payload.get("source_fingerprint") != current_source:
            raise ValueError(
                f"{path}: evaluation source changed; rerun {model} before assembling tables"
            )
        for name, artifact in payload.get("evaluation_artifacts", {}).items():
            artifact_path = Path(artifact["path"])
            if not artifact_path.exists():
                raise ValueError(f"{path}: {name} artifact is missing: {artifact_path}")
            import hashlib
            digest = hashlib.sha256()
            with artifact_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != artifact.get("sha256"):
                raise ValueError(f"{path}: {name} artifact content changed; rerun {model}")
        if payload.get("git_dirty"):
            raise ValueError(f"{path}: result was produced from a dirty worktree")
        actual = payload.get("manifest_fingerprint")
        if actual != manifest["manifest_fingerprint"]:
            raise ValueError(
                f"{path}: manifest mismatch ({actual} != {manifest['manifest_fingerprint']})"
            )
        parsed = _external_rows(payload, manifest)
        rows.extend(parsed[0]); subjects.extend(parsed[1])
    return rows, subjects


def dataset_macro(rows: list[dict]) -> list[dict]:
    per_dataset = defaultdict(list)
    for row in rows:
        key = (
            row["model"], row["method"], row["regime"], row["label_mode"],
            row["subject_relation"], row["configuration_relation"], row["cohort"],
            row["analysis_set"], row["k"], row["dataset"],
        )
        per_dataset[key].append(row["f1_macro"])
    dataset_values = [
        {
            "model": key[0], "method": key[1], "regime": key[2], "label_mode": key[3],
            "subject_relation": key[4], "configuration_relation": key[5],
            "cohort": key[6], "analysis_set": key[7], "k": key[8], "dataset": key[9],
            "f1_macro": float(np.mean(values)),
            "protocol_cells": len(values),
        }
        for key, values in per_dataset.items()
    ]
    groups = defaultdict(list)
    for row in dataset_values:
        key = (
            row["model"], row["method"], row["regime"], row["label_mode"],
            row["subject_relation"], row["configuration_relation"], row["cohort"],
            row["analysis_set"], row["k"],
        )
        groups[key].append(row)
    return [
        {
            "model": key[0], "method": key[1], "regime": key[2], "label_mode": key[3],
            "subject_relation": key[4], "configuration_relation": key[5],
            "cohort": key[6], "analysis_set": key[7], "k": key[8],
            "f1_macro": float(np.mean([row["f1_macro"] for row in values])),
            "datasets": len(values),
        }
        for key, values in groups.items()
    ]


def paired_deltas(subject_rows: list[dict], samples: int = 5_000) -> list[dict]:
    """Dataset-balanced paired subject bootstrap within each evaluation condition.

    The reported point is the mean of per-dataset mean subject deltas. Resampling subjects within
    each fixed dataset therefore estimates the same dataset-balanced quantity instead of allowing a
    large-subject dataset to dominate the interval.
    """
    indexed = defaultdict(dict)
    for row in subject_rows:
        key = (
            row["regime"], row["label_mode"], row["subject_relation"],
            row["configuration_relation"], row["cohort"], row["analysis_set"],
            row["k"], row["cell"], row["seed"], row["dataset"], row["subject"],
        )
        indexed[(row["model"], row["method"])][key] = row["f1_macro"]
    target_key = TARGET_KEY
    if target_key not in indexed:
        return []
    target = indexed[target_key]
    rng = np.random.default_rng(20260817)
    output = []
    for comparator, values in indexed.items():
        if comparator == target_key:
            continue
        common = sorted(set(target) & set(values))
        if not common:
            continue
        by_condition = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for key in common:
            condition = key[:7]
            dataset, subject = key[-2:]
            by_condition[condition][dataset][subject].append(target[key] - values[key])
        for condition, by_dataset_subject in by_condition.items():
            dataset_subject_values = {
                dataset: np.asarray(
                    [np.mean(values) for values in by_subject.values()], dtype=np.float64
                )
                for dataset, by_subject in by_dataset_subject.items()
            }
            dataset_means = np.asarray(
                [values.mean() for values in dataset_subject_values.values()], dtype=np.float64
            )
            draws = np.stack([
                rng.choice(values, size=(samples, len(values)), replace=True).mean(1)
                for values in dataset_subject_values.values()
            ]).mean(0)
            (regime, label_mode, subject_relation, configuration_relation,
             cohort, analysis_set, k) = condition
            output.append({
                "target": f"{target_key[0]}/{target_key[1]}",
                "comparator": f"{comparator[0]}/{comparator[1]}",
                "regime": regime,
                "label_mode": label_mode,
                "subject_relation": subject_relation,
                "configuration_relation": configuration_relation,
                "cohort": cohort,
                "analysis_set": analysis_set,
                "k": k,
                "paired_datasets": len(dataset_subject_values),
                "paired_subjects": int(sum(len(v) for v in dataset_subject_values.values())),
                "delta_f1_macro": float(dataset_means.mean()),
                "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
                "estimand": "dataset-balanced mean paired subject macro-F1 delta",
            })
    return sorted(
        output,
        key=lambda row: (
            row["regime"], row["label_mode"], row["subject_relation"],
            row["configuration_relation"], row["analysis_set"], row["k"], row["comparator"]
        ),
    )


def _markdown(aggregates: list[dict]) -> str:
    lines = [
        "# Matched adaptation results", "",
        "`k` is the number of independent enrolled executions per candidate. External-model",
        "1-NN, prototype, and ridge controls use one equally weighted pooled vector per enrolled",
        "execution and require no gradient fitting. HALO's support comparator consumes one pooled",
        "row per enrolled execution and adds a learned residual to its closed-form support vote.", "",
    ]
    panels = [
        ("Semantic zero-shot", lambda row: (
            row["label_mode"] == "coherent" and row["k"] == 0
            and row["model"] in NATIVE_ZERO_SHOT_MODELS
            and row["method"] == "zero_shot"
        )),
        ("Coherent adaptation comparison", lambda row: (
            row["label_mode"] == "coherent" and row["k"] > 0
            and (row["method"] in {"nearest", "prototype", "ridge"}
                 or (row["model"] == "halo_compare" and row["method"] == "support_comparator"))
        )),
        ("Random-label binding", lambda row: (
            row["label_mode"] == "random_alias" and row["k"] > 0
            and (row["method"] in {"nearest", "prototype", "ridge"}
                 or (row["model"] == "halo_compare" and row["method"] == "support_comparator"))
        )),
    ]
    for title, include in panels:
        selected = sorted(
            (row for row in aggregates if include(row)),
            key=lambda row: (
                row["analysis_set"], row["subject_relation"], row["configuration_relation"],
                row["regime"], row["model"], row["method"], row["k"],
            ),
        )
        lines.extend([
            f"## {title}", "",
            "| analysis set | subject relation | configuration relation | regime | model | method | k | macro F1 | datasets |",
            "|---|---|---|---|---|---|---:|---:|---:|",
        ])
        for row in selected:
            method = (
                "support comparator"
                if row["model"] == "halo_compare" and row["method"] == "support_comparator"
                else "1-NN" if row["method"] == "nearest" else row["method"]
            )
            lines.append(
                f"| {row['analysis_set']} | {row['subject_relation']} | "
                f"{row['configuration_relation']} | {row['regime']} | {row['model']} | "
                f"{method} | {row['k']} | "
                f"{row['f1_macro']:.2f} | {row['datasets']} |"
            )
        if not selected:
            lines.append("| - | - | - | - | - | - | - | - | - |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest, validate_grids=True)
    rows, subjects = load_rows(args.inputs, manifest)
    aggregates = dataset_macro(rows)
    paired = paired_deltas(subjects)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("cells.csv", rows), ("dataset_macro.csv", aggregates)):
        path = args.out_dir / name
        fields = sorted({key for row in values for key in row})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(values)
    (args.out_dir / "paired_deltas.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "tables.md").write_text(_markdown(aggregates) + "\n")
    print(f"assembled {len(rows)} cells from {len(args.inputs)} artifacts -> {args.out_dir}")


if __name__ == "__main__":
    main()
