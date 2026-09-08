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


def _unverified_execution_support(cell: dict) -> bool:
    """Reject missing execution provenance, including legacy TNDA manifests."""
    return cell.get("kind") == "enrollment" and (
        cell.get("execution_identity_known") is False or cell.get("dataset") == "tnda_har"
    )


def _analysis_set(result: dict, cell: dict) -> str:
    if result["kind"] == "zero_shot":
        return "zero_shot"
    if _unverified_execution_support(cell):
        return "unsupported_execution_identity"
    if result.get("cohort", "main") == "secondary_high_support":
        return "secondary_high_support"
    if int(cell["support_ceiling"]) >= MAIN_CURVE_MAX_K:
        return "main_common_k1_8"
    return "supplemental_partial_coverage"


def _model_identity(payload: dict) -> str:
    """``baseline@variant`` when the run carries a variant, else the registry name."""
    return str(payload.get("model") or payload["baseline"])


def _external_rows(payload: dict, manifest: dict) -> tuple[list[dict], list[dict]]:
    model = _model_identity(payload)
    rows, subjects = [], []
    for key, result in payload["results"].items():
        if result.get("status"):
            continue
        cell_id = _cell_id(key, result["kind"])
        if cell_id not in manifest["cells"]:
            raise ValueError(f"result references unknown manifest cell: {cell_id}")
        cell = manifest["cells"][cell_id]
        if _unverified_execution_support(cell):
            continue
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
    seen_models: dict[str, Path] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        identity = _model_identity(payload) if "baseline" in payload else None
        if identity is not None:
            # Two artifacts with one identity would be averaged into a single number by
            # ``dataset_macro`` and silently overwrite each other in ``paired_deltas``. The
            # trained run and its step-0 control are the case this exists for: run the control
            # with ``--variant step0`` so it assembles as ``halo_compare@step0``.
            if identity in seen_models:
                raise ValueError(
                    f"{path}: model identity {identity!r} already loaded from "
                    f"{seen_models[identity]}; give one of the runs a --variant"
                )
            seen_models[identity] = path
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
        if payload.get("git_dirty") is not False or not payload.get("git_commit"):
            raise ValueError(f"{path}: result was produced from a dirty worktree or lacks a source commit")
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


CONDITION_FIELDS = (
    "regime", "label_mode", "subject_relation", "configuration_relation", "cohort",
    "analysis_set", "k",
)


def _dataset_balanced_delta(
    by_dataset_subject: dict[str, dict[str, list[float]]],
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, list[float]]:
    """Mean over datasets of the mean subject delta, with a bootstrap that resamples subjects
    within each fixed dataset so a large-subject dataset cannot dominate the interval."""
    values = {
        dataset: np.asarray([np.mean(v) for v in by_subject.values()], dtype=np.float64)
        for dataset, by_subject in by_dataset_subject.items()
    }
    point = float(np.mean([v.mean() for v in values.values()]))
    draws = np.stack([
        rng.choice(v, size=(samples, len(v)), replace=True).mean(1) for v in values.values()
    ]).mean(0)
    return point, [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _paired_subject_rows(
    indexed: dict, target_key: tuple[str, str], comparator_key: tuple[str, str],
    rng: np.random.Generator, samples: int,
) -> list[dict]:
    target, values = indexed[target_key], indexed[comparator_key]
    common = sorted(set(target) & set(values))
    by_condition = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key in common:
        condition = key[:7]
        dataset, subject = key[-2:]
        by_condition[condition][dataset][subject].append(target[key] - values[key])
    output = []
    for condition, by_dataset_subject in by_condition.items():
        delta, ci95 = _dataset_balanced_delta(by_dataset_subject, rng, samples)
        (regime, label_mode, subject_relation, configuration_relation,
         cohort, analysis_set, k) = condition
        output.append({
            "target": f"{target_key[0]}/{target_key[1]}",
            "comparator": f"{comparator_key[0]}/{comparator_key[1]}",
            "regime": regime,
            "label_mode": label_mode,
            "subject_relation": subject_relation,
            "configuration_relation": configuration_relation,
            "cohort": cohort,
            "analysis_set": analysis_set,
            "k": k,
            "paired_datasets": len(by_dataset_subject),
            "paired_subjects": int(sum(len(v) for v in by_dataset_subject.values())),
            "delta_f1_macro": delta,
            "ci95": ci95,
            "estimand": "dataset-balanced mean paired subject macro-F1 delta",
        })
    return output


def _index_subjects(subject_rows: list[dict]) -> dict:
    indexed = defaultdict(dict)
    for row in subject_rows:
        key = (
            row["regime"], row["label_mode"], row["subject_relation"],
            row["configuration_relation"], row["cohort"], row["analysis_set"],
            row["k"], row["cell"], row["seed"], row["dataset"], row["subject"],
        )
        indexed[(row["model"], row["method"])][key] = row["f1_macro"]
    return indexed


def paired_deltas(subject_rows: list[dict], samples: int = 5_000) -> list[dict]:
    """Dataset-balanced paired subject bootstrap of HALO against every other model and readout,
    within each evaluation condition."""
    indexed = _index_subjects(subject_rows)
    targets = sorted(key for key in indexed
                     if key[0].split("@", 1)[0] == TARGET_KEY[0]
                     and key[1] == TARGET_KEY[1])
    rng = np.random.default_rng(20260817)
    output = []
    for target in targets:
        for comparator in sorted(indexed):
            if comparator != target:
                output.extend(_paired_subject_rows(indexed, target, comparator, rng, samples))
    return sorted(
        output,
        key=lambda row: (
            row["target"], row["regime"], row["label_mode"], row["subject_relation"],
            row["configuration_relation"], row["analysis_set"], row["k"], row["comparator"]
        ),
    )


def _split_identity(model: str) -> tuple[str, str | None]:
    base, _, variant = model.partition("@")
    return base, (variant or None)


def variant_deltas(
    rows: list[dict], subject_rows: list[dict], samples: int = 5_000,
) -> list[dict]:
    """Every ``base@variant`` run against its own base, per readout and condition.

    This is one table for two questions. A perturbation variant (``harnet@query_orientation``)
    gives the heterogeneity-axis retention: variant minus matched, per model. A control variant
    (``halo_compare@step0``) gives the paired step-0 gain with the sign flipped. The point is
    computed from cell rows (so zero-shot and subject-unattributed cells are included); the
    Subject-macro deltas and their intervals are separate fields, not intervals for the cell point.
    """
    by_model: dict[str, dict[tuple, float]] = defaultdict(dict)
    for row in rows:
        key = (row["method"], *(row[field] for field in CONDITION_FIELDS),
               row["dataset"], row["cell"], row["seed"])
        by_model[row["model"]][key] = float(row["f1_macro"])
    indexed = _index_subjects(subject_rows)
    rng = np.random.default_rng(20260905)
    output = []
    for model in sorted(by_model):
        base, variant = _split_identity(model)
        if variant is None or base not in by_model:
            continue
        common = sorted(set(by_model[model]) & set(by_model[base]))
        grouped: dict[tuple, dict[str, list[tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for key in common:
            condition, dataset = key[:1 + len(CONDITION_FIELDS)], key[-3]
            grouped[condition][dataset].append((by_model[base][key], by_model[model][key]))
        for condition, by_dataset in grouped.items():
            method = condition[0]
            base_means = [np.mean([b for b, _ in pairs]) for pairs in by_dataset.values()]
            variant_means = [np.mean([v for _, v in pairs]) for pairs in by_dataset.values()]
            record = {
                "model": model, "base": base, "variant": variant, "method": method,
                **dict(zip(CONDITION_FIELDS, condition[1:])),
                "datasets": len(by_dataset),
                "base_f1_macro": float(np.mean(base_means)),
                "variant_f1_macro": float(np.mean(variant_means)),
                "delta_f1_macro": float(np.mean(variant_means) - np.mean(base_means)),
                "ci95": None,
                "subject_delta_f1_macro": None,
                "subject_ci95": None,
                "estimand": "dataset-balanced mean macro-F1, variant minus base",
            }
            target_key, base_key = (model, method), (base, method)
            if target_key in indexed and base_key in indexed:
                paired = [
                    row for row in _paired_subject_rows(indexed, target_key, base_key, rng, samples)
                    if tuple(row[field] for field in CONDITION_FIELDS) == tuple(condition[1:])
                ]
                if paired:
                    record["subject_delta_f1_macro"] = paired[0]["delta_f1_macro"]
                    record["subject_ci95"] = paired[0]["ci95"]
                    record["subject_estimand"] = paired[0]["estimand"]
                    record["paired_subjects"] = paired[0]["paired_subjects"]
            output.append(record)
    return sorted(
        output,
        key=lambda row: (row["base"], row["variant"], row["method"],
                         *(str(row[field]) for field in CONDITION_FIELDS)),
    )


def _variant_markdown(variants: list[dict]) -> list[str]:
    if not variants:
        return []
    lines = [
        "## Variant deltas (variant minus base)", "",
        "A perturbation variant reads as retention under one heterogeneity axis; a step-0 variant",
        "reads as the training gain with the sign flipped. Cell deltas use dataset-macro F1.",
        "Subject deltas and their 95% intervals are reported separately: they use dataset-balanced",
        "paired subject macro-F1 and are not confidence intervals for the cell delta.", "",
        "| base | variant | method | analysis set | subject relation | configuration relation | "
        "regime | label mode | k | base | variant | cell delta | subject delta | subject ci95 | datasets |",
        "|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in variants:
        interval = row["subject_ci95"]
        ci = "" if interval is None else f"[{interval[0]:.2f}, {interval[1]:.2f}]"
        subject_delta = ("" if row["subject_delta_f1_macro"] is None
                         else f"{row['subject_delta_f1_macro']:+.2f}")
        lines.append(
            f"| {row['base']} | {row['variant']} | {row['method']} | {row['analysis_set']} | "
            f"{row['subject_relation']} | {row['configuration_relation']} | {row['regime']} | "
            f"{row['label_mode']} | {row['k']} | {row['base_f1_macro']:.2f} | {row['variant_f1_macro']:.2f} | "
            f"{row['delta_f1_macro']:+.2f} | {subject_delta} | {ci} | {row['datasets']} |"
        )
    lines.append("")
    return lines


def _markdown(aggregates: list[dict], variants: list[dict] | None = None) -> str:
    lines = [
        "# Matched adaptation results", "",
        "`k` is the number of independent enrolled executions per candidate. External-model",
        "1-NN, prototype, and ridge controls use one equally weighted pooled vector per enrolled",
        "execution and require no gradient fitting. HALO's support comparator consumes one pooled",
        "row per enrolled execution and adds a learned residual to its closed-form support vote.", "",
        "Scores pool query-window predictions within each cell, then average cells/seeds within",
        "each dataset and give datasets equal weight. They are not subject-macro scores.",
        "Paired subject-macro differences and their bootstrap intervals are a separate estimand.",
        "Available-coverage averages below must not be ranked when coverage differs; use the",
        "common-coverage table and per-dataset results for comparisons.", "",
    ]
    panels = [
        ("Semantic zero-shot", lambda row: (
            row["label_mode"] == "coherent" and row["k"] == 0
            and row["model"].split("@", 1)[0] in NATIVE_ZERO_SHOT_MODELS
            and row["method"] == "zero_shot"
        )),
        ("Coherent adaptation comparison", lambda row: (
            row["label_mode"] == "coherent" and row["k"] > 0
            and (row["method"] in {"nearest", "prototype", "ridge"}
                 or (row["model"].split("@", 1)[0] == "halo_compare" and row["method"] == "support_comparator"))
        )),
        ("Random-label binding", lambda row: (
            row["label_mode"] == "random_alias" and row["k"] > 0
            and (row["method"] in {"nearest", "prototype", "ridge"}
                 or (row["model"].split("@", 1)[0] == "halo_compare" and row["method"] == "support_comparator"))
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
                if row["model"].split("@", 1)[0] == "halo_compare" and row["method"] == "support_comparator"
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
    lines.extend(_variant_markdown(variants or []))
    return "\n".join(lines)


def zero_shot_coverage(rows: list[dict], models: list[str]) -> tuple[list[dict], str]:
    """Compare exactly the same stream/cell roster, not just matching dataset names."""
    eligible = [r for r in rows if r["k"] == 0 and r["method"] == "zero_shot"
                and r["label_mode"] == "coherent" and r["model"] in models]
    cells = {model: {r["cell"] for r in eligible if r["model"] == model} for model in models}
    common = set.intersection(*cells.values()) if cells else set()
    matched = [r for r in eligible if r["cell"] in common]
    lines = ["## Zero-shot coverage", "",
             "Unsupported cells are unavailable, not zero scores. The common comparison uses",
             "the intersection of stream cells across all supplied native zero-shot models.", "",
             "| model | available datasets | available cells | common cells |",
             "|---|---|---:|---:|"]
    for model in sorted(models):
        datasets = sorted({r["dataset"] for r in eligible if r["model"] == model})
        lines.append(f"| {model} | {', '.join(datasets) or 'none'} | {len(cells[model])} | {len(common)} |")
    lines += ["", "### Common-coverage zero-shot scores", "",
              "| model | regime | macro F1 | datasets |", "|---|---|---:|---:|"]
    for r in dataset_macro(matched):
        lines.append(f"| {r['model']} | {r['regime']} | {r['f1_macro']:.2f} | {r['datasets']} |")
    if not matched:
        lines.append("No common zero-shot cells; no matched aggregate can be reported.")
    return matched, "\n".join(lines)


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
    variants = variant_deltas(rows, subjects)
    native_models = sorted({_model_identity(payload) for path in args.inputs
                            if (payload := json.loads(path.read_text()))["baseline"]
                            in NATIVE_ZERO_SHOT_MODELS})
    common_rows, coverage = zero_shot_coverage(rows, native_models)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Reuse the same aggregation with one dataset at a time for inspectable breakdowns.
    per_dataset = [{**r, "dataset": dataset}
                   for dataset in sorted({r["dataset"] for r in rows})
                   for r in dataset_macro([r for r in rows if r["dataset"] == dataset])]
    for name, values in (("cells.csv", rows), ("dataset_macro.csv", aggregates),
                         ("per_dataset.csv", per_dataset),
                         ("zero_shot_common_cells.csv", common_rows)):
        path = args.out_dir / name
        fields = sorted({key for row in values for key in row})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(values)
    (args.out_dir / "paired_deltas.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "variant_deltas.json").write_text(
        json.dumps(variants, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "tables.md").write_text(_markdown(aggregates, variants) + "\n" + coverage + "\n")
    print(f"assembled {len(rows)} cells from {len(args.inputs)} artifacts -> {args.out_dir}")


if __name__ == "__main__":
    main()
