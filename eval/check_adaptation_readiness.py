"""Fail-loud readiness audit for the matched zero-shot and enrollment protocol."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import baselines
from baselines.base import BaselineAdapter
from eval.enrollment_protocol import ACTION_REGIMES, PROTOCOL_NAME, iter_cells, load_manifest


PAPER_MODELS = ("halo_compare", "harnet", "unimts", "imagebind", "normwear")
NATIVE_ZERO_SHOT_MODELS = {"halo_compare", "unimts", "imagebind", "normwear"}


def audit(manifest_path: Path, baseline_names=PAPER_MODELS) -> dict:
    manifest = load_manifest(manifest_path, validate_grids=True)
    blockers, warnings = [], []
    if manifest.get("protocol_name") != PROTOCOL_NAME:
        blockers.append(
            f"protocol is {manifest.get('protocol_name')!r}, expected {PROTOCOL_NAME!r}"
        )
    if len(manifest["seeds"]) < 5:
        blockers.append("fewer than five serialized support seeds")
    if manifest["support_counts"][:5] != [0, 1, 2, 4, 8]:
        blockers.append("main k curve is not exactly 0,1,2,4,8")

    zero_datasets = {
        cell["dataset"] for _, cell in iter_cells(manifest, kinds=["zero_shot"])
        if cell["status"] == "ok"
    }
    positive_datasets = {
        cell["dataset"] for _, cell in iter_cells(manifest, kinds=["enrollment"])
        if cell["status"] == "ok"
    }
    expected = set(manifest["datasets"])
    if zero_datasets != expected:
        blockers.append(f"missing zero-shot datasets: {sorted(expected - zero_datasets)}")
    if positive_datasets != expected:
        warnings.append(
            "no valid positive-k relation for datasets: "
            f"{sorted(expected - positive_datasets)}"
        )

    baseline_status = {}
    for name in baseline_names:
        adapter = baselines.REGISTRY.get(name)
        if adapter is None:
            blockers.append(f"baseline adapter is not registered: {name}")
            continue
        features_overridden = type(adapter).window_features is not BaselineAdapter.window_features
        direct_candidate_override = (
            type(adapter).predict_candidates is not BaselineAdapter.predict_candidates
            or adapter.tier in {"conse", "cosine"}
        )
        cached_prediction = (
            type(adapter).predict_candidates_from_features
            is not BaselineAdapter.predict_candidates_from_features
        )
        candidates_overridden = direct_candidate_override or cached_prediction
        native_enrollment = adapter.supports_native_enrollment()
        native_zero_shot = adapter.supports_native_zero_shot()
        reported_zero_shot = name in NATIVE_ZERO_SHOT_MODELS
        baseline_status[name] = {
            "adapter": f"{type(adapter).__module__}.{type(adapter).__name__}",
            "frozen_features": features_overridden,
            "reported_at_zero_shot": reported_zero_shot,
            "native_zero_shot": native_zero_shot,
            "candidate_override": candidates_overridden,
            "cached_feature_prediction": cached_prediction,
            "native_enrollment": native_enrollment,
        }
        try:
            artifact_paths = (
                adapter.evaluation_artifacts(None)
                if native_zero_shot or native_enrollment
                else adapter.feature_artifacts(None)
            )
        except (AttributeError, KeyError, TypeError):
            artifact_paths = {}
        baseline_status[name]["artifacts"] = {
            artifact_name: {
                "path": str(path), "exists": Path(path).is_file(),
            }
            for artifact_name, path in artifact_paths.items()
        }
        missing_artifacts = [
            artifact_name for artifact_name, record in baseline_status[name]["artifacts"].items()
            if not record["exists"]
        ]
        if missing_artifacts:
            blockers.append(f"{name}: missing evaluation artifacts: {missing_artifacts}")
        if not features_overridden:
            blockers.append(f"{name}: no frozen window feature interface")
        if native_zero_shot != reported_zero_shot:
            blockers.append(
                f"{name}: native zero-shot capability disagrees with the report roster"
            )
        if reported_zero_shot and not candidates_overridden:
            blockers.append(f"{name}: cannot score the manifest candidate roster")
        if reported_zero_shot and not cached_prediction:
            blockers.append(f"{name}: k=0 would require a second sensor encoding pass")
        if name == "halo_compare":
            checkpoint = getattr(importlib.import_module(type(adapter).__module__), "_CKPT", None)
            baseline_status[name]["checkpoint"] = str(checkpoint) if checkpoint else None
            baseline_status[name]["checkpoint_exists"] = bool(
                checkpoint is not None and Path(checkpoint).is_file()
            )
            if not native_enrollment:
                blockers.append("halo_compare: current model has no native enrollment path")
            if not baseline_status[name]["checkpoint_exists"]:
                blockers.append(
                    "halo_compare: checkpoint is missing; set HALO_COMPARE_CKPT to the trained "
                    "comparison checkpoint"
                )

    enrollment = [cell for _, cell in iter_cells(manifest, kinds=["enrollment"])]
    ceilings = {}
    secondary_ready = 0
    for cell in enrollment:
        ceilings[str(cell["support_ceiling"])] = ceilings.get(str(cell["support_ceiling"]), 0) + 1
        secondary_ready += int(cell.get("secondary_high_support", {}).get("status") == "ok")
    common_main = [
        cell for cell in enrollment
        if cell["status"] == "ok" and int(cell["support_ceiling"]) >= 8
    ]
    common_relations = {}
    for cell in common_main:
        key = f"{cell['subject_relation']}/{cell['configuration_relation']}"
        record = common_relations.setdefault(key, {"cells": 0, "datasets": set()})
        record["cells"] += 1
        record["datasets"].add(cell["dataset"])
    common_relations = {
        key: {"cells": value["cells"], "datasets": sorted(value["datasets"])}
        for key, value in common_relations.items()
    }
    primary = common_relations.get("cross_subject/same_configuration")
    if not primary:
        blockers.append("no fixed-cohort cross-subject/same-configuration k=1..8 curve")
    same_subject = common_relations.get("same_subject/same_configuration")
    if same_subject is None or len(same_subject["datasets"]) < 3:
        warnings.append(
            "same-subject/same-configuration k=1..8 is underpowered; do not use it as a "
            "multi-dataset headline"
        )
    unattributed = sorted({
        cell["dataset"] for cell in enrollment if cell["subject_relation"] == "unattributed"
    })
    return {
        "ready": not blockers,
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "datasets": manifest["datasets"],
        "action_regimes": {key: list(value) for key, value in ACTION_REGIMES.items()},
        "zero_shot_datasets": sorted(zero_datasets),
        "positive_k_datasets": sorted(positive_datasets),
        "positive_relation_support_ceilings": ceilings,
        "secondary_k16_relations": secondary_ready,
        "fixed_main_curve_relations": common_relations,
        "subject_unattributed_datasets": unattributed,
        "baseline_status": baseline_status,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--baselines", nargs="*", default=list(PAPER_MODELS))
    args = parser.parse_args()
    report = audit(args.manifest, baseline_names=tuple(args.baselines))
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
