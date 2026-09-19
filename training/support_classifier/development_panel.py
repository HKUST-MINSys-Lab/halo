"""Evaluate support-classifier checkpoints on a fixed internal robustness panel.

The panel uses only subject-held-out splits of the active training roster. Every condition reuses
the same deterministic support episodes; only the recording transform changes. This is a fast
development diagnostic, not a replacement for sealed evaluation.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from halo.paths import CACHE_DIR
from data.scripts.augmentations import AugmentationConfig
from data.scripts.curate.deployment_policy import SUPERVISED_HEAD_TRAIN_DATASETS
from model.blocks import AttentionSpec
from model.support.factory import build_classifier_from_blob
from model.tokenizer.sensor_tokens import CONDITIONING_SCHEMA_V2, LEGACY_CONDITIONING_SCHEMA
from training.support_classifier.collate import SupportCollate
from training.support_classifier.corpus import support_corpus_from_index
from training.support_classifier.sampling import (
    DEFAULT_ACQUISITION_MIX,
    DEFAULT_ENROLLMENT_K,
    DEFAULT_ENROLLMENT_MIX,
    DEFAULT_LABEL_SUBSET,
    DEFAULT_PARTIAL_COVERAGE,
    DEFAULT_P_GT_PRESENT,
    DEFAULT_QUERIES_PER_SUPPORT_SET,
    DEFAULT_SAME_SUBJECT_PROBABILITY,
    DEFAULT_VARIABLE_SUPPORT_PROBABILITY,
    DEFAULT_WINDOWS_PER_EXECUTION,
)
from training.support_classifier.train import draw_kwargs_from_args, make_label_text, validate
from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain_data import CorpusIndex, MultiResolutionCollate, PretrainDataset


def _value(blob: dict, name: str, default):
    return blob.get("trajectory", {}).get(name, blob.get("args", {}).get(name, default))


def _conditioning_schema(blob: dict) -> str:
    configured = blob.get("config", {}).get("conditioning_schema")
    if configured is not None:
        return str(configured)
    return (
        CONDITIONING_SCHEMA_V2
        if any(key.startswith("structured_conditioner.") for key in blob.get("encoder", {}))
        else LEGACY_CONDITIONING_SCHEMA
    )


def _draw_kwargs(blob: dict) -> dict:
    return draw_kwargs_from_args(SimpleNamespace(
        classifier="residual",
        p_gt_present=_value(blob, "p_gt_present", DEFAULT_P_GT_PRESENT),
        p_mask_candidate=_value(blob, "p_mask_candidate", 0.25),
        p_mask_gt=_value(blob, "p_mask_gt", 0.10),
        same_subject_probability=_value(
            blob, "same_subject_probability", DEFAULT_SAME_SUBJECT_PROBABILITY,
        ),
        label_subset=tuple(_value(blob, "label_subset", DEFAULT_LABEL_SUBSET)),
        mode=_value(blob, "mode", "compatible"),
        enrollment_k=tuple(_value(blob, "enrollment_k", DEFAULT_ENROLLMENT_K)),
        acquisition_mix=tuple(_value(blob, "acquisition_mix", DEFAULT_ACQUISITION_MIX)),
        enrollment_mix=tuple(_value(blob, "enrollment_mix", DEFAULT_ENROLLMENT_MIX)),
        partial_coverage=tuple(_value(blob, "partial_coverage", DEFAULT_PARTIAL_COVERAGE)),
        variable_support_probability=_value(
            blob, "variable_support_probability", DEFAULT_VARIABLE_SUPPORT_PROBABILITY,
        ),
        counterfactual_enrollment_probability=0.0,
        queries_per_support_set=_value(
            blob, "queries_per_support_set", DEFAULT_QUERIES_PER_SUPPORT_SET,
        ),
        windows_per_execution=_value(
            blob, "windows_per_execution", DEFAULT_WINDOWS_PER_EXECUTION,
        ),
    ))


def _dataset(index: CorpusIndex, blob: dict, *, rate_p: float, modality_p: float) -> PretrainDataset:
    augmented = rate_p > 0 or modality_p > 0
    return PretrainDataset(
        index,
        index.val,
        augment=augmented,
        two_view=False,
        augmentation_config=(
            AugmentationConfig.phase_a(rate_p=rate_p, channel_dropout_p=modality_p)
            if augmented else None
        ),
        neutral_acquisition_text=bool(_value(blob, "neutral_acquisition_text", False)),
        conditioning_schema=_conditioning_schema(blob),
        multi_device_probability=float(_value(blob, "multi_device_probability", 0.5)),
        max_devices=int(_value(blob, "max_devices", 4)),
    )


def evaluate_checkpoint(
    path: Path, *, device: torch.device, support_sets: int,
    panel_seeds: tuple[int, ...],
) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    seed = int(_value(blob, "data_seed", 20260901))
    window_seconds = float(_value(blob, "window_seconds", 8.0))
    resolutions = tuple(float(value) for value in _value(
        blob, "resolutions", blob["config"].get("eval_resolutions", (0.5, 1.0, 2.0, 4.0)),
    ))
    index = CorpusIndex(
        datasets=SUPERVISED_HEAD_TRAIN_DATASETS,
        alignment="native",
        max_per_stream=_value(blob, "max_per_stream", None),
        seed=seed,
        window_seconds=window_seconds,
    )
    corpus = support_corpus_from_index(index, split="val")
    collate = SupportCollate(MultiResolutionCollate(fixed_patch_seconds=resolutions))
    encoder = build_encoder(blob, device, training=False)
    classifier, _ = build_classifier_from_blob(blob, device=device)
    text = make_label_text(corpus.all_labels, device)
    draw_kwargs = _draw_kwargs(blob)
    clean = _dataset(index, blob, rate_p=0.0, modality_p=0.0)
    rate = _dataset(index, blob, rate_p=0.5, modality_p=0.0)
    modality = _dataset(index, blob, rate_p=0.0, modality_p=0.5)

    def forced(*, acquisition=None, enrollment=None) -> dict:
        value = dict(draw_kwargs)
        if acquisition is not None:
            names = ("compatible", "cross_placement", "cross_dataset")
            value["acquisition_mix"] = tuple(float(name == acquisition) for name in names)
            # A zero-support set has no acquisition relation. Exclude it from an acquisition panel
            # so every sampled row actually measures the named condition.
            value["enrollment_mix"] = (0.5, 0.5, 0.0)
        if enrollment is not None:
            names = ("complete", "partial", "zero")
            value["enrollment_mix"] = tuple(float(name == enrollment) for name in names)
        return value

    conditions = {
        "clean_mixed": (clean, draw_kwargs),
        "rate_resample_mixture": (rate, draw_kwargs),
        "gyro_dropout_mixture": (modality, draw_kwargs),
        "clean_acquisition_compatible": (
            clean, forced(acquisition="compatible"),
        ),
        "clean_acquisition_cross_placement": (
            clean, forced(acquisition="cross_placement"),
        ),
        "clean_acquisition_cross_dataset": (
            clean, forced(acquisition="cross_dataset"),
        ),
        "clean_enrollment_complete": (clean, forced(enrollment="complete")),
        "clean_enrollment_partial": (clean, forced(enrollment="partial")),
        "clean_enrollment_zero": (clean, forced(enrollment="zero")),
    }
    replicates: dict[str, dict[str, dict]] = {}
    for panel_seed in panel_seeds:
        seed_metrics = {}
        for name, (dataset, panel_draw_kwargs) in conditions.items():
            seed_metrics[name] = validate(
                encoder=encoder,
                classifier=classifier,
                classifier_mode="residual",
                corpus=corpus,
                dataset=dataset,
                collate=collate,
                text_of=text,
                device=device,
                episodes_count=support_sets,
                episodes_per_step=int(_value(blob, "episodes_per_step", 4)),
                seed=seed + 91_003 + int(panel_seed),
                draw_kwargs=panel_draw_kwargs,
                executor=None,
                deployment_matched=True,
            )
        replicates[str(panel_seed)] = seed_metrics

    metrics: dict[str, dict[str, float]] = {}
    metrics_sd: dict[str, dict[str, float]] = {}
    for condition in conditions:
        rows = [replicates[str(panel_seed)][condition] for panel_seed in panel_seeds]
        keys = sorted(set.intersection(*(set(row) for row in rows)))
        metrics[condition] = {}
        metrics_sd[condition] = {}
        for key in keys:
            values = [row[key] for row in rows]
            if not values or not all(isinstance(value, (int, float)) for value in values):
                continue
            metrics[condition][key] = float(statistics.fmean(values))
            metrics_sd[condition][key] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return {
        "checkpoint": str(path),
        "step": int(blob.get("step", -1)),
        "source_commit": blob.get("git", {}).get("commit"),
        "support_sets_per_condition": support_sets,
        "panel_seed_offsets": list(panel_seeds),
        "window_seconds": window_seconds,
        "resolutions": list(resolutions),
        "panel_contracts": {
            "mixed": "default acquisition and enrollment mixtures",
            "acquisition": "named acquisition regime; 50/50 complete/partial enrollment",
            "enrollment": "named enrollment regime; default acquisition mixture",
        },
        "metrics": metrics,
        "metrics_sd": metrics_sd,
        "replicates": replicates,
    }


def _metric(row: dict, key: str, sd_row: dict | None = None) -> str:
    value = row.get(key)
    if value is None:
        return "n/a"
    if sd_row is None or key not in sd_row:
        return f"{100 * float(value):.2f}"
    return f"{100 * float(value):.2f} +/- {100 * float(sd_row[key]):.2f}"


def _paired_arm_deltas(runs: list[dict]) -> list[dict]:
    """Arm-minus-reference deltas on identical panel draws, never unpaired endpoint ranks."""
    if len(runs) < 2:
        return []
    reference = runs[0]
    keys = (
        "validation/enrolled_dataset_macro_f1",
        "validation/zero_shot_dataset_macro_f1",
        "validation/scenario/comparison/all/enrolled/net_gain",
    )
    output = []
    for run in runs[1:]:
        common_seeds = sorted(set(reference["replicates"]) & set(run["replicates"]))
        for condition in sorted(set(reference["metrics"]) & set(run["metrics"])):
            for key in keys:
                deltas = []
                for panel_seed in common_seeds:
                    left = reference["replicates"][panel_seed][condition].get(key)
                    right = run["replicates"][panel_seed][condition].get(key)
                    if left is not None and right is not None:
                        deltas.append(float(right) - float(left))
                if not deltas:
                    continue
                output.append({
                    "reference": reference["checkpoint"],
                    "checkpoint": run["checkpoint"],
                    "condition": condition,
                    "metric": key,
                    "n_panel_seeds": len(deltas),
                    "mean_delta": float(statistics.fmean(deltas)),
                    "sd_delta": float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0,
                    "deltas": deltas,
                })
    return output


def _markdown(report: dict) -> str:
    lines = [
        "# Internal Classifier Development Panel",
        "",
        "Subject-held-out development data only. Values are percentages. Clean/rate/dropout "
        "reuse one episode panel; each condition-specific panel is fixed across checkpoints.",
        "",
        "| checkpoint | condition | enrolled macro-F1 | zero-shot macro-F1 | soft-vote acc. | "
        "classifier acc. | rescue | overturn | net gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    prefix = "validation/scenario/comparison/all/enrolled"
    for run in report["runs"]:
        checkpoint = Path(run["checkpoint"]).parent.name
        for condition, row in run["metrics"].items():
            sd_row = run["metrics_sd"].get(condition, {})
            lines.append(
                f"| {checkpoint} @ {run['step']} | {condition} | "
                f"{_metric(row, 'validation/enrolled_dataset_macro_f1', sd_row)} | "
                f"{_metric(row, 'validation/zero_shot_dataset_macro_f1', sd_row)} | "
                f"{_metric(row, prefix + '/soft_vote_accuracy', sd_row)} | "
                f"{_metric(row, prefix + '/classifier_accuracy', sd_row)} | "
                f"{_metric(row, prefix + '/rescue_rate', sd_row)} | "
                f"{_metric(row, prefix + '/overturn_rate', sd_row)} | "
                f"{_metric(row, prefix + '/net_gain', sd_row)} |"
            )
    lines.extend([
        "",
        "Cells are mean +/- sample SD across the declared panel seeds. The `soft vote` column is "
        "the temperature-0.07 differentiable support vote, not literal cosine 1-NN. `rescue` means "
        "soft vote wrong and classifier correct. `overturn` means soft vote correct "
        "and classifier wrong. `net gain = classifier accuracy - soft-vote accuracy` on enrolled "
        "episodes only.",
    ])
    if report.get("paired_arm_deltas"):
        lines.extend([
            "", "## Paired Arm Deltas", "",
            "Each value is arm minus the first checkpoint on the identical panel draw.", "",
            "| checkpoint | condition | metric | mean delta | SD | panel seeds |",
            "|---|---|---|---:|---:|---:|",
        ])
        for row in report["paired_arm_deltas"]:
            lines.append(
                f"| {Path(row['checkpoint']).parent.name} | {row['condition']} | "
                f"{row['metric']} | {100 * row['mean_delta']:.2f} | "
                f"{100 * row['sd_delta']:.2f} | {row['n_panel_seeds']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--support-sets", type=int, default=64)
    parser.add_argument(
        "--panel-seeds", type=int, nargs="+", default=[0, 1, 2],
        help="deterministic panel-seed offsets; at least three are required for decisions",
    )
    parser.add_argument(
        "--out", type=Path,
        default=CACHE_DIR / "evaluations" / "development_panel_20260917",
    )
    args = parser.parse_args()
    if args.support_sets < 1 or args.support_sets > 256:
        parser.error("support-sets must be in [1, 256]")
    if len(set(args.panel_seeds)) < 3:
        parser.error("at least three distinct panel seeds are required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    runs = [
        evaluate_checkpoint(
            path, device=device, support_sets=args.support_sets,
            panel_seeds=tuple(dict.fromkeys(args.panel_seeds)),
        )
        for path in args.checkpoints
    ]
    report = {
        "protocol": "internal-classifier-development-panel-v2-20260918",
        "device": str(device),
        "runs": runs,
        "paired_arm_deltas": _paired_arm_deltas(runs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.out.with_suffix(".md").write_text(_markdown(report))
    print(_markdown(report))


if __name__ == "__main__":
    main()
