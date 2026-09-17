"""Evaluate support-classifier checkpoints on a fixed internal robustness panel.

The panel uses only subject-held-out splits of the active training roster. Every condition reuses
the same deterministic support episodes; only the recording transform changes. This is a fast
development diagnostic, not a replacement for sealed evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from data.scripts.augmentations import AugmentationConfig
from data.scripts.curate.deployment_policy import SUPERVISED_HEAD_TRAIN_DATASETS
from model.blocks import AttentionSpec
from model.support.residual_classifier import ResidualClassifierConfig, build_support_classifier
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
        multi_device_probability=float(_value(blob, "multi_device_probability", 0.5)),
        max_devices=int(_value(blob, "max_devices", 4)),
    )


def evaluate_checkpoint(
    path: Path, *, device: torch.device, support_sets: int,
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
    classifier = build_support_classifier(
        AttentionSpec(**blob["attention_spec"]),
        ResidualClassifierConfig(**blob["classifier_config"]),
    ).to(device).eval()
    classifier.load_state_dict(blob["classifier"])
    text = make_label_text(corpus.all_labels, device)
    draw_kwargs = _draw_kwargs(blob)
    conditions = {
        "clean": _dataset(index, blob, rate_p=0.0, modality_p=0.0),
        "rate_downsample_mixture": _dataset(index, blob, rate_p=0.5, modality_p=0.0),
        "gyro_dropout_mixture": _dataset(index, blob, rate_p=0.0, modality_p=0.5),
    }
    metrics = {}
    for name, dataset in conditions.items():
        metrics[name] = validate(
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
            seed=seed + 91_003,
            draw_kwargs=draw_kwargs,
            executor=None,
            deployment_matched=True,
        )
    return {
        "checkpoint": str(path),
        "step": int(blob.get("step", -1)),
        "source_commit": blob.get("git", {}).get("commit"),
        "support_sets_per_condition": support_sets,
        "window_seconds": window_seconds,
        "resolutions": list(resolutions),
        "metrics": metrics,
    }


def _metric(row: dict, key: str) -> str:
    value = row.get(key)
    return "n/a" if value is None else f"{100 * float(value):.2f}"


def _markdown(report: dict) -> str:
    lines = [
        "# Internal Classifier Development Panel",
        "",
        "Subject-held-out development data only. Values are percentages. The same deterministic "
        "episode panel is reused across recording conditions.",
        "",
        "| checkpoint | condition | enrolled macro-F1 | zero-shot macro-F1 | neighbor acc. | "
        "classifier acc. | rescue | overturn | net gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    prefix = "validation/scenario/comparison/all/enrolled"
    for run in report["runs"]:
        checkpoint = Path(run["checkpoint"]).parent.name
        for condition, row in run["metrics"].items():
            lines.append(
                f"| {checkpoint} @ {run['step']} | {condition} | "
                f"{_metric(row, 'validation/enrolled_dataset_macro_f1')} | "
                f"{_metric(row, 'validation/zero_shot_dataset_macro_f1')} | "
                f"{_metric(row, prefix + '/neighbor_accuracy')} | "
                f"{_metric(row, prefix + '/classifier_accuracy')} | "
                f"{_metric(row, prefix + '/rescue_rate')} | "
                f"{_metric(row, prefix + '/overturn_rate')} | "
                f"{_metric(row, prefix + '/net_gain')} |"
            )
    lines.extend([
        "",
        "`rescue` means neighbor wrong and classifier correct. `overturn` means neighbor correct "
        "and classifier wrong. `net gain = classifier accuracy - neighbor accuracy` on enrolled "
        "episodes only.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--support-sets", type=int, default=64)
    parser.add_argument(
        "--out", type=Path,
        default=Path("training/support_classifier/evaluations/development_panel_20260917"),
    )
    args = parser.parse_args()
    if args.support_sets < 1 or args.support_sets > 256:
        parser.error("support-sets must be in [1, 256]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    report = {
        "protocol": "internal-classifier-development-panel-v1-20260917",
        "device": str(device),
        "runs": [
            evaluate_checkpoint(path, device=device, support_sets=args.support_sets)
            for path in args.checkpoints
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.out.with_suffix(".md").write_text(_markdown(report))
    print(_markdown(report))


if __name__ == "__main__":
    main()
