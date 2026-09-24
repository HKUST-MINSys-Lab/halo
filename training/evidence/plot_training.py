"""Render a compact Phase-B telemetry dashboard from one run-specific JSONL history."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("event") != "run_start":
            rows.append(value)
    return rows


def _series(rows, key, field="mean"):
    points = []
    for row in rows:
        value = row.get("metrics", {}).get(key, {}).get(field)
        if value is not None:
            points.append((row["step"], value))
    return points


def _validation_series(rows, key):
    points, previous = [], object()
    for row in rows:
        value = row.get("validation", {}).get(key)
        if value is not None and value != previous:
            points.append((row["step"], value)); previous = value
    return points


def _line(axis, points, label, **kwargs):
    if points:
        x, y = zip(*points)
        axis.plot(x, y, label=label, **kwargs)


def render(telemetry_dir: Path, output: Path | None = None) -> Path:
    telemetry_dir = Path(telemetry_dir)
    latest = json.loads((telemetry_dir / "phase_b_telemetry_latest.json").read_text())
    rows = _load(Path(latest["history_file"]))
    output = output or telemetry_dir / "telemetry.png"
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    stage = latest.get("stage")
    if stage == "predictor":
        _line(axes[0, 0], _series(rows, "loss_over_random"), "CE / random CE")
        _line(axes[0, 0], _series(rows, "chance_normalized_train_accuracy"), "chance-normalized acc")
        for name, label in (
            ("semantic_k0", "CE: k=0"),
            ("partial_unenrolled", "CE: unenrolled"),
            ("coherent_enrolled", "CE: enrolled"),
            ("alias_enrolled", "CE: alias"),
        ):
            _line(axes[0, 0], _series(rows, f"query_group_loss/{name}"), label)
        axes[0, 0].set_title("Training objective")
        for key, label in (
            ("decoder_grad_norm", "decoder"), ("retriever_grad_norm", "retriever"),
            ("retriever_task_grad_norm", "retrieval task"),
            ("tokenizer_grad_norm", "tokenizer"), ("preclip_grad_norm", "total pre-clip"),
        ):
            _line(axes[0, 1], _series(rows, key), label)
        axes[0, 1].set_yscale("symlog", linthresh=1e-4); axes[0, 1].set_title("Gradient norms")
        for key, label in (
            ("provided_support_recall_at_k", "provided-support recall"),
            ("provided_support_pool_mass", "support pool mass"),
            ("true_label_pool_mass", "true-label pool mass"),
            ("true_support_task_promoted_fraction", "task promotes support"),
            ("background_task_promoted_fraction", "task promotes background"),
        ):
            _line(axes[0, 2], _series(rows, key), label)
        axes[0, 2].set_title("Evidence use")
        for key, label in (
            ("candidate_to_evidence_attention_mass", "attention: evidence"),
            ("candidate_to_query_attention_mass", "attention: query"),
            ("candidate_to_candidate_attention_mass", "attention: candidates"),
            ("candidate_to_label_attention_mass", "attention: background labels"),
            ("candidate_attention_normalized_entropy", "attention entropy"),
        ):
            _line(axes[1, 0], _series(rows, key), label)
        _line(
            axes[1, 0],
            _validation_series(rows, "retrieval_roster_jaccard_to_previous"),
            "roster overlap", marker="o",
        )
        axes[1, 0].set_title("Attention and roster health")
        for key, label in (
            ("macro_cell_ba", "held-out"), ("identity_macro_cell_ba", "identity"),
            ("train_macro_cell_ba", "matched train"),
            ("support_k0_macro_cell_ba", "k=0"),
            ("support_k1_macro_cell_ba", "k=1"),
            ("support_k2_macro_cell_ba", "k=2"),
            ("support_k4_macro_cell_ba", "k=4"),
            ("support_k8_macro_cell_ba", "k=8"),
            ("support_k0_identity_macro_cell_ba", "k=0 identity"),
        ):
            _line(axes[1, 1], _validation_series(rows, key), label, marker="o")
        axes[1, 1].set_title("Fixed canaries")
        for name in ("signal", "text", "role", "slot", "time", "group", "relation"):
            _line(axes[0, 3], _series(rows, f"component_scale/{name}"), name)
        axes[0, 3].set_title("Token component scales")
        for key, label in (
            ("component_grad_rms/query_projection", "query"),
            ("component_grad_rms/evidence_projection", "evidence"),
            ("component_grad_rms/text_projection", "text"),
            ("component_grad_rms/time_projection", "time"),
            ("component_grad_rms/acquisition_relation_embeddings", "relations"),
            ("component_grad_rms/coreference_slot_embeddings", "slot"),
            ("component_grad_rms/component_scales", "scale gates"),
        ):
            _line(axes[1, 3], _series(rows, key), label)
        axes[1, 3].set_yscale("symlog", linthresh=1e-5)
        axes[1, 3].set_title("Input-path gradient RMS")
    else:
        _line(axes[0, 0], _series(rows, "loss"), "train BCE")
        _line(axes[0, 0], _validation_series(rows, "bce"), "validation BCE", marker="o")
        axes[0, 0].set_title("Confidence loss")
        _line(axes[0, 1], _series(rows, "grad_norm"), "head")
        axes[0, 1].set_title("Gradient norm")
        for key, label in (
            ("target_positive_rate", "target positive"),
            ("predicted_confidence_mean", "mean confidence"),
            ("predicted_confidence_std", "confidence std"),
        ):
            _line(axes[0, 2], _series(rows, key), label)
        axes[0, 2].set_title("Output distribution")
        for key, label in (("auroc", "AUROC"), ("auprc", "AUPRC"), ("ece", "ECE")):
            _line(axes[1, 0], _validation_series(rows, key), label, marker="o")
        axes[1, 0].set_title("Calibration validation")
        for key, label in (
            ("mean_confidence_positive", "positive"),
            ("mean_confidence_negative", "negative"),
            ("positive_rate", "base rate"),
        ):
            _line(axes[1, 1], _validation_series(rows, key), label, marker="o")
        axes[1, 1].set_title("Confidence separation")
    for key, label in (
        ("step_seconds", "step seconds"),
        ("gpu_allocated_gib", "allocated GiB"),
        ("gpu_reserved_gib", "reserved GiB"),
    ):
        _line(axes[1, 2], _series(rows, key), label)
    axes[1, 2].set_title("Performance")
    for axis in axes.flat:
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=7)
    figure.suptitle(f"Phase B {stage} · {latest.get('run_id')}", fontsize=11)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    figure.savefig(temporary, dpi=140)
    plt.close(figure)
    temporary.replace(output)
    return output
