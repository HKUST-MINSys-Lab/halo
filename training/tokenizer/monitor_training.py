"""Minute-level, CPU-only health monitor for a running Phase-A pretrain.

Reads ``log.jsonl`` and ``run_config.json`` from a run directory, writes atomic ``health.json`` and
``health.txt`` snapshots, and optionally refreshes ``telemetry.png``. It never imports Torch or
touches the training GPU.

Run once (appropriate for a periodically waking agent):

    python -m training.tokenizer.monitor_training --run-dir training/tokenizer/outputs/<run> --render

Or keep a lightweight local watcher alive:

    python -m training.tokenizer.monitor_training --run-dir ... --render --watch 60
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 2
SEVERITY = {"green": 0, "warning": 1, "critical": 2}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(value)
    tmp.replace(path)


def _finite(value) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def _median(records: list[dict], key: str) -> float:
    values = [float(row[key]) for row in records if key in row and _finite(row[key])]
    return statistics.median(values) if values else float("nan")


def _selection(row: dict) -> float:
    for key in ("development_transfer_knn_ba", "val_knn_label_stream_ba", "val_knn_ba"):
        if _finite(row.get(key)):
            return float(row[key])
    return float("nan")


def assess(run_dir: Path, stale_seconds: float = 120.0) -> dict:
    log_path = run_dir / "log.jsonl"
    rows = load_jsonl(log_path)
    train = [row for row in rows if "total" in row]
    val = [row for row in rows if any(key in row for key in (
        "development_transfer_knn_ba", "val_knn_label_stream_ba", "val_knn_ba",
    ))]
    events = [row for row in rows if "event" in row]
    try:
        config = json.loads((run_dir / "run_config.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}

    now = time.time()
    heartbeat_age = now - log_path.stat().st_mtime if log_path.exists() else float("inf")
    latest = train[-1] if train else {}
    latest_step = int(max((row.get("step", 0) for row in rows), default=0))
    planned_steps = int(config.get("steps", latest_step or 0))
    warmup_steps = int(config.get("warmup_steps", 1_000))
    alerts: list[dict[str, str]] = []

    def alert(level: str, code: str, message: str) -> None:
        alerts.append({"severity": level, "code": code, "message": message})

    if not train:
        alert("critical", "no_training_records", "No scalar training record is available.")
    elif heartbeat_age > stale_seconds and latest_step < planned_steps:
        alert("critical", "stale_heartbeat",
              f"Training log has not changed for {heartbeat_age:.0f} seconds.")

    future_mode = config.get("jepa_mode", "masked") == "future"
    critical_numeric = (
        "total", "future/loss", "physical/loss", "collapse/total", "jepa", "vicreg",
        "grad/total_preclip", "grad/encoder", "grad/future_predictor",
        "grad/physical_decoder", "grad/frontend", "descriptor/loss", "grad/sensor_fold",
        "grad/descriptor_projection", "grad/bias_projection", "grad/descriptor_head",
        "repr_encoder/effective_rank", "repr_retrieval/effective_rank",
        "repr_projector/effective_rank", "frontend/observable_fraction",
        "frontend/dead_kernel_fraction",
    )
    for row in train[-12:]:
        bad = [key for key in critical_numeric if key in row and not _finite(row[key])]
        if bad:
            alert("critical", "nonfinite_telemetry",
                  f"Non-finite values at step {row.get('step')}: {', '.join(bad)}.")
            break

    if latest.get("jepa_zero_target_frac_window", 0.0) > 0:
        alert("critical", "jepa_zero_targets",
              f"{latest['jepa_zero_target_frac_window']:.2%} of recent JEPA-eligible windows "
              "had no target.")
    if future_mode and int(latest.get("future/leakage_count", 0)) != 0:
        alert("critical", "future_context_leakage",
              f"Future-context leakage count is {latest['future/leakage_count']}.")
    if future_mode and latest_step > warmup_steps:
        horizon_counts = [value for key, value in latest.items()
                          if key.startswith("future/horizon_") and key.endswith("_count")]
        if horizon_counts and any(int(value) == 0 for value in horizon_counts):
            alert("warning", "empty_future_horizon",
                  "At least one configured future-horizon bin received no target in the latest batch.")
        resolution_counts = [value for key, value in latest.items()
                             if key.startswith("future/resolution_") and key.endswith("_count")]
        if resolution_counts and any(int(value) == 0 for value in resolution_counts):
            alert("warning", "empty_future_resolution",
                  "At least one frontend resolution received no target in the latest batch.")
        physical_rows = [row for row in train[-6:]
                         if _finite(row.get("physical/improvement_over_zero"))]
        physical_gain = _median(physical_rows, "physical/improvement_over_zero")
        if (latest_step > warmup_steps + 500 and len(physical_rows) >= 4
                and float(config.get("physical_weight", 0.0)) > 0.0
                and math.isfinite(physical_gain) and physical_gain <= 0.0):
            alert("warning", "physical_decoder_below_zero_baseline",
                  f"Recent physical-decoder improvement over zero is {physical_gain:.4f}.")
    if (not future_mode and config.get("token_granularity") == "sensor"
            and latest_step > warmup_steps
            and float(config.get("descriptor_weight", 0.0)) > 0.0
            and float(latest.get("descriptor/target_window_fraction", 0.0)) <= 0.0):
        alert("critical", "descriptor_no_targets",
              "No descriptor-mask target was produced in the latest telemetry window.")
    if (not future_mode and config.get("token_granularity") == "sensor"
            and float(config.get("descriptor_weight", 0.0)) > 0.0
            and latest_step > warmup_steps + 500):
        descriptor_rows = [
            row for row in train[-12:]
            if _finite(row.get("descriptor/top1")) and _finite(row.get("descriptor/chance_top1"))
            and float(row["descriptor/chance_top1"]) > 0
        ]
        if len(descriptor_rows) >= 4:
            descriptor_top1 = statistics.median(
                float(row["descriptor/top1"]) for row in descriptor_rows
            )
            descriptor_chance = statistics.median(
                float(row["descriptor/chance_top1"]) for row in descriptor_rows
            )
            if descriptor_top1 <= 1.25 * descriptor_chance:
                alert(
                    "warning", "descriptor_not_learning",
                    f"Descriptor retrieval top-1 ({descriptor_top1:.1%}) is near its "
                    f"candidate-set chance level ({descriptor_chance:.1%}).",
                )
    if latest.get("data/input_finite_fraction", 1.0) < 1.0:
        alert("critical", "nonfinite_input",
              f"Recent input finite fraction is {latest['data/input_finite_fraction']:.6f}.")
    if config.get("frontend") == "multispan" and latest_step > warmup_steps:
        frontend_grad = _median(train[-6:], "grad/frontend")
        if not math.isfinite(frontend_grad):
            alert("critical", "frontend_gradient_missing",
                  "Multi-span frontend gradient telemetry is missing after warmup.")
        elif frontend_grad <= 0.0:
            alert("critical", "frontend_no_gradient",
                  "The learnable multi-span frontend has zero recent gradient norm.")
        dead_fraction = float(latest.get("frontend/dead_kernel_fraction", float("nan")))
        observable = float(latest.get("frontend/observable_fraction", float("nan")))
        if not math.isfinite(dead_fraction) or not math.isfinite(observable):
            alert("critical", "frontend_health_missing",
                  "Multi-span kernel observability/dead-kernel telemetry is missing after warmup.")
        elif dead_fraction > 0.0:
            alert("warning", "dead_frontend_kernels",
                  f"{dead_fraction:.1%} of multi-span kernels have near-zero response variance.")
        if math.isfinite(observable) and observable <= 0.0:
            alert("critical", "frontend_unobservable",
                  "No multi-span kernel is observable at the sampled acquisition rates.")
    if int(latest.get("amp/consecutive_skips", 0)) >= 3:
        alert("critical", "amp_repeated_skips", "At least three optimizer updates were skipped in a row.")
    elif int(latest.get("amp/skipped_updates_window", 0)) > 0:
        alert("warning", "amp_update_skip", "AMP skipped an optimizer update in the latest window.")

    recent = train[-12:]
    if latest_step > warmup_steps:
        encoder_rank = float(latest.get("repr_encoder/effective_rank", float("nan")))
        d_model = int(config.get("d_model", 256))
        jepa_margin = _median(recent, "jepa/margin")
        companion_margin = (_median(recent, "physical/improvement_over_zero")
                            if future_mode else _median(recent, "vicreg/margin"))
        if (math.isfinite(encoder_rank) and encoder_rank < max(4.0, 0.05 * d_model)
                and jepa_margin < 0.05 and companion_margin <= 0.0):
            alert("warning", "possible_representation_collapse",
                  "Encoder rank, predictive margin, and the auxiliary objective are "
                  "simultaneously weak.")

        objective_names = (("future", "physical", "collapse") if future_mode
                           else ("jepa", "vicreg"))
        geometry = [row for row in train
                    if any(f"grad_objective/{name}_share" in row
                           for name in objective_names)]
        if geometry:
            shares = {
                name: _median(geometry[-3:], f"grad_objective/{name}_share")
                for name in objective_names
            }
            finite_shares = {name: value for name, value in shares.items()
                             if math.isfinite(value)}
            weak = {name: value for name, value in finite_shares.items() if value < 0.02}
            dominant = {name: value for name, value in finite_shares.items() if value > 0.90}
            if weak or dominant:
                values = ", ".join(f"{name}={value:.1%}"
                                   for name, value in finite_shares.items())
                alert("warning", "objective_gradient_imbalance",
                      f"Recent encoder-gradient shares are imbalanced: {values}.")

        clip = _median(recent, "grad/clip_coefficient")
        if math.isfinite(clip) and clip < 0.05:
            alert("warning", "severe_gradient_clipping",
                  f"Median recent gradient clip coefficient is {clip:.3f}.")

    throughput = [float(row["perf/steps_per_s"]) for row in train
                  if row.get("step", 0) > warmup_steps and _finite(row.get("perf/steps_per_s"))]
    if len(throughput) >= 8:
        baseline = statistics.median(throughput[:-3]) if len(throughput) > 3 else float("nan")
        current = statistics.median(throughput[-3:])
        if math.isfinite(baseline) and current < 0.7 * baseline:
            alert("warning", "throughput_regression",
                  f"Recent throughput {current:.2f} step/s is over 30% below baseline {baseline:.2f}.")

    observed = latest.get("data/source_share_window", {})
    target = latest.get("data/source_share_target", {})
    source_errors = {key: float(observed.get(key, 0.0)) - float(value)
                     for key, value in target.items()}
    balance_examples = int(latest.get("data/examples_window", 0))
    if (balance_examples >= 2_048 and source_errors
            and max(abs(value) for value in source_errors.values()) > 0.05):
        worst = max(source_errors, key=lambda key: abs(source_errors[key]))
        alert("warning", "source_balance_drift",
              f"Rolling share for {worst} differs from target by {source_errors[worst]:+.1%}.")

    val_values = [_selection(row) for row in val if math.isfinite(_selection(row))]
    if len(val_values) >= 3 and all(value < max(val_values) - 0.10 for value in val_values[-2:]):
        alert("warning", "validation_regression",
              "The last two checkpoint-selection scores are over 0.10 below the run best.")
    if val and not (run_dir / "last.pt").exists():
        alert("warning", "missing_last_checkpoint", "Validation exists but last.pt is absent.")

    status = "green"
    for item in alerts:
        if SEVERITY[item["severity"]] > SEVERITY[status]:
            status = item["severity"]

    selection_values = [_selection(row) for row in val if math.isfinite(_selection(row))]
    progress = latest_step / planned_steps if planned_steps > 0 else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "run_dir": str(run_dir),
        "step": latest_step,
        "planned_steps": planned_steps,
        "progress": progress,
        "heartbeat_age_s": heartbeat_age,
        "alerts": alerts,
        "performance": {
            "steps_per_s": latest.get("perf/steps_per_s"),
            "examples_per_s": latest.get("perf/examples_per_s"),
            "eta_minutes": latest.get("perf/eta_minutes"),
            "allocated_gib": latest.get("memory/allocated_gib"),
            "reserved_gib": latest.get("memory/reserved_gib"),
        },
        "loss": {
            "total": latest.get("total"),
            "future_weighted": latest.get("loss_weighted/future"),
            "physical_weighted": latest.get("loss_weighted/physical"),
            "collapse_weighted": latest.get("loss_weighted/collapse"),
            "physical_improvement_over_zero": latest.get(
                "physical/improvement_over_zero"),
            "collapse_min_std": latest.get("collapse/min_std"),
            "jepa_weighted": latest.get("loss_weighted/jepa"),
            "vicreg_weighted": latest.get("loss_weighted/vicreg"),
            "descriptor_weighted": latest.get("loss_weighted/descriptor"),
            "descriptor_top1": latest.get("descriptor/top1"),
            "descriptor_candidates": latest.get("descriptor/candidates"),
            "descriptor_chance_top1": latest.get("descriptor/chance_top1"),
            "jepa_margin": latest.get("jepa/margin"),
            "vicreg_margin": latest.get("vicreg/margin"),
            "vicreg_min_std": latest.get("vicreg/min_std"),
        },
        "gradients": {
            "total_preclip": latest.get("grad/total_preclip"),
            "clip_coefficient": latest.get("grad/clip_coefficient"),
            "jepa_share": latest.get("grad_objective/jepa_share"),
            "future_share": latest.get("grad_objective/future_share"),
            "physical_share": latest.get("grad_objective/physical_share"),
            "collapse_share": latest.get("grad_objective/collapse_share"),
            "objective_cosine": latest.get("grad_cosine/jepa_vs_vicreg"),
            "future_physical_cosine": latest.get("grad_cosine/future_vs_physical"),
            "future_collapse_cosine": latest.get("grad_cosine/future_vs_collapse"),
            "amp_scale": latest.get("amp/scale"),
            "amp_skipped_total": latest.get("amp/skipped_updates_total"),
            "sensor_fold": latest.get("grad/sensor_fold"),
            "descriptor_projection": latest.get("grad/descriptor_projection"),
            "bias_projection": latest.get("grad/bias_projection"),
            "descriptor_head": latest.get("grad/descriptor_head"),
            "future_predictor": latest.get("grad/future_predictor"),
            "physical_decoder": latest.get("grad/physical_decoder"),
            "frontend": latest.get("grad/frontend"),
        },
        "frontend": {
            "duration_gate": latest.get("duration/gate"),
            "observable_fraction": latest.get("frontend/observable_fraction"),
            "dead_kernel_fraction": latest.get("frontend/dead_kernel_fraction"),
            "response_std_mean": latest.get("frontend/response_std_mean"),
            "edge_support_mean": latest.get("frontend/edge_support_mean"),
        },
        "representation": {
            "encoder_effective_rank": latest.get("repr_encoder/effective_rank"),
            "encoder_min_std": latest.get("repr_encoder/min_std"),
            "visible_effective_rank": latest.get("repr_retrieval/effective_rank"),
            "visible_min_std": latest.get("repr_retrieval/min_std"),
            "projector_effective_rank": latest.get("repr_projector/effective_rank"),
            "teacher_effective_rank": latest.get("repr_teacher/effective_rank"),
        },
        "data": {
            "source_share": observed,
            "source_target": target,
            "source_error": source_errors,
            "batches_window": latest.get("data/batches_window"),
            "examples_window": latest.get("data/examples_window"),
            "stream_share": latest.get("data/stream_share_window", {}),
            "channel_count_share": latest.get("data/channel_count_share_window", {}),
            "patch_pair_share": latest.get("data/patch_pair_share_window", {}),
            "augmentation_rate": latest.get("data/augmentation_rate_window", {}),
            "zero_target_fraction": latest.get("jepa_zero_target_frac_window"),
            "ineligible_fraction": latest.get("jepa_ineligible_frac_window"),
            "context_fraction": latest.get("future/context_fraction_tokens"),
            "target_fraction": latest.get("future/target_fraction_tokens"),
            "leakage_count": latest.get("future/leakage_count"),
            "descriptor_target_window_fraction": latest.get(
                "descriptor/target_window_fraction"),
            "input_finite_fraction": latest.get("data/input_finite_fraction"),
            "input_abs_max": latest.get("data/input_abs_max"),
            "input_rms": latest.get("data/input_rms"),
        },
        "validation": {
            "points": len(val),
            "latest_selection": selection_values[-1] if selection_values else None,
            "best_selection": max(selection_values) if selection_values else None,
            "latest_development_transfer": (
                val[-1].get("development_transfer_knn_ba") if val else None
            ),
            "latest_global_knn": val[-1].get("val_knn_ba") if val else None,
            "latest_conse_label_stream": val[-1].get("val_conse_label_stream_ba") if val else None,
        },
        "events": events[-5:],
    }


def render_text(report: dict) -> str:
    perf = report["performance"]
    val = report["validation"]
    lines = [
        f"STATUS: {report['status'].upper()}",
        f"step: {report['step']:,}/{report['planned_steps']:,} ({report['progress']:.1%})",
        f"heartbeat: {report['heartbeat_age_s']:.0f}s ago",
        f"throughput: {perf.get('steps_per_s') or float('nan'):.2f} step/s",
        f"ETA: {perf.get('eta_minutes') or float('nan'):.1f} min",
        f"selection: latest={val.get('latest_selection')} best={val.get('best_selection')}",
    ]
    if report["alerts"]:
        lines.append("alerts:")
        lines.extend(
            f"- [{item['severity']}] {item['code']}: {item['message']}"
            for item in report["alerts"]
        )
    else:
        lines.append("alerts: none")
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path, *, stale_seconds: float, render: bool) -> dict:
    report = assess(run_dir, stale_seconds=stale_seconds)
    _atomic_text(run_dir / "health.json", json.dumps(report, indent=2) + "\n")
    summary = render_text(report)
    _atomic_text(run_dir / "health.txt", summary)
    if render and (run_dir / "log.jsonl").exists():
        from training.tokenizer.plot_training import render as render_plot
        render_plot(run_dir / "log.jsonl", run_dir / "telemetry.png")
    print(summary, end="", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--render", action="store_true", help="also refresh telemetry.png")
    parser.add_argument("--watch", type=float, default=0.0,
                        help="refresh every N seconds; 0 runs once")
    parser.add_argument("--stale-seconds", type=float, default=120.0)
    args = parser.parse_args()
    while True:
        write_report(args.run_dir, stale_seconds=args.stale_seconds, render=args.render)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
