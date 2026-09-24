"""CPU-only health monitor for Phase-B predictor and confidence training."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
SEVERITY = {"green": 0, "warning": 1, "critical": 2}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_history(path: Path, run_id: str | None) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (run_id is None or value.get("run_id") == run_id):
            rows.append(value)
    return rows


def _metric(snapshot: dict, name: str, field: str = "mean") -> float:
    try:
        return float(snapshot["metrics"][name][field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def assess(telemetry_dir: Path, stale_seconds: float = 150.0) -> dict:
    latest_path = Path(telemetry_dir) / "phase_b_telemetry_latest.json"
    snapshot = _load(latest_path)
    alerts: list[dict[str, str]] = []

    def alert(level: str, code: str, message: str) -> None:
        alerts.append({"severity": level, "code": code, "message": message})

    now = time.time()
    generated = float(snapshot.get("generated_unix_seconds", 0.0))
    age = now - generated if generated else float("inf")
    step = int(snapshot.get("step", 0))
    metadata = snapshot.get("metadata", {})
    planned = int(metadata.get("planned_steps", step or 0))
    final = snapshot.get("event") == "run_end" or (planned > 0 and step >= planned)
    stage = snapshot.get("stage", "unknown")
    history = _load_history(
        Path(snapshot.get("history_file", "")), snapshot.get("run_id")
    ) if snapshot else []

    if not snapshot:
        alert("critical", "no_telemetry", "No Phase-B telemetry snapshot is available.")
    elif age > stale_seconds and not final:
        alert("critical", "stale_heartbeat", f"Telemetry has not updated for {age:.0f} seconds.")
    if snapshot.get("nonfinite"):
        bad = ", ".join(f"{key}={count}" for key, count in snapshot["nonfinite"].items())
        alert("critical", "nonfinite_values", f"Non-finite values occurred: {bad}.")

    if stage == "predictor" and (snapshot.get("metrics") or snapshot.get("validation")):
        warmup = int(metadata.get("warmup_steps", 0))
        mix = snapshot.get("mix", {})
        label_modes = mix.get("label_mode", {})
        expected_label_modes = {"coherent", "random_alias"}
        if label_modes and not set(label_modes).issubset(expected_label_modes):
            alert(
                "critical", "unexpected_label_mode",
                f"Phase-B training emitted unexpected label modes: {label_modes}.",
            )
        physical_modes = mix.get("physical_view_mode", {})
        if physical_modes and set(physical_modes) != {"clean"}:
            alert(
                "critical", "unexpected_physical_view",
                f"Minimal Phase-B training emitted non-clean views: {physical_modes}.",
            )
        support_mix = mix.get("support_count", {})
        support_total = sum(int(value) for value in support_mix.values())
        if support_total >= 32:
            zero = int(support_mix.get("0", 0))
            coherent = int(label_modes.get("coherent", 0))
            aliases = int(label_modes.get("random_alias", 0))
            if coherent != zero or aliases != support_total - zero:
                alert(
                    "critical", "label_support_contract_broken",
                    "Phase-B requires coherent names exactly for k=0 and arbitrary names exactly "
                    f"for k>0; observed support={support_mix}, labels={label_modes}.",
                )
            if zero == 0:
                alert(
                    "warning", "missing_zero_support_draws",
                    f"No k=0 episode occurred in {support_total} recent independent draws.",
                )
            if zero == support_total:
                alert(
                    "critical", "missing_supported_draws",
                    f"All {support_total} recent episodes used k=0.",
                )
        if step > warmup:
            for name in ("decoder_grad_norm",):
                maximum = _metric(snapshot, name, "max")
                if math.isfinite(maximum) and maximum == 0.0:
                    alert("critical", "dead_gradient", f"{name} remained zero in the latest window.")
        if step > warmup:
            maximum = _metric(snapshot, "retriever_grad_norm", "max")
            if math.isfinite(maximum) and maximum == 0.0:
                alert(
                    "critical", "dead_gradient",
                    "retriever_grad_norm remained zero after optimizer warm-up.",
                )
            grad_ratio = _metric(snapshot, "retriever_to_decoder_grad_rms_ratio")
            if not math.isfinite(grad_ratio):
                # Compatibility with telemetry emitted before size-normalized gradients existed.
                grad_ratio = _metric(snapshot, "retriever_to_decoder_grad_ratio")
            if math.isfinite(grad_ratio) and grad_ratio < 1e-4:
                alert(
                    "warning", "weak_retriever_gradient",
                    f"Retriever/decoder gradient RMS ratio is only {grad_ratio:.2e} in the latest "
                    "window; retrieval may be learning too slowly to change the roster.",
                )
        clipped = _metric(snapshot, "gradient_clipped_fraction")
        if step > warmup and math.isfinite(clipped) and clipped > 0.8:
            alert("warning", "persistent_clipping",
                  f"{clipped:.0%} of recent steps exceeded the gradient clip threshold.")
        pool_share = _metric(snapshot, "pool_weight_max_share")
        if math.isfinite(pool_share) and pool_share > 0.8:
            alert("warning", "pool_collapse",
                  f"One evidence item receives {pool_share:.0%} mean maximum pooling mass.")
        row_share = _metric(snapshot, "selected_row_max_share")
        if math.isfinite(row_share) and row_share > 0.25:
            alert("warning", "memory_row_collapse",
                  f"One memory row supplies {row_share:.0%} of recent selections.")
        support_recall = _metric(snapshot, "provided_support_recall_at_k")
        if step > warmup and math.isfinite(support_recall) and support_recall < 0.05:
            alert(
                "warning", "low_provided_support_recall",
                f"Only {support_recall:.1%} of supported queries retrieve enrolled evidence. "
                "Background evidence is valid, but a near-zero rate leaves enrollment unusable.",
            )
        evidence_attention = _metric(snapshot, "candidate_to_evidence_attention_mass")
        query_attention = _metric(snapshot, "candidate_to_query_attention_mass")
        name_attention = sum(
            value for value in (
                _metric(snapshot, "candidate_to_candidate_attention_mass"),
                _metric(snapshot, "candidate_to_label_attention_mass"),
            ) if math.isfinite(value)
        )
        attention_entropy = _metric(snapshot, "candidate_attention_normalized_entropy")
        if math.isfinite(evidence_attention) and evidence_attention > 0.95:
            alert(
                "critical", "evidence_role_starvation",
                f"Evidence keys receive {evidence_attention:.1%} of candidate attention; candidate, "
                "label, and query roles are being numerically starved.",
            )
        if math.isfinite(query_attention) and query_attention < 0.01:
            alert(
                "warning", "query_attention_inert",
                f"Candidate tokens assign only {query_attention:.1%} attention to query patches.",
            )
        if math.isfinite(evidence_attention) and name_attention < 0.01:
            alert(
                "warning", "name_attention_inert",
                f"Candidate/background label tokens receive only {name_attention:.1%} attention.",
            )
        if math.isfinite(attention_entropy) and attention_entropy < 0.10:
            alert(
                "warning", "attention_collapse",
                f"Candidate attention normalized entropy is {attention_entropy:.3f}.",
            )
        # The readout is the whole prediction, so a collapsed spread between the best and worst
        # candidate logit means the model has stopped discriminating at all — a constant predictor
        # that cross-entropy alone can look tolerable on an unbalanced candidate mix.
        spread = _metric(snapshot, "candidate_logit_spread", "max")
        if step > warmup and math.isfinite(spread) and spread < 1e-3:
            alert(
                "warning", "collapsed_candidate_logits",
                f"Best-to-worst candidate logit spread is {spread:.2e}; the readout is emitting a "
                "constant prediction.",
            )
        component_names = {
            key for row in history[-6:] for key in row.get("metrics", {})
            if key.startswith("component_grad_norm/")
        }
        dead_components = []
        for key in sorted(component_names):
            samples = [
                float(row["metrics"][key].get("max", 0.0))
                for row in history[-6:] if key in row.get("metrics", {})
            ]
            if len(samples) >= 3 and all(value == 0.0 for value in samples[-3:]):
                dead_components.append(key.removeprefix("component_grad_norm/"))
        if step > warmup and dead_components:
            alert("warning", "dead_components",
                  "Zero gradient in three consecutive probes: "
                  + ", ".join(dead_components) + ".")
        incomplete_components = []
        for key in sorted(
            key for key in snapshot.get("metrics", {})
            if key.startswith("component_grad_coverage/")
        ):
            coverage = _metric(snapshot, key, "min")
            if math.isfinite(coverage) and coverage < 0.999:
                incomplete_components.append(
                    f"{key.removeprefix('component_grad_coverage/')}={coverage:.1%}"
                )
        if step > warmup and incomplete_components:
            alert(
                "warning",
                "incomplete_component_gradient_coverage",
                "Some trainable parameters were absent from the latest backward graph: "
                + ", ".join(incomplete_components) + ".",
            )
        tokenizer_active = _metric(snapshot, "tokenizer_active", "max") > 0.5
        if tokenizer_active:
            dead_tokenizer_components = []
            for key in sorted(
                key for row in history[-6:] for key in row.get("metrics", {})
                if key.startswith("tokenizer_component_grad_norm/")
            ):
                samples = [
                    float(row["metrics"][key].get("max", 0.0))
                    for row in history[-6:] if key in row.get("metrics", {})
                ]
                if len(samples) >= 3 and all(value == 0.0 for value in samples[-3:]):
                    dead_tokenizer_components.append(
                        key.removeprefix("tokenizer_component_grad_norm/")
                    )
            if dead_tokenizer_components:
                alert(
                    "warning",
                    "dead_tokenizer_components",
                    "Zero gradient in three active tokenizer probes: "
                    + ", ".join(dead_tokenizer_components) + ".",
                )
        component_scales = {
            key.removeprefix("component_scale/"): _metric(snapshot, key)
            for key in snapshot.get("metrics", {}) if key.startswith("component_scale/")
        }
        finite_scales = [value for value in component_scales.values() if math.isfinite(value)]
        if len(finite_scales) == 7:
            scale_ratio = max(finite_scales) / max(min(finite_scales), 1e-12)
            if scale_ratio > 10.0:
                alert(
                    "critical", "component_scale_collapse",
                    f"Token component scales span {scale_ratio:.1f}x: {component_scales}.",
                )
            elif scale_ratio > 4.0:
                alert(
                    "warning", "component_scale_imbalance",
                    f"Token component scales span {scale_ratio:.1f}x: {component_scales}.",
                )
        subspace_count = int(metadata.get("n_retrieval_heads", 4))
        subspace_mass = [
            _metric(snapshot, f"subspace_{index}_mass") for index in range(subspace_count)
        ]
        finite_mass = [value for value in subspace_mass if math.isfinite(value)]
        if finite_mass and (max(finite_mass) > 0.8 or min(finite_mass) < 0.01):
            alert("warning", "retrieval_subspace_collapse",
                  f"Recent retrieval-subspace mass is {[round(value, 3) for value in finite_mass]}.")
        validation = snapshot.get("validation", {})
        retriever_drift = validation.get("retriever_ema_relative_drift_from_initial")
        val_every = int(metadata.get("val_every", 0))
        if retriever_drift is not None and val_every \
                and step >= warmup + 3 * val_every \
                and float(retriever_drift) < 1e-5:
            alert(
                "warning", "retriever_not_moving",
                "Retriever EMA projection remains effectively identical to initialization.",
            )
        gain = validation.get("adaptation_macro_cell_ba_gain")
        if gain is not None and step > warmup and float(gain) < -0.05:
            alert("warning", "worse_than_identity",
                  f"Held-out adaptation score is {float(gain):.3f} below identity control.")
        k0_score = validation.get("support_k0_macro_cell_ba")
        k0_identity = validation.get("support_k0_identity_macro_cell_ba")
        if k0_score is not None and k0_identity is not None \
                and float(k0_score) < float(k0_identity) - 0.05:
            alert(
                "warning", "zero_support_below_identity",
                f"Held-out k=0 BA {float(k0_score):.3f} is below the identity retrieval control "
                f"{float(k0_identity):.3f}.",
            )
        support_drop = validation.get("support_removal_true_probability_drop")
        if support_drop is not None and step > warmup and float(support_drop) <= 0.01:
            alert(
                "warning", "support_removal_inert",
                f"Removing enrolled support changes true-label probability by only "
                f"{float(support_drop):.3f}.",
            )
        shuffle_drop = validation.get("support_label_shuffle_true_probability_drop")
        if shuffle_drop is not None and step > warmup and float(shuffle_drop) <= 0.01:
            alert(
                "warning", "support_labels_inert",
                f"Shuffling enrollment labels changes true-label probability by only "
                f"{float(shuffle_drop):.3f}.",
            )
        eligible = validation.get("checkpoint_eligible")
        if eligible is False and step > warmup:
            alert(
                "warning", "checkpoint_mechanism_ineligible",
                "The latest checkpoint does not yet match the retrieval-vote control and respond "
                "positively to both support interventions.",
            )
        gap = validation.get("train_validation_macro_cell_ba_gap")
        gap_points = []
        previous_gap_marker = object()
        for row in history:
            row_validation = row.get("validation", {})
            marker = row_validation.get("macro_cell_ba")
            row_gap = row_validation.get("train_validation_macro_cell_ba_gap")
            if marker is not None and row_gap is not None and marker != previous_gap_marker:
                gap_points.append(float(row_gap))
                previous_gap_marker = marker
        baseline_gap = gap_points[0] if gap_points else None
        if gap is not None and baseline_gap is not None and step > warmup \
                and float(gap) > 0.25 and float(gap) > baseline_gap + 0.15:
            alert("warning", "large_train_validation_gap",
                  f"Train/held-out BA gap grew from {baseline_gap:.3f} to {float(gap):.3f}.")
        validation_points = []
        previous_value = object()
        for row in history:
            value = row.get("validation", {}).get("macro_cell_ba")
            if value is not None and value != previous_value:
                validation_points.append(float(value)); previous_value = value
        if len(validation_points) >= 3:
            best = max(validation_points)
            if all(value < best - 0.10 for value in validation_points[-2:]):
                alert("warning", "validation_regression",
                      "The latest two held-out scores are over 0.10 below the run best.")

    if stage == "confidence" and snapshot.get("metrics"):
        positive_rate = _metric(snapshot, "target_positive_rate")
        if math.isfinite(positive_rate) and not 0.02 <= positive_rate <= 0.98:
            alert("warning", "confidence_target_imbalance",
                  f"Recent confidence-target positive rate is {positive_rate:.1%}.")
        score_mean = _metric(snapshot, "predicted_confidence_mean")
        score_std = _metric(snapshot, "predicted_confidence_std")
        if math.isfinite(score_mean) and math.isfinite(score_std) \
                and score_std < 0.005 and (score_mean < 0.05 or score_mean > 0.95):
            alert("warning", "confidence_saturation",
                  f"Confidence outputs are nearly constant at {score_mean:.3f}.")
        validation = snapshot.get("validation", {})
        auroc = validation.get("auroc")
        if auroc is not None and float(auroc) < 0.48:
            alert("warning", "confidence_inverted",
                  f"Validation AUROC is {float(auroc):.3f}, below random ranking.")
        auprc = validation.get("auprc")
        prevalence = validation.get("positive_rate")
        if auprc is not None and prevalence is not None \
                and float(auprc) < float(prevalence) - 0.02:
            alert("warning", "confidence_below_base_rate",
                  f"Validation AUPRC {float(auprc):.3f} is below the "
                  f"{float(prevalence):.3f} positive-rate baseline.")
        ece = validation.get("ece")
        if ece is not None and float(ece) > 0.15:
            alert("warning", "confidence_miscalibrated",
                  f"Validation expected calibration error is {float(ece):.3f}.")
        positive = validation.get("mean_confidence_positive")
        negative = validation.get("mean_confidence_negative")
        if positive is not None and negative is not None and float(positive) <= float(negative):
            alert("warning", "confidence_no_separation",
                  "Mean confidence for correct predictions does not exceed incorrect predictions.")

    status = "green"
    for item in alerts:
        if SEVERITY[item["severity"]] > SEVERITY[status]:
            status = item["severity"]
    elapsed = float(snapshot.get("elapsed_seconds", 0.0))
    rate = step / elapsed if step > 0 and elapsed > 0 else float("nan")
    eta = (planned - step) / rate / 60.0 if rate > 0 and planned >= step else float("nan")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "stage": stage,
        "run_id": snapshot.get("run_id"),
        "step": step,
        "planned_steps": planned,
        "progress": step / planned if planned else 0.0,
        "heartbeat_age_s": age,
        "steps_per_s": rate,
        "eta_minutes": eta,
        "alerts": alerts,
        "history_points": len(history),
        "metrics": snapshot.get("metrics", {}),
        "mix": snapshot.get("mix", {}),
        "strata": snapshot.get("strata", {}),
        "validation": snapshot.get("validation", {}),
        "telemetry": str(latest_path),
    }


def render_text(report: dict) -> str:
    rate = report["steps_per_s"]
    eta = report["eta_minutes"]
    lines = [
        f"STATUS: {report['status'].upper()} ({report['stage']})",
        f"run: {report.get('run_id')}",
        f"step: {report['step']:,}/{report['planned_steps']:,} ({report['progress']:.1%})",
        f"heartbeat: {report['heartbeat_age_s']:.0f}s ago",
        f"throughput: {rate:.2f} step/s; ETA: {eta:.1f} min",
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


def write_report(telemetry_dir: Path, *, stale_seconds: float = 150.0, render: bool = False) -> dict:
    report = assess(telemetry_dir, stale_seconds=stale_seconds)
    _atomic_text(Path(telemetry_dir) / "health.json", json.dumps(report, indent=2) + "\n")
    text = render_text(report)
    _atomic_text(Path(telemetry_dir) / "health.txt", text)
    if render:
        from training.evidence.plot_training import render as render_plot
        render_plot(telemetry_dir)
    print(text, end="", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--watch", type=float, default=0.0)
    parser.add_argument("--stale-seconds", type=float, default=150.0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    while True:
        write_report(
            args.telemetry_dir, stale_seconds=args.stale_seconds, render=args.render
        )
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
