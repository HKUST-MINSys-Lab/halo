"""Regression tests for run-safe Phase-B telemetry and health assessment."""

from __future__ import annotations

import json

from training.evidence.monitor_training import assess, write_report
from training.evidence.telemetry import PhaseBTelemetry
from training.evidence.policy import PHASE_B_DEV_DATASETS, PHASE_B_TEST_DATASETS
from data.scripts.curate.deployment_policy import PRIMARY_EVAL_DATASETS


def test_new_run_immediately_replaces_stale_latest_and_keeps_separate_history(tmp_path):
    first = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="first")
    first.start(step=10, elapsed_seconds=5, metadata={"planned_steps": 100})
    first.update({"loss": 1.0})
    first.emit(step=10, elapsed_seconds=5, force=True)

    second = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="second")
    second.start(step=0, elapsed_seconds=0, metadata={"planned_steps": 100})
    latest = json.loads((tmp_path / "phase_b_telemetry_latest.json").read_text())
    assert latest["run_id"] == "second"
    assert latest["event"] == "run_start"
    assert first.jsonl != second.jsonl
    assert first.jsonl.exists() and second.jsonl.exists()


def test_nonfinite_and_stratified_metrics_are_visible_to_monitor(tmp_path):
    telemetry = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="audit")
    telemetry.start(
        step=0,
        elapsed_seconds=0,
        metadata={"planned_steps": 100, "warmup_steps": 0, "grad_clip": 1.0},
    )
    telemetry.update(
        {
            "loss": 1.0,
            "bad": float("nan"),
            "decoder_grad_norm": 1.0,
            "retriever_grad_norm": 1.0,
            "hard_forward_max_abs_error": 0.0,
        },
        categories={"episode_type": "ordinary_few_support"},
        strata={"episode_type": "ordinary_few_support"},
    )
    payload = telemetry.emit(step=5, elapsed_seconds=2, force=True)
    assert payload["nonfinite"] == {"bad": 1}
    assert payload["strata"]["episode_type"]["ordinary_few_support"]["loss"]["mean"] == 1.0
    report = assess(tmp_path)
    assert report["status"] == "critical"
    assert any(item["code"] == "nonfinite_values" for item in report["alerts"])
    telemetry.update({"loss": 0.5})
    later = telemetry.emit(step=6, elapsed_seconds=3, force=True)
    assert later["nonfinite"] == {"bad": 1}
    final = telemetry.emit(step=6, elapsed_seconds=3, force=True, final=True)
    assert final["event"] == "run_end"
    assert final["metrics"] == later["metrics"]


def test_monitor_writes_human_machine_and_plot_outputs(tmp_path):
    telemetry = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="plot")
    telemetry.start(step=0, elapsed_seconds=0, metadata={"planned_steps": 10})
    telemetry.update({"loss": 1.0, "grad_norm": 0.2, "target_positive_rate": 0.5,
                      "predicted_confidence_mean": 0.5, "predicted_confidence_std": 0.1})
    telemetry.emit(step=1, elapsed_seconds=1, force=True)
    write_report(tmp_path, render=True)
    assert (tmp_path / "health.json").exists()
    assert "STATUS:" in (tmp_path / "health.txt").read_text()
    assert (tmp_path / "telemetry.png").stat().st_size > 1_000


def test_predictor_monitor_checks_learned_retrieval_and_support_use(tmp_path):
    telemetry = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="retrieval")
    telemetry.start(
        step=0,
        elapsed_seconds=0,
        metadata={
            "planned_steps": 1000,
            "warmup_steps": 100,
            "n_retrieval_heads": 4,
        },
    )
    telemetry.update({
        "decoder_grad_norm": 1.0,
        "retriever_grad_norm": 1.0,
        "retriever_to_decoder_grad_ratio": 1.0,
        "gradient_clipped_fraction": 0.0,
        "provided_support_recall_at_k": 0.01,
    })
    telemetry.set_validation({
        "macro_cell_ba": 0.2,
        "adaptation_macro_cell_ba_gain": 0.0,
        "support_removal_true_probability_drop": 0.0,
        "support_label_shuffle_true_probability_drop": 0.0,
    })
    telemetry.emit(step=600, elapsed_seconds=60, force=True)
    codes = {item["code"] for item in assess(tmp_path)["alerts"]}
    assert {
        "low_provided_support_recall",
        "support_removal_inert",
        "support_labels_inert",
    } <= codes


def test_predictor_monitor_detects_role_attention_starvation_and_weak_credit(tmp_path):
    telemetry = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="attention")
    telemetry.start(
        step=0,
        elapsed_seconds=0,
        metadata={"planned_steps": 1000, "warmup_steps": 10},
    )
    telemetry.update({
        "decoder_grad_norm": 1.0,
        "retriever_grad_norm": 1e-6,
        "retriever_to_decoder_grad_ratio": 1.0,
        "retriever_to_decoder_grad_rms_ratio": 1e-6,
        "candidate_to_evidence_attention_mass": 0.99,
        "candidate_to_query_attention_mass": 0.003,
        "candidate_to_candidate_attention_mass": 0.004,
        "candidate_to_label_attention_mass": 0.003,
        "candidate_attention_normalized_entropy": 0.05,
    })
    telemetry.emit(step=100, elapsed_seconds=10, force=True)
    codes = {item["code"] for item in assess(tmp_path)["alerts"]}
    assert {
        "weak_retriever_gradient",
        "evidence_role_starvation",
        "query_attention_inert",
        "name_attention_inert",
        "attention_collapse",
    } <= codes


def test_predictor_monitor_detects_incomplete_component_gradient_coverage(tmp_path):
    telemetry = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="gradient-coverage")
    telemetry.start(
        step=0,
        elapsed_seconds=0,
        metadata={"planned_steps": 1000, "warmup_steps": 10},
    )
    telemetry.update({
        "decoder_grad_norm": 1.0,
        "component_grad_norm/time_projection": 0.1,
        "component_grad_coverage/time_projection": 0.5,
    })
    telemetry.emit(step=100, elapsed_seconds=10, force=True)
    codes = {item["code"] for item in assess(tmp_path)["alerts"]}
    assert "incomplete_component_gradient_coverage" in codes


def test_predictor_monitor_checks_minimal_episode_contract(tmp_path):
    telemetry = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="episodes")
    telemetry.start(step=0, elapsed_seconds=0, metadata={"planned_steps": 100})
    for _ in range(32):
        telemetry.update(
            {"loss": 1.0},
            categories={
                "episode_type": "ordinary_few_support",
                "support_count": "2",
                "label_mode": "random_alias",
                "physical_view_mode": "clean",
            },
        )
    telemetry.emit(step=1, elapsed_seconds=1, force=True)
    codes = {item["code"] for item in assess(tmp_path)["alerts"]}
    assert "missing_zero_support_draws" in codes
    assert "unexpected_label_mode" not in codes
    assert "unexpected_physical_view" not in codes
    assert "label_support_contract_broken" not in codes


def test_predictor_monitor_detects_label_support_contract_break(tmp_path):
    telemetry = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="label-contract")
    telemetry.start(step=0, elapsed_seconds=0, metadata={"planned_steps": 100})
    for _ in range(32):
        telemetry.update(
            {"loss": 1.0},
            categories={
                "episode_type": "ordinary_few_support",
                "support_count": "2",
                "label_mode": "coherent",
                "physical_view_mode": "clean",
            },
        )
    telemetry.emit(step=1, elapsed_seconds=1, force=True)
    codes = {item["code"] for item in assess(tmp_path)["alerts"]}
    assert "label_support_contract_broken" in codes


def test_predictor_monitor_detects_token_component_scale_imbalance(tmp_path):
    telemetry = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="component-scales")
    telemetry.start(
        step=0, elapsed_seconds=0,
        metadata={"planned_steps": 1000, "warmup_steps": 10},
    )
    scales = {
        "signal": 0.2, "text": 1.0, "role": 1.0, "slot": 1.0,
        "time": 1.0, "group": 1.0, "relation": 1.0,
    }
    telemetry.update({f"component_scale/{name}": value for name, value in scales.items()})
    telemetry.emit(step=100, elapsed_seconds=10, force=True)
    codes = {item["code"] for item in assess(tmp_path)["alerts"]}
    assert "component_scale_imbalance" in codes


def test_predictor_monitor_detects_zero_support_checkpoint_regression(tmp_path):
    telemetry = PhaseBTelemetry(
        tmp_path, interval_seconds=60, run_id="k0-guard", stage="predictor"
    )
    telemetry.start(
        step=0, elapsed_seconds=0,
        metadata={"planned_steps": 1000, "warmup_steps": 10},
    )
    telemetry.set_validation({
        "support_k0_macro_cell_ba": 0.20,
        "support_k0_identity_macro_cell_ba": 0.30,
    })
    telemetry.emit(step=100, elapsed_seconds=10, force=True)
    codes = {item["code"] for item in assess(tmp_path)["alerts"]}
    # Reported separately from the current support-use eligibility checks.
    assert "zero_support_below_identity" in codes
    assert "zero_support_guard_failed" not in codes


def test_external_development_and_test_rosters_are_complete_and_disjoint():
    assert set(PHASE_B_DEV_DATASETS).isdisjoint(PHASE_B_TEST_DATASETS)
    assert set(PHASE_B_DEV_DATASETS) | set(PHASE_B_TEST_DATASETS) == set(PRIMARY_EVAL_DATASETS)
