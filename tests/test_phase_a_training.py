"""Focused tests for Phase-A evaluation and resume control paths."""

import subprocess

import pytest
import torch

import training.tokenizer.pretrain as pretrain_module
from training.tokenizer.pretrain import (
    PipelineAModel,
    PretrainConfig,
    capture_source_provenance,
    hydrate_calibrated_objective_weights,
    label_group_balanced_acc,
    prepare_output_dir,
    stratified_eval_subset,
)
from training.tokenizer.pretrain_data import WindowKey


def test_batch_1024_recipe_preserves_reference_sample_budget_and_ema_timebase():
    cfg = PretrainConfig()
    assert cfg.batch_size == 1024
    assert cfg.steps == 7_500
    assert cfg.batch_size * cfg.steps == 256 * 30_000
    assert cfg.warmup_steps * cfg.batch_size == 1_000 * 256
    assert cfg.lr == pytest.approx(6e-4)
    assert cfg.weight_decay == pytest.approx(0.1)
    assert cfg.jepa_ema_decay == pytest.approx(0.996 ** 4)
    assert cfg.val_every == 500


def test_live_encoder_carries_checkpoint_evaluation_grid():
    cfg = PretrainConfig(
        d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
        token_granularity="sensor", multiresolution=True,
    )
    enc = PipelineAModel(cfg).encoder
    assert enc.multiresolution is True
    assert enc.eval_resolution_pair == tuple(cfg.future_patch_durations)
    assert enc.min_resolution_ratio == cfg.min_resolution_ratio


def test_checkpoint_without_dft_metadata_uses_historical_capacity():
    from training.tokenizer.eval_transfer import build_encoder

    cfg = PretrainConfig(
        d_model=32, num_layers=1, num_heads=4, dim_feedforward=64,
        dft_size=256, token_granularity="sensor", multiresolution=False,
    )
    original = PipelineAModel(cfg).encoder
    legacy_config = vars(cfg).copy()
    legacy_config.pop("dft_size")
    restored = build_encoder(
        {"config": legacy_config, "encoder": original.state_dict()}, "cpu",
    )
    assert restored.filterbank.S == 256


def test_phase_a_checkpoint_can_add_a_fresh_learnable_recording_pool_for_head_training():
    """A label-free checkpoint has no classifier pool, but head training may add one explicitly."""
    from training.tokenizer.eval_transfer import build_encoder

    cfg = PretrainConfig(
        d_model=32, num_layers=1, num_heads=4, dim_feedforward=64,
        token_granularity="sensor", multiresolution=False,
    )
    original = PipelineAModel(cfg).encoder
    restored = build_encoder(
        {"config": vars(cfg).copy(), "encoder": original.state_dict()}, "cpu",
        training=True, learnable_recording_pool=True,
    )
    assert restored.recording_pool is not None
    assert restored.recording_pool.query.requires_grad


def test_eval_subset_covers_streams_before_refilling_large_source():
    keys = (
        [WindowKey(0, i, 7) for i in range(100)]
        + [WindowKey(1, i, 7) for i in range(2)]
        + [WindowKey(2, 0, 7)]
    )
    chosen = stratified_eval_subset(keys, per_label=6, seed=3)
    assert len(chosen) == 6
    assert {key.stream_i for key in chosen} == {0, 1, 2}
    assert chosen == stratified_eval_subset(keys, per_label=6, seed=3)


def test_label_group_metric_does_not_let_large_cell_hide_failure():
    true = torch.tensor([0] * 10 + [0])
    pred = torch.tensor([0] * 10 + [1])
    groups = ["large"] * 10 + ["small"]
    assert label_group_balanced_acc(pred, true, groups) == pytest.approx(0.5)


def test_resume_hydrates_applied_calibration_weights_as_frozen_state():
    cfg = PretrainConfig()
    saved = {
        "objective_calibration_mode": "apply",
        "objective_calibration_at": 2000,
        "objective_calibration_batches": 73,
        "objective_target_jepa_share": 0.4,
        "jepa_weight": 19.0,
        "vicreg_weight": 0.67,
    }
    applied = hydrate_calibrated_objective_weights(cfg, saved, saved_step=2500)
    assert applied
    assert cfg.jepa_weight == 19.0
    assert cfg.vicreg_weight == 0.67
    assert cfg.objective_calibration_at == 2000
    assert cfg.objective_calibration_batches == 73
    assert cfg.objective_target_jepa_share == 0.4
    assert cfg.objective_calibration_mode == "apply"


def test_resume_hydrates_calibration_schedule_before_it_has_run():
    cfg = PretrainConfig()
    saved = {
        **vars(PretrainConfig()),
        "objective_calibration_mode": "apply",
        "objective_calibration_at": 2000,
        "objective_calibration_batches": 50,
    }
    assert hydrate_calibrated_objective_weights(cfg, saved, saved_step=1000)
    assert cfg.objective_calibration_at == 2000
    assert cfg.objective_calibration_mode == "apply"


def test_resume_leaves_explicit_calibration_override_for_validation_to_reject():
    cfg = PretrainConfig(objective_calibration_at=7)
    hydrate_calibrated_objective_weights(
        cfg, {"objective_calibration_at": 2000}, saved_step=100,
        explicit_fields={"objective_calibration_at"},
    )
    assert cfg.objective_calibration_at == 7


def test_force_removes_all_known_run_artifacts_but_preserves_unknown_files(tmp_path):
    for name in (
        "last.pt", "log.jsonl", "run_config.json", "objective_calibration.json",
        "objective_calibration_resolved.json", "source.patch", "source_provenance.json",
        "runtime_provenance.json", "health.json", "health.txt", "telemetry.png",
    ):
        (tmp_path / name).write_text("stale")
    (tmp_path / "notes.txt").write_text("keep")

    prepare_output_dir(tmp_path, force=True, smoke=False, resume=False)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["notes.txt"]


def test_source_provenance_excludes_every_modules_runtime_outputs(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "model" / "tokenizer" / "tracked.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "model/tokenizer/tracked.py"], cwd=tmp_path, check=True)
    subprocess.run([
        "git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-qm", "initial",
    ], cwd=tmp_path, check=True)

    tracked.write_text("VALUE = 2\n")
    output = tmp_path / "training" / "evidence" / "outputs" / "run"
    output.mkdir(parents=True)
    (output / "result.json").write_text('{"metric": 1}\n')
    source = tmp_path / "training" / "tokenizer" / "new_module.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("READY = True\n")
    unrelated = tmp_path / "training" / "evidence" / "unrelated.py"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text("CHANGING = True\n")
    monkeypatch.setattr(pretrain_module, "_repo_root", lambda: tmp_path)

    provenance = capture_source_provenance(tmp_path / "unused", write=False)
    patch = provenance["_patch"].decode("utf-8")
    assert "tracked.py" in patch
    assert "new_module.py" in patch
    assert "result.json" not in patch
    assert "unrelated.py" not in patch
    assert provenance["untracked_source_files"] == ["training/tokenizer/new_module.py"]
