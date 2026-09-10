"""One-batch activation and gradient check for the default future-JEPA model.

This uses real corpus windows on CPU and exercises the same student, EMA-teacher, predictor,
physical decoder, and patch-collapse paths as the trainer. It is diagnostic only; it does not
update a checkpoint.

Run: /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.grad_check
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.tokenizer.future_jepa import (
    balanced_future_latent_loss,
    balanced_physical_loss,
    combine_future_losses,
    fixed_filterbank_physical_targets,
    gather_token_rows,
    make_future_target_plan,
    normalized_teacher_target,
    patch_variance_covariance,
)
from training.tokenizer.losses_repr import fold_analysis_to_sensors
from training.tokenizer.pretrain import PipelineAModel, PretrainConfig
from training.tokenizer.pretrain_data import (
    SEED,
    CorpusIndex,
    MultiResolutionCollate,
    PretrainDataset,
)

OUT = Path(__file__).resolve().parent / "outputs" / "grad_check"


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def module_grad_norm(module: nn.Module) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cpu")
    cfg = PretrainConfig(
        d_model=64,
        num_layers=2,
        num_heads=4,
        dim_feedforward=128,
        device=str(device),
        text_conditioning="factored",
        token_granularity="sensor",
        multiresolution=True,
        future_teacher_top_layers=2,
        future_predictor_dim=64,
        future_predictor_heads=4,
        descriptor_weight=0.0,
    )
    index = CorpusIndex(max_per_stream=200, seed=SEED)
    diagnostic_keys = [
        index.train[i]
        for i in np.linspace(0, len(index.train) - 1, 32, dtype=np.int64)
    ]
    dataset = PretrainDataset(index, diagnostic_keys, augment=False, two_view=False)
    batch = MultiResolutionCollate(
        fixed_patch_seconds=cfg.future_patch_durations, seed=SEED,
    )([dataset[i] for i in range(len(diagnostic_keys))])

    model = PipelineAModel(cfg).to(device)
    frontend = model.encoder.filterbank
    frontend.reset_norm_accumulator()
    frontend.accumulate_norm_stats(
        batch["patches"], batch["rates"], batch["patch_len"],
        patch_mask=batch["patch_padding_mask"],
        channel_mask=batch["channel_mask"],
        source_rate_hz=batch["source_rates"],
    )
    frontend.finalize_norm_stats()
    target_frontend = model.physical_target_analyzer
    target_frontend.load_state_dict(frontend.state_dict())
    teacher = copy.deepcopy(model.encoder).eval().requires_grad_(False)

    patches = batch["patches"].float()
    rates = batch["rates"]
    lengths = batch["patch_len"]
    positions = batch["positions"]
    durations = batch["patch_durations"]
    resolutions = batch["resolution_ids"]
    patch_valid = batch["patch_padding_mask"]
    channel_mask = batch["channel_mask"]
    sensor_id = batch["sensor_id"]
    n_sensors = max(map(len, batch["sensor_texts"]))

    plan = make_future_target_plan(
        batch["patch_starts"], batch["patch_ends"], patch_valid, resolutions,
        context_fraction=cfg.future_context_fraction,
        horizon_bins_seconds=cfg.future_horizon_bins_seconds,
        generator=torch.Generator().manual_seed(SEED),
    )
    analysis = model.encoder.analyze(
        patches, rates, lengths, source_rate_hz=batch["source_rates"],
    )
    sensor_tokens = model.encoder.project_tokens(
        analysis, sensor_id=sensor_id, channel_mask=channel_mask,
        n_sensors=n_sensors,
    )
    descriptors, descriptor_ids = model.encoder.encode_sensor_descriptors_unique(
        batch["sensor_texts"], device,
    )
    student_tokens = sensor_tokens * plan.context_mask[:, :, None, None].to(sensor_tokens.dtype)
    student = model.encoder.encode(
        student_tokens, None, None, positions,
        patch_durations=durations,
        resolution_ids=resolutions,
        channel_mask=channel_mask,
        patch_padding_mask=plan.context_mask,
        sensor_descriptors=descriptors,
        sensor_id=sensor_id,
        sensor_text_ids=descriptor_ids,
        return_retrieval_tokens=False,
    )

    with torch.no_grad():
        teacher_tokens = teacher.project_tokens(
            analysis.detach(), sensor_id=sensor_id, channel_mask=channel_mask,
            n_sensors=n_sensors,
        )
        teacher_output = teacher.encode(
            teacher_tokens, None, None, positions,
            patch_durations=durations,
            resolution_ids=resolutions,
            channel_mask=channel_mask,
            patch_padding_mask=patch_valid,
            sensor_descriptors=descriptors,
            sensor_id=sensor_id,
            sensor_text_ids=descriptor_ids,
            return_retrieval_tokens=False,
            return_layer_states=True,
        )
    teacher_grid = normalized_teacher_target(
        teacher_output["layer_states"], top_k=cfg.future_teacher_top_layers,
    )
    context_valid = plan.context_mask.unsqueeze(2) & student["sensor_present"].unsqueeze(1)
    prediction, target_indices, query_valid = model.future_predictor(
        student["tokens"], context_valid, plan.target_mask,
        positions, durations, resolutions, plan.horizon_seconds,
        student["descriptor"], student["sensor_present"],
    )
    teacher_targets = gather_token_rows(teacher_grid, target_indices)
    query_resolutions = resolutions.gather(1, target_indices[..., 0])
    future_loss = balanced_future_latent_loss(
        prediction, teacher_targets, query_valid, query_resolutions,
        num_resolutions=len(cfg.future_patch_durations),
    )

    target_analysis = target_frontend.analyze(
        patches, rates, lengths, source_rate_hz=batch["source_rates"],
    )
    physical_channels, physical_channel_valid = fixed_filterbank_physical_targets(
        target_analysis,
        n_bands=target_frontend.n_bands,
        use_resolution_mask=target_frontend.use_resolution_mask,
        use_amplitude=target_frontend.use_amplitude,
        use_dc=target_frontend.use_dc,
        include_amplitude=False,
    )
    physical_grid, _ = fold_analysis_to_sensors(
        physical_channels, sensor_id, channel_mask, n_sensors=n_sensors,
    )
    physical_valid_grid, _ = fold_analysis_to_sensors(
        physical_channel_valid.to(physical_channels.dtype),
        sensor_id, channel_mask, n_sensors=n_sensors,
    )
    physical_targets = gather_token_rows(physical_grid, target_indices)
    physical_valid = gather_token_rows(physical_valid_grid, target_indices)
    physical_prediction = model.physical_decoder(prediction)
    physical_loss = balanced_physical_loss(
        physical_prediction, physical_targets, query_valid, query_resolutions,
        physical_valid, num_resolutions=len(cfg.future_patch_durations),
    )
    collapse = patch_variance_covariance(
        student["tokens"], context_valid,
        resolution_ids=resolutions,
        num_resolutions=len(cfg.future_patch_durations),
        variance_weight=cfg.collapse_variance_weight,
        covariance_weight=cfg.collapse_covariance_weight,
        target_std=cfg.collapse_target_std,
    )
    loss = combine_future_losses(
        future_loss, physical_loss, collapse.total,
        future_weight=cfg.jepa_weight,
        physical_weight=cfg.physical_weight,
        collapse_weight=cfg.collapse_weight,
    )
    model.zero_grad(set_to_none=True)
    loss.total.backward()

    active_modules = {"encoder": model.encoder, **model.pretraining_heads()}
    gradients = {name: module_grad_norm(module) for name, module in active_modules.items()}
    dead = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and (parameter.grad is None or not bool(parameter.grad.any()))
    ]
    leakage = int((plan.context_mask & plan.target_mask).sum())
    leakage += int((plan.context_mask
                    & (batch["patch_ends"] > plan.context_end[:, None] + 1e-7)).sum())
    leakage += int((plan.target_mask
                    & (batch["patch_starts"] < plan.context_end[:, None] - 1e-7)).sum())
    report = {
        "device": str(device),
        "batch": [int(patches.shape[0]), int(patches.shape[1]), n_sensors],
        "loss": {
            "total": float(loss.total.detach()),
            "future": float(future_loss.detach()),
            "physical": float(physical_loss.detach()),
            "collapse": float(collapse.total.detach()),
        },
        "activation_rms": {
            "sensor_tokens": rms(sensor_tokens),
            "visible_student_states": rms(student["tokens"][context_valid]),
            "teacher_targets": rms(teacher_targets[query_valid]),
            "future_predictions": rms(prediction[query_valid]),
            "physical_predictions": rms(physical_prediction[query_valid]),
        },
        "gradient_norms": gradients,
        "dead_parameters": dead,
        "planner": {
            "eligible_fraction": float(plan.eligible.float().mean()),
            "target_queries": int(query_valid.sum()),
            "leakage_count": leakage,
        },
        "checks": {
            "teacher_frozen": not any(parameter.requires_grad for parameter in teacher.parameters()),
            "text_encoder_frozen": not any(
                parameter.grad is not None for parameter in model.encoder.text_encoder.parameters()
            ),
            "no_dead_trainable_parameters": not dead,
            "all_active_modules_receive_gradients": all(value > 0 for value in gradients.values()),
            "finite_loss": bool(torch.isfinite(loss.total)),
            "future_context_has_no_leakage": leakage == 0,
            "future_targets_exist": bool(query_valid.any()),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"GRAD CHECK: {'PASS' if all(report['checks'].values()) else 'ISSUES'}")


if __name__ == "__main__":
    main()
