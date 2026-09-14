"""Shared encoder plumbing for the retained support-classification control.

This module deliberately contains only the batch-to-encoder contract used by
``training.support_classifier``.  It does not retain the historical Phase-B
episode planner, retrieval bank, or admissibility machinery.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from model.tokenizer.encoder import ROPE_MIN_PERIOD_S, SetTokenizerEncoder
from training.tokenizer.pretrain import DFT_SIZE, TRAIN_DATASETS


def materialize_deferred_patches(batch: dict, device: torch.device) -> torch.Tensor:
    """Reconstruct the reference DFT-padded patches from one compact recording transfer."""
    compact = batch.get("compact_data")
    if compact is None:
        return batch["patches"].to(device, non_blocking=True)
    if batch.get("patches") is not None:
        raise ValueError("a deferred batch cannot carry both compact data and materialized patches")
    values = compact.to(device, non_blocking=True)
    starts = batch["patch_start_samples"].to(device, non_blocking=True)
    lengths = batch["patch_len"].to(device, non_blocking=True)
    B, samples, channels = values.shape
    P = starts.shape[1]
    offset = torch.arange(DFT_SIZE, device=device).view(1, 1, DFT_SIZE)
    valid = offset < lengths.unsqueeze(-1)
    source = (starts.unsqueeze(-1) + offset).clamp(max=max(samples - 1, 0))
    source = source + torch.arange(B, device=device).view(B, 1, 1) * samples
    patches = values.reshape(B * samples, channels).index_select(0, source.reshape(-1))
    patches = patches.reshape(B, P, DFT_SIZE, channels)
    return patches * valid.unsqueeze(-1).to(patches.dtype)


def encode_batch(encoder: SetTokenizerEncoder, batch: dict, device: torch.device) -> dict:
    """Encode a heterogeneous batch while preserving the encoder's metadata contract."""
    if encoder.trunk != "temporal" or encoder.token_granularity != "sensor":
        raise ValueError("support classification requires a temporal sensor-granularity encoder")
    metadata = {}
    if getattr(encoder, "requires_stream_metadata", False):
        metadata = {
            "streams": batch.get("streams"),
            "sources": batch.get("sources"),
            "gravity_state": batch.get("gravity_state"),
        }
    return encoder(
        materialize_deferred_patches(batch, device),
        batch["rates"].to(device, non_blocking=True),
        batch["patch_len"].to(device, non_blocking=True),
        batch["role_texts"],
        batch["positions"].to(device, non_blocking=True),
        patch_durations=batch["patch_durations"].to(device, non_blocking=True),
        resolution_ids=(batch["resolution_ids"].to(device, non_blocking=True)
                        if "resolution_ids" in batch else None),
        channel_mask=batch["channel_mask"].to(device, non_blocking=True),
        patch_padding_mask=batch["patch_padding_mask"].to(device, non_blocking=True),
        sensor_texts=batch["sensor_texts"],
        sensor_id=batch["sensor_id"].to(device, non_blocking=True),
        device_id=(batch["device_id"].to(device, non_blocking=True)
                   if batch.get("device_id") is not None else None),
        source_rate_hz=(batch["channel_source_rates"] if batch.get("channel_source_rates") is not None
                        else batch["source_rates"]).to(device, non_blocking=True),
        return_retrieval_tokens=True,
        **metadata,
    )


def autocast(device: torch.device):
    """Use the project-wide CUDA mixed-precision policy without a CPU special case."""
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext())


def install_compiled_transformer(encoder: SetTokenizerEncoder) -> bool:
    """Compile only the stable tensor core, leaving ragged text/batch orchestration eager.

    The runtime hook is checkpoint-neutral: state-dict names and serialized architecture remain
    unchanged. Dynamic shapes cover the small set of channel-width buckets used by multi-device
    training without compiling a separate model.
    """
    if next(encoder.parameters()).device.type != "cuda":
        return False
    import torch._dynamo.config as dynamo_config
    import torch._functorch.config as functorch_config

    functorch_config.donated_buffer = False
    functorch_config.backward_pass_autocast = "off"
    # A requested compiler must either compile or fail loudly.  Silent eager fallback makes the
    # run log claim an optimization that is not actually active and invalidates speed estimates.
    dynamo_config.suppress_errors = False
    encoder._compiled_transformer_forward = torch.compile(
        encoder.transformer.forward, dynamic=True,
    )
    return True


def build_random_encoder(
    device: torch.device,
    frontend: str = "fixed",
    *,
    neutral_acquisition_text: bool = False,
    duration_range: tuple[float, float] | None = None,
    num_resolutions: int = 2,
    frontend_kwargs: dict | None = None,
) -> tuple[SetTokenizerEncoder, dict]:
    """Construct the small random-init encoder used by support-control smoke runs."""
    use_duration_embedding = duration_range is not None
    duration_min, duration_max = duration_range or (0.4, 1.5)
    frontend_kwargs = dict(frontend_kwargs or {})
    rope_min_period = float(frontend_kwargs.get("rope_min_period", ROPE_MIN_PERIOD_S))
    frontend_kwargs["rope_min_period"] = rope_min_period
    config = {
        "frontend": frontend,
        "d_model": 128,
        "dft_size": DFT_SIZE,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "trunk": "temporal",
        "descriptor_prediction": False,
        "text_conditioning": "factored",
        "token_granularity": "sensor",
        "sensor_bias_dim": 14,
        "use_sensor_bias_conditioning": False,
        "use_sensor_isolated_retrieval": False,
        "neutral_acquisition_text": bool(neutral_acquisition_text),
        "learnable_recording_pool": True,
        "multiresolution": False,
        "use_duration_embedding": use_duration_embedding,
        "duration_min_seconds": float(duration_min),
        "duration_max_seconds": float(duration_max),
        "num_resolutions": int(num_resolutions),
        "rope_min_period": rope_min_period,
        "short_patch_choices": [0.4],
        "long_patch_choices": [1.5],
        "val_resolution_pair": [0.5, 1.5],
        "train_datasets": list(TRAIN_DATASETS),
    }
    # Serialize fixed-filterbank analysis choices with random-init controls.  Phase-A checkpoints
    # already carry their own config and are reconstructed by eval_transfer instead.
    for key in ("use_polarization", "polarization_energy_kappa"):
        if key in frontend_kwargs:
            config[key] = frontend_kwargs[key]
    encoder = SetTokenizerEncoder(
        d_model=128,
        num_layers=3,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1,
        dft_size=DFT_SIZE,
        frontend=frontend,
        trunk="temporal",
        descriptor_prediction=False,
        text_conditioning="factored",
        token_granularity="sensor",
        use_sensor_bias_conditioning=False,
        learnable_recording_pool=True,
        use_duration_embedding=use_duration_embedding,
        duration_min_seconds=duration_min,
        duration_max_seconds=duration_max,
        num_resolutions=num_resolutions,
        **frontend_kwargs,
    ).to(device)
    return encoder, config
