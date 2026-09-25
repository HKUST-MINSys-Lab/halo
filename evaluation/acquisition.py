"""HALO's runtime acquisition vector per stream (device/placement prose + modality, gravity and rate
facts through the checkpoint's own conditioner). Moved from the rung-2 runner so every rung can use
it without depending on that runner."""

from __future__ import annotations

import weakref

import numpy as np
import torch
import torch.nn.functional as F

from baselines.data import EvalStream, MultiDeviceEvalStream
from training.tokenizer.pretrain_data import (
    STREAM_SOURCE_RATE_HZ, _stream_gravity_state, modalities_present, stream_sensor_texts,
    structured_sensor_metadata,
)

_HALO_ACQUISITION_VECTOR_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


@torch.no_grad()
def halo_acquisition_vector(
    stream: EvalStream | MultiDeviceEvalStream,
    encoder: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """One runtime acquisition vector for a stream, using the checkpoint's own conditioner.

    The vector contains only information available at inference: device/placement prose plus exact
    modality, gravity and effective-rate facts. Composite devices are balanced exactly like the
    recording path: sensors within device, then devices within recording.
    """
    if not hasattr(encoder, "descriptor_proj"):
        return np.zeros((int(getattr(encoder, "d_model")),), dtype=np.float32)
    members = stream.devices if isinstance(stream, MultiDeviceEvalStream) else [stream]
    cache_key = tuple((
            member.dataset, member.stream, float(member.rate_hz),
            (None if member.effective_source_rate_hz is None
             else float(member.effective_source_rate_hz)),
            tuple(np.asarray(member.mask, dtype=np.bool_).tolist()),
            member.gravity_state, member.perturbation,
        ) for member in members)
    encoder_cache = _HALO_ACQUISITION_VECTOR_CACHE.setdefault(encoder, {})
    cached = encoder_cache.get(cache_key)
    if cached is not None:
        return cached.copy()
    per_device = []
    for member in members:
        mask = torch.as_tensor(member.mask, dtype=torch.bool)
        modalities = modalities_present(mask.tolist())
        gravity = member.gravity_state or _stream_gravity_state(member.dataset, member.stream)
        _, sensor_texts, _ = stream_sensor_texts(
            member.dataset, member.stream,
            gravity_removed=gravity == "removed",
            has_accel="accel" in modalities, has_gyro="gyro" in modalities,
            neutral=bool(getattr(encoder, "neutral_acquisition_text", False)),
            conditioning_schema=getattr(encoder, "conditioning_schema", "legacy-combined-text-v1"),
        )
        descriptors, inverse = encoder.encode_sensor_descriptors_unique([sensor_texts], device)
        descriptor = descriptors.index_select(0, inverse.clamp_min(0).reshape(-1)).reshape(
            1, len(sensor_texts), -1,
        )
        text_embedding = encoder.descriptor_proj.embed(descriptor)
        if hasattr(encoder, "structured_conditioner"):
            modality, gravity_ids = structured_sensor_metadata(
                has_accel="accel" in modalities, has_gyro="gyro" in modalities,
                gravity_state=gravity,
            )
            source_rate = min(
                float(member.rate_hz),
                float(member.effective_source_rate_hz
                      if member.effective_source_rate_hz is not None
                      else STREAM_SOURCE_RATE_HZ.get(
                          f"{member.dataset}/{member.stream}", member.rate_hz,
                      )),
            )
            rates = torch.tensor(
                [[[float(member.rate_hz), source_rate]] * len(sensor_texts)],
                dtype=torch.float32, device=device,
            )
            valid = torch.ones((1, len(sensor_texts)), dtype=torch.bool, device=device)
            structured = encoder.structured_conditioner.embed(
                modality.unsqueeze(0).to(device), gravity_ids.unsqueeze(0).to(device), rates, valid,
            )
            acquisition = F.normalize(text_embedding.float() + structured.float(), dim=-1)
        else:
            acquisition = F.normalize(text_embedding.float(), dim=-1)
        per_device.append(acquisition.mean(dim=1).squeeze(0))
    value = F.normalize(torch.stack(per_device).mean(dim=0), dim=-1).cpu().numpy().astype(np.float32)
    encoder_cache[cache_key] = value
    return value.copy()


def halo_acquisition_rows(
    stream: EvalStream | MultiDeviceEvalStream,
    encoder: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    vector = halo_acquisition_vector(stream, encoder, device)
    return np.broadcast_to(vector, (stream.n_windows, len(vector))).copy()
