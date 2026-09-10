"""Released baseline encoders adapted to the application MotionSequence contract."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import baselines
from applications.motion_monitoring.data.contracts import RawRecording
from applications.motion_monitoring.sequence import (
    MotionSequence,
    _motion_sequence,
    measured_rate_hz,
    select_stream,
    stream_to_patch_batch,
)
from baselines.data import EvalStream


PRIMARY_APPLICATION_BASELINES = ("harnet", "unimts", "normwear")
OPTIONAL_APPLICATION_BASELINES = ("imagebind",)


def _artifact_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class BaselineMotionEncoder(nn.Module):
    """Expose one released frozen baseline as timestamped patch embeddings.

    The baseline keeps its published input preprocessing and receptive-field
    duration. Consecutive embeddings are emitted at ``stride_seconds`` intervals.
    No classifier, ConSE bridge, or label text is used by the application tasks.
    """

    def __init__(self, name: str, *, device: str | torch.device = "cpu") -> None:
        super().__init__()
        try:
            adapter = baselines.REGISTRY[name]
        except KeyError as error:
            raise KeyError(f"unknown baseline encoder {name!r}") from error
        if (
            adapter.__class__.window_features
            is baselines.BaselineAdapter.window_features
        ):
            raise ValueError(
                f"baseline {name!r} does not expose frozen window features"
            )
        if adapter.contract.window_sec is None:
            raise ValueError(
                f"baseline {name!r} has no declared physical receptive-field duration"
            )
        self.name = name
        self.adapter = adapter
        self.window_seconds = float(adapter.contract.window_sec)
        if self.window_seconds <= 0:
            raise ValueError("baseline receptive-field duration must be positive")
        # Existing task smokes discover device through parameters. This inert,
        # non-trainable anchor keeps the same interface without pretending the
        # external model belongs to the task optimizer.
        self._device_anchor = nn.Parameter(
            torch.empty(0, device=torch.device(device)), requires_grad=False
        )
        self._state: dict[str, Any] | None = None

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def _setup(self) -> dict[str, Any]:
        if self._state is None:
            self._state = self.adapter.setup_features(self.device)
        return self._state

    def _eval_stream(
        self,
        recording: RawRecording,
        batch,
        indices: list[int],
        length: int,
    ) -> EvalStream:
        stream = batch.stream
        windows = batch.patches[indices, :length].cpu().numpy()
        count = len(indices)
        return EvalStream(
            dataset=recording.dataset,
            stream=f"{stream.stream_id}_{stream.placement}",
            alignment="application_native",
            windows=windows,
            gt=["unlabeled"] * count,
            subjects=np.asarray([recording.subject_id] * count, dtype=object),
            channels=list(("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")),
            rate_hz=batch.sampling_rate_hz,
            mask=batch.channel_mask.cpu().numpy(),
            eval_labels=["unlabeled"],
            gravity_state=stream.gravity_state,
        )

    @torch.no_grad()
    def encode_recording(
        self,
        recording: RawRecording,
        *,
        stream_id: str | None = None,
        patch_seconds: float | None = None,
        stride_seconds: float = 1.0,
    ) -> MotionSequence:
        stream = select_stream(recording, stream_id)
        if self.name in {"harnet", "harnet_matched", "unimts"}:
            required = {"acc_x", "acc_y", "acc_z"}
            if not required.issubset(stream.channels):
                raise ValueError(f"{self.name} requires a complete acceleration triad")
            if stream.gravity_state != "present":
                raise ValueError(f"{self.name} requires gravity-present acceleration")
        duration = (
            self.window_seconds if patch_seconds is None else float(patch_seconds)
        )
        if duration <= 0:
            raise ValueError("patch duration must be positive")
        rate = measured_rate_hz(stream)
        max_samples = max(512, int(np.ceil(rate * duration * 1.1)) + 2)
        batch = stream_to_patch_batch(
            stream,
            patch_seconds=duration,
            stride_seconds=stride_seconds,
            max_patch_samples=max_samples,
        )
        groups: dict[int, list[int]] = defaultdict(list)
        for index, length in enumerate(batch.patch_len.tolist()):
            groups[int(length)].append(index)

        features: torch.Tensor | None = None
        state = self._setup()
        for length, indices in groups.items():
            eval_stream = self._eval_stream(recording, batch, indices, length)
            encoded = np.asarray(
                self.adapter.window_features(eval_stream, state, self.device),
                dtype=np.float32,
            )
            if encoded.ndim != 2 or encoded.shape[0] != len(indices):
                raise ValueError(
                    f"{self.name} returned {encoded.shape}; expected "
                    f"({len(indices)}, feature)"
                )
            if not np.isfinite(encoded).all():
                raise FloatingPointError(f"{self.name} emitted non-finite features")
            tensor = torch.from_numpy(encoded).to(self.device)
            if features is None:
                features = torch.zeros(
                    len(batch.patches),
                    tensor.shape[1],
                    dtype=tensor.dtype,
                    device=self.device,
                )
            elif tensor.shape[1] != features.shape[1]:
                raise ValueError(
                    f"{self.name} changed feature width between patch lengths"
                )
            features[torch.as_tensor(indices, device=self.device)] = tensor
        if features is None:
            raise RuntimeError(f"{self.name} did not encode any physical window")
        return _motion_sequence(recording, batch.to(self.device), features)

    def provenance(self) -> dict[str, object]:
        state = self._setup()
        artifacts = self.adapter.feature_artifacts(state)
        artifact_provenance = {}
        for name, raw_path in artifacts.items():
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"{self.name} feature artifact {name!r} is missing: {path}"
                )
            artifact_provenance[name] = {
                "path": str(path.resolve()),
                "sha256": _artifact_digest(path),
                "bytes": path.stat().st_size,
            }
        return {
            "kind": "released_frozen_baseline",
            "name": self.name,
            "window_seconds": self.window_seconds,
            "stride_seconds_default": 1.0,
            "input_contract": {
                "channels": list(self.adapter.contract.channels or ()),
                "rate_hz": self.adapter.contract.rate_hz,
                "window_sec": self.adapter.contract.window_sec,
            },
            "feature_config": self.adapter.feature_config(state),
            "artifacts": artifact_provenance,
        }
