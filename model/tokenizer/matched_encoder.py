"""Baseline backbones wearing HALO's support-classifier encoder contract (matched-corpus M2).

Purpose: answer "is our encoder the contribution, or is it the objective and curriculum?" by
swapping a baseline architecture in for :class:`SetTokenizerEncoder` while the corpus, the episode
sampler, the classifier, the readouts and the evaluation stay byte-identical. See
``docs/design/MATCHED_CORPUS_PLAN_20260915.md`` §1 (level M2).

Relationship to :mod:`model.tokenizer.baseline_backbone`
-------------------------------------------------------
That module answers a different question — *frozen released* backbones feeding the retired Phase-B
evidence engine — so it hard-codes ``freeze=True``, wraps the trunk in ``torch.no_grad()`` and
returns ``pooled=None``. All three are wrong for M2, which must train the backbone and must produce
a pooled recording vector. This module is therefore a sibling rather than a subclass; the window
reconstruction below follows the same algorithm and the crop/pad rules are imported from it so the
two cannot drift.

What M2 holds constant, and what it deliberately does not
--------------------------------------------------------
Constant: corpus, episodes, classifier, readouts, evaluation, optimiser, step budget.
Not constant: input contract. Each backbone gets the rate, duration, channel set and preprocessing
its own authors specify, because feeding a baseline something it was never designed to read would
measure our adaptation rather than its architecture. That is the same rule the released-checkpoint
rows already follow.

Capacity is reported, never matched (plan §5.6).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as audio_transforms

from training.tokenizer.pretrain_data import CHANNELS

from .baseline_backbone import _center_crop_or_wrap, _start_crop_or_wrap

ACC_SLICE = slice(0, 3)
GYRO_SLICE = slice(3, 6)

#: Input contracts, from each model's own published preprocessing. ``channels`` is how many of the
#: canonical ``CHANNELS`` order the trunk consumes; ``clip`` is the sample count it was trained on.
BACKBONE_CONTRACTS = {
    # LiMU-BERT: 20 Hz, 6-axis, one-second clips (20 samples), acceleration in g.
    "limubert": {"rate_hz": 20.0, "clip": 20, "channels": 6, "dim": 72, "crop": "clip"},
    # harnet5: 30 Hz, accelerometer triad, 5 s (150 samples).
    "harnet": {"rate_hz": 30.0, "clip": 150, "channels": 3, "dim": 512, "crop": "centre"},
}


def reconstruct_native_window(patches, patch_len, patch_padding_mask):
    """Contiguous native-rate signal ``(B, T, C)`` rebuilt from DFT-padded patches.

    Same algorithm as ``baseline_backbone.BaselineRowEncoder.forward``: patch storage is padded to
    the DFT width, so concatenating raw patch buffers would splice zeros into the signal. Indexing
    through the cumulative *valid* length skips that padding.
    """
    B, P, S, C = patches.shape
    device = patches.device
    valid_len = patch_len.clamp_min(0) * patch_padding_mask.long()
    cumulative = torch.cat([valid_len.new_zeros((B, 1)), valid_len.cumsum(dim=1)], dim=1).float()
    total = cumulative[:, -1].clamp_min(1.0)
    max_total = max(int(total.max().item()), 1)
    index = torch.arange(max_total, device=device).view(1, -1).expand(B, -1)
    index = torch.minimum(index, (total.long() - 1).unsqueeze(1))
    patch_of = (torch.searchsorted(cumulative, index.contiguous(), right=True) - 1).clamp(0, P - 1)
    offset = (index - cumulative.gather(1, patch_of)).long().clamp_min(0)
    offset = torch.minimum(offset, (patch_len.gather(1, patch_of) - 1).clamp_min(0))
    flat = patches.reshape(B, P * S, C)
    window = flat.gather(1, (patch_of * S + offset).unsqueeze(-1).expand(-1, -1, C))
    return window, total.long()


def _reinitialise(module: nn.Module) -> None:
    """Fresh weights for a random-init control, initialised per module type.

    Initialising by tensor rank instead would zero every one-dimensional parameter, which includes
    normalisation *scale* (gamma). A zero gamma outputs zeros and kills the gradient, so the arm
    would train to nothing and look like an architecture failure.
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


class _LiMUBertTrunk(nn.Module):
    """LiMU-BERT encoder stack, mean-pooled over its 20-sample clip.

    The architecture is the one vendored in ``baselines/limubert_x/adapter.py`` (an exact transcript
    of the released pretraining module), so M2 and the released-checkpoint row share a definition
    and cannot silently diverge.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        from baselines.limubert_x.adapter import _Backbone, CHECKPOINT

        self.net = _Backbone()
        if pretrained:
            state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
            self.net.load_state_dict(state, strict=False)
        self.out_dim = BACKBONE_CONTRACTS["limubert"]["dim"]

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """``(n, 20, 6)`` clips -> ``(n, dim)``."""
        return self.net(clips).mean(dim=1)


class _HarnetTrunk(nn.Module):
    """harnet5's ResNet feature extractor, randomly initialised by default.

    M2 asks what the *architecture* is worth on our corpus, so the default is random init: loading
    UK Biobank weights would reintroduce exactly the corpus advantage this arm exists to remove.
    ``pretrained=True`` is available only to reproduce the M0 row.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        from baselines.harnet.adapter import _load_harnet

        module = _load_harnet(2, torch.device("cpu"))
        self.net = getattr(module, "feature_extractor", module)
        # ``_load_harnet`` hands back a frozen, eval-mode deployment model: the released adapter only
        # ever extracts features from it. M2 trains the trunk, so re-enable gradients and training
        # mode explicitly. Without this the arm silently trains the projection alone and would
        # answer a question nobody asked.
        self.net.requires_grad_(True)
        self.net.train()
        if not pretrained:
            self.net.apply(_reinitialise)
        self.out_dim = BACKBONE_CONTRACTS["harnet"]["dim"]

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """``(n, 150, 3)`` clips -> ``(n, dim)``."""
        feature = self.net(clips.transpose(1, 2))
        return feature.flatten(1) if feature.dim() > 2 else feature


class MatchedCorpusEncoder(nn.Module):
    """A baseline trunk presented to the support classifier as a recording encoder.

    Produces the three tensors ``training.support_classifier`` consumes: ``pooled`` (B, d_model),
    ``descriptor`` and ``sensor_present``. Devices are pooled with an unweighted mean over the
    devices actually present, which is the parameter-free rule; the learned attention pool is
    HALO's own component and giving it to a baseline would confound the encoder comparison with
    HALO's fusion design.
    """

    # Guards in ``encode_batch`` read these.
    trunk = "temporal"
    token_granularity = "sensor"
    use_sensor_isolated_retrieval = False
    requires_stream_metadata = False

    def __init__(self, backbone: str, *, d_model: int = 128, pretrained: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        if backbone not in BACKBONE_CONTRACTS:
            raise ValueError(f"backbone must be one of {sorted(BACKBONE_CONTRACTS)}")
        self.backbone_name = backbone
        contract = BACKBONE_CONTRACTS[backbone]
        self.in_hz = float(contract["rate_hz"])
        self.clip = int(contract["clip"])
        self.n_channels = int(contract["channels"])
        self.crop_rule = contract["crop"]
        self.d_model = int(d_model)
        self.pretrained = bool(pretrained)

        self.net = (_LiMUBertTrunk(pretrained) if backbone == "limubert"
                    else _HarnetTrunk(pretrained))
        # A trainable projection to the classifier's width. Without it the comparison would be
        # decided by whichever trunk happens to emit the classifier's dimension.
        self.proj = nn.Sequential(
            nn.Linear(self.net.out_dim, 2 * self.d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * self.d_model, self.d_model),
        )
        self.row_norm = nn.LayerNorm(self.d_model)
        object.__setattr__(self, "_resamplers", {})

    # ------------------------------------------------------------------ helpers
    def _resample(self, x: torch.Tensor, source_hz: float) -> torch.Tensor:
        """Anti-aliased resample of ``(n, T, C)`` to the trunk's rate, caching the sinc kernel."""
        source = int(round(float(source_hz)))
        target = int(round(self.in_hz))
        if source == target:
            return x
        key = (source, target, x.device, x.dtype)
        resampler = self._resamplers.get(key)
        if resampler is None:
            resampler = audio_transforms.Resample(source, target, dtype=x.dtype).to(x.device)
            self._resamplers[key] = resampler
        return resampler(x.transpose(1, 2).contiguous()).transpose(1, 2)

    def _prepare(self, signal: torch.Tensor) -> torch.Tensor:
        """Crop or wrap-pad to the trunk's clip length, by that model's own rule."""
        if self.crop_rule == "centre":
            return _center_crop_or_wrap(signal, self.clip)
        return _start_crop_or_wrap(signal, self.clip)

    def _encode_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """``(n, T, C)`` at the trunk's rate -> ``(n, out_dim)``.

        LiMU-BERT has a 20-sample positional-embedding contract, so a longer window is split into
        non-overlapping one-second clips and averaged — the same rule its evaluation adapter uses.
        Any other trunk consumes one clip.
        """
        if self.backbone_name != "limubert":
            return self.net(self._prepare(signal))
        n, length, channels = signal.shape
        count = max(1, length // self.clip)
        usable = count * self.clip
        if usable < length:
            signal = signal[:, :usable]
        elif usable > length:
            signal = self._prepare(signal)
            usable, count = self.clip, 1
        clips = signal.reshape(n * count, self.clip, channels)
        return self.net(clips).reshape(n, count, -1).mean(dim=1)

    # ------------------------------------------------------- the encoder contract
    def forward(self, patches, rates, patch_len, role_texts, positions, *,
                patch_durations=None, resolution_ids=None, channel_mask=None,
                patch_padding_mask=None, sensor_texts=None, sensor_id=None, device_id=None,
                source_rate_hz=None, return_retrieval_tokens=False, **_ignored):
        if sensor_id is None or channel_mask is None or patch_padding_mask is None:
            raise ValueError("MatchedCorpusEncoder needs sensor_id, channel_mask and patch padding")
        B = patches.shape[0]
        device = patches.device
        window, lengths = reconstruct_native_window(patches, patch_len, patch_padding_mask)

        n_sensors = int(sensor_id.max().item()) + 1 if sensor_id.numel() else 1
        if device_id is None or not bool((device_id > 0).any()):
            device_of_sensor = torch.zeros((B, n_sensors), dtype=torch.long, device=device)
        else:
            device_of_sensor = device_id.to(device)
        n_devices = int(device_of_sensor.max().item()) + 1

        rows = patches.new_zeros((B, n_devices, self.d_model))
        present = torch.zeros((B, n_devices), dtype=torch.bool, device=device)
        rates_host = rates.detach().cpu().tolist()
        lengths_host = lengths.detach().cpu().tolist()

        prepared, destinations = [], []
        # A composite recording carries more than the six canonical channels: each device
        # contributes its own sensors, so `channel_mask` may be 12 or 24 wide. Map every channel to
        # its device through its sensor, then take that device's live channels in canonical order.
        # Scanning only the first six columns would have silently fed device 0's signal for every
        # device, which is the multi-device comparison HALO is strongest at.
        channel_device = device_of_sensor.gather(
            1, sensor_id.clamp(0, n_devices * 8).clamp(max=device_of_sensor.shape[1] - 1),
        )
        for slot in range(n_devices):
            live = channel_mask.bool() & channel_device.eq(slot)
            complete = live.sum(dim=1) >= self.n_channels
            if not bool(complete.any()):
                continue
            # Stable argsort puts the live channels first while preserving their canonical order,
            # so the trunk receives acc x/y/z (then gyro x/y/z) exactly as its authors expect.
            order = torch.argsort((~live).to(torch.int8), dim=1, stable=True)[:, :self.n_channels]
            keep = torch.nonzero(complete, as_tuple=True)[0]
            groups: dict[tuple[float, int], list[int]] = {}
            for row in keep.detach().cpu().tolist():
                groups.setdefault((float(rates_host[row]), int(lengths_host[row])), []).append(row)
            for (rate, length), indices in groups.items():
                group = torch.tensor(indices, dtype=torch.long, device=device)
                selected = order.index_select(0, group)
                signal = window.index_select(0, group)[:, :length]
                signal = torch.gather(
                    signal, 2, selected.unsqueeze(1).expand(-1, length, -1),
                ).float()
                prepared.append(self._resample(signal, rate))
                destinations.append(torch.stack([group, torch.full_like(group, slot)], dim=1))

        if prepared:
            # One trunk call per (rate, length, device) group would be thousands of tiny kernels per
            # step. Lengths differ after resampling, so pad to the longest and let the clip rule cut.
            longest = max(part.shape[1] for part in prepared)
            batch = torch.cat([self._prepare_to(part, longest) for part in prepared], dim=0)
            destination = torch.cat(destinations, dim=0)
            feature = self._encode_signal(batch)
            with torch.autocast(device_type=device.type, enabled=False):
                projected = self.row_norm(self.proj(feature.float()))
            rows[destination[:, 0], destination[:, 1]] = projected.to(rows.dtype)
            present[destination[:, 0], destination[:, 1]] = True

        if not bool(present.any()):
            raise ValueError("no window in this batch satisfied the backbone's channel contract")
        weight = present.unsqueeze(-1).to(rows.dtype)
        pooled = (rows * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

        # The residual classifier never reads the descriptor (see train.py: it is not among the
        # classifier's arguments), but ``recording_rows`` requires a present-shaped tensor. Emit a
        # deterministic unit descriptor rather than an uninitialised one.
        descriptor = patches.new_ones((B, max(n_sensors, 1), 1))
        sensor_present = torch.zeros((B, max(n_sensors, 1)), dtype=torch.bool, device=device)
        for sensor in range(max(n_sensors, 1)):
            owned = (sensor_id == sensor) & channel_mask
            sensor_present[:, sensor] = owned.any(dim=1)
        return {
            "pooled": pooled,
            "descriptor": descriptor,
            "sensor_present": sensor_present,
            "device_present": present.unsqueeze(1),
            "per_patch": None,
            "tokens": None,
            "sensor_context": None,
            "descriptor_pred": None,
            "retrieval_tokens": None,
        }

    @staticmethod
    def _prepare_to(signal: torch.Tensor, length: int) -> torch.Tensor:
        """Right-pad a resampled group to a shared length by edge repetition."""
        if signal.shape[1] >= length:
            return signal[:, :length]
        pad = signal[:, -1:].expand(-1, length - signal.shape[1], -1)
        return torch.cat([signal, pad], dim=1)


def build_matched_encoder(backbone: str, *, d_model: int = 128, pretrained: bool = False,
                          device: torch.device | None = None) -> MatchedCorpusEncoder:
    encoder = MatchedCorpusEncoder(backbone, d_model=d_model, pretrained=pretrained)
    return encoder.to(device) if device is not None else encoder
