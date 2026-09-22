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
    # LiMU-BERT-X: 10 Hz, 6-axis, two-second clips (20 samples), acceleration in g.
    #
    # The rate was 20.0 here until 2026-09-22, which fed the trunk one-second clips and silently
    # halved the physical time each positional embedding covers. Three sources agree on 10 Hz:
    # the released checkpoint's positional-embedding table is (20, 72), so the contract is 20
    # positions; the MobiCom'25 deployment paper states "we reduced the IMU data sampling rate
    # from 20 Hz to 10 Hz"; and the released adapter encodes at TARGET_HZ = 10.0 with
    # native_window_sec = 2.0. The stale "20" in the literature is the mask width (20 of 120
    # samples) and the original SenSys classifier's slice length, neither of which is the input
    # window. `tests/test_matched_encoder_fidelity.py` now pins this against the adapter itself.
    "limubert": {"rate_hz": 10.0, "clip": 20, "channels": 6, "dim": 72, "crop": "clip"},
    # harnet5: 30 Hz, accelerometer triad, 5 s (150 samples).
    # ``min_clip`` is the shortest input the trunk tolerates. harnet5's ResNet pads circularly and
    # raises on a signal shorter than its 5 s contract, so it never shortens; UniMTS's ST-GCN is
    # fully convolutional in time and accepts the real window length.
    "harnet": {"rate_hz": 30.0, "clip": 150, "min_clip": 150, "channels": 3, "dim": 512,
               "crop": "centre"},
    # UniMTS: 20 Hz, accelerometer triad, 10 s (200 samples), placed on a 22-node SMPL skeleton.
    "unimts": {"rate_hz": 20.0, "clip": 200, "min_clip": 64, "channels": 3, "dim": 512,
               "crop": "start", "trunk_chunk": 64},
}

#: UniMTS fuses placements inside its graph, so it consumes every device in one forward pass
#: instead of being pooled afterwards.
NATIVE_DEVICE_FUSION = {"unimts"}


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


_JOINT_CACHE: dict[str, int] = {}


def sensor_joint(text: str) -> int:
    """SMPL joint index for one sensor description, using UniMTS's own placement tables.

    Matching on the *sensor* text rather than the stream key is what makes multi-device work: a
    composite recording has one stream key but one placement per device, and UniMTS's whole design
    is that different placements are different graph nodes.
    """
    from baselines.unimts.adapter import (
        DEFAULT_JOINT, _PLACEMENT_KEYWORDS, _SIDE_PLACEMENT_JOINTS,
    )

    key = str(text).lower()
    if key in _JOINT_CACHE:
        return _JOINT_CACHE[key]
    joint = DEFAULT_JOINT
    for keyword, value in list(_SIDE_PLACEMENT_JOINTS) + list(_PLACEMENT_KEYWORDS):
        if str(keyword).replace("_", " ").lower() in key:
            joint = int(value)
            break
    _JOINT_CACHE[key] = int(joint)
    return int(joint)


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


class _UniMTSTrunk(nn.Module):
    """UniMTS's ST-GCN over a 22-node SMPL skeleton, randomly initialised by default.

    Constructed directly from the vendored repository rather than by loading the released
    checkpoint, because M2 asks what the *architecture* is worth on our corpus and the released
    weights carry the synthetic-mocap corpus this arm exists to remove. The surrounding CLIP text
    tower is never built: all arms use HALO's shared label-text path.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        import sys

        from baselines.unimts.adapter import UNIMTS_REPO

        # UniMTS imports ``from model import ST_GCN_18`` and HALO also owns a package named
        # ``model``. Whichever is imported first poisons the other through sys.modules, so snapshot
        # HALO's package, import against the repo-local module, then restore. Same isolation the
        # released adapter uses; the instantiated class keeps its own module object afterwards.
        saved = {name: module for name, module in list(sys.modules.items())
                 if name == "model" or name.startswith("model.")}
        for name in saved:
            sys.modules.pop(name, None)
        added = str(UNIMTS_REPO) not in sys.path
        if added:
            sys.path.insert(0, str(UNIMTS_REPO))
        try:
            from model import ST_GCN_18  # noqa: E402  (repo-local import)

            self.net = ST_GCN_18(in_channels=3, edge_importance_weighting=True)
        finally:
            for name in list(sys.modules):
                if name == "model" or name.startswith("model."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)
            if added and str(UNIMTS_REPO) in sys.path:
                sys.path.remove(str(UNIMTS_REPO))
        if pretrained:
            from baselines.unimts.adapter import UniMTSAdapter

            loaded = UniMTSAdapter().setup(torch.device("cpu"))["model"]
            self.net.load_state_dict(loaded.model.acc.state_dict())
            del loaded
        self.net.requires_grad_(True)
        self.net.train()
        self.out_dim = BACKBONE_CONTRACTS["unimts"]["dim"]

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """``(n, T, 22, 3)`` accelerations in m/s^2 -> ``(n, dim)``."""
        x = grid.permute(0, 3, 1, 2).unsqueeze(-1)      # (n, 3, T, 22, 1)
        return self.net(x).squeeze(-1).squeeze(-1)


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
                 dropout: float = 0.1, compile_trunk: bool = False):
        super().__init__()
        if backbone not in BACKBONE_CONTRACTS:
            raise ValueError(f"backbone must be one of {sorted(BACKBONE_CONTRACTS)}")
        self.backbone_name = backbone
        contract = BACKBONE_CONTRACTS[backbone]
        self.in_hz = float(contract["rate_hz"])
        self.clip = int(contract["clip"])
        self.n_channels = int(contract["channels"])
        self.crop_rule = contract["crop"]
        self.min_clip = int(contract.get("min_clip", contract["clip"]))
        self.d_model = int(d_model)
        self.pretrained = bool(pretrained)

        self.fuses_devices_natively = backbone in NATIVE_DEVICE_FUSION
        # A 22-joint x 200-frame graph at full episode batch exceeds 24 GB of activations. Gradient
        # checkpointing recomputes each chunk's activations in the backward pass instead of storing
        # them: numerically identical, roughly 30% more compute, and it keeps the RECIPE intact.
        # Shrinking episodes-per-step instead would change the objective and void the comparison.
        self.trunk_chunk = int(contract.get("trunk_chunk", 0))
        self.n_joints = 22
        self.net = (_LiMUBertTrunk(pretrained) if backbone == "limubert"
                    else _UniMTSTrunk(pretrained) if backbone == "unimts"
                    else _HarnetTrunk(pretrained))
        # A trainable projection to the classifier's width. Without it the comparison would be
        # decided by whichever trunk happens to emit the classifier's dimension.
        self.proj = nn.Sequential(
            nn.Linear(self.net.out_dim, 2 * self.d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * self.d_model, self.d_model),
        )
        self.row_norm = nn.LayerNorm(self.d_model)
        object.__setattr__(self, "_resamplers", {})
        # Compilation is opt-in: measured 1.28x on the UniMTS graph, but it costs a warm-up and it
        # recompiles per input shape. Chunking already fixes the shape except for the final partial
        # chunk, which ``_run_trunk`` pads so only one graph is ever built.
        if compile_trunk and backbone == "unimts":
            # UniMTS's ST-GCN is defined in a repo-local module also called ``model``. The adapter
            # restores HALO's package after import, so TorchDynamo cannot resolve the class's
            # globals and fails with "module 'model' has no attribute 'torch'". Compilation works
            # on the other trunks, which have no name collision.
            raise ValueError(
                "compile_trunk is unsupported for unimts: its repo-local 'model' module collides "
                "with HALO's package and TorchDynamo cannot resolve the class globals"
            )
        self.compile_trunk = bool(compile_trunk)
        object.__setattr__(
            self, "_compiled_net",
            torch.compile(self.net, dynamic=False) if compile_trunk else None,
        )

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

    def _clip_length(self, groups) -> int:
        """One clip length for the whole call, so groups concatenate into a single trunk batch.

        Real batches share a window duration, so after resampling every group has the same length;
        taking the minimum is a guard for the mixed-duration case rather than the normal path.
        """
        if self.pretrained or self.crop_rule == "clip":
            return self.clip
        shortest = min(int(part.shape[1]) for part in groups)
        return int(max(min(self.clip, shortest), self.min_clip))

    def _prepare(self, signal: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """Crop or wrap-pad to the trunk's clip length, by that model's own rule.

        When the real signal is shorter than the released clip convention and there is no released
        weight to match (a random-init M2 arm), the *real* length is used instead of wrap-padding.
        Repeating 40 frames of an 8 s window to reach UniMTS's 10 s training convention fabricates
        a quarter of the input; the graph is fully convolutional in time, so the shorter clip is
        both faithful and 1.3x faster. A ``--matched-pretrained`` arm keeps the released length so
        it can reproduce that model's own contract.
        """
        length = self.clip if length is None else int(length)
        if self.crop_rule == "centre":
            return _center_crop_or_wrap(signal, length)
        return _start_crop_or_wrap(signal, length)

    def _encode_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """One homogeneous group ``(n, T, C)`` at the trunk's rate -> ``(n, out_dim)``."""
        return self._encode_groups([signal])

    def _encode_groups(self, groups: list[torch.Tensor]) -> torch.Tensor:
        """Encode several homogeneous groups in one trunk call, each cut by its own length.

        Single-clip trunks (harnet, UniMTS) crop or wrap each group to the clip length and then
        concatenate. LiMU-BERT has a 20-sample positional-embedding contract, so a longer window
        becomes several non-overlapping one-second clips whose features are averaged -- the rule its
        released adapter uses. Groups may yield different clip counts, so they are padded to a
        common count and the average is taken over each row's REAL clips only.
        """
        if self.backbone_name != "limubert":
            length = self._clip_length(groups)
            return self._run_trunk(
                torch.cat([self._prepare(part, length) for part in groups], dim=0))

        counts, usable = [], []
        for part in groups:
            if part.shape[1] < self.clip:
                part = self._prepare(part)
            count = max(1, part.shape[1] // self.clip)
            usable.append(part[:, :count * self.clip])
            counts.append(count)
        widest = max(counts)
        padded = []
        for part, count in zip(usable, counts):
            if count < widest:
                filler = part.new_zeros(
                    (part.shape[0], (widest - count) * self.clip, part.shape[2]))
                part = torch.cat([part, filler], dim=1)
            padded.append(part)
        batch = torch.cat(padded, dim=0)
        n, _, channels = batch.shape
        features = self._run_trunk(batch.reshape(n * widest, self.clip, channels))
        features = features.reshape(n, widest, -1)
        live = torch.cat([
            torch.full((part.shape[0],), count, device=batch.device)
            for part, count in zip(padded, counts)
        ])
        keep = (torch.arange(widest, device=batch.device).unsqueeze(0) < live.unsqueeze(1))
        keep = keep.to(features.dtype).unsqueeze(-1)
        return (features * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)

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

        if self.fuses_devices_natively:
            return self._forward_native_fusion(
                window, lengths, rates, channel_mask, sensor_id, device_of_sensor,
                sensor_texts, n_devices, n_sensors, patches,
            )

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
            # Every group is cut to the trunk's own clip contract FIRST, using its own length, and
            # only then concatenated into one call. Padding groups to the batch's longest signal
            # before cutting (an earlier version) fabricated samples that the trunk consumed: for
            # LiMU-BERT they became extra one-second clips that diluted the mean, and for harnet
            # they shifted the centre crop into repeated data. Cutting per group and batching once
            # is both correct and a single kernel launch.
            feature = self._encode_groups(prepared)
            destination = torch.cat(destinations, dim=0)
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



    def _run_trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Trunk forward, gradient-checkpointed in chunks when the contract asks for it."""
        if not self.trunk_chunk or not torch.is_grad_enabled() or not self.training:
            if not self.trunk_chunk:
                return self.net(x)
            return torch.cat([self.net(x[i:i + self.trunk_chunk])
                              for i in range(0, len(x), self.trunk_chunk)], dim=0)
        from torch.utils.checkpoint import checkpoint

        runner = self._compiled_net or self.net
        out = []
        for start in range(0, len(x), self.trunk_chunk):
            chunk = x[start:start + self.trunk_chunk]
            valid = len(chunk)
            if self._compiled_net is not None and valid < self.trunk_chunk:
                # One compiled graph instead of one per trailing-chunk size. The padding rows are
                # discarded immediately and cannot influence real rows.
                chunk = torch.cat([chunk, chunk[-1:].expand(self.trunk_chunk - valid, *chunk.shape[1:])], dim=0)
            out.append(checkpoint(runner, chunk, use_reentrant=False)[:valid])
        return torch.cat(out, dim=0)

    # --------------------------------------------------- native multi-placement fusion
    def _forward_native_fusion(self, window, lengths, rates, channel_mask, sensor_id,
                               device_of_sensor, sensor_texts, n_devices, n_sensors, patches):
        """One skeleton grid per window, every device placed at its own joint.

        UniMTS fuses placements *inside* its graph, so pooling per device afterwards would replace
        its own contribution with ours. Feeding every device into one forward pass is both faithful
        to the model and the setting most favourable to it (matched-corpus plan §5.4).
        """
        device = window.device
        B = window.shape[0]
        channel_device = device_of_sensor.gather(
            1, sensor_id.clamp(max=device_of_sensor.shape[1] - 1),
        )
        rates_host = rates.detach().cpu().tolist()
        lengths_host = lengths.detach().cpu().tolist()

        # Joint per (window, device), read from the device's own sensor description.
        # Resolve joints on the host. Reading ``device_of_sensor[row, sensor]`` inside the loop
        # forced a GPU synchronisation per sensor -- roughly a thousand stalls per step -- so the
        # whole assignment is done on one CPU copy and transferred once.
        joints_host = [[0] * n_devices for _ in range(B)]
        if sensor_texts:
            device_host = device_of_sensor.detach().cpu().tolist()
            for row, texts in enumerate(sensor_texts):
                for sensor, text in enumerate(texts):
                    if text is None or sensor >= len(device_host[row]):
                        continue
                    slot = int(device_host[row][sensor])
                    if 0 <= slot < n_devices:
                        joints_host[row][slot] = sensor_joint(text)
        joints = torch.tensor(joints_host, dtype=torch.long, device=device)

        present = torch.zeros((B, n_devices), dtype=torch.bool, device=device)
        groups: dict[tuple[float, int], list[int]] = {}
        for row in range(B):
            groups.setdefault(
                (float(rates_host[row]), int(lengths_host[row])), []).append(row)

        features, owners = [], []
        for (rate, length), indices in groups.items():
            group = torch.tensor(indices, dtype=torch.long, device=device)
            signal = window.index_select(0, group)[:, :length]
            resampled = self._resample(signal.float(), rate)
            resampled = self._prepare(resampled, self._clip_length([resampled]))
            clip = resampled.shape[1]
            grid = resampled.new_zeros((len(group), clip, self.n_joints, 3))
            used = torch.zeros((len(group), n_devices), dtype=torch.bool, device=device)
            for slot in range(n_devices):
                live = channel_mask.index_select(0, group).bool() & \
                    channel_device.index_select(0, group).eq(slot)
                complete = live.sum(dim=1) >= 3
                if not bool(complete.any()):
                    continue
                order = torch.argsort((~live).to(torch.int8), dim=1, stable=True)[:, :3]
                triad = torch.gather(
                    resampled, 2, order.unsqueeze(1).expand(-1, clip, -1),
                )
                # UniMTS trains on m/s^2; our grids store g.
                triad = triad * 9.80665 * complete.view(-1, 1, 1).to(triad.dtype)
                node = joints.index_select(0, group)[:, slot]
                rows = torch.arange(len(group), device=device)
                grid[rows, :, node, :] = grid[rows, :, node, :] + triad
                used[:, slot] = complete
            features.append(self._run_trunk(grid))
            owners.append(group)
            present[group] = used

        if not bool(present.any()):
            raise ValueError("no window in this batch satisfied the backbone's channel contract")
        feature = torch.cat(features, dim=0)
        owner = torch.cat(owners, dim=0)
        with torch.autocast(device_type=device.type, enabled=False):
            projected = self.row_norm(self.proj(feature.float()))
        pooled = projected.new_zeros((B, self.d_model))
        # A window whose devices never met the channel contract was encoded from an all-zero
        # skeleton. Zero it rather than letting an empty grid's embedding look like evidence.
        pooled[owner] = (projected * present.index_select(0, owner).any(dim=1, keepdim=True)
                         .to(projected.dtype)).to(pooled.dtype)

        descriptor = patches.new_ones((B, max(n_sensors, 1), 1))
        sensor_present = torch.zeros((B, max(n_sensors, 1)), dtype=torch.bool, device=device)
        for sensor in range(max(n_sensors, 1)):
            sensor_present[:, sensor] = ((sensor_id == sensor) & channel_mask).any(dim=1)
        return {
            "pooled": pooled, "descriptor": descriptor, "sensor_present": sensor_present,
            "device_present": present.unsqueeze(1), "per_patch": None, "tokens": None,
            "sensor_context": None, "descriptor_pred": None, "retrieval_tokens": None,
        }

    @staticmethod
    def _prepare_to(signal: torch.Tensor, length: int) -> torch.Tensor:
        """Right-pad a resampled group to a shared length by edge repetition."""
        if signal.shape[1] >= length:
            return signal[:, :length]
        pad = signal[:, -1:].expand(-1, length - signal.shape[1], -1)
        return torch.cat([signal, pad], dim=1)


def build_matched_encoder(backbone: str, *, d_model: int = 128, pretrained: bool = False,
                          device: torch.device | None = None,
                          compile_trunk: bool = False) -> MatchedCorpusEncoder:
    encoder = MatchedCorpusEncoder(backbone, d_model=d_model, pretrained=pretrained,
                                   compile_trunk=compile_trunk)
    return encoder.to(device) if device is not None else encoder
