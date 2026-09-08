"""Held-out-config transfer probe for a frozen Phase-1 encoder.

The internal pretraining val-kNN is subject-disjoint but WITHIN the training datasets.
This measures the thing we actually care about: does the frozen representation cluster
activities on **held-out eval datasets** (unseen placements/devices/subjects)? For each
eval dataset we encode its windows with the frozen encoder and run a subject-disjoint
kNN balanced accuracy over that dataset's OWN labels — pure representation transfer, no
ConSE/text head involved (that's Pipeline B).

NOT the baseline-table number (which is ConSE macro-F1) — it's a fast, honest transfer
signal to decide whether Pipeline A cleared the bar before building Pipeline B.

Run:  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.eval_transfer \
        --checkpoint training/tokenizer/outputs/pretrain_native/best.pt
      # NB: pretrain_native/best.pt is the real trained model (val_ba 0.659). The default
      # outputs/pretrain/ dir holds only smoke/debug runs — do NOT evaluate that one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data.scripts.eda.grid_io import discover_grids
from model.tokenizer.encoder import SetTokenizerEncoder
from training.tokenizer.pretrain_data import (DFT_SIZE, modalities_present,
                                              stream_channel_descriptions,
                                              stream_sensor_bias, stream_sensor_texts,
                                              _stream_gravity_state,
                                              MultiResolutionCollate, MultiScaleCollate,
                                              STREAM_SOURCE_RATE_HZ, VAL_RESOLUTION_PAIR)
from data.scripts.curate.deployment_policy import PHASE_A_TRANSFER_DATASETS, stream_specs

# Held-out eval streams are derived from the deployment policy, not maintained as a second manual
# roster. Multi-stream datasets retain one cell per placement and are aggregated dataset-first below.
EVAL_STREAMS = tuple(
    (dataset, spec.stream_id)
    for dataset in PHASE_A_TRANSFER_DATASETS
    for spec in stream_specs(dataset, "primary")
)
PATCH_SECONDS = 1.0
KNN_K = 5
SEED = 20260718

# Phase-A CHECKPOINT SELECTION roster — named explicitly as (dataset, stream) pairs rather than
# derived from EVAL_STREAMS, because the selection sources are deliberately NOT the "primary"
# deployment streams and must not drift when that policy changes.
#
# WHY WIDER THAN THE THREE PHONE COHORTS IT REPLACES. The 2026-08-17 regression was largest on
# WRIST cells (SPAR -0.209, Upper Limb Use -0.178) and on RealWorld's static postures (-0.142),
# while the phone-pocket cohorts moved least (MotionSense -0.070). A selection metric built only
# from phone pockets is close to blind to the exact failure it exists to catch. ExtraSensory
# contributes free-living wrist and in-hand cells at a 25 Hz acquisition clock stored at 50 Hz.
#
# EXCLUSIVITY. Every dataset here is development-only: never in a Phase-A training corpus, never in
# the Phase-B test roster, never in the sealed confirmation roster. ExtraSensory is listed in
# `OPTIONAL_PHASE_A_DATASETS` as a scale source; selecting on it forfeits that use, and
# `assert_selection_roster_is_untrained` enforces the separation at run start rather than trusting
# a comment.
# `scored` decides whether a source enters the selection SCALAR. Unscored sources are still encoded
# and reported, including their posture canary; they are diagnostics, not optimization targets.
#
# ExtraSensory is deliberately unscored. Measured 2026-08-18 across three encoders, its score is
# MONOTONICALLY ANTI-CORRELATED with training: random init 0.307, failed 27k 0.274, old-good 4k
# 0.251. Folding an anti-correlated term into the scalar biases selection toward undertrained
# checkpoints, which is a worse defect than the phone-only blindness it was added to fix. Its
# free-living self-reported labels are the likely cause. It is retained because its WRIST posture
# canary is the orientation instrument this roster exists for, and that canary is read directly.
PHASE_A_SELECTION_STREAMS = (
    ("motionsense", "phone_front_pocket", True),
    ("realworld", "phone_waist", True),
    ("shoaib", "phone_right_pocket", True),
    ("extrasensory", "watch_wrist", False),
    ("extrasensory", "phone_hand", False),
)
PHASE_A_SELECTION_DATASETS = tuple(dict.fromkeys(d for d, _, _ in PHASE_A_SELECTION_STREAMS))
PHASE_A_SCORED_SELECTION_DATASETS = tuple(
    dict.fromkeys(d for d, _, scored in PHASE_A_SELECTION_STREAMS if scored)
)

# Measured floor for any Phase-A arm: a randomly initialised encoder of the reference architecture,
# scored by this exact function through the isolated retrieval rows (2026-08-18, seed 20260718).
# The fixed physical filterbank makes a random trunk a genuinely strong kNN baseline -- a random
# projection roughly preserves the distances the tokenizer already encodes -- so an arm that does
# not clear this floor has learned nothing usable for retrieval.
#
#   random init      0.8012          <- this floor
#   old-good 4k      0.8577          (+0.057; the entire measured value of Phase-A SSL to date)
#   rejected 27k     0.7161          (-0.085; training below random init)
#
# The margin is small. Treat "beat the floor" as necessary, not sufficient.
PHASE_A_RANDOM_INIT_FLOOR = 0.8012

# Selection runs every `selection_every` steps inside training, so it must stay cheap and stable.
# ExtraSensory alone carries 659k windows across its three streams; encoding them all would cost
# more than the training interval it is meant to score. The subsample is stratified by
# (label, subject) and drawn from a fixed seed, so the metric is comparable across arms and steps.
SELECTION_MAX_WINDOWS_PER_STREAM = 6_000

# Static postures are separated ONLY by the gravity direction in the device frame: they carry no
# distinguishing periodic content. A representation trained to be orientation-invariant collapses
# them to chance while its overall score decays only gently, which is precisely how the 2026-08-17
# failure hid. Tracking this subset turns that failure mode into a monitored metric.
POSTURE_LABELS = ("lying", "sitting", "standing")


def subject_holdout(subjects: np.ndarray, dataset: str) -> set:
    """Deterministic 50% holdout, shared by every stream of the same cohort."""
    unique = sorted(set(subjects.tolist()))
    dataset_seed = SEED + int(hashlib.sha256(dataset.encode()).hexdigest()[:8], 16) % 1_000_000
    rng = np.random.default_rng(dataset_seed)
    rng.shuffle(unique)
    return set(unique[:max(1, len(unique) // 2)])


def build_encoder(ckpt: dict, device, *, training: bool = False) -> torch.nn.Module:
    c = ckpt["config"]
    backbone = c.get("encoder_backbone")
    if backbone in {"harnet", "unimts"}:
        from model.tokenizer.baseline_backbone import BaselineRowEncoder

        if not bool(c.get("freeze_backbone", True)):
            raise ValueError(
                "unsupported legacy checkpoint: the encoder comparison permits only frozen "
                "third-party backbones"
            )
        enc = BaselineRowEncoder(
            backbone, d_model=int(c.get("d_model", 128)), freeze=True, device=device,
        ).to(device)
        enc.load_state_dict(ckpt["encoder"])
        enc.retrieval_granularity = c.get("retrieval_granularity", "window")
        return enc.train() if training else enc.eval()
    frontend = c.get("frontend", "fixed")
    sensor_bias_dim = c.get("sensor_bias_dim")
    if sensor_bias_dim is None:
        # Compatibility for the short-lived sensor checkpoints written before the dimension was
        # serialized. Infer the exact historical input width instead of guessing and failing load.
        bias_weight = ckpt.get("encoder", {}).get("bias_proj.mlp.0.weight")
        sensor_bias_dim = int(bias_weight.shape[1]) if bias_weight is not None else 14
    use_sensor_bias_conditioning = c.get("use_sensor_bias_conditioning")
    if use_sensor_bias_conditioning is None:
        # Historical sensor-granularity checkpoints always carried this projection. New checkpoints
        # serialize the flag and leave measured stream statistics outside the encoder trunk.
        use_sensor_bias_conditioning = any(
            key.startswith("bias_proj.") for key in ckpt.get("encoder", {})
        )
    descriptor_prediction = c.get("descriptor_prediction")
    if descriptor_prediction is None:
        descriptor_prediction = any(
            key.startswith("descriptor_head.") for key in ckpt.get("encoder", {})
        )
    kw = dict(
        d_model=c["d_model"], num_layers=c["num_layers"], num_heads=c["num_heads"],
        dim_feedforward=c["dim_feedforward"],
        dropout=float(c.get("dropout", 0.1)) if training else 0.0,
        dft_size=DFT_SIZE,
        frontend=frontend,                                  # reconstruct the ACTUAL arm (was: always filterbank)
        trunk=c.get("trunk", "dual"),
        descriptor_prediction=bool(descriptor_prediction),
        use_duration_embedding=(c.get("multiresolution", False)
                                and c.get("token_granularity", "channel") == "channel"),
        duration_min_seconds=min(c.get("short_patch_choices", (0.4,))),
        duration_max_seconds=max(c.get("long_patch_choices", (1.5,))),
        duration_gate_init=c.get("duration_gate_init", 0.1),
        rope_min_period=0.4 if c.get("multiresolution", False) else 0.5,
        text_conditioning=c.get("text_conditioning", "per_channel"),  # reconstruct the ACTUAL arm
        token_granularity=c.get("token_granularity", "channel"),
        sensor_bias_dim=int(sensor_bias_dim),
        use_sensor_bias_conditioning=bool(use_sensor_bias_conditioning),
        use_sensor_isolated_retrieval=bool(c.get("use_sensor_isolated_retrieval", False)),
        gate_bias_init=c.get("gate_bias_init", -2.0),
    )
    kw.update(                                              # fixed / learnable filterbank hyperparams
        center_shift_fraction=c.get("center_shift_fraction", 0.45),
        bandwidth_factor_max=c.get("bandwidth_factor_max", 1.5),
        compression_gain_max=c.get("compression_gain_max", 2.0),
        filter_shape_min=c.get("filter_shape_min", 1.5),
        filter_shape_max=c.get("filter_shape_max", 2.5),
        adaptive_gate_init=c.get("adaptive_gate_init", 0.1),
    )
    enc = SetTokenizerEncoder(**kw)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval_resolution_pair = tuple(c.get("val_resolution_pair", VAL_RESOLUTION_PAIR))
    enc.eval_patch_seconds = float(c.get("patch_seconds", PATCH_SECONDS))
    enc.min_resolution_ratio = float(c.get("min_resolution_ratio", 1.75))
    enc.multiresolution = bool(c.get("multiresolution", False))
    enc.retrieval_granularity = c.get("retrieval_granularity", "patch")
    enc = enc.to(device)
    return enc.train() if training else enc.eval()


#: Fixed probe spec. A window length that does NOT divide evenly into the patch grid, so the probe
#: exercises partial-tail handling (duration-weighted pooling + the tail floor) — the exact code the
#: F1/F7 fixes changed. Never edit these numbers: doing so invalidates every stored fingerprint.
_PROBE = dict(n=2, samples=128, rate=50.0, dataset="wisdm", stream="phone_pocket")


@torch.no_grad()
def embedding_fingerprint(enc, device) -> torch.Tensor:
    """Deterministic BEHAVIOURAL fingerprint of the whole embedding path -> (n, d) float32 cpu.

    Weights + corpus + vocab fingerprints cannot detect a change to the *code* that turns them into
    an embedding (pooling, tail selection, text construction). A pre-F1-pooling memory bank paired
    with a post-F1 encoder passed every existing guard while storing vectors from a different
    function. This runs a FIXED synthetic window through the real ``encode_dataset`` path, so ANY
    behavioural change to that path shows up as a different fingerprint — no version bump to
    remember. Compared with a tolerance (not hashed) so CPU/GPU float noise can't false-alarm.
    """
    g = torch.Generator().manual_seed(20260724)
    data = torch.randn(_PROBE["n"], _PROBE["samples"], 6, generator=g).numpy()
    texts = [f"probe channel {i}" for i in range(6)]
    return encode_dataset(enc, data, texts, device, _PROBE["rate"], gravity_state="present",
                          channel_mask=[True] * 6, dataset=_PROBE["dataset"],
                          stream=_PROBE["stream"]).float().cpu()


@torch.no_grad()
def patch_embedding_fingerprint(enc, device) -> torch.Tensor:
    """Deterministic fingerprint of valid patch vectors and their structural metadata."""
    g = torch.Generator().manual_seed(20260724)
    data = torch.randn(_PROBE["n"], _PROBE["samples"], 6, generator=g).numpy()
    texts = [f"probe channel {i}" for i in range(6)]
    encoded = encode_dataset_detailed(
        enc, data, texts, device, _PROBE["rate"], gravity_state="present",
        channel_mask=[True] * 6, dataset=_PROBE["dataset"], stream=_PROBE["stream"],
    )
    pieces = [
        encoded["patch_Z"].float().flatten().cpu(),
        encoded["patch_window"].float().cpu(),
        encoded["patch_time"].float().cpu(),
        encoded["patch_duration"].float().cpu(),
        encoded["patch_resolution"].float().cpu(),
    ]
    return torch.cat(pieces)


@torch.no_grad()
def sensor_embedding_fingerprint(enc, device) -> torch.Tensor:
    """Deterministic fingerprint of the per-(patch, sensor) export used by schema-4 banks."""
    if getattr(enc, "token_granularity", "channel") != "sensor":
        raise ValueError("sensor embedding fingerprint requires a sensor-granularity encoder")
    g = torch.Generator().manual_seed(20260724)
    data = torch.randn(_PROBE["n"], _PROBE["samples"], 6, generator=g).numpy()
    texts = [f"probe channel {i}" for i in range(6)]
    encoded = encode_dataset_detailed(
        enc, data, texts, device, _PROBE["rate"], gravity_state="present",
        channel_mask=[True] * 6, dataset=_PROBE["dataset"], stream=_PROBE["stream"],
        export_sensor_rows=True,
    )
    pieces = [
        encoded["sensor_Z"].float().flatten().cpu(),
        encoded["sensor_window"].float().cpu(),
        encoded["sensor_slot"].float().cpu(),
        encoded["sensor_time"].float().cpu(),
        encoded["sensor_duration"].float().cpu(),
        encoded["sensor_resolution"].float().cpu(),
    ]
    return torch.cat(pieces)


def encode_dataset_detailed(enc, data, texts, device, rate: float, gravity_state=None,
                            channel_mask=None, dataset=None, stream=None,
                            source_rate: float | None = None,
                            lengths=None,
                            _require_patches: bool = True,
                            requires_grad: bool = False,
                            neutral_text: bool = False,
                            export_sensor_rows: bool = False,
                            eval_patching: str = "checkpoint",
                            batch_size: int = 256) -> dict[str, torch.Tensor]:
    """Encode windows and retain both pooled and valid per-patch representations.

    Returns:
      pooled          (N, d)
      patch_Z         (M, d)
      patch_window    (M,) local input-window row
      patch_time      (M,) window-relative patch-center seconds
      patch_duration  (M,) represented physical seconds
      patch_resolution (M,) 0=single/short grid, 1=long grid

    ``dataset``/``stream`` are only needed for a FACTORED encoder (to build the role/sensor text);
    the default per_channel path uses the ``texts`` (per-channel descriptions) exactly as before.

    Only patches marked valid by ``patch_padding_mask`` are exported. This is important for the
    Phase-B memory bank: padding tokens must never become retrievable evidence. The historical
    :func:`encode_dataset` API below remains a pooled-only compatibility wrapper.
    """
    if eval_patching not in {"checkpoint", "fixed-1s", "multiresolution"}:
        raise ValueError(f"unknown eval_patching mode {eval_patching!r}")
    use_multiresolution = (
        getattr(enc, "multiresolution", enc.use_duration_embedding)
        if eval_patching == "checkpoint" else eval_patching == "multiresolution"
    )
    single_patch_seconds = (
        enc.eval_patch_seconds if eval_patching == "checkpoint" else PATCH_SECONDS
    )
    collate = (
        MultiResolutionCollate(fixed_patch_seconds=enc.eval_resolution_pair,
                               min_resolution_ratio=enc.min_resolution_ratio)
        if use_multiresolution else MultiScaleCollate(fixed_patch_seconds=single_patch_seconds)
    )
    if source_rate is None:
        source_rate = STREAM_SOURCE_RATE_HZ.get(f"{dataset}/{stream}", float(rate))
    source_rate = min(float(source_rate), float(rate))
    enc_texts = texts
    cmask = (torch.ones(6, dtype=torch.bool) if channel_mask is None
             else torch.as_tensor(channel_mask, dtype=torch.bool))
    if cmask.shape != (6,):
        raise ValueError(f"channel_mask must have shape (6,), got {tuple(cmask.shape)}")
    # Sensor granularity always needs the per-sensor text (role text is retired once xyz folds into
    # one token), so it takes the factored text path regardless of `text_conditioning`.
    factored = (getattr(enc, "text_conditioning", "per_channel") == "factored"
                or getattr(enc, "token_granularity", "channel") == "sensor")
    if factored:
        if dataset is None or stream is None:
            raise ValueError("a factored encoder needs dataset+stream to build the factored "
                             "(role/sensor) text conditioning; pass them to encode_dataset()")
        # One presence rule for both the sensor TEXT (which fixes N and sensor_id) and the sensor
        # BIAS (which must supply exactly N rows). Deriving them from separate expressions is how
        # they drift apart.
        modalities = modalities_present(cmask.tolist())
        role_texts, sensor_texts, sensor_id_list = stream_sensor_texts(
            dataset,
            stream,
            gravity_removed=(None if gravity_state is None else gravity_state == "removed"),
            has_accel="accel" in modalities,
            has_gyro="gyro" in modalities,
            neutral=neutral_text,
        )
        sensor_id_t = torch.tensor(sensor_id_list, dtype=torch.long)
        sensor_bias = stream_sensor_bias(dataset, stream, modalities)
    embs = []
    sensor_z_parts, sensor_window_parts, sensor_slot_parts = [], [], []
    sensor_time_parts, sensor_duration_parts, sensor_resolution_parts = [], [], []
    patch_z_parts = []
    patch_window_parts = []
    patch_time_parts = []
    patch_duration_parts = []
    patch_resolution_parts = []
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if lengths is not None and len(lengths) != len(data):
        raise ValueError("lengths must align 1:1 with data windows")
    grad_context = torch.enable_grad if requires_grad else torch.no_grad
    for start in range(0, len(data), batch_size):
        with grad_context():
            items = []
            stop = min(start + batch_size, len(data))
            for local, row in enumerate(range(start, stop)):
                valid = int(lengths[row]) if lengths is not None else data.shape[1]
                window = torch.tensor(np.asarray(data[row, :valid]), dtype=torch.float32)
                item = {"data": window, "rate": rate, "source_rate": source_rate,
                        "texts": enc_texts, "label_id": 0,
                        "channel_mask": cmask, "gravity_state": gravity_state, "source": "eval"}
                if factored:
                    item["role_texts"] = role_texts
                    item["sensor_texts"] = sensor_texts
                    item["sensor_id"] = sensor_id_t
                    item["sensor_bias"] = sensor_bias
                items.append(item)
            batch = collate(items)
            plen = batch["patch_len"]
            out = enc(
                batch["patches"].to(device), batch["rates"].to(device),
                plen.to(device),
                batch["role_texts"] if factored else batch["texts"],
                batch["positions"].to(device),
                patch_durations=(batch["patch_durations"].to(device)
                                 if "patch_durations" in batch else None),
                resolution_ids=(batch["resolution_ids"].to(device)
                                if "resolution_ids" in batch else None),
                channel_mask=batch["channel_mask"].to(device),
                patch_padding_mask=batch["patch_padding_mask"].to(device),
                sensor_texts=(batch["sensor_texts"] if factored else None),
                sensor_id=(batch["sensor_id"].to(device) if factored else None),
                source_rate_hz=batch["source_rates"].to(device),
                sensor_bias=(batch["sensor_bias"].to(device)
                             if getattr(enc, "token_granularity", "channel") == "sensor" else None),
            )
            embs.append(out["pooled"] if requires_grad else out["pooled"].cpu())
            if "per_patch" not in out:
                if _require_patches:
                    raise KeyError("encoder detailed export requires an out['per_patch'] tensor")
                continue

            valid = batch["patch_padding_mask"].bool()
            per_patch = out["per_patch"]
            valid_on_output = valid.to(per_patch.device)
            local_rows = (
                torch.arange(len(items), dtype=torch.long).unsqueeze(1)
                .expand_as(valid) + int(start)
            )
            if "patch_durations" in batch:
                durations = batch["patch_durations"].float()
            else:
                patch_len = batch["patch_len"].float()
                rates = batch["rates"].float().clamp_min(1e-8)
                durations = (
                    patch_len / rates
                    if patch_len.ndim == 1
                    else patch_len / rates.unsqueeze(1)
                )
            resolutions = (
                batch["resolution_ids"].long()
                if "resolution_ids" in batch
                else torch.zeros_like(batch["positions"], dtype=torch.long)
            )

            patch_z_parts.append(per_patch[valid_on_output])
            metadata_device = per_patch.device if requires_grad else torch.device("cpu")
            patch_window_parts.append(local_rows[valid].to(metadata_device))
            patch_time_parts.append(batch["positions"].float()[valid].to(metadata_device))
            patch_duration_parts.append(durations[valid].to(metadata_device))
            patch_resolution_parts.append(resolutions[valid].to(metadata_device))

            # PER-SENSOR rows (design-of-record bank layout). New checkpoints expose a
            # sensor-isolated temporal representation here; legacy checkpoints explicitly retain
            # their historical post-cross-sensor rows through their serialized architecture flag.
            if export_sensor_rows and "sensor_present" in out:
                tokens = out.get("retrieval_tokens")                  # (B,P,S,d)
                if tokens is None:
                    raise KeyError("sensor-row export requires out['retrieval_tokens']")
                present = out["sensor_present"]                       # (B,S)
                keep = valid_on_output.unsqueeze(2) & present.unsqueeze(1)   # (B,P,S)
                b_idx, p_idx, s_idx = keep.nonzero(as_tuple=True)
                sensor_z_parts.append(tokens[b_idx, p_idx, s_idx])
                sensor_window_parts.append((b_idx.cpu() + int(start)).to(metadata_device))
                sensor_slot_parts.append(s_idx.to(metadata_device))
                sensor_time_parts.append(
                    batch["positions"].float().to(keep.device)[b_idx, p_idx].to(metadata_device))
                sensor_duration_parts.append(
                    durations.to(keep.device)[b_idx, p_idx].to(metadata_device))
                sensor_resolution_parts.append(
                    resolutions.to(keep.device)[b_idx, p_idx].to(metadata_device))

    pooled = torch.cat(embs)
    d_model = int(pooled.shape[-1])
    return {
        "pooled": pooled,
        "patch_Z": (
            torch.cat(patch_z_parts)
            if patch_z_parts else torch.empty((0, d_model), dtype=pooled.dtype)
        ),
        "patch_window": (
            torch.cat(patch_window_parts)
            if patch_window_parts else torch.empty(0, dtype=torch.long)
        ),
        "patch_time": (
            torch.cat(patch_time_parts)
            if patch_time_parts else torch.empty(0, dtype=torch.float32)
        ),
        "patch_duration": (
            torch.cat(patch_duration_parts)
            if patch_duration_parts else torch.empty(0, dtype=torch.float32)
        ),
        "patch_resolution": (
            torch.cat(patch_resolution_parts)
            if patch_resolution_parts else torch.empty(0, dtype=torch.long)
        ),
        # Per-(patch, sensor) rows — empty unless export_sensor_rows AND the encoder is at sensor
        # granularity. `sensor_slot` indexes into the stream's sensor_texts order, which is how a
        # consumer recovers modality, descriptor and sensor_bias for the row.
        "sensor_Z": (
            torch.cat(sensor_z_parts)
            if sensor_z_parts else torch.empty((0, d_model), dtype=pooled.dtype)
        ),
        "sensor_window": (
            torch.cat(sensor_window_parts)
            if sensor_window_parts else torch.empty(0, dtype=torch.long)
        ),
        "sensor_slot": (
            torch.cat(sensor_slot_parts)
            if sensor_slot_parts else torch.empty(0, dtype=torch.long)
        ),
        "sensor_time": (
            torch.cat(sensor_time_parts)
            if sensor_time_parts else torch.empty(0, dtype=torch.float32)
        ),
        "sensor_duration": (
            torch.cat(sensor_duration_parts)
            if sensor_duration_parts else torch.empty(0, dtype=torch.float32)
        ),
        "sensor_resolution": (
            torch.cat(sensor_resolution_parts)
            if sensor_resolution_parts else torch.empty(0, dtype=torch.long)
        ),
    }


@torch.no_grad()
def encode_dataset(enc, data, texts, device, rate: float, gravity_state=None,
                   channel_mask=None, dataset=None, stream=None,
                   source_rate: float | None = None,
                   lengths=None,
                   neutral_text: bool = False,
                   eval_patching: str = "checkpoint") -> torch.Tensor:
    """Compatibility API: raw windows at native rate -> pooled embeddings only."""
    return encode_dataset_detailed(
        enc, data, texts, device, rate, gravity_state=gravity_state,
        channel_mask=channel_mask, dataset=dataset, stream=stream,
        source_rate=source_rate, lengths=lengths,
        neutral_text=neutral_text,
        eval_patching=eval_patching,
        _require_patches=False,
    )["pooled"]


def knn_balanced_acc(train_z, train_y, test_z, test_y, k=KNN_K) -> float:
    labels = sorted(set(train_y) & set(test_y))
    if not labels:
        return float("nan")
    label_to_id = {label: index for index, label in enumerate(labels)}
    keep_train = [index for index, label in enumerate(train_y) if label in label_to_id]
    keep_test = [index for index, label in enumerate(test_y) if label in label_to_id]
    train = train_z[keep_train].float()
    test = test_z[keep_test].float()
    train_ids = torch.tensor([label_to_id[train_y[index]] for index in keep_train])
    test_ids = torch.tensor([label_to_id[test_y[index]] for index in keep_test])
    k = min(int(k), len(train))
    if k < 1:
        return float("nan")
    predictions = []
    train_norm = train.square().sum(dim=1).unsqueeze(0)
    for start in range(0, len(test), 512):
        query = test[start:start + 512]
        distances = query.square().sum(dim=1, keepdim=True) + train_norm - 2 * query @ train.T
        neighbors = distances.topk(k, dim=1, largest=False).indices
        votes = torch.nn.functional.one_hot(
            train_ids[neighbors], num_classes=len(labels),
        ).sum(dim=1)
        # argmax gives a deterministic sorted-label tie break, unlike max(set(...), key=count).
        predictions.append(votes.argmax(dim=1))
    predicted = torch.cat(predictions)
    per_class = [
        float((predicted[test_ids == label_id] == label_id).float().mean())
        for label_id in range(len(labels)) if bool((test_ids == label_id).any())
    ]
    return float(np.mean(per_class))


def assert_selection_roster_is_untrained(train_datasets) -> None:
    """Fail loudly if a checkpoint-selection source is also being trained on.

    Selecting on a trained source silently converts the metric from held-out transfer into a
    training probe -- the exact defect that let the 2026-08-17 run report its best checkpoint while
    external transfer degraded. A comment cannot enforce this; a start-of-run assertion can.
    """
    overlap = sorted(set(PHASE_A_SELECTION_DATASETS) & set(train_datasets or ()))
    if overlap:
        raise ValueError(
            f"Phase-A selection datasets {overlap} are in the training corpus. Selection must be "
            f"held out; either drop them from --datasets or edit PHASE_A_SELECTION_STREAMS."
        )


def _selection_subsample(labels: np.ndarray, subjects: np.ndarray, cap: int) -> np.ndarray:
    """Deterministic (label, subject)-stratified row subsample, or all rows when already small.

    Round-robin over strata rather than a proportional draw, so a rare posture class keeps enough
    rows to be scored at all -- the canary below is worthless if `lying` gets three windows.
    """
    total = len(labels)
    if total <= cap:
        return np.arange(total)
    rng = np.random.default_rng(SEED)
    strata: dict[tuple, list[int]] = {}
    for index, key in enumerate(zip(labels.tolist(), subjects.tolist())):
        strata.setdefault(key, []).append(index)
    for rows in strata.values():
        rng.shuffle(rows)
    order = sorted(strata)
    chosen: list[int] = []
    depth = 0
    while len(chosen) < cap:
        added = False
        for key in order:
            rows = strata[key]
            if depth < len(rows):
                chosen.append(rows[depth])
                added = True
                if len(chosen) >= cap:
                    break
        if not added:
            break
        depth += 1
    return np.sort(np.asarray(chosen, dtype=np.int64))


@torch.no_grad()
def development_transfer_score(
    enc: SetTokenizerEncoder,
    device: torch.device,
    datasets: tuple[str, ...] = PHASE_A_SELECTION_DATASETS,
    *,
    patching: str = "checkpoint",
    max_windows_per_stream: int = SELECTION_MAX_WINDOWS_PER_STREAM,
) -> tuple[float, dict[str, float]]:
    """Subject-disjoint kNN on development-only acquisition sources.

    Phase A uses this to select checkpoints. Datasets are averaged equally, so a source with more
    windows cannot dominate. Scored on the SENSOR ROWS Phase B actually stores, not on a pooled
    embedding no downstream stage ever sees.

    The returned dict carries per-dataset scores plus `posture/<dataset>` canaries for sources that
    contain static postures. Those keys are diagnostics and are excluded from the scalar mean.
    """
    if not datasets:
        raise ValueError("development transfer selection requires at least one dataset")
    wanted = set(datasets)
    roster = {dataset for dataset, _, _ in PHASE_A_SELECTION_STREAMS}
    unknown = wanted - roster
    if unknown:
        raise ValueError(
            f"selection datasets are not in PHASE_A_SELECTION_STREAMS: {sorted(unknown)}"
        )
    refs = {(ref.dataset, ref.stream): ref for ref in discover_grids("native")}
    was_training = enc.training
    enc.eval()
    scores: dict[str, list[float]] = {dataset: [] for dataset in datasets}
    posture: dict[str, list[float]] = {}
    scored_datasets: set[str] = set()
    try:
        for dataset, stream, scored in PHASE_A_SELECTION_STREAMS:
            if dataset not in wanted:
                continue
            if scored:
                scored_datasets.add(dataset)
            ref = refs.get((dataset, stream))
            if ref is None:
                raise FileNotFoundError(f"missing development grid {dataset}/{stream}")
            labels_all = np.asarray(ref.labels)
            subjects_all = np.asarray(ref.subjects)
            take = _selection_subsample(labels_all, subjects_all, max_windows_per_stream)
            labels = labels_all[take]
            subjects = subjects_all[take]
            encoded = encode_dataset_detailed(
                enc, ref.load_data()[take], stream_channel_descriptions(dataset, stream), device,
                ref.rate_hz, _stream_gravity_state(dataset, stream), channel_mask=ref.mask,
                dataset=dataset, stream=stream, lengths=ref.load_lengths()[take],
                eval_patching=patching, export_sensor_rows=True,
            )
            if encoded["sensor_Z"].numel():
                rows = encoded["sensor_Z"].float()
                owners = encoded["sensor_window"].long().to(rows.device)
                z = rows.new_zeros((len(labels), rows.shape[1]))
                count = rows.new_zeros((len(labels), 1))
                z.index_add_(0, owners, rows)
                count.index_add_(0, owners, rows.new_ones((len(rows), 1)))
                z = (z / count.clamp_min(1.0)).cpu()
            else:
                z = encoded["pooled"]
            hold = subject_holdout(subjects, dataset)
            train_rows = np.asarray([subject not in hold for subject in subjects])
            test_rows = ~train_rows
            if not train_rows.any() or not test_rows.any():
                raise ValueError(f"{dataset}/{stream} cannot form a subject-disjoint split")
            scores[dataset].append(knn_balanced_acc(
                z[train_rows], labels[train_rows].tolist(),
                z[test_rows], labels[test_rows].tolist(),
            ))
            # Posture canary: rescore restricted to the gravity-defined static classes.
            static = np.isin(labels, POSTURE_LABELS)
            if static.sum() and len(set(labels[static].tolist())) >= 2:
                train_static = train_rows & static
                test_static = test_rows & static
                if train_static.any() and test_static.any():
                    value = knn_balanced_acc(
                        z[train_static], labels[train_static].tolist(),
                        z[test_static], labels[test_static].tolist(),
                    )
                    if value == value:                       # skip NaN (no shared static labels)
                        posture.setdefault(dataset, []).append(value)
    finally:
        enc.train(was_training)
    missing = [dataset for dataset, rows in scores.items() if not rows]
    if missing:
        raise FileNotFoundError(f"no development streams found for {missing}")
    per_dataset = {dataset: float(np.mean(rows)) for dataset, rows in scores.items()}
    scalar_terms = [value for dataset, value in per_dataset.items() if dataset in scored_datasets]
    if not scalar_terms:
        raise ValueError(
            "no SCORED selection dataset was evaluated; the scalar would be undefined. Requested "
            f"{sorted(wanted)}, of which none is marked scored in PHASE_A_SELECTION_STREAMS."
        )
    mean = float(np.mean(scalar_terms))
    for dataset, rows in posture.items():
        per_dataset[f"posture/{dataset}"] = float(np.mean(rows))
    return mean, per_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--parity", action="store_true",
                    help="Also score the PARITY arm (neutral text: modality+axis only, no "
                         "placement/device/gravity) and report the gap. The gap IS the measured "
                         "value of acquisition-config conditioning — BASELINE_FAIRNESS_POLICY.md §5.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write the JSON (default: <checkpoint dir>/transfer_eval.json)")
    ap.add_argument(
        "--patching", choices=("checkpoint", "fixed-1s", "multiresolution"),
        default="checkpoint",
        help="evaluation patch grid; use one explicit mode for controlled checkpoint comparisons",
    )
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    print(f"loaded {args.checkpoint.name}: step {ckpt['step']}, "
          f"internal val_ba {ckpt['val_ba']:.3f}, git {ckpt['git']}", flush=True)

    refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
    arms = {"full": False, "parity": True} if args.parity else {"full": False}
    arm_results: dict[str, dict] = {}
    arm_stream_results: dict[str, dict] = {}
    arm_means: dict[str, float] = {}

    for arm, neutral in arms.items():
        stream_results = {}
        print(f"\n--- arm: {arm} ({'neutral text' if neutral else 'full conditioning'}) ---",
              flush=True)
        for dataset, stream in EVAL_STREAMS:
            ref = refs.get((dataset, stream))
            if ref is None:
                continue
            data = ref.load_data()
            labels = np.asarray(ref.labels)
            subjects = np.asarray(ref.subjects)
            texts = stream_channel_descriptions(dataset, stream, neutral=neutral)
            z = encode_dataset(enc, data, texts, device, ref.rate_hz,
                               _stream_gravity_state(dataset, stream),
                               channel_mask=ref.mask, dataset=dataset, stream=stream,
                               lengths=ref.load_lengths(),
                               neutral_text=neutral, eval_patching=args.patching)

            # subject-disjoint 50/50 split
            hold = subject_holdout(subjects, dataset)
            tr = [i for i in range(len(data)) if subjects[i] not in hold]
            te = [i for i in range(len(data)) if subjects[i] in hold]
            ba = knn_balanced_acc(z[tr], labels[tr].tolist(), z[te], labels[te].tolist())
            n_lab = len(set(labels.tolist()))
            cell = f"{dataset}/{stream}"
            stream_results[cell] = {
                "dataset": dataset,
                "stream": stream,
                "knn_ba": round(ba, 4),
                "windows": len(data),
                "labels": n_lab,
                "chance": round(1 / n_lab, 3),
            }
            print(f"  {cell:42s} kNN-BA={ba:.3f}  ({n_lab} labels, chance {1/n_lab:.3f}, "
                  f"{len(data)} windows)", flush=True)
        results = {}
        for dataset in PHASE_A_TRANSFER_DATASETS:
            cells = [row for row in stream_results.values() if row["dataset"] == dataset]
            if not cells:
                continue
            results[dataset] = {
                "knn_ba": round(float(np.mean([row["knn_ba"] for row in cells])), 4),
                "streams": len(cells),
                "windows": sum(row["windows"] for row in cells),
                "per_stream": {row["stream"]: row["knn_ba"] for row in cells},
            }
        arm_results[arm] = results
        arm_stream_results[arm] = stream_results
        arm_means[arm] = float(np.mean([r["knn_ba"] for r in results.values()]))

    results, mean = arm_results["full"], arm_means["full"]
    print(f"\nHELD-OUT-CONFIG TRANSFER: mean kNN-BA = {mean:.3f} across {len(results)} datasets")
    payload = {"checkpoint": str(args.checkpoint), "step": ckpt["step"],
               "internal_val_ba": ckpt["val_ba"], "per_dataset": results,
               "per_stream": arm_stream_results["full"],
               "mean_knn_ba": round(mean, 4), "patching": args.patching}
    if args.parity:
        gap = mean - arm_means["parity"]
        payload["parity"] = {
            "per_dataset": arm_results["parity"],
            "per_stream": arm_stream_results["parity"],
            "mean_knn_ba": round(arm_means["parity"], 4),
            "conditioning_gain": round(gap, 4),
            "per_dataset_gain": {d: round(results[d]["knn_ba"]
                                          - arm_results["parity"][d]["knn_ba"], 4)
                                 for d in results},
        }
        print(f"PARITY (neutral text):        mean kNN-BA = {arm_means['parity']:.3f}")
        print(f"CONDITIONING GAIN (full - parity) = {gap:+.4f}")
        print("  Interpretation: this gap is the measured value of acquisition-config "
              "conditioning.\n  A gain of ~0 means the placement/device/gravity text is inert.")
    out = args.out or (args.checkpoint.parent / "transfer_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
