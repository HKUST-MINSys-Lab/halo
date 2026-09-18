"""Pipeline A Phase-1 data pipeline (pretraining corpus + sampler + collate).

Design decisions carried in from the gates:
  * Corpus = the dedicated label-free recipe's **native** grids by default (labelled evaluation sets
    are never touched); labelled expanded/matched recipes remain historical controls. Native sampling
    RATE (no 60 Hz resample) + canonical labels + 6-ch pad+mask. The filterbank tokenizer is
    rate-invariant, so HALO trains on the corpus's REAL native rates (20/50/100 Hz) instead of a
    homogenized 60 Hz base. Corpus-matched layout-locked baseline adapters now begin from the same
    native windows and perform their required anti-aliased resampling locally. Source-balanced
    sampling (no per-stream cap) spreads each activity across configs.
  * Subject-disjoint train/val split per dataset.
  * Label-free hierarchical temperature sampling is the sole Phase-A sampler.
  * The reference augmentation is one independent SO(3) rotation per positive view. More aggressive
    signal, sensor, rate, crop, and text transforms remain explicit ablations.
  * Fixed one-second patches by default. Native sampling RATE still varies per sample and true patch
    lengths vary for final partial contexts; the filterbank consumes both explicitly.
  * Axis role plus device/placement text come from deployment policy; exact modality, gravity and
    rate facts use acquisition-conditioning-v2 structured tensors. Text augmentation is disabled
    in the reference recipe; absent channels carry a channel_mask and never fake confidence.
"""

from __future__ import annotations

import hashlib
import random as stdlib_random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data.scripts.augmentations import AugmentationConfig, IMUAugmenter, IMUSample
from data.scripts.curate.deployment_policy import (
    CORPUS_MATCHED_TRAIN_DATASETS,
    EXPANDED_PHASE_A_TRAIN_DATASETS,
    LABEL_FREE_PRETRAIN_DATASETS,
)
from data.scripts.eda.grid_io import GridRef, discover_grids
from data.scripts.labels.canonical_labels import canonicalize
from model.tokenizer.sensor_tokens import (
    CONDITIONING_SCHEMA_V2,
    GRAVITY_NOT_APPLICABLE,
    GRAVITY_PRESENT,
    GRAVITY_REMOVED,
    GRAVITY_UNKNOWN,
    MODALITY_ACCELEROMETER,
    MODALITY_GYROSCOPE,
    LEGACY_CONDITIONING_SCHEMA,
)

# ----------------------------------------------------------------------------------------------
# Corpus configuration
# ----------------------------------------------------------------------------------------------
# hapt DROPPED: the sweep confirmed it is the UCI-HAR re-release — same 30 subjects /
# recordings (per-window NCC 0.98 vs uci_har), so keeping both leaks near-duplicate val
# windows into train across the pair. uci_har is the canonical windowed release; keep it.
# This module-level roster is the historical labelled recipe used by explicit controls. The CLI's
# design-of-record default is the disjoint label-free roster resolved in pretrain.py.
TRAIN_DATASETS = EXPANDED_PHASE_A_TRAIN_DATASETS
# Historical optional sources for the labelled recipe. Label-free scale sources are resolved from
# deployment_policy so additions cannot silently fall into the validation partition.
OPTIONAL_PHASE_A_DATASETS = ("extrasensory", "nhanes", "hmog", "kneepad")
PHASE_A_ONLY_DATASETS = frozenset(LABEL_FREE_PRETRAIN_DATASETS)
UNLABELED_LABEL = "__unlabeled__"
WINDOW_SECONDS = 6.0
PATCH_SECONDS = 1.0
VAL_SUBJECT_FRACTION = 0.10      # subject-disjoint val within the train datasets
# Explicit multi-resolution ablation settings. The reference path above uses PATCH_SECONDS only.
PATCH_SECONDS_CHOICES = (0.5, 0.75, 1.0, 1.5, 2.0)
SHORT_PATCH_SECONDS_CHOICES = (0.4, 0.5, 0.6, 0.7, 0.8)
LONG_PATCH_SECONDS_CHOICES = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
MIN_RESOLUTION_RATIO = 1.75
MIN_TAIL_FRACTION = 0.25          # F7: drop a resolution's tail patch if it covers < this fraction of
                                 # a full patch (and it isn't the only patch) — avoids degenerate
                                 # 1-sample tokens whose duration is clamped to the embedding floor.
VAL_RESOLUTION_PAIR = (0.5, 2.0)
# Hard ceiling on the per-batch TOKEN count (batch x patches). Peak VRAM tracks tokens, not
# windows, and patch_seconds is drawn PER BATCH — so without this, memory is a random variable:
# measured P swings 12->22 at fixed batch. The current sensor-granularity encoder was profiled at
# 7.46 GiB for batch 384 and 10.06 GiB for batch 512 with every resolution pair enabled on a 24 GiB
# RTX 4090. The fixed 0.5/1.0/2.0 s JEPA grid remains under this 16,384-token ceiling at batch 512;
# the dense-hop continuous frontend has its own output grid and must be profiled separately.
# Set 0 to disable.
MAX_BATCH_TOKENS = 16_384
MULTI_DEVICE_TRAIN_DATASETS = frozenset({"realdisp", "xrf_v2", "dsads", "forth_trace"})
# Covers the largest declared future-JEPA patch in the label-free corpus: 240 Hz x 2.0 s = 480
# samples. Keeping one power-of-two capacity across collate, encoder, frozen physical targets and
# evaluation avoids frontend-specific truncation or a run that succeeds only on low-rate sources.
DFT_SIZE = 512
# Four-second filterbank patches must not make FFT cost proportional to a 100--240 Hz acquisition
# clock.  Forty Hz clears the 15 Hz physical analysis band with margin and bounds 4/8 s patches at
# 160/320 samples, respectively.  The conversion is enabled only when a requested resolution is at
# least four seconds, preserving the established <=2 s collate bit-for-bit.
FILTERBANK_LONG_PATCH_ANALYSIS_HZ = 40.0
# Streams whose SOURCE (acquisition) rate differs from the rate the grid is stored at, because a
# converter resampled them onto the dataset-wide grid rate. Upsampling cannot create information, so
# the filterbank must take its Nyquist/observability bound from the ACQUISITION rate while the DFT
# bin->frequency mapping keeps using the true array rate. Without this, xrf_v2's AirPods stream
# (captured at 25 Hz, stored at 50 Hz) was advertised as having all 32 bands observable when only
# ~27 carry real signal — the rest is interpolation artifact. Authority: each dataset's convert.py.
STREAM_SOURCE_RATE_HZ = {
    "xrf_v2/airpods_ear": 25.0,      # AirPods Pro ear IMU @25 Hz, upsampled 25->50 in convert.py
    "extrasensory/watch_wrist": 25.0, # Pebble acquisition clock; converter stores at 50 Hz
    # Phone clocks vary (~30-200 Hz). Thirty Hz is the conservative observed
    # floor, so bands above 15 Hz are never claimed as physically observable.
    "extrasensory/phone_pocket": 30.0,
    "extrasensory/phone_hand": 30.0,
    # MM-Fit's packet-aware converter stores every device on a shared 100 Hz grid. The earbud is
    # acquired at ~85 Hz, so interpolation cannot make bands above its native Nyquist observable.
    "mmfit/left_ear": 85.0,
}
CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
SEED = 20260718

PLACEMENT_WORDS = {
    "waist": "the waist", "wrist": "the wrist", "pocket": "a trouser pocket",
    "hip": "the hip", "belt": "the belt", "thigh": "the thigh",
    "back": "the lower back", "ankle": "the ankle", "chest": "the chest",
}


_DEVICE_WORDS = {"phone": "phone", "watch": "watch", "watch_proxy": "phone",
                 "device": "wearable device"}


def stream_channel_descriptions(dataset: str, stream: str, *, neutral: bool = False) -> list[str]:
    """Per-channel base text for a deployment stream — HALO's configuration-conditioning input.

    Uses the StreamSpec's curated `placement` + `device_profile` from the deployment policy (e.g.
    "smart glasses on the head" / "the left wrist" / "an earbud in the ear"), which distinguishes
    left vs right, head vs ear vs pocket, etc. Falls back to stream-id tokens only if no spec exists.
    """
    try:
        from data.scripts.curate.deployment_policy import get_stream_spec
        spec = get_stream_spec(dataset, stream)
        place = spec.placement if spec.placement.startswith(("the ", "a ", "an ", "smart")) \
            else f"the {spec.placement}"
        device = _DEVICE_WORDS.get(spec.device_profile, spec.device_profile.replace("_", " "))
        gravity_removed = (spec.gravity_state == "removed")
    except (KeyError, ValueError, ImportError):
        tokens = stream.lower().split("_")
        device = "phone" if "phone" in tokens else ("watch" if "watch" in tokens else "device")
        place = next((PLACEMENT_WORDS[w] for w in tokens if w in PLACEMENT_WORDS), "the body")
        gravity_removed = False
    where = f"{place} ({device})"
    # Gravity state is a real acquisition-config axis: accelerometer from gravity-removed streams
    # (kuhar, xrf_v2/airpods_ear) has |DC|~0 vs ~1 g for gravity-present streams. Only the
    # accelerometer carries it (the gyroscope is unaffected). The clause mirrors the sibling
    # deployment_policy.channel_description() and lets the gravity-removal augmentation skip
    # streams that already contain user acceleration rather than trusting a magnitude heuristic.
    grav = "; gravity removed" if gravity_removed else "; includes gravity"
    if neutral:
        # PARITY ARM (BASELINE_FAIRNESS_POLICY.md §5): modality + axis only. Placement, device and
        # gravity state — every acquisition-config fact — are stripped, leaving the identity text a
        # fixed-layout baseline also has. The gap against the full arm IS the value of conditioning.
        return ([f"accelerometer {a}-axis" for a in "xyz"]
                + [f"gyroscope {a}-axis" for a in "xyz"])
    return ([f"accelerometer {a}-axis worn at {where}{grav}" for a in "xyz"]
            + [f"gyroscope {a}-axis worn at {where}" for a in "xyz"])


# Fixed intra-sensor channel ROLE text (AXIS ONLY — never modality/placement/device). One entry per
# CHANNELS slot; constant across the whole corpus. Modality (accel vs gyro) is NOT a role fact — it
# moved to the per-sensor text below, because accel and gyro are now modelled as two distinct
# modality-level SENSORS (docs/design/TEXT_CONDITIONING.md). Role is the purely positional x/y/z axis.
# NOTE: axis is encoded as TEXT ("x"/"y"/"z") so it flows through the same frozen-LM pooler as every
# other text source. A tiny LEARNED 3-way role embedding would be cleaner (only three axes; SBERT-
# embedding single letters is wasteful) but that is a model-side change to FactoredChannelTextFusion —
# kept as text here to avoid over-engineering the data path.
_CHANNEL_ROLE_TEXT = {
    "acc_x": "x", "acc_y": "y", "acc_z": "z",
    "gyro_x": "x", "gyro_y": "y", "gyro_z": "z",
}


_SENSOR_BIAS_PATH = "data/scripts/curate/sensor_bias.json"
_SENSOR_BIAS_CACHE: dict | None = None
SENSOR_BIAS_FIELDS = (
    "gravity_magnitude", "gravity_presence", "noise_floor", "quantization_step",
    "clip_fraction", "rate_fidelity", "rest_bias",
)
SENSOR_BIAS_DIM = 2 * len(SENSOR_BIAS_FIELDS)  # z-scored values + explicit support bits


def load_sensor_bias() -> dict:
    """Frozen per-sensor ``sensor_bias`` artifact, keyed ``(dataset, stream, modality)``.

    Built offline by ``data.scripts.curate.sensor_bias`` over the Phase-A *training subjects* only,
    using activity-invariant channel physics. Loaded once and cached: it is a fixed measurement, not
    something that varies per batch, and the z-scoring statistics inside it are frozen so descriptors
    stay comparable across streams (Phase-B compares them directly).
    """
    global _SENSOR_BIAS_CACHE
    if _SENSOR_BIAS_CACHE is None:
        import json
        from pathlib import Path
        path = Path(_SENSOR_BIAS_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing — build it with "
                "`python -m data.scripts.curate.sensor_bias --build`"
            )
        payload = json.loads(path.read_text())
        if tuple(payload.get("fields", ())) != SENSOR_BIAS_FIELDS:
            raise ValueError(
                f"{path} has stale sensor-bias fields {payload.get('fields')}; rebuild it with "
                "`python -m data.scripts.curate.sensor_bias --build`"
            )
        _SENSOR_BIAS_CACHE = {
            "fields": payload["fields"],
            "provenance": payload.get("provenance", {}),
            "by_sensor": {
                (r["dataset"], r["stream"], r["modality"]):
                    [*r["z"], *(float(value) for value in r["supported"])]
                for r in payload["sensors"]
            },
        }
    return _SENSOR_BIAS_CACHE


def validate_sensor_bias_training_corpus(
    refs: Sequence[GridRef], datasets: Sequence[str], data_seed: int,
) -> None:
    """Fail closed when the frozen bias artifact does not match this Phase-A training split.

    Evaluation may honestly use an unknown (all-zero, unsupported) descriptor for a held-out sensor.
    Training may not: silently missing a requested stream or normalising with different datasets or
    validation subjects changes the conditioning function and invalidates the run.
    """
    artifact = load_sensor_bias()
    provenance = artifact.get("provenance", {})
    expected_datasets = tuple(datasets)
    actual_datasets = tuple(provenance.get("datasets", ()))
    if actual_datasets != expected_datasets:
        raise ValueError(
            "sensor_bias.json was built for datasets "
            f"{actual_datasets}, but this run requests {expected_datasets}; rebuild it with "
            "`python -m data.scripts.curate.sensor_bias --build --datasets ...`"
        )
    if int(provenance.get("data_seed", -1)) != int(data_seed):
        raise ValueError(
            f"sensor_bias.json uses data_seed={provenance.get('data_seed')}, but the run uses "
            f"{data_seed}; rebuild the artifact for the same subject split"
        )
    if provenance.get("subjects") != "train_only":
        raise ValueError("sensor_bias.json must be built from Phase-A training subjects only")

    val_subjects = validation_subjects_for_refs(refs, seed=data_seed)
    subject_payload = "\n".join(f"{dataset}\t{subject}" for dataset, subject in sorted(val_subjects))
    split_hash = hashlib.sha256(subject_payload.encode("utf-8")).hexdigest()
    if provenance.get("validation_subjects_sha256") != split_hash:
        raise ValueError(
            "sensor_bias.json does not match the current validation-subject split; rebuild it"
        )

    expected_keys = set()
    for ref in refs:
        mask = np.asarray(ref.mask, dtype=bool)
        if mask[:3].any():
            expected_keys.add((ref.dataset, ref.stream, "accel"))
        if mask[3:6].any():
            expected_keys.add((ref.dataset, ref.stream, "gyro"))
    missing = sorted(expected_keys - set(artifact["by_sensor"]))
    if missing:
        raise ValueError(f"sensor_bias.json is missing training sensors: {missing}")


def modalities_present(channel_mask: Sequence[bool]) -> list[str]:
    """Modalities carried by a 6-slot channel mask, in ``stream_sensor_texts`` order.

    ONE rule, shared by every caller that has to line sensors up. ``stream_sensor_texts`` fixes the
    sensor count N and the ``sensor_id`` map from these same flags, and ``stream_sensor_bias`` must
    return exactly N rows — so a caller that decides presence with ``.all()`` while the text path
    uses ``.any()`` turns a partial triad (some but not all axes live) into a sensor that has a
    description and no bias row, which the encoder's shape check rejects.

    A modality counts as present when ANY of its axes is live: the axis-validity indicator inside
    ``SensorFold`` is what handles a dead axis, so a two-axis accelerometer is still an
    accelerometer rather than an absent sensor.
    """
    mask = [bool(value) for value in channel_mask]
    if len(mask) != len(CHANNELS):
        raise ValueError(f"channel_mask must have {len(CHANNELS)} slots, got {len(mask)}")
    return [name for name, axes in (("accel", mask[:3]), ("gyro", mask[3:6])) if any(axes)]


def stream_sensor_bias(dataset: str, stream: str, modalities: Sequence[str]) -> torch.Tensor:
    """(N_sensors, F) bias rows for one stream, ordered to match ``stream_sensor_texts``.

    A sensor with no entry gets a ZERO row — z-scored space puts zero at the corpus mean, i.e. "no
    evidence either way", which is the honest default for a sensor we have not measured. It is NOT
    the same claim as "average", and Phase B distinguishes the two via the artifact's `supported`
    flags; here the zero simply keeps the tensor dense.
    """
    artifact = load_sensor_bias()
    width = 2 * len(artifact["fields"])
    rows = [artifact["by_sensor"].get((dataset, stream, m), [0.0] * width) for m in modalities]
    return torch.tensor(rows, dtype=torch.float32)


def _pad_sensor_rows(batch: list[dict], key: str, width: int | None = None) -> torch.Tensor | None:
    """Stack a ragged per-sensor field to (B, N_max, ...), zero-padded.

    Sensor count varies across a batch (accel-only streams carry one, acc+gyro carry two), so the
    per-sensor conditioning tensors are ragged exactly like ``sensor_texts``. Padding rows are zero
    and are masked downstream by ``sensor_present``, which is derived from the text-id table — a
    padded slot never has a real description, so it can never be mistaken for a measured sensor.
    """
    if key not in batch[0] or batch[0][key] is None:
        return None
    rows = [item[key] for item in batch]
    n_max = max(r.shape[0] for r in rows)
    if width is None:
        width = rows[0].shape[1] if rows[0].dim() > 1 else None
    shape = (len(rows), n_max) + ((width,) if width is not None else ())
    out = torch.zeros(shape, dtype=rows[0].dtype)
    for i, r in enumerate(rows):
        out[i, :r.shape[0]] = r
    return out


def _attach_conditioning_metadata(out: dict, batch: list[dict]) -> None:
    """Attach conditioning-v2 fields when present, rejecting partially populated batches.

    The collators also serve channel-granularity baselines and small frontend unit tests that do
    not consume sensor metadata. Keeping the fields optional here preserves that generic surface;
    the v2 HALO encoder itself requires all three and fails before use if they are absent.
    """
    keys = ("sensor_modality", "sensor_gravity", "sensor_rates_hz")
    present = [[item.get(key) is not None for item in batch] for key in keys]
    if any(any(rows) and not all(rows) for rows in present):
        raise ValueError("conditioning-v2 metadata must be populated for every row or no rows")
    if any(all(rows) for rows in present) and not all(all(rows) for rows in present):
        raise ValueError("conditioning-v2 requires modality, gravity and rates together")
    if all(all(rows) for rows in present):
        for key in keys:
            value = _pad_sensor_rows(batch, key)
            if value is None:
                raise RuntimeError(f"failed to collate conditioning-v2 field {key!r}")
            out[key] = value


def _merge_conditioning_metadata(out: dict, items: Sequence[dict]) -> None:
    """Concatenate per-device conditioning rows into one composite recording."""
    keys = ("sensor_modality", "sensor_gravity", "sensor_rates_hz")
    present = [[item.get(key) is not None for item in items] for key in keys]
    if any(any(rows) and not all(rows) for rows in present):
        raise ValueError("conditioning-v2 metadata must be populated for every device or no devices")
    if any(all(rows) for rows in present) and not all(all(rows) for rows in present):
        raise ValueError("conditioning-v2 requires modality, gravity and rates together")
    if all(all(rows) for rows in present):
        for key in keys:
            out[key] = torch.cat([torch.as_tensor(item[key]) for item in items], dim=0)


def _pad_channel_rows(batch: list[dict], key: str, *, value=0) -> torch.Tensor | None:
    """Pad a ragged per-channel tensor; composite recordings may carry more than six slots."""
    if key not in batch[0] or batch[0][key] is None:
        return None
    rows = [torch.as_tensor(item[key]) for item in batch]
    width = max(len(row) for row in rows)
    out = torch.full((len(rows), width), value, dtype=rows[0].dtype)
    for index, row in enumerate(rows):
        out[index, :len(row)] = row
    return out


def _pad_channel_text(batch: list[dict], key: str) -> list[list[str | None]]:
    width = max(len(item.get(key) or ()) for item in batch)
    return [list(item.get(key) or ()) + [None] * (width - len(item.get(key) or ())) for item in batch]


def stream_sensor_texts(
    dataset: str,
    stream: str,
    *,
    gravity_removed: bool | None = None,
    has_accel: bool = True,
    has_gyro: bool = True,
    neutral: bool = False,
    conditioning_schema: str = CONDITIONING_SCHEMA_V2,
) -> tuple[list[str], list[str], list[int]]:
    """Runtime device/placement text for each present modality-level sensor.

    Accel and gyro are modelled as two distinct modality-level SENSORS, so a stream factors as:
      * ``role_texts``  — one string per CHANNELS slot, AXIS ONLY ("x"/"y"/"z"); corpus-constant.
      * ``sensor_texts``— one device/placement sentence per PRESENT modality. Co-located accel and
                          gyro intentionally receive the same sentence; modality, gravity and rate
                          are exact structured fields under acquisition-conditioning-v2.
      * ``sensor_id``   — length-6 map: accel channels -> the accel sensor's index, gyro channels ->
                          the gyro sensor's index. Absent-modality channels are ``channel_mask``-masked
                          and pool to nothing, so their id only needs to be a valid index.

    Dataset, subject, activity, modality, gravity and rate must never enter this text.
    """
    if conditioning_schema not in {CONDITIONING_SCHEMA_V2, LEGACY_CONDITIONING_SCHEMA}:
        raise ValueError(f"unsupported conditioning schema {conditioning_schema!r}")
    role_texts = [_CHANNEL_ROLE_TEXT[c] for c in CHANNELS]
    from data.scripts.curate.deployment_policy import get_stream_spec
    spec = get_stream_spec(dataset, stream)
    if spec.device_profile not in _DEVICE_WORDS:
        raise ValueError(
            f"{dataset}/{stream} has non-runtime device profile {spec.device_profile!r}; "
            "register a supported runtime device role before using it with HALO"
        )
    place = spec.placement if spec.placement.startswith(("the ", "a ", "an ", "smart")) \
        else f"the {spec.placement}"
    device = _DEVICE_WORDS[spec.device_profile]
    if neutral:
        device_placement = "a device at an unspecified placement"
    else:
        device_placement = f"a {device} located at {place}"
    if conditioning_schema == LEGACY_CONDITIONING_SCHEMA:
        removed = (spec.gravity_state == "removed") if gravity_removed is None \
            else bool(gravity_removed)
        gravity = "gravity removed" if removed else "includes gravity"
        if neutral:
            accel_text, gyro_text = "an accelerometer", "a gyroscope"
        else:
            accel_partner = (
                "recorded alongside a gyroscope" if has_gyro else "recorded without a gyroscope"
            )
            gyro_partner = (
                "recorded alongside an accelerometer" if has_accel
                else "recorded without an accelerometer"
            )
            accel_text = f"a {device} accelerometer on {place}; {gravity}; {accel_partner}"
            gyro_text = f"a {device} gyroscope on {place}; {gyro_partner}"
    else:
        accel_text = gyro_text = device_placement
    # Emit only the modalities actually present. Absent-modality channels are channel_mask-masked, so
    # their sensor_id just needs to stay a valid index into sensor_texts.
    sensor_texts: list[str] = []
    accel_id = gyro_id = 0
    if has_accel:
        accel_id = len(sensor_texts)
        sensor_texts.append(accel_text)
    if has_gyro:
        gyro_id = len(sensor_texts)
        sensor_texts.append(gyro_text)
    if not sensor_texts:                     # defensive: neither modality flagged present
        raise ValueError(f"{dataset}/{stream} has neither accelerometer nor gyroscope")
    sensor_id = [accel_id if c.startswith("acc") else gyro_id for c in CHANNELS]
    return role_texts, sensor_texts, sensor_id


def structured_sensor_metadata(
    *, has_accel: bool, has_gyro: bool, gravity_state: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return modality and gravity IDs in the same order as ``stream_sensor_texts``."""
    modalities: list[int] = []
    gravity: list[int] = []
    if has_accel:
        state = {
            "present": GRAVITY_PRESENT,
            "removed": GRAVITY_REMOVED,
            "unknown": GRAVITY_UNKNOWN,
            None: GRAVITY_UNKNOWN,
        }.get(gravity_state)
        if state is None:
            raise ValueError(f"invalid accelerometer gravity state {gravity_state!r}")
        modalities.append(MODALITY_ACCELEROMETER)
        gravity.append(state)
    if has_gyro:
        modalities.append(MODALITY_GYROSCOPE)
        gravity.append(GRAVITY_NOT_APPLICABLE)
    if not modalities:
        raise ValueError("a stream must contain accelerometer or gyroscope")
    return torch.tensor(modalities, dtype=torch.long), torch.tensor(gravity, dtype=torch.long)


_GRAVITY_STATE_CACHE: dict[tuple[str, str], str | None] = {}


def _stream_gravity_state(dataset: str, stream: str) -> str | None:
    """Cached authoritative gravity state used by physics-aware augmentations and text."""
    key = (dataset, stream)
    if key not in _GRAVITY_STATE_CACHE:
        try:
            from data.scripts.curate.deployment_policy import get_stream_spec
            _GRAVITY_STATE_CACHE[key] = get_stream_spec(dataset, stream).gravity_state
        except (KeyError, ValueError, ImportError):
            _GRAVITY_STATE_CACHE[key] = None
    return _GRAVITY_STATE_CACHE[key]


# ----------------------------------------------------------------------------------------------
# Corpus index
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowKey:
    stream_i: int
    window_i: int
    label_id: int


def validation_subjects_for_refs(
    refs: Sequence[GridRef],
    *,
    seed: int = SEED,
    phase_a_only_datasets: frozenset[str] = PHASE_A_ONLY_DATASETS,
    rng: np.random.Generator | None = None,
) -> set[tuple[str, str]]:
    """Return the exact subject-disjoint validation split used by encoder training.

    This is shared with offline artifacts derived from the training corpus. Keeping the split in one
    function prevents filterbank/bias calibration from silently seeing validation subjects while the
    optimizer does not. A subject is eligible for validation only when every label they carry remains
    represented by at least one training subject. This makes a smaller validation fold preferable to
    creating a validation-only class that the optimizer can never learn.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    selected: set[tuple[str, str]] = set()
    by_dataset: dict[str, set[str]] = {}
    subj_labels: dict[tuple[str, str], set[str]] = {}
    for ref in refs:
        by_dataset.setdefault(ref.dataset, set()).update(ref.subjects)
        for subject, label in zip(ref.subjects, ref.labels):
            subj_labels.setdefault((ref.dataset, subject), set()).add(canonicalize(label))
    for dataset, subjects in sorted(by_dataset.items()):
        if dataset in phase_a_only_datasets:
            continue
        ordered = sorted(subjects)
        rng.shuffle(ordered)
        n_val = max(1, int(round(len(ordered) * VAL_SUBJECT_FRACTION)))
        # MM-Fit releases workout ids, not the person mapping. A generic workout-disjoint split is
        # therefore not person-disjoint. The paper identifies these five workouts collectively as
        # its previously-unseen-participant test set; holding out the complete set is the only split
        # the public metadata can prove is person-disjoint.
        if dataset == "mmfit":
            guaranteed = {"w00", "w05", "w12", "w13", "w20"}
            missing = guaranteed - set(ordered)
            if missing:
                raise ValueError(f"MM-Fit is missing published cross-subject workouts: {sorted(missing)}")
            selected.update((dataset, subject) for subject in sorted(guaranteed))
            continue
        remaining_label_subjects: dict[str, int] = {}
        for subject in ordered:
            for label in subj_labels.get((dataset, subject), set()):
                remaining_label_subjects[label] = remaining_label_subjects.get(label, 0) + 1

        # Only labels occurring in at least two subjects can be represented on both sides of a
        # subject-disjoint split. Singleton-subject labels remain optimizer-only and are measured
        # later on the sealed datasets rather than becoming impossible validation targets.
        need = {label for label, count in remaining_label_subjects.items() if count >= 2}
        picked: list[str] = []

        def can_hold_out(subject: str) -> bool:
            return all(
                remaining_label_subjects[label] > 1
                for label in subj_labels.get((dataset, subject), set())
            )

        def hold_out(subject: str) -> None:
            picked.append(subject)
            for label in subj_labels.get((dataset, subject), set()):
                remaining_label_subjects[label] -= 1

        while len(picked) < n_val and need:
            best = max(
                (s for s in ordered if s not in picked and can_hold_out(s)),
                key=lambda s: (len(subj_labels.get((dataset, s), set()) & need), s),
                default=None,
            )
            if best is None or not (subj_labels.get((dataset, best), set()) & need):
                break
            hold_out(best)
            need -= subj_labels.get((dataset, best), set())
        for subject in ordered:
            if len(picked) >= n_val:
                break
            if subject not in picked and can_hold_out(subject):
                hold_out(subject)
        selected.update((dataset, subject) for subject in picked)
    return selected


class CorpusIndex:
    """Discover, curate, subject-split, and label the lazy pretraining corpus index."""

    def __init__(self, max_per_stream: int | None = None, seed: int = SEED,
                 datasets: Sequence[str] = TRAIN_DATASETS, alignment: str = "native",
                 window_seconds: float | None = None):
        # HALO trains on the "native" grids: native sampling RATE (no 60 Hz resample — the filterbank
        # is rate-invariant, so real rates beat a homogenized base + synthetic rate aug) with the
        # canonical labels + 6-ch pad+mask layout this loader expects. The 60 Hz "harmonised" grids
        # remain the layout-locked baselines' source; they are not used here.
        self.alignment = alignment
        self.max_per_stream = max_per_stream   # retained for the checkpoint corpus fingerprint (F5)
        self.seed = seed
        self.datasets = tuple(datasets)
        self.window_seconds = window_seconds
        self.refs: list[GridRef] = [
            r for r in discover_grids(alignment, window_seconds=window_seconds)
            if r.dataset in set(datasets)
        ]
        if not self.refs:
            raise FileNotFoundError(f"no {alignment} train grids found — build grids first "
                                    f"(python -m data.scripts.build_grids --alignment {alignment})")
        # A zero-window placeholder is not a usable dataset. Treat it as missing
        # so a failed/empty optional conversion cannot satisfy an explicit roster.
        materialized_datasets = {ref.dataset for ref in self.refs if ref.n_windows > 0}
        missing_datasets = sorted(set(datasets) - materialized_datasets)
        if missing_datasets:
            raise FileNotFoundError(
                f"requested {alignment} datasets have no grids: {missing_datasets}. "
                "Build them explicitly or remove them from --datasets; silently training on a "
                "partial roster would make the run unattributable."
            )
        self.stream_datasets = [r.dataset for r in self.refs]   # stream_i -> dataset (for the sampler)
        rng = np.random.default_rng(seed)

        # Windows the plausibility scan rejected as physically impossible (accel/gyro beyond any
        # consumer full-scale range). Cached by data.scripts.scan_implausible so indexing stays lazy.
        from data.scripts.scan_implausible import load as _load_implausible
        quality_window_seconds = 6.0 if window_seconds is None else float(window_seconds)
        self.implausible = _load_implausible(
            alignment, require=True, window_seconds=quality_window_seconds,
        )
        # Byte-identical repeated windows — a device re-emitting a stale buffer, not motion
        # (ExtraSensory's Pebble does this for hours at a time). Merged into the same drop set:
        # both are "this window is not an observation", and CorpusIndex applies one filter.
        from data.scripts.scan_duplicates import load as _load_duplicates
        self.duplicates = _load_duplicates(
            alignment, require=True, window_seconds=quality_window_seconds,
        )
        self.excluded = {
            key: self.implausible.get(key, set()) | self.duplicates.get(key, set())
            for key in set(self.implausible) | set(self.duplicates)
        }

        # Subject-disjoint split per dataset, chosen to COVER AS MANY LABELS as the budget allows
        # while retaining at least one training subject for every label.
        # A purely random 10% draw left whole labels with zero val windows (e.g. `sleeping`: 15,100
        # train / 0 val; `table_tennis`: 216/0), so val_knn_ba / val_conse_ba / best.pt selection
        # silently omitted whole labels while the code claimed all of them. Greedy set-cover over
        # subjects fixes that without touching disjointness (a subject is still wholly train or val)
        # or creating validation-only labels.
        val_subjects = validation_subjects_for_refs(self.refs, seed=seed, rng=rng)

        # balanced selection + label map (train labels only)
        label_ids: dict[str, int] = {}
        self.train: list[WindowKey] = []
        self.val: list[WindowKey] = []
        self.n_implausible_dropped = 0
        self.n_duplicate_dropped = 0
        chosen_by_stream: dict[int, np.ndarray] = {}
        if max_per_stream is not None:
            by_dataset: dict[str, list[int]] = {}
            for stream_i, ref in enumerate(self.refs):
                by_dataset.setdefault(ref.dataset, []).append(stream_i)
            for dataset, stream_indices in by_dataset.items():
                if dataset in MULTI_DEVICE_TRAIN_DATASETS and len(stream_indices) > 1:
                    refs = [self.refs[index] for index in stream_indices]
                    if not all(ref.event_ids_explicit for ref in refs):
                        raise ValueError(f"{dataset}: aligned multi-device training needs explicit event ids")
                    common = set(refs[0].event_ids)
                    for ref in refs[1:]:
                        common.intersection_update(ref.event_ids)
                    ordered = np.asarray(sorted(common), dtype=object)
                    if len(ordered) > max_per_stream:
                        ordered = ordered[np.sort(rng.choice(
                            len(ordered), size=max_per_stream, replace=False,
                        ))]
                    selected_events = set(ordered.tolist())
                    for stream_i, ref in zip(stream_indices, refs):
                        chosen_by_stream[stream_i] = np.asarray([
                            row for row, event in enumerate(ref.event_ids)
                            if event in selected_events
                        ], dtype=np.int64)
                else:
                    for stream_i in stream_indices:
                        n = self.refs[stream_i].n_windows
                        chosen_by_stream[stream_i] = (
                            np.arange(n) if n <= max_per_stream
                            else np.sort(rng.choice(n, size=max_per_stream, replace=False))
                        )
        for stream_i, ref in enumerate(self.refs):
            n = ref.n_windows
            bad = self.implausible.get(ref.key, set())
            dup = self.duplicates.get(ref.key, set())
            chosen = np.arange(n) if max_per_stream is None else chosen_by_stream[stream_i]
            for w in np.sort(chosen):
                if int(w) in bad:                    # physically impossible window — drop, never clip
                    self.n_implausible_dropped += 1
                    continue
                if int(w) in dup:                    # stale re-emitted buffer, not an observation
                    self.n_duplicate_dropped += 1
                    continue
                # Canonicalize again at the consumption boundary. New synonym decisions therefore
                # take effect immediately on an existing grid, while clean grid rebuilds still write
                # the same canonical labels at source.
                label = canonicalize(ref.labels[int(w)])
                if label not in label_ids:
                    label_ids[label] = len(label_ids)
                key = WindowKey(stream_i, int(w), label_ids[label])
                if (ref.dataset, ref.subjects[int(w)]) in val_subjects:
                    self.val.append(key)
                else:
                    self.train.append(key)
        self.label_ids = label_ids
        subject_ids: dict[tuple[str, str], int] = {}
        self.train_subject_ids = []
        for key in self.train:
            ref = self.refs[key.stream_i]
            subject = (ref.dataset, str(ref.subjects[key.window_i]))
            self.train_subject_ids.append(
                subject_ids.setdefault(subject, len(subject_ids))
            )
        # Which labels the VAL split can actually score. best.pt is selected on val_knn_ba, so a
        # label with zero val windows is silently unscored — surface it instead of claiming
        # "all classes are scored" (audit F1).
        val_labels = {k.label_id for k in self.val}
        metric_labels = {
            label: label_id for label, label_id in label_ids.items()
            if label != UNLABELED_LABEL
        }
        self.val_missing_labels = sorted(
            label for label, label_id in metric_labels.items() if label_id not in val_labels
        )
        self.n_val_labels = len(val_labels)
        self.n_semantic_labels = len(metric_labels)
        # Shuffle val so any truncated eval subset (embed() caps at val_max_windows) is a
        # representative cross-dataset sample — index.val is otherwise stream-ordered, so a
        # 2k cap saw only capture24 (alphabetically first) = 8 of 56 labels.
        rng.shuffle(self.val)

    def summary(self) -> str:
        extra = ""
        if getattr(self, "n_implausible_dropped", 0):
            extra += f" · {self.n_implausible_dropped} implausible dropped"
        if getattr(self, "n_duplicate_dropped", 0):
            extra += f" · {self.n_duplicate_dropped} duplicate dropped"
        if getattr(self, "val_missing_labels", None):
            extra += f" · val scores {self.n_val_labels}/{self.n_semantic_labels} labels " \
                     f"(missing: {', '.join(self.val_missing_labels[:4])})"
        elif hasattr(self, "n_val_labels"):
            extra += f" · val scores ALL {self.n_val_labels} labels"
        return (f"{len(self.refs)} streams · {len(self.train)} train / {len(self.val)} val "
                f"windows · {len(self.label_ids)} labels{extra}")


# ----------------------------------------------------------------------------------------------
# Dataset + augmentation
# ----------------------------------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    """Reseed np.random AND stdlib random per worker (the augmenter uses both)."""
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed + worker_id)
    stdlib_random.seed(seed + worker_id + 1)


@contextmanager
def _item_randomness(seed: int):
    """Replay module-level augmentation draws without perturbing the caller's RNG state."""
    numpy_state = np.random.get_state()
    python_state = stdlib_random.getstate()
    np.random.seed(int(seed) % (2 ** 32))
    stdlib_random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        stdlib_random.setstate(python_state)


class PretrainDataset(Dataset):
    """One item = one augmented window: variable (T', 6) data + rate + texts + label.

    ``two_view`` (historical masked-JEPA/VICReg control): also emit an independently augmented second view
    window under ``item["view_b"]`` (its own signal augmentation, rate, channel_mask and
    augmentation-consistent channel text). The collate patchifies it into the ``*_b`` keys."""

    def __init__(self, index: CorpusIndex, keys: list[WindowKey],
                 augment: bool = True, two_view: bool = False,
                 augmentation_config: AugmentationConfig | None = None,
                 rotation_pairing: str = "shared",
                 neutral_acquisition_text: bool = False,
                 conditioning_schema: str = CONDITIONING_SCHEMA_V2,
                 multi_device_probability: float = 0.0,
                 max_devices: int = 4,
                 build_aligned_device_index: bool = False):
        self.index = index
        self.keys = keys
        self.two_view = two_view
        self.augment_enabled = bool(augment)
        self.neutral_acquisition_text = bool(neutral_acquisition_text)
        if conditioning_schema not in {CONDITIONING_SCHEMA_V2, LEGACY_CONDITIONING_SCHEMA}:
            raise ValueError(f"unsupported conditioning schema {conditioning_schema!r}")
        self.conditioning_schema = conditioning_schema
        if not 0.0 <= multi_device_probability <= 1.0:
            raise ValueError("multi_device_probability must be in [0,1]")
        if max_devices < 2:
            raise ValueError("max_devices must be at least 2")
        self.multi_device_probability = float(multi_device_probability)
        self.max_devices = int(max_devices)
        if rotation_pairing not in {"shared", "independent"}:
            raise ValueError("rotation_pairing must be 'shared' or 'independent'")
        cfg = (augmentation_config or AugmentationConfig.phase_a()) \
            if augment else AugmentationConfig.none()
        nuisance_cfg, config_cfg = cfg.split_by_group()
        if rotation_pairing == "independent" and cfg.rotation_3d.enabled:
            nuisance_cfg.rotation_3d.enabled = True
            nuisance_cfg.rotation_3d.p = cfg.rotation_3d.p
            config_cfg.rotation_3d.enabled = False
        self.config_augmenter = IMUAugmenter(config_cfg)
        self.nuisance_augmenter = IMUAugmenter(nuisance_cfg)
        self._data_cache: dict[int, np.ndarray] = {}
        self._length_cache: dict[int, np.ndarray] = {}
        self._aligned_devices = self._build_aligned_device_index() \
            if self.multi_device_probability > 0 or build_aligned_device_index else {}

    def _build_aligned_device_index(self) -> dict[int, tuple[int, ...]]:
        groups: dict[tuple[str, str], list[int]] = {}
        for position, key in enumerate(self.keys):
            ref = self.index.refs[key.stream_i]
            if ref.dataset not in MULTI_DEVICE_TRAIN_DATASETS:
                continue
            groups.setdefault((ref.dataset, str(ref.event_ids[key.window_i])), []).append(position)
        result = {}
        for positions in groups.values():
            streams = {self.index.refs[self.keys[position].stream_i].stream for position in positions}
            if len(streams) < 2:
                continue
            labels = {self.keys[position].label_id for position in positions}
            if len(labels) != 1:
                raise ValueError("aligned device rows disagree on label")
            ordered = tuple(sorted(positions, key=lambda p: self.index.refs[self.keys[p].stream_i].stream))
            for position in ordered:
                result[position] = ordered
        return result

    def aligned_device_members(self, position: int) -> dict[str, int]:
        """Return this window's exact-time-aligned devices by stable stream identifier.

        This small public contract is intentionally separate from ``item_with_rng``.  Episode
        construction can therefore select one device relationship for a whole query/support set
        before any signal is loaded, rather than independently composing every row.
        """
        peers = self._aligned_devices.get(int(position), ())
        return {
            self.index.refs[self.keys[peer].stream_i].stream: int(peer)
            for peer in peers
        }

    def __len__(self) -> int:
        return len(self.keys)

    def _grid(self, stream_i: int) -> np.ndarray:
        if stream_i not in self._data_cache:
            # Copy-on-write mappings are writable from PyTorch's perspective but never modify the
            # persisted grid. This permits allocation-free tensor views on the plain classifier
            # path without PyTorch's undefined-behaviour warning for read-only NumPy buffers.
            path = self.index.refs[stream_i].grid_dir / "data.npy"
            self._data_cache[stream_i] = np.load(path, mmap_mode="c")
        return self._data_cache[stream_i]

    def _lengths(self, stream_i: int) -> np.ndarray:
        if stream_i not in self._length_cache:
            self._length_cache[stream_i] = self.index.refs[stream_i].load_lengths()
        return self._length_cache[stream_i]

    @staticmethod
    def _clone_sample(sample: IMUSample) -> IMUSample:
        """Clone mutable signal/metadata before drawing one independent nuisance view."""
        return IMUSample(
            data=sample.data.clone(),
            channel_names=list(sample.channel_names),
            sampling_rate=float(sample.sampling_rate),
            channel_descriptions=list(sample.channel_descriptions),
            channel_mask=(list(sample.channel_mask) if sample.channel_mask is not None else None),
            role_descriptions=(list(sample.role_descriptions)
                               if sample.role_descriptions is not None else None),
            sensor_descriptions=(list(sample.sensor_descriptions)
                                 if sample.sensor_descriptions is not None else None),
            sensor_id=(list(sample.sensor_id) if sample.sensor_id is not None else None),
            gravity_state=sample.gravity_state,
            applied_augmentations=list(sample.applied_augmentations),
        )

    def _raw_sample(
        self,
        ref,
        key: WindowKey,
        base_texts: list[str],
        *,
        copy_data: bool = True,
    ) -> IMUSample:
        valid_length = int(self._lengths(key.stream_i)[key.window_i])
        source = self._grid(key.stream_i)[key.window_i, :valid_length]
        if copy_data:
            # Augmenters mutate samples, so their input must never alias the memory-mapped grid.
            source = np.array(source, dtype=np.float32, copy=True)
        else:
            # The plain classification path only reads this view and immediately copies it into
            # compact collate storage. Avoid a redundant per-recording allocation beforehand.
            source = np.asarray(source, dtype=np.float32)
        window = torch.from_numpy(source)
        role_texts, sensor_texts, sensor_id = stream_sensor_texts(
            ref.dataset, ref.stream,
            has_accel=bool(any(ref.mask[:3])), has_gyro=bool(any(ref.mask[3:])),
            neutral=self.neutral_acquisition_text,
            conditioning_schema=getattr(self, "conditioning_schema", CONDITIONING_SCHEMA_V2),
        )
        return IMUSample(
            data=window,
            channel_names=list(CHANNELS),
            sampling_rate=ref.rate_hz,
            channel_descriptions=list(base_texts),
            channel_mask=[bool(m) for m in ref.mask],   # real vs zero-padded channels (F10b)
            role_descriptions=role_texts,
            sensor_descriptions=sensor_texts,
            sensor_id=sensor_id,
            gravity_state=_stream_gravity_state(ref.dataset, ref.stream),
        )

    def _sample_to_slots(self, ref, base_texts: list[str], slot: dict,
                         sample: IMUSample) -> dict:
        """Scatter one already-augmented view back into the canonical six-slot layout."""

        # channel_dropout REMOVES channels from the tensor (e.g. gyro drop -> (T',3)).
        # Scatter survivors back into the canonical 6-slot layout and mask the rest —
        # the same pad+mask contract the grids use.
        data6 = sample.data.new_zeros(sample.data.shape[0], len(CHANNELS))
        mask6 = torch.zeros(len(CHANNELS), dtype=torch.bool)
        texts6 = list(base_texts)
        for j, name in enumerate(sample.channel_names):
            i = slot[name]
            data6[:, i] = sample.data[:, j]
            mask6[i] = bool(ref.mask[i])
            texts6[i] = sample.channel_descriptions[j]    # keep augmented text
        # Enforce the pad+mask contract: augmentations (jitter etc.) write noise into
        # grid-masked zero-filled channels — a masked slot must stay exactly zero.
        data6[:, ~mask6] = 0.0
        # Scatter the fully augmented role text back to canonical slots just like the signal. Sensor
        # text is already sensor-level and carries any gravity/phrase/dropout augmentation.
        role_texts6 = [_CHANNEL_ROLE_TEXT[c] for c in CHANNELS]
        sensor_id6 = torch.zeros(len(CHANNELS), dtype=torch.long)
        for j, name in enumerate(sample.channel_names):
            i = slot[name]
            if sample.role_descriptions is not None:
                role_texts6[i] = sample.role_descriptions[j]
            if sample.sensor_id is not None:
                sensor_id6[i] = int(sample.sensor_id[j])
        # Acquisition rate = the widest band the HARDWARE ever measured, which no later processing
        # can widen. Two stages can only ever narrow it:
        #   * the converter (STREAM_SOURCE_RATE_HZ records streams stored above their capture clock),
        #   * the rate augmentation, which is a real anti-aliased resample_poly.
        # So the bound is min(hardware, augmented). Scaling the hardware rate by the augmentation
        # ratio (the previous formula) is wrong in BOTH directions: upsampling wisdm 20 -> 94 Hz
        # advertised 94 Hz of measured bandwidth (32 observable bands where only 26 carry signal),
        # and downsampling xrf_v2's 25 Hz AirPods stream to 21.9 Hz reported 10.9 Hz, hiding 5 real
        # bands. This feeds the encoder tokens, so an over-claim teaches the filterbank that
        # interpolation artifacts are measurable signal.
        hardware_rate = STREAM_SOURCE_RATE_HZ.get(f"{ref.dataset}/{ref.stream}", float(ref.rate_hz))
        source_rate = min(float(hardware_rate), float(sample.sampling_rate))
        sensor_texts_out = list(sample.sensor_descriptions or ())
        if not sensor_texts_out:
            raise ValueError("an augmented Phase-A sample has no sensor description")
        modality, gravity = self._structured_fields(sample, len(sensor_texts_out))
        return {
            "data": data6,                                # (T', 6) canonical slots
            "rate": float(sample.sampling_rate),
            "source_rate": source_rate,
            "channel_source_rates": torch.full(
                (len(CHANNELS),), source_rate, dtype=torch.float32),
            "texts": texts6,
            # Factored text (docs/design/TEXT_CONDITIONING.md §4b): carried per view so the VICReg
            # second view gets its OWN independently-augmented role/sensor text. label_id is
            # view-independent and lives in __getitem__ below.
            "role_texts": role_texts6,
            "sensor_texts": sensor_texts_out,
            "sensor_id": sensor_id6,
            "sensor_modality": modality,
            "sensor_gravity": gravity,
            "sensor_rates_hz": torch.tensor(
                [[float(sample.sampling_rate), source_rate]] * len(sensor_texts_out),
                dtype=torch.float32,
            ),
            "device_id": torch.zeros(len(sensor_texts_out), dtype=torch.long),
            # Placement group id per sensor. The sensor-mask JEPA objective uses this to refuse
            # cross-placement prediction; within one stream every sensor shares a placement, so
            # this is constant here and becomes meaningful when paired streams are fused.
            "sensor_placement": torch.zeros(
                len(sensor_texts_out), dtype=torch.long),
            "channel_mask": mask6,
            "gravity_state": sample.gravity_state,
            "augmentations": tuple(sample.applied_augmentations),
        }

    def _plain_sample_to_slots(self, ref, sample: IMUSample) -> dict:
        """Zero-copy metadata assembly for the non-augmented classification/eval path."""
        if sample.channel_names != list(CHANNELS) or sample.data.shape[1] != len(CHANNELS):
            raise ValueError("plain grid samples must already use the canonical six-channel layout")
        sensor_texts = list(sample.sensor_descriptions or ())
        if not sensor_texts:
            raise ValueError("a plain sample has no sensor description")
        hardware_rate = STREAM_SOURCE_RATE_HZ.get(
            f"{ref.dataset}/{ref.stream}", float(ref.rate_hz),
        )
        source_rate = min(float(hardware_rate), float(sample.sampling_rate))
        modality, gravity = self._structured_fields(sample, len(sensor_texts))
        return {
            "data": sample.data,
            "rate": float(sample.sampling_rate),
            "source_rate": source_rate,
            "channel_source_rates": torch.full(
                (len(CHANNELS),), source_rate, dtype=torch.float32),
            "texts": list(sample.channel_descriptions),
            "role_texts": list(sample.role_descriptions or (_CHANNEL_ROLE_TEXT[c] for c in CHANNELS)),
            "sensor_texts": sensor_texts,
            "sensor_id": torch.as_tensor(sample.sensor_id, dtype=torch.long),
            "sensor_modality": modality,
            "sensor_gravity": gravity,
            "sensor_rates_hz": torch.tensor(
                [[float(sample.sampling_rate), source_rate]] * len(sensor_texts),
                dtype=torch.float32,
            ),
            "device_id": torch.zeros(len(sensor_texts), dtype=torch.long),
            "sensor_placement": torch.zeros(len(sensor_texts), dtype=torch.long),
            "channel_mask": torch.as_tensor(sample.channel_mask, dtype=torch.bool),
            "gravity_state": sample.gravity_state,
            "augmentations": (),
        }

    @staticmethod
    def _structured_fields(sample: IMUSample, n_sensors: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive exact modality IDs after any channel dropout/config augmentation."""
        if sample.sensor_id is None:
            raise ValueError("sensor_id is required for conditioning-v2 metadata")
        live = (sample.channel_mask if sample.channel_mask is not None
                else [True] * len(sample.channel_names))
        if len(live) != len(sample.channel_names):
            raise ValueError("channel_mask must align with channel_names")
        modality = []
        for sensor in range(n_sensors):
            names = [name for name, owner, exists in zip(
                sample.channel_names, sample.sensor_id, live,
            ) if bool(exists) and int(owner) == sensor]
            kinds = {"accel" if name.startswith("acc_") else "gyro" if name.startswith("gyro_")
                     else "unsupported" for name in names}
            if kinds == {"accel"}:
                modality.append(MODALITY_ACCELEROMETER)
            elif kinds == {"gyro"}:
                modality.append(MODALITY_GYROSCOPE)
            else:
                raise ValueError(f"sensor {sensor} has unsupported or mixed modality channels {names}")
        gravity = [
            ({"present": GRAVITY_PRESENT, "removed": GRAVITY_REMOVED,
              "unknown": GRAVITY_UNKNOWN, None: GRAVITY_UNKNOWN}.get(sample.gravity_state)
             if kind == MODALITY_ACCELEROMETER else GRAVITY_NOT_APPLICABLE)
            for kind in modality
        ]
        if any(value is None for value in gravity):
            raise ValueError(f"invalid accelerometer gravity state {sample.gravity_state!r}")
        return torch.tensor(modality, dtype=torch.long), torch.tensor(gravity, dtype=torch.long)

    def _can_use_plain_sample(self) -> bool:
        """Return whether the current, possibly test-overridden augmenters are no-ops."""
        if self.two_view:
            return False
        return all(
            not getattr(augmenter.cfg, name).enabled
            for augmenter in (self.config_augmenter, self.nuisance_augmenter)
            for name in AugmentationConfig.ORDER
        )

    def _single_item(self, i: int) -> dict:
        key = self.keys[i]
        ref = self.index.refs[key.stream_i]
        base_texts = stream_channel_descriptions(
            ref.dataset, ref.stream, neutral=self.neutral_acquisition_text,
        )
        slot = {c: k for k, c in enumerate(CHANNELS)}
        plain = not self.augment_enabled and self._can_use_plain_sample()
        raw = self._raw_sample(ref, key, base_texts, copy_data=not plain)
        if plain:
            view = self._plain_sample_to_slots(ref, raw)
            sensor_target_texts = list(view["sensor_texts"])
            view["sensor_target_texts"] = sensor_target_texts
            return {
                **view,
                "label_id": key.label_id,
                "source": ref.dataset,
                "stream": ref.key,
                "window_index": key.window_i,
                "subject": f"{ref.dataset}:{ref.subjects[key.window_i]}",
            }
        # Historical masked control: draw acquisition CONFIG once. Both VICReg views then
        # independently draw only nuisance
        # variation from clones of that configured sample. This is both the intended semantics and
        # substantially cheaper than replaying every CONFIG transform under saved global RNG state.
        configured = self.config_augmenter(raw)
        # Descriptor-mask prediction must target acquisition SEMANTICS, not the random surface form
        # drawn later by the nuisance paraphrase augmentation. Otherwise two equivalent phrasings of
        # the same sensor become false negatives and the hidden signal cannot determine which wording
        # happened to be sampled. Capture the stable target after configuration-changing transforms
        # (gravity/channel set) and before independently paraphrasing each positive view.
        sensor_target_texts = list(configured.sensor_descriptions or ())
        if not sensor_target_texts:
            raise ValueError("a configured Phase-A sample has no descriptor target")
        view = self._sample_to_slots(
            ref, base_texts, slot,
            self.nuisance_augmenter(self._clone_sample(configured)),
        )
        view["sensor_target_texts"] = sensor_target_texts
        item = {
            **view,                                       # data/rate/texts/role_texts/sensor_texts/sensor_id/channel_mask/gravity_state
            "label_id": key.label_id,
            "source": ref.dataset,                        # for per-source telemetry
            "stream": ref.key,
            "window_index": key.window_i,
            "subject": f"{ref.dataset}:{ref.subjects[key.window_i]}",
        }
        if self.two_view:
            # Second view: independent nuisance realization over the exact same acquisition config.
            view_b = self._sample_to_slots(
                ref, base_texts, slot,
                self.nuisance_augmenter(self._clone_sample(configured)),
            )
            view_b["sensor_target_texts"] = list(sensor_target_texts)
            item["view_b"] = {
                **view_b,
                "label_id": key.label_id,
                "source": ref.dataset,
            }
        return item

    def item_with_rng(self, i: int, rng: np.random.Generator) -> dict:
        """Load one item with replayable device composition and signal augmentation."""
        def replay_seed() -> int:
            # Derive augmentation randomness from the structural RNG without advancing it. Device
            # composition therefore remains episode-identical to runs predating augmentation
            # replay, including when augmentation probabilities are zero.
            shadow = np.random.default_rng()
            shadow.bit_generator.state = rng.bit_generator.state
            return int(shadow.integers(0, np.iinfo(np.int64).max, dtype=np.int64))

        peers = self._aligned_devices.get(i, ())
        use_composite = bool(peers) and rng.random() < self.multi_device_probability
        if not use_composite:
            seed = replay_seed()
            with _item_randomness(seed):
                return self._single_item(i)
        others = [position for position in peers if position != i]
        count = int(rng.integers(2, min(self.max_devices, len(peers)) + 1))
        chosen = [i, *rng.choice(others, size=count - 1, replace=False).tolist()]
        seed = replay_seed()
        # Canonical stream order makes device IDs stable while query/support subsets remain
        # independently sampled by separate dataset accesses. Replay the same transform draw for
        # every aligned device: rate and modality perturbations describe one acquisition event and
        # must preserve a common timeline before the device rows can be merged.
        chosen.sort(key=lambda p: self.index.refs[self.keys[p].stream_i].stream)
        items = []
        for position in chosen:
            with _item_randomness(seed):
                items.append(self._single_item(position))
        return merge_device_items(items)

    def item_with_members(self, members: Sequence[int], rng: np.random.Generator) -> dict:
        """Load a pre-planned aligned device subset with one shared augmentation replay.

        ``members`` must describe one physical event.  The planner supplies canonical stream
        order, but we enforce that invariant here too because a malformed composite would make
        device-set evaluation deceptively easy.
        """
        chosen = tuple(int(position) for position in members)
        if not chosen:
            raise ValueError("a device-set plan needs at least one member")
        maps = [self.aligned_device_members(position) for position in chosen]
        if len(chosen) > 1 and (not maps[0] or any(position not in maps[0].values() for position in chosen)):
            raise ValueError("device-set members must be exact-time-aligned peers")
        ordered = tuple(sorted(chosen, key=lambda p: self.index.refs[self.keys[p].stream_i].stream))
        shadow = np.random.default_rng()
        shadow.bit_generator.state = rng.bit_generator.state
        seed = int(shadow.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        items = []
        for position in ordered:
            with _item_randomness(seed):
                items.append(self._single_item(position))
        out = items[0] if len(items) == 1 else merge_device_items(items)
        out["device_set"] = tuple(
            self.index.refs[self.keys[position].stream_i].stream for position in ordered
        )
        return out

    def __getitem__(self, i: int) -> dict:
        # DataLoader compatibility. Trainer prefetch uses a step-owned generator instead.
        return self.item_with_rng(i, np.random.default_rng(int(i)))


def _capped_probabilities(
    scores: dict[str, float],
    max_share: float | None,
) -> dict[str, float]:
    """Normalize positive source scores while enforcing a feasible probability ceiling."""
    if not scores or any(value <= 0 for value in scores.values()):
        raise ValueError("source scores must be non-empty and strictly positive")
    total = float(sum(scores.values()))
    probabilities = {key: float(value) / total for key, value in scores.items()}
    if max_share is None:
        return probabilities
    if not 0.0 < float(max_share) <= 1.0:
        raise ValueError("max_dataset_share must be in (0, 1]")

    # Fewer than 1/max_share datasets makes the requested ceiling mathematically impossible.
    ceiling = max(float(max_share), 1.0 / len(probabilities))
    fixed: dict[str, float] = {}
    remaining = dict(probabilities)
    while remaining:
        budget = 1.0 - sum(fixed.values())
        remainder_total = sum(remaining.values())
        proposed = {
            key: budget * value / remainder_total
            for key, value in remaining.items()
        }
        over = [key for key, value in proposed.items() if value > ceiling + 1e-12]
        if not over:
            fixed.update(proposed)
            break
        for key in over:
            fixed[key] = ceiling
            del remaining[key]
    return fixed


class TemperatureSampler(Sampler[int]):
    """Hierarchical label-free corpus sampler (NO activity-class balancing).

    Dataset mass starts proportional to ``n_dataset ** alpha`` and can be explicitly capped.
    When subject IDs are supplied, each dataset's mass is then split as
    ``P(subject | dataset) ∝ n_subject ** subject_alpha`` before drawing a window uniformly
    from that subject.
      * alpha = 1  -> proportional to dataset size (large corpora dominate),
      * alpha = 0  -> uniform per source (every dataset equally likely),
      * alpha = 0.25 + a 0.25 cap is the Phase-A recipe.
    Draws without replacement inside each batch and with replacement across batches, yielding
    exactly ``num_samples // batch_size`` full batches.
    There is no per-class structure; activity labels are unused."""

    def __init__(self, keys: list[WindowKey], stream_datasets: list[str], num_samples: int,
                 batch_size: int, alpha: float = 0.5, seed: int = SEED,
                 subject_ids: Sequence[int] | None = None,
                 subject_alpha: float = 1.0,
                 max_dataset_share: float | None = None,
                 batch_group_ids: Sequence[int] | None = None):
        from collections import Counter
        datasets = [stream_datasets[k.stream_i] for k in keys]
        counts = Counter(datasets)
        if float(alpha) < 0:
            raise ValueError("alpha must be nonnegative")
        if not 0.0 <= float(subject_alpha) <= 1.0:
            raise ValueError("subject_alpha must be in [0, 1]")
        if subject_ids is not None and len(subject_ids) != len(keys):
            raise ValueError("subject_ids must align 1:1 with keys")
        if batch_group_ids is not None and len(batch_group_ids) != len(keys):
            raise ValueError("batch_group_ids must align 1:1 with keys")
        self.dataset_probabilities = _capped_probabilities(
            {dataset: count ** float(alpha) for dataset, count in counts.items()},
            max_dataset_share,
        )

        if subject_ids is None:
            row_weights = [
                self.dataset_probabilities[dataset] / counts[dataset]
                for dataset in datasets
            ]
        else:
            subject_counts = Counter(zip(datasets, subject_ids))
            subject_scores: dict[str, float] = {}
            for (dataset, _subject), count in subject_counts.items():
                subject_scores[dataset] = (
                    subject_scores.get(dataset, 0.0) + count ** float(subject_alpha)
                )
            row_weights = []
            for dataset, subject in zip(datasets, subject_ids):
                count = subject_counts[(dataset, subject)]
                within_dataset = (
                    count ** (float(subject_alpha) - 1.0)
                    / subject_scores[dataset]
                )
                row_weights.append(self.dataset_probabilities[dataset] * within_dataset)
        self.weights = torch.tensor(row_weights, dtype=torch.double)
        self.batch_group_ids = (torch.as_tensor(batch_group_ids, dtype=torch.long)
                                if batch_group_ids is not None else None)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        if self.num_samples <= 0 or self.batch_size < 2:
            raise ValueError("num_samples must be positive and batch_size must be at least 2")
        if self.batch_size > len(keys):
            raise ValueError("batch_size cannot exceed the number of indexed windows")
        if self.num_samples % self.batch_size:
            raise ValueError("num_samples must be divisible by batch_size")
        self.epoch = 0

    @staticmethod
    def _draw_with_replacement(
        weights: torch.Tensor, count: int, generator: torch.Generator,
    ) -> torch.Tensor:
        """Draw from arbitrarily large categorical tables without the 2^24 torch limit."""
        max_categories = 1 << 24
        if weights.numel() <= max_categories:
            return torch.multinomial(weights, count, replacement=True, generator=generator)
        # Hierarchical draw: choose a bounded contiguous block by its exact mass, then a row
        # inside that block. This is mathematically the same categorical distribution while
        # keeping every torch.multinomial call within its supported category count.
        chunk = 1 << 20
        starts = torch.arange(0, weights.numel(), chunk, device=weights.device)
        masses = torch.stack([weights[start:start + chunk].sum() for start in starts])
        choices = torch.multinomial(masses, count, replacement=True, generator=generator)
        result = torch.empty(count, dtype=torch.long)
        for choice in torch.unique(choices, sorted=True).tolist():
            output = torch.nonzero(choices == choice).flatten()
            start = int(starts[choice])
            local = torch.multinomial(
                weights[start:start + chunk], len(output), replacement=True, generator=generator,
            )
            result[output] = local + start
        return result

    @staticmethod
    def _draw_unique_batches(
        weights: torch.Tensor,
        n_batches: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Weighted rows with replacement across batches and without replacement within one."""
        n_samples = n_batches * batch_size
        extra = max(batch_size * 4, n_samples // 500)
        draws = TemperatureSampler._draw_with_replacement(weights, n_samples + extra, generator)
        batches = draws[:n_samples].clone().view(-1, batch_size)
        ordered = batches.sort(dim=1).values
        duplicate_rows = (
            ordered[:, 1:] == ordered[:, :-1]
        ).any(dim=1).nonzero().flatten().tolist()
        extras = draws[n_samples:].tolist()
        extra_pos = 0
        for row_index in duplicate_rows:
            row = batches[row_index].tolist()
            seen: set[int] = set()
            for position, value in enumerate(row):
                if value not in seen:
                    seen.add(value)
                    continue
                while True:
                    if extra_pos >= len(extras):
                        extras.extend(TemperatureSampler._draw_with_replacement(
                            weights, max(batch_size * 4, 4096), generator,
                        ).tolist())
                    replacement = extras[extra_pos]
                    extra_pos += 1
                    if replacement not in seen:
                        row[position] = replacement
                        seen.add(replacement)
                        break
            batches[row_index] = torch.tensor(row, dtype=batches.dtype)
        return batches

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        # Advance the seed per epoch so the calibration pass and each training epoch draw
        # different windows on each new sampler pass.
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        bs = self.batch_size
        n_batches = self.num_samples // bs
        if self.batch_group_ids is None:
            batches = self._draw_unique_batches(self.weights, n_batches, bs, gen)
        else:
            groups = torch.unique(self.batch_group_ids, sorted=True)
            rows_by_group = [torch.nonzero(self.batch_group_ids == group).flatten()
                             for group in groups]
            if any(len(rows) < bs for rows in rows_by_group):
                raise ValueError("every batch group must contain at least batch_size distinct rows")
            mass = torch.stack([self.weights.index_select(0, rows).sum()
                                for rows in rows_by_group])
            batch_groups = torch.multinomial(mass, n_batches, replacement=True, generator=gen)
            batches = torch.empty(n_batches, bs, dtype=torch.long)
            for group_index, rows in enumerate(rows_by_group):
                destinations = torch.nonzero(batch_groups == group_index).flatten()
                if not len(destinations):
                    continue
                local = self._draw_unique_batches(
                    self.weights.index_select(0, rows), len(destinations), bs, gen,
                )
                selected = rows.index_select(0, local.reshape(-1)).view_as(local)
                batches.index_copy_(0, destinations, selected)
        for batch in batches:
            yield from batch.tolist()


# ----------------------------------------------------------------------------------------------
# Multi-scale collate
# ----------------------------------------------------------------------------------------------
def _batch_identity_seed(batch: list[dict], seed: int) -> int:
    """Stable batch-local RNG seed that contains no activity-label information."""
    digest = hashlib.blake2b(digest_size=8, person=b"halo-patch")
    digest.update(int(seed).to_bytes(8, "little", signed=True))
    for position, item in enumerate(batch):
        identity = (
            position,
            item.get("source", "?"),
            item.get("stream", "?"),
            item.get("window_index", -1),
        )
        digest.update(repr(identity).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little")


def merge_device_items(items: Sequence[dict]) -> dict:
    """Combine exact-time-aligned single-device rows into one variable-channel recording."""
    if len(items) < 2:
        raise ValueError("a composite training/evaluation item needs at least two devices")
    first = items[0]
    for item in items[1:]:
        if item["data"].shape[0] != first["data"].shape[0] or not np.isclose(item["rate"], first["rate"]):
            raise ValueError("HALO composite members must share one sampled timeline")
        if item.get("label_id") != first.get("label_id"):
            raise ValueError("HALO composite members disagree on the activity label")
    sensor_offsets = np.cumsum([0] + [len(item.get("sensor_texts") or ()) for item in items[:-1]])
    sensor_ids = [torch.as_tensor(item["sensor_id"], dtype=torch.long) + int(offset)
                  for item, offset in zip(items, sensor_offsets)]
    sensor_texts = [text for item in items for text in item.get("sensor_texts", ())]
    target_texts = [text for item in items
                    for text in item.get("sensor_target_texts", item.get("sensor_texts", ()))]
    out = {
        **first,
        "data": torch.cat([torch.as_tensor(item["data"]) for item in items], dim=1),
        "texts": [text for item in items for text in item["texts"]],
        "role_texts": [text for item in items for text in item.get("role_texts", item["texts"])],
        "sensor_texts": sensor_texts,
        "sensor_target_texts": target_texts,
        "sensor_id": torch.cat(sensor_ids),
        "device_id": torch.cat([
            torch.full((len(item.get("sensor_texts") or ()),), device, dtype=torch.long)
            for device, item in enumerate(items)
        ]),
        "sensor_placement": torch.cat([
            torch.full((len(item.get("sensor_texts") or ()),), device, dtype=torch.long)
            for device, item in enumerate(items)
        ]),
        "sensor_bias": torch.cat([torch.as_tensor(item["sensor_bias"]) for item in items], dim=0)
        if all(item.get("sensor_bias") is not None for item in items) else None,
        "channel_mask": torch.cat([torch.as_tensor(item["channel_mask"]) for item in items]),
        # Kept as a scalar compatibility field for older callers. Per-channel bandwidth is
        # carried separately so a low-rate device cannot hide another device's valid bands.
        "source_rate": min(float(item.get("source_rate", item["rate"])) for item in items),
        "channel_source_rates": torch.cat([
            torch.as_tensor(item.get("channel_source_rates", torch.full(
                (torch.as_tensor(item["channel_mask"]).numel(),),
                float(item.get("source_rate", item["rate"])))))
            for item in items
        ]),
        "stream": "+".join(str(item.get("stream", "?")) for item in items),
        "augmentations": tuple(value for item in items for value in item.get("augmentations", ())),
    }
    _merge_conditioning_metadata(out, items)
    return out


def _physical_patch_bounds(num_samples: int, rate_hz: float,
                           patch_seconds: float) -> list[tuple[int, int]]:
    """Partition samples using rounded physical-time boundaries.

    Repeatedly stepping by ``round(rate * seconds)`` accumulates rounding error. At 51.2 Hz a
    six-second, 307-sample context became six 51-sample patches plus a meaningless one-sample token.
    Rounding each absolute boundary distributes that sample across the six intended supports.
    """
    if num_samples <= 0:
        return []
    span = float(rate_hz) * float(patch_seconds)
    if span <= 0:
        raise ValueError("rate_hz and patch_seconds must be positive")
    count = max(1, int(np.ceil(num_samples / span - 1e-9)))
    bounds: list[tuple[int, int]] = []
    for patch in range(count):
        start = int(round(patch * span))
        if start >= num_samples:
            break
        end = min(num_samples, int(round((patch + 1) * span)))
        end = max(end, start + 1)
        bounds.append((start, end))
    return bounds


def _bounded_analysis_view(data: np.ndarray, rate_hz: float, durations: Sequence[float]) -> tuple[np.ndarray, float]:
    """Anti-alias a long-resolution view to the bounded physical analysis clock.

    The filterbank maps FFT bins back to Hz using the returned rate, while callers retain the
    original source rate separately for observability metadata.  This is not a second view or an
    acquisition-rate augmentation: it is the one signal supplied to every resolution in a
    long-patch batch, so overlapping resolutions remain temporally consistent.
    """
    if max(map(float, durations), default=0.0) < 4.0 or rate_hz <= FILTERBANK_LONG_PATCH_ANALYSIS_HZ:
        return data, float(rate_hz)
    from fractions import Fraction
    from scipy.signal import resample_poly

    target = FILTERBANK_LONG_PATCH_ANALYSIS_HZ
    ratio = Fraction(target / float(rate_hz)).limit_denominator(10_000)
    was_tensor = torch.is_tensor(data)
    values = resample_poly(
        np.asarray(data.detach().cpu() if was_tensor else data, dtype=np.float32),
        ratio.numerator, ratio.denominator, axis=0,
    )
    expected = max(1, int(round(len(data) * target / float(rate_hz))))
    if len(values) > expected:
        values = values[:expected]
    elif len(values) < expected:
        values = np.pad(values, ((0, expected - len(values)), (0, 0)), mode="edge")
    values = np.asarray(values, dtype=np.float32)
    if was_tensor:
        values = torch.from_numpy(values).to(dtype=data.dtype)
    return values, float(target)


def _bounded_analysis_views(
    batch: Sequence[dict], durations: Sequence[float],
) -> list[tuple[torch.Tensor, float]]:
    """Vectorized equivalent of :func:`_bounded_analysis_view` for collate workers.

    Grid rows in one support batch overwhelmingly share a rate and length. Designing and applying
    one polyphase filter to that stack is the same independent operation along each row's time
    axis, while avoiding hundreds of tiny SciPy calls.
    """
    result: list[tuple[torch.Tensor, float] | None] = [None] * len(batch)
    groups: dict[tuple[float, tuple[int, ...]], list[int]] = {}
    long_grid = max(map(float, durations), default=0.0) >= 4.0
    for index, item in enumerate(batch):
        data = torch.as_tensor(item["data"], dtype=torch.float32)
        rate = float(item["rate"])
        if not long_grid or rate <= FILTERBANK_LONG_PATCH_ANALYSIS_HZ:
            result[index] = (data, rate)
        else:
            groups.setdefault((rate, tuple(data.shape)), []).append(index)

    if groups:
        from fractions import Fraction
        from scipy.signal import resample_poly

        target = FILTERBANK_LONG_PATCH_ANALYSIS_HZ
        for (rate, shape), indices in groups.items():
            values = np.stack([
                np.asarray(batch[index]["data"], dtype=np.float32) for index in indices
            ])
            ratio = Fraction(target / rate).limit_denominator(10_000)
            values = resample_poly(values, ratio.numerator, ratio.denominator, axis=1)
            expected = max(1, int(round(shape[0] * target / rate)))
            if values.shape[1] > expected:
                values = values[:, :expected]
            elif values.shape[1] < expected:
                values = np.pad(
                    values, ((0, 0), (0, expected - values.shape[1]), (0, 0)), mode="edge",
                )
            values = np.asarray(values, dtype=np.float32)
            for row, index in enumerate(indices):
                result[index] = (torch.from_numpy(values[row]), float(target))
    if any(value is None for value in result):
        raise RuntimeError("bounded analysis failed to materialize every batch row")
    return result  # type: ignore[return-value]


class MultiScaleCollate:
    """Draw ONE patch_seconds per batch; patchify each sample at its OWN rate.

    Trailing patches the (possibly rate-shortened / cropped) window cannot fill are flagged in
    ``patch_padding_mask`` and never treated as real.

    Gravity is not aligned. The tokenizer's signed-DC feature preserves gravity direction, while
    SO(3) augmentation teaches mounting robustness. Canonicalizing pitch/roll would erase posture
    information and cancel that augmentation.

    Output: patches (B, P, S, 6) zero-padded · patch_len (B,P) · rates (B,) ·
    positions (B, P) s · channel_mask (B, 6) · patch_padding_mask (B, P) True=real ·
    texts · labels.
    """

    def __init__(self, dft_size: int = DFT_SIZE,
                 patch_choices: Sequence[float] = PATCH_SECONDS_CHOICES,
                 fixed_patch_seconds: float | None = None, seed: int = SEED,
                 two_view: bool = False):
        self.dft_size = dft_size
        self.patch_choices = tuple(patch_choices)
        self.fixed = fixed_patch_seconds
        # Historical masked control: patchify the independent VICReg view into `*_b` keys.
        self.two_view = two_view
        self.seed = seed

    def _patch_seconds(self, batch: list[dict]) -> float:
        if self.fixed is not None:
            return self.fixed
        # Key the deterministic draw to recording identity, never to activity labels. Using Python's
        # randomized ``hash`` also made the same run differ across worker processes.
        key = _batch_identity_seed(batch, self.seed)
        return float(np.random.default_rng(key).choice(self.patch_choices))

    def __call__(self, batch: list[dict]) -> dict:
        ps = self._patch_seconds(batch)
        out = self._collate_impl(batch, ps)
        if self.two_view and batch and "view_b" in batch[0]:
            # Historical masked control: the positive view uses the same patch duration.
            out_b = self._collate_impl([item["view_b"] for item in batch], ps)
            for k in ("patches", "patch_len", "rates", "source_rates", "channel_source_rates", "positions",
                      "patch_durations", "patch_starts", "patch_ends", "resolution_ids",
                      "resolution_count", "texts",
                      "role_texts", "sensor_texts", "sensor_target_texts", "sensor_id",
                      "sensor_modality", "sensor_gravity", "sensor_rates_hz",
                      "device_id",
                      "sensor_placement",
                      "channel_mask", "patch_padding_mask", "augmentations"):
                out[f"{k}_b"] = out_b[k]
            if "sensor_bias" in out_b:
                out["sensor_bias_b"] = out_b["sensor_bias"]
        return out

    def _collate_impl(self, batch: list[dict], ps: float) -> dict:
        bounds_by_shape = {}
        bounds = []
        prepared = []
        for item in batch:
            data, rate = _bounded_analysis_view(item["data"], float(item["rate"]), (ps,))
            prepared.append((data, rate))
            key = (data.shape[0], float(rate))
            if key not in bounds_by_shape:
                bounds_by_shape[key] = _physical_patch_bounds(*key, ps)
            bounds.append(bounds_by_shape[key])
        P = max(1, max(map(len, bounds)))
        B = len(batch)
        channels = max(data.shape[1] for data, _ in prepared)
        patches = torch.zeros(B, P, self.dft_size, channels)
        # Fill small metadata arrays on the CPU without a tensor dispatch per scalar.
        patch_len = np.zeros((B, P), dtype=np.int64)
        patch_durations = np.zeros((B, P), dtype=np.float32)
        patch_pad = np.zeros((B, P), dtype=bool)     # True = real patch
        rates = np.zeros(B, dtype=np.float32)
        source_rates = np.zeros(B, dtype=np.float32)
        positions = np.zeros((B, P), dtype=np.float32)
        patch_starts = np.zeros((B, P), dtype=np.float32)
        patch_ends = np.zeros((B, P), dtype=np.float32)

        for b, (item, (data, rate)) in enumerate(zip(batch, prepared)):
            if int(np.ceil(rate * ps)) > self.dft_size:
                raise ValueError(
                    f"patch length {int(np.ceil(rate * ps))} exceeds dft_size {self.dft_size}"
                )
            usable = 0
            for p, (start, end) in enumerate(bounds[b]):
                length = end - start
                patches[b, p, :length, :data.shape[1]] = data[start:end]
                patch_len[b, p] = length
                patch_durations[b, p] = length / rate
                positions[b, p] = (start + 0.5 * length) / rate
                patch_starts[b, p] = start / rate
                patch_ends[b, p] = end / rate
                usable += 1
            patch_pad[b, :usable] = True
            rates[b] = rate
            source_rates[b] = float(item.get("source_rate", rate))
        out = {
            "patches": patches,
            "patch_len": torch.from_numpy(patch_len),
            "rates": torch.from_numpy(rates),
            "source_rates": torch.from_numpy(source_rates),
            "channel_source_rates": _pad_channel_rows(batch, "channel_source_rates"),
            "positions": torch.from_numpy(positions),
            "patch_durations": torch.from_numpy(patch_durations),
            "patch_starts": torch.from_numpy(patch_starts),
            "patch_ends": torch.from_numpy(patch_ends),
            "resolution_ids": torch.where(
                torch.from_numpy(patch_pad),
                torch.zeros((B, P), dtype=torch.long),
                torch.full((B, P), -1, dtype=torch.long),
            ),
            "resolution_count": 1,
            "patch_seconds": ps,
            "texts": _pad_channel_text(batch, "texts"),
            # Factored text conditioning (docs/design/TEXT_CONDITIONING.md §4b), read ONLY by the
            # factored path. Tolerant of manual items (eval/tests) that omit them — those keep the
            # legacy per_channel path where these are unused (None), so the default is unaffected.
            "role_texts": _pad_channel_text(batch, "role_texts"),
            "sensor_texts": [item.get("sensor_texts") for item in batch],
            "sensor_target_texts": [
                item.get("sensor_target_texts", item.get("sensor_texts")) for item in batch
            ],
            "sensor_id": _pad_channel_rows(batch, "sensor_id"),
            "device_id": _pad_sensor_rows(batch, "device_id"),
            # Sensor-placement metadata is used only to constrain physically valid JEPA masks.
            "sensor_placement": _pad_sensor_rows(batch, "sensor_placement"),
            "labels": torch.tensor([item["label_id"] for item in batch]),
            "sources": [item.get("source", "?") for item in batch],   # per-window dataset (telemetry)
            "streams": [item.get("stream", "?") for item in batch],
            "window_indices": torch.tensor([item.get("window_index", -1) for item in batch]),
            "subjects": [item.get("subject", "?") for item in batch],
            "augmentations": [item.get("augmentations", ()) for item in batch],
            "channel_mask": _pad_channel_rows(batch, "channel_mask"),
            "patch_padding_mask": torch.from_numpy(patch_pad),
        }
        # Legacy checkpoint evaluation can inject the frozen artifact explicitly. New Phase-A
        # datasets omit it, so no source-specific statistics enter or burden the training path.
        if "sensor_bias" in batch[0]:
            out["sensor_bias"] = _pad_sensor_rows(batch, "sensor_bias")
        _attach_conditioning_metadata(out, batch)
        return out


class MultiResolutionCollate:
    """Present several physical-duration patch grids in the same token sequence.

    Unlike ``MultiScaleCollate``, this collate retains partial tail patches and emits a true
    length for every token. Tokens are sorted by physical center time, with ``resolution_ids``
    identifying each duration in ascending order. Signal augmentation has already happened once
    in ``PretrainDataset``; every grid therefore describes exactly the same augmented view.

    With no fixed durations, the Phase-A augmentation retains its historical random short/long
    pair. A fixed sequence may contain two or more durations, which is used by the scale-aware
    comparison experiment.
    """

    def __init__(
        self,
        dft_size: int = DFT_SIZE,
        short_choices: Sequence[float] = SHORT_PATCH_SECONDS_CHOICES,
        long_choices: Sequence[float] = LONG_PATCH_SECONDS_CHOICES,
        fixed_patch_seconds: Sequence[float] | None = None,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
        min_resolution_ratio: float = MIN_RESOLUTION_RATIO,
        seed: int = SEED,
        two_view: bool = False,
    ):
        self.dft_size = int(dft_size)
        self.short_choices = tuple(float(x) for x in short_choices)
        self.long_choices = tuple(float(x) for x in long_choices)
        self.fixed = fixed_patch_seconds
        self.max_batch_tokens = int(max_batch_tokens)
        self.min_resolution_ratio = float(min_resolution_ratio)
        self.seed = int(seed)
        # Historical masked control: patchify the independent VICReg view into `*_b` keys.
        self.two_view = bool(two_view)
        self._valid_pairs = tuple(
            (short, long)
            for short in self.short_choices
            for long in self.long_choices
            if long >= self.min_resolution_ratio * short
        )
        if self.fixed is not None:
            fixed = tuple(sorted(set(map(float, self.fixed))))
            if len(fixed) < 2 or fixed[0] <= 0:
                raise ValueError("fixed multi-resolution durations require at least two positive values")
            if fixed[-1] < self.min_resolution_ratio * fixed[0]:
                raise ValueError("fixed resolution pair does not satisfy min_resolution_ratio")
            self.fixed = fixed
        elif not self._valid_pairs:
            raise ValueError("no short/long duration pair satisfies min_resolution_ratio")

    def _patch_seconds(self, batch: list[dict]) -> tuple[float, ...]:
        if self.fixed is not None:
            return self.fixed
        key = _batch_identity_seed(batch, self.seed)
        rng = np.random.default_rng(key)
        pairs = self._valid_pairs
        if self.max_batch_tokens:
            # Peak VRAM scales with the TOKEN count B x P, and P is a function of the
            # patch_seconds drawn for THIS batch: short draws tile the window more finely.
            # Measured, P swings 12 -> 22 at a fixed batch size, so memory was a random
            # variable and an unlucky draw could OOM a batch size that had survived the
            # previous sixty steps. Restrict the draw to pairs whose token count fits the
            # budget; if none do, take the coarsest pair (fewest tokens) rather than fail.
            n = len(batch)
            max_seconds = max(
                float(item["data"].shape[0]) / max(float(item["rate"]), 1e-9) for item in batch
            )
            def _tokens(pair: tuple[float, ...]) -> int:
                return n * sum(int(np.ceil(max_seconds / p)) for p in pair)
            affordable = [p for p in pairs if _tokens(p) <= self.max_batch_tokens]
            pairs = affordable or [max(pairs, key=lambda p: p[0] + p[1])]
        return pairs[int(rng.integers(len(pairs)))]

    def __call__(self, batch: list[dict]) -> dict:
        pair = self._patch_seconds(batch)
        out = self._collate_impl(batch, pair)
        if self.two_view and batch and "view_b" in batch[0]:
            # Historical masked control: the positive view uses the same resolution pair.
            out_b = self._collate_impl([item["view_b"] for item in batch], pair)
            for k in ("patches", "patch_len", "rates", "source_rates", "channel_source_rates", "positions", "patch_durations",
                      "resolution_ids", "texts", "role_texts", "sensor_texts",
                      "sensor_target_texts", "sensor_id",
                      "sensor_modality", "sensor_gravity", "sensor_rates_hz",
                      "device_id",
                      "sensor_placement",
                      "channel_mask", "patch_padding_mask", "augmentations"):
                out[f"{k}_b"] = out_b[k]
            if "sensor_bias" in out_b:
                out["sensor_bias_b"] = out_b["sensor_bias"]
        return out

    def deferred(self, batch: list[dict]) -> dict:
        """Collate metadata but defer DFT-padded patch materialization to the accelerator.

        Support-classifier workers use this path because transferring a recording once is much
        cheaper than transferring three resolution grids that are mostly zero tokens. The normal
        collate remains the reference path for pretraining, calibration and compatibility tests.
        """
        if self.two_view:
            raise ValueError("deferred multi-resolution collation does not support two-view data")
        return self._collate_impl(batch, self._patch_seconds(batch), defer_patch_values=True)

    def _collate_impl(
        self,
        batch: list[dict],
        pair: tuple[float, ...],
        *,
        defer_patch_values: bool = False,
    ) -> dict:
        B = len(batch)
        rates = torch.zeros(B)
        source_rates = torch.zeros(B)
        channel_mask = _pad_channel_rows(batch, "channel_mask")
        all_entries: list[list[tuple]] = []
        analysis_data: list[torch.Tensor] = []

        bounded_views = _bounded_analysis_views(batch, pair)
        for b, item in enumerate(batch):
            data, rate = bounded_views[b]
            analysis_data.append(data)
            rates[b] = rate
            source_rates[b] = float(item.get("source_rate", item["rate"]))

            entries = []
            for resolution_id, duration in enumerate(pair):
                nominal_n = max(1, int(np.ceil(rate * duration)))
                if nominal_n > self.dft_size:
                    raise ValueError(
                        f"patch length {nominal_n} exceeds dft_size {self.dft_size}"
                    )
                min_tail = max(4, int(MIN_TAIL_FRACTION * nominal_n))   # F7 tail floor
                res_start = len(entries)
                for start, end in _physical_patch_bounds(data.shape[0], rate, duration):
                    n = end - start
                    if n <= 0:
                        continue
                    # F7: a sub-fraction TAIL patch (e.g. UniMiB's 1-sample remainder) is a degenerate
                    # token — its ~0.02 s duration is clamped to the duration-embedding floor and it
                    # still joins self-attention. Drop it rather than feed a physically meaningless
                    # token; but never drop the ONLY patch of a resolution (a window shorter than one
                    # patch keeps its single partial). Duration-weighting (F1) handles the rest.
                    # Preserve the complete measured interval.  A short terminal token is
                    # explicitly duration-weighted downstream, while dropping it gave HALO
                    # less evidence than the baselines from the same requested recording.
                    start_s, end_s = start / rate, end / rate
                    entries.append((
                        0.5 * (start_s + end_s), resolution_id, start_s, end_s,
                        n / rate, n, start, end,
                    ))
            # Physical-time order keeps RoPE and contiguous per-resolution masking meaningful.
            # Short tokens precede long tokens only when their centers are exactly equal.
            entries.sort(key=lambda x: (x[0], x[1]))
            all_entries.append(entries)

        P = max((len(entries) for entries in all_entries), default=1)
        channels = max(item["data"].shape[1] for item in batch)
        patches = (None if defer_patch_values
                   else torch.zeros(B, P, self.dft_size, channels))
        patch_len = torch.zeros(B, P, dtype=torch.long)
        patch_pad = torch.zeros(B, P, dtype=torch.bool)
        positions = torch.zeros(B, P)
        patch_durations = torch.zeros(B, P)
        patch_starts = torch.zeros(B, P)
        patch_ends = torch.zeros(B, P)
        patch_start_samples = torch.zeros(B, P, dtype=torch.long)
        resolution_ids = torch.full((B, P), -1, dtype=torch.long)

        for b, entries in enumerate(all_entries):
            for p, (center, rid, start, end, duration, n,
                    sample_start, sample_end) in enumerate(entries):
                if patches is not None:
                    values = analysis_data[b][sample_start:sample_end]
                    patches[b, p, :n, :values.shape[1]] = values.to(dtype=patches.dtype)
                patch_len[b, p] = n
                patch_pad[b, p] = True
                positions[b, p] = center
                patch_durations[b, p] = duration
                patch_starts[b, p] = start
                patch_ends[b, p] = end
                patch_start_samples[b, p] = sample_start
                resolution_ids[b, p] = rid

        compact_data = None
        compact_lengths = None
        if defer_patch_values:
            max_samples = max((len(values) for values in analysis_data), default=1)
            compact_data = torch.zeros(B, max_samples, channels)
            compact_lengths = torch.zeros(B, dtype=torch.long)
            for b, values in enumerate(analysis_data):
                compact_data[b, :len(values), :values.shape[1]] = values
                compact_lengths[b] = len(values)

        out = {
            "patches": patches,
            "patch_len": patch_len,
            "rates": rates,
            "source_rates": source_rates,
            "channel_source_rates": _pad_channel_rows(batch, "channel_source_rates"),
            "positions": positions,
            "patch_durations": patch_durations,
            "patch_starts": patch_starts,
            "patch_ends": patch_ends,
            "patch_start_samples": patch_start_samples,
            "resolution_ids": resolution_ids,
            "resolution_count": len(pair),
            "patch_seconds": pair,
            "texts": _pad_channel_text(batch, "texts"),
            # Factored text conditioning (docs/design/TEXT_CONDITIONING.md §4b), read ONLY by the
            # factored path. Tolerant of manual items (eval/tests) that omit them — those keep the
            # legacy per_channel path where these are unused (None), so the default is unaffected.
            "role_texts": _pad_channel_text(batch, "role_texts"),
            "sensor_texts": [item.get("sensor_texts") for item in batch],
            "sensor_target_texts": [
                item.get("sensor_target_texts", item.get("sensor_texts")) for item in batch
            ],
            "sensor_id": _pad_channel_rows(batch, "sensor_id"),
            "device_id": _pad_sensor_rows(batch, "device_id"),
            # Sensor-placement metadata is used only to constrain physically valid JEPA masks.
            "sensor_placement": _pad_sensor_rows(batch, "sensor_placement"),
            "labels": torch.tensor([item["label_id"] for item in batch]),
            "sources": [item.get("source", "?") for item in batch],
            "streams": [item.get("stream", "?") for item in batch],
            "window_indices": torch.tensor([item.get("window_index", -1) for item in batch]),
            "subjects": [item.get("subject", "?") for item in batch],
            "augmentations": [item.get("augmentations", ()) for item in batch],
            "channel_mask": channel_mask,
            "patch_padding_mask": patch_pad,
        }
        if defer_patch_values:
            out["compact_data"] = compact_data
            out["compact_lengths"] = compact_lengths
        if "sensor_bias" in batch[0]:
            out["sensor_bias"] = _pad_sensor_rows(batch, "sensor_bias")
        _attach_conditioning_metadata(out, batch)
        return out
