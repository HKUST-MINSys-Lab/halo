"""Build the frozen archetypal memory bank for Phase-B (the evidence engine).

This is the initial expensive pass in Phase B. We encode the training corpus with the frozen
Pipeline-A encoder and cache pooled/patch vectors, metadata, and exact source pointers. Frozen
Phase-B training operates on the cache. Optional ``ema_finetune`` training keeps the global cache
detached and re-encodes only bounded selected source windows. The prior contract is indexed by
``docs/LEGACY.md``.

Cached bank (``memory_bank.pt``):
    Z          (N, d)  float16   pooled frozen-encoder embeddings (L2-normalizable downstream)
    y          (N,)    int64     global-vocab label index (canonicalized; -1 dropped)
    subj       (N,)    int64     composite "dataset:subject" id (for subject-disjoint episodes)
    cfg        (N,)    int64     stream id "dataset/stream" (config bucket)
    vocab      list[str]         current global label vocabulary (row i == label index i)
    label_text (L, 384) float32  frozen-SBERT embedding of each vocab label (the ConSE text space)
    subj_names / cfg_names        int->string decoders for subj / cfg
    backbone   dict              provenance (ckpt path, step, val_ba, git, content fingerprint)
    patch      dict              valid patch vectors + parent window/event/config,
                                center time, duration, resolution, and verification metadata
    source_row (N,) int64        original native-grid row for exact live re-encoding

The pooled keys are intentionally unchanged: they are the T2.0 control. Schema 3 adds the patch
    table, exact source provenance, and a behavioral patch-path probe. ``--sensor-rows`` emits schema 5
with the per-(patch, sensor) table required by the current admissibility design.

Memory is built from CLEAN (un-augmented) encodings — a retrieval bank of jittered
vectors would be matching against noise. Label/query augmentation is a *training-loop*
concern (the learned t-kernel), not a memory concern.

Run:
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    HALO_CKPT=training/tokenizer/outputs/phase_a_fixed_1s_rotation_20260817/best.pt \
      $PY -m training.evidence.build_memory --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import torch

from data.scripts.eda.grid_io import discover_grids, grid_corpus_fingerprint
from data.scripts.labels.canonical_labels import NON_SEMANTIC_LABELS, canonicalize
from eval.scoring import get_sbert_encoder
from training.tokenizer.eval_transfer import (
    build_encoder,
    embedding_fingerprint,
    encode_dataset_detailed,
    patch_embedding_fingerprint,
    sensor_embedding_fingerprint,
)
from training.tokenizer.pretrain_data import (
    TRAIN_DATASETS,
    _stream_gravity_state,
    stream_channel_descriptions,
    stream_sensor_bias,
    stream_sensor_texts,
)
from training.evidence.policy import ARCHIVE_BUDGET_WINDOWS
from training.evidence.device import resolve_device

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_CKPT = (
    _REPO / "training/tokenizer/outputs/phase_a_fixed_1s_rotation_20260817/best.pt"
)
_DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "memory_bank.pt"
_GLOBAL_LABELS = _REPO / "data/labels/global_labels.json"


def _load_vocab(ckpt: dict) -> list[str]:
    """Load the current supervised vocabulary without silently relabelling old checkpoints.

    A label-free encoder checkpoint legitimately contains only ``__unlabeled__`` and is paired
    with the current supervised bank vocabulary. Historical labelled checkpoints, however, are
    reproducible only with the exact semantic vocabulary they recorded.
    """
    vocab = list(json.loads(_GLOBAL_LABELS.read_text())["labels"])
    checkpoint_labels = {
        canonicalize(str(label)) for label in ckpt.get("label_ids", {})
        if canonicalize(str(label)) not in NON_SEMANTIC_LABELS
    }
    if checkpoint_labels and checkpoint_labels != set(vocab):
        raise ValueError(
            "checkpoint semantic vocabulary differs from data/labels/global_labels.json; "
            "rebuild with its frozen vocabulary or retrain the checkpoint rather than silently "
            f"changing label indices (checkpoint={len(checkpoint_labels)}, current={len(vocab)})"
        )
    return vocab


def _backbone_fp(ckpt_path: Path) -> str:
    return hashlib.sha256(ckpt_path.read_bytes()).hexdigest() if ckpt_path.exists() else ""


def _population_fingerprint(fields: dict[str, torch.Tensor]) -> str:
    """Hash selected source identities and patch layout without re-hashing learned vectors."""
    digest = hashlib.sha256()
    for name in sorted(fields):
        value = torch.as_tensor(fields[name]).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()[:16]


def label_config_balanced_keep(
    labels: torch.Tensor,
    configs: torch.Tensor,
    max_per_label: int | None,
    *,
    seed: int = 20260720,
) -> torch.Tensor:
    """Select whole windows with an equalized configuration budget inside every label."""
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    configs = torch.as_tensor(configs, dtype=torch.long).cpu()
    if labels.shape != configs.shape:
        raise ValueError("labels and configs must have the same shape")
    if max_per_label is None:
        return torch.ones(len(labels), dtype=torch.bool)
    if max_per_label < 1:
        raise ValueError("max_per_label must be positive or None")

    keep = torch.zeros(len(labels), dtype=torch.bool)
    for label in sorted(labels.unique().tolist()):
        label_rows = torch.nonzero(labels.eq(label), as_tuple=True)[0]
        if len(label_rows) <= max_per_label:
            keep[label_rows] = True
            continue
        groups = {
            config: label_rows[configs[label_rows].eq(config)]
            for config in sorted(configs[label_rows].unique().tolist())
        }

        # Water-fill the per-label budget: equal quota for well-populated configurations, while
        # preserving every row from configurations smaller than their fair share.
        quotas: dict[int, int] = {}
        active = dict(groups)
        remaining = max_per_label
        while active:
            share = remaining // len(active)
            if share == 0:
                for config in sorted(active)[:remaining]:
                    quotas[config] = 1
                break
            small = [
                config for config, rows in active.items()
                if len(rows) <= share
            ]
            if small:
                for config in small:
                    quotas[config] = len(active[config])
                    remaining -= len(active[config])
                    del active[config]
                continue
            base, extra = divmod(remaining, len(active))
            for position, config in enumerate(sorted(active)):
                quotas[config] = base + int(position < extra)
            break

        for config, quota in quotas.items():
            rows = groups[config]
            if quota >= len(rows):
                chosen = rows
            else:
                generator = torch.Generator().manual_seed(
                    int(seed + 1_000_003 * label + 9_973 * config)
                )
                chosen = rows[torch.randperm(len(rows), generator=generator)[:quota]]
            keep[chosen] = True
    return keep


def archive_budget_balanced_keep(
    labels: torch.Tensor,
    configs: torch.Tensor,
    budget: int,
    *,
    seed: int = 20260720,
) -> torch.Tensor:
    """Spend one global archive budget across labels, then configurations.

    Scarce labels are retained in full. Their unused quota is redistributed to common labels, so
    the result uses the requested budget whenever enough windows exist. Each label's allocation is
    then divided across its available acquisition configurations.
    """
    n = len(labels)
    if budget < 1:
        raise ValueError("archive budget must be positive")
    if n <= budget:
        return torch.ones(n, dtype=torch.bool)
    counts = {int(label): int(labels.eq(label).sum()) for label in labels.unique().tolist()}
    quotas = {label: 0 for label in counts}
    remaining = int(budget)
    active = set(counts)
    while active and remaining:
        share = max(1, remaining // len(active))
        progressed = False
        for label in sorted(tuple(active)):
            available = counts[label] - quotas[label]
            take = min(share, available, remaining)
            quotas[label] += take
            remaining -= take
            progressed |= take > 0
            if quotas[label] == counts[label]:
                active.remove(label)
            if remaining == 0:
                break
        if not progressed:
            break

    keep = torch.zeros(n, dtype=torch.bool)
    for label, quota in quotas.items():
        label_rows = torch.nonzero(labels.eq(label), as_tuple=True)[0]
        label_cfg = configs[label_rows]
        cfg_values = sorted(map(int, label_cfg.unique().tolist()))
        cfg_quota = max(1, quota // len(cfg_values))
        chosen_parts = []
        for config in cfg_values:
            rows = label_rows[label_cfg.eq(config)]
            take = min(cfg_quota, len(rows))
            generator = torch.Generator().manual_seed(
                int(seed + 1_000_003 * label + 9_973 * config)
            )
            chosen_parts.append(rows[torch.randperm(len(rows), generator=generator)[:take]])
        chosen = torch.cat(chosen_parts) if chosen_parts else label_rows[:0]
        if len(chosen) < quota:
            unchosen = label_rows[~torch.isin(label_rows, chosen)]
            generator = torch.Generator().manual_seed(int(seed + 31_337 * label))
            fill = unchosen[torch.randperm(len(unchosen), generator=generator)[:quota - len(chosen)]]
            chosen = torch.cat([chosen, fill])
        elif len(chosen) > quota:
            generator = torch.Generator().manual_seed(int(seed + 65_537 * label))
            chosen = chosen[torch.randperm(len(chosen), generator=generator)[:quota]]
        keep[chosen] = True
    return keep


def _patch_ordinals(local_patch_window: torch.Tensor) -> torch.Tensor:
    """Zero-based patch row within each source window, preserving deterministic export order."""
    if len(local_patch_window) == 0:
        return torch.empty(0, dtype=torch.long)
    if bool((local_patch_window[1:] < local_patch_window[:-1]).any()):
        raise RuntimeError("detailed patch export is not grouped by source window")
    _, counts = torch.unique_consecutive(local_patch_window, return_counts=True)
    starts = counts.cumsum(0) - counts
    return torch.arange(len(local_patch_window)) - torch.repeat_interleave(starts, counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.environ.get("HALO_CKPT", _DEFAULT_CKPT)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--archive-budget-windows", type=int, default=ARCHIVE_BUDGET_WINDOWS,
                    help="one balanced archive-size budget; rare labels are retained first")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--gate-out", type=Path, default=None,
                    help="where to bind a gate embedded in an episodic checkpoint; defaults beside "
                         "the bank as admissibility_gate.pt")
    ap.add_argument("--sensor-rows", action="store_true",
                    help="emit one row per (patch, SENSOR) instead of per patch (design of record "
                         "bank layout). Requires a sensor-granularity encoder; roughly doubles the "
                         "row count on 6-channel streams and lets an accel-only query match accel "
                         "rows exactly instead of against a channel-set-mismatched embedding.")
    args = ap.parse_args()
    device = resolve_device(args.device)
    if args.archive_budget_windows < 1:
        ap.error("--archive-budget-windows must be positive")
    # Fixed engineering guard: the global archive policy below, rather than this scan ceiling,
    # determines the represented memory distribution.
    scan_cap_per_stream = 50_000

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"encoder checkpoint missing at {args.checkpoint}. Point --checkpoint / HALO_CKPT "
            "at the frozen sensor-granularity Phase-A run.")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    sensor_rows = bool(args.sensor_rows)
    if sensor_rows and getattr(enc, "token_granularity", "channel") != "sensor":
        # Fail loud. Silently falling back to patch rows would produce a bank that LOOKS
        # per-sensor to every downstream consumer and is not — the class of defect this repo has
        # been bitten by before (stale fingerprints, inert fixes).
        raise ValueError(
            "--sensor-rows requires a checkpoint trained with token_granularity='sensor'; "
            f"this one is '{getattr(enc, 'token_granularity', 'channel')}'")
    for p in enc.parameters():
        p.requires_grad_(False)
    d_model = int(ckpt["config"]["d_model"])
    selection_value = ckpt.get("selection_value", ckpt.get("val_ba"))
    if selection_value is None:
        raise ValueError("checkpoint records neither selection_value nor val_ba")
    selection_metric = ckpt.get("selection_metric", "val_ba")
    git_revision = ckpt.get("git", "unknown")
    print(f"[memory] encoder {args.checkpoint.name}: step {ckpt['step']}, "
          f"{selection_metric} {float(selection_value):.3f}, git {git_revision}, "
          f"frontend={ckpt['config'].get('frontend')}, "
          f"MR={ckpt['config'].get('multiresolution')}, d={d_model}", flush=True)

    vocab = _load_vocab(ckpt)
    label_to_idx = {l: i for i, l in enumerate(vocab)}
    rng = np.random.RandomState(20260720)

    # Label-free Phase A and supervised memory construction intentionally use disjoint corpora.
    # Historical labelled checkpoints retain their frozen training roster for reproducibility.
    checkpoint_labels = {
        canonicalize(str(label)) for label in ckpt.get("label_ids", {})
        if canonicalize(str(label)) not in NON_SEMANTIC_LABELS
    }
    phase_a_roster = tuple(ckpt["config"].get("train_datasets") or TRAIN_DATASETS)
    if checkpoint_labels:
        roster = phase_a_roster
        roster_source = "labelled checkpoint"
    else:
        from data.scripts.curate.deployment_policy import SUPERVISED_HEAD_TRAIN_DATASETS
        roster = tuple(SUPERVISED_HEAD_TRAIN_DATASETS)
        roster_source = "supervised bank policy (label-free Phase A checkpoint)"
    print(f"[memory] roster from {roster_source}: {sorted(roster)}", flush=True)
    refs = sorted((r for r in discover_grids("native") if r.dataset in roster),
                  key=lambda r: r.key)
    Z_parts, y_parts, subj_parts, cfg_parts, event_parts, event_verified_parts = [], [], [], [], [], []
    source_row_parts = []
    patch_Z_parts, patch_y_parts, patch_subj_parts = [], [], []
    patch_cfg_parts, patch_sensor_parts = [], []
    sensor_Z_parts, sensor_window_parts, sensor_slot_parts = [], [], []
    sensor_time_parts, sensor_duration_parts, sensor_resolution_parts = [], [], []
    sensor_y_parts, sensor_subj_parts, sensor_cfg_parts, sensor_event_parts = [], [], [], []
    sensor_modality_parts, sensor_gravity_parts, sensor_bias_parts = [], [], []
    patch_window_parts, patch_event_parts, patch_event_verified_parts = [], [], []
    patch_time_parts, patch_duration_parts, patch_resolution_parts, patch_ordinal_parts = [], [], [], []
    subj_names: dict[str, int] = {}
    cfg_names: dict[str, int] = {}
    cfg_rates: dict[int, float] = {}
    event_names: dict[str, int] = {}
    bank_streams: dict[str, int] = {}       # ref.key -> encoded window count (corpus provenance, F3)
    bank_datasets: set[str] = set()
    sensor_descriptions: dict[int, list[dict]] = {}
    next_window = 0

    # Quality screens. build_memory reads grids directly rather than through CorpusIndex, so it
    # must apply the same exclusions the pretraining corpus does — otherwise the bank holds windows
    # the encoder was never trained on, including ExtraSensory's stale-buffer repeats whose labels
    # contradict each other (see data/scripts/scan_duplicates).
    from data.scripts.scan_duplicates import load as _load_duplicates
    from data.scripts.scan_implausible import load as _load_implausible
    excluded_by_stream = _load_duplicates("native", require=True)
    for _key, _idx in _load_implausible("native", require=True).items():
        excluded_by_stream[_key] = excluded_by_stream.get(_key, set()) | _idx

    for ref in refs:
        gl = np.array([label_to_idx.get(canonicalize(l), -1) for l in ref.labels])
        excluded = excluded_by_stream.get(ref.key)
        if excluded:
            idx = np.fromiter(excluded, dtype=np.int64, count=len(excluded))
            stale = idx[idx >= len(gl)]
            if stale.size:      # cache predates a grid rebuild — refuse to guess which windows moved
                raise RuntimeError(
                    f"{ref.key}: quality cache indexes window {int(stale.max())} but the grid has "
                    f"{len(gl)}. Re-run data.scripts.scan_implausible and scan_duplicates."
                )
            gl = gl.copy()
            gl[idx] = -1
            print(f"[memory]   {ref.key}: {len(excluded)} windows excluded by quality screens",
                  flush=True)
        keep = np.where(gl >= 0)[0]
        if keep.size == 0:
            print(f"[memory]   {ref.key}: 0 in-vocab windows, skipped", flush=True)
            continue
        if keep.size > scan_cap_per_stream:
            keep = np.sort(rng.choice(keep, scan_cap_per_stream, replace=False))
        data = np.asarray(ref.load_data()[keep])
        texts = stream_channel_descriptions(ref.dataset, ref.stream)
        gs = _stream_gravity_state(ref.dataset, ref.stream)
        encoded = encode_dataset_detailed(
            enc, data, texts, device, float(ref.rate_hz), gs,
            channel_mask=ref.mask, dataset=ref.dataset, stream=ref.stream,
            lengths=np.asarray(ref.load_lengths())[keep],
            export_sensor_rows=sensor_rows,
        )
        z = encoded["pooled"]   # (n, d) cpu; unchanged pooled compatibility path
        cfg_id = cfg_names.setdefault(ref.key, len(cfg_names))
        cfg_rates[cfg_id] = float(ref.rate_hz)
        subj_arr = np.asarray(ref.subjects)[keep]
        s_ids = np.array([subj_names.setdefault(f"{ref.dataset}:{s}", len(subj_names))
                          for s in subj_arr], dtype=np.int64)
        event_arr = np.asarray(ref.event_ids, dtype=object)[keep]
        e_ids = np.array(
            [event_names.setdefault(str(event), len(event_names)) for event in event_arr],
            dtype=np.int64,
        )
        event_verified = torch.full(
            (keep.size,), bool(ref.event_ids_explicit), dtype=torch.bool
        )
        local_patch_window = encoded["patch_window"].long()
        patch_window_parts.append(local_patch_window + next_window)
        # ``encode_dataset_detailed`` returns pooled vectors and every metadata field on the CPU but
        # leaves patch vectors on the encoder device (the live fine-tuning path needs them there).
        # Narrow to fp16 first, then land on the CPU: otherwise the whole archive accumulates in
        # VRAM stream by stream (~1.9 GB at the current corpus size) and torch.save writes CUDA
        # tensors into the bank, which every consumer then has to map back.
        patch_Z_parts.append(encoded["patch_Z"].to(torch.float16).cpu())
        if sensor_rows:
            local_sensor_window = encoded["sensor_window"].long().cpu()
            slot = encoded["sensor_slot"].long().cpu()
            sensor_Z_parts.append(encoded["sensor_Z"].to(torch.float16).cpu())
            sensor_window_parts.append(local_sensor_window + next_window)
            sensor_slot_parts.append(slot)
            sensor_time_parts.append(encoded["sensor_time"].float().cpu())
            sensor_duration_parts.append(encoded["sensor_duration"].float().cpu())
            sensor_resolution_parts.append(encoded["sensor_resolution"].long().cpu())
            sensor_y_parts.append(torch.from_numpy(gl[keep].astype(np.int64))[local_sensor_window])
            sensor_subj_parts.append(torch.from_numpy(s_ids)[local_sensor_window])
            sensor_cfg_parts.append(
                torch.full((len(local_sensor_window),), cfg_id, dtype=torch.int64))
            sensor_event_parts.append(torch.from_numpy(e_ids)[local_sensor_window])
            # Modality and bias per row, indexed by the SAME slot ordering stream_sensor_texts and
            # stream_sensor_bias use (accel first when present, then gyro). Deriving both from one
            # `modalities` list is what keeps slot -> (modality, bias, descriptor) consistent.
            modalities = [m for m, present in (("accel", bool(any(ref.mask[:3]))),
                                               ("gyro", bool(any(ref.mask[3:])))) if present]
            slot_modality = torch.tensor([0 if m == "accel" else 1 for m in modalities],
                                         dtype=torch.int64)
            sensor_modality_parts.append(slot_modality[slot])
            slot_gravity = torch.tensor([
                int(m == "accel" and gs == "removed") for m in modalities
            ], dtype=torch.int64)
            sensor_gravity_parts.append(slot_gravity[slot])
            sensor_bias_parts.append(stream_sensor_bias(ref.dataset, ref.stream, modalities)[slot])
            _, descriptions, _ = stream_sensor_texts(
                ref.dataset,
                ref.stream,
                gravity_removed=gs == "removed",
                has_accel="accel" in modalities,
                has_gyro="gyro" in modalities,
            )
            sensor_descriptions[cfg_id] = [
                {
                    "slot": index,
                    "modality": 0 if modality == "accel" else 1,
                    "gravity": int(modality == "accel" and gs == "removed"),
                    "text": descriptions[index],
                }
                for index, modality in enumerate(modalities)
            ]
        patch_y_parts.append(torch.from_numpy(gl[keep].astype(np.int64))[local_patch_window])
        patch_subj_parts.append(torch.from_numpy(s_ids)[local_patch_window])
        patch_cfg_parts.append(
            torch.full((len(local_patch_window),), cfg_id, dtype=torch.int64)
        )
        patch_sensor_parts.append(
            torch.full((len(local_patch_window),), cfg_id, dtype=torch.int64)
        )
        patch_event_parts.append(torch.from_numpy(e_ids)[local_patch_window])
        patch_event_verified_parts.append(event_verified[local_patch_window])
        patch_time_parts.append(encoded["patch_time"].float())
        patch_duration_parts.append(encoded["patch_duration"].float())
        patch_resolution_parts.append(encoded["patch_resolution"].long())
        patch_ordinal_parts.append(_patch_ordinals(local_patch_window))
        Z_parts.append(z.to(torch.float16))
        y_parts.append(torch.from_numpy(gl[keep].astype(np.int64)))
        subj_parts.append(torch.from_numpy(s_ids))
        cfg_parts.append(torch.full((keep.size,), cfg_id, dtype=torch.int64))
        event_parts.append(torch.from_numpy(e_ids))
        event_verified_parts.append(event_verified)
        source_row_parts.append(torch.from_numpy(keep.astype(np.int64)))
        next_window += int(keep.size)
        bank_streams[ref.key] = int(keep.size)
        bank_datasets.add(ref.dataset)
        print(f"[memory]   {ref.key}: {keep.size} windows, {len(set(subj_arr))} subjects", flush=True)

    Z = torch.cat(Z_parts)
    y = torch.cat(y_parts)
    subj = torch.cat(subj_parts)
    cfg = torch.cat(cfg_parts)
    event = torch.cat(event_parts)
    event_verified = torch.cat(event_verified_parts)
    source_row = torch.cat(source_row_parts)
    patch = {
        "Z": torch.cat(patch_Z_parts),
        "y": torch.cat(patch_y_parts),
        "subj": torch.cat(patch_subj_parts),
        "cfg": torch.cat(patch_cfg_parts),
        # Each row currently comes from one deployment stream, so sensor-group identity equals cfg.
        # A separate field keeps the structural contract explicit for future multi-sensor streams.
        "sensor": torch.cat(patch_sensor_parts),
        "window": torch.cat(patch_window_parts),
        "event": torch.cat(patch_event_parts),
        "event_verified": torch.cat(patch_event_verified_parts),
        "time": torch.cat(patch_time_parts),
        "duration": torch.cat(patch_duration_parts),
        "resolution": torch.cat(patch_resolution_parts),
        "ordinal": torch.cat(patch_ordinal_parts),
    }
    sensor_table = None
    if sensor_rows:
        sensor_table = {
            "Z": torch.cat(sensor_Z_parts),
            "window": torch.cat(sensor_window_parts),
            "slot": torch.cat(sensor_slot_parts),
            "time": torch.cat(sensor_time_parts),
            "duration": torch.cat(sensor_duration_parts),
            "resolution": torch.cat(sensor_resolution_parts),
            "y": torch.cat(sensor_y_parts),
            "subj": torch.cat(sensor_subj_parts),
            "cfg": torch.cat(sensor_cfg_parts),
            "event": torch.cat(sensor_event_parts),
            "modality": torch.cat(sensor_modality_parts),
            "gravity": torch.cat(sensor_gravity_parts),
            "bias": torch.cat(sensor_bias_parts),
        }

    # One global budget, distributed label -> configuration. This replaces independently tunable
    # stream/label caps whose interactions made the effective memory population hard to explain.
    archive_was_applied = len(y) > args.archive_budget_windows
    if archive_was_applied:
        keep_mask = archive_budget_balanced_keep(y, cfg, args.archive_budget_windows)
        before = len(y)
        remap = torch.full((before,), -1, dtype=torch.long)
        remap[keep_mask] = torch.arange(int(keep_mask.sum()), dtype=torch.long)
        keep_patch = keep_mask[patch["window"]]
        patch = {name: values[keep_patch] for name, values in patch.items()}
        patch["window"] = remap[patch["window"]]
        if sensor_table is not None:
            keep_sensor = keep_mask[sensor_table["window"]]
            sensor_table = {name: values[keep_sensor] for name, values in sensor_table.items()}
            sensor_table["window"] = remap[sensor_table["window"]]
        Z, y, subj, cfg, event, event_verified, source_row = (
            Z[keep_mask], y[keep_mask], subj[keep_mask], cfg[keep_mask], event[keep_mask],
            event_verified[keep_mask], source_row[keep_mask],
        )
        print(f"[memory] balanced archive budget {args.archive_budget_windows}: "
              f"{before} -> {len(y)} windows", flush=True)

    sbert = get_sbert_encoder()
    label_text = torch.from_numpy(sbert(vocab).astype(np.float32))   # (L, 384) L2-normalized

    n_per_label = np.bincount(y.numpy(), minlength=len(vocab))
    print(f"[memory] bank: {Z.shape[0]} windows · d={Z.shape[1]} · {len(subj_names)} subjects · "
          f"{len(cfg_names)} configs · {int((n_per_label > 0).sum())}/{len(vocab)} labels present",
          flush=True)
    print(f"[memory] size: Z={Z.numel() * 2 / 1e6:.0f} MB (fp16)", flush=True)
    print(f"[memory] patch table: {len(patch['Z'])} valid patches · "
          f"{patch['Z'].numel() * 2 / 1e6:.0f} MB (fp16)", flush=True)

    from training.evidence.bank_guard import (
        bank_fingerprint, text_embedding_probe, vocab_fingerprint,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    population_fields = {
        "y": y, "subj": subj, "cfg": cfg, "event": event,
        "event_verified": event_verified, "source_row": source_row,
        **{f"patch_{name}": value for name, value in patch.items() if name != "Z"},
    }
    if sensor_table is not None:
        population_fields.update({
            f"sensor_{name}": value for name, value in sensor_table.items() if name != "Z"
        })
    population_fp = _population_fingerprint(population_fields)
    represented_refs = [ref for ref in refs if ref.key in bank_streams]
    # Per-sensor table (design of record). Absent unless --sensor-rows, so every existing consumer
    # sees exactly the schema-3 contract it already reads.
    if sensor_table is not None:
        widths = {k: len(v) for k, v in sensor_table.items()}
        if len(set(widths.values())) != 1:
            raise ValueError(f"per-sensor table columns disagree in length: {widths}")
        print(f"[memory] sensor table: {len(sensor_table['Z'])} rows "
              f"({len(sensor_table['Z']) / max(len(patch['Z']), 1):.2f}x the patch table) · "
              f"{sensor_table['Z'].numel() * 2 / 1e6:.0f} MB (fp16)", flush=True)

    payload = {
        "schema_version": 5 if sensor_rows else 3,
        "Z": Z, "y": y, "subj": subj, "cfg": cfg, "event": event,
        "event_verified": event_verified, "source_row": source_row,
        # Versioned patch evidence. The legacy pooled keys above intentionally remain unchanged so
        # official pooled controls and old adapters keep their exact data contract.
        "patch": patch,
        **({"sensor": sensor_table} if sensor_table is not None else {}),
        **({"sensor_descriptions": sensor_descriptions} if sensor_table is not None else {}),
        "vocab": vocab, "vocab_fp": vocab_fingerprint(vocab), "label_text": label_text,
        "text_encoder": {
            "model": "all-MiniLM-L6-v2",
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
        },
        "text_embed_probe": text_embedding_probe(),
        "subj_names": {v: k for k, v in subj_names.items()},
        "cfg_names": {v: k for k, v in cfg_names.items()},
        "cfg_rate_hz": cfg_rates,
        "event_names": {v: k for k, v in event_names.items()},
        "d_model": d_model,
        "backbone": {"checkpoint": str(args.checkpoint), "step": int(ckpt["step"]),
                     "val_ba": float(selection_value), "git": git_revision,
                     "selection_metric": selection_metric,
                     "selection_value": float(selection_value),
                     "fingerprint": _backbone_fp(args.checkpoint),
                     "frontend": ckpt["config"].get("frontend"),
                     "multiresolution": ckpt["config"].get("multiresolution")},
        "archive_budget_windows": int(args.archive_budget_windows),
        "source_scan_cap_per_stream": scan_cap_per_stream,
        "source_alignment": "native",
        "population_fp": population_fp,
        "balance_policy": {
            "archive": "global_label_config_balanced_v1",
            "archive_applied": archive_was_applied,
        },
        # Bind the bank to the corpus it was built from AND to the Phase-A corpus the encoder was
        # trained on (F3): a matching vocabulary + backbone is not enough — a bank encoded over a
        # different dataset roster / cap than the encoder saw is unattributable.
        "corpus": {
            "datasets": sorted(bank_datasets),
            "n_streams": len(bank_streams),
            "streams": bank_streams,                       # ref.key -> encoded window count
            "n_encoded_windows": int(sum(bank_streams.values())),
            "phase_a_corpus_fp": ckpt.get("corpus_fingerprint"),   # None for pre-fingerprint ckpts
            "phase_a_corpus": ckpt.get("corpus"),
            "source_grid_fp": grid_corpus_fingerprint("native", represented_refs),
        },
        # BEHAVIOURAL fingerprint of the embedding path that produced Z. Corpus/weight/roster
        # fingerprints all stay identical when the encode CODE changes (e.g. the F1 pooling fix),
        # so without this a stale bank passes every guard while storing vectors from a different
        # function. Consumers re-run the probe and compare (see bank_guard).
        "embed_probe": embedding_fingerprint(enc, device),
        "patch_embed_probe": patch_embedding_fingerprint(enc, device),
        **({"sensor_embed_probe": sensor_embedding_fingerprint(enc, device)} if sensor_rows else {}),
    }
    payload["bank_fp"] = bank_fingerprint(payload)
    torch.save(payload, str(args.out))
    print(f"-> {args.out}", flush=True)
    gate_state = ckpt.get("admissibility_gate")
    if gate_state is not None:
        from training.evidence.admissibility_gate import AdmissibilityGate
        from training.evidence.gate_predictor import (
            NATIVE_JOINT_EPISODIC_REGIME,
            save_gate,
        )

        gate = AdmissibilityGate(
            rank=int(ckpt.get("gate_rank") or ckpt["config"].get("gate_rank", 8)),
            text_dim=int(gate_state["sensor_proj.weight"].shape[1]),
        )
        for name in ("known_sensors", "known_concepts"):
            known = gate_state.get(name)
            if known is not None:
                setattr(gate, name, torch.zeros_like(known))
        gate.load_state_dict(gate_state)
        gate_path = args.gate_out or args.out.with_name("admissibility_gate.pt")
        save_gate(
            gate,
            gate_path,
            provenance={
                "resolvability_checkpoint_sha256": payload["backbone"]["fingerprint"],
                "joint_checkpoint": str(args.checkpoint),
                "joint_checkpoint_step": int(ckpt["step"]),
                "selection_metric": selection_metric,
                "selection_value": float(selection_value),
                "training_datasets": list(roster),
                "phase_a_training_datasets": list(phase_a_roster),
            },
            bank=payload,
            training_regime=NATIVE_JOINT_EPISODIC_REGIME,
        )
        print(f"-> {gate_path} (joint gate bound to this bank)", flush=True)


if __name__ == "__main__":
    main()
