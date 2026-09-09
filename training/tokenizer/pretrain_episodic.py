"""Train the compact HALO evidence engine on independent adaptation episodes.

Each episode has its own candidate set, query executions, enrolled support, and fixed-size
stratified memory bank.  Several episodes share one encoder forward for throughput, but never share
candidate identities or evidence scores.  The exact compact deployment path is optimized end to end:

    temporal sensor encoder -> one pooled row per six-second window
                            -> contextual scalar reranking -> corrected nearest neighbour

The default input is clean.  Signal augmentation is an explicit experiment, not hidden in the
reference recipe.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import pickle
import time
from array import array
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.scripts.augmentations import AugmentationConfig
from model.blocks import AttentionSpec
from model.evidence.engine import PHASE_B_VERSION, EngineConfig, EvidenceEngine
from model.evidence.evidence_reranker import EvidenceRerankerConfig
from model.tokenizer.encoder import ROPE_MIN_PERIOD_S, SetTokenizerEncoder
from training.evidence.episode_labels import encode_neutral_aliases, episode_label_set
from training.tokenizer.episodic import (
    BankSpec,
    EpisodePlan,
    EpisodeSpec,
    EpisodicCollate,
    GroupedEpisodicBatchSampler,
    bank_index,
    build_episode_plans,
    build_shared_stream_plans,
    eligible_labels,
    episode_binding,
    episode_row_roles,
    label_window_table,
    LiveRows,
    live_recording_rows,
    macro_f1,
    matched_support_variants,
    retrieval_alignment_loss,
    provenance_lift,
    sample_bank_positions,
    select_rows,
    stream_label_table,
)
from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain import (
    capture_source_provenance,
    corpus_fingerprint,
    representation_health,
)
from training.tokenizer.pretrain_data import (
    DFT_SIZE,
    PATCH_SECONDS,
    TRAIN_DATASETS,
    CorpusIndex,
    MultiScaleCollate,
    PretrainDataset,
    _seed_worker,
    _stream_gravity_state,
)

SEED = 20260818
#: The validation draw is deliberately NOT tied to the training seed. Measured 2026-08-20: four
#: replicates of one configuration differing only in seed scored 0.4499 / 0.5881 / 0.4545 / 0.4519,
#: between-run sd 0.068 -- and the step-0 baseline alone had sd 0.073, so essentially all of it was
#: WHICH CONCEPTS the seed happened to hold out (16 to 23 of them across those runs). Pinning the
#: draw makes two runs comparable; leaving it tied to --seed made every arm comparison read noise.
VAL_SEED = 20260901
EPISODE_PLAN_CACHE_SCHEMA = 1
EPISODE_PLAN_CHUNK_SIZE = 512


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_plan_cache_dir() -> Path:
    """Machine-local cache root; override without adding a routine training knob."""
    configured = os.environ.get("HALO_EPISODE_PLAN_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "halo" / "episode_plans"


def _array_sha256(values: np.ndarray) -> str:
    """Content identity for an array used by deterministic episode sampling."""
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _episode_planner_source_sha256() -> str:
    """Invalidate cached plans whenever their construction logic changes."""
    objects = (
        EpisodeSpec,
        EpisodePlan,
        build_episode_plans,
        build_shared_stream_plans,
        _make_plans,
        _make_plan_chunk,
        _build_training_core_plans,
        grouped_candidate_schedule,
        alias_curriculum,
        support_count_grid,
    )
    source = "\n".join(inspect.getsource(value) for value in objects)
    return hashlib.sha256(source.encode()).hexdigest()


def _episode_plan_cache_identity(
    *,
    keys,
    subjects: np.ndarray,
    streams: np.ndarray,
    executions: np.ndarray,
    pool: Sequence[int],
    spec: EpisodeSpec,
    n_episodes: int,
    seed: int,
    support_schedule: Sequence[int] | None,
    alias_schedule: np.ndarray | None,
    candidate_schedule: Sequence[int] | None,
) -> tuple[str, dict[str, object]]:
    """Return a complete, inspectable identity for a deterministic core-plan build."""
    key_rows = np.fromiter(
        (value for key in keys for value in (key.stream_i, key.window_i, key.label_id)),
        dtype=np.int64,
        count=len(keys) * 3,
    ).reshape(len(keys), 3)
    manifest: dict[str, object] = {
        "schema": EPISODE_PLAN_CACHE_SCHEMA,
        "planner_source_sha256": _episode_planner_source_sha256(),
        "numpy_version": np.__version__,
        "keys_sha256": _array_sha256(key_rows),
        "subjects_sha256": _array_sha256(subjects),
        "streams_sha256": _array_sha256(streams),
        "executions_sha256": _array_sha256(executions),
        "pool": [int(value) for value in pool],
        "spec": dataclasses.asdict(spec),
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "support_schedule": (
            None if support_schedule is None else [int(value) for value in support_schedule]
        ),
        "alias_schedule_sha256": (
            None if alias_schedule is None else _array_sha256(alias_schedule)
        ),
        "candidate_schedule_sha256": (
            None if candidate_schedule is None
            else _array_sha256(np.asarray(candidate_schedule, dtype=np.int64))
        ),
    }
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()[:24], manifest


def _load_or_build_episode_plans(
    cache_dir: Path | None,
    cache_key: str,
    manifest: dict[str, object],
    n_expected: int,
    builder: Callable[[], list[EpisodePlan]],
) -> tuple[list[EpisodePlan], bool, Path | None]:
    """Load deterministic core plans, rebuilding atomically after a miss or corrupt entry."""
    cache_path = None if cache_dir is None else cache_dir / f"{cache_key}.pkl"
    if cache_path is not None and cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            plans = payload["plans"]
            if payload.get("manifest") != manifest:
                raise ValueError("manifest mismatch")
            if len(plans) != n_expected or not all(isinstance(plan, EpisodePlan) for plan in plans):
                raise ValueError("plan count or type mismatch")
            return plans, True, cache_path
        except (EOFError, OSError, pickle.PickleError, TypeError, ValueError, KeyError) as exc:
            print(f"[phase-b] ignoring invalid episode-plan cache {cache_path}: {exc}", flush=True)

    plans = builder()
    if len(plans) != n_expected:
        raise RuntimeError(f"episode planner returned {len(plans)} plans, expected {n_expected}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                pickle.dump(
                    {"manifest": manifest, "plans": plans},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    return plans, False, cache_path


def subject_ids_for(index: CorpusIndex, keys) -> np.ndarray:
    """Global subject id per key position."""
    seen: dict[tuple[str, str], int] = {}
    values = []
    for key in keys:
        ref = index.refs[key.stream_i]
        identity = (ref.dataset, str(ref.subjects[key.window_i]))
        values.append(seen.setdefault(identity, len(seen)))
    return np.asarray(values, dtype=np.int64)


def encoder_comparison_keys(index: CorpusIndex, keys) -> tuple[list, set[str]]:
    """Keep only complete, gravity-present accelerometer windows for the matched three-arm corpus."""
    kept = []
    excluded: set[str] = set()
    for key in keys:
        ref = index.refs[key.stream_i]
        compatible = bool(np.asarray(ref.mask, dtype=bool)[:3].all()) and (
            _stream_gravity_state(ref.dataset, ref.stream) == "present"
        )
        if compatible:
            kept.append(key)
        else:
            excluded.add(ref.key)
    if not kept:
        raise ValueError("the encoder-comparison compatibility filter removed every window")
    return kept, excluded


def stream_ids_for(index: CorpusIndex, keys) -> tuple[np.ndarray, dict[str, int]]:
    """Acquisition stream id per key position."""
    names: dict[str, int] = {}
    values = [names.setdefault(index.refs[key.stream_i].key, len(names)) for key in keys]
    return np.asarray(values, dtype=np.int64), names


def execution_ids_for(index: CorpusIndex, keys) -> np.ndarray:
    """Physical execution id; synchronous streams from one event share an id."""
    names: dict[tuple[str, str, str], int] = {}
    values = []
    for key in keys:
        ref = index.refs[key.stream_i]
        identity = (
            ref.dataset,
            str(ref.subjects[key.window_i]),
            str(ref.event_ids[key.window_i]),
        )
        values.append(names.setdefault(identity, len(names)))
    return np.asarray(values, dtype=np.int64)


def label_text_matrix(encoder: SetTokenizerEncoder, label_ids: dict, device) -> torch.Tensor:
    """Frozen MiniLM embedding for every canonical corpus label."""
    from eval.scoring import get_sbert_encoder

    id_to_label = {value: key for key, value in label_ids.items()}
    labels = [id_to_label[i] for i in range(len(id_to_label))]
    return torch.from_numpy(get_sbert_encoder()(labels)).to(device)


def split_label_pool(pool: list[int], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic, disjoint objective-train and validation concept pools."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout fraction must be in (0,1)")
    values = np.asarray(sorted(pool), dtype=np.int64)
    if len(values) < 4:
        raise ValueError("at least four eligible labels are needed for a concept holdout")
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    n_val = min(max(2, int(round(len(values) * fraction))), len(values) - 2)
    return sorted(values[n_val:].tolist()), sorted(values[:n_val].tolist())


def replace_counts(spec: EpisodeSpec, pool_size: int) -> EpisodeSpec:
    """Drop candidate counts that cannot fit while preserving every other episode field."""
    ceiling = pool_size - (1 if spec.background_windows else 0)
    counts = tuple(count for count in spec.candidate_counts if count <= ceiling)
    if not counts:
        raise ValueError(
            f"no candidate count {spec.candidate_counts} fits pool size {pool_size} "
            f"with ceiling {ceiling}"
        )
    return dataclasses.replace(spec, candidate_counts=counts)


def encode_batch(encoder: SetTokenizerEncoder, batch: dict, device: torch.device) -> dict:
    """Encode every window from several independent episodes in one heterogeneous forward."""
    if encoder.trunk != "temporal" or encoder.token_granularity != "sensor":
        raise ValueError("compact Phase B requires a temporal, sensor-granularity encoder")
    metadata = {}
    if getattr(encoder, "requires_stream_metadata", False):
        metadata = {
            "streams": batch.get("streams"),
            "sources": batch.get("sources"),
            "gravity_state": batch.get("gravity_state"),
        }
    encoded = encoder(
        batch["patches"].to(device, non_blocking=True),
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
        source_rate_hz=batch["source_rates"].to(device, non_blocking=True),
        return_retrieval_tokens=True,
        **metadata,
    )
    return encoded


_DEVICE_BATCH_FIELDS = (
    "patches", "rates", "patch_len", "positions", "patch_durations", "resolution_ids", "channel_mask",
    "patch_padding_mask", "sensor_id", "source_rates", "labels", "sensor_bias",
)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    """Issue one nonblocking transfer per tensor and reuse those tensors throughout the step."""
    if device.type == "cpu":
        return batch
    moved = dict(batch)
    for name in _DEVICE_BATCH_FIELDS:
        value = moved.get(name)
        if torch.is_tensor(value):
            moved[name] = value.to(device, non_blocking=True)
    return moved


def _offset_plan(plan: EpisodePlan, offset: int) -> EpisodePlan:
    """Move query/support positions into a combined train+validation dataset."""
    return dataclasses.replace(
        plan,
        query_positions=tuple(value + offset for value in plan.query_positions),
        support_positions=tuple(value + offset for value in plan.support_positions),
    )


_BANK_BUILD_STATE = None


def _sample_training_bank(job: tuple[int, int]) -> array:
    """Process worker for deterministic, compact training-bank construction."""
    index, seed = job
    plans, bank, bank_spec = _BANK_BUILD_STATE
    plan = plans[index]
    remaining = bank_spec.n_windows - plan.n_support
    if remaining < 0:
        raise ValueError(
            f"episode support uses {plan.n_support} windows but the bank holds only "
            f"{bank_spec.n_windows}"
        )
    positions = sample_bank_positions(
        bank, spec=bank_spec, n_windows=remaining,
        exclude_labels=plan.candidates, rng=np.random.default_rng(seed + index),
    )
    if positions and max(positions) >= 2 ** 32:
        raise ValueError("corpus position exceeds the compact uint32 episode-plan format")
    return array("I", positions)


def _attach_training_banks(
    plans: list[EpisodePlan], bank, bank_spec: BankSpec, seed: int,
) -> list[EpisodePlan]:
    """Attach stratified banks, parallelizing only the large production build.

    Fork workers share the read-only corpus tables and return compact uint32 arrays. Every episode
    has its own seed, so results are deterministic and independent of worker scheduling. Small tests
    and smoke runs stay serial, and platforms without fork retain the reference path.
    """
    global _BANK_BUILD_STATE
    _BANK_BUILD_STATE = (plans, bank, bank_spec)
    workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    use_pool = len(plans) >= 512 and workers > 1 and "fork" in mp.get_all_start_methods()
    jobs = [(index, seed) for index in range(len(plans))]
    try:
        if use_pool:
            with mp.get_context("fork").Pool(workers) as pool:
                backgrounds = pool.map(_sample_training_bank, jobs, chunksize=32)
        else:
            backgrounds = [_sample_training_bank(job) for job in jobs]
    finally:
        _BANK_BUILD_STATE = None
    return [
        dataclasses.replace(plan, background_positions=background)
        for plan, background in zip(plans, backgrounds)
    ]


def _matched_validation_plans(
    base_plans: list[EpisodePlan], support_grid: tuple[int, ...], bank,
    bank_spec: BankSpec, train_offset: int, seed: int,
) -> list[EpisodePlan]:
    """Build matched k-curves: same query, candidates, and nested background bank at every k."""
    rng = np.random.default_rng(seed)
    out = []
    for raw_base in base_plans:
        base = _offset_plan(raw_base, train_offset)
        full_background = sample_bank_positions(
            bank, spec=bank_spec, n_windows=bank_spec.n_windows,
            exclude_labels=base.candidates, rng=rng,
        )
        for variant in matched_support_variants(base, support_grid):
            keep = bank_spec.n_windows - variant.n_support
            out.append(dataclasses.replace(
                variant, background_positions=array("I", full_background[:keep]),
            ))
    return out


def alias_curriculum(
    n_episodes: int, episodes_per_step: int, target: float, warmup_steps: int, ramp_steps: int,
) -> np.ndarray | None:
    """Per-episode alias fraction: 0 for `warmup_steps`, then linear to `target` over `ramp_steps`.

    Random-alias episodes are a CAPABILITY TARGET (physiotherapy-style labels such as "exercise 1",
    where the name carries no usable linguistics), not an augmentation. Presenting them from step 0
    at full strength asks the model to learn evidence-only classification before it can classify at
    all. The curriculum builds the coherent base first and introduces the harder regime once there
    is something to make harder. `warmup_steps=0 and ramp_steps=0` reproduces the flat schedule
    exactly, so the default path is unchanged.
    """
    if warmup_steps <= 0 and ramp_steps <= 0:
        return None
    step = np.arange(n_episodes) // max(episodes_per_step, 1)
    if ramp_steps > 0:
        fraction = np.clip((step - warmup_steps) / float(ramp_steps), 0.0, 1.0)
    else:
        fraction = (step >= warmup_steps).astype(np.float64)
    return (fraction * float(target)).astype(np.float64)


def support_count_grid(max_support: int, regime: str = "unified") -> tuple[int, ...]:
    """Support schedule for a unified, zero-shot-only, or enrollment-only Phase-B head."""
    if regime not in {"unified", "zero-shot", "enrollment"}:
        raise ValueError(f"unknown Phase-B regime: {regime}")
    if regime == "zero-shot":
        return (0,)
    if regime == "enrollment" and max_support < 1:
        raise ValueError("enrollment-only training requires max_support >= 1")
    canonical = (0, 1, 2, 4, 8, 16)
    values = [value for value in canonical if value <= max_support]
    if max_support not in values:
        values.append(max_support)
    result = tuple(sorted(set(values)))
    return tuple(value for value in result if value > 0) if regime == "enrollment" else result


def episode_distribution(plans: list[EpisodePlan]) -> dict[str, object]:
    """Compact, persisted audit of the actual task shapes sampled for a run."""
    if not plans:
        raise ValueError("cannot summarize an empty episode plan")

    def counts(values) -> dict[str, int]:
        return {str(key): int(value) for key, value in sorted(Counter(values).items())}

    return {
        "episodes": len(plans),
        "candidate_count": counts(len(plan.candidates) for plan in plans),
        "support_k": counts(plan.support_k for plan in plans),
        "query_label_count": counts(len(set(plan.query_slot)) for plan in plans),
        "query_windows": counts(plan.n_query for plan in plans),
        "support_windows_mean": float(np.mean([plan.n_support for plan in plans])),
        "support_windows_max": max(plan.n_support for plan in plans),
        "enrolled_candidates_mean": float(np.mean([
            len(plan.enrolled_slots) for plan in plans
        ])),
        "enrolled_query_labels_mean": float(np.mean([
            len(set(plan.query_slot) & set(plan.enrolled_slots)) for plan in plans
        ])),
        "enrolled_distractors_mean": float(np.mean([
            len(set(plan.enrolled_slots) - set(plan.query_slot)) for plan in plans
        ])),
    }


def _make_plans(
    table, stream_table, pool, spec, streams, executions, *, n_episodes, seed, schedule=None,
    alias_schedule=None, candidate_schedule=None,
) -> list[EpisodePlan]:
    if spec.shared_query_stream:
        return build_shared_stream_plans(
            table, stream_table, pool, n_episodes=n_episodes, spec=spec, seed=seed,
            stream_ids=streams, support_schedule=schedule, execution_ids=executions,
            alias_schedule=alias_schedule, candidate_schedule=candidate_schedule,
        )
    return build_episode_plans(
        table, pool, n_episodes=n_episodes, spec=spec, seed=seed,
        support_schedule=schedule, stream_ids=streams, execution_ids=executions,
        alias_schedule=alias_schedule, candidate_schedule=candidate_schedule,
    )


_PLAN_BUILD_STATE = None


def _make_plan_chunk(bounds: tuple[int, int]) -> list[EpisodePlan]:
    """Build one deterministic core-plan chunk in a fork worker or the parent process."""
    start, end = bounds
    (table, stream_table, pool, spec, streams, executions, seed, support_schedule,
     alias_schedule, candidate_schedule) = _PLAN_BUILD_STATE
    local_support = (
        None if support_schedule is None
        else tuple(support_schedule[index % len(support_schedule)] for index in range(start, end))
    )
    local_alias = None if alias_schedule is None else alias_schedule[start:end]
    local_candidates = (
        None if candidate_schedule is None else candidate_schedule[start:end]
    )
    return _make_plans(
        table, stream_table, pool, spec, streams, executions,
        n_episodes=end - start,
        # Fixed chunk boundaries and seeds make output independent of process count and scheduling.
        seed=seed + start,
        schedule=local_support,
        alias_schedule=local_alias,
        candidate_schedule=local_candidates,
    )


def _build_training_core_plans(
    table, stream_table, pool, spec, streams, executions, *, n_episodes, seed, schedule=None,
    alias_schedule=None, candidate_schedule=None,
) -> list[EpisodePlan]:
    """Build independent chunks in parallel without changing episode ordering or distributions."""
    global _PLAN_BUILD_STATE
    _PLAN_BUILD_STATE = (
        table, stream_table, pool, spec, streams, executions, seed, schedule,
        alias_schedule, candidate_schedule,
    )
    jobs = [
        (start, min(start + EPISODE_PLAN_CHUNK_SIZE, n_episodes))
        for start in range(0, n_episodes, EPISODE_PLAN_CHUNK_SIZE)
    ]
    workers = min(8, max(1, (os.cpu_count() or 2) // 2), len(jobs))
    use_pool = n_episodes >= EPISODE_PLAN_CHUNK_SIZE and workers > 1 \
        and "fork" in mp.get_all_start_methods()
    try:
        if use_pool:
            print(
                f"[phase-b] cold plan build: {workers} processes, "
                f"{EPISODE_PLAN_CHUNK_SIZE}-episode deterministic chunks",
                flush=True,
            )
            with mp.get_context("fork").Pool(workers) as pool_handle:
                chunks = pool_handle.map(_make_plan_chunk, jobs, chunksize=1)
        else:
            chunks = [_make_plan_chunk(job) for job in jobs]
    finally:
        _PLAN_BUILD_STATE = None
    return [plan for chunk in chunks for plan in chunk]


def grouped_candidate_schedule(
    steps: int, episodes_per_step: int, candidate_counts: tuple[int, ...], seed: int,
) -> np.ndarray:
    """Balanced random step order with one tensor shape shared by every episode in a step."""
    if steps < 1 or episodes_per_step < 1 or not candidate_counts:
        raise ValueError("steps, episodes_per_step, and candidate_counts must be nonempty")
    step_counts = np.resize(np.asarray(candidate_counts, dtype=np.int64), steps)
    np.random.default_rng(seed).shuffle(step_counts)
    return np.repeat(step_counts, episodes_per_step)


def _grad_norm(parameters) -> float:
    gradients = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(gradients))) if gradients else 0.0


def prepare_live_batch(
    encoded: dict, batch: dict, plans: list[EpisodePlan], offsets, device: torch.device,
) -> tuple[LiveRows, torch.Tensor]:
    """Flatten a shared encoder output once and attach every episode's support bindings.

    Grouped episodes occupy disjoint contiguous ranges in the DataLoader batch. Re-flattening the
    same encoded tensor separately for query and memory in every episode repeated modality checks,
    descriptor normalization, nonzero scans, and small host-to-device copies eight times per default
    optimizer step. One canonical row table is both cheaper and less likely to let role-specific
    plumbing drift.
    """
    labels = batch["labels"].to(device, non_blocking=True)
    binding_values: list[int] = []
    expected_size = 0
    for plan, offset in zip(plans, offsets):
        if int(offset) != expected_size:
            raise RuntimeError("grouped episode offsets are not contiguous")
        binding_values.extend([-1] * plan.n_query)
        binding_values.extend(int(slot) for slot in plan.support_slot)
        binding_values.extend([-1] * plan.n_background)
        expected_size += len(plan.flat_positions())
    if expected_size != len(labels):
        raise RuntimeError(
            f"episode plans cover {expected_size} windows but the batch contains {len(labels)}"
        )
    binding = torch.tensor(binding_values, dtype=torch.long, device=device)
    return live_recording_rows(
        encoded, batch, labels=labels, enrolled_candidate=binding,
    ), labels


def _live_episode_rows(
    live: LiveRows, plan: EpisodePlan, row_offset: int,
) -> tuple[LiveRows, LiveRows]:
    """Slice one independent episode from the shared encoder-row table."""
    window = live.window
    query_end = row_offset + plan.n_query
    episode_end = row_offset + len(plan.flat_positions())
    query_rows = torch.nonzero(
        window.ge(row_offset) & window.lt(query_end), as_tuple=True,
    )[0]
    memory_rows = torch.nonzero(
        window.ge(query_end) & window.lt(episode_end), as_tuple=True,
    )[0]
    return select_rows(live, query_rows), select_rows(live, memory_rows)


def _finish_episode(
    result: dict,
    plan: EpisodePlan,
    query: LiveRows,
    memory: LiveRows,
    query_i: torch.Tensor,
    label_text: torch.Tensor,
    *,
    aux_weight: float,
    aux_temperature: float,
    no_regression_weight: float,
    reference_mode: str,
) -> dict:
    """Apply the common objective and telemetry contract to one engine result."""
    expected_windows = query_i.to(result["query_window"].device)
    if not torch.equal(result["query_window"], expected_windows):
        raise RuntimeError(
            "recording-level engine output and episode query order disagree; predictions would be "
            "assigned to the wrong windows"
        )
    logits = result["logits"]
    base_logits = result["base_logits"]
    target = torch.tensor(plan.query_slot, dtype=torch.long, device=logits.device)
    task_per_query = F.cross_entropy(logits, target, reduction="none")
    base_per_query = F.cross_entropy(base_logits.detach(), target, reduction="none")
    one_nn_logits = result["enrolled_1nn_logits"].detach()
    one_nn_available = result["enrolled_1nn_available"]
    one_nn_per_query = F.cross_entropy(one_nn_logits, target, reduction="none")
    one_nn_valid = (
        one_nn_available.gather(1, target.unsqueeze(1)).squeeze(1)
        & one_nn_available.sum(dim=1).ge(2)
    )
    if reference_mode == "base_nearest":
        reference_per_query = base_per_query
    elif reference_mode == "enrolled_1nn":
        reference_per_query = torch.where(one_nn_valid, one_nn_per_query, base_per_query)
    else:
        raise ValueError(f"unknown no-regression reference: {reference_mode}")
    no_regression = F.relu(task_per_query - reference_per_query).mean()
    loss = task_per_query.mean() + float(no_regression_weight) * no_regression
    aux = loss.new_zeros(())
    if aux_weight:
        aux = retrieval_alignment_loss(
            result["scores"], query.rows.label, memory.rows.label, label_text,
            temperature=aux_temperature,
        )
        loss = loss + aux_weight * aux
    prediction = logits.argmax(dim=1)
    base_prediction = base_logits.argmax(dim=1)
    one_nn_prediction = one_nn_logits.argmax(dim=1)
    out = {
        "loss": loss,
        "task_loss": float(task_per_query.mean().detach()),
        "base_loss": float(base_per_query.mean().detach()),
        "reference_loss": float(reference_per_query.mean().detach()),
        "enrolled_1nn_loss": (
            float(one_nn_per_query[one_nn_valid].mean().detach())
            if bool(one_nn_valid.any()) else 0.0
        ),
        "no_regression_loss": float(no_regression.detach()),
        "aux_loss": float(aux.detach()),
        "truth": target.detach().cpu().tolist(),
        "prediction": prediction.detach().cpu().tolist(),
        "base_prediction": base_prediction.detach().cpu().tolist(),
        "enrolled_1nn_prediction": one_nn_prediction.detach().cpu().tolist(),
        "enrolled_1nn_valid": one_nn_valid.detach().cpu().tolist(),
        "truth_label": [plan.candidates[int(slot)] for slot in target.detach().cpu()],
        "prediction_label": [plan.candidates[int(slot)] for slot in prediction.detach().cpu()],
        "support_k": plan.support_k,
        "label_mode": plan.label_mode,
        "memory_rows": len(memory.rows.feature),
    }
    if "stats" in result:
        stats = dict(result["stats"])
        selected_bound = memory.rows.enrolled_candidate[result["selected"]]
        stats["retrieval/true_support_selected_share"] = float(
            selected_bound.eq(target.unsqueeze(1)).float().mean()
        )
        stats["episode/accuracy"] = float(prediction.eq(target).float().mean())
        stats["episode/base_accuracy"] = float(base_prediction.eq(target).float().mean())
        stats["episode/enrolled_1nn_coverage"] = float(one_nn_valid.float().mean())
        stats["episode/enrolled_1nn_accuracy"] = (
            float(one_nn_prediction[one_nn_valid].eq(target[one_nn_valid]).float().mean())
            if bool(one_nn_valid.any()) else 0.0
        )
        stats["episode/changed_to_correct"] = float(
            ((prediction == target) & (base_prediction != target)).float().mean()
        )
        stats["episode/changed_to_wrong"] = float(
            ((prediction != target) & (base_prediction == target)).float().mean()
        )
        stats["episode/prediction_changed"] = float(
            prediction.ne(base_prediction).float().mean()
        )
        stats["episode/candidate_count"] = float(len(plan.candidates))
        stats["episode/query_label_count"] = float(len(set(plan.query_slot)))
        stats["episode/enrolled_candidate_count"] = float(len(plan.enrolled_slots))
        query_slots = set(plan.query_slot)
        stats["episode/enrolled_query_label_count"] = float(
            len(query_slots & set(plan.enrolled_slots))
        )
        stats["episode/enrolled_distractor_count"] = float(
            len(set(plan.enrolled_slots) - query_slots)
        )
        stats["episode/chance_accuracy"] = 1.0 / len(plan.candidates)
        stats.update(representation_health(query.rows.feature, "query_repr"))
        out["stats"] = stats
    return out


def run_episode(
    engine: EvidenceEngine,
    batch: dict,
    plan: EpisodePlan,
    label_text: torch.Tensor,
    alias_text: torch.Tensor,
    device: torch.device,
    *,
    encoded_out: dict | None = None,
    live_out: LiveRows | None = None,
    labels_out: torch.Tensor | None = None,
    row_offset: int = 0,
    seed: int = 0,
    collect_stats: bool = False,
    aux_weight: float = 0.0,
    aux_temperature: float = 0.1,
    no_regression_weight: float = 0.0,
    reference_mode: str = "base_nearest",
) -> dict:
    """Run one independent episode through the exact compact deployment rule."""
    if engine.encoder is None:
        raise ValueError("training requires an engine with an attached encoder")
    labels = (batch["labels"].to(device, non_blocking=True)
              if labels_out is None else labels_out)
    expected = torch.tensor(plan.expected_labels(), dtype=labels.dtype, device=device)
    actual = labels[row_offset:row_offset + len(expected)]
    if not torch.equal(actual, expected):
        raise RuntimeError("episode plan and DataLoader batch disagree on query/support labels")

    encoded = encode_batch(engine.encoder, batch, device) if encoded_out is None else encoded_out
    query_i, support_i, background_i = episode_row_roles(plan, row_offset=row_offset)
    if live_out is None:
        binding = episode_binding(plan, len(labels), row_offset=row_offset).to(device)
        query = live_recording_rows(
            encoded, batch, labels=labels, enrolled_candidate=binding, select=query_i,
        )
        memory = live_recording_rows(
            encoded, batch, labels=labels, enrolled_candidate=binding,
            select=torch.cat([support_i, background_i]),
        )
    else:
        # Rows are ordered by their source-window index, and each episode is one contiguous batch
        # range. Select row views without rebuilding layout metadata or normalizing descriptors.
        query, memory = _live_episode_rows(live_out, plan, row_offset)
    if len(memory.rows.feature) == 0:
        raise RuntimeError("episode produced no live memory rows")

    candidate_ids = torch.tensor(plan.candidates, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    labels_local = episode_label_set(
        candidate_ids, label_text, mode=plan.label_mode, rng=rng,
        alias_embeddings=alias_text,
    )
    generator = torch.Generator().manual_seed(seed)
    result = engine(
        query.rows, memory.rows, labels_local.embeddings, label_text,
        generator=generator, collect_stats=collect_stats,
    )
    return _finish_episode(
        result, plan, query, memory, query_i, label_text,
        aux_weight=aux_weight, aux_temperature=aux_temperature,
        no_regression_weight=no_regression_weight, reference_mode=reference_mode,
    )


def run_episode_group(
    engine: EvidenceEngine,
    plans: list[EpisodePlan],
    offsets: Sequence[int],
    live: LiveRows,
    labels: torch.Tensor,
    label_text: torch.Tensor,
    alias_text: torch.Tensor,
    *,
    seed: int,
    collect_stats: bool = False,
    aux_weight: float = 0.0,
    aux_temperature: float = 0.1,
    no_regression_weight: float = 0.0,
    reference_mode: str = "base_nearest",
) -> list[dict]:
    """Run one same-C optimizer group through the vectorized active evidence path."""
    if len(plans) != len(offsets) or len({len(plan.candidates) for plan in plans}) != 1:
        raise ValueError("a vectorized optimizer group requires same-C plans and aligned offsets")
    queries, memories, query_indices, candidate_texts, generators = [], [], [], [], []
    for episode, (plan, offset) in enumerate(zip(plans, offsets)):
        expected = torch.tensor(plan.expected_labels(), dtype=labels.dtype, device=labels.device)
        actual = labels[int(offset):int(offset) + len(expected)]
        if not torch.equal(actual, expected):
            raise RuntimeError("episode plan and DataLoader batch disagree on query/support labels")
        query_i, _support_i, _background_i = episode_row_roles(plan, row_offset=int(offset))
        query, memory = _live_episode_rows(live, plan, int(offset))
        if len(memory.rows.feature) == 0:
            raise RuntimeError("episode produced no live memory rows")
        local_seed = seed + episode
        label_set = episode_label_set(
            torch.tensor(plan.candidates, dtype=torch.long, device=labels.device),
            label_text, mode=plan.label_mode, rng=np.random.default_rng(local_seed),
            alias_embeddings=alias_text,
        )
        queries.append(query)
        memories.append(memory)
        query_indices.append(query_i)
        candidate_texts.append(label_set.embeddings)
        generators.append(torch.Generator().manual_seed(local_seed))
    engine_results = engine.forward_many(
        [query.rows for query in queries], [memory.rows for memory in memories],
        torch.stack(candidate_texts), label_text,
        generators=generators, collect_stats=collect_stats,
    )
    return [
        _finish_episode(
            result, plan, query, memory, query_i, label_text,
            aux_weight=aux_weight, aux_temperature=aux_temperature,
            no_regression_weight=no_regression_weight, reference_mode=reference_mode,
        )
        for result, plan, query, memory, query_i
        in zip(engine_results, plans, queries, memories, query_indices)
    ]


def training_group_results(
    engine: EvidenceEngine,
    batch: dict,
    plans: list[EpisodePlan],
    offsets: Sequence[int],
    encoded: dict,
    live: LiveRows,
    labels: torch.Tensor,
    label_text: torch.Tensor,
    alias_text: torch.Tensor,
    *,
    seed: int,
    collect_stats: bool,
    args,
) -> list[dict]:
    """One switch between the optimized active path and its sequential test oracle."""
    common = dict(
        aux_weight=args.retrieval_aux_weight,
        aux_temperature=args.retrieval_aux_temperature,
        no_regression_weight=args.no_regression_weight,
        reference_mode=args.reference_mode,
    )
    if not args.sequential_episodes:
        return run_episode_group(
            engine, plans, offsets, live, labels, label_text, alias_text,
            seed=seed, collect_stats=collect_stats, **common,
        )
    return [
        run_episode(
            engine, batch, plan, label_text, alias_text, labels.device,
            encoded_out=encoded, live_out=live, labels_out=labels,
            row_offset=int(offset), seed=seed + episode,
            collect_stats=collect_stats, **common,
        )
        for episode, (plan, offset) in enumerate(zip(plans, offsets))
    ]


def select_validation_metric(report: dict, *, include_aliases: bool) -> dict:
    """Attach the checkpoint score for the objective the run actually optimizes."""
    if include_aliases:
        report["selection_metric"] = "mean(coherent_curve_mean, alias_positive_curve_mean)"
        report["selection_score"] = 0.5 * (
            report["coherent_mean_macro_f1"] + report["alias_mean_macro_f1"]
        )
    else:
        report["selection_metric"] = "coherent_curve_mean"
        report["selection_score"] = report["coherent_mean_macro_f1"]
    return report


@torch.no_grad()
def validate(
    engine: EvidenceEngine, loader: DataLoader, plans: list[EpisodePlan],
    label_text: torch.Tensor, alias_text: torch.Tensor, device: torch.device, seed: int,
    episodes_per_batch: int, *, select_on_aliases: bool = True,
    reference_mode: str = "base_nearest",
) -> dict:
    engine.eval()
    cells: dict[tuple[str, int], tuple[list[int], list[int]]] = {}
    base_cells: dict[tuple[str, int], tuple[list[int], list[int]]] = {}
    candidate_cells: dict[tuple[str, int, int], tuple[list[int], list[int]]] = {}
    base_candidate_cells: dict[tuple[str, int, int], tuple[list[int], list[int]]] = {}
    one_nn_cells: dict[tuple[str, int], tuple[list[int], list[int]]] = {}
    learned_on_one_nn_cells: dict[tuple[str, int], tuple[list[int], list[int]]] = {}
    losses: list[float] = []
    episode = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        with _autocast(device):
            encoded = encode_batch(engine.encoder, batch, device)
            step_plans = plans[episode:episode + episodes_per_batch]
            offsets = np.cumsum([0] + [len(plan.flat_positions())
                                        for plan in step_plans[:-1]])
            live, labels = prepare_live_batch(encoded, batch, step_plans, offsets, device)
            results = [
                run_episode(
                    engine, batch, plan, label_text, alias_text, device,
                    encoded_out=encoded, live_out=live, labels_out=labels,
                    row_offset=int(offset), seed=seed + episode + local,
                    reference_mode=reference_mode,
                )
                for local, (plan, offset) in enumerate(zip(step_plans, offsets))
            ]
        for plan, result in zip(step_plans, results):
            losses.append(float(result["loss"]))
            bucket = cells.setdefault((plan.label_mode, plan.support_k), ([], []))
            # Candidate slots are randomized episode-local identities. Scoring those slots as if
            # slot 0 meant one stable class across episodes produces a plausible but meaningless F1.
            bucket[0].extend(result["truth_label"])
            bucket[1].extend(result["prediction_label"])
            base_bucket = base_cells.setdefault((plan.label_mode, plan.support_k), ([], []))
            base_bucket[0].extend(result["truth_label"])
            base_bucket[1].extend([
                plan.candidates[int(slot)] for slot in result["base_prediction"]
            ])
            candidate_key = (plan.label_mode, plan.support_k, len(plan.candidates))
            candidate_bucket = candidate_cells.setdefault(candidate_key, ([], []))
            candidate_bucket[0].extend(result["truth_label"])
            candidate_bucket[1].extend(result["prediction_label"])
            base_candidate_bucket = base_candidate_cells.setdefault(candidate_key, ([], []))
            base_candidate_bucket[0].extend(result["truth_label"])
            base_candidate_bucket[1].extend([
                plan.candidates[int(slot)] for slot in result["base_prediction"]
            ])
            one_nn_bucket = one_nn_cells.setdefault((plan.label_mode, plan.support_k), ([], []))
            learned_subset = learned_on_one_nn_cells.setdefault(
                (plan.label_mode, plan.support_k), ([], []),
            )
            for truth, prediction, learned_prediction, valid in zip(
                result["truth_label"], result["enrolled_1nn_prediction"],
                result["prediction_label"],
                result["enrolled_1nn_valid"],
            ):
                if valid:
                    one_nn_bucket[0].append(truth)
                    one_nn_bucket[1].append(plan.candidates[int(prediction)])
                    learned_subset[0].append(truth)
                    learned_subset[1].append(learned_prediction)
        episode += len(step_plans)
    if episode != len(plans):
        raise RuntimeError(f"validation consumed {episode} plans, expected {len(plans)}")
    curve = {
        f"{mode}/k={support}": {
            "macro_f1": macro_f1(truth, prediction),
            "queries": len(truth),
        }
        for (mode, support), (truth, prediction) in sorted(cells.items())
    }
    base_curve = {
        f"{mode}/k={support}": {
            "macro_f1": macro_f1(truth, prediction),
            "queries": len(truth),
        }
        for (mode, support), (truth, prediction) in sorted(base_cells.items())
    }
    one_nn_curve = {
        f"{mode}/k={support}": {
            "macro_f1": macro_f1(truth, prediction),
            "queries": len(truth),
        }
        for (mode, support), (truth, prediction) in sorted(one_nn_cells.items())
        if truth
    }
    learned_on_one_nn_curve = {
        f"{mode}/k={support}": {
            "macro_f1": macro_f1(truth, prediction),
            "queries": len(truth),
        }
        for (mode, support), (truth, prediction) in sorted(learned_on_one_nn_cells.items())
        if truth
    }
    candidate_curve = {
        f"{mode}/C={candidate_count}/k={support}": {
            "macro_f1": macro_f1(truth, prediction),
            "queries": len(truth),
        }
        for (mode, support, candidate_count), (truth, prediction)
        in sorted(candidate_cells.items())
    }
    base_candidate_curve = {
        f"{mode}/C={candidate_count}/k={support}": {
            "macro_f1": macro_f1(truth, prediction),
            "queries": len(truth),
        }
        for (mode, support, candidate_count), (truth, prediction)
        in sorted(base_candidate_cells.items())
    }
    coherent = [row["macro_f1"] for key, row in curve.items() if key.startswith("coherent/")]
    aliases = [row["macro_f1"] for key, row in curve.items() if key.startswith("random_alias/")]
    report = {
        "loss": float(np.mean(losses)),
        "curve": curve,
        "coherent_mean_macro_f1": float(np.mean(coherent)) if coherent else 0.0,
        "alias_mean_macro_f1": float(np.mean(aliases)) if aliases else 0.0,
        "base_curve": base_curve,
        "curve_by_candidate_count": candidate_curve,
        "base_curve_by_candidate_count": base_candidate_curve,
        "enrolled_1nn_curve": one_nn_curve,
        "learned_on_enrolled_1nn_subset_curve": learned_on_one_nn_curve,
        "reference_mode": reference_mode,
    }
    report["learned_minus_base"] = {
        key: curve[key]["macro_f1"] - base_curve[key]["macro_f1"] for key in curve
    }
    coherent_delta = [
        value for key, value in report["learned_minus_base"].items()
        if key.startswith("coherent/")
    ]
    report["coherent_mean_learned_minus_base"] = (
        float(np.mean(coherent_delta)) if coherent_delta else 0.0
    )
    report["learned_minus_enrolled_1nn"] = {
        key: learned_on_one_nn_curve[key]["macro_f1"] - row["macro_f1"]
        for key, row in one_nn_curve.items() if key in learned_on_one_nn_curve
    }
    one_nn_delta = [
        value for key, value in report["learned_minus_enrolled_1nn"].items()
        if key.startswith("coherent/")
    ]
    report["coherent_mean_learned_minus_enrolled_1nn"] = (
        float(np.mean(one_nn_delta)) if one_nn_delta else 0.0
    )
    total_queries = sum(row["queries"] for row in curve.values())
    report["enrolled_1nn_reference_queries"] = sum(
        row["queries"] for row in one_nn_curve.values()
    )
    report["enrolled_1nn_reference_coverage"] = (
        report["enrolled_1nn_reference_queries"] / max(total_queries, 1)
    )
    # Alias behavior influences checkpoint ordering only when alias episodes are present in the
    # objective. Runs with the default coherent-only objective do not spend periodic validation
    # compute on an unsupported task; arbitrary-activity evaluation is a separate sealed protocol.
    select_validation_metric(report, include_aliases=select_on_aliases)
    engine.train()
    if engine.encoder is not None and not any(
        parameter.requires_grad for parameter in engine.encoder.parameters()
    ):
        # A frozen encoder is a deterministic feature extractor. Recursive engine.train() would
        # otherwise reactivate its dropout even though its parameters cannot adapt to that noise.
        engine.encoder.eval()
    return report


def _autocast(device: torch.device):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext())


def profile_training(
    engine, train_loader, train_plans, label_text, alias_text, optimizer, scheduler,
    device, args,
) -> None:
    """Where a training step actually goes: phase timers over real steps, then an op-level table.

    Phase timers synchronise at section boundaries, which inflates the total a little but makes
    the attribution honest. The op table runs unsynchronised so overlap is visible there instead.
    """
    encoder = engine.encoder
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def mark():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    phases = {name: 0.0 for name in
              ("data_wait", "encode", "episodes", "backward", "optimizer")}
    measured = 0
    iterator = iter(train_loader)
    group_cursor = 0

    def next_profile_group():
        nonlocal iterator, group_cursor
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        n_groups = len(train_plans) // args.episodes_per_step
        group = group_cursor % n_groups
        group_cursor += 1
        start = group * args.episodes_per_step
        return batch, train_plans[start:start + args.episodes_per_step]

    for step in range(1, args.profile_steps + 6):
        timing = step > 5                                   # warmup steps excluded
        t0 = mark()
        batch, plans = next_profile_group()
        batch = batch_to_device(batch, device)
        t1 = mark()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            encoded = encode_batch(encoder, batch, device)
            t2 = mark()
            offsets = np.cumsum([0] + [len(plan.flat_positions()) for plan in plans[:-1]])
            live, labels = prepare_live_batch(encoded, batch, plans, offsets, device)
            results = training_group_results(
                engine, batch, plans, offsets, encoded, live, labels, label_text, alias_text,
                seed=args.seed + step * 10_000, collect_stats=False, args=args,
            )
            loss = torch.stack([result["loss"] for result in results]).mean()
        if not bool(torch.isfinite(loss)):
            shapes = [(len(plan.candidates), plan.support_k, plan.n_support)
                      for plan in plans]
            raise FloatingPointError(
                f"non-finite profile loss at step {step}; episode (C,k,support)={shapes}"
            )
        t3 = mark()
        loss.backward()
        t4 = mark()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], args.grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        t5 = mark()
        if timing:
            measured += 1
            for name, span in zip(phases, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                phases[name] += span

    total = sum(phases.values())
    print(f"\n=== phase breakdown over {measured} steps "
          f"(bank {args.bank_windows}, {args.episodes_per_step} episodes/step) ===")
    for name, span in phases.items():
        print(f"  {name:10s} {span / measured * 1000:7.1f} ms  {span / total * 100:5.1f}%")
    print(f"  {'TOTAL':10s} {total / measured * 1000:7.1f} ms")
    if device.type == "cuda":
        print(f"  {'PEAK VRAM':10s} {torch.cuda.max_memory_allocated(device) / 2 ** 30:7.2f} GiB "
              f"allocated / {torch.cuda.max_memory_reserved(device) / 2 ** 30:.2f} GiB reserved")

    from torch.profiler import ProfilerActivity, profile
    activities = [ProfilerActivity.CPU] + (
        [ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=activities) as prof:
        for step in range(3):
            batch, plans = next_profile_group()
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                encoded = encode_batch(encoder, batch, device)
                offsets = np.cumsum([0] + [len(plan.flat_positions())
                                            for plan in plans[:-1]])
                live, labels = prepare_live_batch(encoded, batch, plans, offsets, device)
                results = training_group_results(
                    engine, batch, plans, offsets, encoded, live, labels,
                    label_text, alias_text, seed=args.seed + step,
                    collect_stats=False, args=args,
                )
                loss = torch.stack([result["loss"] for result in results]).mean()
            loss.backward()
            optimizer.step()
    sort = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort, row_limit=14))


def _random_encoder(
    device: torch.device,
    frontend: str = "fixed",
    *,
    neutral_acquisition_text: bool = False,
    duration_range: tuple[float, float] | None = None,
    num_resolutions: int = 2,
    frontend_kwargs: dict | None = None,
) -> tuple[SetTokenizerEncoder, dict]:
    use_duration_embedding = duration_range is not None
    duration_min, duration_max = duration_range or (0.4, 1.5)
    frontend_kwargs = dict(frontend_kwargs or {})
    rope_min_period = float(frontend_kwargs.get("rope_min_period", ROPE_MIN_PERIOD_S))
    frontend_kwargs["rope_min_period"] = rope_min_period
    config = {
        "frontend": frontend,
        "d_model": 128,
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
        "multiresolution": False,
        "use_duration_embedding": use_duration_embedding,
        "duration_min_seconds": float(duration_min),
        "duration_max_seconds": float(duration_max),
        "num_resolutions": int(num_resolutions),
        "rope_min_period": rope_min_period,
        # Constructor bounds remain valid even though multiresolution is disabled. Keeping the
        # ordinary bounds makes this checkpoint reconstructible by the shared loader.
        "short_patch_choices": [0.4],
        "long_patch_choices": [1.5],
        "val_resolution_pair": [0.5, 1.5],
        "train_datasets": list(TRAIN_DATASETS),
    }
    encoder = SetTokenizerEncoder(
        d_model=128, num_layers=3, num_heads=4, dim_feedforward=256, dropout=0.1,
        dft_size=DFT_SIZE, frontend=frontend, trunk="temporal",
        descriptor_prediction=False, text_conditioning="factored",
        token_granularity="sensor", use_sensor_bias_conditioning=False,
        use_duration_embedding=use_duration_embedding,
        duration_min_seconds=duration_min, duration_max_seconds=duration_max,
        num_resolutions=num_resolutions,
        **frontend_kwargs,
    ).to(device)
    return encoder, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument(
        "--phase-b-regime", choices=("unified", "zero-shot", "enrollment"),
        required=True,
        help="episode objective for this head; the two specialized heads must share one frozen "
             "encoder checkpoint",
    )
    parser.add_argument("--datasets", nargs="+", default=list(TRAIN_DATASETS))
    parser.add_argument("--steps", type=int, default=35_000)
    parser.add_argument("--episodes-per-step", type=int, default=8)
    parser.add_argument(
        "--candidate-counts", type=int, nargs="+", default=[2, 4, 8, 16],
        help="total candidate-label roster sizes sampled during training; only "
             "--query-labels-per-episode of them supply queries",
    )
    parser.add_argument(
        "--query-labels-per-episode", type=int, default=4,
        help="candidate labels that supply query recordings in each episode; remaining candidates "
             "are genuine distractors and still compete in the loss",
    )
    parser.add_argument("--queries-per-candidate", type=int, default=4)
    parser.add_argument("--max-support", type=int, default=16)
    parser.add_argument("--bank-windows", type=int, default=512)
    # DEFAULT 0.0 (2026-08-22 decision): random-alias episodes are OUT of the training objective
    # for now — build the strongest coherent base first. Measured at seed 20260830 (20k steps,
    # matched): removal is a wash on coherent (0.5718 vs 0.5672 final) and costs the alias
    # capability (0.4906 -> 0.2934), so this trades the physio "exercise 1" regime for
    # simplicity, not for coherent gains. Restore with --alias-episode-fraction 0.5 (optionally
    # --alias-warmup-steps/--alias-ramp-steps for the curriculum).
    parser.add_argument("--alias-episode-fraction", type=float, default=0.0)
    parser.add_argument("--alias-warmup-steps", type=int, default=0,
                        help="steps with NO random-alias episodes before the curriculum starts")
    parser.add_argument("--alias-ramp-steps", type=int, default=0,
                        help="steps to ramp the alias fraction linearly up to "
                             "--alias-episode-fraction (0 = step change)")
    parser.add_argument("--disjointness", choices=("subject", "stream"), default="stream")
    parser.add_argument("--no-shared-query-stream", action="store_true")
    parser.add_argument("--holdout-label-fraction", type=float, default=0.2)
    parser.add_argument("--encoder-backbone", choices=("ours", "harnet", "unimts"), default="ours",
                        help="encoder-comparison arm; harnet/unimts are always frozen")
    parser.add_argument("--encoder-comparison", action="store_true",
                        help="compare the same recording-level engine across all encoder arms")
    parser.add_argument("--frontend", choices=("fixed", "learnable", "continuous"), default="fixed",
                        help="HALO feature extractor: fixed physical filterbank, constrained-learnable "
                             "filterbank, or continuous-time kernel bank (random-init runs only)")
    parser.add_argument("--correction-gain-init", type=float, default=0.05,
                        help="initial residual correction scale in raw cosine units")
    parser.add_argument("--max-score-correction", type=float, default=0.5,
                        help="hard bound on each recording-pair correction in cosine units")
    parser.add_argument("--semantic-scale", type=float, default=1.0,
                        help="penalty scale mapping a corpus recording label to candidate text")
    parser.add_argument("--surrogate-temperature", type=float, default=0.05,
                        help="smooth-max temperature used only for corrected-nearest gradients")
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--frontend-lr-scale", type=float, default=0.1,
                        help="learning-rate multiplier for continuous/constrained analysis parameters")
    parser.add_argument("--frontend-reg-weight", type=float, default=1e-3,
                        help="pull learned frontend analysis parameters toward their physical init")
    parser.add_argument("--engine-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--retrieval-aux-weight", type=float, default=0.0,
                        help="weight on the retrieval alignment loss (0 = the measured baseline)")
    parser.add_argument("--retrieval-aux-temperature", type=float, default=0.1)
    parser.add_argument("--no-regression-weight", type=float, default=1.0,
                        help="extra penalty when the learned residual gives the target more loss "
                             "than its detached regime reference (raw nearest for zero-shot, "
                             "enrolled 1NN where defined for enrollment)")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--neutral-acquisition-text", action="store_true",
        help="replace device/placement/gravity descriptions with modality-only sensor text; "
             "the matched causal ablation keeps every signal, episode, and model setting fixed",
    )
    parser.add_argument("--calib-batches", type=int, default=5,
                        help="episode batches used to fit frozen filterbank normalization for "
                             "--random-init; checkpoints must already carry fitted statistics")
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-episodes", type=int, default=32)
    parser.add_argument("--telemetry-every", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-seed", type=int, default=VAL_SEED,
                        help="seed for the concept holdout and the validation episodes, kept "
                             "SEPARATE from --seed so every run is scored on the same draw")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="measure this many real steps (phase timers + op table), then exit")
    parser.add_argument(
        "--sequential-episodes", action="store_true",
        help="debug/reference path: run each evidence episode separately instead of the equivalent "
             "vectorized active path",
    )
    args = parser.parse_args()

    if (args.checkpoint is None) == (not args.random_init):
        parser.error("choose exactly one of --checkpoint or --random-init")
    if args.encoder_backbone != "ours" and not args.encoder_comparison:
        parser.error("third-party backbones require --encoder-comparison")
    if args.encoder_comparison and not args.random_init:
        parser.error("the three-arm encoder comparison starts from --random-init")
    if args.encoder_backbone != "ours" and args.frontend != "fixed":
        parser.error("--frontend applies only to the HALO encoder arm")
    if args.encoder_backbone != "ours" and args.neutral_acquisition_text:
        parser.error("--neutral-acquisition-text applies only to the HALO encoder arm")
    if args.encoder_backbone != "ours" and args.freeze_encoder:
        parser.error("the baseline projection must remain trainable in the encoder comparison")
    if args.phase_b_regime != "unified" and (args.checkpoint is None or not args.freeze_encoder):
        parser.error(
            "specialized zero-shot/enrollment heads require --checkpoint and --freeze-encoder so "
            "both heads use exactly the same stable representation"
        )
    if args.phase_b_regime == "zero-shot" and (
        args.alias_episode_fraction or args.alias_warmup_steps or args.alias_ramp_steps
    ):
        parser.error(
            "zero-shot episodes have no enrollment and cannot support random aliases; leave the "
            "alias curriculum disabled"
        )
    if args.encoder_comparison and args.augment:
        parser.error("the matched encoder comparison uses clean windows; omit --augment")
    if (args.steps < 1 or args.episodes_per_step < 1 or args.query_labels_per_episode < 1
            or args.bank_windows < 1
            or args.calib_batches < 1):
        parser.error(
            "steps, episodes-per-step, query-labels-per-episode, bank-windows, and "
            "calib-batches must be positive"
        )
    effective_max_support = 0 if args.phase_b_regime == "zero-shot" else args.max_support
    if effective_max_support > args.bank_windows:
        parser.error(
            "--max-support cannot exceed --bank-windows because even one enrolled candidate "
            "would not fit in memory"
        )
    if (args.alias_episode_fraction > 0.0
            and args.bank_windows < max(args.candidate_counts) * effective_max_support):
        parser.error(
            "random aliases require every candidate to be enrolled, so --bank-windows must be at "
            "least max(candidate-counts) * max-support when aliases are enabled"
        )
    if args.semantic_scale <= 0.0 or args.surrogate_temperature <= 0.0:
        parser.error("--semantic-scale and --surrogate-temperature must be positive")
    if not 0.0 < args.correction_gain_init < args.max_score_correction:
        parser.error("correction gain must lie between zero and --max-score-correction")
    if args.warmup_steps < 0 or args.warmup_steps > args.steps:
        parser.error("--warmup-steps must lie in [0, steps]")
    if args.frontend_lr_scale <= 0 or args.frontend_reg_weight < 0:
        parser.error("--frontend-lr-scale must be positive and --frontend-reg-weight nonnegative")
    if args.smoke:
        args.steps, args.val_every, args.val_episodes = 3, 3, 2
        args.warmup_steps, args.num_workers, args.calib_batches = 1, 0, 1
        args.datasets = ["uci_har", "wisdm", "mhealth", "pamap2"]
        args.candidate_counts, args.query_labels_per_episode, args.max_support = [4, 8], 4, 2
        args.bank_windows = 32

    effective_max_support = 0 if args.phase_b_regime == "zero-shot" else args.max_support
    reference_mode = (
        "base_nearest" if args.phase_b_regime == "zero-shot" else "enrolled_1nn"
    )
    args.reference_mode = reference_mode

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    index = CorpusIndex(seed=args.seed, datasets=tuple(args.datasets))
    print(f"[phase-b] corpus: {index.summary()}", flush=True)
    train_keys, val_keys = list(index.train), list(index.val)
    excluded_streams: set[str] = set()
    if args.encoder_comparison:
        train_keys, excluded_train = encoder_comparison_keys(index, train_keys)
        val_keys, excluded_val = encoder_comparison_keys(index, val_keys)
        excluded_streams = excluded_train | excluded_val
        print(
            f"[phase-b] encoder-comparison corpus: {len(train_keys)} train / "
            f"{len(val_keys)} val windows; excluded streams={sorted(excluded_streams)}",
            flush=True,
        )
    combined_keys = train_keys + val_keys
    val_offset = len(train_keys)

    base_spec = EpisodeSpec(
        candidate_counts=tuple(args.candidate_counts),
        query_labels_per_episode=args.query_labels_per_episode,
        queries_per_candidate=args.queries_per_candidate,
        max_support=effective_max_support,
        # The episode planner requires a nonempty fallback on k=0. It is replaced below by the
        # fixed-size bank, and never reaches the DataLoader.
        background_windows=1,
        max_memory_windows=args.bank_windows,
        alias_episode_fraction=(
            0.0 if args.phase_b_regime == "zero-shot" else args.alias_episode_fraction
        ),
        disjointness=args.disjointness,
        shared_query_stream=(args.disjointness == "stream" and not args.no_shared_query_stream),
    )
    train_subject = subject_ids_for(index, train_keys)
    val_subject = subject_ids_for(index, val_keys)
    train_stream, _ = stream_ids_for(index, train_keys)
    val_stream, _ = stream_ids_for(index, val_keys)
    train_execution = execution_ids_for(index, train_keys)
    val_execution = execution_ids_for(index, val_keys)
    train_table = label_window_table(train_keys, train_subject)
    val_table = label_window_table(val_keys, val_subject)
    train_stream_table = stream_label_table(train_keys, train_subject, train_stream)
    val_stream_table = stream_label_table(val_keys, val_subject, val_stream)
    eligible = eligible_labels(
        train_table, base_spec, train_stream, execution_ids=train_execution,
    )
    train_pool, heldout = split_label_pool(eligible, args.holdout_label_fraction, args.val_seed)
    val_eligible = set(eligible_labels(
        val_table, base_spec, val_stream, execution_ids=val_execution,
    ))
    val_pool = sorted(set(heldout) & val_eligible)
    train_spec = replace_counts(base_spec, len(train_pool))
    val_spec = replace_counts(base_spec, len(val_pool))
    print(f"[phase-b] concepts: {len(train_pool)} train / {len(val_pool)} held out", flush=True)

    n_train = args.steps * args.episodes_per_step
    train_alias = alias_curriculum(
        n_train, args.episodes_per_step, args.alias_episode_fraction,
        args.alias_warmup_steps, args.alias_ramp_steps,
    )
    support_grid = support_count_grid(effective_max_support, args.phase_b_regime)
    train_candidate_schedule = grouped_candidate_schedule(
        args.steps, args.episodes_per_step, train_spec.candidate_counts, args.seed + 17,
    )
    if train_alias is not None:
        print(f"[phase-b] alias curriculum: 0.0 for {args.alias_warmup_steps} steps, "
              f"ramp over {args.alias_ramp_steps} -> {args.alias_episode_fraction}", flush=True)
    plan_cache_key, plan_cache_manifest = _episode_plan_cache_identity(
        keys=train_keys,
        subjects=train_subject,
        streams=train_stream,
        executions=train_execution,
        pool=train_pool,
        spec=train_spec,
        n_episodes=n_train,
        seed=args.seed,
        support_schedule=support_grid,
        alias_schedule=train_alias,
        candidate_schedule=train_candidate_schedule,
    )
    plan_started = time.perf_counter()
    print(f"[phase-b] preparing {n_train:,} deterministic training episodes", flush=True)
    train_plans, plan_cache_hit, plan_cache_path = _load_or_build_episode_plans(
        _episode_plan_cache_dir(),
        plan_cache_key,
        plan_cache_manifest,
        n_train,
        lambda: _build_training_core_plans(
            train_table, train_stream_table, train_pool, train_spec,
            train_stream, train_execution,
            n_episodes=n_train, seed=args.seed, schedule=support_grid,
            alias_schedule=train_alias, candidate_schedule=train_candidate_schedule,
        ),
    )
    bank_spec = BankSpec(n_windows=args.bank_windows)
    # A held-out concept is absent from the whole Phase-B objective, not merely absent as a query.
    # Letting its labeled rows enter the background bank would expose its name and representation
    # to attention before validation and turn "held out" into "seen only as a distractor".
    train_pool_set = set(train_pool)
    objective_bank_table = {
        stream: {label: by_subject for label, by_subject in by_label.items()
                 if label in train_pool_set}
        for stream, by_label in train_stream_table.items()
    }
    objective_bank_table = {
        stream: by_label for stream, by_label in objective_bank_table.items() if by_label
    }
    train_bank = bank_index(objective_bank_table)
    plan_seconds = time.perf_counter() - plan_started
    bank_started = time.perf_counter()
    train_plans = _attach_training_banks(train_plans, train_bank, bank_spec, args.seed + 1)
    bank_seconds = time.perf_counter() - bank_started
    train_episode_distribution = episode_distribution(train_plans)
    print(
        f"[phase-b] episode plans ready: core={plan_seconds:.1f}s "
        f"({'cache hit' if plan_cache_hit else 'cache miss'}) banks={bank_seconds:.1f}s",
        flush=True,
    )
    print(
        f"[phase-b] episode distribution: "
        f"C={train_episode_distribution['candidate_count']} "
        f"k={train_episode_distribution['support_k']} "
        f"query-labels={train_episode_distribution['query_label_count']} "
        f"max-support-windows={train_episode_distribution['support_windows_max']} "
        f"mean-enrolled-query={train_episode_distribution['enrolled_query_labels_mean']:.2f} "
        f"mean-enrolled-distractor={train_episode_distribution['enrolled_distractors_mean']:.2f}",
        flush=True,
    )

    max_support = max(support_grid)
    coherent_base = _make_plans(
        val_table, val_stream_table, val_pool,
        dataclasses.replace(val_spec, alias_episode_fraction=0.0),
        val_stream, val_execution, n_episodes=args.val_episodes,
        seed=args.val_seed + 2, schedule=(max_support,),
    )
    coherent_val = _matched_validation_plans(
        coherent_base, support_grid, train_bank, bank_spec, val_offset, args.val_seed + 3,
    )
    positive_grid = tuple(value for value in support_grid if value > 0)
    alias_base = _make_plans(
        val_table, val_stream_table, val_pool,
        dataclasses.replace(val_spec, alias_episode_fraction=1.0),
        val_stream, val_execution, n_episodes=args.val_episodes,
        seed=args.val_seed + 4, schedule=(max_support,),
    )
    alias_val = _matched_validation_plans(
        alias_base, positive_grid, train_bank, bank_spec, val_offset, args.val_seed + 5,
    ) if positive_grid and args.alias_episode_fraction > 0.0 else []
    val_plans = coherent_val + alias_val

    augmentation = AugmentationConfig.phase_b_generic() if args.augment else AugmentationConfig.none()
    dataset = PretrainDataset(
        index, combined_keys, augment=args.augment, augmentation_config=augmentation,
        neutral_acquisition_text=args.neutral_acquisition_text,
    )
    collate = EpisodicCollate(MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS))
    loader_kwargs = dict(
        dataset=dataset, collate_fn=collate, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", worker_init_fn=_seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        batch_sampler=GroupedEpisodicBatchSampler(train_plans, args.episodes_per_step),
        **loader_kwargs,
    )
    val_group = math.gcd(len(val_plans), args.episodes_per_step)
    val_loader = DataLoader(
        batch_sampler=GroupedEpisodicBatchSampler(val_plans, val_group), **loader_kwargs,
    )

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        encoder = build_encoder(checkpoint, device, training=True)
        config = dict(checkpoint["config"])
        if bool(config.get("neutral_acquisition_text", False)) != args.neutral_acquisition_text:
            parser.error(
                "--neutral-acquisition-text must match the source checkpoint; mixing full and "
                "neutral descriptions within one training run invalidates the ablation"
            )
        source = str(args.checkpoint)
    else:
        if args.encoder_backbone != "ours":
            from model.tokenizer.baseline_backbone import BaselineRowEncoder
            encoder = BaselineRowEncoder(args.encoder_backbone, d_model=128,
                                         freeze=True, device=device).to(device)
            config = {"encoder_backbone": args.encoder_backbone,
                      "freeze_backbone": True, "d_model": 128,
                      "retrieval_granularity": "window"}
            trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            print(f"[phase-b] backbone={args.encoder_backbone} "
                  "FROZEN "
                  f"trainable encoder params {trainable:,}", flush=True)
            if args.encoder_backbone == "harnet":
                from baselines.harnet.adapter import SSL_HUB_REPO, SSL_HUB_TAG, HARNET_NAME
                source = f"pretrained:{SSL_HUB_REPO}:{SSL_HUB_TAG}/{HARNET_NAME}"
            else:
                from baselines.unimts.adapter import UNIMTS_CKPT
                source = (
                    f"pretrained:{UNIMTS_CKPT.resolve()}"
                    f"#sha256={_file_sha256(UNIMTS_CKPT)}"
                )
            config["backbone_source"] = source
        else:
            encoder, config = _random_encoder(
                device, frontend=args.frontend,
                neutral_acquisition_text=args.neutral_acquisition_text,
            )
            source = "random-init"
            if args.encoder_comparison:
                encoder.retrieval_granularity = "window"
                config["retrieval_granularity"] = "window"
    if encoder.trunk != "temporal" or encoder.token_granularity != "sensor":
        raise SystemExit(
            "Phase B accepts only the compact temporal sensor encoder; train/reload Phase A with "
            "--trunk temporal --token-granularity sensor"
        )
    if getattr(encoder, "use_sensor_bias_conditioning", False):
        raise SystemExit("compact Phase B does not consume the legacy source-specific sensor bias")
    # A third-party backbone has no physical filterbank to calibrate and no mask token to freeze:
    # it arrives pretrained with its own input contract, handled inside BaselineRowEncoder.
    is_baseline_backbone = not hasattr(encoder, "filterbank")
    frontend = None if is_baseline_backbone else encoder.filterbank
    if is_baseline_backbone:
        pass
    elif args.random_init:
        print(f"[phase-b] calibrating filterbank on {args.calib_batches} episode batches", flush=True)
        frontend.reset_norm_accumulator()
        for batch_index, batch in enumerate(train_loader):
            batch = batch_to_device(batch, device)
            frontend.accumulate_norm_stats(
                batch["patches"].to(device, non_blocking=True),
                batch["rates"].to(device, non_blocking=True),
                batch["patch_len"].to(device, non_blocking=True),
                patch_mask=batch["patch_padding_mask"].to(device, non_blocking=True),
                channel_mask=batch["channel_mask"].to(device, non_blocking=True),
                source_rate_hz=batch["source_rates"].to(device, non_blocking=True),
            )
            if batch_index + 1 >= args.calib_batches:
                break
        frontend.finalize_norm_stats()
    elif not bool(frontend._norm_fitted.item()):
        raise SystemExit(
            "the supplied Phase-A checkpoint has no fitted filterbank normalization; refusing "
            "to train Phase B on identity-scaled physical features"
        )
    if not is_baseline_backbone:
        encoder.mask_token.requires_grad_(False)
    if args.freeze_encoder:
        encoder.requires_grad_(False)

    spec = encoder.attention_spec
    engine_config = EngineConfig(
        spec=spec,
        trunk_layers=int(encoder.transformer.num_layers),
        semantic_scale=args.semantic_scale,
        surrogate_temperature=args.surrogate_temperature,
        reranker=EvidenceRerankerConfig(
            correction_gain_init=args.correction_gain_init,
            max_correction=args.max_score_correction,
        ),
    )
    # Released backbone constructors consume different amounts of random state. Reset at the exact
    # boundary of the shared evidence engine so all comparison arms receive identical scorer/mixer
    # initialisation and dropout streams; otherwise the "encoder-only" comparison changes two things.
    torch.manual_seed(args.seed + 10_003)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + 10_003)
    engine = EvidenceEngine(encoder, engine_config).to(device).train()
    if args.freeze_encoder:
        encoder.eval()
    label_text = F.normalize(label_text_matrix(encoder, index.label_ids, device).float(), dim=-1)
    from eval.scoring import get_sbert_encoder

    alias_text = F.normalize(
        encode_neutral_aliases(get_sbert_encoder(), device).float(), dim=-1,
    )

    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    frontend_params = (
        [p for p in frontend.adaptation_parameters() if p.requires_grad]
        if frontend is not None else []
    )
    frontend_param_ids = {id(parameter) for parameter in frontend_params}
    encoder_base_params = [p for p in encoder_params if id(p) not in frontend_param_ids]
    reranker_params = [p for p in engine.reranker.parameters() if p.requires_grad]
    reranker_output_params = [
        engine.reranker.row_head.weight, engine.reranker.correction_gain_logit,
    ]
    output_ids = {id(parameter) for parameter in reranker_output_params}
    reranker_input_params = [p for p in reranker_params if id(p) not in output_ids]
    # A ladder rung that substitutes a fixed stand-in leaves a stage with no parameters at all;
    # AdamW rejects an empty group, so only non-empty ones are handed over.
    groups = []
    if encoder_base_params:
        groups.append({"params": encoder_base_params, "lr": args.encoder_lr,
                       "weight_decay": args.weight_decay, "name": "encoder"})
    if frontend_params:
        groups.append({"params": frontend_params,
                       "lr": args.encoder_lr * args.frontend_lr_scale,
                       "weight_decay": 0.0, "name": "frontend"})
    if reranker_params:
        groups.append({"params": reranker_params, "lr": args.engine_lr,
                       "weight_decay": args.weight_decay, "name": "reranker"})
    if not groups:
        raise SystemExit("every stage is frozen; there is nothing to train")
    optimizer = torch.optim.AdamW(groups)

    def lr_factor(step: int) -> float:
        if args.warmup_steps and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    provenance = capture_source_provenance(
        args.out, write=True,
        roots=("training/tokenizer", "training/evidence", "model/tokenizer",
               "model/evidence", "model/blocks.py", "data/datasets", "data/scripts"),
    )
    provenance.pop("_patch", None)
    config.update({
        "phase_b_version": PHASE_B_VERSION,
        "engine_config": dataclasses.asdict(engine_config),
        "train_datasets": list(args.datasets),
        "phase_b": {key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(args).items()},
    })
    run_info = {
        "phase_b_version": PHASE_B_VERSION,
        "source_checkpoint": source,
        "corpus": index.summary(),
        "effective_corpus": {
            "train_windows": len(train_keys), "val_windows": len(val_keys),
            "excluded_streams": sorted(excluded_streams),
        },
        "corpus_fingerprint": corpus_fingerprint(index),
        "train_concepts": train_pool,
        "heldout_concepts": val_pool,
        "episode_spec": dataclasses.asdict(train_spec),
        "episode_plan_cache": {
            "schema": EPISODE_PLAN_CACHE_SCHEMA,
            "key": plan_cache_key,
            "hit": plan_cache_hit,
            "path": str(plan_cache_path),
            "core_seconds": plan_seconds,
            "bank_seconds": bank_seconds,
        },
        "train_episode_distribution": train_episode_distribution,
        "phase_b_regime": args.phase_b_regime,
        "no_regression_reference": reference_mode,
        "bank_spec": dataclasses.asdict(bank_spec),
        "train_provenance_lift": provenance_lift(train_plans, train_stream),
        "parameters": engine.parameter_report(),
        "config": config,
        "source_provenance": provenance,
    }
    (args.out / "run_config.json").write_text(json.dumps(run_info, indent=2, default=str))
    telemetry_path = args.out / "log.jsonl"
    telemetry_path.write_text("")

    def record(row: dict) -> None:
        with telemetry_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    best = {"score": -1.0, "step": 0}

    def payload(step: int, report: dict) -> dict:
        return {
            "phase_b_version": PHASE_B_VERSION,
            "encoder": encoder.state_dict(),
            "evidence_engine": engine.state_dict(),
            "config": config,
            "phase_b_regime": args.phase_b_regime,
            "no_regression_reference": reference_mode,
            "label_ids": index.label_ids,
            "step": step,
            "selection_metric": report["selection_metric"],
            "selection_value": report["selection_score"],
            "validation": report,
            "source_provenance": provenance,
            "corpus_fingerprint": run_info["corpus_fingerprint"],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

    def run_validation(step: int) -> dict:
        report = validate(
            engine, val_loader, val_plans, label_text, alias_text, device, args.val_seed + 10_000,
            val_group, select_on_aliases=args.alias_episode_fraction > 0.0,
            reference_mode=reference_mode,
        )
        record({"step": step, "kind": "validation", **report})
        if report["selection_score"] > best["score"]:
            best.update(score=report["selection_score"], step=step)
            torch.save(payload(step, report), args.out / "best.pt")
        return report

    if args.profile_steps:
        profile_training(engine, train_loader, train_plans, label_text, alias_text,
                         optimizer, scheduler, device, args)
        return

    initial = run_validation(0)
    for step, batch in enumerate(train_loader, 1):
        batch = batch_to_device(batch, device)
        log_step = step == 1 or step % args.telemetry_every == 0
        if frontend is not None and hasattr(frontend, "request_runtime_telemetry"):
            frontend.request_runtime_telemetry(log_step)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            encoded = encode_batch(encoder, batch, device)
            start = (step - 1) * args.episodes_per_step
            plans = train_plans[start:start + args.episodes_per_step]
            offsets = np.cumsum([0] + [len(plan.flat_positions()) for plan in plans[:-1]])
            live, labels = prepare_live_batch(encoded, batch, plans, offsets, device)
            results = training_group_results(
                engine, batch, plans, offsets, encoded, live, labels, label_text, alias_text,
                seed=args.seed + step * 10_000,
                collect_stats=(step == 1 or step % args.telemetry_every == 0), args=args,
            )
            loss = torch.stack([result["loss"] for result in results]).mean()
            frontend_reg = (
                frontend.adaptation_regularization()
                if frontend is not None else loss.new_zeros(())
            )
            loss = loss + args.frontend_reg_weight * frontend_reg
        if not bool(torch.isfinite(loss)):
            shapes = [(len(plan.candidates), plan.support_k, plan.n_support)
                      for plan in plans]
            raise FloatingPointError(
                f"non-finite training loss at step {step}; episode (C,k,support)={shapes}"
            )
        loss.backward()
        # These diagnostics launch a reduction for every parameter tensor and synchronize when
        # converted to Python floats. Compute them only on steps that will actually be recorded.
        encoder_grad = _grad_norm(encoder_params) if log_step else 0.0
        frontend_grad = _grad_norm(frontend_params) if log_step else 0.0
        reranker_grad = _grad_norm(reranker_params) if log_step else 0.0
        reranker_input_grad = _grad_norm(reranker_input_params) if log_step else 0.0
        reranker_output_grad = _grad_norm(reranker_output_params) if log_step else 0.0
        preclip = float(torch.nn.utils.clip_grad_norm_(
            [p for group in groups for p in group["params"]], args.grad_clip,
            error_if_nonfinite=True,
        ))
        optimizer.step()
        scheduler.step()

        if log_step:
            stats = [result.get("stats", {}) for result in results]
            keys = sorted({key for row in stats for key in row})
            one_nn_pairs = [
                (truth, prediction)
                for result in results
                for truth, prediction, valid in zip(
                    result["truth"], result["enrolled_1nn_prediction"],
                    result["enrolled_1nn_valid"],
                ) if valid
            ]
            query_count = sum(len(result["truth"]) for result in results)
            row = {
                "step": step,
                "kind": "train",
                "loss": float(loss.detach()),
                "task_loss": float(np.mean([r["task_loss"] for r in results])),
                "aux_loss": float(np.mean([r["aux_loss"] for r in results])),
                "base_loss": float(np.mean([r["base_loss"] for r in results])),
                "reference_loss": float(np.mean([r["reference_loss"] for r in results])),
                "enrolled_1nn_loss": float(np.mean([
                    r["enrolled_1nn_loss"] for r in results
                ])),
                "no_regression_loss": float(np.mean([
                    r["no_regression_loss"] for r in results
                ])),
                "accuracy": float(np.mean([
                    np.mean(np.equal(result["truth"], result["prediction"])) for result in results
                ])),
                "enrolled_1nn_reference_coverage": len(one_nn_pairs) / max(query_count, 1),
                "enrolled_1nn_accuracy": (
                    float(np.mean([truth == prediction for truth, prediction in one_nn_pairs]))
                    if one_nn_pairs else 0.0
                ),
                "encoder_grad_norm": encoder_grad,
                "frontend_grad_norm": frontend_grad,
                "frontend_reg": float(frontend_reg.detach()),
                "frontend_reg_weighted": float(
                    (args.frontend_reg_weight * frontend_reg).detach()),
                "reranker_grad_norm": reranker_grad,
                "reranker_input_grad_norm": reranker_input_grad,
                "reranker_output_grad_norm": reranker_output_grad,
                "total_preclip_grad_norm": preclip,
                "clip_coefficient": min(1.0, args.grad_clip / max(preclip, 1e-12)),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            row.update({f"lr_{group.get('name', index)}": group["lr"]
                        for index, group in enumerate(optimizer.param_groups)})
            if frontend is not None:
                row.update(frontend.adaptation_summary())
                if hasattr(frontend, "runtime_summary"):
                    row.update(frontend.runtime_summary())
            row.update({key: float(np.mean([value[key] for value in stats if key in value]))
                        for key in keys})
            record(row)
        if step % args.val_every == 0 or step == args.steps:
            final = run_validation(step)

    torch.save(payload(args.steps, final), args.out / "last.pt")
    summary = {
        "steps": args.steps,
        "best": best,
        "initial_validation": initial,
        "final_validation": final,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
