"""Train the Phase-B predictor on clean, independently sampled memory episodes.

Each episode samples a candidate count and one support count ``k``. Every candidate receives ``k``
random eligible executions in memory. ``k=0`` uses coherent activity names for semantic prediction;
positive-k episodes assign fresh arbitrary names so prediction requires support-to-name binding.
Candidate cross-entropy is the only objective. Hard top-k retrieval remains entirely query-driven;
support identity constructs memory and telemetry but never selects evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.scoring import get_sbert_encoder
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from model.evidence.relational_decoder import (
    RelationalDecoderConfig,
    RelationalEvidenceDecoder,
    build_coreference_slots,
    build_window_groups,
    label_text_votes,
    retrieval_vote_base,
)
from training.evidence.bank_guard import (
    assert_embedding_path_current,
    assert_bank_matches_backbone,
    assert_bank_current,
    assert_patch_bank,
    assert_patch_embedding_path_current,
    bank_fingerprint,
)
from training.evidence.labeltext import build_label_variants, ensemble_text
from training.evidence.episode_labels import encode_neutral_aliases, episode_label_set
from training.evidence.device import resolve_device
from training.evidence.folds import VALIDATION_QUERY_POLICY, phase_b_fold_masks
from training.evidence.patch_episodes import (
    EpisodeMemoryView,
    PatchTable,
    assemble_evidence,
    build_episode_memory_view,
    describe_episode_composition,
    simultaneous_stream_pairs,
    support_capacity_by_label,
)
from training.evidence.live_encoder import PatchViewSpec, SourcePatchEncoder
from training.evidence.policy import (
    ACTIVE_REFRESH_STEPS,
    ACTIVE_WINDOWS_PER_LABEL,
    CANDIDATE_COUNT_RANGE,
    EPISODES_PER_STEP,
    RETRIEVAL_VOTE_SCALE,
    EPISODE_TYPES,
    LABEL_TEXT_MODES,
    PHASE_B_TRAINING_REGIME,
    PHYSICAL_VIEW_MODES,
    QUERIES_PER_EPISODE,
    RETRIEVAL_PROJECTION_EMA,
    RETRIEVAL_SUBSPACE_DIM,
    RETRIEVAL_SUBSPACES,
    RETRIEVAL_TEMPERATURE,
    SUPPORT_COUNT_RANGE,
    VALIDATION_CURVE_CANDIDATE_COUNT,
    VALIDATION_SUPPORT_COUNTS,
    TOKENIZER_EMA_DECAY,
    TOKENIZER_FINETUNE_WARMUP_STEPS,
    TOKENIZER_LR_SCALE,
    PhaseBPolicy,
)
from training.evidence.telemetry import PhaseBTelemetry
from training.tokenizer.eval_transfer import build_encoder

_DIR = Path(__file__).resolve().parent / "outputs"
_DEFAULT_BANK = _DIR / "memory_bank.pt"
_DEFAULT_OUT = _DIR / "patch_evidence_predictor.pt"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAMILY_PATH = _REPO_ROOT / "data/labels/activity_families.json"
_PHASE_B_BEHAVIOR_PATHS = (
    "training/evidence/train_patch_decoder.py",
    "training/evidence/patch_episodes.py",
    "training/evidence/policy.py",
    "training/evidence/episode_labels.py",
    "training/evidence/labeltext.py",
    "training/evidence/live_encoder.py",
    "model/evidence/relational_decoder.py",
    "model/evidence/patch_retrieval.py",
)
SEED = 20260725
# Gradient decomposition retains the current episode graph, so sample it periodically rather than on
# every optimizer update. The fixed cadence also prevents an end-of-step heartbeat from repeatedly
# firing just before the following step has a chance to observe that telemetry is due.
RETRIEVAL_DIAGNOSTIC_STEPS = 100
@dataclass(frozen=True)
class AdaptationEpisodeSpec:
    episode_type: str
    support_count: int
    candidate_count: int
    label_mode: str = "coherent"
    physical_view_mode: str = "clean"

    @property
    def enrollment_shape(self) -> str:
        if self.support_count == 0:
            return "zero"
        return "full"

    def __post_init__(self) -> None:
        if self.episode_type not in EPISODE_TYPES:
            raise ValueError(f"unknown episode type {self.episode_type!r}")
        if self.support_count < 0:
            raise ValueError("support_count must be nonnegative")
        if self.candidate_count < 2:
            raise ValueError("candidate_count must be at least two")
        expected_label_mode = "coherent" if self.support_count == 0 else "random_alias"
        if self.label_mode != expected_label_mode:
            raise ValueError(
                f"support_count={self.support_count} requires label_mode="
                f"{expected_label_mode!r}"
            )
        if self.physical_view_mode != "clean":
            raise ValueError("minimal Phase-B episodes require clean views")
        expected = "semantic_zero_support" if self.support_count == 0 else "ordinary_few_support"
        if self.episode_type != expected:
            raise ValueError(
                f"support_count={self.support_count} requires episode_type={expected!r}"
            )


def validation_canary_cases(recipes, fold_pools):
    """Return the complete recipe-by-transfer-fold validation grid.

    Keeping this Cartesian product explicit prevents checkpoint-selection folds from accidentally
    receiving different mixtures of support counts, label modes, or enrollment shapes.
    """
    return [
        (fold_name, fold_pool, recipe_index, recipe)
        for fold_name, fold_pool in fold_pools
        for recipe_index, recipe in enumerate(recipes)
    ]


class EpisodeSampler:
    """Draw independent, fixed-policy episodes without a curriculum."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def sample_batch(self, count: int) -> list[AdaptationEpisodeSpec]:
        if count < 1:
            raise ValueError("episodes_per_step must be positive")
        support_low, support_high = SUPPORT_COUNT_RANGE
        candidate_low, candidate_high = CANDIDATE_COUNT_RANGE
        result = []
        for _ in range(count):
            support_count = int(self.rng.integers(support_low, support_high + 1))
            result.append(AdaptationEpisodeSpec(
                episode_type=(
                    "semantic_zero_support" if support_count == 0
                    else "ordinary_few_support"
                ),
                support_count=support_count,
                candidate_count=int(self.rng.integers(candidate_low, candidate_high + 1)),
                label_mode=("coherent" if support_count == 0 else "random_alias"),
            ))
        return result



def atomic_torch_save(payload: dict, path: Path) -> None:
    """Replace a checkpoint atomically so interruption cannot leave a partial artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def phase_b_source_fingerprint(paths=None) -> str:
    """Fingerprint every source file that defines predictor-training behavior."""
    digest = hashlib.sha256()
    for value in paths or _PHASE_B_BEHAVIOR_PATHS:
        path = Path(value)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Phase-B behavior source is missing: {path}")
        name = str(path.relative_to(_REPO_ROOT)) if path.is_relative_to(_REPO_ROOT) else str(path)
        payload = path.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _update_structured_hash(digest, value) -> None:
    """Hash nested canary state without relying on pickle implementation details."""
    if value is None:
        digest.update(b"none")
    elif is_dataclass(value) and not isinstance(value, type):
        digest.update(f"dataclass:{type(value).__qualname__}".encode("utf-8"))
        for field in fields(value):
            digest.update(field.name.encode("utf-8"))
            _update_structured_hash(digest, getattr(value, field.name))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}".encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(f"ndarray:{array.dtype}:{array.shape}".encode("utf-8"))
        digest.update(array.tobytes())
    elif isinstance(value, np.generic):
        _update_structured_hash(digest, value.item())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_structured_hash(digest, key)
            _update_structured_hash(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}:{len(value)}".encode("utf-8"))
        for item in value:
            _update_structured_hash(digest, item)
    elif isinstance(value, (str, int, float, bool)):
        digest.update(
            f"scalar:{type(value).__name__}:{value!r}".encode("utf-8")
        )
    else:
        raise TypeError(f"unsupported canary fingerprint value: {type(value).__name__}")


def structured_fingerprint(value) -> str:
    digest = hashlib.sha256()
    _update_structured_hash(digest, value)
    return digest.hexdigest()


def milestone_checkpoint_path(output: Path, step: int) -> Path:
    return output.parent / f"{output.stem}.milestones" / f"step_{step:06d}{output.suffix}"


def balanced_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    per_class = [float((pred[true == label] == label).mean()) for label in np.unique(true)]
    return float(np.mean(per_class)) if per_class else float("nan")


def checkpoint_is_better(metrics: dict, best: dict) -> bool:
    """Rank only checkpoints that demonstrate learned use of the enrolled evidence."""
    if not bool(metrics.get("checkpoint_eligible", False)):
        return False
    if not bool(best.get("checkpoint_eligible", False)):
        return True
    return float(metrics["adaptation_selection_score"]) > float(
        best["adaptation_selection_score"]
    )


def checkpoint_eligibility(metrics: dict) -> tuple[bool, dict[str, bool]]:
    """Predeclared mechanism checks for a deployable learned decoder checkpoint."""
    checks = {
        "matches_closed_form_low_k_control": (
            np.isfinite(metrics.get("selection_low_k_adaptation_gain", float("nan")))
            and metrics["selection_low_k_adaptation_gain"] >= 0.0
        ),
        "benefits_from_support_presence": (
            np.isfinite(metrics.get("support_removal_true_probability_drop", float("nan")))
            and metrics["support_removal_true_probability_drop"] > 0.0
        ),
        "uses_support_label_binding": (
            np.isfinite(
                metrics.get("support_label_shuffle_true_probability_drop", float("nan"))
            )
            and metrics["support_label_shuffle_true_probability_drop"] > 0.0
        ),
    }
    return all(checks.values()), checks


def deranged_support_view(view: EpisodeMemoryView, *, shift: int = 1) -> EpisodeMemoryView:
    """Keep every memory row fixed while assigning support to the wrong candidate."""
    n_candidates = len(view.candidate_ids)
    if n_candidates < 2:
        raise ValueError("support-label derangement needs at least two candidates")
    shift = int(shift) % n_candidates
    if shift == 0:
        raise ValueError("support-label derangement cannot use a zero shift")
    binding = view.support_candidate.clone()
    binding[view.support_mask] = (binding[view.support_mask] + shift) % n_candidates
    return replace(view, support_candidate=binding)


def nested_support_view(
    view: EpisodeMemoryView,
    patch: dict,
    index_rows: torch.Tensor,
    support_count: int,
) -> EpisodeMemoryView:
    """Derive a nested k-shot view from one maximum-support canary."""
    if support_count < 0:
        raise ValueError("support_count must be nonnegative")
    support_units = torch.where(
        torch.as_tensor(patch["event_verified"])[index_rows.detach().cpu()].bool().to(
            view.support_mask.device
        ),
        torch.as_tensor(patch["event"])[index_rows.detach().cpu()].long().to(
            view.support_mask.device
        ) + int(torch.as_tensor(patch["window"]).max()) + 1,
        torch.as_tensor(patch["window"])[index_rows.detach().cpu()].long().to(
            view.support_mask.device
        ),
    )
    keep = torch.zeros_like(view.support_mask)
    realized = torch.zeros_like(view.support_units_per_candidate)
    for candidate_position in range(len(view.candidate_ids)):
        members = (
            view.support_mask & view.support_candidate.eq(candidate_position)
        )
        units = torch.unique(support_units[members], sorted=True)
        chosen = units[:support_count]
        if len(chosen):
            keep |= members & torch.isin(support_units, chosen)
        realized[candidate_position] = len(chosen)
    originally_enrolled = view.support_units_per_candidate.gt(0)
    if support_count > 0 and bool(
        (realized[originally_enrolled] < support_count).any()
    ):
        raise ValueError("maximum-support canary cannot realize the requested nested k")
    allowed = view.allowed & (
        ~view.support_mask.view(1, 1, -1) | keep.view(1, 1, -1)
    )
    binding = torch.where(
        keep, view.support_candidate, torch.full_like(view.support_candidate, -1)
    )
    return replace(
        view,
        allowed=allowed,
        support_mask=keep,
        support_candidate=binding,
        support_units_per_candidate=realized,
        episode_type=(
            "semantic_zero_support" if support_count == 0 else view.episode_type
        ),
    )


def label_index(candidates: torch.Tensor, n_vocab: int, device) -> torch.Tensor:
    position = torch.full((n_vocab,), -1, device=device, dtype=torch.long)
    position[candidates] = torch.arange(len(candidates), device=device)
    return position

def sample_text_tables(variants: torch.Tensor, generator: torch.Generator):
    """Independently sample evidence and candidate phrasings for every vocabulary label."""
    n_labels, n_variants, _ = variants.shape
    ev_pick = torch.randint(n_variants, (n_labels,), generator=generator, device=variants.device)
    cand_pick = torch.randint(n_variants, (n_labels,), generator=generator, device=variants.device)
    rows = torch.arange(n_labels, device=variants.device)
    return variants[rows, ev_pick], variants[rows, cand_pick]


def parameter_gradient_norm(parameters, device) -> torch.Tensor:
    """L2 norm of present gradients, returning zero before a module enters the graph."""
    total = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().square().sum()
    return total.sqrt()


def parameter_gradient_statistics(parameters, device) -> dict[str, torch.Tensor]:
    """Size-normalized gradient health for a trainable parameter collection.

    Raw L2 norms grow with module size. RMS permits comparisons between paths, while coverage
    distinguishes a disconnected parameter from a connected parameter whose derivative is zero in
    the current batch. These reductions run only on the telemetry cadence.
    """
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    total_elements = sum(parameter.numel() for parameter in parameters)
    squared = torch.zeros((), dtype=torch.float32, device=device)
    covered_elements = 0
    nonzero_elements = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        squared = squared + gradient.square().sum()
        covered_elements += parameter.numel()
        nonzero_elements = nonzero_elements + gradient.count_nonzero().float()
    denominator = max(total_elements, 1)
    return {
        "norm": squared.sqrt(),
        "rms": (squared / denominator).sqrt(),
        "coverage": torch.tensor(
            covered_elements / denominator, dtype=torch.float32, device=device
        ),
        "nonzero_fraction": nonzero_elements / denominator,
    }


def decoder_gradient_components(decoder) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    """Return disjoint, exhaustive groups for Phase-B decoder gradient telemetry."""
    components = {
        "relational_attention": tuple(decoder.blocks.parameters()),
        "candidate_readout": tuple(decoder.readout.parameters()),
        "input_layer_norm": tuple(decoder.in_ln.parameters()),
        "output_layer_norm": tuple(decoder.final_ln.parameters()),
        "role_embeddings": tuple(decoder.role_emb.parameters()),
        "coreference_slot_embeddings": tuple(decoder.slot_emb.parameters()),
        "window_group_embeddings": tuple(decoder.group_emb.parameters()),
        "acquisition_relation_embeddings": tuple(
            parameter
            for module in (
                decoder.same_config_emb,
                decoder.same_subject_emb,
                decoder.same_sensor_emb,
            )
            for parameter in module.parameters()
        ),
        "query_projection": tuple(decoder.proj_query.parameters()),
        "evidence_projection": tuple(decoder.proj_evidence.parameters()),
        "text_projection": tuple(decoder.proj_text.parameters()),
        "time_projection": tuple(decoder.proj_time.parameters()),
        "component_scales": tuple(decoder.component_log_scale.parameters()),
    }
    grouped = [id(parameter) for parameters in components.values() for parameter in parameters]
    expected = [id(parameter) for parameter in decoder.parameters() if parameter.requires_grad]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(expected):
        raise RuntimeError("decoder gradient telemetry groups must partition trainable parameters")
    return components


def tokenizer_gradient_components(encoder) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    """Partition the inference-path parameters used by optional Phase-B fine-tuning."""
    duration_projection = getattr(encoder, "duration_proj", None)
    duration_parameters = tuple(duration_projection.parameters()) \
        if duration_projection is not None else ()
    duration_gate = getattr(encoder, "duration_gate_logit", None)
    if duration_gate is not None and duration_gate.requires_grad:
        duration_parameters += (duration_gate,)
    components = {
        "filterbank": tuple(
            parameter for parameter in encoder.filterbank.parameters()
            if parameter.requires_grad
        ),
        "channel_conditioning": tuple(
            parameter for parameter in encoder.fusion.parameters()
            if parameter.requires_grad
        ),
        "duration_conditioning": duration_parameters,
        "context_transformer": tuple(
            parameter for parameter in encoder.transformer.parameters()
            if parameter.requires_grad
        ),
    }
    components = {name: parameters for name, parameters in components.items() if parameters}
    grouped = [id(parameter) for parameters in components.values() for parameter in parameters]
    expected = [id(parameter) for parameter in encoder.parameters() if parameter.requires_grad]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(expected):
        raise RuntimeError("tokenizer gradient telemetry groups must partition trainable parameters")
    return components


@torch.no_grad()
def update_tokenizer_ema(ema_encoder, online_encoder, decay: float = TOKENIZER_EMA_DECAY) -> None:
    """Momentum-update trainable encoder state; fixed buffers are copied exactly."""
    for ema_parameter, online_parameter in zip(
        ema_encoder.parameters(), online_encoder.parameters(), strict=True
    ):
        ema_parameter.lerp_(online_parameter.detach(), 1.0 - decay)
    for ema_buffer, online_buffer in zip(
        ema_encoder.buffers(), online_encoder.buffers(), strict=True
    ):
        ema_buffer.copy_(online_buffer.detach())


def synthetic_smoke_bank() -> dict:
    """Small source-shaped bank spanning every current label, config, and subject split."""
    from eval.data import load_global_labels
    from training.evidence.bank_guard import vocab_fingerprint

    generator = torch.Generator().manual_seed(SEED)
    vocab = list(load_global_labels())
    n_labels, d, n_cfg, n_subj = len(vocab), 32, 4, 4
    centers = F.normalize(torch.randn(n_labels, d, generator=generator), dim=-1)
    ys, subjs, cfgs = [], [], []
    for label in range(n_labels):
        for config in range(n_cfg):
            for subject in range(n_subj):
                ys.append(label); cfgs.append(config); subjs.append(subject)
    y = torch.tensor(ys, dtype=torch.long)
    subj = torch.tensor(subjs, dtype=torch.long)
    cfg = torch.tensor(cfgs, dtype=torch.long)
    Z = F.normalize(centers[y] + 0.05 * torch.randn(len(y), d, generator=generator), dim=-1)
    event = torch.arange(len(y), dtype=torch.long)
    event_verified = torch.zeros(len(y), dtype=torch.bool)
    window = torch.arange(len(y)).repeat_interleave(2)
    patch_Z = F.normalize(
        Z[window] + 0.08 * torch.randn(len(window), d, generator=generator), dim=-1
    )
    patch = {
        "Z": patch_Z.half(), "y": y[window], "subj": subj[window], "cfg": cfg[window],
        "sensor": cfg[window],
        "window": window, "event": event[window],
        "event_verified": event_verified[window],
        "time": torch.tensor([0.5, 1.5] * len(y), dtype=torch.float32),
        "duration": torch.ones(len(window)),
        "resolution": torch.tensor([0, 1] * len(y), dtype=torch.long),
        "ordinal": torch.tensor([0, 1] * len(y), dtype=torch.long),
    }
    bank = {
        "schema_version": 3, "population_fp": "synthetic-smoke",
        "Z": Z.half(), "y": y, "subj": subj, "cfg": cfg,
        "event": event, "event_verified": event_verified, "patch": patch,
        "source_row": torch.arange(len(y), dtype=torch.long),
        "patch_embed_probe": torch.zeros(1), "embed_probe": torch.zeros(1),
        "vocab": vocab, "vocab_fp": vocab_fingerprint(vocab),
        "d_model": d, "subj_names": {i: f"smoke-subject-{i}" for i in range(n_subj)},
        "cfg_names": {i: f"smoke/config-{i}" for i in range(n_cfg)},
        "cfg_rate_hz": {i: 50.0 + 10.0 * i for i in range(n_cfg)},
        "event_names": {i: f"smoke-event-{i}" for i in range(len(y))},
        "backbone": {
            "checkpoint": "synthetic-smoke", "step": 0, "val_ba": 0.0,
            "git": "synthetic-smoke", "fingerprint": "synthetic-smoke",
        },
        "corpus": {
            "datasets": ["synthetic-smoke"], "n_streams": n_cfg,
            "streams": {f"synthetic/config-{i}": len(y) // n_cfg for i in range(n_cfg)},
            "n_encoded_windows": len(y), "phase_a_corpus_fp": "synthetic-smoke",
        },
        "archive_budget_windows": len(y), "source_scan_cap_per_stream": len(y),
        "source_alignment": "native",
    }
    bank["bank_fp"] = bank_fingerprint(bank)
    return bank


def load_activity_families(vocab: list[str]) -> tuple[torch.Tensor, dict[str, list[str]], str]:
    payload = json.loads(_FAMILY_PATH.read_text())
    families = {str(name): list(labels) for name, labels in payload["families"].items()}
    flattened = [label for labels in families.values() for label in labels]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(vocab):
        raise ValueError("activity_families.json must map every current vocabulary label exactly once")
    family_index = {}
    for index, labels in enumerate(families.values()):
        family_index.update({label: index for label in labels})
    ids = torch.tensor([family_index[label] for label in vocab], dtype=torch.long)
    fingerprint = hashlib.sha256(_FAMILY_PATH.read_bytes()).hexdigest()[:16]
    return ids, families, fingerprint


def choose_candidates(
    query_labels: torch.Tensor,
    n_candidates: int,
    n_vocab: int,
    text_table: torch.Tensor,
    physical_centroids: torch.Tensor,
    *,
    truth_present: bool,
    rng: np.random.Generator,
    allowed_vocab: torch.Tensor | None = None,
    hard_fraction: float = 0.5,
) -> torch.Tensor:
    """Candidate set mixing nearest confusable labels with random distractors.

    The active minimal trainer passes ``hard_fraction=0`` and samples all distractors uniformly.
    The optional hard branch remains available to the parked confidence experiment. All-random
    distractors make most episodes
    trivial — the full vocabulary rarely lands two similar activities in the same set by chance —
    while all-near distractors drop the easy cases the model also has to get right. Nearness averages
    label-text cosine with physical-centroid cosine
    so that neither modality alone defines "confusable".

    An earlier revision offered five named modes (random / language / motion_family / physical /
    mixed) drawn at random per episode. Nothing ever measured a difference between them, and four of
    the five produced episodes that were uniformly easy or uniformly hard.
    """
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in [0, 1]")
    device = text_table.device
    truth = torch.unique(query_labels).long()
    allowed = (
        torch.arange(n_vocab, device=device)
        if allowed_vocab is None else allowed_vocab.to(device=device, dtype=torch.long)
    )
    required = truth if truth_present else truth.new_empty(0)
    if len(required) > n_candidates:
        raise ValueError(
            f"candidate budget {n_candidates} cannot contain {len(required)} query labels"
        )
    pool = allowed[~torch.isin(allowed, truth)]
    n_distractors = n_candidates - len(required)
    if n_distractors < 1 and not truth_present:
        raise ValueError("truth-absent candidate set needs at least one distractor")
    if n_distractors > len(pool):
        n_distractors = len(pool)

    hard_count = min(int(round(n_distractors * hard_fraction)), len(pool))
    if hard_count:
        score = 0.5 * (
            F.normalize(text_table[pool], dim=-1)
            @ F.normalize(text_table[truth], dim=-1).t()
        ).max(dim=1).values + 0.5 * (
            F.normalize(physical_centroids[pool], dim=-1)
            @ F.normalize(physical_centroids[truth], dim=-1).t()
        ).max(dim=1).values
        near = pool[score.topk(hard_count).indices]
    else:
        near = pool[:0]
    remainder_pool = pool[~torch.isin(pool, near)]
    random_count = n_distractors - len(near)
    if random_count:
        chosen = np.concatenate([
            near.cpu().numpy(),
            rng.choice(remainder_pool.cpu().numpy(), size=random_count, replace=False),
        ])
    else:
        chosen = near.cpu().numpy()
    result = torch.cat([
        required.to(device),
        torch.as_tensor(chosen, device=device, dtype=torch.long),
    ])
    return result[torch.argsort(result)]


def sample_queries(
    pool: torch.Tensor,
    labels: torch.Tensor,
    y: torch.Tensor,
    n: int,
    rng: np.random.Generator,
    *,
    config_ids: torch.Tensor | None = None,
    subject_ids: torch.Tensor | None = None,
    label_alpha: float = 0.5,
    subject_alpha: float = 0.5,
) -> torch.Tensor:
    """Draw queries hierarchically across label, configuration, subject, then window."""
    candidates = pool[torch.isin(y[pool], labels)]
    if not len(candidates):
        raise ValueError("no query windows have the requested episode labels")
    if not 0.0 <= label_alpha <= 1.0 or not 0.0 <= subject_alpha <= 1.0:
        raise ValueError("label_alpha and subject_alpha must be in [0, 1]")

    candidate_labels = y[candidates].long()
    label_counts = torch.bincount(
        candidate_labels, minlength=int(y.max()) + 1
    ).float().clamp_min(1)
    if config_ids is None:
        # Compatibility path: the default alpha=0.5 reproduces the historical inverse-sqrt
        # per-row query weights.
        weights = label_counts[candidate_labels].pow(label_alpha - 1.0)
    else:
        if len(config_ids) != len(y):
            raise ValueError("config_ids must align 1:1 with y")
        candidate_configs = config_ids[candidates].long()
        if bool((candidate_configs < 0).any()):
            raise ValueError("config_ids must be nonnegative")
        config_base = int(candidate_configs.max()) + 1
        label_config_code = candidate_labels * config_base + candidate_configs
        unique_lc, lc_inverse, lc_counts = torch.unique(
            label_config_code, sorted=True, return_inverse=True, return_counts=True
        )
        lc_labels = torch.div(unique_lc, config_base, rounding_mode="floor")
        configs_per_label = torch.bincount(
            lc_labels, minlength=len(label_counts)
        ).float().clamp_min(1)
        label_mass = label_counts[candidate_labels].pow(label_alpha)

        if subject_ids is None:
            within_config = lc_counts[lc_inverse].float().reciprocal()
        else:
            if len(subject_ids) != len(y):
                raise ValueError("subject_ids must align 1:1 with y")
            candidate_subjects = subject_ids[candidates].long()
            if bool((candidate_subjects < 0).any()):
                raise ValueError("subject_ids must be nonnegative")
            subject_base = int(candidate_subjects.max()) + 1
            lcs_code = label_config_code * subject_base + candidate_subjects
            unique_lcs, lcs_inverse, lcs_counts = torch.unique(
                lcs_code, sorted=True, return_inverse=True, return_counts=True
            )
            lcs_lc_code = torch.div(unique_lcs, subject_base, rounding_mode="floor")
            lcs_lc_index = torch.searchsorted(unique_lc, lcs_lc_code)
            subject_score_sum = torch.zeros(
                len(unique_lc), dtype=torch.float32, device=pool.device
            )
            subject_score_sum.index_add_(
                0, lcs_lc_index, lcs_counts.float().pow(subject_alpha)
            )
            within_config = (
                lcs_counts[lcs_inverse].float().pow(subject_alpha - 1.0)
                / subject_score_sum[lc_inverse]
            )
        weights = (
            label_mass
            / configs_per_label[candidate_labels]
            * within_config
        )
    generator = torch.Generator(device=pool.device).manual_seed(
        int(rng.integers(0, 2**31 - 1))
    )
    draw = min(n, len(candidates))
    return candidates[torch.multinomial(
        weights, draw, replacement=len(candidates) < n, generator=generator
    )]


def sample_queries_covering_labels(
    pool: torch.Tensor,
    labels: torch.Tensor,
    y: torch.Tensor,
    n: int,
    rng: np.random.Generator,
    *,
    config_ids: torch.Tensor,
    subject_ids: torch.Tensor,
    required_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Guarantee candidate coverage, then fill the episode with balanced query draws."""
    generator = torch.Generator(device=labels.device).manual_seed(
        int(rng.integers(0, 2**31 - 1))
    )
    required_labels = (
        labels[:0] if required_labels is None else torch.unique(required_labels.long())
    )
    if bool((~torch.isin(required_labels, labels)).any()):
        raise ValueError("required query labels must be members of the episode candidate set")
    if len(required_labels) > n:
        raise ValueError("query count is smaller than required query-label coverage")
    remaining_labels = labels[~torch.isin(labels, required_labels)]
    extra_count = min(n - len(required_labels), len(remaining_labels))
    extra = remaining_labels[torch.randperm(
        len(remaining_labels), generator=generator, device=labels.device,
    )[:extra_count]]
    chosen_labels = torch.cat([required_labels, extra])
    rows = [
        sample_queries(
            pool, label.view(1), y, 1, rng,
            config_ids=config_ids, subject_ids=subject_ids,
            label_alpha=0.0, subject_alpha=0.5,
        )
        for label in chosen_labels
    ]
    remaining = n - len(rows)
    if remaining > 0:
        rows.append(sample_queries(
            pool, labels, y, remaining, rng,
            config_ids=config_ids, subject_ids=subject_ids,
            label_alpha=0.0, subject_alpha=0.5,
        ))
    result = torch.cat(rows)
    generator = torch.Generator(device=result.device).manual_seed(
        int(rng.integers(0, 2**31 - 1))
    )
    order = torch.randperm(len(result), generator=generator, device=result.device)
    return result[order]


class ActiveSupportUnits:
    """Distinct active support units, grouped by label and by (label, subject).

    Feasibility below is decided entirely by *how many distinct support units* the active index
    holds for a label, optionally restricted to or excluding one subject. Computing that with
    ``torch.unique(active_unit[mask])`` inside a per-label, per-subject loop cost one GPU
    synchronization per pair — measured at 42 ms per call, ~2.9 calls per episode, 69% of Phase-B
    training time. These tables answer the same three questions from two sorts, and each lookup
    returns exactly what the corresponding ``torch.unique`` returned: an ascending array of unit ids.

    The tables are built on CPU because this is index bookkeeping over the active rows, not tensor
    maths; keeping it off the GPU removes the synchronizations rather than merely batching them.
    """

    __slots__ = ("_by_label", "_by_label_subject", "_sole_subject_counts")

    def __init__(self, active_y: torch.Tensor, active_subj: torch.Tensor,
                 active_unit: torch.Tensor):
        label = active_y.detach().cpu().numpy()
        subject = active_subj.detach().cpu().numpy()
        unit = active_unit.detach().cpu().numpy()

        # Distinct (label, subject, unit) triples, ordered by label, then subject, then unit.
        order = np.lexsort((unit, subject, label))
        sorted_label, sorted_subject, sorted_unit = label[order], subject[order], unit[order]
        if len(sorted_label):
            fresh = np.empty(len(sorted_label), dtype=bool)
            fresh[0] = True
            fresh[1:] = (
                (sorted_label[1:] != sorted_label[:-1])
                | (sorted_subject[1:] != sorted_subject[:-1])
                | (sorted_unit[1:] != sorted_unit[:-1])
            )
            triple_label = sorted_label[fresh]
            triple_subject = sorted_subject[fresh]
            triple_unit = sorted_unit[fresh]
        else:
            triple_label = triple_subject = triple_unit = np.empty(0, dtype=np.int64)

        self._by_label_subject = self._group(
            triple_unit,
            np.stack([triple_label, triple_subject], axis=1) if len(triple_label)
            else np.empty((0, 2), dtype=np.int64),
        )

        # Re-key the same triples by (label, unit) to get per-label units and, for each unit, how
        # many distinct subjects carry it.
        by_unit = np.lexsort((triple_unit, triple_label))
        pair_label = triple_label[by_unit]
        pair_unit = triple_unit[by_unit]
        pair_subject = triple_subject[by_unit]
        if len(pair_label):
            starts = np.empty(len(pair_label), dtype=bool)
            starts[0] = True
            starts[1:] = (
                (pair_label[1:] != pair_label[:-1]) | (pair_unit[1:] != pair_unit[:-1])
            )
            head = np.flatnonzero(starts)
            subject_count = np.diff(np.r_[head, len(pair_label)])
            unit_label = pair_label[head]
            unit_unit = pair_unit[head]
            # A unit is invisible once its only subject is excluded, and only then.
            sole_subject = np.where(subject_count == 1, pair_subject[head], -1)
        else:
            unit_label = unit_unit = sole_subject = np.empty(0, dtype=np.int64)

        self._by_label = self._group(unit_unit, unit_label.reshape(-1, 1))

        confined = sole_subject >= 0
        self._sole_subject_counts: dict[tuple[int, int], int] = {}
        if bool(confined.any()):
            keys, counts = np.unique(
                np.stack([unit_label[confined], sole_subject[confined]], axis=1),
                axis=0, return_counts=True,
            )
            self._sole_subject_counts = {
                (int(key[0]), int(key[1])): int(count) for key, count in zip(keys, counts)
            }

    @staticmethod
    def _group(values: np.ndarray, keys: np.ndarray) -> dict:
        """Slice `values` into the contiguous runs named by `keys` (already grouped)."""
        grouped: dict = {}
        if not len(keys):
            return grouped
        starts = np.empty(len(keys), dtype=bool)
        starts[0] = True
        starts[1:] = (keys[1:] != keys[:-1]).any(axis=1)
        head = np.flatnonzero(starts)
        for start, end in zip(head, np.r_[head[1:], len(keys)]):
            key = keys[start]
            grouped[int(key[0]) if key.shape[0] == 1 else tuple(int(k) for k in key)] = (
                values[start:end]
            )
        return grouped

    _EMPTY = np.empty(0, dtype=np.int64)

    def for_label(self, label: int) -> np.ndarray:
        """== torch.unique(active_unit[active_y == label])"""
        return self._by_label.get(int(label), self._EMPTY)

    def for_label_subject(self, label: int, subject: int) -> np.ndarray:
        """== torch.unique(active_unit[(active_y == label) & (active_subj == subject)])"""
        return self._by_label_subject.get((int(label), int(subject)), self._EMPTY)

    def count_excluding_subject(self, label: int, subject: int) -> int:
        """== torch.unique(active_unit[(active_y == label) & (active_subj != subject)]).numel()"""
        return (
            len(self.for_label(label))
            - self._sole_subject_counts.get((int(label), int(subject)), 0)
        )


# The tables are a pure function of (bank, active index), but the sampler asks for them roughly 23
# times per optimizer step while `ACTIVE_REFRESH_STEPS` moves the active index once every few steps,
# so rebuilding per call still redid ~11 ms of sorting many times more often than its inputs changed.
# Keyed on the exact row bytes, so a refreshed or reordered index can never read a stale table; the
# bank is held in the entry so its `id` cannot be recycled under us. Several entries are retained
# because training, validation and the training canaries each carry their own active index.
_ACTIVE_SUPPORT_UNITS_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_ACTIVE_SUPPORT_UNITS_CACHE_SIZE = 4


class QuerySupportPool:
    """CPU grouping of an immutable query pool by label and subject-support identity."""

    __slots__ = ("pool", "by_label")

    def __init__(self, bank: dict, pool: torch.Tensor, unit_offset: int):
        pool_host = pool.detach().cpu().long()
        window_y = torch.as_tensor(bank["y"], dtype=torch.long)
        window_subj = torch.as_tensor(bank["subj"], dtype=torch.long)
        window_event = torch.as_tensor(bank["event"], dtype=torch.long)
        window_verified = torch.as_tensor(bank["event_verified"], dtype=torch.bool)
        pool_unit = torch.where(
            window_verified[pool_host], window_event[pool_host] + unit_offset, pool_host
        )
        pool_y = window_y[pool_host]
        pool_subj = window_subj[pool_host]
        self.pool = pool
        self.by_label: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        order = torch.argsort(pool_y, stable=True)
        labels, counts = torch.unique_consecutive(pool_y[order], return_counts=True)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        for position, label in enumerate(labels.tolist()):
            member = order[int(offsets[position]):int(offsets[position + 1])]
            self.by_label[int(label)] = (
                pool_host[member], pool_subj[member], pool_unit[member]
            )


_QUERY_SUPPORT_POOL_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_QUERY_SUPPORT_POOL_CACHE_SIZE = 8


def query_support_pool(bank: dict, pool: torch.Tensor, unit_offset: int) -> QuerySupportPool:
    """Return immutable host-side query groupings without copying the full pool every episode."""
    key = (id(bank), pool.device.type, int(pool.data_ptr()), int(pool.numel()), unit_offset)
    cached = _QUERY_SUPPORT_POOL_CACHE.get(key)
    if cached is not None and cached[1].pool is pool:
        _QUERY_SUPPORT_POOL_CACHE.move_to_end(key)
        return cached[1]
    grouped = QuerySupportPool(bank, pool, unit_offset)
    _QUERY_SUPPORT_POOL_CACHE[key] = (bank, grouped)
    while len(_QUERY_SUPPORT_POOL_CACHE) > _QUERY_SUPPORT_POOL_CACHE_SIZE:
        _QUERY_SUPPORT_POOL_CACHE.popitem(last=False)
    return grouped


def active_support_units(bank: dict, index_rows: torch.Tensor) -> tuple[int, ActiveSupportUnits]:
    """Return ``(unit_offset, tables)`` for this active index, rebuilding only when it changes."""
    rows = index_rows.detach().cpu().long()
    key = (id(bank), rows.numpy().tobytes())
    cached = _ACTIVE_SUPPORT_UNITS_CACHE.get(key)
    if cached is not None:
        _ACTIVE_SUPPORT_UNITS_CACHE.move_to_end(key)
        return cached[1], cached[2]

    patch = bank["patch"]
    active_y = torch.as_tensor(patch["y"])[rows].long()
    active_subj = torch.as_tensor(patch["subj"])[rows].long()
    active_window = torch.as_tensor(patch["window"])[rows].long()
    active_event = torch.as_tensor(patch["event"])[rows].long()
    active_verified = torch.as_tensor(patch["event_verified"])[rows].bool()
    unit_offset = int(torch.as_tensor(patch["window"]).max()) + 1
    active_unit = torch.where(active_verified, active_event + unit_offset, active_window)
    tables = ActiveSupportUnits(active_y, active_subj, active_unit)

    _ACTIVE_SUPPORT_UNITS_CACHE[key] = (bank, unit_offset, tables)
    while len(_ACTIVE_SUPPORT_UNITS_CACHE) > _ACTIVE_SUPPORT_UNITS_CACHE_SIZE:
        _ACTIVE_SUPPORT_UNITS_CACHE.popitem(last=False)
    return unit_offset, tables


def prepare_support_feasible_query_pool(
    pool: torch.Tensor,
    index_rows: torch.Tensor,
    bank: dict,
    labels: torch.Tensor,
    *,
    support_count: int,
    episode_type: str,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reserve support identities before drawing a shared episode query batch.

    A Phase-B support overlay is shared by every query in an episode. Sampling many query
    executions first can therefore exclude more than the single unit assumed by a ``k + 1``
    capacity check. Ordinary episodes reserve ``k`` active support units and remove those identities
    from the query pool. Same-subject episodes additionally choose one real subject per candidate
    and reserve support from that person. Cross-subject episodes choose one query subject per label
    only when at least ``k`` units remain on other subjects.
    """
    if support_count == 0:
        return labels, pool
    device = pool.device
    # Host-side index bookkeeping, memoized on the active index — see `active_support_units`.
    unit_offset, active_units = active_support_units(bank, index_rows)
    grouped_pool = query_support_pool(bank, pool, unit_offset)

    feasible_labels = []
    query_parts = []
    for label in labels.tolist():
        grouped = grouped_pool.by_label.get(int(label))
        if grouped is None:
            continue
        label_pool, label_pool_subj, label_pool_units = grouped
        if episode_type == "cross_subject_few_support":
            viable_subjects = []
            for subject in torch.unique(label_pool_subj).tolist():
                if active_units.count_excluding_subject(label, subject) >= support_count:
                    viable_subjects.append(int(subject))
            if not viable_subjects:
                continue
            query_subject = int(rng.choice(np.asarray(viable_subjects)))
            query_part = label_pool[label_pool_subj.eq(query_subject)]
        elif episode_type == "same_subject_enrollment":
            viable = []
            for subject in torch.unique(label_pool_subj).tolist():
                subject_mask = label_pool_subj.eq(subject)
                subject_pool = label_pool[subject_mask]
                units = active_units.for_label_subject(label, subject)
                if len(units) < support_count:
                    continue
                reserved = torch.as_tensor(
                    rng.choice(units, size=support_count, replace=False),
                    dtype=torch.long,
                )
                query_part = subject_pool[
                    ~torch.isin(label_pool_units[subject_mask], reserved)
                ]
                if len(query_part):
                    viable.append((int(subject), query_part))
            if not viable:
                continue
            _, query_part = viable[int(rng.integers(len(viable)))]
        else:
            units = active_units.for_label(label)
            if len(units) < support_count:
                continue
            reserved = torch.as_tensor(
                rng.choice(units, size=support_count, replace=False),
                dtype=torch.long,
            )
            query_part = label_pool[~torch.isin(label_pool_units, reserved)]
        if len(query_part):
            feasible_labels.append(label)
            query_parts.append(query_part)

    if not feasible_labels:
        return labels[:0], pool[:0]
    return (
        torch.tensor(feasible_labels, device=device, dtype=torch.long),
        torch.cat(query_parts).to(device=device, dtype=pool.dtype),
    )


def support_feasible_labels(
    pool: torch.Tensor,
    index_rows: torch.Tensor,
    bank: dict,
    labels: torch.Tensor,
    *,
    support_count: int,
    episode_type: str,
) -> torch.Tensor:
    """Filter labels by whether at least one valid support/query split exists.

    Unlike ``prepare_support_feasible_query_pool``, this predicate consumes no RNG and constructs no
    episode query pool. Candidate selection uses it once; the randomized support reservation is then
    performed exactly once for the selected candidates.
    """
    if support_count == 0:
        return labels
    unit_offset, active_units = active_support_units(bank, index_rows)
    grouped_pool = query_support_pool(bank, pool, unit_offset)
    feasible = []
    for label in labels.detach().cpu().tolist():
        grouped = grouped_pool.by_label.get(int(label))
        if grouped is None:
            continue
        _rows, subjects, units = grouped
        if episode_type == "cross_subject_few_support":
            valid = any(
                active_units.count_excluding_subject(label, subject) >= support_count
                for subject in torch.unique(subjects).tolist()
            )
        elif episode_type == "same_subject_enrollment":
            valid = False
            for subject in torch.unique(subjects).tolist():
                member = subjects.eq(subject)
                support_units = active_units.for_label_subject(label, subject)
                query_units = torch.unique(units[member]).numpy()
                if len(support_units) >= support_count and (
                    len(support_units) > support_count
                    or bool(np.isin(query_units, support_units, invert=True).any())
                ):
                    valid = True
                    break
        else:
            support_units = active_units.for_label(label)
            query_units = torch.unique(units).numpy()
            valid = (
                len(support_units) >= support_count
                and (
                    len(support_units) > support_count
                    or bool(np.isin(query_units, support_units, invert=True).any())
                )
            )
        if valid:
            feasible.append(int(label))
    return torch.tensor(feasible, dtype=torch.long, device=labels.device)


def physical_label_centroids(Z: torch.Tensor, y: torch.Tensor, n_vocab: int) -> torch.Tensor:
    centroids = Z.new_zeros(n_vocab, Z.shape[1])
    centroids.index_add_(0, y, Z)
    counts = torch.bincount(y, minlength=n_vocab).to(Z.dtype).unsqueeze(1)
    return F.normalize(centroids / counts.clamp_min(1), dim=-1)


def family_holdout_labels(
    family_ids: torch.Tensor,
    labels_present_in_val: torch.Tensor,
    n_families: int,
) -> torch.Tensor:
    """Reserve complete canonical motion families that are actually represented in validation."""
    candidates = []
    present = set(labels_present_in_val.detach().cpu().tolist())
    for family in torch.unique(family_ids).tolist():
        members = torch.nonzero(family_ids.eq(family), as_tuple=True)[0]
        represented = [label for label in members.tolist() if label in present]
        if len(represented) >= 2:
            candidates.append((family, represented))
    if len(candidates) < n_families:
        raise ValueError("not enough complete activity families have two validation labels")
    # Deterministic: prefer the most represented families, then stable family id.
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    selected_ids = {family for family, _ in candidates[:n_families]}
    return torch.nonzero(
        torch.isin(family_ids, torch.tensor(sorted(selected_ids))), as_tuple=True
    )[0]


def build_decoder(cfg: dict):
    """Construct the sole supported Phase-B relational decoder."""
    return RelationalEvidenceDecoder(RelationalDecoderConfig(
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
    ))


# The exact tensor entries `training_episode_telemetry` reads. Listing them keeps the host copy
# narrow: everything else in `aux` (notably the attention maps and candidate states) stays put.
_TELEMETRY_TENSOR_KEYS = frozenset({
    "hard_logits",
    "evidence_mask",
    "evidence_local_index",
    "evidence_support",
    "evidence_support_candidate",
    "realized_true_support",
    "pool_weights",
    "evidence_label",
    "evidence_head",
    "evidence_window",
    "raw_retrieval_index",
    "raw_retrieval_valid",
})


def readout_telemetry(aux: dict) -> dict:
    """Telemetry for the sole supported relational readout."""
    result = {
        "candidate_logit_abs_mean": float(aux["logit_abs_mean"]),
        "candidate_logit_spread": float(aux["logit_spread"]),
        "background_label_tokens": float(aux["n_label_tokens"]),
    }
    for name in (
        "retrieval_score_mean",
        "retrieval_score_within_row_std",
        "retrieval_attention_bias_abs_mean",
        "retrieval_attention_bias_abs_max",
        "candidate_attention_normalized_entropy",
        "candidate_to_candidate_attention_mass",
        "candidate_to_label_attention_mass",
        "candidate_to_query_attention_mass",
        "candidate_to_evidence_attention_mass",
    ):
        if name in aux:
            result[name] = float(aux[name])
    return result


def _same_as_any_query_patch(ev_value, q_value, q_mask):
    """(B,M) True where an evidence row matches ANY valid query patch on this attribute.

    A query window can span several sensors and, after event expansion, several source rows, so
    "same configuration as the query" is a property of the set, not of an arbitrary first patch.
    """
    return ((ev_value.unsqueeze(1) == q_value.unsqueeze(2)) & q_mask.unsqueeze(2)).any(1)


def relational_decode(
    dec: RelationalEvidenceDecoder,
    *,
    query,
    evidence,
    ev_Z: torch.Tensor,
    ev_label_text: torch.Tensor,
    ev_y: torch.Tensor,
    ev_window: torch.Tensor,
    ev_cfg: torch.Tensor,
    ev_subj: torch.Tensor,
    ev_sensor: torch.Tensor,
    ev_time: torch.Tensor,
    ev_support: torch.Tensor,
    ev_support_candidate: torch.Tensor,
    candidate_text: torch.Tensor,
    canonical_text: torch.Tensor,
    generator: torch.Generator,
    score_temperature: float | None = None,
    return_attention: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Assemble the relational token set for one episode and read out candidate logits."""
    n_candidates = candidate_text.shape[-2]
    slots = build_coreference_slots(
        ev_y, evidence.mask, ev_support, ev_support_candidate, n_candidates,
        n_slots=dec.cfg.n_slots, generator=generator,
    )
    q_group, ev_group = build_window_groups(
        query.window, query.mask, ev_window, evidence.mask,
        n_groups=dec.cfg.n_groups, generator=generator,
    )
    logits, aux = dec(
        cand_text=candidate_text,
        label_text=canonical_text[slots.label_ids],
        label_mask=slots.label_mask,
        slot_ids=slots.slot_ids,
        zq=query.Z, q_mask=query.mask,
        zev=ev_Z, ev_mask=evidence.mask,
        ev_slot=slots.ev_slot, ev_support_mask=ev_support,
        ev_score=evidence.scores,
        score_temperature=score_temperature,
        q_time=query.time, ev_time=ev_time,
        q_group=q_group, ev_group=ev_group,
        ev_same_config=_same_as_any_query_patch(ev_cfg, query.cfg, query.mask),
        ev_same_subject=_same_as_any_query_patch(ev_subj, query.subj, query.mask),
        ev_same_sensor=_same_as_any_query_patch(ev_sensor, query.sensor, query.mask),
        return_aux=True,
        return_attention=return_attention,
    )
    # The confidence head consumes a candidate mass, a per-row vote and a pooling weight. The
    # model's own posterior IS the candidate mass; the per-row text vote survives only as an
    # agreement statistic. The head is fitted afterwards with the predictor frozen, so it
    # calibrates against these definitions.
    aux.update({
        "evidence": torch.softmax(logits, dim=1),
        "votes": label_text_votes(ev_label_text, candidate_text),
        "pool_weights": evidence.weights,
        "label_token_ids": slots.label_ids,
        "label_token_mask": slots.label_mask,
        "n_label_tokens": int(slots.label_mask.sum(1).max()),
    })
    return logits, aux


def _episode_view_specs(
    query,
    view: EpisodeMemoryView,
    index_rows: torch.Tensor,
    patch: dict,
    rng: np.random.Generator,
    *,
    physical_view_mode: str,
) -> tuple[list[PatchViewSpec], list[PatchViewSpec]]:
    """Build exact source views for the minimal clean-embedding recipe."""
    if physical_view_mode != "clean":
        raise ValueError("minimal Phase-B training supports only clean physical views")
    support_global = index_rows.detach().cpu()[view.support_rows.detach().cpu()]
    return (
        [PatchViewSpec() for _ in range(query.Z.shape[0])],
        [PatchViewSpec() for _ in range(len(support_global))],
    )


def prepare_adaptation_views(
    query,
    view: EpisodeMemoryView,
    bank: dict,
    index_rows: torch.Tensor,
    selector_z: torch.Tensor,
    memory_index: torch.Tensor,
    retriever: PatchSubspaceRetriever,
    *,
    rng: np.random.Generator,
    live_source: SourcePatchEncoder | None,
    selector_encoder=None,
    online_encoder=None,
    online_requires_grad: bool = False,
    physical_view_mode: str = "clean",
    reuse_stored_clean: bool = False,
    view_specs: tuple[list[PatchViewSpec], list[PatchViewSpec]] | None = None,
    encoded_query: torch.Tensor | None = None,
    encoded_support: torch.Tensor | None = None,
):
    """Create episode-specific query/support vectors and selector keys."""
    if live_source is None:
        return query, query, selector_z, memory_index
    if physical_view_mode == "clean" and reuse_stored_clean and not online_requires_grad:
        # Stored fp16 vectors are the clean frozen-encoder path. Re-encoding them only reproduces
        # the same representation and needlessly serializes CPU augmentation/loading with the GPU.
        return query, query, selector_z, memory_index
    query_specs, support_specs = (
        _episode_view_specs(
            query, view, index_rows, bank["patch"], rng,
            physical_view_mode=physical_view_mode,
        ) if view_specs is None else view_specs
    )
    if len(query_specs) != query.Z.shape[0] or len(support_specs) != len(view.support_rows):
        raise ValueError("precomputed physical-view specs do not align with query/support rows")
    selector_query = (
        live_source.reencode_query_views(
            query, selector_encoder, query_specs, requires_grad=False
        )
        if encoded_query is None else
        live_source.query_with_embeddings(query, encoded_query)
    )
    online_model = online_encoder if online_encoder is not None else selector_encoder
    online_query = live_source.reencode_query_views(
        query, online_model, query_specs, requires_grad=online_requires_grad
    ) if online_requires_grad else selector_query

    memory_online = selector_z
    selector_episode_index = memory_index
    support_local = view.support_rows
    if len(support_local):
        support_global = index_rows.detach().cpu()[support_local.detach().cpu()]
        support_selector = (
            live_source.encode_patch_rows_with_views(
                support_global, support_specs, selector_encoder, requires_grad=False
            )
            if encoded_support is None else encoded_support
        )
        support_online = (
            live_source.encode_patch_rows_with_views(
                support_global, support_specs, online_model, requires_grad=online_requires_grad
            )
            if online_requires_grad else support_selector
        )
        memory_online = selector_z.index_copy(
            0, support_local.to(selector_z.device), support_online.to(selector_z)
        )
        replacement_index = retriever.project(
            F.normalize(support_selector.to(selector_z), dim=-1), ema=True
        ).detach()
        selector_episode_index = memory_index.index_copy(
            0, support_local.to(memory_index.device), replacement_index
        )
    return selector_query, online_query, memory_online, selector_episode_index


def encode_frozen_adaptation_views_batch(
    episodes: list[tuple],
    index_rows: torch.Tensor,
    live_source: SourcePatchEncoder,
    selector_encoder,
) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
    """Encode requested live frozen-tokenizer views in one stream-grouped pass.

    Episode semantics remain independent. This only flattens the raw query/support occurrences
    before ``SourcePatchEncoder`` groups them by acquisition stream, which turns dozens of tiny GPU
    encoder launches into one useful batch per represented stream. Identical
    ``(source window, PatchViewSpec)`` keys are encoded once.
    """
    flat_rows: list[torch.Tensor] = []
    flat_specs: list[PatchViewSpec] = []
    spans: dict[int, tuple[slice, slice]] = {}
    cursor = 0
    for episode_index, (query, view, view_specs, physical_view_mode) in enumerate(episodes):
        if physical_view_mode == "clean":
            continue
        if view_specs is None:
            raise ValueError("batched frozen views require precomputed physical-view specs")
        query_specs, support_specs = view_specs
        query_rows = query.row[query.mask].detach().cpu().long()
        if bool((query_rows < 0).any()):
            raise ValueError("external queries cannot be batched from the source bank")
        query_occurrence_specs = live_source.query_occurrence_view_specs(query, query_specs)
        query_slice = slice(cursor, cursor + len(query_rows))
        cursor = query_slice.stop
        flat_rows.append(query_rows)
        flat_specs.extend(query_occurrence_specs)

        support_local = view.support_rows.detach().cpu()
        support_rows = index_rows.detach().cpu()[support_local].long()
        support_slice = slice(cursor, cursor + len(support_rows))
        cursor = support_slice.stop
        if len(support_rows):
            flat_rows.append(support_rows)
            flat_specs.extend(support_specs)
        spans[episode_index] = (query_slice, support_slice)

    if not flat_rows:
        return [None] * len(episodes)
    rows = torch.cat(flat_rows)
    if len(rows) != len(flat_specs):
        raise RuntimeError("flattened Phase-B view rows and augmentation specs do not align")
    encoded = live_source.encode_patch_rows_with_views(
        rows, flat_specs, selector_encoder, requires_grad=False
    )

    result: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(episodes)
    for episode_index in spans:
        query_slice, support_slice = spans[episode_index]
        result[episode_index] = (encoded[query_slice], encoded[support_slice])
    return result


def prepare_frozen_adaptation_views_batch(
    episodes: list[tuple],
    bank: dict,
    index_rows: torch.Tensor,
    selector_z: torch.Tensor,
    memory_index: torch.Tensor,
    retriever: PatchSubspaceRetriever,
    live_source: SourcePatchEncoder,
    selector_encoder,
    *,
    encoded_views: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
) -> list[tuple | None]:
    """Project one batched set of frozen view embeddings into episode-local memory views."""
    if encoded_views is None:
        encoded_views = encode_frozen_adaptation_views_batch(
            episodes, index_rows, live_source, selector_encoder
        )
    if len(encoded_views) != len(episodes):
        raise ValueError("encoded frozen views do not align with adaptation episodes")
    result: list[tuple | None] = [None] * len(episodes)
    for episode_index, (query, view, view_specs, physical_view_mode) in enumerate(episodes):
        if physical_view_mode == "clean":
            continue
        encoded = encoded_views[episode_index]
        if encoded is None:
            raise ValueError("live episode view is missing its frozen embeddings")
        encoded_query, encoded_support = encoded
        result[episode_index] = prepare_adaptation_views(
            query, view, bank, index_rows, selector_z, memory_index, retriever,
            rng=np.random.default_rng(0),
            live_source=live_source,
            selector_encoder=selector_encoder,
            online_encoder=None,
            online_requires_grad=False,
            physical_view_mode=physical_view_mode,
            reuse_stored_clean=True,
            view_specs=view_specs,
            encoded_query=encoded_query,
            encoded_support=encoded_support,
        )
    return result


def decode_adaptation_episode(
    dec: RelationalEvidenceDecoder,
    retriever: PatchSubspaceRetriever,
    bank: dict,
    index_rows: torch.Tensor,
    selector_z: torch.Tensor,
    memory_index: torch.Tensor,
    query,
    view: EpisodeMemoryView,
    canonical_text: torch.Tensor,
    candidate_text: torch.Tensor,
    *,
    policy: PhaseBPolicy,
    rng: np.random.Generator,
    live_source: SourcePatchEncoder | None = None,
    selector_encoder=None,
    online_encoder=None,
    online_requires_grad: bool = False,
    physical_view_mode: str = "clean",
    evidence_budget: int | None = None,
    score_temperature: float | None = None,
    view_specs: tuple[list[PatchViewSpec], list[PatchViewSpec]] | None = None,
    prepared_views: tuple | None = None,
    return_attention: bool = False,
    return_confidence_features: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Retrieve with learned query keys and apply the relational evidence decoder."""
    device = next(dec.parameters()).device
    patch = bank["patch"]
    selector_query, online_query, memory_online, selector_episode_index = (
        prepare_adaptation_views(
            query, view, bank, index_rows, selector_z, memory_index, retriever,
            rng=rng, live_source=live_source, selector_encoder=selector_encoder,
            online_encoder=online_encoder, online_requires_grad=online_requires_grad,
            physical_view_mode=physical_view_mode,
            reuse_stored_clean=policy.tokenizer_mode == "frozen",
            view_specs=view_specs,
        )
        if prepared_views is None else prepared_views
    )
    selector_query.Z = F.normalize(selector_query.Z, dim=-1)
    online_query.Z = F.normalize(online_query.Z, dim=-1)
    memory_online = F.normalize(memory_online, dim=-1)
    effective_evidence_budget = (
        policy.evidence_budget if evidence_budget is None else int(evidence_budget)
    )
    topk = policy.topk_per_subspace(
        int(query.mask.sum(1).max()), evidence_budget=effective_evidence_budget
    )
    retrieval = retriever.retrieve(
        selector_query.Z, selector_episode_index, view.allowed, topk,
        query_mask=query.mask,
    )
    online_score = retriever.score_selected(
        online_query.Z, memory_online, retrieval.index
    )
    evidence = assemble_evidence(
        retrieval, online_score, index_rows,
        max_evidence=effective_evidence_budget,
        tau=RETRIEVAL_TEMPERATURE if score_temperature is None else score_temperature,
    )
    if evidence.local_index is None:
        raise RuntimeError("assembled evidence omitted active-index row identities")
    ev_Z = memory_online[evidence.local_index]
    ev_idx = evidence.index
    ev_support = view.support_mask[evidence.local_index]
    if online_requires_grad and live_source is not None and online_encoder is not None:
        live_selected = F.normalize(
            live_source.reencode_evidence(
                evidence, online_encoder, requires_grad=True
            ),
            dim=-1,
        )
        # Provided support keeps its episode-specific clean view; selected background rows use a
        # fresh live forward so optional tokenizer fine-tuning receives the hard-path gradient.
        ev_Z = torch.where(ev_support.unsqueeze(-1), ev_Z, live_selected)

    def ev_field(name, dtype=None):
        value = torch.as_tensor(patch[name])[ev_idx.detach().cpu()].to(device)
        return value.to(dtype=dtype) if dtype is not None else value

    ev_y = ev_field("y", torch.long)
    ev_window = ev_field("window", torch.long)
    ev_cfg = ev_field("cfg", torch.long)
    ev_sensor = ev_field("sensor", torch.long)
    # Keep the raw binding: -1 means "not support". `build_coreference_slots` needs the sentinel to
    # tell a genuine candidate 0 from an unbound row, and clamping here would hide that.
    ev_support_binding = view.support_candidate[evidence.local_index]
    ev_support_position = ev_support_binding.clamp_min(0)
    ev_label_text = canonical_text[ev_y].clone()
    if bool(ev_support.any()):
        ev_label_text = torch.where(
            ev_support.unsqueeze(-1),
            candidate_text[ev_support_position],
            ev_label_text,
        )
    hard_logits, aux = relational_decode(
        dec,
        query=online_query, evidence=evidence, ev_Z=ev_Z,
        ev_label_text=ev_label_text, ev_y=ev_y, ev_window=ev_window,
        ev_cfg=ev_cfg, ev_subj=ev_field("subj", torch.long), ev_sensor=ev_sensor,
        ev_time=ev_field("time", torch.float32),
        ev_support=ev_support, ev_support_candidate=ev_support_binding,
        candidate_text=candidate_text, canonical_text=canonical_text,
        generator=torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1))),
        score_temperature=score_temperature,
        return_attention=return_attention,
    )
    identity_logits = retrieval_vote_base(
        ev_label_text, candidate_text, evidence.weights, scale=RETRIEVAL_VOTE_SCALE
    )
    if return_confidence_features:
        # Confidence calibration is parked and runs as a separate frozen-predictor stage. Avoid its
        # candidate-wise Python/device work on every predictor train/eval call.
        from model.evidence.confidence import confidence_features
        aux["confidence_features"] = confidence_features(
            aux["evidence"], evidence.scores, aux["votes"], aux["pool_weights"],
            ev_mask=evidence.mask, ev_sensor_id=ev_sensor,
        )
    logits = hard_logits

    aux.update({
        "hard_logits": hard_logits,
        "retrieval_scores": evidence.scores,
        "retrieval_prior": evidence.weights,
        "raw_retrieval_index": retrieval.index,
        "raw_retrieval_valid": retrieval.valid,
        "evidence_mask": evidence.mask,
        "evidence_index": evidence.index,
        "evidence_local_index": evidence.local_index,
        "evidence_window": ev_window,
        "evidence_label": ev_y,
        "evidence_support": ev_support,
        "evidence_support_candidate": ev_support_position,
        "evidence_head": evidence.head,
        "candidate_ids": view.candidate_ids,
        "query_label": view.query_label,
        "realized_true_support": view.support_units_per_candidate[
            label_index(view.candidate_ids, int(canonical_text.shape[0]), device)[view.query_label]
        ],
        "retrieval_topk": topk,
        "evidence_budget": effective_evidence_budget,
        "score_temperature": (
            dec.cfg.score_temperature if score_temperature is None else score_temperature
        ),
        "identity_logits": identity_logits,
    })
    return logits, aux


def training_episode_telemetry(
    *,
    spec: AdaptationEpisodeSpec,
    candidates: torch.Tensor,
    target: torch.Tensor,
    true_label: torch.Tensor,
    aux: dict,
    candidate_loss: torch.Tensor,
    normalized_candidate_loss: torch.Tensor,
    index_size: int,
    draw_attempts: int,
    composition: dict,
    detailed: bool,
    retrieval_score_gradient: torch.Tensor | None = None,
) -> tuple[dict, dict, dict]:
    """Detach one independent episode into scalar telemetry and sampling categories."""
    categories = {
        "episode_type": spec.episode_type,
        "label_mode": spec.label_mode,
        "physical_view_mode": spec.physical_view_mode,
        "enrollment_shape": spec.enrollment_shape,
        "support_count": str(spec.support_count),
        "candidate_count": str(len(candidates)),
        "target_position": target.detach().cpu().tolist(),
    }
    strata = {key: categories[key] for key in (
        "episode_type", "label_mode", "physical_view_mode", "enrollment_shape",
        "support_count", "candidate_count",
    )}
    if not detailed:
        # Full evidence/attention telemetry is published on the minute cadence. In between, one
        # compact transfer preserves continuous objective and episode health without copying
        # several evidence tensors to the host eight times per optimizer step.
        chance = 1.0 / len(candidates)
        compact = torch.stack([
            candidate_loss.detach(),
            normalized_candidate_loss.detach(),
            aux["hard_logits"].detach().argmax(1).eq(target).float().mean(),
            aux["realized_true_support"].detach().float().mean(),
        ]).cpu().tolist()
        metrics = {
            "candidate_loss": compact[0],
            "loss_over_random": compact[1],
            "loss_improvement_over_random": 1.0 - compact[1],
            "train_accuracy": compact[2],
            "chance_normalized_train_accuracy": (compact[2] - chance) / (1.0 - chance),
            "episode_draw_attempts": float(draw_attempts),
            "realized_true_support": compact[3],
        }
        return metrics, categories, strata

    with torch.no_grad():
        # Roughly twenty-five `float(...)` conversions follow, and on CUDA each one is a device
        # synchronization. Bring the tensors this function reads across once instead. Only the
        # keys read below are copied, so the attention maps stay on the device when unused. The
        # values are identical; this changes when the transfer happens, not what is computed.
        aux = {
            key: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            for key, value in aux.items()
            if not isinstance(value, torch.Tensor) or key in _TELEMETRY_TENSOR_KEYS
        }
        target = target.detach().cpu()
        true_label = true_label.detach().cpu()
        if retrieval_score_gradient is not None:
            retrieval_score_gradient = retrieval_score_gradient.detach().cpu()
        probability = torch.softmax(aux["hard_logits"], dim=1)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(1)
        entropy = entropy / max(np.log(len(candidates)), 1e-8)
        top2 = probability.topk(min(2, len(candidates)), dim=1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0)
        accuracy = float(aux["hard_logits"].argmax(1).eq(target).float().mean())
        chance = 1.0 / len(candidates)

        evidence_mask = aux["evidence_mask"]
        selected = aux["evidence_local_index"][evidence_mask]
        selected_counts = torch.bincount(selected, minlength=index_size).float()
        selected_concentration = float(
            selected_counts.max() / selected_counts.sum().clamp_min(1)
        )
        selected_true_support = (
            aux["evidence_support"]
            & aux["evidence_support_candidate"].eq(target.unsqueeze(1))
            & evidence_mask
        )
        enrolled_query = aux["realized_true_support"].gt(0)
        support_recall = (
            float(selected_true_support.any(1)[enrolled_query].float().mean())
            if bool(enrolled_query.any()) else None
        )

        pool_weights = aux["pool_weights"].masked_fill(~evidence_mask, 0.0)
        pool_entropy = -(
            pool_weights * pool_weights.clamp_min(1e-12).log()
        ).sum(1)
        pool_valid = evidence_mask.sum(1).clamp_min(2).float()
        support_pool_mass = (
            pool_weights * aux["evidence_support"].to(pool_weights.dtype)
        ).sum(1)
        true_label_pool_mass = (
            pool_weights
            * aux["evidence_label"].eq(true_label.unsqueeze(1)).to(pool_weights.dtype)
        ).sum(1)
        target_hist = torch.bincount(target, minlength=len(candidates)).float()
        head_mass = torch.zeros(RETRIEVAL_SUBSPACES, device=pool_weights.device)
        head_mass.scatter_add_(
            0, aux["evidence_head"][evidence_mask], pool_weights[evidence_mask]
        )
        head_mass = head_mass / head_mass.sum().clamp_min(1e-8)
        unique_windows = float(np.mean([
            torch.unique(aux["evidence_window"][b, row]).numel()
            for b, row in enumerate(evidence_mask)
        ]))
        unique_labels = float(np.mean([
            torch.unique(aux["evidence_label"][b, row]).numel()
            for b, row in enumerate(evidence_mask)
        ]))

        metrics = {
            "candidate_loss": float(candidate_loss),
            "loss_over_random": float(normalized_candidate_loss),
            "loss_improvement_over_random": 1.0 - float(normalized_candidate_loss),
            "train_accuracy": accuracy,
            "chance_normalized_train_accuracy": (accuracy - chance) / (1.0 - chance),
            "selected_row_max_share": selected_concentration,
            "provided_support_recall_at_k": support_recall,
            "provided_support_pool_mass": float(support_pool_mass.mean()),
            "true_label_pool_mass": float(true_label_pool_mass.mean()),
            "enrolled_query_fraction": float(enrolled_query.float().mean()),
            "pool_normalized_entropy": float((pool_entropy / pool_valid.log()).mean()),
            "pool_effective_evidence": float(pool_entropy.exp().mean()),
            "pool_weight_max_share": float(pool_weights.max(1).values.mean()),
            "candidate_normalized_entropy": float(entropy.mean()),
            "candidate_top1_margin": float(margin.mean()),
            "target_position_max_share": float(target_hist.max() / target_hist.sum()),
            "unique_evidence_windows": unique_windows,
            "unique_evidence_labels": unique_labels,
            "episode_draw_attempts": float(draw_attempts),
            "realized_true_support": float(aux["realized_true_support"].float().mean()),
            **readout_telemetry(aux),
            **{
                f"subspace_{head}_mass": float(value)
                for head, value in enumerate(head_mass)
            },
            **{
                f"enrolled_{name}": float(value)
                for name, value in composition.items() if isinstance(value, int)
            },
        }
        if retrieval_score_gradient is not None:
            score_gradient = retrieval_score_gradient.detach()
            valid_gradient = score_gradient[evidence_mask]
            if valid_gradient.numel():
                promoted = score_gradient < 0
                support_selected = selected_true_support
                background_selected = evidence_mask & ~support_selected
                metrics.update({
                    "selected_score_task_grad_abs_mean": float(valid_gradient.abs().mean()),
                    "selected_score_task_promoted_fraction": float(
                        promoted[evidence_mask].float().mean()
                    ),
                    "true_support_task_promoted_fraction": (
                        float(promoted[support_selected].float().mean())
                        if bool(support_selected.any()) else None
                    ),
                    "background_task_promoted_fraction": (
                        float(promoted[background_selected].float().mean())
                        if bool(background_selected.any()) else None
                    ),
                    "background_task_promotion_share": float(
                        (-score_gradient).clamp_min(0)[background_selected].sum()
                        / (-score_gradient).clamp_min(0)[evidence_mask].sum().clamp_min(1e-12)
                    ),
                    "true_support_score_task_grad_mean": (
                        float(score_gradient[support_selected].mean())
                        if bool(support_selected.any()) else None
                    ),
                    "background_score_task_grad_mean": (
                        float(score_gradient[background_selected].mean())
                        if bool(background_selected.any()) else None
                    ),
                    "true_support_score_task_grad_abs_mean": (
                        float(score_gradient[support_selected].abs().mean())
                        if bool(support_selected.any()) else None
                    ),
                    "background_score_task_grad_abs_mean": (
                        float(score_gradient[background_selected].abs().mean())
                        if bool(background_selected.any()) else None
                    ),
                })
        if detailed:
            overlap_values = []
            raw_index = aux["raw_retrieval_index"].detach().cpu()
            raw_valid = aux["raw_retrieval_valid"].detach().cpu()
            for batch_index in range(raw_index.shape[0]):
                for query_patch in range(raw_index.shape[1]):
                    for left in range(RETRIEVAL_SUBSPACES):
                        left_rows = torch.unique(raw_index[batch_index, query_patch, left][
                            raw_valid[batch_index, query_patch, left]
                        ])
                        for right in range(left + 1, RETRIEVAL_SUBSPACES):
                            right_rows = torch.unique(raw_index[batch_index, query_patch, right][
                                raw_valid[batch_index, query_patch, right]
                            ])
                            union = torch.unique(torch.cat([left_rows, right_rows])).numel()
                            if union:
                                overlap_values.append(
                                    float(torch.isin(left_rows, right_rows).sum()) / union
                                )
            if overlap_values:
                metrics["raw_subspace_topk_jaccard"] = float(np.mean(overlap_values))

    return metrics, categories, strata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=_DEFAULT_BANK)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Phase-A checkpoint; required by --tokenizer-mode ema_finetune")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--episodes-per-step", type=int, default=EPISODES_PER_STEP,
                    help="independently randomized adaptation episodes per optimizer step")
    ap.add_argument("--queries-per-episode", type=int, default=QUERIES_PER_EPISODE,
                    help="query executions sharing one episode's candidate/support context")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--grad-clip", type=float, default=20.0)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--evidence-budget", type=int, default=64,
                    help="sole retrieval-capacity knob; K and contribution caps are derived")
    ap.add_argument(
        "--retriever-init-predictor", type=Path, default=None,
        help="diagnostic arm: initialize retrieval from an existing predictor artifact",
    )
    ap.add_argument(
        "--freeze-retriever", action="store_true",
        help="diagnostic arm: train only the decoder while keeping retrieval projections fixed",
    )
    ap.add_argument("--tokenizer-mode", choices=("frozen", "ema_finetune"), default="frozen")
    ap.add_argument("--val-families", type=int, default=2,
                    help="complete canonical activity families excluded from every training role")
    ap.add_argument("--val-frac-cfg", type=float, default=0.2)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-episodes", type=int, default=15,
                    help="fixed clean C=8 canaries: coherent k=0 plus alias k=1/2/4/8, "
                         "across three transfer folds")
    ap.add_argument("--val-queries", type=int, default=32)
    ap.add_argument("--label-variants", type=int, default=0,
                    help="optional label-text augmentation; disabled in the minimal recipe")
    ap.add_argument("--telemetry-seconds", type=float, default=60.0)
    ap.add_argument("--telemetry-dir", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=200,
                    help="write a resumable trainer state every N optimizer steps")
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume from a .last.pt trainer state produced by this command")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="run a two-step frozen CPU integration test on a synthetic bank")
    ap.add_argument("--real-smoke", action="store_true",
                    help="run three frozen steps through the real bank, grids, and encoder")
    args = ap.parse_args()
    if args.smoke and args.real_smoke:
        ap.error("--smoke and --real-smoke are mutually exclusive")
    if args.smoke:
        if args.tokenizer_mode != "frozen":
            ap.error("synthetic --smoke supports frozen mode; live fine-tuning is tested on grids")
        args.device = "cpu"
        args.steps = 2
        args.episodes_per_step = 4
        args.queries_per_episode = 2
        args.evidence_budget = 8
        args.val_every = 1
        args.val_episodes = 3
        args.val_queries = 4
        args.val_frac_cfg = 0.5
        args.val_families = 1
        args.warmup_steps = 1
        args.telemetry_seconds = 0.01
        args.save_every = 1
        args.out = Path("/tmp/halo_phase_b_predictor_smoke.pt") if args.out == _DEFAULT_OUT else args.out
    if args.real_smoke:
        if args.tokenizer_mode != "frozen":
            ap.error("--real-smoke checks the default frozen launch path")
        args.steps = 3
        args.episodes_per_step = 4
        args.queries_per_episode = 2
        args.evidence_budget = 16
        args.val_every = 1
        args.val_episodes = 4
        args.val_queries = 8
        args.warmup_steps = 1
        args.save_every = 1
        args.telemetry_seconds = 0.01
        args.out = (
            Path("/tmp/halo_phase_b_predictor_real_smoke.pt")
            if args.out == _DEFAULT_OUT else args.out
        )
    if args.steps < 1 or args.queries_per_episode < 1 or args.val_episodes < 1 \
            or args.val_queries < 1 or args.val_every < 1 or args.save_every < 1:
        ap.error(
            "steps, queries-per-episode, validation sizes, val-every, and save-every "
            "must be positive"
        )
    if args.episodes_per_step < 1:
        ap.error("episodes-per-step must be positive")
    if args.val_episodes > 15:
        ap.error("val-episodes cannot exceed the declared 15-point canary grid")
    if args.warmup_steps < 0 or args.grad_clip <= 0:
        ap.error("warmup-steps must be nonnegative and grad-clip must be positive")
    if args.telemetry_seconds <= 0:
        ap.error("telemetry-seconds must be positive")
    if args.freeze_retriever and args.retriever_init_predictor is None:
        ap.error("--freeze-retriever requires --retriever-init-predictor")
    if args.tokenizer_mode == "ema_finetune" \
            and args.steps <= TOKENIZER_FINETUNE_WARMUP_STEPS:
        ap.error(
            "ema_finetune needs more than "
            f"{TOKENIZER_FINETUNE_WARMUP_STEPS} steps so the tokenizer is actually updated"
        )
    policy = PhaseBPolicy(args.evidence_budget, args.tokenizer_mode)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bank = synthetic_smoke_bank() if args.smoke else torch.load(
        args.bank, map_location="cpu", weights_only=True
    )
    if not args.smoke:
        assert_bank_current(bank, context="train_patch_decoder")
    assert_patch_bank(bank, context="train_patch_decoder")
    table = PatchTable(bank)
    Z = F.normalize(bank["Z"].float(), dim=-1).to(device)
    y = bank["y"].long().to(device)
    subj = bank["subj"].long().to(device)
    cfg = bank["cfg"].long().to(device)
    vocab = list(bank["vocab"])
    n_vocab, d = len(vocab), Z.shape[1]
    family_ids, activity_families, family_fp = load_activity_families(vocab)
    family_ids = family_ids.to(device)

    sbert = get_sbert_encoder()
    text = ensemble_text(vocab, sbert, 8, train_only=True).to(device)
    alias_embeddings = encode_neutral_aliases(sbert, device)
    variants = (
        build_label_variants(vocab, sbert, args.label_variants, train_only=True).to(device)
        if args.label_variants > 0 else None
    )
    text_gen = torch.Generator(device=device).manual_seed(args.seed)

    # Training occupies the non-held-subject/non-held-configuration quadrant. Held-family
    # validation uses every other quadrant, separately exposing subject, configuration, and joint
    # transfer instead of silently discarding the two off-diagonal cells.
    cfg_ids = torch.unique(cfg).cpu().numpy()
    rng.shuffle(cfg_ids)
    n_val_cfg = max(1, int(len(cfg_ids) * args.val_frac_cfg))
    val_cfg = torch.tensor(cfg_ids[:n_val_cfg], device=device)
    subjects = torch.unique(subj).cpu().numpy()
    rng.shuffle(subjects)
    n_val_subject = max(1, int(len(subjects) * args.val_frac_cfg))
    val_subject = torch.tensor(subjects[:n_val_subject], device=device)
    fold_masks = phase_b_fold_masks(cfg, subj, val_cfg, val_subject)
    raw_val_pool = torch.nonzero(fold_masks.validation, as_tuple=True)[0]
    base_train_pool = torch.nonzero(fold_masks.train_base, as_tuple=True)[0]
    if not len(raw_val_pool) or not len(base_train_pool):
        raise SystemExit("empty train/validation fold after config x subject split")
    represented_for_validation = torch.unique(y[raw_val_pool])
    represented_for_validation = represented_for_validation[
        torch.isin(represented_for_validation, torch.unique(y[base_train_pool]))
    ]
    heldout_labels = family_holdout_labels(
        family_ids.cpu(), represented_for_validation.cpu(), args.val_families
    )
    heldout_labels = heldout_labels.to(device)
    train_pool = base_train_pool[~torch.isin(y[base_train_pool], heldout_labels)]
    val_pool = raw_val_pool[torch.isin(y[raw_val_pool], heldout_labels)]
    if len(torch.unique(y[val_pool])) < 2 or len(torch.unique(y[train_pool])) < 3:
        raise SystemExit("activity-family holdout left too few train or validation labels")
    # Hard-distractor centroids may use only the non-validation fold. In particular, never compute
    # a held-out-family centroid from the query windows later used to select a checkpoint.
    physical = physical_label_centroids(
        Z[base_train_pool], y[base_train_pool], n_vocab
    )
    memory_window_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    memory_window_mask[train_pool] = True
    val_memory_window_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    val_memory_window_mask[base_train_pool] = True
    retriever = PatchSubspaceRetriever(
        d, RETRIEVAL_SUBSPACES, RETRIEVAL_SUBSPACE_DIM, RETRIEVAL_PROJECTION_EMA
    ).to(device)
    retriever_init = None
    if args.retriever_init_predictor is not None:
        retriever_init = torch.load(
            args.retriever_init_predictor, map_location="cpu", weights_only=True
        )
        expected_bank_fp = bank.get("bank_fp") or bank_fingerprint(bank)
        if retriever_init.get("bank_fp") != expected_bank_fp:
            raise SystemExit("retriever initializer was built against a different memory bank")
        initializer_cfg = retriever_init.get("cfg", {})
        expected_shape = (d, RETRIEVAL_SUBSPACES, RETRIEVAL_SUBSPACE_DIM)
        actual_shape = (
            int(initializer_cfg.get("d_model", -1)),
            int(initializer_cfg.get("n_subspaces", -1)),
            int(initializer_cfg.get("subspace_dim", -1)),
        )
        if actual_shape != expected_shape:
            raise SystemExit(
                f"retriever initializer shape {actual_shape} != required {expected_shape}"
            )
        retriever.load_state_dict(retriever_init["retriever"])
    if args.freeze_retriever:
        for parameter in retriever.parameters():
            parameter.requires_grad_(False)
    decoder_cfg = {
        "d_model": d, "n_layers": args.layers, "n_heads": args.heads,
    }
    dec = build_decoder(decoder_cfg).to(device)
    params = dec.param_groups(args.weight_decay)
    if not args.freeze_retriever:
        params.append({"params": retriever.parameters(), "weight_decay": args.weight_decay})
    live_source = online_encoder = ema_encoder = None
    checkpoint = None
    if not args.smoke:
        checkpoint_path = args.checkpoint or Path(bank["backbone"]["checkpoint"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"fine-tuning checkpoint not found at {checkpoint_path}; pass --checkpoint"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert_bank_matches_backbone(bank, checkpoint, context="train_patch_decoder fine-tune")
        checkpoint_fp = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_fp != bank["backbone"].get("fingerprint"):
            raise SystemExit("fine-tuning checkpoint is not the encoder that built the memory bank")
        ema_encoder = build_encoder(checkpoint, device, training=False)
        assert_embedding_path_current(
            bank, ema_encoder, device, context="train_patch_decoder fine-tune"
        )
        assert_patch_embedding_path_current(
            bank, ema_encoder, device, context="train_patch_decoder fine-tune"
        )
        live_source = SourcePatchEncoder(bank, device)
        for parameter in ema_encoder.parameters():
            parameter.requires_grad_(False)
        if policy.tokenizer_mode == "ema_finetune":
            online_encoder = build_encoder(checkpoint, device, training=True)
            # The frozen LM is lazy/non-stateful. Reuse the instance/cache warmed by the probes.
            online_encoder.text_encoder = ema_encoder.text_encoder
            # Phase B encodes complete patches and never executes the Phase-A masked-token path.
            # Exclude its learned mask vector from AdamW rather than silently decaying a parameter
            # that cannot receive a task gradient.
            online_encoder.mask_token.requires_grad_(False)
            params.append({
                "params": [p for p in online_encoder.parameters() if p.requires_grad],
                "weight_decay": args.weight_decay,
                "lr": args.lr * TOKENIZER_LR_SCALE,
            })
    opt = torch.optim.AdamW(params, lr=args.lr)

    def lr_factor(step_index: int) -> float:
        if args.warmup_steps and step_index < args.warmup_steps:
            return float(step_index + 1) / args.warmup_steps
        progress = (step_index - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    index_rows = table.sample_index_rows(
        memory_window_mask, ACTIVE_WINDOWS_PER_LABEL, np.random.default_rng(args.seed + 2)
    )
    val_index_rows = table.sample_index_rows(
        val_memory_window_mask, ACTIVE_WINDOWS_PER_LABEL, np.random.default_rng(args.seed + 3)
    )
    # Which streams were captured at the same instant. Used only to describe what an episode drew;
    # it never gates candidate selection.
    simultaneous_pairs = simultaneous_stream_pairs(bank["cfg"], bank["event"])
    selector_z = F.normalize(
        torch.as_tensor(bank["patch"]["Z"])[index_rows].float().to(device), dim=-1
    )
    memory_index = retriever.build_index(selector_z)
    episode_sampler = EpisodeSampler(rng)

    def build_selector_index(rows, *, encoder=None):
        if encoder is None:
            values = torch.as_tensor(bank["patch"]["Z"])[rows].float().to(device)
        else:
            values = live_source.encode_patch_rows(
                rows, encoder, requires_grad=False
            ).detach().to(device)
        return retriever.build_index(F.normalize(values, dim=-1))

    def make_adaptation_episode(
        pool,
        rows,
        spec: AdaptationEpisodeSpec,
        *,
        count,
        local_rng,
        validation=False,
    ):
        present = torch.unique(y[pool])
        capacity = support_capacity_by_label(bank["patch"], rows, n_vocab).to(device)
        required = spec.support_count + (0 if validation else 1)
        present = present[capacity[present] >= required]
        present = support_feasible_labels(
            pool,
            rows,
            bank,
            present,
            support_count=spec.support_count,
            episode_type=spec.episode_type,
        )
        allowed_vocab = present if validation else present[
            ~torch.isin(present, heldout_labels)
        ]
        candidate_count = min(spec.candidate_count, len(allowed_vocab))
        if candidate_count < 2:
            raise ValueError(
                "no eligible support-feasible labels for an adaptation episode: "
                f"type={spec.episode_type}, support={spec.support_count}, "
                f"eligible_labels={len(allowed_vocab)}"
            )
        seed_count = min(2, candidate_count - 1)
        seed = local_rng.choice(
            allowed_vocab.detach().cpu().numpy(), size=seed_count, replace=False
        )
        seed_labels = torch.as_tensor(seed, device=device, dtype=torch.long)
        candidates = choose_candidates(
            seed_labels,
            candidate_count,
            n_vocab,
            text,
            physical,
            truth_present=True,
            rng=local_rng,
            allowed_vocab=allowed_vocab,
            hard_fraction=0.0,
        )
        # choose_candidates may have fewer rows when the represented pool is tiny.
        candidates = candidates[torch.isin(candidates, allowed_vocab)]
        feasible_candidates, episode_query_pool = prepare_support_feasible_query_pool(
            pool,
            rows,
            bank,
            candidates,
            support_count=spec.support_count,
            episode_type=spec.episode_type,
            rng=local_rng,
        )
        if len(feasible_candidates) != len(candidates):
            raise ValueError(
                "no eligible support-feasible labels for an adaptation episode"
            )
        qi = sample_queries_covering_labels(
            episode_query_pool, candidates, y, count, local_rng,
            config_ids=cfg, subject_ids=subj,
        )
        episode_window_mask = memory_window_mask
        if validation:
            episode_window_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
            episode_window_mask[pool] = True
        query = table.gather_queries(
            qi,
            device,
            expand_verified_events=True,
            allowed_window_mask=episode_window_mask,
        )
        # Candidate selection guarantees every candidate can supply k independent support units.
        # The view samples those units uniformly. Retrieval receives only the resulting allowed-row
        # mask; it is never told which rows are support and never has support appended to its top-k.
        view = build_episode_memory_view(
            bank["patch"], rows, query, y[qi], candidates,
            support_count=spec.support_count,
            episode_type=spec.episode_type,
            label_mode=spec.label_mode,
            rng=local_rng,
        )
        return qi, query, view

    # Fixed canaries hold query rows, candidate identities and text fixed within each curve. The
    # held-family curves select checkpoints, while matched training-family curves expose the gap.
    canary_support_counts = (1, 2) if args.smoke else VALIDATION_SUPPORT_COUNTS
    canary_candidate_count = 4 if args.smoke else VALIDATION_CURVE_CANDIDATE_COUNT
    # Five points per fold match the information available during training: coherent k=0 and
    # arbitrary-name k=1/2/4/8. The positive-k aliases are fixed within a curve so only support
    # count changes along that curve.
    validation_curves = (
        ("coherent_zero", "semantic_zero_support", "coherent", (0,)),
        ("alias_full", "ordinary_few_support", "random_alias", canary_support_counts),
    )
    training_curves = validation_curves
    val_selector_z = F.normalize(
        torch.as_tensor(bank["patch"]["Z"])[val_index_rows].float().to(device), dim=-1
    )
    train_canary_index_rows = index_rows.clone()
    train_canary_selector_z = selector_z.detach().clone()
    validation_query_pools = []
    for fold_name in ("subject_only", "configuration_only", "joint"):
        relation_pool = val_pool[getattr(fold_masks, fold_name)[val_pool]]
        if len(torch.unique(y[relation_pool])) < 2:
            raise RuntimeError(
                f"validation relation {fold_name!r} has fewer than two held-family labels"
            )
        validation_query_pools.append((fold_name, relation_pool))

    def build_fixed_canaries(pool, rows, *, curves, count, validation, seed_offset):
        canaries = []
        cases = (
            validation_canary_cases(curves, validation_query_pools)
            if validation else
            [(None, pool, curve_index, curve) for curve_index, curve in enumerate(curves)]
        )
        base_canaries = []
        for case_index, (fold_relation, episode_pool, curve_index, curve) in enumerate(cases):
            curve_name, episode_type, label_mode, support_values = curve
            episode_seed = args.seed + seed_offset * 1000 + case_index
            local_rng = np.random.default_rng(episode_seed)
            max_support = max(support_values)
            built = None
            max_spec = AdaptationEpisodeSpec(
                episode_type=episode_type,
                support_count=max_support,
                candidate_count=canary_candidate_count,
                label_mode=label_mode,
            )
            for _attempt in range(50):
                try:
                    qi, query, max_view = make_adaptation_episode(
                        episode_pool, rows, max_spec,
                        count=args.val_queries, local_rng=local_rng,
                        validation=validation,
                    )
                    if len(max_view.candidate_ids) != canary_candidate_count:
                        raise ValueError("fixed validation curve did not realize all candidates")
                    label_set = episode_label_set(
                        max_view.candidate_ids, text, mode=label_mode,
                        rng=local_rng,
                        alias_embeddings=alias_embeddings,
                        canonical_names=vocab,
                    )
                    built = qi, query, max_view, label_set
                    break
                except ValueError:
                    continue
            if built is None:
                split = "validation" if validation else "training"
                raise RuntimeError(
                    f"could not construct {split} nested adaptation curve {curve_name!r} "
                    f"with C={canary_candidate_count}, k={max_support}"
                )
            qi, query, max_view, label_set = built
            for support in support_values:
                view = nested_support_view(max_view, bank["patch"], rows, support)
                spec = replace(
                    max_spec,
                    episode_type=(
                        "semantic_zero_support" if support == 0 else max_spec.episode_type
                    ),
                    support_count=support,
                )
                base_canaries.append({
                    "spec": spec,
                    "qi": qi,
                    "query": query,
                    "view": view,
                    "candidate_text": label_set.embeddings,
                    "candidate_phrases": label_set.phrases,
                    "seed": episode_seed,
                    "requested_support": support,
                    "fold_relation": fold_relation,
                    "curve": curve_name,
                })
        if count > len(base_canaries):
            raise ValueError(
                f"requested {count} fixed canaries but the declared grid contains "
                f"{len(base_canaries)}"
            )
        for base in base_canaries[:count]:
            for physical_view_mode in PHYSICAL_VIEW_MODES:
                canaries.append({
                    **base,
                    "spec": replace(base["spec"], physical_view_mode=physical_view_mode),
                })
        return canaries

    val_specs = build_fixed_canaries(
        val_pool, val_index_rows, curves=validation_curves, count=args.val_episodes,
        validation=True, seed_offset=1,
    )
    train_canary_specs = build_fixed_canaries(
        train_pool, train_canary_index_rows,
        curves=training_curves,
        count=min(args.val_episodes, 1 + len(canary_support_counts)),
        validation=False, seed_offset=2,
    )
    for canaries, rows in (
        (val_specs, val_index_rows),
        (train_canary_specs, train_canary_index_rows),
    ):
        for canary in canaries:
            canary["view_specs"] = _episode_view_specs(
                canary["query"], canary["view"], rows, bank["patch"],
                np.random.default_rng(canary["seed"]),
                physical_view_mode=canary["spec"].physical_view_mode,
            )

    val_encoded_views = train_canary_encoded_views = None
    if policy.tokenizer_mode == "frozen" and live_source is not None:
        val_encoded_views = encode_frozen_adaptation_views_batch(
            [
                (
                    canary["query"], canary["view"], canary["view_specs"],
                    canary["spec"].physical_view_mode,
                )
                for canary in val_specs
            ],
            val_index_rows, live_source, ema_encoder,
        )
        train_canary_encoded_views = encode_frozen_adaptation_views_batch(
            [
                (
                    canary["query"], canary["view"], canary["view_specs"],
                    canary["spec"].physical_view_mode,
                )
                for canary in train_canary_specs
            ],
            train_canary_index_rows, live_source, ema_encoder,
        )
    current_canary_fp = structured_fingerprint({
        "validation_index_rows": val_index_rows,
        "validation_canaries": val_specs,
        "training_index_rows": train_canary_index_rows,
        "training_canaries": train_canary_specs,
    })

    initial_validation_rosters = None
    previous_validation_rosters = None
    initial_validation_retriever = None
    previous_validation_retriever = None

    @torch.no_grad()
    def evaluate():
        nonlocal initial_validation_rosters, previous_validation_rosters
        nonlocal initial_validation_retriever, previous_validation_retriever
        dec.eval(); retriever.eval()
        if ema_encoder is not None:
            ema_encoder.eval()
        all_pred, all_identity_pred, all_true = [], [], []
        per_cell, identity_per_cell, true_mass, positive_support_recall = [], [], [], []
        cell_records = []
        support_removal_drop = []
        support_label_shuffle_drop = []
        support_removal_by_view = {mode: [] for mode in PHYSICAL_VIEW_MODES}
        support_label_shuffle_by_view = {mode: [] for mode in PHYSICAL_VIEW_MODES}
        current_validation_rosters = []
        fold_predictions = {
            name: {"pred": [], "true": []}
            for name in ("subject_only", "configuration_only", "joint")
        }
        eval_selector_z = val_selector_z
        if policy.tokenizer_mode == "ema_finetune":
            eval_selector_z = F.normalize(
                live_source.encode_patch_rows(
                    val_index_rows, ema_encoder, requires_grad=False
                ).detach().to(device),
                dim=-1,
            )
        val_memory_index = retriever.build_index(eval_selector_z)
        val_prepared_views = None
        if policy.tokenizer_mode == "frozen" and live_source is not None:
            val_prepared_views = prepare_frozen_adaptation_views_batch(
                [
                    (
                        canary["query"], canary["view"], canary["view_specs"],
                        canary["spec"].physical_view_mode,
                    )
                    for canary in val_specs
                ],
                bank, val_index_rows, eval_selector_z, val_memory_index, retriever,
                live_source, ema_encoder, encoded_views=val_encoded_views,
            )
        for canary_index, canary in enumerate(val_specs):
            spec = canary["spec"]
            qi, view = canary["qi"], canary["view"]
            prepared_views = (
                None if val_prepared_views is None else val_prepared_views[canary_index]
            )
            logits, aux = decode_adaptation_episode(
                dec, retriever, bank, val_index_rows, eval_selector_z, val_memory_index,
                canary["query"], view, text, canary["candidate_text"],
                policy=policy,
                rng=np.random.default_rng(canary["seed"]),
                live_source=live_source, selector_encoder=ema_encoder,
                online_requires_grad=False,
                physical_view_mode=spec.physical_view_mode,
                view_specs=canary["view_specs"],
                prepared_views=prepared_views,
            )
            true = y[qi]
            pred = view.candidate_ids[logits.argmax(1)]
            identity_pred = view.candidate_ids[aux["identity_logits"].argmax(1)]
            cell_ba = balanced_accuracy(pred.cpu().numpy(), true.cpu().numpy())
            identity_cell_ba = balanced_accuracy(
                identity_pred.cpu().numpy(), true.cpu().numpy()
            )
            target_position = label_index(view.candidate_ids, n_vocab, device)[true]
            normalized_ce = float(
                F.cross_entropy(logits, target_position)
                / max(np.log(len(view.candidate_ids)), 1e-8)
            )
            per_cell.append((spec.support_count, spec.physical_view_mode, cell_ba))
            identity_per_cell.append(
                (spec.support_count, spec.physical_view_mode, identity_cell_ba)
            )
            cell_records.append({
                "support": spec.support_count,
                "candidate_count": len(view.candidate_ids),
                "enrolled_candidates": int(view.support_units_per_candidate.gt(0).sum()),
                "requested_support": canary["requested_support"],
                "physical_view_mode": spec.physical_view_mode,
                "episode_type": spec.episode_type,
                "label_mode": spec.label_mode,
                "enrollment_shape": spec.enrollment_shape,
                "curve": canary["curve"],
                "ba": cell_ba,
                "identity_ba": identity_cell_ba,
                "loss_over_random": normalized_ce,
            })
            for batch_index in range(len(aux["evidence_index"])):
                row = aux["evidence_mask"][batch_index]
                current_validation_rosters.append(frozenset(
                    aux["evidence_index"][batch_index, row].cpu().tolist()
                ))
            all_pred.extend(pred.cpu().tolist())
            all_identity_pred.extend(identity_pred.cpu().tolist())
            all_true.extend(true.cpu().tolist())
            for fold_name in fold_predictions:
                member = getattr(fold_masks, fold_name)[qi]
                fold_predictions[fold_name]["pred"].extend(pred[member].cpu().tolist())
                fold_predictions[fold_name]["true"].extend(true[member].cpu().tolist())
            selected_true_support = (
                aux["evidence_support"]
                & aux["evidence_support_candidate"].eq(target_position.unsqueeze(1))
                & aux["evidence_mask"]
            )
            mass = (aux["pool_weights"] * selected_true_support).sum(1)
            true_mass.extend(mass.cpu().tolist())
            enrolled_query = aux["realized_true_support"].gt(0)
            if bool(enrolled_query.any()):
                enrolled_recall = selected_true_support.any(1)[enrolled_query].float().cpu().tolist()
                positive_support_recall.extend(enrolled_recall)
                cell_records[-1]["true_support_recall_at_k"] = float(
                    np.mean(enrolled_recall)
                )
            if spec.support_count > 0:
                cell_records[-1]["enrolled_query_fraction"] = float(
                    enrolled_query.float().mean()
                )
            if spec.support_count > 0:
                removed_view = replace(
                    view,
                    allowed=view.allowed & ~view.support_mask.view(1, 1, -1),
                    support_mask=torch.zeros_like(view.support_mask),
                    support_candidate=torch.full_like(view.support_candidate, -1),
                    support_units_per_candidate=torch.zeros_like(
                        view.support_units_per_candidate
                    ),
                )
                removed_logits, _ = decode_adaptation_episode(
                    dec, retriever, bank, val_index_rows, eval_selector_z,
                    val_memory_index, canary["query"], removed_view, text,
                    canary["candidate_text"], policy=policy,
                    rng=np.random.default_rng(canary["seed"]),
                    live_source=live_source,
                    selector_encoder=ema_encoder, online_requires_grad=False,
                    physical_view_mode=spec.physical_view_mode,
                    view_specs=(canary["view_specs"][0], []),
                    prepared_views=(
                        None if prepared_views is None else (
                            prepared_views[0], prepared_views[1],
                            eval_selector_z, val_memory_index,
                        )
                    ),
                )
                normal_probability = torch.softmax(aux["hard_logits"], dim=1)
                removed_probability = torch.softmax(removed_logits, dim=1)
                row = torch.arange(len(target_position), device=device)
                removal_values = (
                    normal_probability[row, target_position]
                    - removed_probability[row, target_position]
                ).cpu().tolist()
                support_removal_drop.extend(removal_values)
                support_removal_by_view[spec.physical_view_mode].extend(removal_values)

                # Keep the candidate text fixed while assigning every enrolled execution the next
                # candidate's label. A model that genuinely uses enrollment labels must react.
                shuffled_support_candidate = view.support_candidate.clone()
                shuffled_support_candidate[view.support_mask] = (
                    shuffled_support_candidate[view.support_mask] + 1
                ) % len(view.candidate_ids)
                shuffled_view = replace(
                    view, support_candidate=shuffled_support_candidate,
                )
                shuffled_logits, _ = decode_adaptation_episode(
                    dec, retriever, bank, val_index_rows, eval_selector_z,
                    val_memory_index, canary["query"], shuffled_view, text,
                    canary["candidate_text"], policy=policy,
                    rng=np.random.default_rng(canary["seed"]),
                    live_source=live_source,
                    selector_encoder=ema_encoder, online_requires_grad=False,
                    physical_view_mode=spec.physical_view_mode,
                    view_specs=canary["view_specs"],
                    prepared_views=prepared_views,
                )
                shuffled_probability = torch.softmax(shuffled_logits, dim=1)
                shuffle_values = (
                    normal_probability[row, target_position]
                    - shuffled_probability[row, target_position]
                ).cpu().tolist()
                support_label_shuffle_drop.extend(shuffle_values)
                support_label_shuffle_by_view[spec.physical_view_mode].extend(shuffle_values)

        zero = [score for support, _, score in per_cell if support == 0]
        low = [score for support, _, score in per_cell if support != 0]
        macro_cell_ba = float(np.mean([score for _, _, score in per_cell]))
        identity_macro_cell_ba = float(
            np.mean([score for _, _, score in identity_per_cell])
        )
        metrics = {
            "macro_cell_ba": macro_cell_ba,
            "identity_macro_cell_ba": identity_macro_cell_ba,
            "adaptation_macro_cell_ba_gain": macro_cell_ba - identity_macro_cell_ba,
            "ba": balanced_accuracy(np.asarray(all_pred), np.asarray(all_true)),
            "identity_ba": balanced_accuracy(
                np.asarray(all_identity_pred), np.asarray(all_true)
            ),
            "loss_over_random": float(np.mean([
                record["loss_over_random"] for record in cell_records
            ])),
            "zero_support_ba": float(np.mean(zero)) if zero else float("nan"),
            "positive_support_ba": float(np.mean(low)) if low else float("nan"),
            "positive_support_recall_at_k": (
                float(np.mean(positive_support_recall))
                if positive_support_recall else float("nan")
            ),
            "mean_retrieved_true_support_mass": float(np.mean(true_mass)),
            "support_removal_true_probability_drop": (
                float(np.mean(support_removal_drop)) if support_removal_drop else float("nan")
            ),
            "support_label_shuffle_true_probability_drop": (
                float(np.mean(support_label_shuffle_drop))
                if support_label_shuffle_drop else float("nan")
            ),
            "support_fallback_fraction": float(np.mean([
                record["support"] != record["requested_support"] for record in cell_records
            ])),
        }
        if initial_validation_rosters is None:
            initial_validation_rosters = list(current_validation_rosters)
        if previous_validation_rosters is None:
            previous_validation_rosters = list(current_validation_rosters)
        if initial_validation_retriever is None:
            initial_validation_retriever = retriever.ema_proj.detach().clone()
        if previous_validation_retriever is None:
            previous_validation_retriever = retriever.ema_proj.detach().clone()

        def roster_jaccard(left, right):
            union = left | right
            return len(left & right) / len(union) if union else 1.0

        metrics.update({
            "retrieval_roster_jaccard_to_initial": float(np.mean([
                roster_jaccard(now, initial)
                for now, initial in zip(current_validation_rosters, initial_validation_rosters)
            ])),
            "retrieval_roster_jaccard_to_previous": float(np.mean([
                roster_jaccard(now, previous)
                for now, previous in zip(current_validation_rosters, previous_validation_rosters)
            ])),
            "retrieval_roster_changed_fraction": float(np.mean([
                now != previous
                for now, previous in zip(current_validation_rosters, previous_validation_rosters)
            ])),
            "retriever_ema_relative_drift_from_initial": float(
                (retriever.ema_proj - initial_validation_retriever).norm()
                / initial_validation_retriever.norm().clamp_min(1e-12)
            ),
            "retriever_ema_relative_drift_since_validation": float(
                (retriever.ema_proj - previous_validation_retriever).norm()
                / previous_validation_retriever.norm().clamp_min(1e-12)
            ),
        })
        previous_validation_rosters = list(current_validation_rosters)
        previous_validation_retriever = retriever.ema_proj.detach().clone()
        for support in (0, *VALIDATION_SUPPORT_COUNTS):
            selected_records = [record for record in cell_records if record["support"] == support]
            if selected_records:
                metrics[f"support_k{support}_macro_cell_ba"] = float(np.mean([
                    record["ba"] for record in selected_records
                ]))
                metrics[f"support_k{support}_identity_macro_cell_ba"] = float(np.mean([
                    record["identity_ba"] for record in selected_records
                ]))
                metrics[f"support_k{support}_loss_over_random"] = float(np.mean([
                    record["loss_over_random"] for record in selected_records
                ]))
                recalls = [
                    record["true_support_recall_at_k"] for record in selected_records
                    if "true_support_recall_at_k" in record
                ]
                if recalls:
                    metrics[f"support_k{support}_true_support_recall_at_k"] = float(
                        np.mean(recalls)
                    )
        for candidate_count in (VALIDATION_CURVE_CANDIDATE_COUNT,):
            selected_records = [
                record for record in cell_records
                if record["support"] == 0 and record["candidate_count"] == candidate_count
            ]
            if selected_records:
                metrics[f"support_k0_c{candidate_count}_macro_cell_ba"] = float(np.mean([
                    record["ba"] for record in selected_records
                ]))
                metrics[f"support_k0_c{candidate_count}_identity_macro_cell_ba"] = float(
                    np.mean([record["identity_ba"] for record in selected_records])
                )
        for fold_name, values in fold_predictions.items():
            if values["true"]:
                metrics[f"validation_fold/{fold_name}_ba"] = balanced_accuracy(
                    np.asarray(values["pred"]), np.asarray(values["true"])
                )
                metrics[f"validation_fold/{fold_name}_queries"] = len(values["true"])
        for episode_type in EPISODE_TYPES:
            selected_records = [
                record for record in cell_records if record["episode_type"] == episode_type
            ]
            if selected_records:
                metrics[f"episode/{episode_type}_macro_cell_ba"] = float(np.mean([
                    record["ba"] for record in selected_records
                ]))
        for label_mode in LABEL_TEXT_MODES:
            selected_records = [
                record for record in cell_records if record["label_mode"] == label_mode
            ]
            if selected_records:
                metrics[f"label_mode/{label_mode}_macro_cell_ba"] = float(np.mean([
                    record["ba"] for record in selected_records
                ]))
        for enrollment_shape in ("zero", "full"):
            selected_records = [
                record for record in cell_records
                if record["enrollment_shape"] == enrollment_shape
            ]
            if selected_records:
                metrics[f"enrollment/{enrollment_shape}_macro_cell_ba"] = float(np.mean([
                    record["ba"] for record in selected_records
                ]))
                metrics[f"enrollment/{enrollment_shape}_identity_macro_cell_ba"] = float(
                    np.mean([record["identity_ba"] for record in selected_records])
                )
        for physical_view_mode in PHYSICAL_VIEW_MODES:
            view_scores = [
                score for _, mode, score in per_cell if mode == physical_view_mode
            ]
            identity_view_scores = [
                score for _, mode, score in identity_per_cell
                if mode == physical_view_mode
            ]
            metrics[f"{physical_view_mode}_macro_cell_ba"] = float(np.mean(view_scores))
            metrics[f"{physical_view_mode}_identity_macro_cell_ba"] = float(
                np.mean(identity_view_scores)
            )
            metrics[f"{physical_view_mode}_support_removal_true_probability_drop"] = (
                float(np.mean(support_removal_by_view[physical_view_mode]))
                if support_removal_by_view[physical_view_mode] else float("nan")
            )
            metrics[f"{physical_view_mode}_support_label_shuffle_true_probability_drop"] = (
                float(np.mean(support_label_shuffle_by_view[physical_view_mode]))
                if support_label_shuffle_by_view[physical_view_mode] else float("nan")
            )

        low_support_records = [
            record for record in cell_records if record["support"] in {1, 2}
        ]
        if not low_support_records:
            raise RuntimeError("fixed validation canaries contain no k=1/2 adaptation cells")
        selection_low_k_ba = float(np.mean([
            record["ba"] for record in low_support_records
        ]))
        selection_low_k_identity_ba = float(np.mean([
            record["identity_ba"] for record in low_support_records
        ]))
        selection_low_k_gain = selection_low_k_ba - selection_low_k_identity_ba
        zero_support_ba = metrics["support_k0_macro_cell_ba"]
        zero_support_identity_ba = metrics["support_k0_identity_macro_cell_ba"]
        # Absolute low-k quality ranks checkpoints only after the mechanism checks establish that
        # the learned arm matches its closed-form control and reacts correctly to both interventions.
        adaptation_score = selection_low_k_ba
        metrics.update({
            "selection_low_k_ba": selection_low_k_ba,
            "selection_low_k_identity_ba": selection_low_k_identity_ba,
            "selection_low_k_adaptation_gain": selection_low_k_gain,
            "zero_support_identity_delta": zero_support_ba - zero_support_identity_ba,
            "adaptation_selection_score": adaptation_score,
            "checkpoint_selection_score": adaptation_score,
        })
        eligible, eligibility_checks = checkpoint_eligibility(metrics)
        metrics.update({
            "checkpoint_eligible": eligible,
            **{
                f"checkpoint_eligibility/{name}": passed
                for name, passed in eligibility_checks.items()
            },
        })

        # Matched fixed training canaries use the same episode recipe and metrics. Their purpose is
        # diagnosis, not checkpoint selection: the held-out-family score above remains authoritative.
        train_selector = train_canary_selector_z
        if policy.tokenizer_mode == "ema_finetune":
            train_selector = F.normalize(
                live_source.encode_patch_rows(
                    train_canary_index_rows, ema_encoder, requires_grad=False
                ).detach().to(device),
                dim=-1,
            )
        train_index = retriever.build_index(train_selector)
        train_prepared_views = None
        if policy.tokenizer_mode == "frozen" and live_source is not None:
            train_prepared_views = prepare_frozen_adaptation_views_batch(
                [
                    (
                        canary["query"], canary["view"], canary["view_specs"],
                        canary["spec"].physical_view_mode,
                    )
                    for canary in train_canary_specs
                ],
                bank, train_canary_index_rows, train_selector, train_index, retriever,
                live_source, ema_encoder, encoded_views=train_canary_encoded_views,
            )
        train_records = []
        for canary_index, canary in enumerate(train_canary_specs):
            spec = canary["spec"]
            logits, aux = decode_adaptation_episode(
                dec, retriever, bank, train_canary_index_rows, train_selector, train_index,
                canary["query"], canary["view"], text, canary["candidate_text"],
                policy=policy,
                rng=np.random.default_rng(canary["seed"]),
                live_source=live_source, selector_encoder=ema_encoder,
                online_requires_grad=False,
                physical_view_mode=spec.physical_view_mode,
                view_specs=canary["view_specs"],
                prepared_views=(
                    None if train_prepared_views is None
                    else train_prepared_views[canary_index]
                ),
            )
            true = y[canary["qi"]]
            target_position = label_index(
                canary["view"].candidate_ids, n_vocab, device
            )[true]
            pred = canary["view"].candidate_ids[logits.argmax(1)]
            identity_pred = canary["view"].candidate_ids[aux["identity_logits"].argmax(1)]
            train_records.append({
                "support": spec.support_count,
                "enrolled_candidates": int(
                    canary["view"].support_units_per_candidate.gt(0).sum()
                ),
                "ba": balanced_accuracy(pred.cpu().numpy(), true.cpu().numpy()),
                "identity_ba": balanced_accuracy(
                    identity_pred.cpu().numpy(), true.cpu().numpy()
                ),
                "loss_over_random": float(
                    F.cross_entropy(logits, target_position)
                    / max(np.log(len(canary["view"].candidate_ids)), 1e-8)
                ),
            })
        metrics["train_macro_cell_ba"] = float(np.mean([
            record["ba"] for record in train_records
        ]))
        metrics["train_identity_macro_cell_ba"] = float(np.mean([
            record["identity_ba"] for record in train_records
        ]))
        metrics["train_loss_over_random"] = float(np.mean([
            record["loss_over_random"] for record in train_records
        ]))
        metrics["train_validation_macro_cell_ba_gap"] = (
            metrics["train_macro_cell_ba"] - metrics["macro_cell_ba"]
        )
        return metrics

    best = {"checkpoint_selection_score": -float("inf"), "checkpoint_eligible": False}
    best_step = 0
    best_state = None
    best_control = {"selection_low_k_identity_ba": -float("inf")}
    best_control_step = 0
    best_control_state = None
    t0 = time.time()
    active_refreshes = 0
    state_path = args.out.with_name(f"{args.out.stem}.last{args.out.suffix}")
    run_config = {
        "steps": args.steps,
        "episodes_per_step": args.episodes_per_step,
        "queries_per_episode": args.queries_per_episode,
        "queries_per_step": args.episodes_per_step * args.queries_per_episode,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "grad_clip": args.grad_clip,
        "layers": args.layers,
        "heads": args.heads,
        "evidence_budget": args.evidence_budget,
        "tokenizer_mode": args.tokenizer_mode,
        "retriever_init_predictor": (
            str(args.retriever_init_predictor) if args.retriever_init_predictor else None
        ),
        "freeze_retriever": bool(args.freeze_retriever),
        "val_families": args.val_families,
        "val_frac_cfg": args.val_frac_cfg,
        "val_episodes": args.val_episodes,
        "val_queries": args.val_queries,
        "label_variants": args.label_variants,
        "seed": args.seed,
    }
    current_bank_fp = bank.get("bank_fp") or bank_fingerprint(bank)
    current_source_fp = phase_b_source_fingerprint()
    train_cfg_ids = torch.unique(cfg[train_pool]).cpu().tolist()
    train_subject_ids = torch.unique(subj[train_pool]).cpu().tolist()
    cfg_name_map = bank.get("cfg_names", {})
    subject_name_map = bank.get("subj_names", {})

    def metadata_name(mapping, index):
        if isinstance(mapping, dict):
            return str(mapping.get(index, mapping.get(str(index), index)))
        return str(mapping[index])

    phase_b_train_configs = [metadata_name(cfg_name_map, index) for index in train_cfg_ids]
    phase_b_train_subjects = [
        metadata_name(subject_name_map, index) for index in train_subject_ids
    ]
    predictor_metadata = {
        "cfg": {
            **decoder_cfg,
            "n_subspaces": RETRIEVAL_SUBSPACES,
            "n_retrieval_heads": RETRIEVAL_SUBSPACES,
            "episodes_per_step": args.episodes_per_step,
            "queries_per_episode": args.queries_per_episode,
            "queries_per_step": args.episodes_per_step * args.queries_per_episode,
            "subspace_dim": RETRIEVAL_SUBSPACE_DIM,
            "subspace_ema": RETRIEVAL_PROJECTION_EMA,
            "tokenizer_mode": policy.tokenizer_mode,
        },
        "episode_cfg": {
            "episode_types": list(EPISODE_TYPES),
            "candidate_count_range": list(CANDIDATE_COUNT_RANGE),
            "support_count_range": list(SUPPORT_COUNT_RANGE),
            "physical_view_modes": list(PHYSICAL_VIEW_MODES),
            "clean_physical_view_share": 1.0,
            "label_text_modes": list(LABEL_TEXT_MODES),
            "mixture": "independent_uniform_candidate_and_support_count_draws",
            "batch_structure": "independent_episodes",
            "episodes_per_step": args.episodes_per_step,
            "queries_per_episode": args.queries_per_episode,
            "query_balance": "hierarchical",
            "query_subject_alpha": 0.5,
            "candidate_support_policy": "same_uniformly_sampled_k_for_every_candidate",
            "support_source_policy": "uniform_eligible_real_execution_excluding_query_execution",
            "physical_views": "clean_stored_patch_embeddings_only",
            "candidate_difficulty": "uniform_random_labels_no_hard_distractor_mining",
            "retrieval_selection": "learned_query_driven_topk_without_support_insertion",
        },
        "retrieval_cfg": {**policy.as_dict(), "index_seed": args.seed + 2},
        "phase_b_policy": policy.as_dict(),
        "memory_schema": int(bank["schema_version"]),
        "bank_fp": current_bank_fp,
        "backbone": bank["backbone"],
        "vocab": vocab,
        "phase_b_train_labels": [
            vocab[index] for index in torch.unique(y[train_pool]).cpu().tolist()
        ],
        "phase_b_train_config_names": phase_b_train_configs,
        "phase_b_train_datasets": sorted({
            name.split("/", 1)[0] for name in phase_b_train_configs
        }),
        "phase_b_train_subject_names": phase_b_train_subjects,
        "heldout_labels": heldout_labels.cpu().tolist(),
        "heldout_activity_families": sorted({
            name for name, labels in activity_families.items()
            if set(labels) & {vocab[index] for index in heldout_labels.cpu().tolist()}
        }),
        "activity_family_fp": family_fp,
        "fold": {
            "validation_config_ids": val_cfg.cpu().tolist(),
            "validation_subject_ids": val_subject.cpu().tolist(),
            "validation_query_policy": VALIDATION_QUERY_POLICY,
            "window_counts": {
                "train_base": int(fold_masks.train_base.sum()),
                "validation_subject_only": int(fold_masks.subject_only.sum()),
                "validation_configuration_only": int(fold_masks.configuration_only.sum()),
                "validation_joint": int(fold_masks.joint.sum()),
                "validation_query_pool": int(len(val_pool)),
            },
            "validation_query_pool": {
                "labels": int(torch.unique(y[val_pool]).numel()),
                "subjects": int(torch.unique(subj[val_pool]).numel()),
                "configurations": int(torch.unique(cfg[val_pool]).numel()),
            },
        },
        "optimizer_cfg": {
            "lr": args.lr, "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps, "grad_clip": args.grad_clip,
            "tokenizer_lr_scale": TOKENIZER_LR_SCALE,
            "tokenizer_ema_decay": TOKENIZER_EMA_DECAY,
            "tokenizer_finetune_warmup_steps": TOKENIZER_FINETUNE_WARMUP_STEPS,
        },
        "objective": "candidate_cross_entropy",
        "objective_cfg": {
            "candidate_gradient_loss": "raw_cross_entropy",
            "candidate_diagnostic_normalization": "divide_by_log_candidate_count",
            "reduction": "mean_queries_then_equal_mean_across_episodes",
        },
        "diagnostic_arm": {
            "freeze_retriever": bool(args.freeze_retriever),
            "retriever_initializer": (
                str(args.retriever_init_predictor)
                if args.retriever_init_predictor is not None else None
            ),
            "retriever_initializer_step": (
                retriever_init.get("checkpoint_step") if retriever_init is not None else None
            ),
        },
        "phase_b_schema_version": 8,
        "training_regime": PHASE_B_TRAINING_REGIME,
        "training_source_fp": current_source_fp,
        "validation_canary_fp": current_canary_fp,
    }

    def snapshot_predictor_state() -> dict:
        state = {
            "decoder": {
                key: value.detach().cpu().clone() for key, value in dec.state_dict().items()
            },
            "retriever": {
                key: value.detach().cpu().clone()
                for key, value in retriever.state_dict().items()
            },
        }
        if online_encoder is not None:
            state["tokenizer_online"] = {
                key: value.detach().cpu().clone()
                for key, value in online_encoder.state_dict().items()
            }
            state["tokenizer_ema"] = {
                key: value.detach().cpu().clone()
                for key, value in ema_encoder.state_dict().items()
            }
        return state

    def make_predictor_payload(
        model_state: dict,
        *,
        checkpoint_step: int,
        checkpoint_metrics: dict,
        selection: str,
        predictor_mode: str = "relational_decoder",
    ) -> dict:
        if predictor_mode not in {"relational_decoder", "closed_form_retrieval_vote"}:
            raise ValueError(f"unknown predictor mode {predictor_mode!r}")
        return {
            **model_state,
            **predictor_metadata,
            "checkpoint_step": checkpoint_step,
            "checkpoint_metrics": dict(checkpoint_metrics),
            # Retain the historical names for evaluator and artifact compatibility.
            "best_step": checkpoint_step,
            "best_metrics": dict(checkpoint_metrics),
            "checkpoint_selection": selection,
            "predictor_mode": predictor_mode,
        }

    def save_trainer_state(step: int) -> None:
        state = {
            "kind": "phase_b_patch_decoder_trainer_state_v5",
            "step": step,
            "elapsed_seconds": time.time() - t0,
            "run_config": run_config,
            "bank_fp": current_bank_fp,
            "training_regime": PHASE_B_TRAINING_REGIME,
            "training_source_fp": current_source_fp,
            "validation_canary_fp": current_canary_fp,
            "decoder": dec.state_dict(),
            "retriever": retriever.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "tokenizer_online": (
                online_encoder.state_dict() if online_encoder is not None else None
            ),
            "tokenizer_ema": (
                ema_encoder.state_dict() if online_encoder is not None else None
            ),
            "best": best,
            "best_step": best_step,
            "best_state": best_state,
            "best_control": best_control,
            "best_control_step": best_control_step,
            "best_control_state": best_control_state,
            "index_rows": index_rows.detach().cpu(),
            "selector_z": selector_z.detach().cpu(),
            "active_refreshes": active_refreshes,
            "rng": {
                "numpy_generator": rng.bit_generator.state,
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "text_generator": text_gen.get_state(),
            },
        }
        atomic_torch_save(state, state_path)

    start_step = 0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("kind") != "phase_b_patch_decoder_trainer_state_v5":
            raise SystemExit(
                "--resume is not a protocol-bound Phase-B trainer state; "
                "legacy states cannot safely restore checkpoint selection"
            )
        if resume.get("bank_fp") != current_bank_fp:
            raise SystemExit("resume checkpoint was built against a different memory bank")
        if resume.get("training_regime") != PHASE_B_TRAINING_REGIME:
            raise SystemExit("resume checkpoint was built under a different training regime")
        if resume.get("training_source_fp") != current_source_fp:
            raise SystemExit("Phase-B behavior source changed since the resume state was written")
        if resume.get("validation_canary_fp") != current_canary_fp:
            raise SystemExit("fixed Phase-B validation canaries changed since the resume state")
        if resume.get("run_config") != run_config:
            differing = sorted(
                key for key in set(run_config) | set(resume.get("run_config", {}))
                if run_config.get(key) != resume.get("run_config", {}).get(key)
            )
            raise SystemExit(f"resume configuration differs in: {differing}")
        start_step = int(resume["step"])
        if start_step >= args.steps:
            raise SystemExit(
                f"resume state is already at step {start_step}, not below --steps {args.steps}"
            )
        dec.load_state_dict(resume["decoder"])
        retriever.load_state_dict(resume["retriever"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        if online_encoder is not None:
            if resume.get("tokenizer_online") is None or resume.get("tokenizer_ema") is None:
                raise SystemExit("ema_finetune resume state lacks tokenizer weights")
            online_encoder.load_state_dict(resume["tokenizer_online"])
            ema_encoder.load_state_dict(resume["tokenizer_ema"])
        best = dict(resume["best"])
        best_step = int(resume["best_step"])
        best_state = resume.get("best_state")
        best_control = dict(resume["best_control"])
        best_control_step = int(resume["best_control_step"])
        best_control_state = resume.get("best_control_state")
        index_rows = resume["index_rows"].long().cpu()
        selector_z = resume["selector_z"].float().to(device)
        memory_index = retriever.build_index(selector_z)
        active_refreshes = int(resume.get("active_refreshes", 0))
        saved_rng = resume["rng"]
        rng.bit_generator.state = saved_rng["numpy_generator"]
        torch.set_rng_state(saved_rng["torch"].cpu())
        if device.type == "cuda" and saved_rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved_rng["cuda"]])
        text_gen.set_state(saved_rng["text_generator"].cpu())
        t0 = time.time() - float(resume.get("elapsed_seconds", 0.0))
        print(f"[patch-dec] resumed {args.resume} at step {start_step}", flush=True)

    telemetry = PhaseBTelemetry(
        args.telemetry_dir or (args.out.parent / "telemetry" / args.out.stem),
        interval_seconds=args.telemetry_seconds,
        stage="predictor",
    )
    telemetry.start(
        step=start_step,
        elapsed_seconds=time.time() - t0,
        metadata={
            "planned_steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "val_every": args.val_every,
            "grad_clip": args.grad_clip,
            "n_retrieval_heads": RETRIEVAL_SUBSPACES,
            "output": str(args.out),
            "bank_fp": current_bank_fp,
            "tokenizer_mode": args.tokenizer_mode,
            "resume": str(args.resume) if args.resume is not None else None,
            "run_config": run_config,
            "numerical_policy": {
                "patch_embeddings": "l2_normalized",
                "retrieval_attention_prior": "log_softmax_over_valid_plus_log_count",
                "token_inputs": "unit_direction_components_with_learned_positive_scales_then_ln",
                "candidate_loss": "raw_cross_entropy_equal_mean_across_episodes",
            },
        },
    )

    if args.resume is None:
        # Score both the learned arm and its deterministic closed-form control before any update.
        # An ineligible learned run ships the strongest measured control state rather than arbitrary
        # decoder initialization.
        initial_metrics = evaluate()
        telemetry.set_validation(initial_metrics)
        initial_state = snapshot_predictor_state()
        if checkpoint_is_better(initial_metrics, best):
            best = dict(initial_metrics)
            best_step = 0
            best_state = initial_state
        best_control = dict(initial_metrics)
        best_control_step = 0
        best_control_state = initial_state
        atomic_torch_save(
            make_predictor_payload(
                initial_state,
                checkpoint_step=0,
                checkpoint_metrics=initial_metrics,
                selection="step0_untrained_reference",
            ),
            milestone_checkpoint_path(args.out, 0),
        )
        telemetry.emit(step=0, elapsed_seconds=time.time() - t0, force=True)

    for step in range(start_step + 1, args.steps + 1):
        step_started = time.perf_counter()
        dec.train(); retriever.train()
        if online_encoder is not None:
            online_encoder.train()
        retriever.proj.requires_grad_(not args.freeze_retriever)
        tokenizer_active = (
            online_encoder is not None
            and step > TOKENIZER_FINETUNE_WARMUP_STEPS
        )
        if step > 1 and (step - 1) % ACTIVE_REFRESH_STEPS == 0:
            index_rows = table.sample_index_rows(
                memory_window_mask,
                ACTIVE_WINDOWS_PER_LABEL,
                np.random.default_rng(args.seed + 10_000 + step),
            )
            selector_values = (
                live_source.encode_patch_rows(
                    index_rows, ema_encoder, requires_grad=False
                ).detach().to(device)
                if tokenizer_active else
                torch.as_tensor(bank["patch"]["Z"])[index_rows].float().to(device)
            )
            selector_z = F.normalize(selector_values, dim=-1)
            memory_index = retriever.build_index(selector_z)
            active_refreshes += 1

        opt.zero_grad(set_to_none=True)
        telemetry_due = (
            telemetry.due() or step == start_step + 1
            or step % RETRIEVAL_DIAGNOSTIC_STEPS == 0
            or step % args.val_every == 0
        )
        specs = episode_sampler.sample_batch(args.episodes_per_step)
        if args.smoke:
            specs = [
                AdaptationEpisodeSpec(
                    episode_type=(
                        "semantic_zero_support" if spec.support_count == 0
                        else "ordinary_few_support"
                    ),
                    support_count=min(spec.support_count, 1),
                    candidate_count=spec.candidate_count,
                    label_mode=(
                        "coherent" if min(spec.support_count, 1) == 0 else "random_alias"
                    ),
                )
                for spec in specs
            ]
        training_evidence_budget = policy.evidence_budget
        training_score_temperature = RETRIEVAL_TEMPERATURE
        episode_records = []
        step_loss = 0.0
        hard_grad_sum = torch.zeros_like(retriever.proj) if telemetry_due else None

        # Prepare all independent episode metadata before any forward pass.
        prepared = []
        for episode_number, spec in enumerate(specs):
            episode_error = None
            for attempt in range(20):
                t_ev, t_cand = (
                    (text, text) if variants is None else sample_text_tables(variants, text_gen)
                )
                try:
                    qi, query, view = make_adaptation_episode(
                        train_pool, index_rows, spec, count=args.queries_per_episode,
                        local_rng=rng, validation=False,
                    )
                    episode_labels = episode_label_set(
                        view.candidate_ids,
                        t_cand,
                        mode=spec.label_mode,
                        rng=rng,
                        alias_embeddings=alias_embeddings,
                        canonical_names=vocab,
                    )
                    break
                except ValueError as exc:
                    if not any(token in str(exc) for token in (
                        "no eligible", "no evidence", "support units", "no episodic memory",
                        "no cross-subject/configuration support",
                    )):
                        raise
                    episode_error = exc
            else:
                raise RuntimeError(
                    f"could not draw feasible adaptation episode {episode_number + 1}/"
                    f"{len(specs)} in 20 attempts"
                ) from episode_error
            item = {
                "spec": spec,
                "qi": qi,
                "query": query,
                "view": view,
                "evidence_text": t_ev,
                "candidate_text": episode_labels.embeddings,
                "decode_seed": int(rng.integers(1, 2**31 - 1)),
                "attempt": attempt + 1,
            }
            prepared.append(item)

        if live_source is not None:
            for item in prepared:
                spec = item["spec"]
                item["view_specs"] = _episode_view_specs(
                    item["query"], item["view"], index_rows, bank["patch"],
                    np.random.default_rng(item["decode_seed"]),
                    physical_view_mode=spec.physical_view_mode,
                )

        if live_source is not None and policy.tokenizer_mode == "frozen":
            batched_views = prepare_frozen_adaptation_views_batch(
                [
                    (
                        item["query"], item["view"], item.get("view_specs"),
                        item["spec"].physical_view_mode,
                    )
                    for item in prepared
                ],
                bank, index_rows, selector_z, memory_index, retriever,
                live_source, ema_encoder,
            )
            for item, prepared_views in zip(prepared, batched_views, strict=True):
                item["prepared_views"] = prepared_views

        for item in prepared:
            candidates = item["view"].candidate_ids
            true_label = y[item["qi"]]
            target = label_index(candidates, n_vocab, device)[true_label]
            if bool((target < 0).any()):
                raise RuntimeError("answerable episode omitted a query label from candidates")
            item.update({
                "true_label": true_label,
                "target": target,
            })

        def forward_item(item, *, attention=None):
            spec = item["spec"]
            return decode_adaptation_episode(
                dec, retriever, bank, index_rows, selector_z, memory_index,
                item["query"], item["view"],
                item["evidence_text"], item["candidate_text"],
                policy=policy,
                rng=np.random.default_rng(item["decode_seed"]),
                live_source=live_source,
                selector_encoder=ema_encoder,
                online_encoder=online_encoder,
                online_requires_grad=tokenizer_active,
                physical_view_mode=spec.physical_view_mode,
                evidence_budget=training_evidence_budget,
                score_temperature=training_score_temperature,
                view_specs=item.get("view_specs"),
                prepared_views=item.get("prepared_views"),
                return_attention=telemetry_due if attention is None else attention,
            )

        def diagnostic_gradients(objective, aux_values):
            score_inputs = [
                aux["retrieval_scores"] for aux in aux_values
                if aux["retrieval_scores"].requires_grad
            ]
            inputs = ([retriever.proj] if retriever.proj.requires_grad else []) + score_inputs
            if not telemetry_due or not inputs:
                return None, [None] * len(aux_values)
            gradients = torch.autograd.grad(
                objective, inputs, retain_graph=True, allow_unused=True,
            )
            cursor = 0
            hard_gradient = None
            if retriever.proj.requires_grad:
                hard_gradient = gradients[0]
                cursor = 1
            score_gradients = []
            for aux in aux_values:
                if aux["retrieval_scores"].requires_grad:
                    score_gradients.append(gradients[cursor])
                    cursor += 1
                else:
                    score_gradients.append(None)
            return hard_gradient, score_gradients

        def record_item(item, logits, aux, candidate_loss, retrieval_score_gradient):
            spec = item["spec"]
            view = item["view"]
            normalized_candidate_loss = candidate_loss / max(
                np.log(len(view.candidate_ids)), 1e-8
            )
            composition = (
                describe_episode_composition(
                    bank["patch"], index_rows, item["query"], view, simultaneous_pairs
                )
                if telemetry_due else {}
            )
            episode_metrics, categories, strata = training_episode_telemetry(
                spec=spec,
                candidates=view.candidate_ids,
                target=item["target"],
                true_label=item["true_label"],
                aux=aux,
                candidate_loss=candidate_loss.detach(),
                normalized_candidate_loss=normalized_candidate_loss.detach(),
                index_size=len(index_rows),
                draw_attempts=item["attempt"],
                composition=composition,
                detailed=telemetry_due,
                retrieval_score_gradient=retrieval_score_gradient,
            )
            episode_records.append({
                "metrics": episode_metrics,
                "categories": categories,
                "strata": strata,
                "retrieval_topk": int(aux["retrieval_topk"]),
                "evidence_count": (
                    float(aux["evidence_mask"].sum(1).float().mean())
                    if telemetry_due else None
                ),
                "evidence_budget": int(aux["evidence_budget"]),
                "score_temperature": float(aux["score_temperature"]),
            })

        for episode_number, item in enumerate(prepared):
            logits, aux = forward_item(item)
            candidate_loss = F.cross_entropy(logits, item["target"])
            objective = candidate_loss / len(prepared)
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError(
                    f"non-finite Phase-B loss at step {step}, episode {episode_number + 1}"
                )
            hard_grad, score_grads = diagnostic_gradients(objective, [aux])
            if hard_grad is not None:
                hard_grad_sum.add_(hard_grad.detach())
            objective.backward()
            step_loss += float(objective.detach())
            record_item(item, logits, aux, candidate_loss, score_grads[0])

        dec_gradient_stats = parameter_gradient_statistics(dec.parameters(), device)
        retriever_gradient_stats = parameter_gradient_statistics(retriever.parameters(), device)
        dec_grad = dec_gradient_stats["norm"]
        retrieval_grad = retriever_gradient_stats["norm"]
        tokenizer_gradient_stats = {
            name: torch.tensor(0.0, device=device)
            for name in ("norm", "rms", "coverage", "nonzero_fraction")
        }
        if online_encoder is not None:
            tokenizer_gradient_stats = parameter_gradient_statistics(
                online_encoder.parameters(), device
            )
        tokenizer_grad = tokenizer_gradient_stats["norm"]
        component_gradients = {}
        if telemetry_due:
            for name, parameters in decoder_gradient_components(dec).items():
                statistics = parameter_gradient_statistics(parameters, device)
                for statistic, value in statistics.items():
                    component_gradients[f"component_grad_{statistic}/{name}"] = float(value)
            if online_encoder is not None:
                for name, parameters in tokenizer_gradient_components(online_encoder).items():
                    statistics = parameter_gradient_statistics(parameters, device)
                    for statistic, value in statistics.items():
                        component_gradients[
                            f"tokenizer_component_grad_{statistic}/{name}"
                        ] = float(value)
        trainable_params = [
            parameter for parameter in [*dec.parameters(), *retriever.parameters()]
            if parameter.requires_grad
        ]
        if online_encoder is not None:
            trainable_params += [p for p in online_encoder.parameters() if p.requires_grad]
        preclip_grad = torch.nn.utils.clip_grad_norm_(
            trainable_params, args.grad_clip, error_if_nonfinite=True,
        )
        opt.step(); sched.step()
        if not args.freeze_retriever:
            retriever.update_ema()
        if tokenizer_active:
            update_tokenizer_ema(ema_encoder, online_encoder)
        task_objective = float(np.mean([
            record["metrics"]["loss_over_random"] for record in episode_records
        ]))
        optimizer_metrics = {
            "loss": step_loss,
            "task_objective": task_objective,
            "decoder_grad_norm": float(dec_grad),
            "decoder_grad_rms": float(dec_gradient_stats["rms"]),
            "decoder_grad_coverage": float(dec_gradient_stats["coverage"]),
            "decoder_grad_nonzero_fraction": float(
                dec_gradient_stats["nonzero_fraction"]
            ),
            "retriever_grad_norm": float(retrieval_grad),
            "retriever_grad_rms": float(retriever_gradient_stats["rms"]),
            "retriever_grad_coverage": float(retriever_gradient_stats["coverage"]),
            "retriever_grad_nonzero_fraction": float(
                retriever_gradient_stats["nonzero_fraction"]
            ),
            "retriever_to_decoder_grad_ratio": float(
                retrieval_grad / dec_grad.clamp_min(1e-12)
            ),
            "retriever_to_decoder_grad_rms_ratio": float(
                retriever_gradient_stats["rms"]
                / dec_gradient_stats["rms"].clamp_min(1e-12)
            ),
            "tokenizer_grad_norm": float(tokenizer_grad),
            "tokenizer_grad_rms": float(tokenizer_gradient_stats["rms"]),
            "tokenizer_grad_coverage": float(tokenizer_gradient_stats["coverage"]),
            "tokenizer_grad_nonzero_fraction": float(
                tokenizer_gradient_stats["nonzero_fraction"]
            ),
            "preclip_grad_norm": float(preclip_grad),
            "gradient_clipped_fraction": float(float(preclip_grad) > args.grad_clip),
            "learning_rate": float(opt.param_groups[0]["lr"]),
            "episodes_per_step": float(len(specs)),
            "queries_per_step": float(len(specs) * args.queries_per_episode),
            "training_evidence_budget": float(training_evidence_budget),
            "training_retrieval_temperature": float(training_score_temperature),
            "retriever_trainable": float(not args.freeze_retriever),
            "tokenizer_active": float(tokenizer_active),
            "step_seconds": float(time.perf_counter() - step_started),
        }
        if device.type == "cuda":
            optimizer_metrics.update({
                "gpu_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                "gpu_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            })
        if telemetry_due:
            task_grad_norm = hard_grad_sum.norm()
            optimizer_metrics.update({
                "hard_retriever_probe_grad_norm": float(task_grad_norm),
                "retriever_task_grad_norm": float(task_grad_norm),
                **component_gradients,
            })
            optimizer_metrics.update({
                f"component_scale/{name}": float(value.detach())
                for name, value in dec.component_scales().items()
            })
        for record in episode_records:
            telemetry.update(
                record["metrics"], categories=record["categories"], strata=record["strata"]
            )
        telemetry.update(optimizer_metrics)

        if step == 1 or step % args.val_every == 0:
            metrics = evaluate()
            telemetry.set_validation(metrics)
            better = checkpoint_is_better(metrics, best)
            milestone_state = snapshot_predictor_state()
            if better:
                best, best_step = dict(metrics), step
                best_state = milestone_state
            if metrics["selection_low_k_identity_ba"] > best_control[
                "selection_low_k_identity_ba"
            ]:
                best_control = dict(metrics)
                best_control_step = step
                best_control_state = milestone_state
            milestone_path = milestone_checkpoint_path(args.out, step)
            atomic_torch_save(
                make_predictor_payload(
                    milestone_state,
                    checkpoint_step=step,
                    checkpoint_metrics=metrics,
                    selection="unselected_validation_milestone",
                ),
                milestone_path,
            )
            def episode_mean(key):
                values = [
                    record["metrics"].get(key) for record in episode_records
                    if record["metrics"].get(key) is not None
                ]
                return float(np.mean(values)) if values else None

            def category_mix(key):
                values = [record["categories"][key] for record in episode_records]
                return {value: values.count(value) for value in sorted(set(values))}

            tokenizer_ema_drift = 0.0
            if online_encoder is not None:
                delta_sq = sum(
                    (online.detach().float() - ema.detach().float()).square().sum()
                    for online, ema in zip(
                        online_encoder.parameters(), ema_encoder.parameters(), strict=True
                    )
                )
                base_sq = sum(
                    ema.detach().float().square().sum() for ema in ema_encoder.parameters()
                )
                tokenizer_ema_drift = float(torch.sqrt(delta_sq / base_sq.clamp_min(1e-12)))
            print(json.dumps({
                "step": step,
                "loss": round(step_loss, 5),
                "episodes_per_step": len(specs),
                "queries_per_episode": args.queries_per_episode,
                "queries_per_step": len(specs) * args.queries_per_episode,
                "episode_type_mix": category_mix("episode_type"),
                "enrollment_shape_mix": category_mix("enrollment_shape"),
                "label_mode_mix": category_mix("label_mode"),
                "physical_view_mix": category_mix("physical_view_mode"),
                "candidate_count_mix": category_mix("candidate_count"),
                "support_count_mix": category_mix("support_count"),
                "realized_true_support_mean": round(
                    episode_mean("realized_true_support") or 0.0, 3
                ),
                "retrieved_true_mass": round(episode_mean("true_label_pool_mass") or 0.0, 4),
                "provided_support_recall_at_k": (
                    round(episode_mean("provided_support_recall_at_k"), 4)
                    if episode_mean("provided_support_recall_at_k") is not None else None
                ),
                "head_usage": [
                    round(episode_mean(f"subspace_{head}_mass") or 0.0, 4)
                    for head in range(RETRIEVAL_SUBSPACES)
                ],
                "decoder_grad_norm": round(float(dec_grad), 4),
                "retriever_grad_norm": round(float(retrieval_grad), 4),
                "tokenizer_grad_norm": round(float(tokenizer_grad), 4),
                "tokenizer_active": tokenizer_active,
                "tokenizer_ema_relative_drift": round(tokenizer_ema_drift, 6),
                "active_index_refreshes": active_refreshes,
                "preclip_grad_norm": round(float(preclip_grad), 4),
                "candidate_logit_spread": round(
                    episode_mean("candidate_logit_spread") or 0.0, 4
                ),
                "lr": opt.param_groups[0]["lr"],
                "retrieval_topk": sorted({
                    record["retrieval_topk"] for record in episode_records
                }),
                "evidence_count_mean": round(float(np.mean([
                    record["evidence_count"] for record in episode_records
                ])), 2),
                "unique_evidence_windows_mean": round(
                    episode_mean("unique_evidence_windows") or 0.0, 2
                ),
                **{key: round(value, 4) for key, value in metrics.items()},
                "best_checkpoint_selection_score": round(
                    best["checkpoint_selection_score"], 4
                ) if np.isfinite(best["checkpoint_selection_score"]) else None,
                "elapsed_s": round(time.time() - t0, 1),
            }), flush=True)
        # Emit only after a step that collected the expensive diagnostics. If the wall-clock
        # interval expires during validation, the following step observes `due()` and publishes a
        # complete health snapshot rather than one missing attention/credit-assignment metrics.
        emitted = (
            telemetry.emit(
                step=step, elapsed_seconds=time.time() - t0,
                force=step % RETRIEVAL_DIAGNOSTIC_STEPS == 0,
            )
            if telemetry_due else None
        )
        if emitted is not None:
            print(json.dumps({
                "telemetry": str(telemetry.latest),
                "step": step,
                "window_seconds": round(emitted["window_seconds"], 2),
            }), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_trainer_state(step)

    learned_checkpoint = best_state is not None
    selected_state = best_state if learned_checkpoint else best_control_state
    selected_metrics = best if learned_checkpoint else best_control
    selected_step = best_step if learned_checkpoint else best_control_step
    if selected_state is None:
        raise RuntimeError("training completed without a learned or control checkpoint")
    payload = make_predictor_payload(
        selected_state,
        checkpoint_step=selected_step,
        checkpoint_metrics=selected_metrics,
        selection=(
            "held_family_low_k_eligible_relational_decoder"
            if learned_checkpoint else
            "closed_form_retrieval_vote_fallback_no_eligible_learned_checkpoint"
        ),
        predictor_mode=(
            "relational_decoder" if learned_checkpoint else "closed_form_retrieval_vote"
        ),
    )
    atomic_torch_save(payload, args.out)
    telemetry.emit(
        step=args.steps, elapsed_seconds=time.time() - t0, force=True, final=True
    )
    print(
        f"[patch-dec] selected {payload['predictor_mode']} step {selected_step}: "
        f"{selected_metrics} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
