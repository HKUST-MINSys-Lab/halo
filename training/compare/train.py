"""Fine-tune the encoder and the support-only comparator end to end.

THE LOSS
--------
Two regimes, interleaved in one stream at the sampler's rate ``p``:

* **few-shot**: the answer and every distractor have enrolled support.
* **zero-shot**: the answer remains among the candidate labels, but no candidate has enrolled
  support. Compatible examples carrying other labels form unbound background evidence.

Both use cross-entropy on the true candidate. This matches deployment: zero-shot means that the
answer has no enrolled example, not that the correct answer is absent from the decision roster.

Cross-entropy is averaged over all episodes. Consequently ``p`` is the actual objective-mixture
weight instead of merely controlling whether each regime appears in a batch.

END TO END, FROM SCRATCH, IS THE DEFAULT
----------------------------------------
An earlier draft of this file required a Phase-A warm start, citing an encoder whose effective rank
collapsed 24 -> 9 within 300 steps of random init. That reading was wrong, and the checkpoints say
so: **every** compact-engine checkpoint on disk — including ``long_4h_20260821``, the one that led
33 of 40 enrolment columns at selection 0.5424 — carries ``phase_a_checkpoint=None``. They were all
trained end to end from random init.

What actually happened is that the rank collapse was measured on a 6,000-step schedule, where the
apparent peak at step 1,750 was head saturation rather than convergence. At 35,000 steps the same
from-scratch recipe produced the best result this project has. So rank collapse is a *telemetry
signal worth watching*, not a demonstrated failure mode, and ``encoder/effective_rank`` is logged
for exactly that reason.

``--phase-a`` remains available for the warm-start arm. Which of the two wins at a 35k schedule has
never been tested head to head; that comparison is an experiment, not a settled question.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from data.scripts.curate import deployment_policy
from model.blocks import AttentionSpec
from model.evidence.comparator import ComparatorConfig, SupportComparator, comparator_logits
from training.compare.corpus import support_corpus_from_index
from training.compare.sampling import (
    DEFAULT_LABEL_SUBSET,
    DEFAULT_P_GT_PRESENT,
    DEFAULT_SAME_SUBJECT_PROBABILITY,
    DEFAULT_SUPPORT,
    Episode,
    SupportCorpus,
    balanced_query_indices,
    draw_batch,
)
from training.tokenizer.episodic import EpisodicCollate
from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain_data import (
    PATCH_SECONDS,
    CorpusIndex,
    MultiScaleCollate,
    PretrainDataset,
)
from training.tokenizer.pretrain import (
    capture_runtime_provenance,
    capture_source_provenance,
    corpus_fingerprint,
    prepare_output_dir,
    write_source_provenance,
)
from training.tokenizer.pretrain_episodic import _autocast, _random_encoder, encode_batch

TAU_SUPPORT = 0.07      # closed-form weighting temperature
VOTE_SCALE = 10.0
PROVENANCE_ROOTS = (
    "training/compare", "training/tokenizer", "model/evidence/comparator.py",
    "model/tokenizer", "model/blocks.py", "data/scripts", "data/datasets",
    "baselines/halo_compare", "baselines/base.py", "eval/run_adaptation_baselines.py",
    "eval/enrollment_protocol.py", "eval/data.py", "eval/scoring.py",
)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _parameter_grad_norm(parameters) -> float:
    pieces = [parameter.grad.detach().float().square().sum()
              for parameter in parameters if parameter.grad is not None]
    if not pieces:
        return 0.0
    return float(torch.stack(pieces).sum().sqrt())


def _learning_rate_scale(step: int, *, warmup: int, total: int) -> float:
    if step <= warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _load_items(dataset, positions: list[int], executor: ThreadPoolExecutor | None) -> list[dict]:
    if executor is None:
        return [dataset[position] for position in positions]
    return list(executor.map(dataset.__getitem__, positions))


# ------------------------------------------------------------------ text tower
def label_text_matrix(labels: list[str], device) -> torch.Tensor:
    """Frozen MiniLM embeddings for verbatim label strings.

    The same frozen sentence encoder every scored path in the repo uses, so a label's vector here
    is the vector the evaluation harness would give it.
    """
    from eval.scoring import get_sbert_encoder

    embeddings = torch.from_numpy(get_sbert_encoder()(list(labels))).to(device)
    return F.normalize(embeddings.float(), dim=-1)


# ------------------------------------------------------------------ batching
def recording_rows(encoded: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """One pooled feature and one acquisition descriptor per encoded recording.

    The descriptor is the normalised mean over the sensors that are actually present, matching
    ``training.tokenizer.episodic.live_recording_rows`` exactly. A recording may carry both an
    accelerometer and a gyroscope, so the acquisition configuration lives in this vector rather
    than in a scalar modality code.
    """
    pooled = encoded.get("pooled")
    descriptor = encoded.get("descriptor")
    present = encoded.get("sensor_present")
    if pooled is None or descriptor is None or present is None:
        raise KeyError("the comparator needs pooled, descriptor and sensor_present outputs")
    if bool((~present.any(dim=1)).any()):
        raise ValueError("an encoded recording carries no real sensor")
    weight = present.unsqueeze(-1).to(descriptor.dtype)
    merged = (descriptor * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
    return pooled, F.normalize(merged.float(), dim=-1)


def episode_positions(episodes: list[Episode], corpus: SupportCorpus) -> list[int]:
    """Every dataset position one step needs, query rows first within each episode."""
    positions: list[int] = []
    for episode in episodes:
        positions.append(corpus.recordings[episode.query].window_index)
        positions.extend(corpus.recordings[i].window_index for i in episode.support)
    return positions


def split_encoded(
    pooled: torch.Tensor,
    descriptor: torch.Tensor,
    episodes: list[Episode],
) -> dict[str, torch.Tensor]:
    """Regroup one flat encoder output into padded per-episode query and support tensors."""
    device = pooled.device
    B = len(episodes)
    K = max((len(episode.support) for episode in episodes), default=0)
    d = pooled.shape[-1]
    z = descriptor.shape[-1]

    query_feature = pooled.new_zeros((B, 1, d))
    query_descriptor = descriptor.new_zeros((B, 1, z))
    support_feature = pooled.new_zeros((B, K, d))
    support_descriptor = descriptor.new_zeros((B, K, z))
    support_mask = torch.zeros((B, K), dtype=torch.bool, device=device)

    cursor = 0
    for row, episode in enumerate(episodes):
        query_feature[row, 0] = pooled[cursor]
        query_descriptor[row, 0] = descriptor[cursor]
        cursor += 1
        for slot in range(len(episode.support)):
            support_feature[row, slot] = pooled[cursor]
            support_descriptor[row, slot] = descriptor[cursor]
            support_mask[row, slot] = True
            cursor += 1
    if cursor != pooled.shape[0]:
        raise RuntimeError(
            f"episodes cover {cursor} rows but the encoder returned {pooled.shape[0]}"
        )
    return {
        "query_feature": query_feature,
        "query_descriptor": query_descriptor,
        "query_mask": torch.ones((B, 1), dtype=torch.bool, device=device),
        "support_feature": support_feature,
        "support_descriptor": support_descriptor,
        "support_mask": support_mask,
    }


def episode_text(
    episodes: list[Episode],
    corpus: SupportCorpus,
    text_of,
    device,
) -> dict[str, torch.Tensor]:
    """Candidate text, support-label text, bindings and slots for one batch of episodes."""
    B = len(episodes)
    C = max(len(episode.candidates) for episode in episodes)
    K = max((len(episode.support) for episode in episodes), default=0)
    z = text_of("walking").shape[-1]

    candidate_text = torch.zeros((B, C, z), device=device)
    support_label_text = torch.zeros((B, K, z), device=device)
    support_bound = torch.full((B, K), -1, dtype=torch.long, device=device)
    candidate_slot = torch.zeros((B, C), dtype=torch.long, device=device)
    candidate_mask = torch.zeros((B, C), dtype=torch.bool, device=device)

    for row, episode in enumerate(episodes):
        for slot, label in enumerate(episode.candidates):
            candidate_text[row, slot] = text_of(label)
            candidate_mask[row, slot] = True
        # Slot 0 means unbound. Sequential positive slots are pure within-episode coreference tags;
        # candidate order is already randomized by the sampler, so another RNG is unnecessary and
        # would make the supposedly fixed validation draw change between checkpoints.
        if len(episode.candidates) >= 64:
            raise ValueError("candidate roster exceeds the 63 bound slots available to the model")
        candidate_slot[row, : len(episode.candidates)] = 1 + torch.arange(
            len(episode.candidates), device=device,
        )
        for slot, index in enumerate(episode.support):
            support_label_text[row, slot] = text_of(corpus.recordings[index].label)
            support_bound[row, slot] = episode.support_candidate[slot]
    return {
        "candidate_text": candidate_text,
        "support_label_text": support_label_text,
        "support_bound": support_bound,
        "candidate_slot": candidate_slot,
        "candidate_mask": candidate_mask,
    }


# ------------------------------------------------------------------ loss
def episode_loss(
    logits: torch.Tensor,           # (B, C)
    episodes: list[Episode],
    text: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Cross-entropy on the true candidate, averaged separately for each episode regime."""
    device = logits.device
    masked = logits.masked_fill(~text["candidate_mask"], float("-inf"))
    log_probability = torch.log_softmax(masked, dim=-1)

    few_shot = torch.tensor(
        [index for index, e in enumerate(episodes) if not e.is_zero_shot],
        dtype=torch.long, device=device,
    )
    zero_shot = torch.tensor(
        [index for index, e in enumerate(episodes) if e.is_zero_shot],
        dtype=torch.long, device=device,
    )

    target = torch.tensor([episode.gt_slot for episode in episodes], dtype=torch.long, device=device)
    per_episode = F.nll_loss(log_probability, target, reduction="none")

    few_ce = logits.new_zeros(())
    if len(few_shot):
        few_ce = per_episode[few_shot].mean()

    zero_ce = logits.new_zeros(())
    if len(zero_shot):
        zero_ce = per_episode[zero_shot].mean()
    # Per-regime means are telemetry. The actual mean lets p control each regime's expected share;
    # summing the means would give a rare regime the same gradient as a common one.
    loss = per_episode.mean()
    accuracy = masked.argmax(dim=-1).eq(target).float()
    return {
        "loss": loss,
        "ce": per_episode.mean().detach(),
        "few_shot_ce": few_ce.detach(),
        "zero_shot_ce": zero_ce.detach(),
        "accuracy": accuracy.detach(),
    }


def effective_rank(features: torch.Tensor) -> float:
    """exp(entropy of the normalised singular spectrum) — the collapse watchdog."""
    matrix = features.detach().float().flatten(0, -2)
    if matrix.shape[0] < 2:
        return float("nan")
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)
    share = singular / singular.sum().clamp_min(1e-12)
    entropy = -(share * share.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


# ------------------------------------------------------------------ training
def build_dataset(index: CorpusIndex, args) -> PretrainDataset:
    return PretrainDataset(
        index, index.train, augment=False, two_view=False,
        neutral_acquisition_text=args.neutral_acquisition_text,
    )


def run_step(
    *,
    episodes: list[Episode],
    corpus: SupportCorpus,
    dataset: PretrainDataset,
    collate,
    encoder,
    comparator: SupportComparator | None,
    text_of,
    device: torch.device,
    center: bool = False,
    executor: ThreadPoolExecutor | None = None,
) -> dict:
    positions = episode_positions(episodes, corpus)
    batch = collate(_load_items(dataset, positions, executor))
    encoded = encode_batch(encoder, batch, device)
    pooled, descriptor = recording_rows(encoded)

    rows = split_encoded(pooled, descriptor, episodes)
    text = episode_text(episodes, corpus, text_of, device)
    output = comparator_logits(
        comparator,
        candidate_text=text["candidate_text"],
        query_feature=rows["query_feature"],
        query_descriptor=rows["query_descriptor"],
        query_mask=rows["query_mask"],
        support_feature=rows["support_feature"],
        support_descriptor=rows["support_descriptor"],
        support_label_text=text["support_label_text"],
        support_bound=text["support_bound"],
        support_mask=rows["support_mask"],
        candidate_slot=text["candidate_slot"],
        candidate_mask=text["candidate_mask"],
        temperature=TAU_SUPPORT,
        vote_scale=VOTE_SCALE,
        center=center,
    )
    loss = episode_loss(output["logits"], episodes, text)
    return {**loss, **output, "pooled": pooled, "text": text, "rows": rows}


def prediction_telemetry(result: dict, episodes: list[Episode]) -> dict[str, float]:
    """Cheap behavioral diagnostics for the learned residual and closed-form support rule."""
    mask = result["text"]["candidate_mask"]
    target = torch.tensor(
        [episode.gt_slot for episode in episodes], device=result["logits"].device,
    )
    learned = result["logits"].detach().masked_fill(~mask, float("-inf")).argmax(dim=-1)
    base = result["base_logits"].detach().masked_fill(~mask, float("-inf")).argmax(dim=-1)
    weights = result["support_weight"].detach()
    support_mask = weights > 0
    entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=1)
    entropy = torch.where(support_mask.any(dim=1), entropy, torch.zeros_like(entropy))
    return {
        "accuracy/learned": float(learned.eq(target).float().mean()),
        "accuracy/base": float(base.eq(target).float().mean()),
        "accuracy/learned_minus_base": float(
            learned.eq(target).float().mean() - base.eq(target).float().mean()
        ),
        "vote/top1_support_mass": float(weights.max(dim=1).values.mean()),
        "vote/effective_support_rows": float(entropy.exp().mean()),
        "vote/support_entropy": float(entropy.mean()),
        "comparator/residual_abs_mean": float(result["residual"].detach().abs().mean()),
        "sampler/candidate_padding_fraction": float((~mask).float().mean()),
    }


def support_ablation_telemetry(
    result: dict,
    episodes: list[Episode],
    comparator: SupportComparator,
    *,
    center: bool,
) -> dict[str, float]:
    """Validation-only controls showing whether the learned path uses support content."""
    rows = result["rows"]
    text = result["text"]
    target = torch.tensor(
        [episode.gt_slot for episode in episodes], device=result["logits"].device,
    )
    candidate_mask = text["candidate_mask"]

    def accuracy(support_label_text, support_bound, support_mask) -> float:
        output = comparator_logits(
            comparator,
            candidate_text=text["candidate_text"],
            query_feature=rows["query_feature"],
            query_descriptor=rows["query_descriptor"],
            query_mask=rows["query_mask"],
            support_feature=rows["support_feature"],
            support_descriptor=rows["support_descriptor"],
            support_label_text=support_label_text,
            support_bound=support_bound,
            support_mask=support_mask,
            candidate_slot=text["candidate_slot"],
            candidate_mask=candidate_mask,
            temperature=TAU_SUPPORT,
            vote_scale=VOTE_SCALE,
            center=center,
        )
        prediction = output["logits"].masked_fill(
            ~candidate_mask, float("-inf"),
        ).argmax(dim=-1)
        return float(prediction.eq(target).float().mean())

    no_support_accuracy = accuracy(
        torch.zeros_like(text["support_label_text"]),
        torch.full_like(text["support_bound"], -1),
        torch.zeros_like(rows["support_mask"]),
    )

    # Rotate label/binding pairs while keeping the sensor rows fixed. This preserves tensor shape
    # and label marginals but destroys the evidence-to-label association.
    shuffled_text = text["support_label_text"].clone()
    shuffled_bound = text["support_bound"].clone()
    for row in range(len(episodes)):
        count = int(rows["support_mask"][row].sum())
        if count > 1:
            shuffled_text[row, :count] = torch.roll(shuffled_text[row, :count], shifts=1, dims=0)
            shuffled_bound[row, :count] = torch.roll(
                shuffled_bound[row, :count], shifts=1, dims=0,
            )
    shuffled_accuracy = accuracy(
        shuffled_text, shuffled_bound, rows["support_mask"],
    )
    learned_accuracy = prediction_telemetry(result, episodes)["accuracy/learned"]
    return {
        "validation/accuracy/no_support": no_support_accuracy,
        "validation/accuracy/shuffled_support_labels": shuffled_accuracy,
        "validation/support_lift_over_none": learned_accuracy - no_support_accuracy,
        "validation/support_lift_over_shuffled": learned_accuracy - shuffled_accuracy,
    }


def calibrate_frontend(
    encoder,
    dataset: PretrainDataset,
    corpus: SupportCorpus,
    collate,
    rng: np.random.Generator,
    device: torch.device,
    *,
    batches: int,
    batch_size: int,
    executor: ThreadPoolExecutor | None,
) -> None:
    """Fit frozen physical-feature statistics on a source/label-balanced corpus sample."""
    frontend = getattr(encoder, "filterbank", None)
    if frontend is None or not hasattr(frontend, "reset_norm_accumulator"):
        return
    frontend.reset_norm_accumulator()
    for _ in range(batches):
        indices = balanced_query_indices(corpus, rng, batch_size)
        positions = [corpus.recordings[index].window_index for index in indices]
        batch = collate(_load_items(dataset, positions, executor))
        frontend.accumulate_norm_stats(
            batch["patches"].to(device),
            batch["rates"].to(device),
            batch["patch_len"].to(device),
            patch_mask=batch["patch_padding_mask"].to(device),
            channel_mask=batch["channel_mask"].to(device),
            source_rate_hz=batch.get("source_rates", batch["rates"]).to(device),
        )
    frontend.finalize_norm_stats()
    if not bool(frontend._norm_fitted.item()):
        raise RuntimeError("filterbank normalization calibration did not complete")


@torch.no_grad()
def validate(
    *,
    encoder,
    comparator,
    corpus: SupportCorpus,
    dataset: PretrainDataset,
    collate,
    text_of,
    device: torch.device,
    episodes_count: int,
    episodes_per_step: int,
    seed: int,
    draw_kwargs: dict,
    center: bool,
    executor: ThreadPoolExecutor | None,
) -> dict[str, float]:
    """Evaluate a fixed subject-held-out episode draw without consuming test datasets."""
    was_encoder_training = encoder.training
    was_comparator_training = comparator.training
    encoder.eval(); comparator.eval()
    rng = np.random.default_rng(seed)
    episodes, sampler = draw_batch(corpus, rng, batch_size=episodes_count, **draw_kwargs)
    rows: list[dict] = []
    ablations: list[dict] = []
    losses: list[float] = []
    ranks: list[float] = []
    for start in range(0, len(episodes), episodes_per_step):
        group = episodes[start:start + episodes_per_step]
        with _autocast(device):
            result = run_step(
                episodes=group, corpus=corpus, dataset=dataset, collate=collate,
                encoder=encoder, comparator=comparator, text_of=text_of, device=device,
                center=center, executor=executor,
            )
        rows.append(prediction_telemetry(result, group))
        ablations.append(support_ablation_telemetry(
            result, group, comparator, center=center,
        ))
        losses.append(float(result["loss"]))
        ranks.append(effective_rank(result["pooled"]))
    if was_encoder_training:
        encoder.train()
    if was_comparator_training:
        comparator.train()
    keys = sorted({key for row in rows for key in row})
    ablation_keys = sorted({key for row in ablations for key in row})
    return {
        "validation/loss": float(np.mean(losses)),
        "validation/encoder_effective_rank": float(np.mean(ranks)),
        **{f"validation/{key}": float(np.mean([row[key] for row in rows])) for key in keys},
        **{key: float(np.mean([row[key] for row in ablations])) for key in ablation_keys},
        **{f"validation/{key}": value for key, value in sampler.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a", type=Path, default=None,
                        help="optional Phase-A checkpoint for the warm-start arm. Omit for the "
                             "default end-to-end-from-scratch recipe, which is what every compact "
                             "checkpoint on disk actually used")
    parser.add_argument("--frontend", choices=("fixed", "learnable", "continuous"),
                        default="fixed",
                        help="front end for a from-scratch encoder; the design of record is fixed")
    parser.add_argument("--center-features", action=argparse.BooleanOptionalAction, default=True,
                        help="subtract each episode's mean feature before similarity and attention, "
                             "so only how rows DIFFER can drive the decision. ON by default; "
                             "--no-center-features is the ablation. Centering changes the step-0 "
                             "function, so the two arms are compared by raw score at matched seeds, "
                             "never by paired gain")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="replace known artifacts in a non-empty output directory")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume model, optimizer, schedule and RNG state from a checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=35_000)
    parser.add_argument("--episodes-per-step", type=int, default=8)
    parser.add_argument("--support-size", type=int, default=DEFAULT_SUPPORT)
    parser.add_argument("--p-gt-present", type=float, default=DEFAULT_P_GT_PRESENT)
    parser.add_argument("--same-subject-probability", type=float,
                        default=DEFAULT_SAME_SUBJECT_PROBABILITY)
    parser.add_argument("--label-subset", type=int, nargs=2, default=list(DEFAULT_LABEL_SUBSET))
    parser.add_argument("--mode", choices=("compatible", "near_miss", "unfiltered"),
                        default="compatible")
    parser.add_argument("--neutral-acquisition-text", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Arm A (default): hide acquisition prose; use --no-neutral-... for B")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--data-seed", type=int, default=20260901,
                        help="fixed subject split/corpus seed; keep constant across model replicates")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-episodes", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--calib-batches", type=int, default=20)
    parser.add_argument("--calib-batch-size", type=int, default=256)
    parser.add_argument("--loader-workers", type=int, default=8,
                        help="threads used to fetch independent mmap-backed windows")
    parser.add_argument("--max-per-stream", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="three steps on capped real data with telemetry; launches nothing long")
    args = parser.parse_args()

    if args.smoke:
        args.steps = min(args.steps, 3)
        args.warmup_steps = min(args.warmup_steps, max(0, args.steps - 1))
        args.log_every = 1
        args.val_every = args.steps
        args.val_episodes = min(args.val_episodes, 4)
        args.checkpoint_every = args.steps
        args.calib_batches = min(args.calib_batches, 1)
        args.calib_batch_size = min(args.calib_batch_size, 32)
        args.loader_workers = 0
        args.max_per_stream = args.max_per_stream or 200

    if args.steps < 1 or not 0 <= args.warmup_steps < args.steps:
        parser.error("steps must be positive and warmup-steps must be in [0, steps)")
    if min(args.episodes_per_step, args.support_size, args.val_every, args.val_episodes,
           args.checkpoint_every, args.calib_batches, args.calib_batch_size) < 1:
        parser.error("episode, support, validation, checkpoint and calibration counts must be positive")
    if args.loader_workers < 0:
        parser.error("loader-workers must be nonnegative")
    if not 0.0 <= args.p_gt_present <= 1.0:
        parser.error("p-gt-present must be in [0,1]")
    if not 0.0 <= args.same_subject_probability <= 1.0:
        parser.error("same-subject-probability must be in [0,1]")
    if args.label_subset[0] < 2 or args.label_subset[1] < args.label_subset[0]:
        parser.error("label-subset must be LOW HIGH with 2 <= LOW <= HIGH")
    if args.label_subset[1] >= ComparatorConfig().n_slots:
        parser.error("label-subset HIGH must fit the comparator's 63 bound candidate slots")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    prepare_output_dir(
        args.out, force=args.force, smoke=args.smoke, resume=args.resume is not None,
    )
    source = capture_source_provenance(args.out, write=False, roots=PROVENANCE_ROOTS)
    runtime = capture_runtime_provenance(device)
    runtime["mixed_precision"] = "bfloat16" if device.type == "cuda" else "float32"
    runtime["dynamic_loss_scaling"] = False

    index = CorpusIndex(
        max_per_stream=args.max_per_stream, seed=args.data_seed,
        datasets=deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, alignment="native",
    )
    print(f"[compare] corpus: {index.summary()}", flush=True)
    corpus = support_corpus_from_index(index)
    val_corpus = support_corpus_from_index(index, split="val")
    print(f"[compare] support corpus: {corpus.summary()}", flush=True)
    print(f"[compare] validation corpus: {val_corpus.summary()}", flush=True)

    dataset = build_dataset(index, args)
    val_dataset = PretrainDataset(
        index, index.val, augment=False, two_view=False,
        neutral_acquisition_text=args.neutral_acquisition_text,
    )
    collate = EpisodicCollate(MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS))
    executor = (
        ThreadPoolExecutor(max_workers=args.loader_workers, thread_name_prefix="compare-data")
        if args.loader_workers else None
    )

    resume_blob = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume is not None else None
    )

    if resume_blob is not None:
        encoder_config = dict(resume_blob["config"])
        encoder = build_encoder(resume_blob, device, training=True)
        spec = AttentionSpec(**resume_blob["attention_spec"])
        comparator = SupportComparator(
            spec, ComparatorConfig(**resume_blob["comparator_config"]),
        ).to(device)
        comparator.load_state_dict(resume_blob["comparator"])
        print(f"[compare] resuming {args.resume} at step {resume_blob['step']}", flush=True)
    elif args.phase_a is None:
        # The design-of-record recipe: one stage, everything but the frozen text tower and the
        # filterbank's normalisation statistics starts random.
        encoder, encoder_config = _random_encoder(
            device, args.frontend, neutral_acquisition_text=args.neutral_acquisition_text,
        )
        encoder = encoder.to(device).train()
        print(f"[compare] end-to-end from scratch (frontend={args.frontend}, "
              f"d_model={encoder_config['d_model']})", flush=True)
    else:
        # Local checkpoint written by our own Phase-A trainer; its `config` holds plain Python
        # values alongside the tensors, which weights_only=True refuses to unpickle.
        checkpoint = torch.load(args.phase_a, map_location="cpu", weights_only=False)
        encoder_config = dict(checkpoint["config"])
        checkpoint_frontend = str(encoder_config.get("frontend", "fixed"))
        if checkpoint_frontend != args.frontend:
            raise SystemExit(
                f"--frontend={args.frontend} does not match the Phase-A checkpoint's "
                f"frontend={checkpoint_frontend}"
            )
        phase_a_neutral = bool(encoder_config.get("neutral_acquisition_text", False))
        if phase_a_neutral != args.neutral_acquisition_text:
            raise SystemExit(
                "--neutral-acquisition-text must match the Phase-A checkpoint. Arm A's claim is "
                "that the encoder never saw acquisition text at ANY stage; mixing the two stages "
                "would quietly void it."
            )
        encoder = build_encoder(checkpoint, device, training=True)
        print(f"[compare] warm-started from {args.phase_a}", flush=True)
    if resume_blob is None:
        spec = AttentionSpec(d_model=encoder.d_model, n_heads=4, ffn_mult=2, dropout=0.1)
        comparator = SupportComparator(spec, ComparatorConfig()).to(device)
    if hasattr(encoder, "mask_token"):
        encoder.mask_token.requires_grad_(False)

    frontend = getattr(encoder, "filterbank", None)
    if resume_blob is None and args.phase_a is None:
        print(f"[compare] calibrating frontend on {args.calib_batches} balanced batches", flush=True)
        calibrate_frontend(
            encoder, dataset, corpus, collate, rng, device,
            batches=args.calib_batches, batch_size=args.calib_batch_size, executor=executor,
        )
    elif frontend is not None and hasattr(frontend, "_norm_fitted") \
            and not bool(frontend._norm_fitted.item()):
        raise SystemExit("checkpoint frontend has no fitted normalization statistics")

    text_cache: dict[str, torch.Tensor] = {}

    def text_of(label: str) -> torch.Tensor:
        if label not in text_cache:
            text_cache[label] = label_text_matrix([label], device)[0]
        return text_cache[label]

    comparator_params = [parameter for parameter in comparator.parameters() if parameter.requires_grad]
    encoder_params = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"name": "comparator", "params": comparator_params, "lr": args.lr},
        {"name": "encoder", "params": encoder_params,
         "lr": args.lr * args.encoder_lr_scale},
    ], weight_decay=args.weight_decay)

    trajectory = {
        key: value for key, value in {
            "steps": args.steps,
            "frontend": args.frontend,
            "center_features": args.center_features,
            "support_size": args.support_size,
            "p_gt_present": args.p_gt_present,
            "same_subject_probability": args.same_subject_probability,
            "label_subset": list(args.label_subset),
            "mode": args.mode,
            "neutral_acquisition_text": args.neutral_acquisition_text,
            "lr": args.lr,
            "encoder_lr_scale": args.encoder_lr_scale,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "data_seed": args.data_seed,
        }.items()
    }
    start_step = 0
    if resume_blob is not None:
        if resume_blob.get("trajectory") != trajectory:
            raise SystemExit("resume trajectory differs from the checkpoint configuration")
        optimizer.load_state_dict(resume_blob["optimizer"])
        start_step = int(resume_blob["step"])
        torch.set_rng_state(resume_blob["rng"]["torch"])
        if device.type == "cuda" and resume_blob["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(resume_blob["rng"]["cuda"])
        rng.bit_generator.state = resume_blob["rng"]["numpy_generator"]
        random.setstate(resume_blob["rng"]["python"])
        if start_step >= args.steps:
            raise SystemExit("resume checkpoint already reached or exceeded --steps")

    corpus_fp = corpus_fingerprint(index)
    if resume_blob is not None:
        if resume_blob.get("corpus_fingerprint") != corpus_fp:
            raise SystemExit("resume corpus fingerprint differs from the checkpoint")
        saved_source = resume_blob.get("source_provenance", resume_blob.get("git"))
        if not isinstance(saved_source, dict) or any(
            saved_source.get(key) != source.get(key) for key in ("head", "patch_sha256")
        ):
            raise SystemExit(
                "resume source fingerprint differs from the checkpoint; use its recorded source "
                "or start a new run"
            )
    serial_source = write_source_provenance(args.out, source)
    (args.out / "runtime_provenance.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (args.out / "run_config.json").write_text(json.dumps({
        "args": {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()},
        "trajectory": trajectory,
        "corpus": index.summary(),
        "corpus_fingerprint": corpus_fp,
        "source_provenance": serial_source,
        "runtime_provenance": runtime,
    }, indent=2) + "\n")

    config = dict(encoder_config)
    config["neutral_acquisition_text"] = bool(args.neutral_acquisition_text)
    config["center_features"] = bool(args.center_features)

    latest_validation: dict[str, float] | None = resume_blob.get("validation") if resume_blob else None
    best_accuracy = float(resume_blob.get("best_validation_accuracy", -1.0)) if resume_blob else -1.0

    def payload(step: int) -> dict:
        return {
            "config": config,
            "encoder": encoder.state_dict(),
            "comparator": comparator.state_dict(),
            "comparator_config": dataclasses.asdict(comparator.cfg),
            "attention_spec": dataclasses.asdict(spec),
            "args": {key: (str(value) if isinstance(value, Path) else value)
                     for key, value in vars(args).items()},
            "trajectory": trajectory,
            "phase_a": None if args.phase_a is None else str(args.phase_a),
            "step": step,
            "validation": latest_validation,
            "best_validation_accuracy": best_accuracy,
            "optimizer": optimizer.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "numpy_generator": rng.bit_generator.state,
                "python": random.getstate(),
            },
            "git": serial_source,
            "source_provenance": serial_source,
            "runtime_provenance": runtime,
            "corpus": index.summary(),
            "corpus_fingerprint": corpus_fp,
        }

    log_path = args.out / "log.jsonl"
    started = time.perf_counter()
    draw_kwargs = {
        "support_size": args.support_size,
        "p_gt_present": args.p_gt_present,
        "same_subject_probability": args.same_subject_probability,
        "label_subset": tuple(args.label_subset),
        "mode": args.mode,
    }

    def run_validation(step: int) -> dict[str, float]:
        nonlocal latest_validation, best_accuracy
        latest_validation = validate(
            encoder=encoder, comparator=comparator, corpus=val_corpus, dataset=val_dataset,
            collate=collate, text_of=text_of, device=device,
            episodes_count=args.val_episodes, episodes_per_step=args.episodes_per_step,
            seed=args.data_seed + 91_003, draw_kwargs=draw_kwargs,
            center=args.center_features, executor=executor,
        )
        latest_validation["step"] = float(step)
        with log_path.open("a") as handle:
            handle.write(json.dumps({"kind": "validation", **latest_validation}) + "\n")
        score = latest_validation["validation/accuracy/learned"]
        if score > best_accuracy:
            best_accuracy = score
            _atomic_torch_save(payload(step), args.out / "best_internal.pt")
        return latest_validation

    if resume_blob is None:
        run_validation(0)
        _atomic_torch_save(payload(0), args.out / "initial.pt")

    for step in range(start_step + 1, args.steps + 1):
        scale = _learning_rate_scale(step, warmup=args.warmup_steps, total=args.steps)
        for group, base in zip(optimizer.param_groups,
                               (args.lr, args.lr * args.encoder_lr_scale)):
            group["lr"] = base * scale

        episodes, telemetry = draw_batch(
            corpus, rng,
            batch_size=args.episodes_per_step,
            **draw_kwargs,
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            result = run_step(
                episodes=episodes, corpus=corpus, dataset=dataset, collate=collate,
                encoder=encoder, comparator=comparator, text_of=text_of, device=device,
                center=args.center_features, executor=executor,
            )
        if not bool(torch.isfinite(result["loss"])):
            raise FloatingPointError(f"non-finite comparison loss at step {step}")
        result["loss"].backward()
        log_step = step % args.log_every == 0 or step == 1
        encoder_grad = _parameter_grad_norm(encoder_params) if log_step else 0.0
        comparator_grad = _parameter_grad_norm(comparator_params) if log_step else 0.0
        preclip = float(torch.nn.utils.clip_grad_norm_(
            comparator_params + encoder_params, args.grad_clip, error_if_nonfinite=True,
        ))
        optimizer.step()

        if log_step:
            behavior = prediction_telemetry(result, episodes)
            row = {
                "kind": "train",
                "step": step,
                "loss": float(result["loss"].detach()),
                "loss/ce": float(result["ce"]),
                "loss/few_shot_ce": float(result["few_shot_ce"]),
                "loss/zero_shot_ce": float(result["zero_shot_ce"]),
                "encoder/effective_rank": effective_rank(result["pooled"]),
                "gradient/encoder_norm": encoder_grad,
                "gradient/comparator_norm": comparator_grad,
                "gradient/total_preclip_norm": preclip,
                "gradient/clip_coefficient": min(1.0, args.grad_clip / max(preclip, 1e-12)),
                "lr/comparator": optimizer.param_groups[0]["lr"],
                "lr/encoder": optimizer.param_groups[1]["lr"],
                "elapsed_s": round(time.perf_counter() - started, 1),
                **telemetry,
                **behavior,
                **comparator.telemetry(),
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"[compare] step {step:>6} loss {row['loss']:.4f} "
                f"(few {row['loss/few_shot_ce']:.4f} zero {row['loss/zero_shot_ce']:.4f}) "
                f"acc {row['accuracy/learned']:.2f}/{row['accuracy/base']:.2f} "
                f"rank {row['encoder/effective_rank']:.1f} "
                f"K {row['sampler/mean_support_size']:.1f}",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.steps:
            report = run_validation(step)
            print(
                f"[compare] validation step {step}: learned "
                f"{report['validation/accuracy/learned']:.3f}, base "
                f"{report['validation/accuracy/base']:.3f}",
                flush=True,
            )
        if step % args.checkpoint_every == 0 or step == args.steps:
            _atomic_torch_save(payload(step), args.out / "last.pt")

    if executor is not None:
        executor.shutdown(wait=True)
    _atomic_torch_save(payload(args.steps), args.out / "last.pt")
    print(f"[compare] wrote {args.out / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
