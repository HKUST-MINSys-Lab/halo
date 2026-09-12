"""Train HALO's support-conditioned semantic classifier.

Each episode contains a query recording, enrolled support recordings with their labels, and a
declared candidate roster. A shared motion encoder produces one vector per recording. Separate
zero-shot and enrolled token-mixer heads then contextualise the relevant token set before scoring.
The optional ``neighbors`` mode is the deliberately simpler differentiable support-vote control.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
from model.tokenizer.multispan_kernel import MS_SPANS_S, MS_FRAMES_PER_SPAN
from model.support.token_mixer import SupportTokenMixer, TokenMixerConfig
from training.support_classifier.corpus import support_corpus_from_index
from training.support_classifier.sampling import (
    DEFAULT_ENROLLMENT_K,
    DEFAULT_LABEL_SUBSET,
    DEFAULT_P_GT_PRESENT,
    DEFAULT_QUERIES_PER_SUPPORT_SET,
    DEFAULT_SAME_SUBJECT_PROBABILITY,
    DEFAULT_SUPPORT,
    DEFAULT_WINDOWS_PER_EXECUTION,
    Episode,
    SupportCorpus,
    balanced_query_indices,
    draw_batch,
)
from training.support_classifier.collate import SupportCollate
from training.support_classifier.encoding import autocast, build_random_encoder, encode_batch
from training.support_classifier.neighbors import DEFAULT_TEMPERATURE, differentiable_neighbor_logits
from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain_data import (
    PATCH_SECONDS,
    CorpusIndex,
    MultiResolutionCollate,
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

TAU_SUPPORT = DEFAULT_TEMPERATURE  # compatibility alias for saved experiment notes
def make_optimizer(param_groups, *, weight_decay: float, device: torch.device):
    """AdamW; the fused CUDA kernel updates every parameter in one launch."""
    return torch.optim.AdamW(
        param_groups, weight_decay=weight_decay, fused=(device.type == "cuda"),
    )
PROVENANCE_ROOTS = (
    "training/support_classifier", "training/tokenizer", "model/support",
    "model/tokenizer", "model/blocks.py", "data/scripts", "data/datasets",
    "baselines/base.py",
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


# ------------------------------------------------------------------ prefetching
def episode_rng(data_seed: int, step: int) -> np.random.Generator:
    """The generator that draws training step ``step``.

    Seeding per step (rather than advancing one generator) makes the episode sequence a pure
    function of ``data_seed``: any worker can draw any step, resume needs no generator state, and
    the calibration draws before training cannot shift it.
    """
    return np.random.default_rng([int(data_seed), int(step)])


def _prefetch_worker(corpus, dataset, collate, data_seed, batch_size, draw_kwargs,
                     requests, results) -> None:
    torch.set_num_threads(1)          # forked workers must not oversubscribe the cores
    while True:
        step = requests.get()
        if step is None:
            # close() discards unused prefetched batches; do not wait for their queue feeder.
            results.cancel_join_thread()
            return
        try:
            episodes, telemetry = draw_batch(
                corpus, episode_rng(data_seed, step), batch_size=batch_size, **draw_kwargs,
            )
            batch = collate([dataset[position] for position in episode_positions(episodes, corpus)])
            results.put((step, episodes, telemetry, batch, None))
        except Exception as error:      # noqa: BLE001 - surfaced in the parent, which re-raises
            results.put((step, None, None, None, repr(error)))


class PrefetchLoader:
    """Draw, load and collate the NEXT training steps in worker processes while the GPU trains.

    Measured 2026-09-05 (8 episodes/step, capped corpus): draw + load + collate is about 29 ms
    of single-threaded Python per step against about 13 ms of GPU work, and thread pools cannot
    overlap it because the per-window work holds the GIL. Forked processes can. Workers are
    forked from the parent so they share the corpus index and memory-mapped grids for free; they
    never touch CUDA, exactly like ``torch.utils.data.DataLoader`` workers.

    Step ``s`` is always drawn with :func:`episode_rng`, so the sequence is identical with any
    number of workers, including zero, and identical across a resume.
    """

    def __init__(self, corpus, dataset, collate, *, data_seed: int, batch_size: int,
                 draw_kwargs: dict, workers: int = 4, depth: int = 2, start_step: int = 1):
        if workers < 1:
            raise ValueError("PrefetchLoader needs at least one worker; use draw_batch directly")
        import torch.multiprocessing as mp

        context = mp.get_context("fork")
        self._requests = context.Queue()
        self._results = context.Queue()
        self._pending: dict[int, tuple] = {}
        self._consumed: set[int] = set()
        self._first_step = int(start_step)
        self._next_request = int(start_step)
        self._ahead = int(depth) * int(workers)
        self._processes = [
            context.Process(
                target=_prefetch_worker,
                args=(corpus, dataset, collate, data_seed, batch_size, draw_kwargs,
                      self._requests, self._results),
                daemon=True,
            )
            for _ in range(int(workers))
        ]
        for process in self._processes:
            process.start()
        self._fill(self._next_request + self._ahead)

    def _fill(self, upto: int) -> None:
        while self._next_request < upto:
            self._requests.put(self._next_request)
            self._next_request += 1

    def get(self, step: int):
        """Episodes, sampler telemetry and the collated batch for training step ``step``."""
        if step in self._consumed:
            raise ValueError(f"step {step} has already been consumed")
        if step < self._first_step and step not in self._pending:
            raise ValueError(
                f"step {step} precedes this loader's start_step {self._first_step}; it was never "
                "requested and no worker will produce it"
            )
        self._fill(step + 1 + self._ahead)
        while step not in self._pending:
            from queue import Empty

            try:
                done, episodes, telemetry, batch, error = self._results.get(timeout=1.0)
            except Empty:
                dead = [p for p in self._processes if not p.is_alive()]
                if dead:
                    raise RuntimeError(
                        f"prefetch worker exited while waiting for step {step}: "
                        f"{[(p.pid, p.exitcode) for p in dead]}"
                    )
                continue
            if error is not None:
                raise RuntimeError(f"prefetch worker failed at step {done}: {error}")
            self._pending[done] = (episodes, telemetry, batch)
        self._consumed.add(step)
        return self._pending.pop(step)

    def close(self) -> None:
        for _ in self._processes:
            self._requests.put(None)
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


# ------------------------------------------------------------------ text tower
def label_text_matrix(labels: list[str], device) -> torch.Tensor:
    """Frozen MiniLM embeddings for verbatim label strings.

    The same frozen sentence encoder every scored path in the repo uses, so a label's vector here
    is the vector the evaluation harness would give it.
    """
    from baselines.text import get_sbert_encoder

    embeddings = torch.from_numpy(get_sbert_encoder()(list(labels))).to(device)
    return F.normalize(embeddings.float(), dim=-1)


class LabelTextTable:
    """Every corpus label's frozen text vector in ONE matrix, addressed by integer id.

    Built once per run. ``episode_text`` then gathers whole (B, C, z) and (B, K, z) tensors with
    a single index op instead of writing one row per candidate and per support row — the per-row
    writes cost thousands of tiny kernels and autograd nodes per step (measured 2026-09-05:
    7.4 ms forward and about 6 ms of backward per 8-episode step, against 0.4 ms vectorised).
    """

    def __init__(self, labels, device):
        self.labels = tuple(sorted(set(str(label) for label in labels)))
        self.index = {label: position for position, label in enumerate(self.labels)}
        self.matrix = label_text_matrix(list(self.labels), device)
        self.device = self.matrix.device
        self.dim = int(self.matrix.shape[-1])

    def __call__(self, label: str) -> torch.Tensor:
        """One label's vector; kept so a caller can still ask for a single row."""
        if label not in self.index:
            self.labels = self.labels + (label,)
            self.index[label] = len(self.labels) - 1
            self.matrix = torch.cat([self.matrix, label_text_matrix([label], self.device)])
        return self.matrix[self.index[label]]

    def ids(self, labels) -> list[int]:
        missing = [label for label in labels if label not in self.index]
        for label in missing:
            self(label)
        return [self.index[label] for label in labels]


def make_label_text(labels, device) -> LabelTextTable:
    return LabelTextTable(labels, device)


# ------------------------------------------------------------------ batching
def recording_rows(encoded: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """One pooled feature and one acquisition descriptor per encoded recording.

    The descriptor is the normalised mean over the sensors that are actually present. A recording
    may carry both an accelerometer and a gyroscope, so acquisition metadata lives in this vector
    rather than in a scalar modality code.
    """
    pooled = encoded.get("pooled")
    descriptor = encoded.get("descriptor")
    present = encoded.get("sensor_present")
    if pooled is None or descriptor is None or present is None:
        raise KeyError("the support classifier needs pooled, descriptor and sensor_present outputs")
    if bool((~present.any(dim=1)).any()):
        raise ValueError("an encoded recording carries no real sensor")
    weight = present.unsqueeze(-1).to(descriptor.dtype)
    merged = (descriptor * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
    return pooled, F.normalize(merged.float(), dim=-1)


def episode_positions(episodes: list[Episode], corpus: SupportCorpus) -> list[int]:
    """Unique dataset positions required by a batch, including execution-pooling windows."""
    recording_indices: list[int] = []
    seen: set[int] = set()
    for episode in episodes:
        groups = episode.support_window_groups or tuple((index,) for index in episode.support)
        for index in (episode.query, *(value for group in groups for value in group)):
            if index not in seen:
                seen.add(index)
                recording_indices.append(index)
    return [corpus.recordings[index].window_index for index in recording_indices]


def split_encoded(
    pooled: torch.Tensor,
    descriptor: torch.Tensor,
    episodes: list[Episode],
    corpus: SupportCorpus,
) -> dict[str, torch.Tensor]:
    """Pool support executions once, then reuse them across queries sharing a support set."""
    device = pooled.device
    B = len(episodes)
    K = max((len(episode.support) for episode in episodes), default=0)
    recording_indices: list[int] = []
    seen: set[int] = set()
    for episode in episodes:
        groups = episode.support_window_groups or tuple((index,) for index in episode.support)
        for index in (episode.query, *(value for group in groups for value in group)):
            if index not in seen:
                seen.add(index)
                recording_indices.append(index)
    if len(recording_indices) != pooled.shape[0]:
        raise RuntimeError(
            f"batch references {len(recording_indices)} unique windows but encoder returned "
            f"{pooled.shape[0]} rows"
        )
    encoded_row = {recording: row for row, recording in enumerate(recording_indices)}

    unique_groups: list[tuple[int, ...]] = []
    group_id: dict[tuple[int, ...], int] = {}
    for episode in episodes:
        groups = episode.support_window_groups or tuple((index,) for index in episode.support)
        if len(groups) != len(episode.support):
            raise ValueError("one support window group is required per support execution")
        for group in groups:
            if not group:
                raise ValueError("support execution window groups cannot be empty")
            if group not in group_id:
                group_id[group] = len(unique_groups)
                unique_groups.append(group)

    if unique_groups:
        width = max(map(len, unique_groups))
        group_rows = np.zeros((len(unique_groups), width), dtype=np.int64)
        group_valid = np.zeros((len(unique_groups), width), dtype=bool)
        for row, group in enumerate(unique_groups):
            group_rows[row, :len(group)] = [encoded_row[index] for index in group]
            group_valid[row, :len(group)] = True
        gather = torch.from_numpy(group_rows).to(device)
        valid = torch.from_numpy(group_valid).to(device).unsqueeze(-1)
        denom = valid.sum(1).clamp_min(1).to(pooled.dtype)
        group_feature = (pooled[gather] * valid.to(pooled.dtype)).sum(1) / denom
        group_descriptor = F.normalize(
            (descriptor[gather] * valid.to(descriptor.dtype)).sum(1)
            / valid.sum(1).clamp_min(1).to(descriptor.dtype), dim=-1,
        )
    else:
        group_feature = pooled.new_zeros((1, pooled.shape[-1]))
        group_descriptor = descriptor.new_zeros((1, descriptor.shape[-1]))

    query_position = np.asarray([encoded_row[e.query] for e in episodes], dtype=np.int64)
    support_position = np.zeros((B, K), dtype=np.int64)
    support_valid = np.zeros((B, K), dtype=bool)
    for row, episode in enumerate(episodes):
        groups = episode.support_window_groups or tuple((index,) for index in episode.support)
        count = len(groups)
        support_position[row, :count] = [group_id[group] for group in groups]
        support_valid[row, :count] = True
    query_index = torch.from_numpy(query_position).to(device)
    support_index = torch.from_numpy(support_position).to(device)
    support_mask = torch.from_numpy(support_valid).to(device)
    keep = support_mask.unsqueeze(-1)
    return {
        "query_feature": pooled[query_index].unsqueeze(1),
        "query_descriptor": descriptor[query_index].unsqueeze(1),
        "query_mask": torch.ones((B, 1), dtype=torch.bool, device=device),
        "support_feature": group_feature[support_index] * keep.to(pooled.dtype),
        "support_descriptor": group_descriptor[support_index] * keep.to(descriptor.dtype),
        "support_mask": support_mask,
    }


def episode_text(
    episodes: list[Episode],
    corpus: SupportCorpus,
    text_of,
    device,
) -> dict[str, torch.Tensor]:
    """Candidate text, support-label text, bindings and slots for one batch of episodes.

    ``text_of`` is a :class:`LabelTextTable`. All ids are assembled in numpy and each text tensor
    is one gather from the table; padded slots gather row 0 and are zeroed by their mask.
    """
    if not isinstance(text_of, LabelTextTable):
        raise TypeError("episode_text needs a LabelTextTable (see make_label_text)")
    B = len(episodes)
    C = max(len(episode.candidates) for episode in episodes)
    K = max((len(episode.support) for episode in episodes), default=0)

    candidate_id = np.zeros((B, C), dtype=np.int64)
    candidate_valid = np.zeros((B, C), dtype=bool)
    support_label_id = np.zeros((B, K), dtype=np.int64)
    support_valid = np.zeros((B, K), dtype=bool)
    bound = np.full((B, K), -1, dtype=np.int64)
    candidate_slot = np.zeros((B, C), dtype=np.int64)
    support_pair_slot = np.zeros((B, K), dtype=np.int64)
    for row, episode in enumerate(episodes):
        # Slot 0 means padding.  Tags are deterministic pseudo-random permutations attached to
        # the logical candidate/support pair, not tensor position. Reordering a set and carrying
        # these tags with it is therefore exactly equivariant; across training episodes their
        # values cannot become a label or row-position lookup table.
        n_candidates, n_support = len(episode.candidates), len(episode.support)
        digest = hashlib.sha256(repr((episode.query, episode.candidates, episode.support)).encode()).digest()
        tag_rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        candidate_slot[row, :n_candidates] = tag_rng.permutation(n_candidates) + 1
        support_pair_slot[row, :n_support] = tag_rng.permutation(n_support) + 1
        candidate_id[row, :n_candidates] = text_of.ids(episode.candidates)
        candidate_valid[row, :n_candidates] = True
        support_label_id[row, :n_support] = text_of.ids(
            [corpus.recordings[index].label for index in episode.support]
        )
        support_valid[row, :n_support] = True
        bound[row, :n_support] = episode.support_candidate
    candidate_mask = torch.from_numpy(candidate_valid).to(device)
    support_mask = torch.from_numpy(support_valid).to(device)
    matrix = text_of.matrix
    return {
        "candidate_text": (
            matrix[torch.from_numpy(candidate_id).to(device)]
            * candidate_mask.unsqueeze(-1).to(matrix.dtype)
        ),
        "support_label_text": (
            matrix[torch.from_numpy(support_label_id).to(device)]
            * support_mask.unsqueeze(-1).to(matrix.dtype)
        ),
        "support_bound": torch.from_numpy(bound).to(device),
        "candidate_slot": torch.from_numpy(candidate_slot).to(device),
        "support_pair_slot": torch.from_numpy(support_pair_slot).to(device),
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

    # Several queries may share one support roster. Give each independently drawn support set equal
    # weight even when a sparse source cannot supply the requested number of distinct query
    # executions. Legacy episodes carry id -1 and retain the ordinary query mean.
    support_set_ids = torch.tensor(
        [episode.support_set_id for episode in episodes], dtype=torch.long, device=device,
    )
    if bool(support_set_ids.ge(0).all()):
        unique_ids = list(dict.fromkeys(episode.support_set_id for episode in episodes))
        set_losses = torch.stack([
            per_episode[[index for index, episode in enumerate(episodes)
                         if episode.support_set_id == set_id]].mean()
            for set_id in unique_ids
        ])
        set_is_zero = torch.tensor([
            next(episode.is_zero_shot for episode in episodes
                 if episode.support_set_id == set_id)
            for set_id in unique_ids
        ], dtype=torch.bool, device=device)
        # The two heads solve different information conditions.  Weight their set means equally
        # whenever both occur, rather than letting whichever regime produced more query windows
        # determine shared-encoder learning.
        few_ce = set_losses[~set_is_zero].mean() if bool((~set_is_zero).any()) else logits.new_zeros(())
        zero_ce = set_losses[set_is_zero].mean() if bool(set_is_zero.any()) else logits.new_zeros(())
        active = [term for term, present in ((few_ce, bool((~set_is_zero).any())),
                                             (zero_ce, bool(set_is_zero.any()))) if present]
        loss = torch.stack(active).mean()
    else:
        few_ce = per_episode[few_shot].mean() if len(few_shot) else logits.new_zeros(())
        zero_ce = per_episode[zero_shot].mean() if len(zero_shot) else logits.new_zeros(())
        active = [term for term, present in ((few_ce, len(few_shot) > 0),
                                             (zero_ce, len(zero_shot) > 0)) if present]
        loss = torch.stack(active).mean()
    accuracy = masked.argmax(dim=-1).eq(target).float()
    return {
        "loss": loss,
        "ce": loss.detach(),
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


def _macro_f1_names(truth: list[str], prediction: list[str]) -> float:
    """Unweighted class F1 without adding sklearn to the training hot path."""
    scores = []
    for label in sorted(set(truth)):
        tp = sum(t == label and p == label for t, p in zip(truth, prediction))
        fp = sum(t != label and p == label for t, p in zip(truth, prediction))
        fn = sum(t == label and p != label for t, p in zip(truth, prediction))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores)) if scores else 0.0


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
    classifier: SupportTokenMixer | None,
    classifier_mode: str,
    text_of,
    device: torch.device,
    executor: ThreadPoolExecutor | None = None,
    batch: dict | None = None,
) -> dict:
    """``batch`` lets a prefetching loader hand over an already-collated batch for these episodes;
    otherwise the windows are loaded and collated here, on the calling thread."""
    if batch is None:
        positions = episode_positions(episodes, corpus)
        batch = collate(_load_items(dataset, positions, executor))
    encoded = encode_batch(encoder, batch, device)
    pooled, descriptor = recording_rows(encoded)

    rows = split_encoded(pooled, descriptor, episodes, corpus)
    text = episode_text(episodes, corpus, text_of, device)
    query = rows["query_feature"].squeeze(1)
    if classifier_mode == "neighbors":
        if any(episode.is_zero_shot for episode in episodes):
            raise ValueError("differentiable neighbors only supports enrolled episodes")
        logits, weight = differentiable_neighbor_logits(
            query, rows["support_feature"], text["support_bound"], rows["support_mask"],
            text["candidate_mask"], temperature=TAU_SUPPORT,
        )
        output = {"logits": logits, "support_weight": weight}
    else:
        if classifier is None:
            raise ValueError("token-mixer mode requires a classifier")
        output = classifier(
            is_zero_shot=torch.tensor([episode.is_zero_shot for episode in episodes],
                                      dtype=torch.bool, device=device),
            query_feature=query,
            support_feature=rows["support_feature"],
            support_label_text=text["support_label_text"],
            support_bound=text["support_bound"], support_mask=rows["support_mask"],
            support_pair_slot=text["support_pair_slot"],
            candidate_text=text["candidate_text"], candidate_mask=text["candidate_mask"],
            candidate_slot=text["candidate_slot"],
        )
    loss = episode_loss(output["logits"], episodes, text)
    return {**loss, **output, "pooled": pooled, "text": text, "rows": rows,
            "readout": classifier_mode}


def prediction_telemetry(result: dict, episodes: list[Episode]) -> dict[str, float]:
    """Cheap behavioral diagnostics for the learned residual and closed-form support rule."""
    mask = result["text"]["candidate_mask"]
    target = torch.tensor(
        [episode.gt_slot for episode in episodes], device=result["logits"].device,
    )
    learned = result["logits"].detach().masked_fill(~mask, float("-inf")).argmax(dim=-1)
    weights = result.get("support_weight")
    if weights is None:
        weights = result["logits"].new_zeros(result["rows"]["support_mask"].shape)
    weights = weights.detach()
    # Soft weights may be exactly zero for reasons other than padding. The structural mask is the
    # honest definition of whether an episode has support, especially for direct k=0 episodes.
    support_mask = result["rows"]["support_mask"]
    has_support = support_mask.any(dim=1)
    entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=1)
    entropy = torch.where(has_support, entropy, torch.zeros_like(entropy))
    effective_rows = torch.where(has_support, entropy.exp(), torch.zeros_like(entropy))
    metrics = {
        "accuracy/learned": float(learned.eq(target).float().mean()),
        "vote/top1_support_mass": float(weights.max(dim=1).values.mean()) if weights.shape[1] else 0.0,
        "vote/total_support_mass_min": float(weights.sum(dim=1).min()),
        "vote/total_support_mass_mean": float(weights.sum(dim=1).mean()),
        "vote/effective_support_rows": float(effective_rows.mean()),
        "vote/support_entropy": float(entropy.mean()),
        "sampler/candidate_padding_fraction": float((~mask).float().mean()),
    }
    return metrics


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
    classifier,
    classifier_mode: str,
    corpus: SupportCorpus,
    dataset: PretrainDataset,
    collate,
    text_of,
    device: torch.device,
    episodes_count: int,
    episodes_per_step: int,
    seed: int,
    draw_kwargs: dict,
    executor: ThreadPoolExecutor | None,
    deployment_matched: bool = False,
) -> dict[str, float]:
    """Evaluate a fixed subject-held-out episode draw without consuming test datasets."""
    was_encoder_training = encoder.training
    was_classifier_training = classifier.training if classifier is not None else False
    encoder.eval()
    if classifier is not None:
        classifier.eval()
    rng = np.random.default_rng(seed)
    # Sparse held-out datasets can have only a small fraction of drawable query executions. A
    # larger retry budget preserves their predeclared representation in the fixed panel instead of
    # silently dropping the source or substituting an easier one. This runs outside the hot path.
    episodes, sampler = draw_batch(
        corpus, rng, batch_size=episodes_count, max_attempts_per_episode=256, **draw_kwargs,
    )
    rows: list[dict] = []
    losses: list[float] = []
    ranks: list[float] = []
    group_sizes: list[int] = []
    loss_group_sizes: list[int] = []
    # Zero-support and enrolled recognition are different information conditions.  Keep their
    # panels separate so checkpoint selection cannot trade one off against the other invisibly.
    learned_by_regime: dict[bool, dict[str, tuple[list[str], list[str]]]] = {
        False: {}, True: {},
    }
    if deployment_matched:
        by_set: dict[int, list[Episode]] = {}
        for episode in episodes:
            by_set.setdefault(episode.support_set_id, []).append(episode)
        support_sets = list(by_set.values())
        validation_groups = [
            [episode for support_set in support_sets[start:start + episodes_per_step]
             for episode in support_set]
            for start in range(0, len(support_sets), episodes_per_step)
        ]
        loss_group_sizes = [
            len(support_sets[start:start + episodes_per_step])
            for start in range(0, len(support_sets), episodes_per_step)
        ]
    else:
        validation_groups = [
            episodes[start:start + episodes_per_step]
            for start in range(0, len(episodes), episodes_per_step)
        ]
        loss_group_sizes = [len(group) for group in validation_groups]
    for group in validation_groups:
        group_sizes.append(len(group))
        with autocast(device):
            result = run_step(
                episodes=group, corpus=corpus, dataset=dataset, collate=collate,
                encoder=encoder, classifier=classifier, classifier_mode=classifier_mode,
                text_of=text_of, device=device, executor=executor,
            )
        rows.append(prediction_telemetry(result, group))
        losses.append(float(result["loss"]))
        ranks.append(effective_rank(result["pooled"]))
        candidate_mask = result["text"]["candidate_mask"]
        learned_slot = result["logits"].masked_fill(
            ~candidate_mask, float("-inf"),
        ).argmax(dim=-1).tolist()
        for index, episode in enumerate(group):
            recording = corpus.recordings[episode.query]
            learned = learned_by_regime[episode.is_zero_shot].setdefault(
                recording.dataset, ([], []),
            )
            learned[0].append(recording.label)
            learned[1].append(episode.candidates[int(learned_slot[index])])
    if was_encoder_training:
        encoder.train()
    if was_classifier_training:
        classifier.train()
    keys = sorted({key for row in rows for key in row})
    learned_dataset_f1 = {
        regime: {
            dataset: _macro_f1_names(truth, prediction)
            for dataset, (truth, prediction) in rows.items()
        }
        for regime, rows in learned_by_regime.items()
    }
    enrolled_f1 = learned_dataset_f1[False]
    zero_f1 = learned_dataset_f1[True]
    return {
        "validation/loss": float(np.average(losses, weights=loss_group_sizes)),
        "validation/encoder_effective_rank": float(np.mean(ranks)),
        "validation/enrolled_dataset_macro_f1": (
            float(np.mean(list(enrolled_f1.values()))) if enrolled_f1 else float("nan")
        ),
        "validation/zero_shot_dataset_macro_f1": (
            float(np.mean(list(zero_f1.values()))) if zero_f1 else float("nan")
        ),
        "validation/learned_dataset_macro_f1": (
            float(np.mean(list(enrolled_f1.values()))) if enrolled_f1 else float("nan")
        ),
        "validation/selection_dataset_macro_f1": (
            float(np.mean(list(enrolled_f1.values()))) if enrolled_f1 else float("-inf")
        ),
        **{
            f"validation/enrolled/dataset/{dataset}/learned_macro_f1": value
            for dataset, value in enrolled_f1.items()
        },
        **{
            f"validation/zero_shot/dataset/{dataset}/learned_macro_f1": value
            for dataset, value in zero_f1.items()
        },
        **{f"validation/{key}": float(np.average([row[key] for row in rows], weights=group_sizes))
           for key in keys},
        **{f"validation/{key}": value for key, value in sampler.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a", type=Path, default=None,
                        help="optional Phase-A checkpoint for the warm-start arm. Omit for the "
                             "default end-to-end-from-scratch recipe, which is what every compact "
                             "checkpoint on disk actually used")
    parser.add_argument("--frontend", choices=("fixed", "learnable", "continuous", "multispan"),
                        default="fixed",
                        help="front end for a from-scratch encoder; the design of record is fixed")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="replace known artifacts in a non-empty output directory")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume model, optimizer, schedule and RNG state from a checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=35_000)
    parser.add_argument("--episodes-per-step", type=int, default=8,
                        help="independently sampled support sets per optimizer step")
    parser.add_argument("--support-size", type=int, default=DEFAULT_SUPPORT,
                        help="legacy non-deployment sampler only; the paper path uses C * k rows")
    parser.add_argument("--p-gt-present", type=float, default=None)
    parser.add_argument("--same-subject-probability", type=float, default=None,
                        help="requested same-user share when feasible; staged default 0.2")
    parser.add_argument("--enrollment-k", type=int, nargs="+", default=list(DEFAULT_ENROLLMENT_K),
                        help="deployment enrollment executions per candidate sampled by training")
    parser.add_argument("--queries-per-support-set", type=int,
                        default=DEFAULT_QUERIES_PER_SUPPORT_SET)
    parser.add_argument("--windows-per-execution", type=int,
                        default=DEFAULT_WINDOWS_PER_EXECUTION,
                        help="maximum windows averaged into each training support execution")
    parser.add_argument("--label-subset", type=int, nargs=2, default=None,
                        help="candidate-count range; defaults to the support sampler's balanced range")
    parser.add_argument("--mode", choices=("compatible", "near_miss", "unfiltered"),
                        default="compatible")
    parser.add_argument("--neutral-acquisition-text", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="override checkpoint acquisition-text mode; omitted inherits Phase-A")
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=None,
                        help="default: train encoder and classifier together")
    parser.add_argument("--classifier", choices=("token_mixer", "neighbors"),
                        default="token_mixer",
                        help="token_mixer is the full semantic head; neighbors is the simple "
                             "differentiable support-vote control")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=None,
                        help="encoder LR multiplier on --lr; default 1.0 for the from-scratch "
                             "neighbor encoder and 0.05 for head/legacy training")
    parser.add_argument("--frontend-lr-scale", type=float, default=1.0,
                        help="extra multiplier on the encoder LR for the continuous frontend's "
                             "analysis bank (kernel coefficients, envelopes, gains); the fixed "
                             "filterbank has no such parameters")
    parser.add_argument("--frontend-reg-weight", type=float, default=0.0,
                        help="weight of the continuous frontend's pull toward its Gabor "
                             "initialisation; 0 = fully learned bank, as in the 2026-09-08 runs")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--data-seed", type=int, default=20260901,
                        help="fixed subject split/corpus seed; keep constant across model replicates")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=2_500)
    parser.add_argument("--val-episodes", type=int, default=None,
                        help="fixed support-set count for staged readouts (default: eight per "
                             "eligible held-out dataset); legacy episode count otherwise "
                             "(default 64)")
    parser.add_argument("--val-repeats-per-dataset", type=int, default=8,
                        help="internal subject-held-out support sets per eligible training dataset when "
                             "--val-episodes is omitted")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--milestone-every", type=int, default=5_000,
                        help="retain a numbered checkpoint at this interval; last.pt is still "
                             "updated at --checkpoint-every")
    parser.add_argument("--calib-batches", type=int, default=20)
    parser.add_argument("--calib-batch-size", type=int, default=256)
    parser.add_argument("--loader-workers", type=int, default=4,
                        help="forked worker PROCESSES that draw, load and collate upcoming steps "
                             "while the GPU trains (PrefetchLoader). 0 = synchronous on the main "
                             "thread. The episode sequence is identical for any value. Thread "
                             "pools were removed: the per-window work holds the GIL and 8 "
                             "threads measured 2.5-3.5x slower than none (2026-09-05)")
    parser.add_argument("--max-per-stream", type=int, default=None)
    parser.add_argument("--patch-seconds", type=float, default=PATCH_SECONDS,
                        help="single filterbank patch duration; ignored with --resolutions")
    parser.add_argument("--spans", type=float, nargs="+", default=list(MS_SPANS_S),
                        metavar="SECONDS",
                        help="multispan frontend: physical kernel spans, one token grid per span")
    parser.add_argument("--frames-per-span", type=int, default=MS_FRAMES_PER_SPAN,
                        help="multispan frontend: envelope frames per span (token stride = "
                             "span / this)")
    parser.add_argument("--resolutions", type=float, nargs="+", default=None,
                        metavar="SECONDS",
                        help="encode each recording on two or more explicitly tagged "
                             "physical-time patch grids")
    parser.add_argument("--smoke", action="store_true",
                        help="three steps on capped real data with telemetry; launches nothing long")
    args = parser.parse_args()
    if args.val_episodes is None:
        args.val_episodes = 64
    if args.label_subset is None:
        args.label_subset = list(DEFAULT_LABEL_SUBSET)
    if args.encoder_lr_scale is None:
        # A random end-to-end encoder needs the base optimizer LR.  A warm-started Phase-A
        # encoder is deliberately updated more conservatively.
        args.encoder_lr_scale = 0.05 if args.phase_a is not None or args.resume is not None else 1.0
    if args.same_subject_probability is None:
        args.same_subject_probability = DEFAULT_SAME_SUBJECT_PROBABILITY
    if args.freeze_encoder is None:
        args.freeze_encoder = False
    if args.freeze_encoder and args.phase_a is None and args.resume is None:
        parser.error("frozen head training requires --phase-a ENCODER_CHECKPOINT")
    if args.p_gt_present is None:
        args.p_gt_present = DEFAULT_P_GT_PRESENT
    if args.classifier == "neighbors":
        # A neighbor vote has no candidate-only k=0 path.  Make the control honest rather than
        # quietly giving it the semantic token mixer's zero-shot machinery.
        args.p_gt_present = 1.0

    if args.smoke:
        args.steps = min(args.steps, 3)
        args.warmup_steps = min(args.warmup_steps, max(0, args.steps - 1))
        args.log_every = 1
        args.val_every = args.steps
        args.val_episodes = min(args.val_episodes, 4)
        args.checkpoint_every = args.steps
        args.milestone_every = args.steps
        args.calib_batches = min(args.calib_batches, 1)
        args.calib_batch_size = min(args.calib_batch_size, 32)
        args.max_per_stream = args.max_per_stream or 200

    if args.steps < 1 or not 0 <= args.warmup_steps < args.steps:
        parser.error("steps must be positive and warmup-steps must be in [0, steps)")
    if min(args.episodes_per_step, args.support_size, args.queries_per_support_set,
           args.windows_per_execution, args.log_every, args.val_every, args.val_episodes,
           args.checkpoint_every, args.milestone_every,
           args.calib_batches, args.calib_batch_size, args.val_repeats_per_dataset) < 1:
        parser.error("episode, support, validation, checkpoint and calibration counts must be positive")
    if args.loader_workers < 0:
        parser.error("loader-workers must be nonnegative")
    if not 0.0 <= args.p_gt_present <= 1.0:
        parser.error("p-gt-present must be in [0,1]")
    if not 0.0 <= args.same_subject_probability <= 1.0:
        parser.error("same-subject-probability must be in [0,1]")
    if args.encoder_lr_scale <= 0:
        parser.error("encoder-lr-scale must be positive")
    if args.frontend_lr_scale <= 0 or args.frontend_reg_weight < 0:
        parser.error("frontend-lr-scale must be positive and frontend-reg-weight nonnegative")
    if not args.enrollment_k or any(value < 1 for value in args.enrollment_k):
        parser.error("enrollment-k values must be positive")
    if args.label_subset[0] < 2 or args.label_subset[1] < args.label_subset[0]:
        parser.error("label-subset must be LOW HIGH with 2 <= LOW <= HIGH")
    if args.label_subset[1] > TokenMixerConfig().max_candidates:
        parser.error("label-subset HIGH exceeds the token mixer's candidate capacity")
    if args.patch_seconds <= 0:
        parser.error("patch-seconds must be positive")
    if args.resolutions is not None:
        if len(args.resolutions) < 2 or any(value <= 0 for value in args.resolutions):
            parser.error("resolutions requires at least two positive durations")
        if len(set(args.resolutions)) != len(args.resolutions):
            parser.error("resolutions must not contain duplicate durations")
        args.resolutions = sorted(args.resolutions)
        if args.resolutions[-1] < 1.75 * args.resolutions[0]:
            parser.error("resolutions must span at least a 1.75x duration ratio")
    if args.frontend == "continuous" and args.resolutions is not None:
        parser.error(
            "the continuous frontend emits one physical patch grid (set with --patch-seconds); "
            "multi-resolution token grids are not defined for it yet"
        )
    if args.frontend == "multispan":
        if args.resolutions is not None:
            parser.error("--resolutions is a filterbank token-grid option; the multispan frontend "
                         "defines its own grid with --spans")
        if len(args.spans) < 2 or any(s <= 0 for s in args.spans) \
                or len(set(args.spans)) != len(args.spans):
            parser.error("spans must be at least two distinct positive durations in seconds")
        if args.frames_per_span < 1:
            parser.error("frames-per-span must be positive")
        args.spans = sorted(float(s) for s in args.spans)

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
    # Resolve acquisition conditioning before datasets are constructed.  A warm-start must use
    # precisely the text regime embedded in its checkpoint; making callers repeat that bit was a
    # fragile and unnecessary source of mismatched Phase-A/heads runs.
    resume_blob = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume is not None else None
    )
    source_config = (
        dict(resume_blob["config"]) if resume_blob is not None
        else dict(torch.load(args.phase_a, map_location="cpu", weights_only=False)["config"])
        if args.phase_a is not None else None
    )
    checkpoint_neutral = (
        bool(source_config.get("neutral_acquisition_text", False))
        if source_config is not None else False
    )
    if args.neutral_acquisition_text is None:
        args.neutral_acquisition_text = checkpoint_neutral
    elif source_config is not None and bool(args.neutral_acquisition_text) != checkpoint_neutral:
        parser.error(
            "--neutral-acquisition-text contradicts the Phase-A/support checkpoint; omit the "
            "flag to inherit the checkpoint's acquisition-conditioning regime"
        )
    source = capture_source_provenance(args.out, write=False, roots=PROVENANCE_ROOTS)
    runtime = capture_runtime_provenance(device)
    runtime["mixed_precision"] = "bfloat16" if device.type == "cuda" else "float32"
    runtime["dynamic_loss_scaling"] = False

    # The classifier has no --datasets override, so the active roster is the only path; gate it
    # anyway so a future override cannot bypass the retirement.
    deployment_policy.assert_no_retired_sources(deployment_policy.SUPERVISED_HEAD_TRAIN_DATASETS)
    index = CorpusIndex(
        max_per_stream=args.max_per_stream, seed=args.data_seed,
        datasets=deployment_policy.SUPERVISED_HEAD_TRAIN_DATASETS, alignment="native",
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
    base_collate = (
        MultiResolutionCollate(fixed_patch_seconds=tuple(args.resolutions))
        if args.resolutions is not None
        else MultiScaleCollate(fixed_patch_seconds=args.patch_seconds)
    )
    collate = SupportCollate(base_collate)
    # Calibration and validation load on this thread; training steps come from the prefetcher.
    executor = None

    draw_kwargs = {
        "p_gt_present": args.p_gt_present,
        "same_subject_probability": args.same_subject_probability,
        "label_subset": tuple(args.label_subset),
        "mode": args.mode,
        "semantic_zero_shot": True,
        "deployment_matched": True,
        "enrollment_k": tuple(args.enrollment_k),
        "queries_per_support_set": args.queries_per_support_set,
        "windows_per_execution": args.windows_per_execution,
    }
    # Fork the prefetch workers NOW, before the encoder, the text tower or any library thread
    # exists: forking a process that has live threads can deadlock the child on a lock a thread
    # held at fork time. The workers only need the corpus, the dataset and the collate.
    loader = (
        PrefetchLoader(
            corpus, dataset, collate, data_seed=args.data_seed,
            batch_size=args.episodes_per_step, draw_kwargs=draw_kwargs,
            workers=args.loader_workers,
            start_step=(int(resume_blob["step"]) if resume_blob is not None else 0) + 1,
        )
        if args.loader_workers else None
    )

    if resume_blob is not None:
        if resume_blob.get("architecture_version") != "support_token_mixer_v1":
            raise SystemExit(
                "cannot resume a retired scalar-reranker checkpoint; start a new token-mixer run"
            )
        encoder_config = dict(resume_blob["config"])
        encoder = build_encoder(resume_blob, device, training=True)
        spec = AttentionSpec(**resume_blob["attention_spec"])
        classifier = (
            SupportTokenMixer(spec, TokenMixerConfig(**resume_blob["classifier_config"])).to(device)
            if args.classifier == "token_mixer" else None
        )
        if classifier is not None:
            classifier.load_state_dict(resume_blob["classifier"])
        print(f"[compare] resuming {args.resume} at step {resume_blob['step']}", flush=True)
    elif args.phase_a is None:
        # The design-of-record recipe: one stage, everything but the frozen text tower and the
        # filterbank's normalisation statistics starts random.
        multispan = args.frontend == "multispan"
        # The token grid's durations: the frontend's spans for multispan, the collate's patch
        # grids for --resolutions, otherwise one grid (no duration embedding).
        grid_durations = (
            [float(s) for s in args.spans] if multispan
            else [float(v) for v in args.resolutions] if args.resolutions is not None
            else None
        )
        frontend_kwargs = None
        if args.frontend == "continuous":
            frontend_kwargs = {"patch_seconds": float(args.patch_seconds)}
        elif multispan:
            frontend_kwargs = {
                "spans": tuple(grid_durations), "frames_per_span": int(args.frames_per_span),
                # RoPE's fastest period spans two of the finest group's frame strides.
                "rope_min_period": 2.0 * grid_durations[0] / args.frames_per_span,
            }
        encoder, encoder_config = build_random_encoder(
            device, args.frontend, neutral_acquisition_text=args.neutral_acquisition_text,
            duration_range=(
                (grid_durations[0], grid_durations[-1]) if grid_durations is not None else None
            ),
            num_resolutions=(len(grid_durations) if grid_durations is not None else 2),
            frontend_kwargs=frontend_kwargs,
        )
        encoder_config.update({
            "multiresolution": grid_durations is not None,
            "token_grid_owner": "frontend" if multispan else "collate",
            "use_duration_embedding": grid_durations is not None,
            "num_resolutions": len(grid_durations) if grid_durations is not None else 2,
            **({"spans": grid_durations, "frames_per_span": int(args.frames_per_span),
                "rope_min_period": frontend_kwargs["rope_min_period"]} if multispan else {}),
            "patch_seconds": float(args.patch_seconds),
            "short_patch_choices": [
                float(grid_durations[0]) if grid_durations is not None
                else 0.4
            ],
            "long_patch_choices": [
                float(grid_durations[-1]) if grid_durations is not None
                else 1.5
            ],
            "eval_resolutions": (
                list(grid_durations) if grid_durations is not None else [0.5, 1.5]
            ),
        })
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
        checkpoint_patch_seconds = float(encoder_config.get("patch_seconds", PATCH_SECONDS))
        if checkpoint_frontend == "continuous" and not math.isclose(
            checkpoint_patch_seconds, float(args.patch_seconds),
        ):
            raise SystemExit(
                f"--patch-seconds={args.patch_seconds} does not match the continuous checkpoint's "
                f"patch grid of {checkpoint_patch_seconds} s"
            )
        encoder = build_encoder(checkpoint, device, training=True, learnable_recording_pool=True)
        encoder_config["learnable_recording_pool"] = True
        print(f"[compare] warm-started from {args.phase_a}", flush=True)
    if resume_blob is None:
        spec = AttentionSpec(d_model=encoder.d_model, n_heads=4, ffn_mult=2, dropout=0.1)
        classifier = SupportTokenMixer(spec, TokenMixerConfig()).to(device) \
            if args.classifier == "token_mixer" else None
    print(f"[compare] classifier={args.classifier}", flush=True)
    if hasattr(encoder, "mask_token"):
        encoder.mask_token.requires_grad_(False)
    if args.freeze_encoder:
        encoder.requires_grad_(False)
        encoder.eval()

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

    text_of = make_label_text(
        list(corpus.all_labels) + list(val_corpus.all_labels), device,
    )

    classifier_params = ([] if classifier is None else
                         [parameter for parameter in classifier.parameters() if parameter.requires_grad])
    encoder_params = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    # The continuous frontend's analysis bank gets its own group so its learning rate can be scaled
    # and scheduled like the others. The fixed filterbank has no such parameters and adds no group,
    # which keeps every existing checkpoint's optimizer state resumable.
    frontend_params = (
        [parameter for parameter in frontend.adaptation_parameters() if parameter.requires_grad]
        if frontend is not None and hasattr(frontend, "adaptation_parameters") else []
    )
    frontend_ids = {id(parameter) for parameter in frontend_params}
    encoder_trunk_params = [p for p in encoder_params if id(p) not in frontend_ids]
    # Base rates live here, not in the groups: ``load_state_dict`` replaces group dicts wholesale.
    base_lrs: dict[str, float] = {"encoder": args.lr * args.encoder_lr_scale}
    param_groups = [{"name": "encoder", "params": encoder_trunk_params,
                     "lr": base_lrs["encoder"]}]
    if classifier_params:
        base_lrs["classifier"] = args.lr
        param_groups.insert(0, {"name": "classifier", "params": classifier_params,
                                "lr": base_lrs["classifier"]})
    if frontend_params:
        base_lrs["frontend"] = args.lr * args.encoder_lr_scale * args.frontend_lr_scale
        param_groups.append({"name": "frontend", "params": frontend_params,
                             "lr": base_lrs["frontend"]})
    optimizer = make_optimizer(param_groups, weight_decay=args.weight_decay, device=device)

    trajectory = {
        key: value for key, value in {
            "steps": args.steps,
            "episodes_per_step": args.episodes_per_step,
            "frontend": args.frontend,
            "patch_seconds": args.patch_seconds,
            "resolutions": args.resolutions,
            "support_size": args.support_size,
            "enrollment_k": list(args.enrollment_k),
            "queries_per_support_set": args.queries_per_support_set,
            "windows_per_execution": args.windows_per_execution,
            "p_gt_present": args.p_gt_present,
            "same_subject_probability": args.same_subject_probability,
            "label_subset": list(args.label_subset),
            "mode": args.mode,
            "neutral_acquisition_text": args.neutral_acquisition_text,
            "classifier": args.classifier,
            "freeze_encoder": args.freeze_encoder,
            "lr": args.lr,
            "encoder_lr_scale": args.encoder_lr_scale,
            "frontend_lr_scale": args.frontend_lr_scale,
            "frontend_reg_weight": args.frontend_reg_weight,
            "spans": list(args.spans) if args.frontend == "multispan" else None,
            "frames_per_span": args.frames_per_span if args.frontend == "multispan" else None,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "data_seed": args.data_seed,
        }.items()
    }
    start_step = 0
    if resume_blob is not None:
        saved_trajectory = dict(resume_blob.get("trajectory") or {})
        saved_trajectory.setdefault("classifier", "token_mixer")
        saved_trajectory.setdefault("freeze_encoder", False)
        saved_trajectory.setdefault("enrollment_k", list(DEFAULT_ENROLLMENT_K))
        saved_trajectory.setdefault("queries_per_support_set", DEFAULT_QUERIES_PER_SUPPORT_SET)
        saved_trajectory.setdefault("windows_per_execution", DEFAULT_WINDOWS_PER_EXECUTION)
        saved_trajectory.setdefault("patch_seconds", PATCH_SECONDS)
        legacy_resolutions = saved_trajectory.pop("resolution_pair", None)
        saved_trajectory.setdefault("resolutions", legacy_resolutions)
        saved_trajectory.setdefault("frontend_lr_scale", 1.0)
        saved_trajectory.setdefault("frontend_reg_weight", 0.0)
        saved_trajectory.setdefault("spans", None)
        saved_trajectory.setdefault("frames_per_span", None)
        if saved_trajectory != trajectory:
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

    latest_validation: dict[str, float] | None = resume_blob.get("validation") if resume_blob else None
    best_accuracy = float(resume_blob.get("best_validation_accuracy", -1.0)) if resume_blob else -1.0
    best_loss = float(resume_blob.get("best_validation_loss", float("inf"))) \
        if resume_blob else float("inf")
    best_dataset_f1 = float(resume_blob.get("best_validation_dataset_macro_f1", -1.0)) \
        if resume_blob else -1.0

    def payload(step: int) -> dict:
        return {
            "config": config,
            "encoder": encoder.state_dict(),
            "architecture_version": "support_token_mixer_v1",
            "classifier": None if classifier is None else classifier.state_dict(),
            "classifier_config": None if classifier is None else dataclasses.asdict(classifier.cfg),
            "attention_spec": dataclasses.asdict(spec),
            "args": {key: (str(value) if isinstance(value, Path) else value)
                     for key, value in vars(args).items()},
            "trajectory": trajectory,
            "phase_a": None if args.phase_a is None else str(args.phase_a),
            "step": step,
            "validation": latest_validation,
            "best_validation_accuracy": best_accuracy,
            "best_validation_loss": best_loss,
            "best_validation_dataset_macro_f1": best_dataset_f1,
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

    def run_validation(step: int) -> dict[str, float]:
        nonlocal latest_validation, best_accuracy, best_loss, best_dataset_f1
        latest_validation = validate(
            encoder=encoder, classifier=classifier, classifier_mode=args.classifier,
            corpus=val_corpus, dataset=val_dataset,
            collate=collate, text_of=text_of, device=device,
            episodes_count=args.val_episodes, episodes_per_step=args.episodes_per_step,
            seed=args.data_seed + 91_003, draw_kwargs=draw_kwargs,
            executor=executor,
            deployment_matched=True,
        )
        latest_validation["step"] = float(step)
        with log_path.open("a") as handle:
            handle.write(json.dumps({"kind": "validation", **latest_validation}) + "\n")
        best_accuracy = max(best_accuracy, latest_validation["validation/accuracy/learned"])
        score = latest_validation["validation/selection_dataset_macro_f1"]
        loss = latest_validation["validation/loss"]
        improved = score > best_dataset_f1 or (
            math.isclose(score, best_dataset_f1) and loss < best_loss
        )
        if improved:
            best_dataset_f1 = score
            best_loss = loss
            _atomic_torch_save(payload(step), args.out / "best_internal.pt")
        return latest_validation

    if resume_blob is None:
        run_validation(0)
        _atomic_torch_save(payload(0), args.out / "initial.pt")

    for step in range(start_step + 1, args.steps + 1):
        scale = _learning_rate_scale(step, warmup=args.warmup_steps, total=args.steps)
        for group in optimizer.param_groups:
            group["lr"] = base_lrs[group["name"]] * scale

        if loader is not None:
            episodes, telemetry, batch = loader.get(step)
        else:
            episodes, telemetry = draw_batch(
                corpus, episode_rng(args.data_seed, step),
                batch_size=args.episodes_per_step, **draw_kwargs,
            )
            batch = None
        log_step = step % args.log_every == 0 or step == 1
        if frontend is not None and hasattr(frontend, "request_runtime_telemetry"):
            frontend.request_runtime_telemetry(log_step)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device):
            result = run_step(
                episodes=episodes, corpus=corpus, dataset=dataset, collate=collate,
                encoder=encoder, classifier=classifier, classifier_mode=args.classifier,
                text_of=text_of, device=device, executor=executor, batch=batch,
            )
        frontend_reg = None
        if args.frontend_reg_weight > 0 and frontend is not None \
                and hasattr(frontend, "adaptation_regularization"):
            frontend_reg = frontend.adaptation_regularization()
        total_loss = (result["loss"] if frontend_reg is None
                      else result["loss"] + args.frontend_reg_weight * frontend_reg)
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(f"non-finite comparison loss at step {step}")
        total_loss.backward()
        encoder_grad = _parameter_grad_norm(encoder_params) if log_step else 0.0
        classifier_grad = _parameter_grad_norm(classifier_params) if log_step else 0.0
        duration_parameters = (
            list(encoder.duration_proj.parameters()) + [encoder.duration_gate_logit]
            if getattr(encoder, "use_duration_embedding", False) else []
        )
        duration_grad = _parameter_grad_norm(duration_parameters) if log_step else 0.0
        frontend_grad = _parameter_grad_norm(frontend_params) if log_step and frontend_params else 0.0
        frontend_summary = (
            {**frontend.adaptation_summary(), **frontend.runtime_summary()}
            if log_step and frontend_params and hasattr(frontend, "adaptation_summary") else {}
        )
        preclip = float(torch.nn.utils.clip_grad_norm_(
            classifier_params + encoder_params, args.grad_clip, error_if_nonfinite=True,
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
                "gradient/classifier_norm": classifier_grad,
                "gradient/duration_embedding_norm": duration_grad,
                "gradient/frontend_norm": frontend_grad,
                "loss/frontend_reg": (
                    float(frontend_reg.detach()) if frontend_reg is not None else 0.0
                ),
                "gradient/total_preclip_norm": preclip,
                "gradient/clip_coefficient": min(1.0, args.grad_clip / max(preclip, 1e-12)),
                "lr/classifier": next((group["lr"] for group in optimizer.param_groups
                                        if group["name"] == "classifier"), 0.0),
                "lr/encoder": next((group["lr"] for group in optimizer.param_groups
                                     if group["name"] == "encoder"), 0.0),
                "lr/frontend": next((group["lr"] for group in optimizer.param_groups
                                      if group["name"] == "frontend"), 0.0),
                "encoder/duration_gate": (
                    float(torch.sigmoid(encoder.duration_gate_logit.detach()))
                    if getattr(encoder, "use_duration_embedding", False) else 0.0
                ),
                "elapsed_s": round(time.perf_counter() - started, 1),
                **telemetry,
                **behavior,
                **({} if classifier is None else classifier.telemetry()),
                **frontend_summary,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            accuracy_text = f"{row['accuracy/learned']:.2f}"
            if "accuracy/base" in row:
                accuracy_text += f"/{row['accuracy/base']:.2f}"
            print(
                f"[compare] step {step:>6} loss {row['loss']:.4f} "
                f"(few {row['loss/few_shot_ce']:.4f} zero {row['loss/zero_shot_ce']:.4f}) "
                f"acc {accuracy_text} rank {row['encoder/effective_rank']:.1f} "
                f"K {row['sampler/mean_support_size']:.1f}",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.steps:
            report = run_validation(step)
            message = (
                f"[compare] validation step {step}: model "
                f"{report['validation/accuracy/learned']:.3f}, "
                f"dataset-macro F1 {report['validation/selection_dataset_macro_f1']:.3f}, "
                f"loss {report['validation/loss']:.3f}"
            )
            if "validation/accuracy/base" in report:
                message += f", neighbor floor {report['validation/accuracy/base']:.3f}"
            print(message, flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            _atomic_torch_save(payload(step), args.out / "last.pt")
        if step % args.milestone_every == 0 or step == args.steps:
            milestone_dir = args.out / "checkpoints"
            milestone_dir.mkdir(exist_ok=True)
            _atomic_torch_save(payload(step), milestone_dir / f"step_{step:06d}.pt")

    if loader is not None:
        loader.close()
    _atomic_torch_save(payload(args.steps), args.out / "last.pt")
    print(f"[compare] wrote {args.out / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
