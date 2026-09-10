"""Bounded, full-corpus training-throughput probe; never writes model checkpoints.

Run with OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m training.support_classifier.profile --out /tmp/profile.json.
Timing excludes setup and validation. Episode hashes make before/after sampling changes visible.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from data.scripts.curate.deployment_policy import EXPANDED_PHASE_A_TRAIN_DATASETS
from model.blocks import AttentionSpec
from model.support.comparator import ComparatorConfig, SupportComparator
from training.support_classifier.corpus import support_corpus_from_index
from training.support_classifier.train import (
    PrefetchLoader, build_dataset, calibrate_frontend, make_label_text, make_optimizer, run_step,
)
from training.support_classifier.collate import SupportCollate
from training.support_classifier.encoding import autocast, build_random_encoder
from training.tokenizer.pretrain_data import CorpusIndex, MultiScaleCollate, PATCH_SECONDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or min(args.workers) < 1 or not 0 <= args.warmup < args.steps <= 100:
        parser.error("positive worker counts and 0 <= warmup < steps <= 100 required")
    torch.set_num_threads(2)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    index = CorpusIndex(datasets=EXPANDED_PHASE_A_TRAIN_DATASETS, alignment="native",
                        max_per_stream=None, seed=20260901)
    corpus = support_corpus_from_index(index)
    args.neutral_acquisition_text = True
    dataset = build_dataset(index, args)
    collate = SupportCollate(MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS))
    # Fork CPU workers before constructing any CUDA models or text-tower threads.
    loaders = {n: PrefetchLoader(corpus, dataset, collate, data_seed=20260901,
                                batch_size=args.episodes, draw_kwargs={}, workers=n)
               for n in args.workers}
    results = []
    try:
        for workers, loader in loaders.items():
            torch.manual_seed(7)
            encoder, _ = build_random_encoder(device, "fixed", neutral_acquisition_text=True)
            encoder.train()
            encoder.mask_token.requires_grad_(False)
            comparator = SupportComparator(AttentionSpec(d_model=128, n_heads=4, ffn_mult=2,
                                                        dropout=.1), ComparatorConfig()).to(device)
            calibrate_frontend(encoder, dataset, corpus, collate, np.random.default_rng(7), device,
                               batches=1, batch_size=128, executor=None)
            text = make_label_text(corpus.all_labels, device)
            parameters = [p for m in (encoder, comparator) for p in m.parameters() if p.requires_grad]
            optimizer = make_optimizer([
                {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": 1.5e-5},
                {"params": comparator.parameters(), "lr": 3e-4},
            ], weight_decay=.05, device=device)
            times, waits = [], []
            digest = hashlib.sha256()
            torch.cuda.reset_peak_memory_stats()
            for step in range(1, args.steps + 1):
                torch.cuda.synchronize()
                started = time.perf_counter()
                episodes, _, batch = loader.get(step)
                if args.pin_memory:
                    batch = {k: v.pin_memory() if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                wait = time.perf_counter() - started
                optimizer.zero_grad(set_to_none=True)
                with autocast(device):
                    result = run_step(episodes=episodes, corpus=corpus, dataset=dataset,
                                      collate=collate, encoder=encoder, comparator=comparator,
                                      text_of=text, device=device, center=True, batch=batch)
                if not bool(torch.isfinite(result["loss"])):
                    raise FloatingPointError("non-finite profile loss")
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                if step > args.warmup:
                    times.append(time.perf_counter() - started)
                    waits.append(wait)
                digest.update(json.dumps([dataclasses.asdict(e) for e in episodes], sort_keys=True).encode())
            record = {"workers": workers, "episodes": args.episodes,
                      "mean_step_ms": 1000 * float(np.mean(times)),
                      "median_step_ms": 1000 * float(np.median(times)),
                      "mean_loader_wait_ms": 1000 * float(np.mean(waits)),
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "pinned": batch["patches"].is_pinned(), "episode_sha256": digest.hexdigest(),
                      "loss": float(result["loss"].detach())}
            print(json.dumps(record), flush=True)
            results.append(record)
            del encoder, comparator, optimizer, parameters, result
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        for loader in loaders.values():
            loader.close()


if __name__ == "__main__":
    main()
