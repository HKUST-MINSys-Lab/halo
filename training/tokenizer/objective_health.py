"""Retired future-JEPA configuration audit, retained for reproducibility."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.tokenizer.multispan_kernel import MS_FRAME_RATE_HZ, MultiSpanKernelTokenizer
from training.tokenizer.future_jepa import DEFAULT_HORIZON_BINS_SECONDS, make_future_target_plan
from training.tokenizer.pretrain import _corpus_datasets, validate_source_window_contract
from training.tokenizer.pretrain_data import (
    CorpusIndex, MultiResolutionCollate, MultiScaleCollate, PretrainDataset, SEED,
    TemperatureSampler, _seed_worker, modalities_present,
)

DEFAULT_DURATIONS = (0.5, 1.0, 2.0)


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size), "min": round(float(array.min()), 4),
        "p10": round(float(np.percentile(array, 10)), 4),
        "median": round(float(np.median(array)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "max": round(float(array.max()), 4), "mean": round(float(array.mean()), 4),
    }


def _multispan_metadata(batch: dict, durations: tuple[float, ...], frame_rate: int) -> dict:
    real_duration = (batch["patch_ends"] * batch["patch_padding_mask"]).amax(dim=1)
    frontend = MultiSpanKernelTokenizer(d_model=8, spans=durations, frame_rate_hz=frame_rate)
    meta = frontend.token_metadata(real_duration)
    half = 0.5 * meta["durations"]
    return {
        "patch_starts": meta["positions"] - half,
        "patch_ends": meta["positions"] + half,
        "patch_padding_mask": meta["token_mask"],
        "resolution_ids": meta["resolution_ids"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-retired-jepa", action="store_true",
                        help="acknowledge that this is a historical JEPA-only diagnostic")
    parser.add_argument("--frontend", choices=("fixed", "multispan"), default="fixed")
    parser.add_argument("--corpus", choices=("label_free", "expanded", "matched"),
                        default="label_free")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--durations", nargs="+", type=float, default=DEFAULT_DURATIONS)
    parser.add_argument("--multispan-frame-rate-hz", type=int, default=MS_FRAME_RATE_HZ)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.allow_retired_jepa:
        parser.error(
            "future-JEPA is retired from the active recipe; pass --allow-retired-jepa only "
            "for historical reproducibility"
        )
    if args.batches <= 0 or args.batch <= 0 or args.multispan_frame_rate_hz <= 0:
        parser.error("batches, batch size, and multispan frame rate must be positive")
    durations = tuple(sorted(set(float(value) for value in args.durations)))
    if len(durations) < 2 or any(not np.isfinite(value) or value <= 0 for value in durations):
        parser.error("durations must contain at least two distinct finite positive values")

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    datasets = tuple(args.datasets) if args.datasets else _corpus_datasets(args.corpus)
    from data.scripts.curate.deployment_policy import LABEL_FREE_PRETRAIN_DATASETS
    requested = set(datasets)
    label_free = set(LABEL_FREE_PRETRAIN_DATASETS)
    if requested & label_free and not requested.issubset(label_free):
        parser.error("cannot audit mixed six- and eight-second source-window contracts")
    index = CorpusIndex(seed=SEED, datasets=datasets)
    if requested.issubset(label_free):
        from data.pretraining.corpus_plan import PRETRAIN_WINDOW_SECONDS
        validate_source_window_contract(index, PRETRAIN_WINDOW_SECONDS)
    dataset = PretrainDataset(index, index.train, augment=False, two_view=False)
    groups = [len(modalities_present(index.refs[key.stream_i].mask)) for key in index.train]
    sampler = TemperatureSampler(
        index.train, index.stream_datasets, num_samples=args.batches * args.batch,
        alpha=0.25, seed=0, batch_size=args.batch,
        subject_ids=index.train_subject_ids, subject_alpha=0.5,
        max_dataset_share=0.25, batch_group_ids=groups,
    )
    collate = (MultiResolutionCollate(fixed_patch_seconds=durations, seed=0, two_view=False)
               if args.frontend == "fixed"
               else MultiScaleCollate(fixed_patch_seconds=1.0, two_view=False))
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch, drop_last=True,
                        collate_fn=collate, num_workers=0, worker_init_fn=_seed_worker)

    context_tokens: list[float] = []
    target_tokens: list[float] = []
    context_fractions: list[float] = []
    target_fractions: list[float] = []
    target_resolutions = [0] * len(durations)
    target_horizons = [0] * len(DEFAULT_HORIZON_BINS_SECONDS)
    missing_resolution = missing_horizon = ineligible = no_target = overlap = leakage = 0
    source_counts: dict[str, int] = {}
    source_targets: dict[str, list[int]] = {}
    generator = torch.Generator().manual_seed(SEED)
    for batch in loader:
        meta = (batch if args.frontend == "fixed"
                else _multispan_metadata(batch, durations, args.multispan_frame_rate_hz))
        plan = make_future_target_plan(
            meta["patch_starts"], meta["patch_ends"], meta["patch_padding_mask"],
            meta["resolution_ids"], horizon_bins_seconds=DEFAULT_HORIZON_BINS_SECONDS,
            generator=generator,
        )
        valid = meta["patch_padding_mask"]
        valid_count = valid.sum(1).clamp_min(1)
        context_count = plan.context_mask.sum(1)
        target_count = plan.target_mask.sum(1)
        context_tokens.extend(context_count.float().tolist())
        target_tokens.extend(target_count.float().tolist())
        context_fractions.extend((context_count / valid_count).float().tolist())
        target_fractions.extend((target_count / valid_count).float().tolist())
        ineligible += int((~plan.eligible).sum())
        no_target += int((plan.eligible & target_count.eq(0)).sum())
        overlap += int((plan.context_mask & plan.target_mask).sum())
        boundary = plan.context_end[:, None]
        leakage += int((plan.context_mask & (meta["patch_ends"] > boundary + 1e-7)).sum())
        leakage += int((plan.target_mask & (meta["patch_starts"] < boundary - 1e-7)).sum())
        resolution_counts = []
        for rid in range(len(durations)):
            count = int((plan.target_mask & meta["resolution_ids"].eq(rid)).sum())
            target_resolutions[rid] += count
            resolution_counts.append(count)
        missing_resolution += int(any(count == 0 for count in resolution_counts))
        horizon_counts = []
        for idx, (lo, hi) in enumerate(DEFAULT_HORIZON_BINS_SECONDS):
            count = int((plan.target_mask & plan.horizon_seconds.ge(lo)
                         & plan.horizon_seconds.lt(hi)).sum())
            target_horizons[idx] += count
            horizon_counts.append(count)
        missing_horizon += int(any(count == 0 for count in horizon_counts))
        for source, count in zip(batch["sources"], target_count.tolist()):
            source_counts[source] = source_counts.get(source, 0) + 1
            source_targets.setdefault(source, []).append(int(count))

    windows = args.batches * args.batch
    report = {
        "frontend": args.frontend, "corpus": args.corpus, "datasets": list(datasets),
        "batches": args.batches, "windows": windows,
        "durations_seconds": durations,
        "multispan_frame_rate_hz": args.multispan_frame_rate_hz,
        "future_jepa": {
            "context_tokens_per_window": distribution(context_tokens),
            "target_tokens_per_window": distribution(target_tokens),
            "context_fraction_of_real_tokens": distribution(context_fractions),
            "target_fraction_of_real_tokens": distribution(target_fractions),
            "ineligible_windows": ineligible, "ineligible_fraction": round(ineligible / windows, 4),
            "eligible_windows_without_target": no_target,
            "context_target_overlap_tokens": overlap, "boundary_leak_tokens": leakage,
            "target_count_by_resolution": {
                f"{duration:g}s": target_resolutions[i] for i, duration in enumerate(durations)
            },
            "target_count_by_horizon": {
                f"[{lo:g},{hi:g})s": target_horizons[i]
                for i, (lo, hi) in enumerate(DEFAULT_HORIZON_BINS_SECONDS)
            },
            "batches_missing_a_resolution": missing_resolution,
            "batches_missing_a_horizon": missing_horizon,
            "mean_targets_by_source": {
                source: round(float(np.mean(values)), 3)
                for source, values in sorted(source_targets.items())
            },
        },
        "sampled_source_share": {
            source: round(count / windows, 4) for source, count in sorted(source_counts.items())
        },
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
