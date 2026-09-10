"""CPU/data health report for the default multi-horizon future-JEPA objective.

The diagnostic draws real temperature-sampled windows, creates the exact aligned 0.5/1.0/1.5
second grids used by training, and audits context/target eligibility, horizon coverage, resolution
alignment, and future leakage. It intentionally does not instantiate the neural model.

Run: /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.objective_health
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.tokenizer.future_jepa import (
    DEFAULT_HORIZON_BINS_SECONDS,
    make_future_target_plan,
)
from training.tokenizer.pretrain_data import (
    CorpusIndex,
    MultiResolutionCollate,
    PretrainDataset,
    SEED,
    TemperatureSampler,
    _seed_worker,
    modalities_present,
)

OUT = Path(__file__).resolve().parent / "outputs" / "objective_health"
N_BATCHES = 20
BATCH_SIZE = 256
PATCH_DURATIONS = (0.5, 1.0, 1.5)


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "min": round(float(array.min()), 4),
        "p10": round(float(np.percentile(array, 10)), 4),
        "median": round(float(np.median(array)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "max": round(float(array.max()), 4),
        "mean": round(float(array.mean()), 4),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    index = CorpusIndex(seed=SEED)
    dataset = PretrainDataset(index, index.train, augment=False, two_view=False)
    sensor_batch_groups = [
        len(modalities_present(index.refs[key.stream_i].mask)) for key in index.train
    ]
    sampler = TemperatureSampler(
        index.train,
        index.stream_datasets,
        num_samples=N_BATCHES * BATCH_SIZE,
        alpha=0.25,
        seed=0,
        batch_size=BATCH_SIZE,
        subject_ids=index.train_subject_ids,
        subject_alpha=0.5,
        max_dataset_share=0.25,
        batch_group_ids=sensor_batch_groups,
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=BATCH_SIZE,
        drop_last=True,
        collate_fn=MultiResolutionCollate(
            fixed_patch_seconds=PATCH_DURATIONS, seed=0, two_view=False,
        ),
        num_workers=0,
        worker_init_fn=_seed_worker,
    )

    context_tokens: list[float] = []
    target_tokens: list[float] = []
    context_fractions: list[float] = []
    target_fractions: list[float] = []
    target_resolutions = [0 for _ in PATCH_DURATIONS]
    target_horizons = [0 for _ in DEFAULT_HORIZON_BINS_SECONDS]
    batches_missing_resolution = 0
    batches_missing_horizon = 0
    ineligible_windows = 0
    eligible_without_target = 0
    overlap_tokens = 0
    boundary_leaks = 0
    source_counts: dict[str, int] = {}
    source_targets: dict[str, list[int]] = {}
    source_ineligible: dict[str, int] = {}

    generator = torch.Generator().manual_seed(SEED)
    for batch in loader:
        plan = make_future_target_plan(
            batch["patch_starts"],
            batch["patch_ends"],
            batch["patch_padding_mask"],
            batch["resolution_ids"],
            horizon_bins_seconds=DEFAULT_HORIZON_BINS_SECONDS,
            generator=generator,
        )
        valid = batch["patch_padding_mask"]
        valid_counts = valid.sum(dim=1).clamp_min(1)
        context_counts = plan.context_mask.sum(dim=1)
        target_counts = plan.target_mask.sum(dim=1)
        context_tokens.extend(context_counts.float().tolist())
        target_tokens.extend(target_counts.float().tolist())
        context_fractions.extend((context_counts / valid_counts).float().tolist())
        target_fractions.extend((target_counts / valid_counts).float().tolist())
        ineligible_windows += int((~plan.eligible).sum())
        eligible_without_target += int((plan.eligible & target_counts.eq(0)).sum())
        overlap_tokens += int((plan.context_mask & plan.target_mask).sum())

        boundary = plan.context_end[:, None]
        boundary_leaks += int((plan.context_mask
                               & (batch["patch_ends"] > boundary + 1e-7)).sum())
        boundary_leaks += int((plan.target_mask
                               & (batch["patch_starts"] < boundary - 1e-7)).sum())

        batch_resolution_counts = []
        for rid in range(len(PATCH_DURATIONS)):
            count = int((plan.target_mask & batch["resolution_ids"].eq(rid)).sum())
            target_resolutions[rid] += count
            batch_resolution_counts.append(count)
        batches_missing_resolution += int(any(count == 0 for count in batch_resolution_counts))

        batch_horizon_counts = []
        for lo, hi in DEFAULT_HORIZON_BINS_SECONDS:
            selected = plan.target_mask & plan.horizon_seconds.ge(lo) & plan.horizon_seconds.lt(hi)
            count = int(selected.sum())
            target_horizons[len(batch_horizon_counts)] += count
            batch_horizon_counts.append(count)
        batches_missing_horizon += int(any(count == 0 for count in batch_horizon_counts))

        for source, count, eligible in zip(
            batch["sources"], target_counts.tolist(), plan.eligible.tolist(),
        ):
            source_counts[source] = source_counts.get(source, 0) + 1
            source_targets.setdefault(source, []).append(int(count))
            source_ineligible[source] = source_ineligible.get(source, 0) + int(not eligible)

    windows = N_BATCHES * BATCH_SIZE
    report = {
        "batches": N_BATCHES,
        "windows": windows,
        "patch_durations_seconds": PATCH_DURATIONS,
        "future_jepa": {
            "context_tokens_per_window": distribution(context_tokens),
            "target_tokens_per_window": distribution(target_tokens),
            "context_fraction_of_real_tokens": distribution(context_fractions),
            "target_fraction_of_real_tokens": distribution(target_fractions),
            "ineligible_windows": ineligible_windows,
            "ineligible_fraction": round(ineligible_windows / windows, 4),
            "eligible_windows_without_target": eligible_without_target,
            "context_target_overlap_tokens": overlap_tokens,
            "boundary_leak_tokens": boundary_leaks,
            "target_count_by_resolution": {
                f"{duration:g}s": target_resolutions[rid]
                for rid, duration in enumerate(PATCH_DURATIONS)
            },
            "target_count_by_horizon": {
                f"[{lo:g},{hi:g})s": target_horizons[i]
                for i, (lo, hi) in enumerate(DEFAULT_HORIZON_BINS_SECONDS)
            },
            "batches_missing_a_resolution": batches_missing_resolution,
            "batches_missing_a_horizon": batches_missing_horizon,
            "mean_targets_by_source": {
                source: round(float(np.mean(values)), 3)
                for source, values in sorted(source_targets.items())
            },
            "ineligible_fraction_by_source": {
                source: round(source_ineligible[source] / count, 4)
                for source, count in sorted(source_counts.items())
            },
        },
        "sampled_source_share": {
            source: round(count / windows, 4)
            for source, count in sorted(source_counts.items())
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
