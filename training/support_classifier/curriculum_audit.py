"""Audit the realised deployment curriculum without loading sensor tensors.

This is deliberately sampler-only: it builds the same active training corpus and calls the same
``draw_batch`` path as the trainer, then records what was actually feasible. It is therefore cheap
enough to run before every long classifier experiment and cannot consume sealed evaluation data.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from data.scripts.curate.deployment_policy import SUPERVISED_HEAD_TRAIN_DATASETS
from training.support_classifier.corpus import support_corpus_from_index
from training.support_classifier.sampling import (
    DEFAULT_ACQUISITION_MIX,
    DEFAULT_ENROLLMENT_K,
    DEFAULT_ENROLLMENT_MIX,
    DEFAULT_LABEL_SUBSET,
    DEFAULT_PARTIAL_COVERAGE,
    DEFAULT_P_GT_PRESENT,
    DEFAULT_QUERIES_PER_SUPPORT_SET,
    DEFAULT_SAME_SUBJECT_PROBABILITY,
    DEFAULT_VARIABLE_SUPPORT_PROBABILITY,
    DEFAULT_WINDOWS_PER_EXECUTION,
    draw_batch,
)
from training.tokenizer.pretrain_data import CorpusIndex


def _shares(counter: collections.Counter, total: int) -> dict[str, float]:
    return {str(key): value / max(total, 1) for key, value in sorted(counter.items())}


def run_audit(*, support_sets: int, batch_size: int, seed: int, window_seconds: float) -> dict:
    index = CorpusIndex(
        datasets=SUPERVISED_HEAD_TRAIN_DATASETS,
        alignment="native",
        max_per_stream=None,
        seed=seed,
        window_seconds=window_seconds,
    )
    corpus = support_corpus_from_index(index)
    rng = np.random.default_rng(seed)
    draw_kwargs = {
        "p_gt_present": DEFAULT_P_GT_PRESENT,
        "same_subject_probability": DEFAULT_SAME_SUBJECT_PROBABILITY,
        "label_subset": DEFAULT_LABEL_SUBSET,
        "mode": "compatible",
        "semantic_zero_shot": True,
        "deployment_matched": True,
        "enrollment_k": DEFAULT_ENROLLMENT_K,
        "acquisition_mix": DEFAULT_ACQUISITION_MIX,
        "enrollment_mix": DEFAULT_ENROLLMENT_MIX,
        "partial_coverage": DEFAULT_PARTIAL_COVERAGE,
        "variable_support_probability": DEFAULT_VARIABLE_SUPPORT_PROBABILITY,
        "require_query_support": False,
        "queries_per_support_set": DEFAULT_QUERIES_PER_SUPPORT_SET,
        "windows_per_execution": DEFAULT_WINDOWS_PER_EXECUTION,
        "p_mask_candidate": 0.25,
        "p_mask_gt": 0.10,
    }

    acquisition = collections.Counter()
    enrollment = collections.Counter()
    datasets = collections.Counter()
    per_dataset_acquisition: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    per_dataset_enrollment: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    per_dataset_fallback = collections.Counter()
    labels_by_acquisition: dict[str, set[str]] = collections.defaultdict(set)
    datasets_by_acquisition: dict[str, set[str]] = collections.defaultdict(set)
    key_families_by_acquisition: dict[str, set[str]] = collections.defaultdict(set)
    candidate_counts = collections.Counter()
    support_k = collections.Counter()
    fallback = truth_enrolled = cross_placement_cross_dataset = 0
    query_count = 0
    sampled_sets = 0
    telemetry_rows: list[tuple[dict[str, float], int]] = []

    while sampled_sets < support_sets:
        current = min(batch_size, support_sets - sampled_sets)
        episodes, telemetry = draw_batch(corpus, rng, batch_size=current, **draw_kwargs)
        telemetry_rows.append((telemetry, current))
        groups: dict[int, list] = collections.defaultdict(list)
        for episode in episodes:
            groups[episode.support_set_id].append(episode)
        for group in groups.values():
            representative = group[0]
            query = corpus.recordings[representative.query]
            key = corpus.key_of(query)
            acquisition[representative.acquisition_regime] += 1
            enrollment[representative.enrollment_regime] += 1
            datasets[query.dataset] += 1
            per_dataset_acquisition[query.dataset][representative.acquisition_regime] += 1
            per_dataset_enrollment[query.dataset][representative.enrollment_regime] += 1
            per_dataset_fallback[query.dataset] += int(representative.curriculum_fallback)
            labels_by_acquisition[representative.acquisition_regime].add(query.label)
            datasets_by_acquisition[representative.acquisition_regime].add(query.dataset)
            key_families_by_acquisition[representative.acquisition_regime].add(
                f"{key.device_family}:{key.site}:{'+'.join(key.channels)}:{key.gravity_state}"
            )
            candidate_counts[len(representative.candidates)] += 1
            if not representative.is_zero_shot:
                support_k[representative.support_per_candidate] += 1
            fallback += int(representative.curriculum_fallback)
            if representative.acquisition_regime == "cross_placement":
                support_datasets = {
                    corpus.recordings[index].dataset for index in representative.support
                }
                cross_placement_cross_dataset += int(
                    bool(support_datasets - {query.dataset})
                )
            for episode in group:
                query_count += 1
                truth_enrolled += int(
                    episode.gt_slot < len(episode.support_counts)
                    and episode.support_counts[episode.gt_slot] > 0
                )
        sampled_sets += len(groups)

    telemetry_keys = sorted({key for row, _ in telemetry_rows for key in row})
    mean_telemetry = {
        key: float(np.average(
            [row[key] for row, _ in telemetry_rows if key in row],
            weights=[weight for row, weight in telemetry_rows if key in row],
        ))
        for key in telemetry_keys
    }
    enrolled_sets = sampled_sets - enrollment["zero"]
    return {
        "protocol": "support-curriculum-audit-v2-20260918",
        "seed": seed,
        "window_seconds": window_seconds,
        "requested_support_sets": support_sets,
        "realised_support_sets": sampled_sets,
        "realised_queries": query_count,
        "training_datasets": list(SUPERVISED_HEAD_TRAIN_DATASETS),
        "corpus": index.summary(),
        "acquisition_share": _shares(acquisition, sampled_sets),
        "acquisition_share_enrolled": {
            name: acquisition[name] / max(enrolled_sets, 1)
            for name in ("compatible", "cross_placement", "cross_dataset")
        },
        "enrollment_share": _shares(enrollment, sampled_sets),
        "query_dataset_share": _shares(datasets, sampled_sets),
        "per_dataset": {
            dataset: {
                "n_support_sets": int(count),
                "fallback_share": per_dataset_fallback[dataset] / max(count, 1),
                "acquisition_share": _shares(per_dataset_acquisition[dataset], count),
                "enrollment_share": _shares(per_dataset_enrollment[dataset], count),
            }
            for dataset, count in sorted(datasets.items())
        },
        "candidate_count_share": _shares(candidate_counts, sampled_sets),
        "support_k_share_enrolled": _shares(support_k, sum(support_k.values())),
        "curriculum_fallback_share": fallback / sampled_sets,
        "truth_enrolled_query_share": truth_enrolled / max(query_count, 1),
        "cross_placement_cross_dataset_share": (
            cross_placement_cross_dataset / max(acquisition["cross_placement"], 1)
        ),
        "coverage": {
            name: {
                "dataset_count": len(datasets_by_acquisition[name]),
                "datasets": sorted(datasets_by_acquisition[name]),
                "query_label_count": len(labels_by_acquisition[name]),
                "query_labels": sorted(labels_by_acquisition[name]),
                "query_acquisition_key_count": len(key_families_by_acquisition[name]),
            }
            for name in ("compatible", "cross_placement", "cross_dataset")
        },
        "sampler_telemetry": mean_telemetry,
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Support Curriculum Audit",
        "",
        f"Protocol: `{report['protocol']}`. Seed: `{report['seed']}`. "
        f"Window: `{report['window_seconds']}` s.",
        "",
        f"Sampled **{report['realised_support_sets']:,} support sets** and "
        f"**{report['realised_queries']:,} queries** from the active training roster.",
        "",
        "| condition | realised share | datasets | labels | acquisition keys |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, share in report["acquisition_share_enrolled"].items():
        coverage = report["coverage"].get(name, {})
        lines.append(
            f"| acquisition/{name} (enrolled only) | {share:.3f} | "
            f"{coverage.get('dataset_count', 0)} | "
            f"{coverage.get('query_label_count', 0)} | "
            f"{coverage.get('query_acquisition_key_count', 0)} |"
        )
    for name, share in report["enrollment_share"].items():
        lines.append(f"| enrollment/{name} | {share:.3f} | - | - | - |")
    lines.extend([
        "",
        f"- Curriculum fallback share: **{report['curriculum_fallback_share']:.3f}**",
        f"- Truth-enrolled query share: **{report['truth_enrolled_query_share']:.3f}**",
        "- Cross-placement sets containing another dataset: "
        f"**{report['cross_placement_cross_dataset_share']:.3f}**",
        "",
        "## Dataset Coverage",
        "",
    ])
    for name in ("compatible", "cross_placement", "cross_dataset"):
        coverage = report["coverage"][name]
        lines.append(f"- `{name}`: {', '.join(coverage['datasets']) or 'none'}")
    lines.extend([
        "", "## Per-Dataset Delivered Curriculum", "",
        "| dataset | sets | fallback | compatible | cross placement | cross dataset | zero |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, row in report["per_dataset"].items():
        acquisition = row["acquisition_share"]
        enrollment = row["enrollment_share"]
        lines.append(
            f"| {dataset} | {row['n_support_sets']} | {row['fallback_share']:.3f} | "
            f"{acquisition.get('compatible', 0.0):.3f} | "
            f"{acquisition.get('cross_placement', 0.0):.3f} | "
            f"{acquisition.get('cross_dataset', 0.0):.3f} | "
            f"{enrollment.get('zero', 0.0):.3f} |"
        )
    lines.extend(["", "## Candidate And Support Distributions", ""])
    lines.append("- Candidate count: " + ", ".join(
        f"C={key}: {value:.3f}" for key, value in report["candidate_count_share"].items()
    ))
    lines.append("- Enrolled K: " + ", ".join(
        f"K={key}: {value:.3f}" for key, value in report["support_k_share_enrolled"].items()
    ))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-sets", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument(
        "--out", type=Path,
        default=Path("training/support_classifier/evaluations/curriculum_audit_20260917"),
    )
    args = parser.parse_args()
    if args.support_sets < 1 or args.batch_size < 1:
        parser.error("support-sets and batch-size must be positive")
    report = run_audit(
        support_sets=args.support_sets,
        batch_size=args.batch_size,
        seed=args.seed,
        window_seconds=args.window_seconds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.out.with_suffix(".md").write_text(_markdown(report))
    print(json.dumps({
        "json": str(args.out.with_suffix('.json')),
        "markdown": str(args.out.with_suffix('.md')),
        "acquisition_share": report["acquisition_share"],
        "acquisition_share_enrolled": report["acquisition_share_enrolled"],
        "enrollment_share": report["enrollment_share"],
        "curriculum_fallback_share": report["curriculum_fallback_share"],
    }, indent=2))


if __name__ == "__main__":
    main()
