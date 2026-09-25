"""Simulated causal deployments from the supervised, subject-split training corpus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.support_classifier.sampling import SupportCorpus


@dataclass(frozen=True)
class MemoryEpisode:
    roster: tuple[str, ...]
    rows: tuple[int, ...]
    verified: tuple[bool, ...]
    dataset: str
    requested_regime: str = "matched"
    realized_mismatch_fraction: float = 0.0
    realized_cross_dataset_fraction: float = 0.0


def eligible_memory_datasets(corpus: SupportCorpus) -> tuple[str, ...]:
    corpus.ensure_indexes()
    cache_key = ("memory_reader_eligible_datasets_v1",)
    if cache_key in corpus.deployment_dataset_cache:
        return corpus.deployment_dataset_cache[cache_key]
    units_by_dataset: dict[str, set[tuple[str, str]]] = {}
    for record in corpus.recordings:
        units_by_dataset.setdefault(record.dataset, set()).add((record.subject, record.execution))
    eligible = []
    for dataset, labels in corpus.query_labels_by_dataset.items():
        if len(labels) < 2:
            continue
        if len(units_by_dataset.get(dataset, ())) >= 2:
            eligible.append(dataset)
    result = tuple(sorted(eligible))
    corpus.deployment_dataset_cache[cache_key] = result
    return result


def draw_memory_episode(
    corpus: SupportCorpus, rng: np.random.Generator, *, max_history: int = 12,
    max_candidates: int = 8, cross_dataset_probability: float = 0.25,
    dataset: str | None = None,
) -> MemoryEpisode:
    """One recording per execution, with variable class balance and enrollment availability.

    Labels of unverified rows are used only by the training objective. The model receives no
    dataset/source identifier or hidden class marginal. Arrival order is a simulation, not
    physical longitudinal order.
    """
    if max_history < 1 or max_candidates < 2 or not 0 <= cross_dataset_probability <= 0.5:
        raise ValueError("need positive history and at least two candidate slots")
    corpus.ensure_indexes()
    eligible = [(name, corpus.query_labels_by_dataset[name])
                for name in eligible_memory_datasets(corpus)]
    if not eligible:
        raise ValueError("no dataset has two usable labels")
    if dataset is None:
        dataset, available = eligible[int(rng.integers(len(eligible)))]
    else:
        available = next((labels for name, labels in eligible if name == dataset), None)
        if available is None:
            raise ValueError(f"dataset {dataset!r} has fewer than two usable labels")
    c = int(rng.integers(2, min(max_candidates, len(available)) + 1))
    roster = tuple(sorted(str(label) for label in rng.choice(available, size=c, replace=False)))
    total = int(rng.integers(2, max_history + 2))
    # Dirichlet concentration varies by episode; a low value makes minority/missing classes
    # without scripting artificial label mistakes.
    concentration = float(rng.choice([0.3, 1.0, 4.0]))
    prior = rng.dirichlet(np.full(c, concentration))
    query_label = roster[int(rng.choice(c, p=prior))]
    query_choices = corpus.query_by_dataset_label[(dataset, query_label)]
    query_row = int(query_choices[int(rng.integers(len(query_choices)))])
    query = corpus.recordings[query_row]
    query_key = corpus.key_of(query)
    regimes = ("matched", "cross_placement", "cross_dataset")
    regime = str(rng.choice(regimes, p=[1 - 2 * cross_dataset_probability,
                                       cross_dataset_probability, cross_dataset_probability]))
    same_subject = bool(rng.integers(2))
    other_labels = tuple(label for label in available if label not in roster)
    rows: list[int] = []
    used: set[tuple[str, str, str]] = {(query.dataset, query.subject, query.execution)}
    for _ in range(total * 16):
        if len(rows) >= total - 1:
            break
        distractor = bool(other_labels and rng.random() < 0.15)
        label = str(rng.choice(other_labels)) if distractor else roster[int(rng.choice(c, p=prior))]
        source = dataset
        if regime == "cross_dataset" and not distractor:
            source_options = [name for name, other_label in corpus.query_by_dataset_label
                              if name != dataset and other_label == label]
            if source_options:
                source = str(rng.choice(source_options))
        choices = corpus.query_by_dataset_label.get((source, label), ())
        if not choices:
            continue
        row = int(choices[int(rng.integers(len(choices)))])
        record = corpus.recordings[row]
        if regime == "matched" and corpus.key_of(record) != query_key:
            continue
        if regime == "cross_placement" and corpus.key_of(record) == query_key:
            continue
        if same_subject and source == dataset and record.subject != query.subject:
            continue
        unit = (record.dataset, record.subject, record.execution)
        if unit in used:
            continue
        used.add(unit)
        rows.append(row)
    if not rows:
        # Some source/key/subject combinations are genuinely infeasible. Fall back to one other
        # execution in this deployment and record the realized mismatch below.
        for _ in range(32):
            label = str(rng.choice(available))
            choices = corpus.query_by_dataset_label.get((dataset, label), ())
            if not choices:
                continue
            row = int(choices[int(rng.integers(len(choices)))])
            record = corpus.recordings[row]
            if (record.dataset, record.subject, record.execution) not in used:
                rows.append(row)
                break
    if not rows:
        fallback = [row for (source, _), candidates in corpus.query_by_dataset_label.items()
                    if source == dataset for row in candidates]
        rng.shuffle(fallback)
        for row in fallback:
            record = corpus.recordings[row]
            unit = (record.dataset, record.subject, record.execution)
            if unit not in used:
                rows.append(row)
                break
    if not rows:
        raise ValueError("not enough execution-distinct rows for an online memory episode")
    rows.append(query_row)
    enrollment_probability = float(rng.choice([0.0, 0.25, 0.75, 1.0]))
    verified = tuple(bool(i < len(rows) - 1 and corpus.recordings[row].label in roster
                          and rng.random() < enrollment_probability)
                     for i, row in enumerate(rows))
    mismatch = sum(corpus.key_of(corpus.recordings[row]) != query_key for row in rows[:-1])
    cross_dataset = sum(corpus.recordings[row].dataset != dataset for row in rows[:-1])
    return MemoryEpisode(roster, tuple(rows), verified, dataset, regime,
                         mismatch / (len(rows) - 1), cross_dataset / (len(rows) - 1))
