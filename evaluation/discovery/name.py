"""Naming discovered clusters: the encoder's own cluster × roster score matrix, and three ways of
turning it into an assignment. The oracle − model gap is the naming cost.

* ``oracle_hungarian`` — clusters matched to labels by the ground-truth confusion (the number the
  clustering literature reports; uses labels, so it is an upper bound, not a capability).
* ``model_hungarian``  — one-to-one matching on the encoder's cluster × name scores; no labels.
* ``greedy``           — each cluster takes its arg-max name; collisions allowed; no labels.

Scores come from :func:`evaluation.zero_shot.zero_shot_scores` applied to cluster centroids, so
HALO is named through its closed-form SBERT bridge, UniMTS/NormWear natively, and HARNet/LiMU-BERT
through the ConSE bank bridge (disclosed in every row).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.discovery.cluster import centroids_of


def confusion(cluster_labels: np.ndarray, truth_ids: np.ndarray, k: int, n_classes: int) -> np.ndarray:
    matrix = np.zeros((k, n_classes), dtype=np.int64)
    valid = truth_ids >= 0
    np.add.at(matrix, (cluster_labels[valid], truth_ids[valid]), 1)
    return matrix


def oracle_assignment(cluster_labels: np.ndarray, truth_ids: np.ndarray, k: int,
                      n_classes: int) -> dict[int, int]:
    """Hungarian matching on the confusion matrix (maximise agreement)."""
    rows, cols = linear_sum_assignment(-confusion(cluster_labels, truth_ids, k, n_classes))
    return {int(r): int(c) for r, c in zip(rows, cols)}


def model_assignment(scores: np.ndarray) -> dict[int, int]:
    """Hungarian matching on the encoder's (k, C) cluster × name scores, no labels."""
    rows, cols = linear_sum_assignment(-np.asarray(scores, dtype=np.float64))
    return {int(r): int(c) for r, c in zip(rows, cols)}


def greedy_assignment(scores: np.ndarray) -> dict[int, int]:
    return {int(r): int(c) for r, c in enumerate(np.asarray(scores).argmax(axis=1))}


def assignment_accuracy(cluster_labels: np.ndarray, truth_ids: np.ndarray,
                        assignment: dict[int, int]) -> float:
    valid = truth_ids >= 0
    if not valid.any():
        return float("nan")
    predicted = np.asarray([assignment.get(int(c), -1) for c in cluster_labels], dtype=np.int64)
    return float((predicted[valid] == truth_ids[valid]).mean())


def name_clusters(features_raw: np.ndarray, cluster_labels: np.ndarray, k: int,
                  score_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """(k, C) scores of each cluster centroid (in the model's *raw* feature space) against the roster."""
    return np.asarray(score_fn(centroids_of(features_raw, cluster_labels, k)), dtype=np.float32)


def naming_outcomes(cluster_labels: np.ndarray, truth_ids: np.ndarray, k: int, n_classes: int,
                    scores: np.ndarray) -> dict:
    oracle = oracle_assignment(cluster_labels, truth_ids, k, n_classes)
    model = model_assignment(scores)
    greedy = greedy_assignment(scores)
    acc = {
        "naming_oracle_hungarian": assignment_accuracy(cluster_labels, truth_ids, oracle),
        "naming_model_hungarian": assignment_accuracy(cluster_labels, truth_ids, model),
        "naming_greedy": assignment_accuracy(cluster_labels, truth_ids, greedy),
    }
    acc["naming_cost"] = acc["naming_oracle_hungarian"] - acc["naming_model_hungarian"]
    acc["naming_collisions_greedy"] = int(k - len(set(greedy.values())))
    return {**acc, "oracle_assignment": oracle, "model_assignment": model}
