"""Metrics shared by rungs 1 and 3 (and the retired discovery readout) — one definition each.

Classification metrics delegate to :func:`baselines.scoring.classification_metrics`, so every
rung's accuracy and macro-F1 are the sealed table's definitions (macro-F1 over ground-truth ∪
predicted classes, balanced accuracy over ground-truth classes, all in percent). Clustering
metrics are thin, guarded wrappers over scikit-learn and reported on their native [0, 1] scale;
the one accuracy among them (Hungarian) is also a fraction here and is scaled by the caller.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics as skm

from baselines import scoring


def classification(truth_names, pred_names, *, f1_classes=None, recall_classes=None) -> dict:
    """The sealed table's metric set: f1_macro, balanced_accuracy, accuracy, f1_weighted, counts."""
    return scoring.classification_metrics(
        list(truth_names), list(pred_names), f1_classes=f1_classes, recall_classes=recall_classes,
    )


def _pair(truth, pred) -> tuple[np.ndarray, np.ndarray]:
    truth, pred = np.asarray(truth), np.asarray(pred)
    if truth.shape != pred.shape:
        raise ValueError("truth and prediction must have the same shape")
    return truth, pred


def ami(truth, pred) -> float:
    t, p = _pair(truth, pred)
    return float(skm.adjusted_mutual_info_score(t, p)) if len(t) else float("nan")


def ari(truth, pred) -> float:
    t, p = _pair(truth, pred)
    return float(skm.adjusted_rand_score(t, p)) if len(t) else float("nan")


def nmi(truth, pred) -> float:
    """Reported for comparability with prior HAR clustering work only; biased w.r.t. cluster count."""
    t, p = _pair(truth, pred)
    return float(skm.normalized_mutual_info_score(t, p)) if len(t) else float("nan")


def hungarian_accuracy(truth_ids, cluster_labels) -> float:
    """Fraction of windows correct after the best one-to-one cluster→class matching (an oracle)."""
    t, c = _pair(np.asarray(truth_ids, dtype=np.int64), np.asarray(cluster_labels, dtype=np.int64))
    if len(t) == 0:
        return float("nan")
    if (t < 0).any() or (c < 0).any():
        raise ValueError("ids must be non-negative")
    matrix = np.zeros((int(c.max()) + 1, int(t.max()) + 1), dtype=np.int64)
    np.add.at(matrix, (c, t), 1)
    rows, cols = linear_sum_assignment(-matrix)
    return float(matrix[rows, cols].sum() / len(t))


def silhouette(features, labels) -> float:
    x, labels = np.asarray(features, dtype=np.float32), np.asarray(labels)
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or n_clusters >= len(x):
        return float("nan")
    return float(skm.silhouette_score(x, labels))


def davies_bouldin(features, labels) -> float:
    x, labels = np.asarray(features, dtype=np.float32), np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(skm.davies_bouldin_score(x, labels))
