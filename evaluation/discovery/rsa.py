"""Representational Similarity Analysis: does the arrangement of activity centroids in the
encoder's space mirror the arrangement of their names in language?

Spearman correlation between two condensed distance matrices (Kriegeskorte 2008; Dwivedi & Roig,
CVPR 2019). No projection between the spaces is fitted, so the encoders without a text head
(HARNet, LiMU-BERT) are scored exactly like the others. Two variants are reported:

* ``rsa_class``   — centroids of the *true-label* groups: the geometry claim, free of clustering error.
* ``rsa_cluster`` — centroids of the *discovered* clusters, mapped to labels by the oracle Hungarian
                    assignment: the end-to-end number.

Registered caveat (Bertram et al. 2026, arXiv:2605.05907): alignment can be carried by a few
dimensions, so a top-PC ablation and a shuffled-name null are computed alongside.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from evaluation.discovery.cluster import centroids_of, l2_normalise


def rdm(vectors: np.ndarray, *, metric: str = "cosine") -> np.ndarray:
    """Condensed pairwise-distance vector of the rows of ``vectors``."""
    return pdist(np.asarray(vectors, dtype=np.float64), metric=metric)


def rsa(centroids: np.ndarray, text_vectors: np.ndarray) -> float:
    """Spearman ρ between the centroid RDM and the text RDM; NaN below three rows (no pairs to rank)."""
    if len(centroids) != len(text_vectors):
        raise ValueError("centroids and text vectors must be aligned row for row")
    if len(centroids) < 3:
        return float("nan")
    rho = spearmanr(rdm(centroids), rdm(text_vectors)).correlation
    return float(rho) if rho is not None else float("nan")


def class_centroids(features: np.ndarray, truth: np.ndarray, classes: list[str]) -> np.ndarray:
    """(C, D) mean feature per true label, in ``classes`` order; absent classes are zero rows."""
    x = np.asarray(features, dtype=np.float32)
    truth = np.asarray(truth, dtype=object)
    index = {label: i for i, label in enumerate(classes)}
    ids = np.asarray([index.get(t, -1) for t in truth], dtype=np.int64)
    out = np.zeros((len(classes), x.shape[1]), dtype=np.float32)
    for c in range(len(classes)):
        members = x[ids == c]
        if len(members):
            out[c] = members.mean(axis=0)
    return out


def cluster_centroids_by_label(features: np.ndarray, cluster_labels: np.ndarray,
                               cluster_to_label: dict[int, int], n_classes: int) -> np.ndarray:
    """(C, D) centroid of the cluster the oracle assignment mapped to each label (zero if none)."""
    x = np.asarray(features, dtype=np.float32)
    k = int(cluster_labels.max()) + 1 if len(cluster_labels) else 0
    per_cluster = centroids_of(x, cluster_labels, k) if k else np.zeros((0, x.shape[1]), np.float32)
    out = np.zeros((n_classes, x.shape[1]), dtype=np.float32)
    for cluster, label in cluster_to_label.items():
        if 0 <= label < n_classes and cluster < k:
            out[label] = per_cluster[cluster]
    return out


def shuffled_null(centroids: np.ndarray, text_vectors: np.ndarray, *, draws: int = 200,
                  seed: int = 0) -> dict:
    """RSA under random permutations of the name rows: what ρ looks like with no correspondence."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        values.append(rsa(centroids, text_vectors[rng.permutation(len(text_vectors))]))
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values)),
            "p95": float(np.nanpercentile(values, 95)), "draws": int(draws)}


def dimension_ablation(features: np.ndarray, truth: np.ndarray, classes: list[str],
                       text_vectors: np.ndarray, *, drop_top: tuple[int, ...] = (1, 4, 16),
                       seed: int = 0) -> dict[str, float]:
    """``rsa_class`` after projecting out the top-m principal components, for each m in ``drop_top``.

    If ρ collapses when a handful of components are removed, the alignment was carried by them.
    """
    x = l2_normalise(features)
    max_m = min(max(drop_top), x.shape[1] - 1, max(1, x.shape[0] - 1))
    pca = PCA(n_components=max_m, random_state=seed).fit(x)
    out = {}
    for m in drop_top:
        m_eff = min(int(m), max_m)
        basis = pca.components_[:m_eff]                      # (m, D), orthonormal rows
        centred = x - pca.mean_
        residual = centred - (centred @ basis.T) @ basis + pca.mean_   # top-m components projected out
        out[f"drop_top_{m}"] = rsa(class_centroids(residual, truth, classes), text_vectors)
    return out
