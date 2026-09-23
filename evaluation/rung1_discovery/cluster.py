"""Clustering of frozen features: k-means++ (10 restarts) with Ward as a same-result robustness
check, K estimation, and the label-free UMAP-silhouette proxy. Nothing here sees a label."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

ALGORITHMS = ("kmeans_pp_10", "ward")
DEFAULT_K_RANGE = tuple(range(2, 31))


def l2_normalise(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), np.float32(1e-12))


def prepare_features(features: np.ndarray, *, pca_dim: int | None = None, seed: int = 0) -> np.ndarray:
    """L2-normalise; optionally PCA to ``pca_dim`` and re-normalise (encoder widths span 72–4608)."""
    x = l2_normalise(features)
    if pca_dim is not None:
        if pca_dim < 1:
            raise ValueError("pca_dim must be positive")
        dim = min(int(pca_dim), x.shape[1], max(1, x.shape[0] - 1))
        x = PCA(n_components=dim, random_state=seed).fit_transform(x).astype(np.float32)
        x = l2_normalise(x)
    return x


@dataclass(frozen=True)
class Clustering:
    labels: np.ndarray        # (N,) cluster ids in [0, k)
    centroids: np.ndarray     # (k, D) means in the clustered space
    algorithm: str
    k: int
    inertia: float | None


def centroids_of(features: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    out = np.zeros((k, x.shape[1]), dtype=np.float32)
    for cluster in range(k):
        members = x[labels == cluster]
        if len(members):
            out[cluster] = members.mean(axis=0)
    return out


def cluster_cell(features: np.ndarray, *, k: int, algorithm: str = "kmeans_pp_10",
                 restarts: int = 10, seed: int = 0) -> Clustering:
    x = np.asarray(features, dtype=np.float32)
    if k < 2 or k > len(x):
        raise ValueError(f"k={k} must lie in [2, n_samples={len(x)}]")
    if algorithm == "kmeans_pp_10":
        model = KMeans(n_clusters=k, init="k-means++", n_init=restarts, random_state=seed).fit(x)
        return Clustering(model.labels_.astype(np.int64), model.cluster_centers_.astype(np.float32),
                          algorithm, k, float(model.inertia_))
    if algorithm == "ward":
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x).astype(np.int64)
        return Clustering(labels, centroids_of(x, labels, k), algorithm, k, None)
    raise ValueError(f"unknown clustering algorithm {algorithm!r}; choose from {ALGORITHMS}")


def _safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(x):
        return float("nan")
    return float(silhouette_score(x, labels))


def _safe_davies_bouldin(x: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(davies_bouldin_score(x, labels))


def estimate_k(features: np.ndarray, *, k_range: tuple[int, ...] = DEFAULT_K_RANGE,
               restarts: int = 3, seed: int = 0) -> dict:
    """K̂ by silhouette (argmax) and Davies–Bouldin (argmin) over ``k_range``; |K̂ − K| is a metric."""
    x = np.asarray(features, dtype=np.float32)
    curve = []
    for k in k_range:
        if k >= len(x):
            break
        labels = KMeans(n_clusters=k, init="k-means++", n_init=restarts, random_state=seed).fit_predict(x)
        curve.append({"k": int(k), "silhouette": _safe_silhouette(x, labels),
                      "davies_bouldin": _safe_davies_bouldin(x, labels)})
    if not curve:
        raise ValueError("no feasible k in k_range for this many samples")
    finite = [row for row in curve if np.isfinite(row["silhouette"]) and np.isfinite(row["davies_bouldin"])]
    if not finite:
        raise ValueError("silhouette/Davies-Bouldin undefined for every k")
    return {
        "k_hat_silhouette": int(max(finite, key=lambda r: r["silhouette"])["k"]),
        "k_hat_davies_bouldin": int(min(finite, key=lambda r: r["davies_bouldin"])["k"]),
        "curve": curve,
    }


def umap_silhouette(features: np.ndarray, cluster_labels: np.ndarray, *, n_components: int = 5,
                    seed: int = 0) -> float:
    """Lowe et al.'s label-free proxy: silhouette of the *cluster* labels in a UMAP-reduced space.

    Uses cluster ids only, so a deployment with no ground truth can compute it. UMAP is imported
    lazily; the dependency is present in the environment but optional for the package.
    """
    import umap  # noqa: WPS433 - optional dependency

    x = np.asarray(features, dtype=np.float32)
    dim = min(int(n_components), max(1, x.shape[1] - 1))
    embedded = umap.UMAP(n_components=dim, random_state=seed).fit_transform(x)
    return _safe_silhouette(np.asarray(embedded, dtype=np.float32), cluster_labels)
