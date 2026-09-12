"""Opt-in, representation-only diagnostics for support-classification experiments.

The functions here never alter embeddings, fit a classifier, or choose a checkpoint.  They turn a
labelled feature matrix into compact numerical summaries and figures that make a frozen-versus-
adapted encoder comparison inspectable.  ``sealed_eval`` calls this only with an explicit flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_MAX_POINTS = 2_000
DEFAULT_PAIR_SAMPLES = 20_000


def _normalise(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("features must be a non-empty finite (N,D) matrix")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _balanced_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = sorted(set(labels.tolist()))
    per_class = max(1, maximum // max(1, len(classes)))
    chosen = []
    for label in classes:
        rows = np.flatnonzero(labels == label)
        if len(rows) > per_class:
            rows = rng.choice(rows, size=per_class, replace=False)
        chosen.extend(rows.tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def _effective_rank(features: np.ndarray) -> float:
    singular = np.linalg.svd(features - features.mean(axis=0, keepdims=True), compute_uv=False)
    probability = singular / max(float(singular.sum()), 1e-12)
    probability = np.maximum(probability, 1e-12)
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _sample_similarity(features: np.ndarray, labels: np.ndarray, *, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Balanced same-label and different-label cosine samples without an O(N²) matrix."""
    rng = np.random.default_rng(seed)
    by_label = {label: np.flatnonzero(labels == label) for label in sorted(set(labels.tolist()))}
    positive_labels = [label for label, rows in by_label.items() if len(rows) >= 2]
    if not positive_labels or len(by_label) < 2:
        return np.empty(0), np.empty(0)
    positive, negative = [], []
    for _ in range(count):
        label = positive_labels[int(rng.integers(len(positive_labels)))]
        pair = rng.choice(by_label[label], size=2, replace=False)
        positive.append(float(features[pair[0]] @ features[pair[1]]))
        other = [item for item in by_label if item != label]
        other_label = other[int(rng.integers(len(other)))]
        negative.append(float(features[pair[0]] @ features[rng.choice(by_label[other_label])]))
    return np.asarray(positive), np.asarray(negative)


def embedding_summary(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    pair_samples: int = DEFAULT_PAIR_SAMPLES,
    seed: int = 20260912,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Return metrics and bounded arrays used by the diagnostic plots.

    The nearest-neighbour result is leave-one-out on a balanced bounded subset.  It is a geometry
    diagnostic only, not a replacement for the execution-disjoint enrolled evaluation protocol.
    """
    values = _normalise(features)
    y = np.asarray(labels, dtype=object)
    if len(y) != len(values):
        raise ValueError("labels must align one-to-one with features")
    keep = _balanced_indices(y, max_points, seed)
    z, y_small = values[keep], y[keep]
    positive, negative = _sample_similarity(z, y_small, count=pair_samples, seed=seed)
    similarity = z @ z.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(axis=1)
    loo_purity = float(np.mean(y_small[nearest] == y_small)) if len(z) > 1 else float("nan")
    labels_unique = sorted(set(y_small.tolist()))
    centroids = np.stack([_normalise(z[y_small == label].mean(axis=0, keepdims=True))[0]
                          for label in labels_unique])
    centroid_similarity = centroids @ centroids.T
    own = np.asarray([z[index] @ centroids[labels_unique.index(label)] for index, label in enumerate(y_small)])
    other = (z @ centroids.T).copy()
    other[np.arange(len(z)), [labels_unique.index(label) for label in y_small]] = -np.inf
    margin = own - other.max(axis=1)
    summary = {
        "n_embeddings": int(len(values)),
        "dimension": int(values.shape[1]),
        "n_labels": int(len(labels_unique)),
        "sampled_embeddings": int(len(z)),
        "effective_rank": _effective_rank(values),
        "leave_one_out_neighbor_purity": loo_purity,
        "same_label_cosine_mean": float(positive.mean()) if len(positive) else float("nan"),
        "different_label_cosine_mean": float(negative.mean()) if len(negative) else float("nan"),
        "centroid_margin_mean": float(margin.mean()),
        "centroid_margin_p10": float(np.quantile(margin, 0.10)),
    }
    artifacts = {"features": z, "labels": y_small, "positive": positive, "negative": negative,
                 "centroid_similarity": centroid_similarity,
                 "centroid_labels": np.asarray(labels_unique, dtype=object)}
    return summary, artifacts


def write_embedding_diagnostics(
    features: np.ndarray,
    labels: Sequence[str],
    out: Path,
    *,
    title: str,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = 20260912,
) -> dict:
    """Write JSON plus PCA, similarity, and centroid figures for one representation matrix."""
    summary, artifacts = embedding_summary(features, labels, max_points=max_points, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment contract, not numerical logic
        raise RuntimeError("embedding plots require matplotlib") from error

    z = artifacts["features"]
    centered = z - z.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ right[:2].T
    names = artifacts["labels"]
    visible = sorted(set(names.tolist()))
    fig, axis = plt.subplots(figsize=(8, 6))
    for label in visible:
        rows = names == label
        axis.scatter(projection[rows, 0], projection[rows, 1], s=10, alpha=0.65, label=str(label))
    axis.set(title=title, xlabel="PCA 1", ylabel="PCA 2")
    axis.legend(fontsize=6, ncol=min(3, max(1, len(visible) // 8 + 1)), loc="best")
    fig.tight_layout(); fig.savefig(out / "pca_by_label.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(artifacts["positive"], bins=40, alpha=0.65, density=True, label="same label")
    axis.hist(artifacts["negative"], bins=40, alpha=0.65, density=True, label="different label")
    axis.set(title=title, xlabel="cosine similarity", ylabel="density")
    axis.legend(); fig.tight_layout(); fig.savefig(out / "cosine_distributions.png", dpi=180); plt.close(fig)

    matrix = artifacts["centroid_similarity"]
    fig, axis = plt.subplots(figsize=(max(6, len(visible) * 0.35), max(5, len(visible) * 0.3)))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set(title=title, xticks=range(len(visible)), yticks=range(len(visible)),
             xticklabels=visible, yticklabels=visible)
    axis.tick_params(axis="x", labelrotation=90, labelsize=6); axis.tick_params(axis="y", labelsize=6)
    fig.colorbar(image, ax=axis, label="centroid cosine similarity")
    fig.tight_layout(); fig.savefig(out / "label_centroid_similarity.png", dpi=180); plt.close(fig)
    return summary
