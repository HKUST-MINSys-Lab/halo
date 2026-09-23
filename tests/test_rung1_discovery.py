"""Rung 1 unit tests: transparent synthetic geometry with known answers. No cached data needed."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from evaluation.metrics import ami, ari, hungarian_accuracy
from evaluation.rung1_discovery.cluster import cluster_cell, estimate_k, prepare_features
from evaluation.rung1_discovery.name import naming_outcomes
from evaluation.rung1_discovery.rsa import class_centroids, rsa, shuffled_null
from evaluation.rung1_discovery.run import evaluate_cell


def _blobs(seed=0, k=3, n=60, dim=8):
    x, y = make_blobs(n_samples=n, centers=k, n_features=dim, cluster_std=0.05, random_state=seed)
    return x.astype(np.float32), y


@pytest.mark.parametrize("algorithm", ["kmeans_pp_10", "ward"])
def test_separable_blobs_are_recovered_exactly(algorithm):
    x, y = _blobs()
    clustering = cluster_cell(prepare_features(x), k=3, algorithm=algorithm, seed=0)
    assert ami(y, clustering.labels) > 0.99
    assert ari(y, clustering.labels) > 0.99
    assert hungarian_accuracy(y, clustering.labels) == 1.0
    assert clustering.centroids.shape == (3, 8)


def test_k_estimate_recovers_the_true_number_of_blobs():
    x, _ = _blobs(k=4, n=120)
    estimate = estimate_k(prepare_features(x), k_range=tuple(range(2, 8)), seed=0)
    assert estimate["k_hat_silhouette"] == 4
    assert estimate["k_hat_davies_bouldin"] == 4


def test_prepare_features_normalises_and_reduces():
    x, _ = _blobs(dim=16)
    raw = prepare_features(x)
    assert np.allclose(np.linalg.norm(raw, axis=1), 1.0, atol=1e-5)
    reduced = prepare_features(x, pca_dim=4)
    assert reduced.shape == (len(x), 4)
    assert np.allclose(np.linalg.norm(reduced, axis=1), 1.0, atol=1e-5)


def test_rsa_is_one_under_an_isometry_and_near_zero_under_shuffling():
    rng = np.random.default_rng(0)
    text = rng.normal(size=(6, 5)).astype(np.float32)
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    centroids = (text @ rotation).astype(np.float32)          # same pairwise cosine geometry
    assert rsa(centroids, text) == pytest.approx(1.0)
    null = shuffled_null(centroids, text, draws=300, seed=1)
    assert abs(null["mean"]) < 0.25 and null["p95"] < 1.0


def test_rsa_is_undefined_below_three_classes():
    assert np.isnan(rsa(np.zeros((2, 4)), np.zeros((2, 4))))


def test_class_centroids_follow_class_order_and_zero_absent_classes():
    x = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    truth = np.array(["a", "a", "b"], dtype=object)
    cent = class_centroids(x, truth, ["b", "a", "c"])
    assert np.allclose(cent, [[0, 1], [1, 0], [0, 0]])


def test_naming_oracle_bounds_model_which_bounds_greedy():
    labels = np.array([0, 0, 1, 1, 2, 2])
    truth = np.array([0, 0, 1, 1, 2, 2])
    perfect = np.eye(3, dtype=np.float32)
    out = naming_outcomes(labels, truth, 3, 3, perfect)
    assert out["naming_oracle_hungarian"] == out["naming_model_hungarian"] == out["naming_greedy"] == 1.0
    assert out["naming_cost"] == 0.0
    # Two clusters both prefer name 0: greedy collides, Hungarian resolves it, oracle is unaffected.
    collide = np.array([[0.9, 0.1, 0.0], [0.8, 0.7, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    out = naming_outcomes(labels, truth, 3, 3, collide)
    assert out["naming_oracle_hungarian"] == 1.0
    assert out["naming_model_hungarian"] == 1.0
    assert out["naming_greedy"] < out["naming_model_hungarian"]
    assert out["naming_collisions_greedy"] == 1


def test_evaluate_cell_produces_every_row_kind_with_the_expected_values():
    x, y = _blobs(k=3, n=90, dim=8)
    classes = ["walk", "run", "sit"]
    truth = np.asarray([classes[i] for i in y], dtype=object)
    rng = np.random.default_rng(0)
    text = rng.normal(size=(3, 5)).astype(np.float32)
    # A score function that names a centroid by which blob it is nearest to: perfect naming.
    blob_centres = np.stack([x[y == i].mean(axis=0) for i in range(3)])

    def score_fn(centroids):
        d = ((centroids[:, None, :] - blob_centres[None]) ** 2).sum(-1)
        return -d

    rows = evaluate_cell(features_raw=x, naming_features=x, truth=truth, classes=classes,
                         text_vectors=text, score_fn=score_fn, seed=0, pca_dims=(None, 4),
                         k_range=range(2, 6), umap=False, null_draws=20)
    kinds = {(r["method"], r["k_setting"], r["pca_dim"]) for r in rows}
    assert {("kmeans_pp_10", "k_true", 0), ("ward", "k_true", 0), ("kmeans_pp_10", "k_true", 4),
            ("kmeans_pp_10", "k_hat_silhouette", 0)} <= kinds
    for row in rows:
        if row["k_setting"] == "k_true":
            assert row["ami"] > 0.99 and row["hungarian_accuracy_pct"] == 100.0
            assert row["naming_model_hungarian_pct"] == 100.0 and row["naming_cost_pct"] == 0.0
            assert row["k_hat_silhouette"] == 3 and row["k_abs_error_silhouette"] == 0
        assert np.isfinite(row["rsa_class"]) or row["k_true"] < 3
        assert "rsa_class_null_mean" in row and "rsa_cluster" in row
    raw_rows = [r for r in rows if r["pca_dim"] == 0 and r["k_setting"] == "k_true"]
    assert all("rsa_class_drop_top_1" in r for r in raw_rows)


def test_evaluate_cell_skips_naming_without_a_score_function():
    x, y = _blobs()
    classes = ["a", "b", "c"]
    truth = np.asarray([classes[i] for i in y], dtype=object)
    rows = evaluate_cell(features_raw=x, naming_features=None, truth=truth, classes=classes,
                         text_vectors=np.eye(3, 4, dtype=np.float32), score_fn=None, seed=0,
                         pca_dims=(None,), estimate=False, umap=False, null_draws=5)
    assert rows and all("naming_model_hungarian_pct" not in r for r in rows)
