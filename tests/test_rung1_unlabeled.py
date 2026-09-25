"""Rung 1 unit tests: the EM-Dirichlet port, the pool protocol, and the controls — all synthetic."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from evaluation.controls import balanced_pool_filter, disjoint_class_split
from evaluation.rung1_unlabeled.ncurve import (
    CellSplit, inductive_predictions, nested_pool_draws, run_cell, shared_support_set, split_scored_pool,
)
from evaluation.rung1_unlabeled.transductive import (
    assign_clusters, paper_lambda, transduce, transduce_numpy,
)
from evaluation.zero_shot import probability_features


def _dirichlet_mixture(seed=0, n_per=60, concentration=40.0):
    """Three components peaked on distinct roster slots; rows are already probability features."""
    rng = np.random.default_rng(seed)
    alphas = np.full((3, 3), 1.0)
    np.fill_diagonal(alphas, concentration)
    z = np.concatenate([rng.dirichlet(alphas[k], size=n_per) for k in range(3)]).astype(np.float32)
    y = np.repeat(np.arange(3), n_per)
    return z, y


def test_paper_lambda_follows_the_released_integer_division_rule():
    assert float(paper_lambda(7, 100)) == 100.0          # int(7/5)=1
    assert float(paper_lambda(4, 100)) == 0.0            # int(4/5)=0: partition term vanishes
    assert float(paper_lambda(12, 100)) == 200.0
    assert float(paper_lambda(6, 50, k_eff=6)) == 50.0   # few-shot: int(C/k_eff)*N


@pytest.mark.parametrize("n_classes", [3, 4, 7, 12])
def test_transduce_defaults_to_lambda_equal_to_n_for_every_roster_size(n_classes):
    # k_eff defaults to the roster size, so the partition term neither vanishes below five classes
    # nor doubles at ten; padded roster slots do not count towards the roster.
    torch.manual_seed(0)
    z = torch.softmax(torch.randn(2, 9, n_classes + 2), dim=-1)
    mask = torch.zeros(2, n_classes + 2, dtype=torch.bool)
    mask[:, :n_classes] = True
    rows = torch.ones(2, 9, dtype=torch.bool)
    rows[1, 6:] = False
    result = transduce(z, candidate_mask=mask, row_mask=rows, n_iter=1, n_iter_mm=2, early_stop=False)
    assert result.lam.tolist() == [9.0, 6.0]


def test_transduce_ignores_an_enclosing_bf16_autocast():
    torch.manual_seed(0)
    z = torch.softmax(3 * torch.randn(1, 40, 5), dim=-1)
    reference = transduce(z, n_iter=3, n_iter_mm=10, early_stop=False)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_result = transduce(z, n_iter=3, n_iter_mm=10, early_stop=False)
    assert autocast_result.logits.dtype == torch.float32
    assert torch.equal(autocast_result.logits, reference.logits)


def test_zero_shot_recovers_a_dirichlet_mixture_and_matchings_agree():
    z, y = _dirichlet_mixture()
    preds, u, info = transduce_numpy(z, n_iter=10, n_iter_mm=200, assignment="identity")
    assert (preds == y).mean() > 0.95
    assert np.allclose(u.sum(axis=1), 1.0, atol=1e-5)
    graph, _, _ = transduce_numpy(z, n_iter=10, n_iter_mm=200, assignment="graph")
    basic, _, _ = transduce_numpy(z, n_iter=10, n_iter_mm=200, assignment="basic")
    assert (graph == preds).mean() > 0.95 and (basic == preds).mean() > 0.95
    assert info["few_shot"] is False and len(info["cluster_sizes"]) == 3


def test_few_shot_supports_pin_component_identity_on_an_ambiguous_pool():
    rng = np.random.default_rng(1)
    # Two components; the pool's probability features lean the WRONG way (a biased zero-shot
    # bridge), supports say otherwise. Few-shot must follow the supports.
    z = np.stack([rng.dirichlet([3.0, 2.0]) for _ in range(40)] + [rng.dirichlet([2.0, 3.0]) for _ in range(40)]).astype(np.float32)
    y = np.repeat([1, 0], 40)                       # true class is the opposite of the leaning
    support_z = np.array([[0.05, 0.95]] * 6 + [[0.95, 0.05]] * 6, dtype=np.float32)
    support_labels = np.array([0] * 6 + [1] * 6)   # class 0 supports look like slot 1, and so on
    zero_shot, _, _ = transduce_numpy(z, n_iter=10, n_iter_mm=200)
    few_shot, _, info = transduce_numpy(z, support_z=support_z, support_labels=support_labels,
                                        n_iter=10, n_iter_mm=200)
    assert (few_shot == y).mean() > (zero_shot == y).mean()
    assert info["few_shot"] is True


def test_candidate_mask_excludes_padded_slots_exactly():
    z, y = _dirichlet_mixture(n_per=30)
    padded = np.concatenate([z, np.zeros((len(z), 2), dtype=np.float32)], axis=1)   # 5 slots, 3 valid
    mask = torch.tensor([[True, True, True, False, False]])
    zt = torch.as_tensor(padded)[None]
    with torch.no_grad():
        out = transduce(zt, candidate_mask=mask, n_iter=5, n_iter_mm=50)
        ref = transduce(torch.as_tensor(z)[None], n_iter=5, n_iter_mm=50)
    assert torch.all(out.u[0, :, 3:] == 0)
    assert torch.allclose(out.u[0, :, :3], ref.u[0], atol=1e-4)
    assert (out.u[0].argmax(-1).numpy() == y).mean() > 0.95


def test_the_unrolled_procedure_is_differentiable_end_to_end():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 3, requires_grad=True)
    z = torch.softmax(logits, dim=-1)
    result = transduce(z, n_iter=2, n_iter_mm=5, early_stop=False)
    target = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1], [2, 2, 1, 1, 0, 0, 2, 1]])
    loss = -torch.log_softmax(result.logits, dim=-1).gather(-1, target[..., None]).mean()
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0
    # few-shot path too
    logits2 = torch.randn(1, 6, 3, requires_grad=True)
    sz = torch.softmax(torch.randn(1, 3, 3), dim=-1)
    so = F.one_hot(torch.tensor([[0, 1, 2]]), 3).float()
    r2 = transduce(torch.softmax(logits2, -1), support_z=sz, support_onehot=so, n_iter=2, n_iter_mm=5, early_stop=False)
    (-r2.logits.logsumexp(-1).mean()).backward()
    assert torch.isfinite(logits2.grad).all()


def test_assign_clusters_identity_and_basic_map_to_valid_slots():
    z, _ = _dirichlet_mixture(n_per=10)
    zt = torch.as_tensor(z)[None]
    with torch.no_grad():
        result = transduce(zt, n_iter=3, n_iter_mm=20)
    for mode in ("identity", "basic", "graph"):
        preds = assign_clusters(result.u, zt, mode=mode)
        assert preds.shape == (1, 30) and preds.min() >= 0 and preds.max() < 3


def test_probability_features_are_rows_on_the_simplex_for_every_kind():
    scores = np.array([[0.9, 0.1, -0.2], [0.2, 0.8, 0.0]])
    for kind in ("cosine", "distance"):
        p, info = probability_features(scores, kind)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6) and (p > 0).all()
        assert info["kind"] == kind and info["temperature"] == 30.0
    assert "distance_scale" in probability_features(-np.abs(scores), "distance")[1]
    p, _ = probability_features(np.array([[0.2, 0.3, 0.5]]), "probability")
    assert np.allclose(p, [[0.2, 0.3, 0.5]])


def _fake_stream(n_exec=10, per_exec=6, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    execution_ids = np.repeat([f"e{i}" for i in range(n_exec)], per_exec).astype(object)
    truth_ids = rng.integers(0, n_classes, size=n_exec * per_exec)
    return execution_ids, truth_ids


def test_split_is_deterministic_execution_disjoint_and_draws_are_nested():
    execution_ids, truth_ids = _fake_stream()
    valid = np.arange(len(truth_ids))
    a = split_scored_pool(execution_ids, valid, fraction=0.2, seed_parts=("cell",))
    b = split_scored_pool(execution_ids, valid, fraction=0.2, seed_parts=("cell",))
    assert np.array_equal(a.scored, b.scored) and np.array_equal(a.pool, b.pool)
    assert not set(execution_ids[a.scored]) & set(execution_ids[a.pool])
    assert 0 < len(a.scored) < len(valid) and len(a.scored) + len(a.pool) == len(valid)
    draws = nested_pool_draws(a.pool, (5, 10, "all"), seed_parts=("cell",))
    assert set(draws["5"]) <= set(draws["10"]) <= set(draws["all"]) == set(a.pool.tolist())
    assert len(draws["5"]) == 5 and len(draws["all"]) == len(a.pool)


def test_shared_support_set_is_per_class_and_reports_infeasibility():
    execution_ids, truth_ids = _fake_stream(n_exec=12)
    split = split_scored_pool(execution_ids, np.arange(len(truth_ids)), fraction=0.2, seed_parts=("s",))
    rows, labels = shared_support_set(split.pool, truth_ids, 2, 3, seed_parts=("s",))
    assert len(rows) == 6 and np.array_equal(np.bincount(labels), [2, 2, 2])
    assert set(rows) <= set(split.pool.tolist())
    assert shared_support_set(split.pool, truth_ids, 10_000, 3) is None
    assert shared_support_set(split.pool, truth_ids, 0, 3)[0].size == 0


def test_inductive_k0_is_the_zero_shot_argmax_and_k1_is_the_prototype_rule():
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    rows = np.array([0, 1, 2])
    assert np.array_equal(inductive_predictions(scores=scores, features=None, support_rows=np.zeros(0, int),
                                                support_labels=np.zeros(0, int), n_classes=2, rows=rows), [1, 0, 1])
    features = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    preds = inductive_predictions(scores=scores[:2], features=features, support_rows=np.array([0, 1]),
                                  support_labels=np.array([0, 1]), n_classes=2, rows=np.array([2, 3]))
    assert np.array_equal(preds, [0, 1])


def test_run_cell_emits_the_grid_and_the_reproduction_row_with_the_sealed_metric_set():
    z, y = _dirichlet_mixture(seed=3, n_per=40)
    scores = np.log(z + 1e-9) / 30.0              # so that softmax(30 * scores) ≈ z
    execution_ids = np.repeat([f"e{i}" for i in range(12)], 10).astype(object)
    split = split_scored_pool(execution_ids, np.arange(len(y)), fraction=0.2, seed_parts=("c",))
    draws = nested_pool_draws(split.pool, (10, "all"), seed_parts=("c",))
    rows = run_cell(scores_all=scores, score_kind="cosine", features_all=z, truth_ids=y,
                    classes=["a", "b", "c"], split=split, pool_draws=draws, ks=(0, 1),
                    transduce_kwargs={"n_iter": 3, "n_iter_mm": 20, "early_stop": False},
                    seed_parts=("c",))
    repro = [r for r in rows if r["scope"] == "all_windows"]
    assert len(repro) == 1 and repro[0]["method"] == "inductive" and repro[0]["k"] == 0
    assert {"f1_macro", "balanced_accuracy", "accuracy"} <= set(repro[0])
    grid = [r for r in rows if r["scope"] == "scored" and r.get("status", "ok") == "ok"]
    assert {(r["method"], r["k"], r.get("N_label", "0")) for r in grid} >= {
        ("inductive", 0, "0"), ("transductive_clip_v1", 0, "10"), ("transductive_clip_v1", 0, "all"),
        ("inductive", 1, "0"), ("transductive_clip_v1", 1, "all")}
    for r in grid:
        if r["method"] == "transductive_clip_v1":
            assert r["n_transduced"] == r["n_scored"] + r["N"]
            assert "pool_marginal_entropy" in r and "collapsed_components" in r
    assert {r["assignment"] for r in grid if r["method"] != "inductive" and r["k"] == 0} == {"identity", "graph"}
    assert all(r["assignment"] == "identity" for r in grid if r["method"] != "inductive" and r["k"] == 1)


def test_distance_calibration_uses_only_scored_rows_and_pool_metrics_exclude_them():
    scores = np.array([[0.0, -1.0], [-1.0, 0.0], [-100.0, 0.0], [0.0, -100.0]])
    split = CellSplit(scored=np.array([0, 1]), pool=np.array([2, 3]),
                      n_executions_scored=2, n_executions_pool=2)
    kwargs = dict(score_kind="distance", features_all=None, truth_ids=np.array([0, 1, 0, 1]),
                  classes=["a", "b"], split=split, pool_draws={"0": np.array([], dtype=int)},
                  ks=(0,), assignments=("identity",),
                  transduce_kwargs={"n_iter": 2, "n_iter_mm": 5, "early_stop": False})
    first = run_cell(scores_all=scores, **kwargs)
    changed = scores.copy()
    changed[2:] *= 100
    second = run_cell(scores_all=changed, **kwargs)
    transduced = lambda rows: next(row for row in rows if row["method"] == "transductive_clip_v1")
    assert transduced(first)["distance_scale"] == transduced(second)["distance_scale"]
    assert transduced(first)["pool_class_counts"] == [0, 0]
    assert transduced(first)["pool_classes_present"] == 0
    assert transduced(first)["f1_macro"] == transduced(second)["f1_macro"]


def test_balanced_pool_filter_yields_a_uniform_marginal_from_labels():
    truth_ids = np.array([0] * 30 + [1] * 6 + [2] * 12)
    pool = np.arange(len(truth_ids))
    balanced = balanced_pool_filter(truth_ids, 3, seed_parts=("b",))(pool)
    counts = np.bincount(truth_ids[balanced], minlength=3)
    assert counts.max() - counts.min() <= 0 or counts[1] == 6      # limited by the rarest class
    assert len(set(balanced)) == len(balanced) and set(balanced) <= set(pool)


def test_balanced_pool_control_keeps_n_and_excludes_support_rows():
    truth_ids = np.array([0] * 20 + [1] * 12 + [2] * 10)
    requested = np.arange(8)
    available = np.arange(4, len(truth_ids))
    balanced = balanced_pool_filter(truth_ids, 3, seed_parts=("size",))(
        requested, available_rows=available,
    )
    counts = np.bincount(truth_ids[balanced], minlength=3)
    assert len(balanced) == len(requested)
    assert set(balanced) <= set(available)
    assert counts.max() - counts.min() <= 1


def test_disjoint_class_split_separates_scored_and_pool_classes():
    truth_ids = np.array([0, 1, 2, 3] * 20)
    split = CellSplit(scored=np.arange(0, 40), pool=np.arange(40, 80), n_executions_scored=1, n_executions_pool=1)
    new, kept, held = disjoint_class_split(split, truth_ids, 4, holdout_fraction=0.5, seed_parts=("d",))
    assert len(kept) == 2 and len(held) == 2 and not set(kept) & set(held)
    assert set(truth_ids[new.scored]) <= set(kept) and set(truth_ids[new.pool]) <= set(held)


def test_row_mask_excludes_padded_rows_exactly():
    z, y = _dirichlet_mixture(n_per=20)
    zt = torch.as_tensor(z)[None]
    padded = torch.cat([zt, torch.full((1, 7, 3), 1.0 / 3)], dim=1)
    row_mask = torch.tensor([[True] * len(z) + [False] * 7])
    with torch.no_grad():
        ref = transduce(zt, n_iter=4, n_iter_mm=30)
        out = transduce(padded, row_mask=row_mask, n_iter=4, n_iter_mm=30)
    assert torch.all(out.u[0, len(z):] == 0)
    assert torch.allclose(out.u[0, :len(z)], ref.u[0], atol=1e-4)
    assert torch.allclose(out.alpha, ref.alpha, atol=1e-3) and float(out.lam[0]) == float(ref.lam[0])


# ---------------------------------------------------------------- embedding affinity (2026-09-24)
from evaluation.rung1_unlabeled.ncurve import neighbour_purity  # noqa: E402
from evaluation.rung1_unlabeled.transductive import knn_affinity  # noqa: E402


def _noisy_text(seed=0, n_per=150, n_classes=6, dims=32, geometry="clean"):
    """A noisy text head (softmax at T = 30 of weak class evidence plus per-window noise) over
    embeddings that are either clean class clusters or grouped by an unrelated 'subject'."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n_per)
    s = np.eye(n_classes)[y] * 0.25 + 0.105 * rng.standard_normal((len(y), n_classes))
    z = np.exp(30 * s)
    z /= z.sum(1, keepdims=True)
    if geometry == "clean":
        emb = rng.standard_normal((n_classes, dims))[y] + 0.6 * rng.standard_normal((len(y), dims))
    else:
        subject = rng.integers(0, n_classes, len(y))
        emb = rng.standard_normal((n_classes, dims))[subject] + 0.3 * rng.standard_normal((len(y), dims))
    return z.astype(np.float32), emb.astype(np.float32), y


def test_mu_zero_or_no_embeddings_is_the_published_method_exactly():
    z, emb, _ = _noisy_text()
    zt, et = torch.from_numpy(z)[None], torch.from_numpy(emb)[None]
    plain = transduce(zt, n_iter=4, n_iter_mm=20, early_stop=False)
    with_zero = transduce(zt, embeddings=et, mu=0.0, n_iter=4, n_iter_mm=20, early_stop=False)
    assert torch.equal(plain.logits, with_zero.logits) and with_zero.neighbours is None


def test_affinity_lets_clean_geometry_correct_a_noisy_text_head():
    z, emb, y = _noisy_text(geometry="clean")
    kwargs = {"n_iter": 20, "n_iter_mm": 200, "early_stop": False}
    plain, _, _ = transduce_numpy(z, **kwargs)
    affine, _, info = transduce_numpy(z, embeddings=emb, mu=1.0, knn=10, **kwargs)
    assert (affine == y).mean() > (plain == y).mean() + 0.03
    assert info["mu"] == 1.0 and info["knn"] == 10 and info["neighbours"].shape == (len(y), 10)
    assert neighbour_purity(y, info["neighbours"]) > 0.8


def test_affinity_does_not_collapse_when_neighbours_are_uninformative():
    # Neighbours grouped by subject, not class: the vote must be roughly neutral, never the
    # one-class collapse that voting on the current assignments produced.
    z, emb, y = _noisy_text(geometry="subject")
    kwargs = {"n_iter": 20, "n_iter_mm": 200, "early_stop": False}
    plain, _, _ = transduce_numpy(z, **kwargs)
    affine, _, info = transduce_numpy(z, embeddings=emb, mu=1.0, knn=10, **kwargs)
    assert neighbour_purity(y, info["neighbours"]) < 0.3
    assert (affine == y).mean() > (plain == y).mean() - 0.03
    assert len(np.unique(affine)) == len(np.unique(y))


def test_knn_affinity_skips_self_and_padded_rows_and_carries_gradient():
    emb = torch.randn(1, 6, 4, requires_grad=True)
    rows = torch.tensor([[True, True, True, True, False, False]])
    idx, weight = knn_affinity(emb, rows, knn=10)
    assert idx.shape == (1, 6, 5)
    for i in range(4):
        chosen = idx[0, i][weight[0, i] > 0].tolist()
        assert i not in chosen and not ({4, 5} & set(chosen))
    assert float(weight[0, 4:].abs().sum()) == 0.0              # padded rows have no neighbours
    weight.sum().backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()


def test_neighbour_purity_ignores_out_of_roster_rows():
    truth = np.array([0, 0, 1, -1])
    neighbours = np.array([[1, 3], [0, 2], [0, 1], [0, 1]])
    # counted pairs: (0,1)=same, (1,0)=same, (1,2)=diff, (2,0)=diff, (2,1)=diff -> 2/5
    assert neighbour_purity(truth, neighbours) == pytest.approx(0.4)
