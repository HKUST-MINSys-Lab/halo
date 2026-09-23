"""Phase 4 unit tests: the unlabelled-pool curriculum (sampler) and the pooled readouts (trainer)
on synthetic corpora and random tensors. run_step end-to-end is exercised by the training smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from training.support_classifier.sampling import Episode, attach_pool, draw_batch, draw_episode
from training.support_classifier.train import (
    episode_recording_indices, pooled_episode_logits, probability_features_torch, split_pool,
)

# The sampler tests' synthetic corpus builder, loaded by path so this file makes no assumption
# about ``tests`` being a package.
_spec = importlib.util.spec_from_file_location(
    "sampling_fixture", Path(__file__).with_name("test_support_classifier_sampling.py"))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
_corpus = _fixture._corpus


def _episode(corpus, seed=0, **kwargs) -> Episode:
    rng = np.random.default_rng(seed)
    for _ in range(50):
        episode = draw_episode(corpus, rng, support_size=4, p_gt_present=1.0, label_subset=(3, 4), **kwargs)
        if episode is not None and episode.support:
            return episode
    raise AssertionError("no enrolled episode could be drawn")


def _units(corpus, indices):
    return {(corpus.recordings[i].dataset, corpus.recordings[i].subject, corpus.recordings[i].execution)
            for i in indices}


def test_attach_pool_draws_from_the_regime_without_touching_query_or_supports():
    corpus = _corpus(sites=("left_wrist", "right_thigh"), subjects_per_label=5, windows_per_subject=4)
    corpus.ensure_indexes()
    episode = _episode(corpus)
    rng = np.random.default_rng(1)
    pooled = attach_pool(corpus, rng, episode, pool_size=12, pool_regime="compatible",
                         concentration=1.0, distractor_fraction=0.25, coverage=(0.5, 1.0))
    assert 0 < len(pooled.pool) <= 12 and pooled.pool_regime == "compatible"
    assert episode.query not in pooled.pool and not set(pooled.pool) & set(episode.support)
    query_unit = _units(corpus, [episode.query])
    assert not _units(corpus, pooled.pool) & query_unit           # never the query's execution
    assert len(pooled.pool_marginal) == len(episode.candidates)
    assert abs(sum(pooled.pool_marginal) - 1.0) < 1e-6
    assert pooled.pool_distractors <= 3
    # distractor rows carry labels outside the roster; roster rows inside it
    labels = [corpus.recordings[i].label for i in pooled.pool]
    assert sum(l not in episode.candidates for l in labels) == pooled.pool_distractors
    # untouched fields
    assert pooled.support == episode.support and pooled.candidates == episode.candidates
    # deterministic under the same generator state
    again = attach_pool(corpus, np.random.default_rng(1), episode, pool_size=12, pool_regime="compatible",
                        concentration=1.0, distractor_fraction=0.25, coverage=(0.5, 1.0))
    assert again.pool == pooled.pool


def test_attach_pool_regimes_and_edge_cases():
    corpus = _corpus(sites=("left_wrist", "right_thigh"), subjects_per_label=5, windows_per_subject=4)
    corpus.ensure_indexes()
    episode = _episode(corpus)
    query_site = corpus.recordings[episode.query].stream
    cross = attach_pool(corpus, np.random.default_rng(2), episode, pool_size=8, pool_regime="cross_placement")
    if cross.pool:
        assert all(corpus.recordings[i].stream != query_site for i in cross.pool)
    else:
        assert cross.pool_regime == "unavailable"
    assert attach_pool(corpus, np.random.default_rng(0), episode, pool_size=0, pool_regime="compatible") is episode
    with pytest.raises(ValueError):
        attach_pool(corpus, np.random.default_rng(0), episode, pool_size=4, pool_regime="compatible", coverage=(0.0, 1.0))
    with pytest.raises(ValueError):
        attach_pool(corpus, np.random.default_rng(0), episode, pool_size=4, pool_regime="compatible", distractor_fraction=1.0)


def test_draw_batch_attaches_pools_and_reports_them_only_when_asked():
    corpus = _corpus(sites=("left_wrist",), subjects_per_label=5, windows_per_subject=4)
    off, telemetry_off = draw_batch(corpus, np.random.default_rng(0), batch_size=6, support_size=4,
                                    p_gt_present=1.0, label_subset=(3, 4))
    assert all(e.pool == () and e.pool_regime == "none" for e in off)
    assert telemetry_off["sampler/pool_mean_size"] == 0.0 and telemetry_off["sampler/pool_attached_fraction"] == 0.0
    on, telemetry_on = draw_batch(corpus, np.random.default_rng(0), batch_size=6, support_size=4,
                                  p_gt_present=1.0, label_subset=(3, 4), pool_size=10,
                                  pool_regime_mix=(1.0, 0.0, 0.0), pool_distractor_fraction=0.2)
    assert all(e.pool or e.pool_regime == "unavailable" for e in on)
    assert telemetry_on["sampler/pool_mean_size"] > 0
    for key in ("sampler/pool_attached_fraction", "sampler/pool_mean_coverage",
                "sampler/pool_distractor_fraction", "sampler/pool_gt_present_fraction"):
        assert key in telemetry_on
    # pool rows are part of the rows the encoder must see
    with_pool = episode_recording_indices(on)
    assert set(i for e in on for i in e.pool) <= set(with_pool)
    with pytest.raises(ValueError):
        draw_batch(corpus, np.random.default_rng(0), batch_size=2, pool_size=-1)


def _tensors(B=3, K=4, P=5, C=3, D=8, T=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(B, D, generator=g, requires_grad=True)
    support = torch.randn(B, K, D, generator=g, requires_grad=True)
    pool = torch.randn(B, P, D, generator=g, requires_grad=True)
    support_mask = torch.ones(B, K, dtype=torch.bool)
    support_bound = torch.tensor([[0, 1, 2, 0]] * B)
    pool_mask = torch.ones(B, P, dtype=torch.bool)
    pool_mask[1, 3:] = False                                     # one task with a short pool
    candidate_text = torch.randn(B, C, T, generator=g)
    candidate_mask = torch.ones(B, C, dtype=torch.bool)
    p_text = nn.Linear(D, T)
    return dict(query=query, support_feature=support, support_mask=support_mask, support_bound=support_bound,
                pool_feature=pool, pool_mask=pool_mask, candidate_text=candidate_text,
                candidate_mask=candidate_mask, p_text=p_text, temperature=30.0)


def test_split_pool_pads_and_masks():
    class E:  # minimal stand-in with the two fields split_pool reads
        def __init__(self, query, support, pool):
            self.query, self.support, self.pool, self.support_window_groups = query, support, pool, ()
    episodes = [E(0, (1, 2), (3, 4, 5)), E(6, (7,), (8,)), E(9, (), ())]
    pooled = torch.arange(10, dtype=torch.float32)[:, None].repeat(1, 2)
    feature, mask = split_pool(pooled, episodes)
    assert feature.shape == (3, 3, 2) and mask.tolist() == [[True, True, True], [True, False, False], [False, False, False]]
    assert feature[0, :, 0].tolist() == [3.0, 4.0, 5.0] and feature[1, 1:].abs().sum() == 0


@pytest.mark.parametrize("mode", ["transductive", "soft_kmeans"])
def test_pooled_readouts_return_finite_masked_logits_with_gradients(mode):
    t = _tensors()
    logits = pooled_episode_logits(mode=mode, unroll={"n_iter": 2, "n_iter_mm": 5, "early_stop": False}, **t)
    assert logits.shape == (3, 3) and torch.isfinite(logits).all()
    loss = -torch.log_softmax(logits, -1)[:, 0].sum()
    loss.backward()
    for name in ("query", "pool_feature"):
        assert t[name].grad is not None and torch.isfinite(t[name].grad).all() and float(t[name].grad.abs().sum()) > 0
    assert t["p_text"].weight.grad is not None or mode == "soft_kmeans"


def test_transductive_readout_ignores_padded_pool_rows():
    t = _tensors(B=1, P=4)
    t["pool_mask"] = torch.ones(1, 4, dtype=torch.bool)
    full = pooled_episode_logits(mode="transductive", unroll={"n_iter": 3, "n_iter_mm": 10, "early_stop": False}, **t).detach()
    padded = dict(t)
    padded["pool_feature"] = torch.cat([t["pool_feature"].detach(), torch.zeros(1, 3, 8)], dim=1)
    padded["pool_mask"] = torch.tensor([[True] * 4 + [False] * 3])
    out = pooled_episode_logits(mode="transductive", unroll={"n_iter": 3, "n_iter_mm": 10, "early_stop": False}, **padded).detach()
    assert torch.allclose(full, out, atol=1e-4)


def test_probability_features_torch_masks_invalid_candidates():
    t = _tensors()
    t["candidate_mask"][:, 2] = False
    z = probability_features_torch(t["pool_feature"], t["p_text"], t["candidate_text"], t["candidate_mask"], 30.0)
    assert torch.all(z[:, :, 2] == 0) and torch.allclose(z.sum(-1), torch.ones(3, 5), atol=1e-5)
