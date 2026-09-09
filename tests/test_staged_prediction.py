"""Contracts for the staged encoder experiment and separately trained attention heads."""

import dataclasses

import pytest
import torch
from torch.nn import functional as F

from model.blocks import AttentionSpec
from model.evidence.comparator import ComparatorConfig, SupportComparator, comparator_logits
from model.evidence.prediction import DualRecordingAttention, RecordingAttentionHead, neighbor_logits


def fixture():
    torch.manual_seed(17)
    return dict(
        query=torch.randn(2, 16), candidates=torch.randn(2, 3, 12),
        candidate_mask=torch.ones(2, 3, dtype=torch.bool),
        support=torch.randn(2, 6, 16), support_labels=torch.randn(2, 6, 12),
        support_mask=torch.ones(2, 6, dtype=torch.bool),
        instance_ids=torch.tensor([[5, 2, 6, 4, 1, 3], [1, 4, 2, 6, 5, 3]]),
    )


def head():
    return RecordingAttentionHead(AttentionSpec(16, 4, 2, 0), text_dim=12, n_layers=2).eval()


def test_pair_permutation_and_candidate_permutation():
    data = fixture()
    model = head()
    scores = model(**data)
    permutation = torch.tensor([4, 2, 5, 0, 3, 1])
    other = dict(data)
    for key in ("support", "support_labels", "support_mask", "instance_ids"):
        other[key] = data[key][:, permutation]
    torch.testing.assert_close(model(**other), scores, atol=2e-6, rtol=2e-6)
    other = {**data, "candidates": data["candidates"].flip(1)}
    torch.testing.assert_close(model(**other), scores.flip(1), atol=2e-6, rtol=2e-6)


def test_pairing_matters_and_duplicate_instance_ids_fail():
    data = fixture()
    model = head()
    other = {**data, "support_labels": data["support_labels"].flip(1)}
    assert not torch.allclose(model(**data), model(**other))
    with pytest.raises(ValueError, match="distinct"):
        model(**{**data, "instance_ids": torch.ones_like(data["instance_ids"])})


def test_padding_does_not_affect_valid_predictions_or_gradients():
    data = fixture()
    model = head()
    expected = model(**data)
    for key in ("support", "support_labels"):
        data[key] = torch.cat([data[key], torch.randn(2, 3, data[key].shape[-1]) * 100], 1)
        data[key].requires_grad_()
    data["support_mask"] = F.pad(data["support_mask"], (0, 3), value=False)
    data["instance_ids"] = F.pad(data["instance_ids"], (0, 3), value=0)
    output = model(**data)
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-6)
    output.square().sum().backward()
    for key in ("support", "support_labels"):
        assert data[key].grad[:, :6].norm() > 0
        assert data[key].grad[:, 6:].count_nonzero() == 0


def test_both_heads_get_gradients_and_zero_shot_ignores_support():
    data = fixture()
    data["query"].requires_grad_()
    data["support"].requires_grad_()
    model = DualRecordingAttention(AttentionSpec(16, 4, 2, 0), text_dim=12, n_layers=2).eval()
    enrolled = torch.tensor([False, True])
    output = model(**data, enrolled=enrolled)
    F.cross_entropy(output, torch.tensor([0, 1])).backward()
    assert data["query"].grad.norm(dim=-1).min() > 0
    assert data["support"].grad[0].count_nonzero() == 0
    assert data["support"].grad[1].norm() > 0
    for branch in (model.zero_shot, model.enrollment):
        for module in (branch.signal, branch.text, branch.stack):
            assert all(p.grad is not None and p.grad.norm() > 0 for p in module.parameters())
    other = {**data, "support": data["support"] * 100,
             "support_labels": data["support_labels"] * -100}
    torch.testing.assert_close(model(**other, enrolled=enrolled)[0], output[0])
    assert not any(a is b for a in model.zero_shot.parameters() for b in model.enrollment.parameters())


def test_neighbor_objective_is_correct_mass_and_both_roles_learn():
    data = fixture()
    q = data["query"].requires_grad_()
    s = data["support"].requires_grad_()
    bound = torch.tensor([[0, 0, 1, 1, 2, 2]]).expand(2, -1)
    logits, weight = neighbor_logits(q, s, bound, data["support_mask"], data["candidate_mask"])
    targets = torch.tensor([0, 2])
    loss = F.cross_entropy(logits, targets)
    expected = -((weight * bound.eq(targets[:, None])).sum(1)).log().mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert q.grad.norm(dim=-1).min() > 0
    assert s.grad.norm(dim=-1).min() > 0
    assert torch.isfinite(q.grad).all() and torch.isfinite(s.grad).all()


def test_neighbor_padding_empty_sets_and_unbound_rejection():
    data = fixture()
    bound = torch.tensor([[0, 0, 1, 1, 2, 2]]).expand(2, -1)
    logits, _ = neighbor_logits(data["query"], data["support"], bound,
                                data["support_mask"], data["candidate_mask"])
    data["support_mask"][:, -1] = False
    data["support"].requires_grad_()
    output, weight = neighbor_logits(data["query"], data["support"], bound,
                                    data["support_mask"], data["candidate_mask"])
    output.sum().backward()
    assert data["support"].grad[:, -1].count_nonzero() == 0
    assert weight[:, -1].count_nonzero() == 0
    empty, _ = neighbor_logits(data["query"], data["support"][:, :0], bound[:, :0],
                              data["support_mask"][:, :0], data["candidate_mask"])
    assert empty.count_nonzero() == 0
    with pytest.raises(ValueError, match="enrolled"):
        neighbor_logits(data["query"], data["support"], bound - 1,
                        data["support_mask"], data["candidate_mask"])


@pytest.mark.parametrize("readout", ["dual_attention", "neighbors"])
def test_checkpoint_roundtrip_and_dispatch(readout):
    data = fixture()
    spec = AttentionSpec(16, 4, 2, 0)
    config = ComparatorConfig(text_dim=12, readout=readout)
    model = SupportComparator(spec, config).eval()
    clone = SupportComparator(spec, ComparatorConfig(**dataclasses.asdict(config))).eval()
    clone.load_state_dict(model.state_dict())
    kwargs = dict(candidate_text=data["candidates"], query_feature=data["query"][:, None],
                  query_descriptor=torch.zeros(2, 1, 12), query_mask=torch.ones(2, 1, dtype=torch.bool),
                  support_feature=data["support"], support_descriptor=data["support_labels"],
                  support_label_text=data["support_labels"], support_mask=data["support_mask"],
                  support_bound=torch.tensor([[0, 0, 1, 1, 2, 2]]).expand(2, -1),
                  candidate_mask=data["candidate_mask"], center=False, instance_ids=data["instance_ids"])
    a, b = comparator_logits(model, **kwargs), comparator_logits(clone, **kwargs)
    torch.testing.assert_close(a["logits"], b["logits"])
    if readout == "neighbors":
        assert sum(p.numel() for p in model.parameters()) == 0
    with pytest.raises(ValueError, match="center"):
        comparator_logits(model, **{**kwargs, "center": True})


def test_native_zero_shot_adapter_does_not_build_a_bank(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from baselines.halo_compare.adapter import HALOCompareAdapter
    from baselines.base import UnsupportedEvaluationCell

    adapter = HALOCompareAdapter()
    features = np.zeros((2, 16), dtype=np.float32)
    state = dict(feature_owner={id(features): SimpleNamespace(dataset="test", stream="wrist")},
                 comparator=SupportComparator(AttentionSpec(16, 4, 2, 0),
                    ComparatorConfig(text_dim=12, readout="dual_attention")).eval(),
                 device=torch.device("cpu"), center=False,
                 sbert=lambda texts: np.ones((len(texts), 12), dtype=np.float32))
    monkeypatch.setattr(adapter, "_stream_rows", lambda *args: (
        torch.randn(2, 16), torch.randn(2, 12), None))
    monkeypatch.setattr(adapter, "_zero_shot_draws", lambda *args: pytest.fail("bank was accessed"))
    predictions, info = adapter.predict_candidates_from_features(features, ["a", "b"], state, "cpu")
    assert len(predictions) == 2 and info["support_rows"] == 0
    state["comparator"] = SupportComparator(AttentionSpec(16, 4, 2, 0),
                                            ComparatorConfig(readout="neighbors"))
    with pytest.raises(UnsupportedEvaluationCell, match="no native zero-shot"):
        adapter.predict_candidates_from_features(features, ["a", "b"], state, "cpu")


def test_execution_pooling_reuses_support_encoding_and_backpropagates():
    from training.compare.sampling import Episode, Recording, SupportCorpus
    from training.compare.train import episode_positions, split_encoded
    from data.scripts.curate.compatibility import AcquisitionKey

    recordings = [Recording(0, i, "ds", "wrist", "a" if i < 3 else "b",
                            f"s{i}", f"e{i}") for i in range(6)]
    corpus = SupportCorpus(recordings, [AcquisitionKey(
        "watch", "wrist", ("acc_x", "acc_y", "acc_z"), "present")], [("ds", "wrist")])
    common = dict(support=(1, 3), support_candidate=(0, 1), candidates=("a", "b"),
                  gt_slot=0, mode="compatible", requested_support=2,
                  shrunk=False,
                  support_window_groups=((1, 2), (3, 4)), support_per_candidate=1,
                  support_set_id=0)
    episodes = [Episode(query=0, **common), Episode(query=5, **{**common, "gt_slot": 1})]
    assert episode_positions(episodes, corpus) == [0, 1, 2, 3, 4, 5]
    pooled = torch.arange(6, dtype=torch.float32).view(6, 1).requires_grad_()
    descriptor = F.normalize(torch.randn(6, 4), dim=-1)
    rows = split_encoded(pooled, descriptor, episodes, corpus)
    torch.testing.assert_close(rows["support_feature"][:, :, 0],
                               torch.tensor([[1.5, 3.5], [1.5, 3.5]]))
    rows["support_feature"].sum().backward()
    assert pooled.grad[1:5].eq(1).all()  # two reuses times 1/2 pooling weight
    assert pooled.grad[[0, 5]].count_nonzero() == 0


def test_neighbor_execution_pooling_matches_cosine_evaluation_and_backpropagates():
    from model.evidence.prediction import cosine_execution_pool
    from training.compare.sampling import Episode, Recording, SupportCorpus
    from training.compare.train import split_encoded
    from data.scripts.curate.compatibility import AcquisitionKey

    recordings = [Recording(0, i, "ds", "wrist", "a", f"s{i}", f"e{i}") for i in range(3)]
    corpus = SupportCorpus(recordings, [AcquisitionKey(
        "watch", "wrist", ("acc_x", "acc_y", "acc_z"), "present")], [("ds", "wrist")])
    episode = Episode(
        query=0, support=(1,), support_candidate=(0,), candidates=("a",), gt_slot=0,
        mode="compatible", requested_support=1, shrunk=False,
        support_window_groups=((1, 2),), support_per_candidate=1, support_set_id=0,
    )
    pooled = torch.tensor([[1.0, 0.0], [10.0, 0.0], [0.0, 1.0]], requires_grad=True)
    descriptor = F.normalize(torch.randn(3, 4), dim=-1)
    rows = split_encoded(pooled, descriptor, [episode], corpus, cosine_support=True)
    expected = cosine_execution_pool(pooled[torch.tensor([1, 2])])
    torch.testing.assert_close(rows["support_feature"][0, 0], expected)
    rows["support_feature"][0, 0, 0].backward()
    assert pooled.grad[1:].abs().sum() > 0


def test_random_encoder_persists_the_exact_rope_period(monkeypatch):
    import training.tokenizer.pretrain_episodic as module

    class DummyEncoder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to(self, _device):
            return self

    monkeypatch.setattr(module, "SetTokenizerEncoder", DummyEncoder)
    encoder, config = module._random_encoder(
        torch.device("cpu"), "multispan",
        duration_range=(0.5, 1.5), num_resolutions=3,
        frontend_kwargs={
            "spans": (0.5, 1.0, 1.5), "frames_per_span": 8,
            "rope_min_period": 0.125,
        },
    )
    assert encoder.kwargs["rope_min_period"] == 0.125
    assert config["rope_min_period"] == 0.125

    default_encoder, default_config = module._random_encoder(torch.device("cpu"))
    assert default_encoder.kwargs["rope_min_period"] == module.ROPE_MIN_PERIOD_S
    assert default_config["rope_min_period"] == module.ROPE_MIN_PERIOD_S


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast dtype behaviour is a CUDA property")
@pytest.mark.parametrize("readout", ["dual_attention", "neighbors"])
def test_staged_readouts_score_in_fp32_under_autocast(readout):
    """Autocast runs ``einsum`` in bf16 even on ``.float()`` inputs. That quantised the kNN logits
    and made ``index_copy`` into the fp32 output fail on CUDA at the first validation step."""
    data = fixture()
    device = torch.device("cuda")
    model = SupportComparator(AttentionSpec(16, 4, 2, 0),
                              ComparatorConfig(text_dim=12, readout=readout)).to(device).eval()
    bound = torch.tensor([[0, 0, 1, 1, 2, 2]]).expand(2, -1).to(device)
    kwargs = dict(candidate_text=data["candidates"].to(device),
                  query_feature=data["query"][:, None].to(device),
                  query_descriptor=torch.zeros(2, 1, 12, device=device),
                  query_mask=torch.ones(2, 1, dtype=torch.bool, device=device),
                  support_feature=data["support"].to(device),
                  support_descriptor=data["support_labels"].to(device),
                  support_label_text=data["support_labels"].to(device),
                  support_mask=data["support_mask"].to(device), support_bound=bound,
                  candidate_mask=data["candidate_mask"].to(device), center=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = comparator_logits(model, **kwargs)
        # a mixed batch: one zero-shot episode, one enrolled, as the trainer produces
        mixed = dict(kwargs)
        mixed["support_mask"] = kwargs["support_mask"].clone(); mixed["support_mask"][0] = False
        mixed["support_bound"] = bound.clone(); mixed["support_bound"][0] = -1
        out_mixed = comparator_logits(model, **mixed)
    for result in (out, out_mixed):
        assert result["logits"].dtype == torch.float32 and result["base_logits"].dtype == torch.float32
        assert torch.isfinite(result["logits"]).all()
