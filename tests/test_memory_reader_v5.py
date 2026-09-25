"""Contract tests for the experimental causal memory reader."""

import numpy as np
import pytest
import torch

from model.blocks import AttentionSpec
from model.support.factory import build_classifier_from_blob
from model.support.memory_classifier import (
    ARCHITECTURE_VERSION, MemoryReaderClassifier, MemoryReaderConfig,
)
from training.support_classifier.memory_bank import DeploymentMemory
from training.support_classifier.memory_episodes import (
    draw_memory_episode, eligible_memory_datasets,
)
from training.support_classifier.sampling import Recording, SupportCorpus
from training.support_classifier.train_memory import memory_size_bin


def _model():
    spec = AttentionSpec(d_model=8, n_heads=2, ffn_mult=2)
    return MemoryReaderClassifier(spec, MemoryReaderConfig(text_dim=6, hidden_dim=8,
                                                         num_heads=2, max_entries=4))


def _inputs():
    torch.manual_seed(4)
    return torch.randn(8), torch.randn(8), torch.randn(3, 6)


def test_empty_bank_is_exact_semantic_and_round_trips():
    model = _model()
    q, acq, labels = _inputs()
    bank = DeploymentMemory("x", capacity=4)
    out = model(q, acq, labels, *bank.tensors(model, ("a", "b", "c"), labels,
                                              empty_device=q.device))
    torch.testing.assert_close(out.probabilities, model.semantic(q, labels), rtol=0, atol=0)
    blob = {"architecture_version": ARCHITECTURE_VERSION, "attention_spec": {
        "d_model": 8, "n_heads": 2, "ffn_mult": 2,
    }, "classifier_config": vars(model.cfg), "classifier": model.state_dict()}
    restored, version = build_classifier_from_blob(blob)
    assert version == ARCHITECTURE_VERSION
    torch.testing.assert_close(restored.semantic(q, labels), out.probabilities, rtol=0, atol=0)


def test_memory_order_equivariance_and_gradient_through_history():
    model = _model()
    q, acq, labels = _inputs()
    q.requires_grad_()
    motion = torch.randn(3, 8, requires_grad=True)
    acquisition = torch.randn(3, 8, requires_grad=True)
    evidence = torch.softmax(torch.randn(3, 3), -1)
    verified = torch.tensor([-1, 1, -1])
    out = model(q, acq, labels, motion, acquisition, evidence, verified)
    order = torch.tensor([2, 0, 1])
    shuffled = model(q, acq, labels, motion[order], acquisition[order], evidence[order],
                     verified[order])
    torch.testing.assert_close(out.probabilities, shuffled.probabilities, atol=1e-6, rtol=1e-6)
    (-out.probabilities[1].log()).backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert motion.grad is not None and motion.grad.abs().sum() > 0
    assert acquisition.grad is not None and acquisition.grad.abs().sum() > 0
    assert model.p_text.weight.grad is not None
    assert model.trust[-1].weight.grad is not None
    assert model.gate[-1].weight.grad is not None


def test_predict_before_insert_and_roster_recompute():
    model = _model()
    q, acq, labels = _inputs()
    bank = DeploymentMemory("x", capacity=4)
    first = bank.predict_then_insert(model, recording_id="r1", execution_id="e1",
                                    motion=q, acquisition=acq, labels=("a", "b", "c"),
                                    candidate_text=labels)
    assert len(bank.entries) == 1 and bank.seen == 1
    torch.testing.assert_close(first.probabilities, first.semantic, rtol=0, atol=0)
    assert bank.entries[0].verified_label is None
    assert bank.entries[0].observation_index == 0
    assert bank.entries[0].original_roster == ("a", "b", "c")
    new = ("c", "a", "b")
    _, _, rescored, _ = bank.tensors(model, new, labels[[2, 0, 1]], empty_device=q.device)
    torch.testing.assert_close(rescored[0], model.semantic(q, labels[[2, 0, 1]]))
    bank.verify("r1", "b")
    assert bank.entries[0].verified_label == "b"
    with pytest.raises(ValueError, match="duplicate"):
        bank.insert(recording_id="r2", execution_id="e1", motion=q, acquisition=acq,
                    original_probabilities=first.semantic, roster=("a", "b", "c"))


def test_eviction_protects_verified_and_ignores_pseudo_class():
    model = _model()
    q, acq, labels = _inputs()
    bank = DeploymentMemory("x", capacity=2)
    for i in range(3):
        bank.insert(recording_id=f"r{i}", execution_id=f"e{i}", motion=q + i,
                    acquisition=acq, original_probabilities=model.semantic(q + i, labels),
                    roster=("a", "b", "c"), verified_label="a" if i == 0 else None)
    assert len(bank.entries) == 2 and bank.seen == 3
    assert any(e.recording_id == "r0" for e in bank.entries)


def test_full_verified_bank_counts_unretained_observation():
    model = _model()
    q, acq, labels = _inputs()
    bank = DeploymentMemory("x", capacity=1)
    bank.insert(recording_id="r0", execution_id="e0", motion=q, acquisition=acq,
                original_probabilities=model.semantic(q, labels), roster=("a", "b", "c"),
                verified_label="a")
    bank.insert(recording_id="r1", execution_id="e1", motion=q, acquisition=acq,
                original_probabilities=model.semantic(q, labels), roster=("a", "b", "c"))
    assert bank.seen == 2 and len(bank.entries) == 1
    assert bank.entries[0].recording_id == "r0"


def test_verified_label_outside_new_roster_has_no_false_vote():
    model = _model()
    q, acq, labels = _inputs()
    bank = DeploymentMemory("x")
    bank.insert(recording_id="r", execution_id="e", motion=q,
                acquisition=acq, original_probabilities=model.semantic(q, labels),
                roster=("a", "b", "c"), verified_label="a")
    new_roster = ("d", "e", "f")
    tensors = bank.tensors(model, new_roster, labels, empty_device=q.device)
    assert tensors[2].sum() == 0
    result = model(q, acq, labels, *tensors)
    torch.testing.assert_close(result.probabilities, result.semantic, rtol=0, atol=0)


def test_bank_snapshot_round_trip_and_version_guard():
    model = _model()
    q, acq, labels = _inputs()
    bank = DeploymentMemory("checkpoint-sha", capacity=2)
    bank.insert(recording_id="r", execution_id="e", motion=q,
                acquisition=acq, original_probabilities=model.semantic(q, labels),
                roster=("a", "b", "c"), acquisition_metadata={"site": "wrist"})
    restored = DeploymentMemory.from_state_dict(
        bank.state_dict(), model_version="checkpoint-sha", device=q.device,
    )
    assert restored.seen == 1 and restored.entries[0].acquisition_metadata == {"site": "wrist"}
    torch.testing.assert_close(restored.entries[0].motion, q)
    with pytest.raises(ValueError, match="version"):
        DeploymentMemory.from_state_dict(bank.state_dict(), model_version="other",
                                         device=q.device)


def test_sampler_does_not_duplicate_execution_or_verify_final_query():
    recordings = []
    for label in ("walk", "run", "sit"):
        for i in range(15):
            recordings.append(Recording(0, len(recordings), "d", "s", label,
                                        str(i % 5), f"{label}-{i}"))
    corpus = SupportCorpus(recordings, keys=[("phone",)], stream_names=[("d", "s")])
    episode = draw_memory_episode(corpus, np.random.default_rng(5), max_history=7)
    assert len(episode.rows) >= 2
    assert len({recordings[i].execution for i in episode.rows}) == len(episode.rows)
    assert not episode.verified[-1]
    assert recordings[episode.rows[-1]].label in episode.roster
    assert eligible_memory_datasets(corpus) == ("d",)


def test_memory_size_telemetry_bins_are_stable():
    assert [memory_size_bin(i) for i in (0, 1, 2, 3, 4, 7, 8, 64)] == [
        "0", "1", "2_3", "2_3", "4_7", "4_7", "8_plus", "8_plus",
    ]
