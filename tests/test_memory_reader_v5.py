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
from training.support_classifier.train_memory import (
    EVALUATION_PROTOCOL_KEYS, indexed_corpus_fingerprint, memory_size_bin,
    validate_evaluation_protocol,
)


def test_evaluation_protocol_refuses_a_different_validation_population():
    saved = {key: 1 for key in EVALUATION_PROTOCOL_KEYS}
    validate_evaluation_protocol(saved, saved)
    for key in EVALUATION_PROTOCOL_KEYS:
        changed = {**saved, key: 2}
        with pytest.raises(ValueError, match=key):
            validate_evaluation_protocol(changed, saved)


def test_corpus_fingerprint_includes_exact_screened_rows(monkeypatch):
    from types import SimpleNamespace
    from training.support_classifier import train_memory
    from training.tokenizer.pretrain_data import WindowKey

    monkeypatch.setattr(train_memory, "corpus_fingerprint", lambda index: "same-grid")
    base = SimpleNamespace(train=[WindowKey(0, 1, 0)], val=[WindowKey(0, 2, 0)],
                           excluded={"source/stream": {5}})
    changed = SimpleNamespace(train=list(base.train), val=list(base.val),
                              excluded={"source/stream": {6}})
    assert indexed_corpus_fingerprint(base) != indexed_corpus_fingerprint(changed)


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
    _, _, rescored, _, _ = bank.tensors(model, new, labels[[2, 0, 1]], empty_device=q.device)
    torch.testing.assert_close(rescored[0], model.semantic(q, labels[[2, 0, 1]]))
    bank.verify("r1", "b")
    assert bank.entries[0].verified_label == "b"
    with pytest.raises(ValueError, match="duplicate"):
        bank.insert(recording_id="r1", execution_id="e9", motion=q, acquisition=acq,
                    original_probabilities=first.semantic, roster=("a", "b", "c"))
    # A later unlabelled window of an enrolled execution is observed, not retained.
    bank.insert(recording_id="r2", execution_id="e1", motion=q, acquisition=acq,
                original_probabilities=first.semantic, roster=("a", "b", "c"))
    assert [e.recording_id for e in bank.entries] == ["r1"] and bank.seen == 2


def test_later_window_of_an_unlabelled_execution_replaces_it():
    model = _model()
    q, acq, labels = _inputs()
    bank = DeploymentMemory("x", capacity=4)
    for i in range(3):
        bank.insert(recording_id=f"w{i}", execution_id="walk-bout", motion=q + i, acquisition=acq,
                    original_probabilities=model.semantic(q + i, labels), roster=("a", "b", "c"))
    assert [e.recording_id for e in bank.entries] == ["w2"] and bank.seen == 3


def test_eviction_protects_verified_and_ignores_pseudo_class():
    model = _model()
    q, acq, labels = _inputs()
    bank = DeploymentMemory("x", capacity=1)
    for i in range(3):
        bank.insert(recording_id=f"r{i}", execution_id=f"e{i}", motion=q + i,
                    acquisition=acq, original_probabilities=model.semantic(q + i, labels),
                    roster=("a", "b", "c"), verified_label="a" if i == 0 else None)
    assert len(bank.entries) == 2 and bank.seen == 3
    assert any(e.recording_id == "r0" for e in bank.entries)
    assert sum(e.verified_label is None for e in bank.entries) == 1


def test_verified_enrollments_are_never_capped_by_unlabelled_capacity():
    model = _model()                              # max_entries = 4
    q, acq, labels = _inputs()
    bank = DeploymentMemory("x", capacity=1)
    for i in range(6):                            # more enrollments than the reader's size scale
        bank.insert(recording_id=f"v{i}", execution_id=f"v{i}", motion=q + i, acquisition=acq,
                    original_probabilities=model.semantic(q + i, labels), roster=("a", "b", "c"),
                    verified_label="abc"[i % 3])
    for i in range(2):
        bank.insert(recording_id=f"u{i}", execution_id=f"u{i}", motion=q - i, acquisition=acq,
                    original_probabilities=model.semantic(q - i, labels), roster=("a", "b", "c"))
    assert sum(e.verified_label is not None for e in bank.entries) == 6
    assert sum(e.verified_label is None for e in bank.entries) == 1
    out = model(q, acq, labels, *bank.tensors(model, ("a", "b", "c"), labels, empty_device=q.device))
    assert torch.isfinite(out.probabilities).all()


def test_gates_never_see_label_identity_and_trust_is_bounded():
    # Relabelling the roster (same evidence values, different label texts) must not change the
    # reader's reliability weights or its semantic/memory blend weights: neither gate may see
    # label identity. Trust stays within its bound however large the MLP output.
    from model.support.memory_classifier import TRUST_LIMIT

    model = _model()
    q, acq, labels = _inputs()
    motion, acquisition = torch.randn(3, 8), torch.randn(3, 8)
    evidence = torch.softmax(torch.randn(3, 3), -1)
    verified = torch.tensor([-1, 1, -1])
    base = model(q, acq, labels, motion, acquisition, evidence, verified)
    other = model(q, acq, torch.randn(3, 6), motion, acquisition, evidence, verified)
    torch.testing.assert_close(base.reliability, other.reliability, rtol=0, atol=0)
    with torch.no_grad():
        model.trust[-1].bias.fill_(1e6)
        model.trust[-1].weight.normal_(std=1e3)
    extreme = model(q, acq, labels, motion, acquisition, evidence, verified)
    similarity = torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(motion, dim=-1).T
    logits = (extreme.reliability.log() - similarity / model.cfg.neighbor_temperature)
    assert float(logits.max() - logits.min()) <= 2 * TRUST_LIMIT + 1e-4


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


def test_verified_entry_keeps_its_original_zero_shot_prediction_as_feedback():
    model = _model()
    q, acq, labels = _inputs()
    bank = DeploymentMemory("x")
    original = torch.tensor([0.8, 0.1, 0.1])
    bank.insert(recording_id="r", execution_id="e", motion=q, acquisition=acq,
                original_probabilities=original, roster=("a", "b", "c"))
    with torch.no_grad():
        original[:] = torch.tensor([0.0, 1.0, 0.0])
    bank.verify("r", "b")
    torch.testing.assert_close(bank.entries[0].original_probabilities, torch.tensor([0.8, 0.1, 0.1]))
    tensors = bank.tensors(model, ("a", "b", "c"), labels, empty_device=q.device)
    torch.testing.assert_close(tensors[2][0], torch.tensor([0.8, 0.1, 0.1]))
    assert tensors[3].tolist() == [1]  # the verified label, not the original argmax
    torch.testing.assert_close(tensors[4][0, :3], torch.tensor([1.0, 0.1, 0.7]))
    assert 0 < tensors[4][0, 3] < 1
    assert tensors[4][0, 4] == pytest.approx(1 / 3)

    # The feedback refers to the historical roster, even when current candidates are reordered.
    reordered = bank.tensors(model, ("c", "b", "a"), labels[[2, 1, 0]], empty_device=q.device)
    torch.testing.assert_close(reordered[4], tensors[4])
    torch.testing.assert_close(reordered[2][0], model.semantic(q, labels[[2, 1, 0]]))
    restored = DeploymentMemory.from_state_dict(bank.state_dict(), model_version="x", device=q.device)
    torch.testing.assert_close(restored.tensors(model, ("a", "b", "c"), labels,
                                                empty_device=q.device)[4], tensors[4])


def test_zero_shot_feedback_needs_a_verified_label_in_both_rosters():
    model = _model()
    q, acq, labels = _inputs()
    bank = DeploymentMemory("x")
    bank.insert(recording_id="unlabelled", execution_id="e1", motion=q, acquisition=acq,
                original_probabilities=torch.tensor([0.7, 0.3]), roster=("a", "b"))
    bank.insert(recording_id="late-label", execution_id="e2", motion=q, acquisition=acq,
                original_probabilities=torch.tensor([0.7, 0.3]), roster=("a", "b"),
                verified_label="c")
    tensors = bank.tensors(model, ("a", "b", "c"), labels, empty_device=q.device)
    assert torch.count_nonzero(tensors[4]) == 0
    assert tensors[3].tolist() == [-1, 2]
    assert tensors[2][1, 2] > 0  # current-roster semantic score, not historical feedback
    with pytest.raises(ValueError, match="unverified memory"):
        model(q, acq, labels, *tensors[:4], torch.ones(2, 5))


def test_feedback_is_label_blind_but_has_a_gradient_path():
    model = _model()
    q, acq, labels = _inputs()
    motion, acquisition = torch.randn(2, 8), torch.randn(2, 8)
    evidence = torch.softmax(torch.randn(2, 3), -1)
    verified = torch.tensor([1, -1])
    feedback = torch.tensor([[1.0, 0.1, 0.7, 0.6, 1 / 3], [0.0, 0.0, 0.0, 0.0, 0.0]])
    result = model(q, acq, labels, motion, acquisition, evidence, verified, feedback)
    renamed = model(q, acq, torch.randn_like(labels), motion, acquisition, evidence,
                    verified, feedback)
    torch.testing.assert_close(result.reliability, renamed.reliability, rtol=0, atol=0)
    (-result.probabilities[1].log()).backward()
    assert model.evidence_proj.weight.grad is not None
    assert model.evidence_proj.weight.grad[:, -5:].abs().sum() > 0


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
    assert [memory_size_bin(i) for i in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64)] == [
        "0", "1", "2_3", "2_3", "4_7", "4_7", "8_15", "8_15", "16_31", "16_31", "32_plus", "32_plus",
    ]
