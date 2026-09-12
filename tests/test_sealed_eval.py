"""Contract tests for the sealed support-evaluation runner."""

from __future__ import annotations

import numpy as np
import torch

from baselines.data import EvalStream
from model.blocks import AttentionSpec
from model.support.token_mixer import SupportTokenMixer, TokenMixerConfig
from training.support_classifier.sealed_eval import (
    DEFAULT_K,
    _halo_token_mixer_predictions,
    _differentiable_neighbor_predictions,
    _readout_predictions,
    _training_bank_conse_predictions,
    build_manifest,
    manifest_fingerprint,
)


def test_default_support_curve_includes_large_enrollment_counts():
    assert DEFAULT_K == (0, 1, 2, 4, 8, 16, 32, 64, 128)


def _stream() -> EvalStream:
    # Three classes, three execution-disjoint examples each.  The vectors are deliberately
    # linearly separable so every common readout has a transparent expected result.
    labels = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    return EvalStream(
        dataset="toy", stream="wrist", alignment="native",
        windows=np.zeros((9, 4, 3), dtype=np.float32), gt=labels,
        subjects=np.asarray(["s1", "s2", "s3"] * 3), channels=["acc_x", "acc_y", "acc_z"],
        rate_hz=50.0, mask=np.ones(3, dtype=bool), eval_labels=["a", "b", "c"],
        execution_ids=np.asarray([f"e{i}" for i in range(9)], dtype=object),
        execution_identity_known=True,
    )


def test_manifest_is_execution_disjoint_and_deterministic():
    stream = _stream()
    first = build_manifest(stream, 1, seed=7)
    second = build_manifest(stream, 1, seed=7)
    assert first == second
    assert manifest_fingerprint(first) == manifest_fingerprint(second)
    assert first
    for plan in first:
        assert stream.execution_ids[plan.query] not in set(stream.execution_ids[list(plan.support)])
        assert len(plan.support) == len(stream.eval_labels)


def test_common_readouts_share_the_same_manifest_and_recover_separable_features():
    stream = _stream()
    features = np.repeat(np.eye(3, dtype=np.float32), 3, axis=0)
    plans = build_manifest(stream, 1, seed=9)
    predictions = _readout_predictions(features, np.asarray(stream.gt, dtype=object),
                                       stream.eval_labels, plans)
    truth = [stream.gt[plan.query] for plan in plans]
    assert set(predictions) == {"1nn", "prototype", "ridge"}
    assert all(prediction == truth for prediction in predictions.values())
    soft = _differentiable_neighbor_predictions(
        features, stream.eval_labels, plans, torch.device("cpu"), batch_size=3,
    )
    assert soft == truth


def test_k_zero_manifest_contains_no_enrollment_rows():
    plans = build_manifest(_stream(), 0)
    assert plans and all(not plan.support and not plan.support_labels for plan in plans)


def test_k_zero_uses_training_bank_neighbor_before_conse(monkeypatch):
    captured = {}

    def fake_conse(probs, train_labels, target_labels, *, top_T):
        captured["probs"] = probs.copy()
        captured["train_labels"] = list(train_labels)
        captured["target_labels"] = list(target_labels)
        captured["top_T"] = top_T
        return [target_labels[index] for index in probs.argmax(axis=1)], {}

    monkeypatch.setattr("training.support_classifier.sealed_eval.scoring.conse_predict", fake_conse)
    prediction, info = _training_bank_conse_predictions(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]], dtype=np.float32),
        np.asarray([0, 1, 2]),
        ["walk", "sit", "run"],
        ["walking", "sitting", "running"],
        torch.device("cpu"),
        query_batch_size=1,
        reference_batch_size=2,
    )
    assert prediction == ["walking", "sitting"]
    np.testing.assert_array_equal(captured["probs"], np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]))
    assert captured["top_T"] == 1
    assert info["zero_support_protocol"] == "training_bank_1nn_conse_v1"
    assert info["reference_rows"] == 3


def test_halo_token_mixer_uses_the_same_enrolled_manifest(tmp_path, monkeypatch):
    """The learned HALO readout must be executable, not a table-only placeholder."""
    stream = _stream()
    plans = build_manifest(stream, 1, seed=11)
    spec = AttentionSpec(d_model=4, n_heads=1, ffn_mult=1, dropout=0.0)
    mixer = SupportTokenMixer(spec, TokenMixerConfig(n_layers=0))
    checkpoint = tmp_path / "support.pt"
    torch.save({
        "architecture_version": "support_token_mixer_v1",
        "classifier": mixer.state_dict(),
        "classifier_config": {"n_layers": 0},
        "attention_spec": {"d_model": 4, "n_heads": 1, "ffn_mult": 1, "dropout": 0.0},
    }, checkpoint)

    class _Text:
        matrix = torch.zeros((3, 384), dtype=torch.float32)

        @staticmethod
        def ids(labels):
            return [{"a": 0, "b": 1, "c": 2}[label] for label in labels]

    monkeypatch.setattr("training.support_classifier.sealed_eval.make_label_text",
                        lambda labels, device: _Text())
    features = np.repeat(np.eye(3, 4, dtype=np.float32), 3, axis=0)
    predicted = _halo_token_mixer_predictions(features, stream, plans, checkpoint,
                                              torch.device("cpu"), batch_size=3)
    assert len(predicted) == len(plans)
    assert set(predicted) <= set(stream.eval_labels)

    zero = _halo_token_mixer_predictions(
        features, stream, build_manifest(stream, 0), checkpoint, torch.device("cpu"), batch_size=3,
    )
    assert len(zero) == stream.n_windows
    assert set(zero) <= set(stream.eval_labels)
