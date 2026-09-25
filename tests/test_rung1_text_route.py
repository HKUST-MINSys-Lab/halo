"""HALO's rung-1 zero-shot route: a checkpoint with a learned ``p_text`` is scored through it — the
head the pool training arm is trained through — and only a checkpoint without one falls back to the
refitted ridge bridge. Synthetic; the label encoder is a fixed random table."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.zero_shot import checkpoint_text_projection, zero_shot_scores
from model.blocks import AttentionSpec
from model.support.evidence_gated_classifier import (
    ARCHITECTURE_VERSION, EvidenceGatedClassifierConfig, EvidenceGatedSupportClassifier,
)
from training.support_classifier.train import probability_features_torch

D_MODEL, TEXT_DIM = 32, 384
LABELS = ["walking", "sitting", "cycling", "running"]


def _sbert(texts):
    table = np.random.default_rng(7).standard_normal((len(LABELS), TEXT_DIM)).astype(np.float32)
    return table[[LABELS.index(t) for t in texts]]


def _blob(seed=0) -> dict:
    torch.manual_seed(seed)
    head = EvidenceGatedSupportClassifier(
        AttentionSpec(d_model=D_MODEL, n_heads=4, ffn_mult=2, dropout=0.0),
        EvidenceGatedClassifierConfig(text_dim=TEXT_DIM),
    ).eval()
    with torch.no_grad():
        head.p_text.weight.normal_()
        head.p_text.bias.normal_()
    return {"architecture_version": ARCHITECTURE_VERSION, "classifier": head.state_dict(),
            "classifier_config": dataclasses.asdict(head.cfg), "attention_spec": dataclasses.asdict(head.spec)}


def test_checkpoint_without_a_learned_head_has_no_text_projection():
    assert checkpoint_text_projection({"classifier": None}) is None
    assert checkpoint_text_projection({}) is None


def test_p_text_route_matches_the_trainers_probability_features_exactly():
    blob = _blob()
    projection = checkpoint_text_projection(blob)
    assert projection is not None and projection[0].shape == (D_MODEL, TEXT_DIM)
    features = np.random.default_rng(1).standard_normal((10, D_MODEL)).astype(np.float32)
    scores, info = zero_shot_scores(name="halo", features=features, candidates=LABELS,
                                    device=torch.device("cpu"), halo_p_text=projection, sbert=_sbert)
    assert info["route"] == "halo_p_text"

    # The trainer's unrolled readout sees softmax(T * cos(p_text(f), text)); the evaluator must
    # produce the same simplex rows from the same checkpoint.
    head = EvidenceGatedSupportClassifier(AttentionSpec(**blob["attention_spec"]),
                                          EvidenceGatedClassifierConfig(**blob["classifier_config"]))
    head.load_state_dict(blob["classifier"])
    text = torch.from_numpy(_sbert(LABELS))[None].expand(1, -1, -1)
    trainer = probability_features_torch(torch.from_numpy(features)[None], head.p_text, text,
                                         torch.ones(1, len(LABELS), dtype=torch.bool), 30.0)[0]
    evaluator = F.softmax(30.0 * torch.from_numpy(scores), dim=-1)
    assert torch.allclose(trainer, evaluator, atol=1e-5)


def test_without_p_text_the_bridge_route_is_used_and_labelled():
    bridge = np.random.default_rng(2).standard_normal((D_MODEL, TEXT_DIM)).astype(np.float32)
    features = np.random.default_rng(3).standard_normal((5, D_MODEL)).astype(np.float32)
    _, info = zero_shot_scores(name="halo", features=features, candidates=LABELS,
                               device=torch.device("cpu"), halo_bridge=bridge, sbert=_sbert)
    assert info["route"] == "halo_text_bridge"


def test_only_text_space_projections_are_used():
    # T1's contextual head also has an attribute called p_text, but it is a d x d encoder-space
    # alignment; it must fall back to the bridge rather than crash scoring.
    assert checkpoint_text_projection({"architecture_version": "support_contextual_mixture_v1",
                                       "classifier": {"p_text.weight": torch.zeros(4, 4)}}) is None
