"""The evaluation adapter must compute the SAME function the trainer trained.

Training support rows are random subset means of raw pooled encoder outputs at the encoder's native
scale (``training.compare.train.recording_rows`` → ``split_encoded``). The adapter uses the less
noisy full-execution mean. Neither side L2-normalises. This matters beyond the
closed-form cosine, which is scale-free: episode centering averages query and support rows together,
and the learned residual was fitted to raw-scale rows. A unit-norm support row next to a raw-scale
query row would let the query dominate the episode mean and hand the residual an input it never saw.

These tests drive ``HALOCompareAdapter.predict_enrollment`` with synthetic encoder outputs and
compare it against ``comparator_logits`` called exactly as ``training.compare.train.run_step`` calls
it, with the residual head deliberately non-zero so the learned path is exercised.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from baselines.halo_compare.adapter import HALOCompareAdapter
from model.blocks import AttentionSpec
from model.evidence.comparator import ComparatorConfig, SupportComparator, comparator_logits
from training.compare.train import TAU_SUPPORT, VOTE_SCALE

D_MODEL = 16
TEXT_DIM = 384
RAW_SCALE = 9.0   # far from unit norm, like the real trunk's pooled output


def test_zero_shot_support_passes_selected_valid_lengths(monkeypatch):
    from types import SimpleNamespace
    import training.tokenizer.eval_transfer as transfer

    data = np.zeros((2, 300, 6), dtype=np.float32)
    ref = SimpleNamespace(channels=[f"{s}_{a}" for s in ("acc", "gyro") for a in "xyz"],
                          mask=np.ones(6, bool), rate_hz=50., load_data=lambda: data,
                          load_lengths=lambda: np.array([108, 200]))
    captured = []

    def encode(enc, windows, *args, **kwargs):
        captured.append(kwargs["lengths"].tolist())
        return {"pooled": torch.ones(len(windows), 8)}

    monkeypatch.setattr(transfer, "encode_dataset_detailed", encode)
    encoder = SimpleNamespace(text_encoder=SimpleNamespace(
        encode_pooled=lambda texts, device: torch.ones(len(texts), 384)))
    state = {"_grid_refs": {("harmes", "watch_wrist"): ref},
             "encoder": encoder, "device": torch.device("cpu")}
    rows = [SimpleNamespace(dataset="harmes", stream="watch_wrist", window_index=i, label="walking")
            for i in (1, 0)]
    HALOCompareAdapter()._encode_training_recordings(rows, state)
    assert captured == [[200, 108]]


def test_halo_restores_cached_feature_stream_identity():
    stream, features, state = object(), np.zeros((2, 8), np.float32), {"feature_owner": {}}
    HALOCompareAdapter().restore_window_features(stream, features, state, "cpu")
    assert state["feature_owner"][id(features)] is stream


@dataclass
class _Stream:
    dataset: str
    stream: str


def _sbert(texts):
    """Deterministic unit text vectors keyed by content, process-independent."""
    out = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
        vector = np.random.default_rng(seed).standard_normal(TEXT_DIM).astype(np.float32)
        out.append(vector / np.linalg.norm(vector))
    return np.stack(out)


def _rows(n: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    feature = torch.randn(n, D_MODEL, generator=generator) * RAW_SCALE
    descriptor = F.normalize(torch.randn(n, TEXT_DIM, generator=generator), dim=-1)
    return feature, descriptor, feature.numpy()


@pytest.fixture(params=["sensor_only", "fused"])
def comparator(request):
    torch.manual_seed(0)
    spec = AttentionSpec(d_model=D_MODEL, n_heads=4, ffn_mult=2, dropout=0.0)
    module = SupportComparator(spec, ComparatorConfig(
        text_dim=TEXT_DIM, n_layers=1, readout=request.param,
        use_descriptor=(request.param == "fused"),
    ))
    # A live learned path: zero init would make every representation choice invisible.
    torch.nn.init.normal_(module.head.weight, std=0.5)
    return module.eval()


PLAN = {
    "subject": "s1",
    "candidate_names": ["a", "b", "c"],
    # Executions of one, two and three windows, so pooling is exercised.
    "support_execution_rows": [[[0], [1]], [[2, 3], [4]], [[5], [6, 7, 8]]],
    "support_execution_ids": [["e0", "e1"], ["e2", "e3"], ["e4", "e5"]],
    "query_rows": list(range(8)),
    "query_execution_ids": ["q"],
}


def _run_adapter(monkeypatch, comparator, *, center: bool, relation: str = "identical"):
    query = _rows(8, seed=1)
    support = _rows(9, seed=2)
    streams = {"q": query, "s": support}
    captured = {}
    original_score = HALOCompareAdapter._score

    def stream_rows(self, stream, state):
        return streams[stream.stream]

    def score(self, state, *args):
        captured["support_feature"] = args[2].detach().clone()
        captured["query_feature"] = args[0].detach().clone()
        return original_score(self, state, *args)

    monkeypatch.setattr(HALOCompareAdapter, "_stream_rows", stream_rows)
    monkeypatch.setattr(HALOCompareAdapter, "_score", score)
    import eval.perturbation as perturbation

    monkeypatch.setattr(perturbation, "compatibility_relation", lambda q, s: relation)
    state = {
        "device": torch.device("cpu"), "sbert": _sbert, "comparator": comparator,
        "center": center, "streams": {}, "feature_owner": {},
    }
    predictions, info = HALOCompareAdapter().predict_enrollment(
        _Stream("synthetic", "q"), _Stream("synthetic", "s"), PLAN, 2,
        list(PLAN["candidate_names"]), state, torch.device("cpu"), seed=0,
    )
    return predictions, info, captured, query, support


def _training_layout_logits(comparator, query, support, *, center: bool):
    """``comparator_logits`` exactly as the trainer's ``run_step`` builds its inputs."""
    q_feature, q_descriptor, _ = query
    s_feature, s_descriptor, _ = support
    candidate_text = F.normalize(torch.from_numpy(_sbert(PLAN["candidate_names"])), dim=-1)
    rows, descriptors, bound = [], [], []
    for slot, executions in enumerate(PLAN["support_execution_rows"]):
        for execution in executions[:2]:
            index = torch.as_tensor(execution)
            rows.append(s_feature[index].mean(0))            # raw scale, like a training row
            descriptors.append(F.normalize(s_descriptor[index].mean(0), dim=0))
            bound.append(slot)
    support_feature = torch.stack(rows)
    support_descriptor = torch.stack(descriptors)
    support_bound = torch.as_tensor(bound)
    B, K, C = q_feature.shape[0], len(rows), len(PLAN["candidate_names"])
    with torch.no_grad():
        return comparator_logits(
            comparator,
            candidate_text=candidate_text.unsqueeze(0).expand(B, -1, -1),
            query_feature=q_feature.unsqueeze(1),
            query_descriptor=q_descriptor.unsqueeze(1),
            query_mask=torch.ones(B, 1, dtype=torch.bool),
            support_feature=support_feature.unsqueeze(0).expand(B, -1, -1),
            support_descriptor=support_descriptor.unsqueeze(0).expand(B, -1, -1),
            support_label_text=candidate_text[support_bound].unsqueeze(0).expand(B, -1, -1),
            support_bound=support_bound.unsqueeze(0).expand(B, -1),
            support_mask=torch.ones(B, K, dtype=torch.bool),
            candidate_slot=(1 + torch.arange(C)).unsqueeze(0).expand(B, -1),
            candidate_mask=torch.ones(B, C, dtype=torch.bool),
            temperature=TAU_SUPPORT, vote_scale=VOTE_SCALE, center=center,
        )["logits"], support_feature


@pytest.mark.parametrize("center", [True, False])
def test_adapter_support_rows_are_raw_execution_means(monkeypatch, comparator, center):
    _, info, captured, _, support = _run_adapter(monkeypatch, comparator, center=center)
    s_feature = support[0]
    expected = torch.stack([
        s_feature[torch.as_tensor(execution)].mean(0)
        for executions in PLAN["support_execution_rows"] for execution in executions[:2]
    ])
    assert torch.allclose(captured["support_feature"], expected, atol=1e-6)
    norms = captured["support_feature"].norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms)), (
        "support rows were L2-normalised; training rows never are"
    )
    # Query and support must live at one scale, or centering is dominated by the query.
    query_norm = captured["query_feature"].norm(dim=-1).mean()
    assert 0.2 < float(norms.mean() / query_norm) < 5.0
    assert info["support_representation"] == "full_execution_mean_at_encoder_scale"
    assert info["training_support_estimator"] == "random_window_subset_mean_at_encoder_scale"


@pytest.mark.parametrize("center", [True, False])
def test_adapter_predictions_match_the_training_layout(monkeypatch, comparator, center):
    predictions, _, _, query, support = _run_adapter(monkeypatch, comparator, center=center)
    logits, _ = _training_layout_logits(comparator, query, support, center=center)
    expected = [PLAN["candidate_names"][int(i)] for i in logits.argmax(1)]
    assert predictions == expected


def test_learned_path_is_live_in_the_parity_check(comparator):
    """Guard against the parity tests passing trivially with a zero residual."""
    query, support = _rows(8, seed=1), _rows(9, seed=2)
    learned, _ = _training_layout_logits(comparator, query, support, center=True)
    comparator.head.weight.data.zero_()
    closed, _ = _training_layout_logits(comparator, query, support, center=True)
    assert not torch.allclose(learned, closed)


@pytest.mark.parametrize("relation", ["near_miss", "incompatible"])
def test_arm_a_refuses_acquisition_incompatible_support(monkeypatch, comparator, relation):
    """Arm A's deployed rule: exemplars that do not share the query's key are filtered out, and a
    support set with nothing left is an unsupported cell rather than a different mechanism."""
    from baselines.base import UnsupportedEvaluationCell
    import baselines.halo_compare.adapter as module

    monkeypatch.setattr(module, "_NEUTRAL_ACQUISITION_TEXT", True)
    with pytest.raises(UnsupportedEvaluationCell, match="Arm A admits only"):
        _run_adapter(monkeypatch, comparator, center=True, relation=relation)


@pytest.mark.parametrize("relation", ["identical", "near_miss", "incompatible"])
def test_arm_b_attends_over_any_support_and_records_the_relation(monkeypatch, comparator, relation):
    import baselines.halo_compare.adapter as module

    monkeypatch.setattr(module, "_NEUTRAL_ACQUISITION_TEXT", False)
    predictions, info, _, _, _ = _run_adapter(monkeypatch, comparator, center=True, relation=relation)
    assert len(predictions) == len(PLAN["query_rows"])
    assert info["support_compatibility"] == relation


# ----------------------------------------------------------------------------- staged readouts
def _staged_comparator(readout: str):
    torch.manual_seed(0)
    spec = AttentionSpec(d_model=D_MODEL, n_heads=4, ffn_mult=2, dropout=0.0)
    return SupportComparator(spec, ComparatorConfig(
        text_dim=TEXT_DIM, n_layers=1, readout=readout,
    )).eval()


@pytest.mark.parametrize("readout", ["dual_attention", "neighbors"])
def test_staged_adapter_predictions_match_the_training_layout(monkeypatch, readout):
    """Staged heads preserve raw scale, instance order and no-centering at evaluation."""
    comparator = _staged_comparator(readout)
    predictions, info, captured, query, support = _run_adapter(
        monkeypatch, comparator, center=False,
    )
    logits, expected_support = _training_layout_logits(comparator, query, support, center=False)
    assert torch.allclose(captured["support_feature"], expected_support, atol=1e-6)
    assert predictions == [PLAN["candidate_names"][int(i)] for i in logits.argmax(1)]
    assert info["support_compatibility"] == "identical"


def test_staged_readouts_refuse_centering(monkeypatch):
    with pytest.raises(ValueError, match="center=False"):
        _run_adapter(monkeypatch, _staged_comparator("dual_attention"), center=True)
