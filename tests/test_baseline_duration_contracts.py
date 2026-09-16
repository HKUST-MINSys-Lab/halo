"""Fast contract tests for 4/8/16-second evidence budgets."""

import numpy as np
import pytest
import torch
from dataclasses import replace

from baselines import data as baseline_data
from baselines.data import EvalStream
from baselines.harnet.adapter import HarnetAdapter
from baselines.limubert_x.adapter import _window_features as limubert_features
from baselines.normwear.adapter import _normwear_groups
from baselines.unimts.adapter import UniMTSAdapter


def _stream(seconds: int, rate: int, *, channels: int = 6) -> EvalStream:
    names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"][:channels]
    values = np.linspace(0.1, 1.0, seconds * rate * channels, dtype=np.float32)
    values = values.reshape(1, seconds * rate, channels)
    return EvalStream(
        dataset="toy", stream="phone_waist", alignment="native", windows=values,
        gt=["walk"], subjects=np.asarray(["s0"]), channels=names, rate_hz=float(rate),
        mask=np.ones(channels, dtype=bool), eval_labels=["walk"],
        event_ids=np.asarray(["e0"], dtype=object), execution_ids=np.asarray(["e0"], dtype=object),
        lengths=np.asarray([seconds * rate]), window_seconds=float(seconds),
    )


def test_stream_discovery_only_uses_unqualified_grid_for_historical_six_seconds(
        tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_data, "DATASETS_DIR", tmp_path)
    legacy = tmp_path / "toy" / "grids" / "native" / "legacy"
    qualified = tmp_path / "toy" / "grids" / "native" / "qualified" / "w4"
    legacy.mkdir(parents=True)
    qualified.mkdir(parents=True)
    (legacy / "meta.json").write_text("{}")
    (qualified / "meta.json").write_text("{}")

    assert baseline_data.list_streams("toy", "native", window_seconds=4.0) == ["qualified"]
    assert baseline_data.list_streams("toy", "native", window_seconds=6.0) == ["legacy"]
    assert baseline_data.list_streams("toy", "native", window_seconds=16.0) == []


def test_quality_screen_uses_requested_budget_not_rate_quantized_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_data, "DATASETS_DIR", tmp_path)
    grid = tmp_path / "toy" / "grids" / "native" / "phone" / "w4"
    grid.mkdir(parents=True)
    np.save(grid / "data.npy", np.zeros((1, 1025, 3), dtype=np.float32))
    np.save(grid / "mask.npy", np.ones(3, dtype=bool))
    np.save(grid / "lengths.npy", np.asarray([1025], dtype=np.int64))
    (grid / "meta.json").write_text(
        '{"labels":["walk"],"subjects":["s0"],'
        '"channels":["acc_x","acc_y","acc_z"],"rate_hz":256.0,'
        '"window_seconds":4.00390625,"lengths_file":"lengths.npy"}'
    )
    (tmp_path / "toy" / "eval_labels.json").write_text(
        '{"labels":["walk"],"streams":{"phone":["walk"]}}'
    )
    seen = []
    monkeypatch.setattr(
        baseline_data, "_quality_excluded",
        lambda dataset, stream, alignment, seconds: (seen.append(seconds) or np.zeros(0, dtype=int), "applied"),
    )

    loaded = baseline_data.load_eval_stream(
        "toy", "phone", alignment="native", window_seconds=4.0, apply_quality_screen=True,
    )

    assert seen == [4.0]
    assert loaded.window_seconds == pytest.approx(4.00390625)
    assert loaded.quality_screen == "applied"


def test_harnet_feature_only_state_does_not_claim_historical_conse_head():
    adapter = HarnetAdapter()
    assert adapter.evaluation_artifacts({"model": object()}) == {}
    assert "conse_head" in adapter.evaluation_artifacts({"model": object(), "temperature": 1.0})


class _HARNet(torch.nn.Module):
    def feature_extractor(self, value):
        return torch.stack((value.mean((1, 2)), value.square().mean((1, 2))), dim=1).unsqueeze(-1)


class _LiMU(torch.nn.Module):
    def forward(self, value):
        # The released trunk returns one hidden state per sample.
        mean = value.mean(dim=-1, keepdim=True)
        return mean.expand(-1, -1, 72)


class _UniMTS:
    def __init__(self):
        self.rows = 0

    def encode_image(self, value):
        self.rows += len(value)
        mean = value.mean((1, 2, 3, 4))
        return torch.stack((mean, torch.ones_like(mean)), dim=1)


@pytest.mark.parametrize("seconds", [4, 8, 16])
def test_released_adapters_consume_complete_requested_duration(seconds):
    harnet = HarnetAdapter().window_features(
        _stream(seconds, 30, channels=3), {"model": _HARNet()}, torch.device("cpu"),
    )
    limubert = limubert_features(_stream(seconds, 100), _LiMU(), torch.device("cpu"))
    unimts_model = _UniMTS()
    unimts = UniMTSAdapter().window_embeddings(
        _stream(seconds, 20, channels=3), {"model": unimts_model}, torch.device("cpu"),
    )
    normwear, _ = _normwear_groups(_stream(seconds, 65))

    assert harnet.shape == (1, 2)
    assert limubert.shape == (1, 72)
    assert unimts.shape == (1, 2)
    # UniMTS accepts the full requested interval in one native forward pass.
    assert unimts_model.rows == 1
    assert len(normwear) == 1
    assert normwear[0][0].shape == (1, 6, seconds * 65)


def test_normwear_grouped_preprocessing_matches_individual_rows():
    base = _stream(8, 50)
    windows = np.concatenate([base.windows, base.windows * 1.7, base.windows * 0.6], axis=0)
    lengths = np.asarray([400, 317, 400], dtype=np.int64)
    stream = replace(
        base, windows=windows, gt=["walk"] * 3, subjects=np.asarray(["a", "b", "c"]),
        event_ids=np.asarray(["e0", "e1", "e2"], dtype=object),
        execution_ids=np.asarray(["e0", "e1", "e2"], dtype=object), lengths=lengths,
    )
    grouped, channels = _normwear_groups(stream)
    assert channels == 6
    by_owner = {int(owner): values[index]
                for values, owners in grouped for index, owner in enumerate(owners)}
    for row in range(3):
        single = replace(
            stream, windows=windows[row:row + 1], gt=["walk"], subjects=stream.subjects[row:row + 1],
            event_ids=stream.event_ids[row:row + 1], execution_ids=stream.execution_ids[row:row + 1],
            lengths=lengths[row:row + 1],
        )
        expected_groups, _ = _normwear_groups(single)
        expected = expected_groups[0][0][0]
        np.testing.assert_allclose(by_owner[row], expected, rtol=2e-6, atol=2e-6)
