"""Rung 3 unit tests: the LoRA wrapper, every cached-feature treatment, the fixed-budget fit loop,
and run_cell on synthetic separable features. Raw-window treatments need real encoders and are
exercised by the CLI smoke, not here."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_blobs

from evaluation.rung1_unlabeled.ncurve import CellSplit
from evaluation.rung3_finetune.finetune import (
    FineTuneConfig, enrollment_frozen_predictions, fit_head, linear_probe_predictions, run_cell,
    small_classifier_predictions,
)
from evaluation.rung3_finetune.lora import LoRALinear, apply_lora, lora_parameters, trainable_parameter_count


def test_lora_is_the_identity_at_init_and_trains_only_the_adapter():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 5), nn.GELU(), nn.Linear(5, 3))
    x = torch.randn(4, 6)
    before = net(x).detach().clone()
    wrapped = apply_lora(net, rank=2, alpha=4.0)
    assert wrapped == ["0", "2"]
    assert torch.allclose(net(x), before)                              # B is zero at init
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in trainable) and len(lora_parameters(net)) == 4
    assert trainable_parameter_count(net) == 2 * 6 + 5 * 2 + 2 * 5 + 3 * 2
    # after a step the merged Linear equals the wrapped module
    net(x).sum().backward()
    with torch.no_grad():
        for p in lora_parameters(net):
            p -= 0.1 * p.grad
    layer = net[0]
    assert isinstance(layer, LoRALinear)
    assert torch.allclose(layer(x), layer.merged()(x), atol=1e-6)


def _blob_features(seed=0, k=3, n=90, dim=16):
    x, y = make_blobs(n_samples=n, centers=k, n_features=dim, cluster_std=0.1, random_state=seed)
    return x.astype(np.float32), y.astype(np.int64)


def _supports(y, k, seed=0):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for c in np.unique(y):
        pick = rng.choice(np.flatnonzero(y == c), size=k, replace=False)
        rows.extend(pick.tolist()); labels.extend([int(c)] * k)
    return np.asarray(rows), np.asarray(labels)


def test_cached_feature_treatments_solve_separable_blobs():
    x, y = _blob_features()
    support_rows, support_labels = _supports(y, 2)
    scored = np.setdiff1d(np.arange(len(y)), support_rows)
    cfg = FineTuneConfig(steps=60, lr=1e-2, seed=0)
    frozen = enrollment_frozen_predictions(x, support_rows, support_labels, scored, 3)
    probe, info = linear_probe_predictions(x, support_rows, support_labels, scored, 3, cfg)
    small, info2 = small_classifier_predictions(x, support_rows, support_labels, scored, 3, cfg, torch.device("cpu"))
    assert (frozen == y[scored]).mean() == 1.0
    assert (probe == y[scored]).mean() == 1.0 and info["trainable_params"] == 3 * 16 + 3
    assert (small == y[scored]).mean() > 0.95 and info2["trainable_params"] > 0
    assert info2["final_loss"] < info2["first_loss"]


def test_fit_head_reduces_loss_through_a_trainable_encoder():
    torch.manual_seed(0)
    x, y = _blob_features(dim=8)
    table = torch.as_tensor(x)
    encoder = nn.Linear(8, 4)
    head = nn.Linear(4, 3)
    support_rows, support_labels = _supports(y, 4)
    history = fit_head(lambda idx: encoder(table[torch.as_tensor(idx)]), list(encoder.parameters()), head,
                       support_rows, support_labels, FineTuneConfig(steps=40, lr=5e-2, encoder_lr_scale=1.0),
                       torch.device("cpu"))
    assert history["steps"] == 40 and history["final_loss"] < history["first_loss"]


def test_run_cell_scores_every_cached_treatment_on_the_scored_set_and_declares_infeasible_k():
    x, y = _blob_features(n=120)
    split = CellSplit(scored=np.arange(0, 30), pool=np.arange(30, 120), n_executions_scored=1, n_executions_pool=3)
    rows = run_cell(name="toy", stream=None, features=x, truth_ids=y, classes=["a", "b", "c"], split=split,
                    ks=(1, 2, 1000), treatments=("enrollment_frozen", "linear_probe", "small_classifier"),
                    cfg=FineTuneConfig(steps=30, lr=1e-2), device=torch.device("cpu"), seed_parts=("t",))
    ok = [r for r in rows if r["status"] == "ok"]
    assert {(r["method"], r["k"]) for r in ok} == {(m, k) for m in ("enrollment_frozen", "linear_probe", "small_classifier") for k in (1, 2)}
    assert all({"f1_macro", "accuracy", "balanced_accuracy"} <= set(r) for r in ok)
    assert all(r["n_scored"] == 30 and r["n_support"] == 3 * r["k"] for r in ok)
    assert any(r["status"] == "n/a" and r["k"] == 1000 for r in rows)
    # the enrollment side of the crossover and the probe agree on trivially separable data
    assert all(r["accuracy"] == 100.0 for r in ok if r["method"] in ("enrollment_frozen", "linear_probe"))


def test_run_cell_reports_unsupported_raw_treatments_instead_of_approximating():
    x, y = _blob_features(n=60)
    split = CellSplit(scored=np.arange(0, 20), pool=np.arange(20, 60), n_executions_scored=1, n_executions_pool=1)
    rows = run_cell(name="normwear", stream=object(), features=x, truth_ids=y, classes=["a", "b", "c"], split=split,
                    ks=(1,), treatments=("lora",), cfg=FineTuneConfig(steps=1), device=torch.device("cpu"))
    assert rows and rows[0]["status"] == "n/a" and "no fine-tuning path" in rows[0]["reason"]
