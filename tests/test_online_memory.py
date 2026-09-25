"""Online predict-then-update evaluation of the v5 memory reader: synthetic contract tests."""

import numpy as np
import torch

from evaluation.online_memory.stream import arrival_order, run_stream, summarise
from model.blocks import AttentionSpec
from model.support.memory_classifier import MemoryReaderClassifier, MemoryReaderConfig


def _reader():
    torch.manual_seed(0)
    return MemoryReaderClassifier(AttentionSpec(d_model=8, n_heads=2, ffn_mult=2),
                                  MemoryReaderConfig(text_dim=6, hidden_dim=8, num_heads=2, max_entries=16))


def _cell(n_exec=12, per_exec=3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_exec * per_exec
    truth = np.repeat(np.arange(n_exec) % 3, per_exec)
    executions = np.repeat(np.arange(n_exec), per_exec).astype(object)
    motion = rng.standard_normal((n, 8)).astype(np.float32)
    acquisition = np.tile(rng.standard_normal(8).astype(np.float32), (n, 1))
    return truth, executions, motion, acquisition


def test_arrival_order_keeps_recordings_contiguous_and_is_seeded():
    executions = np.array(["a", "a", "b", "b", "c"], dtype=object)
    order = arrival_order(executions, np.arange(5), seed=1)
    assert sorted(order.tolist()) == list(range(5))
    runs = [executions[i] for i in order]
    assert all(runs[i] == runs[i + 1] or runs[i + 1] not in runs[:i + 1] for i in range(4))
    assert (arrival_order(executions, np.arange(5), seed=1) == order).all()


def test_first_prediction_has_empty_memory_and_memory_grows_causally():
    reader = _reader()
    truth, executions, motion, acquisition = _cell()
    result = run_stream(reader, motion=motion, acquisition=acquisition, truth_ids=truth,
                        execution_ids=executions, roster=("a", "b", "c"), candidate_text=torch.randn(3, 6),
                        scored_rows=np.arange(len(truth)), seed=0, device=torch.device("cpu"))
    assert result.memory_size[0] == 0
    assert (np.diff(result.memory_size) >= 0).all()          # one entry per execution, never shrinks here
    assert result.memory_size[-1] <= 12
    assert (result.semantic[0] == result.reader[0])            # empty memory: reader == semantic
    row = summarise(result, ("a", "b", "c"))
    assert row["n_predictions"] == len(truth) and row["n_mem_0"] == 1


def test_enrolled_setting_starts_with_k_verified_entries():
    reader = _reader()
    truth, executions, motion, acquisition = _cell()
    enrolled = np.array([0, 1, 3, 4, 6, 7])                      # two per class, some sharing a recording
    result = run_stream(reader, motion=motion, acquisition=acquisition, truth_ids=truth,
                        execution_ids=executions, roster=("a", "b", "c"), candidate_text=torch.randn(3, 6),
                        scored_rows=np.arange(9, len(truth)), seed=0, device=torch.device("cpu"),
                        setting="enrolled", enrolled_rows=enrolled)
    assert result.memory_size[0] == len(enrolled)
