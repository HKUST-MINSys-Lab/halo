"""Pure core of the online evaluation: replay one cell as a stream through a v5 memory reader.

Protocol (one cell, one arrival order):

* The stream is rung 1's scored set S (a fixed 20 % of executions) in a seeded order: executions are
  shuffled, windows within an execution keep their recorded order (a deployment sees a recording's
  windows consecutively).
* ``setting="unlabelled"`` (the tier-1 condition): memory starts empty and every window is inserted,
  unverified, **after** it is predicted.
* ``setting="enrolled"`` (the tier-2 condition): k verified windows per class from the
  execution-disjoint pool P are inserted first; the stream then proceeds as above.
* Every window is predicted before it is inserted. For each prediction three readouts come from the
  same forward pass: the learned reader, its fixed equal-blend vote (the floor), and the no-memory
  semantic path. Metrics are on S.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from training.support_classifier.memory_bank import DeploymentMemory
from training.support_classifier.train_memory import MEMORY_SIZE_BINS, memory_size_bin


@dataclass(frozen=True)
class StreamResult:
    order: np.ndarray                  # row indices of S in arrival order
    truth: np.ndarray                  # roster index per arrival
    reader: np.ndarray                 # predicted roster index per arrival
    fixed: np.ndarray
    semantic: np.ndarray
    memory_size: np.ndarray            # retained entries before each prediction
    semantic_weight: np.ndarray        # mean per-candidate semantic weight per prediction


def arrival_order(execution_ids: np.ndarray, rows: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle executions, keep each execution's windows in their recorded order."""
    rows = np.asarray(rows, dtype=np.int64)
    ex = np.asarray(execution_ids, dtype=object)[rows]
    groups: dict = {}
    for row, e in zip(rows.tolist(), ex.tolist()):
        groups.setdefault(e, []).append(row)
    keys = sorted(groups, key=str)
    np.random.default_rng(seed).shuffle(keys)
    return np.asarray([r for k in keys for r in sorted(groups[k])], dtype=np.int64)


@torch.no_grad()
def run_stream(reader, *, motion: np.ndarray, acquisition: np.ndarray, truth_ids: np.ndarray,
               execution_ids: np.ndarray, roster: tuple[str, ...], candidate_text: torch.Tensor,
               scored_rows: np.ndarray, seed: int, device: torch.device, setting: str = "unlabelled",
               enrolled_rows: np.ndarray | None = None, model_version: str = "v5",
               capacity: int | None = None) -> StreamResult:
    if setting not in {"unlabelled", "enrolled"}:
        raise ValueError(f"unknown setting {setting!r}")
    if setting == "enrolled" and (enrolled_rows is None or not len(enrolled_rows)):
        raise ValueError("the enrolled setting needs enrolled_rows from the pool")
    reader.eval()
    bank = DeploymentMemory(model_version, capacity=int(capacity or reader.cfg.max_entries))
    m = torch.as_tensor(np.asarray(motion, dtype=np.float32), device=device)
    a = torch.as_tensor(np.asarray(acquisition, dtype=np.float32), device=device)
    text = candidate_text.to(device)
    if setting == "enrolled":
        for row in np.asarray(enrolled_rows, dtype=np.int64).tolist():
            # Each enrolled window is its own entry: k supports may share a recording, and the
            # bank's one-entry-per-execution rule (meant for a live stream) would otherwise merge
            # them and silently shrink k.
            bank.insert(recording_id=f"enrol:{row}", execution_id=f"enrol:{row}",
                        motion=m[row], acquisition=a[row],
                        original_probabilities=reader.semantic(m[row], text),
                        roster=roster, verified_label=roster[int(truth_ids[row])])
    order = arrival_order(execution_ids, scored_rows, seed)
    out = {k: [] for k in ("reader", "fixed", "semantic", "size", "weight")}
    for row in order.tolist():
        out["size"].append(len(bank.entries))
        readout = bank.predict_then_insert(
            reader, recording_id=f"win:{row}", execution_id=f"exec:{execution_ids[row]}",
            motion=m[row], acquisition=a[row], labels=roster, candidate_text=text,
        )
        out["reader"].append(int(readout.probabilities.argmax()))
        out["fixed"].append(int(readout.fixed.argmax()))
        out["semantic"].append(int(readout.semantic.argmax()))
        out["weight"].append(float(readout.semantic_weight.mean()))
    return StreamResult(order=order, truth=np.asarray(truth_ids)[order],
                        reader=np.asarray(out["reader"]), fixed=np.asarray(out["fixed"]),
                        semantic=np.asarray(out["semantic"]), memory_size=np.asarray(out["size"]),
                        semantic_weight=np.asarray(out["weight"]))


def summarise(result: StreamResult, roster: tuple[str, ...]) -> dict:
    """Macro-F1 / accuracy per readout, corrections vs new errors, and accuracy by memory size."""
    from evaluation.metrics import classification

    names = np.asarray(roster, dtype=object)
    row = {}
    for key in ("reader", "fixed", "semantic"):
        metrics = classification(names[result.truth], names[getattr(result, key)], f1_classes=list(roster))
        row[f"{key}_f1_macro"] = metrics["f1_macro"]
        row[f"{key}_accuracy"] = metrics["accuracy"]
    right = result.reader == result.truth
    sem_right = result.semantic == result.truth
    row["corrections"] = int((right & ~sem_right).sum())
    row["new_errors"] = int((~right & sem_right).sum())
    row["n_predictions"] = int(len(result.truth))
    row["mean_semantic_weight"] = float(result.semantic_weight.mean()) if len(result.truth) else None
    for size_bin in MEMORY_SIZE_BINS:
        sel = np.asarray([memory_size_bin(int(s)) == size_bin for s in result.memory_size])
        for key in ("reader", "fixed", "semantic"):
            row[f"acc_{key}_mem_{size_bin}"] = (float(100 * (getattr(result, key)[sel] == result.truth[sel]).mean())
                                                if sel.any() else None)
        row[f"n_mem_{size_bin}"] = int(sel.sum())
    return row
