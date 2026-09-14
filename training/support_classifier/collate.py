"""Small data additions required by the retained support-classification control."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from training.tokenizer.pretrain_data import _pad_sensor_rows


@dataclass(frozen=True)
class BucketedSupportBatch:
    """Compact single- and multi-device sub-batches plus their original row order."""

    batches: tuple[dict, ...]
    row_indices: tuple[torch.Tensor, ...]
    restore_order: torch.Tensor
    row_count: int


class SupportCollate:
    """Carry per-window gravity and optional sensor-bias metadata through the base collate."""

    def __init__(self, base):
        self.base = base

    def __call__(self, batch: list[dict]) -> dict:
        return self._add_support_metadata(self.base(batch), batch)

    def _add_support_metadata(self, out: dict, batch: list[dict]) -> dict:
        """Attach metadata not owned by the generic tokenizer collate."""
        out["gravity_state"] = [item.get("gravity_state") for item in batch]
        if "sensor_bias" in batch[0]:
            out["sensor_bias"] = _pad_sensor_rows(batch, "sensor_bias")
        return out

    def bucketed(self, batch: list[dict]) -> BucketedSupportBatch:
        """Collate ordinary and multi-device rows separately without changing row order.

        Multi-device rows have 12--24 channels while ordinary rows have six. A single wide row
        must not force every recording in a support episode through a 24-channel FFT. Channel
        counts have only a handful of values; retaining one ordinary bucket and one multi-device
        bucket removes most padding without creating a series of tiny encoder launches.
        """
        if not batch:
            raise ValueError("cannot collate an empty support batch")
        groups: dict[int, list[tuple[int, dict]]] = {}
        for row, item in enumerate(batch):
            channels = int(item["data"].shape[1])
            # Single-device rows dominate the corpus and stay at their exact six channels. Pool
            # the much smaller multi-device tail into one padded group: this trades a little
            # padding among 12/18/24-channel rows for two encoder calls and two IPC payloads per
            # step instead of as many as four.
            group = 6 if channels <= 6 else 7
            groups.setdefault(group, []).append((row, item))

        batches = []
        indices = []
        flattened = []
        for channels in sorted(groups):
            members = groups[channels]
            row_index = torch.tensor([row for row, _ in members], dtype=torch.long)
            items = [item for _, item in members]
            deferred = getattr(self.base, "deferred", None)
            dense = deferred(items) if deferred is not None else self.base(items)
            batches.append(self._add_support_metadata(dense, items))
            indices.append(row_index)
            flattened.extend(row_index.tolist())
        restore = torch.tensor(flattened, dtype=torch.long).argsort()
        return BucketedSupportBatch(
            batches=tuple(batches), row_indices=tuple(indices),
            restore_order=restore, row_count=len(batch),
        )
