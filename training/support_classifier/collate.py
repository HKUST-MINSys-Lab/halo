"""Small data additions required by the retained support-classification control."""

from __future__ import annotations

from training.tokenizer.pretrain_data import _pad_sensor_rows


class SupportCollate:
    """Carry per-window gravity and optional sensor-bias metadata through the base collate."""

    def __init__(self, base):
        self.base = base

    def __call__(self, batch: list[dict]) -> dict:
        out = self.base(batch)
        out["gravity_state"] = [item.get("gravity_state") for item in batch]
        if "sensor_bias" in batch[0]:
            out["sensor_bias"] = _pad_sensor_rows(batch, "sensor_bias")
        return out
