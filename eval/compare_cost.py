"""Measure what the comparison model costs to run, on a named device.

This script measures only the comparator forward over already-encoded synthetic rows. It reports
the device by name and repeats the measurement for several support-row counts. It does **not** time
query encoding, enrolment encoding, data loading, or a full evaluation cell, so its output must not
be presented as end-to-end deployment latency.

Run::

    python -m eval.compare_cost --checkpoint <path> --device cuda
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model.blocks import AttentionSpec
from model.evidence.comparator import ComparatorConfig, SupportComparator, comparator_logits


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine() or "unknown CPU"


def parameter_report(encoder: torch.nn.Module, comparator: torch.nn.Module) -> dict[str, int]:
    def count(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    frozen_text = 0
    text_encoder = getattr(encoder, "text_encoder", None)
    if text_encoder is not None:
        frozen_text = sum(p.numel() for p in text_encoder.parameters())
    encoder_total = count(encoder) - (
        sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
        if text_encoder is not None else 0
    )
    return {
        "encoder_trainable": int(encoder_total),
        "comparator_trainable": int(count(comparator)),
        "total_trainable": int(encoder_total + count(comparator)),
        "frozen_text_tower": int(frozen_text),
    }


@torch.no_grad()
def time_comparator(
    comparator: SupportComparator,
    *,
    device: torch.device,
    support: int,
    candidates: int = 8,
    repeats: int = 50,
    warmup: int = 10,
) -> dict[str, float]:
    """Milliseconds for one query against an already-encoded support set."""
    d = comparator.spec.d_model
    z = comparator.cfg.text_dim
    episode = {
        "candidate_text": F.normalize(torch.randn(1, candidates, z, device=device), dim=-1),
        "query_feature": torch.randn(1, 1, d, device=device),
        "query_descriptor": F.normalize(torch.randn(1, 1, z, device=device), dim=-1),
        "query_mask": torch.ones(1, 1, dtype=torch.bool, device=device),
        "support_feature": torch.randn(1, support, d, device=device),
        "support_descriptor": F.normalize(torch.randn(1, support, z, device=device), dim=-1),
        "support_label_text": F.normalize(torch.randn(1, support, z, device=device), dim=-1),
        "support_bound": torch.randint(-1, candidates, (1, support), device=device),
        "support_mask": torch.ones(1, support, dtype=torch.bool, device=device),
        "candidate_slot": (1 + torch.arange(candidates, device=device)).unsqueeze(0),
        "candidate_mask": torch.ones(1, candidates, dtype=torch.bool, device=device),
    }
    for _ in range(warmup):
        comparator_logits(comparator, **episode)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        comparator_logits(comparator, **episode)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / repeats
    return {"support": support, "query_ms": elapsed * 1000.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--support-grid", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    # Our own checkpoint; `config` carries plain Python values beside the tensors.
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from training.tokenizer.eval_transfer import build_encoder

    encoder = build_encoder(blob, device, training=False).eval()
    spec = AttentionSpec(**blob["attention_spec"])
    comparator = SupportComparator(spec, ComparatorConfig(**blob["comparator_config"]))
    comparator.load_state_dict(blob["comparator"])
    comparator = comparator.to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows = [
        time_comparator(
            comparator, device=device, support=k,
            candidates=args.candidates, repeats=args.repeats,
        )
        for k in args.support_grid
    ]
    report = {
        "device": device_name(device),
        "device_type": device.type,
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(blob.get("step", 0)),
        "measurement": "measured, not derived; arithmetic mean of repeated forwards after warmup",
        "parameters": parameter_report(encoder, comparator),
        "query_latency_by_support": rows,
        "peak_memory_gib": (
            float(torch.cuda.max_memory_allocated(device)) / 1024**3
            if device.type == "cuda" else None
        ),
        "note": (
            "query_ms is comparator latency only. It excludes query encoding; support recordings "
            "are encoded once at enrolment and reused."
        ),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
