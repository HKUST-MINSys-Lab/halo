"""Write the support control's exact closed-form step-zero checkpoint.

The retained sensor-only reweighter is zero-initialised, so its learned and closed-form support
votes are identical at step zero. This artifact preserves that paired control for a bounded
support-classification experiment.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from model.blocks import AttentionSpec
from model.support.comparator import (
    ComparatorConfig,
    SupportComparator,
    comparator_logits,
)

IDENTITY_TOLERANCE = 1e-6


def assert_identity_at_init(
    comparator: SupportComparator,
    *,
    batch: int = 3,
    candidates: int = 5,
    support: int = 7,
    seed: int = 0,
    center: bool = True,
) -> float:
    """Check the comparator equals the closed-form vote at init; return the largest gap."""

    generator = torch.Generator().manual_seed(seed)
    d = comparator.spec.d_model
    z = comparator.cfg.text_dim
    normal = lambda *shape: torch.randn(*shape, generator=generator)  # noqa: E731

    episode = {
        "candidate_text": torch.nn.functional.normalize(normal(batch, candidates, z), dim=-1),
        "query_feature": normal(batch, 2, d),
        "query_descriptor": torch.nn.functional.normalize(normal(batch, 2, z), dim=-1),
        "query_mask": torch.ones(batch, 2, dtype=torch.bool),
        "support_feature": normal(batch, support, d),
        "support_descriptor": torch.nn.functional.normalize(normal(batch, support, z), dim=-1),
        "support_label_text": torch.nn.functional.normalize(normal(batch, support, z), dim=-1),
        "support_bound": torch.randint(-1, candidates, (batch, support), generator=generator),
        "support_mask": torch.ones(batch, support, dtype=torch.bool),
        "candidate_slot": (1 + torch.arange(candidates)).unsqueeze(0).expand(batch, candidates),
        "candidate_mask": torch.ones(batch, candidates, dtype=torch.bool),
    }
    with torch.no_grad():
        learned = comparator_logits(comparator, center=center, **episode)["logits"]
        closed = comparator_logits(None, center=center, **episode)["logits"]
    gap = float((learned - closed).abs().max())
    if gap > IDENTITY_TOLERANCE:
        raise AssertionError(
            f"the comparator is not its own closed-form vote at initialisation (gap {gap:.3e}). "
            "The untrained floor and every paired step-0 comparison assume this equality; a "
            "non-zero residual head at init silently invalidates both."
        )
    return gap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a", type=Path, required=True,
                        help="the Phase-A checkpoint the paired run warm-starts from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--center-features", action=argparse.BooleanOptionalAction, default=True,
                        help="must match the arm this control is paired against")
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    args.comparator_readout = "sensor_only"

    torch.manual_seed(args.seed)
    # Written by our own Phase-A trainer; `config` holds plain Python values beside the tensors.
    checkpoint = torch.load(args.phase_a, map_location="cpu", weights_only=False)
    d_model = int(checkpoint["config"].get("d_model", 128))
    spec = AttentionSpec(d_model=d_model, n_heads=args.n_heads, ffn_mult=2, dropout=0.1)
    comparator = SupportComparator(spec, ComparatorConfig(
        n_layers=args.n_layers, readout=args.comparator_readout,
    ))

    gap = assert_identity_at_init(comparator, center=args.center_features)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({
        # Written in the trained checkpoint's exact format, so the SAME evaluation adapter scores
        # the control and the trained arm. A control scored through a different code path is not a
        # control.
        "config": {**checkpoint["config"], "center_features": bool(args.center_features)},
        "encoder": checkpoint["encoder"],
        "comparator": comparator.state_dict(),
        "comparator_config": dataclasses.asdict(comparator.cfg),
        "attention_spec": dataclasses.asdict(spec),
        "args": {"phase_a": str(args.phase_a),
                 "center_features": bool(args.center_features)},
        "step": 0,
        "identity_gap": gap,
        "phase_a": str(args.phase_a),
    }, args.out / "step0.pt")
    (args.out / "step0.json").write_text(json.dumps({
        "step": 0,
        "identity_gap": gap,
        "tolerance": IDENTITY_TOLERANCE,
        "phase_a": str(args.phase_a),
        "d_model": d_model,
        "neutral_acquisition_text": neutral,
        "note": "the comparator at initialisation is exactly the closed-form support vote",
    }, indent=2) + "\n")
    print(f"[step0] identity gap {gap:.3e} (tolerance {IDENTITY_TOLERANCE:.0e}) -> {args.out}")


if __name__ == "__main__":
    main()
