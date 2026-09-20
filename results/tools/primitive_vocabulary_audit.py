"""Label-side audit of a primitive vocabulary: pick the profile temperature, find collisions.

This runs entirely on label text — no model, no training, no sensor data — so it is the first
evidence about whether a vocabulary can work at all, and it is how ``DEFAULT_PROFILE_TEMPERATURE``
is fixed a priori rather than from any evaluation result.

It reports, per temperature:
  * mean normalised profile entropy (1.0 = uniform and therefore useless, 0.0 = one-hot);
  * the share of label pairs whose profiles are near-identical (a collision), over the training
    vocabulary, the sealed vocabulary, and both together;
  * the nearest neighbour of a few reference labels, which is the readable version of the same
    thing.

A collision between two labels an IMU genuinely cannot separate is correct. A collision between
two separable labels is a missing axis.

Usage: python results/tools/primitive_vocabulary_audit.py [--version V] [--sealed-artifact DIR]
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import torch

from model.support.primitive_semantics import (
    PRIMITIVE_VOCABULARY_VERSION, PrimitiveSemanticConfig, PrimitiveSemanticHead,
    axis_names, axis_slices, primitive_names, primitive_value_matrix, vocabulary_hash,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SEALED = REPO / "results/artifacts/baselines_sealed_v5_20260918/results.json.gz"
REFERENCE_LABELS = ("walking", "sitting", "running", "cycling", "squats", "jumping_jacks",
                    "lying_down", "typing", "vacuuming", "stair_climbing")


def sealed_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    with gzip.open(path, "rt") as handle:
        rows = json.load(handle)
    labels: set[str] = set()
    for row in rows if isinstance(rows, list) else rows.get("rows", []):
        labels.update((row.get("per_label_f1") or {}).keys())
    return sorted(labels)


def training_labels() -> list[str]:
    from baselines.data import load_global_labels

    return sorted({str(label) for label in load_global_labels()})


def collision_rate(profiles: torch.Tensor, threshold: float) -> tuple[float, int]:
    """Share of label pairs whose profiles agree above ``threshold`` (cosine on the flat vector)."""
    unit = torch.nn.functional.normalize(profiles, dim=-1)
    similarity = unit @ unit.T
    n = similarity.shape[0]
    upper = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
    hits = (similarity > threshold) & upper
    return float(hits.sum()) / max(1, int(upper.sum())), int(hits.sum())


def label_profiles(labels: list[str], version: str, temperature: float, device) -> torch.Tensor:
    """The label side exactly as the model computes it: identity projection, no training."""
    from training.support_classifier.train import label_text_matrix

    head = PrimitiveSemanticHead(
        8, PrimitiveSemanticConfig(version=version, combiner="fixed",
                                   profile_temperature=temperature),
        values=primitive_value_matrix(version, device),
    ).eval()
    text = label_text_matrix(labels, device).unsqueeze(0)
    mask = torch.ones(1, len(labels), dtype=torch.bool, device=device)
    with torch.no_grad():
        return head.candidate_profile(text, mask)[0]


def report(labels: list[str], name: str, temperature: float, version: str,
           threshold: float, device) -> None:
    profiles = label_profiles(labels, version, temperature, device)
    slices = axis_slices(version)
    entropies = []
    for block in slices:
        part = profiles[:, block]
        entropy = -(part.clamp_min(1e-12).log() * part).sum(dim=-1)
        entropies.append(entropy / torch.tensor(float(part.shape[-1])).log())
    mean_entropy = float(torch.stack(entropies, dim=-1).mean())
    rate, count = collision_rate(profiles, threshold)
    print(f"  {name:>10s} n={len(labels):4d}  mean normalised entropy {mean_entropy:.3f}  "
          f"collisions {rate * 100:5.2f}% ({count})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=PRIMITIVE_VOCABULARY_VERSION)
    parser.add_argument("--sealed-artifact", type=Path, default=DEFAULT_SEALED)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.2, 0.1, 0.05, 0.03, 0.02, 0.01])
    parser.add_argument("--collision-threshold", type=float, default=0.99)
    parser.add_argument("--neighbours-at", type=float, default=0.05)
    args = parser.parse_args()
    device = torch.device("cpu")

    print(f"vocabulary {args.version}  hash {vocabulary_hash(args.version)[:16]}")
    print(f"axes: {', '.join(axis_names(args.version))}")
    print(f"primitives: {len(primitive_names(args.version))}\n")

    train = training_labels()
    sealed = sealed_labels(args.sealed_artifact)
    both = sorted(set(train) | set(sealed))
    print(f"training labels {len(train)} | sealed labels {len(sealed)} | union {len(both)}")
    print(f"sealed labels unseen in training: "
          f"{sorted(set(sealed) - set(train))}\n")

    for temperature in args.temperatures:
        print(f"temperature {temperature}")
        for labels, name in ((train, "training"), (sealed, "sealed"), (both, "union")):
            if labels:
                report(labels, name, temperature, args.version, args.collision_threshold, device)

    print(f"\nnearest neighbour by profile at temperature {args.neighbours_at}:")
    profiles = label_profiles(both, args.version, args.neighbours_at, device)
    unit = torch.nn.functional.normalize(profiles, dim=-1)
    similarity = unit @ unit.T
    similarity.fill_diagonal_(-1.0)
    index = {label: position for position, label in enumerate(both)}
    for label in REFERENCE_LABELS:
        if label not in index:
            continue
        row = index[label]
        best = int(similarity[row].argmax())
        print(f"  {label:>16s} -> {both[best]:<24s} ({float(similarity[row, best]):.3f})")

    print("\nprofile of the reference labels (top primitive per axis):")
    names = primitive_names(args.version)
    slices = axis_slices(args.version)
    for label in REFERENCE_LABELS:
        if label not in index:
            continue
        profile = profiles[index[label]]
        top = []
        for block, axis in zip(slices, axis_names(args.version)):
            part = profile[block]
            top.append(f"{names[block.start + int(part.argmax())].split('/')[1]}")
        print(f"  {label:>16s}: {' · '.join(top)}")


if __name__ == "__main__":
    main()
