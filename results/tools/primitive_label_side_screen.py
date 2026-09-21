"""Step 6.1 of the primitive-path repair: does the label side generalise to labels it has not seen?

Holds a deterministic subset of the annotated training labels out of the frozen text-to-profile
map, fits the map on the rest, and scores the held-out labels' predicted per-axis argmax against
their written annotations. That is the direct measure of "does the label side reach labels it has
never seen", and it is what SBERT-cosine-to-sentences failed at (about half random per axis).

Runs on CPU with the frozen sentence encoder only: no training, no sensor data, no sealed labels.

Usage: python results/tools/primitive_label_side_screen.py [--held-out 30] [--seed 0]
"""
from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F

from model.support.primitive_annotations import (
    ANNOTATION_VERSION, ANNOTATIONS, annotated_labels, annotation_profile_matrix,
)
from model.support.primitive_semantics import (
    PrimitiveSemanticConfig, PrimitiveSemanticHead, axis_names, axis_slices, primitive_value_matrix,
)


def per_axis_agreement(predicted: torch.Tensor, truth: torch.Tensor) -> tuple[float, list[float]]:
    slices = axis_slices()
    hits = []
    for block in slices:
        hits.append((predicted[:, block].argmax(-1) == truth[:, block].argmax(-1)).float().mean().item())
    return sum(hits) / len(hits), hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=ANNOTATION_VERSION)
    parser.add_argument("--held-out", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    args = parser.parse_args()
    from training.support_classifier.train import label_text_matrix

    device = torch.device("cpu")
    labels, profiles = annotation_profile_matrix(args.version, smoothing=0.0)
    labels = list(labels)
    order = list(range(len(labels)))
    random.Random(args.seed).shuffle(order)
    held = sorted(order[:args.held_out]); kept = sorted(order[args.held_out:])
    text = F.normalize(label_text_matrix(labels, device), dim=-1)
    anchors, anchor_profiles = text[kept], profiles[kept]
    queries, truth = text[held], profiles[held]
    axes = axis_names()

    print(f"annotations {args.version}: {len(labels)} labels, {len(held)} held out of the fit")
    print("chance per axis (majority value among anchors): "
          f"{sum(anchor_profiles[:, b].argmax(-1).bincount().max().item() / len(kept) for b in axis_slices()) / len(axes):.2f}")

    print("\nannotated label side (similarity-weighted average of anchors):")
    for temperature in args.temperatures:
        weight = torch.softmax(queries @ anchors.T / temperature, dim=-1)
        predicted = weight @ anchor_profiles
        mean, hits = per_axis_agreement(predicted, truth)
        print(f"  T={temperature:<5} agreement {mean:.2f}   " + " ".join(f"{a[:5]}={h:.2f}" for a, h in zip(axes, hits)))

    print("\nsentence-cosine label side (T7), same held-out labels:")
    head = PrimitiveSemanticHead(8, PrimitiveSemanticConfig(combiner="fixed"), values=primitive_value_matrix(device=device)).eval()
    with torch.no_grad():
        predicted = head.candidate_profile(queries.unsqueeze(0), torch.ones(1, len(held), dtype=torch.bool))[0]
    mean, hits = per_axis_agreement(predicted, truth)
    print(f"  agreement {mean:.2f}   " + " ".join(f"{a[:5]}={h:.2f}" for a, h in zip(axes, hits)))

    print("\nheld-out labels and their nearest anchors at T=0.05:")
    weight = torch.softmax(queries @ anchors.T / 0.05, dim=-1)
    for row, index in enumerate(held[:8]):
        top = weight[row].topk(3)
        print(f"  {labels[index]:40s} <- " + ", ".join(f"{labels[kept[j]]} ({w:.2f})" for w, j in zip(top.values.tolist(), top.indices.tolist())))


if __name__ == "__main__":
    main()
