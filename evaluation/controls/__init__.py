"""Registered controls for rungs 1 and 2. Each is a pure function over row indices and labels so a
control run differs from the main run in exactly one declared way (docs/overview/roadmap.md,
"Registered threats and their matched controls")."""

from evaluation.controls.balanced_pool import balanced_pool_filter
from evaluation.controls.disjoint_classes import disjoint_class_split

CONTROLS = ("none", "balanced_pool", "disjoint_classes")

__all__ = ["CONTROLS", "balanced_pool_filter", "disjoint_class_split"]
