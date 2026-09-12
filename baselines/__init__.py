"""External encoder adapters for the current support-conditioned HAR comparison.

Each baseline lives in its own subfolder ``baselines/<name>/`` with an ``adapter.py``
that subclasses a tier in :mod:`baselines.base` and is decorated with ``@register``.
Dropping such a subfolder in **auto-registers** it here — no edit to this file needed.
Historical native/ConSE scoring methods remain available in the adapter modules for reproduction,
but they are not the primary comparison. See ``docs/baselines/BASELINES.md`` for the live roster and
verified input contracts.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .base import (  # noqa: F401
    REGISTRY, register, BaselineAdapter, ConSEAdapter, CosineAdapter, InputContract,
    UnsupportedEvaluationCell, load_gt, score, global_labels,
)

# Auto-import every baselines/<name>/adapter.py for its @register side effects.
_pkg_dir = Path(__file__).resolve().parent
for _sub in sorted(p for p in _pkg_dir.iterdir() if p.is_dir() and (p / "adapter.py").exists()):
    importlib.import_module(f"{__name__}.{_sub.name}.adapter")

__all__ = ["REGISTRY", "register", "BaselineAdapter", "ConSEAdapter", "CosineAdapter",
           "InputContract", "UnsupportedEvaluationCell", "load_gt", "score", "global_labels"]
