"""Repository and runtime paths.

No runtime module should infer layout from its own ``__file__`` depth.  Every location can be
redirected without editing source, which keeps downloaded corpora and large caches outside a
checkout when desired while preserving the historical in-tree defaults.
"""

from __future__ import annotations

import os
from pathlib import Path


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


REPO_ROOT = _path("HALO_REPO_ROOT", Path(__file__).resolve().parents[1])
DATA_DIR = _path("HALO_DATA_DIR", REPO_ROOT / "data")
DATASETS_DIR = _path("HALO_DATASETS_DIR", DATA_DIR / "datasets")
PRETRAINING_DIR = _path("HALO_PRETRAINING_DIR", DATA_DIR / "pretraining")
RUNS_DIR = _path("HALO_RUNS_DIR", REPO_ROOT / "runs")
CACHE_DIR = _path("HALO_CACHE_DIR", REPO_ROOT / "cache")
RESULTS_DIR = _path("HALO_RESULTS_DIR", REPO_ROOT / "results")
REFERENCES_DIR = _path("HALO_REFERENCES_DIR", REPO_ROOT / "references")

