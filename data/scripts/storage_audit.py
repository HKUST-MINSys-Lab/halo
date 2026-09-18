"""Inventory HALO data storage and identify mechanically reclaimable duplicates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

from halo.paths import DATASETS_DIR, REPO_ROOT, RUNS_DIR

CORE_DATASETS = DATASETS_DIR
APPLICATION_SOURCES = (
    REPO_ROOT / "applications" / "motion_monitoring" / "data" / "sources"
)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "quality" / "storage_inventory.json"
OUTPUT_ROOTS = (
    RUNS_DIR / "tokenizer",
    REPO_ROOT / "training" / "evidence" / "outputs",
    REPO_ROOT / "applications" / "motion_monitoring" / "artifacts",
)
LAYER_NAMES = ("downloads", "raw", "sessions", "grids", "processed")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for directory, names, filenames in os.walk(root, followlinks=False):
        names[:] = [name for name in names if name != "__pycache__"]
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            if not path.is_symlink() and path.suffix != ".part":
                yield path


def apparent_bytes(root: Path) -> int:
    """Return logical file bytes without following symlinks or partial downloads."""

    total = 0
    for path in _files(root):
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _entry(root: Path) -> dict[str, object]:
    return {
        "path": _display_path(root),
        "total_bytes": apparent_bytes(root),
        "layers": {
            name: apparent_bytes(root / name)
            for name in LAYER_NAMES
            if (root / name).exists()
        },
    }


def _entries(root: Path) -> dict[str, dict[str, object]]:
    if not root.exists():
        return {}
    return {
        path.name: _entry(path)
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name != "__pycache__"
    }


def _embedded_git_histories(roots: Iterable[Path]) -> list[dict[str, object]]:
    histories: list[dict[str, object]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob(".git"):
            if path.is_dir():
                histories.append(
                    {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "bytes": apparent_bytes(path),
                    }
                )
    return sorted(histories, key=lambda item: int(item["bytes"]), reverse=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_duplicates(
    roots: Iterable[Path], *, minimum_bytes: int
) -> list[dict[str, object]]:
    if minimum_bytes <= 0:
        return []
    by_size: dict[int, list[Path]] = defaultdict(list)
    for root in roots:
        for path in _files(root):
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            if size >= minimum_bytes:
                by_size[size].append(path)

    by_digest: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        for path in paths:
            by_digest[(size, _sha256(path))].append(path)

    groups: list[dict[str, object]] = []
    for (size, digest), paths in by_digest.items():
        if len(paths) < 2:
            continue
        inodes = {(path.stat().st_dev, path.stat().st_ino) for path in paths}
        groups.append(
            {
                "bytes_per_copy": size,
                "copies": len(paths),
                "distinct_inodes": len(inodes),
                "reclaimable_bytes_if_one_copy_retained": max(0, len(inodes) - 1)
                * size,
                "sha256": digest,
                "paths": [_display_path(path) for path in paths],
            }
        )
    return sorted(
        groups,
        key=lambda item: int(item["reclaimable_bytes_if_one_copy_retained"]),
        reverse=True,
    )


def build_inventory(*, hash_minimum_bytes: int) -> dict[str, object]:
    usage = shutil.disk_usage(REPO_ROOT)
    scan_roots = (CORE_DATASETS, APPLICATION_SOURCES)
    core_datasets = _entries(CORE_DATASETS)
    application_sources = _entries(APPLICATION_SOURCES)
    output_roots = {
        _display_path(path): apparent_bytes(path)
        for path in OUTPUT_ROOTS
        if path.exists()
    }
    embedded_git_histories = _embedded_git_histories(scan_roots)
    exact_duplicate_groups = _exact_duplicates(
        scan_roots, minimum_bytes=hash_minimum_bytes
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filesystem": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
        "summary": {
            "core_dataset_bytes": sum(
                int(entry["total_bytes"]) for entry in core_datasets.values()
            ),
            "application_source_bytes": sum(
                int(entry["total_bytes"]) for entry in application_sources.values()
            ),
            "tracked_output_bytes": sum(output_roots.values()),
            "embedded_git_history_bytes": sum(
                int(entry["bytes"]) for entry in embedded_git_histories
            ),
            "exact_duplicate_reclaimable_bytes": sum(
                int(entry["reclaimable_bytes_if_one_copy_retained"])
                for entry in exact_duplicate_groups
            ),
        },
        "core_datasets": core_datasets,
        "application_sources": application_sources,
        "output_roots": output_roots,
        "embedded_git_histories": embedded_git_histories,
        "exact_duplicate_groups": exact_duplicate_groups,
        "hash_minimum_bytes": hash_minimum_bytes,
        "notes": [
            "Sizes are apparent file bytes; symlinks, __pycache__, and .part files are excluded.",
            "An exact duplicate is not automatically safe to delete; logical path contracts still apply.",
            "Downloaded archives and derived sessions/grids are reported separately because they have different reproducibility roles.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--hash-min-mib",
        type=float,
        default=20.0,
        help="hash same-size files at least this large; zero disables duplicate hashing",
    )
    args = parser.parse_args()
    if args.hash_min_mib < 0:
        parser.error("--hash-min-mib must be non-negative")
    inventory = build_inventory(
        hash_minimum_bytes=int(args.hash_min_mib * 1024 * 1024)
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
