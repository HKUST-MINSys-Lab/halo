"""Prepare MobiAct v2 for prospective evaluation.

The setup is intentionally explicit. It accepts either an official archive supplied by the user or
an already-extracted raw directory; it never downloads a third-party mirror. It then converts the
release, materializes native 4/8/16-second grids, writes a duration-specific capacity ledger, and
refreshes the required duplicate/implausible quality screens.

Examples
--------
``python3 -m data.datasets.mobiact.setup --archive /path/MobiAct_Dataset_v2.0.zip``
``python3 -m data.datasets.mobiact.setup --raw-root data/datasets/mobiact/downloads/extracted``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

from .convert import HERE, PROTOCOL_VERSION, convert

DEFAULT_WINDOWS = (4.0, 8.0, 16.0)
RAW_ROOT = HERE / "downloads"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_destination(root: Path, name: str) -> Path:
    destination = (root / name).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ValueError(f"archive entry escapes destination: {name!r}")
    return destination


def extract_archive(archive: Path, destination: Path) -> Path:
    """Extract one trusted archive without permitting path traversal or stale-file mixing."""
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"MobiAct archive does not exist: {archive}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive) as payload:
            for member in payload.infolist():
                _safe_destination(destination, member.filename)
            payload.extractall(destination)
    elif suffixes.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(archive) as payload:
            for member in payload.getmembers():
                if member.issym() or member.islnk():
                    raise ValueError(f"refusing link-bearing archive member: {member.name!r}")
                _safe_destination(destination, member.name)
            payload.extractall(destination, filter="data")
    else:
        raise ValueError(f"unsupported MobiAct archive type: {archive.name}")
    return destination


def _grid_meta(window_seconds: float) -> dict:
    token = f"w{float(window_seconds):g}".replace(".", "p")
    path = HERE / "grids" / "native" / "phone_trouser_pocket" / token / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"MobiAct grid was not created: {path}")
    return json.loads(path.read_text())


def finalize_protocol(window_seconds: tuple[float, ...]) -> dict:
    """Freeze labels available on both subject partitions at each evidence budget."""
    protocol_path = HERE / "eval_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    reference = set(protocol["reference_subjects"])
    query = set(protocol["query_subjects"])
    candidate_labels: dict[str, list[str]] = {}
    capacity: dict[str, dict] = {}
    for seconds in sorted(set(map(float, window_seconds))):
        meta = _grid_meta(seconds)
        labels = np.asarray(meta["labels"], dtype=object)
        subjects = np.asarray(meta["subjects"], dtype=object)
        present = sorted(set(labels))
        by_label: dict[str, dict] = {}
        eligible: list[str] = []
        for label in present:
            label_rows = labels == label
            reference_rows = int(np.count_nonzero(label_rows & np.isin(subjects, list(reference))))
            query_rows = int(np.count_nonzero(label_rows & np.isin(subjects, list(query))))
            by_label[str(label)] = {
                "reference_windows": reference_rows,
                "query_windows": query_rows,
                "eligible": bool(reference_rows and query_rows),
            }
            if reference_rows and query_rows:
                eligible.append(str(label))
        if len(eligible) < 2:
            raise ValueError(
                f"MobiAct {seconds:g}s has only {len(eligible)} labels across both frozen partitions"
            )
        key = f"{seconds:g}"
        candidate_labels[key] = eligible
        capacity[key] = {"n_windows": len(labels), "labels": by_label}
    protocol["candidate_labels_by_window_seconds"] = candidate_labels
    protocol["capacity"] = capacity
    protocol["status"] = "ready"
    protocol["protocol_version"] = PROTOCOL_VERSION
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (HERE / "eval_labels.json").write_text(json.dumps({
        "dataset": "mobiact", "protocol_version": PROTOCOL_VERSION,
        "source": "Native MobiAct v2 activity names; duration panels are frozen in eval_protocol.json.",
        "labels": protocol["candidate_labels"],
    }, indent=2, sort_keys=True) + "\n")
    return protocol


def _quality_screen(window_seconds: tuple[float, ...]) -> None:
    """Refresh only MobiAct entries while preserving quality screens for other datasets."""
    for seconds in sorted(set(map(float, window_seconds))):
        for module in ("data.scripts.scan_duplicates", "data.scripts.scan_implausible"):
            subprocess.run(
                [sys.executable, "-m", module, "--alignment", "native", "--datasets", "mobiact",
                 "--window-seconds", f"{seconds:g}"],
                check=True,
            )


def prepare(
    *, archive: Path | None, raw_root: Path | None, windows: tuple[float, ...],
    split_seed: int, query_fraction: float,
) -> dict:
    if archive is not None and raw_root is not None:
        raise ValueError("provide either --archive or --raw-root, not both")
    if archive is not None:
        raw_root = extract_archive(archive, RAW_ROOT / "extracted")
        archive_hash = _sha256(archive)
    elif raw_root is not None:
        raw_root = raw_root.resolve()
        archive_hash = None
    else:
        raise ValueError("one of --archive or --raw-root is required")
    if not windows or any(seconds <= 0 for seconds in windows):
        raise ValueError("window durations must be positive")

    convert(raw_root, split_seed=split_seed, query_fraction=query_fraction, archive_sha256=archive_hash)
    grids = HERE / "grids"
    if grids.exists():
        # Generated artifacts only; rebuilding prevents an old unqualified phantom grid from being
        # discovered alongside the duration-qualified contract.
        shutil.rmtree(grids)
    from data.scripts.build_grids import build
    for seconds in sorted(set(map(float, windows))):
        build(datasets=("mobiact",), alignments=("native",), window_seconds=seconds)
    protocol = finalize_protocol(windows)
    _quality_screen(windows)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--raw-root", type=Path)
    parser.add_argument("--window-seconds", type=float, nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--split-seed", type=int, default=20260918)
    parser.add_argument("--query-fraction", type=float, default=0.30)
    args = parser.parse_args()
    protocol = prepare(
        archive=args.archive, raw_root=args.raw_root, windows=tuple(args.window_seconds),
        split_seed=args.split_seed, query_fraction=args.query_fraction,
    )
    print(
        f"[mobiact] ready: {len(protocol['reference_subjects'])} reference / "
        f"{len(protocol['query_subjects'])} query subjects",
        flush=True,
    )


if __name__ == "__main__":
    main()
