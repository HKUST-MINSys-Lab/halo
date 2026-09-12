"""Validate the shared ExtraSensory raw archives without copying them.

The labelled and label-free adapters consume the same upstream archives.  Keeping a
second 7.5 GB copy under ``data/pretraining`` would waste disk and make provenance
ambiguous, so this module only validates an existing archive directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from data.datasets.extrasensory.fetch import FILES


REPO = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO / "data" / "datasets" / "extrasensory" / "downloads"
REQUIRED_ARCHIVES = ("raw_acc.zip", "watch_acc.zip", "labels.zip", "cv5Folds.zip")


def validate_archives(
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    expected_sizes: Mapping[str, int] | None = None,
) -> dict[str, Path]:
    """Return archives whose size and ZIP directory are valid.

    Opening the central directory catches truncation without inflating and hashing all
    7.5 GB on every conversion. The fetch path additionally checks published byte sizes.
    """
    paths = {name: raw_dir / name for name in REQUIRED_ARCHIVES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing shared ExtraSensory archives; fetch them with "
            "`python -m data.datasets.extrasensory.fetch` first: " + ", ".join(missing)
        )
    for name, path in paths.items():
        if expected_sizes is not None and path.stat().st_size != expected_sizes[name]:
            raise ValueError(
                f"{path}: expected {expected_sizes[name]:,} bytes, got {path.stat().st_size:,}"
            )
        try:
            with ZipFile(path) as archive:
                archive.infolist()
        except BadZipFile as exc:
            raise ValueError(f"{path}: invalid zip archive") from exc
    return paths


def _download_resumable(url: str, destination: Path, expected_size: int) -> None:
    """Download one archive with an atomic, resumable ``.part`` file."""
    if destination.is_file() and destination.stat().st_size == expected_size:
        print(f"[extrasensory_pretrain] present: {destination.name}")
        return
    if destination.exists():
        raise ValueError(
            f"{destination}: wrong size ({destination.stat().st_size:,}; expected {expected_size:,})"
        )
    part = destination.with_suffix(destination.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected_size:
        part.unlink()
        offset = 0
    headers = {"User-Agent": "HALO-dataset-fetch/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        partial_response = getattr(response, "status", None) == 206
        content_range = response.headers.get("Content-Range") if hasattr(response, "headers") else None
        if offset and partial_response and content_range is not None:
            expected_prefix = f"bytes {offset}-"
            if not content_range.startswith(expected_prefix):
                raise RuntimeError(
                    f"{url}: server resumed at the wrong byte: {content_range!r}, "
                    f"expected prefix {expected_prefix!r}"
                )
        if offset and not partial_response:
            # Some mirrors ignore Range. Restart rather than append a second full archive.
            offset = 0
        mode = "ab" if offset else "wb"
        with part.open(mode) as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    actual = part.stat().st_size
    if actual != expected_size:
        raise RuntimeError(
            f"{url}: incomplete download retained for resume: {actual:,}/{expected_size:,} bytes"
        )
    os.replace(part, destination)
    print(f"[extrasensory_pretrain] downloaded: {destination.name} ({actual:,} bytes)")


def fetch(raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, Path]:
    """Fetch missing archives into the one shared raw directory, then validate them."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ARCHIVES:
        url, size = FILES[name]
        _download_resumable(url, raw_dir / name, size)
    return validate_archives(
        raw_dir,
        expected_sizes={name: FILES[name][1] for name in REQUIRED_ARCHIVES},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="do not download; validate all shared archives and their published sizes",
    )
    args = parser.parse_args()
    expected = {name: FILES[name][1] for name in REQUIRED_ARCHIVES}
    paths = (
        validate_archives(args.raw_dir, expected_sizes=expected)
        if args.validate_only
        else fetch(args.raw_dir)
    )
    total = sum(path.stat().st_size for path in paths.values())
    print(f"[extrasensory_pretrain] validated {len(paths)} shared archives ({total:,} bytes)")
    print(f"[extrasensory_pretrain] raw directory: {args.raw_dir.resolve()}")


if __name__ == "__main__":
    main()
