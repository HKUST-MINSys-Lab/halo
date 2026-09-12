"""Validate the already-local public Capture-24 source archive.

Capture-24 is shared with ``data/datasets/capture24``. This adapter intentionally does not
download a second copy: its fetch stage validates the official raw participant files that are
already present, then its converter writes a source-isolated, label-free view under
``data/pretraining/capture24_pretrain``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO / "data" / "datasets" / "capture24" / "downloads" / "capture24"
EXPECTED_PARTICIPANTS = 151


def validate_source(source_root: Path = SOURCE_ROOT) -> tuple[Path, ...]:
    """Return the complete ordered participant-file set or fail without downloading."""
    participants = tuple(sorted(source_root.glob("P*.csv.gz")))
    if len(participants) != EXPECTED_PARTICIPANTS:
        raise FileNotFoundError(
            "Capture-24 raw source is incomplete. Expected 151 P*.csv.gz files under "
            f"{source_root}; found {len(participants)}. Restore the existing Capture-24 archive "
            "before converting."
        )
    return participants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    participants = validate_source(args.source_root)
    print(f"Capture-24 source ready: {len(participants)} participant files at {args.source_root}")


if __name__ == "__main__":
    main()
