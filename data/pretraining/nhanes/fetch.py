"""Download a deterministic, bounded subset of public NHANES PAX80_G archives.

The complete 2011-2012 release is approximately 1.04 TB compressed.  This
fetcher therefore has no "download everything" mode: callers must choose a
positive subject count or explicit SEQN identifiers. Selection is deterministic
for a seed and does not favor unusually small/short recordings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"
INDEX_URL = "https://ftp.cdc.gov/pub/pax_g/"
DEFAULT_SEED = 20260726


class _ArchiveLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.subjects: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        name = Path(urlparse(href).path).name
        match = re.fullmatch(r"(\d+)\.tar\.bz2", name, re.I)
        if match:
            self.subjects.add(match.group(1))


def available_subjects() -> list[str]:
    request = urllib.request.Request(INDEX_URL, headers={"User-Agent": "HALO-dataset-fetch/1.0"})
    with urllib.request.urlopen(request) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = _ArchiveLinkParser()
    parser.feed(html)
    subjects = sorted(parser.subjects)
    if len(subjects) < 6_000:
        raise RuntimeError(
            f"NHANES index parse returned only {len(subjects)} subjects; upstream layout changed"
        )
    return subjects


def select_subjects(available: list[str], count: int, seed: int) -> list[str]:
    if count <= 0:
        raise ValueError("--subjects must be positive; full-corpus downloads are intentionally blocked")
    ordered = sorted(
        available,
        key=lambda seqn: hashlib.sha256(f"{seed}:{seqn}".encode()).hexdigest(),
    )
    return sorted(ordered[: min(count, len(ordered))])


def _remote_size(url: str) -> int:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "HALO-dataset-fetch/1.0"}
    )
    with urllib.request.urlopen(request) as response:
        return int(response.headers["Content-Length"])


def _download(seqn: str) -> dict:
    url = f"{INDEX_URL}{seqn}.tar.bz2"
    destination = DOWNLOADS / f"{seqn}.tar.bz2"
    expected = _remote_size(url)
    if destination.exists() and destination.stat().st_size == expected:
        print(f"[nhanes] present {seqn}: {expected / 1e6:.1f} MB")
        return {"seqn": seqn, "url": url, "bytes": expected}
    tmp = destination.with_suffix(".tar.bz2.part")
    request = urllib.request.Request(url, headers={"User-Agent": "HALO-dataset-fetch/1.0"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    if tmp.stat().st_size != expected:
        raise RuntimeError(
            f"{seqn}: incomplete download: expected {expected}, got {tmp.stat().st_size}"
        )
    os.replace(tmp, destination)
    print(f"[nhanes] downloaded {seqn}: {expected / 1e6:.1f} MB")
    return {"seqn": seqn, "url": url, "bytes": expected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subjects", type=int, help="deterministic number of participants")
    group.add_argument("--ids", nargs="+", help="explicit NHANES SEQN identifiers")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    available = available_subjects()
    if args.ids:
        unknown = sorted(set(args.ids) - set(available))
        if unknown:
            raise ValueError(f"SEQN identifiers not present in PAX80_G: {unknown}")
        selected = sorted(set(args.ids))
    else:
        selected = select_subjects(available, args.subjects, args.seed)

    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    records = [_download(seqn) for seqn in selected]
    (DOWNLOADS / "subset_manifest.json").write_text(
        json.dumps(
            {
                "source": INDEX_URL,
                "cycle": "NHANES 2011-2012 PAX80_G",
                "selection_seed": None if args.ids else args.seed,
                "subjects": records,
                "total_bytes": sum(record["bytes"] for record in records),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[nhanes] ready: {len(records)} participants, "
        f"{sum(record['bytes'] for record in records) / 1e9:.2f} GB compressed"
    )


if __name__ == "__main__":
    main()
