"""Fetch a deterministic, bounded, IMU-only subset of Meta's Nymeria / NymeriaPlus.

Nymeria (ECCV 2024, arXiv 2406.09905) and NymeriaPlus (arXiv 2603.18496) are
~1,200 sequences / 300 h / 264 participants / 50 locations.  The FULL release is
approximately **80 TB**, so this fetcher has no "download everything" mode:
callers must choose a positive sequence count, and a hard `--max-gb` budget is
enforced against the filtered plan *before* a single byte moves.

Access is licence-gated (CC BY-NC 4.0) and this module deliberately does not try
to work around that.  The user must:

  1. accept the licence at https://www.projectaria.com/datasets/nymeria/
  2. tick the download groups ``body_raw`` and ``timesync_and_imu`` (only those)
  3. receive the URL manifest by email and save it as ``downloads/url.json``

Only two of the release's download groups are ever requested here (group->file
layout verified against ``nymeriaplus/layout.py`` in
github.com/facebookresearch/nymeria_dataset):

  ``body_raw``          ``body/xdata.healthcheck``, ``body/xdata.mvnx``, ``body/xdata.npz``
                        -- the Xsens MVN Link 17-tracker suit stream at 240 Hz.
  ``timesync_and_imu``  ``recording_{head,lwrist,rwrist,observer}/data/motion.vrs``
                        -- IMU-only VRS containers.  **No image streams**, which is
                        what keeps this a ~tens-of-GB pull instead of 80 TB.

Two download paths, both resumable and byte-identical in their result:

  * **preferred** -- shell out to the official ``nymeriaplus-download`` CLI, but
    hand it a *filtered* url.json containing only the selected sequences and
    groups.  A user therefore cannot pull the whole release by accident even if
    they pass a full manifest.
  * **fallback**  -- if the CLI is absent, download the filtered URL list with the
    stdlib.  Resume is driven primarily by on-disk destination size (which is
    independent of any log-file naming convention) and secondarily by our own
    completion markers under ``<out>/.download_logs/``.

Usage::

    python -m data.pretraining.nymeria.fetch --sequences 40 --max-gb 120
    python -m data.pretraining.nymeria.fetch --url-json /path/url.json --dry-run
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"
DEFAULT_URL_JSON = DOWNLOADS / "url.json"
DEFAULT_SEED = 20260909
DEFAULT_MAX_GB = 120.0
#: Bounded default used when neither --sequences nor --sequence-ids nor --all-sequences is
#: given, so a bare `python -m ...fetch` (which is how data.pretraining.build_corpus invokes
#: it) is safe rather than an argparse error or an 80 TB pull.
DEFAULT_SEQUENCES = 40
LICENCE_URL = "https://www.projectaria.com/datasets/nymeria/"
DOWNLOADER_CLI = "nymeriaplus-download"
LOG_DIR_NAME = ".download_logs"

#: The only two groups this module ever requests.
BODY_GROUP = "body_raw"
IMU_GROUP = "timesync_and_imu"
DEFAULT_GROUPS = (BODY_GROUP, IMU_GROUP)

#: Per-group file layout, verified against ``nymeriaplus/layout.py``.  Used by the
#: stdlib fallback to place a downloaded file at the same relative path the official
#: CLI would use, when the url.json record carries only a bare basename.
GROUP_LAYOUT: Mapping[str, tuple[str, ...]] = {
    BODY_GROUP: (
        "body/xdata.healthcheck",
        "body/xdata.mvnx",
        "body/xdata.npz",
    ),
    IMU_GROUP: (
        "recording_head/data/motion.vrs",
        "recording_lwrist/data/motion.vrs",
        "recording_rwrist/data/motion.vrs",
        "recording_observer/data/motion.vrs",
    ),
}

# url.json record field names.  The manifest is issued per-user by Meta and its exact
# key spelling is NOT public, so every accessor below is tolerant and the first match
# in each tuple wins.  `parse_catalog` raises with a shape dump rather than guessing
# when none of these are found.
_URL_KEYS = ("download_url", "url", "cdn_url", "signed_url", "href")
_SIZE_KEYS = ("file_size_bytes", "size_bytes", "filesize", "file_size", "size", "bytes")
_NAME_KEYS = ("filename", "file_name", "name", "path", "relative_path")
_SHA1_KEYS = ("sha1sum", "sha1", "checksum")
_USER_AGENT = "HALO-dataset-fetch/1.0"

_MISSING_URL_JSON = f"""\
Nymeria is licence-gated (CC BY-NC 4.0) and cannot be fetched without your own
URL manifest.  No url.json was found at: {{path}}

Do this once:
  1. Open {LICENCE_URL} and accept the CC BY-NC 4.0 licence
     (email registration; turnaround is typically ~48 h).
  2. In the Aria Dataset Explorer, select the sequences you want (or all of
     them -- this fetcher filters them down locally before downloading).
  3. When choosing download groups, tick EXACTLY these two and nothing else:
         {BODY_GROUP}          (Xsens 17-IMU suit: body/xdata.{{healthcheck,mvnx,npz}})
         {IMU_GROUP}   (IMU-only VRS: recording_*/data/motion.vrs)
     Do NOT tick any group containing image/video streams; the full release is
     ~80 TB and the video groups are essentially all of it.
  4. Save the emailed manifest to {{path}} (or pass --url-json PATH).

Then re-run, e.g.:
     python -m data.pretraining.nymeria.fetch --sequences 40 --max-gb 120
"""


class LicenceError(RuntimeError):
    """The user has not supplied their own licence-gated url.json."""


class BudgetExceeded(RuntimeError):
    """The filtered download plan is larger than the caller's --max-gb budget."""


class CatalogShapeError(RuntimeError):
    """The url.json did not have a shape this module recognises."""


@dataclass(frozen=True)
class FileRecord:
    """One downloadable file inside one (sequence, group)."""

    sequence: str
    group: str
    url: str
    filename: str
    size_bytes: int | None
    sha1: str | None

    @property
    def relative_path(self) -> str:
        """Destination path relative to the download root, mirroring the official CLI."""
        name = self.filename.replace("\\", "/").lstrip("/")
        if "/" in name:
            return f"{self.sequence}/{name}"
        for candidate in GROUP_LAYOUT.get(self.group, ()):
            if candidate.rsplit("/", 1)[-1] == name:
                return f"{self.sequence}/{candidate}"
        # Unknown basename for this group: keep it, namespaced, rather than guessing
        # a layout slot.  `convert.py` reports what it could not find.
        return f"{self.sequence}/{self.group}/{name}"


@dataclass
class Plan:
    """The filtered set of files this run intends to fetch."""

    records: list[FileRecord] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)
    groups: tuple[str, ...] = ()

    @property
    def known_bytes(self) -> int:
        return sum(r.size_bytes or 0 for r in self.records)

    @property
    def unknown_size_count(self) -> int:
        return sum(1 for r in self.records if r.size_bytes is None)


# --------------------------------------------------------------------------------------
# url.json parsing
# --------------------------------------------------------------------------------------


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _coerce_size(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _records_from_node(sequence: str, group: str, node: Any) -> list[FileRecord]:
    """Flatten one group's payload into FileRecords.

    Accepts a single record dict, a list of record dicts, or a mapping of
    name -> record dict.  Anything without a recognisable URL field is skipped.
    """
    if isinstance(node, Mapping):
        url = _first(node, _URL_KEYS)
        if isinstance(url, str):
            name = _first(node, _NAME_KEYS)
            return [
                FileRecord(
                    sequence=sequence,
                    group=group,
                    url=url,
                    filename=str(name) if name else url.split("?", 1)[0].rsplit("/", 1)[-1],
                    size_bytes=_coerce_size(_first(node, _SIZE_KEYS)),
                    sha1=(lambda s: str(s) if s else None)(_first(node, _SHA1_KEYS)),
                )
            ]
        out: list[FileRecord] = []
        for key, child in node.items():
            for record in _records_from_node(sequence, group, child):
                # A mapping key is a better filename than a URL basename.
                if record.filename == record.url.split("?", 1)[0].rsplit("/", 1)[-1]:
                    record = FileRecord(
                        sequence, group, record.url, str(key), record.size_bytes, record.sha1
                    )
                out.append(record)
        return out
    if isinstance(node, list):
        out = []
        for child in node:
            out.extend(_records_from_node(sequence, group, child))
        return out
    if isinstance(node, str) and node.startswith(("http://", "https://")):
        return [
            FileRecord(
                sequence, group, node, node.split("?", 1)[0].rsplit("/", 1)[-1], None, None
            )
        ]
    return []


def locate_container(raw: Any, groups: Iterable[str] = DEFAULT_GROUPS) -> dict:
    """Return the mutable ``{sequence: {group: ...}}`` mapping inside ``raw``.

    The returned object is a *reference into* ``raw``, so pruning it prunes the
    document, which is how :func:`write_filtered_url_json` preserves whatever
    envelope the official CLI expects.
    """
    wanted = set(groups)

    def looks_like_container(node: Any) -> bool:
        if not isinstance(node, Mapping) or not node:
            return False
        for value in node.values():
            if isinstance(value, Mapping) and wanted & set(value.keys()):
                return True
        return False

    if not isinstance(raw, Mapping):
        raise CatalogShapeError(
            f"url.json top level is {type(raw).__name__}, expected an object"
        )
    for key in ("sequences", "sequence", "data", "items"):
        node = raw.get(key)
        if looks_like_container(node):
            return node  # type: ignore[return-value]
    if looks_like_container(raw):
        return dict(raw) if not isinstance(raw, dict) else raw
    # One more level of envelope (e.g. {"nymeria": {"sequences": {...}}}).
    for value in raw.values():
        if isinstance(value, Mapping):
            try:
                return locate_container(value, wanted)
            except CatalogShapeError:
                continue
    raise CatalogShapeError(
        "could not find a {sequence: {group: ...}} mapping containing any of "
        f"{sorted(wanted)} in url.json. Top-level keys: {sorted(map(str, raw.keys()))[:20]}. "
        "Either the manifest was issued without those download groups (re-request it "
        f"with {BODY_GROUP} and {IMU_GROUP} ticked), or the manifest schema changed."
    )


def parse_catalog(raw: Any, groups: Iterable[str] = DEFAULT_GROUPS) -> dict[str, dict[str, list[FileRecord]]]:
    """Normalise a url.json document into ``{sequence: {group: [FileRecord, ...]}}``."""
    container = locate_container(raw, groups)
    catalog: dict[str, dict[str, list[FileRecord]]] = {}
    for sequence, payload in container.items():
        if not isinstance(payload, Mapping):
            continue
        per_group: dict[str, list[FileRecord]] = {}
        for group, node in payload.items():
            records = _records_from_node(str(sequence), str(group), node)
            if records:
                per_group[str(group)] = records
        if per_group:
            catalog[str(sequence)] = per_group
    if not catalog:
        raise CatalogShapeError("url.json contained no downloadable records")
    return catalog


def load_url_json(path: Path) -> Any:
    if not path.exists():
        # str.replace, not str.format: the message contains literal brace text
        # ("body/xdata.{healthcheck,mvnx,npz}") that format() would read as a field.
        raise LicenceError(_MISSING_URL_JSON.replace("{path}", str(path)))
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CatalogShapeError(f"{path} is not valid JSON: {exc}") from exc


# --------------------------------------------------------------------------------------
# selection + budget
# --------------------------------------------------------------------------------------


def select_sequences(available: Sequence[str], count: int, seed: int) -> list[str]:
    """Deterministically choose ``count`` sequence uids (same scheme as nhanes/fetch.py)."""
    if count <= 0:
        raise ValueError(
            "--sequences must be positive; full-release downloads are intentionally "
            "blocked (Nymeria is ~80 TB)"
        )
    ordered = sorted(
        sorted(set(available)),
        key=lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest(),
    )
    return sorted(ordered[: min(count, len(ordered))])


def build_plan(
    catalog: Mapping[str, Mapping[str, list[FileRecord]]],
    sequences: Sequence[str],
    groups: Sequence[str],
) -> Plan:
    plan = Plan(groups=tuple(groups), sequences=sorted(sequences))
    unknown = sorted(set(sequences) - set(catalog))
    if unknown:
        raise ValueError(f"sequences not present in url.json: {unknown[:10]}")
    for sequence in plan.sequences:
        for group in groups:
            for record in catalog[sequence].get(group, ()):
                plan.records.append(record)
    if not plan.records:
        raise ValueError(
            f"the selected sequences carry none of the groups {list(groups)}; "
            "re-request the manifest with those groups ticked"
        )
    return plan


def check_budget(plan: Plan, max_gb: float, allow_unknown_size: bool = False) -> None:
    """Refuse a plan larger than ``max_gb``, or one whose size cannot be established."""
    if plan.unknown_size_count and not allow_unknown_size:
        raise BudgetExceeded(
            f"{plan.unknown_size_count}/{len(plan.records)} files in the filtered plan "
            "declare no size, so the --max-gb budget cannot be enforced. Re-run with "
            "--allow-unknown-size only if you accept an unbounded download."
        )
    gb = plan.known_bytes / 1e9
    if gb > max_gb:
        raise BudgetExceeded(
            f"filtered plan is {gb:.1f} GB across {len(plan.records)} files "
            f"({len(plan.sequences)} sequences x {list(plan.groups)}), over the "
            f"--max-gb budget of {max_gb:.1f} GB. Lower --sequences or raise --max-gb."
        )


def write_filtered_url_json(raw: Any, plan: Plan, destination: Path) -> Path:
    """Write a copy of ``raw`` pruned to exactly ``plan``'s sequences and groups.

    This is the safety property that makes the official CLI usable: it only ever
    sees the sequences and groups we selected, so no invocation of it can pull the
    full 80 TB release.
    """
    pruned = copy.deepcopy(raw)
    container = locate_container(pruned, plan.groups)
    keep_sequences = set(plan.sequences)
    keep_groups = set(plan.groups)
    for sequence in list(container.keys()):
        if str(sequence) not in keep_sequences:
            del container[sequence]
            continue
        payload = container[sequence]
        if isinstance(payload, dict):
            for group in list(payload.keys()):
                if str(group) not in keep_groups:
                    del payload[group]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(pruned, indent=2) + "\n")
    return destination


# --------------------------------------------------------------------------------------
# downloading
# --------------------------------------------------------------------------------------


def marker_path(out_dir: Path, record: FileRecord) -> Path:
    safe = record.relative_path.replace("/", "__")
    return out_dir / LOG_DIR_NAME / f"{safe}.json"


def is_complete(out_dir: Path, record: FileRecord) -> bool:
    """Resume check.

    On-disk size is the primary signal because it does not depend on any log-file
    naming convention (ours or the official CLI's).  The marker is a secondary
    signal used when the record declares no size.
    """
    destination = out_dir / record.relative_path
    if record.size_bytes is not None:
        return destination.exists() and destination.stat().st_size == record.size_bytes
    return destination.exists() and marker_path(out_dir, record).exists()


def _write_marker(out_dir: Path, record: FileRecord, size: int) -> None:
    path = marker_path(out_dir, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sequence": record.sequence,
                "group": record.group,
                "relative_path": record.relative_path,
                "bytes": size,
                "sha1": record.sha1,
                "writer": "halo.data.pretraining.nymeria.fetch",
            },
            indent=2,
        )
        + "\n"
    )


def _extract_archive(out_dir: Path, record: FileRecord, archive_path: Path) -> None:
    """Materialise a group archive in the layout produced by the official downloader."""
    install_root = out_dir / record.sequence
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = Path(member.filename)
            if member.is_dir() or name.is_absolute() or ".." in name.parts:
                if not member.is_dir():
                    raise RuntimeError(f"unsafe archive member {member.filename!r}")
                continue
            parts = name.parts
            if parts and parts[0] == record.sequence:
                parts = parts[1:]
            if not parts:
                continue
            target = install_root.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def download_record(out_dir: Path, record: FileRecord) -> int:
    """Stdlib download of one file with atomic replace and size/sha1 verification."""
    destination = out_dir / record.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_complete(out_dir, record):
        size = destination.stat().st_size
        if zipfile.is_zipfile(destination):
            # An older interrupted fallback run can have a verified archive without its
            # extracted payload. Re-materialise it idempotently before declaring success.
            _extract_archive(out_dir, record, destination)
        print(f"[nymeria] present {record.relative_path} ({size / 1e6:.1f} MB)")
        _write_marker(out_dir, record, size)
        return size
    tmp = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(record.url, headers={"User-Agent": _USER_AGENT})
    digest = hashlib.sha1()
    size = 0
    with urllib.request.urlopen(request) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if record.size_bytes is not None and size != record.size_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{record.relative_path}: incomplete download, expected {record.size_bytes} got {size}"
        )
    if record.sha1 and digest.hexdigest() != record.sha1.lower():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{record.relative_path}: sha1 mismatch")
    os.replace(tmp, destination)
    if zipfile.is_zipfile(destination):
        # The official downloader expands group archives. The fallback must leave the same layout
        # or convert.py will see a successful download but no body/xdata.mvnx or motion.vrs files.
        _extract_archive(out_dir, record, destination)
    _write_marker(out_dir, record, size)
    print(f"[nymeria] downloaded {record.relative_path} ({size / 1e6:.1f} MB)")
    return size


def official_cli() -> str | None:
    """Path to ``nymeriaplus-download`` if it is on PATH, else None."""
    return shutil.which(DOWNLOADER_CLI)


def run_official(filtered_json: Path, out_dir: Path, workers: int) -> None:
    cli = official_cli()
    if cli is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError(f"{DOWNLOADER_CLI} not on PATH")
    command = [cli, "-i", str(filtered_json), "-o", str(out_dir), "-y", "-n", str(workers)]
    print(f"[nymeria] $ {' '.join(command)}")
    subprocess.run(command, check=True)


def fetch(
    *,
    url_json: Path = DEFAULT_URL_JSON,
    out_dir: Path = DOWNLOADS,
    sequences: int | None = None,
    sequence_ids: Sequence[str] | None = None,
    all_sequences: bool = False,
    groups: Sequence[str] = DEFAULT_GROUPS,
    seed: int = DEFAULT_SEED,
    max_gb: float = DEFAULT_MAX_GB,
    workers: int = 4,
    allow_unknown_size: bool = False,
    use_official: bool = True,
    dry_run: bool = False,
) -> dict:
    raw = load_url_json(url_json)
    catalog = parse_catalog(raw, groups)
    available = sorted(catalog)
    if sequence_ids:
        selected = sorted(set(sequence_ids))
        seed_used = None
    elif all_sequences:
        # The url.json is itself the user's explicit selection on the Aria site, so "all of it"
        # is a real intent. --max-gb is still the rail that stops an 80 TB accident.
        selected = available
        seed_used = None
        print(f"[nymeria] --all-sequences: taking every sequence in the manifest ({len(available)})")
    else:
        if sequences is None:
            sequences = DEFAULT_SEQUENCES
            print(
                f"[nymeria] no --sequences given; using the bounded default of "
                f"{DEFAULT_SEQUENCES}. Pass --sequences N, or --all-sequences to take every "
                "sequence in your manifest."
            )
        selected = select_sequences(available, sequences, seed)
        seed_used = seed
    plan = build_plan(catalog, selected, groups)
    check_budget(plan, max_gb, allow_unknown_size)

    print(
        f"[nymeria] plan: {len(plan.sequences)}/{len(available)} sequences, "
        f"groups={list(plan.groups)}, {len(plan.records)} files, "
        f"{plan.known_bytes / 1e9:.2f} GB "
        f"({plan.unknown_size_count} unsized) -- budget {max_gb:.0f} GB"
    )
    if dry_run:
        print("[nymeria] --dry-run: nothing downloaded")
        return _manifest(plan, seed_used, url_json, out_dir, path="dry-run", fetched=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LOG_DIR_NAME).mkdir(exist_ok=True)
    filtered = out_dir / "url.filtered.json"
    write_filtered_url_json(raw, plan, filtered)

    fetched = 0
    if use_official and official_cli() is not None:
        run_official(filtered, out_dir, workers)
        path = "official-cli"
        for record in plan.records:
            destination = out_dir / record.relative_path
            if destination.exists():
                size = destination.stat().st_size
                fetched += size
                _write_marker(out_dir, record, size)
    else:
        path = "stdlib-fallback"
        if use_official:
            print(
                f"[nymeria] {DOWNLOADER_CLI} not on PATH "
                "(pip install git+https://github.com/facebookresearch/nymeria_dataset); "
                "using the stdlib fallback"
            )
        for record in plan.records:
            fetched += download_record(out_dir, record)

    manifest = _manifest(plan, seed_used, url_json, out_dir, path=path, fetched=fetched)
    (out_dir / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[nymeria] ready: {fetched / 1e9:.2f} GB under {out_dir}")
    return manifest


def _manifest(
    plan: Plan, seed: int | None, url_json: Path, out_dir: Path, *, path: str, fetched: int
) -> dict:
    per_sequence: dict[str, dict[str, dict]] = {}
    for record in plan.records:
        entry = per_sequence.setdefault(record.sequence, {}).setdefault(
            record.group, {"files": [], "bytes": 0}
        )
        entry["files"].append(record.relative_path)
        entry["bytes"] += record.size_bytes or 0
    return {
        "dataset": "nymeria",
        "source": LICENCE_URL,
        "licence": "CC BY-NC 4.0",
        "url_json": str(url_json),
        "out_dir": str(out_dir),
        "download_path": path,
        "selection_seed": seed,
        "groups": list(plan.groups),
        "num_sequences": len(plan.sequences),
        "sequences": per_sequence,
        "planned_bytes": plan.known_bytes,
        "unsized_files": plan.unknown_size_count,
        "fetched_bytes": fetched,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--sequences",
        type=int,
        help=f"deterministic number of sequences (default {DEFAULT_SEQUENCES})",
    )
    group.add_argument("--sequence-ids", nargs="+", help="explicit Nymeria sequence uids")
    group.add_argument(
        "--all-sequences",
        action="store_true",
        help="take every sequence in your manifest; --max-gb still applies",
    )
    parser.add_argument("--url-json", type=Path, default=DEFAULT_URL_JSON)
    parser.add_argument("--out", type=Path, default=DOWNLOADS)
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-unknown-size", action="store_true")
    parser.add_argument(
        "--no-official-cli",
        action="store_true",
        help=f"skip {DOWNLOADER_CLI} even if present and use the stdlib fallback",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan, download nothing")
    args = parser.parse_args(argv)

    try:
        fetch(
            url_json=args.url_json,
            out_dir=args.out,
            sequences=args.sequences,
            sequence_ids=args.sequence_ids,
            all_sequences=args.all_sequences,
            groups=tuple(args.groups),
            seed=args.seed,
            max_gb=args.max_gb,
            workers=args.workers,
            allow_unknown_size=args.allow_unknown_size,
            use_official=not args.no_official_cli,
            dry_run=args.dry_run,
        )
    except (LicenceError, BudgetExceeded, CatalogShapeError) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
