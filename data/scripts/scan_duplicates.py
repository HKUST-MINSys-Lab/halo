"""Scan the built grids for BYTE-IDENTICAL repeated windows and cache their indices.

Two windows of real motion are never bit-for-bit equal. When they are, the source device
re-emitted a stale buffer instead of sampling. ExtraSensory's Pebble watch does this in long
runs: subject ``0A986513`` has one group of 178 identical recordings spanning 10,620 s, and
subject ``3600D531`` is 87.7% duplicates. NHANES does it too, but benignly — its repeats are
motionless non-wear blocks quantised to the same 0.001 g triple.

Two rules, because the two cases differ in what we can still trust:

* **All members share one label** -> exactly one of them is a genuine observation. Keep the
  first, drop the rest. The signal is real; only the repeats are fabricated.
* **Members disagree on the label** -> the buffer went stale across a label change, so the
  window is real but its label is unknowable. Drop the WHOLE group. Keeping an arbitrary
  member injects a known-wrong label at least half the time. On ExtraSensory's wrist stream
  this is 36.9% of duplicate pairs, dominated by lying<->sitting (4,413 pairs).

Duplicates are only ever matched WITHIN one stream. Simultaneous multi-placement streams
(xrf_v2, nfi_fared) are meant to describe the same event from different sensors.

Like ``scan_implausible``, results are cached to a small JSON so ``CorpusIndex`` stays lazy.

Run:  python -m data.scripts.scan_duplicates
      python -m data.scripts.scan_duplicates --alignment non_harmonised
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from halo.paths import DATA_DIR

OUT = DATA_DIR / "quality" / "duplicate_windows.json"


def _duration_tag(window_seconds: float) -> str:
    return f"w{float(window_seconds):g}".replace(".", "p")


def cache_path(alignment: str, window_seconds: float = 6.0) -> Path:
    """One cache per alignment; native keeps the historical filename."""
    suffix = "" if alignment == "native" else f"_{alignment}"
    duration = "" if np.isclose(window_seconds, 6.0) else f"_{_duration_tag(window_seconds)}"
    return OUT.with_name(f"{OUT.stem}{suffix}{duration}{OUT.suffix}")

#: Windows are hashed in blocks so a 4 GB grid is never fully resident.
BLOCK = 4096


def _discover(discover_grids, alignment: str, window_seconds: float):
    try:
        return discover_grids(alignment, window_seconds=window_seconds)
    except TypeError:
        # Compatibility for narrow synthetic test doubles predating duration-qualified grids.
        if np.isclose(window_seconds, 6.0):
            return discover_grids(alignment)
        raise


def scan_stream(data, labels, lengths=None) -> tuple[list[int], int, int]:
    """Return (indices to drop, n duplicate groups, n groups dropped whole for label conflict)."""
    groups: dict[bytes, list[int]] = {}
    n = data.shape[0]
    if lengths is None:
        lengths = np.full(n, data.shape[1], dtype=np.int32)
    for start in range(0, n, BLOCK):
        block = np.asarray(data[start:start + BLOCK], dtype=np.float32)
        for j, window in enumerate(block):
            length = int(lengths[start + j])
            payload = length.to_bytes(4, "little") + window[:length].tobytes()
            digest = hashlib.blake2b(payload, digest_size=16).digest()
            groups.setdefault(digest, []).append(start + j)

    drop: list[int] = []
    n_groups = n_conflict = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        n_groups += 1
        if len({labels[i] for i in members}) > 1:    # stale across a label change
            n_conflict += 1
            drop.extend(members)                     # label unknowable -> drop the whole group
        else:
            drop.extend(members[1:])                 # keep one genuine observation
    return sorted(drop), n_groups, n_conflict


def scan(alignment: str = "native", datasets: Sequence[str] | None = None,
         window_seconds: float = 6.0) -> dict:
    """Scan all grids, or safely refresh only named datasets in an existing cache.

    Incremental refresh retains old exclusions only after checking that every unselected
    stream still has the cached fingerprint.  It therefore saves I/O after an isolated grid
    rebuild without silently reusing a stale duplicate screen.
    """
    from data.scripts.eda.grid_io import discover_grids, grid_corpus_fingerprint

    refs = _discover(discover_grids, alignment, window_seconds)
    requested = set(datasets or ())
    selected_refs = refs
    bad: dict[str, list[int]] = {}
    stats: list[tuple] = []
    if requested:
        available = {ref.dataset for ref in refs}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"no {alignment} grids for requested dataset(s): {missing}")
        path = cache_path(alignment, window_seconds)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is required for an incremental refresh; run a full scan first"
            )
        prior = json.loads(path.read_text())
        fingerprints = prior.get("stream_fingerprints", {})
        stale = [
            ref.key for ref in refs if ref.dataset not in requested
            and fingerprints.get(ref.key) != grid_corpus_fingerprint(alignment, [ref])
        ]
        if stale:
            raise ValueError(
                "cannot incrementally refresh while unselected streams have changed; "
                f"run a full scan (stale/missing: {stale[:5]})"
            )
        selected_refs = [ref for ref in refs if ref.dataset in requested]
        selected_keys = {ref.key for ref in selected_refs}
        bad = {key: value for key, value in prior.get("windows", {}).items() if key not in selected_keys}
        stats = [tuple(row) for row in prior.get("summary", []) if row[0] not in selected_keys]
    for ref in sorted(selected_refs, key=lambda r: r.key):
        if ref.n_windows == 0:
            continue
        drop, n_groups, n_conflict = scan_stream(
            ref.load_data(), ref.labels, ref.load_lengths(),
        )
        if drop:
            bad[ref.key] = drop
            stats.append((ref.key, len(drop), ref.n_windows, n_groups, n_conflict))
    return {"alignment": alignment, "window_seconds": float(window_seconds),
            "grid_fingerprint": grid_corpus_fingerprint(alignment, refs),
            "stream_fingerprints": {
                ref.key: grid_corpus_fingerprint(alignment, [ref]) for ref in refs
            },
            "windows": bad, "summary": stats}


def load(alignment: str = "native", *, require: bool = False,
         window_seconds: float = 6.0) -> dict[str, set[int]]:
    """stream key -> set of window indices to exclude.

    ``require=True`` refuses to return an empty screen. A missing or wrong-alignment cache is
    indistinguishable from "this corpus has no duplicates", so on a remote snapshot that shipped
    without ``data/quality/duplicate_windows.json`` training would silently readmit every stale
    ExtraSensory buffer. Callers that depend on the screen should pass require=True.
    """
    path = cache_path(alignment, window_seconds)
    if not path.exists():
        if require:
            raise FileNotFoundError(
                f"{path} is missing — run `python -m data.scripts.scan_duplicates "
                f"--alignment {alignment}`. Training "
                "without it silently readmits byte-identical stale-buffer windows."
            )
        return {}
    blob = json.loads(path.read_text())
    if blob.get("alignment") != alignment:
        if require:
            raise ValueError(
                f"{path} was built for alignment {blob.get('alignment')!r}, not {alignment!r}; "
                "re-run data.scripts.scan_duplicates for this alignment."
            )
        return {}
    if not np.isclose(float(blob.get("window_seconds", 6.0)), float(window_seconds)):
        if require:
            raise ValueError(f"{path} was built for a different window duration")
        return {}
    from data.scripts.eda.grid_io import discover_grids, grid_corpus_fingerprint
    stream_fingerprints = blob.get("stream_fingerprints")
    if stream_fingerprints:
        stale = [
            ref.key for ref in _discover(discover_grids, alignment, window_seconds)
            if stream_fingerprints.get(ref.key) != grid_corpus_fingerprint(alignment, [ref])
        ]
        current_matches = not stale
    else:  # backwards-compatible validation for pre-per-stream cache fixtures/artifacts
        current_matches = blob.get("grid_fingerprint") == grid_corpus_fingerprint(alignment)
        stale = []
    if not current_matches:
        if require:
            raise ValueError(
                f"{path} does not match the current {alignment} grids; re-run "
                "`python -m data.scripts.scan_duplicates` after every grid rebuild"
                + (f" (stale/missing streams: {stale[:5]})" if stale else ".")
            )
        return {}
    return {k: set(v) for k, v in blob.get("windows", {}).items()}


def main() -> None:
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", default="native",
                        choices=("native", "harmonised", "non_harmonised"))
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="incrementally refresh only these datasets; requires current unchanged cache")
    parser.add_argument("--window-seconds", type=float, default=6.0)
    args = parser.parse_args()
    blob = scan(args.alignment, args.datasets, args.window_seconds)
    path = cache_path(args.alignment, args.window_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2) + "\n")
    total = sum(len(v) for v in blob["windows"].values())
    for key, n, tot, groups, conflict in blob["summary"]:
        print(f"  {key:28s} {n:6d}/{tot:7d} windows dropped "
              f"({n / tot:5.1%}) · {groups} groups, {conflict} label-conflicted")
    print(f"-> {total} duplicate windows cached for exclusion in {path}")


if __name__ == "__main__":
    main()
