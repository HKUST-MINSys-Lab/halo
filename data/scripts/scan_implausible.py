"""Scan the built grids for PHYSICALLY IMPOSSIBLE windows and cache their indices.

Some source releases contain windows beyond any consumer MEMS full-scale range. The widest ranges
shipped on phones/watches are +/-2000 dps (34.907 rad/s) and +/-16 g; a sample beyond either is not
vigorous motion, it is a scale/acquisition artifact. Training on it teaches the filterbank a band
energy that no real device can produce. BOTH modalities are screened, so accelerometer-only streams
(capture24, extrasensory, nhanes, unimib_shar) are covered — they used to be skipped entirely.

We DROP such windows rather than clip them — clipping fabricates a plausible value for data we know
is wrong. The scan is cached to a small JSON so ``CorpusIndex`` stays lazy (it must not read every
grid at construction time).

Run:  python -m data.scripts.scan_implausible
      python -m data.scripts.scan_implausible --alignment non_harmonised
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

#: Widest consumer gyroscope full-scale range (+/-2000 degrees/second), in rad/s. A window whose
#: |gyro| exceeds this on ANY axis/sample cannot have been produced by the declared hardware.
GYRO_RAIL_RAD_S = float(np.radians(2000.0))          # 34.907 rad/s

#: Widest consumer accelerometer full-scale range (+/-16 g). Same argument as the gyro rail: a
#: sample beyond it is a scale/acquisition artifact, not motion. This is the only screen an
#: ACCEL-ONLY stream can get, and until it existed those streams (capture24, extrasensory, nhanes,
#: unimib_shar) were passed through entirely unscreened.
ACCEL_RAIL_G = 16.0

OUT = Path(__file__).resolve().parents[2] / "data" / "quality" / "implausible_windows.json"


def _discover(discover_grids, alignment: str, window_seconds: float):
    try:
        return discover_grids(alignment, window_seconds=window_seconds)
    except TypeError:
        if np.isclose(window_seconds, 6.0):
            return discover_grids(alignment)
        raise


def _duration_tag(window_seconds: float) -> str:
    return f"w{float(window_seconds):g}".replace(".", "p")


def cache_path(alignment: str, window_seconds: float = 6.0) -> Path:
    """One cache per alignment; native keeps the historical filename."""
    suffix = "" if alignment == "native" else f"_{alignment}"
    duration = "" if np.isclose(window_seconds, 6.0) else f"_{_duration_tag(window_seconds)}"
    return OUT.with_name(f"{OUT.stem}{suffix}{duration}{OUT.suffix}")


def scan(alignment: str = "native", datasets: Sequence[str] | None = None,
         window_seconds: float = 6.0) -> dict:
    """Scan all grids, or safely refresh only named datasets in an existing cache.

    A full corpus can be hundreds of GB.  After rebuilding one source, re-reading every
    unrelated grid is needless I/O, but accepting its old exclusion entries without a
    fingerprint check would be unsafe.  Incremental refresh therefore requires a current
    cache and rejects it if any unselected stream changed.
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
        bad = {
            key: value for key, value in prior.get("windows", {}).items()
            if key not in {ref.key for ref in selected_refs}
        }
        stats = [
            tuple(row) for row in prior.get("summary", [])
            if row[0] not in {ref.key for ref in selected_refs}
        ]
    for ref in sorted(selected_refs, key=lambda r: r.key):
        if ref.n_windows == 0:
            continue
        mask = np.asarray(ref.mask, dtype=bool)
        data = ref.load_data()
        over = np.zeros(ref.n_windows, dtype=bool)
        gyro_peak = accel_peak = 0.0
        nonfinite_windows = 0
        # Block-wise so a multi-GB grid is never fully resident (capture24 is 1.5M windows).
        for start in range(0, ref.n_windows, 4096):
            block = np.asarray(data[start:start + 4096], dtype=np.float32)
            finite = np.isfinite(block).all(axis=(1, 2))
            nonfinite_windows += int((~finite).sum())
            a_peak = np.where(np.isfinite(block[:, :, :3]),
                              np.abs(block[:, :, :3]), 0.0).max(axis=(1, 2))
            accel_peak = max(accel_peak, float(a_peak.max()))
            hit = ~finite | (a_peak > ACCEL_RAIL_G)
            if mask[3:].any():                        # accel-only streams have no gyro to check
                g_peak = np.where(np.isfinite(block[:, :, 3:]),
                                  np.abs(block[:, :, 3:]), 0.0).max(axis=(1, 2))
                gyro_peak = max(gyro_peak, float(g_peak.max()))
                hit |= g_peak > GYRO_RAIL_RAD_S
            over[start:start + len(hit)] = hit
        idx = np.nonzero(over)[0]
        if idx.size:
            bad[ref.key] = sorted(int(i) for i in idx)
            stats.append((ref.key, int(idx.size), ref.n_windows, gyro_peak, accel_peak,
                          nonfinite_windows))
    return {"alignment": alignment, "window_seconds": float(window_seconds),
            "grid_fingerprint": grid_corpus_fingerprint(alignment, refs),
            "stream_fingerprints": {
                ref.key: grid_corpus_fingerprint(alignment, [ref]) for ref in refs
            },
            "gyro_rail_rad_s": GYRO_RAIL_RAD_S,
            "accel_rail_g": ACCEL_RAIL_G, "windows": bad, "summary": stats}


def load(alignment: str = "native", *, require: bool = False,
         window_seconds: float = 6.0) -> dict[str, set[int]]:
    """Load exclusions, optionally requiring a cache for the exact current grid corpus."""
    path = cache_path(alignment, window_seconds)
    if not path.exists():
        if require:
            raise FileNotFoundError(
                f"{path} is missing — run `python -m data.scripts.scan_implausible "
                f"--alignment {alignment}`."
            )
        return {}
    blob = json.loads(path.read_text())
    if blob.get("alignment") != alignment:
        if require:
            raise ValueError(
                f"{path} was built for alignment {blob.get('alignment')!r}, not {alignment!r}; "
                "re-run data.scripts.scan_implausible for this alignment."
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
                "`python -m data.scripts.scan_implausible` after every grid rebuild"
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
    print(f"gyro rail = {GYRO_RAIL_RAD_S:.3f} rad/s (2000 dps) · accel rail = {ACCEL_RAIL_G} g")
    for key, n, tot, gyro_peak, accel_peak, nonfinite in blob["summary"]:
        print(f"  {key:28s} {n:5d}/{tot:6d} windows exceed a rail "
              f"or are non-finite (non-finite {nonfinite}; peak gyro {gyro_peak:.2f} rad/s, "
              f"peak |accel| {accel_peak:.2f} g)")
    print(f"-> {total} windows cached for exclusion in {path}")


if __name__ == "__main__":
    main()
