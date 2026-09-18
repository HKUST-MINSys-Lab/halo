"""Regenerate `data/datasets/<ds>/recordings.json` from the already-converted session listing.

A **recording** is one continuous physical capture. A **session** is one contiguous label block cut
out of it. Converters emit one session per block, so two sessions of the same (subject, label) are
frequently seconds apart inside a single bout — which makes them useless as independent enrollment
executions, however many of them there are. Measured 2026-08-11 over the ten 2026-08 additions:

    dataset          sessions/(subj,label)   recordings/(subj,label)   overcount
    opportunity            30 / 234                  6 / 6               10.2x
    dsads                   2 /  15                  1 / 1                3.2x
    mmfit                   3 /   4                  1 / 1                3.0x
    forth_trace             1 /   7                  1 / 1                1.7x
    realdisp                2 /  12                  2 / 6                1.1x
    monipar                 7 /   9                  7 / 9                1.0x   <- the real testbed

`eval_enrollment`'s `window_level_ids` gate does not catch this: it fires at
`singleton_execution_share > 0.95`, and these blocks carry several windows each (opportunity 0.66).
The gate detects *window*-level ids; this defect is *within-recording block*-level ids.

The grouping rule is owned by each converter as a `recording_id(session_id) -> str` function, so it
stays next to the code that minted the ids and is covered by the converter's own docstring. This
script only re-applies that pure function to the session directories already on disk, which means an
existing conversion does not have to be re-run against the raw archives to gain the map.

A dataset with no `recording_id` is declaring that each session IS its own recording — true for
monipar (one weekly visit), spar (one bout per file), phytmo (one file per series), kneepad (one
trial directory) and every legacy source.

Run:  python -m data.scripts.curate.build_recording_maps            # every dataset that defines one
      python -m data.scripts.curate.build_recording_maps --check    # verify on disk, write nothing
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from halo.paths import DATASETS_DIR

DATASETS = DATASETS_DIR

# Sources whose converter groups sessions onto a coarser physical capture. Keep this list explicit:
# silently discovering `recording_id` would make a typo in a converter change the leakage unit.
GROUPED_DATASETS = (
    "opportunity",
    "dsads",
    "forth_trace",
    "mmfit",
    "realdisp",
    "upper_limb_use",
)


def recording_map(dataset: str) -> dict[str, str]:
    """`{session_id: recording_id}` for one dataset, from its converter's own rule."""
    module = importlib.import_module(f"data.datasets.{dataset}.convert")
    rule = getattr(module, "recording_id", None)
    if rule is None:
        raise SystemExit(f"{dataset}/convert.py defines no recording_id(); remove it from "
                         "GROUPED_DATASETS or add the rule.")
    sessions = sorted(p.name for p in (DATASETS / dataset / "sessions").iterdir() if p.is_dir())
    if not sessions:
        raise SystemExit(f"{dataset}: no converted sessions on disk; run its converter first.")
    return {session: rule(session) for session in sessions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=list(GROUPED_DATASETS))
    parser.add_argument("--check", action="store_true",
                        help="compare against what is on disk and exit non-zero on a mismatch")
    args = parser.parse_args()

    stale: list[str] = []
    for dataset in args.datasets:
        mapping = recording_map(dataset)
        path = DATASETS / dataset / "recordings.json"
        groups = len(set(mapping.values()))
        if args.check:
            on_disk = json.loads(path.read_text()) if path.exists() else None
            status = "ok" if on_disk == mapping else "STALE"
            if status == "STALE":
                stale.append(dataset)
        else:
            path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
            status = "written"
        print(f"{dataset:16s} {len(mapping):6d} sessions -> {groups:5d} recordings  "
              f"({len(mapping) / groups:.1f} blocks each)  [{status}]", flush=True)

    if stale:
        raise SystemExit(f"stale recordings.json for: {', '.join(stale)}")


if __name__ == "__main__":
    main()
