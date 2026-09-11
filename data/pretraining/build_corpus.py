"""One repeatable command to fetch, convert, and grid the label-free pretraining corpus.

    python -m data.pretraining.build_corpus --plan
    python -m data.pretraining.build_corpus --stage fetch   --datasets nhanes --yes
    python -m data.pretraining.build_corpus --stage convert --datasets nhanes
    python -m data.pretraining.build_corpus --stage grids   --datasets nhanes

Each stage is independently resumable, because each is idempotent on its own outputs: a
fetcher skips archives it already has, a converter rewrites ``sessions/`` from whatever is
downloaded, and the grid builder rewrites ``grids/`` from whatever is converted. Re-running
the whole thing after an interruption is therefore always safe and never re-downloads.

Fetching is opt-in per run (``--yes``). Several of these sources are hundreds of gigabytes
in transit, and a download that large should never start as a side effect of a command
someone ran to inspect the plan.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from data.pretraining.corpus_plan import (
    CANDIDATE_SOURCES,
    CORPUS_PLAN,
    DEFERRED_SOURCES,
    DEFAULT_BUDGET_GB,
    SourcePlan,
    PRETRAIN_WINDOW_SECONDS,
    format_plan,
    total_gigabytes,
)

REPO = Path(__file__).resolve().parents[2]
PRETRAIN_ROOT = Path(__file__).resolve().parent

#: Grid datasets a source produces. Most sources are one dataset; Nymeria records two
#: device families at different rates, so it converts into two dataset directories.
GRID_DATASETS = {
    "nhanes": ("nhanes",),
    "nymeria_xsens": ("nymeria_xsens",),
    "nymeria_aria": ("nymeria_aria",),
    "ego_exo4d": ("ego_exo4d",),
    "synthetic_imu": ("synthetic_imu",),
}

#: Where each planned source's fetch/convert modules live. Nymeria's two dataset
#: directories share one fetcher and one converter, because they come from one download.
MODULE_ROOT = {
    "nhanes": "data.pretraining.nhanes",
    "nymeria_xsens": "data.pretraining.nymeria",
    "nymeria_aria": "data.pretraining.nymeria",
    "ego_exo4d": "data.pretraining.ego_exo4d",
    "synthetic_imu": "data.pretraining.synthetic_imu",
}

# The top-level plan is deliberately explicit about every source's bounded selection.
# Fetchers have different interfaces, so invoking them bare either fails (NHANES/Ego-Exo4D)
# or quietly selects a development-sized default (Nymeria/synthetic IMU).
FETCH_ARGS = {
    "nhanes": ("--subjects", "3000"),
    "nymeria_xsens": ("--all-sequences", "--max-gb", "120"),
    "nymeria_aria": ("--all-sequences", "--max-gb", "120"),
    # Ego-Exo4D must be fetched in bounded batches and converted with --keep-existing because
    # its VRS input is intentionally discarded after each batch. See its README for the loop.
    "ego_exo4d": None,
    # All three sources are access-gated. Passing them explicitly prevents the converter's
    # convenience default (100STYLE only) from misrepresenting the corpus plan.
    "synthetic_imu": ("amass", "motion_x", "100style"),
}

CONVERT_ARGS = {
    "nhanes": (),
    "nymeria_xsens": (),
    "nymeria_aria": (),
    "ego_exo4d": None,
    "synthetic_imu": ("--sources", "amass", "motion_x", "100style"),
}


def _selected(datasets: Sequence[str] | None) -> tuple[SourcePlan, ...]:
    if not datasets:
        return CORPUS_PLAN
    known = {
        source.dataset: source
        for source in CORPUS_PLAN + DEFERRED_SOURCES + CANDIDATE_SOURCES
    }
    unknown = [name for name in datasets if name not in known]
    if unknown:
        raise SystemExit(
            f"unknown source(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known))}"
        )
    return tuple(known[name] for name in datasets)


def _module_exists(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _run(command: list[str]) -> int:
    print(f"  $ {' '.join(command)}", flush=True)
    return subprocess.call(command, cwd=REPO)


def _stage_fetch(sources: Sequence[SourcePlan], approved: bool) -> int:
    gated = [source for source in sources if source.access == "gated"]
    if gated:
        print(
            "Licence-gated sources in this selection: "
            + ", ".join(source.dataset for source in gated)
            + "\nTheir fetchers print the exact access steps and will refuse to run until "
              "the credentials or URL manifest exist. Nothing here bypasses a licence.\n"
        )
    if not approved:
        print(
            f"Refusing to start downloads without --yes. The full selection moves roughly "
            f"{total_gigabytes(tuple(sources)):.0f} GB of GRID output and considerably more "
            f"in transit (Ego-Exo4D alone is ~1 TB of VRS, streamed then deleted)."
        )
        return 1
    failures = 0
    seen: set[str] = set()
    for source in sources:
        module = MODULE_ROOT[source.dataset]
        if module in seen:
            continue  # one download serves both Nymeria dataset directories
        seen.add(module)
        if not _module_exists(f"{module}.fetch"):
            print(f"[skip] {source.dataset}: no fetch module at {module}.fetch")
            failures += 1
            continue
        args = FETCH_ARGS[source.dataset]
        if args is None:
            print(f"[manual-batch] {source.dataset}: fetch/convert it with the documented "
                  "bounded streaming loop; refusing a one-shot fetch that cannot preserve "
                  "the planned corpus.")
            failures += 1
            continue
        failures += bool(_run([sys.executable, "-m", f"{module}.fetch", *args]))
    return failures


def _stage_convert(sources: Sequence[SourcePlan]) -> int:
    failures = 0
    seen: set[str] = set()
    for source in sources:
        module = MODULE_ROOT[source.dataset]
        if module in seen:
            continue
        seen.add(module)
        if not _module_exists(f"{module}.convert"):
            print(f"[skip] {source.dataset}: no convert module at {module}.convert")
            failures += 1
            continue
        args = CONVERT_ARGS[source.dataset]
        if args is None:
            print(f"[manual-batch] {source.dataset}: conversion is coupled to its bounded "
                  "fetch loop; run the documented --keep-existing command.")
            failures += 1
            continue
        failures += bool(_run([sys.executable, "-m", f"{module}.convert", *args]))
    return failures


def _stage_grids(sources: Sequence[SourcePlan]) -> int:
    names = [name for source in sources for name in GRID_DATASETS.get(source.dataset, ())]
    if not names:
        return 0
    # Only the native regime: these sources exist to train HALO's rate-invariant frontend,
    # and the harmonised/non_harmonised regimes exist for the layout-locked baselines and
    # the evaluation path, neither of which may touch a label-free source.
    failed = bool(
        _run(
            [sys.executable, "-m", "data.scripts.build_grids",
             "--dataset", *names, "--alignment", "native",
             "--window-seconds", str(PRETRAIN_WINDOW_SECONDS)]
        )
    )
    if failed:
        return 1
    # CorpusIndex requires current quality caches. Regenerating them here makes a completed grid
    # stage operational rather than leaving the next training invocation to fail on stale caches.
    failures = 0
    for module in ("data.scripts.scan_implausible", "data.scripts.scan_duplicates"):
        failures += bool(_run([sys.executable, "-m", module, "--alignment", "native",
                               "--datasets", *names]))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", action="store_true",
                        help="Print the corpus plan and disk estimate, then exit.")
    parser.add_argument("--stage", choices=("fetch", "convert", "grids", "all"),
                        default="all")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Restrict to these sources (default: the whole plan).")
    parser.add_argument("--budget-gb", type=float, default=DEFAULT_BUDGET_GB,
                        help="Refuse a selection whose grid output exceeds this.")
    parser.add_argument("--yes", action="store_true",
                        help="Approve the download stage. Required for --stage fetch/all.")
    args = parser.parse_args()

    sources = _selected(args.datasets)
    estimate = total_gigabytes(sources)

    if args.plan:
        print(format_plan(sources))
        print()
        print(f"Grid output: {estimate:.1f} GB of {args.budget_gb:.0f} GB budget.")
        if CANDIDATE_SOURCES:
            print("\nCandidates excluded from the budget (access terms unconfirmed):")
            print(format_plan(CANDIDATE_SOURCES))
        return

    candidates = [source.dataset for source in sources if source.candidate]
    if candidates:
        raise SystemExit(
            f"{', '.join(candidates)}: access terms are unconfirmed, so this source is not "
            "buildable. Confirm the licence and move it into CORPUS_PLAN first."
        )
    if estimate > args.budget_gb:
        raise SystemExit(
            f"selection needs {estimate:.1f} GB of grids, over the {args.budget_gb:.0f} GB "
            "budget. Narrow --datasets or raise --budget-gb deliberately."
        )

    print(format_plan(sources))
    print(f"\nGrid output: {estimate:.1f} GB of {args.budget_gb:.0f} GB budget.\n")

    failures = 0
    if args.stage in ("fetch", "all"):
        print("== fetch ==")
        failures += _stage_fetch(sources, approved=args.yes)
        if failures:
            raise SystemExit(f"{failures} fetch stage(s) failed; refusing dependent stages.")
    if args.stage in ("convert", "all"):
        print("== convert ==")
        failures += _stage_convert(sources)
        if failures:
            raise SystemExit(f"{failures} conversion stage(s) failed; refusing grid build.")
    if args.stage in ("grids", "all"):
        print("== grids ==")
        failures += _stage_grids(sources)
    if failures:
        raise SystemExit(f"{failures} stage(s) failed; see the output above.")


if __name__ == "__main__":
    main()
