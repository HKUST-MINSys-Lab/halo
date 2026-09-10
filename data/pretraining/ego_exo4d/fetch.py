"""Download a deterministic, bounded subset of Ego-Exo4D head-IMU VRS files.

Ego-Exo4D is 1,422 hours of egocentric+exocentric capture from 740+ participants
at 13 sites, recorded on Project Aria glasses. We want exactly one thing from it:
the head-mounted IMU. There is no IMU-only download part, so the smallest part
that contains it is ``take_vrs_noimagestream`` (995.592 GB, "VRS files for each
take without image stream data"). This fetcher therefore has no "download
everything" mode -- callers must choose a bounded take count or explicit uids,
and every plan is checked against a byte budget before a single object is pulled.

Prerequisites (all enforced up front, never bypassed):

1. A signed Ego-Exo4D License Agreement -- https://ego4d-data.org/ .
   Approval takes about two days; do it before you need the data.
2. ``pip install ego4d>=1.7.1`` (provides the ``egoexo`` CLI; 1.7.1+ enables the
   fast download path) and ``pip install awscli``.
3. ``aws configure`` with the access key/secret issued with the licence. A named
   profile is passed through with ``--s3-profile``.

Typical use::

    python -m data.pretraining.ego_exo4d.fetch --takes 40 --max-gb 120
    python -m data.pretraining.ego_exo4d.fetch --takes 40 --dry-run

The ``metadata`` part (0.046 GB) is always fetched first: it carries takes.json,
which is what makes take selection deterministic and the byte budget estimable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"
MANIFEST_PATH = DOWNLOADS / "fetch_manifest.json"

CLI_NAME = "egoexo"
PIP_PACKAGE = "ego4d>=1.7.1"
LICENSE_URL = "https://ego4d-data.org/"
DOCS_URL = "https://docs.ego-exo4d-data.org/download/"

METADATA_PART = "metadata"
IMU_PART = "take_vrs_noimagestream"
# Documented size of `take_vrs_noimagestream` in the v2 release (docs.ego-exo4d-data.org/download/).
IMU_PART_BYTES = int(995.592 * 1e9)

DEFAULT_SEED = 20260909
DEFAULT_MAX_GB = 120.0


class MissingPrerequisite(RuntimeError):
    """Raised when the CLI, the licence, or the AWS client is not set up."""


class BudgetExceeded(RuntimeError):
    """Raised when a download plan is larger than the caller's byte budget."""


# --------------------------------------------------------------------------
# Prerequisites
# --------------------------------------------------------------------------


def _cli_help() -> str:
    return (
        f"the `{CLI_NAME}` CLI was not found on PATH.\n"
        f"  1. Sign the Ego-Exo4D License Agreement at {LICENSE_URL}\n"
        "     (approval takes about 2 days -- there is no way around it and none is attempted here).\n"
        f"  2. pip install '{PIP_PACKAGE}'   # >=1.7.1 enables the fast download path\n"
        "  3. pip install awscli && aws configure   # use the keys issued with the licence\n"
        f"  See {DOCS_URL}"
    )


def _aws_help(profile: str | None) -> str:
    which = f"profile '{profile}'" if profile else "default credentials"
    return (
        f"no AWS client configuration found for {which}.\n"
        "  pip install awscli\n"
        "  aws configure            # access key + secret from your Ego-Exo4D licence email\n"
        "  (press Enter twice to accept the default region and output format)\n"
        f"  Then re-run, adding --s3-profile <name> if you used a named profile.\n"
        f"  If you have no keys yet, sign the License Agreement at {LICENSE_URL} first "
        "(approval takes about 2 days)."
    )


def require_cli(which=shutil.which) -> str:
    path = which(CLI_NAME)
    if not path:
        raise MissingPrerequisite(_cli_help())
    return path


def aws_configured(profile: str | None = None, env: Mapping[str, str] | None = None) -> bool:
    """True when an AWS client looks usable for `profile`.

    Deliberately permissive about *how* credentials are supplied (env vars, shared
    config, SSO, instance role): we only refuse when there is no sign of any.
    """
    env = os.environ if env is None else env
    if not profile and env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        return True
    if env.get("AWS_PROFILE") and not profile:
        profile = env["AWS_PROFILE"]
    home = Path(env.get("AWS_SHARED_CREDENTIALS_FILE", "")) if env.get(
        "AWS_SHARED_CREDENTIALS_FILE"
    ) else Path(env.get("HOME", "~")).expanduser() / ".aws" / "credentials"
    config = Path(env.get("AWS_CONFIG_FILE", "")) if env.get("AWS_CONFIG_FILE") else Path(
        env.get("HOME", "~")
    ).expanduser() / ".aws" / "config"
    texts = [path.read_text(errors="replace") for path in (home, config) if path.is_file()]
    if not texts:
        return False
    if profile is None:
        return True
    return any(f"[{profile}]" in text or f"[profile {profile}]" in text for text in texts)


def require_aws(profile: str | None = None, env: Mapping[str, str] | None = None) -> None:
    if not aws_configured(profile, env):
        raise MissingPrerequisite(_aws_help(profile))


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def find_metadata_file(root: Path, name: str = "takes.json") -> Path | None:
    """Locate a metadata JSON under `root`, shallowest match first.

    The downloader has placed metadata directly in the output directory and, in
    other releases, under a nested release directory. Searching by depth keeps
    the converter working either way.
    """
    direct = root / name
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(name), key=lambda p: (len(p.parts), str(p)))
    return matches[0] if matches else None


def load_takes(root: Path) -> list[dict]:
    path = find_metadata_file(root, "takes.json")
    if path is None:
        raise FileNotFoundError(
            f"takes.json not found under {root}; run this fetcher without --skip-metadata, "
            f"or `{CLI_NAME} -o {root} --parts {METADATA_PART}` by hand (0.046 GB)."
        )
    takes = json.loads(path.read_text())
    if isinstance(takes, dict):  # tolerate a {take_uid: take} mapping
        takes = list(takes.values())
    return [take for take in takes if take.get("take_uid")]


def filter_takes(
    takes: Sequence[Mapping],
    universities: Sequence[str] | None = None,
    uids: Sequence[str] | None = None,
) -> list[dict]:
    result = list(takes)
    if universities:
        wanted = {name.lower() for name in universities}
        result = [t for t in result if str(t.get("university_name", "")).lower() in wanted]
    if uids:
        wanted_uids = set(uids)
        result = [
            t
            for t in result
            if t.get("take_uid") in wanted_uids or t.get("capture_uid") in wanted_uids
        ]
    return result


def select_takes(takes: Sequence[Mapping], count: int, seed: int) -> list[dict]:
    """Deterministic, seed-stable subset over sorted take uids.

    Ordering by a salted hash (rather than by uid, duration, or university) avoids
    favouring any site or any unusually short take, and re-running with the same
    seed and the same metadata release reproduces the same set exactly.
    """
    if count <= 0:
        raise ValueError("--takes must be positive; full-corpus downloads are intentionally blocked")
    ordered = sorted(
        sorted(takes, key=lambda t: str(t["take_uid"])),
        key=lambda t: hashlib.sha256(f"{seed}:{t['take_uid']}".encode()).hexdigest(),
    )
    return sorted(ordered[: min(count, len(ordered))], key=lambda t: str(t["take_uid"]))


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def estimate_bytes(
    selected: Sequence[Mapping],
    population: Sequence[Mapping] | None = None,
) -> tuple[int, str]:
    """Estimate the download size of `selected`, returning (bytes, how).

    takes.json publishes no per-file size for the VRS objects, so this is always
    an ESTIMATE and the returned string says which one:

    * "duration_share" -- the documented 995.592 GB part size apportioned by each
      take's ``duration_sec`` over the whole release. VRS payload is dominated by
      time-series sensors once the image streams are gone, so duration is the best
      proxy available offline.
    * "uniform_per_take" -- 995.592 GB / (number of takes), when durations are
      missing.
    """
    population = list(population) if population else list(selected)
    total_duration = sum(float(t.get("duration_sec") or 0.0) for t in population)
    selected_duration = sum(float(t.get("duration_sec") or 0.0) for t in selected)
    if total_duration > 0 and selected_duration > 0:
        return int(IMU_PART_BYTES * selected_duration / total_duration), "duration_share"
    if population:
        return int(IMU_PART_BYTES * len(selected) / len(population)), "uniform_per_take"
    return 0, "uniform_per_take"


def check_budget(plan_bytes: int, max_gb: float, how: str = "estimate") -> None:
    if max_gb <= 0:
        raise ValueError("--max-gb must be positive")
    if plan_bytes > max_gb * 1e9:
        raise BudgetExceeded(
            f"plan is ~{plan_bytes / 1e9:.1f} GB ({how} -- not a measured size) which exceeds "
            f"--max-gb {max_gb:g}. Lower --takes, narrow --universities, or raise --max-gb "
            "deliberately."
        )


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def build_command(
    out_dir: Path,
    parts: Sequence[str],
    uids: Sequence[str] = (),
    universities: Sequence[str] = (),
    splits: Sequence[str] = (),
    s3_profile: str | None = None,
    num_workers: int | None = None,
) -> list[str]:
    command = [CLI_NAME, "-o", str(out_dir), "--parts", *parts, "-y"]
    if uids:
        command += ["--uids", *uids]
    if universities:
        command += ["--universities", *universities]
    if splits:
        command += ["--splits", *splits]
    if s3_profile:
        command += ["--s3_profile", s3_profile]
    if num_workers:
        command += ["--num_workers", str(num_workers)]
    return command


def _run(command: Sequence[str], runner=subprocess.run) -> None:
    print(f"[ego_exo4d] $ {' '.join(command)}", flush=True)
    result = runner(list(command))
    code = getattr(result, "returncode", 0)
    if code:
        raise RuntimeError(f"{CLI_NAME} exited with status {code}: {' '.join(command)}")


def realized_take_dirs(root: Path, takes: Sequence[Mapping]) -> dict[str, bool]:
    """Which of the planned takes actually have a directory on disk (resumability)."""
    present: dict[str, bool] = {}
    for take in takes:
        name = take.get("take_name") or ""
        root_dir = take.get("root_dir") or (f"takes/{name}" if name else "")
        candidates = [root / root_dir, root / "takes" / name] if name else [root / root_dir]
        present[str(take["take_uid"])] = any(
            path.is_dir() and any(path.rglob("*.vrs")) for path in candidates if str(path) != str(root)
        )
    return present


def fetch(
    out_dir: Path = DOWNLOADS,
    takes_count: int | None = None,
    uids: Sequence[str] = (),
    universities: Sequence[str] = (),
    splits: Sequence[str] = (),
    seed: int = DEFAULT_SEED,
    max_gb: float = DEFAULT_MAX_GB,
    s3_profile: str | None = None,
    num_workers: int | None = None,
    dry_run: bool = False,
    skip_metadata: bool = False,
    runner=subprocess.run,
) -> dict:
    if takes_count is None and not uids:
        raise ValueError(
            "choose --takes N or --uids ...; there is deliberately no unbounded mode "
            f"({IMU_PART} is {IMU_PART_BYTES / 1e9:.0f} GB in full)."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    require_cli()
    require_aws(s3_profile)

    if not skip_metadata:
        _run(
            build_command(out_dir, [METADATA_PART], s3_profile=s3_profile, num_workers=num_workers),
            runner=runner,
        )

    population = load_takes(out_dir)
    pool = filter_takes(population, universities=universities, uids=uids)
    if not pool:
        raise ValueError(
            "no takes matched the requested universities/uids; "
            f"metadata lists {len(population)} takes."
        )
    selected = select_takes(pool, takes_count, seed) if takes_count else sorted(
        pool, key=lambda t: str(t["take_uid"])
    )

    plan_bytes, how = estimate_bytes(selected, population)
    print(
        f"[ego_exo4d] plan: {len(selected)} takes, "
        f"~{plan_bytes / 1e9:.1f} GB ESTIMATED ({how}; takes.json publishes no per-file sizes)",
        flush=True,
    )
    check_budget(plan_bytes, max_gb, how)

    record = {
        "source": DOCS_URL,
        "release": "Ego-Exo4D v2",
        "part": IMU_PART,
        "part_total_bytes": IMU_PART_BYTES,
        "selection_seed": seed if takes_count else None,
        "requested_takes": takes_count,
        "universities": list(universities),
        "splits": list(splits),
        "explicit_uids": list(uids),
        "estimated_bytes": plan_bytes,
        "estimate_method": how,
        "estimate_is_measured": False,
        "max_gb": max_gb,
        "takes": [
            {
                "take_uid": t["take_uid"],
                "take_name": t.get("take_name"),
                "capture_uid": t.get("capture_uid"),
                "participant_uid": t.get("participant_uid"),
                "university_name": t.get("university_name"),
                "duration_sec": t.get("duration_sec"),
                "root_dir": t.get("root_dir"),
            }
            for t in selected
        ],
    }

    if dry_run:
        record["dry_run"] = True
        print("[ego_exo4d] dry run: nothing downloaded")
        _write_manifest(out_dir, record)
        return record

    _run(
        build_command(
            out_dir,
            [IMU_PART],
            uids=[str(t["take_uid"]) for t in selected],
            universities=universities,
            splits=splits,
            s3_profile=s3_profile,
            num_workers=num_workers,
        ),
        runner=runner,
    )

    present = realized_take_dirs(out_dir, selected)
    record["realized_take_uids"] = sorted(uid for uid, ok in present.items() if ok)
    record["missing_take_uids"] = sorted(uid for uid, ok in present.items() if not ok)
    record["bytes_on_disk"] = sum(
        path.stat().st_size for path in out_dir.rglob("*.vrs") if path.is_file()
    )
    _write_manifest(out_dir, record)
    print(
        f"[ego_exo4d] ready: {len(record['realized_take_uids'])}/{len(selected)} takes on disk, "
        f"{record['bytes_on_disk'] / 1e9:.2f} GB of VRS"
    )
    if record["missing_take_uids"]:
        print(
            f"[ego_exo4d] note: {len(record['missing_take_uids'])} planned takes are absent. "
            "--splits/--views filters are applied by the CLI *after* uid selection, so a "
            "split filter can legitimately shrink the realized set. Re-running is a no-op "
            "for takes already on disk."
        )
    return record


def _write_manifest(out_dir: Path, record: dict) -> None:
    path = Path(out_dir) / MANIFEST_PATH.name
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[ego_exo4d] wrote {path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=DOWNLOADS)
    parser.add_argument("--takes", type=int, help="bounded, deterministic number of takes")
    parser.add_argument("--uids", nargs="+", default=[], help="explicit take_uid / capture_uid values")
    parser.add_argument("--universities", "-u", nargs="+", default=[], help="e.g. cmu unc sfu")
    parser.add_argument("--splits", nargs="+", default=[], choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB)
    parser.add_argument("--s3-profile", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--skip-metadata", action="store_true", help="reuse an existing takes.json")
    parser.add_argument("--dry-run", action="store_true", help="plan and budget-check only")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        fetch(
            out_dir=args.output_dir,
            takes_count=args.takes,
            uids=args.uids,
            universities=args.universities,
            splits=args.splits,
            seed=args.seed,
            max_gb=args.max_gb,
            s3_profile=args.s3_profile,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
            skip_metadata=args.skip_metadata,
        )
    except (MissingPrerequisite, BudgetExceeded, ValueError, FileNotFoundError) as error:
        print(f"[ego_exo4d] {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
