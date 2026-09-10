"""Obtain the motion-capture corpora (and body models) the virtual-IMU synthesis needs.

Three of the four sources are registration-gated and therefore **cannot be
scripted**: AMASS, Motion-X++ and Embody 3D each require you to create an
account, accept a licence, and receive per-user download links.  Those are
``manual`` entries here, in the same style as the gated sources in
``data/scripts/download_datasets.py``: this prints the page, the exact files,
and where to drop them, and does nothing else.

100STYLE is CC BY 4.0 on Zenodo and **is** downloaded automatically.  Its file
names are resolved from the Zenodo REST API at run time rather than hardcoded:
the record could not be reached to verify the file list when this module was
written (Zenodo answered 403/504 to non-browser clients from that network), so
guessing ``.../files/100STYLE.zip`` would have been an invented URL.  Asking the
API costs one request and cannot be wrong.

The SMPL / SMPL-H **body models** are a separate gate again (a different
account, on a different site, per model) and are needed only for the AMASS and
Motion-X++ paths -- the 100STYLE BVH path skins nothing and needs no model.

Usage (from the repo root)::

    python -m data.pretraining.synthetic_imu.fetch --list        # print every instruction
    python -m data.pretraining.synthetic_imu.fetch 100style      # the one auto source
    python -m data.pretraining.synthetic_imu.fetch amass         # prints manual steps
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"
MANIFEST = DOWNLOADS / "fetch_manifest.json"
USER_AGENT = "HALO-dataset-fetch/1.0"

ZENODO_RECORD_100STYLE = "8127870"
ZENODO_API = "https://zenodo.org/api/records/"


@dataclass(frozen=True)
class Source:
    name: str
    kind: str                 # "zenodo" | "manual"
    subdir: str
    licence: str
    homepage: str
    note: str
    record: Optional[str] = None


SOURCES: dict[str, Source] = {
    # ---- automatic ----
    "100style": Source(
        name="100style",
        kind="zenodo",
        subdir="100style",
        licence="CC BY 4.0",
        homepage="https://www.ianxmason.com/100style/",
        record=ZENODO_RECORD_100STYLE,
        note=(
            "100STYLE (Mason et al.): 100 locomotion styles, XSens capture at 60 fps, >4M frames "
            "(~19 h), BVH named <Style>/<Style>_<MovementType>.bvh plus Dataset_List.csv and "
            "Frame_Cuts.csv. Fully open (CC BY 4.0), so this is the smoke-test source: it needs no "
            "body model, because BVH already carries a posed skeleton. "
            f"Zenodo record {ZENODO_RECORD_100STYLE}; file names are resolved from the API."
        ),
    ),

    # ---- registration-gated corpora ----
    "amass": Source(
        name="amass",
        kind="manual",
        subdir="amass",
        licence="MPG non-commercial scientific research licence (no redistribution, single user)",
        homepage="https://amass.is.tue.mpg.de/",
        note=(
            "AMASS (~40 h, 344 subjects, 11k motions, SMPL-H). THE PRIMARY SOURCE.\n"
            "      1. Register and accept the licence at https://amass.is.tue.mpg.de/ .\n"
            "      2. Under 'Download', take the 'SMPL+H G' archive of each sub-dataset you want\n"
            "         (CMU, BMLrub, KIT, ... ). Start with CMU if you only want one.\n"
            "      3. Extract so the layout is\n"
            "           downloads/amass/<SubDataset>/<Subject>/<sequence>_poses.npz\n"
            "         which is exactly how the archives already unpack.\n"
            "      Also needs the SMPL-H body model: see the 'body_models' entry below.\n"
            "      Licence: non-commercial research only, NO redistribution, cite Mahmood et al. ICCV 2019."
        ),
    ),
    "motion_x": Source(
        name="motion_x",
        kind="manual",
        subdir="motion_x",
        licence="CC BY-NC-SA 4.0 annotations + each sub-dataset's own licence",
        homepage="https://github.com/IDEA-Research/Motion-X",
        note=(
            "Motion-X++ (181 h, 120.5k sequences, SMPL-X, 30 fps). SECONDARY SOURCE.\n"
            "      1. Request access with the Google Form linked from\n"
            "         https://github.com/IDEA-Research/Motion-X (per-subset approval).\n"
            "      2. Motion-X ships ANNOTATIONS ONLY: you must separately obtain each underlying\n"
            "         corpus (AMASS, AIST++, HAA500, HuMMan, IDEA400, ...) under its own licence.\n"
            "      3. The 2025-03 Motion-X++ re-release is at\n"
            "         https://huggingface.co/datasets/YuhongZhang/Motion-Xplusplus .\n"
            "      4. Extract SMPL-X motion arrays to downloads/motion_x/<subset>/<sequence>.npy .\n"
            "      NOTE: we skin Motion-X++ with SMPL-H, not SMPL-X (see convert.iter_motion_x)."
        ),
    ),
    "embody3d": Source(
        name="embody3d",
        kind="manual",
        subdir="embody3d",
        licence="Meta 'XRCIA' licence (non-commercial research; full text NOT verified)",
        homepage="https://www.meta.com/emerging-tech/codec-avatars/embody-3d/",
        note=(
            "Embody 3D (Meta Codec Avatars Lab, arXiv:2510.16258): 500 h, 439 participants, 54M frames.\n"
            "      *** CANDIDATE ONLY -- NO LOADER IS WIRED. convert.iter_embody3d raises. ***\n"
            "      What was confirmed (2026-09-09):\n"
            "        - Official tooling: https://github.com/facebookresearch/embody-3d (CC BY-NC 4.0\n"
            "          for the CODE), with src/download.py driven by a download.txt of signed links.\n"
            "        - Access is gated by a release form on the Meta page above; approval returns\n"
            "          21 per-user download links. There is no anonymous URL to script.\n"
            "        - The repo README states the DATASET is under the 'XRCIA' licence.\n"
            "      What could NOT be confirmed: the XRCIA licence text itself. Until somebody reads\n"
            "      it and confirms the terms permit this use, do not ingest this corpus."
        ),
    ),

    # ---- gated body models (needed only for the SMPL-family sources) ----
    "body_models": Source(
        name="body_models",
        kind="manual",
        subdir="body_models",
        licence="MPG non-commercial research licence (SMPL, SMPL-H/MANO); separate account per site",
        homepage="https://smpl.is.tue.mpg.de/",
        note=(
            "SMPL / SMPL-H body models -- needed to turn AMASS or Motion-X++ poses into the mesh\n"
            "      vertex positions the sensors sit on. NOT needed for 100STYLE.\n"
            "      1. SMPL-H (what AMASS uses): register at https://mano.is.tue.mpg.de/ and download\n"
            "         'Extended SMPL+H model (used in AMASS project)'  ->  smplh.tar.xz\n"
            "         Extract, then place the per-gender models as\n"
            "           downloads/body_models/SMPLH_{NEUTRAL,MALE,FEMALE}.npz\n"
            "         (the archive's <gender>/model.npz layout is also accepted as-is).\n"
            "      2. SMPL (only if you want the 24-joint model): register at\n"
            "         https://smpl.is.tue.mpg.de/ and download 'SMPL for Python' ->\n"
            "         SMPL_python_v.1.1.0.zip, placing SMPL_{NEUTRAL,MALE,FEMALE}.pkl in the same\n"
            "         directory.\n"
            "      These files are non-redistributable: they must be fetched per user, per licence.\n"
            "      `python -m data.pretraining.synthetic_imu.smpl_adapter` is not a downloader; the\n"
            "      converter fails with a pointed error until these are present."
        ),
    ),
}


def _request(url: str, headers: Optional[dict] = None) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})


def _digests(path: Path) -> dict[str, str]:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _download_resumable(url: str, destination: Path, expected_bytes: Optional[int] = None) -> int:
    """Stream ``url`` to ``destination``, resuming a ``.part`` file if one exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected_bytes is None or destination.stat().st_size == expected_bytes):
        print(f"    present  {destination.name} ({destination.stat().st_size / 1e6:.1f} MB)")
        return destination.stat().st_size

    partial = destination.with_suffix(destination.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    mode = "wb"
    if have:
        try:
            with urllib.request.urlopen(_request(url, {"Range": f"bytes={have}-"})) as response:
                if response.status == 206:
                    mode = "ab"
                    print(f"    resuming {destination.name} at {have / 1e6:.1f} MB")
                    with partial.open(mode) as output:
                        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
                else:                     # server ignored the range: start over
                    have, mode = 0, "wb"
        except urllib.error.HTTPError:
            have, mode = 0, "wb"
    if mode == "wb":
        print(f"    GET      {url}")
        with urllib.request.urlopen(_request(url)) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)

    size = partial.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise RuntimeError(f"{destination.name}: expected {expected_bytes} bytes, got {size}")
    os.replace(partial, destination)
    print(f"    done     {destination.name} ({size / 1e6:.1f} MB)")
    return size


def fetch_zenodo(source: Source, raw_dir: Path = DOWNLOADS) -> list[dict]:
    """Resolve a Zenodo record's file list from the API, then download each file."""
    target = raw_dir / source.subdir
    url = f"{ZENODO_API}{source.record}"
    print(f"[{source.name}] resolving {url}")
    with urllib.request.urlopen(_request(url, {"Accept": "application/json"})) as response:
        record = json.loads(response.read().decode("utf-8"))

    files = record.get("files") or []
    if not files:
        raise RuntimeError(
            f"Zenodo record {source.record} listed no files. Open "
            f"https://zenodo.org/records/{source.record} in a browser and download manually into "
            f"{target}."
        )
    results = []
    for entry in files:
        name = entry.get("key") or entry.get("filename")
        link = (entry.get("links") or {}).get("self") or entry.get("links", {}).get("download")
        size = entry.get("size")
        destination = target / name
        _download_resumable(link, destination, size)
        results.append(
            {
                "file": name,
                "url": link,
                "bytes": destination.stat().st_size,
                "zenodo_checksum": entry.get("checksum"),
                **_digests(destination),
            }
        )
    print(f"[{source.name}] unzip the archive(s) in {target} before running convert.py")
    return results


def print_manual(source: Source) -> None:
    print(f"[{source.name}] MANUAL -- {source.licence}")
    print(f"      homepage: {source.homepage}")
    print(f"      drop into: {DOWNLOADS / source.subdir}/")
    print(f"      {source.note}")


def fetch(name: str, raw_dir: Path = DOWNLOADS) -> dict:
    source = SOURCES[name]
    entry = {
        "source": name,
        "kind": source.kind,
        "licence": source.licence,
        "homepage": source.homepage,
        "directory": str((raw_dir / source.subdir).resolve()),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if source.kind == "manual":
        print_manual(source)
        present = sorted(p.name for p in (raw_dir / source.subdir).glob("*")) if (raw_dir / source.subdir).exists() else []
        entry.update({"status": "manual", "files_present": present})
        return entry
    try:
        entry.update({"status": "ok", "zenodo_record": source.record, "files": fetch_zenodo(source, raw_dir)})
    except Exception as error:            # noqa: BLE001 - record the failure, do not abort the run
        print(f"[{name}] FAILED: {error}")
        entry.update({"status": "failed", "error": str(error)})
    return entry


def write_manifest(entries: list[dict], manifest_path: Path = MANIFEST) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if manifest_path.exists():
        previous = {e["source"]: e for e in json.loads(manifest_path.read_text()).get("sources", [])}
    for entry in entries:
        previous[entry["source"]] = entry
    manifest_path.write_text(
        json.dumps(
            {
                "module": "data/pretraining/synthetic_imu",
                "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sources": [previous[key] for key in sorted(previous)],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[synthetic_imu] manifest -> {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="*", metavar="SOURCE",
                        help=f"sources to fetch, from {{{', '.join(SOURCES)}}} "
                             "(default: the auto-downloadable ones)")
    parser.add_argument("--list", action="store_true", help="print every source's access steps and exit")
    parser.add_argument("--raw-dir", type=Path, default=DOWNLOADS)
    args = parser.parse_args()

    if args.list:
        for source in SOURCES.values():
            if source.kind == "manual":
                print_manual(source)
            else:
                print(f"[{source.name}] AUTO ({source.kind}) -- {source.licence}")
                print(f"      homepage: {source.homepage}")
                print(f"      drop into: {args.raw_dir / source.subdir}/")
                print(f"      {source.note}")
            print()
        return 0

    unknown = sorted(set(args.sources) - set(SOURCES))
    if unknown:
        parser.error(f"unknown source(s) {unknown}; choose from {sorted(SOURCES)}")
    names = args.sources or [n for n, s in SOURCES.items() if s.kind != "manual"]
    entries = [fetch(name, args.raw_dir) for name in names]
    write_manifest(entries, args.raw_dir / "fetch_manifest.json")

    print("\n=== summary ===")
    for entry in entries:
        print(f"  {entry['source']:12s} {entry['status']}")
    manual = [n for n, s in SOURCES.items() if s.kind == "manual"]
    print(f"\nManual (run with --list for full steps): {', '.join(manual)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
