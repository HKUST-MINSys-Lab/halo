"""Download raw HALO datasets into ``data/datasets/<ds>/downloads/``.

Most sources below were verified working on 2026-07-12. The entry itself records sources that need
manual access or a dedicated setup workflow; notably MobiAct must use its official annotated release
and is intentionally not fetched from a third-party mirror. Two mechanisms:

  * **direct**  — an HTTP(S) archive (UCI / uni-mannheim); downloaded with the stdlib and unzipped
                  in place (nested UCI zips are unpacked too).
  * **kaggle**  — a Kaggle dataset slug; needs ``~/.kaggle/kaggle.json`` credentials. Some Kaggle
                  datasets require accepting their terms on the website first (mobiact returns 403
                  until you do) — those are marked ``manual``.

Gated sources with no scriptable URL (shoaib / capture24 / inclusivehar) are ``manual``: the note
gives the page + what to drop where. After downloading, run the per-dataset converter
(``python -m data.datasets.<ds>.convert``) then ``python -m data.scripts.build_grids``.

Usage (from the repo root):
    python -m data.scripts.download_datasets                # all auto-downloadable datasets
    python -m data.scripts.download_datasets uci_har hhar   # only these
"""

from __future__ import annotations

import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from halo.paths import REPO_ROOT

REPO = REPO_ROOT


@dataclass(frozen=True)
class Source:
    name: str
    kind: str                       # "direct" | "kaggle" | "manual"
    url: Optional[str] = None       # direct URL
    kaggle: Optional[str] = None    # kaggle slug
    note: str = ""
    manual: bool = False            # requires human action even if a url/slug exists


SOURCES: dict[str, Source] = {
    # ---- direct HTTP (UCI + uni-mannheim), auto-downloadable ----
    "uci_har": Source("uci_har", "direct",
        url="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"),
    "pamap2": Source("pamap2", "direct",
        url="http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"),
    "mhealth": Source("mhealth", "direct",
        url="https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"),
    "wisdm": Source("wisdm", "direct",
        url="https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"),
    "hhar": Source("hhar", "direct",
        url="https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"),
    "hapt": Source("hapt", "direct",
        url="https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"),
    "harth": Source("harth", "direct",
        url="https://archive.ics.uci.edu/static/public/779/harth.zip",
        note="Placement stress-test set (thigh + lower back); role='stress', not in the primary build."),
    "realworld": Source("realworld", "direct",
        url="http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip"),

    # ---- Kaggle (needs ~/.kaggle/kaggle.json) ----
    "unimib_shar": Source("unimib_shar", "kaggle", kaggle="wangboluo/unimib-shar-dataset"),
    "kuhar": Source("kuhar", "kaggle", kaggle="niloy333/kuhar"),
    "mobiact": Source("mobiact", "manual", manual=True,
        note="Use only the official annotated MobiAct release, not MobiFall. After obtaining the "
             "archive from https://bmi.lab.uowm.gr/the-mobiact-dataset/, place it under "
             "data/datasets/mobiact/downloads/ and run python -m data.datasets.mobiact.setup "
             "--archive <archive>."),

    # ---- gated, no scriptable URL ----
    "shoaib": Source("shoaib", "manual", manual=True,
        note="Sensors Activity Dataset (Shoaib et al.). Download from "
             "https://www.utwente.nl/en/eemcs/ps/research/dataset/ into data/datasets/shoaib/downloads/."),
    "capture24": Source("capture24", "manual", manual=True,
        note="CAPTURE-24 (~4 GB, CC BY). Download capture24.zip from Oxford ORA "
             "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001 into "
             "data/datasets/capture24/downloads/ and unzip."),
    "inclusivehar": Source("inclusivehar", "manual", manual=True,
        note="InclusiveHAR CSV from Mendeley doi:10.17632/r78dn3f6nc.4 "
             "https://data.mendeley.com/datasets/r78dn3f6nc/4 into data/datasets/inclusivehar/downloads/."),
}


def _dest(name: str) -> Path:
    d = REPO / "data" / "datasets" / name / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unzip_all(d: Path) -> None:
    """Unzip src.zip, then any nested zips it produced (UCI nests a second archive)."""
    seen: set[str] = set()
    for _ in range(3):  # a couple of nesting levels is enough
        zips = [p for p in d.glob("*.zip") if p.name not in seen]
        if not zips:
            break
        for z in zips:
            seen.add(z.name)
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(d)
            except zipfile.BadZipFile:
                print(f"    ! {z.name} is not a valid zip, skipping")


def _download_direct(src: Source) -> bool:
    d = _dest(src.name)
    zip_path = d / "src.zip"
    print(f"[{src.name}] GET {src.url}")
    req = Request(src.url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as r, open(zip_path, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"[{src.name}] {zip_path.stat().st_size / 1e6:.0f} MB, unzipping")
    _unzip_all(d)
    print(f"[{src.name}] done -> {sorted(p.name for p in d.iterdir())[:8]}")
    return True


def _download_kaggle(src: Source) -> bool:
    d = _dest(src.name)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    print(f"[{src.name}] kaggle {src.kaggle}")
    api.dataset_download_files(src.kaggle, path=str(d), unzip=True, quiet=False)
    print(f"[{src.name}] done -> {sorted(p.name for p in d.iterdir())[:8]}")
    return True


def download(name: str) -> bool:
    src = SOURCES[name]
    if src.manual or src.kind == "manual":
        print(f"[{name}] MANUAL: {src.note}")
        return False
    try:
        return _download_direct(src) if src.kind == "direct" else _download_kaggle(src)
    except Exception as e:  # noqa: BLE001 — report and continue with the rest
        print(f"[{name}] FAILED: {e}")
        return False


def main() -> int:
    names = sys.argv[1:] or [n for n, s in SOURCES.items() if not s.manual and s.kind != "manual"]
    ok = {n: download(n) for n in names}
    print("\n=== summary ===")
    for n, good in ok.items():
        print(f"  {n:14s} {'ok' if good else 'manual/failed'}")
    manual = [n for n, s in SOURCES.items() if (s.manual or s.kind == 'manual')]
    print(f"\nManual (see notes above): {', '.join(manual)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
