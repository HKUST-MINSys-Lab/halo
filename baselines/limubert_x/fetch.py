"""Fetch the author-released LiMU-BERT-X checkpoint with integrity verification."""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path

from .adapter import CHECKPOINT

URL = "https://raw.githubusercontent.com/WANDS-HKUST/LIMU-BERT_Experience/main/weights/limu_bert_x.pt"
SHA256 = "65697e7017020f36429a221382253f1fc370fdfd493cfb71667f95846c18516c"


def fetch(destination: Path = CHECKPOINT) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256:
        return destination
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        with urllib.request.urlopen(URL, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest != SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"LiMU-BERT-X checkpoint checksum mismatch: {digest}")
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    print(fetch())
