"""Regenerate the journal index from dated Markdown entries."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "docs" / "journal"

PREAMBLE = """# Project journal

A dated, append-only record of architecture pivots, experiments, findings, and decisions. Historical
entries preserve what was believed at the time; current behavior is defined only by
[`docs/contracts/`](../contracts/) and [`docs/overview/`](../overview/).

## Rules

- Use one `YYYY-MM-DD-short-slug.md` file per entry.
- Never silently rewrite a historical conclusion. Add a later entry that names what it supersedes.
- Keep entries self-contained and link to code, artifacts, contracts, or commits.
- Run `python tools/generate_journal_index.py` after adding or moving an entry.

## Index

| date | entry | title |
|---|---|---|
"""


def title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# "):
            return line[2:].strip().replace("|", "\\|")
    return path.stem


def main() -> None:
    rows = []
    for path in sorted(JOURNAL.glob("20??-??-??-*.md")):
        rows.append(f"| {path.name[:10]} | [{path.name}]({path.name}) | {title(path)} |")
    (JOURNAL / "README.md").write_text(PREAMBLE + "\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
