from __future__ import annotations

import re
from pathlib import Path

from data.scripts.curate.deployment_policy import (
    LABEL_FREE_PRETRAIN_DATASETS,
    SEALED_TEST_EVAL_DATASETS,
    SUPERVISED_HEAD_TRAIN_DATASETS,
)


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_live_markdown_links_resolve() -> None:
    roots = [DOCS / "README.md", *(DOCS / "overview").glob("*.md"), *(DOCS / "contracts").glob("*.md")]
    failures: list[str] = []
    for path in roots:
        text = path.read_text(encoding="utf-8")
        for raw_target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = raw_target.split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            if not (path.parent / target).resolve().exists():
                failures.append(f"{path.relative_to(ROOT)} -> {raw_target}")
    assert not failures, "broken live documentation links:\n" + "\n".join(failures)


def test_design_contract_matches_active_patch_spans() -> None:
    design = (DOCS / "contracts" / "design_of_record.md").read_text(encoding="utf-8")
    assert "0.5, 1.0, 2.0, and 4.0 seconds" in design
    assert "0.5, 1.0, and 1.5 seconds" not in design


def test_dataset_contract_matches_code_rosters() -> None:
    ledger = (DOCS / "contracts" / "data_policy.md").read_text(encoding="utf-8")
    design = (DOCS / "contracts" / "design_of_record.md").read_text(encoding="utf-8")
    for dataset in LABEL_FREE_PRETRAIN_DATASETS:
        assert f"`{dataset}`" in ledger
    for dataset in SUPERVISED_HEAD_TRAIN_DATASETS:
        assert f"`{dataset}`" in design
    for dataset in SEALED_TEST_EVAL_DATASETS:
        assert f"`{dataset}`" in design


def test_no_living_document_is_left_in_design_directory() -> None:
    design_dir = DOCS / "design"
    assert not design_dir.exists() or not list(design_dir.glob("*.md"))
