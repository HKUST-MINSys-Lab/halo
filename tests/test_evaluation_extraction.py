"""Phase 0 gate: the evaluation/ package is an exact extraction of sealed_eval.py, not a rewrite.

Three fast checks run everywhere; the fourth (bit-exact results on a cached sealed cell) needs a
golden artifact produced at the anchor tag and is skipped with instructions when it is absent.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_TAG = "hist/v3-support-conditioned/pre-evaluation-package-20260923"
SEALED_EVAL = "training/support_classifier/sealed_eval.py"
MOVED = {
    "features": [
        "feature_cache_schema", "FeatureMemoryCache", "_cache_key", "_file_hash_for_stat",
        "_file_hash", "_halo_features", "_baseline_feature_state", "_load_or_encode",
    ],
    "manifests": [
        "QueryPlan", "sealed_cells", "duration_cells", "evaluation_cells", "_aligned_labels",
        "_stable_choice", "build_manifest", "manifest_fingerprint",
    ],
    "zero_shot": ["_normalise", "_training_bank_conse_predictions", "_build_training_reference_bank"],
    "provenance": ["validate_result_rows", "_atomic_json", "_run_provenance"],
}
MOVED_CONSTANTS = {"features": ["FEATURE_CACHE_SCHEMA", "UNCHANGED_BASELINE_FEATURE_CACHE_SCHEMA"],
                   "manifests": ["SEED"], "zero_shot": ["TRAINING_BANK_ZERO_SHOT"]}


def _definition_source(text: str, name: str) -> str:
    """Exact source of one top-level definition, decorators included."""
    tree = ast.parse(text)
    lines = text.split("\n")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return "\n".join(lines[start - 1:node.end_lineno])
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    raise KeyError(name)


def _anchor_source() -> str | None:
    try:
        return subprocess.run(
            ["git", "show", f"{ANCHOR_TAG}:{SEALED_EVAL}"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def test_every_moved_definition_is_byte_identical_to_the_anchor_tag():
    anchor = _anchor_source()
    if anchor is None:
        pytest.skip(f"anchor tag {ANCHOR_TAG} is not available in this checkout")
    for module, names in list(MOVED.items()) + list(MOVED_CONSTANTS.items()):
        new_text = (ROOT / "evaluation" / f"{module}.py").read_text()
        for name in names:
            assert _definition_source(new_text, name) == _definition_source(anchor, name), \
                f"{module}.{name} differs from the anchor-tag source: extraction retyped something"


def test_sealed_eval_reexports_every_moved_name_from_the_new_module():
    sealed = importlib.import_module("training.support_classifier.sealed_eval")
    for module, names in MOVED.items():
        target = importlib.import_module(f"evaluation.{module}")
        for name in names:
            assert getattr(sealed, name) is getattr(target, name), name
            assert getattr(target, name).__module__ == f"evaluation.{module}", name
    for module, names in MOVED_CONSTANTS.items():
        target = importlib.import_module(f"evaluation.{module}")
        for name in names:
            assert getattr(sealed, name) == getattr(target, name), name


def test_evaluation_package_never_imports_the_sealed_runner():
    for path in (ROOT / "evaluation").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            module = getattr(node, "module", None) or ""
            names = [a.name for a in getattr(node, "names", [])] if isinstance(node, ast.Import) else []
            assert "sealed_eval" not in module and not any("sealed_eval" in n for n in names), \
                f"{path.relative_to(ROOT)} imports sealed_eval: the dependency must be one-way"


def test_no_moved_definition_remains_in_sealed_eval():
    text = (ROOT / SEALED_EVAL).read_text()
    tree = ast.parse(text)
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    defined |= {n.targets[0].id for n in tree.body if isinstance(n, ast.Assign)
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    duplicated = defined & {n for ns in MOVED.values() for n in ns} \
        | defined & {n for ns in MOVED_CONSTANTS.values() for n in ns}
    assert not duplicated, f"still defined in sealed_eval as well as evaluation/: {sorted(duplicated)}"


@pytest.mark.slow
def test_sealed_results_on_a_cached_cell_are_bit_identical_to_the_anchor_golden(tmp_path):
    """Functional gate. Produce the golden ONCE at the anchor tag, then point at it:

        git worktree add /tmp/halo-anchor hist/v3-support-conditioned/pre-evaluation-package-20260923
        cd /tmp/halo-anchor && python -m training.support_classifier.sealed_eval \\
            --out /tmp/halo-anchor-golden --models halo harnet5 --k 0 8 --window-seconds 4 \\
            --halo-checkpoint <v4 checkpoint> --feature-cache <shared feature cache> --bootstrap 0
        HALO_EXTRACTION_GOLDEN=/tmp/halo-anchor-golden/results.json pytest -m slow tests/test_evaluation_extraction.py

    The same invocation is then run from this tree into ``tmp_path`` and every row compared. The
    cell restriction is applied through the same CLI, so the comparison is on identical manifests.
    """
    golden = os.environ.get("HALO_EXTRACTION_GOLDEN")
    if not golden:
        pytest.skip("set HALO_EXTRACTION_GOLDEN to a results.json produced at the anchor tag")
    invocation = os.environ.get("HALO_EXTRACTION_INVOCATION")
    if not invocation:
        pytest.skip("set HALO_EXTRACTION_INVOCATION to the exact sealed_eval argument string used for the golden")
    out = tmp_path / "current"
    subprocess.run(
        ["python", "-m", "training.support_classifier.sealed_eval", "--out", str(out), *invocation.split()],
        cwd=ROOT, check=True,
    )
    golden_rows = json.loads(Path(golden).read_text())
    current_rows = json.loads((out / "results.json").read_text())
    assert current_rows == golden_rows, "sealed results changed across the extraction"
