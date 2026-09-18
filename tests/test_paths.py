"""The package has one stable, environment-overridable path contract."""

from pathlib import Path

from halo import paths


def test_default_paths_are_rooted_in_the_checkout():
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert paths.DATASETS_DIR == paths.REPO_ROOT / "data" / "datasets"
    assert paths.PRETRAINING_DIR == paths.REPO_ROOT / "data" / "pretraining"
    assert paths.RUNS_DIR == paths.REPO_ROOT / "runs"
    assert paths.CACHE_DIR == paths.REPO_ROOT / "cache"


def test_path_contract_contains_no_relative_runtime_locations():
    for name in ("REPO_ROOT", "DATASETS_DIR", "PRETRAINING_DIR", "RUNS_DIR", "CACHE_DIR",
                 "RESULTS_DIR", "REFERENCES_DIR"):
        assert isinstance(getattr(paths, name), Path)
        assert getattr(paths, name).is_absolute()
