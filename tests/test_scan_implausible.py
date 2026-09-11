"""Regressions for fail-closed physical-plausibility cache loading."""

import json

import numpy as np
import pytest


def test_implausible_cache_is_required_and_grid_fingerprinted(tmp_path, monkeypatch):
    from data.scripts import scan_implausible as si
    from data.scripts.eda import grid_io

    cache = tmp_path / "implausible_windows.json"
    monkeypatch.setattr(si, "OUT", cache)
    with pytest.raises(FileNotFoundError):
        si.load(require=True)

    cache.write_text(json.dumps({
        "alignment": "native", "grid_fingerprint": "old", "windows": {},
    }))
    monkeypatch.setattr(grid_io, "grid_corpus_fingerprint", lambda alignment: "current")
    assert si.load(require=False) == {}
    with pytest.raises(ValueError, match="does not match"):
        si.load(require=True)


def test_implausible_cache_paths_do_not_overwrite_alignments(tmp_path, monkeypatch):
    from data.scripts import scan_implausible as si
    monkeypatch.setattr(si, "OUT", tmp_path / "implausible_windows.json")
    assert si.cache_path("native").name == "implausible_windows.json"
    assert si.cache_path("non_harmonised").name == "implausible_windows_non_harmonised.json"


def test_implausible_cache_allows_absent_optional_cached_streams(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from data.scripts import scan_implausible as si
    from data.scripts.eda import grid_io
    cache = tmp_path / "implausible_windows.json"
    cache.write_text(json.dumps({
        "alignment": "native",
        "stream_fingerprints": {"required/stream": "ok", "optional/stream": "extra"},
        "windows": {},
    }))
    monkeypatch.setattr(si, "OUT", cache)
    monkeypatch.setattr(grid_io, "discover_grids",
                        lambda alignment: [SimpleNamespace(key="required/stream")])
    monkeypatch.setattr(grid_io, "grid_corpus_fingerprint",
                        lambda alignment, refs=None: "ok")
    assert si.load("native", require=True) == {}


def test_implausible_scan_excludes_nonfinite_windows(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from data.scripts import scan_implausible as si
    from data.scripts.eda import grid_io

    data = np.zeros((3, 8, 6), dtype=np.float32)
    data[1, 2, 4] = np.nan
    ref = SimpleNamespace(
        key="example/watch", n_windows=3, mask=(True,) * 6,
        load_data=lambda: data,
    )
    monkeypatch.setattr(grid_io, "discover_grids", lambda alignment: [ref])
    monkeypatch.setattr(grid_io, "grid_corpus_fingerprint", lambda alignment, refs=None: "fp")
    blob = si.scan("native")
    assert blob["windows"] == {"example/watch": [1]}
    assert blob["summary"][0][-1] == 1


def test_incremental_implausible_refresh_preserves_only_fingerprinted_streams(tmp_path, monkeypatch):
    """A rebuilt source need not reread the corpus, but stale other sources must refuse."""
    from types import SimpleNamespace
    from data.scripts import scan_implausible as si
    from data.scripts.eda import grid_io

    changed = SimpleNamespace(
        key="changed/watch", dataset="changed", n_windows=1, mask=(True,) * 6,
        load_data=lambda: np.full((1, 8, 6), 17.0, dtype=np.float32),
    )
    stable = SimpleNamespace(key="stable/watch", dataset="stable", n_windows=0)
    cache = tmp_path / "implausible_windows.json"
    cache.write_text(json.dumps({
        "alignment": "native", "stream_fingerprints": {"stable/watch": "stable-fp"},
        "windows": {"stable/watch": [3]}, "summary": [],
    }))
    monkeypatch.setattr(si, "OUT", cache)
    monkeypatch.setattr(grid_io, "discover_grids", lambda alignment: [changed, stable])
    monkeypatch.setattr(
        grid_io, "grid_corpus_fingerprint",
        lambda alignment, refs=None: "all-fp" if refs is None else f"{refs[0].dataset}-fp",
    )
    blob = si.scan("native", ["changed"])
    assert blob["windows"] == {"stable/watch": [3], "changed/watch": [0]}

    cache.write_text(json.dumps({
        "alignment": "native", "stream_fingerprints": {"stable/watch": "outdated"},
        "windows": {}, "summary": [],
    }))
    with pytest.raises(ValueError, match="unselected streams have changed"):
        si.scan("native", ["changed"])
