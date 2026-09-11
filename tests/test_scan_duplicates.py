"""Regressions for the byte-identical-window screen (data/scripts/scan_duplicates)."""

import json

import numpy as np
import pytest

from data.scripts.scan_duplicates import OUT, scan_stream


def _grid(windows):
    return np.asarray(windows, dtype=np.float32)


def test_unique_windows_are_never_dropped():
    rng = np.random.default_rng(0)
    data = _grid(rng.normal(size=(8, 30, 6)))
    drop, groups, conflict = scan_stream(data, ["walking"] * 8)
    assert drop == [] and groups == 0 and conflict == 0


def test_same_label_group_keeps_exactly_one_genuine_observation():
    rng = np.random.default_rng(1)
    stale = rng.normal(size=(30, 6)).astype(np.float32)
    data = _grid([stale, rng.normal(size=(30, 6)), stale, stale])
    drop, groups, conflict = scan_stream(data, ["sitting"] * 4)
    assert drop == [2, 3]          # index 0 survives, its two repeats do not
    assert (groups, conflict) == (1, 0)


def test_label_conflicted_group_is_dropped_whole():
    """The ExtraSensory case: the Pebble buffer went stale across a label change, so the
    window is real but its label is unknowable — keeping any member injects a wrong label."""
    rng = np.random.default_rng(2)
    stale = rng.normal(size=(30, 6)).astype(np.float32)
    data = _grid([stale, stale, rng.normal(size=(30, 6))])
    drop, groups, conflict = scan_stream(data, ["lying", "sitting", "walking"])
    assert drop == [0, 1]          # NOT [1] — the survivor would be a coin flip
    assert (groups, conflict) == (1, 1)


def test_scan_spans_hash_block_boundaries():
    """Duplicates must be found even when the pair straddles the 4096-window read block."""
    from data.scripts import scan_duplicates

    rng = np.random.default_rng(3)
    data = _grid(rng.normal(size=(20, 4, 6)))
    data[19] = data[0]
    monkey = scan_duplicates.BLOCK
    try:
        scan_duplicates.BLOCK = 8          # force three blocks
        drop, groups, _ = scan_duplicates.scan_stream(data, ["walking"] * 20)
    finally:
        scan_duplicates.BLOCK = monkey
    assert drop == [19] and groups == 1


def test_cached_scan_covers_the_stale_pebble_stream():
    """The committed cache must actually carry the ExtraSensory finding, not an empty dict."""
    if not OUT.exists():
        pytest.skip("duplicate scan has not been run in this checkout")
    blob = json.loads(OUT.read_text())
    assert blob["alignment"] == "native"
    windows = blob["windows"]
    if "extrasensory/watch_wrist" not in windows:
        pytest.skip("extrasensory grids are not built in this checkout")
    assert len(windows["extrasensory/watch_wrist"]) > 10_000
    assert len(set(windows["extrasensory/watch_wrist"])) == \
        len(windows["extrasensory/watch_wrist"])       # indices are unique


def test_load_require_refuses_to_silently_return_an_empty_screen(tmp_path, monkeypatch):
    """A missing cache must not look like 'this corpus has no duplicates'."""
    from data.scripts import scan_duplicates as sd

    monkeypatch.setattr(sd, "OUT", tmp_path / "absent.json")
    assert sd.load(require=False) == {}
    with pytest.raises(FileNotFoundError):
        sd.load(require=True)

    (tmp_path / "absent.json").write_text(json.dumps({"alignment": "harmonised", "windows": {}}))
    assert sd.load("native", require=False) == {}
    with pytest.raises(ValueError):
        sd.load("native", require=True)


def test_load_require_rejects_a_cache_for_old_grid_content(tmp_path, monkeypatch):
    from data.scripts import scan_duplicates as sd
    from data.scripts.eda import grid_io

    cache = tmp_path / "duplicate_windows.json"
    cache.write_text(json.dumps({
        "alignment": "native", "grid_fingerprint": "old", "windows": {},
    }))
    monkeypatch.setattr(sd, "OUT", cache)
    monkeypatch.setattr(grid_io, "grid_corpus_fingerprint", lambda alignment: "current")

    assert sd.load("native", require=False) == {}
    with pytest.raises(ValueError, match="does not match"):
        sd.load("native", require=True)


def test_duplicate_cache_paths_do_not_overwrite_alignments(tmp_path, monkeypatch):
    from data.scripts import scan_duplicates as sd
    monkeypatch.setattr(sd, "OUT", tmp_path / "duplicate_windows.json")
    assert sd.cache_path("native").name == "duplicate_windows.json"
    assert sd.cache_path("non_harmonised").name == "duplicate_windows_non_harmonised.json"


def test_duplicate_cache_allows_absent_optional_cached_streams(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from data.scripts import scan_duplicates as sd
    from data.scripts.eda import grid_io
    cache = tmp_path / "duplicate_windows.json"
    cache.write_text(json.dumps({
        "alignment": "native",
        "stream_fingerprints": {"required/stream": "ok", "optional/stream": "extra"},
        "windows": {},
    }))
    monkeypatch.setattr(sd, "OUT", cache)
    monkeypatch.setattr(grid_io, "discover_grids",
                        lambda alignment: [SimpleNamespace(key="required/stream")])
    monkeypatch.setattr(grid_io, "grid_corpus_fingerprint",
                        lambda alignment, refs=None: "ok")
    assert sd.load("native", require=True) == {}


def test_incremental_duplicate_refresh_preserves_only_fingerprinted_streams(tmp_path, monkeypatch):
    """Incremental quality refresh is fail-closed for untouched grids."""
    from types import SimpleNamespace
    from data.scripts import scan_duplicates as sd
    from data.scripts.eda import grid_io

    repeated = np.zeros((2, 4, 6), dtype=np.float32)
    changed = SimpleNamespace(
        key="changed/watch", dataset="changed", n_windows=2, labels=("x", "x"),
        load_data=lambda: repeated, load_lengths=lambda: np.array([4, 4], dtype=np.int32),
    )
    stable = SimpleNamespace(key="stable/watch", dataset="stable", n_windows=0)
    cache = tmp_path / "duplicate_windows.json"
    cache.write_text(json.dumps({
        "alignment": "native", "stream_fingerprints": {"stable/watch": "stable-fp"},
        "windows": {"stable/watch": [9]}, "summary": [],
    }))
    monkeypatch.setattr(sd, "OUT", cache)
    monkeypatch.setattr(grid_io, "discover_grids", lambda alignment: [changed, stable])
    monkeypatch.setattr(
        grid_io, "grid_corpus_fingerprint",
        lambda alignment, refs=None: "all-fp" if refs is None else f"{refs[0].dataset}-fp",
    )
    blob = sd.scan("native", ["changed"])
    assert blob["windows"] == {"stable/watch": [9], "changed/watch": [1]}

    cache.write_text(json.dumps({
        "alignment": "native", "stream_fingerprints": {"stable/watch": "outdated"},
        "windows": {}, "summary": [],
    }))
    with pytest.raises(ValueError, match="unselected streams have changed"):
        sd.scan("native", ["changed"])


def test_orphan_session_directories_are_not_ingested(tmp_path, monkeypatch):
    """A converter re-run that drops sessions leaves the old directories behind, and the grid
    builder's glob would happily ingest them — defeating the very fix that dropped them. MM-Fit's
    gap fix removed 61 labelled sets; without this guard all 61 would have survived in the grid.
    `labels.json` is the converter's declaration of what it emitted, so it is the authority.
    """
    import json
    import pandas as pd
    from data.scripts import build_grids
    from data.scripts.curate.deployment_policy import stream_specs

    root = tmp_path / "data" / "datasets" / "mmfit"
    (root / "sessions").mkdir(parents=True)
    real = json.loads((build_grids.REPO / "data" / "datasets" / "mmfit" / "labels.json").read_text())
    kept = sorted(real)[0]
    for name in (kept, "w99_orphan_from_an_earlier_run_00"):
        target = root / "sessions" / name
        target.mkdir()
        pd.DataFrame({
            "timestamp_sec": [0.0, 0.01],
            **{f"left_wrist_{axis}": [0.0, 0.0]
               for axis in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")},
            "activity": ["squats"] * 2,
            "subject": ["w00"] * 2,
        }).to_parquet(target / "data.parquet", index=False)
    (root / "labels.json").write_text(json.dumps({kept: ["squats"]}))
    (root / "metadata.json").write_text(json.dumps({"sampling_rate_hz": 100.0}))
    monkeypatch.setattr(build_grids, "REPO", tmp_path)

    spec = next(s for s in stream_specs("mmfit", None) if s.stream_id == "left_wrist")
    seen = [frame.attrs["halo_session_id"] for frame, _, _ in build_grids.iter_sessions("mmfit", spec)]
    assert seen == [kept], f"orphan directory was ingested: {seen}"
