import json
from pathlib import Path

import numpy as np

from data.scripts.eda.grid_io import (
    discover_grids,
    matching_refs,
    output_subdir,
    sample_indices,
    triad_indices,
)


def _write_grid(root: Path, dataset: str, stream: str, mask: list[bool], *,
                duration_dir: str | None = None) -> None:
    grid = root / dataset / "grids" / "harmonised" / stream
    if duration_dir is not None:
        grid /= duration_dir
    grid.mkdir(parents=True, exist_ok=True)
    labels = ["walking", "walking", "sitting", "walking"]
    subjects = ["s1", "s1", "s2", "s3"]
    data = np.arange(4 * 8 * 6, dtype=np.float32).reshape(4, 8, 6)
    np.save(grid / "data.npy", data)
    np.save(grid / "mask.npy", np.asarray(mask, dtype=bool))
    (grid / "meta.json").write_text(json.dumps({
        "dataset": dataset,
        "stream_id": stream,
        "alignment": "harmonised",
        "rate_hz": 4.0,
        "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "labels": labels,
        "subjects": subjects,
    }))


def test_discovery_modalities_and_selector_order(tmp_path: Path) -> None:
    _write_grid(tmp_path, "alpha", "phone", [1, 1, 1, 1, 1, 1])
    _write_grid(tmp_path, "beta", "watch", [1, 1, 1, 0, 0, 0])

    refs = discover_grids(datasets_dir=tmp_path)

    assert [ref.key for ref in refs] == ["alpha/phone", "beta/watch"]
    assert triad_indices(refs[0], "gyro") == (3, 4, 5)
    assert triad_indices(refs[1], "gyro") is None
    selected = matching_refs(refs, "walking", ["beta/watch", "alpha/phone"])
    assert [ref.key for ref in selected] == ["beta/watch", "alpha/phone"]


def test_watch_stream_is_not_mistaken_for_duration_directory(tmp_path: Path) -> None:
    _write_grid(tmp_path, "alpha", "watch_wrist", [1, 1, 1, 0, 0, 0])
    _write_grid(tmp_path, "beta", "phone", [1, 1, 1, 0, 0, 0], duration_dir="w4")

    refs = discover_grids(datasets_dir=tmp_path)

    assert [ref.key for ref in refs] == ["alpha/watch_wrist"]
    assert [ref.key for ref in discover_grids(datasets_dir=tmp_path, window_seconds=4)] == ["beta/phone"]


def test_default_never_promotes_a_longer_window_to_six_seconds(tmp_path: Path) -> None:
    _write_grid(tmp_path, "alpha", "phone", [1, 1, 1, 0, 0, 0], duration_dir="w16")
    _write_grid(tmp_path, "beta", "watch", [1, 1, 1, 0, 0, 0], duration_dir="w6")
    assert [ref.key for ref in discover_grids(datasets_dir=tmp_path)] == ["beta/watch"]


def test_explicit_six_second_discovery_prefers_w6_and_falls_back_per_stream(tmp_path: Path) -> None:
    _write_grid(tmp_path, "alpha", "watch_wrist", [1, 1, 1, 0, 0, 0])
    _write_grid(tmp_path, "alpha", "watch_wrist", [1, 1, 1, 1, 1, 1], duration_dir="w6")
    _write_grid(tmp_path, "beta", "phone", [1, 1, 1, 0, 0, 0])

    refs = discover_grids(datasets_dir=tmp_path, window_seconds=6.0)

    assert [(ref.key, ref.grid_dir.name) for ref in refs] == [
        ("alpha/watch_wrist", "w6"), ("beta/phone", "phone"),
    ]
    implicit = discover_grids(datasets_dir=tmp_path)
    assert [(ref.key, ref.grid_dir.name) for ref in implicit] == [
        ("alpha/watch_wrist", "w6"), ("beta/phone", "phone"),
    ]


def test_sampling_is_deterministic_and_prefers_distinct_subjects(tmp_path: Path) -> None:
    _write_grid(tmp_path, "alpha", "phone", [1, 1, 1, 1, 1, 1])
    ref = discover_grids(datasets_dir=tmp_path)[0]

    first = sample_indices(ref, "walking", count=2, seed=42)
    second = sample_indices(ref, "walking", count=2, seed=42)

    assert np.array_equal(first, second)
    assert len({ref.subjects[index] for index in first}) == 2


def test_output_subdir_creates_nested_analysis_directory(tmp_path: Path) -> None:
    destination = output_subdir(tmp_path, "activity_signatures", "summaries")

    assert destination == tmp_path / "activity_signatures" / "summaries"
    assert destination.is_dir()
