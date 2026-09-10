"""Offline tests for the Ego-Exo4D head-IMU pretraining module.

Ego-Exo4D is licence-gated and ~1 TB, and `projectaria_tools` is not installed in
this tree, so every test here runs against a fake VRS provider injected into
`sys.modules`. The point is that the whole convert path -- stream selection, unit
conversion, anti-aliasing, resampling, gap splitting, and the on-disk contract --
is exercised without the library or the data.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.pretraining.ego_exo4d import convert as convert_mod
from data.pretraining.ego_exo4d import fetch as fetch_mod

RATE_OUT = convert_mod.RATE_HZ
NATIVE_HZ = 800.0


# --------------------------------------------------------------------------
# Fake projectaria_tools
# --------------------------------------------------------------------------


class _FakeStreamId:
    def __init__(self, raw: str) -> None:
        self.raw = str(raw)

    def __eq__(self, other) -> bool:
        return isinstance(other, _FakeStreamId) and other.raw == self.raw

    def __hash__(self) -> int:
        return hash(self.raw)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StreamId({self.raw})"


class _FakeImuRecord:
    def __init__(self, t_ns: float, accel, gyro) -> None:
        self.capture_timestamp_ns = t_ns
        self.accel_msec2 = accel
        self.gyro_radsec = gyro


class _FakeProvider:
    """Serves one synthetic IMU stream keyed by its Aria label."""

    def __init__(self, streams: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        self._streams = streams
        self._labels = {"imu-right": "1202-1", "imu-left": "1202-2"}

    def get_stream_id_from_label(self, label: str):
        if label not in self._streams:
            return None
        return _FakeStreamId(self._labels[label])

    def _by_id(self, stream_id):
        for label, raw in self._labels.items():
            if getattr(stream_id, "raw", None) == raw and label in self._streams:
                return self._streams[label]
        raise KeyError(stream_id)

    def get_num_data(self, stream_id) -> int:
        try:
            return len(self._by_id(stream_id)[0])
        except KeyError:
            return 0

    def get_imu_data_by_index(self, stream_id, index: int):
        stamps, accel, gyro = self._by_id(stream_id)
        return _FakeImuRecord(stamps[index], accel[index], gyro[index])


def _install_fake_aria(monkeypatch, provider_factory) -> None:
    root = types.ModuleType("projectaria_tools")
    core = types.ModuleType("projectaria_tools.core")
    data_provider = types.ModuleType("projectaria_tools.core.data_provider")
    stream_id_mod = types.ModuleType("projectaria_tools.core.stream_id")

    data_provider.create_vrs_data_provider = lambda path: provider_factory(path)
    stream_id_mod.StreamId = _FakeStreamId
    core.data_provider = data_provider
    core.stream_id = stream_id_mod
    root.core = core

    for name, module in {
        "projectaria_tools": root,
        "projectaria_tools.core": core,
        "projectaria_tools.core.data_provider": data_provider,
        "projectaria_tools.core.stream_id": stream_id_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


# --------------------------------------------------------------------------
# Synthetic signal helpers
# --------------------------------------------------------------------------


def _stream(duration_s: float = 12.0, rate: float = NATIVE_HZ, gap_at: float | None = None,
            gap_len: float = 0.0, tone_hz: float | None = None, tone_amp: float = 1.0,
            start_ns: float = 1_234_567_890.0):
    """Synthetic IMU: stationary 1 g on z, optional tone on x, optional dropout."""
    t = np.arange(0.0, duration_s, 1.0 / rate)
    if gap_at is not None:
        t = t[(t < gap_at) | (t >= gap_at + gap_len)]
    accel = np.zeros((len(t), 3))
    accel[:, 2] = convert_mod.GRAVITY_MS2  # a still head: 1 g on one axis
    if tone_hz is not None:
        # phase offset: a tone whose samples land on grid zero-crossings would make the
        # anti-alias assertion vacuous, so never test one.
        accel[:, 0] = tone_amp * convert_mod.GRAVITY_MS2 * np.sin(2 * np.pi * tone_hz * t + 0.7)
    gyro = np.zeros((len(t), 3))
    gyro[:, 1] = 0.25
    return t * 1e9 + start_ns, accel, gyro


def _take(uid: str = "take-0001", participant=657, capture="cap-abc", name="cmu_bike01_2"):
    return {
        "take_uid": uid,
        "take_name": name,
        "root_dir": f"takes/{name}",
        "capture_uid": capture,
        "participant_uid": participant,
        "university_name": "cmu",
        "duration_sec": 60.0,
        "capture": {"cameras": [{"cam_id": "aria01", "is_ego": True, "device_type": "aria"}]},
    }


def _stage_take(root: Path, take: dict) -> Path:
    take_dir = root / take["root_dir"]
    take_dir.mkdir(parents=True, exist_ok=True)
    vrs = take_dir / "aria01_noimagestreams.vrs"
    vrs.write_bytes(b"fake-vrs")
    return vrs


def _write_metadata(root: Path, takes: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "takes.json").write_text(json.dumps(takes))


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


def test_stationary_imu_reads_one_g():
    stamps, accel, gyro = _stream(duration_s=10.0)
    frames = convert_mod.session_frames((stamps - stamps[0]) / 1e9, accel, gyro, "p657")
    assert len(frames) == 1
    acc = frames[0][list(convert_mod.ACC_COLUMNS)].to_numpy(np.float64)
    assert convert_mod.stationary_gravity_g(acc) == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(acc, axis=1).mean() == pytest.approx(1.0, abs=1e-3)


def test_gyro_passes_through_in_rad_per_second():
    stamps, accel, gyro = _stream(duration_s=10.0)
    frames = convert_mod.session_frames((stamps - stamps[0]) / 1e9, accel, gyro, "p657")
    gy = frames[0]["gyro_y"].to_numpy(np.float64)
    assert gy.mean() == pytest.approx(0.25, abs=1e-3)


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------


def test_resample_lands_on_a_uniform_200hz_grid():
    stamps, accel, gyro = _stream(duration_s=10.0)
    t = (stamps - stamps[0]) / 1e9
    grid, values = convert_mod.resample_uniform(t, np.hstack([accel, gyro]))
    steps = np.diff(grid)
    assert grid[0] == 0.0
    assert np.allclose(steps, 1.0 / RATE_OUT, atol=1e-12)
    assert len(grid) == pytest.approx(10.0 * RATE_OUT, abs=2)
    assert values.shape == (len(grid), 6)


def test_resample_absorbs_timestamp_jitter():
    rng = np.random.default_rng(0)
    stamps, accel, gyro = _stream(duration_s=8.0)
    t = (stamps - stamps[0]) / 1e9
    t = np.sort(t + rng.normal(0.0, 1e-4, len(t)))  # device-clock jitter
    t -= t[0]
    grid, _ = convert_mod.resample_uniform(t, np.hstack([accel, gyro]))
    assert np.allclose(np.diff(grid), 1.0 / RATE_OUT, atol=1e-12)


@pytest.mark.parametrize("tone_hz", [260.0, 310.0, 390.0])
def test_high_frequency_tone_is_attenuated(tone_hz):
    """Tones legal at 800 Hz must not fold back onto the 200 Hz grid.

    260 Hz would alias to 60 Hz at full amplitude without the low-pass, so this
    fails loudly if the anti-alias step is ever removed.
    """
    stamps, accel, gyro = _stream(duration_s=10.0, tone_hz=tone_hz)
    t = (stamps - stamps[0]) / 1e9
    grid, values = convert_mod.resample_uniform(t, np.hstack([accel / convert_mod.GRAVITY_MS2, gyro]))
    interior = slice(int(RATE_OUT), -int(RATE_OUT))  # ignore FIR edge transients
    assert np.abs(values[interior, 0]).max() < 0.05  # input amplitude was 1.0 g


def test_low_frequency_tone_survives_with_amplitude():
    stamps, accel, gyro = _stream(duration_s=10.0, tone_hz=5.0)
    t = (stamps - stamps[0]) / 1e9
    grid, values = convert_mod.resample_uniform(t, np.hstack([accel / convert_mod.GRAVITY_MS2, gyro]))
    interior = slice(int(RATE_OUT), -int(RATE_OUT))
    assert np.abs(values[interior, 0]).max() == pytest.approx(1.0, rel=0.03)


def test_antialias_is_zero_phase_in_the_passband():
    stamps, accel, _ = _stream(duration_s=6.0, tone_hz=5.0)
    t = (stamps - stamps[0]) / 1e9
    filtered = convert_mod.antialias(accel[:, [0]], convert_mod.native_rate(t))
    interior = slice(int(NATIVE_HZ), -int(NATIVE_HZ))
    reference = accel[interior, 0]
    assert np.corrcoef(filtered[interior, 0], reference)[0, 1] > 0.999


# --------------------------------------------------------------------------
# Gap splitting
# --------------------------------------------------------------------------


def test_gap_longer_than_half_a_second_splits_the_session():
    stamps, accel, gyro = _stream(duration_s=20.0, gap_at=10.0, gap_len=1.0)
    t = (stamps - stamps[0]) / 1e9
    assert len(convert_mod.split_on_gaps(t)) == 2
    frames = convert_mod.session_frames(t, accel, gyro, "p657")
    assert len(frames) == 2
    for frame in frames:
        assert frame["timestamp_sec"].iloc[0] == 0.0
        assert np.allclose(np.diff(frame["timestamp_sec"].to_numpy()), 1.0 / RATE_OUT, atol=1e-12)
    # No sample bridges the dropout: each part is shorter than the wall-clock span.
    assert sum(len(f) for f in frames) < 20.0 * RATE_OUT


def test_short_gap_does_not_split():
    stamps, accel, gyro = _stream(duration_s=20.0, gap_at=10.0, gap_len=0.2)
    t = (stamps - stamps[0]) / 1e9
    assert len(convert_mod.split_on_gaps(t)) == 1
    assert len(convert_mod.session_frames(t, accel, gyro, "p657")) == 1


def test_segments_shorter_than_one_window_are_dropped():
    stamps, accel, gyro = _stream(duration_s=13.0, gap_at=3.0, gap_len=1.0)
    t = (stamps - stamps[0]) / 1e9
    assert len(convert_mod.split_on_gaps(t)) == 2  # 3 s + 9 s
    frames = convert_mod.session_frames(t, accel, gyro, "p657")
    assert len(frames) == 1  # the 3 s part is below the 6 s window
    assert len(frames[0]) >= convert_mod.MIN_SESSION_SECONDS * RATE_OUT


def test_split_parts_keep_the_stream_token_matchable():
    single = convert_mod.session_id("take-1", "p657", 1, 1)
    part_two = convert_mod.session_id("take-1", "p657", 2, 3)
    assert single == "egoexo_take-1_p657_aria_head"
    assert convert_mod.STREAM_TOKEN in part_two
    assert part_two.startswith("egoexo_take-1_p657_aria_head")


# --------------------------------------------------------------------------
# Subject identity
# --------------------------------------------------------------------------


def test_subject_prefers_participant_uid():
    assert convert_mod.subject_id(_take()) == ("p657", "participant_uid")


def test_subject_falls_back_to_capture_uid():
    subject, source = convert_mod.subject_id(_take(participant=None))
    assert subject == "capture_cap-abc"
    assert source == "capture_uid_fallback"


def test_subject_never_invented_per_take():
    with pytest.raises(ValueError, match="invent a per-take subject id"):
        convert_mod.subject_id(_take(participant=None, capture=None))


def test_two_takes_of_one_participant_share_a_subject():
    a = convert_mod.subject_id(_take(uid="t1"))[0]
    b = convert_mod.subject_id(_take(uid="t2", name="cmu_bike01_3"))[0]
    assert a == b  # subject-disjoint splits cannot leak a person across folds


# --------------------------------------------------------------------------
# IMU stream selection
# --------------------------------------------------------------------------


def test_imu_right_is_preferred(monkeypatch):
    right = _stream(duration_s=8.0)
    left = _stream(duration_s=8.0, rate=1000.0)
    _install_fake_aria(monkeypatch, lambda path: _FakeProvider({"imu-right": right, "imu-left": left}))
    provider = convert_mod.open_provider(Path("x.vrs"))
    label, _, count = convert_mod.resolve_imu_stream(provider)
    assert label == "imu-right"
    assert count == len(right[0])


def test_falls_back_to_imu_left(monkeypatch):
    left = _stream(duration_s=8.0, rate=1000.0)
    _install_fake_aria(monkeypatch, lambda path: _FakeProvider({"imu-left": left}))
    provider = convert_mod.open_provider(Path("x.vrs"))
    label, _, _ = convert_mod.resolve_imu_stream(provider)
    assert label == "imu-left"


def test_missing_projectaria_tools_names_the_pip_package(monkeypatch):
    for name in list(sys.modules):
        if name.startswith("projectaria_tools"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    real_import = importlib.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("projectaria_tools"):
            raise ImportError("No module named 'projectaria_tools'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(convert_mod.MissingDependency, match="pip install projectaria-tools"):
        convert_mod.open_provider(Path("x.vrs"))


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


@pytest.fixture()
def converted(tmp_path, monkeypatch):
    raw, out = tmp_path / "downloads", tmp_path / "out"
    takes = [
        _take(uid="take-0001", participant=657, capture="cap-a", name="cmu_bike01_1"),
        _take(uid="take-0002", participant=657, capture="cap-a", name="cmu_bike01_2"),
        _take(uid="take-0003", participant=None, capture="cap-b", name="unc_cook01_1"),
    ]
    _write_metadata(raw, takes)
    for take in takes:
        _stage_take(raw, take)
    streams = {
        "cmu_bike01_1": _stream(duration_s=20.0),
        "cmu_bike01_2": _stream(duration_s=20.0, gap_at=10.0, gap_len=1.0),
        "unc_cook01_1": _stream(duration_s=20.0, rate=1000.0),
    }

    def factory(path):
        name = Path(path).parent.name
        label = "imu-left" if name == "unc_cook01_1" else "imu-right"
        return _FakeProvider({label: streams[name]})

    _install_fake_aria(monkeypatch, factory)
    assert convert_mod.convert(raw_dir=raw, output_dir=out, max_hours_per_take=1.0)
    return out


def test_parquet_columns_and_dtypes(converted):
    sessions = sorted((converted / "sessions").iterdir())
    assert sessions, "no sessions written"
    for session in sessions:
        assert convert_mod.STREAM_TOKEN in session.name
        assert session.name.startswith("egoexo_")
        frame = pd.read_parquet(session / "data.parquet")
        assert list(frame.columns) == [
            "timestamp_sec", "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z", "subject",
        ]
        assert frame["timestamp_sec"].dtype == np.float64
        for column in convert_mod.SIGNAL_COLUMNS:
            assert frame[column].dtype == np.float32
        assert pd.api.types.is_object_dtype(frame["subject"]) or pd.api.types.is_string_dtype(
            frame["subject"]
        )
        assert frame["subject"].nunique() == 1
        stamps = frame["timestamp_sec"].to_numpy()
        assert stamps[0] == 0.0
        assert np.all(np.diff(stamps) > 0)
        assert np.allclose(np.diff(stamps), 1.0 / RATE_OUT, atol=1e-12)


def test_labels_json_marks_every_session_unlabeled(converted):
    labels = json.loads((converted / "labels.json").read_text())
    sessions = {path.name for path in (converted / "sessions").iterdir()}
    assert set(labels) == sessions
    assert all(value == ["__unlabeled__"] for value in labels.values())


def test_manifest_keys_and_values(converted):
    manifest = json.loads((converted / "manifest.json").read_text())
    for key in (
        "dataset_name", "source", "num_subjects", "sampling_rate_hz",
        "channels", "unit", "gravity_state", "phase_a_only", "note",
    ):
        assert key in manifest
    assert manifest["sampling_rate_hz"] == 200.0
    assert manifest["unit"] == "g"
    assert manifest["gravity_state"] == "present"
    assert manifest["phase_a_only"] is True
    assert manifest["channels"] == list(convert_mod.SIGNAL_COLUMNS)
    assert manifest["num_subjects"] == 2  # p657 (two takes) + the capture-level fallback
    assert manifest["subject_id_source"] == "mixed"
    assert manifest["subject_id_fallback_takes"] == 1
    assert manifest["imu_streams"] == {"imu-right": 2, "imu-left": 1}
    assert manifest["gap_split_seconds"] == 0.5
    assert manifest["stationary_gravity_g_median"] == pytest.approx(1.0, abs=1e-2)
    assert "capture-level subject id" in manifest["note"]


def test_metadata_keys_and_values(converted):
    metadata = json.loads((converted / "metadata.json").read_text())
    for key in (
        "dataset", "display_name", "sampling_rate_hz", "pre_windowed", "streaming_grid",
        "role", "phase_a_only", "activities", "num_subjects", "channels",
        "core_channels", "extra_channels", "grid_dtype", "placement", "note",
    ):
        assert key in metadata
    assert metadata["dataset"] == "ego_exo4d"
    assert metadata["sampling_rate_hz"] == 200.0
    assert metadata["pre_windowed"] is False
    assert metadata["streaming_grid"] is True
    assert metadata["role"] == "pretrain_scale"
    assert metadata["phase_a_only"] is True
    assert metadata["activities"] == []
    assert metadata["num_subjects"] is None
    assert metadata["core_channels"] == {c: c for c in convert_mod.SIGNAL_COLUMNS}
    # Two bytes a sample is what keeps the label-free corpus inside its disk budget.
    assert metadata["grid_dtype"] == "float16"


def test_checked_in_descriptor_matches_what_convert_writes(converted):
    """The static metadata.json makes the source discoverable before any data lands."""
    static = json.loads((Path(convert_mod.DS_DIR) / "metadata.json").read_text())
    generated = json.loads((converted / "metadata.json").read_text())
    assert set(static) == set(generated)
    for key in (
        "dataset", "sampling_rate_hz", "pre_windowed", "streaming_grid", "role",
        "phase_a_only", "activities", "num_subjects", "channels", "core_channels",
        "extra_channels", "grid_dtype", "placement", "display_name",
    ):
        assert static[key] == generated[key], key


def test_gapped_take_produces_two_parts(converted):
    names = sorted(path.name for path in (converted / "sessions").iterdir())
    parts = [name for name in names if "take-0002" in name]
    assert len(parts) == 2
    assert all(convert_mod.STREAM_TOKEN in name for name in parts)
    assert {name.rsplit("_", 1)[1] for name in parts} == {"part01", "part02"}


# --------------------------------------------------------------------------
# fetch.py
# --------------------------------------------------------------------------


def _fetch_takes(count: int = 50) -> list[dict]:
    return [
        {
            "take_uid": f"take-{i:04d}",
            "take_name": f"take_{i:04d}",
            "root_dir": f"takes/take_{i:04d}",
            "capture_uid": f"cap-{i // 3:04d}",
            "participant_uid": 100 + i // 3,
            "university_name": ["cmu", "unc", "sfu"][i % 3],
            "duration_sec": 60.0 + i,
        }
        for i in range(count)
    ]


def test_seeded_take_subset_is_deterministic():
    takes = _fetch_takes()
    first = fetch_mod.select_takes(takes, 7, seed=42)
    second = fetch_mod.select_takes(list(reversed(takes)), 7, seed=42)
    assert [t["take_uid"] for t in first] == [t["take_uid"] for t in second]
    assert len(first) == 7
    assert [t["take_uid"] for t in first] == sorted(t["take_uid"] for t in first)


def test_a_different_seed_gives_a_different_subset():
    takes = _fetch_takes()
    a = {t["take_uid"] for t in fetch_mod.select_takes(takes, 7, seed=1)}
    b = {t["take_uid"] for t in fetch_mod.select_takes(takes, 7, seed=2)}
    assert a != b


def test_subset_is_a_prefix_chain():
    """Growing --takes must extend the previous set, so a re-run resumes."""
    takes = _fetch_takes()
    small = {t["take_uid"] for t in fetch_mod.select_takes(takes, 5, seed=9)}
    large = {t["take_uid"] for t in fetch_mod.select_takes(takes, 9, seed=9)}
    assert small <= large


def test_zero_takes_is_refused():
    with pytest.raises(ValueError, match="intentionally blocked"):
        fetch_mod.select_takes(_fetch_takes(), 0, seed=1)


def test_max_gb_refuses_an_oversized_plan():
    takes = _fetch_takes(5000)
    selected = fetch_mod.select_takes(takes, 4000, seed=3)
    plan_bytes, how = fetch_mod.estimate_bytes(selected, takes)
    assert how == "duration_share"
    with pytest.raises(fetch_mod.BudgetExceeded, match="exceeds --max-gb"):
        fetch_mod.check_budget(plan_bytes, fetch_mod.DEFAULT_MAX_GB, how)


def test_max_gb_admits_a_small_plan():
    takes = _fetch_takes(5000)
    selected = fetch_mod.select_takes(takes, 20, seed=3)
    plan_bytes, how = fetch_mod.estimate_bytes(selected, takes)
    fetch_mod.check_budget(plan_bytes, fetch_mod.DEFAULT_MAX_GB, how)
    assert 0 < plan_bytes < fetch_mod.DEFAULT_MAX_GB * 1e9


def test_estimate_falls_back_to_uniform_when_durations_are_missing():
    takes = [{"take_uid": f"t{i}"} for i in range(100)]
    plan_bytes, how = fetch_mod.estimate_bytes(takes[:10], takes)
    assert how == "uniform_per_take"
    assert plan_bytes == pytest.approx(fetch_mod.IMU_PART_BYTES * 0.1, rel=1e-6)


def test_command_always_requests_the_noimagestream_part(tmp_path):
    command = fetch_mod.build_command(
        tmp_path, [fetch_mod.IMU_PART], uids=["a", "b"], universities=["cmu"],
        splits=["train"], s3_profile="egoexo",
    )
    assert "take_vrs_noimagestream" in command
    assert command[:2] == ["egoexo", "-o"]
    assert "--uids" in command and "a" in command and "b" in command
    assert "--universities" in command and "cmu" in command
    assert "--splits" in command and "train" in command
    assert "--s3_profile" in command and "egoexo" in command
    assert "-y" in command


def test_missing_cli_message_is_actionable():
    with pytest.raises(fetch_mod.MissingPrerequisite) as error:
        fetch_mod.require_cli(which=lambda _name: None)
    text = str(error.value)
    assert "ego4d>=1.7.1" in text
    assert "aws configure" in text
    assert "https://ego4d-data.org/" in text
    assert "2 days" in text


def test_missing_aws_message_is_actionable(tmp_path):
    env = {"HOME": str(tmp_path)}
    assert not fetch_mod.aws_configured(None, env)
    with pytest.raises(fetch_mod.MissingPrerequisite) as error:
        fetch_mod.require_aws("egoexo", env)
    text = str(error.value)
    assert "pip install awscli" in text
    assert "aws configure" in text
    assert "--s3-profile" in text


def test_aws_profile_detected_in_shared_credentials(tmp_path):
    aws = tmp_path / ".aws"
    aws.mkdir()
    (aws / "credentials").write_text("[egoexo]\naws_access_key_id = AKIA\n")
    env = {"HOME": str(tmp_path)}
    assert fetch_mod.aws_configured("egoexo", env)
    assert not fetch_mod.aws_configured("other", env)


def test_dry_run_plans_without_downloading(tmp_path, monkeypatch):
    root = tmp_path / "downloads"
    _write_metadata(root, _fetch_takes(60))
    calls: list[list[str]] = []
    monkeypatch.setattr(fetch_mod, "require_cli", lambda **_kwargs: "/usr/bin/egoexo")
    monkeypatch.setattr(fetch_mod, "require_aws", lambda *_a, **_k: None)

    def runner(command):
        calls.append(list(command))
        raise AssertionError("dry run must not invoke the downloader for data")

    record = fetch_mod.fetch(
        out_dir=root, takes_count=5, seed=11, dry_run=True, skip_metadata=True, runner=runner
    )
    assert calls == []
    assert record["dry_run"] is True
    assert record["part"] == "take_vrs_noimagestream"
    assert record["estimate_is_measured"] is False
    assert len(record["takes"]) == 5
    manifest = json.loads((root / "fetch_manifest.json").read_text())
    assert manifest["selection_seed"] == 11
    assert [t["take_uid"] for t in manifest["takes"]] == [
        t["take_uid"] for t in record["takes"]
    ]


def test_fetch_refuses_unbounded_downloads(tmp_path):
    with pytest.raises(ValueError, match="deliberately no unbounded mode"):
        fetch_mod.fetch(out_dir=tmp_path)


def test_fetch_writes_a_resumable_manifest(tmp_path, monkeypatch):
    root = tmp_path / "downloads"
    takes = _fetch_takes(30)
    _write_metadata(root, takes)
    monkeypatch.setattr(fetch_mod, "require_cli", lambda **_kwargs: "/usr/bin/egoexo")
    monkeypatch.setattr(fetch_mod, "require_aws", lambda *_a, **_k: None)
    commands: list[list[str]] = []

    def runner(command):
        commands.append(list(command))
        if fetch_mod.IMU_PART in command:
            # Simulate the downloader landing only two of the planned takes.
            for uid in command[command.index("--uids") + 1: command.index("--uids") + 3]:
                take = next(t for t in takes if t["take_uid"] == uid)
                path = root / take["root_dir"]
                path.mkdir(parents=True, exist_ok=True)
                (path / "aria01_noimagestreams.vrs").write_bytes(b"x" * 16)
        return types.SimpleNamespace(returncode=0)

    record = fetch_mod.fetch(
        out_dir=root, takes_count=4, seed=5, max_gb=1000.0, skip_metadata=True, runner=runner
    )
    assert len(commands) == 1
    assert len(record["realized_take_uids"]) == 2
    assert len(record["missing_take_uids"]) == 2
    assert record["bytes_on_disk"] == 32
    assert json.loads((root / "fetch_manifest.json").read_text())["realized_take_uids"] == record[
        "realized_take_uids"
    ]


def test_university_filter():
    takes = _fetch_takes(30)
    filtered = fetch_mod.filter_takes(takes, universities=["cmu"])
    assert filtered and all(t["university_name"] == "cmu" for t in filtered)


def test_metadata_lookup_finds_nested_takes_json(tmp_path):
    nested = tmp_path / "v2" / "metadata"
    nested.mkdir(parents=True)
    (nested / "takes.json").write_text("[]")
    assert fetch_mod.find_metadata_file(tmp_path) == nested / "takes.json"


# --------------------------------------------------------------------------
# Incremental conversion (the fetch/convert/delete batch loop)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subject", ["p657", "capture_d37b73eb-fa42-43a6-8115-56832996ebd7"]
)
@pytest.mark.parametrize("part,total", [(1, 1), (2, 3)])
def test_session_id_round_trips(subject, part, total):
    uid = "13f01c79-5bfd-42f5-90ce-ee350aa1c3ad"
    sid = convert_mod.session_id(uid, subject, part, total)
    assert convert_mod.parse_session_id(sid) == (uid, subject, part)


def test_parse_rejects_a_foreign_session_id():
    with pytest.raises(ValueError, match="not an ego_exo4d session id"):
        convert_mod.parse_session_id("nhanes_65559_watch_wrist")


def _one_take_tree(uid, name, participant, duration=20.0, rate=NATIVE_HZ):
    take = _take(uid=uid, participant=participant, capture=f"cap-{uid}", name=str(name))
    return take, _stream(duration_s=duration, rate=rate)


def test_keep_existing_carries_earlier_batches_forward(tmp_path, monkeypatch):
    """The README loop deletes each batch's VRS, so a rebuild would be unrecoverable."""
    raw, out = tmp_path / "downloads", tmp_path / "out"
    first, first_stream = _one_take_tree("take-0001", "cmu_bike01_1", 657)
    second, second_stream = _one_take_tree("take-0002", "unc_cook01_1", 658)

    def run(takes, streams, **kwargs):
        _write_metadata(raw, takes)
        for take in takes:
            _stage_take(raw, take)
        _install_fake_aria(
            monkeypatch, lambda path: _FakeProvider({"imu-right": streams[Path(path).parent.name]})
        )
        return convert_mod.convert(raw_dir=raw, output_dir=out, **kwargs)

    assert run([first], {"cmu_bike01_1": first_stream})
    batch_one = {path.name for path in (out / "sessions").iterdir()}
    assert batch_one

    # Batch one's VRS is gone by the time batch two arrives.
    for path in raw.rglob("*.vrs"):
        path.unlink()
    assert run([second], {"unc_cook01_1": second_stream}, keep_existing=True)

    names = {path.name for path in (out / "sessions").iterdir()}
    assert batch_one < names  # strictly grown, nothing lost
    labels = json.loads((out / "labels.json").read_text())
    assert set(labels) == names
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["num_takes"] == 2
    assert manifest["takes_converted_this_run"] == 1
    assert manifest["num_subjects"] == 2
    assert manifest["num_sessions"] == len(names)


def test_without_keep_existing_the_tree_is_rebuilt(tmp_path, monkeypatch):
    raw, out = tmp_path / "downloads", tmp_path / "out"
    first, first_stream = _one_take_tree("take-0001", "cmu_bike01_1", 657)
    second, second_stream = _one_take_tree("take-0002", "unc_cook01_1", 658)
    _write_metadata(raw, [first])
    _stage_take(raw, first)
    _install_fake_aria(monkeypatch, lambda path: _FakeProvider({"imu-right": first_stream}))
    assert convert_mod.convert(raw_dir=raw, output_dir=out)
    batch_one = {path.name for path in (out / "sessions").iterdir()}

    _write_metadata(raw, [second])
    _stage_take(raw, second)
    _install_fake_aria(monkeypatch, lambda path: _FakeProvider({"imu-right": second_stream}))
    assert convert_mod.convert(raw_dir=raw, output_dir=out)
    names = {path.name for path in (out / "sessions").iterdir()}
    assert not (batch_one & names)


def test_incremental_run_still_reports_a_carried_fallback_subject(tmp_path, monkeypatch):
    """A clean second batch must not paper over a fallback subject in the first."""
    raw, out = tmp_path / "downloads", tmp_path / "out"
    fallback = _take(uid="take-0001", participant=None, capture="cap-x", name="cmu_bike01_1")
    clean = _take(uid="take-0002", participant=42, capture="cap-y", name="unc_cook01_1")
    stream = _stream(duration_s=20.0)

    def run(take, **kwargs):
        _write_metadata(raw, [take])
        _stage_take(raw, take)
        _install_fake_aria(monkeypatch, lambda path: _FakeProvider({"imu-right": stream}))
        return convert_mod.convert(raw_dir=raw, output_dir=out, **kwargs)

    assert run(fallback)
    assert json.loads((out / "manifest.json").read_text())["subject_id_source"] == (
        "capture_uid_fallback"
    )
    assert run(clean, keep_existing=True)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["subject_id_source"] == "mixed"
    assert manifest["subject_id_fallback_takes"] == 1
    assert manifest["subject_id_fallback_subjects"] == 1
    assert "capture-level subject id" in manifest["note"]
