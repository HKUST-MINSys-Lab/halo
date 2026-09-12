"""Offline tests for the Nymeria label-free pretraining module.

Nothing here downloads anything and nothing here needs the real 80 TB release:
every fixture is synthesised in ``tmp_path``.  ``projectaria_tools`` is not
installed in this tree, so the VRS path is exercised against a fake module
injected into ``sys.modules``.
"""

from __future__ import annotations

import inspect
import json
import math
import sys
import types
import zipfile

import numpy as np
import pandas as pd
import pytest

from data.pretraining.nymeria import convert, fetch


# ======================================================================================
# fixtures
# ======================================================================================

# The MVN Link tracker labels this converter cares about, in the order they appear in
# the synthetic file.  `T8` is the trunk tracker that maps to the `xsens_sternum` token.
_LABELS = (
    "Pelvis",
    "T8",
    "Head",
    "RightUpperArm",
    "RightForeArm",
    "LeftForeArm",
    "LeftUpperArm",
    "LeftUpperLeg",
    "RightUpperLeg",
    "LeftLowerLeg",
    "RightLowerLeg",
)


def _mvnx_text(frames, *, frame_rate=240.0, with_sensor_fields=True, indices=None):
    """A minimal but structurally faithful MVNX document.

    ``frames`` is a list of dicts with keys ``free`` (n_sensors x 3),
    ``quat`` (n_sensors x 4, w-first) and ``omega`` (n_segments x 3).
    """
    segments = "".join(
        f'<segment label="{label}" id="{i + 1}"/>' for i, label in enumerate(_LABELS)
    )
    sensors = "".join(f'<sensor label="{label}"/>' for label in _LABELS)
    indices = list(range(len(frames))) if indices is None else list(indices)
    body = []
    for index, frame in zip(indices, frames):
        free = " ".join(f"{v:.9f}" for v in np.asarray(frame["free"], float).ravel())
        quat = " ".join(f"{v:.9f}" for v in np.asarray(frame["quat"], float).ravel())
        omega = " ".join(f"{v:.9f}" for v in np.asarray(frame["omega"], float).ravel())
        if with_sensor_fields:
            payload = (
                f"<sensorFreeAcceleration>{free}</sensorFreeAcceleration>"
                f"<sensorOrientation>{quat}</sensorOrientation>"
                f"<angularVelocity>{omega}</angularVelocity>"
            )
        else:
            payload = (
                f"<acceleration>{free}</acceleration>"
                f"<orientation>{quat}</orientation>"
                f"<angularVelocity>{omega}</angularVelocity>"
            )
        body.append(
            f'<frame type="normal" index="{index}" time="{index * 1000.0 / frame_rate:.4f}">'
            f"{payload}</frame>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mvnx version="4">'
        '<mvn version="2021.2" build="1"/>'
        f'<subject label="P0042" frameRate="{frame_rate:g}" '
        f'segmentCount="{len(_LABELS)}" sensorCount="{len(_LABELS)}">'
        f"<segments>{segments}</segments>"
        f"<sensors>{sensors}</sensors>"
        '<frames segmentCount="%d" sensorCount="%d">'
        '<frame type="identity"><orientation>1 0 0 0</orientation></frame>'
        "%s</frames>"
        "</subject></mvnx>"
    ) % (len(_LABELS), len(_LABELS), "".join(body))


def _stationary_frames(count, *, quat=(1.0, 0.0, 0.0, 0.0)):
    n = len(_LABELS)
    return [
        {
            "free": np.zeros((n, 3)),
            "quat": np.tile(np.asarray(quat, float), (n, 1)),
            "omega": np.zeros((n, 3)),
        }
        for _ in range(count)
    ]


def _write_mvnx(tmp_path, frames, **kwargs):
    path = tmp_path / "xdata.mvnx"
    path.write_text(_mvnx_text(frames, **kwargs))
    return path


# ======================================================================================
# 1. rotate-and-add-gravity math
# ======================================================================================


def test_stationary_sensor_reads_one_g_and_zero_gyro(tmp_path):
    """A motionless tracker must read |a| = 1 g with gravity PRESENT and ~0 rad/s."""
    recording = convert.parse_mvnx(_write_mvnx(tmp_path, _stationary_frames(4)))
    assert recording.acceleration_source == "sensor"
    assert recording.frame_rate == pytest.approx(240.0)

    signals = {s.token: s for s in convert.mvnx_stream_signals(recording)}
    assert set(signals) == set(convert.XSENS_STREAMS)

    for token, signal in signals.items():
        magnitude = np.linalg.norm(signal.acc_g, axis=1)
        assert np.allclose(magnitude, 1.0, atol=1e-6), token
        # Identity orientation + Z-up global frame -> gravity sits entirely on acc_z.
        assert np.allclose(signal.acc_g, [0.0, 0.0, 1.0], atol=1e-6), token
        assert np.allclose(signal.gyro_rads, 0.0, atol=1e-12), token


def test_rotated_stationary_sensor_still_reads_one_g(tmp_path):
    """Rotating the tracker moves gravity between axes but never changes |a| = 1 g."""
    # 90 deg about global X: quaternion (cos45, sin45, 0, 0), w-first.
    half = math.sqrt(0.5)
    frames = _stationary_frames(3, quat=(half, half, 0.0, 0.0))
    recording = convert.parse_mvnx(_write_mvnx(tmp_path, frames))
    signal = convert.mvnx_stream_signals(recording)[0]
    assert np.allclose(np.linalg.norm(signal.acc_g, axis=1), 1.0, atol=1e-6)
    # R maps device->global; Rx(+90) sends device +Y to global +Z, so the device
    # reads the whole of gravity on its +Y axis.
    assert np.allclose(signal.acc_g, [0.0, 1.0, 0.0], atol=1e-6)


def test_free_acceleration_adds_to_gravity_in_the_device_frame():
    """Known linear acceleration on top of gravity, identity orientation."""
    free = np.array([[0.0, 0.0, convert.GRAVITY_MS2], [convert.GRAVITY_MS2, 0.0, 0.0]])
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
    acc = convert.free_acceleration_to_device_g(free, quat)
    assert np.allclose(acc, [[0.0, 0.0, 2.0], [1.0, 0.0, 1.0]], atol=1e-9)


def test_gyro_is_rotated_into_the_device_frame(tmp_path):
    """Global angular velocity must land in the sensor's own axes, in rad/s."""
    half = math.sqrt(0.5)
    n = len(_LABELS)
    frames = [
        {
            "free": np.zeros((n, 3)),
            "quat": np.tile(np.array([half, half, 0.0, 0.0]), (n, 1)),
            "omega": np.tile(np.array([0.0, 0.0, 2.0]), (n, 1)),
        }
    ]
    recording = convert.parse_mvnx(_write_mvnx(tmp_path, frames))
    signal = convert.mvnx_stream_signals(recording)[0]
    # R^T sends global +Z onto device +Y for a +90 deg rotation about X.
    assert np.allclose(signal.gyro_rads, [0.0, 2.0, 0.0], atol=1e-9)
    assert recording.angular_unit == "rad"
    assert recording.angular_unit_source == "assumed_mvnx_si_default"


def test_degree_override_converts_angular_velocity(tmp_path):
    n = len(_LABELS)
    frames = [
        {
            "free": np.zeros((n, 3)),
            "quat": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
            "omega": np.tile(np.array([180.0, 0.0, 0.0]), (n, 1)),
        }
    ]
    recording = convert.parse_mvnx(_write_mvnx(tmp_path, frames), angular_unit="deg")
    assert recording.angular_unit_source == "cli_override"
    signal = convert.mvnx_stream_signals(recording)[0]
    assert np.allclose(signal.gyro_rads[:, 0], math.pi, atol=1e-9)


def test_segment_fallback_when_sensor_fields_absent(tmp_path):
    recording = convert.parse_mvnx(
        _write_mvnx(tmp_path, _stationary_frames(2), with_sensor_fields=False)
    )
    assert recording.acceleration_source == "segment"
    signal = convert.mvnx_stream_signals(recording)[0]
    assert np.allclose(np.linalg.norm(signal.acc_g, axis=1), 1.0, atol=1e-6)


def test_sternum_token_maps_to_the_t8_tracker(tmp_path):
    recording = convert.parse_mvnx(_write_mvnx(tmp_path, _stationary_frames(2)))
    signals = {s.token: s for s in convert.mvnx_stream_signals(recording)}
    assert signals["xsens_sternum"].detail["label"] == "T8"
    assert signals["xsens_rthigh"].detail["label"] == "RightUpperLeg"
    assert signals["xsens_rshank"].detail["label"] == "RightLowerLeg"


def test_mvnx_dropped_frames_become_a_time_gap(tmp_path):
    """A 240-frame index jump at 240 Hz is a 1 s gap and must split the session."""
    frames = _stationary_frames(6)
    indices = [0, 1, 2, 242, 243, 244]
    recording = convert.parse_mvnx(_write_mvnx(tmp_path, frames, indices=indices))
    blocks = convert.contiguous_blocks(recording.times_sec)
    assert blocks == [(0, 3), (3, 6)]


# ======================================================================================
# 2. url.json filtering and the --max-gb refusal
# ======================================================================================


def _fake_url_json(sequence_count=6, *, group_bytes=10_000_000_000, sizes=True):
    sequences = {}
    for i in range(sequence_count):
        uid = f"20230607_s{i}_indoor_ace{i:03d}"
        sequences[uid] = {
            "body_xdata_mvnx": {
                "download_url": f"https://example.invalid/{uid}/xdata.mvnx",
                "filename": f"Nymeria_v0.0_{uid}_body_xdata.mvnx",
                **({"file_size_bytes": group_bytes} if sizes else {}),
            },
            # A group we must never request, sized so that including it would blow
            # any sane budget.
            "recording_head_rgb": {
                "video.vrs": {
                    "download_url": f"https://example.invalid/{uid}/rgb.vrs",
                    "filename": "video.vrs",
                    "file_size_bytes": 400_000_000_000,
                }
            },
        }
    return {"sequences": sequences}


def test_catalog_parse_and_group_filter():
    raw = _fake_url_json()
    catalog = fetch.parse_catalog(raw)
    assert len(catalog) == 6
    plan = fetch.build_plan(catalog, sorted(catalog)[:2], fetch.DEFAULT_GROUPS)
    assert {r.group for r in plan.records} == set(fetch.DEFAULT_GROUPS)
    assert "recording_head_rgb" not in {r.group for r in plan.records}
    assert len(plan.records) == 2
    assert plan.known_bytes == 2 * 10_000_000_000


def test_sequence_selection_is_deterministic_and_bounded():
    available = [f"seq{i:03d}" for i in range(50)]
    first = fetch.select_sequences(available, 7, seed=20260909)
    assert first == fetch.select_sequences(available, 7, seed=20260909)
    assert len(first) == 7
    assert set(first) <= set(available)
    assert first != fetch.select_sequences(available, 7, seed=1)
    assert first == sorted(first)
    with pytest.raises(ValueError):
        fetch.select_sequences(available, 0, seed=1)


def test_max_gb_refusal():
    catalog = fetch.parse_catalog(_fake_url_json())
    plan = fetch.build_plan(catalog, sorted(catalog), fetch.DEFAULT_GROUPS)
    assert plan.known_bytes / 1e9 == pytest.approx(60.0)
    with pytest.raises(fetch.BudgetExceeded, match="over the --max-gb budget"):
        fetch.check_budget(plan, max_gb=50.0)
    fetch.check_budget(plan, max_gb=200.0)  # under budget: no raise


def test_unknown_sizes_refuse_unless_explicitly_allowed():
    catalog = fetch.parse_catalog(_fake_url_json(sizes=False))
    plan = fetch.build_plan(catalog, sorted(catalog)[:1], fetch.DEFAULT_GROUPS)
    with pytest.raises(fetch.BudgetExceeded, match="declare no size"):
        fetch.check_budget(plan, max_gb=1000.0)
    fetch.check_budget(plan, max_gb=1000.0, allow_unknown_size=True)


def test_filtered_url_json_keeps_the_envelope_and_drops_everything_else(tmp_path):
    raw = _fake_url_json()
    catalog = fetch.parse_catalog(raw)
    keep = sorted(catalog)[:2]
    plan = fetch.build_plan(catalog, keep, fetch.DEFAULT_GROUPS)
    destination = fetch.write_filtered_url_json(raw, plan, tmp_path / "url.filtered.json")
    pruned = json.loads(destination.read_text())
    assert sorted(pruned["sequences"]) == keep
    for payload in pruned["sequences"].values():
        assert sorted(payload) == sorted(fetch.DEFAULT_GROUPS)
    # The source document must not have been mutated in place.
    assert len(raw["sequences"]) == 6
    assert "recording_head_rgb" in raw["sequences"][keep[0]]


def test_missing_url_json_raises_an_actionable_licence_error(tmp_path):
    with pytest.raises(fetch.LicenceError) as excinfo:
        fetch.load_url_json(tmp_path / "url.json")
    message = str(excinfo.value)
    assert "projectaria.com/datasets/nymeria" in message
    assert "body_xdata_mvnx" in message


def test_unrecognised_url_json_shape_is_reported_not_guessed():
    with pytest.raises(fetch.CatalogShapeError):
        fetch.parse_catalog({"totally": "different"})


def test_record_relative_path_uses_the_verified_group_layout():
    record = fetch.FileRecord(
        "seq1", "body_xdata_mvnx", "https://x/y", "release_body_xdata.mvnx", 1, None
    )
    assert record.relative_path == "seq1/body/xdata.mvnx"


def test_completed_release_named_mvnx_is_reported_present(tmp_path):
    """The fallback maps the release filename to the converter's stable layout."""
    destination = tmp_path / "seq1" / "body" / "xdata.mvnx"
    destination.parent.mkdir(parents=True)
    destination.write_text("mvnx payload")
    record = fetch.FileRecord(
        "seq1", "body_xdata_mvnx", "https://unused.invalid/release_body_xdata.mvnx",
        "release_body_xdata.mvnx",
        destination.stat().st_size, None,
    )
    assert fetch.download_record(tmp_path, record) == destination.stat().st_size
    assert (tmp_path / "seq1" / "body" / "xdata.mvnx").read_text() == "mvnx payload"


def test_fetch_dry_run_never_touches_the_network(tmp_path, monkeypatch):
    (tmp_path / "url.json").write_text(json.dumps(_fake_url_json()))

    def explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("dry run must not download")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", explode)
    monkeypatch.setattr(fetch, "run_official", explode)
    manifest = fetch.fetch(
        url_json=tmp_path / "url.json",
        out_dir=tmp_path / "out",
        sequences=2,
        max_gb=120.0,
        dry_run=True,
    )
    assert manifest["num_sequences"] == 2
    assert manifest["groups"] == list(fetch.DEFAULT_GROUPS)
    assert manifest["selection_seed"] == fetch.DEFAULT_SEED
    assert manifest["fetched_bytes"] == 0


# ======================================================================================
# 3. gap splitting
# ======================================================================================


def test_gap_split_on_synthetic_nonuniform_timestamps():
    part_a = np.arange(0, 1000) / 1000.0            # 1 s at 1 kHz
    part_b = part_a[-1] + 0.75 + np.arange(0, 500) / 1000.0   # after a 0.75 s gap
    times = np.concatenate([part_a, part_b])
    blocks = convert.contiguous_blocks(times, max_gap_seconds=0.5)
    assert blocks == [(0, 1000), (1000, 1500)]


def test_small_jitter_is_not_a_gap():
    rng = np.random.default_rng(0)
    times = np.cumsum(1.0 / 1000.0 + rng.normal(0, 2e-6, 5000))
    assert convert.contiguous_blocks(times, max_gap_seconds=0.5) == [(0, 5000)]


def test_backwards_clock_is_a_boundary():
    times = np.array([0.0, 0.001, 0.002, 0.0015, 0.0025])
    assert convert.contiguous_blocks(times) == [(0, 3), (3, 5)]


def test_one_missing_xsens_frame_is_a_boundary_not_time_compression():
    times = np.array([0.0, 1.0 / 240.0, 3.0 / 240.0, 4.0 / 240.0])
    assert convert.contiguous_blocks(times, expected_rate_hz=240.0) == [(0, 2), (2, 4)]


def test_gap_split_produces_part_suffixed_session_ids_keeping_the_stream_token():
    session = convert._session_id("20230607_s0_x", "P0042", "aria_lwrist", 2, 3)
    assert session == "nymeria_20230607_s0_x_P0042_aria_lwrist_p02"
    assert "aria_lwrist" in session  # the policy layer matches on this substring
    single = convert._session_id("20230607_s0_x", "P0042", "aria_lwrist", 1, 1)
    assert single.endswith("aria_lwrist")


# ======================================================================================
# 4. decimation
# ======================================================================================


@pytest.mark.parametrize("rate_in", [800.0, 1000.0])
def test_decimation_keeps_5hz_and_kills_300hz(rate_in):
    duration = 8.0
    t = np.arange(int(duration * rate_in)) / rate_in
    signal = np.column_stack([np.sin(2 * np.pi * 5.0 * t), np.sin(2 * np.pi * 300.0 * t)])
    out = convert.decimate_to(signal, rate_in, convert.ARIA_RATE_HZ)

    assert len(out) == pytest.approx(duration * convert.ARIA_RATE_HZ, rel=0.02)
    # Drop filter transients at both ends before measuring amplitude.
    edge = int(0.5 * convert.ARIA_RATE_HZ)
    kept, removed = out[edge:-edge, 0], out[edge:-edge, 1]
    assert np.max(np.abs(kept)) == pytest.approx(1.0, abs=0.02)
    assert np.max(np.abs(removed)) < 0.01
    # And the 5 Hz tone is still at 5 Hz, not aliased.
    spectrum = np.abs(np.fft.rfft(kept * np.hanning(len(kept))))
    peak = np.fft.rfftfreq(len(kept), 1.0 / convert.ARIA_RATE_HZ)[np.argmax(spectrum)]
    assert peak == pytest.approx(5.0, abs=0.4)


def test_snap_nominal_rate():
    assert convert.snap_nominal_rate(999.4) == 1000.0
    assert convert.snap_nominal_rate(801.2) == 800.0
    assert convert.snap_nominal_rate(333.0) == 333.0  # nothing close: keep the estimate


def test_resample_uniform_recovers_a_tone_from_a_jittered_clock():
    rng = np.random.default_rng(3)
    n = 4000
    times = np.arange(n) / 1000.0 + rng.normal(0, 5e-6, n)
    times = np.maximum.accumulate(times)
    values = np.sin(2 * np.pi * 5.0 * times)[:, None]
    grid, out = convert.resample_uniform(times, values, 1000.0)
    assert np.allclose(np.diff(grid), 1e-3, atol=1e-12)
    assert np.max(np.abs(out[:, 0] - np.sin(2 * np.pi * 5.0 * grid))) < 1e-3


# ======================================================================================
# 5. the Aria VRS path, against a fake projectaria_tools
# ======================================================================================


class _FakeImuSample:
    def __init__(self, timestamp_ns, accel, gyro):
        self.capture_timestamp_ns = timestamp_ns
        self.accel_msec2 = accel
        self.gyro_radsec = gyro


class _FakeProvider:
    def __init__(self, samples, label="imu-right"):
        self._samples = samples
        self._label = label

    def get_stream_id_from_label(self, label):
        return "1202-1" if label == self._label else None

    def get_num_data(self, stream_id):
        return len(self._samples)

    def get_imu_data_by_index(self, stream_id, index):
        return self._samples[index]


def _install_fake_aria(monkeypatch, provider):
    root = types.ModuleType("projectaria_tools")
    core = types.ModuleType("projectaria_tools.core")
    data_provider = types.ModuleType("projectaria_tools.core.data_provider")
    data_provider.create_vrs_data_provider = lambda path: provider
    core.data_provider = data_provider
    root.core = core
    monkeypatch.setitem(sys.modules, "projectaria_tools", root)
    monkeypatch.setitem(sys.modules, "projectaria_tools.core", core)
    monkeypatch.setitem(sys.modules, "projectaria_tools.core.data_provider", data_provider)


def _aria_samples(rate=1000.0, seconds=14.0, gap_after=None, gap_seconds=0.75):
    samples = []
    t = 0.0
    for i in range(int(rate * seconds)):
        if gap_after is not None and i == gap_after:
            t += gap_seconds
        phase = 2 * np.pi * 5.0 * t
        samples.append(
            _FakeImuSample(
                int(t * 1e9),
                [0.0, 0.0, convert.GRAVITY_MS2 + 0.5 * math.sin(phase)],
                [0.1, 0.0, 0.0],
            )
        )
        t += 1.0 / rate
    return samples


def test_read_aria_imu_prefers_imu_right(monkeypatch, tmp_path):
    _install_fake_aria(monkeypatch, _FakeProvider(_aria_samples(seconds=0.01)))
    times, accel, gyro, label = convert.read_aria_imu(tmp_path / "motion.vrs")
    assert label == "imu-right"
    assert times.shape == (10,) and accel.shape == (10, 3) and gyro.shape == (10, 3)


def test_read_aria_imu_falls_back_to_imu_left(monkeypatch, tmp_path):
    _install_fake_aria(monkeypatch, _FakeProvider(_aria_samples(seconds=0.01), label="imu-left"))
    _, _, _, label = convert.read_aria_imu(tmp_path / "motion.vrs")
    assert label == "imu-left"


def test_missing_projectaria_tools_is_an_actionable_error(monkeypatch, tmp_path):
    for name in list(sys.modules):
        if name.startswith("projectaria_tools"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(
        convert.importlib if hasattr(convert, "importlib") else convert, "__name__", convert.__name__
    )
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("projectaria_tools"):
            raise ImportError("no module named projectaria_tools")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(RuntimeError, match="pip install projectaria-tools"):
        convert.read_aria_imu(tmp_path / "motion.vrs")


def test_aria_stream_signals_decimate_and_split(monkeypatch, tmp_path):
    _install_fake_aria(monkeypatch, _FakeProvider(_aria_samples(seconds=20.0, gap_after=8000)))
    times, accel, gyro, label = convert.read_aria_imu(tmp_path / "motion.vrs")
    signals = convert.aria_stream_signals(times, accel, gyro, "aria_lwrist", label)
    assert len(signals) == 2  # the 0.75 s gap split the recording
    for signal in signals:
        assert signal.rate_hz == 200.0
        assert signal.detail["native_rate_hz"] == 1000.0
        assert signal.detail["imu_stream"] == "imu-right"
        assert len(signal.acc_g) % int(200.0 * convert.WINDOW_SECONDS) == 0
        # m/s^2 -> g, gravity present.
        assert np.median(np.linalg.norm(signal.acc_g, axis=1)) == pytest.approx(1.0, abs=0.1)
        # The gyro is a DC 0.1 rad/s; check the steady state, past the FIR startup.
        edge = int(0.5 * signal.rate_hz)
        assert np.allclose(signal.gyro_rads[edge:-edge, 0], 0.1, atol=1e-3)


# ======================================================================================
# 6. output contract
# ======================================================================================


def _build_sequence(root, name, *, frames=240 * 13, with_aria=True):
    sequence = root / name
    (sequence / "body").mkdir(parents=True)
    (sequence / "body" / "xdata.mvnx").write_text(_mvnx_text(_stationary_frames(frames)))
    (sequence / "metadata.json").write_text(json.dumps({"uid": name, "participant_id": "P0042"}))
    if with_aria:
        for recording in convert.ARIA_STREAMS:
            (sequence / recording / "data").mkdir(parents=True)
            (sequence / recording / "data" / "motion.vrs").write_text("stub")
    return sequence


def test_convert_writes_the_repo_session_contract(tmp_path, monkeypatch):
    raw = tmp_path / "downloads"
    raw.mkdir()
    _build_sequence(raw, "20230607_s0_indoor_a", with_aria=True)
    _install_fake_aria(monkeypatch, _FakeProvider(_aria_samples(seconds=14.0)))

    out = tmp_path / "out"
    assert convert.convert(raw_dir=raw, output_dir=out)

    xsens = out / convert.XSENS_DATASET
    aria = out / convert.ARIA_DATASET
    for directory, rate, expected_tokens in (
        (xsens, 240.0, set(convert.XSENS_STREAMS)),
        (aria, 200.0, set(convert.ARIA_STREAMS.values())),
    ):
        labels = json.loads((directory / "labels.json").read_text())
        assert labels, directory
        assert all(value == ["__unlabeled__"] for value in labels.values())
        tokens = {token for token in expected_tokens if any(token in s for s in labels)}
        assert tokens == expected_tokens

        metadata = json.loads((directory / "metadata.json").read_text())
        assert metadata["sampling_rate_hz"] == rate
        assert metadata["dataset"] == directory.name
        assert metadata["grid_dtype"] == "float16"

        manifest = json.loads((directory / "manifest.json").read_text())
        assert set(manifest) >= {
            "dataset_name",
            "source",
            "num_subjects",
            "sampling_rate_hz",
            "channels",
            "unit",
            "gravity_state",
            "phase_a_only",
            "note",
        }
        assert manifest["sampling_rate_hz"] == rate
        assert manifest["unit"] == "g"
        assert manifest["gravity_state"] == "present"
        assert manifest["phase_a_only"] is True
        assert manifest["num_subjects"] == 1

        for session_id in labels:
            frame = pd.read_parquet(directory / "sessions" / session_id / "data.parquet")
            assert list(frame.columns) == [
                "timestamp_sec",
                "acc_x",
                "acc_y",
                "acc_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "subject",
            ]
            assert frame["timestamp_sec"].dtype == np.float64
            for column in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"):
                assert frame[column].dtype == np.float32, column
            assert frame["subject"].map(type).eq(str).all()
            assert frame["subject"].nunique() == 1
            assert frame["subject"].iloc[0] == "P0042"

            stamps = frame["timestamp_sec"].to_numpy()
            assert stamps[0] == 0.0
            assert np.all(np.diff(stamps) > 0)
            assert np.allclose(np.diff(stamps), 1.0 / rate, atol=1e-9)
            # Whole source windows only, so no grid window can cross a gap.
            assert len(frame) % int(rate * convert.WINDOW_SECONDS) == 0
            assert np.median(np.linalg.norm(
                frame[["acc_x", "acc_y", "acc_z"]].to_numpy(), axis=1
            )) == pytest.approx(1.0, abs=0.1)


def test_convert_writes_a_union_labels_file_and_a_rateless_module_manifest(tmp_path, monkeypatch):
    raw = tmp_path / "downloads"
    raw.mkdir()
    _build_sequence(raw, "20230607_s0_indoor_a")
    _install_fake_aria(monkeypatch, _FakeProvider(_aria_samples(seconds=14.0)))
    out = tmp_path / "out"
    convert.convert(raw_dir=raw, output_dir=out)

    union = json.loads((out / "labels.json").read_text())
    per_dataset = {}
    for name in (convert.XSENS_DATASET, convert.ARIA_DATASET):
        per_dataset.update(json.loads((out / name / "labels.json").read_text()))
    assert union == per_dataset

    module = json.loads((out / "manifest.json").read_text())
    # The module manifest must NOT claim a single rate: that is the whole reason
    # the module splits into two dataset directories.
    assert "sampling_rate_hz" not in module
    assert set(module["datasets"]) == {convert.XSENS_DATASET, convert.ARIA_DATASET}
    assert module["datasets"][convert.XSENS_DATASET]["sampling_rate_hz"] == 240.0
    assert module["datasets"][convert.ARIA_DATASET]["sampling_rate_hz"] == 200.0


def test_convert_streams_flag_selects_one_modality(tmp_path):
    raw = tmp_path / "downloads"
    raw.mkdir()
    _build_sequence(raw, "20230607_s0_indoor_a", with_aria=False)
    out = tmp_path / "out"
    assert convert.convert(raw_dir=raw, output_dir=out, streams=("xsens",))
    assert (out / convert.XSENS_DATASET / "labels.json").exists()
    assert not (out / convert.ARIA_DATASET / "labels.json").exists()


def test_max_hours_per_sequence_caps_output(tmp_path):
    raw = tmp_path / "downloads"
    raw.mkdir()
    _build_sequence(raw, "20230607_s0_indoor_a", frames=240 * 60, with_aria=False)
    out = tmp_path / "out"
    convert.convert(
        raw_dir=raw, output_dir=out, streams=("xsens",), max_hours_per_sequence=12.0 / 3600.0
    )
    labels = json.loads((out / convert.XSENS_DATASET / "labels.json").read_text())
    for session_id in labels:
        frame = pd.read_parquet(out / convert.XSENS_DATASET / "sessions" / session_id / "data.parquet")
        # The 12-second cap is applied before the source contract: retain the largest complete
        # eight-second JEPA window rather than emitting a partial window across a later grid seam.
        assert len(frame) == 240 * 8


def test_subject_without_verified_identity_is_rejected(tmp_path):
    sequence = tmp_path / "20230607_s0_indoor_a"
    sequence.mkdir()
    with pytest.raises(ValueError, match="no verified participant"):
        convert.resolve_subject(sequence, mvnx_subject=None)
    (sequence / "metadata.json").write_text(json.dumps({"session": {"participant": "P7"}}))
    assert convert.resolve_subject(sequence) == ("P7", "metadata.json:participant")


def test_official_sequence_uid_supplies_recurring_subject_not_generic_mvn_label(tmp_path):
    first = tmp_path / "20231222_s0_denise_carter_act2_d8mdi5"
    second = tmp_path / "20231222_s0_denise_carter_act3_3itytl"
    first.mkdir()
    second.mkdir()
    expected = ("denise_carter", "official_sequence_uid:pseudonym")
    assert convert.resolve_subject(first, mvnx_subject="MVN System") == expected
    assert convert.resolve_subject(second, mvnx_subject="MVN System") == expected


# ======================================================================================
# 7. npz path
# ======================================================================================


def test_npz_with_unknown_keys_refuses_and_dumps_the_real_schema(tmp_path):
    path = tmp_path / "xdata.npz"
    np.savez(path, mystery=np.zeros((4, 3)), other=np.ones(2))
    with pytest.raises(convert.NpzSchemaUnknown) as excinfo:
        convert.load_npz_recording(path)
    message = str(excinfo.value)
    assert "mystery" in message and '"shape"' in message
    assert "--body-source mvnx" in message
    assert convert.inspect_npz(path)["other"] == {"shape": [2], "dtype": "float64"}


def test_npz_with_recognised_keys_matches_the_mvnx_math(tmp_path):
    frames, sensors = 5, len(_LABELS)
    path = tmp_path / "xdata.npz"
    np.savez(
        path,
        sensorFreeAcceleration=np.zeros((frames, sensors, 3)),
        sensorOrientation=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, sensors, 1)),
        angularVelocity=np.zeros((frames, sensors, 3)),
        sensorLabels=np.array(_LABELS),
        frameRate=np.array([240.0]),
    )
    recording = convert.load_npz_recording(path)
    assert recording.frame_rate == 240.0
    assert recording.acceleration_source == "sensor"
    signals = {s.token: s for s in convert.mvnx_stream_signals(recording)}
    assert set(signals) == set(convert.XSENS_STREAMS)
    assert np.allclose(signals["xsens_head"].acc_g, [0.0, 0.0, 1.0], atol=1e-9)


# ======================================================================================
# 8. authored metadata.json files
# ======================================================================================


def test_default_output_lands_where_corpus_roots_resolves_the_two_datasets(tmp_path, monkeypatch):
    """The dataset dirs are SIBLINGS of this package, not children of it.

    `corpus_roots.dataset_root("nymeria_xsens")` resolves to `data/pretraining/nymeria_xsens`,
    and `build_corpus._stage_convert` runs this converter with no arguments, so the DEFAULT
    output dir has to be the corpus root.
    """
    from data.scripts.curate import corpus_roots

    assert convert.CORPUS_ROOT == corpus_roots.PRETRAIN_ROOT
    parameters = inspect.signature(convert.convert).parameters
    assert parameters["output_dir"].default == convert.CORPUS_ROOT

    # And a converted dataset is discoverable by name through their resolver.
    raw = tmp_path / "downloads"
    raw.mkdir()
    _build_sequence(raw, "20230607_s0_indoor_a", with_aria=False)
    root = tmp_path / "pretraining"
    convert.convert(raw_dir=raw, output_dir=root, streams=("xsens",))
    monkeypatch.setattr(corpus_roots, "CORPUS_ROOTS", (tmp_path / "datasets", root))
    monkeypatch.setattr(corpus_roots, "PRETRAIN_ROOT", root)
    assert corpus_roots.dataset_root(convert.XSENS_DATASET) == root / convert.XSENS_DATASET
    assert corpus_roots.is_pretraining(convert.XSENS_DATASET)
    assert corpus_roots.dataset_names() == (convert.XSENS_DATASET,)


def test_the_module_package_is_not_itself_a_dataset_directory():
    """corpus_roots treats any dir holding metadata.json as a dataset.

    A metadata.json here would make 'nymeria' a phantom third dataset beside the two
    real ones, and would be picked up by dataset_names(PRETRAIN_ROOT).
    """
    assert not (convert.DS_DIR / "metadata.json").exists()
    assert (convert.DS_DIR / "module_metadata.json").exists()


def test_authored_metadata_declares_the_two_rates():
    root = convert.DS_DIR
    module = json.loads((root / "module_metadata.json").read_text())
    assert module["is_module_descriptor"] is True
    assert "sampling_rate_hz" not in module
    assert module["sampling_rate_hz_by_dataset"] == {
        convert.XSENS_DATASET: 240.0,
        convert.ARIA_DATASET: 200.0,
    }
    assert module["phase_a_only"] is True
    assert module["role"] == "pretrain_scale"

    xsens = json.loads(convert.METADATA_TEMPLATES[convert.XSENS_DATASET].read_text())
    aria = json.loads(convert.METADATA_TEMPLATES[convert.ARIA_DATASET].read_text())
    assert xsens["sampling_rate_hz"] == 240.0
    assert aria["sampling_rate_hz"] == 200.0
    assert set(xsens["stream_tokens"]) == set(convert.XSENS_STREAMS)
    assert set(aria["stream_tokens"]) == set(convert.ARIA_STREAMS.values())
    for meta in (xsens, aria):
        assert meta["unit"] == "g"
        assert meta["gravity_state"] == "present"
        assert meta["pre_windowed"] is False
        assert meta["streaming_grid"] is True
        assert meta["phase_a_only"] is True
        assert meta["role"] == "pretrain_scale"
        # The data/pretraining disk budget assumes float16 grids.
        assert meta["grid_dtype"] == "float16"
