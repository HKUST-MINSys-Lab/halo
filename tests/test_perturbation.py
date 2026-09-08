"""Controlled acquisition perturbations: each axis changes exactly what it claims."""

from __future__ import annotations

import numpy as np
import pytest

from data.scripts.curate.compatibility import are_compatible, is_near_miss
from eval.data import EvalStream
from eval.perturbation import (
    Perturbation,
    acquisition_key_for,
    compatibility_relation,
    perturb_stream,
)

CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def _stream(*, n=6, t=300, rate=50.0, gyro=True, seed=0):
    """A synthetic view of a REAL curated stream (phone at the waist, acc+gyro, gravity present)
    so the acquisition key can be computed; the signal is synthetic."""
    rng = np.random.default_rng(seed)
    windows = rng.standard_normal((n, t, 6)).astype(np.float32) * 0.2
    windows[..., 2] += 1.0                          # gravity on one accelerometer axis
    windows[1] = windows[0]                         # two identical windows in different executions
    mask = np.array([True, True, True, gyro, gyro, gyro])
    if not gyro:
        windows[..., 3:] = 0.0
    return EvalStream(
        dataset="inclusivehar", stream="phone_waist", alignment="native",
        windows=windows, gt=["walking"] * n, subjects=np.asarray(["s1"] * n, dtype=object),
        channels=list(CHANNELS), rate_hz=rate, mask=mask, eval_labels=["walking", "sitting"],
        execution_ids=np.asarray(["e0", "e1", "e1", "e2", "e2", "e2"], dtype=object),
    )


def test_perturbation_validation_and_label():
    assert Perturbation("orientation", "query").label == "query_orientation_per_stream"
    assert Perturbation("rate", "support", rate_hz=25).label == "support_rate_25hz"
    assert Perturbation("gravity", "query").label == "query_gravity"
    with pytest.raises(ValueError):
        Perturbation("placement", "query")
    with pytest.raises(ValueError):
        Perturbation("rate", "both")
    with pytest.raises(ValueError):
        Perturbation("orientation", "query", rotation_unit="subject")


def test_input_is_never_modified_and_identity_is_carried():
    stream = _stream()
    before = stream.windows.copy()
    out = perturb_stream(stream, Perturbation("orientation", "query"))
    assert np.array_equal(stream.windows, before)
    assert stream.perturbation is None
    assert out.perturbation == "query_orientation_per_stream"
    assert list(out.execution_ids) == list(stream.execution_ids)
    with pytest.raises(ValueError, match="already carries"):
        perturb_stream(out, Perturbation("gravity", "query"))


def test_rate_axis_resamples_and_tells_the_model_the_new_rate():
    out = perturb_stream(_stream(), Perturbation("rate", "query", rate_hz=25.0))
    assert out.rate_hz == 25.0
    assert out.windows.shape == (6, 150, 6)
    assert compatibility_relation(_stream(), out) == "identical", "rate is not in the key"
    from baselines.base import UnsupportedEvaluationCell
    with pytest.raises(UnsupportedEvaluationCell, match="rate shift needs"):
        perturb_stream(_stream(), Perturbation("rate", "query", rate_hz=60.0))


def test_channel_axis_masks_the_gyroscope_with_exact_zeros():
    base = _stream()
    out = perturb_stream(base, Perturbation("channel", "query"))
    assert out.mask.tolist() == [True, True, True, False, False, False]
    assert np.all(out.windows[..., 3:] == 0.0)
    assert np.array_equal(out.windows[..., :3], base.windows[..., :3])
    key_before, key_after = acquisition_key_for(base), acquisition_key_for(out)
    assert not are_compatible(key_before, key_after)
    assert is_near_miss(key_before, key_after)
    assert compatibility_relation(base, out) == "near_miss"
    from baselines.base import UnsupportedEvaluationCell
    with pytest.raises(UnsupportedEvaluationCell, match="accelerometer-only"):
        perturb_stream(_stream(gyro=False), Perturbation("channel", "query"))


def test_orientation_axis_is_a_proper_rotation_shared_per_unit():
    base = _stream()
    per_stream = perturb_stream(base, Perturbation("orientation", "query", rotation_unit="stream"))
    norms = lambda s: np.linalg.norm(s.windows[..., :3], axis=-1)  # noqa: E731
    assert np.allclose(norms(per_stream), norms(base), atol=1e-4)
    assert not np.allclose(per_stream.windows, base.windows)
    # windows 0 and 1 are identical inputs: one rotation per stream keeps them identical
    assert np.allclose(per_stream.windows[0], per_stream.windows[1])
    per_execution = perturb_stream(
        base, Perturbation("orientation", "query", rotation_unit="execution"),
    )
    # ... one rotation per execution separates them (e0 vs e1) ...
    assert not np.allclose(per_execution.windows[0], per_execution.windows[1])
    # ... while windows of one execution share a rotation (windows 3,4,5 are e2).
    rotated = per_execution.windows[3:6, :, :3]
    original = base.windows[3:6, :, :3]
    fitted = np.linalg.lstsq(original.reshape(-1, 3), rotated.reshape(-1, 3), rcond=None)[0]
    assert np.allclose(original.reshape(-1, 3) @ fitted, rotated.reshape(-1, 3), atol=1e-4)
    assert compatibility_relation(base, per_stream) == "identical"


def test_orientation_is_deterministic_under_seed():
    a = perturb_stream(_stream(), Perturbation("orientation", "query", seed=1))
    b = perturb_stream(_stream(), Perturbation("orientation", "query", seed=1))
    c = perturb_stream(_stream(), Perturbation("orientation", "query", seed=2))
    assert a.perturbation != c.perturbation
    assert np.array_equal(a.windows, b.windows)
    assert not np.allclose(a.windows, c.windows)


def test_gravity_axis_removes_the_dc_component_and_relabels_the_view():
    base = _stream()
    out = perturb_stream(base, Perturbation("gravity", "query"))
    assert out.gravity_state == "removed"
    assert abs(float(out.windows[..., 2].mean())) < 0.05, "gravity DC should be gone"
    assert np.array_equal(out.windows[..., 3:], base.windows[..., 3:]), "gyro untouched"
    assert out.channel_descriptions is not None
    assert all("gravity removed" in text for text in out.channel_descriptions[:3])
    assert compatibility_relation(base, out) == "near_miss"
    from baselines.base import UnsupportedEvaluationCell
    with pytest.raises(UnsupportedEvaluationCell, match="gravity-present"):
        perturb_stream(out.__class__(**{**out.__dict__, "perturbation": None}),
                       Perturbation("gravity", "query"))


def test_acquisition_key_reads_the_curated_spec():
    key = acquisition_key_for(_stream())
    assert key.device_family == "phone" and key.site == "waist"
    assert key.gravity_state == "present"
    from data.scripts.curate.compatibility import stream_key

    assert key == stream_key("inclusivehar", "phone_waist"), "unperturbed view == curated key"
    assert len(key.channels) == 6
