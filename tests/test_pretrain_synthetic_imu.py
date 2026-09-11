"""Offline tests for the synthetic virtual-IMU pretraining module.

No motion-capture data and no gated body models are needed: the physics is
driven with motions whose accelerometer/gyroscope readings are known in closed
form, and the SMPL adapter is exercised against a hand-built toy body model.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.pretraining.synthetic_imu import convert, smpl_adapter
from data.pretraining.synthetic_imu.synthesis import (
    DEFAULT_PLACEMENTS,
    GRAVITY_M_S2,
    PLACEMENTS,
    RealismConfig,
    SynthesisConfig,
    Trajectory,
    angular_velocity,
    matrix_to_quaternion,
    quaternion_to_rotvec,
    resample_trajectory,
    rotation_exp,
    rotation_log,
    smoothing_response,
    synthesize,
)

RATE = 60.0
UP = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _trajectory(positions: np.ndarray, orientations: np.ndarray | None = None,
                placements=DEFAULT_PLACEMENTS, rate: float = RATE) -> Trajectory:
    """Broadcast a single-site motion to every placement."""
    num_sites = len(placements)
    if positions.ndim == 2:
        positions = np.repeat(positions[:, None, :], num_sites, axis=1)
    if orientations is None:
        orientations = np.tile(np.eye(3), (positions.shape[0], num_sites, 1, 1))
    elif orientations.ndim == 3:
        orientations = np.repeat(orientations[:, None], num_sites, axis=1)
    return Trajectory(positions, orientations, rate, tuple(placements))


def _times(count: int, rate: float = RATE) -> np.ndarray:
    return np.arange(count, dtype=np.float64) / rate


# --------------------------------------------------------------------------- #
# rotation utilities
# --------------------------------------------------------------------------- #


def test_rotation_exp_log_roundtrip_including_near_pi():
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(200, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = np.concatenate(
        [rng.uniform(0.0, np.pi, 196), np.array([1e-9, 1e-7, np.pi - 1e-6, np.pi - 1e-9])]
    )
    rotvecs = axes * angles[:, None]
    matrices = rotation_exp(rotvecs)

    identity = np.einsum("...ij,...kj->...ik", matrices, matrices)
    assert np.allclose(identity, np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(matrices), 1.0, atol=1e-12)

    recovered = rotation_log(matrices)
    # log is only defined up to the 2*pi wrap; compare the rotations, not the vectors.
    assert np.allclose(rotation_exp(recovered), matrices, atol=1e-7)


def test_quaternion_roundtrip():
    rng = np.random.default_rng(1)
    rotvecs = rng.normal(scale=0.7, size=(50, 3))
    matrices = rotation_exp(rotvecs)
    assert np.allclose(quaternion_to_rotvec(matrix_to_quaternion(matrices)), rotvecs, atol=1e-10)


# --------------------------------------------------------------------------- #
# physics: the four cases with closed-form answers
# --------------------------------------------------------------------------- #


def test_stationary_body_reads_one_g_on_every_placement():
    count = 300
    imu = synthesize(_trajectory(np.zeros((count, 3))), SynthesisConfig(up_axis="z"))

    magnitude = np.linalg.norm(imu.acc_g, axis=-1)
    assert magnitude.shape[1] == len(DEFAULT_PLACEMENTS)
    assert np.allclose(magnitude, 1.0, atol=1e-9), "a resting accelerometer must read 1 g"
    # Sensor frame == world frame here, so the whole 1 g sits on +z (world up).
    assert np.allclose(imu.acc_g[..., 2], 1.0, atol=1e-9)
    assert np.allclose(imu.acc_g[..., :2], 0.0, atol=1e-9)
    assert np.allclose(imu.gyro_rad_s, 0.0, atol=1e-12)


def test_free_fall_reads_zero_g_not_two_g():
    """The classic sign error.

    A body in free fall has world acceleration ``-g``; specific force is
    ``a - g_vec = -G*up + G*up = 0``.  Getting the sign backwards gives 2 g, and
    getting the whole gravity term backwards gives -1 g, so this test pins both.
    """
    count = 300
    times = _times(count)
    positions = np.zeros((count, 3))
    positions[:, 2] = 10.0 - 0.5 * GRAVITY_M_S2 * times**2      # falling along -z
    imu = synthesize(_trajectory(positions), SynthesisConfig(up_axis="z"))

    assert np.allclose(imu.acc_g, 0.0, atol=1e-9), (
        "free fall must read 0 g; a reading near 2 g means gravity was subtracted "
        "with the wrong sign, and -1 g means the gravity vector itself is flipped"
    )
    assert np.abs(imu.acc_g).max() < 1e-6
    assert not np.allclose(np.linalg.norm(imu.acc_g, axis=-1), 2.0)


def test_constant_linear_velocity_reads_gravity_only():
    count = 300
    times = _times(count)
    positions = np.stack([1.4 * times, -0.6 * times, 0.2 * times], axis=-1)
    imu = synthesize(_trajectory(positions), SynthesisConfig(up_axis="z"))

    assert np.allclose(np.linalg.norm(imu.acc_g, axis=-1), 1.0, atol=1e-9)
    assert np.allclose(imu.acc_g[..., 2], 1.0, atol=1e-9)
    assert np.allclose(imu.gyro_rad_s, 0.0, atol=1e-12)


@pytest.mark.parametrize("axis, rate_rad_s", [(0, 1.7), (1, -2.3), (2, 0.9)])
def test_constant_rotation_recovers_rate_and_axis(axis, rate_rad_s):
    count = 400
    times = _times(count)
    rotvec = np.zeros((count, 3))
    rotvec[:, axis] = rate_rad_s * times
    orientations = rotation_exp(rotvec)
    imu = synthesize(
        _trajectory(np.zeros((count, 3)), orientations, placements=("pelvis",)),
        SynthesisConfig(up_axis="z"),
    )

    gyro = imu.gyro_rad_s[:, 0, :]
    # Sign convention: omega = log(R[t]^T R[t+s]) / (s dt) is the rate in the
    # SENSOR frame, and rotating the body about +axis at +rate must read +rate.
    assert np.allclose(gyro[:, axis], rate_rad_s, atol=1e-9), (
        f"expected {rate_rad_s} rad/s about axis {axis}, got {gyro[:, axis].mean()}"
    )
    others = [i for i in range(3) if i != axis]
    assert np.allclose(gyro[:, others], 0.0, atol=1e-9)
    assert np.sign(gyro[:, axis].mean()) == np.sign(rate_rad_s)

    # Rotating about world up with the body upright keeps gravity on the sensor's
    # z axis; rotating about a horizontal axis sweeps it. Either way |a| == 1 g.
    assert np.allclose(np.linalg.norm(imu.acc_g, axis=-1), 1.0, atol=1e-6)


def test_circular_motion_matches_centripetal_acceleration():
    radius, frequency = 0.30, 0.5
    omega = 2.0 * np.pi * frequency
    count = 900
    times = _times(count)
    positions = np.stack(
        [radius * np.cos(omega * times), radius * np.sin(omega * times), np.full(count, 1.2)],
        axis=-1,
    )
    # smooth_n=1 is the plain central difference, so the only error is the
    # O((omega*dt)^2) truncation of the stencil itself.
    imu = synthesize(
        _trajectory(positions, placements=("pelvis",)),
        SynthesisConfig(up_axis="z", smooth_n=1, gyro_stride=1),
    )

    force = imu.acc_g[:, 0, :] * GRAVITY_M_S2          # back to m/s^2
    horizontal = force[:, :2]                          # gravity is entirely on z
    assert np.allclose(force[:, 2], GRAVITY_M_S2, atol=1e-9)

    expected = omega**2 * radius
    assert np.allclose(np.linalg.norm(horizontal, axis=-1), expected, rtol=1e-3), (
        f"centripetal magnitude should be omega^2 r = {expected:.4f} m/s^2"
    )
    # ...and it must point at the centre, i.e. opposite the radius vector.
    radial = positions[1:-1, :2] / radius
    assert np.all(np.sum(horizontal * radial, axis=-1) < 0.0)


# --------------------------------------------------------------------------- #
# smoothing: attenuates, but does not shift
# --------------------------------------------------------------------------- #


def test_smoothing_introduces_no_phase_lag():
    amplitude, frequency = 0.05, 1.0
    omega = 2.0 * np.pi * frequency
    count = 1200
    times = _times(count)
    positions = np.zeros((count, 3))
    positions[:, 0] = amplitude * np.sin(omega * times)

    config = SynthesisConfig(up_axis="z")                  # the smoothed default
    imu = synthesize(_trajectory(positions, placements=("pelvis",)), config)

    pad = config.pad
    reference = -amplitude * omega**2 * np.sin(omega * times)[pad : count - pad]
    measured = imu.acc_g[:, 0, 0] * GRAVITY_M_S2

    # Zero-lag correlation must be the maximum over a +/- 12-sample search.
    def _correlation(shift: int) -> float:
        a = measured[12 + shift : len(measured) - 12 + shift]
        b = reference[12 : len(reference) - 12]
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    lags = np.arange(-12, 13)
    scores = np.array([_correlation(int(s)) for s in lags])
    assert lags[int(np.argmax(scores))] == 0, "smoothing must be zero-phase (symmetric stencil)"
    assert scores.max() > 0.9999

    # Amplitude is attenuated by exactly the documented sinc^2 response.
    gain = float(np.linalg.norm(measured) / np.linalg.norm(reference))
    assert np.isclose(gain, float(smoothing_response(frequency, config)), rtol=2e-3)


def test_smoothing_default_keeps_the_null_outside_our_analysis_band():
    """The reason the default is smooth_n=2 rather than TransPose's 4.

    Our tokenizer analyses up to ~14 Hz. The stencil's first transfer null sits
    at ``rate / n``: 15 Hz for n=4, which is *inside* that band, versus 30 Hz
    (Nyquist) for n=2. These are the exact numbers quoted in the docstring and
    README, so if either drifts this test fails.
    """
    default = SynthesisConfig()
    assert default.smooth_n == 2
    assert smoothing_response(1.0, default) > 0.996        # <0.4% loss at 1 Hz
    assert smoothing_response(5.0, default) == pytest.approx(0.912, abs=0.005)
    assert smoothing_response(14.0, default) == pytest.approx(0.460, abs=0.005)
    assert smoothing_response(30.0, default) < 1e-6        # null at rate / n == Nyquist

    transpose = SynthesisConfig(smooth_n=4)
    assert smoothing_response(5.0, transpose) == pytest.approx(0.684, abs=0.005)
    assert smoothing_response(15.0, transpose) < 1e-6      # null INSIDE our band
    assert smoothing_response(14.0, default) > smoothing_response(14.0, transpose)


def test_unsmoothed_stencil_is_more_faithful_at_high_frequency():
    assert smoothing_response(6.0, SynthesisConfig(smooth_n=1)) > smoothing_response(
        6.0, SynthesisConfig(smooth_n=2)
    ) > smoothing_response(6.0, SynthesisConfig(smooth_n=4))


# --------------------------------------------------------------------------- #
# structure, determinism, resampling
# --------------------------------------------------------------------------- #


def test_edges_are_trimmed_never_zero_padded():
    count = 300
    config = SynthesisConfig(up_axis="z", smooth_n=4, gyro_stride=1)
    imu = synthesize(_trajectory(np.zeros((count, 3))), config)
    assert imu.num_frames == count - 2 * config.pad
    # Zero-padding would have left 0 g (a fabricated free fall) at the ends.
    assert np.isclose(np.linalg.norm(imu.acc_g[0, 0]), 1.0)
    assert np.isclose(np.linalg.norm(imu.acc_g[-1, 0]), 1.0)


def test_defaults_are_deterministic_and_realism_is_off():
    config = SynthesisConfig()
    assert config.realism.enabled is False
    positions = np.cumsum(np.random.default_rng(3).normal(scale=1e-3, size=(300, 3)), axis=0)
    first = synthesize(_trajectory(positions), config)
    second = synthesize(_trajectory(positions), config)
    assert np.array_equal(first.acc_g, second.acc_g)
    assert np.array_equal(first.gyro_rad_s, second.gyro_rad_s)

    noisy = synthesize(
        _trajectory(positions), SynthesisConfig(realism=RealismConfig(enabled=True, seed=7))
    )
    assert not np.allclose(noisy.acc_g, first.acc_g)
    repeat = synthesize(
        _trajectory(positions), SynthesisConfig(realism=RealismConfig(enabled=True, seed=7))
    )
    assert np.array_equal(noisy.acc_g, repeat.acc_g), "a fixed seed must reproduce"


def test_resampling_preserves_the_physics():
    """A 30 fps source upsampled to 60 Hz still reads 1 g and the right rate."""
    count = 200
    times = _times(count, 30.0)
    rotvec = np.zeros((count, 3))
    rotvec[:, 2] = 1.1 * times
    trajectory = _trajectory(np.zeros((count, 3)), rotation_exp(rotvec),
                             placements=("pelvis",), rate=30.0)
    imu = synthesize(trajectory, SynthesisConfig(up_axis="z"))

    assert imu.rate_hz == 60.0
    assert np.allclose(np.linalg.norm(imu.acc_g, axis=-1), 1.0, atol=1e-6)
    assert np.allclose(imu.gyro_rad_s[:, 0, 2], 1.1, atol=1e-6)


def test_resample_is_identity_at_the_native_rate():
    trajectory = _trajectory(np.zeros((50, 3)), placements=("pelvis",))
    assert resample_trajectory(trajectory, RATE) is trajectory


def test_downsampling_lowpasses_high_frequency_position_jitter():
    """A 50 Hz mocap artefact cannot alias into a 60 Hz synthetic IMU stream."""
    source_rate, target_rate = 120.0, 60.0
    times = _times(1200, source_rate)
    positions = np.zeros((len(times), 3))
    positions[:, 0] = 1e-3 * np.sin(2.0 * np.pi * 50.0 * times)
    source = _trajectory(positions, placements=("pelvis",), rate=source_rate)
    decimated = resample_trajectory(source, target_rate)
    # The 50 Hz tone is above the target Nyquist (30 Hz), so it must be strongly removed.
    assert np.std(decimated.positions[:, 0, 0]) < 1e-5


def test_angular_velocity_forward_form_matches_the_centred_form_at_constant_rate():
    count = 100
    rotvec = np.zeros((count, 3))
    rotvec[:, 1] = -0.8 * _times(count)
    orientations = rotation_exp(rotvec)
    centred = angular_velocity(orientations, RATE, stride=1, centred=True)
    forward = angular_velocity(orientations, RATE, stride=1, centred=False)
    assert np.allclose(centred[:, 1], -0.8, atol=1e-9)
    assert np.allclose(forward[:, 1], -0.8, atol=1e-9)


def test_up_axis_is_honoured():
    imu = synthesize(_trajectory(np.zeros((100, 3)), placements=("pelvis",)),
                     SynthesisConfig(up_axis="y"))
    assert np.allclose(imu.acc_g[..., 1], 1.0, atol=1e-9)
    assert np.allclose(imu.acc_g[..., 2], 0.0, atol=1e-9)


def test_too_short_a_sequence_raises_rather_than_padding():
    config = SynthesisConfig()                       # pad == 2
    # Exactly 2*pad frames leaves nothing after trimming, so it must refuse
    # rather than invent padded samples.
    with pytest.raises(ValueError, match="frames"):
        synthesize(_trajectory(np.zeros((2 * config.pad, 3)), placements=("pelvis",)), config)
    kept = synthesize(_trajectory(np.zeros((2 * config.pad + 1, 3)), placements=("pelvis",)), config)
    assert kept.num_frames == 1


# --------------------------------------------------------------------------- #
# placements
# --------------------------------------------------------------------------- #


def test_eight_placements_with_virt_tokens_and_declared_provenance():
    assert len(DEFAULT_PLACEMENTS) == 8
    assert set(DEFAULT_PLACEMENTS) == {
        "head", "sternum", "pelvis", "lforearm", "rforearm", "rupperarm", "rthigh", "rshank",
    }
    for name in DEFAULT_PLACEMENTS:
        spec = PLACEMENTS[name]
        assert spec.token == f"virt_{name}"
        assert spec.provenance, f"{name} has no provenance string"
        if spec.smpl_vertex is None:
            assert spec.provenance.startswith("UNVERIFIED"), (
                f"{name} has no vertex index, so its provenance must say so plainly"
            )
        else:
            assert spec.provenance.startswith("VERIFIED"), (
                f"{name} asserts vertex {spec.smpl_vertex} and must cite a published source"
            )
            assert 0 <= spec.smpl_vertex < 6890
    # The six TransPose indices, verbatim from preprocess.py L34-35.
    assert PLACEMENTS["lforearm"].smpl_vertex == 1961 and PLACEMENTS["lforearm"].smpl_joint == 18
    assert PLACEMENTS["rforearm"].smpl_vertex == 5424 and PLACEMENTS["rforearm"].smpl_joint == 19
    assert PLACEMENTS["rshank"].smpl_vertex == 4662 and PLACEMENTS["rshank"].smpl_joint == 5
    assert PLACEMENTS["head"].smpl_vertex == 411 and PLACEMENTS["head"].smpl_joint == 15
    assert PLACEMENTS["pelvis"].smpl_vertex == 3021 and PLACEMENTS["pelvis"].smpl_joint == 0
    # ...and the IMUPoser thigh index.
    assert PLACEMENTS["rthigh"].smpl_vertex == 4362 and PLACEMENTS["rthigh"].smpl_joint == 2
    assert PLACEMENTS["sternum"].smpl_vertex is None
    assert PLACEMENTS["rupperarm"].smpl_vertex is None


# --------------------------------------------------------------------------- #
# BVH path (no body model required)
# --------------------------------------------------------------------------- #


_TINY_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine3
  {
    OFFSET 0.0 40.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Head
    {
      OFFSET 0.0 25.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0.0 10.0 0.0
      }
    }
  }
}
MOTION
Frames: @COUNT@
Frame Time: 0.016667
@ROWS@
"""


def _bvh(rows: list[str]) -> str:
    return _TINY_BVH.replace("@COUNT@", str(len(rows))).replace("@ROWS@", "\n".join(rows))


def _tiny_bvh(count: int = 900) -> str:
    return _bvh(
        [" ".join(["0.0", "100.0", f"{0.5 * index:.4f}"] + ["0.0"] * 9) for index in range(count)]
    )


def test_bvh_parser_and_forward_kinematics():
    joints, motion, frame_time = convert.parse_bvh(_tiny_bvh(10))
    assert [j.name for j in joints] == ["Hips", "Spine3", "Head"]
    assert joints[2].parent == 1
    assert motion.shape == (10, 12)
    assert np.isclose(1.0 / frame_time, 60.0, atol=0.01)

    positions, rotations = convert.bvh_forward_kinematics(joints, motion)
    # Offsets are centimetres by default: head sits 0.65 m above the root.
    assert np.allclose(positions[:, 2, 1] - positions[:, 0, 1], 0.65, atol=1e-9)
    assert np.allclose(rotations, np.eye(3), atol=1e-12)


def test_bvh_trajectory_resolves_placements_by_joint_name():
    joints, motion, frame_time = convert.parse_bvh(_tiny_bvh(300))
    positions, rotations = convert.bvh_forward_kinematics(joints, motion)
    trajectory = convert.trajectory_from_bvh(joints, positions, rotations, 1.0 / frame_time)
    assert set(trajectory.placements) == {"head", "sternum", "pelvis"}

    imu = synthesize(trajectory, SynthesisConfig(up_axis="y"))
    # Constant-velocity translation along +z: gravity only, on the y axis.
    assert np.allclose(np.linalg.norm(imu.acc_g, axis=-1), 1.0, atol=1e-6)


def test_motionbuilder_bvh_joint_aliases_cover_all_virtual_sites():
    """100STYLE uses MotionBuilder names at the proximal end of each limb bone."""
    names = {
        "Hips", "Chest", "Head", "LeftElbow", "RightElbow", "RightShoulder",
        "RightHip", "RightKnee",
    }
    resolved = {
        placement
        for placement, spec in PLACEMENTS.items()
        if names.intersection(spec.bvh_joint_candidates)
    }
    assert resolved == set(DEFAULT_PLACEMENTS)


def test_bvh_rotation_channel_order_is_applied_intrinsically():
    """Zrotation then Xrotation must compose as Rz @ Rx, not Rx @ Rz."""
    text = _bvh([" ".join(["0.0"] * 3 + ["90.0", "45.0", "0.0"] + ["0.0"] * 6)] * 3)
    joints, motion, _ = convert.parse_bvh(text)
    _, rotations = convert.bvh_forward_kinematics(joints, motion)
    expected = rotation_exp(np.array([0.0, 0.0, np.pi / 2])) @ rotation_exp(np.array([np.pi / 4, 0.0, 0.0]))
    assert np.allclose(rotations[0, 0], expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# SMPL adapter (toy model offline; real model behind a skip guard)
# --------------------------------------------------------------------------- #


def _toy_body_model(num_joints: int = 4, num_vertices: int = 12) -> smpl_adapter.BodyModel:
    rng = np.random.default_rng(11)
    v_template = rng.normal(scale=0.3, size=(num_vertices, 3))
    j_regressor = np.zeros((num_joints, num_vertices))
    for joint in range(num_joints):          # each joint is the mean of 3 vertices
        j_regressor[joint, joint * 3 : joint * 3 + 3] = 1.0 / 3.0
    weights = np.zeros((num_vertices, num_joints))
    for vertex in range(num_vertices):
        weights[vertex, vertex // 3] = 1.0   # rigidly bound, so LBS == the joint transform
    return smpl_adapter.BodyModel(
        v_template=v_template,
        shapedirs=np.zeros((num_vertices, 3, 2)),
        posedirs=np.zeros((num_vertices, 3, 9 * (num_joints - 1))),
        j_regressor=j_regressor,
        weights=weights,
        parents=np.array([-1, 0, 1, 2]),
        model_type="toy",
        gender="neutral",
    )


def test_smpl_forward_rest_pose_is_the_template_plus_translation():
    model = _toy_body_model()
    count = 5
    poses = np.zeros((count, 3 * model.num_joints))
    trans = np.tile(np.array([0.1, -0.2, 1.0]), (count, 1))
    vertices, rotations = smpl_adapter.forward(model, poses, trans, np.zeros(2), [0, 4, 9])

    assert np.allclose(rotations, np.eye(3), atol=1e-12)
    assert np.allclose(vertices, model.v_template[[0, 4, 9]] + trans[:, None, :], atol=1e-12)


def test_smpl_forward_root_rotation_rotates_the_whole_body():
    model = _toy_body_model()
    count = 4
    poses = np.zeros((count, 3 * model.num_joints))
    poses[:, 2] = np.pi / 3                     # yaw the root
    trans = np.zeros((count, 3))
    vertices, rotations = smpl_adapter.forward(model, poses, trans, np.zeros(2), [0, 4, 9])

    yaw = rotation_exp(np.array([0.0, 0.0, np.pi / 3]))
    assert np.allclose(rotations[:, 0], yaw, atol=1e-12)
    assert np.allclose(rotations[:, 3], yaw, atol=1e-12)   # inherited down the chain
    # With only the root rotated, the whole body is rigidly yawed about the root
    # JOINT centre (not the world origin): v' = R (v - J0) + J0.
    root = model.j_regressor[0] @ model.v_template
    expected = (yaw @ (model.v_template[[0, 4, 9]] - root).T).T + root
    assert np.allclose(vertices, expected, atol=1e-12)


def test_smpl_forward_rejects_a_short_pose_vector():
    model = _toy_body_model()
    with pytest.raises(ValueError, match="pose vector"):
        smpl_adapter.forward(model, np.zeros((3, 6)), np.zeros((3, 3)), np.zeros(2), [0])


def test_missing_body_model_raises_a_pointed_error(tmp_path):
    assert smpl_adapter.find_model_path("smplh", "neutral", tmp_path) is None
    with pytest.raises(smpl_adapter.BodyModelMissing, match="registration-gated"):
        smpl_adapter.load_body_model("smplh", "neutral", tmp_path)


@pytest.mark.skipif(
    smpl_adapter.find_model_path("smplh", "neutral") is None,
    reason="gated SMPL-H body model not present (expected in CI and on a fresh clone)",
)
def test_real_smplh_model_smoke():
    model = smpl_adapter.load_body_model("smplh", "neutral")
    assert model.v_template.shape == (6890, 3)
    assert model.num_joints >= 22
    count = 200
    poses = np.zeros((count, 3 * model.num_joints))
    trans = np.zeros((count, 3))
    trajectory = smpl_adapter.trajectory_from_smpl(model, poses, trans, np.zeros(10), 60.0)
    assert trajectory.placements == DEFAULT_PLACEMENTS
    imu = synthesize(trajectory, SynthesisConfig(up_axis="z"))
    assert np.allclose(np.linalg.norm(imu.acc_g, axis=-1), 1.0, atol=1e-6)


def test_embody3d_refuses_to_run(tmp_path):
    with pytest.raises(NotImplementedError, match="documented placeholder"):
        list(convert.iter_embody3d(tmp_path, DEFAULT_PLACEMENTS))
    assert "embody3d" in convert.SOURCES and "embody3d" not in {"amass", "motion_x", "100style"}


# --------------------------------------------------------------------------- #
# end-to-end output contract
# --------------------------------------------------------------------------- #


def _synthetic_sequence(seconds: float = 12.0, name: str = "run01") -> convert.SourceSequence:
    count = int(seconds * RATE)
    times = _times(count)
    positions = np.zeros((count, len(DEFAULT_PLACEMENTS), 3))
    orientations = np.zeros((count, len(DEFAULT_PLACEMENTS), 3, 3))
    for site in range(len(DEFAULT_PLACEMENTS)):
        positions[:, site, 0] = 0.4 * np.sin(2 * np.pi * (0.7 + 0.1 * site) * times)
        positions[:, site, 2] = 1.0 + 0.05 * site
        rotvec = np.zeros((count, 3))
        rotvec[:, site % 3] = 0.6 * np.sin(2 * np.pi * 0.5 * times)
        orientations[:, site] = rotation_exp(rotvec)
    return convert.SourceSequence(
        source="amass",
        subject="amass_CMU_01",
        name=name,
        trajectory=Trajectory(positions, orientations, RATE, DEFAULT_PLACEMENTS),
        up_axis="z",
    )


def test_end_to_end_output_contract(tmp_path):
    stats = convert.emit_sessions([_synthetic_sequence()], output_dir=tmp_path)

    assert stats["sessions"] == 8
    assert stats["failed"] == 0

    labels = json.loads((tmp_path / "labels.json").read_text())
    assert len(labels) == 8
    assert all(value == ["__unlabeled__"] for value in labels.values())

    expected = "synthimu_amass_CMU_01_run01_virt_rthigh"
    assert expected in labels, sorted(labels)
    for session in labels:
        assert session.startswith("synthimu_")
        assert "_virt_" in session, "every session id must carry a virt_ stream token"
        assert any(session.endswith(PLACEMENTS[p].token) for p in DEFAULT_PLACEMENTS)

    frame = pd.read_parquet(tmp_path / "sessions" / expected / "data.parquet")
    assert list(frame.columns) == [
        "timestamp_sec", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "subject",
    ]
    assert frame["timestamp_sec"].dtype == np.float64
    for channel in convert.OUTPUT_COLUMNS:
        assert frame[channel].dtype == np.float32, channel
    assert frame["subject"].map(type).eq(str).all()
    assert frame["subject"].nunique() == 1
    assert frame["subject"].iloc[0] == "amass_CMU_01"

    stamps = frame["timestamp_sec"].to_numpy()
    assert stamps[0] == 0.0
    assert np.allclose(np.diff(stamps), 1.0 / 60.0, atol=1e-12), "timestamps must be uniform 1/60 s"
    assert len(frame) >= 6 * 60

    # Physically plausible: gravity present, so |acc| sits around 1 g.
    magnitude = np.linalg.norm(frame[["acc_x", "acc_y", "acc_z"]].to_numpy(), axis=-1)
    assert 0.2 < magnitude.mean() < 5.0
    assert np.isfinite(frame[list(convert.OUTPUT_COLUMNS)].to_numpy()).all()


def test_end_to_end_manifest_and_metadata(tmp_path):
    convert.emit_sessions([_synthetic_sequence()], output_dir=tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert {
        "dataset_name", "source", "num_subjects", "sampling_rate_hz",
        "channels", "unit", "gravity_state", "phase_a_only", "note",
    } <= set(manifest)
    assert manifest["sampling_rate_hz"] == 60.0
    assert manifest["gravity_state"] == "present"
    assert manifest["phase_a_only"] is True
    assert manifest["num_subjects"] == 1
    assert manifest["channels"] == list(convert.OUTPUT_COLUMNS)
    assert manifest["unit"] == "acc: g; gyro: rad/s"
    assert "2602.11064" in manifest["note"], "the sim-to-real caveat must travel with the data"

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert {
        "dataset", "display_name", "sampling_rate_hz", "pre_windowed", "streaming_grid",
        "role", "phase_a_only", "synthetic", "activities", "num_subjects",
        "channels", "core_channels", "placement", "note",
    } <= set(metadata)
    assert metadata["dataset"] == "synthetic_imu"
    assert metadata["sampling_rate_hz"] == 60.0
    assert metadata["pre_windowed"] is False
    assert metadata["streaming_grid"] is True
    assert metadata["role"] == "pretrain_scale"
    assert metadata["synthetic"] is True
    assert metadata["activities"] == []
    assert metadata["num_subjects"] is None
    assert sorted(metadata["placement"]) == sorted(DEFAULT_PLACEMENTS)


def test_short_sequences_are_dropped_not_concatenated(tmp_path):
    stats = convert.emit_sessions(
        [_synthetic_sequence(seconds=3.0, name="tooshort"), _synthetic_sequence(name="keeper")],
        output_dir=tmp_path,
    )
    assert stats["sessions"] == 8
    assert stats["dropped_short"] == 8
    sessions = sorted(p.name for p in (tmp_path / "sessions").iterdir())
    assert all("tooshort" not in name for name in sessions)
    assert all("keeper" in name for name in sessions)


def test_session_ids_are_unique_across_sequences_and_subjects(tmp_path):
    sequences = [
        _synthetic_sequence(name="run01"),
        _synthetic_sequence(name="run02"),
    ]
    stats = convert.emit_sessions(sequences, output_dir=tmp_path)
    assert stats["sessions"] == 16
    assert len(json.loads((tmp_path / "labels.json").read_text())) == 16


def test_no_sessions_written_when_everything_is_too_short(tmp_path):
    stats = convert.emit_sessions([_synthetic_sequence(seconds=2.0)], output_dir=tmp_path)
    assert stats["sessions"] == 0
    assert not (tmp_path / "manifest.json").exists()


def test_100style_loader_end_to_end_from_a_bvh_on_disk(tmp_path):
    """The one source that needs no gated body model, driven through the real loader."""
    style_dir = tmp_path / "downloads" / "100style" / "Neutral"
    style_dir.mkdir(parents=True)
    (style_dir / "Neutral_FW.bvh").write_text(_tiny_bvh(600))

    sequences = list(convert.iter_100style(tmp_path / "downloads" / "100style", DEFAULT_PLACEMENTS))
    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.source == "100style"
    assert sequence.subject == "100style_actor_01"
    assert sequence.up_axis == "y"          # BVH is y-up, unlike AMASS
    assert set(sequence.trajectory.placements) == {"head", "sternum", "pelvis"}

    out = tmp_path / "out"
    stats = convert.emit_sessions(sequences, output_dir=out)
    assert stats["sessions"] == 3
    ids = sorted(json.loads((out / "labels.json").read_text()))
    assert ids == [
        "synthimu_100style_actor_01_neutral_fw_virt_head",
        "synthimu_100style_actor_01_neutral_fw_virt_pelvis",
        "synthimu_100style_actor_01_neutral_fw_virt_sternum",
    ]
    frame = pd.read_parquet(out / "sessions" / ids[0] / "data.parquet")
    # Constant-velocity walk-through: gravity only, and it lands on the y axis.
    assert np.allclose(frame["acc_y"].to_numpy(), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(frame[["acc_x", "acc_y", "acc_z"]].to_numpy(), axis=-1), 1.0, atol=1e-5)


def test_100style_loader_ignores_macos_archive_metadata(tmp_path):
    root = tmp_path / "100style"
    real = root / "Neutral" / "Neutral_FW.bvh"
    metadata = root / "__MACOSX" / "Neutral" / "._Neutral_FW.bvh"
    real.parent.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    real.write_text(_tiny_bvh(300))
    metadata.write_bytes(b"not a BVH motion file")

    sequences = list(convert.iter_100style(root, DEFAULT_PLACEMENTS))
    assert len(sequences) == 1
    assert sequences[0].name == "neutral_fw"


def test_missing_source_directory_is_reported_not_fatal(tmp_path, capsys):
    assert list(convert.iter_sources(["amass", "100style"], tmp_path, DEFAULT_PLACEMENTS, None)) == []
    printed = capsys.readouterr().out
    assert "amass" in printed and "100style" in printed
