"""Virtual-IMU synthesis from body-motion trajectories (pure, no I/O).

This module turns *kinematics* — a time series of world-frame sensor positions
and world-frame segment orientations — into what a strapdown accelerometer and
rate gyroscope bolted to those segments would have recorded.  It is deliberately
ignorant of SMPL, BVH, AMASS and files: the corpus adapters
(:mod:`data.pretraining.synthetic_imu.smpl_adapter`, the BVH reader in
:mod:`data.pretraining.synthetic_imu.convert`) are responsible for producing a
:class:`Trajectory`, and everything below is unit-testable against motions with
closed-form answers.

Method (TransPose / DIP / IMUPoser / UniMTS all use this recipe)
---------------------------------------------------------------
Let ``p[t]`` be the world position of the sensor site (m) and ``R[t]`` the
rotation taking sensor-frame vectors into the world frame (segment orientation).

1. **World acceleration** by a symmetric second difference with stride ``n``::

       a_world[t] = (p[t-n] - 2 p[t] + p[t+n]) / (n * dt)^2

2. **Specific force** — what an accelerometer actually measures.  A proof mass
   is held against gravity, so the sensor reads ``a - g`` with
   ``g = -G * up_hat``, i.e. gravity is *added* along world up::

       f_world[t] = a_world[t] + G * up_hat        (G = 9.80665 m/s^2)

   Stationary  -> ``f = +G * up``   (magnitude 1 g)
   Free fall   -> ``a = -G * up``  -> ``f = 0``   (magnitude 0 g)

   The free-fall case is the one everybody gets backwards; ``tests/`` asserts it.

3. **Into the sensor frame and into g**::

       acc[t] = R[t]^T f_world[t] / G          (dimensionless, units of g)

4. **Angular velocity** from the relative rotation of consecutive frames.  With
   ``R`` mapping sensor -> world, ``R[t]^T R[t+1] = exp([w]_x dt)`` where ``w``
   is expressed in the *sensor* frame, so::

       omega[t] = log(R[t]^T R[t+s]) / (s * dt)          [rad/s, sensor frame]

   The default is the *centred* form ``log(R[t-s]^T R[t+s]) / (2 s dt)``, which
   is the same estimator evaluated symmetrically about ``t`` and therefore has
   no half-sample time shift; pass ``gyro_centred=False`` for the literal
   forward difference above.

Smoothing, and why the default is what it is
--------------------------------------------
A second difference is a high-pass with gain ``(2/dt)^2`` at Nyquist: run it on
raw 60 fps mocap and the output is dominated by solver jitter, not by the body.
The standard fix, and the one in TransPose's released ``preprocess.py``, is to
widen the stencil rather than to pre-filter: differencing over ``n`` samples on
each side is algebraically ``(low-pass) * (second difference)`` with a
**symmetric, and therefore exactly zero-phase**, FIR kernel.  No group delay is
introduced, which matters because these channels are later windowed and
compared against each other and must not be sheared apart.

Its transfer function relative to an ideal ``-w^2`` differentiator is
``sinc^2(w n dt / 2)``, which is 1 at DC and has its first null at
``rate / n``.

**The default here is ``smooth_n = 2``, not TransPose's 4, and that is a
deliberate departure.**  At 60 Hz, ``n = 4`` puts the first transfer null at
15 Hz — *inside* the band our tokenizer analyses (``f_max`` ~14 Hz) — and costs
32% of the amplitude already at 5 Hz.  TransPose can afford that because it is
feeding a pose estimator whose content is essentially all below 5 Hz; we cannot.
``n = 2`` moves the null to 30 Hz (exactly Nyquist, where only aliased solver
noise lives), keeps the response within 0.4% of ideal at 1 Hz, 91% at 5 Hz and
46% at 14 Hz, and still cuts the white-noise gain of the stencil by 4x relative
to ``n = 1``.  Use ``smooth_n=4`` for bit-comparable TransPose parity and
``smooth_n=1`` for the raw central difference (what the closed-form physics
tests use, so that the assertions are against exact values).
:func:`smoothing_response` computes all of these numbers, and the tests check
them, so none of the above has to be taken on trust.

Edges are **trimmed**, not zero-padded.  TransPose pads the first and last
frames with zeros, which fabricates a 0 g reading — physically a free fall — at
every sequence boundary.  We drop ``pad = max(smooth_n, gyro_stride)`` samples
from each end instead, so no synthesised sample is an artefact of padding.

Realism knobs
-------------
Additive white noise, constant bias and placement jitter are implemented in
:class:`RealismConfig` and are **all off by default**.  With the defaults the
output is a deterministic function of the input trajectory: no RNG is touched.

Units contract (matches the repo-wide session contract)
-------------------------------------------------------
``acc`` in **g**, gravity **present**, sensor frame.  ``gyro`` in **rad/s**,
sensor frame.  Output rate 60 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Optional, Sequence

import numpy as np


# CODATA / ISO 80000 standard gravity. The same constant the repo uses to move
# between g and m/s^2 elsewhere; do not "round to 9.81" here.
GRAVITY_M_S2 = 9.80665

TARGET_RATE_HZ = 60.0

#: World "up" unit vectors for the axis conventions we meet in practice.
UP_VECTORS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


# --------------------------------------------------------------------------- #
# Placements
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Placement:
    """One virtual sensor site.

    ``smpl_vertex`` is an index into the 6890-vertex SMPL/SMPL-H template mesh;
    ``smpl_joint`` is an index into the 24-joint SMPL kinematic tree and gives
    the *segment orientation* the sensor is rigidly attached to.  ``provenance``
    records where the number came from, verbatim, so a reader can check it
    without trusting this file.
    """

    name: str
    token: str
    smpl_vertex: Optional[int]
    smpl_joint: int
    bvh_joint_candidates: tuple[str, ...]
    description: str
    provenance: str


# PROVENANCE OF THE VERTEX INDICES (verified 2026-09-09 by reading the sources)
#
# TransPose (Yi, Zhou & Xu, SIGGRAPH 2021) -- github.com/Xinyu-Yi/TransPose,
# `preprocess.py` lines 34-35 (NOT config.py, which has no such mask):
#
#     vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
#     ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
#
# ordered [left forearm, right forearm, left lower leg, right lower leg, head,
# pelvis], matching the DIP-IMU sensor order.
#
# IMUPoser -- github.com/FIGLAB/IMUPoser, `scripts/1. Preprocessing/
# 1. preprocess_all.py` lines 33-34: same arms/head/pelvis, but the leg sensors
# are at the THIGH rather than the shank:
#
#     vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
#     ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
#
# so vertex 4362 / joint 2 (right hip, which drives the right femur) is a
# published thigh index and is what `rthigh` uses.
#
# For contrast, DIP (eth-ait/dip18, `data_synthesis/genSynData.py` line 27) uses
# a *different* set -- [1962, 5431, 1096, 4583, 412, 3021] -- sharing only the
# pelvis with TransPose. Vertex choice at this level of detail is not
# standardised across the literature; we follow TransPose, plus IMUPoser for the
# thigh, and say so.
#
# `sternum` and `rupperarm` appear in NO published virtual-IMU mask, so they get
# `smpl_vertex=None` and fall back to the joint origin. A vertex index guessed
# off a mesh render is not a citation; inventing one would silently move a
# sensor by an unknown amount. The joint-origin fallback is a *documented
# approximation*, not a fabricated vertex.
PLACEMENTS: dict[str, Placement] = {
    "head": Placement(
        name="head",
        token="virt_head",
        smpl_vertex=411,
        smpl_joint=15,
        bvh_joint_candidates=("Head", "head", "Neck1"),
        description="forehead / head-mounted unit",
        provenance="VERIFIED: TransPose preprocess.py L34-35, vi_mask[4]=411 / ji_mask[4]=15 (SMPL joint 15 = head)",
    ),
    "sternum": Placement(
        name="sternum",
        token="virt_sternum",
        smpl_vertex=None,
        smpl_joint=9,
        bvh_joint_candidates=("Spine3", "Spine2", "Chest", "Spine1"),
        description="upper chest / sternum patch",
        provenance="UNVERIFIED: no published virtual-IMU mask places a sensor on the sternum. No vertex index is asserted; falls back to the SMPL joint 9 (spine3) origin, which is INSIDE the torso rather than on the skin — a documented approximation.",
    ),
    "pelvis": Placement(
        name="pelvis",
        token="virt_pelvis",
        smpl_vertex=3021,
        smpl_joint=0,
        bvh_joint_candidates=("Hips", "hip", "Pelvis"),
        description="lower back / waist unit",
        provenance="VERIFIED: TransPose preprocess.py L34-35, vi_mask[5]=3021 / ji_mask[5]=0 (SMPL joint 0 = pelvis). The one index TransPose, DIP and IMUPoser all agree on.",
    ),
    "lforearm": Placement(
        name="lforearm",
        token="virt_lforearm",
        smpl_vertex=1961,
        smpl_joint=18,
        # 100STYLE's MotionBuilder BVH calls the joint at the proximal end of
        # this segment ``LeftElbow``.  In a BVH hierarchy its local rotation
        # drives the forearm; omitting that spelling silently removed the site.
        bvh_joint_candidates=("LeftForeArm", "LeftElbow", "LeftHand", "lradius"),
        description="left wrist / forearm (watch position)",
        provenance="VERIFIED: TransPose preprocess.py L34-35, vi_mask[0]=1961 / ji_mask[0]=18 (SMPL joint 18 = left elbow, drives the left forearm)",
    ),
    "rforearm": Placement(
        name="rforearm",
        token="virt_rforearm",
        smpl_vertex=5424,
        smpl_joint=19,
        # See ``lforearm``: 100STYLE names the proximal forearm joint Elbow.
        bvh_joint_candidates=("RightForeArm", "RightElbow", "RightHand", "rradius"),
        description="right wrist / forearm (watch position)",
        provenance="VERIFIED: TransPose preprocess.py L34-35, vi_mask[1]=5424 / ji_mask[1]=19 (SMPL joint 19 = right elbow, drives the right forearm)",
    ),
    "rupperarm": Placement(
        name="rupperarm",
        token="virt_rupperarm",
        smpl_vertex=None,
        smpl_joint=17,
        # MotionBuilder's ``RightShoulder`` rotation drives the upper-arm bone.
        bvh_joint_candidates=("RightArm", "RightShoulder", "rhumerus"),
        description="right upper arm band",
        provenance="UNVERIFIED: no published virtual-IMU mask places a sensor on the upper arm. No vertex index is asserted; falls back to the SMPL joint 17 (right shoulder, which drives the right humerus) origin — a documented approximation.",
    ),
    "rthigh": Placement(
        name="rthigh",
        token="virt_rthigh",
        smpl_vertex=4362,
        smpl_joint=2,
        # MotionBuilder's ``RightHip`` rotation drives the femur.
        bvh_joint_candidates=("RightUpLeg", "RightHip", "rfemur"),
        description="right thigh (pocket / thigh-worn)",
        provenance="VERIFIED: IMUPoser preprocess_all.py L33-34, vi_mask[3]=4362 / ji_mask[3]=2 (SMPL joint 2 = right hip, drives the right femur). NOT from TransPose, whose leg sensors sit at the knee.",
    ),
    "rshank": Placement(
        name="rshank",
        token="virt_rshank",
        smpl_vertex=4662,
        smpl_joint=5,
        # MotionBuilder's ``RightKnee`` rotation drives the lower-leg bone.
        bvh_joint_candidates=("RightLeg", "RightKnee", "rtibia"),
        description="right shank / lower leg",
        provenance="VERIFIED: TransPose preprocess.py L34-35, vi_mask[3]=4662 / ji_mask[3]=5 (SMPL joint 5 = right knee, drives the right shank)",
    ),
}

DEFAULT_PLACEMENTS: tuple[str, ...] = (
    "head",
    "sternum",
    "pelvis",
    "lforearm",
    "rforearm",
    "rupperarm",
    "rthigh",
    "rshank",
)

PLACEMENT_TOKENS: tuple[str, ...] = tuple(PLACEMENTS[n].token for n in DEFAULT_PLACEMENTS)


# --------------------------------------------------------------------------- #
# Rotation utilities (numpy only; no scipy, so the physics is auditable here)
# --------------------------------------------------------------------------- #


def rotation_exp(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vectors ``(..., 3)`` -> rotation matrices ``(..., 3, 3)``."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec, axis=-1)
    small = theta < 1e-8
    safe = np.where(small, 1.0, theta)
    axis = rotvec / safe[..., None]
    # Taylor-continue the trig coefficients so theta -> 0 is exact, not 0/0.
    sin_t = np.where(small, theta - theta**3 / 6.0, np.sin(theta))
    cos_t = np.where(small, 1.0 - theta**2 / 2.0, np.cos(theta))
    kx, ky, kz = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(kx)
    skew = np.stack(
        [
            np.stack([zero, -kz, ky], axis=-1),
            np.stack([kz, zero, -kx], axis=-1),
            np.stack([-ky, kx, zero], axis=-1),
        ],
        axis=-2,
    )
    eye = np.broadcast_to(np.eye(3), skew.shape).copy()
    outer = axis[..., :, None] * axis[..., None, :]
    return (
        cos_t[..., None, None] * eye
        + sin_t[..., None, None] * skew
        + (1.0 - cos_t)[..., None, None] * outer
    )


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` -> unit quaternions ``(..., 4)`` as ``(w, x, y, z)``.

    Shepperd's branch on the largest of ``{trace, m00, m11, m22}``; the branch
    exists to avoid the catastrophic cancellation the naive trace formula
    suffers near 180 degrees.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    trace = m00 + m11 + m22

    def _branch(a, b, c, d):
        return np.stack([a, b, c, d], axis=-1)

    # Each branch: take 2*sqrt(1 + <largest diagonal-ish term>) as the pivot.
    s0 = np.sqrt(np.maximum(1.0 + trace, 1e-20)) * 2.0
    q0 = _branch(0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0)
    s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-20)) * 2.0
    q1 = _branch((m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1)
    s2 = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-20)) * 2.0
    q2 = _branch((m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2)
    s3 = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-20)) * 2.0
    q3 = _branch((m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3)

    use0 = trace > 0.0
    use1 = (~use0) & (m00 >= m11) & (m00 >= m22)
    use2 = (~use0) & (~use1) & (m11 >= m22)
    quat = np.where(
        use0[..., None],
        q0,
        np.where(use1[..., None], q1, np.where(use2[..., None], q2, q3)),
    )
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-20)


def quaternion_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Unit quaternions ``(..., 4)`` (w, x, y, z) -> rotation vectors ``(..., 3)``."""
    quat = np.asarray(quat, dtype=np.float64)
    # Canonicalise to the w >= 0 hemisphere so the returned angle is in [0, pi].
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    w = np.clip(quat[..., 0], -1.0, 1.0)
    xyz = quat[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    small = sin_half < 1e-10
    # angle / sin(angle/2) -> 2 + angle^2/12 as angle -> 0.
    scale = np.where(small, 2.0 + angle**2 / 12.0, angle / np.maximum(sin_half, 1e-20))
    return xyz * scale[..., None]


def rotation_log(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` -> rotation vectors ``(..., 3)``."""
    return quaternion_to_rotvec(matrix_to_quaternion(matrix))


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix in Frobenius norm (polar/SVD projection)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    u, _, vt = np.linalg.svd(matrix)
    result = u @ vt
    # Guard against a reflection sneaking in when the input is badly degenerate.
    det = np.linalg.det(result)
    flip = np.ones(result.shape[:-1])
    flip[..., -1] = np.sign(np.where(det == 0.0, 1.0, det))
    return u @ (flip[..., None] * vt)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RealismConfig:
    """Sensor-imperfection knobs. **All disabled by default** (``enabled=False``).

    Enabling any of them makes synthesis stochastic; ``seed`` then fixes the
    draw so a run is still reproducible.  Defaults are order-of-magnitude
    figures for a consumer MEMS IMU, not a calibration of any specific part.
    """

    enabled: bool = False
    acc_noise_g: float = 0.01          # white noise std, g
    gyro_noise_rad_s: float = 0.01     # white noise std, rad/s
    acc_bias_g: float = 0.02           # per-session constant offset, g
    gyro_bias_rad_s: float = 0.01      # per-session constant offset, rad/s
    placement_jitter_m: float = 0.0    # sensor site displacement std, metres
    orientation_jitter_rad: float = 0.0  # sensor mounting misalignment std, rad
    seed: int = 0


@dataclass(frozen=True)
class SynthesisConfig:
    """Everything that controls the physics.  See the module docstring for ``smooth_n``."""

    target_rate_hz: float = TARGET_RATE_HZ
    #: Stride of the symmetric second-difference stencil (see module docstring).
    smooth_n: int = 2
    #: Stride of the angular-velocity difference, in samples.
    gyro_stride: int = 1
    #: Evaluate the gyro difference symmetrically about t (zero phase).
    gyro_centred: bool = True
    #: World up axis of the *source* corpus. AMASS/SMPL is z-up; BVH is usually y-up.
    up_axis: str = "z"
    realism: RealismConfig = field(default_factory=RealismConfig)

    def __post_init__(self) -> None:
        if self.smooth_n < 1:
            raise ValueError("smooth_n must be >= 1")
        if self.gyro_stride < 1:
            raise ValueError("gyro_stride must be >= 1")
        if self.up_axis not in UP_VECTORS:
            raise ValueError(f"up_axis must be one of {sorted(UP_VECTORS)}, got {self.up_axis!r}")
        if self.target_rate_hz <= 0:
            raise ValueError("target_rate_hz must be positive")

    @property
    def pad(self) -> int:
        """Samples trimmed from each end (never zero-padded — see module docstring)."""
        return max(self.smooth_n, self.gyro_stride)


# --------------------------------------------------------------------------- #
# Trajectory containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Trajectory:
    """Kinematics of P virtual sensor sites over T frames.

    positions      (T, P, 3) float, world frame, **metres**
    orientations   (T, P, 3, 3) float, sensor-frame -> world-frame rotations
    rate_hz        sampling rate of the arrays above
    placements     names, len P, aligned with axis 1
    """

    positions: np.ndarray
    orientations: np.ndarray
    rate_hz: float
    placements: tuple[str, ...]

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        orientations = np.asarray(self.orientations, dtype=np.float64)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(f"positions must be (T, P, 3); got {positions.shape}")
        if orientations.shape != positions.shape[:2] + (3, 3):
            raise ValueError(
                f"orientations must be (T, P, 3, 3) matching positions {positions.shape}; "
                f"got {orientations.shape}"
            )
        if len(self.placements) != positions.shape[1]:
            raise ValueError(
                f"placements has {len(self.placements)} entries but positions has "
                f"{positions.shape[1]} sites"
            )
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "orientations", orientations)
        object.__setattr__(self, "placements", tuple(self.placements))

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.rate_hz


@dataclass(frozen=True)
class VirtualIMU:
    """Synthesised signals.  ``acc`` in g (gravity present), ``gyro`` in rad/s."""

    acc_g: np.ndarray        # (T, P, 3), sensor frame
    gyro_rad_s: np.ndarray   # (T, P, 3), sensor frame
    rate_hz: float
    placements: tuple[str, ...]

    @property
    def num_frames(self) -> int:
        return int(self.acc_g.shape[0])

    @property
    def timestamps_sec(self) -> np.ndarray:
        return np.arange(self.num_frames, dtype=np.float64) / self.rate_hz


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


def _slerp(quat_a: np.ndarray, quat_b: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Shortest-arc slerp between two arrays of quaternions."""
    dot = np.sum(quat_a * quat_b, axis=-1, keepdims=True)
    quat_b = np.where(dot < 0.0, -quat_b, quat_b)
    dot = np.abs(dot).clip(0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-8
    # Fall back to lerp when the two rotations are nearly identical.
    weight_a = np.where(near, 1.0 - weight, np.sin((1.0 - weight) * theta) / np.maximum(sin_theta, 1e-20))
    weight_b = np.where(near, weight, np.sin(weight * theta) / np.maximum(sin_theta, 1e-20))
    out = weight_a * quat_a + weight_b * quat_b
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-20)


def resample_trajectory(trajectory: Trajectory, target_rate_hz: float) -> Trajectory:
    """Resample to ``target_rate_hz`` while preserving the target Nyquist limit.

    Resampling happens **before** differencing, so the finite-difference stencil
    always sees a uniform ``dt`` and the stated smoothing bandwidth is the real
    one.  Differencing first and then resampling the derivative would alias.
    """
    if abs(trajectory.rate_hz - target_rate_hz) < 1e-9:
        return trajectory

    source_times = np.arange(trajectory.num_frames, dtype=np.float64) / trajectory.rate_hz
    duration = source_times[-1]
    count = int(np.floor(duration * target_rate_hz)) + 1
    if count < 2:
        raise ValueError(
            f"sequence of {trajectory.duration_seconds:.3f} s is too short to resample "
            f"to {target_rate_hz} Hz"
        )
    target_times = np.arange(count, dtype=np.float64) / target_rate_hz

    index = np.clip(np.searchsorted(source_times, target_times, side="right") - 1, 0, len(source_times) - 2)
    span = source_times[index + 1] - source_times[index]
    weight = ((target_times - source_times[index]) / span)[:, None]

    if trajectory.rate_hz > target_rate_hz:
        # Interpolation alone aliases high-frequency mocap solver jitter into the IMU band. Filter
        # position and rotation-matrix coordinates before decimation, then project the latter back
        # onto SO(3). This is the same physical-time low-pass policy used by real-source converters.
        from scipy.signal import resample_poly

        ratio = Fraction(target_rate_hz / trajectory.rate_hz).limit_denominator(1000)
        positions = resample_poly(
            trajectory.positions, ratio.numerator, ratio.denominator, axis=0
        )[:count]
        orientations = orthonormalize(resample_poly(
            trajectory.orientations, ratio.numerator, ratio.denominator, axis=0
        )[:count])
    else:
        positions = (
            trajectory.positions[index] * (1.0 - weight[..., None])
            + trajectory.positions[index + 1] * weight[..., None]
        )
        quats = matrix_to_quaternion(trajectory.orientations)      # (T, P, 4)
        orientations = rotation_exp(
            quaternion_to_rotvec(_slerp(quats[index], quats[index + 1], weight[..., None]))
        )
    return Trajectory(positions, orientations, float(target_rate_hz), trajectory.placements)


# --------------------------------------------------------------------------- #
# The physics
# --------------------------------------------------------------------------- #


def world_acceleration(positions: np.ndarray, rate_hz: float, smooth_n: int = 2) -> np.ndarray:
    """Symmetric second difference with stride ``smooth_n``.

    Returns ``(T - 2*smooth_n, ..., 3)`` in m/s^2: the ``smooth_n`` frames at
    each end have no complete stencil and are dropped rather than padded.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[0] <= 2 * smooth_n:
        raise ValueError(
            f"need more than {2 * smooth_n} frames for a stride-{smooth_n} stencil; "
            f"got {positions.shape[0]}"
        )
    step = smooth_n / rate_hz
    return (
        positions[: -2 * smooth_n] - 2.0 * positions[smooth_n:-smooth_n] + positions[2 * smooth_n :]
    ) / (step * step)


def specific_force(world_accel: np.ndarray, up_axis: str = "z") -> np.ndarray:
    """Add gravity: ``f = a + G * up``.  Input and output m/s^2, world frame."""
    up = UP_VECTORS[up_axis]
    return np.asarray(world_accel, dtype=np.float64) + GRAVITY_M_S2 * up


def to_sensor_frame(world_vectors: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors into the sensor frame: ``R^T v``."""
    world_vectors = np.asarray(world_vectors, dtype=np.float64)
    orientations = np.asarray(orientations, dtype=np.float64)
    return np.einsum("...ji,...j->...i", orientations, world_vectors)


def angular_velocity(
    orientations: np.ndarray,
    rate_hz: float,
    stride: int = 1,
    centred: bool = True,
) -> np.ndarray:
    """Sensor-frame angular velocity in rad/s.

    ``centred``  : ``log(R[t-s]^T R[t+s]) / (2 s dt)``, zero phase, drops ``s``
                   frames at each end.
    ``not centred``: ``log(R[t]^T R[t+s]) / (s dt)``, the literal forward
                   difference; drops ``s`` frames at the end only.  Its estimate
                   is centred on ``t + s/2``, i.e. it lags by half a stride.
    """
    orientations = np.asarray(orientations, dtype=np.float64)
    if centred:
        if orientations.shape[0] <= 2 * stride:
            raise ValueError(f"need more than {2 * stride} frames for a centred stride-{stride} gyro")
        relative = np.einsum(
            "...ji,...jk->...ik", orientations[: -2 * stride], orientations[2 * stride :]
        )
        return rotation_log(relative) * (rate_hz / (2.0 * stride))
    if orientations.shape[0] <= stride:
        raise ValueError(f"need more than {stride} frames for a stride-{stride} gyro")
    relative = np.einsum("...ji,...jk->...ik", orientations[:-stride], orientations[stride:])
    return rotation_log(relative) * (rate_hz / stride)


def _apply_realism(
    acc_g: np.ndarray,
    gyro_rad_s: np.ndarray,
    config: RealismConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    num_sites = acc_g.shape[1]
    if config.acc_bias_g:
        acc_g = acc_g + rng.normal(0.0, config.acc_bias_g, size=(1, num_sites, 3))
    if config.gyro_bias_rad_s:
        gyro_rad_s = gyro_rad_s + rng.normal(0.0, config.gyro_bias_rad_s, size=(1, num_sites, 3))
    if config.acc_noise_g:
        acc_g = acc_g + rng.normal(0.0, config.acc_noise_g, size=acc_g.shape)
    if config.gyro_noise_rad_s:
        gyro_rad_s = gyro_rad_s + rng.normal(0.0, config.gyro_noise_rad_s, size=gyro_rad_s.shape)
    return acc_g, gyro_rad_s


def _perturb_placement(trajectory: Trajectory, config: RealismConfig) -> Trajectory:
    """Displace each site by a fixed random offset and misalign its mounting."""
    if not (config.placement_jitter_m or config.orientation_jitter_rad):
        return trajectory
    rng = np.random.default_rng(config.seed + 1)
    num_sites = trajectory.positions.shape[1]
    positions = trajectory.positions
    orientations = trajectory.orientations
    if config.placement_jitter_m:
        # Offset is fixed in the SENSOR frame, so it rotates with the segment —
        # a strap that sits 2 cm off is off in body coordinates, not world ones.
        offset = rng.normal(0.0, config.placement_jitter_m, size=(1, num_sites, 3))
        positions = positions + np.einsum("...ij,...j->...i", orientations, np.broadcast_to(offset, positions.shape))
    if config.orientation_jitter_rad:
        misalign = rotation_exp(rng.normal(0.0, config.orientation_jitter_rad, size=(num_sites, 3)))
        orientations = np.einsum("tpij,pjk->tpik", orientations, misalign)
    return Trajectory(positions, orientations, trajectory.rate_hz, trajectory.placements)


def synthesize(trajectory: Trajectory, config: Optional[SynthesisConfig] = None) -> VirtualIMU:
    """Trajectory -> virtual IMU.  See the module docstring for the full recipe.

    With ``config.realism.enabled == False`` (the default) this is a pure,
    deterministic function of ``trajectory``.
    """
    config = config or SynthesisConfig()
    if config.realism.enabled:
        trajectory = _perturb_placement(trajectory, config.realism)
    trajectory = resample_trajectory(trajectory, config.target_rate_hz)

    pad = config.pad
    if trajectory.num_frames <= 2 * pad:
        raise ValueError(
            f"resampled sequence has {trajectory.num_frames} frames; need more than "
            f"{2 * pad} for smooth_n={config.smooth_n}, gyro_stride={config.gyro_stride}"
        )

    rate = trajectory.rate_hz

    # --- accelerometer -----------------------------------------------------
    accel = world_acceleration(trajectory.positions, rate, config.smooth_n)
    force = specific_force(accel, config.up_axis)
    trim = pad - config.smooth_n
    force = force[trim : force.shape[0] - trim] if trim else force
    acc_g = to_sensor_frame(force, trajectory.orientations[pad : trajectory.num_frames - pad]) / GRAVITY_M_S2

    # --- gyroscope ---------------------------------------------------------
    gyro = angular_velocity(trajectory.orientations, rate, config.gyro_stride, config.gyro_centred)
    if config.gyro_centred:
        lead = pad - config.gyro_stride
        gyro = gyro[lead : gyro.shape[0] - lead] if lead else gyro
    else:
        # Forward difference consumes `stride` frames at the tail only; align its
        # sample t with the accelerometer's by slicing the same window out.
        gyro = gyro[pad : pad + acc_g.shape[0]]

    if gyro.shape[0] != acc_g.shape[0]:  # pragma: no cover - guarded by the maths above
        raise AssertionError(f"channel length mismatch: acc {acc_g.shape[0]} vs gyro {gyro.shape[0]}")

    if config.realism.enabled:
        acc_g, gyro = _apply_realism(acc_g, gyro, config.realism)

    return VirtualIMU(
        acc_g=acc_g,
        gyro_rad_s=gyro,
        rate_hz=rate,
        placements=trajectory.placements,
    )


def smoothing_response(frequency_hz: np.ndarray | float, config: SynthesisConfig) -> np.ndarray:
    """Magnitude response of the stride-``smooth_n`` stencil vs an ideal ``-w^2``.

    ``sinc^2(w n dt / 2)``.  Exposed so the README's numbers can be checked and
    so a caller can see how much of the band it is giving up.
    """
    omega = 2.0 * np.pi * np.asarray(frequency_hz, dtype=np.float64)
    half = omega * config.smooth_n / (2.0 * config.target_rate_hz)
    return np.where(np.abs(half) < 1e-12, 1.0, (np.sin(half) / np.where(half == 0, 1.0, half)) ** 2)


__all__ = [
    "GRAVITY_M_S2",
    "TARGET_RATE_HZ",
    "UP_VECTORS",
    "Placement",
    "PLACEMENTS",
    "DEFAULT_PLACEMENTS",
    "PLACEMENT_TOKENS",
    "RealismConfig",
    "SynthesisConfig",
    "Trajectory",
    "VirtualIMU",
    "rotation_exp",
    "rotation_log",
    "matrix_to_quaternion",
    "quaternion_to_rotvec",
    "orthonormalize",
    "resample_trajectory",
    "world_acceleration",
    "specific_force",
    "to_sensor_frame",
    "angular_velocity",
    "synthesize",
    "smoothing_response",
]
