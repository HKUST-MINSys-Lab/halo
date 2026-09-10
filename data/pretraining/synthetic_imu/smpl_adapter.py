"""Thin SMPL / SMPL-H forward pass: body poses -> the vertex + joint trajectories
that :mod:`data.pretraining.synthetic_imu.synthesis` consumes.

This is deliberately the *only* part of the module that needs the gated body
model files.  Keeping it separate is what makes the virtual-IMU physics fully
testable offline: the physics takes a
:class:`~data.pretraining.synthetic_imu.synthesis.Trajectory` as input, and this
file is one of several ways to produce one (the BVH reader in ``convert.py`` is
another, and it needs no body model at all).

The body models are registration-gated and are **not** redistributable:

  * SMPL   -- https://smpl.is.tue.mpg.de   (SMPL_{FEMALE,MALE,NEUTRAL}.pkl)
  * SMPL-H -- https://mano.is.tue.mpg.de   (the "Extended SMPL+H model" archive)

Run ``python -m data.pretraining.synthetic_imu.fetch --list`` for the exact
files and where to drop them.  With the files absent, :func:`load_body_model`
raises :class:`BodyModelMissing` and every caller degrades to a clear message
rather than a stack trace.

Implementation notes
--------------------
Standard linear blend skinning with shape and pose blend shapes, in numpy:

    v_shaped  = v_template + shapedirs . betas
    J         = J_regressor @ v_shaped
    v_posed   = v_shaped + posedirs . (R(theta)[1:] - I)
    G_k       = G_parent(k) @ [R_k | J_k - J_parent(k)]
    v_world   = sum_k w_k (G_k @ [I | -J_k]) @ [v_posed, 1]

Only the handful of vertices we actually sample are skinned, so the cost is
O(T * 8) rather than O(T * 6890).  The **joint** global rotations ``G_k[:3,:3]``
are the segment orientations the virtual sensors are strapped to, which is also
what TransPose uses (its ``ji_mask``).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .synthesis import (
    DEFAULT_PLACEMENTS,
    PLACEMENTS,
    Trajectory,
    rotation_exp,
)

MODELS_DIR = Path(__file__).resolve().parent / "downloads" / "body_models"

#: Candidate filenames per (model type, gender), newest naming first.
_MODEL_FILENAMES = {
    ("smplh", "neutral"): ("SMPLH_NEUTRAL.npz", "SMPLH_NEUTRAL.pkl", "neutral/model.npz"),
    ("smplh", "male"): ("SMPLH_MALE.npz", "SMPLH_MALE.pkl", "male/model.npz"),
    ("smplh", "female"): ("SMPLH_FEMALE.npz", "SMPLH_FEMALE.pkl", "female/model.npz"),
    ("smpl", "neutral"): ("SMPL_NEUTRAL.pkl", "SMPL_NEUTRAL.npz", "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"),
    ("smpl", "male"): ("SMPL_MALE.pkl", "SMPL_MALE.npz", "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"),
    ("smpl", "female"): ("SMPL_FEMALE.pkl", "SMPL_FEMALE.npz", "basicModel_f_lbs_10_207_0_v1.0.0.pkl"),
}


class BodyModelMissing(FileNotFoundError):
    """The gated SMPL / SMPL-H files are not present on this machine."""


@dataclass(frozen=True)
class BodyModel:
    """The subset of an SMPL-family model that skinning needs."""

    v_template: np.ndarray      # (V, 3)
    shapedirs: np.ndarray       # (V, 3, B)
    posedirs: np.ndarray        # (V, 3, 9*(K-1))
    j_regressor: np.ndarray     # (K, V)
    weights: np.ndarray         # (V, K)
    parents: np.ndarray         # (K,) int, parents[0] == -1
    model_type: str
    gender: str

    @property
    def num_joints(self) -> int:
        return int(self.parents.shape[0])

    @property
    def num_betas(self) -> int:
        return int(self.shapedirs.shape[-1])


def _as_dense(array) -> np.ndarray:
    if hasattr(array, "toarray"):          # scipy sparse J_regressor in the .pkl models
        array = array.toarray()
    return np.asarray(array, dtype=np.float64)


def _read_model_file(path: Path) -> dict:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as handle:
            return {key: handle[key] for key in handle.files}
    with path.open("rb") as handle:
        # The original SMPL pickles are Python 2 and carry chumpy arrays; latin1
        # is the standard incantation for reading them from Python 3.
        return pickle.load(handle, encoding="latin1")


def find_model_path(model_type: str, gender: str, models_dir: Path = MODELS_DIR) -> Optional[Path]:
    """Locate a body model file, or ``None`` if it is not on disk."""
    key = (model_type.lower(), gender.lower())
    for name in _MODEL_FILENAMES.get(key, ()):
        candidate = models_dir / name
        if candidate.exists():
            return candidate
    # Tolerate an arbitrary extraction layout: search by filename anywhere below.
    if models_dir.exists():
        wanted = {name.rsplit("/", 1)[-1].lower() for name in _MODEL_FILENAMES.get(key, ())}
        for candidate in sorted(models_dir.rglob("*")):
            if candidate.is_file() and candidate.name.lower() in wanted:
                return candidate
    return None


def load_body_model(
    model_type: str = "smplh",
    gender: str = "neutral",
    models_dir: Path = MODELS_DIR,
) -> BodyModel:
    """Load SMPL/SMPL-H parameters, or raise :class:`BodyModelMissing`."""
    path = find_model_path(model_type, gender, models_dir)
    if path is None:
        raise BodyModelMissing(
            f"no {model_type.upper()} '{gender}' body model under {models_dir}. "
            "These files are registration-gated and cannot be downloaded automatically; "
            "run `python -m data.pretraining.synthetic_imu.fetch --list` for instructions."
        )
    raw = _read_model_file(path)
    kintree = np.asarray(raw["kintree_table"])
    parents = kintree[0].astype(np.int64).copy()
    parents[0] = -1
    shapedirs = _as_dense(raw["shapedirs"])
    posedirs = _as_dense(raw["posedirs"])
    return BodyModel(
        v_template=_as_dense(raw["v_template"]),
        shapedirs=shapedirs,
        posedirs=posedirs.reshape(posedirs.shape[0], 3, -1),
        j_regressor=_as_dense(raw["J_regressor"]),
        weights=_as_dense(raw["weights"]),
        parents=parents,
        model_type=model_type.lower(),
        gender=gender.lower(),
    )


def _global_transforms(rotations: np.ndarray, joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Compose the kinematic chain.  ``rotations`` (T, K, 3, 3), ``joints`` (K, 3)."""
    num_frames, num_joints = rotations.shape[:2]
    transforms = np.zeros((num_frames, num_joints, 4, 4), dtype=np.float64)
    transforms[..., 3, 3] = 1.0
    transforms[:, 0, :3, :3] = rotations[:, 0]
    transforms[:, 0, :3, 3] = joints[0]
    for joint in range(1, num_joints):
        parent = int(parents[joint])
        local = np.zeros((num_frames, 4, 4), dtype=np.float64)
        local[..., 3, 3] = 1.0
        local[:, :3, :3] = rotations[:, joint]
        local[:, :3, 3] = joints[joint] - joints[parent]
        transforms[:, joint] = transforms[:, parent] @ local
    return transforms


def forward(
    body_model: BodyModel,
    poses: np.ndarray,
    translations: np.ndarray,
    betas: np.ndarray,
    vertex_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Skin ``vertex_indices`` only.

    poses         (T, 3*K) axis-angle, root first, SMPL convention
    translations  (T, 3) metres
    betas         (B,) shape coefficients
    returns       vertices (T, len(vertex_indices), 3), joint rotations (T, K, 3, 3)
    """
    poses = np.asarray(poses, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    num_joints = body_model.num_joints
    if poses.shape[-1] < 3 * num_joints:
        raise ValueError(
            f"pose vector has {poses.shape[-1]} entries, need at least {3 * num_joints} "
            f"for a {num_joints}-joint {body_model.model_type.upper()} model"
        )
    poses = poses[:, : 3 * num_joints].reshape(-1, num_joints, 3)
    num_frames = poses.shape[0]

    betas = np.asarray(betas, dtype=np.float64).ravel()
    num_betas = min(betas.shape[0], body_model.num_betas)
    v_shaped = body_model.v_template + body_model.shapedirs[..., :num_betas] @ betas[:num_betas]
    joints = body_model.j_regressor @ v_shaped

    rotations = rotation_exp(poses)                                   # (T, K, 3, 3)
    pose_feature = (rotations[:, 1:] - np.eye(3)).reshape(num_frames, -1)

    index = np.asarray(list(vertex_indices), dtype=np.int64)
    # Pose blend shapes and skinning weights, restricted to the sampled vertices.
    v_posed = v_shaped[index] + np.einsum(
        "vcp,tp->tvc", body_model.posedirs[index, :, : pose_feature.shape[1]], pose_feature
    )

    transforms = _global_transforms(rotations, joints, body_model.parents)
    # Remove the rest-pose joint translation so the transform maps rest -> posed.
    relative = transforms.copy()
    relative[..., :3, 3] -= np.einsum("tkij,kj->tki", transforms[..., :3, :3], joints)

    skinned = np.einsum("vk,tkij->tvij", body_model.weights[index], relative)
    homogeneous = np.concatenate([v_posed, np.ones(v_posed.shape[:2] + (1,))], axis=-1)
    vertices = np.einsum("tvij,tvj->tvi", skinned, homogeneous)[..., :3]
    return vertices + translations[:, None, :], transforms[..., :3, :3]


def trajectory_from_smpl(
    body_model: BodyModel,
    poses: np.ndarray,
    translations: np.ndarray,
    betas: np.ndarray,
    rate_hz: float,
    placements: Sequence[str] = DEFAULT_PLACEMENTS,
) -> Trajectory:
    """Build a :class:`Trajectory` for ``placements`` from an SMPL pose sequence.

    Sites with a published mesh-vertex index use it; sites without one
    (``smpl_vertex is None`` -- see ``PLACEMENTS`` provenance) fall back to the
    joint origin.  Segment orientation is always the global rotation of the
    placement's ``smpl_joint``.
    """
    specs = [PLACEMENTS[name] for name in placements]
    unknown = [s.smpl_joint for s in specs if s.smpl_joint >= body_model.num_joints]
    if unknown:
        raise ValueError(f"placement joints {unknown} exceed the model's {body_model.num_joints} joints")

    vertex_specs = [s for s in specs if s.smpl_vertex is not None]
    vertices, joint_rotations = forward(
        body_model, poses, translations, betas, [s.smpl_vertex for s in vertex_specs] or [0]
    )

    # Joint world positions, needed for the vertex-less fallbacks.
    poses_arr = np.asarray(poses, dtype=np.float64)[:, : 3 * body_model.num_joints]
    rotations = rotation_exp(poses_arr.reshape(-1, body_model.num_joints, 3))
    betas_arr = np.asarray(betas, dtype=np.float64).ravel()
    num_betas = min(betas_arr.shape[0], body_model.num_betas)
    rest_joints = body_model.j_regressor @ (
        body_model.v_template + body_model.shapedirs[..., :num_betas] @ betas_arr[:num_betas]
    )
    transforms = _global_transforms(rotations, rest_joints, body_model.parents)
    joint_positions = transforms[..., :3, 3] + np.asarray(translations, dtype=np.float64)[:, None, :]

    columns = []
    vertex_cursor = 0
    for spec in specs:
        if spec.smpl_vertex is None:
            columns.append(joint_positions[:, spec.smpl_joint])
        else:
            columns.append(vertices[:, vertex_cursor])
            vertex_cursor += 1
    positions = np.stack(columns, axis=1)
    orientations = np.stack([joint_rotations[:, s.smpl_joint] for s in specs], axis=1)
    return Trajectory(positions, orientations, float(rate_hz), tuple(placements))


__all__ = [
    "MODELS_DIR",
    "BodyModel",
    "BodyModelMissing",
    "find_model_path",
    "load_body_model",
    "forward",
    "trajectory_from_smpl",
]
