"""Synthesise virtual IMU sessions from mocap corpora into the HALO session contract.

This drives :mod:`data.pretraining.synthetic_imu.synthesis` over whatever motion
capture is present under ``downloads/`` and writes one session per
(sequence x placement).

Output (identical in shape to every other converted dataset in the repo)::

    sessions/<session_id>/data.parquet
        timestamp_sec  float64   seconds from session start, uniform 1/60 s
        acc_x/y/z      float32   units g, gravity PRESENT, sensor frame
        gyro_x/y/z     float32   units rad/s, sensor frame
        subject        string    e.g. "amass_CMU_01"
    labels.json    {session_id: ["__unlabeled__"]}
    manifest.json  provenance + units
    metadata.json  role="pretrain_scale", phase_a_only, synthetic=True

``session_id`` is ``synthimu_<subject>_<sequence>_virt_<placement>``, e.g.
``synthimu_amass_CMU_01_run01_virt_rthigh``.  The **virt_** prefix on the stream
token is load-bearing: no downstream policy, grid or eval should ever confuse a
simulated stream for a measured one, and grepping for ``virt_`` finds all of
them.

Short sequences: DROPPED, not concatenated
------------------------------------------
Anything shorter than ``--min-seconds`` (default 8 s, our JEPA source window) after edge
trimming is discarded.  Concatenating clips would be the other option and we
reject it deliberately: splicing two unrelated motions creates a position
discontinuity, and the very next stage is a second derivative, which turns that
discontinuity into a multi-hundred-g impulse.  Those impulses are the largest
values in the corpus, they are pure artefact, and a masked-reconstruction
objective would happily spend capacity on them.  Losing short clips is cheap;
teaching the model a fake transient is not.

Sim-to-real caveat
------------------
Darwish, Nicholson & Doherty (2026, arXiv:2602.11064) find large-scale mocap
pretraining gives only marginal gains on its own, and helps mainly when *mixed*
with real recordings.  This module is a **diversity source**, never the base of
the corpus; see README.md.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from data.pretraining.corpus_plan import PRETRAIN_WINDOW_SECONDS

from .synthesis import (
    DEFAULT_PLACEMENTS,
    PLACEMENTS,
    SynthesisConfig,
    TARGET_RATE_HZ,
    Trajectory,
    rotation_exp,
    synthesize,
)

DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"
UNLABELED = "__unlabeled__"
WINDOW_SECONDS = PRETRAIN_WINDOW_SECONDS
OUTPUT_COLUMNS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
SOURCES = ("amass", "motion_x", "100style", "embody3d")

_SAFE = re.compile(r"[^0-9a-zA-Z]+")


def _slug(text: str) -> str:
    return _SAFE.sub("_", str(text)).strip("_").lower()


@dataclass(frozen=True)
class SourceSequence:
    """One mocap sequence, already resolved to sensor-site kinematics."""

    source: str
    subject: str          # stable subject id, e.g. "amass_CMU_01"
    name: str             # sequence name, unique within the subject
    trajectory: Trajectory
    up_axis: str = "z"


# --------------------------------------------------------------------------- #
# BVH (100STYLE) -- needs no body model, so this is the always-available source
# --------------------------------------------------------------------------- #


@dataclass
class _BvhJoint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str]


def parse_bvh(text: str, scale_to_metres: float = 0.01) -> tuple[list[_BvhJoint], np.ndarray, float]:
    """Parse a BVH file into (joints, motion frames, frame time).

    ``scale_to_metres`` converts the file's length unit.  BVH carries no unit
    declaration; 100STYLE and most Motion-Builder exports are in centimetres,
    hence the 0.01 default.  ``convert`` re-checks the result by asserting a
    plausible standing height, so a wrong unit is caught rather than silently
    scaling every acceleration by 100.
    """
    tokens = text.split("\n")
    joints: list[_BvhJoint] = []
    stack: list[int] = []
    line_no = 0

    while line_no < len(tokens):
        parts = tokens[line_no].split()
        line_no += 1
        if not parts:
            continue
        head = parts[0].upper()
        if head in {"ROOT", "JOINT"}:
            parent = stack[-1] if stack else -1
            joints.append(_BvhJoint(parts[1], parent, np.zeros(3), []))
            stack.append(len(joints) - 1)
        elif head == "END":                      # "End Site" -- no channels, skip it
            stack.append(-2)
        elif head == "OFFSET":
            if stack and stack[-1] >= 0:
                joints[stack[-1]].offset = np.array([float(v) for v in parts[1:4]]) * scale_to_metres
        elif head == "CHANNELS":
            joints[stack[-1]].channels = [c for c in parts[2:]]
        elif head == "}":
            stack.pop()
        elif head == "MOTION":
            break

    frame_count = None
    frame_time = None
    while line_no < len(tokens):
        stripped = tokens[line_no].strip()
        line_no += 1
        low = stripped.lower()
        if low.startswith("frames"):
            frame_count = int(stripped.split(":")[1])
        elif low.startswith("frame time"):
            frame_time = float(stripped.split(":")[1])
            break
    if frame_time is None or frame_count is None:
        raise ValueError("BVH file has no MOTION header (Frames / Frame Time)")

    rows = []
    for raw in tokens[line_no:]:
        parts = raw.split()
        if parts:
            rows.append([float(v) for v in parts])
        if len(rows) == frame_count:
            break
    motion = np.asarray(rows, dtype=np.float64)
    if motion.shape[0] < 2:
        raise ValueError("BVH file has fewer than two motion frames")
    return joints, motion, frame_time


_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def bvh_forward_kinematics(
    joints: Sequence[_BvhJoint],
    motion: np.ndarray,
    scale_to_metres: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """BVH channels -> global joint positions (T, J, 3) and rotations (T, J, 3, 3)."""
    num_frames = motion.shape[0]
    num_joints = len(joints)
    positions = np.zeros((num_frames, num_joints, 3))
    rotations = np.zeros((num_frames, num_joints, 3, 3))
    cursor = 0
    for index, joint in enumerate(joints):
        local_translation = np.tile(joint.offset, (num_frames, 1))
        local_rotation = np.tile(np.eye(3), (num_frames, 1, 1))
        for channel in joint.channels:
            values = motion[:, cursor]
            cursor += 1
            axis = _AXIS_INDEX[channel[0].upper()]
            if channel.lower().endswith("position"):
                local_translation[:, axis] = values * scale_to_metres
            else:
                # BVH angles are degrees, and channel ORDER is application order:
                # each successive rotation is applied in the frame of the ones
                # before it, i.e. right-multiplication.
                rotvec = np.zeros((num_frames, 3))
                rotvec[:, axis] = np.deg2rad(values)
                local_rotation = local_rotation @ rotation_exp(rotvec)
        if joint.parent < 0:
            rotations[:, index] = local_rotation
            positions[:, index] = local_translation
        else:
            parent_rotation = rotations[:, joint.parent]
            rotations[:, index] = parent_rotation @ local_rotation
            positions[:, index] = positions[:, joint.parent] + np.einsum(
                "tij,tj->ti", parent_rotation, local_translation
            )
    if cursor != motion.shape[1]:
        raise ValueError(
            f"BVH channel count {cursor} does not match motion width {motion.shape[1]}"
        )
    return positions, rotations


def trajectory_from_bvh(
    joints: Sequence[_BvhJoint],
    positions: np.ndarray,
    rotations: np.ndarray,
    rate_hz: float,
    placements: Sequence[str] = DEFAULT_PLACEMENTS,
) -> Trajectory:
    """Select the placement joints from a BVH skeleton by name.

    BVH gives us joint frames, not a body surface: a BVH-sourced sensor sits at
    the joint centre, whereas an AMASS-sourced one sits on the skin.  The
    difference is a few centimetres of lever arm and it is a real, documented
    limitation of the BVH path (README.md, "Placement provenance").
    """
    by_name = {joint.name.lower(): index for index, joint in enumerate(joints)}
    columns, orientations, resolved = [], [], []
    for name in placements:
        spec = PLACEMENTS[name]
        index = next((by_name[c.lower()] for c in spec.bvh_joint_candidates if c.lower() in by_name), None)
        if index is None:
            continue
        columns.append(positions[:, index])
        orientations.append(rotations[:, index])
        resolved.append(name)
    if not resolved:
        raise ValueError(
            "no requested placement matched this skeleton; joints are "
            f"{sorted(by_name)[:20]}"
        )
    return Trajectory(np.stack(columns, 1), np.stack(orientations, 1), float(rate_hz), tuple(resolved))


def iter_100style(
    root: Path,
    placements: Sequence[str],
    limit: Optional[int] = None,
    scale_to_metres: float = 0.01,
) -> Iterator[SourceSequence]:
    """100STYLE BVH (Zenodo 8127870, CC BY 4.0). Layout: ``<Style>/<Style>_<gait>.bvh``."""
    # The official zip also contains an equally large ``__MACOSX`` tree whose
    # ``._*.bvh`` files are AppleDouble metadata, not motion captures.
    files = sorted(
        path for path in root.rglob("*.bvh")
        if "__MACOSX" not in path.parts and not any(part.startswith(".") for part in path.parts)
    )
    for count, path in enumerate(files):
        if limit is not None and count >= limit:
            return
        joints, motion, frame_time = parse_bvh(path.read_text(errors="replace"), scale_to_metres)
        positions, rotations = bvh_forward_kinematics(joints, motion, scale_to_metres)
        style = path.parent.name if path.parent != root else path.stem.split("_")[0]
        yield SourceSequence(
            source="100style",
            # 100STYLE varies gait style for one actor; style is not a person identity.
            # Keeping it as the subject would make subject-balanced sampling claim diversity
            # that this source does not provide.
            subject="100style_actor_01",
            name=_slug(path.stem),
            trajectory=trajectory_from_bvh(joints, positions, rotations, 1.0 / frame_time, placements),
            up_axis="y",     # BVH from MotionBuilder / 100STYLE is y-up
        )


# --------------------------------------------------------------------------- #
# AMASS / Motion-X++ (SMPL-family; need the gated body model)
# --------------------------------------------------------------------------- #


def _amass_rate(payload) -> float:
    for key in ("mocap_framerate", "mocap_frame_rate", "frame_rate", "fps"):
        if key in payload:
            return float(np.asarray(payload[key]).ravel()[0])
    raise KeyError("AMASS npz carries no frame-rate field")


def iter_amass(
    root: Path,
    placements: Sequence[str],
    limit: Optional[int] = None,
    model_type: str = "smplh",
) -> Iterator[SourceSequence]:
    """AMASS ``*_poses.npz``. Layout: ``downloads/amass/<SubDataset>/<Subject>/<seq>.npz``.

    AMASS is z-up and in metres, and its ``poses`` are SMPL-H axis-angle
    (156 = 52 joints x 3); only the first 22 body joints matter for us.
    """
    from .smpl_adapter import load_body_model, trajectory_from_smpl

    files = [p for p in sorted(root.rglob("*.npz")) if p.name != "shape.npz"]
    cache: dict[str, object] = {}
    emitted = 0
    for path in files:
        if limit is not None and emitted >= limit:
            return
        with np.load(path, allow_pickle=True) as payload:
            keys = set(payload.files)
            if not {"poses", "trans", "betas"} <= keys:
                continue
            poses = np.asarray(payload["poses"], dtype=np.float64)
            trans = np.asarray(payload["trans"], dtype=np.float64)
            betas = np.asarray(payload["betas"], dtype=np.float64)
            gender = str(np.asarray(payload["gender"]).ravel()[0]) if "gender" in keys else "neutral"
            rate = _amass_rate(payload)
        gender = gender.replace("b'", "").strip("'\" ").lower()
        gender = gender if gender in {"male", "female", "neutral"} else "neutral"
        if gender not in cache:
            cache[gender] = load_body_model(model_type, gender)
        relative = path.relative_to(root)
        sub_dataset = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        subject = relative.parts[1] if len(relative.parts) > 2 else path.stem
        yield SourceSequence(
            source="amass",
            subject=f"amass_{_slug(sub_dataset)}_{_slug(subject)}",
            name=_slug(path.stem.replace("_poses", "")),
            trajectory=trajectory_from_smpl(cache[gender], poses, trans, betas, rate, placements),
            up_axis="z",
        )
        emitted += 1


def iter_motion_x(
    root: Path,
    placements: Sequence[str],
    limit: Optional[int] = None,
) -> Iterator[SourceSequence]:
    """Motion-X++ SMPL-X motion files (``*.npy``, 30 fps, 322-D per frame).

    Motion-X++ stores SMPL-X parameters; the first 66 entries of each frame's
    vector are the root orientation plus 21 body-joint axis-angles, which is
    exactly the block the SMPL-H body model consumes.  Hands, face and
    expression are ignored -- no virtual sensor sits on them.  We skin with
    SMPL-H rather than SMPL-X, which is an approximation of a few millimetres at
    the sampled sites (the two share a body topology but not a mesh), and it is
    recorded as such in manifest.json.
    """
    from .smpl_adapter import load_body_model, trajectory_from_smpl

    files = sorted(root.rglob("*.npy"))
    body_model = load_body_model("smplh", "neutral") if files else None
    for count, path in enumerate(files):
        if limit is not None and count >= limit:
            return
        raw = np.load(path, allow_pickle=True)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] < 69:
            continue
        poses = np.zeros((raw.shape[0], 156))
        poses[:, :66] = raw[:, :66]
        trans = raw[:, 309:312] if raw.shape[1] >= 312 else np.zeros((raw.shape[0], 3))
        betas = raw[0, 312:322] if raw.shape[1] >= 322 else np.zeros(10)
        relative = path.relative_to(root)
        # Motion-X++ does not expose a stable actor identifier in this layout.  A
        # per-recording group is honest for sampling, unlike treating the action/category
        # directory as a subject.
        subject = f"recording_{_slug(str(relative.with_suffix('')))}"
        yield SourceSequence(
            source="motion_x",
            subject=f"motionx_{_slug(subject)}",
            name=_slug(path.stem),
            trajectory=trajectory_from_smpl(body_model, poses, trans, betas, 30.0, placements),
            up_axis="z",
        )


def iter_embody3d(root: Path, placements: Sequence[str], limit: Optional[int] = None):
    """Embody 3D (Meta Codec Avatars, arXiv:2510.16258) -- NOT WIRED, refuses to run.

    Confirmed 2026-09-09: the dataset *is* released, with official tooling at
    github.com/facebookresearch/embody-3d (code under CC BY-NC 4.0) and access
    gated behind a release form on meta.com; approval returns 21 per-user signed
    download links consumed by that repo's ``src/download.py``.  The repo's
    README states the **dataset** is under Meta's "XRCIA" licence.

    NOT confirmed: the XRCIA licence text.  Without having read the terms we
    cannot say this corpus may be ingested, and with no data on disk we cannot
    know its on-disk schema either.  So this is a documented placeholder that
    refuses, rather than a loader guessing at a layout under unread terms.
    """
    raise NotImplementedError(
        "embody3d is a documented placeholder, not a loader: access is gated behind a Meta "
        "release form and the dataset's XRCIA licence text has not been verified. "
        "See README.md ('Embody 3D') and `fetch.py --list` before wiring this."
    )


SOURCE_LOADERS = {
    "amass": iter_amass,
    "motion_x": iter_motion_x,
    "100style": iter_100style,
    "embody3d": iter_embody3d,
}

SOURCE_DIRS = {
    "amass": "amass",
    "motion_x": "motion_x",
    "100style": "100style",
    "embody3d": "embody3d",
}


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #


def session_id(sequence: SourceSequence, placement: str) -> str:
    return f"synthimu_{sequence.subject}_{sequence.name}_{PLACEMENTS[placement].token}"


def sequence_frames(
    sequence: SourceSequence,
    config: Optional[SynthesisConfig] = None,
) -> dict[str, pd.DataFrame]:
    """Synthesise one sequence into ``{session_id: frame}`` (one frame per placement)."""
    config = replace(config or SynthesisConfig(), up_axis=sequence.up_axis)
    imu = synthesize(sequence.trajectory, config)
    timestamps = imu.timestamps_sec
    frames = {}
    for column, placement in enumerate(imu.placements):
        frame = pd.DataFrame(
            {
                "timestamp_sec": timestamps,
                "acc_x": imu.acc_g[:, column, 0].astype(np.float32),
                "acc_y": imu.acc_g[:, column, 1].astype(np.float32),
                "acc_z": imu.acc_g[:, column, 2].astype(np.float32),
                "gyro_x": imu.gyro_rad_s[:, column, 0].astype(np.float32),
                "gyro_y": imu.gyro_rad_s[:, column, 1].astype(np.float32),
                "gyro_z": imu.gyro_rad_s[:, column, 2].astype(np.float32),
            }
        )
        frame["subject"] = sequence.subject
        frames[session_id(sequence, placement)] = frame
    return frames


def emit_sessions(
    sequences: Iterable[SourceSequence],
    output_dir: Path = DS_DIR,
    config: Optional[SynthesisConfig] = None,
    min_seconds: float = WINDOW_SECONDS,
    clean: bool = True,
) -> dict:
    """Write sessions + labels.json + manifest.json + metadata.json.  Returns stats."""
    config = config or SynthesisConfig()
    output_dir = Path(output_dir)
    sessions_dir = output_dir / "sessions"
    if clean and sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = {}
    subjects: set[str] = set()
    sources: set[str] = set()
    placements_seen: set[str] = set()
    stats = {"sequences": 0, "sessions": 0, "dropped_short": 0, "failed": 0, "seconds": 0.0}
    min_samples = int(round(min_seconds * config.target_rate_hz))

    for sequence in sequences:
        stats["sequences"] += 1
        try:
            frames = sequence_frames(sequence, config)
        except ValueError as error:      # too short to difference, or malformed
            stats["dropped_short"] += 1
            print(f"[synthetic_imu] skip {sequence.source}/{sequence.name}: {error}")
            continue
        except Exception as error:       # noqa: BLE001 - one bad file must not kill the run
            stats["failed"] += 1
            print(f"[synthetic_imu] FAILED {sequence.source}/{sequence.name}: {error}")
            continue
        for sid, frame in frames.items():
            if len(frame) < min_samples:
                stats["dropped_short"] += 1
                continue
            destination = sessions_dir / sid
            destination.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(destination / "data.parquet", index=False)
            labels[sid] = [UNLABELED]
            subjects.add(sequence.subject)
            sources.add(sequence.source)
            placements_seen.add(sid.rsplit("_virt_", 1)[-1])
            stats["sessions"] += 1
            stats["seconds"] += len(frame) / config.target_rate_hz

    if not labels:
        return stats

    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "Synthetic virtual IMU from motion capture",
                "source": sorted(sources),
                "num_subjects": len(subjects),
                "sampling_rate_hz": config.target_rate_hz,
                "channels": list(OUTPUT_COLUMNS),
                "unit": "acc: g; gyro: rad/s",
                "gravity_state": "present",
                "phase_a_only": True,
                "synthesis": {
                    "method": "TransPose/DIP virtual IMU: second central difference of sensor-site "
                              "position + gravity, rotated into the segment frame; angular velocity "
                              "from the rotation-matrix log of consecutive segment orientations",
                    "smooth_n": config.smooth_n,
                    "gyro_stride": config.gyro_stride,
                    "gyro_centred": config.gyro_centred,
                    "realism_enabled": config.realism.enabled,
                    "edge_handling": "trimmed (never zero-padded)",
                    "min_seconds": min_seconds,
                    "short_sequence_policy": "dropped (never concatenated)",
                    "smplx_approximated_by_smplh": "motion_x" in sources,
                },
                "note": "SIMULATED, not measured. Sim-to-real gap: mocap-only pretraining gives "
                        "marginal gains alone (Darwish, Nicholson & Doherty 2026, arXiv:2602.11064); "
                        "use as a diversity source mixed with real data, never as the corpus base. "
                        "No activity labels; reserved marker __unlabeled__.",
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic_imu",
                "display_name": "Synthetic virtual IMU (mocap-derived)",
                "sampling_rate_hz": float(config.target_rate_hz),
                "pre_windowed": False,
                "streaming_grid": True,
                "role": "pretrain_scale",
                "phase_a_only": True,
                "synthetic": True,
                "activities": [],
                "num_subjects": None,
                # Label-free scale sources store grids at half precision; see
                # data/scripts/build_grids._store_dtype and docs/data/PRETRAINING_CORPUS.md.
                "grid_dtype": "float16",
                "channels": list(OUTPUT_COLUMNS),
                "core_channels": {name: name for name in OUTPUT_COLUMNS},
                "placement": sorted(placements_seen),
                "note": "Virtual IMU synthesised from motion capture (AMASS / Motion-X++ / 100STYLE). "
                        "Stream tokens are prefixed 'virt_' so simulated streams are never mistaken "
                        "for measured ones. Phase-A only; unlabeled.",
            },
            indent=2,
        )
        + "\n"
    )
    return stats


def iter_sources(
    sources: Sequence[str],
    raw_dir: Path,
    placements: Sequence[str],
    sequences_per_source: Optional[int],
) -> Iterator[SourceSequence]:
    for source in sources:
        root = raw_dir / SOURCE_DIRS[source]
        if not root.exists():
            print(
                f"[synthetic_imu] {source}: nothing under {root}; run "
                f"`python -m data.pretraining.synthetic_imu.fetch {source}`"
            )
            continue
        yield from SOURCE_LOADERS[source](root, placements, sequences_per_source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=["100style"], choices=SOURCES)
    parser.add_argument("--sequences", type=int, default=None,
                        help="max sequences per source (default: all present)")
    parser.add_argument("--placements", nargs="+", default=list(DEFAULT_PLACEMENTS),
                        choices=list(PLACEMENTS))
    parser.add_argument("--raw-dir", type=Path, default=DOWNLOADS)
    parser.add_argument("--output-dir", type=Path, default=DS_DIR)
    parser.add_argument("--min-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--smooth-n", type=int, default=SynthesisConfig.smooth_n)
    parser.add_argument("--target-rate-hz", type=float, default=TARGET_RATE_HZ)
    parser.add_argument("--realism", action="store_true",
                        help="enable sensor noise/bias/placement jitter (OFF by default)")
    args = parser.parse_args()

    from .synthesis import RealismConfig

    config = SynthesisConfig(
        target_rate_hz=args.target_rate_hz,
        smooth_n=args.smooth_n,
        realism=RealismConfig(enabled=args.realism),
    )
    stats = emit_sessions(
        iter_sources(args.sources, args.raw_dir, args.placements, args.sequences),
        output_dir=args.output_dir,
        config=config,
        min_seconds=args.min_seconds,
    )
    print(f"[synthetic_imu] {stats}")
    if not stats["sessions"]:
        raise SystemExit("no synthetic sessions were produced")
    print(f"[synthetic_imu] {stats['seconds'] / 3600:.2f} h across {stats['sessions']} sessions")


if __name__ == "__main__":
    main()
