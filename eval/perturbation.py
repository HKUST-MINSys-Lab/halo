"""Controlled acquisition perturbations for the matched adaptation protocol.

WHY THIS EXISTS
---------------
The primary enrollment curve draws support from the same stream as the query, so acquisition never
differs between the two by construction. The heterogeneity-axis experiment (design doc §7) is the
controlled complement: take the frozen ``adaptation_v2`` cells, change ONE physical property of the
query or of the support recordings at test time, and read the same k-curve again. Every model
receives identical perturbed inputs, so the drop is a property of the model, not of the data.

Four axes, taken from ``training/diagnostics/baseline_heterogeneity.py`` (which applied them to a
bounded zero-shot subset, never to the manifest):

* ``rate`` — anti-aliased polyphase resampling to a lower rate; the new rate is what the model is
  told. The compatibility key ignores rate on purpose, so this axis tests that decision.
* ``channel`` — the gyroscope triad is zeroed and masked out. This CHANGES the acquisition key.
* ``orientation`` — one proper SO(3) rotation applied to accelerometer and gyroscope triads. The
  default draws ONE rotation per stream (a remounted device); ``execution`` draws one per enrolled
  execution, ``window`` one per window (the diagnostic's choice). The key is unchanged.
* ``gravity`` — the accelerometer's gravity component is removed with a 0.4 Hz low-pass subtraction
  and the stream is relabelled gravity-removed for any model that reads that metadata. CHANGES the
  acquisition key.

Which side is perturbed matters. ``query``: the user's device differs from the exemplars — the §7
figure. ``support``: the exemplars came from a different setup — the incompatible-exemplar cell the
two-arm comparison needs (Arm A filters such support out and reports the cell unsupported; Arm B
attends over it).

Perturbed views carry ``EvalStream.perturbation`` so per-stream caches never mix them with the
grid as converted. Nothing here touches the manifest; its fingerprint is over the unperturbed grid.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal as sps

from data.scripts.curate import deployment_policy
from data.scripts.curate.compatibility import (
    AcquisitionKey,
    acquisition_key,
    are_compatible,
    is_near_miss,
)
from eval.data import EvalStream
from baselines.base import UnsupportedEvaluationCell

AXES = ("rate", "channel", "orientation", "gravity")
SIDES = ("query", "support")
ROTATION_UNITS = ("stream", "execution", "window")
DEFAULT_SEED = 20260905
DEFAULT_RATE_HZ = 25.0
GRAVITY_CUTOFF_HZ = 0.4


@dataclass(frozen=True)
class Perturbation:
    """One controlled change, applied to one side of every cell."""

    axis: str
    side: str
    seed: int = DEFAULT_SEED
    rate_hz: float = DEFAULT_RATE_HZ
    rotation_unit: str = "stream"

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}, got {self.axis!r}")
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")
        if self.rotation_unit not in ROTATION_UNITS:
            raise ValueError(f"rotation_unit must be one of {ROTATION_UNITS}")
        if self.axis == "rate" and not self.rate_hz > 0:
            raise ValueError("rate_hz must be positive")

    @property
    def label(self) -> str:
        """Run identity suffix, e.g. ``query_orientation_per_stream``; also the model variant."""
        parts = [self.side, self.axis]
        if self.axis == "rate":
            parts.append(f"{self.rate_hz:g}hz")
        if self.axis == "orientation":
            parts.append(f"per_{self.rotation_unit}")
        if self.seed != DEFAULT_SEED:
            parts.append(f"seed{self.seed}")
        return "_".join(parts)

    def as_dict(self) -> dict:
        return {**dataclasses.asdict(self), "label": self.label}


# --------------------------------------------------------------------------- metadata helpers

def _spec(stream: EvalStream):
    return deployment_policy.get_stream_spec(stream.dataset, stream.stream)


def gravity_state_of(stream: EvalStream) -> str:
    """The stream's gravity state: its override when set, else the curated spec's."""
    if getattr(stream, "gravity_state", None) is not None:
        return str(stream.gravity_state)
    try:
        return str(_spec(stream).gravity_state)
    except (KeyError, ValueError):
        return "unknown"


def _triad(stream: EvalStream, prefix: str) -> list[int] | None:
    """Column indices of one sensor's x/y/z, only when all three are present and valid."""
    mask = np.asarray(stream.mask, dtype=bool)
    columns = [index for index, name in enumerate(stream.channels) if name.startswith(prefix)]
    if len(columns) != 3 or not mask[columns].all():
        return None
    return columns


def acquisition_key_for(stream: EvalStream) -> AcquisitionKey:
    """The stream's acquisition key as it is NOW: curated spec, then the view's own mask and
    gravity state on top, so a perturbed view reports the configuration it actually presents."""
    spec = _spec(stream)
    declared = tuple(spec.required) + tuple(spec.optional)
    channels = getattr(stream, "channels", None)
    if channels is None or getattr(stream, "mask", None) is None:
        # A bare (dataset, stream) reference means the grid as curated.
        return acquisition_key(
            device_profile=spec.device_profile, placement=spec.placement,
            channels=declared, gravity_state=gravity_state_of(stream),
        )
    mask = np.asarray(stream.mask, dtype=bool)
    present = [
        channel for channel in declared
        if any(name.startswith(channel) and bool(valid)
               for name, valid in zip(stream.channels, mask))
    ]
    return acquisition_key(
        device_profile=spec.device_profile,
        placement=spec.placement,
        channels=present,
        gravity_state=gravity_state_of(stream),
    )


def compatibility_relation(query: EvalStream, support: EvalStream) -> str:
    """``identical`` (Arm A admits it), ``near_miss`` (Arm B2's case) or ``incompatible``."""
    query_key = acquisition_key_for(query)
    support_key = acquisition_key_for(support)
    if are_compatible(query_key, support_key):
        return "identical"
    if is_near_miss(query_key, support_key):
        return "near_miss"
    return "incompatible"


# --------------------------------------------------------------------------- the axes

def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform proper rotation from a random unit quaternion (Marsaglia)."""
    q = rng.standard_normal(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _rotation_groups(stream: EvalStream, unit: str) -> np.ndarray:
    """Group id per window: which windows share one rotation."""
    n = stream.n_windows
    if unit == "stream":
        return np.zeros(n, dtype=np.int64)
    if unit == "window":
        return np.arange(n, dtype=np.int64)
    ids = stream.execution_ids if stream.execution_ids is not None else stream.block_ids
    if ids is None:
        raise ValueError(
            f"{stream.dataset}/{stream.stream}: per-execution rotation needs execution ids"
        )
    _, inverse = np.unique(np.asarray(ids).astype(str), return_inverse=True)
    return inverse.astype(np.int64)


def _shift_orientation(stream: EvalStream, p: Perturbation) -> dict:
    data = np.array(stream.windows, dtype=np.float32, copy=True)
    triads = [t for t in (_triad(stream, "acc"), _triad(stream, "gyro")) if t is not None]
    if not triads:
        raise UnsupportedEvaluationCell(f"{stream.dataset}/{stream.stream}: no complete triad to rotate")
    groups = _rotation_groups(stream, p.rotation_unit)
    rng = np.random.default_rng(p.seed)
    for group in np.unique(groups):
        rotation = _random_rotation(rng)
        rows = np.flatnonzero(groups == group)
        for columns in triads:
            data[np.ix_(rows, np.arange(data.shape[1]), columns)] = np.einsum(
                "ij,ntj->nti", rotation, data[np.ix_(rows, np.arange(data.shape[1]), columns)],
            )
    return {"windows": data}


def _shift_rate(stream: EvalStream, p: Perturbation) -> dict:
    native = float(stream.rate_hz)
    if not 0 < p.rate_hz < native:
        raise UnsupportedEvaluationCell(
            f"{stream.dataset}/{stream.stream}: rate shift needs 0 < rate_hz < native "
            f"({p.rate_hz} vs {native})"
        )
    ratio = Fraction(p.rate_hz / native).limit_denominator(50)
    data = sps.resample_poly(
        np.asarray(stream.windows, dtype=np.float32), ratio.numerator, ratio.denominator, axis=1,
    ).astype(np.float32)
    lengths = getattr(stream, "lengths", None)
    if lengths is not None:
        lengths = np.minimum(
            (np.asarray(lengths) * ratio.numerator + ratio.denominator - 1) // ratio.denominator,
            data.shape[1],
        )
    return {"windows": data, "rate_hz": native * ratio.numerator / ratio.denominator,
            "lengths": lengths}


def _shift_channel(stream: EvalStream, p: Perturbation) -> dict:
    del p
    gyro = [index for index, name in enumerate(stream.channels) if name.startswith("gyro")]
    mask = np.asarray(stream.mask, dtype=bool).copy()
    if not gyro or not mask[gyro].any():
        raise UnsupportedEvaluationCell(f"{stream.dataset}/{stream.stream}: already accelerometer-only")
    data = np.array(stream.windows, dtype=np.float32, copy=True)
    data[..., gyro] = 0.0          # exact zeros with a mask, never fabricated (START_HERE §5.5)
    mask[gyro] = False
    return {"windows": data, "mask": mask}


def _gravity_removed_texts(stream: EvalStream) -> list[str] | None:
    """Arm B's acquisition text must say what the signal now is."""
    from training.tokenizer.pretrain_data import stream_channel_descriptions

    try:
        texts = (list(stream.channel_descriptions)
                 if stream.channel_descriptions is not None
                 else stream_channel_descriptions(stream.dataset, stream.stream))
    except Exception:  # pragma: no cover - a stream without curated text keeps none
        return None
    return [text.replace("; includes gravity", "; gravity removed") for text in texts]


def _shift_gravity(stream: EvalStream, p: Perturbation) -> dict:
    del p
    if gravity_state_of(stream) != "present":
        raise UnsupportedEvaluationCell(
            f"{stream.dataset}/{stream.stream}: gravity_state is "
            f"{gravity_state_of(stream)!r}; only gravity-present sources can be perturbed"
        )
    columns = _triad(stream, "acc")
    if columns is None:
        raise UnsupportedEvaluationCell(f"{stream.dataset}/{stream.stream}: no complete accelerometer triad")
    data = np.array(stream.windows, dtype=np.float32, copy=True)
    wn = GRAVITY_CUTOFF_HZ / (float(stream.rate_hz) / 2.0)
    coefficients = sps.butter(2, wn, btype="low") if 0 < wn < 1 else None
    lengths = (np.asarray(stream.lengths) if stream.lengths is not None
               else np.full(stream.n_windows, data.shape[1]))
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        index = np.ix_(rows, np.arange(int(length)), columns)
        signal = data[index]
        low = (signal.mean(axis=1, keepdims=True) if length <= 12 or coefficients is None
               else sps.filtfilt(*coefficients, signal, axis=1))
        data[index] = signal - low
    return {
        "windows": data.astype(np.float32),
        "gravity_state": "removed",
        "channel_descriptions": _gravity_removed_texts(stream),
    }


_SHIFT = {
    "rate": _shift_rate,
    "channel": _shift_channel,
    "orientation": _shift_orientation,
    "gravity": _shift_gravity,
}


def perturb_stream(stream: EvalStream, p: Perturbation) -> EvalStream:
    """A new view of ``stream`` with one axis changed; the input is never modified."""
    if getattr(stream, "perturbation", None) is not None:
        raise ValueError(f"stream already carries perturbation {stream.perturbation!r}")
    changes = _SHIFT[p.axis](stream, p)
    lengths = changes.get("lengths", stream.lengths)
    if lengths is not None:
        valid = np.arange(changes["windows"].shape[1])[None, :] < np.asarray(lengths)[:, None]
        changes["windows"][~valid] = 0.0
    return dataclasses.replace(stream, perturbation=p.label, **changes)
