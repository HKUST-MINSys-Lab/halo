"""Deterministic length planning for variable-duration motion episodes."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


@dataclass(frozen=True)
class DurationPlan:
    """The recording-duration policy for one logical optimizer update."""

    mode: str
    cap_seconds: float | None
    source_duration_seconds: float | None

    def __post_init__(self) -> None:
        if self.mode not in {"complete", "cropped"}:
            raise ValueError("duration mode must be complete or cropped")
        if self.mode == "complete" and self.cap_seconds is not None:
            raise ValueError("a complete plan cannot have a duration cap")
        if self.mode == "cropped" and (
            self.cap_seconds is None or self.cap_seconds <= 0
        ):
            raise ValueError("a cropped plan needs a positive duration cap")


def central_duration_cap(
    durations_seconds: Sequence[float], rng: random.Random
) -> float:
    """Draw an observed duration from the central ranks of a logical batch.

    Selecting an observed duration avoids a corpus-specific magic value. The
    shortest and longest examples are excluded whenever the batch has at least
    four distinct positions, so one outlier does not determine the allocation.
    """

    durations = sorted(float(value) for value in durations_seconds)
    if not durations or any(value <= 0 for value in durations):
        raise ValueError("durations must be non-empty and positive")
    if len(durations) < 4:
        return durations[len(durations) // 2]
    lower = len(durations) // 4
    upper = len(durations) - lower
    return durations[rng.randrange(lower, upper)]


def plan_duration(
    durations_seconds: Sequence[float],
    *,
    complete_probability: float,
    rng: random.Random,
) -> DurationPlan:
    """Choose complete coverage or one shared crop cap for a logical batch."""

    if not 0.0 <= complete_probability <= 1.0:
        raise ValueError("complete probability must be in [0, 1]")
    if rng.random() < complete_probability:
        return DurationPlan("complete", None, None)
    cap = central_duration_cap(durations_seconds, rng)
    return DurationPlan("cropped", cap, cap)


def padding_fraction(lengths: Sequence[int]) -> float:
    """Fraction of allocated sequence positions that contain only batch padding."""

    values = [int(value) for value in lengths]
    if not values or any(value <= 0 for value in values):
        raise ValueError("lengths must be non-empty positive integers")
    allocated = len(values) * max(values)
    return 1.0 - sum(values) / allocated


def nearest_length_indices(
    durations_seconds: Sequence[float],
    *,
    count: int,
    target_seconds: float,
    rng: random.Random,
) -> list[int]:
    """Pick a reproducibly shuffled set of durations closest to one target."""

    durations = [float(value) for value in durations_seconds]
    if count < 1 or count > len(durations) or target_seconds <= 0:
        raise ValueError("invalid length-matched selection")
    # Shuffle before stable sorting so equal durations do not acquire source-order bias.
    indices = list(range(len(durations)))
    rng.shuffle(indices)
    indices.sort(key=lambda index: abs(durations[index] - target_seconds))
    return indices[:count]
