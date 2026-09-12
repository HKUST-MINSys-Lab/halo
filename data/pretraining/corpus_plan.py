"""The declarative plan for the label-free pretraining corpus.

One table, imported by the orchestrator, the tests, and the docs generator, so a disk
estimate quoted in prose can never drift from the one the build actually enforces.

Sizing arithmetic, stated once
------------------------------
A stream-hour on disk is ``3600 * rate_hz * channels * bytes_per_sample``. Grids for these
sources are ``float16`` (see ``data.scripts.build_grids._store_dtype``), so
``bytes_per_sample`` is 2. The common grid always has six slots, so an 80 Hz accel-only source
uses 3.46 GB per 1,000 stream-hours; at
240 Hz with a gyro it is 10.4 GB.

"Stream-hours" is not wall-clock hours. A sequence recorded by eight body sensors
contributes eight stream-hours per wall-clock hour, because each placement is a separate
token stream the encoder sees independently.

Rate choices
------------
Aria IMUs sample at 800 Hz and 1 kHz. We store 200 Hz. The frontend's highest analysis
frequency is ~14 Hz, so a 200 Hz store is already an order of magnitude above anything the
model reads, and keeping the native rate would cost 4-5x for content that is discarded in
the first layer. The native rate is preserved in the session manifest and travels with the
window as ``source_rate_hz``, so the tokenizer still knows what was actually acquired.

Xsens stays at its native 240 Hz because that is already close to the store target and
resampling would add an interpolation artefact for no saving worth having.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

#: Bytes per stored sample. float16; see ``build_grids._store_dtype`` for why not int16.
BYTES_PER_SAMPLE = 2

#: The disk budget for grid arrays. Raw source archives and temporary conversion files need
#: additional transient space. Capture-24 contributes the largest block, but its raw archive is
#: already present locally and is shared with the labelled-data archive.
DEFAULT_BUDGET_GB = 22.0

# The representation learner needs enough past context to predict all three future-horizon bins.
# Keep this separate from data.scripts.build_grids.WINDOW_SECONDS: six-second labelled grids remain
# a reproducibility contract for historical classification experiments, while the label-free JEPA
# corpus is rebuilt at the eight-second context selected for the current experiment.
PRETRAIN_WINDOW_SECONDS = 8.0


@dataclass(frozen=True)
class SourcePlan:
    """One label-free source's intended contribution to the corpus."""

    dataset: str
    #: Human-readable description of the slice we take, not the whole dataset.
    take: str
    #: Independent token streams per wall-clock hour of recording (placements x devices).
    streams: int
    #: Wall-clock hours of recording we intend to convert.
    wall_hours: float
    rate_hz: float
    channels: int
    #: Wave 1 sources are the corpus base; wave 2 is mixed in as a diversity source.
    wave: int
    #: Fetch mechanism: "auto" (scriptable), "gated" (licence/registration first).
    access: str
    #: Physical width written by the common grid contract. Accel-only sources still use
    #: six slots because absent gyro channels are zero-padded and masked.
    stored_channels: int | None = None
    note: str = ""
    #: Set when the source is understood but its access terms are not yet confirmed.
    candidate: bool = False

    @property
    def stream_hours(self) -> float:
        return self.streams * self.wall_hours

    @property
    def gigabytes(self) -> float:
        channels = self.stored_channels if self.stored_channels is not None else self.channels
        samples = self.stream_hours * 3600.0 * self.rate_hz * channels
        return samples * BYTES_PER_SAMPLE / 1e9


#: The corpus of record, sized to the grid budget above. Editing this table is a protocol change: it
#: alters what the encoder is pretrained on, so it belongs in a commit of its own with the
#: rationale, not as a side effect of a code change.
CORPUS_PLAN: Tuple[SourcePlan, ...] = (
    SourcePlan(
        dataset="capture24_pretrain",
        take="all 151 participants; continuous raw wrist recordings, labels ignored",
        streams=1,
        wall_hours=3_883.0,
        rate_hz=100.0,
        channels=3,
        stored_channels=6,
        wave=1,
        access="auto",
        note="The full public Capture-24 release is already local: 151 people and 3,883 raw "
             "hours of dominant-wrist Axivity acceleration. The label-free adapter reads only "
             "timestamps and accelerometer axes from the original participant files, so activity "
             "annotations neither select samples nor define session boundaries.",
    ),
    SourcePlan(
        dataset="nymeria_xsens",
        take="40 deterministic sequences, 11 of 17 body placements",
        streams=11,
        wall_hours=300.0 * 40.0 / 1_064.0,
        rate_hz=240.0,
        channels=6,
        wave=1,
        access="gated",
        note="A roughly 71 GB raw subset selected deterministically from the 1,064 sequences "
             "that expose Xsens MVNX. It provides simultaneous head, torso, bilateral arm and "
             "leg motion; hands, feet, and non-target trackers are excluded.",
    ),
    SourcePlan(
        dataset="extrasensory_pretrain",
        take="all 60 participants; wrist watch and valid hand/pocket phone recordings",
        streams=3,
        # The three placement streams are not simultaneous. This is their measured combined
        # full-eight-second yield (305,279 windows = 678.4 stream-hours) expressed as the
        # plan's conventional three-stream equivalent.
        wall_hours=678.4 / 3.0,
        rate_hz=50.0,
        channels=3,
        stored_channels=6,
        wave=1,
        access="auto",
        note="Measured production conversion yields about 678 stream-hours of real free-living "
             "phone/watch snippets. "
             "The three placements are not assumed simultaneous; activity annotations are "
             "ignored and each raw acquisition remains a separate session.",
    ),
)

# Understood adapters deliberately excluded from the first corpus of record. The first JEPA
# comparison should establish whether Nymeria Xsens plus inexpensive acquisition diversity is
# sufficient before adding a second gated device family or an AWS-backed source.
DEFERRED_SOURCES: Tuple[SourcePlan, ...] = (
    SourcePlan(
        dataset="nhanes",
        take="128 participants x 12 h, motion-aware hour selection",
        streams=1,
        wall_hours=1_536.0,
        rate_hz=80.0,
        channels=3,
        stored_channels=6,
        wave=1,
        access="auto",
        note="Adapter retained for an explicit non-dominant-wrist ablation. The official CDC "
             "host was measured at roughly 0.6 MB/s aggregate for bounded parallel range "
             "requests, so the 128-participant production slice is deferred in favour of the "
             "already-local Capture-24 corpus. Re-examined 2026-09-11: it occupies the SAME "
             "role as Capture-24 (one free-living wrist accelerometer stream) while the "
             "materialized pilot is 95 stream-hours from 8 participants against Capture-24's "
             "3,883 hours from 151. A second wrist source is not a fourth role.",
    ),
    SourcePlan(
        dataset="synthetic_imu",
        take="AMASS + Motion-X++ (+ 100STYLE), 8 virtual placements",
        streams=8,
        wall_hours=225.0,
        rate_hz=60.0,
        channels=6,
        wave=2,
        access="gated",
        note="Ablation only: simulated motion can test scale, but the main corpus uses real IMU "
             "so sim-to-real artefacts cannot explain its result.",
    ),
    SourcePlan(
        dataset="nymeria_aria",
        take="head + both wrists, one IMU per device, all sequences",
        streams=3,
        wall_hours=300.0,
        rate_hz=200.0,
        channels=6,
        wave=1,
        access="gated",
        note="Deferred: the available release manifest does not expose the compact motion-only "
             "Aria assets expected by the current converter.",
    ),
    SourcePlan(
        dataset="ego_exo4d",
        take="head Aria IMU from all takes",
        streams=1,
        wall_hours=221.26,
        rate_hz=200.0,
        channels=6,
        wave=1,
        access="gated",
        note="Deferred: avoid AWS and large VRS acquisition until the selected real-IMU corpus is "
             "measured as insufficient.",
    ),
)

#: Sources that are understood and wired but whose access terms are unconfirmed. They are
#: excluded from the budget until someone confirms the licence.
CANDIDATE_SOURCES: Tuple[SourcePlan, ...] = (
    SourcePlan(
        dataset="embody3d",
        take="500 h, 439 participants, tracked 3D motion -> virtual IMU",
        streams=8,
        wall_hours=500.0,
        rate_hz=60.0,
        channels=6,
        wave=2,
        access="gated",
        candidate=True,
        note="Ten times AMASS. arXiv 2510.16258, toolbox at github.com/facebookresearch/"
             "embody-3d. The toolbox is CC BY-NC 4.0 but the DATA carries a separate Meta "
             "XRCIA licence whose terms could not be retrieved on 2026-09-09; confirm them "
             "before adding it to CORPUS_PLAN.",
    ),
)


def total_gigabytes(plan: Tuple[SourcePlan, ...] = CORPUS_PLAN) -> float:
    return sum(source.gigabytes for source in plan)


def total_stream_hours(plan: Tuple[SourcePlan, ...] = CORPUS_PLAN) -> float:
    return sum(source.stream_hours for source in plan)


def format_plan(plan: Tuple[SourcePlan, ...] = CORPUS_PLAN) -> str:
    """Render the plan as the table that appears in the docs."""
    name_width = max(16, *(len(source.dataset) for source in plan))
    header = f"{'source':<{name_width}} {'wave':>4} {'streams':>7} {'stream-h':>10} {'rate':>6} {'GB':>7}  take"
    rows = [header, "-" * len(header)]
    for source in plan:
        rows.append(
            f"{source.dataset:<{name_width}} {source.wave:>4} {source.streams:>7} "
            f"{source.stream_hours:>10,.0f} {source.rate_hz:>5.0f}Hz {source.gigabytes:>7.1f}  "
            f"{source.take}"
        )
    rows.append("-" * len(header))
    rows.append(
        f"{'TOTAL':<{name_width}} {'':>4} {'':>7} {total_stream_hours(plan):>10,.0f} {'':>6} "
        f"{total_gigabytes(plan):>7.1f}"
    )
    return "\n".join(rows)
