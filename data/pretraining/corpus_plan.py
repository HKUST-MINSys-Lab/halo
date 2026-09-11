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
#: additional transient space.
DEFAULT_BUDGET_GB = 180.0

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


#: The corpus of record, sized to ~180 GB of grid arrays. Editing this table is a protocol change: it
#: alters what the encoder is pretrained on, so it belongs in a commit of its own with the
#: rationale, not as a side effect of a code change.
CORPUS_PLAN: Tuple[SourcePlan, ...] = (
    SourcePlan(
        dataset="nhanes",
        take="3,000 participants x 12 h, motion-aware hour selection",
        streams=1,
        wall_hours=36_000.0,
        rate_hz=80.0,
        channels=3,
        stored_channels=6,
        wave=1,
        access="auto",
        note="Breadth of SUBJECTS on the wrist placement every evaluation set uses. The "
             "motion-aware hour budget keeps a one-third still minority so sleep and "
             "sedentary posture stay represented.",
    ),
    SourcePlan(
        dataset="nymeria_xsens",
        take="all 1,100 sequences, 8 of 17 body placements",
        streams=8,
        wall_hours=300.0,
        rate_hz=240.0,
        channels=6,
        wave=1,
        access="gated",
        note="The only source with real IMU on head, torso, both arms and a leg "
             "simultaneously. The 9 dropped Xsens sensors are left/right mirrors of "
             "placements already kept.",
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
        note="Real glasses and wristband IMUs, hardware-synchronised with the Xsens suit "
             "above, so the same motion is observed from three device classes at once.",
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
        note="The head placement at 740+ subjects. The release has 1,286 camera-hours but "
             "221 ego-camera hours, which is the relevant one-IMU stream budget. The ~1 TB "
             "image-free VRS download is the real cost and is streamed then deleted.",
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
        note="Diversity of motion vocabulary, never the base of the corpus: published "
             "evidence (arXiv 2602.11064) is that mocap pretraining alone gives marginal "
             "gains from the sim-to-real gap and helps mainly when mixed with real data.",
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
    header = f"{'source':<16} {'wave':>4} {'streams':>7} {'stream-h':>10} {'rate':>6} {'GB':>7}  take"
    rows = [header, "-" * len(header)]
    for source in plan:
        rows.append(
            f"{source.dataset:<16} {source.wave:>4} {source.streams:>7} "
            f"{source.stream_hours:>10,.0f} {source.rate_hz:>5.0f}H {source.gigabytes:>7.1f}  "
            f"{source.take}"
        )
    rows.append("-" * len(header))
    rows.append(
        f"{'TOTAL':<16} {'':>4} {'':>7} {total_stream_hours(plan):>10,.0f} {'':>6} "
        f"{total_gigabytes(plan):>7.1f}"
    )
    return "\n".join(rows)
