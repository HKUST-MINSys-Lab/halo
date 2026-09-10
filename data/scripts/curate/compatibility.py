"""Acquisition compatibility for support-set construction (IMWUT `imwut/compare` line).

WHY THIS FILE EXISTS
--------------------
The comparison model is handed K labelled recordings alongside each query. Which recordings are
*allowed* to be handed over is a deployment consideration, not something the model should have to
learn: a pocket-phone query is not compared against smartwatch examples. This module is that
filter, and nothing here is claimed as a research contribution.

WHY IT IS NOT `applications/motion_monitoring/data/compatibility.py`
-------------------------------------------------------------------
That module keys off a live :class:`SensorStream` and normalises ``placement`` by lowercasing and
collapsing whitespace. The corpus stores ``StreamSpec.placement`` as free-text English, so that
normalisation leaves *six distinct keys for the wrist*::

    "the left wrist" · "left wrist" · "the wrist" · "wrist" · "dominant wrist" · "the right wrist"

Under exact-string matching the compatible pool fragments into near-singletons and cross-dataset
support becomes impossible, which breaks the core arm rather than only the experiment. This module
replaces the string with an explicit, checked-in **site** taken from a closed vocabulary.

TWO RELATIONS, NOT A LADDER
---------------------------
Per the design decision of 2026-09-03 the compatibility relation is binary:

* :func:`are_compatible` — identical key. This is what the core arm's sampler enforces.
* :func:`is_near_miss` — same device family, sites in the same equivalence group, key not
  identical. Left wrist against right wrist; a left trouser pocket against a back pocket. This is
  the only "further away" relation the experiment uses.

Anything outside those two (a wrist against an ankle, a watch against a phone) is **out of scope**.
We deliberately do not define a graded degradation ladder, because we do not claim one.

LATERALITY IS PRESERVED, PROSE IS NOT
-------------------------------------
``"the left wrist"`` and ``"left wrist"`` are the same site and collapse to one key. ``left_wrist``
and ``right_wrist`` stay distinct sites that are *equivalent* — so they are a near miss, never
identical. That is exactly the distinction the experiment needs: prose variation is noise, body
laterality is a real (if small) acquisition difference.

ALWAYS KEY OFF ``placement``, NEVER ``stream_id``
-------------------------------------------------
``nfi_fared``'s stream is called ``wrist`` but its curated placement is "the dominant forearm".
Every function here reads the spec, never the identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from data.scripts.curate import deployment_policy


# --------------------------------------------------------------------------- device families

#: ``StreamSpec.device_profile`` is already a coarse form factor. It is mapped explicitly rather
#: than pattern-matched so an unrecognised profile fails loudly.
#:
#: ``watch_proxy`` stays separate from ``watch``: shoaib straps a *phone* to the wrist, which has a
#: different mass and mounting rigidity from a watch. Merging them would assert an equivalence we
#: have not measured.
#:
#: ``device`` is the corpus's catch-all for body-worn research IMUs, and it also covers earbuds and
#: smart glasses. It is coarse on its own; the placement site is what separates an ear-worn unit
#: from a shin-mounted one.
DEVICE_FAMILY: dict[str, str] = {
    "phone": "phone",
    "watch": "watch",
    "watch_proxy": "watch_proxy",
    "device": "body_imu",
    "non_deployment": "non_deployment",
}


# --------------------------------------------------------------------------- placement sites

#: Every ``StreamSpec.placement`` string in the corpus, mapped to a closed-vocabulary site.
#: An unmapped string is a hard error (:func:`placement_site`), never a silent fallback — a new
#: dataset must state where its sensor sits before it can enter a support set.
PLACEMENT_SITE: dict[str, str] = {
    # ---- wrist ------------------------------------------------------------
    "the left wrist": "left_wrist",
    "left wrist": "left_wrist",
    "the right wrist": "right_wrist",
    "right wrist": "right_wrist",
    # Handedness cannot be resolved to a physical side, so these stay side-agnostic. A source that
    # says only "the wrist" is pooled with one that says "dominant": single-wrist studies place the
    # device on the preferred wrist by convention, and treating them as different configurations
    # would strand monipar with no compatible training partner at all.
    "the wrist": "wrist_unspecified",
    "wrist": "wrist_unspecified",
    "dominant wrist": "wrist_unspecified",
    "the dominant wrist": "wrist_unspecified",
    # But an explicitly NON-dominant wrist is not the same observation — the dominant wrist does the
    # manipulation — so it is its own site rather than pooled with the above. Free to separate:
    # nhanes is an optional scale source, not part of any named training recipe.
    "the non-dominant wrist": "non_dominant_wrist",
    # upper_limb_use labels by clinical status, and affected vs unaffected is the entire contrast
    # that dataset exists to measure. Pooling them would let an unaffected-arm recording support a
    # query about the affected arm the moment cross-configuration cells are ever enabled. They are
    # wrist-EQUIVALENT (so still a near miss) but never identical.
    "the wrist of the more-affected arm": "affected_wrist",
    "the wrist of the less-affected arm": "unaffected_wrist",
    # ---- forearm ----------------------------------------------------------
    "the left forearm": "left_forearm",
    "the right forearm": "right_forearm",
    "the dominant forearm": "forearm_unspecified",
    # ---- upper arm --------------------------------------------------------
    "the left upper arm": "left_upper_arm",
    "the right upper arm": "right_upper_arm",
    # ---- hand -------------------------------------------------------------
    "the hand": "hand",
    # ---- waist / hip / belt ----------------------------------------------
    "waist": "waist",
    "front-right hip": "hip",
    "belt/holster": "belt",
    # ---- pocket -----------------------------------------------------------
    "the left trouser pocket": "left_pocket",
    "left trouser pocket": "left_pocket",
    "left front trouser pocket": "left_pocket",
    "the right trouser pocket": "right_pocket",
    "right trouser pocket": "right_pocket",
    "a trouser pocket": "pocket_unspecified",
    "trouser pocket": "pocket_unspecified",
    "pocket": "pocket_unspecified",
    "front pocket": "front_pocket",
    # ---- thigh ------------------------------------------------------------
    "the left thigh": "left_thigh",
    "the right thigh": "right_thigh",
    "thigh": "thigh_unspecified",
    # ---- shin / calf ------------------------------------------------------
    "the left shin": "left_shin",
    "the right shin": "right_shin",
    "the left calf": "left_calf",
    "the right calf": "right_calf",
    # "Shank" is the whole lower-leg segment an inertial suit tracks; shin and calf are its
    # anterior and posterior faces. Distinct sites, but equivalent — see EQUIVALENT_SITES.
    "the left shank": "left_shank",
    "the right shank": "right_shank",
    # ---- knee / ankle -----------------------------------------------------
    "the outer side of the left knee": "left_knee",
    "the outer side of the right knee": "right_knee",
    "the left ankle": "left_ankle",
    # ---- trunk ------------------------------------------------------------
    "the back": "back",
    "lower back": "lower_back",
    "the lower back": "lower_back",
    "the chest": "chest",
    "the torso": "torso",
    # Nymeria's suit and its synthetic counterpart mount at the sternum. That is the chest,
    # so it is grouped with it below rather than pretending to be a separate observation.
    "the sternum": "sternum",
    # The pelvis is its own site, NOT pooled with waist/hip/belt: a pelvis-strapped suit
    # tracker moves with the body segment, while a belt or waistband unit can rotate and slip
    # against it. Grouping them would let one support a query about the other.
    "the pelvis": "pelvis",
    # ---- head -------------------------------------------------------------
    "an earbud in the ear": "ear_unspecified",
    "an earbud in the left ear": "left_ear",
    "the head (smart glasses)": "head",
    # A head-strapped suit tracker observes the same site as glasses do; the device family
    # already separates them in the key, so the SITE should not also pretend to.
    "the head": "head",
    # ---- muscle bellies (kneepad; mapped for completeness, not used for Nov 1) ---
    "the left gastrocnemius": "left_gastrocnemius",
    "the right gastrocnemius": "right_gastrocnemius",
    "the left hamstrings": "left_hamstrings",
    "the right hamstrings": "right_hamstrings",
    "the left rectus femoris": "left_rectus_femoris",
    "the right rectus femoris": "right_rectus_femoris",
    "the left tibialis anterior": "left_tibialis_anterior",
    "the right tibialis anterior": "right_tibialis_anterior",
}


#: Sites that count as "similar placement" for :func:`is_near_miss`. Each group is a set of sites
#: that differ only by body side or by a pocket variant — the two cases named in the design
#: decision. Sites absent from every group are equivalent to themselves alone.
#:
#: Deliberately NOT grouped:
#:   * ``wrist`` with ``forearm`` — a watch and a strapped forearm IMU sit on different lever arms.
#:   * ``pocket`` with ``waist``/``hip``/``belt`` — a pocketed phone swings with the thigh, a belt
#:     unit moves with the torso. Those are different signals, not a placement nuance.
#:   * anything spanning limbs (wrist with ankle), which the decision puts out of scope entirely.
EQUIVALENT_SITES: tuple[frozenset[str], ...] = (
    frozenset({
        "left_wrist", "right_wrist", "wrist_unspecified",
        "non_dominant_wrist", "affected_wrist", "unaffected_wrist",
    }),
    frozenset({"left_forearm", "right_forearm", "forearm_unspecified"}),
    frozenset({"left_upper_arm", "right_upper_arm"}),
    frozenset({"left_pocket", "right_pocket", "pocket_unspecified", "front_pocket"}),
    frozenset({"waist", "hip", "belt"}),
    frozenset({"left_thigh", "right_thigh", "thigh_unspecified"}),
    frozenset({
        "left_shin", "right_shin", "left_calf", "right_calf",
        "left_shank", "right_shank",
    }),
    frozenset({"chest", "sternum"}),
    frozenset({"left_knee", "right_knee"}),
    frozenset({"back", "lower_back"}),
    frozenset({"left_ear", "ear_unspecified"}),
    frozenset({"left_gastrocnemius", "right_gastrocnemius"}),
    frozenset({"left_hamstrings", "right_hamstrings"}),
    frozenset({"left_rectus_femoris", "right_rectus_femoris"}),
    frozenset({"left_tibialis_anterior", "right_tibialis_anterior"}),
)


@dataclass(frozen=True, order=True)
class AcquisitionKey:
    """What must agree for two recordings to be directly comparable.

    Sampling rate is deliberately absent: the physical-Hz filterbank is rate-invariant by
    construction, so rate is nuisance variation within one configuration rather than a different
    configuration. Device model is absent for the same reason.
    """

    device_family: str
    site: str
    channels: tuple[str, ...]
    gravity_state: str


def placement_site(placement: str) -> str:
    """Closed-vocabulary site for one curated placement string."""

    text = " ".join(placement.strip().lower().split())
    if not text:
        raise ValueError("placement must be non-empty")
    try:
        return PLACEMENT_SITE[text]
    except KeyError as error:
        raise KeyError(
            f"placement {placement!r} has no site in PLACEMENT_SITE. Add it explicitly — a new "
            "acquisition configuration must state where the sensor sits before its recordings may "
            "enter a support set."
        ) from error


def device_family(device_profile: str) -> str:
    """Coarse form factor for one curated ``device_profile``."""

    text = device_profile.strip().lower()
    try:
        return DEVICE_FAMILY[text]
    except KeyError as error:
        raise KeyError(
            f"device_profile {device_profile!r} has no family in DEVICE_FAMILY; add it explicitly"
        ) from error


def site_group(site: str) -> frozenset[str]:
    """The equivalence group containing ``site``; a singleton when it has no equivalents."""

    for group in EQUIVALENT_SITES:
        if site in group:
            return group
    return frozenset({site})


def acquisition_key(
    *,
    device_profile: str,
    placement: str,
    channels: Sequence[str],
    gravity_state: str,
) -> AcquisitionKey:
    """Build the key from curated stream metadata."""

    if gravity_state not in {"present", "removed", "unknown"}:
        raise ValueError(f"invalid gravity state: {gravity_state!r}")
    ordered = tuple(sorted(set(channels)))
    if not ordered:
        raise ValueError("a stream must declare at least one channel")
    return AcquisitionKey(
        device_family=device_family(device_profile),
        site=placement_site(placement),
        channels=ordered,
        gravity_state=gravity_state,
    )


def stream_key(dataset: str, stream: str) -> AcquisitionKey:
    """Key for one corpus stream, read from its :class:`StreamSpec`."""

    spec = deployment_policy.get_stream_spec(dataset, stream)
    channels = tuple(spec.required) + tuple(spec.optional)
    return acquisition_key(
        device_profile=spec.device_profile,
        placement=spec.placement,
        channels=channels,
        gravity_state=spec.gravity_state,
    )


def are_compatible(first: AcquisitionKey, second: AcquisitionKey) -> bool:
    """True when two configurations are identical, which is the core arm's support rule."""

    return first == second


def is_near_miss(first: AcquisitionKey, second: AcquisitionKey) -> bool:
    """True for the experiment's "no perfectly compatible example" case.

    Same device family, sites in one equivalence group, and not already compatible. Channels and
    gravity may differ; anything outside the group is out of scope rather than a further tier.
    """

    if are_compatible(first, second):
        return False
    if first.device_family != second.device_family:
        return False
    return second.site in site_group(first.site)


def corpus_keys(
    datasets: Iterable[str] | None = None,
) -> dict[tuple[str, str], AcquisitionKey]:
    """Every ``(dataset, stream) -> key``, for auditing the corpus.

    ``datasets=None`` covers every curated stream, including roles other than ``primary``.
    """

    wanted = None if datasets is None else set(datasets)
    keys: dict[tuple[str, str], AcquisitionKey] = {}
    for spec in deployment_policy.STREAM_SPECS:
        if wanted is not None and spec.dataset not in wanted:
            continue
        keys[(spec.dataset, spec.stream_id)] = acquisition_key(
            device_profile=spec.device_profile,
            placement=spec.placement,
            channels=tuple(spec.required) + tuple(spec.optional),
            gravity_state=spec.gravity_state,
        )
    return keys
