"""Acquisition-key tests (IMWUT handoff W1).

The support-set filter is the one place the comparison model is given help it did not learn. If it
is wrong the core arm is either impossible (pool fragments to singletons) or dishonest (support
that no deployment would have). These tests pin the two failure modes the sweep actually found.
"""

from __future__ import annotations

import pytest

from data.scripts.curate import deployment_policy
from data.scripts.curate.compatibility import (
    EQUIVALENT_SITES,
    PLACEMENT_SITE,
    AcquisitionKey,
    acquisition_key,
    are_compatible,
    corpus_keys,
    device_family,
    is_near_miss,
    placement_site,
    site_group,
    stream_key,
)


def test_every_curated_placement_has_a_site():
    """A new dataset must declare where its sensor sits; there is no silent fallback."""
    missing = sorted({
        spec.placement
        for spec in deployment_policy.STREAM_SPECS
        if " ".join(spec.placement.strip().lower().split()) not in PLACEMENT_SITE
    })
    assert not missing, f"placements with no site: {missing}"


def test_every_curated_device_profile_has_a_family():
    for spec in deployment_policy.STREAM_SPECS:
        device_family(spec.device_profile)


def test_prose_variants_of_one_site_collapse():
    """The bug this module exists to fix: six spellings of the wrist were six keys."""
    assert placement_site("the left wrist") == placement_site("left wrist")
    assert placement_site("the right wrist") == placement_site("right wrist")
    # A source that says only "the wrist" is pooled with one that says "dominant": single-wrist
    # studies use the preferred wrist by convention, and splitting them would strand monipar with
    # no compatible training partner at all.
    for text in ("the wrist", "wrist", "dominant wrist", "the dominant wrist"):
        assert placement_site(text) == "wrist_unspecified"


def test_semantically_distinct_wrists_do_not_collapse():
    """Wording is noise; handedness and clinical status are not.

    An explicitly non-dominant wrist is a different observation from the dominant one, and
    upper_limb_use's affected/unaffected pair IS the contrast that dataset exists to measure.
    Pooling them would let an unaffected-arm recording support an affected-arm query the moment
    cross-configuration cells are enabled. They stay wrist-EQUIVALENT (a near miss) but never
    identical.
    """
    sites = {
        placement_site("the non-dominant wrist"),
        placement_site("the wrist of the more-affected arm"),
        placement_site("the wrist of the less-affected arm"),
        placement_site("the wrist"),
    }
    assert len(sites) == 4, f"distinct wrist descriptions collapsed together: {sites}"
    # ...but all of them remain mutually near-miss-eligible.
    group = site_group("wrist_unspecified")
    assert {"non_dominant_wrist", "affected_wrist", "unaffected_wrist"} <= group


def test_separating_the_wrists_cost_no_compatible_pool():
    """The split was made only because it was free; assert that it stayed free.

    Originally checked against monipar; that source left the roster on 2026-09-11, so the guard
    now covers every live watch-wrist evaluation stream instead.
    """
    from data.scripts.curate.compatibility import are_compatible

    train = set(deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS)
    keys = corpus_keys()
    for eval_dataset in deployment_policy.PRIMARY_EVAL_DATASETS:
        for spec in deployment_policy.stream_specs(eval_dataset, "primary"):
            key = stream_key(eval_dataset, spec.stream_id)
            if key.device_family != "watch":
                continue
            partners = [
                1 for (dataset, _stream), other in keys.items()
                if dataset in train and are_compatible(key, other)
            ]
            assert partners, f"{eval_dataset}/{spec.stream_id} has no compatible training stream"


def test_laterality_is_preserved_not_collapsed():
    """Left and right are distinct sites, so they are a near miss rather than identical."""
    assert placement_site("the left wrist") != placement_site("the right wrist")


def test_unknown_placement_raises_loudly():
    with pytest.raises(KeyError):
        placement_site("strapped to the left eyebrow")


def test_key_reads_placement_not_stream_id():
    """nfi_fared's stream is called 'wrist' but the sensor is on the forearm."""
    assert stream_key("nfi_fared", "wrist").site == "forearm_unspecified"


def test_cross_dataset_wrist_streams_are_compatible():
    """The core arm needs support drawn from other datasets, which requires this to hold."""
    a = stream_key("dsads", "left_wrist")
    b = stream_key("forth_trace", "left_wrist")
    assert are_compatible(a, b)
    assert a.device_family == "body_imu"


def test_gravity_removed_is_never_compatible_with_gravity_present():
    """kuhar is the corpus's gravity-removed phone stream; its DC component is ~0, not ~1 g."""
    removed = stream_key("kuhar", "phone_waist")
    present = stream_key("hhar", "phone_waist")
    assert removed.gravity_state == "removed"
    assert present.gravity_state == "present"
    assert not are_compatible(removed, present)


def test_left_and_right_wrist_are_a_near_miss():
    """The decision's own example: you may compare a left wrist against a right wrist."""
    left = stream_key("dsads", "left_wrist")
    right = stream_key("dsads", "right_wrist")
    assert not are_compatible(left, right)
    assert is_near_miss(left, right)
    assert is_near_miss(right, left)


def test_pocket_variants_are_a_near_miss():
    left = stream_key("xrf_v2", "left_pocket")
    right = stream_key("xrf_v2", "right_pocket")
    assert not are_compatible(left, right)
    assert is_near_miss(left, right)


def test_wrist_against_ankle_is_out_of_scope():
    """Not a further tier — simply not a relation we define."""
    wrist = stream_key("forth_trace", "left_wrist")
    ankle = stream_key("forth_trace", "left_ankle")
    assert not are_compatible(wrist, ankle)
    assert not is_near_miss(wrist, ankle)


def test_watch_against_phone_is_out_of_scope():
    watch = stream_key("wisdm", "watch_wrist")
    phone = stream_key("wisdm", "phone_pocket")
    assert not are_compatible(watch, phone)
    assert not is_near_miss(watch, phone)


def test_wrist_and_forearm_stay_separate():
    """A watch and a strapped forearm IMU sit on different lever arms; not asserted equivalent."""
    assert site_group("left_wrist").isdisjoint(site_group("left_forearm"))


def test_pocket_and_waist_stay_separate():
    """A pocketed phone swings with the thigh; a belt unit moves with the torso."""
    assert site_group("left_pocket").isdisjoint(site_group("waist"))


def test_near_miss_is_not_reflexive_and_is_symmetric():
    key = stream_key("dsads", "left_wrist")
    assert not is_near_miss(key, key)
    other = stream_key("dsads", "right_wrist")
    assert is_near_miss(key, other) == is_near_miss(other, key)


def test_equivalence_groups_are_disjoint():
    """A site in two groups would make near-miss ill-defined."""
    seen: set[str] = set()
    for group in EQUIVALENT_SITES:
        assert seen.isdisjoint(group), f"site in two groups: {seen & group}"
        seen |= group


def test_rate_is_not_part_of_the_key():
    """The filterbank is rate-invariant by construction, so rate is nuisance, not configuration."""
    fields = set(AcquisitionKey.__dataclass_fields__)
    assert fields == {"device_family", "site", "channels", "gravity_state"}


def test_channels_are_order_insensitive():
    first = acquisition_key(
        device_profile="watch", placement="the left wrist",
        channels=("acc_x", "acc_y", "acc_z"), gravity_state="present",
    )
    second = acquisition_key(
        device_profile="watch", placement="left wrist",
        channels=("acc_z", "acc_x", "acc_y"), gravity_state="present",
    )
    assert are_compatible(first, second)


def test_channel_mismatch_breaks_compatibility_but_allows_near_miss():
    """acc-only against acc+gyro at the same site is a real deployment mismatch."""
    acc_only = acquisition_key(
        device_profile="watch", placement="the left wrist",
        channels=("acc_x", "acc_y", "acc_z"), gravity_state="present",
    )
    with_gyro = acquisition_key(
        device_profile="watch", placement="the right wrist",
        channels=("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        gravity_state="present",
    )
    assert not are_compatible(acc_only, with_gyro)
    assert is_near_miss(acc_only, with_gyro)


def test_corpus_keys_cover_every_curated_stream():
    keys = corpus_keys()
    assert len(keys) == len(deployment_policy.STREAM_SPECS)


def test_training_corpus_has_multi_dataset_pools():
    """At least one key must draw support from several datasets, or no episode is cross-dataset."""
    train = set(deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS)
    pools: dict[AcquisitionKey, set[str]] = {}
    for (dataset, _stream), key in corpus_keys().items():
        if dataset in train:
            pools.setdefault(key, set()).add(dataset)
    # Truthful hardware profiles still leave cross-source pools; they no longer manufacture a
    # watch equivalence for wrist-mounted Xsens/Shimmer research nodes.
    assert max(len(datasets) for datasets in pools.values()) >= 3
