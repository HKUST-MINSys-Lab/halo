"""Tests for the harmonised canonical-label vocabulary."""

import json

from data.scripts.labels.canonical_labels import SYNONYMS, canonicalize

TRAIN = ["uci_har", "hhar", "pamap2", "wisdm", "kuhar", "unimib_shar", "hapt", "mhealth", "capture24"]


def _all_train_labels():
    labels = set()
    for ds in TRAIN:
        labels.update(json.load(open(f"data/datasets/{ds}/metadata.json")).get("activities", []))
    return labels


def test_map_is_idempotent():
    for raw in _all_train_labels():
        c = canonicalize(raw)
        assert canonicalize(c) == c, f"{raw} -> {c} is not idempotent"


def test_no_canonical_is_a_merged_away_synonym():
    canon = {canonicalize(l) for l in _all_train_labels()}
    stray = canon & set(SYNONYMS)
    assert not stray, f"these canonical names are themselves synonyms (double-merge): {stray}"


def test_specific_synonym_merges():
    assert canonicalize("laying") == "lying"
    assert canonicalize("bicycling") == "cycling"
    assert canonicalize("ascending_stairs") == "walking_upstairs"
    assert canonicalize("going_down_stairs") == "walking_downstairs"
    assert canonicalize("climbing_stairs") == "stairs"
    assert canonicalize("sit_to_stand") == "standing_up_from_sitting"
    assert canonicalize("lie_to_stand") == "standing_up_from_lying"
    assert canonicalize("standing_up_from_laying") == "standing_up_from_lying"
    assert canonicalize("pushups") == "push_up"
    assert canonicalize("situps") == "sit_up"
    assert canonicalize("squats") == "squat"
    assert canonicalize("jump_front_and_back") == "jump_front_back"
    assert canonicalize("jump_rope") == "rope_jumping"
    assert canonicalize("arms_frontal_elevation") == "frontal_elevation_arms"
    assert canonicalize("knees_bending_crouching") == "knees_bending"
    assert canonicalize("transition_from_sitting_to_standing") == "standing_up_from_sitting"
    assert canonicalize("transition_from_standing_to_sitting") == "sitting_down"
    assert canonicalize("talking_sitting") == "sitting_and_talking"


def test_near_but_distinct_activities_are_kept():
    # canonical merge, NOT aggressive taxonomy merge: these must stay separate.
    for x in ["jogging", "running", "stairs", "falling_forward", "rope_jumping", "jump_front_back"]:
        assert canonicalize(x) == x, f"{x} was wrongly merged"


def test_vocab_shrinks_yet_covers_every_activity():
    raw = _all_train_labels()
    canon = {canonicalize(l) for l in raw}
    assert len(canon) < len(raw)                      # synonyms actually merged
    assert all(canonicalize(l) in canon for l in raw)  # every activity still representable
