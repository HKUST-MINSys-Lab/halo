"""Unified activity-label vocabulary for every canonical training view.

When we harmonise, we also unify how activities are NAMED. Different training datasets describe the
same activity in different words — uci_har "laying" vs "lying"; pamap2 "ascending_stairs" vs
uci_har "walking_upstairs"; hapt "sit_to_stand" vs kuhar "standing_up_from_sitting". Canonical merge
maps each such raw label to ONE canonical name, so the harmonised corpus never describes the same
activity two ways.

Only GENUINE synonyms (and one spelling/scheme normalization) are merged — every distinct activity is
kept. In particular we deliberately DO NOT merge near-but-different activities: `jogging` ≠ `running`
(mhealth records both), each fall type stays separate, and wisdm's ambiguous `stairs` is kept as its
own label (it is neither specifically up nor down).

Scope: the harmonised **training** vocabulary only. Held-out eval datasets keep their native label
strings — they are the zero-shot targets. See docs/DATA_HETEROGENEITY.md.

`tests/test_canonical_labels.py` asserts every training label resolves, the map is idempotent, and no
canonical name is itself a merged-away synonym.
"""

from __future__ import annotations

# raw training label -> canonical label, with the reason it is a synonym.
# Anything NOT listed here is already its own canonical label.
SYNONYMS = {
    # --- posture spelling ---
    "laying": "lying",                             # uci_har spelling of "lying"
    "standing_up_from_laying": "standing_up_from_lying",  # kuhar/unimib "laying" -> "lying"

    # --- cycling ---
    "bicycling": "cycling",                        # capture24 vs hhar/pamap2/mhealth

    # --- stairs (direction preserved: ascending == up, descending == down) ---
    "ascending_stairs": "walking_upstairs",        # pamap2
    "going_up_stairs": "walking_upstairs",         # unimib_shar
    # FORTH-TRACE and mHealth both use this label for ascent AND descent. Inventing an upward
    # direction corrupts the target; retain the same undirected concept as WISDM's `stairs`.
    "climbing_stairs": "stairs",
    "climbingup": "walking_upstairs",              # realworld
    "descending_stairs": "walking_downstairs",     # pamap2
    "going_down_stairs": "walking_downstairs",     # unimib_shar
    "climbingdown": "walking_downstairs",          # realworld

    # --- posture transitions: hapt uses "X_to_Y"; kuhar/unimib use a descriptive name.
    #     Unify to the descriptive scheme (and normalize laying->lying). ---
    "sit_to_stand": "standing_up_from_sitting",    # == kuhar/unimib "standing_up_from_sitting"
    "lie_to_stand": "standing_up_from_lying",      # == kuhar/unimib "standing_up_from_laying"
    "stand_to_sit": "sitting_down",                # == unimib "sitting_down"
    "stand_to_lie": "lying_down_from_standing",     # == unimib "lying_down_from_standing"
    "lie_to_sit": "sitting_up_from_lying",         # hapt-only; named to match the scheme
    "sit_to_lie": "lying_down_from_sitting",       # hapt-only; named to match the scheme

    # Forth-TRACE uses sentence-like transition names for the same endpoints. Talking variants are
    # intentionally not merged because they are compound activities rather than plain transitions.
    "transition_from_sitting_to_standing": "standing_up_from_sitting",
    "transition_from_standing_to_sitting": "sitting_down",

    # --- exercise spelling/number ---
    # MM-Fit uses plural concatenated names; KU-HAR/PHYTMO use singular underscore forms. The source
    # protocols describe the same exercise, so keeping two classes would make checkpoint selection
    # reward dataset identity rather than activity structure.
    "pushups": "push_up",
    "situps": "sit_up",
    "squats": "squat",

    # --- equivalent source wording ---
    "jump_front_and_back": "jump_front_back",      # REALDISP vs MHEALTH
    "jump_rope": "rope_jumping",                  # REALDISP vs PAMAP2
    "arms_frontal_elevation": "frontal_elevation_arms",  # REALDISP vs MHEALTH
    "knees_bending_crouching": "knees_bending",   # REALDISP vs MHEALTH label description
    "talking_sitting": "sitting_and_talking",     # KU-HAR vs Forth-TRACE
}

# Reserved data-plumbing markers are not activity concepts and must never enter
# ConSE candidate vocabularies or Phase-B evidence labels.
NON_SEMANTIC_LABELS = frozenset({"__unlabeled__"})


def canonicalize(label: str) -> str:
    """Map a raw training label to its canonical name (identity if it has no synonym)."""
    return SYNONYMS.get(label, label)
