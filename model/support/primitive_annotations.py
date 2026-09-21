"""Written primitive profiles for the TRAINING labels: the knowledge half of the common ground.

The 2026-09-21 post-mortem of T7 located the primitive path's failure on its label side: a label's
profile was SBERT similarity to primitive sentences, which the pre-training audit had measured at
about half random per axis. The sensor side learned to decompose recordings; the label side never
knew what the primitives meant, so on a foreign vocabulary it was confidently wrong.

This module writes the label side down. Every one of the 155 supervised training labels gets one
value per axis of ``primitives-v1`` (see ``primitive_semantics``). A frozen map fitted on these
pairs (``PrimitiveSemanticHead`` with ``label_side="annotated"``) then gives an unseen label a
profile by inheriting from annotated labels that are *near it in text space* -- so "burpee"
inherits from jumping, squats and push-ups, which share its attributes even though none shares its
name. Averaging attributes is meaningful where averaging identities is not.

Protocol discipline: **training labels only.** A test asserts that the key set equals the
supervised training vocabulary exactly, and the text is hashed so an edit cannot silently reuse a
version name. Some training concepts also occur verbatim in sealed datasets; those are explicitly
seen-vocabulary evaluations, not evaluation-only annotations. The scrambled version permutes the
profiles across labels and is the control that separates grounding from capacity.

Each entry is ``intensity rhythm impact travel posture regularity upper_limbs lower_limbs trunk
head`` in vocabulary order. Treadmill locomotion is ``stationary``: the body does not move across
the ground, and the rhythm and impact axes carry the gait signature.
"""

from __future__ import annotations

import hashlib

from model.support.primitive_semantics import (
    PRIMITIVE_VOCABULARY_VERSION, VOCABULARIES, axis_names, axis_slices, n_primitives,
)

ANNOTATION_VERSION = "annotations-v1"
ANNOTATION_VOCABULARY = PRIMITIVE_VOCABULARY_VERSION

_ANNOTATIONS_V1_TEXT = """
answering_phone: light none none stationary upright single_burst irregular still still still
applying_hand_cream: light slow none stationary upright steady rhythmic still still still
arms_frontal_crossing: light slow none stationary upright steady rhythmic still still still
arms_inner_rotation: light slow none stationary upright steady rhythmic still still still
arms_lateral_elevation: light slow none stationary upright steady rhythmic still still still
baking: light none none stationary upright varied irregular irregular irregular irregular
bathroom_cleaning_bathtub: moderate slow none stationary bent_over varied irregular irregular irregular irregular
bathroom_cleaning_mirror: light slow none stationary upright varied rhythmic still irregular irregular
bathroom_cleaning_toilet: moderate slow none stationary bent_over varied irregular irregular irregular irregular
bathroom_cleaning_washbasin: light slow none stationary bent_over varied rhythmic still irregular irregular
brushing_teeth: light fast none stationary upright steady rhythmic still still still
build_with_lego: light none none stationary upright varied irregular still still still
clapping: light fast light stationary upright steady rhythmic still still still
cleaning_bathroom: moderate slow none stationary bent_over varied irregular irregular irregular irregular
cleaning_door: light slow none stationary upright varied rhythmic still irregular irregular
cleaning_shelf: light slow none stationary upright varied rhythmic still irregular irregular
cleaning_table: light slow none stationary bent_over varied rhythmic still irregular irregular
cleaning_the_cupboard: light slow none stationary upright varied irregular still irregular irregular
climbing_stairs_and_talking: moderate stepping light travelling upright steady rhythmic rhythmic rhythmic irregular
cutting_cardboard: light slow none stationary bent_over varied irregular still irregular still
cutting_fruit: light fast none stationary bent_over steady rhythmic still still still
cutting_plants_near_water: moderate slow none stationary bent_over varied irregular irregular irregular irregular
cutting_vegetables: light fast none stationary bent_over steady rhythmic still still still
cutting_wood: vigorous slow hard stationary bent_over steady rhythmic still rhythmic still
cycling: moderate slow none travelling bent_over steady still rhythmic still still
cycling_on_an_exercise_bike_in_horizontal_position: moderate slow none stationary horizontal steady still rhythmic still still
cycling_on_an_exercise_bike_in_vertical_position: moderate slow none stationary upright steady still rhythmic still still
disinfecting_hands: light fast none stationary upright steady rhythmic still still still
doing_a_puzzle: still none none stationary upright varied irregular still still still
drawing: light none none stationary upright varied irregular still still still
dribbling: moderate fast light stationary bent_over steady rhythmic irregular rhythmic still
drinking: light none none stationary upright single_burst irregular still still irregular
dusting_furniture: light slow none stationary upright varied rhythmic irregular irregular irregular
dusting_model_cars: light slow none stationary upright varied rhythmic still still still
eating_chips: light none none stationary upright varied irregular still still irregular
eating_fruit: light none none stationary upright varied irregular still still irregular
eating_pasta: light none none stationary upright varied irregular still still irregular
eating_sandwich: light none none stationary upright varied irregular still still irregular
eating_soup: light slow none stationary upright steady rhythmic still still irregular
elliptic_bike: moderate slow none stationary upright steady rhythmic rhythmic still still
emptying_dishwasher: light none none stationary changing varied irregular irregular irregular irregular
exercising_on_a_cross_trainer: moderate slow none stationary upright steady rhythmic rhythmic still still
exercising_on_a_stepper: moderate stepping light stationary upright steady still rhythmic still still
feeding_the_chickens: light none none travelling upright varied irregular irregular irregular irregular
feeding_the_dog: light none none stationary bent_over single_burst irregular still irregular irregular
floor_cleaning: moderate slow none travelling bent_over varied rhythmic irregular irregular irregular
folding_clothes: light none none stationary upright varied irregular still still still
folding_laundry: light none none stationary upright varied irregular still still still
frontal_elevation_arms: light slow none stationary upright steady rhythmic still still still
frontal_hand_claps: light fast light stationary upright steady rhythmic still still still
gardening_with_a_rake: moderate slow none travelling bent_over steady rhythmic irregular rhythmic still
hanging_up_laundry: light none none stationary upright varied irregular still irregular irregular
heels_alternately_to_the_backside: moderate stepping light stationary upright steady still rhythmic still still
ironing_laundry: light slow none stationary upright steady rhythmic still still still
jogging: vigorous fast hard travelling upright steady rhythmic rhythmic rhythmic rhythmic
jump_front_back: vigorous fast hard stationary upright steady still rhythmic rhythmic rhythmic
jump_sideways: vigorous fast hard stationary upright steady still rhythmic rhythmic rhythmic
jump_up: vigorous fast hard stationary upright steady rhythmic rhythmic rhythmic rhythmic
jump_with_legs_and_arms_open_and_closed: vigorous fast hard stationary upright steady rhythmic rhythmic rhythmic rhythmic
jumping: vigorous fast hard stationary upright steady rhythmic rhythmic rhythmic rhythmic
kicking: moderate slow light stationary upright single_burst still irregular irregular still
knees_alternately_bending_forward: moderate stepping light stationary upright steady still rhythmic still still
knees_alternately_to_the_breast: moderate stepping light stationary upright steady still rhythmic rhythmic still
knees_bending: moderate slow none stationary changing steady still rhythmic rhythmic still
lateral_bend: light slow none stationary changing steady rhythmic still rhythmic rhythmic
lateral_bend_with_the_arm_up: light slow none stationary changing steady rhythmic still rhythmic rhythmic
lying: still none none stationary horizontal steady still still still still
lying_down_from_standing: light none none stationary changing single_burst irregular irregular irregular irregular
lying_on_back: still none none stationary horizontal steady still still still still
lying_on_right_side: still none none stationary horizontal steady still still still still
making_tea: light none none stationary upright varied irregular still still still
moving_around_in_an_elevator: light none none stationary upright varied irregular irregular irregular irregular
opening_curtains: light none none stationary upright single_burst irregular still still still
opening_envelope: light none none stationary upright single_burst irregular still still still
opening_windows: light none none stationary upright single_burst irregular still still still
petting_the_dog: light slow none stationary bent_over steady rhythmic still still still
picking_fruit: light none none stationary upright varied irregular irregular irregular irregular
picking_up: light none none stationary changing single_burst irregular still irregular irregular
playing_basketball: vigorous fast hard travelling upright varied irregular irregular irregular irregular
playing_catch: moderate none light stationary upright varied irregular irregular irregular irregular
playing_with_the_dog: moderate none none travelling changing varied irregular irregular irregular irregular
pouring_water: light none none stationary upright single_burst irregular still still still
push_up: vigorous slow none stationary horizontal steady rhythmic still rhythmic still
putting_away_dishes: light none none stationary changing varied irregular irregular irregular irregular
reach_heels_backwards: light slow none stationary changing steady still rhythmic rhythmic still
reading: still none none stationary upright steady still still still still
refill: light none none stationary upright single_burst irregular still still still
refill_bucket: light none none stationary bent_over single_burst irregular still irregular still
refill_water: light none none stationary upright single_burst irregular still still still
removing_plants_near_water: moderate none none stationary bent_over varied irregular irregular irregular irregular
removing_protective_film: light none none stationary upright varied irregular still still still
repeated_standing_and_lying: moderate slow none stationary changing steady irregular irregular rhythmic irregular
repeated_standing_and_sitting: moderate slow none stationary changing steady still rhythmic rhythmic still
repetitive_forward_stretching: light slow none stationary changing steady rhythmic still rhythmic rhythmic
rope_jumping: vigorous fast hard stationary upright steady rhythmic rhythmic rhythmic rhythmic
rotation_on_the_knees: light slow none stationary bent_over steady still still rhythmic rhythmic
rowing: moderate slow none stationary upright steady rhythmic rhythmic rhythmic still
running: vigorous fast hard travelling upright steady rhythmic rhythmic rhythmic rhythmic
running_on_a_treadmill: vigorous fast hard stationary upright steady rhythmic rhythmic rhythmic rhythmic
shoulders_high_amplitude_rotation: light slow none stationary upright steady rhythmic still still still
shoulders_low_amplitude_rotation: light slow none stationary upright steady rhythmic still still still
sit_up: moderate slow none stationary changing steady still still rhythmic rhythmic
sitting: still none none stationary upright steady still still still still
sitting_and_talking: still none none stationary upright varied irregular still still irregular
sitting_down: light none none stationary changing single_burst still irregular irregular still
sorting_documents: light none none stationary upright varied irregular still still still
sowing_seeds: light none none travelling bent_over varied irregular irregular irregular irregular
stairs: moderate stepping light travelling upright steady rhythmic rhythmic rhythmic still
standing: still none none stationary upright steady still still still still
standing_still_in_an_elevator: still none none stationary upright steady still still still still
standing_up_from_lying: light none none stationary changing single_burst irregular irregular irregular irregular
standing_up_from_sitting: light none none stationary changing single_burst still irregular irregular still
stretching: light slow none stationary changing steady rhythmic still rhythmic still
sweeping_the_yard: moderate slow none travelling bent_over steady rhythmic irregular rhythmic still
table_tennis: moderate none light stationary upright varied irregular irregular irregular irregular
taking_medicine: light none none stationary upright single_burst irregular still still irregular
talking_standing: still none none stationary upright varied irregular still still irregular
throwing_garbage: light none none stationary upright single_burst irregular still irregular still
tidy_up_the_cupboard: light none none stationary upright varied irregular irregular irregular irregular
tidy_up_the_wardrobe: light none none stationary upright varied irregular irregular irregular irregular
toggling_lamp: light none none stationary upright single_burst irregular still still still
transition_from_climbing_stairs_and_talking_to_walking_and_talking: moderate stepping light travelling upright varied rhythmic rhythmic rhythmic irregular
transition_from_climbing_stairs_to_walking: moderate stepping light travelling upright varied rhythmic rhythmic rhythmic still
transition_from_sitting_and_talking_to_standing: light none none stationary changing single_burst irregular irregular irregular irregular
transition_from_standing_to_climbing_stairs: moderate stepping light travelling upright varied rhythmic rhythmic rhythmic still
transition_from_standing_to_sitting_and_talking: light none none stationary changing single_burst irregular irregular irregular irregular
transition_from_standing_to_walking: light stepping light travelling upright varied rhythmic rhythmic still still
transition_from_walking_to_standing: light stepping light travelling upright varied rhythmic rhythmic still still
trunk_twist_with_arms_outstretched: light slow none stationary upright steady rhythmic still rhythmic rhythmic
trunk_twist_with_elbows_bent: light slow none stationary upright steady rhythmic still rhythmic rhythmic
typing: light fast none stationary upright steady rhythmic still still still
upper_trunk_and_lower_body_opposite_twist: light slow none stationary upright steady rhythmic rhythmic rhythmic still
using_mouse: light none none stationary upright varied irregular still still still
using_phone: light none none stationary upright varied irregular still still still
vacuum_cleaning: moderate slow none travelling upright steady rhythmic irregular irregular irregular
waist_bend_reaching_the_foot_with_the_opposite_hand: light slow none stationary changing steady rhythmic still rhythmic rhythmic
waist_bends_forward: light slow none stationary changing steady still still rhythmic rhythmic
waist_rotation: light slow none stationary upright steady still still rhythmic rhythmic
walking: moderate stepping light travelling upright steady rhythmic rhythmic still still
walking_and_talking: moderate stepping light travelling upright steady rhythmic rhythmic still irregular
walking_backwards: light stepping light travelling upright steady rhythmic rhythmic still still
walking_downstairs: moderate stepping light travelling upright steady rhythmic rhythmic still still
walking_in_a_parking_lot: moderate stepping light travelling upright steady rhythmic rhythmic still still
walking_in_circles: moderate stepping light travelling upright steady rhythmic rhythmic still irregular
walking_on_a_treadmill_in_flat_position: moderate stepping light stationary upright steady rhythmic rhythmic still still
walking_on_a_treadmill_in_inclined_position: moderate stepping light stationary upright steady rhythmic rhythmic still still
walking_upstairs: moderate stepping light travelling upright steady rhythmic rhythmic rhythmic still
washing_dishes: light slow none stationary bent_over varied rhythmic still still still
washing_hands: light fast none stationary bent_over steady rhythmic still still still
watering_plants: light none none travelling upright varied irregular irregular irregular irregular
window_cleaning: light slow none stationary upright steady rhythmic still irregular irregular
working_in_the_garden: moderate none none travelling bent_over varied irregular irregular irregular irregular
working_on_the_computer: light none none stationary upright varied irregular still still still
writing: light none none stationary upright varied irregular still still still
writing_on_blackboard: light slow none stationary upright varied rhythmic still still still
"""


def _parse(text: str, vocabulary_version: str) -> dict[str, tuple[str, ...]]:
    axes = axis_names(vocabulary_version)
    allowed = {axis: {name for name, _ in values} for axis, values in VOCABULARIES[vocabulary_version]}
    out: dict[str, tuple[str, ...]] = {}
    for line in text.strip().splitlines():
        label, _, rest = line.partition(":")
        values = tuple(rest.split())
        if len(values) != len(axes):
            raise ValueError(f"{label}: expected {len(axes)} values, got {len(values)}")
        for axis, value in zip(axes, values):
            if value not in allowed[axis]:
                raise ValueError(f"{label}: {value!r} is not a value of axis {axis!r}")
        if label in out:
            raise ValueError(f"{label} is annotated twice")
        out[label] = values
    return out


ANNOTATIONS_V1: dict[str, tuple[str, ...]] = _parse(_ANNOTATIONS_V1_TEXT, ANNOTATION_VOCABULARY)


def _scrambled(annotations: dict[str, tuple[str, ...]], seed: int = 0) -> dict[str, tuple[str, ...]]:
    """Profiles permuted across labels: the grounding control. Same profiles, wrong owners."""
    import random

    labels = sorted(annotations)
    profiles = [annotations[label] for label in labels]
    random.Random(seed).shuffle(profiles)
    return dict(zip(labels, profiles))


ANNOTATIONS: dict[str, dict[str, tuple[str, ...]]] = {ANNOTATION_VERSION: ANNOTATIONS_V1}
SCRAMBLED_ANNOTATION_VERSION = f"{ANNOTATION_VERSION}-scrambled"
ANNOTATIONS[SCRAMBLED_ANNOTATION_VERSION] = _scrambled(ANNOTATIONS_V1)


def annotation_hash(version: str = ANNOTATION_VERSION) -> str:
    digest = hashlib.sha256()
    for label in sorted(ANNOTATIONS[version]):
        digest.update(label.encode())
        digest.update(" ".join(ANNOTATIONS[version][label]).encode())
    return digest.hexdigest()


def annotated_labels(version: str = ANNOTATION_VERSION) -> tuple[str, ...]:
    return tuple(sorted(ANNOTATIONS[version]))


def annotation_profile_matrix(version: str = ANNOTATION_VERSION, *, smoothing: float = 0.05,
                              vocabulary_version: str = ANNOTATION_VOCABULARY):
    """(N, K) profiles, one row per annotated label in sorted order, each axis summing to 1.

    ``smoothing`` spreads a little mass over the other values of each axis so no label-side
    probability is exactly zero; the chosen value keeps ``1 - smoothing + smoothing / V``.
    """
    import torch

    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    labels = annotated_labels(version)
    axes = axis_names(vocabulary_version)
    slices = axis_slices(vocabulary_version)
    value_index = {axis: {name: i for i, (name, _) in enumerate(values)}
                   for axis, values in VOCABULARIES[vocabulary_version]}
    matrix = torch.zeros((len(labels), n_primitives(vocabulary_version)), dtype=torch.float32)
    for row, label in enumerate(labels):
        for axis, block, value in zip(axes, slices, ANNOTATIONS[version][label]):
            width = block.stop - block.start
            matrix[row, block] = smoothing / width
            matrix[row, block.start + value_index[axis][value]] += 1.0 - smoothing
    return labels, matrix
