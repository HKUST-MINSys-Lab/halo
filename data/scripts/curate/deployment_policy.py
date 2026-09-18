"""Phone/watch deployment channel policy for HALO data preprocessing.

Raw converted datasets intentionally remain lossless. This module defines the
deployment-scoped view consumed by HALO and EDA: one physical phone or watch
stream with acceleration and, when trustworthy and co-located, gyroscope data.
Every curated frame therefore has exactly three or six sensor channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


STANDARD_CHANNEL_ORDER = (
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
)

# ---------------------------------------------------------------------------------------------
# Training rosters
# ---------------------------------------------------------------------------------------------
# Two kinds of roster live here and they must not be confused. FROZEN rosters are historical
# facts: the corpus a published baseline was fitted on, the corpus a dated checkpoint was trained
# on. They are literal tuples so that editing the ACTIVE roster can never silently rewrite what a
# past result meant. The ACTIVE roster is what a training script reaches for today.

# FROZEN. The corpus every retrained baseline was fitted on before the 2026-08 expansion. hapt was
# REMOVED from it (F11): a near-exact duplicate of uci_har (same 30 subjects, NCC 0.98) that was
# never in the trained corpus and double-counted the roster (13 vs the real 12).
CORPUS_MATCHED_TRAIN_DATASETS = (
    "uci_har",
    "hhar",
    "pamap2",
    "wisdm",
    "kuhar",
    "unimib_shar",
    "mhealth",
    "capture24",
    "sp_sw_har",
    "nfi_fared",
    "harmes",
    "xrf_v2",
)

# FROZEN. The 18-source design of record from 2026-08-12 to 2026-09-10. Every September 2026
# compact-engine and neighbours checkpoint, and the matched baseline tables, were produced on it.
# A checkpoint records its own roster in its config, so loading one never consults this; it exists
# so the number "18" in those reports still resolves to a definite list.
EXPANDED_18_TRAIN_DATASETS = CORPUS_MATCHED_TRAIN_DATASETS + (
    "dsads",
    "forth_trace",
    "opportunity",
    "realdisp",
    "mmfit",
    "phytmo",
)

# Sources withdrawn from the ACTIVE roster on 2026-09-10 after the labelled-corpus audit
# (docs/data/LABELLED_CORPUS_AUDIT_20260909.md). Their data stays on disk and their StreamSpecs
# stay below, so they remain loadable for reproduction; they are simply no longer trained on by
# default. `assert_no_retired_sources` is the gate every training entry point passes through.
RETIRED_TRAIN_DATASETS = {
    "uci_har": "91.5% of consecutive same-subject windows share an exact 50% sample overlap and "
               "the comparator's execution unit collapses to the window, so a support can be a "
               "half-copy of the query; 2.56 s windows against a 6 s context; its 6 labels all "
               "exist elsewhere.",
    "sp_sw_har": "1.0 s windows (a single patch, no temporal context to mask), 47.5% overlapping "
                 "windows, 1.0 h total; its 5 labels all exist elsewhere.",
    "mhealth": "gyroscope is sample-and-hold (72% exact consecutive repeats, effective ~13 Hz on "
               "a stream declared 50 Hz); 1.9 h, 10 subjects; labels covered by pamap2/dsads.",
    "opportunity": "4 subjects, 4 labels, 37% full windows. Four people cannot carry a cross-"
                   "subject comparison signal, yet uniform dataset sampling gave it 5.6% of "
                   "queries.",
    "capture24": "Promoted wholesale to the source-isolated, label-free Capture-24 JEPA corpus "
                 "on 2026-09-11. Keeping it in the supervised roster would reuse the same 151 "
                 "people and recordings across pretraining and support-conditioned fitting.",
    # Retired 2026-09-11 after a verified cost/benefit review of the head-training roster. The
    # corpus-shape finding behind it: only 11 of 937 (acquisition key, label) cells are reachable
    # from two or more datasets, so under the default `compatible` sampling mode 98.8% of episodes
    # draw every support from the query's own dataset. A source earns its place by covering an
    # evaluation configuration (phone pocket/waist/hip, watch right wrist) or by adding subjects at
    # a key another source shares; a source that adds a private key, a private vocabulary and a
    # structural defect adds maintenance surface and nothing the claim needs.
    #
    # NOT retired, and why (each re-measured on the grids on disk, 2026-09-11):
    #   hhar   - the only compatible training source for the phone-waist evaluation streams
    #            (realworld, inclusivehar); the 2026-09-10 converter rewrite kept the Galaxy S+ as an
    #            honest accel-only stream and anti-aliases to 50 Hz on the real clock.
    #   wisdm  - the only compatible source for shoaib's right pocket and the only near-miss watch
    #            source for tnda_har/ut_complex; both 2026-09-10 sweep defects are gone: 100% of
    #            sessions now run at 20 Hz and gyro exact-repeat fraction is 0.002 (was >30% for
    #            17/51 subjects).
    #   kuhar  - 89 subjects, the largest cross-subject pool in the corpus, at a phone-waist key
    #            with 8/8 of realworld's labels. Gravity-removed at source (median |acc| 0.09 g),
    #            so it can never be `compatible` with a gravity-present evaluation stream; it is
    #            a near-miss source for all three waist/hip streams. Its repeated stand/sit and
    #            stand/lie protocols and circular walking retain their own semantic labels.
    "pamap2": "3.3k windows, 0.9% of the corpus, yet dataset-first query sampling gave it 1/13 of "
              "all queries (an 8.6x up-weight); 9 subjects; body-IMU wrist that is neither "
              "compatible nor near-miss with any evaluation stream; only 2 labels no other source "
              "has. No defect, no coverage, a sampler distortion.",
    "unimib_shar": "Pre-windowed 3 s clips (median 3.02 s; 0% of windows fill the 6 s context), "
                   "the same structural mismatch that retired uci_har and sp_sw_har: half of "
                   "every comparison is padding. Accelerometer-only, and at most a near-miss to "
                   "any evaluation stream. The overlap-leak and lost-source reasons given on "
                   "2026-09-10 are withdrawn: the source CSVs are on disk and recordings.json maps "
                   "11,771 windows to 1,910 trials, which the execution unit already honours.",
    # Retired 2026-09-11 (second pass) to reach the eight-source head roster. Measured against the
    # six-source general-HAR evaluation roster: every one of the 11 cross-source (key, label)
    # cells in the corpus survives without these three, every evaluation stream keeps its
    # compatible or near-miss training partner, and the 20/25/50/51.2/100 Hz rate spread is kept.
    "mmfit": "Its compatible coverage of shoaib's right pocket and the watch right wrist is "
             "already supplied by wisdm and harmes; 8.5k windows, 21 subjects, gym exercises "
             "sharing no label with any evaluation set and no cross-source cell.",
    "phytmo": "Physical-therapy exercise protocol (Sci. Data 2022): rehabilitation content the "
              "programme has moved away from. Its six acquisition keys shared with realdisp carry "
              "zero shared labels, so the pooling they appeared to offer never produced a "
              "cross-source cell.",
    "nfi_fared": "Two dataset-private keys (lower back, forearm_unspecified) that pool with "
                 "nothing and match no evaluation stream; 14 subjects. Its 100 Hz and eval-label "
                 "coverage are supplied by kuhar and dsads. Swapping it for dsads keeps all 11 "
                 "cross-source cells and restores the 25 Hz anchor.",
}

# ACTIVE. The datasets whose raw source frames are curated to the 6-slot acc/gyro layout by
# `curate_frame` (`EXPECTED_PRIMARY_CHANNELS`, the channel-policy tests). NARROWER than the
# training corpus: the direct-converted multi-placement sources below train too but never touch
# this curation path, so they are NOT primary. See `DIRECT_CONVERTED_TRAIN_DATASETS`.
PRIMARY_TRAIN_DATASETS = tuple(
    dataset for dataset in CORPUS_MATCHED_TRAIN_DATASETS if dataset not in RETIRED_TRAIN_DATASETS
)

# ACTIVE. 2026-08-12: training datasets built by their OWN converters straight to native grids, not
# through the deployment channel-policy curation. They carry StreamSpecs (placement/device text) but
# no `curate_frame` source frame, so they are training-but-not-primary. Together with
# PRIMARY_TRAIN_DATASETS these are exactly TRAIN_DATASETS (F11 guard). The evaluation roster is
# intentionally absent; see SEALED_TEST_EVAL_DATASETS below.
DIRECT_CONVERTED_TRAIN_DATASETS = tuple(
    dataset for dataset in ("dsads", "forth_trace", "opportunity", "realdisp", "mmfit", "phytmo")
    if dataset not in RETIRED_TRAIN_DATASETS
)

# ACTIVE design-of-record support-classifier training corpus (8 sources since 2026-09-11).
# Adding or removing a
# dataset here is an experimental-protocol change: it needs a StreamSpec, converter, grids, quality
# caches, a regenerated data/labels/global_labels.json, and a corpus-matched baseline disclosure.
EXPANDED_PHASE_A_TRAIN_DATASETS = (
    PRIMARY_TRAIN_DATASETS + DIRECT_CONVERTED_TRAIN_DATASETS
)


def assert_no_retired_sources(datasets: Sequence[str], *, allow: bool = False) -> None:
    """Refuse to train on a retired source unless the caller says so explicitly.

    Every training entry point passes its resolved roster through here, so a retired dataset can
    only re-enter training via a deliberate ``--allow-retired`` (historical reproduction), never
    via a stale default, a hand-typed ``--datasets`` list, or an ablation subset that predates the
    retirement.
    """
    hit = sorted(set(datasets) & set(RETIRED_TRAIN_DATASETS))
    if hit and not allow:
        reasons = "\n".join(f"  {name}: {RETIRED_TRAIN_DATASETS[name]}" for name in hit)
        raise ValueError(
            f"retired training source(s) requested: {', '.join(hit)}.\n{reasons}\n"
            "These were withdrawn on 2026-09-10 (docs/data/LABELLED_CORPUS_AUDIT_20260909.md). "
            "Pass --allow-retired only to reproduce a run that predates the retirement."
        )


# Label-free sources living in `data/pretraining/`, sized and documented in
# `data/pretraining/corpus_plan.py` and `docs/data/PRETRAINING_CORPUS.md`. Every session in
# these carries `__unlabeled__`, so they can only ever feed representation pretraining: they
# are structurally incapable of reaching a label vocabulary, a validation probe, or the
# supervised application heads. Grid construction remains explicit so fetching a source cannot
# silently enlarge an existing run; the encoder trainer's named ``label_free`` recipe selects them.
PRETRAIN_SCALE_DATASETS = (
    "capture24_pretrain",
    "nymeria_xsens",
    "extrasensory_pretrain",
)

# ---------------------------------------------------------------------------------------------
# The pretraining / supervised split (decision 2026-09-09)
# ---------------------------------------------------------------------------------------------
# Until now one roster did both jobs: the encoder pretrained on the same 18 labelled corpora the
# comparator and classification head were then trained and selected on. That conflates two
# questions it should be possible to answer separately.
#
#   * Does self-supervised pretraining produce a useful representation? Asking that on data the
#     supervised stage also sees means a gain can always be read as the encoder having already
#     met those subjects, devices and activities.
#   * Does the head generalise? Its training corpus should be the labelled one, and nothing about
#     it should have leaked into the encoder's view of the world.
#
# So the two rosters are now disjoint by construction. The encoder pretrains ONLY on label-free
# sources; the head trains ONLY on labelled ones. The practical consequence is that every
# labelled corpus becomes genuinely out-of-sample for the encoder, which is a stronger and much
# more honest transfer claim than the previous arrangement could support.
#
# The historical recipes below keep their names and their contents so past runs stay reproducible.
# What changed is which roster each STAGE reaches for by default.

#: Encoder pretraining. Label-free by construction: every session in these carries
#: ``__unlabeled__``, so no activity annotation can reach the representation.
LABEL_FREE_PRETRAIN_DATASETS = PRETRAIN_SCALE_DATASETS

#: Supervised/episodic stage: the comparator and classification head. Labelled corpora only.
#: Same object as ``EXPANDED_PHASE_A_TRAIN_DATASETS`` (the active roster), under the name that
#: says what it is for. The frozen 18 is ``EXPANDED_18_TRAIN_DATASETS``.
SUPERVISED_HEAD_TRAIN_DATASETS = EXPANDED_PHASE_A_TRAIN_DATASETS


def assert_pretraining_is_label_free(datasets: Sequence[str]) -> None:
    """Raise if a labelled corpus is about to enter encoder pretraining.

    The check is structural rather than a roster lookup: a source is labelled iff it lives in
    ``data/datasets/``. That way a newly added source is judged by where it sits on disk, not by
    whether someone remembered to update a tuple here.

    A source present in NEITHER tree is not labelled, it is unbuilt, and saying otherwise would
    send the reader looking for a leak that does not exist. Unbuilt sources are left to
    ``CorpusIndex``, which already refuses to train on a roster whose grids are missing and
    names them.
    """
    from data.scripts.curate.corpus_roots import LABELLED_ROOT, _is_dataset_dir

    labelled = [name for name in datasets if _is_dataset_dir(LABELLED_ROOT / name)]
    if labelled:
        raise ValueError(
            "label-free pretraining was asked to train on labelled corpora: "
            f"{', '.join(sorted(labelled))}. These are the supervised stage's data; using them "
            "here makes any downstream gain unattributable. Use --corpus expanded if you "
            "deliberately want the historical combined recipe."
        )

# Optional scale sources are fully specified and buildable, but are not silently mixed into either
# named paper recipe. They must be requested explicitly in build_grids and with pretrain.py
# --datasets so an additional data-scaling experiment remains attributable.
OPTIONAL_PHASE_A_DATASETS = (
    "extrasensory",
    "nhanes",
    "hmog",
    # Real rehabilitation data, but only 4.7% of trials reach Phase-A's six-second window and all
    # placements are muscle-belly sensors. Keep it explicitly available as a short/stress study.
    "kneepad",
)

# Established zero-shot datasets plus the three 2026-08 held-out datasets. These are the single
# source of truth used by baseline table construction and by Phase-B's dev/test partition.
ESTABLISHED_EVAL_DATASETS = (
    "motionsense",
    "realworld",
    "shoaib",
    "inclusivehar",
    "usc_had",
    "tnda_har",
    "ut_complex",
)

# Withdrawn from evaluation on 2026-09-11. These three arrived with the movement-monitoring
# clinical pivot, which is archived (tag `archive-pre-classifier-cleanup-20260911`). The live
# question is general human activity recognition, and a clinical-population symptom or
# rehabilitation-exercise corpus does not measure it. Their converters, StreamSpecs and grids are
# left intact so an archived protocol still reproduces; they are simply no longer a live roster.
RETIRED_EVAL_DATASETS = {
    "monipar": "Parkinson's disease motor-symptom monitoring via weekly smartwatch visits "
               "(Frontiers in Neurology 2023). Clinical symptom measurement, not general HAR.",
    "spar": "Shoulder physiotherapy exercise recognition (Physiol. Meas. 2018). Rehabilitation "
            "exercise protocol, not general HAR.",
    "upper_limb_use": "Relative arm use in patients with hemiparesis (J. Rehab. Assist. Technol. "
                      "Eng. 2021). Clinical population and a clinical readout. Separately, only "
                      "72% of its cells can form a k=8 episode (10th percentile k=3), so its "
                      "support curve bent for a sampling reason rather than a modelling one.",
    "tnda_har": "General HAR and wanted, but the UniMTS bundle we convert from ships no "
                "per-sample subject id: all 3,353 windows carry subject=\"unknown\", so no "
                "cross-subject support can ever be drawn and it can only produce k=0 rows. Restore "
                "it by fetching the original release (IEEE DataPort 10.21227/4epb-pg26, sign-in) "
                "and converting with participants.",
}

#: Sealed test roster: every evaluation dataset. Three disjoint source roles (decision reaffirmed
#: 2026-09-11): label-free encoder pretraining, supervised head training, and sealed test. There is
#: no separate development-source roster. No checkpoint, temperature, threshold, or roster choice may read these;
#: they are touched once. Three phone placements (front pocket, waist x2, right pocket, hip) and
#: one watch wrist, 50 and 100 Hz, all gravity-present; inclusivehar includes participants with
#: physical disabilities.
SEALED_TEST_EVAL_DATASETS = (
    "motionsense",
    "realworld",
    "shoaib",
    "inclusivehar",
    "usc_had",
    "ut_complex",
)

PRIMARY_EVAL_DATASETS = SEALED_TEST_EVAL_DATASETS

# Prospective sources are outside the historical sealed-six aggregate. They use an explicit scope
# and are reported per dataset, so adding one cannot silently rewrite an established mean.
PROSPECTIVE_EVAL_DATASETS = (
    "mobiact",
)

# The held-out-configuration transfer probe is EMPTY under the three-role rule: every labelled
# source is either a head-training source or sealed, and a probe run during training may read
# neither (sealed data would be spent; head-training data is the same role the encoder is
# being prepared for, so it is not "transfer"). Offline diagnostics may pass explicit datasets to
# `training.tokenizer.eval_transfer`; nothing runs on a sealed source before the final test.
PHASE_A_TRANSFER_DATASETS: Tuple[str, ...] = ()

EXCLUDED_PRIMARY_DATASETS = {
    "dsads": "torso/limb IMUs do not match the phone-pocket/waist or watch-wrist deployment",
    "harth": "lower-back and thigh accelerometers are retained only as a placement stress test",
    "opportunity": "back and upper/lower-arm IMUs are appendix-only, not phone/watch inputs",
    "recgym": "per-axis min-max normalization destroyed physical scale and gravity",
}


@dataclass(frozen=True)
class StreamSpec:
    dataset: str
    stream_id: str
    device_profile: str
    placement: str
    required: Mapping[str, Tuple[str, ...]]
    optional: Mapping[str, Tuple[str, ...]]
    gravity_state: str
    role: str = "primary"
    session_contains: Tuple[str, ...] = ()
    session_excludes: Tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class CuratedMetadata:
    dataset: str
    stream_id: str
    device_profile: str
    placement: str
    gravity_state: str
    channels: Tuple[str, ...]
    source_channels: Mapping[str, Tuple[str, ...]]
    note: str


@dataclass(frozen=True)
class MultiDeviceCell:
    """One declared simultaneous-placement evaluation cell, in fixed device order."""

    dataset: str
    cell_id: str
    stream_ids: Tuple[str, ...]


def _xyz(prefix: str) -> Dict[str, Tuple[str, ...]]:
    return {f"acc_{axis}": (f"{prefix}{axis}",) for axis in "xyz"}


def _gyro(prefix: str) -> Dict[str, Tuple[str, ...]]:
    return {f"gyro_{axis}": (f"{prefix}{axis}",) for axis in "xyz"}


def _total_acc(acc_prefix: str, gravity_prefix: str) -> Dict[str, Tuple[str, ...]]:
    return {
        f"acc_{axis}": (f"{acc_prefix}{axis}", f"{gravity_prefix}{axis}")
        for axis in "xyz"
    }


_GENERIC_ACC = _xyz("acc_")
_GENERIC_GYRO = _gyro("gyro_")


def _multi_placement(
    dataset: str,
    placements: Mapping[str, Tuple[str, str]],
    *,
    role: str = "primary",
    gravity_state: str = "present",
    session_contains: Tuple[str, ...] = (),
    note: str = "",
) -> Tuple[StreamSpec, ...]:
    """Streams for a converter that writes every placement into ONE frame with a column prefix.

    `placements` maps the column prefix to `(device_profile, placement_text)`. Sharing one frame is
    what makes the placements simultaneous downstream: `build_grids` gives every stream of a session
    the same event id, so window ordinal *i* of one placement is the same instant as ordinal *i* of
    the others.
    """
    return tuple(
        StreamSpec(dataset, prefix, profile, placement,
                   _xyz(f"{prefix}_acc_"), _gyro(f"{prefix}_gyro_"), gravity_state,
                   role=role, session_contains=session_contains, note=note)
        for prefix, (profile, placement) in placements.items()
    )


STREAM_SPECS: Tuple[StreamSpec, ...] = (
    # Training datasets.
    StreamSpec("uci_har", "phone_waist", "phone", "waist",
               _xyz("total_acc_"), _gyro("body_gyro_"), "present"),
    StreamSpec("hhar", "phone_waist", "phone", "waist",
               _GENERIC_ACC, _GENERIC_GYRO, "present", session_contains=("hhar_imu_",)),
    StreamSpec("hhar", "phone_waist_accel_only", "phone", "waist",
               _GENERIC_ACC, {}, "present", session_contains=("hhar_accel_only_",),
               note="Samsung Galaxy S+ phone; the public HHAR release contains no gyro rows."),
    StreamSpec("pamap2", "watch_wrist", "device", "the dominant wrist",
               _xyz("hand_acc16_"), _gyro("hand_gyro_"), "present",
               note="Colibri wireless IMU on the dominant wrist; uses the +/-16g range. Chest, ankle, "
                    "6g, mag, temperature, HR, and invalid orientation are pruned."),
    StreamSpec("wisdm", "phone_pocket", "phone", "the right trouser pocket",
               _xyz("phone_accel_"), _gyro("phone_gyro_"), "present",
               session_contains=("phone_",), session_excludes=("_gyro_",),
               note="Phone in the right trouser pocket; mixed raw clocks are anti-aliased to 20 Hz."),
    StreamSpec("wisdm", "watch_wrist", "watch", "the dominant wrist",
               _xyz("watch_accel_"), _gyro("watch_gyro_"), "present",
               session_contains=("watch_",), session_excludes=("_gyro_",),
               note="Smartwatch on the dominant wrist; mixed raw clocks are anti-aliased to 20 Hz."),
    StreamSpec("sp_sw_har", "phone_pocket", "phone", "left front trouser pocket",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               session_contains=("_sp_",),
               note="Paired phone+watch TUG capture; smartphone (orientation-variable pocket)."),
    StreamSpec("sp_sw_har", "watch_wrist", "watch", "left wrist",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               session_contains=("_sw_",),
               note="Paired phone+watch TUG capture; smartwatch on the left wrist."),
    StreamSpec("nfi_fared", "back", "device", "the lower back",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               session_contains=("_back_",),
               note="NFI-FARED lower-back strapped IMU (rare placement); gyro deg/s->rad/s in convert."),
    StreamSpec("harmes", "watch_wrist", "watch", "the right wrist",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               note="WearOS smartwatch, 15 fine-grained kitchen/bathroom hand ADLs. Acc m/s^2 "
                    "(gravity present); gyro rad/s. Left-wrist Puck.js excluded (unrecoverable gyro)."),
    # XRF V2 (WWADL) 'Plus' release: five-position body IMU + AirPods ear IMU; 30 indoor ADLs,
    # 16 volunteers (upgraded from the old 3-subject WWADL_open subset).
    # Device->placement is read from the h5's OWN `device_order` field at convert time
    # (self-describing, so the Plus-vs-WWADL_open ordering difference cannot bite). Acc g, gyro rad/s.
    StreamSpec("xrf_v2", "glasses", "device", "the head (smart glasses)",
               _GENERIC_ACC, _GENERIC_GYRO, "present", session_contains=("_glasses_",),
               note="Head-worn smart-glasses IMU — a placement absent elsewhere in the corpus. "
                    "Placement 'the head (smart glasses)' keeps 'head' in the per-channel text AND "
                    "renders the factored sensor text as 'on the head (smart glasses)' — a single "
                    "'on', not the old 'on smart glasses on the head' double."),
    StreamSpec("xrf_v2", "left_wrist", "device", "the left wrist",
               _GENERIC_ACC, _GENERIC_GYRO, "present", session_contains=("_left_wrist_",)),
    StreamSpec("xrf_v2", "right_wrist", "device", "the right wrist",
               _GENERIC_ACC, _GENERIC_GYRO, "present", session_contains=("_right_wrist_",)),
    StreamSpec("xrf_v2", "left_pocket", "device", "the left trouser pocket",
               _GENERIC_ACC, _GENERIC_GYRO, "present", session_contains=("_left_pocket_",)),
    StreamSpec("xrf_v2", "right_pocket", "device", "the right trouser pocket",
               _GENERIC_ACC, _GENERIC_GYRO, "present", session_contains=("_right_pocket_",)),
    StreamSpec("xrf_v2", "airpods_ear", "device", "an earbud in the ear",
               _GENERIC_ACC, _GENERIC_GYRO, "removed", session_contains=("_airpods_ear_",),
               note="AirPods Pro ear IMU @25 Hz (Plus release): user acceleration (gravity REMOVED) "
                    "+ gyro rad/s. Ear placement absent elsewhere in the corpus."),
    StreamSpec("nfi_fared", "wrist", "device", "the dominant forearm",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               session_contains=("_arm_",),
               note="NFI-FARED dominant-FOREARM strapped IMU (per the NFI/Hi-OSCAR paper; not the wrist); gyro deg/s->rad/s in convert."),
    StreamSpec("kuhar", "phone_waist", "phone", "waist",
               _GENERIC_ACC, _GENERIC_GYRO, "removed"),
    StreamSpec("unimib_shar", "phone_pocket", "phone", "trouser pocket",
               _GENERIC_ACC, {}, "present"),
    StreamSpec("hapt", "phone_waist", "phone", "waist",
               _GENERIC_ACC, _GENERIC_GYRO, "present"),
    StreamSpec("mhealth", "watch_wrist", "device", "right wrist",
               _xyz("arm_acc_"), _gyro("arm_gyro_"), "present",
               note="Shimmer2 research IMU on the right lower arm: co-located acc + gyro (6-ch). "
                    "mHealth's gyro is somewhat "
                    "sample-and-hold but real, so it is kept; chest, ankle, ECG, and magnetometer are pruned."),
    StreamSpec("capture24", "watch_wrist", "watch", "dominant wrist",
               _GENERIC_ACC, {}, "present"),

    # Historical optional encoder-pretraining sources. ExtraSensory has per-example phone placement
    # labels; the converter prunes unknown/bag/table examples before assigning one of these
    # streams. The active source-isolated label-free views are declared immediately below.
    StreamSpec("extrasensory", "phone_pocket", "phone", "a trouser pocket",
               _GENERIC_ACC, {}, "present", role="phase_a_scale",
               session_contains=("_phone_pocket_",),
               note="Free-living personal-phone raw acceleration; the authors' subject/platform "
                    "split controls unit conversion and real clocks control resampling to 50 Hz. "
                    "Explicit pocket label required."),
    StreamSpec("extrasensory", "phone_hand", "phone", "the hand",
               _GENERIC_ACC, {}, "present", role="phase_a_scale",
               session_contains=("_phone_hand_",),
               note="Free-living personal-phone raw acceleration; explicit in-hand label required."),
    StreamSpec("extrasensory", "watch_wrist", "watch", "the wrist",
               _GENERIC_ACC, {}, "present", role="phase_a_scale",
               session_contains=("_watch_wrist_",),
               note="Pebble watch acceleration at a 25 Hz acquisition clock, stored at 50 Hz; "
                    "accelerometer only."),
    StreamSpec("nhanes", "watch_wrist", "watch", "the non-dominant wrist",
               _GENERIC_ACC, {}, "present", role="phase_a_scale",
               session_contains=("_watch_wrist",),
               note="NHANES PAX80_G ActiGraph acceleration at 80 Hz; unlabeled pretraining-only "
                    "bounded subset with released QC intervals applied."),
    StreamSpec("capture24_pretrain", "watch_wrist", "watch", "the dominant wrist",
               _GENERIC_ACC, {}, "present", role="phase_a_scale",
               session_contains=("_watch_wrist",),
               note="Label-free continuous view of Capture-24 Axivity AX3 acceleration at 100 Hz "
                    "in g. Activity annotations are not read; only real timestamp gaps split a "
                    "participant recording."),
    StreamSpec("extrasensory_pretrain", "phone_pocket", "phone", "a trouser pocket",
               _GENERIC_ACC, {}, "present", role="phase_a_scale",
               session_contains=("_phone_pocket",),
               note="Label-free view of ExtraSensory raw phone captures. Android m/s^2 and "
                    "iPhone g are normalized to g; placement annotations are metadata only."),
    StreamSpec("extrasensory_pretrain", "phone_hand", "phone", "the hand",
               _GENERIC_ACC, {}, "present", role="phase_a_scale",
               session_contains=("_phone_hand",),
               note="Label-free view of ExtraSensory raw phone captures with an explicit "
                    "in-hand placement annotation."),
    StreamSpec("extrasensory_pretrain", "watch_wrist", "watch", "the wrist",
               _GENERIC_ACC, {}, "present", role="phase_a_scale",
               session_contains=("_watch_wrist",),
               note="Label-free Pebble watch acceleration, acquired at approximately 25 Hz "
                    "and stored at 50 Hz in g."),
    # Label-free pretraining scale sources (data/pretraining/). Every session carries
    # __unlabeled__, so these never reach a label vocabulary, a validation probe, or the
    # support-classifier episodes; role="phase_a_scale" keeps them out of every default build.
    #
    # Nymeria records one motion with three device families at once, hardware-synchronised.
    # That simultaneity is the point: it is the only public source where the same movement is
    # observed from a body suit, glasses, and wristbands together, which is what a
    # cross-configuration claim needs. Keep every head, torso, and bilateral limb tracker;
    # hands, feet, and other non-target suit trackers remain excluded.
    *(
        StreamSpec("nymeria_xsens", f"xsens_{token}", "device", placement,
                   _GENERIC_ACC, _GENERIC_GYRO, "present", role="phase_a_scale",
                   session_contains=(f"_xsens_{token}",),
                   note="Xsens MVN Link tracker at 240 Hz. Device-frame acceleration in g "
                        "with gravity restored and angular velocity in rad/s, both derived "
                        "from the suit's sensor-level fields.")
        for token, placement in (
            ("head", "the head"),
            ("sternum", "the sternum"),
            ("pelvis", "the pelvis"),
            ("lforearm", "the left forearm"),
            ("rforearm", "the right forearm"),
            ("lupperarm", "the left upper arm"),
            ("rupperarm", "the right upper arm"),
            ("lthigh", "the left thigh"),
            ("rthigh", "the right thigh"),
            ("lshank", "the left shank"),
            ("rshank", "the right shank"),
        )
    ),
    *(
        StreamSpec("nymeria_aria", f"aria_{token}", "device", placement,
                   _GENERIC_ACC, _GENERIC_GYRO, "present", role="phase_a_scale",
                   session_contains=(f"_aria_{token}",),
                   note="Project Aria IMU decimated to 200 Hz from a native 800 Hz/1 kHz "
                        "clock; the native rate travels with the window as source_rate_hz.")
        for token, placement in (
            ("head", "the head (smart glasses)"),
            ("lwrist", "the left wrist"),
            ("rwrist", "the right wrist"),
        )
    ),
    StreamSpec("ego_exo4d", "aria_head", "device", "the head (smart glasses)",
               _GENERIC_ACC, _GENERIC_GYRO, "present", role="phase_a_scale",
               session_contains=("_aria_head",),
               note="Project Aria head IMU across 740+ participants, decimated to 200 Hz "
                    "from a native 800 Hz/1 kHz clock. Video is never downloaded."),
    # Virtual sensors synthesised from motion capture. The `virt_` token is deliberate: a
    # simulated stream must never be mistaken for a measurement anywhere downstream, and it
    # makes every synthetic row greppable in one pass.
    *(
        StreamSpec("synthetic_imu", f"virt_{token}", "device", placement,
                   _GENERIC_ACC, _GENERIC_GYRO, "present", role="phase_a_scale",
                   session_contains=(f"_virt_{token}",),
                   note="SYNTHETIC: finite-differenced from SMPL body motion at 60 Hz, not "
                        "measured. Diversity source only; mocap pretraining alone gives "
                        "marginal gains from the sim-to-real gap (arXiv 2602.11064).")
        for token, placement in (
            ("head", "the head"),
            ("sternum", "the sternum"),
            ("pelvis", "the pelvis"),
            ("lforearm", "the left forearm"),
            ("rforearm", "the right forearm"),
            ("rupperarm", "the right upper arm"),
            ("rthigh", "the right thigh"),
            ("rshank", "the right shank"),
        )
    ),

    StreamSpec("hmog", "phone_hand", "phone", "the hand",
               _GENERIC_ACC, _GENERIC_GYRO, "present", role="phase_a_scale",
               session_contains=("_phone_hand",),
               note="Samsung Galaxy S4 held during reading, writing, or map navigation while "
                    "sitting/walking. Native 100 Hz acc m/s^2 + gyro rad/s; converter synchronizes "
                    "the two event clocks and splits source gaps."),

    # Primary evaluation datasets.
    StreamSpec("motionsense", "phone_front_pocket", "phone", "front pocket",
               _total_acc("acc_", "gravity_"), _GENERIC_GYRO, "present",
               note="Total acceleration is reconstructed from iOS userAcceleration + gravity; attitude is QA-only."),
    StreamSpec("realworld", "phone_waist", "phone", "waist",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               note="Gyro is retained only when the converted waist stream actually contains a complete finite triad."),
    StreamSpec("realworld", "phone_forearm", "phone", "the forearm",
               _xyz("forearm_acc_"), _gyro("forearm_gyro_"), "present"),
    StreamSpec("realworld", "phone_thigh", "phone", "the thigh",
               _xyz("thigh_acc_"), _gyro("thigh_gyro_"), "present"),
    StreamSpec("mobiact", "phone_trouser_pocket", "phone", "trouser pocket",
               _GENERIC_ACC, _GENERIC_GYRO, "present"),
    StreamSpec("shoaib", "phone_right_pocket", "phone", "right trouser pocket",
               _xyz("right_pocket_acc_"), _gyro("right_pocket_gyro_"), "present"),
    StreamSpec("inclusivehar", "phone_waist", "phone", "waist",
               _total_acc("acc_", "gravity_"), _GENERIC_GYRO, "present",
               note="Total acceleration is reconstructed from iOS userAcceleration + gravity; attitude is QA-only."),
    StreamSpec("usc_had", "phone_hip", "phone", "front-right hip",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               note="MotionNode IMU on the hip; accelerometer native in g, gyroscope converted "
                    "from deg/s to rad/s in the converter. UniMTS eval suite."),
    StreamSpec("tnda_har", "watch_wrist", "watch", "right wrist",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               note="Right-wrist IMU from the UniMTS TNDA-HAR bundle (cols 12:18); accel m/s^2 (gravity present), gyro rad/s."),
    # UT-Complex is a phone strapped to the wrist, not a smartwatch.  Keep the historical stream
    # slug so existing converted artifacts remain addressable, but keep it out of genuine-watch
    # compatibility pools.
    StreamSpec("ut_complex", "watch_wrist", "watch_proxy", "the right wrist",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               note="Wrist-mounted phone, retained as a smartwatch-placement proxy only; complex hand-gesture activities. Accel m/s^2 (gravity present)."),

    # SPAR — consumer Apple Watch, 7 shoulder physiotherapy exercises, 20 subjects x both
    # shoulders. A rehabilitation-framing EVALUATION source: its concepts are absent from the
    # training vocabulary, so they score as genuinely unseen. Left and right are performed
    # SEQUENTIALLY, so the two streams are a cross-placement axis, not a synchronous pair.
    StreamSpec("spar", "watch_left_wrist", "watch", "the left wrist",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               session_contains=("_left_wrist",),
               note="Apple Watch 2/3, 50 Hz, accel already g (gravity present) and gyro already "
                    "rad/s -- no unit conversion. One session = one continuous 20-repetition bout "
                    "(median 42 s), so enrollment from this source is within-session; see "
                    "docs/data/APPLICATION_DATASETS.md."),
    StreamSpec("spar", "watch_right_wrist", "watch", "the right wrist",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               session_contains=("_right_wrist",),
               note="Right-shoulder counterpart of spar/watch_left_wrist; same device and units."),

    # MONIPAR — weekly at-home Parkinson's monitoring on a consumer smartwatch. The corpus's only
    # verified ACROSS-SESSION enrollment source: one converted session is one weekly visit, so two
    # sessions of the same subject and exercise are a week apart rather than seconds apart.
    # Accelerometer only.
    # Placement text is "the wrist", not "the more-affected wrist". The cohort-specific wording was
    # false for a third of this stream: subjects hc01-hc07 are healthy controls with no affected
    # side, and they contribute 4,069 of 12,079 windows (33.7%). HALO conditions on this text, so a
    # description that only fits the patient cohort mis-describes every control window. The clinical
    # detail lives in the note below, where nothing conditions on it. Splitting monipar into
    # cohort-specific streams would be the alternative, but the two cohorts wear the same device on
    # the same limb — the difference is pathology, not acquisition configuration.
    StreamSpec("monipar", "watch_wrist", "watch", "the wrist",
               _GENERIC_ACC, {}, "present",
               note="TicWatch S2 (Mobvoi) consumer smartwatch, accelerometer only (3-channel), "
                    "m/s^2 with gravity. Per the paper patients wore the watch on the wrist with "
                    "the greatest presence of motor symptoms, as judged by the attending "
                    "physician, while healthy controls (hc01-hc07) wore it on the dominant hand; "
                    "the stream therefore carries no single affected-side semantics. "
                    "Source clock is 49.9-52.9 Hz depending on subgroup and is resampled to a true "
                    "50 Hz in the converter, because the tremor bands this dataset exists to "
                    "measure would otherwise sit 5.7% off."),

    # Shoaib simultaneous placement views. The wrist stream is explicitly a phone-on-wrist proxy,
    # but all four streams are retained so the declared multi-device cell can be evaluated.
    StreamSpec("shoaib", "phone_left_pocket", "phone", "left trouser pocket",
               _xyz("left_pocket_acc_"), _gyro("left_pocket_gyro_"), "present"),
    StreamSpec("shoaib", "phone_belt", "phone", "belt/holster",
               _xyz("belt_acc_"), _gyro("belt_gyro_"), "present"),
    StreamSpec("shoaib", "watch_wrist_proxy", "watch_proxy", "right wrist",
               _xyz("wrist_acc_"), _gyro("wrist_gyro_"), "present",
               note="A wrist-mounted smartphone is a placement proxy, not a true smartwatch."),
    StreamSpec("harth", "stress_lower_back", "non_deployment", "lower back",
               _xyz("back_acc_"), {}, "present", role="stress"),
    StreamSpec("harth", "stress_thigh", "non_deployment", "thigh",
               _xyz("thigh_acc_"), {}, "present", role="stress"),
)


MULTI_DEVICE_EVAL_CELLS: Tuple[MultiDeviceCell, ...] = (
    MultiDeviceCell(
        "realworld", "phone_forearm+phone_thigh+phone_waist",
        ("phone_forearm", "phone_thigh", "phone_waist"),
    ),
    MultiDeviceCell(
        "shoaib", "phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt",
        ("phone_left_pocket", "phone_right_pocket", "watch_wrist_proxy", "phone_belt"),
    ),
    # Scenario-only: the publication split, not local workout ids, establishes the held-out
    # participant boundary before these simultaneous streams are sampled.
    MultiDeviceCell(
        "mmfit", "left_wrist+right_wrist+right_pocket+left_ear",
        ("left_wrist", "right_wrist", "right_pocket", "left_ear"),
    ),
)


# --- Multi-placement rehabilitation / displacement sources (2026-08) ------------------------------
# Each of these converters writes every placement of one recording into a single frame with a column
# prefix, so the streams below are simultaneous by construction. As of 2026-08-12, six of the ten
# (see DIRECT_CONVERTED_TRAIN_DATASETS) train — their placement diversity is the point, and
# opportunity/realdisp/mmfit populate the resolvability paired contrast, which measures on training
# subjects. They are NOT in PRIMARY_TRAIN_DATASETS (they skip the curate_frame path). monipar, spar
# and upper_limb_use remain held-out eval-only (monipar is the sole across-session enrollment testbed).

STREAM_SPECS += _multi_placement("realdisp", {
    # Manual Table 4 order. The ideal / self / mutual placement regime is in the session id.
    "rla": ("device", "the right forearm"),
    "rua": ("device", "the right upper arm"),
    "back": ("device", "the back"),
    "lua": ("device", "the left upper arm"),
    "lla": ("device", "the left forearm"),
    "rc": ("device", "the right calf"),
    "rt": ("device", "the right thigh"),
    "lt": ("device", "the left thigh"),
    "lc": ("device", "the left calf"),
}, note="Xsens MTx, 50 Hz, m/s^2 with gravity, gyro rad/s. The same subject and exercise recur "
        "under instructor-placed (ideal), subject-placed (self) and deliberately displaced "
        "(mutual4..7) regimes — a real change-of-configuration pair, not a synthetic rotation.")

STREAM_SPECS += _multi_placement("forth_trace", {
    "left_wrist": ("device", "the left wrist"),
    "right_wrist": ("device", "the right wrist"),
    "torso": ("device", "the torso"),
    "right_thigh": ("device", "the right thigh"),
    "left_ankle": ("device", "the left ankle"),
}, note="Shimmer nodes at 51.2 Hz; m/s^2 with gravity, gyro converted deg/s -> rad/s in the "
        "converter. Carries a simultaneous bilateral wrist pair and 9 explicitly labelled postural "
        "transitions. Participant part4 is excluded upstream because its five annotation tracks "
        "describe different takes; part8 is retained after its 1,828 s clock gap is split.")

# Anatomy from Barshan & Yuksek section 3, NOT from UCI's block names: the units sit on the chest,
# both wrists and the outer sides of both knees. UCI calls the blocks T/RA/LA/RL/LL, which would
# have put a wrist sensor into the placement text as "the right arm".
STREAM_SPECS += _multi_placement("dsads", {
    "chest": ("device", "the chest"),
    "right_wrist": ("device", "the right wrist"),
    "left_wrist": ("device", "the left wrist"),
    "right_knee": ("device", "the outer side of the right knee"),
    "left_knee": ("device", "the outer side of the left knee"),
}, role="stress",
   note="Xsens MTx at 25 Hz; m/s^2 with gravity, gyro rad/s. role='stress' because dsads is in "
        "EXCLUDED_PRIMARY_DATASETS on placement grounds -- torso and limb units are not a "
        "phone-pocket or watch-wrist deployment. Build it with `build_grids --dataset dsads`, which "
        "selects every role.")

STREAM_SPECS += _multi_placement("opportunity", {
    "back": ("device", "the back"),
    "right_upper_arm": ("device", "the right upper arm"),
    "right_lower_arm": ("device", "the right forearm"),
    "left_upper_arm": ("device", "the left upper arm"),
    "left_lower_arm": ("device", "the left forearm"),
}, role="stress",
   note="Xsens units at 30 Hz. The converter rescales the release's milli-g accelerometer and "
        "milli-rad/s gyroscope to g and rad/s. Labels are the Locomotion track; the mid-level "
        "gesture track is unsupported at a 6 s window (measured median instance 2.6 s). "
        "role='stress' because opportunity is in EXCLUDED_PRIMARY_DATASETS on placement grounds.")

STREAM_SPECS += _multi_placement("phytmo", {
    "right_arm": ("device", "the right upper arm"),
    "left_arm": ("device", "the left upper arm"),
    "right_forearm": ("device", "the right forearm"),
    "left_forearm": ("device", "the left forearm"),
}, session_contains=("_upper_",),
   note="Upper-limb trials: four magneto-inertial units at 100 Hz, g with gravity, gyro converted "
        "deg/s -> rad/s. Labels distinguish correct from deliberately incorrect execution.")

STREAM_SPECS += _multi_placement("phytmo", {
    "right_thigh": ("device", "the right thigh"),
    "left_thigh": ("device", "the left thigh"),
    "right_shin": ("device", "the right shin"),
    "left_shin": ("device", "the left shin"),
}, session_contains=("_lower_",),
   note="Lower-limb trials: the anterior surface of both thighs and both shins. Same device, rate "
        "and units as the upper-limb set; a recording carries one set or the other, never both.")

STREAM_SPECS += _multi_placement("mmfit", {
    "left_wrist": ("watch", "the left wrist"),
    "right_wrist": ("watch", "the right wrist"),
    "right_pocket": ("phone", "the right trouser pocket"),
    "left_ear": ("device", "an earbud in the left ear"),
}, note="The corpus's cleanest cross-configuration source: one repetition captured on two "
        "smartwatches, a pocketed phone and an earbud at once. The four devices log at 85-212 Hz on "
        "a shared wall clock (measured skew 0.06 s) and are resampled onto one 100 Hz grid.")

# KneE-PAD: muscle-belly sensors on the thigh and calf. Real knee-pathology patients performing
# correct and clinically-defined incorrect exercise variants, but the placement is outside the
# phone/watch deployment envelope AND only 4.7% of trials reach a 6 s window. It is an explicit
# opt-in short/stress study rather than part of the default Phase-A corpus.
STREAM_SPECS += _multi_placement("kneepad", {
    "right_rectus_femoris": ("device", "the right rectus femoris"),
    "right_hamstrings": ("device", "the right hamstrings"),
    "right_tibialis_anterior": ("device", "the right tibialis anterior"),
    "right_gastrocnemius": ("device", "the right gastrocnemius"),
    "left_rectus_femoris": ("device", "the left rectus femoris"),
    "left_hamstrings": ("device", "the left hamstrings"),
    "left_tibialis_anterior": ("device", "the left tibialis anterior"),
    "left_gastrocnemius": ("device", "the left gastrocnemius"),
}, role="stress",
   note="Delsys Trigno Avanti at 148.15 Hz, g with gravity, gyro converted deg/s -> rad/s. "
        "device_profile is 'device', not 'non_deployment': deployment_streams() drops "
        "non_deployment entirely, so those streams can never be gridded at all (which is why "
        "harth's stress streams have no native grid). role='stress' is what keeps kneepad out of "
        "every primary query, and it is the right knob for that.")

# upper-limb-use: one wrist band per arm, so a session carries plain acc_/gyro_ columns and the arm
# is routed from the session id. Controls are described by anatomical side; the released patient
# CSVs do not record which side is affected, so those two are described by impairment.
STREAM_SPECS += tuple(
    StreamSpec("upper_limb_use", stream_id, "device", placement,
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               session_contains=contains, session_excludes=excludes,
               note="Custom wrist-worn IMU at 50 Hz, g with gravity, gyro rad/s. Labels are 15 "
                    "functional ADLs annotated from video by therapists.")
    for stream_id, placement, contains, excludes in (
        ("control_left_wrist", "the left wrist", ("_left_wrist_",), ()),
        ("control_right_wrist", "the right wrist", ("_right_wrist_",), ()),
        # "_affected_wrist_" is a substring of "_unaffected_wrist_", so the affected stream has to
        # exclude it explicitly or it would swallow both arms of every patient.
        ("patient_affected_wrist", "the wrist of the more-affected arm",
         ("_affected_wrist_",), ("_unaffected_wrist_",)),
        ("patient_unaffected_wrist", "the wrist of the less-affected arm",
         ("_unaffected_wrist_",), ()),
    )
)


_BY_DATASET: Dict[str, Tuple[StreamSpec, ...]] = {}
for _dataset in {spec.dataset for spec in STREAM_SPECS}:
    _BY_DATASET[_dataset] = tuple(spec for spec in STREAM_SPECS if spec.dataset == _dataset)


def stream_specs(dataset: str, role: Optional[str] = "primary") -> Tuple[StreamSpec, ...]:
    specs = _BY_DATASET.get(dataset, ())
    return specs if role is None else tuple(spec for spec in specs if spec.role == role)


def deployment_streams(
    placement_strict: bool = False,
    role: Optional[str] = "primary",
) -> Tuple[StreamSpec, ...]:
    """Every device stream in the corpus, for the harmonised/deployment build.

    - ``placement_strict=True``  → **phone streams only** ("harmonised-strict"). Watch-placement
      datasets (pamap2, mhealth, capture24) and wisdm's watch stream are dropped, because mixing
      phone and watch placement is a problem for placement-blind models.
    - ``placement_strict=False`` → **all wearables** ("harmonised"): phone + watch + body-strapped
      ``device`` IMUs (e.g. nfi_fared back/wrist). A session recorded on multiple devices contributes
      one separate single-device sample each.

    ``watch_proxy`` / ``non_deployment`` streams are excluded from compatibility pooling. A proxy
    can still be named in the sealed roster and materialized explicitly for its disclosed
    placement-proxy evaluation cell.
    """
    keep = {"phone"} if placement_strict else {"phone", "watch", "device"}
    return tuple(
        s for s in STREAM_SPECS
        if (role is None or s.role == role) and s.device_profile in keep
    )


def get_stream_spec(dataset: str, stream_id: str) -> StreamSpec:
    for spec in _BY_DATASET.get(dataset, ()):
        if spec.stream_id == stream_id:
            return spec
    raise KeyError(f"No deployment stream {dataset}/{stream_id}")


def session_stream_specs(
    dataset: str,
    session_id: str,
    role: str = "primary",
) -> Tuple[StreamSpec, ...]:
    """Return policy streams that can be represented by a converted session id."""
    matches = []
    for spec in stream_specs(dataset, role):
        if spec.session_contains and not all(token in session_id for token in spec.session_contains):
            continue
        if any(token in session_id for token in spec.session_excludes):
            continue
        matches.append(spec)
    return tuple(matches)


def _sources_available(frame: pd.DataFrame, sources: Sequence[str]) -> bool:
    if not all(source in frame.columns for source in sources):
        return False
    return all(np.isfinite(pd.to_numeric(frame[source], errors="coerce")).any() for source in sources)


def channel_names_for_frame(frame: pd.DataFrame, spec: StreamSpec) -> Tuple[str, ...]:
    """Return the standardized 3- or 6-channel schema available in ``frame``."""
    missing = [name for name, sources in spec.required.items() if not _sources_available(frame, sources)]
    if missing:
        raise ValueError(
            f"{spec.dataset}/{spec.stream_id}: missing required deployment channels {missing}; "
            f"available columns={list(frame.columns)}"
        )

    names = list(spec.required)
    optional_names = list(spec.optional)
    if optional_names and all(_sources_available(frame, spec.optional[name]) for name in optional_names):
        names.extend(optional_names)
    return tuple(name for name in STANDARD_CHANNEL_ORDER if name in names)


def curate_frame(frame: pd.DataFrame, spec: StreamSpec) -> Tuple[pd.DataFrame, CuratedMetadata]:
    """Select one deployment stream and rename/derive it to the common channel schema.

    A tuple of source columns means sum them elementwise. This is used only for
    iOS total acceleration reconstruction (userAcceleration + gravity).
    """
    channel_names = channel_names_for_frame(frame, spec)
    source_map = {**spec.required, **spec.optional}
    out = pd.DataFrame(index=frame.index)
    if "timestamp_sec" in frame.columns:
        out["timestamp_sec"] = pd.to_numeric(frame["timestamp_sec"], errors="coerce")

    for output_name in channel_names:
        sources = source_map[output_name]
        values = np.zeros(len(frame), dtype=np.float64)
        for source in sources:
            values += pd.to_numeric(frame[source], errors="coerce").to_numpy(dtype=np.float64)
        out[output_name] = values

    if "activity" in frame.columns:
        out["activity"] = frame["activity"].values

    metadata = CuratedMetadata(
        dataset=spec.dataset,
        stream_id=spec.stream_id,
        device_profile=spec.device_profile,
        placement=spec.placement,
        gravity_state=spec.gravity_state,
        channels=channel_names,
        source_channels={name: tuple(source_map[name]) for name in channel_names},
        note=spec.note,
    )
    return out.reset_index(drop=True), metadata


def channel_description(metadata: CuratedMetadata, channel_name: str) -> str:
    modality = "accelerometer" if channel_name.startswith("acc_") else "gyroscope"
    axis = channel_name[-1].upper()
    gravity = ""
    if modality == "accelerometer":
        gravity = "; gravity removed" if metadata.gravity_state == "removed" else "; includes gravity"
    return (
        f"{metadata.device_profile} {modality} {axis}-axis at {metadata.placement}{gravity}"
    )


def all_source_channels(dataset: str, role: str = "primary") -> Tuple[str, ...]:
    channels = {
        source
        for spec in stream_specs(dataset, role)
        for sources in (*spec.required.values(), *spec.optional.values())
        for source in sources
    }
    return tuple(sorted(channels))
