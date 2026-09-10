"""Accelerometer UNIT canonicalization → g.

This module normalizes exactly ONE heterogeneity axis: the accelerometer UNIT, so that a still,
gravity-present window reads |acc| ≈ 1.0 g. (HALO's signed DC/gravity feature needs g — a per-channel
DC of 1.0 means "gravity", which is only true in g.) The gyroscope is NEVER touched.

Pipeline ordering (important — this is why the logic here is so small):

    raw session → deployment_policy.curate_*  →  accel_units.to_g  →  windowing  →  baseline_view

`deployment_policy` runs FIRST and owns GRAVITY: it selects the phone/watch device stream and, for the
iOS sets, reconstructs total acceleration = userAcceleration + gravity (both in g). So by the time a
stream reaches this module, gravity is already handled and each accelerometer is in its dataset's
NATIVE unit — all we do here is a scalar rescale (g stays; m/s² → ÷9.80665).

The per-dataset unit is a fixed property of how each dataset was recorded. The rationale for every
dataset is documented in `docs/DATA_HETEROGENEITY.md`; `tests/test_accel_units.py` asserts that every
dataset in the deployment policy is classified here exactly once, so a new dataset cannot be added
without a documented unit decision.
"""

from __future__ import annotations

import numpy as np

GRAVITY_MS2 = 9.80665

# --- Accelerometer already in g → scale 1.0 --------------------------------------------------------
# Native-g recordings: capture24 + harth (Axivity AX3, reported in g); uci_har (Android, exported as
# `total_acc` in g — NOT the gravity-removed `body_acc`); hapt (Android, ~1.02 g).
# iOS sets (motionsense, inclusivehar): deployment_policy already summed userAcceleration + gravity,
# both in g, so they ARRIVE here in g.
ACC_UNIT_G = frozenset({
    "uci_har", "hapt", "capture24", "harth", "motionsense", "inclusivehar",
    "usc_had",  # MotionNode accelerometer reported in g (USC-HAD Readme); |acc|~1.07 g still.
    "sp_sw_har",  # converter already divides raw m/s² by 9.81 -> g (see convert.py).
    "nfi_fared",  # NFI-FARED accelerometer reported in g (verified at-rest |acc|~1.00).
    "xrf_v2",     # 5-pos IMU acc in g (verified still |acc|~0.9-1.0); AirPods total acc=grav+userAccel (g).
    "extrasensory",  # converter normalizes Android m/s^2, iPhone g (author split), and watch milli-g.
    "nhanes",        # CDC PAX80_G release is calibrated triaxial acceleration in g.
    # Label-free pretraining sources. Every one of these converters emits g at the session
    # contract boundary, so nothing is rescaled here. They are listed anyway because the
    # policy test requires a documented unit decision per dataset, and "the converter already
    # did it" is a decision that should be written down rather than inferred.
    "nymeria_xsens",   # Xsens sensor-frame acceleration is m/s^2 at source; the converter
                       # restores gravity and divides by 9.80665 before writing sessions.
    "nymeria_aria",    # Project Aria reports m/s^2; the converter divides by 9.80665.
    "ego_exo4d",       # Same Aria device family and the same conversion.
    "synthetic_imu",   # SYNTHETIC: finite-differenced from mocap, expressed in g by
                       # construction (see synthetic_imu/synthesis.py).
    "spar",          # Apple Watch 2/3 export already in g, gravity present. Verified over all 280
                     # files: the lowest-gyro decile of each file has median |acc| = 1.026 g. Whole-
                     # file medians run 0.99-2.47 g because these are vigorous arm exercises, so the
                     # quiescent decile is the honest gravity reference (see convert.py).
    "phytmo",        # CSV header states "Accelerometer X (g)"; measured median |acc| 1.008 g.
    "kneepad",       # release metadata.txt Table 2 states "Accelerometer (x,y,z) unit G";
                     # measured median |acc| 1.011 g.
    "upper_limb_use",  # wrist-band export in g; measured median |acc| 1.016 g.
    "opportunity",   # source is milli-g (column_names.txt: round(original / 9.8 * 1000)); the
                     # converter already divides by 1000, so it ARRIVES here in g (1.002 g).
})

# --- Accelerometer in m/s² → scale 1/9.80665 -------------------------------------------------------
# Android / Shimmer recordings. kuhar is here too: it is m/s² but gravity-REMOVED (linear
# acceleration); the unit rescale is identical — gravity STATE is deployment_policy's concern and is
# never fabricated here.
# unimib_shar: the RAW UniMiB .npy release is m/s² (|acc|~9.8, gravity present); the converter reads
# that (not the z-scored Kaggle CSV). Reclassified from G on 2026-07-12 after the debug sweep found the
# CSV was z-score-normalized.
ACC_UNIT_MS2 = frozenset({
    "hhar", "pamap2", "wisdm", "kuhar", "mhealth", "realworld", "mobiact", "shoaib",
    "tnda_har", "ut_complex", "unimib_shar",  # accelerometer in m/s^2 (gravity present) -> rescale to g.
    "harmes",  # WearOS right-wrist acc in m/s^2 (verified at-rest |acc|~9.81) -> rescale to g.
    "hmog",    # Samsung Galaxy S4 accelerometer in m/s^2; sitting-session |acc|~9.81, gravity present.
    "monipar",     # Monipar_README.txt: "Accelerometer x-axis (unit m/s^2)"; measured 9.773.
    "realdisp",    # Xsens MTx; measured median |acc| 9.860 over all 9 units and 46 logs.
    "forth_trace",  # Shimmer nodes; measured median |acc| 9.924. (Gyro deg/s is converted in the
                    # converter -- accel_units never touches the gyroscope.)
    "dsads",       # Xsens MTx; measured median |acc| 9.731.
    "mmfit",       # smartwatch/phone/earbud Android exports; measured median |acc| 9.887.
})


def accel_scale_factor(dataset: str) -> float:
    """Scalar that brings `dataset`'s (already gravity-handled) accelerometer to g."""
    if dataset in ACC_UNIT_G:
        return 1.0
    if dataset in ACC_UNIT_MS2:
        return 1.0 / GRAVITY_MS2
    raise KeyError(
        f"{dataset!r} has no accelerometer-unit classification. Add it to ACC_UNIT_G or ACC_UNIT_MS2 "
        f"in data/scripts/accel_units.py and document why in docs/DATA_HETEROGENEITY.md.")


def to_g(dataset: str, acc: np.ndarray) -> np.ndarray:
    """Return the accelerometer array rescaled to g (pure unit rescale; gravity already handled)."""
    return np.asarray(acc, dtype=np.float32) * accel_scale_factor(dataset)


def is_accel_channel(name: str) -> bool:
    """True iff a channel name is an accelerometer axis (never gyro/mag): `acc_x`, `hand_acc16_x`, ..."""
    return "acc" in name.lower()
