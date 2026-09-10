"""Regression tests for the phone/watch deployment-scoped data view."""

import numpy as np
import pandas as pd
import pytest

from training.tokenizer.pretrain_data import TRAIN_DATASETS


def test_primary_train_datasets_matches_trainer_roster():
    """F11: the deployment-policy rosters and the trainer's TRAIN_DATASETS are ONE manifest — they
    must not drift (hapt was in one and not the other, double-counting 13 vs 12).

    Since 2026-08-12 TRAIN_DATASETS is the union of two disjoint kinds: PRIMARY_TRAIN_DATASETS (raw
    frames curated by `curate_frame`) and DIRECT_CONVERTED_TRAIN_DATASETS (built straight to native
    grids, no curation path). Every training dataset must be declared in exactly one, so a new one
    cannot be added to the trainer without also declaring how its channels are produced."""
    from data.scripts.curate.deployment_policy import (
        CORPUS_MATCHED_TRAIN_DATASETS as _matched,
        DIRECT_CONVERTED_TRAIN_DATASETS as _direct,
        EXPANDED_PHASE_A_TRAIN_DATASETS as _expanded,
        PRIMARY_TRAIN_DATASETS as _ptd,
        RETIRED_TRAIN_DATASETS,
    )
    assert set(_ptd).isdisjoint(_direct), "a dataset cannot be both curated and direct-converted"
    union = set(_ptd) | set(_direct)
    assert union == set(TRAIN_DATASETS), (
        f"roster drift: policy declares {sorted(union - set(TRAIN_DATASETS))} extra, "
        f"missing {sorted(set(TRAIN_DATASETS) - union)}")
    assert len(_ptd) + len(_direct) == len(TRAIN_DATASETS) == 14
    # The matched recipe is FROZEN history (12 sources incl. the retired ones); the active primary
    # roster is that list minus the retirements. They must never be the same object again.
    assert set(_ptd) == set(_matched) - set(RETIRED_TRAIN_DATASETS)
    assert len(_matched) == 12
    assert _expanded == TRAIN_DATASETS

from data.scripts.curate.deployment_policy import (
    EXCLUDED_PRIMARY_DATASETS,
    PRIMARY_EVAL_DATASETS,
    PRIMARY_TRAIN_DATASETS,
    STANDARD_CHANNEL_ORDER,
    all_source_channels,
    channel_description,
    curate_frame,
    get_stream_spec,
    session_stream_specs,
    stream_specs,
)


EXPECTED_PRIMARY_CHANNELS = {
    "uci_har": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "hhar": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "pamap2": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    # WISDM has two primary logical devices; both share the same intended schema.
    "wisdm": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "kuhar": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "unimib_shar": ("acc_x", "acc_y", "acc_z"),
    "hapt": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "mhealth": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "capture24": ("acc_x", "acc_y", "acc_z"),
    "sp_sw_har": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "nfi_fared": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "harmes": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "xrf_v2": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "motionsense": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "realworld": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "mobiact": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "shoaib": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "inclusivehar": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "usc_had": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "tnda_har": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "ut_complex": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "monipar": ("acc_x", "acc_y", "acc_z"),
    "spar": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
    "upper_limb_use": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
}


def _full_source_frame(dataset, spec, n=8):
    columns = {"timestamp_sec": np.arange(n, dtype=float) / 50.0}
    for source in all_source_channels(dataset, role=spec.role):
        columns[source] = np.ones(n, dtype=float)
    return pd.DataFrame(columns)


@pytest.mark.parametrize("dataset", PRIMARY_TRAIN_DATASETS + PRIMARY_EVAL_DATASETS)
def test_primary_streams_prune_to_three_or_six_channels(dataset):
    specs = stream_specs(dataset, role="primary")
    assert specs, f"{dataset} needs at least one primary deployment stream"
    for spec in specs:
        curated, metadata = curate_frame(_full_source_frame(dataset, spec), spec)
        sensor_columns = tuple(c for c in curated.columns if c != "timestamp_sec")
        if dataset == "hhar" and spec.stream_id == "phone_waist_accel_only":
            assert sensor_columns == ("acc_x", "acc_y", "acc_z")
        else:
            assert sensor_columns == EXPECTED_PRIMARY_CHANNELS[dataset]
        assert metadata.channels == sensor_columns
        assert len(sensor_columns) in (3, 6)
        assert sensor_columns == tuple(c for c in STANDARD_CHANNEL_ORDER if c in sensor_columns)
        assert not any("mag" in c or "ecg" in c or "temp" in c for c in curated.columns)


def test_uci_uses_total_acceleration_not_body_acceleration():
    spec = get_stream_spec("uci_har", "phone_waist")
    assert spec.required["acc_x"] == ("total_acc_x",)
    assert "body_acc_x" not in all_source_channels("uci_har")


def test_pamap2_keeps_only_wrist_acc16_and_gyro():
    sources = set(all_source_channels("pamap2"))
    assert sources == {
        "hand_acc16_x", "hand_acc16_y", "hand_acc16_z",
        "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    }


def test_mhealth_keeps_right_wrist_imu_acc_and_gyro():
    sources = set(all_source_channels("mhealth"))
    assert sources == {
        "arm_acc_x", "arm_acc_y", "arm_acc_z", "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",
    }
    assert not any("chest" in s or "ankle" in s or "ecg" in s or "mag" in s for s in sources)


def test_ios_total_acceleration_is_reconstructed_before_pruning():
    spec = get_stream_spec("motionsense", "phone_front_pocket")
    frame = _full_source_frame("motionsense", spec)
    curated, metadata = curate_frame(frame, spec)
    # userAcceleration=1 plus gravity=1 for each synthetic axis.
    assert np.allclose(curated[["acc_x", "acc_y", "acc_z"]].to_numpy(), 2.0)
    assert metadata.gravity_state == "present"
    assert not any(name.startswith("gravity_") or name.startswith("attitude_") for name in curated.columns)


def test_wisdm_legacy_sessions_keep_accel_sessions_and_drop_gyro_only_sessions():
    phone = session_stream_specs("wisdm", "phone_accel_1600_A_0000")
    watch = session_stream_specs("wisdm", "watch_accel_1600_A_0000")
    assert [s.stream_id for s in phone] == ["phone_pocket"]
    assert [s.stream_id for s in watch] == ["watch_wrist"]
    assert session_stream_specs("wisdm", "phone_gyro_1600_A_0000") == ()
    assert session_stream_specs("wisdm", "watch_gyro_1600_A_0000") == ()


def test_wisdm_missing_gyro_remains_accelerometer_only():
    spec = get_stream_spec("wisdm", "phone_pocket")
    frame = pd.DataFrame({
        "timestamp_sec": [0.0, 0.05],
        "phone_accel_x": [1.0, 1.1],
        "phone_accel_y": [2.0, 2.1],
        "phone_accel_z": [3.0, 3.1],
    })
    curated, metadata = curate_frame(frame, spec)
    assert metadata.channels == ("acc_x", "acc_y", "acc_z")
    assert tuple(curated.columns) == ("timestamp_sec", "acc_x", "acc_y", "acc_z")


def test_shoaib_primary_and_diagnostic_placements_are_separate():
    assert [s.stream_id for s in stream_specs("shoaib", "primary")] == ["phone_right_pocket"]
    assert {s.stream_id for s in stream_specs("shoaib", "diagnostic")} == {
        "phone_left_pocket", "phone_belt", "watch_wrist_proxy",
    }
    assert not any("upper_arm" in source for source in all_source_channels("shoaib", "diagnostic"))


def test_non_deployment_datasets_are_not_primary():
    assert "dsads" not in PRIMARY_TRAIN_DATASETS
    assert "harth" not in PRIMARY_EVAL_DATASETS
    assert stream_specs("dsads", "primary") == ()
    assert stream_specs("harth", "primary") == ()
    assert {"dsads", "harth"}.issubset(EXCLUDED_PRIMARY_DATASETS)


def test_new_heldout_datasets_are_default_eval_only():
    heldout = {"monipar", "spar", "upper_limb_use"}
    assert heldout <= set(PRIMARY_EVAL_DATASETS)
    assert heldout.isdisjoint(TRAIN_DATASETS)
    assert all(stream_specs(dataset, "primary") for dataset in heldout)


def test_default_grid_build_covers_expanded_training_roster():
    from data.scripts.build_grids import build_stream_specs
    built = {spec.dataset for spec in build_stream_specs()}
    assert built == set(TRAIN_DATASETS) | set(PRIMARY_EVAL_DATASETS)
    assert {"dsads", "mmfit"} <= built
    assert "opportunity" not in built, "retired 2026-09-10; must leave the default build"
    assert {"extrasensory", "nhanes", "hmog", "kneepad"}.isdisjoint(built)


def test_channel_text_names_device_placement_axis_and_gravity_state():
    spec = get_stream_spec("kuhar", "phone_waist")
    _, metadata = curate_frame(_full_source_frame("kuhar", spec), spec)
    assert channel_description(metadata, "acc_x") == (
        "phone accelerometer X-axis at waist; gravity removed"
    )
    assert channel_description(metadata, "gyro_z") == "phone gyroscope Z-axis at waist"


def test_deployment_streams_non_strict_includes_phone_watch_and_device():
    from data.scripts.curate.deployment_policy import deployment_streams
    streams = deployment_streams(placement_strict=False)
    # non-strict ("harmonised", all-wearable) admits phone + watch + body-strapped `device`
    # IMUs (nfi_fared back/wrist); phone-strict (below) keeps only phone.
    assert {s.device_profile for s in streams} == {"phone", "watch", "device"}
    datasets = {s.dataset for s in streams}
    assert {"pamap2", "mhealth", "capture24"}.issubset(datasets)   # watch datasets present
    assert "nfi_fared" in datasets                                   # body-strapped device present
    assert sum(s.dataset == "wisdm" for s in streams) == 2          # wisdm phone + watch


def test_deployment_streams_strict_is_phone_only():
    from data.scripts.curate.deployment_policy import deployment_streams
    streams = deployment_streams(placement_strict=True)
    assert all(s.device_profile == "phone" for s in streams)
    datasets = {s.dataset for s in streams}
    assert not ({"pamap2", "mhealth", "capture24"} & datasets)      # watch datasets dropped
    assert [s.stream_id for s in streams if s.dataset == "wisdm"] == ["phone_pocket"]


def test_strict_streams_are_a_strict_subset():
    from data.scripts.curate.deployment_policy import deployment_streams
    strict = {(s.dataset, s.stream_id) for s in deployment_streams(placement_strict=True)}
    full = {(s.dataset, s.stream_id) for s in deployment_streams(placement_strict=False)}
    assert strict < full
