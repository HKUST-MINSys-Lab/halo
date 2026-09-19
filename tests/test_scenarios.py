"""Deployment-heterogeneity scenarios 2-7 — unit tests for derived streams and cross enrolment."""

from __future__ import annotations

import numpy as np
import pytest

from baselines.data import EvalStream, source_slice_fingerprint
from training.support_classifier import run_scenarios as runner
from training.support_classifier.scenarios import (
    CrossEnrolment,
    build_cross_manifest,
    derive_accel_only,
    derive_resampled,
    shared_candidates,
)
from training.support_classifier.run_scenarios import (
    ACTIVE_SCENARIOS,
    Task,
    _matched_within_reference,
    _requires_cross_support_features,
    device_set_variants,
)

CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def test_scenario_cli_defaults_to_the_representative_budget(monkeypatch):
    import argparse
    import sys

    class Parsed(Exception):
        pass

    def capture(parser):
        assert parser.get_default("k") == [0, 1, 4, 8, 32]
        assert parser.get_default("window_seconds") == [8.0]
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(sys, "argv", ["halo-scenarios", "--out", "unused"])
    with pytest.raises(Parsed):
        runner.main()


def test_active_scenario_roster_excludes_retired_compound_cold_start():
    assert ACTIVE_SCENARIOS == (
        "s1_partial_coverage",
        "s2_cross_placement",
        "s3_cross_dataset",
        "s4_missing_modality",
        "s5_rate_mismatch",
        "s6_new_domain",
        "s7_device_set",
    )


def test_device_set_variants_rotate_all_devices_and_have_exact_relation_shapes():
    variants = device_set_variants(("forearm", "thigh", "waist"))
    singles = [row for row in variants if row[0] == "support_single_query_full"]
    assert {row[2][0] for row in singles} == {"forearm", "thigh", "waist"}
    assert all(len(query) > len(support) for name, query, support, _ in variants
               if name == "support_single_query_full")
    assert all(len(query) == len(support) == 2 and len(set(query) & set(support)) == 1
               for name, query, support, _ in variants if name == "partial_overlap")
    assert all(not (set(query) & set(support))
               for name, query, support, _ in variants if name == "disjoint_single")


def make_stream(
    *, dataset="toy", stream="site_a", labels=("walking", "running"), n_per_label=6,
    rate_hz=50.0, subjects_per_label=2, executions_per_label=3, seed=0, channels=None,
    window_len=100,
) -> EvalStream:
    rng = np.random.default_rng(seed)
    channels = list(channels or CHANNELS)
    gt, subjects, executions = [], [], []
    for label in labels:
        for index in range(n_per_label):
            gt.append(label)
            subjects.append(f"s{index % subjects_per_label}")
            executions.append(f"{label}-e{index % executions_per_label}")
    n = len(gt)
    return EvalStream(
        dataset=dataset, stream=stream, alignment="native",
        windows=rng.normal(size=(n, window_len, len(channels))).astype(np.float32),
        gt=gt, subjects=np.asarray(subjects, dtype=object), channels=channels,
        rate_hz=rate_hz, mask=np.ones(len(channels), dtype=bool),
        eval_labels=list(labels), window_seconds=window_len / rate_hz,
        event_ids=np.arange(n), execution_ids=np.asarray(executions, dtype=object),
        block_ids=np.asarray(executions, dtype=object), execution_granularity="recording",
    )


def test_zero_support_cross_stream_does_not_require_support_features():
    query = make_stream(stream="query")
    support = make_stream(stream="support", seed=1)
    task = Task(
        scenario="s2_cross_placement", variant="zero_support",
        query_stream=query, support_stream=support, plans=(),
        candidates=tuple(query.eval_labels), offset=query.n_windows,
        severity={"L": 0, "S": 0, "P": 1, "C": 1},
    )

    assert not _requires_cross_support_features(task, 0)
    assert _requires_cross_support_features(task, 1)


@pytest.mark.parametrize("coverage", [None, "partial"])
def test_neighbor_encoder_zero_support_uses_training_bank_bridge(monkeypatch, tmp_path, coverage):
    import torch
    from training.support_classifier import run_scenarios as runner
    from training.support_classifier.partial_coverage import CoverageCell
    from training.support_classifier.sealed_eval import build_manifest

    stream = make_stream()
    task = Task(
        scenario="s2_cross_placement", variant="zero_support",
        query_stream=stream, support_stream=stream,
        plans=tuple(build_manifest(stream, 0)), candidates=tuple(stream.eval_labels),
        offset=0, severity={"L": 0, "S": 0, "P": 0, "C": 0},
        coverage=(None if coverage is None else CoverageCell(
            supported=("walking",), hidden=("running",), coverage=0.5, requested_coverage=0.5,
        )),
    )
    features = np.ones((stream.n_windows, 4), dtype=np.float32)
    monkeypatch.setattr(runner, "_load_or_encode", lambda **kwargs: (features, "features"))
    monkeypatch.setattr(runner, "_parameter_count_m", lambda *args, **kwargs: 0.8)
    monkeypatch.setattr(runner, "conse_scores", lambda *args: np.tile([1., 0.], (len(features), 1)))
    banks = {"halo": (features, np.zeros(len(features), dtype=int), ["walking"], "bank")}
    rows = runner.score_task(
        task, models=["halo"], device=torch.device("cpu"), cache_dir=tmp_path,
        halo_checkpoint=tmp_path / "neighbors.pt", halo_has_classifier=False,
        banks=banks, k=0, window_seconds=2., bootstrap=0,
    )
    assert not any(row["status"] in {"failed", "inapplicable"} for row in rows)
    assert any(row["readout"] == "zero-shot-native-or-bridge" and row["status"] == "ok"
               for row in rows)


# ------------------------------------------------------- Scenario 4: modality drop


def test_accel_only_zeroes_and_masks_the_gyroscope():
    stream = make_stream()
    derived = derive_accel_only(stream)
    gyro = [index for index, name in enumerate(CHANNELS) if name.startswith("gyro")]
    assert np.all(derived.windows[:, :, gyro] == 0.0)
    assert not derived.mask[gyro].any(), "an absent sensor must be masked, not merely zeroed"
    assert derived.mask[:3].all()


def test_accel_only_does_not_mutate_the_original_stream():
    stream = make_stream()
    before = stream.windows.copy()
    derive_accel_only(stream)
    assert np.array_equal(stream.windows, before)
    assert stream.mask.all()


def test_accel_only_preserves_metadata_identity_but_marks_the_derived_view():
    derived = derive_accel_only(make_stream())
    assert derived.stream == "site_a"
    assert derived.perturbation == "accel_only"


def test_accel_only_refuses_a_stream_without_a_gyroscope():
    with pytest.raises(ValueError, match="no gyroscope"):
        derive_accel_only(make_stream(channels=["acc_x", "acc_y", "acc_z"]))


# --------------------------------------------------------- Scenario 5: rate change


@pytest.mark.parametrize("target", [20.0, 25.0, 100.0])
def test_resampling_changes_length_but_preserves_physical_duration(target):
    stream = make_stream(rate_hz=50.0, window_len=100)
    derived = derive_resampled(stream, target)
    assert derived.rate_hz == target
    assert derived.windows.shape[1] == pytest.approx(100 * target / 50.0, abs=1)
    duration = derived.windows.shape[1] / derived.rate_hz
    assert duration == pytest.approx(stream.windows.shape[1] / stream.rate_hz, rel=0.02)


def test_resampling_to_the_same_rate_is_an_identity():
    stream = make_stream(rate_hz=50.0)
    assert derive_resampled(stream, 50.0) is stream


def test_resampling_preserves_known_physical_source_rate_below_grid_rate():
    stream = make_stream(dataset="xrf_v2", stream="airpods_ear", rate_hz=50.0)
    derived = derive_resampled(stream, 100.0)
    assert derived.effective_source_rate_hz == 25.0


def test_resampling_preserves_each_window_duration_and_updates_lengths():
    stream = make_stream(rate_hz=50.0, window_len=100)
    stream.lengths = np.asarray([100, 75, 50, 100, 80, 60, 100, 90, 70, 100, 85, 65])
    derived = derive_resampled(stream, 100.0)
    assert np.allclose(derived.lengths / derived.rate_hz, stream.lengths / stream.rate_hz,
                       atol=1.0 / derived.rate_hz)
    assert derived.effective_source_rate_hz == 50.0
    assert derived.stream == stream.stream
    for row, length in enumerate(derived.lengths):
        assert np.all(derived.windows[row, length:] == 0.0)


def test_resampling_a_single_valid_sample_stays_finite_and_constant():
    stream = make_stream(rate_hz=50.0, window_len=100)
    stream.lengths = np.full(stream.n_windows, 100, dtype=np.int64)
    stream.lengths[0] = 1
    expected = stream.windows[0, 0].copy()
    derived = derive_resampled(stream, 20.0)
    assert derived.lengths[0] == 1
    assert np.isfinite(derived.windows).all()
    assert np.allclose(derived.windows[0, 0], expected)


def test_derived_views_cannot_collide_with_the_original_feature_cache_identity():
    stream = make_stream(rate_hz=50.0, window_len=100)
    assert source_slice_fingerprint(stream) != source_slice_fingerprint(derive_accel_only(stream))
    assert source_slice_fingerprint(stream) != source_slice_fingerprint(derive_resampled(stream, 100.0))


def test_downsampling_attenuates_high_frequency_content():
    """A decimated stream must lose the band above its new Nyquist, not alias it back in."""
    t = np.arange(200) / 100.0
    signal = np.sin(2 * np.pi * 40.0 * t)            # 40 Hz, above the 20 Hz target Nyquist
    stream = make_stream(rate_hz=100.0, window_len=200, n_per_label=1, labels=("walking",))
    stream.windows[:] = signal[None, :, None].astype(np.float32)
    derived = derive_resampled(stream, 20.0)
    assert np.abs(derived.windows).mean() < 0.5 * np.abs(stream.windows).mean()


def test_resampling_refuses_a_nonpositive_rate():
    with pytest.raises(ValueError):
        derive_resampled(make_stream(), 0.0)


# --------------------------------------------------- Scenarios 2/3/7: cross enrolment


def test_shared_candidates_uses_the_query_streams_wording():
    query = make_stream(labels=("walking", "running"))
    support = make_stream(dataset="other", labels=("walking", "cycling"))
    roster, mapping = shared_candidates(query, support)
    assert roster == ("walking",)
    assert mapping == {"walking": "walking"}


def test_cross_manifest_offsets_support_rows_into_the_concatenated_matrix():
    query = make_stream(stream="site_a")
    support = make_stream(stream="site_b", seed=1)
    cross = build_cross_manifest(query, support, k=1, seed=7)
    assert cross.offset == len(query.gt)
    for plan in cross.plans:
        assert plan.query < cross.offset, "queries index the query stream"
        assert all(row >= cross.offset for row in plan.support), "supports index the support stream"
        assert all(row - cross.offset < len(support.gt) for row in plan.support)


def test_cross_manifest_never_enrols_a_query_against_its_own_execution():
    """Simultaneously recorded placements share execution ids; that is the leak this blocks."""
    query = make_stream(stream="site_a")
    support = make_stream(stream="site_b", seed=1)      # identical execution id scheme
    cross = build_cross_manifest(query, support, k=1, seed=7)
    q_exec = np.asarray(query.execution_ids, dtype=object)
    s_exec = np.asarray(support.execution_ids, dtype=object)
    assert cross.plans
    for plan in cross.plans:
        for row in plan.support:
            assert s_exec[row - cross.offset] != q_exec[plan.query]


def test_cross_manifest_can_require_the_same_or_a_different_subject():
    query = make_stream(stream="site_a")
    support = make_stream(stream="site_b", seed=1)
    q_subj = np.asarray(query.subjects)
    s_subj = np.asarray(support.subjects)
    same = build_cross_manifest(query, support, k=1, seed=7, same_subject=True)
    other = build_cross_manifest(query, support, k=1, seed=7, same_subject=False)
    assert same.plans and other.plans
    for plan in same.plans:
        assert all(s_subj[r - same.offset] == q_subj[plan.query] for r in plan.support)
    for plan in other.plans:
        assert all(s_subj[r - other.offset] != q_subj[plan.query] for r in plan.support)


def test_cross_dataset_subject_relation_requires_an_identity_map():
    query = make_stream(dataset="dataset_a", stream="a")
    support = make_stream(dataset="dataset_b", stream="b", seed=1)
    with pytest.raises(ValueError, match="identity map"):
        build_cross_manifest(query, support, k=1, seed=7, same_subject=False)


def test_matched_reference_uses_identical_queries_and_candidates():
    query = make_stream(stream="site_a")
    support = make_stream(stream="site_b", seed=1)
    cross = build_cross_manifest(query, support, k=1, seed=7, same_subject=False)
    scenario, control = _matched_within_reference(
        cross, query, 1, seed=7, same_subject=False, relation="control",
    )
    assert [plan.query for plan in scenario] == [plan.query for plan in control]
    assert scenario
    assert all(max(plan.support, default=-1) < len(query.gt) for plan in control)


def test_cross_manifest_draws_exactly_k_per_candidate():
    query = make_stream(stream="site_a")
    support = make_stream(stream="site_b", seed=1)
    cross = build_cross_manifest(query, support, k=2, seed=7)
    for plan in cross.plans:
        counts = {label: plan.support_labels.count(label) for label in set(plan.support_labels)}
        assert set(counts) == set(cross.candidates)
        assert set(counts.values()) == {2}


def test_cross_manifest_drops_queries_that_cannot_form_an_honest_episode():
    query = make_stream(stream="site_a")
    support = make_stream(stream="site_b", seed=1, n_per_label=1, executions_per_label=1)
    cross = build_cross_manifest(query, support, k=5, seed=7)
    assert cross.plans == (), "an impossible episode is dropped, never padded"


def test_cross_manifest_is_deterministic():
    query, support = make_stream(stream="a"), make_stream(stream="b", seed=1)
    first = build_cross_manifest(query, support, k=1, seed=7)
    again = build_cross_manifest(query, support, k=1, seed=7)
    assert first.plans == again.plans
    assert first.fingerprint == again.fingerprint


def test_cross_manifest_refuses_a_roster_of_fewer_than_two_shared_labels():
    query = make_stream(labels=("walking", "running"))
    support = make_stream(dataset="other", labels=("walking", "cycling"))
    with pytest.raises(ValueError, match="at least two"):
        build_cross_manifest(query, support, k=1, seed=7)


def test_cross_manifest_rejects_candidates_absent_from_either_stream():
    query, support = make_stream(stream="a"), make_stream(stream="b", seed=1)
    with pytest.raises(ValueError, match="unavailable"):
        build_cross_manifest(query, support, k=1, seed=7, candidates=("walking", "swimming"))


def test_cross_features_concatenation_is_validated():
    query, support = make_stream(stream="a"), make_stream(stream="b", seed=1)
    cross = build_cross_manifest(query, support, k=1, seed=7)
    q = np.zeros((len(query.gt), 4))
    s = np.ones((len(support.gt), 4))
    joined = cross.features(q, s)
    assert joined.shape == (len(query.gt) + len(support.gt), 4)
    assert np.array_equal(joined[cross.offset:], s)
    with pytest.raises(ValueError, match="offset"):
        cross.features(np.zeros((3, 4)), s)
    with pytest.raises(ValueError, match="dimension"):
        cross.features(q, np.ones((len(support.gt), 8)))


def test_cross_manifest_excludes_queries_whose_truth_is_outside_the_shared_roster():
    """An unanswerable query would depress every model equally and hide the real comparison."""
    query = make_stream(labels=("walking", "running", "cycling"))
    support = make_stream(dataset="other", labels=("walking", "running"), seed=1)
    cross = build_cross_manifest(query, support, k=1, seed=7)
    assert set(cross.candidates) == {"walking", "running"}
    truth = np.asarray(query.gt, dtype=object)
    assert cross.plans
    assert all(truth[plan.query] in cross.candidates for plan in cross.plans)
    assert not any(truth[plan.query] == "cycling" for plan in cross.plans)


def test_cross_manifest_aligns_support_labels_to_the_stream_vocabulary():
    """Regression: a grid's raw spelling must be canonicalised before the concept map is applied.

    MM-Fit stores ``push_up`` in ``gt`` but registers ``pushups`` in ``eval_labels``.  Matching the
    raw spelling against a roster built from aligned labels silently produced an empty manifest with
    no error, which is the worst possible failure for an evaluation harness.
    """
    query = make_stream(labels=("pushups", "squats"))
    support = make_stream(dataset="other", labels=("pushups", "squats"), seed=1)
    support.gt[:] = ["push_up" if label == "pushups" else "squat" for label in support.gt]
    cross = build_cross_manifest(query, support, k=1, seed=7)
    assert set(cross.candidates) == {"pushups", "squats"}
    assert cross.plans, "aligned support labels must still form episodes"
    for plan in cross.plans:
        assert set(plan.support_labels) == {"pushups", "squats"}


def test_cross_manifest_ignores_support_rows_with_no_registered_label():
    query = make_stream(labels=("walking", "running"))
    support = make_stream(dataset="other", labels=("walking", "running"), seed=1)
    support.gt[0] = "not_a_registered_activity"
    cross = build_cross_manifest(query, support, k=1, seed=7)
    assert cross.plans
    for plan in cross.plans:
        assert all(row - cross.offset != 0 for row in plan.support)
def test_evidence_diagnostics_are_opt_in_and_oracle_is_never_implicit():
    assert runner._evidence_diagnostic_requests(None, include_oracle=False) == (False, False)
    assert runner._evidence_diagnostic_requests(
        frozenset({"halo-classifier-label-meaning-only"}), include_oracle=False,
    ) == (True, False)
    assert runner._evidence_diagnostic_requests(None, include_oracle=True) == (False, True)
