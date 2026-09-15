"""Scenario 1 (partial enrollment coverage) — unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from training.support_classifier.partial_coverage import (
    CoverageCell,
    choose_hidden_candidates,
    hide_supports,
    hybrid_predictions,
    support_only_predictions,
    truth_split,
    zscore,
)
from training.support_classifier.sealed_eval import (
    QueryPlan,
    _readout_predictions,
    manifest_fingerprint,
)

CANDIDATES = ("walking", "running", "sitting", "standing")


def _plans(k: int = 2, n_queries: int = 6) -> list[QueryPlan]:
    """Episodes whose support rows are laid out at a predictable offset per candidate."""
    plans = []
    for query in range(n_queries):
        support, labels = [], []
        for slot, label in enumerate(CANDIDATES):
            for repeat in range(k):
                support.append(100 + slot * 10 + repeat)
                labels.append(label)
        plans.append(QueryPlan(query=query, support=tuple(support), support_labels=tuple(labels)))
    return plans


def _features(dim: int = 8, rows: int = 200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, dim))


# ----------------------------------------------------------------- cell choice


def test_choose_hidden_is_stable_and_partitions_the_roster():
    first = choose_hidden_candidates(CANDIDATES, coverage=0.5, seed_parts=("a", 1))
    again = choose_hidden_candidates(CANDIDATES, coverage=0.5, seed_parts=("a", 1))
    assert first == again
    assert set(first.supported) | set(first.hidden) == set(CANDIDATES)
    assert not set(first.supported) & set(first.hidden)
    assert len(first.supported) == 2


def test_choose_hidden_varies_with_seed_parts():
    cells = {
        choose_hidden_candidates(CANDIDATES, coverage=0.5, seed_parts=("cell", index)).hidden
        for index in range(12)
    }
    assert len(cells) > 1, "hidden sets must depend on the cell identity"


@pytest.mark.parametrize("coverage", [0.01, 0.2, 0.5, 0.8, 0.99])
def test_every_cell_keeps_one_supported_and_one_hidden(coverage):
    cell = choose_hidden_candidates(CANDIDATES, coverage=coverage, seed_parts=("x",))
    assert cell.supported and cell.hidden


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, 1.5])
def test_degenerate_coverage_is_refused(coverage):
    with pytest.raises(ValueError):
        choose_hidden_candidates(CANDIDATES, coverage=coverage, seed_parts=("x",))


def test_single_candidate_roster_is_refused():
    with pytest.raises(ValueError):
        choose_hidden_candidates(("walking",), coverage=0.5, seed_parts=("x",))


# ------------------------------------------------------------------- hiding


def test_hide_supports_removes_exactly_the_hidden_candidates():
    cell = CoverageCell(supported=("running", "walking"), hidden=("sitting", "standing"),
                        coverage=0.5, requested_coverage=0.5)
    plans = _plans()
    hidden_plans = hide_supports(plans, cell)
    for before, after in zip(plans, hidden_plans):
        assert after.query == before.query
        assert set(after.support_labels) == {"running", "walking"}
        # Surviving rows keep their original identity and order.
        kept = [row for row, label in zip(before.support, before.support_labels)
                if label in {"running", "walking"}]
        assert list(after.support) == kept


def test_hiding_nothing_is_a_no_op_and_preserves_the_manifest_fingerprint():
    cell = CoverageCell(supported=CANDIDATES, hidden=(), coverage=1.0, requested_coverage=1.0)
    plans = _plans()
    assert hide_supports(plans, cell) == plans
    assert manifest_fingerprint(hide_supports(plans, cell)) == manifest_fingerprint(plans)


def test_truth_split_partitions_queries_by_enrolment():
    cell = CoverageCell(supported=("walking",), hidden=("running",),
                        coverage=0.5, requested_coverage=0.5)
    supported, missing = truth_split(["walking", "running", "walking"], cell)
    assert supported.tolist() == [0, 2]
    assert missing.tolist() == [1]


# -------------------------------------------------------------------- zscore


def test_zscore_standardises_live_entries_and_zeroes_the_rest():
    scores = np.array([[1.0, 2.0, 3.0, 99.0]])
    mask = np.array([[True, True, True, False]])
    out = zscore(scores, mask)
    assert out[0, 3] == 0.0
    assert out[0, :3].mean() == pytest.approx(0.0, abs=1e-12)
    assert out[0, :3].std() == pytest.approx(1.0)


def test_zscore_returns_zeros_for_constant_or_degenerate_rows():
    assert np.allclose(zscore(np.array([[5.0, 5.0, 5.0]])), 0.0)
    one_live = zscore(np.array([[5.0, 1.0]]), np.array([[True, False]]))
    assert np.allclose(one_live, 0.0), "a single live entry carries no ranking information"


# ------------------------------------------------------- support-only readouts


def test_support_only_readouts_never_name_a_hidden_candidate():
    cell = choose_hidden_candidates(CANDIDATES, coverage=0.5, seed_parts=("cell",))
    plans = hide_supports(_plans(), cell)
    out = support_only_predictions(_features(), CANDIDATES, plans, cell)
    for readout, predicted in out.items():
        assert predicted, readout
        assert set(predicted) <= set(cell.supported), readout


def test_support_only_readouts_produce_no_nan_for_empty_classes():
    cell = choose_hidden_candidates(CANDIDATES, coverage=0.25, seed_parts=("cell",))
    plans = hide_supports(_plans(), cell)
    with np.errstate(invalid="raise"):
        out = support_only_predictions(_features(), CANDIDATES, plans, cell)
    assert all(all(isinstance(label, str) for label in values) for values in out.values())


def test_full_coverage_reproduces_the_sealed_readouts_exactly():
    """The restricted implementation must be the same function on a complete roster."""
    cell = CoverageCell(supported=CANDIDATES, hidden=(), coverage=1.0, requested_coverage=1.0)
    plans = _plans()
    features = _features()
    labels = np.array([""] * len(features), dtype=object)
    reference = _readout_predictions(features, labels, CANDIDATES, plans, device=None)
    mine = support_only_predictions(features, CANDIDATES, plans, cell)
    for readout in ("1nn", "prototype", "ridge"):
        assert mine[readout] == reference[readout], readout


def test_support_only_refuses_an_episode_with_no_enrolment_left():
    cell = CoverageCell(supported=("walking",), hidden=tuple(c for c in CANDIDATES if c != "walking"),
                        coverage=0.25, requested_coverage=0.25)
    empty = [QueryPlan(query=0, support=(), support_labels=())]
    with pytest.raises(ValueError):
        support_only_predictions(_features(), CANDIDATES, empty, cell)


# -------------------------------------------------------------------- hybrid


def test_hybrid_can_name_a_hidden_candidate():
    """The whole point of the scenario: support-only readouts cannot, the hybrid can."""
    cell = CoverageCell(supported=("walking", "running"), hidden=("sitting", "standing"),
                        coverage=0.5, requested_coverage=0.5)
    plans = hide_supports(_plans(n_queries=4), cell)
    features = _features(seed=3)
    # Text evidence overwhelmingly favours a candidate that carries no enrolment.
    text = np.zeros((len(features), len(CANDIDATES)))
    text[:, CANDIDATES.index("sitting")] = 10.0
    predicted = hybrid_predictions(text, features, CANDIDATES, plans, cell)
    assert set(predicted) == {"sitting"}


def test_hybrid_follows_support_when_text_is_uninformative():
    cell = CoverageCell(supported=("walking", "running"), hidden=("sitting", "standing"),
                        coverage=0.5, requested_coverage=0.5)
    plans = hide_supports(_plans(n_queries=5), cell)
    features = _features(seed=5)
    flat_text = np.zeros((len(features), len(CANDIDATES)))
    predicted = hybrid_predictions(flat_text, features, CANDIDATES, plans, cell)
    support_only = support_only_predictions(features, CANDIDATES, plans, cell)["1nn"]
    assert predicted == support_only


def test_hybrid_rejects_a_mismatched_text_matrix():
    cell = choose_hidden_candidates(CANDIDATES, coverage=0.5, seed_parts=("cell",))
    plans = hide_supports(_plans(), cell)
    with pytest.raises(ValueError):
        hybrid_predictions(np.zeros((200, 2)), _features(), CANDIDATES, plans, cell)


# ------------------------------------------------------------- reporting splits

class _FakeStream:
    dataset = "toy"
    stream = "toy_stream"
    window_seconds = 8.0

    def __init__(self, truth):
        self.gt = list(truth)
        self.eval_labels = list(CANDIDATES)
        self.subjects = np.arange(len(truth)) % 3


def _patch_labels(monkeypatch, truth):
    import training.support_classifier.run_partial_coverage as runner
    monkeypatch.setattr(runner, "_aligned_labels",
                        lambda stream: np.asarray(stream.gt, dtype=object))
    return runner


def test_emit_rows_splits_by_enrolment_and_picks_honest_metrics(monkeypatch):
    truth = ["walking", "sitting", "walking", "sitting"]
    runner = _patch_labels(monkeypatch, truth)
    cell = CoverageCell(supported=("walking",), hidden=("sitting",),
                        coverage=0.5, requested_coverage=0.5)
    plans = [QueryPlan(query=i, support=(10,), support_labels=("walking",))
             for i in range(len(truth))]
    rows = runner.emit_rows(_FakeStream(truth), plans, ["walking"] * 4, cell,
                            model="m", readout="1nn", k=1, window_seconds=8.0,
                            bootstrap=0, manifest="abc")
    by_split = {row["coverage_split"]: row for row in rows}
    assert set(by_split) == {"all", "truth_enrolled", "truth_unenrolled"}
    assert by_split["all"]["primary_metric"] == "f1_macro"
    assert by_split["all"]["f1_macro"] is not None
    for split in ("truth_enrolled", "truth_unenrolled"):
        row = by_split[split]
        assert row["primary_metric"] == "balanced_accuracy"
        assert row["f1_macro"] is not None
        assert "f1_macro_note" in row
    # Always answering the one enrolled label: perfect on enrolled truth, zero on unenrolled.
    assert by_split["truth_enrolled"]["balanced_accuracy"] == pytest.approx(100.0)
    assert by_split["truth_unenrolled"]["balanced_accuracy"] == pytest.approx(0.0)
    assert by_split["truth_enrolled"]["n_queries"] == 2
    assert by_split["truth_unenrolled"]["n_queries"] == 2


def test_emit_rows_bootstraps_wrong_predictions_outside_the_conditional_truth(monkeypatch):
    """A hidden-label query predicted as an enrolled label is an error, not a CI crash."""
    truth = ["walking", "sitting", "walking", "sitting"]
    runner = _patch_labels(monkeypatch, truth)
    cell = CoverageCell(supported=("walking",), hidden=("sitting",),
                        coverage=0.5, requested_coverage=0.5)
    plans = [QueryPlan(query=i, support=(10,), support_labels=("walking",))
             for i in range(len(truth))]
    rows = runner.emit_rows(_FakeStream(truth), plans, ["walking"] * len(truth), cell,
                            model="m", readout="1nn", k=1, window_seconds=8.0,
                            bootstrap=8, manifest="abc")
    hidden = {row["coverage_split"]: row for row in rows}["truth_unenrolled"]
    assert hidden["status"] == "ok"
    assert hidden["balanced_accuracy"] == pytest.approx(0.0)
    assert hidden["bootstrap_B"] == 8
    assert hidden["balanced_accuracy_ci_lo"] == pytest.approx(0.0)


def test_emit_rows_carries_the_coverage_configuration(monkeypatch):
    truth = ["walking", "sitting"]
    runner = _patch_labels(monkeypatch, truth)
    cell = CoverageCell(supported=("walking",), hidden=("sitting",),
                        coverage=0.5, requested_coverage=0.5)
    plans = [QueryPlan(query=i, support=(10,), support_labels=("walking",)) for i in range(2)]
    rows = runner.emit_rows(_FakeStream(truth), plans, ["walking"] * 2, cell,
                            model="halo", readout="halo-classifier", k=1,
                            window_seconds=8.0, bootstrap=0, manifest="abc")
    for row in rows:
        assert row["hidden_candidates"] == ["sitting"]
        assert row["supported_candidates"] == ["walking"]
        assert row["coverage_fingerprint"] == cell.fingerprint
        assert row["scenario"] == "partial_coverage_v1"


def test_cannot_attempt_row_is_disclosed_not_scored(monkeypatch):
    runner = _patch_labels(monkeypatch, ["walking"])
    cell = CoverageCell(supported=("walking",), hidden=("sitting",),
                        coverage=0.5, requested_coverage=0.5)
    rows = runner.cannot_attempt_rows(_FakeStream(["walking"]), cell, model="limubert_x",
                                      readout="hybrid-text-support", k=1, window_seconds=8.0,
                                      reason="no text path")
    assert len(rows) == 1
    assert rows[0]["status"] == "cannot_attempt"
    assert "f1_macro" not in rows[0], "a model that cannot attempt the split must not be scored 0"
