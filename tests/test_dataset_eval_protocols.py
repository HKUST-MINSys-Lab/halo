from __future__ import annotations

from data.datasets.mmfit.protocol import load_protocol


def test_mmfit_publication_reference_and_query_workouts_do_not_overlap():
    protocol = load_protocol()
    assert not set(protocol["reference_workouts"]) & set(protocol["query_workouts"])
    assert protocol["role"] == "scenario_only"
