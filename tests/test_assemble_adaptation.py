import pytest

from eval.assemble_adaptation import (
    _external_rows,
    _markdown,
    _unverified_execution_support,
    dataset_macro,
    load_rows,
    paired_deltas,
    variant_deltas,
    zero_shot_coverage,
)


def test_variant_native_rows_are_visible_and_get_paired_comparisons():
    rows = [
        _row(model="halo_compare@sensor_only_trained", method="support_comparator",
             label_mode="coherent", k=1, score=60),
        _row(model="unimts", method="nearest", label_mode="coherent", k=1, score=50),
    ]
    result = paired_deltas(rows, samples=10)
    assert len(result) == 1
    assert result[0]["delta_f1_macro"] == 10
    assert result[0]["target"] == "halo_compare@sensor_only_trained/support_comparator"
    assert "halo_compare@sensor_only_trained | support comparator" in _markdown(dataset_macro(rows))


def test_zero_shot_common_coverage_matches_cells_not_just_datasets():
    rows = []
    for model, cells in (("halo_compare@trained", ["left"]), ("unimts", ["left", "right"])):
        for cell in cells:
            row = _row(model=model, method="zero_shot", label_mode="coherent", k=0, score=50)
            row["cell"] = cell
            rows.append(row)
    common, text = zero_shot_coverage(rows, ["halo_compare@trained", "unimts"])
    assert len(common) == 2
    assert {r["cell"] for r in common} == {"left"}
    assert "halo_compare@trained" in _markdown(dataset_macro(rows))
    assert "Common-coverage" in text
    common, text = zero_shot_coverage(rows, ["halo_compare@trained", "unimts", "normwear"])
    assert common == []
    assert "No common" in text


@pytest.mark.parametrize("dirty,commit", [(True, "abc"), (None, None), (False, None)])
def test_publication_preflight_rejects_unpinned_source(monkeypatch, dirty, commit):
    import eval.run_adaptation_baselines as runner
    monkeypatch.setattr(runner, "_git_provenance", lambda: {"git_dirty": dirty, "git_commit": commit})
    with pytest.raises(RuntimeError, match="committed, clean"):
        runner.require_publication_source()


def test_publication_preflight_accepts_clean_commit(monkeypatch):
    import eval.run_adaptation_baselines as runner
    monkeypatch.setattr(runner, "_git_provenance", lambda: {"git_dirty": False, "git_commit": "abc"})
    assert runner.require_publication_source()["git_commit"] == "abc"


def _row(*, model, method, label_mode, k, score, subject="dataset:s1"):
    return {
        "model": model,
        "method": method,
        "regime": "ordinary",
        "label_mode": label_mode,
        "k": k,
        "cell": "dataset/stream/source/match/cohort",
        "seed": 7,
        "subject": subject,
        "f1_macro": score,
        "dataset": subject.split(":", 1)[0],
        "subject_relation": "cross_subject",
        "configuration_relation": "same_configuration",
        "cohort": "main",
        "analysis_set": "main_common_k1_8",
    }


def test_paired_deltas_do_not_pool_label_modes_or_support_counts():
    rows = []
    for label_mode, k, target, comparator in (
        ("coherent", 1, 50.0, 40.0),
        ("coherent", 2, 60.0, 55.0),
        ("random_alias", 1, 45.0, 45.0),
    ):
        rows.extend([
            _row(
                model="halo_compare", method="support_comparator",
                label_mode=label_mode, k=k, score=target,
            ),
            _row(
                model="baseline", method="nearest",
                label_mode=label_mode, k=k, score=comparator,
            ),
        ])

    results = paired_deltas(rows, samples=50)

    assert [
        (row["label_mode"], row["k"], row["delta_f1_macro"])
        for row in results
    ] == [
        ("coherent", 1, 10.0),
        ("coherent", 2, 5.0),
        ("random_alias", 1, 0.0),
    ]


def test_paired_deltas_average_repeated_cells_within_subject():
    rows = []
    for cell, target, comparator in (("cell-a", 50.0, 40.0), ("cell-b", 30.0, 40.0)):
        target_row = _row(
            model="halo_compare", method="support_comparator",
            label_mode="coherent", k=1, score=target,
        )
        comparator_row = _row(
            model="baseline", method="nearest",
            label_mode="coherent", k=1, score=comparator,
        )
        target_row["cell"] = cell
        comparator_row["cell"] = cell
        rows.extend([target_row, comparator_row])

    result = paired_deltas(rows, samples=50)[0]

    assert result["paired_subjects"] == 1
    assert result["delta_f1_macro"] == 0.0


def test_paired_deltas_give_datasets_equal_weight():
    rows = []
    for dataset, subjects, delta in (("a", 1, 0.0), ("b", 9, 100.0)):
        for subject_index in range(subjects):
            for model, method, score in (
                ("halo_compare", "support_comparator", delta),
                ("baseline", "nearest", 0.0),
            ):
                rows.append(_row(
                    model=model, method=method, label_mode="coherent", k=1, score=score,
                    subject=f"{dataset}:s{subject_index}",
                ))

    result = paired_deltas(rows, samples=50)[0]

    assert result["paired_datasets"] == 2
    assert result["paired_subjects"] == 10
    assert result["delta_f1_macro"] == 50.0


def test_external_rows_preserve_protocol_relations_and_do_not_pool_them():
    cells = {
        "dataset/q/from_s/same_configuration/same_subject": {
            "dataset": "dataset", "kind": "enrollment", "support_ceiling": 8,
            "subject_relation": "same_subject", "configuration_relation": "same_configuration",
        },
        "dataset/q/from_s/same_configuration/cross_subject": {
            "dataset": "dataset", "kind": "enrollment", "support_ceiling": 8,
            "subject_relation": "cross_subject", "configuration_relation": "same_configuration",
        },
    }
    results = {}
    for relation, score in (("same_subject", 10.0), ("cross_subject", 90.0)):
        cell = f"dataset/q/from_s/same_configuration/{relation}"
        results[f"{cell}/coherent/seed1/k1"] = {
            "kind": "enrollment", "regime": "ordinary", "label_mode": "coherent",
            "support_count": 1, "seed": 1, "cohort": "main",
            "nearest": {"f1_macro": score}, "subject_results": {},
        }
    rows, _ = _external_rows(
        {"baseline": "baseline", "methods": ["nearest"], "results": results},
        {"cells": cells},
    )

    aggregates = dataset_macro(rows)

    assert {row["subject_relation"] for row in rows} == {"same_subject", "cross_subject"}
    assert len(aggregates) == 2
    assert {row["f1_macro"] for row in aggregates} == {10.0, 90.0}


def test_zero_shot_markdown_omits_non_native_harnet_bridge():
    aggregates = [
        {
            "model": model,
            "method": "zero_shot",
            "regime": "ordinary",
            "label_mode": "coherent",
            "k": 0,
            "f1_macro": 25.0,
            "datasets": 1,
            "subject_relation": "none",
            "configuration_relation": "none",
            "cohort": "zero_shot",
            "analysis_set": "zero_shot",
        }
        for model in ("halo_compare", "harnet", "unimts")
    ]

    table = _markdown(aggregates)

    assert "| ordinary | halo_compare | zero_shot |" in table
    assert "| ordinary | unimts | zero_shot |" in table
    assert "| ordinary | harnet | zero_shot |" not in table


def _enrollment_payload(model_field, score):
    results = {
        "dataset/q/from_s/same_configuration/cross_subject/coherent/seed1/k1": {
            "kind": "enrollment", "regime": "ordinary", "label_mode": "coherent",
            "support_count": 1, "seed": 1, "cohort": "main",
            "nearest": {"f1_macro": score}, "subject_results": {},
        }
    }
    payload = {"baseline": "harnet", "methods": ["nearest"], "results": results}
    if model_field is not None:
        payload["model"] = model_field
    return payload


_CELLS = {
    "dataset/q/from_s/same_configuration/cross_subject": {
        "dataset": "dataset", "kind": "enrollment", "support_ceiling": 8,
        "subject_relation": "cross_subject", "configuration_relation": "same_configuration",
    },
}


def test_variant_identity_becomes_the_model_column():
    """A step-0 control of the same adapter must assemble as its own model, never pooled."""
    trained, _ = _external_rows(_enrollment_payload(None, 90.0), {"cells": _CELLS})
    control, _ = _external_rows(_enrollment_payload("harnet@step0", 10.0), {"cells": _CELLS})
    assert {row["model"] for row in trained} == {"harnet"}
    assert {row["model"] for row in control} == {"harnet@step0"}
    aggregates = dataset_macro(trained + control)
    assert {(row["model"], row["f1_macro"]) for row in aggregates} == {
        ("harnet", 90.0), ("harnet@step0", 10.0),
    }


def test_two_artifacts_with_one_identity_are_refused(tmp_path):
    """Without a variant the two runs would be averaged into one number; refuse instead."""
    import json

    import baselines
    from eval.run_adaptation_baselines import _source_fingerprint

    fingerprint = _source_fingerprint(baselines.REGISTRY["harnet"])
    manifest = {"manifest_fingerprint": "m", "cells": _CELLS}
    paths = []
    for index, score in enumerate((90.0, 10.0)):
        payload = {
            **_enrollment_payload(None, score), "schema_version": 2,
            "source_fingerprint": fingerprint, "evaluation_artifacts": {},
            "git_dirty": False, "git_commit": "test-commit", "manifest_fingerprint": "m",
        }
        path = tmp_path / f"run{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    with pytest.raises(ValueError, match="already loaded"):
        load_rows(paths, manifest)
    # The same two runs assemble once one of them carries a variant.
    payload = json.loads(paths[1].read_text())
    payload["model"] = "harnet@step0"
    paths[1].write_text(json.dumps(payload))
    rows, _ = load_rows(paths, manifest)
    assert {row["model"] for row in rows} == {"harnet", "harnet@step0"}


def test_unverified_executions_leave_every_enrollment_analysis_set():
    single = {
        "kind": "enrollment", "support_ceiling": 8,
        "execution_identity_known": False,
        "seeds": {"1": {"plans": [{"support_execution_rows": [[[0], [1]], [[2], [3]]]}]}},
        "secondary_high_support": {"seeds": {"1": {"plans": []}}},
    }
    pooled = {
        "kind": "enrollment", "support_ceiling": 8,
        "seeds": {"1": {"plans": [{"support_execution_rows": [[[0, 1], [2]], [[3], [4, 5]]]}]}},
    }
    assert _unverified_execution_support(single)
    assert not _unverified_execution_support(pooled)
    assert not _unverified_execution_support({"kind": "zero_shot"})
    assert _unverified_execution_support({"kind": "enrollment", "dataset": "tnda_har"})
    cells = {
        "dataset/q/from_s/same_configuration/cross_subject": {
            **_CELLS["dataset/q/from_s/same_configuration/cross_subject"], **single,
        },
    }
    rows, _ = _external_rows(_enrollment_payload(None, 50.0), {"cells": cells})
    assert rows == [], "unverified enrollment must not appear as scored supplementary results"


def test_variant_deltas_pair_each_variant_with_its_base_only():
    """``harnet@query_orientation`` is read against ``harnet``; ``halo_compare@step0`` against
    ``halo_compare``; nothing is paired across models, and unrelated variants are skipped."""
    rows = []
    for model, score in (("harnet", 60.0), ("harnet@query_orientation", 45.0),
                         ("unimts", 70.0), ("unimts@query_orientation", 68.0),
                         ("halo_compare", 55.0), ("halo_compare@step0", 50.0),
                         ("orphan@x", 1.0)):
        for dataset in ("d1", "d2"):
            rows.append({
                **_row(model=model, method="nearest", label_mode="coherent", k=1,
                       score=score + (5.0 if dataset == "d2" else 0.0),
                       subject=f"{dataset}:s1"),
            })
    subjects = [dict(row) for row in rows]
    out = variant_deltas(rows, subjects, samples=200)
    by_model = {row["model"]: row for row in out}
    assert set(by_model) == {"harnet@query_orientation", "unimts@query_orientation",
                             "halo_compare@step0"}
    assert by_model["harnet@query_orientation"]["delta_f1_macro"] == pytest.approx(-15.0)
    assert by_model["unimts@query_orientation"]["delta_f1_macro"] == pytest.approx(-2.0)
    assert by_model["halo_compare@step0"]["delta_f1_macro"] == pytest.approx(-5.0)
    assert by_model["harnet@query_orientation"]["base_f1_macro"] == pytest.approx(62.5)
    assert by_model["harnet@query_orientation"]["datasets"] == 2
    ci = by_model["harnet@query_orientation"]["subject_ci95"]
    assert ci is not None and ci[0] <= -15.0 <= ci[1]


def test_variant_deltas_do_not_pool_conditions():
    rows = []
    for k, score in ((1, 40.0), (2, 60.0)):
        rows.append(_row(model="harnet", method="nearest", label_mode="coherent", k=k, score=50.0))
        rows.append(_row(model="harnet@query_gravity", method="nearest", label_mode="coherent",
                         k=k, score=score))
    out = variant_deltas(rows, [], samples=10)
    assert {(row["k"], round(row["delta_f1_macro"], 6)) for row in out} == {(1, -10.0), (2, 10.0)}
    assert all(row["ci95"] is None for row in out), "no subject rows, no interval"


def test_variant_cell_and_subject_estimands_are_reported_separately():
    from eval.assemble_adaptation import _variant_markdown
    base = _row(model="harnet", method="nearest", label_mode="coherent", k=1, score=50.)
    variant = {**base, "model": "harnet@x", "f1_macro": 60.}
    rows = [base, variant]
    subjects = [base, {**variant, "f1_macro": 80.}]
    result = variant_deltas(rows, subjects, samples=10)[0]
    assert result["delta_f1_macro"] == 10.
    assert result["ci95"] is None
    assert result["subject_delta_f1_macro"] == 30.
    assert result["subject_ci95"] == [30., 30.]
    markdown = "\n".join(_variant_markdown([result]))
    assert "label mode" in markdown and "coherent" in markdown
    assert "cell delta" in markdown and "subject delta" in markdown
