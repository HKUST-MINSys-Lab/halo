from eval.assemble_adaptation import _external_rows, _markdown, dataset_macro, paired_deltas


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
