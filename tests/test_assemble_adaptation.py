from eval.assemble_adaptation import _markdown, paired_deltas


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
        }
        for model in ("halo_compare", "harnet", "unimts")
    ]

    table = _markdown(aggregates)

    assert "| ordinary | halo_compare | zero_shot |" in table
    assert "| ordinary | unimts | zero_shot |" in table
    assert "| ordinary | harnet | zero_shot |" not in table
