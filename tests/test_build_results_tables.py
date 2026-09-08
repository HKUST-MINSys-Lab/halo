import pytest

from eval.build_results_tables import (
    MODEL_NAMES,
    _validate_current_cells,
    table_label_efficiency,
    table_per_dataset,
    table_zero_shot,
)


def test_zero_shot_compares_only_matched_coverage():
    cells = [dict(model=m, method="zero_shot", label_mode="coherent", dataset=d,
                  cell=d, k=0, f1_macro=score)
             for m, d, score in [("halo_compare", "easy", 90), ("unimts", "easy", 95),
                                 ("unimts", "hard", 30)]]
    text = table_zero_shot(cells)
    assert "| UniMTS | **95.00** |" in text
    assert "62.50" not in text
    assert "Matched held-out datasets: 1; cells: 1" in text
    assert "| HALO (ours) | 1 | 1 |" in text
    assert "| UniMTS | 2 | 2 |" in text


def test_label_efficiency_excludes_random_alias_rows() -> None:
    cells = [
        {
            "model": model,
            "method": method,
            "regime": regime,
            "label_mode": "coherent",
            "dataset": "example",
            "k": "1",
            "f1_macro": "80.0" if method == "support_comparator" else "40.0",
        }
        for model in MODEL_NAMES
        for regime in ("ordinary", "specialized_novel")
        for method in (("support_comparator", "nearest", "prototype", "ridge")
                       if model == "halo_compare" else ("nearest", "prototype", "ridge"))
    ] + [
        {
            "model": "halo_compare",
            "method": "support_comparator",
            "regime": "ordinary",
            "label_mode": "random_alias",
            "dataset": "example",
            "k": "1",
            "f1_macro": "20.0",
        },
    ]

    table = table_label_efficiency(cells)

    assert "| HALO / support comparator | **80.00** |" in table
    assert "HARNet / 1-NN" in table
    assert "HARNet / prototype" in table
    assert "HARNet / ridge" in table
    assert "linear_head" not in table
    assert "50.00" not in table
    assert "ordinary" not in table
    assert "specialized" not in table


def test_report_excludes_self_pretrained_models_and_non_native_zero_shot() -> None:
    cells = [
        {
            "model": model,
            "method": "zero_shot",
            "label_mode": "coherent",
            "dataset": "example",
            "k": "0",
            "f1_macro": "50.0",
        }
        for model in (*MODEL_NAMES, "crosshar", "limubert")
    ]

    table = table_zero_shot(cells)

    assert "HALO (ours)" in table
    assert "UniMTS" in table
    assert "ImageBind" in table
    assert "NormWear" in table
    assert "HARNet" not in table
    assert "CrossHAR" not in table
    assert "LIMU-BERT" not in table


def test_current_report_rejects_missing_matched_readout() -> None:
    cells = [
        {
            "model": model,
            "method": method,
            "k": "1",
        }
        for model in (*MODEL_NAMES, "crosshar", "limubert")
        for method in (("support_comparator", "nearest", "prototype", "ridge")
                       if model == "halo_compare" else ("nearest", "prototype", "ridge"))
        if not (model == "harnet" and method == "ridge")
    ]

    with pytest.raises(ValueError, match="harnet/ridge"):
        _validate_current_cells(cells)


def test_per_dataset_table_keeps_datasets_separate() -> None:
    cells = [
        {
            "model": model,
            "method": method,
            "regime": "ordinary",
            "label_mode": "coherent",
            "dataset": dataset,
            "k": "1",
            "f1_macro": "90.0" if dataset == "inclusivehar" else "10.0",
        }
        for dataset in ("inclusivehar", "usc_had")
        for model in MODEL_NAMES
        for method in (("support_comparator", "nearest", "prototype", "ridge")
                       if model == "halo_compare" else ("nearest", "prototype", "ridge"))
    ] + [
        {
            "model": model,
            "method": "zero_shot",
            "regime": "ordinary",
            "label_mode": "coherent",
            "dataset": dataset,
            "k": "0",
            "f1_macro": "50.0",
        }
        for dataset in ("inclusivehar", "usc_had")
        for model in MODEL_NAMES
    ]

    table = table_per_dataset(cells)

    assert "#### Inclusive-HAR" in table
    assert "#### USC-HAD" in table
    assert "ordinary" not in table
    assert "specialized" not in table
    assert "| HALO / support comparator | **90.00** |" in table
    assert "| HALO / support comparator | **10.00** |" in table
    assert "CrossHAR" not in table
    assert "LIMU-BERT" not in table


def test_step0_control_rows_are_listed_beside_the_trained_row() -> None:
    """A ``halo_compare@step0`` run appears as its own labelled row and never replaces HALO."""
    def cell(model, method, score):
        return {
            "model": model, "method": method, "regime": "ordinary", "label_mode": "coherent",
            "dataset": "example", "k": "1", "f1_macro": score,
        }

    cells = [cell("halo_compare", "support_comparator", "80.0"),
             cell("halo_compare@step0", "support_comparator", "40.0")]
    for model in MODEL_NAMES:
        cells.extend(cell(model, method, "30.0") for method in ("nearest", "prototype", "ridge"))
    table = table_label_efficiency(cells)
    assert "| HALO / support comparator | **80.00** |" in table
    assert "| HALO step-0 control (untrained) | 40.00 |" in table
    without = table_label_efficiency([c for c in cells if c["model"] != "halo_compare@step0"])
    assert "step-0" not in without
