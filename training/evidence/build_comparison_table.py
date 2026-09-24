"""Assemble the matched Phase-B method-comparison table from enrollment result files.

The table that `docs/results/PHASE_B_TRAINING_STATUS.md` §4 reports was produced by a script that
was never committed, and it aggregates as an unweighted mean over three cells. That weighting is
not defensible here: RealWorld contributes 203 of 2,953 queries (6.9%) over 3 subjects with a
2-way decision, yet carries a third of the reported number. This module replaces it with a
committed, re-runnable aggregation that reports **query-weighted** means, keeps the per-cell
numbers next to them, and prints a chance floor beside every score so "above chance" is a fact on
the page instead of an inference.

Several arms (checkpoints) can be compared at once. Aggregation is restricted to the cells that
every arm scored successfully, so the comparison is matched by construction; any cell dropped for
that reason is listed rather than silently omitted.

Two floors are reported, because macro-F1 has two different degenerate strategies:

    chance          uniform random over C candidates      100/C
    constant        always predict one candidate          100 * 2/(C*(C+1))

Usage:
    python -m training.evidence.build_comparison_table \
        --arm step0 out/enroll_step0_coherent.json out/enroll_step0_aliases.json \
        --arm step1000 out/enroll_step1000_coherent.json out/enroll_step1000_aliases.json \
        --out-prefix training/evidence/outputs/diagnostics/phase_b_20260808/stage1_step0_control/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

# Columns that come from the predictor being scored, and therefore differ between arms.
ARM_METRICS = {
    "full_engine_f1": "f1_macro",
    "identity_vote_f1": "identity_f1_macro",
    "support_removed_f1": "support_removed_f1_macro",
    "support_shuffled_f1": "support_label_shuffled_f1_macro",
}
# Columns computed directly from the frozen Phase-A embeddings and the declared support. They do
# not involve the decoder or the retriever, so every arm must reproduce them exactly.
SHARED_METRICS = {
    "prototype_f1": "prototype_f1_macro",
    "ridge_f1": "ridge_head_f1_macro",
}
SHARED_TOLERANCE = 1e-6


def load_arm(paths: list[Path]) -> dict:
    """Merge one arm's result files (typically coherent + neutral-alias runs)."""
    cells, meta = {}, {}
    for path in paths:
        payload = json.loads(path.read_text())
        label_mode = "neutral_alias" if payload.get("random_aliases") else "coherent"
        for key, result in payload["results"].items():
            if result.get("status") or not result.get("queries"):
                continue
            cell = (label_mode, _support_of(key), _cell_of(key))
            if cell in cells:
                raise SystemExit(f"duplicate cell {cell} across {paths}")
            cells[cell] = result
        for field in ("predictor", "predictor_fp", "predictor_step", "predictor_selection",
                      "predictor_mode",
                      "untrained_control", "training_regime", "bank_fp", "checkpoint_fp", "seed",
                      "evaluation_regime", "evaluation_source_fp", "evaluation_protocol_fp"):
            value = payload.get(field)
            if field in meta and meta[field] != value:
                # Only the label-mode-specific fields may differ between an arm's two files.
                raise SystemExit(
                    f"arm files disagree on {field!r}: {meta[field]!r} vs {value!r}. These are not "
                    "two halves of one arm."
                )
            meta[field] = value
    return {"cells": cells, "meta": meta}


def assert_matched_evaluation_provenance(arms: dict) -> None:
    """Reject comparisons assembled across different evaluator code or cohort protocols."""
    for field in ("evaluation_regime", "evaluation_source_fp", "evaluation_protocol_fp"):
        values = {name: arm["meta"].get(field) for name, arm in arms.items()}
        if any(value is None for value in values.values()):
            raise SystemExit(
                f"comparison requires {field} in every enrollment artifact; got {values}. "
                "Re-run evaluation with the current evaluator."
            )
        if len(set(values.values())) != 1:
            raise SystemExit(
                f"arms were produced by different evaluation protocols ({field}): {values}"
            )


def _support_of(key: str) -> int:
    return int(key.rsplit("/", 1)[1].removeprefix("k"))


def _cell_of(key: str) -> str:
    return key.rsplit("/", 1)[0]


def _mode_of(cell: str) -> str:
    """Return the subject relation from legacy or current protocol cell ids.

    Same-subject and cross-subject are different protocols over different query cohorts, so they
    are grouped separately rather than pooled. Pooling would also double-count any dataset whose
    same-subject curve is unsupported: its k=0 plan then falls back to the cross-subject cohort, and
    the identical 2,100 queries would be summed twice.
    """
    for value in reversed(cell.split("/")):
        if value in {"same_subject", "cross_subject"}:
            return value
    raise ValueError(f"cell has no subject relation: {cell}")


def _configuration_of(cell: str) -> str:
    for value in reversed(cell.split("/")):
        if value in {"same_configuration", "cross_configuration"}:
            return value
    return "same_configuration"


def _enrollment_shape_of(cell: str, support: int) -> str:
    for value in reversed(cell.split("/")):
        if value in {"zero", "partial", "full"}:
            return value
    return "zero" if support == 0 else "full"


def candidate_count(result: dict) -> float:
    """Query-weighted mean candidate count, for the chance floors."""
    subjects = result.get("subject_results") or {}
    total = sum(entry["queries"] for entry in subjects.values())
    if not total:
        return float("nan")
    return sum(
        entry["candidate_count"] * entry["queries"] for entry in subjects.values()
    ) / total


def floors(n_candidates: float) -> tuple[float, float]:
    if not math.isfinite(n_candidates) or n_candidates < 1:
        return float("nan"), float("nan")
    return 100.0 / n_candidates, 100.0 * 2.0 / (n_candidates * (n_candidates + 1.0))


def weighted(values: list[tuple[float, int]]) -> float:
    """Query-weighted mean over (value, weight); NaN-valued cells are excluded with their weight."""
    usable = [(v, w) for v, w in values if v is not None and math.isfinite(v)]
    total = sum(w for _, w in usable)
    if not total:
        return float("nan")
    return sum(v * w for v, w in usable) / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="+", action="append", required=True,
                    metavar="NAME PATH", help="arm name followed by its enrollment result files")
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    arms = {}
    for entry in args.arm:
        if len(entry) < 2:
            ap.error(f"--arm needs a name and at least one path, got {entry}")
        name, paths = entry[0], [Path(p) for p in entry[1:]]
        if name in arms:
            ap.error(f"duplicate arm name {name!r}")
        arms[name] = load_arm(paths)

    # Shared controls catch most cohort drift, but not every evaluator change is guaranteed to move
    # a prototype or ridge score. Require the explicit protocol and source identities as well.
    assert_matched_evaluation_provenance(arms)

    # Matched by construction: keep only the cells every arm scored.
    per_arm_cells = [set(arm["cells"]) for arm in arms.values()]
    shared_cells = set.intersection(*per_arm_cells)
    dropped = sorted(set.union(*per_arm_cells) - shared_cells)
    if not shared_cells:
        raise SystemExit("no cell was scored by every arm; nothing is comparable")

    # The support-free controls must be identical across arms. If they are not, the arms did not
    # see the same query cohort and no comparison between them is valid.
    for cell in sorted(shared_cells):
        for column, field in SHARED_METRICS.items():
            values = [arm["cells"][cell].get(field) for arm in arms.values()]
            present = [v for v in values if v is not None]
            if present and max(present) - min(present) > SHARED_TOLERANCE:
                raise SystemExit(
                    f"{column} differs across arms at {cell}: {values}. This control does not use "
                    "the decoder or retriever, so the arms are not scoring the same cohort."
                )

    rows, cell_rows = [], []
    groups = defaultdict(list)
    for label_mode, support, cell in sorted(shared_cells):
        groups[(
            label_mode, support, _mode_of(cell), _configuration_of(cell),
            _enrollment_shape_of(cell, support),
        )].append(cell)

    for (label_mode, support, mode, configuration, enrollment_shape), cells in sorted(
        groups.items()
    ):
        for name, arm in arms.items():
            queries = sum(arm["cells"][(label_mode, support, c)]["queries"] for c in cells)
            row = {
                "label_mode": label_mode, "support_k": support, "mode": mode, "arm": name,
                "configuration": configuration, "enrollment_shape": enrollment_shape,
                "cells": len(cells), "queries": queries,
            }
            for column, field in {**ARM_METRICS, **SHARED_METRICS}.items():
                row[column] = weighted([
                    (arm["cells"][(label_mode, support, c)].get(field),
                     arm["cells"][(label_mode, support, c)]["queries"]) for c in cells
                ])
                row[f"{column}_unweighted"] = weighted([
                    (arm["cells"][(label_mode, support, c)].get(field), 1) for c in cells
                ])
            chance, constant = zip(*[
                floors(candidate_count(arm["cells"][(label_mode, support, c)])) for c in cells
            ])
            counts = [arm["cells"][(label_mode, support, c)]["queries"] for c in cells]
            row["chance_f1"] = weighted(list(zip(chance, counts)))
            row["constant_predictor_f1"] = weighted(list(zip(constant, counts)))
            rows.append(row)

            for c in cells:
                result = arm["cells"][(label_mode, support, c)]
                cell_chance, cell_constant = floors(candidate_count(result))
                cell_rows.append({
                    "label_mode": label_mode, "support_k": support, "mode": mode,
                    "configuration": configuration, "enrollment_shape": enrollment_shape,
                    "arm": name, "cell": c,
                    "queries": result["queries"],
                    "candidates": round(candidate_count(result), 3),
                    **{col: result.get(field)
                       for col, field in {**ARM_METRICS, **SHARED_METRICS}.items()},
                    "chance_f1": cell_chance, "constant_predictor_f1": cell_constant,
                })

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, table in (("_by_cell.csv", cell_rows), (".csv", rows)):
        path = args.out_prefix.with_name(args.out_prefix.name + suffix)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"-> {path}")

    summary_path = args.out_prefix.with_name(args.out_prefix.name + ".json")
    summary_path.write_text(json.dumps({
        "arms": {name: arm["meta"] for name, arm in arms.items()},
        "matched_cells": sorted("/".join((m, f"k{k}", c)) for m, k, c in shared_cells),
        "dropped_cells": ["/".join((m, f"k{k}", c)) for m, k, c in dropped],
        "weighting": "query_weighted_primary_unweighted_reported_alongside",
        "rows": rows,
    }, indent=2) + "\n")
    print(f"-> {summary_path}")

    if dropped:
        print(f"\ndropped {len(dropped)} cell(s) not scored by every arm:")
        for mode, support, cell in dropped:
            print(f"  {mode}/k{support}/{cell}")

    header = ["label_mode", "k", "mode", "config", "shape", "arm", "n", "engine",
              "identity", "prototype", "ridge", "chance", "const"]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join([
            row["label_mode"], str(row["support_k"]), row["mode"], row["configuration"],
            row["enrollment_shape"], row["arm"],
            str(row["queries"]),
            f"{row['full_engine_f1']:.2f}", f"{row['identity_vote_f1']:.2f}",
            f"{row['prototype_f1']:.2f}" if math.isfinite(row["prototype_f1"]) else "n/a",
            f"{row['ridge_f1']:.2f}" if math.isfinite(row["ridge_f1"]) else "n/a",
            f"{row['chance_f1']:.2f}", f"{row['constant_predictor_f1']:.2f}",
        ]) + " |")


if __name__ == "__main__":
    main()
