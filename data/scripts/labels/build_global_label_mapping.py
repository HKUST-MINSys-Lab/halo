#!/usr/bin/env python3
"""Global canonical vocabulary for support-conditioned training and historical ConSE controls.

The support classifier and historical ConSE controls key on ONE closed vocabulary: the set of every
supervised training dataset's activities. We build it from the **canonical** labels
(`data.scripts.labels.canonical_labels`) so a synonym never occupies two output classes.

SOURCE OF TRUTH = THE GRIDS, NOT metadata.json.
Earlier versions read ``metadata.json["activities"]`` over a hardcoded 9-dataset list. That was wrong
twice over: the list was stale (it still contained `hapt` and predated sp_sw_har / nfi_fared / harmes /
xrf_v2), and four sources in that historical 12-source roster did not declare `activities`. The result
was a 59-label vocabulary while the grids actually contain 93 canonical labels — so **11.5% of all
training windows (39,413 / 343,235) were silently dropped** as "out of vocabulary" by historical
ConSE head fitting, concentrated in the fine-grained ADLs from the newer
datasets. Notably `elevator_up` / `elevator_down` existed in the corpus but were discarded, while
usc_had's elevator classes were being reported as ~0 F1.

Deriving from the grid labels cannot drift from what the models actually see. metadata.json is still
cross-checked and any disagreement is reported, but it is never the source.

Regenerate whenever the supervised training datasets or their grids change. The vocabulary defines
historical ConSE label indices, so cached ConSE heads must be rebuilt afterwards. The adapters compare
their cached `labels` against this file and refit automatically on mismatch. The current HALO support
classifier derives its candidates directly from the episode corpus and does not consume this file.

    python -m data.scripts.labels.build_global_label_mapping           # writes data/labels/global_labels.json
    python -m data.scripts.labels.build_global_label_mapping --dry-run # show the diff, write nothing
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

from data.scripts.labels.canonical_labels import NON_SEMANTIC_LABELS, canonicalize

REPO = Path(__file__).resolve().parents[3]
OUT_PATH = REPO / "data" / "labels" / "global_labels.json"


def _train_datasets() -> List[str]:
    """The canonical training-dataset list, imported lazily from the training pipeline so this
    script can never drift from what actually gets trained on."""
    from training.tokenizer.pretrain_data import TRAIN_DATASETS
    return sorted(TRAIN_DATASETS)


def global_label_vocabulary(train_datasets: Optional[List[str]] = None,
                            alignment: str = "native"):
    """(sorted canonical vocabulary, per-label window counts) taken from the GRIDS."""
    from data.scripts.eda.grid_io import discover_grids
    datasets = set(train_datasets or _train_datasets())
    counts: Counter = Counter()
    for ref in discover_grids(alignment):
        if ref.dataset not in datasets:
            continue
        for raw in ref.labels:
            label = canonicalize(raw)
            if label not in NON_SEMANTIC_LABELS:
                counts[label] += 1
    return sorted(counts), counts


def _metadata_declared(datasets: List[str]):
    """Canonical labels declared in metadata.json, plus the datasets that declare none."""
    declared, missing = set(), []
    for ds in datasets:
        p = REPO / "data" / "datasets" / ds / "metadata.json"
        acts = json.loads(p.read_text()).get("activities") if p.exists() else None
        if not acts:
            missing.append(ds)
            continue
        declared.update(
            canonicalize(a) for a in acts if canonicalize(a) not in NON_SEMANTIC_LABELS
        )
    return declared, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the diff, write nothing")
    ap.add_argument("--alignment", default="native")
    args = ap.parse_args()

    datasets = _train_datasets()
    vocab, counts = global_label_vocabulary(datasets, args.alignment)
    total = sum(counts.values())
    print(f"[vocab] {len(datasets)} training datasets: {datasets}")
    print(f"[vocab] {len(vocab)} canonical labels over {total} windows ({args.alignment} grids)")

    # cross-check against metadata.json (informational only — grids win)
    declared, missing = _metadata_declared(datasets)
    if missing:
        print(f"[vocab] NOTE: no 'activities' declared in metadata.json for: {missing} "
              f"(harmless — the vocabulary comes from the grids)")
    only_meta = sorted(declared - set(vocab))
    if only_meta:
        print(f"[vocab] NOTE: declared in metadata but absent from grids: {only_meta}")

    prev = json.loads(OUT_PATH.read_text())["labels"] if OUT_PATH.exists() else []
    added, removed = sorted(set(vocab) - set(prev)), sorted(set(prev) - set(vocab))
    if prev:
        unlocked = sum(counts[l] for l in added)
        print(f"[vocab] previous vocabulary: {len(prev)} labels")
        print(f"[vocab]   + {len(added)} ADDED   (windows they unlock: {unlocked} = "
              f"{100 * unlocked / max(1, total):.2f}%)")
        for l in added:
            print(f"[vocab]        + {l:34} {counts[l]:>7} windows")
        print(f"[vocab]   - {len(removed)} REMOVED (not present in any grid): {removed}")

    if args.dry_run:
        print("[vocab] --dry-run: nothing written")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "description": ("Canonical supervised-training vocabulary for historical ConSE controls. "
                        f"Sorted union of canonicalized labels observed in the {args.alignment} GRIDS "
                        f"of {len(datasets)} training datasets — derived from the data, never from "
                        "metadata.json. Generated by data.scripts.labels.build_global_label_mapping "
                        "— do not hand-edit. Regenerating invalidates every cached ConSE head; "
                        "adapters must rebuild it before use."),
        "labels": vocab,
        "num_labels": len(vocab),
        "train_datasets": datasets,
        "alignment": args.alignment,
        "window_counts": {l: counts[l] for l in vocab},
    }, indent=2) + "\n")
    print(f"[vocab] wrote {OUT_PATH} — {len(vocab)} canonical labels")
    print("[vocab] cached ConSE heads are now stale; adapters must validate/rebuild before use")


if __name__ == "__main__":
    main()
