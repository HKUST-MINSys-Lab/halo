"""Print the corpus acquisition-key table and measure support-pool viability.

This is the artifact the user reviews by hand before the support sampler is trusted (handoff W1).
It answers three questions that decide whether the core arm is feasible at all:

1. Does every curated placement map to a site? (An unmapped one is a hard error.)
2. How large is the compatible pool for a typical training query — in streams, recordings and
   windows? The sampler needs K support rows drawn from recordings that are not the query's own
   and not its subject's, so a pool of one stream is a problem.
3. Does every evaluation stream have a compatible *training* partner? The zero-shot row draws its
   candidate-excluded support from the training corpus, so a key with no training partner leaves
   that row undefined. That is a finding to report, not something to paper over.

Run::

    python -m data.scripts.curate.audit_compatibility
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from data.scripts.curate import deployment_policy
from data.scripts.curate.compatibility import (
    AcquisitionKey,
    are_compatible,
    corpus_keys,
    is_near_miss,
)

REPO = Path(__file__).resolve().parents[3]
GRIDS = REPO / "data" / "datasets"


def window_count(dataset: str, stream: str, alignment: str = "native") -> int | None:
    """Windows in one grid, read from meta only. ``None`` when the grid is absent."""

    meta = GRIDS / dataset / "grids" / alignment / stream / "meta.json"
    if not meta.exists():
        return None
    try:
        payload = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return None
    labels = payload.get("labels")
    return len(labels) if isinstance(labels, list) else None


def _key_text(key: AcquisitionKey) -> str:
    channels = "acc+gyro" if len(key.channels) > 3 else "acc"
    gravity = "g-" if key.gravity_state == "removed" else "g+"
    return f"{key.device_family}/{key.site}/{channels}/{gravity}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", default="native")
    args = parser.parse_args()

    train = list(deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS)
    # Keep the audit tied to the live protocol roster.  A previous hard-coded seven-dataset list
    # silently omitted MotionSense, RealWorld and Shoaib after adaptation_v2 restored them.
    evaluation = list(deployment_policy.PRIMARY_EVAL_DATASETS)

    keys = corpus_keys()
    print(f"curated streams: {len(keys)}   distinct keys: {len(set(keys.values()))}\n")

    # ---------------------------------------------------------------- the table
    print("=" * 100)
    print("STREAM -> ACQUISITION KEY")
    print("=" * 100)
    print(f"{'role':<6}{'dataset':<16}{'stream':<24}{'key':<44}{'windows':>9}")
    for (dataset, stream), key in sorted(keys.items()):
        role = "train" if dataset in train else ("EVAL" if dataset in evaluation else "-")
        count = window_count(dataset, stream, args.alignment)
        shown = f"{count:,}" if count is not None else "-"
        print(f"{role:<6}{dataset:<16}{stream:<24}{_key_text(key):<44}{shown:>9}")

    # ------------------------------------------------- training pools by key
    train_pool: dict[AcquisitionKey, list[tuple[str, str, int]]] = defaultdict(list)
    for (dataset, stream), key in keys.items():
        if dataset not in train:
            continue
        count = window_count(dataset, stream, args.alignment)
        if count:
            train_pool[key].append((dataset, stream, count))

    print("\n" + "=" * 100)
    print("TRAINING POOLS (a query with this key draws its compatible support from here)")
    print("=" * 100)
    print(f"{'key':<44}{'streams':>8}{'datasets':>10}{'windows':>12}")
    singleton_dataset = 0
    for key, members in sorted(train_pool.items(), key=lambda kv: -sum(m[2] for m in kv[1])):
        datasets = {member[0] for member in members}
        windows = sum(member[2] for member in members)
        if len(datasets) < 2:
            singleton_dataset += 1
        print(f"{_key_text(key):<44}{len(members):>8}{len(datasets):>10}{windows:>12,}")
    print(
        f"\ntraining keys: {len(train_pool)}   "
        f"keys whose pool is a single dataset: {singleton_dataset}"
    )
    print(
        "A single-dataset pool still trains, but every support row then shares the query's "
        "dataset, so the episode cannot reward cross-dataset comparison."
    )

    # ----------------------------------------- evaluation coverage (open question 1)
    print("\n" + "=" * 100)
    print("EVALUATION STREAM COVERAGE (zero-shot support is drawn from TRAINING)")
    print("=" * 100)
    uncovered = []
    for (dataset, stream), key in sorted(keys.items()):
        if dataset not in evaluation:
            continue
        exact = [
            (d, s) for (d, s), other in keys.items()
            if d in train and are_compatible(key, other)
        ]
        near = [
            (d, s) for (d, s), other in keys.items()
            if d in train and is_near_miss(key, other)
        ]
        status = "ok" if exact else ("NEAR-MISS ONLY" if near else "NONE")
        if not exact:
            uncovered.append((dataset, stream, status))
        print(
            f"{dataset:<16}{stream:<24}{_key_text(key):<44}"
            f"exact={len(exact):<3} near={len(near):<3} {status}"
        )
    if uncovered:
        print("\nFINDING — these evaluation streams have no exactly compatible training stream:")
        for dataset, stream, status in uncovered:
            print(f"  {dataset}/{stream}: {status}")
        print(
            "The k=0 row for these streams is undefined under the agreed mechanism. Report this "
            "to the user; do not invent a fallback."
        )
    else:
        print("\nEvery evaluation stream has at least one exactly compatible training stream.")

    # --------------------------------------------------- near-miss availability (Arm B2)
    print("\n" + "=" * 100)
    print("NEAR-MISS AVAILABILITY (Arm B2 needs same-family, equivalent-site, non-identical pairs)")
    print("=" * 100)
    pairs = 0
    for key in train_pool:
        partners = [other for other in train_pool if is_near_miss(key, other)]
        if partners:
            pairs += 1
            print(f"{_key_text(key):<44}-> {len(partners)} near-miss key(s)")
    print(f"\ntraining keys with at least one near-miss partner: {pairs}/{len(train_pool)}")
    if pairs == 0:
        print("Arm B2 cannot be run on this corpus. Report it rather than widening the relation.")


if __name__ == "__main__":
    main()
