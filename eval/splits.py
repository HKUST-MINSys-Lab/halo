"""Shared, dataset-stratified subject split manifest (REMEDIATION_PLAN Phase 1.1 — H7/H8).

**The problem this replaces.** Every model used to call `scoring.subject_disjoint_split(S, seed)`
on *its own* aggregate subject array. Two consequences, both measured:

  * **H8 — models disagreed on folds.** harnet excludes gravity-removed streams, which changes the
    subject universe *before* the shuffle, so **16.5% of the 363 shared subjects landed in a
    different train/val/test fold than HALO's.** The two models were therefore selected and
    calibrated on different data, which silently breaks the "same protocol" claim.
  * **H7 — the split was not stratified.** Shuffling the aggregate pool left **HHAR, MHEALTH and
    PAMAP2 with zero validation subjects**, and NFI-FARED with a single val subject carrying a
    single label. Epoch selection and temperature calibration were effectively blind to those
    datasets.

**The fix.** Split subjects **within each dataset**, once, deterministically, and cache the result.
Every model then *looks up* assignments instead of reshuffling — so folds are identical no matter
which streams a given model can consume. A model that must skip a stream simply contributes fewer
windows; it does not move anybody's subject.

Subject ids are ``"dataset:subject"``, matching what the adapters already build.

    python -m eval.splits              # build/refresh data/labels/subject_splits.json
    python -m eval.splits --describe   # show per-dataset fold sizes
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "data" / "labels" / "subject_splits.json"
SPLIT_SEED = 3431                      # matches the historical FIT_SEED
FRACS = (0.8, 0.1, 0.1)                # train / val / test, by SUBJECT within each dataset

# Authors' released workout partition. MM-Fit does not publish the person identity behind every
# workout id, so randomly splitting workout ids cannot honestly be called person-disjoint. Keep
# their train/validation roles and put both published test regimes in test.
MMFIT_SPLIT = {
    "train": frozenset({"w01", "w02", "w03", "w04", "w06", "w07", "w08", "w16", "w17", "w18"}),
    "val": frozenset({"w14", "w15", "w19"}),
    "test": frozenset({"w00", "w05", "w09", "w10", "w11", "w12", "w13", "w20"}),
}


def _split_one_dataset(subjects: Sequence[str], seed: int) -> Dict[str, str]:
    """Assign each subject of ONE dataset to train/val/test. Deterministic given (subjects, seed).

    Guarantees val and test get >=1 subject whenever the cohort has >=3. Cohorts of 1-2 subjects
    cannot be split three ways: they go entirely to train (and are reported), because a 1-subject
    val fold produces a meaningless selection signal — which is exactly the NFI-FARED pathology.
    """
    uniq = sorted(set(subjects))
    rng = np.random.RandomState(seed)
    perm = list(rng.permutation(uniq))
    n = len(perm)
    if n < 3:
        return {s: "train" for s in perm}
    n_val = max(1, int(n * FRACS[1]))
    n_test = max(1, int(n * FRACS[2]))
    n_train = max(1, n - n_val - n_test)
    out = {}
    for s in perm[:n_train]:
        out[s] = "train"
    for s in perm[n_train:n_train + n_val]:
        out[s] = "val"
    for s in perm[n_train + n_val:]:
        out[s] = "test"
    return out


# Datasets that are NOT in HALO's TRAIN_DATASETS but ARE in some baseline's head-fit corpus.
# They must still be in the manifest, or their subjects fall through to the train fold and the
# model's val/test folds are silently built from a different population than everyone else's.
# The four sources retired from training on 2026-09-10 belong here too: their subjects are still
# in the checked-in manifest and in the frozen 18-source baseline corpus, and a regeneration that
# dropped them would let those subjects fall through to "train" for any model that still scores
# on the old roster.
EXTRA_MANIFEST_DATASETS = ("hapt", "uci_har", "sp_sw_har", "mhealth", "opportunity")

# Cohorts that are THE SAME PEOPLE under different dataset names. Both members must land in the
# same fold or a subject's data appears on both sides of the split.
#
# hapt is the extended re-release of UCI-HAR: identical 30-participant cohort, `user01..user30`
# against `subject01..subject30`, and `baselines/crosshar/prep.py` measures per-window NCC 0.98
# between them. harnet's DEFAULT (legacy) corpus includes hapt while the manifest covered only
# uci_har, so all 30 hapt subjects fell through to train while their uci_har counterparts were
# harnet's val (epoch selection) and test (temperature calibration) folds -- an optimistic bias on
# the strongest baseline in the table.
ALIASED_COHORTS: Dict[str, Tuple[str, str, str]] = {
    # alias_dataset: (canonical_dataset, alias_subject_prefix, canonical_subject_prefix)
    "hapt": ("uci_har", "user", "subject"),
}


def _canonical_key(dataset: str, subject: str) -> str:
    """Map an aliased cohort member onto the canonical subject it duplicates."""
    alias = ALIASED_COHORTS.get(dataset)
    if alias is None:
        return f"{dataset}:{subject}"
    canon_ds, a_pre, c_pre = alias
    if subject.startswith(a_pre):
        return f"{canon_ds}:{c_pre}{subject[len(a_pre):]}"
    return f"{dataset}:{subject}"


def build_manifest(seed: int = SPLIT_SEED) -> dict:
    """Per-dataset stratified subject assignment over the whole training corpus."""
    from data.scripts.eda.grid_io import discover_grids
    from training.tokenizer.pretrain_data import TRAIN_DATASETS

    wanted = set(TRAIN_DATASETS) | set(EXTRA_MANIFEST_DATASETS)
    per_dataset: Dict[str, List[str]] = {}
    aliased: Dict[str, List[str]] = {}
    for ref in discover_grids("native"):
        if ref.dataset not in wanted:
            continue
        # An aliased dataset does not get its own split -- it inherits the canonical cohort's.
        bucket = aliased if ref.dataset in ALIASED_COHORTS else per_dataset
        bucket.setdefault(ref.dataset, [])
        bucket[ref.dataset].extend(str(s) for s in ref.subjects)

    assignment: Dict[str, str] = {}
    summary: Dict[str, Dict[str, int]] = {}
    tiny: List[str] = []
    for ds in sorted(per_dataset):
        # seed per dataset so adding a dataset never reshuffles the others
        ds_seed = seed + int(hashlib.sha256(ds.encode()).hexdigest()[:8], 16) % 100000
        if ds == "mmfit":
            present = set(per_dataset[ds])
            published = set().union(*MMFIT_SPLIT.values())
            if present != published:
                raise RuntimeError(
                    f"MM-Fit workout roster differs from the published split: "
                    f"missing={sorted(published - present)}, extra={sorted(present - published)}"
                )
            assign = {workout: fold for fold, workouts in MMFIT_SPLIT.items()
                      for workout in workouts}
        else:
            assign = _split_one_dataset(per_dataset[ds], ds_seed)
        counts = {"train": 0, "val": 0, "test": 0}
        for subj, fold in assign.items():
            assignment[f"{ds}:{subj}"] = fold
            counts[fold] += 1
        summary[ds] = counts
        if counts["val"] == 0:
            tiny.append(ds)

    # Aliased cohorts inherit their canonical twin's fold, so the same person is never split.
    alias_summary: Dict[str, Dict[str, int]] = {}
    for ds in sorted(aliased):
        counts = {"train": 0, "val": 0, "test": 0}
        for subj in sorted(set(aliased[ds])):
            canon = _canonical_key(ds, subj)
            fold = assignment.get(canon)
            if fold is None:
                raise RuntimeError(
                    f"{ds}:{subj} aliases {canon}, which is not in the manifest. Either the "
                    f"canonical dataset is missing from TRAIN_DATASETS or the subject-prefix "
                    f"mapping in ALIASED_COHORTS is wrong.")
            assignment[f"{ds}:{subj}"] = fold
            counts[fold] += 1
        alias_summary[ds] = counts
        summary[ds] = counts

    return {"seed": seed, "fracs": list(FRACS), "assignment": assignment,
            "per_dataset": summary, "datasets_without_val": tiny,
            "aliased": {k: list(ALIASED_COHORTS[k]) for k in alias_summary},
            "n_subjects": len(assignment)}


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"subject split manifest missing at {MANIFEST_PATH}. Build it:\n"
            f"    python -m eval.splits")
    return json.loads(MANIFEST_PATH.read_text())


def manifest_fingerprint(manifest: dict | None = None) -> str:
    m = manifest or load_manifest()
    return hashlib.sha256(json.dumps(m["assignment"], sort_keys=True).encode()).hexdigest()[:16]


def split_indices(subject_ids: Sequence[str],
                  manifest: dict | None = None,
                  allow_unknown: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map an array of ``"dataset:subject"`` ids to (train_idx, val_idx, test_idx).

    **Unknown subjects raise.** The previous behaviour — silently assigning them to TRAIN — was
    documented as "reported by the caller if it cares", and no caller ever did. Two real failures
    came out of that silence:

      * harnet's default corpus contains ``hapt``, which was absent from the manifest, so 30
        subjects got fabricated fold membership (now fixed by ``EXTRA_MANIFEST_DATASETS``);
      * a *total* miss (e.g. a subject-id format change upstream) degrades with no error at all:
        empty val fold -> ``np.mean([])`` = nan -> ``nan > best_acc`` is never True -> ``best_sd``
        stays None -> the adapters' ``if best_sd is not None`` guard keeps the LAST epoch, and
        Phase 1.3 then calibrates temperature on an empty tensor. Nothing raises anywhere.

    ``allow_unknown=True`` restores the old lenient behaviour for callers that genuinely score
    data outside the training corpus; it still routes unknowns to TRAIN, never to val/test.
    """
    m = manifest or load_manifest()
    a = m["assignment"]
    ids = [str(s) for s in subject_ids]
    missing = sorted({s for s in ids if s not in a})
    if missing and not allow_unknown:
        by_ds: Dict[str, int] = {}
        for s in missing:
            by_ds[s.split(":", 1)[0]] = by_ds.get(s.split(":", 1)[0], 0) + 1
        raise KeyError(
            f"{len(missing)} of {len(ids)} subject ids are absent from the split manifest "
            f"{MANIFEST_PATH.name}: " + ", ".join(f"{d} x{n}" for d, n in sorted(by_ds.items()))
            + f" (e.g. {missing[:3]}). These would silently land in the TRAIN fold, so val/test "
            f"would be built from a different population than every other model's. Fix: add the "
            f"dataset to TRAIN_DATASETS or EXTRA_MANIFEST_DATASETS in eval/splits.py and rebuild "
            f"with `python -m eval.splits`, or pass allow_unknown=True if that is truly intended.")
    folds = np.array([a.get(s, "train") for s in ids])
    tr, va, te = (np.where(folds == "train")[0], np.where(folds == "val")[0],
                  np.where(folds == "test")[0])
    if len(va) == 0 or len(te) == 0:
        raise ValueError(
            f"split produced an empty fold (train={len(tr)}, val={len(va)}, test={len(te)}) over "
            f"{len(set(ids))} unique subjects. Epoch selection and temperature calibration both "
            f"need a non-empty fold; continuing would silently keep the last epoch and calibrate "
            f"on nothing.")
    return tr, va, te


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--describe", action="store_true", help="print the manifest, write nothing")
    args = ap.parse_args()

    m = build_manifest()
    print(f"[splits] {m['n_subjects']} subjects over {len(m['per_dataset'])} datasets "
          f"(seed {m['seed']}, per-dataset stratified {FRACS})")
    print(f"{'dataset':16}{'train':>7}{'val':>6}{'test':>6}")
    for ds, c in sorted(m["per_dataset"].items()):
        flag = "   <-- cohort too small to split" if c["val"] == 0 else ""
        print(f"{ds:16}{c['train']:>7}{c['val']:>6}{c['test']:>6}{flag}")
    if m["datasets_without_val"]:
        print(f"[splits] NOTE: no val subjects for {m['datasets_without_val']} "
              f"(cohort < 3 subjects; all windows go to train)")
    if args.describe:
        print("[splits] --describe: nothing written")
        return
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    print(f"[splits] wrote {MANIFEST_PATH}  fp={manifest_fingerprint(m)}")
    print("[splits] !! every cached ConSE head is now stale (split changed) and must be refit")


if __name__ == "__main__":
    main()
