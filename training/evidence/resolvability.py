"""``resolvability`` — can a given acquisition configuration witness a given concept?

THE MEASUREMENT THE CONTRIBUTION RESTS ON (docs/design/DESIGN_OF_RECORD.md). The admissibility gate
learns whether one physical sensor can witness one candidate concept. Runtime combines the query-side
and evidence-side sensor scores; this module supplies the train-only per-sensor measurements.

THE CLAIM, STATED SO IT CAN FAIL
--------------------------------
A phone in a pocket is strong evidence about gait and none at all about what the arms are doing. If
that is true, then for a fixed concept the discriminability of that concept should vary SHARPLY and
SYSTEMATICALLY with sensor placement — and it should do so on the SAME physical events, not merely
across datasets with different subjects and protocols.

Two estimators, in increasing order of how much they prove:

1. Per-sensor resolvability — for every (stream, modality, label), how well does that physical sensor
   separate the label from the rest of its own protocol? One-vs-rest, subject-disjoint kNN, rescaled
   so 0 = chance and 1 = perfect.

2. ``paired_contrast`` — restricted to active datasets where several streams record the SAME
   sessions simultaneously (xrf_v2, realdisp, mmfit, DSADS, FORTH-TRACE, and NFI-FARED). Same
   subjects, same events, same clock: the ONLY thing that differs is where the sensor sits. A large
   per-label spread across simultaneous streams is the jumping-jacks effect, measured. A small spread
   falsifies the premise, and would mean the admissibility gate has nothing to gate on.

Estimator 2 is the one that can kill the thesis, which is why it is reported separately rather than
averaged into estimator 1.

WHY ONE-VS-REST AND NOT ACCURACY
--------------------------------
A stream's overall accuracy conflates "this configuration is good" with "this protocol is easy".
Resolvability is a property of a (configuration, concept) PAIR, so each label is scored against its
own chance level within its own stream, and the score is the excess over chance.

Run:
    python -m training.evidence.resolvability --build --checkpoint <phase_a>/best.pt
    python -m training.evidence.resolvability --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

OUT_PATH = Path("training/evidence/outputs/resolvability.json")

# Datasets whose streams are simultaneous views of the same sessions. Only these support the
# paired contrast; everywhere else "different placement" is confounded with different subjects and
# protocols, which is exactly the confound the contrast exists to remove.
PAIRED_DATASETS = ("xrf_v2", "realdisp", "mmfit", "dsads", "forth_trace", "nfi_fared")

MAX_WINDOWS_PER_STREAM = 3000
MIN_WINDOWS_PER_LABEL = 20
MIN_SUBJECTS = 2
KNN_K = 5
SEED = 20260812


def _one_vs_rest_resolvability(
    z: np.ndarray, labels: np.ndarray, subjects: np.ndarray, k: int = KNN_K,
) -> dict[str, float]:
    """Per-label excess-over-chance separability, subject-disjoint.

    For each label: hold out half the subjects, ask a kNN over the remaining half whether a query
    belongs to this label or not, and score balanced accuracy. Rescaled ``2*(ba - 0.5)`` so 0 means
    "this configuration cannot tell this concept from the rest of its protocol" and 1 means perfect.
    Clipped at 0: a below-chance estimate is noise, not negative information.
    """
    rng = np.random.default_rng(SEED)
    subject_list = sorted(set(subjects.tolist()))
    if len(subject_list) < MIN_SUBJECTS:
        return {}
    rng.shuffle(subject_list)
    held = set(subject_list[: max(1, len(subject_list) // 2)])
    train_idx = np.array([i for i in range(len(z)) if subjects[i] not in held])
    test_idx = np.array([i for i in range(len(z)) if subjects[i] in held])
    if train_idx.size == 0 or test_idx.size == 0:
        return {}

    zt = torch.from_numpy(z[train_idx]).float()
    zq = torch.from_numpy(z[test_idx]).float()
    zt = torch.nn.functional.normalize(zt, dim=1)
    zq = torch.nn.functional.normalize(zq, dim=1)
    sim = zq @ zt.t()                                        # (Q, T) cosine
    kk = min(k, zt.shape[0])
    nn_idx = sim.topk(kk, dim=1).indices.numpy()

    out: dict[str, float] = {}
    for label in sorted(set(labels.tolist())):
        pos_train = (labels[train_idx] == label)
        pos_test = (labels[test_idx] == label)
        if pos_test.sum() < MIN_WINDOWS_PER_LABEL or pos_train.sum() < MIN_WINDOWS_PER_LABEL:
            continue
        if (~pos_test).sum() == 0:
            continue
        vote = pos_train[nn_idx].mean(axis=1) > 0.5          # kNN says "this label"
        tpr = float(vote[pos_test].mean())
        tnr = float((~vote[~pos_test]).mean())
        ba = 0.5 * (tpr + tnr)
        out[str(label)] = round(max(0.0, 2.0 * (ba - 0.5)), 4)
    return out


def _pool_sensor_windows(encoded: dict, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Duration-pool one sensor's patch rows to one vector per source window.

    Multi-resolution rows are pooled within each resolution first, then the available resolutions are
    averaged. This mirrors the encoder's equal treatment of resolutions and prevents the denser patch
    grid from receiving more weight merely because it emits more rows.
    """
    row = encoded["sensor_slot"].long().eq(int(slot))
    if not bool(row.any()):
        return torch.empty((0, encoded["sensor_Z"].shape[-1])), torch.empty(0, dtype=torch.long)
    z = encoded["sensor_Z"][row].float().cpu()
    window = encoded["sensor_window"][row].long().cpu()
    duration = encoded["sensor_duration"][row].float().cpu()
    resolution = encoded["sensor_resolution"][row].long().cpu()
    pooled, windows = [], []
    for source_window in window.unique(sorted=True).tolist():
        per_resolution = []
        in_window = window.eq(source_window)
        for value in resolution[in_window].unique(sorted=True).tolist():
            selected = in_window & resolution.eq(value)
            weight = duration[selected].clamp_min(1e-8)
            per_resolution.append((z[selected] * weight.unsqueeze(1)).sum(0) / weight.sum())
        pooled.append(torch.stack(per_resolution).mean(0))
        windows.append(source_window)
    return torch.stack(pooled), torch.tensor(windows, dtype=torch.long)


def build(checkpoint: Path, device: str = "cuda", limit_streams: int | None = None) -> dict:
    """Measure per-(sensor, concept) resolvability on Phase-A TRAINING subjects only."""
    from data.scripts.eda.grid_io import discover_grids
    from data.scripts.scan_duplicates import load as load_duplicates
    from data.scripts.scan_implausible import load as load_implausible
    from training.tokenizer.eval_transfer import build_encoder, encode_dataset_detailed
    from training.tokenizer.pretrain_data import (
        SEED as PHASE_A_SEED,
        _stream_gravity_state,
        modalities_present,
        stream_channel_descriptions,
        stream_sensor_texts,
        validation_subjects_for_refs,
    )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, dev)
    if getattr(enc, "token_granularity", "channel") != "sensor":
        raise ValueError(
            "resolvability requires a sensor-granularity Phase-A checkpoint; pooled channel-era "
            "embeddings cannot supervise a per-sensor admissibility gate"
        )
    print(f"encoder: {checkpoint} step {ckpt.get('step')} git {ckpt.get('git')}", flush=True)

    roster = tuple(ckpt.get("config", {}).get("train_datasets") or ())
    if not roster:
        raise ValueError(
            "checkpoint does not record config.train_datasets; refusing to guess a roster for a "
            "training-derived admissibility artifact"
        )
    all_refs = sorted(
        (ref for ref in discover_grids("native") if ref.dataset in set(roster)),
        key=lambda r: (r.dataset, r.stream),
    )
    represented = {ref.dataset for ref in all_refs if ref.n_windows > 0}
    missing = sorted(set(roster) - represented)
    if missing:
        raise FileNotFoundError(f"checkpoint training datasets have no native grids: {missing}")
    # Derive the Phase-A subject split from the full checkpoint roster. ``limit_streams`` is a debug
    # work cap only and must not silently change which subjects count as training observations.
    val_subjects = validation_subjects_for_refs(all_refs, seed=PHASE_A_SEED)
    refs = all_refs[:limit_streams] if limit_streams else all_refs
    val_payload = "\n".join(f"{d}\t{s}" for d, s in sorted(val_subjects))
    val_hash = hashlib.sha256(val_payload.encode()).hexdigest()
    bad_by_stream = load_implausible("native", require=True)
    duplicate_by_stream = load_duplicates("native", require=True)

    def eligible_rows(ref) -> np.ndarray:
        subjects = np.asarray(ref.subjects)
        excluded = bad_by_stream.get(ref.key, set()) | duplicate_by_stream.get(ref.key, set())
        stale = sorted(index for index in excluded if index >= ref.n_windows)
        if stale:
            raise RuntimeError(
                f"{ref.key}: quality cache indexes window {stale[-1]} but the grid has "
                f"{ref.n_windows}; rebuild the quality scans"
            )
        return np.asarray([
            i for i, subject in enumerate(subjects)
            if (ref.dataset, str(subject)) not in val_subjects and i not in excluded
        ], dtype=np.int64)

    eligible_by_key = {ref.key: eligible_rows(ref) for ref in all_refs}
    paired_event_subset: dict[str, set[str]] = {}
    for dataset in PAIRED_DATASETS:
        dataset_refs = [ref for ref in all_refs if ref.dataset == dataset]
        if len(dataset_refs) < 2:
            continue
        if not all(ref.event_ids_explicit for ref in dataset_refs):
            raise ValueError(
                f"{dataset}: paired contrast requires explicit shared event identifiers"
            )
        event_maps = []
        for ref in dataset_refs:
            rows = eligible_by_key[ref.key]
            events = np.asarray(ref.event_ids, dtype=object)
            labels = np.asarray(ref.labels, dtype=object)
            subjects = np.asarray(ref.subjects, dtype=object)
            event_maps.append({
                str(events[index]): (str(labels[index]), str(subjects[index])) for index in rows
            })
        common = set.intersection(*(set(mapping) for mapping in event_maps))
        if not common:
            raise ValueError(f"{dataset}: paired streams have no shared eligible event identifiers")
        for event in common:
            signatures = {mapping[event] for mapping in event_maps}
            if len(signatures) != 1:
                raise ValueError(
                    f"{dataset}: simultaneous event {event!r} has inconsistent "
                    f"(label, subject) metadata {sorted(signatures)}"
                )
        ordered = np.asarray(sorted(common), dtype=object)
        if len(ordered) > MAX_WINDOWS_PER_STREAM:
            event_seed = int(hashlib.sha256(
                f"{SEED}:{dataset}:paired-events".encode()
            ).hexdigest()[:16], 16)
            selected = np.random.default_rng(event_seed).choice(
                ordered, MAX_WINDOWS_PER_STREAM, replace=False
            )
            ordered = np.sort(selected)
        paired_event_subset[dataset] = set(map(str, ordered.tolist()))

    per_sensor: dict[str, dict] = {}
    for ref in refs:
        labels_all = np.asarray(ref.labels)
        subjects_all = np.asarray(ref.subjects)
        eligible = eligible_by_key[ref.key]
        if ref.dataset in paired_event_subset:
            events = np.asarray(ref.event_ids, dtype=object)
            selected_events = paired_event_subset[ref.dataset]
            eligible = eligible[
                np.asarray([str(events[index]) in selected_events for index in eligible], dtype=bool)
            ]
        elif len(eligible) > MAX_WINDOWS_PER_STREAM:
            stream_seed = int(hashlib.sha256(
                f"{SEED}:{ref.key}".encode()
            ).hexdigest()[:16], 16)
            eligible = np.sort(np.random.default_rng(stream_seed).choice(
                eligible, MAX_WINDOWS_PER_STREAM, replace=False
            ))
        data = np.asarray(ref.load_data()[eligible])
        labels = labels_all[eligible]
        subjects = subjects_all[eligible]
        if len(set(subjects.tolist())) < MIN_SUBJECTS:
            print(f"  skip {ref.dataset}/{ref.stream}: <{MIN_SUBJECTS} subjects", flush=True)
            continue
        try:
            encoded = encode_dataset_detailed(
                enc, data, stream_channel_descriptions(ref.dataset, ref.stream),
                dev, ref.rate_hz, _stream_gravity_state(ref.dataset, ref.stream),
                channel_mask=ref.mask, dataset=ref.dataset, stream=ref.stream,
                lengths=np.asarray(ref.load_lengths())[eligible],
                export_sensor_rows=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to encode required resolvability stream {ref.dataset}/{ref.stream}"
            ) from exc
        modalities = modalities_present(ref.mask)
        _, sensor_texts, _ = stream_sensor_texts(
            ref.dataset, ref.stream,
            has_accel="accel" in modalities, has_gyro="gyro" in modalities,
        )
        for slot, (modality, sensor_text) in enumerate(zip(modalities, sensor_texts, strict=True)):
            z, source_window = _pool_sensor_windows(encoded, slot)
            scores = _one_vs_rest_resolvability(
                z.numpy(), labels[source_window], subjects[source_window]
            )
            if not scores:
                continue
            key = f"{ref.dataset}/{ref.stream}::{modality}"
            per_sensor[key] = {
                "dataset": ref.dataset, "stream": ref.stream, "slot": slot,
                "modality": modality, "sensor_text": sensor_text,
                "n_windows": int(len(source_window)), "labels": scores,
            }
            mean = float(np.mean(list(scores.values())))
            print(f"  {key}: {len(scores)} labels, mean resolvability {mean:.3f}", flush=True)

    return {
        "schema_version": 2,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": ckpt.get("step"),
        "training_datasets": sorted(roster),
        "phase_a_data_seed": PHASE_A_SEED,
        "validation_subjects_sha256": val_hash,
        "scope": "phase_a_training_subjects_only_per_sensor",
        "paired_event_counts": {
            dataset: len(events) for dataset, events in sorted(paired_event_subset.items())
        },
        "per_sensor": per_sensor,
        "paired_contrast": paired_contrast(per_sensor),
    }


def paired_contrast(per_sensor: dict) -> dict:
    """Per-label spread across SIMULTANEOUS streams — the falsifiable half.

    Restricted to datasets whose streams record the same sessions at the same time, so subjects,
    protocol and clock are held fixed and placement is the only variable. Reports, per label, the
    best and worst configuration and the gap between them.

    A large mean gap is the jumping-jacks effect measured. A small one says placement barely matters
    for concept identifiability, which would leave the admissibility gate with nothing to gate on —
    a result that must be reported, not smoothed away.
    """
    out: dict[str, dict] = {}
    for dataset in PAIRED_DATASETS:
        sensors = {k: v for k, v in per_sensor.items() if v["dataset"] == dataset}
        for modality in ("accel", "gyro"):
            streams = {k: v for k, v in sensors.items() if v["modality"] == modality}
            if len(streams) < 2:
                continue
            by_label: dict[str, dict[str, float]] = defaultdict(dict)
            for key, payload in streams.items():
                for label, score in payload["labels"].items():
                    by_label[label][payload["stream"]] = score
            rows = {}
            for label, scores in by_label.items():
                if len(scores) < 2:
                    continue                                 # not observed by 2+ simultaneous streams
                best = max(scores, key=scores.get)
                worst = min(scores, key=scores.get)
                rows[label] = {
                    "best_stream": best, "best": scores[best],
                    "worst_stream": worst, "worst": scores[worst],
                    "gap": round(scores[best] - scores[worst], 4),
                    "n_streams": len(scores),
                }
            if rows:
                gaps = [r["gap"] for r in rows.values()]
                out[f"{dataset}::{modality}"] = {
                    "dataset": dataset, "modality": modality,
                    "n_labels": len(rows),
                    "mean_gap": round(float(np.mean(gaps)), 4),
                    "max_gap": round(float(np.max(gaps)), 4),
                    "concept_dependence": _concept_dependence(by_label),
                    "labels": rows,
                }
    return out


def _concept_dependence(by_label: dict[str, dict[str, float]]) -> dict:
    """Does the placement ranking DEPEND on the concept, or is it a fixed quality ordering?

    THE SHARPEST CLAIM IN THE DESIGN. "Some placements are just better sensors" would be a boring
    finding and would not need an admissibility gate — a single per-placement weight would do. The
    contribution requires that a placement good for one concept is bad for another: a pocket phone
    witnesses squats and not bicep curls, a wrist the reverse.

    Measured as the mean Pearson correlation between streams' resolvability profiles across labels.
      * correlation near +1 -> a fixed quality ordering; a scalar per placement would suffice, and
        the (config, config, CONCEPT) structure is unnecessary.
      * correlation near 0 or negative -> which placement wins depends on the concept, and the gate
        genuinely needs its third argument.
    """
    streams = sorted({s for scores in by_label.values() for s in scores})
    labels = [l for l, scores in by_label.items() if len(scores) >= 2]
    if len(streams) < 2 or len(labels) < 3:
        return {"n_pairs": 0, "mean_correlation": None,
                "note": "need >=2 streams and >=3 shared labels"}
    correlations, inversions, compared = [], 0, 0
    for i, a in enumerate(streams):
        for b in streams[i + 1:]:
            shared = [l for l in labels if a in by_label[l] and b in by_label[l]]
            if len(shared) < 3:
                continue
            va = np.array([by_label[l][a] for l in shared])
            vb = np.array([by_label[l][b] for l in shared])
            if va.std() == 0 or vb.std() == 0:
                continue
            correlations.append(float(np.corrcoef(va, vb)[0, 1]))
            # An INVERSION: a beats b on one label and b beats a on another, both decisively.
            diff = va - vb
            if (diff > 0.15).any() and (diff < -0.15).any():
                inversions += 1
            compared += 1
    if not correlations:
        return {"n_pairs": 0, "mean_correlation": None, "note": "no comparable stream pairs"}
    return {
        "n_pairs": compared,
        "mean_correlation": round(float(np.mean(correlations)), 4),
        "inverting_pairs": inversions,
        "inverting_fraction": round(inversions / compared, 4),
    }


# ----------------------------------------------------------------------------------------------
# Consumption by the admissibility gate
# ----------------------------------------------------------------------------------------------

def load(path: Path = OUT_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — build it with "
            "`python -m training.evidence.resolvability --build --checkpoint <ckpt>`")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 2 or payload.get("scope") != \
            "phase_a_training_subjects_only_per_sensor" or "per_sensor" not in payload:
        raise ValueError(
            f"{path} is a legacy/leaky stream-level table. Rebuild it from a sensor-granularity "
            "checkpoint; old tables may include evaluation datasets and cannot fit the gate."
        )
    roster = set(payload.get("training_datasets", ()))
    measured = {value.get("dataset") for value in payload["per_sensor"].values()}
    if None in measured or not measured.issubset(roster):
        raise ValueError(
            f"{path} contains sensor rows outside its recorded Phase-A roster: "
            f"{sorted(value for value in measured - roster if value is not None)}"
        )
    return payload


# ``gate_tensor`` LIVED HERE AND WAS DELETED 2026-08-12.
#
# It answered "can this configuration witness this concept" by a dictionary read keyed on the literal
# "<dataset>/<stream>" string and an exact label match, with a neutral default on a miss. Against a
# novel vocabulary every entry defaulted, the gate became a uniform multiplier, and it provably could
# not change the argmax — no behaviour at all in exactly the open-vocabulary case it existed for.
# `training/evidence/admissibility_gate.py` replaces it with a function of text.
#
# THIS TABLE IS STILL LOAD-BEARING as the gate's warm start (`gate_predictor.fit_from_table`). It is
# not an independent validation set after fitting. Generalisation is measured with held-out folds in
# `gate_extrapolation`; reporting fit-table correlation as independent evidence would be leakage.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(
                        "training/tokenizer/outputs/phase_a_fixed_1s_rotation_20260817/best.pt"
                    ))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-streams", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if args.build:
        payload = build(args.checkpoint, args.device, args.limit_streams)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\n-> {args.out}  ({len(payload['per_sensor'])} sensors)")
        _print_report(payload)
    elif args.report:
        _print_report(load(args.out))
    else:
        ap.error("pass --build or --report")


def _print_report(payload: dict) -> None:
    contrast = payload.get("paired_contrast", {})
    print("\n=== PAIRED CONTRAST — simultaneous streams, same events, placement the only variable ===")
    if not contrast:
        print("  none: no paired dataset had 2+ scorable simultaneous streams")
        return
    for dataset, info in contrast.items():
        cd = info.get("concept_dependence", {})
        corr = cd.get("mean_correlation")
        print(f"\n{dataset}: {info['n_labels']} labels across simultaneous streams · "
              f"mean gap {info['mean_gap']:.3f} · max gap {info['max_gap']:.3f}")
        if corr is not None:
            print(f"    concept-dependence: stream-profile correlation {corr:+.3f} · "
                  f"{cd['inverting_pairs']}/{cd['n_pairs']} stream pairs INVERT "
                  f"({cd['inverting_fraction']:.0%})")
        worst = sorted(info["labels"].items(), key=lambda kv: -kv[1]["gap"])[:6]
        for label, row in worst:
            print(f"    {label:28s} best {row['best']:.2f} ({row['best_stream']})"
                  f"   worst {row['worst']:.2f} ({row['worst_stream']})   gap {row['gap']:.2f}")
    gaps = [i["mean_gap"] for i in contrast.values()]
    corrs = [i["concept_dependence"]["mean_correlation"] for i in contrast.values()
             if i.get("concept_dependence", {}).get("mean_correlation") is not None]
    print(f"\nOVERALL mean placement gap: {float(np.mean(gaps)):.3f}")
    if corrs:
        print(f"OVERALL stream-profile correlation: {float(np.mean(corrs)):+.3f}")
        print("  Near +1 would mean a fixed placement-quality ordering, and a scalar per placement")
        print("  would suffice. Near 0 means WHICH placement wins depends on the concept — which is")
        print("  what makes the gate's (config, config, concept) third argument necessary.")
    print("  A large gap is the premise of the admissibility gate, measured on identical events.")
    print("  A gap near zero would mean placement barely affects concept identifiability — which")
    print("  would leave the gate with nothing to gate on, and must be reported as such.")


if __name__ == "__main__":
    main()
