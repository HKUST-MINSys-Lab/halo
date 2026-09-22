"""Sensor-side screen: do the learned primitives fire on the recordings they name?

The T7 post-mortem's cause 5 was that the wrong side of the primitive bridge was validated. The
label side was audited against hand-written expectations before training; the sensor side never
was. So nobody checked the question the whole design rests on -- does ``impact/hard`` actually
fire on jumping recordings? -- and a branch that was confidently wrong on foreign vocabulary
shipped into a 40k arm.

This is that check, and the design entry registers it as the first thing to run on a trained
primitive arm, before its sealed evaluation is read.

METHOD. Encode held-out-subject windows from the training corpus, take the sensor-side per-axis
primitive distribution the head computes from the pooled recording vector alone, and score it
against the written annotation for that window's label. For every primitive v on axis a, the
target is "this recording's label was annotated with v on axis a" and the score is the head's own
probability mass on v. Area under the ROC curve summarises it. A head that has learned to
decompose movement scores well above 0.5; a head whose primitives are a re-encoding of label
identity scores near chance on any primitive that does not happen to separate the labels present.

Two controls run alongside, because an AUC on its own proves little:

  * ``shuffled`` -- window labels permuted. Destroys the recording/annotation correspondence
    while preserving both marginals. Must land at 0.5.
  * ``prior`` -- the head's output replaced by the corpus-mean profile, i.e. the best possible
    label-blind guess. Beating this is what "the sensor side is reading the recording" means;
    a primitive can score a high AUC purely by being common in one frequent label.

Nothing here touches sealed or scenario data, and nothing here selects a checkpoint.

    .venv/bin/python results/tools/primitive_sensor_side_screen.py \
        --checkpoint runs/support-classifier/halo_t8_40k_20260921/last.pt --windows 1500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based ROC AUC. Ties share their average rank, as they must."""
    positive = positive.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks within tied score groups so a constant score gives exactly 0.5.
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(scores) + 1):
        if index == len(scores) or sorted_scores[index] != sorted_scores[start]:
            ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--windows", type=int, default=1500,
                        help="held-out windows to encode (balanced across labels)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="optional JSON report path")
    args = parser.parse_args()
    os.chdir(REPO)

    from data.scripts.curate.deployment_policy import SUPERVISED_HEAD_TRAIN_DATASETS
    from model.support.factory import build_classifier_from_blob
    from model.support.primitive_annotations import ANNOTATIONS, annotation_hash
    from model.support.primitive_semantics import axis_names, axis_slices, primitive_names
    from training.support_classifier.collate import SupportCollate
    from training.support_classifier.corpus import support_corpus_from_index
    from training.support_classifier.encoding import encode_batch
    from training.support_classifier.train import _load_items
    from training.tokenizer.eval_transfer import build_encoder
    from training.tokenizer.pretrain_data import CorpusIndex, MultiResolutionCollate, PretrainDataset

    device = torch.device(args.device)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = build_encoder(blob, device, training=False)
    classifier, version = build_classifier_from_blob(blob, device=device)
    head = getattr(classifier, "primitive_head", None)
    if head is None:
        raise SystemExit(f"{args.checkpoint} has no primitive head; nothing to screen")

    config = blob["config"]
    index = CorpusIndex(datasets=SUPERVISED_HEAD_TRAIN_DATASETS, alignment="native",
                        max_per_stream=4000, seed=20260901, window_seconds=8.0)
    dataset = PretrainDataset(index, index.val, augment=False, two_view=False,
                              conditioning_schema=config["conditioning_schema"],
                              multi_device_probability=0.0, max_devices=4)
    corpus = support_corpus_from_index(index, split="val")
    collate = SupportCollate(MultiResolutionCollate(
        fixed_patch_seconds=tuple(config["eval_resolutions"])))

    annotations = ANNOTATIONS[head.cfg.annotation_version]
    axes, slices = axis_names(head.cfg.version), axis_slices(head.cfg.version)
    # primitive_names() returns "axis/value"; the annotations store the bare value.
    values_of = tuple(name.split("/", 1)[1] for name in primitive_names(head.cfg.version))

    # Balanced sample over the annotated labels present in the held-out split: an unbalanced draw
    # would let one frequent label decide every AUC.
    by_label = defaultdict(list)
    for record in corpus.recordings:
        if record.label in annotations:
            by_label[record.label].append(record.window_index)
    if not by_label:
        raise SystemExit("no annotated training labels present in the held-out split")
    rng = np.random.default_rng(args.seed)
    per_label = max(1, args.windows // len(by_label))
    positions, labels = [], []
    for label, windows in sorted(by_label.items()):
        chosen = rng.choice(windows, size=min(per_label, len(windows)), replace=False)
        positions.extend(int(w) for w in chosen)
        labels.extend([label] * len(chosen))
    print(f"[screen] {len(positions)} windows over {len(by_label)} annotated labels "
          f"({per_label} per label, held-out subjects)", flush=True)

    profiles = []
    with torch.no_grad():
        for start in range(0, len(positions), args.batch):
            chunk = positions[start:start + args.batch]
            batch = collate(_load_items(dataset, chunk, None))
            pooled = encode_batch(encoder, batch, device)["pooled"]
            profiles.append(head.profile(pooled).float().cpu().numpy())
            if start % (args.batch * 10) == 0:
                print(f"[screen] encoded {start + len(chunk)}/{len(positions)}", flush=True)
    profile = np.concatenate(profiles, axis=0)                      # (N, n_primitives)
    labels = np.asarray(labels, dtype=object)

    # Ground truth: the written annotation's value for each axis of each window's label. An
    # annotation is a tuple in axis order, so the axis INDEX selects the value.
    truth = {a: np.asarray([annotations[label][a] for label in labels], dtype=object)
             for a in range(len(axes))}
    shuffled = labels[rng.permutation(len(labels))]
    truth_shuffled = {a: np.asarray([annotations[label][a] for label in shuffled], dtype=object)
                      for a in range(len(axes))}
    prior = profile.mean(axis=0, keepdims=True).repeat(len(profile), axis=0)

    report, per_axis = {}, {}
    for axis_index, (axis, block) in enumerate(zip(axes, slices)):
        scores = {}
        for column in range(block.start, block.stop):
            value = values_of[column]
            positive = truth[axis_index] == value
            if positive.sum() == 0 or positive.sum() == len(positive):
                continue        # this primitive is never (or always) used; AUC undefined
            scores[value] = {
                "auc": round(auc(profile[:, column], positive), 4),
                "shuffled": round(auc(profile[:, column],
                                      truth_shuffled[axis_index] == value), 4),
                "prior": round(auc(prior[:, column], positive), 4),
                "support": int(positive.sum()),
            }
        if scores:
            report[axis] = scores
            per_axis[axis] = {
                key: round(float(np.mean([v[key] for v in scores.values()])), 4)
                for key in ("auc", "shuffled", "prior")
            }

    flat = [v for axis in report.values() for v in axis.values()]
    overall = {key: round(float(np.mean([v[key] for v in flat])), 4)
               for key in ("auc", "shuffled", "prior")}

    print("\n=== sensor-side primitive screen ===")
    print(f"checkpoint {args.checkpoint}  ({version})")
    print(f"vocabulary {head.cfg.version}  annotations {head.cfg.annotation_version} "
          f"({annotation_hash(head.cfg.annotation_version)[:16]}...)")
    print(f"{len(positions)} held-out windows, {len(by_label)} labels, "
          f"{len(flat)} primitives scored\n")
    print(f"{'axis':<14}{'AUC':>8}{'shuffled':>10}{'prior':>8}   {'lift over prior':>16}")
    for axis, values in per_axis.items():
        lift = values["auc"] - values["prior"]
        print(f"{axis:<14}{values['auc']:>8.3f}{values['shuffled']:>10.3f}"
              f"{values['prior']:>8.3f}{lift:>+17.3f}")
    print(f"{'OVERALL':<14}{overall['auc']:>8.3f}{overall['shuffled']:>10.3f}"
          f"{overall['prior']:>8.3f}{overall['auc'] - overall['prior']:>+17.3f}")

    ranked = sorted(((v["auc"], f"{axis}/{name}", v)
                     for axis, block in report.items() for name, v in block.items()),
                    reverse=True)
    print("\nbest five primitives")
    for value, name, entry in ranked[:5]:
        print(f"  {name:<28}AUC {value:.3f}   prior {entry['prior']:.3f}   n={entry['support']}")
    print("worst five primitives")
    for value, name, entry in ranked[-5:]:
        print(f"  {name:<28}AUC {value:.3f}   prior {entry['prior']:.3f}   n={entry['support']}")

    payload = {
        "checkpoint": args.checkpoint, "classifier": version,
        "vocabulary": head.cfg.version, "annotations": head.cfg.annotation_version,
        "annotation_hash": annotation_hash(head.cfg.annotation_version),
        "windows": len(positions), "labels": len(by_label),
        "split": "val (subject held out) of the supervised training corpus",
        "overall": overall, "per_axis": per_axis, "per_primitive": report,
    }
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
