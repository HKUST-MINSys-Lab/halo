"""Training-telemetry dashboard for a support-classifier run (reads ``<run>/log.jsonl``).

Panels are chosen for diagnosing the learned classifier, not for decoration:
losses and auxiliaries; internal validation (seen-label vs label-held-out); branch accuracy on the
training batch and on the label-held-out panel; semantic reliance (the router) with quantiles;
support-correction magnitude and temperatures; gradient norms; sampler realised view shares.
Keys absent in a run (e.g. a residual or neighbours arm) are skipped, so the same script serves
every architecture.  ``--compare`` overlays a second run's validation curves.

Usage: python results/tools/plot_telemetry.py RUN_DIR --out FIG.png [--compare LABEL=RUN_DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(run: Path):
    rows = [json.loads(line) for line in open(run / "log.jsonl")]
    train = [r for r in rows if r.get("kind") == "train"]
    val = [r for r in rows if r.get("kind") == "validation"]
    return train, val


def series(rows, key):
    pts = [(r["step"], r[key]) for r in rows if isinstance(r.get(key), (int, float))
           and np.isfinite(r[key])]
    if not pts:
        return None, None
    x, y = zip(*pts)
    return np.asarray(x, float), np.asarray(y, float)


def smooth(y, n):
    if y is None or len(y) < 3 or n <= 1:
        return y
    n = min(n, len(y))
    kernel = np.ones(n) / n
    pad = np.concatenate([np.full(n // 2, y[0]), y, np.full(n - n // 2 - 1, y[-1])])
    return np.convolve(pad, kernel, mode="valid")


PANELS = [
    ("Training loss (smoothed)", "train", [
        ("loss", "total"), ("loss/few_shot_ce", "enrolled CE"), ("loss/zero_shot_ce", "zero-support CE"),
        ("aux/branch_preservation", "aux: branch preservation"),
        ("aux/best_path_non_regression", "aux: best-path regret")]),
    ("Internal validation (selection signals)", "val", [
        ("validation/selection_seen_scenario_balanced_f1", "seen labels, scenario-balanced"),
        ("validation/selection_open_vocabulary_f1", "label-held-out"),
        ("validation/enrolled_dataset_macro_f1", "enrolled dataset-macro"),
        ("validation/zero_shot_dataset_macro_f1", "zero-support dataset-macro"),
        ("validation/selection_dataset_macro_f1", "selection score")]),
    ("Branch accuracy, training batch (smoothed)", "train", [
        ("classifier/branch_accuracy/semantic_status", "label meaning"),
        ("classifier/branch_accuracy/support_status", "support vote"),
        ("classifier/branch_accuracy/refined_support", "contextual support"),
        ("accuracy/learned", "final")]),
    ("Branch accuracy, label-held-out panel", "val", [
        ("validation/open_vocabulary/classifier/branch_accuracy/semantic_status", "label meaning"),
        ("validation/open_vocabulary/classifier/branch_accuracy/support_status", "support vote"),
        ("validation/open_vocabulary/classifier/branch_accuracy/refined_support", "contextual support"),
        ("validation/open_vocabulary/accuracy/learned", "final")]),
    ("Semantic reliance (router), training (smoothed)", "train", [
        ("classifier/semantic_reliance_p10", "p10"), ("classifier/semantic_reliance_p50", "p50"),
        ("classifier/semantic_reliance_p90", "p90"),
        ("classifier/semantic_reliance_direct_support", "mean, truth has support"),
        ("classifier/semantic_reliance_no_direct_support", "mean, truth has no support")]),
    ("Semantic reliance, validation", "val", [
        ("validation/classifier/semantic_reliance", "seen labels"),
        ("validation/open_vocabulary/classifier/semantic_reliance", "label-held-out"),
        ("validation/classifier/semantic_reliance_saturation_high", "seen: share > high"),
        ("validation/open_vocabulary/classifier/semantic_reliance_saturation_high", "held-out: share > high")]),
    ("Best-branch rescue / harm and regret (smoothed)", "train", [
        ("classifier/best_branch_rescue_rate", "rescue rate"), ("classifier/best_branch_harm_rate", "harm rate"),
        ("classifier/positive_regret_fraction", "positive-regret fraction")]),
    ("Support correction & temperatures", "train", [
        ("classifier/support_correction_rms", "correction RMS"), ("classifier/support_correction_p95", "correction p95"),
        ("classifier/correction_scale", "correction scale"), ("classifier/tau_support", "tau support"),
        ("classifier/tau_semantic", "tau semantic"), ("classifier/tau_label", "tau label")]),
    ("Gradient norms, pre-clip (smoothed, log)", "train", [
        ("gradient/total_preclip_norm", "total"), ("gradient/classifier_norm", "classifier"),
        ("gradient/encoder_norm", "encoder")]),
    ("Sampler: realised view shares (smoothed)", "train", [
        ("sampler/view_enrollment_zero_fraction", "zero-support views"),
        ("sampler/view_enrollment_partial_fraction", "partial views"),
        ("sampler/view_enrollment_complete_fraction", "complete views"),
        ("sampler/counterfactual_episode_fraction", "counterfactual episodes")]),
    ("Encoder effective rank", "both", [
        ("encoder/effective_rank", "train batch"), ("validation/encoder_effective_rank", "validation")]),
    ("Validation loss", "val", [
        ("validation/loss", "seen labels"), ("validation/open_vocabulary/loss", "label-held-out")]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--smooth", type=int, default=20)
    parser.add_argument("--compare", action="append", default=[], help="LABEL=RUN_DIR; overlays validation")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, val = read(args.run)
    others = []
    for spec in args.compare:
        label, _, path = spec.partition("=")
        others.append((label, read(Path(path))))
    panels = [p for p in PANELS if any(
        series(train if src != "val" else val, key)[0] is not None
        or (src == "both" and series(val, key)[0] is not None) for key, _ in [(k, n) for k, n in p[2]]
        for src in [p[1]])]
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, src, keys) in zip(axes, panels):
        for key, name in keys:
            rows_src = val if key.startswith("validation/") else train
            x, y = series(rows_src, key)
            if x is None:
                continue
            if rows_src is train:
                y = smooth(y, args.smooth)
                ax.plot(x, y, lw=1.4, label=name)
            else:
                line, = ax.plot(x, y, lw=1.6, marker="o", ms=3, label=name)
                for label, (_, oval) in others:
                    ox, oy = series(oval, key)
                    if ox is not None:
                        ax.plot(ox, oy, lw=1.0, ls="--", color=line.get_color(), alpha=0.6,
                                label=f"{name} [{label}]")
        if "log" in title:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(args.title or args.run.name)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out} ({len(panels)} panels)")


if __name__ == "__main__":
    main()
