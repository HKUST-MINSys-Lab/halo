"""Render the Phase-A JSONL telemetry as a CPU-only live dashboard.

Pass ``--watch N`` to refresh periodically. Rendering is atomic, so another process can safely read
the PNG while it is being updated.

Run:
  python -m training.tokenizer.plot_training \
    --log training/tokenizer/outputs/pretrain_native/log.jsonl [--watch 30]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def load(log_path: Path):
    step_recs, val_recs = [], []
    for line in log_path.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        is_validation = any(key in r for key in (
            "development_transfer_knn_ba", "val_knn_label_stream_ba", "val_knn_ba",
        ))
        (val_recs if is_validation else step_recs).append(r)
    return step_recs, val_recs


def series(recs, key):
    xs, ys = [], []
    for r in recs:
        if r.get(key) is not None:
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def dict_series(recs, key):
    """{sub_key: ([steps], [values])} from records carrying a {sub_key: value} dict under `key`."""
    out: dict = {}
    for r in recs:
        for s, v in (r.get(key) or {}).items():
            out.setdefault(s, ([], []))
            out[s][0].append(r["step"])
            out[s][1].append(v)
    return out


def ewma(values, alpha=0.2):
    out = []
    for value in values:
        out.append(value if not out else alpha * value + (1.0 - alpha) * out[-1])
    return out


def selection_value(row):
    for key in ("development_transfer_knn_ba", "val_knn_label_stream_ba", "val_knn_ba"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return float("nan")


def _line(axis, records, key, label=None, *, smooth=False, **kwargs):
    x, y = series(records, key)
    if not x:
        return
    if smooth and len(y) > 2:
        # The faint raw underlay sets its OWN alpha/width, so drop any caller-supplied width or
        # colour rather than passing them twice -- every `smooth=True` caller passes lw, which
        # made --render a hard TypeError.
        underlay = {k: v for k, v in kwargs.items()
                    if k not in ("color", "lw", "linewidth", "alpha")}
        axis.plot(x, y, alpha=0.2, lw=0.6, **underlay)
        y = ewma(y)
    axis.plot(x, y, label=label or key, **kwargs)


def _legend(axis, **kwargs):
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, **kwargs)


def render(log_path: Path, out_path: Path):
    S, V = load(log_path)
    fig, ax = plt.subplots(4, 3, figsize=(18, 15))
    future_mode = any("future/loss" in row for row in S)

    loss_keys = (("total", "total"),
                 (("loss_weighted/future", "future") if future_mode
                  else ("loss_weighted/jepa", "masked JEPA")),
                 (("loss_weighted/physical", "physical") if future_mode
                  else ("loss_weighted/descriptor", "descriptor")),
                 (("loss_weighted/collapse", "collapse") if future_mode
                  else ("loss_weighted/vicreg", "VICReg")))
    for key, label in loss_keys:
        fallback = {"loss_weighted/jepa": "jepa", "loss_weighted/vicreg": "vicreg"}.get(key)
        _line(ax[0, 0], S, key if series(S, key)[0] else fallback, label, smooth=True, lw=1.1)
    ax[0, 0].set_title("weighted losses (EWMA)"); _legend(ax[0, 0], fontsize=7)

    component_keys = ((
        ("physical/loss", "physical"),
        ("physical/zero_baseline", "zero baseline"),
        ("collapse/variance", "variance penalty"),
        ("collapse/covariance", "covariance penalty"),
    ) if future_mode else (
        ("vicreg/invariance", "invariance"), ("vicreg/variance", "variance"),
        ("vicreg/covariance", "covariance"), ("vicreg/min_std", "min std"),
    ))
    for key, label in component_keys:
        _line(ax[0, 1], S, key, label, smooth=True, lw=1.0)
    ax[0, 1].set_title("physical and collapse terms" if future_mode else "VICReg components")
    _legend(ax[0, 1], fontsize=7)

    quality_keys = ((
        ("jepa/margin", "future target margin"),
        ("physical/improvement_over_zero", "physical gain over zero"),
        ("collapse/min_std", "visible min std"),
        ("future/eligible_fraction", "eligible fraction"),
    ) if future_mode else (
        ("jepa/margin", "JEPA margin"), ("vicreg/margin", "VICReg margin"),
        ("descriptor/top1", "descriptor top-1"),
        ("descriptor/chance_top1", "descriptor chance"),
    ))
    for key, label in quality_keys:
        _line(ax[0, 2], S, key, label, smooth=True, lw=1.1)
    ax[0, 2].axhline(0.0, color="black", lw=0.6)
    ax[0, 2].set_title("predictive health" if future_mode
                       else "pair margins and descriptor retrieval")
    _legend(ax[0, 2], fontsize=7)

    gradient_keys = ((
        ("grad/total_preclip", "total"), ("grad/encoder", "encoder"),
        ("grad/frontend", "frontend"),
        ("grad/future_predictor", "future predictor"),
        ("grad/physical_decoder", "physical decoder"),
        ("grad/descriptor_projection", "descriptor projection"),
    ) if future_mode else (
        ("grad/total_preclip", "total"), ("grad/encoder", "encoder"),
        ("grad/jepa_predictor", "JEPA head"),
        ("grad/vicreg_projector", "VICReg head"),
        ("grad/descriptor_head", "descriptor head"),
        ("grad/bias_projection", "bias projection"),
    ))
    for key, label in gradient_keys:
        _line(ax[1, 0], S, key, label, lw=0.9)
    ax[1, 0].set_yscale("log"); ax[1, 0].set_title("gradient norms (pre-clip)")
    _legend(ax[1, 0], fontsize=7)

    objective_names = (("future", "physical", "collapse") if future_mode
                       else ("jepa", "vicreg"))
    for name in objective_names:
        key, label = f"grad_objective/{name}", name
        _line(ax[1, 1], S, key, label, marker="o", ms=3, lw=1.0)
    ax[1, 1].set_yscale("log"); ax[1, 1].set_title("objective gradients on encoder")
    _legend(ax[1, 1], fontsize=7)
    geometry = ax[1, 1].twinx()
    share_name = "future" if future_mode else "jepa"
    _line(geometry, S, f"grad_objective/{share_name}_share", f"{share_name} share",
          color="tab:green", marker=".", lw=0.8)
    cosine_key = ("grad_cosine/future_vs_physical" if future_mode
                  else "grad_cosine/jepa_vs_vicreg")
    _line(geometry, S, cosine_key, "objective cosine", color="tab:red", marker=".", lw=0.8)
    geometry.set_ylim(-1.05, 1.05)
    _legend(geometry, fontsize=6, loc="lower right")

    for key, label in (("repr_encoder/effective_rank", "pooled encoder"),
                       (("repr_retrieval/effective_rank", "visible patch states")
                        if future_mode else ("repr_projector/effective_rank", "projector")),
                       ("repr_teacher/effective_rank", "EMA teacher")):
        _line(ax[1, 2], S, key, label, lw=1.0)
    ax[1, 2].set_title("representation effective rank"); _legend(ax[1, 2], fontsize=7)

    spread_prefix = "repr_retrieval" if future_mode else "repr_projector"
    for key, label in (("repr_encoder/min_std", "pooled min"),
                       ("repr_encoder/mean_std", "pooled mean"),
                       (f"{spread_prefix}/min_std", "patch min"),
                       (f"{spread_prefix}/mean_std", "patch mean")):
        _line(ax[2, 0], S, key, label, smooth=True, lw=0.9)
    ax[2, 0].set_title("representation spread"); _legend(ax[2, 0], fontsize=7)

    for key, label, style in (
        ("development_transfer_knn_ba", "development transfer (selection)", "g-o"),
        ("val_knn_label_stream_ba", "kNN label/stream (selection)", "k-o"),
        ("val_knn_ba", "kNN global", "k--"),
        ("val_conse_label_stream_ba", "ConSE label/stream", "b-s"),
        ("val_conse_ba", "ConSE global", "b--"),
    ):
        x, y = series(V, key)
        if x:
            ax[2, 1].plot(x, y, style, ms=3, lw=1.3, label=label)
    for _, (xs, ys) in sorted(dict_series(V, "val_ba_by_source").items()):
        ax[2, 1].plot(xs, ys, lw=0.6, alpha=0.25, color="gray")
    ax[2, 1].set_title("validation (gray = per-source kNN)"); _legend(ax[2, 1], fontsize=6)

    latest = S[-1] if S else {}
    observed = latest.get("data/source_share_window", {})
    target = latest.get("data/source_share_target", {})
    names = sorted(set(observed) | set(target))
    if names:
        y = list(range(len(names)))
        ax[2, 2].barh([v - 0.18 for v in y], [observed.get(k, 0) for k in names], 0.36,
                      label="observed")
        ax[2, 2].barh([v + 0.18 for v in y], [target.get(k, 0) for k in names], 0.36,
                      label="target")
        ax[2, 2].set_yticks(y, names, fontsize=6)
    ax[2, 2].set_title("rolling dataset share"); _legend(ax[2, 2], fontsize=7)

    frontend_keys = (
        ("frontend/observable_fraction", "observable kernels"),
        ("frontend/dead_kernel_fraction", "dead kernels"),
        ("frontend/response_std_mean", "response std"),
        ("duration/gate", "duration gate"),
    )
    has_frontend = any(
        series(S, key)[0] for key, _ in frontend_keys if key.startswith("frontend/")
    )
    aug = latest.get("data/augmentation_rate_window", {})
    if has_frontend:
        for key, label in frontend_keys:
            _line(ax[3, 0], S, key, label, smooth=True, lw=1.0)
        ax[3, 0].set_title("frontend health")
        _legend(ax[3, 0], fontsize=7)
    elif aug:
        names = sorted(aug)
        ax[3, 0].barh(range(len(names)), [aug[k] for k in names])
        display = {
            "channel_dropout": "channel drop",
            "channel_text_dropout": "channel-text drop",
            "channel_text_phrase": "text paraphrase",
            "rotation_3d": "SO(3) rotation",
            "sensor_text_dropout": "sensor-text drop",
            "window_crop": "window crop",
        }
        ax[3, 0].set_yticks(range(len(names)), [display.get(name, name) for name in names], fontsize=6)
        ax[3, 0].set_xlim(0, 1)
        ax[3, 0].set_title("realized augmentation rate")

    _line(ax[3, 1], S, "perf/steps_per_s", "steps/s", lw=1.0, color="tab:green")
    ax[3, 1].set_title("throughput"); ax[3, 1].set_ylabel("steps/s")
    mem = ax[3, 1].twinx()
    _line(mem, S, "memory/allocated_gib", "allocated GiB", lw=0.9, color="tab:purple")
    _line(mem, S, "memory/reserved_gib", "reserved GiB", lw=0.9, color="tab:orange")
    mem.set_ylabel("GiB")
    _legend(ax[3, 1], fontsize=7, loc="upper left")
    _legend(mem, fontsize=7, loc="upper right")

    ax[3, 2].axis("off")
    selection_scores = [selection_value(row) for row in V]
    selection_scores = [value for value in selection_scores if math.isfinite(value)]
    best = max(selection_scores, default=float("nan"))
    best_c = max((r.get("val_conse_label_stream_ba", r.get("val_conse_ba", float("nan")))
                  for r in V), default=float("nan"))
    clip_values = series(S, "grad/clip_coefficient")[1]
    clipped_fraction = (sum(v < 1.0 for v in clip_values) / len(clip_values)
                        if clip_values else float("nan"))
    objective_summary = (
        f"future margin : {latest.get('jepa/margin', float('nan')):.3f}\n"
        f"physical gain : {latest.get('physical/improvement_over_zero', float('nan')):.3f}\n"
        f"future leak   : {latest.get('future/leakage_count', 0)}\n"
        f"ineligible    : {latest.get('jepa_ineligible_frac_window', float('nan')):.2%}\n"
        if future_mode else
        f"zero targets  : {latest.get('jepa_zero_target_frac_window', float('nan')):.2%}\n"
        f"descriptor tgt: {latest.get('descriptor/target_window_fraction', float('nan')):.2%}\n"
        f"descriptor top: {latest.get('descriptor/top1', float('nan')):.2%}\n"
        f"descriptor rnd: {latest.get('descriptor/chance_top1', float('nan')):.2%}\n"
    )
    txt = (f"step          : {S[-1]['step'] if S else 0}\n"
           f"val points   : {len(V)}\n"
           f"best selection: {best:.3f}\n"
           f"best ConSE LS : {best_c:.3f}\n"
           f"last total    : {series(S, 'total')[1][-1] if series(S, 'total')[1] else float('nan'):.3f}\n"
           f"steps/s       : {latest.get('perf/steps_per_s', float('nan')):.2f}\n"
           f"ETA minutes   : {latest.get('perf/eta_minutes', float('nan')):.1f}\n"
           f"{objective_summary}"
           f"input finite  : {latest.get('data/input_finite_fraction', float('nan')):.6f}\n"
           f"input abs max : {latest.get('data/input_abs_max', float('nan')):.2f}\n"
           f"AMP skips     : {latest.get('amp/skipped_updates_total', 0)}\n"
           f"logged clipped: {clipped_fraction:.1%}")
    ax[3, 2].text(0.03, 0.95, txt, va="top", family="monospace", fontsize=10)

    for axis in ax.flat:
        if axis.axison:
            axis.set_xlabel("step" if not axis.patches else "")
            axis.grid(alpha=0.18)

    fig.suptitle(f"training telemetry — {log_path.parent.name}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    tmp.replace(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--watch", type=float, default=0.0, help="re-render every N seconds (0 = once)")
    args = ap.parse_args()
    out = args.out or args.log.parent / "telemetry.png"
    while True:
        if args.log.exists():
            render(args.log, out)
            print(f"-> {out}", flush=True)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
