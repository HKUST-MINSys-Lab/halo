"""Comprehensive quality battery for a frozen Phase-1 encoder.

Characterizes the representation along the axes that matter for feeding an evidence
engine.  It uses supervised-training sources only; sealed evaluation belongs exclusively to the
support-protocol runner.

Probes (all on supervised-training sources, subject-disjoint unless noted):
  1. DISCRIMINABILITY   kNN-BA + linear-probe macro-F1 per dataset; per-class + confusion
  2. INVARIANCE         embedding drift (cosine / rel-L2) under rotation / gain / rate, and
                        kNN-BA when the QUERY is transformed vs a clean bank — the thesis test
  3. CROSS-PLACEMENT    wisdm phone->watch retrieval (same activities, different device/site;
                        TRAIN data — a placement-transfer probe, flagged)
  4. MISSING-CHANNEL    full-IMU vs accel-only kNN-BA (graceful degradation)
  5. LEARNED vs HANDCRAFTED   encoder kNN-BA vs the M0 gravity-aligned band-energy features
  6. SPACE HEALTH       effective rank of the embedding (collapse check)

Run:  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.eval_quality \
        --checkpoint training/tokenizer/outputs/pretrain_native/best.pt
      # NB: pretrain_native/best.pt is the real trained model (val_ba 0.659). The default
      # outputs/pretrain/ dir holds only smoke/debug runs — do NOT evaluate that one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.scripts.eda.grid_io import discover_grids
from training.tokenizer.eval_transfer import (EVAL_STREAMS, build_encoder, encode_dataset,
                                              knn_balanced_acc)
from training.tokenizer.pretrain_data import stream_channel_descriptions, _stream_gravity_state

RATE = 60.0
SEED = 20260718
# Report is written next to the evaluated checkpoint (args.checkpoint.parent), NOT a hardcoded dir —
# so evaluating pretrain_native/best.pt never drops its report into the smoke outputs/pretrain/ folder.


# --------------------------------------------------------------------------- transforms
def _rand_so3(rng):
    q = rng.normal(size=4); w, x, y, z = q / np.linalg.norm(q)
    return torch.tensor([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]], dtype=torch.float32)


def transform_windows(data, kind, rng, base_rate):
    """Return a nuisance-transformed copy of (N,T,6) windows at the stream's NATIVE `base_rate`."""
    x = torch.tensor(np.asarray(data), dtype=torch.float32)
    if kind == "rotation":
        r = _rand_so3(rng)
        x[:, :, :3] = x[:, :, :3] @ r.T
        x[:, :, 3:] = x[:, :, 3:] @ r.T
    elif kind == "gain":
        x = x * float(rng.uniform(0.7, 1.3))
    elif kind == "rate":
        from scipy.signal import resample_poly
        y = resample_poly(x.numpy(), 1, 2, axis=1)      # halve the native rate (anti-aliased)
        return torch.tensor(y, dtype=torch.float32), base_rate / 2.0
    return x, base_rate


# --------------------------------------------------------------------------- encoding
@torch.no_grad()
def encode(enc, data, texts, device, rate=RATE, accel_only=False, gravity_state=None,
           channel_mask=None, dataset=None, stream=None):
    x = data if torch.is_tensor(data) else torch.tensor(np.asarray(data), dtype=torch.float32)
    cmask = (torch.ones(6, dtype=torch.bool) if channel_mask is None
             else torch.as_tensor(channel_mask, dtype=torch.bool).clone())
    if accel_only:
        cmask[3:] = False
    x = x.clone().float()
    if accel_only:
        x[:, :, 3:] = 0.0
    return encode_dataset(enc, x, texts, device, rate, gravity_state,
                          channel_mask=cmask, dataset=dataset, stream=stream)


def subj_split(subjects, rng):
    subj = sorted(set(subjects.tolist())); rng.shuffle(subj)
    hold = set(subj[: max(1, len(subj)//2)])
    tr = [i for i in range(len(subjects)) if subjects[i] not in hold]
    te = [i for i in range(len(subjects)) if subjects[i] in hold]
    return tr, te


def effective_rank(z):
    s = torch.linalg.svdvals(z - z.mean(0)); p = (s/s.sum()).clamp(min=1e-12)
    return float(torch.exp(-(p*p.log()).sum()))


def linear_probe(train_z, train_y, test_z, test_y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    mu, sd = train_z.mean(0), train_z.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(((train_z - mu)/sd).numpy(), train_y)
    pred = clf.predict(((test_z - mu)/sd).numpy())
    return float(f1_score(test_y, pred, average="macro"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
    rng = np.random.default_rng(SEED)
    report = {"checkpoint": str(args.checkpoint), "step": ckpt["step"]}

    # ---------- 1. discriminability + 4. missing-channel + 5. learned-vs-handcrafted ----
    from training.tokenizer.probe_robustness import FAMILIES
    disc_stream = {}
    for dataset, stream in EVAL_STREAMS:
        ref = refs.get((dataset, stream))
        if ref is None:
            continue
        full = np.asarray(ref.load_lengths()) == ref.shape[1]
        data = ref.load_data()[full]
        labels = np.asarray(ref.labels)[full]
        subj = np.asarray(ref.subjects)[full]
        texts = stream_channel_descriptions(dataset, stream)
        gstate = _stream_gravity_state(dataset, stream)
        z = encode(enc, data, texts, device, rate=ref.rate_hz, gravity_state=gstate,
                   channel_mask=ref.mask, dataset=dataset, stream=stream)
        tr, te = subj_split(subj, rng)
        y_tr, y_te = labels[tr].tolist(), labels[te].tolist()
        knn = knn_balanced_acc(z[tr], y_tr, z[te], y_te)
        lin = linear_probe(z[tr], y_tr, z[te], y_te)
        z_ao = encode(enc, data, texts, device, rate=ref.rate_hz, accel_only=True,
                      gravity_state=gstate, channel_mask=ref.mask,
                      dataset=dataset, stream=stream)
        knn_ao = knn_balanced_acc(z_ao[tr], y_tr, z_ao[te], y_te)
        # handcrafted grav_band_energy on the same split (at the stream's native rate)
        hand = torch.tensor(np.stack([
            FAMILIES["grav_band_energy"](np.asarray(w[:, :3], np.float64),
                                         np.asarray(w[:, 3:], np.float64), ref.rate_hz)
            for w in data]), dtype=torch.float32)
        hand = torch.where(torch.isfinite(hand), hand, hand.nan_to_num())
        hand = (hand - hand.mean(0)) / (hand.std(0) + 1e-6)
        knn_hand = knn_balanced_acc(hand[tr], y_tr, hand[te], y_te)
        cell = f"{dataset}/{stream}"
        disc_stream[cell] = {"dataset": dataset, "stream": stream,
                             "knn_ba": round(knn, 3), "linear_f1": round(lin, 3),
                             "knn_accel_only": round(knn_ao, 3),
                             "knn_handcrafted": round(knn_hand, 3),
                             "eff_rank": round(effective_rank(z), 1),
                             "labels": len(set(labels.tolist()))}
        print(f"  {cell:42s} kNN {knn:.3f} | linF1 {lin:.3f} | accel-only {knn_ao:.3f} "
              f"| handcrafted {knn_hand:.3f} | rank {effective_rank(z):.0f}", flush=True)
    metric_names = ("knn_ba", "linear_f1", "knn_accel_only", "knn_handcrafted", "eff_rank")
    disc = {}
    for dataset in sorted({row["dataset"] for row in disc_stream.values()}):
        rows = [row for row in disc_stream.values() if row["dataset"] == dataset]
        disc[dataset] = {name: round(float(np.mean([row[name] for row in rows])), 3)
                         for name in metric_names}
        disc[dataset]["streams"] = len(rows)
        disc[dataset]["per_stream"] = {row["stream"]: row["knn_ba"] for row in rows}
    report["discriminability"] = disc
    report["discriminability_per_stream"] = disc_stream

    # ---------- 2. invariance (drift + kNN-under-transform) ----------
    # Keep this diagnostic entirely inside the supervised roster.  A quality probe must never
    # casually consume a sealed test stream just because it has a convenient phone placement.
    inv = {}
    from data.scripts.curate.deployment_policy import SUPERVISED_HEAD_TRAIN_DATASETS, stream_specs
    candidates = [
        (dataset, spec.stream_id)
        for dataset in SUPERVISED_HEAD_TRAIN_DATASETS
        for spec in stream_specs(dataset, "primary")
        if (dataset, spec.stream_id) in refs
    ]
    if not candidates:
        raise RuntimeError("no supervised stream is available for the invariance diagnostic")
    dataset, stream = candidates[0]
    ref = refs[(dataset, stream)]
    full = np.asarray(ref.load_lengths()) == ref.shape[1]
    data = ref.load_data()[full]
    labels = np.asarray(ref.labels)[full]
    subj = np.asarray(ref.subjects)[full]
    texts = stream_channel_descriptions(dataset, stream)
    z0 = encode(enc, data, texts, device, rate=ref.rate_hz,
                channel_mask=ref.mask, dataset=dataset, stream=stream)
    tr, te = subj_split(subj, rng)
    base_knn = knn_balanced_acc(z0[tr], labels[tr].tolist(), z0[te], labels[te].tolist())
    for kind in ("rotation", "gain", "rate"):
        xt, rt = transform_windows(data, kind, np.random.default_rng(SEED), ref.rate_hz)
        zt = encode(enc, xt, texts, device, rate=rt,
                    channel_mask=ref.mask, dataset=dataset, stream=stream)
        cos = torch.nn.functional.cosine_similarity(z0, zt, dim=1).mean().item()
        rel = ((z0 - zt).norm(dim=1) / (z0.norm(dim=1) + 1e-9)).mean().item()
        # classify TRANSFORMED query against CLEAN bank
        knn_t = knn_balanced_acc(z0[tr], labels[tr].tolist(), zt[te], labels[te].tolist())
        inv[kind] = {"emb_cosine": round(cos, 3), "emb_rel_l2": round(rel, 3),
                     "knn_transformed_query": round(knn_t, 3)}
        print(f"  invariance/{kind:9s} emb-cos {cos:.3f} rel-L2 {rel:.3f} "
              f"| kNN(clean {base_knn:.3f} -> transformed {knn_t:.3f})", flush=True)
    inv["clean_knn"] = round(base_knn, 3)
    report["invariance"] = inv

    # ---------- 3. cross-placement (wisdm phone -> watch; TRAIN data, flagged) ----------
    try:
        rp = refs[("wisdm", "phone_pocket")]; rw = refs[("wisdm", "watch_wrist")]
        kp = np.asarray(rp.load_lengths()) == rp.shape[1]
        kw = np.asarray(rw.load_lengths()) == rw.shape[1]
        zp = encode(enc, rp.load_data()[kp], stream_channel_descriptions("wisdm", "phone_pocket"),
                    device, rate=rp.rate_hz, channel_mask=rp.mask,
                    dataset="wisdm", stream="phone_pocket")
        zw = encode(enc, rw.load_data()[kw], stream_channel_descriptions("wisdm", "watch_wrist"),
                    device, rate=rw.rate_hz, channel_mask=rw.mask,
                    dataset="wisdm", stream="watch_wrist")
        yp = np.asarray(rp.labels)[kp].astype(str).tolist()
        yw = np.asarray(rw.labels)[kw].astype(str).tolist()
        p2w = knn_balanced_acc(zp, yp, zw, yw)
        w2p = knn_balanced_acc(zw, yw, zp, yp)
        report["cross_placement_wisdm(train-data)"] = {"phone->watch": round(p2w, 3),
                                                       "watch->phone": round(w2p, 3)}
        print(f"  cross-placement wisdm phone->watch {p2w:.3f} | watch->phone {w2p:.3f} "
              f"(TRAIN data)", flush=True)
    except KeyError:
        pass

    means = {k: round(float(np.mean([d[k] for d in disc.values()])), 3)
             for k in ("knn_ba", "linear_f1", "knn_accel_only", "knn_handcrafted")}
    report["means"] = means
    print(f"\nMEANS across held-out: {means}")
    out = args.checkpoint.parent / "quality_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
