"""Per-encoder temperature for rung 1's probability features (temperature scaling, Guo et al. 2017).

Why: EM-Dirichlet consumes each window as a probability vector, softmax(T * score). One fixed T
made the six providers' inputs incomparable (2026-09-25 smoke tests): UniMTS's native scores are
tightly bunched, so at T = 30 its features were essentially uniform (normalised entropy 1.00) while
HALO's were near one-hot, and the transductive method's behaviour was set by score scale rather than
by the encoder. Each encoder therefore gets the T that makes its zero-shot probabilities calibrated
— maximum likelihood of the true label — on labelled data it was allowed to see: subject-held-out
windows of the eight supervised training sources. No sealed data is read; the procedure is identical
for every encoder.

Calibration set. The subject split is the training corpus's own (``validation_subjects_for_refs``
with the v4 data seed), so HALO's ``p_text`` never trained on these subjects. Rows come from the same
training-bank streams the ConSE bridge uses (rebuilt exactly, so cached features are reused), each
scored against its own dataset's label roster:

* ``halo``: its ``p_text`` (or, for a checkpoint without one, a ridge bridge refitted on the
  *non-held-out* bank rows only);
* ``harnet5`` / ``harnet10`` / ``limubert_x``: training-bank 1-NN + ConSE with a bank of the
  non-held-out subjects only (otherwise every window would find itself);
* ``unimts`` / ``normwear``: their native text heads.

The loss is dataset-balanced mean negative log-likelihood; ``distance`` scores (NormWear) are divided
by their per-dataset standard deviation first, mirroring ``probability_features``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from baselines import scoring
from baselines.data import list_streams, load_eval_stream, load_global_labels
from data.scripts.curate.deployment_policy import SUPERVISED_HEAD_TRAIN_DATASETS
from evaluation.zero_shot import (
    SCORE_KIND, TRAINING_BANK_ZERO_SHOT, _training_bank_conse_predictions, fit_halo_text_bridge,
    halo_text_scores, zero_shot_feature_role,
)
from training.support_classifier.sampling import MIN_RECORDING_SECONDS

CALIBRATION_VERSION = "temperature-calibration-v1"
DATA_SEED = 20260901                 # the v4 corpus split seed (training ``--data-seed``)
LOG_T_BOUNDS = (np.log(0.1), np.log(10_000.0))


def held_out_subjects(window_seconds: float, *, data_seed: int = DATA_SEED) -> set[tuple[str, str]]:
    """The training corpus's own subject-held-out validation subjects."""
    from training.tokenizer.pretrain_data import CorpusIndex, validation_subjects_for_refs

    index = CorpusIndex(max_per_stream=4, seed=data_seed, datasets=SUPERVISED_HEAD_TRAIN_DATASETS,
                        alignment="native", window_seconds=float(window_seconds))
    return {(str(d), str(s)) for d, s in validation_subjects_for_refs(index.refs, seed=data_seed)}


def _bank_streams(name: str, window_seconds: float):
    """Yield the exact training-bank streams ``_build_training_reference_bank`` encodes (same rows,
    same perturbation tag), so their cached features are reused, with per-row subjects kept."""
    train_labels = load_global_labels()
    for dataset in SUPERVISED_HEAD_TRAIN_DATASETS:
        for stream_id in list_streams(dataset, alignment="native", window_seconds=window_seconds):
            stream = load_eval_stream(dataset, stream_id, alignment="native", apply_quality_screen=True,
                                      candidate_labels=train_labels, window_seconds=window_seconds)
            aligned = np.asarray(scoring.align_ground_truth_labels(stream.gt, train_labels), dtype=object)
            lengths = (np.asarray(stream.lengths, dtype=np.int64) if stream.lengths is not None
                       else np.full(stream.n_windows, stream.windows.shape[1], dtype=np.int64))
            keep = np.flatnonzero((aligned != None) & (lengths / float(stream.rate_hz) >= MIN_RECORDING_SECONDS))  # noqa: E711
            if not len(keep):
                continue
            bank_stream = replace(
                stream, windows=stream.windows[keep], gt=[stream.gt[row] for row in keep],
                subjects=np.asarray(stream.subjects)[keep],
                event_ids=(np.asarray(stream.event_ids)[keep] if stream.event_ids is not None else None),
                execution_ids=(np.asarray(stream.execution_ids)[keep] if stream.execution_ids is not None else None),
                block_ids=(np.asarray(stream.block_ids)[keep] if stream.block_ids is not None else None),
                lengths=lengths[keep], perturbation=f"training-bank-min-{MIN_RECORDING_SECONDS:g}s",
            )
            yield dataset, stream_id, bank_stream, [str(label) for label in aligned[keep]]


def fit_temperature(score_sets: Sequence[tuple[np.ndarray, np.ndarray]], kind: str) -> dict:
    """Dataset-balanced maximum-likelihood temperature for softmax(T * s)."""
    prepared = []
    for scores, truth in score_sets:
        s = np.asarray(scores, dtype=np.float64)
        if kind == "distance":
            s = s / (float(s.std()) or 1.0)
        prepared.append((s, np.asarray(truth, dtype=np.int64)))

    def nll(log_t: float) -> float:
        t = float(np.exp(log_t))
        losses = []
        for s, y in prepared:
            z = t * s
            z = z - z.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            losses.append(-logp[np.arange(len(y)), y].mean())
        return float(np.mean(losses))

    result = minimize_scalar(nll, bounds=LOG_T_BOUNDS, method="bounded", options={"xatol": 1e-4})
    t = float(np.exp(result.x))
    at_bound = bool(abs(result.x - LOG_T_BOUNDS[0]) < 1e-3 or abs(result.x - LOG_T_BOUNDS[1]) < 1e-3)

    def ece(temp: float) -> float:
        conf, correct = [], []
        for s, y in prepared:
            z = temp * s
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
            conf.append(p.max(1))
            correct.append(p.argmax(1) == y)
        conf, correct = np.concatenate(conf), np.concatenate(correct)
        bins = np.minimum((conf * 15).astype(int), 14)
        return float(sum(abs(conf[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
                         for b in range(15) if (bins == b).any()))

    return {"temperature": t, "nll": nll(result.x), "nll_at_30": nll(np.log(30.0)),
            "ece": ece(t), "ece_at_30": ece(30.0), "at_search_bound": at_bound,
            "accuracy": float(np.mean(np.concatenate([s.argmax(1) == y for s, y in prepared]))),
            "n_windows": int(sum(len(y) for _, y in prepared)), "n_datasets": len(prepared)}


def calibrate(scorer, name: str, window_seconds: float, *, held_out: set[tuple[str, str]],
              device: torch.device, max_rows_per_dataset: int = 1500, seed: int = 0) -> dict:
    """Fit ``name``'s temperature on held-out training-source windows (see the module docstring)."""
    rng = np.random.default_rng(seed)
    streams: dict[str, list] = {}
    needs_bank_features = name in TRAINING_BANK_ZERO_SHOT     # HALO + the ConSE providers
    for dataset, stream_id, bank_stream, labels in _bank_streams(name, window_seconds):
        is_held = np.asarray([(dataset, str(s)) in held_out for s in bank_stream.subjects], dtype=bool)
        if not needs_bank_features:
            # Native text heads score only the held-out windows (below); encoding the whole bank
            # through them would cost a full-corpus pass for nothing.
            if is_held.any():
                streams.setdefault(dataset, []).append((bank_stream, None, np.asarray(labels, dtype=object), is_held))
            continue
        try:
            features, _ = scorer.features(name, bank_stream)
        except Exception as exc:                       # an unsupported stream for this provider
            if exc.__class__.__name__ == "UnsupportedEvaluationCell":
                continue
            raise
        streams.setdefault(dataset, []).append((bank_stream, np.asarray(features, dtype=np.float32),
                                                np.asarray(labels, dtype=object), is_held))
    train_labels = load_global_labels()
    label_id = {label: i for i, label in enumerate(train_labels)}
    kind = SCORE_KIND.get(name, "cosine")
    bank_rows = [(f[~h], np.asarray([label_id[x] for x in lab[~h]], dtype=np.int64))
                 for parts in streams.values() for _, f, lab, h in parts if f is not None and (~h).any()]
    bank_features = np.concatenate([b[0] for b in bank_rows]) if bank_rows else None
    bank_ids = np.concatenate([b[1] for b in bank_rows]) if bank_rows else None
    bridge = None
    if name == "halo" and getattr(scorer, "halo_p_text", None) is None:
        bridge = fit_halo_text_bridge(bank_features, bank_ids, train_labels, sbert=scorer.sbert)[0]
    score_sets, per_dataset = [], {}
    for dataset, parts in sorted(streams.items()):
        feats = (np.concatenate([f[h] for _, f, _, h in parts]) if parts[0][1] is not None else None)
        labs = np.concatenate([lab[h] for _, _, lab, h in parts]) if parts else np.zeros(0, dtype=object)
        if len(labs) == 0:
            continue
        roster = sorted({str(x) for _, _, lab, _ in parts for x in lab})
        if len(roster) < 2:
            continue
        pick = rng.permutation(len(labs))[:max_rows_per_dataset]
        subset_streams = []
        if name in TRAINING_BANK_ZERO_SHOT and name != "halo":
            scores, _ = _training_bank_conse_predictions(feats[pick], bank_features, bank_ids, train_labels,
                                                         roster, device, return_scores=True)
        elif name == "halo":
            if bridge is not None:
                scores = halo_text_scores(feats[pick], bridge, roster, sbert=scorer.sbert)
            else:
                scores, _ = scorer.scores(name, feats[pick], roster, window_seconds)
        else:                                          # native text heads need their own feature role
            # Encode only the sampled held-out rows: map ``pick`` (over the concatenated held rows
            # of this dataset) back to (stream, row) and encode each stream's selection once.
            owners = [(i, r) for i, (_, _, _, h) in enumerate(parts) for r in np.flatnonzero(h)]
            chosen = [owners[j] for j in pick]
            native_rows: dict[tuple[int, int], np.ndarray] = {}
            for i in sorted({i for i, _ in chosen}):
                bank_stream = parts[i][0]
                rows_i = np.asarray(sorted(r for j, r in chosen if j == i), dtype=np.int64)
                sub = replace(bank_stream, windows=bank_stream.windows[rows_i],
                              gt=[bank_stream.gt[r] for r in rows_i],
                              subjects=np.asarray(bank_stream.subjects)[rows_i],
                              event_ids=(np.asarray(bank_stream.event_ids)[rows_i]
                                         if bank_stream.event_ids is not None else None),
                              execution_ids=(np.asarray(bank_stream.execution_ids)[rows_i]
                                             if bank_stream.execution_ids is not None else None),
                              block_ids=(np.asarray(bank_stream.block_ids)[rows_i]
                                         if bank_stream.block_ids is not None else None),
                              lengths=np.asarray(bank_stream.lengths)[rows_i],
                              perturbation=f"{bank_stream.perturbation}-heldout-calibration-s{seed}")
                try:
                    encoded = scorer.features(name, sub, role=zero_shot_feature_role(name))[0]
                except Exception as exc:               # e.g. gravity-removed accel for UniMTS
                    if exc.__class__.__name__ == "UnsupportedEvaluationCell":
                        continue
                    raise
                for r, vector in zip(rows_i.tolist(), np.asarray(encoded)):
                    native_rows[(i, r)] = vector
            usable = [n for n, key in enumerate(chosen) if key in native_rows]
            if not usable:
                continue
            pick = np.asarray(pick)[usable]
            native = np.stack([native_rows[chosen[n]] for n in usable])
            scores, _ = scorer.scores(name, native, roster, window_seconds)
        truth = np.asarray([roster.index(str(x)) for x in labs[pick]], dtype=np.int64)
        score_sets.append((np.asarray(scores, dtype=np.float64), truth))
        per_dataset[dataset] = {"n": int(len(truth)), "roster_size": len(roster),
                                "accuracy": float((np.asarray(scores).argmax(1) == truth).mean())}
    if not score_sets:
        raise RuntimeError(f"no held-out calibration windows for {name}")
    out = fit_temperature(score_sets, kind)
    out.update({"model": name, "kind": kind, "window_seconds": float(window_seconds),
                "version": CALIBRATION_VERSION, "data_seed": DATA_SEED, "per_dataset": per_dataset,
                "route": ("halo_text_bridge_heldout_refit" if bridge is not None else
                          "halo_p_text" if name == "halo" else
                          "training_bank_conse_heldout_subjects" if name in TRAINING_BANK_ZERO_SHOT else "native"),
                "held_out_subjects": len(held_out)})
    return out


def calibrate_all(scorer, names: Sequence[str], window_seconds: float, *, device: torch.device,
                  out_path: Path | None = None, cached: Path | None = None) -> dict[str, dict]:
    """Calibrate every provider (or reuse a previous calibration file for the same checkpoints)."""
    if cached is not None and Path(cached).is_file():
        blob = json.loads(Path(cached).read_text())
        missing = [n for n in names if n not in blob.get("models", {})]
        if not missing:
            return {n: blob["models"][n] for n in names}
    held = held_out_subjects(window_seconds)
    result = {name: calibrate(scorer, name, window_seconds, held_out=held, device=device) for name in names}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "version": CALIBRATION_VERSION, "window_seconds": float(window_seconds),
            "fingerprint": hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest(),
            "models": result}, indent=2) + "\n")
    return result
