"""Zero-target-enrollment scoring: the training-bank ConSE bridge and its reference bank.

Extracted verbatim from training/support_classifier/sealed_eval.py on 2026-09-23 (Phase 0 of
docs/journal/2026-09-22-rung1-rung1-implementation-plan.md). sealed_eval re-imports these names.
"""

from __future__ import annotations

import baselines
import hashlib
import json
import numpy as np
import torch
from dataclasses import replace
from pathlib import Path
from typing import Sequence
from baselines import scoring
from baselines.data import list_streams, load_eval_stream, load_global_labels
from data.scripts.curate.deployment_policy import (
    SUPERVISED_HEAD_TRAIN_DATASETS,
    assert_no_retired_sources,
)
from training.support_classifier.sampling import MIN_RECORDING_SECONDS
from evaluation.features import FeatureMemoryCache, _load_or_encode


def _normalise(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), np.float32(1e-12))


@torch.no_grad()
def _training_bank_conse_predictions(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    reference_label_ids: np.ndarray,
    train_labels: Sequence[str],
    target_labels: Sequence[str],
    device: torch.device,
    *,
    query_batch_size: int = 1024,
    reference_batch_size: int = 32768,
    return_scores: bool = False,
) -> tuple[list[str] | np.ndarray, dict]:
    """Bridge a training-bank nearest neighbour onto an unseen candidate vocabulary.

    ``k=0`` means zero *target-dataset enrollment*, not zero prior labelled evidence. The closest
    training-corpus recording supplies a distribution over the fixed training vocabulary; ConSE
    then maps that distribution to the target label strings. Chunking the exact matrix search
    bounds VRAM without changing the selected neighbour.
    """
    query = _normalise(query_features).astype(np.float32, copy=False)
    reference = _normalise(reference_features).astype(np.float32, copy=False)
    label_ids = np.asarray(reference_label_ids, dtype=np.int64)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("query and training-bank features must be matching matrices")
    if label_ids.shape != (len(reference),) or not len(reference):
        raise ValueError("training-bank labels must align with a non-empty feature matrix")
    if label_ids.min() < 0 or label_ids.max() >= len(train_labels):
        raise ValueError("training-bank label id outside the declared training vocabulary")

    nearest: list[np.ndarray] = []
    reference_device = [
        torch.as_tensor(reference[start:start + reference_batch_size], device=device)
        for start in range(0, len(reference), reference_batch_size)
    ]
    for start in range(0, len(query), query_batch_size):
        q = torch.as_tensor(query[start:start + query_batch_size], device=device)
        best_score = torch.full((len(q),), -torch.inf, device=device)
        best_row = torch.zeros((len(q),), dtype=torch.long, device=device)
        offset = 0
        for block in reference_device:
            score, row = (q @ block.T).max(dim=1)
            replace = score > best_score
            best_score = torch.where(replace, score, best_score)
            best_row = torch.where(replace, row + offset, best_row)
            offset += len(block)
        nearest.append(best_row.cpu().numpy())
    nearest_rows = np.concatenate(nearest)
    probs = np.zeros((len(query), len(train_labels)), dtype=np.float32)
    probs[np.arange(len(query)), label_ids[nearest_rows]] = 1.0
    if return_scores:
        output: list[str] | np.ndarray = scoring.conse_score_matrix(
            probs, train_labels, target_labels, top_T=1,
        )
        info = {"top_T": 1}
    else:
        output, info = scoring.conse_predict(probs, train_labels, target_labels, top_T=1)
    info.update({
        "zero_support_protocol": "training_bank_1nn_conse_v1",
        "reference_rows": int(len(reference)),
        "reference_labels": int(len(np.unique(label_ids))),
    })
    del reference_device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, info


def _build_training_reference_bank(
    *,
    name: str,
    device: torch.device,
    cache_dir: Path,
    halo_checkpoint: Path | None,
    halo_state: tuple[torch.nn.Module, str] | None = None,
    baseline_state: dict | None = None,
    cache_read_dirs: Sequence[Path] = (),
    memory_cache: FeatureMemoryCache | None = None,
    window_seconds: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Encode the active labelled training roster for zero-target-enrollment scoring."""
    assert_no_retired_sources(SUPERVISED_HEAD_TRAIN_DATASETS)
    train_labels = load_global_labels()
    label_to_id = {label: index for index, label in enumerate(train_labels)}
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    provenance: list[dict] = []
    excluded: list[dict] = []
    for dataset in SUPERVISED_HEAD_TRAIN_DATASETS:
        streams = list_streams(dataset, alignment="native", window_seconds=window_seconds)
        if not streams:
            raise FileNotFoundError(f"zero-shot training bank has no native grid for {dataset}")
        for stream_id in streams:
            stream = load_eval_stream(
                dataset, stream_id, alignment="native", apply_quality_screen=True,
                candidate_labels=train_labels, window_seconds=window_seconds,
            )
            if stream.quality_screen != "applied":
                raise RuntimeError(
                    f"{dataset}/{stream_id}: quality screen unavailable ({stream.quality_screen})"
                )
            aligned = np.asarray(
                scoring.align_ground_truth_labels(stream.gt, train_labels), dtype=object,
            )
            lengths = (
                np.asarray(stream.lengths, dtype=np.int64)
                if stream.lengths is not None
                else np.full(stream.n_windows, stream.windows.shape[1], dtype=np.int64)
            )
            eligible_duration = lengths / float(stream.rate_hz) >= MIN_RECORDING_SECONDS
            keep = np.flatnonzero((aligned != None) & eligible_duration)  # noqa: E711
            if not len(keep):
                continue
            bank_stream = replace(
                stream,
                windows=stream.windows[keep],
                gt=[stream.gt[row] for row in keep],
                subjects=np.asarray(stream.subjects)[keep],
                event_ids=(np.asarray(stream.event_ids)[keep] if stream.event_ids is not None else None),
                execution_ids=(
                    np.asarray(stream.execution_ids)[keep]
                    if stream.execution_ids is not None else None
                ),
                block_ids=(np.asarray(stream.block_ids)[keep] if stream.block_ids is not None else None),
                lengths=lengths[keep],
                perturbation=f"training-bank-min-{MIN_RECORDING_SECONDS:g}s",
            )
            if name == "limubert_x":
                required = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
                channel_index = {channel: index for index, channel in enumerate(bank_stream.channels)}
                missing = [channel for channel in required if channel not in channel_index]
                masked = [
                    channel for channel in required
                    if channel in channel_index and not bool(bank_stream.mask[channel_index[channel]])
                ]
                if missing or masked:
                    excluded.append({
                        "dataset": dataset,
                        "stream": stream_id,
                        "reason": f"LiMU-BERT-X requires measured six-axis IMU; "
                                  f"missing={missing}, masked={masked}",
                    })
                    continue
            try:
                features, fingerprint = _load_or_encode(
                    name=name,
                    stream=bank_stream,
                    device=device,
                    cache_dir=cache_dir,
                    halo_checkpoint=halo_checkpoint,
                    halo_state=halo_state,
                    baseline_state=baseline_state,
                    cache_read_dirs=cache_read_dirs,
                    memory_cache=memory_cache,
                )
            except baselines.UnsupportedEvaluationCell as error:
                excluded.append({"dataset": dataset, "stream": stream_id, "reason": str(error)})
                continue
            feature_parts.append(np.asarray(features, dtype=np.float32))
            label_parts.append(np.asarray([label_to_id[str(aligned[row])] for row in keep], dtype=np.int64))
            provenance.append({
                "dataset": dataset,
                "stream": stream_id,
                "rows": int(len(keep)),
                "feature_fingerprint": fingerprint,
                "quality_excluded": int(stream.n_quality_excluded),
            })
    if not feature_parts:
        raise RuntimeError(f"no valid labelled training references for {name}")
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    fingerprint = hashlib.sha256(json.dumps({
        "provider": name,
        "roster": list(SUPERVISED_HEAD_TRAIN_DATASETS),
        "vocabulary": train_labels,
        "parts": provenance,
        "excluded": excluded,
        "protocol": "training_bank_1nn_conse_v1",
        "window_seconds": float(window_seconds),
    }, sort_keys=True).encode()).hexdigest()
    manifest_dir = cache_dir / "bank_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    duration_token = f"{float(window_seconds):g}".replace(".", "p")
    (manifest_dir / f"{name}__w{duration_token}.json").write_text(json.dumps({
        "provider": name,
        "window_seconds": float(window_seconds),
        "fingerprint": fingerprint,
        "vocabulary": train_labels,
        "included": provenance,
        "excluded": excluded,
        "n_rows": int(len(labels)),
    }, indent=2) + "\n")
    return features, labels, train_labels, fingerprint


# ---------------------------------------------------------------------------------------------
# Added 2026-09-23: one zero-shot score matrix for every provider (rung 1's only model input).
# ---------------------------------------------------------------------------------------------

import torch.nn.functional as F  # noqa: E402

# Providers whose k=0 rule is the training-bank 1-NN + ConSE bridge; the others score candidates
# natively in their own text-aligned space. Moved verbatim from sealed_eval on 2026-09-23.
TRAINING_BANK_ZERO_SHOT = frozenset({"halo", "harnet5", "harnet10", "limubert_x"})

# How each provider's native score matrix is scaled. Every route but NormWear's yields a cosine
# (HALO: bridge cosine in SBERT space; bank bridge: ConSE cosine in SBERT space; UniMTS: its own
# shared space). NormWear's released rule is a negative L1 distance.
SCORE_KIND = {"halo": "cosine", "harnet5": "cosine", "harnet10": "cosine", "limubert_x": "cosine",
              "unimts": "cosine", "normwear": "distance"}


def zero_shot_feature_role(name: str) -> str:
    """Which cached representation the provider's zero-shot rule consumes."""
    return "enrollment" if name in TRAINING_BANK_ZERO_SHOT else "native_zero_shot"


def fit_halo_text_bridge(bank_features: np.ndarray, bank_label_ids: np.ndarray,
                         train_labels: Sequence[str], *, sbert=None) -> tuple[np.ndarray, float, dict]:
    """Closed-form ridge from HALO pooled features to frozen SBERT label vectors.

    The same map the v4 head initialises ``p_text`` with (``train.fit_text_projection``), fitted
    here on the training reference bank so HALO enters rung 1 as encoder + bridge — symmetric with
    the ConSE bridge the representation-tier baselines get — and with no learned head.
    """
    from training.support_classifier.train import fit_text_projection

    sbert = sbert or scoring.get_sbert_encoder()
    targets = np.asarray(sbert(list(train_labels)), dtype=np.float32)[np.asarray(bank_label_ids, dtype=np.int64)]
    x = torch.as_tensor(np.asarray(bank_features, dtype=np.float32))
    t = torch.as_tensor(targets)
    w, alpha = fit_text_projection(x, t)
    cosine = F.cosine_similarity(x @ w, t, dim=-1).mean()
    return w.numpy(), float(alpha), {"n": int(len(x)), "mean_cosine": float(cosine)}


def halo_text_scores(features: np.ndarray, bridge: np.ndarray, candidates: Sequence[str], *,
                     sbert=None) -> np.ndarray:
    sbert = sbert or scoring.get_sbert_encoder()
    text = _normalise(np.asarray(sbert(list(candidates)), dtype=np.float32))
    projected = _normalise(np.asarray(features, dtype=np.float32) @ bridge)
    return projected @ text.T


def zero_shot_scores(*, name: str, features: np.ndarray, candidates: Sequence[str],
                     device: torch.device, adapter_state: dict | None = None,
                     bank: tuple | None = None, halo_bridge: np.ndarray | None = None,
                     sbert=None) -> tuple[np.ndarray, dict]:
    """(N, C) zero-target-enrollment scores over ``candidates``, higher is better, any provider.

    ``features`` must be the representation named by :func:`zero_shot_feature_role`. ``bank`` is
    the ``_build_training_reference_bank`` tuple for bank-bridge providers; ``halo_bridge`` is the
    matrix from :func:`fit_halo_text_bridge`.
    """
    candidates = list(candidates)
    if name == "halo":
        if halo_bridge is None:
            raise ValueError("halo zero-shot scores need a fitted text bridge (fit_halo_text_bridge)")
        return halo_text_scores(features, halo_bridge, candidates, sbert=sbert), \
            {"route": "halo_text_bridge", "kind": "cosine"}
    if name in TRAINING_BANK_ZERO_SHOT:
        if bank is None:
            raise ValueError(f"{name} zero-shot scores need the training reference bank")
        reference, label_ids, train_labels, _ = bank
        scores, info = _training_bank_conse_predictions(
            features, reference, label_ids, train_labels, candidates, device, return_scores=True,
        )
        return np.asarray(scores, dtype=np.float32), {"route": "training_bank_conse", "kind": "cosine", **info}
    if adapter_state is None:
        raise ValueError(f"{name} zero-shot scores need the adapter state from setup_features")
    adapter = baselines.REGISTRY[name]
    scores = adapter.candidate_scores_from_features(np.asarray(features), candidates, adapter_state, device)
    return np.asarray(scores, dtype=np.float32), {"route": "native", "kind": SCORE_KIND.get(name, "cosine")}


def probability_features(scores: np.ndarray, kind: str, *, temperature: float = 30.0,
                         eps: float = 1e-15) -> tuple[np.ndarray, dict]:
    """Rows on the unit simplex, as transductive-CLIP consumes them.

    ``cosine``: softmax(T * s) with T = 30, verbatim from the released config (``T: 30``).
    ``distance``: NormWear's negative-L1 scores have no bounded scale, so they are divided by the
    matrix's standard deviation before the same temperature — a disclosed, provider-specific
    deviation recorded in the returned info. ``probability``: already a distribution; clipped and
    renormalised.
    """
    s = np.asarray(scores, dtype=np.float64)
    info = {"kind": kind, "temperature": float(temperature)}
    if kind == "probability":
        p = np.clip(s, eps, None)
        return (p / p.sum(axis=1, keepdims=True)).astype(np.float32), info
    if kind == "distance":
        scale = float(s.std()) or 1.0
        s = s / scale
        info["distance_scale"] = scale
    elif kind != "cosine":
        raise ValueError(f"unknown score kind {kind!r}")
    z = temperature * s
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32), info


class ProviderScorer:
    """Feature loading and zero-shot score functions for every provider, sharing banks and bridges.

    One object per run: it holds the released-adapter states (NormWear is instantiated once), the
    HALO encoder, the per-duration training reference banks and HALO text bridges. Both rung
    drivers use it so the six providers are scored by exactly one code path.
    """

    def __init__(self, *, models: Sequence[str], device: torch.device, cache_dir: Path,
                 halo_checkpoint: Path | None, cache_read_dirs: Sequence[Path] = (),
                 memory_cache: FeatureMemoryCache | None = None, sbert=None):
        from training.tokenizer.eval_transfer import build_encoder

        from evaluation.features import _file_hash

        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_read_dirs = tuple(Path(p) for p in cache_read_dirs)
        self.memory_cache = memory_cache
        self.halo_checkpoint = halo_checkpoint
        self.halo_state = None
        if "halo" in models:
            if halo_checkpoint is None:
                raise ValueError("--halo-checkpoint is required when the model list includes halo")
            blob = torch.load(halo_checkpoint, map_location="cpu", weights_only=False)
            self.halo_state = (build_encoder(blob, device).eval(), _file_hash(halo_checkpoint))
        self.states = {name: baselines.REGISTRY[name].setup_features(device)
                       for name in models if name != "halo"}
        self.banks: dict[tuple[str, float], tuple] = {}
        self.bridges: dict[float, np.ndarray] = {}
        self._sbert = sbert

    @property
    def sbert(self):
        if self._sbert is None:
            self._sbert = scoring.get_sbert_encoder()
        return self._sbert

    def features(self, name: str, stream, *, role: str = "enrollment") -> tuple[np.ndarray, str]:
        return _load_or_encode(
            name=name, stream=stream, device=self.device, cache_dir=self.cache_dir,
            halo_checkpoint=self.halo_checkpoint, baseline_state=self.states.get(name),
            halo_state=self.halo_state, cache_read_dirs=self.cache_read_dirs,
            memory_cache=self.memory_cache, feature_role=role,
        )

    def bank(self, name: str, window_seconds: float) -> tuple:
        key = (name, float(window_seconds))
        if key not in self.banks:
            self.banks[key] = _build_training_reference_bank(
                name=name, device=self.device, cache_dir=self.cache_dir,
                halo_checkpoint=self.halo_checkpoint, halo_state=self.halo_state,
                baseline_state=self.states.get(name), cache_read_dirs=self.cache_read_dirs,
                memory_cache=self.memory_cache, window_seconds=float(window_seconds),
            )
        return self.banks[key]

    def halo_bridge(self, window_seconds: float) -> np.ndarray:
        key = float(window_seconds)
        if key not in self.bridges:
            bank = self.bank("halo", key)
            self.bridges[key] = fit_halo_text_bridge(bank[0], bank[1], bank[2], sbert=self.sbert)[0]
        return self.bridges[key]

    def scores(self, name: str, features: np.ndarray, classes: Sequence[str],
               window_seconds: float) -> tuple[np.ndarray, dict]:
        if name == "halo":
            return zero_shot_scores(name=name, features=features, candidates=classes, device=self.device,
                                    halo_bridge=self.halo_bridge(window_seconds), sbert=self.sbert)
        if name in TRAINING_BANK_ZERO_SHOT:
            return zero_shot_scores(name=name, features=features, candidates=classes, device=self.device,
                                    bank=self.bank(name, window_seconds))
        return zero_shot_scores(name=name, features=features, candidates=classes, device=self.device,
                                adapter_state=self.states[name])

    def score_fn(self, name: str, classes: Sequence[str], window_seconds: float):
        classes = list(classes)
        return lambda features: self.scores(name, features, classes, window_seconds)[0]
