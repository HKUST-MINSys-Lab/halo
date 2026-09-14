"""NormWear baseline adapter (bespoke MSiTF / L1 text-matching tier).

NormWear (Luo et al., arXiv 2412.09758) is a channel-INDEPENDENT ViT over ricker-CWT
scalograms of each sensor channel, with a query-conditioned MSiTF aggregator that fuses
the per-channel patch tokens into a single 2048-d vector aligned to a frozen Clinical
TinyLlama (~1.1B) text encoder. Zero-shot HAR compares the signal embedding to each
candidate label's TinyLlama embedding by MANHATTAN (L1) distance, argmin.

This is NEITHER plain ConSE nor plain cosine: the sensor/text spaces are asymmetric and
NormWear's native metric is L1 (a dot product has the wrong sign/geometry). So we subclass
:class:`BaselineAdapter` directly and override :meth:`predict` with the bespoke matching.

Input contract (verified in ``docs/baselines/BASELINES.md``):
  * channel-INDEPENDENT -> we feed all REAL acc+gyro channels (drop zero-pad/phantom via
    ``stream.mask``); never fabricated channels into its cross-channel pool.
  * native rate ~65 Hz (its ricker CWT scales are 65 Hz-tuned; ``get_embedding`` only
    resamples internally above 256 Hz, so we resample each window to 65 Hz ourselves and
    pass ``sampling_rate=65`` honestly).
  * 6 s windows -> 390 samples @ 65 Hz.
  * NormWear's per-channel preprocessing: linear de-trend (removes the static gravity DC)
    then amplitude-normalize by mean|x| into its ~unit regime. Unit-agnostic.

Ported faithfully from the working legacy adapter
``legacy_code/val_scripts/human_activity_recognition/evaluate_normwear.py`` +
``benchmark_data/scripts/preprocess_normwear.py`` (the 65 Hz resample, which the legacy did
offline, is folded in here per-window). Reuses the on-disk NormWear repo model code + the
released checkpoints; a clean vendored copy under ``baselines/normwear/repo/`` is a follow-up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

from baselines import data as eval_data

from ..base import BaselineAdapter, InputContract, register

# ---------------------------------------------------------------------------
# On-disk NormWear repo + released checkpoints (reused; heavy — do NOT re-download).
# Overridable via env for a future vendored copy under baselines/normwear/repo/.
# NORMWEAR_REPO_PARENT is the package parent so `NormWear.*` relative imports resolve.
# ---------------------------------------------------------------------------
_DEFAULT_AUX = Path("/home/alex/code/HALO/legacy_code/auxiliary_repos")
NORMWEAR_REPO_PARENT = Path(os.environ.get("NORMWEAR_REPO_PARENT", str(_DEFAULT_AUX)))
NORMWEAR_REPO = NORMWEAR_REPO_PARENT / "NormWear"
BACKBONE_CKPT = NORMWEAR_REPO / "checkpoints" / "normwear_pretrain_ckpt.pth"
MSITF_CKPT = NORMWEAR_REPO / "checkpoints" / "normwear_msitf_zeroshot_last_checkpoint-5.pth"
CLINICAL_LM_ID = "muzammil-eds/tinyllama-2.5T-Clinical-v2"
CLINICAL_LM_REVISION = "60edacbb121c27e4e10b33db00ee8f54b307b175"
CLINICAL_LM_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--muzammil-eds--tinyllama-2.5T-Clinical-v2"
    / "snapshots" / CLINICAL_LM_REVISION
)
CLINICAL_LM_REF = CLINICAL_LM_CACHE.parents[1] / "refs" / "main"

EMB_DIM = 2048
TARGET_HZ = 65                # NormWear native rate; ricker-CWT scales are tuned for it.
WINDOW_65 = TARGET_HZ * 6     # 390 samples = 6 s @ 65 Hz.
QUERY = "What is the current activity?"            # native 'activity' question_template[0]
ANSWER_TEMPLATE = "This subject is presently {}."  # native 'activity' answer_template[0]


# ---------------------------------------------------------------------------
# NormWear repo glue (ported verbatim from the legacy evaluate_normwear.py).
# ---------------------------------------------------------------------------
def _load_normwear_model(device):
    """NormWearZeroShot(backbone + MSiTF aggregator + frozen Clinical-TinyLlama text head),
    everything frozen, with the pure-torch ricker CWT enabled. TinyLlama loads from HF on
    first run (cached thereafter)."""
    if not CLINICAL_LM_REF.is_file() or CLINICAL_LM_REF.read_text().strip() != CLINICAL_LM_REVISION:
        raise RuntimeError(
            "NormWear Clinical TinyLlama cache is not pinned to the audited revision "
            f"{CLINICAL_LM_REVISION}"
        )
    if str(NORMWEAR_REPO_PARENT) not in sys.path:
        sys.path.insert(0, str(NORMWEAR_REPO_PARENT))
    from NormWear.zero_shot.msitf_fusion import NormWearZeroShot  # noqa: E402

    model = NormWearZeroShot(
        weight_path=str(BACKBONE_CKPT),
        msitf_ckpt=str(MSITF_CKPT),
        use_query=True,
        rel_only=False,
    ).to(device).eval()
    if (not CLINICAL_LM_CACHE.is_dir()
            or CLINICAL_LM_REF.read_text().strip() != CLINICAL_LM_REVISION):
        raise RuntimeError("NormWear text-model revision changed during setup")
    model.sensor_model.optimized_cwt = True   # avoid the removed scipy.signal.cwt path
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def _compute_query(model) -> torch.Tensor:
    """One task-query embedding (1, 2048) reused for every window (native HAR protocol)."""
    return model.txt_encode([QUERY])


@torch.no_grad()
def _signal_encode_np(model, x_np, query, device):
    """GPU-correct reimplementation of NormWearZeroShot.signal_encode.

    The released signal_encode does ``device = x.device`` then get_embedding does
    ``x.numpy()`` — self-contradictory on GPU. get_embedding wants a numpy/CPU input and
    uses its ``device`` arg to move the spectrogram + backbone onto the GPU. We pass numpy
    x + device=cuda and replicate the exact query-broadcast + aggregator call.
    """
    precision = os.environ.get("NORMWEAR_PRECISION", "fp16").lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError("NORMWEAR_PRECISION must be fp16 or fp32")
    use_amp = torch.device(device).type == "cuda" and precision == "fp16"
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        sensor_out = model.sensor_model.get_embedding(
            x_np, sampling_rate=TARGET_HZ, device=device
        )  # (bn,nvar,P,768)
        q = query.expand(sensor_out.shape[0], query.shape[1]) if query.shape[0] == 1 else query
        bn, nvar, P, E = sensor_out.shape
        q = q.unsqueeze(1).expand(bn, nvar * P, q.shape[1])
        return model.aggregator(
            sensor_out,
            q,
            device=device,
            rel_only=model.rel_only,
            use_query=model.use_query,
        )  # (bn,2048)


@torch.no_grad()
def _encode_labels(label_strings: Sequence[str], model, device) -> np.ndarray:
    """(L, 2048) TinyLlama embeddings of the candidate labels in NormWear's answer template."""
    # De-underscore to match the shared baseline text convention.
    sents = [ANSWER_TEMPLATE.format(l.replace("_", " ").strip()) for l in label_strings]
    return model.txt_encode(sents).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Preprocessing (folds in preprocess_normwear.py's 65 Hz resample, per-window).
# ---------------------------------------------------------------------------
def _to_normwear_input(windows: np.ndarray, mask: np.ndarray, rate_hz: float) -> np.ndarray:
    """(N, T, C) native windows -> (N, Creal, 390) NormWear-ready float32.

    Drops zero-pad/phantom channels (channel-independent model), resamples each window to
    65 Hz via anti-aliased polyphase filtering, de-trends (removes the static gravity DC),
    and amplitude-normalizes by mean|x| — NormWear's native per-channel preprocessing.
    """
    from scipy import signal as _sig

    mask = np.asarray(mask, dtype=bool)
    X = np.asarray(windows, dtype=np.float64)[:, :, mask]   # (N, T, Creal) real channels only
    if X.shape[2] == 0:
        raise ValueError("NormWear: no real channels after masking — nothing to encode.")

    # Resample each 6 s window to exactly 390 samples (65 Hz). resample_poly is exact when the
    # native rate divides evenly; pad/truncate a tiny rounding drift to lock the length.
    orig = int(round(rate_hz))
    if orig != TARGET_HZ:
        g = np.gcd(TARGET_HZ, orig)
        X = _sig.resample_poly(X, TARGET_HZ // g, orig // g, axis=1)
    if X.shape[1] < WINDOW_65:
        X = np.pad(X, ((0, 0), (0, WINDOW_65 - X.shape[1]), (0, 0)), mode="edge")
    X = X[:, :WINDOW_65, :]

    X = np.transpose(X, (0, 2, 1))                          # (N, Creal, 390)
    X = _sig.detrend(X, axis=2, type="linear")             # remove linear trend incl. gravity DC
    X = X / (np.mean(np.abs(X), axis=2, keepdims=True) + 1e-6)
    return np.ascontiguousarray(X, dtype=np.float32)       # C-contiguous: calc_cwt uses .view()


def _normwear_chunks(stream) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Prepare native 6 s chunks while retaining every declared device and sample.

    Rows are grouped by their valid-length tuple before resampling.  Evaluation grids usually
    contain thousands of equal-length rows; batching them turns thousands of tiny SciPy calls into
    one polyphase operation per device and length group without changing the physical-time rule.
    """
    members = stream.devices if isinstance(stream, eval_data.MultiDeviceEvalStream) else [stream]
    chunks: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    n_rows = members[0].n_windows
    if any(member.n_windows != n_rows for member in members):
        raise ValueError("aligned NormWear devices disagree on window count")
    valid_lengths = np.stack([
        np.asarray(member.lengths, dtype=np.int64)
        if member.lengths is not None
        else np.full(n_rows, member.windows.shape[1], dtype=np.int64)
        for member in members
    ], axis=1)
    groups: dict[tuple[int, ...], list[int]] = {}
    for row, length_tuple in enumerate(valid_lengths.tolist()):
        groups.setdefault(tuple(int(value) for value in length_tuple), []).append(row)

    from scipy import signal as _sig
    total_channels = 0
    for length_tuple, group_rows in groups.items():
        rows = np.asarray(group_rows, dtype=np.int64)
        prepared: list[np.ndarray] = []
        target_length = None
        for member, valid in zip(members, length_tuple):
            raw = member.windows[rows, :valid]
            # Reuse the released preprocessing one physical 6 s chunk at a time below; here only
            # resample to its native 65 Hz clock and retain real channels in declared order.
            mask = np.asarray(member.mask, dtype=bool)
            values = np.asarray(raw[:, :, mask], dtype=np.float64)
            source = int(round(member.rate_hz))
            if source != TARGET_HZ:
                divisor = np.gcd(TARGET_HZ, source)
                values = _sig.resample_poly(values, TARGET_HZ // divisor, source // divisor, axis=1)
            if target_length is None:
                target_length = values.shape[1]
            if values.shape[1] != target_length:
                raise ValueError("aligned NormWear devices disagree on physical window duration")
            prepared.append(values)
        assert target_length is not None
        combined = np.concatenate(prepared, axis=2)
        total_channels = combined.shape[2]
        chunk_count = int(np.ceil(target_length / WINDOW_65))
        padded_length = chunk_count * WINDOW_65
        if padded_length != target_length:
            combined = np.pad(
                combined, ((0, 0), (0, padded_length - target_length), (0, 0)), mode="edge",
            )
        chunks.append(combined.reshape(len(rows) * chunk_count, WINDOW_65, total_channels))
        owners.append(np.repeat(rows, chunk_count))
        chunk_weights = np.ones(chunk_count, dtype=np.float32)
        remainder = target_length % WINDOW_65
        if remainder:
            chunk_weights[-1] = remainder / WINDOW_65
        weights.append(np.tile(chunk_weights, len(rows)))

    values = np.transpose(np.concatenate(chunks, axis=0), (0, 2, 1))
    values = _sig.detrend(values, axis=2, type="linear")
    values /= np.mean(np.abs(values), axis=2, keepdims=True) + 1e-6
    return (np.ascontiguousarray(values, dtype=np.float32), np.concatenate(owners),
            np.concatenate(weights).astype(np.float32, copy=False), total_channels)


@register
class NormWearAdapter(BaselineAdapter):
    """NormWear zero-shot: MSiTF signal embedding vs candidate-label TinyLlama embeddings,
    matched by minimum Manhattan (L1) distance (its native metric)."""

    name = "normwear"
    tier = "bespoke"
    contract = InputContract(channels=None, rate_hz=float(TARGET_HZ), native_window_sec=6.0)
    supports_multi_device = True

    def supports_native_zero_shot(self) -> bool:
        return True

    def evaluation_artifacts(self, state):
        return {
            "backbone": BACKBONE_CKPT,
            "zero_shot_fusion": MSITF_CKPT,
            "text_model": CLINICAL_LM_CACHE / "model.safetensors",
            "text_tokenizer": CLINICAL_LM_CACHE / "tokenizer.json",
        }

    def evaluation_source_paths(self):
        return (NORMWEAR_REPO,)

    def evaluation_config(self, state):
        return {
            "input_rate_hz": TARGET_HZ,
            "input_samples": WINDOW_65,
            "real_channels_only": True,
            "native_metric": "manhattan_l1",
            "text_model": CLINICAL_LM_ID,
            "text_model_revision": CLINICAL_LM_REVISION,
            "inference_precision": os.environ.get("NORMWEAR_PRECISION", "fp16").lower(),
        }

    def setup(self, device):
        model = _load_normwear_model(device)
        query_emb = _compute_query(model)   # (1, 2048), reused for every window
        return {"model": model, "query_emb": query_emb}

    @torch.no_grad()
    def window_features(self, stream: eval_data.EvalStream, state, device) -> np.ndarray:
        model, query_emb = state["model"], state["query_emb"]
        inputs, owners, weights, n_channels = _normwear_chunks(stream)
        outputs = []
        configured_batch = os.environ.get("NORMWEAR_BATCH")
        # CWT/backbone activation memory scales approximately with batch * measured channels.
        # Keep that product bounded so six-channel cells can fill the 4090 while native
        # multi-device cells automatically use a safe batch. An explicit environment override is
        # retained for reproduction on different hardware.
        batch = (int(configured_batch) if configured_batch is not None
                 else min(128, max(1, 768 // n_channels)))
        if batch <= 0:
            raise ValueError("NORMWEAR_BATCH must be positive")
        for start in range(0, len(inputs), batch):
            embedding = _signal_encode_np(
                model, inputs[start:start + batch], query_emb, device
            )
            outputs.append(embedding.float().cpu().numpy())
        encoded = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        result = np.zeros((stream.n_windows, encoded.shape[1]), dtype=np.float32)
        denom = np.zeros(stream.n_windows, dtype=np.float32)
        for feature, owner, weight in zip(encoded, owners, weights):
            result[owner] += feature * weight
            denom[owner] += weight
        state["_last_n_channels_used"] = int(n_channels)
        return result / denom[:, None]

    @torch.no_grad()
    def predict(self, stream: eval_data.EvalStream, state, device) -> Tuple[List[str], dict]:
        return self.predict_candidates(stream, stream.eval_labels, state, device)

    @torch.no_grad()
    def predict_candidates(self, stream, candidates, state, device) -> Tuple[List[str], dict]:
        win = self.window_features(stream, state, device)                      # (N, 2048)
        predictions, info = self.predict_candidates_from_features(
            win, candidates, state, device
        )
        info["n_channels_used"] = int(state.get("_last_n_channels_used", 0))
        return predictions, info

    def predict_candidates_from_features(self, features, candidates, state, device):
        win = np.asarray(features)

        candidates = list(candidates)
        cache = state.setdefault("_candidate_embedding_cache", {})
        cache_key = tuple(candidates)
        if cache_key not in cache:
            cache[cache_key] = _encode_labels(
                candidates, state["model"], device
            ).astype(np.float32, copy=False)
        lab = cache[cache_key]                                                 # (L, 2048)

        # scipy's C implementation preserves NormWear's native L1 rule without allocating
        # the otherwise very large (windows, labels, 2048) broadcast temporary.
        from scipy.spatial.distance import cdist

        dist = cdist(win, lab, metric="cityblock")                            # (N, L)
        idx = dist.argmin(axis=1)
        preds = [candidates[i] for i in idx]

        info = {
            "predicted_classes": sorted(set(preds)),
            "match_metric": "l1_argmin",
        }
        return preds, info
