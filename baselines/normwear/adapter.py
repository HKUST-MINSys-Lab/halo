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
import types
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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
    # SciPy removed signal.cwt, while the release's fixed-length torch approximation is not
    # numerically equivalent. Install a batched torch implementation of the released variable
    # wavelet-length rule instead.
    model.sensor_model.calc_cwt = types.MethodType(_released_calc_cwt, model.sensor_model)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _ricker(points: int, width: float, *, device, dtype) -> torch.Tensor:
    positions = torch.arange(points, device=device, dtype=dtype) - (points - 1.0) / 2.0
    width_t = torch.as_tensor(width, device=device, dtype=dtype)
    amplitude = 2.0 / (torch.sqrt(3.0 * width_t) * torch.pi ** 0.25)
    ratio = positions.square() / width_t.square()
    return amplitude * (1.0 - ratio) * torch.exp(-0.5 * ratio)


def _released_cwt_torch(values: torch.Tensor) -> torch.Tensor:
    """SciPy ``signal.cwt(..., ricker, arange(.1, 65))`` in batched torch.

    SciPy chooses ``min(10 * width, signal_length)`` independently at each scale. Grouping equal
    lengths keeps the implementation batched without replacing that defining rule by one oversized
    kernel. Output is ``(batch, time, 65)`` with exactly the input temporal length.
    """
    if values.ndim != 2:
        raise ValueError("NormWear CWT expects (batch,time)")
    length = int(values.shape[1])
    if length <= 0:
        raise ValueError("NormWear CWT received an empty time axis")
    scales = [0.1 + index for index in range(65)]
    groups: dict[int, list[tuple[int, float]]] = {}
    for index, scale in enumerate(scales):
        points = max(1, min(int(10.0 * scale), length))
        groups.setdefault(points, []).append((index, scale))
    result = torch.empty(
        (values.shape[0], len(scales), length), device=values.device, dtype=values.dtype,
    )
    signal = values.unsqueeze(1)
    for points, entries in groups.items():
        kernels = torch.stack([
            _ricker(points, scale, device=values.device, dtype=values.dtype)
            for _, scale in entries
        ]).unsqueeze(1)
        convolved = F.conv1d(signal, kernels.flip(-1), padding="same")
        # CUDA autocast may emit fp16 while ``values`` and the destination are fp32. Keep CWT
        # storage in the input dtype, matching the released path's explicit ``.float()`` boundary.
        result[:, [index for index, _ in entries]] = convolved.to(result.dtype)
    return result.transpose(1, 2)


def _released_calc_cwt(self, x, device=torch.device("cpu")):
    values = torch.as_tensor(x, dtype=torch.float32, device=device)
    batch, channels, length = values.shape
    first = values[:, :, 1:] - values[:, :, :-1]
    second = first[:, :, 1:] - first[:, :, :-1]
    aligned = torch.stack((values[:, :, 2:], first[:, :, 1:], second), dim=2)
    transformed = _released_cwt_torch(aligned.reshape(batch * channels * 3, length - 2))
    return transformed.reshape(batch, channels, 3, length - 2, 65)


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
def _backbone_encode_np(model, x_np, device):
    """Released downstream representation: patch-mean, then flatten real channels."""
    precision = os.environ.get("NORMWEAR_PRECISION", "fp16").lower()
    use_amp = torch.device(device).type == "cuda" and precision == "fp16"
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        tokens = model.sensor_model.get_embedding(x_np, sampling_rate=TARGET_HZ, device=device)
        # Upstream downstream tasks pool patches and preserve every real channel. Do not average
        # channels merely to manufacture cross-configuration compatibility: that changes the
        # representation and previously cost substantial enrolled accuracy.
        pooled = tokens.mean(dim=2)
        return pooled.reshape(pooled.shape[0], -1)


@torch.no_grad()
def _encode_labels(label_strings: Sequence[str], model, device) -> np.ndarray:
    """(L, 2048) TinyLlama embeddings of the candidate labels in NormWear's answer template."""
    # De-underscore to match the shared baseline text convention.
    sents = [ANSWER_TEMPLATE.format(l.replace("_", " ").strip()) for l in label_strings]
    return model.txt_encode(sents).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Preprocessing (folds in preprocess_normwear.py's 65 Hz resample, per-window).
# ---------------------------------------------------------------------------
def _normwear_groups(stream) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Prepare one released variable-length input per recording.

    Rows are grouped by their valid-length tuple before resampling.  Evaluation grids usually
    contain thousands of equal-length rows; batching them turns thousands of tiny SciPy calls into
    one polyphase operation per device and length group without changing the physical-time rule.
    """
    members = stream.devices if isinstance(stream, eval_data.MultiDeviceEvalStream) else [stream]
    prepared_groups: list[tuple[np.ndarray, np.ndarray]] = []
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
            # Resample to the model's 65 Hz clock and retain real channels in declared order.
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
        values = np.transpose(combined, (0, 2, 1))
        values = _sig.detrend(values, axis=2, type="linear")
        values /= np.mean(np.abs(values), axis=2, keepdims=True) + 1e-6
        prepared_groups.append((np.ascontiguousarray(values, dtype=np.float32), rows))

    return prepared_groups, total_channels


@register
class NormWearAdapter(BaselineAdapter):
    """NormWear zero-shot: MSiTF signal embedding vs candidate-label TinyLlama embeddings,
    matched by minimum Manhattan (L1) distance (its native metric)."""

    name = "normwear"
    tier = "bespoke"
    contract = InputContract(channels=None, rate_hz=float(TARGET_HZ), native_window_sec=None)
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
            "input_samples": "variable, full evaluation interval",
            "real_channels_only": True,
            "native_metric": "manhattan_l1",
            "text_model": CLINICAL_LM_ID,
            "text_model_revision": CLINICAL_LM_REVISION,
            "inference_precision": os.environ.get("NORMWEAR_PRECISION", "fp16").lower(),
            "cwt_implementation": "released-variable-length-ricker-torch-v1",
            "window_policy": "released variable-length single pass",
            "native_zero_shot_feature": "MSiTF_query_conditioned_2048",
            "enrollment_feature": "backbone_mean_patch_then_channel_768",
        }

    def setup(self, device):
        model = _load_normwear_model(device)
        query_emb = _compute_query(model)   # (1, 2048), reused for every window
        return {"model": model, "query_emb": query_emb}

    @torch.no_grad()
    def window_features(self, stream: eval_data.EvalStream, state, device) -> np.ndarray:
        return self._encode_stream(stream, state, device, native=False)

    @torch.no_grad()
    def native_zero_shot_features_for_stream(self, stream, state, device) -> np.ndarray:
        return self._encode_stream(stream, state, device, native=True)

    def feature_config(self, state):
        return {
            **self.evaluation_config(state),
            "feature_role": "common_enrollment",
            "feature_dim": "768 x real sensor channels",
            "feature_layer": "released backbone patch tokens",
            "feature_readout": "mean patches then flatten real channels",
            "feature_readout_provenance": "released downstream recipe",
            "metadata_inputs": "none",
        }

    def native_feature_config(self, state):
        return {
            **self.evaluation_config(state),
            "feature_role": "native_zero_shot",
            "feature_dim": EMB_DIM,
            "feature_layer": "released MSiTF query-conditioned aggregator",
            "metadata_inputs": "fixed activity query text; candidate label text in TinyLlama bridge",
        }

    def _encode_stream(self, stream, state, device, *, native: bool) -> np.ndarray:
        model, query_emb = state["model"], state["query_emb"]
        groups, n_channels = _normwear_groups(stream)
        configured_batch = os.environ.get("NORMWEAR_BATCH")
        # CWT/backbone activation memory scales approximately with batch * measured channels.
        # Keep that product bounded so six-channel cells can fill the 4090 while native
        # multi-device cells automatically use a safe batch. An explicit environment override is
        # retained for reproduction on different hardware.
        # RTX 4090 profiling over the released CWT/backbone path found that filling all available
        # memory was slower than a moderate batch (six channels: 32 beat 64/128).  Keep a roughly
        # constant 192 channel-windows per launch; this also scales down naturally for composites.
        batch = (int(configured_batch) if configured_batch is not None
                 else min(64, max(1, 192 // n_channels)))
        if batch <= 0:
            raise ValueError("NORMWEAR_BATCH must be positive")
        result = None
        for inputs, owners in groups:
            outputs = []
            for start in range(0, len(inputs), batch):
                if native:
                    embedding = _signal_encode_np(
                        model, inputs[start:start + batch], query_emb, device
                    )
                else:
                    embedding = _backbone_encode_np(model, inputs[start:start + batch], device)
                outputs.append(embedding.float().cpu().numpy())
            encoded = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
            if result is None:
                result = np.empty((stream.n_windows, encoded.shape[1]), dtype=np.float32)
            elif result.shape[1] != encoded.shape[1]:
                raise RuntimeError("NormWear feature dimension changed between length groups")
            result[owners] = encoded
        if result is None:
            raise ValueError("NormWear received an empty evaluation stream")
        state["_last_n_channels_used"] = int(n_channels)
        return result

    @torch.no_grad()
    def predict(self, stream: eval_data.EvalStream, state, device) -> Tuple[List[str], dict]:
        return self.predict_candidates(stream, stream.eval_labels, state, device)

    @torch.no_grad()
    def predict_candidates(self, stream, candidates, state, device) -> Tuple[List[str], dict]:
        win = self.native_zero_shot_features_for_stream(stream, state, device)  # (N, 2048)
        predictions, info = self.predict_candidates_from_features(
            win, candidates, state, device
        )
        info["n_channels_used"] = int(state.get("_last_n_channels_used", 0))
        return predictions, info

    def predict_candidates_from_features(self, features, candidates, state, device):
        scores = self.candidate_scores_from_features(features, candidates, state, device)
        candidates = list(candidates)
        idx = scores.argmax(axis=1)
        preds = [candidates[i] for i in idx]
        return preds, {
            "predicted_classes": sorted(set(preds)),
            "match_metric": "l1_argmin",
        }

    def candidate_scores_from_features(self, features, candidates, state, device):
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

        return -cdist(win, lab, metric="cityblock")                           # higher is better
