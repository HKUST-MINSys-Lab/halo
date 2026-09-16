"""UniMTS adapter (cosine tier): a text-aligned ST-GCN accelerometer encoder + fine-tuned CLIP
text tower -> per-window 512-d embeddings scored by cosine similarity to label-text embeddings.

UniMTS (Zhang et al., NeurIPS'24) is an ST-GCN over a 22-node SMPL skeleton graph contrastively
aligned to a fine-tuned CLIP ViT-B/32 text tower. Zero-shot HAR = cosine similarity between a
window's IMU embedding and each candidate label string's text embedding in the shared 512-d CLIP
space -> the "cosine" adapter tier (its own text tower, no ConSE bridge).

Verified input contract (from the released checkpoint + our BASELINES.md):
  * accelerometer-ONLY, 3 channels (gyro/stft branches OFF in the released weights, in_channels=3);
  * 20 Hz, 10 s = 200 samples (short windows wrap-padded), accel in m/s^2 WITH gravity;
  * placement = each physical IMU is written into ONE joint's 3 accel channels of a fixed 22-joint
    SMPL skeleton, the other 21 joints zero-filled (UniMTS trains with random-joint masking, so
    zero joints are valid at inference);
  * no per-window normalization (an internal BatchNorm handles scaling).

This port keeps the legacy preprocessing faithful (val_scripts/.../evaluate_unimts.py), adapted to
the v2 grid loader: the adapter now receives NATIVE windows (N,T,6) in g at stream.rate_hz instead
of pre-baked 20 Hz limubert grids, so it (a) selects the 3 accel channels by name, (b) resamples
native -> 20 Hz, (c) wrap-pads/truncates to 200 samples, and (d) converts g -> m/s^2 (x9.80665).

NOTE (follow-up): this reuses the UniMTS model code + checkpoint from the legacy repo path via
sys.path. A clean vendored copy under ``baselines/unimts/repo/`` (re-clone from citation.json's
data_or_code_url) is a follow-up.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import resample_poly

from baselines import data as eval_data
from ..base import CosineAdapter, InputContract, UnsupportedEvaluationCell, register

# --- reused-on-disk locations (legacy repo + released checkpoint; see module docstring) ---
_LEGACY_ROOT = Path("/home/alex/code/HALO/legacy_code")
UNIMTS_REPO = _LEGACY_ROOT / "auxiliary_repos" / "UniMTS"
UNIMTS_CKPT = UNIMTS_REPO / "checkpoint" / "UniMTS.pth"
LABEL_DICTIONARIES = Path(__file__).resolve().parent / "sealed_label_dictionaries.json"

# --- input config (verified against the released code) ---
GRAVITY_MS2 = 9.80665  # our grids store accel in g; UniMTS expects m/s^2 WITH gravity
TARGET_HZ = 20.0       # UniMTS operates at 20 Hz
PAD_LEN = 200          # 10 s @ 20 Hz; every window is wrap-padded/truncated to this
N_JOINTS = 22          # SMPL skeleton graph nodes
EMB_DIM = 512          # shared CLIP ViT-B/32 space

# Datasets whose accel is gravity-REMOVED (linear accel): |acc| never reaches 1 g and the signed
# DC/gravity cue is legitimately ~0. UniMTS needs gravity-present accel, so it must NOT be scored on
# these — the adapter discloses them as N/A rather than report a physically-invalid number (#91b).
# Mirrors legacy accel_units.GRAVITY_REMOVED.
GRAVITY_INCOMPATIBLE = frozenset({"kuhar"})

# Placement -> SMPL joint index. A single IMU stream is placed at one joint; others zero-filled.
# Joint semantics (from the graph child->parent tree + UniMTS data.py assignments):
#   0 = pelvis/root; left leg = 1 hip, 2 knee, 3 ankle, 4 foot;
#   right leg = 5 hip, 6 knee, 7 ankle, 8 foot; 9 = spine1; 17/21 = L/R wrist.
# Matched on the stream id first (placement-derived), then a per-dataset fallback, then pelvis.
DEFAULT_JOINT = 0
_SIDE_PLACEMENT_JOINTS = [
    # Exact deployment stream ids come first. This keeps simultaneous left/right devices on
    # distinct SMPL joints instead of letting the generic ``pocket`` rule collapse both to R-hip.
    ("phone_left_pocket", 1), ("phone_right_pocket", 5),
    ("phone_belt", 9), ("watch_wrist_proxy", 21),
    ("phone_forearm", 20), ("phone_thigh", 5), ("phone_waist", 9),
    ("left rectus femoris", 1), ("left_rectus_femoris", 1),
    ("left hamstrings", 1), ("left_hamstrings", 1),
    ("right rectus femoris", 5), ("right_rectus_femoris", 5),
    ("right hamstrings", 5), ("right_hamstrings", 5),
    ("left tibialis anterior", 2), ("left_tibialis_anterior", 2),
    ("left gastrocnemius", 2), ("left_gastrocnemius", 2),
    ("right tibialis anterior", 6), ("right_tibialis_anterior", 6),
    ("right gastrocnemius", 6), ("right_gastrocnemius", 6),
    ("left wrist", 17), ("left_wrist", 17),
    ("right wrist", 21), ("right_wrist", 21),
    ("left forearm", 16), ("left_forearm", 16),
    ("right forearm", 20), ("right_forearm", 20),
    ("left upper arm", 15), ("left_upper_arm", 15),
    ("right upper arm", 19), ("right_upper_arm", 19),
    ("left chest", 14), ("left_chest", 14),
    ("right chest", 18), ("right_chest", 18),
]
_PLACEMENT_KEYWORDS = [
    ("wrist", 21), ("forearm", 20), ("upper arm", 19), ("upper_arm", 19),
    ("pocket", 5), ("thigh", 5), ("hip", 5),
    ("waist", 9), ("lower_back", 9), ("lowerback", 9), ("lumbar", 9),
    ("belt", 9), ("chest", 9), ("back", 9),
]
# Per-dataset fallback (ported verbatim from the legacy JOINT_BY_DS).
JOINT_BY_DS = {
    "motionsense": 5,    # front trouser pocket -> R-hip
    "mobiact": 5,        # trouser pocket -> R-hip
    "realworld": 9,      # waist -> spine1
    "inclusivehar": 9,   # waist -> spine1
    "harth": 9,          # lower back / thigh -> spine1
    "shoaib": 5,         # multi-position stream -> R-hip default
}


def _joint_for(stream) -> int:
    name = f"{stream.dataset}/{stream.stream}".lower()
    for keyword, joint in _SIDE_PLACEMENT_JOINTS:
        if keyword in name:
            return joint
    for kw, j in _PLACEMENT_KEYWORDS:
        if kw in name:
            return j
    return JOINT_BY_DS.get(stream.dataset, DEFAULT_JOINT)


def _accel_indices(channels):
    """Indices of the 3 accelerometer channels (acc_x/y/z) in grid order."""
    idx = {c: i for i, c in enumerate(channels)}
    missing = [c for c in ("acc_x", "acc_y", "acc_z") if c not in idx]
    if missing:
        raise ValueError(f"UniMTS needs accel channels {('acc_x','acc_y','acc_z')}; "
                         f"missing {missing} in {channels}")
    return [idx["acc_x"], idx["acc_y"], idx["acc_z"]]


def _resample_to_20hz(acc: np.ndarray, rate_hz: float) -> np.ndarray:
    """(N,T,3) at rate_hz -> 20 Hz with polyphase anti-alias filtering."""
    _, length, _ = acc.shape
    # A source-valid short tail may contain fewer than half a target-rate sample.
    # Preserve one measured sample for the declared wrap-padding path instead of
    # producing an empty temporal axis that NumPy cannot pad.
    expected = max(1, int(round(length * TARGET_HZ / float(rate_hz))))
    if expected == length and float(rate_hz) == TARGET_HZ:
        return acc
    ratio = Fraction(TARGET_HZ / float(rate_hz)).limit_denominator(1000)
    out = resample_poly(
        acc.astype(np.float64), ratio.numerator, ratio.denominator, axis=1
    )
    # scipy returns ceil(T*up/down); keep the duration contract exact under non-integer source rates.
    if out.shape[1] > expected:
        out = out[:, :expected]
    elif out.shape[1] < expected:
        out = np.pad(out, ((0, 0), (0, expected - out.shape[1]), (0, 0)), mode="edge")
    return out.astype(np.float32)


@register
class UniMTSAdapter(CosineAdapter):
    name = "unimts"
    # accel-only, 20 Hz, 10 s window (short windows wrap-padded internally).
    contract = InputContract(channels=("acc_x", "acc_y", "acc_z"), rate_hz=20.0,
                             native_window_sec=10.0)
    supports_multi_device = True

    def evaluation_artifacts(self, state):
        return {"released_checkpoint": UNIMTS_CKPT}

    def evaluation_source_paths(self):
        return (UNIMTS_REPO,)

    def evaluation_config(self, state):
        return {
            "input_rate_hz": TARGET_HZ,
            "input_samples": PAD_LEN,
            "input_channels": ["acc_x", "acc_y", "acc_z"],
            "label_text_source": "sealed source-backed dataset dictionaries",
            "label_dictionary_sha256": __import__("hashlib").sha256(
                LABEL_DICTIONARIES.read_bytes()
            ).hexdigest(),
            "window_policy": "published_wrap_pad_or_truncate",
            "metadata_inputs": "sensor placement mapped to released SMPL joint; candidate label text",
        }

    def feature_config(self, state):
        return {
            "input_rate_hz": TARGET_HZ,
            "input_samples": PAD_LEN,
            "input_channels": ["acc_x", "acc_y", "acc_z"],
            "feature_layer": "acc_st_gcn",
            "window_policy": "published_wrap_pad_or_truncate",
            "metadata_inputs": "sensor placement mapped to released SMPL joint",
        }

    def setup(self, device):
        """Load ContrastiveModule (acc-only ST-GCN + fine-tuned CLIP text tower) with UniMTS.pth.

        The pretrained state_dict is ContrastiveModule.model.state_dict(): CLIP text tower (minus
        visual) + logit_scale/text_projection + acc.* ST-GCN. It loads into model.model strict=True.
        """
        import torch

        # UniMTS uses the generic top-level import ``from model import ST_GCN_18``. HALO also owns a
        # package named ``model``; without isolation, whichever adapter runs first poisons the other
        # through sys.modules. Snapshot HALO's package, load UniMTS against its repo-local module, then
        # restore the snapshot. The instantiated classes retain their module objects, so inference is
        # unaffected after the import names are cleaned up.
        saved_model_modules = {
            name: module for name, module in list(sys.modules.items())
            if name == "model" or name.startswith("model.")
        }
        for name in saved_model_modules:
            sys.modules.pop(name, None)
        added_path = str(UNIMTS_REPO) not in sys.path
        if added_path:
            sys.path.insert(0, str(UNIMTS_REPO))
        try:
            sys.modules.pop("contrastive", None)
            from contrastive import ContrastiveModule  # noqa: E402  (repo-local import)
        finally:
            for name in list(sys.modules):
                if name == "model" or name.startswith("model."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_model_modules)
            if added_path:
                sys.path.remove(str(UNIMTS_REPO))

        args = SimpleNamespace(gyro=0, stft=0, stage="evaluation")  # acc-only, no finetune head
        model = ContrastiveModule(args).to(device)
        sd = torch.load(str(UNIMTS_CKPT), map_location=device, weights_only=True)
        missing, unexpected = model.model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, (
            f"UniMTS load mismatch: missing={missing} unexpected={unexpected}")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return {"model": model}

    def is_incompatible(self, dataset):
        if dataset in GRAVITY_INCOMPATIBLE:
            return "gravity-removed accel; UniMTS needs gravity-present"
        return None

    def window_embeddings(self, stream, state, device, batch=256) -> np.ndarray:
        """(N,512) L2-normalized IMU embeddings.

        native (N,T,6) g -> accel 3ch -> resample to 20 Hz -> g->m/s^2 -> place at the placement's
        SMPL joint (others zero) -> wrap-pad/truncate to 200 -> (N,3,200,22,1) -> ST-GCN -> (N,512).
        """
        import torch

        model = state["model"]
        members = stream.devices if isinstance(stream, eval_data.MultiDeviceEvalStream) else [stream]
        joints = [_joint_for(member) for member in members]
        if len(set(joints)) != len(joints):
            raise UnsupportedEvaluationCell(
                f"UniMTS device-to-joint collision: {list(zip(stream.device_ids, joints))}"
            )
        n_windows = members[0].n_windows
        rows: list[np.ndarray] = []
        for row in range(n_windows):
            per_device = []
            target_length = None
            for member in members:
                valid = int(member.lengths[row]) if member.lengths is not None else member.windows.shape[1]
                acc = member.windows[row:row + 1, :valid, _accel_indices(member.channels)]
                acc = _resample_to_20hz(np.asarray(acc, np.float32), member.rate_hz)[0] * GRAVITY_MS2
                if target_length is None:
                    target_length = len(acc)
                if len(acc) != target_length:
                    raise ValueError("aligned UniMTS devices disagree on physical window duration")
                per_device.append(acc)
            assert target_length is not None
            allx = np.zeros((target_length, N_JOINTS, 3), np.float32)
            for acc, joint in zip(per_device, joints):
                allx[:, joint, :] = acc
            if target_length >= PAD_LEN:
                # Released ``load_custom_data`` uses ``resampled_data[:, :padding_size]``.
                allx = allx[:PAD_LEN]
            else:
                repeats = int(np.ceil(PAD_LEN / max(target_length, 1)))
                allx = np.tile(allx, (repeats, 1, 1))[:PAD_LEN]
            rows.append(allx)

        output = []
        for start in range(0, len(rows), batch):
            x = torch.from_numpy(np.asarray(rows[start:start + batch])).to(device) \
                .permute(0, 3, 1, 2).unsqueeze(-1)
            e = model.encode_image(x)
            output.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        if not output:
            raise ValueError("UniMTS received an empty evaluation stream")
        output = np.concatenate(output, axis=0)
        return output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)

    def encode_labels(self, labels, state, device) -> np.ndarray:
        """Encode the released protocol's joined dataset label-dictionary strings."""
        import clip
        import torch

        model = state["model"]
        blob = json.loads(LABEL_DICTIONARIES.read_text())
        joined = blob["joined_text"]
        texts = [joined.get(label, label.replace("_", " ").strip()) for label in labels]
        with torch.no_grad():
            tok = clip.tokenize(texts, truncate=True).to(device)
            encoded = model.encode_text(tok)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        return encoded.float().cpu().numpy()

    def input_accounting(self, stream) -> dict:
        member = stream.devices[0] if isinstance(stream, eval_data.MultiDeviceEvalStream) else stream
        lengths = (np.asarray(member.lengths, dtype=np.float64) if member.lengths is not None
                   else np.full(member.n_windows, member.windows.shape[1], dtype=np.float64))
        durations = lengths / float(member.rate_hz)
        padded = np.maximum(10.0 - durations, 0.0)
        truncated = np.maximum(durations - 10.0, 0.0)
        return {"padded": bool(np.any(padded > 1e-9)),
                "padded_fraction": float(padded.sum() / (10.0 * len(durations))) if durations.size else 0.0,
                "truncated": bool(np.any(truncated > 1e-9)),
                "truncated_fraction": float(truncated.sum() / durations.sum()) if durations.sum() else 0.0,
                "consumed_samples": PAD_LEN, "published_window_contract": True}
