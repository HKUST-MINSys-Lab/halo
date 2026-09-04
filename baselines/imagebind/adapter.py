"""ImageBind adapter (cosine tier): Meta's 6-modality joint embedding space, scored
zero-shot on HAR via its IMU<->text towers.

ImageBind (Girdhar et al., CVPR'23) binds vision/text/audio/depth/thermal/IMU into ONE
1024-d space by contrastive alignment to the image tower. Its IMU tower was trained on
Ego4D HEAD-MOUNTED IMU (via the IMU2CLIP pipeline), so zero-shot phone/watch HAR = cosine
similarity between a window's IMU embedding and each candidate label string's text embedding
in the shared 1024-d space -> the "cosine" adapter tier (its own text tower, no ConSE bridge).

Expectation (this is the point, not a bug): the IMU tower saw only single-placement,
head-mounted motion, so it transfers POORLY to phone/watch HAR — prior work (UniMTS,
NeurIPS'24) measures ImageBind at ~12.5% zero-shot accuracy / ~7.8 macro-F1. A low-but-
non-degenerate score on motionsense is the CORRECT "generic multimodal binding doesn't
transfer" reference floor; we only guard against degeneracy (>1 predicted class, no crash).

Input contract (sourced from ImageBind's architecture + Meta's IMU2CLIP preprocessing, since
the released ImageBind repo ships NO ``load_and_transform_imu_data``):
  * IMU tensor shape (B, 6, 2000): 6 channels = accel xyz + gyro xyz, 2000 time steps.
    - 6 channels & 2000 steps are hard architectural constants: ``IMUPreprocessor(img_size=[6,
      2000], kernel_size=8)`` patchifies into 250 non-overlapping length-8 patches -> a
      Linear(in_features=48=6*8) stem (imagebind_model.py).
  * 200 Hz, 2000 samples = 10 s. Meta's IMU2CLIP (the pipeline that generated ImageBind's IMU
    training data) resamples every Ego4D clip to 200 Hz (``resampleIMU``) and pads/crops to
    ``round(duration_sec)*200`` (``padIMU``); img_size[1]=2000 => a 10 s window at 200 Hz.
  * accel in m/s^2 WITH gravity, gyro in rad/s — Ego4D IMU native units (IMU2CLIP passes raw
    resampled values, NO per-sample mean/std normalization). Our grids store accel in g, so we
    x9.80665; our gyro is already rad/s (motionsense gyro |.|~O(1-20), consistent with rad/s).
  * short windows are zero-padded to 2000 (IMU2CLIP's ``padIMU`` appends zeros); longer ones
    are cropped to the first 2000 samples.

Reused ImageBind code: the model (``imagebind_model.imagebind_huge``) and text tokenizer
(``multimodal_preprocessors.SimpleTokenizer`` == the tokenizer behind
``data.load_and_transform_text``). Resampling mirrors IMU2CLIP's ``torchaudio.functional.
resample``. We L2-normalize both towers' outputs ourselves (ImageBind's postprocessors L2-norm
then apply a per-modality LearnableLogitScaling, so raw outputs are scaled off the unit sphere).

Install: cloned github.com/facebookresearch/ImageBind into ``baselines/imagebind/repo/``
(the pip ``imagebind`` package is a 1.1 kB empty stub). The repo's ``imagebind/__init__.py``
eagerly imports ``data``, which pulls ``pytorchvideo`` (broken on modern torchvision); we don't
need video/audio, so we register a package stub in ``sys.modules`` that exposes the repo's
submodules WITHOUT running that __init__, and import the model + tokenizer directly.
Checkpoint ``imagebind_huge.pth`` (~4.5 GB) is cached under ``repo/.checkpoints/`` (gitignored).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from ..base import CosineAdapter, InputContract, register

# --- on-disk locations (cloned upstream repo + released checkpoint; both gitignored) ---
_HERE = Path(__file__).resolve().parent
IMAGEBIND_REPO = _HERE / "repo"
IMAGEBIND_PKG = IMAGEBIND_REPO / "imagebind"
IMAGEBIND_CKPT = IMAGEBIND_REPO / ".checkpoints" / "imagebind_huge.pth"
BPE_PATH = IMAGEBIND_PKG / "bpe" / "bpe_simple_vocab_16e6.txt.gz"

# --- input config (verified against the architecture + IMU2CLIP preprocessing) ---
GRAVITY_MS2 = 9.80665   # our grids store accel in g; ImageBind/Ego4D IMU expects m/s^2 WITH gravity
TARGET_HZ = 200.0       # IMU2CLIP resamples every clip to 200 Hz
PAD_LEN = 2000          # img_size[1]: 10 s @ 200 Hz; short windows zero-padded, long ones cropped
N_CH = 6                # accel xyz + gyro xyz (Linear stem in_features = 6*kernel_size(8) = 48)
EMB_DIM = 1024          # out_embed_dim of imagebind_huge

_IMU_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")


def _ensure_imagebind_importable():
    """Make ImageBind's model + tokenizer importable WITHOUT running the repo package's
    ``__init__`` (which imports ``data`` -> ``pytorchvideo``, broken on modern torchvision and
    unneeded here). We insert a synthetic ``imagebind`` package whose ``__path__`` points at the
    repo so ``imagebind.models.*`` submodule imports resolve, but the __init__ body never runs.
    """
    if str(IMAGEBIND_PKG.parent) not in sys.path:
        sys.path.insert(0, str(IMAGEBIND_PKG.parent))
    if "imagebind" not in sys.modules:
        pkg = types.ModuleType("imagebind")
        pkg.__path__ = [str(IMAGEBIND_PKG)]  # namespace so submodules import, __init__ skipped
        sys.modules["imagebind"] = pkg


def _imu_channel_indices(channels):
    """(6,) indices into the grid's channels for accel xyz + gyro xyz, or None per channel if a
    channel is absent (e.g. an accel-only dataset has no gyro -> those are zero-filled)."""
    idx = {c: i for i, c in enumerate(channels)}
    if not all(c in idx for c in ("acc_x", "acc_y", "acc_z")):
        raise ValueError(f"ImageBind IMU needs accel channels acc_x/y/z; got {channels}")
    return [idx.get(c) for c in _IMU_CHANNELS]


def _resample(sig: np.ndarray, rate_hz: float):
    """(N,T,6) at rate_hz -> (N,T2,6) at 200 Hz via torchaudio.functional.resample (as IMU2CLIP)."""
    import torch
    import torchaudio

    if abs(rate_hz - TARGET_HZ) < 1e-6:
        return sig
    x = torch.from_numpy(np.ascontiguousarray(sig)).float().permute(0, 2, 1).contiguous()  # (N,6,T)
    y = torchaudio.functional.resample(x, orig_freq=int(round(rate_hz)), new_freq=int(TARGET_HZ))
    return y.permute(0, 2, 1).numpy()  # (N,T2,6)


def _pad_or_crop(sig: np.ndarray) -> np.ndarray:
    """(N,T,6) -> (N,2000,6): crop to first 2000 or zero-pad the tail (IMU2CLIP ``padIMU``)."""
    N, T, C = sig.shape
    if T >= PAD_LEN:
        return sig[:, :PAD_LEN]
    out = np.zeros((N, PAD_LEN, C), np.float32)
    out[:, :T] = sig
    return out


@register
class ImageBindAdapter(CosineAdapter):
    name = "imagebind"
    # 6-ch accel+gyro, 200 Hz, 10 s window (short windows zero-padded internally).
    contract = InputContract(channels=_IMU_CHANNELS, rate_hz=TARGET_HZ, window_sec=PAD_LEN / TARGET_HZ)

    def evaluation_artifacts(self, state):
        return {"released_checkpoint": IMAGEBIND_CKPT, "text_vocabulary": BPE_PATH}

    def evaluation_source_paths(self):
        return (IMAGEBIND_REPO,)

    def feature_artifacts(self, state):
        # Application tasks consume only the IMU tower; text tokenization is not
        # part of the cached signal representation.
        return {"released_checkpoint": IMAGEBIND_CKPT}

    def evaluation_config(self, state):
        return {
            "input_rate_hz": TARGET_HZ,
            "input_samples": PAD_LEN,
            "input_channels": list(_IMU_CHANNELS),
        }

    def feature_config(self, state):
        return {
            "input_rate_hz": TARGET_HZ,
            "input_samples": PAD_LEN,
            "input_channels": list(_IMU_CHANNELS),
            "feature_layer": "imu_tower",
        }

    def _setup_imu(self, device):
        import torch

        _ensure_imagebind_importable()
        from imagebind.models import imagebind_model  # noqa: E402
        from imagebind.models.imagebind_model import ModalityType  # noqa: E402

        if not IMAGEBIND_CKPT.exists():
            raise FileNotFoundError(
                f"ImageBind checkpoint missing at {IMAGEBIND_CKPT}. Download it once:\n"
                f"  curl -L -o {IMAGEBIND_CKPT} "
                f"https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
            )
        model = imagebind_model.imagebind_huge(pretrained=False)
        sd = torch.load(str(IMAGEBIND_CKPT), map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, (
            f"ImageBind load mismatch: missing={missing} unexpected={unexpected}")
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return {"model": model, "ModalityType": ModalityType}

    def setup_features(self, device):
        """Load the IMU tower without the text tokenizer unused by task caches."""

        return self._setup_imu(device)

    def setup(self, device):
        """Load frozen imagebind_huge (1024-d joint space) from the cached checkpoint."""
        state = self._setup_imu(device)
        from imagebind.models.multimodal_preprocessors import SimpleTokenizer  # noqa: E402

        tokenizer = SimpleTokenizer(bpe_path=str(BPE_PATH))
        return {**state, "tokenizer": tokenizer}

    def window_embeddings(self, stream, state, device, batch=64) -> np.ndarray:
        """(N,1024) L2-normalized IMU embeddings.

        native (N,T,6) g -> accel->m/s^2 (gyro rad/s untouched) -> resample to 200 Hz ->
        zero-pad/crop to 2000 -> (N,6,2000) -> ImageBind IMU tower -> (N,1024).
        Absent channels (e.g. gyro on an accel-only dataset) are zero-filled.
        """
        import torch

        model, ModalityType = state["model"], state["ModalityType"]
        cidx = _imu_channel_indices(stream.channels)
        win = np.asarray(stream.windows, np.float32)                    # (N,T,C) in g
        N, T, _ = win.shape
        sig = np.zeros((N, T, N_CH), np.float32)
        for k, src in enumerate(cidx):
            if src is not None:
                sig[:, :, k] = win[:, :, src]
        sig[:, :, 0:3] *= GRAVITY_MS2                                   # accel g -> m/s^2; gyro as-is
        sig = _resample(sig, stream.rate_hz)                           # (N,T2,6) @ 200 Hz
        sig = _pad_or_crop(sig)                                         # (N,2000,6)

        embs = []
        for s in range(0, N, batch):
            x = torch.from_numpy(sig[s:s + batch]).to(device).permute(0, 2, 1).contiguous()  # (b,6,2000)
            with torch.no_grad():
                e = model({ModalityType.IMU: x})[ModalityType.IMU]     # (b,1024), L2-norm*logit_scale
            e = e / e.norm(dim=-1, keepdim=True)                       # pure unit-norm for cosine
            embs.append(e.float().cpu().numpy())
        return np.concatenate(embs, axis=0)

    def encode_labels(self, labels, state, device) -> np.ndarray:
        """(L,1024) L2-normalized text embeddings via ImageBind's text tower + SimpleTokenizer
        (the tokenizer behind data.load_and_transform_text). Raw label string, no template."""
        import torch

        model, ModalityType, tok = state["model"], state["ModalityType"], state["tokenizer"]
        # de-underscore to MATCH eval.scoring's ConSE humanization (scoring.py:400) -> symmetric label text (audit Q1).
        toks = torch.cat([tok(s.replace("_", " ").strip()).unsqueeze(0) for s in labels], dim=0).to(device)  # (L,77)
        with torch.no_grad():
            t = model({ModalityType.TEXT: toks})[ModalityType.TEXT]    # (L,1024)
            t = t / t.norm(dim=-1, keepdim=True)
        return t.float().cpu().numpy()
