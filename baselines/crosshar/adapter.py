"""CrossHAR adapter (ConSE tier).

CrossHAR is a self-supervised IMU method (masked-reconstruction + contrastive) —
it ships NO released weights, so its backbone is pretrained ON OUR CORPUS by
``baselines/crosshar/train.py`` (see that file for the full-run command). This
adapter loads that self-pretrained backbone and, exactly like the harnet adapter,
fits a softmax head over our current global training vocabulary
(``data/labels/global_labels.json``) so CrossHAR is scored zero-shot through the
ConSE bridge.

Ported from the working legacy adapter + ``evaluate_crosshar.py``. Carried over:

  * INPUT CONTRACT (verified in BASELINES.md): 6-ch acc+gyro, 20 Hz, 120 samples
    (6 s), with per-channel InstanceNorm (zero-mean/unit-var per axis per window)
    that DISCARDS gravity/DC. Grid windows (native rate/length, g) are mapped to
    that contract by :mod:`baselines.crosshar.prep` and InstanceNorm'd here.
  * ARCHITECTURE (base_v1): a 1-layer, 72-d masked Transformer encoder. The
    module below reproduces the upstream ``MaskedModel4Pretrain`` state_dict
    layout exactly so the self-pretrained checkpoint loads strict.

LEAKAGE-SAFE HEAD-FIT (mirrors harnet): the head is fit on FROZEN backbone
features (mean-pooled sequence embedding) over the current training corpus, selecting
the best epoch on a SUBJECT-DISJOINT held-out fold — no source subject appears in
both the head-train and head-selection fold. Eval targets are a separate held-out
cohort, so there is no target leakage (structural in ZS-XD). The fitted head is
cached to ``baselines/crosshar/crosshar_conse_head.pt`` (gitignored) with a
vocab stamp; it re-fits automatically if the global vocabulary changes.

Backbone checkpoint: ``baselines/crosshar/crosshar_backbone.pt`` (gitignored),
produced by ``train.py``. If it is absent, :meth:`setup` fails loud with the
command to create it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.base import (ConSEAdapter, InputContract, fit_fingerprint, global_labels, make_probe,
                            register)
from baselines.crosshar import prep
from data.scripts.labels.canonical_labels import canonicalize
from eval import scoring

# --- contract / architecture (base_v1) ---
FEATURE_NUM = 6
HIDDEN = 72
HIDDEN_FF = 144
N_LAYERS = 1
N_HEADS = 4
SEQ_LEN = 120
FEAT_DIM = HIDDEN

# --- head-fit hyperparameters (parity with the harnet ConSE head-fit) ---
FIT_EPOCHS = 100
FIT_BATCH = 512
FIT_LR = 1e-3
FIT_SEED = 3431
EMBED_BATCH = 512
HEAD_FIT_MAX_PER_STREAM = 20000   # cap per stream for a tractable head-fit

_HERE = Path(__file__).resolve().parent
_BACKBONE_CKPT = _HERE / "crosshar_backbone.pt"
_HEAD_CACHE = _HERE / "crosshar_conse_head.pt"


def _backbone_fp() -> str:
    """Content hash of the pretrained backbone — stamped into the head cache so a re-pretrained
    backbone invalidates a head that was fit on the old backbone's feature space."""
    import hashlib
    return hashlib.sha256(_BACKBONE_CKPT.read_bytes()).hexdigest() if _BACKBONE_CKPT.exists() else ""


# =============================================================================
# Backbone (reproduces upstream MaskedModel4Pretrain state_dict layout)
# =============================================================================

def _gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _split_last(x, shape):
    shape = list(shape)
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -int(np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def _merge_last(x, n_dims):
    s = x.size()
    return x.view(*s[:-n_dims], -1)


def _fit_fp(vocab) -> str:
    """The cache-validity fingerprint. Called by BOTH the loader and the writer.

    Phase 1.6 originally computed this inside `_fit_head` only, so it was never compared against
    anything -- four writers, zero readers. A cached head therefore survived changes to the subject
    split, the seed, the probe width and the per-stream cap, silently reporting pre-fix numbers.
    """
    from baselines.base import PROBE_SPEC
    from eval.splits import manifest_fingerprint
    return fit_fingerprint(model='crosshar', vocab=list(vocab), split=manifest_fingerprint(),
                           hp=[FIT_EPOCHS, FIT_BATCH, FIT_LR, FIT_SEED], probe=PROBE_SPEC,
                           cap=HEAD_FIT_MAX_PER_STREAM, backbone='crosshar-selfpretrained',
                           prep=prep.PREP_SCHEMA)


class _LayerNorm(nn.Module):
    def __init__(self, hidden, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta = nn.Parameter(torch.zeros(hidden))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        return self.gamma * (x - u) / torch.sqrt(s + self.eps) + self.beta


class _Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(FEATURE_NUM, HIDDEN)
        self.pos_embed = nn.Embedding(SEQ_LEN, HIDDEN)
        self.norm = _LayerNorm(HIDDEN)

    def forward(self, x):
        pos = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), x.size(1))
        e = self.norm(self.lin(x))
        return self.norm(e + self.pos_embed(pos))


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Linear(HIDDEN, HIDDEN)
        self.proj_k = nn.Linear(HIDDEN, HIDDEN)
        self.proj_v = nn.Linear(HIDDEN, HIDDEN)
        self.n_heads = N_HEADS

    def forward(self, x):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (_split_last(t, (self.n_heads, -1)).transpose(1, 2) for t in (q, k, v))
        scores = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.size(-1)), dim=-1)
        h = (scores @ v).transpose(1, 2).contiguous()
        return _merge_last(h, 2)


class _PWFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN, HIDDEN_FF)
        self.fc2 = nn.Linear(HIDDEN_FF, HIDDEN)

    def forward(self, x):
        return self.fc2(_gelu(self.fc1(x)))


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = _Embeddings()
        self.n_layers = N_LAYERS
        self.attn = _Attention()
        self.proj = nn.Linear(HIDDEN, HIDDEN)
        self.norm1 = _LayerNorm(HIDDEN)
        self.pwff = _PWFF()
        self.norm2 = _LayerNorm(HIDDEN)

    def forward(self, x):
        h = self.embed(x)
        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class _Backbone(nn.Module):
    """Encoder-only wrapper whose keys match the upstream MaskedModel4Pretrain
    checkpoint (fc/linear/norm/decoder retained for a strict load; forward
    returns the encoder sequence output like ``output_embed=True``)."""
    def __init__(self):
        super().__init__()
        self.transformer = _Transformer()
        self.fc = nn.Linear(HIDDEN, HIDDEN)
        self.linear = nn.Linear(HIDDEN, HIDDEN)
        self.norm = _LayerNorm(HIDDEN)
        self.decoder = nn.Linear(HIDDEN, FEATURE_NUM)

    def forward(self, x):
        return self.transformer(x)          # (N, 120, 72)


def _instance_norm(x_n_t_c: np.ndarray) -> np.ndarray:
    """Per-channel InstanceNorm per window (CrossHAR's IMUDataset normalization)."""
    inst = nn.InstanceNorm1d(FEATURE_NUM)
    t = torch.from_numpy(x_n_t_c.transpose(0, 2, 1)).float()  # (N, 6, 120)
    return inst(t).numpy().transpose(0, 2, 1)                 # (N, 120, 6)


@torch.no_grad()
def _features(backbone: nn.Module, x6: np.ndarray, device) -> np.ndarray:
    """(N, 120, 6) native g -> (N, 72) mean-pooled frozen-backbone features."""
    xin = _instance_norm(x6)
    feats = []
    for s in range(0, len(xin), EMBED_BATCH):
        b = torch.from_numpy(xin[s:s + EMBED_BATCH]).float().to(device)
        feats.append(backbone(b).mean(dim=1).cpu().numpy())   # mean-pool over time
    return np.concatenate(feats, axis=0).astype(np.float32)


# =============================================================================
# adapter
# =============================================================================

@register
class CrossHARAdapter(ConSEAdapter):
    """CrossHAR (masked-Transformer + contrastive), ConSE tier."""

    name = "crosshar"
    contract = InputContract(channels=list(prep.SIX_CHANNELS), rate_hz=prep.TARGET_HZ,
                             window_sec=prep.TARGET_LEN / prep.TARGET_HZ)

    # ---- setup: load self-pretrained backbone + fit (or load cached) head ----
    def setup(self, device):
        vocab = global_labels()
        backbone = self._load_backbone(device)

        # Phase 1.4: identical 2-layer probe for every ConSE-tier model (see base.make_probe)
        # H4: make_probe draws from the GLOBAL torch RNG. Unseeded, two runs get different
        # heads (measured max init-weight delta 0.971) -- and Phase 1.4 grew the probe from
        # ~24k params to ~180k, so init variance now moves the reported number. harnet has
        # seeded since the H4 audit; these three never did.
        torch.manual_seed(FIT_SEED)
        head = make_probe(FEAT_DIM, len(vocab)).to(device)
        cached = self._load_cached_head(vocab, device)
        if cached is not None:
            head_sd, temperature = cached
            head.load_state_dict(head_sd)
        else:
            temperature = self._fit_head(backbone, head, vocab, device)
        head.eval()
        return {"backbone": backbone, "head": head, "temperature": temperature}

    def evaluation_artifacts(self, state):
        return {"backbone": _BACKBONE_CKPT,
                "backbone_metadata": _BACKBONE_CKPT.with_suffix(".meta.json"),
                "conse_head": _HEAD_CACHE}

    def evaluation_config(self, state):
        return {
            "input_rate_hz": prep.TARGET_HZ,
            "input_samples": prep.TARGET_LEN,
            "train_datasets": list(prep.TRAIN_DATASETS),
            "head_fit_max_per_stream": HEAD_FIT_MAX_PER_STREAM,
        }

    def _load_backbone(self, device) -> nn.Module:
        if not _BACKBONE_CKPT.exists():
            raise FileNotFoundError(
                f"CrossHAR backbone checkpoint missing at {_BACKBONE_CKPT}. "
                "Self-pretrain it on our corpus first:\n"
                "  python -m baselines.crosshar.train")
        meta_path = _BACKBONE_CKPT.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if (meta.get("schema_version") != 2
                or meta.get("prep_schema") != prep.PREP_SCHEMA
                or meta.get("corpus_profile") != "expanded_phase_a"
                or meta.get("recipe") != "full"
                or meta.get("batch_size") != 512
                or meta.get("max_per_stream") != 20_000
                or not meta.get("channel_augmentation")
                or tuple(meta.get("train_datasets", ())) != tuple(prep.TRAIN_DATASETS)):
            raise RuntimeError(
                f"CrossHAR backbone at {_BACKBONE_CKPT} is unstamped or does not match the "
                "locked expanded-corpus recipe. Re-pretrain before scoring:\n"
                "  python -m baselines.crosshar.train --recipe full --gpu")
        backbone = _Backbone().to(device)
        sd = torch.load(str(_BACKBONE_CKPT), map_location=device, weights_only=True)
        backbone.load_state_dict(sd)
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        return backbone

    def _load_cached_head(self, vocab, device):
        if not _HEAD_CACHE.exists():
            return None
        blob = torch.load(str(_HEAD_CACHE), map_location=device, weights_only=True)
        if list(blob.get("labels", [])) != list(vocab):
            return None
        if blob.get("fit_fp") != _fit_fp(vocab):
            return None                       # split/probe/seed/cap/backbone changed -> refit
        # Re-fit if the backbone changed (e.g. after the deferred full pretrain overwrites it):
        # a head fit on the old backbone's feature space is invalid for new features.
        if blob.get("backbone_fp") != _backbone_fp():
            return None
        if "temperature" not in blob:
            return None   # pre-calibration cache -> re-fit (#82)
        return blob["head"], float(blob["temperature"])

    def _fit_head(self, backbone, head, vocab, device):
        """Fit the softmax head on frozen backbone features over the training
        corpus, selecting the best epoch on a SUBJECT-DISJOINT held-out fold."""
        label_to_idx = {l: i for i, l in enumerate(vocab)}
        fit_device = device if isinstance(device, torch.device) else torch.device(device)
        backbone.to(fit_device)

        feats, labs, subjs, used = [], [], [], []
        for ds, stream, x6, raw_labels, subjects in prep.iter_train_streams(
                max_per_stream=HEAD_FIT_MAX_PER_STREAM, seed=FIT_SEED):
            gl = np.array([label_to_idx.get(canonicalize(l), -1) for l in raw_labels])
            keep = gl >= 0
            if not keep.any():
                continue
            feats.append(_features(backbone, x6[keep], fit_device))
            labs.append(gl[keep])
            subjs.append(np.array([f"{ds}:{s}" for s in subjects[keep]]))
            used.append(f"{ds}/{stream}")
        print(f"[crosshar] head-fit corpus: {used}")

        X = np.concatenate(feats, 0)
        Y = np.concatenate(labs, 0)
        S = np.concatenate(subjs, 0)

        # Phase 1.1 (H7/H8): SHARED, per-dataset-stratified subject manifest. Previously each
        # model reshuffled its own aggregate subject pool, so excluding a stream moved 16.5% of
        # shared subjects into different folds than other models, and 3 datasets got ZERO val
        # subjects. Folds are now identical across models regardless of stream coverage.
        from eval.splits import split_indices   # lazy
        ti, vi, tei = split_indices(S)
        Xt = torch.from_numpy(X[ti]).float()
        Yt = torch.from_numpy(Y[ti]).long()
        Xv = torch.from_numpy(X[vi]).float().to(fit_device)
        Yv = Y[vi]
        Xte = torch.from_numpy(X[tei]).float().to(fit_device) if len(tei) else Xv
        Yte = Y[tei] if len(tei) else Yv

        head.to(fit_device)
        opt = torch.optim.Adam(head.parameters(), lr=FIT_LR)
        crit = nn.CrossEntropyLoss()
        rng = np.random.RandomState(FIT_SEED)
        n = len(Xt)
        best_acc, best_sd = -1.0, None
        for _ in range(FIT_EPOCHS):
            head.train()
            perm = rng.permutation(n)
            for s in range(0, n, FIT_BATCH):
                bi = perm[s:s + FIT_BATCH]
                opt.zero_grad()
                loss = crit(head(Xt[bi].to(fit_device)), Yt[bi].to(fit_device))
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                # Phase 1.2 (H7b): select on BALANCED accuracy, not window accuracy — we
                # report macro-F1 and the corpus has ~530x class imbalance.
                _p = head(Xv).argmax(1).cpu().numpy()
                va = float(np.mean([(_p[Yv == c] == c).mean() for c in np.unique(Yv)]))
            if va > best_acc:
                best_acc = va
                best_sd = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if best_sd is not None:
            head.load_state_dict(best_sd)

        # Temperature-scale on the SAME subject-disjoint val fold so ConSE weights top-K by
        # calibrated confidences, not raw over-confident softmax (#82).
        head.eval()
        with torch.no_grad():
            # Phase 1.3 (#12): calibrate on the THIRD fold, which was previously computed and
            # discarded. Reusing the selection fold made the temperature mildly optimistic.
            _cal_X, _cal_Y = (Xte, Yte) if len(tei) else (Xv, Yv)
            temperature = scoring.fit_temperature(head(_cal_X).cpu().numpy(), _cal_Y)

        n_val_subj = len(set(S[vi]))
        print(f"[crosshar] fitted head: val_acc={best_acc:.3f}, T={temperature:.3f} "
              f"over {n_val_subj} held-out subjects")
        torch.save({"head": {k: v.cpu() for k, v in head.state_dict().items()},
                    "labels": list(vocab), "fit_fp": _fit_fp(vocab), "backbone_fp": _backbone_fp(),
                    "temperature": float(temperature)}, str(_HEAD_CACHE))
        backbone.to(device)
        head.to(device)
        return float(temperature)

    # ---- window_probs: (N, 59) softmax over the global vocab ------------------
    def window_probs(self, stream, state, device) -> np.ndarray:
        backbone, head = state["backbone"], state["head"]
        T = float(state.get("temperature", 1.0))    # calibrated temperature (#82)
        x6 = prep.grid_to_contract(stream.windows, stream.channels, stream.rate_hz, stream.lengths)
        feats = _features(backbone, x6, device)
        probs = []
        with torch.no_grad():
            for s in range(0, len(feats), EMBED_BATCH):
                b = torch.from_numpy(feats[s:s + EMBED_BATCH]).float().to(device)
                probs.append(F.softmax(head(b) / T, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def window_features(self, stream, state, device) -> np.ndarray:
        x6 = prep.grid_to_contract(stream.windows, stream.channels, stream.rate_hz, stream.lengths)
        return _features(state["backbone"], x6, device)

    def predict_candidates_from_features(self, features, candidates, state, device):
        head = state["head"]
        temperature = float(state.get("temperature", 1.0))
        values = []
        with torch.no_grad():
            for start in range(0, len(features), EMBED_BATCH):
                batch = torch.from_numpy(features[start:start + EMBED_BATCH]).float().to(device)
                values.append(F.softmax(head(batch) / temperature, dim=1).cpu().numpy())
        return scoring.conse_predict(np.concatenate(values), global_labels(), list(candidates))
