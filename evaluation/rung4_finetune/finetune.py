"""The rung-4 treatments, every one fitted on the same k windows per class and scored on the same
scored set. Budgets are fixed a priori (steps, lr); nothing selects on the scored set.

Cached-feature treatments (all six providers, NormWear included):
* ``linear_probe``      — one linear layer on frozen features (TransfHAR's on-device mechanism),
                          fitted by deterministic L-BFGS logistic regression.
* ``small_classifier``  — the frozen-projection MLP from the corpus-matched work
                          (``FrozenBaselineProjection``) plus a linear head, fitted by SGD.
* ``enrollment_frozen`` — the rung-3 parameter-free class-prototype readout on the same supports:
                          the enrollment side of the crossover, on identical windows.

Raw-window treatments (HALO from its checkpoint; HARNet-5 / LiMU-BERT-X / UniMTS as released
trunks under the encoder contract via ``MatchedCorpusEncoder(pretrained=True)``; NormWear has no
fine-tuning path — its authors never fine-tune and it has no trunk under the contract — so it is
declared unsupported rather than approximated):
* ``lora``               — LoRA on every ``nn.Linear`` of the encoder, base frozen.
* ``full_finetune``      — every encoder parameter trainable.
* ``scratch_specialist`` — the same architecture from random initialisation: the "just retrain"
                           comparator, on exactly the same k windows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import baselines
from evaluation.metrics import classification
from evaluation.rung2_unlabeled.ncurve import CellSplit, inductive_predictions, shared_support_set
from evaluation.rung4_finetune.lora import apply_lora, lora_parameters, trainable_parameter_count
from evaluation.zero_shot import _normalise

TREATMENTS = ("enrollment_frozen", "linear_probe", "small_classifier", "lora", "full_finetune",
              "scratch_specialist")
CACHED_FEATURE_TREATMENTS = frozenset({"enrollment_frozen", "linear_probe", "small_classifier"})
RAW_WINDOW_TREATMENTS = frozenset({"lora", "full_finetune", "scratch_specialist"})
# Released trunks that exist under the encoder contract (model/tokenizer/matched_encoder.py).
MATCHED_BACKBONE = {"harnet5": "harnet", "limubert_x": "limubert", "unimts": "unimts"}


@dataclass(frozen=True)
class FineTuneConfig:
    steps: int = 300
    lr: float = 1e-3
    encoder_lr_scale: float = 0.1      # LoRA / full fine-tune: encoder lr = lr * scale
    batch_size: int = 32
    weight_decay: float = 0.05
    lora_rank: int = 8
    lora_alpha: float = 16.0
    projection_dim: int = 128
    probe_c: float = 1.0               # inverse L2 strength of the logistic-regression probe
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------------------
# cached-feature treatments
# ---------------------------------------------------------------------------------------------

def enrollment_frozen_predictions(features: np.ndarray, support_rows: np.ndarray, support_labels: np.ndarray,
                                  rows: np.ndarray, n_classes: int) -> np.ndarray:
    return inductive_predictions(scores=np.zeros((len(features), n_classes), np.float32), features=features,
                                 support_rows=support_rows, support_labels=support_labels,
                                 n_classes=n_classes, rows=rows)


def linear_probe_predictions(features: np.ndarray, support_rows: np.ndarray, support_labels: np.ndarray,
                             rows: np.ndarray, n_classes: int, cfg: FineTuneConfig) -> tuple[np.ndarray, dict]:
    from sklearn.linear_model import LogisticRegression

    z = _normalise(features)
    present = np.unique(support_labels)
    if len(present) < 2:
        raise ValueError("a probe needs supports from at least two classes")
    model = LogisticRegression(C=cfg.probe_c, max_iter=2000, random_state=cfg.seed)
    model.fit(z[support_rows], support_labels)
    preds = model.predict(z[rows]).astype(np.int64)
    return preds, {"trainable_params": int(model.coef_.size + model.intercept_.size),
                   "probe_iterations": int(np.max(model.n_iter_))}


class _ProjectionClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, d_model: int):
        super().__init__()
        from training.support_classifier.frozen_baseline_adaptation import FrozenBaselineProjection

        self.projection = FrozenBaselineProjection(in_dim, d_model=d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.projection(x))


def small_classifier_predictions(features: np.ndarray, support_rows: np.ndarray, support_labels: np.ndarray,
                                 rows: np.ndarray, n_classes: int, cfg: FineTuneConfig,
                                 device: torch.device) -> tuple[np.ndarray, dict]:
    table = torch.as_tensor(_normalise(features), device=device)
    torch.manual_seed(cfg.seed)
    model = _ProjectionClassifier(table.shape[1], n_classes, cfg.projection_dim).to(device)
    history = fit_head(lambda idx: model.projection(table[torch.as_tensor(idx, device=device)]),
                       list(model.projection.parameters()), model.head, support_rows, support_labels, cfg, device)
    with torch.no_grad():
        preds = model(table[torch.as_tensor(rows, device=device)]).argmax(-1).cpu().numpy()
    return preds.astype(np.int64), {**history, "trainable_params": trainable_parameter_count(model)}


# ---------------------------------------------------------------------------------------------
# raw-window treatments
# ---------------------------------------------------------------------------------------------

def build_encoder_for(name: str, treatment: str, *, halo_checkpoint, device: torch.device) -> nn.Module:
    """A trainable encoder for ``name`` under ``treatment``; raises for providers without a path."""
    from training.tokenizer.eval_transfer import build_encoder

    if name == "halo":
        if halo_checkpoint is None:
            raise ValueError("halo fine-tuning needs --halo-checkpoint")
        blob = torch.load(halo_checkpoint, map_location="cpu", weights_only=False)
        encoder = build_encoder(blob, device, training=True)
        if treatment == "scratch_specialist":
            from model.tokenizer.matched_encoder import _reinitialise

            encoder.apply(_reinitialise)
        return encoder
    if name in MATCHED_BACKBONE:
        from model.tokenizer.matched_encoder import build_matched_encoder

        return build_matched_encoder(MATCHED_BACKBONE[name], device=device,
                                     pretrained=(treatment != "scratch_specialist")).train()
    raise baselines.UnsupportedEvaluationCell(
        f"{name} has no fine-tuning path: no trunk under the encoder contract"
        + (" and its authors never fine-tune" if name == "normwear" else ""))


def set_treatment(encoder: nn.Module, treatment: str, cfg: FineTuneConfig) -> tuple[list[nn.Parameter], dict]:
    if treatment == "lora":
        wrapped = apply_lora(encoder, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
        return lora_parameters(encoder), {"lora_layers": len(wrapped)}
    if treatment in {"full_finetune", "scratch_specialist"}:
        for parameter in encoder.parameters():
            parameter.requires_grad_(True)
        return [p for p in encoder.parameters()], {}
    raise ValueError(f"{treatment} is not a raw-window treatment")


def encode_rows(encoder: nn.Module, stream, rows: np.ndarray, device: torch.device, *,
                requires_grad: bool) -> torch.Tensor:
    """Pooled recording vectors for ``rows`` of an EvalStream, through the same path the sealed
    evaluator uses, with gradients when asked."""
    from training.tokenizer.eval_transfer import encode_dataset_detailed
    from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

    rows = np.asarray(rows, dtype=np.int64)
    lengths = None if getattr(stream, "lengths", None) is None else np.asarray(stream.lengths)[rows]
    out = encode_dataset_detailed(
        encoder, stream.windows[rows], stream_channel_descriptions(stream.dataset, stream.stream), device,
        stream.rate_hz, _stream_gravity_state(stream.dataset, stream.stream), channel_mask=stream.mask,
        dataset=stream.dataset, stream=stream.stream,
        source_rate=(stream.effective_source_rate_hz if stream.effective_source_rate_hz is not None else None),
        lengths=lengths, requires_grad=requires_grad, batch_size=max(1, len(rows)), _require_patches=False,
    )
    pooled = out["pooled"]
    return pooled if requires_grad else pooled.detach()


def fit_head(encode_fn: Callable[[np.ndarray], torch.Tensor], trainable: Sequence[nn.Parameter],
             head: nn.Module, support_rows: np.ndarray, support_labels: np.ndarray, cfg: FineTuneConfig,
             device: torch.device, *, module: nn.Module | None = None) -> dict:
    """Fixed-budget cross-entropy fit of ``head`` (and ``trainable`` encoder parameters) on the
    supports. Frozen encoders are encoded once; trainable ones are re-encoded every step."""
    support_rows = np.asarray(support_rows, dtype=np.int64)
    labels = torch.as_tensor(np.asarray(support_labels, dtype=np.int64), device=device)
    groups = [{"params": list(head.parameters()), "lr": cfg.lr}]
    trainable = [p for p in trainable if p.requires_grad]
    if trainable:
        groups.append({"params": trainable, "lr": cfg.lr * cfg.encoder_lr_scale})
    optimiser = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(cfg.steps, 1))
    rng = np.random.default_rng(cfg.seed)
    cached = None if trainable else encode_fn(support_rows).detach()
    losses = []
    for _ in range(cfg.steps):
        idx = rng.choice(len(support_rows), size=min(cfg.batch_size, len(support_rows)),
                         replace=len(support_rows) < cfg.batch_size)
        features = cached[torch.as_tensor(idx, device=device)] if cached is not None else encode_fn(support_rows[idx])
        loss = F.cross_entropy(head(features.float()), labels[torch.as_tensor(idx, device=device)])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        optimiser.step()
        schedule.step()
        losses.append(float(loss.detach()))
    return {"steps": int(cfg.steps), "first_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None}


@torch.no_grad()
def predict_rows(encode_fn: Callable[[np.ndarray], torch.Tensor], head: nn.Module, rows: np.ndarray,
                 *, batch_size: int = 512) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    out = []
    for start in range(0, len(rows), batch_size):
        out.append(head(encode_fn(rows[start:start + batch_size]).float()).argmax(-1).cpu().numpy())
    return np.concatenate(out).astype(np.int64) if out else np.zeros(0, np.int64)


def raw_window_predictions(name: str, treatment: str, stream, support_rows: np.ndarray,
                           support_labels: np.ndarray, rows: np.ndarray, n_classes: int,
                           cfg: FineTuneConfig, device: torch.device, *, halo_checkpoint) -> tuple[np.ndarray, dict]:
    torch.manual_seed(cfg.seed)
    encoder = build_encoder_for(name, treatment, halo_checkpoint=halo_checkpoint, device=device)
    trainable, info = set_treatment(encoder, treatment, cfg)
    encoder.train() if trainable else encoder.eval()
    with torch.no_grad():
        width = int(encode_rows(encoder, stream, support_rows[:1], device, requires_grad=False).shape[-1])
    head = nn.Linear(width, n_classes).to(device)
    history = fit_head(lambda r: encode_rows(encoder, stream, r, device, requires_grad=bool(trainable)),
                       trainable, head, support_rows, support_labels, cfg, device)
    encoder.eval()
    preds = predict_rows(lambda r: encode_rows(encoder, stream, r, device, requires_grad=False), head, rows)
    info.update(history)
    info["trainable_params"] = trainable_parameter_count(encoder) + trainable_parameter_count(head)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds, info


# ---------------------------------------------------------------------------------------------
# one (provider, cell)
# ---------------------------------------------------------------------------------------------

def run_cell(*, name: str, stream, features: np.ndarray | None, truth_ids: np.ndarray, classes: Sequence[str],
             split: CellSplit, ks: Sequence[int], treatments: Sequence[str], cfg: FineTuneConfig,
             device: torch.device, halo_checkpoint=None, seed_parts: Sequence[object] = ()) -> list[dict]:
    """Every rung-4 row for one (provider, cell): treatments × k on the scored set, supports drawn
    from the pool partition exactly as rung 2 draws them (same seed parts ⇒ same windows)."""
    classes = list(classes)
    C = len(classes)
    names = np.asarray(classes, dtype=object)
    rows: list[dict] = []
    for k in ks:
        if k < 1:
            continue
        support = shared_support_set(split.pool, truth_ids, k, C, seed_parts=(*seed_parts, "k", k))
        if support is None:
            rows.append({"method": None, "k": k, "status": "n/a",
                         "reason": f"pool lacks {k} execution-disjoint windows for every class"})
            continue
        support_rows, support_labels = support
        for treatment in treatments:
            base = {"method": treatment, "k": k, "n_support": int(len(support_rows)),
                    "n_scored": int(len(split.scored)), "config": cfg.as_dict()}
            try:
                if treatment == "enrollment_frozen":
                    if features is None:
                        raise ValueError("cached features required")
                    preds, info = enrollment_frozen_predictions(features, support_rows, support_labels,
                                                                split.scored, C), {"trainable_params": 0}
                elif treatment == "linear_probe":
                    preds, info = linear_probe_predictions(features, support_rows, support_labels, split.scored, C, cfg)
                elif treatment == "small_classifier":
                    preds, info = small_classifier_predictions(features, support_rows, support_labels,
                                                               split.scored, C, cfg, device)
                elif treatment in RAW_WINDOW_TREATMENTS:
                    if stream is None:
                        raise ValueError("raw-window treatments need the stream")
                    preds, info = raw_window_predictions(name, treatment, stream, support_rows, support_labels,
                                                         split.scored, C, cfg, device, halo_checkpoint=halo_checkpoint)
                else:
                    raise ValueError(f"unknown treatment {treatment!r}")
            except baselines.UnsupportedEvaluationCell as exc:
                rows.append({**base, "status": "n/a", "reason": str(exc)})
                continue
            rows.append({**base, "status": "ok",
                         **classification(names[truth_ids[split.scored]], names[preds]), **info})
    return rows
