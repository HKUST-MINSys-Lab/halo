"""Self-supervised LiMU-BERT pretraining ON OUR TRAINING CORPUS.

LiMU-BERT ships no released weights; it is an SSL *method* (masked
reconstruction). This script pools the design-of-record training datasets' grids into
LiMU-BERT's input contract (6-ch acc+gyro, 20 Hz, 120 samples — see
:mod:`baselines.limubert.prep`) and drives the UPSTREAM LiMU-BERT SSL pipeline
(model + Trainer + masking, reused from
``auxiliary_repos/LIMU-BERT-Public`` via ``sys.path``) to produce the backbone
checkpoint ``baselines/limubert/limubert_backbone.pt`` that the adapter loads.

  * Vendoring the upstream repo under ``baselines/limubert/repo/`` is a follow-up;
    for now it is imported from the legacy ``auxiliary_repos`` tree.
  * Run length is a CLI arg so a SMOKE run (few epochs, small subset) proves the
    pipeline end-to-end while the FULL run reproduces the paper-scale pretrain.

FULL same-data pretrain (published batch/objective; exposure matched after the
18-source expansion):

    python -m baselines.limubert.train --recipe full --gpu

SMOKE (what we run to prove the wiring; under-trained by design):

    python -m baselines.limubert.train --recipe smoke --gpu
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.limubert import prep

# Upstream LiMU-BERT repo (reused via sys.path; vendoring under repo/ is a follow-up).
LIMU_REPO = Path("/home/alex/code/HALO/legacy_code/auxiliary_repos/LIMU-BERT-Public")

_HERE = Path(__file__).resolve().parent
BACKBONE_CKPT = _HERE / "limubert_backbone.pt"
METADATA_SCHEMA = 2


def _seed_worker(worker_id):
    """Reseed numpy AND stdlib-random per DataLoader worker.

    LiMU-BERT's span-mask draws from both ``np.random`` (utils.span_mask) and
    stdlib ``random`` (utils.bert_mask -> random.sample). Forked workers inherit
    identical global RNG state, so without this they emit correlated/duplicate
    masks across workers -- shrinking mask diversity and changing the SSL
    objective. torch assigns each worker a unique ``initial_seed()``; derive both
    RNGs from it so masking stays diverse AND the run stays reproducible.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Self-pretrain LiMU-BERT on our corpus")
    ap.add_argument("--recipe", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=3431)
    ap.add_argument("--max-per-stream", type=int,
                    help="per-stream window cap; 0 means uncapped")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--num-workers", type=int,
                    help="DataLoader workers. 0 = single-process masking (robust to CPU contention: "
                         "no worker-starvation stall, degrades gracefully instead of collapsing).")
    ap.add_argument("--output", type=Path,
                    help="checkpoint destination; defaults to the canonical adapter checkpoint")
    args = ap.parse_args(argv)

    defaults = ({"epochs": 143, "batch_size": 128,
                 "max_per_stream": 20_000, "num_workers": 4}
                if args.recipe == "full" else
                {"epochs": 2, "batch_size": 128,
                 "max_per_stream": 800, "num_workers": 4})
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    if str(LIMU_REPO) not in sys.path:
        sys.path.insert(0, str(LIMU_REPO))
    import models as lb_models
    import train as lb_train
    from config import PretrainModelConfig, TrainConfig, MaskConfig
    from utils import LIBERTDataset4Pretrain, Preprocess4Mask

    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print(f"[limubert] device={device}")
    if args.gpu and device.type != "cuda":
        raise SystemExit("--gpu requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    started = time.perf_counter()

    cap = args.max_per_stream or None
    data = prep.build_pretrain_array(max_per_stream=cap, seed=args.seed, shuffle=False)
    print(f"[limubert] corpus: {data.shape[0]} windows, shape {data.shape}")

    # 2026-08-22 audit F1: upstream's Preprocess4Normalization divides accel by 9.8 because its
    # datasets store m/s²; the model is designed to see acc ≈ 1 in g units. Our grids ALREADY
    # store g, so the division here fed the backbone acc ≈ 0.10 against gyro O(0.1-0.5) — ~10x
    # off the designed acc:gyro balance. Grids-in-g are the upstream post-normalization
    # convention, so no scaling is applied (must match adapter._normalize).

    model_cfg = PretrainModelConfig(hidden=72, hidden_ff=144, feature_num=6,
                                    n_layers=4, n_heads=4, seq_len=120, emb_norm=True)
    train_cfg = TrainConfig(seed=args.seed, batch_size=args.batch_size, lr=args.lr,
                            n_epochs=args.epochs, warmup=0.1, save_steps=100000,
                            total_steps=200000000)
    mask_cfg = MaskConfig(mask_ratio=0.15, mask_alpha=6, max_gram=10,
                          mask_prob=0.8, replace_prob=0.0)

    # Normalization applied once above (value-identical); per-item work is now masking only.
    pipeline = [Preprocess4Mask(mask_cfg)]
    # Reproduce upstream's seeded 80/10/10 random partition without allocating
    # labels that the self-supervised objective never consumes.
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    order = np.arange(len(data))
    np.random.shuffle(order)
    n_train, n_val = int(0.8 * len(data)), int(0.1 * len(data))
    d_train = data[order[:n_train]]
    d_test = data[order[n_train:n_train + n_val]]
    del data, order
    print(f"[limubert] train={len(d_train)} val={len(d_test)}")

    ds_train = LIBERTDataset4Pretrain(d_train, pipeline=pipeline)
    ds_test = LIBERTDataset4Pretrain(d_test, pipeline=pipeline)
    # num_workers>0 overlaps the CPU span-masking (the measured bottleneck) with GPU
    # compute; _seed_worker keeps each worker's mask RNG independent (see above).
    # pin_memory + persistent_workers + prefetch trim per-epoch overhead. Pure-speed,
    # faithful (identical objective, just parallelized). Measured ~1.6x on the box.
    nw = args.num_workers
    dl_extra = dict(num_workers=nw, persistent_workers=True, prefetch_factor=4,
                    worker_init_fn=_seed_worker) if nw > 0 else dict(num_workers=0)
    ld_train = DataLoader(ds_train, shuffle=True, batch_size=train_cfg.batch_size,
                          pin_memory=True, **dl_extra)
    ld_test = DataLoader(ds_test, shuffle=False, batch_size=train_cfg.batch_size,
                         pin_memory=True, **dl_extra)

    model = lb_models.LIMUBertModel4Pretrain(model_cfg)
    criterion = nn.MSELoss(reduction="none")
    adam_extra = {"fused": True} if device.type == "cuda" else {}
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, **adam_extra)

    output = (args.output or BACKBONE_CKPT).resolve()
    save_prefix = (_HERE / "pretrain_run" / "model" if args.output is None else
                   output.parent / f"{output.stem}_run" / "model")
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    trainer = lb_train.Trainer(train_cfg, model, optimizer, str(save_prefix), device)

    def func_loss(m, batch):
        mask_seqs, masked_pos, seqs = batch
        return criterion(m(mask_seqs, masked_pos), seqs)

    def func_forward(m, batch):
        mask_seqs, masked_pos, seqs = batch
        return m(mask_seqs, masked_pos), seqs

    def func_evaluate(seqs, pred):
        return criterion(pred, seqs).mean().cpu().numpy()

    trainer.pretrain(func_loss, func_forward, func_evaluate, ld_train, ld_test)

    # Trainer.pretrain reloads the best state into `model`; save it as the
    # canonical checkpoint the adapter loads.
    torch.save(model.state_dict(), str(output))
    metadata = {
        "schema_version": METADATA_SCHEMA,
        "model": "limubert",
        "corpus_profile": "expanded_phase_a",
        "acc_convention": "g",
        "train_datasets": list(prep.TRAIN_DATASETS),
        "prep_schema": prep.PREP_SCHEMA,
        "input_contract": {"rate_hz": prep.TARGET_HZ, "samples": prep.TARGET_LEN,
                           "channels": list(prep.SIX_CHANNELS), "acc_convention": "g"},
        "recipe": args.recipe,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_per_stream": args.max_per_stream,
        "seed": args.seed,
        "train_windows": len(ds_train),
        "validation_windows": len(ds_test),
        "optimizer_steps": len(ld_train) * args.epochs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[limubert] saved backbone -> {output}")


if __name__ == "__main__":
    main()
