"""Self-supervised CrossHAR pretraining ON OUR TRAINING CORPUS.

CrossHAR ships no released weights; it is an SSL *method* (masked reconstruction +
contrastive NT-Xent). This script pools the design-of-record training datasets' grids
into CrossHAR's input contract (6-ch acc+gyro, 20 Hz, 120 samples — see
:mod:`baselines.crosshar.prep`) and drives the UPSTREAM CrossHAR SSL pipeline
(model + Trainer + Contrastive + NT-Xent loss + masking/augmentation, reused from
``auxiliary_repos/CrossHAR`` via ``sys.path``) to produce the backbone checkpoint
``baselines/crosshar/crosshar_backbone.pt`` that the adapter loads.

  * Vendoring the upstream repo under ``baselines/crosshar/repo/`` is a follow-up;
    for now it is imported from the legacy ``auxiliary_repos`` tree.
  * Run length is a CLI arg so a SMOKE run (few epochs, small subset, no channel
    augmentation) proves the pipeline end-to-end, while the FULL run reproduces
    the paper-scale pretrain.

FULL same-data pretrain (published batch/objective; exposure matched after the
18-source expansion):

    python -m baselines.crosshar.train --recipe full --gpu

SMOKE (what we run to prove the wiring; under-trained by design):

    python -m baselines.crosshar.train --recipe smoke --gpu
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from baselines.crosshar import prep

# Upstream CrossHAR repo (reused via sys.path; vendoring under repo/ is a follow-up).
CROSSHAR_REPO = Path("/home/alex/code/HALO/legacy_code/auxiliary_repos/CrossHAR")

_HERE = Path(__file__).resolve().parent
BACKBONE_CKPT = _HERE / "crosshar_backbone.pt"
METADATA_SCHEMA = 2


def _seed_worker(worker_id):
    """Reseed numpy AND stdlib-random per DataLoader worker.

    CrossHAR's span-mask draws from both ``np.random`` (utils.span_mask) and stdlib
    ``random`` (utils.bert_mask -> random.sample), and masks each item TWICE (two
    contrastive views). Forked workers inherit identical global RNG state, so without
    this they emit correlated/duplicate masks -- changing the SSL objective. torch
    gives each worker a unique ``initial_seed()``; derive both RNGs from it so masking
    stays diverse AND the run stays reproducible.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


class _ChunkedPretrainDataset(Dataset):
    """CrossHAR's upstream two-view dataset with bounded host-memory use.

    The upstream class transforms the complete array at once. At the expanded
    recipe's sixfold corpus size, float64 augmentation temporaries can approach
    host RAM capacity. Chunking changes only the order in which random draws are
    consumed; the transform distributions, InstanceNorm, masks, and returned
    tensor contract are unchanged.
    """

    def __init__(self, data, pipeline, data_transform, *, chunk_size=8192):
        self.pipeline = pipeline
        self.aug1 = np.empty_like(data, dtype=np.float32)
        self.aug2 = np.empty_like(data, dtype=np.float32)
        normalizer = nn.InstanceNorm1d(data.shape[2])
        with torch.no_grad():
            for start in range(0, len(data), chunk_size):
                stop = min(start + chunk_size, len(data))
                weak, strong = data_transform(data[start:stop])
                for source, target in ((weak, self.aug1), (strong, self.aug2)):
                    values = torch.from_numpy(source).transpose(1, 2)
                    normalized = normalizer(values).transpose(1, 2).numpy()
                    target[start:stop] = normalized

    def __getitem__(self, index):
        instance_1 = self.aug1[index]
        instance_2 = self.aug2[index]
        for proc in self.pipeline:
            instance_1 = proc(instance_1)
            instance_2 = proc(instance_2)
        mask_seq_1, masked_pos_1, seq_1 = instance_1
        mask_seq_2, masked_pos_2, seq_2 = instance_2
        return (torch.from_numpy(mask_seq_1), torch.from_numpy(masked_pos_1).long(),
                torch.from_numpy(seq_1), torch.from_numpy(mask_seq_2),
                torch.from_numpy(masked_pos_2).long(), torch.from_numpy(seq_2))

    def __len__(self):
        return len(self.aug1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Self-pretrain CrossHAR on our corpus")
    ap.add_argument("--recipe", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--epochs-cl", type=int,
                    help="# of final epochs that add the contrastive loss")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=3431)
    ap.add_argument("--max-per-stream", type=int,
                    help="per-stream window cap; 0 means uncapped")
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None,
                    help="apply the published six-way paired-axis augmentation")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--num-workers", type=int,
                    help="DataLoader workers. 0 = single-process masking (robust to CPU contention: "
                         "no worker-starvation stall on a shared box, degrades gracefully instead).")
    ap.add_argument("--output", type=Path,
                    help="checkpoint destination; defaults to the canonical adapter checkpoint")
    args = ap.parse_args(argv)

    defaults = ({"epochs": 52, "epochs_cl": 18, "batch_size": 512,
                 "max_per_stream": 20_000, "augment": True, "num_workers": 8}
                if args.recipe == "full" else
                {"epochs": 2, "epochs_cl": 1, "batch_size": 256,
                 "max_per_stream": 800, "augment": False, "num_workers": 4})
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if not 0 <= args.epochs_cl <= args.epochs:
        ap.error("--epochs-cl must be between 0 and --epochs")

    if str(CROSSHAR_REPO) not in sys.path:
        sys.path.insert(0, str(CROSSHAR_REPO))
    import models as ch_models
    import train as ch_train
    from Contrastive import Contrastive
    from config import PretrainModelConfig, TrainConfig, MaskConfig
    from augmentations import DataTransform
    from utils import Preprocess4Mask

    # FIX (audit P0): the upstream CrossHAR Trainer.run() switches models to .eval() for
    # validation each epoch and NEVER restores .train(), so every epoch after the first
    # trains with dropout/BatchNorm frozen -- degrading the Contrastive head's BatchNorm1d
    # (its running stats freeze at epoch-0 values) through the entire contrastive phase.
    # Restore train mode after each validation. Patched here rather than in the un-versioned
    # vendored repo so the correctness fix is tracked in git.
    _ch_orig_run = ch_train.Trainer.run
    def _run_then_restore_train(self, *a, **k):
        out = _ch_orig_run(self, *a, **k)
        self.masked_model.train()
        self.Contrastive_model.train()
        return out
    ch_train.Trainer.run = _run_then_restore_train

    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print(f"[crosshar] device={device}")
    if args.gpu and device.type != "cuda":
        raise SystemExit("--gpu requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    started = time.perf_counter()

    cap = args.max_per_stream or None
    data = prep.build_pretrain_array(max_per_stream=cap, seed=args.seed, shuffle=False)
    print(f"[crosshar] corpus: {data.shape[0]} windows, shape {data.shape}")

    if args.augment:
        permutations = np.asarray(list(itertools.permutations(range(3))), dtype=np.int64)
        orders = np.concatenate((permutations, permutations + 3), axis=1)
        data = np.concatenate([data[:, :, order] for order in orders], axis=0).astype(
            np.float32, copy=False)
        print(f"[crosshar] after channel_aug: {data.shape[0]} windows")

    data = np.ascontiguousarray(data, dtype=np.float32)

    model_cfg = PretrainModelConfig(feature_num=6, hidden=72, hidden_ff=144,
                                    n_layers=1, n_heads=4, seq_len=120, emb_norm=True)
    train_cfg = TrainConfig(seed=args.seed, batch_size=args.batch_size, lr=args.lr,
                            n_epochs=args.epochs, n_epochs_cl=args.epochs_cl,
                            warmup=0.1, save_steps=100000, total_steps=200000000)
    mask_cfg = MaskConfig(mask_ratio=0.15, mask_alpha=6, max_gram=10,
                          mask_prob=0.8, replace_prob=0.0)

    pipeline = [Preprocess4Mask(mask_cfg)]
    # This is upstream's seeded 80/10/10 random split without a multi-gigabyte
    # dummy SSL-label allocation. Only train and validation are consumed.
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    order = np.arange(len(data))
    np.random.shuffle(order)
    n_train, n_val = int(0.8 * len(data)), int(0.1 * len(data))
    d_train = data[order[:n_train]]
    d_test = data[order[n_train:n_train + n_val]]
    del data, order
    print(f"[crosshar] train={len(d_train)} val={len(d_test)}")

    ds_train = _ChunkedPretrainDataset(d_train, pipeline, DataTransform)
    ds_test = _ChunkedPretrainDataset(d_test, pipeline, DataTransform)
    del d_train, d_test
    # num_workers>0 overlaps CrossHAR's per-item span-masking -- which runs TWICE per
    # item (two contrastive views) and is the dominant cost -- with GPU compute.
    # _seed_worker keeps each worker's mask RNG independent (see above). drop_last kept
    # (NT-Xent needs a fixed batch size). Pure-speed, faithful. Measured ~1.6x fp32.
    nw = args.num_workers
    dl_extra = dict(num_workers=nw, persistent_workers=True, prefetch_factor=4,
                    worker_init_fn=_seed_worker) if nw > 0 else dict(num_workers=0)
    ld_train = DataLoader(ds_train, shuffle=True, batch_size=train_cfg.batch_size,
                          drop_last=True, pin_memory=True, **dl_extra)
    ld_test = DataLoader(ds_test, shuffle=False, batch_size=train_cfg.batch_size,
                         drop_last=True, pin_memory=True, **dl_extra)
    if len(ld_train) == 0 or len(ld_test) == 0:
        raise SystemExit(
            f"batch_size={train_cfg.batch_size} too large for train={len(ds_train)}/"
            f"val={len(ds_test)} windows (drop_last empties a loader); lower --batch-size "
            "or raise --max-per-stream.")

    masked_model = ch_models.MaskedModel4Pretrain(model_cfg).to(device)
    contrastive_model = Contrastive().to(device)
    criterion = nn.MSELoss(reduction="none")
    adam_extra = {"fused": True} if device.type == "cuda" else {}
    opt_m = torch.optim.Adam(masked_model.parameters(), lr=train_cfg.lr, **adam_extra)
    opt_c = torch.optim.Adam(contrastive_model.parameters(), lr=train_cfg.lr, **adam_extra)

    output = (args.output or BACKBONE_CKPT).resolve()
    save_prefix = (_HERE / "pretrain_run" / "model" if args.output is None else
                   output.parent / f"{output.stem}_run" / "model")
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    # NT-Xent is stateless. Upstream rebuilding its 1024x1024 exclusion mask on
    # every batch costs about 8 ms at batch 512 on this host.
    original_ntxent = ch_train.NTXentLoss
    ntxent_cache = {}
    def _cached_ntxent(*a, **k):
        device_arg = k.get("device", a[0] if a else device)
        batch_arg = k.get("batch_size", a[1] if len(a) > 1 else args.batch_size)
        key = (str(device_arg), int(batch_arg))
        if key not in ntxent_cache:
            ntxent_cache[key] = original_ntxent(*a, **k)
        return ntxent_cache[key]
    ch_train.NTXentLoss = _cached_ntxent
    trainer = ch_train.Trainer(train_cfg, masked_model, opt_m, contrastive_model, opt_c,
                               str(save_prefix), device, batch_size=train_cfg.batch_size,
                               criterion=criterion)
    trainer.pretrain(ld_train, ld_test)

    # Regression guard for the audit-P0 .train()-restore patch: pretrain's last op is a
    # validation run() (which sets .eval()); our patch follows it with .train(). If the
    # models are in eval mode here, the patch is inactive and the contrastive phase trained
    # with frozen BatchNorm -- fail loudly rather than ship a silently-degraded backbone.
    assert masked_model.training and contrastive_model.training, (
        "CrossHAR left in eval mode after pretrain -- the .train()-restore patch is inactive "
        "(contrastive-phase BatchNorm would be frozen).")

    # Trainer saves the best masked model to "<prefix>_masked_6_1.pt"; promote it
    # to the canonical checkpoint the adapter loads.
    best = save_prefix.parent / "model_masked_6_1.pt"
    sd = torch.load(str(best), map_location="cpu", weights_only=True)
    torch.save(sd, str(output))
    metadata = {
        "schema_version": METADATA_SCHEMA,
        "model": "crosshar",
        "corpus_profile": "expanded_phase_a",
        "train_datasets": list(prep.TRAIN_DATASETS),
        "prep_schema": prep.PREP_SCHEMA,
        "input_contract": {"rate_hz": prep.TARGET_HZ, "samples": prep.TARGET_LEN,
                           "channels": list(prep.SIX_CHANNELS),
                           "acc_convention": "g_then_per_window_instance_norm"},
        "recipe": args.recipe,
        "epochs": args.epochs,
        "contrastive_epochs": args.epochs_cl,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_per_stream": args.max_per_stream,
        "channel_augmentation": bool(args.augment),
        "seed": args.seed,
        "train_windows": len(ds_train),
        "validation_windows": len(ds_test),
        "optimizer_steps": len(ld_train) * args.epochs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[crosshar] saved backbone -> {output}")


if __name__ == "__main__":
    main()
