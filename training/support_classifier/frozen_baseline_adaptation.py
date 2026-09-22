"""Adapt a frozen released baseline encoder to our training corpus.

WHY THIS EXISTS, AND WHY IT IS NOT `--encoder-arch`. The matched-corpus arms
(`MatchedCorpusEncoder`) train a baseline *architecture* end to end, which requires the trunk's
preprocessing to be reimplemented inside HALO's encoder contract. That is affordable for HARNet,
LiMU-BERT and UniMTS. It is not affordable for NormWear: a measured 200 GPU-hours for one 40k arm,
plus gradient checkpointing to fit at all, plus stripping a ``@torch.no_grad()`` from the authors'
released code.

More importantly, full training is **not NormWear's protocol**. Across 11 datasets and 18
applications its authors only ever freeze the encoder and fit a closed-form probe on the
features -- logistic regression by Newton's method, ridge by Cholesky -- and they run no
random-init control anywhere. Their framing is a "starting point ... with minimal tuning". So the
treatment implemented here is the one its authors apply, given our corpus.

THE FIDELITY PROPERTY THAT MOTIVATES THE DESIGN. Features are produced by the baseline's **own**
``window_features``, in both this script and the evaluator. Nothing about anyone's resampling,
detrending or normalization is reimplemented here. That matters because the one place this project
*did* reimplement a released preprocessing contract, it diverged silently: the matched LiMU-BERT
entry fed 20 Hz clips to a checkpoint whose positional-embedding table pins it at 10 Hz. This path
cannot drift that way, because there is only one implementation.

WHAT IS TRAINED. A projection identical in shape to ``MatchedCorpusEncoder``'s -- Linear -> GELU ->
Dropout -> Linear -> LayerNorm, into ``d_model`` -- fitted with the differentiable-neighbour
objective, which is parameter-free and is exactly what the 1-NN readout measures. It is a strict
generalisation of the authors' linear probe: same frozen encoder, a learned map optimised for
neighbour retrieval rather than a logistic head.

BECAUSE THE TRUNK IS FROZEN, its features are fixed, so the corpus is encoded **once** and cached.
That is what turns 200 GPU-hours into about 35 minutes: measured 108 windows/s forward-only at
batch 64, 223,587 windows, and a 1.9 GiB cache.

    # encode the corpus once, then fit the projection on the cache
    .venv/bin/python -m training.support_classifier.frozen_baseline_adaptation \\
        --baseline normwear --steps 40000 --out runs/support-classifier/frozen_normwear_<date>

The resulting checkpoint is consumed by the evaluators via ``--baseline-projection
normwear=<path>``, which applies the projection to that baseline's frozen features before the
readout. Report these rows as a project method, never as a native baseline result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines import REGISTRY, UnsupportedEvaluationCell
from baselines.data import EvalStream
from data.scripts.curate.deployment_policy import SUPERVISED_HEAD_TRAIN_DATASETS
from training.support_classifier.corpus import support_corpus_from_index
from training.support_classifier.neighbors import differentiable_neighbor_logits
from training.support_classifier.sampling import draw_batch
from training.tokenizer.pretrain_data import CorpusIndex, PretrainDataset

#: Canonical channel order of a training-corpus window, matching the grids the adapters expect.
CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
#: Cache format. Bump when the stored quantity changes, so a stale cache can never be read.
CACHE_SCHEMA = "frozen-baseline-feature-v1-20260922"
PROJECTION_SCHEMA = "frozen-baseline-projection-v1-20260922"


class FrozenBaselineProjection(nn.Module):
    """The only trainable object here. Deliberately the same shape as the matched arms' adapter.

    Keeping the head identical across arms is what lets a frozen-baseline row and a matched-corpus
    row be read side by side: any difference is the encoder, not the head that reads it.
    """

    def __init__(self, in_dim: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.in_dim, self.d_model = int(in_dim), int(d_model)
        self.proj = nn.Sequential(
            nn.Linear(self.in_dim, 2 * self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.d_model, self.d_model),
        )
        self.row_norm = nn.LayerNorm(self.d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.row_norm(self.proj(features.float()))


def _stream_for(dataset, window_indices, *, name, stream_name, labels, subjects):
    """An ``EvalStream`` over training-corpus windows, so the adapter's own code path applies.

    Every field the adapters read is populated from the grid the trainer itself uses. Nothing is
    resampled, cropped or normalized here: that is the adapter's job and the whole point.
    """
    items = [dataset[int(index)] for index in window_indices]
    windows = np.stack([np.asarray(item["data"], dtype=np.float32) for item in items], axis=0)
    mask = np.asarray(items[0]["channel_mask"], dtype=bool)
    if any(not np.array_equal(np.asarray(item["channel_mask"], dtype=bool), mask)
           for item in items):
        raise ValueError(f"{name}/{stream_name}: channel mask is not constant within a stream")
    rate = float(items[0]["rate"])
    return EvalStream(
        dataset=name, stream=stream_name, alignment="native", windows=windows,
        gt=list(labels), subjects=np.asarray(subjects, dtype=object),
        channels=list(CHANNELS), rate_hz=rate, mask=mask,
        eval_labels=sorted(set(labels)),
        window_seconds=float(windows.shape[1]) / rate,
    )


def slot_channels(block: np.ndarray, mask: np.ndarray, per_channel: int) -> np.ndarray:
    """Place each real channel's sub-vector at its canonical slot, zero-filling absent channels.

    NormWear is channel-independent and its enrollment feature is ``768 x n_real_channels``, so an
    accelerometer-only stream returns 2304 and a six-axis stream 4608. One corpus-wide matrix needs
    a fixed width, and there are only three ways to get one.

    Averaging over channels is ruled out by the adapter itself: "Do not average channels merely to
    manufacture cross-configuration compatibility: that changes the representation and previously
    cost substantial enrolled accuracy." Fitting a separate projection per channel count would put
    the configurations in different learned spaces, which is precisely what a support vote across
    heterogeneous streams must not do.

    So the sub-vectors are scattered into fixed per-channel slots in canonical order and missing
    channels are zeros -- the same way absence is represented everywhere else in this pipeline. No
    value is altered, nothing is mixed across channels, and a channel the device never carried
    contributes nothing to a cosine comparison.
    """
    n_real = int(mask.sum())
    if block.shape[1] != n_real * per_channel:
        raise ValueError(f"expected {n_real}x{per_channel} channel-structured features, "
                         f"got width {block.shape[1]}")
    out = np.zeros((block.shape[0], len(CHANNELS) * per_channel), dtype=np.float32)
    for source, target in enumerate(np.flatnonzero(mask)):
        out[:, target * per_channel:(target + 1) * per_channel] = \
            block[:, source * per_channel:(source + 1) * per_channel]
    return out


def cache_features(baseline: str, corpus, dataset, device, cache_path: Path,
                   *, batch: int = 64) -> tuple[np.ndarray, dict]:
    """Encode every corpus recording through the baseline's own ``window_features``, once.

    Returns an ``(n_recordings, D)`` matrix aligned to ``corpus.recordings`` order, so an episode's
    recording index is directly a row index. Streams the adapter declares unsupported become
    all-zero rows and are recorded in the provenance rather than silently dropped -- a zero row
    cannot win a cosine neighbour vote, which is the honest behaviour for "this model cannot read
    this stream".
    """
    adapter = REGISTRY[baseline]
    state = adapter.setup_features(device)
    # Grouped by length as well as stream: a corpus window can be short (a truncated tail), and
    # the released adapters batch by valid length for exactly this reason. Mixing lengths into one
    # array would either fail to stack or force a pad this script has no business inventing.
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for position, record in enumerate(corpus.recordings):
        length = int(np.asarray(dataset[int(record.window_index)]["data"]).shape[0])
        groups[(record.dataset, record.stream, length)].append(position)

    features: np.ndarray | None = None
    per_channel: int | None = None
    unsupported: dict[str, str] = {}
    started = time.perf_counter()
    for seen, ((name, stream_name, _length), positions) in enumerate(sorted(groups.items()),
                                                                    start=1):
        records = [corpus.recordings[p] for p in positions]
        try:
            rows = []
            for start in range(0, len(positions), batch):
                chunk = records[start:start + batch]
                stream = _stream_for(
                    dataset, [r.window_index for r in chunk], name=name,
                    stream_name=stream_name, labels=[r.label for r in chunk],
                    subjects=[r.subject for r in chunk],
                )
                rows.append(np.asarray(
                    adapter.features_for_stream(stream, state, device), dtype=np.float32))
            block = np.concatenate(rows, axis=0)
        except UnsupportedEvaluationCell as reason:
            unsupported[f"{name}/{stream_name}"] = str(reason)
            continue
        mask = np.asarray(dataset[int(records[0].window_index)]["channel_mask"], dtype=bool)
        n_real = int(mask.sum())
        # Detect a channel-structured feature (NormWear) on the first stream and slot every later
        # stream the same way. A fixed-width trunk (harnet, limubert, unimts) leaves this at None.
        if per_channel is None and block.shape[1] % max(n_real, 1) == 0 and n_real:
            candidate = block.shape[1] // n_real
            per_channel = candidate if candidate * len(CHANNELS) != block.shape[1] else None
            if per_channel is not None:
                print(f"[cache] channel-structured features: {per_channel} per channel", flush=True)
        if per_channel is not None:
            block = slot_channels(block, mask, per_channel)
        if features is None:
            features = np.zeros((len(corpus.recordings), block.shape[1]), dtype=np.float32)
        elif block.shape[1] != features.shape[1]:
            raise ValueError(
                f"{name}/{stream_name} produced width {block.shape[1]}, expected "
                f"{features.shape[1]}; this baseline's feature width varies in a way "
                "slot_channels could not reconcile")
        features[np.asarray(positions)] = block
        print(f"[cache] {seen}/{len(groups)} {name}/{stream_name} "
              f"n={len(positions)} elapsed={(time.perf_counter() - started) / 60:.1f} min",
              flush=True)
    if features is None:
        raise SystemExit(f"{baseline} could not encode any training stream")

    provenance = {
        "cache_schema": CACHE_SCHEMA, "baseline": baseline,
        "n_recordings": int(features.shape[0]), "dim": int(features.shape[1]),
        "unsupported_streams": unsupported,
        "encode_minutes": round((time.perf_counter() - started) / 60, 2),
        "feature_artifacts": adapter.feature_artifacts(state),
        "feature_config": adapter.feature_config(state),
        "per_channel_dim": per_channel,
        "channel_slots": list(CHANNELS) if per_channel else None,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    cache_path.with_suffix(".json").write_text(json.dumps(provenance, indent=2, default=str))
    return features, provenance


def _episode_tensors(episodes, features: torch.Tensor):
    """Pack drawn episodes into the dense (query, support, candidate) tensors the vote needs."""
    device = features.device
    batch = len(episodes)
    max_support = max(len(e.support) for e in episodes)
    max_candidates = max(len(e.candidates) for e in episodes)
    support = features.new_zeros((batch, max_support, features.shape[1]))
    support_mask = torch.zeros((batch, max_support), dtype=torch.bool, device=device)
    support_slot = torch.zeros((batch, max_support), dtype=torch.long, device=device)
    candidate_mask = torch.zeros((batch, max_candidates), dtype=torch.bool, device=device)
    query = features.new_zeros((batch, features.shape[1]))
    target = torch.zeros((batch,), dtype=torch.long, device=device)
    for row, episode in enumerate(episodes):
        query[row] = features[episode.query]
        for column, (index, slot) in enumerate(zip(episode.support, episode.support_candidate)):
            support[row, column] = features[index]
            support_slot[row, column] = int(slot)
            support_mask[row, column] = True
        candidate_mask[row, :len(episode.candidates)] = True
        target[row] = int(episode.gt_slot)
    return query, support, support_slot, support_mask, candidate_mask, target


def train_projection(features: np.ndarray, corpus, *, steps: int, batch_size: int, lr: float,
                     d_model: int, seed: int, device, temperature: float, log_every: int):
    """Fit the projection with the parameter-free differentiable-neighbour objective."""
    table = torch.from_numpy(features).to(device)
    model = FrozenBaselineProjection(features.shape[1], d_model=d_model).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(steps, 1))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    history = []
    for step in range(1, steps + 1):
        episodes, _ = draw_batch(
            corpus, rng, batch_size=batch_size, deployment_matched=True,
            enrollment_k=(1, 2, 4, 8, 16, 32), queries_per_support_set=4,
            windows_per_execution=2, acquisition_mix=(0.5, 0.25, 0.25),
            enrollment_mix=(1.0, 0.0, 0.0), require_query_support=True,
            max_attempts_per_episode=64,
        )
        if not episodes:
            continue
        query, support, slot, support_mask, candidate_mask, target = _episode_tensors(
            episodes, table)
        projected_query = model(query)
        projected_support = model(support.reshape(-1, support.shape[-1])).reshape(
            support.shape[0], support.shape[1], -1)
        # The same closed-form centred soft vote the `neighbors` classifier uses, so this arm and
        # the matched-corpus arms are fitted against an identical objective.
        centre = (projected_support * support_mask.unsqueeze(-1)).sum(dim=1) \
            / support_mask.sum(dim=1, keepdim=True).clamp_min(1)
        logits, _ = differentiable_neighbor_logits(
            projected_query - centre, projected_support - centre.unsqueeze(1), slot,
            support_mask, candidate_mask, temperature=temperature,
        )
        loss = F.cross_entropy(logits, target)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
        if step % log_every == 0 or step == 1:
            accuracy = float((logits.argmax(dim=-1) == target).float().mean())
            history.append({"step": step, "loss": float(loss.detach()), "accuracy": accuracy})
            print(f"[fit] step {step:>6} loss {float(loss.detach()):.4f} acc {accuracy:.3f}",
                  flush=True)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, choices=sorted(REGISTRY),
                        help="released baseline whose frozen features are adapted")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="reused if present; defaults to cache/frozen_baseline/<baseline>.npy")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--support-sets", type=int, default=4,
                        help="episodes drawn per step, matching the trainer's default shape")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--support-temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--max-per-stream", type=int, default=4000)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    index = CorpusIndex(datasets=SUPERVISED_HEAD_TRAIN_DATASETS, alignment="native",
                        max_per_stream=args.max_per_stream, seed=args.seed,
                        window_seconds=args.window_seconds)
    dataset = PretrainDataset(index, index.train, augment=False, two_view=False,
                              conditioning_schema="acquisition-conditioning-v2",
                              multi_device_probability=0.0, max_devices=4)
    corpus = support_corpus_from_index(index, split="train")
    print(f"[corpus] {len(corpus.recordings)} recordings, {len(corpus.all_labels)} labels",
          flush=True)

    cache_path = args.feature_cache or Path("cache/frozen_baseline") / f"{args.baseline}.npy"
    sidecar = cache_path.with_suffix(".json")
    if cache_path.is_file() and sidecar.is_file():
        provenance = json.loads(sidecar.read_text())
        features = np.load(cache_path)
        if (provenance.get("cache_schema") != CACHE_SCHEMA
                or features.shape[0] != len(corpus.recordings)):
            raise SystemExit(f"{cache_path} does not match this corpus or schema; delete it")
        print(f"[cache] reusing {cache_path} {features.shape}", flush=True)
    else:
        features, provenance = cache_features(
            args.baseline, corpus, dataset, device, cache_path, batch=args.encode_batch)
        print(f"[cache] wrote {cache_path} {features.shape} in "
              f"{provenance['encode_minutes']:.1f} min", flush=True)

    model, history = train_projection(
        features, corpus, steps=args.steps, batch_size=args.support_sets, lr=args.lr,
        d_model=args.d_model, seed=args.seed, device=device,
        temperature=args.support_temperature, log_every=args.log_every,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    trainable = sum(p.numel() for p in model.parameters())
    payload = {
        "projection_schema": PROJECTION_SCHEMA,
        "baseline": args.baseline,
        "projection": {k: v.cpu() for k, v in model.state_dict().items()},
        "in_dim": model.in_dim, "d_model": model.d_model,
        "trainable_parameters": trainable,
        "steps": args.steps, "support_sets": args.support_sets, "lr": args.lr,
        "seed": args.seed, "support_temperature": args.support_temperature,
        "window_seconds": args.window_seconds,
        "train_datasets": sorted(SUPERVISED_HEAD_TRAIN_DATASETS),
        "corpus_fingerprint": hashlib.sha256(
            json.dumps({"n": len(corpus.recordings),
                        "labels": sorted(corpus.all_labels)}, sort_keys=True).encode()
        ).hexdigest(),
        "feature_cache": provenance,
        "history": history,
    }
    torch.save(payload, args.out / "projection.pt")
    (args.out / "summary.json").write_text(json.dumps(
        {k: v for k, v in payload.items() if k != "projection"}, indent=2, default=str))
    print(f"[done] {args.out/'projection.pt'}  trainable={trainable:,}  "
          f"frozen trunk={args.baseline}", flush=True)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- evaluation side
def load_projection(path, device=None):
    """Rebuild a trained projection from its checkpoint, for use inside an evaluator."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("projection_schema") != PROJECTION_SCHEMA:
        raise ValueError(f"{path}: expected {PROJECTION_SCHEMA}, got "
                         f"{blob.get('projection_schema')!r}")
    model = FrozenBaselineProjection(blob["in_dim"], d_model=blob["d_model"])
    model.load_state_dict(blob["projection"])
    model.eval().requires_grad_(False)
    if device is not None:
        model = model.to(device)
    return model, blob


def stream_channel_mask(stream) -> np.ndarray:
    """The channel mask to slot by, for a single-device or a multi-device evaluation cell.

    A ``MultiDeviceEvalStream`` has no ``mask`` of its own; its members each carry one, and
    ``features_for_stream`` pools the members' feature vectors. Pooling requires equal widths, so
    a cell whose devices disagreed on channel count would already have failed upstream -- but
    assert it rather than assume it, because slotting by the wrong mask would permute the feature
    space silently, which is the exact failure this module is built to avoid.
    """
    if hasattr(stream, "mask"):
        return np.asarray(stream.mask, dtype=bool)
    masks = [np.asarray(device.mask, dtype=bool) for device in stream.devices]
    if any(not np.array_equal(m, masks[0]) for m in masks[1:]):
        raise ValueError(
            f"{getattr(stream, 'cell_id', stream)}: devices disagree on which channels are real, "
            "so their pooled feature has no single channel slotting")
    return masks[0]


def apply_projection(model, blob, features: np.ndarray, mask, device=None) -> np.ndarray:
    """Map frozen baseline features through the trained projection.

    Channel slotting is repeated here exactly as at fit time, from the stream's own channel mask,
    so a channel-structured baseline lands in the same slots it was fitted in. Getting this wrong
    would silently permute the feature space between training and evaluation, which is the class
    of bug that motivated the fidelity gate in the first place.
    """
    per_channel = (blob.get("feature_cache") or {}).get("per_channel_dim")
    values = np.asarray(features, dtype=np.float32)
    if per_channel:
        mask = np.asarray(mask, dtype=bool)
        expected = int(mask.sum()) * int(per_channel)
        if values.shape[1] != expected:
            # A composite cell hands a channel-independent trunk every device's channels at once,
            # so the feature is wider than any single placement. The projection was fitted only on
            # single-device windows (the cache is built with multi_device_probability = 0), so
            # there is no honest slotting for this and no fitted space to map it into. Decline the
            # cell through the sanctioned channel rather than invent a width.
            raise UnsupportedEvaluationCell(
                f"frozen-baseline projection was fitted on single-device features "
                f"({model.in_dim}-wide); this cell produced {values.shape[1]}")
        values = slot_channels(values, mask, int(per_channel))
    if values.shape[1] != model.in_dim:
        raise UnsupportedEvaluationCell(
            f"projection expects width {model.in_dim}, got {values.shape[1]}")
    with torch.no_grad():
        tensor = torch.from_numpy(values)
        if device is not None:
            tensor = tensor.to(device)
        return model(tensor).float().cpu().numpy()


def parse_projection_arguments(values, device=None) -> dict:
    """``name=path`` pairs -> {baseline: (module, blob)}; refuses a mismatched baseline name."""
    out = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"--baseline-projection expects name=path, got {item!r}")
        name, path = item.split("=", 1)
        model, blob = load_projection(path, device=device)
        if blob.get("baseline") != name:
            raise ValueError(f"{path} was fitted for {blob.get('baseline')!r}, not {name!r}")
        out[name] = (model, blob, str(path))
    return out
