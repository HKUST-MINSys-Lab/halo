"""Experimental v5 online-memory training; does not read or select on sealed data.

Example: uv run python -m training.support_classifier.train_memory --source-checkpoint
  runs/support-classifier/halo_t6_40k_20260920/last.pt --out runs/support-classifier/v5_screen
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from data.scripts.curate import deployment_policy
from model.blocks import AttentionSpec
from model.support.memory_classifier import (
    ARCHITECTURE_VERSION, MemoryReaderClassifier, MemoryReaderConfig,
)
from training.support_classifier.collate import SupportCollate
from training.support_classifier.corpus import (
    open_vocabulary_holdout_labels, support_corpus_from_index,
)
from training.support_classifier.memory_bank import DeploymentMemory
from training.support_classifier.memory_episodes import (
    MemoryEpisode, draw_memory_episode, eligible_memory_datasets,
)
from training.support_classifier.train import (
    PROVENANCE_ROOTS, LabelTextTable, _atomic_torch_save, encode_recording_rows,
)
from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain import corpus_fingerprint, capture_source_provenance, write_source_provenance
from training.tokenizer.pretrain_data import CorpusIndex, MultiResolutionCollate, PretrainDataset
from evaluation.provenance import _atomic_json

MEMORY_SIZE_BINS = ("0", "1", "2_3", "4_7", "8_15", "16_31", "32_plus")
EVALUATION_PROTOCOL_KEYS = (
    "seed", "val_episodes", "max_history", "max_per_stream", "reader_holdout_fraction",
    "fine_tune_encoder", "train_text_projection",
)


def validate_evaluation_protocol(current: dict, saved: dict) -> None:
    mismatches = [key for key in EVALUATION_PROTOCOL_KEYS if key not in saved or
                  current.get(key) != saved[key]]
    if mismatches:
        raise ValueError(f"evaluation protocol differs from checkpoint: {mismatches}; "
                         "pass its training settings explicitly")


def memory_size_bin(size: int) -> str:
    if size < 0:
        raise ValueError("memory size cannot be negative")
    return "0" if size == 0 else "1" if size == 1 else "2_3" if size < 4 else \
        "4_7" if size < 8 else "8_15" if size < 16 else "16_31" if size < 32 else "32_plus"


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def indexed_corpus_fingerprint(index: CorpusIndex) -> str:
    """Bind v5 to both grid content and the exact quality-screened train/validation rows."""
    digest = hashlib.sha256(corpus_fingerprint(index).encode())
    for split in (index.train, index.val):
        for key in split:
            digest.update(f"{key.stream_i}:{key.window_i}:{key.label_id};".encode())
        digest.update(b"|")
    for stream in sorted(index.excluded):
        digest.update(stream.encode())
        for row in sorted(index.excluded[stream]):
            digest.update(f"{row},".encode())
    return digest.hexdigest()


def episode_loss(
    episode: MemoryEpisode, corpus, dataset, collate, encoder, classifier,
    text: LabelTextTable, device: torch.device, *, counterfactual: bool,
    reader_holdout: frozenset[str] = frozenset(),
) -> tuple[torch.Tensor, dict[str, float], list[tuple[str, str, str, str]]]:
    items = [dataset[corpus.recordings[row].window_index] for row in episode.rows]
    encoded = collate.bucketed(items)
    motion, acquisition, _ = encode_recording_rows(encoder, encoded, device)
    candidate_text = text.matrix[text.ids(episode.roster)]
    bank = DeploymentMemory("current-step", capacity=classifier.cfg.max_entries)
    losses = []
    observations: list[tuple[str, str, str, str]] = []
    counts = dict(scored=0, correct=0, semantic_correct=0, fixed_correct=0,
                  corrections=0, new_errors=0, retained=0, seen=0, verified=0,
                  empty_scored=0, empty_correct=0, predicted_scored=0, predicted_correct=0,
                  verified_scored=0, verified_correct=0, mixed_scored=0, mixed_correct=0,
                  semantic_weight_sum=0, reliability_max_sum=0,
                  holdout_scored=0, holdout_correct=0,
                  counterfactual_views=0)
    for size_bin in MEMORY_SIZE_BINS:
        for key in ("scored", "reader_correct", "semantic_correct", "fixed_correct"):
            counts[f"size_{size_bin}_{key}"] = 0
    for i, row in enumerate(episode.rows):
        record = corpus.recordings[row]
        tensors = bank.tensors(classifier, episode.roster, candidate_text, empty_device=device)
        readout = classifier(motion[i], acquisition[i], candidate_text, *tensors)
        if record.label in episode.roster and not episode.verified[i]:
            target = episode.roster.index(record.label)
            losses.append(-readout.probabilities[target].clamp_min(1e-8).log())
            if counterfactual and i == len(episode.rows) - 1 and bank.entries:
                # Same query and roster under a different realistic history: no entries, then
                # only predicted or only verified entries. There is no synthetic label flip.
                mask = torch.tensor([e.verified_label is None for e in bank.entries], device=device)
                keep = (mask if i % 2 else ~mask) if bool(mask.any()) and bool((~mask).any()) \
                    else torch.zeros_like(mask)
                view = classifier(motion[i], acquisition[i], candidate_text,
                                  *(tensor[keep] for tensor in tensors))
                losses.append(-view.probabilities[target].clamp_min(1e-8).log())
                counts["counterfactual_views"] += 1
            pred = int(readout.probabilities.argmax())
            semantic_pred = int(readout.semantic.argmax())
            counts["scored"] += 1
            counts["correct"] += pred == target
            counts["semantic_correct"] += semantic_pred == target
            counts["fixed_correct"] += int(readout.fixed.argmax()) == target
            observations.append((record.label, episode.roster[pred],
                                 episode.roster[semantic_pred],
                                 episode.roster[int(readout.fixed.argmax())]))
            counts["corrections"] += pred == target and semantic_pred != target
            counts["new_errors"] += pred != target and semantic_pred == target
            size_bin = memory_size_bin(len(bank.entries))
            counts[f"size_{size_bin}_scored"] += 1
            counts[f"size_{size_bin}_reader_correct"] += pred == target
            counts[f"size_{size_bin}_semantic_correct"] += semantic_pred == target
            counts[f"size_{size_bin}_fixed_correct"] += int(readout.fixed.argmax()) == target
            if record.label in reader_holdout:
                counts["holdout_scored"] += 1
                counts["holdout_correct"] += pred == target
            sources = {e.verified_label is not None for e in bank.entries}
            regime = ("empty" if not sources else "mixed" if len(sources) == 2 else
                      "verified" if True in sources else "predicted")
            counts[f"{regime}_scored"] += 1
            counts[f"{regime}_correct"] += pred == target
            counts["semantic_weight_sum"] += float(readout.semantic_weight.mean().detach())
            if readout.reliability.numel():
                counts["reliability_max_sum"] += float(readout.reliability.max().detach())
        bank.insert(
            recording_id=f"{record.dataset}:{record.stream}:{record.window_index}",
            execution_id=f"{record.dataset}:{record.subject}:{record.execution}",
            motion=motion[i], acquisition=acquisition[i],
            original_probabilities=readout.semantic,
            roster=episode.roster,
            verified_label=record.label if episode.verified[i] else None,
            acquisition_metadata={"dataset": record.dataset, "stream": record.stream},
        )
    counts["retained"] = len(bank.entries)
    counts["seen"] = bank.seen
    counts["verified"] = sum(e.verified_label is not None for e in bank.entries)
    if not losses:
        raise RuntimeError("online episode has no supervised query")
    return (torch.stack(losses).mean(),
            {key: float(value) for key, value in counts.items()}, observations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume a v5 last.pt with identical curriculum and optimizer")
    parser.add_argument("--evaluate-checkpoint", type=Path, default=None,
                        help="validate a trained v5 snapshot without fitting; never reads sealed data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--episodes-per-step", type=int, default=4)
    parser.add_argument("--val-episodes", type=int, default=64)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--max-history", type=int, default=63,
                        help="longest simulated history; the default fills the reader's 64-entry "
                             "size scale so deployment memories are not longer than any trained on")
    parser.add_argument("--max-per-stream", type=int, default=None)
    parser.add_argument("--reader-holdout-fraction", type=float, default=0.2,
                        help="globally held-out labels for reader optimization; source v4 encoder saw them")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--fine-tune-encoder", action="store_true")
    parser.add_argument("--train-text-projection", action="store_true",
                        help="also train v4's p_text; off by default to preserve its text projection "
                             "(fine-tuning the encoder still changes the no-memory control)")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.episodes_per_step < 1 or args.val_episodes < 1 or args.val_every < 1:
        parser.error("step and episode counts must be positive")
    if args.max_history < 1 or args.lr <= 0 or args.encoder_lr_scale <= 0:
        parser.error("max history and learning rates must be positive")
    if not 0 <= args.reader_holdout_fraction < 1:
        parser.error("reader holdout fraction must be in [0,1)")
    if args.smoke:
        args.steps, args.val_episodes, args.val_every = 2, 4, 1
        args.max_per_stream = min(args.max_per_stream or 48, 48)
    if args.resume is not None and args.evaluate_checkpoint is not None:
        parser.error("resume and evaluate-checkpoint are mutually exclusive")
    if args.resume is not None and args.resume.resolve().parent != args.out.resolve():
        parser.error("resume checkpoint must be inside the same output directory")
    if args.out.exists() and any(args.out.iterdir()) and args.resume is None:
        parser.error("output directory must be absent or empty")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    if source.get("architecture_version") != "support_classifier_v4":
        parser.error("v5 initialization currently requires a v4 source checkpoint")
    source_hash = checkpoint_sha256(args.source_checkpoint)
    source_provenance = capture_source_provenance(
        args.out, write=False, roots=PROVENANCE_ROOTS,
    )
    source_record = {key: value for key, value in source_provenance.items() if key != "_patch"}
    spec = AttentionSpec(**source["attention_spec"])
    head = MemoryReaderClassifier(spec, MemoryReaderConfig(
        text_dim=source["classifier_config"]["text_dim"], hidden_dim=spec.d_model,
        semantic_temperature=float(source["classifier_config"]["text_temperature"]),
        neighbor_temperature=float(source["classifier_config"]["temperature"]),
        max_entries=max(64, args.max_history + 1),
    )).to(device)
    head.p_text.load_state_dict({
        "weight": source["classifier"]["p_text.weight"],
        "bias": source["classifier"]["p_text.bias"],
    })
    if not args.train_text_projection:
        # Cross-entropy on the training vocabulary would otherwise retune v4's zero-shot text
        # path, so v5's "no memory" control would no longer be v4 and unseen-label transfer could
        # regress (as T7's did).
        head.p_text.requires_grad_(False)
    encoder = build_encoder(source, device, training=args.fine_tune_encoder)
    if not args.fine_tune_encoder:
        encoder.requires_grad_(False)
        encoder.eval()
    deployment_policy.assert_no_retired_sources(deployment_policy.SUPERVISED_HEAD_TRAIN_DATASETS)
    # The subject split must be the source encoder's: CorpusIndex's validation subjects depend on
    # its seed, and with v5's own seed 12 of 23 "held-out" subjects were v4 training subjects
    # (2026-09-25 sweep). ``--seed`` still drives episodes and the reader label holdout.
    source_args = source["args"]
    split_seed = int(source_args.get("data_seed", source_args["seed"]))
    index = CorpusIndex(max_per_stream=args.max_per_stream, seed=split_seed,
                        datasets=deployment_policy.SUPERVISED_HEAD_TRAIN_DATASETS,
                        alignment="native", window_seconds=float(source["args"]["window_seconds"]))
    corpus_fp = indexed_corpus_fingerprint(index)
    holdout = open_vocabulary_holdout_labels(index, fraction=args.reader_holdout_fraction,
                                             seed=args.seed)
    train_corpus = support_corpus_from_index(index, exclude_labels=holdout)
    val_corpus = support_corpus_from_index(index, split="val")
    common_dataset = dict(augment=False, two_view=False,
                          neutral_acquisition_text=bool(source["config"].get("neutral_acquisition_text", False)),
                          conditioning_schema=source["config"]["conditioning_schema"])
    train_data = PretrainDataset(index, index.train, **common_dataset)
    val_data = PretrainDataset(index, index.val, **common_dataset)
    durations = source["args"].get("resolutions") or source["config"].get("eval_resolutions")
    if not durations:
        parser.error("source checkpoint lacks its fixed-resolution tokenization contract")
    collate = SupportCollate(MultiResolutionCollate(fixed_patch_seconds=tuple(durations)))
    labels = sorted(set(train_corpus.all_labels) | set(val_corpus.all_labels))
    text = LabelTextTable(labels, device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in head.parameters() if p.requires_grad], "lr": args.lr},
        {"params": [p for p in encoder.parameters() if p.requires_grad],
         "lr": args.lr * args.encoder_lr_scale},
    ], weight_decay=0.01)
    config = {"architecture_version": ARCHITECTURE_VERSION,
              "source_checkpoint_sha256": source_hash, "source_step": source.get("step"),
              "training_source": list(deployment_policy.SUPERVISED_HEAD_TRAIN_DATASETS),
              "reader_heldout_labels": list(holdout),
              "subject_split_seed": split_seed,
              "corpus_fingerprint": corpus_fp,
              "semantic_control": ("source_v4_text_with_finetuned_encoder" if args.fine_tune_encoder
                                   else "source_v4_text_with_frozen_encoder"),
              "source_provenance": source_record,
              "args": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()}}
    if args.resume is None:
        _atomic_json(args.out / "run_config.json", config)
        write_source_provenance(args.out, source_provenance)

    def validate(step: int) -> dict:
        head.eval()
        encoder.eval()
        totals: dict[str, dict[str, float]] = {}
        predictions: dict[str, list[tuple[str, str, str, str]]] = {}
        acquisition: dict[str, list[float]] = {}
        with torch.no_grad():
            datasets = eligible_memory_datasets(val_corpus)
            if not datasets:
                raise ValueError("development corpus has no execution-distinct online episode")
            for j in range(args.val_episodes):
                episode = draw_memory_episode(val_corpus, np.random.default_rng(args.seed + 900000 + j),
                                              max_history=args.max_history,
                                              dataset=datasets[j % len(datasets)])
                loss, stats, observed = episode_loss(
                    episode, val_corpus, val_data, collate, encoder,
                    head, text, device, counterfactual=False,
                    reader_holdout=frozenset(holdout),
                )
                predictions.setdefault(episode.dataset, []).extend(observed)
                acquisition.setdefault(episode.dataset, []).append(
                    episode.realized_mismatch_fraction,
                )
                row = totals.setdefault(episode.dataset, dict(loss=0., episodes=0.,
                    **{key: 0. for key in stats if key not in {"retained", "seen", "verified"}}))
                row["loss"] += float(loss)
                row["episodes"] += 1
                for key in row.keys() - {"loss", "episodes"}:
                    row[key] += stats[key]
        for row in totals.values():
            for key in ("correct", "semantic_correct", "fixed_correct", "corrections", "new_errors"):
                row[key + "_rate"] = row[key] / max(1, row["scored"])
            for regime in ("empty", "predicted", "verified", "mixed"):
                row[f"{regime}_accuracy"] = (
                    row[f"{regime}_correct"] / row[f"{regime}_scored"]
                    if row[f"{regime}_scored"] else None
                )
            row["mean_semantic_weight"] = row["semantic_weight_sum"] / max(1, row["scored"])
            row["mean_top_reliability"] = row["reliability_max_sum"] / max(1, row["scored"])
            for size_bin in MEMORY_SIZE_BINS:
                n = row[f"size_{size_bin}_scored"]
                for method in ("reader", "semantic", "fixed"):
                    row[f"size_{size_bin}_{method}_accuracy"] = (
                        row[f"size_{size_bin}_{method}_correct"] / n if n else None
                    )
            row["reader_holdout_accuracy"] = (
                row["holdout_correct"] / row["holdout_scored"]
                if row["holdout_scored"] else None
            )
            row["loss"] /= row["episodes"]
        for dataset, observed in predictions.items():
            truth = [item[0] for item in observed]
            if not truth:
                continue
            labels_in_truth = sorted(set(truth))
            for name, column in (("reader", 1), ("no_memory", 2), ("fixed_vote", 3)):
                guess = [item[column] for item in observed]
                totals[dataset][f"{name}_macro_f1"] = float(
                    f1_score(truth, guess, labels=labels_in_truth, average="macro",
                             zero_division=0),
                )
                totals[dataset][f"{name}_balanced_accuracy"] = float(
                    np.mean([sum(p == label for t, p in zip(truth, guess) if t == label)
                             / sum(t == label for t in truth) for label in labels_in_truth]),
                )
            totals[dataset]["realized_acquisition_mismatch_fraction"] = float(
                np.mean(acquisition[dataset]),
            )
        return {"step": step, "datasets": totals,
                "semantic_control": config["semantic_control"],
                "reader_heldout_labels": list(holdout),
                "macro_accuracy": float(np.mean([v["correct_rate"] for v in totals.values()])),
                "macro_f1": float(np.mean([v["reader_macro_f1"] for v in totals.values()
                                            if "reader_macro_f1" in v]))}

    def save(name: str, step: int, result: dict) -> None:
        _atomic_torch_save({"architecture_version": ARCHITECTURE_VERSION,
                    "attention_spec": asdict(spec), "classifier_config": asdict(head.cfg),
                    "classifier": head.state_dict(), "encoder": encoder.state_dict(),
                    "config": source["config"], "source_checkpoint_sha256": source_hash,
                    "source_step": source.get("step"), "step": step, "validation": result,
                    "optimizer": optimizer.state_dict(), "training_args": config["args"],
                    "corpus_fingerprint": corpus_fp,
                    "source_provenance": source_record,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                   args.out / name)

    best = -1.0
    if args.evaluate_checkpoint is not None:
        snapshot = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        if snapshot.get("architecture_version") != ARCHITECTURE_VERSION or \
                snapshot.get("source_checkpoint_sha256") != source_hash:
            parser.error("evaluation checkpoint is not v5 or does not match the source encoder")
        if snapshot.get("source_provenance") != source_record:
            parser.error("evaluation code differs from the checkpoint's source provenance")
        if snapshot.get("corpus_fingerprint") != corpus_fp:
            parser.error("evaluation corpus differs from the checkpoint's indexed grids or quality screens")
        try:
            validate_evaluation_protocol(config["args"], snapshot.get("training_args") or {})
        except ValueError as exc:
            parser.error(str(exc))
        head.load_state_dict(snapshot["classifier"], strict=True)
        encoder.load_state_dict(snapshot["encoder"], strict=True)
        result = validate(int(snapshot["step"]))
        (args.out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    start_step = 1
    if args.resume is not None:
        snapshot = torch.load(args.resume, map_location="cpu", weights_only=False)
        if snapshot.get("architecture_version") != ARCHITECTURE_VERSION or \
                snapshot.get("source_checkpoint_sha256") != source_hash:
            parser.error("resume checkpoint is not v5 or has a different source encoder")
        if snapshot.get("source_provenance") != source_record:
            parser.error("resume code differs from the checkpoint's source provenance")
        if snapshot.get("corpus_fingerprint") != corpus_fp:
            parser.error("resume corpus differs from the checkpoint's indexed grids or quality screens")
        old = snapshot.get("training_args") or {}
        allowed = {"steps", "out", "device", "resume", "evaluate_checkpoint"}
        mismatch = [key for key, value in old.items()
                    if key not in allowed and config["args"].get(key) != value]
        if mismatch:
            parser.error(f"resume curriculum/optimizer differs: {sorted(mismatch)}")
        if snapshot.get("optimizer") is None:
            parser.error("resume checkpoint has no optimizer state")
        head.load_state_dict(snapshot["classifier"], strict=True)
        encoder.load_state_dict(snapshot["encoder"], strict=True)
        optimizer.load_state_dict(snapshot["optimizer"])
        if snapshot.get("torch_rng") is None:
            parser.error("resume checkpoint lacks its torch RNG state")
        torch.set_rng_state(snapshot["torch_rng"])
        if device.type == "cuda":
            if snapshot.get("cuda_rng") is None:
                parser.error("CUDA resume checkpoint lacks its CUDA RNG state")
            torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
        start_step = int(snapshot["step"]) + 1
        best = float(snapshot["validation"].get("macro_f1", -1.0))
        best_path = args.out / "best_internal.pt"
        if best_path.exists():
            previous_best = torch.load(best_path, map_location="cpu", weights_only=False)
            best = max(best, float(previous_best["validation"].get("macro_f1", -1.0)))
        if args.steps < start_step:
            parser.error("--steps must exceed the checkpoint's completed step")
        previous_config = json.loads((args.out / "run_config.json").read_text())
        config["resume_history"] = [*previous_config.get("resume_history", []), {
            "checkpoint": str(args.resume), "completed_step": start_step - 1,
            "previous_steps": previous_config["args"]["steps"], "requested_steps": args.steps,
        }]
        config["initial_training_args"] = previous_config.get("initial_training_args",
                                                            previous_config["args"])
        _atomic_json(args.out / "run_config.json", config)
    log = (args.out / "log.jsonl").open("a" if args.resume else "w")
    try:
        for step in range(start_step, args.steps + 1):
            head.train()
            if args.fine_tune_encoder:
                encoder.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            counts = []
            episodes = []
            for j in range(args.episodes_per_step):
                rng = np.random.default_rng(args.seed + step * 1009 + j)
                episode = draw_memory_episode(train_corpus, rng, max_history=args.max_history)
                episodes.append(episode)
                loss, stats, _ = episode_loss(episode, train_corpus, train_data, collate,
                                              encoder, head, text, device,
                                              counterfactual=True)
                losses.append(loss)
                counts.append(stats)
            batch_loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(batch_loss)):
                raise FloatingPointError("non-finite v5 training objective")
            batch_loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + [p for p in encoder.parameters() if p.requires_grad], 1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            record = {"step": step, "loss": float(batch_loss.detach()), "grad_norm": float(grad),
                      "semantic_control": config["semantic_control"],
                      "mean_retained": float(np.mean([v["retained"] for v in counts])),
                      "mean_seen": float(np.mean([v["seen"] for v in counts])),
                      "mean_verified": float(np.mean([v["verified"] for v in counts])),
                      "train_scored": float(sum(v["scored"] for v in counts)),
                      "counterfactual_views": float(sum(v["counterfactual_views"] for v in counts)),
                      "train_corrections": float(sum(v["corrections"] for v in counts)),
                      "train_new_errors": float(sum(v["new_errors"] for v in counts))}
            for regime in ("matched", "cross_placement", "cross_dataset"):
                record[f"requested_{regime}_episodes"] = sum(
                    episode.requested_regime == regime for episode in episodes
                )
            record["realized_acquisition_mismatch_fraction"] = float(np.mean([
                episode.realized_mismatch_fraction for episode in episodes
            ]))
            record["realized_cross_dataset_fraction"] = float(np.mean([
                episode.realized_cross_dataset_fraction for episode in episodes
            ]))
            if step % args.val_every == 0 or step == args.steps:
                result = validate(step)
                record["validation"] = result
                save("last.pt", step, result)
                if result["macro_f1"] > best:
                    best = result["macro_f1"]
                    save("best_internal.pt", step, result)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % args.val_every == 0 or args.smoke:
                print(json.dumps(record), flush=True)
    finally:
        log.close()


if __name__ == "__main__":
    main()
