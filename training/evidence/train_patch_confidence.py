"""Calibrate Phase-B reject confidence with a frozen patch evidence predictor.

The predictor is never updated here. The confidence target is one only when the ground truth is an
allowed candidate and the frozen predictor selects it; truth-absent and incorrect episodes target
zero. This learns selective prediction without introducing an ``UNKNOWN`` candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.scoring import get_sbert_encoder
from model.evidence.confidence import (
    EvidenceConfidenceHead,
    aurc,
    binary_auprc,
    binary_auroc,
    expected_calibration_error,
)
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from training.evidence.bank_guard import (
    assert_artifact_matches_bank,
    assert_bank_matches_backbone,
    assert_bank_current,
    assert_embedding_path_current,
    assert_patch_bank,
    assert_patch_embedding_path_current,
    bank_fingerprint,
)
from training.evidence.labeltext import ensemble_text
from training.evidence.episode_labels import encode_neutral_aliases, episode_label_set
from training.evidence.device import resolve_device
from training.evidence.folds import VALIDATION_QUERY_POLICY, phase_b_fold_masks
from training.evidence.patch_episodes import (
    PatchTable,
    build_episode_memory_view,
    support_capacity_by_label,
)
from training.evidence.live_encoder import SourcePatchEncoder
from training.evidence.policy import (
    ACTIVE_WINDOWS_PER_LABEL,
    CANDIDATE_COUNT_RANGE,
    EPISODE_TYPES,
    LABEL_TEXT_MODES,
    PHASE_B_TRAINING_REGIME,
    PHYSICAL_VIEW_MODES,
    SUPPORT_COUNT_RANGE,
    PhaseBPolicy,
)
from training.evidence.train_patch_decoder import (
    build_decoder,
    choose_candidates,
    atomic_torch_save,
    decode_adaptation_episode,
    physical_label_centroids,
    sample_queries,
    sample_queries_covering_labels,
    synthetic_smoke_bank,
)
from training.evidence.telemetry import PhaseBTelemetry
from training.tokenizer.eval_transfer import build_encoder

_DIR = Path(__file__).resolve().parent / "outputs"
_DEFAULT_BANK = _DIR / "memory_bank.pt"
_DEFAULT_PREDICTOR = _DIR / "patch_evidence_predictor.pt"
_DEFAULT_OUT = _DIR / "patch_evidence_confidence.pt"
SEED = 20260807


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=_DEFAULT_BANK)
    ap.add_argument("--predictor", type=Path, default=_DEFAULT_PREDICTOR)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Phase-A checkpoint used to construct a fine-tuned predictor")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--truth-absent-prob", type=float, default=0.25)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-episodes", type=int, default=8)
    ap.add_argument("--val-queries", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--telemetry-seconds", type=float, default=60.0)
    ap.add_argument("--telemetry-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.device = "cpu"; args.steps = 2; args.batch = 4
        args.val_every = 1; args.val_episodes = 2; args.val_queries = 4
        args.save_every = 1
        args.telemetry_seconds = 0.01
        args.predictor = Path("/tmp/halo_phase_b_predictor_smoke.pt")
        args.out = Path("/tmp/halo_phase_b_confidence_smoke.pt") \
            if args.out == _DEFAULT_OUT else args.out
    if args.steps < 1 or args.batch < 1 or args.val_every < 1 or args.save_every < 1:
        ap.error("steps, batch, val-every, and save-every must be positive")
    if not 0.0 < args.truth_absent_prob < 1.0:
        ap.error("truth-absent-prob must be in (0,1)")
    if args.telemetry_seconds <= 0:
        ap.error("telemetry-seconds must be positive")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bank = synthetic_smoke_bank() if args.smoke else torch.load(
        args.bank, map_location="cpu", weights_only=True
    )
    if not args.smoke:
        assert_bank_current(bank, context="train_patch_confidence")
    assert_patch_bank(bank, context="train_patch_confidence")
    predictor = torch.load(args.predictor, map_location="cpu", weights_only=True)
    if not args.smoke:
        assert_artifact_matches_bank(
            predictor, bank, context="train_patch_confidence",
            artifact_name="patch evidence predictor",
        )
    if predictor.get("objective") != (
        "candidate_cross_entropy_plus_matched_support_counterfactuals"
    ):
        raise SystemExit(
            "confidence calibration requires the current counterfactual Phase-B predictor artifact"
        )
    if predictor.get("training_regime") != PHASE_B_TRAINING_REGIME:
        raise SystemExit(
            "confidence calibration requires a predictor trained with the current adaptation regime"
        )

    cfg = predictor["cfg"]
    decoder = build_decoder(cfg).to(device)
    decoder.load_state_dict(predictor["decoder"])
    retriever = PatchSubspaceRetriever(
        cfg["d_model"], cfg["n_subspaces"], cfg["subspace_dim"], cfg["subspace_ema"]
    ).to(device)
    retriever.load_state_dict(predictor["retriever"])
    decoder.eval(); retriever.eval()
    for parameter in list(decoder.parameters()) + list(retriever.parameters()):
        parameter.requires_grad_(False)
    retrieval_cfg = predictor["retrieval_cfg"]
    policy = PhaseBPolicy(
        int(retrieval_cfg["evidence_budget"]), cfg.get("tokenizer_mode", "frozen")
    )
    live_source = tokenizer = None
    if not args.smoke:
        checkpoint_path = args.checkpoint or Path(bank["backbone"]["checkpoint"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"fine-tuned predictor checkpoint not found at {checkpoint_path}; pass --checkpoint"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert_bank_matches_backbone(bank, checkpoint, context="train_patch_confidence")
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() \
                != bank["backbone"].get("fingerprint"):
            raise SystemExit("confidence checkpoint is not the encoder that built the memory bank")
        tokenizer = build_encoder(checkpoint, device)
        assert_embedding_path_current(
            bank, tokenizer, device, context="train_patch_confidence"
        )
        assert_patch_embedding_path_current(
            bank, tokenizer, device, context="train_patch_confidence"
        )
        if policy.tokenizer_mode == "ema_finetune":
            tokenizer.load_state_dict(predictor["tokenizer_ema"])
        tokenizer.eval()
        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)
        live_source = SourcePatchEncoder(bank, device)

    table = PatchTable(bank)
    Z = F.normalize(bank["Z"].float(), dim=-1).to(device)
    y = bank["y"].long().to(device)
    subj = bank["subj"].long().to(device)
    config = bank["cfg"].long().to(device)
    vocab = list(bank["vocab"])
    n_vocab = len(vocab)
    heldout = torch.tensor(predictor["heldout_labels"], device=device, dtype=torch.long)
    fold = predictor["fold"]
    val_cfg = torch.tensor(fold["validation_config_ids"], device=device)
    val_subject = torch.tensor(fold["validation_subject_ids"], device=device)
    if fold.get("validation_query_policy") != VALIDATION_QUERY_POLICY:
        raise SystemExit("predictor uses an obsolete Phase-B validation-query policy")
    fold_masks = phase_b_fold_masks(config, subj, val_cfg, val_subject)
    base_train = torch.nonzero(fold_masks.train_base, as_tuple=True)[0]
    train_pool = base_train[~torch.isin(y[base_train], heldout)]
    val_pool = torch.nonzero(
        fold_masks.validation & torch.isin(y, heldout), as_tuple=True
    )[0]
    if not len(train_pool) or not len(val_pool):
        raise SystemExit("predictor fold does not provide confidence train/validation queries")
    physical = physical_label_centroids(Z[base_train], y[base_train], n_vocab)

    train_memory_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    train_memory_mask[train_pool] = True
    val_memory_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    val_memory_mask[base_train] = True
    train_index_rows = table.sample_index_rows(
        train_memory_mask, ACTIVE_WINDOWS_PER_LABEL,
        np.random.default_rng(args.seed + 1),
    )
    val_index_rows = table.sample_index_rows(
        val_memory_mask, ACTIVE_WINDOWS_PER_LABEL,
        np.random.default_rng(args.seed + 2),
    )

    def build_index(rows):
        memory = (
            bank["patch"]["Z"][rows].float().to(device)
            if policy.tokenizer_mode == "frozen" or live_source is None else
            live_source.encode_patch_rows(rows, tokenizer, requires_grad=False).to(device)
        )
        memory = F.normalize(memory, dim=-1)
        return memory, retriever.build_index(memory)

    train_selector_z, train_memory_index = build_index(train_index_rows)
    val_selector_z, val_memory_index = build_index(val_index_rows)
    sbert = get_sbert_encoder()
    text = ensemble_text(vocab, sbert, 8, train_only=True).to(device)
    alias_embeddings = encode_neutral_aliases(sbert, device)
    train_vocab = torch.arange(n_vocab, device=device)
    train_vocab = train_vocab[~torch.isin(train_vocab, heldout)]
    episode_cfg = predictor["episode_cfg"]

    @torch.no_grad()
    def draw_features(
        pool, memory_mask, index_rows, selector_z, memory_index,
        allowed_vocab, local_rng,
        *, truth_present: bool, count: int,
    ):
        last_error = None
        for _ in range(20):
            episode_type = str(local_rng.choice(EPISODE_TYPES))
            physical_view_mode = str(local_rng.choice(
                episode_cfg.get("physical_view_modes", PHYSICAL_VIEW_MODES)
            ))
            support_count = 0 if episode_type == "semantic_zero_support" \
                else int(local_rng.integers(SUPPORT_COUNT_RANGE[0], SUPPORT_COUNT_RANGE[1] + 1))
            label_mode = "coherent" if support_count == 0 \
                else str(local_rng.choice(LABEL_TEXT_MODES))
            capacity = support_capacity_by_label(
                bank["patch"], index_rows, n_vocab
            ).to(device)
            query_labels = torch.unique(y[pool])
            query_labels = query_labels[torch.isin(query_labels, allowed_vocab)]
            supportable = torch.nonzero(capacity >= support_count, as_tuple=True)[0]
            supportable = supportable[torch.isin(supportable, allowed_vocab)]
            if len(query_labels) < 1 or len(supportable) < 2:
                continue

            low, high = episode_cfg.get("candidate_count_range", CANDIDATE_COUNT_RANGE)
            candidate_count = min(
                int(local_rng.integers(low, high + 1)), len(supportable)
            )
            if truth_present:
                present = query_labels[capacity[query_labels] >= support_count]
                if len(present) < 2:
                    continue
                candidate_count = min(candidate_count, len(present))
                seed_count = min(2, candidate_count - 1)
                seed_labels = torch.as_tensor(
                    local_rng.choice(present.cpu().numpy(), size=seed_count, replace=False),
                    device=device, dtype=torch.long,
                )
                candidates = choose_candidates(
                    seed_labels, candidate_count, n_vocab, text, physical,
                    truth_present=True, rng=local_rng, allowed_vocab=present,
                )
                qi = sample_queries_covering_labels(
                    pool, candidates, y, count, local_rng,
                    config_ids=config, subject_ids=subj,
                )
            else:
                seed_count = min(4, len(query_labels))
                seed_labels = torch.as_tensor(
                    local_rng.choice(
                        query_labels.cpu().numpy(), size=seed_count, replace=False
                    ),
                    device=device, dtype=torch.long,
                )
                qi = sample_queries(
                    pool, seed_labels, y, count, local_rng,
                    config_ids=config, subject_ids=subj, label_alpha=0.0,
                    subject_alpha=float(episode_cfg.get("query_subject_alpha", 0.5)),
                )
                eligible_candidates = supportable[~torch.isin(supportable, torch.unique(y[qi]))]
                if len(eligible_candidates) < 2:
                    continue
                candidate_count = min(candidate_count, len(eligible_candidates))
                candidates = choose_candidates(
                    y[qi], candidate_count, n_vocab, text, physical,
                    truth_present=False, rng=local_rng,
                    allowed_vocab=eligible_candidates,
                )
            query = table.gather_queries(
                qi, device, expand_verified_events=True,
                allowed_window_mask=memory_mask,
            )
            try:
                view = build_episode_memory_view(
                    bank["patch"], index_rows, query, y[qi], candidates,
                    support_count=support_count, episode_type=episode_type,
                    label_mode=label_mode, rng=local_rng,
                    truth_present=truth_present,
                )
                labels = episode_label_set(
                    candidates, text, mode=label_mode, rng=local_rng,
                    alias_embeddings=alias_embeddings,
                    canonical_names=vocab,
                )
                logits, aux = decode_adaptation_episode(
                    decoder, retriever, bank, index_rows, selector_z,
                    memory_index, query, view, text, labels.embeddings,
                    policy=policy,
                    rng=local_rng,
                    live_source=live_source, selector_encoder=tokenizer,
                    online_requires_grad=False,
                    physical_view_mode=physical_view_mode,
                    return_confidence_features=True,
                )
            except ValueError as error:
                last_error = error
                continue
            prediction = candidates[logits.argmax(1)]
            target = prediction.eq(y[qi]) if truth_present else torch.zeros_like(
                prediction, dtype=torch.bool
            )
            return aux["confidence_features"].detach(), target.float(), {
                "truth_present": truth_present,
                "correct": float(target.float().mean()),
                "candidate_count": len(candidates),
                "true_support": support_count,
                "episode_type": episode_type,
                "label_mode": label_mode,
                "physical_view_mode": physical_view_mode,
            }
        raise RuntimeError("could not draw a feasible confidence episode") from last_error

    val_features, val_targets = [], []
    val_rng = np.random.default_rng(args.seed + 3)
    for index in range(args.val_episodes):
        features, targets, _ = draw_features(
            val_pool, val_memory_mask, val_index_rows, val_selector_z,
            val_memory_index,
            torch.arange(n_vocab, device=device), val_rng,
            truth_present=index % 2 == 0, count=args.val_queries,
        )
        val_features.append(features); val_targets.append(targets)
    val_features = torch.cat(val_features)
    val_targets = torch.cat(val_targets)
    if not bool(val_targets.bool().any()) or bool(val_targets.bool().all()):
        raise SystemExit(
            "confidence validation has only one target class; the predictor/calibration fold "
            "cannot support a meaningful reject-confidence fit"
        )

    head = EvidenceConfidenceHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    @torch.no_grad()
    def evaluate():
        head.eval()
        logit = head(val_features)
        loss = head.loss(logit, val_targets)
        score = torch.sigmoid(logit).cpu().numpy()
        target = val_targets.cpu().numpy().astype(bool)
        return {
            "bce": float(loss),
            "auroc": binary_auroc(score, target),
            "auprc": binary_auprc(score, target),
            "aurc": aurc(1.0 - score, target),
            "ece": expected_calibration_error(score, target),
            "brier": float(np.mean((score - target.astype(np.float32)) ** 2)),
            "positive_rate": float(target.mean()),
            "mean_confidence": float(score.mean()),
            "mean_confidence_positive": float(score[target].mean()),
            "mean_confidence_negative": float(score[~target].mean()),
        }

    best = {"bce": float("inf")}; best_state = None; best_step = 0
    started = time.time()
    predictor_fp = hashlib.sha256(args.predictor.read_bytes()).hexdigest()
    state_path = args.out.with_name(f"{args.out.stem}.last{args.out.suffix}")
    run_config = {
        "steps": args.steps, "batch": args.batch, "lr": args.lr,
        "weight_decay": args.weight_decay,
        "truth_absent_probability": args.truth_absent_prob,
        "val_episodes": args.val_episodes, "val_queries": args.val_queries,
        "seed": args.seed,
    }

    def save_state(step: int) -> None:
        atomic_torch_save({
            "kind": "phase_b_confidence_trainer_state_v1",
            "step": step,
            "elapsed_seconds": time.time() - started,
            "run_config": run_config,
            "bank_fp": bank.get("bank_fp") or bank_fingerprint(bank),
            "predictor_fp": predictor_fp,
            "confidence": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best": best, "best_state": best_state, "best_step": best_step,
            "rng": {"numpy_generator": rng.bit_generator.state,
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
        }, state_path)

    start_step = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("kind") != "phase_b_confidence_trainer_state_v1":
            raise SystemExit("--resume is not a Phase-B confidence trainer state")
        if state.get("bank_fp") != (bank.get("bank_fp") or bank_fingerprint(bank)) \
                or state.get("predictor_fp") != predictor_fp:
            raise SystemExit("confidence resume state does not match the predictor and bank")
        if state.get("run_config") != run_config:
            raise SystemExit("confidence resume configuration differs from the saved run")
        start_step = int(state["step"])
        if start_step >= args.steps:
            raise SystemExit("confidence resume state has already reached --steps")
        head.load_state_dict(state["confidence"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best = dict(state["best"]); best_state = state["best_state"]
        best_step = int(state["best_step"])
        rng.bit_generator.state = state["rng"]["numpy_generator"]
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if device.type == "cuda" and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])
        started = time.time() - float(state.get("elapsed_seconds", 0.0))
        print(f"[confidence] resumed {args.resume} at step {start_step}", flush=True)

    telemetry = PhaseBTelemetry(
        args.telemetry_dir or (args.out.parent / "telemetry" / args.out.stem),
        interval_seconds=args.telemetry_seconds,
        stage="confidence",
    )
    telemetry.start(
        step=start_step,
        elapsed_seconds=time.time() - started,
        metadata={
            "planned_steps": args.steps,
            "warmup_steps": 0,
            "grad_clip": 1.0,
            "output": str(args.out),
            "predictor": str(args.predictor),
            "predictor_fp": predictor_fp,
            "bank_fp": bank.get("bank_fp") or bank_fingerprint(bank),
            "resume": str(args.resume) if args.resume is not None else None,
            "run_config": run_config,
        },
    )

    for step in range(start_step + 1, args.steps + 1):
        step_started = time.perf_counter()
        truth_present = bool(rng.random() >= args.truth_absent_prob)
        features, target, episode = draw_features(
            train_pool, train_memory_mask, train_index_rows, train_selector_z,
            train_memory_index,
            train_vocab, rng, truth_present=truth_present, count=args.batch,
        )
        head.train()
        logit = head(features)
        loss = head.loss(logit, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite confidence loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            head.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step(); scheduler.step()
        if step == 1 or step % args.val_every == 0:
            metrics = evaluate()
            telemetry.set_validation(metrics)
            if metrics["bce"] < best["bce"]:
                best = dict(metrics); best_step = step
                best_state = {
                    key: value.detach().cpu().clone() for key, value in head.state_dict().items()
                }
            print(json.dumps({
                "step": step, "loss": round(float(loss.detach()), 5),
                "grad_norm": round(float(grad_norm), 4), **episode,
                **{key: round(value, 4) for key, value in metrics.items()},
                "elapsed_s": round(time.time() - started, 1),
            }), flush=True)
        with torch.no_grad():
            score = torch.sigmoid(logit)
            prediction = score >= 0.5
            telemetry_metrics = {
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "gradient_clipped_fraction": float(float(grad_norm) > 1.0),
                "target_positive_rate": float(target.mean()),
                "predicted_confidence_mean": float(score.mean()),
                "predicted_confidence_std": float(score.float().std(unbiased=False)),
                "predicted_confidence_min": float(score.min()),
                "predicted_confidence_max": float(score.max()),
                "threshold_accuracy": float(prediction.eq(target.bool()).float().mean()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "step_seconds": float(time.perf_counter() - step_started),
            }
            if device.type == "cuda":
                telemetry_metrics.update({
                    "gpu_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                    "gpu_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                    "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                })
        category_values = {
            "truth_present": str(bool(episode["truth_present"])),
            "episode_type": episode["episode_type"],
            "label_mode": episode["label_mode"],
            "physical_view_mode": episode["physical_view_mode"],
            "support_count": str(episode["true_support"]),
            "candidate_count": str(episode["candidate_count"]),
        }
        telemetry.update(
            telemetry_metrics,
            categories=category_values,
            strata=category_values,
        )
        emitted = telemetry.emit(step=step, elapsed_seconds=time.time() - started)
        if emitted is not None:
            print(json.dumps({
                "telemetry": str(telemetry.latest),
                "step": step,
                "window_seconds": round(emitted["window_seconds"], 2),
            }), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_state(step)

    if best_state is None:
        raise RuntimeError("confidence calibration completed without a valid checkpoint")
    payload = {
        "confidence": best_state,
        "objective": "correct_and_answerable_bce",
        "predictor": str(args.predictor), "predictor_fp": predictor_fp,
        "bank_fp": bank.get("bank_fp") or bank_fingerprint(bank),
        "memory_schema": int(bank["schema_version"]), "vocab": vocab,
        "truth_absent_probability": args.truth_absent_prob,
        "best_step": best_step, "best_metrics": best,
    }
    atomic_torch_save(payload, args.out)
    telemetry.emit(
        step=args.steps, elapsed_seconds=time.time() - started, force=True, final=True
    )
    print(f"[confidence] best step {best_step}: {best} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
