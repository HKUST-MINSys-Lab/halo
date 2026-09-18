"""Bounded profile of the current support-classifier training recipe.

This profiles the deployment-shaped residual classifier or differentiable-neighbour control.
Optional validation only uses the training roster's held-out subjects; no checkpoints or sealed
results are written. Worker-count variants run sequentially.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from data.scripts.curate.deployment_policy import SUPERVISED_HEAD_TRAIN_DATASETS
from training.support_classifier.collate import SupportCollate
from training.support_classifier.corpus import support_corpus_from_index
from training.support_classifier.encoding import (
    autocast,
    build_random_encoder,
    install_compiled_transformer,
)
from training.support_classifier.neighbors import differentiable_neighbor_logits
from model.blocks import AttentionSpec
from model.support.residual_classifier import ResidualClassifierConfig, ResidualSupportClassifier
from training.support_classifier.train import (
    TAU_SUPPORT,
    PrefetchLoader,
    build_dataset,
    calibrate_frontend,
    episode_loss,
    episode_text,
    encode_recording_rows,
    make_label_text,
    make_optimizer,
    initialise_text_projection,
    split_encoded,
    validate,
    draw_kwargs_from_args,
)
from training.support_classifier.sampling import (
    DEFAULT_ACQUISITION_MIX, DEFAULT_ENROLLMENT_K, DEFAULT_ENROLLMENT_MIX,
    DEFAULT_LABEL_SUBSET, DEFAULT_PARTIAL_COVERAGE, DEFAULT_P_GT_PRESENT,
    DEFAULT_SAME_SUBJECT_PROBABILITY, DEFAULT_VARIABLE_SUPPORT_PROBABILITY,
)
from training.tokenizer.pretrain_data import CorpusIndex, MultiResolutionCollate, PretrainDataset


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _grad_norm(module: torch.nn.Module) -> float:
    pieces = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(pieces).sum().sqrt()) if pieces else 0.0


def _gradient_breakdown(encoder, classifier) -> dict[str, float]:
    """Pre-clip block norms for diagnosing scale without changing the training graph."""
    modules = {
        "encoder": encoder,
        **({"encoder/filterbank": encoder.filterbank}
           if getattr(encoder, "filterbank", None) is not None else {}),
        **({"encoder/sensor_fold": encoder.sensor_fold}
           if getattr(encoder, "sensor_fold", None) is not None else {}),
        **({"encoder/transformer": encoder.transformer}
           if getattr(encoder, "transformer", None) is not None else {}),
        **({"encoder/recording_pool": encoder.recording_pool}
           if getattr(encoder, "recording_pool", None) is not None else {}),
        **({"encoder/duration_proj": encoder.duration_proj}
           if getattr(encoder, "duration_proj", None) is not None else {}),
        **({"encoder/text_conditioner": encoder.descriptor_proj}
           if getattr(encoder, "descriptor_proj", None) is not None else {}),
        **({"encoder/structured_conditioner": encoder.structured_conditioner}
           if getattr(encoder, "structured_conditioner", None) is not None else {}),
    }
    if classifier is not None:
        modules.update({
            "classifier": classifier,
            "classifier/signal_proj": classifier.signal_proj,
            "classifier/attention": classifier.metric_stack,
            "classifier/support_residual": classifier.r_support_head,
            "classifier/candidate_residual": classifier.r_candidate_head,
            "classifier/text_bridge": classifier.p_text,
        })
    return {name: _grad_norm(module) for name, module in modules.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--classifier", choices=("residual", "neighbors"), default="residual")
    parser.add_argument("--support-temperature", type=float, default=TAU_SUPPORT,
                        help="diagnostic support-vote temperature")
    parser.add_argument("--text-temperature", type=float, default=0.07,
                        help="diagnostic query-to-label temperature")
    parser.add_argument("--support-sets", type=int, default=4,
                        help="independent support sets per optimizer step")
    parser.add_argument("--queries-per-support-set", type=int, default=4)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--resolutions", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--trace", action="store_true", help="profile one additional step")
    parser.add_argument("--validation-support-sets", type=int, default=0,
                        help="optional bounded timing of subject-held-out validation (0 disables)")
    parser.add_argument("--multi-device-probability", type=float, default=0.5)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--max-devices", type=int, default=4)
    parser.add_argument("--polarization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-transformer", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--encoder-arch", default="halo",
                        choices=("halo", "limubert", "harnet", "unimts"),
                        help="profile a matched-corpus M2 arm instead of HALO's encoder")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.support_sets < 1 or min(args.workers) < 1 \
            or not 0 <= args.warmup < args.steps <= 100:
        parser.error("positive worker counts and 0 <= warmup < steps <= 100 required")
    if not 0 <= args.validation_support_sets <= 64:
        parser.error("validation timing is bounded to 0..64 support sets")
    if args.support_temperature <= 0 or args.text_temperature <= 0:
        parser.error("diagnostic temperatures must be positive")

    torch.set_num_threads(2)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    setup_started = time.perf_counter()
    index = CorpusIndex(
        datasets=SUPERVISED_HEAD_TRAIN_DATASETS, alignment="native",
        max_per_stream=None, seed=20260901, window_seconds=args.window_seconds,
    )
    corpus = support_corpus_from_index(index)
    dataset_args = argparse.Namespace(
        neutral_acquisition_text=False,
        multi_device_probability=args.multi_device_probability,
        max_devices=args.max_devices,
        rate_augmentation_probability=0.0,
        modality_dropout_probability=0.0,
    )
    dataset = build_dataset(index, dataset_args)
    collate = SupportCollate(MultiResolutionCollate(
        fixed_patch_seconds=tuple(args.resolutions),
    ))
    recipe_args = argparse.Namespace(
        classifier=args.classifier,
        p_gt_present=(DEFAULT_P_GT_PRESENT if args.classifier == "residual" else 1.0),
        p_mask_candidate=0.25,
        p_mask_gt=0.10,
        same_subject_probability=DEFAULT_SAME_SUBJECT_PROBABILITY,
        label_subset=DEFAULT_LABEL_SUBSET,
        mode="compatible",
        enrollment_k=DEFAULT_ENROLLMENT_K,
        acquisition_mix=DEFAULT_ACQUISITION_MIX,
        enrollment_mix=(DEFAULT_ENROLLMENT_MIX if args.classifier == "residual"
                        else (2.0 / 3.0, 1.0 / 3.0, 0.0)),
        partial_coverage=DEFAULT_PARTIAL_COVERAGE,
        variable_support_probability=DEFAULT_VARIABLE_SUPPORT_PROBABILITY,
        queries_per_support_set=args.queries_per_support_set,
        windows_per_execution=2,
    )
    draw_kwargs = draw_kwargs_from_args(recipe_args)

    corpus_setup_seconds = time.perf_counter() - setup_started
    print(f"[profile] full training corpus ready in {corpus_setup_seconds:.1f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for workers in args.workers:
        # Fork before CUDA/text initialization. Only this variant's workers exist during timing.
        loader = PrefetchLoader(
            corpus, dataset, collate.bucketed, data_seed=20260901,
            batch_size=args.support_sets, draw_kwargs=draw_kwargs, workers=workers,
        )
        try:
            torch.manual_seed(7)
            if args.encoder_arch != "halo":
                from model.tokenizer.matched_encoder import build_matched_encoder

                encoder = build_matched_encoder(args.encoder_arch, device=device).train()
            else:
                encoder, _ = build_random_encoder(
                    device, "fixed", neutral_acquisition_text=False,
                    duration_range=(min(args.resolutions), max(args.resolutions)),
                    num_resolutions=len(args.resolutions),
                    frontend_kwargs={
                        "use_polarization": args.polarization,
                        "polarization_energy_kappa": 0.05,
                    },
                )
            encoder.train()
            if hasattr(encoder, "mask_token"):
                encoder.mask_token.requires_grad_(False)
            if args.compile_transformer and args.encoder_arch == "halo":
                install_compiled_transformer(encoder)
            if getattr(encoder, "filterbank", None) is not None:
                calibrate_frontend(
                    encoder, dataset, corpus, collate, np.random.default_rng(7), device,
                    batches=1, batch_size=64, executor=None,
                )
            text = make_label_text(corpus.all_labels, device)
            classifier = (ResidualSupportClassifier(
                AttentionSpec(d_model=encoder.d_model, n_heads=4, ffn_mult=2, dropout=0.1),
                ResidualClassifierConfig(
                    temperature=args.support_temperature,
                    text_temperature=args.text_temperature,
                ),
            ).to(device).train() if args.classifier == "residual" else None)
            if classifier is not None:
                initialise_text_projection(
                    classifier, encoder, dataset, corpus, collate, text,
                    np.random.default_rng(7), device, batches=1, batch_size=64, executor=None,
                )
            parameters = [p for p in encoder.parameters() if p.requires_grad]
            if classifier is not None:
                parameters += list(classifier.parameters())
            optimizer = make_optimizer(
                [{"name": "encoder", "params": parameters, "lr": 3e-4}],
                weight_decay=.05, device=device,
            )
            timings = {name: [] for name in (
                "step", "loader_wait", "encode", "recording_pool", "episode_assembly",
                "neighbors_loss", "backward", "clip", "optimizer",
            )}
            digest = hashlib.sha256()
            shape_samples = []
            torch.cuda.reset_peak_memory_stats()
            trace = None
            preclip_gradient_norms = {}
            for step in range(1, args.steps + 1):
                if args.trace and step == args.steps:
                    trace = torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
                    ])
                    trace.__enter__()
                torch.cuda.synchronize()
                started = time.perf_counter()
                loaded = loader.get(step)
                # Current loaders attach per-episode device-set plans. The profile does not
                # aggregate scenario telemetry, but it must consume the same prepared batches
                # as training so its timing remains representative.
                episodes, _, batch = loaded[:3]
                loader_done = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)

                marks = [_event() for _ in range(8)]
                marks[0].record()
                with autocast(device):
                    pooled, descriptor, _ = encode_recording_rows(encoder, batch, device)
                    marks[1].record()
                    marks[2].record()
                    rows = split_encoded(pooled, descriptor, episodes, corpus)
                    episode_vectors = episode_text(episodes, corpus, text, device)
                    marks[3].record()
                    query = rows["query_feature"].squeeze(1)
                    if classifier is None:
                        logits, weights = differentiable_neighbor_logits(
                            query, rows["support_feature"], episode_vectors["support_bound"],
                            rows["support_mask"], episode_vectors["candidate_mask"],
                            temperature=args.support_temperature,
                        )
                    else:
                        output = classifier(query_feature=query,
                            support_feature=rows["support_feature"], support_mask=rows["support_mask"],
                            **episode_vectors)
                        logits, weights = output["logits"], output["support_weight"]
                    loss = episode_loss(logits, episodes, episode_vectors)["loss"]
                    marks[4].record()
                loss.backward()
                marks[5].record()
                if step == args.steps:
                    preclip_gradient_norms = _gradient_breakdown(encoder, classifier)
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                marks[6].record()
                optimizer.step()
                marks[7].record()
                torch.cuda.synchronize()
                finished = time.perf_counter()
                if trace is not None:
                    trace.__exit__(None, None, None)
                    trace.export_chrome_trace(str(args.out.with_suffix(f".w{workers}.trace.json")))
                    args.out.with_suffix(f".w{workers}.operators.txt").write_text(
                        trace.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))

                if step > args.warmup and trace is None:
                    timings["step"].append(1000 * (finished - started))
                    timings["loader_wait"].append(1000 * (loader_done - started))
                    for name, left, right in (
                        ("encode", 0, 1), ("recording_pool", 1, 2),
                        ("episode_assembly", 2, 3), ("neighbors_loss", 3, 4),
                        ("backward", 4, 5), ("clip", 5, 6), ("optimizer", 6, 7),
                    ):
                        timings[name].append(marks[left].elapsed_time(marks[right]))
                    dense_batches = batch.batches
                    shape_samples.append({
                        "recordings": int(batch.row_count),
                        "patches": max(int(part["patch_len"].shape[1]) for part in dense_batches),
                        "channels": max(int(part["compact_data"].shape[-1]) for part in dense_batches),
                        "queries": len(episodes),
                        "max_candidates": int(episode_vectors["candidate_mask"].shape[1]),
                        "zero_support_query_fraction": sum(e.is_zero_shot for e in episodes) / len(episodes),
                        "support_padding_fraction": (1 - float(rows["support_mask"].float().mean())
                                                     if rows["support_mask"].numel() else 0.0),
                        "token_padding_fraction": 1 - float((1 + 2 * rows["support_mask"].sum()
                            / len(episodes) + episode_vectors["candidate_mask"].sum() / len(episodes))
                            / (1 + 2 * rows["support_mask"].shape[1] + episode_vectors["candidate_mask"].shape[1])),
                        "max_support_rows": int(rows["support_mask"].shape[1]),
                        "valid_support_rows": int(rows["support_mask"].sum()),
                        "host_batch_mib": sum(
                            value.numel() * value.element_size()
                            for part in dense_batches for value in part.values()
                            if isinstance(value, torch.Tensor)
                        ) / 2**20,
                    })
                digest.update(json.dumps(
                    [dataclasses.asdict(episode) for episode in episodes], sort_keys=True,
                ).encode())

            record = {
                "classifier": args.classifier,
                "support_temperature": args.support_temperature,
                "text_temperature": args.text_temperature,
                "corpus_setup_seconds": corpus_setup_seconds,
                "draw_kwargs": draw_kwargs,
                "encoder_params": sum(p.numel() for p in encoder.parameters()),
                "classifier_params": sum(p.numel() for p in classifier.parameters()) if classifier else 0,
                "workers": workers,
                "support_sets": args.support_sets,
                "queries_per_support_set": args.queries_per_support_set,
                "resolutions": args.resolutions,
                "multi_device_probability": args.multi_device_probability,
                "window_seconds": args.window_seconds,
                "polarization": args.polarization,
                "compile_transformer": args.compile_transformer,
                "timing_notes": {
                    "encode": "Includes recording pooling, not just patch encoding.",
                    "recording_pool": "Empty event boundary; pooling is included in encode.",
                    "neighbors_loss": "Selected classifier forward plus episode loss.",
                    "compile_transformer": "Requested only; compilation errors are fatal rather than silently eager.",
                    "optimizer_only_minutes_35k": "Includes loader wait; excludes calibration, validation, checkpointing and telemetry.",
                    "gradient_norms": "Final step after clipping; encoder includes recording_pool.",
                    "preclip_gradient_norms": "Final step before clipping; nested blocks overlap their parent totals.",
                },
                **{f"mean_{name}_ms": _mean(values) for name, values in timings.items()},
                "median_step_ms": float(np.median(timings["step"])),
                "p90_step_ms": float(np.percentile(timings["step"], 90)),
                "optimizer_only_minutes_35k": _mean(timings["step"]) * 35000 / 60000,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "episode_sha256": digest.hexdigest(),
                "mean_shape": {
                    key: _mean([sample[key] for sample in shape_samples])
                    for key in shape_samples[0]
                },
                "loss": float(loss.detach()),
                "gradient_norms": {
                    name: float(torch.stack([p.grad.float().norm().square() for p in module.parameters()
                                             if p.grad is not None]).sum().sqrt())
                    for name, module in {
                        "encoder": encoder,
                        **({"recording_pool": encoder.recording_pool}
                           if getattr(encoder, "recording_pool", None) is not None else {}),
                        **({"classifier": classifier, "trunk": classifier.metric_stack,
                            "support_residual": classifier.r_support_head,
                            "candidate_residual": classifier.r_candidate_head,
                            "text_bridge": classifier.p_text} if classifier else {}),
                    }.items()
                    if any(p.grad is not None for p in module.parameters())
                },
                "preclip_gradient_norms": preclip_gradient_norms,
                "score_magnitudes": ({
                    name: {
                        "mean_abs": float(output[name][episode_vectors["candidate_mask"]]
                                          .detach().float().abs().mean()),
                        "std": float(output[name][episode_vectors["candidate_mask"]]
                                     .detach().float().std()),
                        "max_abs": float(output[name][episode_vectors["candidate_mask"]]
                                         .detach().float().abs().max()),
                    }
                    for name in (
                        "logits", "metric_part", "text_part", "r_candidate", "text_score",
                    )
                } if classifier is not None else {}),
                "timing_samples_ms": timings,
                "shape_samples": shape_samples,
            }
            if args.validation_support_sets:
                val_corpus = support_corpus_from_index(index, split="val")
                val_dataset = PretrainDataset(
                    index, index.val, augment=False, two_view=False,
                    neutral_acquisition_text=False,
                    multi_device_probability=args.multi_device_probability,
                    max_devices=args.max_devices,
                )
                torch.cuda.synchronize()
                validation_start = time.perf_counter()
                with torch.no_grad():
                    validate(
                        encoder=encoder, classifier=classifier, classifier_mode=args.classifier,
                        corpus=val_corpus, dataset=val_dataset, collate=collate, text_of=text,
                        device=device, episodes_count=args.validation_support_sets,
                        episodes_per_step=args.support_sets, seed=20260901,
                        draw_kwargs=draw_kwargs, executor=None, deployment_matched=True,
                    )
                torch.cuda.synchronize()
                record["validation_support_sets"] = args.validation_support_sets
                record["validation_seconds"] = time.perf_counter() - validation_start
            print(json.dumps({k:v for k,v in record.items() if k not in ("timing_samples_ms", "shape_samples")}), flush=True)
            results.append(record)
            del encoder, optimizer, parameters, pooled, descriptor, rows, loss
            torch.cuda.empty_cache()
        finally:
            loader.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
