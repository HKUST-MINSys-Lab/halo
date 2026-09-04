"""HALO comparison model, scored on the shared adaptation manifest.

Two faces from one checkpoint, exactly as the design specifies:

* ``window_features`` — the encoder's pooled row per evaluation window, consumed by the shared
  nearest / prototype / ridge / linear-head readouts. Apples-to-apples with every baseline's
  frozen-feature row.
* **native enrollment** — the deployed mechanism. The manifest's support rows *are* the support
  set: they are already execution-disjoint from the query and drawn from the same stream, hence
  config-compatible by construction. There is **no corpus bank** at k >= 1; that stage was removed
  from the model, not hidden here.

THE ZERO-SHOT ROW
-----------------
At k = 0 there is no support, so the deployed mechanism has nothing to compare against. Per the
2026-09-03 decision the k = 0 row keeps the same mechanism and draws its support from the
*training* corpus: recordings that are config-compatible with the query and whose labels exclude
every candidate. The model still cannot find the answer among them; it can only note which
training recordings the query resembles and let their label text bridge to the candidates.

**It is an ensemble of small, training-shaped comparisons, not one large one** (revised 2026-09-04).
A single 64-row draw spanning many labels is off-distribution: the sampler trains on episodes of
2-14 candidate labels with K = 32, so a wide flat support set asks the comparator a question it has
never been asked. Instead the row is scored R times, each draw being a handful of seen labels with
their example recordings, shaped exactly like a training episode; the per-draw candidate scores are
then combined. Two things improve at once — the comparison matches the training distribution, and
averaging over draws removes the support draw as a nuisance variable. That variable is known to
matter here: on this codebase the *draw* once contributed five times the run-to-run scatter of the
thing being measured.

Ensembling is also the intervention with the best track record on this project: averaging eight
label paraphrases was the single largest Phase-B gain (45.3 -> 47.5 macro-F1), and it has no learned
parameters, which is the category of change that has historically survived its controls here.

Six streams in the active ``adaptation_v2`` manifest have no compatible training partner:
MotionSense's front-pocket stream, USC-HAD's hip stream, and all four Upper Limb Use research-IMU
wrist streams. For those the k = 0 row is unsupported. The broader curation audit also lists two
unsupported Shoaib streams, but the manifest skips them. The evaluation runner must record an
unsupported cell rather than silently substitute a different mechanism or pad with incompatible
rows.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch.nn.functional as F

from baselines.base import BaselineAdapter, InputContract, UnsupportedEvaluationCell, register

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

_CKPT = Path(os.environ.get(
    "HALO_COMPARE_CKPT", _REPO / "training/compare/outputs/arm_a/last.pt",
))
#: Arm A is the headline configuration: the encoder is never told the acquisition configuration.
_NEUTRAL_ACQUISITION_TEXT = os.environ.get("HALO_NEUTRAL_ACQUISITION_TEXT", "1") not in {"0", ""}
# The k = 0 row is an ENSEMBLE of training-shaped comparisons (see the module docstring). These
# defaults reproduce the training episode shape exactly — 4 labels x 8 executions = K 32, inside
# the sampler's 2-14 label range — so the deployed mechanism is asked the kind of question it trained
# on. R = 8 follows the text-ensemble precedent, where averaging 8 paraphrases was worth +2.2 macro-F1
# and was the single largest Phase-B gain.
ZERO_SHOT_DRAWS = int(os.environ.get("HALO_COMPARE_ZERO_SHOT_DRAWS", "8"))
ZERO_SHOT_LABELS_PER_DRAW = int(os.environ.get("HALO_COMPARE_ZERO_SHOT_LABELS", "4"))
ZERO_SHOT_ROWS_PER_LABEL = int(os.environ.get("HALO_COMPARE_ZERO_SHOT_ROWS_PER_LABEL", "8"))
#: probability | standardized | logprob — see :func:`_ensemble`.
ZERO_SHOT_ENSEMBLE = os.environ.get("HALO_COMPARE_ZERO_SHOT_ENSEMBLE", "probability")
ZERO_SHOT_SEED = 20260901
QUERY_CHUNK = 1024


def _ckpt_sha256() -> str:
    return hashlib.sha256(_CKPT.read_bytes()).hexdigest() if _CKPT.exists() else ""


def _six_slot(windows, channels, mask):
    """Scatter a grid's native channels into the encoder's canonical 6-slot pad+mask layout.

    Accelerometer-only sources store 3 channels — monipar among the evaluation streams, capture24
    among the training pools — while the encoder's contract is the fixed 6 slots with a validity
    mask, which is what ``PretrainDataset`` builds for training. Passing a native 3-wide mask
    straight through raises inside ``encode_dataset_detailed``; the pre-existing ``halo_compact``
    adapter does exactly that and shares this defect on monipar's native alignment.

    Masked slots are written as exact zeros, never as fabricated channels.
    """
    from training.tokenizer.pretrain_data import CHANNELS

    native = np.asarray(windows, dtype=np.float32)
    native_mask = np.asarray(mask, dtype=bool)
    if native.shape[-1] == len(CHANNELS) and native_mask.shape[0] == len(CHANNELS):
        return native, native_mask

    slot = {name: index for index, name in enumerate(CHANNELS)}
    out = np.zeros((native.shape[0], native.shape[1], len(CHANNELS)), dtype=np.float32)
    out_mask = np.zeros(len(CHANNELS), dtype=bool)
    for position, name in enumerate(channels):
        if name not in slot:
            continue
        out[:, :, slot[name]] = native[:, :, position]
        out_mask[slot[name]] = bool(native_mask[position])
    out[:, :, ~out_mask] = 0.0
    return out, out_mask


def _ensemble(logits, mode: str):
    """Combine per-draw candidate scores. ``logits`` is (draws, queries, candidates).

    * ``probability`` — mean of each draw's softmax. Scale-free per draw, so a draw that happens to
      produce large logits cannot dominate. The default, and the conservative choice.
    * ``standardized`` — z-score each draw's logits across candidates, then average. The "calibrated"
      variant: it keeps relative margins rather than flattening them through a softmax, while still
      removing per-draw scale.
    * ``logprob`` — mean of log-softmax, i.e. a geometric mean over draws (product of experts).
      Sharper, but one confident-and-wrong draw can veto the rest.
    """
    import torch as T

    if mode == "probability":
        return T.softmax(logits.float(), dim=-1).mean(dim=0)
    if mode == "logprob":
        return T.log_softmax(logits.float(), dim=-1).mean(dim=0)
    if mode == "standardized":
        values = logits.float()
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return ((values - mean) / std).mean(dim=0)
    raise ValueError(f"unknown zero-shot ensemble mode {mode!r}")


@register
class HALOCompareAdapter(BaselineAdapter):
    """Recognise by comparing the query to the episode's own labelled support recordings."""

    name = "halo_compare"
    tier = "evidence"
    contract = InputContract()          # native rate and channels; the tokenizer is the contract

    # ------------------------------------------------------------------ setup
    def setup(self, device):
        import torch as T
        from eval.scoring import get_sbert_encoder
        from model.blocks import AttentionSpec
        from model.evidence.comparator import ComparatorConfig, SupportComparator
        from training.tokenizer.eval_transfer import build_encoder

        if not _CKPT.exists():
            raise FileNotFoundError(
                f"HALO compare checkpoint missing at {_CKPT}. Set HALO_COMPARE_CKPT or run "
                "training.compare.train."
            )
        blob = T.load(str(_CKPT), map_location="cpu", weights_only=False)
        device = T.device(device)
        encoder = build_encoder(blob, device, training=False).eval()

        spec = AttentionSpec(**blob["attention_spec"])
        comparator = SupportComparator(spec, ComparatorConfig(**blob["comparator_config"]))
        comparator.load_state_dict(blob["comparator"])
        comparator = comparator.to(device).eval()

        trained_neutral = bool(blob.get("args", {}).get("neutral_acquisition_text", False))
        if trained_neutral != _NEUTRAL_ACQUISITION_TEXT:
            raise ValueError(
                f"{_CKPT.name} was trained with neutral_acquisition_text={trained_neutral} but is "
                f"being evaluated with {_NEUTRAL_ACQUISITION_TEXT}. Arm A's claim is that the "
                "encoder is never told the acquisition configuration; evaluating the other way "
                "round would report a different model than the one that was trained."
            )
        # Evaluation must centre exactly as training did, or the deployed function is not the one
        # that was trained. Older checkpoints predate the flag and were trained uncentred.
        center = bool(blob.get("config", {}).get("center_features",
                                                 blob.get("args", {}).get("center_features", False)))
        return {
            "center": center,
            "encoder": encoder, "comparator": comparator, "device": device,
            "sbert": get_sbert_encoder(), "streams": {}, "feature_owner": {},
            "checkpoint_step": int(blob.get("step", 0)),
            "neutral_acquisition_text": _NEUTRAL_ACQUISITION_TEXT,
            "zero_shot_support": {},
        }

    # ------------------------------------------------------- stream encoding
    def _stream_rows(self, stream, state):
        """Pooled feature and acquisition descriptor for every window of one eval stream."""
        from training.tokenizer.eval_transfer import encode_dataset_detailed
        from training.tokenizer.pretrain_data import (
            _stream_gravity_state,
            stream_channel_descriptions,
            stream_sensor_texts,
        )

        key = (stream.dataset, stream.stream)
        if key in state["streams"]:
            return state["streams"][key]
        device, encoder = state["device"], state["encoder"]

        texts = (
            stream_channel_descriptions(stream.dataset, stream.stream, neutral=True)
            if _NEUTRAL_ACQUISITION_TEXT else
            (stream.channel_descriptions
             if getattr(stream, "channel_descriptions", None) is not None
             else stream_channel_descriptions(stream.dataset, stream.stream))
        )
        gravity_state = (stream.gravity_state
                         if getattr(stream, "gravity_state", None) is not None
                         else _stream_gravity_state(stream.dataset, stream.stream))
        from training.tokenizer.pretrain_data import CHANNELS

        windows, mask = _six_slot(stream.windows, stream.channels, stream.mask)
        detailed = encode_dataset_detailed(
            encoder, windows, texts, device, stream.rate_hz,
            gravity_state=gravity_state, channel_mask=mask,
            dataset=stream.dataset, stream=stream.stream,
            neutral_text=_NEUTRAL_ACQUISITION_TEXT, export_sensor_rows=False,
        )
        feature = detailed["pooled"].to(device)

        _, sensor_texts, _ = stream_sensor_texts(
            stream.dataset, stream.stream,
            gravity_removed=(gravity_state == "removed"),
            has_accel=any(c.startswith("acc") for c, m in zip(CHANNELS, mask) if m),
            has_gyro=any(c.startswith("gyro") for c, m in zip(CHANNELS, mask) if m),
            neutral=_NEUTRAL_ACQUISITION_TEXT,
        )
        descriptors = encoder.text_encoder.encode_pooled(sensor_texts, device=device)
        descriptor = F.normalize(
            F.normalize(descriptors.float(), dim=-1).mean(dim=0, keepdim=True), dim=-1,
        ).expand(len(feature), -1)

        payload = (feature, descriptor, detailed["pooled"].float().cpu().numpy())
        state["streams"][key] = payload
        return payload

    # ------------------------------------------------ standard adaptation face
    def window_features(self, stream, state, device) -> np.ndarray:
        _, _, pooled = self._stream_rows(stream, state)
        state["feature_owner"][id(pooled)] = (stream.dataset, stream.stream)
        return pooled

    def supports_native_enrollment(self) -> bool:
        return True

    def supports_native_zero_shot(self) -> bool:
        return True

    # ---------------------------------------------------------- the mechanism
    def _score(self, state, query_feature, query_descriptor,
               support_feature, support_descriptor, support_label_text, support_bound,
               candidate_text):
        """Run the comparator over one padded episode and return per-query candidate logits."""
        import torch as T
        from model.evidence.comparator import comparator_logits

        device = state["device"]
        n_query = query_feature.shape[0]
        n_support = support_feature.shape[0]
        n_candidates = candidate_text.shape[0]

        out = comparator_logits(
            state["comparator"],
            candidate_text=candidate_text.unsqueeze(0).expand(n_query, -1, -1),
            query_feature=query_feature.unsqueeze(1),
            query_descriptor=query_descriptor.unsqueeze(1),
            query_mask=T.ones((n_query, 1), dtype=T.bool, device=device),
            support_feature=support_feature.unsqueeze(0).expand(n_query, -1, -1),
            support_descriptor=support_descriptor.unsqueeze(0).expand(n_query, -1, -1),
            support_label_text=support_label_text.unsqueeze(0).expand(n_query, -1, -1),
            support_bound=support_bound.unsqueeze(0).expand(n_query, -1),
            support_mask=T.ones((n_query, n_support), dtype=T.bool, device=device),
            candidate_slot=(1 + T.arange(n_candidates, device=device)).unsqueeze(0).expand(
                n_query, -1,
            ),
            candidate_mask=T.ones((n_query, n_candidates), dtype=T.bool, device=device),
            center=state["center"],
        )
        return out["logits"]

    def predict_enrollment(
        self, query_stream, support_stream, plan, support_count, candidate_texts,
        state, device, *, seed,
    ) -> Tuple[List[str], dict]:
        import torch as T

        query_feature, query_descriptor, _ = self._stream_rows(query_stream, state)
        support_feature_all, support_descriptor_all, _ = self._stream_rows(support_stream, state)

        canonical = list(plan["candidate_names"])
        if len(candidate_texts) != len(canonical):
            raise ValueError("candidate text roster must align with the manifest roster")
        candidate_text = F.normalize(T.from_numpy(
            state["sbert"]([text.replace("_", " ") for text in candidate_texts])
        ).float(), dim=-1).to(state["device"])

        # The manifest already chose which executions are enrolled for each candidate; the support
        # set is exactly those rows. No corpus bank, no retrieval, no selection.
        support_features = []
        support_descriptors = []
        bound: list[int] = []
        for slot, executions in enumerate(plan["support_execution_rows"]):
            for execution in executions[:support_count]:
                index = T.as_tensor(execution, dtype=T.long, device=state["device"])
                support_features.append(F.normalize(
                    support_feature_all.index_select(0, index).mean(dim=0), dim=0,
                ))
                support_descriptors.append(F.normalize(
                    support_descriptor_all.index_select(0, index).mean(dim=0), dim=0,
                ))
                bound.append(slot)
        if not support_features:
            raise ValueError("native enrollment received no support rows")
        support_bound = T.as_tensor(bound, dtype=T.long, device=state["device"])
        support_feature = T.stack(support_features)
        support_descriptor = T.stack(support_descriptors)
        # An enrolled row's label IS its candidate's text, which is what binds it.
        support_label_text = candidate_text.index_select(0, support_bound)

        requested = [int(row) for row in plan["query_rows"]]
        predictions: list[str] = []
        with T.no_grad():
            for start in range(0, len(requested), QUERY_CHUNK):
                chunk = requested[start:start + QUERY_CHUNK]
                wanted = T.as_tensor(chunk, dtype=T.long, device=state["device"])
                logits = self._score(
                    state,
                    query_feature.index_select(0, wanted),
                    query_descriptor.index_select(0, wanted),
                    support_feature, support_descriptor, support_label_text, support_bound,
                    candidate_text,
                )
                predictions.extend(canonical[int(i)] for i in logits.argmax(1).cpu().tolist())
        return predictions, {
            "mechanism": "support_comparator",
            "support_rows": int(len(support_features)),
            "corpus_rows": 0,
            "enrolled_executions": int(len(canonical) * support_count),
            "support_representation": "one_normalized_mean_vector_per_execution",
        }

    # ------------------------------------------------------------ zero shot
    def _compatible_pool(self, stream, state):
        """Training recordings sharing this stream's acquisition key, grouped by label."""
        from data.scripts.curate import deployment_policy
        from data.scripts.curate.compatibility import are_compatible, stream_key
        from training.compare.sampling import build_support_corpus

        key = (stream.dataset, stream.stream)
        if key not in state["zero_shot_support"]:
            query_key = stream_key(stream.dataset, stream.stream)
            corpus = state.get("_corpus")
            if corpus is None:
                corpus = build_support_corpus(
                    deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS,
                    max_per_stream=4000, seed=ZERO_SHOT_SEED,
                )
                state["_corpus"] = corpus
            streams = {
                index for index, other in enumerate(corpus.keys)
                if are_compatible(query_key, other)
            }
            by_label: dict[str, list] = {}
            for recording in corpus.recordings:
                if recording.stream_index in streams:
                    by_label.setdefault(recording.label, []).append(recording)
            state["zero_shot_support"][key] = by_label
        return state["zero_shot_support"][key]

    def _zero_shot_draws(self, stream, state, candidates: Sequence[str]):
        """R independent, training-shaped support sets of seen labels excluding every candidate.

        Each draw is a handful of labels with their example recordings — the shape the sampler
        trains on — rather than one wide flat set. Drawing is deterministic under ``ZERO_SHOT_SEED``
        so the row is reproducible.
        """
        by_label = self._compatible_pool(stream, state)
        if not by_label:
            return None, (
                "no training stream shares this acquisition configuration, so the deployed "
                "mechanism has nothing admissible to compare against at k=0"
            )
        banned = {str(text).replace("_", " ").lower() for text in candidates}
        eligible = sorted(
            label for label in by_label
            if str(label).replace("_", " ").lower() not in banned
        )
        if len(eligible) < 2:
            return None, (
                "fewer than two compatible training labels remain once the candidates are "
                "excluded, so a comparison would not be a decision"
            )

        rng = np.random.default_rng(ZERO_SHOT_SEED)
        draws = []
        for _ in range(ZERO_SHOT_DRAWS):
            n_labels = min(ZERO_SHOT_LABELS_PER_DRAW, len(eligible))
            chosen = rng.choice(eligible, size=n_labels, replace=False)
            rows = []
            for label in chosen:
                pool = by_label[str(label)]
                by_execution: dict[tuple[str, str, str], list] = {}
                for recording in pool:
                    by_execution.setdefault(
                        (recording.dataset, recording.subject, recording.execution), []
                    ).append(recording)
                units = list(by_execution)
                take = min(ZERO_SHOT_ROWS_PER_LABEL, len(units))
                picked = rng.choice(len(units), size=take, replace=False)
                for position in picked:
                    members = by_execution[units[int(position)]]
                    rows.append(members[int(rng.integers(len(members)))])
            if rows:
                draws.append(rows)
        if not draws:
            return None, "no compatible training recording could be drawn"
        return draws, None

    def _encode_training_recordings(self, recordings, state):
        """Encode training-corpus windows into support rows, grouped by their source stream.

        The training grids are not evaluation streams, so they are loaded directly and pushed
        through the same encoder path. Grouping by stream matters: channel text, gravity state and
        sampling rate are per-stream, and the encoder is told each group's own.
        """
        import torch as T
        from data.scripts.eda.grid_io import discover_grids
        from training.tokenizer.eval_transfer import encode_dataset_detailed
        from training.tokenizer.pretrain_data import (
            CHANNELS,
            _stream_gravity_state,
            stream_channel_descriptions,
            stream_sensor_texts,
        )

        refs = state.setdefault("_grid_refs", None)
        if refs is None:
            refs = {(ref.dataset, ref.stream): ref for ref in discover_grids("native")}
            state["_grid_refs"] = refs

        grouped: dict[tuple[str, str], list] = {}
        for recording in recordings:
            grouped.setdefault((recording.dataset, recording.stream), []).append(recording)

        features, descriptors, labels = [], [], []
        for (dataset, stream_id), members in grouped.items():
            ref = refs[(dataset, stream_id)]
            rows = np.asarray([m.window_index for m in members], dtype=np.int64)
            windows, mask = _six_slot(ref.load_data()[rows], ref.channels, ref.mask)
            gravity_state = _stream_gravity_state(dataset, stream_id)
            texts = stream_channel_descriptions(
                dataset, stream_id, neutral=_NEUTRAL_ACQUISITION_TEXT,
            )
            detailed = encode_dataset_detailed(
                state["encoder"], windows, texts, state["device"], float(ref.rate_hz),
                gravity_state=gravity_state, channel_mask=mask,
                dataset=dataset, stream=stream_id,
                neutral_text=_NEUTRAL_ACQUISITION_TEXT, export_sensor_rows=False,
            )
            _, sensor_texts, _ = stream_sensor_texts(
                dataset, stream_id,
                gravity_removed=(gravity_state == "removed"),
                has_accel=any(c.startswith("acc") for c, m in zip(CHANNELS, mask) if m),
                has_gyro=any(c.startswith("gyro") for c, m in zip(CHANNELS, mask) if m),
                neutral=_NEUTRAL_ACQUISITION_TEXT,
            )
            encoded = state["encoder"].text_encoder.encode_pooled(
                sensor_texts, device=state["device"],
            )
            descriptor = F.normalize(
                F.normalize(encoded.float(), dim=-1).mean(dim=0, keepdim=True), dim=-1,
            )
            pooled = detailed["pooled"].to(state["device"])
            features.append(pooled)
            descriptors.append(descriptor.expand(len(pooled), -1))
            labels.extend(m.label for m in members)
        return T.cat(features), T.cat(descriptors), labels

    def predict_candidates_from_features(self, features, candidates, state, device):
        """The k=0 row: an ensemble of training-shaped comparisons, then one vote.

        Each draw asks the same question the sampler asks at training time — "which of these few
        seen activities does the query resemble?" — and answers it in candidate space through the
        label-text bridge. The draws are combined before the argmax, so no single unlucky support
        set decides the row. No support set can contain the answer: every candidate label is
        excluded from all of them.
        """
        import torch as T

        owner = state["feature_owner"].get(id(features))
        if owner is None:
            raise ValueError(
                "halo_compare scores k=0 with its own mechanism over recording rows; the features "
                "passed here were not produced by this adapter's window_features"
            )
        from eval import data as eval_data

        stream = eval_data.load_eval_stream(*owner)
        candidates = list(candidates)
        draws, reason = self._zero_shot_draws(stream, state, candidates)
        if draws is None:
            # An honest unsupported row. Substituting ConSE, or padding with incompatible
            # recordings, would report a different mechanism under this model's name.
            raise UnsupportedEvaluationCell(
                f"zero-shot unsupported for {owner[0]}/{owner[1]}: {reason}"
            )

        query_feature, query_descriptor, _ = self._stream_rows(stream, state)
        candidate_text = F.normalize(T.from_numpy(
            state["sbert"]([c.replace("_", " ") for c in candidates])
        ).float(), dim=-1).to(state["device"])

        per_draw: list[T.Tensor] = []
        draw_labels: list[list[str]] = []
        with T.no_grad():
            for rows in draws:
                support_feature, support_descriptor, labels = self._encode_training_recordings(
                    rows, state,
                )
                support_label_text = F.normalize(T.from_numpy(
                    state["sbert"]([label.replace("_", " ") for label in labels])
                ).float(), dim=-1).to(state["device"])
                # Every support row is unbound: none carries a candidate's label, by construction.
                support_bound = T.full(
                    (len(labels),), -1, dtype=T.long, device=state["device"],
                )
                chunks = []
                for start_row in range(0, len(query_feature), QUERY_CHUNK):
                    stop = start_row + QUERY_CHUNK
                    chunks.append(self._score(
                        state,
                        query_feature[start_row:stop], query_descriptor[start_row:stop],
                        support_feature, support_descriptor, support_label_text, support_bound,
                        candidate_text,
                    ))
                per_draw.append(T.cat(chunks, dim=0))
                draw_labels.append(sorted(set(labels)))

        combined = _ensemble(T.stack(per_draw), ZERO_SHOT_ENSEMBLE)
        predictions = [candidates[int(i)] for i in combined.argmax(1).cpu().tolist()]
        # Agreement across draws is the honest read on whether the ensemble is doing work: if every
        # draw already agreed there was nothing to average, and if none agree the row is noise.
        votes = T.stack([logits.argmax(1) for logits in per_draw])
        agreement = float((votes == votes.mode(dim=0).values.unsqueeze(0)).float().mean())
        return predictions, {
            "mechanism": "support_comparator_zero_shot_ensemble",
            "predicted_classes": sorted(set(predictions)),
            "draws": len(per_draw),
            "labels_per_draw": ZERO_SHOT_LABELS_PER_DRAW,
            "rows_per_draw": int(len(draw_labels[0]) and len(draws[0])),
            "ensemble": ZERO_SHOT_ENSEMBLE,
            "draw_agreement": agreement,
            # How much the ensemble actually had to average over. A compatible pool with few
            # labels left after excluding the candidates makes every draw pick the SAME labels,
            # and the ensemble degenerates to resampling rows. Measured range across the
            # evaluation streams: 7 compatible training labels for inclusivehar/phone_waist
            # against 86 for tnda_har/watch_wrist. Reported per row rather than assumed.
            "distinct_label_sets": len({tuple(labels) for labels in draw_labels}),
            "support_labels": sorted({label for labels in draw_labels for label in labels}),
            "support_policy": "config_compatible_training_rows_excluding_candidates",
        }

    def evaluation_config(self, state) -> dict:
        return {
            "checkpoint": str(_CKPT),
            "checkpoint_sha256": _ckpt_sha256(),
            "checkpoint_step": state.get("checkpoint_step"),
            "neutral_acquisition_text": _NEUTRAL_ACQUISITION_TEXT,
            "mechanism": "support_comparator",
            "center_features": state.get("center"),
            "corpus_bank_at_positive_k": False,
            "zero_shot_draws": ZERO_SHOT_DRAWS,
            "zero_shot_labels_per_draw": ZERO_SHOT_LABELS_PER_DRAW,
            "zero_shot_rows_per_label": ZERO_SHOT_ROWS_PER_LABEL,
            "zero_shot_ensemble": ZERO_SHOT_ENSEMBLE,
            "zero_shot_seed": ZERO_SHOT_SEED,
            "zero_shot_support_policy": "config_compatible_training_rows_excluding_candidates",
        }

    def evaluation_artifacts(self, state):
        return {"checkpoint": _CKPT} if _CKPT.exists() else {}

    def evaluation_source_paths(self):
        return (
            _REPO / "model" / "evidence" / "comparator.py",
            _REPO / "training" / "compare",
            _REPO / "data" / "scripts" / "curate" / "compatibility.py",
        )
