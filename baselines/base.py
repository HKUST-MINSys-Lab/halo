"""Baseline adapter framework for the ZS-XD evaluation protocol (v2).

Each baseline is a small adapter that subclasses one of two tiers and is
registered with :func:`register`:

  * :class:`ConSEAdapter` — a closed-vocabulary classifier. It emits a per-window
    softmax over the GLOBAL training vocabulary
    (``data/labels/global_labels.json``); the base bridges that to the target
    dataset's label strings with ConSE (Norouzi et al., 2014).
  * :class:`CosineAdapter` — a text-aligned model with its own text tower. It
    emits per-window sensor embeddings and label embeddings in a shared space;
    the base scores by cosine similarity (no bridge).

An adapter declares its INPUT CONTRACT (channels / rate / window) so a per-baseline
resampler (added later) can honour it, implements ``setup`` plus its one tier
method, and is decorated with ``@register``. The base owns the shared plumbing —
ground truth, the ConSE/cosine scoring, subject-stratified CIs — via
:mod:`eval.scoring` and :mod:`eval.data`, so there is NO per-baseline dispatch
code and no per-baseline ground-truth handling (the source of the legacy label
bugs). No concrete baseline lives here yet; drop a ``<name>/adapter.py`` in to add one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval import data as eval_data
from eval import scoring

# Populated by @register at import time: name -> adapter instance.
REGISTRY: Dict[str, "BaselineAdapter"] = {}


class UnsupportedEvaluationCell(RuntimeError):
    """A model cannot score one protocol cell without changing its declared mechanism."""


def register(cls):
    """Class decorator: instantiate and add to REGISTRY under ``cls.name``."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} must set a non-empty `name` to register")
    REGISTRY[cls.name] = cls()
    return cls


@dataclass(frozen=True)
class InputContract:
    """What a baseline expects its sensor input to look like.

    A ``None`` field means "accepts native" (the adapter handles it internally).
    Consumed by a per-baseline resampler (added with the concrete adapters); the
    base only records it so the eval driver can honour each baseline's contract
    instead of feeding every model the same tensor.
    """
    channels: Optional[Sequence[str]] = None  # required channel names/order, or None
    rate_hz: Optional[float] = None           # required sampling rate, or None
    window_sec: Optional[float] = None        # required window length in seconds, or None


class BaselineAdapter:
    """Base adapter. Subclass :class:`ConSEAdapter` or :class:`CosineAdapter`.

    Subclasses set `name` + `tier`, declare `contract`, and implement `setup`
    plus their tier's prediction method. The shared :meth:`evaluate` turns a
    dataset/stream into the v2 metric bundle.
    """
    name: str = ""
    tier: str = ""  # "conse" | "cosine"
    contract: InputContract = InputContract()

    def setup(self, device):
        """Load model + artifacts once; return an opaque state object."""
        raise NotImplementedError

    def setup_features(self, device):
        """Load only what :meth:`window_features` needs.

        Most adapters use the same state for native prediction and frozen
        features.  Adapters whose classifier setup performs additional fitting
        may override this hook so representation-cache generation never trains
        an irrelevant prediction head.
        """

        return self.setup(device)

    def feature_artifacts(self, state) -> Dict[str, Path]:
        """Artifacts that determine :meth:`window_features`."""

        return self.evaluation_artifacts(state)

    def feature_config(self, state) -> dict:
        """Configuration that determines :meth:`window_features`."""

        return self.evaluation_config(state)

    def is_incompatible(self, dataset: str) -> Optional[str]:
        """Return a short reason if this model CANNOT be validly scored on
        `dataset` (e.g. a gravity-dependent model on a gravity-removed set), else
        None. The driver records such a case as an explicit, disclosed N/A cell —
        not silently scored, and not counted as a failure."""
        return None

    def predict(self, stream: eval_data.EvalStream, state, device) -> Tuple[List[str], dict]:
        """Per-window predictions over ``stream.eval_labels`` (aligned 1:1 with
        ``stream.windows``) plus an info dict. Implemented by each tier."""
        raise NotImplementedError

    def predict_candidates(
        self, stream: eval_data.EvalStream, candidates: Sequence[str], state, device
    ) -> Tuple[List[str], dict]:
        """Predict over an explicitly frozen candidate roster."""
        if list(candidates) != list(stream.eval_labels):
            raise NotImplementedError(
                f"{self.name} cannot override its candidate roster"
            )
        return self.predict(stream, state, device)

    def window_features(self, stream: eval_data.EvalStream, state, device) -> np.ndarray:
        """Frozen per-window representation used by the matched enrollment protocol.

        Concrete adapters must expose the representation immediately before their published
        classifier or native text-matching rule.  Keeping this method explicit prevents the
        adaptation evaluator from accidentally training on logits for one model and trunk features
        for another.
        """
        raise NotImplementedError(f"{self.name} does not expose frozen window features")

    def predict_candidates_from_features(
        self, features: np.ndarray, candidates: Sequence[str], state, device
    ) -> Tuple[List[str], dict]:
        """Predict from :meth:`window_features` without re-encoding the signal."""
        raise NotImplementedError(f"{self.name} cannot predict from cached window features")

    def supports_native_enrollment(self) -> bool:
        """Whether this model has a deployment-time enrollment mechanism of its own.

        The shared nearest/prototype/ridge/linear-head controls operate on
        :meth:`window_features`. A model with a native memory mechanism implements this hook so the
        evaluator can score that mechanism on the exact same serialized episodes without teaching
        the generic runner model-specific behavior.
        """
        return False

    def supports_native_zero_shot(self) -> bool:
        """Whether candidate scoring is part of the released/deployed model itself.

        Text-aligned cosine models satisfy this through their native shared space. A locally fitted
        ConSE bridge does not: it remains available for historical analyses but is not a native
        zero-shot result in the current comparison.
        """
        return self.tier == "cosine"

    def predict_enrollment(
        self,
        query_stream: eval_data.EvalStream,
        support_stream: eval_data.EvalStream,
        plan: dict,
        support_count: int,
        candidate_texts: Sequence[str],
        state,
        device,
        *,
        seed: int,
    ) -> Tuple[List[str], dict]:
        """Native enrollment predictions aligned to ``plan['query_rows']``."""
        raise NotImplementedError(f"{self.name} has no native enrollment mechanism")

    def evaluation_artifacts(self, state) -> Dict[str, Path]:
        """Named files whose content determines an evaluation result.

        Adapters should include released/self-trained checkpoints and fitted heads. The runner
        hashes these files into every result artifact; source hashes alone cannot detect a silently
        replaced checkpoint.
        """
        return {}

    def evaluation_source_paths(self) -> Sequence[Path]:
        """Additional shared source trees that determine this adapter's predictions."""
        return ()

    def evaluation_config(self, state) -> dict:
        """Small, JSON-serializable adapter settings that affect evaluation semantics."""
        return {}

    def evaluate(
        self,
        dataset: str,
        stream: str,
        *,
        alignment: str = "non_harmonised",
        device="cpu",
        state=None,
    ) -> dict:
        """Score this baseline on one dataset/stream -> v2 metric bundle.

        Loads the grid, checks compatibility, runs the tier's `predict`, restricts
        to windows whose ground truth is in the candidate vocabulary, and computes
        the metric bundle. Returns ``{"status": "n/a", "reason": ...}`` for a
        disclosed incompatible dataset (no invalid number is produced).
        """
        na = self.is_incompatible(dataset)
        if na is not None:
            return {"status": "n/a", "reason": na, "dataset": dataset, "stream": stream}

        s = eval_data.load_eval_stream(dataset, stream, alignment=alignment)
        if state is None:
            state = self.setup(device)

        preds, info = self.predict(s, state, device)
        if len(preds) != s.n_windows:
            raise ValueError(
                f"{self.name}: predicted {len(preds)} labels for {s.n_windows} windows "
                f"({dataset}/{stream}) — predictions must align 1:1 with windows."
            )

        gt, subjects, keep_idx = scoring.filter_ground_truth(s.gt, s.subjects, s.eval_labels)
        preds = [preds[i] for i in keep_idx]
        return score(gt, preds, subjects, extra=info)


class ConSEAdapter(BaselineAdapter):
    """Closed-vocabulary classifier scored via the ConSE bridge."""
    tier = "conse"

    def window_probs(self, stream: eval_data.EvalStream, state, device) -> np.ndarray:
        """Per-window softmax over the GLOBAL training labels: (N, K), aligned
        1:1 with ``stream.windows``. K == len(global_labels())."""
        raise NotImplementedError

    def predict(self, stream, state, device) -> Tuple[List[str], dict]:
        return self.predict_candidates(stream, stream.eval_labels, state, device)

    def predict_candidates(self, stream, candidates, state, device) -> Tuple[List[str], dict]:
        probs = np.asarray(self.window_probs(stream, state, device))
        vocab = global_labels()
        if probs.shape[1] != len(vocab):
            raise ValueError(
                f"{self.name}: window_probs has {probs.shape[1]} columns but the "
                f"global vocabulary has {len(vocab)} labels."
            )
        return scoring.conse_predict(probs, vocab, list(candidates))


class CosineAdapter(BaselineAdapter):
    """Text-aligned model scored by cosine similarity (no bridge)."""
    tier = "cosine"

    def window_embeddings(self, stream: eval_data.EvalStream, state, device) -> np.ndarray:
        """Per-window L2-normalized sensor embeddings: (N, D)."""
        raise NotImplementedError

    def encode_labels(self, labels: Sequence[str], state, device) -> np.ndarray:
        """Encode the target label strings into the same space: (L, D)."""
        raise NotImplementedError

    def predict(self, stream, state, device) -> Tuple[List[str], dict]:
        return self.predict_candidates(stream, stream.eval_labels, state, device)

    def predict_candidates(self, stream, candidates, state, device) -> Tuple[List[str], dict]:
        emb = np.asarray(self.window_embeddings(stream, state, device))      # (N, D)
        return self.predict_candidates_from_features(emb, candidates, state, device)

    def predict_candidates_from_features(self, features, candidates, state, device):
        emb = np.asarray(features)
        candidates = list(candidates)
        cache = state.setdefault("_candidate_embedding_cache", {})
        cache_key = tuple(candidates)
        if cache_key not in cache:
            cache[cache_key] = np.asarray(
                self.encode_labels(candidates, state, device), dtype=np.float32
            )
        lab = cache[cache_key]                                                # (L, D)
        sims = emb @ lab.T                                                    # (N, L)
        preds = scoring.predict_from_similarity(sims, candidates)
        return preds, {"predicted_classes": sorted(set(preds))}

    def window_features(self, stream, state, device) -> np.ndarray:
        return np.asarray(self.window_embeddings(stream, state, device), dtype=np.float32)


# =============================================================================
# Shared ground truth + scoring (offset-free v2)
# =============================================================================

def load_gt(
    dataset: str,
    stream: str,
    alignment: str = "non_harmonised",
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Canonical ground truth for a dataset/stream: ``(eval_labels, gt_names,
    subjects, keep_idx)``.

    Reads the per-window label strings and subject ids directly from the grid
    (offset-free — never the legacy code-offset path) and drops windows whose
    label is outside the dataset's candidate vocabulary. `keep_idx` indexes the
    original N windows, so any per-window model output aligns via
    ``output[keep_idx]``.
    """
    s = eval_data.load_eval_stream(dataset, stream, alignment=alignment)
    gt_names, subjects, keep_idx = scoring.filter_ground_truth(s.gt, s.subjects, s.eval_labels)
    return s.eval_labels, gt_names, subjects, keep_idx


def score(gt_names, pred_names, subjects, extra: Optional[dict] = None) -> dict:
    """v2 metric bundle: macro-F1 (primary) + balanced-acc + subject CIs + per-class."""
    m = scoring.classification_metrics(gt_names, pred_names)
    # Small-cohort datasets (few subjects) get a jagged, over-wide subject bootstrap → use the
    # leave-one-subject-out jackknife CI instead; larger cohorts keep the bootstrap (#7).
    import numpy as _np
    if len(_np.unique(subjects)) < scoring.SMALL_COHORT_MAX_SUBJECTS:
        m.update(scoring.subject_groupkfold_ci(gt_names, pred_names, subjects, metric="f1_macro"))
    else:
        m.update(scoring.subject_bootstrap_ci(gt_names, pred_names, subjects, metric="f1_macro"))
    m["per_class_f1"] = scoring.per_class_f1(gt_names, pred_names)
    if extra:
        m.update(extra)
    return m


def fit_fingerprint(**parts) -> str:
    """Hash of everything that changes what a fitted ConSE head IS (Phase 1.6 / audit H6).

    The old caches validated only `labels` (and later corpus mode), so they silently survived
    changes to the subject split, the seed, the probe architecture, the backbone checkpoint, the
    per-stream cap and the optimizer settings — an audit demonstrated a fabricated cache with
    ``n_windows=1`` being accepted. Any change to a part here invalidates the cache and forces a
    refit, which is the safe default: a wrong cache produces a silently wrong number.
    """
    import hashlib
    import json
    blob = json.dumps({k: parts[k] for k in sorted(parts)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


PROBE_HIDDEN = 512          # the ONE hidden width; fingerprints derive from this, never a literal
PROBE_SPEC = f"2layer-{PROBE_HIDDEN}"


def make_probe(feat_dim: int, n_classes: int, hidden: int = PROBE_HIDDEN):
    """The ONE probe architecture every ConSE-tier baseline fits on its frozen features.

    Phase 1.4. Previously harnet used the official 2-layer ``EvaClassifier``
    (feat→512→n, ~300k params) while halo, crosshar and limubert used a single
    ``nn.Linear(feat→n)`` (~24k) — so harnet was given **strictly more probe capacity than
    everyone else, including us**. That is not a "same probe" comparison.

    Unifying on the 2-layer form: leaves harnet exactly at its paper's head, and *strengthens*
    the other three (it also moves LiMU-BERT toward its paper's non-linear downstream classifier,
    where a linear probe understates it). Strengthening the baselines is the honest direction —
    if our numbers survive it they are better earned.

    Shape matches ssl-wearables ``EvaClassifier``: Linear(feat, hidden) → ReLU → Linear(hidden, n).
    """
    import torch.nn as nn
    return nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))


def global_labels() -> List[str]:
    """The global ConSE training-label vocabulary."""
    return eval_data.load_global_labels()
