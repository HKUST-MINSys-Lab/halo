"""One construction path for learned HALO support classifiers, shared by trainer and evaluators.

Architecture identifiers are persisted in checkpoints as ``architecture_version``; unknown or
partially specified checkpoints fail here rather than being loaded with ``strict=False``.
"""

from __future__ import annotations

import torch

from model.blocks import AttentionSpec
from model.support.contextual_classifier import (
    ARCHITECTURE_VERSION as LEGACY_CONTEXTUAL_ARCHITECTURE,
    ContextualClassifierConfig, ContextualSupportClassifier,
)
from model.support.contextual_residual_classifier import (
    ARCHITECTURE_VERSION as CONTEXTUAL_RESIDUAL_ARCHITECTURE,
    ContextualResidualClassifierConfig, ContextualResidualSupportClassifier,
)
from model.support.evidence_aware_classifier import (
    ARCHITECTURE_VERSION as EVIDENCE_AWARE_ARCHITECTURE,
    EvidenceAwareClassifierConfig, EvidenceAwareSupportClassifier,
)
from model.support.evidence_gated_classifier import (
    ARCHITECTURE_VERSION as EVIDENCE_GATED_ARCHITECTURE,
    EvidenceGatedClassifierConfig, EvidenceGatedSupportClassifier,
)
from model.support.residual_classifier import ResidualClassifierConfig, build_support_classifier

RESIDUAL_ARCHITECTURES = frozenset({"support_classifier_v2", "support_classifier_v3"})
LEARNED_CLASSIFIER_ARCHITECTURES = RESIDUAL_ARCHITECTURES | {
    LEGACY_CONTEXTUAL_ARCHITECTURE, CONTEXTUAL_RESIDUAL_ARCHITECTURE,
    EVIDENCE_AWARE_ARCHITECTURE, EVIDENCE_GATED_ARCHITECTURE,
}
# ``contextual`` is the CLI mode, not a checkpoint family.  New runs use v2 while v1 remains
# explicitly addressable for checkpoint loading and historical evaluation.
CONTEXTUAL_ARCHITECTURE = EVIDENCE_AWARE_ARCHITECTURE
CONTEXTUAL_CHECKPOINT_ARCHITECTURES = frozenset({CONTEXTUAL_RESIDUAL_ARCHITECTURE, EVIDENCE_AWARE_ARCHITECTURE})
MODE_TO_ARCHITECTURE = {"residual": "support_classifier_v3", "contextual": EVIDENCE_AWARE_ARCHITECTURE,
                        "evidence_gated": EVIDENCE_GATED_ARCHITECTURE,
                        "token_mixer": "support_token_mixer_v1"}

# One authoritative lifecycle registry for humans and tooling. Historical architectures remain
# strictly loadable so old results are reproducible, but they must not be mistaken for active
# experiment choices. The CLI mode ``contextual`` always constructs the active experimental v2.
CLASSIFIER_ARCHITECTURE_STATUS = {
    "support_token_mixer_v1": "retired-reproduction-only",
    "support_classifier_v2": "historical-checkpoint-only",
    # Promoted 2026-09-18, superseded 2026-09-21 by the T6 recipe of support_classifier_v4 ("v4").
    "support_classifier_v3": "superseded-control",
    LEGACY_CONTEXTUAL_ARCHITECTURE: "abandoned-negative-result",
    CONTEXTUAL_RESIDUAL_ARCHITECTURE: "abandoned-negative-result",
    # v2 completed its matched sealed + scenario evaluation on 2026-09-19 and did not beat the
    # v3 control (router collapsed onto label meaning); see docs/results/RESULTS.md.
    EVIDENCE_AWARE_ARCHITECTURE: "abandoned-negative-result",
    # Promoted 2026-09-21 as "v4", in its T6 recipe (see PROMOTED_RECIPE). T7 remains an active
    # experiment on the same architecture string.
    EVIDENCE_GATED_ARCHITECTURE: "promoted-control",
}
# Human-facing names, adopted 2026-09-20. `v3` is the promoted classifier; every experimental
# replacement is T-numbered in the order it was trained ("T" for try). The architecture strings are
# NOT numbered consistently with the tries -- `support_classifier_v4` is T4, not a successor to
# `support_classifier_v3` -- which is why this mapping is explicit. Tries T4, T5 and T6 all share
# the v4 architecture string and differ by recipe (config flags and curriculum); this maps an
# architecture to its BASE try. See docs/results/RESULTS.md, "Classifier naming".
CLASSIFIER_TRY_NAME = {
    "support_classifier_v3": "v3",
    "support_contextual_mixture_v1": "T1",
    "support_contextual_residual_v1": "T2",
    "support_evidence_aware_v2": "T3",
    "support_classifier_v4": "T4",
}


def classifier_architecture_base_try_name(architecture: str) -> str:
    """Human-facing base name for an architecture, not a particular training recipe."""
    try:
        return CLASSIFIER_TRY_NAME[architecture]
    except KeyError as exc:
        raise ValueError(f"no try name registered for {architecture!r}") from exc


def classifier_try_name(architecture: str, *, trajectory: dict | None = None) -> str:
    """Return a recipe-aware experimental name for a persisted classifier run.

    ``support_classifier_v4`` is shared by T4, T5, and T6. Its architecture string alone is
    therefore intentionally insufficient: callers must supply the saved trajectory to avoid
    silently labelling one recipe as another.
    """
    if architecture != EVIDENCE_GATED_ARCHITECTURE:
        return classifier_architecture_base_try_name(architecture)
    if trajectory is None:
        raise ValueError("support_classifier_v4 requires its saved trajectory for a T-number")
    mode = trajectory.get("text_corruption_mode", "replace")
    probability = float(trajectory.get("text_corruption_probability", 0.0))
    calibrated = bool(trajectory.get("unenrolled_calibration", False))
    if mode == "auxiliary" and calibrated:
        # The T6 recipe was promoted on 2026-09-21 and is called v4 from then on; T7 adds the
        # primitive semantic branch and stays experimental.
        return "T7" if trajectory.get("semantic_mode", "text") != "text" else "v4"
    if probability == 0.0:
        return "T5"
    if mode == "replace" and not calibrated:
        return "T4"
    raise ValueError(f"unrecognized support_classifier_v4 recipe: {trajectory!r}")


PROMOTED_CLASSIFIER_ARCHITECTURE = EVIDENCE_GATED_ARCHITECTURE
PROMOTED_CLASSIFIER_NAME = "v4"
# The promoted classifier is a RECIPE on support_classifier_v4, not the architecture alone: the
# T6 training recipe, i.e. corruption as a gate-only auxiliary on every eligible episode plus the
# label-blind unenrolled calibration term, with the promoted cosine semantic path. These are the
# trainer's defaults for --classifier evidence_gated since 2026-09-21; the T4 recipe remains
# reachable with --text-corruption-mode replace --no-unenrolled-calibration.
PROMOTED_RECIPE = {
    "text_corruption_mode": "auxiliary",
    "text_corruption_probability": 1.0,
    "unenrolled_calibration": True,
    "semantic_mode": "text",
}
SUPERSEDED_CLASSIFIER_ARCHITECTURE = "support_classifier_v3"
# The active experiment is T7, a recipe (semantic_mode != "text") on the promoted architecture.
ACTIVE_EXPERIMENTAL_CLASSIFIER_ARCHITECTURE = EVIDENCE_GATED_ARCHITECTURE
ABANDONED_CLASSIFIER_ARCHITECTURES = frozenset({
    LEGACY_CONTEXTUAL_ARCHITECTURE,
    CONTEXTUAL_RESIDUAL_ARCHITECTURE,
    EVIDENCE_AWARE_ARCHITECTURE,
})
CONTEXTUAL_READOUTS = ("halo-classifier-semantic-only", "halo-classifier-support-floor",
                       "halo-classifier-contextual-support")
EVIDENCE_AWARE_READOUTS = (
    "halo-classifier-label-meaning-only", "halo-classifier-unmodified-support-vote",
    "halo-classifier-contextual-support-vote",
)
# v4 branch readouts: the final blend, its label-meaning branch, its trust-weighted support vote,
# and the untrusted closed-form vote the trust weights are a residual on.
EVIDENCE_GATED_READOUTS = (
    "halo-classifier-label-meaning-only", "halo-classifier-support-vote",
    "halo-classifier-untrusted-support-vote",
    "halo-classifier-text-off-blend", "halo-classifier-trust-off-blend",
)
# Emitted only for checkpoints whose semantic mode combines both paths: the attribution question
# for the primitive semantic path is what each half of the semantic branch contributes.
EVIDENCE_GATED_SEMANTIC_READOUTS = (
    "halo-classifier-semantic-text-only", "halo-classifier-semantic-primitives-only",
)
LEGACY_CONTEXTUAL_READOUTS = (
    "halo-classifier-semantic-only", "halo-classifier-support-only",
    "halo-classifier-fixed-half-mixture",
)


def classifier_architecture(mode: str) -> str | None:
    return MODE_TO_ARCHITECTURE.get(mode)


def classifier_architecture_status(architecture: str) -> str:
    """Return the declared lifecycle state of a persisted classifier architecture."""
    try:
        return CLASSIFIER_ARCHITECTURE_STATUS[architecture]
    except KeyError as exc:
        raise ValueError(f"unknown support-classifier architecture {architecture!r}") from exc


def build_classifier_from_blob(blob: dict, *, device=None, overrides: dict | None = None):
    """Rebuild a learned head from a checkpoint payload and load its state strictly.

    Returns ``(head, architecture_version)``. ``overrides`` may replace residual-config fields
    (e.g. ``residual_enabled``) for ablation readouts; they are rejected for other architectures.
    """
    version = blob.get("architecture_version")
    if blob.get("classifier") is None or blob.get("classifier_config") is None \
            or blob.get("attention_spec") is None:
        raise ValueError("checkpoint does not contain a learned support classifier")
    spec = AttentionSpec(**blob["attention_spec"])
    config = dict(blob["classifier_config"])
    if version in RESIDUAL_ARCHITECTURES:
        if version == "support_classifier_v2":
            config.setdefault("normalized_token_composition", False)
        config.update(overrides or {})
        head = build_support_classifier(spec, ResidualClassifierConfig(**config))
    elif version == LEGACY_CONTEXTUAL_ARCHITECTURE:
        if overrides:
            raise ValueError("residual ablation flags are not defined for the contextual head")
        head = ContextualSupportClassifier(spec, ContextualClassifierConfig(**config))
    elif version == CONTEXTUAL_RESIDUAL_ARCHITECTURE:
        if overrides:
            raise ValueError("residual ablation flags are not defined for the contextual head")
        head = ContextualResidualSupportClassifier(
            spec, ContextualResidualClassifierConfig(**config),
        )
    elif version == EVIDENCE_AWARE_ARCHITECTURE:
        if overrides:
            raise ValueError("residual ablation flags are not defined for the evidence-aware head")
        head = EvidenceAwareSupportClassifier(spec, EvidenceAwareClassifierConfig(**config))
    elif version == EVIDENCE_GATED_ARCHITECTURE:
        # Only the two ablation switches may be overridden; a readout must not silently change
        # capacity, bounds or temperatures, which would make its rows incomparable.
        allowed = {"trust_enabled", "text_term_enabled"}
        if set(overrides or {}) - allowed:
            raise ValueError(f"evidence-gated overrides are limited to {sorted(allowed)}")
        config.update(overrides or {})
        head = EvidenceGatedSupportClassifier(spec, EvidenceGatedClassifierConfig(**config))
    else:
        raise ValueError(f"unsupported support-classifier architecture {version!r}")
    head.load_state_dict(blob["classifier"], strict=True)
    expected_primitive = blob.get("primitive_provenance")
    if expected_primitive is not None:
        primitive = getattr(head, "primitive_head", None)
        if primitive is None or primitive.provenance != expected_primitive:
            raise ValueError("primitive vocabulary provenance differs from the checkpoint")
    if device is not None:
        head = head.to(device)
    return head.eval(), version


def checkpoint_architecture(checkpoint) -> str | None:
    return torch.load(checkpoint, map_location="cpu", weights_only=False).get("architecture_version")
