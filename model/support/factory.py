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
from model.support.residual_classifier import ResidualClassifierConfig, build_support_classifier

RESIDUAL_ARCHITECTURES = frozenset({"support_classifier_v2", "support_classifier_v3"})
LEARNED_CLASSIFIER_ARCHITECTURES = RESIDUAL_ARCHITECTURES | {
    LEGACY_CONTEXTUAL_ARCHITECTURE, CONTEXTUAL_RESIDUAL_ARCHITECTURE,
    EVIDENCE_AWARE_ARCHITECTURE,
}
# ``contextual`` is the CLI mode, not a checkpoint family.  New runs use v2 while v1 remains
# explicitly addressable for checkpoint loading and historical evaluation.
CONTEXTUAL_ARCHITECTURE = EVIDENCE_AWARE_ARCHITECTURE
CONTEXTUAL_CHECKPOINT_ARCHITECTURES = frozenset({CONTEXTUAL_RESIDUAL_ARCHITECTURE, EVIDENCE_AWARE_ARCHITECTURE})
MODE_TO_ARCHITECTURE = {"residual": "support_classifier_v3", "contextual": EVIDENCE_AWARE_ARCHITECTURE,
                        "token_mixer": "support_token_mixer_v1"}

# One authoritative lifecycle registry for humans and tooling. Historical architectures remain
# strictly loadable so old results are reproducible, but they must not be mistaken for active
# experiment choices. The CLI mode ``contextual`` always constructs the active experimental v2.
CLASSIFIER_ARCHITECTURE_STATUS = {
    "support_token_mixer_v1": "retired-reproduction-only",
    "support_classifier_v2": "historical-checkpoint-only",
    "support_classifier_v3": "promoted-control",
    LEGACY_CONTEXTUAL_ARCHITECTURE: "abandoned-negative-result",
    CONTEXTUAL_RESIDUAL_ARCHITECTURE: "abandoned-negative-result",
    EVIDENCE_AWARE_ARCHITECTURE: "active-experimental",
}
PROMOTED_CLASSIFIER_ARCHITECTURE = "support_classifier_v3"
ACTIVE_EXPERIMENTAL_CLASSIFIER_ARCHITECTURE = EVIDENCE_AWARE_ARCHITECTURE
ABANDONED_CLASSIFIER_ARCHITECTURES = frozenset({
    LEGACY_CONTEXTUAL_ARCHITECTURE,
    CONTEXTUAL_RESIDUAL_ARCHITECTURE,
})
CONTEXTUAL_READOUTS = ("halo-classifier-semantic-only", "halo-classifier-support-floor",
                       "halo-classifier-contextual-support")
EVIDENCE_AWARE_READOUTS = (
    "halo-classifier-label-meaning-only", "halo-classifier-unmodified-support-vote",
    "halo-classifier-contextual-support-vote",
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
    else:
        raise ValueError(f"unsupported support-classifier architecture {version!r}")
    head.load_state_dict(blob["classifier"], strict=True)
    if device is not None:
        head = head.to(device)
    return head.eval(), version


def checkpoint_architecture(checkpoint) -> str | None:
    return torch.load(checkpoint, map_location="cpu", weights_only=False).get("architecture_version")
