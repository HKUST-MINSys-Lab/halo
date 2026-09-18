"""One construction path for learned HALO support classifiers, shared by trainer and evaluators.

Architecture identifiers are persisted in checkpoints as ``architecture_version``; unknown or
partially specified checkpoints fail here rather than being loaded with ``strict=False``.
"""

from __future__ import annotations

import torch

from model.blocks import AttentionSpec
from model.support.contextual_classifier import (
    ARCHITECTURE_VERSION as CONTEXTUAL_ARCHITECTURE,
    ContextualClassifierConfig, ContextualSupportClassifier,
)
from model.support.residual_classifier import ResidualClassifierConfig, build_support_classifier

RESIDUAL_ARCHITECTURES = frozenset({"support_classifier_v2", "support_classifier_v3"})
LEARNED_CLASSIFIER_ARCHITECTURES = RESIDUAL_ARCHITECTURES | {CONTEXTUAL_ARCHITECTURE}
MODE_TO_ARCHITECTURE = {"residual": "support_classifier_v3", "contextual": CONTEXTUAL_ARCHITECTURE,
                        "token_mixer": "support_token_mixer_v1"}
CONTEXTUAL_READOUTS = ("halo-classifier-semantic-only", "halo-classifier-support-only",
                       "halo-classifier-fixed-half-mixture")


def classifier_architecture(mode: str) -> str | None:
    return MODE_TO_ARCHITECTURE.get(mode)


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
    elif version == CONTEXTUAL_ARCHITECTURE:
        if overrides:
            raise ValueError("residual ablation flags are not defined for the contextual head")
        head = ContextualSupportClassifier(spec, ContextualClassifierConfig(**config))
    else:
        raise ValueError(f"unsupported support-classifier architecture {version!r}")
    head.load_state_dict(blob["classifier"], strict=True)
    if device is not None:
        head = head.to(device)
    return head.eval(), version


def checkpoint_architecture(checkpoint) -> str | None:
    return torch.load(checkpoint, map_location="cpu", weights_only=False).get("architecture_version")
