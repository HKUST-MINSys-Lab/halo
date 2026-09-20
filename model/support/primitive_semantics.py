"""Primitive-driven semantic alignment: a grounded, compositional alternative to ``p_text``.

Design of record: ``docs/journal/2026-09-20-primitive-semantic-path-design.md``.

The promoted semantic path is one linear map from a pooled sensor vector into frozen sentence-
embedding space, compared with the candidate label's embedding by cosine.  It reaches an unseen
label only when that label's embedding happens to sit near a training label's, which is why a
foreign vocabulary scores at chance.

This module keeps the comparison but changes what is compared.  A fixed vocabulary of
**primitives** — grouped into mutually exclusive **axes**, each primitive carrying one sentence —
is embedded once with the same frozen sentence encoder.  The sensor side learns to decompose the
pooled recording into a distribution over each axis.  A candidate is then scored by how well its
label embedding agrees with the primitives the recording expressed.  An unseen label needs no
annotation and no training: it only needs to be *describable* by the same primitives.

Two facts measured on the label side before any of this was trained (2026-09-20 audit):

* sentence similarity is a poor judge of attribute entailment — against hand-written expectations
  the per-axis argmax of ``cos(label, primitive sentence)`` agreed 52% of the time for abstract
  sentences and 50% for concrete embodied ones, against roughly 30% chance;
* rewriting the sentences to *name exemplar activities* lifted that to 78%, i.e. essentially all
  of the signal available to a sentence encoder here is label-to-label similarity.

Naming activities in the vocabulary would encode our evaluation vocabulary into a supposedly fixed
artefact, so the sentences here deliberately name none.  Instead the fixed cosine is treated as an
*initialisation* of a learnable compatibility function (the DeViSE/ALE family) rather than as the
truth: ``combiner="projection"`` learns a single shared linear map applied identically to primitive
sentences and to candidate labels.

Why that is safe where the abandoned contextual lineage was not:

* the projection is **shared**: it applies the same learned metric to training and unseen labels.
  This preserves an out-of-vocabulary compatibility contract, but does not itself rule out
  overfitting; that needs held-out-label validation. A per-primitive free value vector would add a
  separate failure mode, which is why the value bank is frozen and never a parameter;
* it is **identity-initialised**, so step 0 is exactly the fixed cosine design;
* the primitive bottleneck stays, so a uniform axis is algebraically neutral across candidates
  (see ``agreement``). This is not, by itself, evidence about sensor observability;
* the **scrambled-sentence control** is registered as its own vocabulary version. It tests how
  primitive sentence grouping contributes, but retains the same meaningful sentences and cannot
  alone establish a capacity-versus-grounding conclusion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- vocabulary
# Changing ANY sentence, axis or ordering below is a new vocabulary version: checkpoints and
# results are keyed on `PRIMITIVE_VOCABULARY_VERSION` and on the hash of this exact text.
PRIMITIVE_VOCABULARY_VERSION = "primitives-v1"

# Global axes describe the whole recording once.  Region axes say which parts of the body move and
# how; they deliberately share one three-value menu and parallel wording, so the region noun is
# what distinguishes them.  Within an axis the sentences share a frame and differ in one idea.
# No sentence names an activity: the vocabulary must not encode the evaluation's label set.
PRIMITIVE_VOCABULARY_V1: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("intensity", (
        ("still", "The person is at rest and hardly moving at all."),
        ("light", "The person is moving gently with little effort."),
        ("moderate", "The person is moving at a comfortable, moderate effort."),
        ("vigorous", "The person is moving hard and strenuously, exerting a lot of effort."),
    )),
    ("rhythm", (
        ("none", "The movement does not repeat; it is a one-off or continuously changing motion."),
        ("slow", "The person repeats a slow, unhurried motion over and over."),
        ("stepping", "The person repeats a regular stepping motion, about twice each second."),
        ("fast", "The person repeats a quick motion many times a second."),
    )),
    ("impact", (
        ("none", "There are no jolts or landings; the movement is smooth."),
        ("light", "There are light, regular footfalls or taps."),
        ("hard", "There are hard landings or heavy jolts on each contact."),
    )),
    ("travel", (
        ("stationary", "The person stays in one place."),
        ("travelling", "The person moves from one place to another across the ground."),
    )),
    ("posture", (
        ("upright", "The person is upright, with the torso vertical."),
        ("horizontal", "The person is horizontal, with the body flat and reclined."),
        ("bent_over", "The person is bent or leaning forward, crouched or stooped."),
        ("changing", "The person changes posture, moving between lower and upper positions."),
    )),
    ("regularity", (
        ("steady", "The movement is steady and repeats the same way each time."),
        ("varied", "The movement is irregular and keeps changing from one moment to the next."),
        ("single_burst", "There is one brief movement, followed by stillness."),
    )),
    ("upper_limbs", (
        ("still", "The arms and hands are still."),
        ("rhythmic", "The arms or hands move in a regular, repeated pattern."),
        ("irregular",
         "The arms or hands move in varied, non-repeating ways, handling or reaching for things."),
    )),
    ("lower_limbs", (
        ("still", "The legs and feet are still."),
        ("rhythmic", "The legs move in a regular, repeated alternating pattern."),
        ("irregular", "The legs move in varied, non-repeating ways, shifting and stepping about."),
    )),
    ("trunk", (
        ("still", "The torso stays still."),
        ("rhythmic", "The torso moves in a regular, repeated pattern, rising, falling or twisting."),
        ("irregular", "The torso moves in varied, non-repeating ways, turning or leaning as needed."),
    )),
    ("head", (
        ("still", "The head is held still."),
        ("rhythmic", "The head moves in a regular, repeated pattern, nodding or bobbing."),
        ("irregular", "The head turns and moves in varied, non-repeating ways."),
    )),
)


def _scrambled(version: str, seed: int = 0):
    """Same sentences, reassigned to primitives at random: the grounding control."""
    import random

    sentences = [sentence for _, values in VOCABULARIES[version] for _, sentence in values]
    random.Random(seed).shuffle(sentences)
    index, axes = 0, []
    for axis, values in VOCABULARIES[version]:
        rebuilt = []
        for name, _ in values:
            rebuilt.append((name, sentences[index]))
            index += 1
        axes.append((axis, tuple(rebuilt)))
    return tuple(axes)


VOCABULARIES: dict[str, tuple] = {PRIMITIVE_VOCABULARY_VERSION: PRIMITIVE_VOCABULARY_V1}
SCRAMBLED_VERSION = f"{PRIMITIVE_VOCABULARY_VERSION}-scrambled"
VOCABULARIES[SCRAMBLED_VERSION] = _scrambled(PRIMITIVE_VOCABULARY_VERSION)


def axis_names(version: str = PRIMITIVE_VOCABULARY_VERSION) -> tuple[str, ...]:
    return tuple(axis for axis, _ in VOCABULARIES[version])


def axis_sizes(version: str = PRIMITIVE_VOCABULARY_VERSION) -> tuple[int, ...]:
    return tuple(len(values) for _, values in VOCABULARIES[version])


def primitive_names(version: str = PRIMITIVE_VOCABULARY_VERSION) -> tuple[str, ...]:
    return tuple(f"{axis}/{name}" for axis, values in VOCABULARIES[version] for name, _ in values)


def primitive_sentences(version: str = PRIMITIVE_VOCABULARY_VERSION) -> tuple[str, ...]:
    return tuple(sentence for _, values in VOCABULARIES[version] for _, sentence in values)


def n_primitives(version: str = PRIMITIVE_VOCABULARY_VERSION) -> int:
    return sum(axis_sizes(version))


def axis_slices(version: str = PRIMITIVE_VOCABULARY_VERSION) -> tuple[slice, ...]:
    out, start = [], 0
    for size in axis_sizes(version):
        out.append(slice(start, start + size))
        start += size
    return tuple(out)


def vocabulary_hash(version: str = PRIMITIVE_VOCABULARY_VERSION) -> str:
    """Hash of the exact text, so an edited sentence cannot silently reuse a version name."""
    digest = hashlib.sha256()
    for axis, values in VOCABULARIES[version]:
        digest.update(axis.encode())
        for name, sentence in values:
            digest.update(name.encode())
            digest.update(sentence.encode())
    return digest.hexdigest()


def primitive_value_matrix(version: str = PRIMITIVE_VOCABULARY_VERSION, device=None) -> torch.Tensor:
    """Frozen sentence embeddings of the vocabulary, from the repo's one text encoder."""
    from training.support_classifier.train import label_text_matrix

    return label_text_matrix(list(primitive_sentences(version)), device)


# --------------------------------------------------------------------------- head
COMBINERS = ("fixed", "projection")


@dataclass(frozen=True)
class PrimitiveSemanticConfig:
    version: str = PRIMITIVE_VOCABULARY_VERSION
    text_dim: int = 384
    combiner: str = "projection"
    # Rank of the shared compatibility projection. Full rank is identity-initialised and already
    # capacity-limited by being shared across every label; a smaller rank limits it further.
    projection_rank: int = 384
    # Temperature of the label-side softmax within an axis, and the scale of the final score.
    profile_temperature: float = 0.05
    logit_scale: float = 10.0

    def __post_init__(self) -> None:
        if self.version not in VOCABULARIES:
            raise ValueError(f"unknown primitive vocabulary {self.version!r}")
        if self.combiner not in COMBINERS:
            raise ValueError(f"combiner must be one of {COMBINERS}")
        if not 1 <= self.projection_rank <= self.text_dim:
            raise ValueError("projection rank must be in [1, text_dim]")
        if self.profile_temperature <= 0 or self.logit_scale <= 0:
            raise ValueError("temperature and logit scale must be positive")


class PrimitiveSemanticHead(nn.Module):
    """Pooled sensor vector + candidate label embeddings -> candidate log-probabilities.

    Takes no input the promoted head does not already receive: the label side is the same frozen
    ``candidate_text`` tensor, so label-text corruption permutes the primitive profiles with it and
    the two semantic paths can never disagree about which roster they are scoring.
    """

    def __init__(self, d_model: int, cfg: PrimitiveSemanticConfig | None = None,
                 values: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg or PrimitiveSemanticConfig()
        self.slices = axis_slices(self.cfg.version)
        self.axes = axis_names(self.cfg.version)
        k = n_primitives(self.cfg.version)
        if values is None:
            values = primitive_value_matrix(self.cfg.version)
        if values.shape != (k, self.cfg.text_dim):
            raise ValueError("primitive value bank does not match the vocabulary")
        # Frozen: a trainable value bank would drift into a lookup over the training vocabulary,
        # which is the failure this whole design is avoiding.
        self.register_buffer("values", F.normalize(values.float(), dim=-1), persistent=True)
        self.keys = nn.Linear(d_model, k)
        if self.cfg.combiner == "projection":
            # ONE shared map, applied identically to primitive sentences and candidate labels.
            # This preserves the same transformation contract for unseen labels; it does not by
            # itself prevent overfitting, which is tested with held-out-label development data.
            # Identity-initialised: step 0 is the fixed design.
            weight = torch.zeros(self.cfg.projection_rank, self.cfg.text_dim)
            torch.nn.init.eye_(weight)
            self.projection = nn.Parameter(weight)
        else:
            # Keep the historical state key for checkpoint compatibility, but make the fixed
            # comparison genuinely fixed and normalized rather than an unconstrained side path.
            self.register_buffer("axis_weight", torch.ones(len(self.slices)), persistent=True)

    # ------------------------------------------------------------------ pieces
    def profile(self, query_feature: torch.Tensor) -> torch.Tensor:
        """Per-axis distributions over primitives, from the pooled recording vector alone."""
        logits = self.keys(query_feature.float())
        profile = torch.empty_like(logits)
        for block in self.slices:
            profile[:, block] = torch.softmax(logits[:, block], dim=-1)
        return profile

    def _compatibility(self, candidate_text: torch.Tensor) -> torch.Tensor:
        """(B, C, K) compatibility between each candidate label and each primitive."""
        labels = F.normalize(candidate_text.float(), dim=-1)
        if self.cfg.combiner == "fixed":
            return labels @ self.values.T
        projected_values = F.normalize(self.values @ self.projection.T, dim=-1)
        projected_labels = F.normalize(labels @ self.projection.T, dim=-1)
        return projected_labels @ projected_values.T

    def candidate_profile(self, candidate_text: torch.Tensor,
                          candidate_mask: torch.Tensor) -> torch.Tensor:
        """The label side, softmaxed within each axis: a fixed function of the label string."""
        compatibility = self._compatibility(candidate_text)
        profile = torch.empty_like(compatibility)
        for block in self.slices:
            profile[..., block] = torch.softmax(
                compatibility[..., block] / self.cfg.profile_temperature, dim=-1,
            )
        return torch.where(candidate_mask.unsqueeze(-1), profile, torch.zeros_like(profile))

    def agreement(self, profile: torch.Tensor, candidate_profile: torch.Tensor) -> torch.Tensor:
        """Per-axis dot product of two distributions.

        Averaged over axes, so the result is in [0, 1] regardless of how many axes a vocabulary
        has and ``logit_scale`` means the same thing across vocabulary versions.

        A uniform sensor block contributes ``1/V`` to EVERY candidate and therefore cancels in the
        softmax over candidates. That is what makes observability implied: a head that cannot see a
        region from the devices present may output a uniform block for it and pay nothing, while a
        confident wrong block costs. No mask is declared and none is learned.
        """
        if self.cfg.combiner == "fixed":
            terms = [torch.einsum("bv,bcv->bc", profile[:, block], candidate_profile[..., block])
                     for block in self.slices]
            return torch.stack(terms, dim=0).mean(dim=0)
        return torch.einsum("bk,bck->bc", profile, candidate_profile) / len(self.slices)

    def forward(self, query_feature: torch.Tensor, candidate_text: torch.Tensor,
                candidate_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if candidate_text.shape[:2] != candidate_mask.shape:
            raise ValueError("candidate text and mask disagree on batch or roster size")
        if candidate_text.shape[-1] != self.cfg.text_dim:
            raise ValueError("candidate text width does not match the vocabulary's text dimension")
        safe_candidate_text = torch.where(
            candidate_mask.unsqueeze(-1), candidate_text, torch.zeros_like(candidate_text),
        )
        profile = self.profile(query_feature)
        candidate = self.candidate_profile(safe_candidate_text, candidate_mask)
        score = self.cfg.logit_scale * self.agreement(profile, candidate)
        logits = torch.log_softmax(score.masked_fill(~candidate_mask, float("-inf")), dim=-1)
        entropy = torch.stack([
            -(profile[:, block].clamp_min(1e-12).log() * profile[:, block]).sum(dim=-1)
            for block in self.slices
        ], dim=-1)
        return {"logits": logits, "primitive_profile": profile,
                "candidate_primitive_profile": candidate, "axis_entropy": entropy}

    # ------------------------------------------------------------------ reporting
    def telemetry(self) -> dict[str, float]:
        if self.cfg.combiner == "projection":
            weight = self.projection.detach()
            eye = torch.eye(weight.shape[0], weight.shape[1], device=weight.device)
            return {"classifier/primitive_projection_drift": float((weight - eye).abs().mean())}
        return {}

    @property
    def provenance(self) -> dict:
        return {"primitive_vocabulary_version": self.cfg.version,
                "primitive_vocabulary_hash": vocabulary_hash(self.cfg.version),
                "primitive_combiner": self.cfg.combiner,
                "primitive_projection_rank": self.cfg.projection_rank}
