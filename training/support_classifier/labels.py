"""Episode-local label names for bounded support-classification episodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


_PREFIXES = (
    "routine", "protocol", "sequence", "exercise", "pattern", "motion", "task", "drill",
)
_CODEWORDS = (
    "amber", "birch", "cobalt", "delta", "ember", "fjord", "garnet", "harbor",
    "indigo", "juniper", "kelp", "lilac", "marble", "nectar", "opal", "pearl",
    "quartz", "raven", "silver", "topaz", "umber", "violet", "willow", "xenon",
    "yarrow", "zephyr", "acorn", "brisk", "cinder", "dawn", "elm", "flint",
)


@dataclass(frozen=True)
class EpisodeLabelSet:
    """Text presented to candidates and provided support in one episode."""

    mode: str
    phrases: tuple[str, ...]
    embeddings: torch.Tensor


def neutral_alias_vocabulary() -> tuple[str, ...]:
    """Finite cacheable vocabulary with no intentional activity semantics."""
    return tuple(f"{prefix} {word}" for prefix in _PREFIXES for word in _CODEWORDS)


def encode_neutral_aliases(sbert, device) -> torch.Tensor:
    aliases = neutral_alias_vocabulary()
    return torch.from_numpy(sbert(list(aliases)).astype(np.float32)).to(device)


def episode_label_set(
    canonical_ids: torch.Tensor,
    canonical_text: torch.Tensor,
    *,
    mode: str,
    rng: np.random.Generator,
    alias_embeddings: torch.Tensor | None = None,
    canonical_names: list[str] | tuple[str, ...] | None = None,
) -> EpisodeLabelSet:
    """Choose a one-to-one label namespace for an episode.

    Random aliases are meaningful only when support is present. The caller enforces that
    information-theoretic constraint; this function only constructs the namespace.
    """
    canonical_ids = canonical_ids.long().to(canonical_text.device)
    if mode == "coherent":
        phrases = tuple(
            canonical_names[int(value)] if canonical_names is not None else str(int(value))
            for value in canonical_ids.detach().cpu()
        )
        return EpisodeLabelSet(mode, phrases, canonical_text[canonical_ids])
    if mode != "random_alias":
        raise ValueError(f"unknown episode label mode {mode!r}")
    if alias_embeddings is None:
        raise ValueError("random_alias mode requires pre-encoded neutral aliases")
    if len(canonical_ids) > len(alias_embeddings):
        raise ValueError("neutral alias vocabulary is smaller than the candidate set")
    chosen = rng.choice(len(alias_embeddings), size=len(canonical_ids), replace=False)
    aliases = neutral_alias_vocabulary()
    return EpisodeLabelSet(
        mode,
        tuple(aliases[int(index)] for index in chosen),
        alias_embeddings[torch.as_tensor(chosen, device=alias_embeddings.device)],
    )
