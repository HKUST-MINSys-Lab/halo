"""Shared frozen text encoder used by the retained support-classifier control."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

_SBERT_CACHE: dict[str, object] = {}
_EMBEDDING_CACHE: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}


def get_sbert_encoder(model_name: str = "all-MiniLM-L6-v2") -> Callable[[Sequence[str]], np.ndarray]:
    """Return the shared, normalized MiniLM label-text encoder."""
    if model_name not in _SBERT_CACHE:
        from sentence_transformers import SentenceTransformer
        _SBERT_CACHE[model_name] = SentenceTransformer(model_name)
    encoder = _SBERT_CACHE[model_name]

    def encode(labels: Sequence[str]) -> np.ndarray:
        key = (model_name, tuple(labels))
        if key not in _EMBEDDING_CACHE:
            text = [label.replace("_", " ") for label in labels]
            _EMBEDDING_CACHE[key] = np.asarray(
                encoder.encode(text, normalize_embeddings=True), dtype=np.float32
            )
        return _EMBEDDING_CACHE[key]

    return encode
