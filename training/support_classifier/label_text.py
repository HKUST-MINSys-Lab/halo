"""Shared label-text tooling for the support-classification control — the single source of truth for
paraphrase ensembling and (optional) fine-grained descriptions.

Used by the retained support-classification control so candidate labels are anchored in the
same frozen SBERT space (raw canonical name plus optional template/synonym paraphrases).

- ``global_label_paraphrases`` merges every per-dataset synonym/template table into one
  dataset-agnostic pool. This is load-bearing: ``augment_label(label, dataset_name="")``
  falls into a generic fallback that ignores the synonym tables, so the global vocab labels
  (owned by no single dataset) would otherwise collapse to near-duplicate template wraps.
- ``ensemble_text`` returns the (L, 384) L2-normalized mean SBERT anchor per label. Variant
  0 is always the canonical name (so E=1 reproduces the plain-ConSE text); higher E adds
  paraphrase variants; a fine-grained **description** (T2.1), when present, is appended as an
  extra anchor. The description file is optional — absent ⇒ this is a no-op and the mechanism
  reproduces the exact tier-1 text configuration.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
_DESCRIPTIONS = _REPO / "data/labels/label_descriptions.json"
SBERT_DIM = 384


@lru_cache(maxsize=2)
def global_label_paraphrases(train_only: bool = True):
    """(synonyms, templates): merged paraphrase pool from the dataset augmentation configs.

    ``train_only=True`` (DEFAULT, and required for any reported number) restricts the merge to
    TRAINING datasets. ``DATASET_CONFIGS`` contains hand-authored synonym/template tables keyed by
    the HELD-OUT EVAL datasets too (motionsense, realworld, shoaib), and merging those leaked
    eval-specific phrasing — e.g. 'jogging' picked up the motionsense template 'mobile phone sensing
    light running'. That gave HALO target-label text no baseline received, worth +1.4 macro-F1
    measured, while violating our own rule "never fit anything on the candidate labels"
    (EVIDENCE_ENGINE_TIER2.md §1.2 / FINDINGS §6 F1).

    ``train_only=False`` reproduces the old (contaminated) behaviour and exists ONLY so the size of
    that confound stays measurable. Never use it for a reported result.
    """
    from data.scripts.labels.label_augmentation import DATASET_CONFIGS
    allowed = set(DATASET_CONFIGS)
    if train_only:
        from training.tokenizer.pretrain_data import TRAIN_DATASETS
        allowed = set(DATASET_CONFIGS) & set(TRAIN_DATASETS)
    synonyms: dict[str, set] = {}
    templates: set = set()
    for name in sorted(allowed):
        cfg = DATASET_CONFIGS[name]
        for lab, forms in cfg.get("synonyms", {}).items():
            synonyms.setdefault(lab, set()).update(forms)
        templates.update(cfg.get("templates", []))
    return ({k: sorted(v) for k, v in synonyms.items()}, sorted(templates) or ["{}"])


@lru_cache(maxsize=1)
def load_descriptions() -> dict:
    """Optional ``{label -> fine-grained description}`` (T2.1). ``{}`` if the file is absent."""
    if _DESCRIPTIONS.exists():
        return dict(json.loads(_DESCRIPTIONS.read_text()))
    return {}


def label_variant_rows(labels, E: int, seed: int = 0, use_descriptions: bool = True,
                       train_only: bool = True):
    """List of E (+1 if descriptions) rows, each a list of surface strings per label.

    Row 0 = canonical names (``label.replace("_", " ")``). Rows 1..E-1 = ``template.format(synonym)``
    sampled deterministically. A trailing description row is appended for any label that has one.
    """
    syn, templates = global_label_paraphrases(train_only)
    # `train_only` gates the SYNONYM tables but not this file. label_descriptions.json is
    # hand-authored and unscoped, so if it is ever written for eval labels it would put
    # human-written descriptions of the eval vocabulary into the target-label text -- the exact
    # F1 contamination train_only exists to prevent, through a different door. Refuse to apply
    # descriptions when the caller asked for training-only text.
    descs = load_descriptions() if (use_descriptions and not train_only) else {}
    if use_descriptions and train_only and load_descriptions():
        raise ValueError(
            "use_descriptions=True with train_only=True: label_descriptions.json is not scoped to "
            "training datasets, so applying it would reintroduce eval-label text. Either scope the "
            "file and relax this guard, or pass use_descriptions=False.")
    rng = random.Random(seed)
    rows = []
    for e in range(max(1, E)):
        if e == 0:
            rows.append([l.replace("_", " ") for l in labels])
        else:
            rows.append([
                templates[rng.randrange(len(templates))].format(
                    rng.choice(syn.get(l, [l.replace("_", " ")])))
                for l in labels
            ])
    if descs:
        rows.append([descs.get(l, l.replace("_", " ")) for l in labels])
    return rows


def ensemble_text(labels, sbert, E: int = 8, seed: int = 0,
                  use_descriptions: bool = True, train_only: bool = True) -> torch.Tensor:
    """(L, 384) L2-normalized mean SBERT over the paraphrase (+ description) variants.

    NOTE on EVAL parity: ``E=1, use_descriptions=False`` reduces to the bare label string, which is
    exactly what the ConSE baselines embed (``eval/scoring.py``: ``l.replace("_", " ")``). Any
    reported HALO-vs-baseline number must either use that, or apply the same ensemble to every model.
    """
    rows = label_variant_rows(labels, E, seed, use_descriptions, train_only)
    acc = np.zeros((len(labels), SBERT_DIM), dtype=np.float32)
    for r in rows:
        acc += sbert(r).astype(np.float32)
    v = torch.from_numpy(acc / len(rows))
    return F.normalize(v, dim=-1)


def build_label_variants(labels, sbert, K: int = 16, seed: int = 0,
                         use_descriptions: bool = True, train_only: bool = True) -> torch.Tensor:
    """(L, K, 384) L2-normalized SBERT of K *individual* paraphrase variants per label.

    Unlike ``ensemble_text`` (which averages into one anchor), this keeps the variants separate so
    training can sample a DIFFERENT surface form per episode. That is what stops the model from
    memorizing a fixed set of label vectors and forces it to work off text semantics — the
    mechanism we need for genuinely unseen label strings (FINDINGS §2: gains were confined to
    seen-label cells, r=-0.973 against unseen-label fraction). Variant 0 is always the canonical name.
    """
    rows = label_variant_rows(labels, K, seed, use_descriptions, train_only)
    embs = np.stack([sbert(r).astype(np.float32) for r in rows], axis=1)   # (L, K, 384)
    return F.normalize(torch.from_numpy(embs), dim=-1)
