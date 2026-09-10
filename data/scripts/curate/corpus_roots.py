"""Where a dataset's converted sessions and grids live.

HALO keeps source material in two sibling trees under ``data/``:

``datasets/``
    Labelled sources. Every activity-recognition corpus used for evaluation probes, the
    Phase-B evidence bank, and the application tasks. A session here has real activity
    annotations in ``labels.json``.

``pretraining/``
    Label-free scale sources for representation pretraining ONLY. Every session carries the
    reserved ``__unlabeled__`` marker, so these sources can never contribute to label
    vocabulary construction, validation probes, or the evidence bank — the existing
    ``__unlabeled__`` guards already enforce that, and the directory split makes the
    intent visible on disk instead of only in a roster tuple.

The split is organizational. Nothing about the on-disk contract differs between the two
trees: the same converters, the same ``sessions/<id>/data.parquet``, the same grid layout.
Code should therefore resolve a dataset by NAME through :func:`dataset_root` rather than
joining ``data/datasets`` itself, so a source can move between trees without a code change.

A name must not exist in both trees. :func:`dataset_root` raises on ambiguity rather than
silently preferring one, because a half-finished move that leaves two copies would
otherwise train on whichever one happened to sort first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

REPO = Path(__file__).resolve().parents[3]

LABELLED_ROOT = REPO / "data" / "datasets"
PRETRAIN_ROOT = REPO / "data" / "pretraining"

#: Search order for :func:`dataset_root`. Labelled first only so the common case resolves
#: in one stat call; ambiguity is an error, not a precedence rule.
CORPUS_ROOTS: Tuple[Path, ...] = (LABELLED_ROOT, PRETRAIN_ROOT)

#: Reserved activity marker carried by every session in ``pretraining/``.
UNLABELED = "__unlabeled__"


def _is_dataset_dir(path: Path) -> bool:
    """A dataset directory is one that declares itself with a ``metadata.json``.

    Checking for the declaration rather than mere directory existence keeps ``__pycache__``,
    scratch folders, and a bare ``downloads/`` left by an interrupted fetch out of the roster.
    """
    return path.is_dir() and (path / "metadata.json").exists()


def roots_under(repo: Path) -> Tuple[Path, ...]:
    """The corpus roots of an arbitrary repository tree.

    Callers that can be pointed at a different tree (``build_grids`` under test) derive their
    roots from their own ``REPO`` and pass them in, so root resolution keeps one implementation
    instead of each caller reinventing the search and the ambiguity rule.
    """
    return (repo / "data" / "datasets", repo / "data" / "pretraining")


def dataset_root(
    dataset: str,
    *,
    must_exist: bool = True,
    roots: Optional[Tuple[Path, ...]] = None,
) -> Path:
    """Return the directory holding ``dataset``'s sessions, grids, and metadata.

    ``must_exist=False`` returns the path a NEW dataset would occupy (in the labelled tree)
    when it is present in neither, which is what a converter writing its first output needs.
    ``roots`` overrides the search path; see :func:`roots_under`.
    """
    search = CORPUS_ROOTS if roots is None else roots
    found = [root / dataset for root in search if _is_dataset_dir(root / dataset)]
    if len(found) > 1:
        locations = ", ".join(str(path) for path in found)
        raise ValueError(
            f"dataset {dataset!r} exists in more than one corpus root ({locations}). "
            "A source lives in exactly one tree; remove the stale copy."
        )
    if found:
        return found[0]
    if must_exist:
        names = ", ".join(str(root) for root in search)
        raise FileNotFoundError(
            f"no dataset {dataset!r} under any corpus root ({names}). Run its converter first."
        )
    return search[0] / dataset


def is_pretraining(dataset: str) -> bool:
    """Whether ``dataset`` lives in the label-free pretraining tree."""
    return _is_dataset_dir(PRETRAIN_ROOT / dataset)


def dataset_names(root: Optional[Path] = None) -> Tuple[str, ...]:
    """Every declared dataset name, optionally restricted to one root."""
    roots = (root,) if root is not None else CORPUS_ROOTS
    names = {
        path.name
        for search in roots
        if search.exists()
        for path in sorted(search.iterdir())
        if _is_dataset_dir(path)
    }
    return tuple(sorted(names))


def iter_dataset_dirs(root: Optional[Path] = None) -> Iterator[Path]:
    """Yield every dataset directory across the corpus roots."""
    for name in dataset_names(root):
        yield dataset_root(name)


def grid_search_roots() -> Tuple[Path, ...]:
    """Roots a grid discovery glob must cover (existing ones only)."""
    return tuple(root for root in CORPUS_ROOTS if root.exists())
