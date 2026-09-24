"""Result-row validation, atomic JSON, and run provenance shared by every rung.

Extracted verbatim from training/support_classifier/sealed_eval.py on 2026-09-23 (Phase 0 of
docs/journal/2026-09-22-rung1-rung2-implementation-plan.md). sealed_eval re-imports these names.
"""

from __future__ import annotations

import baselines
import json
import numpy as np
import os
import platform
import subprocess
import torch
from pathlib import Path
from typing import Sequence
from evaluation.features import FEATURE_CACHE_SCHEMA, _file_hash


def validate_result_rows(
    rows: Sequence[dict],
    *,
    expected_cells: Sequence[tuple[float, str, str]],
    models: Sequence[str],
    k_values: Sequence[int],
) -> None:
    """Reject incomplete or internally inconsistent result artifacts before publication."""
    if not rows:
        raise RuntimeError("sealed evaluation produced no rows")
    required = {
        "model", "readout", "window_seconds", "k", "dataset", "stream", "status",
        "parameters_m", "native_open_set_labels", "native_support_conditioning",
        "published_few_label_finetuning", "padded", "padded_fraction",
    }
    coverage: set[tuple[float, str, str, str, int]] = set()
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"result row {index} is missing fields: {missing}")
        if row["model"] == "harnet":
            raise RuntimeError("ambiguous model identity 'harnet'; use harnet5 or harnet10")
        if row["status"] not in {"ok", "n/a"}:
            raise RuntimeError(f"result row {index} has non-publishable status {row['status']!r}")
        if not np.isfinite(float(row["padded_fraction"])) or float(row["padded_fraction"]) < 0:
            raise RuntimeError(f"result row {index} has invalid padding accounting")
        if row["status"] == "ok":
            for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
                if metric in row and not np.isfinite(float(row[metric])):
                    raise RuntimeError(f"result row {index} has non-finite {metric}")
        coverage.add((float(row["window_seconds"]), str(row["dataset"]), str(row["stream"]),
                      str(row["model"]), int(row["k"])))
    expected = {
        (float(duration), dataset, stream, model, int(k))
        for duration, dataset, stream in expected_cells
        for model in models for k in k_values
    }
    missing_cells = sorted(expected - coverage)
    if missing_cells:
        preview = missing_cells[:5]
        raise RuntimeError(
            f"sealed evaluation is partial: {len(missing_cells)} model/cell/k combinations missing; "
            f"first={preview}"
        )


def _atomic_json(path: Path, value: object) -> None:
    """Write evaluator state without exposing a partial JSON document to monitors."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _run_provenance(argv: list[str], *, device: torch.device, halo_checkpoint: Path | None) -> dict:
    """Persist the evaluator revision and model identity beside every sealed result."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1"], text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, None
    checkpoint = halo_checkpoint.resolve() if halo_checkpoint is not None else None
    return {
        "protocol": "sealed-support-conditioned-v3-20260920",
        "argv": argv,
        "git_revision": revision,
        "dirty_worktree": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "halo_checkpoint": str(checkpoint) if checkpoint else None,
        "halo_checkpoint_sha256": _file_hash(checkpoint) if checkpoint else None,
        "baseline_adapters": {
            name: type(baselines.REGISTRY[name]).__qualname__
            for name in sorted(baselines.REGISTRY)
        },
        "feature_cache_schema": FEATURE_CACHE_SCHEMA,
    }


# ---------------------------------------------------------------------------------------------
# Rung / method / readout-version registry (added 2026-09-23; docs/overview/roadmap.md, code plan
# rule 1). An artifact cannot be written without declaring all three, so a results file can never
# be mistaken for a different protocol. Rung 2 keeps its existing runner and adopts this at
# migration time.
# ---------------------------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402
from enum import Enum, IntEnum  # noqa: E402


class Rung(IntEnum):
    # Numbering of 2026-09-23: the discovery readout (roster and K unknown) was dropped from the plan
    # as not interesting enough; it keeps value 0 so the retired code still runs and can never be
    # mistaken for a rung of the paper.
    DISCOVERY = 0      # retired; roster and K unknown
    UNLABELED = 1      # roster known, unlabelled pool grows, no labels ever
    FROZEN = 2         # k labels, parameters frozen (the existing sealed/scenario tables)
    FINETUNE = 3       # k labels, fine-tuning allowed


class Method(str, Enum):
    # discovery (retired)
    KMEANS_PP_10 = "kmeans_pp_10"
    WARD = "ward"
    # rung 1
    INDUCTIVE = "inductive"                    # N=0: the published per-window readout
    TRANSDUCTIVE_CLIP_V1 = "transductive_clip_v1"
    # rung 2
    NEIGHBOURS = "neighbours"
    CLASSIFIER_V4 = "classifier_v4"
    # rung 3
    LINEAR_PROBE = "linear_probe"
    SMALL_CLASSIFIER = "small_classifier"
    LORA = "lora"
    FULL_FINETUNE = "full_finetune"
    SCRATCH_SPECIALIST = "scratch_specialist"
    # The rung-2 parameter-free readout re-run on rung 3's own support draw and scored set, so the
    # enrollment-vs-fine-tuning crossover is measured on identical windows. Named distinctly so it
    # is never confused with the sealed table's per-query-manifest rows.
    ENROLLMENT_FROZEN = "enrollment_frozen"


METHODS_BY_RUNG: dict[Rung, frozenset[Method]] = {
    Rung.DISCOVERY: frozenset({Method.KMEANS_PP_10, Method.WARD}),
    Rung.UNLABELED: frozenset({Method.INDUCTIVE, Method.TRANSDUCTIVE_CLIP_V1}),
    Rung.FROZEN: frozenset({Method.NEIGHBOURS, Method.CLASSIFIER_V4}),
    Rung.FINETUNE: frozenset({Method.LINEAR_PROBE, Method.SMALL_CLASSIFIER, Method.LORA,
                              Method.FULL_FINETUNE, Method.SCRATCH_SPECIALIST,
                              Method.ENROLLMENT_FROZEN}),
}

# Bump when the procedure changes, as sealed-manifest-v2 / deployment-scenarios-v5 do today.
READOUT_VERSION: dict[Rung, str] = {
    Rung.DISCOVERY: "discovery-v1",
    Rung.UNLABELED: "ncurve-v1",
    Rung.FROZEN: "sealed-manifest-v2",
    Rung.FINETUNE: "finetune-v1",
}


@dataclass(frozen=True)
class ArtifactProvenance:
    """What every rung-1/3 (and retired discovery) artifact must declare before a row can be written.

    ``method`` and ``encoder`` may be fixed for the whole artifact or left ``None``, in which case
    every ``status == "ok"`` row must carry its own, and each is validated against the rung's
    registered methods. Discovery artifacts hold six encoders x two algorithms, so they use the
    per-row form; a single-arm artifact fixes both.
    """

    rung: Rung
    checkpoint_fingerprint: str       # sha256 over the encoder checkpoint(s) / released artifacts
    manifest_fingerprint: str         # sealed manifest / cell construction fingerprint
    method: Method | None = None
    encoder: str | None = None        # "halo" | baseline name | "matched:<backbone>"
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method is not None and self.method not in METHODS_BY_RUNG[self.rung]:
            raise ValueError(f"{self.method.value} is not a registered method for rung {int(self.rung)}")
        for name in ("checkpoint_fingerprint", "manifest_fingerprint"):
            if not getattr(self, name):
                raise ValueError(f"artifact provenance requires a non-empty {name}")
        if self.encoder is not None and not self.encoder:
            raise ValueError("encoder must be non-empty when fixed at the artifact level")

    @property
    def readout_version(self) -> str:
        return READOUT_VERSION[self.rung]

    def stamp(self, row: dict) -> dict:
        """Return a copy of ``row`` carrying the registry fields; existing keys must agree.

        Rows whose ``status`` is not ``"ok"`` (unsupported cells) are stamped with the rung and
        readout version only; they need no method because nothing was computed.
        """
        out = {**row, "rung": int(self.rung), "readout_version": self.readout_version}
        if row.get("status", "ok") != "ok":
            return out
        allowed = {m.value for m in METHODS_BY_RUNG[self.rung]}
        for key, fixed in (("method", None if self.method is None else self.method.value),
                           ("encoder", self.encoder)):
            value = row.get(key)
            if fixed is not None:
                if value is not None and value != fixed:
                    raise ValueError(f"row {key}={value!r} disagrees with provenance {fixed!r}")
                out[key] = fixed
            elif not value:
                raise ValueError(f"row must declare {key!r} when it is not fixed at the artifact level")
        if out["method"] not in allowed:
            raise ValueError(f"{out['method']!r} is not a registered method for rung {int(self.rung)}")
        return out

    def as_dict(self) -> dict:
        return {"rung": int(self.rung), "readout_version": self.readout_version,
                "method": None if self.method is None else self.method.value, "encoder": self.encoder,
                "checkpoint_fingerprint": self.checkpoint_fingerprint,
                "manifest_fingerprint": self.manifest_fingerprint, **self.extra}


def write_artifact(out_dir: Path, rows: Sequence[dict], provenance: ArtifactProvenance, *,
                   argv: Sequence[str], device: torch.device,
                   halo_checkpoint: Path | None = None, name: str = "results") -> Path:
    """Write ``<name>.json`` with every row stamped, beside ``run_provenance.json``.

    Refuses rows that disagree with the declared provenance. ``_run_provenance`` (the existing
    sealed-eval record: git commit, platform, torch, checkpoint hash) is extended, not replaced.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped = [provenance.stamp(dict(row)) for row in rows]
    _atomic_json(out_dir / f"{name}.json", stamped)
    record = _run_provenance(list(argv), device=device, halo_checkpoint=halo_checkpoint)
    record["artifact"] = {
        **provenance.as_dict(),
        "methods_present": sorted({r["method"] for r in stamped if "method" in r}),
        "encoders_present": sorted({r["encoder"] for r in stamped if "encoder" in r}),
        "rows": len(stamped),
    }
    _atomic_json(out_dir / "run_provenance.json", record)
    return out_dir / f"{name}.json"
