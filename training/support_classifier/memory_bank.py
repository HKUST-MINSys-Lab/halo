"""Bounded deployment-local memory with strict predict-before-insert semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch.nn import functional as F

from model.support.memory_classifier import MemoryReadout, MemoryReaderClassifier


@dataclass(frozen=True)
class MemoryEntry:
    recording_id: str
    execution_id: str
    observation_index: int
    motion: torch.Tensor
    acquisition: torch.Tensor
    original_probabilities: torch.Tensor
    original_roster: tuple[str, ...]
    model_version: str
    verified_label: str | None = None
    acquisition_metadata: Mapping[str, str | float | bool] | None = None


class DeploymentMemory:
    """One deployment only; no hidden truth is ever inferred from an unlabelled insert."""

    def __init__(self, model_version: str, *, capacity: int = 64):
        if not model_version or capacity < 1:
            raise ValueError("model version and positive capacity are required")
        self.model_version = model_version
        self.capacity = capacity
        self.seen = 0
        self.entries: list[MemoryEntry] = []

    def _evict(self) -> None:
        if len(self.entries) <= self.capacity:
            return
        unlabelled = [e for e in self.entries if e.verified_label is None]
        if not unlabelled:
            raise ValueError("verified enrollments exceed memory capacity")
        # Keep recent samples, but preferentially evict a redundant nearby motion/acquisition
        # sample. No pseudo-class quotas: an incorrect prediction must not control retention.
        removable = sorted(unlabelled, key=lambda e: e.observation_index)[:max(1, len(unlabelled) // 2)]
        vectors = torch.stack([F.normalize(torch.cat((e.motion, e.acquisition)).float(), dim=0)
                               for e in self.entries])
        similarity = vectors @ vectors.T
        similarity.fill_diagonal_(-1)
        eviction = max(removable, key=lambda e: (
            float(similarity[self.entries.index(e)].max()), -e.observation_index,
        ))
        self.entries = [entry for entry in self.entries if entry is not eviction]

    def insert(
        self, *, recording_id: str, execution_id: str, motion: torch.Tensor,
        acquisition: torch.Tensor, original_probabilities: torch.Tensor,
        roster: tuple[str, ...], verified_label: str | None = None,
        acquisition_metadata: Mapping[str, str | float | bool] | None = None,
    ) -> MemoryEntry:
        if not recording_id or not execution_id:
            raise ValueError("recording and execution identities are required")
        if len(roster) < 2 or len(set(roster)) != len(roster):
            raise ValueError("the candidate roster must contain distinct labels")
        if original_probabilities.shape != (len(roster),) or not bool(
            torch.isfinite(original_probabilities).all()
        ) or bool((original_probabilities < 0).any()) or not torch.isclose(
            original_probabilities.sum(), original_probabilities.new_tensor(1.0), atol=1e-4
        ):
            raise ValueError("original probabilities must be a normalized roster distribution")
        if any(e.recording_id == recording_id or e.execution_id == execution_id
               for e in self.entries):
            raise ValueError("duplicate recording or execution in one deployment bank")
        full_of_verified = (len(self.entries) >= self.capacity
                            and all(e.verified_label is not None for e in self.entries))
        if full_of_verified and verified_label is not None:
            raise ValueError("verified enrollments already fill memory capacity")
        entry = MemoryEntry(
            recording_id, execution_id, self.seen, motion, acquisition,
            original_probabilities.detach(), tuple(roster), self.model_version,
            verified_label, dict(acquisition_metadata or {}),
        )
        self.seen += 1
        if full_of_verified:
            return entry  # Observed, but not retained; verified evidence stays protected.
        self.entries.append(entry)
        self._evict()
        return entry

    def verify(self, recording_id: str, label: str) -> None:
        for i, entry in enumerate(self.entries):
            if entry.recording_id == recording_id:
                self.entries[i] = replace(entry, verified_label=label)
                return
        raise KeyError(recording_id)

    def state_dict(self) -> dict:
        """Portable snapshot; tensor values are detached because deployment never updates weights."""
        return {
            "schema": "deployment-memory-v5-1", "model_version": self.model_version,
            "capacity": self.capacity, "seen": self.seen,
            "entries": [{
                "recording_id": entry.recording_id,
                "execution_id": entry.execution_id,
                "observation_index": entry.observation_index,
                "motion": entry.motion.detach().cpu(),
                "acquisition": entry.acquisition.detach().cpu(),
                "original_probabilities": entry.original_probabilities.detach().cpu(),
                "original_roster": entry.original_roster,
                "model_version": entry.model_version,
                "verified_label": entry.verified_label,
                "acquisition_metadata": dict(entry.acquisition_metadata or {}),
            } for entry in self.entries],
        }

    @classmethod
    def from_state_dict(
        cls, state: dict, *, model_version: str, device: torch.device,
    ) -> "DeploymentMemory":
        if state.get("schema") != "deployment-memory-v5-1" or \
                state.get("model_version") != model_version:
            raise ValueError("memory schema or model version does not match the reader")
        bank = cls(model_version, capacity=int(state["capacity"]))
        bank.seen = int(state["seen"])
        if bank.seen < 0 or len(state["entries"]) > bank.capacity:
            raise ValueError("invalid memory snapshot count or capacity")
        recording_ids: set[str] = set()
        execution_ids: set[str] = set()
        previous_index = -1
        for raw in state["entries"]:
            if raw["model_version"] != model_version:
                raise ValueError("memory snapshot mixes model versions")
            index = int(raw["observation_index"])
            if index <= previous_index or index >= bank.seen:
                raise ValueError("memory observation indices are not causal and unique")
            previous_index = index
            if raw["recording_id"] in recording_ids or raw["execution_id"] in execution_ids:
                raise ValueError("memory snapshot contains duplicate recordings/executions")
            recording_ids.add(raw["recording_id"])
            execution_ids.add(raw["execution_id"])
            bank.entries.append(MemoryEntry(
                recording_id=str(raw["recording_id"]),
                execution_id=str(raw["execution_id"]),
                observation_index=index,
                motion=raw["motion"].to(device),
                acquisition=raw["acquisition"].to(device),
                original_probabilities=raw["original_probabilities"].to(device),
                original_roster=tuple(raw["original_roster"]),
                model_version=model_version,
                verified_label=raw.get("verified_label"),
                acquisition_metadata=dict(raw.get("acquisition_metadata") or {}),
            ))
        return bank

    def tensors(
        self, classifier: MemoryReaderClassifier, labels: tuple[str, ...],
        candidate_text: torch.Tensor, *, empty_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(labels) != candidate_text.shape[0] or len(set(labels)) != len(labels):
            raise ValueError("candidate text and distinct roster labels must match")
        if not self.entries:
            d = classifier.spec.d_model
            return (torch.empty((0, d), device=empty_device),
                    torch.empty((0, d), device=empty_device),
                    torch.empty((0, len(labels)), device=empty_device),
                    torch.empty(0, dtype=torch.long, device=empty_device))
        motion = torch.stack([e.motion for e in self.entries])
        acquisition = torch.stack([e.acquisition for e in self.entries])
        evidence = []
        verified = []
        for entry in self.entries:
            if entry.model_version != self.model_version:
                raise ValueError("stale memory entry belongs to a different model version")
            # A changed roster requires a fresh semantic score; old probabilities are retained
            # only as provenance, never redistributed over a different label set.
            if entry.verified_label is not None and entry.verified_label not in labels:
                prob = entry.original_probabilities.new_zeros(len(labels))
            else:
                prob = (entry.original_probabilities if entry.original_roster == labels else
                        classifier.semantic(entry.motion, candidate_text).detach())
            evidence.append(prob)
            verified.append(labels.index(entry.verified_label)
                            if entry.verified_label in labels else -1)
        return (motion, acquisition, torch.stack(evidence),
                torch.tensor(verified, dtype=torch.long, device=motion.device))

    def predict_then_insert(
        self, classifier: MemoryReaderClassifier, *, recording_id: str,
        execution_id: str, motion: torch.Tensor, acquisition: torch.Tensor,
        labels: tuple[str, ...], candidate_text: torch.Tensor,
        verified_label: str | None = None,
        acquisition_metadata: Mapping[str, str | float | bool] | None = None,
    ) -> MemoryReadout:
        tensors = self.tensors(classifier, labels, candidate_text, empty_device=motion.device)
        readout = classifier(motion, acquisition, candidate_text, *tensors)
        self.insert(
            recording_id=recording_id, execution_id=execution_id, motion=motion,
            acquisition=acquisition, original_probabilities=readout.semantic,
            roster=labels, verified_label=verified_label,
            acquisition_metadata=acquisition_metadata,
        )
        return readout
