"""Single-source Phase-B recipe for the minimal memory-adaptation experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


# These are recipe constants, not independent experiment knobs. The active index is sampled in
# source-window units so a future trainable tokenizer can refresh it without re-encoding thousands
# of unrelated one-off patches.
ARCHIVE_BUDGET_WINDOWS = 250_000
ACTIVE_WINDOWS_PER_LABEL = 16
# How often the retrievable memory bank is redrawn from the reservoir. At 100 a 3000-step run only
# ever made ~13% of the 248k-window reservoir reachable — the rest was encoded, stored and never
# retrievable. At 5 that becomes ~71%. The cadence used to be pinned by cost: redrawing took ~300 ms
# because the sampler rescanned every window once per label. That scan is now precomputed, so a
# refresh costs ~60-90 ms. At the five-step cadence it remains a small part of the optimized
# roughly 19-minute RTX 4090 run while exposing most of the archive over 3,000 steps.
ACTIVE_REFRESH_STEPS = 5
RETRIEVAL_SUBSPACES = 4
RETRIEVAL_SUBSPACE_DIM = 64
RETRIEVAL_PROJECTION_EMA = 0.995
RETRIEVAL_TEMPERATURE = 0.07
RETRIEVAL_OVERSAMPLE = 8
MIN_TOPK_PER_SUBSPACE = 4
MAX_TOPK_PER_SUBSPACE = 32

# Training deliberately has two information regimes. A zero-support draw uses coherent activity
# names and tests semantic prediction. Every positive-k draw assigns fresh, meaningless names to
# the candidates, so the answer can be recovered only by binding retrieved support to its
# episode-local name. Subject identity and configuration never constrain support sampling.
EPISODE_TYPES = (
    "semantic_zero_support",
    "ordinary_few_support",
)

# Inclusive ranges sampled independently for every episode. All candidates in an episode receive
# the same k. This makes k the single, direct description of how many labeled executions per
# candidate are available in memory.
SUPPORT_COUNT_RANGE = (0, 8)
CANDIDATE_COUNT_RANGE = (2, 16)

# The fixed validation canaries enumerate instead of sampling, deliberately: they are the
# checkpoint-selection target and the generalization-gap probe, so they must be byte-identical
# across runs and across checkpoints. Training sampling and validation fixtures are different
# requirements and no longer share a constant.
VALIDATION_SUPPORT_COUNTS = (1, 2, 4, 8)
# Checkpoint curves use one candidate set size so k=1 and k=8 differ only in enrolled support.
VALIDATION_CURVE_CANDIDATE_COUNT = 8

# One optimizer update is an intentionally small collection of adaptation tasks. These
# two values describe compute shape, not model behavior: every episode below gets its own candidate
# set, random support draw, and query executions. Keeping them as
# named recipe constants prevents the old ambiguous `batch` from being mistaken for episode count.
EPISODES_PER_STEP = 8
QUERIES_PER_EPISODE = 8

LABEL_TEXT_MODES = ("coherent", "random_alias")
PHYSICAL_VIEW_MODES = ("clean",)

# Scale of the closed-form retrieval vote. Not on the forward path: it sets the units of the
# untrained retrieval + text-ensemble control that every Phase-B result is quoted against, so the
# control and the learned logits stay numerically comparable.
RETRIEVAL_VOTE_SCALE = 10.0

# External evaluation is deliberately split so repeated development runs do not consume every
# dataset that is supposed to support the final claim. The test roster is only selected explicitly.
PHASE_B_DEV_DATASETS = ("motionsense", "realworld", "shoaib")
PHASE_B_TEST_DATASETS = ("inclusivehar", "usc_had", "tnda_har", "ut_complex")

# Artifact guard for the complete predictor recipe. Bump this whenever a behavioral training path
# changes even if the decoder parameter schema stays compatible.
PHASE_B_TRAINING_REGIME = "episodic_memory_adaptation_v22_alias_bound_support"

# Evaluation artifacts are independently versioned from training artifacts. A training-only change
# must not invalidate a held-out protocol, while a cohort/scoring/runtime-memory change must never
# be allowed into a comparison table under an old protocol name.
PHASE_B_EVALUATION_REGIME = "enrollment_adaptation_v2_paired_execution_protocol"

# Fine-tuning constants. The online tokenizer uses a deliberately small LR, while an EMA copy
# supplies stable retrieval keys and the inference representation.
TOKENIZER_LR_SCALE = 0.05
TOKENIZER_EMA_DECAY = 0.999
TOKENIZER_FINETUNE_WARMUP_STEPS = 100


@dataclass(frozen=True)
class PhaseBPolicy:
    """The only public retrieval-capacity choice is the final evidence budget."""

    evidence_budget: int = 64
    tokenizer_mode: str = "frozen"

    def __post_init__(self) -> None:
        if self.evidence_budget < 8:
            raise ValueError("evidence_budget must be at least 8")
        if self.tokenizer_mode not in {"frozen", "ema_finetune"}:
            raise ValueError("tokenizer_mode must be 'frozen' or 'ema_finetune'")

    def topk_per_subspace(
        self, query_patches: int, *, evidence_budget: int | None = None
    ) -> int:
        """Derive selector K for the deployment or explicitly expanded training budget."""
        q = max(1, int(query_patches))
        budget = self.evidence_budget if evidence_budget is None else int(evidence_budget)
        if budget < self.evidence_budget:
            raise ValueError("evidence_budget cannot be below the deployment evidence budget")
        raw = math.ceil(
            RETRIEVAL_OVERSAMPLE * budget / (q * RETRIEVAL_SUBSPACES)
        )
        return max(MIN_TOPK_PER_SUBSPACE, min(MAX_TOPK_PER_SUBSPACE, raw))

    def as_dict(self, *, query_patches: int | None = None) -> dict:
        value = asdict(self)
        value.update({
            "active_windows_per_label": ACTIVE_WINDOWS_PER_LABEL,
            "active_refresh_steps": ACTIVE_REFRESH_STEPS,
            "n_subspaces": RETRIEVAL_SUBSPACES,
            "subspace_dim": RETRIEVAL_SUBSPACE_DIM,
            "subspace_ema": RETRIEVAL_PROJECTION_EMA,
            "max_evidence": self.evidence_budget,
            "tau_retr": RETRIEVAL_TEMPERATURE,
            "dynamic_topk_range": [MIN_TOPK_PER_SUBSPACE, MAX_TOPK_PER_SUBSPACE],
            "training_retrieval": "fixed_and_identical_to_deployment",
        })
        if query_patches is not None:
            value["topk_per_subspace"] = self.topk_per_subspace(query_patches)
        return value
