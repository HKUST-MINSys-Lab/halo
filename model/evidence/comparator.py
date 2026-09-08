"""Prediction readout dispatch, including historical comparison checkpoints.

Current staged readouts (2026-09-08): ``dual_attention`` uses separate semantic/enrollment weights
and explicit paired tokens; ``neighbors`` is a parameter-free encoder-training control. Their
implementation lives in ``prediction.py``. The description below applies to legacy readouts only.

WHAT THIS REPLACES
------------------
The Phase-B engine retrieved a top-k slice from a large frozen corpus bank and mixed it. Three
measurements retired that design:

* retrieval ranked by *acquisition configuration*, not activity — same-activity rows from another
  device sat at the 39th percentile, near chance;
* a learned retrieval stage was worth exactly nothing over plain cosine (+0.0000 paired);
* the readout, not the retrieval, carried ~80% of all learning.

So the retrieval stage is gone. The support set is handed to the model — chosen by an explicit
compatibility filter that is a deployment consideration, not a learned quantity — and the
comparator attends over all of it. K is small enough that there is nothing to select.

THE READOUT
-----------
At step 0, candidate ``c`` receives a closed-form weighted vote over support rows::

    score(c) = sum_e  w(query, e, c) * vote(e, c)

Here ``w`` is the fixed softmax of query/support cosine similarity, shared across candidates.
``vote`` is 1 when support row ``e`` is enrolled against candidate ``c`` and otherwise the rectified
cosine between the support row's label text and the candidate text.  The learned set-attention stack
does not replace ``w`` directly: it reads all candidate, query and support tokens and adds one
residual scalar to each candidate's closed-form logit.

There are no per-candidate parameters anywhere. An unseen candidate is scored by the same operation
as a seen one, and permuting candidates permutes the logits.

IDENTITY AT INITIALISATION
--------------------------
``residual_head`` is zero-initialised, so at step 0 the comparator's logits are *exactly* the
closed-form vote over the same support set. That closed form is the untrained floor every result is
quoted against, and the step-0 control depends on the equality being exact rather than approximate.
``tests/test_comparator.py`` asserts it to 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import AttentionSpec, ScaledSum, SetAttentionStack

# Token roles. Every token carries exactly one, added as a direction rather than concatenated.
(ROLE_CANDIDATE, ROLE_QUERY, ROLE_QUERY_DESC,
 ROLE_SUPPORT, ROLE_SUPPORT_DESC, ROLE_SUPPORT_LABEL) = range(6)
N_ROLES = 6

#: Slot 0 means "not bound to any candidate". Query rows always use it.
UNBOUND_SLOT = 0


#: Explicit names preserve the behavior of previously saved checkpoints.
#: the ablation arm and the shape of every checkpoint written before that date.
READOUTS = ("sensor_only", "fused", "dual_attention", "neighbors")


@dataclass(frozen=True)
class ComparatorConfig:
    """Shape and initialisation of the comparator.

    ``n_slots`` bounds how many candidates one episode may carry. The slot embedding is an
    episode-randomised coreference tag, not a label identity: it lets attention notice that a
    support row and a candidate refer to the same thing without ever learning what that thing is.

    ``readout`` selects what the learned part is allowed to see:

    * ``"sensor_only"`` (default) — attention runs over the query's and the support rows' SIGNAL
      vectors only, and emits one zero-initialised scalar per support row that shifts that row's
      logit before the closed-form softmax. Label text and candidate text never enter the learned
      path: sensors are compared with sensors, labels with labels, paired by index in the vote.
      This keeps the learned part from becoming a label-text-to-motion bridge — the pathway every
      earlier learned text component turned out to hurt through — and makes the residual a pure
      "which examples do I trust" decision. In a zero-shot episode it still reweights the
      background rows; what it cannot touch is the label-to-candidate relation, which stays the
      frozen text cosine.
    * ``"fused"`` — each support row's signal, sensor description and label text are fused into
      one token, attention runs over [candidates | query | support], and one zero-initialised
      scalar per CANDIDATE is added to its logit. Strictly more capacity, including a learned
      label-to-candidate relation; kept as the ablation arm.

    ``use_descriptor`` adds the frozen sensor-description embedding to each ``sensor_only`` row.
    It is constant within an Arm A episode (neutral text + exact-key support), so it is OFF there
    and ON for Arm B, where support may mix configurations. ``fused`` always consumes it.
    """

    text_dim: int = 384
    n_layers: int = 2
    n_slots: int = 64
    #: Content leads at initialisation. Role and coreference stay visible without jointly
    #: outweighing the signal they are supposed to annotate.
    identity_gain_init: float = 0.25
    readout: str = "sensor_only"
    use_descriptor: bool = False
    max_instances: int = 512

    def __post_init__(self) -> None:
        if self.readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {self.readout!r}")
        if self.max_instances < 1 or self.n_layers < 0:
            raise ValueError("instance capacity must be positive and depth nonnegative")
        if self.readout == "dual_attention" and self.n_layers == 0:
            raise ValueError("dual attention needs at least one attention block")


def comparator_config_from_checkpoint(saved: dict) -> ComparatorConfig:
    """Rebuild the config a checkpoint was trained with.

    Checkpoints written before 2026-09-07 predate ``readout`` and were all the fused design; they
    must load as such rather than silently as the current default, or ``load_state_dict`` would
    fail on the head names — and, worse, a step-0 control would be paired against the wrong
    function.
    """
    config = dict(saved)
    if "readout" not in config:
        config["readout"] = "fused"
        config.setdefault("use_descriptor", True)
    return ComparatorConfig(**config)


class SupportComparator(nn.Module):
    """The learned part of the comparison. See :class:`ComparatorConfig` for the two readouts."""

    def __init__(self, spec: AttentionSpec, cfg: ComparatorConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or ComparatorConfig()
        d = spec.d_model
        if self.cfg.readout in ("dual_attention", "neighbors"):
            if self.cfg.use_descriptor:
                raise ValueError("staged readouts use encoder vectors and labels, not descriptors")
            if self.cfg.readout == "dual_attention":
                from .prediction import DualRecordingAttention
                self.prediction = DualRecordingAttention(
                    spec, text_dim=self.cfg.text_dim, n_layers=self.cfg.n_layers,
                    max_instances=self.cfg.max_instances,
                    identity_gain=self.cfg.identity_gain_init,
                )
            return
        self.proj_signal = nn.Linear(d, d)
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.slot_emb = nn.Embedding(self.cfg.n_slots, d)
        self.compose = ScaledSum(3, init=[1.0, self.cfg.identity_gain_init,
                                          self.cfg.identity_gain_init])
        self.stack = SetAttentionStack(spec, self.cfg.n_layers)
        # Whichever readout, the head is one shared scalar with NO bias, zero-initialised, so the
        # whole module is exactly the closed-form vote at step 0. A shared constant would cancel in
        # the softmax and be permanently unidentifiable.
        if self.cfg.readout == "fused":
            self.proj_text = nn.Linear(self.cfg.text_dim, d)
            # A support's feature, descriptor and label must stay associated even when it is not
            # bound to a candidate. Separate set tokens lose that association.
            self.support_fusion = nn.Linear(3 * d, d)
            self.residual_head = nn.Linear(d, 1, bias=False)
            nn.init.zeros_(self.residual_head.weight)
        else:
            if self.cfg.use_descriptor:
                self.proj_text = nn.Linear(self.cfg.text_dim, d)
            self.shift_head = nn.Linear(d, 1, bias=False)
            nn.init.zeros_(self.shift_head.weight)

    @property
    def head(self) -> nn.Linear:
        return self.residual_head if self.cfg.readout == "fused" else self.shift_head

    # ------------------------------------------------------------------ tokens
    def _token(self, content: torch.Tensor, role: int, slot: torch.Tensor) -> torch.Tensor:
        role_vec = self.role_emb(torch.full(
            content.shape[:-1], role, dtype=torch.long, device=content.device,
        ))
        return self.compose(content, role_vec, self.slot_emb(slot))

    def forward(
        self,
        *,
        candidate_text: torch.Tensor,        # (B, C, Z)  frozen text of each candidate label
        query_feature: torch.Tensor,         # (B, Q, d)  encoder rows for the query recording
        query_descriptor: torch.Tensor,      # (B, Q, Z)  the query's sensor text
        query_mask: torch.Tensor,            # (B, Q)     True = a real row
        support_feature: torch.Tensor,       # (B, K, d)  one pooled row per support recording
        support_descriptor: torch.Tensor,    # (B, K, Z)  each support recording's sensor text
        support_label_text: torch.Tensor,    # (B, K, Z)  each support recording's verbatim label
        support_mask: torch.Tensor,          # (B, K)     True = a real support row
        candidate_slot: torch.Tensor,        # (B, C)     episode-randomised coreference tags
        support_slot: torch.Tensor,          # (B, K)     tag of the candidate a row is bound to
        candidate_mask: torch.Tensor | None = None,  # (B, C) True = a real candidate
    ) -> torch.Tensor:
        """``fused``: ``(B, C)`` candidate residual logits. ``sensor_only``: ``(B, K)`` support
        logit shifts. Both are exactly zero at initialisation."""

        B, C, _ = candidate_text.shape
        Q = query_feature.shape[1]
        device = query_feature.device
        if candidate_mask is None:
            candidate_mask = torch.ones((B, C), dtype=torch.bool, device=device)
        if candidate_mask.shape != (B, C):
            raise ValueError("candidate_mask must have shape (batch, candidates)")
        valid_slots = candidate_slot[candidate_mask]
        if bool(valid_slots.le(UNBOUND_SLOT).any()):
            raise ValueError("slot 0 is reserved for unbound tokens")
        if bool(valid_slots.ge(self.cfg.n_slots).any()):
            raise ValueError("candidate slot exceeds ComparatorConfig.n_slots")

        query_slot = torch.full((B, Q), UNBOUND_SLOT, dtype=torch.long, device=device)

        if self.cfg.readout == "sensor_only":
            # Sensors against sensors. No candidate token, no label text: the association between a
            # row's signal and its label lives in the vote's index, not in attention.
            query = self._token(self.proj_signal(query_feature), ROLE_QUERY, query_slot)
            content = self.proj_signal(support_feature)
            if self.cfg.use_descriptor:
                content = content + self.proj_text(support_descriptor)
            support = self._token(content, ROLE_SUPPORT, support_slot)
            hidden = self.stack(
                torch.cat([query, support], dim=1),
                key_padding_mask=torch.cat([query_mask, support_mask], dim=1),
            )
            with torch.autocast(device_type=device.type, enabled=False):
                shift = self.shift_head(hidden[:, Q:].float()).squeeze(-1)
            return shift.masked_fill(~support_mask, 0.0)

        candidate = self._token(self.proj_text(candidate_text), ROLE_CANDIDATE, candidate_slot)
        query = self._token(self.proj_signal(query_feature), ROLE_QUERY, query_slot)
        query_desc = self._token(self.proj_text(query_descriptor), ROLE_QUERY_DESC, query_slot)
        support = self._token(self.proj_signal(support_feature), ROLE_SUPPORT, support_slot)
        support_desc = self._token(
            self.proj_text(support_descriptor), ROLE_SUPPORT_DESC, support_slot,
        )
        support_label = self._token(
            self.proj_text(support_label_text), ROLE_SUPPORT_LABEL, support_slot,
        )
        support_row = self.support_fusion(torch.cat(
            [support, support_desc, support_label], dim=-1,
        ))

        tokens = torch.cat(
            [candidate, query, query_desc, support_row], dim=1,
        )
        valid = torch.cat([
            candidate_mask,
            query_mask, query_mask,
            support_mask,
        ], dim=1)

        hidden = self.stack(tokens, key_padding_mask=valid)
        with torch.autocast(device_type=device.type, enabled=False):
            residual = self.residual_head(hidden[:, :C].float()).squeeze(-1)
        return residual.masked_fill(~candidate_mask, 0.0)

    def telemetry(self) -> dict[str, float]:
        if self.cfg.readout == "neighbors":
            return {"prediction/parameter_free": 1.0}
        if self.cfg.readout == "dual_attention":
            return {
                f"prediction/{name}/log_scale": float(head.log_scale.detach())
                for name, head in (("zero_shot", self.prediction.zero_shot),
                                   ("enrollment", self.prediction.enrollment))
            }
        gains = self.compose.log_gain.detach().exp()
        return {
            # Whichever readout: the norm of the one zero-initialised head, so a flat line here
            # means the learned part never woke up.
            "comparator/residual_head_norm": float(self.head.weight.detach().norm()),
            "comparator/readout_is_fused": float(self.cfg.readout == "fused"),
            "comparator/content_gain": float(gains[0]),
            "comparator/identity_gain_mean": float(gains[1:].mean()),
        }


def support_vote(
    *,
    candidate_text: torch.Tensor,        # (B, C, Z)  L2-normalised
    support_label_text: torch.Tensor,    # (B, K, Z)  L2-normalised
    support_bound: torch.Tensor,         # (B, K)     candidate index, -1 = not a candidate's label
    support_mask: torch.Tensor,          # (B, K)
    weights: torch.Tensor,               # (B, K, C)  non-negative
) -> torch.Tensor:
    """The closed-form readout: ``sum_e weight(e,c) * vote(e,c)``.

    An enrolled support row votes 1 for the candidate it is bound to and nothing for the others.
    Any other row votes the rectified cosine between its own label text and each candidate's text.
    Neither branch has a per-candidate parameter, so the rule is identical for a candidate the
    model has never seen.
    """

    B, C, _ = candidate_text.shape
    K = support_label_text.shape[1]
    device = candidate_text.device

    index = torch.arange(C, device=device).view(1, 1, C)
    bound = support_bound.to(device).view(B, K, 1)

    enrolled_vote = (bound.ge(0) & bound.eq(index)).to(weights.dtype)
    semantic = F.relu(torch.bmm(
        F.normalize(support_label_text.float(), dim=-1),
        F.normalize(candidate_text.float(), dim=-1).transpose(1, 2),
    )).to(weights.dtype)
    # A row bound to a candidate speaks only through the identity vote; letting it also vote
    # semantically would count the same evidence twice, with the duplicate landing on whichever
    # other candidates happen to share vocabulary with its label.
    text_vote = semantic * bound.lt(0).to(weights.dtype)

    vote = enrolled_vote + text_vote
    masked = weights * support_mask.unsqueeze(-1).to(weights.dtype)
    return (masked * vote).sum(dim=1)


def center_episode(
    query_feature: torch.Tensor,      # (B, Q, d)
    support_feature: torch.Tensor,    # (B, K, d)
    query_mask: torch.Tensor,         # (B, Q)
    support_mask: torch.Tensor,       # (B, K)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract each episode's own mean feature from its query and support rows.

    WHY THIS MIGHT MATTER (it is an arm to measure, not an assumption). Retrieval in the previous
    design ranked by *acquisition configuration* rather than activity, at a measured 7.0x lift.
    Configuration is very close to a common mode within an episode: the support rows all share the
    query's acquisition key by construction, so whatever they have in common is mostly the thing we
    do NOT want the similarity to key on. Removing the episode mean leaves only how the rows differ
    from each other, which is the quantity a discriminator should be using.

    The same trick is standard elsewhere for the same reason — hubness correction in retrieval,
    all-but-the-top in word embeddings, prototype centering in few-shot learning.

    METHODOLOGY WARNING: centering changes the step-0 function, so a centered run may NOT be
    compared to an uncentered one by paired gain. Compare raw scores at matched seeds instead.
    """
    rows = torch.cat([query_feature, support_feature], dim=1)
    mask = torch.cat([query_mask, support_mask], dim=1).unsqueeze(-1).to(rows.dtype)
    mean = (rows * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return query_feature - mean, support_feature - mean


def comparator_logits(
    comparator: SupportComparator | None,
    *,
    candidate_text: torch.Tensor,
    query_feature: torch.Tensor,
    query_descriptor: torch.Tensor,
    query_mask: torch.Tensor,
    support_feature: torch.Tensor,
    support_descriptor: torch.Tensor,
    support_label_text: torch.Tensor,
    support_bound: torch.Tensor,
    support_mask: torch.Tensor,
    candidate_slot: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
    vote_scale: float = 10.0,
    center: bool = True,
    instance_ids: torch.Tensor | None = None,
    enrollment_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Score every candidate for every episode in the batch.

    ``comparator=None`` is the untrained floor: cosine similarity between the query and each
    support recording supplies the weights, and nothing is learned anywhere. With a comparator
    whose head is still zero the two paths agree exactly, which is what makes the step-0 control a
    control rather than an approximation.

    A ``sensor_only`` comparator shifts each support row's logit BEFORE the softmax (``shift`` is
    already in units of the temperature-scaled logit); a ``fused`` one adds a residual to each
    candidate's logit AFTER the vote. ``base_logits`` is always the unshifted closed-form vote, so
    ``residual = logits - base_logits`` is the learned contribution under either readout, and
    ``support_weight`` is the weight the model actually used.

    ``center`` removes the episode's mean feature first (see :func:`center_episode`) and is ON by
    default. It is applied to BOTH the closed-form similarity and the comparator's signal input, so
    the two stay the same function of the same features, and identity-at-init survives it.
    """

    B, C, _ = candidate_text.shape
    K = support_feature.shape[1]
    if candidate_mask is None:
        candidate_mask = torch.ones((B, C), dtype=torch.bool, device=candidate_text.device)

    if comparator is not None and comparator.cfg.readout in ("dual_attention", "neighbors"):
        from .prediction import neighbor_logits
        if center:
            raise ValueError("staged readouts require center=False; preserve encoder geometry")
        valid_query = query_mask.unsqueeze(-1).to(query_feature.dtype)
        pooled_query = (query_feature * valid_query).sum(1) / valid_query.sum(1).clamp_min(1)
        enrolled_mask = support_mask & support_bound.ge(0)
        base, weights = neighbor_logits(
            pooled_query, support_feature, support_bound.masked_fill(~enrolled_mask, 0),
            enrolled_mask, candidate_mask, temperature,
        )
        if comparator.cfg.readout == "neighbors":
            if bool((support_mask & ~enrolled_mask).any()):
                raise ValueError("neighbors cannot interpret unbound background labels")
            logits = base
        else:
            logits = comparator.prediction(
                query=pooled_query, candidates=candidate_text, candidate_mask=candidate_mask,
                support=support_feature, support_labels=support_label_text,
                support_mask=support_mask, enrolled=(enrolled_mask.any(dim=1)
                    if enrollment_override is None else enrollment_override),
                instance_ids=instance_ids,
            )
        # Difference is diagnostic only. Mask candidate padding before subtraction/reduction.
        residual = (logits - base).masked_fill(~candidate_mask | (base < -1e20), 0.0)
        return {"logits": logits, "base_logits": base, "residual": residual,
                "support_weight": weights}

    if center:
        query_feature, support_feature = center_episode(
            query_feature, support_feature, query_mask, support_mask,
        )

    # Pool the query's rows into one direction, then score it against every support recording.
    valid_query = query_mask.unsqueeze(-1).to(query_feature.dtype)
    pooled = (query_feature * valid_query).sum(dim=1) / valid_query.sum(dim=1).clamp_min(1e-6)
    similarity = torch.bmm(
        F.normalize(pooled.float(), dim=-1).unsqueeze(1),
        F.normalize(support_feature.float(), dim=-1).transpose(1, 2),
    ).squeeze(1)                                                   # (B, K)
    similarity = similarity.masked_fill(~support_mask, float("-inf"))

    # Softmax over support rows, shared across candidates. With every row masked out (K = 0 or an
    # all-empty support set) the vote is zero and the logits fall back to the text path alone.
    empty = ~support_mask.any(dim=1, keepdim=True)
    # Only all-empty episodes need finite dummy logits; real episodes must keep
    # masked rows at -inf so padding never takes probability mass.
    safe = torch.where(empty, torch.zeros_like(similarity), similarity)
    scaled = safe / temperature

    def support_weights(logit: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(logit, dim=1)
        weights = torch.where(support_mask, weights, torch.zeros_like(weights))
        weights = torch.where(empty, torch.zeros_like(weights), weights)
        return weights.unsqueeze(-1).expand(B, K, C)

    def vote(weights: torch.Tensor) -> torch.Tensor:
        return vote_scale * support_vote(
            candidate_text=candidate_text,
            support_label_text=support_label_text,
            support_bound=support_bound,
            support_mask=support_mask,
            weights=weights,
        )

    base_weights = support_weights(scaled)
    base = vote(base_weights)
    used_weights = base_weights
    logits = base
    if comparator is not None:
        if candidate_slot is None:
            raise ValueError("a comparator needs episode-randomised candidate slots")
        support_slot = torch.where(
            support_bound.ge(0),
            torch.gather(candidate_slot, 1, support_bound.clamp_min(0)),
            torch.full_like(support_bound, UNBOUND_SLOT),
        )
        learned = comparator(
            candidate_text=candidate_text,
            query_feature=query_feature,
            query_descriptor=query_descriptor,
            query_mask=query_mask,
            support_feature=support_feature,
            support_descriptor=support_descriptor,
            support_label_text=support_label_text,
            support_mask=support_mask,
            candidate_slot=candidate_slot,
            support_slot=support_slot,
            candidate_mask=candidate_mask,
        )
        if comparator.cfg.readout == "sensor_only":
            used_weights = support_weights(scaled + learned.to(scaled.dtype))
            logits = vote(used_weights)
        else:
            logits = base + learned

    return {"logits": logits, "base_logits": base, "residual": logits - base,
            "support_weight": used_weights[..., 0]}
