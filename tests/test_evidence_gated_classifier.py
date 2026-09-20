"""Contract tests for ``support_classifier_v4`` (the evidence-gated head).

These encode the acceptance criteria of
``docs/journal/2026-09-20-classifier-v4-evidence-gated-design.md``. The three that exist because a
previous generation failed them are marked: label blindness, no-deletion, and nonzero (not merely
finite) gradient reach.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from model.blocks import AttentionSpec
from model.support.evidence_gated_classifier import (
    ARCHITECTURE_VERSION, N_GATE_FEATURES, N_TRUST_FEATURES,
    EvidenceGatedClassifierConfig, EvidenceGatedSupportClassifier,
)
from model.support.factory import (
    CLASSIFIER_ARCHITECTURE_STATUS, CLASSIFIER_TRY_NAME, build_classifier_from_blob,
    classifier_architecture_base_try_name,
    classifier_try_name,
)
from training.support_classifier.neighbors import differentiable_neighbor_logits

D_MODEL, TEXT_DIM = 32, 384


def make_head(**overrides) -> EvidenceGatedSupportClassifier:
    spec = AttentionSpec(d_model=D_MODEL, n_heads=4, ffn_mult=2, dropout=0.0)
    cfg = EvidenceGatedClassifierConfig(text_dim=TEXT_DIM, **overrides)
    return EvidenceGatedSupportClassifier(spec, cfg).eval()


def make_episode(*, b=3, c=4, k=6, seed=0, zero_support_rows=(), device="cpu"):
    """A batch with mixed enrollment: some candidates unenrolled, optionally whole rows zero."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn(b, D_MODEL, generator=generator, device="cpu").to(device)
    support = torch.randn(b, k, D_MODEL, generator=generator, device="cpu").to(device)
    bound = torch.randint(0, c - 1, (b, k), generator=generator, device="cpu").to(device)
    support_mask = torch.ones(b, k, dtype=torch.bool, device=device)
    support_mask[:, -1] = False  # one padded slot everywhere
    for row in zero_support_rows:
        support_mask[row] = False
    bound = torch.where(support_mask, bound, torch.full_like(bound, -1))
    candidate_mask = torch.ones(b, c, dtype=torch.bool, device=device)
    candidate_mask[:, -1] = False  # one padded candidate everywhere
    text = F.normalize(torch.randn(b, c, TEXT_DIM, generator=generator, device="cpu"), dim=-1).to(device)
    text = text * candidate_mask.unsqueeze(-1)
    support_text = text.gather(
        1, bound.clamp_min(0).unsqueeze(-1).expand(-1, -1, TEXT_DIM),
    ) * support_mask.unsqueeze(-1)
    return {
        "query_feature": query, "support_feature": support * support_mask.unsqueeze(-1),
        "support_label_text": support_text, "support_bound": bound,
        "support_mask": support_mask,
        "support_pair_slot": torch.arange(1, k + 1, device=device).unsqueeze(0).expand(b, -1),
        "candidate_text": text, "candidate_mask": candidate_mask,
        "candidate_slot": torch.arange(1, c + 1, device=device).unsqueeze(0).expand(b, -1),
    }


# --------------------------------------------------------------------- structure
def test_architecture_version_and_feature_widths_are_pinned():
    assert ARCHITECTURE_VERSION == "support_classifier_v4"
    head = make_head()
    batch = make_episode()
    out = head(**batch)
    assert out["trust_features"].shape[-1] == N_TRUST_FEATURES
    assert out["gate_features"].shape[-1] == N_GATE_FEATURES


def test_config_rejects_an_unbounded_blend():
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            EvidenceGatedClassifierConfig(lambda_max=bad)
    with pytest.raises(ValueError):
        EvidenceGatedClassifierConfig(trust_scale=0.0)


def test_support_vote_starts_at_the_shared_helper():
    """The metric path starts at the closed-form differentiable-neighbour vote.

    Not bit-for-bit: the gate output layers are initialised at std 1e-3 rather than exactly zero,
    because a zero output weight leaves the trust MLP permanently dead (its output bias cancels in
    the softmax over supports). Exact equality is asserted separately under ``trust_override=0``.
    """
    head = make_head()
    batch = make_episode()
    out = head(**batch)
    q = batch["query_feature"].float()
    mean = ((batch["support_feature"] * batch["support_mask"].unsqueeze(-1)).sum(1)
            / batch["support_mask"].sum(1, keepdim=True).clamp_min(1))
    reference, _ = differentiable_neighbor_logits(
        q - mean, batch["support_feature"].float() - mean[:, None], batch["support_bound"],
        batch["support_mask"], batch["candidate_mask"], temperature=head.cfg.temperature,
    )
    assert torch.allclose(out["metric_logits"], reference, atol=2e-2)
    assert torch.allclose(out["neighbor_logits"], reference, atol=1e-6)
    exact = head(**batch, trust_override=0.0)
    assert torch.allclose(exact["metric_logits"], reference, atol=1e-6)


def test_lambda_respects_its_bound_and_is_forced_to_one_without_support():
    head = make_head()
    batch = make_episode()
    out = head(**batch)
    k_c, valid = out["k_c"], batch["candidate_mask"]
    enrolled = k_c.gt(0) & valid
    assert bool(enrolled.any()) and bool((k_c.eq(0) & valid).any())
    assert torch.all(out["lambda"].masked_select(enrolled) <= head.cfg.lambda_max + 1e-6)
    assert torch.allclose(
        out["lambda"].masked_select(k_c.eq(0) & valid),
        torch.ones(int((k_c.eq(0) & valid).sum())),
    )


def test_trust_is_bounded_and_zero_on_padded_slots():
    head = make_head()
    batch = make_episode()
    with torch.no_grad():  # a non-degenerate gate, not the zero-init one
        head.trust_mlp[-1].weight.normal_(std=3.0)
        head.trust_mlp[-1].bias.fill_(5.0)
    out = head(**batch)
    assert torch.all(out["trust"].abs() <= head.cfg.trust_scale + 1e-6)
    assert torch.all(out["trust"].masked_select(~batch["support_mask"]) == 0.0)


# --------------------------------------------------------------------- the three that matter
def test_gates_are_blind_to_label_identity():
    """Regression for the v1/v2/v3-contextual collapse: no gate input may carry label identity.

    Replacing every candidate's text (and therefore the semantic path) must leave the support
    vote, the trust weights and the blend weight bit-identical. Only the text branch may move.
    """
    head = make_head()
    batch = make_episode(seed=1)
    with torch.no_grad():  # non-degenerate gates, so blindness is not trivially satisfied
        head.trust_mlp[-1].weight.normal_(std=1.0)
        head.gate_mlp[-1].weight.normal_(std=1.0)
    first = head(**batch)
    scrambled = dict(batch)
    roll = torch.roll(batch["candidate_text"], shifts=1, dims=1) * batch["candidate_mask"].unsqueeze(-1)
    scrambled["candidate_text"] = roll
    scrambled["support_label_text"] = torch.zeros_like(batch["support_label_text"])
    scrambled["candidate_slot"] = torch.flip(batch["candidate_slot"], dims=(1,))
    scrambled["support_pair_slot"] = torch.flip(batch["support_pair_slot"], dims=(1,))
    second = head(**scrambled)
    for key in ("trust", "lambda", "metric_logits", "neighbor_logits", "gate_features",
                "trust_features"):
        assert torch.equal(first[key], second[key]), f"{key} depends on label identity"
    assert not torch.allclose(first["text_logits"], second["text_logits"])


def test_support_path_can_never_be_deleted():
    """With lambda pinned at its bound and every support maximally distrusted, support still moves
    the argmax. A blend that can reach 1.0 would make this impossible; that is how v2 collapsed."""
    head = make_head(text_temperature=1.0)
    b, c, k = 1, 3, 4
    device = "cpu"
    query = torch.zeros(b, D_MODEL)
    query[0, 0] = 1.0
    support = torch.zeros(b, k, D_MODEL)
    support[0, 0, 0] = 1.0   # candidate 0: identical to the query
    support[0, 1, 1] = 1.0   # candidate 1: orthogonal
    support[0, 2, 1] = 1.0
    support[0, 3, 1] = 1.0
    bound = torch.tensor([[0, 1, 1, 1]])
    support_mask = torch.ones(b, k, dtype=torch.bool)
    candidate_mask = torch.ones(b, c, dtype=torch.bool)
    text = torch.zeros(b, c, TEXT_DIM)
    text[0, 1, 0] = 1.0      # text points hard at candidate 1
    text[0, 0, 1] = 1.0
    text[0, 2, 2] = 1.0
    batch = {
        "query_feature": query, "support_feature": support,
        "support_label_text": text.gather(1, bound.unsqueeze(-1).expand(-1, -1, TEXT_DIM)),
        "support_bound": bound, "support_mask": support_mask,
        "support_pair_slot": torch.arange(1, k + 1).unsqueeze(0),
        "candidate_text": text, "candidate_mask": candidate_mask,
        "candidate_slot": torch.arange(1, c + 1).unsqueeze(0),
    }
    # Make the constructed disagreement deterministic: the text path votes for candidate 1,
    # while the support path strongly votes for candidate 0.
    with torch.no_grad():
        head.p_text.weight.zero_()
        head.p_text.bias.zero_()
        head.p_text.weight[0, 0] = 1.0
    pinned = head(**batch, lambda_override=head.cfg.lambda_max,
                  trust_override=-head.cfg.trust_scale)
    without_support = head(**batch, lambda_override=1.0)
    assert not torch.allclose(pinned["logits"], without_support["logits"]), \
        "support evidence vanished from the output at the lambda bound"
    gap = (pinned["logits"] - without_support["logits"]).abs().max()
    assert float(gap.detach()) > 1e-3
    assert pinned["logits"].argmax(dim=1).item() != without_support["logits"].argmax(dim=1).item()


def test_every_gate_parameter_receives_a_nonzero_gradient():
    """v2's equivalent test asserted *finite* gradients and passed with 34 dead tensors."""
    head = make_head()
    batch = make_episode(seed=2, zero_support_rows=(2,))
    out = head(**batch)
    target = torch.tensor([0, 1, 0])
    loss = F.cross_entropy(out["logits"].masked_fill(~batch["candidate_mask"], -1e30), target)
    loss.backward()
    gated = {name: parameter for name, parameter in head.named_parameters()
             if name.startswith(("trust_mlp", "gate_mlp", "trust_norm", "gate_norm",
                                 "lambda_prior"))}
    assert gated, "no gate parameters found"
    for name, parameter in gated.items():
        assert parameter.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient"
        assert float(parameter.grad.abs().sum()) > 0.0, f"{name} received exactly zero gradient"


# --------------------------------------------------------------------- invariances and safety
def test_support_permutation_invariance():
    head = make_head()
    batch = make_episode(seed=3)
    with torch.no_grad():
        head.trust_mlp[-1].weight.normal_(std=1.0)
        head.gate_mlp[-1].weight.normal_(std=1.0)
    first = head(**batch)
    order = torch.randperm(batch["support_feature"].shape[1])
    permuted = dict(batch)
    for key in ("support_feature", "support_label_text", "support_bound", "support_mask",
                "support_pair_slot"):
        permuted[key] = batch[key].index_select(1, order)
    second = head(**permuted)
    assert torch.allclose(first["logits"], second["logits"], atol=1e-5)
    assert torch.allclose(first["lambda"], second["lambda"], atol=1e-5)
    assert torch.allclose(
        first["trust"].index_select(1, order), second["trust"], atol=1e-5,
    )


def test_support_permutation_invariance_with_tied_similarities():
    """Ties must not turn a support-set statistic into an accidental row-position feature."""
    head = make_head()
    with torch.no_grad():
        head.trust_mlp[-1].weight.normal_(std=1.0)
        head.gate_mlp[-1].weight.normal_(std=1.0)
    batch = make_episode(b=1, c=3, k=4, seed=31)
    query = torch.zeros_like(batch["query_feature"])
    query[:, 0] = 1.0
    support = torch.zeros_like(batch["support_feature"])
    support[:, :, 0] = 1.0  # every valid support has exactly the same cosine to the query
    batch["query_feature"] = query
    batch["support_feature"] = support
    batch["support_mask"][:] = True
    batch["support_bound"][:] = torch.tensor([[0, 1, 0, 1]])
    first = head(**batch)
    order = torch.tensor([1, 0, 3, 2])
    permuted = dict(batch)
    for key in ("support_feature", "support_label_text", "support_bound", "support_mask",
                "support_pair_slot"):
        permuted[key] = batch[key].index_select(1, order)
    second = head(**permuted)
    assert torch.allclose(first["logits"], second["logits"], atol=1e-6)
    assert torch.allclose(first["lambda"], second["lambda"], atol=1e-6)
    assert torch.allclose(first["trust"].index_select(1, order), second["trust"], atol=1e-6)


def test_candidate_permutation_equivariance():
    head = make_head()
    batch = make_episode(seed=4, c=4)
    with torch.no_grad():
        head.gate_mlp[-1].weight.normal_(std=1.0)
    first = head(**batch)
    order = torch.tensor([1, 0, 2, 3])  # keep the padded candidate last
    permuted = dict(batch)
    permuted["candidate_text"] = batch["candidate_text"].index_select(1, order)
    permuted["candidate_mask"] = batch["candidate_mask"].index_select(1, order)
    inverse = torch.argsort(order)
    permuted["support_bound"] = torch.where(
        batch["support_mask"], inverse[batch["support_bound"].clamp_min(0)],
        torch.full_like(batch["support_bound"], -1),
    )
    second = head(**permuted)
    assert torch.allclose(
        first["logits"].index_select(1, order), second["logits"], atol=1e-5,
    )


def test_padding_may_hold_non_finite_values():
    """Masked slots are replaced, not multiplied by zero, so NaN padding cannot poison a gate."""
    head = make_head()
    batch = make_episode(seed=5)
    poisoned = dict(batch)
    support = batch["support_feature"].clone()
    support[:, -1] = float("nan")          # the padded support slot
    poisoned["support_feature"] = support
    text = batch["candidate_text"].clone()
    text[:, -1] = float("inf")             # the padded candidate slot
    poisoned["candidate_text"] = text
    out = head(**poisoned)
    valid = batch["candidate_mask"]
    assert torch.isfinite(out["logits"].masked_select(valid)).all()
    assert torch.isfinite(out["lambda"]).all() and torch.isfinite(out["trust"]).all()
    assert torch.isfinite(out["gate_features"]).all()
    assert torch.isfinite(out["trust_features"]).all()


def test_padding_may_hold_non_finite_values_in_backward():
    """Padding safety is a differentiable contract, not merely a finite forward pass."""
    head = make_head()
    batch = make_episode(seed=51)
    batch["query_feature"] = batch["query_feature"].requires_grad_(True)
    batch["support_feature"] = batch["support_feature"].masked_fill(
        ~batch["support_mask"].unsqueeze(-1), float("nan"),
    ).requires_grad_(True)
    batch["candidate_text"] = batch["candidate_text"].masked_fill(
        ~batch["candidate_mask"].unsqueeze(-1), float("inf"),
    )
    out = head(**batch)
    torch.nn.functional.cross_entropy(out["logits"], torch.zeros(3, dtype=torch.long)).backward()
    assert torch.isfinite(batch["query_feature"].grad).all()
    assert torch.isfinite(batch["support_feature"].grad.masked_select(
        batch["support_mask"].unsqueeze(-1)
    )).all()
    for _, parameter in head.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_rank_is_padding_invariant_for_negative_cosines():
    similarity = torch.tensor([[-0.8, -0.2]])
    mask = torch.ones_like(similarity, dtype=torch.bool)
    short = EvidenceGatedSupportClassifier._normalised_rank(
        similarity, mask, mask.sum(dim=1, keepdim=True),
    )
    padded = torch.cat((similarity, torch.zeros((1, 3))), dim=1)
    padded_mask = torch.cat((mask, torch.zeros((1, 3), dtype=torch.bool)), dim=1)
    long = EvidenceGatedSupportClassifier._normalised_rank(
        padded, padded_mask, padded_mask.sum(dim=1, keepdim=True),
    )
    assert torch.equal(short, long[:, :2])
    assert torch.all(long[:, 2:] == 0)


def test_whole_batch_without_support_falls_back_to_meaning():
    head = make_head()
    batch = make_episode(seed=6, zero_support_rows=(0, 1, 2))
    out = head(**batch)
    valid = batch["candidate_mask"]
    assert torch.allclose(out["lambda"].masked_select(valid), torch.ones(int(valid.sum())))
    assert torch.allclose(
        out["logits"].masked_select(valid), out["text_logits"].masked_select(valid), atol=1e-6,
    )


def test_text_stop_gradient_detaches_only_the_selected_rows():
    head = make_head()
    batch = make_episode(seed=7)
    batch["query_feature"] = batch["query_feature"].requires_grad_(True)
    stop = torch.tensor([True, False, False])
    out = head(**batch, text_stop_gradient=stop)
    out["semantic_logits"].sum().backward()
    grad = batch["query_feature"].grad
    assert float(grad[0].abs().sum()) == 0.0
    assert float(grad[1].abs().sum()) > 0.0


def test_ablation_overrides_reproduce_the_named_branches():
    head = make_head()
    batch = make_episode(seed=8)
    # Text-off is defined only for complete enrollment.
    batch["support_bound"][:] = torch.tensor([[0, 1, 2, 0, 1, -1]]).expand(3, -1)
    text_off = head(**batch, lambda_override=0.0)
    enrolled = text_off["k_c"].gt(0) & batch["candidate_mask"]
    assert torch.allclose(
        text_off["logits"].masked_select(enrolled),
        text_off["metric_part"].masked_select(enrolled), atol=1e-6,
    )
    trust_off = head(**batch, trust_override=0.0)
    assert torch.allclose(trust_off["metric_logits"], trust_off["neighbor_logits"], atol=1e-6)


def test_text_off_rejects_any_unenrolled_candidate():
    head = make_head()
    with pytest.raises(ValueError, match="unenrolled"):
        head(**make_episode(seed=52), lambda_override=0.0)


def test_branch_logits_names_are_stable():
    head = make_head()
    out = head(**make_episode(seed=9))
    for branch, key in (("final", "logits"), ("support", "metric_part"),
                        ("support_floor", "neighbor_logits"), ("semantic", "text_logits")):
        assert torch.equal(head.branch_logits(out, branch), out[key])
    with pytest.raises(ValueError):
        head.branch_logits(out, "router")


def test_checkpoint_round_trip_is_strict_and_exact():
    head = make_head()
    with torch.no_grad():
        head.gate_mlp[-1].weight.normal_(std=0.5)
        head.trust_mlp[-1].bias.fill_(0.25)
    blob = {
        "architecture_version": ARCHITECTURE_VERSION,
        "classifier": head.state_dict(),
        "classifier_config": dataclasses.asdict(head.cfg),
        "attention_spec": dataclasses.asdict(head.spec),
    }
    restored, version = build_classifier_from_blob(blob)
    assert version == ARCHITECTURE_VERSION
    batch = make_episode(seed=10)
    with torch.no_grad():
        assert torch.equal(head(**batch)["logits"], restored(**batch)["logits"])
    ablated, _ = build_classifier_from_blob(blob, overrides={"trust_enabled": False})
    assert ablated.cfg.trust_enabled is False
    with pytest.raises(ValueError):
        build_classifier_from_blob(blob, overrides={"lambda_max": 0.99})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gates_run_in_fp32_under_autocast(dtype):
    head = make_head()
    batch = make_episode(seed=11)
    with torch.autocast(device_type="cpu", dtype=dtype, enabled=dtype is torch.bfloat16):
        out = head(**batch)
    assert out["lambda"].dtype == torch.float32
    assert out["trust"].dtype == torch.float32
    assert torch.isfinite(out["logits"].masked_select(batch["candidate_mask"])).all()


# --------------------------------------------------------------------- the curriculum
class _StubEpisode:
    """Only the fields ``corrupt_candidate_text`` reads."""

    def __init__(self, *, zero_shot: bool, gt_slot: int, support_counts: tuple[int, ...],
                 support_set_id: int = 0):
        self.is_zero_shot = zero_shot
        self.gt_slot = gt_slot
        self.support_counts = support_counts
        self.support_set_id = support_set_id


def _corruption_batch():
    episodes = [
        _StubEpisode(zero_shot=False, gt_slot=0, support_counts=(2, 1, 0)),   # eligible
        _StubEpisode(zero_shot=True, gt_slot=0, support_counts=()),           # zero support
        _StubEpisode(zero_shot=False, gt_slot=2, support_counts=(1, 1, 0)),   # truth unenrolled
        _StubEpisode(zero_shot=False, gt_slot=1, support_counts=(1, 3, 1)),   # eligible
    ]
    text = {
        "candidate_text": torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3),
        "candidate_mask": torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool,
        ),
    }
    return episodes, text


def test_corruption_selects_only_answerable_episodes():
    """A corrupted episode must remain answerable from support, or it is pure loss noise."""
    import numpy as np

    from training.support_classifier.train import corrupt_candidate_text

    episodes, text = _corruption_batch()
    mask = corrupt_candidate_text(
        text, episodes, probability=1.0, rng=np.random.default_rng(0), device="cpu",
    )
    assert mask.tolist() == [True, False, False, True]


def test_corruption_deranges_only_enrolled_candidate_text():
    import numpy as np

    from training.support_classifier.train import corrupt_candidate_text

    episodes, text = _corruption_batch()
    original = text["candidate_text"].clone()
    mask = corrupt_candidate_text(
        text, episodes, probability=1.0, rng=np.random.default_rng(1), device="cpu",
    )
    for row in range(len(episodes)):
        valid = text["candidate_mask"][row]
        moved = ~torch.isclose(text["candidate_text"][row], original[row]).all(dim=-1)
        if mask[row]:
            enrolled = torch.zeros_like(valid)
            enrolled[:len(episodes[row].support_counts)] = torch.tensor(
                [count > 0 for count in episodes[row].support_counts], dtype=torch.bool,
            )
            enrolled &= valid
            assert bool(moved[enrolled].all()), "an enrolled candidate kept its own label text"
            assert not bool(moved[valid & ~enrolled].any()), "unenrolled text was corrupted"
            assert not bool(moved[~valid].any()), "a padded slot was rewritten"
            before = {tuple(v.tolist()) for v in original[row][enrolled]}
            after = {tuple(v.tolist()) for v in text["candidate_text"][row][enrolled]}
            assert before == after, "corruption invented text instead of permuting the roster"
        else:
            assert not bool(moved.any())


def test_corruption_is_deterministic_for_a_seed_and_off_by_default():
    import numpy as np

    from training.support_classifier.train import corrupt_candidate_text

    first_episodes, first = _corruption_batch()
    second_episodes, second = _corruption_batch()
    a = corrupt_candidate_text(
        first, first_episodes, probability=0.5, rng=np.random.default_rng(7), device="cpu",
    )
    b = corrupt_candidate_text(
        second, second_episodes, probability=0.5, rng=np.random.default_rng(7), device="cpu",
    )
    assert torch.equal(a, b)
    assert torch.equal(first["candidate_text"], second["candidate_text"])
    episodes, untouched = _corruption_batch()
    original = untouched["candidate_text"].clone()
    assert corrupt_candidate_text(
        untouched, episodes, probability=0.0, rng=np.random.default_rng(0), device="cpu",
    ) is None
    assert torch.equal(untouched["candidate_text"], original)


def test_every_lifecycle_architecture_has_a_try_name():
    """The architecture strings are not numbered consistently with the tries, so the mapping is
    explicit and must stay complete. See docs/results/RESULTS.md, "Classifier naming"."""
    assert classifier_architecture_base_try_name(ARCHITECTURE_VERSION) == "T4"
    with pytest.raises(ValueError, match="requires its saved trajectory"):
        classifier_try_name(ARCHITECTURE_VERSION)
    assert classifier_try_name(ARCHITECTURE_VERSION, trajectory={
        "text_corruption_mode": "replace", "text_corruption_probability": 0.25,
        "unenrolled_calibration": False,
    }) == "T4"
    assert classifier_try_name(ARCHITECTURE_VERSION, trajectory={
        "text_corruption_probability": 0.0,
    }) == "T5"
    assert classifier_try_name(ARCHITECTURE_VERSION, trajectory={
        "text_corruption_mode": "auxiliary", "text_corruption_probability": 1.0,
        "unenrolled_calibration": True,
    }) == "T6"
    assert classifier_try_name("support_classifier_v3") == "v3"
    missing = set(CLASSIFIER_ARCHITECTURE_STATUS) - set(CLASSIFIER_TRY_NAME)
    # The retired token mixer predates the scheme and is deliberately unnamed.
    assert missing == {"support_token_mixer_v1", "support_classifier_v2"}, missing
    assert len(set(CLASSIFIER_TRY_NAME.values())) == len(CLASSIFIER_TRY_NAME), "duplicate try name"


# --------------------------------------------------------------------- T6: calibration and gate-only view
def test_unenrolled_calibration_is_off_by_default_and_label_blind():
    """Off by default (T4 checkpoints unchanged); on, it shifts only unenrolled candidates, by an
    amount that depends on roster coverage and size alone -- never on which label."""
    spec = AttentionSpec(d_model=D_MODEL, n_heads=4, ffn_mult=2, dropout=0.0)
    off = EvidenceGatedSupportClassifier(spec, EvidenceGatedClassifierConfig(text_dim=TEXT_DIM)).eval()
    assert off.unenrolled_mlp is None
    on = EvidenceGatedSupportClassifier(spec, EvidenceGatedClassifierConfig(
        text_dim=TEXT_DIM, unenrolled_calibration=True)).eval()
    on.load_state_dict({**on.state_dict(), **off.state_dict()}, strict=True)
    with torch.no_grad():
        on.unenrolled_mlp[-1].weight.normal_(std=1.0); on.unenrolled_mlp[-1].bias.fill_(0.7)
    batch = make_episode(seed=21)
    a, b = off(**batch), on(**batch)
    valid = batch["candidate_mask"]; enrolled = a["k_c"].gt(0) & valid; unenrolled = a["k_c"].eq(0) & valid
    assert bool(unenrolled.any()) and bool(enrolled.any())
    torch.testing.assert_close(a["logits"][enrolled], b["logits"][enrolled])
    shift = (b["logits"] - a["logits"])
    assert float(shift[unenrolled].abs().min().detach()) > 1e-4
    # The same shift for every unenrolled candidate of an episode: it cannot prefer one label.
    for row in range(valid.shape[0]):
        row_shift = shift[row][unenrolled[row]]
        if row_shift.numel() > 1:
            assert float((row_shift - row_shift[0]).abs().max()) < 1e-6
    # And identical under a relabelled roster with the same coverage.
    scrambled = dict(batch)
    scrambled["candidate_text"] = torch.roll(batch["candidate_text"], 1, dims=1) * valid.unsqueeze(-1)
    torch.testing.assert_close(on(**scrambled)["unenrolled_bias"], b["unenrolled_bias"])


def test_unenrolled_calibration_features_preserve_coverage_and_roster_size():
    """Regression for two-feature LayerNorm, which erased both calibration inputs."""
    k_c = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])
    masks = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    features = EvidenceGatedSupportClassifier._unenrolled_calibration_features(k_c, masks)
    # Same roster, different coverage; same coverage, different roster. Both must remain visible.
    assert features[0, 0] != features[1, 0]
    assert features[1, 1] != features[2, 1]
    assert not torch.equal(features[0], features[1])
    assert not torch.equal(features[1], features[2])


def test_gate_only_view_reaches_exactly_the_blend_gate():
    """T6's auxiliary contract: the corrupted view's loss may train lambda and nothing else."""
    spec = AttentionSpec(d_model=D_MODEL, n_heads=4, ffn_mult=2, dropout=0.0)
    head = EvidenceGatedSupportClassifier(spec, EvidenceGatedClassifierConfig(
        text_dim=TEXT_DIM, unenrolled_calibration=True, semantic_mode="text+primitives"))
    with torch.no_grad():
        head.primitive_head.values.copy_(F.normalize(torch.randn_like(head.primitive_head.values), dim=-1))
    batch = make_episode(seed=22)
    batch["query_feature"] = batch["query_feature"].requires_grad_(True)
    batch["support_feature"] = batch["support_feature"].requires_grad_(True)
    out = head(**batch, gate_only=True)
    valid = batch["candidate_mask"]
    loss = F.cross_entropy(out["logits"].masked_fill(~valid, -1e30), torch.tensor([0, 1, 0]))
    loss.backward()
    live = {name for name, p in head.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert live, "the gate received no gradient at all"
    assert all(name.startswith(("gate_mlp", "gate_norm", "lambda_prior")) for name in live), sorted(live)
    assert any(name.startswith("gate_mlp") for name in live)
    for tensor in (batch["query_feature"], batch["support_feature"]):
        assert tensor.grad is None or float(tensor.grad.abs().sum()) == 0.0, "encoder features were trained"
    # Every branch diagnostic is detached as well, so an auxiliary loss added later cannot train
    # the semantic path from a corrupted view by reaching through one of them.
    for key in ("semantic_logits", "text_logits", "primitive_logits"):
        assert not out[key].requires_grad, f"{key} kept a live graph under gate_only"


def test_corrupted_view_helper_leaves_clean_text_untouched():
    import numpy as np

    from training.support_classifier.train import corrupted_view

    spec = AttentionSpec(d_model=D_MODEL, n_heads=4, ffn_mult=2, dropout=0.0)
    head = EvidenceGatedSupportClassifier(spec, EvidenceGatedClassifierConfig(text_dim=TEXT_DIM))
    batch = make_episode(seed=23, b=4, c=4)
    original = batch["candidate_text"].clone()
    k_c = head(**batch)["k_c"]
    episodes = [_StubEpisode(zero_shot=False, gt_slot=int(k_c[i].argmax()),
                             support_counts=tuple(int(x) for x in k_c[i][batch["candidate_mask"][i]]))
                for i in range(4)]
    text = {key: batch[key] for key in ("candidate_text", "candidate_mask", "support_label_text",
                                        "support_bound", "support_pair_slot", "candidate_slot")}
    rows = {"support_feature": batch["support_feature"], "support_mask": batch["support_mask"]}
    probe = corrupted_view(head, query=batch["query_feature"], rows=rows, text=text,
                           episodes=episodes, device="cpu", probability=1.0,
                           rng=np.random.default_rng(0), with_gradient=False)
    assert torch.equal(text["candidate_text"], original), "the clean roster was modified"
    assert probe["corrupted_view_accuracy"] is not None
    assert not probe["aux/corrupted_view_loss"].requires_grad
    assert 0.0 <= float(probe["corrupted_view_accuracy"]) <= 1.0
    trained = corrupted_view(head, query=batch["query_feature"], rows=rows, text=text,
                             episodes=episodes, device="cpu", probability=1.0,
                             rng=np.random.default_rng(0), with_gradient=True)
    assert trained["aux/corrupted_view_loss"].requires_grad
    assert corrupted_view(head, query=batch["query_feature"], rows=rows, text=text,
                          episodes=episodes, device="cpu", probability=0.0,
                          rng=np.random.default_rng(0), with_gradient=True)["corrupted_view_accuracy"] is None


def test_corrupted_auxiliary_loss_is_balanced_by_support_set():
    from training.support_classifier.train import _balanced_corrupted_view_loss

    per_row = torch.tensor([1.0, 3.0, 10.0])
    corrupted = torch.tensor([True, True, True])
    episodes = [
        _StubEpisode(zero_shot=False, gt_slot=0, support_counts=(1, 1), support_set_id=4),
        _StubEpisode(zero_shot=False, gt_slot=0, support_counts=(1, 1), support_set_id=4),
        _StubEpisode(zero_shot=False, gt_slot=0, support_counts=(1, 1), support_set_id=9),
    ]
    # Equal support-set weighting: mean([mean(1, 3), mean(10)]) = 6, not row mean 14/3.
    assert _balanced_corrupted_view_loss(per_row, corrupted, episodes) == pytest.approx(6.0)
