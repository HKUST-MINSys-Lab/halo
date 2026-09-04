"""Support-only comparator tests (IMWUT handoff W4).

The properties pinned here are the ones the paper's controls depend on. Identity-at-init is not a
nicety: the untrained floor and the paired step-0 control are both defined as "this module with a
zero residual head", so if that equality is approximate every reported gain is approximate too.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from model.blocks import AttentionSpec
from model.evidence.comparator import (
    ComparatorConfig,
    SupportComparator,
    comparator_logits,
    support_vote,
)

SPEC = AttentionSpec(d_model=32, n_heads=4, ffn_mult=2, dropout=0.0)
TEXT = 16


def _episode(B=2, C=4, Q=3, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    candidate_text = F.normalize(torch.randn(B, C, TEXT, generator=g), dim=-1)
    support_label_text = F.normalize(torch.randn(B, K, TEXT, generator=g), dim=-1)
    bound = torch.full((B, K), -1, dtype=torch.long)
    # Enrol the first two support rows of every episode against candidates 0 and 1, and let their
    # label text BE that candidate's text, which is what a real enrolled row looks like.
    for b in range(B):
        for row in range(min(2, K)):
            bound[b, row] = row
            support_label_text[b, row] = candidate_text[b, row]
    return {
        "candidate_text": candidate_text,
        "query_feature": torch.randn(B, Q, SPEC.d_model, generator=g),
        "query_descriptor": F.normalize(torch.randn(B, Q, TEXT, generator=g), dim=-1),
        "query_mask": torch.ones(B, Q, dtype=torch.bool),
        "support_feature": torch.randn(B, K, SPEC.d_model, generator=g),
        "support_descriptor": F.normalize(torch.randn(B, K, TEXT, generator=g), dim=-1),
        "support_label_text": support_label_text,
        "support_bound": bound,
        "support_mask": torch.ones(B, K, dtype=torch.bool),
        "candidate_slot": (1 + torch.arange(C)).unsqueeze(0).expand(B, C).contiguous(),
        "candidate_mask": torch.ones(B, C, dtype=torch.bool),
    }


def _comparator():
    return SupportComparator(SPEC, ComparatorConfig(text_dim=TEXT, n_layers=2, n_slots=16)).eval()


def test_identity_at_init_matches_the_closed_form_vote():
    """At step 0 the comparator IS the untrained floor, exactly. Every control depends on this."""
    episode = _episode()
    with torch.no_grad():
        trained = comparator_logits(_comparator(), **episode)
        floor = comparator_logits(None, **episode)
    assert torch.allclose(trained["logits"], floor["logits"], atol=1e-6)
    assert torch.allclose(trained["residual"], torch.zeros_like(trained["residual"]), atol=1e-8)


def test_support_permutation_invariance():
    """Support is a set; reordering it must not move a single logit."""
    episode = _episode()
    comparator = _comparator()
    with torch.no_grad():
        comparator.residual_head.weight.normal_(std=0.05)  # wake the head so this is a real test
        before = comparator_logits(comparator, **episode)["logits"]
        order = torch.tensor([3, 0, 5, 1, 4, 2])
        shuffled = dict(episode)
        for key in ("support_feature", "support_descriptor", "support_label_text",
                    "support_bound", "support_mask"):
            shuffled[key] = episode[key][:, order]
        after = comparator_logits(comparator, **shuffled)["logits"]
    assert torch.allclose(before, after, atol=1e-5)


def test_candidate_permutation_equivariance():
    """Permuting candidates permutes the logits identically — no per-candidate parameters."""
    episode = _episode()
    comparator = _comparator()
    with torch.no_grad():
        comparator.residual_head.weight.normal_(std=0.05)
        before = comparator_logits(comparator, **episode)["logits"]
        order = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(order)
        permuted = dict(episode)
        permuted["candidate_text"] = episode["candidate_text"][:, order]
        permuted["candidate_slot"] = episode["candidate_slot"][:, order]
        bound = episode["support_bound"].clone()
        bound[bound.ge(0)] = inverse[bound[bound.ge(0)]]
        permuted["support_bound"] = bound
        after = comparator_logits(comparator, **permuted)["logits"]
    assert torch.allclose(before[:, order], after, atol=1e-5)


def test_empty_support_set_runs_and_scores_zero_vote():
    """k = 0 is a real operating point; it must not divide by zero or NaN."""
    episode = _episode(K=4)
    episode["support_mask"] = torch.zeros_like(episode["support_mask"])
    with torch.no_grad():
        out = comparator_logits(_comparator(), **episode)
    assert torch.isfinite(out["logits"]).all()
    assert torch.allclose(out["base_logits"], torch.zeros_like(out["base_logits"]))


def test_zero_support_rows_are_allowed():
    """K = 0 with no support tensor rows at all."""
    episode = _episode(K=0)
    with torch.no_grad():
        out = comparator_logits(_comparator(), **episode)
    assert torch.isfinite(out["logits"]).all()


def test_enrolled_row_votes_only_for_its_own_candidate():
    """An enrolled row's evidence must not leak to candidates that share vocabulary with it."""
    B, C, K = 1, 3, 1
    candidate_text = F.normalize(torch.randn(B, C, TEXT), dim=-1)
    vote = support_vote(
        candidate_text=candidate_text,
        support_label_text=candidate_text[:, :1],          # its label IS candidate 0
        support_bound=torch.tensor([[0]]),
        support_mask=torch.ones(B, K, dtype=torch.bool),
        weights=torch.ones(B, K, C),
    )
    assert torch.allclose(vote[0, 0], torch.tensor(1.0))
    assert torch.allclose(vote[0, 1:], torch.zeros(2), atol=1e-6)


def test_unbound_row_votes_by_label_text_similarity():
    """This branch is what makes an unseen candidate scorable, so it must actually fire."""
    B, C, K = 1, 2, 1
    candidate_text = F.normalize(torch.randn(B, C, TEXT), dim=-1)
    label = F.normalize(candidate_text[:, :1] + 0.05 * torch.randn(B, 1, TEXT), dim=-1)
    vote = support_vote(
        candidate_text=candidate_text,
        support_label_text=label,
        support_bound=torch.tensor([[-1]]),
        support_mask=torch.ones(B, K, dtype=torch.bool),
        weights=torch.ones(B, K, C),
    )
    assert vote[0, 0] > vote[0, 1]
    assert (vote >= 0).all()          # rectified: a negative cosine never votes against


def test_masked_support_rows_do_not_vote():
    episode = _episode(K=6)
    masked = dict(episode)
    masked["support_mask"] = episode["support_mask"].clone()
    masked["support_mask"][:, 3:] = False
    with torch.no_grad():
        full = comparator_logits(None, **episode)["logits"]
        part = comparator_logits(None, **masked)["logits"]
    assert not torch.allclose(full, part)
    assert torch.isfinite(part).all()


def test_padded_candidates_cannot_change_real_logits():
    """Candidate padding is a batch shape detail, not information available to attention."""
    episode = _episode(B=1, C=4, seed=17)
    comparator = _comparator()
    with torch.no_grad():
        comparator.residual_head.weight.normal_(std=0.05)
        base = comparator_logits(comparator, **episode)["logits"]
        padded = dict(episode)
        padded["candidate_text"] = torch.cat([
            episode["candidate_text"], torch.randn(1, 2, TEXT),
        ], dim=1)
        padded["candidate_slot"] = torch.cat([
            episode["candidate_slot"], torch.zeros(1, 2, dtype=torch.long),
        ], dim=1)
        padded["candidate_mask"] = torch.tensor([[True, True, True, True, False, False]])
        after = comparator_logits(comparator, **padded)["logits"]
    assert torch.allclose(base, after[:, :4], atol=1e-5)


def test_bound_candidates_cannot_use_the_unbound_slot():
    episode = _episode(B=1)
    episode["candidate_slot"][0, 0] = 0
    with torch.no_grad(), pytest.raises(ValueError, match="reserved"):
        comparator_logits(_comparator(), **episode)


def test_verbatim_duplicate_labels_are_not_a_problem():
    """Two candidates with identical text receive identical votes. That is correct, not a bug."""
    B, C, K = 1, 3, 2
    shared = F.normalize(torch.randn(B, 1, TEXT), dim=-1)
    candidate_text = torch.cat([shared, shared, F.normalize(torch.randn(B, 1, TEXT), dim=-1)], 1)
    vote = support_vote(
        candidate_text=candidate_text,
        support_label_text=F.normalize(torch.randn(B, K, TEXT), dim=-1),
        support_bound=torch.full((B, K), -1, dtype=torch.long),
        support_mask=torch.ones(B, K, dtype=torch.bool),
        weights=torch.ones(B, K, C),
    )
    assert torch.allclose(vote[0, 0], vote[0, 1], atol=1e-6)


def test_every_parameter_receives_gradient_after_two_steps():
    """A dead parameter means a component that cannot be credited or blamed."""
    comparator = SupportComparator(SPEC, ComparatorConfig(text_dim=TEXT, n_layers=2, n_slots=16))
    optimizer = torch.optim.SGD(comparator.parameters(), lr=0.1)
    episode = _episode()
    for _ in range(2):
        optimizer.zero_grad()
        out = comparator_logits(comparator, **episode)
        out["logits"].square().mean().backward()
        optimizer.step()
    optimizer.zero_grad()
    out = comparator_logits(comparator, **episode)
    out["logits"].square().mean().backward()
    dead = [name for name, p in comparator.named_parameters()
            if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0]
    assert not dead, f"parameters with no gradient: {dead}"


def test_residual_is_differentiable_through_the_encoder_features():
    """Support rows are encoded per query WITH gradients; that is the point of the design."""
    episode = _episode()
    episode["support_feature"] = episode["support_feature"].clone().requires_grad_(True)
    comparator = _comparator()
    with torch.no_grad():
        comparator.residual_head.weight.normal_(std=0.05)
    comparator_logits(comparator, **episode)["logits"].sum().backward()
    assert episode["support_feature"].grad is not None
    assert episode["support_feature"].grad.abs().sum() > 0


def test_batch_episodes_are_independent():
    """One episode's support must never reach another's logits."""
    episode = _episode(B=2, seed=3)
    comparator = _comparator()
    with torch.no_grad():
        comparator.residual_head.weight.normal_(std=0.05)
        both = comparator_logits(comparator, **episode)["logits"]
        single = {k: v[:1] for k, v in episode.items()}
        first = comparator_logits(comparator, **single)["logits"]
    assert torch.allclose(both[:1], first, atol=1e-5)


# ---------------------------------------------------------------- centering arm
def test_centering_removes_the_episode_mean():
    """The point of the arm: after centering, what rows have in common is gone."""
    from model.evidence.comparator import center_episode

    episode = _episode(K=6)
    q, s = center_episode(
        episode["query_feature"], episode["support_feature"],
        episode["query_mask"], episode["support_mask"],
    )
    rows = torch.cat([q, s], dim=1)
    assert torch.allclose(rows.mean(dim=1), torch.zeros_like(rows.mean(dim=1)), atol=1e-5)


def test_centering_respects_the_masks():
    """Padded rows must not drag the mean; only real rows define the common mode."""
    from model.evidence.comparator import center_episode

    episode = _episode(K=6)
    mask = episode["support_mask"].clone()
    mask[:, 4:] = False
    episode["support_feature"][:, 4:] = 1e3          # padded garbage
    q, s = center_episode(
        episode["query_feature"], episode["support_feature"], episode["query_mask"], mask,
    )
    real = torch.cat([q, s[:, :4]], dim=1)
    assert torch.allclose(real.mean(dim=1), torch.zeros_like(real.mean(dim=1)), atol=1e-3)


def test_centering_changes_the_scores():
    """If it were a no-op there would be nothing to measure."""
    episode = _episode()
    with torch.no_grad():
        plain = comparator_logits(None, center=False, **episode)["logits"]
        centered = comparator_logits(None, center=True, **episode)["logits"]
    assert not torch.allclose(plain, centered, atol=1e-4)


def test_centering_keeps_identity_at_init():
    """The control must stay exact in the centered arm too, or that arm has no floor."""
    episode = _episode()
    with torch.no_grad():
        trained = comparator_logits(_comparator(), center=True, **episode)["logits"]
        floor = comparator_logits(None, center=True, **episode)["logits"]
    assert torch.allclose(trained, floor, atol=1e-6)


def test_centering_keeps_support_permutation_invariance():
    episode = _episode()
    comparator = _comparator()
    with torch.no_grad():
        comparator.residual_head.weight.normal_(std=0.05)
        before = comparator_logits(comparator, center=True, **episode)["logits"]
        order = torch.tensor([3, 0, 5, 1, 4, 2])
        shuffled = dict(episode)
        for key in ("support_feature", "support_descriptor", "support_label_text",
                    "support_bound", "support_mask"):
            shuffled[key] = episode[key][:, order]
        after = comparator_logits(comparator, center=True, **shuffled)["logits"]
    assert torch.allclose(before, after, atol=1e-5)
