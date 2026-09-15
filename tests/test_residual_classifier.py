import pytest
import torch

from model.blocks import AttentionSpec
from model.support.residual_classifier import ResidualClassifierConfig, ResidualSupportClassifier
from training.support_classifier.neighbors import differentiable_neighbor_logits
from training.support_classifier.sampling import Episode
from training.support_classifier.train import episode_loss, fit_text_projection
from training.support_classifier.encoding import build_random_encoder
from training.tokenizer.eval_transfer import build_encoder


def _episode(*, k=6, c=3, d=12):
    torch.manual_seed(3)
    query = torch.randn(2, d)
    support = torch.randn(2, k, d)
    bound = torch.arange(k).remainder(c).unsqueeze(0).expand(2, -1).clone()
    mask = torch.ones(2, k, dtype=torch.bool)
    candidate = torch.randn(2, c, 384)
    return dict(query_feature=query, support_feature=support,
                support_label_text=candidate.gather(1, bound.unsqueeze(-1).expand(-1, -1, 384)),
                support_bound=bound, support_mask=mask,
                support_pair_slot=torch.arange(1, k + 1).unsqueeze(0).expand(2, -1),
                candidate_text=candidate, candidate_mask=torch.ones(2, c, dtype=torch.bool),
                candidate_slot=torch.arange(1, c + 1).unsqueeze(0).expand(2, -1))


def test_initial_residual_head_equals_neighbor_floor_without_centring():
    values = _episode()
    head = ResidualSupportClassifier(AttentionSpec(d_model=12, n_heads=3, dropout=0),
                                     ResidualClassifierConfig(centring="none", n_layers=1))
    got = head(**values)["logits"]
    expected, _ = differentiable_neighbor_logits(
        values["query_feature"], values["support_feature"], values["support_bound"],
        values["support_mask"], values["candidate_mask"], temperature=.07,
    )
    assert torch.equal(got, expected)


def test_initial_residual_head_equals_support_centred_neighbor_floor():
    values = _episode()
    head = ResidualSupportClassifier(AttentionSpec(d_model=12, n_heads=3, dropout=0),
                                     ResidualClassifierConfig(centring="support_mean", n_layers=1))
    got = head(**values)["logits"]
    mean = values["support_feature"].mean(dim=1)
    query = values["query_feature"] - mean
    support = values["support_feature"] - mean[:, None]
    expected, _ = differentiable_neighbor_logits(
        query, support, values["support_bound"], values["support_mask"],
        values["candidate_mask"], temperature=.07,
    )
    assert torch.equal(got, expected)


def test_supportless_candidate_is_finite_text_only():
    values = _episode()
    values["support_mask"][:, values["support_bound"][0] == 1] = False
    head = ResidualSupportClassifier(AttentionSpec(d_model=12, n_heads=3, dropout=0),
                                     ResidualClassifierConfig(centring="none", n_layers=1))
    result = head(**values)
    assert torch.isfinite(result["logits"][:, 1]).all()
    uniform_log_prior = -torch.log(torch.tensor(3.0))
    assert torch.equal(result["logits"][:, 1], result["text_score"][:, 1] + uniform_log_prior)


def test_partial_enrollment_neutral_text_has_no_missing_candidate_bonus():
    values = _episode()
    values["support_mask"][:, values["support_bound"][0] == 1] = False
    head = ResidualSupportClassifier(AttentionSpec(d_model=12, n_heads=3, dropout=0),
                                     ResidualClassifierConfig(centring="none", n_layers=1)).eval()
    with torch.no_grad():
        head.p_text.weight.zero_()
        head.p_text.bias.zero_()
    logits = head(**values)["logits"]
    # The absent candidate receives the uniform log prior, not the arbitrary 0 that previously
    # beat every non-positive neighbour log-probability by construction.
    assert torch.equal(logits[:, 1], torch.full((2,), -torch.log(torch.tensor(3.0))))
    assert not logits[:, 1].eq(logits.max(dim=1).values).all()


def test_zero_support_episode_has_finite_text_scores_and_live_output_heads():
    values = _episode(k=0)
    head = ResidualSupportClassifier(AttentionSpec(d_model=12, n_heads=3, dropout=0),
                                     ResidualClassifierConfig(centring="support_mean", n_layers=1))
    result = head(**values)
    loss = result["logits"].sum()
    loss.backward()
    assert torch.isfinite(result["logits"]).all()
    assert head.r_candidate_head.weight.grad is not None
    assert head.p_text.weight.grad is not None


def test_mixed_regime_loss_is_invariant_to_query_order():
    episodes = [
        Episode(0, (), (), ("a", "b"), 0, "compatible", 0, False, True, support_set_id=0),
        Episode(1, (9,), (0,), ("a", "b"), 0, "compatible", 1, False, False, support_set_id=0),
        Episode(2, (9,), (0,), ("a", "b"), 0, "compatible", 1, False, False, support_set_id=1),
        Episode(3, (), (), ("a", "b"), 0, "compatible", 0, False, True, support_set_id=2),
    ]
    logits = torch.tensor([[0., 3.], [3., 0.], [2., 0.], [0., 5.]])
    text = {"candidate_mask": torch.ones(4, 2, dtype=torch.bool)}
    first = episode_loss(logits, episodes, text)["loss"]
    order = [1, 0, 2, 3]
    second = episode_loss(logits[order], [episodes[index] for index in order], text)["loss"]
    assert torch.equal(first, second)


def test_mixed_support_backward_reaches_every_trainable_parameter():
    values = _episode()
    values["support_mask"][:, values["support_bound"][0] == 1] = False
    head = ResidualSupportClassifier(
        AttentionSpec(d_model=12, n_heads=3, dropout=0),
        ResidualClassifierConfig(centring="support_mean", n_layers=1),
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = head(**values)
        torch.nn.functional.cross_entropy(result["logits"], torch.tensor([0, 2])).backward()
        optimizer.step()
    missing = [name for name, parameter in head.named_parameters() if parameter.grad is None]
    assert missing == []
    assert head.r_support_head.weight.grad.abs().sum() > 0
    assert head.r_candidate_head.weight.grad.abs().sum() > 0
    assert head.p_text.weight.grad.abs().sum() > 0


def test_text_bridge_uses_the_same_raw_query_with_or_without_residual_trunk():
    values = _episode(k=0)
    spec = AttentionSpec(d_model=12, n_heads=3, dropout=0)
    active = ResidualSupportClassifier(spec, ResidualClassifierConfig(n_layers=1))
    inactive = ResidualSupportClassifier(
        spec, ResidualClassifierConfig(n_layers=1, residual_enabled=False),
    )
    inactive.load_state_dict(active.state_dict())
    assert torch.equal(active(**values)["text_score"], inactive(**values)["text_score"])


def test_score_components_recompose_logits_exactly():
    values = _episode()
    head = ResidualSupportClassifier(
        AttentionSpec(d_model=12, n_heads=3, dropout=0),
        ResidualClassifierConfig(centring="support_mean", n_layers=1),
    )
    with torch.no_grad():
        head.r_support_head.weight.normal_()
        head.r_candidate_head.weight.normal_()
        head.lambda_table.fill_(0.4)
    result = head(**values)
    expected = result["metric_part"] + result["text_part"] + result["r_candidate"]
    assert torch.equal(result["logits"], expected)
    assert result["base_part"].shape == result["logits"].shape


def test_residual_off_equals_support_centred_neighbor_floor():
    values = _episode()
    head = ResidualSupportClassifier(
        AttentionSpec(d_model=12, n_heads=3, dropout=0),
        ResidualClassifierConfig(centring="support_mean", n_layers=1,
                                 residual_enabled=False, text_term_enabled=False),
    )
    got = head(**values)["logits"]
    mean = values["support_feature"].mean(dim=1)
    expected, _ = differentiable_neighbor_logits(
        values["query_feature"] - mean,
        values["support_feature"] - mean[:, None],
        values["support_bound"], values["support_mask"], values["candidate_mask"],
        temperature=.07,
    )
    assert torch.equal(got, expected)


def test_supportless_residual_and_text_ablation_fails_loudly():
    values = _episode(k=0)
    head = ResidualSupportClassifier(
        AttentionSpec(d_model=12, n_heads=3, dropout=0),
        ResidualClassifierConfig(residual_enabled=False, text_term_enabled=False),
    )
    with pytest.raises(ValueError, match="supportless candidates"):
        head(**values)


def test_text_projection_closed_form_recovers_known_linear_map():
    torch.manual_seed(9)
    x = torch.randn(64, 12, dtype=torch.float64)
    expected = torch.randn(12, 20, dtype=torch.float64)
    w, alpha = fit_text_projection(x, x @ expected, ridge_fraction=0.0)
    assert alpha == 0
    assert torch.allclose(w, expected, atol=1e-10, rtol=1e-10)


def test_checkpoint_round_trip_preserves_corpus_mean_and_logits():
    values = _episode()
    spec = AttentionSpec(d_model=12, n_heads=3, dropout=0)
    cfg = ResidualClassifierConfig(centring="corpus_mean", n_layers=1)
    original = ResidualSupportClassifier(spec, cfg).eval()
    original.set_corpus_mean(torch.randn(12))
    expected = original(**values)["logits"]
    restored = ResidualSupportClassifier(spec, cfg).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    assert torch.equal(restored.corpus_mean, original.corpus_mean)
    assert torch.equal(restored(**values)["logits"], expected)


def test_support_checkpoint_serializes_and_restores_its_dft_size():
    encoder, config = build_random_encoder(
        torch.device("cpu"), "fixed", neutral_acquisition_text=False,
        duration_range=(.5, 4.0), num_resolutions=4,
        frontend_kwargs={"use_polarization": True},
    )
    assert config["dft_size"] == encoder.filterbank.S == 512
    restored = build_encoder({
        "architecture_version": "support_classifier_v3", "config": config,
        "encoder": encoder.state_dict(),
    }, torch.device("cpu"))
    assert restored.filterbank.S == 512

    # Known v1/v2 support checkpoints wrote 512-point weights before the config field existed.
    legacy = dict(config)
    legacy.pop("dft_size")
    restored_legacy = build_encoder({
        "architecture_version": "support_classifier_v2", "config": legacy,
        "encoder": encoder.state_dict(),
    }, torch.device("cpu"))
    assert restored_legacy.filterbank.S == 512


def test_unfitted_corpus_mean_and_unimplemented_csls_fail_loudly():
    values = _episode()
    head = ResidualSupportClassifier(
        AttentionSpec(d_model=12, n_heads=3, dropout=0),
        ResidualClassifierConfig(centring="corpus_mean", n_layers=1),
    )
    with pytest.raises(ValueError, match="fitted corpus mean"):
        head(**values)
    with pytest.raises(ValueError, match="CSLS"):
        ResidualClassifierConfig(centring="csls")


def test_removing_top_support_matches_neighbor_floor_sensitivity():
    values = _episode(k=6, c=3)
    head = ResidualSupportClassifier(
        AttentionSpec(d_model=12, n_heads=3, dropout=0),
        ResidualClassifierConfig(centring="none", n_layers=1),
    )
    baseline = head(**values)["logits"].argmax(dim=1)
    similarity = torch.einsum(
        "bd,bkd->bk", torch.nn.functional.normalize(values["query_feature"], dim=-1),
        torch.nn.functional.normalize(values["support_feature"], dim=-1),
    )
    removed = {key: value.clone() if torch.is_tensor(value) else value for key, value in values.items()}
    removed["support_mask"][torch.arange(2), similarity.argmax(dim=1)] = False
    head_after = head(**removed)["logits"].argmax(dim=1)
    q = torch.nn.functional.normalize(values["query_feature"], dim=-1)
    s = torch.nn.functional.normalize(values["support_feature"], dim=-1)
    direct, _ = differentiable_neighbor_logits(
        q, s, values["support_bound"], removed["support_mask"], values["candidate_mask"],
        temperature=.07,
    )
    assert torch.equal(head_after != baseline, direct.argmax(dim=1) != baseline)


def test_regime_split_routes_and_shares_nothing():
    """Two complete heads, routed by whether an episode retains any support."""
    from model.support.residual_classifier import (
        RegimeSplitSupportClassifier, build_support_classifier,
    )
    values = _episode()
    spec = AttentionSpec(d_model=12, n_heads=3, dropout=0)
    cfg = ResidualClassifierConfig(centring="none", n_layers=1, regime_split=True)
    head = build_support_classifier(spec, cfg)
    assert isinstance(head, RegimeSplitSupportClassifier)
    assert not (set(id(p) for p in head.zero_head.parameters())
                & set(id(p) for p in head.few_head.parameters()))

    # Row 0 keeps its supports, row 1 is fully masked: they must take different heads.
    values["support_mask"] = values["support_mask"].clone()
    values["support_mask"][1] = False
    got = head(**values)["logits"]
    enrolled, _ = differentiable_neighbor_logits(
        values["query_feature"][:1], values["support_feature"][:1], values["support_bound"][:1],
        values["support_mask"][:1], values["candidate_mask"][:1], temperature=.07,
    )
    # Untrained: the enrolled row is still exactly the neighbour floor; the zero row is text only.
    assert torch.equal(got[:1], enrolled)
    assert torch.isfinite(got[1]).all()
    head.zero_head.p_text.weight.data.add_(1.0)
    moved = head(**values)["logits"]
    assert not torch.equal(moved[1], got[1])      # zero head owns the zero row
    assert torch.equal(moved[:1], got[:1])        # and cannot touch the enrolled row
