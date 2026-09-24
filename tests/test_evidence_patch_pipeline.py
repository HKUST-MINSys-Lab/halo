"""Regression tests for source-aware patch evidence, learned retrieval, and confidence."""


import numpy as np
import pytest
import torch
import torch.nn.functional as F

from data.scripts.eda.grid_io import GridRef
from model.evidence.confidence import (
    EvidenceConfidenceHead,
    binary_auprc,
    confidence_features,
    expected_calibration_error,
)
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from training.evidence.build_memory import archive_budget_balanced_keep, label_config_balanced_keep
from training.evidence.bank_guard import assert_patch_bank
from training.evidence.patch_episodes import (
    EpisodeMemoryView,
    PatchTable,
    assemble_evidence,
    build_episode_memory_view,
    describe_episode_composition,
    queries_from_encoded,
    simultaneous_stream_pairs,
)
from training.evidence.policy import (
    CANDIDATE_COUNT_RANGE,
    EPISODE_TYPES,
    LABEL_TEXT_MODES,
    PHYSICAL_VIEW_MODES,
    SUPPORT_COUNT_RANGE,
    PhaseBPolicy,
)
from training.evidence.live_encoder import PatchViewSpec, SourcePatchEncoder
from training.tokenizer.eval_transfer import encode_dataset_detailed
from training.evidence.train_patch_decoder import (
    AdaptationEpisodeSpec,
    EpisodeSampler,
    _episode_view_specs,
    checkpoint_eligibility,
    checkpoint_is_better,
    decoder_gradient_components,
    deranged_support_view,
    family_holdout_labels,
    load_activity_families,
    milestone_checkpoint_path,
    nested_support_view,
    parameter_gradient_norm,
    parameter_gradient_statistics,
    phase_b_source_fingerprint,
    prepare_support_feasible_query_pool,
    sample_queries,
    sample_queries_covering_labels,
    structured_fingerprint,
    validation_canary_cases,
)
from training.evidence.subject_style import SubjectStyle, apply_subject_style
from training.evidence.runtime_memory import build_enrollment_memory
from training.evidence.eval_enrollment import (
    _support_and_query_rows,
    build_paired_enrollment_plans,
    paired_subject_summary,
    phase_b_evaluation_source_fingerprint,
    predictor_logits,
    summarize_protocol_capabilities,
)
from training.evidence.build_comparison_table import assert_matched_evaluation_provenance
from training.evidence.device import resolve_device


def _bank():
    # Four windows, two patches each. Windows 0/1 are simultaneous placements of event 10.
    window = torch.arange(4).repeat_interleave(2)
    y = torch.tensor([0, 0, 1, 2])
    subj = torch.tensor([0, 0, 1, 2])
    cfg = torch.tensor([0, 1, 0, 1])
    event = torch.tensor([10, 10, 20, 30])
    verified = torch.tensor([True, True, False, False])
    patch = {
        "Z": torch.randn(8, 8).half(),
        "y": y[window],
        "subj": subj[window],
        "cfg": cfg[window],
        "sensor": cfg[window],
        "window": window,
        "event": event[window],
        "event_verified": verified[window],
        "time": torch.tensor([0.5, 1.5] * 4),
        "duration": torch.ones(8),
        "resolution": torch.tensor([0, 1] * 4),
        "ordinal": torch.tensor([0, 1] * 4),
    }
    return {
        "schema_version": 3,
        "population_fp": "test-population",
        "patch_embed_probe": torch.zeros(1),
        "Z": torch.randn(4, 8).half(),
        "y": y, "subj": subj, "cfg": cfg, "event": event,
        "event_verified": verified,
        "source_row": torch.arange(4),
        "source_alignment": "native",
        "cfg_rate_hz": {0: 50.0, 1: 100.0},
        "patch": patch,
    }


def _support_feasibility_bank(subjects):
    n = len(subjects)
    values = torch.arange(n)
    return {
        "y": torch.zeros(n, dtype=torch.long),
        "subj": torch.as_tensor(subjects, dtype=torch.long),
        "event": values,
        "event_verified": torch.ones(n, dtype=torch.bool),
        "patch": {
            "y": torch.zeros(n, dtype=torch.long),
            "subj": torch.as_tensor(subjects, dtype=torch.long),
            "window": values,
            "event": values,
            "event_verified": torch.ones(n, dtype=torch.bool),
        },
    }


def test_query_pool_reserves_all_requested_ordinary_support_units():
    bank = _support_feasibility_bank([0] * 9)
    labels, pool = prepare_support_feasible_query_pool(
        torch.arange(9), torch.arange(9), bank, torch.tensor([0, 1]),
        support_count=8, episode_type="ordinary_few_support",
        rng=np.random.default_rng(4),
    )
    # Label 1 is absent and is removed; the one remaining query unit leaves eight for support.
    assert labels.tolist() == [0]
    assert len(pool) == 1


def test_cross_subject_query_pool_leaves_requested_support_on_other_people():
    bank = _support_feasibility_bank([0] * 8 + [1])
    labels, pool = prepare_support_feasible_query_pool(
        torch.arange(9), torch.arange(9), bank, torch.tensor([0]),
        support_count=8, episode_type="cross_subject_few_support",
        rng=np.random.default_rng(7),
    )
    assert labels.tolist() == [0]
    assert pool.tolist() == [8]


def test_same_subject_query_pool_reserves_support_from_the_query_person():
    bank = _support_feasibility_bank([0] * 5 + [1] * 4)
    labels, pool = prepare_support_feasible_query_pool(
        torch.arange(9), torch.arange(9), bank, torch.tensor([0]),
        support_count=4, episode_type="same_subject_enrollment",
        rng=np.random.default_rng(7),
    )
    assert labels.tolist() == [0]
    assert len(pool) == 1
    assert len(torch.unique(bank["subj"][pool])) == 1


def test_patch_bank_foreign_keys_and_event_expansion():
    bank = _bank()
    assert_patch_bank(bank, context="test")
    table = PatchTable(bank)
    query = table.gather_queries(torch.tensor([0]), "cpu")
    assert query.mask.sum() == 4
    assert set(query.window[query.mask].tolist()) == {0, 1}
    assert set(query.sensor[query.mask].tolist()) == {0, 1}


def test_index_coreset_stratifies_label_by_config_and_resolution():
    bank = _bank()
    rows = PatchTable(bank).sample_index_rows(
        torch.ones(4, dtype=torch.bool), windows_per_label=2, rng=np.random.default_rng(1)
    )
    label0 = rows[bank["patch"]["y"][rows].eq(0)]
    groups = set(zip(
        bank["patch"]["cfg"][label0].tolist(),
        bank["patch"]["resolution"][label0].tolist(),
    ))
    assert groups == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_index_coreset_does_not_prefer_low_config_ids_when_budget_is_tight():
    bank = _bank()
    n = 20
    bank["Z"] = torch.randn(n, 8).half()
    bank["y"] = torch.zeros(n, dtype=torch.long)
    bank["subj"] = torch.arange(n)
    bank["cfg"] = torch.arange(n)
    bank["event"] = torch.arange(n)
    bank["event_verified"] = torch.zeros(n, dtype=torch.bool)
    bank["source_row"] = torch.arange(n)
    bank["patch"] = {
        "Z": torch.randn(n, 8).half(), "y": bank["y"], "subj": bank["subj"],
        "cfg": bank["cfg"], "sensor": bank["cfg"], "window": torch.arange(n),
        "event": bank["event"], "event_verified": bank["event_verified"],
        "time": torch.full((n,), 0.5), "duration": torch.ones(n),
        "resolution": torch.zeros(n, dtype=torch.long),
        "ordinal": torch.zeros(n, dtype=torch.long),
    }
    table = PatchTable(bank)
    first = table.sample_index_rows(
        torch.ones(n, dtype=torch.bool), 5, np.random.default_rng(1)
    )
    second = table.sample_index_rows(
        torch.ones(n, dtype=torch.bool), 5, np.random.default_rng(2)
    )
    assert len(first) == len(second) == 5
    assert set(first.tolist()) != set(range(5))
    assert set(first.tolist()) != set(second.tolist())


def test_memory_label_cap_is_configuration_balanced():
    labels = torch.tensor([0] * 110 + [1] * 5)
    configs = torch.tensor([0] * 100 + [1] * 10 + [0] * 5)
    keep = label_config_balanced_keep(labels, configs, max_per_label=20, seed=7)
    kept_label0 = configs[keep & labels.eq(0)]
    assert len(kept_label0) == 20
    assert torch.bincount(kept_label0, minlength=2).tolist() == [10, 10]
    assert int((keep & labels.eq(1)).sum()) == 5


def test_global_archive_budget_is_exact_and_preserves_rare_labels():
    labels = torch.tensor([0] * 100 + [1] * 10 + [2] * 2)
    configs = torch.tensor([0] * 50 + [1] * 50 + [0] * 10 + [1, 2])
    keep = archive_budget_balanced_keep(labels, configs, budget=40, seed=3)
    assert int(keep.sum()) == 40
    assert int((keep & labels.eq(2)).sum()) == 2
    assert set(configs[keep & labels.eq(0)].tolist()) == {0, 1}


def test_phase_b_policy_derives_query_driven_topk_from_evidence_budget():
    policy = PhaseBPolicy(evidence_budget=64)
    assert policy.topk_per_subspace(16) == 8
    assert policy.topk_per_subspace(32) == 4


def test_phase_b_source_fingerprint_binds_file_names_and_contents(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("VALUE = 1\n")
    second.write_text("VALUE = 1\n")
    original = phase_b_source_fingerprint([first, second])
    second.write_text("VALUE = 2\n")
    assert phase_b_source_fingerprint([first, second]) != original
    assert phase_b_source_fingerprint([second, first]) != original


def test_phase_b_evaluation_fingerprint_binds_evaluator_sources(tmp_path):
    first = tmp_path / "eval.py"
    first.write_text("PROTOCOL = 1\n")
    original = phase_b_evaluation_source_fingerprint([first])
    first.write_text("PROTOCOL = 2\n")
    assert phase_b_evaluation_source_fingerprint([first]) != original


def test_validation_canaries_cross_every_recipe_with_every_transfer_fold():
    recipes = [("zero", 0), ("few", 4)]
    folds = [("subject", torch.tensor([1])), ("config", torch.tensor([2])),
             ("joint", torch.tensor([3]))]
    cases = validation_canary_cases(recipes, folds)
    assert len(cases) == len(recipes) * len(folds)
    assert {(case[0], case[3]) for case in cases} == {
        (fold_name, recipe) for fold_name, _ in folds for recipe in recipes
    }


def test_comparison_rejects_missing_or_mixed_evaluator_provenance():
    common = {
        "evaluation_regime": "v1",
        "evaluation_source_fp": "source-a",
        "evaluation_protocol_fp": "protocol-a",
    }
    assert_matched_evaluation_provenance({
        "step0": {"meta": dict(common)}, "trained": {"meta": dict(common)},
    })
    with pytest.raises(SystemExit, match="evaluation_source_fp"):
        assert_matched_evaluation_provenance({
            "step0": {"meta": dict(common)},
            "trained": {"meta": {**common, "evaluation_source_fp": "source-b"}},
        })
    with pytest.raises(SystemExit, match="evaluation_protocol_fp"):
        assert_matched_evaluation_provenance({
            "step0": {"meta": dict(common)},
            "legacy": {"meta": {**common, "evaluation_protocol_fp": None}},
        })


def test_protocol_capabilities_do_not_overclaim_unsupported_transfers():
    protocol = {
        "same": {
            "status": "ok", "support_ceiling": 8,
            "subject_relation": "cross_subject",
            "configuration_relation": "same_configuration",
        },
        "pseudo_execution": {
            "status": "unverified_window_level_execution_ids", "support_ceiling": 0,
            "subject_relation": "same_subject",
            "configuration_relation": "same_configuration",
        },
    }
    capabilities = summarize_protocol_capabilities(protocol)
    assert capabilities["cross_subject_enrollment"]
    assert not capabilities["same_subject_enrollment"]
    assert not capabilities["cross_configuration_enrollment"]
    assert len(capabilities["limitations"]) == 2


def test_structured_fingerprint_detects_canary_content_and_order():
    first = {
        "rows": torch.tensor([1, 2]),
        "spec": ["coherent", 1],
    }
    changed = {
        "rows": torch.tensor([1, 3]),
        "spec": ["coherent", 1],
    }
    reordered = {
        "rows": torch.tensor([1, 2]),
        "spec": [1, "coherent"],
    }
    assert structured_fingerprint(first) == structured_fingerprint(first)
    assert structured_fingerprint(first) != structured_fingerprint(changed)
    assert structured_fingerprint(first) != structured_fingerprint(reordered)


def test_milestone_checkpoint_path_is_separate_and_step_specific(tmp_path):
    output = tmp_path / "predictor.pt"
    assert milestone_checkpoint_path(output, 200) == (
        tmp_path / "predictor.milestones" / "step_000200.pt"
    )


def test_explicit_cuda_request_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="pass --device cpu explicitly"):
        resolve_device("cuda")
    assert resolve_device("cpu") == torch.device("cpu")


def test_adaptation_episode_removes_candidate_background_and_equalizes_support():
    n_labels, windows_per_label = 3, 4
    y = torch.arange(n_labels).repeat_interleave(windows_per_label)
    n = len(y)
    window = torch.arange(n)
    patch = {
        "Z": torch.randn(n, 8), "y": y, "subj": torch.arange(n),
        "cfg": torch.zeros(n, dtype=torch.long), "sensor": torch.zeros(n, dtype=torch.long),
        "window": window, "event": window, "event_verified": torch.zeros(n, dtype=torch.bool),
        "time": torch.ones(n), "duration": torch.ones(n),
        "resolution": torch.zeros(n, dtype=torch.long), "ordinal": torch.zeros(n, dtype=torch.long),
    }
    bank = {
        "Z": torch.randn(n, 8), "y": y, "subj": torch.arange(n),
        "cfg": torch.zeros(n, dtype=torch.long), "event": window,
        "event_verified": torch.zeros(n, dtype=torch.bool), "patch": patch,
    }
    query = PatchTable(bank).gather_queries(torch.tensor([0, 4, 8]), "cpu")
    view = build_episode_memory_view(
        patch, torch.arange(n), query, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]),
        support_count=2, episode_type="ordinary_few_support", label_mode="random_alias",
        rng=np.random.default_rng(3),
    )
    assert view.support_units_per_candidate.tolist() == [2, 2, 2]
    for label in range(n_labels):
        candidate_rows = patch["y"].eq(label)
        assert torch.equal(
            view.allowed[0, 0, candidate_rows], view.support_mask[candidate_rows]
        )
    with pytest.raises(ValueError, match="unanswerable"):
        build_episode_memory_view(
            patch, torch.arange(n), query, torch.tensor([0, 1, 2]),
            torch.tensor([0, 1, 2]), support_count=0,
            episode_type="semantic_zero_support", label_mode="random_alias",
            rng=np.random.default_rng(3),
        )

    absent_query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")
    absent = build_episode_memory_view(
        patch, torch.arange(n), absent_query, torch.tensor([0]), torch.tensor([1, 2]),
        support_count=1, episode_type="ordinary_few_support", label_mode="coherent",
        rng=np.random.default_rng(4), truth_present=False,
    )
    assert not bool(absent.allowed[..., patch["y"].eq(0)].any())
    for label in (1, 2):
        label_rows = patch["y"].eq(label)
        assert torch.equal(
            absent.allowed[0, 0, label_rows], absent.support_mask[label_rows]
        )
    with pytest.raises(ValueError, match="cannot include"):
        build_episode_memory_view(
            patch, torch.arange(n), absent_query, torch.tensor([0]), torch.tensor([0, 1]),
            support_count=1, episode_type="ordinary_few_support", label_mode="coherent",
            rng=np.random.default_rng(4), truth_present=False,
        )


def test_episode_sampler_covers_the_declared_clean_ranges():
    sampler = EpisodeSampler(np.random.default_rng(4))
    specs = [
        spec for _ in range(1000)
        for spec in sampler.sample_batch(8)
    ]

    assert set(spec.episode_type for spec in specs) == set(EPISODE_TYPES)
    assert set(spec.physical_view_mode for spec in specs) == set(PHYSICAL_VIEW_MODES)
    assert set(spec.enrollment_shape for spec in specs) == {"zero", "full"}
    assert set(spec.label_mode for spec in specs) == set(LABEL_TEXT_MODES)

    candidate_counts = [spec.candidate_count for spec in specs]
    support_counts = [spec.support_count for spec in specs]
    assert (min(candidate_counts), max(candidate_counts)) == CANDIDATE_COUNT_RANGE
    assert (min(support_counts), max(support_counts)) == SUPPORT_COUNT_RANGE


def test_episode_sampler_invariants_hold_for_every_sampled_episode():
    sampler = EpisodeSampler(np.random.default_rng(9))
    specs = [
        spec for _ in range(200)
        for spec in sampler.sample_batch(8)
    ]
    for spec in specs:
        assert spec.label_mode == (
            "coherent" if spec.support_count == 0 else "random_alias"
        )
        assert spec.physical_view_mode == "clean"
        assert (spec.episode_type == "semantic_zero_support") == (spec.support_count == 0)


def test_adaptation_episode_spec_prevents_semantic_shortcut_and_unanswerable_aliases():
    with pytest.raises(ValueError, match="requires label_mode='random_alias'"):
        AdaptationEpisodeSpec(
            episode_type="ordinary_few_support", support_count=1,
            candidate_count=4, label_mode="coherent",
        )
    with pytest.raises(ValueError, match="requires label_mode='coherent'"):
        AdaptationEpisodeSpec(
            episode_type="semantic_zero_support", support_count=0,
            candidate_count=4, label_mode="random_alias",
        )


def test_episode_sampler_is_deterministic_and_takes_any_positive_batch():
    assert (
        EpisodeSampler(np.random.default_rng(17)).sample_batch(8)
        == EpisodeSampler(np.random.default_rng(17)).sample_batch(8)
    )
    assert len(EpisodeSampler(np.random.default_rng(1)).sample_batch(6)) == 6
    with pytest.raises(ValueError, match="positive"):
        EpisodeSampler(np.random.default_rng(1)).sample_batch(0)


def test_episode_sampler_has_no_step_dependent_curriculum():
    first = EpisodeSampler(np.random.default_rng(31)).sample_batch(100)
    second = EpisodeSampler(np.random.default_rng(31)).sample_batch(100)
    assert first == second
    assert {spec.support_count for spec in first} <= set(range(9))
    assert all(2 <= spec.candidate_count <= 16 for spec in first)


def test_partial_episode_query_draw_covers_enrolled_and_unenrolled_sides():
    y = torch.arange(8).repeat_interleave(3)
    pool = torch.arange(len(y))
    selected = sample_queries_covering_labels(
        pool,
        torch.arange(8),
        y,
        4,
        np.random.default_rng(5),
        config_ids=torch.zeros_like(y),
        subject_ids=torch.arange(len(y)),
        required_labels=torch.tensor([1, 6]),
    )
    assert {1, 6} <= set(y[selected].tolist())


def test_episode_view_specs_are_clean_identity_only():
    bank = _bank()
    query = PatchTable(bank).gather_queries(torch.tensor([2]), "cpu")
    support_mask = torch.zeros(8, dtype=torch.bool)
    support_mask[0] = True
    view = EpisodeMemoryView(
        allowed=torch.ones(1, query.Z.shape[1], 8, dtype=torch.bool),
        support_mask=support_mask,
        support_candidate=torch.zeros(8, dtype=torch.long),
        candidate_ids=torch.tensor([0, 1]),
        query_label=torch.tensor([1]),
        support_units_per_candidate=torch.tensor([1, 0]),
        episode_type="same_subject_enrollment",
        label_mode="coherent",
    )
    rows = torch.arange(8)
    clean_query, clean_support = _episode_view_specs(
        query, view, rows, bank["patch"], np.random.default_rng(1),
        physical_view_mode="clean",
    )
    assert all(spec == PatchViewSpec() for spec in clean_query + clean_support)

    with pytest.raises(ValueError, match="only clean"):
        _episode_view_specs(
            query, view, rows, bank["patch"], np.random.default_rng(1),
            physical_view_mode="augmented",
        )


def test_subject_style_preserves_gravity_scale_shape_and_masked_slots():
    rate = 50.0
    t = np.arange(300) / rate
    data = np.zeros((300, 6), dtype=np.float32)
    data[:, 0] = 0.2 * np.sin(2 * np.pi * 1.5 * t)
    data[:, 2] = 1.0
    before_mean = data[:, :3].mean(0)
    styled = apply_subject_style(
        data, rate, [True, True, True, False, False, False],
        SubjectStyle(1.08, 1.1, 0.9, 0.2),
    )
    assert styled.shape == data.shape
    assert np.isfinite(styled).all()
    assert np.linalg.norm(styled[:, :3].mean(0) - before_mean) < 0.05
    assert np.count_nonzero(styled[:, 3:]) == 0

    batch = np.stack([data, data.copy()])
    style = SubjectStyle(1.08, 1.1, 0.9, 0.2)
    batched = apply_subject_style(
        batch, rate, [True, True, True, False, False, False], style,
    )
    serial = np.stack([
        apply_subject_style(
            window, rate, [True, True, True, False, False, False], style,
        )
        for window in batch
    ])
    assert np.allclose(batched, serial, atol=1e-6, rtol=1e-6)


def test_runtime_enrollment_appends_equal_support_without_mutating_base_bank():
    bank = _bank()
    original_patch_count = len(bank["patch"]["Z"])
    encoded = {
        "patch_Z": torch.randn(8, 8),
        "patch_window": torch.arange(4).repeat_interleave(2),
        "patch_time": torch.tensor([0.5, 1.5] * 4),
        "patch_duration": torch.ones(8),
        "patch_resolution": torch.tensor([0, 1] * 4),
    }
    canonical = F.normalize(torch.randn(3, 6), dim=-1)
    candidates = F.normalize(torch.randn(2, 6), dim=-1)
    memory = build_enrollment_memory(
        bank, torch.arange(8), F.normalize(bank["patch"]["Z"].float(), dim=-1),
        encoded, torch.tensor([0, 1]), torch.tensor([0, 1]), canonical, candidates,
    )
    assert len(bank["patch"]["Z"]) == original_patch_count
    assert len(memory.index_rows) == 12
    assert memory.support_mask.sum() == 4
    assert memory.support_units_per_candidate.tolist() == [1, 1]
    query = queries_from_encoded(
        encoded, torch.tensor([2, 3]), "cpu", sensor_id=memory.runtime_sensor_id
    )
    view = memory.episode_view(query, torch.tensor([0, 1]), label_mode="coherent")
    assert view.allowed.all()
    assert torch.equal(view.query_label, memory.candidate_ids)
    assert bool(query.sensor.eq(memory.runtime_sensor_id).all())

    partial = build_enrollment_memory(
        bank, torch.arange(8), F.normalize(bank["patch"]["Z"].float(), dim=-1),
        encoded, torch.tensor([0, 1]), torch.tensor([0, 0]), canonical, candidates,
        support_subject_ids=torch.tensor([10, 11]),
    )
    assert partial.support_units_per_candidate.tolist() == [2, 0]
    assert set(partial.bank["patch"]["subj"][partial.support_mask.cpu()].tolist()) == {10, 11}
    identified_query = queries_from_encoded(
        encoded, torch.tensor([2, 3]), "cpu", sensor_id=partial.runtime_sensor_id,
        subject_ids=torch.tensor([10, 12]),
    )
    assert identified_query.subj[:, 0].tolist() == [10, 12]

    cross_config = build_enrollment_memory(
        bank, torch.arange(8), F.normalize(bank["patch"]["Z"].float(), dim=-1),
        encoded, torch.tensor([0, 1]), torch.tensor([0, 1]), canonical, candidates,
        support_subject_ids=torch.tensor([10, 12]),
        query_matches_support_config=False,
    )
    support_cfg = cross_config.bank["patch"]["cfg"][cross_config.support_mask.cpu()]
    assert not bool(support_cfg.eq(cross_config.runtime_sensor_id).any())

    zero = build_enrollment_memory(
        bank, torch.arange(8), F.normalize(bank["patch"]["Z"].float(), dim=-1),
        encoded, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        canonical, candidates,
    )
    assert len(zero.index_rows) == 8
    assert zero.support_units_per_candidate.tolist() == [0, 0]


def test_runtime_enrollment_can_exclude_candidate_concepts_from_base_archive():
    bank = _bank()
    encoded = {
        "patch_Z": torch.randn(8, 8),
        "patch_window": torch.arange(4).repeat_interleave(2),
        "patch_time": torch.tensor([0.5, 1.5] * 4),
        "patch_duration": torch.ones(8),
        "patch_resolution": torch.tensor([0, 1] * 4),
    }
    canonical = F.normalize(torch.randn(3, 6), dim=-1)
    candidates = F.normalize(torch.randn(2, 6), dim=-1)
    memory = build_enrollment_memory(
        bank, torch.arange(8), F.normalize(bank["patch"]["Z"].float(), dim=-1),
        encoded, torch.tensor([0, 1]), torch.tensor([0, 1]), canonical, candidates,
        excluded_base_labels=torch.tensor([0, 1]),
    )
    base_count = int((~memory.support_mask).sum())
    assert base_count == 2
    assert set(memory.bank["patch"]["y"][:base_count].tolist()) == {2}
    assert set(memory.bank["patch"]["y"][base_count:].tolist()) == {3, 4}


def test_same_subject_enrollment_excludes_the_whole_support_execution():
    labels = np.asarray(["walk"] * 6 + ["sit"] * 6, dtype=object)
    subjects = np.asarray(["s1"] * 12, dtype=object)
    executions = np.asarray(
        ["walk_a"] * 3 + ["walk_b"] * 3 + ["sit_a"] * 3 + ["sit_b"] * 3,
        dtype=object,
    )
    support, position, query = _support_and_query_rows(
        labels, subjects, executions, "s1", ["walk", "sit"],
        support_count=1, mode="same_subject", rng=np.random.default_rng(5),
    )
    assert position.tolist() == [0, 1]
    assert not set(executions[support]).intersection(executions[query])
    assert set(labels[query]) == {"walk", "sit"}


def test_paired_enrollment_plan_uses_nested_support_and_fixed_queries():
    labels, subjects, executions = [], [], []
    for subject in ("s1", "s2"):
        for label in ("walk", "sit"):
            for execution in range(3):
                labels.extend([label, label])
                subjects.extend([subject, subject])
                executions.extend([f"{subject}:{label}:{execution}"] * 2)
    plans, coverage = build_paired_enrollment_plans(
        np.asarray(labels, dtype=object),
        np.asarray(subjects, dtype=object),
        np.asarray(executions, dtype=object),
        ["walk", "sit"],
        requested_support=[0, 1, 2, 4], mode="same_subject", seed=7,
    )
    assert coverage["support_ceiling"] == 2
    assert coverage["subjects"] == 2
    for plan in plans:
        assert all(len(rows) == 2 for rows in plan.support_rows)
        support_executions = {
            executions[row] for rows in plan.support_rows for row in rows
        }
        query_executions = {executions[row] for row in plan.query_rows}
        assert support_executions.isdisjoint(query_executions)
        assert all(rows[:1] == (rows[0],) for rows in plan.support_rows)


def test_paired_enrollment_plan_supports_cross_configuration_sources():
    query_labels = np.repeat(
        np.asarray(["walk", "walk", "walk", "sit", "sit", "sit"], dtype=object), 2
    )
    query_subjects = np.asarray(["s1"] * 12, dtype=object)
    query_executions = np.repeat(
        np.asarray(["w0", "w1", "w2", "s0", "s1", "s2"], dtype=object), 2
    )
    support_labels = query_labels.copy()
    support_subjects = query_subjects.copy()
    support_executions = query_executions.copy()
    plans, coverage = build_paired_enrollment_plans(
        query_labels, query_subjects, query_executions, ["walk", "sit"],
        requested_support=[0, 1, 2], mode="same_subject", seed=11,
        support_labels=support_labels,
        support_subjects=support_subjects,
        support_execution_ids=support_executions,
    )
    assert coverage["support_ceiling"] == 2
    assert len(plans) == 1
    plan = plans[0]
    selected = {support_executions[row] for rows in plan.support_rows for row in rows}
    remaining = {query_executions[row] for row in plan.query_rows}
    assert selected.isdisjoint(remaining)
    assert set(query_labels[plan.query_rows]) == {"walk", "sit"}


def test_paired_enrollment_rejects_window_level_execution_ids():
    labels = np.asarray(["walk", "sit"] * 4, dtype=object)
    subjects = np.asarray(["s1"] * len(labels), dtype=object)
    executions = np.asarray([f"window_{index}" for index in range(len(labels))], dtype=object)
    plans, coverage = build_paired_enrollment_plans(
        labels, subjects, executions, ["walk", "sit"],
        requested_support=[0, 1, 2], mode="same_subject", seed=1,
    )
    assert plans == []
    assert coverage["status"] == "unverified_window_level_execution_ids"


def test_enrollment_summary_uses_paired_subject_differences():
    records = {
        "s1": {
            "f1_macro": 80.0, "identity_f1_macro": 60.0,
            "support_removed_f1_macro": 50.0,
            "support_label_shuffled_f1_macro": 40.0,
            "prototype_f1_macro": 70.0, "ridge_head_f1_macro": 75.0,
        },
        "s2": {
            "f1_macro": 40.0, "identity_f1_macro": 50.0,
            "support_removed_f1_macro": 45.0,
            "support_label_shuffled_f1_macro": 30.0,
            "prototype_f1_macro": 35.0, "ridge_head_f1_macro": 45.0,
        },
    }
    summary = paired_subject_summary(records, seed=3, samples=200)
    assert summary["learned_f1_macro"] == 60.0
    assert summary["adaptation_gain_over_identity"] == 5.0
    assert summary["gain_over_support_removed"] == 12.5
    assert summary["adaptation_gain_over_identity_ci95"] == [-10.0, 20.0]
    assert summary["independent_unit"] == "subject"


def test_gradient_telemetry_is_zero_before_tokenizer_finetuning_activates():
    module = torch.nn.Linear(3, 2)
    norm = parameter_gradient_norm(module.parameters(), torch.device("cpu"))
    assert norm.shape == ()
    assert float(norm) == 0.0


def test_gradient_statistics_are_size_normalized_and_report_disconnected_parameters():
    connected = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    disconnected = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    connected.grad = torch.tensor([3.0, 4.0])
    statistics = parameter_gradient_statistics(
        (connected, disconnected), torch.device("cpu")
    )
    assert statistics["norm"] == pytest.approx(5.0)
    assert statistics["rms"] == pytest.approx(2.5)
    assert statistics["coverage"] == pytest.approx(0.5)
    assert statistics["nonzero_fraction"] == pytest.approx(0.5)


def test_decoder_gradient_groups_are_disjoint_and_exhaustive():
    from model.evidence.relational_decoder import (
        RelationalDecoderConfig,
        RelationalEvidenceDecoder,
    )

    decoder = RelationalEvidenceDecoder(RelationalDecoderConfig(d_model=32, n_heads=4))
    groups = decoder_gradient_components(decoder)
    grouped = [id(parameter) for parameters in groups.values() for parameter in parameters]
    expected = [id(parameter) for parameter in decoder.parameters() if parameter.requires_grad]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(expected)
    assert {
        "time_projection", "acquisition_relation_embeddings", "input_layer_norm"
    } <= set(groups)


def test_source_patch_encoder_recovers_rows_once_and_preserves_gradients(tmp_path, monkeypatch):
    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.use_duration_embedding = False
            self.eval_resolution_pair = (0.5, 1.0)
            self.min_resolution_ratio = 1.75
            self.text_conditioning = "per_channel"

        def forward(self, patches, sampling_rate_hz, patch_len_samples, channel_texts, positions,
                    patch_padding_mask=None, **kwargs):
            scalar = patches.mean(dim=(2, 3)) * self.scale
            per_patch = scalar.unsqueeze(-1).repeat(1, 1, 4)
            mask = patch_padding_mask.to(per_patch.dtype)
            pooled = (per_patch * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            return {"pooled": pooled, "per_patch": per_patch}

    data = np.arange(2 * 120 * 6, dtype=np.float32).reshape(2, 120, 6)
    np.save(tmp_path / "data.npy", data)
    ref = GridRef(
        dataset="wisdm", stream="phone_pocket", alignment="native", rate_hz=50.0,
        channels=("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        mask=(True,) * 6, labels=("walking", "walking"), subjects=("a", "b"),
        event_ids=("a", "b"), event_ids_explicit=False, shape=data.shape, grid_dir=tmp_path,
    )
    monkeypatch.setattr("training.evidence.live_encoder.discover_grids", lambda _: [ref])
    encoder = TinyEncoder()
    encoded = encode_dataset_detailed(
        encoder, data, ["x"] * 6, torch.device("cpu"), 50.0,
        channel_mask=[True] * 6,
    )
    parent = encoded["patch_window"].long()
    counts = torch.bincount(parent, minlength=2)
    starts = counts.cumsum(0) - counts
    ordinal = torch.arange(len(parent)) - torch.repeat_interleave(starts, counts)
    bank = {
        "schema_version": 3, "d_model": 4,
        "source_alignment": "native",
        "Z": encoded["pooled"], "source_row": torch.arange(2),
        "cfg": torch.zeros(2, dtype=torch.long),
        "cfg_names": {0: ref.key}, "corpus": {"datasets": [ref.dataset]},
        "patch": {
            "Z": encoded["patch_Z"], "window": parent,
            "cfg": torch.zeros(len(parent), dtype=torch.long), "ordinal": ordinal,
            "time": encoded["patch_time"], "duration": encoded["patch_duration"],
            "resolution": encoded["patch_resolution"],
        },
    }
    source = SourcePatchEncoder(bank, "cpu", verify_fingerprint=False)
    rows = torch.tensor([0, len(parent) - 1, 0])
    live = source.encode_patch_rows(rows, encoder, requires_grad=True)
    live.square().mean().backward()
    assert live.shape == (3, 4)
    assert torch.equal(live[0], live[2])
    assert encoder.scale.grad is not None and float(encoder.scale.grad.abs()) > 0

    encoder.zero_grad(set_to_none=True)
    specs = [PatchViewSpec(None, 17), PatchViewSpec(None, 19), PatchViewSpec(None, 17)]
    augmented = source.encode_patch_rows_with_views(
        rows, specs, encoder, requires_grad=True
    )
    repeated = source.encode_patch_rows_with_views(
        rows, specs, encoder, requires_grad=True
    )
    serial = torch.cat([
        source.encode_patch_rows_with_views(
            rows[i:i + 1], specs[i:i + 1], encoder, requires_grad=True
        )
        for i in range(len(rows))
    ])
    assert torch.equal(augmented, repeated)
    assert torch.equal(augmented[0], augmented[2])
    assert torch.allclose(augmented, serial, atol=1e-6, rtol=1e-6)
    augmented.square().mean().backward()
    assert encoder.scale.grad is not None and float(encoder.scale.grad.abs()) > 0


def test_phase_b_queries_balance_configs_and_temper_subjects():
    # One label: config 0 has a 90/10 subject split, config 1 has ten windows. Hierarchical
    # sampling should make configs ~50/50 and the large subject ~75% within config 0, not 90%.
    y = torch.zeros(110, dtype=torch.long)
    cfg = torch.tensor([0] * 100 + [1] * 10)
    subj = torch.tensor([0] * 90 + [1] * 10 + [2] * 10)
    pool = torch.arange(len(y))
    draws = []
    rng = np.random.default_rng(8)
    for _ in range(2000):
        draws.append(int(sample_queries(
            pool, torch.tensor([0]), y, 1, rng,
            config_ids=cfg, subject_ids=subj, label_alpha=0.0,
        )[0]))
    drawn = torch.tensor(draws)
    config0_fraction = float(cfg[drawn].eq(0).float().mean())
    config0 = drawn[cfg[drawn].eq(0)]
    large_subject_fraction = float(subj[config0].eq(0).float().mean())
    assert 0.45 < config0_fraction < 0.55
    assert 0.70 < large_subject_fraction < 0.80


def test_external_query_packing_uses_noncolliding_structural_ids():
    encoded = {
        "patch_Z": torch.randn(5, 8),
        "patch_window": torch.tensor([0, 0, 1, 1, 1]),
        "patch_time": torch.tensor([0.5, 1.5, 0.5, 1.5, 2.5]),
        "patch_duration": torch.ones(5),
        "patch_resolution": torch.tensor([0, 1, 0, 0, 1]),
    }
    query = queries_from_encoded(encoded, torch.tensor([0, 1]), "cpu", sensor_id=-1)
    assert query.mask.sum(1).tolist() == [2, 3]
    assert (query.window[query.mask] < 0).all()
    assert query.sensor[query.mask].eq(-1).all()


def test_patch_bank_guard_rejects_wrong_parent_metadata():
    bank = _bank()
    bank["patch"]["event"][0] = 999
    with pytest.raises(SystemExit, match="patch.event disagrees"):
        assert_patch_bank(bank, context="test")


def test_patch_bank_guard_requires_valid_sensor_identity():
    bank = _bank()
    del bank["patch"]["sensor"]
    with pytest.raises(SystemExit, match="missing \\['sensor'\\]"):
        assert_patch_bank(bank, context="test")

    bank = _bank()
    bank["patch"]["sensor"][0] = -1
    with pytest.raises(SystemExit, match="nonnegative integer structural sensor ids"):
        assert_patch_bank(bank, context="test")


def test_subspace_retrieval_is_independent_masked_and_differentiable():
    torch.manual_seed(3)
    retriever = PatchSubspaceRetriever(8, n_subspaces=3, subspace_dim=4, ema_decay=0.5)
    query = F.normalize(torch.randn(2, 2, 8), dim=-1)
    memory = F.normalize(torch.randn(9, 8), dim=-1)
    index = retriever.build_index(memory)
    allowed = torch.ones(2, 2, 9, dtype=torch.bool)
    allowed[0, 0, :8] = False
    result = retriever.retrieve(query, index, allowed, k=4)
    assert result.index.shape == (2, 2, 3, 4)
    assert result.valid[0, 0].sum() == 3  # one eligible row independently in each head
    assert set(result.index[0, 0][result.valid[0, 0]].tolist()) == {8}
    score = retriever.score_selected(query, memory, result.index)
    score[result.valid].sum().backward()
    assert retriever.proj.grad is not None
    assert float(retriever.proj.grad.abs().sum()) > 0


def test_subspace_retrieval_ignores_padded_query_slots_without_requiring_memory():
    retriever = PatchSubspaceRetriever(8, 2, 4)
    query = F.normalize(torch.randn(1, 3, 8), dim=-1)
    memory = F.normalize(torch.randn(5, 8), dim=-1)
    allowed = torch.zeros(1, 3, 5, dtype=torch.bool)
    allowed[:, :2] = True
    result = retriever.retrieve(
        query, retriever.build_index(memory), allowed, k=2,
        query_mask=torch.tensor([[True, True, False]]),
    )
    assert result.valid[:, :2].all()
    assert not result.valid[:, 2].any()


def test_subspace_ema_update_tracks_the_online_projection():
    torch.manual_seed(4)
    retriever = PatchSubspaceRetriever(8, 3, 4, ema_decay=0.5)
    old = retriever.ema_proj.clone()
    with torch.no_grad():
        retriever.proj.add_(1)
    retriever.update_ema()
    assert torch.allclose(retriever.ema_proj, old.lerp(retriever.proj, 0.5))


def test_assemble_evidence_uses_learned_score_order_and_global_deduplication():
    from model.evidence.patch_retrieval import PatchRetrieval

    # Both heads select row 0. Only its highest-scoring occurrence may reach the roster.
    retrieval = PatchRetrieval(
        index=torch.tensor([[[[0, 2], [0, 3]]]]),
        score=torch.ones(1, 1, 2, 2),
        valid=torch.ones(1, 1, 2, 2, dtype=torch.bool),
    )
    online = torch.tensor([[[[0.9, 0.8], [0.95, 0.6]]]], requires_grad=True)
    result = assemble_evidence(
        retrieval, online, torch.arange(8), max_evidence=4, tau=0.1,
    )
    assert result.index[result.mask].tolist() == [0, 2, 3]
    assert result.head[result.mask].tolist() == [1, 0, 1]
    assert torch.allclose(result.weights.sum(1), torch.ones(1))
    assert len(torch.unique(result.head[result.mask])) == 2
    (result.weights * result.scores).sum().backward()
    assert online.grad is not None and bool((online.grad != 0).any())
    assert float(online.grad[0, 0, 0, 0]) == 0.0


def test_vectorized_evidence_assembly_matches_reference_scan():
    from model.evidence.patch_retrieval import PatchRetrieval

    generator = torch.Generator().manual_seed(23)
    B, Q, H, K, N = 3, 5, 4, 6, 17
    local = torch.randint(N, (B, Q, H, K), generator=generator)
    valid = torch.rand(B, Q, H, K, generator=generator) > 0.2
    valid[:, 0, 0, 0] = True
    score = torch.randn(B, Q, H, K, generator=generator)
    retrieval = PatchRetrieval(local, torch.zeros_like(score), valid)
    global_rows = torch.arange(N) * 3 + 11
    result = assemble_evidence(
        retrieval, score, global_rows, max_evidence=9, tau=0.2,
    )

    for b in range(B):
        reference = []
        for q in range(Q):
            for h in range(H):
                for k in range(K):
                    if valid[b, q, h, k]:
                        reference.append((
                            int(global_rows[local[b, q, h, k]]),
                            float(score[b, q, h, k]), h, q,
                        ))
        reference.sort(key=lambda value: value[1], reverse=True)
        unique = []
        seen = set()
        for row in reference:
            if row[0] not in seen:
                seen.add(row[0]); unique.append(row)
            if len(unique) == 9:
                break
        n = int(result.mask[b].sum())
        assert result.index[b, :n].tolist() == [row[0] for row in unique]
        assert result.head[b, :n].tolist() == [row[2] for row in unique]
        assert result.query_patch[b, :n].tolist() == [row[3] for row in unique]


def test_confidence_features_are_finite_and_candidate_entropy_is_normalized():
    evidence2 = torch.tensor([[1.0, 1.0]])
    evidence8 = torch.ones(1, 8)
    score = torch.tensor([[0.8, 0.7]])
    weight = torch.tensor([[0.5, 0.5]])
    votes2 = torch.ones(1, 2, 2)
    votes8 = torch.ones(1, 2, 8)
    f2 = confidence_features(evidence2, score, votes2, weight)
    f8 = confidence_features(evidence8, score, votes8, weight)
    assert torch.isfinite(f2).all() and torch.isfinite(f8).all()
    assert torch.allclose(f2[:, 2], f8[:, 2], atol=1e-6)  # normalized concentration
    head = EvidenceConfidenceHead()
    assert head(f2).shape == (1,)


def test_confidence_quality_metrics_reward_correct_ranking_and_calibration():
    target = np.asarray([0, 0, 1, 1], dtype=bool)
    good = np.asarray([0.05, 0.2, 0.8, 0.95])
    reversed_score = good[::-1]
    assert binary_auprc(good, target) == pytest.approx(1.0)
    assert binary_auprc(reversed_score, target) < 0.5
    assert expected_calibration_error(good, target) < 0.2


def _single_stream_bank():
    """One label recorded on ONE stream by two different people, plus a distractor label."""
    window = torch.arange(4).repeat_interleave(2)
    y = torch.tensor([0, 0, 1, 1])
    subj = torch.tensor([0, 1, 0, 1])
    cfg = torch.zeros(4, dtype=torch.long)          # every window on the same stream
    event = torch.arange(4)
    verified = torch.zeros(4, dtype=torch.bool)
    patch = {
        "Z": torch.randn(8, 8).half(), "y": y[window], "subj": subj[window],
        "cfg": cfg[window], "sensor": cfg[window], "window": window,
        "event": event[window], "event_verified": verified[window],
        "time": torch.tensor([0.5, 1.5] * 4), "duration": torch.ones(8),
        "resolution": torch.tensor([0, 1] * 4), "ordinal": torch.tensor([0, 1] * 4),
    }
    return {"schema_version": 3, "Z": torch.randn(4, 8).half(), "y": y, "subj": subj,
            "cfg": cfg, "event": event, "event_verified": verified,
            "source_row": torch.arange(4), "patch": patch}


def test_cross_subject_support_works_for_a_label_recorded_on_only_one_stream():
    # 35 of the 93 corpus labels exist on a single stream. Requiring acquisition disjointness would
    # drop every one of them from this regime; only the person has to differ.
    bank = _single_stream_bank()
    query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")   # label 0, subject 0
    view = build_episode_memory_view(
        bank["patch"], torch.arange(8), query, torch.tensor([0]), torch.tensor([0]),
        support_count=1, episode_type="cross_subject_few_support",
        label_mode="coherent", rng=np.random.default_rng(0),
    )
    support = view.support_rows
    assert len(support), "a single-stream label must still be able to supply cross-subject support"
    assert set(bank["patch"]["subj"][support].tolist()) == {1}, "support must be another person"
    assert set(bank["patch"]["cfg"][support].tolist()) == {0}, "same stream is allowed, not excluded"


def test_same_subject_enrollment_uses_a_distinct_execution_from_the_same_real_person():
    window = torch.arange(4)
    subject = torch.tensor([0, 0, 1, 1])
    patch = {
        "Z": torch.randn(4, 8), "y": torch.zeros(4, dtype=torch.long), "subj": subject,
        "cfg": torch.zeros(4, dtype=torch.long), "sensor": torch.zeros(4, dtype=torch.long),
        "window": window, "event": window, "event_verified": torch.zeros(4, dtype=torch.bool),
        "time": torch.ones(4), "duration": torch.ones(4),
        "resolution": torch.zeros(4, dtype=torch.long), "ordinal": torch.zeros(4, dtype=torch.long),
    }
    bank = {
        "Z": torch.randn(4, 8), "y": patch["y"], "subj": subject,
        "cfg": patch["cfg"], "event": window,
        "event_verified": patch["event_verified"], "patch": patch,
    }
    query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")
    view = build_episode_memory_view(
        patch, torch.arange(4), query, torch.tensor([0]), torch.tensor([0]),
        support_count=1, episode_type="same_subject_enrollment",
        label_mode="coherent", rng=np.random.default_rng(2),
    )
    support = view.support_rows
    assert patch["subj"][support].tolist() == [0]
    assert patch["window"][support].tolist() == [1]


def test_episode_composition_names_what_the_support_actually_was():
    bank = _single_stream_bank()
    query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")
    view = build_episode_memory_view(
        bank["patch"], torch.arange(8), query, torch.tensor([0]), torch.tensor([0]),
        support_count=1, episode_type="cross_subject_few_support",
        label_mode="coherent", rng=np.random.default_rng(0),
    )
    report = describe_episode_composition(bank["patch"], torch.arange(8), query, view)
    assert report["enrolled_examples"] == 1
    assert report["performed_by_a_different_person"] == 1
    assert report["performed_by_the_same_person"] == 0
    assert report["from_the_query_s_own_stream"] == 1
    assert report["from_a_second_stream_worn_simultaneously"] == 0
    assert report["synthetic_persona"].startswith("a different persona")


def test_simultaneous_stream_pairs_separate_synchronous_from_independent_streams():
    # Streams 0/1 share event 10 (captured together); stream 2 covers the same activity alone.
    cfg = torch.tensor([0, 1, 2, 2])
    event = torch.tensor([10, 10, 20, 30])
    assert simultaneous_stream_pairs(cfg, event) == {frozenset((0, 1))}


def test_composition_separates_paired_rig_support_from_independent_stream_support():
    # Streams 0 and 1 are a worn-together rig (event 10 spans both). The query is a LATER execution
    # on stream 0, so its support on stream 1 is a different session on the paired device.
    window = torch.arange(4).repeat_interleave(2)
    y = torch.zeros(4, dtype=torch.long)
    subj = torch.tensor([0, 0, 1, 1])
    cfg = torch.tensor([0, 1, 1, 0])
    event = torch.tensor([10, 10, 20, 30])
    verified = torch.tensor([True, True, False, False])
    patch = {
        "Z": torch.randn(8, 8).half(), "y": y[window], "subj": subj[window],
        "cfg": cfg[window], "sensor": cfg[window], "window": window,
        "event": event[window], "event_verified": verified[window],
        "time": torch.tensor([0.5, 1.5] * 4), "duration": torch.ones(8),
        "resolution": torch.tensor([0, 1] * 4), "ordinal": torch.tensor([0, 1] * 4),
    }
    bank = {"schema_version": 3, "Z": torch.randn(4, 8).half(), "y": y, "subj": subj,
            "cfg": cfg, "event": event, "event_verified": verified,
            "source_row": torch.arange(4), "patch": patch}
    pairs = simultaneous_stream_pairs(bank["cfg"], bank["event"])
    assert pairs == {frozenset((0, 1))}
    query = PatchTable(bank).gather_queries(torch.tensor([3]), "cpu")   # stream 0, subject 1
    view = build_episode_memory_view(
        bank["patch"], torch.arange(8), query, torch.tensor([0]), torch.tensor([0]),
        support_count=1, episode_type="ordinary_few_support",
        label_mode="coherent", rng=np.random.default_rng(1),
    )
    report = describe_episode_composition(bank["patch"], torch.arange(8), query, view, pairs)
    assert report["enrolled_examples"] == 1
    assert report["from_a_second_stream_worn_simultaneously"] == 1
    assert report["from_the_query_s_own_stream"] == 0
    assert report["synthetic_persona"] == "none applied"


def test_composition_reports_both_own_and_paired_streams_in_one_support_event():
    window = torch.arange(4).repeat_interleave(2)
    y = torch.zeros(4, dtype=torch.long)
    subj = torch.tensor([0, 0, 1, 1])
    cfg = torch.tensor([0, 1, 0, 1])
    event = torch.tensor([10, 10, 20, 20])
    verified = torch.ones(4, dtype=torch.bool)
    patch = {
        "Z": torch.randn(8, 8).half(), "y": y[window], "subj": subj[window],
        "cfg": cfg[window], "sensor": cfg[window], "window": window,
        "event": event[window], "event_verified": verified[window],
        "time": torch.tensor([0.5, 1.5] * 4), "duration": torch.ones(8),
        "resolution": torch.tensor([0, 1] * 4), "ordinal": torch.tensor([0, 1] * 4),
    }
    bank = {
        "schema_version": 3, "Z": torch.randn(4, 8).half(), "y": y, "subj": subj,
        "cfg": cfg, "event": event, "event_verified": verified,
        "source_row": torch.arange(4), "patch": patch,
    }
    pairs = simultaneous_stream_pairs(bank["cfg"], bank["event"])
    query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")
    view = build_episode_memory_view(
        bank["patch"], torch.arange(8), query, torch.tensor([0]), torch.tensor([0]),
        support_count=1, episode_type="ordinary_few_support",
        label_mode="coherent", rng=np.random.default_rng(2),
    )
    report = describe_episode_composition(bank["patch"], torch.arange(8), query, view, pairs)
    assert report["enrolled_examples"] == 1
    assert report["from_the_query_s_own_stream"] == 1
    assert report["from_a_second_stream_worn_simultaneously"] == 1


def test_composition_does_not_attribute_distractor_support_to_unrelated_queries():
    bank = _single_stream_bank()
    query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")
    view = build_episode_memory_view(
        bank["patch"], torch.arange(8), query, torch.tensor([0]), torch.tensor([0, 1]),
        support_count=1, episode_type="ordinary_few_support",
        label_mode="coherent", rng=np.random.default_rng(3),
    )
    report = describe_episode_composition(bank["patch"], torch.arange(8), query, view)
    assert report["enrolled_examples_without_matching_query"] == 1
    assert report["performed_by_a_different_person"] == 1
    assert report["from_the_query_s_own_stream"] == 1


def test_activity_family_mapping_is_complete_and_drives_validation_holdout():
    """Families no longer pick distractors, but they still define the held-out validation family."""
    import json

    vocab = json.load(open("data/labels/global_labels.json"))["labels"]
    family_ids, families, fingerprint = load_activity_families(vocab)
    assert len(family_ids) == len(vocab)
    assert len(fingerprint) == 16
    heldout = family_holdout_labels(family_ids, torch.arange(len(vocab)), 1)
    chosen_family = torch.unique(family_ids[heldout])
    assert len(chosen_family) == 1
    expected = torch.nonzero(family_ids.eq(chosen_family[0]), as_tuple=True)[0]
    assert set(heldout.tolist()) == set(expected.tolist())


def _mixed_overlap_bank(n_labels=3, windows_per_label=4):
    """Minimal synthetic patch bank: one window per patch, one subject per window."""
    y = torch.arange(n_labels).repeat_interleave(windows_per_label)
    n = len(y)
    window = torch.arange(n)
    patch = {
        "Z": torch.randn(n, 8), "y": y, "subj": torch.arange(n),
        "cfg": torch.zeros(n, dtype=torch.long), "sensor": torch.zeros(n, dtype=torch.long),
        "window": window, "event": window, "event_verified": torch.zeros(n, dtype=torch.bool),
        "time": torch.ones(n), "duration": torch.ones(n),
        "resolution": torch.zeros(n, dtype=torch.long), "ordinal": torch.zeros(n, dtype=torch.long),
    }
    bank = {
        "Z": torch.randn(n, 8), "y": y, "subj": torch.arange(n),
        "cfg": torch.zeros(n, dtype=torch.long), "event": window,
        "event_verified": torch.zeros(n, dtype=torch.bool), "patch": patch,
    }
    return bank, patch, n


def test_partially_enrolled_episode_keeps_unenrolled_candidates_out_of_memory():
    """Fix 0: some candidates enrolled, the rest recognizable only by name + related background.

    This is the regime the evidence engine exists for. With a uniform support count every episode
    is either fully un-enrolled (no substrate at all) or fully enrolled (a prototype over the
    declared support is optimal and neither retrieval nor language is needed).
    """
    # FIVE labels but only three candidates, so labels 3 and 4 are genuine background and the
    # "background survives" assertion below is not vacuous.
    bank, patch, n = _mixed_overlap_bank(n_labels=5)
    query = PatchTable(bank).gather_queries(torch.tensor([0, 4, 8]), "cpu")
    view = build_episode_memory_view(
        patch, torch.arange(n), query, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]),
        support_count=[2, 0, 1], episode_type="ordinary_few_support", label_mode="coherent",
        rng=np.random.default_rng(7),
    )
    assert view.support_units_per_candidate.tolist() == [2, 0, 1]
    # Candidate 1 is un-enrolled: it has NO support rows and its concept stays erased from
    # background memory, so it can never be recognized by retrieving its own stored examples.
    unenrolled = patch["y"].eq(1)
    assert not bool(view.support_mask[unenrolled].any())
    assert not bool(view.allowed[..., unenrolled].any())
    # The enrolled candidates are visible only through their own restored support.
    for label, expected in ((0, 2), (2, 1)):
        rows = patch["y"].eq(label)
        assert torch.equal(view.allowed[0, 0, rows], view.support_mask[rows])
        assert int(view.support_candidate[view.support_mask & rows][0]) == (0 if label == 0 else 2)
        assert len(torch.unique(torch.arange(n)[view.support_mask & rows])) >= expected
    # Non-candidate background (labels 3 and 4) is untouched and stays available as the semantic
    # bridge the un-enrolled candidate has to be recognized through.
    background = ~torch.isin(patch["y"], torch.tensor([0, 1, 2]))
    assert int(background.sum()) == 8
    assert bool(view.allowed[..., background].all())


def test_partial_enrollment_rejects_meaningless_names_and_length_mismatch():
    bank, patch, n = _mixed_overlap_bank()
    query = PatchTable(bank).gather_queries(torch.tensor([0, 4, 8]), "cpu")
    common = dict(
        patch=patch, index_rows=torch.arange(n), query=query,
        query_label=torch.tensor([0, 1, 2]), candidates=torch.tensor([0, 1, 2]),
        rng=np.random.default_rng(0),
    )
    # An un-enrolled candidate under an episode-local alias carries no information at all.
    with pytest.raises(ValueError, match="unanswerable"):
        build_episode_memory_view(
            **common, support_count=[1, 0, 1],
            episode_type="ordinary_few_support", label_mode="random_alias",
        )
    with pytest.raises(ValueError, match="3 candidates"):
        build_episode_memory_view(
            **common, support_count=[1, 1],
            episode_type="ordinary_few_support", label_mode="coherent",
        )
    with pytest.raises(ValueError, match="semantic_zero_support requires"):
        build_episode_memory_view(
            **common, support_count=[0, 1, 0],
            episode_type="semantic_zero_support", label_mode="coherent",
        )


def test_decoder_score_temperature_matches_the_retrieval_temperature():
    """The decoder divides the retrieval score by its own constant before using it as an attention
    bias, while `assemble_evidence` divides the same scores by `RETRIEVAL_TEMPERATURE` to form the
    pool weights that telemetry reports. If the two drift apart, `pool_normalized_entropy` silently
    stops describing how concentrated the decoder's attention actually is — and that metric is what
    the `pool_collapse` alert watches. The model package cannot import the training package, so the
    constant is duplicated; this is the guard on that duplication.
    """
    from model.evidence.relational_decoder import RelationalDecoderConfig
    from training.evidence.policy import RETRIEVAL_TEMPERATURE

    assert RelationalDecoderConfig().score_temperature == RETRIEVAL_TEMPERATURE


def test_training_and_deployment_use_one_fixed_retrieval_recipe():
    policy = PhaseBPolicy(evidence_budget=64)
    assert policy.as_dict()["training_retrieval"] == "fixed_and_identical_to_deployment"
    assert policy.topk_per_subspace(32) == policy.topk_per_subspace(
        32, evidence_budget=64
    )
    with pytest.raises(ValueError, match="below the deployment"):
        policy.topk_per_subspace(8, evidence_budget=32)


def test_checkpoint_selection_requires_mechanism_evidence_then_ranks_quality():
    ineligible = {"adaptation_selection_score": 0.9, "checkpoint_eligible": False}
    fallback = {"adaptation_selection_score": 0.1, "checkpoint_eligible": False}
    worse = {"adaptation_selection_score": 0.05, "checkpoint_eligible": True}
    better = {"adaptation_selection_score": 0.2, "checkpoint_eligible": True}
    best = {"adaptation_selection_score": 0.3, "checkpoint_eligible": True}

    assert not checkpoint_is_better(ineligible, fallback)
    assert checkpoint_is_better(better, fallback)
    assert checkpoint_is_better(best, better)
    assert not checkpoint_is_better(worse, better)
    assert not checkpoint_is_better(better, best)
    assert not checkpoint_is_better(dict(better), better)

    eligible, checks = checkpoint_eligibility({
        "selection_low_k_adaptation_gain": 0.0,
        "support_removal_true_probability_drop": 0.01,
        "support_label_shuffle_true_probability_drop": 0.02,
    })
    assert eligible and all(checks.values())
    failed, failed_checks = checkpoint_eligibility({
        "selection_low_k_adaptation_gain": -0.01,
        "support_removal_true_probability_drop": 0.01,
        "support_label_shuffle_true_probability_drop": 0.02,
    })
    assert not failed and not failed_checks["matches_closed_form_low_k_control"]


def test_artifact_predictor_mode_selects_the_declared_path():
    learned = torch.tensor([[1.0, 0.0]])
    control = torch.tensor([[0.0, 2.0]])
    aux = {"identity_logits": control}
    assert predictor_logits(learned, aux, "relational_decoder") is learned
    assert predictor_logits(learned, aux, "closed_form_retrieval_vote") is control
    with pytest.raises(ValueError, match="unknown predictor mode"):
        predictor_logits(learned, aux, "invalid")


def test_nested_support_views_are_subsets_and_derangement_has_no_fixed_binding():
    bank = _bank()
    support_mask = torch.tensor([True, True, False, False, True, True, False, False])
    support_candidate = torch.tensor([0, 0, -1, -1, 1, 1, -1, -1])
    full = EpisodeMemoryView(
        allowed=torch.ones(1, 2, 8, dtype=torch.bool),
        support_mask=support_mask,
        support_candidate=support_candidate,
        candidate_ids=torch.tensor([0, 1]),
        query_label=torch.tensor([0]),
        support_units_per_candidate=torch.tensor([1, 1]),
        episode_type="ordinary_few_support",
        label_mode="coherent",
    )
    zero = nested_support_view(full, bank["patch"], torch.arange(8), 0)
    one = nested_support_view(full, bank["patch"], torch.arange(8), 1)
    assert not bool(zero.support_mask.any())
    assert torch.equal(one.support_mask, full.support_mask)
    assert bool((zero.allowed & full.support_mask.view(1, 1, -1)).any()) is False
    shuffled = deranged_support_view(one)
    assert torch.equal(shuffled.support_mask, one.support_mask)
    assert bool((shuffled.support_candidate[one.support_mask]
                 != one.support_candidate[one.support_mask]).all())
