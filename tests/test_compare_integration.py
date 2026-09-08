"""Integration tests for the comparison line's data path (IMWUT handoff W6/W8).

These cover the seams that unit tests on synthetic corpora cannot reach: the bridge from the
sampler's recordings to positions in the real Phase-A dataset, and the zero-shot support draw that
decides which evaluation streams have a k=0 row at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from data.scripts.curate import deployment_policy
from data.scripts.curate.compatibility import are_compatible, stream_key


@dataclass
class _Stream:
    """Only the two fields the zero-shot support draw reads off an evaluation stream."""

    dataset: str
    stream: str


@pytest.fixture(scope="module")
def small_index():
    from training.tokenizer.pretrain_data import CorpusIndex

    return CorpusIndex(
        max_per_stream=40, seed=3, datasets=("dsads", "forth_trace"), alignment="native",
    )


def test_bridge_positions_address_the_real_dataset(small_index):
    """``window_index`` must be a position in the dataset the trainer builds, or batches misalign."""
    from training.compare.corpus import support_corpus_from_index

    corpus = support_corpus_from_index(small_index)
    assert len(corpus) > 0
    for recording in corpus.recordings[:200]:
        key = small_index.train[recording.window_index]
        ref = small_index.refs[key.stream_i]
        assert ref.dataset == recording.dataset
        assert ref.stream == recording.stream
        assert str(ref.subjects[key.window_i]) == recording.subject


def test_bridge_labels_match_the_corpus_vocabulary(small_index):
    from training.compare.corpus import support_corpus_from_index

    corpus = support_corpus_from_index(small_index)
    vocabulary = set(small_index.label_ids)
    assert {recording.label for recording in corpus.recordings} <= vocabulary


def test_bridge_keys_agree_with_the_stream_spec(small_index):
    from training.compare.corpus import support_corpus_from_index

    corpus = support_corpus_from_index(small_index)
    for index, (dataset, stream) in enumerate(corpus.stream_names):
        assert corpus.keys[index] == stream_key(dataset, stream)


def test_episodes_drawn_over_the_real_corpus_obey_the_rules(small_index):
    """The sampler's guarantees must survive contact with real subjects and executions."""
    import numpy as np

    from training.compare.corpus import support_corpus_from_index
    from training.compare.sampling import draw_batch

    corpus = support_corpus_from_index(small_index)
    episodes, telemetry = draw_batch(
        corpus, np.random.default_rng(0), batch_size=24, support_size=8, p_gt_present=0.5,
    )
    assert episodes
    for episode in episodes:
        query = corpus.recordings[episode.query]
        query_key = corpus.key_of(query)
        for index in episode.support:
            other = corpus.recordings[index]
            assert (other.dataset, other.subject, other.execution) != (
                query.dataset, query.subject, query.execution,
            )
            if episode.subject_relation == "same_subject":
                assert (other.dataset, other.subject) == (query.dataset, query.subject)
            else:
                assert (other.dataset, other.subject) != (query.dataset, query.subject)
            assert are_compatible(query_key, corpus.key_of(other))
        units = {
            (corpus.recordings[index].dataset, corpus.recordings[index].subject,
             corpus.recordings[index].execution)
            for index in episode.support
        }
        assert len(units) == len(episode.support)
    assert 0.0 <= telemetry["sampler/realised_gt_rate"] <= 1.0


@pytest.mark.parametrize("dataset,stream", [("upper_limb_use", "control_left_wrist")])
def test_streams_without_a_compatible_partner_are_reported_unsupported(dataset, stream):
    """The audit's finding, pinned so it cannot be silently 'fixed' by widening the relation.

    ``upper_limb_use`` mounts a research IMU on the wrist while every training wrist stream is a
    watch, so no training recording shares its configuration. The honest k=0 answer is that the
    deployed mechanism does not apply, not a different mechanism wearing its name.
    """
    from baselines.halo_compare.adapter import HALOCompareAdapter
    from training.compare.sampling import build_support_corpus

    corpus = build_support_corpus(
        deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, max_per_stream=20, seed=0,
    )
    state = {"zero_shot_support": {}, "_corpus": corpus}
    draws, reason = HALOCompareAdapter()._zero_shot_draws(
        _Stream(dataset, stream), state, ["walking", "reaching"],
    )
    assert draws is None
    assert "acquisition configuration" in reason


def test_zero_shot_support_excludes_every_candidate_label():
    """The support set must not contain the answer, or k=0 is not zero-shot."""
    from baselines.halo_compare.adapter import HALOCompareAdapter
    from training.compare.sampling import build_support_corpus

    corpus = build_support_corpus(
        deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, max_per_stream=200, seed=0,
    )
    state = {"zero_shot_support": {}, "_corpus": corpus}
    candidates = ["walking", "sitting", "standing"]
    draws, reason = HALOCompareAdapter()._zero_shot_draws(
        _Stream("inclusivehar", "phone_waist"), state, candidates,
    )
    assert draws is not None, reason
    from data.scripts.labels.canonical_labels import canonicalize

    banned = {canonicalize(c) for c in candidates}
    for rows in draws:
        executions = {
            (recording.dataset, recording.subject, recording.execution)
            for recording in rows
        }
        assert len(executions) == len(rows)
        for recording in rows:
            assert canonicalize(recording.label) not in banned


def test_zero_shot_support_excludes_synonyms_of_candidates():
    """A candidate written in the dataset's own words must ban its canonical training synonym.

    realworld's ``climbingup`` and tnda_har's ``ascending_stairs`` canonicalise to the training
    label ``walking_upstairs``. Under a string-only ban those rows survived and voted for the
    candidate through label text, which is enrolled support under another name (found 2026-09-05).
    """
    from baselines.halo_compare.adapter import HALOCompareAdapter
    from data.scripts.labels.canonical_labels import canonicalize
    from training.compare.sampling import build_support_corpus

    corpus = build_support_corpus(
        deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, max_per_stream=200, seed=0,
    )
    state = {"zero_shot_support": {}, "_corpus": corpus}
    candidates = ["climbingup", "climbingdown", "walking"]
    concepts = {canonicalize(c) for c in candidates}
    assert "walking_upstairs" in concepts, "the fixture no longer exercises a synonym"
    draws, reason = HALOCompareAdapter()._zero_shot_draws(
        _Stream("realworld", "phone_waist"), state, candidates,
    )
    assert draws is not None, reason
    drawn = {recording.label for rows in draws for recording in rows}
    assert "walking_upstairs" not in drawn and "walking_downstairs" not in drawn
    assert not ({canonicalize(label) for label in drawn} & concepts)


def test_zero_shot_support_is_configuration_compatible():
    from baselines.halo_compare.adapter import HALOCompareAdapter
    from training.compare.sampling import build_support_corpus

    corpus = build_support_corpus(
        deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, max_per_stream=200, seed=0,
    )
    state = {"zero_shot_support": {}, "_corpus": corpus}
    query_key = stream_key("inclusivehar", "phone_waist")
    draws, reason = HALOCompareAdapter()._zero_shot_draws(
        _Stream("inclusivehar", "phone_waist"), state, ["walking"],
    )
    assert draws is not None, reason
    for rows in draws:
        for recording in rows:
            assert are_compatible(query_key, stream_key(recording.dataset, recording.stream))


def test_zero_shot_support_never_draws_from_an_evaluation_dataset():
    """Support comes from TRAINING only; drawing from an eval dataset would leak."""
    from baselines.halo_compare.adapter import HALOCompareAdapter
    from training.compare.sampling import build_support_corpus

    corpus = build_support_corpus(
        deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, max_per_stream=200, seed=0,
    )
    state = {"zero_shot_support": {}, "_corpus": corpus}
    draws, _ = HALOCompareAdapter()._zero_shot_draws(
        _Stream("inclusivehar", "phone_waist"), state, ["walking"],
    )
    training = set(deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS)
    for rows in draws:
        for recording in rows:
            assert recording.dataset in training


def test_evaluation_datasets_are_absent_from_the_training_corpus():
    """The whole comparison rests on this; assert it rather than trusting the roster."""
    evaluation = set(deployment_policy.PRIMARY_EVAL_DATASETS)
    assert evaluation.isdisjoint(deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS)


# ---------------------------------------------------------------- ensemble
def test_draws_match_the_training_episode_shape():
    """The k=0 comparison must be the shape the sampler trains on, or it is off-distribution."""
    from baselines.halo_compare import adapter as A
    from training.compare.sampling import DEFAULT_LABEL_SUBSET, build_support_corpus

    corpus = build_support_corpus(
        deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, max_per_stream=200, seed=0,
    )
    state = {"zero_shot_support": {}, "_corpus": corpus}
    draws, reason = A.HALOCompareAdapter()._zero_shot_draws(
        _Stream("inclusivehar", "phone_waist"), state, ["walking"],
    )
    assert draws is not None, reason
    assert len(draws) == A.ZERO_SHOT_DRAWS
    low, high = DEFAULT_LABEL_SUBSET
    for rows in draws:
        labels = {r.label for r in rows}
        assert low <= len(labels) <= high, "draw label count is outside the trained range"


def test_draws_are_deterministic_and_not_all_identical():
    """Reproducible under the fixed seed, but genuinely different draws or there is no ensemble."""
    from baselines.halo_compare.adapter import HALOCompareAdapter
    from training.compare.sampling import build_support_corpus

    corpus = build_support_corpus(
        deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, max_per_stream=200, seed=0,
    )
    def draw():
        state = {"zero_shot_support": {}, "_corpus": corpus}
        got, _ = HALOCompareAdapter()._zero_shot_draws(
            _Stream("inclusivehar", "phone_waist"), state, ["walking"],
        )
        return [sorted({r.label for r in rows}) for rows in got]

    first, second = draw(), draw()
    assert first == second, "the k=0 row must be reproducible"
    assert len({tuple(labels) for labels in first}) > 1, "every draw picked the same labels"


def test_ensemble_modes_agree_on_a_unanimous_batch():
    """When every draw says the same thing, no combiner may disagree with it."""
    import torch

    from baselines.halo_compare.adapter import _ensemble

    logits = torch.zeros(5, 3, 4)
    logits[..., 2] = 10.0                      # every draw prefers candidate 2
    for mode in ("probability", "standardized", "logprob"):
        assert torch.equal(
            _ensemble(logits, mode).argmax(-1), torch.full((3,), 2),
        ), mode


def test_probability_ensemble_resists_a_single_extreme_draw():
    """The reason 'probability' is the default: one huge-logit draw must not decide the row."""
    import torch

    from baselines.halo_compare.adapter import _ensemble

    logits = torch.zeros(4, 1, 3)
    logits[:3, 0, 0] = 2.0                      # three draws mildly prefer candidate 0
    logits[3, 0, 1] = 500.0                     # one draw screams for candidate 1
    assert int(_ensemble(logits, "probability").argmax(-1)) == 0
    assert int(_ensemble(logits, "logprob").argmax(-1)) == 1   # documented sharper behaviour


def test_unknown_ensemble_mode_is_refused():
    import pytest as _pytest
    import torch

    from baselines.halo_compare.adapter import _ensemble

    with _pytest.raises(ValueError):
        _ensemble(torch.zeros(2, 1, 3), "average")


def test_accelerometer_only_streams_are_scattered_into_the_canonical_slots():
    """Regression: a 3-channel source raised inside the encoder's 6-slot contract.

    monipar (evaluation) and capture24 (the only training pool compatible with it) are both
    accelerometer-only, so this path is on the critical route for a real evaluation dataset. The
    pre-existing halo_compact adapter passes the native mask straight through and shares the defect.
    """
    import numpy as np

    from baselines.halo_compare.adapter import _six_slot

    native = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    windows, mask = _six_slot(native, ["acc_x", "acc_y", "acc_z"], np.array([True, True, True]))
    assert windows.shape == (2, 4, 6)
    assert mask.tolist() == [True, True, True, False, False, False]
    assert np.array_equal(windows[:, :, :3], native)
    assert not windows[:, :, 3:].any(), "absent channels must be exact zeros, never fabricated"


def test_six_slot_is_a_no_op_on_already_canonical_input():
    import numpy as np

    from baselines.halo_compare.adapter import _six_slot

    native = np.random.randn(2, 4, 6).astype(np.float32)
    mask = np.ones(6, dtype=bool)
    windows, out_mask = _six_slot(native, list("abcdef"), mask)
    assert np.array_equal(windows, native) and np.array_equal(out_mask, mask)


# ---------------------------------------------------------------- adaptation_v2
def test_adaptation_v2_roster_and_protocol():
    """The ten-dataset manifest, and the guard that stops v1 and v2 rows sharing a table."""
    import gzip
    import json
    from pathlib import Path

    from eval.enrollment_protocol import ADAPTATION_V1_DATASETS, ADAPTATION_V2_DATASETS

    path = Path("eval/manifests/adaptation_v2.json.gz")
    assert path.exists(), "adaptation_v2 has not been built"
    manifest = json.load(gzip.open(path))

    assert manifest["protocol_name"] == "halo_matched_adaptation_v2"
    assert set(manifest["datasets"]) == set(ADAPTATION_V2_DATASETS)
    assert set(ADAPTATION_V1_DATASETS) < set(ADAPTATION_V2_DATASETS), "v2 must be a superset"

    v1 = json.load(gzip.open("eval/manifests/adaptation_v1.json.gz"))
    assert v1["protocol_name"] != manifest["protocol_name"], (
        "v1 and v2 must carry different protocol names or the assembler will happily mix them"
    )


def test_adaptation_v2_rebalances_the_ordinary_regime():
    """The reason for adding them: ordinary activities were badly outnumbered."""
    import gzip
    import json
    from collections import Counter

    def ok_cells(path):
        manifest = json.load(gzip.open(path))
        return Counter(
            (c["kind"], c["regime"]) for c in manifest["cells"].values() if c["status"] == "ok"
        )

    v1, v2 = ok_cells("eval/manifests/adaptation_v1.json.gz"), ok_cells(
        "eval/manifests/adaptation_v2.json.gz")
    assert v2[("enrollment", "ordinary")] > v1[("enrollment", "ordinary")]
    assert v2[("zero_shot", "ordinary")] > v1[("zero_shot", "ordinary")]
    # Nothing may be lost from the clinical side by rebalancing.
    assert v2[("enrollment", "specialized_novel")] >= v1[("enrollment", "specialized_novel")]
    assert v2[("zero_shot", "specialized_novel")] >= v1[("zero_shot", "specialized_novel")]


def test_hapt_can_never_enter_an_evaluation_manifest():
    """hapt is the same 30 subjects as uci_har, which is in training. That is a direct leak."""
    import pytest as _pytest

    from eval.enrollment_protocol import NEVER_EVALUATE, build_manifest

    assert "hapt" in NEVER_EVALUATE
    with _pytest.raises(ValueError, match="never be evaluated"):
        build_manifest(datasets=("hapt",))


def test_manifest_records_streams_it_could_not_materialize():
    """A missing conversion must be visible, not silently dropped."""
    import gzip
    import json

    manifest = json.load(gzip.open("eval/manifests/adaptation_v2.json.gz"))
    assert "skipped_streams" in manifest
    # shoaib declares four placements; only phone_right_pocket has a native grid.
    assert any(name.startswith("shoaib/") for name in manifest["skipped_streams"])


def test_prefetch_loader_reproduces_the_synchronous_draw(small_index):
    """Worker processes must hand back exactly the episodes step-seeded drawing would produce,
    in any order of completion, with the collated batch covering those episodes' windows."""
    from training.compare.corpus import support_corpus_from_index
    from training.compare.sampling import draw_batch
    from training.compare.train import PrefetchLoader, episode_positions, episode_rng
    from training.tokenizer.episodic import EpisodicCollate
    from training.tokenizer.pretrain_data import PATCH_SECONDS, MultiScaleCollate, PretrainDataset

    corpus = support_corpus_from_index(small_index)
    dataset = PretrainDataset(
        small_index, small_index.train, augment=False, two_view=False,
        neutral_acquisition_text=True,
    )
    collate = EpisodicCollate(MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS))
    kwargs = dict(support_size=6, p_gt_present=0.5, label_subset=(2, 4), mode="compatible",
                  same_subject_probability=0.5)
    loader = PrefetchLoader(
        corpus, dataset, collate, data_seed=11, batch_size=3, draw_kwargs=kwargs,
        workers=2, depth=2, start_step=1,
    )
    try:
        for step in (1, 2, 3, 4):
            episodes, telemetry, batch = loader.get(step)
            expected, _ = draw_batch(corpus, episode_rng(11, step), batch_size=3, **kwargs)
            assert episodes == expected
            assert batch["patches"].shape[0] == len(episode_positions(episodes, corpus))
            assert "sampler/zero_shot_rate" in telemetry
    finally:
        loader.close()
    # Two different seeds do not share a sequence; the same seed does.
    a, _ = draw_batch(corpus, episode_rng(11, 2), batch_size=3, **kwargs)
    b, _ = draw_batch(corpus, episode_rng(12, 2), batch_size=3, **kwargs)
    assert a != b
