"""Support-set sampler tests (IMWUT handoff W5).

The sampler *is* the contribution, so its rules are pinned harder than usual. Two of these tests
exist because the corresponding mistake was made before in this repo: support that shares the
query's execution (the leakage unit), and a support set padded to size with rows that do not
belong.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.scripts.curate.compatibility import (
    AcquisitionKey,
    are_compatible,
    is_near_miss,
)
from training.compare.sampling import (
    Recording,
    SupportCorpus,
    draw_batch,
    draw_episode,
)


def _key(site: str, family: str = "watch", gravity: str = "present") -> AcquisitionKey:
    return AcquisitionKey(
        device_family=family, site=site,
        channels=("acc_x", "acc_y", "acc_z"), gravity_state=gravity,
    )


def _corpus(
    *,
    labels=("walking", "running", "sitting", "standing"),
    subjects_per_label=4,
    windows_per_subject=3,
    sites=("left_wrist",),
) -> SupportCorpus:
    """A synthetic corpus with known structure, so every rule is checkable by construction."""
    recordings: list[Recording] = []
    keys: list[AcquisitionKey] = []
    stream_names: list[tuple[str, str]] = []
    for stream_index, site in enumerate(sites):
        keys.append(_key(site))
        stream_names.append((f"ds{stream_index}", site))
        for label in labels:
            for subject in range(subjects_per_label):
                for window in range(windows_per_subject):
                    recordings.append(Recording(
                        stream_index=stream_index,
                        window_index=len(recordings),
                        dataset=f"ds{stream_index}",
                        stream=site,
                        label=label,
                        subject=f"{site}-s{subject}",
                        execution=f"{site}-{label}-s{subject}",
                    ))
    corpus = SupportCorpus(recordings=recordings, keys=keys, stream_names=stream_names)
    for index, recording in enumerate(recordings):
        key = keys[recording.stream_index]
        corpus.by_key.setdefault(key, []).append(index)
        corpus.by_key_label.setdefault((key, recording.label), []).append(index)
    distinct = list(corpus.by_key)
    for key in distinct:
        corpus.near_miss_keys[key] = [o for o in distinct if is_near_miss(key, o)]
    return corpus


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_query_is_never_in_its_own_support():
    corpus = _corpus()
    rng = _rng()
    for _ in range(200):
        episode = draw_episode(corpus, rng, support_size=8)
        assert episode is not None
        assert episode.query not in episode.support


def test_cross_subject_support_is_subject_disjoint_from_the_query():
    corpus = _corpus()
    rng = _rng(1)
    for _ in range(200):
        episode = draw_episode(corpus, rng, support_size=8, same_subject_probability=0.0)
        assert episode is not None
        query = corpus.recordings[episode.query]
        for index in episode.support:
            other = corpus.recordings[index]
            assert (other.dataset, other.subject) != (query.dataset, query.subject)


def test_same_subject_enrollment_uses_a_distinct_execution():
    corpus = _corpus(windows_per_subject=1)
    # Give every label two physical repetitions for each person.
    expanded = []
    for recording in corpus.recordings:
        expanded.extend([
            recording,
            Recording(**{**recording.__dict__, "window_index": len(corpus.recordings) + len(expanded),
                         "execution": recording.execution + "-repeat"}),
        ])
    corpus = SupportCorpus(expanded, corpus.keys, corpus.stream_names)
    rng = _rng(101)
    seen = 0
    for _ in range(200):
        episode = draw_episode(
            corpus, rng, support_size=8, p_gt_present=1.0,
            same_subject_probability=1.0,
        )
        assert episode is not None
        if episode.subject_relation != "same_subject":
            continue
        seen += 1
        query = corpus.recordings[episode.query]
        for index in episode.support:
            other = corpus.recordings[index]
            assert (other.dataset, other.subject) == (query.dataset, query.subject)
            assert other.execution != query.execution
    assert seen > 0


def test_support_never_shares_the_query_execution():
    """The leakage unit. Two blocks of one capture are not independent examples."""
    corpus = _corpus()
    rng = _rng(2)
    for _ in range(200):
        episode = draw_episode(corpus, rng, support_size=8)
        query = corpus.recordings[episode.query]
        for index in episode.support:
            assert corpus.recordings[index].execution != query.execution


def test_compatible_mode_gives_identical_keys():
    corpus = _corpus(sites=("left_wrist", "right_wrist"))
    rng = _rng(3)
    for _ in range(200):
        episode = draw_episode(corpus, rng, support_size=8, mode="compatible")
        query_key = corpus.key_of(corpus.recordings[episode.query])
        for index in episode.support:
            assert are_compatible(query_key, corpus.key_of(corpus.recordings[index]))


def test_near_miss_mode_gives_equivalent_but_not_identical_keys():
    corpus = _corpus(sites=("left_wrist", "right_wrist"))
    rng = _rng(4)
    drawn = 0
    for _ in range(200):
        episode = draw_episode(corpus, rng, support_size=8, mode="near_miss")
        if episode is None:
            continue
        drawn += 1
        query_key = corpus.key_of(corpus.recordings[episode.query])
        for index in episode.support:
            other = corpus.key_of(corpus.recordings[index])
            assert not are_compatible(query_key, other)
            assert is_near_miss(query_key, other)
    assert drawn > 0, "the near-miss corpus produced no episodes at all"


def test_realised_gt_rate_tracks_p():
    """p is a probability, not a quota; the realised rate is what telemetry must report."""
    corpus = _corpus()
    for p in (0.0, 0.5, 1.0):
        _, telemetry = draw_batch(
            corpus, _rng(7), batch_size=400, support_size=8, p_gt_present=p,
        )
        assert abs(telemetry["sampler/realised_gt_rate"] - p) < 0.08


def test_zero_shot_keeps_answer_as_candidate_but_excludes_all_candidate_support():
    corpus = _corpus()
    rng = _rng(8)
    seen = 0
    for _ in range(300):
        episode = draw_episode(corpus, rng, support_size=8, p_gt_present=0.0)
        if not episode.is_zero_shot:
            continue
        seen += 1
        query = corpus.recordings[episode.query]
        assert episode.candidates[episode.gt_slot] == query.label
        assert all(slot == -1 for slot in episode.support_candidate)
        for index in episode.support:
            assert corpus.recordings[index].label not in episode.candidates
    assert seen > 0


def test_few_shot_episode_contains_the_answer_as_an_enrolled_row():
    corpus = _corpus()
    rng = _rng(9)
    for _ in range(200):
        episode = draw_episode(corpus, rng, support_size=8, p_gt_present=1.0)
        if episode.is_zero_shot:
            continue
        query = corpus.recordings[episode.query]
        assert episode.candidates[episode.gt_slot] == query.label
        bound = {
            episode.support_candidate[i]
            for i, index in enumerate(episode.support)
            if corpus.recordings[index].label == query.label
        }
        assert episode.gt_slot in bound


def test_every_few_shot_candidate_has_a_distinct_support_execution():
    corpus = _corpus(windows_per_subject=4)
    rng = _rng(901)
    for _ in range(100):
        episode = draw_episode(
            corpus, rng, support_size=3, p_gt_present=1.0,
            label_subset=(2, 14), same_subject_probability=0.0,
        )
        assert episode is not None
        assert len(episode.candidates) <= 3
        assert set(episode.support_candidate) == set(range(len(episode.candidates)))
        units = {
            (corpus.recordings[index].dataset, corpus.recordings[index].subject,
             corpus.recordings[index].execution)
            for index in episode.support
        }
        assert len(units) == len(episode.support)


def test_support_labels_are_balanced_within_one():
    corpus = _corpus()
    rng = _rng(10)
    for _ in range(100):
        episode = draw_episode(corpus, rng, support_size=8, p_gt_present=1.0)
        assert episode is not None
        counts: dict[int, int] = {}
        for slot in episode.support_candidate:
            counts[slot] = counts.get(slot, 0) + 1
        if len(counts) > 1:
            assert max(counts.values()) - min(counts.values()) <= 1


def test_every_few_shot_support_row_is_bound_to_a_real_candidate():
    corpus = _corpus()
    rng = _rng(11)
    for _ in range(100):
        episode = draw_episode(corpus, rng, support_size=8, p_gt_present=1.0)
        assert episode is not None
        for i, index in enumerate(episode.support):
            slot = episode.support_candidate[i]
            assert 0 <= slot < len(episode.candidates)
            assert corpus.recordings[index].label == episode.candidates[slot]


def test_short_pool_shrinks_rather_than_pads():
    """Padding with foreign rows would silently break the compatibility rule."""
    corpus = _corpus(labels=("walking", "running"), subjects_per_label=2, windows_per_subject=1)
    rng = _rng(12)
    episode = draw_episode(corpus, rng, support_size=64, p_gt_present=1.0,
                           same_subject_probability=0.0)
    assert episode is not None
    assert episode.shrunk
    assert len(episode.support) < 64
    query_key = corpus.key_of(corpus.recordings[episode.query])
    for index in episode.support:
        assert are_compatible(query_key, corpus.key_of(corpus.recordings[index]))


def test_shrink_is_reported_in_telemetry():
    corpus = _corpus(labels=("walking", "running"), subjects_per_label=2, windows_per_subject=1)
    _, telemetry = draw_batch(
        corpus, _rng(13), batch_size=32, support_size=64, p_gt_present=1.0,
        same_subject_probability=0.0,
    )
    assert telemetry["sampler/shrunk_episode_fraction"] > 0.0
    assert telemetry["sampler/mean_support_size"] < 64


def test_candidates_are_verbatim_and_may_repeat_text():
    """No canonicalisation, no dedup — the readout handles near-identical labels by design."""
    corpus = _corpus(labels=("walking", "walking upstairs", "running", "sitting"))
    rng = _rng(14)
    episode = draw_episode(corpus, rng, support_size=8)
    for label in episode.candidates:
        assert label in {"walking", "walking upstairs", "running", "sitting"}


def test_at_least_two_candidates_always():
    """A single candidate is not a decision."""
    corpus = _corpus()
    rng = _rng(15)
    for _ in range(200):
        episode = draw_episode(corpus, rng, support_size=8, label_subset=(2, 2))
        assert len(episode.candidates) >= 2


def test_determinism_under_a_fixed_seed():
    corpus = _corpus()
    first = draw_episode(corpus, _rng(99), support_size=8)
    second = draw_episode(corpus, _rng(99), support_size=8)
    assert first == second


def test_support_rows_are_distinct_physical_executions():
    corpus = _corpus(windows_per_subject=8)
    episodes, telemetry = draw_batch(corpus, _rng(102), batch_size=64, support_size=12)
    for episode in episodes:
        units = {
            (corpus.recordings[index].dataset, corpus.recordings[index].subject,
             corpus.recordings[index].execution)
            for index in episode.support
        }
        assert len(units) == len(episode.support)
    assert telemetry["sampler/duplicate_support_execution_mean"] == 0.0


def test_query_sampling_is_dataset_then_label_balanced():
    small = _corpus(labels=("walking", "running"), windows_per_subject=1,
                    sites=("left_wrist", "right_wrist"))
    # Inflate one dataset/label without changing its probability under the hierarchical draw.
    inflated = list(small.recordings)
    source = [r for r in small.recordings if r.dataset == "ds0" and r.label == "walking"]
    for repeat in range(30):
        for recording in source:
            inflated.append(Recording(**{
                **recording.__dict__, "window_index": len(inflated),
                "execution": f"{recording.execution}-extra-{repeat}",
            }))
    corpus = SupportCorpus(inflated, small.keys, small.stream_names)
    episodes, _ = draw_batch(
        corpus, _rng(103), batch_size=1000, support_size=2, p_gt_present=1.0,
        same_subject_probability=0.0,
    )
    ds0 = sum(corpus.recordings[e.query].dataset == "ds0" for e in episodes) / len(episodes)
    walking = sum(corpus.recordings[e.query].label == "walking" for e in episodes) / len(episodes)
    assert abs(ds0 - 0.5) < 0.08
    assert abs(walking - 0.5) < 0.08


def test_isolated_configuration_returns_none_rather_than_guessing():
    """One subject, one execution: nothing admissible. The caller counts this, we do not invent."""
    corpus = _corpus(labels=("walking",), subjects_per_label=1, windows_per_subject=1)
    assert draw_episode(corpus, _rng(16), support_size=8) is None


def test_draw_batch_raises_when_nothing_is_drawable():
    corpus = _corpus(labels=("walking",), subjects_per_label=1, windows_per_subject=1)
    with pytest.raises(RuntimeError):
        draw_batch(corpus, _rng(17), batch_size=4, support_size=8)


def test_draw_batch_never_returns_a_silently_short_batch(monkeypatch):
    corpus = _corpus()
    calls = 0
    original = draw_episode

    def mostly_unusable(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs) if calls == 1 else None

    monkeypatch.setattr("training.compare.sampling.draw_episode", mostly_unusable)
    with pytest.raises(RuntimeError, match="silently smaller batch"):
        draw_batch(corpus, _rng(18), batch_size=4, support_size=8,
                   max_attempts_per_episode=2)


def test_execution_derivation_matches_the_evaluation_loader():
    """The sampler and eval/data.py must agree on the leakage unit."""
    from training.compare.sampling import _execution_ids

    event_ids = ("ds:sessionA:0", "ds:sessionA:1", "ds:sessionB:0")
    derived = _execution_ids("does_not_exist", event_ids)
    assert list(derived) == ["ds:sessionA", "ds:sessionA", "ds:sessionB"]


# ---------------------------------------------------------------- loss regression
def test_regime_ce_is_finite_with_ragged_candidate_counts():
    import torch

    from training.compare.sampling import Episode
    from training.compare.train import episode_loss

    episodes = [
        Episode(query=0, support=(), support_candidate=(), candidates=("a", "b", "c"),
                gt_slot=2, mode="compatible", requested_support=0, shrunk=False,
                zero_shot=True),
        Episode(query=1, support=(), support_candidate=(), candidates=("a", "b"),
                gt_slot=0, mode="compatible", requested_support=0, shrunk=False),
    ]
    mask = torch.tensor([[True, True, True], [True, True, False]])
    text = {
        "candidate_mask": mask,
        "candidate_text": torch.nn.functional.normalize(torch.randn(2, 3, 8), dim=-1),
    }
    out = episode_loss(torch.randn(2, 3), episodes, text)
    assert torch.isfinite(out["loss"]), out
    assert torch.isfinite(out["few_shot_ce"]) and torch.isfinite(out["zero_shot_ce"])


def test_episode_mix_probability_controls_the_loss_weight():
    """A rare regime must not be silently reweighted to equal a common regime."""
    import torch

    from training.compare.sampling import Episode
    from training.compare.train import episode_loss

    episodes = [
        Episode(query=i, support=(), support_candidate=(), candidates=("a", "b"),
                gt_slot=0, mode="compatible", requested_support=0, shrunk=False,
                zero_shot=(i == 2))
        for i in range(3)
    ]
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    text = {"candidate_mask": torch.ones(3, 2, dtype=torch.bool)}
    out = episode_loss(logits, episodes, text)
    expected = torch.nn.functional.cross_entropy(logits, torch.zeros(3, dtype=torch.long))
    assert torch.allclose(out["loss"], expected)
    assert not torch.allclose(out["loss"], out["few_shot_ce"] + out["zero_shot_ce"])
