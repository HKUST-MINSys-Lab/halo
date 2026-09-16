"""Tests for the Phase-1 pretraining data pipeline (real grids; skip when absent)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1]
         / "data/datasets/hhar/grids/harmonised").exists(),
    reason="harmonised train grids not built",
)

from training.tokenizer.pretrain_data import (  # noqa: E402
    CHANNELS,
    DFT_SIZE,
    PATCH_SECONDS_CHOICES,
    CorpusIndex,
    MultiResolutionCollate,
    MultiScaleCollate,
    PretrainDataset,
    TemperatureSampler,
    WindowKey,
    stream_channel_descriptions,
    SENSOR_BIAS_DIM,
    SENSOR_BIAS_FIELDS,
    stream_sensor_bias,
)
from data.scripts.labels.canonical_labels import canonicalize  # noqa: E402


@pytest.fixture(scope="module")
def index():
    return CorpusIndex(max_per_stream=150, seed=7)


def test_corpus_index_is_subject_disjoint(index):
    train_subj = {(index.refs[k.stream_i].dataset, index.refs[k.stream_i].subjects[k.window_i])
                  for k in index.train}
    val_subj = {(index.refs[k.stream_i].dataset, index.refs[k.stream_i].subjects[k.window_i])
                for k in index.val}
    assert not (train_subj & val_subj), "train/val share subjects"
    assert index.train and index.val


def test_validation_never_removes_a_label_from_training(index):
    from training.tokenizer.pretrain_data import validation_subjects_for_refs

    val_subjects = validation_subjects_for_refs(index.refs, seed=index.seed)
    train_labels = {
        canonicalize(label)
        for ref in index.refs
        for subject, label in zip(ref.subjects, ref.labels)
        if (ref.dataset, subject) not in val_subjects
    }
    val_labels = {
        canonicalize(label)
        for ref in index.refs
        for subject, label in zip(ref.subjects, ref.labels)
        if (ref.dataset, subject) in val_subjects
    }
    assert val_labels <= train_labels


def test_retired_mmfit_partition_remains_reproducible():
    """Retirement must not silently break reproduction of the historical MM-Fit split."""
    from data.scripts.eda.grid_io import discover_grids
    from training.tokenizer.pretrain_data import validation_subjects_for_refs

    refs = [ref for ref in discover_grids("native") if ref.dataset == "mmfit"]
    if not refs:
        pytest.skip("historical MM-Fit grids are not materialized")
    selected = validation_subjects_for_refs(
        refs,
        seed=7,
        phase_a_only_datasets=frozenset(),
    )
    assert {subject for dataset, subject in selected if dataset == "mmfit"} == {
        "w00", "w05", "w12", "w13", "w20",
    }


def test_corpus_index_uses_canonical_labels(index):
    """Existing grids may predate a synonym decision; the loader boundary must still merge it."""
    assert all(canonicalize(label) == label for label in index.label_ids)
    assert "pushups" not in index.label_ids


def test_checked_in_global_vocabulary_matches_expanded_grids():
    import json
    from data.scripts.labels.build_global_label_mapping import global_label_vocabulary
    from training.tokenizer.pretrain_data import TRAIN_DATASETS
    expected, _ = global_label_vocabulary(list(TRAIN_DATASETS), "native")
    artifact = json.load(open("data/labels/global_labels.json"))
    assert artifact["labels"] == expected
    assert artifact["train_datasets"] == sorted(TRAIN_DATASETS)


def test_corpus_excludes_eval_datasets(index):
    from data.scripts.curate.deployment_policy import SEALED_TEST_EVAL_DATASETS

    datasets = {r.dataset for r in index.refs}
    for banned in SEALED_TEST_EVAL_DATASETS:
        assert banned not in datasets, f"eval dataset {banned} leaked into pretraining"


def test_item_canonical_slots_and_mask(index):
    ds = PretrainDataset(index, index.train[:32], augment=True)
    torch.manual_seed(0)
    import random as stdlib_random

    import numpy as np
    np.random.seed(0)
    stdlib_random.seed(0)
    for i in range(16):
        item = ds[i]
        assert item["data"].shape[1] == 6
        assert item["channel_mask"].shape == (6,)
        assert len(item["texts"]) == 6
        # masked-out slots must be zero-filled
        for c in range(6):
            if not item["channel_mask"][c]:
                assert torch.allclose(item["data"][:, c],
                                      torch.zeros_like(item["data"][:, c]))


def test_channel_dropout_prunes_all_sensor_metadata_atomically(index):
    """Dropping gyro must not leave a stale gyro descriptor, placement row, or bias row."""
    from data.scripts.augmentations import AugmentationConfig, IMUAugmenter

    key = next(key for key in index.train if all(index.refs[key.stream_i].mask))
    ds = PretrainDataset(index, [key], augment=False)
    cfg = AugmentationConfig.none()
    cfg.channel_dropout.enabled = True
    cfg.channel_dropout.p = 1.0
    ds.config_augmenter = IMUAugmenter(cfg)
    item = ds[0]
    assert item["channel_mask"].tolist() == [True, True, True, False, False, False]
    assert len(item["sensor_texts"]) == 1
    assert item["sensor_placement"].shape == (1,)


def test_natively_accel_only_stream_has_no_phantom_gyro_metadata(index):
    key = next(key for key in index.train if not any(index.refs[key.stream_i].mask[3:]))
    item = PretrainDataset(index, [key], augment=False)[0]
    assert len(item["sensor_texts"]) == 1
    assert item["sensor_placement"].shape == (1,)


def test_neutral_acquisition_text_keeps_modality_but_removes_configuration(index):
    key = next(key for key in index.train if all(index.refs[key.stream_i].mask))
    item = PretrainDataset(
        index, [key], augment=False, neutral_acquisition_text=True,
    )[0]

    assert item["sensor_texts"] == ["an accelerometer", "a gyroscope"]
    assert all("wrist" not in text and "pocket" not in text and "gravity" not in text
               for text in item["texts"])
    assert item["sensor_id"].tolist() == [0, 0, 0, 1, 1, 1]


def test_sensor_bias_retains_observation_support_bits():
    bias = stream_sensor_bias("wisdm", "phone_pocket", ["accel", "gyro"])
    support = bias[:, len(SENSOR_BIAS_FIELDS):]
    assert bias.shape == (2, SENSOR_BIAS_DIM)
    assert set(support.unique().tolist()) <= {0.0, 1.0}
    # Gravity fields are measured for accelerometers and unsupported for gyroscopes. WISDM's
    # 20-Hz clock cannot support a >20-Hz noise-floor estimate for either modality.
    assert support[0, :2].tolist() == [1.0, 1.0]
    assert support[1, :2].tolist() == [0.0, 0.0]
    assert support[:, 2].tolist() == [0.0, 0.0]


def test_two_views_share_every_acquisition_configuration(index):
    ds = PretrainDataset(index, index.train[:96], augment=True, two_view=True)
    for i in range(96):
        item = ds[i]
        other = item["view_b"]
        assert item["rate"] == other["rate"]
        assert item["source_rate"] == other["source_rate"]
        assert torch.equal(item["channel_mask"], other["channel_mask"])
        assert torch.equal(item["sensor_placement"], other["sensor_placement"])
        assert item["sensor_target_texts"] == other["sensor_target_texts"]


def test_item_with_rng_replays_acquisition_perturbations(index):
    from data.scripts.augmentations import AugmentationConfig

    key = next(key for key in index.train if all(index.refs[key.stream_i].mask))
    config = AugmentationConfig.phase_a(rate_p=1.0, channel_dropout_p=1.0)
    dataset = PretrainDataset(index, [key], augment=True, augmentation_config=config)
    first = dataset.item_with_rng(0, np.random.default_rng(91))
    second = dataset.item_with_rng(0, np.random.default_rng(91))
    assert torch.equal(first["data"], second["data"])
    assert first["rate"] == second["rate"]
    assert first["augmentations"] == second["augmentations"]
    assert first["sensor_texts"] == second["sensor_texts"]


def test_augmentation_replay_does_not_consume_structural_rng_without_peers(index):
    key = index.train[0]
    dataset = PretrainDataset(index, [key], augment=False, multi_device_probability=0.0)
    actual = np.random.default_rng(314)
    expected = np.random.default_rng(314)
    dataset.item_with_rng(0, actual)
    assert actual.integers(0, 2**31) == expected.integers(0, 2**31)


def test_rotation_pairing_is_explicit(index):
    from data.scripts.augmentations import AugmentationConfig

    cfg = AugmentationConfig.phase_a(rotation_p=1.0)
    shared = PretrainDataset(
        index, index.train[:8], augment=True, two_view=True,
        augmentation_config=cfg, rotation_pairing="shared",
    )
    independent = PretrainDataset(
        index, index.train[:8], augment=True, two_view=True,
        augmentation_config=cfg, rotation_pairing="independent",
    )
    for i in range(8):
        item = shared[i]
        assert torch.equal(item["data"], item["view_b"]["data"])
    differs = False
    for i in range(8):
        item = independent[i]
        differs |= not torch.allclose(item["data"], item["view_b"]["data"])
    assert differs


def test_descriptor_target_is_stable_across_random_surface_paraphrases(index):
    """Equivalent wording variants must not become different descriptor-retrieval classes."""
    from data.scripts.augmentations import AugmentationConfig, IMUAugmenter

    key = next(key for key in index.train if all(index.refs[key.stream_i].mask))
    ds = PretrainDataset(index, [key], augment=False, two_view=True)
    cfg = AugmentationConfig.none()
    cfg.channel_text_phrase.enabled = True
    cfg.channel_text_phrase.p = 1.0
    ds.nuisance_augmenter = IMUAugmenter(cfg)

    visible_forms = set()
    target_forms = set()
    for _ in range(24):
        item = ds[0]
        visible_forms.add(tuple(item["sensor_texts"]))
        visible_forms.add(tuple(item["view_b"]["sensor_texts"]))
        target_forms.add(tuple(item["sensor_target_texts"]))
        target_forms.add(tuple(item["view_b"]["sensor_target_texts"]))
    assert len(visible_forms) > 1, "test did not exercise sensor-text paraphrase variation"
    assert len(target_forms) == 1, "surface paraphrases fragmented the semantic target"


@pytest.mark.parametrize("ps", PATCH_SECONDS_CHOICES)
def test_collate_shapes_and_positions(index, ps):
    ds = PretrainDataset(index, index.train[:8], augment=False)
    collate = MultiScaleCollate(fixed_patch_seconds=ps)
    out = collate([ds[i] for i in range(8)])
    P = out["patches"].shape[1]
    assert P == max(1, round(6.0 / ps))
    assert out["patches"].shape == (8, P, DFT_SIZE, 6)
    assert out["positions"].shape == (8, P)
    assert out["patch_starts"].shape == (8, P)
    assert out["patch_ends"].shape == (8, P)
    assert out["resolution_ids"].shape == (8, P)
    assert out["resolution_count"] == 1
    # Positions are centers of the *realized* sample boundaries.  A final or source-native
    # partial window need not begin with an exact ``ps``-second patch, so asserting ``ps / 2``
    # here would reject the intended physical-time partitioning for otherwise valid sources.
    assert torch.allclose(
        out["positions"], 0.5 * (out["patch_starts"] + out["patch_ends"]),
    )
    assert (out["resolution_ids"][out["patch_padding_mask"]] == 0).all()
    assert (out["resolution_ids"][~out["patch_padding_mask"]] == -1).all()
    # Every REAL patch carries at least one sample. Padding patches are legitimately empty
    # (see test_patch_padding_mask_flags_phantom_patches); asserting >= 1 on them only ever
    # passed while the first eight training rows happened to be full-length windows, and the
    # 2026-09-10 roster change surfaced a short recording in that slice.
    real = out["patch_padding_mask"]
    assert (out["patch_len"][real] >= 1).all()
    assert (out["patch_len"][~real] == 0).all()


def test_collate_handles_per_sample_rates(index):
    """Rate augmentation gives every sample its own rate; the collate must produce
    per-sample patch lengths, not assume a shared one."""
    ds = PretrainDataset(index, index.train[:24], augment=True)
    import random as stdlib_random

    import numpy as np
    np.random.seed(3)
    stdlib_random.seed(3)
    torch.manual_seed(3)
    items = [ds[i] for i in range(24)]
    rates = {round(it["rate"], 1) for it in items}
    out = MultiScaleCollate(fixed_patch_seconds=1.0)(items)
    if len(rates) > 1:                      # rate aug fired at least once
        assert len(set(out["patch_len"][:, 0].tolist())) > 1, \
            "per-sample rates must yield per-sample patch lengths"
    real = out["patch_padding_mask"]
    expected = out["rates"].unsqueeze(1).expand_as(out["patch_len"])
    # Every full patch follows rate*duration; only an honest final partial patch may be shorter.
    assert (out["patch_len"][real].float() <= expected[real] + 1.0).all()


def test_noninteger_rate_uses_physical_boundaries_without_a_one_sample_remainder():
    item = {
        "data": torch.randn(307, 6), "rate": 51.2, "texts": ["x"] * 6,
        "label_id": 0, "channel_mask": torch.ones(6, dtype=torch.bool),
        "gravity_state": "present", "source": "synthetic",
    }
    out = MultiScaleCollate(fixed_patch_seconds=1.0)([item])
    real = out["patch_padding_mask"][0]
    lengths = out["patch_len"][0, real].tolist()
    assert len(lengths) == 6
    assert sum(lengths) == 307
    assert min(lengths) == 51 and max(lengths) == 52
    assert torch.allclose(
        out["positions"][0, real],
        torch.tensor([0.4980469, 1.4941406, 2.5, 3.5058594, 4.5019531, 5.4980469]),
        atol=1e-5,
    )


def test_multiresolution_collate_covers_signal_and_retains_partial_tails():
    item = {
        "data": torch.randn(300, 6), "rate": 50.0, "texts": ["x"] * 6,
        "label_id": 0, "channel_mask": torch.ones(6, dtype=torch.bool),
        "gravity_state": "present", "source": "synthetic",
    }
    out = MultiResolutionCollate(fixed_patch_seconds=(0.4, 1.4))([item])
    real = out["patch_padding_mask"][0]
    assert out["patch_len"].shape == out["positions"].shape
    assert set(out["resolution_ids"][0, real].tolist()) == {0, 1}
    assert torch.all(out["positions"][0, real][1:] >= out["positions"][0, real][:-1])
    for rid in (0, 1):
        m = real & out["resolution_ids"][0].eq(rid)
        assert out["patch_starts"][0, m].min() == 0
        assert abs(float(out["patch_ends"][0, m].max()) - 6.0) < 1e-6
    # 1.4 seconds at 50 Hz leaves an honest 0.4-second final patch.
    long = real & out["resolution_ids"][0].eq(1)
    assert out["patch_len"][0, long].tolist()[-1] == 20
    assert torch.allclose(
        out["patch_durations"][0, real],
        out["patch_len"][0, real].float() / out["rates"][0],
    )


def test_multiresolution_collate_supports_three_explicit_scales():
    item = {
        "data": torch.randn(300, 6), "rate": 50.0, "texts": ["x"] * 6,
        "label_id": 0, "channel_mask": torch.ones(6, dtype=torch.bool),
        "gravity_state": "present", "source": "synthetic",
    }
    out = MultiResolutionCollate(fixed_patch_seconds=(0.5, 1.0, 1.5))([item])
    real = out["patch_padding_mask"][0]
    assert out["resolution_count"] == 3
    assert out["patch_seconds"] == (0.5, 1.0, 1.5)
    assert set(out["resolution_ids"][0, real].tolist()) == {0, 1, 2}
    for rid in range(3):
        selected = real & out["resolution_ids"][0].eq(rid)
        assert out["patch_starts"][0, selected].min() == 0
        assert out["patch_ends"][0, selected].max() == 6.0


@pytest.mark.parametrize("rate", [200.0, 240.0])
def test_default_long_patch_fits_high_rate_pretraining_sources(rate):
    item = {
        "data": torch.randn(round(6 * rate), 6), "rate": rate, "texts": ["x"] * 6,
        "label_id": 0, "channel_mask": torch.ones(6, dtype=torch.bool),
        "gravity_state": "present", "source": "synthetic",
    }
    out = MultiResolutionCollate(fixed_patch_seconds=(0.5, 1.0, 1.5))([item])
    long = out["patch_padding_mask"][0] & out["resolution_ids"][0].eq(2)
    assert out["patch_len"][0, long].max() == round(1.5 * rate)
    assert out["patches"].shape[2] == DFT_SIZE


def test_collate_carries_realized_augmentation_trace_for_both_views(index):
    ds = PretrainDataset(index, index.train[:4], augment=True, two_view=True)
    items = [ds[i] for i in range(4)]
    out = MultiResolutionCollate(
        fixed_patch_seconds=(0.4, 1.4), two_view=True,
    )(items)
    assert out["augmentations"] == [item["augmentations"] for item in items]
    assert out["augmentations_b"] == [item["view_b"]["augmentations"] for item in items]


def test_stream_text_parses_placement():
    texts = stream_channel_descriptions("hhar", "phone_waist")
    assert len(texts) == 6
    assert "waist" in texts[0] and "phone" in texts[0]
    assert "gyroscope" in texts[3]


def test_stream_text_uses_rich_distinct_placement():
    """Channel text must use the StreamSpec placement, not collapse distinct configs (review #2)."""
    def t(ds, st):
        return stream_channel_descriptions(ds, st)[0]
    lw, rw = t("xrf_v2", "left_wrist"), t("xrf_v2", "right_wrist")
    assert "left wrist" in lw and "right wrist" in rw and lw != rw     # L/R not collapsed
    assert "head" in t("xrf_v2", "glasses")                            # glasses = head, not "body"
    assert "ear" in t("xrf_v2", "airpods_ear")                         # ear placement preserved
    assert t("xrf_v2", "left_pocket") != t("xrf_v2", "right_pocket")   # L/R pockets distinct
    assert "forearm" in t("nfi_fared", "wrist")                        # NFI = forearm (per paper)
    assert "lower back" in t("nfi_fared", "back")


def test_temperature_sampler_draws_unique_deterministic_batches():
    keys = [WindowKey(i % 2, i, 0) for i in range(100)]
    sampler_a = TemperatureSampler(
        keys, ["a", "b"], num_samples=40, batch_size=40,
        seed=3,
    )
    sampler_b = TemperatureSampler(
        keys, ["a", "b"], num_samples=40, batch_size=40,
        seed=3,
    )
    batch_a, batch_b = list(sampler_a), list(sampler_b)
    assert batch_a == batch_b
    assert len(batch_a) == len(set(batch_a)) == 40


def test_temperature_sampler_caps_datasets_and_tempers_subjects():
    # Five datasets make a 25% ceiling feasible. Dataset 0 is 10x larger than every other source;
    # within it, subject 0 has 90 windows and subject 1 has 10.
    sizes = [100, 10, 10, 10, 10]
    keys, datasets, subjects = [], [], []
    for stream, size in enumerate(sizes):
        for window in range(size):
            keys.append(WindowKey(stream, window, 0))
            datasets.append(f"d{stream}")
            subjects.append(int(window >= 90) if stream == 0 else 0)
    sampler = TemperatureSampler(
        keys, [f"d{i}" for i in range(5)], num_samples=100, batch_size=10,
        alpha=0.25, subject_ids=subjects, subject_alpha=0.5,
        max_dataset_share=0.25,
    )
    assert sampler.dataset_probabilities["d0"] == pytest.approx(0.25)
    assert max(sampler.dataset_probabilities.values()) <= 0.25 + 1e-12
    assert sum(sampler.dataset_probabilities.values()) == pytest.approx(1.0)

    d0_rows = torch.tensor([i for i, dataset in enumerate(datasets) if dataset == "d0"])
    d0_weight = sampler.weights[d0_rows].sum()
    subject0 = sampler.weights[d0_rows[:90]].sum() / d0_weight
    expected = 90**0.5 / (90**0.5 + 10**0.5)
    assert float(subject0) == pytest.approx(expected)


def test_temperature_sampler_can_batch_by_sensor_count_without_changing_marginals():
    keys = [WindowKey(0, i, 0) for i in range(400)]
    groups = [1] * 160 + [2] * 240
    sampler = TemperatureSampler(
        keys, ["source"], num_samples=100 * 20, batch_size=20,
        alpha=0.0, seed=9, batch_group_ids=groups,
    )
    draws = list(sampler)
    for start in range(0, len(draws), 20):
        batch_groups = {groups[index] for index in draws[start:start + 20]}
        assert len(batch_groups) == 1
        assert len(set(draws[start:start + 20])) == 20
    observed_group_one = sum(groups[index] == 1 for index in draws) / len(draws)
    assert observed_group_one == pytest.approx(0.4, abs=0.08)


def test_no_hapt_uci_leak(index):
    """hapt (UCI-HAR re-release) must be dropped from the corpus (sweep finding E).

    uci_har itself was retired on 2026-09-10 (window-overlap leakage), so it no longer serves as
    the "still present" anchor this test once used; the guard is now that neither half of the
    duplicate, nor any other retired source, reaches the corpus by default.
    """
    from data.scripts.curate.deployment_policy import RETIRED_TRAIN_DATASETS
    datasets = {r.dataset for r in index.refs}
    assert "hapt" not in datasets
    assert datasets.isdisjoint(RETIRED_TRAIN_DATASETS), sorted(datasets & set(RETIRED_TRAIN_DATASETS))
    # The active supervised roster is intentionally curated over time.  Keep this guard tied to
    # the policy authority rather than an obsolete literal count.
    from training.tokenizer.pretrain_data import TRAIN_DATASETS
    assert datasets == set(TRAIN_DATASETS)


def test_patch_padding_mask_flags_phantom_patches(index):
    """Every filled patch is flagged real; any trailing unfilled patch is flagged pad
    AND is exactly zero (sweep findings B/9/17)."""
    ds = PretrainDataset(index, index.train[:16], augment=True)
    out = MultiScaleCollate(fixed_patch_seconds=2.0)([ds[i] for i in range(16)])
    pad = out["patch_padding_mask"]
    patches = out["patches"]
    assert pad.shape == (16, patches.shape[1])
    assert pad[:, 0].all()                              # first patch always real
    for b in range(16):
        for p in range(patches.shape[1]):
            if not pad[b, p]:
                assert torch.count_nonzero(patches[b, p]) == 0, "phantom patch not zero"


def test_short_window_yields_at_least_one_patch(index):
    """A window shorter than one patch at the drawn scale (e.g. sp_sw_har's 1.0 s TUG
    windows at ps=1.5) must still get exactly one REAL short patch spanning the whole
    window — never an all-padding window (which would pool to a degenerate embedding).
    patch_len is honest (< round(rate*ps))."""
    ds = PretrainDataset(index, index.train[:32], augment=True)
    items = [ds[i] for i in range(32)]
    for ps in PATCH_SECONDS_CHOICES:
        out = MultiScaleCollate(fixed_patch_seconds=ps)(items)
        real_per_win = out["patch_padding_mask"].sum(1)
        assert (real_per_win >= 1).all(), f"all-padding window at ps={ps}"
        assert torch.isfinite(out["patches"]).all()
        # every real patch's declared length fits inside the window it came from
        real = out["patch_padding_mask"]
        assert (out["patch_len"][real] >= 1).all()


def test_window_crop_varies_observation_length():
    """P5 window-crop keeps a random contiguous sub-window (session-length invariance), floored at
    min_samples, and never lengthens."""
    from data.scripts.augmentations import AugmentationConfig, IMUAugmenter, IMUSample
    import numpy as np
    cfg = AugmentationConfig.none()
    cfg.window_crop.enabled = True
    cfg.window_crop.p = 1.0
    aug = IMUAugmenter(cfg)
    np.random.seed(0)
    lens = []
    for _ in range(128):
        s = IMUSample(data=torch.zeros(360, 6), channel_names=list(CHANNELS),
                      sampling_rate=60.0, channel_descriptions=["x"] * 6)
        lens.append(aug(s).data.shape[0])
    lens = np.asarray(lens)
    assert lens.max() <= 360 and lens.min() >= int(0.5 * 360)     # in [min_frac*T, T], never longer
    assert len(set(lens.tolist())) > 10                          # genuinely variable
    # floor: a window already near the min_samples floor is not cropped below it
    short = IMUSample(data=torch.zeros(40, 6), channel_names=list(CHANNELS), sampling_rate=50.0,
                      channel_descriptions=["x"] * 6)
    assert aug(short).data.shape[0] >= 32


def test_phase_a_reference_recipe_is_clean_and_rotation_is_an_explicit_shared_ablation():
    from data.scripts.augmentations import AugmentationConfig
    cfg = AugmentationConfig.phase_a()
    enabled = {name for name in cfg.ORDER if getattr(cfg, name).enabled}
    assert enabled == set()
    cfg = AugmentationConfig.phase_a(rotation_p=1.0)
    nuisance, config = cfg.split_by_group()
    assert not nuisance.rotation_3d.enabled
    assert config.rotation_3d.enabled
    assert config.rotation_3d.p == 1.0


def test_gravity_align_primitive_respects_removed_state():
    """The EDA alignment primitive respects authoritative gravity-removed metadata."""
    from model.tokenizer.preprocess import gravity_align
    w = torch.zeros(1, 120, 6)
    w[:, :, 0] = 1.0                                   # 1 g DC on x — the heuristic WOULD rotate it
    _, _, aligned_auto = gravity_align(w.clone(), list(CHANNELS), 50.0)
    _, r_rem, aligned_rem = gravity_align(w.clone(), list(CHANNELS), 50.0, gravity_state="removed")
    assert bool(aligned_auto[0])                       # heuristic aligns a strong-DC window
    assert not bool(aligned_rem[0])                    # authoritative 'removed' skips it
    assert torch.allclose(r_rem[0], torch.eye(3))      # ...and returns identity (no rotation)


def test_collate_fallback_position_is_window_center():
    """F4a: a window shorter than one patch emits ONE short patch whose position is the window's
    TRUE center (0.5*T/rate), not the nominal ps/2."""
    item = {"data": torch.randn(100, 6), "rate": 100.0, "texts": ["x"] * 6, "label_id": 0,
            "channel_mask": torch.ones(6, dtype=torch.bool), "gravity_state": "present"}
    out = MultiScaleCollate(fixed_patch_seconds=1.5)([item])   # 100 samples @100Hz = 1.0 s < 1.5 s patch
    assert int(out["patch_len"][0, 0]) == 100                  # whole window in one short patch
    assert abs(float(out["positions"][0, 0]) - 0.5) < 1e-4     # 0.5 s (not the nominal 0.75)


def test_wisdm_native_grid_is_full_six_channel(index):
    """F2: wisdm native grids carry REAL gyro (merged), not accel-only [1,1,1,0,0,0]."""
    import numpy as np
    from data.scripts.eda.grid_io import discover_grids
    wisdm = [r for r in discover_grids("native") if r.dataset == "wisdm"]
    if not wisdm:
        pytest.skip("wisdm native grids not built")
    for r in wisdm:
        assert all(r.mask), f"{r.key} mask has padded channels {r.mask}"
        data = r.load_data()
        assert float(np.abs(np.asarray(data[:200, :, 3:])).mean()) > 0.0, "gyro is all-zero"


def test_knn_scores_unsupported_query_labels_as_failures():
    """F1: knn_balanced_acc scores every QUERY label; a class absent from the support scores 0
    instead of being intersected away (which inflated the metric)."""
    from training.tokenizer.pretrain import knn_balanced_acc
    train_z = torch.tensor([[0., 0.], [0.01, 0.], [0., 0.01]])
    train_y = torch.tensor([0, 0, 0])                          # support has only class 0
    test_z = torch.tensor([[0., 0.], [0.02, 0.], [5., 5.], [6., 6.]])
    test_y = torch.tensor([0, 0, 1, 1])                        # query has classes 0 and 1
    assert abs(knn_balanced_acc(train_z, train_y, test_z, test_y, k=3) - 0.5) < 1e-9


def test_collate_default_does_not_rotate_gravity():
    """Collation preserves gravity direction for the tokenizer's signed-DC feature."""
    w = torch.zeros(120, 6)
    w[:, 0] = 1.0                                            # 1 g DC on x (gravity present)
    item = {"data": w, "rate": 50.0, "texts": ["x"] * 6, "label_id": 0,
            "channel_mask": torch.ones(6, dtype=torch.bool), "gravity_state": "present"}
    d = MultiScaleCollate(fixed_patch_seconds=1.0)([item])
    dc_d = d["patches"][0, 0, :int(d["patch_len"][0, 0]), :3].mean(0)
    assert dc_d[0].abs() > 0.9 and dc_d[2].abs() < 0.1, dc_d


def test_source_rate_is_bounded_by_the_hardware_clock_after_rate_augmentation():
    """Observability is min(hardware, augmented). Resampling never widens measured bandwidth.

    Regression for the pre-2026-07-26 formula ``src0 * augmented / stored``, which advertised
    94 Hz of bandwidth for wisdm windows upsampled from 20 Hz (32 observable bands where only 26
    carry signal) and hid 5 real bands when xrf_v2's 25 Hz AirPods stream was downsampled.
    """
    from training.tokenizer.pretrain_data import STREAM_SOURCE_RATE_HZ

    def bound(key, stored_hz, augmented_hz):
        hw = STREAM_SOURCE_RATE_HZ.get(key, stored_hz)
        return min(float(hw), float(augmented_hz))

    # No converter resample: the stored rate IS the hardware rate.
    assert bound("wisdm/phone_pocket", 20.0, 94.15) == 20.0     # upsample cannot add bandwidth
    assert bound("wisdm/phone_pocket", 20.0, 12.0) == 12.0      # downsample genuinely narrows it
    assert bound("capture24/watch_wrist", 100.0, 41.0) == 41.0

    # Converter stored these ABOVE their capture clock; the clock still bounds them.
    assert bound("xrf_v2/airpods_ear", 50.0, 81.43) == 25.0
    assert bound("xrf_v2/airpods_ear", 50.0, 21.88) == 21.88    # not 25 * 21.88/50 = 10.94
    assert bound("extrasensory/watch_wrist", 50.0, 90.0) == 25.0


def test_token_budget_caps_the_per_batch_patch_seconds_draw():
    """Peak VRAM tracks batch x patches, and patch_seconds is drawn PER BATCH — so an
    unlucky short draw could OOM a batch size that survived the previous sixty steps."""
    import numpy as np
    import torch
    from training.tokenizer.pretrain_data import MultiResolutionCollate

    # 6 s windows at 100 Hz: the 0.4 s short grid gives 15 patches, the 1.5 s long grid 4.
    batch = [{"data": torch.zeros(600, 6), "rate": 100.0, "label_id": 0} for _ in range(32)]

    uncapped = MultiResolutionCollate(max_batch_tokens=0)
    tight = MultiResolutionCollate(max_batch_tokens=32 * 12)      # only coarse pairs fit

    def tokens(collate):
        short, long = collate._patch_seconds(batch)
        return len(batch) * sum(int(np.ceil(6.0 / p)) for p in (short, long))

    assert tokens(tight) <= 32 * 12
    # The cap must actually bind: the unconstrained pool contains draws that exceed it.
    assert any(
        len(batch) * sum(int(np.ceil(6.0 / p)) for p in pair) > 32 * 12
        for pair in uncapped._valid_pairs
    )
    # Every admissible draw stays inside the budget, not just the one we happened to get.
    for _ in range(25):
        assert tokens(tight) <= 32 * 12


def test_token_budget_falls_back_to_the_coarsest_pair_when_nothing_fits():
    """An impossible budget must degrade to the fewest-token pair, never raise."""
    import torch
    from training.tokenizer.pretrain_data import MultiResolutionCollate

    batch = [{"data": torch.zeros(600, 6), "rate": 100.0, "label_id": 0} for _ in range(8)]
    collate = MultiResolutionCollate(max_batch_tokens=1)
    short, long = collate._patch_seconds(batch)
    assert (short, long) == max(collate._valid_pairs, key=lambda p: p[0] + p[1])


def test_patch_duration_draw_is_independent_of_activity_labels():
    """Phase A is label-free, including the stochastic patch-resolution schedule."""
    import torch
    from training.tokenizer.pretrain_data import MultiResolutionCollate, MultiScaleCollate

    base = [
        {
            "data": torch.zeros(300, 6), "rate": 50.0, "label_id": 0,
            "source": "example", "stream": "watch_wrist", "window_index": index,
        }
        for index in range(16)
    ]
    relabeled = [{**item, "label_id": 91 - i} for i, item in enumerate(base)]

    assert MultiScaleCollate(seed=5)._patch_seconds(base) == \
        MultiScaleCollate(seed=5)._patch_seconds(relabeled)
    assert MultiResolutionCollate(seed=5, max_batch_tokens=0)._patch_seconds(base) == \
        MultiResolutionCollate(seed=5, max_batch_tokens=0)._patch_seconds(relabeled)


def test_four_second_resolution_uses_bounded_physical_analysis_rate():
    """Long filterbank patches stay below DFT capacity without changing physical duration."""
    from training.tokenizer.pretrain_data import (
        DFT_SIZE, FILTERBANK_LONG_PATCH_ANALYSIS_HZ, MultiResolutionCollate,
    )

    item = {
        "data": np.zeros((int(6 * 240), 6), dtype=np.float32),
        "rate": 240.0,
        "source_rate": 240.0,
        "texts": ["x"] * 6,
        "label_id": 0,
        "channel_mask": torch.ones(6, dtype=torch.bool),
    }
    output = MultiResolutionCollate(fixed_patch_seconds=(0.5, 1.0, 2.0, 4.0))([item])
    assert output["rates"].tolist() == [FILTERBANK_LONG_PATCH_ANALYSIS_HZ]
    assert output["source_rates"].tolist() == [240.0]
    assert int(output["patch_len"].max()) <= DFT_SIZE
    four_second = output["resolution_ids"].eq(3) & output["patch_padding_mask"]
    assert torch.allclose(output["patch_durations"][four_second], torch.tensor([4.0, 2.0]))
    assert torch.isclose(output["patch_durations"][four_second].sum(), torch.tensor(6.0))
