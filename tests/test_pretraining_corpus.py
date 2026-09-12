"""The label-free pretraining tree: root resolution, compact grids, and the corpus plan.

These cover the SHARED plumbing that lets a source live outside ``data/datasets`` and be
stored at half the bytes. Per-source fetch/convert behaviour is tested alongside each
source (``test_pretrain_nymeria.py``, ``test_pretrain_ego_exo4d.py``,
``test_pretrain_synthetic_imu.py``, ``test_scale_dataset_converters.py``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from data.pretraining import corpus_plan
from data.scripts.assembly.assemble import Grid
from data.scripts import build_grids
from data.scripts.build_grids import _grid_root, _save, _store_dtype
from data.scripts.curate import corpus_roots
from data.scripts.curate.deployment_policy import StreamSpec
from data.scripts.eda import grid_io

REPO = Path(__file__).resolve().parents[1]


def _declare(root: Path, name: str, **metadata) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(json.dumps({"dataset": name, **metadata}))
    return directory


def _isolated_repo(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Point ``build_grids`` at a throwaway repo tree and return its two corpus roots.

    ``build_grids`` resolves dataset directories under its own module-level ``REPO`` so that a
    test can redirect the whole pipeline at a temporary tree. Exercising that contract here,
    rather than patching the resolver's globals, is what keeps this suite honest about how the
    production code actually finds data.
    """
    labelled = tmp_path / "data" / "datasets"
    pretrain = tmp_path / "data" / "pretraining"
    labelled.mkdir(parents=True)
    pretrain.mkdir(parents=True)
    monkeypatch.setattr(build_grids, "REPO", tmp_path)
    return labelled, pretrain


# ---------------------------------------------------------------------------------------
# Corpus roots
# ---------------------------------------------------------------------------------------

def test_dataset_root_finds_a_source_in_either_tree(tmp_path, monkeypatch):
    labelled, pretrain = tmp_path / "datasets", tmp_path / "pretraining"
    _declare(labelled, "uci_har")
    _declare(pretrain, "nhanes")
    monkeypatch.setattr(corpus_roots, "CORPUS_ROOTS", (labelled, pretrain))
    monkeypatch.setattr(corpus_roots, "PRETRAIN_ROOT", pretrain)

    assert corpus_roots.dataset_root("uci_har") == labelled / "uci_har"
    assert corpus_roots.dataset_root("nhanes") == pretrain / "nhanes"
    assert corpus_roots.is_pretraining("nhanes")
    assert not corpus_roots.is_pretraining("uci_har")
    assert corpus_roots.dataset_names() == ("nhanes", "uci_har")


def test_a_dataset_present_in_both_trees_is_an_error_not_a_precedence_rule(tmp_path, monkeypatch):
    """A half-finished move leaves two copies; silently preferring one trains on a guess."""
    labelled, pretrain = tmp_path / "datasets", tmp_path / "pretraining"
    _declare(labelled, "nhanes")
    _declare(pretrain, "nhanes")
    monkeypatch.setattr(corpus_roots, "CORPUS_ROOTS", (labelled, pretrain))

    with pytest.raises(ValueError, match="more than one corpus root"):
        corpus_roots.dataset_root("nhanes")


def test_missing_dataset_raises_but_can_resolve_a_destination_for_a_new_one(tmp_path, monkeypatch):
    labelled, pretrain = tmp_path / "datasets", tmp_path / "pretraining"
    labelled.mkdir()
    monkeypatch.setattr(corpus_roots, "CORPUS_ROOTS", (labelled, pretrain))
    monkeypatch.setattr(corpus_roots, "LABELLED_ROOT", labelled)

    with pytest.raises(FileNotFoundError, match="no dataset 'ghost'"):
        corpus_roots.dataset_root("ghost")
    assert corpus_roots.dataset_root("ghost", must_exist=False) == labelled / "ghost"


def test_a_bare_directory_without_metadata_is_not_a_dataset(tmp_path, monkeypatch):
    """An interrupted fetch leaves ``downloads/``; it must not join the roster."""
    labelled = tmp_path / "datasets"
    (labelled / "half_fetched" / "downloads").mkdir(parents=True)
    monkeypatch.setattr(corpus_roots, "CORPUS_ROOTS", (labelled,))
    assert corpus_roots.dataset_names() == ()


def test_the_real_repo_has_nhanes_in_the_pretraining_tree_only():
    assert corpus_roots.is_pretraining("nhanes")
    assert corpus_roots.dataset_root("nhanes") == corpus_roots.PRETRAIN_ROOT / "nhanes"
    assert not (corpus_roots.LABELLED_ROOT / "nhanes").exists()


# ---------------------------------------------------------------------------------------
# Compact (float16) grids
# ---------------------------------------------------------------------------------------

def test_store_dtype_defaults_to_float32_and_honours_a_float16_declaration(tmp_path, monkeypatch):
    labelled, _ = _isolated_repo(tmp_path, monkeypatch)
    _declare(labelled, "plain")
    _declare(labelled, "compact", grid_dtype="float16")

    assert _store_dtype("plain") == np.dtype(np.float32)
    assert _store_dtype("compact") == np.dtype(np.float16)


def test_an_unsupported_grid_dtype_is_refused(tmp_path, monkeypatch):
    labelled, _ = _isolated_repo(tmp_path, monkeypatch)
    _declare(labelled, "weird", grid_dtype="int8")
    with pytest.raises(ValueError, match="unsupported grid_dtype"):
        _store_dtype("weird")


def test_the_real_pretraining_sources_all_declare_float16():
    """The 120 GB budget assumes it; a source that forgets silently doubles its footprint."""
    for name in corpus_roots.dataset_names(corpus_roots.PRETRAIN_ROOT):
        assert _store_dtype(name) == np.dtype(np.float16), name


def test_grid_root_prefers_an_explicit_override_and_otherwise_resolves(tmp_path, monkeypatch):
    labelled, _ = _isolated_repo(tmp_path, monkeypatch)
    _declare(labelled, "src")
    assert _grid_root(tmp_path / "scratch", "src") == tmp_path / "scratch" / "src"
    assert _grid_root(None, "src") == labelled / "src"


def _tiny_grid(dataset: str, values: np.ndarray) -> Grid:
    return Grid(
        data=values,
        mask=np.ones(values.shape[2], dtype=bool),
        channels=tuple(f"acc_{axis}" for axis in "xyz")[: values.shape[2]],
        labels=["__unlabeled__"] * len(values),
        alignment="native",
        dataset=dataset,
        rate_hz=80.0,
        event_ids=[f"{dataset}:e:{i}" for i in range(len(values))],
    )


def test_a_float16_grid_round_trips_through_save_and_discovery(tmp_path, monkeypatch):
    _, root = _isolated_repo(tmp_path, monkeypatch)
    _declare(root, "compact", grid_dtype="float16", sampling_rate_hz=80.0)

    rng = np.random.default_rng(0)
    # Realistic magnitudes: acceleration in g, gravity present.
    values = (rng.standard_normal((4, 480, 3)).astype(np.float32) * 0.4 + 1.0)
    spec = StreamSpec("compact", "watch_wrist", "watch", "wrist", {}, {}, "present")
    _save(None, spec, _tiny_grid("compact", values), ["s1"] * 4)

    refs = grid_io.discover_grids("native", datasets_dir=root)
    assert [ref.key for ref in refs] == ["compact/watch_wrist"]
    ref = refs[0]
    assert ref.store_dtype == np.dtype(np.float16)

    stored = np.asarray(ref.load_data(), dtype=np.float32)
    # float16 spacing at ~1 g is 2**-10; the stored signal must match the source to that.
    assert np.abs(stored - values).max() < 2e-3
    meta = json.loads((ref.grid_dir / "meta.json").read_text())
    assert meta["store_dtype"] == "float16"


def test_float16_storage_preserves_the_band_energy_the_frontend_reads(tmp_path, monkeypatch):
    """The saving is only free if it is inaudible to the analysis, so measure that directly."""
    _, root = _isolated_repo(tmp_path, monkeypatch)
    _declare(root, "compact", grid_dtype="float16", sampling_rate_hz=80.0)

    rate, seconds = 80.0, 6.0
    t = np.arange(int(rate * seconds)) / rate
    # 1 g of gravity plus a 2 Hz gait-band tone, the regime the filterbank actually reads.
    signal = 1.0 + 0.3 * np.sin(2 * np.pi * 2.0 * t)
    values = np.stack([np.stack([signal] * 3, axis=-1)]).astype(np.float32)

    spec = StreamSpec("compact", "watch_wrist", "watch", "wrist", {}, {}, "present")
    _save(None, spec, _tiny_grid("compact", values), ["s1"])
    stored = np.asarray(
        np.load(root / "compact" / "grids" / "native" / "watch_wrist" / "data.npy"),
        dtype=np.float32,
    )

    reference = np.abs(np.fft.rfft(values[0, :, 0] - values[0, :, 0].mean()))
    replayed = np.abs(np.fft.rfft(stored[0, :, 0] - stored[0, :, 0].mean()))
    peak = int(np.argmax(reference))
    assert peak == int(np.argmax(replayed))
    assert abs(replayed[peak] - reference[peak]) / reference[peak] < 1e-3


def test_streaming_grid_refuses_mixed_channel_masks(tmp_path, monkeypatch):
    """A stream cannot silently inherit the first session's gyro availability."""
    spec = StreamSpec("mixed", "watch_wrist", "watch", "wrist", {}, {}, "present")
    first = _tiny_grid("mixed", np.ones((1, 4, 6), dtype=np.float32))
    second = replace(
        _tiny_grid("mixed", np.ones((1, 4, 6), dtype=np.float32)),
        mask=np.array([True, True, True, False, False, False]),
    )
    grids = iter((first, second))
    monkeypatch.setattr(build_grids, "_session_grid", lambda *args, **kwargs: next(grids))

    def sessions():
        return iter(((None, 80.0, "s1"), (None, 80.0, "s2")))

    with pytest.raises(ValueError, match="inconsistent session grid.*mask"):
        build_grids._write_streaming_grid(
            tmp_path, "mixed", spec, sessions, alignment="native", resample_to=None,
            canonical_labels=True, view="native",
        )


def test_discovery_spans_both_corpus_roots(tmp_path, monkeypatch):
    labelled, pretrain = _isolated_repo(tmp_path, monkeypatch)
    _declare(labelled, "labelled_src", sampling_rate_hz=50.0)
    _declare(pretrain, "compact", grid_dtype="float16", sampling_rate_hz=80.0)

    values = np.ones((2, 480, 3), dtype=np.float32)
    for dataset in ("labelled_src", "compact"):
        spec = StreamSpec(dataset, "watch_wrist", "watch", "wrist", {}, {}, "present")
        _save(None, spec, _tiny_grid(dataset, values), ["s1", "s1"])

    monkeypatch.setattr(grid_io, "grid_search_roots", lambda: (labelled, pretrain))
    keys = {ref.key for ref in grid_io.discover_grids("native")}
    assert keys == {"labelled_src/watch_wrist", "compact/watch_wrist"}


# ---------------------------------------------------------------------------------------
# The corpus plan
# ---------------------------------------------------------------------------------------

def test_the_plan_fits_the_stated_budget():
    total = corpus_plan.total_gigabytes()
    assert total <= corpus_plan.DEFAULT_BUDGET_GB, f"plan is {total:.1f} GB"
    # Headroom is deliberate, but an enormous gap would mean the plan silently shrank.
    assert total > 0.8 * corpus_plan.DEFAULT_BUDGET_GB, f"plan is only {total:.1f} GB"


def test_source_sizing_is_the_documented_arithmetic():
    source = next(s for s in corpus_plan.CORPUS_PLAN if s.dataset == "capture24_pretrain")
    expected = (
        source.streams * source.wall_hours * 3600.0
        * source.rate_hz * source.stored_channels * corpus_plan.BYTES_PER_SAMPLE / 1e9
    )
    assert source.gigabytes == pytest.approx(expected)
    assert source.stream_hours == pytest.approx(source.streams * source.wall_hours)
    assert source.stored_channels == 6


def test_multi_placement_sources_count_one_stream_hour_per_placement():
    xsens = next(s for s in corpus_plan.CORPUS_PLAN if s.dataset == "nymeria_xsens")
    assert xsens.streams == 11
    assert xsens.stream_hours == pytest.approx(11 * xsens.wall_hours)


def test_candidates_are_excluded_from_the_budget():
    planned = {source.dataset for source in corpus_plan.CORPUS_PLAN}
    for candidate in corpus_plan.CANDIDATE_SOURCES:
        assert candidate.candidate
        assert candidate.dataset not in planned


def test_pretraining_sources_are_opt_in_and_never_join_a_default_build():
    """Building these grids must not silently enlarge anyone's training run.

    Two independent gates have to hold. ``deployment_streams`` (what a default grid build
    materialises) must not name them, and they must be absent from the named training
    rosters, which is what ``CorpusIndex`` filters on. Either gate alone would be enough;
    both are asserted because losing one silently changes what every run trains on.
    """
    from data.scripts.curate.deployment_policy import (
        EXPANDED_PHASE_A_TRAIN_DATASETS,
        PRETRAIN_SCALE_DATASETS,
        PRIMARY_EVAL_DATASETS,
        deployment_streams,
    )

    default_build = {spec.dataset for spec in deployment_streams(placement_strict=False)}
    for dataset in PRETRAIN_SCALE_DATASETS:
        assert dataset not in default_build, dataset
        assert dataset not in EXPANDED_PHASE_A_TRAIN_DATASETS, dataset
        # A label-free source can never be an evaluation source: it has no labels to score.
        assert dataset not in PRIMARY_EVAL_DATASETS, dataset


def test_every_planned_source_has_streams_declared_in_the_policy():
    """A planned source with no StreamSpec would fetch and convert, then grid to nothing."""
    from data.scripts.curate.deployment_policy import (
        PRETRAIN_SCALE_DATASETS,
        stream_specs,
    )

    for source in corpus_plan.CORPUS_PLAN:
        assert source.dataset in PRETRAIN_SCALE_DATASETS, source.dataset
        specs = stream_specs(source.dataset, role="phase_a_scale")
        assert specs, f"{source.dataset} has no streams in the deployment policy"
        # The plan's per-hour stream multiplier must match the declared placements, or the
        # disk estimate is fiction.
        assert len(specs) == source.streams, (
            f"{source.dataset}: plan says {source.streams} streams, policy declares {len(specs)}"
        )


def test_synthetic_data_is_never_the_base_of_the_corpus():
    """Mocap-derived signal is a wave-2 diversity source; arXiv 2602.11064 is why."""
    synthetic = next(s for s in corpus_plan.DEFERRED_SOURCES if s.dataset == "synthetic_imu")
    assert synthetic.wave == 2
    assert synthetic.dataset not in {source.dataset for source in corpus_plan.CORPUS_PLAN}


# ---------------------------------------------------------------------------------------
# Pretraining and the supervised head train on disjoint data (decision 2026-09-09)
# ---------------------------------------------------------------------------------------

def test_the_encoder_and_the_head_never_share_a_corpus():
    """The whole point of the split: a downstream gain must be attributable.

    If the encoder pretrains on the same corpora the support classifier is
    trained and selected on, an improvement can always be read as the encoder having already
    met those subjects, devices and activities rather than as a better representation.
    """
    from data.scripts.curate.deployment_policy import (
        LABEL_FREE_PRETRAIN_DATASETS,
        SUPERVISED_HEAD_TRAIN_DATASETS,
    )

    overlap = set(LABEL_FREE_PRETRAIN_DATASETS) & set(SUPERVISED_HEAD_TRAIN_DATASETS)
    assert not overlap, f"encoder and head share corpora: {sorted(overlap)}"


def test_every_label_free_source_is_excluded_from_subject_validation():
    """Unlabelled sources train the encoder; labelled sources never enter its validation split."""
    from data.scripts.curate.deployment_policy import LABEL_FREE_PRETRAIN_DATASETS
    from training.tokenizer.pretrain_data import PHASE_A_ONLY_DATASETS

    assert PHASE_A_ONLY_DATASETS == frozenset(LABEL_FREE_PRETRAIN_DATASETS)


def test_every_label_free_source_really_is_label_free():
    """Membership of the roster is not evidence; where it sits on disk is."""
    from data.scripts.curate.deployment_policy import LABEL_FREE_PRETRAIN_DATASETS

    for dataset in LABEL_FREE_PRETRAIN_DATASETS:
        assert corpus_roots.is_pretraining(dataset), dataset
        assert not (corpus_roots.LABELLED_ROOT / dataset).exists(), dataset


def test_a_labelled_corpus_cannot_enter_label_free_pretraining():
    from data.scripts.curate.deployment_policy import assert_pretraining_is_label_free

    assert_pretraining_is_label_free(["nhanes", "nymeria_xsens"])  # fine
    with pytest.raises(ValueError, match="capture24"):
        assert_pretraining_is_label_free(["nhanes", "capture24"])


def test_an_unbuilt_source_is_not_reported_as_a_leak():
    """Absent is not labelled; saying so sends the reader hunting a leak that isn't there."""
    from data.scripts.curate.deployment_policy import assert_pretraining_is_label_free

    assert_pretraining_is_label_free(["a_source_nobody_has_built_yet"])


def test_the_named_recipes_resolve_and_only_label_free_is_label_free():
    from training.tokenizer.pretrain import _corpus_datasets

    label_free = _corpus_datasets("label_free")
    assert set(label_free) == set(corpus_plan.CORPUS_PLAN[i].dataset
                                  for i in range(len(corpus_plan.CORPUS_PLAN)))
    for historical in ("expanded", "matched"):
        roster = _corpus_datasets(historical)
        assert roster, historical
        # The historical recipes are labelled corpora, kept for reproduction.
        assert not set(roster) & set(label_free), historical


# ---------------------------------------------------------------------------------------
# The orchestrator's guards
# ---------------------------------------------------------------------------------------

def _orchestrate(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "data.pretraining.build_corpus", *args],
        cwd=REPO, capture_output=True, text=True,
    )


def test_plan_flag_reports_without_touching_anything():
    result = _orchestrate("--plan")
    assert result.returncode == 0
    assert "TOTAL" in result.stdout and "capture24_pretrain" in result.stdout


def test_orchestrator_uses_each_source_cli_contract(monkeypatch):
    """A source's convenience default must not silently shrink the planned corpus."""
    from data.pretraining import build_corpus

    commands: list[list[str]] = []
    monkeypatch.setattr(build_corpus, "_module_exists", lambda _: True)
    monkeypatch.setattr(build_corpus, "_run", lambda command: commands.append(command) or 0)
    sources = tuple(source for source in corpus_plan.CORPUS_PLAN if source.dataset != "ego_exo4d")
    assert build_corpus._stage_fetch(sources, approved=True) == 0
    assert build_corpus._stage_convert(sources) == 0
    rendered = [" ".join(command) for command in commands]
    assert any("capture24_pretrain.fetch" in command for command in rendered)
    assert any("nymeria.fetch --sequences 40 --max-gb 120 --workers 8" in command for command in rendered)
    assert any("extrasensory_pretrain.fetch" in command for command in rendered)
    assert any("nymeria.convert --streams xsens" in command for command in rendered)
    assert any("extrasensory_pretrain.convert" in command for command in rendered)


def test_grid_stage_uses_the_jepa_source_window_contract(monkeypatch):
    from data.pretraining import build_corpus

    commands: list[list[str]] = []
    monkeypatch.setattr(build_corpus, "_run", lambda command: commands.append(command) or 0)
    source = next(item for item in corpus_plan.CORPUS_PLAN if item.dataset == "capture24_pretrain")
    assert build_corpus._stage_grids((source,)) == 0
    grid_command = next(command for command in commands if "data.scripts.build_grids" in command)
    index = grid_command.index("--window-seconds")
    assert float(grid_command[index + 1]) == corpus_plan.PRETRAIN_WINDOW_SECONDS


def test_fetching_refuses_without_explicit_approval():
    """Hundreds of gigabytes must never move as a side effect of inspecting the plan."""
    result = _orchestrate("--stage", "fetch", "--datasets", "capture24_pretrain")
    assert result.returncode != 0
    assert "--yes" in result.stdout


def test_a_selection_over_budget_is_refused():
    result = _orchestrate("--stage", "grids", "--budget-gb", "1")
    assert result.returncode != 0
    assert "budget" in (result.stdout + result.stderr)


def test_a_candidate_source_is_not_buildable():
    result = _orchestrate("--stage", "grids", "--datasets", "embody3d")
    assert result.returncode != 0
    assert "unconfirmed" in (result.stdout + result.stderr)


def test_an_unknown_source_is_named_not_silently_skipped():
    result = _orchestrate("--plan", "--datasets", "not_a_source")
    assert result.returncode != 0
    assert "unknown source" in (result.stdout + result.stderr)
