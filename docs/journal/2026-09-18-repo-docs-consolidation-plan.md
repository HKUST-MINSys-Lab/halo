# Repository, documentation and naming consolidation plan (2026-09-18, second pass)

**Status:** plan, nothing executed. Every count and path below was measured on the working tree at
`093d8e6` plus the uncommitted conditioning-v2 and MobiAct work, then re-verified in a second pass
(§7 lists what the second pass changed). The plan is written so that a separate agent can execute
each phase and a reviewer can verify each acceptance check without having read this conversation.

## 0. Where the code actually lives today (measured)

| fact | value |
|---|---|
| working repo | `/home/alex/code/HALO/halo`, `main` at `093d8e6`, in sync with `origin/main` |
| `origin` | `github.com/alxdofficial/halo`, **PUBLIC**, created 2026-07-12, last push 2026-09-18 01:39 UTC, 56 MB |
| lab organisation | `HKUST-MINSys-Lab`; the account is an active member with repository-creation rights; org plan is free |
| lab repository | `HKUST-MINSys-Lab/tsfm` (private, ADMIN permission): the **v1 code base**, description "time series foundational model for IMU sensing", branches `master` and `cleanup/codebase-refactor`, last push 2026-03-09 |
| legacy checkout | `/home/alex/code/HALO/legacy_code` (14 GB), branch `V2` at `16cd318`, remotes `lab` (the repo above) and `origin` = `alxdofficial/tsfm` (personal, private). `V2` is fully pushed to the personal `tsfm`; its 55 commits are on **no** lab branch. The checkout carries 116 uncommitted deletions and 12 modifications, i.e. a half-finished cleanup |
| no lab repository named `halo` exists | none of the 17 org repositories match |
| remote tags on the personal `halo` | 10, all pushed: six `archive-*-2026091x`, three `phaseb-*-20260824`, `results-pre-vocab-fix` |
| remote branches | `main`, `archive/imwut-comparison-pre-cleanup-20260910`, `archive/pre-application-main-20260830`; local-only: `archive/phaseb-vector8-vote-20260824`, worktree branch `codex/phase-b-decoder-diagnostics-20260811` (569 MB checkout, idle since August) |
| sibling directories | `paper/` (own repo `alxdofficial/Alex-TSFM-Mobicom26`, personal), `trace-paper/` (git, no remote), `archive/` (189 MB, not git: anon submission 2026-03, rebuttal v1), `halo_archives/` (28 GB, not git, one snapshot that a tag already covers), `code/docs` (520 KB stray) |
| tracked weight | `.git` 71 MB; 20 MB of the tracked tree is result artifacts under `training/support_classifier/evaluations/` (170 files, 72 PNGs); 1.4 MB under `docs/results/artifacts/`; four `hapt` debug PNGs tracked despite the ignore rule; `references/` tracks 138 citation files, its 388 MB of PDFs are ignored; no credentials found by pattern scan; `.claude/` and `comms/` are untracked |
| **untracked weight inside the code tree** | `training/support_classifier/evaluations/` 48 GB (feature caches: 18 GB, 5.9 GB, 4.2 GB, 1.9 GB ...), `training/support_classifier/outputs/` 2.9 GB (checkpoints), `training/tokenizer/outputs/` 392 MB, `data/datasets/` 171 GB and `data/pretraining/` 106 GB of ignored downloads and grids that live *beside* the converter code |
| code | 58,058 non-test lines in 200 modules; 62 test files; 945 tests pass |
| docs | 64 Markdown files; 22 linked from no other document; 7 cite paths that do not exist; the entry documents describe a retired classifier and retired patch spans |

So the answer to "have we been pushing to the lab repo": no. Everything since 2026-07-12 has gone
to a public personal repository; the lab repository holds the March v1 tree under the old code
name; the v2 legacy branch exists only on a personal repository. Act on the visibility first
(§2, step 1); it does not need the rest of the plan.

## 1. Name

The v1 expansion was "Heterogeneity-Aware Language-aligned Open-set" IMU foundation model. Two of
those words no longer describe the system: the design of record says explicitly that language
alone is not the claim, and "foundation model" is the framing the supervisor meeting rejected. The
contribution as stated on 2026-09-15 is a very small heterogeneity-friendly model with ultra
few-shot adaptation, compatible with variable channels, rates, devices and placements, conditioned
on metadata and text, trained by a curriculum for partial-information deployments.

Decisions taken with Alex on 2026-09-18: H and A are **Heterogeneity-Adaptive**; "low-shot" is
out.

Recommended expansion: **HALO: Heterogeneity-Adaptive, Lightweight, Open-vocabulary activity
recognition** (tagline: "support-conditioned zero- and few-shot recognition from heterogeneous
wearable IMUs in under a million parameters").

- *Heterogeneity-Adaptive*: the channel, rate, device, placement and multi-device axes, the
  acquisition-conditioning contract, and adaptation to a deployment from its support set rather
  than a fixed configuration.
- *Lightweight*: the 0.79 M-parameter encoder plus a 0.13 M classifier against baselines of 4.5 M
  to 1.3 B; this is the word Alex used first ("super lightweight") and the axis the paper's
  parameter table already makes.
- *Open-vocabulary*: the declared-candidate-text interface, the thing that separates the
  classifier from every released baseline; it does not claim open-set rejection, which is why
  the v1 "Open-set" is dropped.
- Few-shot adaptation lives in the tagline and the first README sentence, not the acronym; the
  Adaptive already gestures at it.

Alternatives for L and O if the group prefers: "Label-conditioned, Open-vocabulary" (redundant);
"Lightweight, Open-set" (implies rejection of unknowns, which the system does not do). Whichever is
chosen, it appears in exactly one place, the README's first sentence, and every other document
links there. The v1 expansion stays only in `history.md` as the name's origin.

## 2. GitHub, branches and archives

Target state: one lab-owned repository is the source of truth, the personal repositories are
private mirrors or archived, and every historical era is reachable through one naming scheme
documented in one file.

1. **Immediately, independent of everything else:** make `alxdofficial/halo` private
   (`gh repo edit alxdofficial/halo --visibility private`). It has been public since July with the
   unpublished resubmission code, results and journal. One line, reversible.
2. Create `HKUST-MINSys-Lab/halo` (private). Push `main`, the two `archive/*` branches and all
   ten tags. Set it as `origin`; keep the personal repo as remote `personal` for a transition
   period, then archive it on GitHub (read-only) rather than delete it, because paper drafts and
   memory notes cite its URL.
3. **Legacy repositories.** Push `legacy_code`'s `V2` branch to the lab `tsfm` (it is on no lab
   branch today). Then rename `tsfm` to `halo-v1-v2-legacy`, add a top-level README pointing at
   `HKUST-MINSys-Lab/halo` and set the GitHub "archived" flag; renaming keeps old links working.
   The half-finished deletion in the local checkout (116 deletions, 12 edits) must be either
   committed as "retire benchmark_data" or discarded before the push, not left. After the push is
   verified, the 14 GB local checkout can go; `alxdofficial/tsfm` becomes a private mirror or is
   archived.
4. **One naming scheme for history.** Today there are four: `archive-<topic>-<date>` tags,
   `phaseb-<topic>-<date>` tags, `results-pre-vocab-fix`, and `archive/<topic>-<date>` branches.
   Adopt annotated tags only, named `hist/<era>/<topic>-<yyyymmdd>` with era one of
   `v1-language-aligned`, `v2-evidence-engine`, `v2-applications`, `v3-support-conditioned`.
   The prefix is `hist/`, not `archive/`, so that tags never share a name with the `archive/*`
   branches while both exist (Git reports such refs as ambiguous). Retag the ten existing tags
   under that scheme, keep the old names for one release cycle, then delete them; delete the two
   `archive/*` branches once their tips are tagged (branches invite commits, tags do not); delete
   the local-only `archive/phaseb-vector8-vote-20260824` after confirming it equals its tag.
5. **Worktrees and siblings.** Tag and remove the `codex/phase-b-decoder-diagnostics-20260811`
   worktree. Verify that `halo_archives/imwut-comparison-pre-cleanup-20260910` equals the tag of
   the same name, then delete the 28 GB copy. Move `archive/` (March submission and rebuttal)
   into the paper repository, and move the paper repository itself into the lab organisation
   while at it. Delete `code/docs` after diffing against the tracked docs. Give `trace-paper` a
   remote or fold it into the paper repository.
6. **Branch policy, written into `CONTRIBUTING.md`:** `main` protected; `feat/<topic>`,
   `fix/<topic>`, `exp/<topic>-<yyyymmdd>` for experiment branches that must be tagged or deleted
   within a week of the result being recorded; parallel agents work on branches, never in one
   shared checkout of `main`, which is what produced the mid-flight test failures in each of the
   last three sweeps.
7. **Tracked weight.** Before the lab push: move the 20 MB of tracked evaluation artifacts into
   `results/artifacts/<run>/` under a size policy (JSON and Markdown tracked; PNGs and gzipped
   rows tracked only for promoted runs; everything else ignored); untrack the four `hapt` debug
   PNGs; fix `pyproject.toml`, which lists an `eval*` package that does not exist. No history
   rewrite is needed at 71 MB.

## 3. Code

### 3.1 Inventory by fate (import graph plus the design of record; exact line counts)

| fate | modules | lines |
|---|---|---:|
| **live core** | `model/tokenizer/{encoder, filterbank, sensor_tokens, transformer, baseline_backbone, matched_encoder}`, `model/blocks`, `model/support/residual_classifier`, `training/support_classifier/{train, sampling, corpus, collate, encoding, neighbors, sealed_eval, run_scenarios, scenarios, partial_coverage, run_partial_coverage, merge_sealed_results, curriculum_audit, development_panel}`, `training/tokenizer/{pretrain_data, eval_transfer}`, `data/scripts/{augmentations, build_grids, scan_duplicates, scan_implausible, download_datasets}`, `data/scripts/curate/*` except `sensor_bias`, `data/scripts/labels/{canonical_labels, build_global_label_mapping}`, `data/scripts/eda/grid_io`, `data/scripts/assembly/*`, `baselines/*` | 26,369 |
| **live converters** (16 active datasets) | `data/datasets/{dsads, forth_trace, harmes, hhar, inclusivehar, kuhar, mmfit, mobiact, motionsense, realdisp, realworld, shoaib, usc_had, ut_complex, wisdm, xrf_v2}` | 5,080 |
| **retired JEPA pretraining** | `training/tokenizer/{pretrain (3,877), future_jepa, losses_repr, monitor_training, plot_training, objective_health, grad_check, ablation_subset, eval_quality, diagnostics/frontend/*}` | 7,441 |
| **retired label-free corpus** | `data/pretraining/*` (nymeria, ego_exo4d, nhanes, synthetic_imu, capture24_pretrain, extrasensory_pretrain, build_corpus, corpus_plan) | 6,963 |
| **retired frontends, classifier, conditioning** | `model/tokenizer/{continuous_kernel, multispan_kernel}`, `model/support/token_mixer` (only its four role constants are still used), `data/scripts/curate/sensor_bias`; plus the per-channel fusion classes inside `model/tokenizer/channel_text` and the M1 primitives `model/tokenizer/{primitives, preprocess}` (exported by the package `__init__`, used by nothing live) | 2,511 (+ about 700 inside the three mixed files) |
| **dormant tooling** | `data/scripts/labels/label_augmentation`, `data/scripts/debug/*`, `data/scripts/eda/plot_*`, `data/scripts/eda/{inventory_streams, orientation}`, `data/scripts/storage_audit`, `training/support_classifier/{profile, neighbor_diagnostics, summarize_classifier_isolation, representation_diagnostics}`, `training/support_classifier/evaluations/residual_audit_20260914/probes.py` | 4,468 |
| **unreferenced** | `training/support_classifier/{label_text, labels, random_control}`, `tests/applications/` (only `__pycache__`), `training/diagnostics/` (empty package) | 230 |
| **retired converters** (19 datasets in no roster) | `capture24, duo_gait, extrasensory, hapt, harth, hmog, kneepad, mhealth, monipar, nfi_fared, opportunity, pamap2, phytmo, sp_sw_har, spar, tnda_har, uci_har, unimib_shar, upper_limb_use` and their 25 `StreamSpec` registrations (41 datasets registered, 16 active) | 4,996 |

Retired and dormant code is 22,600 of 58,058 lines, and the *shared* data loader and encoder
builder sit in a package named after a retired training stage. That, not the line count, is what
makes the tree hard to read.

### 3.2 Prerequisites that the first pass missed

These must land before any file moves, because the code pins its own layout.

- **Twenty repository-relative path constants** (`Path(__file__).resolve().parents[2]` in
  `sampling.py`, `corpus.py`, `build_grids.py`, `scan_*`, `download_datasets.py`, `parents[3]`
  in `grid_io.py` and `build_global_label_mapping.py`, `EVALUATION_ROOT = parent/"evaluations"`
  in `run_scenarios.py`, two CLI defaults of `training/support_classifier/evaluations/zero_shot_feature_cache`,
  `OUT_DIR` constants in the JEPA trainer, `PROVENANCE_ROOTS` in `train.py`). Moving a file one
  level deeper silently changes what `parents[k]` means. Prerequisite: one `halo/paths.py` with
  `REPO_ROOT`, `DATASETS_DIR`, `PRETRAINING_DIR`, `RUNS_DIR`, `CACHE_DIR`, `RESULTS_DIR`, each
  overridable by an environment variable, and every constant above replaced by it. This is a
  logic change and gets its own commit and test.
- **Runtime data lives inside the code tree.** 51 GB of caches and checkpoints under
  `training/`, and 277 GB of downloads and grids under `data/datasets/<name>/` and
  `data/pretraining/<name>/` beside the converter code. Moving those directories is a physical
  move of hundreds of gigabytes and breaks every path cited in the result records (for example
  the promoted checkpoint path and SHA in the 2026-09-17 record). Decision: **the data tree stays
  where it is**; `DATASETS_DIR` keeps pointing at `data/datasets/`, and retired converters are
  marked, not moved (§3.3). Checkpoints and caches move once, to top-level ignored `runs/` and
  `cache/`, with `RUNS_DIR`/`CACHE_DIR` pointing there and symlinks left at the old locations for
  one cycle so the recorded paths still resolve.
- **Cache discovery** (`_existing_feature_cache_dirs` walks `EVALUATION_ROOT` recursively) and
  **provenance** (`capture_source_provenance` hashes `PROVENANCE_ROOTS`; resume compares the
  hash) both change meaning under a move. Expected consequences, to be written down rather than
  discovered: every checkpoint written before the move resumes only with
  `--allow-resume-source-drift`, and the feature cache must be re-pointed, not re-encoded.
- **Checkpoints do not pickle module-qualified classes** (verified on the promoted, the
  differentiable-neighbours and a JEPA checkpoint: only built-in, torch and numpy types), so a
  package rename does not break `torch.load`. The 25 `weights_only=False` call sites are fine.
- **Loading retired-frontend checkpoints.** `build_encoder` constructs `continuous` and
  `multispan` frontends and the channel-granularity fusion for old checkpoints. A flat rule "no
  import from `archive/`" would break that. Rule as adopted: live code never imports from
  `archive/` at module level; `build_encoder` may lazily import a retired frontend only when the
  checkpoint config names it and the caller passed an explicit acknowledgment flag (the
  `--allow-retired-jepa-checkpoint` pattern already exists); the CI check forbids eager imports.
- **Import paths change for 200 modules, 62 tests and every command in the docs.** Add a
  one-cycle compatibility shim (`training/__init__.py`, `model/__init__.py`, `data/scripts/...`
  re-exporting from `halo.*` with a deprecation warning) and `[project.scripts]` console entries
  (`halo-train`, `halo-sealed-eval`, `halo-scenarios`, `halo-audit-curriculum`,
  `halo-dev-panel`) so that documents cite stable commands instead of module paths.

### 3.3 Target layout

Moves with `git mv` in commits that contain no logic change *after* the prerequisites above.

```text
halo/
  halo/                      # the package
    paths.py                 # the only place that knows where data, runs, caches and results are
    data/                    # loaders and curation the live model needs
      corpus.py              # <- training/tokenizer/pretrain_data.py (CorpusIndex, PretrainDataset, collates)
      grids.py               # <- data/scripts/eda/grid_io.py
      augment.py             # <- data/scripts/augmentations.py
      policy/                # <- data/scripts/curate/{deployment_policy, compatibility, corpus_roots, accel_units, channels, build_recording_maps}
      labels.py              # <- data/scripts/labels/canonical_labels.py
      quality/               # <- scan_duplicates, scan_implausible, build_grids
    model/
      frontend/filterbank.py, polarization (unchanged), sensor_tokens.py, text.py (<- channel_text.TokenTextEncoder only)
      encoder.py, transformer.py, blocks.py, matched_encoder.py, baseline_backbone.py
      classifier/residual.py, neighbors.py, roles.py (<- the four constants from token_mixer)
    train/                   # <- training/support_classifier/{train, sampling, corpus, collate, encoding}
      cli.py, episodes.py, telemetry.py, validate.py   # split of the 2,316-line train.py
    eval/
      streams.py             # <- baselines/data.py (EvalStream, loaders; used by HALO too)
      encode.py              # <- training/tokenizer/eval_transfer.py (build_encoder, encode_*)
      readouts.py            # shared manifest, feature-cache and readout code of sealed_eval and run_scenarios
      sealed.py, scenarios.py, partial_coverage.py, merge.py
    baselines/               # adapters unchanged
    diagnostics/             # curriculum_audit, development_panel, profile, neighbor_diagnostics
  data/datasets/<name>/      # UNCHANGED location: converter code beside its ignored downloads/grids;
                             #   retired datasets get a RETIRED.md and are excluded from packaging and from the live policy
  data/pretraining/          # unchanged location, marked retired
  archive/                   # retired-but-runnable: jepa/ (pretrain.py + friends), frontends/ (continuous, multispan), classifier/token_mixer.py, conditioning/ (channel fusion, sensor_bias), primitives/
  tools/                     # eda plots, debug scripts, storage audit (not imported by the package)
  runs/                      # ignored: checkpoints (<- training/*/outputs)
  cache/                     # ignored: feature caches, zero-shot banks (<- training/support_classifier/evaluations/*cache*)
  results/                   # promoted records + tracked artifacts (see §4)
  docs/
  tests/                     # mirrored: tests/data, tests/model, tests/train, tests/eval, tests/baselines, tests/archive
```

Rules that go with it:

- `archive/` follows the lazy-import rule in §3.2; a CI test built from the import-graph script
  used for this inventory asserts it.
- `StreamSpec` registrations for retired datasets move to `data/datasets/retired_policy.py`,
  loaded only by retired converters; the live policy shrinks to 16 sources.
- The three unreferenced modules are deleted (their history stays in git); `tests/applications`
  and the empty `training/diagnostics` package are deleted.
- `sealed_eval.py` and `run_scenarios.py` share `readouts.py`; the two runners keep their CLIs.
  This is the one refactor with logic risk and gets the reproducibility gate in §5.
- The `--frontend continuous|multispan` choices leave the trainer CLI; the design of record says
  they are not active.

### 3.4 Order of operations for code

1. Freeze: a window in which no other agent edits `training/`, `model/` or `data/scripts/`.
   Land conditioning-v2 and MobiAct first, on branches.
2. Prerequisites (§3.2): `paths.py`, runtime-directory move with symlinks, console scripts,
   compatibility shim. Full suite after each.
3. Delete unreferenced modules; untrack artifacts (§2.7).
4. Move retired code into `archive/` with its tests; add the lazy-import CI test.
5. Move live code into the package layout, one subsystem per commit, suite after each.
6. Split `train.py`; extract `readouts.py`; run the reproducibility gate.
7. Mark retired converters and trim the policy file.

## 4. Documentation

### 4.1 What is wrong now

- The three entry documents contradict the code: `README.md` describes "a semantic token mixer
  with separate zero-support and enrolled-support weights" and patch spans of 0.5, 1.0 and 1.5 s;
  the active classifier is the residual head and the spans are 0.5, 1, 2 and 4 s at 8 s windows.
  `START_HERE.md` step 4 describes the token mixer. `docs/README.md` lists the readiness repair
  plan of 2026-09-13 as live and "pending implementation". `DESIGN_OF_RECORD.md` carries the same
  1.5 s table and describes its own replacement as "not yet implemented".
- `docs/design/` holds 33 files of which four are living contracts and 29 are dated plans,
  sweeps, audits and findings; nothing in the tree says which is which. 22 documents are linked
  from nowhere; 7 cite files that do not exist; 1 relative link is broken.
- Results are stated in three places (`docs/results/RESULTS.md`, the dated
  `docs/results/2026-09-17-*.md` records, Markdown beside artifacts under
  `training/.../evaluations/`), and the sealed table still shows rows the protocol replaced.
- The journal is declared immutable and permission-gated, but the debug sweeps and audits that
  record findings live in `docs/design/`, so the trajectory the journal promises is split.

### 4.2 Target structure

```text
docs/
  README.md                 # the only index; one paragraph per section below
  overview/
    thesis.md               # name, expansion, one-paragraph contribution, what is NOT claimed
    architecture.md         # current encoder + classifier with the parameter table; links to contracts
    history.md              # eras, hist/ tags, what each archive contains
  contracts/                # living; each carries "last verified against code on <date>" and one test that checks a fact from it
    design_of_record.md, acquisition_conditioning.md, evaluation_protocol.md, curriculum.md, baselines.md, data_policy.md
  results/
    RESULTS.md              # promoted numbers only, one table per protocol version, each row naming its artifact
    artifacts/<run>/        # tracked JSON + md for promoted runs only
  journal/                  # KEPT under its existing name and rules; dated immutable entries; the sweeps, audits, plans and
                            #   result records from docs/design and docs/results move INTO it under YYYY-MM-DD-<slug> names
    README.md               # the index table, generated
  archive/                  # superseded design docs kept for citation, each with a banner naming what superseded it
```

Rules:

- A document is a contract (edited in place, dated "last verified"), a journal entry (dated,
  immutable, superseded by a later entry) or an archive (frozen). No fourth kind.
- Moving files into `docs/journal/` touches a folder whose README forbids changes without Alex's
  explicit permission; that permission is a gate on Phase D, not something an agent infers.
- Every number, default or path in a contract is reproducible from the tree; the link and path
  checker from this sweep runs in CI.
- Contracts never describe unimplemented designs as current: the contextual-classifier plan is a
  journal entry until it lands.
- One result table per protocol version. The 2026-09-17 scenario record and its readout caveat
  (baseline 1-NN beside fusion) move into `RESULTS.md` under the v4 heading; Markdown beside
  artifacts becomes a generated appendix, never a place where a result is first stated.

### 4.3 Order of operations for docs

1. `overview/thesis.md` with the name from §1 and the contribution statement; rewrite the README
   against it. This is the single highest-value edit in the plan.
2. With permission, move every dated file from `docs/design/` and `docs/results/` into
   `docs/journal/` under `YYYY-MM-DD-<slug>`, adding `Supersedes:` and `Superseded by:` lines.
3. Rewrite the contracts against the code, one at a time, each with its verifying test (patch
   spans, roster, readout policy, conditioning schema, curriculum shares).
4. Regenerate `RESULTS.md` from artifacts; move the stale sealed table to `archive/` with its
   protocol version.
5. Run the link and path checker; fix the seven dangling paths and the broken link.

## 5. Acceptance checks (all mechanical)

| check | how | pass condition |
|---|---|---|
| no eager import from `archive/` | the import-graph script, run in CI | zero module-level edges from `halo/` into `archive/`; lazy imports only inside `build_encoder` |
| no unreferenced module | same script | every module has a code importer, a test importer or a `__main__` |
| no layout-pinned path | grep for `parents[` and literal `training/`/`data/` path strings outside `paths.py` | zero hits |
| tests | `pytest tests -q` | 945 passing today; the count may fall only by tests deleted with their archived module |
| reproducibility | re-encode one sealed cell per checkpoint era against its cached features; rebuild one sealed and one scenario manifest | bit-identical features (bf16 tolerance) and identical support rows, as verified for the promoted checkpoint in this sweep |
| resume | resume the promoted checkpoint for one step with `--allow-resume-source-drift` | reaches the optimizer step |
| docs | link and path checker | zero dangling paths, zero broken links, zero orphans outside `journal/` |
| docs claims | one test per contract | patch spans, roster, readout policy, schema string, curriculum shares equal the code |
| repository weight | `git ls-files` size by directory | no tracked PNG outside `results/artifacts/`; nothing tracked under `runs/` or `cache/` |
| visibility | `gh repo view` | personal repos private or archived; lab `halo` private; legacy repo archived; `V2` on a lab branch |

## 6. Effort and sequencing

| phase | what | size | can run in parallel with |
|---|---|---|---|
| A | visibility fix, lab repo creation, legacy push, retagging, worktree and sibling cleanup (§2) | half a day | everything |
| B | prerequisites: `paths.py`, runtime-directory move, console scripts, shim (§3.2); then delete unreferenced code, untrack artifacts, move retired code to `archive/` | one and a half days, needs the freeze for `paths.py` | D |
| C | package layout moves, `train.py` split, shared `readouts.py` (§3.4 steps 5-7) | two to three days, needs the freeze | nothing in `training/`, `model/` |
| D | thesis, README, journal move (with permission), contract rewrite, results regeneration (§4.3) | two days | A, B |
| E | acceptance checks wired into CI (§5) | half a day | after B and D |

Three agents at most at any time: one on A then E, one on B then C, one on D. B and C must not
start until the conditioning-v2 and dataset branches are merged, and no experiment should be
launched between the start of B and C's reproducibility gate, because the feature caches, the
checkpoint loader and the path constants are exactly what is being moved.

## 7. What the second pass changed

- Line counts are now exact per category (they were rounded and undercounted the live core by
  about 2,000 lines); `primitives`/`preprocess` moved from "check" to retired, `baseline_backbone`
  and `matched_encoder` confirmed live.
- The "moves only, no logic change" premise was wrong: twenty path constants and 51 GB of runtime
  data inside the code tree make `paths.py` and a runtime-directory move prerequisites (§3.2).
- `data/datasets/` is no longer moved; 277 GB of ignored data lives beside the converters, so
  retired converters are marked in place instead.
- The tag prefix changed from `archive/` to `hist/` to avoid ambiguity with the existing
  `archive/*` branches.
- The archive-import rule now allows the lazy path that loading retired-frontend checkpoints
  needs, with an explicit acknowledgment flag.
- Checkpoint loading under a package rename was verified safe (no module-qualified pickles);
  resume-after-move and cache re-pointing are called out as expected consequences.
- The legacy `V2` branch is on the personal `tsfm`, not the lab's, and the legacy checkout has
  a half-finished deletion; both are now steps in §2.3.
- The journal is kept under its own name and rules, with the permission gate stated, instead of
  being renamed to `records/`.
- Console scripts and a one-cycle compatibility shim were added so the rename does not break
  every cited command.
- The name changed to Heterogeneity-Adaptive, Lightweight, Open-vocabulary per Alex's two
  messages; few-shot moved to the tagline.
