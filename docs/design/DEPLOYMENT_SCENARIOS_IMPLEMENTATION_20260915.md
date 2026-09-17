# Deployment-heterogeneity scenarios — implementation handoff (2026-09-15)

> **Historical v3 handoff:** the active v4 protocol retired `s8_cold_start` on 2026-09-17.
> References below to eight scenarios describe the archived implementation, not the current roster.

**Status:** repaired and performance-audited after the 2026-09-15 implementation review. Focused
regression tests and an all-scenario real-data smoke pass for HALO, HARNet, LiMU-BERT-X, UniMTS,
and NormWear; **no full comparison run has been made.** Implements
`DEPLOYMENT_SCENARIOS_PLAN_20260915.md`. Written for another agent to review and debug.

## 1. What was added

| file | role |
|---|---|
| `training/support_classifier/partial_coverage.py` | Scenario 1 primitives: hidden-candidate choice, support hiding, truth splitting, coverage-safe readouts, and equal-weight normalized fusion |
| `training/support_classifier/scenarios.py` | Scenarios 2-8 primitives: accel-only and resampled derived streams, cross-stream enrolment manifests, shared-label rosters |
| `training/support_classifier/run_partial_coverage.py` | Scenario 1 standalone runner; also the home of `emit_rows`, `conse_scores`, `native_text_scores`, reused by the unified runner |
| `training/support_classifier/run_scenarios.py` | Unified runner for all eight scenarios, with severity stamping |
| `tests/test_partial_coverage.py` | 27 tests |
| `tests/test_scenarios.py` | 23 tests |

The scenario code shares the audited sealed-evaluation feature and readout paths rather than
maintaining separate model adapters.

## 2. How to run

```bash
cd /home/alex/code/HALO/halo
/home/alex/code/HALO/legacy_code/.venv/bin/python -m training.support_classifier.run_scenarios \
  --out training/support_classifier/evaluations/scenarios_20260915 \
  --models halo harnet5 harnet10 limubert_x unimts normwear \
  --halo-checkpoint training/support_classifier/outputs/halo_fixed_mr_residual_v3_curriculum12_40k_20260916/best_internal.pt
```

The default full grid is `k = 0, 1, 2, 4, 8, 16, 32, 64` at 4, 8, and 16 seconds. Pass explicit
subsets only for a bounded diagnostic.

`--smoke` runs up to two cells per scenario, one k, no bootstrap. Two cells are necessary for
matched scenarios because they exercise both the control and perturbation. `--scenarios` selects a subset.
`--max-tasks-per-scenario` bounds a long sweep. Output includes `results.json`, `failures.json`,
`manifests.jsonl`, `predictions.jsonl`, `paired_deltas.json`, `model_artifacts.json`,
`run_provenance.json`, `results.csv`, and `RESULTS.md`;
a scenario that produces no evaluable cell is written to `failures.json` rather than skipped.

Feature arrays are content-addressed by provider artifacts, source slice, and extraction config.
The default shared cache is
`training/support_classifier/evaluations/shared_sealed-feature-v4-20260913`; individually validated
entries from older evaluation directories are reused automatically. Use
`--no-prior-feature-cache-reuse` only for a deliberate cold-cache diagnostic. A bounded 2 GiB RAM
LRU avoids repeated disk loads within a run and is adjustable with `--feature-memory-cache-gib`.

## 3. Design decisions a reviewer should check

1. **Hiding is per cell, not per query.** One hidden candidate set per (dataset, stream, window,
   coverage draw),
   so support counts stay uniform and the batched readouts stay valid. The experimental unit is a
   deployment configuration.
2. **Support-only readouts can never name a hidden candidate.** 1-NN, prototype and ridge score only
   enrolled candidates, so a query whose truth is hidden is necessarily wrong. That is the
   measurement of the capability gap, not a bug.
3. **Equal-weight normalized fusion is fixed and untrained.** Per query: z-score the semantic row
   from the model's declared zero-support route over the full roster, z-score class-wise maximum
   cosine similarity over the enrolled subset, and add with weights `1 + 1`. A hidden candidate
   competes on semantic evidence alone. Nothing is fitted, so no baseline is handicapped and none
   is credited with a tuned combiner. `conse_scores` reproduces the exact matrix
   `scoring.conse_predict` takes its arg-max over, so a bridge model is ranked by its own
   zero-support rule.
   This is the primary external-baseline classifier for every enrolled scenario, not only partial
   coverage. Support-only 1-NN is always reported beside fusion; prototype and ridge are opt-in
   diagnostics.
4. **Truth-restricted metrics are explicit.** Enrolled-truth and unenrolled-truth rows report
   macro F1 over the observed truth/prediction union and balanced accuracy over classes present in
   truth. The output names the split and also reports false-enrolment pull for unenrolled truth.
5. **A model with no text path is recorded as `cannot_attempt`, never scored 0.** Scoring it zero
   would be a strawman.
6. **Cross-stream enrolment never uses the query's own execution**, even across simultaneously
   recorded placements, where the same execution appears in both streams. Overridable with
   `allow_same_execution=True`; the default is off.
7. **Queries whose truth falls outside the shared roster are excluded**, not scored as failures.
   They are unanswerable by every model and would depress all scores equally.

## 4. Bugs found by the smoke (already fixed, tests added)

* **Support labels were read raw instead of aligned to the stream's vocabulary.** MM-Fit stores
  `push_up` in `gt` but registers `pushups` in `eval_labels`, so the concept map matched nothing and
  Scenarios 7 and 8 produced **empty manifests with no error**. Fixed in `build_cross_manifest`;
  regression test `test_cross_manifest_aligns_support_labels_to_the_stream_vocabulary`.
* **Truth-restricted macro F1** (item 4 above).
* **A scenario yielding zero cells was silently a no-op**; it is now a recorded failure.
* **Composite streams have `cell_id`, not `stream`**, which crashed row emission for Scenarios 7-8.
* **A zero bootstrap crashed the CI helper**; smoke mode now skips the CI and records `bootstrap_B`.

## 5. Repo state changes outside the new files — please verify

Building the 8 s grids for the new-domain datasets invalidated two quality caches, which then
blocked **every composite load repo-wide**. Both were rebuilt:

```bash
python -m data.scripts.build_grids --dataset mmfit spar upper_limb_use --alignment native --window-seconds 8
python -m data.scripts.scan_duplicates  --alignment native --window-seconds 8
python -m data.scripts.scan_implausible --alignment native --window-seconds 8
```

`data/quality/duplicate_windows_w8.json` and `implausible_windows_w8.json` are therefore modified.
I verified the rebuild is safe: **the excluded-window sets are byte-identical for every
pre-existing stream**, only the ten new streams were added, and no shared fingerprint changed. The
sealed protocol is unaffected. Backups of both files are in this session's scratchpad. **Re-verify
this independently** — it is the one change that touches sealed evaluation inputs.

## 6. Smoke results (historical one-cell HALO check, k=4, 8 s, no bootstrap)

Not results. A wiring check, single cell, no baselines except where noted. Severity is `L/S/P/C`.

| scenario | severity | 1-NN | HALO classifier |
|---|---|---:|---:|
| s1 partial coverage (all) | 0/2/0/0 | 37.3 | 80.9 |
| s1 truth unenrolled (balanced acc.) | | 0.0 | 86.0 |
| s2 cross placement | 0/0/2/0 | 20.4 | 25.6 |
| s3 cross dataset | 0/0/3/0 | 77.3 | 73.9 |
| s4 missing modality | 0/0/0/1 | 58.1 | 66.9 |
| s5 rate mismatch (20 Hz query) | 0/0/0/2 | 67.1 | 79.3 |
| s6 new domain (MM-Fit) | 3/0/0/0 | 86.0 | 84.2 |
| s7 device set | 0/0/0/3 | 48.5 | 55.5 |
| s8 cold start (all) | 3/2/2/3 | 22.2 | 24.1 |
| s8 truth unenrolled (balanced acc.) | | 0.0 | 9.2 |

Baseline equal-weight normalized fusion, Scenario 1 only, same cell, balanced accuracy on unenrolled truth:
UniMTS 56.9, LiMU-BERT-X 39.8, HARNet 34.3, against HALO's 86.0. **One cell of one dataset with no
confidence intervals; treat as wiring evidence only.**

## 7. Review decisions now enforced

1. `conse_scores` and the bridge prediction share the same score construction; native models use
   adapter-owned candidate scores rather than assuming cosine similarity.
2. The hybrid z-score combination when a candidate has exactly one enrolled support at k=1, and when
   the text row is degenerate.
3. `derive_resampled` uses SciPy polyphase resampling on valid prefixes, updates valid lengths, and
   preserves original acquisition identity and effective bandwidth.
4. Whether excluding the query's own execution is right for Scenario 2a (same subject, other
   placement). It is the conservative choice; the opposite choice is defensible and is one flag.
5. Scenario 8's support stream is `right_wrist` while the query composite is
   `left_wrist+right_pocket`. Confirm that is the intended cold-start story.
6. Truth-restricted rows are emitted only for genuine partial-coverage cells.

## 7b. Corrected baseline input

NormWear now returns pooled backbone tokens for representation readouts and negative native L1
distance for zero-shot candidate scoring. Its feature-cache schema was bumped. The original finding
is preserved in `NORMWEAR_READOUT_FINDING_20260915.md`.

## 8. Not done

* No full comparison run. Bounded real-data HALO and all-primary-baseline probes passed.
* RealWorld's four ungridded sites (chest, head, shin, upper arm) and Shoaib's upper arm are still
  not gridded, so Scenario 2 covers 3 RealWorld and 4 Shoaib placements only.
* New-domain 4/8/16-second grids and task construction were exercised. Optional additional
  RealWorld/Shoaib placements remain outside the current roster.
* The runner writes JSON, CSV, Markdown, per-query decisions, matched deltas, manifests, and model
  provenance. Figure generation remains a post-run reporting step.

## 9. Performance audit

The 2026-09-15 optimization pass made no changes to episodes, predictions, metrics, or model
semantics:

* Paired subject bootstrap now accumulates per-subject confusion matrices and vectorizes the same
  seeded resamples. It is bit-for-bit equivalent to the former per-replicate sklearn estimator in
  parity tests, reducing a typical 1,000-replicate comparison from about 4.1 s to 0.006 s.
* Partial-coverage support readouts and equal-weight normalized fusion are batched. The support roster,
  max-over-enrolments rule, z-score definition, and predictions are unchanged.
* Scenario task limiting now stops construction at the requested limit instead of constructing and
  discarding the full suite. A two-task rate-mismatch smoke fell from 12.4 s to 1.6 s in task setup.
* Composite views, source fingerprints, model-artifact hashes, and feature matrices are cached
  with bounded memory and content validation. Manifest generation retains the frozen NumPy draw
  and canonical JSON fingerprint; optimization must not alter the seed-to-row mapping.
* External baselines always report cosine 1-NN beside equal-weight normalized fusion. Fusion stays
  the declared primary deployment readout, but never hides the representation-only floor.
* HALO residual heads are loaded once per checkpoint/configuration rather than once per task. One
  unloadable stream is recorded without discarding sibling datasets in the scenario cell.
* Sealed evaluation retains released provider states for the process, keeps a bounded decoded
  feature LRU, and computes each model/stream candidate-semantic matrix once for the full k curve.
  Those quantities are k-independent; redoing them at every support count was pure overhead.
* ConSE and HALO label tokens now use one process-wide frozen MiniLM instance and embedding cache
  instead of loading the same 22M-parameter model twice.
* HALO extraction uses a measured-safe CUDA batch of 512. NormWear uses 32 six-channel chunks on
  this RTX 4090; that was the fastest measured point and stays below 8 GiB peak VRAM.
* Matched controls are released after their final perturbation instead of retaining every
  per-query prediction in memory until the end.

An all-model smoke over all eight scenarios produced 407 result rows, 71 matched-delta rows, and
zero failures in 10.6 minutes from a partly cold cache. Six cells were explicitly `unsupported`
because LiMU-BERT-X requires measured six-axis IMU and the relevant streams lacked gyroscope data;
none were silently scored. With those one-time features materialized, a two-cell HALO partial-
coverage smoke took 17.5 s. First-use feature extraction, especially NormWear and the labelled
training reference banks, remains the dominant cold-run cost; later runs reuse exact validated
artifacts.
