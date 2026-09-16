# Debug sweep: scenario evaluator and Stage A curriculum (2026-09-16)

**Report only. Nothing was fixed.** Scope: the committed scenario evaluator
(`run_scenarios.py`, `scenarios.py`, `partial_coverage.py`, `sealed_eval.py`, `scoring.py`,
baseline adapters) and the **uncommitted** Stage A curriculum (`sampling.py`, `train.py`,
`residual_classifier.py`, `pretrain_data.py`). Method: three independent code readers plus direct
probes on the real corpus, the real run artifacts and the GPU. Every P1 below was confirmed by a
probe, not by reading alone. Suite at sweep time: 901 passed, 1 skipped.

## 0. The one result that changes the story

The Stage A summary compares the new curriculum only against the old **neighbours-trained**
encoder, and says so. The isolating comparison — the **promoted residual-v3 classifier** (same
objective, same 40k budget, old curriculum) on the **byte-identical manifests** — exists in
`scenarios_full_20260916` (its HALO rows used checkpoint `9b6070d5…`) but is never reported. It is
unfavourable:

| scenario, split=all, primary cells | k | promoted classifier | Stage A classifier | Δ | promoted 1-NN | Stage A 1-NN | Δ |
|---|:-:|---:|---:|---:|---:|---:|---:|
| partial enrollment | 1 | 51.8 | 49.9 | −1.9 | 28.8 | 27.7 | −1.1 |
| cross placement | 1 | 62.8 | 59.7 | −3.2 | 54.6 | 54.5 | −0.1 |
| cross dataset | 1 | 70.7 | 69.6 | −1.1 | 63.4 | 65.4 | +2.0 |
| missing modality | 1 | 55.6 | 54.3 | −1.3 | 53.6 | 53.8 | +0.2 |
| rate mismatch | 1 | 63.2 | 56.6 | **−6.6** | 57.7 | 56.0 | −1.7 |
| new domain | 1 | 39.9 | 38.1 | −1.8 | 39.5 | 39.9 | +0.5 |
| device-set mismatch | 1 | 66.6 | 64.1 | −2.5 | 61.9 | 62.7 | +0.8 |
| cold start | 1 | 23.0 | 23.3 | +0.3 | 17.1 | 23.8 | **+6.7** |
| rate mismatch | 8 | 69.6 | 62.3 | **−7.3** | 67.0 | 66.4 | −0.6 |
| cross dataset | 8 | 74.5 | 73.4 | −1.2 | 69.1 | 72.0 | +2.9 |
| cold start | 8 | 27.4 | 26.2 | −1.2 | 21.0 | 29.6 | **+8.6** |

The classifier is worse under Stage A on seven of eight scenarios, including the three it targets.
The encoder floor improves on cross-dataset and cold start. Whether this is the curriculum or the
defects in §1 cannot be separated yet, which is the point of reporting them together.

## 1. Curriculum (uncommitted) — confirmed defects

**C1 — P1. "Cross-placement" episodes are mostly not cross-placement.** `_keys_for` restricts the
*support* keys to a different site (`sampling.py:333-340`), but `_additional_queries`
(`sampling.py:631-680`) draws the three extra queries per support set **from those support keys**,
so the extra queries sit at a support placement. Probe on the real corpus, clean (non-fallback)
cross-placement sets, 150 batches: base queries had **0 of 6,894** same-site support rows; the
additional queries had same-site rows in **113 of 138 episodes (82%)** and none shared the base
query's site. Only 106 of 274 episodes labelled `cross_placement` (39%) are cross-placement for
their own query; per-episode that is ~5% of all training episodes against a requested 25%. Every
downstream reader of `acquisition_regime` inherits the mislabel: `scenario/acquisition/*` training
telemetry (`train.py:764`) and the run's own `sampler/acquisition_cross_placement_fraction`
(mean 0.271 over the Stage A log). Cross-dataset is **not** affected: the same function requires
`unit[0] == base.dataset` (`:659`), and purity measured 1.000.

**C2 — P1 (disclosure, quantified). The realized mixture is far from the requested one.** Stage A
log means over 801 batches: compatible **0.631 / cross-placement 0.271 / cross-dataset 0.098**
against 0.50 / 0.25 / 0.25; `curriculum_fallback_fraction` 0.176. With C1 applied, the model
trained on roughly 63% compatible, ~10% cross-dataset, and a cross-placement bucket that was mostly
compatible-for-its-query. The fallback rule is correctly disclosed in telemetry, but the design
document states target shares as if delivered, and the experiment's negative result in §0 was
obtained under this realized mixture, not the designed one.

**C3 — P2. Three recorded hyper-parameters are inert under the new path.** With
`deployment_matched=True`, `enrollment_regime` is always set, so `shared_masked` is never `None`
and the historical masking branch (`sampling.py:983-993`) never runs: `p_mask_candidate=0.25` and
`p_mask_gt=0.10` do nothing. `p_gt_present=0.5` is overridden to 0/1 by regime (`:1210`). All three
are still written to `run_config.json` and the resume trajectory as if active. Stage A's provenance
therefore claims masking it did not apply.

**C4 — P2 (verified by probe). Resuming any pre-gate checkpoint fails.** `train.py:1857-1859`
migrates `classifier_config.regime_split` but not `adaptive_text_gate` / `text_gate_hidden`; a
fresh `asdict(cfg)` contains them, the saved sub-dict does not, so the equality check exits. The
Stage A checkpoint itself lacks both keys.

**C5 — P2 (agent-verified). `regime_split` + `adaptive_text_gate` + `text_term_enabled=False`
crashes in the evaluator's `halo-classifier-residual-off` row** (`residual_classifier.py:394-410`:
destination guarded, source not; member returns `None`).

**C6 — P2. The gate is fed a learned quantity, not deterministic evidence.** `metric_logits`
(post-residual, `residual_classifier.py:304→321`) drives `margin`/`disagreement`, opening a
`r_support → gate → λ` path the design document excludes. The k=0 prior shares the same MLP
residual (`:242`), so enrolled-episode gradients move the zero-support weight — the k=0/k>0
coupling that the regime-split experiment was run to rule out.

**C7 — P3. Reporting denominators.** `unequal_support_count_fraction` (0.297 in the log) divides by
all support sets including zero-shot; the design target ("half of enrolled sets") is per enrolled
set, where the realized value is 0.35 and the k=1 gate caps it near 0.375. The `classifier/lambda_*`
telemetry under the gate reports the base prior only, under the same keys as the non-gate arm.

**C8 — P3. The promoted command line no longer reproduces the promoted curriculum.** Defaults
changed: `--enrollment-k` (1,2,4,8 → 1…32), and `--acquisition-mix` / `--enrollment-mix` now
default to the Stage A mixture. Re-running the promoted invocation without flags trains Stage A.

**C9 — P1 (confirmed on the real corpus). Cross-condition query labels are an alphabetical
prefix.** `_eligible_deployment_datasets` stops collecting viable query labels at eight
(`sampling.py:1127`), iterating a **sorted** vocabulary, and every non-compatible draw is forced to
pick its query label from that cache (`:1268`). Dumped cache: RealDisp cross-placement queries are
`arms_frontal_crossing … heels_alternately_to_the_backside` — 8 of 33 labels, **never walking,
running, jogging or jumping**; xrf_v2 is `answering_phone … opening_curtains`; HARMES
`applying_hand_cream … build_with_lego`. For the *enrolled* cross-dataset variant RealDisp has
exactly one viable label, `walking`, and dsads two. The mismatch arm of the curriculum therefore
trains on a deterministic alphabetical slice of each vocabulary, and the locomotion classes the
scenario evaluator scores are excluded from it by construction. The comment claims "eight labels
retain useful diversity"; that would be true of a sample, not of a prefix.

**C10 — P1. Zero-enrollment episodes carry an acquisition regime they cannot have.** Acquisition
and enrollment are drawn independently, and the zero-support branch stamps `acquisition_regime =
mode` (`sampling.py:882`). About a quarter of every "cross-placement"/"cross-dataset" support set
has no support rows and is indistinguishable from a compatible zero-shot episode. Reader's probe:
15/45 cross-placement and 4/20 cross-dataset episodes had empty support. Together with C1 and C2,
the share of training episodes that actually present *mismatched evidence* is roughly 6% + 6%
against a designed 25% + 25%.

**C11 — P2 (confirmed). "Different body site" is a string inequality.** HARMES `watch_wrist` is
`right_wrist`; WISDM `watch_wrist` is `wrist_unspecified`; same device family, channels and
gravity, so each is the other's **only** cross-placement partner. Those episodes are wrist-to-wrist,
i.e. a cross-dataset pair labelled cross-placement. HARMES's whole cross-placement roster is
`['brushing_teeth', 'drinking']`, both WISDM labels.

**C12 — P2. `--mode` is dead on the deployment path and the trajectory records it.**
`acquisition_names` is the fixed 3-tuple whenever `acquisition_mix` is not `None`, and argparse
never gives `None` (`train.py:1128`); `kwargs["mode"]` is overridden per support set
(`sampling.py:1285`). `trajectory["mode"] = "compatible"` for a run that was 27% cross-placement,
and resume validation compares that inert field.

**C13 — P2. Regime telemetry is wrong in two independent ways.** (a) `scenario/*/loss` uses
**unmasked** logits (`train.py:750`) while the objective masks padded candidates (`:522`), so
padding width leaks into the per-regime loss and correlates with regime. (b)
`weighted_present_metrics` (`train.py:602-614`) weights each group by its total episode count, not
by how many episodes matched the condition, so every validation `scenario/*/loss`, `accuracy` and
`semantic_weight` cell is mis-weighted; only `fraction` is right. (c) `support_per_candidate = k`
is the **pre-downsample** value and `shrunk=False` is hard-coded (`:1006-1008`), so an episode drawn
at k=32 and thinned to one row per candidate is logged in the k=32 bucket.

**C14 — P2. Two resume traps.** `--no-adaptive-text-gate` bypasses the resume guard, which tests
only the positive spelling (`train.py:1434-1438`). A checkpoint with an empty `trajectory` block
resumes with `acquisition_mix = enrollment_mix = None` and silently trains the pre-curriculum
sampler (`:1396-1399` + `:1425`).

**C15 — P3. Sampler cost.** ~17–32 ms of single-threaded Python per support set (`_available_units`
scans keys × labels twice; `feasible_by_k` calls `_can_draw_candidate_count` up to 12×). Absorbed by
the forked workers today (loader wait stayed at ~21 ms of a 55 ms step in my profile), but it is
now the dominant prefetch term. `sampler/eligible_dataset_count` is computed for a regime the run
never draws (`p_gt_present=0.5`, `mode=compatible`), so it says 8 sources are eligible while only
dsads, forth_trace and xrf_v2 can form enrolled cross-dataset episodes and kuhar can form no
cross-placement episode at all — which is the real cause of the 0.098 realized cross-dataset share.

**Open, needs a GPU probe.** Whether `_adaptive_lambda` is dtype-safe under bf16 autocast:
`sums`/`squares` are `k_c.new_zeros` and the similarity is explicitly `.float()`; if `k_c` arrives
as bf16 the `scatter_add_` mismatches. Run `test_residual_classifier.py` with the gate on inside
`torch.autocast("cuda", torch.bfloat16)` and assert finite λ and nonzero gate gradients.

**Working-tree note.** The sampler reader observed another agent reverting `partial_coverage.py`,
`run_scenarios.py`, `sealed_eval.py` and `test_partial_coverage.py` *during* the sweep; at the time
this file was written those four were back at HEAD. Line numbers for `sampling.py` refer to the
uncommitted version present at sweep time.

**Verified correct (curriculum).** Deterministic under seed; 0.084 s/batch; no support from the
query's execution (2,043 episodes); `same_subject` and `cross_subject` honoured (1.000 / 0
violations); cross-dataset purity 1.000; partial enrollment independent of the query label
(P(truth enrolled) 0.519 vs mean coverage 0.513, 530 episodes); `support_counts` consistent; the
adaptive gate is off by default and the working-tree classifier is **bit-identical** to HEAD on
the promoted checkpoint (max |Δlogit| = 0.0, λ telemetry equal); no ground-truth leakage into the
gate inputs (agent trace).

## 2. Scenario evaluator (committed, already ran) — confirmed findings

**E1 — P1. The same-subject cross-placement rung silently vanished.** `build_cross_manifest`
needs ≥k same-person supports from another execution for **every** candidate; Shoaib
`phone_right_pocket` has one execution per (subject, label) in 70/70 pairs, RealWorld forearm in
49/54. The result: **0 of 72** s2 manifests are `same_subject`; the P=1 rung the plan defines is
absent from the run, and the empty-scenario guard never fires because cross-subject cells exist.
The severity axis in the results therefore has no P=1 column and nothing says why.

**E2 — P2 (latent, did not bite here). Paired-delta class sets.** `f1_macro_delta` is the
difference of two per-side macro-F1s over each side's own GT∪pred set, while the CI-bracketed
`f1_macro_difference` uses the union set. In this run the two agree in all 584 rows (max gap 0.0),
but a scenario that induces new false-positive classes would make the headline column disagree
with its own interval.

**E3 — P2. Two "scenario 1" runners are not the same scenario.** `run_scenarios` uses cross-subject
manifests; the standalone `run_partial_coverage` uses `build_manifest`, which permits same-person
enrolment, and both stamp `partial_coverage_v1`. The published run used the former; the latter is
a trap for anyone merging outputs.

**E4 — P2 (latent). `cannot_attempt_rows` hard-codes `scenario=partial_coverage_v1`** and takes
no metadata, so an s6/s8 "cannot attempt" row would be filed under s1 with no variant or
severity. No such row occurred in the published run (statuses were only `ok`/`unsupported`).

**E5 — P2. Restricted-split reporting.** The exhaustive `RESULTS.md` prints macro-F1 for
`truth_enrolled` / `truth_unenrolled` rows without the explicit label set that makes it meaningful
and without the `primary_metric` / `balanced_accuracy` columns; the CSV also drops `primary_metric`,
`coverage`, `hidden_candidates` and every `severity_*` field. The `SUMMARY.md` tables use the
`all` split and are unaffected.

**E6 — P2. Every pre-existing sealed feature cache is orphaned.** `source_slice_fingerprint` now
hashes `effective_source_rate_hz` and `perturbation` for every stream (`baselines/data.py:167-168`),
so the cache key changed for **unperturbed** streams too, and `FEATURE_CACHE_SCHEMA` was not
bumped. Probe: the promoted sealed cache holds no file for the current key of
`motionsense/phone_front_pocket`. Six sealed cache directories (78 HALO files each) are now dead
weight that "reuse prior cache" will scan and never hit. Numerically harmless (E7), operationally a
silent full re-encode on the next sealed run.

**E7 — cleared. Batch size 512 versus 256 does not change sealed HALO features.** GPU probe on the
promoted checkpoint, 3,693 windows: max |Δ| = 0.0, mean cosine 1.0, zero 1-NN neighbour flips,
repeat-run identical.

**E8 — cleared. The bootstrap change is bit-identical for the sealed k-curve.** Old and new
`subject_bootstrap_ci` give identical intervals for `f1_macro` (the only metric the sealed path
uses); the new code additionally makes truth-restricted balanced accuracy computable where the old
code raised.

**E9 — P2 (latent). k=0 label filter missing in `build_manifest`'s zero-support branch**
(`sealed_eval.py:251-252`); currently every roster grid's extra labels canonicalise into a
candidate, so it does not bite, but it is the only k=0 path without the guard.

**E10 — P2. One unattemptable cell discards all paired deltas for its (scenario, window, k)**
(`_PairedDeltaTracker.finish` raises; caught; `output` dropped), and one bad stream aborts all
of s5 at that k (per-scenario `try`, unlike s4's per-cell handling).

**E11 — P3 (latent). `_effective_source_rate` falls back to the grid rate, not the source-rate
table**, so a table-listed stream (`mmfit/left_ear` 85 Hz, `xrf_v2/airpods_ear` 25 Hz) that is rate
perturbed would over-state its bandwidth. No listed stream is in the current perturbed roster.

**E12 — P3.** Zero-shot bank cache path is a hard-coded relative string; `_halo_residual_predictions`
reloads the checkpoint per task; support-side feature fingerprints are discarded in cross rows;
`results.json` is rewritten after every task; `_file_hash`'s mtime-keyed cache can return a stale
digest for a same-size, same-mtime rewrite.

**Verified correct (evaluator), over the real run.** Manifests byte-identical across the HALO and
baseline runs (SHA-256 match). Over all **746,240** persisted plans: no support from the query's
execution; no cross-subject plan shares a person with its supports; no hidden label carries
support; per-candidate counts equal k; hidden rosters invariant across k in all 11 coverage cells.
The 34,618 cross-dataset plans with same-dataset supports are exactly the `condition=control`
matched cells; every primary cross-dataset plan is pure. Derived streams keep their registered
identity and cannot collide in the cache. Summary aggregates for s1/s3/s4/s5/s7 recompute exactly
from the raw rows (primary cells, `all` split). LiMU-BERT's 98 gyro-less cells are `unsupported`,
never scored.

## 3. Priority for whoever fixes

1. C1, C9 and C10 together, then re-measure §0 — the curriculum result is uninterpretable while
   the mismatch arm is mostly compatible episodes drawn from an alphabetical label prefix.
2. E1 — restore or explicitly record the missing P=1 rung.
3. C4 and E6 — one-line migrations that currently block resume and silently re-encode.
4. C3, C8, C12 and C13 — provenance and telemetry: stop recording inert knobs as active, keep the
   promoted command line reproducible, and make the per-regime numbers mean what they say.
5. C6 — decide whether the gate may see `metric_logits`; the design says it may not.
6. E3/E4/E5 — reporting hygiene before any table leaves the repo.
