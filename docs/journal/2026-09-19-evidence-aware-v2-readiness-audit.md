# Evidence-aware v2: implementation and readiness audit

Date: 2026-09-19. Source: `2c5f7c9` on `fix/training-eval-readiness-20260918`.
Status: report only; production code and historical results unchanged.
Reference: [approved implementation handoff](2026-09-19-new-classifier-design-handoff.md).

## Verdict

The main classifier structure matches the discussion, but the implementation does not yet meet
the full handoff. Do not launch a production experiment and call it the completed repair.
The earlier completion statement overstated what the passing tests established. There are real
objective errors, incomplete counterfactual/validation work, and missing scenario diagnostics.
These findings do not establish that the intended architecture is ineffective: it has not yet
been tested as specified.

## Confirmed findings

### F1. P1: Grouped best-path loss implements a different objective

`training/support_classifier/objectives.py:187` applies softplus per view, takes the arithmetic
group mean, then applies softplus again. The handoff specifies raw regret, group log-mean-exp,
then one softplus. With one view, identical branches, and temperature 0.1, actual loss is
0.10986123 versus specified 0.06931472. This does not satisfy the single-view acceptance test.
It also averages away a difficult view differently from the specified worst-view-sensitive
aggregation. Implement the exact stable logsumexp formula with separate persisted group
temperature, and test ordering, duplicated groups, and worsening individual views.

The reference always includes refined support, even for zero-support rows. With semantic/final
probabilities [0.1, 0.9] and an unavailable uniform support distribution, the loss is 2.19722462
instead of the specified zero-regret softplus baseline 0.06931472. Mask the unavailable expert
before taking the detached maximum.

### F2. P1: Counterfactual defaults break the neighbors control

`train.py:723,1485` forwards probability 0.25 to all classifiers. The expansion in
`sampling.py:1459` ignores `require_query_support`. A probe with that guard true and probability
one produced complete, truth-enrolled partial, truth-unenrolled partial, and zero-support views.
`run_step` rejects zero-support episodes in neighbors mode; truth-unenrolled views also violate
its intended supervision contract. Existing residual training also acquires a new curriculum
silently. Scope the new recipe to v2, enforce classifier-compatible views, and version independent
versus grouped sampling separately. Keep a reproducible ordinary-episode ablation.

### F3. P1: Most counterfactual axes and the scheduler are absent

`sampling.py:344,1459` implements enrollment only. It replaces at most one complete episode per
batch, not a feasibility-balanced schedule of query/candidate groups. There are no grouped
compatible/cross-placement, compatible/cross-dataset, rate, modality, or device-set interventions,
no unavailable-view reasons, and no configured cap on views. Existing independent scenario draws
are useful but are not the same-query counterfactual mechanism in the handoff.

The raw-row table in `prepare_training_batch` is keyed by recording identity alone. That is
correct for enrollment reuse; adding different observations of one recording requires an
observation key including the intervention. Otherwise a clean and perturbed view would reuse the
same encoding. Preserve query identity separately from observation identity.

### F4. P1: Final CE is not averaged by counterfactual group

`train.py:566` still groups by `(support_set_id, zero_shot)` and never reads counterfactual ids.
Other queries sharing a support roster are averaged with three enrolled views of the selected
query, while its zero-support view enters a separately balanced regime. Grouping only the
auxiliary does not implement the specified final-loss weighting. Aggregate explicit groups first
and define the zero/enrolled balancing once, with tests covering mixed independent/grouped rows.

`objectives.py:142` can also merge an independent row with an explicit group: values [1, 9] and
ids [3, -1] produce a single mean [5]. Generated large random ids make this uncommon, but it is
a real violation of the helper's general contract. Remap existing ids before assigning unique
private ids.

### F5. P1: Validation and checkpoint selection do not implement the agreed protocol

`train.py:1250-1410` uses the ordinary subject-held-out validation corpus with optional enrollment
expansion. No source/label-held-out panel or controlled rate/modality/device counterfactual panel
is created. Independent training-source draws do not establish those generalization conditions.

`train.py:1383` flat-averages overlapping summaries (acquisition, enrollment, truth, and clean
complete). One episode appears in several panels; this is not equal family weighting followed
by equal zero/enrolled weighting. The selection change also affects older classifiers.
`run_validation` lacks the regret tie-breaker, `best_zero_support.pt`, and `best_enrolled.pt`.
Use a predeclared immutable panel and architecture-versioned selection policy. Persist counts,
availability reasons, split definitions, and fingerprints. Never select on sealed data.

### F6. P2: Pre-context support status ignores off-roster supports

`evidence_aware_classifier.py:181` calls the neighbor helper only on exact bindings. All-off-roster
supports produce a uniform status distribution and zero support weights. The refined branch
still uses their contextual label compatibility. In a two-row probe, status was [1/3, 1/3, 1/3]
while one refined distribution was [0.7942, 0.0037, 0.2021] even at zero correction. Thus the
contextualizer is not shown the actual initial evidence in precisely the off-roster condition.
Build the pre-context vote over every valid support, with exact bindings for roster labels and
pre-context semantic binding for others. Reuse the same initial score contract in refinement.

The returned `query_support_cosine` is divided by learned temperature at line 190. The handoff
requires raw cosine. Separate raw cosine from temperature-scaled sensor logits. The status branch
also uses a fixed temperature while the refined branch learns one, which should be explicit or
unified to keep the reference interpretation clear.

### F7. P2: Correction stabilization and evidence routing differ from the plan

`evidence_aware_classifier.py:203` retains a tanh pair state and an unnormalized candidate linear
map. There is a learned uncapped scale, but not the specified normalized pair/candidate
projections. Projection norm and explicit scale can both grow, making the scale alone misleading.
Implement normalized states with a deliberate, tested zero-initialization strategy; simply
normalizing a zero vector can itself create a large initial derivative.

At line 218 the router additionally receives the raw three evidence scalars directly. The
handoff supplies them through the existing token projection and contextualized candidate state.
This bypass survives disabling token injection and would confound an evidence-injection ablation.
Remove it or explicitly revise the design and ablation contract before a run.

### F8. P2: Loss/ablation switches are missing or silently ignored

The v2 config in `objectives.py:125` has only a global enable flag and scale. Trainer CLI
`--contextual-aux-comparisons` is still persisted with v1 names, but the v2 call at `train.py:846`
does not forward it. Selecting an empty/subset list therefore does not select v2 losses.
The required final, branch-preservation, and grouped-best-path switches, and the evidence-injection
ablation switch, are absent. Wire explicit versioned options through configuration, resume guards,
and tests. Grouped semantic CE currently deduplicates enrollment views by group, which is useful;
future query-side interventions need distinct semantic observation ids.

### F9. P2: Scenario evaluation lacks the new branch diagnostics

`run_scenarios.py:1453` accepts only the old readout set. Scoring emits the classifier and 1-NN
but not status support, refined support, semantic status, oracle headroom, branch regret, or
rescue/overturn relative to status support. Prediction sinks retain labels, not branch
probabilities, so these measurements cannot be recovered by adding a results table alone.
Add one shared forward returning all branch probabilities per task, then derive diagnostics on
the same manifests and matched controls. Include NLL/Brier/calibration diagnostics where defined.

Aggregate branch dispatch is present, but `branch_logits(..., 'semantic_status')` raises ValueError
despite being a required stable name. It works only with `semantic`. New readout names are omitted
from the learned-head parameter accounting set in `sealed_eval.py:2089`, and v2 reuses v1 output
aliases despite the handoff's explicit versioned naming requirement.

### F10. P2: Sampler and validation telemetry are misleading after expansion

`sampling.py:1459` replaces `episodes` but leaves `regime_episodes` and `support_set_episodes`
unchanged. A five-view probe had 2 complete, 2 partial, and 1 zero, yet reported enrollment
fractions 1.0/0.0/0.0. Pre-expansion support-set counts can be useful if labeled as such, but they
cannot substitute for realized-view counts. Some numerators use pre-expansion rows while their
denominators use the new zero-shot count, so the same-subject rate can be invalid.
Derived views also retain stale masked-candidate and acquisition-regime fields: the new zero view
was stamped compatible rather than not-applicable.

`weighted_present_metrics` does not treat `semantic_reliance` as conditional, so validation
weights a rare condition by the full batch instead of its number of selected rows. Module-level
router/evidence/correction gradients, branch regret distributions, support-only semantic-invariance
checks, requested versus realized axis shares, and encoding-reuse measurements are missing.

### F11. P2: Historical continuation and profiling need explicit version handling

Strict historical factory loading works. However `train.py:2415` serializes every contextual
classifier as v2, even if the resume branch constructed a v1 head. The global sampler-schema bump
currently blocks old continuation earlier; relaxing that guard alone would expose a mislabeled
checkpoint. Serialize the actual instantiated architecture and retain historical schemas/selection
policies when continuing older runs.

The new v2 checkpoint also fails a default resume after training with nondefault episodes-per-step:
the resume override list does not inherit that field. The measured mismatch was saved 1 versus
current default 4. An explicit matching flag works around it; add inheritance while rejecting
explicit conflicting overrides.

`profile.py:238` still creates the v1 head for `--classifier contextual` and profiles the old loss.
It cannot establish speed/memory for v2 plus groups. Update it before relying on a 40-minute
estimate. Shared row encoding is present, but classifier execution pads all flattened views to
maximum S, so zero/partial views can waste attention work beside large complete views.

### F12. P3: Mixed-precision portability and test coverage

Fresh CUDA bf16 forward/backward with three small updates passed. CPU bf16 fails at the indexed
status assignment with Float destination/BFloat16 source. The active CPU training path uses fp32,
so this is not a production CUDA blocker. Use explicit fp32 for sensitive score reductions and
consistent assignment types if claiming CPU autocast support.

There are only four v2-specific tests; they do not verify the grouped formula, disabled switches,
other counterfactual axes, selection policy, or full evaluator diagnostics. Passing the existing
suite was insufficient evidence of full handoff conformance. Add behavioral acceptance tests
for the missing contracts rather than only shape/finite checks.

## Verified working

- Uncentered query-only semantic status is exactly independent of changed support content in
  an fp32 evaluation probe. One semantic map is shared across enrollment regimes.
- Query/support sensor vectors are support-mean centered for the support path.
- Support voting is soft; there is no forced 1-NN selection inside this classifier.
- Status evidence is added to existing support/candidate tokens; roles, recording/label instance
  tags, and the shared attention stack remain present.
- The final output is a normalized, candidate-dependent probability mixture with no free-logit
  residual. Zero-support output reduces to the semantic path within numerical tolerance.
- At initialization with exact bindings, refined/status support agree within 1.91e-6 in fp32.
- The CUDA bf16 synthetic forward/backward probe passed three updates with finite gradients.
- Real training-source CUDA smoke: 3 optimizer steps, 2 loading workers, forced enrollment
  grouping, finite losses, validation, and checkpoint write. Losses were 3.4716, 2.1697, 1.3806;
  these are mechanical checks, not evidence of representation/generalization quality.
- Targeted existing tests: 100 passed in 2.49 seconds. This audit did not rerun the full suite.
- The common evaluator dispatcher scored synthetic k=0 and enrolled plans for all three
  checkpoint families. No sealed recording was scored or used to tune a change in this audit.

Smoke files are in `/tmp/halo-v2-audit-20260919`; they are disposable and not experimental results.

## Comparison anchors

| comparison | architecture | checkpoint | selected step | SHA-256 |
|---|---|---|---:|---|
| current best HALO | support_classifier_v3 | runs/support-classifier/halo_residual_stage_a_40k_20260918/last.pt | 40000 | 56b2cb29fdf093bee26292987fae1048b701ef8560643f2e9868d7a7acf5cc74 |
| previous proposed classifier | support_contextual_residual_v1 | runs/support-classifier/halo_mr_filterbank_contextual_residual_v1_e2e_40k_20260919/best_internal.pt | 35000 | 471ec05f45b83e563de30e489a875b57ed4e77b1002eba59b6319e8b7ff7ebdd |
| new classifier | support_evidence_aware_v2 | no production checkpoint yet | n/a | n/a |

Both historical checkpoint files exist, match the saved hashes, and load strictly through the
current factory. Their decompressed manifest files are byte-identical:

- Aggregate SHA-256: `3117900ff269125ae3cdefe3d3fd1067ad0bd9338b6dcf8dcf7ff8630d0fb985`.
- Scenario SHA-256: `0b39f297102fe3d4d54194de99e72d5e98c53e9a96e16571273c1ab813ca96d2`.

Their existing final accuracy/F1 comparisons can be reused on those manifests. A new classifier
does not require re-encoding external baseline features. New v2 encoder weights do require their
own features, and previously unrecorded branch diagnostics require forwards on the relevant heads.
Generate model-specific result rows and compare only common eligible dataset/scenario/window/k
cells. Keep architecture, checkpoint hash, selection rule, and readout explicit.

The best model used fixed final step 40k; the prior proposal used validation-selected step 35k;
v2 plans a new selection rule and curriculum. These can be compared as complete systems but not
called an architecture-only ablation. For causal attribution use matched-budget runs and a common,
predeclared development selection policy. Existing sealed results must not choose that policy.

## Repair order

1. Correct grouped loss, expert availability, explicit loss switches, and neighbor safeguards.
2. Complete one-axis counterfactual observation construction, group weighting, and truthful metrics.
3. Reconcile pre-context off-roster evidence, normalized corrections, and router inputs with the spec.
4. Build/freeze the development panels and architecture-specific checkpoint selection/resume.
5. Add branch diagnostics to the scenario runner and complete result provenance/accounting.
6. Update the profiler; rerun focused invariance/gradient/resume/evaluator tests and a brief real-data
   smoke. Only then launch a production comparison and populate the results table.
