# New classifier design: implementation handoff

**Date:** 2026-09-19  
**Status:** approved implementation plan; no code changes or training authorized by this entry.  
**Supersedes:** the implementation decisions in
[`2026-09-19-evidence-aware-classifier-repair-plan.md`](2026-09-19-evidence-aware-classifier-repair-plan.md).
That entry's diagnosis, literature review, and historical results remain valid. This handoff removes
the proposed direct router-supervision and paired-router losses after discussion with Alex.

## 1. Objective

Modify the current proposed contextual residual classifier in place into the **new classifier
design**. The classifier must preserve two independently auditable evidence paths, show their
pre-context predictions to the episode contextualizer, and learn a candidate-specific normalized
mixture. Training uses counterfactual episode groups that hold the recognition problem fixed while
varying one realistic deployment heterogeneity.

The only active loss families are:

1. final classification;
2. branch preservation;
3. best-path non-regression, aggregated over each counterfactual group.

There is no direct router target, no paired-view routing loss, no scenario-specific instruction to
trust one branch, and no fixed cap on learned support corrections.

## 2. Compatibility and naming

- User-facing name: **new classifier design** until a paper-facing name is chosen.
- Persisted architecture identifier: `support_evidence_aware_v2`.
- Keep `--classifier contextual` as the active CLI selection; it constructs v2 after this change.
- Do not overwrite `support_contextual_residual_v1` checkpoint semantics. Its class and strict
  loader remain available for reproducing the 2026-09-19 result.
- The factory must dispatch by persisted `architecture_version`, never by guessing from keys.
- A v1 checkpoint cannot resume as v2. Warm-starting v2 from v1 is not part of the first experiment.
- Historical result documents remain unchanged. Living contracts change only after tests pass.

Likely files:

- `model/support/contextual_residual_classifier.py`: retain or move the v1 class; add the v2 model.
- `model/support/factory.py`: make `contextual` construct v2 and keep strict v1 loading.
- `training/support_classifier/objectives.py`: grouped loss implementation.
- `training/support_classifier/sampling.py`: counterfactual group construction and schema.
- `training/support_classifier/train.py`: grouped execution, loss integration, telemetry, validation,
  checkpoint selection, and persisted trajectory.
- `training/support_classifier/sealed_eval.py`, `run_scenarios.py`, and
  `run_partial_coverage.py`: v2 loading and diagnostic readouts.
- `tests/test_contextual_residual_classifier.py`: retain v1 tests.
- Add focused v2 model, objective, sampler, evaluator, and checkpoint-selection tests rather than
  making one large test module cover every contract.

## 3. Model contract

### 3.1 Inputs that remain unchanged

For a batch with `B` views, `S` padded supports, `C` padded candidates, sensor width `E`,
acquisition width `A`, text width `T`, and model width `D`:

- query recording vector: `[B, E]`;
- support recording vectors: `[B, S, E]`;
- query and support acquisition-conditioning vectors: `[B, A]` and `[B, S, A]`;
- support-label text vectors: `[B, S, T]`;
- candidate-label text vectors: `[B, C, T]`;
- support-to-candidate binding, support mask, candidate mask, and support pair slots.

The encoder, learnable recording pool, acquisition conditioning, role embeddings, and support
recording/label instance tags remain end-to-end trainable. Runtime acquisition information remains
implicit in the recording-conditioning vectors; no extra acquisition token is required.

### 3.2 Stable pre-context evidence paths

Compute both paths before the set-attention stack.

#### Support status quo

1. Centre query and support sensor vectors using the existing support-set mean contract.
2. Compute cosine query-to-support similarity for every valid support.
3. Apply a softmax over valid supports. The query is never forced to choose one support.
4. Map each support label to candidate labels:
   - use its exact candidate binding when the support label is in the candidate roster;
   - use normalized support-label/candidate-label semantic compatibility only for a genuinely
     off-roster support label.
5. Sum support mass by candidate and normalize across valid candidates.
6. Keep unsupported candidates finite and neutral rather than impossible. With no support, the
   status-quo support distribution is uniform and marked unavailable.

Return normalized `support_status_logp: [B, C]` and raw
`query_support_cosine: [B, S]`. The result must be permutation-invariant over supports.

#### Semantic status quo

Project the **uncentred, query-only** pre-context query content and pre-context candidate-label
content into one metric space, compute cosine similarity, apply the learned semantic temperature,
and normalize over candidates. Never use the support-centred query `q0` for this path because its
value changes when the support roster changes.

Use one shared semantic status-quo map for zero- and nonzero-support views. For identical query,
candidate labels, and acquisition input, `semantic_status_logp` must be bit-identical regardless of
which support tokens are present. The current v1 split between zero-support and enrolled semantic
maps must not survive in this status-quo path.

### 3.3 Evidence injection into existing tokens

Evidence describes an entity and therefore becomes a feature of that entity, not another token.

For support `i`:

```text
support_token[i] += support_evidence_projection(query_support_cosine[i])
```

For candidate `j`, with `C_valid` valid candidates:

```text
support_relative[j]  = support_status_logp[j]  + log(C_valid)
semantic_relative[j] = semantic_status_logp[j] + log(C_valid)

candidate_token[j] += candidate_evidence_projection(
    support_relative[j],
    semantic_relative[j],
    has_direct_support[j],
)
```

- `has_direct_support` is binary.
- Do not add raw support count or effective sample size in v2.
- Use one shared projection for every support and one shared projection for every candidate so
  variable `S` and `C` require no new parameters.
- Zero-initialize the final linear projection of each evidence adapter. This preserves the current
  token content at initialization while allowing immediate gradients into that linear layer.
- Add projected evidence through the existing normalized token-composition mechanism. Do not
  create scalar tokens, positional candidate identities, or `S x C` pair tokens.

### 3.4 Contextualizer

Run the query, support-recording, support-label, and evidence-augmented candidate tokens through the
existing permutation-safe `SetAttentionStack`. Preserve:

- query/support/support-label/candidate role embeddings;
- one shared instance tag for each support recording and its support-label token;
- no instance or positional identity for candidates;
- all current support and candidate padding masks.

The contextualizer now reasons from both entity content and explicit status-quo evidence. It does
not emit unrestricted candidate logits.

### 3.5 Refined support path

Construct the support path from auditable pair evidence:

```text
pair_score[i,j] =
    query_support_sensor_score[i]
    + support_label_candidate_score[i,j]
    + contextual_correction[i,j]
```

The contextual correction reads contextualized query, support, support-label, and candidate states.
Replace the current fixed `tanh` correction cap with normalized projections and a learned scale:

```text
pair_state[i] = normalize(f(query_h, support_h[i], support_label_h[i]))
candidate_state[j] = normalize(g(candidate_h[j]))
correction[i,j] = exp(log_correction_scale) * dot(pair_state[i], candidate_state[j])
```

Initialize the candidate projection or learned scale so correction is exactly zero initially. The
scale is learned and has no fixed maximum. Weight decay, finite-value checks, ordinary global
gradient clipping, and normalized projections provide stability.

Normalize valid pair scores, sum pair probability by candidate, and renormalize over candidates to
obtain `refined_support_logp: [B, C]`. At initialization and with exact enrolled labels, this must
match `support_status_logp` within numerical tolerance.

### 3.6 Candidate-specific normalized mixture

A shared MLP reads the contextualized query and candidate state. Candidate evidence is already in
the candidate state. It emits one semantic-reliance logit per valid candidate:

```text
g[j] = sigmoid(router(query_h, candidate_h[j]))
```

For any row with no support, force `g[j] = 1` for valid candidates because the support expert is
structurally unavailable. This is availability masking, not scenario-specific trust supervision.
For enrolled rows, initialize the router near support-dominant behavior, matching v1's conservative
semantic initialization, but leave it fully learnable.

Combine in probability space and renormalize because `g` varies by candidate:

```text
mixed_mass[j] =
    (1 - g[j]) * exp(refined_support_logp[j])
    + g[j] * exp(semantic_status_logp[j])

final_logp = masked_log_softmax(log(mixed_mass), candidate_mask)
```

Implement this with `logaddexp` and clamped log gates for mixed-precision stability. There is no
third free candidate-logit path and no post-mixture residual.

### 3.7 Required output dictionary

Return at least:

- `logits` / `final_logp`;
- `support_status_logits`;
- `refined_support_logits`;
- `semantic_status_logits`;
- `semantic_reliance`;
- `query_support_cosine`;
- `support_weight` and contextual pair probability;
- `support_correction`;
- `candidate_has_direct_support`, `has_any_support`, and `k_c`;
- support and semantic temperatures and learned correction scale.

Diagnostic branch names must be intuitive and stable:

- `final`;
- `support_status`;
- `refined_support`;
- `semantic_status`.

Do not reuse v1 output names with changed meanings without a compatibility alias local to the v1
evaluator.

## 4. Counterfactual episode groups

### 4.1 Definition

A counterfactual group fixes:

- one physical query execution;
- one candidate-label roster and ordering;
- one ground-truth candidate slot;
- one subject/split provenance;
- one deterministic group seed.

Views vary exactly one declared deployment axis. Do not create a Cartesian product of all axes in
one group. This keeps the comparison attributable and group size bounded.

Add a group-level structure rather than inferring groups from adjacent rows:

```text
CounterfactualGroup
    group_id
    query
    candidates
    gt_slot
    axis
    views: tuple[Episode, ...]
```

Each `Episode` additionally records `counterfactual_group_id`, `counterfactual_view`,
`counterfactual_axis`, and a machine-readable intervention description. Persist the sampler schema
version and realized view roster in checkpoints.

### 4.2 Mapping to the evaluation scenarios

| evaluation heterogeneity | training counterfactual group | invariant | varied views |
|---|---|---|---|
| partial enrollment | enrollment | query, candidates | complete, partial truth-enrolled, partial truth-unenrolled, zero |
| cross placement | support acquisition | query, candidates | compatible and honest within-dataset cross-placement support |
| cross dataset | support source | query, candidates | compatible and honest shared-label cross-dataset support |
| missing modality | modality nuisance | physical execution, candidates | clean and truthful query-side or support-side gyro removal |
| sampling-rate mismatch | rate nuisance | physical execution, candidates | clean and anti-aliased lower-rate query or support observation |
| new domain | no synthetic equivalent | none | measured through source-held-out and label-held-out development panels |
| device-set mismatch | device composition | physical execution, candidates | matched, overlap/superset, and disjoint sets where real aligned devices permit |

Cold start remains excluded. Support-label shuffling is a negative diagnostic only and is not a
default training scenario because it is not a normal deployment heterogeneity.

For support-only axes, the raw query and semantic status quo are identical across views. For
query-side rate/modality/device perturbations, the query refers to the same physical execution but
its observed sensor input changes; branch references are therefore computed separately per view.

### 4.3 Construction rules

1. Draw the query and candidate roster before choosing view-specific supports.
2. The ground-truth label is always in the candidate roster.
3. Enrollment masking is independent of the ground truth; truth-enrolled and truth-unenrolled are
   consequences recorded by the sampler, except when deliberately constructing both named views
   inside an enrollment group.
4. Every support remains execution-distinct from the query and obeys declared subject, dataset,
   placement, modality, gravity, and device compatibility rules.
5. A requested view that is infeasible is recorded as unavailable; never silently substitute a
   different heterogeneity. A valid group has at least two views.
6. Select group axes with a feasibility-aware balanced scheduler. Log requested and realized shares.
7. Use all feasible levels for the selected axis, capped by a configurable maximum. The initial
   active recipe should use the clean/control view plus one or two heterogeneous views, not every
   Cartesian combination.
8. Keep ordinary ungrouped sampling available behind a flag for an exact ablation.

### 4.4 Efficient execution

- Form the union of physical recording rows used by every view in a batch and encode each unique
  row once.
- Reuse query vectors, candidate text vectors, and acquisition vectors across applicable views.
- Gather view-specific support tensors from the shared encoded-row table.
- Count `groups_per_step` separately from flattened `views_per_step`.
- Preserve current channel-width bucketing, deferred patch materialization, bf16 autocast, pinned
  transfer, and multi-device batching.
- Profile group assembly, CPU loading, unique encoder rows, classifier time, GPU utilization, and
  peak memory before setting the production group/view counts.

## 5. Training objectives

### 5.1 Final classification

Compute ordinary candidate cross-entropy for every view. Average views within a group, then average
groups, so an axis with more feasible views does not receive more weight:

```text
L_final = mean_group(mean_view(CE(final_logp, truth)))
```

Keep zero-support and enrolled group contributions balanced when both are present, matching the
current intent without duplicating one semantic loss for every support-only counterfactual view.

### 5.2 Branch preservation

One modular loss family keeps both evidence paths independently useful:

- semantic-status cross-entropy for each unique query/candidate observation;
- refined-support cross-entropy on views where the truth has direct enrollment;
- contextual-support non-regression against the detached support status quo on those same views.

Average only active subterms so enabling or disabling one does not silently rescale the family.
Do not train the support path to infer an unenrolled truth from distractor supports; semantics and
the final mixture own that case.

### 5.3 Counterfactual best-path non-regression

For each view `m`, compute true-class log-odds:

```text
q_support[m] = log_odds(refined_support_logp[m], truth)
q_semantic[m] = log_odds(semantic_status_logp[m], truth)
q_final[m] = log_odds(final_logp[m], truth)

q_reference[m] = max(detach(q_support[m]), detach(q_semantic[m]))
regret[m] = q_reference[m] - q_final[m]
```

If support is structurally absent, the reference is semantic only. Aggregate within the complete
counterfactual group before applying the smooth non-regression penalty:

```text
group_regret = tau_group * log(mean(exp(regret[m] / tau_group)))
L_best_group = tau * softplus(group_regret / tau)
L_best = mean_group(L_best_group)
```

Use stable `logsumexp`. The normalization by group size keeps the value comparable across group
sizes. Margin remains zero. The detached reference prevents either branch from degrading itself to
make the target easier.

There is no router-supervision loss and no paired-view routing-direction loss in v2.

### 5.4 Total loss and switches

```text
L_total = L_final + w_aux * mean(active(L_branch, L_best))
```

- Retain the existing default auxiliary scale and temperature for the first matched experiment
  rather than introducing an uncalibrated search.
- `final`, `branch_preservation`, and `best_path_group` are separate configuration switches.
- Persist all switches, weights, temperatures, sampler settings, and schema versions in trajectory
  metadata. Resume must reject a mismatch.
- No loss freezes a component or routes gradients manually. All non-reference computation remains
  end-to-end differentiable through classifier, acquisition conditioning, recording pool, and
  encoder.

## 6. Validation and checkpoint selection

Build a deterministic development panel from training-source validation subjects only. It must
contain fixed counterfactual groups, a fixed fingerprint, and both source-held-out and label-held-out
views. Sealed datasets never participate in checkpoint selection.

### 6.1 Required validation strata

- zero support versus enrolled;
- complete, partial truth-enrolled, and partial truth-unenrolled;
- compatible, cross-placement, and cross-dataset support where feasible;
- clean versus missing-modality, rate-mismatch, and device-set views;
- source-held-out datasets;
- canonical labels held out from classifier training episodes.

### 6.2 Selection metric

For each fixed validation cell, compute macro-F1 per dataset. Give each deployment family equal
weight, then give zero-support and enrolled summaries equal weight when both are available. A cell
declared unavailable when the panel is built remains unavailable for every checkpoint.

Select `best_internal.pt` by:

1. highest scenario-balanced dataset-macro-F1;
2. on an exact tie, lower mean final cross-entropy;
3. then lower counterfactual positive regret relative to the better branch.

Also retain diagnostic `best_zero_support.pt` and `best_enrolled.pt`; neither is automatically
promoted. Keep the historical enrolled-only selection metric in logs for comparison, but stop using
it to choose v2.

## 7. Evaluation contract

### 7.1 Standard aggregate evaluation

Run the existing sealed k-curve and per-dataset protocol on immutable manifests and identical query
windows. Report the new classifier's `final` readout as the primary HALO classifier row. Preserve
the agreed recording-window and k grids from the evaluation contract; do not embed a new grid in
the model implementation.

Diagnostic HALO rows, not headline alternatives:

- support status quo;
- refined support;
- semantic status quo;
- encoder 1-NN where the protocol defines it;
- oracle better-branch upper bound, clearly labeled non-deployable.

Report accuracy, macro-F1, and balanced accuracy per dataset and overall. Keep NLL, Brier score,
and calibration error as diagnostics.

### 7.2 Deployment-scenario evaluation

Run the existing seven realistic scenarios and their matched controls. Use the same manifests for
the promoted current classifier and v2. Report, per dataset and scenario:

- final accuracy, macro-F1, and balanced accuracy;
- scenario-minus-control paired delta;
- support-status, refined-support, and semantic-status performance;
- final regret to the better branch;
- oracle better-branch headroom;
- rescue and harmful-overturn rates relative to support status;
- results split by truth enrolled versus unenrolled.

Do not add cold start. Support-label shuffle and metadata shuffle remain diagnostic sensitivity
tests, not deployment scenarios or headline rows.

### 7.3 Comparability

- Do not rerun external baselines solely because the HALO classifier changed when their cached
  features, manifests, windows, and readout protocol remain identical.
- Rerun the current promoted HALO classifier on any newly versioned manifest required for a paired
  comparison.
- Persist protocol version, manifest fingerprint, checkpoint SHA, source commit, architecture
  version, parameter count, window duration, k, dataset, scenario, and readout in every result row.

## 8. Telemetry

### 8.1 Sampler and efficiency

- requested/realized counterfactual axis shares;
- group-size distribution and unavailable-view reasons;
- views per group and groups per step;
- truth-enrollment rate per view;
- source, subject relation, acquisition relation, rate, modality, and device-set distributions;
- unique encoded rows versus naive flattened rows and reuse ratio;
- load, collate, transfer, encoder, classifier, backward, and optimizer timings;
- peak allocated/reserved GPU memory and GPU utilization sample.

### 8.2 Loss and branch quality

- final, branch-preservation, and grouped best-path losses separately;
- status-support, refined-support, semantic, final, and oracle accuracy/macro-F1;
- true-class log-odds for every branch;
- per-view and per-group regret: mean, median, p90, maximum, and positive fraction;
- rescue, overturn, preserve, and both-wrong rates against support status;
- branch metrics split by counterfactual axis/view, k bucket, truth enrollment, dataset, and device
  relation.

### 8.3 Learned behavior

- semantic reliance mean, standard deviation, p10/p50/p90, and saturation fractions near 0 and 1;
- semantic reliance split by support availability, truth enrollment, scenario, and branch advantage;
- support-correction mean absolute value, RMS, p95, and learned scale;
- support attention entropy and effective support rows;
- maximum semantic-status difference among support-only views of one group; expected exactly zero;
- prediction sensitivity to support removal, support-label shuffle, and metadata shuffle.

These are diagnostics only; semantic reliance is not directly supervised.

### 8.4 Optimization health

- gradient norms for encoder, recording pool, acquisition text/structured conditioners, evidence
  projections, contextualizer, support correction, semantic projection, and mixture router;
- total pre-clip norm and clip coefficient;
- non-finite checks for every branch, gate, correction, and loss;
- encoder effective rank, learned temperatures, learning rates, and parameter/update norms.

Log compact scalar summaries at the current telemetry interval. Store expensive histograms or
per-example rows only at validation checkpoints.

## 9. Tests and smoke acceptance

### 9.1 Model tests

- support status is invariant to support permutation;
- semantic status is bit-identical when only support content, order, or enrollment changes;
- zero support reduces exactly to semantic status;
- evidence projections modify the intended existing tokens and create no new token role;
- candidate permutation is equivariant;
- padded supports/candidates cannot change valid outputs;
- v2 initialization preserves the support floor on enrolled rows within declared tolerance;
- candidate-specific mixture is normalized, finite, and bf16-safe;
- no fixed support-correction cap remains;
- gradients reach every intended component from the three active loss families;
- variable `S`, variable `C`, multi-device input, and all-support-masked rows work.

### 9.2 Sampler tests

- every group shares query identity, candidate order, truth slot, and split provenance;
- exactly one declared axis varies;
- every support is execution-disjoint and satisfies its declared relation;
- partial enrollment is independent of truth except in explicitly named paired views;
- infeasible views are reported, not substituted;
- deterministic replay under the same seed;
- no evaluation source enters training or checkpoint selection;
- realized group/view telemetry agrees with enumerated episodes.

### 9.3 Objective tests

- detached references receive no gradient while all prediction paths do;
- group loss is invariant to view ordering;
- duplicating an identical view does not change normalized group regret materially;
- a single-view group reproduces ordinary best-path non-regression;
- worsening one view while holding others fixed cannot lower group best-path loss;
- disabled terms are exact zero and do not change the main loss;
- group averaging prevents larger groups from receiving larger weight.

### 9.4 Checkpoint and evaluator tests

- v1 and v2 round-trip strictly through the factory;
- v1 result reproduction is unchanged;
- v2 resume rejects sampler, loss, or architecture trajectory drift;
- all four diagnostic branches use the same encoded features and manifest rows;
- aggregate and scenario runners load v2 acquisition vectors correctly;
- scenario/control paired rows share their declared query/candidate identity;
- result rows carry complete provenance and intuitive readout names.

### 9.5 Short smoke sequence

1. CPU shape/invariance tests.
2. CUDA bf16 forward/backward on synthetic grouped episodes.
3. Three optimizer steps on real training data with every active loss finite.
4. Deliberate tiny-batch overfit where different views favor different branches; final output must
   reduce regret without either branch collapsing.
5. Deterministic validation/checkpoint round-trip and one-cell aggregate/scenario evaluator smoke.
6. Profile 50 warm steps before choosing production groups per step. No full run before these pass.

## 10. Ablation and promotion sequence

Use the same initialization, training corpus, step budget, seed set, validation panel, and evaluation
manifests:

1. stable status-quo branches and normalized mixture, ordinary episodes;
2. add evidence injection;
3. add branch-preservation loss;
4. add counterfactual grouping and grouped best-path non-regression.

The minimal required publication comparison is current promoted classifier versus complete v2.
The ladder is diagnostic and can use short screening runs before one full matched run.

Promote v2 only if all hold:

- scenario-balanced development macro-F1 improves over the current promoted classifier;
- enrolled high-k final performance does not materially regress from refined support;
- zero-support performance does not regress from semantic status;
- positive regret and harmful-overturn rate fall across multiple datasets/scenarios, not only mean;
- gains survive label-held-out and source-held-out development;
- no sealed result influenced checkpoint or hyperparameter selection.

## 11. Recommended commit sequence

1. `model: add stable evidence paths and v2 checkpoint schema`
2. `model: inject status evidence and add normalized candidate mixture`
3. `train: add counterfactual episode groups and row reuse`
4. `train: add grouped branch and non-regression objectives`
5. `train: add scenario-balanced validation and telemetry`
6. `eval: add v2 diagnostic readouts and paired scenario metrics`
7. `tests: complete v2 acceptance coverage`
8. `docs: promote new classifier contract after smoke acceptance`

Each commit must leave v1 loadable and the focused test subset green. Do not rewrite historical
artifacts or start a long training run during implementation.
