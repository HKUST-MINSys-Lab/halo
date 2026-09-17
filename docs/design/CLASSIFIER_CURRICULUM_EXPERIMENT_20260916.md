# Classifier Curriculum Experiment - 2026-09-16

**Status:** approved staged experiment. This document records the hypotheses, implementation
order, controls, and reporting contract. It supplements `SUPPORT_CLASSIFIER_DESIGN_20260914.md`;
it does not redefine the sealed evaluation protocol.

> **Protocol update (2026-09-17):** active comparisons use the seven-scenario v4 roster. Cold-start
> rows below are retained only as historical v3 measurements and are not promoted.

## Question

Can deployment-shaped training make HALO's learned classifier improve on its own neighbour floor
when enrollment is incomplete or query and support acquisition conditions differ?

The 2026-09-16 scenario evaluation found the largest classifier gains under partial enrollment,
cross-placement support, and cross-dataset support. It also found regressions when strong enrolled
sensor evidence was overridden by the semantic term. This experiment targets those two behaviors.

## Fixed controls

- Fixed multiresolution filterbank encoder: 0.5, 1, 2, and 4 second patches in 8 second windows.
- End-to-end encoder and residual-classifier training for 40,000 optimizer steps.
- The current eight-source supervised corpus, subject-held-out validation split, seed, optimizer,
  candidate policy, and multi-device policy.
- The same seven-scenario evaluator, at `k in {1, 8}`, with immutable episode manifests and the
  same HALO 1-NN, prototype, ridge, and learned-classifier readouts.
- Checkpoint selection uses internal validation only. Scenario results never select a checkpoint.

## Stage A - curriculum changes 1 and 2

### Acquisition challenge mixture

Each support set requests one acquisition condition:

| condition | target share | contract |
|---|---:|---|
| compatible | 0.50 | query and support acquisition keys are identical |
| cross placement | 0.25 | same dataset, device family, channels, and gravity convention; different anatomical site group |
| cross dataset | 0.25 | identical acquisition key, but every support execution comes from another dataset |

Only real recordings form mismatched pairs. If a dataset cannot form a requested condition, the
sampler draws from the feasible conditions using the declared weights and reports both requested
and realized shares. It never duplicates executions or relaxes subject/execution separation.

These are requested shares, not quotas. Corpus feasibility determines the realized mixture and the
logged realized shares are authoritative. Cross-placement excludes left/right/unspecified variants
of one anatomical site and never substitutes another dataset. Cross-dataset coverage is limited to
the acquisition keys genuinely shared by the eight-source corpus.

### Enrollment mixture

Enrollment is sampled before the query's true label is inspected:

| condition | target share | contract |
|---|---:|---|
| complete | 0.50 | every candidate has enrolled support |
| partial | 0.25 | an independently selected candidate subset has support |
| zero | 0.25 | no candidate has support |

One support set and one enrolled-candidate subset are shared by its query executions. In partial
episodes, coverage is sampled between 25% and 75%, with at least one enrolled and one unenrolled
candidate. Support counts come from `{1, 2, 4, 8, 16, 32}` where distinct executions permit them.
Half of enrolled support sets use unequal per-candidate counts obtained by downsampling an honestly
drawn execution-distinct roster. Dataset-first query sampling remains in force so large datasets
do not monopolize high-k training.

## Stage B - curriculum change 3 and model change 4 (postponed)

Stage B starts from the same initialization and repeats the complete 40,000-step run with Stage A
plus the following changes. It is not a continuation of Stage A.

On 2026-09-16 this ablation was stopped around step 20,000 so reporting could prioritize the
completed Stage A results and matched neighbour controls. No Stage B checkpoint is promoted into
the primary results. Its partial run remains a diagnostic artifact only.

The stopped Stage B run predates the 2026-09-16 sampler corrections and is not a valid matched
ablation of the current curriculum. A new comparison must train both arms from scratch on the
current sampler and score both on `deployment-scenarios-v4-20260917` manifests.

The command line does not infer an experiment stage. A Stage B run must explicitly include
`--rate-augmentation-probability 0.25 --modality-dropout-probability 0.20
--adaptive-text-gate`; omitting them is Stage A. Persisted `run_config.json`, trajectory fields,
and telemetry are the authority, not an output-directory name.

### Acquisition perturbations

- Anti-aliased rate resampling is applied independently to recording occurrences with probability
  0.25. Half of applicable draws are sampled at or below the hardware rate to include bandwidth
  loss; the other half retains the full 15--100 Hz range. Physical duration, achieved rate,
  source-rate observability, and metadata remain truthful. Training and evaluation use the same
  line-edge polyphase padding.
- Complete gyroscope removal is applied independently with probability 0.20 where a gyroscope is
  present. Channel masks and acquisition descriptions are updated by the existing augmentation
  implementation.
- Clean examples remain the majority. Neither perturbation depends on the query label.

### Evidence-dependent semantic gate

Replace the support-count lookup `lambda(k)` with a small shared gate evaluated per candidate. Its
inputs are deterministic evidence statistics:

- log support count;
- strongest and mean query-support cosine;
- within-candidate support spread;
- margin over the strongest competing candidate;
- disagreement between sensor-vote and text scores.

The gate returns a non-negative semantic weight. A zero-support candidate keeps a separately
learned zero-support prior. The enrolled gate is initialized to reproduce the previous support-count
weights, and the residual heads remain identity-initialized. Telemetry records gate values by
support count, sensor margin, and enrollment regime.

## Decision criteria

Compare Stage A and Stage B with the 2026-09-16 reference checkpoint at identical scenario cells.
Report macro-F1 and paired deltas for:

- complete versus partial enrollment;
- truth enrolled versus truth unenrolled;
- compatible, cross-placement, and cross-dataset support;
- missing modality, sampling-rate mismatch, device-set mismatch, new domain, and cold start;
- HALO classifier versus HALO 1-NN and ridge on the same encoder.

A change is useful only if gains are broad across cells and do not come from hiding a material
regression in the ordinary complete-enrollment condition. Stage B should specifically reduce
correct neighbour decisions overturned by text while retaining text-driven rescues when the truth
has no support.

## Deferred change 5

Explicit acquisition-description tokens are deferred until Stages A and B establish whether the
training challenge and evidence gate are sufficient. They require a separate correct/omitted/
shuffled-descriptor ablation and are not part of these runs.

## Differentiable-neighbour encoder control

Two `DN + 1-NN` rows are kept distinct:

- The historical checkpoint at
  `halo_fixed_mr_neighbors_8s_4res_e2e_20260913_continue95k` is the pre-curriculum reference. It
  used `k in {1, 2, 4, 8}` and no acquisition/enrollment challenge mixture.
- A curriculum-matched control is trained with the same acquisition mixture, candidate policy,
  `k in {1, 2, 4, 8, 16, 32}`, and variable support counts as the staged classifier runs.

The DN objective requires a true-class neighbour. Its matched curriculum therefore excludes
zero-support episodes and, in partial enrollment episodes, hides distractor classes while drawing
queries only from enrolled classes. This is not equivalent to the learned classifier's harder
query-independent partial/zero curriculum and must not be described as such. It isolates whether
the acquisition curriculum improves the encoder geometry that a plain 1-NN readout uses.

## Execution record

### Stage A - completed 2026-09-16

- Output: `training/support_classifier/outputs/halo_fixed_mr_residual_v3_curriculum12_40k_20260916`
- Scenario evaluation: `training/support_classifier/evaluations/scenarios_curriculum12_20260916`
- Best internal checkpoint: step 30,000, selected on subject-held-out validation only
- Best validation macro-F1: 0.760
- Validation macro-F1 rose from 0.518 at step 2,500 to 0.760 at step 30,000; it did not
  exhibit the historical early-convergence failure.

The table reports unweighted means over the perturbed scenario cells. `Delta` compares Stage A
with the fixed 2026-09-14 control checkpoint on identical scenario manifests.

| scenario | k | 1-NN delta (points) | classifier delta (points) |
|---|---:|---:|---:|
| partial enrollment | 1 | -1.1 | -1.9 |
| partial enrollment | 8 | -0.3 | -0.5 |
| cross placement | 1 | -0.1 | -3.2 |
| cross placement | 8 | -0.6 | -2.8 |
| cross dataset | 1 | +2.0 | -1.1 |
| cross dataset | 8 | +2.9 | -1.2 |
| missing modality | 1 | +0.2 | -1.3 |
| missing modality | 8 | +1.0 | +0.0 |
| sampling-rate mismatch | 1 | -1.7 | -6.6 |
| sampling-rate mismatch | 8 | -0.6 | -7.3 |
| new domain | 1 | +0.5 | -1.8 |
| new domain | 8 | +0.0 | +1.4 |
| device-set mismatch | 1 | +0.8 | -2.5 |
| device-set mismatch | 8 | -1.8 | -2.2 |
| cold start | 1 | +6.7 | +0.3 |
| cold start | 8 | +8.6 | -1.2 |

Stage A improved encoder geometry most clearly for cross-dataset and cold-start transfer, but the
learned classifier generally failed to retain those gains. Its large sampling-rate regression is
the main reason Stage B includes explicit rate perturbation. The final held-out validation draw
realized 54.7% compatible, 39.1% cross-placement, and 6.3% cross-dataset episodes because 21.9%
of requested conditions were infeasible under strict subject/execution separation. This fallback
is recorded rather than silently relaxing the episode contract.

### Stage B - implementation acceptance

Stage B adds deterministic rate/modality perturbations and the adaptive semantic gate. Its smoke
acceptance requires finite mixed-precision forward/backward, nonzero gradients in every gate
layer, deterministic perturbation replay, aligned multi-device timelines, and telemetry split by
acquisition regime, enrollment regime, truth enrollment, query/support perturbation, support
count, and sensor-evidence margin. The 3-step real-data smoke passed these checks before the full
run was launched.

## Bounded diagnostics - 2026-09-17

Two cheap diagnostics now gate any replacement 40,000-step run:

- `training/support_classifier/curriculum_audit.py` samples the active curriculum without loading
  sensor tensors. Its 2,000-support-set report is stored in
  `training/support_classifier/evaluations/curriculum_audit_20260917.{md,json}`.
- `training/support_classifier/development_panel.py` scores checkpoints on one deterministic,
  subject-held-out episode panel under clean, rate-resampled, and gyroscope-dropout recording
  conditions. It also reports exact neighbour-to-classifier rescues and harmful overturns. Its
  report is stored in
  `training/support_classifier/evaluations/development_panel_20260917.{md,json}`.

The sampler audit measured 2,000 support sets and 7,457 queries. Among enrolled sets, the realized
acquisition mixture was 74.8% compatible, 15.1% cross-placement, and 10.1% cross-dataset, rather
than the requested 50/25/25. Cross-placement was feasible for four datasets and cross-dataset for
three. The strict corrected cross-placement path had zero sets containing support from another
dataset. These are corpus feasibility limits, not sampler leakage.

The development panel is diagnostic because both evaluated checkpoints predate the corrected
sampler. On the identical clean panel, Stage A improved over its own neighbour floor by 11.1
accuracy points; the stopped Stage B checkpoint improved by 9.5. Stage B was also lower in
dataset-macro F1 under clean, rate-resampled, and gyroscope-dropout conditions. This does not
separate perturbations from the adaptive gate, but it rejects the claim that the stopped combined
run already demonstrated a robustness gain.

Before another long run, use matched short screens from the same initialization and current source:

1. corrected Stage A (neither perturbations nor adaptive gate);
2. rate/modality perturbations only;
3. adaptive gate only;
4. perturbations plus adaptive gate.

The fixed panel, correction telemetry, seeds, validation cadence, and all other hyperparameters
must be identical. The screen is only a filter: a promising arm still requires a complete run and
sealed evaluation before promotion.

The completed screen and decision are recorded in
`docs/results/2026-09-17-classifier-curriculum-screen.md`. In brief, adaptive-gate-only preserved
clean performance and improved partial, zero-support, and cross-dataset panels; perturbations-only
improved gyroscope-dropout robustness; their combination underperformed both. The combined recipe
is therefore not promoted.
