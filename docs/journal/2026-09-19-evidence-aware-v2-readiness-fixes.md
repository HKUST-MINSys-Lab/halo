# Evidence-aware v2 readiness fixes

Date: 2026-09-19. Supersedes the readiness verdict in
[`2026-09-19-evidence-aware-v2-readiness-audit.md`](2026-09-19-evidence-aware-v2-readiness-audit.md)
for the active enrollment-counterfactual experiment. Historical results are unchanged.

## Corrected contracts

- The best-path auxiliary now computes raw per-view regret, a normalized temperature-scaled
  group `logmeanexp`, and exactly one smooth hinge. A structurally unavailable support path is
  excluded from the detached reference. Explicit and private group identifiers cannot collide.
- Final cross-entropy averages views within each explicit counterfactual group before averaging
  independent zero-support, enrolled, and grouped families.
- Counterfactual enrollment is enabled by default only for the evidence-aware contextual
  classifier. Neighbors and historical residual heads default to ordinary episodes and reject an
  explicit nonzero counterfactual setting.
- The support status path now uses all support rows. Exact labels bind directly; genuinely
  off-roster support labels use pre-context label-to-candidate semantic compatibility. Raw cosine
  is returned separately and the learned support temperature is used consistently.
- Evidence injection and contextual pair correction use normalized, well-conditioned projections
  behind exactly-zero scalar gates. This preserves the initial status quo without normalizing a
  zero vector. The candidate router reads only contextualized query and candidate states.
- Branch preservation, grouped best-path non-regression, evidence injection, and group
  temperature are independent persisted switches. Resume checks include them and
  `episodes_per_step`; checkpoint architecture metadata is derived from the constructed class.
- Validation selection now averages deployment families instead of overlapping panels directly,
  breaks exact score/loss ties by lower positive regret, and saves `best_zero_support.pt` and
  `best_enrolled.pt` beside `best_internal.pt`.
- Training telemetry now includes realized post-expansion view shares, router quantiles and
  saturation, masked correction RMS/p95, branch accuracy, positive regret, rescue, and harm.
  Scenario evaluation emits the three auditable branches and a clearly named non-deployable
  better-branch oracle for v2.

## Numerical finding and repair

The first forced-counterfactual smoke exposed a classifier pre-clip gradient norm of approximately
`1.5e12`, followed by `1.1e7`. The cause was `F.normalize` applied to exactly zero-initialized
evidence/candidate projections. The equivalent zero-output initialization now normalizes a
well-conditioned projection and multiplies it by an exactly-zero learned scalar. In the repeated
three-step real-data smoke, classifier gradient norms were `6.45`, `7.92`, and `8.88`; encoder
norms were `37.23`, `24.04`, and `17.23`. All losses remained finite, validation completed, and
the checkpoint round-trip was strict.

## Verification

- Focused model/sampler/loader/evaluator suite: 169 tests passed before the final full-suite run.
- Real-data training smoke: three optimizer steps with counterfactual enrollment forced on every
  batch, validation at steps 0 and 3, and all three checkpoint roles written.
- Scenario smoke: scenario 1, `k=1`, 8-second windows, two datasets, 42 result rows, zero task
  failures, with all v2 diagnostic readouts present.
- CPU bfloat16, support permutation, off-roster binding, exact initial support-path equality,
  grouped-loss ordering, group-id isolation, auxiliary switches, and strict checkpoint reload all
  have regression tests.

## Deliberate scope boundary

The active grouped intervention is enrollment only. Acquisition, rate, modality, and device-set
challenges remain present as independently sampled curriculum conditions, but they are not called
same-query counterfactual groups. Implementing those honestly requires observation identities that
distinguish transformed copies of one recording; the current loader correctly deduplicates by
physical recording and must not be bypassed with duplicated aliases.

The internal panel remains subject-held-out across the eight training sources. It is not described
as source-held-out or label-held-out. A genuine source/label holdout changes the optimizer corpus
and requires a predeclared split experiment; it must not be simulated after training or selected
ad hoc. These two extensions are future experiment work, not hidden prerequisites of the repaired
enrollment-counterfactual run.
