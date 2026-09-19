# Evidence-aware v2 second-review fixes

Date: 2026-09-19. Supersedes the readiness conclusion in
[`2026-09-19-evidence-aware-v2-readiness-fixes.md`](2026-09-19-evidence-aware-v2-readiness-fixes.md).

## Verified findings

The second independent review correctly identified four result-affecting problems: expanded
counterfactual queries received excess main-loss weight; device-set planning could vary across the
four enrollment views; routing/checkpoint selection had no label-held-out signal; and the scenario
runner emitted a truth-dependent branch oracle by default. It also found real contract or hygiene
issues in off-roster support smoothing, support-floor naming, mixed-precision consistency,
programmatic neighbor guards, masked-candidate telemetry, acquisition normalization, duplicated
architecture sets, and the gradient-reach test.

The report's claim that an unbounded contextual-correction scale was a defect was not accepted. The
approved handoff explicitly rejects a manually chosen cap: normalized pair/candidate projections,
weight decay, finite checks, and global clipping are the declared safeguards. The file-system tensor
sharing strategy also remains because it prevents the measured file-descriptor exhaustion; PyTorch's
shared-memory manager owns normal cleanup.

## Repairs

- Counterfactual views now collapse to one query unit before the ordinary support-set and
  zero/enrolled balancing. A gradient-weight regression test proves that all four views jointly
  receive the same weight as the sibling query they replaced.
- Device planning groups by explicit counterfactual identity before support tuples, so enrollment
  is the only intervention across those views.
- The evidence-aware recipe uses a deterministic 20% per-source target to form a globally excluded
  canonical-label panel.
  The selected labels never enter optimizer episodes from any source. Primary checkpoint selection
  equally weights seen-label and label-held-out, subject-held-out deployment-family scores. Legacy
  recipes retain their historical corpus and selection metric.
- The scenario oracle is opt-in and its rows are machine-marked non-deployable diagnostics.
- Support aggregation now adds one uniform pseudo-support continuously and lets that prior decay
  with real support count. Off-roster semantic mass no longer switches the prior off.
- `support_status` uses a learned temperature; `support_floor`/`neighbor_logits` is the fixed 0.07
  control. Support-label binding has a separate learned temperature.
- The classifier head executes in fp32 inside CUDA autocast; the encoder remains bf16. The measured
  fp32/autocast classifier difference is exactly zero.
- Validation selection is architecture-scoped: historical arms use their enrolled dataset-macro
  rule, while evidence-aware v2 uses the declared scenario-balanced plus open-vocabulary rule.
- The default CLI classifier is the promoted residual control. Evidence-aware v2 must be selected
  explicitly with `--classifier contextual`.
- Group log-mean-exp is vectorized; partial counterfactual telemetry records withheld candidates;
  programmatic neighbor draws reject zero-support counterfactuals; query/support acquisition vectors
  are normalized symmetrically; evaluator architecture constants are shared; acquisition caches use
  weak encoder-object keys rather than recyclable memory addresses.

## Verification

A three-step real-data CUDA smoke completed end to end with the global label holdout, both internal
validation panels, finite gradients, checkpoint selection, and checkpoint writes. The smoke reserved
16 of 133 labels on its capped corpus; the full corpus deterministically reserves 25 of 155 labels,
leaving 130 optimizer labels and held-out coverage in all eight sources. Targeted evaluator tests
confirm that branch diagnostics remain explicitly selectable while the oracle is absent by default.
