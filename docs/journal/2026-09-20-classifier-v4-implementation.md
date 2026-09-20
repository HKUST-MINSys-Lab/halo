# Classifier v4 implementation notes, and four deviations from the design entry

Date: 2026-09-20. Implements
[2026-09-20-classifier-v4-evidence-gated-design.md](2026-09-20-classifier-v4-evidence-gated-design.md),
which remains the design of record. This entry records what was built, where the implementation
had to deviate from that design, and what has *not* been built. **Nothing has been trained beyond a
six-step smoke run.**

## What was built

| file | contents |
|---|---|
| `model/support/evidence_gated_classifier.py` | `support_classifier_v4`: closed-form vote and text log-softmax, 11 per-support and 13 per-candidate statistics, two MLPs (`t_j`, `lambda_c`), the bounded blend, `branch_logits` |
| `model/support/factory.py` | v4 registered; construction from a checkpoint; `EVIDENCE_GATED_READOUTS`; **v2 moved to `abandoned-negative-result`** and v4 is now the active experiment |
| `training/support_classifier/train.py` | `--classifier evidence_gated`, `--text-corruption-probability`, `corrupt_candidate_text`, dispatch, checkpoint version, resume, lambda/trust telemetry |
| `training/support_classifier/sealed_eval.py` | `_halo_evidence_gated_predictions`, dispatch, and the three branch rows per enrolled cell |
| `training/support_classifier/run_scenarios.py` | v4 routing, and opt-in branch readouts (no oracle row) |
| `tests/test_evidence_gated_classifier.py` | 21 tests: structure, bounds, the three acceptance tests, invariances, NaN safety, checkpoint round-trip, autocast, and the corruption curriculum |

The head is **50,615 parameters** against v3's 1,728,404, because it has no attention stack at all;
`p_text` (128x384) is most of it. The two gates are about 1.5k parameters together.

## Deviations from the design entry

**1. Step-0 parity is near, not exact — and the reason matters.** The design asked for zero-init
output layers and bit-for-bit parity with the closed-form vote. Zero-init is wrong here: the trust
head's output *bias* is shift-invariant (a constant added to every support cancels in the softmax
over supports), so with a zero output weight the trust MLP would receive **no gradient at all,
ever**, not merely at step 0. Both output layers are therefore initialised at std 1e-3, which keeps
step 0 within a few 1e-3 of the closed-form vote and leaves every parameter trainable from step
one. Exact equality is asserted separately under `trust_override=0`. The acceptance test was
rewritten accordingly, and the nonzero-gradient test is what caught this.

**2. Text corruption applies only to truth-enrolled episodes.** The design said "episodes with
k >= 1". That is wrong for truth-unenrolled queries: with no support for the true candidate its
blend weight is forced to 1, its logit is pure (now scrambled) text, and no parameter the head
controls can reduce that loss. Those rows are pure noise, so eligibility is now "the ground-truth
candidate has support". Zero-support episodes were already excluded.

**3. The CLI mode is `--classifier evidence_gated`, not `--classifier residual` plus flags.** The
design said "residual lineage", meaning the additive, bounded discipline; a separate mode keeps the
promoted v3 default untouched and makes checkpoints unambiguous. `--no-residual` maps to
`trust_enabled=False` and `--no-text-term` to `text_term_enabled=False`, reusing the v3 flag names.

**4. `log C` is a feature, not folded away.** The design's candidate table listed it; an earlier
draft folded it into coverage. It is a 13th channel, so margin and entropy are comparable across
roster sizes.

## Verification

* Full suite: **1028 passed, 1 skipped**. One pre-existing test (`test_evidence_aware_classifier.
  py::test_every_learned_checkpoint_architecture_has_an_explicit_lifecycle`) asserted that v2 is the
  active experiment; it now asserts v2 is abandoned and that whatever is active carries the
  `active-experimental` status. That is a fact our own 2026-09-19 result changed.
* Real-data training smoke on CUDA: 6 steps, two validations, `--text-corruption-probability 1.0`,
  checkpoint written, finite losses, classifier gradient norms 0.7-2.7 and encoder 4.6-26.7.
  Realised corruption fraction equals `1 - zero_shot_rate` exactly, as designed.
* Sealed-evaluator smoke with the smoke checkpoint (k=1, 8 s, 13 cells): all four readouts emitted.
  The trust-weighted vote equals the untrusted vote to the displayed precision (trust is near zero
  at init), the vote tracks 1-NN (55.0 against 56.0), and the blend sits between the vote and the
  untrained label-meaning branch (43.6 between 55.0 and 14.4).
* Scenario smoke (partial coverage, k=1, opt-in branch readouts): 105 rows, zero failures. The
  support vote scores 0.0 on truth-unenrolled queries by construction while the blend scores above
  zero, which is the partial-coverage capability the design is aimed at.
* Probes: zero-width support (lambda forced to 1, logits equal the text path); the rank feature
  matches a brute-force reference; the "best support of another candidate" feature is exact;
  forward passes are deterministic; CPU and CUDA agree to 1.4e-6; lambda falls 0.55 -> 0.30 as k
  goes 1 -> 8 and stays at 1.0 for unenrolled candidates.

## Not built

* **The label-held-out diagnostic panel.** The design asks for lambda plotted against whether text
  was actually right, on labels held out of *selection only*. The existing open-vocabulary holdout
  removes labels from the optimiser, which is a different and more costly thing, so that screen
  currently has no honest substrate. The other three 3k-screen manipulations (k, corruption,
  cross-placement supports) are measurable today from the telemetry added here.
* The 3k screens themselves, and the v3 text-only / base scenario readouts that are step 1 of the
  design's sequencing.
* Any training run. The arm is not launched.
