# T6: what T4 and the corruption-free control taught, put into one design

Date: 2026-09-20. Status: **built, smoke-tested, not trained.** T6 is a recipe on the
`support_classifier_v4` architecture, not a new architecture string; see "Classifier naming" in
[the results record](../results/RESULTS.md).

## What the evidence pinned down

1. **The collapse is a property of the objective, not the architecture.** T1, T2, T3 and the
   corruption-free T4 control all collapsed onto label meaning; T4 with corruption did not. The
   bound on λ contained the damage (−15 rather than −40 against its own vote); the curriculum is
   what prevented it. Both are needed.
2. **Internal validation prefers the collapsed model** (enrolled 0.787 against 0.778). No routing
   decision may be selected on that panel.
3. **v3's centred soft support vote is the best component we have.** It beats hard 1-NN by about a
   point at every k in every arm, and no learned reweighting has clearly improved the vote itself.
4. **Per-candidate λ works, but cross-entropy alone picks the wrong operating point.** Partial
   coverage truth-enrolled accuracy 66.7 → 80.8 is real; truth-unenrolled 53.9 → 44.5 is the
   price, and T4 has no label-blind term that can recalibrate the two populations. v3's
   `r_candidate` could, but it was the identity channel.
5. **Replace-mode corruption costs the encoder** about half of T4's 1.6-point cosine-1-NN gap,
   because a corrupted episode displaces a clean one and its semantic gradient with it.
6. **v3 is strictly more expressive than T4.** T4's contribution is separating per-candidate
   adaptivity from label-identity access, not capacity.

## The two changes

**Corruption as a gate-only auxiliary, not a data replacement.** Every truth-enrolled episode
contributes its clean loss, which trains everything, plus a second loss on the same episode with a
deranged roster, computed with `gate_only=True`: encoder features, trust, the semantic branch and
the calibration term are all detached, so that loss can reach exactly the blend gate (`gate_mlp`,
`lambda_prior`). Asserted by `test_gate_only_view_reaches_exactly_the_blend_gate`.

Three consequences. The representation loses nothing, which should recover the curriculum's half
of the encoder gap. Corruption runs on 100% of eligible episodes rather than 25%, so routing gets
four times the signal. And the design principle becomes literal: *representation trains on clean
data; routing trains on the distribution where meaning is unreliable.* Those were always two jobs,
and T4 paid for doing them with one loss.

**A label-blind calibration term for unenrolled candidates.** A candidate with no support of its
own receives `text_logit + b(coverage, log |roster|)`, where `b` is a 69-parameter MLP. It cannot
see which label it is; it can only learn how much to favour "the answer is something nobody
enrolled" as a function of how much of the roster is enrolled. It is identical for every unenrolled
candidate of an episode (so it cannot prefer a label), trains on clean views only (a corrupted view
has an enrolled truth by construction and would teach it the wrong lesson), and is off by default.
Asserted by `test_unenrolled_calibration_is_off_by_default_and_label_blind`.

**Unchanged from T4:** the centred soft vote as the floor, the exact-floor residual trick, the
bounded label-blind trust and λ, the convex blend with `λ_max = 0.85`, no attention, no learned
weight between semantic halves.

## The collapse detector

`corrupted_view` doubles as a probe: accuracy on a deranged roster at k ≥ 1, reported as
`curriculum/corrupted_view_accuracy` in every training log step and, with a fixed derangement,
in every validation. A head that has stopped looking at support evidence scores near zero here; a
disciplined one keeps its support accuracy. It is label-blind by construction and would have
flagged T1, T2, T3 and the corruption-free control within the first few thousand steps — the
runs internal validation rated most highly.

## Recipe

```
halo-train --classifier evidence_gated --text-corruption-mode auxiliary --unenrolled-calibration \
           --steps 40000 --loader-workers 16 --no-compile-transformer --encode-chunk-rows 256
```

`--text-corruption-probability` defaults to 1.0 in auxiliary mode; `--text-corruption-aux-weight`
defaults to 1.0. Primary checkpoint `last.pt` at step 40,000, as for every arm since 2026-09-18.

## Verification

* Full suite 1055 passed, 1 skipped; 3 new contract tests.
* Real-data CUDA smoke, 6 steps with both mechanisms on: finite losses, all telemetry present in
  training and validation, checkpoint written with `unenrolled_calibration=True` and
  `text_corruption_mode=auxiliary` recorded.
* Sealed-evaluator smoke (k=1, 8 s, 13 cells): the checkpoint loads through the factory with the
  new config field and every T4 readout emits.

## Predictions, stated before the run

* Encoder cosine 1-NN at 8 s, k=8: back near 72 (T4 70.4, corruption-free 71.1, v3 72.0).
* λ ordered by evidence with no candidate at the bound, as in T4; corrupted-view accuracy well
  above zero throughout training.
* Partial coverage at k=8: truth-enrolled near 80 *and* truth-unenrolled back above 50, which
  neither T4 nor v3 achieved.
* Deficit against its own support vote at k=128: no worse than T4's −1.9.

If the encoder does not recover, the remaining gap is not the curriculum and the label-blind
attention ablation (v3's set-attention stack over query and support embeddings, feeding trust
only, never λ) is the next probe.

## Sequencing still recommended

1. **v3 + corruption** (replace mode) — cheapest, follows directly from the observation that v3
   is more expressive than T4. Prediction: reduces v3's high-k over-trust but cannot move partial
   coverage, because its λ is per-bucket.
2. **T6** as above.
