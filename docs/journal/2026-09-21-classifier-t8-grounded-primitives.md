# T8: the primitive path with a grounded label side

Date: 2026-09-21. Status: **built, smoke-tested, not trained.** T8 is a recipe on
`support_classifier_v4`: the promoted v4 recipe plus the primitive semantic branch, with the two
repairs the [T7 post-mortem](2026-09-21-v4-promotion-and-primitive-path-postmortem.md) ranked as
able to deliver the design intent.

## What T7 got wrong, in one line each

The label side was SBERT similarity to primitive sentences — about half random per axis — so the
"common ground" had knowledge on the sensor side only; the learnable projection could repair that
only for training labels, and did so by memorising them; and the log-space product of the two
semantic halves let a confidently wrong half veto the other.

## The two changes

**1. The label side is written down, and unseen labels inherit from it.**
`model/support/primitive_annotations.py` gives every one of the 155 supervised training labels one
value per axis of `primitives-v1` — and only those labels; a test asserts the key set equals the
training vocabulary exactly, and the text is hashed. The head (`label_side="annotated"`) carries
the frozen text embeddings of those labels and their profiles as buffers, and an unseen label's
profile is the similarity-weighted average of the anchors' profiles at a fixed temperature
(`0.05`, chosen a priori from the screen below). A convex combination of per-axis distributions is
itself one, so nothing is renormalised, and the weights are auditable: which annotated labels an
unseen label inherited from is a softmax over the anchors.

Why this generalises where T7 could not: "burpee" no longer has to sit near one *correct*
training label; it only has to sit near labels that *share its attributes* — jumping, squats,
push-ups — and their averaged profile is still "vigorous, hard impacts, posture changing".
Averaging attributes is meaningful where averaging identities is not. The primitive sentences
leave the scoring path entirely, which also retires the exemplar-bias concern.

The learnable repair is now a 32×32 identity-initialised bilinear form in primitive space
(`combiner="bilinear"`, 1,024 parameters) instead of a 384×384 projection in text space. The
primitive head is 5,152 parameters against T7's 151,584.

**2. The semantic halves combine as an OR, floored.**
`semantic_combination="mixture"`: each half's probabilities are floored at `1e-3`, averaged with
equal fixed weights, renormalised. The branch can never fall below half the better half, and no
half can be confident beyond the floor. Asserted by `test_mixture_never_falls_below_half_the_
better_half`. T7's log-space product remains available as `log_sum` and is the default, so T7
checkpoints are unchanged.

## The pre-registered screen (Step 6.1) — run, on CPU, before anything else

`results/tools/primitive_label_side_screen.py`: 30 annotated labels held out of the fit at random
(seed 0), the map fitted on the other 125, per-axis argmax of the held-out labels' predicted
profiles scored against their written annotations.

| label side | per-axis agreement on 30 held-out labels |
|---|---:|
| **annotated, T = 0.05** | **0.82** |
| annotated, T = 0.02 / 0.1 / 0.2 | 0.82 / 0.78 / 0.71 |
| sentence cosine (T7), same labels | 0.34 |
| majority value among anchors (chance) | 0.57 |

The T7 label side scores close to chance on labels it has not seen; the annotated one reaches
0.82. Inheritance is legible: `disinfecting_hands ← washing_hands (0.96)`, `drawing ← writing
(0.60)`, `cleaning_door ← window_cleaning (0.47)`.

Caveat, stated plainly: held-out *training* labels have near-duplicates among the anchors
(`clapping ← frontal_hand_claps 0.99`), so 0.82 is an upper bound. A genuinely foreign label will
have weaker neighbours. MM-Fit is the real test, and it is still one dataset with five subjects.

## Verification

* Full suite 1072 passed, 1 skipped; 14 new contract tests (annotation coverage and validity, the
  scrambled control, one-distribution-per-axis, own-profile recovery, convexity of inherited
  profiles, combiner/label-side compatibility, bilinear trainability, mixture lower bound and
  floor, T7 default unchanged, gate-only isolation with the annotated side, checkpoint round trip,
  try-name resolution to `T8`).
* Six-step CUDA smoke with the full recipe: finite losses, primitive and corrupted-view telemetry
  present, anchors and profiles round-trip in the checkpoint, `classifier_try_name` returns `T8`.
* Sealed-evaluator smoke (k = 0, 1; 8 s): the checkpoint loads with the new config fields and
  every readout emits, including both semantic halves at k = 0.

## Recipe

```
halo-train --classifier evidence_gated --semantic-mode text+primitives \
           --primitive-label-side annotated --semantic-combination mixture \
           --steps 40000 --loader-workers 16 --no-compile-transformer --encode-chunk-rows 256
```

(`--text-corruption-mode auxiliary` and `--unenrolled-calibration` are the v4 defaults;
`--primitive-combiner` defaults to `bilinear` for the annotated side.) Primary checkpoint `last.pt`
at step 40,000. The scrambled control is `--primitive-annotations annotations-v1-scrambled`.

## Predictions, registered before the run

Against v4 (T6) on identical manifests:

1. **MM-Fit at k ≥ 1 does not regress** — the T7 failure. Under the mixture the semantic branch
   cannot fall below half its better half, so a wrong primitive half costs at most a bounded
   amount; the classifier should stay within 2 points of v4's 61.0 at k = 8.
2. **MM-Fit at k = 0 improves on v4's 10.5**, and the primitive half alone is *above* chance there
   (T7's was 3.7 against 10.0). This is the claim the design exists for.
3. **Sealed k = 0 at least matches T7's 51.9**, i.e. the seen-vocabulary complementarity survives
   the change of label side.
4. **The scrambled-annotation control is worse than the grounded arm on MM-Fit at k = 0.** If it
   is not, the gain is capacity, not grounding, and the primitive path should be shelved.

Miss 1 and the combination rule is still wrong; miss 2 and the label side still carries no
transferable knowledge; miss 4 and it was never about primitives.

## Not done

The sensor-side AUC screen (does `impact/hard` fire on jumping recordings?) needs a trained
checkpoint and is the first thing to run on T8's 3k-step telemetry, before the sealed evaluation is
read. A second foreign-vocabulary evaluation source remains predeclared but not added.
