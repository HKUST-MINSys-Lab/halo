# v4 promoted (the T6 recipe), and why the primitive semantic path failed on foreign vocabulary

Date: 2026-09-21. Two records in one entry: the promotion decision, and the post-mortem of T7.

## Promotion

**T6 is promoted and is called `v4` from today.** It is the `support_classifier_v4` architecture in
its T6 recipe — corruption as a gate-only auxiliary on every eligible episode, plus the label-blind
unenrolled calibration term, with the promoted cosine semantic path — and that recipe is now the
trainer's default for `--classifier evidence_gated`, which is itself now the default classifier.
`v3` (`support_classifier_v3`, promoted 2026-09-18) is superseded. The T4 recipe stays reachable
with `--text-corruption-mode replace --no-unenrolled-calibration`.

Basis, all on identical manifests ([record](../results/RESULTS.md)): v4 beats v3 by +1.5 to +2.4
dataset-balanced macro F1 at every k ≥ 4 on the sealed aggregate, matches it at k = 1–2, trails by
1.5 at k = 0; its deficit against its own support vote is −0.9 at k = 128 against v3's −3.6; it
leads v3 on the new-domain, device-set and missing-modality scenarios at every k, and trails on
cross placement at k ≥ 1 by at most 1.6. Its head is 50,684 parameters against v3's 1,728,404.

Registry: `PROMOTED_CLASSIFIER_ARCHITECTURE`, `PROMOTED_RECIPE` and `PROMOTED_CLASSIFIER_NAME` in
`model/support/factory.py`; `classifier_try_name` returns `v4` for the promoted recipe and `T7`
for the primitive-branch recipe on the same architecture.

## Why the primitive semantic path did not deliver its design intent

The intent was specific: language alignment alone cannot reach labels outside the training
vocabulary, so a fixed vocabulary of movement primitives would be the *common ground* — something
both a recording and an unseen label could be expressed in. T7 (v4 plus that branch) is the best
sealed arm produced so far, and it fails on exactly the condition the branch was built for.

### What the numbers say

| | seen vocabulary (sealed, 8 s, k=0) | foreign vocabulary (MM-Fit, k=8) |
|---|---:|---:|
| support vote | – | 47.5 |
| semantic, text half | 50.6 | 10.2 |
| semantic, primitive half | 42.9 | **3.7** (chance 10.0) |
| full classifier | **51.9** | **26.8** |

On seen vocabulary the halves are complementary: the combination beats both. On foreign vocabulary
the primitive half is not uninformative, it is *confidently wrong* — below chance — and the
classifier lands twenty points under its own support vote.

### Five causes, in order of importance

**1. The label side was never grounded. It was SBERT similarity renamed.** A label's primitive
profile was `softmax_axis(cos(SBERT(label), SBERT(primitive sentence)))`. The label-side audit run
*before* training measured that at 52 % per-axis agreement with physical expectation, against
roughly 30 % chance; sentence similarity is not attribute entailment, and SBERT has no way to
judge that "sitting" implies "no jolts". So "burpee" received noise. The sensor side genuinely
learned to decompose recordings (primitive entropy fell from 0.94 to 0.58), but the bridge had one
grounded half. A label side that is *consistent but wrong* supports transfer only to labels that
sit near training labels in SBERT space — which is precisely the generalisation the text path
already had. It contributes no new knowledge; on foreign labels it contributes noise.

**2. The learnable projection could only repair training labels, and did so by memorising them.**
The shared 384×384 compatibility map is trained by classification cross-entropy on 155 labels. What
it learns is to make the decomposition of a walking recording match whatever noise-profile the
label side assigned to "walking". The primitives thereby become a 32-dimensional re-encoding of
training-label identity: a lookup wearing a vocabulary. That is why the primitive half alone scores
42.9 on sealed labels (which are near training labels) and 3.7 on MM-Fit. Below chance is the
fingerprint of a sharpened, memorised metric applied out of distribution — it does not merely fail
to match, it confidently mismatches.

**3. Log-space equal-weight fusion made a wrong half a veto.** `log_softmax(text + primitives)` is
a product of two distributions: an AND. A half that assigns near-zero probability to the true label
cannot be outvoted by the other half, because log-probabilities have unbounded negative range. So
when the primitive half was confidently wrong, the entire semantic branch was — and at k = 8, with
λ around 0.2, that still flipped the argmax. The combination rule turned "primitives useless here"
into "primitives destructive here". A probability-space mixture (an OR) cannot be dragged below
half the better half; it would have contained this.

**4. Nothing anywhere in training could see "foreign".** λ is label-blind by design, and that
design is correct: it can learn to distrust meaning *when evidence is strong*, never *because the
vocabulary is unfamiliar*. The corruption curriculum manufactures wrong text, but the wrongness it
teaches is a random derangement, not a coherent foreign vocabulary that scores confidently. There
was therefore no mechanism in the pipeline that could dial the primitive half down where it fails.

**5. The wrong side was validated, and the pre-registered screens were skipped.** The audit checked
the label side against hand-written expectations; nobody checked the sensor side against physical
truth (does `impact/hard` fire on jumping recordings?). The design entry called for per-primitive
AUC on held-out subjects and for the scrambled-vocabulary control before any 40k arm, and both were
sequenced past. The primary metric, sealed k = 0, is dominated by seen vocabulary, so its +1.7 gain
could not distinguish learned grounding from a better lookup — the scrambled control would have.
And the suite has exactly one foreign-vocabulary source, so the failure surfaced in one cell.

### What this says about the idea

"Primitives as common ground" requires that *both* sides be expressed in primitives by something
that knows what the primitives mean. The sensor side can learn that from data. The label side
cannot learn it from 155 labels' cross-entropy — it needs knowledge injected. SBERT cosine was a
proxy for that knowledge; the audit measured it as about half random, and the run confirmed the
audit. The idea is not refuted; this implementation of its label side is.

### Repairs, ranked by cost and by what they can establish

1. **Bound the semantic branch's confidence and mix in probability space.** Cheapest; targets
   cause 3 directly. Would recover most of the MM-Fit loss. Would not make primitives *help* on
   foreign vocabulary — it stops them hurting.
2. **Run the scrambled-vocabulary control.** Decides whether the seen-vocabulary gain is grounding
   at all. If scrambled matches grounded, cause 2 is the whole story.
3. **Ground the label side with knowledge, not similarity.** Annotate the 155 training labels'
   profiles (LLM- or hand-written; training labels only, never sealed), and fit
   `SBERT(label) → profile` on them. An unseen label then inherits attributes from labels that
   *share its attributes*, not from labels that share its name — averaging attributes is
   meaningful where averaging identities is not. This is the version that could actually deliver
   the design intent, and it is the one that addresses cause 1.
4. **Run the sensor-side AUC screen** before any further arm, so the decomposition is checked
   against physical truth rather than assumed.
5. **Predeclare a second foreign-vocabulary evaluation source.** MM-Fit is one dataset, one stream,
   five subjects; a promotion decision should not rest on it, and neither should a repair.
