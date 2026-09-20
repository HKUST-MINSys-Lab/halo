# Primitive-driven semantic alignment: design, label-side audit, and implementation

Date: 2026-09-20. Status: **built, off by default, not trained.** Extends
[the v4 classifier design](2026-09-20-classifier-v4-evidence-gated-design.md) at exactly one seam,
its semantic branch. Section 12 of that entry recorded this as "discussed, not agreed"; this entry
supersedes that section.

## 1. The problem

The promoted semantic path is `cos(p_text(pooled_sensor), SBERT(label))`: one linear map, trained
on 155 labels. It reaches an unseen label only when that label's embedding happens to sit near a
training label's. MM-Fit's ten gym exercises score **6.6** macro F1 at `k=0`, below the 10.0 of
chance, while one example per class lifts the same encoder to 46.3 — the movements are represented,
the language interface is not.

## 2. The design

A fixed vocabulary of **32 primitives in 10 mutually exclusive axes** — 6 global (intensity,
rhythm, impact, travel, posture, regularity) and 4 regional (upper limbs, lower limbs, trunk,
head), each regional axis carrying the same three-value menu (still / rhythmic / irregular). Every
primitive has one sentence. The sensor side learns keys that decompose the pooled recording vector
into one distribution per axis; the label side is a *fixed function of the label string*; the score
is their agreement.

Three properties are deliberate:

* **Resolution matches sparse sensing.** Four regions, not seven; three motion states, not five.
  A wrist cannot testify to elbow-versus-shoulder, and nobody wears a motion-capture suit daily.
  The vocabulary describes coordinated whole-body patterns a few sensors can infer, and stops there.
* **Observability is implied, never declared.** Agreement is a per-axis dot product averaged over
  axes, so a *uniform* block contributes the same constant to every candidate and cancels in the
  softmax. A head that cannot see the head region from a wrist may output a uniform block and pay
  nothing; a confident wrong block costs. There is no mask table and nothing learned to switch axes
  off. Asserted by `test_uniform_axis_is_neutral_across_candidates`.
* **Whole-body vocabulary, any device set.** The head runs after the recording pool, so an
  arbitrary number of devices and any modality subset produce the same 32 numbers.

## 3. The label-side audit, and what it changed

Run before any training, on label text alone (`results/tools/primitive_vocabulary_audit.py`).
Per-axis argmax of `cos(label, primitive sentence)` against hand-written physical expectations,
5 labels x 10 axes, chance about 30%:

| label side | agreement |
|---|---:|
| abstract sentences, bare label | 52% |
| abstract sentences, templated label ("A person is walking.") | 54% |
| concrete embodied sentences, no activity named | 50% |
| sentences naming exemplar activities ("hard landings, such as jumping, running or stomping") | **78%** |

**Sentence similarity is not attribute entailment.** SBERT cannot judge that "sitting" implies "no
jolts"; it judges that "sitting" resembles "sitting quietly". Essentially all of the usable signal
is label-to-label similarity, which is why only the exemplar variant scores well.

That rules the exemplar variant out: naming activities would write our evaluation vocabulary into a
supposedly fixed artefact, and any MM-Fit gain would partly be an artefact of word choice.
`test_vocabulary_names_no_activity_from_our_corpora` enforces this, and it caught two real
instances during implementation ("at the pace of walking", "lying down with the body horizontal"),
both rewritten.

So the fixed cosine is kept as an **initialisation of a learnable compatibility function** (the
DeViSE/ALE family) rather than as the truth. `combiner="projection"` learns one shared linear map
applied identically to primitive sentences and to candidate labels.

Why this is not the collapse we spent a week diagnosing:

* the projection is **shared** — whatever it does to a training label it does to an unseen one, so
  it cannot single out a vocabulary. A per-primitive free value vector could, which is why the
  value bank is a frozen buffer and never a parameter (`test_value_bank_is_frozen`);
* it is **identity-initialised**, so step 0 is exactly the fixed design;
* the primitive bottleneck and its uniform-block neutrality survive;
* combining with the text path is a **fixed equal-weight sum in log space**, never a learned
  weight. A learned weight between two label-derived paths is the router we removed
  (`test_semantic_combination_has_no_learned_weight`).

Label-side state at the frozen `primitives-v1` (hash `cab859a68e89012d`), temperature 0.05:
mean normalised profile entropy 0.788, 4 near-identical label pairs in 14,706, and nearest
neighbours that are right where it matters (cycling→biking, running→jogging, walking→walking
downstairs). Per-axis argmaxes remain noisy: that noise is what the projection exists to repair,
and whether it does is the experiment.

## 4. What was built

| file | contents |
|---|---|
| `model/support/primitive_semantics.py` | the vocabulary, its hash and scrambled control, `PrimitiveSemanticHead` |
| `model/support/evidence_gated_classifier.py` | `semantic_mode` ∈ {`text`, `primitives`, `text+primitives`} plus four primitive config fields; `semantic_logits`; `branch_logits` gains `semantic_text` and `semantic_primitives` |
| `training/support_classifier/train.py` | `--semantic-mode`, `--primitive-vocabulary`, `--primitive-combiner`, `--primitive-projection-rank`; entropy, drift and per-half branch-accuracy telemetry; resume and checkpoint provenance |
| `training/support_classifier/{sealed_eval,run_scenarios}.py` | `halo-classifier-semantic-text-only` and `halo-classifier-semantic-primitives-only`, emitted at `k=0` as well as `k>0` |
| `tests/test_primitive_semantics.py` | 17 contract tests |
| `results/tools/primitive_vocabulary_audit.py` | the label-side audit above |

**No new plumbing was needed.** The label side is `candidate_text`, which v4 already receives, so
the primitive head takes no new input, the evaluators pass nothing extra, and label-text corruption
permutes the primitive profiles with the text automatically — the two semantic halves can never
disagree about which roster they are scoring.

`semantic_mode="text"` is the default and constructs no primitive head at all, so a v4 checkpoint
trained today is unchanged. The primitive head adds 148,512 parameters, almost all of them the
384x384 projection.

## 5. Verification

* Full suite **1046 passed, 1 skipped**.
* Real-data CUDA training smoke, `--semantic-mode text+primitives --text-corruption-probability
  0.5`: 6 steps, two validations, checkpoint written, all telemetry present, the frozen value bank
  round-trips inside the checkpoint.
* Sealed smoke (`k=0,1`, 8 s, 13 cells) and scenario smoke (MM-Fit, `k=0,1`): every readout emits,
  including both semantic halves at `k=0`, where the blend equals the semantic branch and the
  attribution question is which half produced the answer.
* One bug found and fixed by the smoke: agreement sums over 10 axes, so it lives in [0, 10], and
  multiplying by `logit_scale=10` spanned 100 logits and made the path confidently wrong at
  initialisation (zero-support loss 9.8 against `log(155)=5.0`). Agreement is now averaged over
  axes, which also makes `logit_scale` mean the same thing across vocabulary versions.

## 6. The evaluation programme

Four arms, one recipe, one seed, identical manifests: **v4-text** (control), **v4-primitives**,
**v4-text+primitives**, and **v4-primitives-scrambled** (`--primitive-vocabulary
primitives-v1-scrambled`). The scrambled arm is not optional: if it matches the grounded arm, the
gain came from capacity, not from meaning, exactly as the scrambled-vocabulary control once
exposed the old mixer's gain as semantic rather than structural.

Primary cells: MM-Fit `k=0` (6.6 today, chance 10.0); sealed `k=0` across the six datasets, which
must not regress; cross-placement `k=0` (42.2); partial-coverage truth-unenrolled accuracy (53.9).

3k screens before the arms: per-primitive AUC on held-out subjects; **axis entropy versus device
set**, where a region no device can see should sit near maximal entropy — that is the implied mask,
and it is measurable from the telemetry added here; and identity-versus-learned projection on
MM-Fit, which measures whether the compatibility is memorising the 155 training profiles.

## 7. Risks, stated plainly

* **Fixed equal-weight fusion can dilute a strong path with a weak one.** Released-baseline fusion
  scores 40.4 against its own 1-NN at 66.9 for exactly this reason. `text+primitives` may therefore
  land below `text`. That is why all three modes are separate arms, and why the combination rule is
  declared now rather than tuned later.
* **The projection can overfit the 155 training profiles** even though it cannot see label
  identity. The identity-versus-learned control is the gate.
* **MM-Fit is two cells.** If the unseen-vocabulary claim is to rest on more, a second new-domain
  source must be predeclared *before* any primitive result exists. That is a roster decision and is
  not taken here.
