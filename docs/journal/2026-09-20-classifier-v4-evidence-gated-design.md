# Classifier v4: evidence-gated blend on the promoted residual head (agreed design)

Date: 2026-09-20. Status: **implemented; no full-budget training yet.** This entry records the design discussion
that followed the evidence-aware v2 negative result
([results record](../results/RESULTS.md#the-evidence-aware-v2-follow-up)). It is the handoff for
whoever implements it. Supersedes the contextual-lineage handoff in
[2026-09-19-new-classifier-design-handoff.md](2026-09-19-new-classifier-design-handoff.md) as the
active classifier direction; that lineage (`support_contextual_mixture_v1`,
`support_contextual_residual_v1`, `support_evidence_aware_v2`) is closed.

## 1. What the record says the classifier must do

All numbers: v5 protocol, 8-second windows, identical manifests, promoted `support_classifier_v3`
("residual arm") unless stated. Situations as the project defines them:

| situation | measured state | headroom | fixable by the classifier? |
|---|---|---|---|
| zero-shot, seen vocabulary | HALO semantic 51.7 vs UniMTS native 33.1 | encoder / text alignment only | no (λ is forced to 1 at k=0) |
| zero-shot, unseen vocabulary (MM-Fit) | 6.6, below chance | needs a different semantic path | no |
| truth enrolled, complete | classifier **loses to its own closed-form base** from k≥4: 71.5 vs 73.3 (k=8), 74.4 vs 78.0 (k=128); base beats hard 1-NN by ~1 at every k | +1.8 … +3.6 | **yes** — over-trust of text at high evidence |
| truth not enrolled, negatives enrolled (partial coverage, k=8, accuracy) | classifier 66.7 enrolled / 53.9 unenrolled (hmean 59.7); 1-NN 88.0 / 0.0; chance ≈13 | enrolled half back toward 88 while keeping ~54 → hmean ≈67 | **yes** — needs a *per-candidate* trade-off; a global λ is the trade-off v3 already makes |
| adjacent, imperfect supports: cross-dataset | one support: 1-NN 69.0 < semantic 74.1, classifier 75.1; eight: 1-NN 78.0, classifier 81.0; control 85.6 | +4.6 | **yes** — λ must depend jointly on k and evidence quality |
| cross-placement / device set | all paths weak: 42–50 vs control 67; device set classifier = 1-NN at every k, control 79 | 14–20 | **no** — representation; a router cannot create information neither path has |
| missing modality | classifier ≤ 1-NN from k=8 | ~2 | yes, same fix as "truth enrolled" |

Two further facts from the v3 checkpoint (`halo_residual_stage_a_40k_20260918/last.pt`):

* v3 **already learned to lower λ with k** on its own: bucket table 1.61 / 1.03 / 0.82 / 0.68 / 0.51
  for k = 0 / 1 / 2 / 4 / 8+. Preferring support at high k is not what needs teaching.
* v3 has **three** text-driven output terms, not one: `λ·text_score`; `r_support`, produced by a
  stack that reads support-label and candidate-label tokens; and `r_candidate`, a free per-candidate
  logit from the same tokens. The last two are unbounded side channels for label text. The 3k
  adaptive-gate screen (`screen_20260917_adaptive_gate_only_3k`) collapsed its bucket biases to
  0.001 and let the gate MLP carry λ, and that MLP's feature set includes
  `disagreement = text_cosine − sensor_score`, a semantic-confidence input.

Baseline equal-weight fusion (a fixed blend with a weak semantic partner) scores 40.4 for UniMTS
against its own 1-NN at 66.9: fixed blending with an unreliable partner destroys value; adaptivity
is what makes blending worth doing.

## 2. Why the contextual lineage collapsed, in one sentence

Every training label is in-vocabulary and text embeddings separate them almost perfectly, so the
loss-minimising router on the training distribution routes everything to the semantic branch;
the collapse was the optimum of the objective, not an optimisation bug, and a free per-candidate gate
that can *delete* the support path was the capability that let it happen (v2: reliance prior stayed
at 0.006 while the per-candidate router output went to 0.97–0.99). Capacity was not the cause:
v3's head has 1.73M parameters, v2's 1.51M.

## 3. Design principle

**Label text touches the output at exactly one place, weighted by `λ_c ≤ λ_max`. Every learnable
part of the head reads evidence statistics only.** Nothing about high-k behaviour is hard-coded;
what is fixed is the *shape* (bounds and blindness), not the policy.

Order of computation (the reverse of "contextualise first, then branch"):

1. closed-form paths: the centred soft support vote and the text log-softmax, from cosines;
2. statistics derived from those paths (below);
3. two small learned gates on the statistics: per-support trust `t_j`, per-candidate `λ_c`;
4. the blend.

No attention anywhere in the first build. Context between supports and between candidates enters
only through **fixed permutation-invariant pooling** (means, maxima, ranks, margins, entropy) in
the feature vectors; there is no learned mixing between rows. Learned set interaction has bought
shortcuts every time in this project.

## 4. Mechanism 1 — per-support trust inside the vote

Replaces `r_support`.

```
t_j  = T · tanh( MLP_t(f_j) )                       T = 2.0 (fixed)
w_j  ∝ exp( cos(q, s_j) / τ + t_j )                  over valid supports
v_c  = Σ_{j∈S_c} w_j ;   metric_c = log v_c
metric = base + (corrected − zero_corrected)          keep v3's exact-floor residual trick
```

`f_j`, with `s_j = cos(q, s_j)`, `S_c` the supports of candidate `c(j)`, `S` all supports:

| feature | answers |
|---|---|
| `s_j` | absolute closeness |
| `s_j − mean(S_c)`, `s_j − max(S_c)` | outlier or anchor within its own candidate |
| `z_j = (s_j − mean(S)) / (std(S) + ε)` | closeness relative to the episode's scale |
| `s_j − max(S \ S_c)` | discriminative, or matched by a rival's support |
| `rank(s_j) / |S|` | scale-robust version of the above |
| `log1p(|S_c|)`, flag `|S_c| = 1` | sibling count; the flag stops zero spread reading as perfect agreement |
| `mean(S)`, `max(S)`, `std(S)` | is the query far from everything (mismatched condition) |

Raw and standardised forms are both kept on purpose: z-scores survive similarity-scale drift during
training; raw values keep "the query is far from every support", which standardisation erases.

## 5. Mechanism 2 — per-candidate bounded λ

Replaces the bucket-only λ and the adaptive gate. Computed **after** the trust stage, on the
reweighted vote.

```
λ_c = λ_max · sigmoid( b[bucket(k_c)] + MLP_λ(g_c) )    k_c ≥ 1 ;   λ_max = 0.85 (declared a priori)
λ_c = 1                                                 k_c = 0  (forced, as in v3)
text_c   = log_softmax( cos_text / τ_text ) over the roster
logits_c = (1 − λ_c) · metric_c + λ_c · text_c
```

Both terms are log-probabilities, so λ is a genuine blend; `λ_max` guarantees an enrolled
candidate's support keeps ≥15 % weight. `b` is initialised so that step 0 reproduces v3's learned
table on the new scale.

`g_c`:

| feature | answers |
|---|---|
| `log v_c` | the support path's opinion |
| `log v_c − log v_runner-up` | how decisive |
| entropy of `v` over the roster | confident anywhere at all |
| `log1p(k_c)`, flag `k_c = 1` | evidence quantity |
| `max(S_c)`, `mean(S_c)`, `std(S_c)` | evidence quality and agreement |
| `mean(t_j)` over `S_c` | the trust stage's summary of its witnesses |
| `max(S)`, `mean(S)` | is the query well matched to anything |
| coverage = fraction of roster with `k > 0` | how much of the answer space support can see (partial coverage) |
| `log C` | roster size, so margin and entropy are comparable across rosters |

## 6. Deliberately absent

* Anything from the text path in either gate: `text_cosine`, text margin, text entropy,
  `disagreement`. Text *confidence* is a compressed label-identity signal on the training
  vocabulary (v2's held-out-label loss rose to 5.6 at ~0.5 accuracy: confidently wrong).
* `r_candidate` (removed; an ablation flag at most, never on by default).
* Candidate slot index, dataset/source identity, raw query/support embeddings, label tokens.
* Acquisition vectors in the first build. They *are* available (the encoder exposes them; the
  trainer passes `query_acquisition` / `support_acquisition`; the evaluator reconstructs them per
  stream), and a query–support acquisition-mismatch feature is the natural extension if the
  paired cross-placement cells show the trust weights not reacting.
* Evidence injection, learned temperatures, attention over label tokens, counterfactual groups,
  grouped losses, path-improvement auxiliaries.

## 7. Numerical hygiene (required, not optional)

All features bounded; padded rows masked to zero *before* LayerNorm; `std` with an ε floor
(k=1 has exactly zero variance; sqrt′(0) is infinite); gates computed in fp32 inside autocast;
MLPs: LayerNorm → hidden 32 → GELU → scalar, zero-initialised output layer; `−inf` masking only
on the final softmax, finite fills elsewhere. A test must feed NaN/inf padding and assert finite
loss and gradients.

## 8. Training signal — the part no architecture supplies

Nothing in the current curriculum says "text is unreliable here". Add **label-text corruption** on
a fraction `p = 0.25` (fixed a priori) of episodes with `k ≥ 1`: permute candidate text embeddings
across the roster, or replace them with placeholder-string embeddings. The gates are text-blind and
cannot detect corruption; that is the point. Corruption lowers the *expected* reliability of text
as a function of support-evidence quality, so the optimal `λ(g_c)` moves toward support wherever
support evidence is strong and stays high only where it is weak or absent — the policy learned from
the objective rather than written in.

* Corrupt only `k ≥ 1` episodes; a corrupted `k = 0` episode is unlearnable noise for the text
  alignment.
* **Stop-gradient the text branch on corrupted episodes** (detach `text_c`), so corruption trains
  `t` and `λ`, never `p_text` or the encoder's text alignment.
* Plain cross-entropy. No auxiliaries, no counterfactual groups.

## 9. Acceptance tests before any 40k run

Unit tests:

1. **Step-0 parity**: v4 with zero-initialised gates and `b` set from v3's table reproduces v3's
   logits on a fixed batch (bit-for-bit in fp32).
2. **Gradient reach asserts nonzero** for every parameter of `MLP_t` and `MLP_λ` after one step
   on a real batch (v2's test asserted finite only; 34 tensors had zero grad).
3. **No deletion**: with `λ = λ_max` and `t_j = −T` on every support, the support term still
   moves the argmax on a constructed disagreement case.
4. NaN/inf padding safety; support-permutation and candidate-permutation invariance.

3k-step screens on the internal panel, v3 as control, measuring `λ` and `t` directly:

* k from 1 → 8: mean `λ` on enrolled candidates falls;
* text corruption on validation episodes: accuracy drop smaller than v3's;
* supports cross-placement relative to the query: mean `t_j` on those supports falls relative to
  matched supports;
* `λ` against whether text was actually right, on the label-held-out validation panel (diagnostic
  panel only — no labels removed from optimisation): the curve must slope, not sit flat.

If `λ` or `t` does not move under these manipulations, the design is inert and work stops at 3k.

## 10. Evaluation plan

Fixed 40k budget, `last.pt` primary, Stage A recipe plus `--text-corruption 0.25`, one arm, sealed
+ scenarios on the existing manifests. Readouts: full head; text-off (`λ = 0`); trust-off (`t = 0`);
and two diagnostics of the **existing v3 checkpoint** in the scenario runner that the record
cannot currently answer — a text-only readout (how much of the 53.9 unenrolled accuracy is semantic
alone versus negative evidence) and the base readout (whether over-trust also costs points in the
mismatch scenarios). Both are readouts of existing checkpoints and need no training; run them first.

## 11. Sequencing

1. v3 text-only and base readouts in the scenario runner; rerun the residual arm's scenarios.
2. `support_classifier_v4` in the residual lineage (`--classifier residual` plus new flags; not a
   fourth contextual generation), unit tests, smoke.
3. 3k screens with the acceptance manipulations.
4. One 40k arm, sealed + scenarios, results record.

## 12. Discussed, not agreed: a compositional semantic path

The semantic path is a single linear map from pooled sensor feature to text space; nothing in it
generalises compositionally, which is why unseen vocabulary (MM-Fit) sits at chance. Proposed for a
later discussion: quantise sensor features into a learned codebook of primitives ("arm forward",
"small vibration", "pocket device") and let codes vote for candidates. It can help only if the
**text side is grounded in the same primitives** (codes carry descriptions, or labels decompose
into codes); otherwise it is the same sensor→text map with a bottleneck. Prior art:
attribute-based zero-shot recognition and semantic-attribute ZS-HAR. Risks to design around:
domain-specific codes from eight sources, and the codes→candidates step becoming a learned mixer
over labels (same blindness discipline required). If built: modular, off by default.
