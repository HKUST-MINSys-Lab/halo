# Support classifier design — 2026-09-14

**Status:** design of record for the learned support-conditioned classifier. Core v2 implementation,
strict checkpoint round-trip, unit contracts, and a three-step real-data smoke completed
2026-09-14; no training or sealed evaluation has been launched. The full suite passes (788 passed,
1 skipped). The development-only centring and cross-placement diagnostics remain pending and are
not a training-readiness blocker.
**Readiness update (2026-09-14):** the subsequent independent
[classifier audit and profile](../journal/RESIDUAL_CLASSIFIER_AUDIT_PROFILE_20260914.md)
confirmed and fixed encoder checkpoint reconstruction, mixed-regime loss weighting,
partial-enrollment score calibration, resume configuration, telemetry, and parameter accounting.
The corrected checkpoint format is `support_classifier_v3`; v2 remains historical-evaluation-only.
Supersedes the two-head `SupportTokenMixer` as the intended learned readout; differentiable
neighbours remains the unchanged parameter-free control.

**Handoff.** §8 is the implementation brief. Standing constraints: "implement" means **build +
tests + a short smoke** — do **not** launch a training run, do **not** run sealed evaluation, and do
**not** run the §2.2 centring ablation on sealed data; all three need Alex's explicit go. The other
agent may be editing concurrently; unexpected edits are not corruption — note them, never revert.
Interpreter: `/home/alex/code/HALO/legacy_code/.venv/bin/python`, always from
`/home/alex/code/HALO/halo`.

**Evidence base.** Sealed comparison of 2026-09-13 (`docs/results/RESULTS.md`), the readout table
in `training/support_classifier/evaluations/sealed_comparison_20260913/`, and the neighbour-control
diagnostics of 2026-09-14
(`training/support_classifier/evaluations/neighbor_diagnostics_20260914/REPORT.md`). Those
diagnostics were run on cached **sealed** embeddings and are hypothesis-generating; every
architectural choice below must be confirmed on **training-subject-held-out validation** before it
is adopted. There is no third data split (see `halo-train-eval-only`); "development data" in this
document means subject-held-out validation inside the training roster.

---

## 1. What the classifier is for — the measured deficiencies

The encoder plus differentiable neighbours is already a strong readout. The classifier's job is
confined to three regimes where the parameter-free vote is measurably short, and nowhere else.

| regime | measurement (8 s, sealed) | what NN cannot do |
|---|---|---|
| **k = 0** | HALO 34.1 via training-bank ConSE bridge; HARNet 37.2. Yet k=0→k=1 recovers +40 points on ut_complex (20.7→66.5) and realworld (24.8→59.3): the representation separates the classes, the sensor→text bridge does not. | Score a candidate with no enrolled example. |
| **k = 1** | Every readout identical (65.3): one support per class, nothing to reweight. On wrong predictions the correct support is in the top-5 **92.5%** of the time — it is nearly always close, just not closest. | Bring a second source of evidence to break a near-tie. |
| **k ≥ 8** | Ridge +2.33 over 1-NN, soft vote +1.65, prototype +0.75. **15.6%** of wrong soft-vote predictions already have the correct label's best support ranked **first** — the vote was outvoted by several mediocre wrong-label supports. 72.6% have a correct support in the top-5. | Reweight supports by trustworthiness; recover rank-1 evidence the vote diluted. |

Stability: enrollment-draw std is 0.36–0.54 macro-F1, so a classifier delta must clear ~1 point to
be real. Support sensitivity: removing the top support flips 16.5% of predictions, a random one
0.42% — the vote correctly relies on strong evidence, and the classifier must not diffuse that.

**Not targets.** inclusivehar (flat k-curve 34→38: encoder/data, not readout). Cross-device
interaction (hierarchical mean already gives +9 to +12 at k=1, above UniMTS's native fusion).
High-k margin over baselines (all encoders converge under NN).

**Why not just ship ridge.** Ridge is the measured ceiling at k≥8 but it is a per-query linear
solve — materially slower, and not an inference-time readout. The learned residual in §2.3 is the
amortised form of what ridge computes on the supports.

## 2. The design

One scoring rule for every k. Everything learned is a **residual on top of differentiable
neighbours**, initialised so the untrained classifier equals the control exactly.

### 2.1 Inputs (inherited unchanged)

Pooled query vector `q`; support vectors `s_i` each bound to candidate `c(i)` via the integer
`support_bound` (kept as a plain index, never an embedding); candidate label texts `t_c` from the
**frozen** SBERT encoder; per-candidate support count `k_c`, which may be 0.

Token set `[ q | S | S-labels | T ]` with the existing role embeddings and both tag systems:
the **pair tag** binds `s_i` to its own label token (same instance); the **candidate tag** binds a
support-label token to the candidate token carrying the same label. Slot order is shuffled per
episode so neither tag table can become a label lookup. A candidate with `k_c = 0` simply has no
support or support-label tokens — the k=0 episode is this set with every support masked, not a
special case.

### 2.2 Base score — differentiable neighbours on **centred** vectors (inside the classifier)

**Where this lives.** Centroid subtraction is the first, parameter-free step **of the learned
classifier**, which is a HALO-only component — no baseline has a support-conditioned learned head.
The shared readouts (`training/support_classifier/neighbors.py` differentiable neighbours, 1-NN,
prototype, ridge) are **unchanged and uncentred**, applied identically to every provider; that is
where the fairness contract lives and nothing here touches it. Centred differentiable neighbours
must therefore never be reported as HALO's parameter-free number in a shared-readout row — it is
reported only as "HALO classifier, residual disabled".

**The step.** Before any similarity is computed, subtract the episode's support centroid from
every support *and* from the query, then L2-normalise:

```
μ        = mean_i s_i                                        over all valid supports
q'       = normalize(q  − μ)
s'_i     = normalize(s_i − μ)
```

The query is centred by the **support** centroid, not by itself: a single point cannot be centred
on itself, and using the support mean keeps the readout non-transductive over queries. At k=1, μ is
the class-balanced centroid over the C supports, so the operation is defined at every k.

Why: every support from a dataset carries that dataset's device / placement / subject-population
signature, and the query carries the same. That shared component dominates the cosine and squashes
the discriminative residual — the k=1 near-tie the diagnostics found. Centring removes the shared
component rather than describing it, which is why it can work where the `sensor_bias` descriptor
did not (`9b7d75d`). The same operation was already found load-bearing in the *training* loss
(`aa5ff15`: the uncentred cosine term was self-extinguishing; `d9956a3`: centring became the
default) but was never applied to the readout. This is SimpleShot's centre-then-L2-normalise
(Wang et al., 2019) applied per episode.

Note what is **not** a centring: subtracting the *mean similarity* per query
(`cos(q,s_i) − mean_j cos(q,s_j)`) shifts every support by the same constant and is invisible to
both softmax and argmax. Only subtracting the mean *vector* changes the geometry.

Then differentiable neighbours, verbatim on the centred vectors:

```
a_i      = cos(q', s'_i) / τ                                 per support
w_i      = softmax_i(a_i)                                    over all valid supports
vote_c   = Σ_{i : c(i)=c} w_i
base_c   = log(vote_c)                                       (−∞ when k_c = 0)
```

This is the same vote as `neighbors.py`, computed on centred vectors inside the classifier. The
residual in §2.3 sits on top of it. The uncentred `neighbors.py` control is untouched.

**Ablation, run before the residual is built** — which centring the classifier's base uses. Four
variants on HALO features, identical supports, k=1 and k=8, training-subject holdout, using the
existing `training/support_classifier/neighbor_diagnostics.py` harness (its matched-decision-rule
framework; the variants are additional rules in it):

| variant | μ | transductive? |
|---|---|---|
| uncentred (today) | — | no |
| **episode support mean** (default above) | mean of the episode's supports | no (over queries) |
| training-corpus mean | mean over the training corpus, fixed | no |
| CSLS hub penalty | per-*support* mean similarity to its nearest queries, subtracted | yes |

Expect the largest effect at k=1 and on cross-placement episodes. Whichever variant wins becomes
the classifier's base; the shared control is unaffected. This ablation also gives the
**attribution** number: classifier-with-residual-disabled *is* centred DN, so the gap between it
and the uncentred control is the free preprocessing gain, and the gap between it and the full
classifier is the learned gain. Report both.

### 2.3 Learned residual — scalars, not rewritten vectors

The attention stack sees the whole token set and emits **scalars**, not refined vectors:

```
r_i  = residual per support row      (from the refined support token)
r_c  = residual per candidate        (from the refined candidate token)
```

The corrected score:

```
a'_i     = cos(q', s'_i) / τ  +  r_i                      (centred vectors from §2.2)
w'_i     = softmax_i(a'_i)
vote'_c  = Σ_{i : c(i)=c} w'_i
logit_c  = log(vote'_c)  +  λ(k_c) · text_c  +  r_c
text_c   = cos(P_text(q), t_c) / τ_text
```

Why scalars: the diagnosed k≥8 failure is a *voting* failure (rank-1 correct evidence outvoted),
which a per-support reweighting `r_i` targets directly; and scalars cannot rewrite the embedding
geometry that already works. Vector refinement (FEAT-style) is an **ablation step up**, not the
default.

`r_i` may depend on the query, the support, its label text, the candidate texts, the tags and
roles — everything in the set. That is how it can learn "this support is far from its own label's
text, trust it less" or "the query is closer to another candidate's name than to this support's
label."

### 2.4 The text term

`P_text : d → 384` maps the refined query into SBERT space; `t_c` is raw SBERT, with **no learned
map on the text side** (`label_proj` removed). `λ(k)` is one learned scalar per bucket
`{0, 1, 2, 4, 8+}`.

`P_text` is initialised from a **closed-form least-squares fit** of encoder features to SBERT
label embeddings over the training corpus, so k=0 starts from a real predictor rather than noise.

### 2.5 Initialisation — the classifier is the control at step 0

- Residual heads for `r_i`, `r_c`: final linear layer **zero-initialised**. A zero-initialised
  output layer still receives gradient (its weight gradient is upstream-grad × activation), so the
  branch is live, not dead. If a multiplicative gate is used instead, initialise it *near* zero,
  never at exactly `σ(−∞)`, which would cut the gradient.
- `λ(0) = 1`, `λ(k>0) = 0`.
- Result: for every candidate with supports, `logit_c == base_c` (the **centred** vote) bit-for-bit;
  for a candidate with none, `logit_c == text_c` from the closed-form fit.

**This equality is a required test**, not a design intention.

### 2.6 Parameterisation

Shared attention trunk; **separate output projections** for the metric side (`r_i`) and the text
side (`P_text`, `r_c`), and separate `λ`. The two objectives — metric comparison and cross-modal
alignment — are different and keep their own weights at the output. The trunk is shared by default
because the zero-shot side trains on ~90 label texts while the few-shot side trains on millions of
pairs; ablate against a fully separate trunk. What is **not** separable is the scoring rule: a
candidate with no supports must be scoreable inside any episode, which the two-head design cannot
do.

## 3. Training

- Episodes with **mixed per-candidate k**. Support count is drawn per candidate, not per episode.
- **Label holdout inside every episode:** a random subset of candidates has its supports masked
  (`k_c = 0`) even though examples exist. Those candidates can only be scored through `text_c`, so
  the loss rewards landing near the *semantic* location in SBERT space rather than memorising which
  of the ~90 training labels is which. This is the mechanism-level fix for the failure that made
  every previous text head lose to its untrained control.
- Cross-entropy over candidates. Subject-disjoint, execution-disjoint supports as now.
- Encoder tuned jointly at the existing reduced learning-rate multiplier.
- Differentiable neighbours is trained and reported as the unchanged control arm.
- Configuration/placement descriptors are **not** inputs yet — see §6.

## 3b. Build order

1. Centring ablation (§2.2) — no training, one afternoon, decides the classifier's base.
2. Cross-placement NN measurement (§6) — same harness, decides whether descriptors are needed.
3. Classifier (§2.3–§2.6, §3) on top of the winning control.

## 4. Acceptance bars (fixed before any run)

1. **Untrained classifier == centred differentiable neighbours, bit-for-bit**, on a real batch
   (the classifier's own floor, HALO-only).
1b. The **shared** control rows — uncentred differentiable neighbours, 1-NN, prototype, ridge —
   are unchanged for every provider and are what bars 2–3 compare against.
2. On training-subject-held-out validation, at 8 s: ≥ soft vote at every k; **≥ ridge at k ≥ 16**;
   **no worse than NN at k = 1**; **≥ 37.2 at k = 0** (HARNet's bridge on sealed; use the
   validation analogue as the working bar).
3. Deltas must exceed the enrollment-draw noise (~1 macro-F1).
4. `|r_i|`, `|r_c|`, and `λ(k)` reported as telemetry. If they train to ~0 the classifier is not
   earning its keep and that is the result.

## 5. Required diagnostics (from the 2026-09-14 report, adopted)

- **Error stratification** at each k: correct label's best support ranked (a) first, (b) in top-5
  but not first, (c) absent from top-5. Gains must come from (a) and (b). Gains concentrated in (c)
  are suspicious — the classifier would be inventing evidence, not recovering it.
- **Support-sensitivity preservation:** top-support removal should still flip predictions at a
  rate comparable to the control (~16%); a collapse toward the random-removal rate (~0.4%) means
  the residual has diffused evidence.
- **Information-source ablation:** signal only → + label text → + acquisition descriptors → full
  token set with tags and roles.
- **Zero-shot decomposition** on a held-out-label validation split: is a k=0 failure a retrieval
  failure, a source-label→target-vocabulary mapping failure, or candidate ambiguity?
- **Harness:** reuse `training/support_classifier/neighbor_diagnostics.py`, which reconstructs
  execution-disjoint episodes from cached embeddings and runs matched decision rules on identical
  supports without encoder inference. The centring ablation is four more decision rules in it.
  Extend it with two measures it lacks: **hubness** (how often each support is chosen as nearest
  neighbour relative to its share — if high, centring helps and CSLS helps more) and
  **cross-placement stratification** on the aligned realworld/shoaib streams (query from one
  placement, supports from another).

## 6. Open questions — deliberately not built

- **Configuration/placement tokens.** The mixer sees no acquisition descriptor. Before adding one,
  measure how much NN degrades on cross-placement episodes (query from one aligned placement,
  supports from another) using the realworld/shoaib streams. If the drop is large, descriptors are
  justified and cross-placement episodes are what make them load-bearing (commit `9b7d75d`:
  conditioning nothing depends on trains to inertness). If small, do not add them.
- **Vector refinement** (FEAT-style) as the ablation above scalar residuals.
- **Whether λ(8+) → 0.** If text evidence remains useful at high k, report it.

## 7. Provenance

Design discussion 2026-09-13/14 with Alex. Independent neighbour-control diagnostics by the other
agent, 2026-09-14, confirmed the k=1 / k≥8 error structure and motivated the scalar-residual form
in §2.3 and the diagnostics in §5. Prior negative results this design is built to avoid:
`halo-phaseB-tier2-build` (trained decoder 44.2 vs 46.7 untrained), `halo-phaseB-step0-control`
(training pushed zero-shot below chance), and `halo-mixer-semantic-finding` (text path carries
semantics when embeddings are meaningful; scrambled vocabulary inverts the gain).

## 8. Implementation brief

### 8.1 Files and symbols

| action | path | what |
|---|---|---|
| **new** | `model/support/residual_classifier.py` | `ResidualSupportClassifier`, `ResidualClassifierConfig` — the §2 head |
| keep | `model/support/token_mixer.py` | `SupportTokenMixer` stays **untouched** so v1 checkpoints still load and reproduce |
| edit | `training/support_classifier/sampling.py` | per-candidate support masking (§8.4) |
| edit | `training/support_classifier/train.py` | `--classifier residual`; construction; least-squares init; checkpoint payload; telemetry |
| edit | `training/support_classifier/sealed_eval.py` | dispatch on `architecture_version`; two new readout names (§8.7) |
| edit | `training/support_classifier/neighbor_diagnostics.py` | centring variants as decision rules; hubness; cross-placement strata (§8.8) |
| **leave alone** | `training/support_classifier/neighbors.py` | the shared control. **No edits.** |
| **new** | `tests/test_residual_classifier.py` | §8.9 |

Reuse from `token_mixer.py` by import, not copy: `ROLE_QUERY/SUPPORT/SUPPORT_LABEL/CANDIDATE`,
`N_ROLES`, and the `_compose` pattern (role + candidate tag + pair tag). Reuse
`model/blocks.SetAttentionStack(spec, n_layers)` as the trunk.

### 8.2 Config

```python
@dataclass(frozen=True)
class ResidualClassifierConfig:
    text_dim: int = 384
    n_layers: int = 2
    max_candidates: int = 256
    max_supports: int = 8192
    temperature: float = 0.07            # τ for the vote; matches neighbors.DEFAULT_TEMPERATURE
    text_temperature: float = 0.07       # τ_text
    centring: str = "support_mean"       # "none" | "support_mean" | "corpus_mean"
    lambda_buckets: tuple[int, ...] = (0, 1, 2, 4, 8)   # k_c -> bucket by largest lower bound
    residual_enabled: bool = True        # False == "classifier, residual disabled" (attribution row)
    text_term_enabled: bool = True
    shared_trunk: bool = True            # False == fully separate trunks (ablation)
```

Reject invalid values in `__post_init__` exactly as `TokenMixerConfig` does. Every field is
serialised into the checkpoint (§8.6).

CSLS remains an explicitly transductive diagnostics-only rule. The trainable head and CLI reject
it rather than silently treating it as uncentred cosine similarity.

### 8.3 Forward

```python
def forward(self, *,
            query_feature: Tensor,        # (B, d)
            support_feature: Tensor,      # (B, K, d)   K may be 0
            support_label_text: Tensor,   # (B, K, 384) frozen SBERT of each support's label
            support_bound: Tensor,        # (B, K) long, candidate index per support; plain index
            support_mask: Tensor,         # (B, K) bool
            support_pair_slot: Tensor,    # (B, K) long, shuffled pair tags
            candidate_text: Tensor,       # (B, C, 384) frozen SBERT
            candidate_mask: Tensor,       # (B, C) bool
            candidate_slot: Tensor,       # (B, C) long, shuffled candidate tags
            corpus_mean: Tensor | None = None,   # (d,) only for centring="corpus_mean"
) -> dict[str, Tensor]:
    # returns {"logits": (B,C), "support_weight": (B,K), "k_c": (B,C),
    #          "r_support": (B,K), "r_candidate": (B,C), "text_score": (B,C), "lambda": (B,C)}
```

No `is_zero_shot` argument. **`k_c` is derived**, never passed:
`k_c = scatter_add(support_mask.float(), support_bound) -> (B, C)`. An episode with `K == 0` or
all-false `support_mask` is the k=0 case and needs no special code path.

Steps, in order — each is one small function so the tests in §8.9 can hit them individually:

1. `centre(q, S, support_mask, corpus_mean) -> (q', S')` per §2.2. `"support_mean"`: μ = masked
   mean of S per row; subtract from q and S; L2-normalise. `"none"`: L2-normalise only.
   `"corpus_mean"`: subtract the passed buffer. `"csls"`: see §8.8; at inference it needs the query
   batch, so implement it as a post-hoc adjustment on `a_i` rather than a vector transform.
2. `base_vote(q', S', support_bound, support_mask, candidate_mask, τ)` — **call
   `neighbors.differentiable_neighbor_logits` directly** on the centred vectors. Do not reimplement
   the vote. This is what makes acceptance bar 1 a test rather than a hope.
3. Token composition: `[q | S | S-labels | T]` with roles and tags via `_compose`. Content
   projections: `signal_proj` on q and S (LayerNorm + Linear, as today); **no `label_proj`** — the
   text tokens enter as raw SBERT through a fixed, non-trainable linear map to `d` (e.g. a random
   orthogonal projection registered as a buffer) so the trunk can attend over them without a
   learnable map on the text side.
4. Trunk: `SetAttentionStack(spec, n_layers)` with `key_padding_mask` from the masks. If
   `shared_trunk=False`, instantiate two.
5. Residual heads, both `Linear(d, 1)` with **weight and bias zero-initialised**:
   `r_support = head_s(hidden[support rows]).squeeze(-1) * support_mask`,
   `r_candidate = head_c(hidden[candidate rows]).squeeze(-1) * candidate_mask`.
6. Corrected vote: `a'_i = a_i + r_support_i`, then the same softmax/scatter as step 2 (reuse the
   function with an `extra_logit` argument, or recompute inline with identical arithmetic — the
   bit-identity test decides which is acceptable).
7. Text term: `P_text: Linear(d, 384)` on the **refined** query token;
   `text_c = cos(P_text(q̃), t_c) / τ_text`. `λ = lambda_table[bucket(k_c)]`, a `(len(buckets),)`
   parameter initialised `[1, 0, 0, 0, 0]`.
8. `logits_c = log(vote'_c).masked_fill(k_c==0, 0) + λ_c · text_c + r_candidate_c`, then
   `masked_fill(~candidate_mask, -1e30)`. Note the `k_c==0` fill: `log(vote)` is `−∞` there and
   must contribute **0**, not `−∞`, so a support-less candidate is scored by text alone.

When `residual_enabled=False`: skip steps 3–6, use `a_i` unmodified, `r_candidate = 0`. When
`text_term_enabled=False`: `λ = 0`. Both must be pure config switches, no code paths deleted.

### 8.4 Sampler — per-candidate support masking

`_draw_k_per_candidate` (`sampling.py:442`) draws exactly `k` executions for **every** candidate
and fails the episode if any candidate has fewer. Keep it. Add one step **after** it, at episode
construction:

- New sampler argument `p_mask_candidate: float` (default 0.25; trainer flag
  `--p-mask-candidate`). For each candidate **that is not the ground truth** with probability
  `p_mask_candidate`, drop all of its supports from `Episode.support` / `support_candidate`.
  Additionally, with probability `p_mask_gt` (default 0.10; `--p-mask-gt`), drop the ground-truth
  candidate's supports — this is the only way the head is trained to pick a **correct** answer from
  text alone while other candidates have examples.
- Record the masked set on the episode: new field `masked_candidates: tuple[int, ...]`.
- **Never mask every candidate**; if the draw would leave zero supports in the episode, the
  episode is a legitimate k=0 episode and `Episode.zero_shot` is set as today.
- The existing `zero_shot=True` episodes (`semantic_zero_shot`) stay; they are the all-masked case.

`episode_loss` (`train.py:506`): keep the cross-entropy and the support-set-id weighting. The
regime split (`few_shot` vs `zero_shot` index sets) can remain for telemetry, but the head no
longer needs it — remove the equal-regime weighting *only* if Alex agrees; default is to keep it.

### 8.5 Trainer wiring (`train.py`)

- `--classifier` choices become `("token_mixer", "neighbors", "residual")`. `"residual"` builds
  `ResidualSupportClassifier(spec, ResidualClassifierConfig(...))`. Config fields come from new
  flags: `--centring`, `--p-mask-candidate`, `--p-mask-gt`, `--no-residual`, `--no-text-term`,
  `--separate-trunk`, `--text-temperature`.
- `run_step` (`train.py:603`): in the `else` branch, if the classifier is a
  `ResidualSupportClassifier`, call it **without** `is_zero_shot`. Pass `corpus_mean` when
  `centring == "corpus_mean"`.
- **Least-squares init of `P_text`** (§2.4). Immediately after `calibrate_frontend` and before the
  first optimiser step: run the same calibration batches through the encoder, collect pooled
  features `X (N, d)` and each window's label SBERT vector `T (N, 384)` via the existing
  `text_of`; solve ridge `W = (XᵀX + αI)⁻¹ XᵀT` with `α = 1e-2 · trace(XᵀX)/d`; set
  `P_text.weight = Wᵀ`, `P_text.bias = 0`. Log `α`, `N`, and the fit's mean cosine on the
  calibration set as telemetry. Store `{"n": N, "alpha": α}` in the checkpoint as `p_text_init`.
- **Corpus mean** for `centring="corpus_mean"`: the masked mean of the same `X`, registered as a
  buffer on the classifier and saved with it.
- Telemetry per step: `classifier/mean_abs_r_support`, `classifier/mean_abs_r_candidate`,
  `classifier/lambda_<bucket>` for every bucket, `classifier/masked_candidate_fraction`,
  `classifier/text_score_gt_minus_max_other`.
- Param groups: classifier params at `args.lr` as today; `lambda_table` in the classifier group.

### 8.6 Checkpoint and round-trip

Payload (`train.py`): `architecture_version = "support_classifier_v3"`;
`classifier = state_dict()`; `classifier_config = dataclasses.asdict(cfg)`; `attention_spec`;
`p_text_init`. **v1 checkpoints (`support_token_mixer_v1`) must still load and evaluate
unchanged** — dispatch on the version string in both the resume path (`train.py:1181`) and the
evaluator. A test loads a v1 checkpoint and asserts the old readout path is taken.

### 8.7 Evaluator wiring (`sealed_eval.py`)

`_halo_token_mixer_predictions` (`:534`) currently hard-checks `support_token_mixer_v1`. Refactor
into a dispatcher: v1 → existing code, untouched; v2 (historical) and v3 → new
`_halo_residual_predictions` that builds the same inputs but **never** passes `is_zero_shot`, and
runs the head once per manifest batch for any k including 0. Do not mix v1 and v2 logic in one
function.

Two readout names, both HALO-only rows:

- `halo-classifier` — the full head.
- `halo-classifier-residual-off` — same checkpoint with `residual_enabled=False,
  text_term_enabled=False` forced at load; this **is** centred differentiable neighbours and is
  the attribution row of §2.2. It is **never** labelled as a shared readout.

The shared readouts (`1nn`, `prototype`, `ridge`, `differentiable-neighbors`) are computed exactly
as today on the same features. Do not add centring to them.

At k=0 the evaluator currently prefers a v1 mixer's zero-shot head, then falls back to the
training-bank ConSE bridge (`:936-960`). For v2 do the same: prefer `halo-classifier`, keep the
bridge row as a disclosed comparison, name both.

### 8.8 `neighbor_diagnostics.py` extensions (the §2.2 ablation harness)

Add decision rules `centred_support_mean`, `centred_corpus_mean`, `csls` alongside the existing
matched rules, on HALO features only (this is a HALO-classifier ablation, not a shared readout).
CSLS: `a_i ← 2·cos(q', s'_i) − r_q − r_{s_i}` with `r_x` = mean cosine to its 10 nearest
neighbours in the other set (Conneau et al.); transductive over the query batch, say so in the
output. Add **hubness**: for each support, the count of queries for which it is rank-1, divided by
its expected share; report the max and the Gini. Add **cross-placement strata** on realworld and
shoaib: query from placement A, supports from placement B ≠ A, reported per (A, B) pair. The
harness reads cached embeddings only; it runs on the **training-subject-held-out** cache, not the
sealed cache, unless Alex says otherwise.

### 8.9 Tests — `tests/test_residual_classifier.py`

1. **Bit-identity at init (bar 1).** Random `(B,K,C)` episode with every candidate having ≥1
   support. Fresh head, `centring="none"`, `residual_enabled=True`, `λ` at init. Assert
   `torch.equal(head(...)["logits"], differentiable_neighbor_logits(...)[0])`. Repeat with
   `centring="support_mean"` against the same function called on the centred vectors.
2. **k_c = 0 candidate scored by text only.** Mask one candidate's supports; assert its logit
   equals `λ(0)·text_c` and is finite; other candidates unchanged to 1e-6.
3. **All-masked episode is k=0.** `K=0` tensors → logits equal `text` for all candidates; no NaN.
4. **Gradients reach every parameter** from a single backward: `r` heads, `λ`, `P_text`, trunk.
   Assert no parameter has a `None` grad and the zero-initialised heads have non-zero grads.
5. **Residual off == centred DN**, config switch only.
6. **`P_text` least-squares init** on a synthetic `X, T` with a known linear relation recovers it
   to 1e-4; `P_text.bias` is zero.
7. **Sampler masking**: over 10k draws, masked non-GT fraction ≈ `p_mask_candidate` ±0.02, GT
   masked ≈ `p_mask_gt`; never all candidates masked unless `zero_shot=True`; `masked_candidates`
   never contains an index with surviving supports.
8. **Checkpoint round-trip**: save → load → identical logits; a v1 checkpoint still loads via the
   v1 path.
9. **Evaluator dispatch**: `halo-classifier-residual-off` row equals a direct
   `differentiable_neighbor_logits` call on centred features for a small manifest.
10. **Shared readouts untouched**: `neighbors.py` byte-identical to HEAD (`git diff --quiet`);
    `1nn/prototype/ridge` outputs on a fixed feature file unchanged.
11. **Support-sensitivity preservation** (from §5): on a synthetic episode, removing the top
    support changes the untrained head's prediction exactly when it changes DN's.

### 8.10 Smoke

`python -m training.support_classifier.train --classifier residual --steps 20 --smoke` on the
training roster; assert it runs end to end, writes a v2 checkpoint, and that the telemetry keys in
§8.5 appear. Then run the evaluator's **unit path** on a synthetic stream, not on sealed data.
**Stop there and report.**

### 8.11 Do not

- Do not edit `neighbors.py`.
- Do not add centring, or any HALO-specific step, to `1nn`, `prototype`, `ridge`, or
  `differentiable-neighbors` in the evaluator.
- Do not add a learned map on the text side.
- Do not pass `is_zero_shot` into the new head or branch on it inside it.
- Do not delete or alter `SupportTokenMixer` or the v1 evaluator path.
- Do not add configuration/placement descriptor tokens (§6 is open).
- Do not launch training, the centring ablation on sealed data, or sealed evaluation.
