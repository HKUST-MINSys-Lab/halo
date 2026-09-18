# Contextual semantic-voting classifier: implementation plan

Date: 2026-09-17.
Status: approved design direction, implementation planned. No model code, training, or evaluation
is changed by this document. The implementation on this date remains the v3 residual classifier.

## 1. Scope and decisions

Prerequisite audit and separate experiment tracks:
[the heterogeneity and metadata audit](2026-09-17-heterogeneity-metadata-audit.md).
In particular, cross-stream manifests currently rewrite support labels into the query vocabulary;
preserving source wording for semantic-transfer experiments needs an explicit protocol revision,
not just changing the classifier's text gather. The new head must accept independent label text,
but historical manifest behavior must not be silently reinterpreted.

Replace the learned classifier, not the encoder, recording pool, source roster, evaluation
manifests, baseline fusion, or differentiable-neighbor control. The new classifier contextualizes
all supplied evidence before making either kind of comparison. It combines a support-based
distribution and a direct semantic distribution using a learned weight for each candidate.

The first implementation retains the current unified classifier parameter sharing. An empty
support set changes the input tokens and available scoring paths, not the weights. Do not revive
the historical independent zero/few-shot heads as an incidental part of this change. They remain
a separate, explicitly named experiment if requested. The encoder remains shared in either case.

Use the existing two-layer pre-LayerNorm set transformer, residual connections, nonlinear FFNs,
and final normalization. Start with its existing capacity: D=128, four heads, FFN width 256,
dropout 0.1. Do not enlarge the encoder or change the curriculum at the same time.

No hard top-k selection; every valid supplied support participates. No direct candidate-logit MLP,
support scalar residual, candidate scalar residual, count-bucket lambda, or hand-crafted gate
features in the new head. The gate produces a mixture weight, not an unrestricted additive logit.
This encourages explicit comparison but cannot guarantee that the model uses supports well.

## 2. What the source currently does

| Component | Current source | Required change |
|---|---|---|
| Recording embeddings | `encode_recording_rows` and `split_encoded` in `training/support_classifier/train.py` encode query/support rows through the same encoder and pool | Reuse; verify both gradient paths and repeated-row accumulation |
| Attention content | `model/support/residual_classifier.py` projects sensor content, uses a fixed text projection, and adds roles, pair tags, and bound-candidate tags | Learn text input projection; retain roles and sensor-label pairing, remove support-to-candidate identity tags |
| Sensor score | Original centered query/support cosine plus an attention-produced scalar correction | Cosine of projected, post-attention sensor states |
| Support vote | `support_bound` scatter-add to exact candidate indices | Similarity-based vote from contextualized support-label states to candidate states |
| Direct semantic score | Original query through `p_text` against original candidate text | Projected, post-attention query against post-attention candidate states |
| Combination | Additive text term with count-bucket lambda, optional statistics gate, and candidate residual | Candidate-local sigmoid MLP and normalized probability mixture |
| Training support text | `episode_text` already gathers actual labels from `corpus.recordings` | Preserve this correct behavior; do not replace with candidate-index lookup |
| Evaluation support text | `_halo_residual_predictions` reconstructs text using candidate slots | New scorer must gather actual `plan.support_labels`, including labels outside the candidate set |
| Model loading | v2/v3 checks distributed across trainer and evaluators | Explicit new architecture dispatch, retaining strict historical loading |

The old model's exact centered-neighbor initialization does not apply to the new model. Never
claim the replacement cannot perform worse than 1-NN. Support-centroid subtraction is also not an
implicit part of the new comparator: start with projected contextual states and normalization as
defined below. Keep the old centered floor as a clearly labelled diagnostic, not the new default.

## 3. Inputs, tokens, and masks

Let B be episode-query rows, S the padded number of supplied support recordings, C the padded
number of candidates, E the encoder's recording-vector dimension, and T the text dimension.
Currently E=D=128 and T=384, but an explicit E-to-D adapter should allow frozen baseline encoders
with different output dimensions without changing the classifier algorithm.

| Input | Shape |
|---|---|
| Query recording vector | `[B,E]` |
| Support recording vectors | `[B,S,E]` |
| Support-label text embeddings | `[B,S,T]` |
| Candidate-label text embeddings | `[B,C,T]` |
| Support/candidate masks | `[B,S]`, `[B,C]` |
| Support-pair tag IDs | `[B,S]` |

Use a shared `LayerNorm(E) -> Linear(E,D)` adapter for query and support recordings, and a shared
`LayerNorm(T) -> Linear(T,D)` adapter for support-label and candidate-label text. Keep the text
encoder frozen; these adapters and the downstream head are trainable.

Compose `[query | support recordings | support labels | candidate labels]`, giving
`[B,1+2*S+C,D]`. Add the four existing role embeddings. The sensor token and label token for one
support share one instance tag. Use the existing normalized token-composition helper and
conservative initial tag contributions; record its initialization in the configuration.

Do not attach a matching candidate ID to support tokens. That would bypass the intended semantic
label bridge. Candidate tokens need their text and role, not an arbitrary candidate-ID embedding.
Support-pair IDs remain local, unique among valid supports, and randomized during training. At
evaluation use deterministic IDs carried with the logical pairs. Their reassignment sensitivity
is a separate diagnostic from permutation equivariance.

Zero invalid input rows before token adapters; keep all padding masked as attention keys, zero
padded outputs before score reductions, and apply score masks separately. Masking attention keys
alone does not sanitize non-finite padding. There must always be a valid query and at least one valid candidate.
There is no cross-query attention: batching queries must not make one query depend on another.
Supports are contextualized independently for each query, even if their encoder outputs are shared.

Preserve acquisition conditioning already present in the encoder. This change does not introduce
an explicit acquisition-description token. Record the encoder's acquisition-text flags; do not
claim the head directly observes metadata that the configured encoder does not supply.

## 4. Exact scoring rule

After attention, extract `q:[B,D]`, `s:[B,S,D]`, `l:[B,S,D]`, `c:[B,C,D]`.

### 4.1 Post-attention projections

Start with two bias-free learned D-to-D linear maps, initialized to identity:

```
q_m = L2Normalize(P_motion(q))
s_m = L2Normalize(P_motion(s))
l_t = L2Normalize(P_text(l))
c_t = L2Normalize(P_text(c))
```

Share the motion map between query and support, and the text map between both label types.
The semantic comparison uses `q_m` and `c_t`, so both maps learn a common comparison space.
Do not add another independent query-semantic MLP initially. More capacity or a separate semantic
query projection would be an ablation, not a prerequisite for this design.

### 4.2 Support path

```
a[b,s]   = dot(q_m[b], s_m[b,s]) / tau_motion
log_w    = masked_log_softmax(a, over supports)
b[b,s,c] = dot(l_t[b,s], c_t[b,c]) / tau_label
log_L    = masked_log_softmax(b, over candidates)
log_p_support[b,c] = logsumexp_s(log_w[b,s] + log_L[b,s,c])
```

Interpretation: a support's sensor similarity determines its vote weight; its label similarity
determines how that vote is distributed among candidates. There is no `support_bound` in this
calculation, and no test of whether the support's label equals a candidate string. Unequal counts
can legitimately affect total evidence; do not silently add class balancing or drop duplicates.
Measure count sensitivity in the existing unequal-enrollment conditions.

### 4.3 Semantic path

```
z_sem[b,c] = dot(q_m[b], c_t[b,c]) / tau_semantic
log_p_semantic = masked_log_softmax(z_sem, over candidates)
```

Both query and candidate states have already attended to all available valid tokens. There is no
pre-attention query-to-text shortcut and no extra raw-candidate prior logit.

### 4.4 Learned combination

For each candidate concatenate contextualized states before the final comparison projections:

```
x[b,c] = concat(q[b], c[b,c])                         # [B,C,2D]
g      = Linear(GELU(Linear(LayerNorm(x)))).squeeze(-1) # [B,C]
alpha  = sigmoid(g)                                   # [B,C]
```

The same MLP is applied to every candidate: `2D -> D/2 -> 1` (256 -> 64 -> 1 at D=128).
Its parameter count is independent of C. Use an expanded query view, not C separate modules.
Initialize the final bias to zero and its weights to small nonzero values so alpha starts near
0.5 while gradients reach the earlier gate layer on the first informative batch. Test rather
than assume that this initialization leaves every path trainable.

For rows with support, combine in log space:

```
log_mass = logaddexp(logsigmoid(-g) + log_p_support,
                     logsigmoid(g) + log_p_semantic)
log_p    = log_mass - logsumexp(log_mass over valid candidates)
```

Return `log_p:[B,C]` as the classifier's logits contract. Invalid candidates are negative infinity.
The existing cross-entropy/NLL is compatible with these normalized log probabilities.

Candidate-specific alpha requires the final renormalization. Alpha is a local, pre-normalization
mixture weight, not a directly interpretable final fraction of the prediction. Branch softmaxes
normalize mass but do not guarantee calibrated confidence. Log branch entropy and calibration
diagnostics on internal validation, without fitting test-specific temperatures.

### 4.5 Empty support and numerical behavior

For an episode with no valid support, return `log_p_semantic` directly. Do not execute an all-masked
support softmax, fabricate uniform support probabilities, or let an unobserved-support gate alter
semantic scores. Mixed zero/enrolled batches must follow this rule per row; an all-empty batch
must also support S=0. Gate and support-only parameters have no reason to receive gradients from
an all-zero-support batch; check them on enrolled batches instead.

Use BF16 autocast for adapters/attention/MLPs on supported hardware. Cast to FP32 before L2
normalization, cosine products, temperatures, log-softmax, logsumexp, and mixture arithmetic.
Use standard normalization epsilon and numerically stable log-sigmoid functions. Parameterize
each positive temperature with softplus plus a numerical floor and initialize all three from the
existing 0.07 scale. These are learned calibration parameters, not three hand-tuned schedules.
Record values and gradients; do not use clamps that silently strand a parameter at a bound.

## 5. Code changes and compatibility

### Commit A: model and independent contracts

Add `model/support/contextual_classifier.py` with `ContextualClassifierConfig` and
`ContextualSupportClassifier`. Prefer the public CLI name `contextual` and architecture identifier
`support_contextual_mixture_v1`. Keep the residual and legacy mixer modules unchanged for loading
old checkpoints. Reuse `SetAttentionStack` and `ScaledSum`; do not copy transformer internals.

The new forward method accepts the inputs in section 3. `support_bound` and `candidate_slot` are
not required for inference. Keep candidate membership indices only in episode construction,
ground-truth/control readouts, and telemetry outside this head. Return `logits`, `support_weight`,
branch log probabilities, `semantic_weight`, `has_support`, and temperatures. Large intermediate
label-score matrices or attention weights are opt-in diagnostics, not mandatory return values.

Add a small explicit architecture/config factory shared by trainer and evaluators. Preserve strict
state loading and old defaults; unknown versions fail clearly. Do not accept an old state dict
with `strict=False` and call that a migration.

### Commit B: trainer, initialization, and telemetry

Update `training/support_classifier/train.py`: CLI selection, config construction, `run_step`,
optimizer groups, checkpoint serialization/loading, parameter accounting, validation, and resume
trajectory. Audit `development_panel.py` and other head construction sites as well.

Retain corrected acquisition/enrollment sampling and the existing support-set/regime-weighted
cross-entropy. No new objective, augmentation, trainable text encoder, or auxiliary loss. Partial
enrollment remains independent of the truth. Do not quietly enable the optional rate/modality
or old adaptive-gate experiments. Snapshot all realized settings in the new run configuration.

Skip legacy `initialise_text_projection`/lambda calibration for this head; its separate `p_text`
does not exist. Preserve necessary encoder/frontend calibration using clean train-only data and
the same initialization artifact across matched arms. An encoder warm start loads encoder and
pool weights only; head/optimizer start fresh. A true resume requires matching architecture,
trajectory, optimizer, scheduler, and RNG states.

Use the current query/support encoding and deduplication path, without detach or inference mode
in the end-to-end arm. Frozen-encoder experiments explicitly disable encoder gradients but still
train the head. Only cache encoder outputs across optimization steps when the encoder and pooling
are frozen. With an updating encoder, re-encode both query and support recordings each step.

Telemetry must stop assuming `r_candidate`, `r_support`, `lambda`, or `metric_part` exist. Log:

- Loss, accuracy, macro F1 and support coverage by existing enrollment/acquisition conditions.
- Semantic-only, support-only, and mixture predictions from the same attention forward; note
  that these are readout decompositions, not absence-of-support interventions.
- Alpha mean/spread/saturation on supported episodes, plus branch entropy, disagreement,
  learned temperatures, support entropy and effective support count.
- Rescue/overturn rates versus raw 1-NN on identical valid supports; retain the old centered DN
  floor only under its distinct name. No neighbor accuracy for an empty-support episode.
- Gradient norms for encoder, recording pool, adapters, attention, comparison projections, gate,
  and temperatures at a bounded diagnostic interval; finite-value checks and clipping frequency.
- Realized C/S distributions, padding fraction, time/step, data-wait time, and peak GPU memory.

Every subgroup metric needs its own count/fraction for aggregation. Do not average absent groups
as zero or expose nominal k as a realized count. Heavy counterfactual forwards belong in periodic
internal validation, not every training step.

### Commit C: evaluation and result provenance

Update `sealed_eval.py`, `run_scenarios.py`, `development_panel.py`, and their classifier loaders
to dispatch by the new architecture. Reuse a cached loaded head per checkpoint/device/config.
In the new scorer build a text table over the union of actual support labels and candidates,
not by gathering candidate vectors at `support_bound`. Test an off-roster support label explicitly.

Preserve current information conditions: the learned HALO classifier currently receives no
support tokens at k=0 and uses its semantic path. The DN training-bank bridge is a different
readout. Supporting arbitrary supplied label text does not authorize silently adding a training
bank to the new classifier, especially not at k=0. Such a bank would need a separate protocol.

Keep dataset/subject/execution exclusions, candidate rosters, identical manifests, window lengths,
baseline equal-weight fusion and reporting rules unchanged. Evaluate new head as `halo-classifier`
with architecture/checkpoint identifiers and an explicit human-readable variant name. Do not
merge its rows with residual-v3 rows. Historical scores remain historical; no new sealed scores
are produced by implementation or smokes.

Replace residual-specific diagnostic labels for this architecture with semantic-only,
support-only, fixed-half mixture, and learned mixture. Support/label counterfactuals must rerun
attention because contextualized states depend on the evidence. Architecture/config hashes must
invalidate head predictions, while encoder-feature caches remain reusable only if the encoder,
pool, preprocessing, metadata and existing feature-cache contract are identical.

### Commit D: documentation and handoff

After the above contracts pass, update the implemented-design sections of `DESIGN_OF_RECORD.md`,
the experiment roadmap, CLI examples, and the classifier documentation. Keep the 2026-09-14
residual design explicitly historical, not overwritten with different equations. Link this plan,
record tests/profile artifacts and implementation commit, and state whether training is authorized.

## 6. Acceptance tests before any substantial run

1. Reference arithmetic: vectorized support voting and log-mixture match a tiny FP64 loop;
   distributions sum to one, and a constant alpha=0.5 matches the arithmetic probability average.
2. Masks: ragged C/S, S=0, all-empty and mixed batches, one candidate, one support, unsupported
   candidates, arbitrary padded values, and large logits produce finite valid losses/gradients.
   Missing candidates fail early; off-roster support labels are legal, with no candidate binding.
3. Set behavior: jointly permuting support sensor/label/mask/tag rows preserves output; permuting
   candidate text/masks permutes predictions. Adding masked padding and changing sibling queries
   cannot change a query's valid predictions. Reassigning pair IDs is measured separately.
4. Gradient reach: on a nondegenerate enrolled batch, gradients reach q/s encoder outputs, both
   pooling paths, token adapters, attention/FFNs, both projections, gate layers and temperatures.
   Verify accumulation when one encoded recording is reused. A zero-support batch trains its
   semantic path without requiring gradients in unused parameters. Frozen encoders stay frozen.
5. Counterfactuals: swapping which sensor belongs to which label changes controlled predictions;
   source labels not in candidates can still vote; zero-support scoring equals semantic scoring.
   Random-model sensitivity is only a mechanical test, not proof of learned support use.
6. Numerics: FP32/BF16 forward and backward are finite with measured tolerances; no branch
   softmax is taken over padding alone. Check duplicate/unequal support-count behavior explicitly.
7. Serialization: strict new-head round trip, real historical v3 round trip, encoder-only warm
   start, and short uninterrupted-versus-resumed training agree within the relevant RNG tolerance.
8. Wiring: a few real-corpus train/validate/save/load steps; a bounded internal scorer test of
   zero, complete, partial and mismatched acquisition episodes. Test the sealed runner's code
   with synthetic/internal fixtures, not by repeatedly opening sealed test data during development.

Use focused unit tests first, then one three-to-five-step real-data smoke. Run adjacent sampling,
loader, classifier, and evaluator regression tests. No long diagnostic scripts are needed to
implement this change, and no full training/evaluation is authorized by the planning request.

## 7. Efficiency and bounded profiling

Use SDPA and batched matrix products. The additional label bridge costs O(B*S*C*D), while full
attention costs O(B*(1+2*S+C)^2*D). Thus attention at large support counts, not the tiny gate,
is the principal scaling risk. A frozen sensor vector does not mean its query-conditioned
attention output can be cached across queries.

Retain recording-length bucketing and deferred collate. Bucket head calls by actual token length
if existing batching permits it, and microbatch independent queries at evaluation to limit memory.
Never change support membership or drop rows to meet a memory cap. Support-pair capacity must
cover the declared feasible C*k; exceeding it needs a clear preflight failure, not truncation.

Start with the straightforward batched implementation. If the label-score matrix alone is too
large, chunk the independent support axis and combine its vote using exact logsumexp accumulation;
the support normalization must still cover all valid supports. Do not normalize chunks separately
or replace global self-attention with independent support chunks.

Profile forward/backward on small, typical, and large *observed* episodes, plus k=32/64 where
feasible. Keep timing runs bounded (warmup then roughly 20 measured steps per representative
case). Record compile startup separately if trying compilation. Compare new head, residual head,
and encoder-only time with the same shapes, dtype, and encoder. Report steady-state median/p95,
peak memory and estimated total-run time; do not promise a 40-minute run without measurement.

## 8. Experiment order after implementation is accepted

1. A short internal screen with one identical, fixed encoder checkpoint: old residual versus new
   contextual head, both head-only trained on identical corrected-curriculum manifests. This
   isolates the head change; existing short screens are context, not a matched experiment.
2. On the new head, report both paths, fixed-half mixture, learned mixture, and support-removal/
   pairing interventions. Use internal-held-out subjects only. Branch-only scores after attention
   do not establish causal support value; removal/shuffling reruns provide that evidence.
3. Train the accepted new head end to end, preserving gradient flow through both recording paths;
   compare with a curriculum-matched DN-trained encoder + 1-NN and the frozen-encoder head arm.
4. Select checkpoints using the existing fixed internal protocol, report its regime breakdown,
   and only then run the frozen aggregate/scenario manifests and update results.

Expected improvement is a hypothesis: contextual motion comparison may resolve acquisition
mismatch, semantic voting may transfer label relationships, and the gate may handle partial
coverage. The first implementation changes several parts together and cannot alone attribute a
gain to any one of them. No gain, superiority to 1-NN, or calibrated semantic trust is guaranteed.
