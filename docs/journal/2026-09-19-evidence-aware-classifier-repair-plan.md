# Evidence-aware classifier repair plan

**Date:** 2026-09-19  
**Status:** agreed repair direction; literature review complete; implementation pending.  
**Supersedes:** no historical result. This is the proposed successor to
`support_contextual_residual_v1`, whose result remains reproducible and unpromoted.

## Why a repair is needed

The bounded contextual classifier trained and optimized normally, but its learned decision logic
did not preserve the strongest available evidence on sealed labels. At 8 seconds, the same trained
encoder reached 72.0/75.4/76.3 macro-F1 with the unmodified soft support vote at
`k=8/32/128`, while the complete classifier reached 64.0/67.6/70.3. The learned support path helped
at `k=1` but fell behind the support floor as enrollment grew. The semantic-only readout also fell
from 47.7 at `k=1` to 32.0 at `k=128`, showing that support-conditioned contextualization changed
the semantic expert in a harmful way. Internal validation did not expose this because its labels
come from the training vocabulary.

The repair does not remove either evidence source. It makes their current predictions explicit to
the reasoner and trains routing with controlled changes to support evidence.

## Agreed model contract

### Stable status-quo paths

Before learned episode reasoning, compute two differentiable normalized candidate distributions:

1. **Support status quo:** the centered soft support vote over every valid support; no hard nearest
   support selection.
2. **Semantic status quo:** direct query-to-candidate-label matching that does not change merely
   because additional support tokens were supplied.

For candidate count `C`, expose each log probability relative to the uniform distribution:

```text
support_evidence[j]  = log p_support[j]  + log C
semantic_evidence[j] = log p_semantic[j] + log C
```

Zero denotes uninformative evidence independently of `C`. Positive and negative values denote
evidence for and against the candidate.

### Evidence belongs to existing entity tokens

Do not create one token per scalar or one token per support-candidate pair.

- Add raw query-to-support cosine similarity to the corresponding **support recording token**
  through a shared trainable evidence projection.
- Add support-relative log probability, semantic-relative log probability, and a
  `has_direct_support` indicator to the corresponding **candidate token** through a shared
  trainable evidence projection.
- Keep support-label/candidate compatibility as a vectorized pairwise calculation after attention,
  or test it later as an attention bias. Do not materialize `S x C` pair tokens.

Evidence is a feature of an existing entity, not a new role. Query, support, support-label, and
candidate role embeddings remain unchanged. A support recording and its label retain the same
instance tag. Candidate labels receive no fake support-instance tag and no positional identity.
Each new evidence contribution has its own learned composition scale and begins conservatively.

Raw support count is deliberately excluded from the first version. Soft voting already describes
the distribution of evidence, and a count can reward duplicated weak examples. The binary direct-
support indicator is retained because "no demonstration exists" differs from "a demonstration
exists but disagrees." Add an effective-sample-size statistic only if an ablation establishes that
the score distributions and support tokens are insufficient.

### Contextual reasoning and outputs

Run the evidence-augmented query, support, support-label, candidate-label, and acquisition tokens
through the set contextualizer. It may produce:

- a candidate-specific adjustment for every support-candidate pair;
- candidate-specific relative reliance on support and semantic evidence.

The reasoner must not emit an unrestricted candidate logit that bypasses both evidence paths. A
support correction is zero-initialized but is not assigned an arbitrary fixed magnitude cap. Use
normalization, a learned scale, weight decay, and ordinary global gradient safeguards for numerical
stability. Following the literature review below, the final combiner is a normalized two-expert
mixture with candidate-specific learned reliance, not an unrestricted residual logit generator.

## Agreed training contract

### Main, branch-preservation, and routing objectives

Ordinary candidate cross-entropy remains necessary because both evidence paths may be wrong. Keep
an independently measurable cross-entropy on each status-quo branch so joint training cannot improve
the fused answer by silently destroying one expert. These branch losses are ordinary predictions,
not scenario-specific instructions, and their weights are modular configuration values.

Retain the generic non-regression objective on true-class log-odds as a guardrail:

```text
reference_quality = max(detach(support_quality), detach(semantic_quality))
L_best_path = softplus((reference_quality - final_quality) / temperature)
```

Use zero required margin: matching an already-confident best path can be the optimal attainable
answer. The reference is detached so neither expert can weaken itself to make routing easier.

This loss is not the routing teacher. The previous classifier already had a best-path-style
objective and still learned poor routing. Train the router explicitly from detached per-episode,
per-candidate branch quality. The target is which branch has the larger true-class log-odds, with
the target softened near a tie. This is the same generic rule for every scenario and does not name a
dataset, acquisition regime, or support count.

### Paired counterfactual episodes

For one base query, candidate set, and ground truth, construct two views:

```text
A: an ordinary support draw
B: one randomly selected support-evidence intervention
```

Candidate interventions include partial enrollment, removal of the truth's enrollment,
configuration-mismatched but valid support, shuffled support-label binding, and zero support. Both
views receive the same main and best-path losses. Query and candidate encodings, and overlapping
support encodings, should be deduplicated within the batch.

Do not hard-code that a named scenario is intrinsically trustworthy or untrustworthy. For each
view, measure the detached relative true-class advantage of semantic versus support evidence. The
router preference should move in the same direction between paired views:

```text
if semantic_advantage(B) > semantic_advantage(A):
    semantic_preference(B) >= semantic_preference(A)
else:
    support_preference(B) >= support_preference(A)
```

This is one scenario-agnostic relation: when changing only support evidence changes which expert is
better, learned routing should change in the same direction. No component is manually frozen or
assigned to a scenario-specific objective.

The paired relation supplements, rather than replaces, the per-view routing target. The per-view
target teaches which expert is currently stronger; the paired relation teaches that the decision
must respond when support evidence changes while query and candidate set remain fixed.

## Required controls and telemetry

Before a full run, report:

- stable support and semantic status-quo performance;
- refined support, refined semantic, and final performance;
- final regret relative to the detached better path;
- routing preference for candidates with and without direct support;
- routing movement between paired views and agreement with measured branch advantage;
- support-label-shuffle and support-removal sensitivity;
- metadata-shuffle sensitivity on matched and mismatched acquisitions;
- correction magnitude and gradients through encoder, recording pool, acquisition conditioners,
  evidence projections, contextualizer, correction head, and router;
- label-held-out and source-held-out development results, not only subject-held-out results on the
  training vocabulary.

The first cheap diagnostic is an oracle branch-selection upper bound on existing artifacts. If an
oracle that selects the better expert per query provides little gain, branch quality must be fixed
before routing. If the oracle gain is material, routing is the primary opportunity.

## Literature review

The review used primary conference papers and official proceedings. The exact HALO combination is
not a standard named architecture, but the failure and each proposed repair have close precedents.

### What has already been observed

1. **Useful branches can compete during joint training.** Wang et al. show that naive multimodal
   joint training can generalize worse than the best unimodal model because modalities overfit at
   different rates. Huang et al. give a complementary optimization account: a branch with a small
   early advantage can become the winning modality while the other remains under-trained. Peng et
   al. likewise measure dominant-modality suppression and modulate optimization from each branch's
   current contribution. This closely matches HALO's full classifier falling below its support-only
   readout; more evidence and more capacity do not guarantee a better decision.
2. **Adaptive semantic/support fusion is established.** AM3 treats visual examples and label
   semantics as independent evidence sources and learns a category-conditioned mixture rather than
   forcing them into one aligned representation. Its gains are largest in the lowest-shot regime.
   SEGA takes an even more conservative visual-dominated approach, using semantics to select or
   reweight visual features. These results support preserving the support path and using semantics
   adaptively, not allowing semantic context to rewrite it unconditionally.
3. **Set contextualization is useful, but it is not itself a safeguard.** Matching Networks support
   soft attention over all demonstrations, and FEAT shows that a permutation-invariant Transformer
   can adapt embeddings to the current candidate task. Neither result implies that arbitrary
   contextualization must outperform its unmodified metric input. HALO therefore needs the stable
   status-quo paths and explicit non-regression measurements around its contextualizer.
4. **Routing should be trained from expert competence.** Learning-to-defer work formulates routing
   around the probability that an expert is correct, and develops calibrated losses rather than
   asking a generic fused predictor to discover routing indirectly. HALO's support and semantic
   paths are not human experts, but the transferable principle is direct supervision from observed
   branch competence. This is stronger and more identifiable than the existing best-path penalty
   alone.
5. **Related episodes are a recognized fix for brittle episodic learning.** MELR deliberately
   constructs two episodes with the same classes but different instances, then enforces
   cross-episode consistency to reduce sensitivity to poorly sampled supports. Meta-augmentation
   work separately shows that task-level variation is needed to prevent a meta-learner from solving
   training tasks while ignoring the adaptation mechanism. HALO's paired support intervention is a
   narrower, deployment-grounded form of the same idea: hold query and candidates fixed, vary one
   support condition, and supervise the resulting change in expert preference.

### Primary sources

- Chen Xing et al., [Adaptive Cross-Modal Few-shot Learning (AM3), NeurIPS
  2019](https://proceedings.neurips.cc/paper/2019/hash/d790c9e6c0b5e02c87b375e782ac01bc-Abstract.html).
- Oriol Vinyals et al., [Matching Networks for One Shot Learning, NeurIPS
  2016](https://proceedings.neurips.cc/paper/2016/hash/90e1357833654983612fb05e3ec9148c-Abstract.html).
- Han-Jia Ye et al., [Few-Shot Learning via Embedding Adaptation With Set-to-Set Functions
  (FEAT), CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.html).
- Fengyuan Yang et al., [SEGA: Semantic Guided Attention on Visual Prototype for Few-Shot
  Learning, WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Yang_SEGA_Semantic_Guided_Attention_on_Visual_Prototype_for_Few-Shot_Learning_WACV_2022_paper.html).
- Weiyao Wang et al., [What Makes Training Multi-Modal Classification Networks Hard?, CVPR
  2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html).
- Yu Huang et al., [Modality Competition: What Makes Joint Training of Multi-modal Network Fail
  in Deep Learning? (Provably), ICML
  2022](https://proceedings.mlr.press/v162/huang22e.html).
- Xiaokang Peng et al., [Balanced Multimodal Learning via On-the-Fly Gradient Modulation, CVPR
  2022](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html).
- Hussein Mozannar and David Sontag, [Consistent Estimators for Learning to Defer to an Expert,
  ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html).
- Rajeev Verma et al., [Learning to Defer to Multiple Experts, AISTATS
  2023](https://proceedings.mlr.press/v206/verma23a.html).
- Nanyi Fei et al., [MELR: Meta-Learning via Modeling Episode-Level Relationships for Few-Shot
  Learning, ICLR 2021](https://openreview.net/pdf?id=D3PcGLdMx0).
- Janarthanan Rajendran et al., [Meta-Learning Requires Meta-Augmentation, NeurIPS
  2020](https://proceedings.neurips.cc/paper/2020/hash/3e5190eeb51ebe6c5bbc54ee8950c548-Abstract.html).
- Beyza Ermis et al., [Towards Robust Episodic Meta-Learning, UAI
  2021](https://proceedings.mlr.press/v161/ermis21a.html).

## Decisions after research

The literature resolves the architecture direction but does not substitute for HALO-specific
ablations:

1. Use a **normalized two-expert mixture** as the primary combination, with one learned semantic
   reliance value per candidate. This makes the two paths and their routing auditable. Preserve the
   support-pair correction as a zero-initialized refinement inside the support expert; do not add an
   unrestricted third logit path.
2. Use **both** direct routing supervision from detached branch advantage and paired preference
   ranking. The direct term identifies the better expert in each view; the paired term teaches
   sensitivity to a controlled support change. The main final cross-entropy remains authoritative
   when both experts are wrong.
3. Start with **exactly two views** per selected base episode and exactly one support intervention.
   This is sufficient to identify the desired relation and avoids a combinatorial episode expansion.
   Make paired-episode probability and intervention distribution configurable.
4. Keep pairwise support-label/candidate compatibility in the vectorized support vote for the first
   implementation. Do not add an attention relation bias until an ablation shows the candidate
   evidence and contextual tokens are insufficient.
5. Do not add raw support count or effective sample size initially. Reconsider only if routing
   errors remain after evidence distributions and the direct-support indicator are available.
6. Build label-held-out development folds entirely from the eight training sources. Rotate canonical
   labels into development-only folds while preserving source and subject separation; sealed
   datasets remain untouched. Report both ordinary source-held-out validation and label-held-out
   routing diagnostics.
7. Do not adopt Gradient-Blending or on-the-fly gradient modulation in the first repair. Those
   methods validate the branch-competition diagnosis, but their repeated optimization estimates add
   machinery. Separate branch losses and competence telemetry provide the smaller first test.

## Implementation and experiment order

1. Run the artifact-only oracle branch-selection analysis and quantify headroom by `k`, dataset,
   and deployment scenario.
2. Add stable status-quo score computation and telemetry. Prove that supplying or permuting support
   tokens cannot change the pre-context semantic status quo.
3. Add evidence projections to existing support and candidate tokens, preserving permutation and
   instance-tag contracts.
4. Replace the current final combiner with the candidate-wise normalized two-expert mixture and
   retain the zero-initialized support correction.
5. Add branch-preservation, direct routing, and best-path guardrail losses behind independent
   configuration switches. Unit-test detach boundaries and zero-weight equivalence.
6. Add two-view counterfactual episode construction and paired routing preference loss. Unit-test
   that query/candidates are identical, only one declared support intervention changes, and shared
   encodings are reused.
7. Add the required telemetry and label-held-out development split, then run short overfit and smoke
   tests. Do not launch a full training run until support-only and semantic-only controls remain
   stable and routing beats both on a deliberately mixed synthetic batch.
8. Run one controlled ablation ladder: status-quo paths; mixture only; plus direct routing; plus
   paired episodes. Promote nothing unless the final model reduces regret to the better path on both
   ordinary and label-held-out development sets without degrading the support floor at higher `k`.

No training or evaluation result is authorized by this design record. A later implementation record
must name the selected decisions, tests, and exact protocol version.
