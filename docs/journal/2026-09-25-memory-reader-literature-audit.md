# Memory-reader literature audit: storage and token contract

Date: 2026-09-25. Status: **research addendum; design not implemented or evaluated**.
Refines the storage, tokenization and reader requirements in the earlier
[deployment memory-reader proposal](2026-09-25-deployment-memory-reader-proposal.md). The proposal
remains outside the registered three-rung paper protocol. This entry does not change rung 1.

## What the papers actually use

- [AdaNPC, ICML 2023](https://proceedings.mlr.press/v202/zhang23am.html) keeps feature/label pairs
  and adds a test feature with its **predicted** label after prediction. This supports causal
  predict-then-insert memory, but its pseudo-label is not verified evidence.
- [TDA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Karmanov_Efficient_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2024_paper.html)
  keeps feature-key/pseudo-label-value caches. Its positive and negative caches use different
  confidence conditions. A full probability/logit vector can preserve both positive and negative
  evidence without requiring two physical caches; whether HALO can use negative evidence safely
  is an empirical question.
- [LaplacianShot](https://arxiv.org/abs/2006.15486) exploits **relations among unlabelled
  features**, not only each feature's direct similarity to the query. A reader that receives only
  per-entry query similarities and pseudo-label votes may miss local density and disagreement.
- [RoTTA, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yuan_Robust_Test-Time_Adaptation_in_Dynamic_Scenarios_CVPR_2023_paper.html)
  uses timeliness and uncertainty under correlated streams. Its category balancing uses predicted
  classes, which may propagate prediction bias; HALO should compare this with acquisition and
  motion-diversity eviction rather than assume one policy works.
- [Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html) motivates
  permutation-invariant processing of entries and offers efficient set attention when a bank
  grows. It does not establish that three tokens per recording are sufficient.

## Revised bank schema

The minimal persistent record is:

```text
motion_vector                  current checkpoint's pooled recording representation
acquisition_vector             current checkpoint's per-recording acquisition summary
source_acquisition_metadata    modalities, effective rates, device/placement fields for audit
base_semantic_logits           original unnormalised scores on the admission roster
admission_roster_ref           ordered label descriptions and scorer text embeddings
evidence_source                verified | model_prediction | no_label_evidence
verified_label_ref             present only if confirmed, including labels outside today's roster
observed_at                   deployment observation index
recording_id, execution_id     deduplication and leakage checks
model_version, calibration_id  invalidate or re-encode incompatible cached entries
```

Keep the full original logits, not only an argmax, entropy, or probability-weighted text vector.
Retain calibration parameters with their version; probabilities and uncertainty are derived on
read. Entropy is a measure of score concentration, **not** automatically the probability that a
prediction is correct. A calibration fit from development data can help but must be checked after
shift ([Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)).

For a changed candidate roster, use the frozen checkpoint's semantic projection and stored motion
representation to score the **new** label descriptions. Keep the admission roster and logits for
provenance and comparison. Do not map an old roster's normalized probabilities onto new labels.
The current HALO v4 text path scores projected recording features against candidate embeddings;
the proposal must test that its exact chosen no-memory path is reproducible from the stored
representation before claiming roster-change support. If the required inputs are not cached,
retain the recording for re-encoding or declare that roster change unsupported. Checkpoint changes
always require re-encoding or discarding entries.

An entry with no trusted in-roster label remains useful as an unlabelled motion point. It must not
be forced into the nearest candidate merely to produce a vote. A verified enrollment may have an
outside-roster label and still contribute to geometry, with zero direct in-roster label mass.

## Three tokens, with a full-data sidecar

Keep the three-token starting point. The motion token represents the stored recording feature;
the acquisition token represents its separate acquisition summary; the label-evidence token
contains a compact summary of the **current-roster** score distribution, its entropy/margin,
evidence source, and age/availability features. A verified label uses a distinct provenance role.
Role embeddings and an entry-group relation must be permutation equivariant across entries.

The label-evidence token is a summary only. The full `(entry, candidate)` score distribution
and valid-label mask remain sidecar tensors used by voting and by candidate-specific readout.
One weighted average of label embeddings cannot preserve two competing hypotheses or absent
roster classes. Padded candidates and unavailable labels need explicit masks; zero vectors alone
would confuse missing information with low evidence.

Unlabelled adaptation requires a path from **relations within memory** to the decision. In the
first reader, compute masked, causal neighbourhood features at read time: query-to-entry
similarity, local embedding density, agreement/disagreement among nearby entries' original
semantic distributions, and query-to-entry acquisition mismatch. Feed these as features to the
entry reliability module alongside the three tokens. Use only earlier entries; exclude duplicate
executions and handle banks with fewer than two distinct entries. This keeps a fixed number of
stored tokens per entry while testing whether geometry helps beyond repeating semantic guesses.
Memory-to-memory attention is a later alternative if fixed neighbourhood summaries prove too
restrictive; its cost and shortcut risks must be measured against the fixed vote.

## What to verify before implementation is considered successful

1. One checkpoint can reconstruct each stored admission-roster distribution and re-score a new
   roster from the cached feature within numerical tolerance; changing checkpoint invalidates it.
2. Full distributions preserve uncertain/negative evidence. Compare full, hard-argmax and
   verified-only values; measure calibration and errors from wrong confident entries.
3. Permuting memory entries leaves predictions unchanged within floating-point tolerance;
   changing relative age or provenance changes the appropriate reader input.
4. On the same causal stream and memory, compare no memory, fixed similarity vote, reader without
   neighbourhood features, and reader with them. In the unlabelled-only case, gains over the
   fixed vote must survive class imbalance and misleading-neighbour controls.
5. Measure memory-size and roster-size scaling, as well as multi-device cases where one pooled
   acquisition vector may lose device-set detail. Add more acquisition tokens only if that failure
   is observed.

The evidence supports feature keys, uncertain label values, causal age, provenance, and local
geometry as ingredients. It does **not** show that a three-token learned reader will outperform
the fixed vote on HALO's IMU streams. That remains the experiment's central falsifiable question.
