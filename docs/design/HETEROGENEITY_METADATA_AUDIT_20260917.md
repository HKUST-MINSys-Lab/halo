# Heterogeneity, metadata, and domain-shift audit

Date: 2026-09-17. Source revision: `a5e3e71`, with the unimplemented contextual-classifier plan
present in the working tree. Status: initial code/metadata/publication audit and experiment
separation decisions. No classifier, sampler, converter, checkpoint, or evaluation result changed.
No training or model-evaluation run launched. This is not a full checkpoint-exposure certification.

## 1. Decisions from the discussion

Separate the following questions rather than treating every difficulty as one augmentation recipe:

1. **Deployment heterogeneity:** incomplete or unequal enrollment; different subjects, placements,
   devices or datasets between query and support; and different target activity domains. These
   motivate the support-conditioned classifier and its curriculum.
2. **Measurement robustness:** rate/bandwidth changes, missing modalities, noise and justified
   coordinate changes. These are separate controlled augmentation experiments. A clean-condition
   cost for a difficult-condition gain can be acceptable, but both must be measured and disclosed.
3. **Metadata value and encoding:** establish what sampling-rate information, duration conditioning
   and acquisition text contribute, before replacing the language model or fusion method.
4. **Matched classifier:** later freeze each released baseline encoder, train the same HALO
   classifier with a declared training budget, and evaluate. Explicitly deferred; do not launch now.

The earlier zero-shot panels do not establish a robust positive or negative gate effect. Both used
one training seed but different episode panels. Training-seed variation is not measured by that
comparison. Repeated matched draws/seeds can resolve it later; do not infer a new design from it now.

## 2. Initial findings

### A. The current new-domain description overclaims label novelty (high confidence)

`training/support_classifier/run_scenarios.py` describes s6 as labels no model trained on, but
`NEW_DOMAIN_CELLS` is a dataset/stream list with no checkpoint-specific concept-exposure test.
Dataset novelty and unseen activity concepts are different properties.

A sub-second, standard-library-only check compared `data/datasets/*/eval_labels.json` through
`canonicalize` with the persisted 155-label `data/labels/global_labels.json` vocabulary:

| Registered dataset vocabulary | Labels | Exact/canonical matches to HALO vocabulary | Not matched by that dictionary |
|---|---:|---:|---:|
| MotionSense | 6 | 6 | 0 |
| RealWorld | 8 | 8 | 0 |
| Shoaib | 7 | 7 | 0 |
| InclusiveHAR | 6 | 4 | 2 |
| USC-HAD | 12 | 4 | 8 |
| UT-Complex | 13 | 8 | 5 |
| MM-Fit | 10 | 2 | 8 |
| SPAR | 7 | 0 | 7 |
| Upper Limb Use | 15 | 0 | 15 |

These are vocabulary counts, not scored-query counts, checkpoint exposure counts, or validated
novel-concept counts. The persisted vocabulary must be reconciled with the actual corpus and
each checkpoint's lineage before asserting what that checkpoint saw.

Concrete examples:

- MM-Fit's `pushups` and `situps` match training `push_up` and `sit_up`. The original
  [MM-Fit site](https://mmfit.github.io/) describes those exercises. MM-Fit is not uniformly
  unseen-label recognition for HALO even if its recordings are held out.
- Upper Limb Use's `typing_on_a_keyboard`, `writing_with_a_pen`, and `walking_25_metres` do not
  exactly match the dictionary but are clearly related to training `typing`, `writing`, `walking`.
  Protocol equivalence still needs review; string difference is not proof of conceptual novelty.
- USC-HAD's `walking_forward` and UT-Complex's `biking` illustrate vocabulary granularity/spelling
  differences. Do not mark these automatically novel or automatically equivalent.

Recommended correction: call s6 **activity-domain shift**, then attach per-model exposure
annotations. Distinguish known concept/new recording domain, related or finer-grained concept,
verified held-out concept, and unresolved exposure. Do not silently alter stored scenario IDs or
recompute old scores. The s6 source lists also include retired SPAR/Upper Limb Use entries; reconcile
the live scenario manifest and retirement policy before claiming all listed cells were evaluated.

### B. Cross-dataset evaluation tests shared concepts, not arbitrary label transfer

`scenarios.shared_candidates` intersects canonical concepts. `build_cross_manifest` selects real
support rows and records a source-to-query label map, but emits each support's presented label in
the **query vocabulary**. The current residual scorer further gathers candidate text by binding.

This is a defensible ontology-aligned comparison, not automatically a bug. It does mean that a
cross-dataset gain alone does not demonstrate a learned linguistic bridge between source label
descriptions and target descriptions. It is also not the same as injecting a training recording
and pretending it is a sample of an unavailable test activity.

For the new semantic-voting head, preserve both source label and mapped concept in the manifest;
do not lose the source wording before text encoding. Keep current ontology-aligned episodes as
the reproducible control and give any source-wording condition a distinct protocol version.
Support eligibility must depend on reviewed concepts, not whichever text embedding is closest.

Checks read as correct in this pass: same-dataset execution exclusions; offsetting support rows
when concatenating query/support features; refusing unsupported candidate concepts; and refusing
claims of same/different person across different datasets without an identity map. Disjoint dataset
names do not independently prove disjoint real people.

### C. USC-HAD uses an intentional deployment-role abstraction, but provenance is conflated

`deployment_policy.py` registers `usc_had/phone_hip` with `device_profile="phone"`, while its note
identifies MotionNode. `stream_sensor_texts` consumes that profile and describes a phone sensor.

The [USC-HAD paper, sections MotionNode and Data Collection](https://sipi.usc.edu/had/mi_ubicomp_sagaware12.pdf)
identifies a dedicated MotionNode IMU, connected by wire, placed in a phone pouch at the front-right
hip. The intended model-facing abstraction is nevertheless **phone-like deployment at the hip**:
common language such as "phone in a pocket" is preferred over an obscure hardware product name when
it describes the deployment role the model should generalize across. That is a defensible policy, not
an automatic metadata error.

The remaining issue is provenance: one `device_profile` field currently carries both physical
hardware and model-facing deployment semantics. Preserve two explicit facts in future metadata:
`hardware_kind/model` (MotionNode IMU) and `deployment_role` (phone-like device in a front-right hip
pouch). Generate model text from the declared deployment role, while retaining hardware truth for
audits, compatibility analysis and cache provenance. Historical artifacts must keep the description
policy/version that produced them; no existing cache should be silently reinterpreted.

### D. Realized heterogeneity is constrained by available labels and configurations

Reuse the existing [curriculum audit](../../training/support_classifier/evaluations/curriculum_audit_20260917.md),
not another long run: 2,000 support sets and 7,457 queries. Among enrolled sets, compatible,
cross-placement, and cross-dataset shares were 74.8%, 15.1%, and 10.1%. Fallback was 22.5% overall.
Cross-dataset draws covered three body-sensor datasets and five labels; cross-placement draws
covered four datasets. Corrected cross-placement sets contained no other-dataset supports.

This is useful but limited exposure, not broad coverage of every deployment configuration. Add
source-pair, concept, device, placement, subject-relation, C and realized-k counts to the audit
ledger. Report eligible and excluded counts. Avoid manufacturing questionable label equivalences
solely to increase diversity or satisfy a target sampling percentage.

## 3. What the metadata currently does

| Input | Current path | Suitable contribution test |
|---|---|---|
| Sampling/storage rate | Physical filterbank frequency grid and physical time calculations | Preserve truthful rates; measure robustness to correctly resampled observations |
| Effective source rate | Nyquist/observability support after resampling | Separate physical correctness from any optional learned observability-feature ablation |
| Patch duration | Determines physical features and resolution support; normalized log-duration MLP plus gated addition before temporal attention | Remove only learned duration conditioning while keeping identical patches/features to isolate its value |
| Sensor configuration text | Frozen MiniLM, masked mean pooling and L2 normalization, then learned MLP/LayerNorm and vector-valued gated addition to sensor tokens | Full text versus modality-only text versus conditioning disabled, with matched retraining |
| Label text | Frozen MiniLM sentence embeddings used by the classifier | A separate experiment; do not change label and acquisition language representations together |

Sources: `model/tokenizer/filterbank.py`, `encoder.py`, `channel_text.py`, `sensor_tokens.py`,
`training/tokenizer/pretrain_data.py`, `training/support_classifier/train.py`, `baselines/text.py`.
The four recent screen configs record `neutral_acquisition_text=false`, so full acquisition
descriptions were enabled. The current classifier does not receive an additional acquisition
descriptor token; its access is through the conditioned sensor representation.

Text implementation caveats for a future language-model comparison:

- Default backend is `all-MiniLM-L6-v2`, 384 dimensions. Acquisition text uses manually pooled
  transformer states; label text uses SentenceTransformer.encode. Verify equivalence at the
  default backend rather than assuming arbitrary replacement backends have the same pooling.
- `TokenTextEncoder` accepts a model name, but the sensor conditioner is constructed for 384
  dimensions. A 768-dimensional alternative is not currently a safe drop-in flag change.
- The frozen acquisition tower is excluded from the encoder state dict. Pin model revision,
  tokenizer, pooling, normalization, maximum length and descriptor-cache fingerprint for repeats.
- The neutral-text option retains modality while removing device, placement and gravity wording;
  it is not a pure all-text-off intervention. Use exact names for these different ablations.
- Current registry-based text has no intentional dataset-name or activity-label field, but
  configuration coverage can correlate with labels. Test text-only/metadata-only prediction on
  internal data and configuration-disjoint generalization before claiming semantic understanding.

First compare full text, neutral modality-only text, no learned conditioning, and a parameter-matched
non-linguistic configuration encoding. Then consider phrase paraphrases and one alternate language
backend. Hold the label-text tower fixed. A test-time metadata knockout measures reliance; only a
matched retraining comparison measures whether the metadata improved the learned model.
Gate magnitudes, vector norms, and gradients are diagnostics, not causal contribution measures.

## 4. Label and metadata audit completion criteria

Produce an inspectable table for each actually used query/support source pair:

`source activity/code -> source description -> canonical concept -> target activity/description`

Add relation (`equivalent`, `broader`, `narrower`, `related`, `ambiguous`), source publication/release,
device, placement/laterality, modalities, units, gravity convention, source rate, execution and
subject provenance, and number of eligible query/support executions. Attach metadata/hash versions.

Only confirmed equivalences become hard support-label bindings. Preserve broader/narrower and
related labels for explicitly declared semantic-transfer tests, not as interchangeable ground truth.
Do not map vague stairs to ascent or merge transitions with repeated transition bouts. Do not
merge labels only because they have high language-model similarity. Audit class collisions within
one target vocabulary instead of letting dictionary insertion order choose a target label.

Use corpus inventories and already-produced manifests first. This pass inspected representative
converters and public sources, not every publication, raw recording, or label definition; the
complete pair-level semantic ledger remains outstanding.

## 5. Baseline exposure: what is known and what remains unverified

Exposure must be tied to the **loaded checkpoint**, not a model-family name. Keep separate fields
for sensor pretraining, supervised adaptation, semantic-alignment training, prior benchmark use,
and unknown private/text pretraining. Match data copies, participants and releases where possible.
An author evaluating a benchmark is not proof that the released backbone trained on that benchmark.
Conversely, a held-out recording dataset does not imply its activity concepts were unseen.

| Model | Primary-source evidence inspected | Scope of defensible conclusion now |
|---|---|---|
| HARNet | [Official release documentation](https://raw.githubusercontent.com/OxWearables/ssl-wearables/main/README.md) lists UK Biobank pretraining and downstream WISDM/RealWorld benchmarks | Distinguish released pretraining trunk from downstream task heads; RealWorld was a known author benchmark, not a guaranteed untouched benchmark |
| UniMTS | [Official repository](https://github.com/xiyuanzh/UniMTS) specifies synthetic motion derived from HumanML3D and text alignment; real HAR datasets are downstream benchmarks | Different recording source is plausible; unseen target activity concepts are not established by source disjointness |
| LiMU-BERT-X | [Official Experience repository](https://github.com/WANDS-HKUST/LIMU-BERT_Experience) describes released pretraining on courier sensor data | Do not revive the retracted MotionSense/Shoaib-pretraining claim; private record/subject-level exposure cannot be fully audited from the release |
| NormWear | [Paper appendix A, table 5](https://arxiv.org/html/2412.09758v2) enumerates nine backbone pretraining datasets, including PPG-Dalia; native prediction also uses an alignment component | Audit backbone and MSiTF alignment checkpoint separately; backbone table alone does not certify the entire native predictor's exposure |
| HALO | Active eight-source roster, saved run configs, warm-start checkpoints, and any historical pretraining lineage | Dataset disjointness is verifiable locally only after traversing the actual selected checkpoint lineage; concept overlap already exists in the saved vocabulary |

This initial source review found no evidence sufficient to declare a new baseline leakage event.
It also does not certify no overlap. Use `unknown` where releases lack evidence; do not label
unverifiable exposure as clean. Language models can know activity words without having seen sensor
examples, which is different from sensor-training leakage.

Retain the six main test datasets as useful heterogeneous HAR evaluations for now. Reframe their
claims rather than replacing datasets just because activities overlap. Dedicated unseen-concept
analysis needs a predeclared, reviewed concept-family split and a per-model exposure ledger.
Existing sealed results already informed design; disclose that history and do not present a newly
selected subset of those same results as a fresh untouched confirmatory benchmark.

## 6. Nuisance experiment rules

- Channel storage order with correct axis/modality identifiers is a loader canonicalization test.
  It should not need learned invariance. The current sensor fold assumes xyz order.
- Swapping numeric x/y/z while leaving identifiers unchanged is a changed coordinate convention
  or corrupt metadata, not just reordering. Apply only justified joint triad transforms; not every
  permutation is a physical proper rotation. Preserve accelerometer/gyroscope consistency.
- Rate downsampling must use anti-aliasing and truthful source bandwidth. Upsampling does not
  simulate a genuinely higher-bandwidth acquisition. Never falsify rate metadata as an ablation
  of physical preprocessing correctness.
- Missing modalities genuinely remove information. Train/test a small explicit severity list,
  record whether the affected modality existed, and use model-specific unsupported status where
  the released input contract cannot ingest that condition.
- Scale additive noise relative to physical units or train-derived signal statistics, with stated
  sensor assumptions. Avoid dataset-tuned amplitudes selected from test scores.
- Measure clean score, perturbed score and paired delta on the same queries/supports; report each
  rather than hiding a tradeoff in one aggregate. No full Cartesian perturbation grid by default.

## 7. Sequence of work

1. Complete the high-priority concept-map/device-metadata/exposure audit before modifying the
   classifier's input semantics or claiming OOD performance. Correct confirmed metadata issues
   with provenance/cache invalidation in a separate authorized implementation change.
2. Implement and test the approved contextual classifier on the corrected curriculum, without
   adding nuisance augmentation or changing the text backend simultaneously.
3. Isolate metadata contributions with matched internal experiments, then separate nuisance
   transforms. Repeated paired seeds quantify uncertainty; no automatic long runs.
4. Later perform the matched-classifier comparison: freeze encoder, train adapters/head on
   training-only data, same candidate/support protocol, common internal checkpoint rule and
   declared steps plus data exposure. Keep all native/existing baseline reference rows.

## Reproduction of the lightweight vocabulary check

From the repository root, this reads only JSON vocabulary files and the canonical synonym map:

```python
import json
from pathlib import Path
from data.scripts.labels.canonical_labels import canonicalize

root = Path("data")
train = set(json.loads((root / "labels/global_labels.json").read_text())["labels"])
for name in ("motionsense", "realworld", "shoaib", "inclusivehar", "usc_had",
             "ut_complex", "mmfit", "spar", "upper_limb_use"):
    labels = json.loads((root / "datasets" / name / "eval_labels.json").read_text())["labels"]
    matched = [label for label in labels if canonicalize(label) in train]
    print(name, len(labels), len(matched), len(labels) - len(matched), matched)
```

No raw signal array, model weights, train/test embedding, or GPU computation was read by this check.
