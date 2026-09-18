# Evaluation-dataset expansion and text-pipeline audit

Date: 2026-09-18. Status: **dataset wiring implemented; MobiAct source archive still required**.
No MobiAct score exists. The historical sealed-six roster and all its aggregate means remain
unchanged. MobiAct is opt-in under a separate prospective scope, while MM-Fit is now wired only into
the multi-device/new-domain scenario paths using its publication split.

## 1. Decision summary

Use at most one new download and one already-held-out source:

1. **Add MobiAct as a prospective activity-domain/fall benchmark**, after acquiring and validating
   the official extended release. It supplies subject breadth and genuinely different events that
   the six locomotion-heavy test datasets do not.
2. **Promote MM-Fit from a scenario fixture to the named multi-device scenario benchmark**, using
   the publication's participant-disjoint split. It is already downloaded and materialized, and is
   the best local evidence for simultaneous phone/watch/earbud deployment changes.

This is preferable to adding two more conventional locomotion datasets. MobiAct and MM-Fit fill two
different gaps: activity-domain shift and synchronized configuration shift. Keep the current six
datasets and report every dataset separately; do not fold the additions into old means or relabel old
results as if they used the expanded protocol.

## 2. What the current roster does not measure cleanly

The sealed six are MotionSense, RealWorld, Shoaib, InclusiveHAR, USC-HAD and UT-Complex. They offer
useful subject, placement, rate and vocabulary variation, but most labels are locomotion/posture and
the apparent wrist devices include phone-on-wrist proxies. The scenario suite already references
MM-Fit, SPAR and Upper Limb Use, but SPAR and Upper Limb Use were retired from the live general-HAR
framing. The active protocol therefore lacks both:

- a strong prospective test of falls and short daily-life transitions; and
- a named, genuine, synchronized consumer multi-device benchmark with a defensible split.

Dataset novelty is not label novelty. Every added source must carry a concept-exposure ledger for
each loaded checkpoint, and results must distinguish new recording domain, related/finer concept,
verified unseen concept and unresolved exposure.

## 3. Candidate review

| Source | Primary-source facts | Value | Limitation | Decision |
|---|---|---|---|---|
| **MobiAct extended release** | 66 subjects, more than 3,200 trials, 12 ADLs, four simulated falls and a daily-living scenario; Samsung phone accelerometer/gyroscope in a freely oriented trouser pocket | Large subject pool; pocket-orientation variation; falls, car entry/exit and other events beyond the current locomotion core | Local `downloads/` is empty and the converter is explicitly unverified; existing 13-label metadata reflects an older/partial release; several trials are too short for 16 s | **Add after data validation** |
| **MM-Fit** | 809 minutes, 21 workout sessions, ten exercises, 616 sets/6,160 repetitions; synchronized two watches, phone and earbud | Best local cross-device and multi-device stress test; 4/8/16 s grids already exist | Ten people; release does not expose the full workout-to-person map. Arbitrary workout splits are not subject-disjoint | **Promote only under the paper split** |
| SP-SW-HAR / GeoTec TUG | 23 subjects, 223 repeated TUG executions, paired phone pocket and genuine watch wrist at about 100 Hz | Excellent paired consumer-device provenance and demographics | Transition runs are about 1 s and the current converter emits overlapping 1 s clips; it is incompatible with the common 4/8/16 s single-label protocol without changing the task | Keep as a bounded diagnostic, not a headline addition |
| PAMAP2 | Nine subjects, 18 activities, three 100 Hz body IMUs at wrist/chest/ankle | Public, downloaded, well documented | Small subject count, non-consumer body IMUs, and substantial overlap with placement/domain coverage already supplied elsewhere | Do not add |

Primary sources: [MobiAct project page](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/),
[original MobiAct paper](https://www.scitepress.org/PublishedPapers/2016/57924/57924.pdf),
[MM-Fit project](https://mmfit.github.io/), [MM-Fit paper](https://vradu.uk/UbiComp2021.pdf),
[SP-SW-HAR article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10700515/), and
[PAMAP2 repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring).

## 4. Exact expansion protocol

### 4.1 MobiAct

Before registration:

1. Fetch one identified official release and store its paper/site/licence beside the converter.
2. Replace the phantom grid and regenerate metadata from raw trial files. Verify units, real clocks,
   gravity, gyro availability, subject IDs, execution IDs, labels, missingness and duplicate trials.
3. Freeze native source labels from that release. Reconcile the current 13-label file with the
   extended release's 12 ADLs plus four falls; do not silently mix releases.
4. Reserve subjects before any score is produced. Query/support must be subject- and
   execution-disjoint. Supports are examples supplied by the protocol, never training examples
   mislabeled as unavailable target activities.
5. Materialize 4/8/16 s grids, then freeze each duration's candidate set to labels with real,
   complete windows in both participant partitions. A duration is unavailable where fewer than two
   labels survive; never repeat/pad a short fall or transition to manufacture coverage.
6. Report common-activity and fall/transition subsets separately, plus per-class support capacity.
   Do not use the fall subset to tune the classifier and then call it sealed.

MobiAct should enter as a new protocol version. Existing checkpoints may be evaluated only if their
source lineage proves no MobiAct sensor exposure; new HALO comparison runs must retain that exclusion.

### Implementation command

After the official annotated archive is present, run:

```bash
python -m data.datasets.mobiact.setup \
  --archive /absolute/path/to/MobiAct_Dataset_v2.0.zip
```

The setup refuses MobiFall, validates every annotated CSV and clock, anti-aliases to 50 Hz, preserves
only real complete windows, reserves participant-disjoint reference/query subjects before any score,
refreshes the quality screens, and writes a source hash plus duration-specific capacity ledger.
Use `training.support_classifier.sealed_eval --scope prospective` for the resulting separate table.
Use `training.support_classifier.run_scenarios --include-prospective-mobiact` only when adding the
MobiAct new-domain row to a separately named scenario run. Neither flag can alter the sealed-six mean.

### 4.2 MM-Fit

Use the paper split already recorded by the converter: train workouts 1,2,3,4,6,7,8,16,17,18;
validation 14,15,19; seen-person test 9,10,11; participant-disjoint test 0,5,12,13,20. Headline
queries come only from the participant-disjoint test workouts. The support pool may use the paper
train/validation workouts, with execution exclusions and fixed manifests.

Report MM-Fit as a **scenario benchmark**, not in the six-dataset mean. Include:

- single-device left-watch, right-watch, right-pocket and earbud rows;
- same-execution cross-device support/query rows where the clock/event map proves simultaneity;
- multi-device query and support rows; and
- 4/8/16 s rows, with identical query windows across compared models.

Disclose that workout IDs are not participant IDs and that only the publication's split certifies
the held-out people. Do not use arbitrary local subject splits for a cross-subject claim.

## 5. Current text pipeline

There are two distinct language paths.

### 5.1 Acquisition descriptions

`stream_sensor_texts` creates axis-role strings and one sensor string per present modality. The
sensor string contains deployment device, modality, placement, gravity convention and paired-
modality presence. Frozen `all-MiniLM-L6-v2` produces a 384-dimensional descriptor. For sensor
tokens, `ConditioningProjection` applies a learned MLP, LayerNorm and vector gate before residual
addition. Duration has its own learned gated scalar-to-vector path; source rate separately controls
physical filterbank observability.

Using a common word such as **phone** for a phone-like deployment is intentional. The registry should
nevertheless preserve both `hardware_kind/model` and `deployment_role`, and model text should be
generated from a versioned policy over the latter. That makes an abstraction auditable rather than
mistakable for a hardware assertion.

### 5.2 Activity-label descriptions

The active path replaces underscores and calls `SentenceTransformer.encode(...,
normalize_embeddings=True)` on the bare native/canonical label. Training builds one cached table;
evaluation rebuilds it over the candidate set. A separate paraphrase/description module exists, but
the active `make_label_text` path does not call it. This is currently a bare-label protocol, not a
paraphrase-robust protocol.

The residual-v3 classifier exposes text to attention through a fixed random orthogonal projection.
Its direct semantic score is still `cos(p_text(raw_query), raw_candidate_text)`: post-attention
candidate states do not define that cosine. Attention affects support scalar corrections and a
candidate scalar residual. Therefore the implementation does **not** yet realize the approved
contextual semantic-voting design in `CONTEXTUAL_CLASSIFIER_PLAN_20260917.md`.

## 6. Measured text checks

A local CPU probe compared acquisition `TokenTextEncoder.encode_pooled` against
`SentenceTransformer.encode(normalize_embeddings=True)` for three representative strings. Shapes
were `(3,384)`, diagonal cosine was 1.0, and maximum absolute element difference was exactly 0.0.
There is no pooling mismatch for the current MiniLM backend.

The same probe demonstrates why free text should not be the sole carrier of exact configuration:

| Contrast | MiniLM cosine |
|---|---:|
| phone vs watch, same wrist sentence | 0.9203 |
| pocket vs wrist, otherwise same | 0.9260 |
| gravity present vs removed | 0.9752 |
| paired gyro present vs absent | 0.9785 |
| phone vs generic device | 0.9531 |

Across the active train/test registries there were 74 sensor rows but only 57 exact description
strings. Very high non-identical pairs were dominated by left/right variants (about 0.998). This
semantic smoothing may help transfer, but it is not a reliable encoding of laterality, hardware or
observability. Those facts must stay structured and physically enforced where correctness depends
on them.

Bare activity labels also collapse directional opposites: the highest within-dataset pair was
upstairs/downstairs 0.8678, climbing-up/down 0.8631, ramp-ascent/descent 0.8974, and elevator-up/down
0.8707. General-purpose sentence similarity recognizes relatedness, not necessarily opposition.
MiniLM's official model card describes a general sentence/paragraph similarity encoder trained on
over one billion sentence pairs, not a sensor ontology. This makes it a reasonable small frozen
baseline, not evidence that its raw geometry is optimal for these attributes. The official
Sentence-Transformers documentation describes `all-mpnet-base-v2` as higher quality and MiniLM as
about five times faster, but explicitly recommends task-specific experiments.

Sources: [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[MPNet model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/main/README.md),
and [Sentence-Transformers model guidance](https://github.com/UKPLab/sentence-transformers/blob/04ae2e06/docs/sentence_transformer/pretrained_models.md).

## 7. Text-pipeline findings

### Sound today

- Both active paths use the same frozen backend, mean pooling and L2 normalization.
- The current manual acquisition pooling is exactly equal to the library path in the measured
  default backend.
- Text embeddings are cached, so the frozen LM is not on the steady-state training hot path.
- Acquisition conditioning is gated and residual rather than replacing sensor evidence.
- Eval-specific paraphrase leakage is guarded in the dormant paraphrase tooling.

### Needs correction before a language-backend experiment

1. `TokenTextEncoder` advertises a configurable backend, but the encoder constructs 384-wide
   conditioners and heads. A 768-wide MPNet backend is not a safe flag change.
2. The frozen text tower is omitted from state dicts without pinning model revision, tokenizer,
   pooling/max-length contract or embedding-cache fingerprint. Exact run reproduction is weaker
   than the checkpoint metadata implies.
3. Hardware provenance and deployment-role language share one field. Split them, while retaining
   the intended common-language abstraction.
4. Bare labels are brittle to synonyms and highly similar for antonyms. The dormant paraphrase
   code is not proof of robustness because it is not active.
5. Cross-dataset manifests currently rewrite support labels into the target vocabulary. This is a
   valid ontology-aligned control but does not test whether source wording bridges to target wording.
6. Residual-v3 does not compare post-attention query/candidate states in its semantic path. The
   already-approved contextual classifier plan is the targeted architectural correction.

## 8. Minimal experiment plan

Keep dataset expansion and text changes in separate commits/runs.

### T0: contracts, no training

- Add a versioned descriptor ledger with hardware provenance, deployment role and exact generated
  text. Hash it into feature-cache provenance.
- Pin the text model repository/revision, tokenizer, max length, pooling and normalization.
- Make text dimension dynamic from the loaded backend; reject mismatched checkpoints explicitly.
- Add parity, descriptor-collision and source-label-preservation tests.

### T1: acquisition-text contribution, matched retraining

Train four otherwise identical arms on internal train/validation data:

1. current full deployment-role text;
2. modality-only neutral text;
3. no learned acquisition conditioning; and
4. parameter-matched structured categorical metadata (device role, placement, gravity, modality).

Add one focused fifth arm only if needed: hardware-truth wording versus common deployment-role
wording. Compare clean and cross-placement/device scenario deltas. This directly tests the user's
intended abstraction instead of deciding it by taste.

### T2: label wording, frozen encoder/head budget

Compare bare native labels, one uniform grammatical template, and a train-only paraphrase ensemble.
Apply the same candidate text to HALO and semantic baseline paths. Keep eval-native strings frozen;
do not hand-author test-specific descriptions after seeing scores. Add a distinct source-wording
cross-dataset protocol rather than replacing the ontology-aligned control.

### T3: one language-backend comparison

After T0, compare MiniLM against only `all-mpnet-base-v2`, with the frozen tower precomputed and the
same D-dimensional trainable adapter/head capacity. Judge by per-dataset macro F1, balanced accuracy,
semantic-only k=0, enrolled curves, paraphrase stability and the directional-label collision panel.
Do not change the acquisition and label backends in the same first ablation.

### Required diagnostics

- metadata-only prediction and text-shuffle controls for shortcut detection;
- full/neutral/no-text paired deltas by scenario and dataset;
- source-label versus mapped-label sensitivity;
- label nearest-neighbour/collision report, especially directional opposites;
- gate/projection gradient norms and embedding norms; and
- repeated matched seeds with confidence intervals before a design claim.

The next implementation step should be T0 plus MobiAct acquisition/validation. Do not run model
evaluation on MobiAct until its release, converter, splits, labels and duration coverage are frozen.
