# Deployment-heterogeneity scenarios — evaluation plan (2026-09-15)

**Status:** implemented, repaired after review, and bounded integration-tested on HALO and every
primary baseline. The full comparison has not been run. See
[the review acceptance record](DEPLOYMENT_SCENARIOS_REVIEW_20260915.md#execution-and-acceptance-checklist).
Written after the 2026-09-15 supervisor meeting. Companion to
`ABLATION_PLAN_20260915.md`: that plan explains *why* we lead once we know *where*; this plan finds
where. Nothing here trains a model. Every scenario is evaluation-only on existing checkpoints,
and the executing agent must not modify `docs/journal/`.

## 0. Why this exists

On the current protocol all five models are asked to do the one thing every foundation model is
built for — nearest-neighbour over a good embedding on a fully enrolled, same-dataset, same-device
support set — and the aggregate lines sit within a few points of each other. Our design decisions
were never about that cell. Each of them is about what happens when information is **removed** at
deployment. So the evaluation has to remove information, one axis at a time and then together, and
report a **degradation curve** per model. The contribution is the area between curves.

## 1. The four heterogeneity axes and their severity rungs

Every scenario below is placed on all four axes. Rung 0 is the current sealed protocol.

| axis | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **L** label domain | seen concept, seen phrasing | seen concept, new phrasing | new concept, familiar domain | new domain (rehab, gym, ADL, clinical tasks) |
| **S** support coverage | every candidate has k supports | some candidates unsupported; true label supported | some unsupported; **true label unsupported** | no supports (k=0) |
| **P** support provenance | same dataset, same placement, other subject | same dataset, other placement, same subject | same dataset, other placement, other subject | other dataset, mapped label (and therefore other device, rate, protocol) |
| **C** sensor configuration | same channels, rate, device set as support | modality missing at query (accel-only) | rate mismatch | device-set mismatch (query multi-device, support single, or reverse) |

A scenario's severity is written as `L/S/P/C`, e.g. `0/2/0/0`.

## 2. Ground truth about the data we can use

Verified on disk 2026-09-15 (`data/datasets/`, `deployment_policy.STREAM_SPECS`, per-dataset
`labels.json`):

- **Sealed six** (gridded, in the current protocol): motionsense (front pocket), realworld (waist,
  forearm, thigh; 7 sites in the raw download), shoaib (left pocket, right pocket, belt, wrist-proxy;
  5 sites in the raw download), inclusivehar (waist), usc_had (hip), ut_complex (a phone strapped
  to the wrist, not a smartwatch).
- **Converted, in no training set of ours and, as far as we can establish, of no baseline:** spar (7
  shoulder-rehab exercises, left+right wrist watches), mmfit (10 gym exercises; left wrist, right
  wrist, right pocket, left earbud — synchronised), upper_limb_use (15 ADL tasks, wrist), monipar
  (Parkinson's assessment tasks, wrist), phytmo (physio exercises with correct/incorrect variants,
  8 limb sites), kneepad (knee-rehab, 8 thigh/shank sites), opportunity (locomotion + 5 sites).
- **Label overlap across the sealed six** (for cross-dataset enrolment): {sitting, standing, walking,
  jogging/running, walking_upstairs, walking_downstairs} is shared by motionsense, shoaib,
  ut_complex, usc_had (as walking_forward / running_forward), realworld (climbingup/down, running).
  Canonicalisation already exists (`data/scripts/labels/canonical_labels.py`).
- **Not on disk, not needed:** OpenPack, OCA, C-MHAD. Nothing in this plan requires a download.
- **New-domain grids ready:** mmfit, spar, and upper_limb_use at 4/8/16 s. RealWorld chest / head /
  shin / upper arm and Shoaib upper arm remain optional expansion cells and are not silently
  represented by another placement.

## 3. The eight scenarios

Each scenario states: severity, the deployment story in one sentence, data, construction, the fair
readout for baselines, what is reported, what we predict and **why** (the mechanism that would be
the contribution if the prediction holds), and cost. Cost assumes existing checkpoints and the
current evaluator (~18 min per model per full pass; scenario-specific passes are smaller).

### Scenario 1 — Enrol some, ask about all
**Severity `0/2/0/0`.** *A user enrols examples for the activities they can easily demonstrate and
asks the system to recognise a longer list.*

- **Data:** sealed six, existing grids and manifests.
- **Construction:** from each existing k-support manifest, hide all supports for a random subset of
  candidates. Two sub-cells per k ∈ {1, 4, 8}: (1a) true label among the supported candidates,
  (1b) true label among the **unsupported**. Coverage fraction 50 % of candidates, drawn with the
  same stable hashing the manifests use; the hidden set is recorded in the manifest so every model
  sees the same one.
- **Fair baseline readout:** unsupported candidates cannot be scored by 1-NN. Give every model with a
  text path (HALO, UniMTS, HARNet-bridge) a fixed **untrained hybrid**: per query, z-score the text
  scores over the candidate set and the 1-NN similarities over the supported subset, then take the
  arg-max over the union. Models without a text path (LiMU-BERT-X, NormWear, Mantis) are scored on
  1-NN over supported candidates only and their 1b cell is reported as "cannot attempt", not as a
  number.
- **Report:** macro F1 for 1a, 1b, and the union, per k, per dataset, plus the 8 s dataset-balanced
  aggregate; also the confusion mass flowing from unsupported-truth queries into supported
  candidates (the "false enrolment pull").
- **Prediction and mechanism:** large lead on 1b and a smaller one on the union. HALO's head was
  trained with the true label absent from the support half the time (`p_gt_present=0.5`) and with
  candidate supports randomly hidden; its λ-weighted text term is calibrated against its own vote.
  No baseline has a trained rule for mixing text and support evidence; the hybrid is the fairest
  untrained one. If the lead is small here, the "partial-information curriculum" clause is not a
  contribution.
- **Cost:** manifest variant + hybrid readout, ~1 day build; evaluation ~2 h all models.

### Scenario 2 — Enrol on one body site, deploy on another
**Severity `0/0/1–2/0`.** *The user enrolled with the phone in a pocket; today it is in a belt
holster or on the wrist.*

- **Data:** realworld (grid the remaining raw sites so it has waist, thigh, forearm, upper arm,
  chest, shin, head) and shoaib (left/right pocket, belt, wrist, upper arm). Both datasets record all
  sites simultaneously, so support and query are the *same physical execution* seen from two sites
  in the same-subject variant.
- **Construction:** manifest where support windows come from stream A and query windows from
  stream B, same dataset. Two sub-cells: (2a) same subject, execution-disjoint by time, P=1;
  (2b) other subject, P=2. Rank site pairs by anatomical distance: near (thigh→waist,
  pocket→belt), mid (waist→upper arm), far (wrist→thigh, head→shin). k ∈ {1, 8}.
- **Fair baseline readout:** 1-NN, unchanged; every model gets the same A/B pairs. UniMTS keeps its
  native orientation augmentation — this is the scenario it was designed for.
- **Report:** F1 versus anatomical distance, per model, as a curve; and the drop from the P=0 cell
  for the same (dataset, query site).
- **Prediction and mechanism:** honestly uncertain. Identity-as-text and multi-device training
  should make the encoder map "walking at the waist" and "walking at the wrist" near each other,
  but the curriculum has no cross-placement episodes, so nothing rewarded that explicitly. UniMTS
  may win at near distance. A loss here is informative: it names the missing curriculum term.
- **Cost:** gridding 5 sites ~half a day; manifest builder shares code with Scenario 3; eval ~3 h.

### Scenario 3 — Enrol from a public dataset, not from the user
**Severity `0/0/3/0` (implicitly C: other device model, other rate, other protocol).** *No on-device
enrolment at all: the deployer takes a public corpus with the same activities and uses it as the
support set.*

- **Data:** sealed six, pairwise, over the shared six-label locomotion set. Support dataset ≠ query
  dataset. Use the three same-region pairs to keep placement out of it: motionsense front pocket →
  shoaib right pocket; usc_had hip → realworld waist; shoaib wrist-proxy → ut_complex wrist-proxy.
  Then the same three with placement mismatch as a P=3 + C variant.
- **Construction:** candidate roster = intersection of canonical labels; supports drawn from the
  support dataset's execution-disjoint pool with the same k semantics; query dataset fully held out.
  k ∈ {1, 8, 32}.
- **Fair baseline readout:** 1-NN. Each baseline's adapter already handles rate and channel
  normalisation, so the scenario is about representation, not I/O.
- **Report:** F1 relative to the same query cell with in-dataset support (Scenario 0); the gap is the
  "corpus transfer cost" per model.
- **Prediction and mechanism:** moderate lead, mostly from the physical-Hz filterbank and gravity
  handling making the embedding depend less on the recording protocol. The centring step should
  help here specifically (SimpleShot's motivation is exactly cross-domain support shift).
- **Cost:** shares the cross-stream manifest with Scenario 2; eval ~2 h.

### Scenario 4 — Missing modality at deployment
**Severity `0/0/0/1`.** *Enrolled with accelerometer and gyroscope; the deployment device, or a
power-saving mode, gives accelerometer only.*

- **Data:** the five sealed datasets with a gyroscope (realworld is accel-only already and is
  excluded). Synthesis is channel dropping, which needs no justification.
- **Construction:** (4a) support full, query accel-only; (4b) support accel-only, query full;
  (4c) both accel-only (the reference for "how much was the gyro worth"). k ∈ {1, 8}.
- **Fair baseline readout:** 1-NN. Baselines receive the accel-only window through their existing
  pad-to-6-and-mask contract; HARNet is natively accel-only and unaffected.
- **Report:** F1 for 4a/4b relative to 4c and to the full-channel cell; the asymmetric loss (4a − 4b)
  is the interesting number.
- **Prediction and mechanism:** LiMU-BERT-X collapses (already seen in the failure analysis). HALO
  should lose little because a sensor is a token and its absence is a masked token with its identity
  text gone, not a zeroed channel. HARNet is the real competitor here.
- **Cost:** evaluator flag to drop gyro on one side; eval ~2 h.

### Scenario 5 — Sampling-rate mismatch
**Severity `0/0/0/2`.** *Enrolment at 50 Hz on the user's phone; deployment at 20 Hz on a
low-power wearable, or at 100 Hz on a research IMU.*

- **Data:** sealed six. Synthesis is anti-aliased resampling of the query grid to 20 / 25 / 100 Hz
  before the model's own input pipeline; supports stay native.
- **Construction:** k ∈ {1, 8}, three query rates. The manifest is unchanged; only the query tensors
  are resampled, so this is bit-comparable to the reference cell.
- **Fair baseline readout:** 1-NN. **Caveat to state in the results:** every baseline adapter
  resamples input to its trained rate, so for baselines the scenario measures information loss at 20
  Hz, not rate handling. For HALO the filterbank sits in physical Hz and the Nyquist mask adjusts.
- **Report:** F1 versus query rate, per model.
- **Prediction and mechanism:** small effect for everyone at 25 and 100 Hz; at 20 Hz the constant-Q
  bank above 10 Hz is masked and HALO should degrade gracefully rather than shift. This is the least
  likely scenario to produce a large lead; it is here because "sampling rate" is in the contribution
  sentence and must be measured, not asserted.
- **Cost:** small; eval ~1.5 h.

### Scenario 6 — New domain, new labels
**Severity `3/0/0/0`, then `3/2/0/0`.** *Deploy to rehabilitation, gym, or activities of daily
living, where the label set has nothing in common with locomotion.*

- **Data:** spar (shoulder rehab, wrist), mmfit (gym, use the left-wrist stream here), upper_limb_use
  (ADL, control-subject wrists only; the patient arms are a clinical contrast and stay out). All three
  are in no training corpus of ours and, by their release dates and modalities, in none of the
  baselines'. Grid at 8 s if not already.
- **Construction:** the standard sealed protocol at k ∈ {0, 1, 8}, execution-disjoint,
  cross-subject; then repeat with the Scenario 1 partial-coverage construction (1b) at k=8.
- **Fair baseline readout:** as Scenario 0 (k=0 uses each model's disclosed zero-support path); the
  hybrid for the 1b cell.
- **Report:** per dataset, per k, and the D1 seen/unseen stratification is moot here — everything is
  unseen, which is the point.
- **Prediction and mechanism:** at k=0 no confident prediction; HARNet's bridge and UniMTS's text
  head are real competitors and all three of us are extrapolating. At k=1 and at `3/2` HALO should
  lead, because that is where text and support have to cooperate and only HALO has a head trained
  to do so. The interesting finding is the *shape*: which model's curve bends most sharply from k=0
  to k=1.
- **Cost:** gridding ~half a day if needed; eval ~2 h.

### Scenario 7 — Device-set mismatch
**Severity `0/0/0/3`.** *Enrolled with one device; deployed wearing two or three — or the reverse,
enrolled from a multi-sensor lab session and deployed on a single phone.*

- **Data:** realworld (composite of waist+forearm+thigh, plus the newly gridded sites), shoaib
  (4-device composite), mmfit (wrist+pocket+earbud — a new domain, so also `3/0/0/3`).
- **Construction:** (7a) support single-device from site A, query the full composite; (7b) support
  composite, query single-device at site A; (7c) query composite with one device *added* that the
  support never saw. k ∈ {1, 8}.
- **Fair baseline readout:** the existing `per-device-pooled` rule (encode each device, average
  embeddings) for models without native fusion; UniMTS's native skeleton fusion where it applies.
- **Report:** F1 for 7a/7b/7c against the matched single-device and composite reference cells.
- **Prediction and mechanism:** clear lead on 7a and 7c. The learned recording pool attends over an
  arbitrary set of device-tagged tokens and was trained on random device subsets
  (`multi_device_probability=0.5`, up to 4). Per-device embedding averaging cannot weight devices
  by usefulness and UniMTS's fusion is bound to a fixed skeleton. (This is the clean version of the
  +11.1 / +8.9 result and it is the one the paper should report.)
- **Cost:** composite manifests already exist; A/B support-query pairing shared with Scenario 2;
  eval ~2 h.

### Scenario 8 — Cold-start user in a new domain (compound)
**Severity `3/2/2/3`.** *A new user starts a gym-tracking app on a watch and a phone. The app ships
one enrolment example per exercise, recorded by someone else on a single wrist device, for only
half of the exercises it recognises; the rest are recognised from their names.*

- **Data:** mmfit only. Support: one execution per supported label, from a different subject, from
  the right-wrist stream. Query: the left-wrist + right-pocket + earbud composite (or the largest
  composite the grids support) for a held-out subject. 50 % of labels unsupported, truth balanced
  across supported and unsupported.
- **Construction:** k=1 with the Scenario 1 hiding rule, the Scenario 2 cross-subject/cross-site
  pairing, and the Scenario 7 device-set mismatch, in one manifest. Also report the same manifest
  at k=4.
- **Fair baseline readout:** hybrid text + 1-NN with per-device pooling, i.e. the most generous
  untrained composition of every baseline's parts.
- **Report:** F1 on supported truth, unsupported truth, and union; plus each model's Scenario 0
  number on mmfit so the compound cost is visible.
- **Prediction and mechanism:** this is where the design should separate most, because every
  mechanism above is engaged at once and no baseline has more than one of them. If HALO does *not*
  lead here, the thesis is wrong and we should know that before writing.
- **Cost:** everything is shared with Scenarios 1, 2, 6, 7; eval ~1 h.

## 4. Severity summary

| # | scenario | L | S | P | C | new build | predicted lead |
|---|---|:-:|:-:|:-:|:-:|---|---|
| 1 | enrol some, ask about all | 0 | 2 | 0 | 0 | hidden-support manifests; hybrid readout | **large** (curriculum) |
| 2 | cross-placement enrolment | 0 | 0 | 1–2 | 0 | A/B stream manifests; 5 sites to grid | uncertain |
| 3 | public-corpus enrolment | 0 | 0 | 3 | 0* | cross-dataset manifests; label intersection | moderate (frontend, centring) |
| 4 | missing modality | 0 | 0 | 0 | 1 | gyro-drop flag | moderate (sensor-as-token) |
| 5 | rate mismatch | 0 | 0 | 0 | 2 | query resampling | small |
| 6 | new domain | 3 | 0→2 | 0 | 0 | grid spar / mmfit / upper_limb_use | k=0 unknown; k≥1 moderate |
| 7 | device-set mismatch | 0 | 0 | 0 | 3 | A/B + composite pairing | **large** (learned pool) |
| 8 | cold-start compound | 3 | 2 | 2 | 3 | composition of the above | **largest, or thesis fails** |

\* Scenario 3 changes device model, rate and protocol implicitly through the dataset change.

## 4b. Baseline readout prerequisite (completed)

NormWear previously fed its MSiTF text-alignment head where every other model was fed a
representation, understating its enrolled rows. The adapter now returns pooled backbone features,
its cache schema was bumped, and its native zero-shot path uses its published L1 scoring rule.
`NORMWEAR_READOUT_FINDING_20260915.md` preserves the diagnosis.

The other primary adapters retain their audited representation and native/bridge score paths; the
runner records model artifacts, hashes, parameter counts, and native capabilities in every run.

## 5. Build order and shared infrastructure

Everything reduces to one new manifest builder and two readout additions:

1. **Cross-stream manifests** (`build_manifest_cross(support_stream, query_stream, k, ...)`): support
   pool and query pool may differ in dataset, stream, and subject relation. Records provenance per
   support window. Serves Scenarios 2, 3, 7, 8.
2. **Support hiding** (`hide_candidates(manifest, fraction, seed)`): drops all supports for a stable
   random candidate subset; records the hidden set. Serves 1, 6, 8.
3. **Hybrid readout for text-path baselines** and a "cannot attempt" status for the others.
   Serves 1, 6, 8.
4. **Query-side perturbations**: gyro drop (4) and resampling (5). Applied to the query tensors
   after manifest construction, so manifests stay identical.
5. **Gridding**: spar / mmfit / upper_limb_use are ready at 4/8/16 s. RealWorld and Shoaib's extra
   raw placements are explicitly outside the current roster.

Order: 1 and 3 first (Scenario 1 is the cheapest large-lead candidate and needs no gridding), then
4 (Scenarios 4 and 5, same afternoon), then gridding, then 2 (Scenarios 2, 3, 7), then 8.

## 6. Reporting

One results file, `docs/design/DEPLOYMENT_SCENARIOS_RESULTS_20260915.md`, one section per scenario
in the order run, each with: the manifest fingerprint, the exact command, per-model F1 tables at 8 s
(4 s and 16 s in an appendix), the reference Scenario 0 cell for the same query data, and any
"cannot attempt" cells marked as such. Numbers and provenance only; no interpretation. The
degradation-curve figure (F1 versus severity, one line per model, one panel per axis) is produced
from that file afterwards.

## 7. What is deliberately not here

- Label **phrasing** shift (L=1): cheap and worth a supplement, but every text-path model would be
  hit the same way by a synonym table, so it is unlikely to separate us. Add later as a robustness
  check on whichever scenario we lead.
- Orientation perturbation: UniMTS's home ground, and our gravity referencing makes the outcome
  predictable in their favour under random rotation; the honest cell is Scenario 2, where placement
  changes orientation *and* dynamics.
- Same-subject enrolment as a headline (P=−1): it is a supplement, decided 2026-09-05.
- Anything that trains a model. Matched-corpus baseline retraining is the next plan, once we know
  which scenarios matter.
