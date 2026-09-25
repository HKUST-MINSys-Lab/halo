# Evaluation protocol: support-conditioned HAR

> Last verified against code: 2026-09-24. A score is reportable only when the run records the data split,
> encoder checkpoint, baseline adapter revision, episode manifest, and support/candidate settings.

> **Operational status:** evaluation is organised by the three rungs of the
> [roadmap](../overview/roadmap.md), all under `evaluation/`, sharing one feature-cache, manifest,
> zero-shot and provenance implementation (`evaluation/{features,manifests,zero_shot,provenance,metrics}.py`):
>
> | rung | runner | entry point | status |
> |---|---|---|---|
> | 1 — unlabelled adaptation | `evaluation.rung1_unlabeled.run` | `halo-rung1` | built, not run |
> | 2 — labels, frozen (this document's sealed and scenario protocol) | `evaluation.rung2_frozen.sealed_eval`, `.run_scenarios`, `.run_partial_coverage` | `halo-sealed-eval`, `halo-scenarios` | the promoted results |
> | 3 — labels, fine-tuning | `evaluation.rung3_finetune.run` | `halo-rung3` | built, not run |
>
> Every rung writes artifacts through `evaluation.provenance.write_artifact`, which refuses a row
> without a registered rung, method and readout version. The rung-2 sealed runner materializes
> deterministic execution-disjoint manifests, caches each representation with a checkpoint/artifact
> fingerprint, and writes machine-readable and Markdown per-stream tables. Internal validation from
> `training.support_classifier.train` is not a sealed result.

Rung 1 splits scored windows and unlabeled-pool windows by physical execution within each sealed
cell. It does **not** split these two sets by subject; the runner reports their subject overlap.
Its scored-set macro-F1 uses the complete declared candidate roster as the class set at every N,
and the balanced-pool control keeps N fixed while excluding labeled supports. Historical rung-1
artifacts made before this contract should not be pooled with new N-curves.

## Data separation

1. Split subjects before constructing recordings or episodes.
2. Keep every recording, session, and source segment assigned to one split only.
3. There are two disjoint active source roles: supervised support-classifier training and sealed
   test. There is no separate development-source roster. Subject-held-out folds formed only from
   the eight supervised training sources may select the support-classifier checkpoint.
   The internal splitter keeps at least one training subject for every label. A label observed in
   only one training subject remains optimizer-only rather than moving wholly into validation; it
   is not included in the internal model-selection score unless a subject-disjoint validation
   example exists.
4. Temperatures, candidate distributions, and every other operating decision are fixed a priori
   or read from training data. Touch the sealed test sources once, after those choices are frozen.

The exact active 8/6 source lists are recorded once in
[design_of_record.md](design_of_record.md) and enforced in code by roster tests.

## Episode contract

An episode contains a query recording, a bounded candidate set, and, when `k > 0`, exactly `k`
labelled support recordings per candidate. The candidate count `C`, support count `k`, source
dataset, subject relationship, and recording lengths are part of the manifest. At `k = 0` there is
no target-dataset support bank; the fixed training-corpus reference bank described below remains
available to methods whose declared zero-target-enrollment readout uses it.

Report the preregistered support curve at `k = 0, 1, 2, 4, 8, 16, 32, 64, 128`, rather than
selecting one favorable enrollment count. The 2026-09-12 materialized sealed grids can form every
point for every valid query while excluding the query's physical execution from enrollment;
the manifest builder continues to omit and report a query if a later grid revision cannot do so.
Large candidate-set episodes are required because a small `C` is an easier task; the held-out test
set determines the deployment candidate set.

The complete curve is repeated at **4, 8, and 16 seconds**. Query and support units use the same
physical duration within a cell. The evaluator constructs and fingerprints the raw native-rate
window set and every query/support manifest before loading a model. All providers therefore receive
the same event rows and valid samples; only their published input conversion may differ. A partial
tail remains a partial tail and is identified by its valid length rather than treated as measured
padding.

NormWear's released CWT stack takes two finite differences before a nine-frame patch kernel. For a
retained tail shorter than 11 samples at its 65 Hz model clock, its adapter repeats the final
measured value to reach that architectural minimum. The original valid length remains in the
manifest and source fingerprint; the extension contributes no additional motion.

RealWorld and Shoaib additionally report every declared placement separately and one preregistered
all-device cell. Composite rows require elementwise-identical event, label, subject, and execution
identities. A row rejected by any member's quality screen is removed from the composite for every
model. The ordered device list and exact raw-slice fingerprint are stored in the manifest.

## Compared methods

For each eligible representation, report these readouts on exactly the same episodes:

1. **Equal-weight normalized fusion:** the common primary adaptation rule for external encoders.
   It combines each model's declared native-or-ConSE semantic scores with class-wise maximum cosine
   support scores using independently normalized `1 + 1` weights and no fitted parameter. At
   `k=0`, no support component exists, so this readout reduces to the declared semantic route.
2. **Cosine 1-NN:** a mandatory representation-only companion row for every fully enrolled external
   baseline cell. Prototype and ridge remain optional diagnostic controls. In a fully enrolled
   `k`-shot cell, every candidate has exactly `k` support recordings; `k` never means that only the
   query's ground-truth class receives support.
3. **Released native method:** a separate `k=0` row when the released model exposes a valid route
   to the complete target candidate set. If it does not, record `N/A`; do not relabel a project-added
   ConSE bridge or training-bank neighbour as native behavior. When the native semantic route is
   also the semantic branch of equal-weight fusion, retain both named rows and disclose that their
   predictions are identical at `k=0`.
4. **Differentiable neighbours:** HALO encoder-development objective only; it is not reported as an
   adaptation mechanism for released baseline models.
5. **HALO evidence-gated classifier:** the learned support-conditioned classifier. It computes
   auditable centered support-vote and label-meaning distributions, then uses bounded,
   label-blind candidate gates to choose their relative reliance. It does not use the retired
   contextual-token mixer. Evaluation must reconstruct
   acquisition vectors from each query and support stream separately; cross-stream scenarios may
   not replace them with zeros or reuse the query configuration for supports.

At `k=0`, there is no target-dataset enrollment, but labelled training-corpus evidence remains
available. For HALO and released encoders without a native open-label head, retrieve the nearest
recording from a reference bank built exclusively from `SUPERVISED_HEAD_TRAIN_DATASETS`, convert its
training label into a one-hot training-vocabulary prediction, and use ConSE to bridge that prediction
to the sealed dataset's candidate strings. Models with a released native text-aligned prediction
path retain it. Report the readout as `training-bank-1nn-conse` or `native_zero_support`; do not call
the former native zero-shot. No sealed recording may enter the training bank.

### Enrollment fusion

For every `k > 0` baseline cell, including both complete and partial enrollment, every model makes
one deployable prediction with the fixed **equal-weight normalized fusion** rule:

1. Obtain semantic scores over every candidate from the model's declared `k=0` route: its released
   native candidate scorer where available, otherwise training-bank 1-NN plus ConSE.
2. For each enrolled candidate, take the maximum cosine similarity between the query and that
   candidate's support recordings. In partial-coverage cells, unenrolled candidates have no
   neighbour score.
3. Per query, z-score semantic scores over the full candidate roster and z-score neighbour scores
   over the enrolled candidates. Degenerate components with fewer than two live values or no
   spread contribute zero rather than amplified noise.
4. Add the two components with coefficients `1 + 1` and take one argmax. No parameter, threshold or
   model-specific fusion weight is fitted.

Semantic-only, prototype, and ridge controls are opt-in diagnostics. Cosine 1-NN is always retained
beside the primary fusion row in the complete-enrollment aggregate curve; neither is selected per
cell. Scenario tables use equal-weight normalized fusion as the default external-baseline row and
may retain 1-NN as a diagnostic rather than a headline result. An "either prediction was correct"
oracle is opt-in only, carries `diagnostic_only=true` and `deployable=false`, and may be reported
only as a clearly labelled diagnostic ceiling; it is not deployed accuracy. Partial-coverage results are split into
truth-enrolled, truth-unenrolled and combined queries, because a support-only method cannot name an
unenrolled candidate. Also compute the harmonic mean of truth-enrolled and truth-unenrolled
performance as a diagnostic summary; whether it appears in the main paper or appendix is a later
presentation decision.

The sealed runner always computes external 1-NN. `--baseline-diagnostic-readouts` additionally
enables prototype and ridge; omitting it avoids their repeated ridge work.

## Baseline fairness

The primary external roster is HARNet, LiMU-BERT-X, UniMTS, and NormWear using author-released
checkpoints. HARNet and LiMU-BERT-X supply frozen representations but no released open-label
prediction mechanism, so their `k=0` values use the disclosed training-bank 1-NN plus ConSE bridge.
Each adapter preserves published units, channel order, resampling, crop/padding, normalization, and
masking. No baseline is retrained by this project for the primary table. If a model cannot accept a
recording or cannot expose a representation at the required granularity, mark the combination
unsupported instead of giving it custom privileged preprocessing.

Models consume all valid samples in the shared evidence window. Length-flexible trunks run once;
fixed-length trunks use consecutive native-size chunks and pad only the final partial chunk with the
published rule. Results disclose whether padding occurred and its fraction. UniMTS and NormWear
consume a composite natively. HARNet and LiMU-BERT-X encode every device independently and use the
single shared equal-device mean plus L2 normalization; these rows are labelled
`per-device-pooled`, not native multi-device inference.
NormWear's native semantic path is fixed-width across device sets, but its released downstream
enrollment representation flattens real channels. A cross-configuration enrolled cell with unequal
channel counts is therefore explicitly unsupported rather than repaired with a project-specific
projection or channel average.

HALO is shown with the same frozen-representation readouts and, for a support-classifier checkpoint,
its retrieve-mix-vote readout on the exact same manifest. This distinguishes representation
quality from a task-specific training gain. Upstream training corpus, parameter count, inference
time, peak memory, and known data overlap must be disclosed beside the score.

## Rungs 1 and 3 (pre-registered; no result yet)

Both reuse this document's data separation, the sealed single-device cells and the provider feature
caches, and apply one procedure identically to all six encoders. Details and registered
predictions are in the [roadmap](../overview/roadmap.md); the invariants a reader must know:

- **Rung 1 (k = 0).** Per cell, queries are split by physical execution into a fixed scored set
  (20 %) and a pool; nested pool draws N ∈ {0, 50, 100, 500, 2000, all}; metrics on the scored set
  only. The procedure is EM-Dirichlet (Martin et al., CVPR 2024) over each encoder's zero-shot
  probability vectors (softmax, T = 30), with λ = N and an embedding-affinity term (μ = 1, 10
  neighbours in the encoder's own feature space). HALO's zero-shot scores come from its checkpoint's
  `p_text`; HARNet and LiMU-BERT-X use the training-bank ConSE bridge; UniMTS and NormWear their native
  text heads. The curve's null is N = 0 under the same procedure; the per-window inductive anchor
  must reproduce the sealed k = 0 row. Rows report `neighbour_purity`. Readout version `ncurve-v2`.
- **Rung 3.** Every treatment — linear probe, small classifier, LoRA, full fine-tune, and a
  from-scratch specialist — is fitted on the same k windows per class and scored on rung 1's scored
  set, beside the rung-2 parameter-free readout re-run on the same draw (`enrollment_frozen`).
  Budgets are fixed a priori; nothing selects on the scored set. NormWear has no fine-tuning path
  and is reported unsupported; HARNet-10 gets the cached-feature treatments only. Readout version
  `finetune-v1`.

## Metrics and reporting

Primary classification metrics are per-dataset macro F1 and balanced accuracy. Report accuracy only
as secondary context. Every retained result artifact must nevertheless contain all three metrics:
macro F1, balanced accuracy, and accuracy. Compute confidence intervals by subject, not by
overlapping windows. Publish
per-dataset tables before an aggregate so one large or easy dataset cannot conceal failures.

Two evaluation regimes are retained:

1. **Complete-enrollment aggregate curve:** every candidate receives exactly `k` supports at
   `k>0`; external baselines report both equal-weight normalized fusion and cosine 1-NN. At `k=0`,
   report equal-weight normalized fusion and the released native method where one exists.
2. **Deployment-scenario evaluation:** partial enrollment and the six other preregistered
   heterogeneity scenarios use equal-weight normalized fusion as the default external-baseline
   readout, with cosine 1-NN always emitted beside every enrolled fusion row. HALO classifier rows
   likewise carry the same-encoder cosine 1-NN floor. The active
   `deployment-scenarios-v5-20260918` roster is partial enrollment, cross placement, cross dataset,
   missing modality, rate mismatch, MM-Fit new domain, and device-set mismatch (including the
   MM-Fit composite cell). MobiAct is a separately versioned prospective scope until its official
   archive is prepared; it never enters the sealed-six aggregate.
   The former compound cold-start scenario is retired because it duplicated these factors.
   The scenario CLI defaults to the representative 8-second window and `k=0,1,4,8,32`.
   Larger grids require explicit `--k` / `--window-seconds` overrides; the complete-enrollment
   sealed runner retains its separately configured duration and support grid.

For both regimes, the classifier-attribution experiment freezes each released baseline encoder,
attaches the same HALO learnable classifier, trains only that classifier under the same training
episodes and selection rule, and evaluates on the same manifests. Name these arms `HALO classifier
with frozen <encoder>`, not as native baseline methods. Report trainable parameter count, training
steps, runtime, and seed so the adapter budget is explicit.

For every promoted result, save:

- code commit and checkpoint hashes, including sealed `run_provenance.json`;
- split and episode-manifest hashes;
- raw source-slice hashes, duration, ordered device set, multi-device mode, and padding fraction;
- `C` and `k` distributions;
- per-dataset macro F1, balanced accuracy, and subject-level uncertainty;
- training/evaluation time, peak memory, and unsupported combinations; and
- a short protocol note identifying zero-support versus enrolled-support conditions.

Embedding figures are optional diagnostics, never selection signals.  When enabled, the sealed
runner writes a bounded balanced-sample PCA, same/different-label cosine distributions, label
centroid similarity, and accompanying geometry summary beside the machine-readable result.  These
figures cannot change a manifest, prediction, threshold, or checkpoint.

Results belong in [RESULTS.md](../results/RESULTS.md). Historical generic-HAR and application-pivot
tables are not valid under this protocol and remain archived.
