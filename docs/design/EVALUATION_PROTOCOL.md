# Evaluation protocol: support-conditioned HAR

> Protocol of record, revised 2026-09-13. A score is reportable only when the run records the data split,
> encoder checkpoint, baseline adapter revision, episode manifest, and support/candidate settings.

> **Operational status:** `training.support_classifier.sealed_eval` is the only sealed-test entry
> point. It materializes deterministic execution-disjoint manifests, caches each representation
> with a checkpoint/artifact fingerprint, and writes machine-readable and Markdown per-stream
> tables. Internal validation from `training.support_classifier.train.py` is not a sealed result.
> No result is reportable until this runner has been invoked once with a frozen checkpoint and the
> generated manifest and artifact hashes have been retained beside the table.

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
[DESIGN_OF_RECORD.md](DESIGN_OF_RECORD.md) and enforced in code by roster tests.

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

RealWorld and Shoaib additionally report every declared placement separately and one preregistered
all-device cell. Composite rows require elementwise-identical event, label, subject, and execution
identities. A row rejected by any member's quality screen is removed from the composite for every
model. The ordered device list and exact raw-slice fingerprint are stored in the manifest.

## Compared methods

For each eligible representation, report these readouts on exactly the same episodes:

1. **1-NN:** nearest labelled support, the deployment-simple floor.
2. **Differentiable neighbours:** parameter-free, temperature-scaled soft cosine voting over every
   enrolled support. This is the exact scoring rule used by the encoder-only adaptation control.
3. **Prototype:** class mean of enrolled supports, clearly labelled as a batch-enrollment method.
4. **Ridge:** an adapted linear readout fit only from the episode's enrolled support data, clearly
   labelled as fitted adaptation.
5. **HALO retrieve-mix-vote:** its learned semantic token mixer. For `k > 0`, it jointly attends
   to query, support, support-label, and candidate-label tokens before a soft support vote. For
   `k = 0`, it jointly attends to the query and candidate-label tokens before direct cosine
   scoring. The two paths use separately trained head weights and one shared encoder.

At `k=0`, there is no target-dataset enrollment, but labelled training-corpus evidence remains
available. For HALO and released encoders without a native open-label head, retrieve the nearest
recording from a reference bank built exclusively from `SUPERVISED_HEAD_TRAIN_DATASETS`, convert its
training label into a one-hot training-vocabulary prediction, and use ConSE to bridge that prediction
to the sealed dataset's candidate strings. Models with a released native text-aligned prediction
path retain it. Report the readout as `training-bank-1nn-conse` or `native_zero_support`; do not call
the former native zero-shot. No sealed recording may enter the training bank.

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

HALO is shown with the same frozen-representation readouts and, for a support-classifier checkpoint,
its retrieve-mix-vote readout on the exact same manifest. This distinguishes representation
quality from a task-specific training gain. Upstream training corpus, parameter count, inference
time, peak memory, and known data overlap must be disclosed beside the score.

## Metrics and reporting

Primary classification metrics are per-dataset macro F1 and balanced accuracy. Report accuracy only
as secondary context. Compute confidence intervals by subject, not by overlapping windows. Publish
per-dataset tables before an aggregate so one large or easy dataset cannot conceal failures.

For every promoted result, save:

- code commit and checkpoint hashes;
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
