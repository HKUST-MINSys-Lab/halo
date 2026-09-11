# Evaluation protocol: support-conditioned HAR

> Protocol of record, 2026-09-11. A score is reportable only when the run records the data split,
> encoder checkpoint, baseline adapter revision, episode manifest, and support/candidate settings.

## Data separation

1. Split subjects before constructing recordings or episodes.
2. Keep every recording, session, and source segment assigned to one split only.
3. Use training data to fit parameters and development data for checkpoint selection, temperatures,
   candidate distributions, and any operating decision. Touch the test split once, after freezing
   those choices.
4. A pretraining source cannot also support an unseen-dataset generalization claim. Such a score is
   still a deployment comparison, but must be labelled as such.

## Episode contract

An episode contains a query recording, a bounded candidate set, and a bank of labelled support
recordings. The candidate count `C`, support count `k`, source dataset, subject relationship, and
recording lengths are part of the manifest. Background supports must not reveal the correct
candidate simply through their presence or their acquisition configuration.

Report a support curve at every feasible `k`, including `k=0`, rather than selecting one favorable
enrollment count. Large candidate-set episodes are required because a small `C` is an easier task;
the held-out test set determines the deployment candidate set.

## Compared methods

For each eligible representation, report these readouts on exactly the same episodes:

1. **1-NN:** nearest labelled support, the deployment-simple floor.
2. **Prototype:** class mean of enrolled supports, clearly labelled as a batch-enrollment method.
3. **Ridge:** an adapted linear readout fit only from the episode's enrolled support data, clearly
   labelled as fitted adaptation.
4. **HALO support vote:** the retained fixed vote plus, where trained, its sensor-only support-row
   reweighter.

At `k=0`, no support-based readout can claim enrollment. Report each model's declared zero-support
path separately and do not compare it with an enrolled-support score as though the information were
identical.

## Baseline fairness

The primary external roster is HARNet, UniMTS, and NormWear using author-released checkpoints.
Each adapter preserves published units, channel order, resampling, crop/padding, normalization, and
masking. No baseline is retrained by this project for the primary table. If a model cannot accept a
recording or cannot expose a representation at the required granularity, mark the combination
unsupported instead of giving it custom privileged preprocessing.

HALO is shown both with the same frozen-representation readouts and, when appropriate, in a clearly
separate end-to-end arm. This distinguishes representation quality from a task-specific training
gain. Upstream training corpus, parameter count, inference time, peak memory, and known data overlap
must be disclosed beside the score.

## Metrics and reporting

Primary classification metrics are per-dataset macro F1 and balanced accuracy. Report accuracy only
as secondary context. Compute confidence intervals by subject, not by overlapping windows. Publish
per-dataset tables before an aggregate so one large or easy dataset cannot conceal failures.

For every promoted result, save:

- code commit and checkpoint hashes;
- split and episode-manifest hashes;
- `C` and `k` distributions;
- per-dataset macro F1, balanced accuracy, and subject-level uncertainty;
- training/evaluation time, peak memory, and unsupported combinations; and
- a short protocol note identifying zero-support versus enrolled-support conditions.

Results belong in [RESULTS.md](../results/RESULTS.md). Historical generic-HAR and application-pivot
tables are not valid under this protocol and remain archived.
