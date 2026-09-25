# Deployment memory reader: proposed design for a later classifier experiment

Date: 2026-09-25. Status: **design proposal; not implemented or evaluated**.

This is a HALO-specific, learned extension of the frozen-parameter classifier. It is outside the
registered three-rung paper protocol in [the roadmap](../overview/roadmap.md): rung 1 still uses
the shared EM-Dirichlet/embedding-affinity procedure, and its inference must remain identical for
all six encoders. This proposal neither changes those readouts nor claims a result. If promoted
into a paper experiment, its protocol and scope need a separate decision before sealed testing.

## Question and comparison

Can a learned reader use earlier predictions, unlabelled recordings, and verified enrollments to
improve future classification after deployment, without changing model weights at deployment?
Measure the reader against (1) the same model with no memory and (2) a fixed similarity-weighted
memory vote with the **same** encoder, memory entries, provenance rules, and candidate roster.
The learned-reader claim is its gain over (2), not merely its gain over no memory. Monitor
corrections (no-memory wrong, reader right) and new errors (no-memory right, reader wrong) on the
same queries. No gain is presumed; the fixed vote is the fallback if the reader does not help.

## Memory contract

Each entry is one earlier recording, not one sensor row, window, or attention token. Store:

- Motion representation from the current HALO encoder and recording pool.
- A separate acquisition representation. Keep the encoder's existing early acquisition
  conditioning in the first experiment; expose acquisition again to the reader for comparison.
- Original semantic scores or probabilities **and the ordered candidate roster used to compute
  them**, or a verified enrollment label. Keep provenance distinct: verified label, model
  prediction, or no label evidence. Never promote a model prediction to verified evidence.
- Recording/execution identity, deployment observation index, and encoder/scorer/calibration
  version. Cache entries from a different checkpoint are re-encoded or discarded.

An ordered roster is registered once and referenced by entries; its registry ID is bookkeeping,
never a model input. A probability vector alone does not identify its labels. Cache the full
distribution beside the compact label-evidence token. When a deployment changes candidate roster,
re-score the stored recording representation with the **same frozen scorer** against the new
candidate labels; do not merely renormalize old probabilities or invent probabilities for labels
not previously scored. Preserve the original distribution for provenance. If a verified label is
outside the current roster, keep the entry's motion evidence without assigning it a false
in-roster class.

Use one monotonically increasing observation counter per deployment stream. Increment once for
each newly observed recording, even when it is not retained; synchronized devices that form one
recording count once. At query index `n`, an earlier entry inserted at `j < n` has age `n - j`.
Compute `log1p(age)` at read time. Retrieval, repeated inference, and refinement do not advance
the counter. A delayed verification retains the recording's original `j`; verification arrival
can be recorded separately. This age measures *recordings seen*, not elapsed time. If reliable
physical timestamps exist, elapsed time may be an additional feature with an availability mask;
never impute a missing timestamp as zero age.

Bank scope is one deployment/session. Protect verified enrollments. Bound unlabeled history,
deduplicate by recording/execution identity and overlap, and retain a mixture of recent entries
and motion/acquisition diversity. Avoid predicted-class quotas as the first eviction rule, since
incorrect predictions would determine which classes survive. A per-dataset statistical signature
is not part of the first version.

## Reader and prediction path

Encode each recording independently. Begin with exactly three tokens per memory entry: motion,
acquisition, and label evidence. Use role embeddings and a recording-group relation; shuffling
whole memory entries must leave the output unchanged within floating-point tolerance. A second
motion token and other capacity changes are later ablations, not prerequisites.

1. Score the query against the current candidates with the ordinary no-memory semantic path.
   Record this probability distribution as `p_no_memory`.
2. Compare query motion to bank motion and compute a soft memory vote from verified labels and
   stored uncertain predictions. Multiple entries may vote. Also score this vote with a fixed,
   declared fusion rule; that is the non-learned memory floor.
3. Give the learned reader query/candidate information, memory tokens, and the initial
   similarities and vote. The first reader learns a reliability adjustment **per memory entry**,
   shared across candidates, then recomputes the soft vote. Candidate-specific entry adjustment
   remains a planned extension if shared reliability proves insufficient.
4. Learn a semantic-versus-memory weight per candidate; combine the two evidence paths and
   normalize over the declared roster. Attention may adjust evidence weights and their blend,
   but may not emit an independent unrestricted candidate-logit path. Use one memory read first.

With an empty bank, return `p_no_memory` exactly for that checkpoint and input. With verified
enrollments but no unlabelled history, compare to the existing v4 classifier as a compatibility
control. The learned reader's exact historical v4 scores are not guaranteed after end-to-end
encoder retraining. Keep the memory size bounded and report inference cost as it grows.

At deployment, predict the current recording **before** inserting it into the bank. Store its
original semantic evidence, not the memory-refined output, to limit feedback loops. A later
verified label can supersede predicted evidence without changing its provenance history. The
encoder, scorer, and reader weights stay fixed after deployment.

## Training and evaluation sequence

1. Build the fixed bank, roster registry, causal predict-then-update runner, and fixed vote.
   Compare it with no memory before choosing reader capacity.
2. Freeze the encoder and fit the reader on training-source simulated deployments using ordinary
   classification cross-entropy. Train across empty, unlabelled-only, enrollment-only, and mixed
   memories; vary bank size, candidate roster, partial enrollment, imbalance, natural model
   mistakes, distractors, acquisition mismatch, and device set. Counterfactual episodes can keep
   query and candidates fixed while changing memory. Hidden labels supervise loss only; they
   never enter an unlabelled entry. Detach stored pseudo-probabilities. Fit any confidence
   calibration on training-source development data and fix it before sealed evaluation.
3. If the reader helps the fixed floor, train encoder and reader end-to-end. Recompute memory
   representations with the current encoder during training so training does not use stale
   cached features. Compare the resulting model both with its own no-memory path and with the
   original frozen no-memory model, to expose regressions in the encoder.

Start with cross-entropy and the exact empty-bank identity. Branch preservation and best-path
non-regression remain optional, modular losses to introduce only for a measured failure: the
earlier T3 classifier used them and still collapsed toward semantic evidence. Deliberately wrong
memory is a stress view, not a requirement to copy the no-memory answer when it is wrong.
Monitor memory and semantic reliance by evidence quality, plus reliability-weight concentration
and saturation, before adopting numerical caps from v4.

For evaluation, fix and record stream order before scoring, test several orders, and keep every
query causally ahead of its own insertion. Report accuracy, macro-F1, balanced accuracy,
calibration, corrections/new errors, runtime, and memory footprint versus both memory size and
enrollment count. Include wrong-prediction, wrong-placement, distractor, and disjoint-class
memory controls. A gain from disjoint-class memory alone does not identify its cause. Compare
with no memory, the fixed vote, EM-Dirichlet plus affinity under matched available information,
and v4 with verified enrollments. The existing sealed-test restrictions still apply; do not tune
the reader or bank policy on sealed results.

The present sources mostly lack true longitudinal arrival sequences. Where order is simulated,
call it simulated and do not claim measured adaptation to physical drift over time.
