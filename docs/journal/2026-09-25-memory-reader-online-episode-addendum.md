# Memory reader: online episodes and age decision

Date: 2026-09-25. Status: **proposal update; not implemented or evaluated**.
This addendum follows the [memory-reader proposal](2026-09-25-deployment-memory-reader-proposal.md)
and [literature audit](2026-09-25-memory-reader-literature-audit.md). It supersedes their
recommendation to give the first reader per-entry age or elapsed-time features. Their bank schema,
three-token starting point, evidence provenance and comparisons otherwise stand.

## One deployment episode

An episode is a new deployment with a declared candidate roster and a sequence of recordings from
that deployment. The roster can contain labels absent from classifier training. The first
experiment holds the roster fixed within one episode; a changing roster is a separate registered
stress condition handled by rescoring stored semantic features against the new label text.

1. Begin with an empty bank and the same frozen model for the whole deployment episode.
2. For each next recording, encode it, produce its no-memory semantic prediction, and predict using
   only entries already in the bank. Compute training loss against its hidden label, if training.
3. After scoring, insert the recording's motion/semantic/acquisition representations, its
   **original no-memory** logits with roster provenance, and any genuinely verified enrollment
   label. Unlabelled query labels never enter the bank, even though they supervise the loss.
4. Continue with later recordings. Measure both the number **seen so far** and the number
   **retained** in the bank. Corrections and new errors are always measured against the same
   model's no-memory prediction on the same recording.

Training builds many such episodes from supervised training sources, varying candidate roster,
deployment acquisition, history length, class mix, verified enrollment availability, and natural
prediction mistakes. For most available sources the arrival order is simulated, not real
longitudinal history. Candidate labels for scored recordings remain hidden from the memory until
genuine enrollment is supplied. End-to-end training must represent history with the current
encoder weights rather than a stale feature cache; the exact gradient-through-history strategy
is an implementation/profiling decision, not a property already demonstrated.

## Age is not an initial reader input

The first reader receives neither per-entry age nor physical timestamp features. Bank occupancy
may be supplied as evidence quantity. Observation indices remain **bookkeeping** for causal
replay, leakage checks, duplicate identity, and reconstructing stream order. They are not model
tokens. A relative-age ablation belongs later, only if real drift or stale-history effects are
observed. Dropping age also avoids teaching a spurious chronology from arbitrarily ordered
single-session recordings.

Occupancy and age are different concepts: if entries are evicted, filtered, or the bank reaches
capacity, bank size stops increasing even though many recordings have arrived. Therefore the
first experiment makes no claim that occupancy measures elapsed time or model age. It measures
how much evidence is **currently retained**. Report both retained and seen counts.

## Relationship to the registered rung 1

This is a proposed **online** unlabelled adaptation experiment for HALO's learned reader. The
current registered rung-1 implementation instead constructs fixed execution-disjoint unlabelled
pools of size N and applies the **same transductive inference method to every encoder** on a fixed
scored set. It does not run the sequential predict-then-insert classifier described here. The two
protocols answer related but different questions and must not be presented as the same result.
Changing the paper's rung 1 to the online learned-reader protocol requires a separate decision
and a new pre-registered comparison before reading sealed results.
