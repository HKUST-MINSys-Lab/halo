# V5 online memory reader: implementation record

Date: 2026-09-25. Status: **implemented, smoke-tested, not yet trained or evaluated at scale**.
This implements the earlier [proposal](2026-09-25-deployment-memory-reader-proposal.md) and
[online-episode addendum](2026-09-25-memory-reader-online-episode-addendum.md). Those remain
historical design documents. V5 is an experimental HALO-only online classifier, **not** the
registered fixed-pool rung 1 or a replacement for the promoted v4 classifier.

## Contract and files

- [`memory_classifier.py`](../../model/support/memory_classifier.py) is checkpoint architecture
  `support_memory_reader_v5`. It inherits the v4 encoder and cosine semantic projection. A query
  and each prior entry have motion and acquisition vectors; every entry has three grouped tokens:
  motion, acquisition, and label evidence. Role embeddings distinguish token types. Local attention
  groups one entry's tokens; query-to-bank attention is order invariant over entries. There is no
  positional/age input and no free candidate-logit head.
  Per-entry trust also reads motion-neighbor density and the agreement of nearby entries' label
  evidence, alongside query similarity and acquisition agreement. These statistics are computed
  from the current bank, not from dataset identity or hidden truth.
- [`memory_bank.py`](../../training/support_classifier/memory_bank.py) stores recording/execution
  identity, acquisition metadata as a sidecar, original semantic probabilities with their ordered
  roster, a verified label only when provided, observation index, and model version. It predicts
  before insertion, re-scores on roster changes, protects verified entries, and bounds unlabeled
  history. The current semantic probabilities, not the memory-refined answer, are inserted.
  A verified label outside a changed roster supplies no false in-roster vote; the motion and
  acquisition entry remains available as context when other entries have usable evidence.
  `state_dict`/`from_state_dict` snapshot a deployment-local bank with a schema and strict model
  version check. Snapshot tensors are detached; snapshots are inference state, not training caches.
- [`memory_episodes.py`](../../training/support_classifier/memory_episodes.py) draws simulated
  arrival sequences from execution-distinct training rows. Roster size, class balance, memory
  length, distractors, cross-dataset rows, and verified-label rate vary. Hidden query labels feed
  the loss only. The final query is always in the roster and unverified. Within an episode, a
  counterfactual view of that same query/roster removes or subsets history.
- [`train_memory.py`](../../training/support_classifier/train_memory.py) initializes from a v4
  checkpoint, freezes the encoder by default, and supports `--fine-tune-encoder`. It encodes the
  episode once, differentiates through query and prior-memory motion/acquisition vectors, and
  trains by classification cross-entropy. It selects `best_internal.pt` by per-dataset macro-F1
  on subject-disjoint development rows. It also writes `last.pt`, `log.jsonl`, and an independent
  `--evaluate-checkpoint` development readout. No sealed data are read.
  `--resume` restores optimizer and torch RNG state and checks the source checkpoint and
  curriculum arguments before continuing in the same output directory.

No age, training dataset identity, source name, or hidden label is a learned reader input. The
acquisition metadata sidecar is for audit and re-encoding; the encoder's acquisition vector is
the learned reader input. A reader-label holdout is globally excluded from **v5 optimization**,
but the v4 source encoder may have seen those labels. This is a reader-transfer diagnostic, not
an unseen-label claim about the whole model.

## Controls and telemetry

Every scored query produces no-memory semantic, fixed equal semantic/memory vote, and learned
reader predictions under the same causal bank. Validation logs per-dataset accuracy, macro-F1,
balanced accuracy, corrections/new errors, empty/predicted/verified/mixed-memory accuracy,
reader-heldout-label accuracy, semantic weight, reliability concentration, loss, and number of
scored examples. It also reports accuracy of all three paths by retained-bank size (0, 1, 2-3,
4-7, 8+). Training logs objective, gradient norm, seen/retained/verified counts and matched
counterfactual view count. An empty bank returns the semantic distribution **exactly**.

The fixed vote uses cosine-neighbor weights over all retained recordings and stored uncertain
semantic distributions or exact verified one-hots; it is a floor, not a literature baseline.
The current v5 model uses one read and one per-entry reliability adjustment shared by all
candidates, followed by a per-candidate semantic/memory gate and normalization.

## Reproduction and limits

```bash
uv run python -m training.support_classifier.train_memory \
  --source-checkpoint runs/support-classifier/halo_t6_40k_20260920/last.pt \
  --out runs/support-classifier/v5_screen --smoke
uv run python -m training.support_classifier.train_memory \
  --source-checkpoint runs/support-classifier/halo_t6_40k_20260920/last.pt \
  --evaluate-checkpoint runs/support-classifier/v5_screen/best_internal.pt \
  --out runs/support-classifier/v5_screen_recheck --smoke
```

Two-step frozen-encoder and end-to-end GPU smokes, independent checkpoint reload, and contract
tests completed on 2026-09-25. In the two-step end-to-end smoke, 84 of 91 floating-point encoder
state tensors changed from the source checkpoint; the reader tests also verify gradients reach
query, history motion/acquisition, semantic projection, reliability and gate. A two-step resumed
run matched a continuous run bit-exactly in the head state. Smoke scores are deliberately not
result claims. No long v5
training, no sealed evaluation, and no head-to-head claim versus v4 have been made. Before
promoting v5 into the paper, freeze an **online** ordering/roster protocol, compare v5 to its
same-encoder no-memory and fixed-vote controls and to v4 under matched verified enrollments,
and report multiple order seeds. Simulated arrival order is not a longitudinal health dataset.
