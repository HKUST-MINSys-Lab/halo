# Debug sweep of everything built 2026-09-22 to 09-25, and the fixes

Date: 2026-09-25. Status: **fixed and unit-tested; a two-step v5 smoke ran on training data; no
sealed data read, no result produced.** Scope: the `evaluation/` package and its extraction, rung 1
(evaluator, controls, training arm), rung 3 and the corpus-matched trunks, and the v5 online-memory
reader ([implementation record](2026-09-25-v5-online-memory-implementation.md)). Four read-only
auditors plus a direct read of v5; findings were reproduced before fixing.

## Fixed

| # | issue | where | fix |
|---|---|---|---|
| 1 | **v5's "subject-disjoint" validation leaked into the source encoder's training set.** `train_memory` built its own split with its seed (47); v4's split uses `--data-seed 20260901`. Recomputed: 12 of 23 v5 validation subjects were v4 training subjects (KU-HAR 7, WISDM 4, REALDISP 1). | `train_memory.py` | split with the source checkpoint's `data_seed`; recorded as `subject_split_seed` |
| 2 | Rung 3's small classifier was scored with dropout active (no `eval()`). | `rung3_finetune/finetune.py` | `model.eval()` before scoring |
| 3 | Rung 3's LoRA wrapped only `nn.Linear`: for HARNet (ResNet) and UniMTS (ST-GCN) that is the projection head only — 0 % of the pretrained trunk adapted. | `rung3_finetune/lora.py` | `LoRAConv` for ungrouped `Conv1d`/`Conv2d` (rank-r conv with the base kernel + zero-init 1×1), identity at init |
| 4 | The balanced-pool control (09-25 rewrite) drew independent pools at each N from one advancing random stream — not nested. | `controls/balanced_pool.py` | one class-interleaved order per available-row set; the balanced pool of size N is its prefix, independent of call order |
| 5 | The disjoint-class control scored macro-F1 over the full roster, half of which never occurs in its scored set — capped near 50. | `rung1_unlabeled/ncurve.py`, `run.py` | F1 over the kept classes (`macro_f1_class_policy = scored_classes_only`) |
| 6 | Rung-1 training capped the centered logits' RMS at 1 before cross-entropy: the correct class could never exceed ~0.75 probability for 5–10 candidates, a floor under the loss (NormFace's no-scale problem). | `train.py` | cap at `ROSTER_LOGIT_RMS = 4` (reachable > 0.999 for 2–20 candidates); curriculum contract updated |
| 8 | Rung 3 resampled one model two ways: cached-feature treatments through each released adapter (`np.interp`; `resample_poly`), raw-window treatments through a torchaudio sinc resampler. | `model/tokenizer/matched_encoder.py` | the matched trunks now call each adapter's own method (linear interpolation for HARNet, `resample_poly` for LiMU-BERT-X / UniMTS); a test pins equality |
| 9 | v5's trust and blend gates saw label identity (label-text evidence tokens, candidate text in the gate) and trust was unbounded — the route by which T1/T3 collapsed onto label meaning; with verified entries of every class, trust could act as a private classifier. | `model/support/memory_classifier.py` | evidence token = label-blind statistics (entropy, confidence, margin, provenance); gate sees per-candidate evidence values only; trust bounded by `TRUST_LIMIT = 2`; test: relabelling the roster leaves reliability weights unchanged |
| 10 | v5 trained on histories ≤ 12 but deploys to 64 entries, and fed the gate `n/64` directly. | `train_memory.py`, reader | `--max-history 63` default; size feature clipped at 1; telemetry bins to 32+ |
| 11 | v5 retrained v4's `p_text` with training-vocabulary cross-entropy, so its "no-memory" control was no longer v4's zero-shot. | `train_memory.py` | `p_text` frozen unless `--train-text-projection` |
| 12 | The v5 bank raised when verified enrollments exceeded 64 (v4 is evaluated to k = 128 per class) and on a second window of the same execution (normal in a live stream). | `memory_bank.py`, reader | capacity bounds unlabelled entries only; verified are never evicted; one entry per execution (a later unlabelled window replaces an unlabelled one, and is observed-not-retained if the execution is enrolled); the reader no longer raises above `max_entries` |
| 13 | `checkpoint_text_projection` took the retired T1 head's d×d `p_text` for a text-space projection and crashed. | `evaluation/zero_shot.py` | architecture whitelist (v2, v3, v4, v5) plus an output-width check; others fall back to the ridge bridge |
| 14 | The "from-scratch" HARNet specialist kept pretrained BatchNorm running statistics. | `matched_encoder._reinitialise` | `reset_running_stats()` |
| 15 | Rung 3 skipped raw-window treatments whenever cached features were unsupported; matched LiMU-BERT loaded non-strictly; rung-3 provenance claimed `subject_independent: True` for an execution-level split. | `rung3_finetune/run.py`, `matched_encoder.py` | cached treatments get their own `n/a` rows and raw-window treatments still run; `strict=True`; provenance corrected |

## Checked and not a bug

- **7. "Rung-1 training always shows full roster coverage."** Coverage `(1, 1)` makes every class
  *eligible*, but window counts are a multinomial over a Dirichlet-weighted class marginal, so small
  pools already miss classes, as the evaluator's random prefixes do. Unchanged.
- A v5 checkpoint passed to the rung-2 sealed runner fails with a clear error rather than being
  misread.

## Checked clean

The extraction is byte-identical to `hist/v3-support-conditioned/pre-evaluation-package-20260923`;
the EM-Dirichlet port matches the reference implementation (max |Δu| 8e-7 on random inputs); roster
and row masking; the affinity kNN (self/padding exclusion, chunking); the uncontrolled N-curve's
execution-disjoint split and nested draws; pool sampling excludes the query's execution; fp32 under
bf16; resume persistence; `--pool-size 0` bit-identity; the published rung-2 tables are untouched by
the `p_text` route.

## Consequences

- v5 state dicts change shape (evidence and gate layers): v5 smoke checkpoints from before this entry
  do not load. No v5 run beyond smokes existed.
- Rung-3 LoRA rows for HARNet and UniMTS now adapt the trunk; trainable-parameter counts rise
  accordingly and are recorded per row.
- Suite: 1,132 passed, 2 skipped. v5 smoke (2 steps, training corpus only): split seed 20260901,
  histories to 63, `p_text` frozen.
