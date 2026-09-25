# Rung 3: heads on each model's own features, and three support draws

Date: 2026-09-25. Status: **code and tests; nothing run.** Follows the
[debug sweep](2026-09-25-debug-sweep-and-fixes.md) and a review of the concurrent session's fixes
(committed separately as `52a3f3d`).

1. **No random projection in the baselines' fine-tuning path.** The matched trunk always wrapped
   the released trunk in a freshly initialised 512→256→128 MLP plus LayerNorm. Under LoRA that MLP
   was frozen (a random bottleneck with adapters on it); under full fine-tuning it was an extra
   layer trained from k windows that HALO, whose head sits on its own pooled vector, does not have;
   and it made the baselines' linear probe (adapter features) incomparable with their LoRA row.
   `MatchedCorpusEncoder(projection=False)` now exposes the trunk's own feature, and rung 3 uses it
   for HARNet-5, LiMU-BERT-X and UniMTS. The corpus-matched training arms keep their trainable
   projection (unchanged).
2. **Three support draws per (cell, k).** One draw made k = 1-2 rows mostly a function of which
   windows were drawn. `--support-draws` (default 3) repeats each (cell, k) on independent support
   sets with independent fit seeds; draw 0 is the original set (and rung 1's k > 0 draw). This
   triples rung 3's cost.
