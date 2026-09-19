# HALO evidence-aware v2, step 40k — scenario branch diagnostics

Run date: 2026-09-19.
Architecture `support_evidence_aware_v2` ("evidence-aware v2"), trained end to end for 40,000
steps from code `a9e8d91` (= lab `main` at launch) with `--classifier contextual --steps 40000`,
Stage A curriculum defaults and the v2 defaults: counterfactual enrollment probability 0.25 and a
20% per-source open-vocabulary label holdout (25 of 155 canonical labels never enter optimization).
**Checkpoint: `last.pt`, step 40,000** — the fixed-budget primary declared before any sealed number
existed, the same policy as the 2026-09-18 residual/neighbours/contextual arms. SHA-256
`d0961938c05479d550dfa00b4c67948353ffbf1560171d12f34eb135151cba52`. The step-32,500 `best_internal.pt` companion lives in its own `..._step32500_...` artifacts.

**Result: negative, not promoted.** The router routes almost every candidate to the label-meaning
branch (validated semantic reliance 0.97-0.99 from step 7,500), so the final classifier equals its
label-meaning branch and ignores enrolled support at every k. Two further confounds are recorded in
`docs/results/RESULTS.md`: the label holdout removed walking, sitting, walking_upstairs, stairs and
cycling (17 of 52 sealed classes), and the evaluator's `label_seen_in_training` flag does not
subtract the holdout, so the seen/unseen split in these rows is wrong for this checkpoint.

Diagnostic (non-headline) scenario run on the same manifests with
`--readouts halo-classifier-label-meaning-only halo-classifier-unmodified-support-vote
halo-classifier-contextual-support-vote`: 4,583 rows, zero task failures. Note that the evaluator's
"unmodified support vote" is the fixed-temperature (0.07) support floor, not the head's learned-
temperature support-status branch. At k=8 the unmodified support vote beats the label-meaning
branch by 18-51 points in every scenario except partial coverage, and the learned contextual
correction lowers the support vote in all seven scenarios.
