# HALO evidence-aware v2, step 40k — deployment scenarios

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

The `deployment-scenarios-v5-20260918` suite completed all seven scenarios at 8 seconds and
`k = 0, 1, 4, 8, 32`: 2,395 rows, zero task failures, manifests identical to the released-baseline
scenario artifact on all 737 shared cells. Deployable readouts only (the truth-dependent branch
oracle is opt-in and was not requested). The classifier sits at 25-44 macro F1 in every scenario
regardless of k; its near-zero paired mismatch costs reflect that it ignores the support set, not
robustness.

Branch diagnostics for the same checkpoint are in `halo_evidence_aware_v2_step40k_scenario_branches_v5_20260919`.
