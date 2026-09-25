# Rung 1 smoke tests: everything runs; one comparability problem found

Date: 2026-09-25. Status: **smoke tests only (1-2 sealed cells, reduced grids); not results.**
Artifacts under `runs/evaluations/smoke_rung1/` and `runs/support-classifier/smoke_*` (not promoted).

## What ran

| component | command (abridged) | outcome |
|---|---|---|
| six encoders under EM-Dirichlet + affinity | `halo-rung1 --cells 2 --pool-sizes 0 50 all` | 96 rows, 80 s, no failures |
| controls | `--control balanced_pool`, `--control disjoint_classes`, `--affinity-mu 0` | all run |
| HALO trained through EM-Dirichlet | `halo-train --smoke --rung1-training --pool-mode transductive --pool-size 100` | 18 s, finite loss, checkpoint written and scored by `halo-rung1` |
| neighbours-arm checkpoint (no `p_text`) | `halo-rung1 --encoder-label halo:neighbours` | takes the ridge-bridge route as designed |
| corpus-matched HARNet arm | `halo-train --smoke --encoder-arch harnet --classifier neighbors`, then `halo-rung1 --encoder-label matched:harnet` | trains and scores |

## Checks that passed

- Every baseline's k = 0 anchor equals its published sealed k = 0 row exactly (2 cells, 5 baselines).
- HALO v4's k = 0 anchor equals the published sealed `halo-classifier` k = 0 row on **all 11**
  sealed 8 s cells: with no supports, v4's calibration term shifts every candidate equally, so the
  classifier reduces to its `p_text` arg-max. The same-day note that the anchor "must not be
  presented as a reproduction" was over-cautious and is corrected in the docs.
- λ = N, masking, controls, routes (`halo_p_text` / `halo_text_bridge` / ConSE / native) as specified.

## Findings

1. **Neighbours are partly the same recording (diagnostic added).** 14-56 % of a window's 10
   embedding neighbours come from the same physical execution (highest for HALO and on RealWorld),
   so `neighbour_purity` partly measures within-recording similarity. Not leakage — a deployment's own
   stream has adjacent windows — but rows now also report `neighbour_same_execution` and
   `neighbour_purity_other_execution`. Among other-execution neighbours HALO still leads
   (MotionSense 0.94 vs UniMTS 0.88 vs HARNet-5 0.74).
2. **A single temperature makes the providers' probability features incomparable (open decision).**
   EM-Dirichlet consumes softmax(T · score) with the released T = 30 for every provider. Measured
   on two cells: UniMTS's native scores are tightly bunched (mean top-1/top-2 gap 0.001), so its
   features are essentially uniform (normalized entropy 1.00, mean max probability 0.14-0.19);
   HALO, NormWear and the ConSE providers are sharp (entropy 0.03-0.35; HALO 46-75 % of rows above
   0.99). Consequences seen in the smoke tests: with μ = 0 UniMTS falls from 42.4 (anchor) to 28.5;
   with 5 × 50 EM iterations instead of 20 × 200 it falls to 11.9 while HALO is unchanged. The
   transductive method's behaviour is therefore set by each provider's score scale, which the
   comparison should not depend on. Options are listed in the session record; nothing changed yet.
3. The matched HARNet arm scores 38.8 at k = 0 after three training steps: its zero-shot route is a
   ridge bridge fitted on labelled training data, and random convolutional features are informative.
   Bridge-route rows should be read as "encoder + supervised bridge".
