# Tier 1 (rung 1): corpus-matched HARNet arm — 2026-09-26

**What:** the HARNet trunk trained from its released weights' architecture on **our** 8 training
sources with our pipeline (`halo-train --encoder-arch harnet --classifier neighbors --steps 40000`,
differentiable-neighbour head), then scored under exactly the tier-1 protocol of
`results/artifacts/rung1_tier1_20260925`. It separates "our corpus and training recipe" from "the
HALO architecture/text" in the tier-1 gap. Checkpoint
`runs/support-classifier/matched_harnet_40k_20260925/last.pt` (sha256 prefix cbf7741f7951460e);
training config and log in this directory.

**Route:** the neighbour head has no text projection, so zero-shot scores come from a ridge text
bridge refit on the non-held-out training bank (`halo_text_bridge_heldout_refit`) — close to, but
not identical with, the ConSE bridge the released HARNet-5/-10 use. Calibrated T = 7.43.

## Result (dataset-balanced macro-F1)

| encoder | anchor | N=0 | N=all | pool effect |
|---|---:|---:|---:|---:|
| HALO v4 | 47.4 | 50.6 | 51.9 | +1.3 |
| **HARNet, corpus-matched** | 33.7 | 39.3 | 40.8 | +1.5 |
| UniMTS (released) | 31.9 | 35.1 | 34.7 | −0.4 |
| HARNet-5 (released) | 31.2 | 30.5 | 31.1 | +0.6 |

The matched arm is the strongest non-HALO row and gains from the pool about as much as HALO does.
Of HALO's ~21-point lead over released HARNet-5 at N=all, ~10 points are reproduced by training
HARNet on our corpus with our pipeline; ~11 remain with HALO. Per dataset: the matched arm beats
released HARNet-5 on 5 of 6 datasets but trails HALO on all 6 (see `RESULTS.md`).

## Caveats

- One seed. Internal validation (training sources) peaked at 0.595 (step 32.5k) and ended at 0.580;
  `last.pt` is used per the declared policy.
- Different zero-shot route from HALO (ridge bridge vs `p_text`), so the anchor comparison mixes
  route and representation; the N-curve comparison is within-arm.
- LiMU-BERT and UniMTS matched arms are not trained yet (UniMTS ≈ 13.6 h; cloud).
