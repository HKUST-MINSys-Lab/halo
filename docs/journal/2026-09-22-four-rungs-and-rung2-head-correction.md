# Addendum: four rungs, and rung 2 does not use the v4 head

Date: 2026-09-22 (late). Status: **decision record; nothing run.** Supersedes two points in
[the evening entry](2026-09-22-three-regimes-and-unsupervised-adaptation-decisions.md); everything
else there stands. Live plan: [`docs/overview/roadmap.md`](../overview/roadmap.md).

## 1. Four rungs, not three regimes

Alex restated the ladder as four rungs, hardest first: (1) new dataset, nothing known; (2) roster
known, no labels; (3) some labels, parameters frozen; (4) some labels, fine-tuning allowed. The
evening entry's "regime 3" fused rungs 3 and 4. The roadmap and thesis now use four rungs; the
evening entry's numbering is left as written, and the roadmap header says how they map.

**Rung 3 is the existing work** — v4's 333 sealed and 737 scenario cells — and is the case study.
It is written as an **encoder** result with v4 as the vehicle, because that is what the numbers say
(sealed aggregate, 8 s, dataset-balanced macro-F1, RESULTS.md): with no learned head, the HALO
encoder under plain 1-NN beats every baseline under the same readout at every k (70.6 vs UniMTS
66.9 at k=8, 76.0 vs 73.9 at k=128); the learned head adds value at k ∈ {0, 1} (51.7 vs 38.6 at
k=0) and trails its own parameter-free floor at k ≥ 8 (71.5 vs 73.3 at k=8; 73.4 vs 77.0 at k=32).
Presenting the head as the claim would invite the high-k regression, which was the old roadmap's
open question and is now simply not the paper's question.

**Rung 4's priority is flipped** from Alex's first phrasing ("above 80 %, preferably beating the
baselines"): HALO fine-tuned ≥ every baseline fine-tuned at matched k is the must; the crossover
against a specialist trained from scratch on the same k windows is the absolute claim; ~77–80 %
(DAGHAR LODO — harmonised, ~5 classes, 225 subjects) is context, not pass/fail. Our protocol is six
unharmonised datasets at ~8.7 classes scored by macro-F1, so an absolute target on it is not
apples-to-apples and committing to one would manufacture a miss. Alex agreed.

## 2. Rung 2 does not use the v4 head; `ROLE_UNLABELED` withdrawn

The evening entry (§3, item 4) registered a fifth token role, `ROLE_UNLABELED`, acting through
v4's residual, as the mechanism by which unlabelled pools reach the model. Alex asked whether v4 is
used on rung 2 at all. It is not, and it cannot be: the invariant is that inference is identical for
all six encoders, so **the transductive method is the head**, and a learned head between HALO's
encoder and that method would be a private inference path — exactly what would let a reviewer
attribute the gain to the head rather than the encoder and its training. The pool enters training
where it enters inference: through the unrolled transductive iterations. What is trained is the
encoder, the recording pool and `p_text` (HALO's k=0 text prototypes). The v4 head — residual
stack, gated blend, corruption auxiliary, unenrolled calibration — is the rung-3 vehicle only.

The mechanism I had written was Design B (HALO keeps its own head and consumes unlabelled tokens),
which is incompatible with the invariant agreed the same evening. Alex's question caught the
contradiction; recorded here so the paper does not inherit it.

Consequences, now in the roadmap:

- **Two N=0 checks**, not one: architectural bit-identity for the same checkpoint with the pool
  removed; and the retrained encoder's own N=0 k-curve must not fall below v4's encoder at any k.
  The sealed table is the N=0 column for v4's encoder, not for the new arm.
- **Three baseline tiers**: frozen released checkpoint + method (off-the-shelf reality);
  corpus-matched arms + method (removes "you just saw our corpus" — the arms exist, trained with
  differentiable neighbours, so this is a new readout on existing checkpoints and costs no
  training); HALO unrolled.
- **A four-step HALO ladder**, all from random initialisation on our corpus: plain CE → +
  heterogeneity curriculum (= v4's encoder) → + unlabelled-pool episodes under the
  differentiable-neighbour objective, no unrolling (objective-matched to the corpus-matched arms) →
  + unrolled transductive loss. Step 3 exists so that "our encoder is better" and "our loss is
  better" can be told apart; without it the tier-2 comparison would confound architecture with
  training objective.
- **Optional tier 4:** the unrolled loss dropped into HARNet's matched arm (32 min) as a transfer
  check. Not required for the claim; a reviewer asks tier 2's question first.
