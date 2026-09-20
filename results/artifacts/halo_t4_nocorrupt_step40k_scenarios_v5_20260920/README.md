# T4 corruption-free arm, step 40k — deployment scenarios

Run date: 2026-09-20.
The T4 architecture (`support_classifier_v4`) trained identically to
`halo_evidence_gated_v4_40k_20260920` — same code `ba5c3c5`, same seed, same 40,000 steps, same
gate bounds — with **one flag changed: `--text-corruption-probability 0`**. It is an arm of T4,
not a new try, and it exists to answer whether the label-text corruption curriculum caused T4's
encoder regression. Primary checkpoint `last.pt`, step 40,000, declared before any sealed number
existed.

**Result: the curriculum is load-bearing, and it is what keeps lambda honest.** Without it lambda
goes to 0.81-0.82 at EVERY support count (flat, with 1.3% of candidates pinned at the 0.85 bound)
instead of ordering itself by evidence, and the classifier falls 6 to 15 points BELOW its own
support vote — a far worse version of the pathology this design was built to fix, contained only
by the bound. Internal validation is meanwhile *higher* than the corrupted arm's (enrolled 0.787
against 0.778), which is the pathology in one line: the panel shares the training vocabulary, where
trusting label meaning is the correct policy.

The encoder does partially recover (cosine 1-NN 71.1 against T4's 70.4 at 8 s, k=8; v3 scores
72.0), so roughly half of T4's 1.6-point encoder gap is attributable to the curriculum and the
rest is not explained here.

Every scenario is worse than both T4 and v3, most sharply the new-domain cell (42.5 against T4's
59.0) and partial coverage (49.6 against 58.1). Manifests identical to the released-baseline
scenario artifact on all 737 shared cells.
