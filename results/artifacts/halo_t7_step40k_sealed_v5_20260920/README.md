# T7, step 40k — sealed evaluation

Run date: 2026-09-20.
T7 is the T6 recipe plus the primitive semantic branch (`--semantic-mode text+primitives`), trained
end to end for 40,000 steps from code `3876dd0`. Primary checkpoint `last.pt`, step 40,000,
declared before any sealed number existed. Designs:
`docs/journal/2026-09-20-classifier-t6-design.md` and
`docs/journal/2026-09-20-primitive-semantic-path-design.md`.

**Result: the best sealed arm produced so far, and disqualified from promotion by one scenario.**
T7 beats the promoted v3 control at every k on the sealed aggregate, including k=0 where T6
trailed, and its deficit against its own support vote is down to -0.4 at k=128 (v3: -3.6). The two
semantic halves are genuinely complementary on seen vocabulary: at 8 s k=0 the text half alone
scores 50.6, the primitive half alone 42.9, and their fixed equal-weight combination 51.9.

But on MM-Fit, the only foreign-vocabulary source, the primitive half is **confidently wrong**
(3.7 macro F1 against a 10.0 chance level) rather than merely uninformative. Because both semantic
paths are log-probabilities with unbounded negative range, that half can flip an argmax even at a
small blend weight, and the classifier falls to 45.8 at k=8 where T6 reaches 61.0 and its own
support vote reaches 47.5. This is dilution of a strong path by a weak one, in exactly the
condition the primitive vocabulary was designed to serve.

All 39 cells; manifests identical to the released-baseline artifact on all 333 shared cells. At 8
seconds the classifier scores 51.9/62.7/73.1/76.2/77.9 at `k=0/1/8/32/128` against v3's
51.7/62.4/71.5/73.4/74.4 and T6's 50.2/62.4/72.7/75.8/76.5.
