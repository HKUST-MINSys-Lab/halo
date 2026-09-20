# T7, step 40k — deployment scenarios

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

Seven scenarios at 8 seconds; manifests identical to the released-baseline scenario artifact on all
737 shared cells. T7 leads v3 on partial coverage (+1.0 at k=8), cross placement (+0.5 at k=8, the
first arm in this lineage to lead there), device set (+2.0), rate mismatch (+0.8) and cross dataset
(+1.6 at k=32), and its partial-coverage split is the best of any arm (62.0 enrolled / 60.2
unenrolled, harmonic mean 61.1). It trails badly on the new-domain cell; see the README summary.
`halo_t7_mmfit_diag` records the per-branch diagnosis of that regression.
