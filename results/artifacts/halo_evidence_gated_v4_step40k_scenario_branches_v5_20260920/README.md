# HALO classifier v4, step 40k — scenario branch diagnostics

Run date: 2026-09-20.
Architecture `support_classifier_v4` (the evidence-gated blend), trained end to end for 40,000
steps from code `ba5c3c5` (= lab `main` at launch) with `--classifier evidence_gated`, the Stage A
curriculum defaults shared with every arm since 2026-09-18, and v4's own defaults: per-support
trust bounded to |t| <= 2, per-candidate blend weight bounded to lambda <= 0.85, and the label-text
corruption curriculum at 0.25. Semantic mode is the default `text`; the primitive semantic path is
a separate experiment and was not enabled. **Checkpoint: `last.pt`, step 40,000**, the fixed-budget
primary declared before any sealed number existed. SHA-256 prefix `4e0070891dd2c7842bbd`.

**Result: the diagnosed failure is roughly halved, and the arm lands at parity with the promoted
v3 control.** Against its own untrusted support vote the classifier is positive to k=4 and -1.9 at
k=128, where v3 was positive only to k=2 and -3.6 at k=128. Absolute scores match v3 within about a
point because v4's encoder came out slightly weaker (cosine 1-NN 70.4 against 72.0 at 8 s, k=8).

Diagnostic (non-headline) scenario run on the same manifests with the label-meaning branch, the
trust-weighted support vote, the untrusted support vote, and the trust-off blend.
