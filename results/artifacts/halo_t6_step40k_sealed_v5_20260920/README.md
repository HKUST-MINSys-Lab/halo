# T6, step 40k — sealed evaluation

Run date: 2026-09-20.
T6 is the `support_classifier_v4` architecture with two recipe changes over T4, trained end to end
for 40,000 steps from code `3876dd0` (= lab `main` at launch): **corruption as a gate-only
auxiliary** (`--text-corruption-mode auxiliary`, so a deranged-roster view of every eligible
episode trains the blend gate and nothing else, displacing no clean episode) and a **label-blind
calibration term** for candidates with no support of their own (`--unenrolled-calibration`).
Everything else matches T4 and every arm since 2026-09-18. Primary checkpoint `last.pt`, step
40,000, declared before any sealed number existed. Design:
`docs/journal/2026-09-20-classifier-t6-design.md`.

**Result: T6 matches or beats the promoted v3 control nearly everywhere, and the over-trust
pathology is essentially closed.** Against its own untrusted support vote the classifier is
positive through k=8 and only -0.9 at k=128, where v3 is -3.6 and T4 -1.9. It beats v3 by +1.5 to
+2.4 at every k >= 4 on the sealed aggregate, and the encoder regression that made T4 a wash has
recovered (cosine 1-NN 71.7 at 8 s k=8, against T4's 70.4 and v3's 72.0).

All 39 cells, six datasets, 4/8/16-second windows, `k = 0..128`; manifests identical to the
released-baseline artifact on all 333 shared cells. At 8 seconds the classifier scores
50.2/62.4/72.7/75.8/76.5 at `k=0/1/8/32/128` against v3's 51.7/62.4/71.5/73.4/74.4.
