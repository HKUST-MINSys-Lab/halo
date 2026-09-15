# Project journal

A dated, append-only record of how this project has changed: architecture pivots, what was
tried, what worked, what didn't, and why. It exists so a future agent (or Alex, before a
supervisor meeting) can reconstruct the arc of the project and pull material for slides without
re-deriving it from git log and scattered memory notes.

## Rules

- **One file per dated entry**, named `YYYY-MM-DD-short-slug.md`. Multiple entries on the same
  date get distinct slugs.
- **Entries are immutable once written.** Do not edit, rewrite, or delete a past entry to
  "correct" it in place — write a new dated entry that supersedes or corrects it, and say so
  explicitly (`Supersedes: 2026-09-12-....md`). This preserves the actual history of what was
  believed/found at each point, which is the point of a journal.
- **Do not modify any file in this folder without the user's explicit permission**, including
  this README. If something here looks wrong, flag it in a new entry or ask — don't silently fix
  the old one.
- Entries should be self-contained: state the date, what changed or was found, why it matters,
  and links to the relevant docs/commits/memory files rather than duplicating their content.
- This folder is a narrative supplement to `docs/HISTORY.md` (which tracks *artifacts* — tags,
  branches, archived systems) and `docs/results/RESULTS.md` (which tracks *promoted results*).
  The journal tracks the *reasoning and trajectory* between those.

## Index

| date | entry | summary |
|---|---|---|
| 2026-09-12 | [project-history-v1-to-now.md](2026-09-12-project-history-v1-to-now.md) | Retrospective: HALO v1 → v2 → clinical pivot (abandoned) → current support-conditioned design. Motivation and contribution at each stage. |
| 2026-09-12 | [jepa-and-encoder-findings.md](2026-09-12-jepa-and-encoder-findings.md) | JEPA pretraining implemented and evaluated: it does not beat end-to-end training outside noise, with a proposed explanation. Encouraging results (k up to 128, low parameter count vs baselines). Open encoder-design questions. |
| 2026-09-12 | [long-window-proposal.md](2026-09-12-long-window-proposal.md) | Proposal (not yet built): long analysis windows for the continuous-kernel arm, with the execution-length measurement that says pretrain-long / adapt-short is the only feasible shape. |
| 2026-09-12 | [fixed-filterbank-decision.md](2026-09-12-fixed-filterbank-decision.md) | **Decision:** focus on the fixed multiresolution filterbank and extend it to 0.5/1/2/4/8 s patches (8 s windows); continuous-kernel arm demoted to ablation. Frequency-resolution and execution-length measurements, plus supporting literature. |
| 2026-09-12 | [frontend-efficiency-and-jepa-window.md](2026-09-12-frontend-efficiency-and-jepa-window.md) | How to add 4 s/8 s patches without paying for them (per-resolution DFT length = 0.67x today; decimating to a 40 Hz analysis rate = 0.13x), and why the JEPA window should be 16 s rather than 32 s (32 s costs ExtraSensory entirely). |
| 2026-09-12 | [spectral-frontend-literature.md](2026-09-12-spectral-frontend-literature.md) | Literature check on fixing the frontend: acceleration is 99% below 15 Hz and discrimination plateaus at a 40 Hz sampling rate, so a 0.3-15 Hz bank is near-sufficient for frequency content; plus the honest counter-evidence for convolutional depth. |
| 2026-09-13 | [filterbank-polarization-features.md](2026-09-13-filterbank-polarization-features.md) | **Designed, not built:** gravity-referenced polarization features (`vert`/`circ`/`spin`) for the fixed filterbank, recovering inter-axis phase the magnitude spectrum discards. Published precedent (Samson & Olson 1980; Gonella 1972; Mooers 1973; Mizell 2003; Kobayashi et al. 2011) and how ours differs from the closest prior work. |
| 2026-09-13 | [why-a-simple-encoder.md](2026-09-13-why-a-simple-encoder.md) | **Design rationale:** why a fixed engineered low-parameter frontend beats a learned deep encoder here — the skeleton-descriptor thought experiment, why we refuse IMU→pose (hallucination), why capacity is actively harmful under label scarcity + heterogeneity. Citations: ZARA (ACL 2026), Haresamudram IMWUT 2022, Zeng AAAI 2023, Deep Inertial Poser. Includes the honest wrinkle that ZARA's own ablation credits a *learned* retrieval embedder with +10.6 points. |
| 2026-09-13 | [jepa-value-measured.md](2026-09-13-jepa-value-measured.md) | **Result (sealed, manifest-matched):** k-curve table, fixed filterbank only. JEPA frozen beats random frozen by +9.2 / +12.8 macro-F1 (6/6 streams) — but JEPA-adapted beats random-adapted by only +0.7 / +1.2 with the sign flipping on 2/6 streams. JEPA substitutes for early supervised training rather than adding to it. Zero-shot entirely unmeasured. |
| 2026-09-13 | [retired-jepa-promoted-results.md](2026-09-13-retired-jepa-promoted-results.md) | Exact promoted JEPA k-curve rows, archived when future-JEPA was retired from the active recipe. |
| 2026-09-13 | [mantis-comparison-and-zara-correction.md](2026-09-13-mantis-comparison-and-zara-correction.md) | **Corrects** the ZARA reading in why-a-simple-encoder.md (the ablation's only non-learned control is raw DTW, so it does not show learned>engineered). Plus Mantis as the architectural foil — it interpolates to 512 samples and instance-normalises away gravity, the two things we refuse — what we're missing (calibration, transient structure), and verified baseline parameter counts (MOMENT 385M). |
| 2026-09-14 | [residual-classifier-first-result.md](2026-09-14-residual-classifier-first-result.md) | **Result (sealed):** first evaluation of the learned residual classifier. Zero-shot 38.3 → **49.3** (+13.7 over HARNet, the prior leader); k=1 +1.9 over every parameter-free readout; centring alone worth +1.7 at k=8–32 for free. But the learned residual **regresses −1.9 to −6.3 from k=4 upward**, traced to a λ bucket table ending at 8 and a training k range ending at 8. Encoder quality unchanged by the curriculum change. |
| 2026-09-14 | [evaluation-rebuild.md](2026-09-14-evaluation-rebuild.md) | **Protocol change:** 4/8/16 s evidence budgets, multi-device composite cells, RealWorld/Shoaib placement streams, six enforced fairness guarantees. Records what it invalidates (the 6 s JEPA ladder) and the coverage cost. HALO's parameter-free hierarchical mean beats UniMTS's native skeleton fusion at k=1 (+11.1/+8.9 vs +3.7/+4.1). |
| 2026-09-14 | [baseline-failure-analysis.md](2026-09-14-baseline-failure-analysis.md) | **Per-baseline failure modes** and the scenario matrix: HARNet fails at short windows, UniMTS on wrist/hand activities (−10.2), LiMU-BERT at k=1 and without a gyro, NormWear collapsed (effective rank 1.4 in 2048 dims). Plus the neighbour diagnostics that motivated the classifier: 92.5% top-5 at k=1, 15.6% rank-1-but-outvoted at k=8. |
| 2026-09-14 | [design-narrative.md](2026-09-14-design-narrative.md) | **The argument, end to end:** why JEPA and the continuous kernel were dropped, why the filterbank was improved as it was, what is and is not a controlled comparison, the classifier's results, what went well and what did not, and the limitations. Slide backbone. |
| 2026-09-14 | [RESIDUAL_CLASSIFIER_AUDIT_PROFILE_20260914.md](RESIDUAL_CLASSIFIER_AUDIT_PROFILE_20260914.md) | Independent code audit and GPU training profile of `support_classifier_v3` by the other agent (filename predates this folder's `YYYY-MM-DD-slug` convention). Findings 1-6 implemented and verified. |
| 2026-09-14 | [multi-device-pooling-correction.md](2026-09-14-multi-device-pooling-correction.md) | **Correction** to evaluation-rebuild.md and design-narrative.md: HALO's multi-device fusion is a **learned `RecordingAttentionPool`** over every `patch x sensor` token (0.133M params, device identity supplied as text), **not** the parameter-free hierarchical mean those entries credited. The mean path exists but is overridden in every trained checkpoint. All numbers stand; the "a mean beat a purpose-built graph" framing does not. Owes a pool-vs-mean ablation. |
| 2026-09-14 | [regime-split-vs-shared-classifier.md](2026-09-14-regime-split-vs-shared-classifier.md) | **Result (sealed, frozen-encoder ablation):** fully-separate zero-shot/few-shot classifier parameters vs. the shared-trunk design. Separation loses at k=0 (**−1.1 to −2.2**, up to **−12** on multi-device) but gains a small, consistent edge at k≥4 (**+0.4 to +0.7**, within noise). Confirms the high-k regression is **not** a sharing artifact — both arms still trail their own parameter-free floor — pointing at the λ schedule as the real cause. Net: keep the shared-trunk design as specified. |
