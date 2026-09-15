# 2026-09-14 — Shared vs. fully-separate zero-shot/few-shot classifier parameters

**Status:** result. Sealed, manifest-matched, frozen-encoder ablation.
Answers a direct question: does giving the zero-shot and few-shot regimes their own,
completely disjoint classifier parameters (rather than one scoring rule sharing everything
but the final projection/λ, as designed in
`docs/design/SUPPORT_CLASSIFIER_DESIGN_20260914.md`) change anything?

**Encoder:** identical for both arms — `halo_fixed_mr_neighbors_8s_4res_e2e_20260913_continue95k`
(the neighbours-95k checkpoint), **frozen** (`--freeze-encoder`, `encoder_lr_scale=0.05` irrelevant
since gradients never reach it). Using a neutral, classifier-agnostic base encoder was deliberate:
an earlier attempt warm-started from `halo_fixed_mr_residual_v3_8s_4res_40k_20260914`, whose
encoder had been *co-trained* with the shared-parameter classifier, which would have confounded
"does separation help" with "was the encoder shaped for the shared head." That partial run and its
smoke are kept, not deleted, at `training/support_classifier/outputs/superseded/`.

**Runs**, both 40,000 steps, identical curriculum (`centring=support_mean`, `p_gt_present=0.5`,
`p_mask_candidate=0.25`, `p_mask_gt=0.1`, `enrollment_k=[1,2,4,8]`, `window_seconds=8.0`):

| arm | output dir | params (encoder+classifier) |
|---|---|---:|
| **shared** (design as specified — one trunk, separate output heads only) | `halo_fixed_mr_neighbors95k_frozen_residual_40k_20260914` | 2.203 M |
| **regime-split** (`RegimeSplitSupportClassifier`, `--regime-split`: two complete `ResidualSupportClassifier` instances, zero shared parameters, routed by whether any candidate retains support) | `halo_fixed_mr_neighbors95k_frozen_residual_regimesplit_40k_20260914` | 3.617 M |

The shared-arm run was started by the other agent and stopped mid-training (step 26,800/40,000,
no crash trace, no process found — cause unknown) while this run was in progress. It was **resumed
in place** from its last checkpoint (`--resume ... last.pt`, step 26,500) rather than restarted, so
the same run directory and identity is used throughout; nothing was deleted or renamed. The resume
needed two small, tested fixes, both applied to `training/support_classifier/train.py`:
1. A forward-compatible migration (`saved_trajectory["classifier_config"].setdefault("regime_split",
   False)`) — the checkpoint predates the `regime_split` field, same pattern as the other legacy-field
   migrations already in that function.
2. `--allow-resume-source-drift`, the sanctioned flag for continuing when checked-out source has
   moved since the checkpoint was written; new provenance is recorded.

**Evaluation:** `sealed_halo_fixed_mr_neighbors95k_frozen_residual_shared_40k_20260914` and
`..._regimesplit_40k_20260914`, `--models halo` only (baselines unaffected, already cached).
A real bug was found and fixed en route: `sealed_eval.py`'s `_parameter_count_m` still referenced
`ResidualSupportClassifier` directly after the constructor call sites were switched to the
`build_support_classifier` factory, crashing on any residual checkpoint. Fixed to use the factory
(which also makes the regime-split parameter count correct rather than silently wrong). Full suite
green (794 passed, 1 skipped) before and after.

---

## Result — dataset-balanced macro F1, 8 s windows, single-device, `halo-classifier` readout

| k | shared | regime-split | Δ (split − shared) |
|---:|---:|---:|---:|
| 0 | **45.33** | 44.20 | **−1.13** |
| 1 | 58.39 | 58.42 | +0.03 |
| 4 | 66.66 | 67.16 | +0.50 |
| 8 | 69.37 | 69.92 | +0.55 |
| 16 | 70.97 | 71.54 | +0.57 |
| 32 | 71.57 | 72.00 | +0.42 |
| 64 | 72.19 | 72.90 | +0.71 |
| 128 | 73.97 | 74.65 | +0.68 |

The parameter-free readouts (1-NN, prototype, ridge, centred neighbours / "residual-off") are
**bit-identical** between the two arms at every k, confirming the encoders and features truly
match — the only difference is the classifier.

## Multi-device cells, `halo-classifier`, 8 s (single cell each — read as a signal, not a mean)

| n_devices | k=0 Δ | k=1 Δ | k=8 Δ | k=32 Δ |
|---:|---:|---:|---:|---:|
| 3 | **−12.34** | −0.06 | +2.86 | +1.76 |
| 4 | **−10.26** | −0.64 | −0.26 | −0.13 |

## Interpretation

**Zero-shot (k=0): sharing wins, clearly and consistently.** −1.1 to −2.2 macro F1 single-device
across all three window lengths, and a much larger −10 to −12 on the two multi-device composite
cells. Per-dataset at 8 s, shared wins 5/11 and split wins 6/11 cells, so it is not one dataset
driving it — but the margin on the cells that matter most (RealWorld's three placements, where
shared wins by 6–17 points each) is what drives the aggregate. **Mechanism:** the shared trunk lets
the sensor→text alignment ride on the far richer few-shot metric signal (millions of same-class
pairs vs ~90 label texts), exactly the argument made for sharing in the original design (§2.6). Full
separation removes that cross-training entirely, and the zero-shot head is left to learn cross-modal
alignment from a much thinner signal alone.

**Few-shot (k≥1): separation wins, small and consistent.** Essentially flat at k=1 (+0.03), then a
steady, small edge from k=4 up (+0.4 to +0.7), largest at k=64/128 (+0.7). Plausible mechanism: a
dedicated few-shot head is not fighting the zero-shot objective for the same output projection
capacity — but the effect size here is inside or barely above the ~1-point enrollment-draw noise
established in the 2026-09-14 neighbour diagnostics, so **this direction should not be treated as
established** without a second seed.

**The high-k regression is not fixed by separation.** Both arms still lose to their own
parameter-free floor from around k=4 upward — shared reaches −3.15 macro F1 below centred
neighbours and −5.40 below ridge at k=128; regime-split narrows this only slightly, to −2.25 and
−4.51. This directly tests the diagnosis in
[2026-09-14-residual-classifier-first-result.md](2026-09-14-residual-classifier-first-result.md)
§3, which attributed the regression to the λ bucket schedule (`(0,1,2,4,8)`, so every k from 8 to
128 shares one λ value trained mostly at the low end of that range) rather than to parameter
sharing. **The diagnosis survives**: separating the parameters measurably helps at high k but does
not come close to closing the gap, so the λ schedule (and the short k=1..8 training range) remains
the primary suspect, not the sharing decision.

## Net read

Neither arm dominates. Sharing is better where the two regimes need each other (zero-shot leans on
few-shot's data volume); separation is marginally better where they might otherwise interfere
(high-k), though that margin needs a repeat run to trust. Given zero-shot is currently the system's
biggest lead over every baseline (see
[2026-09-14-residual-classifier-first-result.md](2026-09-14-residual-classifier-first-result.md)),
and separation's few-shot gain is small and still short of ridge either way, **this does not change
the recommendation to keep the shared-trunk design** as specified. It does add weight to fixing the
λ schedule next, since that is now confirmed to be the dominant lever on the high-k regression.

## Caveats

- Single run per arm; the k≥1 separation edge (+0.4 to +0.7) is close to the ~1-point
  enrollment-draw noise floor and needs a second seed before it is treated as real.
- Multi-device deltas are one sealed cell each per (dataset, k) — not an average, no error bars.
- Both encoders are frozen at the neighbours-95k checkpoint; this result says nothing about
  regime-split vs shared under joint encoder+classifier training (the promoted recipe).
- `parameters_m` in the result rows includes the frozen encoder (0.789 M) plus the classifier, so
  2.203 M / 3.617 M above are total, not classifier-only (1.414 M / 2.828 M classifier-only).

## Links

- Design: `docs/design/SUPPORT_CLASSIFIER_DESIGN_20260914.md`
- Prior result and the λ-schedule diagnosis this entry tests:
  [2026-09-14-residual-classifier-first-result.md](2026-09-14-residual-classifier-first-result.md)
- Superseded partial run: `training/support_classifier/outputs/superseded/README.md`
