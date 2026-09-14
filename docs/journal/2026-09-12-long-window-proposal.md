# Design proposal: long analysis windows for the continuous-kernel arm

**Date:** 2026-09-12
**Status:** immutable entry — see `docs/journal/README.md`. This records a **proposal and its
feasibility measurement**, not a result. Nothing here has been built. A later entry should record
what was actually tried and what it produced.
**Origin:** Alex's idea, raised after the encoder/JEPA findings in
`2026-09-12-jepa-and-encoder-findings.md`.

## The idea

The two live encoder arms have genuinely different logics, and the fixed arm's logic is already
coherent:

- **Fixed multiresolution filterbank.** Each patch is a short window; we compute frequency-domain
  features for it; *temporal* structure is supplied later by self-attention across patches. There
  is no compositional stage in between and there doesn't need to be — attention is the
  contextualizer. This is a clean design and it works.
- **Continuous-kernel multi-span.** Copying the fixed arm's "one spectral column per patch, then
  attention" shape throws away the reason for having continuous kernels at all. The proposal:
  start from a **substantially longer analysis window — tens of seconds, possibly a minute** —
  so that the kernels *and* convolutional modules stacked on top of them have room to build a
  hierarchy in the time direction. The output is then what a CNN over an image produces: a
  sequence of vectors along time, each corresponding to a sub-patch of the original window.
  That sequence is directly usable as the JEPA prediction target space, and for classification it
  can be fused by attention pooling — over time, and across the different kernel-duration groups —
  into a single recording vector.

## Why this is the right axis to push on

Two independent reasons, one architectural and one that speaks directly to the JEPA negative
result recorded the same day.

**Window length is a second source of headroom, independent of sampling rate.** The earlier
analysis concluded that low, variable sampling rates (20–240 Hz) leave little room to *stride*
down the way a 16 kHz audio frontend does. That conclusion was about the sample axis; it does not
constrain the frame axis. At a 16 frames/s token grid, an 8 s window is 128 frames and supports
perhaps two stride-2 stages; a 60 s window is 960 frames and supports five, or equivalently a
dilated stack whose receptive field spans tens of seconds. The "not enough room to convolve" ceiling
is real for raw samples and largely dissolves once the window is long.

**It attacks the measured cause of the JEPA failure more directly than the fixes already planned.**
The probes behind `2026-09-12-jepa-and-encoder-findings.md` found the predictive objective
saturating within ~1,500 steps with loss flat across horizon bins (0–1 s, 1–2 s and 2–3 s
indistinguishable), because over a few seconds of free-living wrist motion the future genuinely
does look like the present. On an 8 s window there is no horizon at which that stops being true.
On a 60 s window, predicting 10–30 s ahead requires anticipating *activity transitions and bout
structure*, which is not trivially solved by copying the present. If the aim is for JEPA to learn
temporal dynamics, lengthening the window may matter more than the objective-level fixes already
queued (instance-normalised targets, motion weighting, positional predictor memory).

Note also that the attention-pooling half of the proposal partially exists: `RecordingAttentionPool`
(`model/tokenizer/encoder.py`) is a learned-query masked set-attention pool added during the
2026-09-11 consolidation, currently applied over the flattened token set. Pooling across duration
groups as well as time is a natural extension of it rather than a new mechanism.

## What the data permits (measured 2026-09-12)

The binding constraint is the length of a **continuous single-label execution** — the leakage unit.
A window can never exceed it. Measured across all three roster roles (seconds of real signal per
execution; `frac ≥ 30 s` / `≥ 60 s` are the share of executions long enough to supply one window):

| role | source | median s/exec | ≥ 30 s | ≥ 60 s |
|---|---|---:|---:|---:|
| pretrain | capture24_pretrain | 93,600 | 0.99 | 0.99 |
| pretrain | nymeria_xsens | 932 | 1.00 | 1.00 |
| pretrain | extrasensory_pretrain | 16 | **0.00** | **0.00** (max 40 s) |
| head train | forth_trace | 1,028 | 1.00 | 1.00 |
| head train | realdisp | 929 | 1.00 | 1.00 |
| head train | dsads | 300 | 1.00 | 1.00 |
| head train | wisdm | 180 | 0.99 | 0.99 |
| head train | kuhar | 17.5 | 0.29 | 0.13 |
| head train | hhar | 16.8 | 0.26 | 0.11 |
| head train | harmes | 23.1 | 0.02 | **0.00** |
| head train | xrf_v2 | 8.0 | **0.00** | **0.00** (max 15 s) |
| sealed | realworld | 618 | 1.00 | 1.00 |
| sealed | shoaib / ut_complex | 180 | 1.00 | 1.00 |
| sealed | inclusivehar | 60 | 1.00 | 0.80 |
| sealed | motionsense | 54 | 0.80 | 0.48 |
| sealed | usc_had | 30 | 0.51 | **0.08** |

**Read:** the proposal is comfortably feasible for **pretraining** — Capture-24 (83% of the
label-free corpus) is continuous multi-day and Nymeria runs ~15 min per sequence. ExtraSensory is
chunked at ~16–40 s by construction and would cap out; it is ~15% of label-free windows.

It is **not** feasible at 60 s for the supervised and sealed stages without changing the rosters:
xrf_v2 (max 15 s) and harmes (0% ≥ 60 s) would be lost from head training, usc_had (8%) would be
effectively lost from the sealed test and motionsense halved. At 30 s the damage is smaller but
still real (xrf_v2 gone, harmes ~98% gone, hhar/kuhar ~72% gone).

## The shape this suggests

The asymmetry is not a problem to solve — it is arguably the correct structure, and the
architecture already supports it. The encoder is duration-agnostic by construction (physical-time
positions, RoPE, duration embeddings, token masks), so **pretrain long where the data is
continuous and unlabelled; adapt and evaluate short where labels and episodes live.** Two design
consequences follow if this is pursued:

1. The convolutional stack's dilation schedule should be chosen so that the *early* blocks cover
   1–3 s — the range that still fires on a 6 s episode recording — while later blocks extend to
   tens of seconds and are exercised only during pretraining. A stack whose useful receptive field
   only exists at 60 s would transfer nothing to the downstream stage.
2. Mixed-duration batching is native to the design and should be used rather than excluding
   ExtraSensory: long windows from Capture-24/Nymeria, short from ExtraSensory, with the token
   mask doing its job.

Separately, a modest lengthening of the *episode* recordings (6 s → ~20 s) is feasible for six of
the eight head sources and five of six sealed streams, and would close part of the gap without
touching the rosters. Whether that is worth a grid rebuild is a separate question.

## Cost and risk, for whoever picks this up

- **Compute.** Frontend analysis cost is linear in frames. A 60 s window at 16 frames/s is ~7.5×
  the frames of the current 8 s contract; at constant memory the batch size drops roughly
  proportionally (the multispan run currently reserves ~4.3 GB at batch 384). Expect a 20k-step
  pretraining run to go from ~30 min to a few hours.
- **Data rebuild.** `build_grids.py` already exposes `--window-seconds`, but the grid path layout
  (`grids/<alignment>/<stream>/`) does **not** encode window length, so a rebuild at a new length
  overwrites the existing grids in place. Any experiment here must write to a separate
  `--out-root`, not least because an 8 s-contract pretraining run was in flight on 2026-09-12.
- **Contract guards.** `PRETRAIN_WINDOW_SECONDS = 8.0` (`data/pretraining/corpus_plan.py`) and
  `WINDOW_SECONDS = 6.0` (`data/scripts/build_grids.py`, `training/tokenizer/pretrain_data.py`)
  are enforced by `validate_source_window_contract`; a long-window corpus is a new contract value,
  not a silent override.
- **Ordering.** This does not replace the frontend work in `docs/design/FRONTEND_PLAN_20260912.md`
  — the convolutional stage is the thing that makes a long window *useful*, and it should be built
  and shown to help at the current window length first. Long windows multiply its value; they do
  not substitute for it.

**Related:** `docs/design/FRONTEND_PLAN_20260912.md` (section X9 records this as a gated follow-on),
`docs/design/CONTINUOUS_KERNEL_FRONTEND.md`, `2026-09-12-jepa-and-encoder-findings.md`.
