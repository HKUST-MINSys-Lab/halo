> # ARCHIVED — not the current design. Do not cite.
> Superseded by `../../training/tokenizer/README.md`.
> Kept as a record of what we believed and why; see [`README.md`](README.md) for the
> full archive index and the pre-v4 number trap.

---

# Native Phase-A pretrain — pre-launch manifest (2026-07-19)

> **HISTORICAL MANIFEST.** This predates the consolidated objective and describes the old SupCon /
> A3 / multi-rail recipe. The authoritative current launch recipe — JEPA + augmentation VICReg,
> temperature sampling, and factored conditioning — is in
> [`training/tokenizer/README.md`](../../training/tokenizer/README.md), mirroring the live
> `PretrainConfig`. Use that for any launch.

Provenance record for the from-scratch **native-rate** Phase-1 (HALO representation) run. This
supersedes the old-corpus diagnostics in `PIPELINE_A_PREFLIGHT.md` (8-dataset / 57-label / 60 Hz).
Save this next to the run.

## Provenance
- **Clean revision:** launch from a clean checkout at commit `caf5c57` (or a later frozen commit).
  Per-run output dirs (`training/tokenizer/outputs/pretrain*/`) are now gitignored, so a run does not
  dirty the tree — checkpoints record a clean SHA, not `<sha>-dirty`.
- **Corpus fingerprint:** `99fa9ce6e0af4ad6` (stamped into every checkpoint via `corpus_fingerprint`).
- **Checkpoint contents:** encoder + heads + config + label_ids + git + corpus + corpus_fingerprint +
  optimizer/scheduler/scaler/RNG (warm-resume via `--resume`).

## Corpus (native alignment)
- **20 streams · 300,231 train / 42,909 val windows · 93 labels.** Subject-disjoint split after
  the 2026-07-25 event-grid migration and 95 implausible-window exclusions.
- **Native sampling rates:** 20 Hz (70,205 win) · 50 Hz (86,101) · 100 Hz (186,929). No 60 Hz resample
  — the filterbank is rate-invariant; HALO trains on real native rates + the `rate`/`window_crop` augs.
- All native grids finite, 6-channel `[acc xyz, gyro xyz]` pad+mask, accel in g, canonical labels.

## Model + training config (`PretrainConfig`)
| | |
|---|---|
| d_model / layers / heads / ff | 256 / 6 / 8 / 1024 (~7.3 M trainable) |
| batch | 32 classes × 8 = 256 |
| optimizer | AdamW, lr 3e-4, weight_decay 0.05, grad_clip 1.0 |
| schedule | 1k warmup → cosine to `steps` |
| objectives | A1 masked-latent + A2 SupCon (config-conditional) + A3 grounding (weight 0.1) |
| filterbank norm | calibrated over 50 augmented batches (frozen) |
| val | every 1k steps, kNN-BA, **40 windows/label stratified** (all classes scored), best.pt on val |

**Historical active augmentations (now exposed as `AugmentationConfig.phase_a()`):** window_crop,
channel_dropout (drops the whole gyro triad →
accel-only, the real deployment shift), rotation_3d, gravity-remove (p=0.15), rate (15–100 Hz), scale,
jitter, channel_text_phrase, channel_text_dropout. The permanently disabled label-text and time-warp
augmentation branches were subsequently deleted; Phase A does not consume activity-label text.

**Gravity is NOT aligned (design decision, 2026-07-19).** The signed-DC feature exposes gravity
direction (posture stays readable) and `rotation_3d` teaches pose-robustness; aligning had flattened
every posture's DC to +z and cancelled the rotation aug. Eval/inference match this (no align).
Consequence: with align off, `rotation_3d` (p=0.5, full SO(3)) is now *fully* effective (previously
the align undid most of it) — worth watching in the pilot.

**Augmentation magnitude telemetry** (`python -m training.tokenizer.aug_telemetry`, 200 windows):
value-space augs are gentle (jitter rel-L2 0.03, scale 0.06, amplitude/frequency preserved);
`rotation_3d` is large in rel-L2 (~1.4) but **norm- and cadence-preserving** (a valid re-orientation,
not corruption); `gravity`-remove is the biggest-magnitude physics change (RMS→0.42, ~90% of signal)
— which is why it sits at p=0.15; `rate`/`window_crop`/`channel_dropout` are structural and
physics-faithful (cadence drift ≈0). Nothing looks over-powering.

## Audit status — clean
Native-rate switch + two adversarial audits + external review, all fixes landed and tested (200 CPU
tests pass; CPU smoke train + warm-resume pass): honest coverage-complete val metric; WISDM 6-ch
accel+gyro merge; XRF AirPods off-by-one recovered; SP/NFI gap-split & signal-dedup; sampler without
replacement; partial-patch tail + fallback position; authoritative gravity_state; checkpoint
provenance/resume + output-dir guard.

## Step budget — DECIDE BEFORE THE PAPER RUN
The default `steps=20_000` dates from the old ~96 k-window corpus. Exposure (batch 256):
| steps | examples | epochs of the 305 k corpus |
|---|---|---|
| 20 k | 5.12 M | **16.8** |
| 40 k | 10.24 M | 33.6 |
| 60 k | 15.36 M | 50.4 |

The old run saw ~53 epochs of its (smaller) corpus; matching that exposure on the native corpus is
~**60–64 k steps**. Recommended path: a short **native pilot** (~10 k steps, ~30 min on the 4090) to
confirm the val-kNN curve rises and measure real throughput, then commit to a fixed final budget
(~50–60 k) and run it fresh with cosine annealing to that budget + best.pt selection. Do **not** train
20 k then resume into a 60 k cosine schedule — the LR schedule is set by `cfg.steps`, so that creates a
learning-rate discontinuity. Set the final budget upfront.

## Launch (local RTX 4090)
```bash
CUDA_VISIBLE_DEVICES=0 /home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m training.tokenizer.pretrain --device cuda --steps <FINAL_BUDGET> \
  --out training/tokenizer/outputs/pretrain_native
# Historical resume state was removed after completion; `best.pt` remains for reproduction.
```

## Profiling + config (2026-07-19, RTX 4090)
Profiled on the native corpus and tuned: **batch 512 (64×8)**, **num_workers 12**, **lr 4.2e-4** (√2 of
3e-4 for the 256→512 doubling), **30k steps = 50 epochs**. Step ~110 ms (~9.2 steps/s); GPU-compute
throughput knee is ~512 (no gain past it, and heavy patch scales ps=0.5 lose throughput); peak ~10.6 GB
of 24. Step breakdown: backward 58%, two forwards ~25%, text-encode ~9%, filterbank ~2%. TF32 enabled
for the fp32 regions. Full run ≈ ~1 hour.

**Historical validation result:** val loaders skipped the then-present A3 cadence/eigen DSP and used
parallel/persistent workers; A3 target computation has since been removed from the collator. kNN is vectorized
(cdist) — one val went **~571 s → ~12.6 s** (~45×), bit-identical metrics. Added a live **ConSE
text-cosine zero-shot probe** (`val_conse_ba`) alongside kNN-BA (best.pt still selects on kNN-BA).

**Known micro-opts NOT applied** (each ~1% of step, bit-exact, deferred — not worth churning the core
forward for): (1) filterbank DSP runs twice/step (frozen A1 target + trainable encoder on identical
patches) — could hoist the shared pre-proj feature; (2) channel-text fusion *pool* is text-only but
recomputed in both the masked & clean forwards — could pool once. Fold in if/when the encoder forward
is refactored anyway.

## Non-blocking caveats
- Peak-memory headroom is comfortable (~10.6/24 GB at batch 512; ~18.8 GB at 1024, not worth it).
- XRF contributes ~22–23 % of sampled class slots (it uniquely covers the most labels; 6 diverse
  configs, no single config dominates).
- Label-text synonym coverage is a Phase-B concern; Phase A does not consume activity-label text.
- After the HALO checkpoint: re-fit the ConSE baseline heads (auto-refit on next baseline run; now
  temperature-calibrated) + re-run the baseline table in the baseline env.
