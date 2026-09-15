# Ablation plan — strengthening the contribution claims (2026-09-15)

**Status:** plan, not run. Written for a separate agent to execute. Results come back as
`docs/design/ABLATION_RESULTS_20260915.md` (format in §7); the journal entry is written afterwards
by the planning agent, not the executing agent.

## 0. The claim these ablations serve

> A 2.2M-parameter support-conditioned HAR model that accepts any channel count, sampling rate,
> sensor type and placement through text-conditioned sensor tokens, and answers from label text
> alone, from one enrolled example, or from many, through a single head with a parameter-free floor.
> It is trained by a curriculum that targets partial-information deployment: missing labels in the
> support set, hidden candidates, and missing sensors or devices.

Every clause of that sentence is currently either measured, argued, or owed. The ablations below
turn "argued" into "measured" and tell us which clauses to drop. They are grouped by clause, and
each names the single question it decides.

## 1. Ground rules (read before running anything)

1. **Do not delete, rename or overwrite anything under `training/support_classifier/outputs/` or
   `evaluations/`.** New runs get new directories. If a run is wrong, move it to
   `outputs/superseded/` with a README line, as was done on 2026-09-14.
2. **Do not modify anything under `docs/journal/`.** Results go in the results file named above.
3. **The reference arm is the promoted run**
   `training/support_classifier/outputs/halo_fixed_mr_residual_v3_8s_4res_40k_20260914/best_internal.pt`
   and its sealed evaluation
   `evaluations/sealed_comparison_residual_v3_8s_4res_40k_20260914/`. Its exact arguments are in
   that directory's `run_config.json` under `args`. Every training ablation changes **exactly one**
   argument from that config unless the table below says otherwise.
4. **Same seeds everywhere:** `--seed 20260901 --data-seed 20260901`, except the seed ablation E1.
5. **Naming.** Training runs: `outputs/halo_abl_<axis>_<variant>_40k_20260915`. Evaluations:
   `evaluations/sealed_abl_<axis>_<variant>_40k_20260915`. `<axis>` is the section letter below,
   `<variant>` is a short slug from the table. Example: `halo_abl_A_nopol_40k_20260915`.
6. **Evaluate HALO only.** Pass `--models halo --halo-checkpoint <run>/best_internal.pt` to
   `sealed_eval.py`. Baseline rows are unchanged and are already in the reference evaluation;
   `merge_sealed_results.py` can combine if a full table is wanted. This keeps each evaluation to
   roughly 18 minutes.
7. **Windows and k.** Use the defaults (4/8/16 s; k = 0,1,2,4,8,16,32,64,128). Report 8 s as the
   headline and the others in the appendix.
8. **Cost per training ablation:** ~40 minutes for 40k steps at ~17 steps/s, plus ~18 minutes
   evaluation. Twelve training arms is roughly 12 hours of GPU. Run in the priority order in §6.
9. **Interpreter and cwd:** `cd /home/alex/code/HALO/halo` then
   `/home/alex/code/HALO/legacy_code/.venv/bin/python -m training.support_classifier.train ...`.
10. Run the smoke path (`--smoke`) once for any arm that needs a new flag before launching it at
    40k.

## 2. Code additions required first (small, all opt-in, default behaviour unchanged)

| id | file | change | needed by |
|---|---|---|---|
| X1 | `training/support_classifier/train.py`, `encoding.py` | a `--recording-pool {learned,mean}` flag (default `learned`) that sets `learnable_recording_pool` instead of the current hard-coded `True` at `encoding.py:132/162` and `train.py:1442`. Record it in `run_config.json` and the resume trajectory (with a `setdefault` migration like `regime_split`). | B2 |
| X2 | `training/support_classifier/sealed_eval.py` | a `--halo-pool {checkpoint,mean}` flag (default `checkpoint`). With `mean`, set `encoder.recording_pool = None` after loading so `encoder.py:817` falls through to `hierarchical_device_pool`. Write the choice into every result record and the output directory name. | B1 |
| X3 | `training/support_classifier/train.py`, `model/support/residual_classifier.py` | expose `--lambda-buckets` (ints, default `0 1 2 4 8`) into `ResidualClassifierConfig.lambda_buckets`. Add to the resume trajectory with migration. | C2 |
| X4 | `training/support_classifier/sealed_eval.py` | per-label F1 in each result record (`per_label_f1: {label: f1}`), and a `label_seen_in_training: {label: bool}` map computed at the **concept** level using the same alias machinery as the k=0 bank (`labels.py` / the neutral-alias vocabulary), not by string match. Then two extra aggregate fields per k=0 record: `f1_macro_seen`, `f1_macro_unseen`. | D1 |
| X5 | `training/support_classifier/corpus.py` | `--exclude-placement-sites <site> ...` that drops every training stream whose `PLACEMENT_SITE` resolves to a listed site (use `data/scripts/curate/compatibility.py:PLACEMENT_SITE`). Record the list in `run_config.json`. Refuse to run if it empties any training dataset entirely; print the surviving stream list. | B5 |
| X6 | `training/support_classifier/sealed_eval.py` | two more HALO readout rows beside `halo-classifier` and `halo-classifier-residual-off`: `halo-classifier-text-only` (residuals zeroed, text term live) and `halo-classifier-residual-only` (text term zeroed, residuals live). Same episodes, same features; just the classifier's `forward` with the corresponding term masked. Also record, per k=0 record, the mean and sd of the text-term margin (`target_text − max other_text`), which the trainer already logs at `train.py:712`. | D3, D4 |
| X7 | `training/support_classifier/encoding.py`, `train.py` | `--duration-embedding {on,off}` (default `on`, matching the reference's `use_duration_embedding=True`). Off builds the encoder without the gated duration embedding. Resume trajectory migration as for X1. | D5 / B6 |

X1–X3 and X5 are trainer-side and touch the resume-trajectory validator; run
`tests/test_support_classifier_resume*.py` and `tests/test_residual_classifier.py` after each. X2
and X4 are evaluation-only and must leave every existing result record's fields untouched; verify by
re-running one reference cell and diffing `f1_macro`.

## 3. Ablations by contribution clause

Deltas are always **ablation minus reference**, dataset-balanced macro F1, 8 s, single-device
unless the cell says composite. The noise floor is set by E1 (§3.6); nothing under it is a result.

### 3.1 Clause: "text-conditioned sensor tokens" — is identity-as-text doing work?

| id | arm | one change from reference | cells that decide it | question decided |
|---|---|---|---|---|
| **B4** | `B_neutraltext` | `--neutral-acquisition-text` (removes device / placement / modality identity from the sensor text; axis text stays) | all single-device k=0..128; multi-device composites; the D2 placement split | Does conditioning on acquisition metadata matter at all? If B4 ≈ reference everywhere, the "conditioning with metadata" clause is decoration and must be cut. If it hurts most on the D2 *unseen-site* cells and the composites, the clause is a contribution. |
| **B2** | `B_meanpool` | `--recording-pool mean` (X1): train the full recipe with the parameter-free hierarchical mean instead of the learned attention pool | multi-device composites first; single-device as a check | How much of the +11.1 / +8.9 multi-device gain is the learned pool? This is the fair version of the question (the mean-pooled encoder is trained as such). |
| **B1** | `B_meanpool_evalonly` | evaluation only: reference checkpoint with `--halo-pool mean` (X2) | same as B2 | Lower bound on the pool's value (features were trained for the learned pool). Costs 18 minutes and no training; run it first as a sanity check, but B2 is the number that goes in the paper. |
| **B3** | `B_singledev` | `--multi-device-probability 0.0` | multi-device composites | Do random device-subset episodes in training matter, or does the pool fuse an arbitrary set at test time without ever having seen one? |
| **B5** | `B_holdout_waist` | `--exclude-placement-sites waist` (X5): drops hhar/phone and kuhar/phone waist streams from training | `inclusivehar:phone_waist`, `realworld:phone_waist`, and the RealWorld composites, at every k | The first genuinely **region-unseen** placement cell we will have (see §4). Compare to the same cells in the reference. The cost is what a truly new placement costs; the B4-vs-B5 interaction says whether the text is what pays it back. |

### 3.2 Clause: "answers from label text alone, from one example, or from many, through a single head"

| id | arm | one change from reference | cells | question decided |
|---|---|---|---|---|
| **C2** | `C_k32_lambda` | `--enrollment-k 1 2 4 8 16 32 --lambda-buckets 0 1 2 4 8 16 32` (X3). **Two** changes, deliberately: this is the *repair*, not an isolation. | all k; compare against the centred-neighbours floor **of the same checkpoint** | Does the k≥4 regression disappear when the head is trained where it is evaluated? Pass criterion: classifier ≥ floor − 1.0 at every k ≤ 32, and ≥ floor − 2.0 at 64/128, with k=0 and k=1 within noise of the reference. If it passes, this becomes the new reference for the paper. |
| **C2b** | `C_k32_only` | `--enrollment-k 1 2 4 8 16 32` only (buckets unchanged) | as C2 | Only if C2 passes: attributes the fix between k-range and λ table. Low priority. |
| **C1** | `C_gtalways` | `--p-gt-present 1.0` (the true label always has supports; masking unchanged) | k=0 above all; k=1 | Is the open-set episode construction what earned the k=0 result? If k=0 drops by more than noise, "partial information" is a contribution. If not, it is practice. |
| **C3** | `C_nomask` | `--p-mask-candidate 0 --p-mask-gt 0` | k=1..8 | Does hiding supports during training matter? Lower priority; drop if time is short. |

### 3.3 Clause: the frontend (not in the sentence, but the most novel mechanism)

| id | arm | one change from reference | cells | question decided |
|---|---|---|---|---|
| **A1** | `A_nopol` | `--no-polarization` | all; look hardest at wrist/hand cells (`ut_complex`, `shoaib:watch_wrist_proxy`) and at k=0 | What do the gravity-referenced polarization features buy? This is the isolation the polarization journal entry has owed since 2026-09-13. |
| **A2** | `A_1res` | `--resolutions 1.0` (single 1 s patch grid instead of 0.5/1/2/4) | all; k=0 and 16 s window especially | Is the multiresolution grid still the dominant frontend gain, as the 2026-07 learnable-tokenizer attribution found? Lower priority than A1. |

### 3.4 Clause: "a parameter-free floor" — nothing to run

Already measured: centred neighbours is bit-identical across arms with the same encoder and is
reported as its own row. The one addition is free: the executing agent should include the
`halo-centred-neighbours` row for every ablation arm, since encoder ablations (A1, A2, B2, B4, B5)
move the floor and the classifier together and the paper needs both.

### 3.5 Clause: "2.2M parameters" — nothing to run

The parameter table exists (`parameters_m` in every record). Optional, if cheap: wall-clock per
query on CPU for HALO vs the four baselines at 8 s, k=8, single-threaded. Not required.

### 3.6 Noise floor

| id | arm | change | question |
|---|---|---|---|
| **E1** | `E_seed2` | `--seed 20260902 --data-seed 20260902`, nothing else | The between-seed spread of the reference recipe at every k and cell. Every delta in this plan is read against this. Also settles the two open near-noise findings from 2026-09-14: the regime-split +0.4–0.7 at k≥4 and the 4-point co-training gap at k=0. **Run this first, in parallel with B1.** |

### 3.7 Diagnostics for the three open "why" questions

These are not one-change ablations; they are measurements that explain a result we already have.
They reuse arms above where possible.

**D3 — why did zero-shot improve, and what part of it is the frontend?**
The k=0 gain decomposes into three things that are currently confounded: the encoder (filterbank +
polarization + multiresolution), the classifier's text term, and the seen/unseen label mix. Measure:

| row | what it isolates | how |
|---|---|---|
| bridge on neighbours-95k encoder | k=0 with **no** learned classifier (the 38.3) | already in the 2026-09-13 superseded table; re-run under the current protocol with `--models halo --halo-checkpoint <neighbours95k>` |
| `halo-classifier-text-only` on the reference | k=0 from the text term alone (X6) | at k=0 there is no support, so this should equal `halo-classifier`; if it does not, something other than the text term is scoring at k=0 and that must be named |
| A1 (no polarization) and A2 (single resolution), k=0 only | how much of the k=0 number each frontend piece carries | from the A-arms' evaluations, no extra runs |
| D1 seen/unseen split of each row above | whether the gain is on labels the text projection was fitted to | X4 |

The reading: if A1 and A2 move k=0 by less than noise while the text-only row carries the whole
gain, the frontend did not improve zero-shot — the classifier did, and the sentence in the results
file ("+11.0 of the gain is the classifier rather than the encoder") is confirmed. If A1 moves k=0
on the wrist cells specifically, polarization is contributing discriminative structure that the text
projection can then align. Either answer is fine; we just need to stop guessing which.

**D4 — why does the learned classifier not beat 1-NN / the floor at k ≥ 4?**
The reference checkpoint's learned λ table is `[1.93, 1.13, 1.01, 0.92, 0.79]` for buckets
`k=0, 1, 2–3, 4–7, ≥8`. So the text term still carries 0.79× weight at k=128, where the centred vote
over 128 supports is far sharper than at k=8. The hypothesis is that the text term, not the
residuals, is the damage above k=4. Measure on the reference checkpoint, every k:

| row | expected if the λ hypothesis is right |
|---|---|
| `halo-classifier` | regresses from k=4 (known) |
| `halo-classifier-residual-off` (= centred floor) | the floor (known) |
| `halo-classifier-text-only` (X6) | tracks `halo-classifier` closely at k ≥ 8: the text term alone reproduces the regression |
| `halo-classifier-residual-only` (X6) | ≈ floor or slightly above at every k: the residuals are harmless |

If `residual-only` also regresses, the residuals are part of the problem and C2's λ fix will not be
enough; `r_candidate` gating goes back on the list. C2 is then the confirmation: with λ trained per
bucket to k=32, the text-only row should fall toward zero weight at high k on its own.

Second, cheap and diagnostic of the *mechanism*: the trainer's telemetry already logs
`classifier/mean_abs_r_support`, `mean_abs_r_candidate` and the text margin per step. Plot them
from the reference `log.jsonl` against step; if the text margin keeps growing while the residual
magnitudes plateau near zero, the head learned to solve the training task with text and never
needed the residuals — which is what a k ≤ 8 curriculum with `p_gt_present=0.5` rewards.

**D5 — is metadata conditioning (text, patch duration, sampling rate) actually helping?**
Three different inputs, three different answers required:

| input | how it enters | switch | question |
|---|---|---|---|
| acquisition text (device, modality, placement) | summed into each sensor token | **B4** | covered in §3.1 |
| patch duration | a learned duration embedding behind a sigmoid gate; the gate initialises at 0.10 and the reference checkpoint learned it to **0.20** (neighbours-95k: 0.25), so it is live, not inert | **B6** = `B_nodur`: `--duration-embedding off` (X7), one 40k run | with four patch resolutions in the same sequence, does the model need to be told which resolution a token is, or does the RoPE period already carry it? If B6 ≈ reference, drop the embedding and the claim. |
| sampling rate | **not a conditioning input.** The filterbank places its 32 bands in physical Hz from `sampling_rate_hz`, and `source_rate_hz` only bounds the analysable bandwidth (the Nyquist mask). Rate never reaches the transformer as an embedding. | none needed | the rate-invariance test from the 2026-09-11 sweep already shows bit-stable features under resampling; the paper should say "rate-agnostic by construction", not "rate-conditioned". The only ablation that would test *value* is resampling every source to one rate before the filterbank, which is a data-pipeline change and is out of scope here. |

If B4 ≈ reference **and** B6 ≈ reference, then the honest sentence is that the model is
*compatible* with heterogeneous acquisition through its tokenisation, but is not measurably
*conditioned* on it — and the "conditioning with metadata" clause must be rewritten as
"tokenised by".

## 4. Evaluation-side stratifications (no training, but they gate what we may claim)

**D1 — seen vs unseen labels at k=0.** After X4, re-evaluate the reference checkpoint and report,
per dataset and pooled, `f1_macro_seen` and `f1_macro_unseen` at k=0 and k=1. Three of six sealed
datasets have every label seen at the concept level; the unseen number is the open-set claim, and
right now we do not know it. Expected outcome is a large drop; the question is whether the unseen
number still beats HARNet's and UniMTS's k=0 rows, which must be stratified the same way.

**D2 — placement-site stratification of existing cells.** No new evaluation. Regroup the reference
results (and every B-arm) by whether the evaluation stream's `PLACEMENT_SITE` appears in training:

| group | eval streams | in training? |
|---|---|---|
| site seen | `shoaib:phone_left_pocket`, `shoaib:phone_right_pocket`, `shoaib:watch_wrist_proxy`, `ut_complex:watch_wrist`, `inclusivehar:phone_waist`, `realworld:phone_waist` | yes (exact site) |
| site unseen, region seen | `shoaib:phone_belt` (belt ~ waist), `realworld:phone_forearm` (forearm_unspecified ~ left/right forearm), `motionsense:phone_front_pocket` (~ left/right pocket), `usc_had:phone_hip` (~ waist), `realworld:phone_thigh` (thigh_unspecified ~ left/right thigh) | the text string is new; the body region is not |
| region unseen | none today; B5 creates one | — |

Also note that **every** watch evaluation stream carries the device profile `watch_proxy`, which
does not occur in training (`phone`, `watch`, `device` do). So the wrist cells are already a
device-text shift. Report the three groups as separate averages for the reference and for B4; the
gap between groups is the honest "unseen configuration" number we have without B5, and B4's effect
on the middle group is the direct test of identity-as-text.

## 5. What each outcome means for the paper

| result | keep clause | rewrite clause |
|---|---|---|
| B4 hurts unseen-site and composite cells by > noise | "conditioning with metadata and text" is a contribution | — |
| B4 ≈ reference | — | cut the metadata clause; identity text becomes an implementation detail |
| B2 loses most of the composite gain | learned placement-agnostic pool is the fusion contribution | — |
| B2 ≈ reference on composites | — | fusion is "more devices help"; the pool is not the story |
| C1 drops k=0 by > noise | "partial-information curriculum" is a contribution | — |
| C2 passes | "one head, all regimes" holds; C2 becomes the reference | — |
| C2 fails | — | scope the head to k ≤ 8 and report the floor above it |
| D1 unseen k=0 still leads baselines | open-set claim stands | — |
| D1 unseen k=0 falls below baselines | — | k=0 claim becomes "seen-label text retrieval"; the paper reframes around k=1 |
| A1 drops wrist/hand or k=0 by > noise | polarization is a frontend contribution | — |
| B5 costs more than B4's unseen-site penalty | — | "any placement" becomes "any placement in a seen region" |

## 6. Priority order and run schedule

Run in this order; stop and report after each block so the planner can re-prioritise.

1. **Block 1 (no training, ~1.5 h):** X2, X4, X6 → **B1**, **D1**, **D2**, **D3** (text-only k=0 row + bridge re-run), **D4** (four-row table on the reference). Then start **E1** training.
2. **Block 2 (~2 h each, sequential on one GPU):** **B4**, **C2**, **A1** — one per clause, highest
   information per run.
3. **Block 3:** **C1**, **B2** (needs X1), **B5** (needs X5), **B6** (needs X7).
4. **Block 4 (only if time):** **B3**, **A2**, **C3**, **C2b**.

Thirteen training arms if everything runs; the first eight decide every clause in the sentence.

## 7. Results file format

`docs/design/ABLATION_RESULTS_20260915.md`, one section per arm in the order run, each containing:

1. The exact command line, the output directory, the `best_internal.pt` step, and the evaluation
   directory.
2. One table, 8 s, single-device, dataset-balanced macro F1: rows `halo-classifier` and
   `halo-centred-neighbours`, columns k = 0, 1, 4, 8, 32, 128, each cell as `value (Δ vs reference)`.
3. One table of the multi-device composites at 8 s (`realworld` 3-device, `shoaib` 4-device), same
   rows, k = 0, 1, 8.
4. For B-arms: the D2 three-group table.
5. For D1: the seen/unseen table per dataset, HALO and the two baselines with a k=0 path.
5b. For D3/D4: the four-readout table (`classifier`, `residual-off`, `text-only`, `residual-only`) at every k on the reference, plus the k=0 bridge row from the neighbours-95k checkpoint, plus the telemetry plot from `log.jsonl` (residual magnitudes and text margin vs step) as a PNG under `docs/results/assets/`.
6. Anything that went wrong, verbatim (a crash, a resume, a stalled run, a moved directory).
7. **No interpretation.** Numbers and provenance only; the reading in §5 is the planner's job.

The 4 s and 16 s tables go in an appendix, same layout. Attach the per-cell `results.json` paths.

## 8. Things not in this plan, on purpose

- Cross-placement **training episodes** and label-group **holdout** (the curriculum gaps named in
  the design narrative). Those are design changes, not ablations of the current system; they come
  after we know which clauses survive.
- A second Phase-A / JEPA arm. Retired; see the 2026-09-13 ladder.
- Any change to baselines. They are frozen inputs to this comparison.
- Anything at 16 s / k=128 that is MotionSense-only. Report it, do not read it.
