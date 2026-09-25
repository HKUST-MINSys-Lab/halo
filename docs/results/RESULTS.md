# HALO results record

Promoted results for the support-conditioned HAR design, newest first.

**Where these sit in the paper (2026-09-22 onward):** every table below is **rung 2** of the
three-rung plan — k labelled examples, parameters frozen — and is presented as a case study written
as an *encoder* result, not as the headline. Rungs 1 (unlabelled adaptation) and 3 (fine-tuning for
every model) got their first results on 2026-09-26 (next section); the operational plan is
[the experiment runbook](../overview/experiments.md), the design [the roadmap](../overview/roadmap.md).

## Rung 1 (tier 1) and rung 3 (tier 3) — first results, 2026-09-26

Protocol for both: the 11 sealed single-device 8 s cells; per cell a fixed scored set (20 % of
executions) and an execution-disjoint pool. Rung 1 = EM-Dirichlet + embedding affinity (μ = 1,
10 neighbours), per-encoder temperatures fitted on held-out *training-source* windows, k = 0. Rung 3
supports come from the pool (same draw as rung 1 for draw 0). Dataset-balanced macro-F1.

**Rung 1** — [`results/artifacts/rung1_tier1_20260925`](../../results/artifacts/rung1_tier1_20260925/README.md):

| encoder | anchor (no unlabelled data) | N=0 (scored set only) | N=all | pool effect, μ = 0 (published) |
|---|---:|---:|---:|---:|
| HALO v4 | 47.4 | 50.6 | 51.9 | −2.0 |
| UniMTS | 31.9 | 35.1 | 34.7 | +1.8 |
| HARNet-10 | 32.0 | 31.6 | 31.4 | −1.0 |
| HARNet-5 | 31.2 | 30.5 | 31.1 | −0.4 |
| LiMU-BERT-X | 21.9 | 23.0 | 22.6 | −3.1 |
| NormWear | 10.4 | 16.3 | 18.4 | +2.0 |

Most of the effect is anchor → N=0 (transduction over the scored set); the pool adds ≤ 2 points,
and for HALO only with the affinity term. A pool with *none* of the scored classes still helps
(explaining away, not learning target classes) — read the artifact README before citing.

**Rung 3, frozen features** — [`results/artifacts/rung3_probe_20260925`](../../results/artifacts/rung3_probe_20260925/README.md)
(3 support draws; ± ≈ 5 points at k = 1 for every model): linear probe k = 1 / 4 / 16 — HALO
53.5 / 66.8 / 73.3, UniMTS 45.4 / 51.6 / 60.8, LiMU-BERT-X 34.7 / 41.9 / 50.6, HARNet-5
34.7 / 41.2 / 51.3.

**Rung 3, fine-tuning** — [`results/artifacts/rung3_phaseA_20260926`](../../results/artifacts/rung3_phaseA_20260926/README.md)
(draw 0 only so far): full fine-tune k = 1 / 4 / 16 — HALO 50.1 / 69.7 / 76.2, UniMTS 50.2 / 64.4 /
67.4, HARNet-5 42.6 / 54.9 / 68.1, LiMU-BERT-X 45.1 / 58.3 / 60.8. HALO *frozen* (53.1 / 66.0 / 71.8,
draw 0) is ≥ every baseline's full fine-tune at every k.

**The one rule: never mix protocols.** A number is only comparable to another number produced by
the same protocol on the same manifests. Every table below states its protocol, and superseded
tables are not reproduced here: their numbers live in their immutable artifacts, linked in the index.
No sealed result has ever selected a checkpoint or tuned a threshold.

## Which run is which

Names collide easily in this project, so use this index rather than inferring a run from a
directory name. "Arm" means a training run; "readout" means how a trained checkpoint is scored.

### Classifier naming (adopted 2026-09-20)

**`v4` is the promoted classifier as of 2026-09-21 — the one to beat. It is the T6 recipe on
`support_classifier_v4`. `v3` (`support_classifier_v3`) was promoted from 2026-09-18 to
2026-09-21. Every experimental replacement is `T`-numbered in the order it was trained (T for
"try"); a try that gets promoted keeps its T-number in history and gains a `v` name.** Use these names in discussion and in every new
document; the architecture strings below are what checkpoints and code carry, and they are not
always numbered consistently with the tries (`support_classifier_v4` is T4, not a successor to
`support_classifier_v3`). That mismatch is exactly why this table exists.

| name | architecture string in code and checkpoints | trained | verdict |
|---|---|---|---|
| **v4** | `support_classifier_v4`, T6 recipe (trainer default for `--classifier evidence_gated`) | 2026-09-20 | **promoted 2026-09-21**; beats v3 at every k ≥ 4, over-trust pathology closed |
| v3 | `support_classifier_v3` | 2026-09-18 | promoted 2026-09-18 → 2026-09-21, superseded by v4 |
| T1 | `support_contextual_mixture_v1` | 2026-09-18 | failed: the gate collapsed onto label meaning (0.996) |
| T2 | `support_contextual_residual_v1` | 2026-09-19 | negative |
| T3 | `support_evidence_aware_v2` | 2026-09-19 | negative: router at 0.97-0.99 semantic reliance |
| T4 | `support_classifier_v4` | 2026-09-20 | parity with v3; first try whose routing did not collapse |
| T5 | `support_classifier_v4` with corruption disabled | 2026-09-20 | corruption-free control; routing collapses onto label meaning |
| T6 | T4 with corruption as a gate-only auxiliary and a label-blind unenrolled calibration term ([design](../journal/2026-09-20-classifier-t6-design.md)) | 2026-09-20 | **promoted as v4 on 2026-09-21** |
| T7 | T6 with the primitive semantic branch ([design](../journal/2026-09-20-primitive-semantic-path-design.md)) | 2026-09-20 | best sealed arm; **disqualified by a 15-point foreign-vocabulary regression** |
| T8 | T7 with a written, training-label-only annotated label side and an exact probability-space mixture ([design](../journal/2026-09-21-classifier-t8-grounded-primitives.md)) | built, not trained | held-out label-side screen 0.82 vs T7's 0.34 |

T4 through T8 share the `support_classifier_v4` architecture string and differ by recipe (config
flags and curriculum), which the checkpoint records. The differentiable-neighbours arm is a parameter-free **control**, not a try, and keeps its name.

| run | protocol | encoder conditioning | status | numbers |
|---|---|---|---|---|
| **T7** T6 + primitive semantic branch, step 40k, 2026-09-20 | v5 | acquisition-conditioning-v2 | **best sealed; MM-Fit regression** | [below](#t7-t6-plus-the-primitive-semantic-branch) |
| **v4 (T6)** gate-only corruption + unenrolled calibration, step 40k, 2026-09-20 | v5 | acquisition-conditioning-v2 | **promoted 2026-09-21** | [below](#t6-corruption-as-a-gate-only-auxiliary-plus-unenrolled-calibration) |
| **T5** corruption-free control, step 40k, 2026-09-20 | v5 | acquisition-conditioning-v2 | **control: λ collapses without the curriculum** | [below](#the-corruption-free-control-the-curriculum-is-load-bearing) |
| **T4** evidence-gated blend, step 40k, 2026-09-20 | v5 | acquisition-conditioning-v2 | **completed, parity with v3** | [below](#t4-the-evidence-gated-blend-support_classifier_v4) |
| **T3** evidence-aware, step 40k, 2026-09-19 | v5 | acquisition-conditioning-v2 | **completed negative result** | [below](#t3-the-evidence-aware-classifier-support_evidence_aware_v2) |
| **T3** evidence-aware, step 32,500 companion, 2026-09-19 | v5 | acquisition-conditioning-v2 | **completed negative result** | [below](#t3-the-evidence-aware-classifier-support_evidence_aware_v2) |
| **T2** bounded contextual residual, step 35k, 2026-09-19 | v5 | acquisition-conditioning-v2 | **completed negative result** | [record](../journal/2026-09-19-bounded-contextual-residual-v1-results.md) |
| **v3** residual arm, step 40k, 2026-09-18 | v5 | acquisition-conditioning-v2 | superseded by v4 on 2026-09-21 | below |
| HALO neighbours arm, step 40k, 2026-09-18 | v5 | acquisition-conditioning-v2 | **current** | below |
| **T1** contextualise-first mixture, step 40k, 2026-09-18 | v5 | acquisition-conditioning-v2 | **failed, recorded** | below |
| Released baselines, 2026-09-18 | v5 | n/a | **current** | below |
| HALO residual v3 `curriculum1234`, 2026-09-16 | v4 scenarios | combined-text-v1 | superseded | [artifact](../../results/artifacts/scenarios_halo_classifier_v3_20260917/) |
| HALO residual v3 `curriculum12`, 2026-09-16 | v4 scenarios | combined-text-v1 | superseded | [summary](../../results/artifacts/historical-evaluations/scenarios_curriculum12_20260916/SUMMARY.md) |
| HALO residual v3 40k, 2026-09-14 | 4/8/16 s sealed | combined-text-v1 | superseded | [artifact](../../results/artifacts/historical-evaluations/sealed_comparison_residual_v3_8s_4res_40k_20260914/combined/RESULTS.md) |
| Differentiable-neighbours control, 2026-09-13 | 4/8/16 s sealed | combined-text-v1 | superseded | see the 2026-09-14 artifact |
| Any 6-second-protocol table, 2026-09-12 and earlier | 6 s sealed | pre-factored | retired | [evaluation rebuild](../journal/2026-09-14-evaluation-rebuild.md) |
| Future-JEPA variants | various | various | retired | [retired record](../journal/2026-09-13-retired-jepa-promoted-results.md) |

Three things changed between the superseded rows and the current ones, and any one of them is
enough to make the numbers incomparable: the evaluation protocol (v5 restored the NumPy manifest
draw, corrected k=0 filtering and mandates companion 1-NN rows), the encoder's conditioning schema
(v2 puts device and placement in the text and sensor metadata in a structured conditioner), and the
training sampler (independent enrollment and acquisition draws, plus the joint device-set
curriculum). Historical tables must not be extended with new checkpoints.

## Current record — v5 protocol, 2026-09-18

Sealed evaluation uses `sealed-manifest-v2-20260916`; scenarios use
`deployment-scenarios-v5-20260918`. HALO and the released baselines were scored in separate runs on
**verified-identical manifests**: 333 of 333 shared sealed cells and 737 of 737 shared scenario
cells carry the same manifest fingerprint, with none differing, so the two artifact sets join
exactly.

### The released-baseline half

Five providers (`HARNet-5`, `HARNet-10`, `LiMU-BERT-X`, `UniMTS`, `NormWear`) over the six sealed
datasets, 4/8/16-second windows and `k = 0 … 128`: 39 cells in 42 minutes, 3,354 scored rows and 213
explicit unsupported rows, no failures. The scenario run covers seven scenarios at 8 seconds and
`k = 0, 1, 4, 8, 32`: 11,211 rows over 1,132 task units, no failures. Unsupported model/input
combinations are recorded as such rather than repaired with project-specific adapter logic.
Artifacts: [`baselines_sealed_v5_20260918`](../../results/artifacts/baselines_sealed_v5_20260918/),
[`scenarios_baselines_v5_20260918`](../../results/artifacts/scenarios_baselines_v5_20260918/).

### The HALO half

Two arms trained from scratch on the current sampler, identical except for the classifier:

| arm | classifier objective | checkpoint SHA-256 | wall time |
|---|---|---|---|
| residual | learned residual support classifier, defaults on (`residual_enabled`, `text_term_enabled`, `centring=support_mean`) | `56b2cb29fdf093bee26292987fae1048b701ef8560643f2e9868d7a7acf5cc74` | 52 min |
| neighbours | parameter-free differentiable-neighbours control | `8d63871eb15557631acbf8caa1cbb4e405297ba3d7146bfd209cec13cb7fcdb3` | 54 min |

Recipe: Stage A curriculum at the trainer defaults, 40,000 optimizer steps, 4 episodes per step, enrollment
counts `{1,2,4,8,16,32}`, acquisition mix 0.5/0.25/0.25, enrollment mix 0.5/0.25/0.25, joint device-set
challenge probability 0.5, up to four devices, seed 20260901, bf16, `--encode-chunk-rows 256`. No rate
augmentation, no gyroscope dropout, no adaptive text gate. Conditioning schema
`acquisition-conditioning-v2`. Trained at `28ff48d` plus the chunking patch, evaluated at `4212cf5` on a
clean tree; the exact configurations are stored as `training_run_config.json` inside each artifact.

**Checkpoint policy.** Both arms report the final step-40,000 checkpoint. This was declared before any
sealed evaluation ran. The enrolled-validation argmax rule that selected earlier checkpoints landed on
step 15,000 for the residual arm, where the enrolled metric oscillated between 70.2 and 76.1 after step
12,500 with no trend while zero-support validation kept rising: the argmax checkpoint scores 76.1 enrolled
and 73.5 zero-support on internal validation, the final checkpoint 72.2 and 81.8. For the neighbours arm the
argmax and the final checkpoint are the same weights. A fixed budget states its stopping rule in advance and
does not reward a lucky validation draw. A companion evaluation of the residual argmax checkpoint was
started and not completed.

**The three HALO variants** requested for this comparison, plus one ablation:

1. residual arm, learned classifier (`halo-classifier`): the shipped default;
2. neighbours arm, cosine 1-NN on its encoder (`1nn`);
3. residual arm, cosine 1-NN on its encoder (`1nn`);
4. ablation: residual arm with the residual and text terms switched off (`halo-classifier-residual-off`).
   This is the same checkpoint and module as (1), keeping only the centred support-vote term, i.e.
   the closed-form floor the learned residual is added to. Earlier sections call it "centred neighbours".

### Sealed aggregate, dataset-balanced macro F1, single-device cells

8-second windows (full `k` grid in the artifacts):

| row | k=0 | k=1 | k=8 | k=32 | k=128 |
|---|---:|---:|---:|---:|---:|
| HALO residual / classifier | **51.7** | **62.4** | 71.5 | 73.4 | 74.4 |
| HALO residual / classifier base (ablation) | - | 60.1 | **73.3** | **77.0** | **78.0** |
| HALO residual / 1-NN | - | 60.7 | 72.0 | 74.8 | 77.1 |
| HALO neighbours / 1-NN | - | 60.3 | 70.6 | 74.0 | 76.0 |
| HALO, either arm / training-bank bridge | 38.6 | - | - | - | - |
| UniMTS / 1-NN (native text at k=0) | 33.1 | 53.9 | 66.9 | 71.5 | 73.9 |
| LiMU-BERT-X / 1-NN (bank bridge at k=0) | 24.9 | 48.2 | 63.9 | 69.0 | 71.2 |
| NormWear / 1-NN (native text at k=0) | 11.6 | 46.3 | 60.6 | 67.1 | 71.5 |
| HARNet-10 / 1-NN (bank bridge at k=0) | 34.4 | 38.8 | 52.1 | 59.6 | 64.5 |
| HARNet-5 / 1-NN (bank bridge at k=0) | 36.0 | 40.8 | 51.1 | 57.0 | 60.7 |

4-second windows:

| row | k=0 | k=1 | k=8 | k=32 | k=128 |
|---|---:|---:|---:|---:|---:|
| HALO residual / classifier | **50.3** | **59.3** | 69.0 | 70.6 | 71.6 |
| HALO residual / classifier base | - | 56.2 | **69.8** | **73.6** | **75.2** |
| HALO residual / 1-NN | - | 57.0 | 68.0 | 71.4 | 73.4 |
| HALO neighbours / 1-NN | - | 57.1 | 67.2 | 70.2 | 72.8 |
| UniMTS / 1-NN | 34.0 | 51.2 | 64.0 | 67.9 | 70.7 |
| LiMU-BERT-X / 1-NN | 24.4 | 45.8 | 60.8 | 65.9 | 68.4 |
| NormWear / 1-NN | 11.7 | 45.1 | 59.2 | 64.4 | 68.8 |
| HARNet-10 / 1-NN | 33.6 | 36.2 | 49.2 | 55.4 | 60.7 |
| HARNet-5 / 1-NN | 34.7 | 35.5 | 46.3 | 52.4 | 56.9 |

16-second windows (`k=128` is MotionSense only at this duration and is omitted):

| row | k=0 | k=1 | k=8 | k=32 | k=64 |
|---|---:|---:|---:|---:|---:|
| HALO residual / classifier | **52.1** | **64.6** | 73.5 | 75.4 | 76.1 |
| HALO residual / classifier base | - | 63.4 | **74.8** | **78.6** | **79.5** |
| HALO residual / 1-NN | - | 64.0 | 73.8 | 77.6 | 78.6 |
| HALO neighbours / 1-NN | - | 63.8 | 72.6 | 76.5 | 77.6 |
| UniMTS / 1-NN | 32.2 | 55.1 | 68.0 | 71.5 | 73.7 |
| LiMU-BERT-X / 1-NN | 24.7 | 49.7 | 64.8 | 71.4 | 72.9 |
| NormWear / 1-NN | 11.0 | 47.0 | 62.4 | 68.3 | 70.8 |
| HARNet-10 / 1-NN | 36.7 | 44.8 | 57.7 | 63.5 | 66.5 |
| HARNet-5 / 1-NN | 35.2 | 41.2 | 52.3 | 58.0 | 60.3 |

Reading these tables:

* **HALO leads every released baseline at every enrollment level and every window.** At 8 s the
  zero-enrollment margin is 15.7 points over the best baseline path (51.7 against HARNet-5's bank bridge
  at 36.0); at one example per class it is 8.5 over UniMTS. Of the 51.7, HALO's own training-bank bridge
  accounts for 38.6, so 13.1 points come from the learned classifier rather than the encoder.
* **The learned residual and text term are net negative above two supports, on their own base.**
  Rows (1) and (4) are one model read out two ways; the base wins by 1.8 at k=8 and 3.6 at k=128 at 8 s,
  and by similar margins at 4 and 16 s. The classifier owns k=0 and k=1; the base owns everything above.
  This reproduces the regression recorded on 2026-09-14. That record attributed it partly to a training
  range that stopped at k=8; this run trained on enrollment counts up to 32 and the regression is
  unchanged, so that explanation is falsified. The remaining suspect is the λ bucket schedule
  `(0,1,2,4,8)`, under which every count from 8 to 128 shares one semantic weight.
* **Training toward the support vote does not produce a better encoder.** Read out identically with
  1-NN, the residual-trained encoder beats the neighbours-trained one at every k (8 s: 60.7/72.0/74.8/77.1
  against 60.3/70.6/74.0/76.0). The centred, temperature-weighted vote of the base is worth about a point
  over plain 1-NN on the same features at k≥2.
* The k=0 bridge values are 38.55 (residual) and 38.64 (neighbours); the per-cell values differ in all
  33 cells and the feature fingerprints differ, so the match is a rounding coincidence, not shared features.

#### Per-dataset decomposition, 8-second windows, macro F1

| dataset | row | k=0 | k=1 | k=8 | k=32 | k=128 |
|---|---|---:|---:|---:|---:|---:|
| MotionSense | classifier | 71.4 | 75.9 | 84.5 | 86.7 | 87.6 |
| MotionSense | classifier base | - | 71.3 | 86.2 | 91.3 | 92.4 |
| MotionSense | residual 1-NN | - | 72.9 | 85.2 | 89.9 | 94.0 |
| MotionSense | neighbours 1-NN | - | 75.1 | 85.5 | 88.9 | 91.4 |
| RealWorld | classifier | 46.9 | 55.4 | 69.1 | 71.4 | 72.3 |
| RealWorld | classifier base | - | 55.8 | 71.1 | 75.6 | 77.3 |
| RealWorld | residual 1-NN | - | 56.2 | 68.9 | 71.0 | 71.3 |
| RealWorld | neighbours 1-NN | - | 55.7 | 68.5 | 71.2 | 72.7 |
| Shoaib | classifier | 72.3 | 84.8 | 90.9 | 91.9 | 92.1 |
| Shoaib | classifier base | - | 80.5 | 91.9 | 93.9 | 94.3 |
| Shoaib | residual 1-NN | - | 81.3 | 91.4 | 92.9 | 93.6 |
| Shoaib | neighbours 1-NN | - | 81.6 | 89.8 | 92.0 | 92.8 |
| InclusiveHAR | classifier | 43.7 | 40.1 | 42.6 | 42.7 | 43.4 |
| InclusiveHAR | classifier base | - | 34.6 | 42.2 | 43.1 | 42.3 |
| InclusiveHAR | residual 1-NN | - | 35.4 | 40.8 | 38.4 | 38.8 |
| InclusiveHAR | neighbours 1-NN | - | 31.9 | 38.0 | 37.6 | 37.7 |
| USC-HAD | classifier | 33.0 | 49.9 | 64.2 | 68.5 | 70.7 |
| USC-HAD | classifier base | - | 52.6 | 68.8 | 75.0 | 77.3 |
| USC-HAD | residual 1-NN | - | 52.1 | 67.6 | 76.4 | 82.1 |
| USC-HAD | neighbours 1-NN | - | 54.9 | 66.7 | 74.9 | 79.8 |
| UT-Complex | classifier | 43.1 | 68.3 | 77.7 | 79.3 | 80.4 |
| UT-Complex | classifier base | - | 66.0 | 79.9 | 83.2 | 84.4 |
| UT-Complex | residual 1-NN | - | 66.4 | 78.0 | 80.3 | 82.6 |
| UT-Complex | neighbours 1-NN | - | 62.7 | 74.8 | 79.6 | 81.7 |

InclusiveHAR stays at or below 44 for every HALO readout, as it does for every released baseline; it is
the one sealed source where enrollment barely helps anyone.

#### Multi-device composite cells, 8-second windows, macro F1 (k=1 / 8 / 32)

| dataset | HALO classifier | HALO classifier base | HALO residual 1-NN | HALO neighbours 1-NN | UniMTS native | NormWear native |
|---|---|---|---|---|---|---|
| RealWorld, all devices | 61.1 / 73.2 / 74.3 | 60.7 / 77.5 / 80.9 | 61.4 / 75.3 / 77.5 | 63.2 / 76.6 / 79.1 | 53.8 / 69.7 / 74.4 | 47.9 / 63.9 / 67.7 |
| Shoaib, all devices | 89.9 / 94.5 / 95.1 | 89.7 / 97.0 / 98.1 | 90.3 / 97.6 / 98.3 | 89.4 / 95.3 / 95.6 | 79.5 / 93.2 / 96.3 | 80.2 / 93.4 / 95.1 |

### Deployment scenarios, 8-second windows, macro F1

Mean over primary variant cells (matched controls excluded, whole-query split, cross-subject or the
published participant split).

> **Corrected 2026-09-19.** The first version of this table averaged the matched controls into the
> cross-dataset, missing-modality and rate-mismatch rows, because those scenarios name their controls
> `matched_control_for_<dataset>` / `matched_full_control` rather than `.../matched_control`. Those nine
> rows (all columns) are now recomputed from primary scenario cells only; every other row and every
> paired cost below were unaffected and reproduce exactly. The table is regenerated by
> [`results/tools/v5_report.py`](../../results/tools/v5_report.py), which selects cells by the evaluator's
> own control fields. The baseline column is the strongest released model under cosine 1-NN,
named per row; at k=0 it is the strongest baseline path of any kind.

| scenario | k | HALO classifier | HALO residual 1-NN | HALO neighbours 1-NN | best baseline |
|---|---:|---:|---:|---:|---|
| partial coverage | 1 | **56.7** | 29.3 | 29.4 | 37.8 HARNet fusion; 26.4 UniMTS 1-NN |
| partial coverage | 8 | **57.2** | 31.8 | 31.0 | 38.4 HARNet-10 fusion; 30.4 UniMTS 1-NN |
| partial coverage | 32 | **57.4** | 32.4 | 31.5 | 38.7 HARNet-10 fusion; 31.0 UniMTS 1-NN |
| cross placement | 0 | **42.2** | - | 30.2 (bridge) | 28.4 HARNet-10 fusion |
| cross placement | 1 | **48.0** | 43.9 | 44.0 | 38.8 UniMTS |
| cross placement | 8 | **50.5** | 48.5 | 48.2 | 43.8 UniMTS |
| cross placement | 32 | **50.9** | 48.8 | 48.5 | 44.8 UniMTS |
| cross dataset | 0 | **74.1** | - | 27.0 (bridge) | 26.3 LiMU-BERT-X fusion |
| cross dataset | 1 | **75.1** | 69.0 | 71.0 | 58.7 UniMTS |
| cross dataset | 8 | **81.0** | 78.0 | 76.3 | 65.4 UniMTS |
| cross dataset | 32 | **82.0** | 79.8 | 77.7 | 66.3 UniMTS |
| missing modality | 1 | **58.1** | 55.5 | 54.3 | 57.6 UniMTS |
| missing modality | 8 | 64.8 | 64.6 | 61.9 | **71.3** UniMTS |
| missing modality | 32 | 65.7 | 67.1 | 63.8 | **74.8** UniMTS |
| rate mismatch | 1 | **64.2** | 60.2 | 58.1 | 57.9 UniMTS |
| rate mismatch | 8 | **71.8** | 70.2 | 66.4 | 71.0 UniMTS |
| rate mismatch | 32 | 73.1 | 71.8 | 67.9 | **74.2** UniMTS |
| new domain (MM-Fit) | 0 | 6.6 | - | 5.6 (bridge) | **14.7** UniMTS native |
| new domain (MM-Fit) | 1 | 46.3 | 46.2 | 47.1 | **54.7** LiMU-BERT-X |
| new domain (MM-Fit) | 8 | 57.9 | 57.7 | 57.3 | 57.9 LiMU-BERT-X |
| new domain (MM-Fit) | 32 | **60.6** | 57.5 | 59.5 | 59.2 LiMU-BERT-X |
| device set | 0 | **47.6** | - | 31.5 (bridge, 3 cells) | 34.2 UniMTS fusion |
| device set | 1 | **58.8** | 58.3 | 57.7 | 49.1 UniMTS |
| device set | 8 | **65.7** | 65.6 | 64.6 | 56.3 UniMTS |
| device set | 32 | 66.7 | **66.9** | 66.3 | 57.6 UniMTS |

Cost of each mismatch, scenario minus its exact matched control on the same queries and support
counts, macro F1 at k=8 (subject-paired; more negative is worse):

| scenario | HALO classifier | HALO residual 1-NN | HALO neighbours 1-NN | UniMTS 1-NN | HARNet-10 1-NN |
|---|---:|---:|---:|---:|---:|
| cross placement (30 pairs) | -24.8 | -27.3 | -26.7 | -26.9 | -18.2 |
| cross dataset (8) | -4.6 | -7.4 | -8.3 | -13.4 | -7.0 |
| missing modality (21) | -10.5 | -12.2 | -13.1 | 0.0 | 0.0 |
| rate mismatch (20) | -4.4 | -7.3 | -9.2 | -1.2 | -5.3 |
| device set (55) | -13.2 | -14.6 | -14.7 | -17.9 | -10.2 |

Reading the scenario tables:

* **Partial coverage is the largest lead** and it comes from naming candidates that carry no enrolled
  example, which cosine 1-NN structurally cannot do (its truth-unenrolled split is 0.0 by construction).
* **Cross placement has flipped.** The 2026-09-17 checkpoint trailed UniMTS 1-NN by 3.2 at k=8 on the
  v4 manifests; this checkpoint leads by 6.7, and it is the only model with a zero-enrollment row above
  the fusion floor. The paired cost is still large for everyone; HARNet-10 pays least in absolute terms
  from a much lower matched control.
* **Missing modality remains the one loss above k=1**, and its paired-cost row shows why the comparison
  is asymmetric: HARNet and UniMTS are accelerometer-only, so removing the gyroscope changes nothing for
  them (cost exactly 0.0), while HALO genuinely loses a modality (-10.5). LiMU-BERT-X declares the
  condition unsupported.
* **Rate mismatch** costs HALO's classifier 4.4 points against UniMTS's 1.2; LiMU-BERT-X is unaffected by
  construction because its released contract resamples every input to a fixed clock.
* **New domain is where the open-vocabulary claim stops.** MM-Fit's ten gym exercises are not in the
  training vocabulary. At zero enrollment the classifier is at chance (6.6 macro F1, 9.6% accuracy over
  ten classes). One example per class lifts it to 62.6 and eight to 81.8 on the matched-placement cell,
  so the encoder represents these movements; the language interface does not. On the partial-coverage
  variant, queries whose truth is unenrolled score 1.7 at k=8 against 55.6 when it is enrolled. The
  scenario has two MM-Fit cells and no other source, so its aggregate is thin.
* **Foreign vocabulary and acquisition mismatch compound.** MM-Fit also appears inside cross placement
  and device set. Under device-set mismatch HALO still leads on MM-Fit (45.8 against UniMTS 41.0 at
  k=8), but far below its familiar-vocabulary cells (77.1). Under cross placement on MM-Fit every model
  sits between 12 and 25 at every k, HALO is flat at about 19.5 from k=1 to k=32, and HARNet-10 and
  UniMTS edge ahead at high k. Enrolling a gym exercise at the wrist and deploying at the ear or pocket
  appears close to physically ill-posed; the failure is universal but HALO does not lead it.
* Zero-support rows for the neighbours arm use the training-bank bridge and are much weaker (30.2,
  27.0, 31.5) than the residual classifier's semantic path (42.2, 74.1, 47.6); that arm never trains on
  zero-support episodes, so its zero-support validation is undefined by design.


### T1: the contextualise-first mixture (`support_contextual_mixture_v1`)

A third arm trained the contextualise-first semantic-voting head
(`support_contextual_mixture_v1`, the approved [2026-09-17 plan](../journal/2026-09-17-contextual-classifier-plan.md)):
one attention stack contextualises query, supports, support labels and candidates together, then a
support-vote path and a semantic path are mixed by a per-candidate learned gate. Same recipe as the
other two arms, final step-40,000 checkpoint, 1.43 M classifier parameters against the residual
head's 1.41 M.

**It failed, and the failure is instructive rather than a wasted run.** Sealed, 8 s,
dataset-balanced macro F1:

| readout | k=0 | k=1 | k=8 | k=32 | k=128 |
|---|---:|---:|---:|---:|---:|
| contextual classifier (headline) | 40.6 | 42.8 | 43.7 | 43.8 | 43.8 |
| its semantic branch alone | - | 42.6 | 43.3 | 43.4 | 43.5 |
| its support branch alone | - | 53.4 | 64.0 | 66.6 | 67.6 |
| cosine 1-NN on its encoder | - | 54.9 | 65.8 | 69.2 | 71.7 |
| residual arm classifier, for reference | 51.7 | 62.4 | 71.5 | 73.4 | 74.4 |

* **The headline row is flat**: 1.0 point from one support to 128. A support-conditioned model that
  does not respond to support is not doing its job.
* **The gate collapsed onto the semantic branch.** Measured on real sealed features with twelve
  supports enrolled, the mean semantic weight is **0.996** and every candidate exceeds 0.9. The
  mixture sits 0.001 from the semantic branch and 0.28 from the support branch: the support path is
  computed and then discarded, and the gate selects the weaker branch nearly everywhere.
* **Its own support branch would have been better**, and plain 1-NN on the same features better
  still, so the contextualised vote also underperforms the parameter-free readout it replaced.
* **The encoder came out weaker too**: 54.9 against the residual arm's 60.7 at k=1. Contextualising
  before comparison cost representation quality as well as decision quality.

Scenarios show the same flatness in every condition. The clearest case is the new-domain scenario,
where the residual arm climbs from 6.6 at zero enrollment to 60.6 at k=32 by using supports, while
this arm moves only from 8.0 to 9.3 because its gate ignores them. Its cosine 1-NN readout, on the
same checkpoint, reaches 45.4.

**Internal validation could not detect any of this.** This arm finished at 76.1 enrolled and 81.7
zero-support, better than the residual arm's 72.2 and 81.8 on both. Internal validation draws from
the same 155 training labels, where a semantic path can succeed by memorising the vocabulary; the
sealed datasets use labels the model never saw. Collapsing onto semantics is therefore a winning
strategy on validation and a losing one in deployment. **Treat any future head with a semantic
shortcut as unvalidated until it is scored on out-of-vocabulary labels**, and log gate saturation
during training, which would have exposed this by step 2,500 instead of after a full
train-and-evaluate cycle.

The 2026-09-17 plan deliberately discarded the residual head's closed-form floor and warned never to
claim the replacement could not underperform 1-NN. That warning was correct, and a future revision
of this design should restore an identity-initialised floor so the learned parts must earn their
place.

Artifacts: [`halo_contextual_sealed_v5_20260918`](../../results/artifacts/halo_contextual_sealed_v5_20260918/),
[`halo_contextual_scenarios_v5_20260918`](../../results/artifacts/halo_contextual_scenarios_v5_20260918/).

### T2: the bounded contextual residual (`support_contextual_residual_v1`)

The 2026-09-19 replacement restored a soft support-vote floor, bounded contextual corrections,
per-candidate label-meaning weighting, and modular path-improvement losses. In human-facing tables
it is named **proposed hybrid classifier**; its diagnostics are **learned support matcher**, **soft
support vote**, **nearest support**, and **label-meaning matcher**. It trained cleanly and no longer
collapses entirely onto label meaning, but it is not promoted: at 8 seconds the proposed hybrid
scores 50.0/59.7/64.0/67.6/70.3 macro F1 at `k=0/1/8/32/128`, while the current best HALO classifier
scores 51.7/62.4/71.5/73.4/74.4. The learned support matcher improves the soft vote at one shot
(61.2 versus 59.8) but falls behind as evidence grows. Full training telemetry, branch
decomposition, sealed rows, and scenario rows are in the
[2026-09-19 result record](../journal/2026-09-19-bounded-contextual-residual-v1-results.md).

### T3: the evidence-aware classifier (`support_evidence_aware_v2`)

**What it is.** `support_evidence_aware_v2` computes an auditable support vote and a label-meaning
(semantic) distribution, injects both as evidence into a contextualising attention stack over query,
support, support-label and candidate-label tokens, refines the support comparison with a learned
correction, and mixes the two branches per candidate through a learned router ("semantic reliance").
It adds counterfactual enrollment groups (the same query seen with complete, partial and zero
enrollment), branch-preservation and grouped best-path auxiliaries, and a 20% per-source
open-vocabulary label holdout used only for checkpoint selection.

**Run.** `halo_evidence_aware_v2_40k_20260919`: code `a9e8d91` (= lab `main` at launch),
`--classifier contextual --steps 40000`, otherwise the same Stage A recipe as the residual and
neighbours arms; 42 minutes. Evaluated at `11c9221`, which differs from the training commit only in
docstrings and a lifecycle registry. **Primary checkpoint: `last.pt`, step 40,000**, declared before
any sealed number existed (SHA-256 `d0961938…cba52`). Companion `best_internal.pt` (step 32,500,
v2's seen-plus-held-out selection rule, SHA-256 `be0dd82e…876ea`) was evaluated as well and
lands within about one point of the primary everywhere: 8 s classifier 34.2/34.5/35.4/35.7/36.0 at
`k=0/1/8/32/128`, unmodified support vote 55.1/66.8/70.5/72.2 at `k=1/8/32/128`, and 25-45 macro F1
in every scenario. Checkpoint choice does not change the conclusion.

**Result: negative, not promoted.** Sealed, 8 s, dataset-balanced macro F1 (333/333 manifests identical):

| readout | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **v2 classifier** | 34.1 | 34.3 | 34.5 | 34.7 | 34.9 | 35.1 | 35.2 | 35.3 | 35.3 |
| v2 label-meaning branch only | - | 34.1 | 34.1 | 34.1 | 34.1 | 34.1 | 34.1 | 34.1 | 34.1 |
| v2 unmodified support vote | - | 55.0 | 59.5 | 63.6 | 66.6 | 69.1 | 70.5 | 71.3 | 71.9 |
| v2 contextual support vote | - | 54.3 | 57.2 | 59.3 | 59.9 | 60.2 | 60.0 | 60.5 | 62.1 |
| v2 encoder, cosine 1-NN | - | 55.3 | 59.6 | 63.3 | 66.3 | 68.8 | 70.8 | 72.1 | 73.2 |
| residual arm classifier (current) | 51.7 | 62.4 | 66.3 | 68.7 | 71.5 | 72.7 | 73.4 | 74.1 | 74.4 |
| UniMTS cosine 1-NN | 33.1* | 53.9 | 59.4 | 64.1 | 66.9 | 69.5 | 71.5 | 72.9 | 73.9 |

\* UniMTS native zero-shot. The 4 s and 16 s grids show the same shape (flat classifier near 34-35);
see the artifact and the figure below.

Scenarios, 8 s, `k=8`, mean over primary scenario cells (737/737 manifests identical):

| scenario | v2 classifier | v2 label meaning | v2 unmodified support vote | v2 contextual support vote | residual arm classifier | best baseline 1-NN |
|---|---:|---:|---:|---:|---:|---|
| partial coverage | 34.3 | 34.1 | 31.6 | 30.0 | **57.2** | 30.4 UniMTS |
| cross placement | 25.5 | 25.0 | 43.4 | 41.7 | **50.5** | 43.8 UniMTS |
| cross dataset | 43.8 | 42.9 | 72.8 | 65.5 | **81.0** | 65.4 UniMTS |
| missing modality | 28.7 | 28.1 | 60.9 | 54.1 | 64.8 | **71.3** UniMTS |
| rate mismatch | 34.0 | 33.1 | 64.3 | 58.1 | **71.8** | 71.0 UniMTS |
| new domain (MM-Fit) | 2.4 | 2.5 | 53.9 | 52.0 | 57.9 | 57.9 LiMU-BERT-X |
| device set | 30.0 | 29.5 | 61.1 | 58.3 | **65.7** | 56.3 UniMTS |

The v2 classifier's paired mismatch costs are near zero (cross placement -0.3, device set -0.1).
That is not robustness: a model that ignores the support set cannot be hurt by a mismatched one.

**Diagnosis: two independent failures.**

1. **The router collapsed onto label meaning.** Validated semantic reliance is 0.90 by step 2,500 and
   0.97-0.99 from step 7,500 to the end, on both the seen-label and the label-held-out panels. The final
   classifier therefore equals its label-meaning branch (34.1 flat) and discards support evidence that,
   read by its own unmodified vote, reaches 66.6 at `k=8`. The learned contextual correction also
   degrades the support branch (59.9 against 66.6). The best-path auxiliary registered the problem but
   at weight 0.1 could not overrule the cross-entropy: its median regret rises from 0.3 before step
   20,000 to 5.7 after step 30,000. The held-out-label validation loss rises from 1.1 to 5.6 while its
   accuracy stays near 0.5, i.e. the model became confidently wrong on unseen labels and the label-held-out
   selection signal did not prevent it.
2. **The label holdout removed core sealed activities from training.** The deterministic 20% holdout
   selected `walking`, `sitting`, `walking_upstairs`, `stairs` and `cycling` among its 25 labels: 17 of the
   52 sealed classes. Per-label F1 at 8 s, `k=0`: on labels v2 trained on it scores 44.7 (residual arm
   47.4); on the held-out labels 21.7 (residual arm 63.9). At `k=8` it scores 45.6 on trained labels
   against the residual arm's 74.5, which isolates failure 1. The v2 encoder is also weaker than the
   residual arm's under the same 1-NN readout (66.3 against 72.0 at `k=8`); fewer training labels and
   gradients routed through the label-meaning path are both candidate causes and are not separated here.

**Caveats for anyone re-reading these rows.**

* The evaluator's `label_seen_in_training` flag and `f1_macro_seen/unseen` read the global label list and
  do not subtract a checkpoint's holdout, so those fields are wrong for this checkpoint. The held/kept
  split above was recomputed from `per_label_f1` and the checkpoint's recorded holdout list.
* The evaluator's "unmodified support vote" readout is the fixed-temperature (0.07) support floor, not the
  head's learned-temperature support-status branch.

Figures: [sealed k-curves](../../results/artifacts/promoted-figures/k_curve_evidence_aware_v2_20260919.png),
[scenarios at k=8](../../results/artifacts/promoted-figures/scenarios_k8_evidence_aware_v2_20260919.png),
[holdout decomposition](../../results/artifacts/promoted-figures/holdout_decomposition_evidence_aware_v2_20260919.png),
[training telemetry](../../results/artifacts/promoted-figures/telemetry_evidence_aware_v2_20260919.png)
(regenerate with `results/tools/plot_telemetry.py`). Artifacts:
[`halo_evidence_aware_v2_step40k_sealed_v5_20260919`](../../results/artifacts/halo_evidence_aware_v2_step40k_sealed_v5_20260919/),
[`halo_evidence_aware_v2_step40k_scenarios_v5_20260919`](../../results/artifacts/halo_evidence_aware_v2_step40k_scenarios_v5_20260919/),
[`halo_evidence_aware_v2_step40k_scenario_branches_v5_20260919`](../../results/artifacts/halo_evidence_aware_v2_step40k_scenario_branches_v5_20260919/),
[`halo_evidence_aware_v2_step32500_sealed_v5_20260919`](../../results/artifacts/halo_evidence_aware_v2_step32500_sealed_v5_20260919/),
[`halo_evidence_aware_v2_step32500_scenarios_v5_20260919`](../../results/artifacts/halo_evidence_aware_v2_step32500_scenarios_v5_20260919/).

### T7: T6 plus the primitive semantic branch

**What changed from T6.** One flag: `--semantic-mode text+primitives`. The semantic branch becomes
a fixed equal-weight sum in log space of the promoted cosine path and the primitive path (32
primitives in 10 axes, label side a fixed function of the label string, shared identity-initialised
compatibility projection). Nothing else differs. Design:
[primitive semantic path](../journal/2026-09-20-primitive-semantic-path-design.md).

**Run.** `halo_t7_40k_20260920`, code `3876dd0`, 46 minutes training.

**Verdict: the best sealed arm we have produced, and not promotable, because of one scenario.**

Sealed, 8 s, dataset-balanced macro F1 (333/333 manifests identical):

| readout | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **T7 classifier** | **51.9** | **62.7** | **66.6** | **70.6** | **73.1** | **75.0** | **76.2** | **77.9** |
| T6 classifier | 50.2 | 62.4 | 66.5 | 70.2 | 72.7 | 74.6 | 75.8 | 76.5 |
| v3 classifier (promoted) | 51.7 | 62.4 | 66.3 | 68.7 | 71.5 | 72.7 | 73.4 | 74.4 |
| T7 own support vote | - | 60.8 | 65.4 | 69.7 | 72.7 | 74.7 | 76.3 | 78.3 |
| T7 encoder, cosine 1-NN | - | 61.0 | 65.3 | 69.0 | 71.8 | 73.0 | 74.4 | 75.9 |

T7 leads v3 at **every** k, by +3.5 at k=128, and its deficit against its own support vote is
-0.4 at k=128 against v3's -3.6 and T6's -0.9.

**The two semantic halves are complementary on seen vocabulary.** At 8 s, `k=0` (where the blend is
the semantic branch): text half alone **50.6**, primitive half alone **42.9**, combined **51.9**.
The combination beats both halves, which is the result the primitive design predicted and the
opposite of the dilution that makes released-baseline fusion rows worthless. It also closes T6's
only sealed weakness: k=0 moves 50.2 → 51.9, from 1.5 behind v3 to marginally ahead.

**Scenarios, 8 s** (737/737 manifests identical). T7 leads v3 almost everywhere, including the
cross-placement cell that no arm in this lineage had previously won:

| scenario | k | T7 | T6 | v3 |
|---|---:|---:|---:|---:|
| partial coverage | 8 | **58.2** | 57.2 | 57.2 |
| cross placement | 8 | **51.1** | 48.9 | 50.5 |
| cross dataset | 32 | **83.6** | 82.6 | 82.0 |
| rate mismatch | 32 | **74.7** | 73.1 | 73.1 |
| device set | 32 | **69.1** | 68.3 | 66.7 |
| **new domain (MM-Fit)** | **8** | **45.8** | **61.0** | **57.9** |

Partial coverage is the best split any arm has produced: 62.0 truth-enrolled and 60.2
truth-unenrolled at k=8 (harmonic mean 61.1, against v3's 59.7 and T6's 59.9), finally achieving
the balance the calibration term was added for.

**The disqualifying result, and its mechanism.** On MM-Fit, the only foreign-vocabulary source, T7
scores 45.8 at k=8 where T6 reaches 61.0. A per-branch diagnostic on that scenario
(`halo_t7_mmfit_diag`) shows why:

| MM-Fit readout | k=1 | k=8 |
|---|---:|---:|
| support vote | 42.7 | 47.5 |
| full classifier | 23.3 | 26.8 |
| semantic, text half | 10.2 | 10.2 |
| semantic, primitive half | **3.7** | **3.7** |

The support evidence is intact. Both semantic halves are at or below the 10.0 chance level, and the
primitive half is **confidently wrong rather than uninformative**. Because both semantic paths are
log-probabilities with unbounded negative range, a confidently wrong half can flip an argmax even at
a small blend weight, so the classifier lands 20 points below its own support vote. λ cannot rescue
this: it is label-blind by design, so it can learn "distrust meaning when evidence is strong" but
never "distrust meaning because this vocabulary is foreign".

This is dilution of a strong path by a weak one, in exactly the condition the primitive vocabulary
was built to serve. Two candidate repairs, neither tested: bound the semantic branch's confidence
(clamp its log-probabilities, or raise the primitive profile temperature so it cannot be peaked),
or give λ a label-blind proxy for vocabulary familiarity, which is a new and unproven input class.
The scrambled-vocabulary control remains unrun and would say how much of the seen-vocabulary gain is
grounding rather than capacity.

**T6 was promoted as v4 on 2026-09-21; T7 was not.** T6 beats v3 at every k ≥ 4 with
no scenario regression worse than 1.6, while T7 buys +1.4 sealed macro F1 at k=128 and a 15-point
loss on foreign vocabulary — the capability the project exists to demonstrate.

Artifacts:
[sealed](../../results/artifacts/halo_t7_step40k_sealed_v5_20260920/),
[scenarios](../../results/artifacts/halo_t7_step40k_scenarios_v5_20260920/).

### T6: corruption as a gate-only auxiliary, plus unenrolled calibration

**What changed from T4.** Same `support_classifier_v4` architecture, two recipe changes.
*Corruption became an auxiliary instead of a replacement*: every eligible episode now contributes
its clean loss, which trains everything, **plus** a deranged-roster view computed with
`gate_only=True` whose gradient reaches exactly the blend gate. No clean episode is displaced, and
the enrolled candidates' texts are deranged among themselves so every deliberately wrong semantic
score still has a learnable gate. *A label-blind calibration term* was added for candidates with no
support of their own: `text_logit + b(coverage, log |roster|)`, identical for every unenrolled
candidate of an episode. Design and predictions:
[T6 design](../journal/2026-09-20-classifier-t6-design.md).

**Run.** `halo_t6_40k_20260920`, code `3876dd0`, 45 minutes, `last.pt` at step 40,000 declared
before any sealed number existed.

**Verdict: T6 matches or beats v3 nearly everywhere and the over-trust pathology is essentially
closed. It is the first try that improves on the promoted control.**

Sealed, 8 s, dataset-balanced macro F1 (333/333 manifests identical):

| readout | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **T6 classifier** | 50.2 | 62.4 | 66.5 | **70.2** | **72.7** | **74.6** | **75.8** | **76.5** |
| T4 classifier | 50.2 | 62.1 | 65.5 | 69.0 | 71.0 | 72.9 | 73.9 | 74.3 |
| v3 classifier (promoted) | **51.7** | 62.4 | 66.3 | 68.7 | 71.5 | 72.7 | 73.4 | 74.4 |
| T6 own support vote | - | 60.5 | 65.8 | 69.7 | 72.4 | 74.6 | 76.1 | 77.4 |
| T6 encoder, cosine 1-NN | - | 60.8 | 65.5 | 69.3 | 71.7 | 73.2 | 74.8 | 76.3 |
| v3 encoder, cosine 1-NN | - | 60.7 | 65.9 | 69.6 | 72.0 | 73.6 | 74.8 | 77.1 |

T6 leads v3 by +1.5 to +2.4 at every k ≥ 4, matches it at k=1-2, and trails by 1.5 at k=0.

Each classifier against **its own** support vote — the pathology this lineage exists to fix:

| arm | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **T6** | +1.9 | +0.7 | +0.5 | +0.3 | +0.1 | -0.3 | -0.4 | **-0.9** |
| T4 | +2.3 | +1.0 | +0.5 | -0.3 | -0.5 | -1.1 | -1.8 | -1.9 |
| v3 | +2.3 | +0.2 | -1.5 | -1.8 | -2.1 | -3.6 | -3.5 | -3.6 |

The learned head now adds value through k=16 and costs under a point at the extreme, against v3's
crossover at k=2 and -3.6 at k=128.

**Three of the four predictions registered before the run held.** The encoder recovered as
predicted (1-NN 71.7 at 8 s k=8, against T4's 70.4, T5's (corruption-free) 71.1 and v3's
72.0), confirming that replace-mode corruption was the cause of T4's encoder regression. The k=128
deficit came in at -0.9 against a predicted "no worse than -1.9". λ stayed ordered by evidence
(0.41 / 0.33 / 0.22 at k=1 / 2-7 / ≥8) with no candidate at the 0.85 bound, and the collapse
detector rose from 0.25 to 0.62 and held. **The partial-coverage prediction was wrong** — see below.

**The collapse detector, and why internal validation is worthless here.** T6's internal enrolled
score is 0.727, *below* T4's 0.778 and well below T5's collapsed 0.787.
Its corrupted-view accuracy, measured on deliberately deranged rosters, climbed 0.25 → 0.62. The
panel shares the training vocabulary and rewards trusting label meaning, so it rates the collapsed
model highest; the corrupted-view probe separates them cleanly and costs nothing.

**Scenarios, 8 s (737/737 manifests identical).** T6 beats v3 on:

| scenario | k | T6 | v3 |
|---|---:|---:|---:|
| new domain (MM-Fit) | 0 | **10.5** | 6.6 |
| new domain (MM-Fit) | 8 | **61.0** | 57.9 |
| new domain (MM-Fit) | 32 | **63.0** | 60.6 |
| device set | 0 | **49.9** | 47.6 |
| device set | 32 | **68.3** | 66.7 |
| missing modality | 8 | **66.5** | 64.8 |
| missing modality | 32 | **68.5** | 65.7 |
| cross dataset | 32 | **82.6** | 82.0 |

MM-Fit at k=0 exceeds its 10.0 chance level for the first time. T6 trails v3 on **cross placement
at k ≥ 1** (48.9 against 50.5 at k=8) and marginally on rate mismatch at k=8.

**Partial coverage: the calibration term over-corrected.** Accuracy at k=8:

| arm | all | truth enrolled | truth unenrolled | harmonic mean |
|---|---:|---:|---:|---:|
| T6 | 60.1 | 62.2 | **57.8** | **59.9** |
| T4 | **62.3** | **80.8** | 44.5 | 57.4 |
| v3 | 60.1 | 66.7 | 53.9 | 59.7 |

The prediction was "truth-enrolled near 80 *and* truth-unenrolled above 50". The unenrolled half
came in best-of-three at 57.8, but the enrolled half fell to 62.2, below even v3's 66.7. The term
works — it moves the operating point, and its learned bias settled at +1.11 — but cross-entropy
plus this single scalar lands on a balanced point rather than the asymmetric one we wanted. The
harmonic mean is the best of the three arms by 0.2, which is not a meaningful margin.

**Historical decision at completion.** T6 was initially held pending T7 because it lost at k=0 and
on cross placement. T7 subsequently improved the sealed aggregate but failed the foreign-vocabulary
scenario by 15 points, so T6 was promoted as v4 on 2026-09-21. Its trust saturation remains a
documented limitation (mean |t| 1.85, p95 2.00 against a limit of 2.0).

Artifacts:
[sealed](../../results/artifacts/halo_t6_step40k_sealed_v5_20260920/),
[scenarios](../../results/artifacts/halo_t6_step40k_scenarios_v5_20260920/).
Figures:
[k-curves](../../results/artifacts/promoted-figures/k_curve_t6_20260920.png),
[telemetry](../../results/artifacts/promoted-figures/telemetry_t6_20260920.png).

### T4: the evidence-gated blend (`support_classifier_v4`)

**What it is.** `support_classifier_v4` replaces v3's three text-driven output terms with two
bounded, label-blind gates on top of the same closed-form support vote: a per-support trust scalar
`|t| ≤ 2` that reweights the vote, and a per-candidate blend weight `λ ≤ 0.85` that mixes the vote
with the label-meaning path. Neither gate can see label text, text confidence or slot identity, and
`λ_max < 1` means no setting can delete the support path. A label-text corruption curriculum (25% of
truth-enrolled episodes get a deranged roster, with the semantic branch stop-gradiented) supplies
the signal that prices the reliability of meaning. Design and implementation:
[design](../journal/2026-09-20-classifier-v4-evidence-gated-design.md),
[implementation](../journal/2026-09-20-classifier-v4-implementation.md).

**Run.** `halo_evidence_gated_v4_40k_20260920`, code `ba5c3c5` (= lab `main` at launch), 43 minutes.
Primary checkpoint `last.pt` at step 40,000, declared before any sealed number existed. The head is
50,615 parameters against v3's 1,728,404.

**Verdict: the diagnosed failure is roughly halved; the arm lands at parity with v3 overall.**

Each classifier against **its own** support vote, dataset-balanced macro F1 (positive = the learned
head adds value over its own floor):

| window | arm | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 s | **v4** | +2.3 | +1.0 | +0.5 | -0.3 | -0.5 | -1.1 | -1.8 | **-1.9** |
| 8 s | v3 | +2.3 | +0.2 | -1.5 | -1.8 | -2.1 | -3.6 | -3.5 | **-3.6** |
| 4 s | **v4** | +2.7 | +1.6 | +0.7 | +0.3 | -0.2 | -1.2 | -1.5 | -1.9 |
| 4 s | v3 | +3.1 | +1.4 | -0.2 | -0.8 | -2.1 | -3.0 | -3.2 | -3.5 |
| 16 s | **v4** | +1.2 | +0.5 | +0.1 | +0.1 | -0.7 | -1.0 | -1.1 | -3.2 |
| 16 s | v3 | +1.2 | -0.8 | -1.2 | -1.4 | -2.4 | -3.2 | -3.4 | -4.5 |

The crossover where the head stops helping moves from k=2 to k=8, and the deficit at k=128 halves.
The over-trust is reduced, not eliminated.

Sealed aggregate, 8 s (333/333 manifests identical):

| readout | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v4 classifier | 50.2 | 62.1 | 65.5 | 69.0 | 71.0 | 72.9 | 73.9 | 74.3 | 74.3 |
| v4 untrusted support vote | - | 59.7 | 64.5 | 68.5 | 71.2 | 73.4 | 75.0 | 76.2 | 76.2 |
| v4 trust-weighted support vote | - | 59.3 | 65.0 | 68.6 | 71.2 | 73.3 | 74.7 | 75.9 | 76.0 |
| v4 label-meaning branch | - | 50.2 | 50.2 | 50.2 | 50.2 | 50.2 | 50.2 | 50.2 | 50.2 |
| v4 encoder, cosine 1-NN | - | 59.8 | 64.4 | 68.0 | 70.4 | 72.0 | 74.1 | 75.6 | 76.2 |
| v3 classifier (promoted) | 51.7 | 62.4 | 66.3 | 68.7 | 71.5 | 72.7 | 73.4 | 74.1 | 74.4 |
| v3 encoder, cosine 1-NN | - | 60.7 | 65.9 | 69.6 | 72.0 | 73.6 | 74.8 | 76.5 | 77.1 |

**Why parity rather than a win: the encoder regressed.** v4's encoder scores 70.4 under cosine 1-NN
at k=8 against v3's 72.0, and its zero-shot semantic path 50.2 against 51.7, consistently across all
three window lengths. The head is better at using evidence; the representation it learned is
slightly weaker, and the two cancel. Whether the corruption curriculum causes that (it withholds
semantic gradient on a quarter of enrolled episodes) or it is seed variation is not established
here; a corruption-off arm would separate them.

**The gates behaved as designed, which is what three previous generations failed to do.**
λ orders itself by evidence — 0.54 at k=1, 0.43 at k=2-7, 0.34 at k≥8 — with **no candidate at the
0.85 bound at any point in training**, and trust stays inside its ±2 bound (mean |t| 0.63, p95 1.69
at the end). Contrast v2, whose router reached 0.97-0.99 semantic reliance by step 7,500 and stayed.

**Scenarios, 8 s, k=8** (737/737 manifests identical), v4 against v3:

| scenario | v4 | v3 | Δ |
|---|---:|---:|---:|
| partial coverage | 58.1 | 57.2 | +0.9 |
| new domain (MM-Fit) | 59.0 | 57.9 | +1.1 |
| device set | 66.2 | 65.7 | +0.4 |
| missing modality | 65.0 | 64.8 | +0.2 |
| rate mismatch | 70.8 | 71.8 | -1.0 |
| cross placement | 48.2 | 50.5 | -2.3 |
| cross dataset | 78.2 | 81.0 | -2.8 |

The two losses are the scenarios that most depend on representation quality, consistent with the
weaker encoder rather than with the head.

**Partial coverage: the mechanism worked and chose a different operating point.** Accuracy at k=8:

| arm | all | truth enrolled | truth unenrolled | harmonic mean |
|---|---:|---:|---:|---:|
| v4 classifier | **62.3** | **80.8** | 44.5 | 57.4 |
| v3 classifier | 60.1 | 66.7 | 53.9 | **59.7** |
| either arm, 1-NN | 43.2 | 87.6 | 0.0 | 0.0 |

The per-candidate λ was introduced precisely to stop one global trade-off from being forced on
every candidate, and it did move: truth-enrolled accuracy rises 14.1 points, most of the way to the
1-NN ceiling of 87.6, while truth-unenrolled falls 9.4. Overall accuracy improves, the harmonic
mean does not. The design's stated target was to raise the enrolled half *while keeping* the
unenrolled half near 54; half of that was achieved.

#### The corruption-free control: the curriculum is load-bearing

`halo_t4_nocorrupt_40k_20260920` is **T5**, trained identically to T4 — same code, seed, steps and
gate bounds — with one flag changed, `--text-corruption-probability 0`. The historical directory
name predates the recipe naming convention. It was run to attribute T4's
encoder regression, and it answered a larger question instead.

| 8 s, k=8 | λ at k=1 / k=2-7 / k≥8 | classifier | its own support vote | deficit | encoder 1-NN |
|---|---|---:|---:|---:|---:|
| T4 (corruption 0.25) | 0.44 / 0.40 / 0.22 | 71.0 | 71.2 | **-0.3** | 70.4 |
| T5 corruption-free | **0.81 / 0.78 / 0.82** | 63.7 | 72.4 | **-8.7** | 71.1 |
| v3 | 1.61 / 0.82 / 0.51 (per bucket) | 71.5 | 73.3 | -1.8 | 72.0 |

**Without the curriculum λ stops depending on evidence at all.** It sits at 0.81-0.82 for every
support count — flat, with 1.3% of candidates pinned at the 0.85 bound — instead of falling from
0.44 to 0.22 as evidence accumulates. The classifier then lands 6 to 15 points *below its own
support vote* across the k grid (-6.4 at k=1, -15.0 at k=128), which is the same collapse T1, T2
and T3 produced, contained only by `λ_max`. Every scenario is worse than both T4 and v3, most
sharply the new-domain cell (42.5 against T4's 59.0) and partial coverage (49.6 against 58.1).

**Internal validation is meanwhile higher than the corrupted arm's** — enrolled dataset-macro F1
0.787 against 0.778, zero-support 0.841 against 0.836. That is the whole diagnosis in one line: the
internal panel shares the training vocabulary, where trusting label meaning *is* the correct
policy, so it rates the collapsed arm more highly. No selection rule computed on that panel could
have caught this.

So the corruption curriculum is not a regulariser that happened to help; it is the only thing in
the recipe that prices the reliability of meaning, and the bounded gate alone does not substitute
for it. It does cost a little encoder quality: 1-NN recovers from 70.4 to 71.1 without it, against
v3's 72.0, so roughly half of T4's encoder gap is the curriculum and the other half is unexplained.

Artifacts:
[sealed](../../results/artifacts/halo_t4_nocorrupt_step40k_sealed_v5_20260920/),
[scenarios](../../results/artifacts/halo_t4_nocorrupt_step40k_scenarios_v5_20260920/).

Artifacts:
[sealed](../../results/artifacts/halo_evidence_gated_v4_step40k_sealed_v5_20260920/),
[scenarios](../../results/artifacts/halo_evidence_gated_v4_step40k_scenarios_v5_20260920/),
[branch diagnostics](../../results/artifacts/halo_evidence_gated_v4_step40k_scenario_branches_v5_20260920/).
Figures:
[k-curves against each arm's own vote](../../results/artifacts/promoted-figures/k_curve_evidence_gated_v4_20260920.png),
[training telemetry](../../results/artifacts/promoted-figures/telemetry_evidence_gated_v4_20260920.png).

### Limitations of the current record

* **Conditioning changed.** These checkpoints carry `acquisition-conditioning-v2`; every superseded
  HALO row used the combined-text scheme. These are fresh measurements, not re-scorings.
* **Row-chunked collation.** The dense collate pads each sub-batch to its widest row, so batch
  composition influences encoder outputs. Chunking at a fixed size is deterministic and reproducible
  but not bit-identical to an unchunked run. The sensitivity predates chunking and is unresolved.
* **Checkpoint selection.** The enrolled-validation argmax is inside noise after about step 12,500,
  which is why the fixed step-40,000 budget replaces it here. A companion evaluation of the argmax
  checkpoint was started and not completed.
* **Thin cells.** `k=128` at 16 s is MotionSense only. The new-domain scenario is two MM-Fit cells.
  Same-subject cross-placement cells are excluded from the scenario means as too thin to cite.
* These are deployed-system comparisons, not parameter-matched or corpus-matched ablations. The
  released models differ substantially in size and pretraining corpus. The matched-corpus arms that
  would control for this are planned in
  [the encoder-isolation plan](../journal/2026-09-18-encoder-isolation-plan.md) and not yet run.

Artifacts, each with a README, the generated table, the compressed row payload and manifests, the
training configuration and uncompressed hashes:
[`halo_residual_sealed_v5_20260918`](../../results/artifacts/halo_residual_sealed_v5_20260918/),
[`halo_residual_scenarios_v5_20260918`](../../results/artifacts/halo_residual_scenarios_v5_20260918/),
[`halo_neighbors_sealed_v5_20260918`](../../results/artifacts/halo_neighbors_sealed_v5_20260918/),
[`halo_neighbors_scenarios_v5_20260918`](../../results/artifacts/halo_neighbors_scenarios_v5_20260918/).

## Design facts that outlive their tables

**How HALO fuses devices.** One token per `(patch, sensor)`, where a sensor is one 3-axis modality
on one device, so an accelerometer and a gyroscope on the same wrist stay separate tokens. Device,
modality and placement identity enter through per-sensor text conditioning, and a single learned
`RecordingAttentionPool` query attends over every `patch × sensor` token at once. There is no
separate device-fusion stage and no placement-specific parameter: an unseen placement is
expressible because identity is text. An earlier record credited this gain to a parameter-free
hierarchical device mean; that path exists in `encoder.py` but is overridden by the learned pool in
every trained checkpoint, so it produced none of the numbers in this file. See
[the pooling correction](../journal/2026-09-14-multi-device-pooling-correction.md).

**Why the scenario set is fixed at seven.** Deployment scenarios are a chosen set of stress tests,
not a Cartesian sweep. Multiplying the sealed `k` grid and three window lengths into the scenario
experiment is an exploratory sweep, must use a separate output directory, and must not replace the
representative tables.

**Narrative.** Why Future-JEPA and the continuous kernel were dropped, what is and is not a
controlled comparison, and the open limitations:
[the design narrative](../journal/2026-09-14-design-narrative.md).

## Superseded and retired records

Their numbers are not reproduced here. Each artifact retains every split, readout and confidence
interval, and the index at the top of this file links them.

* **2026-09-17, scenarios v4.** First seven-scenario run with HALO against all five baselines. Its
  headline had HALO leading every scenario, but external models were scored with fusion only; on
  companion 1-NN the ordering reverses for cross placement and missing modality. Superseded by the
  v5 rows above, which mandate both readouts.
  [Results note](../journal/2026-09-17-scenario-evaluation-results.md).
* **2026-09-16, Stage A curriculum diagnostics.** Eight scenarios including the since-retired
  cold-start condition.
  [Summary](../../results/artifacts/historical-evaluations/scenarios_curriculum12_20260916/SUMMARY.md).
* **2026-09-14, sealed 4/8/16 s.** The previously promoted HALO table, with per-dataset and
  multi-device decompositions and the k-curve figure. Its two headline findings, that the learned
  classifier owns `k=0` and `k=1` while its own base wins above `k=2`, both reproduce in the current
  record; the explanation it offered for the high-k regression, a training range stopping at `k=8`,
  is falsified there.
  [Artifact](../../results/artifacts/historical-evaluations/sealed_comparison_residual_v3_8s_4res_40k_20260914/combined/RESULTS.md).
* **2026-09-13, differentiable-neighbours control.** The same encoder configuration with no learned
  classifier, retained inside the 2026-09-14 artifact.
* **2026-09-12 and earlier, 6-second protocol.** Not comparable to anything above; the protocol was
  rebuilt on 2026-09-13. [Evaluation rebuild](../journal/2026-09-14-evaluation-rebuild.md).
* **Future-JEPA variants.** Rows, telemetry and artifacts preserved as historical evidence, not
  active model references. [Retired record](../journal/2026-09-13-retired-jepa-promoted-results.md),
  [measured value](../journal/2026-09-13-jepa-value-measured.md).
