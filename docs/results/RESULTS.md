# HALO results record

Promoted results for the support-conditioned HAR design, newest first.

**The one rule: never mix protocols.** A number is only comparable to another number produced by
the same protocol on the same manifests. Every table below states its protocol, and superseded
tables are not reproduced here: their numbers live in their immutable artifacts, linked in the index.
No sealed result has ever selected a checkpoint or tuned a threshold.

## Which run is which

Names collide easily in this project, so use this index rather than inferring a run from a
directory name. "Arm" means a training run; "readout" means how a trained checkpoint is scored.

| run | protocol | encoder conditioning | status | numbers |
|---|---|---|---|---|
| HALO bounded contextual residual v1, step 35k, 2026-09-19 | v5 | acquisition-conditioning-v2 | **completed negative result** | [record](../journal/2026-09-19-bounded-contextual-residual-v1-results.md) |
| HALO residual arm, step 40k, 2026-09-18 | v5 | acquisition-conditioning-v2 | **current** | below |
| HALO neighbours arm, step 40k, 2026-09-18 | v5 | acquisition-conditioning-v2 | **current** | below |
| HALO contextual arm, step 40k, 2026-09-18 | v5 | acquisition-conditioning-v2 | **failed, recorded** | below |
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
published participant split). The baseline column is the strongest released model under cosine 1-NN,
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
| cross dataset | 1 | **76.8** | 72.0 | 74.1 | 62.6 UniMTS |
| cross dataset | 8 | **83.3** | 81.7 | 80.4 | 72.1 UniMTS |
| cross dataset | 32 | **84.7** | 84.0 | 82.3 | 74.5 UniMTS |
| missing modality | 1 | **60.3** | 58.0 | 57.0 | 57.6 UniMTS |
| missing modality | 8 | 67.5 | 67.7 | 65.2 | **71.3** UniMTS |
| missing modality | 32 | 68.5 | 70.1 | 67.1 | **74.8** UniMTS |
| rate mismatch | 1 | **64.9** | 61.6 | 59.9 | 57.8 UniMTS |
| rate mismatch | 8 | **72.7** | 71.9 | 68.6 | 71.1 UniMTS |
| rate mismatch | 32 | 74.1 | 73.7 | 70.3 | **74.3** UniMTS |
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


### The contextual arm: a recorded negative result

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

### The bounded contextual residual v1 follow-up

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
