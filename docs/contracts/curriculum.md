# Training curriculum

Last verified against code: 2026-09-24. Authority: the argument defaults of
`training/support_classifier/train.py` and the constants at the top of
`training/support_classifier/sampling.py`; `PROMOTED_RECIPE` in `model/support/factory.py`. The
2026-09-16 curriculum experiment that produced most of these defaults is archived at
[`../archive/curriculum-experiment-20260916.md`](../archive/curriculum-experiment-20260916.md).

HALO is trained end to end (encoder, recording pool and support classifier) on the eight
supervised training sources, from random initialisation, in 8-second windows. Every episode is a
query, a declared candidate roster and a support set whose shape is drawn to mirror deployment.
Checkpoints use internal subject-held-out validation only; the sealed six never select anything.

## The v4 recipe (promoted 2026-09-21)

`--classifier evidence_gated` (the default) trains `support_classifier_v4` in the T6 recipe:

| axis | what the sampler draws | default |
|---|---|---|
| enrollment count | supports per candidate | `k ∈ {1, 2, 4, 8, 16, 32}`, up to 32 supports per episode |
| enrollment shape | complete / partial / zero-support episodes | mix `0.50 / 0.25 / 0.25`; partial episodes enrol `25–75 %` of the roster |
| unequal enrollment | per-candidate counts differ within an episode | probability `0.50` |
| acquisition relation | support from the query's configuration / another placement / another dataset | mix `0.50 / 0.25 / 0.25` |
| multi-device | exact event-aligned 2–4-device recordings (realdisp, xrf_v2, dsads, forth_trace) | probability `0.5`, up to 4 devices |
| device-set challenge | coordinated query/support device-set mismatch | probability `0.5` |
| label-text corruption | a deranged-roster view whose gradient reaches only the blend gate (T6: an auxiliary, not a replacement) | every eligible episode, weight `1.0` |
| unenrolled calibration | a label-blind bias for candidates with no support of their own | on |
| windows per execution | distinct windows drawn per physical execution | 2 |

Off by default: rate augmentation, modality dropout, counterfactual enrollment groups. The v4
checkpoint was trained for 40,000 steps (`--steps 40000`; the CLI default is 35,000), 4 episodes per
step, seed 20260901, bf16.

## The unlabelled-pool arm (rung 1's training arm; built, not run)

`--pool-size N --pool-mode {transductive, soft_kmeans}` attaches an unlabelled pool to each
episode, structured the way deployment pools are ([roadmap](../overview/roadmap.md)):

| option | meaning | default |
|---|---|---|
| `--pool-regime-mix` | pool drawn from the query's configuration / another placement / another dataset | `0.5 / 0.25 / 0.25` |
| `--pool-concentration` | Dirichlet concentration of the pool's class marginal (imbalance) | `1.0` |
| `--pool-coverage` | fraction of roster classes present in the pool | `0.5–1.0` |
| `--pool-distractor-fraction` | share of pool windows from labels outside the roster | `0.25` |
| `--pool-temperature` | softmax temperature of the probability features | `30` |
| `--pool-unroll-iters`, `--pool-unroll-mm-iters` | unrolled EM and MM iterations | `5`, `20` |
| `--pool-affinity-mu`, `--pool-affinity-knn` | embedding-affinity term of the transductive readout | `1.0`, `10` |

`transductive` scores pooled episodes through the rung-1 evaluator's own EM-Dirichlet
(`evaluation/rung1_unlabeled/transductive.py`), unrolled, so what HALO is trained through is
exactly what every encoder is scored with. It trains the encoder, the recording pool and `p_text`;
the v4 head's gates are not used on pooled episodes. `soft_kmeans` is the Ren-2018 control (one
E-step under the support vote, no unrolling). With `--pool-size 0` the recipe is bit-identical to v4.
All pool options are persisted in the checkpoint and checked on resume.
