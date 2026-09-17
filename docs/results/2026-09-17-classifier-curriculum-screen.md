# Classifier Curriculum Screen - 2026-09-17

## Scope

This is a bounded **development** experiment, not sealed evaluation. Four arms were trained from
the same initialization for 3,000 optimizer steps on the current corrected sampler. The encoder,
classifier, corpus, seed, optimizer, candidate policy, and checkpoint-selection rule were fixed.

The arms differ only in:

1. corrected Stage A: fixed support-count semantic weight, no rate/modality perturbation;
2. perturbations only: rate probability 0.25 and gyroscope-dropout probability 0.20;
3. adaptive gate only: evidence-dependent semantic gate, no perturbation;
4. perturbations plus adaptive gate.

Checkpoint selection used enrolled dataset-macro F1 on subject-held-out development data. The
condition panels use 64 deterministic support sets each. They do not consume sealed test data.

## Training Screen

| arm | selected step | enrolled macro-F1 | zero-shot macro-F1 | validation loss |
|---|---:|---:|---:|---:|
| corrected Stage A | 2,500 | **65.70** | **55.04** | **1.539** |
| adaptive gate only | 3,000 | 65.04 | 48.36 | 1.575 |
| perturbations only | 3,000 | 63.04 | 45.42 | 1.632 |
| perturbations + gate | 3,000 | 62.60 | 48.69 | 1.581 |

The three 3,000-step selections were still improving at the endpoint. These measurements screen
bad interactions; they are not convergence measurements.

## Recording Robustness

Dataset-macro F1 on the identical mixed episode panel:

| arm | clean | rate mixture | gyro-dropout mixture | gyro delta vs own clean |
|---|---:|---:|---:|---:|
| corrected Stage A | **65.70** | **65.12** | 53.09 | -12.61 |
| adaptive gate only | 65.04 | 64.47 | 53.42 | -11.62 |
| perturbations only | 63.04 | 61.98 | **59.82** | **-3.22** |
| perturbations + gate | 62.60 | 61.04 | 57.78 | -4.82 |

The perturbation arm buys real gyroscope-dropout robustness: +6.73 points over corrected Stage A
under dropout, despite a -2.66 point clean cost. It does not improve the measured rate panel. Rate
and modality perturbations were bundled here, so a rate-only versus modality-only screen is needed
before assigning the gain to one transform.

## Deployment Conditions

Each acquisition or enrollment row below is a separate deterministic panel, not a small slice of
the mixed draw.

| arm | compatible | cross placement | cross dataset | complete | partial | zero |
|---|---:|---:|---:|---:|---:|---:|
| corrected Stage A | **54.74** | **50.67** | 70.40 | 57.45 | 52.07 | 49.64 |
| adaptive gate only | 54.13 | 47.91 | **80.40** | **59.97** | **56.54** | **56.83** |
| perturbations only | 50.60 | 47.11 | 71.34 | 58.94 | 56.48 | 50.73 |
| perturbations + gate | 46.91 | 48.25 | 71.20 | 57.57 | 51.87 | 49.91 |

The adaptive gate is the targeted winner: relative to corrected Stage A it adds +10.00 points on
cross-dataset support, +4.47 on partial enrollment, +7.19 at zero enrollment, and +2.52 on complete
enrollment while giving up 0.61 on compatible support. Its cross-placement score is 2.76 points
lower, so it is not uniformly better.

## Classifier Corrections

On the clean mixed panel, relative to each arm's exact neighbor floor:

| arm | neighbor accuracy | classifier accuracy | rescue | overturn | net gain |
|---|---:|---:|---:|---:|---:|
| corrected Stage A | 47.62 | **59.52** | **14.29** | 2.38 | **+11.90** |
| adaptive gate only | 49.21 | 57.94 | 9.52 | **0.79** | +8.73 |
| perturbations only | 46.03 | 55.56 | 11.90 | 2.38 | +9.52 |
| perturbations + gate | **50.00** | 53.17 | 9.52 | 6.35 | +3.17 |

The gate does what its evidence-dependent design intended in one respect: it cuts harmful
overturns. It is also more conservative and makes fewer rescues, so its aggregate net gain is
smaller than corrected Stage A. The combined arm has a genuine interaction problem: it overturns
more correct neighbors and extracts little benefit from the classifier.

## Decision

- Do not promote or extend the combined perturbation-plus-gate recipe as currently configured.
- The adaptive gate alone is the best candidate for a longer matched run because it targets the
  deployment conditions where semantic reasoning is needed and remains close on clean data.
- Modality perturbation is promising for robustness, but separate rate-only and modality-only
  screens should precede any long perturbation run.
- Explicit metadata tokens remain deferred. The current gate already produces measurable gains in
  partial, zero, and cross-dataset conditions without adding that complexity.

Full machine-readable metrics are in
`training/support_classifier/evaluations/curriculum_ablation_screen_20260917.json`; the compact
generated table is beside it as `.md`. The sampler feasibility audit is in
`training/support_classifier/evaluations/curriculum_audit_20260917.{md,json}`.

## Limitations

- One seed and a 3,000-step budget.
- Internal subject-held-out data only; no sealed-result claim.
- The perturbation arm bundles rate and modality transforms.
- Selected steps differ because the declared validation rule, rather than the final step, chooses
  each checkpoint.
