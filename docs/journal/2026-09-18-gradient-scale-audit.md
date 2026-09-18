# Support-classifier gradient-scale audit

Date: 2026-09-18

## Question

Why does the active HALO residual-classifier recipe spend most logged steps under global
gradient clipping, and does that indicate broken initialization or unstable activations?

## Method

The bounded support-classifier profiler was extended to record pre-clip block norms and score
magnitudes. Four ten-step runs used identical seeds, episodes, and model initialization while
varying only the support-vote and query-to-label temperatures. These are scale diagnostics, not
model-quality experiments.

| support temp | text temp | encoder norm | classifier norm | recording pool | sensor fold | text bridge | attention |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.07 | 0.07 | 17.98 | 9.18 | 12.20 | 10.28 | 8.93 | 0.12 |
| 0.20 | 0.07 | 16.48 | 9.36 | 12.08 | 8.69 | 9.13 | 0.09 |
| 0.07 | 0.20 | 9.14 | 3.53 | 5.80 | 5.45 | 3.05 | 0.14 |
| 0.20 | 0.20 | 6.18 | 3.02 | 4.00 | 3.65 | 2.61 | 0.11 |

At the default temperatures, valid candidate logits had standard deviation 9.06 and maximum
absolute magnitude 27.65. The metric path produces log vote probabilities, so near-zero class
votes naturally approach `log(1e-12) = -27.63`. The text score is also cosine divided by 0.07.
The combination is finite and mathematically defined, but sharp. With text temperature 0.20, text
score standard deviation fell from 1.77 to 0.74 and its bridge gradient fell from 8.93 to 3.05.

## Interpretation

No evidence indicates an activation explosion, NaN boundary, failed normalization, or malformed
initializer. The large gradients are mainly the expected `1 / temperature` amplification from two
sharp cosine paths, propagated through the learned recording pool and sensor fold. The closed-form
text bridge starts useful (calibration cosine about 0.92 in the smoke) but is still wrong on many
candidate decisions, so its sharp zero-support cross-entropy dominates early updates.

Global clipping prevents numerical instability, but it also scales every block together. Under the
default recipe the post-clip attention norm was only 0.0058 while the text bridge was 0.443. Thus a
sharp semantic path can suppress the contextual attention and residual heads that the experiment is
trying to train. This is a learning-balance concern, not merely cosmetic telemetry.

Historical logs confirm persistence rather than a one-step transient: total pre-clip norm was 164.9
at step 1, 37.4 at step 500, 8.70 at step 2,500, 3.17 at step 10,000, 5.82 at step 30,000, and 5.34
at step 40,000, against the fixed clip threshold of 1.0.

## Decision boundary

Do not remove clipping or raise its threshold blindly. That would expose the optimizer to the same
branch imbalance without correcting it. Before another full residual-classifier run, compare short
matched screens at `(support, text)` temperatures `(0.07, 0.07)`, `(0.07, 0.20)`, and
`(0.20, 0.20)`. Select using the fixed internal enrolled and zero-support panels, classifier gain
over its neighbor floor, and clipping/block-norm telemetry. A temperature change is promoted only
if it improves learning balance without reducing either information regime materially.

Longer term, the proposed candidate-specific mixture should normalize the support and semantic
paths before combining them and use a bounded learnable scale. That addresses the source of the
magnitude mismatch more directly than a larger clip threshold.
