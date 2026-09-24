# Phase-B Decoder Diagnostics

Date: 2026-08-11  
Branch: `codex/phase-b-decoder-diagnostics-20260811`  
Base commit: `8b222f5024f2085704c7381e7d106037b3f11346`

## Question

Why does the trained relational decoder usually fail to improve on the retrieval-only identity vote?

The tested predictor is the v22 step-1000 relational decoder. Before diagnostic instrumentation, its
recorded Phase-B source fingerprint exactly matched the isolated source snapshot:
`2ec5703b49819189071a241d538472b5afadcde943d2e27fa3551a27fa6e1d94`.

The sealed test roster was not opened. All external results use the development datasets MotionSense,
RealWorld, and Shoaib. Development grids were copied into the worktree before the final paired runs so
other work on the active checkout could not change the cohort.

## Experiments

1. Restrict learned retrieval to enrolled support rows, excluding background memory.
2. Initialize retrieval from v22, freeze it, and train a fresh decoder for 1,000 matched steps.
3. Compare decoder and identity predictions from the exact same assembled evidence roster.
4. Compare full memory, support-only memory, and background-only memory.
5. Measure attention and remove the decoder query signal, evidence signal, text, or coreference while
   leaving learned retrieval unchanged.
6. Give the model only support bound to the true candidate. This target-leaking arm is a diagnostic of
   candidate binding and readout, not a valid prediction score.

The default and diagnostic evaluators produced byte-equivalent cohort sizes and identical primary
scores on the fixed snapshot. The diagnostic implementation passed 127 focused tests and the synthetic
Phase-B training smoke test.

## Cross-Subject Results

Values are macro F1 percentages, averaged equally over the three development datasets. Each
positive-support row uses 17,837 queries and arbitrary episode-local aliases.

Zero support is evaluated separately on 17,871 queries because an arbitrary alias with no enrolled
example is information-theoretically unanswerable. The `k=0` protocol therefore uses coherent activity
names.

| k=0 coherent | Joint v22 decoder | Frozen-retriever decoder | Retrieval-only semantic vote |
|---|---:|---:|---:|
| MotionSense | 12.05 | 15.30 | 45.26 |
| RealWorld | 8.54 | 11.64 | 10.14 |
| Shoaib | 10.07 | 13.39 | 18.24 |
| Equal-dataset mean | **10.22** | **13.45** | **24.55** |

There is no subject relation at `k=0` because no enrollment subject exists; the evaluator's
`same_subject` and `cross_subject` cells are therefore identical. The vote uses learned retrieval over
background memory and coherent label-text similarity. The decoder loses most of that semantic bridge.

| k | Decoder | Same-roster vote | Support-only decoder | Support-only vote | Background-only decoder |
|---:|---:|---:|---:|---:|---:|
| 1 | 49.40 | 50.22 | 53.39 | 55.23 | 14.05 |
| 2 | 54.78 | 56.70 | 56.62 | 60.14 | 14.05 |
| 4 | 60.64 | 64.71 | 61.74 | 66.18 | 14.05 |
| 8 | 65.37 | 69.37 | 65.91 | 70.45 | 14.05 |

Background rows hurt both methods, particularly at low support, by competing for retrieval capacity and
vote mass. Removing them does not make the decoder beat the vote: the support-only decoder still trails
the support-only vote at every k.

## Paired Corrections

| k | Decoder-only correct | Vote-only correct | Net decoder corrections | Predictions changed |
|---:|---:|---:|---:|---:|
| 1 | 1,220 | 1,481 | -261 | 3,869 |
| 2 | 1,286 | 1,735 | -449 | 4,126 |
| 4 | 1,175 | 1,863 | -688 | 3,937 |
| 8 | 1,156 | 1,850 | -694 | 3,818 |

The decoder is not inert. It changes about 21-23% of predictions and sometimes fixes retrieval errors.
Its correction precision is insufficient: harmful changes outnumber helpful changes, increasingly so as
more support becomes available.

## Causal Input Ablations

| k | Full decoder | No query signal | No evidence signal | No text | No coreference |
|---:|---:|---:|---:|---:|---:|
| 1 | 49.40 | 49.35 | 49.85 | 50.13 | 14.89 |
| 2 | 54.78 | 54.94 | 54.87 | 55.56 | 14.83 |
| 4 | 60.64 | 60.78 | 60.42 | 62.21 | 15.32 |
| 8 | 65.37 | 65.51 | 65.17 | 66.88 | 14.77 |

The decoder's query and evidence vector projections add no measurable cross-subject value after learned
retrieval has selected and scored the roster. Text is neutral or mildly harmful in arbitrary-alias
episodes. Exact support-to-candidate coreference is load-bearing and accounts for almost the entire
adaptation gain.

The final decoder block's candidate attention averaged 70.7% to evidence tokens, 18.8% to candidate
tokens, 10.1% to query tokens, and 0.4% to background-label tokens. Attention mass is not causal use:
removing query or evidence vector content left performance unchanged because retrieval scores and
structural support bindings remained.

## Oracle Candidate Scoring

When retrieval was restricted to support bound to the true candidate, both the decoder and identity vote
achieved 100% F1 at every k. The decoder's mean true-candidate probability was 0.90-0.91, versus 0.98 for
the vote. The readout therefore understands exact candidate binding, but is less decisive than the vote
and loses accuracy when it must compare support from several candidates.

## Frozen-Retrieval Training

A fresh decoder was trained for 1,000 steps with the v22 retriever fixed. It learned normally, selected
step 800, had zero retriever gradient and roster drift by construction, and passed all mechanism gates.

| k | Joint v22 decoder | Frozen-retriever decoder | Same fixed identity vote |
|---:|---:|---:|---:|
| 0, coherent | 10.22 | 13.45 | 24.55 |
| 1 | 49.40 | 49.42 | 50.22 |
| 2 | 54.78 | 55.05 | 56.70 |
| 4 | 60.64 | 61.85 | 64.71 |
| 8 | 65.37 | 65.72 | 69.37 |

Freezing retrieval did not expose a hidden decoder advantage. It modestly improved some external cells
but still remained below the identical vote. Joint nonstationarity is not the primary failure.

## Diagnosis

This is primarily a task-design and model-redundancy issue, not gradient starvation or numerical
instability. In every positive-support training episode, all candidates receive the same k and all names
are arbitrary aliases. The sufficient solution is therefore: retrieve support, follow its exact candidate
binding, and weight labels by retrieval similarity. That is exactly what the closed-form vote already
does. The decoder is given no recurring condition in which richer query-evidence reasoning is necessary
to beat that rule.

At zero support the failure is complementary: coherent text and semantically related background rows are
the only available bridge, but the learned decoder underuses text and scores far below the same-roster
semantic vote. Positive-support training therefore teaches a structural binding shortcut without
preserving the stronger zero-support fallback.

Background evidence is also not serving the intended compositional role. It consumes capacity and hurts
scores; the decoder cannot recover that loss. The learned transformer currently behaves as a noisier,
less decisive approximation to its own retrieval vote.

## Recommended Next Experiment

If the relational decoder is retained, use the same retrieval-only logits for every candidate as an
explicit base and train a zero-initialized gated residual over them. Unlike the removed historical base,
this must not switch between prototype and text branches. It should be one uniform retrieval-vote path,
with the transformer responsible only for corrections. Checkpoint selection must require positive paired
gain over the base.

Keep the clean curriculum, but add one narrowly defined episode type that creates a reason for relational
reasoning: coherent partial enrollment, where some candidates have support and others must use semantic
or background evidence. Do not reintroduce personas or broad augmentation until this simple capability is
demonstrated. If the residual still cannot produce positive paired corrections, the decoder should be
parked and retrieval-only should become the Phase-B model.

## Artifacts

- `training/evidence/outputs/decoder_diagnostics/v22_alias_dev_cross_full_snapshot.json`
- `training/evidence/outputs/decoder_diagnostics/v22_alias_dev_same_full_snapshot.json`
- `training/evidence/outputs/decoder_diagnostics/v22_alias_dev_cross_control.json`
- `training/evidence/outputs/decoder_diagnostics/v22_coherent_dev_k0_snapshot.json`
- `training/evidence/outputs/decoder_diagnostics/frozen_retriever/patch_evidence_predictor.pt`
- `training/evidence/outputs/decoder_diagnostics/frozen_retriever/eval_alias_dev_cross_snapshot.json`
- `training/evidence/outputs/decoder_diagnostics/frozen_retriever/eval_coherent_dev_k0_snapshot.json`
- `training/evidence/outputs/decoder_diagnostics/frozen_retriever/telemetry/`
