# Current architecture

Last verified against code: 2026-09-24.

HALO consumes native-rate accelerometer and optional gyroscope streams. Sampling rate, gravity
availability, and modality presence are structured metadata; device role and placement are natural
language acquisition context. Missing channels are masked rather than fabricated.

The active encoder uses a fixed physical filterbank at `0.5`, `1`, `2`, and `4` seconds. Duration
embeddings distinguish those resolutions before temporal contextualization. Every patch remains a
token until a learned recording pool produces one vector per query or support execution.

Classifier lifecycle is explicit:

| role | architecture | status |
|---|---|---|
| promoted classifier | **v4** = `support_classifier_v4` in the T6 recipe | promoted 2026-09-21; the try-number index is in [RESULTS.md](../results/RESULTS.md#classifier-naming-adopted-2026-09-20) because architecture strings and try numbers do not line up |
| superseded | `support_classifier_v3` | promoted 2026-09-18 → 2026-09-21 |
| recorded negative results | `support_evidence_aware_v2` (T3); T7 and T8 primitive-path recipes | not active; reproduction only |
| encoder-only control | differentiable neighbours | Parameter-free head; not a learned classifier |
| historical negative results | `support_contextual_mixture_v1`, `support_contextual_residual_v1` | Abandoned; checkpoint loading only |
| retired predecessor | `support_token_mixer_v1` | Reproduction only |

The promoted head is
[`model/support/evidence_gated_classifier.py`](../../model/support/evidence_gated_classifier.py)
(`support_classifier_v4`). It takes the parameter-free support vote as its floor, learns a bounded
residual and a gated blend with the label-text term, and cannot emit unrestricted class logits. The
differentiable-neighbour mode removes the learned classifier and remains the representation control.
Rung 1 of the [roadmap](roadmap.md) scores the encoder with no learned head — only its text
projection `p_text` feeds the transductive method — and rung 2's case study reports the
parameter-free vote and plain 1-NN beside the head.

Training draws complete, partial, and zero-enrollment episodes; compatible, cross-placement, and
cross-dataset acquisition relationships; and optional rate, modality, and aligned multi-device
perturbations. Checkpoint selection uses subject-held-out dataset-macro F1, with enrolled and
zero-support panels reported separately.

Code layout: the encoder is under `model/tokenizer/`, the heads under `model/support/`, training
under `training/support_classifier/` (with shared data and encoder-building infrastructure in
`training/tokenizer/`), and every evaluation rung under `evaluation/` — `rung1_unlabeled/`,
`rung2_frozen/` (the sealed and scenario runners), `rung3_finetune/`, plus the shared feature,
manifest, zero-shot and provenance modules.

Future-JEPA, continuous/multispan kernels, the old token mixer, both v1 contextual heads, explicit
admissibility gates, and hidden memory-bank retrieval are retired or abandoned. Their code and
rationale are preserved for reproduction under source modules, `docs/archive/`, `docs/journal/`,
and historical Git tags. The lifecycle strings used by code are authoritative in
`model/support/factory.py`.
