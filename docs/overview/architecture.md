# Current architecture

Last verified against code: 2026-09-19.

HALO consumes native-rate accelerometer and optional gyroscope streams. Sampling rate, gravity
availability, and modality presence are structured metadata; device role and placement are natural
language acquisition context. Missing channels are masked rather than fabricated.

The active encoder uses a fixed physical filterbank at `0.5`, `1`, `2`, and `4` seconds. Duration
embeddings distinguish those resolutions before temporal contextualization. Every patch remains a
token until a learned recording pool produces one vector per query or support execution.

The active learned head is the contextual residual classifier in
[`model/support/contextual_residual_classifier.py`](../../model/support/contextual_residual_classifier.py).
It jointly contextualizes query, support, paired support-label, candidate-label, and runtime
acquisition vectors, but it cannot emit unrestricted logits. Its outputs remain an auditable
centered-neighbor support vote plus candidate-local semantic evidence. The differentiable-neighbor
mode removes the learned classifier and remains the representation-control experiment.

Training draws complete, partial, and zero-enrollment episodes; compatible, cross-placement, and
cross-dataset acquisition relationships; and optional rate, modality, and aligned multi-device
perturbations. Checkpoint selection uses subject-held-out dataset-macro F1, with enrolled and
zero-support panels reported separately.

Future-JEPA, continuous/multispan kernels, the old token mixer, explicit admissibility gates, and
hidden memory-bank retrieval are retired. Their code and rationale are preserved under
`docs/archive/`, `docs/journal/`, and historical Git tags.
