# Model components

## `tokenizer/`

The HALO encoder converts native-rate IMU into contextual physical-time patch vectors and an
optional recording representation.

| module | role |
|---|---|
| `filterbank.py` | fixed physical-Hz filterbank with signed low-frequency features |
| `continuous_kernel.py`, `multispan_kernel.py` | continuous temporal frontend and multispan variant |
| `preprocess.py` | truthful gravity and channel preparation |
| `sensor_tokens.py` | sensor-level token construction and validity masks |
| `transformer.py`, `encoder.py` | temporal/cross-sensor context and representation interface |
| `channel_text.py` | acquisition-configuration conditioning |
| `future_jepa.py` | pretraining-only EMA teacher, predictor, and physical decoder |

The encoder preserves patch-level states until a downstream recording pool is explicitly requested.
The support classifier can train that pool end to end; future-JEPA targets patch states, not pooled
recording vectors.

## `support/`

The active support-conditioned classifier starts from cosine similarity between a query recording
and support recordings. Label bindings turn weighted support rows into candidate scores. The learned
comparator is intentionally narrow: it set-attends only sensor representations plus query/support
role and episode-slot embeddings, then emits a scalar correction for each support row.

The retired evidence-engine modules remain in the repository only for archived checkpoint
compatibility. They are not imported by the active support-classifier path.
