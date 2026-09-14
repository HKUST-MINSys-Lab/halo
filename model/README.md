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
| `future_jepa.py` | retired pretraining-only EMA teacher, predictor, and physical decoder |

The encoder preserves patch-level states until a downstream recording pool is explicitly requested.
The support classifier can train that pool end to end. The retired future-JEPA path targeted patch
states rather than pooled recording vectors.

## `support/`

The active support-conditioned classifier pools each recording to one motion vector. Its semantic
token mixer jointly attends to the query vector, every support vector, each paired support-label
token, and candidate-label tokens. Role embeddings distinguish token type; pair and candidate tags
preserve the support-label-candidate bindings. It then scores query/support cosine similarity and
softly votes support evidence to the candidate roster. A separate head handles the `k=0` query plus
candidate-label condition.
