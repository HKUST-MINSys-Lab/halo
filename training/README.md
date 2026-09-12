# Training

Two live training surfaces remain.

## `tokenizer/`

Optional label-free future-JEPA pretraining for a HALO encoder. The student observes a valid prefix
of an IMU region, predicts later patch states from an EMA teacher, and decodes physical targets from
those predictions. See [JEPA_PRETRAINING_OBJECTIVE.md](../docs/design/JEPA_PRETRAINING_OBJECTIVE.md).

## `support_classifier/`

Support-conditioned HAR training. The same selected encoder embeds query and support recordings.
The semantic token mixer attends jointly over query, support, paired support-label, and candidate
tokens before a soft candidate vote; a separate but mechanically identical head handles `k=0`.
The `neighbors` control removes the mixer and isolates encoder quality. An end-to-end run updates
the HALO encoder, recording pool, and active head through both query and support paths.

The old `evidence/` directory is historical reproducibility code, not a live default. Do not add
new work there.
