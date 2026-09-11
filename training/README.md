# Training

Two live training surfaces remain.

## `tokenizer/`

Optional label-free future-JEPA pretraining for a HALO encoder. The student observes a valid prefix
of an IMU region, predicts later patch states from an EMA teacher, and decodes physical targets from
those predictions. See [JEPA_PRETRAINING_OBJECTIVE.md](../docs/design/JEPA_PRETRAINING_OBJECTIVE.md).

## `support_classifier/`

Support-conditioned HAR training. The same selected encoder embeds query and support recordings;
the comparator learns only to adjust support-row evidence before the explicit candidate vote. A
frozen-encoder run isolates representation quality. An end-to-end run updates the HALO encoder,
recording pool, and comparator through both query and support paths.

The old `evidence/` directory is historical reproducibility code, not a live default. Do not add
new work there.
