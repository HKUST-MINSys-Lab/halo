# Training

Two live training surfaces remain.

## `tokenizer/`

Encoder reconstruction, data-contract, and historical future-JEPA reproducibility utilities. The
future-JEPA launcher is retired and requires an explicit archival acknowledgment; it is not part of
the active training recipe.

## `support_classifier/`

Support-conditioned HAR training. The same selected encoder embeds query and support recordings.
The semantic token mixer attends jointly over query, support, paired support-label, and candidate
tokens before a soft candidate vote; a separate but mechanically identical head handles `k=0`.
The `neighbors` control removes the mixer and isolates encoder quality. An end-to-end run updates
the HALO encoder, recording pool, and active head through both query and support paths.

The default trainer draws a 50/50 mixture of single-device and exact aligned multi-device examples
where a source provides simultaneous placements (`--multi-device-probability`, `--max-devices`).
Checkpoint validation is deterministic and single-device. The sealed evaluator owns the common
4/8/16-second manifests and reports both single-placement and declared composite cells.

The old `evidence/` directory is historical reproducibility code, not a live default. Do not add
new work there.
