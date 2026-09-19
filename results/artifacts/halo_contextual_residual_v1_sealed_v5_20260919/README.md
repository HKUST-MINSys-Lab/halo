# HALO bounded contextual residual v1, sealed evaluation

Run date: 2026-09-19. Architecture `support_contextual_residual_v1`, trained end to end for
40,000 steps. This evaluation uses the internally selected step-35,000 checkpoint, chosen only on
the subject-held-out training-source validation panel. Checkpoint SHA-256:
`471ec05f45b83e563de30e489a875b57ed4e77b1002eba59b6319e8b7ff7ebdd`.

The complete v5 sealed protocol finished all 39 cells over six datasets, 4/8/16-second evidence
windows, and `k = 0, 1, 2, 4, 8, 16, 32, 64, 128`. It produced 2,190 rows without a failed cell.
The episode manifest is byte-identical to the current released-baseline artifact.

This is a valid negative result, not the promoted classifier. At 8 seconds, dataset-balanced macro
F1 for the full classifier is 50.0/59.7/64.0/67.6/70.3 at `k=0/1/8/32/128`; its parameter-free
support floor reaches 59.8/72.0/75.4/76.3 at `k=1/8/32/128`. The learned head therefore retains
useful zero-support behavior but degrades the stronger support evidence as enrollment grows.

`results.json.gz` is the exhaustive result payload, `episode_manifests.json.gz` is the exact
evaluation plan, and `RESULTS.md` is generated directly by the evaluator. Uncompressed hashes and
the exact training configuration are retained beside them.
