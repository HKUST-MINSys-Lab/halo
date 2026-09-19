# HALO bounded contextual residual v1, deployment scenarios

Run date: 2026-09-19. Architecture `support_contextual_residual_v1`, trained end to end for
40,000 steps and evaluated from the internally selected step-35,000 checkpoint. Checkpoint
SHA-256: `471ec05f45b83e563de30e489a875b57ed4e77b1002eba59b6319e8b7ff7ebdd`.

The representative `deployment-scenarios-v5-20260918` suite completed all seven scenarios at
8-second evidence and `k = 0, 1, 4, 8, 32`: 2,395 result rows, 1,132 task units, and zero task
failures. The manifest is byte-identical to the current released-baseline scenario artifact.

The learned classifier remains useful under partial enrollment, where it can name candidates with
no enrolled support, but it does not improve complete-enrollment evidence consistently. This
artifact is retained as a valid design result and is not promoted over the current residual
classifier.

`results.json.gz` contains every metric row, `manifests.jsonl.gz` records the exact episodes,
`paired_deltas.json` contains matched-control comparisons, and `RESULTS.md` is generated directly
from those rows. Uncompressed hashes and the exact training configuration are retained beside
them.
