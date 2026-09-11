# HALO results record

This file is the only promoted result record for the current support-conditioned HAR design.
Historical Phase-B and movement-monitoring results are archived and deliberately not mixed here.

No sealed result from the cleaned protocol is promoted yet.

Every completed run must add a dated section containing:

- code commit, encoder arm, checkpoint hash, and whether the encoder was frozen or end to end;
- training and development episode-manifest fingerprints;
- candidate-count and support-count distributions;
- all retained baseline rows under the same eligible protocol;
- per-test-dataset metrics and uncertainty, then any aggregate; and
- machine-readable artifact paths, run time, peak memory, and known exclusions.

Never select a test checkpoint or threshold by the headline table. Development selection and sealed
test reporting are specified in [EVALUATION_PROTOCOL.md](../design/EVALUATION_PROTOCOL.md).
