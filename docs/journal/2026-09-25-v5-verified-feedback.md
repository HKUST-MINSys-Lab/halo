# V5 verified-label feedback

Date: 2026-09-25. Status: implemented, unit-tested, and two-step smoke-tested; no full training
or sealed evaluation run.
This extends the [v5 implementation record](2026-09-25-v5-online-memory-implementation.md).

Every memory entry keeps a detached copy of its pre-label, no-memory semantic distribution and
the ordered candidate roster used to produce it. A verified label is stored separately, never in
place of that distribution. On a future query, a verified entry votes with its ground-truth label,
while the reader also receives five label-blind feedback scalars: whether a comparison is
available, the original probability of the true label, the original top probability minus that
probability, the original distribution's normalized entropy, and the original roster's chance
level (`1/C`). These values can condition
reliability and the semantic/memory blend; they cannot change the verified vote itself.

The comparison is unavailable for unverified entries, for verified labels absent from the original
roster, or when the verified label is outside the current roster. When the roster changes, current
semantic evidence is recomputed for voting, but the original snapshot and its feedback remain
anchored to the original roster. A snapshot round trip preserves them. The zero-memory prediction
path is unchanged. This changes the reader's `evidence_proj` input width, so older v5 smoke
checkpoints cannot be loaded into this revision; v4 checkpoints are unaffected.
