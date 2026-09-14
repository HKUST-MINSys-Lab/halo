# Fixed-filterbank polarization features

The fixed physical filterbank can append bounded, rotation-aware features for a complete
accelerometer or gyroscope xyz triad. This is an optional fixed analysis block, not a new learned
frontend or a cross-sensor fusion mechanism.

For each physical band, complex FFT coefficients are pooled with `sqrt(H)`, while ordinary band
energy continues to use `H`. For a triad `Z=(Zx,Zy,Zz)`, the block emits the gravity-relative
vertical energy share, circularity, and signed spin, plus a gravity-confidence scalar. The three
values are bounded and silent bands are suppressed by a relative energy gate. Each triad value is
replicated onto its three channel rows, so the existing per-axis projection and sensor folding
contracts are preserved.

Gravity is taken only from raw accelerometer DC. A gyroscope triad may use that direction, but an
incomplete triad, an absent channel, missing triad metadata, or generic non-xyz rows are neutral
zeros. The implementation never invents a zero-filled axis.

`use_polarization=True` and `polarization_energy_kappa=0.05` are the defaults for newly created
fixed-filterbank encoders. A checkpoint lacking the serialized flag reconstructs with the old
98-dimensional analysis input (`False`) for strict state-dict compatibility. With `K=32`, the
enabled input is 195 dimensions.

Future-JEPA's physical decoder deliberately uses a separate, frozen filterbank with polarization
disabled. Its target stays the established band-energy plus signed-DC physical target; the
polarization block is a student representation input, not a reconstruction target.
