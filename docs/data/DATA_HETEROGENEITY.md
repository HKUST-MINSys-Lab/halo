# Heterogeneous IMU data contract

HALO accepts heterogeneous IMU only when the difference is explicit in data and model inputs. This
file is the live contract for converters, grid construction, encoder inputs, and support episodes.

## Preserved metadata

Every source recording must retain stable dataset, subject, session, stream, and recording IDs;
timestamps in seconds; source and effective rate; device/placement description; gravity state;
canonical units; channel order; and per-channel validity. A converter may reject an uncertain field,
but it must not silently guess a measurement or conceal a missing channel.

## Physical channels

The harmonized representation reserves `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`.
Acceleration is in `g`; angular velocity is in `rad/s`. Acceleration-only recordings have masked,
zero-filled gyro slots. A mask means unavailable, not a legitimate zero measurement. NaN, Inf,
clock gaps, and session boundaries are hard boundaries for sampling and target construction.

## Rates and durations

Frontend geometry, patch spans, positions, and JEPA horizons use physical seconds. Native rate is
preserved alongside any frontend-specific resampling. A rate conversion required by an external
baseline belongs in that baseline adapter and is documented as part of its published input contract.

## Metadata conditioning

Configuration text can describe observed channels and their acquisition context. It is not a label
and must not encode dataset identity as a shortcut. The current sensor-only learned support
comparator deliberately does not consume this text; it reasons from query/support sensor vectors
only. Configuration remains available to the encoder where it can explain honest input variation.

## Split and episode rules

Subject, recording, and session provenance must survive all preprocessing so splits occur before
windows and support episodes are constructed. Query/support pairs may only combine configurations
allowed by the declared experiment. When a baseline cannot accept the real channel layout, mark the
combination unsupported instead of replacing the input with an undocumented proxy.
