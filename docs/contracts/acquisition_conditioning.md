# Acquisition Conditioning Contract

**Status:** design of record, schema `acquisition-conditioning-v2`. Last verified against code: 2026-09-24.

This document is the authoritative contract for turning runtime acquisition metadata into HALO
encoder inputs. Dataset converters and evaluation derivations must populate this contract. They
must not construct ad hoc description strings or infer metadata from dataset names.

The encoder exposes a per-sensor learned acquisition vector formed from its projected natural-
language device/placement description plus its structured modality, gravity, and effective-rate
embedding. The support classifier pools sensors within device and devices within recording. This
is the only classifier metadata vector; evaluation reconstructs it with the checkpoint's own
conditioner separately for query and support streams.

## Design

Conditioning has two independent residual branches after physical sensor features are projected to
the encoder width:

1. **Natural-language context:** device role and body placement only. Frozen
   `all-MiniLM-L6-v2` produces a 384-dimensional, mean-pooled and L2-normalized sentence vector.
   A trainable MLP and LayerNorm map it to the sensor-token width. A token-dependent sigmoid gate
   applies it as a residual.
2. **Exact structured context:** modality, gravity convention, and effective source rate. Modality
   and gravity use learned categorical embeddings. Effective rate uses the fixed
   `log2(rate / 50 Hz)` feature followed by a small MLP and LayerNorm. Stored rate remains in the
   batch metadata for provenance but is deliberately not a model input: after resampling it is a
   dataset/device-pipeline identifier rather than additional physical bandwidth. A separate token-dependent
   sigmoid gate applies this branch as a residual.

Patch duration and resolution identity remain structured temporal inputs in the encoder. They are
not repeated in either acquisition branch. The text and structured branches are added in parallel;
neither is used to overwrite the sensor vector.

Both gates initialize with zero weights and bias `-2`, so conditioning begins as a small residual.
The sigmoid is nonzero, so the text projection, structured embeddings, rate MLP, and gates all
receive gradient on the first update.

## Required StreamSpec Fields

Every stream used by HALO must have a registered `StreamSpec` in
`data/scripts/curate/deployment_policy.py`. Missing registrations are errors; there is no stream-name
fallback.

| Field | Required meaning | Rules |
|---|---|---|
| `dataset` | Stable source identifier | Never shown to the model. |
| `stream_id` | Stable physical stream identifier | Never shown to the model and never parsed for facts. |
| `device_profile` | Runtime device role | Use `phone`, `watch`, `watch_proxy`, or `device`. `watch_proxy` means a phone physically strapped at a wrist and is rendered as a phone, never a watch. Use `device` only when a more specific consumer-device role is not justified. |
| `placement` | Human-readable physical placement | State laterality and attachment site when known, for example `the right wrist`, `the left trouser pocket`, or `the head (smart glasses)`. Do not include activity, dataset, subject, disease, or label information. |
| `required` / `optional` | Raw-to-canonical channel mapping | Canonical output is an accel xyz triad and/or gyro xyz triad. A missing modality remains absent. |
| `gravity_state` | Accelerometer convention | Exactly `present`, `removed`, or `unknown`. This does not apply to gyroscope rows. |

The deterministic text renderer emits one device/placement sentence per present modality, such as
`a phone located at the right trouser pocket`. Co-located accelerometer and gyroscope rows receive
the same sentence. This is intentional: modality belongs to the exact structured branch.

The following must never enter the acquisition text: modality, gravity state, sampling rate, patch
duration, dataset name, stream ID, subject ID, activity label, candidate label, split, session ID,
or any property derived from the target.

## Structured Tensor Contract

For a batch with `B` recordings and at most `N` sensor rows:

| Tensor | Shape | Values |
|---|---:|---|
| `sensor_modality` | `(B, N)` | `0=accelerometer`, `1=gyroscope` |
| `sensor_gravity` | `(B, N)` | `0=present`, `1=removed`, `2=unknown`, `3=not applicable` |
| `sensor_rates_hz` | `(B, N, 2)` | `[stored_rate_hz, effective_source_rate_hz]`; only column 1 conditions the model |

Only accelerometer and gyroscope are accepted by schema v2. Magnetometer, barometer, ECG, and other
modalities must be rejected at onboarding rather than mapped to a nearby ID. The enum may be
extended in a future schema revision with a migration and explicit tests.

Gravity is `present`, `removed`, or `unknown` for accelerometer rows and always `not applicable` for
gyroscope rows. Do not use `unknown` to mean `not applicable`.

`stored_rate_hz` is the rate of the tensor presented to the encoder. `effective_source_rate_hz` is
the highest real acquisition bandwidth represented by that tensor:

```text
effective_source_rate_hz = min(hardware_acquisition_rate_hz, stored_rate_hz)
```

Upsampling changes the stored rate but cannot increase the effective source rate. Anti-aliased
downsampling lowers both. Both values must be finite and positive. Internal analysis downsampling
for long filterbank patches does not rewrite these acquisition facts.

The log-rate feature is always computed in float32, including during bfloat16 training, and only
the projected result joins the mixed-precision token stream. This preserves nearby physical rates
such as 50 and 51.2 Hz.

Ragged sensor rows are zero-padded by collate and masked by sensor presence. The conditioner
neutralizes padded values before embedding. A partially populated batch is invalid: modality,
gravity, and both rates must be supplied together for every real row.

## Augmentation And Derived Views

Metadata is derived after acquisition-changing augmentation:

- rate augmentation updates stored rate and caps effective source rate;
- gravity removal updates the accelerometer gravity enum, not the natural-language sentence;
- modality dropout removes the corresponding sensor row and compacts `sensor_id`;
- device/placement text dropout uses `a device at an unspecified placement` but leaves exact
  modality, gravity, and rate fields intact;
- natural-language paraphrase may change wording but not the represented device or placement.

Evaluation perturbations follow the same rules. A resampled scenario must report its resampled
stored rate and the minimum of original source rate and target rate. An accel-only scenario removes
the gyroscope row rather than presenting a zero gyroscope as measured evidence.

## Multi-Device Recordings

Each physical device contributes its own accel and/or gyro sensor rows, text, structured metadata,
and `device_id`. Simultaneous rows may be concatenated only when their sampled timelines and labels
agree. Per-device stored/effective rates are preserved in `sensor_rates_hz`; the lowest device rate
must not overwrite the rate metadata of every other device.

## Units And Channel Order

Canonical channels are ordered `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z` per device.
Acceleration is in **g** and gyroscope is in **rad/s**. Axis roles (`x`, `y`, `z`) remain separate
role metadata and do not belong in the device/placement sentence. Missing channels are masked and
zero-filled only for tensor layout; they are not advertised as present sensors.

## Checkpoint Compatibility

New checkpoints must serialize:

```json
{"conditioning_schema": "acquisition-conditioning-v2"}
```

Checkpoints without this field and without `structured_conditioner.*` weights reconstruct under the
historical `combined-text-v1` contract. They are never silently loaded into schema v2. The legacy
path selects the historical renderer as well as the historical weights: modality, gravity and
partner-presence clauses remain in its sensor sentence. It exists only for exact reproduction; all
newly initialized HALO support-classifier encoders use schema v2. Feature caches are schema-versioned
so legacy and v2 renderer outputs cannot mix.

## Retired Descriptor Objective

The old masked descriptor-retrieval objective predicts the frozen natural-language descriptor. In
schema v2 that descriptor intentionally identifies device role and placement only; co-located
accelerometer and gyroscope rows therefore share a target. The objective does **not** reconstruct
modality, gravity, or rate and must not be reported as doing so. It is disabled in the active
end-to-end classifier recipe and retained only for historical experiments.

## Dataset Onboarding Checklist

1. Register every physical stream with an explicit `StreamSpec`.
2. Verify device role and placement against source documentation; record ambiguity in `note`.
3. Verify native modality presence, units, channel order, hardware rate, and gravity convention.
4. Convert acceleration to g and gyro to rad/s exactly once.
5. Preserve source/session boundaries and real valid lengths.
6. Populate stored and effective source rates without granting bandwidth to interpolation.
7. Run the conditioning contract tests, including accel-only and multi-device cases.
8. Inspect rendered text and confirm it contains only device role and placement.
9. Confirm no activity, label, subject, dataset, or split field reaches either conditioning branch.
10. Update `docs/data/ACTIVE_DATASET_LEDGER.md` with the source-specific decisions.
