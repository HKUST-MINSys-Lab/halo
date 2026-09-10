# Data heterogeneity - per-dataset reference

> **Application status, 2026-08-28:** the verified unit, placement, rate, gravity, and channel
> decisions below remain authoritative. The old train/eval role column describes encoder and HAR
> provenance, not the new application split. Use
> [`APPLICATION_DATASETS.md`](APPLICATION_DATASETS.md) for Tasks 1-3.

> This doc is the mechanics of how we normalize heterogeneity. It is not a contribution claim.
> Application measurements must nevertheless test whether session, remounting, and device variation
> change the representation enough to mimic a movement change. See
> [`MOTIVATION.md`](../design/MOTIVATION.md).

Every non-obvious thing we do to a dataset is recorded here, so we never have to re-derive *why*.
Three modules enforce these decisions in code; this doc is their prose rationale:

- **`data/scripts/curate/deployment_policy.py`** — device/placement/channel selection + gravity reconstruction
- **`data/scripts/curate/accel_units.py`** — accelerometer unit → g (runs *after* deployment_policy)
- **`data/scripts/assembly/baseline_view.py`** — harmonised/non-harmonised views (fixed windowing runs in `build_grids.py`)

Pipeline order (fixed): `raw session → deployment_policy → accel_units → windowing → baseline_view`.
`deployment_policy` owns **gravity**; `accel_units` owns **unit**. Nothing here fabricates data.

## The four heterogeneity axes we normalize (and how)

| Axis | Normalized to | Where |
|---|---|---|
| **Device / placement** | one stream per device: phone (pocket/waist/thigh), watch (wrist), or body-strapped `device` (lower-back, forearm, head-glasses, ear). The **strict** deployment view keeps phone only; the **harmonised** (all-wearable) view keeps phone + watch + `device`. | deployment_policy |
| **Channels** | `[acc_xyz(, gyro_xyz)]`, standard order; ≤ 6 | deployment_policy → baseline_view |
| **Gravity** | present (reconstructed for iOS); kuhar **and** xrf_v2's AirPods stay gravity-removed (never faked) | deployment_policy |
| **Unit** | accelerometer in **g** (|acc| ≈ 1 at rest); gyro untouched | accel_units |

Sampling-rate heterogeneity is **not** flattened in the corpus (it's a first-class comparison axis);
rate is recorded per dataset. `build_grids` materialises **three** regimes (`_ALIGNMENTS`):

- **`harmonised`** — resampled to **60 Hz**, 6-ch pad+mask, canonical labels. A compatibility export
  for fixed-rate, fixed-layout analyses; it is not HALO's training source.
- **`non_harmonised`** — native rate, native 3/6-ch, native labels. The raw eval/baseline source.
- **`native`** — **native rate** (no resample), 6-ch pad+mask, canonical labels. **HALO's source:** the
  filterbank tokenizer is rate-invariant, so HALO trains on the corpus's REAL native rates (20/50/100 Hz)
  plus the `rate`/`window_crop` augmentations — not a 60 Hz base with synthetic rate diversity.

> **The "Channels" column below = the dataset's REAL sensors.** The accelerometer is *always* present
> and *never* removed. Some datasets have no usable gyroscope — it is either physically absent
> (capture24/unimib_shar/harth are accelerometer-only devices) or unrecoverable — so they contribute
> 3 real channels. This is **not** a choice about the tensor: the **harmonised**/**native** views are
> *always* 6-channel `[acc_xyz, gyro_xyz]` (acc-only datasets get **zero-padded + masked** gyro slots),
> and the **non-harmonised** view keeps the native 3 or 6. Nothing is ever "taken out".

## Per-dataset table

| Dataset | Role | Device · placement | Native rate | Accel unit | Gravity | Channels | Special treatment (and why) |
|---|---|---|---|---|---|---|---|
| uci_har | train | phone · waist | 50 Hz | g | present | acc+gyro | Uses **`total_acc`** (gravity present, g), **not** `body_acc` (gravity-removed, ≈0.04 g). |
| hhar | train | phone · waist | 50-200 Hz source, anti-aliased to 50 Hz | m/s² | present | acc+gyro; Galaxy S+ acc only | Separate IMU and accelerometer-only streams preserve the release's low-rate phone class. |
| pamap2 | train | watch · wrist (hand IMU) | 100 Hz | m/s² | present | acc+gyro | Keep only the **wrist ±16 g** IMU; drop chest, ankle, the ±6 g accel, mag, temp, HR, and the invalid orientation quaternion. |
| wisdm | train | phone · right pocket **and** watch · dominant wrist (2 streams) | mixed source clocks, anti-aliased to 20 Hz | m/s² | present | acc+gyro | Acceleration and gyroscope are independently regularized from raw timestamps before joining. |
| kuhar | train | phone · waist | 100 Hz | m/s² | **removed** | acc+gyro | Linear acceleration — DC ≈ 0 at rest by design. **Never fabricate gravity.** Gravity-dependent baselines must skip it. |
| unimib_shar | train | phone · trouser pocket | 50 Hz | m/s² | present | acc only | Distributed overlapping windows are grouped by the release's subject/activity/trial execution id. |
| hapt | excluded duplicate | phone · waist | 50 Hz | g | present | acc+gyro | UCI-HAR re-release from the same 30 subjects/recordings; excluded from Phase A to prevent near-duplicate leakage. |
| mhealth | train | watch · wrist (arm IMU) | 50 Hz | m/s² | present | acc+gyro | Right-lower-arm IMU (co-located acc+gyro). Gyro is somewhat sample-and-hold but **real**, so it is kept as a 6-ch stream. Drop chest, ankle, ECG, mag. |
| capture24 | train | watch · wrist | 100 Hz | g | present | acc only | Free-living Axivity, accelerometer-only. |
| sp_sw_har | train | phone · front pocket **and** watch · wrist (2 streams) | ~102.5 → 100 Hz | g | present | acc+gyro | Paired Timed-Up-and-Go. Per-row labels → **fixed 1.0 s windows** (`turning` is ~1 s; a 6 s window would discard it), `pre_windowed`. Resampled on the real `timestamp` (true ~102.5 Hz, dup timestamps + gaps), not a synthetic clock. |
| nfi_fared | train | body-strapped **lower back** + **dominant forearm** (2 streams) | 100 Hz | g | present | acc+gyro | Forensic activities (incl. transport → `vehicle`, punch/kick/drag as new labels). Gyro **deg/s → rad/s**. SHA-256 dedupe of byte-identical source CSVs (pp10 exp2==exp3). Placement is **forearm** per the Hi-OSCAR paper, not wrist. |
| harmes | train | watch · right wrist | 50 Hz | m/s² | present | acc+gyro | Kitchen/bath ADLs plus open-vocabulary visit-4 activities. **Right wrist only** — the left Puck.js gyro saturates the int16 rail with an undocumented full-scale (unrecoverable), so we do not ship it. Both typed start/end rows and comment-form start/end markers are parsed; corrected a **+1 h clock offset** on 39/71 recordings (DST bug). |
| xrf_v2 | train | 6 streams: L/R **wrist**, L/R **pocket phone**, **head glasses**, **AirPods ear** | 50 Hz (AirPods 25→50) | g | present (**AirPods removed**) | acc+gyro | 16 volunteers, 30 indoor ADLs (arXiv 2501.19034). Device→placement order read from the h5's own `device_order`. Gyro **deg/s → rad/s**. AirPods = iOS **user acceleration (gravity removed)** + gyro; upsampled to 50 Hz. Glasses/ear placements exist nowhere else in the corpus. |
| dsads | expanded train · stress placement | chest, L/R wrist, L/R knee | 25 Hz | m/s² → g | present | acc+gyro | Five simultaneous Xsens units. The distributed 5 s segments are rejoined only after continuity checks; discontinuous joins become hard session boundaries. |
| forth_trace | expanded train | L/R wrist, torso, right thigh, left ankle | 51.2 Hz | m/s² → g | present | acc+gyro | Simultaneous Shimmer nodes; gyro deg/s → rad/s. One participant with irreconcilable annotation tracks is excluded and long node dropouts are hard boundaries. |
| opportunity | expanded train · stress placement | back + bilateral upper/lower arms | 30 Hz | milli-g → g | present | acc+gyro | Simultaneous Xsens units; gyro milli-rad/s → rad/s. Uses the locomotion track because the gesture instances are shorter than the 6 s Phase-A window. |
| realdisp | expanded train | back, bilateral upper/lower arms, thighs, calves | 50 Hz | m/s² → g | present | acc+gyro | Nine simultaneous Xsens units under ideal, self, and induced-displacement placement regimes. Gyro is already rad/s. |
| mmfit | expanded train | L/R wrist, right pocket, left ear | mixed 85–212 Hz → 100 Hz | g | present | acc+gyro | Shared-clock multi-device exercise recordings. Converter aligns devices on wall time and cuts gaps before the common 100 Hz grid. |
| phytmo | expanded train | bilateral arms/forearms or thighs/shins | 100 Hz | g | present | acc+gyro | Correct and deliberately incorrect rehabilitation exercise executions; gyro deg/s → rad/s. Upper- and lower-limb trials have distinct stream sets. |
| kneepad | optional Phase-A stress study | eight thigh/calf muscle sites | 148.15 Hz | g | present | acc+gyro | Clinical knee-pathology exercises; gyro deg/s → rad/s. Only 4.7% of trials reach an honest 6 s window, so it is excluded from the default phone/watch-oriented corpus. |
| extrasensory | optional Phase-A scale | phone · pocket/hand **and** watch · wrist (3 streams) | phone real clock (measured p01 30.3 / p50 34.1 / p99 119.2 Hz; paper's 40 Hz is nominal), watch 25 Hz → stored 50 Hz | mixed → g | present | acc only | Public free-living examples. Android m/s² vs iPhone g comes from the authors' subject/platform split; Pebble milli-g is converted to g. Each raw example is resampled independently and truncated to complete 6 s blocks. Phone data requires exactly one explicit pocket/hand placement; bag/table/unknown are pruned. Ambiguous multi-activity labels are dropped; a single stair direction supersedes its co-occurring generic walking tag. **The Pebble re-emits stale buffers**: 3.3% of wrist windows are byte-identical repeats (one group is 178 recordings over 10,620 s; one subject is 87.7% duplicates), and 36.9% of duplicate pairs carry contradictory labels. Removed by `scan_duplicates`, not by the converter — the defect is upstream. |
| nhanes | optional label-free JEPA scale | watch · non-dominant wrist | 80 Hz | g | present | acc only | Public PAX80_G participant archives. No activity diary exists, so windows use reserved `__unlabeled__` and feed only label-free JEPA; they are excluded from labelled application-task manifests and evaluation. Published QC intervals, non-finite rows, and clock gaps are hard boundaries. Fetching is bounded and deterministic; full-corpus download is intentionally unavailable. **Mostly stillness**: 43.0% of windows are below 0.003 g max-axis std and 13.3% are byte-identical motionless repeats (dropped by `scan_duplicates`); the released QC covers sensor malfunction only, so no non-wear detector is applied. |
| hmog | optional Phase-A scale | phone · hand | 100 Hz | m/s² → g | present | acc+gyro | Samsung Galaxy S4 used for reading, writing, and map navigation while sitting/walking. `Gesture_scenario` is checked against the independent odd/even `TaskID` condition before mapping to `sitting`/`walking`. Accelerometer and gyroscope are synchronized on monotonic event time; gaps over 250 ms are hard boundaries and each continuous block is truncated to complete 6 s windows. One official archive-name typo (`207969.zip` contains unique internal subject `207696`) is recorded explicitly. Materialized: 191,535 windows / 319.23 h; no rail violations or byte-identical duplicate windows. |
| motionsense | eval | phone · front pocket | 50 Hz | iOS → g | present | acc+gyro | iOS: total accel = **userAcceleration + gravity** (both g), reconstructed in deployment_policy; attitude is QA-only. |
| realworld | eval | phone · waist | 50 Hz | m/s² | present | acc(+gyro) | Gyro retained only when the converted waist stream has a complete finite triad. |
| mobiact | eval | phone · trouser pocket | 50 Hz | m/s² | present | acc+gyro | — |
| shoaib | eval | phone · right pocket (primary) | 50 Hz | m/s² | present | acc+gyro | Left-pocket / belt / wrist-proxy kept as **diagnostic** streams (not in the primary score). |
| inclusivehar | eval | phone · waist | 50 Hz | iOS → g | present | acc+gyro | iOS reconstruction as motionsense; ability-stratified cohort. |
| monipar | eval | watch · wrist | 50 Hz | m/s² → g | present | acc only | Parkinson's patients and controls repeat exercises across weekly sessions; primary across-session enrollment testbed. |
| spar | eval | Apple Watch · L/R wrist | 50 Hz | g | present | acc+gyro | Seven shoulder physiotherapy exercises. Left and right streams are separate primary evaluation cells. |
| upper_limb_use | eval | bilateral wrist bands | 50 Hz | g | present | acc+gyro | Controls and hemiparetic patients performing functional ADLs; patient streams distinguish more- and less-affected arms without inventing anatomical side. |
| usc_had | eval | device · front-right hip | 100 Hz | g | present | acc+gyro | MotionNode gyro is converted from deg/s to rad/s in the converter. Harmonised/non-harmonised grids were rebuilt after this unit fix. |
| harth | stress | back / thigh | 50 Hz | g | present | acc only | **Non-deployment** placement — retained only as a placement stress test, never in the primary score. |

## Excluded datasets (kept in `legacy_code`, not in this repo)

| Dataset | Why excluded |
|---|---|
| recgym | Per-axis min-max [0,1] normalization destroyed physical scale and gravity — non-recoverable. |
| paaws | Release format and phone/watch-compatible locations were verified from the official parser and sample file, but this machine cannot access the released collection (HTTP 403). Do not add a converter until actual release bytes can be fetched and smoke-tested. |

## Invariants (asserted by tests)

- Every dataset in `deployment_policy.STREAM_SPECS` is classified in **exactly one** of
  `accel_units.ACC_UNIT_G` / `ACC_UNIT_MS2` (a new dataset cannot land without a unit decision).
- After the pipeline, a still window reads |acc| approximately 1 g for gravity-present datasets and
  approximately 0 for the declared gravity-removed streams (KU-HAR and XRF V2 AirPods).
- Gyroscope channels are never scaled by `accel_units`.
