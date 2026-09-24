# Active Dataset Ledger

**Status:** active protocol ledger. Last verified against code: 2026-09-24 (local processed sessions last checked 2026-09-18).
This is the single source for active data-role, conversion, and fairness disclosures. It covers the
eight supervised-head sources, six sealed evaluation sources, and separately scoped prospective
sources. The three label-free sources are retained below as a historical inventory only; JEPA
pretraining is disconnected from the active recipe.

## Common processing contract

- Raw files are converted to per-recording `sessions/<id>/data.parquet`. Each row has a real
  `timestamp_sec`, a subject ID, canonical activity labels where applicable, and physical IMU
  channels. Session boundaries preserve raw recording/trial or real timestamp gaps; nothing joins
  separate recordings.
- The grid builder materializes native-rate 4, 8, and 16-second windows. It uses the native stream
  clock, anti-aliased polyphase resampling only where a converter documents it, and never uses
  harmonization in the active protocol. Native grids may retain an incomplete final window unless a
  source sets `full_windows_only`; masks carry the actual valid extent and models never treat padded
  samples as signal.
- Input layout is canonical `acc_x/y/z[, gyro_x/y/z]`. Acceleration is converted to **g** exactly
  once (m/s2 sources divided by 9.80665); gyro remains rad/s. Missing gyro is represented as absent
  modality with a mask, never as a claimed measurement. No per-session min-max normalization is
  allowed. Robust normalization statistics are fitted on the active train corpus only.
- Native labels are retained in `labels.json`; canonical labels are a controlled vocabulary used
  only for cross-source comparisons. Acquisition conditioning follows the versioned
  [schema-v2 contract](acquisition_conditioning.md): natural language contains
  device role and placement only, while modality, gravity, and effective source rate are exact
  structured model inputs. Stored rate is retained for provenance only. Evaluation candidate labels
  remain frozen native labels.
- Counts below are **processed session** counts. Duration is the sum of session samples divided by
  the recorded native rate; it is not a claim about raw archive duration. The median is per session.
  Rerun this inventory after rebuilding sessions before changing a protocol claim.

## Retired Label-free Encoder Pretraining

These sources remain structurally label-free (`__unlabeled__` only) and cannot enter the
supervised vocabulary, support bank, validation selection, or sealed evaluation. They total
**5,271.4 stream-hours** across **730 processed streams**. They are not loaded by current training.

| Dataset | Original schema and available subset | Local quantity | Conversion / preprocessing | Why retained |
|---|---|---:|---|---|
| `capture24_pretrain` | CAPTURE-24, dominant-wrist Axivity AX3 triaxial accelerometer at 100 Hz; diary/video labels deliberately unread | 151 people, 151 streams, 3,883.4 h; median 93,900.0 s | Raw `acc_x/y/z` in g with gravity; timestamp gaps split; no gyro is invented; full real windows only, float16 grid | Large free-living wrist motion scale without label exposure |
| `nymeria_xsens` | Nymeria Xsens MVN Link suit, 11 simultaneous placements selected from 17 trackers, accel + gyro at 240 Hz | 39 people, 440 streams, 113.5 h; median 932.0 s | Sensor-frame acceleration reconstructed/reoriented as documented by the converter, gravity restored, g conversion; gyro retained in rad/s; gaps split; full real windows only, float16 grid | Diverse limb/body dynamics and sensor placements |
| `extrasensory_pretrain` | ExtraSensory raw phone/watch accelerometer captures; only deployment-valid phone hand/pocket and watch streams, labels unread | 60 people, 139 streams, 1,274.4 h; median 14,360.0 s | Android m/s2, iPhone g, Pebble milli-g converted to g; each capture independently resampled to 50 Hz; capture/gap ID preserved; full real windows only, float16 grid | Consumer-device and free-living motion without supervised-label leakage |

## Supervised Support-Classifier Training

The active eight-source corpus totals **290.5 processed stream-hours** and **51,160 sessions**.
All source labels are preserved, then mapped to the training ontology only for episode construction.
No sealed or prospective source is in this roster.

| Dataset | Original schema | Local quantity / labels | Subset and preprocessing affecting fairness | Why retained |
|---|---|---:|---|---|
| `hhar` | Smartphone waist pouch, acc+gyro; released devices span 50–200 Hz | 9 people, 2,265 sessions, 26.4 h; median 18.6 s; 6 labels | Anti-aliased to 50 Hz on the real clock. Galaxy S+ is correctly accel-only; no synthetic gyro. | Only compatible phone-waist source for two sealed placements |
| `wisdm` | Phone/watch consumer streams, 20 Hz acc+gyro after conversion | 51 people, 2,012 sessions, 90.5 h; median 179.8 s; 18 labels | Converted to physical units; device/placement streams remain distinct; true source rate is recorded. | Large consumer-device pool and pocket/watch coverage |
| `kuhar` | Smartphone waist acc+gyro at 100 Hz, source linear acceleration (gravity removed) | 89 people, 2,452 sessions, 17.1 h; median 17.5 s; 18 labels | Native 100 Hz; gravity-removed status is explicit, so it is never falsely declared configuration-compatible with gravity-present streams. Backward clock seams are split upstream. | Largest subject pool and high label overlap with held-out waist/hip data |
| `harmes` | Wrist-oriented consumer IMU acc+gyro, 50 Hz | 51 people, 9,808 sessions, 54.6 h; median 23.1 s; 52 labels | Native rate/units, named recording boundaries, explicit wrist acquisition descriptors. | Broad daily-living vocabulary and genuine wrist coverage |
| `xrf_v2` | 50 Hz body IMUs plus 25 Hz AirPods; active streams are explicit sensor placements | 16 people, 32,610 sessions, 74.8 h; median 8.0 s; 30 labels | Each placement/device remains a separate stream; AirPods' gravity state is retained rather than harmonized away. | Cross-placement and earbud/device diversity |
| `dsads` | Five body IMUs, acc+gyro, 25 Hz | 8 people, 218 sessions, 12.7 h; median 300.0 s; 19 labels | Native 25 Hz direct converter; no generic phone/watch relabeling. | Supplies the low-rate anchor and body-motion variation |
| `forth_trace` | Five Shimmer IMUs, 51.2 Hz, basic activities plus explicit transitions | 14 people, 398 sessions, 3.7 h; median 18.0 s; 16 labels | Direct native grid; discontinuities stay session boundaries. | Transition vocabulary and bilateral placement structure |
| `realdisp` | Nine Xsens body IMUs, acc+gyro, 50 Hz under ideal/self/displaced placement regimes | 17 people, 1,397 sessions, 10.6 h; median 21.7 s; 33 labels | Placement regime stays encoded as configuration metadata; no rotation/displacement label leakage. | Controlled sensor-displacement evidence and exercise diversity |

## Historical Sealed Evaluation

The sealed roster is fixed: **MotionSense, RealWorld, Shoaib, InclusiveHAR, USC-HAD, UT-Complex**.
All scores are per dataset and per stream/window duration. Query/support episode manifests are
frozen before provider features are loaded. These data are not allowed in pretraining or supervised
head training.

| Dataset | Original schema | Local quantity / labels | Evaluation handling | Reason |
|---|---|---:|---|---|
| `motionsense` | iPhone 6s front pocket, Core Motion acc+gyro at 50 Hz | 24 people, 360 recordings, 7.8 h; median 59.0 s; 6 labels | Uses phone-front-pocket stream only; raw gravity/attitude extras are not substituted for the standard acc+gyro contract. | Canonical pocket HAR baseline |
| `realworld` | Simultaneous phone waist, forearm, thigh acc+gyro at 50 Hz | 15 people, 132 recordings, 18.6 h; median 620.1 s; 8 labels | Single-placement and declared simultaneous multi-device cells; identical time windows for every provider. | Strong placement/device-set stress test |
| `shoaib` | Four phone placements plus wrist-mounted phone proxy, acc+gyro at 50 Hz | 10 people, 70 recordings, 3.5 h; median 180.0 s; 7 labels | Right-pocket, left-pocket, belt and explicitly marked wrist-proxy streams; proxy is never described as a real watch. | Simultaneous placement and rate/modality controls |
| `inclusivehar` | Waist-mounted iPhone, acc+gyro at 50 Hz; able-bodied and disability participants | 20 people, 120 recordings, 2.2 h; median 64.8 s; 6 labels | Native waist stream, participant-disjoint manifests. | Inclusion-relevant domain variation |
| `usc_had` | Hip-mounted IMU, acc+gyro at 100 Hz | 14 people, 840 recordings, 7.8 h; median 30.0 s; 12 labels | Native 100 Hz hip stream; source activity names are frozen. | Higher-rate hip and directional locomotion vocabulary |
| `ut_complex` | Wrist-mounted phone, acc+gyro at 50 Hz | 10 people, 130 recordings, 6.5 h; median 180.0 s; 13 labels | Native wrist stream, explicitly phone-on-wrist rather than smartwatch. | Complex daily activities at a wrist placement |

## Separate Scenario / Prospective Sources

These sources are not folded into the sealed-six aggregate or the supervised training corpus.

| Dataset | Original schema | Local quantity / labels | Protocol and preprocessing | Reason |
|---|---|---:|---|---|
| `mmfit` | 21 workouts: two watches, right-pocket phone, and earbud, synchronized acc+gyro at 100 Hz; video/pose/heart-rate excluded | 555 device-recording sessions, 3.1 h; median 19.7 s; 10 exercise labels | Publication split only: train+validation workouts provide reference support; participant-disjoint test workouts provide queries. Workout ID is never presented as a person ID. Included only in new-domain and multi-device scenario tasks. | Genuine synchronized consumer multi-device evidence |
| `mobiact` | **Not locally prepared yet.** Official annotated MobiAct v2: trouser-pocket Android phone acc+gyro trials, ADLs and simulated falls | Counts, labels and durations intentionally unset until archive validation | `data.datasets.mobiact.setup` rejects MobiFall, validates official annotated CSVs, splits real gaps, anti-aliases to 50 Hz, reserves subject-disjoint reference/query partitions, and freezes duration-specific label panels with complete real windows only. Use only `sealed_eval --scope prospective` or `run_scenarios --include-prospective-mobiact`. | Prospective activity/fall domain expansion without rewriting historical means |

## Audit and Update Procedure

1. Before changing a data role, update the roster in `data/scripts/curate/deployment_policy.py` and
   this ledger in the same change.
2. After any converter rebuild, recompute session counts and durations from Parquet metadata, rebuild
   the affected native grids, refresh duplicate/implausible screens, and record the new date here.
3. A dataset must not move from prospective/scenario into a headline aggregate without a new frozen
   protocol version, an exposure audit for every evaluated checkpoint, and per-dataset reporting.
