# Plan: correct the evaluation, and evaluate at 4 s / 8 s / 16 s windows

**Written 2026-09-13.** Implementation brief for another agent. "Implement" = **build + tests +
short smoke**. Do **not** run sealed evaluation and do **not** launch long training without Alex's
explicit go.

> **Implementation status (2026-09-13): complete.** The versioned duration grids, immutable
> source/episode fingerprints, 4/8/16-second evaluator, released-model duration adapters,
> RealWorld/Shoaib composite cells, and HALO training/evaluation device plumbing are implemented.
> No sealed model scores were produced as part of this build.
>
> **Verification:** the focused duration/device/data suite passes (127 tests), as does the sealed
> manifest, deployment-policy, grid-build, and released-backbone suite (79 tests). All 39 declared
> cells (33 placement cells and 6 composites) load with quality screening at 4/8/16 seconds;
> k=0 and k=1 manifests cover all 186,420 retained rows. Short released-checkpoint probes passed
> for HARNet, LiMU-BERT-X, UniMTS, and NormWear on real composite data. A three-step CUDA HALO
> trainer smoke sampled multi-device rows, produced finite nonzero encoder/head gradients, wrote a
> checkpoint, and that checkpoint encoded a four-device 16-second sealed slice successfully.

Four pieces. **Order is load-bearing:** C1 (grid path schema) must land before B, or the placement
streams get built twice. C0 (probing) must precede C2 (design), because two earlier drafts of this
plan guessed wrong about what the baselines accept.

| order | piece | what | touches sealed data? |
|---:|---|---|---|
| 1 | A | 4 s patch resolution in the filterbank | no |
| 2 | C0 | probe each baseline's real length tolerance | no |
| 3 | C1 | grid paths encode window length | schema only |
| 4 | B | unlock multi-placement streams (realworld, shoaib) | new sealed streams |
| 5 | C2–C5 | the fair multi-duration evaluation (F1–F5) | yes — needs a go |
| 6 | D1–D4 | multi-device for the baselines (F6) | yes — needs a go |
| 7 | D5a–D5b | multi-device for HALO | yes — confirm before starting |
| — | D5c | cross-device trunk question | **open question, do not build** |

Everything through step 3 can be done without any sealed-data contact. Steps 5–7 need Alex's go.

---

## A. Add a 4-second patch resolution

Extend the fixed multiresolution filterbank ladder from `0.5 / 1 / 2` to `0.5 / 1 / 2 / 4` seconds.

* The only hard blocker is the `N.max() <= self.S` guard in
  `PhysicalFilterbankTokenizer._prep_rate_len` (`model/tokenizer/filterbank.py`), which raises
  `patch_len_samples max {n} exceeds dft_size S=512`.
* **Do not raise `dft_size`.** Decimate to a bounded analysis rate first: at a 40 Hz analysis rate a
  4 s patch is 160 samples and an 8 s patch 320, so `DFT_SIZE` stays 512 and the ladder costs
  ~0.13x the current rDFT work. Decimation must follow `f_max` (15 Hz), never the acquisition rate.
* The multiresolution collate sorts patches by centre time, so 4 s patches **interleave** with the
  others rather than forming a contiguous block. Do not assume contiguity anywhere.
* Update `future_patch_durations`, the resolution ids, and the RoPE minimum period.
* **8 s patches are deliberately out of scope here.** They become viable once C adopts 8 s and 16 s
  windows — raise separately. At N=8 an 8 s patch fills the entire window (one patch, no context at
  that scale), so it is only clearly useful at N=16.

**Tests:** rate invariance across 20/25/50/100/240 Hz at the new resolution; the resolution flag is
correct for a 4 s patch; `DFT_SIZE` unchanged; with 4 s disabled the 0.5/1/2 outputs are
bit-identical to today.

---

## B. Unlock the multi-placement streams in two sealed datasets

Both sources hold simultaneous multi-placement recordings **already on disk**; we discarded them at
conversion. This is the only route to testing placement heterogeneity on sealed data, and it
directly probes UniMTS's published placement-generalisation claim.

### B1. realworld

Positions verified by reading the downloaded `acc_*_csv.zip` members:
**chest, forearm, head, shin, thigh, upperarm, waist**.

> **realworld has no `wrist` position.** The requested set was forearm / thigh / wrist / waist.
> **`forearm` is the wrist/watch-proxy placement** here and is already in the list, so the buildable
> set is **forearm, thigh, waist** (3). If a fourth is wanted, `shin` or `chest` are available;
> `upperarm` is presumably excluded, as it was for shoaib. Build the three and ask before adding.

`data/datasets/realworld/convert.py` hardcodes `TARGET_POSITION = "waist"` and selects the
per-position CSV member by that name. Generalise it to emit one session set per requested position.
Keep the existing waist stream **byte-identical**.

### B2. shoaib

The source CSV holds all five positions in one file (70 columns = 5 x 14; order: left_pocket,
right_pocket, wrist, upper_arm, belt). Build **left_pocket, right_pocket, wrist, belt** —
**exclude upper_arm** (Alex's decision).

`data/scripts/curate/deployment_policy.py` **already declares StreamSpecs** for
`shoaib/phone_left_pocket`, `shoaib/phone_belt` and `shoaib/watch_wrist_proxy`. Use those ids; do
not invent new ones. Shoaib's "wrist" is a **phone strapped to the wrist**, not a smartwatch — name
it a proxy, exactly as `ut_complex/watch_wrist` already is.

### B3. Consequences

* These streams have never been scored, so they remain sealed-clean.
* Adding them **changes the sealed roster and every manifest fingerprint**. Results do not pool with
  the existing ladder. Report as a separate table and say so.
* Register each stream in `deployment_policy.py` with honest placement and device text (the encoder
  conditions on it).
* Re-run the per-source quality screen and gravity check on every new stream. A new placement is not
  automatically valid.

---

## C. The fair multi-duration evaluation (4 s / 8 s / 16 s)

**The question every model is asked, identically:** *given this N-second recording, name the
activity*, with support also N-second recordings. Repeat the whole evaluation at **N = 4, 8, 16**.

**Why not 3 s:** harnet5's trunk *crashes* on 90 samples
(`Calculated padded input size per channel: (4). Kernel size: (5)`). 4 s is the shortest window it
can physically accept.

### C0. FIRST — probe each baseline's real length tolerance. Do not assume.

For each baseline, instantiate the released trunk and run its **representation layer** on inputs of
4, 8 and 16 seconds at the model's native rate. Record the output shape at each length. Chunking
applies **only** where the architecture genuinely refuses a length.

The harnet5 probe is done and is the template (`Resnet.feature_extractor`, 3 channels, 30 Hz):

| input | samples | trunk output | usable as 512-d |
|---|---:|---|---|
| 3 s | 90 | **RuntimeError** | no |
| 4 s | 120 | `(1, 512, 1)` | **yes, natively** |
| 5 s | 150 (native) | `(1, 512, 1)` | yes |
| 6 s | 180 | `(1, 512, 1)` | **yes, natively** |
| 8 s | 240 | `(1, 512, 2)` | yes, after mean over the time axis |
| 16 s | 480 | `(1, 512, 4)` | yes, after mean over the time axis |

**Conclusion for harnet: it needs no chunking at any of 4/8/16, and the current centre-crop is a
pure defect.** Replace `f.flatten(1)` with a **mean over the trailing time axis** — the
global-average-pool the architecture omits — and harnet accepts any window from 4 s upward in a
single pass over the continuous signal. That is strictly better than chunking, because the
convolutions are never cut at a chunk boundary.

Run the identical probe for:
* **NormWear** — its adapter hardcodes 390 samples (6 s @ 65 Hz), but a ricker-CWT front end is
  length-flexible in principle. Weights are not materialised locally; fetch first.
* **UniMTS** — hardcodes 200 samples (10 s @ 20 Hz) with wrap-padding. At N=8 and N=16 it may be
  able to receive its full designed 10 s for the first time, which is a fairness improvement in its
  favour.
* **LiMU-BERT-X** — expected to refuse: `pos_embed` is `nn.Embedding(20, HIDDEN)` and the forward
  does `torch.arange(x.shape[1])`, so >20 timesteps indexes out of range. Confirm, don't assume.

Write the probe as a committed test (`tests/test_baseline_length_tolerance.py`) that records each
model's accepted lengths, so the table cannot silently rot.

### C1. Grid paths must encode window length

`build_grids._save` writes to `grids/<alignment>/<stream_id>/` and stores `window_seconds` only
*inside* `meta.json`. Building 4/8/16 s grids of one stream would **silently overwrite each other**.

Add the window length to the path — `grids/<alignment>/<stream_id>/w<seconds>/` — and update every
reader: `grid_io`, `load_eval_stream`, the pretraining corpus loader, and the feature caches. Watch
the known `grid_io` sort-key gotcha. Ship a migration that leaves existing 6 s grids readable.

### C2. The five fairness guarantees

This is the substance. Each guarantee gets a test.

**F1 — identical window boundaries, chosen model-independently.**
Window boundaries are computed **once per (stream, N)** from the grid, before any model is loaded,
and stored in the manifest. Every model reads the same `[t0, t0 + N]` slice of the same execution.
No adapter may re-window, shift, or re-centre. *Test:* for a fixed query index, hash the raw
pre-resample slice handed to each model and assert all hashes are equal.

**F2 — resample from native, once, per model.**
Each model resamples from the **same native-rate source array**, never from another model's
resampled copy, and never twice. Chained resampling (e.g. 50 → 20 → 30 Hz) compounds filter error
and would silently advantage whoever sits first in the chain. *Test:* assert each adapter's
resample is called with `stream.rate_hz` as the source rate.

**F3 — same evidence, native consumption.**
Give every model the same N seconds. Then:
1. If C0 says the model accepts length N natively, **feed it in one pass** and mean-pool any
   residual time axis in the representation. This is the preferred path.
2. Only where the architecture genuinely refuses, split into `ceil(N / native_seconds)`
   native-length chunks, padding **only the final partial chunk** with that model's own documented
   rule (harnet wrap-pads, UniMTS wrap-pads, NormWear edge-pads, LiMU-BERT edge-pads).
3. Encode at each model's **published representation layer** — the same layer `window_features`
   already returns (harnet's 512-d `feature_extractor` output before `EvaClassifier`; LiMU-BERT's
   clip embedding; NormWear's embedding; UniMTS's sensor encoder before text scoring). Do not invent
   a new tap point.

**F4 — one pooling rule, for everyone including HALO.**
Use the **length-weighted mean already implemented in `baselines/limubert_x/adapter.py`**: full
chunks weight 1, the tail weight `remainder / native_length`, divided by total weight; then
L2-normalise. Lift it into `baselines/base.py` as a shared helper and call it from every adapter.
It is parameter-free, author-neutral, and already in the repo, so we are not inventing a rule and
imposing it. **Scope:** the helper governs *fan-out* only — combining per-chunk or per-device
vectors that our driver produced because a model could not consume the whole input at once. A
model's **internal** pooling (HALO's `RecordingAttentionPool`, NormWear's MSiTF aggregator,
harnet's mean over its conv time axis) is part of that model and is not replaced. *Test:* the helper
is the only fan-out pooling path any adapter calls.

**F5 — padding is disclosed, never hidden.**
Where a model's native window exceeds N, part of its input is repeated signal — its own declared
short-window behaviour, not something we imposed. Mark those cells in the output table and state it
in the results text. **Do not mark them N/A**; that would empty the N=4 column. *Test:* the result
record carries a `padded: bool` and a `padded_fraction: float` field.

### C3. Fix the ambiguous contract field

`InputContract.window_sec` currently means "native chunk length" but reads as "the extent this model
is allowed to see" — that ambiguity is what produced two wrong drafts of this plan and, in harnet's
case, a real defect. Rename it to `native_window_sec`, document it as *the model's native input
length, not the evidence budget*, and add `max_window_sec: Optional[float]` populated from the C0
probe.

### C4. Coverage and yield — measured, and what it costs

Estimated from the current 6 s grids (assumes non-overlapping windows; **report exact retained
counts per cell rather than these numbers**):

| dataset | executions | median | ≥4 s | ≥8 s | ≥16 s | windows @4 s | @8 s | @16 s | (@6 s today) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| motionsense | 360 | 54 s | 100% | 99% | 96% | 6709 | 3270 | 1550 | 4534 |
| realworld | 132 | 618 s | 100% | 100% | 100% | 16824 | 8378 | 4163 | 11237 |
| shoaib | 70 | 180 s | 100% | 100% | 100% | 3150 | 1540 | 770 | 2100 |
| usc_had | 840 | 30 s | 100% | **92%** | **83%** | 6316 | 2956 | 1296 | 4350 |
| inclusivehar | 120 | 60 s | 100% | 100% | 100% | 1859 | 893 | 415 | 1260 |
| ut_complex | 130 | 180 s | 100% | 100% | 100% | 5850 | 2860 | 1430 | 3900 |

Consequences to handle explicitly:

* **The high-k tail thins at 16 s.** An episode needs `k` support windows *per candidate* from
  executions disjoint from the query. usc_had has ~12 classes and only ~1296 windows at 16 s, so
  `k = 64` and `k = 128` will frequently be unformable; inclusivehar at 415 windows is thinner
  still. The manifest builder already drops queries that cannot form an honest episode — keep that
  behaviour, and **report retained query counts per (dataset, N, k) cell**. Do not pad, do not relax
  execution-disjointness, and do not silently shrink the candidate roster.
* **Compare within a duration, never across.** Query counts differ by N, so confidence intervals
  differ. State this in every table caption.
* The N seconds must be **contiguous within a single execution**. No stitching across executions.
* Execution-disjointness between query and support is unchanged and non-negotiable.
* Keep `k = 1, 2, 4, 8, 16, 32, 64, 128`; support units are N-second windows, so k keeps its
  meaning.

### C5. What this costs — confirm with Alex before rebuilding grids

Moving off 6 s **rebuilds every evaluation grid and changes every manifest fingerprint**. The
four-arm JEPA ladder in `docs/journal/2026-09-13-jepa-value-measured.md` was run at 6 s and **will
not be comparable** to anything produced at 4/8/16 s. Either re-run those arms under the new
protocol, or keep the 6 s table as a clearly-labelled historical result. This is the single largest
cost of the change.

---

## D. Multiple concurrent devices

Piece B creates the first sealed cells where one subject wears several devices at the same instant
(realworld: forearm + thigh + waist; shoaib: left_pocket + right_pocket + wrist + belt). All
placement streams share **100% of their event ids**, so alignment is exact and free.

Nothing in the current code consumes that. This section is the complete build: the evaluation data
structure (D1), the plumbing that must change with it (D2), what each baseline can do (D3), the
fairness guarantee (D4), HALO evaluation (D5), and **HALO training (D6)**, which is the largest
single piece and was under-specified in the first draft of this plan.

### D1. `MultiDeviceEvalStream`

`load_eval_stream(dataset, stream_id, ...)` returns one `EvalStream` whose `windows` are
`(n_windows, T, C)` for a single placement. Add a composite:

```
MultiDeviceEvalStream:
    dataset:     str
    cell_id:     str                   # e.g. "realworld/forearm+thigh+waist" — see D2.1
    devices:     list[EvalStream]      # one per placement, in a fixed declared order
    device_ids:  list[str]             # stream ids, e.g. ["forearm","thigh","waist"]
    event_ids:   np.ndarray            # verified identical across members
    quality_screen / n_quality_excluded  # computed on the COMPOSITE — see D2.3
```

Construction rules, each enforced with a raise, never a silent repair:

* **Assert event-id identity** across every member, elementwise and in order. Do not
  intersect-and-reindex: differing ids mean the grids came from different segmentations and the
  "same instant" premise is false.
* Members may differ in native rate and channel set. Do **not** harmonise at load time; every
  adapter resamples from native (guarantee F2).
* Device order is fixed by the declared roster, not by directory listing order.
* Window index `i` refers to the same physical interval in every member.

### D2. Evaluation plumbing that must change with it

These are the items a composite breaks. **D2.2 is a correctness bug**, not a nicety.

**D2.1 — cell identity.** Register multi-device cells explicitly in `deployment_policy.py` with a
stable `cell_id`. **Declared composition policy:** for each multi-placement dataset, evaluate every
single placement on its own (those are ordinary `EvalStream` cells) **plus exactly one all-device
composite**. Do not enumerate all subsets — with 3 and 4 placements that is 4 and 11 extra cells
whose only use would be post-hoc selection.

**D2.2 — the feature cache key must include the device set.**
`sealed_eval._cache_key` (line 367) is currently
`f"{name}|{stream.dataset}|{stream.stream}|{stream.alignment}|{stream.window_seconds:g}|{fingerprint}"`.
It already includes `window_seconds`, so piece C is safe. It has **no device component**, so a
composite cell would silently collide with a single-placement cache and read the wrong features.
Add the ordered device list to the key. *Test:* a composite and each of its members produce
distinct cache keys.

**D2.3 — quality screen is computed once, on the composite.**
`sealed_eval` raises unless `stream.quality_screen == "applied"`. For a composite, run the screen
per member and then **drop a window index if it fails in any member**, so every model sees the
identical retained set (this is what F1/F6 require). Record `quality_excluded` per member *and* for
the composite. Do not let each model apply its own screen.

**D2.4 — `build_manifest` accepts the composite.** It uses `stream.n_windows`,
`stream.execution_ids` and `_aligned_labels(stream)`. All three are shared across members by
construction, but the function must accept the new type and record the device set in the manifest
(see F6). Execution-disjointness applies across the whole composite: a support episode may not draw
from the query's execution in **any** member.

**D2.5 — `_load_or_encode` and per-device fan-out.** Type it for either stream kind. **Put the
per-device loop in `baselines/base.py`, not in four adapters.** Add
`supports_multi_device: bool = False` to `BaselineAdapter`; the driver feeds the composite directly
to adapters that declare `True`, and otherwise loops `window_features` over members and pools with
the shared helper (D4). One code path, one pooling rule.

**D2.6 — `is_incompatible` across a composite.** harnet declares itself N/A on gravity-removed
datasets. If any member of a composite triggers a model's incompatibility rule, **the whole cell is
that model's disclosed N/A** — do not silently drop the offending device, which would give that
model a different evidence set from everyone else.

**D2.7 — the k=0 training-bank bridge.** `TRAINING_BANK_ZERO_SHOT` covers `halo`, `harnet`,
`limubert_x`, comparing the query against a bank of training-corpus recordings. A multi-device query
vector against a single-device bank is a representation mismatch. **Either** build a matching
multi-device bank from the multi-device training sources, **or** declare k=0 N/A for composite
cells. Pick one, record it; do not compare across the mismatch.

### D3. What each baseline can natively do

| model | multi-device | mechanism |
|---|---|---|
| **UniMTS** | **native — its designed use** | ST-GCN over a 22-node SMPL skeleton. Each IMU is written into one joint's 3 accel channels, other joints zero-filled; trained with random-joint masking, so partial skeletons are in-distribution. Several devices = several populated joints. |
| **NormWear** | **native** | Channel-independent ViT over per-channel CWT scalograms with a query-conditioned MSiTF aggregator accepting any channel count. Devices = more channels. |
| **harnet5** | no (3-ch accel, one device) | per-device encode, pooled by the driver (D2.5 + D4) |
| **LiMU-BERT-X** | no (6 ch, one device) | per-device encode, pooled by the driver |
| **HALO** | after D5 | native, per-sensor with device identity |

**UniMTS.** `baselines/unimts/adapter.py` has `_SIDE_PLACEMENT_JOINTS` (stream-id keyword match),
`JOINT_BY_DS` (already containing `"shoaib": 5,  # multi-position stream -> R-hip default`) and
`DEFAULT_JOINT = 0`. Resolve a joint **per device** and populate each. Add explicit assignments for
every new placement (forearm, thigh, waist, left_pocket, right_pocket, wrist, belt).
**Raise on a joint collision** — two devices overwriting one joint would be an invisible fairness
violation. Note in the results that UniMTS is **accel-only**, so multi-device means more joints of
accelerometer, gyroscope discarded, at every device count.

**NormWear.** Extend the `stream.mask` phantom-channel drop to the union across members, in the
declared device order so the channel layout is deterministic. Record `n_channels_used`.
**Disclose the structural limitation:** it linear-detrends each channel (removing gravity) and has
no notion of *which* channel is where, so it receives the extra data but not the geometry. That is a
property of the published model, not a handicap we imposed.

**harnet / LiMU-BERT.** Per-device encode then pool. This extends them past their published
single-device contract and **must be labelled in the table**.

### D4. Fairness guarantee F6, and the shared pooling helper

**F6 — identical device set and time slice, fixed in the manifest.**
The device set is written into the manifest **before any model is loaded**, with the window
boundaries. Every model receives the same devices over the same `[t0, t0+N]` interval. No model may
choose, drop, re-order or substitute devices.
*Test:* for a fixed query index, hash each `(device_id, raw pre-resample slice)` pair per model and
assert the multiset is identical across models.

* **Never fabricate a device.** If a placement has no valid window at an index, that index is
  already excluded by D2.3; the composite never runs short.
* **Label each model's mode** in the table — `native` (UniMTS, NormWear, HALO after D5) or
  `per-device-pooled` (harnet, LiMU-BERT).
* **Report single-device rows alongside composite rows.** The per-model multi-device delta is the
  result; a headline that mixes them answers nothing.

**The pooling helper** is the same length-weighted mean lifted into `baselines/base.py` for F4:
per-device vectors all weight 1, summed, divided by count, L2-normalised. One helper for time chunks
and devices both. Do not write a second pooling function.

### D5. HALO evaluation: data path and hierarchical pooling

**The encoder is already a sensor-set model.** `SensorFold` (`model/tokenizer/sensor_tokens.py:39`)
takes `sensor_id (B,C)` plus `n_sensors`, folds each 3-axis group into a sensor token and returns
`(B,P,S,d)` with a mask, for **arbitrary S**. The trunk attends over sensor tokens as a masked set,
per-sensor text conditioning is factored, and gravity/DC and the polarization features are already
per-triad — the correct granularity, since each device has its own orientation.

**D5a — data path.** The blocker is entirely here:
`CHANNELS = ("acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z")`
(`training/tokenizer/pretrain_data.py:97`) is a fixed 6-slot constant, and
`sensor_id = [accel_id if c.startswith("acc") else gyro_id for c in CHANNELS]` (:375) makes
**"sensor" mean modality, not device**.

1. Variable-length channel layout: `D` devices x up to 2 modalities x 3 axes, in a canonical order
   so `SensorFold`'s fixed-xyz assumption holds.
2. `sensor_id` indexes **(device, modality)** pairs. Two devices with accel+gyro is `S=4, C=12`.
3. **Add `device_id (B,S)`** mapping each sensor slot to its device index. This does not exist
   today and D5b needs it.
4. **`training/support_classifier/encoding.py` must pass `device_id` through** — it currently
   forwards `sensor_id`, `channel_mask`, `sensor_texts`, `gravity_state` and friends, and has no
   device concept. Extend `encode_batch` and the encoder signature together.
5. Per-sensor descriptor text for every device (placement + device type). The mechanism exists; it
   needs the entries. This is what lets HALO tell wrist from thigh — the thing NormWear
   structurally cannot do.
6. **Per-device gravity state.** `gravity_state` is currently per window; each device has its own
   orientation and its own gravity-removed-ness, so it becomes per device. KU-HAR-style
   gravity-removed members must gate their own `vert`/`spin` features, not the whole window's.
7. `MAX_BATCH_TOKENS = 16_384` and `batch_under_token_budget` must account for `S` scaling with
   device count, or multi-device batches silently shrink.

**D5b — hierarchical parameter-free pooling. DECIDED: option 3 — for the mean-pooled path.**

**Correction to the first draft.** HALO has two recording-pooling paths, and the trainer always uses
the learned one: `train.py:1224` and `encoding.py:86/116` set `learnable_recording_pool=True`
**unconditionally**, for `--classifier neighbors` and `token_mixer` alike. So every *trained* arm —
including the differentiable-neighbours arm — already pools with `RecordingAttentionPool`, and
`docs/design/JEPA_REPRESENTATION_EVALUATION.md` counts that pool as part of the adapted encoder.
Only the **frozen** arms (random frozen, JEPA frozen) and any checkpoint without a learned pool use
the parameter-free mean at `encoder.py:595`. Option 3 applies to **that** path.

For the trained path nothing pooling-related changes: `RecordingAttentionPool` flattens
`(B,P,S,d)` to `(B,P*S,d)` and attends over every patch x sensor token, so it can already weight
devices — the tokens carry device identity through per-sensor text conditioning. The D5a data-path
work is all it needs.

`encoder.py:595` currently pools the mean path's stage 1 as a flat masked mean over the sensor axis,
weighted only by validity:

```python
per_patch = (h * weights.unsqueeze(-1)).sum(dim=2) / denom     # (B,P,d)
```

With several devices this weights every (device, modality) slot equally, so a device carrying accel
*and* gyro counts twice as much as an accel-only device purely because of its instrumentation.
Replace stage 1 with a **two-level mean**, using `device_id`:

1. **Within device:** masked mean over that device's sensor slots -> one vector per device per patch.
2. **Across devices:** masked mean over per-device vectors -> `(B,P,d)`.

Stage 2 (per-resolution, duration-weighted patch pooling) is unchanged.

Required properties, each a test:

* **Completely parameter-free.** The frozen arms and the mean-pooled path gain no parameters. (The
  trained neighbours arm's pool is already learned and already counted as encoder; that does not
  change.)
* **Bit-identical for every single-device recording.** With one device, a within-device mean over its
  slots followed by a one-element across-device mean is arithmetically today's flat mean. This is the
  load-bearing regression test: no previously recorded result may move.
* **Each device counts once**, whatever its modality count.
* Per-device vectors available as a diagnostic at no extra cost.

The readout itself (`training/support_classifier/neighbors.py`) needs **no change** — it consumes
`(B,D)` and `(B,K,D)` and is agnostic to how they were pooled. Everything here is upstream of it.

### D6. HALO multi-device TRAINING

Evaluation alone is not enough: a model trained only on single-device windows meets composite cells
out of distribution. This piece was missing from the first draft.

**D6a — composite training samples.** `training/support_classifier/corpus.py` and `sampling.py`
draw windows from one stream. Add the training-side analogue of D1: for a source with aligned
placement streams, a sample may bind several streams at the same event id. Reuse the event-id
identity assertion from D1; do not write a second alignment path.

Training sources with aligned multi-placement streams, verified at 100% shared event ids:
**realdisp (9 placements), xrf_v2 (6), dsads (5), forth_trace (5)** — roughly 500k aligned
combinations currently loaded as independent single-device streams.

**D6b — the device-sampling policy. Three decisions to make and record:**

1. **Mixture ratio.** Evaluation has both single-device and composite cells, so training must see
   both. **Recommended: sample device count per episode**, e.g. 1 device with probability 0.5, else
   2–4 drawn uniformly. Training only on composites specialises the model; training only on singles
   leaves composite cells out of distribution. Record the chosen distribution in the run config.
2. **Subset sampling for wide sources.** realdisp has 9 placements. Using all 9 every time would
   blow the token budget and over-represent one source's geometry. **Draw a random subset per
   episode** from the source's available placements.
3. **Device-set agreement between query and support.** Allowing query and support to carry
   *different* device sets trains exactly the cross-configuration robustness the thesis claims, and
   it is free here. **Recommended: draw query and support device sets independently.** Evaluation
   still fixes both sides per cell (F6); this is a training-time augmentation only. Record it as a
   CONFIG-axis draw, not a NUISANCE one, per commit `9b7d75d` — the model conditions on placement
   text, so it must not be trained to be invariant to it.

**D6c — token budget.** Episode cost now scales with device count. `batch_under_token_budget` must
see the sampled `S`, or batches shrink silently and the effective learning rate schedule drifts.
Log realised tokens-per-batch and device-count histogram in telemetry.

**D6d — smoke only.** Build it, unit-test the sampler (device counts follow the declared
distribution; query/support sets are drawn independently; execution-disjointness holds across every
member), and run a <=50-step smoke. **Do not launch a training run.**

### D7. Staging

D is not required for A/B/C to land.

1. A, C0, C1, B, C2–C5 -> the single-device 4/8/16 s table.
2. D1–D4 -> multi-device for the **baselines** only; needs no HALO change. HALO appears
   per-device-pooled via the shared helper, or absent.
3. D5 -> HALO multi-device evaluation.
4. D6 -> HALO multi-device training. **Confirm with Alex before starting D5a/D6**: they touch the
   shared data path every other experiment depends on.

**D8 — open question, do NOT build.** Neither pooling path creates cross-device **interaction**.
`training/support_classifier/encoding.py:20` asserts `trunk == "temporal"`, and `encoder.py:145`
states *"A temporal trunk never mixes sensors"* — so each device is encoded in an independent lane.
The mean path then averages the lanes. The learned path (`RecordingAttentionPool`) is one query
attending over the lanes' tokens: it can **weight** a wrist token against a thigh token, but there
is no token-to-token attention, so a wrist token's contribution never depends on what the thigh
token says. Wrist-swinging-with-thigh-still — exactly the joint evidence multi-device should buy —
is therefore representable in neither path. The route to it is a sensor-mixing trunk
(`trunk="dual"`), which is a trunk ablation to raise separately. **Do not** change the trunk here.

## Acceptance

* Duration-specific grids and quality artifacts exist for every declared 4/8/16-second cell.
* Every model consumes one immutable model-independent source slice and episode manifest per cell;
  the cache key includes duration, ordered device set, source fingerprint, and model artifact.
* Released baseline length/padding contracts and all multi-device paths are covered by
  `tests/test_baseline_duration_contracts.py`, `tests/test_multi_device_support.py`, and
  `tests/test_sealed_eval.py`.
* Real released checkpoints produce valid composite embeddings through all four baseline paths.
* HALO's training sampler, device-aware encoder/pooling, gradient path, checkpoint round trip, and
  sealed composite encoder path pass the short CUDA probes summarized above.
* No sealed model score was computed while building or validating this machinery.

## Notes

* Land A first — independent, does not touch grids.
* Land C0 next — it is pure measurement and it determines the C2 design.
* Land C1 before B so placement streams are built once, into the new layout.
* **Do not "fix" any baseline's contract to make it fit the budget.** The comparison is only
  meaningful if each model consumes its native geometry; what is equalised is the evidence, not the
  architecture.
* If a probe shows a model accepts a length its paper does not document, prefer the documented path
  and record the discrepancy — we are comparing published models, not our best guess at them.
* The differentiable-neighbours readout (`training/support_classifier/neighbors.py`) itself needs
  **no change** for any of this: it consumes `(B,D)` query and `(B,K,D)` support vectors and is
  agnostic to how they were pooled. Every change in D5b is upstream of it.
* `learnable_recording_pool` is already on for every trained arm (`train.py:1224`); leave it as it
  is. The parameter-free property belongs to the neighbours **readout** and to the frozen arms'
  pooling, not to the trained encoder.
