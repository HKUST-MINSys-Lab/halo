# Debug sweep: the 18 labelled sources, three tracks

> **Resolution ledger, 2026-09-10.** This is the historical finding report, not a statement of
> current behavior. The confirmed mechanical defects have since been fixed and verified:
>
> - WISDM clocks are regularized from real timestamps and anti-aliased to 20 Hz before sensor join.
> - UniMiB's release trial id is now the execution identity for overlap/leakage guards.
> - undirected `climbing_stairs` canonicalizes to `stairs`, not `walking_upstairs`.
> - HHAR is anti-aliased to 50 Hz and retains Galaxy S+ as an honest accelerometer-only stream.
> - HARMES parses visit-4 comment markers, normalizes known annotation aliases, and describes the
>   right wrist consistently.
> - research IMUs are no longer described as watches/phones in the corrected StreamSpecs; WISDM's
>   recorded side qualifiers are retained.
> - sensor quantization uses a robust per-axis temporal-increment statistic; the artifact was
>   rebuilt (Capture-24 now measures about 0.0155 g).
> - MM-Fit uses the authors' released workout partition in the shared split manifest.
> - episodic and controlled-subset entry points reject retired sources; simultaneous-stream
>   resolvability uses the active paired roster.
> - corpus-matched CrossHAR, LiMU-BERT, and HARNet preprocessing starts from the same native rows,
>   trims by valid length, and anti-aliases to each published input contract. Schema guards reject
>   the old fitted checkpoints and heads.
>
> All affected sessions/grids, all three quality-cache alignments, the 202-label vocabulary,
> activity families, subject splits, and sensor-bias artifact were regenerated after these fixes.
> Findings marked `[A]` that require a new modeling policy rather than a mechanical correction
> (for example per-recording RealDisp displacement text) remain research decisions; they were not
> silently encoded as facts by this repair.

> Original 2026-09-10 report: three parallel tracks over all 18 sources in
> `deployment_policy.EXPANDED_18_TRAIN_DATASETS`, including the four retired on 2026-09-10 which
> remain wired behind the retirement gate. **[V]** = reproduced by me during this sweep;
> **[A]** = reported by a track sub-audit with evidence, not independently re-measured.
>
> Tracks: (1) is each source used according to its authors' original intent; (2) is the metadata
> the model conditions on honest (acquisition text, rate, units, gravity, channels); (3) is every
> source wired to training correctly. Per-track notes are in the session scratchpad
> (`sweep_intent.md`, `sweep_metadata.md`, `sweep_wiring.md`, `part1..5.md`).

## 0. What is confirmed correct

Worth stating first, because most of the chain is sound. [A, spot-checked V]

- All 18 datasets route 100% of their sessions to exactly one stream: zero orphans, zero double
  counts. 56 StreamSpecs map bijectively onto 56 native grids, all four grid files present, sample
  retention 0.96-1.07x against the sessions.
- The active 14-source index is subject-disjoint per dataset, every dataset has a non-empty
  validation fold, the subject-split manifest covers all 604 subjects including the retired four,
  and no retired source reaches the index.
- Collate probe over all 47 active streams: patches finite, `channel_mask` equals the grid mask on
  every stream, so **no accelerometer-only stream is conditioned on a phantom gyroscope** [V]; the
  two genuinely upsampled streams (`xrf_v2/airpods_ear` 25→50 Hz, `mmfit/left_ear` 85→100 Hz) carry
  the true-rate override, and the downsampled ones (`hhar`, `sp_sw_har`, `mmfit` watch/phone) need
  none [V].
- All 56 gravity states match measurement; accelerometer units complete and correct; channel sets
  match specs; `placement_site` resolves for all 56 with no collisions; `global_labels.json` and
  `activity_families.json` agree exactly (165 labels); quality caches fresh (85/85 fingerprints).
- Comparator path: all 14 active datasets are query-eligible, zero `(configuration, label)` cells
  lack cross-subject support, 0% episode-draw failure.
- `realdisp` placement mapping verified against the authors' manual, all nine units; `phytmo`
  correct/incorrect labels survive end to end; `dsads` block-to-segment mapping confirmed by
  gyro energy.

## 1. Findings that corrupt training

### 1.1 WISDM's declared 20 Hz is wrong for a third of its windows [V]

The converter keeps raw rows and real timestamps; the grid builder then windows by **row count**
at the dataset's declared rate and never reads `timestamp_sec`. Measured from the converted
sessions:

| stream | sessions actually >30 Hz | windows in those sessions |
|---|---:|---:|
| `wisdm/phone_pocket` | 247 of 920 (27%) | 18,221 of 39,329 (**46%**) |
| `wisdm/watch_wrist` | 259 of 1,168 (22%) | 5,227 of 30,876 (17%) |

25 of the 51 raw phone accelerometer files run at ~50 Hz (verified from raw nanosecond
timestamps). For those, every "6 s" window of 120 rows is really 2.4 s, and because the
filterbank places bands in physical Hz from the declared rate, **every frequency is scaled by 0.4**:
a 2 Hz gait reads as 0.8 Hz. The authors warned of mixed rates in the paper (the raw release has 16 phone subjects at ~50 Hz
and 9 at ~25 Hz [A]). 23,448 of 70,205 WISDM windows are affected, on the second-largest active
source. A second-order effect follows mechanically: the converter joins gyro onto accel with
`merge_asof(nearest)`, so wherever the accel clock is faster the gyro sample is duplicated, giving
17 of 51 phone subjects and 5 of 51 watch subjects more than 30% exact consecutive gyro repeats —
the same sample-and-hold pathology cited when `mhealth` was retired, on an active source. [A] Prior audits compared
`metadata.json` to grid `meta.json` (both say 20) and could not see it.

The same timestamp-versus-declared-rate test over all 18 clears the other 17 [V]. `kuhar` shows
symmetric Android timestamp jitter (7.6% of intervals too fast, 10.2% too slow, no gaps, median
exactly 10 ms), which is honest on average and minor. `phytmo` runs at 100.89 Hz against a declared
100, a 0.9% error [A].

### 1.2 UniMiB has the same overlap leak that retired UCI-HAR, and I missed it [V]

67.4% of consecutive same-subject `unimib_shar` windows share an exact overlap, **median 83% of the
window**, and windows-per-execution is 1.00. The corpus audit recorded "0.3% exact" because its
test looked for an exact *50%* overlap, which UniMiB's ~87% stride cannot satisfy. It should have
been retired on the same grounds as `uci_har` and `sp_sw_har`; it was kept for its eight fall
classes. Correction to the 2026-09-09 audit.

### 1.3 The stairs rule fabricates a direction two sources refused to record [V]

`canonical_labels.py` maps `climbing_stairs → walking_upstairs` under a comment reading
"direction preserved". The two sources that emit that string both define it as bidirectional:
FORTH-TRACE's release table says "climb stairs (up/down)" and mHealth's paper Table 1 says
"Climbing/descending stairs". About 1,310 direction-unknown FORTH-TRACE windows (~18% of the
class) enter `walking_upstairs`. The same file keeps WISDM's ambiguous `stairs` unmerged, which is
the policy this rule violates. [A for the paper citations, V for the mapping and the emitting
sources]

### 1.4 HHAR silently lost its only 50 Hz device class [A]

The release's phone gyroscope file has no rows for the two Samsung Galaxy S+ phones; the converter
requires gyro and `continue`s without logging. The built corpus has 6 of 8 devices, none of them
the 50 Hz class that anchors the low end of HHAR's rate heterogeneity, while the docstring and
manifest still say "50-200 Hz". Separately, the 200 Hz Nexus 4 is decimated 4:1 with `np.interp`
and **no anti-alias filter**, unlike the shared `resample_signal`, so the aliasing artefact is
device-correlated. The 327 MB of watch data is on disk and unused.


### 1.5 HARMES silently drops most fourth-visit recordings [V]

The converter's `event_segments` accepts only `start`/`end` event types. Each participant's fourth
visit is a free-living protocol logged as `Type=comment` with open-vocabulary descriptions
("Cutting wood", "Doing a puzzle"), so those logs yield no segments and the recording vanishes
without a print. Converted sessions carry 57 (participant, visit) recordings: visits 1-3 hold
15/17/19, **visit 4 holds 6** against 20 raw visit-4 recordings on disk (71 raw in total) [V]: 14 recordings
lost, every one of them a fourth visit, about 14 h of free-living open-label wrist data and the one
cross-day visit per participant. The converter
docstring also says the +1 h clock fix hits 4 of 20 participants; measured 39 of 71 recordings,
11 of 20 participants.

## 2. Findings that misdescribe the data to the model

### 2.1 Device profile is assigned by body site, not by hardware [A]

`forth_trace`, `dsads`, `mhealth` and `pamap2` wrist nodes are labelled `watch` while their
identical-hardware siblings on torso and knee are `device`; `realdisp`'s Xsens units are `device`
throughout. `dsads`'s "watch" is a cabled Xsens MTx tethered to a belt-worn hub. Consequence: 22 of
56 streams share a byte-identical descriptor, 13 `AcquisitionKey`s are shared by 2 to 6 streams,
and a 25 Hz strapped Xsens is `are_compatible()` with a 100 Hz Android smartwatch. The descriptor
is model input; `StreamSpec.note`, where the correct fact often lives, is read by no model path.

### 2.2 Placement text drops qualifiers the authors recorded [A]

WISDM's paper says right pants pocket and dominant hand; the model is told "the pocket" and
"the wrist". UCI-HAR's two placement trials (left belt versus self-placed) are collapsed. `kuhar`'s
"waist" is unsourced: no document we hold states a placement, and the one note says "varied".
RealDisp's ideal/self/mutual displacement regimes all emit byte-identical placement text, so the
corpus's only ground-truth displacement source enters as unlabelled nuisance variance. `xrf_v2`
goes further: the release's own `source_files` metadata shows all five body positions are WT53xx
research IMU nodes, yet the model is told "a watch on the left wrist" and "a phone in the left
trouser pocket". `harmes`'s manifest says right wrist while its StreamSpec says dominant wrist.

### 2.3 Rate over-claims through `rate_fidelity` [A]

`uci_har`'s released `total_acc` was Butterworth-filtered at 20 Hz by the authors: measured
24-26 Hz band power is 1.6e-9 of the 5-10 Hz band (versus 1.7e-2 for `hhar` at the same rate), yet
bands 21-25 Hz are advertised as acquired. `mhealth`'s gyroscope is dead (p99 |ω| 1.36 rad/s),
possibly in deg/s per the release README, and 72% sample-and-hold, on a stream declared 50 Hz.
`STREAM_SOURCE_RATE_HZ` is keyed per stream and cannot express "accelerometer faithful, gyroscope
held". Both sources are retired; the mechanism gap is not.

### 2.4 `sensor_bias.quantization_step` is broken [A]

Stored as a stream-wide minimum difference. Capture-24's effective step is ~0.0156 g (my own
measurement: median non-zero step 0.0158 g, 56.6% consecutive repeats [V]); the stored value is
1.49e-8, off by six orders of magnitude, so the coarsest accelerometer in the corpus is
z-scored as the finest. Currently inert because bias conditioning defaults off.

## 3. Findings about intent and protocol

- **Every authors' train/test split is stored and ignored** except MM-Fit's, and MM-Fit's is
  honoured only in the HALO index (`pretrain_data.py`), not in `data/labels/subject_splits.json`,
  which random-splits w00 to val and w11/w15 to test; the paper puts w11 in the seen-subject test
  set, and 14 of 21 workouts come from 3 people, so the **baseline** adapters that read that
  manifest are exposed to person leakage on MM-Fit. [A]
- **Capture-24's 24 h timeline is cut into 13,120 label bouts, each its own execution.** A
  (subject, label) cell holds median 8, mean 12.4, up to 74 executions [V], with 38% of consecutive
  same-cell bouts separated by at most one intervening bout [A]. Harmless under the cross-subject
  relation (the query's subject is excluded outright), but under same-subject enrollment a person's
  adjacent bouts of the same posture are trivially easy support. Also: the official P001-100 /
  P101-151 split is ignored, and the build cap (`max_hours_per_class`) is not in effect, so >99% of
  the annotated hours are gridded. [A]
- **Opportunity's 37% full-window rate is our artefact**: one session per locomotion block gives
  3,620 sessions with a median of one window. It also keeps only the jacket IMUs and drops the
  release's bilateral wrist and hip accelerometers, then is excluded for having "appendix-only"
  placements. The 2,551-instance gesture track is discarded before the parquet. [A]
- **Harmonised and native grids see different corpora**: harmonised excludes partial windows,
  native keeps them. Opportunity 5,757→2,153 windows; FORTH-TRACE 2,466→2,068 with 7 of 9
  transition classes going to zero. HALO and the layout-locked baselines are not trained on the
  same data. [A]
- FORTH-TRACE's part4 exclusion note ("five annotation tracks describe different takes") is
  factually wrong; all five nodes carry the same 29-run sequence. The real fault is one broken
  torso node, and it costs all five placements including a clean bilateral wrist pair. [A]
- `pamap2` drops six optional activities silently; `unimib_shar` clips at the ±2 g rail in 30% of
  windows and 60-70% of every fall class, and discards a recoverable left/right pocket field. [A]
- Six of 18 sources have no usable licence recorded; `nfi_fared` and `xrf_v2` have none at all. [A]

## 4. Findings about wiring

- **`training/tokenizer/pretrain_episodic.py` is an ungated trainer** [V]: no
  `assert_no_retired_sources`, and its `--smoke` hard-codes `["uci_har", "wisdm", "mhealth",
  "pamap2"]`. My 2026-09-10 claim that every training entry point is gated was wrong.
- `--subset` is now dead by construction (its recipe names `uci_har`), while its diagnostic twin
  `eval/tokenizer_metrics.py → build_subset_index` still scores on `uci_har` ungated. [A]
- `validate_sensor_bias_training_corpus` has zero callers; `sensor_bias.json` was built on all 18.
  Inert today because bias conditioning is off. [A]
- `--corpus label_free` cannot run yet: four of its five sources have no grids. Expected, since
  nothing has been downloaded. [A]
- Same-subject enrollment is structurally impossible for `dsads`, `forth_trace`, `mmfit`,
  `realdisp` (0 of 150 forced draws) and rare for `wisdm`/`pamap2`: the grouped execution unit
  blocks it, and the telemetry aggregates it away. `--same-subject-probability` is not realised for
  half the roster. [A]
- Cosmetic: `build_memory._load_vocab()` reads the current vocabulary with no roster check, so an
  18-source checkpoint silently gets the 14-source vocabulary; `resolvability.PAIRED_DATASETS`
  names two retired sources and omits three trained multi-placement ones. [A]

## 4b. The pattern across the intent track [A]

Seven of the eighteen sources were built to study the exact axis HALO claims, and the pipeline
erases it before the model can see it. RealDisp's ideal/self/mutual displacement regimes reach no
code path (`grep mutual` finds two comments). HHAR loses its device identity, its only 50 Hz phone
class, and all four watches. UCI-HAR's two placement conditions and UniMiB's balanced left/right
pocket are discarded. WISDM's right-pocket/dominant-hand qualifiers are dropped. The corpus's
ground-truth configuration variation enters as unlabelled nuisance variance that the invariance
objectives are trained to average away. Separately, four sources (`sp_sw_har`, `nfi_fared`,
`harmes`, `xrf_v2`) have no local review copy, so their side, site and device claims could not be
checked against a publication at all, and `nfi_fared` and `xrf_v2` have no recorded licence.

## 5. Documentation contradictions [A]

`DATA_HETEROGENEITY.md`: `unimib_shar` unit stated as g (code and measurement say m/s²);
`hhar` native rate stated as 50 Hz (it is 50-200, non-anti-aliased); `wisdm` "gyro is optional"
(both streams carry a live triad; the `merge_asof` join reuses 27% of phone gyro samples,
unrecorded); an eval row for `mobiact` whose grid is a zero-window placeholder at
`rate_hz = 0.0`. `xrf_v2`'s StreamSpec comment says 34 ADLs; there are 30. `pamap2`'s
`num_sessions` is stale (4377 vs 106).

## 6. Ranked, for decision

1. WISDM rate mislabel (§1.1): second-largest active source, a third of its windows spectrally
   wrong by 2.5x. Fixable in the converter by resampling on real timestamps or by splitting the
   dataset by acquisition rate.
2. UniMiB overlap leak (§1.2): retire it or fix the execution guard. My earlier audit understated it.
3. `pretrain_episodic.py` ungated (§4): one call and a smoke-roster edit.
4. Stairs-direction fabrication (§1.3): drop the `climbing_stairs` rule or map it to an
   undirected `stairs`.
5. HHAR missing device class and un-anti-aliased decimation (§1.4); HARMES visit-4 loss (§1.5).
6. MM-Fit split in the shared manifest (§3): affects baselines, not HALO.
7. Device profile by site (§2.1) and the RealDisp regime blindness (§2.2): representation
   questions, not bugs, but they decide what "cross-configuration" can mean.
