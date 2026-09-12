# Nymeria — label-free JEPA encoder-pretraining source

Meta Project Aria **Nymeria** (ECCV 2024, arXiv 2406.09905) and **NymeriaPlus**
(arXiv 2603.18496): ~1,200 sequences, 300 hours, 264 participants, 50 locations of
everyday activity. Each ~15-minute sequence simultaneously records

| device | what | rate |
| --- | --- | --- |
| Xsens MVN Link suit | 17 body-worn inertial trackers | 240 Hz |
| Project Aria headset | 2 IMUs (1202-1 and 1202-2) | ~800 Hz / ~1 kHz |
| miniAria wristbands (L + R) | 1 IMU each | ~800 Hz / ~1 kHz |
| observer headset | a **second person** following the participant | — (excluded) |

It is the only source in reach with real IMU on the head, both wrists and every limb
**at the same instant**, which is exactly the cross-configuration structure JEPA pretraining
needs. There are no activity annotations, so every session here carries the reserved
`__unlabeled__` marker: JEPA self-supervision may use it, while label-vocabulary construction and
support-classifier episodes must not.

Licence: **CC BY-NC 4.0** (non-commercial), email-gated.

---

## 1. Access

Nothing in this module tries to bypass the licence.

1. Open <https://www.projectaria.com/datasets/nymeria/> and accept CC BY-NC 4.0.
   Registration turnaround is typically ~48 h.
2. In the Aria Dataset Explorer, pick sequences (filter by location / date /
   scenario) — or select everything; `fetch.py` filters locally before downloading.
3. When choosing download groups, tick exactly this one and nothing else:

   | group | files per sequence |
   | --- | --- |
   | `body_xdata_mvnx` | Xsens MVN Link suit recording (`*.mvnx`) |

   Do not tick any video or VRS group. The Xsens assets alone are about 1.96 TB for
   the 1,064 sequences that expose them, so HALO takes a deterministic bounded subset.
4. Save the emailed manifest to `data/pretraining/nymeria/downloads/url.json`
   (or pass `--url-json PATH`). It is credential-like and time-limited; the local
   `.gitignore` keeps it out of the repo.

Optionally install the official downloader (preferred, and used automatically when
it is on `PATH`):

```bash
pip install git+https://github.com/facebookresearch/nymeria_dataset   # nymeriaplus-download
pip install projectaria-tools                                        # only for optional Aria conversion
```

## 2. Commands

```bash
# See exactly what would be pulled, byte counts included. Downloads nothing.
python -m data.pretraining.nymeria.fetch --sequences 40 --dry-run

# Fetch a deterministic 40-sequence Xsens subset, hard 120 GB ceiling.
python -m data.pretraining.nymeria.fetch --sequences 40 --max-gb 120 --workers 8

# No arguments = the bounded default of 40 sequences (this is how
# data.pretraining.build_corpus --stage fetch invokes it). Everything in your
# manifest instead, still under --max-gb (the supplied full manifest will refuse):
python -m data.pretraining.nymeria.fetch --all-sequences --max-gb 400

# Named sequences instead of a seeded draw.
python -m data.pretraining.nymeria.fetch --sequence-ids 20230607_s0_... 20230608_s1_...

# Convert the corpus-of-record Xsens stream into HALO sessions.
python -m data.pretraining.nymeria.convert --streams xsens

# Xsens only (no projectaria_tools needed), capped at 10 minutes per sequence.
python -m data.pretraining.nymeria.convert --streams xsens --max-hours-per-sequence 0.1667

# Report the real array schema of one body/xdata.npz (see §5).
python -m data.pretraining.nymeria.convert --inspect-npz downloads/<seq>/body/xdata.npz

# Tests (offline, no downloads, no projectaria_tools):
python -m pytest tests/test_pretrain_nymeria.py -q
```

`fetch.py` is deterministic and re-runnable: sequence selection is a SHA-256 order over
sorted sequence uids for a given `--seed` (the same scheme as
`data/datasets/nhanes/fetch.py`), and every run writes
`downloads/fetch_manifest.json` recording the seed, the groups, the exact per-sequence
file list and the byte counts.

### Safety rails

* Taking everything is explicit, never the default: with no arguments the fetcher uses a
  bounded, deterministic 40-sequence draw and says so. `--all-sequences` opts in, and
  `--max-gb` still applies to it.
* The url.json is **filtered to the selected sequences and groups first**, written to
  `downloads/url.filtered.json`, and only that filtered file is handed to
  `nymeriaplus-download`. No invocation can pull the full release by accident.
* `--max-gb` (default **120**) is checked against the filtered plan before any byte
  moves. Over budget → hard refusal.
* If any planned file declares no size, the budget cannot be enforced, so the fetcher
  **refuses** rather than guessing. `--allow-unknown-size` opts into an unbounded pull.
* Resume: the fallback path checks the destination file's on-disk size against the
  manifest's declared size, which is independent of any log-file naming convention, and
  additionally writes markers under `downloads/.download_logs/`. Partial downloads land
  on `*.part` and are only `os.replace`d after size and (when published) SHA-1 verify.

---

## 3. Output contract

**Two dataset directories, not one** (see §4 for why). They are **siblings** of this
package, because `corpus_roots.dataset_root(name)` resolves a dataset to
`data/pretraining/<name>`:

```
data/pretraining/
  nymeria/                          ← this package (code + authored metadata only)
    fetch.py  convert.py  README.md
    module_metadata.json            module descriptor (NOT named metadata.json — see below)
    nymeria_xsens.metadata.json     authored template, copied on convert
    nymeria_aria.metadata.json      authored template, copied on convert
    manifest.json                   generated: module summary, no scalar rate
    labels.json                     generated: union of both datasets' labels
    downloads/                      generated: the licence-gated download tree
  nymeria_xsens/                    240 Hz  ← a real dataset, what build_grids reads
    metadata.json  manifest.json  labels.json  sessions/<session_id>/data.parquet
  nymeria_aria/                     200 Hz  ← a real dataset
    metadata.json  manifest.json  labels.json  sessions/<session_id>/data.parquet
```

This package deliberately holds **no `metadata.json`**. `corpus_roots._is_dataset_dir`
recognises a dataset by exactly that filename, so one here would make `nymeria` a phantom
third dataset picked up by `dataset_names(PRETRAIN_ROOT)`. The module descriptor is
`module_metadata.json`, and each dataset's real `metadata.json` is copied from the
authored template at convert time (the dataset directories are generated output).

Each `data.parquet` is the existing repo-wide session contract:

| column | dtype | meaning |
| --- | --- | --- |
| `timestamp_sec` | float64 | seconds from session start, monotonic, exactly `1/rate` spacing |
| `acc_x/y/z` | float32 | **g**, gravity **present**, device frame |
| `gyro_x/y/z` | float32 | **rad/s**, device frame (omitted entirely if the source has no gyro) |
| `subject` | string | stable participant id |

`labels.json` is `{session_id: ["__unlabeled__"]}` for every session.

### Session ids and stream tokens

`nymeria_<sequence_uid>_<subject>_<stream_token>` , plus `_pNN` when a gap split the
recording (`..._aria_lwrist_p02`). The stream token is always a **matchable substring**,
so `deployment_policy.StreamSpec(session_contains=("_xsens_pelvis",))` works the same way
it does for `xrf_v2` and `wisdm`.

| dataset | tokens |
| --- | --- |
| `nymeria_xsens` (240 Hz) | `xsens_head` `xsens_sternum` `xsens_pelvis` `xsens_lforearm` `xsens_rforearm` `xsens_lupperarm` `xsens_rupperarm` `xsens_lthigh` `xsens_rthigh` `xsens_lshank` `xsens_rshank` |
| `nymeria_aria` (200 Hz) | `aria_head` `aria_lwrist` `aria_rwrist` |

Eleven of the suit's seventeen trackers are taken: head, torso, and both sides of each limb.
Hands, feet, and other non-target suit trackers are excluded. `xsens_sternum` resolves to the MVN
Link trunk tracker, which MVNX labels **`T8`** (some exports say `Sternum`; both are
accepted). `recording_observer` is never converted — it is worn by a second person.

---

## 4. Why two dataset directories (the two-rates decision)

`data/scripts/build_grids.py` reads **one scalar rate per dataset directory**:

```python
ds_dir = REPO / "data" / "datasets" / dataset
native_rate = float(json.loads((ds_dir / "metadata.json").read_text())["sampling_rate_hz"])
```

There is no per-stream rate anywhere in that path, and the `native` alignment builds
grids at exactly this rate with no resampling. The Xsens streams are genuinely 240 Hz and
the Aria streams are genuinely 200 Hz after decimation, so a single `metadata.json`
would have to misdeclare one of them — and because HALO's native-rate arm trusts this
number as physical truth, a misdeclared rate is not cosmetic: it would put every filterbank
centre frequency in the wrong physical place for one of the two modalities.

So the converter emits `nymeria_xsens/` and `nymeria_aria/` as two complete, independent
dataset directories, each with its own honest `sampling_rate_hz`. `module_metadata.json`
is flagged `"is_module_descriptor": true` and carries `sampling_rate_hz_by_dataset` instead
of a scalar; this package's generated `manifest.json` has **no** `sampling_rate_hz` key at
all, so nothing can read a single rate out of it by accident.

This matches what the surrounding infrastructure already expects:
`data/pretraining/corpus_plan.py` plans `nymeria_xsens` at 240 Hz / 11 streams and
`nymeria_aria` at 200 Hz / 3 streams, `build_corpus.py` maps both to the single module
`data.pretraining.nymeria`, and `deployment_policy.py` carries eleven + three `StreamSpec`
entries keyed on exactly the `session_contains` tokens above.

**Still outstanding, and owned elsewhere:** `data/scripts/curate/accel_units.py` lists
`nymeria_xsens` and `nymeria_aria` under `ACC_UNIT_G` with a comment saying the source is
m/s² and "the converter divides by 9.80665" — which is what this converter does, so that
classification is correct as written; but `tests/test_accel_units.py` was failing before
this module existed because `ego_exo4d` and `synthetic_imu` are still unclassified. Also,
the root `.gitignore` only globs `data/datasets/*/`, so generated `sessions/`, `grids/`,
`manifest.json` and `labels.json` under `data/pretraining/*` are currently untracked-but-
not-ignored (true for `nhanes` today as well).

---

## 5. Signal handling — what is verified and what is not

### Xsens MVNX → device frame

The contract needs the **device-frame measurement with gravity present, in g**. MVNX does
not store that, so it is reconstructed:

```
a_device_g = R(sensorOrientation)ᵀ · (sensorFreeAcceleration + [0, 0, +9.80665]) / 9.80665
w_device   = R(sensorOrientation)ᵀ · angularVelocity
```

* `sensorFreeAcceleration` is gravity-**removed** and expressed in the **global** frame, so
  gravity has to be re-added before rotating. `[0, 0, +9.80665]` is the **specific force** a
  stationary accelerometer reads (up, not down) in MVN's default **Z-up** right-handed global
  frame — which is why a motionless tracker comes out at exactly +1 g on its gravity axis.
  `--up-axis` overrides the axis if a recording used a different global convention.
* Sensor-level fields are preferred because they are the actual device measurement. When they
  are absent the segment fields `acceleration` + `orientation` are used identically, and
  `manifest.json` records which path ran under `detail.acceleration_source`
  (`"sensor"` or `"segment"`).
* **MVNX has no sensor-level angular velocity.** The gyroscope is therefore segment
  `angularVelocity` (global frame) rotated by the **sensor's** orientation. That is physically
  exact: a tracker is rigid with its segment, so they share an angular velocity vector, and
  only the frame it is expressed in differs.
* Quaternions are read **real-part-first** (`wxyz`, the MVNX convention); `--quat-order xyzw`
  overrides.
* **Angular units** are taken from the file's own unit declarations when it has any. When it
  declares none, rad/s (the MVNX SI default) is assumed, the run prints a warning, and
  `manifest.json` says `angular_unit_source: "assumed_mvnx_si_default"` rather than pretending
  the unit was verified. A plausibility heuristic additionally warns when the p99 magnitude
  exceeds 35 rad/s (~2000 °/s, above a body-worn MTw's range), which would indicate degrees.
  `--angular-unit deg` forces the conversion.

### Aria VRS → 200 Hz

* `projectaria_tools` (`from projectaria_tools.core import data_provider`) is required. If it
  is missing the converter **fails with the pip package name**; it never silently skips.
  `--streams xsens` runs without it.
* **One IMU per device**: `imu-right` (1202-1, ~800 Hz) preferred, `imu-left` (1202-2, ~1 kHz)
  as fallback. The one actually used is recorded per run in `manifest.json`
  (`detail.imu_stream`).
* Aria reports accel in **m/s² and gyro in rad/s, already in the device frame**. Accel is
  divided by 9.80665 → g, gravity present. Nothing is rotated.
* **Anti-alias then decimate to exactly 200 Hz.** Each contiguous run is linearly interpolated
  onto the sensor's nominal uniform clock (800 or 1000 Hz, snapped from the measured rate),
  then `scipy.signal.resample_poly` applies a Kaiser-window FIR whose cutoff is the 100 Hz
  **output** Nyquist before downsampling. The frontend's f_max is ~14 Hz, so 200 Hz is far
  above anything used and storing 1 kHz would cost 5× for nothing. Tested: a 5 Hz tone keeps
  its amplitude to within 2 %, a 300 Hz tone comes out below 1 % — it is filtered out, not
  folded back onto the passband.

### Gap policy (both modalities)

A gap longer than **0.5 s** is a hard boundary and is **never interpolated across**. The
recording is split into `_p01`, `_p02`, … parts, the stream token stays a matchable substring,
and each part is then truncated to a whole number of eight-second windows — so no downstream grid
window can straddle a gap. A backwards clock step is treated as a gap too, not silently sorted.
For Xsens the clock is the MVNX **frame index** divided by `frameRate`, so a dropped-frame run
shows up as a real gap that the nominal `time` attribute would have hidden.

### NOT VERIFIED — flagged honestly

The real release is licence-gated and 80 TB; none of it was downloaded while building this
module. These are the assumptions that a first real conversion must confirm:

1. **`sensorFreeAcceleration` is global-frame and gravity-removed.** Taken from the task brief
   and the standard Xsens XKF definition of "free acceleration". If it turned out to already be
   sensor-frame, the rotation step would be wrong — check that a still tracker reads |a| ≈ 1 g
   and that walking `acc_z` looks like a gravity-dominated signal, not a zero-mean one.
2. **MVN's global frame is Z-up.** The `--up-axis` flag exists for this. Symptom of a wrong
   guess: a stationary tracker still reads 1 g but on the wrong axis, and the DC/gravity feature
   is systematically rotated.
3. **MVNX angular-velocity units.** MVNX has no universal unit-declaration element, so the
   parser scans every header attribute containing "unit" and falls back to rad/s with a warning
   (see above). MVN *does* export degrees for some angular quantities (joint angles), so this
   needs one real check.
4. **`body/xdata.npz` schema is completely unknown.** The key names in `NPZ_CANDIDATES`
   (`convert.py`) are *candidates*, not observations. `load_npz_recording` **refuses** with a
   full dump of the real keys, shapes and dtypes when none match, rather than guessing; `auto`
   then falls through to the MVNX path. Run `--inspect-npz <path>` on the first downloaded
   sequence, paste the output, and correct `NPZ_CANDIDATES`.
5. **Participant identity.** An explicit participant field in `<seq>/metadata.json` takes
   precedence. Otherwise, the official release UID format
   `date_setup_first_last_activity_recording` supplies the stable participant pseudonym. This
   extracts 236 recurring identities from the issued 1,100-sequence manifest. Generic MVNX labels
   such as `MVN System` are rejected; a nonstandard sequence without another verified identity is
   also rejected. Treating each recording UID as a person would invalidate subject balancing.
6. **url.json schema.** The manifest is issued per user and its key spelling is not public. The
   parser is tolerant (several accepted spellings for url / size / filename / sha1, and it
   accepts a record, a list of records, or a name→record mapping) and raises
   `CatalogShapeError` with the observed top-level keys when it recognises nothing.
7. **The official downloader's `.download_logs/` marker filenames.** Unverified, so resume does
   not depend on them: the size check against the destination file is the primary signal, and
   our own markers are written alongside in the same directory.
8. **`projectaria_tools` IMU accessor names** (`capture_timestamp_ns`, `accel_msec2`,
   `gyro_radsec`). Alternatives are tried and, if none match, the error lists the sample
   object's real attributes.

Items 1–3 and 5 are cheap to confirm on the very first downloaded sequence and should be
confirmed before any pretraining run uses this data.

---

## 6. Tests

`tests/test_pretrain_nymeria.py` — 39 tests, fully offline, no downloads, and
`projectaria_tools` is injected as a fake module so the VRS path runs without the real
library. Covered: the rotate-and-add-gravity math (stationary → exactly 1 g, rotated →
still 1 g on the expected axis, gyro rotated into the device frame, deg→rad override, the
segment fallback, `T8`→`xsens_sternum`), url.json parsing / group filtering / deterministic
selection / `--max-gb` and unknown-size refusals / envelope-preserving pruning / the licence
message, gap splitting on synthetic non-uniform timestamps (including a backwards clock),
decimation of 5 Hz vs 300 Hz from both 800 and 1000 Hz, the npz refusal-with-schema-dump, and
the full output contract end to end (parquet columns and dtypes, `timestamp_sec` spacing,
whole-window truncation, `__unlabeled__` markers, manifest keys, the rateless module manifest, and
that `corpus_roots.dataset_root` finds both converted datasets while this package itself is
not mistaken for one).
