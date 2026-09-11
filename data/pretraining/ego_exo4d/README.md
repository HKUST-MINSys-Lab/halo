# Ego-Exo4D — Aria head IMU (label-free Phase-A pretraining)

Head-mounted Project Aria IMU extracted from Ego-Exo4D takes. No video, no
labels, no benchmark participation — this exists purely to add head-placement
scale to Phase-A self-supervised pretraining.

| | |
|---|---|
| Source | https://docs.ego-exo4d-data.org/ (release v2) |
| Sensor | Project Aria glasses, head |
| Streams kept | one IMU per take: `imu-right` (1202-1, ~800 Hz) preferred, `imu-left` (1202-2, ~1 kHz) fallback |
| Streams dropped | video, audio, eye gaze, trajectory, point cloud, magnetometer, barometer, the second IMU |
| Channels | `acc_{x,y,z}` (g, gravity **present**, device frame), `gyro_{x,y,z}` (rad/s, device frame) |
| Stored rate | uniform 200 Hz |
| Labels | none — every session is `__unlabeled__`; `phase_a_only: true` |
| Scale (full release) | 1,286 camera-hours; 221.26 ego-camera IMU hours, 740+ participants, 13 sites |
| Size | `take_vrs_noimagestream` is **995.592 GB** on the wire; ~1.9 GB for the planned ego-IMU grids at 200 Hz |

## Why this dataset

The corpus has exactly one head-worn stream today (`xrf_v2` smart glasses, 16
subjects). Ego-Exo4D adds head placement at 740+ participants and 13 recording
sites, unlabeled, which is the shape Phase-A wants: many people, many
environments, many device orientations, no vocabulary to overfit.

## Licence and access (no way around it, and none is attempted)

1. **Sign the License Agreement** at <https://ego4d-data.org/>. Approval takes
   about **two days**. Do this first.
2. `pip install 'ego4d>=1.7.1'` — provides the `egoexo` CLI. 1.7.1+ enables the
   fast download path.
3. `pip install awscli && aws configure` — enter the access key and secret from
   the licence email, then press Enter twice to accept the default region and
   output format. If you keep it in a named profile, pass `--s3-profile <name>`.
4. `pip install projectaria-tools` — needed only by `convert.py`, to read VRS.

`fetch.py` checks 1–3 up front and prints exactly these steps if anything is
missing. It never tries to work around the licence.

## Commands

```bash
# Plan only: fetch the 46 MB metadata part, pick a seeded subset, price it.
python -m data.pretraining.ego_exo4d.fetch --takes 40 --max-gb 120 --dry-run

# Download that plan (metadata part first, then take_vrs_noimagestream).
python -m data.pretraining.ego_exo4d.fetch --takes 40 --max-gb 120

# Site- or split-scoped variants.
python -m data.pretraining.ego_exo4d.fetch --takes 100 -u cmu unc sfu --max-gb 200
python -m data.pretraining.ego_exo4d.fetch --uids <take_uid> ... --max-gb 20
python -m data.pretraining.ego_exo4d.fetch --takes 40 --splits train --s3-profile egoexo

# Convert whatever is on disk into HALO sessions.
python -m data.pretraining.ego_exo4d.convert --takes 40 --max-hours-per-take 1
```

`--parts take_vrs_noimagestream` is always passed and is not configurable: it is
the only part that contains the IMU without also containing ~12 TB of video.

### Bounding and reproducibility

* `--takes N` selects N takes by salted-hash order over sorted `take_uid`s
  (`sha256("<seed>:<take_uid>")`). Same seed + same metadata release = same set,
  and growing N *extends* the previous set, so a bigger re-run resumes rather
  than reshuffles.
* `--max-gb` (default 120) refuses the plan before any object is pulled.
  takes.json publishes **no per-file sizes**, so the number is an **estimate**:
  the documented 995.592 GB apportioned by each take's `duration_sec` over the
  whole release (`estimate_method: "duration_share"`), falling back to
  995.592 GB / number-of-takes when durations are absent
  (`"uniform_per_take"`). The manifest records `estimate_is_measured: false`.
* `downloads/fetch_manifest.json` records the seed, the requested filters, the
  selected takes with their participant/capture uids, the estimate and its
  method, then after the download the `realized_take_uids`, `missing_take_uids`
  and `bytes_on_disk`. Re-running is a no-op for takes already present.
* `--splits` is forwarded to the CLI, which applies it *after* uid selection, so
  a split filter can legitimately shrink the realized set below `--takes`. That
  shows up as `missing_take_uids`, not as a silent success.

## Signal decisions

**Units.** Aria reports acceleration in **m/s²** and angular rate in **rad/s**,
both in the device frame, gravity present. We divide acceleration by 9.80665 to
reach the corpus unit of **g** and leave gravity in (the corpus-wide
`gravity_state: present` contract); gyro passes through unchanged. The converter
probes the quietest two seconds of every session and writes the median
`|acc|` to the manifest as `stationary_gravity_g_median`; it should read ~1.000,
and a value outside [0.8, 1.2] prints a loud warning rather than being buried.

**200 Hz.** The frontend's f_max is ~14 Hz. Storing 800 Hz or 1 kHz would cost
4–5× for information the model never reads. 200 Hz still leaves an order of
magnitude of headroom over the band that matters.

**Anti-aliasing.** VRS IMU timestamps are non-uniform device timestamps in
nanoseconds. Each segment is low-passed with a zero-phase FIR (Hamming,
`filtfilt`) whose passband ends at 0.4×200 = **80 Hz** and whose stopband starts
by the new Nyquist at 100 Hz, then interpolated onto a uniform 200 Hz grid
against the *measured* timestamps — so device-clock jitter is absorbed instead of
being assumed away by a nominal-rate `resample_poly`. Zero-phase filtering keeps
passband gain at 1 and introduces no group delay. Tests assert that a 5 Hz tone
survives with amplitude preserved to within 3% and that 260/310/390 Hz tones
(which would otherwise fold to 60/90/10 Hz at full amplitude) are annihilated.

**Gaps.** Aria drops IMU packets. Linearly bridging a 1 s dropout would read as a
slow, smooth head movement that never happened, so **any gap > 0.5 s splits the
session** instead of being interpolated across. Parts are suffixed
`..._aria_head_part01`, `..._aria_head_part02`; `aria_head` stays a matchable
substring so the deployment-policy stream token still resolves. Parts shorter
than one 8 s window are dropped. Each part's `timestamp_sec` restarts at 0.

**Subject identity.** HALO splits are subject-disjoint, so a take must resolve to
a stable person. `takes.json` carries `participant_uid` — a release-wide integer
id joined to `participants.json` — and that is what we use: the subject id is
`p<participant_uid>`, so several takes by one person collapse to one subject.
When `participant_uid` is absent we fall back to `capture_<capture_uid>`: a
capture is one recording session with one wearer, so multiple takes of that
capture still land in one fold; the cost is that one person recorded across
several captures becomes several pseudo-subjects (conservative — it never leaks
a person across folds, it only over-splits). A per-take id is **never** invented,
because that would silently break subject-disjoint splitting. The manifest
reports `subject_id_source` (`participant_uid` / `capture_uid_fallback` /
`mixed`) and `subject_id_fallback_takes`, and the note text carries the warning
into every downstream consumer of the manifest.

## Output contract

```
data/pretraining/ego_exo4d/
  downloads/fetch_manifest.json
  sessions/egoexo_<take_uid>_<subject>_aria_head[_partNN]/data.parquet
  labels.json      {session_id: ["__unlabeled__"]}
  manifest.json
  metadata.json
```

`data.parquet` columns, in order:

| column | dtype | meaning |
|---|---|---|
| `timestamp_sec` | float64 | seconds from session start, monotonic, uniform 1/200 s |
| `acc_x/y/z` | float32 | g, **gravity present**, Aria device frame |
| `gyro_x/y/z` | float32 | rad/s, Aria device frame |
| `subject` | string | `p<participant_uid>` or `capture_<capture_uid>` |

`metadata.json` is checked in as a static descriptor (so the dataset is
discoverable by `data.scripts.curate.corpus_roots` before anything is converted)
and is rewritten by `convert.py` with the realized counts. It declares
`role: "pretrain_scale"`, `phase_a_only: true`,
`pre_windowed: false`, `streaming_grid: true`, `activities: []`, and
`placement: "the head (smart glasses)"` (matching the wording `xrf_v2` already
uses, which renders as "on the head (smart glasses)"), and
`grid_dtype: "float16"` — two bytes a sample is what keeps the label-free corpus
inside its disk budget, and float16 resolves ~0.001 g at 1 g, far below anything
a <14 Hz frontend reads.

## Phase-A only

There are no activity annotations here, and the ones Ego-Exo4D does publish are
video-grounded keystep/narration annotations, not IMU labels. Every session
carries the reserved `__unlabeled__` marker. Label-vocabulary construction,
validation probes, and the Phase-B evidence bank must exclude it.

## Not verified without the data

The following could not be checked offline (no licence, no ~1 TB download) and
are handled defensively rather than assumed:

* The exact on-disk filename of a `take_vrs_noimagestream` object. The docs say
  such files are suffixed `_noimagestreams`, so `find_take_vrs` globs `*.vrs`
  under the take directory and ranks candidates: `noimagestream` first, then a
  cam id the take's own embedded capture record marks `is_ego` + `device_type ==
  "aria"`, then anything containing `aria`. A full-`take_vrs` download works too.
* Whether the metadata part lands at `<out-dir>/takes.json` or under a nested
  release directory — both are searched, shallowest first.
* Real dropout statistics, and therefore how many sessions the 0.5 s gap rule
  actually produces per take.
* How often `participant_uid` is null in practice, and hence how much of the
  corpus falls back to capture-level subject ids. The manifest counts it at
  convert time.

## Processing the 221 h ego-camera IMU subset on a bounded working budget

`corpus_plan.py` books this source at 1 stream x 221.26 ego-camera wall-hours x 200 Hz x 6
channels = **~1.9 GB gridded**, against a ~1 TB image-free VRS download. The
download is therefore streamed and deleted, not hoarded: fetch a batch, convert
it, delete the VRS, advance the seed window.

```bash
for START in 0 40 80 120; do
  python -m data.pretraining.ego_exo4d.fetch --takes $((START + 40)) --max-gb 120
  python -m data.pretraining.ego_exo4d.convert --keep-existing
  # keep the parquet, drop the VRS; the manifest still records what was taken
  find data/pretraining/ego_exo4d/downloads -name '*.vrs' -delete
done
```

Because the seeded subset is a prefix chain, `--takes 80` is `--takes 40` plus 40
new takes, so each round genuinely advances instead of re-downloading. Deleting
the VRS makes `realized_take_uids` in `fetch_manifest.json` the record of what
was already consumed. `--keep-existing` is what makes the loop safe: without it
`convert.py` rebuilds `sessions/` from scratch, which would destroy earlier
batches whose VRS is already gone. With it, existing sessions are carried
forward and their takes are skipped, so each round only converts what is new.

## Tests

`tests/test_pretrain_ego_exo4d.py` runs fully offline: `projectaria_tools` is
monkeypatched with a fake VRS provider that serves synthetic IMU records, so unit
conversion, anti-aliasing, resampling, gap splitting, subject resolution, the
byte budget, the seeded subset, and the whole on-disk contract are all exercised
without the library and without the data.
