# data/pretraining — label-free scale sources

Sources in this tree exist for one purpose: **self-supervised representation pretraining**.
Every session here carries the reserved `__unlabeled__` activity marker, so none of it can
reach label vocabulary construction, validation probes, or the Phase-B evidence bank — the
existing `__unlabeled__` guards enforce that, and this directory makes the intent visible on
disk rather than only in a roster tuple.

Labelled corpora live in [`../datasets`](../datasets) and are unaffected. The on-disk
contract is identical in both trees, so nothing about a source changes when it moves; code
resolves a dataset by name through
[`data/scripts/curate/corpus_roots.py`](../scripts/curate/corpus_roots.py) rather than
joining a fixed root.

## Why this tree exists

The JEPA objective needs no labels, so the ceiling on pretraining data is not annotation
cost but download and disk. The current 18-source labelled corpus supplies 3,117 stream-hours,
of which Capture-24 alone is 82% of the windows. These sources add ~40,200 stream-hours
across placements the labelled corpus barely covers.

## The corpus of record

Run `python -m data.pretraining.build_corpus --plan` for the live table; it is generated
from [`corpus_plan.py`](corpus_plan.py), which is the single source of truth for both the
build and the docs.

| source | take | streams | stream-hours | rate | GB |
|---|---|---:|---:|---:|---:|
| `nhanes` | 3,000 participants x 12 h, motion-aware hours | 1 | 36,000 | 80 Hz | 124.4 |
| `nymeria_xsens` | all 1,100 sequences, 8 of 17 body placements | 8 | 2,400 | 240 Hz | 24.9 |
| `synthetic_imu` | AMASS + Motion-X++ (+ 100STYLE), 8 virtual placements | 8 | 1,800 | 60 Hz | 4.7 |
| **total** | | | **40,200** | | **154.0** |

The `channels` column describes measured channels. The grid stores six physical slots for every
source, zero-padding and masking unavailable gyro, which is why accel-only NHANES costs 124.4 GB
rather than the 62.2 GB that a three-column file alone would occupy.

`embody3d` is wired as a candidate but excluded from the budget: its access terms could not
be confirmed. Confirm the licence before moving it into `CORPUS_PLAN`.

### Why each source is in

- **NHANES** buys breadth of *subjects* on the wrist placement every evaluation set uses.
  Under the sampler's per-subject `n^0.5` tempering, 3,000 people at 12 hours is worth far
  more than 500 people at a full week.
- **Nymeria** is the only source anywhere with real IMU on head, torso, both arms and a leg
  *simultaneously*, hardware-synchronised, so the same motion is observed from three device
  classes at once. That is what a cross-configuration claim needs per hour.
- **Ego-Exo4D** adds the head placement across 740+ subjects. Its disk cost is small; the
  ~1 TB image-free VRS download is the real cost, streamed through convert-then-delete.
- **Synthetic IMU** is a diversity source and never the base. Published evidence
  ([arXiv 2602.11064](https://arxiv.org/abs/2602.11064)) is that large-scale mocap
  pretraining alone gives only marginal gains because of the sim-to-real gap, and that
  synthetic data helps mainly when *mixed* with real data.

## Two storage decisions

**Grids are `float16`.** Halving grid bytes is what makes tens of thousands of stream-hours
fit on one disk, and it costs nothing the model can see: the frontend reads band energies
below ~14 Hz, and float16 resolves ~0.001 g at 1 g, exactly the precision NHANES itself
publishes. float16 rather than a scaled int16 deliberately — both are two bytes, but a
reader that forgets to dequantize an int16 grid trains on values wrong by the scale factor,
while a reader that forgets float16 gets correct values in a narrower dtype. Every read path
in the repo already ends in `np.asarray(..., dtype=np.float32)`, which upcasts transparently.
Declared per dataset as `"grid_dtype": "float16"` in `metadata.json`.

**Aria IMUs are stored at 200 Hz, not their native 800 Hz / 1 kHz.** The frontend's highest
analysis frequency is ~14 Hz, so 200 Hz is already an order of magnitude above anything the
model reads and the native rate would cost 4-5x for content discarded in the first layer.
The native rate is preserved in the session manifest and travels with each window as
`source_rate_hz`, so the tokenizer still knows what was actually acquired. Xsens stays at its
native 240 Hz; resampling it would add an interpolation artefact for no saving worth having.

**Source windows are eight seconds.** Converters truncate only at real session/gap boundaries, and
the corpus grid stage passes the same duration explicitly. Historical labelled grids remain six
seconds; the JEPA trainer rejects a label-free grid built under that old contract.

## Usage

```bash
python -m data.pretraining.build_corpus --plan
python -m data.pretraining.build_corpus --stage fetch   --datasets nhanes --yes
python -m data.pretraining.build_corpus --stage convert --datasets nhanes
python -m data.pretraining.build_corpus --stage grids   --datasets nhanes
```

Each stage is idempotent on its own outputs, so re-running after an interruption is always
safe and never re-downloads. Fetching requires `--yes` per run: several of these sources move
hundreds of gigabytes in transit, and that should never start as a side effect of inspecting
the plan. Licence-gated fetchers refuse to run until the credentials or URL manifest exist;
nothing here bypasses a licence.

Per-source access steps live in each subdirectory's `README.md`.

## Adding a source

A new source needs, in its own subdirectory: `fetch.py` (bounded, deterministic, resumable),
`convert.py` (writes the session contract below), `metadata.json`, `README.md`, and an entry
in `corpus_plan.py`. It also needs a `StreamSpec` in
[`deployment_policy.py`](../scripts/curate/deployment_policy.py) and a reference entry under
[`references/datasets/`](../../references/datasets) — a local publication or official
protocol document, not just a URL.

### Session contract

```
<dataset>/sessions/<session_id>/data.parquet
    timestamp_sec  float64  seconds from session start, monotonic, uniform 1/rate spacing
    acc_x/y/z      float32  units g, gravity PRESENT, device frame
    gyro_x/y/z     float32  units rad/s, device frame        (omit all three if no gyro)
    subject        string   stable subject id, for subject-disjoint splits
<dataset>/labels.json    {session_id: ["__unlabeled__"]}
<dataset>/manifest.json  dataset_name, source, num_subjects, sampling_rate_hz, channels,
                         unit, gravity_state, phase_a_only, note
<dataset>/metadata.json  dataset, display_name, sampling_rate_hz, pre_windowed,
                         streaming_grid, grid_dtype, role, phase_a_only, channels,
                         core_channels, placement, activities, num_subjects, note
```

`session_id` must contain a token naming its stream (`..._xsens_pelvis`, `..._aria_head`,
`..._virt_rthigh`), because `deployment_policy.session_contains` routes sessions to streams
by substring. Synthetic streams carry a `virt_` prefix so they can never be mistaken for real
measurements downstream.
