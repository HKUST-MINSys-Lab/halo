# Synthetic virtual IMU from motion capture

Label-free Phase-A pretraining data, synthesised by simulating what a strapdown
IMU bolted to a moving body would have recorded — the technique behind
TransPose, DIP, PIP, IMUPoser and UniMTS.

```
fetch.py         obtain the corpora + gated body models (100STYLE auto, rest manual)
synthesis.py     the virtual-IMU physics. Pure, no I/O, fully unit-tested
smpl_adapter.py  SMPL / SMPL-H forward pass -> vertex + joint trajectories
convert.py       drive synthesis over a corpus, emit HALO sessions
```

Tests: `tests/test_pretrain_synthetic_imu.py` (offline, needs no mocap and no
body model — the physics is asserted against motions with closed-form answers).

---

## Read this before using the data

> Darwish, Nicholson & Doherty (2026), *arXiv:2602.11064*, find that large-scale
> mocap-derived pretraining gives only **marginal** gains on its own, because of
> the sim-to-real gap, and that synthetic data helps mainly when **mixed with
> real recordings**.

So: this module is a **diversity source**, never the base of the corpus. It buys
placement coverage and motion variety that our real corpus is thin on
(`virt_rupperarm`, `virt_sternum`, unusual gaits). It does not buy scale in any
sense that substitutes for real free-living data. Every session it writes is
marked `synthetic: true`, `phase_a_only: true`, `__unlabeled__`, and carries a
`virt_` stream token, so a mix ratio is always something you choose explicitly
rather than something that happens to you.

What the simulation does **not** model: soft-tissue artefact and strap slop
(the dominant real-world error at the thigh and upper arm), sensor bandwidth and
saturation, real MEMS noise colour and temperature drift, magnetic disturbance,
clock drift between units, and mocap-solver error itself. The optional realism
knobs (below) address only the easiest of these, and are off by default.

---

## Sources

| Source | Size | Format | Licence | Access | Status |
|---|---|---|---|---|---|
| **AMASS** | ~40 h, 344 subjects, 11k motions | SMPL-H, mixed rates | MPG non-commercial research; **no redistribution**, single-user | register at [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) | manual, primary |
| **Motion-X++** | 181 h, 120.5k sequences | SMPL-X, 30 fps | CC BY-NC-SA 4.0 annotations + each sub-dataset's own licence | Google Form via [IDEA-Research/Motion-X](https://github.com/IDEA-Research/Motion-X) | manual, secondary |
| **100STYLE** | ~19 h, 100 styles, >4M frames | BVH, 60 fps | **CC BY 4.0** | [Zenodo 8127870](https://zenodo.org/records/8127870) | **auto** |
| **Embody 3D** | 500 h, 439 participants, 54M frames | — | Meta "XRCIA" (text unverified) | release form at [meta.com](https://www.meta.com/emerging-tech/codec-avatars/embody-3d/) | **placeholder, refuses to run** |

`python -m data.pretraining.synthetic_imu.fetch --list` prints the exact steps
for each. Everything obtained is recorded in `downloads/fetch_manifest.json`.

### 100STYLE is the always-available source

It is the only one that is (a) openly licensed and (b) needs **no body model**:
BVH already carries a posed skeleton, so forward kinematics is enough. That
makes it the smoke-test path — you can go from a clean checkout to real
synthetic sessions with one command and no registrations.

Its file names are resolved from the Zenodo REST API at run time rather than
hardcoded. Zenodo answered 403/504 to non-browser clients when this module was
written, so the file list could not be verified offline, and a hardcoded
The fetcher resolves the declared `100STYLE.zip` BVH asset against the live
Zenodo record before downloading it. It deliberately skips the separate
14.8 GB labelled-data archive because this label-free converter does not use it.

### Embody 3D

Investigated 2026-09-09. **Confirmed**: the dataset is released; official
tooling is [facebookresearch/embody-3d](https://github.com/facebookresearch/embody-3d)
(code under CC BY-NC 4.0), whose `src/download.py` consumes 21 per-user signed
links you receive after submitting a release form on the Meta page; the repo's
README states the **dataset** is licensed under Meta's *XRCIA* licence.

**Not confirmed**: the XRCIA licence text itself. Search results describe it as
a non-commercial research licence, but no canonical text could be retrieved.

Because the terms are unread and no data is on disk to reveal its schema,
`convert.iter_embody3d` raises `NotImplementedError`. It is a documented
placeholder, not a loader. Wiring it needs someone to read XRCIA first.

---

## The physics

Notation: `p[t]` is the world position of the sensor site (m); `R[t]` is the
segment orientation, rotating sensor-frame vectors into the world frame;
`G = 9.80665 m/s²`; `dt = 1/rate`; `up` is the world up unit vector (`z` for
AMASS/SMPL, `y` for BVH — a per-source setting, not a global assumption).

**1 — World acceleration**, symmetric second difference with stride `n`:

```
a_world[t] = ( p[t-n] - 2 p[t] + p[t+n] ) / (n·dt)²
```

**2 — Specific force.** An accelerometer measures the force holding its proof
mass off the case, i.e. `a - g_vec` with `g_vec = -G·up`, so gravity is *added*:

```
f_world[t] = a_world[t] + G · up
```

| motion | `a_world` | reading |
|---|---|---|
| stationary | `0` | `+G·up` → **1 g** |
| free fall | `-G·up` | `0` → **0 g** |
| constant velocity | `0` | **1 g** |

The free-fall case is the classic sign error: getting it backwards gives 2 g,
and flipping the gravity vector gives −1 g. `test_free_fall_reads_zero_g_not_two_g`
pins all three.

**3 — Into the sensor frame, and into g:**

```
acc[t] = Rᵀ[t] · f_world[t] / G        [dimensionless, units of g]
```

**4 — Angular velocity.** With `R` mapping sensor → world,
`R[t]ᵀ R[t+s] = exp([ω]ₓ · s·dt)` where `ω` is in the **sensor** frame:

```
ω[t] = log( R[t]ᵀ R[t+s] ) / (s·dt)     [rad/s, sensor frame]
```

using the rotation-matrix logarithm (axis-angle of the relative rotation). The
default is the centred form `log(R[t-s]ᵀ R[t+s]) / (2s·dt)` — the same estimator
evaluated symmetrically about `t`, so it carries no half-sample time shift.
Sign: rotating the body about `+axis` at `+ω` reads `+ω` on that axis
(`test_constant_rotation_recovers_rate_and_axis`).

### Smoothing: `smooth_n`, and why the default is 2

Differencing twice is a high-pass with gain `(2/dt)²` at Nyquist; run it on raw
60 fps mocap and the output is solver jitter, not body motion. The standard fix
(TransPose's `preprocess.py`) is to **widen the stencil** rather than pre-filter:
a stride-`n` central difference is algebraically `(low-pass) ∘ (2nd difference)`
with a symmetric — hence exactly **zero-phase** — FIR kernel. Its response
relative to an ideal `-ω²` differentiator is `sinc²(ω·n·dt/2)`, first null at
`rate / n`:

| `smooth_n` | 1 Hz | 2 Hz | 5 Hz | 10 Hz | 14 Hz | first null |
|---|---|---|---|---|---|---|
| 1 (raw) | 0.999 | 0.996 | 0.977 | 0.912 | 0.833 | 60 Hz |
| **2 (default)** | **0.996** | **0.985** | **0.912** | **0.684** | **0.460** | **30 Hz** |
| 4 (TransPose) | 0.985 | 0.943 | 0.684 | 0.171 | 0.005 | 15 Hz |

**We depart from TransPose's `n = 4` deliberately.** At 60 Hz it puts the first
transfer null at 15 Hz — *inside* the ~14 Hz band our tokenizer analyses — and
has already thrown away 32% of the amplitude at 5 Hz. TransPose can afford that
because it feeds a pose estimator whose content is essentially sub-5 Hz; we
cannot. `n = 2` moves the null to 30 Hz (exactly Nyquist, where only aliased
solver noise lives), stays within 0.4% of ideal at 1 Hz, and still cuts the
stencil's white-noise gain 4× relative to `n = 1`.

Pass `--smooth-n 4` for TransPose parity, `--smooth-n 1` for the raw central
difference. `synthesis.smoothing_response()` computes the table above, and
`test_smoothing_default_keeps_the_null_outside_our_analysis_band` asserts it, so
none of these numbers has to be taken on trust.

**Edges are trimmed, not zero-padded.** TransPose pads the first and last frames
with zeros, fabricating a 0 g reading — physically a free fall — at every
sequence boundary. We drop `pad = max(smooth_n, gyro_stride)` samples from each
end instead, so no emitted sample is a padding artefact.

### Output rate: 60 Hz

Above our ~14 Hz `f_max` with margin, and the native rate of most mocap
(100STYLE is exactly 60 fps; AMASS sub-datasets are 60/100/120 fps), so the
common case is a downsample or a no-op rather than an invention of detail.
Resampling happens on the **trajectory**, before differencing — linear on
positions, slerp on orientations — so the stencil always sees a uniform `dt`
and the stated smoothing bandwidth is the real one. Differencing first and
resampling the derivative would alias.

### Realism knobs — all OFF by default

`RealismConfig(enabled=False)` by default, so with stock settings synthesis is a
**deterministic** function of the trajectory and touches no RNG. Enabling
(`convert.py --realism`) adds white noise (0.01 g / 0.01 rad/s), a per-session
constant bias (0.02 g / 0.01 rad/s), and optional placement displacement and
mounting misalignment. `seed` fixes the draw. These are order-of-magnitude
consumer-MEMS figures, not a calibration of any part, and they do not model the
error that actually dominates (soft-tissue artefact).

---

## Placement provenance

Eight placements, matching what our real corpus covers. **Vertex indices were
verified on 2026-09-09 by reading the released source files; the two that have
no published index say so and assert none.**

| placement | stream token | SMPL vertex | SMPL joint | source |
|---|---|---|---|---|
| `head` | `virt_head` | 411 | 15 (head) | TransPose `preprocess.py` L34–35, `vi_mask[4]`/`ji_mask[4]` ✅ |
| `sternum` | `virt_sternum` | *none* | 9 (spine3) | **unverified** — no published mask; joint-origin fallback |
| `pelvis` | `virt_pelvis` | 3021 | 0 (pelvis) | TransPose L34–35, `vi_mask[5]`/`ji_mask[5]` ✅ |
| `lforearm` | `virt_lforearm` | 1961 | 18 (left elbow) | TransPose L34–35, `vi_mask[0]`/`ji_mask[0]` ✅ |
| `rforearm` | `virt_rforearm` | 5424 | 19 (right elbow) | TransPose L34–35, `vi_mask[1]`/`ji_mask[1]` ✅ |
| `rupperarm` | `virt_rupperarm` | *none* | 17 (right shoulder) | **unverified** — no published mask; joint-origin fallback |
| `rthigh` | `virt_rthigh` | 4362 | 2 (right hip) | IMUPoser `1. preprocess_all.py` L33–34, `vi_mask[3]`/`ji_mask[3]` ✅ |
| `rshank` | `virt_rshank` | 4662 | 5 (right knee) | TransPose L34–35, `vi_mask[3]`/`ji_mask[3]` ✅ |

The two published masks, verbatim:

```python
# github.com/Xinyu-Yi/TransPose, preprocess.py lines 34-35
# (NOT config.py, which contains no such mask)
vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
#          [L forearm, R forearm, L lower leg, R lower leg, head, pelvis]

# github.com/FIGLAB/IMUPoser, "scripts/1. Preprocessing/1. preprocess_all.py" lines 33-34
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
#          same arms/head/pelvis, but leg sensors at the THIGH (hips) not the shank
```

Caveats you should not skip:

* **Vertex choice is not standardised.** DIP (`eth-ait/dip18`,
  `data_synthesis/genSynData.py` L27) uses a *different* set —
  `[1962, 5431, 1096, 4583, 412, 3021]` — sharing only the pelvis with
  TransPose. We follow TransPose, plus IMUPoser for the thigh, and say so
  rather than pretending there is one canonical answer.
* **`sternum` and `rupperarm` have no published vertex.** They fall back to the
  joint origin (`smpl_vertex = None`), which sits *inside* the body rather than
  on the skin — a few centimetres of lever arm that will slightly under-state
  rotational acceleration at those two sites. This is a documented
  approximation. A vertex index eyeballed off a mesh render is not a citation,
  and inventing one would move a sensor by an unknown amount.
* **The BVH path uses joint frames, not mesh vertices**, for *every* placement:
  BVH gives no body surface. 100STYLE-derived sensors therefore all sit at joint
  centres. Placements are resolved by joint name
  (`Placement.bvh_joint_candidates`), and any that a given skeleton lacks are
  simply skipped — 100STYLE's skeleton yields fewer than eight.
* **Motion-X++ is SMPL-X but is skinned with SMPL-H here.** The first 66 entries
  of each frame (root + 21 body joints) transfer directly; the two models share
  a body topology but not a mesh, so expect millimetre-level differences at the
  sampled sites. Recorded in `manifest.json` as
  `smplx_approximated_by_smplh`.

---

## Output contract

Identical in shape to every other converted dataset in the repo.

```
sessions/<session_id>/data.parquet
    timestamp_sec  float64  seconds from session start, uniform 1/60 s
    acc_x,  acc_y,  acc_z   float32   units g, gravity PRESENT, sensor frame
    gyro_x, gyro_y, gyro_z  float32   units rad/s, sensor frame
    subject        string   stable subject id, e.g. "amass_CMU_01"
labels.json    {session_id: ["__unlabeled__"]}
manifest.json  {dataset_name, source, num_subjects, sampling_rate_hz, channels,
                unit, gravity_state, phase_a_only, synthesis, note}
metadata.json  {dataset, display_name, sampling_rate_hz: 60.0, pre_windowed: false,
                streaming_grid: true, role: "pretrain_scale", phase_a_only: true,
                synthetic: true, activities: [], num_subjects: null, channels,
                core_channels, placement, note}
```

Session ids are `synthimu_<subject>_<sequence>_virt_<placement>`, e.g.

```
synthimu_amass_CMU_01_run01_virt_rthigh
```

One session per (sequence × placement). The **`virt_` prefix on the stream token
is load-bearing**: no downstream policy, grid or eval should ever confuse a
simulated stream for a measured one, and `grep -r virt_` finds every one of them.

**Short sequences are dropped, not concatenated.** Anything under
`--min-seconds` (default 8 s, our JEPA source window) after edge trimming is discarded.
Concatenation was the other option and is rejected on purpose: splicing two
unrelated clips creates a position discontinuity, and the very next stage is a
second derivative, which turns that into a multi-hundred-g impulse. Those
impulses would be the largest values in the corpus, they are pure artefact, and
a masked-reconstruction objective would happily spend capacity on them. Losing
short clips is cheap; teaching the model a fake transient is not.

---

## Commands

```bash
# what each source needs, and where to put it
python -m data.pretraining.synthetic_imu.fetch --list

# the one auto-downloadable source (CC BY 4.0, no registration)
python -m data.pretraining.synthetic_imu.fetch 100style
#   ... then unzip the archives inside downloads/100style/

# manual sources: prints the steps and records them in the fetch manifest
python -m data.pretraining.synthetic_imu.fetch amass motion_x body_models

# smoke run: 20 sequences, no body model needed
python -m data.pretraining.synthetic_imu.convert --sources 100style --sequences 20

# the real thing, once AMASS + SMPL-H are in place
python -m data.pretraining.synthetic_imu.convert \
    --sources amass 100style \
    --placements head sternum pelvis lforearm rforearm rupperarm rthigh rshank

# TransPose-parity smoothing, or raw central difference
python -m data.pretraining.synthetic_imu.convert --sources 100style --smooth-n 4
python -m data.pretraining.synthetic_imu.convert --sources 100style --smooth-n 1

# sensor noise / bias / placement jitter (off unless asked)
python -m data.pretraining.synthetic_imu.convert --sources 100style --realism

# tests (offline; no mocap, no body model)
python -m pytest tests/test_pretrain_synthetic_imu.py -q
```

Generated downloads, sessions, grids, and manifests under `data/pretraining/`
are ignored by the repository. The eight declared virtual placements are wired
as `phase_a_scale` streams in `deployment_policy.py`; build them explicitly with
`python -m data.scripts.build_grids --dataset synthetic_imu --alignment native
--window-seconds 8`. A source skeleton may expose only a subset of placements.
For example, the open 100STYLE BVH release resolves head, sternum, and pelvis;
the other declared streams remain empty until a source with those joints is
converted. Empty streams do not satisfy corpus materialization checks.

---

## Citations

* Yi, Zhou & Xu. *TransPose: Real-time 3D Human Translation and Pose Estimation
  with Six Inertial Sensors.* SIGGRAPH 2021. — the synthesis recipe and five of
  the six vertex indices.
* Huang et al. *Deep Inertial Poser (DIP).* SIGGRAPH Asia 2018. — the original
  AMASS→virtual-IMU pipeline; a different vertex set.
* Mollyn et al. *IMUPoser.* CHI 2023. — the thigh vertex index.
* Mahmood et al. *AMASS: Archive of Motion Capture as Surface Shapes.* ICCV 2019.
* Lin et al. *Motion-X / Motion-X++.* NeurIPS 2023 / arXiv:2501.05098.
* Mason et al. *Real-Time Style Modelling of Human Locomotion (100STYLE).* 2022.
* McLean et al. *Embody 3D.* arXiv:2510.16258, 2025.
* **Darwish, Nicholson & Doherty. arXiv:2602.11064, 2026.** — the sim-to-real
  finding that makes this a diversity source rather than the corpus base.
