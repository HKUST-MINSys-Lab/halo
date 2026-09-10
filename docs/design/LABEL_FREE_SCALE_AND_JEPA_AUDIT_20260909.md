# Label-free pretraining: what the objective actually is, how it compares to best practice, and what data exists

> **Historical audit snapshot.** This document describes the retired masked-JEPA/VICReg default
> that motivated the replacement. It is preserved as evidence, not as configuration guidance. The
> implemented objective is [`JEPA_PRETRAINING_OBJECTIVE.md`](JEPA_PRETRAINING_OBJECTIVE.md).

> Written 2026-09-09 from a read of `training/tokenizer/losses_repr.py`, `training/tokenizer/pretrain.py`,
> `model/tokenizer/encoder.py`, the 2026-08-18 recovery report, a 13-paper primary-source literature
> pass (notes: `JEPA_LITERATURE_NOTES_20260909.md`, [V] = read from the PDF, [S] = secondary), and a
> four-way data survey (population cohorts, egocentric/industry, mocap, non-IMU kinematics).
> Nothing here was trained or launched.

## 1. The objective as coded (not as remembered)

Per step, one six-second context, one-second patches, one token per (patch, sensor):

| leg | what runs | default |
|---|---|---|
| student | clean tokens; a contiguous block of round(0.5·P) patches (all sensors) replaced by a learned [MASK] before descriptor conditioning; 25 % of steps also hide one whole co-located sensor | on |
| teacher | EMA copy (decay 0.984 fixed; cosine ramp to 1.0 exists, unmeasured) on the clean view, full descriptor | on |
| latent prediction | per-token 2-layer MLP predictor (d→d→d) on the student's final tokens; loss 1 − cos against teacher final tokens at masked positions; no layer-norm / instance-norm of targets, no top-K layer averaging | on, weight 1 |
| collapse control | VICReg 25/25/1 on the expander of the POOLED window embedding of two augmented views, half the weight on the sensor-isolated retrieval rows | on, weight 1 |
| decode to physics | `mae_head` MLP on the masked student's tokens → MSE against detached `filterbank.analyze` band features at hidden positions | **off** (`mae_weight = 0`) |
| weighting | one-shot gradient-norm calibration after warmup (JEPA = 45 % share), then frozen | on |

Three facts that differ from the "U-shaped" description of the design:

1. **The decode leg is not running.** It is off by default, its config comment and the loss docstring
   frame it as a *replacement* for JEPA (`--jepa-weight 0 --mae-weight 1`), and the only measurement
   (2026-08-18) was that swap: +0.0027, under the 0.012 noise floor. It was never measured *stacked*
   on top of the latent term, which is the configuration the U-shape describes.
2. **It cannot be enabled on the continuous or multi-span frontend**: `pretrain.py` rejects
   `--mae-weight` there because the head reads `filterbank.proj.in_features`. The anti-collapse
   mechanism actually in force is EMA teacher + pooled VICReg + the `jepa/margin` telemetry.
3. **"Latent state transitions"** is masked-block prediction with bidirectional context, not causal
   next-state prediction. There is no future-tail branch (README).

Also relevant to any scale-up: every compact-engine / neighbours checkpoint that produced the current
best results carries `phase_a_checkpoint=None` (see the header of `training/compare/train.py`). The
Phase-A objective is not in the pipeline that beat UniMTS. Its best measured contribution is +0.057
transfer over a random-init trunk, and the "old-good 4k" checkpoint (0.8577) is still unbeaten by
any later recipe. Scaling the data without fixing the objective would scale a term whose value has
not been re-established.

`training/tokenizer/README.md` still says physical-feature reconstruction "has been removed"; the code
keeps it behind `--mae-weight`.

## 2. Best-practice comparison

Scorecard (full citations and tables in the literature notes):

| choice | verdict | strongest evidence |
|---|---|---|
| teacher sees the clean input, mask on the student side | aligned, high leverage | I-JEPA Table 11: 67.3 vs 56.1 [V] |
| cosine latent loss | aligned; loss family is a minor lever | data2vec Table 7: L1/L2/smooth-L1 within 0.5 [V] |
| ~50 % contiguous temporal mask | aligned for temporal data | data2vec 2.0 speech R = 0.5, B = 5 (Table 9) [V]; CAE 50 % best (Table 7) [V]; SimMTM 50 > 75 [S] |
| hand-crafted physical target rather than raw signal | aligned | MaskFeat Table 2: HOG 83.6 > pixel 82.5 > scratch 81.8 [V] |
| latent prediction PLUS a decode head (the U) | precedented and positive | CAE Table 5: (+dec,+align) 64.1/73.8 vs neither 60.3/71.2 [V]; MAGE Table 7 C+R 77.1 vs 73.3 [S]; V-JEPA's "no pixels" ablation is latent *instead of* pixels, not this configuration |
| VICReg 25/25/1 | aligned, flat region | VICReg Table 7: 5–50 gives 68.1–68.6 [V] |
| whole-sensor masking as a minority event | defensible only as minority | LSM Table 5: sensor-structured is the worst sole strategy [V]; LSM-2 mixes it in for missingness robustness [S] |
| **one** contiguous block | **deviates** | I-JEPA Table 10: 1 block 9.0 → 4 blocks 54.2 [V]; V-JEPA Fig. 8b several small > one large [V] |
| **no target normalisation** | **deviates, in the regime data2vec flags** | data2vec §6: collapse "more likely for modalities where adjacent targets are very correlated and longer spans need to be masked… we address this by normalizing target representations over the sequence" [V]; data2vec 2.0 IN→AVG→LN [V] |
| predictor = per-token MLP, no bottleneck | acceptable (data2vec-style), not I-JEPA-style | I-JEPA Tables 12/14: deep + narrow predictor helps [V]; BYOL's predictor is also a 2-layer MLP |
| VICReg on pooled rows only | gap: blind to per-token collapse | VICReg Table 4: Var/Cov on top of EMA+SG+predictor ≈ neutral (69.3 → 69.5) [V]; the masked-prediction collapse (all masked positions emit the same vector) leaves the pooled row well-conditioned |
| unnormalised band-energy target beside a latent target | risk when re-enabled | MaskFeat Table 13: pixel+HOG 82.3 < HOG 83.6, cause = normalised-vs-unnormalised target mismatch; Table 12a: no normalisation −1.4 [V] |
| decode gradient coupled into the latent path | ablate on/off | CAE (+) [V] vs DINO-WM Table 7 0.92 → 0.80 (−) [S, unverified] |
| one-shot gradient-norm calibration | no named precedent | GradNorm "static" [S]; Xin et al. adaptive weights barely move [S]; label as heuristic |
| last-layer targets | acceptable; cheap win available | data2vec Fig. 2 top-K averaging "very pronounced for speech" [V] |

Note on the literature pass: the reviewer was briefed that the design has no predictor; it does (the
MLP above), so the BYOL "0.2 % without predictor" finding in the notes does not apply. The remaining
predictor point is only that I-JEPA prefers a narrow bottleneck.

### What to change before scaling data (cheap, ordered)

1. **Normalise targets**: instance-normalise teacher tokens over the sequence and average the three
   trunk layers (data2vec). One function in `masked_ema_latent_loss`; removes the collapse mode the
   pooled VICReg cannot see.
2. **Several blocks, not one**: on the multi-span grid (24 + 12 + 6 + 3 tokens per 6 s context)
   there is room for 2–4 blocks per resolution; on the one-second grid the context is too short and
   should lengthen (LSM uses long windows; six patches is the smallest context in this survey).
3. **Re-enable the decode leg properly**: make the target the fixed filterbank analysis regardless of
   frontend (it is parameter-free and computable beside any frontend), log-transform and standardise
   it per window (MaskFeat 12a/13), run it STACKED with a weight sweep (MAGE's curve is unimodal
   around 0.1), and ablate stop-gradient on the decode path (DINO-WM). Re-measure the interpolation
   null at the live 3-patch geometry first; the 2026-08-18 null was measured on single-patch holes.
4. **Add a token-level variance term** at masked positions, or rely on (1); either closes the gap in
   the VICReg placement.
5. Keep the EMA ramp, predictor bottleneck and calibration as knobs; none is load-bearing.

Then re-establish the floor: Phase-A warm start versus random init under the neighbours protocol,
because the best current checkpoints did not use Phase A at all.

## 3. Label-free motion data available

Rate-agnostic frontends make head-mounted 1 kHz IMU, 240 Hz body suits and 60–144 Hz VR trajectories
all admissible without resampling; the fixed filterbank needs only `f_max` coverage.

### Tier 1: open or click-through, large, raw signal

| source | placement / modality | subjects | hours (raw) | rate | access |
|---|---|---|---|---|---|
| **NHANES 2011–14** | wrist accel (ActiGraph GT3X+) | 14,693 | ~2.5 M person-hours | 80 Hz | fully open CDC download |
| **Nymeria** (Meta, ECCV'24) | head Aria + both wrists (miniAria) + 17-IMU Xsens suit | 264 | 300 h | Aria 800 Hz/1 kHz, Xsens 240 Hz | licence click, ~48 h |
| **Ego-Exo4D** | head Aria dual IMU | 740+ | 1,422 h video, IMU throughout | 1 kHz | licence click, ~48 h |
| Ego4D IMU subset | head, GoPro rigs | subset of 931 | ~836 h (secondary figure) | varies | licence click; maintainers flag noise/timestamps |
| **SHL** | phone at hips/bag/torso/hand | **3** | 2,812 h | 100 Hz | registration |
| **PPMI-Verily watch** | wrist accel+gyro, continuous, PD cohort | 149 | >0.5 M person-hours (months per person) | 100 Hz | DUA, no fee |
| **BOXRR-23** | VR head + two hand controllers, 6-DoF | 105,852 | 4.7 M recordings | 60–144 Hz | self-serve DUA, CC BY-NC-SA |
| emg2pose / emg2qwerty | hand pose from mocap + 16-ch sEMG | 193 / 108 | 370 h / 346 h | pose at mocap rate, EMG 2 kHz | open, CC BY-NC-SA |
| CAPTURE-24 | wrist | 151 | 3,883 h | 100 Hz | open (already in corpus) |
| Aria Everyday Activities | head | few | 7.3 h | 800 Hz/1 kHz | open |
| OpenPack / WEAR / HuGaDB / Daphnet / ANDY | various | 10–22 each | 7–54 h each | 30–100 Hz | open (OpenPack already in corpus) |

### Tier 1b: mocap for synthetic IMU (the UniMTS / IMUTube route)

| source | scale | format | licence |
|---|---|---|---|
| AMASS | ~40 h, 344 subjects, 11k motions | SMPL-H | non-commercial, registration |
| Motion-X++ | 181 h, 120k sequences | SMPL-X 30 fps | CC BY-NC-SA + sub-dataset licences |
| HuMMan | 60 M frames, 1,000 subjects | SMPL | S-Lab licence |
| 100STYLE | 19 h, 60 fps, 100 locomotion styles | BVH | CC BY 4.0 |
| BEDLAM / BEDLAM 2.0 | 2,311 motions, 271 bodies | SMPL-X | research only |
| CMU, KIT, AIST++, Fit3D, Human3.6M, BABEL, HumanML3D | 5–43 h each | mixed | mixed; HumanML3D inherits AMASS |

Synthetic IMU carries a known domain gap (no soft tissue, ideal mounting); useful as a diversity
source, not a substitute for real wear.

### Tier 2: application + committee, months, but the largest raw wrist corpora that exist

| source | placement | subjects | hours | access |
|---|---|---|---|---|
| **UK Biobank** | wrist, Axivity AX3, 100 Hz | 96,600 valid | ~16 M person-hours | application, fee (reduced ~£500 for students), ~15 weeks |
| NAKO (German National Cohort) | hip ActiGraph | 63,236 valid | ~10.6 M person-hours | portal + committee; raw export unverified |
| China Kadoorie Biobank | wrist AX3, 100 Hz | 22,511 | ~3.8 M person-hours | inter-institutional DAA; raw export unverified |
| Whitehall II | wrist GENEActiv | 4,006 | 9 days each | DPUK, fee |
| Fenland | wrist GENEActiv 60 Hz (subsample only) | ~2,100 | 6 days each | committee |

### Not obtainable as raw signal

Google LSM (40 M h, 165k people), LSM-2, SensorLM (59.7 M h), SensorFM (>2 B h, 5 M people); Apple
AHMS / RelCon (1 B segments, 87k people, 100 Hz); Samsung HiMAE (80k h PPG): all closed. All of Us
Fitbit: derived metrics only. NSRR cohorts (MESA, HCHS/SOL, STAGES): Actiwatch epoch counts, no raw.
NHANES 2003–06 hip: 1-min counts only. GaitRec: force plates, not IMU.

### Recommended order

1. NHANES 2011–14 (open, the largest raw wrist set without an application; 14.7k subjects).
2. Nymeria and Ego-Exo4D licence requests (48 h turnaround; Nymeria is the only source with real
   IMU on head, both wrists and all limbs simultaneously).
3. UK Biobank application through HKUST now, because of the 15-week lead time.
4. BOXRR-23 DUA and emg2pose for hand/arm kinematics; differentiate 6-DoF trajectories into virtual
   accel/gyro the same way mocap is converted.
5. PPMI-Verily for months-long continuous wrist wear on a clinical population.
6. Mocap → synthetic IMU (AMASS, Motion-X++, 100STYLE) as a diversity augmentation once real data
   is in.

Sampler note: the hierarchical sampler caps any dataset at 25 % of draws (n^0.25 tempering), so
NHANES or UK Biobank cannot swamp the corpus; the per-subject n^0.5 tempering also stops SHL's three
subjects from mattering more than their count.

## 4. Concrete pretraining corpus plan (second pass, 2026-09-09)

Measured locally (RTX 4090, `phase_a_a_clean_20260818` and `phase_a_e_clean_long_20260818`):

| quantity | value |
|---|---|
| Phase-A step, batch 1024, fixed frontend, 6 layers | 102 ms |
| reference run, 7,500 steps = 7.68 M windows | 13 min |
| windows per second | ~10,000 = ~1,000 hours of signal per minute |
| multi-span frontend (measured 2.6x in the compare trainer) | ~35 min per reference run |
| native grid format | (N, 600, 6) float32 = 14.4 KB per 6 s window = 8.6 GB per 1,000 h at 100 Hz, gyro slots stored even when absent |
| current corpus | 1.87 M windows = 3,117 h, 18 sources, 157 GB on disk incl. downloads; capture24 is 82 % of windows |
| free disk | 1.0 TB |

Download logistics (details and citations in `DATA_LOGISTICS_NOTES_20260909.md`): NHANES 2011–14 raw is ~2.2 TB
compressed across 14,693 per-participant tar.bz2 archives of hourly CSVs; Ego-Exo4D's Aria VRS without image
streams is 996 GB; BOXRR-23 is 4.7–5.35 TB; Motion-X++ 50 GB; UK Biobank raw is locked to the Research Analysis
Platform (no off-platform bulk download, £500 + VAT student fee, 15-week approval); Nymeria is 80 TB in full and an
IMU-only download group could not be confirmed; PPMI-Verily raw is request-only with undocumented format.

A fetcher and converter for NHANES already exist (`data/datasets/nhanes/`, bounded subset, 8 subjects, listed under
`OPTIONAL_PHASE_A_DATASETS`). The converter documents that 43 % of NHANES windows are below 0.003 g of motion.

### Chosen set, in order

| wave | source | what to take | stream-hours | setup | transient download | gridded disk (current fmt / int16 accel-only) |
|---|---|---|---|---|---|---|
| 1 | **NHANES 2011–14 wrist** | 1,000–2,000 subjects x 48 h spread across wear, motion-stratified (keep all moving windows, cap still ones) | 48k–96k | LOW: scale the existing fetcher to stream download→convert→delete | 150–300 GB | 410–830 GB / 105–210 GB |
| 1 | **Nymeria** | Xsens 17-sensor body stream (240 Hz) + head + both wrist Aria IMUs; all 300 h | ~6,000 (300 h x ~20 streams) | MEDIUM: licence click + MVNX/VRS converter (2–4 days) | Xsens small; Aria VRS large, IMU-only unconfirmed | ~50 GB / ~15 GB |
| 1 | **Ego-Exo4D head IMU** | ~1/5 of takes (~300 h), extract IMU from VRS and delete | ~600 (2 IMUs) | MEDIUM: ego4d CLI + projectaria_tools (2–3 days, mostly download) | ~200 GB per slice | ~5 GB |
| 1 (opt) | SHL | 100 h per position (hand, hip, bag, torso); 3 subjects only, so never more | 400 | LOW–MEDIUM | ~100 GB | ~4 GB |
| 2 | mocap → synthetic IMU (AMASS, Motion-X++, 100STYLE) | 6–8 virtual placements (TransPose vertex set) | ~1,500–2,000 | MEDIUM–HIGH: synthesis + validate the gap on DIP-IMU/TotalCapture (3–5 days) | ~100 GB | ~15 GB |
| 3 | UK Biobank | only with a cloud budget: training must run on the RAP | — | HIGH: application, 15 weeks, RAP compute | none locally | none locally |
| defer | BOXRR-23, PPMI-Verily, Ego4D IMU | domain (VR), request-only, noisy | — | HIGH | 5 TB / unknown / — | — |

Wave 1 totals: ~55k–105k stream-hours, i.e. 18–34x the current corpus. It does **not** fit the local disk in the
current float32-six-channel format (410–830 GB for NHANES alone against 1.0 TB free). Step 0 is therefore a compact
grid format: int16 (or float16) with only the channels a stream has. At 80 Hz accel-only that is 2.9 KB per window,
1.7 GB per 1,000 h, and the whole wave-1 corpus lands at 150–250 GB.

### Why this set

* NHANES supplies subject count (thousands, versus 151 in capture24) on the placement every eval set uses.
* Nymeria is the only source with real IMU on head, both wrists and every limb at once; it carries the cross-config
  claim per hour better than anything else available.
* Ego-Exo4D adds a placement and a rate (1 kHz) the corpus does not have, from hundreds of subjects.
* The hierarchical sampler caps a dataset at 25 % of draws, so NHANES cannot swamp the mix; SHL's three subjects
  are the reason it is capped at a slice, not the hours it offers.
* Synthetic IMU waits for wave 2 because its value is diversity, not scale, and its domain gap needs measuring.

### Time

| item | estimate |
|---|---|
| NHANES download + convert, 2,000 subjects | ~1 day download, ~0.5–1 day CPU conversion with the pool; converter is pandas over hourly CSVs |
| Nymeria / Ego-Exo4D converters | 2–4 days each of engineering, download in the background |
| compact grid format + loader change | 1–2 days incl. tests |
| training, fixed frontend, 50k steps = 51 M windows ≈ one pass over 85k h | 1.4 h |
| training, multi-span frontend, 50k steps | ~3.6 h |
| training, 200k steps (~4 passes) | 5.6 h fixed / ~15 h multi-span |

Training cost is set by the step budget, not the corpus: the sampler draws with replacement, and at
~1,000 h of signal per minute the GPU is not the bottleneck until the corpus exceeds a few hundred thousand hours.
The loader is: 10k windows/s x 14.4 KB = 144 MB/s of random memmap reads, which the NVMe sustains but the page
cache will not hold for a 200 GB corpus, so the compact format also matters for I/O.

### Caveats specific to the objective

* Free-living wrist is mostly still; masked prediction of a still window is trivial and will dominate the loss
  unless still windows are capped at conversion or the sampler weights by motion. LSM used random 80 % masking on
  minute-level features; nothing in the survey trained a JEPA on raw free-living wrist at this scale.
* Rates of 240 Hz and 1 kHz enter the fixed filterbank only through `f_max`; the continuous / multi-span frontend
  evaluates kernels at the real sample offsets and needs no resampling. Check the collate's rounded boundaries at
  1 kHz before assuming the patch grid holds.
* None of the wave-1 sources overlap the eight-dataset primary evaluation cohort.

## 5. Verification pass before committing (2026-09-09, Scite + Consensus + browser)

User's chosen set: NHANES subset (~20–30 GB), Nymeria, Ego-Exo4D, mocap→synthetic IMU. UK Biobank dropped.

### Pages read in the browser

* **Nymeria GitHub `nymeriaplus/layout.py`** resolves the open question from §4: download groups are per
  (sequence, group), and two of them are exactly what we need without any video:
  `body_raw` = `body/xdata.{healthcheck,mvnx,npz}` (Xsens 17-IMU stream) and `timesync_and_imu` =
  per-recording `data/motion.vrs` for head, left wrist, right wrist, observer (IMU-only VRS). Sequences are
  selected on the Project Aria site (filter by location/date/scenario), a URL JSON is issued, and
  `nymeriaplus-download -i url.json -o out -y` fetches only the chosen groups. Licence CC BY-NC 4.0, email
  registration. 1,100 sequences, ~80 TB only if everything is taken. NymeriaPlus (2026) adds SMPL/MHR motion.
* **Ego-Exo4D CLI docs**: parts table confirms `take_vrs_noimagestream` = 995.6 GB (IMU lives inside VRS;
  no separate IMU part); `--uids` / `--universities` / `--splits` filters allow a slice; licence agreement
  takes ~2 days; downloads via AWS credentials.
* **CDC PAX80_G documentation**: 6,917 participants (2011–12) + 7,776 (2013–14); one tar.bz2 per participant
  holding up to 194 hourly gzipped CSVs (288,000 rows/hour, columns timestamp/X/Y/Z in g, ±6 g range) plus a
  QC-flag CSV (0.26 % of data flagged); ~1.04 TB compressed for 2011–12; non-dominant wrist in ~99 % of cases;
  96 % wore to day 9. Matches the repo's existing fetcher.

### Literature check on the choice

* **Inertia-1** (Xu et al. 2026, arXiv 2607.06617, 18.2 M h): uses **NHANES as its primary pretraining
  source** ("raw high-frequency accelerometer data from 14,000 participants"), adds UK Biobank only for
  scaling studies, finds "data volume and diversity provide a more reliable path to improved transfer than
  increasing model size" and gains from mixing sources. Independent confirmation of the NHANES anchor.
* **Darwish, Nicholson & Doherty 2026** (arXiv 2602.11064): large-scale mocap pretraining "yields only
  marginal gains due to domain mismatch"; synthetic helps when **mixed with real** or scaled far. Keeps
  synthetic IMU in wave 2 as a diversity source, never the base.
* **Nymeria is already the IMU testbed of choice** in 2026 work: MARIO (inertial odometry, "5x larger than
  datasets used in prior work") and WHIP / "Towards Real-World Wearable Motion Reconstruction" both evaluate
  on it.
* Virtual-to-real is now a crowded line (Wonderwall IMWUT 2026; AnyMo 2026 with physics-grounded simulation
  over dense body-surface placements; WIMUSim; SynHAR; Text2IMU). None of them release a real multi-placement
  corpus, which is why Nymeria matters more than another synthetic pipeline.
* No competing open source was found that beats the set: EMHI (28.5 h), Ego-Elec (9 h), MoVi (6.6 h IMU),
  the 17-IMU layout dataset (30 subjects) are all small. IF-D is a vehicle-mounted IMU, not human.
* **New candidate for the mocap wave**: **Embody 3D** (Meta Codec Avatars, arXiv 2510.16258): 500 h of
  tracked 3D motion from 439 participants, 54 M frames, hands + body shape. Ten times AMASS. Release terms
  not yet verified (no GitHub repo under the obvious name); check before relying on it.

### Decision

Set confirmed: NHANES (bounded subset via the existing fetcher, ~2,000 subjects × 24–48 h → ~25 GB in the
compact format), Nymeria (`body_raw` + `timesync_and_imu` groups, all sequences), Ego-Exo4D
(`take_vrs_noimagestream` slice by university/uids), then synthetic IMU from AMASS + Motion-X++ (+ Embody 3D
if its licence allows). UK Biobank dropped.

## 6. The 120 GB corpus (2026-09-09)

Cost model: compact int16 grids, only the channels a stream has, six-second windows, no overlap.
Bytes per stream-hour = 3600 × rate × channels × 2. Aria IMUs (800 Hz / 1 kHz) are stored after anti-aliased
decimation to 200 Hz with `source_rate_hz` kept at the native value; the frontend's f_max is ~14 Hz, so nothing
above 100 Hz Nyquist is ever consumed and storing it would cost 5x for zero information. Xsens stays at its
native 240 Hz.

| source | what | stream-hours | GB | why this much |
|---|---|---|---|---|
| NHANES 2011–14 wrist, 80 Hz accel | ~3,500 subjects × 12 h spread across wear; still windows capped so ~2/3 kept | ~28,000 kept | 48 | subject count is the value; 12 h per person covers day and night; more subjects beats longer wear under the n^0.5 subject tempering |
| Nymeria Xsens, 240 Hz accel+gyro | all 1,100 sequences (264 subjects, 300 h), 8 of 17 placements: head, sternum, pelvis, L/R forearm, R upper arm, R thigh, R shank | 2,400 | 25 | every placement the eval sets use, on the same 300 h of motion; the other 9 sensors are near-duplicates (left/right symmetry) |
| Nymeria Aria head + both wrists, 200 Hz accel+gyro | one IMU per device, all sequences | 900 | 8 | real glasses and wristband IMUs synchronised with the suit |
| Ego-Exo4D head Aria, 200 Hz accel+gyro | all 1,422 h of takes, one IMU per device | 1,400 | 12 | the head placement at 740+ subjects; the download (996 GB VRS) is the cost, not the disk |
| synthetic IMU from AMASS + Motion-X++ (+ Embody 3D if licensed), 60 Hz, 8 virtual placements | all sequences | 1,800 (5,800 with Embody 3D) | 5 (15) | diversity of motion vocabulary; wave 2, mixed in at low weight |
| SHL phone, 100 Hz accel+gyro | 150 h × 4 positions (hand, hip, bag, torso) | 600 | 3 | four phone positions; capped because it is three people |
| existing 18-source corpus, re-encoded | as is | 3,100 | 8 | keeps the current baseline comparable |
| **total** | | **~38,000 (~42,000 with Embody 3D)** | **~109 (~119)** | ~10 GB slack for the NHANES still-window cap landing above 2/3 |

The raw downloads (NHANES ~500 GB transient, Ego-Exo4D ~1 TB transient, Nymeria groups ~100 GB) are streamed
through convert-then-delete; only the grids stay. The current float32 grids (157 GB) are replaced, not kept.

Sampler consequence: NHANES holds ~70 % of the hours but the 25 % dataset cap and n^0.25 tempering bring its
draw share to ~25 %; Nymeria's 11 streams and Ego-Exo4D each land around 15–20 %; synthetic and SHL are minor.
Training: 100k steps × 1,024 = 102 M windows ≈ 170k stream-hours ≈ 4–5 passes; 2.8 h fixed frontend, ~7 h
multi-span on the 4090.

## 7. Built (2026-09-09)

The corpus of §6 is implemented. Operational reference:
[`docs/data/PRETRAINING_CORPUS.md`](../data/PRETRAINING_CORPUS.md). Nothing has been
downloaded or trained.

**New tree.** `data/pretraining/` holds label-free scale sources; `data/datasets/` keeps the
labelled corpora. Every session in the new tree carries `__unlabeled__`, so it is barred from
label vocabularies, validation probes, and the Phase-B bank. Code resolves a dataset by name
through `data/scripts/curate/corpus_roots.py`, so the split costs no call sites; a name
present in both trees raises rather than resolving by precedence. `eval/data.py` still
addresses `data/datasets` directly on purpose, which structurally prevents a label-free
source from being resolved as an evaluation source. `nhanes` moved into the new tree.

**Sizing corrected against the implementation.** §6 quoted 3,500 NHANES subjects and a
~109 GB total on the assumption that window-level still-dropping would keep about two thirds
of the hours. Hour selection is motion-aware but does not shrink the per-subject budget
below 12 h, so the honest figure is 3,000 subjects and 110.6 GB (Ego-Exo4D is 1,286 h, not the 1,422 h
§6 quoted from a secondary source; the paper's own abstract says 1,286). `corpus_plan.py` is now the
single source of truth for the arithmetic, and a test fails if the plan leaves the budget.

**Storage: float16, not int16.** §4 proposed a scaled int16 grid. Built as float16 instead.
Both are two bytes, so the budget is unchanged, but the failure modes are not: a reader that
forgets to dequantize an int16 grid trains on values wrong by the scale factor with nothing
crashing, while a reader that forgets float16 gets correct values in a narrower dtype. Every
read path in the repo already ends in `np.asarray(..., dtype=np.float32)`, so no consumer
needed changing. A test measures the claim rather than asserting it: a 2 Hz gait-band tone
round-trips with its peak bin unchanged and magnitude within 0.1%.

**Aria IMUs stored at 200 Hz** rather than their native 800 Hz / 1 kHz, with the native rate
preserved in the manifest and carried per window as `source_rate_hz`. The frontend's f_max is
~14 Hz, so the native rate would cost 4-5x for content discarded in the first layer.

**One command, three idempotent stages.** `python -m data.pretraining.build_corpus`, with
`--plan`, `--stage fetch|convert|grids|all`, `--datasets`, and a `--budget-gb` that refuses an
over-budget selection. Fetching requires `--yes` on every run, because several sources move
hundreds of gigabytes in transit. Licence-gated fetchers print their access steps and refuse
to run without credentials; nothing bypasses a licence.

**Still open.** `embody3d` is wired as a candidate excluded from the budget until its licence
is confirmed. And the objective fixes in §2 plus a Phase-A-versus-random control still belong
before a full-scale ingest: every checkpoint behind the current best results carries
`phase_a_checkpoint=None`, so the value of the pretraining stage itself is not yet
re-established at any corpus size.

## 8. Pretraining and the supervised head are now disjoint (2026-09-09)

A second decision, made after §7 landed. The encoder previously pretrained on the same 18
labelled corpora the comparator and classification head were trained and selected on, so a
downstream gain could always be read as the encoder having already met those subjects,
devices and activities. The two stages now draw from disjoint trees: the encoder from
`data/pretraining/` (`LABEL_FREE_PRETRAIN_DATASETS`), the head from `data/datasets/`
(`SUPERVISED_HEAD_TRAIN_DATASETS`). Every labelled corpus is consequently out-of-sample for
the encoder.

Selected with `--corpus label_free`. The historical `expanded` and `matched` recipes keep
their names and contents, and `expanded` stays the CLI default until a label-free run has
been measured against it — flipping the default before that would silently change what every
unqualified command trains on. `assert_pretraining_is_label_free` refuses a labelled corpus
in the label-free recipe, judging by where a source sits on disk rather than by a roster
tuple, and treats a source present in neither tree as unbuilt rather than reporting a leak
that does not exist.

The cost of the split is that the label-free corpus is now the encoder's entire view of the
world, so its placement coverage matters more than when labelled data backfilled it. That is
why the plan carries Nymeria and Ego-Exo4D rather than NHANES alone at ten times the hours.
