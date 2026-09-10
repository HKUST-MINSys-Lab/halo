# Raw inertial-data download logistics — research notes (2026-09-09)

Verification standard: numbers below are cited to a specific page/PDF where found. Where I
could not find a directly-stated number, I built an estimate from stated per-file sizes and
show the arithmetic, and flag it "ESTIMATE".

---

## 1. NHANES 2011-2012 & 2013-2014 raw 80 Hz wrist accelerometry (PAX80_G / PAX80_H)

- **Total size**: PAX80_G (2011-2012) ≈ **1.04 TB compressed**, 6,917 participant archives.
  PAX80_H (2013-2014) ≈ **1.17 TB compressed**, 7,776 participant archives.
  Combined ≈ **2.2 TB compressed**.
  Source: CDC NHANES data-file pages (PAX80_G.htm / PAX80_H.htm), confirmed via
  Clevenger blog post "Downloading NHANES Raw Acceleration Data" which independently
  states "almost 30 TB in total" for the *decompressed/uncompressed* CSVs across all
  cycles+NNYFS — i.e. the ~2.2 TB compressed figure expands roughly 13-14x on disk once
  unzipped (consistent with bzip2 compression of repetitive float CSVs).
  - https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAX80_G.htm
  - https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAX80_H.htm
  - https://clevengerkimberly.github.io/2022-10-19-nhanesraw/
- **Fetch method**: Each participant's raw data is one FTP file. There is a public FTP/HTTPS
  folder per cycle listing all files (6,917 for G, 7,776 for H); no single bulk archive.
  CDC also released the **NHANES.RAW80Hz R package** (NCI, epi.grants.cancer.gov) whose
  `downloadAcclRawData80Hz()` function scripts the bulk download.
  - https://epi.grants.cancer.gov/physical/NHANES-RAW80Hz-manual.pdf
- **Per-participant file layout**: One `.tar.bz2` archive per participant containing up to
  194 hourly `.csv` files (one CSV per hour of wear, up to 194 h ≈ 8 days) plus **one data
  quality-control CSV** per participant.
- **Per-hour CSV structure**: columns `HEADER_TIMESTAMP` (YYYY-MM-DD HH:MM:SS.MMM), X, Y, Z.
  A full hour at 80 Hz = 288,000 rows.
- **Units**: g (gravity), values in range roughly ±6.006 g per axis (ActiGraph GT3X+, wrist-worn).
- **Data-quality flags**: QC CSV per participant flags repeated max/min values, impossible
  acceleration spikes, zero/implausible g-values, sensor-malfunction patterns. Reported
  ~0.26% of records flagged across cycles.
  Source: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAX80_G.htm
- **Gotchas**: no video/other modality mixed in (accelerometer-only device), so "inertial
  only" is moot — the whole PAX80 release IS inertial-only. Because of file count and total
  size, CDC/community guidance is to download only what you need and expect long transfer
  times over plain FTP/HTTPS (no S3/CLI-native bulk tool beyond the R package).

---

## 2. UK Biobank raw accelerometer .cwa (Field 90001)

- **Access model**: Raw .cwa files are **not bulk-downloadable off-platform**. They live at
  `/Bulk/Activity/Raw/<dir>/<eid>_90001_4_0.cwa` inside the UK Biobank Research Analysis
  Platform (RAP, DNAnexus-hosted). All processing must happen on RAP (JupyterLab/RStudio/Swiss
  Army knife apps) — individual files can be pulled out with `dx download` but bulk egress off
  the platform is discouraged/restricted by UKB data-access rules.
  - https://community.ukbiobank.ac.uk/hc/en-gb/community/posts/16019555977885
  - https://dnanexus.gitbook.io/uk-biobank-rap/getting-started/data-structure
- **Per-file size**: community-reported **~250 MB raw per participant's 7-day .cwa file**,
  compressing to **~100 MB**.
  Source: UK Biobank community forum thread (search-result synthesis, unverified primary doc)
  — treat as **ESTIMATE-grade but sourced**.
- **Total size (ESTIMATE, arithmetic shown)**: 103,686 participants completed the
  accelerometer sub-study (source: ukbiobank.ac.uk PLOS ONE study page). At ~250 MB/file
  raw: 103,686 × 0.25 GB ≈ **26 TB raw** (≈10.4 TB if using the ~100 MB compressed figure:
  103,686 × 0.1 GB ≈ 10.4 TB). Both bracket the commonly-quoted "many terabytes" language
  used by Oxford's ssl-wearables README. No single authoritative total-TB figure was found
  in official UKB documentation.
  - https://www.ukbiobank.ac.uk/publications/large-scale-population-assessment-of-physical-activity-using-wrist-worn-accelerometers-the-uk-biobank-study/
  - https://github.com/OxWearables/ssl-wearables ("many terabytes")
- **Fees/approval for a HK university student applicant**:
  - Register an account, submit a Material/Data Access Application describing the health-related,
    public-interest research project (any country, academia/industry/charity/government all
    eligible — nationality not a bar).
  - **Student rate: £500 (+VAT)** data-only access fee, valid 3 years from approval, when the
    application is submitted by a student or their supervisor for the student's own project.
    Source: https://community.ukbiobank.ac.uk/hc/en-gb/articles/28612614071709
  - Standard (non-student) data-only fee is higher (tiered by data type/tier); RAP compute is
    billed separately (DNAnexus credits) since raw accelerometry can only be processed on RAP.
  - https://www.ukbiobank.ac.uk/use-our-data/fees/
  - https://www.ukbiobank.ac.uk/use-our-data/fees/financial-support/ (credits/financial-support
    programme exists for those who qualify)

---

## 3. Nymeria (Meta Project Aria, ECCV 2024)

- **Download tool**: `nymeriaplus-download` CLI (facebookresearch/nymeria_dataset repo), e.g.
  `nymeriaplus-download -i /path/to/url.json -o /path/to/outdir -y`, with filtering (`-s`),
  parallel workers (`-n`), resume support. Requires registering via projectaria.com/datasets/nymeria
  to receive the URL manifest (email-gated).
  - https://github.com/facebookresearch/nymeria_dataset
  - https://www.projectaria.com/datasets/nymeria/
- **"Parts"/data groups**: The downloader organizes data into "groups" (defined in
  `nymeriaplus/layout.py`) that can be filtered after browsing sequences in the Aria Dataset
  Explorer (by location/date/activity/annotation availability). The repo/paper describe
  per-sequence modalities: Aria RGB/SLAM/eye-tracking video+IMU+audio+barometer/magnetometer,
  miniAria wristband streams, and Xsens MVN body-motion output. **I could not confirm from
  public docs a documented flag that downloads ONLY the IMU/motion streams while excluding
  video** — the repo mentions per-"group" selective download but the specific group names for
  an IMU-only pull were not visible in the fetched README. This is a gap — recommend checking
  `nymeriaplus/layout.py` directly in the repo, or asking on the project's GitHub issues.
- **Rates**: Aria right IMU 1 kHz, left IMU 800 Hz, magnetometer 10 Hz, barometer 50 Hz;
  Xsens MVN outputs full-body motion at 240 Hz (recorded from 1 kHz Xsens IMUs internally).
  - Source: search-result synthesis of the ECCV paper (arXiv 2406.09905).
- **Formats**: Aria data ships as **VRS** containers (video+IMU+audio multiplexed); body
  motion/pose as **NPZ** (SMPL-style) per the repo's mention of NPZ for SMPL files; no
  confirmed plain-CSV IMU export path in the docs I could fetch.
- **Size**: Full Nymeria dataset ≈ **80 TB** (paper/press quote: "Nymeria and NymeriaPlus
  datasets each provide 1100 sequences, which in total amounts to ~80 TB per dataset").
  IMU-only subset size is **NOT documented** in any page I could reach — flag as unresolved.
  - https://github.com/facebookresearch/nymeria_dataset (fetched README)
  - Scale stats from paper: 1200 sequences / 300 h / 264 participants / 11.7B IMU samples total.

---

## 4. Ego-Exo4D — Aria IMU without video

- **Confirmed**: the CLI download tool exposes a part `take_vrs_noimagestream` — VRS files
  containing IMU (+audio, camera intrinsics/calibration) but **excluding image/video
  streams**. This is the part to use for IMU-only downloads.
- **Size**: `take_vrs_noimagestream` = **995.6 GB (≈1 TB)**, vs. `take_vrs` (full, with images)
  = 12,301.5 GB (≈12 TB). Other parts for context: `take_point_cloud` 6,164.6 GB,
  `take_trajectory` 509.5 GB, `annotations` 10.5 GB. Full dataset ≈ 38.45 TB; recommended
  default subset ≈ 12.11 TB.
  Source: https://docs.ego-exo4d-data.org/download/ (fetched size table)
- **Format**: still VRS (a multiplexed container), not pre-extracted to CSV/NPY — IMU streams
  must be parsed out of the VRS via `projectaria_tools`/VRS reader; the "noimagestream" variant
  just drops the camera payload, cutting size ~12x, not the container format itself.
- **Access**: license agreement + AWS credentials emailed within ~48 h, then the `ego4d` CLI
  (`pip install ego4d`) with `--parts take_vrs_noimagestream`.
  - https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md

---

## 5. Sussex-Huawei Locomotion (SHL)

- **Total size**: official site states collected data **exceeds 950 GB** (2812 h labelled,
  17,562 km travelled). Some third-party summaries cite ~270 GB for a reduced
  "Challenge"/preview subset — treat 950 GB as the authoritative full-dataset figure and
  270 GB as a smaller challenge-specific release, not the same thing.
  - http://www.shl-dataset.org/ (site itself returned a connection error on fetch; number
    corroborated via cached search snippet only — **could not independently re-verify by
    direct fetch**, so treat as **medium-confidence**).
- **Per-position files**: four phone placements (Hand, Torso, Backpack, Trousers' front
  pocket), each phone (Huawei Mate 9) logging accelerometer, gyroscope, magnetometer,
  ambient pressure at ~100 Hz, plus GPS/location and video for some subsets.
- **Motion-only subset**: The "Challenge" releases (e.g. 2018, 2025) redistribute
  preprocessed **motion-sensor-only** feature files — e.g. one Challenge format cited:
  files of 196,072 rows × 500 columns representing 5 s frames at 100 Hz (labelled), with a
  smaller validation split (28,789 rows). This strongly suggests a motion-only subset
  (no video/location) is obtainable via the Challenge downloads rather than the full
  950 GB multimodal release, but I did not verify an explicit "motion-only, full-length,
  non-challenge" download option on the main site (fetch failed).
  - https://www.sussex.ac.uk/strc/research/wearable/locomotion-transportation
  - http://www.shl-dataset.org/challenge-2025/ (indirect, via search snippet)
- **Gotcha**: download is split into 3 parts that must be unzipped into a common folder.

---

## 6. PPMI Verily Study Watch raw accelerometer/gyroscope

- **Sampling rate**: **100 Hz** tri-axial accelerometer + gyroscope for PD cohort
  participants; **30 Hz** for non-PD cohorts; a lower 50 Hz rate occurred for some
  participants early in the study.
  Source: npj Parkinson's Disease paper "Digital outcome measures from smartwatch data
  relate to non-motor features of Parkinson's disease" — https://www.nature.com/articles/s41531-024-00719-w
  (also PMC11137004).
- **Access route**: Verily-derived summary measures (sleep, activity, vitals) are on the
  standard **LONI/IDA** PPMI download portal. **Raw device-level sensor data is NOT on the
  open download portal** — it requires a separate request: "Study Docs → Study Data Request
  Forms → Verily Raw Device Data Request Form," reviewed on request.
  - https://github.com/MJFF-ResearchCommunity/PIE (PPMI Data User Guide references section
    "15.2 Verily study watch derived data," but I could not retrieve the full section text —
    the raw-request-form process is corroborated by a separate search-result synthesis, not
    directly re-read from the PDF, which failed to parse as text).
  - https://www.ppmi-info.org/access-data-specimens/download-data
- **AMP-PD**: AMP-PD hosts a curated/harmonized version of PPMI (and other cohort) data on
  Google Cloud (GCS buckets + BigQuery tables) accessible via the AMP-PD Researcher Workbench;
  whether raw (unaggregated 100 Hz) Verily accelerometer/gyroscope streams are included there
  vs. only derived features was **not confirmed** — flag as open question requiring direct
  AMP-PD data-dictionary check.
  - https://amp-pdrd.org/amp-pd-use-cases , https://amp-pd.org/cloud-architecture
- **File format / size**: not documented in any page I could reach — LONI raw-request process
  suggests format is determined per-request (likely CSV/JSON export from Verily's cloud), no
  public size figure found.

---

## 7. BOXRR-23

- **Total size**: **4.7 TB compressed**, expanding to **>8.0 TB raw/decompressed**.
  4,717,215 recordings from 105,852 users, chunked into 106 per-user-batch archives
  averaging ~45 GB/chunk.
  - https://rdi.berkeley.edu/metaverse/boxrr-23/paper.pdf (arXiv 2310.00430)
  - Hugging Face mirror (cschell/boxrr-23) independently states **~5.35 TB total** — the two
    sources disagree slightly (4.7 vs 5.35 TB); likely different compression/packaging,
    flagging both.
- **Format**: **XROR** (XR Open Recording), a BSON-based binary format for per-frame
  position/orientation/event telemetry. Reference reader/writer library:
  https://github.com/MetaGuard/xror (Python, reads+decompresses XROR in one step, converts
  to/from TILT, BSOR, DAT, JSON).
- **Conversion to per-frame arrays**: use the `xror` Python library to decode a recording into
  its underlying frame-indexed position/quaternion arrays; for BOXRR-23 specifically, the
  companion "kinematic-maze" project (cschell) ships a **conversion script specific to
  BOXRR-23** plus a downstream "Motion-Learning-Toolbox" for further preprocessing.
  - https://cschell.github.io/kinematic-maze/
  - https://github.com/cschell/Motion-Learning-Toolbox
- **Per-file structure**: per-user tarball under `users/`, each containing individual XROR
  replay files; separate BSON `metadata/` files for filtering by user/app (BeatSaber,
  TiltBrush, etc.).
  - https://huggingface.co/datasets/cschell/boxrr-23

---

## 8. AMASS and Motion-X++ — sizes, and IMU-synthesis reference implementation

- **AMASS**: 344 subjects, 11,265 motions, >40 h of mocap unified into SMPL/SMPL+H/DMPL
  parameters, npz format, gated behind registration at amass.is.tue.mpg.de.
  **I could not find an authoritative total-download-size figure** (GB) on the official site,
  GitHub repo, or ICCV paper via web search/fetch — all attempts returned composition stats
  (subjects/motions/hours) but no storage size. **Flag as unresolved**; commonly-repeated
  community figures (e.g., "~18 GB" for SMPL+H) appear in blog posts but I could not trace
  them to a citable primary source, so I am not reporting a number here rather than reporting
  an unverifiable one.
  - https://amass.is.tue.mpg.de/ , https://github.com/nghorbani/amass
- **Motion-X++**: **50.5 GB** total (Hugging Face mirror YuhongZhang/Motion-Xplusplus),
  covering 19.5M whole-body SMPL-X pose annotations over 120.5K sequences, 80.8K RGB videos,
  45.3K audio clips (multimodal — video/audio are part of that 50.5 GB, not separable in the
  HF mirror as inspected).
  - https://huggingface.co/datasets/YuhongZhang/Motion-Xplusplus
  - https://arxiv.org/pdf/2501.05098
- **IMU-synthesis reference implementations** (SMPL/AMASS → virtual IMU):
  - **TransPose** (Yi et al.): places **6 virtual IMUs on SMPL mesh vertices**; orientation
    from compounded joint rotations along the kinematic chain to the pelvis (root);
    acceleration via finite differences of vertex position over time; AMASS resampled from
    60–120 fps down to **25 fps**, following the DIP synthetic-IMU convention; acceleration
    (m/s²) scaled ×30 for network input.
    - https://arxiv.org/pdf/2105.04605 , https://github.com/Xinyu-Yi/TransPose
  - **IMUPoser** (Mollyn et al.): attaches virtual IMUs to specific SMPL vertices —
    **left wrist, right wrist, left front pants pocket, right front pants pocket, scalp**
    (i.e. phone/watch/earbuds placements) — https://arxiv.org/pdf/2304.12518
  - **UniMTS** (Zhang et al., NeurIPS 2024, github.com/xiyuanzh/UniMTS — this is the same
    UniMTS baseline already in the HALO corpus per memory): synthesizes motion time series
    from **BVH motion-skeleton data with all-joint coverage** using **IMUSim**
    (`bvh2ts.py` script under IMUSim's root) rather than AMASS npz directly; spatio-temporal
    graph modeling captures cross-joint relationships to generalize across arbitrary
    device positions/orientations. This is a different pipeline (BVH+IMUSim) from the
    TransPose/IMUPoser vertex-finite-difference approach — worth noting for citation
    accuracy if HALO's docs currently imply UniMTS uses the TransPose-style method.
    - https://github.com/xiyuanzh/unimts , https://arxiv.org/html/2410.19818v1

---

## 9. CAPTURE-24 and Ego4D — sizes for completeness

- **CAPTURE-24**: 151 participants, 3,883 h total accelerometer recording (2,562 h
  annotated). **No GB/TB figure found** in the Nature Scientific Data paper, arXiv page, or
  OxWearables GitHub README via search — only duration/participant counts are given.
  Flag as **unresolved** (already have local copy per user's memory, so this is FYI only).
  - https://www.nature.com/articles/s41597-024-03960-3 , https://github.com/OxWearables/capture24
- **Ego4D** (video+audio+IMU, distinct from Ego-Exo4D): full videos + annotations download
  ≈ **5 TB**. (No separate IMU-only figure found; a downstream benchmark, MMG-Ego4D, uses a
  202 h video-audio-IMU-aligned subset of Ego4D — 167 h unlabeled + 35 h labeled — but that's
  a curated subset size in hours, not a GB figure for the IMU stream alone.)
  - https://ego4d-data.org/ , MMG-Ego4D: https://arxiv.org/pdf/2305.07214

---

## (a) UK Biobank raw-accelerometry SSL training — disk footprint / GPU-days / windowing

Source: Yuan et al. 2024, npj Digital Medicine, "Self-supervised learning for human activity
recognition using 700,000 person-days of wearable data" + OxWearables/ssl-wearables GitHub.

- **Scale**: >100,000 UK Biobank participants wore a wrist accelerometer for 7 days each,
  totalling **>700,000 person-days** and (per the repo's own description) **"many
  terabytes"** of raw free-living data — **no exact TB figure is stated** in either the
  paper text or README snippets I could retrieve.
  - https://github.com/OxWearables/ssl-wearables
  - https://www.nature.com/articles/s41746-024-01062-3
- **Windowing/downsampling**: models (`harnet5`/`harnet10`/`harnet30`) operate on **30 Hz**
  windows of **5 s / 10 s / 30 s** (150 / 300 / 900 samples respectively) — i.e. raw UKB .cwa
  data (recorded ~100 Hz, resampled) is downsampled to 30 Hz before windowing for SSL
  pretraining.
  - https://github.com/OxWearables/ssl-wearables (repo model dimension listings)
- **GPU-days**: **not disclosed** in any page I could reach (README, paper abstract/search
  snippets). The paper likely states this in its Methods/Supplementary section behind the
  full-text paywall/HTML render I could not fully parse — flag as **unresolved, needs direct
  paper PDF read** (I did not have time/tool budget to pull and OCR the full npj Digital
  Medicine PDF in this pass).
- **Storage footprint reported**: **not found** — same caveat as above.
- **Recommendation**: if this number is load-bearing for HALO's design docs, do a targeted
  read of the full paper PDF (nature.com/articles/s41746-024-01062-3) Methods section, which
  was not fully retrievable via the web-fetch summarizer in this pass.

## (b) Google LSM — GPU-days / storage

- **Not found.** No search returned a paper/blog specifically named "Google LSM" with reported
  GPU-days or storage footprint for wearable/IMU self-supervised pretraining in the time
  available. This likely refers to a specific internal Google project (possibly "Large
  Sensor Model" or similar) not indexed publicly, or a name I mis-resolved. **Recommend the
  user clarify the exact paper/project name** (e.g. full title or arXiv ID) so a targeted
  search can be run — a blind search for "Google LSM wearable" returned no relevant hits.

---

## Summary table (sizes only, GB unless noted)

| Dataset | Total size | Confidence | IMU-only subset available? |
|---|---|---|---|
| NHANES PAX80_G+H | ~2.2 TB compressed (~30 TB uncompressed, per blog) | High (CDC pages) | N/A — accel-only device |
| UK Biobank 90001 | ESTIMATE 10–26 TB (103,686 files × 100–250 MB) | Medium | N/A — accel-only device, but RAP-locked |
| Nymeria | ~80 TB full | Medium (press/paper) | Unclear — group-based download exists but IMU-only group not confirmed |
| Ego-Exo4D | 38.45 TB full / 995.6 GB IMU-only VRS (`take_vrs_noimagestream`) | High (docs table) | Yes, confirmed |
| SHL | ~950 GB full (270 GB Challenge subset ≠ same thing) | Medium (fetch failed, search-only) | Likely yes via Challenge motion-only releases |
| PPMI Verily | Unknown | Low | Raw = request-only, format/size undocumented |
| BOXRR-23 | 4.7–5.35 TB compressed / >8 TB raw | High (two sources, slight disagreement) | N/A — motion-only by nature |
| AMASS | Unresolved | N/A | N/A |
| Motion-X++ | 50.5 GB (multimodal, not separable) | Medium (HF mirror only) | No |
| CAPTURE-24 | Unresolved (duration only: 3,883 h) | — | already local |
| Ego4D | ~5 TB full | Medium | No dedicated IMU-only figure found |
