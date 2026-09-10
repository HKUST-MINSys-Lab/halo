# Audit: the 18 labelled training sources

> 2026-09-09. Scope: `deployment_policy.SUPERVISED_HEAD_TRAIN_DATASETS` — 18 datasets, 56 streams,
> 1,963,606 windows, 3,166 stream-hours, measured from the `native` grids. Three questions were
> asked: are they valid, are they bug-free, and do they cover the right deployment scenarios and
> labels. Short answers: **largely valid, not bug-free, and not the right labels.**
>
> **Target (corrected 2026-09-10).** The first draft of this audit judged label coverage against the
> three-task clinical movement-monitoring pivot. **That pivot is abandoned.** The active program is
> the IMWUT comparison line: recognise by comparison, not classification — a query recording scored
> against candidate label *text* (zero-shot, k=0) and against labelled support *recordings*
> (enrollment, k=1..8), cross-subject and cross-configuration. Section 4 has been re-derived against
> that target and the movement-monitoring verdict is retracted; sections 1-3 are target-independent
> and unchanged.
>
> Short answers: **largely valid, not bug-free, and better matched to the target than the first
> draft claimed — with three real coverage limits.**
>
> Findings marked **[V]** were reproduced directly during this audit. Findings marked **[A]** come
> from a sub-audit and are reported with their evidence but were not independently re-measured.
> Nothing here was changed; this is an assessment.

## 1. Validity: the signal is in good shape

This is the strongest part of the corpus and it is worth stating plainly before the defects.

- **No normalisation damage anywhere in the 18.** The z-score signature (per-window mean ~0 and
  std ~1 on every real channel) is 0.000 on all 56 streams, and no stream is confined to [0,1] or
  [-1,1]. `unimib_shar`, whose raw source was once thought lost, is clean. [A]
- **Gravity is correct on all 56 streams.** Gravity-present streams read median |acc| 0.98-1.17 g;
  the two streams that declare gravity removed (`kuhar`, `xrf_v2` AirPods) read 0.117 g and
  0.030 g. Nothing escaped unit conversion into m/s². [A]
- **Low sampling rates are handled honestly, not silently.** Three sources cannot observe the top
  of the 0.3-15 Hz analysis band: `wisdm` at 20 Hz, `dsads` at 25 Hz, `opportunity` at 30 Hz —
  139,187 windows, 7.1% of the corpus. The filterbank's observability mask marks 26 of 32 bands
  usable at 20 Hz and 27 of 32 at 25 Hz, which is *more* conservative than raw Nyquist because it
  accounts for filter bandwidth. This is declared heterogeneity, and it is the design working. [V]
- **No cross-dataset duplication among the 18**, calibrated against the known `hapt`x`uci_har`
  positive control (NCC fraction >0.95: 0.020 for the known duplicate, 0.000 for every tested pair
  inside the roster). `hapt` itself is confirmed absent from the roster. [A]
- **Split integrity holds.** Train and validation are subject-disjoint on all 18; `CorpusIndex`
  keys subjects by `(dataset, subject)` everywhere, which matters because 61 subject-id strings
  collide across datasets. Exact duplicate windows are *removed* from the training pool, not merely
  recorded. The quality caches are fresh, not stale. [A]

## 2. Bugs

### 2.1 Window-overlap leakage in three sources — the most serious finding

Three of the 18 are distributed pre-windowed with overlap, and the comparator's leakage unit
collapses onto the window for them. `training/compare/sampling.py::_execution_ids` derives an
"execution" by stripping an ordinal from the event id; on these sources every window is its own
execution, so the rule "a support execution is always physically distinct from the query" becomes
vacuous.

| source | windows / executions | consecutive same-subject windows sharing an exact 50% overlap |
|---|---:|---:|
| `uci_har` | 10,299 / 10,299 | **91.5%** [V] |
| `sp_sw_har` | 3,757 / 2,621 | **47.5%** [V] |
| `unimib_shar` | 11,771 / 11,771 | 0.3% exact [V], but 26.7% of adjacent pairs exceed NCC 0.99 [A] |
| (control) `capture24` | 118 windows/execution | 0.0% [V] |

So a query and its "support" can be two windows sharing literally half their samples, and the
sampler excludes none of them. Under the comparator's uniform-over-datasets query sampling these
three carry **16.7% of all training queries**. This reaches both the training signal and
`best.pt`, which is selected on `validation/selection_dataset_macro_f1` computed from this corpus.

The evaluation path already has the guard the training path lacks (`eval/data.py` uses
`execution_granularity="block"`, and excludes `tnda_har`, its only source with the same
one-window-per-execution ratio). So published evaluation numbers are not implicated; **model
selection is**. [A]

### 2.2 Two label-mapping errors in KU-HAR

KU-HAR defines `Stand-sit` as *repeatedly standing up **and** sitting down*; the converter maps it
to a single direction, contaminating **28.7% of `standing_up_from_sitting`** and **55.1% of
`standing_up_from_lying`**. Separately `Walk-circle` is mapped to plain `walking`, deleting turning
gait — while `sp_sw_har` carries `turning` as its own label, and MoniPar, the sealed Task-2
evaluation source, is a Parkinson's cohort for whom turning is a primary measurement. [A]

### 2.3 `mhealth` gyroscope is sample-and-hold

72.2% of consecutive gyroscope samples are exact repeats, against 0.0% for the accelerometer in the
same stream — an effective rate near 13 Hz on a stream declared at 50 Hz. [V] The heterogeneity doc
calls it "somewhat sample-and-hold but real", which materially understates it. `mhealth` is only
1.9 hours, but under uniform dataset sampling it is 5.6% of queries, and it misinforms a
rate-conditioned frontend.

### 2.4 A known duplicate is still live in a published baseline

`baselines/harnet/adapter.py`'s `LEGACY_TRAIN_DATASETS` still contains `hapt` alongside `uci_har`,
so the published HARNet comparison row trains on both halves of the duplicate that was removed from
HALO's own roster for exactly that reason. [A]

### 2.5 Documentation contradicts code, and counts are stale

`DATA_HETEROGENEITY.md` states `unimib_shar`'s accelerometer unit is g; `accel_units.py` classifies
it m/s². The code is right (measured output 1.010 g; the doc's reading would give 9.9 g). The same
document says the corpus has 161 canonical labels; the grids, `global_labels.json` and
`CorpusIndex` all say **166**. [A]

Also: `unimib_shar` clips at the ±2 g rail on 0.21% of samples, the only stream above 0.1%; ~1,800
windows corpus-wide have length 1 and no screen rejects them. [A]

## 3. Composition: the sampler inverts the corpus

The raw corpus is extremely lopsided, but the samplers correct it — and the correction is itself the
finding.

- `capture24` is **78.6% of windows and 81% of hours**, from one placement (dominant wrist), three
  channels (no gyroscope), 151 subjects, 10 labels. Everything else combined is 606 hours. [V]
- Label mass is static posture: `sitting` 29.1% + `sleeping` 28.9% = 58%; the top seven labels are
  81.8%. **42.4% of all windows are near-static** (max-axis acc std < 0.02 g). [A]
- The Phase-A sampler tempers `capture24` to 16.0% of draws. The comparator's episodic sampler
  (`_choose_query`) is uniform over datasets, then uniform over labels within a dataset. So the
  imbalance never reaches the training signal. [V]
- **But that inverts the corpus.** Every dataset gets 5.6% of queries regardless of size: [V]

| | windows | share of pool | share of draws | oversampling | subjects |
|---|---:|---:|---:|---:|---:|
| `capture24` | 1,543,573 | 78.6% | 5.6% | **0.1x** | 151 |
| `mhealth` | 1,230 | 0.06% | 5.6% | **88.7x** | 10 |
| `pamap2` | 3,292 | 0.17% | 5.6% | 33.1x | 9 |
| `sp_sw_har` | 3,757 | 0.19% | 5.6% | 29.0x | 23 |

  Five datasets with ten or fewer subjects (`opportunity` 4, `dsads` 8, `hhar` 9, `pamap2` 9,
  `mhealth` 10) supply **27.8% of all training queries from 40 people and 4.2% of the pool**. Meanwhile
  2,560 hours of free-living wrist data contribute 5.6%.
- Three sources emit windows far shorter than the 6 s context: `sp_sw_har` 1.00 s, `uci_har` 2.56 s,
  `unimib_shar` 3.02 s. A one-second window is a single patch, so the masked-prediction objective has
  no temporal context to hide, and a 1 s example is not a comparable observation to a 6 s one. [V]
- `opportunity` is only 37.4% full windows (mean fill 0.68), so it contributes about two thirds of
  its nominal signal. [V]
- `capture24`'s effective amplitude resolution is coarse: 56.6% of consecutive samples are exact
  repeats and the median non-zero sample-to-sample step is 0.0158 g, against 0.9% repeats for
  `wisdm`. The values are *not* on a clean 1/64 g lattice (0% on-lattice at every candidate step), so
  this is packed device resolution rescaled by per-device calibration rather than simple
  quantisation. Real data, but an undocumented noise floor on 79% of the corpus. [V]

## 4. Labels, against the comparison target

Re-derived 2026-09-10. The first draft measured this corpus against movement-monitoring and found it
wanting because exercise and repetition content is only 4.7% of windows. **That criticism is
withdrawn**: for a comparison model over general activity, general activity labels are the target,
not a defect.

Against the actual target the corpus is in good shape on the two things comparison needs most.

- **Labels are not welded to one configuration.** Across 31 distinct acquisition configurations
  (device profile x placement site x channel set x gravity state) and 166 labels: **84% of labels
  appear in at least two configurations and 72% in at least four.** Only 16% are single-configuration.
  A sub-audit's headline that "79% of labels appear in one dataset" is true but misleading — one
  dataset is not one configuration, because the multi-placement sources (`realdisp` 9 streams,
  `phytmo` 8, `xrf_v2` 6, `dsads` and `opportunity` 5 each) each supply many configurations. The
  cross-configuration learning signal is real. [V]
- **Support sets are formable.** Of 921 (configuration, label) cells, the median holds 42 executions
  from 16 subjects. At k=8, 97.1% of cells have enough distinct executions and 86.8% enough distinct
  subjects for a cross-subject support set; only 3% of cells are too thin. Every one of the 166
  labels is carried by at least six subjects. [V]

Three limits remain, and they bear directly on the claim.

1. **40.2% of windows carry a label that exists in exactly one configuration.** [V] This is
   Capture-24: its ten labels appear only on a dominant-wrist accelerometer, and it is 79% of the
   window pool. That block of the corpus teaches label-to-configuration association rather than
   comparison across configurations. It is substantially mitigated by the episodic sampler, which
   reduces Capture-24 to 5.6% of queries, but the mitigation is a sampling choice rather than a
   property of the data.
2. **Label text cannot express the distinctions several classes depend on.** This is first-order
   here because the k=0 head scores a query against candidate label *text*. Measured with the
   repo's own encoder (`all-MiniLM-L6-v2`): [V]

   | label pair | cosine |
   |---|---:|
   | `left_arm_raise` vs `right_arm_raise` | **0.959** |
   | `standing_up_from_sitting` vs `sitting_down_from_standing` | **0.942** |
   | `sitting` vs `sitting_down` | **0.934** |
   | `standing_up_from_lying` vs `lying_down_from_standing` | 0.912 |
   | `walking_upstairs` vs `walking_downstairs` | 0.868 |
   | `walking` vs `jogging` | 0.542 |
   | `sitting` vs `sleeping` | 0.381 |

   Genuinely different activities separate cleanly; **direction of transition and body side do not
   separate at all**. Those classes are close to unreachable for a zero-shot head no matter how good
   the encoder is, and the branch notes already record that "zero-shot remains weak". This is a
   plausible contributing cause and is cheap to test by scoring zero-shot accuracy against pairwise
   label-text similarity. A further 72 of 166 labels have no authored paraphrase, and 24 synonym
   keys are dead. [A]
3. **Support labels are internally noisy on the largest source.** Capture-24's `sitting` class
   (552,295 windows, 28.1% of the corpus) merges 59 source annotations including *office work such
   as writing and typing*, *eating sitting alone* and *using a mobile phone/tablet* — on a
   dominant-wrist sensor, while the same corpus carries `typing`, `writing`, `reading`, `eating_*`
   and `drinking` as separate labels. For a comparison model this is label noise inside the support
   set: a recording enrolled as `sitting` may be a typing recording. Its annotation is also coarse,
   changing 117-259 times per 24 hours from camera review at roughly 30-second cadence, and
   `sleeping` (28.9% of the corpus) rests on 288 diary runs of median 3.2 hours with no camera
   evidence. [A]

## 5. Deployment coverage

Training spans 30 placement sites across phone, watch and body-worn device profiles. Five sites
appear only in evaluation — `affected_wrist`, `unaffected_wrist`, `belt`, `hip`, `front_pocket` —
and **every one has an equivalent trained site** under `compatibility.EQUIVALENT_SITES` (the wrist,
waist and pocket groups). No evaluation placement lacks a trained equivalent. [V]

That is a coverage strength for ordinary transfer, but it is a direct limit on the
cross-configuration half of the claim: the configuration axis actually being tested is device, side
and wording, never a genuinely unseen body site. A reader is entitled to ask whether "unseen
acquisition configuration" means more than "unseen device at a site we trained on".

## 6. Assessment

**Valid:** yes, with two exceptions. Units, gravity, scale and rate honesty are sound across all 56
streams, and the low-rate sources are correctly marked rather than silently corrupted. The
exceptions are `mhealth`'s sample-and-hold gyroscope and `capture24`'s undocumented resolution
floor.

**Bug-free:** no. The window-overlap leakage in `uci_har`, `sp_sw_har` and `unimib_shar` is the
finding that changes a number someone might report, and it is worse under the comparison target than
under any classification target, because query-to-support matching *is* the mechanism: a support
recording can share half its samples with the query. It reaches `best.pt` selection. The KU-HAR
transition mapping is a correctness bug in a small class. The stale `hapt` entry in the HARNet
baseline roster undermines one published comparison row.

**Right coverage and labels:** better than the first draft claimed, with three specific limits. For
recognition by comparison, the corpus supplies what the mechanism needs — 31 acquisition
configurations, 84% of labels seen in two or more of them, every label carried by at least six
subjects, and enough executions to form cross-subject support sets at k=8 in 97% of cells. The
limits are that 40% of windows carry a single-configuration label, that label text cannot express
direction or body side and so caps the zero-shot head on those classes, and that no evaluation
placement is a body site absent from training. None of these is a data-integrity problem; all three
are limits on what can honestly be claimed from this corpus.

## 7. Actioned (2026-09-10)

`uci_har`, `sp_sw_har`, `mhealth` and `opportunity` are retired from the active training roster
(`deployment_policy.RETIRED_TRAIN_DATASETS`, with the reasons above recorded beside each name).
Data and StreamSpecs stay on disk. The active roster is 14 sources; the frozen historical 18 and
12 survive as `EXPANDED_18_TRAIN_DATASETS` and `CORPUS_MATCHED_TRAIN_DATASETS` so dated results
still resolve to a definite list. Every training entry point (`pretrain.py --corpus/--datasets/
--subset`, `compare/train.py`) now passes its roster through `assert_no_retired_sources`, which
refuses a retired source unless `--allow-retired` is given for historical reproduction.
`data/labels/global_labels.json` was regenerated: 166 → 165 labels (`turning` had only one source).

Not changed, deliberately: the episodic sampler's uniform-over-datasets weighting (§3), the
execution-identity guard for `unimib_shar`'s near-duplicates (§2.1), the KU-HAR mapping (§2.2), and
the stale `hapt` entry in the HARNet baseline roster (§2.4). Each is a separate decision.
