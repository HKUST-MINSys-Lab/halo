# The label-free pretraining corpus

> Built 2026-09-09. Operational reference for `data/pretraining/`. The design argument that
> selected these sources is in the active JEPA objective and design documents. The per-source
> access steps are in each source's own `README.md`. This page is the corpus contract and
> operating procedure.

## Why a separate tree

The JEPA objective reads no labels, so the ceiling on pretraining data is download and disk,
not annotation. That makes the natural corpus for it a different set of sources from the one
the evaluation protocol uses, and mixing the two in one directory has a specific failure
mode: a label-free source silently entering a labelled application-task manifest or evaluation.

`data/pretraining/` holds sources whose every session carries the reserved `__unlabeled__`
marker. `data/datasets/` holds labelled sources. The `__unlabeled__` guards already enforced
the separation semantically; the directory split makes it visible on disk and lets a reader
tell at a glance which tree a source belongs to.

Nothing about the on-disk contract differs between the trees. Code resolves a dataset by
**name** through [`data/scripts/curate/corpus_roots.py`](../../data/scripts/curate/corpus_roots.py),
so moving a source between trees is not a code change. A name present in both trees raises
rather than resolving by precedence: a half-finished move leaves two copies, and silently
preferring one would train on whichever sorted first.

The motion-monitoring manifests address `data/datasets` directly. That is deliberate: it
structurally prevents a label-free source from being resolved as an application evaluation source.

## The encoder and the head no longer share data

**Decision 2026-09-09.** Until now one roster did both jobs: the encoder pretrained on the same
labelled corpora used by downstream controls and application-task heads.
That conflates two questions that should be answerable separately. If a downstream number
improves, it can always be read as the encoder having already met those subjects, devices and
activities rather than as a better representation.

So the two stages now draw from disjoint trees:

| stage | corpus | roster |
|---|---|---|
| encoder pretraining | `data/pretraining/` | `LABEL_FREE_PRETRAIN_DATASETS` |
| support-classification control / application heads | `data/datasets/` | task-specific manifests |

Every labelled corpus is therefore genuinely out-of-sample for the encoder, which is a stronger
and more honest transfer claim than the previous arrangement could support.

This is the default recipe; select it explicitly in recorded launch commands with
`python -m training.tokenizer.pretrain --corpus label_free`. The historical
`expanded` (the active labelled roster, 14 sources) and `matched` (the frozen historical 12) recipes keep their names so
earlier runs stay reproducible. `deployment_policy.assert_pretraining_is_label_free` refuses a
labelled corpus in the label-free recipe, checking where a source sits on disk rather than
trusting a roster tuple, and a test asserts the two rosters never intersect.

Label-free sessions and native grids use eight-second source windows. This is independent of the
legacy six-second labelled-grid contract. The trainer checks every materialized label-free grid's
physical duration before constructing a loader and gives the exact rebuild command if stale grids
are found.

One consequence to keep in view: this makes the label-free corpus the *only* thing the encoder
sees, so its placement coverage now matters more than it did. Nymeria provides real
body-mounted motion across locations; NHANES and synthetic IMU prevent the representation from
becoming specific to one acquisition regime.

## Composition

Generated from [`data/pretraining/corpus_plan.py`](../../data/pretraining/corpus_plan.py),
which is the single source of truth for the build, the tests, and this table. Run
`python -m data.pretraining.build_corpus --plan` for the live figures.

| source | take | streams | stream-hours | rate | channels | GB |
|---|---|---:|---:|---:|---:|---:|
| `nhanes` | 3,000 participants x 12 h, motion-aware hours | 1 | 36,000 | 80 Hz | 3 | 124.4 |
| `nymeria_xsens` | all 1,100 sequences, 11 of 17 body placements | 11 | 3,300 | 240 Hz | 6 | 34.2 |
| `synthetic_imu` | AMASS + Motion-X++ (+ 100STYLE), 8 virtual placements | 8 | 1,800 | 60 Hz | 6 | 4.7 |
| **total** | | | **41,100** | | | **163.3** |

Against a 180 GB budget, leaving 16.7 GB of headroom. Accel-only sources still occupy six
grid slots because the common contract zero-pads and masks gyro. For comparison, the labelled Phase-A
corpus supplies 3,117 stream-hours, so this is roughly a 13.6x increase in label-free signal.

**Stream-hours are not wall-clock hours.** A sequence recorded by eight body sensors
contributes eight stream-hours per wall-clock hour, because each placement is a separate
token stream the encoder sees independently.

`embody3d` (500 h, 439 participants) is wired as a candidate and excluded from the budget:
its access terms could not be confirmed as of 2026-09-09. The orchestrator refuses to build a
candidate source until someone confirms the licence and moves it into `CORPUS_PLAN`.

`nymeria_aria` and `ego_exo4d` remain implemented as deferred adapters, but are not part of
the default experiment. This keeps the first comparison focused on one large real-motion source
with two inexpensive diversity controls.

## Two storage decisions

### Grids are float16

Declared per dataset as `"grid_dtype": "float16"` in `metadata.json`, honoured by
`data.scripts.build_grids._store_dtype`.

Halving grid bytes is what makes tens of thousands of stream-hours fit on one disk, and it
costs nothing the model can see. The frontend reads band energies below ~14 Hz, and float16
resolves ~0.001 g at 1 g — exactly the precision the NHANES release itself publishes. A test
measures this rather than asserting it: a 2 Hz gait-band tone round-trips through float16
storage with its peak bin unchanged and its magnitude within 0.1%.

float16 rather than a scaled int16, deliberately. Both are two bytes. But a reader that
forgets to dequantize an int16 grid trains on values wrong by the scale factor and nothing
crashes, whereas a reader that forgets float16 gets correct values in a narrower dtype. Every
read path in the repo already ends in `np.asarray(..., dtype=np.float32)`, which upcasts
float16 transparently, so no consumer needed changing.

### Aria IMUs are stored at 200 Hz

Project Aria devices sample at 800 Hz and 1 kHz. The frontend's highest analysis frequency is
~14 Hz, so a 200 Hz store is already an order of magnitude above anything the model reads,
and keeping the native rate would cost 4-5x for content discarded in the first layer. The
native rate is preserved in the session manifest and travels with each window as
`source_rate_hz`, so the tokenizer still knows what was actually acquired and never mistakes
a decimated stream for a natively slow one.

Xsens stays at its native 240 Hz: close enough to the target that resampling would add an
interpolation artefact for no saving worth having.

## Operating procedure

```bash
# inspect, changes nothing
python -m data.pretraining.build_corpus --plan

# per source, per stage; each stage is idempotent and resumable
python -m data.pretraining.build_corpus --stage fetch   --datasets nhanes --yes
python -m data.pretraining.build_corpus --stage convert --datasets nhanes
python -m data.pretraining.build_corpus --stage grids   --datasets nhanes
```

Fetching requires `--yes` on every run. Several of these sources move hundreds of gigabytes
in transit — Ego-Exo4D's image-free VRS part alone is ~1 TB, streamed through
convert-then-delete — and that must never start as a side effect of a command someone ran to
read the plan. The orchestrator also refuses a selection whose grid output exceeds the
budget, so raising it is always a deliberate act.

Licence-gated fetchers (`nymeria`, `ego_exo4d`, `synthetic_imu`) print the exact access steps
and refuse to run until the credentials or URL manifest exist. Nothing in this tree bypasses
a licence.

### Tooling isolation

The training runtime is `/home/alex/code/HALO/legacy_code/.venv/bin/python`. Do not install
`projectaria-tools` into it: current releases can replace its pinned NumPy stack, which would
invalidate model dependencies. VRS conversion is prepared in the isolated Python 3.12 runtime
`/home/alex/.venvs/halo-nymeria-download/bin/python`, with `projectaria-tools==1.7.1`, NumPy,
SciPy, pandas and PyArrow. Use that interpreter only for `nymeria.convert` and
`ego_exo4d.convert`; training and grid building continue to use the project runtime.

The Nymeria fetcher has a resumable standard-library fallback, so its official downloader does
not need to be installed. The Ego-Exo4D `egoexo` entry point is installed beside the project
interpreter and the fetcher resolves that location even when the interpreter is invoked by
absolute path.

Only the `native` grid regime is built for these sources. The `harmonised` and
`non_harmonised` regimes exist for the layout-locked baselines and the evaluation path,
neither of which may touch a label-free source.

## Cost

Measured on this machine (RTX 4090), from the Phase-A run logs
`phase_a_a_clean_20260818` and `phase_a_e_clean_long_20260818`:

| quantity | value |
|---|---|
| Phase-A step, batch 1024, fixed frontend | 102 ms |
| signal consumed | ~1,000 stream-hours per minute |
| 100k steps = 102 M windows ≈ 4 passes over this corpus | 2.8 h fixed frontend |
| the same with the multi-span frontend (2.6x measured) | ~7 h |

Training cost is set by the step budget, not corpus size: the sampler draws with replacement.
Preprocessing is the slower half. NHANES conversion decompresses every candidate hour to
score it for motion, so budget roughly a day of download and a day of CPU for 3,000
participants; `--selection evenly_spaced` skips scoring and decompresses only the hours it
keeps, at the cost of a corpus with more stillness in it.

## Sampling consequences

The hierarchical sampler is unchanged: `P(dataset) ∝ n^0.25` capped at 25%,
`P(subject | dataset) ∝ n^0.5`, `P(window | subject)` uniform. So although NHANES holds ~85%
of the stream-hours here, its share of draws is capped near a quarter, and Nymeria's eleven
streams and Ego-Exo4D land in the 15-20% range each. The per-subject tempering is also why
NHANES is specified as 3,000 subjects at 12 hours rather than 500 subjects at a full week.

## Known caveats

- **Free-living wrist data is mostly stillness.** The 2026-07 audit measured 43% of NHANES
  windows below 0.003 g of max-axis standard deviation. Masked prediction of a motionless
  window from motionless neighbours teaches nothing, so hour selection is motion-aware by
  default. It deliberately keeps a one-third still minority: sleep and sedentary posture are
  real free-living content and are exactly what the DC and gravity part of the frontend reads.
- **Synthetic IMU is a diversity source, never the base.** Published evidence
  ([arXiv 2602.11064](https://arxiv.org/abs/2602.11064)) is that large-scale mocap
  pretraining alone gives marginal gains because of the sim-to-real gap, and helps mainly
  when mixed with real data. Its streams carry a `virt_` token so they can never be mistaken
  for measurements.
- **No overlap with the evaluation cohort.** None of these sources appears in the eight-dataset
  primary evaluation cohort or the held-out roster.
- **Phase-A's value over a random-init trunk is +0.057 and was measured on the old corpus.**
  Scaling the data does not by itself establish that the pretraining objective earns its
  place: every compact-engine checkpoint that produced the current best results carries
  `phase_a_checkpoint=None`. The objective fixes in the design doc and a Phase-A-versus-random
  control belong before, or alongside, a full-scale ingest.
