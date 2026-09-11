# data/datasets

One subfolder per dataset. Each `<name>/` holds the **dataset-specific** pieces:

- downloads / upstream source (**gitignored** — regenerated from the converter)
- converter / preprocessing script(s) that turn the raw source into per-subject sessions
- metadata, channel descriptions, and per-dataset notes (e.g. which device/placement/channels we keep,
  gravity state, sampling rate, any known data-quality caveats)

Shared, **cross-dataset** logic (unit/gravity canonicalization, the device/channel-selection policy,
harmonised-vs-raw assembly, augmentations, the setup-all entry point) lives in [`../scripts`](../scripts),
not here.

The retained datasets feed support-classifier training and held-out evaluation. A dataset used for
label-free encoder pretraining cannot also support an unseen-dataset generalization claim. Its role
must be recorded in the run manifest and in the promoted result record.

The gridded training corpus and old held-out HAR roster remain available for representation training
and historical reproduction. Complete converted sessions, timestamps, gaps, subject identity, and
recording provenance is authoritative for support-classifier episodes; six-second grids
must not be mistaken for complete recordings.

New data sources still require a locally readable publication or official protocol under
[`../../references/datasets`](../../references/datasets), verified acquisition metadata, and an
explicit training or evaluation role before they enter a reportable experiment.
