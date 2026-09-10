# References: datasets and baselines

This tree stores the local primary-source material used to audit datasets and published encoders in
HALO. A usable entry contains:

- `citation.json`: title, authors/owner, year, venue, stable URLs, licence, and implementation notes.
- `paper.pdf`, `webpage.html`, or another clearly named official technical document.
- `SOURCE.txt`: canonical publication, data, and code URLs.

A `citation.json` plus a URL is not a substitute for the local review copy. New dataset integration
is incomplete until its publication or official protocol documentation is locally readable. Some
review copies are ignored for copyright reasons; `citation.json` and `SOURCE.txt` are the durable,
tracked locators from which a clean checkout can recover them.

## Current inventory

There are 52 dataset entries and 9 baseline entries, with 61 parseable `citation.json` files, 64
locally readable PDFs, and 7 saved HTML documents (counted 2026-09-09). Every entry has a
`citation.json` and `SOURCE.txt`.

Dataset entries:

`100style`, `aidlab_har`, `alameda`, `amass`, `c_mhad`, `capture24`, `cops`, `crossfit`, `dsads`,
`ego_exo4d`, `embody3d`, `extrasensory`, `forth_trace`, `hapt`, `harmes`, `harth`, `hhar`, `hmog`,
`inclusivehar`, `kneepad`, `kuhar`, `mhealth`, `mmfit`, `mobiact`, `monipar`, `motion_x`,
`motionsense`, `nfi_fared`, `nhanes`, `nymeria`, `oca`, `openpack`, `opportunity`, `pamap2`,
`phytmo`, `realdisp`, `realworld`, `recgym`, `recofit`, `rehab120`, `shoaib`, `sp_sw_har`, `spar`,
`tnda_har`, `uci_har`, `unimib_shar`, `upper_limb_use`, `usc_had`, `ut_complex`, `wear`, `wisdm`,
and `xrf_v2`.

Baseline entries retained for representation comparison and historical reproduction:

`crosshar`, `deepconvlstm`, `lanhar`, `limubert`, `llasa`, `moment`, `normwear`, `ssl-wearables`,
`unimts`.

## Application use

A local publication proves provenance; it does not by itself make a dataset suitable for an
application task. The active role of each source, including subject/session requirements and
pretraining-overlap restrictions, is defined in
[`../docs/data/APPLICATION_DATASETS.md`](../docs/data/APPLICATION_DATASETS.md).

## Scale-source evidence

- **Capture-24:** `references/datasets/capture24/paper.pdf`, the published Scientific Data paper.
- **ExtraSensory:** `references/datasets/extrasensory/paper.pdf`, the canonical phone/watch paper.
- **NHANES PAX80_G:** `references/datasets/nhanes/procedures_manual.pdf` plus the locally saved
  `pax80_g_documentation.html`. The CDC release documentation is authoritative for rate, units,
  placement, file schema, missing labels, and QC intervals.
- **PAAWS:** researched but not integrated. The official release endpoint returns HTTP 403 from this
  machine, so a publication alone does not satisfy the data-access gate.

## Known reference debt

The following pre-existing entries still lack a local paper or saved official page: `harmes`,
`nfi_fared`, `sp_sw_har`, `tnda_har`, `upper_limb_use`, `usc_had`, `ut_complex`, `xrf_v2`, and
baseline `deepconvlstm`. Their tracked locators exist, but the local review-copy gate remains open.
Before the final paper audit, close this debt and rerun the reference-link checker.
