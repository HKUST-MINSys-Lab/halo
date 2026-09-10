# HALO documentation

> ## 📍 On branch `imwut/compare`, start with [**IMWUT_START_HERE.md**](IMWUT_START_HERE.md)
>
> This branch's active program is the **IMWUT comparison-model line** ("recognise by comparison,
> not classification"), not the movement-monitoring application described below. The reading order
> in this file is the `main`-branch application line, which is still maintained but is **not** the
> current paper target. `IMWUT_START_HERE.md` says which code is live, which is a previous life,
> and which conventions must not be violated.

`main` contains one active research program: personalized movement monitoring from wearable IMU
representations.

## Reading order

1. [**MOTIVATION.md**](design/MOTIVATION.md) - why the project moved from generic open-label HAR to
   applied movement measurement.
2. [**RESEARCH_TASKS.md**](design/RESEARCH_TASKS.md) - the three agreed tasks and their boundaries.
3. [**TASK1_ARBITRARY_DETECTION.md**](tasks/TASK1_ARBITRARY_DETECTION.md) - Task-1 data construction,
   matching, training, and evaluation design.
4. [**TASK2_CHANGE_QUANTIFICATION.md**](tasks/TASK2_CHANGE_QUANTIFICATION.md) - Task-2 personal
   baselines, alignment, measurements, training data, and validation.
5. [**TASK3_RECURRENT_MOTION_DISCOVERY.md**](tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md) - Task-3 motif
   search, clustering, occupational data, review, and evaluation.
6. [**ANNOTATION_INVENTORY.md**](data/ANNOTATION_INVENTORY.md) - the exact temporal supervision,
   background coverage, and current adapter form of every selected source.
7. [**DESIGN_OF_RECORD.md**](design/DESIGN_OF_RECORD.md) - the shared representation and algorithms.
8. [**ENCODER_HYPOTHESES.md**](design/ENCODER_HYPOTHESES.md) - which representation gaps matter for
   Tasks 1-3, which HALO mechanisms may address them, and the required matched ablations.
9. [**EVALUATION_PROTOCOL.md**](design/EVALUATION_PROTOCOL.md) - shared leakage, metrics, controls, and data
   roles.
10. [**IMPLEMENTATION_PLAN.md**](design/IMPLEMENTATION_PLAN.md) - staged build order and exit criteria.
11. [**APPLICATION_DATASETS.md**](data/APPLICATION_DATASETS.md) - which local datasets can answer each
   task and which are contaminated by existing pretraining.
12. [**STORAGE_INVENTORY.md**](data/STORAGE_INVENTORY.md) - dataset footprint, retention policy, and
   verified reclaim candidates.
13. [**BASELINES.md**](baselines/BASELINES.md) - released representations and raw/physical controls.
14. [**RESULTS.md**](results/RESULTS.md) - promoted application results only; currently a pre-result
   protocol record.

## Implementation references

- [DATA_PIPELINE.md](data/DATA_PIPELINE.md) - converter, session, unit, grid, and quality contracts.
- [DATA_HETEROGENEITY.md](data/DATA_HETEROGENEITY.md) - verified per-source sensor semantics.
- [LABELLED_CORPUS_AUDIT_20260909.md](data/LABELLED_CORPUS_AUDIT_20260909.md) - measured audit of the
  18 labelled training sources: validity, defects, composition, and label coverage against the target.
- [PRETRAINING_CORPUS.md](data/PRETRAINING_CORPUS.md) - the label-free `data/pretraining/` tree:
  composition, the float16 and 200 Hz storage decisions, and the fetch/convert/grid procedure.
- [AUGMENTATIONS.md](design/AUGMENTATIONS.md) - available physical transformations; not automatically
  enabled for application tasks.
- [CONTINUOUS_KERNEL_FRONTEND.md](design/CONTINUOUS_KERNEL_FRONTEND.md) - continuous physical-time
  HALO frontend.
- [JEPA_PRETRAINING_OBJECTIVE.md](design/JEPA_PRETRAINING_OBJECTIVE.md) - implemented multi-horizon,
  physically grounded predictive pretraining objective for the HALO encoder.
- [TEXT_CONDITIONING.md](design/TEXT_CONDITIONING.md) - implemented acquisition-description path;
  retained as an encoder reference, not an application claim.
- [`training/tokenizer/README.md`](../training/tokenizer/README.md) - optional representation
  pretraining and diagnostics.

## Documentation policy

- Active claims have exactly one owner document linked above.
- Each application task has one task document that owns its data construction, task-specific
  algorithm, training, and evaluation protocol. Shared rules remain in the common design documents.
- Generated tables do not become design documents.
- A promoted result is summarized only in `results/RESULTS.md` and points to its versioned artifact.
- Old zero-shot and evidence-engine documents are not duplicated in an active-looking archive. They
  remain available at commit `32267b6` and on the branches that use that design.
- Application terms are used precisely: an unconfirmed recurrence is a **motion motif**, latent
  distance is **difference**, and clinical or ergonomic interpretations require external validation.
