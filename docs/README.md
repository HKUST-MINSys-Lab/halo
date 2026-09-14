# HALO documentation

Start with [START_HERE.md](START_HERE.md). It names the active system, source of truth for the
pretraining corpus, and the exact separation between live and archived work.

Live documents:

- [Readiness repair plan](design/READINESS_REPAIR_PLAN_20260913.md): consolidated verified
  training/evaluation fixes and acceptance gates; pending implementation, not a new model design;
- `design/DESIGN_OF_RECORD.md`: current encoder and support-classifier architecture;
- `design/EXPERIMENT_ROADMAP.md`: active experiment sequence and result-promotion rules;
- `design/EVALUATION_PROTOCOL.md`: support-classification splits, metrics, and reporting rules;
- `baselines/`: retained released-checkpoint roster and comparison policy; and
- `results/RESULTS.md`: promoted result record.

Use [HISTORY.md](HISTORY.md) and the dated [journal](journal/README.md) for retired work,
including the JEPA design, corpus, and results. Do not restore archived documents into this tree
as live design material.
