# HALO documentation

Start with [START_HERE.md](START_HERE.md). It names the active system, source of truth for the
pretraining corpus, and the exact separation between live and archived work.

Live documents:

- `design/DESIGN_OF_RECORD.md`: current encoder and support-classifier architecture;
- `design/JEPA_PRETRAINING_OBJECTIVE.md`: optional label-free future-JEPA objective;
- `design/JEPA_REPRESENTATION_EVALUATION.md`: frozen and encoder-only JEPA comparison harness;
- `design/CONTINUOUS_KERNEL_FRONTEND.md`: continuous multispan encoder arm;
- `design/EXPERIMENT_ROADMAP.md`: active experiment sequence and result-promotion rules;
- `design/EVALUATION_PROTOCOL.md`: support-classification splits, metrics, and reporting rules;
- `data/PRETRAINING_CORPUS.md`: selected label-free corpus, exposure budget, and measured runtime;
- `baselines/`: retained released-checkpoint roster and comparison policy; and
- `results/RESULTS.md`: promoted result record.

Use [HISTORY.md](HISTORY.md) for retired work. Do not restore archived documents into this tree as
live design material.
