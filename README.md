# HALO

HALO is a research codebase for support-conditioned human-activity recognition from heterogeneous
inertial measurement unit (IMU) recordings. The current question is practical: can a
physical-time encoder and a simple comparison head recognize a bounded set of activities when it
is given zero or a small number of labelled support recordings?

```text
native-rate IMU + honest channel metadata
    -> rate-aware HALO encoder
    -> one representation per recording
    -> compare a query with labelled supports
    -> score the supplied candidate activities
```

The project does not claim unconstrained activity recognition from arbitrary label text. The
support set defines the available evidence: zero-support and enrolled-support conditions are
separate, explicitly reported evaluation regimes.

## Current scope

- **Encoder arms:** a one-second fixed physical filterbank control, a fixed multiresolution
  filterbank (0.5, 1.0, and 1.5 seconds), and a continuous multispan frontend at the same spans.
- **No active label-free pretraining:** future-JEPA was evaluated and retired because supervised
  end-to-end encoder adaptation recovered its observed gain. The reproducibility code and results
  remain archived behind an explicit command-line acknowledgment.
- **Classifier:** a semantic token mixer with separate zero-support and enrolled-support weights.
  With enrollment it jointly contextualises query, support, paired support-label, and candidate-label
  tokens before a soft support vote. The parameter-free `neighbors` path is the encoder control.
  Neither path contains the retired admissibility gate or hidden memory-bank retrieval machinery.
- **External comparisons:** author-released HARNet, UniMTS, and NormWear checkpoints, adapted
  faithfully and evaluated under the same support protocol where their input contracts permit it.

Read [the documentation entry point](docs/START_HERE.md) before configuring a run.

## Layout

```text
baselines/                    # retained released-checkpoint adapters and publications
data/                         # labelled data, curation, and label-free pretraining sources
model/tokenizer/              # HALO frontends and contextual encoder
model/support/                # current support-conditioned comparison head
training/tokenizer/           # encoder utilities and archived JEPA reproducibility code
training/support_classifier/  # support-classifier training and validation
docs/                         # current design, data, protocol, and results record
tests/                        # regression tests for the retained surface
```

The previous language-alignment, explicit-admissibility, Phase-B memory-bank evidence engine, and
movement-monitoring application pivots are archived in Git. They are not live implementation
guidance. See [docs/HISTORY.md](docs/HISTORY.md).

## Development

Use the project interpreter for Torch and scientific dependencies:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python -m pytest tests -q
```

Raw downloads, caches, checkpoints, and generated run outputs are ignored by Git. Promoted
protocols and result summaries are tracked with their code.
