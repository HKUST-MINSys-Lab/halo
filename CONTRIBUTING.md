# Contributing

Last verified: 2026-09-24.

**All work happens on `main`**, and is pushed to the lab repository (`origin`). No feature, fix or
experiment branches. Before retiring code or an experiment, tag the last state that contains it as
an annotated `hist/<era>/<what>-<yyyymmdd>` tag and push the tag; see
[docs/overview/history.md](docs/overview/history.md). Summarize every experiment in the journal.

Parallel agents may share `main`: pull before committing, commit small, and do not run a
repository-wide refactor while a training or evaluation run depends on the checkout. A worktree is
fine for an isolated run, but its work comes back to `main`, not to a long-lived branch.

Before committing:

Before merging:

1. Run `uv sync --extra model --extra dev`; add `--extra baselines` when checking released adapters.
2. Run focused tests while editing, then `uv run pytest -q`.
3. Run `git diff --check` and confirm no generated data, checkpoints, caches, or debug plots are
   tracked. Promoted result JSON/Markdown and publication figures belong under `results/artifacts/`.
4. Update living contracts in `docs/contracts/` and the overview in `docs/overview/` when behavior
   changes. Add a dated journal entry for an experiment or decision; do not rewrite old journal
   conclusions (index: `python tools/generate_journal_index.py`).

Runtime locations are owned by `halo/paths.py`. Use its constants instead of deriving the checkout
root from `__file__`; large paths can be redirected with the corresponding `HALO_*` variables.
