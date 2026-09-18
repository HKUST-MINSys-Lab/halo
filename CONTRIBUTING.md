# Contributing

`main` is the protected integration branch. Use `feat/<topic>` for features, `fix/<topic>` for
repairs, and `exp/<topic>-<yyyymmdd>` for experiments. Experiment branches must be summarized in
the journal and tagged or deleted within one week of recording their result.

Parallel agents use separate branches and worktrees. Do not share a mutable `main` checkout while
training, evaluation, or a repository-wide refactor is in progress.

Before merging:

1. Run `uv sync --extra model --extra dev`.
2. Run focused tests while editing, then `uv run pytest -q`.
3. Run `git diff --check` and confirm no generated data, checkpoints, caches, or debug plots are
   tracked. Promoted result JSON/Markdown and publication figures belong under `results/artifacts/`.
4. Update living contracts in `docs/contracts/` when behavior changes. Add a dated journal entry
   for an experiment or historical decision; do not rewrite old journal conclusions.

Runtime locations are owned by `halo/paths.py`. Use its constants instead of deriving the checkout
root from `__file__`; large paths can be redirected with the corresponding `HALO_*` variables.
