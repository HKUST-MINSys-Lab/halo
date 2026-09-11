# External representation adapters

Each retained directory contains the relevant publication material, official repository metadata,
and a thin adapter for an author-released checkpoint.

The active comparison treats every baseline as a representation provider. Its native classifier,
open-vocabulary head, or source-specific task head is not substituted into the common support
protocol. The adapter preserves the model's published input contract and exports an embedding for
the same query/support recordings used by HALO.

Primary roster: `harnet`, `unimts`, and `normwear`.

LiMU-BERT, CrossHAR, and ImageBind are retained only as historical or optional diagnostic assets;
they are not part of the current paper's primary comparison. See
[BASELINES.md](../docs/baselines/BASELINES.md) and
[BASELINE_FAIRNESS_POLICY.md](../docs/baselines/BASELINE_FAIRNESS_POLICY.md).
