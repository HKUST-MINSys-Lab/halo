# External representation adapters

Each retained directory contains the relevant publication material, official repository metadata,
and a thin adapter for an author-released checkpoint.

The active comparison treats every baseline as a representation provider. Its native classifier,
open-vocabulary head, or source-specific task head is not substituted into the common support
protocol. The adapter preserves the model's published input contract and exports an embedding for
the same query/support recordings used by HALO.

Primary roster: `harnet`, `limubert_x`, `unimts`, and `normwear`.

The original locally pretrained LiMU-BERT, CrossHAR, and ImageBind are retained only as historical
or optional diagnostic assets. LiMU-BERT-X is distinct: its primary row uses the authors' released
large-scale checkpoint and no locally fitted prediction head. It has no native open-label output,
so its zero-support cell is `N/A`; enrollment readouts operate on its frozen representation. The
historical models are not part of the current paper's primary comparison. See
[BASELINES.md](../docs/baselines/BASELINES.md) and
[BASELINE_FAIRNESS_POLICY.md](../docs/baselines/BASELINE_FAIRNESS_POLICY.md).
