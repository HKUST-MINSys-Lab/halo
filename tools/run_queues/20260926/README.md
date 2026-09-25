# Run queues, night of 2026-09-25/26 (HKT)

Verbatim copies of the job-queue scripts that produced the tier-1 and tier-3 runs of that night
(the originals live in the git-ignored `runs/evaluations/`). They encode the GPU-memory budgeting:
jobs wait on each other, and tier-3 shards run under a per-process memory cap
(`capped_rung3.py`, `GPU_GIB`). See `docs/journal/2026-09-26-overnight-tier1-tier3-runs.md` and the
runbook `docs/overview/experiments.md`.

| script | what |
|---|---|
| `queue_tier1_controls.sh` | T1-B: μ = 0, balanced pool, disjoint classes, in sequence |
| `queue_tier1_telemetry.sh` | T1-A re-run with per-label telemetry |
| `queue_after_training.sh` | T1-D (last, best_internal), T1-E (1-NN floor), T1-F (matched HARNet train + score) |
| `queue_t1d_t30.sh` | T1-D secondary diagnostic at T = 30 |
| `queue_rung3_phaseA.sh MODEL GIB` | T3-A draw 0 for one model |
| `queue_rung3_phaseA_rest.sh` | staggered launch of the HALO and HARNet-5 draw-0 shards |
| `queue_rung3_phaseA_d12.sh MODEL GIB` | T3-A draws 1–2 for one model, after its draw 0 |
| `resume_unimts_after_training.sh` | SIGCONT for the UniMTS shard paused while training ran |
| `capped_rung3.py` | `halo-rung3` under `torch.cuda.set_per_process_memory_fraction` |
