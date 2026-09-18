# Fresh HALO sealed evaluation: differentiable-neighbours control

Run date: 2026-09-18

Sealed aggregate evaluation of the HALO encoder trained end to end with the parameter-free differentiable-neighbours control (readouts retained: cosine `1nn`; zero enrollment uses the training-bank/ConSE bridge), under
`sealed-manifest-v2-20260916` and feature-cache schema `sealed-feature-v6-20260918`: six sealed datasets,
4/8/16-second evidence windows, `k = 0, 1, 2, 4, 8, 16, 32, 64, 128`.

All 39 protocol cells completed in 15.1 minutes: 933 scored rows,
318 explicit `n/a` rows, no failed rows. Every cell shared with
[`baselines_sealed_v5_20260918`](../baselines_sealed_v5_20260918/) carries an identical manifest fingerprint
(333 of 333), so the two artifacts are directly joinable.

Training: Stage A curriculum at the trainer defaults, 40,000 optimizer steps, 4 episodes per step, enrollment counts {1,2,4,8,16,32}, acquisition mix 0.5/0.25/0.25, enrollment mix 0.5/0.25/0.25, joint device-set challenge probability 0.5, up to 4 devices, seed 20260901, bf16, 16 loader workers, `--encode-chunk-rows 256`. No rate augmentation, no gyroscope dropout, no adaptive text gate. Conditioning schema `acquisition-conditioning-v2`. Corpus fingerprint `dcd2cd874ec7b7ce`. Checkpoint policy declared before any sealed evaluation: the final step-40,000 checkpoint (`last.pt`).
Training run directory: `runs/support-classifier/halo_neighbors_stage_a_40k_20260918` (not tracked); `training_run_config.json` here is
its exact configuration. Checkpoint `last.pt` SHA-256 `8d63871eb15557631acbf8caa1cbb4e405297ba3d7146bfd209cec13cb7fcdb3`.
Evaluator commit `4212cf5a72811932b562c3d30836e0277074b53f`, clean tree.

`results.json.gz` is the exhaustive row payload, `episode_manifests.json.gz` the deterministic episodes,
`RESULTS.md` the table generated from those rows, and `source_hashes.txt` the uncompressed hashes.
