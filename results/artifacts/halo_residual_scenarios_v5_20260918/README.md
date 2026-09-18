# Fresh HALO deployment-scenario evaluation: residual classifier

Run date: 2026-09-18

`deployment-scenarios-v5-20260918` evaluation of the HALO encoder trained end to end with the learned residual support classifier (three readouts retained: `halo-classifier`, its base `halo-classifier-residual-off`, and cosine `1nn`):
seven active scenarios, 8-second evidence, `k = 0, 1, 4, 8, 32`.

`run_metadata.json` records 2395 result rows, 0 task failures and
0 failed rows (2338 scored; the remainder are explicit unsupported, `n/a`
or inapplicable rows). Every scenario cell shared with
[`scenarios_baselines_v5_20260918`](../scenarios_baselines_v5_20260918/) carries an identical manifest
fingerprint, so paired comparisons against the released baselines are exact.

Training: Stage A curriculum at the trainer defaults, 40,000 optimizer steps, 4 episodes per step, enrollment counts {1,2,4,8,16,32}, acquisition mix 0.5/0.25/0.25, enrollment mix 0.5/0.25/0.25, joint device-set challenge probability 0.5, up to 4 devices, seed 20260901, bf16, 16 loader workers, `--encode-chunk-rows 256`. No rate augmentation, no gyroscope dropout, no adaptive text gate. Conditioning schema `acquisition-conditioning-v2`. Corpus fingerprint `dcd2cd874ec7b7ce`. Checkpoint policy declared before any sealed evaluation: the final step-40,000 checkpoint (`last.pt`).
Training run directory: `runs/support-classifier/halo_residual_stage_a_40k_20260918` (not tracked); `training_run_config.json` here is
its exact configuration. Checkpoint `last.pt` SHA-256 `56b2cb29fdf093bee26292987fae1048b701ef8560643f2e9868d7a7acf5cc74`.
Evaluator commit and environment are in `run_provenance.json`.

`results.json.gz` is the exhaustive metric payload, `manifests.jsonl.gz` the deterministic episodes,
`paired_deltas.json` the matched-control comparisons, `RESULTS.md` the generated table and
`source_hashes.txt` the uncompressed hashes.
