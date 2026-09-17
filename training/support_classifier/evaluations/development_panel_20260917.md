# Internal Classifier Development Panel

Subject-held-out development data only. Values are percentages. The same deterministic episode panel is reused across recording conditions.

| checkpoint | condition | enrolled macro-F1 | zero-shot macro-F1 | neighbor acc. | classifier acc. | rescue | overturn | net gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| halo_fixed_mr_residual_v3_curriculum12_40k_20260916 @ 30000 | clean | 71.84 | 73.68 | 55.56 | 66.67 | 15.87 | 4.76 | 11.11 |
| halo_fixed_mr_residual_v3_curriculum12_40k_20260916 @ 30000 | rate_downsample_mixture | 72.85 | 69.40 | 56.35 | 68.25 | 18.25 | 6.35 | 11.90 |
| halo_fixed_mr_residual_v3_curriculum12_40k_20260916 @ 30000 | gyro_dropout_mixture | 69.43 | 54.03 | 46.83 | 63.49 | 20.63 | 3.97 | 16.67 |
| halo_fixed_mr_residual_v3_curriculum1234_40k_20260916 @ 15000 | clean | 70.81 | 65.55 | 55.56 | 65.08 | 12.70 | 3.17 | 9.52 |
| halo_fixed_mr_residual_v3_curriculum1234_40k_20260916 @ 15000 | rate_downsample_mixture | 69.70 | 65.79 | 56.35 | 64.29 | 11.11 | 3.17 | 7.94 |
| halo_fixed_mr_residual_v3_curriculum1234_40k_20260916 @ 15000 | gyro_dropout_mixture | 66.55 | 63.68 | 50.00 | 60.32 | 14.29 | 3.97 | 10.32 |

`rescue` means neighbor wrong and classifier correct. `overturn` means neighbor correct and classifier wrong. `net gain = classifier accuracy - neighbor accuracy` on enrolled episodes only.
