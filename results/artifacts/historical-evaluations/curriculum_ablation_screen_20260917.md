# Internal Classifier Development Panel

Subject-held-out development data only. Values are percentages. Clean/rate/dropout reuse one episode panel; each condition-specific panel is fixed across checkpoints.

| checkpoint | condition | enrolled macro-F1 | zero-shot macro-F1 | neighbor acc. | classifier acc. | rescue | overturn | net gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| screen_20260917_corrected_stage_a_3k @ 2500 | clean_mixed | 65.70 | 55.04 | 47.62 | 59.52 | 14.29 | 2.38 | 11.90 |
| screen_20260917_corrected_stage_a_3k @ 2500 | rate_downsample_mixture | 65.12 | 54.13 | 50.79 | 57.94 | 9.52 | 2.38 | 7.14 |
| screen_20260917_corrected_stage_a_3k @ 2500 | gyro_dropout_mixture | 53.09 | 41.22 | 41.27 | 48.41 | 8.73 | 1.59 | 7.14 |
| screen_20260917_corrected_stage_a_3k @ 2500 | clean_acquisition_compatible | 54.74 | nan | 42.70 | 50.27 | 10.81 | 3.24 | 7.57 |
| screen_20260917_corrected_stage_a_3k @ 2500 | clean_acquisition_cross_placement | 50.67 | nan | 39.06 | 43.23 | 8.33 | 4.17 | 4.17 |
| screen_20260917_corrected_stage_a_3k @ 2500 | clean_acquisition_cross_dataset | 70.40 | nan | 55.88 | 75.00 | 19.12 | 0.00 | 19.12 |
| screen_20260917_corrected_stage_a_3k @ 2500 | clean_enrollment_complete | 57.45 | nan | 52.38 | 53.97 | 2.12 | 0.53 | 1.59 |
| screen_20260917_corrected_stage_a_3k @ 2500 | clean_enrollment_partial | 52.07 | nan | 33.51 | 49.47 | 19.15 | 3.19 | 15.96 |
| screen_20260917_corrected_stage_a_3k @ 2500 | clean_enrollment_zero | nan | 49.64 | n/a | n/a | n/a | n/a | n/a |
| screen_20260917_perturbations_only_3k @ 3000 | clean_mixed | 63.04 | 45.42 | 46.03 | 55.56 | 11.90 | 2.38 | 9.52 |
| screen_20260917_perturbations_only_3k @ 3000 | rate_downsample_mixture | 61.98 | 45.03 | 44.44 | 53.97 | 12.70 | 3.17 | 9.52 |
| screen_20260917_perturbations_only_3k @ 3000 | gyro_dropout_mixture | 59.82 | 46.45 | 42.06 | 52.38 | 11.90 | 1.59 | 10.32 |
| screen_20260917_perturbations_only_3k @ 3000 | clean_acquisition_compatible | 50.60 | nan | 38.92 | 47.03 | 11.89 | 3.78 | 8.11 |
| screen_20260917_perturbations_only_3k @ 3000 | clean_acquisition_cross_placement | 47.11 | nan | 35.94 | 40.10 | 11.46 | 7.29 | 4.17 |
| screen_20260917_perturbations_only_3k @ 3000 | clean_acquisition_cross_dataset | 71.34 | nan | 51.47 | 73.53 | 24.26 | 2.21 | 22.06 |
| screen_20260917_perturbations_only_3k @ 3000 | clean_enrollment_complete | 58.94 | nan | 54.50 | 56.08 | 1.59 | 0.00 | 1.59 |
| screen_20260917_perturbations_only_3k @ 3000 | clean_enrollment_partial | 56.48 | nan | 31.38 | 48.94 | 22.34 | 4.79 | 17.55 |
| screen_20260917_perturbations_only_3k @ 3000 | clean_enrollment_zero | nan | 50.73 | n/a | n/a | n/a | n/a | n/a |
| screen_20260917_adaptive_gate_only_3k @ 3000 | clean_mixed | 65.04 | 48.36 | 49.21 | 57.94 | 9.52 | 0.79 | 8.73 |
| screen_20260917_adaptive_gate_only_3k @ 3000 | rate_downsample_mixture | 64.47 | 44.39 | 48.41 | 57.14 | 9.52 | 0.79 | 8.73 |
| screen_20260917_adaptive_gate_only_3k @ 3000 | gyro_dropout_mixture | 53.42 | 36.28 | 37.30 | 46.83 | 10.32 | 0.79 | 9.52 |
| screen_20260917_adaptive_gate_only_3k @ 3000 | clean_acquisition_compatible | 54.13 | nan | 42.70 | 49.19 | 10.81 | 4.32 | 6.49 |
| screen_20260917_adaptive_gate_only_3k @ 3000 | clean_acquisition_cross_placement | 47.91 | nan | 41.15 | 42.19 | 10.42 | 9.38 | 1.04 |
| screen_20260917_adaptive_gate_only_3k @ 3000 | clean_acquisition_cross_dataset | 80.40 | nan | 55.15 | 80.15 | 26.47 | 1.47 | 25.00 |
| screen_20260917_adaptive_gate_only_3k @ 3000 | clean_enrollment_complete | 59.97 | nan | 53.44 | 54.50 | 3.70 | 2.65 | 1.06 |
| screen_20260917_adaptive_gate_only_3k @ 3000 | clean_enrollment_partial | 56.54 | nan | 32.98 | 52.13 | 24.47 | 5.32 | 19.15 |
| screen_20260917_adaptive_gate_only_3k @ 3000 | clean_enrollment_zero | nan | 56.83 | n/a | n/a | n/a | n/a | n/a |
| screen_20260917_perturbations_plus_gate_3k @ 3000 | clean_mixed | 62.60 | 48.69 | 50.00 | 53.17 | 9.52 | 6.35 | 3.17 |
| screen_20260917_perturbations_plus_gate_3k @ 3000 | rate_downsample_mixture | 61.04 | 49.29 | 48.41 | 53.17 | 10.32 | 5.56 | 4.76 |
| screen_20260917_perturbations_plus_gate_3k @ 3000 | gyro_dropout_mixture | 57.78 | 47.78 | 42.86 | 48.41 | 7.94 | 2.38 | 5.56 |
| screen_20260917_perturbations_plus_gate_3k @ 3000 | clean_acquisition_compatible | 46.91 | nan | 38.92 | 47.03 | 10.27 | 2.16 | 8.11 |
| screen_20260917_perturbations_plus_gate_3k @ 3000 | clean_acquisition_cross_placement | 48.25 | nan | 38.02 | 42.71 | 9.90 | 5.21 | 4.69 |
| screen_20260917_perturbations_plus_gate_3k @ 3000 | clean_acquisition_cross_dataset | 71.20 | nan | 50.74 | 70.59 | 22.79 | 2.94 | 19.85 |
| screen_20260917_perturbations_plus_gate_3k @ 3000 | clean_enrollment_complete | 57.57 | nan | 55.56 | 54.50 | 1.06 | 2.12 | -1.06 |
| screen_20260917_perturbations_plus_gate_3k @ 3000 | clean_enrollment_partial | 51.87 | nan | 31.91 | 47.87 | 22.34 | 6.38 | 15.96 |
| screen_20260917_perturbations_plus_gate_3k @ 3000 | clean_enrollment_zero | nan | 49.91 | n/a | n/a | n/a | n/a | n/a |

`rescue` means neighbor wrong and classifier correct. `overturn` means neighbor correct and classifier wrong. `net gain = classifier accuracy - neighbor accuracy` on enrolled episodes only.
