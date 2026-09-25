# Tier 3 (rung 3): adaptation with k labelled windows per class

Dataset-balanced macro-F1 on the fixed scored set (mean over support draws; ± = mean within-cell std across draws). Heads sit on each model's own features.

## enrollment_frozen

| encoder | k=1 | k=4 | k=16 | trainable params |
|---|---:|---:|---:|---:|
| halo | 53.9 ± 5.3 | 67.1 ± 2.8 | 71.3 ± 1.4 | 0 |
| harnet10 | 33.1 ± 3.6 | 41.4 ± 3.0 | 50.2 ± 3.0 | 0 |
| harnet5 | 36.1 ± 4.3 | 43.2 ± 3.4 | 50.1 ± 2.7 | 0 |
| limubert_x | 42.0 ± 8.0 | 50.0 ± 4.7 | 56.6 ± 3.0 | 0 |
| normwear | 41.7 ± 4.5 | 54.7 ± 3.7 | 61.2 ± 2.4 | 0 |
| unimts | 51.5 ± 7.3 | 55.8 ± 5.8 | 61.1 ± 3.1 | 0 |

## linear_probe

| encoder | k=1 | k=4 | k=16 | trainable params |
|---|---:|---:|---:|---:|
| halo | 53.5 ± 5.2 | 66.8 ± 2.5 | 73.3 ± 1.5 | 903 |
| harnet10 | 33.0 ± 4.0 | 41.2 ± 3.6 | 53.4 ± 2.9 | 7175 |
| harnet5 | 34.7 ± 4.4 | 41.2 ± 3.5 | 51.3 ± 3.0 | 3591 |
| limubert_x | 34.7 ± 7.4 | 41.9 ± 4.4 | 50.6 ± 2.9 | 511 |
| normwear | 40.2 ± 4.9 | 43.6 ± 3.5 | 47.9 ± 3.3 | 32263 |
| unimts | 45.4 ± 5.3 | 51.6 ± 5.3 | 60.8 ± 2.5 | 3591 |

## full_finetune

| encoder | k=1 | k=4 | k=16 | trainable params |
|---|---:|---:|---:|---:|
| halo | 50.1 ± 0.0 | 69.7 ± 0.0 | 76.2 ± 0.0 | 840993 |
| harnet10 | - | - | - | - |
| harnet5 | 42.6 ± 0.0 | 54.9 ± 0.0 | 68.1 ± 0.0 | 4231488 |
| limubert_x | 45.1 ± 0.0 | 58.3 ± 0.0 | 60.8 ± 0.0 | 55950 |
| normwear | - | - | - | - |
| unimts | 50.2 ± 0.0 | 64.4 ± 0.0 | 67.4 ± 0.0 | 5183484 |

## scratch_specialist

| encoder | k=1 | k=4 | k=16 | trainable params |
|---|---:|---:|---:|---:|
| halo | 42.2 ± 0.0 | 59.6 ± 0.0 | 73.2 ± 0.0 | 840993 |
| harnet10 | - | - | - | - |
| harnet5 | 36.3 ± 0.0 | 47.9 ± 0.0 | 63.3 ± 0.0 | 4231488 |
| limubert_x | 41.6 ± 0.0 | 52.2 ± 0.0 | 61.7 ± 0.0 | 55950 |
| normwear | - | - | - | - |
| unimts | 34.7 ± 0.0 | 44.0 ± 0.0 | 59.9 ± 0.0 | 5183484 |

## Per dataset (mean over cells and draws)

### enrollment_frozen

| encoder | k | inclusivehar | motionsense | realworld | shoaib | usc_had | ut_complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| halo | 1 | 28.9 | 74.3 | 50.5 | 76.8 | 46.0 | 47.0 |
| halo | 4 | 41.6 | 86.9 | 64.8 | 85.6 | 62.8 | 60.7 |
| halo | 16 | 47.5 | 91.4 | 71.3 | 86.7 | 66.3 | 64.5 |
| harnet10 | 1 | 21.8 | 46.7 | 28.0 | 45.4 | 25.4 | 31.4 |
| harnet10 | 4 | 26.1 | 60.5 | 34.8 | 56.1 | 32.8 | 37.8 |
| harnet10 | 16 | 30.5 | 74.2 | 44.8 | 63.0 | 41.6 | 47.2 |
| harnet5 | 1 | 27.8 | 54.1 | 27.9 | 50.4 | 30.6 | 25.7 |
| harnet5 | 4 | 21.3 | 68.5 | 38.3 | 59.4 | 33.2 | 38.4 |
| harnet5 | 16 | 29.6 | 72.0 | 45.9 | 66.2 | 41.0 | 46.1 |
| limubert_x | 1 | 17.8 | 47.7 | 40.6 | 64.5 | 38.1 | 43.4 |
| limubert_x | 4 | 15.4 | 66.8 | 48.4 | 66.8 | 59.8 | 43.1 |
| limubert_x | 16 | 20.8 | 74.5 | 52.7 | 74.1 | 66.6 | 51.2 |
| normwear | 1 | 25.5 | 55.3 | 36.6 | 55.8 | 32.6 | 44.3 |
| normwear | 4 | 30.2 | 68.3 | 51.3 | 73.4 | 49.0 | 55.9 |
| normwear | 16 | 33.1 | 78.2 | 61.8 | 77.5 | 54.3 | 62.2 |
| unimts | 1 | 32.6 | 67.3 | 50.5 | 70.3 | 41.4 | 47.1 |
| unimts | 4 | 33.7 | 69.7 | 58.7 | 70.2 | 49.7 | 52.7 |
| unimts | 16 | 34.7 | 86.2 | 63.4 | 73.6 | 52.3 | 56.5 |

### linear_probe

| encoder | k | inclusivehar | motionsense | realworld | shoaib | usc_had | ut_complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| halo | 1 | 26.7 | 73.7 | 50.5 | 74.9 | 46.4 | 48.7 |
| halo | 4 | 41.0 | 87.3 | 64.4 | 85.9 | 63.2 | 59.2 |
| halo | 16 | 50.2 | 93.4 | 73.9 | 88.4 | 67.8 | 65.8 |
| harnet10 | 1 | 22.4 | 48.0 | 27.2 | 46.0 | 24.2 | 30.0 |
| harnet10 | 4 | 23.8 | 61.9 | 33.5 | 58.2 | 33.4 | 36.7 |
| harnet10 | 16 | 30.5 | 78.1 | 47.6 | 68.1 | 43.4 | 52.7 |
| harnet5 | 1 | 29.2 | 53.0 | 27.4 | 46.3 | 28.4 | 23.9 |
| harnet5 | 4 | 21.7 | 69.5 | 36.8 | 56.4 | 30.8 | 31.8 |
| harnet5 | 16 | 34.9 | 73.5 | 46.8 | 66.4 | 41.8 | 44.6 |
| limubert_x | 1 | 19.9 | 43.3 | 34.5 | 48.3 | 30.7 | 31.4 |
| limubert_x | 4 | 16.4 | 56.2 | 44.2 | 54.7 | 48.8 | 31.3 |
| limubert_x | 16 | 24.8 | 63.9 | 53.0 | 69.8 | 52.6 | 39.8 |
| normwear | 1 | 21.4 | 61.3 | 35.3 | 49.5 | 31.6 | 42.4 |
| normwear | 4 | 23.0 | 62.9 | 40.2 | 58.2 | 36.0 | 41.0 |
| normwear | 16 | 28.1 | 66.1 | 47.0 | 63.5 | 42.5 | 40.3 |
| unimts | 1 | 28.6 | 59.5 | 42.9 | 62.0 | 36.4 | 42.6 |
| unimts | 4 | 34.5 | 63.2 | 54.3 | 68.5 | 44.9 | 44.1 |
| unimts | 16 | 35.7 | 87.7 | 61.6 | 75.8 | 52.8 | 51.1 |

### full_finetune

| encoder | k | inclusivehar | motionsense | realworld | shoaib | usc_had | ut_complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| halo | 1 | 34.8 | 57.6 | 49.1 | 62.3 | 43.0 | 54.0 |
| halo | 4 | 40.5 | 87.7 | 66.7 | 85.8 | 67.7 | 69.8 |
| halo | 16 | 48.1 | 92.2 | 76.2 | 88.8 | 76.7 | 75.5 |
| harnet5 | 1 | 20.1 | 58.7 | 43.1 | 55.3 | 36.8 | 41.3 |
| harnet5 | 4 | 13.6 | 70.7 | 52.1 | 75.7 | 61.3 | 56.1 |
| harnet5 | 16 | 30.7 | 88.1 | 63.2 | 86.2 | 70.3 | 70.2 |
| limubert_x | 1 | 23.1 | 54.9 | 45.3 | 55.5 | 46.2 | 45.4 |
| limubert_x | 4 | 30.4 | 65.0 | 55.6 | 84.2 | 59.6 | 55.0 |
| limubert_x | 16 | 35.2 | 80.7 | 55.5 | 84.1 | 59.0 | 50.5 |
| unimts | 1 | 29.9 | 61.0 | 45.4 | 60.6 | 52.0 | 52.4 |
| unimts | 4 | 46.2 | 79.0 | 68.1 | 79.7 | 51.9 | 61.7 |
| unimts | 16 | 29.7 | 87.7 | 70.7 | 91.0 | 59.3 | 66.3 |

### scratch_specialist

| encoder | k | inclusivehar | motionsense | realworld | shoaib | usc_had | ut_complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| halo | 1 | 29.3 | 50.3 | 42.6 | 59.1 | 35.9 | 36.1 |
| halo | 4 | 38.0 | 72.8 | 54.7 | 82.6 | 56.9 | 52.7 |
| halo | 16 | 52.6 | 88.1 | 68.9 | 86.8 | 72.9 | 70.0 |
| harnet5 | 1 | 27.7 | 31.9 | 29.9 | 45.0 | 36.4 | 46.9 |
| harnet5 | 4 | 9.0 | 62.4 | 39.7 | 65.2 | 59.0 | 52.3 |
| harnet5 | 16 | 25.2 | 79.7 | 60.9 | 80.5 | 70.7 | 63.1 |
| limubert_x | 1 | 24.7 | 54.6 | 35.2 | 45.9 | 45.1 | 43.9 |
| limubert_x | 4 | 14.3 | 61.1 | 42.3 | 73.7 | 64.1 | 57.6 |
| limubert_x | 16 | 29.5 | 85.3 | 51.2 | 79.8 | 67.0 | 57.2 |
| unimts | 1 | 19.2 | 44.1 | 27.4 | 44.7 | 33.9 | 38.6 |
| unimts | 4 | 29.8 | 30.7 | 49.1 | 62.0 | 38.8 | 53.7 |
| unimts | 16 | 30.4 | 78.0 | 62.5 | 80.1 | 53.2 | 55.4 |

## Fit diagnostics

Coverage and cost per (encoder, method). A final loss far above 0 means the head did not fit its supports within the fixed budget.

| encoder | method | cells | draws | rows | median fit s | mean final loss |
|---|---|---:|---:|---:|---:|---:|
| halo | enrollment_frozen | 11 | 3 | 99 | - | - |
| halo | linear_probe | 11 | 3 | 99 | - | - |
| halo | full_finetune | 11 | 1 | 33 | 21.33 | 0.003 |
| halo | scratch_specialist | 11 | 1 | 33 | 21.80 | 0.040 |
| harnet10 | enrollment_frozen | 11 | 3 | 99 | - | - |
| harnet10 | linear_probe | 11 | 3 | 99 | - | - |
| harnet5 | enrollment_frozen | 11 | 3 | 99 | - | - |
| harnet5 | linear_probe | 11 | 3 | 99 | - | - |
| harnet5 | full_finetune | 11 | 1 | 33 | 13.54 | 0.012 |
| harnet5 | scratch_specialist | 11 | 1 | 33 | 12.65 | 0.054 |
| limubert_x | enrollment_frozen | 11 | 3 | 99 | - | - |
| limubert_x | linear_probe | 11 | 3 | 99 | - | - |
| limubert_x | full_finetune | 11 | 1 | 33 | 15.18 | 0.495 |
| limubert_x | scratch_specialist | 11 | 1 | 33 | 15.71 | 0.238 |
| normwear | enrollment_frozen | 11 | 3 | 99 | - | - |
| normwear | linear_probe | 11 | 3 | 99 | - | - |
| unimts | enrollment_frozen | 11 | 3 | 99 | - | - |
| unimts | linear_probe | 11 | 3 | 99 | - | - |
| unimts | full_finetune | 11 | 1 | 33 | 41.79 | 0.037 |
| unimts | scratch_specialist | 11 | 1 | 33 | 31.10 | 0.069 |

