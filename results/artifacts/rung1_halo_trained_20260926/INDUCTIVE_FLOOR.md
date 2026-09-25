# Inductive floor: cosine 1-NN k-curve, candidate vs reference

reference: `results/artifacts/halo_evidence_gated_v4_step40k_sealed_v5_20260920`  
candidate: `runs/evaluations/sealed_1nn_halo_rung1_trained_20260925`  
window 8 s; 13 shared cells.

| k | reference | candidate | Δ | cells below reference by > 5 |
|---:|---:|---:|---:|---:|
| 1 | 60.3 | 58.5 | -1.9 | 2 |
| 8 | 70.9 | 68.7 | -2.2 | 1 |
| 32 | 74.6 | 72.1 | -2.5 | 2 |
| 128 | 76.7 | 74.7 | -2.0 | 0 |

## Per cell

| dataset / stream | k | reference | candidate |
|---|---:|---:|---:|
| inclusivehar / phone_waist | 1 | 32.1 | 29.2 |
| inclusivehar / phone_waist | 8 | 36.8 | 32.9 |
| inclusivehar / phone_waist | 32 | 36.7 | 32.4 |
| inclusivehar / phone_waist | 128 | 36.7 | 32.9 |
| motionsense / phone_front_pocket | 1 | 74.5 | 73.0 |
| motionsense / phone_front_pocket | 8 | 86.3 | 83.3 |
| motionsense / phone_front_pocket | 32 | 91.1 | 85.8 |
| motionsense / phone_front_pocket | 128 | 93.5 | 90.5 |
| realworld / phone_forearm | 1 | 55.0 | 54.7 |
| realworld / phone_forearm | 8 | 65.3 | 64.6 |
| realworld / phone_forearm | 32 | 67.6 | 68.1 |
| realworld / phone_forearm | 128 | 69.3 | 70.8 |
| realworld / phone_forearm+phone_thigh+phone_waist | 1 | 65.7 | 62.3 |
| realworld / phone_forearm+phone_thigh+phone_waist | 8 | 75.9 | 71.0 |
| realworld / phone_forearm+phone_thigh+phone_waist | 32 | 79.0 | 74.7 |
| realworld / phone_forearm+phone_thigh+phone_waist | 128 | 80.5 | 77.0 |
| realworld / phone_thigh | 1 | 46.5 | 47.6 |
| realworld / phone_thigh | 8 | 61.7 | 58.4 |
| realworld / phone_thigh | 32 | 65.1 | 61.9 |
| realworld / phone_thigh | 128 | 65.7 | 62.6 |
| realworld / phone_waist | 1 | 68.2 | 61.6 |
| realworld / phone_waist | 8 | 79.8 | 75.7 |
| realworld / phone_waist | 32 | 81.1 | 79.9 |
| realworld / phone_waist | 128 | 81.3 | 81.2 |
| shoaib / phone_belt | 1 | 81.6 | 73.6 |
| shoaib / phone_belt | 8 | 89.7 | 83.9 |
| shoaib / phone_belt | 32 | 91.8 | 85.6 |
| shoaib / phone_belt | 128 | 91.8 | 87.2 |
| shoaib / phone_left_pocket | 1 | 85.6 | 82.0 |
| shoaib / phone_left_pocket | 8 | 93.3 | 91.2 |
| shoaib / phone_left_pocket | 32 | 95.5 | 94.4 |
| shoaib / phone_left_pocket | 128 | 97.0 | 96.0 |
| shoaib / phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 1 | 87.3 | 90.2 |
| shoaib / phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 8 | 95.8 | 97.2 |
| shoaib / phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 32 | 97.0 | 98.5 |
| shoaib / phone_left_pocket+phone_right_pocket+watch_wrist_proxy+phone_belt | 128 | 97.1 | 98.5 |
| shoaib / phone_right_pocket | 1 | 85.3 | 81.4 |
| shoaib / phone_right_pocket | 8 | 96.4 | 92.3 |
| shoaib / phone_right_pocket | 32 | 97.2 | 95.8 |
| shoaib / phone_right_pocket | 128 | 98.4 | 97.2 |
| shoaib / watch_wrist_proxy | 1 | 74.6 | 72.7 |
| shoaib / watch_wrist_proxy | 8 | 79.7 | 82.0 |
| shoaib / watch_wrist_proxy | 32 | 84.5 | 84.0 |
| shoaib / watch_wrist_proxy | 128 | 84.9 | 84.9 |
| usc_had / phone_hip | 1 | 51.2 | 50.5 |
| usc_had / phone_hip | 8 | 66.5 | 64.1 |
| usc_had / phone_hip | 32 | 74.8 | 71.1 |
| usc_had / phone_hip | 128 | 79.8 | 78.4 |
| ut_complex / watch_wrist | 1 | 62.4 | 61.5 |
| ut_complex / watch_wrist | 8 | 74.1 | 75.0 |
| ut_complex / watch_wrist | 32 | 78.3 | 80.4 |
| ut_complex / watch_wrist | 128 | 81.9 | 80.8 |
