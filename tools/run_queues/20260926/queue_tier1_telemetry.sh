#!/usr/bin/env bash
# Re-run the main tier-1 evaluation with per-label F1 / predicted-class telemetry, after the controls.
set -u
cd /home/alex/code/HALO/halo
while pgrep -f queue_tier1_controls.sh > /dev/null; do sleep 30; done
MAIN=runs/evaluations/rung1_tier1_20260925
OUT=runs/evaluations/rung1_tier1_telemetry_20260925; mkdir -p $OUT
.venv/bin/halo-rung1 --out $OUT --halo-checkpoint runs/support-classifier/halo_t6_40k_20260920/last.pt --cache-read-dirs cache/evaluations/feature_cache_halo_t6_40k_20260920_final40k cache/evaluations/feature_cache_baselines_v2_20260917_final cache/evaluations/feature_cache_baselines_v2_20260917 --feature-cache runs/evaluations/rung1_cache --feature-memory-cache-gib 8 --temperature-file $MAIN/temperature_calibration_w8.json --resume > $OUT/run.log 2>&1
echo "$(date) telemetry rerun exit $?" >> runs/evaluations/queue_tier1_controls.log
