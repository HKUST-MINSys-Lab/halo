#!/usr/bin/env bash
# Tier-1 controls, run one after another after the main tier-1 evaluation (GPU memory budget).
set -u
cd /home/alex/code/HALO/halo
MAIN=runs/evaluations/rung1_tier1_20260925
while pgrep -f "halo-rung1 --out $MAIN " > /dev/null; do sleep 30; done
COMMON="--halo-checkpoint runs/support-classifier/halo_t6_40k_20260920/last.pt --cache-read-dirs cache/evaluations/feature_cache_halo_t6_40k_20260920_final40k cache/evaluations/feature_cache_baselines_v2_20260917_final cache/evaluations/feature_cache_baselines_v2_20260917 --feature-cache runs/evaluations/rung1_cache --feature-memory-cache-gib 8 --temperature-file $MAIN/temperature_calibration_w8.json --resume"
for spec in "mu0:--affinity-mu 0" "balanced:--control balanced_pool" "disjoint:--control disjoint_classes"; do
  name=${spec%%:*}; extra=${spec#*:}
  OUT=runs/evaluations/rung1_tier1_${name}_20260925
  mkdir -p $OUT
  echo "[queue] $(date) start $name" >> runs/evaluations/queue_tier1_controls.log
  .venv/bin/halo-rung1 --out $OUT $COMMON $extra > $OUT/run.log 2>&1
  echo "[queue] $(date) end $name exit $?" >> runs/evaluations/queue_tier1_controls.log
done
