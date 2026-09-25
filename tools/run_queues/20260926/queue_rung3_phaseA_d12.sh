#!/usr/bin/env bash
# Tier-3 Phase A, support draws 1-2 (draw 0 is in rung3_phaseA_d0_20260926), after MODEL's draw 0.
# usage: queue_rung3_phaseA_d12.sh MODEL GPU_GIB   (at most 3 attempts; --resume keeps finished cells)
set -u
cd /home/alex/code/HALO/halo
MODEL=$1; export GPU_GIB=$2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "$MODEL done" runs/evaluations/rung3_phaseA_d0_20260926/$MODEL/run.log 2>/dev/null; do sleep 30; done
OUT=runs/evaluations/rung3_phaseA_d12_20260926/$MODEL; mkdir -p $OUT
for attempt in 1 2 3; do
  if .venv/bin/python runs/evaluations/tools/capped_rung3.py --out $OUT --models $MODEL \
      --halo-checkpoint runs/support-classifier/halo_t6_40k_20260920/last.pt \
      --k 1 4 16 --first-draw 1 --support-draws 3 --treatments full_finetune scratch_specialist --resume >> $OUT/run.log 2>&1; then
    echo "[queue] $(date) $MODEL done" >> $OUT/run.log; exit 0
  fi
  echo "[queue] $(date) $MODEL attempt $attempt failed" >> $OUT/run.log; sleep 60
done
echo "[queue] $(date) $MODEL gave up after 3 attempts" >> $OUT/run.log
