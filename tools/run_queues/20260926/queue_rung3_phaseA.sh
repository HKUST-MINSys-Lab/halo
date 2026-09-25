#!/usr/bin/env bash
# Tier-3 Phase A (runbook T3-A), draw 0: full fine-tune + from-scratch specialist, k in {1,4,16},
# one process per model under a GPU memory cap. Draws 1-2 go to a later run with --first-draw 1.
# usage: queue_rung3_phaseA.sh MODEL GPU_GIB   (at most 3 attempts; --resume keeps finished cells)
set -u
cd /home/alex/code/HALO/halo
MODEL=$1; export GPU_GIB=$2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/evaluations/rung3_phaseA_d0_20260926/$MODEL; mkdir -p $OUT
for attempt in 1 2 3; do
  if .venv/bin/python runs/evaluations/tools/capped_rung3.py --out $OUT --models $MODEL \
      --halo-checkpoint runs/support-classifier/halo_t6_40k_20260920/last.pt \
      --k 1 4 16 --support-draws 1 --treatments full_finetune scratch_specialist --resume >> $OUT/run.log 2>&1; then
    echo "[queue] $(date) $MODEL done" >> $OUT/run.log; exit 0
  fi
  echo "[queue] $(date) $MODEL attempt $attempt failed" >> $OUT/run.log; sleep 60
done
echo "[queue] $(date) $MODEL gave up after 3 attempts" >> $OUT/run.log
