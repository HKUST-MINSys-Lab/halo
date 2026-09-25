#!/usr/bin/env bash
# T1-D secondary diagnostic: the rung-1-trained HALO scored at T=30 (its training temperature),
# after the queued T1-D runs finish.
set -u
cd /home/alex/code/HALO/halo
until grep -q "T1-D best_internal" runs/evaluations/queue_after_training.log 2>/dev/null; do sleep 20; done
OUT=runs/evaluations/rung1_halo_trained_last_T30_20260925; mkdir -p $OUT
.venv/bin/halo-rung1 --out $OUT --models halo --encoder-label halo:rung1-trained-last-T30 \
  --halo-checkpoint runs/support-classifier/halo_rung1_40k_20260925/last.pt --temperature 30 \
  --feature-cache runs/evaluations/rung1_cache_trained --feature-memory-cache-gib 6 --resume > $OUT/run.log 2>&1
echo "[queue] $(date) T1-D T30 exit $?" >> runs/evaluations/queue_after_training.log
