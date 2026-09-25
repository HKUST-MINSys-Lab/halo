#!/usr/bin/env bash
# After HALO's rung-1 training ends: score it (T1-D), check its inductive floor (T1-E), then train
# and score the corpus-matched HARNet arm (T1-F). Checkpoint policy declared before any result:
# last.pt (step 40k) is primary; best_internal.pt is a secondary diagnostic.
set -u
cd /home/alex/code/HALO/halo
TRAIN=runs/support-classifier/halo_rung1_40k_20260925
LOG=runs/evaluations/queue_after_training.log
while pgrep -f "halo-train --out $TRAIN " > /dev/null; do sleep 30; done
echo "[queue] $(date) training finished" >> $LOG
CACHES="--cache-read-dirs cache/evaluations/feature_cache_halo_t6_40k_20260920_final40k --feature-cache runs/evaluations/rung1_cache_trained --feature-memory-cache-gib 6"
for ck in last best_internal; do
  OUT=runs/evaluations/rung1_halo_trained_${ck}_20260925; mkdir -p $OUT
  .venv/bin/halo-rung1 --out $OUT --models halo --encoder-label halo:rung1-trained-${ck} --halo-checkpoint $TRAIN/${ck}.pt $CACHES --resume > $OUT/run.log 2>&1
  echo "[queue] $(date) T1-D $ck exit $?" >> $LOG
done
OUT=runs/evaluations/sealed_1nn_halo_rung1_trained_20260925; mkdir -p $OUT
.venv/bin/halo-sealed-eval --out $OUT --models halo --halo-checkpoint $TRAIN/last.pt --k 1 8 32 128 --window-seconds 8 --feature-cache runs/evaluations/rung1_cache_trained > $OUT.log 2>&1
echo "[queue] $(date) T1-E exit $?" >> $LOG
ARM=runs/support-classifier/matched_harnet_40k_20260925; rm -rf $ARM; mkdir -p $ARM
.venv/bin/halo-train --out $ARM --encoder-arch harnet --classifier neighbors --steps 40000 --log-every 1000 > $ARM.log 2>&1
echo "[queue] $(date) T1-F train exit $?" >> $LOG
OUT=runs/evaluations/rung1_matched_harnet_20260925; mkdir -p $OUT
.venv/bin/halo-rung1 --out $OUT --models halo --encoder-label matched:harnet --halo-checkpoint $ARM/last.pt --feature-cache runs/evaluations/rung1_cache_matched --feature-memory-cache-gib 6 --resume > $OUT/run.log 2>&1
echo "[queue] $(date) T1-F eval exit $?" >> $LOG
