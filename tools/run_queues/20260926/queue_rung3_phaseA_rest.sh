#!/usr/bin/env bash
# Staggered Phase-A shards: HALO once rung-1 training frees its ~11 GB; HARNet-5 once LiMU-BERT-X is done.
set -u
cd /home/alex/code/HALO/halo
while pgrep -f "halo-train --out runs/support-classifier/halo_rung1_40k_20260925 " > /dev/null; do sleep 30; done
nohup runs/evaluations/queue_rung3_phaseA.sh halo 3 > /dev/null 2>&1 &
until grep -q "limubert_x done" runs/evaluations/rung3_phaseA_d0_20260926/limubert_x/run.log 2>/dev/null; do sleep 30; done
nohup runs/evaluations/queue_rung3_phaseA.sh harnet5 2 > /dev/null 2>&1 &
