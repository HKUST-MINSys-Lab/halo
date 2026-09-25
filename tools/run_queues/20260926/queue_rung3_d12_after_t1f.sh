#!/usr/bin/env bash
# Matched-HARNet training (T1-F) peaks at ~22 GB, so the HALO and UniMTS draw-1-2 shards wait for it.
set -u
cd /home/alex/code/HALO/halo
while pgrep -f "halo-train --out runs/support-classifier/matched_harnet_40k_20260925" > /dev/null; do sleep 30; done
nohup runs/evaluations/queue_rung3_phaseA_d12.sh halo 3 > /dev/null 2>&1 &
nohup runs/evaluations/queue_rung3_phaseA_d12.sh unimts 4.5 > /dev/null 2>&1 &
