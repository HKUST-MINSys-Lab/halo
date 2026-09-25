#!/usr/bin/env bash
# Resume the paused (SIGSTOP) UniMTS Phase-A shard once rung-1 training has ended.
while pgrep -f "halo-train --out runs/support-classifier/halo_rung1_40k_20260925 " > /dev/null; do sleep 20; done
kill -CONT 491930
