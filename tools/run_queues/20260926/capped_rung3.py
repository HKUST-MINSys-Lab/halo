"""halo-rung3 under a per-process GPU memory cap (GPU_GIB env), so co-scheduled shards cannot OOM
other jobs on the card; the capped process fails alone and --resume picks it up."""
import os, runpy, sys
import torch
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ.get("GPU_GIB", "4")) / total))
sys.argv = ["halo-rung3"] + sys.argv[1:]
runpy.run_module("evaluation.rung3_finetune.run", run_name="__main__")
