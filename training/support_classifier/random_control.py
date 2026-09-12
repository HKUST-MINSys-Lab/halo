"""Create an architecture-matched random encoder control from a JEPA checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from training.tokenizer.eval_transfer import build_encoder

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=20260901); args=ap.parse_args()
    torch.manual_seed(args.seed)
    source=torch.load(args.source,map_location='cpu',weights_only=False)
    encoder=build_encoder(source,torch.device('cpu'),training=False)
    for module in encoder.modules():
        reset=getattr(module,'reset_parameters',None)
        if callable(reset): reset()
    payload={'config':dict(source['config']),'encoder':encoder.state_dict(),'step':0,
             'random_control':True,'source_checkpoint':str(args.source),'seed':args.seed}
    args.out.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,args.out)
if __name__=='__main__': main()
