#!/usr/bin/env python
from __future__ import annotations

import argparse

from rebind4ary.checkpoint import load_encoder_checkpoint
from rebind4ary.data import IDSH5Dataset
from rebind4ary.engine import encode_ids_dataset, make_loader
from rebind4ary.metrics import correlation_summary
from rebind4ary.utils import auto_device
from rebind4ary.whitening import fit_whitener, save_whitener


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--output",required=True); ap.add_argument("--split",default="train"); ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--shrinkage",type=float,default=0.01); ap.add_argument("--eps-relative",type=float,default=1e-4); ap.add_argument("--device",default="auto"); args=ap.parse_args()
    device=auto_device(args.device); enc,_,ck=load_encoder_checkpoint(args.checkpoint,device); ds=IDSH5Dataset(args.data,args.split); x=encode_ids_dataset(enc,make_loader(ds,args.batch_size,False,args.workers,1,"ids"),device)
    residual=x["z_noisy"]-x["z_clean"]; whitener=fit_whitener(residual,args.shrinkage,args.eps_relative); white=(residual-whitener.mean.numpy())@whitener.matrix.numpy().T
    meta={"split":args.split,"num_samples":len(residual),"checkpoint_epoch":int(ck.get("epoch",0)),"raw":correlation_summary(residual),"whitened":correlation_summary(white)}
    save_whitener(args.output,whitener,meta); print({"output":args.output,**meta})


if __name__=="__main__": main()
