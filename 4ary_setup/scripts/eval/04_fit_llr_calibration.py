#!/usr/bin/env python
from __future__ import annotations

import argparse

from rebind4ary.calibration import fit_logistic_calibrator, save_calibrator
from rebind4ary.checkpoint import load_encoder_checkpoint
from rebind4ary.data import IDSH5Dataset
from rebind4ary.engine import encode_ids_dataset, make_loader
from rebind4ary.metrics import binary_metrics_from_llr
from rebind4ary.utils import auto_device
from rebind4ary.whitening import load_whitener


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--whitener",required=True); ap.add_argument("--output",required=True); ap.add_argument("--split",default="train"); ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--newton-steps",type=int,default=30); ap.add_argument("--device",default="auto"); args=ap.parse_args()
    device=auto_device(args.device); enc,_,ck=load_encoder_checkpoint(args.checkpoint,device); wh=load_whitener(args.whitener,"cpu"); ds=IDSH5Dataset(args.data,args.split); x=encode_ids_dataset(enc,make_loader(ds,args.batch_size,False,args.workers,1,"ids"),device)
    obs=(x["z_noisy"]-wh.mean.numpy())@wh.matrix.numpy().T; cal=fit_logistic_calibrator(obs,x["bits"],steps=args.newton_steps); llr=-(obs*cal.weight.numpy()+cal.bias.numpy()); met=binary_metrics_from_llr(llr,x["bits"])
    save_calibrator(args.output,cal,{"split":args.split,"checkpoint_epoch":int(ck.get("epoch",0)),"train_metrics":met}); print({"output":args.output,"train_metrics":met})


if __name__=="__main__": main()
