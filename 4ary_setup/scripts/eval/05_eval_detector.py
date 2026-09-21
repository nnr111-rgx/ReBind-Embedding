#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rebind4ary.calibration import load_calibrator
from rebind4ary.checkpoint import load_encoder_checkpoint
from rebind4ary.data import IDSH5Dataset
from rebind4ary.engine import encode_ids_dataset, make_loader
from rebind4ary.evaluation import evaluate_detector, evaluate_geometry
from rebind4ary.utils import auto_device, save_json
from rebind4ary.whitening import load_whitener


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--whitener",required=True); ap.add_argument("--calibrator",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--split",default="test"); ap.add_argument("--llr-clip",type=float,default=20.0); ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--device",default="auto"); args=ap.parse_args()
    device=auto_device(args.device); enc,_,ck=load_encoder_checkpoint(args.checkpoint,device); wh=load_whitener(args.whitener,"cpu"); cal=load_calibrator(args.calibrator,"cpu")
    tr=IDSH5Dataset(args.data,"train"); ev=IDSH5Dataset(args.data,args.split); trenc=encode_ids_dataset(enc,make_loader(tr,args.batch_size,False,args.workers,1,"ids"),device); _,_,geomcal=evaluate_geometry(trenc)
    x=encode_ids_dataset(enc,make_loader(ev,args.batch_size,False,args.workers,2,"ids"),device); geom,_,_=evaluate_geometry(x,geomcal); overall,bycond,llr,wc,wn=evaluate_detector(x,wh,cal,args.llr_clip)
    result={"split":args.split,"checkpoint_epoch":int(ck.get("epoch",0)),"geometry":geom,"detector":overall}
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); save_json(result,out/"detector_metrics.json"); bycond.to_csv(out/"detector_metrics_by_condition.csv",index=False); np.savez_compressed(out/"llr_outputs.npz",sample_id=x["sample_id"],condition_id=x["condition_id"],bits=x["bits"].astype(np.uint8),llr=llr.astype(np.float32),p=x["params"].astype(np.float32))
    print(result); print(f"Saved evaluation to {out}")


if __name__=="__main__": main()
