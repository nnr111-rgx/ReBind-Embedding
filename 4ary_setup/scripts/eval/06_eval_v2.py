#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rebind4ary.calibration import load_calibrator
from rebind4ary.checkpoint import load_encoder_checkpoint
from rebind4ary.data import IDSH5Dataset
from rebind4ary.engine import encode_ids_dataset, make_loader
from rebind4ary.metrics import binary_metrics_from_llr
from rebind4ary.utils import auto_device, save_json
from rebind4ary.whitening import load_whitener


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--teacher-data"); ap.add_argument("--checkpoint",required=True); ap.add_argument("--whitener",required=True); ap.add_argument("--calibrator",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--split",default="test"); ap.add_argument("--llr-clip",type=float,default=20.0); ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--device",default="auto"); args=ap.parse_args()
    device=auto_device(args.device); enc,_,ck=load_encoder_checkpoint(args.checkpoint,device); wh=load_whitener(args.whitener,"cpu"); cal=load_calibrator(args.calibrator,"cpu"); ds=IDSH5Dataset(args.data,args.split,args.teacher_data); x=encode_ids_dataset(enc,make_loader(ds,args.batch_size,False,args.workers,1,"ids"),device)
    obs=(x["z_noisy"]-wh.mean.numpy())@wh.matrix.numpy().T; llr=-(obs*cal.weight.numpy()+cal.bias.numpy()); llr=np.clip(llr,-args.llr_clip,args.llr_clip); result={"split":args.split,"checkpoint_epoch":int(ck.get("epoch",0)),"student":binary_metrics_from_llr(llr,x["bits"])}
    if "teacher_llr" in x: result["oracle_teacher"]=binary_metrics_from_llr(np.clip(x["teacher_llr"],-args.llr_clip,args.llr_clip),x["bits"])
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); save_json(result,out/"v2_metrics.json"); np.savez_compressed(out/"v2_llr_outputs.npz",sample_id=x["sample_id"],condition_id=x["condition_id"],bits=x["bits"].astype(np.uint8),student_llr=llr.astype(np.float32),teacher_llr=x.get("teacher_llr",np.empty((0,204),dtype=np.float32)))
    print(result); print(f"Saved evaluation to {out}")


if __name__=="__main__": main()
