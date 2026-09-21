#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rebind4ary.checkpoint import load_encoder_checkpoint
from rebind4ary.data import IDSH5Dataset
from rebind4ary.engine import encode_ids_dataset, make_loader
from rebind4ary.evaluation import evaluate_geometry
from rebind4ary.utils import auto_device, save_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--split",default="test"); ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--device",default="auto"); args=ap.parse_args()
    device=auto_device(args.device); enc,_,ck=load_encoder_checkpoint(args.checkpoint,device)
    train=IDSH5Dataset(args.data,"train"); ev=IDSH5Dataset(args.data,args.split)
    tr=encode_ids_dataset(enc,make_loader(train,args.batch_size,False,args.workers,1,"ids"),device)
    _,_,cal=evaluate_geometry(tr)
    x=encode_ids_dataset(enc,make_loader(ev,args.batch_size,False,args.workers,2,"ids"),device)
    m,pred,_=evaluate_geometry(x,cal); m["checkpoint_epoch"]=int(ck.get("epoch",0)); m["split"]=args.split
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); save_json(m,out/"ids_geometry_metrics.json")
    pd.DataFrame({"sample_id":x["sample_id"],"condition_id":x["condition_id"],"true_ed":x["edit_distance"],"pred_ed":pred}).to_csv(out/"ids_ed_scatter.csv",index=False)
    print(m); print(f"Saved evaluation to {out}")


if __name__=="__main__": main()
