#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rebind4ary.checkpoint import load_encoder_checkpoint
from rebind4ary.data import TripletH5Dataset
from rebind4ary.engine import make_loader, move_batch
from rebind4ary.metrics import distance_metrics, fit_linear
from rebind4ary.models import l2
from rebind4ary.utils import auto_device, save_json


@torch.no_grad()
def collect(encoder, loader, device):
    encoder.eval(); raw=[]; truth=[]; ap_raw=[]; an_raw=[]
    for b in tqdm(loader,leave=False):
        b=move_batch(b,device)
        za=encoder(b["anchor"],b["anchor_len"]); zp=encoder(b["positive"],b["positive_len"]); zn=encoder(b["negative"],b["negative_len"])
        dap=l2(za,zp); dan=l2(za,zn); dpn=l2(zp,zn)
        raw.append(torch.cat([dap,dan,dpn]).cpu().numpy())
        truth.append(torch.cat([b["d_ap"],b["d_an"],b["d_pn"]]).cpu().numpy())
        ap_raw.append(dap.cpu().numpy()); an_raw.append(dan.cpu().numpy())
    return np.concatenate(raw),np.concatenate(truth),np.concatenate(ap_raw),np.concatenate(an_raw)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--device",default="auto"); args=ap.parse_args()
    device=auto_device(args.device); enc,_,ck=load_encoder_checkpoint(args.checkpoint,device)
    tr=TripletH5Dataset(args.data,"train"); te=TripletH5Dataset(args.data,"test")
    trl=make_loader(tr,args.batch_size,False,args.workers,1,"triplet"); tel=make_loader(te,args.batch_size,False,args.workers,2,"triplet")
    tr_raw,tr_y,_,_=collect(enc,trl,device); a,b=fit_linear(tr_raw,tr_y)
    raw,y,apd,and_=collect(enc,tel,device); pred=a*raw+b
    m=distance_metrics(pred,y); m.update({"calibration_slope":a,"calibration_intercept":b,"ranking_accuracy":float(np.mean(apd<and_)),"checkpoint_epoch":int(ck.get("epoch",0))})
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); save_json(m,out/"triplet_metrics.json")
    pd.DataFrame({"true_ed":y,"pred_ed":pred,"raw_embedding_distance":raw}).to_csv(out/"ed_scatter.csv",index=False)
    print(m); print(f"Saved evaluation to {out}")


if __name__=="__main__": main()
