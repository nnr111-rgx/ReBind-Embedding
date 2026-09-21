#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from rebind4ary.calibration import load_calibrator
from rebind4ary.checkpoint import load_encoder_checkpoint, save_encoder_checkpoint
from rebind4ary.data import IDSH5Dataset
from rebind4ary.engine import append_history, make_loader, move_batch
from rebind4ary.losses import offdiag_correlation_loss, pairwise_rank_loss
from rebind4ary.models import l2
from rebind4ary.utils import auto_device, set_seed
from rebind4ary.whitening import load_whitener


def pass_epoch(encoder, dist_calib, reference, whitener, llr_calib, loader, device, args, optimizer=None):
    train=optimizer is not None
    encoder.train(train); dist_calib.train(train); reference.eval()
    sums={"loss":0.0,"soft":0.0,"llr_huber":0.0,"hard_bit":0.0,"ed":0.0,"rank":0.0,"preserve":0.0,"white":0.0,"student_bce":0.0,"ber":0.0}
    n=0
    for batch in tqdm(loader,leave=False):
        batch=move_batch(batch,device)
        with torch.set_grad_enabled(train):
            zc=encoder(batch["clean"],batch["clean_len"])
            zn=encoder(batch["noisy"],batch["noisy_len"])
            wc=whitener.transform(zc); wn=whitener.transform(zn)
            student_llr=llr_calib.llr_bit0_over_bit1(wn)
            teacher_llr=batch["teacher_llr"].clamp(-args.teacher_clip,args.teacher_clip)
            temp=float(args.temperature)
            teacher_p1=torch.sigmoid(-teacher_llr/temp)
            soft=F.binary_cross_entropy_with_logits(-student_llr/temp,teacher_p1)*(temp*temp)
            llr_huber=F.smooth_l1_loss(student_llr/args.teacher_clip,teacher_llr/args.teacher_clip)
            hard_bit=F.binary_cross_entropy_with_logits(-student_llr,batch["bits"])
            pred_ed=dist_calib(l2(zc,zn))
            ed=F.smooth_l1_loss(pred_ed/args.distance_max,batch["edit_distance"]/args.distance_max)
            rank=pairwise_rank_loss(pred_ed/args.distance_max,batch["edit_distance"]/args.distance_max)
            with torch.no_grad():
                rc=reference(batch["clean"],batch["clean_len"])
                rn=reference(batch["noisy"],batch["noisy_len"])
            preserve=0.5*(F.mse_loss(zc,rc)+F.mse_loss(zn,rn))
            white=offdiag_correlation_loss(wn-wc)
            loss=(
                args.lambda_soft_distill*soft+
                args.lambda_llr_huber*llr_huber+
                args.lambda_hard_bit*hard_bit+
                args.lambda_ed*ed+
                args.lambda_rank*rank+
                args.lambda_preserve*preserve+
                args.lambda_whiteness*white
            )
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(dist_calib.parameters()),1.0); optimizer.step()
        student_bce=F.binary_cross_entropy_with_logits(-student_llr.detach(),batch["bits"])
        ber=((student_llr.detach()<0).float()!=batch["bits"]).float().mean()
        bs=len(batch["bits"])
        vals={"loss":loss,"soft":soft,"llr_huber":llr_huber,"hard_bit":hard_bit,"ed":ed,"rank":rank,"preserve":preserve,"white":white,"student_bce":student_bce,"ber":ber}
        for k,v in vals.items(): sums[k]+=float(v.detach())*bs
        n+=bs
    return {k:v/n for k,v in sums.items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--teacher-data",required=True)
    ap.add_argument("--v1-checkpoint",required=True)
    ap.add_argument("--whitener",required=True)
    ap.add_argument("--calibrator",required=True)
    ap.add_argument("--save-dir",required=True)
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--lr",type=float,default=1e-5)
    ap.add_argument("--weight-decay",type=float,default=1e-4)
    ap.add_argument("--temperature",type=float,default=2.0)
    ap.add_argument("--teacher-clip",type=float,default=20.0)
    ap.add_argument("--distance-max",type=float,default=32.0)
    ap.add_argument("--lambda-soft-distill",type=float,default=1.0)
    ap.add_argument("--lambda-llr-huber",type=float,default=0.1)
    ap.add_argument("--lambda-hard-bit",type=float,default=0.5)
    ap.add_argument("--lambda-ed",type=float,default=0.2)
    ap.add_argument("--lambda-rank",type=float,default=0.05)
    ap.add_argument("--lambda-preserve",type=float,default=0.05)
    ap.add_argument("--lambda-whiteness",type=float,default=0.01)
    ap.add_argument("--patience",type=int,default=8)
    ap.add_argument("--min-epochs",type=int,default=10)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--seed",type=int,default=4304)
    ap.add_argument("--device",default="auto")
    args=ap.parse_args()
    set_seed(args.seed); device=auto_device(args.device)
    save_dir=Path(args.save_dir); save_dir.mkdir(parents=True,exist_ok=True)
    encoder,dist_calib,_=load_encoder_checkpoint(args.v1_checkpoint,device)
    reference=deepcopy(encoder).eval()
    for p in reference.parameters(): p.requires_grad_(False)
    whitener=load_whitener(args.whitener,device)
    llr_calib=load_calibrator(args.calibrator,device)
    tr=IDSH5Dataset(args.data,"train",args.teacher_data); va=IDSH5Dataset(args.data,"validation",args.teacher_data)
    tl=make_loader(tr,args.batch_size,True,args.workers,args.seed,"ids"); vl=make_loader(va,args.batch_size*2,False,args.workers,args.seed+1,"ids")
    opt=AdamW(list(encoder.parameters())+list(dist_calib.parameters()),lr=args.lr,weight_decay=args.weight_decay)
    best=float("inf"); stale=0
    for epoch in range(1,args.epochs+1):
        a=pass_epoch(encoder,dist_calib,reference,whitener,llr_calib,tl,device,args,opt)
        b=pass_epoch(encoder,dist_calib,reference,whitener,llr_calib,vl,device,args,None)
        row={"epoch":epoch,**{f"train_{k}":v for k,v in a.items()},**{f"val_{k}":v for k,v in b.items()}}
        append_history(save_dir/"history.csv",row); print(row)
        if b["student_bce"]<best:
            best=b["student_bce"]; stale=0
            save_encoder_checkpoint(save_dir/"best.pt",encoder,dist_calib,opt,epoch,b,{"stage":"v2_bcjr_distillation","selection":"validation student BCE","whitener":args.whitener,"calibrator":args.calibrator})
        else: stale+=1
        if epoch>=args.min_epochs and stale>=args.patience: break
    save_encoder_checkpoint(save_dir/"last.pt",encoder,dist_calib,opt,epoch,b,{"stage":"v2_bcjr_distillation"})
    print({"best":str(save_dir/"best.pt"),"best_val_student_bce":best})


if __name__=="__main__": main()
