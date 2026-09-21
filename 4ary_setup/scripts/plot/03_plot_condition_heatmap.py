#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv",required=True); ap.add_argument("--metric",default="ber"); ap.add_argument("--output",required=True); args=ap.parse_args()
    df=pd.read_csv(args.csv); xs=sorted(df["p_sub"].unique()); ys=sorted(df["p_ins"].unique()); mat=np.full((len(ys),len(xs)),np.nan)
    for i,p in enumerate(ys):
        for j,s in enumerate(xs):
            z=df[(df.p_ins==p)&(df.p_sub==s)]
            if len(z): mat[i,j]=float(z.iloc[0][args.metric])
    fig,ax=plt.subplots(figsize=(7,4)); im=ax.imshow(mat,aspect="auto",origin="lower"); ax.set_xticks(range(len(xs)),[f"{x:.2f}" for x in xs]); ax.set_yticks(range(len(ys)),[f"{y:.2f}" for y in ys]); ax.set_xlabel("p_sub"); ax.set_ylabel("p_ins = p_del"); ax.set_title(args.metric)
    for i in range(len(ys)):
        for j in range(len(xs)):
            if np.isfinite(mat[i,j]): ax.text(j,i,f"{mat[i,j]:.3f}",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=ax); fig.tight_layout(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig); print(f"Saved figure: {out}")


if __name__=="__main__": main()
