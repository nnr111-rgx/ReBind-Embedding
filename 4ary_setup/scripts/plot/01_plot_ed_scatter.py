#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv",required=True); ap.add_argument("--output",required=True); ap.add_argument("--max-points",type=int,default=5000); ap.add_argument("--seed",type=int,default=0); args=ap.parse_args()
    df=pd.read_csv(args.csv); n=min(len(df),args.max_points); df=df.sample(n=n,random_state=args.seed) if len(df)>n else df
    true_col="true_ed"; pred_col="pred_ed"
    lo=min(df[true_col].min(),df[pred_col].min()); hi=max(df[true_col].max(),df[pred_col].max())
    fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(df[true_col],df[pred_col],s=9,alpha=.35); ax.plot([lo,hi],[lo,hi],linewidth=1); ax.set_xlabel("True edit distance"); ax.set_ylabel("Predicted edit distance"); ax.set_title("4-ary ReBind edit-distance estimate"); fig.tight_layout()
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig); print(f"Saved figure: {out}")


if __name__=="__main__": main()
