#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv",required=True); ap.add_argument("--output",required=True); ap.add_argument("--y",default="train_loss,val_loss"); args=ap.parse_args()
    df=pd.read_csv(args.csv); cols=[x.strip() for x in args.y.split(",") if x.strip()]
    fig,ax=plt.subplots(figsize=(7,4.5))
    for c in cols:
        if c in df.columns: ax.plot(df["epoch"],df[c],label=c)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Metric"); ax.set_title("Training history"); ax.legend(); fig.tight_layout(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig); print(f"Saved figure: {out}")


if __name__=="__main__": main()
