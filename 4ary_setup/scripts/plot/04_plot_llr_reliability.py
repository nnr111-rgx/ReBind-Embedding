#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--npz",required=True); ap.add_argument("--output",required=True); ap.add_argument("--bins",type=int,default=15); args=ap.parse_args()
    d=np.load(args.npz); bits=np.asarray(d["bits"]).reshape(-1); key="llr" if "llr" in d else "student_llr"; llr=np.asarray(d[key]).reshape(-1); prob1=1/(1+np.exp(np.clip(llr,-60,60))); edges=np.linspace(0,1,args.bins+1); xs=[]; ys=[]
    for i in range(args.bins):
        m=(prob1>=edges[i])&(prob1<(edges[i+1] if i<args.bins-1 else edges[i+1]+1e-12))
        if np.any(m): xs.append(float(prob1[m].mean())); ys.append(float(bits[m].mean()))
    fig,ax=plt.subplots(figsize=(5,5)); ax.plot([0,1],[0,1],linewidth=1); ax.plot(xs,ys,marker="o"); ax.set_xlabel("Predicted P(bit=1)"); ax.set_ylabel("Empirical bit-1 frequency"); ax.set_title("LLR reliability"); ax.set_xlim(0,1); ax.set_ylim(0,1); fig.tight_layout(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig); print(f"Saved figure: {out}")


if __name__=="__main__": main()
